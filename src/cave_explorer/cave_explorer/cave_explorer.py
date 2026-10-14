#!/usr/bin/env python3

import os
import sys

# Advanced 1 prep: the trained artefact classifier (see build_classifier_dataset.py and
# config/artifact_classifier/) is loaded via tflite_runtime rather than full tensorflow - much
# lighter weight for a ROS node that only needs inference, not training.
#
# Installed into its own dedicated directory (see README) rather than the general user
# site-packages - deliberately NOT just `pip install --user`. Two problems with that: (1) the
# ROS-sourced shell environment's PYTHONPATH doesn't include the user site-packages directory
# by default (same gap hit earlier with pypdf), and (2) on this machine the general user
# site-packages already has an unrelated, much newer opencv-python installed (from prior,
# unrelated work) - adding that whole directory to sys.path would silently swap this node's
# cv2 for a different, untested version. A dedicated install directory, containing only
# tflite_runtime and the specific numpy release it needs, avoids both: it's added to sys.path
# explicitly here (no bashrc edit needed on any teammate's machine) without exposing anything
# else that happens to live in the general user site-packages.
#
# MUST happen before cv2/numpy are imported below (not just before tflite_runtime) - numpy is a
# process-wide singleton once imported, so if cv2's own internal numpy import runs first and
# binds the older system numpy, tflite_runtime's compiled extension (needing a newer numpy ABI)
# fails even though the right numpy is sitting right here, unused.
_tflite_install_dir = os.path.expanduser('~/.local/share/cave_explorer_tflite')
if os.path.isdir(_tflite_install_dir) and _tflite_install_dir not in sys.path:
    sys.path.insert(0, _tflite_install_dir)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    tflite = None

import functools
import json
import math
import random
from collections import deque, namedtuple
from enum import Enum

import cv2  # OpenCV2
import numpy as np
import rclpy
import tf2_geometry_msgs
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Point, PointStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from cave_explorer import cave_geometry, science_tour


def wrap_angle(angle):
    """Function to wrap an angle between 0 and 2*Pi"""
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle

def dilate8_mask(mask):
    """
    Grow a boolean grid mask by one cell in all 8 directions (OR together the mask shifted by
    each neighbour offset), without a slow per-cell python loop. Shared by find_frontier_cells()
    and compute_path_distance_grid() - see each for how the result is used.
    """

    out = np.zeros_like(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(np.roll(mask, dr, axis=0), dc, axis=1)
            # np.roll wraps around; clear the wrapped-in edge so it isn't mistaken for a
            # real neighbour on the opposite side of the map
            if dr == -1:
                shifted[-1, :] = False
            elif dr == 1:
                shifted[0, :] = False
            if dc == -1:
                shifted[:, -1] = False
            elif dc == 1:
                shifted[:, 0] = False
            out |= shifted
    return out


def pose2d_to_pose(pose_2d):
    """Convert a Pose2D to a full 3D Pose"""
    pose = Pose()

    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y

    pose.orientation.w = math.cos(pose_2d.theta / 2.0)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)

    return pose


# Perception 2: distance (pixels) within which two bounding boxes are merged into one - see
# merge_overlapping_boxes(). Raised from 10: a single object's silhouette can have a genuine
# visible gap between two parts from some camera angles (e.g. a jagged/spiky crystal model),
# wider than 10px was accounting for - live testing showed the same correctly-identified
# label (e.g. green_alien) still split into two separate boxes for one physical object.
# 30px is still small relative to a real detection's own size (min_area, the smallest
# profiles allow, is already ~300px^2 - roughly a 17x17px box - so this won't casually bridge
# two genuinely distinct nearby objects, since real artefacts in this world are spaced several
# metres apart; it specifically targets gaps within one object's own silhouette.
BOX_MERGE_GAP_PX = 30


def merge_overlapping_boxes(boxes, gap_px=BOX_MERGE_GAP_PX):
    """
    Perception 2: merge bounding boxes that are at or within 'gap_px' of each other (including
    touching/overlapping) into a single box per group, using union-find.

    A single real artefact's colour mask can fragment into several disconnected contours
    (shading, texture, specular highlights on the 3D model), which would otherwise each become
    a separate detection for what's visually one object. Genuinely separate instances of the
    same artefact type elsewhere in the frame (far apart) are left as distinct boxes.
    """

    if len(boxes) <= 1:
        return list(boxes)

    def expand(box):
        x, y, w, h = box
        return (x - gap_px, y - gap_px, x + w + gap_px, y + h + gap_px)

    def overlaps(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    expanded = [expand(box) for box in boxes]
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if overlaps(expanded[i], expanded[j]):
                parent[find(i)] = find(j)

    groups = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for indices in groups.values():
        x1 = min(boxes[i][0] for i in indices)
        y1 = min(boxes[i][1] for i in indices)
        x2 = max(boxes[i][0] + boxes[i][2] for i in indices)
        y2 = max(boxes[i][1] + boxes[i][3] for i in indices)
        merged.append((x1, y1, x2 - x1, y2 - y1))
    return merged


# Perception 2 robustness: two different-label detections are only treated as "the same
# physical object seen twice" (see suppress_overlapping_labels()) if their bounding boxes
# overlap by at least this much IoU (intersection over union). Plain "do they touch at all"
# is not enough: a broad, low-confidence profile (e.g. mossy_boulder/toy_story_alien matching
# wall or floor texture - a known false-positive source, since neither is actually placed in
# this cave world) can produce a box spanning most of the frame, which would otherwise
# "overlap" and wrongly swallow every other small, genuine detection in the same frame just
# by bounding-box containment. IoU penalises that - a tiny real detection inside a huge
# background box has very low IoU - while still catching the actual target case (two
# deliberately-similar profiles both matching one object, producing near-identical boxes).
LABEL_OVERLAP_IOU_THRESHOLD = 0.3

# Perception 2 robustness: two different-label detections are ALSO treated as "the same
# physical object/mount, detected via two different coloured parts of it" if their boxes are
# within this many pixels of each other - even without the IoU overlap above. One mount can
# show two genuinely different-coloured regions close together but not actually overlapping
# (e.g. a stop sign's red octagon and a separate blue backing panel right behind it getting
# matched as ice_formation) - same gap-tolerance concept as merge_overlapping_boxes(), reused
# here for the cross-label case.
CROSS_LABEL_GAP_PX = 30

# Still required even under the gap-tolerance above: guards against a huge, low-confidence
# background-texture match "swallowing" a small real detection just because its box happens
# to sprawl close to or over it - the two boxes must be within this size ratio of each other.
CROSS_LABEL_MAX_AREA_RATIO = 6.0


def bbox_iou(a, b):
    """Intersection over union of two (x, y, w, h) boxes."""

    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection == 0:
        return 0.0
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def is_same_object(a_bbox, b_bbox):
    """
    Perception 2: decide whether two different-label detections are plausibly the same
    physical object/mount - see CROSS_LABEL_GAP_PX and CROSS_LABEL_MAX_AREA_RATIO above.
    """

    area_a = a_bbox[2] * a_bbox[3]
    area_b = b_bbox[2] * b_bbox[3]
    if max(area_a, area_b) > CROSS_LABEL_MAX_AREA_RATIO * max(1, min(area_a, area_b)):
        return False

    if bbox_iou(a_bbox, b_bbox) >= LABEL_OVERLAP_IOU_THRESHOLD:
        return True

    ax, ay, aw, ah = a_bbox
    bx, by, bw, bh = b_bbox
    gap = CROSS_LABEL_GAP_PX
    return not (ax + aw + gap < bx or bx + bw + gap < ax
                or ay + ah + gap < by or by + bh + gap < ay)


def suppress_overlapping_labels(detections, priority_labels=frozenset()):
    """
    Perception 2: when detections of DIFFERENT labels are plausibly the same physical object
    (see is_same_object()) - one physical object/mount matching more than one HSV profile at
    once, rather than several fragments of the same profile (that's merge_overlapping_boxes()
    above) - keep only one and discard the rest.

    Several of ARTIFACT_COLOR_PROFILES' ranges are deliberately similar (see that list's own
    comment, e.g. green_alien vs toy_story_alien) and can both match the same pixels on one
    object, reporting it as two or more different artefact types simultaneously instead of
    one.

    Within a group, a detection whose label is in 'priority_labels' always wins over one that
    isn't, regardless of area - used for the stop sign (image_callback), whose dedicated
    cascade classifier is a far more reliable identification than an incidental colour-profile
    match landing on or near the same spot (e.g. its backing panel occasionally matching
    ice_formation's "icy" range). Otherwise, area is used as the tie-break (the better colour
    match for an object usually covers more of it) - the original per-contour pixel counts
    aren't available any more once merge_overlapping_boxes() has already merged same-label
    fragments into plain boxes.
    """

    if len(detections) <= 1:
        return list(detections)

    parent = list(range(len(detections)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if is_same_object(detections[i].bbox, detections[j].bbox):
                parent[find(i)] = find(j)

    groups = {}
    for i in range(len(detections)):
        groups.setdefault(find(i), []).append(i)

    kept = []
    for indices in groups.values():
        priority_indices = [i for i in indices if detections[i].label in priority_labels]
        candidates = priority_indices or indices
        best = max(candidates, key=lambda i: detections[i].bbox[2] * detections[i].bbox[3])
        kept.append(detections[best])
    return kept


class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    EXPLORE_FRONTIER = 6
    INSPECT_ARTIFACT = 7
    SCIENCE_TOUR = 8
    # Add more!


# A single detected artifact in the current camera frame.
# 'bbox' is (x, y, width, height) in pixel coordinates.
# 'color_rgb' is the colour (as an (r, g, b) tuple) used to draw/label this detection.
Detection = namedtuple('Detection', ['label', 'bbox', 'color_rgb'])

# Perception 2: colour/contour-based detectors for the artefact types, other than the
# stop sign (which is handled separately by the provided cascade classifier).
#
# Each profile thresholds the camera image in HSV space to find pixels that plausibly
# belong to that artefact, then treats sufficiently large connected blobs as detections.
# The HSV ranges below were estimated from the artefact models' base-colour textures
# (see worlds/models/artifacts/*), so treat them as a reasonable starting point rather
# than ground truth - re-tune them (e.g. with cv2 trackbars) against real camera frames
# from the Perception 1 dataset, since simulated lighting will shift these colours.
# Some artefacts (e.g. green_alien vs toy_story_alien, white_sphere vs ice_formation)
# have deliberately similar colours and may be confused with this simple approach.
#
# toy_story_alien, mossy_boulder, AND blue_cube are deliberately NOT included here: none of
# the three is actually placed anywhere in mars_cave.sdf (checked directly against the world
# file - it places exactly 15 artifacts, 3 each of only 5 real models: green_alien,
# green_crystals, white_sphere, ice_formation, mushrooms_blue). blue_cube in particular wasn't
# just an unused profile sitting idle - its hue range (100-125) happened to match
# mushrooms_blue's actual rendered colour well, so it was WINNING the cross-label tie-break
# against mushroom_blue's own profile all session, meaning every "blue_cube" detection/
# inspection logged throughout development was actually mushrooms_blue under the wrong label
# (confirmed by eye against saved Perception 1 dataset photos - every "blue_cube" frame shows
# the mushroom model, never a cube). This is also why mushroom_blue's own profile kept losing
# ground no matter how it was narrowed - it was being deliberately pushed AWAY from the hue
# range the real object actually needs, to dodge a competing profile that was never real to
# begin with. Dropping all three false profiles removes that confusion entirely with zero loss
# of real detection capability.
ARTIFACT_COLOR_PROFILES = [
    {
        'label': 'green_crystals',
        'hsv_lower': (42, 140, 40),
        'hsv_upper': (70, 255, 210),
        'min_area': 300,
        'color_rgb': (0, 200, 0),
    },
    {
        'label': 'green_alien',
        'hsv_lower': (30, 120, 140),
        'hsv_upper': (55, 255, 255),
        'min_area': 300,
        'color_rgb': (0, 255, 140),
    },
    {
        'label': 'white_sphere',
        'hsv_lower': (95, 30, 200),
        'hsv_upper': (120, 140, 255),
        'min_area': 200,
        'color_rgb': (255, 255, 255),
    },
    {
        'label': 'ice_formation',
        'hsv_lower': (90, 20, 90),
        'hsv_upper': (115, 130, 200),
        'min_area': 300,
        'color_rgb': (180, 220, 255),
    },
    {
        'label': 'mushroom_blue',
        # Empirically resampled from the Perception 1 dataset after the hue-only narrowing
        # history above (128, then reverted to 115) turned out to still be guesswork - directly
        # sampling real pixels from confirmed mushroom_blue photos showed its hue (97-121)
        # almost entirely overlaps BOTH white_sphere's (95-120) and ice_formation's (90-120) -
        # they're effectively the same hue band in this world, so no hue split alone can
        # separate them. What actually differs is saturation: mushroom_blue's pixels are far
        # more vivid (median S ~120-125) than white_sphere/ice_formation's pale, washed-out
        # look (median S ~49 for both). Sweeping the saturation floor against all three classes'
        # real sampled images found a clean cliff at S=150: 27/29 mushroom_blue images still
        # match (recall holds right up to the edge - S=160 drops to 4/29), while white_sphere
        # false-positives drop to 3/39 and ice_formation/green_crystals/green_alien/stop_sign/
        # negative all hit zero. Hue range left wide (90-125) since saturation is now doing the
        # real separating work, not hue.
        'hsv_lower': (90, 150, 90),
        'hsv_upper': (125, 255, 255),
        'min_area': 300,
        'color_rgb': (255, 120, 255),
    },
]

# Lookup used to colour Perception 3's RViz markers to match each artefact's Perception 2 detection colour
ARTIFACT_COLOR_BY_LABEL = {profile['label']: profile['color_rgb'] for profile in ARTIFACT_COLOR_PROFILES}

# Advanced 1: trained-classifier refinement of the HSV detections above. The Perception 1
# dataset's own sorting process (manually verifying hundreds of HSV-detected crops by eye)
# surfaced real, persistent cross-label confusion in the live detector itself - most notably
# green_alien/green_crystals (adjacent hue ranges; a distant or partially-lit alien can produce
# a torso-coloured blob that matches green_crystals' profile, sometimes even a BIGGER matching
# area than genuine crystal images) and the white_sphere/ice_formation/mushroom_blue group
# (near-identical pale, low-saturation appearance). A trained classifier - built from that same
# manually-verified dataset via Teachable Machine (see build_classifier_dataset.py) - is used
# here to catch exactly those cases: it re-examines each HSV-matched crop and can confirm,
# correct, or reject (if it predicts 'negative') the HSV label, rather than trusting HSV's
# first guess outright.
#
# Matches the 224x224 RGB input and preprocessing (see classify_crop()) the model was actually
# trained on by Teachable Machine - must stay in sync with any re-export of the model.
CLASSIFIER_INPUT_SIZE = 224
# Crop margin (pixels) around each HSV-detected box before classifying - matches
# build_classifier_dataset.py's CROP_MARGIN_PX exactly, so the classifier sees the same kind of
# crop live that it was trained on (a looser/tighter margin here would be a train/inference
# mismatch, undermining the whole point of training on real detection crops).
CLASSIFIER_CROP_MARGIN_PX = 15
# Below this confidence, the classifier's opinion is treated as inconclusive and the original
# HSV label is kept as-is rather than risking a low-confidence relabel/rejection overriding a
# perfectly good HSV match.
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.6
CLASSIFIER_NEGATIVE_LABEL = 'negative'
# Perception 2's own cascade classifier for stop signs is already a dedicated, reliable
# detector (see detect_stop_signs()) - not part of the HSV cross-label confusion this is meant
# to fix, so stop_sign detections are left untouched rather than re-judged by this classifier.
CLASSIFIER_SKIP_LABELS = frozenset({'stop_sign'})

# The RGB-D camera's colour and depth images share one sensor (single <camera> block in
# mars_explorer.gazebo.xacro), so both use the same intrinsics computed from its resolution/FOV.
CAMERA_WIDTH_PX = 720
CAMERA_HEIGHT_PX = 480
CAMERA_HORIZONTAL_FOV_RAD = 2.0944
CAMERA_FX = (CAMERA_WIDTH_PX / 2.0) / math.tan(CAMERA_HORIZONTAL_FOV_RAD / 2.0)
CAMERA_FY = CAMERA_FX  # assume square pixels; only the horizontal FOV is specified
CAMERA_CX = CAMERA_WIDTH_PX / 2.0
CAMERA_CY = CAMERA_HEIGHT_PX / 2.0

# NOTE: the bridged depth image's header.frame_id is (incorrectly) 'camera_link', which is a
# body-convention frame (x-forward/y-left/z-up) - see <gz_frame_id>camera_link</gz_frame_id> in
# mars_explorer.gazebo.xacro. The pixel -> 3D unprojection below assumes the standard optical
# convention (x-right/y-down/z-forward), so we transform using the *_optical_frame from the URDF
# instead of trusting the message header, otherwise localised positions would be off by the
# fixed -90/0/-90 rpy rotation between the two (see camera_depth_optical_joint in
# mars_explorer.urdf.xacro).
CAMERA_DEPTH_OPTICAL_FRAME = 'camera_depth_optical_frame'

# Perception 3: repeated observations of the same artefact type within this distance (metres)
# of an existing estimate are merged into it (running average) rather than creating a new one.
ARTIFACT_CLUSTER_DISTANCE_M = 1.5

# Perception/Planning 3 robustness: a same-label cluster within this distance (metres) of an
# existing one is treated as a repeat sighting of the same physical object (drifted further
# than ARTIFACT_CLUSTER_DISTANCE_M due to localisation noise) and merged into it too, as long as
# that existing cluster is already CONFIRMED (see ARTIFACT_CONFIRMATION_OBSERVATIONS) - see
# add_artifact_observation(). Gating on "already confirmed" (rather than applying this wider
# radius unconditionally) means two genuinely distinct same-label artefacts placed closer than
# this to each other still each get a chance to independently confirm themselves, instead of
# being merged into one from the very first sighting. Deliberately wider than
# ARTIFACT_CLUSTER_DISTANCE_M to absorb that drift, while still tighter than the spacing between
# genuinely distinct artefacts. Previously only applied to the 3 inspection labels (as
# INSPECTION_DUPLICATE_RADIUS_M) - every artefact type ballooned unbounded ids from drifted
# re-sightings without it, not just the inspected ones, so it's now applied to all of them here.
ARTIFACT_DUPLICATE_RADIUS_M = 3.0

# Perception 3 robustness: a single-frame colour/contour blob can be a stray false positive
# (lighting glare, a misclassified rock texture) rather than a real artefact. A cluster isn't
# logged, counted, or offered up for inspection (see find_inspection_target()) until it reaches
# this many merged observations - filters out one-off noise without needing a smarter detector.
ARTIFACT_CONFIRMATION_OBSERVATIONS = 3

# Planning 1: frontier-based exploration.
# Occupancy grid cells >= this value are treated as occupied (cells are 0-100, or -1 if unknown).
FRONTIER_OCCUPIED_THRESHOLD = 50
# Frontier clusters smaller than this many cells are treated as noise and ignored. Raised from
# an earlier, smaller value: tiny clusters are usually just sensor-shadow slivers (behind a
# rock, a corner the lidar grazed) right next to already-explored corridors, not real
# unexplored territory - letting those count as frontiers was the main cause of the robot
# "mopping up" small scraps near where it'd already been instead of pushing into new areas.
FRONTIER_MIN_CLUSTER_SIZE = 15
# How far (metres) a frontier's score is allowed to "travel" before distance starts
# significantly discounting it - see choose_frontier_goal()'s score formula. Larger means
# distance matters less relative to size, so a big far frontier can still beat a tiny close one.
FRONTIER_DISTANCE_SCALE_M = 5.0
# Candidate frontiers within this distance (metres) of a goal sent in the last
# FRONTIER_REVISIT_WINDOW_S seconds are skipped, so we don't repeatedly re-target the same
# spot (e.g. if Nav2 can't quite reach the frontier itself, or a sensor-shadow sliver flickers
# in and out near a path we've already driven).
FRONTIER_REVISIT_RADIUS_M = 4.0
# How long (seconds) a sent goal keeps excluding nearby frontiers for. Needs to be a *window*,
# not permanent: on a long run, dozens of permanently-excluded 4m zones in a corridor-width
# cave can start blanketing genuinely new nearby frontiers too, forcing the robot to fall back
# to a random (possibly far away) point instead of the closest real unexplored bit - observed
# as excessive backtracking across the whole map on long runs.
FRONTIER_REVISIT_WINDOW_S = 180.0

# Planning 1: after this many consecutive checks with a real map available but no frontier
# candidates found, treat exploration as complete (rather than a one-off gap) and say so
# clearly in the logs - distinct from the same "no candidates" result before the map exists.
EXPLORATION_COMPLETE_STREAK = 5

# Planning 2: artefact types the robot will pause exploration to inspect up close. Chosen as
# the 3 most visually distinct profiles in ARTIFACT_COLOR_PROFILES (per that list's own
# comments, e.g. green_alien/toy_story_alien and white_sphere/ice_formation are more likely
# to be confused with each other, so are left out).
INSPECTION_ARTIFACT_LABELS = ['mushroom_blue', 'white_sphere', 'green_crystals']

# Planning 2: distance (metres) to stop from an artefact for close-range inspection. A floor
# rather than the final figure - see artifact_standoff_distance(), which pushes it out for
# artefact types whose own body is wider than this.
INSPECTION_STANDOFF_DISTANCE_M = 2.0

# Physical radius (metres) of each artefact type: half the larger horizontal extent of the
# model's mesh bounding box, i.e. the mesh's own bbox times the <scale> it's placed with in
# mars_cave.sdf. How far the solid object actually reaches from the point perception reports.
#
# Needed because standoff distances are measured from the artefact's detected centroid, and two
# of the models are wider than the 2.0m standoff above: the green crystal clusters and the ice
# formations are both roughly 4.5 x 5.2m, so their spikes reach ~2.6m from their own centre. A
# 2.0m standoff puts the goal INSIDE the model.
#
# The map can't be relied on to catch that. The lidar is a single horizontal ray at z = 1.0m
# (scan_joint in mars_explorer.urdf.xacro) and both of those models are clusters of thin spikes
# at that height, so most rays pass between the spikes and clear the cells behind them - the
# artefact ends up barely present in the occupancy grid, a few scattered occupied cells with
# apparently free space between them. is_point_navigable() then quite correctly reports a point
# inside the model as free, Nav2 plans straight through it, and the rover wedges against the
# collision mesh, which is the full dense mesh rather than the thin cross-section the lidar saw.
# Seen live as the rover sitting inside a green crystal cluster with its wheels still turning.
#
# So this trusts perception rather than the map: we know a solid object is at that position
# because we just detected it there, and we know how big that type of object is, which is enough
# to keep goals out of it without needing the lidar to agree.
ARTIFACT_PHYSICAL_RADIUS_M = {
    'green_crystals': 2.6,
    'ice_formation': 2.6,
    'green_alien': 1.2,
    'mushroom_blue': 1.2,
    'white_sphere': 0.8,
}

# Used for any label not in the table above, so a new profile added to ARTIFACT_COLOR_PROFILES
# gets sensible-but-cautious treatment rather than silently defaulting to zero clearance.
ARTIFACT_DEFAULT_PHYSICAL_RADIUS_M = 1.3

# Half the diagonal of base_link's 1x1m collision box (see mars_explorer.urdf.xacro, and the
# matching explicit footprint in nav2_params.yaml): the distance from the rover's centre to its
# furthest corner, i.e. how much room its own body needs around a goal point.
ROVER_FOOTPRINT_RADIUS_M = 0.71

# Extra breathing room (metres) on top of artefact radius + rover radius, so a viewpoint is
# comfortably outside the artefact rather than exactly touching it.
ARTIFACT_STANDOFF_MARGIN_M = 0.4

# Narrowest passage (metres) the rover can actually be sent into, as a tour stop. The rover needs
# a free disc of ROVER_FOOTPRINT_RADIUS_M around its centre to sit anywhere and turn, so that is
# a diameter of twice the footprint radius; anything tighter is unreachable by construction and
# Nav2's inflation layer (inflation_radius 1.0m in nav2_params.yaml) will refuse to plan into it
# regardless. Advanced 2 is right to *report* a 0.6m passage - it just isn't somewhere to drive.
MIN_TOUR_PASSAGE_WIDTH_M = 2.0 * ROVER_FOOTPRINT_RADIUS_M

# Planning 3: if a close-range inspection goal doesn't succeed (Nav2 aborts/rejects it - e.g.
# the artefact's standoff point turned out to be unreachable), retry it this many times before
# abandoning that artefact and resuming exploration.
INSPECTION_MAX_RETRIES = 1

# A failed goal that concluded faster than this (seconds) tells us nothing about the artefact:
# the rover never moved, so it cannot have "lost" anything. Nav2 rejects a goal this fast when
# its behaviour tree couldn't get its own action servers to acknowledge the request in time -
# see default_server_timeout in nav2_params.yaml, where the live evidence is written up. It is
# a failure of the navigation stack's internal plumbing, not a fact about the world.
#
# Separating the two cases matters because INSPECTION_MAX_RETRIES is small on purpose. With
# every failure counted alike, two consecutive 0s handshake timeouts - which take well under a
# second between them - were enough to write an artefact off permanently. That is exactly what
# happened to green_crystals #62 in live testing: two identical instant aborts on the same
# 42.6m goal, abandoned without the rover ever having driven towards it.
GOAL_INSTANT_FAILURE_S = 1.0

# How many instant failures (see GOAL_INSTANT_FAILURE_S) an inspection target may accumulate
# before they start counting as real attempts after all. Needed as a separate bound because a
# genuinely unplannable goal also fails instantly, every time, forever - one live goal really
# was rejected with "GridBased failed to generate a valid path" rather than a handshake
# timeout, and without a ceiling here that artefact would be retried for the whole mission.
INSPECTION_MAX_INSTANT_RETRIES = 3

# Planning 1 robustness: how many random points planner_random_walk() tries before giving up
# and holding position, when looking for one that's actually navigable (see is_point_navigable()).
RANDOM_WALK_MAX_ATTEMPTS = 30

# How often (seconds) to print the [STATUS] heartbeat - see log_status().
STATUS_LOG_PERIOD_S = 30.0

# How often (seconds) to publish the GUI status feed - see publish_gui_status(). Much more
# frequent than STATUS_LOG_PERIOD_S since this drives a live-updating display rather than a
# human-readable log heartbeat, where that cadence would feel sluggish.
GUI_STATUS_PERIOD_S = 1.0

# Planning 1/3 robustness: Nav2's own recovery behaviour tree can retry a difficult goal (e.g.
# "Failed to make progress") more or less indefinitely without ever reporting success or
# failure back to us. If a goal hasn't finished within this many seconds, we give up on it
# ourselves - see the watchdog at the top of main_loop() - rather than waiting forever.
# Lowered from 60s (stuck goals rarely resolved themselves after the first ~20-30s anyway, so
# waiting the full 60s was mostly wasted "frozen" time) but kept at 3x the controller's
# progress_checker movement_time_allowance (15s in nav2_params.yaml), not a flat 30s: those two
# values interact - if the timeout is too short relative to the check interval, Nav2's own
# recovery behaviours (spin/backup/wait) don't get enough cycles to actually resolve a
# difficult-but-recoverable spot before we give up on it ourselves.
GOAL_TIMEOUT_S = 45.0

# Planning 3 robustness: RETURN_HOME's distance-scaled timeout assumes the robot can average at
# least this speed (m/s) over the whole trip home - see planner_return_home(). Deliberately
# conservative (well under max_vel_x in nav2_params.yaml) to leave room for turns, narrow
# corridors, and the occasional recovery cycle without falsely running out of patience.
RETURN_HOME_MIN_SPEED_MPS = 0.3

# Advanced 5 robustness: the same idea for science tour legs - see planner_science_tour(). Tour
# legs are the longest goals the mission ever sends: the route is a closed loop over the whole
# explored cave, so a leg routinely spans 20-40m (the first leg is worst, since the tour starts
# from wherever exploration happened to leave the rover and stop 1 is base). The flat
# GOAL_TIMEOUT_S ceiling cancelled every single stop in live testing - 36m to base abandoned at
# 46s, then four more at 42-44s, 0 of 12 stops reached - because 36m in 45s needs 0.8m/s
# sustained with no turning. Same conservative figure as RETURN_HOME_MIN_SPEED_MPS, kept as its
# own constant because the two behaviours are tuned for different reasons and shouldn't drift
# together by accident.
TOUR_MIN_SPEED_MPS = 0.3

# Planning 1/3 robustness: a companion to GOAL_TIMEOUT_S that reacts to Nav2's own
# distance_remaining feedback (see feedback_callback()) rather than waiting out a flat
# duration regardless of whether the robot is actually making progress. If distance_remaining
# hasn't decreased by more than GOAL_PROGRESS_EPSILON_M for this many seconds, the goal is
# abandoned early instead of waiting the full GOAL_TIMEOUT_S.
#
# Deliberately NOT set close to GOAL_TIMEOUT_S's lesson learned in reverse: too short a value
# here would repeat that same mistake - cutting a goal off before Nav2's own progress_checker
# (movement_time_allowance=15s in nav2_params.yaml) has even finished its first stall-detect
# cycle, let alone had a chance to actually run a recovery behaviour (spin/backup/wait) and
# resume making progress.
#
# Raised from 30s (2x movement_time_allowance, room for one detect-then-recover cycle) after
# live testing: once the global costmap's rolling_window bug was fixed, the robot started
# attempting much longer frontier goals (30-50m, across the whole explored map instead of a
# ~15m window) - and over that much longer route, through a cave whose map is still being
# actively discovered, there's more opportunity to hit one genuinely tricky spot (a tight
# corridor, a junction, a plan disrupted by newly-discovered obstacles) that needs more than
# one recovery cycle to resolve. All 3 early-abandons observed in that test run fired at
# 30-32s on long-distance goals that were otherwise making real progress - i.e. Nav2's single
# recovery cycle hadn't finished, not that the goal was truly dead. 40s gives room for closer
# to two cycles, while still meaningfully faster than the 45s hard backstop for a goal that
# really is stuck.
GOAL_PROGRESS_STALL_S = 40.0

# How much (metres) distance_remaining must decrease to count as "real" forward progress,
# rather than feedback noise/jitter resetting the GOAL_PROGRESS_STALL_S timer forever.
GOAL_PROGRESS_EPSILON_M = 0.15

# Advanced 2: how often (seconds) to re-run the cave geometry analysis - see
# update_cave_geometry(). The task asks for the distance transform to be computed/updated
# online, so this runs throughout the mission rather than once at the end. Deliberately much
# slower than the 0.2s main loop: the analysis is a handful of whole-map numpy/OpenCV passes
# (distance transform, blur, four Sobels) plus a full-map BFS for the reachability mask, and the
# results only change as fast as SLAM reveals new cave - several seconds between updates is
# visually continuous in RViz while leaving the CPU to Nav2 and this node's own camera/tf
# callbacks (the same starvation concern documented on compute_path_distance_grid()).
GEOMETRY_UPDATE_PERIOD_S = 5.0

# Advanced 2: clearance (metres) displayed as the top of the colour scale when the distance
# transform is published as an OccupancyGrid overlay - see publish_clearance_map(). Anything
# more open than this saturates. Occupancy grids only carry 0-100, so the clearance has to be
# squashed into that range somehow; this makes the mapping explicit and readable rather than
# dependent on whatever the widest cell on the current map happens to be (which would make the
# colours mean something different every update).
CLEARANCE_DISPLAY_MAX_M = 5.0

# Advanced 2: features closer together than this (metres) in the geometry analysis are treated
# as the same place. Mirrors cave_geometry.MIN_FEATURE_SEPARATION_M, and is also used when
# assembling the science tour so two "different" stops can't be the same spot twice.
GEOMETRY_FEATURE_SEPARATION_M = 4.0

# Advanced 5: the science tour visits between this many locations. The lower bound is the
# task's own requirement (at least 8 meaningful locations); the upper bound keeps the tour
# driveable in a demo run and keeps cost-matrix construction cheap - it costs one full-map BFS
# per location (see build_tour_cost_matrix()), so this is the knob that bounds that work.
TOUR_MIN_LOCATIONS = 8
TOUR_MAX_LOCATIONS = 12

# Advanced 5: how far from an artefact the tour stops - so a science stop is a viewpoint of the
# artefact rather than a point inside it - comes from artifact_standoff_distance(), shared with
# the Planning 2 inspection standoff for consistency between the two behaviours. It's per-label
# rather than a single constant because the artefact's own size decides it; an earlier single
# constant here assumed the artefact would show up in the map as an obstacle to plan around,
# which turned out not to hold (see ARTIFACT_PHYSICAL_RADIUS_M).

# Advanced 5: two tour stops closer together than this (metres) are treated as the same place
# and only one is kept - see collect_science_locations(). Smaller than
# GEOMETRY_FEATURE_SEPARATION_M because stops come from different sources (an artefact viewpoint
# and a chamber centre can legitimately be a few metres apart and still be two different
# reasons to go somewhere); this only needs to catch stops that are effectively identical.
TOUR_MIN_STOP_SEPARATION_M = 2.5

# Advanced 5: how far (metres) to search outward for a navigable cell when a nominal tour
# location isn't driveable - see find_nearest_navigable_point(). Geometry features are always
# in free space by construction, but artefact positions are estimates that can land inside a
# wall or inside the artefact's own footprint, so they need nudging to somewhere reachable.
TOUR_SNAP_SEARCH_RADIUS_M = 3.0


class CaveExplorer(Node):
    def __init__(self):
        super().__init__('cave_explorer_node')

        # Variables/Flags for mapping
        self.xlim_ = [0.0, 0.0]
        self.ylim_ = [0.0, 0.0]

        # Planning 1: full occupancy grid (kept in addition to xlim_/ylim_ above), used for
        # frontier detection. 'map_grid_' is a (height, width) int8 numpy array: -1 = unknown,
        # 0-99 = free (increasing cost), 100 = occupied.
        self.map_grid_ = None
        self.map_resolution_ = None
        self.map_origin_ = None
        self.map_width_ = None
        self.map_height_ = None

        # Variables/Flags for perception
        self.artifact_found_ = False

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.ERROR

        # Planning 1: (x, y, time_sent) of frontier goals sent within the last
        # FRONTIER_REVISIT_WINDOW_S - see that constant's comment above.
        self.recent_frontier_goals_ = []
        self.frontier_markers_pub_ = self.create_publisher(MarkerArray, 'frontier_markers', 1)
        # Planning 1: consecutive ticks with a real map but no frontier candidates - see
        # EXPLORATION_COMPLETE_STREAK and planner_explore_frontier().
        self.no_frontier_streak_ = 0
        # Set once the cave is fully mapped (sticky - never reset back to False). Once true
        # and every known artefact is visited/abandoned, main_loop sends the robot home
        # instead of continuing to explore - see the mission-complete handling there.
        self.exploration_complete_ = False
        # Set once the robot has returned home after exploration_complete_ and no artefacts
        # remain pending - main_loop stops doing anything further once this is true.
        self.mission_complete_ = False

        # Planning 2/3: the artefact cluster currently being approached for close-range
        # inspection (a dict from artifact_clusters_), or None if not inspecting.
        self.inspection_target_ = None
        # Planning 3: ids of artefact clusters already successfully inspected, so each is
        # only visited once, and ids abandoned after repeated failed approach attempts (kept
        # separate from visited_ so the report can distinguish the two; both are excluded from
        # future targeting by find_inspection_target(), otherwise an unreachable artefact would
        # be retried forever instead of actually resuming exploration). Also: how many approach
        # attempts made for the current target so far (see INSPECTION_MAX_RETRIES), and whether
        # the most recently completed Nav2 goal succeeded (set by goal_response_callback/
        # goal_reached_callback below).
        self.visited_artifact_ids_ = set()
        self.abandoned_artifact_ids_ = set()
        self.inspection_attempts_ = 0
        # Instant failures for the current target, counted separately from inspection_attempts_
        # so Nav2 plumbing hiccups can't write an artefact off - see GOAL_INSTANT_FAILURE_S.
        self.inspection_instant_failures_ = 0
        self.last_goal_succeeded_ = None
        # How long the most recently concluded goal lasted, in seconds. Recorded by
        # log_goal_outcome(), which every goal conclusion path already funnels through, so
        # main_loop can tell "Nav2 never started" from "the rover tried and failed".
        self.last_goal_duration_s_ = 0.0
        self.visited_markers_pub_ = self.create_publisher(
            MarkerArray, 'artifact_inspection_status_markers', 1)

        # Perception 3: clustered artefact location estimates.
        # Each entry is a dict: {'id': int, 'label': str, 'position': Point, 'num_observations': int}
        # 'position' is a running average over all observations merged into this cluster.
        # 'id' (Planning 3) is a stable identity for this cluster, used for visited-tracking.
        self.artifact_clusters_ = []
        self.next_artifact_cluster_id_ = 0
        self.marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_artifacts', 10)

        # Initialise CvBridge
        self.cv_bridge_ = CvBridge()

        # Prepare transformation to get robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Action client for nav2
        self.nav2_action_client_ = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().warn('Waiting for navigate_to_pose action...')
        self.nav2_action_client_.wait_for_server()
        self.get_logger().warn('navigate_to_pose connected')
        self.ready_for_next_goal_ = True
        self.declare_parameter('print_feedback', rclpy.Parameter.Type.BOOL)

        # Robustness: tracks the most recently sent goal, so main_loop's watchdog (see
        # GOAL_TIMEOUT_S) can give up on one that's taking too long, and so a late callback
        # for an already-abandoned goal can be recognised as stale and ignored rather than
        # corrupting the state of whatever goal we've since moved on to.
        self.goal_generation_ = 0
        self.goal_sent_time_ = None
        # The flat watchdog ceiling for the current goal - normally GOAL_TIMEOUT_S, but
        # overridden per-goal by planner_go_to_pose2d()'s 'timeout_s' - see
        # planner_return_home() for why RETURN_HOME needs a larger one.
        self.goal_timeout_s_ = GOAL_TIMEOUT_S
        # The in-flight goal's handle (set once accepted - see goal_response_callback), so
        # main_loop's watchdog can actually cancel it with Nav2 when giving up, rather than
        # just locally pretending it's gone while Nav2 keeps executing it regardless.
        self.current_goal_handle_ = None

        # Robustness: tracks progress on the current goal via the robot's own straight-line
        # distance to the goal (x, y) - computed fresh from get_pose_2d() in main_loop's
        # watchdog, NOT from Nav2's distance_remaining feedback. distance_remaining is the
        # length of Nav2's current PLAN, which gets recomputed every time the costmap updates
        # (frequent, since SLAM keeps discovering new area mid-drive) - a replan that finds a
        # slightly longer detour is needed can make it plateau or tick up for a stretch even
        # while the robot is genuinely, correctly moving toward the goal. The longer the goal,
        # the more replans happen en route, the more likely this is to look like a stall when
        # it isn't one - confirmed live: every false stall observed was on a goal 30m+ away,
        # abandoned right at the threshold with the robot already having covered over half the
        # distance. Straight-line distance to the fixed goal point can't be fooled this way -
        # it only decreases as the robot genuinely gets physically closer.
        self.goal_target_x_ = None
        self.goal_target_y_ = None
        self.goal_progress_distance_ = None
        self.goal_progress_time_ = None

        # Logging: straight-line distance to the current goal at the moment it was sent (used
        # to report per-leg average speed when it concludes - see log_goal_outcome()), and the
        # running total across the whole mission (used for MISSION COMPLETE's overall average).
        self.goal_initial_distance_m_ = None
        self.total_distance_m_ = 0.0

        # Publisher for the goal pose visualisation
        self.goal_pose_vis_ = self.create_publisher(PoseStamped, 'goal_pose', 1)

        # Subscribe to the map topic to get current bounds
        self.map_sub_ = self.create_subscription(OccupancyGrid, 'map',  self.map_callback, 1)

        # Prepare image processing
        self.image_detections_pub_ = self.create_publisher(Image, 'detections_image', 1)
        self.declare_parameter('computer_vision_model_filename', rclpy.Parameter.Type.STRING)
        self.computer_vision_model_ = cv2.CascadeClassifier(self.get_parameter('computer_vision_model_filename').value)
        self.classifier_interpreter_, self.classifier_labels_ = self.load_artifact_classifier()
        self.image_sub_ = self.create_subscription(Image, 'camera/image', self.image_callback, 1)

        # Perception 3: depth image, used to localise detected artefacts (see localise_artifacts())
        self.latest_depth_image_ = None
        self.depth_image_sub_ = self.create_subscription(Image, 'camera/depth/image', self.depth_image_callback, 1)

        # Perception 1: dataset collection.
        # If 'dataset_dir' is set, raw camera frames are periodically saved there (at most
        # once every 'dataset_save_period' seconds) to build a training/test image dataset.
        # A 'save_dataset_image' service is also provided to save the current frame on demand,
        # e.g. while teleoperating the robot up to an artefact of interest.
        self.declare_parameter('dataset_dir', '')
        self.declare_parameter('dataset_save_period', 2.0)
        self.dataset_dir_ = self.get_parameter('dataset_dir').value
        self.dataset_save_period_ = self.get_parameter('dataset_save_period').value
        self.last_dataset_save_time_ = self.get_clock().now()
        self.latest_image_ = None
        if self.dataset_dir_:
            os.makedirs(self.dataset_dir_, exist_ok=True)
            self.get_logger().info(f'Saving dataset images to: {self.dataset_dir_}')
        self.save_dataset_image_srv_ = self.create_service(
            Trigger, 'save_dataset_image', self.save_dataset_image_callback)

        # GUI control: the robot does nothing in main_loop() until this is triggered (e.g. by
        # the GUI's "Start Exploring" button), rather than autonomously starting the instant
        # the node launches. Perception (image_callback etc.) keeps running regardless, so
        # artefacts already in view get detected immediately once exploration does start.
        self.mission_started_ = False
        self.start_mission_srv_ = self.create_service(
            Trigger, 'start_mission', self.start_mission_callback)

        # GUI control: pause/resume - see pause_mission_callback()/resume_mission_callback().
        self.mission_paused_ = False
        # Set when pausing cancelled an in-flight goal, so the behaviour it belonged to can tell
        # that apart from the goal genuinely failing - see pause_mission_callback().
        self.goal_cancelled_by_pause_ = False
        self.pause_mission_srv_ = self.create_service(
            Trigger, 'pause_mission', self.pause_mission_callback)
        self.resume_mission_srv_ = self.create_service(
            Trigger, 'resume_mission', self.resume_mission_callback)

        # GUI control: skip straight to RETURN_HOME - see force_return_home_callback().
        self.force_return_home_srv_ = self.create_service(
            Trigger, 'force_return_home', self.force_return_home_callback)

        # Advanced 2: cave geometry analysis - see update_cave_geometry(). 'geometry_' holds the
        # latest result from cave_geometry.analyse_cave() (the clearance map plus the chamber and
        # bottleneck lists), recomputed on a timer as the map grows.
        self.geometry_ = None
        self.geometry_markers_pub_ = self.create_publisher(MarkerArray, 'cave_geometry_markers', 1)
        # The distance transform itself, republished as an OccupancyGrid so it can be displayed
        # in RViz with a normal Map display. Latched (transient-local) QoS to match RViz's own
        # default for Map displays, so a display added mid-run gets the current analysis
        # immediately instead of showing nothing until the next GEOMETRY_UPDATE_PERIOD_S tick.
        self.clearance_map_pub_ = self.create_publisher(
            OccupancyGrid, 'clearance_map',
            QoSProfile(depth=1,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=QoSReliabilityPolicy.RELIABLE))
        # Logging only: the headline widest/narrowest values last reported, so update_cave_geometry()
        # can log when the answer actually changes instead of every few seconds forever.
        self.last_logged_widest_m_ = None
        self.last_logged_narrowest_m_ = None
        self.geometry_timer_ = self.create_timer(
            GEOMETRY_UPDATE_PERIOD_S, self.update_cave_geometry)

        # Advanced 5: science tour. 'tour_' is the planned route (see plan_science_tour()):
        # the chosen locations, the nearest-neighbour baseline, the 2-opt optimised order, and
        # the cost comparison between them. 'tour_stop_index_' is how far through the optimised
        # order execution has got.
        self.tour_ = None
        self.tour_stop_index_ = 0
        self.tour_complete_ = False
        # Set when the tour can't be planned (not enough distinct reachable locations), so it
        # isn't retried on every tick for the rest of the mission.
        self.tour_planning_failed_ = False
        # Set by the start_science_tour service, to run the tour on demand mid-mission (e.g. for
        # a demo) rather than waiting for exploration to finish naturally.
        self.tour_requested_ = False
        self.tour_markers_pub_ = self.create_publisher(MarkerArray, 'science_tour_markers', 1)
        self.start_science_tour_srv_ = self.create_service(
            Trigger, 'start_science_tour', self.start_science_tour_callback)
        # Whether to run the tour automatically once exploration completes. On by default (it's
        # the natural end-of-mission behaviour: the cave is mapped, so now go and revisit
        # everything worth revisiting), but can be turned off to get the plain Planning 1-3
        # behaviour back without editing code.
        self.declare_parameter('science_tour_on_completion', True)

        # Timer for main loop
        self.main_loop_timer_ = self.create_timer(0.2, self.main_loop)

        # Periodic status heartbeat (elapsed time + progress counts) - see log_status(). Set
        # once start_mission_callback() actually starts the clock, not at node construction,
        # so elapsed time doesn't include time spent waiting for the GUI's start button.
        self.start_time_ = None
        self.status_timer_ = self.create_timer(STATUS_LOG_PERIOD_S, self.log_status)

        # GUI data feed: a single JSON-encoded topic (mission status + the full artefact
        # list), published frequently enough for a live-updating display - see
        # publish_gui_status(). Kept as one plain std_msgs/String rather than a custom .msg
        # package, to avoid an interfaces-package rebuild for what's purely a GUI convenience.
        self.gui_status_pub_ = self.create_publisher(String, 'gui_status', 1)
        self.gui_status_timer_ = self.create_timer(GUI_STATUS_PERIOD_S, self.publish_gui_status)
    
    def get_pose_2d(self):
        """Get the 2d pose of the robot"""

        # Lookup the latest transform
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().error(f'Could not transform: {ex}')
            return

        # Return a Pose2D message
        pose = Pose2D()
        pose.x = t.transform.translation.x
        pose.y = t.transform.translation.y

        qw = t.transform.rotation.w
        qz = t.transform.rotation.z

        if qz >= 0.:
            pose.theta = wrap_angle(2. * math.acos(qw))
        else:
            pose.theta = wrap_angle(-2. * math.acos(qw))

        return pose

    def map_callback(self, map_msg: OccupancyGrid):
        """New map received: update x/y bounds, and keep the full grid for frontier detection"""

        # Extract data from message
        map_origin = [map_msg.info.origin.position.x,
                      map_msg.info.origin.position.y]
        map_resolution = map_msg.info.resolution
        map_height = map_msg.info.height
        map_width = map_msg.info.width

        # Set current limits
        self.xlim_ = [map_origin[0], map_origin[0]+map_width*map_resolution]
        self.ylim_ = [map_origin[1], map_origin[1]+map_height*map_resolution]

        # Planning 1: keep the full grid (reshaped to (height, width)) so planner_explore_frontier()
        # can find the boundary between free and unknown space.
        self.map_resolution_ = map_resolution
        self.map_origin_ = map_origin
        self.map_width_ = map_width
        self.map_height_ = map_height
        self.map_grid_ = np.array(map_msg.data, dtype=np.int8).reshape((map_height, map_width))

        # self.get_logger().warn('Map received:')
        # self.get_logger().warn(f'  xlim = [{self.xlim_[0]:.2f}, {self.xlim_[1]:.2f}]')
        # self.get_logger().warn(f'  ylim = [{self.ylim_[0]:.2f}, {self.ylim_[1]:.2f}]')
    
    def image_callback(self, image_msg):
        """
        Recieve an RGB image.
        Use this method to detect artifacts of interest.
        
        A simple method has been provided to begin with for detecting stop signs (which is not what we're actually looking for) 
        adapted from: https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/
        """
    
        # Copy the image message to a cv image
        # see http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

        # Remember the latest raw frame (used by the dataset-collection service below)
        # and periodically save frames to build a Perception 1 image dataset.
        self.latest_image_ = image
        self.maybe_save_dataset_image(image)

        # Perception 2: run all detectors and merge their results. The stop sign's dedicated
        # cascade classifier is given priority over any colour-profile detection landing on
        # the same spot (see suppress_overlapping_labels()'s 'priority_labels') - e.g. its
        # white border/pole occasionally gets mistaken for ice_formation's "icy" colour range.
        detections = suppress_overlapping_labels(
            self.detect_stop_signs(image) + self.detect_color_artifacts(image),
            priority_labels={'stop_sign'})

        # Advanced 1: re-examine each HSV detection with the trained classifier - confirms,
        # relabels (fixes cross-label confusion like green_alien/green_crystals), or drops it
        # entirely (if the classifier is confident it's actually 'negative' - a false positive).
        # See CLASSIFIER_* constants' comments above for why this exists.
        detections = self.refine_detections_with_classifier(image, detections)

        # You can set "artifact_found_" to true to signal to "main_loop" that you have found a artifact
        # Since the "image_callback" and "main_loop" methods can run at the same time you should protect any shared variables
        # with a mutex
        # "artifact_found_" doesn't need a mutex because it's an atomic
        self.artifact_found_ = len(detections) > 0

        # Draw a bounding box + label for each detection
        annotated_image = image.copy()
        for detection in detections:
            x, y, width, height = detection.bbox
            cv2.rectangle(annotated_image, (x, y), (x + width, y + height), detection.color_rgb, 3)
            cv2.putText(annotated_image, detection.label, (x, max(y - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection.color_rgb, 2)

        # Publish the image with the detection bounding boxes
        image_detection_message = self.cv_bridge_.cv2_to_imgmsg(annotated_image, encoding="rgb8")
        self.image_detections_pub_.publish(image_detection_message)

        if self.artifact_found_:
            # No per-frame log here - it fires on nearly every camera frame when something's
            # in view, which drowned out every other log line. add_artifact_observation()
            # below logs each genuinely NEW artefact once, which is the useful signal.
            self.localise_artifacts(detections)

    def depth_image_callback(self, image_msg):
        """Perception 3: remember the latest depth frame, used by localise_artifacts() below"""

        self.latest_depth_image_ = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

    def detect_stop_signs(self, image):
        """
        Detect stop signs using the provided cascade classifier
        (a placeholder for real artefact detection)
        adapted from: https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/
        """

        # The minSize is used to avoid very small detections that are probably noise
        raw_detections = self.computer_vision_model_.detectMultiScale(image, minSize=(20, 20))

        return [Detection('stop_sign', (x, y, w, h), (0, 255, 0))
                for (x, y, w, h) in raw_detections]

    def detect_color_artifacts(self, image):
        """
        Perception 2: detect the non-stop-sign artefact types using HSV colour thresholding
        and contour/blob detection, based on ARTIFACT_COLOR_PROFILES.

        Several profiles have deliberately similar colour ranges (see that list's own comment),
        so the same object can match more than one at once - suppress_overlapping_labels()
        resolves that by keeping only the best (largest-area) match per overlapping region,
        rather than reporting one physical object as several different artefact types.
        """

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        kernel = np.ones((5, 5), np.uint8)

        detections = []
        for profile in ARTIFACT_COLOR_PROFILES:
            mask = cv2.inRange(hsv_image, profile['hsv_lower'], profile['hsv_upper'])

            # Clean up noise, then close small gaps within a single artefact's blob
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [cv2.boundingRect(contour) for contour in contours
                     if cv2.contourArea(contour) >= profile['min_area']]

            # A single real artefact can still produce several disconnected contours (shading,
            # texture, specular highlights), which would otherwise show up as several separate
            # detections for what's visually one object - merge any that are close together.
            for bbox in merge_overlapping_boxes(boxes):
                detections.append(Detection(profile['label'], bbox, profile['color_rgb']))

        return suppress_overlapping_labels(detections)

    def load_artifact_classifier(self):
        """
        Advanced 1: load the trained artefact classifier (see CLASSIFIER_* constants' comment
        above, and build_classifier_dataset.py for how its training data was produced).

        Returns (interpreter, labels) - labels is a list indexed to match the model's output
        tensor order. Returns (None, None) if tflite_runtime isn't installed or no exported
        model is present yet, so the live detector still runs (just without this refinement
        step) rather than crashing a team member's setup that hasn't done this part.
        """

        if tflite is None:
            self.get_logger().warn(
                'tflite_runtime not installed - skipping artefact classifier (HSV-only detection)')
            return None, None

        model_dir = os.path.join(
            get_package_share_directory('cave_explorer'), 'config', 'artifact_classifier')
        model_path = os.path.join(model_dir, 'model.tflite')
        labels_path = os.path.join(model_dir, 'labels.txt')
        if not os.path.exists(model_path) or not os.path.exists(labels_path):
            self.get_logger().warn(
                f'No trained artefact classifier found at {model_dir} - skipping (HSV-only detection)')
            return None, None

        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # labels.txt lines look like "0 green_alien" - the index prefix is Teachable Machine's
        # own class ordering, already matching the model's output tensor order; only the label
        # text after it is needed here.
        labels = []
        with open(labels_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line.split(' ', 1)[1])

        self.get_logger().info(f'Loaded artefact classifier: {len(labels)} classes from {model_dir}')
        return interpreter, labels

    def classify_crop(self, image, bbox):
        """
        Advanced 1: run the trained classifier on the region of 'image' (RGB) given by 'bbox',
        returning (predicted_label, confidence). Returns (None, 0.0) if the classifier isn't
        loaded (see load_artifact_classifier()).

        Crops with the same margin build_classifier_dataset.py used when producing the
        training images (CLASSIFIER_CROP_MARGIN_PX) - the classifier needs to see the same
        kind of crop live that it was actually trained on, or this would quietly become a
        train/inference mismatch and the model's real-world accuracy would no longer match
        what training suggested.
        """

        if self.classifier_interpreter_ is None:
            return None, 0.0

        x, y, w, h = bbox
        height, width = image.shape[:2]
        margin = CLASSIFIER_CROP_MARGIN_PX
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(width, x + w + margin), min(height, y + h + margin)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return None, 0.0

        resized = cv2.resize(crop, (CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE))
        # Teachable Machine's standard MobileNet preprocessing: pixels in [0, 255] -> [-1, 1].
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        input_tensor = np.expand_dims(normalized, axis=0)

        interpreter = self.classifier_interpreter_
        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()
        probabilities = interpreter.get_tensor(output_index)[0]

        best_index = int(np.argmax(probabilities))
        return self.classifier_labels_[best_index], float(probabilities[best_index])

    def refine_detections_with_classifier(self, image, detections):
        """
        Advanced 1: pass each HSV detection (other than CLASSIFIER_SKIP_LABELS) through the
        trained classifier, using it to confirm, relabel, or reject (drop) the HSV guess - see
        CLASSIFIER_* constants' comment above for why this exists and what it fixes.

        A low-confidence classifier opinion (below CLASSIFIER_CONFIDENCE_THRESHOLD) is treated
        as inconclusive rather than acted on - the original HSV label passes through unchanged
        in that case, rather than risking an uncertain classifier result overriding a perfectly
        good HSV match.
        """

        if self.classifier_interpreter_ is None:
            return detections

        refined = []
        for detection in detections:
            if detection.label in CLASSIFIER_SKIP_LABELS:
                refined.append(detection)
                continue

            predicted_label, confidence = self.classify_crop(image, detection.bbox)
            if predicted_label is None or confidence < CLASSIFIER_CONFIDENCE_THRESHOLD:
                refined.append(detection)
            elif predicted_label == CLASSIFIER_NEGATIVE_LABEL:
                self.get_logger().info(
                    f'    CLASSIFIER: rejected HSV {detection.label} as negative '
                    f'(confidence {confidence:.2f})')
                continue  # classifier confident this is a false positive - drop it
            elif predicted_label != detection.label:
                self.get_logger().info(
                    f'    CLASSIFIER: relabelled HSV {detection.label} -> {predicted_label} '
                    f'(confidence {confidence:.2f})')
                color_rgb = ARTIFACT_COLOR_BY_LABEL.get(predicted_label, detection.color_rgb)
                refined.append(Detection(predicted_label, detection.bbox, color_rgb))
            else:
                refined.append(detection)

        return refined

    def maybe_save_dataset_image(self, image):
        """Perception 1: save the current frame to 'dataset_dir_' if enough time has passed"""

        if not self.dataset_dir_:
            return

        elapsed = (self.get_clock().now() - self.last_dataset_save_time_).nanoseconds / 1e9
        if elapsed < self.dataset_save_period_:
            return

        self.last_dataset_save_time_ = self.get_clock().now()
        self.save_image_to_dataset(image)

    def save_image_to_dataset(self, image):
        """Perception 1: write a single frame out to 'dataset_dir_' as a timestamped PNG"""

        timestamp = self.get_clock().now().nanoseconds
        filename = os.path.join(self.dataset_dir_, f'frame_{timestamp}.png')

        # image is RGB (from cv_bridge passthrough); cv2.imwrite expects BGR
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self.get_logger().info(f'Saved dataset image: {filename}')
        return filename

    def save_dataset_image_callback(self, request, response):
        """Service to save the current frame on demand, e.g. while teleoperating up to an artefact"""

        if not self.dataset_dir_:
            response.success = False
            response.message = "The 'dataset_dir' parameter is not set, so there's nowhere to save to"
            return response

        if self.latest_image_ is None:
            response.success = False
            response.message = 'No camera image received yet'
            return response

        filename = self.save_image_to_dataset(self.latest_image_)
        response.success = True
        response.message = f'Saved to {filename}'
        return response

    def start_mission_callback(self, request, response):
        """
        Service to start autonomous exploration - see mission_started_'s comment for why this
        is a manual trigger rather than starting the instant the node launches.
        """

        if self.mission_started_:
            response.success = False
            response.message = 'Mission already started'
            return response

        self.mission_started_ = True
        self.start_time_ = self.get_clock().now()
        self.get_logger().info('Mission started')
        response.success = True
        response.message = 'Mission started'
        return response

    def pause_mission_callback(self, request, response):
        """
        Service to pause: main_loop() returns immediately while paused (see its gate), and the
        in-flight goal is cancelled so the robot actually stops rather than continuing to
        drive toward wherever it was already headed.

        RETURN_HOME is deliberately left running rather than cancelled: main_loop's
        RETURN_HOME result-handling can't tell a manual cancellation apart from a genuine
        failure, so cancelling it here would risk declaring MISSION COMPLETE with "didn't
        quite reach base" on resume, even though the robot was simply paused a few metres from
        home. Letting it finish the (usually short) final leg avoids that, at the cost of not
        instantly freezing during that one specific window.
        """

        if not self.mission_started_:
            response.success = False
            response.message = 'Mission not started yet'
            return response
        if self.mission_paused_:
            response.success = False
            response.message = 'Already paused'
            return response

        self.mission_paused_ = True
        if self.planner_type_ != PlannerType.RETURN_HOME and self.current_goal_handle_ is not None:
            self.current_goal_handle_.cancel_goal_async()
            # Cancelling makes the goal report as not-succeeded, which is indistinguishable from
            # a genuine failure to whatever is tracking the current goal's outcome. Flagged here
            # so a cancelled-by-pause goal can be retried on resume rather than counted as a
            # failure - see advance_science_tour(), where a "failed" tour stop is permanently
            # recorded as missed, so pausing mid-tour (very likely during a demo) would otherwise
            # silently cost a stop.
            self.goal_cancelled_by_pause_ = True
        self.get_logger().info('Mission paused')
        response.success = True
        response.message = 'Mission paused'
        return response

    def resume_mission_callback(self, request, response):
        """Service to resume after pause_mission_callback() - main_loop() picks up normally."""

        if not self.mission_paused_:
            response.success = False
            response.message = 'Not paused'
            return response

        self.mission_paused_ = False
        self.get_logger().info('Mission resumed')
        response.success = True
        response.message = 'Mission resumed'
        return response

    def force_return_home_callback(self, request, response):
        """
        Service to skip straight to RETURN_HOME, as if exploration had completed naturally.
        A pending, already-confirmed inspection target is still visited first (main_loop's
        normal planner-selection order), rather than abruptly abandoning something already
        decided - this only short-circuits further EXPLORE_FRONTIER goals, not inspection.
        """

        if not self.mission_started_:
            response.success = False
            response.message = 'Mission not started yet'
            return response

        self.exploration_complete_ = True
        if self.planner_type_ == PlannerType.EXPLORE_FRONTIER and self.current_goal_handle_ is not None:
            self.current_goal_handle_.cancel_goal_async()
        self.get_logger().info('Force return-home triggered')
        response.success = True
        response.message = 'Returning home'
        return response

    def localise_artifacts(self, detections):
        """
        Perception 3: estimate the world-frame (map) position of each detected artefact and
        merge it into the running per-artefact-type location estimates in 'artifact_clusters_'.

        Direction comes from the detection's pixel location, distance from the depth camera at
        that pixel; together these unproject to a 3D point in the camera frame, which is then
        transformed into the map frame using tf (so it accounts for the robot's current pose
        automatically, rather than us combining robot pose + bearing by hand).
        """

        # The stop sign is a placeholder detector (see image_callback docstring), not a real
        # artefact of interest, so it's excluded from localisation.
        artifact_detections = [d for d in detections if d.label != 'stop_sign']
        if not artifact_detections or self.latest_depth_image_ is None:
            return

        # Look up the camera -> map transform once and reuse it for every detection in this
        # frame (the camera doesn't move between them). See CAMERA_DEPTH_OPTICAL_FRAME's
        # definition above for why we use that frame name rather than the depth image's
        # (incorrect) header.frame_id.
        try:
            camera_to_map_transform = self.tf_buffer.lookup_transform(
                'map', CAMERA_DEPTH_OPTICAL_FRAME, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(f'localise_artifacts: could not transform: {ex}')
            return

        depth_image = self.latest_depth_image_
        depth_height, depth_width = depth_image.shape[:2]

        for detection in artifact_detections:
            x, y, width, height = detection.bbox
            pixel_x = x + width / 2.0
            pixel_y = y + height / 2.0

            # Sample a small patch around the detection's centre pixel and take the median
            # depth, to be robust to individual noisy/missing (inf/NaN) depth readings
            patch_half_size = 2
            row_lo = max(0, int(pixel_y) - patch_half_size)
            row_hi = min(depth_height, int(pixel_y) + patch_half_size + 1)
            col_lo = max(0, int(pixel_x) - patch_half_size)
            col_hi = min(depth_width, int(pixel_x) + patch_half_size + 1)
            depth_patch = depth_image[row_lo:row_hi, col_lo:col_hi]
            valid_depths = depth_patch[np.isfinite(depth_patch) & (depth_patch > 0.0)]
            if valid_depths.size == 0:
                continue
            depth = float(np.median(valid_depths))

            # Unproject the pixel to a 3D point in the camera's optical frame
            # (x-right, y-down, z-forward) using the pinhole camera model
            point_camera = PointStamped()
            point_camera.header.frame_id = CAMERA_DEPTH_OPTICAL_FRAME
            point_camera.point.x = (pixel_x - CAMERA_CX) * depth / CAMERA_FX
            point_camera.point.y = (pixel_y - CAMERA_CY) * depth / CAMERA_FY
            point_camera.point.z = depth

            point_map = tf2_geometry_msgs.do_transform_point(point_camera, camera_to_map_transform)
            self.add_artifact_observation(detection.label, point_map.point)

        self.publish_artifact_markers()

    def add_artifact_observation(self, label, position):
        """
        Perception 3: merge a new observed position for 'label' into an existing nearby cluster
        (running average), or start a new candidate cluster if it's not close to any existing one.

        Two merge radii, in increasing order of looseness:
        - ARTIFACT_CLUSTER_DISTANCE_M: always merge a same-label cluster this close - a normal
          repeat sighting from a similar vantage point.
        - ARTIFACT_DUPLICATE_RADIUS_M: merge a same-label cluster this close ONLY if it's
          already CONFIRMED (see ARTIFACT_CONFIRMATION_OBSERVATIONS) - absorbs drift-induced
          stray detections of an artefact we already know is real, without merging two
          genuinely distinct same-label artefacts placed closer than this to each other before
          either has had a chance to independently confirm itself.

        Logging/counting a brand new cluster is deferred until it reaches
        ARTIFACT_CONFIRMATION_OBSERVATIONS merged observations - see that constant's comment.
        """

        best_cluster, best_distance = None, None
        for cluster in self.artifact_clusters_:
            if cluster['label'] != label:
                continue

            dx = position.x - cluster['position'].x
            dy = position.y - cluster['position'].y
            dz = position.z - cluster['position'].z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)

            is_confirmed = cluster['num_observations'] >= ARTIFACT_CONFIRMATION_OBSERVATIONS
            radius = ARTIFACT_DUPLICATE_RADIUS_M if is_confirmed else ARTIFACT_CLUSTER_DISTANCE_M
            if distance <= radius and (best_distance is None or distance < best_distance):
                best_cluster, best_distance = cluster, distance

        if best_cluster is not None:
            n = best_cluster['num_observations']
            best_cluster['position'].x = (best_cluster['position'].x * n + position.x) / (n + 1)
            best_cluster['position'].y = (best_cluster['position'].y * n + position.y) / (n + 1)
            best_cluster['position'].z = (best_cluster['position'].z * n + position.z) / (n + 1)
            best_cluster['num_observations'] = n + 1
            if best_cluster['num_observations'] == ARTIFACT_CONFIRMATION_OBSERVATIONS:
                self.get_logger().info(
                    f"    FOUND: {label} #{best_cluster['id']} at "
                    f"({best_cluster['position'].x:.1f}, {best_cluster['position'].y:.1f})")
            return

        # Planning 3: a stable id per cluster (distinct from its position in the list, which
        # isn't guaranteed to stay fixed), used to track which specific artefacts have
        # already been inspected - see visited_artifact_ids_. Not logged yet - see
        # ARTIFACT_CONFIRMATION_OBSERVATIONS - in case this is a one-off false positive.
        cluster_id = self.next_artifact_cluster_id_
        self.next_artifact_cluster_id_ += 1

        self.artifact_clusters_.append({
            'id': cluster_id,
            'label': label,
            'position': Point(x=position.x, y=position.y, z=position.z),
            'num_observations': 1,
        })

    def publish_artifact_markers(self):
        """Perception 3: publish one coloured sphere + text label per clustered artefact estimate"""

        marker_array = MarkerArray()
        for i, cluster in enumerate(self.artifact_clusters_):
            color_rgb = ARTIFACT_COLOR_BY_LABEL.get(cluster['label'], (255, 255, 255))
            color = [channel / 255.0 for channel in color_rgb]

            sphere = Marker()
            sphere.header.frame_id = 'map'
            sphere.ns = 'artifacts'
            sphere.id = 2 * i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = cluster['position']
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.6
            sphere.color.a = 1.0
            sphere.color.r, sphere.color.g, sphere.color.b = color
            marker_array.markers.append(sphere)

            label_text = Marker()
            label_text.header.frame_id = 'map'
            label_text.ns = 'artifact_labels'
            label_text.id = 2 * i + 1
            label_text.type = Marker.TEXT_VIEW_FACING
            label_text.action = Marker.ADD
            label_text.pose.position.x = cluster['position'].x
            label_text.pose.position.y = cluster['position'].y
            label_text.pose.position.z = cluster['position'].z + 0.5
            label_text.pose.orientation.w = 1.0
            label_text.scale.z = 0.4
            label_text.color.a = 1.0
            label_text.color.r, label_text.color.g, label_text.color.b = color
            label_text.text = f"{cluster['label']} ({cluster['num_observations']})"
            marker_array.markers.append(label_text)

        self.marker_pub_.publish(marker_array)

    def grid_to_world(self, row, col):
        """Planning 1: convert occupancy-grid (row, col) indices to a map-frame (x, y) point"""

        x = self.map_origin_[0] + (col + 0.5) * self.map_resolution_
        y = self.map_origin_[1] + (row + 0.5) * self.map_resolution_
        return x, y

    def world_to_grid(self, x, y):
        """Planning 2: inverse of grid_to_world() - map-frame (x, y) to occupancy-grid (row, col)"""

        col = int((x - self.map_origin_[0]) / self.map_resolution_)
        row = int((y - self.map_origin_[1]) / self.map_resolution_)
        return row, col

    def is_point_navigable(self, x, y, clearance_cells=3):
        """
        Planning 2: check whether (x, y) is free space (not occupied, not unexplored) in the
        current map, with a small clearance margin - used to validate a candidate inspection
        standoff point before sending it to Nav2, since a point computed purely from the
        artefact's position can otherwise land inside a wall or in unexplored space (which
        Nav2's global planner then simply can't find a path to at all). Also used to validate
        frontier/random-walk goals.

        clearance_cells raised from 2 to 3: a point can be technically "free" right at the edge
        of a wall and still be the kind of tight squeeze that makes the local controller
        struggle to actually execute the approach (repeated "Failed to make progress"), even
        though the global planner found it reachable in principle. A bit more buffer picks
        goals with genuine room around them instead.
        """

        if self.map_grid_ is None:
            return False

        row, col = self.world_to_grid(x, y)
        height, width = self.map_grid_.shape
        row_lo, row_hi = max(0, row - clearance_cells), min(height, row + clearance_cells + 1)
        col_lo, col_hi = max(0, col - clearance_cells), min(width, col + clearance_cells + 1)
        if row_lo >= row_hi or col_lo >= col_hi:
            return False

        patch = self.map_grid_[row_lo:row_hi, col_lo:col_hi]
        return bool(np.all((patch >= 0) & (patch < FRONTIER_OCCUPIED_THRESHOLD)))

    def artifact_standoff_distance(self, label):
        """
        How far from an artefact's detected centroid a viewpoint has to be to stay clear of the
        artefact's own body: its physical radius (see ARTIFACT_PHYSICAL_RADIUS_M) plus the
        rover's, plus a small margin.

        Floored at INSPECTION_STANDOFF_DISTANCE_M so small artefacts are still inspected from
        genuinely close range - this only pushes the viewpoint out for the types that need it,
        rather than backing away from everything to satisfy the widest model in the world.
        """

        radius = ARTIFACT_PHYSICAL_RADIUS_M.get(label, ARTIFACT_DEFAULT_PHYSICAL_RADIUS_M)
        return max(INSPECTION_STANDOFF_DISTANCE_M,
                   radius + ROVER_FOOTPRINT_RADIUS_M + ARTIFACT_STANDOFF_MARGIN_M)

    def is_clear_of_artifacts(self, x, y):
        """
        Is (x, y) outside the physical body of every artefact we currently know about?

        Checks all of them rather than just whichever one we're heading to, because the
        artefacts in this world are often placed in pairs a metre or two apart (a green alien
        standing right beside a crystal cluster, for instance - detected live at (7.9, 3.7) and
        (7.6, 4.3)). A perfectly good standoff point for one can therefore sit inside its
        neighbour, and the same applies to the map-derived tour stops, where a 'narrow passage'
        can turn out to be a gap between two spikes of the same crystal cluster.

        Only confirmed clusters count. An unconfirmed detection could still be a one-off false
        positive, and letting those veto goals would let a bad frame block off real space.
        """

        for cluster in self.artifact_clusters_:
            if cluster['num_observations'] < ARTIFACT_CONFIRMATION_OBSERVATIONS:
                continue
            radius = ARTIFACT_PHYSICAL_RADIUS_M.get(
                cluster['label'], ARTIFACT_DEFAULT_PHYSICAL_RADIUS_M)
            dx = x - cluster['position'].x
            dy = y - cluster['position'].y
            if math.hypot(dx, dy) < radius + ROVER_FOOTPRINT_RADIUS_M:
                return False
        return True

    def find_frontier_cells(self):
        """
        Planning 1: return a boolean mask of 'frontier' cells - free cells that neighbour at
        least one unknown cell, i.e. the boundary between explored and unexplored space.
        """

        grid = self.map_grid_
        free = (grid >= 0) & (grid < FRONTIER_OCCUPIED_THRESHOLD)
        unknown = grid == -1

        return free & dilate8_mask(unknown)

    def cluster_frontier_cells(self, frontier_mask):
        """Planning 1: group frontier cells into connected (8-connectivity) clusters via BFS"""

        visited = np.zeros_like(frontier_mask, dtype=bool)
        height, width = frontier_mask.shape
        clusters = []

        for start_row, start_col in zip(*np.nonzero(frontier_mask)):
            if visited[start_row, start_col]:
                continue

            cluster = []
            queue = deque([(start_row, start_col)])
            visited[start_row, start_col] = True
            while queue:
                r, c = queue.popleft()
                cluster.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width \
                                and frontier_mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

            clusters.append(cluster)

        return clusters

    def compute_path_distance_grid(self, start_row, start_col):
        """
        Planning 1: BFS over free cells from (start_row, start_col), returning a same-shaped
        array of path distance in cells (-1 where unreachable through known free space).

        Used so frontier scoring reflects actual path distance through the cave's corridors,
        rather than straight-line distance - which can be badly misleading in a branching
        maze (a frontier might look close as the crow flies but actually require a long
        detour through the only real corridor, or vice versa).

        Vectorised as a wavefront expansion (dilate8_mask(), one ring of cells per iteration)
        rather than a per-cell python-loop BFS: this is called on every frontier AND every
        inspection-target goal pick (see planner_inspect_artifact()), and profiling showed the
        per-cell version taking 170ms-1000ms+ on realistic map sizes - long enough to noticeably
        starve this node's own tf2/camera callbacks (and compete for CPU with the rest of the
        Nav2 stack) each time it ran, which can itself contribute to the kind of system-wide
        timing hiccups (stale transforms, missed control loops) that looked like unrelated
        environment flakiness. The vectorised version is 3-10x faster and scales far better as
        the map grows over a long run (growing the full map was also made possible by the
        global_costmap rolling_window fix - see nav2_params.yaml).
        """

        grid = self.map_grid_
        height, width = grid.shape
        free = (grid >= 0) & (grid < FRONTIER_OCCUPIED_THRESHOLD)

        dist = np.full((height, width), -1, dtype=np.int32)
        if not free[start_row, start_col]:
            return dist

        dist[start_row, start_col] = 0
        frontier = np.zeros((height, width), dtype=bool)
        frontier[start_row, start_col] = True
        d = 0
        while frontier.any():
            d += 1
            expanded = dilate8_mask(frontier) & free & (dist == -1)
            if not expanded.any():
                break
            dist[expanded] = d
            frontier = expanded

        return dist

    def choose_frontier_goal(self, robot_pose):
        """
        Planning 1: pick the best frontier cluster to explore next.

        Score = cluster size / (1 + path_distance / FRONTIER_DISTANCE_SCALE_M): prefers larger
        frontiers (more new area to reveal), with path distance (through the known free-space
        corridors - see compute_path_distance_grid()) as a secondary tiebreaker rather than a
        dominant factor - so a big far frontier can still beat a tiny close one, which matters
        because small clusters near already-explored corridors are usually just sensor-shadow
        slivers, not real new territory. Clusters smaller than FRONTIER_MIN_CLUSTER_SIZE are
        dropped as likely noise, and clusters near any previously-sent goal are skipped
        (revisitation avoidance).

        Returns (chosen, candidates): 'chosen' is the best candidate dict (or None if none are
        valid), 'candidates' is every candidate considered (for visualisation).
        """

        if self.map_grid_ is None:
            return None, []

        frontier_mask = self.find_frontier_cells()
        clusters = self.cluster_frontier_cells(frontier_mask)

        robot_row, robot_col = self.world_to_grid(robot_pose.x, robot_pose.y)
        path_distance_grid = self.compute_path_distance_grid(robot_row, robot_col)

        # Drop goals older than the revisit window - see FRONTIER_REVISIT_WINDOW_S's comment
        now = self.get_clock().now()
        self.recent_frontier_goals_ = [
            (gx, gy, gt) for gx, gy, gt in self.recent_frontier_goals_
            if (now - gt).nanoseconds / 1e9 < FRONTIER_REVISIT_WINDOW_S
        ]

        candidates = []
        for cluster in clusters:
            if len(cluster) < FRONTIER_MIN_CLUSTER_SIZE:
                continue

            mean_row = sum(r for r, c in cluster) / len(cluster)
            mean_col = sum(c for r, c in cluster) / len(cluster)

            # Snap to the actual cluster cell nearest the centroid, rather than sending Nav2
            # the raw arithmetic mean position directly - for a non-convex cluster (wraps
            # around a corner, L-shaped), the mean can land outside the free/frontier cells
            # altogether, inside a wall or unknown space. Snapping guarantees the goal is one
            # of the real frontier cells found above, which is by definition free space.
            snap_row, snap_col = min(
                cluster, key=lambda rc: (rc[0] - mean_row) ** 2 + (rc[1] - mean_col) ** 2)
            world_x, world_y = self.grid_to_world(snap_row, snap_col)

            if any(math.hypot(world_x - gx, world_y - gy) < FRONTIER_REVISIT_RADIUS_M
                   for gx, gy, _ in self.recent_frontier_goals_):
                continue

            path_cells = path_distance_grid[snap_row, snap_col]
            # Not reachable through known free space at all (e.g. an isolated noise pocket
            # behind a wall) - still included in 'candidates' for visualisation, but excluded
            # from 'reachable' below so it can never actually be chosen. This is a genuine
            # reachability pre-check using the BFS data already computed above, rather than
            # just letting an unreachable frontier's score trend toward (but never quite hit)
            # zero - which could still let it be picked if it ends up the only candidate once
            # nearby real frontiers are excluded by the revisit window, wasting a full
            # GOAL_TIMEOUT_S finding out Nav2 can't reach it either.
            reachable = path_cells >= 0
            distance = path_cells * self.map_resolution_ if reachable else float('inf')
            score = len(cluster) / (1.0 + distance / FRONTIER_DISTANCE_SCALE_M) if reachable else 0.0

            candidates.append({
                'position': (world_x, world_y),
                'size': len(cluster),
                'distance': distance,
                'score': score,
                'reachable': reachable,
            })

        if not candidates:
            return None, []

        reachable_candidates = [c for c in candidates if c['reachable']]
        if not reachable_candidates:
            return None, candidates

        chosen = max(reachable_candidates, key=lambda c: c['score'])
        return chosen, candidates

    def publish_frontier_markers(self, chosen, candidates):
        """Planning 1: visualise every candidate frontier (yellow) and the chosen one (red)"""

        marker_array = MarkerArray()

        candidate_points = Marker()
        candidate_points.header.frame_id = 'map'
        candidate_points.ns = 'frontier_candidates'
        candidate_points.id = 0
        candidate_points.type = Marker.POINTS
        candidate_points.action = Marker.ADD
        candidate_points.pose.orientation.w = 1.0
        candidate_points.scale.x = 0.2
        candidate_points.scale.y = 0.2
        candidate_points.color.a = 1.0
        candidate_points.color.r = 1.0
        candidate_points.color.g = 1.0
        candidate_points.color.b = 0.0
        for candidate in candidates:
            x, y = candidate['position']
            candidate_points.points.append(Point(x=x, y=y, z=0.2))
        marker_array.markers.append(candidate_points)

        chosen_marker = Marker()
        chosen_marker.header.frame_id = 'map'
        chosen_marker.ns = 'frontier_chosen'
        chosen_marker.id = 1
        chosen_marker.type = Marker.SPHERE
        chosen_marker.action = Marker.ADD if chosen else Marker.DELETE
        if chosen:
            x, y = chosen['position']
            chosen_marker.pose.position = Point(x=x, y=y, z=0.3)
            chosen_marker.pose.orientation.w = 1.0
            chosen_marker.scale.x = chosen_marker.scale.y = chosen_marker.scale.z = 0.6
            chosen_marker.color.a = 1.0
            chosen_marker.color.r = 1.0
            chosen_marker.color.g = 0.0
            chosen_marker.color.b = 0.0
        marker_array.markers.append(chosen_marker)

        self.frontier_markers_pub_.publish(marker_array)

    def planner_explore_frontier(self):
        """Planning 1: autonomously explore by driving to the best unexplored frontier"""

        robot_pose = self.get_pose_2d()
        if robot_pose is None:
            return

        chosen, candidates = self.choose_frontier_goal(robot_pose)
        self.publish_frontier_markers(chosen, candidates)

        if chosen is None:
            # planner_random_walk() samples directly from the current map bounds, so (unlike
            # planner_random_goal()'s fixed coordinate list) it always finds a valid point even
            # very early on (e.g. before the first map message arrives) or on an unfamiliar map
            if self.map_grid_ is None:
                self.get_logger().info('    (no map yet - taking a random step until SLAM has data)')
            else:
                self.no_frontier_streak_ += 1
                if self.no_frontier_streak_ >= EXPLORATION_COMPLETE_STREAK:
                    if not self.exploration_complete_:
                        self.get_logger().info(
                            f'    EXPLORATION COMPLETE: no frontiers found for '
                            f'{self.no_frontier_streak_} consecutive checks - the cave appears '
                            'to be fully mapped')
                    self.exploration_complete_ = True
                else:
                    self.get_logger().info('    (no frontiers right now - taking a random step)')
            self.planner_random_walk()
            return

        self.no_frontier_streak_ = 0
        goal_x, goal_y = chosen['position']
        theta = math.atan2(goal_y - robot_pose.y, goal_x - robot_pose.x)
        goal_pose2d = Pose2D(x=goal_x, y=goal_y, theta=theta)

        # Expires after FRONTIER_REVISIT_WINDOW_S - see that constant's comment for why
        self.recent_frontier_goals_.append((goal_x, goal_y, self.get_clock().now()))

        self.planner_go_to_pose2d(goal_pose2d)

    def find_inspection_target(self):
        """
        Planning 2/3: return the first known artefact cluster of a chosen type (see
        INSPECTION_ARTIFACT_LABELS) that hasn't already been inspected or abandoned, or None.
        """

        for cluster in self.artifact_clusters_:
            if cluster['label'] not in INSPECTION_ARTIFACT_LABELS:
                continue
            if cluster['num_observations'] < ARTIFACT_CONFIRMATION_OBSERVATIONS:
                continue  # not yet confirmed - could still be a one-off false positive
            if cluster['id'] in self.visited_artifact_ids_ or cluster['id'] in self.abandoned_artifact_ids_:
                continue
            if self.is_duplicate_of_handled_artifact(cluster):
                # Treat it as the same physical artefact as one we've already dealt with (see
                # ARTIFACT_DUPLICATE_RADIUS_M) - mark it visited too, so this id (and any
                # future ones that spawn near it) won't keep coming back as a "new" target
                self.visited_artifact_ids_.add(cluster['id'])
                continue
            return cluster
        return None

    def is_duplicate_of_handled_artifact(self, cluster):
        """Planning 3: is 'cluster' suspiciously close to a same-label cluster already handled?"""

        handled_ids = self.visited_artifact_ids_ | self.abandoned_artifact_ids_
        for other in self.artifact_clusters_:
            if other['id'] == cluster['id'] or other['id'] not in handled_ids \
                    or other['label'] != cluster['label']:
                continue
            dx = cluster['position'].x - other['position'].x
            dy = cluster['position'].y - other['position'].y
            if math.hypot(dx, dy) < ARTIFACT_DUPLICATE_RADIUS_M:
                return True
        return False

    def publish_visited_artifact_markers(self):
        """
        Planning 3: overlay a small flag above each inspectable-type artefact (see
        INSPECTION_ARTIFACT_LABELS) showing its inspection status: visited (green),
        abandoned after repeated failed attempts (orange), or still pending (red).
        """

        marker_array = MarkerArray()
        for cluster in self.artifact_clusters_:
            if cluster['label'] not in INSPECTION_ARTIFACT_LABELS:
                continue
            if cluster['num_observations'] < ARTIFACT_CONFIRMATION_OBSERVATIONS:
                continue  # not yet confirmed - don't show an unconfirmed detection as "pending"

            if cluster['id'] in self.visited_artifact_ids_:
                color = (0.0, 1.0, 0.0)
            elif cluster['id'] in self.abandoned_artifact_ids_:
                color = (1.0, 0.6, 0.0)
            else:
                color = (1.0, 0.0, 0.0)

            flag = Marker()
            flag.header.frame_id = 'map'
            flag.ns = 'artifact_inspection_status'
            flag.id = cluster['id']
            flag.type = Marker.CYLINDER
            flag.action = Marker.ADD
            flag.pose.position.x = cluster['position'].x
            flag.pose.position.y = cluster['position'].y
            flag.pose.position.z = cluster['position'].z + 0.9
            flag.pose.orientation.w = 1.0
            flag.scale.x = flag.scale.y = 0.25
            flag.scale.z = 0.08
            flag.color.a = 1.0
            flag.color.r, flag.color.g, flag.color.b = color
            marker_array.markers.append(flag)

        self.visited_markers_pub_.publish(marker_array)

    def planner_inspect_artifact(self):
        """
        Planning 2/3: navigate to a close-range standoff viewpoint of self.inspection_target_.

        Tries a handful of candidate approach angles around the artefact, starting with
        approaching from the robot's current side (shortest path) and otherwise spaced around
        it, using the first one that validates as (a) free space in the current map (see
        is_point_navigable()), (b) actually reachable from the robot's current position through
        currently-known free space (reusing the same BFS path-distance grid frontier selection
        uses - see compute_path_distance_grid()), and (c) outside the physical body of every
        known artefact (see is_clear_of_artifacts()). A standoff point computed from the
        artefact's position alone can otherwise land inside a wall, in unexplored space, or on
        the far side of a wall with no known connecting path - which Nav2's global planner then
        can't find a path to at all, rather than just navigating there poorly.

        Check (c) exists because (a) alone isn't sufficient: the widest artefacts are wider than
        the nominal standoff distance, and they show up so sparsely in the lidar's single
        horizontal scan plane that the map reports the inside of them as free space. See
        ARTIFACT_PHYSICAL_RADIUS_M for the full explanation.
        """

        robot_pose = self.get_pose_2d()
        if robot_pose is None or self.inspection_target_ is None:
            return

        target = self.inspection_target_['position']
        dx = robot_pose.x - target.x
        dy = robot_pose.y - target.y
        base_angle = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-3 else 0.0

        path_distance_grid = None
        if self.map_grid_ is not None:
            robot_row, robot_col = self.world_to_grid(robot_pose.x, robot_pose.y)
            path_distance_grid = self.compute_path_distance_grid(robot_row, robot_col)

        def is_reachable(x, y):
            if path_distance_grid is None:
                return True
            row, col = self.world_to_grid(x, y)
            height, width = path_distance_grid.shape
            return 0 <= row < height and 0 <= col < width and path_distance_grid[row, col] >= 0

        candidate_offsets = [0.0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2,
                             3 * math.pi / 4, -3 * math.pi / 4, math.pi]
        standoff = self.artifact_standoff_distance(self.inspection_target_['label'])

        # Try the nominal standoff first, then step outwards. The nominal distance already clears
        # the artefact's own body, but an artefact sitting against a wall or beside a second
        # artefact can have every angle at that exact radius blocked while a slightly wider orbit
        # is completely clear. Stepping out is much better than the unvalidated fallback below.
        goal_x = goal_y = None
        for extra_distance in (0.0, 1.0, 2.0):
            for offset in candidate_offsets:
                angle = base_angle + offset
                distance = standoff + extra_distance
                candidate_x = target.x + math.cos(angle) * distance
                candidate_y = target.y + math.sin(angle) * distance
                if self.is_point_navigable(candidate_x, candidate_y) \
                        and is_reachable(candidate_x, candidate_y) \
                        and self.is_clear_of_artifacts(candidate_x, candidate_y):
                    goal_x, goal_y = candidate_x, candidate_y
                    break
            if goal_x is not None:
                break

        if goal_x is None:
            # Nothing validated (e.g. map not built yet, or the artefact is tightly enclosed) -
            # fall back to the direct approach anyway; the goal watchdog (GOAL_TIMEOUT_S) will
            # recover if Nav2 genuinely can't reach it. Logged so we have evidence of how often
            # this unvalidated fallback actually fires, rather than just suspecting it might.
            self.get_logger().warn(
                f"    (no validated standoff point for {self.inspection_target_['label']} "
                f"#{self.inspection_target_['id']} - falling back to unvalidated direct approach)")
            # Still use the label's own standoff distance, not the 2.0m floor: even unvalidated,
            # a goal outside the artefact's body is far better than one inside it.
            goal_x = target.x + math.cos(base_angle) * standoff
            goal_y = target.y + math.sin(base_angle) * standoff

        theta = math.atan2(target.y - goal_y, target.x - goal_x)  # face the artefact
        self.planner_go_to_pose2d(Pose2D(x=goal_x, y=goal_y, theta=theta))

    def log_episode_start(self, mode_label):
        """
        Logging only: print a divider and a MODE header to mark the start of a new decision
        cycle - every line until the next one belongs to this cycle (heading out, discoveries
        en route, the outcome, and the inspection/mission result it led to).
        """

        self.get_logger().info('-' * 54)
        self.get_logger().info(f'MODE: {mode_label}')

    def log_goal_outcome(self, outcome_text):
        """
        Logging only: report how the current goal concluded - elapsed time, straight-line
        distance, and average speed for this leg - and add that distance to the mission-wide
        running total (see total_distance_m_, used by MISSION COMPLETE's overall average).
        Called from every place a goal can conclude: goal_response_callback (rejected),
        goal_reached_callback (reached/failed), and abandon_current_goal() (timed out/no
        progress/preempted).
        """

        elapsed_s = (self.get_clock().now() - self.goal_sent_time_).nanoseconds / 1e9 \
            if self.goal_sent_time_ is not None else 0.0
        self.last_goal_duration_s_ = elapsed_s

        if self.goal_initial_distance_m_ is None or elapsed_s <= 0:
            self.get_logger().info(f'    <- {outcome_text}')
            return

        # A goal that concluded instantly never moved the rover, so neither the distance nor a
        # speed derived from it means anything - and crediting it to the mission total actively
        # corrupts the headline figure. green_crystals #62's 42.6m goal was rejected twice in 0s
        # and contributed 85.2m the rover never drove, inflating MISSION COMPLETE's average
        # speed. Report the outcome, say how far away the goal had been, and claim nothing else.
        if elapsed_s < GOAL_INSTANT_FAILURE_S:
            self.get_logger().info(
                f'    <- {outcome_text} immediately ({elapsed_s:.2f}s, no movement; '
                f'the goal was {self.goal_initial_distance_m_:.1f}m away)')
            return

        avg_speed = self.goal_initial_distance_m_ / elapsed_s
        self.get_logger().info(
            f'    <- {outcome_text} in {elapsed_s:.0f}s '
            f'({self.goal_initial_distance_m_:.1f}m, avg {avg_speed:.2f} m/s)')
        self.total_distance_m_ += self.goal_initial_distance_m_

    def abandon_current_goal(self, outcome_text='abandoned'):
        """
        Give up on the in-flight Nav2 goal locally - bump goal_generation_ (so a late
        response/result callback for it is recognised as stale and ignored) and clear the
        tracking state - WITHOUT explicitly cancelling it with Nav2 first.

        Every call site of this (the DIVERT preemption, and the timeout/no-progress watchdog
        in main_loop) sends a brand new goal immediately afterwards, in the same tick. Nav2's
        bt_navigator already preempts a goal that's still executing the instant a new one is
        sent to the same action client - that's the standard, intended way Nav2's single-goal
        action servers switch goals, not a fallback. An earlier version of this method called
        current_goal_handle_.cancel_goal_async() here first, which raced against that: the
        very next goal could land in the gap between Nav2 accepting the cancel and the
        behaviour tree actually finishing its halt, getting instantly aborted by Nav2 itself
        (logged as a goal "failed" in 0s, status=6) while the BT reset - a visible stutter/
        stop matching exactly the "randomly pausing" symptom reported live, confirmed by
        bt_navigator's own "Aborting handle" / "Failed to get result for follow_path in node
        halt!" messages appearing right at those moments. Letting Nav2's own preemption
        handle the swap (just by sending the next goal) avoids the race entirely.
        """

        self.log_goal_outcome(outcome_text)
        self.current_goal_handle_ = None
        self.goal_generation_ += 1
        self.last_goal_succeeded_ = False
        self.ready_for_next_goal_ = True
        self.goal_sent_time_ = None

    def planner_go_to_pose2d(self, pose2d, timeout_s=GOAL_TIMEOUT_S):
        """
        Go to a provided 2d pose.

        'timeout_s' overrides the flat watchdog ceiling (see main_loop's watchdog) for this
        goal specifically - see planner_return_home() for why RETURN_HOME needs a larger,
        distance-scaled one instead of the default.
        """

        # Any pause-cancelled goal has been dealt with by whatever was tracking it by now (see
        # pause_mission_callback()); clear the flag so it can't leak into the next behaviour.
        self.goal_cancelled_by_pause_ = False

        # Send a goal to navigate_to_pose with self.nav2_action_client_
        action_goal = NavigateToPose.Goal()
        action_goal.pose.header.stamp = self.get_clock().now().to_msg()
        action_goal.pose.header.frame_id = 'map'
        action_goal.pose.pose = pose2d_to_pose(pose2d)

        # Publish visualisation
        self.goal_pose_vis_.publish(action_goal.pose)

        # Send goal to action server. Tag this goal with a generation number, and have the
        # callbacks below check it against self.goal_generation_ before acting - if the
        # watchdog in main_loop() has since given up on this goal and moved on, these would
        # otherwise be a late, stale response overwriting the state of the *next* goal.
        self.goal_generation_ += 1
        generation = self.goal_generation_
        self.goal_sent_time_ = self.get_clock().now()
        self.goal_timeout_s_ = timeout_s

        # Progress tracking for this goal starts fresh - see GOAL_PROGRESS_STALL_S and
        # goal_target_x_/goal_target_y_'s comment for why this is based on the robot's own
        # pose rather than Nav2's distance_remaining feedback.
        self.goal_target_x_ = pose2d.x
        self.goal_target_y_ = pose2d.y
        self.goal_progress_distance_ = None
        self.goal_progress_time_ = self.goal_sent_time_

        # Logging only (see log_goal_outcome()) - straight-line distance to this goal right
        # now, used to report this leg's average speed once it concludes.
        robot_pose = self.get_pose_2d()
        self.goal_initial_distance_m_ = math.hypot(pose2d.x - robot_pose.x, pose2d.y - robot_pose.y) \
            if robot_pose is not None else None

        self.get_logger().info(f'    -> heading to ({pose2d.x:.2f}, {pose2d.y:.2f})')
        self.send_goal_future_ = self.nav2_action_client_.send_goal_async(
            action_goal,
            feedback_callback=functools.partial(self.feedback_callback, generation=generation))
        self.send_goal_future_.add_done_callback(
            functools.partial(self.goal_response_callback, generation=generation))

    def goal_response_callback(self, future, generation):
        """The requested goal pose has been sent to the action server"""

        if generation != self.goal_generation_:
            return  # stale: we've since given up on this goal (see GOAL_TIMEOUT_S) and moved on

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.log_goal_outcome('rejected')
            self.last_goal_succeeded_ = False
            self.ready_for_next_goal_ = True
            return

        # Keep a handle to the in-flight goal so main_loop's watchdog can actually cancel it
        # with Nav2 (see current_goal_handle_'s comment) rather than just locally pretending
        # it's gone while Nav2 keeps driving it in the background.
        self.current_goal_handle_ = goal_handle

        # Goal accepted: get result when it's completed
        self.get_result_future_ = goal_handle.get_result_async()
        self.get_result_future_.add_done_callback(
            functools.partial(self.goal_reached_callback, generation=generation))

    def feedback_callback(self, feedback_msg, generation):
        """
        Log Nav2's distance_remaining feedback, if requested. Progress-stall detection
        (see GOAL_PROGRESS_STALL_S) is tracked separately in main_loop's watchdog, from the
        robot's own pose rather than this feedback - see goal_target_x_'s comment for why.
        """

        if generation != self.goal_generation_:
            return  # stale: we've since given up on this goal (see GOAL_TIMEOUT_S) and moved on

        if self.get_parameter('print_feedback').value:
            self.get_logger().info(f'{feedback_msg.feedback.distance_remaining:.2f} m remaining')

    def goal_reached_callback(self, future, generation):
        """The requested goal has finished (successfully or not)"""

        if generation != self.goal_generation_:
            return  # stale: we've since given up on this goal (see GOAL_TIMEOUT_S) and moved on

        status = future.result().status
        self.last_goal_succeeded_ = (status == GoalStatus.STATUS_SUCCEEDED)
        self.log_goal_outcome('reached' if self.last_goal_succeeded_ else f'failed (status={status})')
        self.current_goal_handle_ = None
        self.ready_for_next_goal_ = True


    def planner_move_forwards(self, distance):
        """Simply move forward by the specified distance"""

        pose_2d = self.get_pose_2d()

        pose_2d.x += distance * math.cos(pose_2d.theta)
        pose_2d.y += distance * math.sin(pose_2d.theta)

        self.planner_go_to_pose2d(pose_2d)

    def planner_go_to_first_artifact(self):
        """Go to a pre-specified artifact location"""

        goal_pose2d = Pose2D(
            x = 18.1,
            y = 6.6,
            theta = math.pi/2
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_return_home(self):
        """
        Return to the origin.

        Uses a distance-scaled timeout (see RETURN_HOME_MIN_SPEED_MPS) instead of the flat
        GOAL_TIMEOUT_S: RETURN_HOME only happens once, right at the very end, after there's no
        more exploration time left to protect, so there's no benefit to a short ceiling - only
        the cost of cutting off a genuinely slow-but-progressing trip home. Live testing hit
        exactly this: a ~31m trip home abandoned by the flat 45s ceiling with the robot still
        16m short, after RETURN_HOME had already been exempted from the faster progress-stall
        check for the same underlying reason (see that check's comment in main_loop).
        """

        goal_pose2d = Pose2D(
            x = 0.0,
            y = 0.0,
            theta = math.pi
        )

        robot_pose = self.get_pose_2d()
        if robot_pose is not None:
            distance_home_m = math.hypot(robot_pose.x, robot_pose.y)
            timeout_s = max(GOAL_TIMEOUT_S, distance_home_m / RETURN_HOME_MIN_SPEED_MPS)
        else:
            timeout_s = GOAL_TIMEOUT_S

        self.planner_go_to_pose2d(goal_pose2d, timeout_s=timeout_s)

    def planner_random_walk(self):
        """
        Go to the nearest-to-the-robot navigable location out of several random samples
        within the current map bounds.

        Sampling uniformly within xlim_/ylim_ can easily land inside a wall or unexplored
        space for an irregular cave shape (most of the bounding box isn't actual floor) -
        that previously sent Nav2 on goals it could never reach at all, costing a full 60s
        goal-timeout each time. Each candidate is checked with is_point_navigable() before
        being considered. Of the candidates that do validate, the nearest one is chosen
        (rather than just the first found) - otherwise this fallback has no locality
        preference at all, and can send the robot clear across the whole cave instead of to
        the closest unclaimed spot, forcing a long traversal back through already-explored
        corridors (observed as excessive backtracking on long runs).
        """

        robot_pose = self.get_pose_2d()

        candidates = []
        for _ in range(RANDOM_WALK_MAX_ATTEMPTS):
            x = random.uniform(self.xlim_[0], self.xlim_[1])
            y = random.uniform(self.ylim_[0], self.ylim_[1])
            if self.is_point_navigable(x, y):
                candidates.append((x, y))

        if not candidates:
            # No navigable point found (e.g. map not built yet) - hold position rather than
            # risk another long Nav2 failure/recovery cycle on an unreachable random point
            self.get_logger().warn('    (no navigable random point found - holding position)')
            if robot_pose is not None:
                self.planner_go_to_pose2d(robot_pose)
            return

        if robot_pose is not None:
            x, y = min(candidates,
                       key=lambda c: math.hypot(c[0] - robot_pose.x, c[1] - robot_pose.y))
        else:
            x, y = random.choice(candidates)

        goal_pose2d = Pose2D(x=x, y=y, theta=random.uniform(0, 2 * math.pi))
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_random_goal(self):
        """Go to a random location out of a predefined set"""

        # Hand picked set of goal locations
        random_goals = [[15.2, 2.2],
                        [30.7, 2.2],
                        [43.0, 11.3],
                        [36.6, 21.9],
                        [33.0, 30.4],
                        [40.4, 44.3],
                        [51.5, 37.8],
                        [16.0, 24.1],
                        [3.4, 33.5],
                        [7.9, 13.8],
                        [14.2, 37.7]]

        # Select a random location, trying each candidate at most once (rather than looping
        # forever) in case none of them currently fall within the map bounds - e.g. early on,
        # before enough of the map has been seen for any of this hardcoded list to be in range
        goal_valid = False
        for goal_x, goal_y in random.sample(random_goals, len(random_goals)):
            if self.xlim_[0] < goal_x < self.xlim_[1] and self.ylim_[0] < goal_y < self.ylim_[1]:
                goal_valid = True
                break
            self.get_logger().warn(f'Goal [{goal_x}, {goal_y}] out of bounds')

        if not goal_valid:
            self.get_logger().warn('No random goal currently within map bounds - skipping this cycle')
            self.ready_for_next_goal_ = True
            return

        goal_pose2d = Pose2D(
            x = goal_x,
            y = goal_y,
            theta = random.uniform(0, 2*math.pi)
        )
        self.planner_go_to_pose2d(goal_pose2d)

    #######################################################
    # Advanced 2: cave geometry analysis.
    #
    # The analysis itself lives in cave_geometry.py (pure numpy/OpenCV, no ROS) - see that
    # module's docstring for the method. Everything here is the ROS side of it: when it runs,
    # what it runs on, and how the results get into RViz and the logs.

    def find_nearest_navigable_point(self, x, y, clearance_cells=3,
                                     search_radius_m=TOUR_SNAP_SEARCH_RADIUS_M):
        """
        Return the navigable point nearest to (x, y), or None if there isn't one within
        search_radius_m.

        Needed because a "meaningful location" isn't necessarily a drivable one. An artefact's
        estimated position is the artefact itself - i.e. an obstacle - and the estimate can be a
        little off besides, so it often lands inside a wall or in unexplored space. Searching
        outward in rings and taking the first navigable hit keeps the stop as close as possible
        to the thing it's meant to be a stop for, instead of either sending Nav2 somewhere it
        can provably never reach or dropping the location from the tour entirely.
        """

        if self.map_grid_ is None:
            return None
        if self.is_point_navigable(x, y, clearance_cells):
            return x, y

        samples_per_ring = 16
        for ring in range(1, int(search_radius_m / self.map_resolution_) + 1):
            radius_m = ring * self.map_resolution_
            for sample in range(samples_per_ring):
                angle = 2.0 * math.pi * sample / samples_per_ring
                candidate_x = x + radius_m * math.cos(angle)
                candidate_y = y + radius_m * math.sin(angle)
                if self.is_point_navigable(candidate_x, candidate_y, clearance_cells):
                    return candidate_x, candidate_y

        return None

    def compute_reachable_mask(self, robot_pose):
        """
        Advanced 2: boolean mask of cells the robot can actually drive to from where it is now,
        through currently-known free space. Reuses Planning 1's wavefront BFS.

        See cave_geometry.analyse_cave() for why the geometry analysis needs this rather than
        just operating on every free cell on the map.
        """

        start = self.find_nearest_navigable_point(
            robot_pose.x, robot_pose.y, clearance_cells=0, search_radius_m=1.0)
        if start is None:
            # The robot's own cell reads as occupied and nothing free is within a metre - almost
            # certainly a transient SLAM artefact mid-update. Skip this cycle; the next tick's
            # map will have moved on.
            return None

        row, col = self.world_to_grid(*start)
        height, width = self.map_grid_.shape
        if not (0 <= row < height and 0 <= col < width):
            return None

        return self.compute_path_distance_grid(row, col) >= 0

    def update_cave_geometry(self):
        """
        Advanced 2: recompute the cave geometry analysis from the current map and publish it.

        Runs on a timer throughout the mission (see GEOMETRY_UPDATE_PERIOD_S) rather than once
        at the end, because the task asks for the distance transform to be computed and updated
        online - so the widest-area and bottleneck answers visibly refine in RViz as SLAM
        reveals more cave, instead of appearing all at once after the fact.
        """

        if self.map_grid_ is None or self.map_resolution_ is None:
            return

        # Checked rather than just calling get_pose_2d(), which logs an error on every failed
        # lookup - this timer would otherwise spam the log until SLAM comes up.
        if not self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time()):
            return
        robot_pose = self.get_pose_2d()
        if robot_pose is None:
            return

        # Snapshot the map parameters alongside the grid itself, so the published overlay is
        # guaranteed to describe the same grid the analysis ran on even if a new map arrives
        # before this method finishes.
        grid = self.map_grid_
        resolution = self.map_resolution_
        origin = list(self.map_origin_)

        reachable = self.compute_reachable_mask(robot_pose)
        if reachable is None:
            return

        # max_features is higher than Advanced 2 needs on its own (which only has to report the
        # widest area and a bottleneck): the science tour draws its stops from these same lists
        # (see collect_science_locations()), so it needs enough candidates to fill a tour from.
        self.geometry_ = cave_geometry.analyse_cave(grid, resolution, reachable, max_features=8)

        self.publish_clearance_map(self.geometry_['clearance_m'], resolution, origin)
        self.publish_geometry_markers(resolution, origin)
        self.log_geometry_findings()

    def log_geometry_findings(self):
        """
        Advanced 2: report the two headline findings the task asks for - the widest accessible
        area and the narrowest passage.

        Only logs when an answer actually changes by a noticeable amount, rather than every
        GEOMETRY_UPDATE_PERIOD_S: during a long run the values settle early and repeating them
        every few seconds would bury the exploration log, but the moment the robot breaks into a
        bigger chamber or squeezes through a tighter gap is exactly what we want recorded.
        """

        min_change_m = 0.2
        chambers = self.geometry_['chambers']
        bottlenecks = self.geometry_['bottlenecks']

        if chambers:
            widest_m = chambers[0]['clearance_m']
            if self.last_logged_widest_m_ is None \
                    or abs(widest_m - self.last_logged_widest_m_) > min_change_m:
                x, y = self.grid_to_world(chambers[0]['row'], chambers[0]['col'])
                self.get_logger().info(
                    f'    GEOMETRY: widest accessible area at ({x:.1f}, {y:.1f}) - '
                    f'{2.0 * widest_m:.1f}m across ({len(chambers)} chambers found)')
                self.last_logged_widest_m_ = widest_m

        if bottlenecks:
            narrowest_m = bottlenecks[0]['clearance_m']
            if self.last_logged_narrowest_m_ is None \
                    or abs(narrowest_m - self.last_logged_narrowest_m_) > min_change_m:
                x, y = self.grid_to_world(bottlenecks[0]['row'], bottlenecks[0]['col'])
                self.get_logger().info(
                    f'    GEOMETRY: narrowest passage at ({x:.1f}, {y:.1f}) - '
                    f"{bottlenecks[0]['width_m']:.1f}m wide ({len(bottlenecks)} bottlenecks found)")
                self.last_logged_narrowest_m_ = narrowest_m

    def publish_clearance_map(self, clearance_m, resolution, origin):
        """
        Advanced 2: republish the distance transform as an OccupancyGrid, so the task's "display
        the distance transform" can be done with RViz's standard Map display rather than by
        hand-rolling a per-cell marker grid (which would be tens of thousands of markers).

        OccupancyGrid only carries 0-100 per cell, so clearance is scaled against
        CLEARANCE_DISPLAY_MAX_M; walls and unexplored cells are published as -1 (unknown) so
        they stay visually distinct from genuinely zero-clearance floor. Display it with RViz's
        'costmap' colour scheme - the default 'map' scheme is a two-tone black/white ramp, which
        can't show a gradient usefully.
        """

        height, width = clearance_m.shape

        message = OccupancyGrid()
        message.header.frame_id = 'map'
        message.header.stamp = self.get_clock().now().to_msg()
        message.info.resolution = resolution
        message.info.width = width
        message.info.height = height
        message.info.origin.position.x = origin[0]
        message.info.origin.position.y = origin[1]
        message.info.origin.orientation.w = 1.0

        scaled = np.clip(clearance_m / CLEARANCE_DISPLAY_MAX_M * 100.0, 0.0, 100.0).astype(np.int8)
        scaled[clearance_m <= 0.0] = -1  # not free space: wall or unexplored
        message.data = scaled.ravel().tolist()

        self.clearance_map_pub_.publish(message)

    def publish_geometry_markers(self, resolution, origin):
        """
        Advanced 2: draw the detected chambers and bottlenecks in RViz, overlaid on the map.

        Each feature is drawn as a flat disc of exactly the radius the clearance map says fits
        there, so the marker isn't just a pin on a position - it's a direct visual claim about
        the geometry ("this circle fits in this chamber") that can be checked by eye against the
        map underneath it. The best of each kind (widest chamber, narrowest passage) is drawn
        opaque and labelled with its size; the rest are faint, so the headline answers the task
        asks for stand out from the supporting detections.
        """

        marker_array = MarkerArray()

        # Clear everything first - the feature set shrinks as well as grows between updates (a
        # chamber's local maximum can move a few cells, or a bottleneck can stop qualifying once
        # the map fills in around it), and without this those stale markers would persist in
        # RViz forever, accumulating into a misleading pile of "findings".
        clear_all = Marker()
        clear_all.header.frame_id = 'map'
        clear_all.action = Marker.DELETEALL
        marker_array.markers.append(clear_all)

        def world_of(feature):
            return (origin[0] + (feature['col'] + 0.5) * resolution,
                    origin[1] + (feature['row'] + 0.5) * resolution)

        for kind, features, color in (
                ('chamber', self.geometry_['chambers'], (0.1, 0.9, 0.2)),
                ('bottleneck', self.geometry_['bottlenecks'], (1.0, 0.2, 0.1))):
            for index, feature in enumerate(features):
                x, y = world_of(feature)
                is_headline = (index == 0)

                disc = Marker()
                disc.header.frame_id = 'map'
                disc.ns = f'cave_geometry_{kind}'
                disc.id = index
                disc.type = Marker.CYLINDER
                disc.action = Marker.ADD
                disc.pose.position = Point(x=x, y=y, z=0.05)
                disc.pose.orientation.w = 1.0
                # Diameter, not radius: clearance is the distance to the nearest wall, so the
                # circle that actually fits is twice that across.
                disc.scale.x = disc.scale.y = 2.0 * feature['clearance_m']
                disc.scale.z = 0.05
                disc.color.a = 0.55 if is_headline else 0.18
                disc.color.r, disc.color.g, disc.color.b = color
                marker_array.markers.append(disc)

                if not is_headline:
                    continue

                label = Marker()
                label.header.frame_id = 'map'
                label.ns = f'cave_geometry_{kind}_label'
                label.id = index
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position = Point(x=x, y=y, z=1.5)
                label.pose.orientation.w = 1.0
                label.scale.z = 0.8
                label.color.a = 1.0
                label.color.r, label.color.g, label.color.b = color
                if kind == 'chamber':
                    label.text = f"WIDEST AREA\n{2.0 * feature['clearance_m']:.1f}m across"
                else:
                    label.text = f"NARROWEST PASSAGE\n{feature['width_m']:.1f}m wide"
                marker_array.markers.append(label)

        self.geometry_markers_pub_.publish(marker_array)

    #######################################################
    # Advanced 5: science tour planning.
    #
    # The route optimisation itself lives in science_tour.py (pure python, no ROS) - see that
    # module's docstring for the method. Everything here is the robot side of it: choosing which
    # locations are worth visiting, measuring the real travel cost between them, driving the
    # resulting route, and drawing it.

    def find_artifact_viewpoint(self, position, label=None):
        """
        Advanced 5: a navigable point a standoff distance from an artefact, or None.

        Tries approach angles all the way around the artefact and takes the first that
        validates, the same idea as planner_inspect_artifact()'s standoff search - an artefact
        against a wall can only be viewed from some directions, and which ones isn't knowable
        from its position alone.

        The standoff comes from artifact_standoff_distance(label) so a tour stop lands outside
        the artefact's own body rather than inside the wider models, and every candidate is
        checked against is_clear_of_artifacts() afterwards: find_nearest_navigable_point()
        searches outwards from the requested point for free space, and since the inside of a
        crystal cluster reads as free in the map, snapping could otherwise pull the viewpoint
        back into the artefact it was supposed to stand off from.

        Orbits outwards if nothing at the nominal distance works, the same ladder as
        planner_inspect_artifact(). Needed for exactly the same reason: a small artefact right
        beside a big one has its entire nominal standoff ring inside the big one's body. The
        green alien and the crystal cluster were detected 0.67m apart, and the crystal reaches
        3.31m once the rover's own width is counted, which swallows the whole 2.31m ring around
        the alien - without the ladder that artefact is simply dropped from the tour, losing a
        real science stop to a viewpoint that does exist a little further out.
        """

        standoff = self.artifact_standoff_distance(label)
        for extra_distance in (0.0, 1.0, 2.0):
            distance = standoff + extra_distance
            for sample in range(8):
                angle = 2.0 * math.pi * sample / 8
                candidate = self.find_nearest_navigable_point(
                    position.x + distance * math.cos(angle),
                    position.y + distance * math.sin(angle),
                    search_radius_m=1.0)
                if candidate is not None \
                        and self.is_clear_of_artifacts(candidate[0], candidate[1]):
                    return candidate

        return None

    def collect_science_locations(self):
        """
        Advanced 5: choose the locations the tour will visit.

        The task is explicit that the locations must be meaningful rather than arbitrary, so
        every stop here is something the robot actually discovered and can justify:

          * base (0, 0) - the lander. Always the first stop, so the tour is a closed loop that
            starts and ends where a real mission would.
          * confirmed artefacts - the science targets themselves, at a viewpoint standoff rather
            than on top of them. Only ones confirmed by ARTIFACT_CONFIRMATION_OBSERVATIONS
            observations, so a one-off false positive can't send the rover across the cave.
          * chamber centres (Advanced 2) - the widest accessible areas, i.e. the candidate sites
            for anything that needs room.
          * bottlenecks (Advanced 2) - the narrow passages, the places a route plan has to care
            about and worth re-surveying up close.

        The three kinds are drawn against separate quotas instead of just taking the
        highest-priority ones until full: a tour of nothing but artefact viewpoints would be a
        worse survey than a mixed one, and it would also make the tour's content depend entirely
        on how the perception run happened to go. Quotas are deliberately generous relative to
        TOUR_MAX_LOCATIONS, with the shortfall topped up from whatever's left over, so the tour
        still fills up when one category comes back thin.

        Returns the list of stops (each a dict with x, y, label, kind), base first.
        """

        stops = []

        def try_add(x, y, label, kind):
            # No two stops at effectively the same place - a chamber centre and an artefact
            # viewpoint inside that chamber can easily coincide, and the tour would then be
            # "visiting" one spot twice while claiming two locations.
            if any(math.hypot(x - stop['x'], y - stop['y']) < TOUR_MIN_STOP_SEPARATION_M
                   for stop in stops):
                return False
            stops.append({'x': x, 'y': y, 'label': label, 'kind': kind, 'status': 'pending'})
            return True

        try_add(0.0, 0.0, 'base', 'base')

        artifact_candidates = []
        for cluster in self.artifact_clusters_:
            if cluster['num_observations'] < ARTIFACT_CONFIRMATION_OBSERVATIONS:
                continue
            viewpoint = self.find_artifact_viewpoint(cluster['position'], cluster['label'])
            if viewpoint is not None:
                artifact_candidates.append(
                    (viewpoint[0], viewpoint[1], f"{cluster['label']} #{cluster['id']}", 'artifact'))

        # Geometry features are filtered through is_clear_of_artifacts() as well. Advanced 2
        # describes the map honestly, but the map isn't the world: a cluster of crystal spikes
        # reads as a scatter of isolated obstacles in the lidar's single scan plane, so the gaps
        # between the spikes get reported as perfectly genuine narrow passages. One such "passage
        # 0.6m" was logged at (7.0, 5.7), a metre and a half from the crystal cluster at
        # (8.2, 5.2), and was duly added to the tour - a stop the 1x1m rover cannot physically
        # occupy. The analysis isn't wrong; it just mustn't be turned into a driving goal.
        chamber_candidates = []
        bottleneck_candidates = []
        if self.geometry_ is not None:
            for chamber in self.geometry_['chambers']:
                x, y = self.grid_to_world(chamber['row'], chamber['col'])
                navigable = self.find_nearest_navigable_point(x, y)
                if navigable is not None and self.is_clear_of_artifacts(navigable[0], navigable[1]):
                    chamber_candidates.append(
                        (navigable[0], navigable[1],
                         f"chamber {2.0 * chamber['clearance_m']:.1f}m", 'chamber'))
            for bottleneck in self.geometry_['bottlenecks']:
                # A passage narrower than the rover is not a place the rover can stand, whatever
                # the map says. The clearance check below is deliberately relaxed (2 cells, 0.6m)
                # so genuinely tight-but-passable routes still qualify, and that relaxation is
                # exactly what let a 0.6m "passage" through in testing - it sat 0.12m inside an
                # undetected mushroom cluster at (-3.3, 29.2) and was duly toured and missed.
                # Gating on the measured width catches that case without needing perception to
                # have confirmed the artefact first.
                if bottleneck['width_m'] < MIN_TOUR_PASSAGE_WIDTH_M:
                    continue
                x, y = self.grid_to_world(bottleneck['row'], bottleneck['col'])
                navigable = self.find_nearest_navigable_point(x, y, clearance_cells=2)
                if navigable is not None and self.is_clear_of_artifacts(navigable[0], navigable[1]):
                    bottleneck_candidates.append(
                        (navigable[0], navigable[1],
                         f"passage {bottleneck['width_m']:.1f}m", 'bottleneck'))

        quotas = ((artifact_candidates, 6), (chamber_candidates, 4), (bottleneck_candidates, 3))
        for candidates, quota in quotas:
            added = 0
            for x, y, label, kind in candidates:
                if added >= quota or len(stops) >= TOUR_MAX_LOCATIONS:
                    break
                if try_add(x, y, label, kind):
                    added += 1

        # Top up from everything left over if the quotas didn't fill the tour.
        for candidates, _ in quotas:
            for x, y, label, kind in candidates:
                if len(stops) >= TOUR_MAX_LOCATIONS:
                    break
                try_add(x, y, label, kind)

        return stops

    def build_tour_cost_matrix(self, stops):
        """
        Advanced 5: travel cost between every pair of stops, in metres of actual driving.

        Each entry comes from Planning 1's wavefront BFS over the occupancy grid - one full-map
        BFS per stop, reading off the distance to all the others. This is the part of Advanced 5
        the task is most specific about: the cost has to come from the navigation map, not from
        straight-line distance. In a branching cave those are wildly different numbers. Two
        chambers either side of a single wall are metres apart as the crow flies and can be fifty
        metres apart by road, and a tour optimised on straight-line distance would confidently
        order its stops as though the rover could drive through rock - producing a route that is
        provably optimal for a problem the robot doesn't have.

        Unreachable pairs are marked -1 rather than a large number, so the route optimisation can
        tell "far away" apart from "no route at all" and refuse to plan through the latter.
        """

        count = len(stops)
        cost_matrix = [[0.0] * count for _ in range(count)]
        height, width = self.map_grid_.shape

        for i, stop in enumerate(stops):
            row, col = self.world_to_grid(stop['x'], stop['y'])
            distance_grid = self.compute_path_distance_grid(row, col)

            for j, other in enumerate(stops):
                if i == j:
                    continue
                other_row, other_col = self.world_to_grid(other['x'], other['y'])
                if not (0 <= other_row < height and 0 <= other_col < width):
                    cost_matrix[i][j] = -1.0
                    continue
                cells = distance_grid[other_row, other_col]
                cost_matrix[i][j] = float(cells) * self.map_resolution_ if cells >= 0 else -1.0

        return cost_matrix

    def plan_science_tour(self):
        """
        Advanced 5: pick the locations, measure the real travel costs, and optimise the route.

        Returns True if a tour was planned. Stops with no route to or from base are dropped
        before optimising - keeping one would make the whole tour cost meaningless (and send the
        rover off to burn a goal timeout proving Nav2 can't get there either).
        """

        if self.map_grid_ is None:
            return False

        stops = self.collect_science_locations()
        if len(stops) < TOUR_MIN_LOCATIONS:
            self.get_logger().warn(
                f'    SCIENCE TOUR: only found {len(stops)} distinct reachable locations, '
                f'need {TOUR_MIN_LOCATIONS} - skipping the tour')
            return False

        plan_start = self.get_clock().now()
        cost_matrix = self.build_tour_cost_matrix(stops)

        # Drop anything with no route to or from base (index 0), then rebuild the matrix over
        # what's left - simpler and less error-prone than trying to delete rows and columns.
        connected = [0] + [i for i in range(1, len(stops))
                           if cost_matrix[0][i] >= 0.0 and cost_matrix[i][0] >= 0.0]
        if len(connected) < len(stops):
            dropped = len(stops) - len(connected)
            self.get_logger().warn(
                f'    SCIENCE TOUR: dropped {dropped} location(s) with no route to base')
            stops = [stops[i] for i in connected]
            if len(stops) < TOUR_MIN_LOCATIONS:
                self.get_logger().warn(
                    f'    SCIENCE TOUR: only {len(stops)} locations left after dropping '
                    f'unreachable ones - skipping the tour')
                return False
            cost_matrix = self.build_tour_cost_matrix(stops)

        result = science_tour.plan_tour(cost_matrix, start=0)
        plan_s = (self.get_clock().now() - plan_start).nanoseconds / 1e9

        self.tour_ = {'stops': stops, 'cost_matrix': cost_matrix, **result}
        self.tour_stop_index_ = 0

        self.log_tour_plan(plan_s)
        self.publish_tour_markers()
        return True

    def log_tour_plan(self, plan_s):
        """
        Advanced 5: print the planned tour and the baseline-vs-optimised comparison the task
        asks for, in a form that can go straight into the report.
        """

        tour = self.tour_
        self.get_logger().info('-' * 54)
        self.get_logger().info(
            f"SCIENCE TOUR PLANNED: {len(tour['stops'])} locations in {plan_s:.1f}s")
        self.get_logger().info(
            f"    baseline  (nearest neighbour): {tour['baseline_cost_m']:.1f}m")
        self.get_logger().info(
            f"    optimised (2-opt)            : {tour['optimised_cost_m']:.1f}m")
        self.get_logger().info(
            f"    saving: {tour['saved_m']:.1f}m ({tour['saved_percent']:.1f}%)")
        self.get_logger().info('    route:')
        for position, stop_index in enumerate(tour['optimised_order']):
            stop = tour['stops'][stop_index]
            self.get_logger().info(
                f"      {position + 1:2d}. {stop['label']:<24} "
                f"({stop['x']:6.1f}, {stop['y']:6.1f})  [{stop['kind']}]")

    def publish_tour_markers(self):
        """
        Advanced 5: draw the tour in RViz - the stops, and both routes through them.

        Both tours are drawn, not just the chosen one: the whole point of the task is the
        comparison, and seeing the baseline's long cross-cave backtrack next to the optimised
        loop makes the saving obvious at a glance in a way a pair of numbers in a log doesn't.
        The baseline is drawn faint and lower, the optimised route bright and above it.

        The lines are straight between consecutive stops, i.e. a diagram of the visiting ORDER,
        not of the path the robot drives (which follows the cave's corridors - Nav2 plans that).
        The costs that produced this order are the real path distances, though; see
        build_tour_cost_matrix().
        """

        if self.tour_ is None:
            return

        marker_array = MarkerArray()
        clear_all = Marker()
        clear_all.header.frame_id = 'map'
        clear_all.action = Marker.DELETEALL
        marker_array.markers.append(clear_all)

        stops = self.tour_['stops']
        status_colors = {'pending': (1.0, 1.0, 0.0), 'visited': (0.0, 1.0, 0.0),
                         'missed': (1.0, 0.4, 0.0)}

        for position, stop_index in enumerate(self.tour_['optimised_order']):
            stop = stops[stop_index]
            color = status_colors.get(stop['status'], (1.0, 1.0, 1.0))

            sphere = Marker()
            sphere.header.frame_id = 'map'
            sphere.ns = 'science_tour_stops'
            sphere.id = position
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = Point(x=stop['x'], y=stop['y'], z=0.4)
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.7
            sphere.color.a = 1.0
            sphere.color.r, sphere.color.g, sphere.color.b = color
            marker_array.markers.append(sphere)

            label = Marker()
            label.header.frame_id = 'map'
            label.ns = 'science_tour_labels'
            label.id = position
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = Point(x=stop['x'], y=stop['y'], z=1.2)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.6
            label.color.a = 1.0
            label.color.r, label.color.g, label.color.b = color
            label.text = f"{position + 1}. {stop['label']}"
            marker_array.markers.append(label)

        for namespace, order, color, alpha, z, thickness in (
                ('science_tour_baseline', self.tour_['baseline_order'], (0.6, 0.6, 0.6), 0.5, 0.15, 0.08),
                ('science_tour_optimised', self.tour_['optimised_order'], (0.0, 0.9, 1.0), 1.0, 0.35, 0.15)):
            line = Marker()
            line.header.frame_id = 'map'
            line.ns = namespace
            line.id = 0
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale.x = thickness
            line.color.a = alpha
            line.color.r, line.color.g, line.color.b = color
            # The closing leg back to the first stop is included, because the tour is a closed
            # loop - the rover has to get home - and that leg is part of the cost being compared.
            for stop_index in list(order) + [order[0]]:
                stop = stops[stop_index]
                line.points.append(Point(x=stop['x'], y=stop['y'], z=z))
            marker_array.markers.append(line)

        self.tour_markers_pub_.publish(marker_array)

    def planner_science_tour(self):
        """
        Advanced 5: drive to the current stop on the optimised tour.

        Uses a distance-scaled timeout (see TOUR_MIN_SPEED_MPS) instead of the flat
        GOAL_TIMEOUT_S, for the same reason planner_return_home() does: a tour leg crosses the
        explored cave, so the flat ceiling expires mid-leg and the stop is recorded as missed
        even though Nav2 was driving it correctly. Live testing with the flat ceiling reached 0
        of 12 stops, every one cancelled at 42-46s.
        """

        order = self.tour_['optimised_order']
        stop = self.tour_['stops'][order[self.tour_stop_index_]]

        # Face the next stop on arrival, so the rover is already pointed the right way for the
        # following leg (and, at a science stop, is looking across the area rather than at a wall).
        next_stop = self.tour_['stops'][order[(self.tour_stop_index_ + 1) % len(order)]]
        theta = math.atan2(next_stop['y'] - stop['y'], next_stop['x'] - stop['x'])

        # Scale off the straight-line distance the rover actually has to cover from where it is
        # now, not the planned leg length: the tour's first leg starts from wherever exploration
        # left the rover, which is nowhere near the stop the route has it leaving from.
        robot_pose = self.get_pose_2d()
        if robot_pose is not None:
            leg_distance_m = math.hypot(stop['x'] - robot_pose.x, stop['y'] - robot_pose.y)
            timeout_s = max(GOAL_TIMEOUT_S, leg_distance_m / TOUR_MIN_SPEED_MPS)
        else:
            timeout_s = GOAL_TIMEOUT_S

        self.get_logger().info(
            f"    stop {self.tour_stop_index_ + 1}/{len(order)}: {stop['label']} "
            f"(timeout {timeout_s:.0f}s)")
        self.planner_go_to_pose2d(Pose2D(x=stop['x'], y=stop['y'], theta=theta),
                                  timeout_s=timeout_s)

    def advance_science_tour(self):
        """
        Advanced 5: the goal for the current tour stop just finished - record it and move on.

        A stop that failed is recorded as missed and the tour continues to the next one rather
        than retrying: by this point the route is already planned and the remaining stops are
        all still worth visiting, so spending retries (and goal timeouts) on one awkward
        viewpoint at the cost of the rest of the tour is the wrong trade. The missed stop stays
        visible in RViz as an orange marker, so it's reported rather than quietly dropped.

        The one exception is a goal Nav2 threw out before the rover moved (see
        GOAL_INSTANT_FAILURE_S), which is re-sent instead. That trade-off is different because
        an instant rejection costs no driving time at all, so re-sending it doesn't come at the
        expense of the rest of the tour - and the stop hasn't actually been attempted yet.
        Scoring these as misses understated the result: of the 3 stops recorded missed in live
        testing, 2 were instant rejections at 9.6m and 9.1m, so the honest figure was 11/12
        rather than 9/12. Bounded per stop, so an unplannable viewpoint is still reported missed
        rather than stalling the tour on it.
        """

        order = self.tour_['optimised_order']
        stop = self.tour_['stops'][order[self.tour_stop_index_]]

        if self.goal_cancelled_by_pause_:
            # Not a real failure - the goal was cancelled by a pause (see
            # pause_mission_callback()). Leave the index alone so this same stop is re-sent.
            self.goal_cancelled_by_pause_ = False
            self.get_logger().info(f"    (resuming tour stop '{stop['label']}' after pause)")
            return

        instant_retries = stop.get('instant_retries', 0)
        if not self.last_goal_succeeded_ \
                and self.last_goal_duration_s_ < GOAL_INSTANT_FAILURE_S \
                and instant_retries < INSPECTION_MAX_INSTANT_RETRIES:
            # Nav2 rejected the goal before the rover moved - see the docstring. Leave the index
            # alone so this same stop is re-sent, exactly as the pause case above does.
            stop['instant_retries'] = instant_retries + 1
            self.get_logger().warn(
                f"    RETRY: tour stop '{stop['label']}' - Nav2 rejected the goal before the "
                f"rover moved ({stop['instant_retries']}/{INSPECTION_MAX_INSTANT_RETRIES})")
            return

        if self.last_goal_succeeded_:
            stop['status'] = 'visited'
            self.get_logger().info(f"    DONE: reached tour stop '{stop['label']}'")
        else:
            stop['status'] = 'missed'
            self.get_logger().warn(f"    MISSED: tour stop '{stop['label']}' not reached")

        self.tour_stop_index_ += 1
        if self.tour_stop_index_ >= len(order):
            visited = sum(1 for s in self.tour_['stops'] if s['status'] == 'visited')
            self.get_logger().info(
                f"    SCIENCE TOUR COMPLETE: {visited}/{len(order)} stops reached, "
                f"planned route {self.tour_['optimised_cost_m']:.1f}m "
                f"(vs {self.tour_['baseline_cost_m']:.1f}m baseline)")
            self.tour_complete_ = True
            self.tour_requested_ = False

        self.publish_tour_markers()

    def should_run_science_tour(self):
        """
        Advanced 5: whether the tour should be the active behaviour.

        Either exploration has finished (the natural trigger - the cave is mapped, so now go
        back round everything worth a second look) or the tour was explicitly requested via the
        start_science_tour service.
        """

        if self.tour_complete_ or self.tour_planning_failed_:
            return False
        if self.tour_requested_:
            return True

        return (self.exploration_complete_
                and self.get_parameter('science_tour_on_completion').value)

    def ensure_science_tour_planned(self):
        """
        Advanced 5: plan the tour if it hasn't been planned yet. Returns True if there's a tour
        to execute.

        On failure this latches tour_planning_failed_ so planning isn't retried on every tick
        for the rest of the mission - the usual cause is simply not enough distinct reachable
        locations having been found, which won't change once exploration is over.
        """

        if self.tour_ is not None:
            return True

        if not self.plan_science_tour():
            self.tour_planning_failed_ = True
            return False

        return True

    def start_science_tour_callback(self, request, response):
        """
        Advanced 5: service to run the science tour now, instead of waiting for exploration to
        finish on its own. Mainly for demos - the tour is the end-of-mission behaviour, and a
        full cave takes a while to explore.

        The tour is planned from whatever has been discovered so far, so an early call may find
        too few locations to plan one (reported back in the response rather than failing
        silently). Exploration resumes normally once the tour finishes.
        """

        if not self.mission_started_:
            response.success = False
            response.message = 'Mission not started yet'
            return response
        if self.tour_complete_:
            response.success = False
            response.message = 'Science tour already completed'
            return response
        if self.tour_requested_:
            response.success = False
            response.message = 'Science tour already running'
            return response

        self.tour_requested_ = True
        # Planning happens in main_loop (it needs a couple of seconds of whole-map BFS, which
        # shouldn't run inside a service callback), so clear the failure latch in case an
        # earlier attempt gave up when less of the cave had been mapped.
        self.tour_planning_failed_ = False
        if self.current_goal_handle_ is not None:
            self.current_goal_handle_.cancel_goal_async()

        self.get_logger().info('Science tour requested')
        response.success = True
        response.message = 'Science tour starting'
        return response

    #######################################################

    def main_loop(self):
        """
        Set the next goal pose and send to the action server
        See https://docs.nav2.org/concepts/index.html
        """

        # Wait for the GUI's "Start Exploring" button (or a manual
        # `ros2 service call /start_mission std_srvs/srv/Trigger {}`) - see mission_started_'s
        # comment in __init__.
        if not self.mission_started_:
            return

        # Mission already wrapped up (cave fully explored, every known artefact visited or
        # abandoned, robot back at base) - nothing further to do.
        if self.mission_complete_:
            return

        # Paused from the GUI - see pause_mission_callback() for what's already been done to
        # actually stop the robot; here we just stop making any further decisions until resumed.
        if self.mission_paused_:
            return

        # Don't do anything until SLAM is launched
        if not self.tf_buffer.can_transform(
                'map',
                'base_link',
                rclpy.time.Time()):
            self.get_logger().warn('Waiting for transform... Have you launched a SLAM node?')
            return

        # Planning 3 robustness: treat RETURN_HOME as pre-emptible, not a final committed
        # state - if an artefact gets confirmed while already heading home (e.g. spotted in
        # passing), divert to inspect it immediately instead of sailing past and leaving it
        # stranded as a permanent "pending" count. Checked every tick while a RETURN_HOME goal
        # is in flight; sending the new inspection goal below naturally preempts whatever Nav2
        # is still doing with the return-home goal.
        if self.planner_type_ == PlannerType.RETURN_HOME and not self.ready_for_next_goal_:
            preempting_target = self.find_inspection_target()
            if preempting_target is not None:
                self.get_logger().info(
                    f"    DIVERT: '{preempting_target['label']}' #{preempting_target['id']} "
                    f"spotted en route")
                self.abandon_current_goal('preempted')
                self.inspection_target_ = preempting_target
                self.inspection_attempts_ = 0
                self.inspection_instant_failures_ = 0
                self.planner_type_ = PlannerType.INSPECT_ARTIFACT
                self.publish_visited_artifact_markers()
                self.log_episode_start(self.planner_type_.name)
                self.planner_inspect_artifact()
                return

        #######################################################
        # Update flags related to the progress of the current planner

        # Robustness watchdog: Nav2's own recovery behaviour tree can retry a difficult goal
        # (e.g. repeated "Failed to make progress") for a very long time without ever reporting
        # back. If the current goal has been outstanding too long, give up on it ourselves
        # rather than waiting forever - the next goal we send will naturally preempt whatever
        # Nav2 is still doing with this one. Two triggers: a flat ceiling (GOAL_TIMEOUT_S), and
        # a progress-based one (GOAL_PROGRESS_STALL_S) based on the robot's own straight-line
        # distance to the goal - see goal_target_x_'s comment for why that, and not Nav2's
        # distance_remaining feedback, is what's tracked.
        #
        # The progress-based trigger is skipped for SCIENCE_TOUR for the same reason it's skipped
        # for RETURN_HOME, one step worse: a tour leg is the longest goal the mission sends, so
        # straight-line distance to the stop plateaus for long stretches while the rover rounds a
        # chamber or backtracks out of a dead end, and every one of those reads as a stall. With
        # both this check and the flat ceiling applying, live testing reached 0 of 12 tour stops.
        # SCIENCE_TOUR now sends a distance-scaled ceiling too (see planner_science_tour()), which
        # still bounds a genuinely stuck stop - it's recorded as missed and the tour moves on.
        #
        # The progress-based trigger is skipped for RETURN_HOME specifically: live testing hit
        # it firing twice in a row on the final return trip, genuinely stranding the robot ~14m
        # short of base once goal cancellation actually started working (see
        # abandon_current_goal()). RETURN_HOME only happens once, right at the very end, after
        # there's no more exploration time left to protect - there's no speed benefit to
        # abandoning it early, only the cost of a stranded robot and a wrongly-declared "didn't
        # quite reach base". The flat ceiling (self.goal_timeout_s_, normally GOAL_TIMEOUT_S)
        # still applies so it can't hang forever - but RETURN_HOME sends a distance-scaled one
        # instead of the flat 45s (see planner_return_home()), after the same flat ceiling was
        # live-tested hitting a genuinely long ~31m trip home and stranding the robot ~16m short.
        if not self.ready_for_next_goal_ and self.goal_sent_time_ is not None:
            now = self.get_clock().now()
            elapsed_s = (now - self.goal_sent_time_).nanoseconds / 1e9

            robot_pose = self.get_pose_2d()
            if robot_pose is not None and self.goal_target_x_ is not None:
                distance_to_goal = math.hypot(self.goal_target_x_ - robot_pose.x,
                                               self.goal_target_y_ - robot_pose.y)
                if self.goal_progress_distance_ is None \
                        or distance_to_goal < self.goal_progress_distance_ - GOAL_PROGRESS_EPSILON_M:
                    self.goal_progress_distance_ = distance_to_goal
                    self.goal_progress_time_ = now

            stalled_s = (now - self.goal_progress_time_).nanoseconds / 1e9 \
                if self.goal_progress_time_ is not None else 0.0
            if elapsed_s > self.goal_timeout_s_:
                self.abandon_current_goal('timed out')
            elif self.planner_type_ not in (PlannerType.RETURN_HOME, PlannerType.SCIENCE_TOUR) \
                    and stalled_s > GOAL_PROGRESS_STALL_S:
                self.abandon_current_goal('no progress')

        # Check if previous goal still running
        if not self.ready_for_next_goal_:
            # self.get_logger().info(f'Previous goal still running')
            return

        self.ready_for_next_goal_ = False

        #######################################################
        # Select the next planner to execute
        #
        # Planning 1: explore via frontiers by default (see planner_explore_frontier()).
        # Planning 2: pause exploration and approach a chosen-type artefact for close-range
        # inspection when one is known (see planner_inspect_artifact()).
        # Planning 3: switch between the two - each artefact is only inspected once
        # (visited_artifact_ids_), and a failed approach (Nav2 couldn't complete the goal -
        # the closest signal we have to "lost the artefact") is retried once before
        # abandoning that artefact and resuming exploration.
        if self.planner_type_ == PlannerType.INSPECT_ARTIFACT:
            # The previous tick's approach goal (for self.inspection_target_) just finished
            if self.last_goal_succeeded_:
                self.get_logger().info(
                    f"    DONE: {self.inspection_target_['label']} "
                    f"#{self.inspection_target_['id']} inspected")
                self.visited_artifact_ids_.add(self.inspection_target_['id'])
                self.inspection_target_ = None
                self.inspection_attempts_ = 0
                self.inspection_instant_failures_ = 0
            elif self.last_goal_duration_s_ < GOAL_INSTANT_FAILURE_S \
                    and self.inspection_instant_failures_ < INSPECTION_MAX_INSTANT_RETRIES:
                # Nav2 threw the goal out before the rover moved - see GOAL_INSTANT_FAILURE_S.
                # Retry without spending an attempt, because nothing has been learnt about the
                # artefact: counting these was abandoning science targets over handshake
                # timeouts inside the navigation stack. Bounded by INSPECTION_MAX_INSTANT_RETRIES
                # so a genuinely unplannable goal still gets written off rather than looping.
                self.inspection_instant_failures_ += 1
                self.get_logger().warn(
                    f"    RETRY: {self.inspection_target_['label']} "
                    f"#{self.inspection_target_['id']} - Nav2 rejected the goal before the rover "
                    f"moved ({self.inspection_instant_failures_}/{INSPECTION_MAX_INSTANT_RETRIES}), "
                    f"not counting it as a failed approach")
                # inspection_target_ stays set, so it's retried below
            else:
                self.inspection_attempts_ += 1
                if self.inspection_attempts_ > INSPECTION_MAX_RETRIES:
                    self.get_logger().warn(
                        f"    DONE: {self.inspection_target_['label']} "
                        f"#{self.inspection_target_['id']} abandoned after "
                        f"{self.inspection_attempts_} failed attempt(s)"
                        + (f" and {self.inspection_instant_failures_} instant rejection(s)"
                           if self.inspection_instant_failures_ else ''))
                    self.abandoned_artifact_ids_.add(self.inspection_target_['id'])
                    self.inspection_target_ = None
                    self.inspection_attempts_ = 0
                    self.inspection_instant_failures_ = 0
                # else: inspection_target_ stays set, so it's retried below
        elif self.planner_type_ == PlannerType.SCIENCE_TOUR:
            # Advanced 5: the goal for the current tour stop just finished
            self.advance_science_tour()
        elif self.planner_type_ == PlannerType.RETURN_HOME:
            # The "go home" goal just finished (either way) - re-check for any artefact
            # confirmed too late to trigger the en-route preemption above (e.g. confirmed in
            # the last tick before arrival) before actually declaring the mission over, rather
            # than leaving it stranded as a permanent "pending" count.
            pending_target = self.find_inspection_target()
            if pending_target is not None:
                self.get_logger().info(
                    f"    DIVERT: arrived home, but {pending_target['label']} "
                    f"#{pending_target['id']} is still pending - resuming inspection")
                self.inspection_target_ = pending_target
                self.inspection_attempts_ = 0
                self.inspection_instant_failures_ = 0
            else:
                elapsed_s = (self.get_clock().now() - self.start_time_).nanoseconds / 1e9
                outcome = 'reached base' if self.last_goal_succeeded_ else "didn't quite reach base"
                mission_avg_speed = self.total_distance_m_ / elapsed_s if elapsed_s > 0 else 0.0
                self.get_logger().info(
                    f'    MISSION COMPLETE: {outcome} - '
                    f'{len(self.visited_artifact_ids_)} inspected, '
                    f'{len(self.abandoned_artifact_ids_)} abandoned, '
                    f'{elapsed_s:.0f}s total, avg {mission_avg_speed:.2f} m/s')
                self.mission_complete_ = True
                return

        self.publish_visited_artifact_markers()

        if self.inspection_target_ is None:
            self.inspection_target_ = self.find_inspection_target()
            self.inspection_attempts_ = 0
            self.inspection_instant_failures_ = 0

        if self.inspection_target_ is not None:
            self.planner_type_ = PlannerType.INSPECT_ARTIFACT
        elif self.should_run_science_tour() and self.ensure_science_tour_planned():
            # Advanced 5: revisit the science locations on an optimised route. Sits between
            # inspection and return-home: a pending artefact is still inspected first (it's a
            # new discovery, and it also becomes a candidate tour stop once confirmed), but the
            # tour happens before the robot commits to going home and stopping. If no tour can
            # be planned, this falls through to the normal end-of-mission behaviour below.
            self.planner_type_ = PlannerType.SCIENCE_TOUR
        elif self.exploration_complete_:
            # Nothing left to explore and nothing left to inspect - head home and stop
            self.planner_type_ = PlannerType.RETURN_HOME
        else:
            self.planner_type_ = PlannerType.EXPLORE_FRONTIER

        #######################################################
        # Execute the planner by calling the relevant method
        # Add your own planners here!
        self.log_episode_start(self.planner_type_.name)
        if self.planner_type_ == PlannerType.MOVE_FORWARDS:
            self.planner_move_forwards(10)
        elif self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.planner_go_to_first_artifact()
        elif self.planner_type_ == PlannerType.RETURN_HOME:
            self.planner_return_home()
        elif self.planner_type_ == PlannerType.RANDOM_WALK:
            self.planner_random_walk()
        elif self.planner_type_ == PlannerType.RANDOM_GOAL:
            self.planner_random_goal()
        elif self.planner_type_ == PlannerType.EXPLORE_FRONTIER:
            self.planner_explore_frontier()
        elif self.planner_type_ == PlannerType.INSPECT_ARTIFACT:
            self.planner_inspect_artifact()
        elif self.planner_type_ == PlannerType.SCIENCE_TOUR:
            self.planner_science_tour()
        else:
            self.get_logger().error('No valid planner selected')
            self.destroy_node()


        #######################################################

    def log_status(self):
        """Periodic STATUS heartbeat: elapsed time, current mode, and progress counts"""

        if not self.mission_started_:
            return

        elapsed_s = (self.get_clock().now() - self.start_time_).nanoseconds / 1e9
        inspectable = [c for c in self.artifact_clusters_
                       if c['label'] in INSPECTION_ARTIFACT_LABELS
                       and c['num_observations'] >= ARTIFACT_CONFIRMATION_OBSERVATIONS]
        visited = len(self.visited_artifact_ids_)
        abandoned = len(self.abandoned_artifact_ids_)
        pending = len(inspectable) - visited - abandoned

        self.get_logger().info(
            f'STATUS   t={elapsed_s:.0f}s   mode={self.planner_type_.name}   '
            f'visited={visited}   abandoned={abandoned}   pending={pending}')

    def publish_gui_status(self):
        """
        GUI data feed: publish mission status + the full confirmed-artefact inventory as one
        JSON payload - see gui_status_pub_'s comment for why this topic exists and why it's a
        plain JSON string rather than a custom message type.
        """

        if not self.mission_started_:
            payload = {'mission_started': False, 'mission_paused': False, 'mission_complete': False,
                       'mode': 'WAITING', 'elapsed_s': 0.0, 'visited': 0, 'abandoned': 0, 'pending': 0,
                       'avg_speed_mps': 0.0, 'artifacts': [], 'robot_x': None, 'robot_y': None,
                       'goal_x': None, 'goal_y': None, 'xlim': None, 'ylim': None, 'tour': None}
            self.gui_status_pub_.publish(String(data=json.dumps(payload)))
            return

        elapsed_s = (self.get_clock().now() - self.start_time_).nanoseconds / 1e9
        avg_speed_mps = self.total_distance_m_ / elapsed_s if elapsed_s > 0 else 0.0

        # For the GUI's mini-map - see gui.py's MapWidget.
        robot_pose = self.get_pose_2d()
        robot_x = round(robot_pose.x, 2) if robot_pose is not None else None
        robot_y = round(robot_pose.y, 2) if robot_pose is not None else None

        artifacts = []
        for cluster in self.artifact_clusters_:
            if cluster['num_observations'] < ARTIFACT_CONFIRMATION_OBSERVATIONS:
                continue  # not yet confirmed - see ARTIFACT_CONFIRMATION_OBSERVATIONS
            if cluster['label'] not in INSPECTION_ARTIFACT_LABELS:
                status = 'detected'
            elif cluster['id'] in self.visited_artifact_ids_:
                status = 'visited'
            elif cluster['id'] in self.abandoned_artifact_ids_:
                status = 'abandoned'
            else:
                status = 'pending'
            artifacts.append({
                'id': cluster['id'],
                'label': cluster['label'],
                'x': round(cluster['position'].x, 2),
                'y': round(cluster['position'].y, 2),
                'status': status,
            })

        inspectable = [c for c in self.artifact_clusters_
                       if c['label'] in INSPECTION_ARTIFACT_LABELS
                       and c['num_observations'] >= ARTIFACT_CONFIRMATION_OBSERVATIONS]
        visited = len(self.visited_artifact_ids_)
        abandoned = len(self.abandoned_artifact_ids_)
        pending = len(inspectable) - visited - abandoned

        payload = {
            'mission_started': True,
            'mission_paused': self.mission_paused_,
            'mission_complete': self.mission_complete_,
            'mode': self.planner_type_.name,
            'elapsed_s': round(elapsed_s, 1),
            'visited': visited,
            'abandoned': abandoned,
            'pending': pending,
            'avg_speed_mps': round(avg_speed_mps, 2),
            'artifacts': artifacts,
            'robot_x': robot_x,
            'robot_y': robot_y,
            'goal_x': round(self.goal_target_x_, 2) if self.goal_target_x_ is not None else None,
            'goal_y': round(self.goal_target_y_, 2) if self.goal_target_y_ is not None else None,
            'xlim': list(self.xlim_),
            'ylim': list(self.ylim_),
            'tour': self.gui_tour_summary(),
        }
        self.gui_status_pub_.publish(String(data=json.dumps(payload)))

    def gui_tour_summary(self):
        """
        Advanced 5, for the GUI: the optimised route, the baseline it beat, and how far through
        it we are. None until a tour has actually been planned.

        Without this the tour was invisible to anyone driving from the GUI - the only evidence
        it had run at all was the Mode field reading SCIENCE_TOUR, and the saving the task is
        actually assessed on (baseline vs optimised) only ever appeared in the terminal log.
        Stops are listed in visiting order rather than the order they were collected in, since
        that order *is* the result.
        """

        if self.tour_ is None:
            return None

        stops = self.tour_['stops']
        order = self.tour_['optimised_order']
        return {
            'baseline_m': round(self.tour_['baseline_cost_m'], 1),
            'optimised_m': round(self.tour_['optimised_cost_m'], 1),
            'saved_m': round(self.tour_['saved_m'], 1),
            'saved_percent': round(self.tour_['saved_percent'], 1),
            'stop_index': self.tour_stop_index_,
            'complete': self.tour_complete_,
            'stops': [{
                'position': position + 1,
                'label': stops[stop_index]['label'],
                'kind': stops[stop_index]['kind'],
                'x': round(stops[stop_index]['x'], 2),
                'y': round(stops[stop_index]['y'], 2),
                'status': stops[stop_index]['status'],
            } for position, stop_index in enumerate(order)],
        }

def main():
    # Initialise
    rclpy.init()

    # Create the cave explorer
    cave_explorer = CaveExplorer()

    while rclpy.ok():
        rclpy.spin(cave_explorer)