#!/usr/bin/env python3

import functools
import json
import math
import os
import random
from collections import deque, namedtuple
from enum import Enum

import cv2  # OpenCV2
import numpy as np
import rclpy
import tf2_geometry_msgs
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Point, PointStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


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
# toy_story_alien and mossy_boulder are deliberately NOT included here: neither is actually
# placed anywhere in mars_cave.sdf (checked directly against the world file), so every
# detection that would appear under either label is guaranteed to be a false positive -
# mostly wall/floor rock texture, and (for toy_story_alien specifically) a major contributor
# to green_crystals/green_alien getting mislabeled, since its hue/value range overlapped both.
# Dropping them removes that confusion entirely with zero loss of real detection capability.
ARTIFACT_COLOR_PROFILES = [
    {
        'label': 'blue_cube',
        'hsv_lower': (100, 120, 60),
        'hsv_upper': (125, 255, 255),
        'min_area': 300,
        'color_rgb': (30, 90, 255),
    },
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
        # Narrowed from (80, 40, 140)-(130, 180, 255): that window was wide enough to fully
        # contain ice_formation's, white_sphere's, AND blue_cube's hue ranges at once, and
        # live testing confirmed mushroom_blue winning the cross-label tie-break against all
        # three in separate runs. First pass (hue floor 115) cleared ice_formation (max 115)
        # and white_sphere (max 120) but still left a 10-degree overlap with blue_cube
        # (100-125) - live testing confirmed that's exactly where the mushroom's actual
        # rendered colour sits, since it kept winning against blue_cube specifically even
        # after the other two were fixed. Hue floor raised again, past blue_cube's ceiling
        # (125) entirely this time.
        'hsv_lower': (128, 90, 150),
        'hsv_upper': (155, 200, 255),
        'min_area': 300,
        'color_rgb': (255, 120, 255),
    },
]

# Lookup used to colour Perception 3's RViz markers to match each artefact's Perception 2 detection colour
ARTIFACT_COLOR_BY_LABEL = {profile['label']: profile['color_rgb'] for profile in ARTIFACT_COLOR_PROFILES}

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
INSPECTION_ARTIFACT_LABELS = ['blue_cube', 'white_sphere', 'green_crystals']

# Planning 2: distance (metres) to stop from an artefact for close-range inspection.
INSPECTION_STANDOFF_DISTANCE_M = 2.0

# Planning 3: if a close-range inspection goal doesn't succeed (Nav2 aborts/rejects it - e.g.
# the artefact's standoff point turned out to be unreachable), retry it this many times before
# abandoning that artefact and resuming exploration.
INSPECTION_MAX_RETRIES = 1

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
        self.last_goal_succeeded_ = None
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
        self.pause_mission_srv_ = self.create_service(
            Trigger, 'pause_mission', self.pause_mission_callback)
        self.resume_mission_srv_ = self.create_service(
            Trigger, 'resume_mission', self.resume_mission_callback)

        # GUI control: skip straight to RETURN_HOME - see force_return_home_callback().
        self.force_return_home_srv_ = self.create_service(
            Trigger, 'force_return_home', self.force_return_home_callback)

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
        is_point_navigable()) AND (b) actually reachable from the robot's current position
        through currently-known free space (reusing the same BFS path-distance grid frontier
        selection uses - see compute_path_distance_grid()). A standoff point computed from the
        artefact's position alone can otherwise land inside a wall, in unexplored space, or on
        the far side of a wall with no known connecting path - which Nav2's global planner then
        can't find a path to at all, rather than just navigating there poorly.
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
        goal_x = goal_y = None
        for offset in candidate_offsets:
            angle = base_angle + offset
            candidate_x = target.x + math.cos(angle) * INSPECTION_STANDOFF_DISTANCE_M
            candidate_y = target.y + math.sin(angle) * INSPECTION_STANDOFF_DISTANCE_M
            if self.is_point_navigable(candidate_x, candidate_y) and is_reachable(candidate_x, candidate_y):
                goal_x, goal_y = candidate_x, candidate_y
                break

        if goal_x is None:
            # Nothing validated (e.g. map not built yet, or the artefact is tightly enclosed) -
            # fall back to the direct approach anyway; the goal watchdog (GOAL_TIMEOUT_S) will
            # recover if Nav2 genuinely can't reach it. Logged so we have evidence of how often
            # this unvalidated fallback actually fires, rather than just suspecting it might.
            self.get_logger().warn(
                f"    (no validated standoff point for {self.inspection_target_['label']} "
                f"#{self.inspection_target_['id']} - falling back to unvalidated direct approach)")
            goal_x = target.x + math.cos(base_angle) * INSPECTION_STANDOFF_DISTANCE_M
            goal_y = target.y + math.sin(base_angle) * INSPECTION_STANDOFF_DISTANCE_M

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

        if self.goal_initial_distance_m_ is not None and elapsed_s > 0:
            avg_speed = self.goal_initial_distance_m_ / elapsed_s
            self.get_logger().info(
                f'    <- {outcome_text} in {elapsed_s:.0f}s '
                f'({self.goal_initial_distance_m_:.1f}m, avg {avg_speed:.2f} m/s)')
            self.total_distance_m_ += self.goal_initial_distance_m_
        else:
            self.get_logger().info(f'    <- {outcome_text}')

    def abandon_current_goal(self, outcome_text='abandoned'):
        """
        Give up on the in-flight Nav2 goal: cancel it with Nav2 (so it actually stops trying,
        rather than continuing to execute in the background while we've locally moved on to
        something else) and bump goal_generation_ (so a late response/result callback for it
        is recognised as stale and ignored, even on the one path that doesn't immediately send
        a replacement goal - RETURN_HOME's mission-complete branch, which just returns).

        Without this, a goal we'd given up on could still actually succeed with Nav2 minutes
        later, update last_goal_succeeded_/ready_for_next_goal_ out from under whatever we've
        since moved on to, and (for the mission-complete case) make the final [MISSION
        COMPLETE] log wrongly claim "didn't quite reach base" when it actually did, just later
        than our patience allowed.
        """

        self.log_goal_outcome(outcome_text)
        if self.current_goal_handle_ is not None:
            self.current_goal_handle_.cancel_goal_async()
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
            elif self.planner_type_ != PlannerType.RETURN_HOME and stalled_s > GOAL_PROGRESS_STALL_S:
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
            else:
                self.inspection_attempts_ += 1
                if self.inspection_attempts_ > INSPECTION_MAX_RETRIES:
                    self.get_logger().warn(
                        f"    DONE: {self.inspection_target_['label']} "
                        f"#{self.inspection_target_['id']} abandoned after "
                        f"{self.inspection_attempts_} failed attempt(s)")
                    self.abandoned_artifact_ids_.add(self.inspection_target_['id'])
                    self.inspection_target_ = None
                    self.inspection_attempts_ = 0
                # else: inspection_target_ stays set, so it's retried below
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

        if self.inspection_target_ is not None:
            self.planner_type_ = PlannerType.INSPECT_ARTIFACT
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
                       'goal_x': None, 'goal_y': None, 'xlim': None, 'ylim': None}
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
        }
        self.gui_status_pub_.publish(String(data=json.dumps(payload)))

def main():
    # Initialise
    rclpy.init()

    # Create the cave explorer
    cave_explorer = CaveExplorer()

    while rclpy.ok():
        rclpy.spin(cave_explorer)