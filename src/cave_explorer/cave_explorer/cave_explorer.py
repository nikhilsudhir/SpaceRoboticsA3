#!/usr/bin/env python3

import functools
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

def pose2d_to_pose(pose_2d):
    """Convert a Pose2D to a full 3D Pose"""
    pose = Pose()

    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y

    pose.orientation.w = math.cos(pose_2d.theta / 2.0)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)

    return pose


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
        'label': 'toy_story_alien',
        'hsv_lower': (25, 120, 30),
        'hsv_upper': (50, 255, 140),
        'min_area': 300,
        'color_rgb': (0, 140, 70),
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
        'label': 'mossy_boulder',
        'hsv_lower': (8, 45, 35),
        'hsv_upper': (42, 230, 210),
        'min_area': 400,
        'color_rgb': (140, 90, 20),
    },
    {
        'label': 'mushroom_blue',
        'hsv_lower': (80, 40, 140),
        'hsv_upper': (130, 180, 255),
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
# Candidate frontiers within this distance (metres) of a goal we've ever sent before are
# skipped, so we don't repeatedly re-target the same spot (e.g. if Nav2 can't quite reach the
# frontier itself, or a sensor-shadow sliver flickers in and out near a path we've already
# driven). Unlike a short-lived "recent goals" window, recent_frontier_goals_ is never
# truncated - every frontier goal from the whole run is remembered, so the robot can't forget
# it was already here and go back after enough time/goals have passed.
FRONTIER_REVISIT_RADIUS_M = 4.0

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

# Planning 3 robustness: Perception 3's clustering (ARTIFACT_CLUSTER_DISTANCE_M) can
# over-segment one physical artefact into several same-label clusters if repeated
# localisation estimates for it land further apart than that threshold (observed with the
# colour-blob detector, especially at close range where many detections arrive per second).
# A same-label cluster within this distance of one we've already visited/abandoned is treated
# as the same physical object rather than a new one, so we don't keep "rediscovering" and
# re-approaching it under a new id. Deliberately wider than ARTIFACT_CLUSTER_DISTANCE_M (1.5m)
# to absorb that drift, while still being tighter than the spacing between distinct artefacts.
INSPECTION_DUPLICATE_RADIUS_M = 3.0

# Planning 1 robustness: how many random points planner_random_walk() tries before giving up
# and holding position, when looking for one that's actually navigable (see is_point_navigable()).
RANDOM_WALK_MAX_ATTEMPTS = 30

# How often (seconds) to print the [STATUS] heartbeat - see log_status().
STATUS_LOG_PERIOD_S = 30.0

# Planning 1/3 robustness: Nav2's own recovery behaviour tree can retry a difficult goal (e.g.
# "Failed to make progress") more or less indefinitely without ever reporting success or
# failure back to us. If a goal hasn't finished within this many seconds, we give up on it
# ourselves - see the watchdog at the top of main_loop() - rather than waiting forever.
GOAL_TIMEOUT_S = 60.0


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

        # Planning 1: world-frame (x, y) of every frontier goal ever sent this run, so we never
        # re-target the same spot - see FRONTIER_REVISIT_RADIUS_M's comment above.
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

        # Timer for main loop
        self.main_loop_timer_ = self.create_timer(0.2, self.main_loop)

        # Periodic status heartbeat (elapsed time + progress counts) - see log_status()
        self.start_time_ = self.get_clock().now()
        self.status_timer_ = self.create_timer(STATUS_LOG_PERIOD_S, self.log_status)
    
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

        # Perception 2: run all detectors and merge their results
        detections = self.detect_stop_signs(image) + self.detect_color_artifacts(image)

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
            for contour in contours:
                if cv2.contourArea(contour) < profile['min_area']:
                    continue

                bbox = cv2.boundingRect(contour)
                detections.append(Detection(profile['label'], bbox, profile['color_rgb']))

        return detections

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
        (running average), or start a new cluster if it's not close to any existing one.
        """

        for cluster in self.artifact_clusters_:
            if cluster['label'] != label:
                continue

            dx = position.x - cluster['position'].x
            dy = position.y - cluster['position'].y
            dz = position.z - cluster['position'].z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if distance <= ARTIFACT_CLUSTER_DISTANCE_M:
                n = cluster['num_observations']
                cluster['position'].x = (cluster['position'].x * n + position.x) / (n + 1)
                cluster['position'].y = (cluster['position'].y * n + position.y) / (n + 1)
                cluster['position'].z = (cluster['position'].z * n + position.z) / (n + 1)
                cluster['num_observations'] += 1
                return

        # Planning 3: a stable id per cluster (distinct from its position in the list, which
        # isn't guaranteed to stay fixed), used to track which specific artefacts have
        # already been inspected - see visited_artifact_ids_.
        cluster_id = self.next_artifact_cluster_id_
        self.next_artifact_cluster_id_ += 1

        self.artifact_clusters_.append({
            'id': cluster_id,
            'label': label,
            'position': Point(x=position.x, y=position.y, z=position.z),
            'num_observations': 1,
        })
        self.get_logger().info(
            f'[DISCOVER] {label} #{cluster_id} at ({position.x:.1f}, {position.y:.1f})')

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

    def is_point_navigable(self, x, y, clearance_cells=2):
        """
        Planning 2: check whether (x, y) is free space (not occupied, not unexplored) in the
        current map, with a small clearance margin - used to validate a candidate inspection
        standoff point before sending it to Nav2, since a point computed purely from the
        artefact's position can otherwise land inside a wall or in unexplored space (which
        Nav2's global planner then simply can't find a path to at all).
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

        # OR together the unknown mask shifted by each of the 8 neighbour offsets, to find
        # (without a slow per-cell python loop) which free cells have an unknown neighbour
        neighbours_unknown = np.zeros_like(unknown)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.roll(np.roll(unknown, dr, axis=0), dc, axis=1)
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
                neighbours_unknown |= shifted

        return free & neighbours_unknown

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
        """

        grid = self.map_grid_
        height, width = grid.shape
        free = (grid >= 0) & (grid < FRONTIER_OCCUPIED_THRESHOLD)

        dist = np.full((height, width), -1, dtype=np.int32)
        if not free[start_row, start_col]:
            return dist

        dist[start_row, start_col] = 0
        queue = deque([(start_row, start_col)])
        while queue:
            r, c = queue.popleft()
            d = dist[r, c]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width \
                            and free[nr, nc] and dist[nr, nc] == -1:
                        dist[nr, nc] = d + 1
                        queue.append((nr, nc))

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
        height, width = path_distance_grid.shape

        candidates = []
        for cluster in clusters:
            if len(cluster) < FRONTIER_MIN_CLUSTER_SIZE:
                continue

            mean_row = sum(r for r, c in cluster) / len(cluster)
            mean_col = sum(c for r, c in cluster) / len(cluster)
            world_x, world_y = self.grid_to_world(mean_row, mean_col)

            if any(math.hypot(world_x - gx, world_y - gy) < FRONTIER_REVISIT_RADIUS_M
                   for gx, gy in self.recent_frontier_goals_):
                continue

            row_idx = min(max(int(round(mean_row)), 0), height - 1)
            col_idx = min(max(int(round(mean_col)), 0), width - 1)
            path_cells = path_distance_grid[row_idx, col_idx]
            # Not reachable through known free space at all (e.g. an isolated noise pocket) -
            # still include it (for visualisation) but make sure it never scores highest
            distance = path_cells * self.map_resolution_ if path_cells >= 0 else float('inf')

            score = len(cluster) / (1.0 + distance / FRONTIER_DISTANCE_SCALE_M)
            candidates.append({
                'position': (world_x, world_y),
                'size': len(cluster),
                'distance': distance,
                'score': score,
            })

        if not candidates:
            return None, []

        chosen = max(candidates, key=lambda c: c['score'])
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
                self.get_logger().info('No map yet - taking a random step until SLAM has data')
            else:
                self.no_frontier_streak_ += 1
                if self.no_frontier_streak_ >= EXPLORATION_COMPLETE_STREAK:
                    if not self.exploration_complete_:
                        self.get_logger().info(
                            f'EXPLORATION COMPLETE: no frontiers found for '
                            f'{self.no_frontier_streak_} consecutive checks - the cave appears '
                            'to be fully mapped')
                    self.exploration_complete_ = True
                else:
                    self.get_logger().info('No frontiers right now - taking a random step')
            self.planner_random_walk()
            return

        self.no_frontier_streak_ = 0
        goal_x, goal_y = chosen['position']
        theta = math.atan2(goal_y - robot_pose.y, goal_x - robot_pose.x)
        goal_pose2d = Pose2D(x=goal_x, y=goal_y, theta=theta)

        # Never evicted - see FRONTIER_REVISIT_RADIUS_M's comment for why
        self.recent_frontier_goals_.append((goal_x, goal_y))

        self.planner_go_to_pose2d(goal_pose2d)

    def find_inspection_target(self):
        """
        Planning 2/3: return the first known artefact cluster of a chosen type (see
        INSPECTION_ARTIFACT_LABELS) that hasn't already been inspected or abandoned, or None.
        """

        for cluster in self.artifact_clusters_:
            if cluster['label'] not in INSPECTION_ARTIFACT_LABELS:
                continue
            if cluster['id'] in self.visited_artifact_ids_ or cluster['id'] in self.abandoned_artifact_ids_:
                continue
            if self.is_duplicate_of_handled_artifact(cluster):
                # Treat it as the same physical artefact as one we've already dealt with (see
                # INSPECTION_DUPLICATE_RADIUS_M) - mark it visited too, so this id (and any
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
            if math.hypot(dx, dy) < INSPECTION_DUPLICATE_RADIUS_M:
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
        it, using the first one that validates as free space in the current map (see
        is_point_navigable()). A standoff point computed from the artefact's position alone can
        otherwise land inside a wall or in unexplored space if the artefact happens to be close
        to one - which Nav2's global planner then can't find a path to at all, rather than just
        navigating there poorly.
        """

        robot_pose = self.get_pose_2d()
        if robot_pose is None or self.inspection_target_ is None:
            return

        target = self.inspection_target_['position']
        dx = robot_pose.x - target.x
        dy = robot_pose.y - target.y
        base_angle = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-3 else 0.0

        candidate_offsets = [0.0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2,
                             3 * math.pi / 4, -3 * math.pi / 4, math.pi]
        goal_x = goal_y = None
        for offset in candidate_offsets:
            angle = base_angle + offset
            candidate_x = target.x + math.cos(angle) * INSPECTION_STANDOFF_DISTANCE_M
            candidate_y = target.y + math.sin(angle) * INSPECTION_STANDOFF_DISTANCE_M
            if self.is_point_navigable(candidate_x, candidate_y):
                goal_x, goal_y = candidate_x, candidate_y
                break

        if goal_x is None:
            # Nothing validated (e.g. map not built yet, or the artefact is tightly enclosed) -
            # fall back to the direct approach anyway; the goal watchdog (GOAL_TIMEOUT_S) will
            # recover if Nav2 genuinely can't reach it
            goal_x = target.x + math.cos(base_angle) * INSPECTION_STANDOFF_DISTANCE_M
            goal_y = target.y + math.sin(base_angle) * INSPECTION_STANDOFF_DISTANCE_M

        theta = math.atan2(target.y - goal_y, target.x - goal_x)  # face the artefact
        self.planner_go_to_pose2d(Pose2D(x=goal_x, y=goal_y, theta=theta))

    def planner_go_to_pose2d(self, pose2d):
        """Go to a provided 2d pose"""

        # Send a goal to navigate_to_pose with self.nav2_action_client_
        action_goal = NavigateToPose.Goal()
        action_goal.pose.header.stamp = self.get_clock().now().to_msg()
        action_goal.pose.header.frame_id = 'map'
        action_goal.pose.pose = pose2d_to_pose(pose2d)

        # Publish visualisation
        self.goal_pose_vis_.publish(action_goal.pose)

        # Decide whether to show feedback or not
        if self.get_parameter('print_feedback').value:
            feedback_method = self.feedback_callback
        else:
            feedback_method = None

        # Send goal to action server. Tag this goal with a generation number, and have the
        # callbacks below check it against self.goal_generation_ before acting - if the
        # watchdog in main_loop() has since given up on this goal and moved on, these would
        # otherwise be a late, stale response overwriting the state of the *next* goal.
        self.goal_generation_ += 1
        generation = self.goal_generation_
        self.goal_sent_time_ = self.get_clock().now()

        self.get_logger().info(f'  -> goal ({pose2d.x:.2f}, {pose2d.y:.2f})')
        self.send_goal_future_ = self.nav2_action_client_.send_goal_async(
            action_goal,
            feedback_callback=feedback_method)
        self.send_goal_future_.add_done_callback(
            functools.partial(self.goal_response_callback, generation=generation))

    def goal_response_callback(self, future, generation):
        """The requested goal pose has been sent to the action server"""

        if generation != self.goal_generation_:
            return  # stale: we've since given up on this goal (see GOAL_TIMEOUT_S) and moved on

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('  <- rejected')
            self.last_goal_succeeded_ = False
            self.ready_for_next_goal_ = True
            return

        # Goal accepted: get result when it's completed
        self.get_result_future_ = goal_handle.get_result_async()
        self.get_result_future_.add_done_callback(
            functools.partial(self.goal_reached_callback, generation=generation))

    def feedback_callback(self, feedback_msg):
        """Monitor the feedback from the action server"""

        feedback = feedback_msg.feedback

        self.get_logger().info(f'{feedback.distance_remaining:.2f} m remaining')

    def goal_reached_callback(self, future, generation):
        """The requested goal has finished (successfully or not)"""

        if generation != self.goal_generation_:
            return  # stale: we've since given up on this goal (see GOAL_TIMEOUT_S) and moved on

        status = future.result().status
        self.last_goal_succeeded_ = (status == GoalStatus.STATUS_SUCCEEDED)
        if self.last_goal_succeeded_:
            self.get_logger().info('  <- reached')
        else:
            self.get_logger().info(f'  <- failed (status={status})')
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
        """Return to the origin"""

        goal_pose2d = Pose2D(
            x = 0.0,
            y = 0.0,
            theta = math.pi
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_random_walk(self):
        """
        Go to a random, navigable location within the current map bounds.

        Sampling uniformly within xlim_/ylim_ can easily land inside a wall or unexplored
        space for an irregular cave shape (most of the bounding box isn't actual floor) -
        that previously sent Nav2 on goals it could never reach at all, costing a full 60s
        goal-timeout each time. Each candidate is now checked with is_point_navigable()
        before being sent.
        """

        for _ in range(RANDOM_WALK_MAX_ATTEMPTS):
            x = random.uniform(self.xlim_[0], self.xlim_[1])
            y = random.uniform(self.ylim_[0], self.ylim_[1])
            if self.is_point_navigable(x, y):
                goal_pose2d = Pose2D(x=x, y=y, theta=random.uniform(0, 2 * math.pi))
                self.planner_go_to_pose2d(goal_pose2d)
                return

        # No navigable point found (e.g. map not built yet) - hold position rather than risk
        # another long Nav2 failure/recovery cycle on an unreachable random point
        self.get_logger().warn('No navigable random point found - holding position')
        robot_pose = self.get_pose_2d()
        if robot_pose is not None:
            self.planner_go_to_pose2d(robot_pose)

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

        # Mission already wrapped up (cave fully explored, every known artefact visited or
        # abandoned, robot back at base) - nothing further to do.
        if self.mission_complete_:
            return

        # Don't do anything until SLAM is launched
        if not self.tf_buffer.can_transform(
                'map',
                'base_link',
                rclpy.time.Time()):
            self.get_logger().warn('Waiting for transform... Have you launched a SLAM node?')
            return

        #######################################################
        # Update flags related to the progress of the current planner

        # Robustness watchdog: Nav2's own recovery behaviour tree can retry a difficult goal
        # (e.g. repeated "Failed to make progress") for a very long time without ever reporting
        # back. If the current goal has been outstanding too long, give up on it ourselves
        # rather than waiting forever - the next goal we send will naturally preempt whatever
        # Nav2 is still doing with this one.
        if not self.ready_for_next_goal_ and self.goal_sent_time_ is not None:
            elapsed_s = (self.get_clock().now() - self.goal_sent_time_).nanoseconds / 1e9
            if elapsed_s > GOAL_TIMEOUT_S:
                self.get_logger().info(f'[GOAL]   <- timed out after {elapsed_s:.0f}s, abandoning')
                self.last_goal_succeeded_ = False
                self.ready_for_next_goal_ = True
                self.goal_sent_time_ = None

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
                    f"Inspected '{self.inspection_target_['label']}' "
                    f"(id={self.inspection_target_['id']})")
                self.visited_artifact_ids_.add(self.inspection_target_['id'])
                self.inspection_target_ = None
                self.inspection_attempts_ = 0
            else:
                self.inspection_attempts_ += 1
                if self.inspection_attempts_ > INSPECTION_MAX_RETRIES:
                    self.get_logger().warn(
                        f"Abandoning '{self.inspection_target_['label']}' "
                        f"(id={self.inspection_target_['id']}) after "
                        f"{self.inspection_attempts_} failed attempt(s)")
                    self.abandoned_artifact_ids_.add(self.inspection_target_['id'])
                    self.inspection_target_ = None
                    self.inspection_attempts_ = 0
                # else: inspection_target_ stays set, so it's retried below
        elif self.planner_type_ == PlannerType.RETURN_HOME:
            # The final "go home" goal just finished (either way) - mission over, don't fall
            # through to picking another goal below
            elapsed_s = (self.get_clock().now() - self.start_time_).nanoseconds / 1e9
            outcome = 'reached base' if self.last_goal_succeeded_ else "didn't quite reach base"
            self.get_logger().info(
                f'[MISSION COMPLETE] {outcome} after {elapsed_s:.0f}s - '
                f'{len(self.visited_artifact_ids_)} artefact(s) inspected, '
                f'{len(self.abandoned_artifact_ids_)} abandoned')
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
        self.get_logger().info(f'[GOAL] {self.planner_type_.name}')
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
        """Periodic [STATUS] heartbeat: elapsed time, current mode, and progress counts"""

        elapsed_s = (self.get_clock().now() - self.start_time_).nanoseconds / 1e9
        inspectable = [c for c in self.artifact_clusters_ if c['label'] in INSPECTION_ARTIFACT_LABELS]
        visited = len(self.visited_artifact_ids_)
        abandoned = len(self.abandoned_artifact_ids_)
        pending = len(inspectable) - visited - abandoned

        self.get_logger().info(
            f'[STATUS] t={elapsed_s:.0f}s | mode={self.planner_type_.name} | '
            f'artefacts: visited={visited} abandoned={abandoned} pending={pending}')

def main():
    # Initialise
    rclpy.init()

    # Create the cave explorer
    cave_explorer = CaveExplorer()

    while rclpy.ok():
        rclpy.spin(cave_explorer)