# SpaceRoboticsA3

ROS2 workspace for the 49274 Space Robotics team project (brief in [`team_project_2026.pdf`](team_project_2026.pdf)): an autonomously exploring Mars rover that maps a simulated Martian cave, and detects and localises artefacts of interest.

## Layout

This repo holds the source packages, not the colcon workspace itself. Each folder under `src/` is one ROS2 package:

- `src/cave_explorer/` — the `cave_explorer` package (`ament_python`). Robot nodes, launch files, Nav2/SLAM config, and the Gazebo worlds/models/urdf used to simulate it.

As more packages are added, each new one should get its own folder under `src/`.

Inside `src/cave_explorer/`:

- `cave_explorer/cave_explorer.py` — the main ROS2 node. This is where most of the project's decision-making, perception and planning code lives.
- `launch/` — a combined launch file plus the three underlying launch files it includes (see [Running the simulation](#running-the-simulation) below).
- `config/` — Nav2, SLAM, EKF (`robot_localization`), RViz and Gazebo topic-bridge parameters, plus the OpenCV cascade classifier used for the starter artefact detector (`stop_data.xml`).
- `urdf/` — the rover's URDF/xacro description and meshes.
- `worlds/` — the Gazebo cave and surface world files, plus their models/textures.

## Prerequisites

This has been developed and tested on **Ubuntu 22.04** with **ROS2 Humble**. If you don't have ROS2 installed yet, follow the [official Humble installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) first (desktop install recommended, since it includes RViz).

You'll also need **Gazebo Fortress**, which is the 3D simulator used for the cave environment. Install it by following [these instructions](https://gazebosim.org/docs/fortress/install_ubuntu/).

### ROS package dependencies

Install the additional ROS packages this project depends on (the Gazebo↔ROS2 bridge, localisation, SLAM and navigation stacks, and xacro):

```bash
sudo apt update

sudo apt install ros-humble-ros-ign-bridge ros-humble-ros-ign-gazebo

sudo apt install ros-humble-robot-localization

sudo apt install ros-humble-slam-toolbox ros-humble-navigation2 ros-humble-nav2-bringup

sudo apt install ros-humble-xacro
```

`cave_explorer.py` also uses OpenCV via `cv_bridge` for computer vision, which is included in a standard ROS2 desktop install.

The [Mission Control GUI](#mission-control-gui) needs PyQt5:

```bash
sudo apt install python3-pyqt5
```

## Building

You'll need a colcon workspace to build into — if you don't already have one from previous coursework, create one:

```bash
mkdir -p ~/ros2_ws/src
```

Symlink the `cave_explorer` package into your workspace's `src/` directory (the commands below assume this repo is checked out at `~/git/SpaceRoboticsA3` and your workspace is at `~/ros2_ws`; adjust the paths if yours differ), then build:

```bash
ln -s ~/git/SpaceRoboticsA3/src/cave_explorer ~/ros2_ws/src/cave_explorer

cd ~/ros2_ws
colcon build --symlink-install --packages-select cave_explorer
source install/setup.bash
```

`--symlink-install` means Python source changes are picked up without rebuilding — you'll only need to re-run `colcon build` after adding new files, changing `setup.py`/`package.xml`, or editing non-Python files like launch or config files.

Remember to `source ~/ros2_ws/install/setup.bash` in every new terminal you use to run these packages (or add it to your `~/.bashrc`).

This should build without errors. If you see any errors, resolve them before proceeding (double check every dependency above installed correctly, and that you're using ROS2 Humble).

## Running the simulation

### Quick start: one launch file

```bash
ros2 launch cave_explorer cave_explorer_all.launch.py
```

This brings up everything at once — Gazebo/RViz, SLAM + Nav2, and the autonomy node — in a single terminal. It just includes the three launch files below, so it accepts all of their arguments too (e.g. `world:=`, `odom_mode:=`, `print_feedback:=`, `dataset_dir:=` — see below). Use this unless you're actively iterating on your own code (see the tip at the end of this section).

### Or: three separate launch files

There are three launch files, meant to be run together, **one in each of three separate terminals** (each sourced with `source ~/ros2_ws/install/setup.bash` first). Launch them in this order, waiting for each to finish starting up before launching the next.

**1. Start the simulator and visualisation:**

```bash
ros2 launch cave_explorer cave_explorer_startup.launch.py
```

This starts Gazebo (loaded with the Mars cave world and the rover), spawns the robot, bridges its sensor/control topics into ROS2, and opens RViz. This can take a minute or so, especially the first time. Useful arguments:

- `world:=mars_cave.sdf` or `world:=mars_surface.sdf` — which world to load (default: `mars_cave.sdf`).
- `odom_mode:=robot_localization` or `odom_mode:=gazebo` — how the `odom` → `base_link` transform is published (default: `robot_localization`, which fuses odometry and IMU via an EKF).
- `depth_pointcloud:=True` — also bridge the RGB-D camera's depth pointcloud (default: `False`).

**2. Start SLAM and navigation:**

```bash
ros2 launch cave_explorer cave_explorer_navigation.launch.py
```

This starts `slam_toolbox` (builds a map from the laser scan as the robot moves) and the Nav2 path-planning stack. RViz should now show the map building up.

You can manually test navigation at this point: click **2D Goal Pose** in RViz, then click a point on the map, and the robot should plan a path and drive there while avoiding obstacles.

**3. Start the autonomy node:**

```bash
ros2 launch cave_explorer cave_explorer_autonomy.launch.py
```

This runs the `cave_explorer` node (`cave_explorer/cave_explorer.py`), which contains the project's decision-making and computer-vision logic. Out of the box it detects stop signs in the camera image (as a placeholder for real artefact detection), then drives to a hardcoded location, returns home, and starts picking random goals. Pass `print_feedback:=True` to log Nav2 navigation feedback (distance remaining) to the terminal.

**Tip:** while iterating on your own code, running these three separately lets you stop and restart the second and third launch files without restarting Gazebo, which is the slowest part to start up — the combined `cave_explorer_all.launch.py` doesn't offer that.

### Collecting a dataset (Perception 1)

The autonomy node can save camera frames to disk for building a training/test dataset of artefacts. It's off by default; enable it with the `dataset_dir` launch argument:

```bash
ros2 launch cave_explorer cave_explorer_autonomy.launch.py dataset_dir:=/home/$USER/cave_dataset
```

This saves a frame at most once every `dataset_save_period` seconds (default `2.0`) to `dataset_dir` as `frame_<timestamp>.png`. You can also save the current frame on demand at any time (e.g. while lining up a good shot of an artefact with `teleop_twist_keyboard`) by calling:

```bash
ros2 service call /save_dataset_image std_srvs/srv/Trigger {}
```

Drive the robot around (random walk/goals, waypoints, or teleop — see Perception 1 in the project brief) while this is running, then sort the saved images into per-artefact-type folders afterwards.

### Autonomous exploration and artefact inspection (Planning 1–3)

Once all three launch files are running, the robot is ready to explore and inspect artefacts fully autonomously — no manual goals needed — but it waits for an explicit start trigger rather than beginning the instant the autonomy node launches. Start it either from the [GUI](#mission-control-gui) ("Start Exploring" button) or directly:

```bash
ros2 service call /start_mission std_srvs/srv/Trigger {}
```

In RViz:

- **`frontier_markers`** (Planning 1) — yellow points are candidate exploration frontiers (free cells bordering unknown space, clustered), the red sphere is the one currently chosen. The robot heads there, then re-evaluates once it arrives (or the goal fails).
- **`artifact_inspection_status_markers`** (Planning 3) — a small flag above each inspectable artefact: **red** = not yet inspected, **green** = successfully inspected, **orange** = abandoned after repeated failed approach attempts.

Only `blue_cube`, `white_sphere`, and `green_crystals` (`INSPECTION_ARTIFACT_LABELS` in `cave_explorer.py`) trigger close-range inspection — these were judged the most visually distinct of Perception 2's colour profiles. When one is detected, the robot pauses exploring, navigates to a standoff viewpoint (`INSPECTION_STANDOFF_DISTANCE_M`, default 2 m) facing it, then resumes exploring. A failed approach is retried once (`INSPECTION_MAX_RETRIES`) before being abandoned.

### Mission Control GUI

A standalone PyQt5 dashboard (`cave_explorer/gui.py`), separate from RViz, for monitoring a run and starting it:

```bash
ros2 run cave_explorer gui
```

Run it alongside the three launch files (its own terminal, after `cave_explorer_autonomy.launch.py`). It shows:

- **Mission status** — current mode, elapsed time, artefact counts, and average speed.
- **Map** — the real occupancy grid (same data as RViz's Map display, subscribed to directly - not routed through the status feed), with the robot's current position, the active goal, and every confirmed artefact overlaid as a coloured pin.
- **Artefact inventory** — every confirmed artefact (`ARTIFACT_CONFIRMATION_OBSERVATIONS`+ sightings), with its label, position, and status (colour-coded the same as the RViz inspection-status markers: green = visited, orange = abandoned, red = pending, grey = detected-but-not-inspectable).
- **Event timeline** — a scrolling log of discoveries, inspections, abandons, and the final mission summary.

Controls:

- **Start Exploring** — calls the `start_mission` service (see above).
- **Pause / Resume** — stops the robot in place and halts further decision-making (`pause_mission`/`resume_mission` services); re-click to resume. Note: pausing mid-`RETURN_HOME` deliberately lets that final leg finish rather than cancelling it, to avoid a cancelled-return being mistaken for a genuine failure on resume.
- **Force Return Home** — skips remaining exploration and heads home immediately (`force_return_home` service), finishing any artefact already confirmed as a target first rather than abandoning it mid-approach.
- **Export Summary** — writes the current status + full artefact inventory to a text file you choose, for pasting into the project report.
- **Manual Override** (D-pad) — drive the robot directly via `/cmd_vel`. Only ever enabled while the mission is paused, not yet started, or already complete - **never** while `main_loop()` is actively driving, since autonomous and manual `cmd_vel` commands fighting each other would cause erratic motion. The buttons are disabled automatically outside those states, and switching back out of them forces an immediate zero-velocity stop.

All of this is driven by one JSON-encoded topic, `gui_status`, published once a second by `cave_explorer_node` (see `publish_gui_status()`) - a custom message type wasn't worth the `.msg` package/rebuild for what's purely a GUI convenience. Any other tool (or a second GUI instance) can read the same topic.

## Development notes

- The `cave_explorer` console script is registered as an entry point in [`setup.py`](src/cave_explorer/setup.py) and maps to `main()` in `cave_explorer/cave_explorer.py`.
- You're free to edit any part of the template, split the code across multiple files, and add new nodes or launch files — wire any new nodes into `cave_explorer_autonomy.launch.py`, or create additional launch files, as needed.
- After adding new Python files, config, or launch files, re-run `colcon build --symlink-install --packages-select cave_explorer` (with `--symlink-install`, existing tracked files update automatically, but new files need a rebuild to be picked up) and re-source `install/setup.bash`.
