#!/usr/bin/env python3
"""
Tests that a science tour leg is given long enough to actually be driven.

Why this exists: the goal watchdog in main_loop() has a flat ceiling (GOAL_TIMEOUT_S = 45s)
and a progress-stall trigger (GOAL_PROGRESS_STALL_S = 40s), both tuned for frontier goals.
Tour legs are the longest goals the mission sends - the route is a closed loop over the whole
explored cave - so the flat ceiling expires mid-leg and the stop is recorded as 'missed' even
though Nav2 was driving it correctly the whole time.

Observed live: the tour was triggered with the rover at (35.8, 8.9) and stop 1 is base at
(0, 0), 36m away. 36m inside 45s needs 0.8m/s sustained with no turning, so base was abandoned
at 46s, and the next four stops went the same way at 42-44s each - 0 of 12 stops reached, with
the rover never getting near any of them. RETURN_HOME had already been given a distance-scaled
timeout and an exemption from the stall check for exactly this reason (its comment records a
~31m trip home cut off 16m short); SCIENCE_TOUR had longer legs and neither.

What's checked here is the scaling arithmetic and the watchdog exemption, not Nav2 - the point
is that a long leg gets a proportionate budget and a short one still gets the normal floor.

Run with:  python -m pytest src/cave_explorer/test/test_tour_goal_timeout.py -v
"""

import math
import types

import pytest

cave_explorer = pytest.importorskip(
    'cave_explorer.cave_explorer',
    reason='needs a sourced ROS 2 environment (rclpy, cv_bridge)')


class FakePose:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.theta = 0.0


class FakeLogger:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)

    def warn(self, msg):
        self.lines.append(msg)


def make_node(robot_xy, stops, stop_index=0):
    """
    A stand-in carrying only what planner_science_tour() reads: the tour, the stop index, the
    robot pose and a logger. The goal send is captured instead of being made, so the timeout
    the planner chose can be asserted on without a ROS graph or an action server.
    """

    node = types.SimpleNamespace()
    node.tour_ = {
        'stops': stops,
        'optimised_order': list(range(len(stops))),
    }
    node.tour_stop_index_ = stop_index
    node.sent = []
    node.get_pose_2d = lambda: FakePose(*robot_xy) if robot_xy is not None else None
    node.get_logger = lambda: node._logger
    node._logger = FakeLogger()
    node.planner_go_to_pose2d = lambda pose2d, timeout_s=cave_explorer.GOAL_TIMEOUT_S: \
        node.sent.append({'pose': pose2d, 'timeout_s': timeout_s})
    node.planner_science_tour = types.MethodType(
        cave_explorer.CaveExplorer.planner_science_tour, node)
    return node


def stop(label, x, y, kind='artifact'):
    return {'label': label, 'kind': kind, 'x': x, 'y': y, 'status': 'pending'}


def test_the_leg_that_broke_it_live_now_gets_a_realistic_budget():
    """
    The exact geometry from the failing run: rover at (35.8, 8.9), stop 1 is base at (0, 0).
    Under the flat ceiling this was cancelled at 46s having covered a fraction of the distance.
    """

    node = make_node((35.8, 8.9), [stop('base', 0.0, 0.0, 'base'),
                                   stop('green_crystals #0', 5.3, 1.6)])
    node.planner_science_tour()

    assert len(node.sent) == 1
    timeout_s = node.sent[0]['timeout_s']
    distance_m = math.hypot(35.8, 8.9)

    assert timeout_s > cave_explorer.GOAL_TIMEOUT_S, (
        'the leg that failed live must now get more than the flat ceiling that failed it')
    assert timeout_s == pytest.approx(distance_m / cave_explorer.TOUR_MIN_SPEED_MPS)
    # The concrete number the rover needs: ~36.8m at 0.3m/s is just over two minutes.
    assert 115.0 < timeout_s < 130.0


def test_a_short_leg_still_gets_the_normal_floor():
    """
    Scaling must never make a short leg's budget *smaller* than the flat ceiling - a 2m
    reposition between adjacent stops would otherwise get under 7 seconds.
    """

    node = make_node((5.0, 1.6), [stop('green_crystals #0', 5.3, 1.6),
                                  stop('green_alien #1', 8.4, 0.7)])
    node.planner_science_tour()

    assert node.sent[0]['timeout_s'] == cave_explorer.GOAL_TIMEOUT_S


def test_budget_is_measured_from_the_rover_not_the_previous_stop():
    """
    The first leg of the tour starts from wherever exploration left the rover, which is not the
    stop the route has it departing from. Measuring the leg along the planned route instead of
    from the live pose is what under-budgets that first leg - the worst one.
    """

    # Route says base -> green_crystals #0 (a 5.5m hop), but the rover is 36m away at the far
    # end of the cave. The budget has to reflect where it actually is.
    node = make_node((35.8, 8.9), [stop('base', 0.0, 0.0, 'base'),
                                   stop('green_crystals #0', 5.3, 1.6)])
    node.planner_science_tour()

    along_route_m = math.hypot(5.3, 1.6)
    assert node.sent[0]['timeout_s'] > along_route_m / cave_explorer.TOUR_MIN_SPEED_MPS


def test_falls_back_to_the_flat_ceiling_without_a_pose():
    """No TF yet is not a reason to skip sending the goal - it just can't be scaled."""

    node = make_node(None, [stop('base', 0.0, 0.0, 'base'),
                            stop('green_crystals #0', 5.3, 1.6)])
    node.planner_science_tour()

    assert node.sent[0]['timeout_s'] == cave_explorer.GOAL_TIMEOUT_S


def test_the_rover_is_pointed_at_the_following_stop_on_arrival():
    """
    Unchanged behaviour, asserted so the timeout work can't quietly break it: the goal heading
    faces the *next* stop, so the rover is already lined up for the following leg.
    """

    node = make_node((0.0, 0.0), [stop('base', 0.0, 0.0, 'base'),
                                  stop('north', 0.0, 10.0)])
    node.planner_science_tour()

    assert node.sent[0]['pose'].theta == pytest.approx(math.pi / 2)


def test_tour_speed_assumption_is_conservative():
    """
    The scaling is only safe if the assumed speed is below what the rover can really average.
    max_vel_x in nav2_params.yaml is 1.0 m/s; assuming a third of that leaves room for turns,
    narrow passages and the odd recovery cycle without falsely running out of patience.
    """

    assert 0.0 < cave_explorer.TOUR_MIN_SPEED_MPS <= 0.35


def test_science_tour_is_exempt_from_the_progress_stall_trigger():
    """
    The flat ceiling is only half the problem. Straight-line distance to a tour stop plateaus
    whenever the rover rounds a chamber or backs out of a dead end, and GOAL_PROGRESS_STALL_S
    reads that as a stall. RETURN_HOME was already exempted for the same reason; this asserts
    the exemption list in main_loop's watchdog covers SCIENCE_TOUR too.
    """

    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(cave_explorer.CaveExplorer.main_loop))

    # Find the `elif ... and stalled_s > GOAL_PROGRESS_STALL_S:` branch and read the planner
    # types named in that same condition, rather than pattern-matching the source text - the
    # condition spans two lines, and a line-based check silently passes when it's reformatted.
    tree = ast.parse(source)
    exempt = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if any(isinstance(n, ast.Name) and n.id == 'GOAL_PROGRESS_STALL_S'
               for n in ast.walk(node.test)):
            exempt = {n.attr for n in ast.walk(node.test) if isinstance(n, ast.Attribute)}
            break

    assert exempt is not None, 'could not find the progress-stall trigger in main_loop'
    for planner in ('RETURN_HOME', 'SCIENCE_TOUR'):
        assert planner in exempt, (
            f'{planner} must be exempt from the progress-stall trigger - its goals are long '
            f'enough that straight-line distance plateaus without the goal being stuck')
