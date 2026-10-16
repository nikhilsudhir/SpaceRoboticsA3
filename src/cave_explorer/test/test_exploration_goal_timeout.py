#!/usr/bin/env python3
"""
Tests that a far frontier goal is given long enough to actually be driven.

Why this exists: frontier goals used the flat GOAL_TIMEOUT_S (45s) ceiling. Frontiers are chosen
across the whole explored map, so later in a run the best one is routinely 40-50m away by path.
One live run logged 7 frontier goals "timed out in 45s" while the rover was driving normally
(0.8-1.1 m/s average, 36-50m covered). Each cancel sent it toward a fresh frontier, often back
the way it came, so it shuttled along the central corridor: still exploring at 1000s, against a
run that finished exploration and started the science tour by 560s. RETURN_HOME and
SCIENCE_TOUR were already given distance-scaled ceilings for the same reason.

Run with:  python -m pytest src/cave_explorer/test/test_exploration_goal_timeout.py -v
"""

import types

import pytest

cave_explorer = pytest.importorskip(
    'cave_explorer.cave_explorer',
    reason='needs a sourced ROS 2 environment (rclpy, cv_bridge)')

exploration_goal_timeout_s = cave_explorer.exploration_goal_timeout_s
GOAL_TIMEOUT_S = cave_explorer.GOAL_TIMEOUT_S

# (distance covered when cancelled, average speed) for each frontier goal the live run cut off
# while it was still moving at a normal pace
LIVE_CANCELLED_GOALS = [(44.7, 0.99), (45.7, 1.01), (38.8, 0.86), (42.5, 0.94),
                        (49.8, 1.10), (47.7, 1.06), (36.4, 0.81)]


def test_every_goal_the_live_run_cut_off_now_has_room_to_finish():
    # Each had covered at least this much path and was still going, so its frontier was at
    # least this far away. At the speed it was actually doing, it needs this long:
    for covered_m, speed_mps in LIVE_CANCELLED_GOALS:
        assert exploration_goal_timeout_s(covered_m) > covered_m / speed_mps * 1.5


def test_a_near_frontier_still_gets_the_normal_floor():
    assert exploration_goal_timeout_s(0.0) == GOAL_TIMEOUT_S
    assert exploration_goal_timeout_s(10.0) == GOAL_TIMEOUT_S


def test_the_ceiling_grows_with_distance():
    assert exploration_goal_timeout_s(60.0) > exploration_goal_timeout_s(40.0) > GOAL_TIMEOUT_S


def test_speed_assumption_sits_between_the_tour_figure_and_real_driving():
    """
    Below the ~0.8-1.1 m/s the rover really averages on long frontier goals, so a progressing
    goal is not cut short; not as cautious as the tour's figure, since a stuck exploration goal
    is caught by the progress-stall trigger anyway and exploration time is worth protecting.
    """

    assert cave_explorer.TOUR_MIN_SPEED_MPS <= cave_explorer.EXPLORE_MIN_SPEED_MPS <= 0.6


def test_exploration_is_still_cut_off_by_the_progress_stall_trigger():
    """
    The longer ceiling is only safe because a stuck frontier goal is still abandoned after
    GOAL_PROGRESS_STALL_S without progress. Asserts main_loop's stall exemption list doesn't
    include EXPLORE_FRONTIER.
    """

    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(cave_explorer.CaveExplorer.main_loop))
    exempt = None
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.If) and any(
                isinstance(n, ast.Name) and n.id == 'GOAL_PROGRESS_STALL_S'
                for n in ast.walk(node.test)):
            exempt = {n.attr for n in ast.walk(node.test) if isinstance(n, ast.Attribute)}
            break

    assert exempt is not None, 'could not find the progress-stall trigger in main_loop'
    assert 'EXPLORE_FRONTIER' not in exempt


def test_planner_explore_frontier_sends_the_scaled_timeout():
    """
    A stand-in carrying only what planner_explore_frontier() reads, with the goal send captured
    so the timeout it chose can be checked without a ROS graph or an action server.
    """

    sent = {}
    node = types.SimpleNamespace()
    node.get_pose_2d = lambda: types.SimpleNamespace(x=0.0, y=0.0, theta=0.0)
    chosen = {'position': (30.0, 30.0), 'distance': 48.0, 'size': 40, 'small': False}
    node.choose_frontier_goal = lambda pose: (chosen, [chosen])
    node.publish_frontier_markers = lambda chosen, candidates: None
    node.no_frontier_streak_ = 0
    node.exploration_complete_ = False
    node.recent_frontier_goals_ = []
    node.get_clock = lambda: types.SimpleNamespace(now=lambda: 0)
    node.planner_go_to_pose2d = lambda pose2d, timeout_s=GOAL_TIMEOUT_S: sent.update(
        pose=pose2d, timeout_s=timeout_s)

    cave_explorer.CaveExplorer.planner_explore_frontier(node)

    assert sent['timeout_s'] == exploration_goal_timeout_s(48.0)
    assert sent['timeout_s'] > GOAL_TIMEOUT_S
