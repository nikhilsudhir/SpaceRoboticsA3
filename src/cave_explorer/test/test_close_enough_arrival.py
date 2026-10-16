#!/usr/bin/env python3
"""
Tests that an inspection or tour goal the rover has effectively reached is accepted.

Why this exists: live, the rover drove all 11.6m to green_crystals #0's standoff, then sat ~1m
short for 30s ("Failed to make progress" twice) and the goal timed out at 45s - it couldn't
complete the final turn to face the artefact beside the crystal spikes. A re-sent goal 1m away
"succeeded" in 1s. Within GOAL_CLOSE_ENOUGH_M and no longer closing in for
GOAL_CLOSE_ENOUGH_STALL_S now counts as reached, for inspection and tour goals only.

Run with:  python -m pytest src/cave_explorer/test/test_close_enough_arrival.py -v
"""

import ast
import inspect
import textwrap
import types

import pytest

cave_explorer = pytest.importorskip(
    'cave_explorer.cave_explorer',
    reason='needs a sourced ROS 2 environment (rclpy, cv_bridge)')

CaveExplorer = cave_explorer.CaveExplorer


def test_accepting_marks_the_goal_reached_and_frees_the_planner():
    logged = []
    node = types.SimpleNamespace(
        goal_progress_distance_=1.1, current_goal_handle_=object(), goal_generation_=4,
        last_goal_succeeded_=False, ready_for_next_goal_=False, goal_sent_time_=object())
    node.log_goal_outcome = lambda text, success=False: logged.append((text, success))

    CaveExplorer.accept_goal_as_reached(node)

    assert node.last_goal_succeeded_ is True
    assert node.ready_for_next_goal_ is True
    assert node.goal_generation_ == 5  # a late Nav2 result for it is ignored as stale
    assert node.current_goal_handle_ is None
    assert logged and logged[0][1] is True and 'close enough' in logged[0][0]


def watchdog_branches():
    source = textwrap.dedent(inspect.getsource(CaveExplorer.main_loop))
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.If) and any(
                isinstance(n, ast.Name) and n.id == 'GOAL_CLOSE_ENOUGH_M' for n in ast.walk(node.test)):
            return node
    return None


def test_the_watchdog_checks_close_enough_before_timing_out():
    branch = watchdog_branches()
    assert branch is not None, 'main_loop no longer checks GOAL_CLOSE_ENOUGH_M'

    planners = {n.attr for n in ast.walk(branch.test) if isinstance(n, ast.Attribute)}
    assert {'INSPECT_ARTIFACT', 'SCIENCE_TOUR'} <= planners
    assert 'EXPLORE_FRONTIER' not in planners, 'exploration goals keep the normal watchdog'

    # The timeout must be the elif behind it, or a stalled-but-arrived goal still times out first
    orelse = branch.orelse[0] if branch.orelse else None
    assert isinstance(orelse, ast.If) and any(
        isinstance(n, ast.Attribute) and n.attr == 'goal_timeout_s_' for n in ast.walk(orelse.test))


def test_the_thresholds_leave_room_for_normal_arrival():
    # Nav2's own goal checker accepts within 1.0m; close-enough must be looser than that, or it
    # adds nothing, and the stall wait shorter than the 40s no-progress watchdog.
    assert 1.0 < cave_explorer.GOAL_CLOSE_ENOUGH_M <= 2.0
    assert cave_explorer.GOAL_CLOSE_ENOUGH_STALL_S < cave_explorer.GOAL_PROGRESS_STALL_S
