#!/usr/bin/env python3
"""
Tests that small frontiers get explored once the big ones run out, without trapping the rover.

Why this exists: FRONTIER_MIN_CLUSTER_SIZE (15 cells, 4.5 m at 0.3 m resolution) dropped every
smaller frontier as noise. Live, one area near the start was never entered in any run: its
mouth showed less than 4.5 m of frontier, so it never became a goal, and the rover random-walked
with "57 reachable frontier cell(s) remain unseen" in the log. Those cells also held the
completeness count above EXPLORATION_RESIDUAL_FRONTIER_CELLS (40), so exploration never
finished and the science tour never ran.

Now clusters down to FRONTIER_FALLBACK_MIN_CLUSTER_SIZE become goals once no full-size one is
left, and a small frontier still open after FRONTIER_MAX_FALLBACK_ATTEMPTS goals is written off -
not chosen again, and not counted against completeness - so an unfillable scrap can't trap it.

Run with:  python -m pytest src/cave_explorer/test/test_small_frontier_fallback.py -v
"""

import types

import numpy as np
import pytest

cave_explorer = pytest.importorskip(
    'cave_explorer.cave_explorer',
    reason='needs a sourced ROS 2 environment (rclpy, cv_bridge)')

CaveExplorer = cave_explorer.CaveExplorer
MIN_SIZE = cave_explorer.FRONTIER_MIN_CLUSTER_SIZE
FALLBACK_MIN = cave_explorer.FRONTIER_FALLBACK_MIN_CLUSTER_SIZE
MAX_ATTEMPTS = cave_explorer.FRONTIER_MAX_FALLBACK_ATTEMPTS

FREE, OCCUPIED, UNKNOWN = 0, 100, -1
RESOLUTION_M = 0.3


def make_room(big_opening=True, small_opening=True):
    """
    A fully-seen square room inside unknown space. Optionally a wide opening in its top wall
    (a full-size frontier) and a narrow one in its right wall (a small, fallback-size frontier).
    """

    grid = np.full((32, 32), UNKNOWN, dtype=np.int8)
    grid[4:27, 4:27] = OCCUPIED
    grid[5:26, 5:26] = FREE
    if big_opening:
        grid[4, 8:24] = FREE
    if small_opening:
        grid[12:17, 26] = FREE
    return grid


class Clock:
    def now(self):
        return 0


def make_node(grid):
    node = types.SimpleNamespace(
        map_grid_=grid,
        map_origin_=(0.0, 0.0),
        map_resolution_=RESOLUTION_M,
        recent_frontier_goals_=[],
        fallback_frontier_attempts_=[],
        artifact_clusters_=[],
        coverage_measurements=[],
        logged=[],
    )
    node.get_clock = lambda: Clock()
    node.get_logger = lambda: types.SimpleNamespace(info=node.logged.append)
    node.update_exploration_coverage = node.coverage_measurements.append
    for name in ('find_frontier_cells', 'cluster_frontier_cells', 'compute_path_distance_grid',
                 'world_to_grid', 'grid_to_world', 'choose_frontier_goal',
                 'record_fallback_frontier_attempt', 'is_clear_of_artifacts'):
        setattr(node, name, types.MethodType(getattr(CaveExplorer, name), node))
    return node


ROBOT = types.SimpleNamespace(x=15.5 * RESOLUTION_M, y=15.5 * RESOLUTION_M, theta=0.0)


def frontier_sizes(node):
    return sorted(len(c) for c in node.cluster_frontier_cells(node.find_frontier_cells()))


def test_the_fixture_has_one_full_size_and_one_small_frontier():
    sizes = frontier_sizes(make_node(make_room()))
    assert len(sizes) == 2
    assert FALLBACK_MIN <= sizes[0] < MIN_SIZE <= sizes[1]


def test_a_full_size_frontier_is_still_preferred_over_a_small_one():
    chosen, _ = make_node(make_room()).choose_frontier_goal(ROBOT)

    assert chosen is not None and not chosen['small']


def test_a_small_frontier_is_chosen_once_no_full_size_one_is_left():
    # Before the fix this returned None and the rover random-walked instead
    chosen, _ = make_node(make_room(big_opening=False)).choose_frontier_goal(ROBOT)

    assert chosen is not None and chosen['small']
    assert FALLBACK_MIN <= chosen['size'] < MIN_SIZE


def test_a_small_frontier_still_open_after_repeated_attempts_is_written_off():
    node = make_node(make_room(big_opening=False))
    chosen, _ = node.choose_frontier_goal(ROBOT)
    x, y = chosen['position']

    for attempt in range(MAX_ATTEMPTS):
        node.record_fallback_frontier_attempt(x, y, chosen['size'])
        if attempt < MAX_ATTEMPTS - 1:
            again, _ = node.choose_frontier_goal(ROBOT)
            assert again is not None, 'still worth another try before the limit'

    chosen_after, _ = node.choose_frontier_goal(ROBOT)
    assert chosen_after is None


def test_a_written_off_frontier_no_longer_blocks_exploration_completeness():
    node = make_node(make_room(big_opening=False))
    chosen, _ = node.choose_frontier_goal(ROBOT)
    before = node.coverage_measurements[-1]
    assert before['reachable_frontier_cells'] == chosen['size']

    for _ in range(MAX_ATTEMPTS):
        node.record_fallback_frontier_attempt(*chosen['position'], chosen['size'])
    node.choose_frontier_goal(ROBOT)
    after = node.coverage_measurements[-1]

    assert after['reachable_frontier_cells'] == 0
    assert after['written_off_frontier_cells'] == chosen['size']


def test_writing_off_a_small_frontier_does_not_hide_a_full_size_one_nearby():
    node = make_node(make_room())
    small = [c for c in node.choose_frontier_goal(ROBOT)[1] if c['small']][0]
    for _ in range(MAX_ATTEMPTS):
        node.record_fallback_frontier_attempt(*small['position'], small['size'])

    chosen, _ = node.choose_frontier_goal(ROBOT)
    assert chosen is not None and not chosen['small']


def test_attempts_are_logged_so_the_end_game_is_visible_in_the_run_log():
    node = make_node(make_room(big_opening=False))
    for _ in range(MAX_ATTEMPTS):
        node.record_fallback_frontier_attempt(10.0, 4.5, 5)

    assert 'trying a small 5-cell one' in node.logged[0]
    assert 'written off' in node.logged[-1]


def test_measure_reports_written_off_cells_separately():
    grid = make_room(big_opening=False)
    node = make_node(grid)
    frontier = node.find_frontier_cells()
    bfs = node.compute_path_distance_grid(15, 15)

    plain = cave_explorer.measure_exploration_coverage(grid, frontier, bfs, RESOLUTION_M)
    written_off = cave_explorer.measure_exploration_coverage(
        grid, frontier, bfs, RESOLUTION_M, written_off_mask=frontier.copy())

    assert plain['written_off_frontier_cells'] == 0
    assert written_off['reachable_frontier_cells'] == 0
    assert written_off['written_off_frontier_cells'] == plain['reachable_frontier_cells']


def test_no_frontier_goal_is_sent_on_the_tick_exploration_completes():
    """
    Completeness is measured inside choose_frontier_goal(), so it can latch mid-tick. Live, that
    tick still sent a goal to a 4-cell scrap - a 46s detour after "EXPLORATION COMPLETE".
    """

    sent = []
    node = make_node(make_room(big_opening=False))
    node.exploration_complete_ = False
    node.ready_for_next_goal_ = False
    node.no_frontier_streak_ = 0
    node.get_pose_2d = lambda: ROBOT
    node.publish_frontier_markers = lambda chosen, candidates: None
    node.planner_go_to_pose2d = lambda *args, **kwargs: sent.append(args)
    node.planner_random_walk = lambda: sent.append('random walk')

    def complete_during_measurement(counts):
        node.exploration_complete_ = True
    node.update_exploration_coverage = complete_during_measurement

    CaveExplorer.planner_explore_frontier(node)

    assert sent == []
    assert node.ready_for_next_goal_


def test_a_frontier_inside_a_known_artefact_is_never_chosen():
    """
    Live: a 6-cell frontier 1.3m from the centre of the green crystals at (19.8, 35.8) - inside
    the spikes, where the lidar sees gaps - was chosen, and the rover wedged among the spikes.
    """

    from geometry_msgs.msg import Point
    node = make_node(make_room(big_opening=False))
    chosen, _ = node.choose_frontier_goal(ROBOT)
    assert chosen is not None
    cx, cy = chosen['position']

    node.artifact_clusters_ = [{'id': 1, 'label': 'green_crystals', 'num_observations': 10,
                                'position': Point(x=cx + 0.5, y=cy, z=0.0)}]
    blocked, _ = node.choose_frontier_goal(ROBOT)

    assert blocked is None or blocked['position'] != (cx, cy)
    assert node.coverage_measurements[-1]['written_off_frontier_cells'] > 0
