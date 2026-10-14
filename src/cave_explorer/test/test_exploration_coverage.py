"""
Planning 1: tests for the exploration-completeness test that decides when the science tour
starts.

The behaviour being protected here is the one that broke the demo: the rover set off on its
science tour with roughly a third of the cave still unmapped, because the old test treated
"frontier goal selection came back empty" as "the cave is mapped". Those are not the same
claim, and the gap between them is a routine 180-second revisit suppression.

Two things are therefore worth asserting, and they pull in opposite directions:

  - the test must not fire early, whatever goal selection happens to be doing; and
  - it must still fire, on a map that genuinely is finished, or the mission never ends.

Most of these tests build small occupancy grids by hand rather than going near a running node,
which is why measure_exploration_coverage() is a pure function of its arrays.
"""

import ast
import re
import types
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from cave_explorer.cave_explorer import (
    EXPLORATION_MAP_GROWTH_CELLS,
    EXPLORATION_RESIDUAL_FRONTIER_CELLS,
    EXPLORATION_SETTLE_WINDOW_S,
    FRONTIER_REVISIT_WINDOW_S,
    CaveExplorer,
    measure_exploration_coverage,
)

FREE = 0
OCCUPIED = 100
UNKNOWN = -1
RESOLUTION_M = 0.3


# ---------------------------------------------------------------------------------------------
# Test doubles. The node is never constructed; update_exploration_coverage() is bound to a
# namespace carrying only the attributes it actually touches, so these tests stay independent
# of rclpy and of the rest of CaveExplorer's 3000-line constructor.
# ---------------------------------------------------------------------------------------------

class FakeDuration:
    def __init__(self, nanoseconds):
        self.nanoseconds = nanoseconds


class FakeTime:
    """
    Stands in for an rclpy Time. Subtraction has to be defined on the CLASS - Python looks up
    __sub__ on the type, never on the instance, so a SimpleNamespace carrying a __sub__
    attribute is not subtractable.
    """

    def __init__(self, t_s):
        self.t_s = t_s

    def __sub__(self, other):
        return FakeDuration(int((self.t_s - other.t_s) * 1e9))


class FakeClock:
    """A clock the test drives by hand, so settle windows don't cost real wall-clock seconds."""

    def __init__(self):
        self.t_s = 0.0

    def now(self):
        return FakeTime(self.t_s)

    def advance(self, seconds):
        self.t_s += seconds


def make_assessor():
    """A minimal stand-in for the node, with update_exploration_coverage() bound to it."""

    clock = FakeClock()
    logged = []
    node = types.SimpleNamespace(
        coverage_=None,
        known_cell_history_=deque(),
        residual_clear_since_=None,
        exploration_complete_=False,
        get_clock=lambda: clock,
        get_logger=lambda: types.SimpleNamespace(
            info=logged.append, warn=logged.append, error=logged.append),
    )
    node.update_exploration_coverage = types.MethodType(
        CaveExplorer.update_exploration_coverage, node)
    return node, clock, logged


def counts(reachable_frontier_cells, known_cells, reachable_free_cells=500):
    """A coverage measurement, as measure_exploration_coverage() would return it."""

    return {
        'known_cells': known_cells,
        'reachable_free_cells': reachable_free_cells,
        'reachable_frontier_cells': reachable_frontier_cells,
        'mapped_area_m2': reachable_free_cells * RESOLUTION_M ** 2,
        'unseen_frontier_area_m2': reachable_frontier_cells * RESOLUTION_M ** 2,
    }


def settle(node, clock, reachable_frontier_cells=0, known_cells=5000, step_s=20.0,
           duration_s=None):
    """Tick the assessor over a span of simulated time with a steady measurement."""

    if duration_s is None:
        duration_s = EXPLORATION_SETTLE_WINDOW_S + step_s
    elapsed = 0.0
    node.update_exploration_coverage(counts(reachable_frontier_cells, known_cells))
    while elapsed < duration_s:
        clock.advance(step_s)
        elapsed += step_s
        node.update_exploration_coverage(counts(reachable_frontier_cells, known_cells))


# ---------------------------------------------------------------------------------------------
# What "explored" means: reachable unknown, not total unknown.
# ---------------------------------------------------------------------------------------------

def test_unknown_space_sealed_behind_a_wall_does_not_count_as_unexplored():
    """
    The criterion has to be reachable unknown space, not unknown space. A cave is a finite
    pocket inside an infinite unknown grid, so "no unknown cells left" is unsatisfiable and a
    test written that way would never let the mission finish. Here a fully-seen corridor sits
    next to a sealed unknown region: the frontier cells along the wall exist, but no path
    reaches them, so nothing is outstanding.
    """

    grid = np.full((9, 9), UNKNOWN, dtype=np.int8)
    grid[4, 1:5] = FREE            # the corridor the robot has driven and seen all of
    grid[3, 1:5] = OCCUPIED        # walls above and below, so the corridor is closed
    grid[5, 1:5] = OCCUPIED
    grid[4, 0] = OCCUPIED
    grid[4, 5] = OCCUPIED
    # Everything from column 6 rightwards stays UNKNOWN - the rest of the planet.

    frontier_mask = np.zeros_like(grid, dtype=bool)
    frontier_mask[4, 4] = True     # a free cell that does border unknown, but behind a wall

    # BFS from the corridor: it reaches the corridor cells and nothing else.
    path_distance_grid = np.full(grid.shape, -1, dtype=np.int32)
    path_distance_grid[4, 1:5] = [0, 1, 2, 3]

    result = measure_exploration_coverage(grid, frontier_mask, path_distance_grid, RESOLUTION_M)

    # The frontier cell is reachable in this hand-built case, so it is counted - the point of
    # the test is the contrast below, where sealing it off removes it from the total.
    assert result['reachable_frontier_cells'] == 1

    path_distance_grid[4, 4] = -1  # now the frontier cell itself is cut off
    sealed = measure_exploration_coverage(grid, frontier_mask, path_distance_grid, RESOLUTION_M)
    assert sealed['reachable_frontier_cells'] == 0, \
        'unknown space the robot cannot reach must not keep exploration alive forever'
    assert np.count_nonzero(grid == UNKNOWN) > 0, \
        'the grid still holds unknown cells - which is exactly why counting them is wrong'


def test_reachable_unseen_space_is_counted_and_converted_to_area():
    """The headline figure, and the m^2 conversion the GUI and run summary report."""

    grid = np.full((5, 10), UNKNOWN, dtype=np.int8)
    grid[2, 0:6] = FREE
    frontier_mask = np.zeros_like(grid, dtype=bool)
    frontier_mask[2, 4] = True
    frontier_mask[2, 5] = True

    path_distance_grid = np.full(grid.shape, -1, dtype=np.int32)
    path_distance_grid[2, 0:6] = range(6)

    result = measure_exploration_coverage(grid, frontier_mask, path_distance_grid, RESOLUTION_M)

    assert result['reachable_frontier_cells'] == 2
    assert result['unseen_frontier_area_m2'] == pytest.approx(2 * RESOLUTION_M ** 2)
    assert result['reachable_free_cells'] == 6
    assert result['mapped_area_m2'] == pytest.approx(6 * RESOLUTION_M ** 2)


def test_occupied_cells_are_known_but_not_free():
    """
    known_cells drives the map-growth test, so it has to include walls: discovering a new
    chamber's wall is the map growing just as much as discovering its floor is.
    """

    grid = np.array([[FREE, OCCUPIED, UNKNOWN]], dtype=np.int8)
    frontier_mask = np.zeros_like(grid, dtype=bool)
    path_distance_grid = np.array([[0, -1, -1]], dtype=np.int32)

    result = measure_exploration_coverage(grid, frontier_mask, path_distance_grid, RESOLUTION_M)

    assert result['known_cells'] == 2, 'free and occupied are both known'
    assert result['reachable_free_cells'] == 1, 'only the free cell is drivable'


# ---------------------------------------------------------------------------------------------
# The regression: the tour must not start on a partial map.
# ---------------------------------------------------------------------------------------------

def test_exploration_is_not_complete_while_reachable_unknown_space_remains():
    """However long we wait, outstanding reachable unknown space blocks completion."""

    node, clock, _ = make_assessor()
    settle(node, clock,
           reachable_frontier_cells=EXPLORATION_RESIDUAL_FRONTIER_CELLS + 1,
           duration_s=EXPLORATION_SETTLE_WINDOW_S * 3)

    assert not node.exploration_complete_
    assert node.coverage_['residual_clear'] is False


def test_the_bug_that_started_the_tour_early_cannot_recur():
    """
    The specific failure from the demo runs. The old test fired after five consecutive ticks
    with no frontier *goal*, which the revisit window alone can cause: every candidate within
    FRONTIER_REVISIT_RADIUS_M of a recent goal is suppressed, so a rover working a corridor
    sees nothing to drive to while real cave sits just out of range. Run 5's streak reached 3
    of 5 while the widest chamber was still growing from 6.4m to 11.9m.

    Those ticks happen fast - well inside FRONTIER_REVISIT_WINDOW_S. So: simulate that burst,
    with unknown space still outstanding the whole time, and require no completion.
    """

    node, clock, _ = make_assessor()
    for _ in range(12):
        clock.advance(2.0)  # 24s of ticks in total, far inside the 180s revisit window
        node.update_exploration_coverage(counts(reachable_frontier_cells=900, known_cells=4000))

    assert not node.exploration_complete_, \
        'a burst of suppressed-frontier ticks must not be mistaken for a mapped cave'
    assert clock.t_s < FRONTIER_REVISIT_WINDOW_S, \
        'the point of the test is that this all happens inside the revisit window'


def test_a_still_growing_map_blocks_completion_even_with_no_frontier_cells():
    """
    The backstop. If SLAM is still revealing cells then exploration is not finished, whatever
    the frontier count says - this is what catches a frontier figure that is wrong for a reason
    nobody anticipated.
    """

    node, clock, _ = make_assessor()
    known = 1000
    node.update_exploration_coverage(counts(0, known))
    for _ in range(20):
        clock.advance(20.0)
        known += EXPLORATION_MAP_GROWTH_CELLS  # a whole threshold's worth every tick
        node.update_exploration_coverage(counts(0, known))

    assert not node.exploration_complete_
    assert node.coverage_['map_settled'] is False


def test_completion_requires_the_full_settle_window():
    """
    Both conditions can be satisfied instantaneously and still mean nothing - they have to
    hold. Checked just short of the window, then just past it.
    """

    node, clock, _ = make_assessor()
    settle(node, clock, duration_s=EXPLORATION_SETTLE_WINDOW_S - 40.0, step_s=20.0)
    assert not node.exploration_complete_, 'fired before the settle window elapsed'

    clock.advance(60.0)
    node.update_exploration_coverage(counts(0, 5000))
    assert node.exploration_complete_, 'never fired, so the mission would never end'


def test_a_frontier_reappearing_restarts_the_settle_clock():
    """
    Partial progress towards completion must not be bankable. A frontier coming back means the
    quiet period was not evidence of anything, so the window starts again from zero rather
    than resuming where it left off.
    """

    node, clock, _ = make_assessor()
    settle(node, clock, duration_s=EXPLORATION_SETTLE_WINDOW_S - 40.0, step_s=20.0)
    assert not node.exploration_complete_

    # A new chamber opens up.
    clock.advance(20.0)
    node.update_exploration_coverage(counts(EXPLORATION_RESIDUAL_FRONTIER_CELLS + 500, 5000))
    assert node.residual_clear_since_ is None, 'the clock should have been reset, not paused'

    # It is explored again, but only for part of a window.
    settle(node, clock, duration_s=EXPLORATION_SETTLE_WINDOW_S - 40.0, step_s=20.0)
    assert not node.exploration_complete_, \
        'the pre-interruption quiet time must not count towards the new window'


# ---------------------------------------------------------------------------------------------
# The other direction: it still has to finish.
# ---------------------------------------------------------------------------------------------

def test_a_genuinely_finished_map_does_complete():
    node, clock, logged = make_assessor()
    settle(node, clock, reachable_frontier_cells=0, known_cells=8000)

    assert node.exploration_complete_
    assert any('EXPLORATION COMPLETE' in str(line) for line in logged), \
        'the operator needs to be told why the tour is about to start'


def test_residual_sensor_shadow_slivers_do_not_block_completion():
    """
    The threshold is not 0 on purpose. The lidar is a single horizontal ray at z = 1.0m, so it
    grazes rock edges and leaves a few cells that cannot be filled in from any angle. If those
    blocked completion the rover would explore forever and never run its tour.
    """

    node, clock = make_assessor()[:2]
    settle(node, clock, reachable_frontier_cells=EXPLORATION_RESIDUAL_FRONTIER_CELLS)

    assert node.exploration_complete_, \
        'a handful of unfillable sliver cells must not deadlock the mission'


def test_completion_latches_and_is_not_re_announced():
    node, clock, logged = make_assessor()
    settle(node, clock)
    assert node.exploration_complete_
    announcements = sum(1 for line in logged if 'EXPLORATION COMPLETE' in str(line))

    settle(node, clock, duration_s=EXPLORATION_SETTLE_WINDOW_S)
    assert node.exploration_complete_, 'the latch must be sticky'
    assert sum(1 for line in logged if 'EXPLORATION COMPLETE' in str(line)) == announcements, \
        'announced more than once'


# ---------------------------------------------------------------------------------------------
# Degenerate input.
# ---------------------------------------------------------------------------------------------

def test_a_failed_bfs_is_not_mistaken_for_a_finished_map():
    """
    compute_path_distance_grid() returns an all -1 grid when the robot's own cell doesn't read
    as free, which SLAM does transiently. Nothing is then reachable, so the frontier count is 0
    and the residual test would pass for the worst possible reason: complete, with nothing
    mapped. Reachable free space being empty has to invalidate the measurement.
    """

    node, clock = make_assessor()[:2]
    settle(node, clock, reachable_frontier_cells=0, duration_s=EXPLORATION_SETTLE_WINDOW_S * 2)
    assert node.exploration_complete_, 'baseline: this measurement does normally complete'

    node, clock = make_assessor()[:2]
    elapsed = 0.0
    node.update_exploration_coverage(counts(0, 5000, reachable_free_cells=0))
    while elapsed < EXPLORATION_SETTLE_WINDOW_S * 2:
        clock.advance(20.0)
        elapsed += 20.0
        node.update_exploration_coverage(counts(0, 5000, reachable_free_cells=0))

    assert not node.exploration_complete_, \
        'an unreachable robot means the BFS failed, not that the cave is mapped'


def test_the_history_keeps_a_baseline_old_enough_to_measure_growth_against():
    """
    Growth has to be measured over at least a full settle window. The deque prunes from the
    front, so it must keep one sample that is still older than the window rather than trimming
    down to only recent ones - otherwise a slowly-growing map looks settled.
    """

    node, clock = make_assessor()[:2]
    known = 1000
    for _ in range(60):
        clock.advance(10.0)
        known += 10
        node.update_exploration_coverage(counts(0, known))

    oldest_time = node.known_cell_history_[0][0]
    span_s = (clock.now() - oldest_time).nanoseconds / 1e9
    assert span_s >= EXPLORATION_SETTLE_WINDOW_S, \
        f'baseline sample is only {span_s:.0f}s old, less than the settle window'
    assert len(node.known_cell_history_) < 60, 'history grows without bound'


# ---------------------------------------------------------------------------------------------
# Consistency with the real source, so the production path can't drift away from these tests.
# ---------------------------------------------------------------------------------------------

def source_text():
    path = Path(__file__).resolve().parents[1] / 'cave_explorer' / 'cave_explorer.py'
    return path.read_text(encoding='utf-8')


def test_frontier_selection_still_feeds_the_coverage_measurement():
    """
    The measurement is driven from choose_frontier_goal() so it reuses that function's BFS
    instead of running a second one - a full-map BFS per tick is exactly the kind of CPU load
    that caused Nav2's action handshakes to time out. If that call is ever moved, this fails.
    """

    source = source_text()
    assert re.search(r'self\.update_exploration_coverage\(measure_exploration_coverage\(',
                     source), 'choose_frontier_goal() no longer updates the coverage measurement'


def test_the_coverage_measurement_reads_the_unfiltered_frontier_mask():
    """
    It must be passed 'frontier_mask' itself - the raw find_frontier_cells() output - not a
    filtered, clustered or otherwise derived version. The cluster-size and revisit filters
    describe what is worth driving to, not what is still unseen, and mistaking one for the
    other is the original bug.

    Checked by walking the AST and requiring each argument to be a bare name, rather than by
    searching the source text. A substring check is not good enough here, and the difference is
    not academic: np.zeros_like(frontier_mask) contains the string 'frontier_mask' while
    reporting zero unseen cells on every tick, which would declare the cave fully explored
    about a second into the mission. That mutation passed a text-based version of this test.
    """

    source = source_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == 'measure_exploration_coverage']

    assert len(calls) == 1, \
        f'expected exactly one measure_exploration_coverage() call site, found {len(calls)}'

    args = calls[0].args
    assert len(args) == 4, 'measure_exploration_coverage() takes grid, mask, bfs, resolution'

    mask_arg = args[1]
    assert isinstance(mask_arg, ast.Name), (
        f'the frontier mask argument is a {type(mask_arg).__name__} expression, not a plain '
        f'variable - anything derived or filtered here breaks the completeness measurement')
    assert mask_arg.id == 'frontier_mask', \
        f'the measurement reads {mask_arg.id!r} rather than the unfiltered frontier_mask'

    grid_arg = args[0]
    assert isinstance(grid_arg, ast.Attribute) and grid_arg.attr == 'map_grid_', \
        'the measurement must read the raw SLAM grid'


def test_no_frontier_goal_no_longer_declares_exploration_complete():
    """
    The heart of the fix: planner_explore_frontier()'s empty-candidate branch must not set
    exploration_complete_. That single assignment is what sent the rover touring a half-mapped
    cave, and it would be an easy thing to reintroduce while debugging something else.
    """

    source = source_text()
    start = source.index('def planner_explore_frontier')
    end = source.index('def find_inspection_target')
    body = source[start:end]

    assert 'exploration_complete_ = True' not in body, \
        'planner_explore_frontier() must not decide that exploration is complete'
    assert 'update_exploration_coverage' in source, 'the replacement test is missing entirely'


def test_the_settle_window_outlasts_the_revisit_window():
    """
    The settle window has to be longer than FRONTIER_REVISIT_WINDOW_S, or the completeness test
    could again be satisfied entirely inside a period when real frontiers are being suppressed
    and are therefore invisible to it.
    """

    assert EXPLORATION_SETTLE_WINDOW_S > FRONTIER_REVISIT_WINDOW_S, \
        (f'settle window {EXPLORATION_SETTLE_WINDOW_S}s does not outlast the '
         f'{FRONTIER_REVISIT_WINDOW_S}s revisit window, so suppressed frontiers could still '
         f'be mistaken for a finished map')
