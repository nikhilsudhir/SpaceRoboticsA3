#!/usr/bin/env python3
"""
Tests for the Advanced 2 (cave geometry) and Advanced 5 (science tour) logic.

Both modules under test are deliberately ROS-free, which is what makes this possible: the
geometry analysis takes an occupancy grid as a numpy array and the route optimisation takes a
cost matrix, so both can be checked against a synthetic cave with a KNOWN answer, in
milliseconds, without launching Gazebo, SLAM or Nav2. Getting the same evidence from the
simulator would mean driving a rover around and eyeballing RViz markers.

Run with:  python -m pytest src/cave_explorer/test/test_advanced_tasks.py -v
(no ROS environment needed - just numpy, OpenCV and pytest)
"""

import itertools

import cv2
import numpy as np
import pytest

from cave_explorer import cave_geometry, science_tour

RESOLUTION = 0.05  # metres per cell, matching the SLAM map
HEIGHT, WIDTH = 300, 600  # 15m x 30m


def to_cells(metres):
    return int(round(metres / RESOLUTION))


def to_metres(cell_index):
    return (cell_index + 0.5) * RESOLUTION


@pytest.fixture(scope='module')
def synthetic_cave():
    """
    A cave with known dimensions, so the analysis can be checked against ground truth:

        - a 6.0m-wide chamber at (5.0, 7.5)
        - a 0.8m-wide corridor from it, spanning x = 8.0 to 14.4   <- the bottleneck
        - a 3.2m-wide chamber at (16.0, 7.5)
        - a 1.2m-wide corridor on from there
        - a 4.4m-wide chamber at (25.0, 4.0)
        - a strip of unexplored cells at the right-hand edge

    Returns (grid, reachable_mask).
    """

    grid = np.full((HEIGHT, WIDTH), 100, dtype=np.int8)  # solid rock
    rows = np.arange(HEIGHT)[:, None]
    cols = np.arange(WIDTH)[None, :]

    def disc(centre_x, centre_y, radius):
        return ((cols - to_cells(centre_x)) ** 2
                + (rows - to_cells(centre_y)) ** 2) <= to_cells(radius) ** 2

    def box(x0, y0, x1, y1):
        return ((cols >= to_cells(x0)) & (cols <= to_cells(x1))
                & (rows >= to_cells(y0)) & (rows <= to_cells(y1)))

    free = disc(5.0, 7.5, 3.0)
    free |= disc(16.0, 7.5, 1.6)
    free |= disc(25.0, 4.0, 2.2)
    free |= box(8.0, 7.1, 14.4, 7.9)    # 0.8m corridor
    free |= box(17.6, 6.0, 23.0, 7.2)   # 1.2m corridor
    free |= box(22.0, 3.0, 23.5, 7.0)

    grid[free] = 0
    grid[:, 580:] = -1  # unexplored

    # Reachable = the connected component containing the big chamber, which is what the node
    # computes with its wavefront BFS from the robot's position.
    _, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
    reachable = labels == labels[to_cells(7.5), to_cells(5.0)]

    return grid, reachable


@pytest.fixture(scope='module')
def analysis(synthetic_cave):
    grid, reachable = synthetic_cave
    return cave_geometry.analyse_cave(grid, RESOLUTION, reachable, max_features=8)


# ---------------------------------------------------------------- Advanced 2

def test_clearance_map_matches_known_chamber_radius(synthetic_cave):
    """The distance transform's peak should be the radius of the largest chamber (3.0m)."""

    grid, _ = synthetic_cave
    clearance_m, free = cave_geometry.compute_clearance_map(grid, RESOLUTION)

    assert clearance_m.max() == pytest.approx(3.0, abs=0.1)
    # Unknown cells count as walls, so clearance there must be zero - a rover can't assume
    # unexplored space is open.
    assert clearance_m[:, 585:].max() == 0.0
    assert not free[:, 585:].any()


def test_widest_accessible_area_is_the_big_chamber(analysis):
    """The headline Advanced 2 result: the widest accessible area, with its size."""

    widest = analysis['chambers'][0]
    x, y = to_metres(widest['col']), to_metres(widest['row'])

    assert x == pytest.approx(5.0, abs=1.0)
    assert y == pytest.approx(7.5, abs=1.0)
    assert widest['width_m'] == pytest.approx(6.0, abs=0.3)


def test_all_three_chambers_found_with_correct_widths(analysis):
    """Each chamber is found once, and measured accurately (not blurred by the smoothing)."""

    found = {(round(to_metres(c['col'])), round(to_metres(c['row']))): c['width_m']
             for c in analysis['chambers']}

    expected = {(5, 8): 6.0, (16, 8): 3.2, (25, 4): 4.4}
    for (x, y), width in expected.items():
        matches = [w for (fx, fy), w in found.items() if abs(fx - x) <= 1 and abs(fy - y) <= 1]
        assert matches, f'no chamber found near ({x}, {y}); found {sorted(found)}'
        assert matches[0] == pytest.approx(width, abs=0.3)


def test_narrowest_passage_is_the_narrow_corridor(analysis):
    """The other headline Advanced 2 result: a narrow passage / bottleneck."""

    narrowest = analysis['bottlenecks'][0]
    x, y = to_metres(narrowest['col']), to_metres(narrowest['row'])

    # The corridor is a uniform 0.8m wide over x in [8.0, 14.4], so clearance along its
    # centreline is flat and any cell on it is equally the narrowest - including the cells at
    # either mouth. Allow a couple of cells past each end.
    assert 7.8 <= x <= 14.6, f'bottleneck at x={x:.1f} is not in the narrow corridor'
    assert y == pytest.approx(7.5, abs=0.5)
    assert narrowest['width_m'] == pytest.approx(0.8, abs=0.2)


def test_bottlenecks_are_on_passages_not_against_walls(analysis):
    """
    Regression test for the bug the ridge condition fixes (see cave_geometry.find_ridge()).

    Without it, every bottleneck came back at the MIN_BOTTLENECK_CLEARANCE_M floor - arbitrary
    cells hugging a wall or sitting in a concave corner, which have low clearance without being
    passages at all. Every real passage in this cave is at least 0.8m wide, so anything
    materially narrower than that is a wall-hugging false positive.
    """

    assert analysis['bottlenecks'], 'no bottlenecks found at all'
    for bottleneck in analysis['bottlenecks']:
        assert bottleneck['width_m'] >= 0.7, \
            f"bottleneck of {bottleneck['width_m']:.2f}m is narrower than any real passage here"


def test_no_features_reported_in_unexplored_space(analysis):
    for feature in analysis['chambers'] + analysis['bottlenecks']:
        assert feature['col'] < 580


def test_unreachable_void_is_not_reported_as_a_chamber():
    """
    A sealed void inside the rock is wide open but the rover can never drive there, and the
    task asks for the widest ACCESSIBLE area - so the reachability mask has to exclude it.
    """

    grid = np.full((HEIGHT, WIDTH), 100, dtype=np.int8)
    rows = np.arange(HEIGHT)[:, None]
    cols = np.arange(WIDTH)[None, :]

    reachable_chamber = (((cols - to_cells(5.0)) ** 2
                          + (rows - to_cells(7.5)) ** 2) <= to_cells(2.0) ** 2)
    # Bigger, but walled off from everything.
    sealed_void = (((cols - to_cells(20.0)) ** 2
                    + (rows - to_cells(7.5)) ** 2) <= to_cells(4.0) ** 2)
    grid[reachable_chamber | sealed_void] = 0

    result = cave_geometry.analyse_cave(grid, RESOLUTION, reachable_chamber, max_features=5)

    assert result['chambers'], 'the reachable chamber should still be found'
    widest = result['chambers'][0]
    assert to_metres(widest['col']) == pytest.approx(5.0, abs=1.0)
    assert widest['width_m'] == pytest.approx(4.0, abs=0.3)
    for chamber in result['chambers']:
        assert to_metres(chamber['col']) < 15.0, 'reported a chamber inside the sealed void'


def test_reported_features_are_separated(analysis):
    """
    Otherwise the "widest areas" would be several cells of the same chamber - the clearance map
    varies smoothly, so a cell's closest rival is its own neighbour.
    """

    for features in (analysis['chambers'], analysis['bottlenecks']):
        for a, b in itertools.combinations(features, 2):
            separation_m = RESOLUTION * np.hypot(a['row'] - b['row'], a['col'] - b['col'])
            assert separation_m >= cave_geometry.MIN_FEATURE_SEPARATION_M - RESOLUTION


@pytest.mark.parametrize('description,grid_value', [
    ('entirely unexplored', -1),
    ('entirely solid rock', 100),
])
def test_degenerate_maps_report_nothing_without_crashing(description, grid_value):
    """Early in a run the map is empty; the analysis timer still fires."""

    grid = np.full((200, 200), grid_value, dtype=np.int8)
    result = cave_geometry.analyse_cave(grid, RESOLUTION, np.zeros((200, 200), dtype=bool))

    assert result['chambers'] == []
    assert result['bottlenecks'] == []


def test_open_space_has_no_bottlenecks():
    """A map with no walls has no pinch points - reporting one would be a false positive."""

    grid = np.zeros((400, 400), dtype=np.int8)
    result = cave_geometry.analyse_cave(grid, RESOLUTION, np.ones((400, 400), dtype=bool))

    assert result['bottlenecks'] == []
    assert result['chambers'], 'open space is all one big chamber, which should be reported'


# ---------------------------------------------------------------- Advanced 5

def euclidean_cost_matrix(points):
    return [[float(np.hypot(*(a - b))) for b in points] for a in points]


@pytest.fixture(scope='module')
def random_tour():
    """
    Nine random 2D stops, small enough to brute-force the exact optimum for comparison.

    Random points, not a tidy layout: on a line (or a circle) every sensible closed tour costs
    the same, nearest neighbour is already optimal, and the 2-opt stage is never exercised - so
    such a test would pass whether or not the optimisation worked at all.
    """

    points = np.random.default_rng(7).uniform(0.0, 20.0, size=(9, 2))
    cost_matrix = euclidean_cost_matrix(points)
    return cost_matrix, science_tour.plan_tour(cost_matrix, start=0)


def test_optimisation_improves_on_the_baseline(random_tour):
    """The comparison the task asks for: 2-opt should beat nearest neighbour."""

    _, plan = random_tour

    assert plan['optimised_cost_m'] < plan['baseline_cost_m']
    assert plan['saved_m'] == pytest.approx(plan['baseline_cost_m'] - plan['optimised_cost_m'])
    assert plan['saved_percent'] > 0.0


def test_optimised_tour_is_never_cheaper_than_the_true_optimum(random_tour):
    """
    Guards against the cost being miscounted rather than the route being improved - e.g. if the
    closing leg back to base were dropped, every tour would look shorter than possible.
    """

    cost_matrix, plan = random_tour
    optimum = min(science_tour.tour_cost([0] + list(rest), cost_matrix)
                  for rest in itertools.permutations(range(1, len(cost_matrix))))

    assert plan['optimised_cost_m'] >= optimum - 1e-9
    assert plan['baseline_cost_m'] >= optimum - 1e-9
    # 2-opt is a local search with no optimality guarantee, but it should land close.
    assert plan['optimised_cost_m'] <= optimum * 1.15


def test_tour_is_a_valid_closed_route(random_tour):
    cost_matrix, plan = random_tour
    count = len(cost_matrix)

    for order in (plan['baseline_order'], plan['optimised_order']):
        assert order[0] == 0, 'the tour must start at base'
        assert sorted(order) == list(range(count)), 'every stop exactly once'


def test_tour_cost_includes_the_leg_home():
    """The rover has to get back to base, so the tour is a closed loop, not a one-way path."""

    # Three stops in a line: 0 -> 1 -> 2 -> back to 0.
    cost_matrix = [[0.0, 1.0, 2.0],
                   [1.0, 0.0, 1.0],
                   [2.0, 1.0, 0.0]]

    assert science_tour.tour_cost([0, 1, 2], cost_matrix) == pytest.approx(4.0)


def test_unreachable_stops_are_left_out():
    """
    A location with no route to it is marked -1 by the cost matrix builder, and must be dropped
    rather than treated as a cheap or expensive leg - otherwise the tour plans a drive through
    solid rock.
    """

    points = np.random.default_rng(3).uniform(0.0, 20.0, size=(8, 2))
    cost_matrix = euclidean_cost_matrix(points)
    for i in range(len(cost_matrix)):
        if i != 5:
            cost_matrix[i][5] = cost_matrix[5][i] = -1.0

    plan = science_tour.plan_tour(cost_matrix, start=0)

    assert 5 not in plan['optimised_order']
    assert sorted(plan['optimised_order']) == [i for i in range(8) if i != 5]


@pytest.mark.parametrize('count', [1, 2, 3, 4, 12])
def test_small_and_full_size_tours_do_not_crash(count):
    """Fewer than four stops has no 2-opt move available; 12 is the tour size cap."""

    cost_matrix = [[float(abs(i - j)) for j in range(count)] for i in range(count)]
    plan = science_tour.plan_tour(cost_matrix, start=0)

    assert sorted(plan['optimised_order']) == list(range(count))
    assert plan['optimised_cost_m'] <= plan['baseline_cost_m'] + 1e-9
