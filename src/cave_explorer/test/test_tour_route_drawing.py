#!/usr/bin/env python3
"""
Tests that the science tour is drawn along the corridors, not as straight lines through rock.

Why this exists: RViz and the GUI drew each leg as a straight line between stops, so the
optimised route visibly crossed walls and left the map. The costs were always path distances;
now the drawn line is the same BFS path those costs measured. Display only - driving is untouched.

Run with:  python -m pytest src/cave_explorer/test/test_tour_route_drawing.py -v
"""

import types

import numpy as np
import pytest

cave_explorer = pytest.importorskip(
    'cave_explorer.cave_explorer',
    reason='needs a sourced ROS 2 environment (rclpy, cv_bridge)')

CaveExplorer = cave_explorer.CaveExplorer
FREE, WALL = 0, 100


def l_shaped_corridor():
    """Free cells along row 1 and column 8, walls everywhere else: an L around solid rock."""
    grid = np.full((10, 10), WALL, dtype=np.int8)
    grid[1, 1:9] = FREE
    grid[1:9, 8] = FREE
    return grid


def make_node(grid):
    node = types.SimpleNamespace(map_grid_=grid, map_origin_=(0.0, 0.0), map_resolution_=1.0)
    for name in ('compute_path_distance_grid', 'world_to_grid', 'grid_to_world'):
        setattr(node, name, types.MethodType(getattr(CaveExplorer, name), node))
    return node


def test_trace_follows_the_corridor_round_the_corner():
    node = make_node(l_shaped_corridor())
    dist = node.compute_path_distance_grid(1, 1)
    cells = cave_explorer.trace_grid_path(dist, 8, 8)

    assert cells[0] == (1, 1) and cells[-1] == (8, 8)
    assert all(node.map_grid_[r, c] == FREE for r, c in cells), 'the drawn route crossed rock'
    assert all(max(abs(r1 - r2), abs(c1 - c2)) == 1
               for (r1, c1), (r2, c2) in zip(cells, cells[1:]))


def test_an_unreachable_end_gives_no_path():
    node = make_node(l_shaped_corridor())
    dist = node.compute_path_distance_grid(1, 1)
    assert cave_explorer.trace_grid_path(dist, 5, 3) == []


def test_route_points_stay_inside_free_space():
    node = make_node(l_shaped_corridor())
    stops = [{'x': 1.5, 'y': 1.5}, {'x': 8.5, 'y': 8.5}]
    points = CaveExplorer.tour_route_points(node, stops, [0, 1])

    assert points[0] == points[-1] == (1.5, 1.5)  # one closed, unbroken loop
    for x, y in points:
        row, col = node.world_to_grid(x, y)
        assert node.map_grid_[row, col] == FREE, f'route point ({x}, {y}) is inside rock'


def test_the_route_keeps_off_the_walls_in_a_wide_chamber():
    """
    The plain shortest path hugs walls; the rover (Nav2, 1.0m inflation) drives down the middle.
    In a 9m-wide chamber with stops near the left and right ends, the drawn line should keep
    clear of the top and bottom walls rather than skimming along them.
    """

    grid = np.full((40, 40), WALL, dtype=np.int8)
    grid[10:31, 2:38] = FREE            # 21 rows (6.3m) x 36 cols at 0.3m per cell
    node = make_node(grid)
    node.map_resolution_ = 0.3
    stops = [{'x': 3 * 0.3, 'y': 20 * 0.3}, {'x': 36 * 0.3, 'y': 20 * 0.3}]

    points = CaveExplorer.tour_route_points(node, stops, [0, 1])

    for x, y in points:
        row, col = node.world_to_grid(x, y)
        if 6 <= col <= 33:  # away from the stops themselves
            assert 13 <= row <= 27, f'route point ({x:.1f}, {y:.1f}) is within 1m of a wall'
