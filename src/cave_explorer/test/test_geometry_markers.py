#!/usr/bin/env python3
"""
Tests for the Advanced 2 RViz drawing of chambers: magenta rings of the inscribed circle, the
widest one thick and labelled. Display only - the analysis itself is tested in
test_advanced_tasks.py.

Run with:  python -m pytest src/cave_explorer/test/test_geometry_markers.py -v
"""

import math
import types

import numpy as np
import pytest

cave_explorer = pytest.importorskip(
    'cave_explorer.cave_explorer',
    reason='needs a sourced ROS 2 environment (rclpy, cv_bridge)')

from cave_explorer import cave_geometry  # noqa: E402

CaveExplorer = cave_explorer.CaveExplorer


def publish(grid, resolution):
    published = []
    node = types.SimpleNamespace(
        geometry_=cave_geometry.analyse_cave(grid, resolution, grid == 0, max_features=3),
        geometry_markers_pub_=types.SimpleNamespace(publish=published.append))
    CaveExplorer.publish_geometry_markers(node, resolution, [0.0, 0.0])
    return [m for m in published[0].markers if m.ns]


def test_chambers_are_magenta_rings_with_the_widest_labelled():
    resolution = 0.1
    grid = np.full((80, 150), 100, dtype=np.int8)
    rows, cols = np.indices(grid.shape)
    grid[((cols - 35) ** 2 + (rows - 40) ** 2) <= 32 ** 2] = 0   # 6.4m-wide chamber
    grid[37:44, 60:140] = 0                                      # 0.7m corridor

    markers = publish(grid, resolution)
    rings = sorted((m for m in markers if m.ns == 'cave_geometry_chamber'), key=lambda m: m.id)
    widest = rings[0]

    assert widest.type == widest.LINE_STRIP, 'chambers are rings, not filled discs'
    assert (widest.color.r, widest.color.g, widest.color.b) == pytest.approx(cave_explorer.CHAMBER_COLOR)
    assert widest.scale.x == cave_explorer.CHAMBER_RING_HEADLINE_WIDTH_M

    # The ring is the inscribed circle: every point sits one clearance from the centre
    cx = sum(p.x for p in widest.points[:-1]) / (len(widest.points) - 1)
    cy = sum(p.y for p in widest.points[:-1]) / (len(widest.points) - 1)
    radii = [math.hypot(p.x - cx, p.y - cy) for p in widest.points]
    assert max(radii) - min(radii) < 0.01
    assert 2 * radii[0] == pytest.approx(6.4, abs=0.3)

    [label] = [m for m in markers if m.ns == 'cave_geometry_chamber_label']
    assert label.text.startswith('WIDEST:') and label.text.isascii()
    assert (label.color.r, label.color.g, label.color.b) == pytest.approx(cave_explorer.CHAMBER_COLOR)
    assert label.pose.position.y > cy + radii[0], 'the label sits just outside its ring'

    # Bottlenecks are untouched by this change
    assert any(m.ns == 'cave_geometry_bottleneck' and m.type == m.CYLINDER for m in markers)


def test_branch_colours_run_red_through_yellow_to_green():
    red, yellow, green = (c for _, c in cave_explorer.BRANCH_COLOR_STOPS)
    assert cave_explorer.branch_color(1.0) == red
    assert cave_explorer.branch_color(4.5) == pytest.approx(yellow)
    assert cave_explorer.branch_color(20.0) == green


def test_the_skeleton_is_one_connected_line_down_the_corridor():
    if not hasattr(__import__('cv2'), 'ximgproc'):
        pytest.skip('this OpenCV build has no contrib thinning module')
    free = np.zeros((40, 100), dtype=bool)
    free[15:25, 5:95] = True                     # a straight 10-cell corridor
    free[12:15, 48:50] = True                    # a shallow 3-cell nook - should be pruned away

    line = cave_geometry.skeleton(free, prune_cells=5)

    rows, cols = np.nonzero(line)
    assert np.all(np.abs(rows - 19.5) <= 2.0), 'the line runs down the middle'
    assert not line[:17].any(), 'the short nook stub was pruned'
    assert cols.max() - cols.min() > 60


def test_branches_are_drawn_red_in_the_corridor_and_green_in_the_chamber():
    if not hasattr(__import__('cv2'), 'ximgproc'):
        pytest.skip('this OpenCV build has no contrib thinning module')
    resolution = 0.1
    grid = np.full((80, 150), 100, dtype=np.int8)
    rows, cols = np.indices(grid.shape)
    grid[((cols - 35) ** 2 + (rows - 40) ** 2) <= 32 ** 2] = 0   # 6.4m-wide chamber
    grid[37:44, 60:140] = 0                                      # 0.7m corridor

    markers = publish(grid, resolution)
    [branches] = [m for m in markers if m.ns == 'cave_geometry_branches']

    assert branches.type == branches.LINE_LIST
    assert len(branches.points) == len(branches.colors) > 0
    assert len(branches.points) % 2 == 0
    red, _, green = (c for _, c in cave_explorer.BRANCH_COLOR_STOPS)
    in_corridor = [c for p, c in zip(branches.points, branches.colors) if p.x > 9.0]
    at_chamber_centre = [c for p, c in zip(branches.points, branches.colors)
                         if abs(p.x - 3.55) < 0.6 and abs(p.y - 4.05) < 0.6]
    assert in_corridor and all((c.r, c.g, c.b) == pytest.approx(red) for c in in_corridor)
    assert at_chamber_centre and all(c.g > 0.8 and c.r < 0.5 for c in at_chamber_centre), \
        'the middle of the 6.4m chamber should read green'


def test_a_dead_end_taper_takes_the_width_where_the_narrowing_starts():
    """A branch running into a wall shrinks to nothing - its tip must not read as a narrow passage."""
    line = np.zeros((1, 10), dtype=bool)
    line[0, :] = True
    width = np.array([[6.0, 6.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5]], dtype=np.float32)

    shown = cave_geometry.taper_free_widths(line, width)

    assert (shown[0, 4:] == 6.0).all(), 'the taper into the wall is shown at the chamber width'


def test_a_real_narrow_neck_before_a_dead_end_pocket_stays_narrow():
    """Narrow, then opening into a pocket, then the wall: only the pocket's own taper is flattened."""
    line = np.zeros((1, 10), dtype=bool)
    line[0, :] = True
    width = np.array([[6.0, 6.0, 2.0, 1.0, 2.0, 4.0, 4.0, 3.0, 2.0, 1.0]], dtype=np.float32)

    shown = cave_geometry.taper_free_widths(line, width)

    assert shown[0, 3] == 1.0, 'the neck is real and stays narrow'
    assert (shown[0, 7:] == 4.0).all(), "the pocket's taper into the wall is flattened"


def test_a_narrow_constant_width_dead_end_corridor_stays_narrow():
    line = np.zeros((1, 10), dtype=bool)
    line[0, :] = True
    width = np.array([[6.0, 6.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.3]], dtype=np.float32)

    shown = cave_geometry.taper_free_widths(line, width)

    assert np.allclose(shown[0, 2:8], 0.7), 'the narrow corridor is not repainted wide'
