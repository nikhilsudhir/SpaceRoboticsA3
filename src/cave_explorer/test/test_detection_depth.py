#!/usr/bin/env python3
"""
Tests that a detection's distance comes from the artefact, not the wall behind it.

Why this exists: project_detection_to_map() used to read depth from a 5x5 patch at the bbox
centre. For a thin artefact like green_alien, that centre pixel can fall between its limbs onto
the cave wall behind it. Live, one alien sighting was placed 4.4m beyond the real alien (at
(14.7, 21.0) vs (19.1, 20.8)), outside ARTIFACT_DUPLICATE_RADIUS_M, so it started a second
cluster that confirmed as green_crystals and was then "inspected". estimate_detection_depth()
now takes a low percentile (DEPTH_SAMPLE_PERCENTILE) over the whole (inset) box instead.

Run with:  python -m pytest src/cave_explorer/test/test_detection_depth.py -v
"""

import numpy as np
import pytest

cave_explorer = pytest.importorskip(
    'cave_explorer.cave_explorer',
    reason='needs a sourced ROS 2 environment (rclpy, cv_bridge)')

estimate_detection_depth = cave_explorer.estimate_detection_depth

HEIGHT, WIDTH = cave_explorer.CAMERA_HEIGHT_PX, cave_explorer.CAMERA_WIDTH_PX
WALL_M = 9.0
ARTEFACT_M = 5.0


def wall():
    return np.full((HEIGHT, WIDTH), WALL_M, dtype=np.float32)


def draw_side_on_alien(depth, x, y, w, h):
    """
    A side-on alien: body in a narrow column at the left of its box, one arm stretched out to
    the right edge. The box centre is open wall, and the alien covers under a quarter of the box.
    """

    depth[y:y + h, x + w // 6:x + w // 3] = ARTEFACT_M                 # head, body, legs
    depth[y + h // 4:y + h // 4 + h // 12, x + w // 3:x + w] = ARTEFACT_M  # outstretched arm


def test_a_thin_artefact_reads_its_own_depth_not_the_wall_behind_it():
    depth = wall()
    bbox = (300, 100, 120, 240)
    draw_side_on_alien(depth, *bbox)
    x, y, w, h = bbox
    centre_patch = depth[y + h // 2 - 2:y + h // 2 + 3, x + w // 2 - 2:x + w // 2 + 3]
    assert np.median(centre_patch) == WALL_M  # what the old centre-patch sample read

    assert estimate_detection_depth(depth, bbox) == pytest.approx(ARTEFACT_M)


def test_a_solid_artefact_reads_its_front_surface():
    # A sphere of radius 0.4m centred 4.4m away: its front surface spans 4.0-4.4m
    depth = wall()
    x, y, size = 320, 200, 80
    rows, cols = np.mgrid[0:size, 0:size]
    r = np.hypot(rows - size / 2, cols - size / 2) / (size / 2)
    inside = r <= 1.0
    patch = depth[y:y + size, x:x + size]
    patch[inside] = 4.4 - 0.4 * np.sqrt(1.0 - r[inside] ** 2)

    estimate = estimate_detection_depth(depth, (x, y, size, size))

    assert 4.0 <= estimate <= 4.4


def test_a_single_stray_near_pixel_does_not_win():
    depth = wall()
    bbox = (100, 100, 60, 60)
    depth[100:160, 100:160] = ARTEFACT_M
    depth[130, 130] = 0.3

    assert estimate_detection_depth(depth, bbox) == pytest.approx(ARTEFACT_M)


def test_invalid_readings_are_ignored():
    depth = wall()
    bbox = (100, 100, 60, 60)
    depth[100:160, 100:160] = ARTEFACT_M
    depth[100:160, 100:130] = np.nan
    depth[100:115, 130:160] = np.inf
    depth[145:160, 130:160] = 0.0

    assert estimate_detection_depth(depth, bbox) == pytest.approx(ARTEFACT_M)


def test_no_valid_reading_returns_none():
    # Point-blank, below the depth camera's minimum range
    depth = wall()
    depth[100:160, 100:160] = np.nan

    assert estimate_detection_depth(depth, (100, 100, 60, 60)) is None


def test_a_box_hanging_off_the_image_edge_is_clipped():
    depth = wall()
    depth[:, WIDTH - 30:] = ARTEFACT_M

    assert estimate_detection_depth(depth, (WIDTH - 40, -20, 80, 60)) == pytest.approx(ARTEFACT_M)


def test_the_smallest_box_detect_stop_signs_can_return_still_reads():
    # detect_stop_signs() uses minSize=(20, 20)
    depth = wall()
    depth[200:220, 300:320] = ARTEFACT_M

    assert estimate_detection_depth(depth, (300, 200, 20, 20)) == pytest.approx(ARTEFACT_M)
