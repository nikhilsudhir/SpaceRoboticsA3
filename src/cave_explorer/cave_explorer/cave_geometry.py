#!/usr/bin/env python3
"""
Advanced 2: cave geometry analysis.

Pure analysis code (numpy/OpenCV only, no ROS) so it can be tested offline against a saved
occupancy grid without bringing up the simulator. The ROS wiring - when this runs, and how the
results are drawn in RViz - lives in cave_explorer.py.

The method, in one line: build a clearance map (how far every free cell is from the nearest
wall), then read its shape to find the two things the task asks for - the widest accessible
area, and the narrow passages/bottlenecks.

  * Widest area = a MAXIMUM of the clearance map. The cell furthest from any wall is the centre
    of the biggest open chamber, and its clearance value is literally the radius of the largest
    circle that fits there, so the answer comes with its own measurement.

  * Bottleneck = a SADDLE point of the clearance map. This is the key idea, and it's why simply
    looking for low-clearance cells doesn't work: every cell beside a wall has low clearance, so
    that would just find walls. What makes a pinch point different is the shape of the clearance
    map around it. Walking ALONG the corridor through a bottleneck, clearance rises in both
    directions (it opens out into a chamber at either end), so the bottleneck is a minimum along
    that axis. Walking ACROSS the corridor, clearance falls off toward both walls, so it's a
    maximum along that axis. Minimum in one direction and maximum in the perpendicular one is
    the definition of a saddle, which is exactly what the Hessian - the matrix of second
    derivatives of the clearance map - is able to detect. See find_bottlenecks().

Using the derivatives of the distance transform this way is what the task hints at with "the
gradient/Jacobian of the distance transform".
"""

import cv2
import numpy as np

# Cells with an occupancy cost at or above this are treated as walls. Matches
# FRONTIER_OCCUPIED_THRESHOLD in cave_explorer.py (nav_msgs/OccupancyGrid convention:
# -1 unknown, 0-99 increasing cost, 100 definitely occupied).
OCCUPIED_THRESHOLD = 50

# The clearance map is smoothed before its derivatives are taken. Raw SLAM occupancy grids are
# speckled - a single stray occupied cell mid-corridor puts a sharp dent in the clearance map,
# and second derivatives amplify exactly that kind of noise into thousands of spurious "saddle
# points". Sigma is in metres (converted to cells below) so the amount of smoothing is tied to
# real-world scale rather than map resolution: ~0.3m blurs out single-cell speckle while
# preserving the metre-scale structure (corridors, chambers) we actually want to find.
SMOOTHING_SIGMA_M = 0.3

# A chamber must have at least this much clearance (metres) to be worth reporting - below this
# it's a corridor widening, not an open area.
MIN_CHAMBER_CLEARANCE_M = 1.5

# A bottleneck's clearance must fall in this range. Below the minimum it's almost certainly map
# noise (a gap the robot could never fit through anyway); above the maximum it isn't "narrow" in
# any meaningful sense, just a soft waist between two large areas.
MIN_BOTTLENECK_CLEARANCE_M = 0.3
MAX_BOTTLENECK_CLEARANCE_M = 2.5

# Reported features are kept at least this far apart (metres) - see pick_separated_features().
MIN_FEATURE_SEPARATION_M = 4.0

# Saddle candidates must have a near-zero gradient, since a saddle is a critical point of the
# clearance map. On a discrete, smoothed grid nothing is exactly zero, so instead of a fixed
# threshold (which would mean something different on every map) we keep this percentile of the
# flattest candidate cells.
SADDLE_GRADIENT_PERCENTILE = 35.0


def compute_clearance_map(grid, resolution):
    """
    Build the clearance map (distance transform): for every cell, the distance in METRES to the
    nearest wall or unexplored cell.

    Unknown cells are deliberately treated as walls. The robot cannot assume unexplored space is
    open, so clearance near the frontier is reported conservatively - a chamber only reads as
    "wide" once it has actually been observed to be wide, rather than looking wide because the
    map simply stops there. As exploration fills the map in, the clearance values settle.

    Returns (clearance_m, free_mask).
    """

    free = (grid >= 0) & (grid < OCCUPIED_THRESHOLD)

    # distanceTransform measures distance to the nearest ZERO pixel, so free space is the
    # non-zero input and everything else (walls + unknown) is the zero it measures against.
    distance_cells = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)

    return distance_cells * resolution, free


def smooth_clearance(clearance_m, resolution):
    """Gaussian-blur the clearance map - see SMOOTHING_SIGMA_M for why this is necessary."""

    sigma_cells = max(SMOOTHING_SIGMA_M / resolution, 0.5)
    return cv2.GaussianBlur(clearance_m, (0, 0), sigma_cells)


def pick_separated_features(values, candidates_mask, resolution, count, prefer_lowest):
    """
    Greedily pick up to 'count' candidate cells that are the best by 'values' (lowest if
    prefer_lowest, else highest) and at least MIN_FEATURE_SEPARATION_M apart.

    Take the best cell anywhere, blank out everything within the separation radius of it, then
    repeat. Without the blanking step the "top 5 widest areas" would almost always be five cells
    of the same chamber - the clearance map varies smoothly, so a cell's best competitor is its
    own neighbour - and the five distinct places the task actually wants would never appear.

    Done with whole-array numpy operations rather than by sorting the candidates and filtering
    them in python: the candidate mask routinely covers tens of thousands of cells (a saddle
    condition is satisfied over broad regions, not just at isolated points), and iterating over
    that in python every few seconds was enough to starve this node's other callbacks - the same
    problem documented on compute_path_distance_grid() in cave_explorer.py. This way the cost is
    a couple of full-array passes per feature picked, with 'count' in single digits.

    Returns a list of (row, col, value) tuples, best first.
    """

    # Blanked-out and non-candidate cells are set to the value that can never win, so a single
    # argmin/argmax over the whole array finds the next pick without needing a separate mask.
    never_wins = np.inf if prefer_lowest else -np.inf
    work = np.where(candidates_mask, values.astype(np.float64), never_wins)

    radius_cells = max(int(round(MIN_FEATURE_SEPARATION_M / resolution)), 1)
    height, width = values.shape
    row_indices = np.arange(height)[:, None]
    col_indices = np.arange(width)[None, :]

    picked = []
    for _ in range(count):
        flat_index = int(np.argmin(work) if prefer_lowest else np.argmax(work))
        best_value = work.flat[flat_index]
        if not np.isfinite(best_value):
            break  # every candidate has been picked or blanked out

        row, col = divmod(flat_index, width)
        picked.append((row, col, float(values[row, col])))

        within_separation = ((row_indices - row) ** 2
                             + (col_indices - col) ** 2) <= radius_cells ** 2
        work[within_separation] = never_wins

    return picked


def find_ridge(smoothed_m):
    """
    Mask of cells on the ridge of the clearance map - the centreline of a corridor or chamber,
    i.e. the set of points that are locally as far from the walls as it is possible to get. (This
    is the medial axis of the free space, approximated on the grid.)

    This is what separates a passage from a wall, and it's needed because low clearance by
    itself describes both. Clearance is low at a bottleneck AND low at every cell that merely
    happens to sit beside a wall, of which there are thousands - so without this, "the narrowest
    place in the cave" reliably comes back as some arbitrary cell hugging a wall, or a concave
    corner (whose distance transform curves like a saddle too). A cell beside a wall is not on
    the ridge: clearance rises as you step away from the wall, so it is not a maximum in that
    direction.

    A cell qualifies if it is a maximum along either grid axis. The test needs care on flat
    ground: next to a straight wall, clearance is constant in the direction PARALLEL to the
    wall, so a plain "greater than or equal to both neighbours" test passes there and lets every
    wall-adjacent cell back in - which is exactly the bug this function exists to prevent.
    Requiring it to be strictly greater than at least one of the two neighbours excludes those,
    while still accepting a genuine centreline whose maximum is split across two equal cells (an
    even-width corridor, which has no single middle cell).
    """

    def shifted(axis, offset):
        result = np.roll(smoothed_m, offset, axis=axis).astype(np.float32)
        # np.roll wraps around; make the wrapped-in edge lose every comparison, the same
        # concern dilate8_mask() handles in cave_explorer.py.
        if axis == 0 and offset == 1:
            result[0, :] = -np.inf
        elif axis == 0:
            result[-1, :] = -np.inf
        elif offset == 1:
            result[:, 0] = -np.inf
        else:
            result[:, -1] = -np.inf
        return result

    ridge = np.zeros(smoothed_m.shape, dtype=bool)
    for axis in (0, 1):
        before = shifted(axis, 1)
        after = shifted(axis, -1)
        ridge |= ((smoothed_m >= before) & (smoothed_m >= after)
                  & ((smoothed_m > before) | (smoothed_m > after)))

    return ridge


def find_chambers(clearance_m, reachable, resolution, count):
    """
    Find open chambers: the cells furthest from any wall.

    No local-maximum test is needed, because pick_separated_features() already enforces a
    minimum separation - taking the global maximum, excluding everything within
    MIN_FEATURE_SEPARATION_M of it, then taking the next maximum, directly yields the widest
    well-separated areas. That is both simpler than a local-maximum filter and more robust than
    one: a dilation-based local-maximum test behaves badly on the flat plateaus that large open
    areas produce (every cell on a plateau is a local maximum of it).

    Deliberately works on the RAW clearance map, not the smoothed one. Smoothing is only
    necessary where derivatives are taken (see find_bottlenecks()), and it systematically
    understates a chamber's size: the blur pulls in the zeros from outside the free space, which
    shaves the peak - measured at ~0.4m on a 6m chamber in testing, and enough to push a
    genuine 3.2m-wide chamber below MIN_CHAMBER_CLEARANCE_M and out of the results entirely.
    Chamber width is a headline number this task reports, so it's measured on the real
    distance transform.

    Each result carries 'clearance_m' (the radius of the largest circle that fits at that point)
    and 'width_m' (twice that - the distance across). Sorted widest first, so the first entry is
    "the widest accessible area discovered so far".
    """

    candidates_mask = reachable & (clearance_m >= MIN_CHAMBER_CLEARANCE_M)

    return [
        {
            'row': row,
            'col': col,
            'clearance_m': clearance_m_at,
            'width_m': 2.0 * clearance_m_at,
            'kind': 'chamber',
        }
        for row, col, clearance_m_at in pick_separated_features(
            clearance_m, candidates_mask, resolution, count, prefer_lowest=False)
    ]


def find_bottlenecks(clearance_m, smoothed_m, reachable, resolution, count):
    """
    Find narrow passages: saddle points on the ridge of the clearance map. See this module's
    docstring for why a bottleneck is a saddle, and find_ridge() for why the ridge condition is
    needed as well.

    The Hessian's two eigenvalues have opposite signs exactly when the surface curves down in
    one direction and up in the perpendicular one. Rather than computing the eigenvalues, we
    test their product, which is the determinant:

        det(H) = d2/dcol2 * d2/drow2 - (d2/drowdcol)^2 < 0   <=>   eigenvalues differ in sign

    A true critical point also has zero gradient; on a discrete grid we keep the flattest
    candidates instead (see SADDLE_GRADIENT_PERCENTILE).

    The derivative tests use the SMOOTHED clearance map, since second derivatives of a raw SLAM
    occupancy grid are mostly noise, but the thresholds and the reported measurement use the raw
    one - so the gap width quoted for a bottleneck is the real distance transform's answer, not
    a blurred approximation of it.

    'width_m' is the actual width of the gap: clearance is the distance to the nearest wall, so
    the passage is twice that across. Sorted narrowest first, so the first entry is the tightest
    bottleneck in the cave so far.
    """

    # First derivatives (the gradient/Jacobian), then second derivatives of those (the Hessian).
    d_col = cv2.Sobel(smoothed_m, cv2.CV_32F, 1, 0, ksize=3)
    d_row = cv2.Sobel(smoothed_m, cv2.CV_32F, 0, 1, ksize=3)
    d_col_col = cv2.Sobel(d_col, cv2.CV_32F, 1, 0, ksize=3)
    d_row_row = cv2.Sobel(d_row, cv2.CV_32F, 0, 1, ksize=3)
    d_row_col = cv2.Sobel(d_col, cv2.CV_32F, 0, 1, ksize=3)

    hessian_determinant = d_col_col * d_row_row - d_row_col ** 2
    gradient_magnitude = np.hypot(d_col, d_row)

    candidates_mask = (
        (hessian_determinant < 0.0)          # opposite-signed curvature => saddle
        & find_ridge(smoothed_m)             # on the corridor centreline, not against a wall
        & reachable
        & (clearance_m >= MIN_BOTTLENECK_CLEARANCE_M)
        & (clearance_m <= MAX_BOTTLENECK_CLEARANCE_M)
    )
    if not candidates_mask.any():
        return []

    gradient_threshold = np.percentile(gradient_magnitude[candidates_mask],
                                       SADDLE_GRADIENT_PERCENTILE)
    candidates_mask &= gradient_magnitude <= gradient_threshold

    return [
        {
            'row': row,
            'col': col,
            'clearance_m': clearance_m_at,
            'width_m': 2.0 * clearance_m_at,
            'kind': 'bottleneck',
        }
        for row, col, clearance_m_at in pick_separated_features(
            clearance_m, candidates_mask, resolution, count, prefer_lowest=True)
    ]


def analyse_cave(grid, resolution, reachable, max_features=5):
    """
    Run the full Advanced 2 analysis on the current map.

    'reachable' is a boolean mask of cells the robot can actually drive to through known free
    space (computed by the caller with its existing BFS - see compute_path_distance_grid() in
    cave_explorer.py). Restricting to it matters: without it, sealed voids inside the rock and
    noise pockets behind walls show up as enormous "chambers" the rover could never visit, and
    the headline "widest accessible area" result would be wrong in exactly the way the task
    cares about - it asks for the widest ACCESSIBLE area.

    Returns a dict with the clearance map plus up to max_features chambers and bottlenecks, each
    sorted best-first (widest / narrowest).
    """

    clearance_m, free = compute_clearance_map(grid, resolution)
    smoothed_m = smooth_clearance(clearance_m, resolution)
    reachable_free = reachable & free

    return {
        'clearance_m': clearance_m,
        'chambers': find_chambers(clearance_m, reachable_free, resolution, max_features),
        'bottlenecks': find_bottlenecks(clearance_m, smoothed_m, reachable_free,
                                        resolution, max_features),
    }
