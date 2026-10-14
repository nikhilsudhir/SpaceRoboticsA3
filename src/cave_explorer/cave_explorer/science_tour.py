#!/usr/bin/env python3
"""
Advanced 5: science tour planning (TSP-style route optimisation).

Pure route-optimisation code (standard library only, no ROS, no numpy) so it can be tested
offline on a hand-written cost matrix. Choosing WHICH locations to visit, measuring the travel
costs between them, and actually driving the route all live in cave_explorer.py.

The problem: given N science locations and the cost of driving between every pair, find the
cheapest closed loop that visits all of them and returns to base. That is the Travelling
Salesman Problem, so we take the standard two-stage approach:

  1. BASELINE - nearest neighbour. Repeatedly drive to the closest place not yet visited.
     Fast and obvious, but short-sighted: it happily strands a location on the far side of the
     cave and has to make a long dedicated trip back for it at the end.

  2. OPTIMISATION - 2-opt. Repeatedly look for two edges in the tour that can be swapped for a
     cheaper pair, and keep going until no swap helps. This is what untangles the long
     backtracks nearest neighbour leaves behind (see improve_with_2_opt() for why reversing a
     segment is exactly the right move).

Both stages run on the same cost matrix, so the before/after total is a fair comparison and is
what gets reported for the task's "compare optimised vs baseline" requirement.

Note that costs here are PATH distances through the map, not straight-line distances - the
caller builds the matrix by running its wavefront BFS over the occupancy grid. That distinction
is the whole point: two chambers either side of a wall are metres apart in a straight line but
a long way apart by road, and a straight-line tour would confidently plan a route through solid
rock. These functions never see coordinates, only the matrix, so they cannot make that mistake.
"""

# 2-opt stops as soon as a full pass finds no improvement, so it normally terminates on its own
# well before this. The cap only exists so a pathological cost matrix (e.g. ties that let the
# search cycle) can't spin forever inside the control loop.
MAX_2_OPT_PASSES = 50

# Improvements smaller than this (metres) are ignored. Without it, floating-point noise in the
# cost matrix can make a swap look infinitesimally better forever, and the loop never settles.
MIN_IMPROVEMENT_M = 1e-6


def tour_cost(order, cost_matrix):
    """
    Total cost of the closed tour: each leg plus the final leg back to the starting location.

    The tour is closed because the rover has to come home - it has to return to base whether we
    count that leg or not, so leaving it out would mean optimising the wrong quantity (and would
    reward tours that end as far from home as possible).
    """

    return sum(cost_matrix[order[i]][order[(i + 1) % len(order)]]
               for i in range(len(order)))


def nearest_neighbour_order(cost_matrix, start=0):
    """
    Baseline route: from the start, always hop to the nearest location not yet visited.

    Unreachable legs are marked with a negative cost by the caller and are skipped here, so a
    location the robot genuinely cannot get to never ends up in the tour.
    """

    count = len(cost_matrix)
    order = [start]
    unvisited = set(range(count)) - {start}

    while unvisited:
        current = order[-1]
        reachable = [candidate for candidate in unvisited
                     if cost_matrix[current][candidate] >= 0.0]
        if not reachable:
            break
        nearest = min(reachable, key=lambda candidate: cost_matrix[current][candidate])
        order.append(nearest)
        unvisited.discard(nearest)

    return order


def improve_with_2_opt(order, cost_matrix, max_passes=MAX_2_OPT_PASSES):
    """
    Improve a tour with 2-opt: find two edges whose removal and reconnection shortens the tour.

    Take the tour ... A -> B ... C -> D ... Remove edges A->B and C->D, then reconnect as
    A->C ... B->D. Reconnecting the two loose ends the only other possible way requires
    traversing the B...C stretch backwards, which is why the move is implemented as reversing
    that segment. The tour still visits everything exactly once; only two edges changed, so the
    saving is just

        (cost A->B + cost C->D) - (cost A->C + cost B->D)

    and there is no need to re-total the whole tour to evaluate a candidate swap. Index 0 is
    never moved, which keeps the tour starting at base.

    Returns a new list; the input is not modified.
    """

    order = list(order)
    count = len(order)
    if count < 4:
        # With three or fewer stops every tour is the same loop, so there is nothing to swap.
        return order

    for _ in range(max_passes):
        improved = False

        for i in range(1, count - 1):
            for j in range(i + 1, count):
                a, b = order[i - 1], order[i]
                c, d = order[j], order[(j + 1) % count]

                # All four legs must be known-reachable, or the "saving" is meaningless.
                if min(cost_matrix[a][b], cost_matrix[c][d],
                       cost_matrix[a][c], cost_matrix[b][d]) < 0.0:
                    continue

                saving = ((cost_matrix[a][b] + cost_matrix[c][d])
                          - (cost_matrix[a][c] + cost_matrix[b][d]))
                if saving > MIN_IMPROVEMENT_M:
                    order[i:j + 1] = reversed(order[i:j + 1])
                    improved = True

        if not improved:
            # A whole pass found nothing better - this is a 2-opt local optimum.
            break

    return order


def plan_tour(cost_matrix, start=0):
    """
    Run both stages and report the comparison.

    Returns a dict with the baseline order and cost, the optimised order and cost, and the
    saving in metres and percent - the numbers the task asks us to compare.
    """

    baseline_order = nearest_neighbour_order(cost_matrix, start)
    baseline_cost = tour_cost(baseline_order, cost_matrix)

    optimised_order = improve_with_2_opt(baseline_order, cost_matrix)
    optimised_cost = tour_cost(optimised_order, cost_matrix)

    saved_m = baseline_cost - optimised_cost

    return {
        'baseline_order': baseline_order,
        'baseline_cost_m': baseline_cost,
        'optimised_order': optimised_order,
        'optimised_cost_m': optimised_cost,
        'saved_m': saved_m,
        'saved_percent': (100.0 * saved_m / baseline_cost) if baseline_cost > 0.0 else 0.0,
    }
