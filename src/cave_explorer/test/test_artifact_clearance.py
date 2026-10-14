#!/usr/bin/env python3
"""
Tests for the artefact-body clearance logic that keeps driving goals out of the artefacts.

Why this exists: the inspection standoff and the Advanced 5 tour stops are both measured from
an artefact's detected centroid, and two of the models in mars_cave.sdf (the green crystal
clusters and the ice formations, both roughly 5m across) are wider than the old fixed 2.0m
standoff - so the goal was being placed inside the artefact. The occupancy grid doesn't catch
it, because the lidar is a single horizontal ray at z = 1.0m and those two models are clusters
of thin spikes at that height: most rays pass between the spikes, the cells behind them get
cleared, and the inside of the model reads as free space. Observed live as the rover sitting
inside a green crystal cluster with its wheels still turning.

Two different things are checked here:

  * the decision logic - artifact_standoff_distance() and is_clear_of_artifacts() - exercised
    directly, with no ROS graph and no simulator, by binding the methods to a stub that has
    nothing but the artefact list they read.
  * the constants themselves - ARTIFACT_PHYSICAL_RADIUS_M is measured from the artefact meshes,
    so test_radius_table_covers_every_model_in_the_world re-measures them from the actual .obj
    files and the actual <scale> in mars_cave.sdf and fails if the table under-states any of
    them. That's the test that matters for keeping this honest: the table is only as good as the
    world it was measured from, and the world is a file someone can edit.

Run with:  python -m pytest src/cave_explorer/test/test_artifact_clearance.py -v

The mesh-measuring test reads ~44MB of .obj geometry (the ice formation mesh alone is 37MB), so
it takes a few seconds; the logic tests are instant. Deselect it with -k "not world" if you want
only the fast ones.
"""

import math
import os
import types
import xml.etree.ElementTree as ET

import pytest

# The node module imports rclpy, cv_bridge and friends. The pure-logic modules under test in
# test_advanced_tasks.py deliberately don't, but this clearance logic reads the node's own
# artefact list, so it lives on the node class. Skip rather than error when run outside a
# sourced ROS environment, so this file never breaks a plain `pytest` on a bare checkout.
cave_explorer = pytest.importorskip(
    'cave_explorer.cave_explorer',
    reason='needs a sourced ROS 2 environment (rclpy, cv_bridge)')

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD_FILE = os.path.join(PACKAGE_DIR, 'worlds', 'mars_cave.sdf')

# Matches scan_joint in urdf/mars_explorer.urdf.xacro.
LIDAR_HEIGHT_M = 1.0


def make_node(clusters):
    """
    A stand-in for the node carrying only what the methods under test actually read.

    The two methods touch self.artifact_clusters_ and nothing else - no publishers, no clock, no
    map - so binding them to a namespace is enough to test them, and avoids both rclpy.init()
    and the need for a running ROS graph.
    """

    node = types.SimpleNamespace(artifact_clusters_=clusters)
    node.artifact_standoff_distance = types.MethodType(
        cave_explorer.CaveExplorer.artifact_standoff_distance, node)
    node.is_clear_of_artifacts = types.MethodType(
        cave_explorer.CaveExplorer.is_clear_of_artifacts, node)
    return node


def cluster(label, x, y, observations=None):
    if observations is None:
        observations = cave_explorer.ARTIFACT_CONFIRMATION_OBSERVATIONS
    return {
        'label': label,
        'id': abs(hash((label, x, y))) % 1000,
        'position': types.SimpleNamespace(x=float(x), y=float(y), z=0.0),
        'num_observations': observations,
    }


# --------------------------------------------------------------------------------------------
# artifact_standoff_distance()
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('label', sorted(cave_explorer.ARTIFACT_PHYSICAL_RADIUS_M))
def test_standoff_clears_the_artefacts_own_body(label):
    """Standing at the standoff distance must put the whole rover outside the artefact."""

    node = make_node([])
    standoff = node.artifact_standoff_distance(label)
    needed = (cave_explorer.ARTIFACT_PHYSICAL_RADIUS_M[label]
              + cave_explorer.ROVER_FOOTPRINT_RADIUS_M)
    assert standoff >= needed, (
        f'{label}: standoff {standoff:.2f}m is inside the model '
        f'(needs at least {needed:.2f}m)')


def test_wide_artefacts_stand_further_off_than_the_old_fixed_distance():
    """The regression this fixes: 2.0m was inside the crystal clusters and ice formations."""

    node = make_node([])
    for label in ('green_crystals', 'ice_formation'):
        assert node.artifact_standoff_distance(label) > \
            cave_explorer.INSPECTION_STANDOFF_DISTANCE_M


def test_small_artefacts_are_still_inspected_from_close_range():
    """
    The fix must not back the rover away from everything - only from what needs it. Small
    artefacts should keep the original close-range standoff, or the camera gets a worse view
    than before for no reason.
    """

    node = make_node([])
    assert node.artifact_standoff_distance('white_sphere') == \
        pytest.approx(cave_explorer.INSPECTION_STANDOFF_DISTANCE_M)


def test_unknown_label_gets_the_cautious_default():
    """A label with no table entry must not silently get zero clearance."""

    node = make_node([])
    standoff = node.artifact_standoff_distance('something_new_from_perception')
    assert standoff >= (cave_explorer.ARTIFACT_DEFAULT_PHYSICAL_RADIUS_M
                        + cave_explorer.ROVER_FOOTPRINT_RADIUS_M)
    assert standoff >= cave_explorer.INSPECTION_STANDOFF_DISTANCE_M


# --------------------------------------------------------------------------------------------
# is_clear_of_artifacts()
# --------------------------------------------------------------------------------------------

def test_point_inside_a_crystal_cluster_is_rejected():
    """The actual bug: a goal 2.0m from the crystal's centroid is inside the crystal."""

    node = make_node([cluster('green_crystals', 8.2, 5.2)])
    assert not node.is_clear_of_artifacts(8.2 + 2.0, 5.2)


def test_the_computed_standoff_point_is_accepted():
    """And the distance the code now actually uses is outside it, from every angle."""

    node = make_node([cluster('green_crystals', 8.2, 5.2)])
    standoff = node.artifact_standoff_distance('green_crystals')
    for sample in range(16):
        angle = 2.0 * math.pi * sample / 16
        x = 8.2 + standoff * math.cos(angle)
        y = 5.2 + standoff * math.sin(angle)
        assert node.is_clear_of_artifacts(x, y), f'rejected at angle {math.degrees(angle):.0f}'


def test_a_neighbours_body_is_respected_too():
    """
    Artefacts in this world are placed in pairs - a green alien standing beside a crystal
    cluster - so a point that stands off the alien properly can sit inside the crystal. That's
    why the check looks at every artefact rather than only the one being approached.

    Uses a 5m separation so the two sides of the alien give different answers: on the crystal's
    side the point is ruled out by the crystal, and on the far side it is clear. (The real pairs
    in this world sit closer than that - see the tightly packed case below.)
    """

    alien = cluster('green_alien', 0.0, 0.0)
    crystals = cluster('green_crystals', 5.0, 0.0)

    alien_only = make_node([alien])
    both = make_node([alien, crystals])
    standoff = alien_only.artifact_standoff_distance('green_alien')

    # Away from the crystal: a good standoff point either way.
    assert alien_only.is_clear_of_artifacts(-standoff, 0.0)
    assert both.is_clear_of_artifacts(-standoff, 0.0)

    # Towards the crystal: fine for the alien alone, but inside the crystal's body.
    assert alien_only.is_clear_of_artifacts(standoff, 0.0)
    assert not both.is_clear_of_artifacts(standoff, 0.0)


def test_a_small_artefact_beside_a_big_one_needs_a_wider_orbit():
    """
    The tightly packed case, and the reason both standoff searches step outwards instead of
    giving up: the green alien and the crystal cluster were detected live only 0.67m apart, at
    (7.9, 3.7) and (7.6, 4.3). The crystal's body plus the rover's width reaches 3.31m, which
    swallows the alien's entire 2.31m standoff ring - so NO point at the alien's nominal
    distance is usable, and a viewpoint only exists further out.

    Without the outward ladder in planner_inspect_artifact() and find_artifact_viewpoint(),
    this artefact gets no viewpoint at all: the inspection falls back to an unvalidated direct
    approach, and the tour silently drops the stop.
    """

    alien_x, alien_y = 7.9, 3.7
    node = make_node([cluster('green_alien', alien_x, alien_y),
                      cluster('green_crystals', 7.6, 4.3)])
    standoff = node.artifact_standoff_distance('green_alien')

    def clear_angles(distance):
        return [sample for sample in range(16)
                if node.is_clear_of_artifacts(
                    alien_x + distance * math.cos(2.0 * math.pi * sample / 16),
                    alien_y + distance * math.sin(2.0 * math.pi * sample / 16))]

    assert clear_angles(standoff) == [], \
        'expected the crystal to rule out the whole nominal ring around the alien'

    # The largest step the two searches take (standoff + 2.0) has to actually find something,
    # or the ladder is too short to help here.
    assert clear_angles(standoff + 2.0), \
        'the outward ladder found no clear viewpoint either - it is too short'


def test_unconfirmed_detections_do_not_block_goals():
    """
    A single bad frame must not be able to fence off real space. Only clusters confirmed by
    ARTIFACT_CONFIRMATION_OBSERVATIONS observations count.
    """

    unconfirmed = cluster('green_crystals', 8.2, 5.2, observations=1)
    assert make_node([unconfirmed]).is_clear_of_artifacts(8.2, 5.2)

    confirmed = cluster('green_crystals', 8.2, 5.2)
    assert not make_node([confirmed]).is_clear_of_artifacts(8.2, 5.2)


def test_empty_world_accepts_everything():
    """No artefacts known yet (e.g. before the first detection) must not block exploration."""

    node = make_node([])
    assert node.is_clear_of_artifacts(0.0, 0.0)
    assert node.is_clear_of_artifacts(-37.5, 112.0)


# --------------------------------------------------------------------------------------------
# Passage width: the clearance check above needs perception to have confirmed the artefact,
# so it cannot protect a bottleneck stop in a part of the cave the cameras haven't resolved.
# The width gate is the backstop that doesn't depend on perception at all.
# --------------------------------------------------------------------------------------------

def test_passage_gate_rejects_anything_narrower_than_the_rover():
    """
    The rover is a 1x1m box; it needs a free disc of its footprint radius to sit anywhere.
    The 0.6m passage that was toured and missed in testing must now be refused outright.
    """

    assert 0.6 < cave_explorer.MIN_TOUR_PASSAGE_WIDTH_M, \
        'the 0.6m passage that wedged the rover would still be accepted as a tour stop'
    assert cave_explorer.MIN_TOUR_PASSAGE_WIDTH_M == \
        pytest.approx(2.0 * cave_explorer.ROVER_FOOTPRINT_RADIUS_M), \
        'the gate must stay tied to the footprint, not drift into a magic number'


def test_passage_gate_still_allows_genuinely_traversable_passages():
    """
    The gate must not quietly delete the whole bottleneck category - Advanced 5 is supposed to
    tour narrow passages, just not impossible ones. A passage comfortably wider than the rover
    still has to qualify.
    """

    assert cave_explorer.MIN_TOUR_PASSAGE_WIDTH_M < 2.0, \
        'a 2m-wide passage is plainly drivable and must still be eligible'
    assert cave_explorer.MIN_TOUR_PASSAGE_WIDTH_M >= 1.0, \
        'the rover body alone is 1m across; the gate must be at least that'


# --------------------------------------------------------------------------------------------
# The constants, measured against the world files
# --------------------------------------------------------------------------------------------

def mesh_bounding_box(obj_path):
    """Min/max per axis over the mesh's 'v' lines."""

    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    with open(obj_path, 'r', errors='replace') as mesh_file:
        for line in mesh_file:
            if line[:2] == 'v ':
                parts = line.split()
                for axis in range(3):
                    value = float(parts[axis + 1])
                    lo[axis] = min(lo[axis], value)
                    hi[axis] = max(hi[axis], value)
    return lo, hi


def artifact_models_in_world():
    """
    Every artifact_* model in mars_cave.sdf, with the world-space footprint and height implied
    by its mesh and the <scale>/<pose> it's placed with.

    All of them are placed with roll = pi/2, which maps the mesh's +Y axis onto world +Z - so
    the model's height comes from the mesh's Y extent and its horizontal footprint from the X
    and Z extents, not the other way round.
    """

    root = ET.parse(WORLD_FILE).getroot()
    bbox_cache = {}

    for model in root.iter('model'):
        name = model.get('name', '')
        if not name.startswith('artifact'):
            continue
        pose = model.find('pose')
        mesh = model.find('./link/visual/geometry/mesh')
        if pose is None or mesh is None:
            continue

        numbers = [float(value) for value in pose.text.split()]
        pose_z = numbers[2]
        roll = numbers[3] if len(numbers) > 3 else 0.0
        scale = [float(value) for value in mesh.find('scale').text.split()]

        uri = mesh.find('uri').text.strip()
        relative = uri.split('/worlds/', 1)[-1]
        obj_path = os.path.join(PACKAGE_DIR, 'worlds', *relative.split('/'))
        assert os.path.exists(obj_path), f'{name}: mesh missing at {obj_path}'

        if obj_path not in bbox_cache:
            bbox_cache[obj_path] = mesh_bounding_box(obj_path)
        lo, hi = bbox_cache[obj_path]

        if abs(roll - math.pi / 2) < 1e-3:
            height = (hi[1] - lo[1]) * scale[1]
            base_offset = lo[1] * scale[1]
            footprint = ((hi[0] - lo[0]) * scale[0], (hi[2] - lo[2]) * scale[2])
        else:
            height = (hi[2] - lo[2]) * scale[2]
            base_offset = lo[2] * scale[2]
            footprint = ((hi[0] - lo[0]) * scale[0], (hi[1] - lo[1]) * scale[1])

        yield {
            'name': name,
            'mesh': os.path.basename(obj_path),
            'reach_m': max(footprint) / 2.0,
            'top_z_m': pose_z + base_offset + height,
        }


@pytest.fixture(scope='module')
def world_models():
    models = list(artifact_models_in_world())
    assert models, 'no artifact_* models found in mars_cave.sdf'
    return models


def test_radius_table_covers_every_model_in_the_world(world_models):
    """
    Re-measure every artefact model from its own mesh and the world file, and check the radius
    table doesn't under-state any of them.

    Keyed by mesh rather than by label, because the label is perception's name for the thing
    and the mesh is the thing - matching them up is exactly where a table like this goes stale
    (a model swapped or rescaled in the world, with the table left behind).
    """

    mesh_to_label = {
        'green_crystals_game_ready_asset.obj': 'green_crystals',
        'ice_castles_ny__ice_formations.obj': 'ice_formation',
        'green_alien.obj': 'green_alien',
        'mushrooms_blue.obj': 'mushroom_blue',
        'white_sphere.obj': 'white_sphere',
    }

    unmapped = {model['mesh'] for model in world_models} - set(mesh_to_label)
    assert not unmapped, (
        f'world has artefact meshes this test has no label for: {sorted(unmapped)} - '
        f'add them here and to ARTIFACT_PHYSICAL_RADIUS_M')

    for model in world_models:
        label = mesh_to_label[model['mesh']]
        tabled = cave_explorer.ARTIFACT_PHYSICAL_RADIUS_M.get(
            label, cave_explorer.ARTIFACT_DEFAULT_PHYSICAL_RADIUS_M)
        assert tabled >= model['reach_m'] - 1e-6, (
            f"{model['name']} ({model['mesh']}) reaches {model['reach_m']:.2f}m from its centre "
            f"but ARTIFACT_PHYSICAL_RADIUS_M['{label}'] is {tabled:.2f}m")


def test_standoff_is_outside_every_real_model(world_models):
    """End to end on the real world: the standoff the code picks clears the real geometry."""

    node = make_node([])
    mesh_to_label = {
        'green_crystals_game_ready_asset.obj': 'green_crystals',
        'ice_castles_ny__ice_formations.obj': 'ice_formation',
        'green_alien.obj': 'green_alien',
        'mushrooms_blue.obj': 'mushroom_blue',
        'white_sphere.obj': 'white_sphere',
    }

    for model in world_models:
        standoff = node.artifact_standoff_distance(mesh_to_label[model['mesh']])
        needed = model['reach_m'] + cave_explorer.ROVER_FOOTPRINT_RADIUS_M
        assert standoff >= needed, (
            f"{model['name']}: standoff {standoff:.2f}m vs {needed:.2f}m needed to keep the "
            f"rover's corners out of the model")


def test_every_artefact_crosses_the_lidar_plane(world_models):
    """
    Documents why the clearance check can't just be left to the map, and guards the assumption
    behind it: every artefact is tall enough to be hit by the single horizontal scan at
    z = 1.0m, so they are all at least partly visible. The problem is never that they're
    invisible, it's that the thin spiky ones are seen so sparsely that the gaps between their
    spikes read as free space. A model that failed this would be invisible outright - a
    different and much worse problem than the one being fixed here.
    """

    for model in world_models:
        assert model['top_z_m'] >= LIDAR_HEIGHT_M, (
            f"{model['name']} tops out at {model['top_z_m']:.2f}m, entirely below the "
            f'{LIDAR_HEIGHT_M}m lidar plane - it would be invisible to SLAM')
