#!/usr/bin/env python3
"""
Tests that the stop sign exclusion neither deletes real artefacts nor lets ghosts through.

Why this exists: one live run hit both failure modes of remember_stop_sign_position().

  * A real white_sphere (#3, at (45.8, 35.5); mars_cave.sdf has one at (46.7, 35.9)) was
    inspected, then discarded a second later as "turned out to be this stop sign". The nearest
    real stop sign is ~18m away. A single cascade misfire at close range - with no depth there,
    so the rover's own position stood in for the sign - was trusted outright, purged the sphere
    and blanked out its spot for the rest of the mission.
  * An ice_formation ghost (#12, at (30.1, 27.1)) was confirmed right after the classifier
    recognised the stop sign on wall_61 (29.25, 28.35). It landed 1.51m from the sign, just
    outside the old 1.5m exclusion radius.

The fix is STOP_SIGN_CONFIRMATION_OBSERVATIONS (a sign needs several sightings before it
excludes or purges anything) plus a wider STOP_SIGN_EXCLUSION_RADIUS_M.

Run with:  python -m pytest src/cave_explorer/test/test_stop_sign_confirmation.py -v
"""

import math
import types

import pytest

cave_explorer = pytest.importorskip(
    'cave_explorer.cave_explorer',
    reason='needs a sourced ROS 2 environment (rclpy, cv_bridge)')

from geometry_msgs.msg import Point  # noqa: E402  (only importable once ROS is sourced)

CaveExplorer = cave_explorer.CaveExplorer
CONFIRMATIONS = cave_explorer.STOP_SIGN_CONFIRMATION_OBSERVATIONS
RADIUS = cave_explorer.STOP_SIGN_EXCLUSION_RADIUS_M


class FakeLogger:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)


def make_node(clusters=()):
    """A stand-in carrying only the state remember_stop_sign_position() reads and writes."""

    node = types.SimpleNamespace()
    node.stop_sign_positions_ = []
    node.stop_sign_candidates_ = []
    node.stop_sign_counts_ = []
    node.artifact_clusters_ = [
        {'id': cid, 'label': label, 'position': Point(x=x, y=y, z=0.0), 'num_observations': 1}
        for cid, label, x, y in clusters]
    node.visited_artifact_ids_ = set()
    node.abandoned_artifact_ids_ = set()
    node._logger = FakeLogger()
    node.get_logger = lambda: node._logger
    node.purge_artifacts_near_stop_sign = lambda i: CaveExplorer.purge_artifacts_near_stop_sign(node, i)
    return node


def sight(node, x, y, times=1):
    for _ in range(times):
        CaveExplorer.remember_stop_sign_position(node, Point(x=x, y=y, z=0.0))


def cluster_ids(node):
    return {c['id'] for c in node.artifact_clusters_}


def test_a_single_misfire_does_not_purge_the_inspected_white_sphere():
    # The live incident: sphere #3 confirmed and inspected, then one cascade hit at the
    # rover's standoff position ~2m from it.
    node = make_node(clusters=[(3, 'white_sphere', 45.8, 35.5)])
    node.visited_artifact_ids_ = {3}

    sight(node, 46.4, 33.6)

    assert cluster_ids(node) == {3}
    assert 3 in node.visited_artifact_ids_
    assert node.stop_sign_positions_ == []
    assert not any('STOP SIGN' in line for line in node._logger.lines)


def test_an_unconfirmed_sighting_does_not_exclude_future_observations():
    # localise_artifacts() only checks stop_sign_positions_, so a candidate must not land there
    node = make_node()

    sight(node, 46.4, 33.6, times=CONFIRMATIONS - 1)

    assert node.stop_sign_positions_ == []
    assert len(node.stop_sign_candidates_) == 1


def test_a_real_sign_seen_repeatedly_purges_the_ghost_just_outside_the_old_radius():
    # The live incident: ice_formation ghost #12 at 1.51m from the wall_61 stop sign
    node = make_node(clusters=[(12, 'ice_formation', 30.1, 27.1)])
    assert math.hypot(30.1 - 29.25, 27.1 - 28.35) > 1.5

    sight(node, 29.25, 28.35, times=CONFIRMATIONS)

    assert cluster_ids(node) == set()
    assert 12 not in node.visited_artifact_ids_
    assert len(node.stop_sign_positions_) == 1
    assert node.stop_sign_candidates_ == []
    assert any('discarding ice_formation #12' in line for line in node._logger.lines)


def test_noisy_sightings_of_one_sign_merge_into_one_confirmed_position():
    node = make_node()

    sight(node, 29.0, 28.0)
    sight(node, 29.5, 28.7)
    sight(node, 29.25, 28.35)
    sight(node, 29.3, 28.4, times=5)  # further sightings dedupe against the confirmed one

    assert len(node.stop_sign_positions_) == 1
    sign = node.stop_sign_positions_[0]
    assert math.hypot(sign.x - 29.25, sign.y - 28.35) < 0.1
    assert node.stop_sign_candidates_ == []


def test_sightings_of_two_far_apart_signs_are_counted_separately():
    node = make_node()

    for _ in range(CONFIRMATIONS - 1):
        sight(node, 29.25, 28.35)
        sight(node, 23.35, 4.7)

    assert node.stop_sign_positions_ == []
    assert len(node.stop_sign_candidates_) == 2


def test_the_radius_leaves_every_real_artefact_clear_of_every_stop_sign():
    # Stop sign walls and artefact positions from mars_cave.sdf. If the world changes so that
    # a real artefact sits within the radius, the exclusion would hide it.
    stop_signs = [(45.55, 18.2), (7.95, 39.35), (29.25, 28.35), (52.7, 47.15), (23.35, 4.7)]
    artefacts = [(-5.4, -0.3), (8.2, 5.2), (52.0, 5.4), (33.7, 10.6), (20.0, 10.2),
                 (19.1, 20.8), (-3.3, 21.4), (2.9, 23.3), (18.0, 30.2), (-4.7, 30.2),
                 (54.4, 32.7), (19.8, 35.8), (46.7, 35.9), (-4.0, 39.0), (33.5, 45.9)]

    nearest = min(math.hypot(sx - ax, sy - ay) for sx, sy in stop_signs for ax, ay in artefacts)

    assert nearest > 2 * RADIUS


# --- The cascade's own hits are checked by the trained classifier -------------------------------
#
# A second live run (after the 3-sighting rule) still purged five real artefacts - green
# crystals at (7.6, 4.3) and (18.7, 36.1), the green alien at (18.9, 20.8) and the white spheres
# at (46.2, 35.1) and (3.2, 22.4) - none of them within 10m of a stop sign. The cascade behind
# detect_stop_signs() misfires on them for several frames in a row, which clears 3 sightings
# easily. Its hits were exempt from the classifier; now they go through it too, and only a hit
# the classifier itself calls a stop sign is remembered.

Detection = cave_explorer.Detection
CONFIDENT = cave_explorer.STOP_SIGN_VERIFY_CONFIDENCE + 0.05
BIG_BOX = (100, 100, 80, 80)  # above CLASSIFIER_MIN_CROP_AREA_PX


def make_classifier_node(verdict):
    node = types.SimpleNamespace()
    node.classifier_interpreter_ = object()
    node.classify_crop = lambda image, bbox: verdict
    node._logger = FakeLogger()
    node.get_logger = lambda: node._logger
    return node


def refine(verdict, detection):
    node = make_classifier_node(verdict)
    return CaveExplorer.refine_detections_with_classifier(node, None, [detection])


def test_a_cascade_hit_the_classifier_calls_a_white_sphere_becomes_one():
    result = refine(('white_sphere', CONFIDENT), Detection('stop_sign', BIG_BOX, (0, 255, 0)))

    assert [d.label for d in result] == ['white_sphere']
    assert result[0].verified  # a classifier-backed white_sphere vote, not a stop sign


def test_a_cascade_hit_the_classifier_calls_background_is_dropped():
    assert refine(('negative', CONFIDENT), Detection('stop_sign', BIG_BOX, (0, 255, 0))) == []


def test_a_cascade_hit_the_classifier_agrees_with_is_verified():
    result = refine(('stop_sign', CONFIDENT), Detection('stop_sign', BIG_BOX, (0, 255, 0)))

    assert [d.label for d in result] == ['stop_sign']
    assert result[0].verified


def test_an_hsv_fragment_the_classifier_calls_a_stop_sign_is_verified():
    result = refine(('stop_sign', CONFIDENT), Detection('ice_formation', BIG_BOX, (255, 255, 255)))

    assert [(d.label, d.verified) for d in result] == [('stop_sign', True)]


def test_an_unsure_or_too_small_cascade_hit_stays_unverified():
    unsure = refine(('stop_sign', CONFIDENT - 0.2), Detection('stop_sign', BIG_BOX, (0, 255, 0)))
    small = refine(('stop_sign', CONFIDENT), Detection('stop_sign', (100, 100, 20, 20), (0, 255, 0)))

    assert [(d.label, d.verified) for d in unsure] == [('stop_sign', False)]
    assert [(d.label, d.verified) for d in small] == [('stop_sign', False)]


def test_only_verified_stop_signs_are_remembered():
    """
    localise_artifacts() is driven with a stubbed projection: an unverified cascade hit right on
    top of a confirmed white sphere, repeated well past the confirmation count, must not purge it.
    """

    node = make_node(clusters=[(11, 'white_sphere', 46.2, 35.1)])
    node.classifier_interpreter_ = object()
    node.latest_depth_image_ = object()
    node.tf_buffer = types.SimpleNamespace(lookup_transform=lambda *args: None)
    node.project_detection_to_map = lambda d, depth, tf: Point(x=46.2, y=35.1, z=0.0)
    node.remember_stop_sign_position = lambda p: CaveExplorer.remember_stop_sign_position(node, p)
    node.purge_artifacts_near_stop_sign = lambda i: CaveExplorer.purge_artifacts_near_stop_sign(node, i)
    node.add_artifact_observation = lambda label, p, label_vote=True: None
    node.publish_artifact_markers = lambda: None

    for _ in range(CONFIRMATIONS * 3):
        CaveExplorer.localise_artifacts(node, [Detection('stop_sign', BIG_BOX, (0, 255, 0))])
    assert cluster_ids(node) == {11}

    for _ in range(CONFIRMATIONS):
        CaveExplorer.localise_artifacts(
            node, [Detection('stop_sign', BIG_BOX, (0, 255, 0), verified=True)])
    assert cluster_ids(node) == set()


def test_a_borderline_stop_sign_verdict_is_not_verified():
    """
    Live: the cascade fired on part of the green alien at (18.8, 20.8) and the classifier agreed
    "stop_sign" at ordinary confidence on 3 frames, which purged the alien. Such a hit still
    suppresses overlapping detections that frame, but can no longer delete anything.
    """

    borderline = (cave_explorer.CLASSIFIER_CONFIDENCE_THRESHOLD
                  + cave_explorer.STOP_SIGN_VERIFY_CONFIDENCE) / 2
    cascade = refine(('stop_sign', borderline), Detection('stop_sign', BIG_BOX, (0, 255, 0)))
    fragment = refine(('stop_sign', borderline), Detection('ice_formation', BIG_BOX, (255, 255, 255)))

    assert [(d.label, d.verified) for d in cascade] == [('stop_sign', False)]
    assert [(d.label, d.verified) for d in fragment] == [('stop_sign', False)]


def test_a_distant_sign_without_depth_is_not_placed_at_the_rover():
    """
    Live: a real stop sign ~16m away (verified at confidence 1.00) had no depth reading - too far
    for the depth camera - so it was placed at the rover's position, right next to the white
    sphere at (46.2, 34.6) it had just inspected, which was purged. Only a sign filling a big part
    of the image (the rover right up against it) may use the rover's position.
    """

    node = make_node(clusters=[(7, 'white_sphere', 46.2, 34.6)])
    node.classifier_interpreter_ = object()
    node.latest_depth_image_ = object()
    node.tf_buffer = types.SimpleNamespace(lookup_transform=lambda *args: None)
    node.project_detection_to_map = lambda d, depth, tf: None  # no valid depth
    node.get_pose_2d = lambda: types.SimpleNamespace(x=45.5, y=33.5, theta=0.0)
    node.remember_stop_sign_position = lambda p: CaveExplorer.remember_stop_sign_position(node, p)
    node.purge_artifacts_near_stop_sign = lambda i: CaveExplorer.purge_artifacts_near_stop_sign(node, i)
    node.add_artifact_observation = lambda label, p, label_vote=True: None
    node.publish_artifact_markers = lambda: None

    far = (300, 200, 40, 40)
    for _ in range(CONFIRMATIONS * 3):
        CaveExplorer.localise_artifacts(node, [Detection('stop_sign', far, (0, 255, 0), verified=True)])
    assert cluster_ids(node) == {7}

    near = (200, 100, cave_explorer.STOP_SIGN_FALLBACK_MIN_WIDTH_PX, 200)
    for _ in range(CONFIRMATIONS):
        CaveExplorer.localise_artifacts(node, [Detection('stop_sign', near, (0, 255, 0), verified=True)])
    assert cluster_ids(node) == set()


def test_a_sign_only_removes_clusters_it_out_sees():
    """
    Both failure directions seen live. A real artefact the cascade misfired on (green crystals,
    a dozen-plus observations) must survive a handful of false sign sightings. A ghost fragment
    of a real sign misread as white_sphere (a few observations) must go once the real sign has
    been seen more - it reads 1.00 on dozens of frames as the rover passes.
    """

    node = make_node(clusters=[(12, 'green_crystals', 18.7, 36.0),
                               (10, 'white_sphere', 18.9, 36.3)])
    node.artifact_clusters_[0]['num_observations'] = 15
    node.artifact_clusters_[1]['num_observations'] = 5
    node.visited_artifact_ids_ = {12, 10}  # inspection alone protects nothing

    sight(node, 18.8, 36.1, times=CONFIRMATIONS)
    assert cluster_ids(node) == {12, 10}

    sight(node, 18.8, 36.1, times=3)  # 6 sightings > 5 observations
    assert cluster_ids(node) == {12}

    sight(node, 18.8, 36.1, times=9)  # 15, still not more than the crystals' 15
    assert cluster_ids(node) == {12}
