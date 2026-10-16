#!/usr/bin/env python3
"""
Tests that only labels the trained classifier stood behind vote on what an artefact is.

Why this exists: when the classifier is unsure (below CLASSIFIER_CONFIDENCE_THRESHOLD), the raw
HSV label passes through - and HSV is exactly what confuses green_alien with green_crystals.
Live, the real alien at (54.1, 32.7) was confirmed as green_alien with classifier votes at
0.60-0.69, then flipped to green_crystals 0.4s later on silent HSV-only frames, and the rover
diverted to "inspect" it. Unsure frames now still place the artefact but don't vote.

Run with:  python -m pytest src/cave_explorer/test/test_label_votes.py -v
"""

import types

import pytest

cave_explorer = pytest.importorskip(
    'cave_explorer.cave_explorer',
    reason='needs a sourced ROS 2 environment (rclpy, cv_bridge)')

from geometry_msgs.msg import Point  # noqa: E402

CaveExplorer = cave_explorer.CaveExplorer
Detection = cave_explorer.Detection
CONFIRMATIONS = cave_explorer.ARTIFACT_CONFIRMATION_OBSERVATIONS
ALIEN = Point(x=54.1, y=32.7, z=0.0)


def make_node():
    node = types.SimpleNamespace()
    node.artifact_clusters_ = []
    node.next_artifact_cluster_id_ = 0
    node.get_logger = lambda: types.SimpleNamespace(info=lambda m: None, warn=lambda m: None)
    return node


def observe(node, label, vote, times=1):
    for _ in range(times):
        CaveExplorer.add_artifact_observation(node, label, ALIEN, label_vote=vote)


def test_unsure_hsv_frames_cannot_flip_a_classifier_confirmed_alien():
    node = make_node()
    observe(node, 'green_alien', vote=True, times=CONFIRMATIONS)
    observe(node, 'green_crystals', vote=False, times=CONFIRMATIONS * 3)

    [cluster] = node.artifact_clusters_
    assert cluster['label'] == 'green_alien'
    assert cluster['num_observations'] == CONFIRMATIONS * 4  # still used for position


def test_a_cluster_started_by_an_unsure_frame_takes_the_first_real_vote():
    node = make_node()
    observe(node, 'green_crystals', vote=False)
    observe(node, 'green_alien', vote=True)

    assert node.artifact_clusters_[0]['label'] == 'green_alien'


def test_classifier_votes_still_decide_by_majority():
    node = make_node()
    observe(node, 'green_crystals', vote=True, times=2)
    observe(node, 'green_alien', vote=True, times=3)

    assert node.artifact_clusters_[0]['label'] == 'green_alien'


def test_a_classifier_relabel_or_agreement_is_marked_as_a_vote():
    node = types.SimpleNamespace(classifier_interpreter_=object(),
                                 get_logger=lambda: types.SimpleNamespace(info=lambda m: None))
    box = (100, 100, 80, 80)

    def refine(verdict, label):
        node.classify_crop = lambda image, bbox: verdict
        return CaveExplorer.refine_detections_with_classifier(
            node, None, [Detection(label, box, (0, 255, 0))])

    threshold = cave_explorer.CLASSIFIER_CONFIDENCE_THRESHOLD
    assert refine(('green_alien', threshold + 0.05), 'green_crystals')[0].verified
    assert refine(('green_alien', threshold + 0.05), 'green_alien')[0].verified
    assert not refine(('green_alien', threshold - 0.05), 'green_crystals')[0].verified


def test_a_close_view_outweighs_many_distant_ones():
    """
    Live: the alien at (54.2, 32.2) read green_crystals from afar, green_alien up close, then
    flipped back as distant frames piled up while the rover drove away. Votes are weighted by box
    area, so a handful of close-up readings beat many small distant ones.
    """

    node = make_node()
    distant, close = 40 * 60, 160 * 240
    observe(node, 'green_crystals', vote=distant, times=10)
    observe(node, 'green_alien', vote=close, times=3)
    observe(node, 'green_crystals', vote=distant, times=10)

    assert node.artifact_clusters_[0]['label'] == 'green_alien'
