#!/usr/bin/env python3
"""
Tests that a target which turns out up close not to be an inspection type isn't counted.

Why this exists: the green alien at (54.4, 32.7) reads as green_crystals from a distance in every
run. The rover diverted to it, the classifier saw the alien up close, and it was still recorded
as "visited" - inflating the inspection count with an artefact type that isn't inspected.

Run with:  python -m pytest src/cave_explorer/test/test_dismissed_targets.py -v
"""

import ast
import inspect
import textwrap
import types

import pytest

cave_explorer = pytest.importorskip(
    'cave_explorer.cave_explorer',
    reason='needs a sourced ROS 2 environment (rclpy, cv_bridge)')

from geometry_msgs.msg import Point  # noqa: E402

CaveExplorer = cave_explorer.CaveExplorer
CONFIRMED = cave_explorer.ARTIFACT_CONFIRMATION_OBSERVATIONS


def cluster(cid, label, x, y):
    return {'id': cid, 'label': label, 'num_observations': CONFIRMED,
            'position': Point(x=x, y=y, z=0.0), 'label_votes': {label: 1}}


def make_node(clusters):
    node = types.SimpleNamespace(artifact_clusters_=clusters, visited_artifact_ids_=set(),
                                 abandoned_artifact_ids_=set(), dismissed_artifact_ids_=set())
    node.is_duplicate_of_handled_artifact = types.MethodType(
        CaveExplorer.is_duplicate_of_handled_artifact, node)
    return node


def test_a_dismissed_cluster_is_never_targeted_again_even_if_its_label_flips_back():
    alien = cluster(9, 'green_crystals', 54.4, 32.6)  # misread again from afar
    node = make_node([alien])
    node.dismissed_artifact_ids_ = {9}

    assert CaveExplorer.find_inspection_target(node) is None


def test_a_duplicate_cluster_on_a_dismissed_object_is_dismissed_too():
    node = make_node([cluster(9, 'green_alien', 54.4, 32.6),
                      cluster(14, 'green_crystals', 54.9, 33.0)])
    node.dismissed_artifact_ids_ = {9}

    assert CaveExplorer.find_inspection_target(node) is None
    assert 14 in node.dismissed_artifact_ids_
    assert not node.visited_artifact_ids_


def test_main_loop_dismisses_before_it_counts_a_visit():
    source = textwrap.dedent(inspect.getsource(CaveExplorer.main_loop))
    tree = ast.parse(source)
    dismiss = visit = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and any(
                isinstance(n, ast.Name) and n.id == 'INSPECTION_ARTIFACT_LABELS'
                for n in ast.walk(node.test)):
            dismiss = node
            visit = node.orelse[0] if node.orelse else None
            break

    assert dismiss is not None, 'the INSPECT_ARTIFACT branch no longer checks the label on arrival'
    assert 'dismissed_artifact_ids_' in ast.unparse(dismiss.body[0]) + ast.unparse(dismiss)
    assert 'visited_artifact_ids_.add' not in ast.unparse(ast.Module(body=dismiss.body, type_ignores=[]))
    assert visit is not None and 'visited_artifact_ids_.add' in ast.unparse(visit)
