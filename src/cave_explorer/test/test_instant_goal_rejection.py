#!/usr/bin/env python3
"""
Tests that a goal Nav2 threw out before the rover moved is not scored as a failure to reach.

Why this exists: 14 of the 62 goals in a live run came back as 'failed (status=6) in 0s'. The
logs name the cause for 11 of them - "Timed out while waiting for action server to acknowledge
goal request for compute_path_to_pose" (and the same for follow_path). That is Nav2's behaviour
tree failing to get an acknowledgement from its own action servers inside
default_server_timeout, which was 20ms (the upstream default) on a machine running Gazebo with
software rendering. The goals themselves were ordinary reachable coordinates - two of the
instant rejections were only 0.8m and 2.4m away. Exactly one of the 14 was a genuine planning
failure ("GridBased failed to generate a valid path").

Two behaviours were reading those as facts about the world:

  * Planning 3 inspection. INSPECTION_MAX_RETRIES is 1, so two instant rejections - well under
    a second apart - permanently wrote an artefact off. green_crystals #62 was abandoned that
    way, after two identical rejections of the same 42.6m goal, without the rover ever moving.
  * Advanced 5 tour stops, which were recorded 'missed'. Of the 3 misses in that run, 2 were
    instant rejections at 9.6m and 9.1m, so the honest result was 11/12 rather than 9/12.

The real fix is default_server_timeout in nav2_params.yaml; this is the robustness half, so a
hiccup inside the navigation stack cannot cost a science target. Both paths stay bounded, since
a genuinely unplannable goal also fails instantly and would otherwise retry forever.

Run with:  python -m pytest src/cave_explorer/test/test_instant_goal_rejection.py -v
"""

import pathlib
import re
import types

import pytest

cave_explorer = pytest.importorskip(
    'cave_explorer.cave_explorer',
    reason='needs a sourced ROS 2 environment (rclpy, cv_bridge)')

PlannerType = cave_explorer.PlannerType
INSTANT_S = cave_explorer.GOAL_INSTANT_FAILURE_S
MAX_INSTANT = cave_explorer.INSPECTION_MAX_INSTANT_RETRIES


class FakeLogger:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)

    def warn(self, msg):
        self.lines.append(msg)


def make_inspection_node(duration_s, instant_failures=0, attempts=0, cancelled_by_pause=False):
    """
    A stand-in carrying only the state the INSPECT_ARTIFACT branch of main_loop() reads when a
    goal has just concluded. Only that branch is exercised - calling main_loop() itself would
    need a ROS graph, a map and an action server.
    """

    node = types.SimpleNamespace()
    node.inspection_target_ = {'id': 62, 'label': 'green_crystals'}
    node.inspection_attempts_ = attempts
    node.inspection_instant_failures_ = instant_failures
    node.last_goal_succeeded_ = False
    node.last_goal_duration_s_ = duration_s
    node.goal_cancelled_by_pause_ = cancelled_by_pause
    node.visited_artifact_ids_ = set()
    node.abandoned_artifact_ids_ = set()
    node._logger = FakeLogger()
    node.get_logger = lambda: node._logger
    return node


def resolve_inspection_outcome(node):
    """
    The decision the INSPECT_ARTIFACT branch makes, as a function of the state above. Kept in
    step with cave_explorer.py by test_the_source_still_branches_on_the_instant_case below,
    which reads the real source rather than trusting this copy.
    """

    if node.goal_cancelled_by_pause_:
        node.goal_cancelled_by_pause_ = False
    elif node.last_goal_succeeded_:
        node.visited_artifact_ids_.add(node.inspection_target_['id'])
        node.inspection_target_ = None
        node.inspection_attempts_ = 0
        node.inspection_instant_failures_ = 0
    elif node.last_goal_duration_s_ < INSTANT_S \
            and node.inspection_instant_failures_ < MAX_INSTANT:
        node.inspection_instant_failures_ += 1
    else:
        node.inspection_attempts_ += 1
        if node.inspection_attempts_ > cave_explorer.INSPECTION_MAX_RETRIES:
            node.abandoned_artifact_ids_.add(node.inspection_target_['id'])
            node.inspection_target_ = None
            node.inspection_attempts_ = 0
            node.inspection_instant_failures_ = 0
    return node


# --------------------------------------------------------------------------------------------
# Planning 3: inspection
# --------------------------------------------------------------------------------------------

def test_the_case_that_lost_green_crystals_62_now_survives():
    """Two instant rejections in a row must not write the artefact off."""

    node = make_inspection_node(duration_s=0.0)
    resolve_inspection_outcome(node)
    resolve_inspection_outcome(node)

    assert node.abandoned_artifact_ids_ == set(), \
        'the artefact was abandoned without the rover ever having driven towards it'
    assert node.inspection_target_ is not None, 'the target should still be held for a retry'
    assert node.inspection_attempts_ == 0, \
        'an instant rejection must not consume one of the real approach attempts'
    assert node.inspection_instant_failures_ == 2


def test_an_instant_rejection_is_still_bounded():
    """A genuinely unplannable goal fails instantly every time, so it must stop retrying."""

    node = make_inspection_node(duration_s=0.0)
    for _ in range(MAX_INSTANT + cave_explorer.INSPECTION_MAX_RETRIES + 2):
        resolve_inspection_outcome(node)

    assert 62 in node.abandoned_artifact_ids_, \
        'an always-instantly-rejected goal must eventually be given up on'


def test_a_real_approach_failure_still_abandons_after_one_retry():
    """The brief-mandated behaviour (retry once, then abandon) is unchanged for real failures."""

    node = make_inspection_node(duration_s=45.0)
    resolve_inspection_outcome(node)
    assert node.abandoned_artifact_ids_ == set(), 'should still have one retry left'
    assert node.inspection_attempts_ == 1

    resolve_inspection_outcome(node)
    assert 62 in node.abandoned_artifact_ids_, 'second real failure should abandon'


def test_a_pause_still_takes_priority_over_the_instant_check_for_inspection():
    """
    A pause-cancelled approach goal also concludes fast (often in well under
    GOAL_INSTANT_FAILURE_S), so it must be read as the pause case, not as an instant rejection
    or a real failure - exactly the bug the matching tour-path test above guards against, just
    for Planning 3 inspection instead of Advanced 5 tour stops. Before this was fixed, a pause
    mid-approach was indistinguishable from Nav2 genuinely failing the goal, and could burn a
    real attempt (or even abandon the artefact outright) purely from the user pausing.
    """

    node = make_inspection_node(duration_s=0.0, cancelled_by_pause=True)
    resolve_inspection_outcome(node)

    assert node.inspection_target_ is not None, 'a pause must not give up on the target'
    assert node.inspection_attempts_ == 0, 'a pause is not a failed approach attempt'
    assert node.inspection_instant_failures_ == 0, 'a pause is not a Nav2 rejection either'
    assert node.goal_cancelled_by_pause_ is False, 'the pause flag must be consumed'


def test_a_slow_failure_counts_even_after_instant_ones():
    """Instant rejections don't buy extra real attempts."""

    node = make_inspection_node(duration_s=0.0)
    resolve_inspection_outcome(node)

    node.last_goal_duration_s_ = 45.0
    resolve_inspection_outcome(node)
    resolve_inspection_outcome(node)

    assert 62 in node.abandoned_artifact_ids_
    assert node.inspection_instant_failures_ == 0, 'reset when the target is written off'


def test_the_threshold_is_about_not_having_moved_not_about_being_quick():
    """
    A goal that ran for a second or more did drive, however briefly, so it is a real attempt.
    The constant has to sit well below any plausible real approach and well above a handshake.
    """

    assert 0.1 <= INSTANT_S <= 2.0, \
        'too small misses the 0s rejections; too large swallows genuine short approaches'

    node = make_inspection_node(duration_s=INSTANT_S)
    resolve_inspection_outcome(node)
    assert node.inspection_attempts_ == 1, 'at the threshold it must count as a real attempt'
    assert node.inspection_instant_failures_ == 0


# --------------------------------------------------------------------------------------------
# Advanced 5: tour stops
# --------------------------------------------------------------------------------------------

def make_tour_node(duration_s, succeeded=False, stop_index=0, num_stops=3):
    node = types.SimpleNamespace()
    stops = [{'x': float(i), 'y': 0.0, 'label': f'stop {i}', 'kind': 'artifact',
              'status': 'pending'} for i in range(num_stops)]
    node.tour_ = {
        'stops': stops,
        'optimised_order': list(range(num_stops)),
        'optimised_cost_m': 100.0,
        'baseline_cost_m': 110.0,
    }
    node.tour_stop_index_ = stop_index
    node.tour_complete_ = False
    node.tour_requested_ = True
    node.goal_cancelled_by_pause_ = False
    node.last_goal_succeeded_ = succeeded
    node.last_goal_duration_s_ = duration_s
    node.publish_tour_markers = lambda: None
    node._logger = FakeLogger()
    node.get_logger = lambda: node._logger
    node.advance_science_tour = types.MethodType(
        cave_explorer.CaveExplorer.advance_science_tour, node)
    return node


def test_a_tour_stop_rejected_instantly_is_re_sent_not_scored_as_missed():
    """The white_sphere #3 / ice_formation #9 case: 9.6m and 9.1m away, rejected in 0s."""

    node = make_tour_node(duration_s=0.0)
    node.advance_science_tour()

    stop = node.tour_['stops'][0]
    assert stop['status'] == 'pending', f"scored as {stop['status']} without being attempted"
    assert node.tour_stop_index_ == 0, 'the tour must not advance past a stop it never tried'
    assert stop['instant_retries'] == 1


def test_a_tour_stop_that_keeps_being_rejected_is_eventually_reported_missed():
    node = make_tour_node(duration_s=0.0)
    for _ in range(MAX_INSTANT + 1):
        node.advance_science_tour()

    assert node.tour_['stops'][0]['status'] == 'missed'
    assert node.tour_stop_index_ == 1, 'the tour has to move on rather than stall on one stop'


def test_a_tour_stop_the_rover_genuinely_failed_to_reach_is_still_missed():
    """green_crystals #16: 48s, failed backup recovery. A real miss, and reported as one."""

    node = make_tour_node(duration_s=48.0)
    node.advance_science_tour()

    assert node.tour_['stops'][0]['status'] == 'missed'
    assert node.tour_stop_index_ == 1


def test_a_reached_stop_is_unaffected():
    node = make_tour_node(duration_s=20.0, succeeded=True)
    node.advance_science_tour()

    assert node.tour_['stops'][0]['status'] == 'visited'
    assert node.tour_stop_index_ == 1


def test_a_pause_still_takes_priority_over_the_instant_check():
    """A pause-cancelled goal also concludes fast; it must stay the pause case, not a retry."""

    node = make_tour_node(duration_s=0.0)
    node.goal_cancelled_by_pause_ = True
    node.advance_science_tour()

    assert node.tour_stop_index_ == 0
    assert node.goal_cancelled_by_pause_ is False, 'the pause flag must be consumed'
    assert 'instant_retries' not in node.tour_['stops'][0], \
        'a pause is not a Nav2 rejection and must not spend a retry'


# --------------------------------------------------------------------------------------------
# The metric the report quotes
# --------------------------------------------------------------------------------------------

class FakeDuration:
    def __init__(self, nanoseconds):
        self.nanoseconds = nanoseconds


class FakeTime:
    """
    Stands in for an rclpy Time. Subtraction has to be defined on the CLASS - Python looks up
    __sub__ on the type, never on the instance, so a SimpleNamespace carrying a __sub__
    attribute is not subtractable.
    """

    def __init__(self, elapsed_s):
        self._elapsed_s = elapsed_s

    def __sub__(self, _other):
        return FakeDuration(int(self._elapsed_s * 1e9))


class FakeClock:
    def __init__(self, elapsed_s):
        self._elapsed_s = elapsed_s

    def now(self):
        return FakeTime(self._elapsed_s)


def make_logging_node(elapsed_s, goal_distance_m=42.6):
    node = types.SimpleNamespace()
    node.total_distance_m_ = 0.0
    node.last_goal_duration_s_ = None
    node.goal_initial_distance_m_ = goal_distance_m
    node.goal_sent_time_ = FakeTime(0.0)  # only its presence is checked
    node._logger = FakeLogger()
    node.get_logger = lambda: node._logger
    node.get_clock = lambda: FakeClock(elapsed_s)
    node.log_goal_outcome = types.MethodType(
        cave_explorer.CaveExplorer.log_goal_outcome, node)
    return node


def test_a_goal_that_never_moved_contributes_nothing_to_the_mission_distance():
    """
    green_crystals #62's 42.6m goal was rejected twice in 0s and added 85.2m the rover never
    drove, inflating the average speed the report quotes.
    """

    node = make_logging_node(elapsed_s=0.05)
    node.log_goal_outcome('failed (status=6)')

    assert node.total_distance_m_ == 0.0, 'credited distance for a goal that never started'
    assert node.last_goal_duration_s_ == pytest.approx(0.05)
    assert not any('m/s' in line for line in node._logger.lines), \
        'must not report a speed derived from a zero-duration goal'


def test_a_normal_goal_still_counts_towards_the_mission_distance():
    node = make_logging_node(elapsed_s=12.0, goal_distance_m=7.9)
    node.log_goal_outcome('reached')

    assert node.total_distance_m_ == pytest.approx(7.9)
    assert node.last_goal_duration_s_ == pytest.approx(12.0)
    assert any('avg 0.66 m/s' in line for line in node._logger.lines)


# --------------------------------------------------------------------------------------------
# The two halves have to stay in step with the real source
# --------------------------------------------------------------------------------------------

def test_the_source_still_branches_on_the_instant_case():
    """
    resolve_inspection_outcome() above is a copy of logic that lives in main_loop(), which is
    too entangled with ROS to call directly. Guard against the copy silently going stale.
    """

    source = pathlib.Path(cave_explorer.__file__).read_text(encoding='utf-8')
    branch = re.search(
        r'elif self\.last_goal_duration_s_ < GOAL_INSTANT_FAILURE_S\s*\\?\s*\n?\s*'
        r'and self\.inspection_instant_failures_ < INSPECTION_MAX_INSTANT_RETRIES',
        source)
    assert branch is not None, \
        'main_loop no longer special-cases an instantly-rejected inspection goal'


def test_the_source_still_checks_for_a_pause_before_counting_an_inspection_failure():
    """
    Guards the pause-priority branch resolve_inspection_outcome() now also copies - without
    this, the real INSPECT_ARTIFACT branch could silently lose its pause check again (e.g. a
    future refactor) while this file's copy, and every test above using it, kept passing.
    """

    source = pathlib.Path(cave_explorer.__file__).read_text(encoding='utf-8')
    branch = re.search(
        r'if self\.planner_type_ == PlannerType\.INSPECT_ARTIFACT:.{0,400}?'
        r'if self\.goal_cancelled_by_pause_:',
        source, re.DOTALL)
    assert branch is not None, \
        'main_loop no longer checks goal_cancelled_by_pause_ before scoring an INSPECT_ARTIFACT outcome'


def test_nav2_is_no_longer_given_20ms_to_answer_itself():
    """
    The root cause. 20ms is the upstream default and is what produced the 0s rejections on this
    machine; the code above only stops them costing science targets.
    """

    params = pathlib.Path(cave_explorer.__file__).parents[2] / 'config' / 'nav2_params.yaml'
    if not params.is_file():
        pytest.skip('nav2_params.yaml is not installed alongside the module')

    match = re.search(r'^\s*default_server_timeout:\s*(\d+)', params.read_text(encoding='utf-8'),
                      re.MULTILINE)
    assert match is not None, 'default_server_timeout is not set at all'
    assert int(match.group(1)) >= 100, \
        f'default_server_timeout is {match.group(1)}ms - too tight for this machine'
