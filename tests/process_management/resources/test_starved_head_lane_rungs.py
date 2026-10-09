"""The arbiter's service-lane rungs for a starved head whose deficit nothing cheaper closes.

A whole-card head on a card at its edge was refused by the measured identity for room the service lanes'
contexts held, and the ladder had no rung that named them: it emptied, the hold reclaimed nothing, and the
head waited on a fit nothing produced. The lanes now enter the ladder for a starved head, cheapest first and
one per evaluation, only when the permitted rungs would close the deficit, and never when policy withholds
lane reclaim. The first waits out the teardown grace. Each later rung of the same starvation episode is offered
once the previous one is graded on its lane's exit and a reading taken after it, or at the teardown
verification bound when the lane never answers.

A post-processing job's lane moves answer to the same closability test: offered only when no eviction or
demotion is on the ladder, the idle utilities lane before the safety cycle, each only when its return alone
covers the deficit.
"""

from __future__ import annotations

from dataclasses import replace

from horde_worker_regen.process_management.resources.admission_identity import TenantLane
from horde_worker_regen.process_management.resources.reclaim_ladder import teardown_verification_settle_seconds
from horde_worker_regen.process_management.resources.vram_arbiter import (
    _FIRST_PARTY_TEARDOWN_GRACE_SECONDS,
    ActuatorCommandKind,
    DeviceVramState,
    LaneRungGrade,
    MeasuredVramSnapshot,
    VramArbiter,
    VramDisposition,
    VramRequest,
    VramRequestKind,
)
from horde_worker_regen.process_management.scheduling.admission.clearance import StarvedLaneRungEpisode

_TOTAL_MB = 24074.0
_NOISE_MB = 1203.7
_CONTEXT_MB = 487.0
_CANDIDATE_MB = 19424.0


def _lane_state(
    *,
    post_process: bool = True,
    safety: bool = False,
    utilities: bool = False,
    utilities_reserved_mb: float = 554.0,
    device_free_mb: float = 20157.0,
) -> DeviceVramState:
    """The head's own context beside the service lanes named, with no idle sibling and no idle model."""
    lanes = {2: TenantLane.INFERENCE_IDLE}
    reserved = {2: 100.0}
    if post_process:
        lanes[1] = TenantLane.POST_PROCESS
        reserved[1] = 808.0
    if utilities:
        lanes[4] = TenantLane.UTILITIES
        reserved[4] = utilities_reserved_mb
    if safety:
        lanes[0] = TenantLane.SAFETY
    return DeviceVramState(
        total_vram_mb=_TOTAL_MB,
        baseline_mb=1000.0,
        committed_vram_mb=2000.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        device_free_mb=device_free_mb,
        noise_buffer_mb=_NOISE_MB,
        per_process_reserved_mb=reserved,
        lane_by_process_id=lanes,
        marginal_mb=_CONTEXT_MB,
        post_process_context_count=1 if post_process else 0,
        post_process_reclaim_allowed=post_process,
        safety_context_count=1 if safety else 0,
        safety_reclaim_allowed=safety,
        safety_footprint_mb=3044.0 if safety else 0.0,
        utilities_context_count=1 if utilities else 0,
        utilities_reclaim_allowed=utilities,
    )


def _head(*, starved_seconds: float = _FIRST_PARTY_TEARDOWN_GRACE_SECONDS + 1.0, **overrides: object) -> VramRequest:
    request = VramRequest(
        kind=VramRequestKind.MONOLITHIC_DISPATCH,
        job_label="Z-Image-Turbo",
        baseline="z_image_turbo",
        device_index=0,
        target_process_id=2,
        candidate_delta_mb=_CANDIDATE_MB,
        is_head_of_queue=True,
        head_job_id="head",
        starved_seconds=starved_seconds,
    )
    return replace(request, **overrides)  # type: ignore[arg-type]


def _verdict(state: DeviceVramState, request: VramRequest):  # noqa: ANN202
    arbiter = VramArbiter()
    arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: state}))
    return arbiter.evaluate(request)


def _kinds(verdict) -> list[ActuatorCommandKind]:  # noqa: ANN001
    return [command.kind for command in verdict.required_actuations]


def test_a_starved_head_with_an_empty_ladder_is_offered_the_post_processing_lane() -> None:
    """Past the grace, with nothing cheaper left and the lane closing the deficit, the ladder names the lane."""
    verdict = _verdict(_lane_state(), _head())

    assert verdict.disposition is VramDisposition.DEFER
    assert _kinds(verdict) == [ActuatorCommandKind.PAUSE_POST_PROCESS_LANE]


def test_inside_the_grace_no_lane_is_offered() -> None:
    """The lane rungs wait out the same grace the idle-context teardown does."""
    verdict = _verdict(_lane_state(), _head(starved_seconds=1.0))

    assert verdict.disposition is VramDisposition.DEFER
    assert _kinds(verdict) == []


def test_the_rungs_are_cheapest_first_and_one_per_evaluation() -> None:
    """Post-processing before safety before utilities; each evaluation names only the next one."""
    all_lanes = _lane_state(post_process=True, safety=True, utilities=True)
    assert _kinds(_verdict(all_lanes, _head())) == [ActuatorCommandKind.PAUSE_POST_PROCESS_LANE]

    no_pp = _lane_state(post_process=False, safety=True, utilities=True)
    assert _kinds(_verdict(no_pp, _head())) == [ActuatorCommandKind.CYCLE_SAFETY_OFF_GPU]

    only_utilities = _lane_state(post_process=False, safety=False, utilities=True)
    assert _kinds(_verdict(only_utilities, _head())) == [ActuatorCommandKind.PAUSE_UTILITIES_LANE]


def test_a_graded_rung_offers_the_next_lane_without_a_second_grace() -> None:
    """Once the post-processing lane's rung is graded, safety is offered on the next evaluation.

    The waiter's clock restarted when the first rung was applied, so it reads well inside the grace. The grade
    stands for the rest of the episode, which already waited the grace once.
    """
    pp_gone = _lane_state(post_process=False, safety=True, utilities=True)
    request = _head(starved_seconds=0.5, lane_rung_grade=LaneRungGrade.GRADED)

    assert _kinds(_verdict(pp_gone, request)) == [ActuatorCommandKind.CYCLE_SAFETY_OFF_GPU]


def test_a_rung_awaiting_its_grade_offers_no_further_lane() -> None:
    """The next lane waits for the previous one's grade, however long the head has starved."""
    pp_paused = _lane_state(post_process=False, safety=True, utilities=True)
    request = _head(lane_rung_grade=LaneRungGrade.AWAITING_GRADE)

    assert _kinds(_verdict(pp_paused, request)) == []


def test_a_rung_is_graded_on_the_lanes_exit_and_a_later_reading() -> None:
    """The evaluation that sees the lane gone still prices a reading from before it, so the next one grades."""
    episode = StarvedLaneRungEpisode(applied_at=100.0, lane_launches=frozenset({(200, 0)}))

    assert episode.grade(now=100.5, lane_exited=False) is LaneRungGrade.AWAITING_GRADE
    assert episode.grade(now=100.6, lane_exited=True) is LaneRungGrade.AWAITING_GRADE
    assert episode.grade(now=100.7, lane_exited=True) is LaneRungGrade.GRADED
    assert not episode.graded_at_bound


def test_a_rung_the_lane_never_answers_is_graded_at_the_bound() -> None:
    """A lane that never exits cannot hold the head: the rung is graded on whatever the card then reports."""
    bound = teardown_verification_settle_seconds()
    episode = StarvedLaneRungEpisode(applied_at=100.0, lane_launches=frozenset({(200, 0)}))

    assert episode.grade(now=100.0 + bound - 0.1, lane_exited=False) is LaneRungGrade.AWAITING_GRADE
    assert episode.grade(now=100.0 + bound, lane_exited=False) is LaneRungGrade.GRADED
    assert episode.graded_at_bound


def test_a_deficit_the_lanes_cannot_close_offers_nothing() -> None:
    """A lane paused for a head that still cannot fit is churn, so the ladder stays empty."""
    verdict = _verdict(_lane_state(device_free_mb=12000.0), _head())

    assert verdict.disposition is VramDisposition.DEFER
    assert _kinds(verdict) == []


def test_policy_withholds_lane_reclaim() -> None:
    """The operator's out: with lane reclaim withheld no lane rung is ever named."""
    verdict = _verdict(_lane_state(), _head(lane_reclaim_permitted=False))

    assert _kinds(verdict) == []


def test_a_non_head_and_a_post_processing_request_never_reach_the_lane_rungs() -> None:
    """The escalation is the head's alone; a post-processing job keeps its own borrow rules."""
    assert _kinds(_verdict(_lane_state(), _head(is_head_of_queue=False))) == []

    pp_job = _head(kind=VramRequestKind.PP_JOB)
    assert ActuatorCommandKind.PAUSE_POST_PROCESS_LANE not in _kinds(_verdict(_lane_state(), pp_job))


_PP_CANDIDATE_MB = 8000.0


def _pp_job_short_by(
    deficit_mb: float,
    *,
    utilities_reserved_mb: float | None = None,
    **overrides: object,
) -> tuple[DeviceVramState, VramRequest]:
    """A post-processing job on its own lane, short of room by ``deficit_mb``, beside an on-card safety.

    ``utilities_reserved_mb`` adds an idle image-utilities lane holding that reservation.
    """
    state = _lane_state(
        post_process=True,
        safety=True,
        utilities=utilities_reserved_mb is not None,
        utilities_reserved_mb=utilities_reserved_mb or 0.0,
        device_free_mb=_PP_CANDIDATE_MB + _NOISE_MB - deficit_mb,
    )
    request = _head(
        kind=VramRequestKind.PP_JOB,
        target_process_id=1,
        candidate_delta_mb=_PP_CANDIDATE_MB,
        is_head_of_queue=False,
        starved_seconds=0.0,
    )
    return state, replace(request, **overrides)  # type: ignore[arg-type]


def test_a_post_processing_deficit_demotion_covers_is_not_offered_the_safety_cycle() -> None:
    """A small deficit takes the in-place demotion; cycling the safety process waits for a later evaluation."""
    state, request = _pp_job_short_by(154.0)
    state = replace(state, safety_weights_demotable=True)

    assert _kinds(_verdict(state, request)) == [ActuatorCommandKind.DEMOTE_SAFETY_WEIGHTS]


def test_a_post_processing_deficit_beyond_the_safety_footprint_is_not_offered_the_safety_cycle() -> None:
    """Moving safety cannot close a deficit larger than what safety holds, so the move would be pure churn."""
    state, request = _pp_job_short_by(5077.0)

    assert ActuatorCommandKind.CYCLE_SAFETY_OFF_GPU not in _kinds(_verdict(state, request))


def test_a_post_processing_deficit_safety_alone_closes_is_offered_the_safety_cycle() -> None:
    """With no resident to evict and nothing to demote, a deficit within safety's footprint names the move."""
    state, request = _pp_job_short_by(2000.0)

    assert _kinds(_verdict(state, request)) == [ActuatorCommandKind.CYCLE_SAFETY_OFF_GPU]


def test_a_post_processing_job_with_an_evictable_resident_is_not_offered_the_safety_cycle() -> None:
    """The eviction is cheaper and is priced first; the next evaluation decides whether safety must move."""
    state, request = _pp_job_short_by(2000.0, has_reclaimable_idle_model=True)

    assert _kinds(_verdict(state, request)) == [ActuatorCommandKind.EVICT_COLDEST_IDLE_MODEL]


def test_a_post_processing_deficit_the_idle_utilities_lane_closes_is_offered_that_lane() -> None:
    """The utilities lane shares the service pool, so it is idle while post-processing runs and costs no rebuild.

    Pausing it keeps safety on the card, where a check takes a fraction of the time it takes on the CPU.
    """
    state, request = _pp_job_short_by(2000.0, utilities_reserved_mb=2700.0)

    assert _kinds(_verdict(state, request)) == [ActuatorCommandKind.PAUSE_UTILITIES_LANE]


def test_a_post_processing_deficit_the_utilities_lane_cannot_close_is_offered_the_safety_cycle() -> None:
    """A utilities lane holding less than the deficit is churn; safety, which covers it, is moved instead."""
    state, request = _pp_job_short_by(2000.0, utilities_reserved_mb=554.0)

    assert _kinds(_verdict(state, request)) == [ActuatorCommandKind.CYCLE_SAFETY_OFF_GPU]
