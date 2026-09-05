"""The arbiter's service-lane rungs for a starved head whose deficit nothing cheaper closes.

A whole-card head on a card at its edge was refused by the measured identity for room the service lanes'
contexts held, and the ladder had no rung that named them: it emptied, the hold reclaimed nothing, and the
head waited on a fit nothing produced. The lanes now enter the ladder for a starved head, cheapest first and
one per evaluation, only past the teardown grace, only when the permitted rungs would close the deficit, and
never when policy withholds lane reclaim.
"""

from __future__ import annotations

from dataclasses import replace

from horde_worker_regen.process_management.resources.admission_identity import TenantLane
from horde_worker_regen.process_management.resources.vram_arbiter import (
    _FIRST_PARTY_TEARDOWN_GRACE_SECONDS,
    ActuatorCommandKind,
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
    VramDisposition,
    VramRequest,
    VramRequestKind,
)

_TOTAL_MB = 24074.0
_NOISE_MB = 1203.7
_CONTEXT_MB = 487.0
_CANDIDATE_MB = 19424.0


def _lane_state(
    *,
    post_process: bool = True,
    safety: bool = False,
    utilities: bool = False,
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
        reserved[4] = 554.0
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
