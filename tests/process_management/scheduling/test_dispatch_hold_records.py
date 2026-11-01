"""A dispatch hold is recorded with the measured room it was judged against, and its release names the cause.

The hold gate already counted holds and releases; a reader of the stats export could see that a head was
held but not who held the card, what could have returned it, or whether the probe was still ahead. These
tests pin the two records a hold now produces (standing, released) and the room fields on the deferral.
"""

from __future__ import annotations

from unittest.mock import Mock

from loguru import logger

from horde_worker_regen.analysis.log_signatures import SIGNATURES
from horde_worker_regen.process_management.resources.run_metrics import DecisionKind, ResourceStateKind
from horde_worker_regen.process_management.resources.vram_arbiter import DeviceVramState, VramArbiter
from horde_worker_regen.process_management.scheduling.ledgers.dispatch_holds import (
    DISPATCH_HOLD_LIVENESS_SECONDS,
    HoldRemedy,
)
from tests.process_management.scheduling.test_dispatch_residency_reconciliation import (
    _card_below_any_candidate,
    _fitting_state,
    _install_cycle,
    _over_committed_state,
    _scheduler_with_idle_sibling,
    _scheduler_without_sibling,
)


class _Recorder:
    """Captures decision and resource-state records the scheduler emits."""

    def __init__(self) -> None:
        self.decisions: list[dict[str, object]] = []
        self.states: list[dict[str, object]] = []

    def decision(self, **kwargs: object) -> None:
        self.decisions.append(kwargs)

    def state(self, **kwargs: object) -> None:
        self.states.append(kwargs)


async def test_a_hold_records_standing_with_the_room_and_release_with_its_cause() -> None:
    """The first hold pass emits a standing record carrying the room; the admit emits a release record."""
    scheduler, job, target = await _scheduler_without_sibling()
    recorder = _Recorder()
    scheduler._decision_sink = recorder.decision
    scheduler._resource_state_sink = recorder.state

    _install_cycle(scheduler, _over_committed_state())
    assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True
    assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True

    standing = [record for record in recorder.states if record["state"] == "standing"]
    assert len(standing) == 1, "one standing record per hold, not one per pass"
    assert standing[0]["state_kind"] is ResourceStateKind.DISPATCH_HOLD
    inputs = standing[0]["inputs"]
    assert isinstance(inputs, dict)
    assert inputs["model"] == "model_a"
    assert inputs["room_deficit_mb"] > 0
    assert inputs["probe_state"].startswith("waiting_")
    assert "hold_seconds" in inputs and "starved_seconds" in inputs

    deferral = [record for record in recorder.decisions if record["decision_kind"] is DecisionKind.INFERENCE_DISPATCH]
    assert deferral, "the deferral decision is still recorded"
    deferral_inputs = deferral[0]["inputs"]
    assert isinstance(deferral_inputs, dict)
    assert deferral_inputs["room_deficit_mb"] == inputs["room_deficit_mb"]
    assert "room_closable" in deferral_inputs

    _install_cycle(scheduler, _fitting_state())
    assert scheduler._dispatch_residency_reconciliation_holds(job, target) is False

    released = [record for record in recorder.states if record["state"] == "released"]
    assert len(released) == 1
    assert released[0]["reason"] == "natural_free"
    tracked = scheduler._job_tracker.get_tracked_job(job.id_)
    assert tracked is not None and tracked.dispatch_hold_seconds is not None, "the job record carries the hold"
    released_inputs = released[0]["inputs"]
    assert isinstance(released_inputs, dict)
    assert released_inputs["room_deficit_mb"] == inputs["room_deficit_mb"], "the release names what the hold saw"
    assert released_inputs["hold_seconds"] >= 0.0


async def test_no_sink_means_no_record_and_no_failure() -> None:
    """A scheduler without the sink wired holds and releases exactly as before."""
    scheduler, job, target = await _scheduler_without_sibling()
    scheduler._resource_state_sink = None

    _install_cycle(scheduler, _over_committed_state())
    assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True
    _install_cycle(scheduler, _fitting_state())
    assert scheduler._dispatch_residency_reconciliation_holds(job, target) is False
    assert scheduler.dispatch_holds.holds == 1


async def test_the_hold_liveness_predicate_is_bounded_by_the_hold_age() -> None:
    """A standing hold on the head reads alive inside its bound, lapses past it, and clears on release."""
    scheduler, job, target = await _scheduler_without_sibling()
    now = 1000.0
    scheduler._clock = lambda: now

    assert scheduler.dispatch_hold_liveness_active() is False

    _install_cycle(scheduler, _over_committed_state())
    assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True
    assert scheduler.dispatch_hold_liveness_active() is True
    ages = scheduler.dispatch_hold_liveness_seconds()
    assert ages is not None and ages.held_seconds == 0.0 and ages.bound_seconds == DISPATCH_HOLD_LIVENESS_SECONDS
    assert ages.remedy is HoldRemedy.PROBE, "a fresh hold's probe is still ahead"

    now = 1000.0 + DISPATCH_HOLD_LIVENESS_SECONDS - 1.0
    assert scheduler.dispatch_hold_liveness_active() is True
    now = 1000.0 + DISPATCH_HOLD_LIVENESS_SECONDS + 1.0
    assert scheduler.dispatch_hold_liveness_active() is False, "past the bound the hold is a wedge like any other"

    _install_cycle(scheduler, _fitting_state())
    assert scheduler._dispatch_residency_reconciliation_holds(job, target) is False
    assert scheduler.dispatch_hold_liveness_active() is False
    assert scheduler.dispatch_hold_liveness_seconds() is None


async def test_a_ceiling_deny_on_the_only_card_is_not_live_on_its_first_pass() -> None:
    """A hold on a terminal DENY with no other card has no remedy, so recovery does not wait out the bound."""
    scheduler, job, target = await _scheduler_without_sibling()
    scheduler._clock = lambda: 1000.0
    _install_cycle(scheduler, _card_below_any_candidate())

    assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True

    liveness = scheduler.dispatch_hold_liveness_seconds()
    assert liveness is not None and liveness.held_seconds == 0.0
    assert liveness.remedy is HoldRemedy.NONE
    assert scheduler.dispatch_hold_liveness_active() is False


async def test_a_deny_hold_line_and_its_stall_summary_agree_that_nothing_is_reclaimed() -> None:
    """The hold line and the stall summary read one ledger, so neither describes an eviction on a terminal DENY."""
    scheduler, job, target = await _scheduler_without_sibling()
    _install_cycle(scheduler, _card_below_any_candidate())
    captured: list[str] = []
    handler_id = logger.add(lambda message: captured.append(message.record["message"]), level="WARNING")
    try:
        assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True
    finally:
        logger.remove(handler_id)

    hold_lines = [match for line in captured if (match := SIGNATURES["dispatch_residency_hold"].pattern.search(line))]
    assert len(hold_lines) == 1, captured
    assert hold_lines[0].group("model") == "model_a"
    assert hold_lines[0].group("reclaim") == (
        "Nothing is being reclaimed for this hold and no remedy is left on this card"
    )
    assert "next verdicts a fit" not in hold_lines[0].string, "a hold with no remedy does not release on a verdict"

    _bucket, reason = scheduler._classify_dispatch_stall(job, {})
    assert SIGNATURES["dispatch_stall_reconcile_no_remedy"].pattern.search(reason) is not None, reason
    assert "evict" not in reason


async def test_a_defer_with_reclaim_accepted_is_live_to_the_bound() -> None:
    """A hold whose eviction an actuator accepted keeps recovery waiting until the bound, and no longer."""
    scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
    now = 1000.0
    scheduler._clock = lambda: now
    scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]
    _install_cycle(scheduler, _over_committed_state())

    assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True

    liveness = scheduler.dispatch_hold_liveness_seconds()
    assert liveness is not None and liveness.remedy is HoldRemedy.RECLAIM
    now = 1000.0 + DISPATCH_HOLD_LIVENESS_SECONDS - 1.0
    assert scheduler.dispatch_hold_liveness_active() is True
    now = 1000.0 + DISPATCH_HOLD_LIVENESS_SECONDS + 1.0
    assert scheduler.dispatch_hold_liveness_active() is False, "a live remedy does not outlast the bound"


def _over_ceiling_by(
    candidate_mb: float,
    overshoot_mb: float,
    *,
    noise_buffer_mb: float,
    total_mb: float = 16000.0,
) -> DeviceVramState:
    """A card whose foreign floor puts its achievable ceiling ``overshoot_mb`` below ``candidate_mb``.

    Idle residents hold the rest of the card, so it is not converged-empty.
    """
    return DeviceVramState(
        total_vram_mb=total_mb,
        baseline_mb=0.0,
        committed_vram_mb=total_mb,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        device_free_mb=200.0,
        foreign_floor_mb=total_mb - noise_buffer_mb - (candidate_mb - overshoot_mb),
    )


async def test_a_ceiling_deny_pending_convergence_is_live() -> None:
    """An over-ceiling head inside the attempt allowance on a card not yet converged keeps the hold live.

    The arbiter holds the DENY back while the card still has reclaimable state, so the head can reach the
    converged card its one real load is judged on. The probe has not been spent, so the hold has a remedy even
    when no eviction was accepted on this pass.
    """
    scheduler, job, target, _sibling = await _scheduler_with_idle_sibling()
    scheduler.unload_models_from_vram = Mock(return_value=False)  # type: ignore[method-assign]
    decision_sink = Mock(return_value=None)
    scheduler._decision_sink = decision_sink
    _install_cycle(scheduler, _over_committed_state())
    assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True
    candidate_mb = decision_sink.call_args.kwargs["inputs"]["candidate_delta_mb"]
    assert isinstance(candidate_mb, float)

    noise_buffer_mb = _over_committed_state().noise_buffer_mb
    allowance_mb = VramArbiter._ceiling_attempt_allowance_mb(candidate_mb - noise_buffer_mb)
    state = _over_ceiling_by(candidate_mb, allowance_mb / 2.0, noise_buffer_mb=noise_buffer_mb)
    ceiling_mb = state.achievable_ceiling_mb()
    assert ceiling_mb is not None and 0.0 < candidate_mb - ceiling_mb <= VramArbiter._ceiling_attempt_allowance_mb(
        ceiling_mb
    ), "precondition: the candidate sits above the ceiling, inside the attempt allowance"
    _install_cycle(scheduler, state)

    assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True

    reason = decision_sink.call_args.kwargs["reason"]
    assert "achievable ceiling" not in reason, f"precondition: the arbiter issued the terminal DENY ({reason})"
    liveness = scheduler.dispatch_hold_liveness_seconds()
    assert liveness is not None and liveness.remedy is HoldRemedy.PROBE
    assert scheduler.dispatch_hold_liveness_active() is True

    _install_cycle(scheduler, _over_ceiling_by(candidate_mb, 4096.0, noise_buffer_mb=noise_buffer_mb))
    assert scheduler._dispatch_residency_reconciliation_holds(job, target) is True
    assert "achievable ceiling" in decision_sink.call_args.kwargs["reason"], (
        "precondition: beyond the allowance the arbiter issues the terminal DENY"
    )
    assert scheduler.dispatch_hold_liveness_active() is False, "a terminal DENY on the only card has no remedy"
