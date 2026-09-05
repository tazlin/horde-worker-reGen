"""A dispatch hold is recorded with the measured room it was judged against, and its release names the cause.

The hold gate already counted holds and releases; a reader of the stats export could see that a head was
held but not who held the card, what could have returned it, or whether the probe was still ahead. These
tests pin the two records a hold now produces (standing, released) and the room fields on the deferral.
"""

from __future__ import annotations

from horde_worker_regen.process_management.resources.run_metrics import DecisionKind, ResourceStateKind
from horde_worker_regen.process_management.scheduling.inference_scheduler import _DISPATCH_HOLD_LIVENESS_SECONDS
from tests.process_management.scheduling.test_dispatch_residency_reconciliation import (
    _fitting_state,
    _install_cycle,
    _over_committed_state,
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
    assert scheduler.latest_dispatch_reconciliation_holds() == 1


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
    assert ages is not None and ages[0] == 0.0 and ages[1] == _DISPATCH_HOLD_LIVENESS_SECONDS

    now = 1000.0 + _DISPATCH_HOLD_LIVENESS_SECONDS - 1.0
    assert scheduler.dispatch_hold_liveness_active() is True
    now = 1000.0 + _DISPATCH_HOLD_LIVENESS_SECONDS + 1.0
    assert scheduler.dispatch_hold_liveness_active() is False, "past the bound the hold is a wedge like any other"

    _install_cycle(scheduler, _fitting_state())
    assert scheduler._dispatch_residency_reconciliation_holds(job, target) is False
    assert scheduler.dispatch_hold_liveness_active() is False
    assert scheduler.dispatch_hold_liveness_seconds() is None
