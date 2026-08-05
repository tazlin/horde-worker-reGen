"""Each registered gate, attacked on the three properties its declaration promises.

The gate registry says, per gate, what engages it, what releases it, and the bound or backstop that covers
it. A declaration nobody attacks is a comment. This module drives each gate from a state that genuinely
engages it and asserts the declaration holds:

* **engage-has-reachable-release** - drive the declared release condition through real state and assert the
  gate actually opens. The assertion is a positive outcome (the gate reports open, the work proceeds), never
  the absence of a log line, because a failure can produce silence just as easily as success can.
* **no self-inflicted permanent defer** - compose the engaging state purely from the request's own footprint
  or its own prior activity, and assert release or bounded escalation still happens. This is the shape the
  whole-card grace budget failed: reuse re-asks charged the budget that then denied the next ask, so a burst
  of jobs for one held model starved itself.
* **bounded resolution** - under an explicit clock, assert resolution or the named backstop engaging inside
  the registered bound.

Each class states which gate it covers and cites the registry entry, so a declaration cannot be edited into
something the tests no longer check. Gates whose seam does not exist without production refactoring are
named in ``TestGatesWithNoTestSeamYet`` rather than left silently uncovered.
"""

from __future__ import annotations

from unittest.mock import Mock, PropertyMock, patch

import pytest
from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE

from horde_worker_regen.process_management.config.worker_state import PopGate, PopPauseOwner, WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_popper import JobPopper
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.liveness.gate_registry import (
    GateKind,
    GateSurface,
    entry_for,
)
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.scheduling.dispatch_affinity import (
    _AFFINITY_MAX_SKIPS,
    AffinitySkipState,
    affinity_budget_seconds,
    affinity_skip_allowed,
    record_affinity_skip,
)
from horde_worker_regen.process_management.scheduling.governance.preload_admission import (
    preload_concurrency_blocked,
)
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    _ESTABLISH_WINDOW_LIMIT,
    _ESTABLISH_WINDOW_SECONDS,
    _GOVERNOR_DEFER_DWELL_SECONDS,
    _GRACE_BUDGET_SECONDS,
    _GRACE_BUDGET_WINDOW_SECONDS,
    _MIN_HOLD_SECONDS,
    WholeCardGrantKind,
    WholeCardResidencyLedger,
)
from tests.process_management.conftest import (
    make_mock_job,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_testable_process_manager,
)
from tests.process_management.jobs.test_job_popping import _FakeBacklogTracker, _make_popper

_CARD = None
"""The single-GPU / worker-wide residency key, which is a valid device index in its own right."""

_EPOCH = 10_000.0
"""An arbitrary clock origin. Every ledger query takes ``now`` explicitly, so the tests never touch the wall
clock and a bound is asserted by advancing this figure rather than by sleeping."""

"""A step past a boundary the production code compares strictly, so a bound is asserted as elapsed rather
than as exactly reached. Rolling windows drop a charge once its grant time is *before* the cutoff, so the
declared bound is the last instant a hold may stand, not the first instant it must be gone."""


def _assert_registered_hold(surface: GateSurface, key: str) -> None:
    """Assert the gate under test is registered as a hold with a declared release path.

    Binds each hostile class to its declaration, so deleting an entry (or downgrading a hold to an outcome
    to dodge the guardrail) fails the tests that prove its release path instead of quietly removing them.
    """
    entry = entry_for(surface, key)
    assert entry is not None, f"{surface.value}.{key} is attacked here but not registered"
    assert entry.kind is GateKind.HOLD, f"{surface.value}.{key} is attacked as a hold but registered as {entry.kind}"
    assert entry.released_by, f"{surface.value}.{key} declares no release condition"


class TestWholeCardGraceBudget:
    """``whole_card_governor.grace_budget``: the rolling allowance on opening fresh grace windows.

    The budget exists because back-to-back establish/restore churn re-arms the recovery supervisor's grace
    faster than it expires, and a worker whose supervisor is permanently disarmed is exactly where a real
    wedge goes unnoticed. It must brake that churn without ever becoming the reason a servable head is not
    served.
    """

    def test_the_declaration_exists(self) -> None:
        """The gate under attack is the one the registry describes."""
        _assert_registered_hold(GateSurface.WHOLE_CARD_GOVERNOR, "grace_budget")

    def test_a_spent_allowance_replenishes_as_its_charges_age_out(self) -> None:
        """Engage-has-reachable-release: charges leaving the rolling window reopen the card to establishing."""
        ledger = WholeCardResidencyLedger()
        now = _EPOCH
        # Spend past the allowance with physical establishments, each charging its own grace window.
        charge = _GRACE_BUDGET_SECONDS / 2.0
        for index in range(3):
            ledger.record_grant(
                _CARD,
                model=f"model-{index}",
                forecast=None,
                cooldown_until=0.0,
                now=now + index,
                establish_grace_seconds=charge,
            )
        assert ledger.grace_budget_exhausted(_CARD, now=now + 3) is True, "precondition: the allowance is spent"

        status = ledger.grace_budget_status(_CARD, now=now + 3)
        assert status.replenish_in_seconds > 0.0, "a spent allowance must say when it returns"

        # A charge exactly one window old is already out, so the quoted wait is exact: waiting precisely
        # that long reopens the card.
        released_at = now + 3 + status.replenish_in_seconds
        assert ledger.grace_budget_exhausted(_CARD, now=released_at) is False, (
            "the wait the budget itself quotes must be the wait that actually reopens the card"
        )

    def test_a_burst_reusing_one_residency_never_spends_the_allowance(self) -> None:
        """No self-inflicted permanent defer: a job's own prior asks must not fund the refusal of the next.

        The failure this encodes: every job for the model a card already holds re-asked for the residency,
        each ask charged a fresh grace window, and the allowance sized for physical teardown churn was spent
        by jobs that caused none. The queue then stood still behind a brake its own traffic had armed.
        """
        ledger = WholeCardResidencyLedger()
        first = ledger.record_grant(
            _CARD,
            model="heavy",
            forecast=None,
            cooldown_until=_EPOCH + 600.0,
            now=_EPOCH,
            establish_grace_seconds=120.0,
        )
        assert first is WholeCardGrantKind.ESTABLISH

        for ask in range(1, 40):
            kind = ledger.record_grant(
                _CARD,
                model="heavy",
                forecast=None,
                cooldown_until=_EPOCH + 600.0,
                now=_EPOCH + ask,
                establish_grace_seconds=120.0,
            )
            assert kind is WholeCardGrantKind.REUSE, "riding a held residency costs the card no teardown"

        state = ledger.state_for(_CARD)
        assert len(state.grace_charges) == 1, (
            "only the physical establishment may be charged; a burst that tore nothing down must not spend "
            "the allowance that would then deny it the card"
        )
        assert ledger.grace_budget_exhausted(_CARD, now=_EPOCH + 40) is False

    def test_the_refusal_resolves_inside_its_declared_window(self) -> None:
        """Bounded resolution: a spent allowance cannot outlast the rolling window it is measured over."""
        ledger = WholeCardResidencyLedger()
        ledger.record_grant(
            _CARD,
            model="heavy",
            forecast=None,
            cooldown_until=0.0,
            now=_EPOCH,
            establish_grace_seconds=_GRACE_BUDGET_SECONDS * 2,
        )
        assert ledger.grace_budget_exhausted(_CARD, now=_EPOCH) is True

        entry = entry_for(GateSurface.WHOLE_CARD_GOVERNOR, "grace_budget")
        assert entry is not None and entry.bound_seconds is not None
        assert ledger.grace_budget_exhausted(_CARD, now=_EPOCH + entry.bound_seconds) is False, (
            "every charge ages out of the rolling window, so the refusal cannot outlive the declared bound"
        )
        assert entry.bound_seconds == _GRACE_BUDGET_WINDOW_SECONDS

    def test_a_governed_head_stops_asking_for_the_card_after_the_bounded_dwell(self) -> None:
        """Bounded escalation: the brake hands the head to ordinary admission rather than parking it.

        The dwell is the declared backstop. Without it the budget is an absolute park: a card measurably able
        to hold the model sits idle while the queue stands still, which costs far more than the rotation
        churn the brake was protecting against.
        """
        ledger = WholeCardResidencyLedger()
        elapsed, exhausted = ledger.note_governor_defer(_CARD, model="heavy", now=_EPOCH)
        assert (elapsed, exhausted) == (0.0, False), "the wait is anchored at the first deferral of this head"

        _elapsed, still_inside = ledger.note_governor_defer(
            _CARD,
            model="heavy",
            now=_EPOCH + _GOVERNOR_DEFER_DWELL_SECONDS - 1.0,
        )
        assert still_inside is False

        _elapsed, spent = ledger.note_governor_defer(
            _CARD,
            model="heavy",
            now=_EPOCH + _GOVERNOR_DEFER_DWELL_SECONDS,
        )
        assert spent is True, "past the dwell the head stops preferring the card and measured admission decides"

    def test_one_heads_spent_dwell_is_never_inherited_by_the_next(self) -> None:
        """No self-inflicted permanent defer: a fresh head starts its own clock, not the previous head's."""
        ledger = WholeCardResidencyLedger()
        ledger.note_governor_defer(_CARD, model="first", now=_EPOCH)
        elapsed, exhausted = ledger.note_governor_defer(
            _CARD,
            model="second",
            now=_EPOCH + _GOVERNOR_DEFER_DWELL_SECONDS * 3,
        )
        assert (elapsed, exhausted) == (0.0, False), (
            "a head arriving behind a long-deferred one must get its own full dwell, or the queue's own "
            "history decides against it before it has asked once"
        )


class TestWholeCardEstablishRate:
    """``whole_card_governor.establish_rate``: the rolling per-card cap on physical establishments."""

    def test_the_declaration_exists(self) -> None:
        """The gate under attack is the one the registry describes."""
        _assert_registered_hold(GateSurface.WHOLE_CARD_GOVERNOR, "establish_rate")

    def test_the_limit_releases_when_the_oldest_establishment_ages_out(self) -> None:
        """Engage-has-reachable-release: the window is rolling, so waiting it out is the release."""
        ledger = WholeCardResidencyLedger()
        for index in range(_ESTABLISH_WINDOW_LIMIT):
            ledger.record_grant(
                _CARD,
                model=f"model-{index}",
                forecast=None,
                cooldown_until=0.0,
                now=_EPOCH + index,
            )
        assert ledger.establish_rate_exceeded(_CARD, now=_EPOCH + _ESTABLISH_WINDOW_LIMIT) is True

        released_at = _EPOCH + _ESTABLISH_WINDOW_SECONDS + 1.0
        assert ledger.establish_rate_exceeded(_CARD, now=released_at) is False, (
            "the oldest establishment leaving the window must admit the next ask"
        )

    def test_reusing_a_held_residency_is_not_counted_as_an_establishment(self) -> None:
        """No self-inflicted permanent defer: repeated asks for the held model cannot spend the rate."""
        ledger = WholeCardResidencyLedger()
        ledger.record_grant(_CARD, model="heavy", forecast=None, cooldown_until=_EPOCH + 600.0, now=_EPOCH)
        for ask in range(1, 50):
            ledger.record_grant(
                _CARD,
                model="heavy",
                forecast=None,
                cooldown_until=_EPOCH + 600.0,
                now=_EPOCH + ask,
            )
        assert len(ledger.state_for(_CARD).establishments) == 1
        assert ledger.establish_rate_exceeded(_CARD, now=_EPOCH + 50) is False

    def test_the_deferral_cannot_outlast_the_declared_window(self) -> None:
        """Bounded resolution: one window is the longest a rate deferral can last, by construction."""
        ledger = WholeCardResidencyLedger()
        for index in range(_ESTABLISH_WINDOW_LIMIT * 3):
            ledger.record_grant(
                _CARD,
                model=f"model-{index}",
                forecast=None,
                cooldown_until=0.0,
                now=_EPOCH + index,
            )
        entry = entry_for(GateSurface.WHOLE_CARD_GOVERNOR, "establish_rate")
        assert entry is not None and entry.bound_seconds is not None
        latest_establishment = _EPOCH + (_ESTABLISH_WINDOW_LIMIT * 3) - 1
        assert ledger.establish_rate_exceeded(_CARD, now=latest_establishment + entry.bound_seconds) is False


class TestWholeCardMinHold:
    """``scheduler_budget.whole_card_min_hold``: the non-preemptable floor under a fresh residency."""

    def test_the_declaration_exists(self) -> None:
        """The gate under attack is the one the registry describes."""
        _assert_registered_hold(GateSurface.SCHEDULER_BUDGET, "whole_card_min_hold")

    def test_the_floor_lapses_on_its_own_clock(self) -> None:
        """Engage-has-reachable-release, and bounded: the floor is a timer with nothing that can extend it."""
        ledger = WholeCardResidencyLedger()
        ledger.record_grant(_CARD, model="heavy", forecast=None, cooldown_until=0.0, now=_EPOCH)
        assert ledger.min_hold_active(_CARD, now=_EPOCH) is True
        assert ledger.min_hold_active(_CARD, now=_EPOCH + _MIN_HOLD_SECONDS - 1.0) is True
        assert ledger.min_hold_active(_CARD, now=_EPOCH + _MIN_HOLD_SECONDS) is False

    def test_a_burst_for_the_held_model_cannot_keep_re_arming_the_floor(self) -> None:
        """No self-inflicted permanent defer: reuse must not push the floor out ahead of the clock.

        If every ask for the resident model re-armed the floor, a steady stream of that model's own work
        would hold a different-model head off the card indefinitely, and the head's wait would be caused
        entirely by traffic the card was already serving.
        """
        ledger = WholeCardResidencyLedger()
        ledger.record_grant(_CARD, model="heavy", forecast=None, cooldown_until=_EPOCH + 600.0, now=_EPOCH)
        for ask in range(1, int(_MIN_HOLD_SECONDS) + 30):
            ledger.record_grant(
                _CARD,
                model="heavy",
                forecast=None,
                cooldown_until=_EPOCH + 600.0,
                now=_EPOCH + ask,
            )
        assert ledger.min_hold_active(_CARD, now=_EPOCH + _MIN_HOLD_SECONDS) is False, (
            "the floor amortizes one paid teardown; asks that paid for none must not extend it"
        )


class TestAffinityLineSkip:
    """``scheduler_budget.affinity_line_skip``: the bounded bypass of a cold FIFO head."""

    def test_the_declaration_exists(self) -> None:
        """The gate under attack is the one the registry describes."""
        _assert_registered_hold(GateSurface.SCHEDULER_BUDGET, "affinity_line_skip")

    def test_the_head_reclaims_the_slot_once_the_count_ceiling_is_reached(self) -> None:
        """Engage-has-reachable-release: the head's queue position is returned by the count bound."""
        state = AffinitySkipState()
        budget = affinity_budget_seconds(None)
        for skip in range(_AFFINITY_MAX_SKIPS):
            assert affinity_skip_allowed(state, "head", _EPOCH, budget, _AFFINITY_MAX_SKIPS) is True, (
                f"skip {skip} is inside both bounds"
            )
            state = record_affinity_skip(state, "head", _EPOCH)
        assert affinity_skip_allowed(state, "head", _EPOCH, budget, _AFFINITY_MAX_SKIPS) is False, (
            "at the ceiling the bypass stops and dispatch falls back to making room for the head"
        )

    def test_a_steady_stream_of_resident_work_cannot_bypass_one_head_forever(self) -> None:
        """No self-inflicted permanent defer: the head's own displacement history is what closes the window.

        Every skip is charged to the head it passed, so the traffic doing the passing is the traffic that
        ends its own permission. Without that, a queue that happens to hold resident-model work would starve
        a cold head purely by being busy.
        """
        state = AffinitySkipState()
        budget = affinity_budget_seconds(None)
        skips_taken = 0
        clock = _EPOCH
        for _tick in range(500):
            if not affinity_skip_allowed(state, "head", clock, budget, _AFFINITY_MAX_SKIPS):
                break
            state = record_affinity_skip(state, "head", clock)
            skips_taken += 1
            clock += 0.01
        assert skips_taken == _AFFINITY_MAX_SKIPS, (
            "an unbounded supply of bypassing work must still be cut off at the head's declared ceiling"
        )

    def test_the_wall_clock_budget_closes_the_window_even_below_the_count(self) -> None:
        """Bounded resolution: elapsed time alone ends the bypass, so a slow trickle cannot evade the count."""
        budget = affinity_budget_seconds(None)
        state = record_affinity_skip(AffinitySkipState(), "head", _EPOCH)
        assert state.skip_count < _AFFINITY_MAX_SKIPS, "precondition: the count bound is not what closes this"
        assert affinity_skip_allowed(state, "head", _EPOCH + budget - 0.01, budget, _AFFINITY_MAX_SKIPS) is True
        assert affinity_skip_allowed(state, "head", _EPOCH + budget, budget, _AFFINITY_MAX_SKIPS) is False

        entry = entry_for(GateSurface.SCHEDULER_BUDGET, "affinity_line_skip")
        assert entry is not None and entry.bound_seconds is not None
        assert budget <= entry.bound_seconds, "the declared bound must cover the widest budget the ttl can produce"


class TestPreloadConcurrencyGate:
    """``preload_admission.defer_concurrency``: the per-device model-load serialization gate."""

    def test_the_declaration_exists(self) -> None:
        """The gate under attack is the one the registry describes."""
        _assert_registered_hold(GateSurface.PRELOAD_ADMISSION, "defer_concurrency")

    def test_the_gate_opens_the_moment_the_in_flight_load_concludes(self) -> None:
        """Engage-has-reachable-release: the gate is a function of the live count and latches nothing."""
        assert (
            preload_concurrency_blocked(
                num_preloading=1,
                max_concurrent_inference_processes=2,
                very_fast_disk_mode=False,
            )
            is True
        )
        assert (
            preload_concurrency_blocked(
                num_preloading=0,
                max_concurrent_inference_processes=2,
                very_fast_disk_mode=False,
            )
            is False
        ), "however the load ended, the gate opens on the count alone"

    def test_the_gate_is_never_closed_against_an_idle_device(self) -> None:
        """No self-inflicted permanent defer: a device loading nothing cannot decline the first load.

        The gate is derived per cycle from the loading count, so no request can be held by a charge of its
        own that the count no longer contains. Checked across the whole configuration surface, since a
        relaxation that inverted would close the gate on an empty device.
        """
        for ceiling in range(1, 9):
            for very_fast_disk in (False, True):
                assert (
                    preload_concurrency_blocked(
                        num_preloading=0,
                        max_concurrent_inference_processes=ceiling,
                        very_fast_disk_mode=very_fast_disk,
                    )
                    is False
                ), f"an idle device declined a load (ceiling={ceiling}, very_fast_disk={very_fast_disk})"


class TestSafetyBacklogHysteresisLatch:
    """``pop_gate.safety_backlog``: the hysteretic intake latch over the post-inference safety stage."""

    def test_the_declaration_exists(self) -> None:
        """The gate under attack is the one the registry describes."""
        _assert_registered_hold(GateSurface.POP_GATE, "safety_backlog")

    @staticmethod
    def _popper_with_fixed_cap(cap: int) -> tuple[JobPopper, _FakeBacklogTracker]:
        """Return a popper whose safety cap is pinned, with the fake tracker feeding its backlog."""
        popper = _make_popper()
        tracker = _FakeBacklogTracker()
        popper._job_tracker = tracker  # pyrefly: ignore - a stub stands in for the job tracker
        popper._max_safe_safety_backlog = lambda: cap  # pyrefly: ignore - pinning the cap isolates the latch
        return popper, tracker

    def test_the_latch_releases_once_the_backlog_drains_below_its_release_bound(self) -> None:
        """Engage-has-reachable-release: draining the stage is the release, and it is reachable from engaged."""
        cap = 8
        popper, tracker = self._popper_with_fixed_cap(cap)

        tracker.set_backlog(cap)
        assert popper._is_post_inference_backlogged() is True, "precondition: the latch engages at the cap"

        # Hysteresis: the latch deliberately holds while the backlog hovers just under the cap.
        tracker.set_backlog(cap - 1)
        assert popper._is_post_inference_backlogged() is True

        tracker.set_backlog(0)
        assert popper._is_post_inference_backlogged() is False, (
            "a drained safety stage must reopen intake; a latch that survives its own release condition is a wedge"
        )

    def test_the_release_bound_is_reachable_from_the_engaged_state(self) -> None:
        """No self-inflicted permanent defer: the release bound sits strictly below the engage threshold.

        A hysteretic latch whose release bound met or exceeded its engage threshold could never open from the
        state that closed it: the very depth that engaged it would also satisfy holding it.
        """
        cap = 10
        popper, tracker = self._popper_with_fixed_cap(cap)
        tracker.set_backlog(cap)
        assert popper._is_post_inference_backlogged() is True

        for backlog in range(cap, -1, -1):
            tracker.set_backlog(backlog)
            if not popper._is_post_inference_backlogged():
                assert backlog < cap, "the latch must release strictly below the depth that engaged it"
                return
        pytest.fail("the latch never released, even at an empty safety stage")

    def test_a_drained_stage_reopens_intake_within_one_cycle(self) -> None:
        """Bounded resolution: the verdict is deterministic in the current backlog, so release costs one read."""
        popper, tracker = self._popper_with_fixed_cap(4)
        tracker.set_backlog(4)
        assert popper._is_post_inference_backlogged() is True
        tracker.set_backlog(0)
        assert popper._is_post_inference_backlogged() is False
        assert popper._is_post_inference_backlogged() is False, "the verdict is stable, not alternating"


class TestQueueFullPopGate:
    """``pop_gate.queue_full``: the depth cap, and the escalation that separates full from frozen.

    A full queue holds pops back legitimately only while it moves. The two states are told apart by
    movement, not by the gate, so the hostile property here is that the escalation fires on the motionless
    queue and stays quiet on the moving one.
    """

    def test_the_declaration_exists(self) -> None:
        """The gate under attack is the one the registry describes."""
        _assert_registered_hold(GateSurface.POP_GATE, "queue_full")

    @staticmethod
    def _manager_holding_at_queue_full(held_since: float) -> HordeWorkerProcessManager:
        """Return a process manager stamped as holding at the queue-full gate since ``held_since``."""
        manager = make_testable_process_manager()
        manager._state.last_pop_gate = str(PopGate.QUEUE_FULL)
        manager._state.last_pop_gate_since = held_since
        return manager

    def test_a_moving_queue_is_never_escalated(self) -> None:
        """Engage-has-reachable-release: dispatch or completion re-anchors the span, so the gate stays benign."""
        manager = self._manager_holding_at_queue_full(_EPOCH)
        completions = 0

        def _completed() -> int:
            return completions

        with patch.object(
            type(manager._job_tracker),
            "total_num_completed_jobs",
            new_callable=PropertyMock,
            side_effect=_completed,
        ):
            manager._check_full_queue_liveness(_EPOCH)
            for tick in range(1, 40):
                completions += 1
                with patch.object(type(manager), "_full_queue_frozen_line") as line:
                    manager._check_full_queue_liveness(_EPOCH + tick * 60.0)
                assert line.called is False, (
                    "a queue that keeps completing work is at its configured depth doing its job, not frozen"
                )

    def test_a_motionless_full_queue_is_escalated_inside_the_declared_bound(self) -> None:
        """Bounded resolution: the backstop the entry names actually fires, on its own clock."""
        from horde_worker_regen.process_management.process_manager import POP_LIVENESS_FROZEN_QUEUE_SECONDS

        manager = self._manager_holding_at_queue_full(_EPOCH)
        manager._check_full_queue_liveness(_EPOCH)

        with patch.object(type(manager), "_full_queue_frozen_line", return_value="frozen") as line:
            manager._check_full_queue_liveness(_EPOCH + POP_LIVENESS_FROZEN_QUEUE_SECONDS - 1.0)
            assert line.called is False, "the span has not yet run"
            manager._check_full_queue_liveness(_EPOCH + POP_LIVENESS_FROZEN_QUEUE_SECONDS)
            assert line.called is True, (
                "a full queue with nothing dispatched and nothing completed for the whole span is the worker's "
                "most complete stall and must be escalated, not read as depth management"
            )

    def test_the_gate_clearing_disarms_the_escalation(self) -> None:
        """No self-inflicted permanent defer: the escalation is armed by the gate, never by its own history."""
        manager = self._manager_holding_at_queue_full(_EPOCH)
        manager._check_full_queue_liveness(_EPOCH)
        manager._state.last_pop_gate = None
        manager._check_full_queue_liveness(_EPOCH + 10.0)
        assert manager._pop_liveness_frozen_baseline is None, (
            "a released gate must reset the frozen-queue anchor, or the next hold inherits this one's span"
        )


class TestSelfThrottlePopPause:
    """``pop_pause.*``: the shared self-throttle pause every fault backstop arms.

    Its liveness contract is unusually strict: nothing but the deadline gates the lift, so a condition that
    keeps faulting cannot hold the worker paused.
    """

    @pytest.mark.parametrize("key", ["fault_throttle", "ram_pressure", "safety"])
    def test_the_declaration_exists(self, key: str) -> None:
        """Every pause owner is registered as a hold with a declared release."""
        _assert_registered_hold(GateSurface.POP_PAUSE, key)

    def test_the_pause_lapses_on_its_deadline(self) -> None:
        """Engage-has-reachable-release: the deadline elapsing lifts the pause on the next tick."""
        manager = make_testable_process_manager()
        manager._state.self_throttle_paused = True
        manager._state.self_throttle_paused_until = _EPOCH
        manager._state.self_throttle_pause_owner = PopPauseOwner.FAULT_THROTTLE
        manager._state.self_throttle_pause_reason = "a run of resource faults"

        with patch("horde_worker_regen.process_management.process_manager.time.time", return_value=_EPOCH + 1.0):
            manager._apply_self_maintenance_throttle()

        assert manager._state.self_throttle_paused is False
        assert manager._state.self_throttle_pause_owner is None
        assert manager._state.workload_intake_paused is False, "the lifted pause must reopen worker-wide intake"

    def test_an_unrelenting_fault_condition_cannot_extend_the_standing_pause(self) -> None:
        """No self-inflicted permanent defer: the worker's own fault history must not re-arm before the lift.

        A pause that re-armed from the same faults that armed it would hold intake for as long as the
        condition persisted, with no lift and no disclosure of one.
        """
        manager = make_testable_process_manager()
        manager._state.self_throttle_paused = True
        manager._state.self_throttle_paused_until = _EPOCH
        manager._state.self_throttle_pause_owner = PopPauseOwner.FAULT_THROTTLE

        armed: list[float] = []

        def _would_arm(now: float) -> bool:
            armed.append(now)
            return True

        with (
            patch.object(manager, "_arm_resource_fault_throttle", side_effect=_would_arm),
            patch("horde_worker_regen.process_management.process_manager.time.time", return_value=_EPOCH + 1.0),
        ):
            manager._apply_self_maintenance_throttle()

        assert manager._state.self_throttle_paused is False, "the deadline lifts the pause before anything re-arms"
        assert armed == [], "the lapse tick returns without consulting the arming backstops"

    def test_the_pause_is_still_held_before_its_deadline(self) -> None:
        """Bounded resolution: the bound is the deadline itself, and it is honoured in both directions."""
        manager = make_testable_process_manager()
        manager._state.self_throttle_paused = True
        manager._state.self_throttle_paused_until = _EPOCH + 100.0
        manager._state.self_throttle_pause_owner = PopPauseOwner.SAFETY

        with patch("horde_worker_regen.process_management.process_manager.time.time", return_value=_EPOCH + 99.0):
            manager._apply_self_maintenance_throttle()
        assert manager._state.self_throttle_paused is True

        with patch("horde_worker_regen.process_management.process_manager.time.time", return_value=_EPOCH + 100.0):
            manager._apply_self_maintenance_throttle()
        assert manager._state.self_throttle_paused is False


class TestIntakePausedLatch:
    """``pop_gate.intake_paused``: the worker-wide posture that forbids every workload popper from popping."""

    def test_the_declaration_exists(self) -> None:
        """The gate under attack is the one the registry describes."""
        _assert_registered_hold(GateSurface.POP_GATE, "intake_paused")

    @pytest.mark.parametrize(
        "flag",
        ["shutting_down", "supervisor_paused", "self_throttle_paused", "downloads_only_hold", "recovery_parked"],
    )
    def test_each_arming_flow_releases_its_own_hold(self, flag: str) -> None:
        """Engage-has-reachable-release: the posture is a disjunction of live flags and latches nothing extra.

        Clearing the flag that set it is sufficient, so no arming flow can leave a residue that keeps intake
        closed after its own condition has passed.
        """
        state = WorkerState()
        assert state.workload_intake_paused is False, "precondition: a fresh worker accepts work"

        setattr(state, flag, True)
        assert state.workload_intake_paused is True

        setattr(state, flag, False)
        assert state.workload_intake_paused is False, f"clearing {flag} must reopen intake by itself"

    def test_overlapping_holds_release_independently(self) -> None:
        """No self-inflicted permanent defer: one flow's hold cannot outlive that flow through another's."""
        state = WorkerState()
        state.supervisor_paused = True
        state.downloads_only_hold = True
        assert state.workload_intake_paused is True

        state.supervisor_paused = False
        assert state.workload_intake_paused is True, "the remaining hold still stands on its own"

        state.downloads_only_hold = False
        assert state.workload_intake_paused is False, "no residue survives both holds clearing"


class TestKeepSingleInference:
    """``dispatch_stall.keep_single_inference``: the worker-wide hold for a workflow that cannot share."""

    def test_the_declaration_exists(self) -> None:
        """The gate under attack is the one the registry describes."""
        _assert_registered_hold(GateSurface.DISPATCH_STALL, "keep_single_inference")

    @staticmethod
    def _controlnet_xl_slot() -> tuple[ProcessMap, dict[str, Mock], HordeProcessInfo]:
        """A single idle slot still associated with a resident ControlNet-XL job, with its reference."""
        model = "qr-controlnet-sdxl"
        process_info = make_mock_process_info(1, model_name=model, state=HordeProcessState.WAITING_FOR_JOB)
        process_info.last_job_referenced = make_mock_job(model=model, workflow="qr_code")
        reference = {
            model: make_mock_model_reference_record(
                model,
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            ),
        }
        return ProcessMap({1: process_info}), reference, process_info

    def test_the_hold_lifts_when_the_workflow_leaves_the_slot(self) -> None:
        """Engage-has-reachable-release: the hold is re-derived from live state every cycle and latches nothing."""
        process_map, reference, process_info = self._controlnet_xl_slot()
        keep, reason = process_map.keep_single_inference(stable_diffusion_model_reference=reference)
        assert (keep, reason) == (True, "ControlNet XL"), "precondition: the guard is engaged"

        process_info.last_job_referenced = None
        keep_after, _reason_after = process_map.keep_single_inference(stable_diffusion_model_reference=reference)
        assert keep_after is False, "the model leaving the slot must lift the worker-wide hold"

    def test_the_hold_is_derived_only_from_the_live_slot_association(self) -> None:
        """No self-inflicted permanent defer: nothing about past engagements feeds the next verdict.

        Re-asking many times must not accumulate anything, or a workflow that once ran would keep the worker
        single-lane for the rest of the session.
        """
        process_map, reference, process_info = self._controlnet_xl_slot()
        for _ask in range(50):
            keep, _reason = process_map.keep_single_inference(stable_diffusion_model_reference=reference)
            assert keep is True

        process_info.last_job_referenced = make_mock_job(model="stable_diffusion")
        keep_after, _reason_after = process_map.keep_single_inference(stable_diffusion_model_reference=reference)
        assert keep_after is False, "fifty engaged reads must leave no residue behind the fifty-first"


class TestGatesWithNoTestSeamYet:
    """Registered gates whose hostile properties are not yet driven, and what each would need.

    Naming them here keeps the coverage claim honest: the registry is complete, the hostile suite is not,
    and the gap is a list rather than an impression. Each entry states the seam that would close it.
    """

    _UNCOVERED: dict[tuple[GateSurface, str], str] = {
        (GateSurface.DISPATCH_STALL, "overlap_headway"): (
            "the headway fractions are computed inside InferenceScheduler._concurrent_overlap_allowed from "
            "live per-process sampling progress and an arbiter verdict, with no seam that sets progress "
            "directly; driving it needs either a progress-injection seam on the process map or a full "
            "sampling-progress fixture"
        ),
        (GateSurface.POP_GATE, "ram_pressure"): (
            "InferenceScheduler.set_available_ram_mb_provider now pins the reading the hold is derived from, "
            "so the seam exists; what remains is driving a scripted sequence of readings across governance "
            "ticks and observing governance_healthy_but_held arm and disarm around the hold"
        ),
        (GateSurface.PRELOAD_ADMISSION, "defer_budget"): (
            "covered as a property at the arbiter surface in "
            "tests/process_management/resources/test_admission_liveness_matrix.py; a per-gate drive here "
            "would duplicate it rather than add coverage"
        ),
        (GateSurface.PRELOAD_ADMISSION, "defer_ram_pressure"): (
            "the reclamation ladder's terminal rung is its declared backstop and is exercised through the "
            "ram-governor suites; a per-gate drive needs the ReclamationExecutor protocol wired to a host-RAM "
            "sequence fed through InferenceScheduler.set_available_ram_mb_provider"
        ),
    }

    def test_every_uncovered_gate_is_still_registered(self) -> None:
        """An uncovered gate must at least have declared its release path."""
        for (surface, key), seam in self._UNCOVERED.items():
            entry = entry_for(surface, key)
            assert entry is not None, f"{surface.value}.{key} is listed as uncovered but is not registered"
            assert entry.kind is GateKind.HOLD
            assert seam, f"{surface.value}.{key} does not say what seam would cover it"
