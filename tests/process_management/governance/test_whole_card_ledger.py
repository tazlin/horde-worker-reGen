"""Unit tests for the whole-card residency ledger and the co-resident sizing rule."""

from __future__ import annotations

from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from horde_worker_regen.process_management.scheduling.governance import (
    WholeCardPhase,
    WholeCardResidencyLedger,
    WholeCardResidencyMachine,
    max_coresident_for_peak,
)
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    _ESTABLISH_WINDOW_LIMIT,
    _ESTABLISH_WINDOW_SECONDS,
    _GOVERNOR_DEFER_DWELL_SECONDS,
    _GRACE_BUDGET_SECONDS,
    _GRACE_BUDGET_WINDOW_SECONDS,
    _MIN_HOLD_SECONDS,
    WholeCardGrantKind,
)

_NOW = 1_000_000.0
_ESTABLISH_GRACE = 90.0
_RESTORE_GRACE = 30.0


def _granted_ledger(
    device_index: int | None = None,
    *,
    model: str = "heavy-model",
    establish_grace_seconds: float = 0.0,
) -> WholeCardResidencyLedger:
    """Build a ledger with one residency granted at ``_NOW``."""
    ledger = WholeCardResidencyLedger()
    ledger.record_grant(
        device_index,
        model=model,
        forecast=None,
        cooldown_until=_NOW + 300.0,
        now=_NOW,
        establish_grace_seconds=establish_grace_seconds,
    )
    return ledger


class TestLedgerQueries:
    """The ledger answers residency questions without touching live process state."""

    def test_empty_ledger_holds_nothing(self) -> None:
        """A fresh ledger reports no residencies anywhere."""
        ledger = WholeCardResidencyLedger()
        assert ledger.held() == []
        assert ledger.any_held() is False
        assert ledger.holder_for_model("any") == (False, None)
        assert ledger.get(0) is None

    def test_state_for_creates_lazily_and_is_stable(self) -> None:
        """The accessor creates one state per card and returns the same object thereafter."""
        ledger = WholeCardResidencyLedger()
        state = ledger.state_for(None)
        assert ledger.state_for(None) is state

    def test_grant_is_held_and_found_by_model(self) -> None:
        """A granted residency is held and locatable by its model, including on the None key."""
        ledger = _granted_ledger(None)
        assert ledger.any_held() is True
        assert ledger.holder_for_model("heavy-model") == (True, None)
        held = ledger.held()
        assert len(held) == 1
        assert held[0][0] is None

    def test_a_reuse_re_ask_keeps_the_establishment_stamp_and_refreshes_the_cooldown(self) -> None:
        """Re-asking for the model the card already holds keeps the grace anchor and extends the hold.

        The stamp anchors the recovery supervisor's grace window over one physical teardown. A later job for
        the same held model rides that residency without touching the card, so re-opening the window would
        hand the supervisor a fresh excuse for a queue nothing is currently tearing down.
        """
        ledger = _granted_ledger(0)
        kind = ledger.record_grant(
            0,
            model="heavy-model",
            forecast=None,
            cooldown_until=_NOW + 600.0,
            now=_NOW + 50.0,
        )
        state = ledger.state_for(0)
        assert kind is WholeCardGrantKind.REUSE
        assert state.established_at == _NOW
        assert state.cooldown_until == _NOW + 600.0


class TestPhases:
    """The phase query splits a held residency into establishing and holding."""

    def test_no_residency_reads_none(self) -> None:
        """A card without a residency has no model and phase NONE."""
        ledger = WholeCardResidencyLedger()
        assert ledger.phase(0, now=_NOW, establish_grace_seconds=_ESTABLISH_GRACE) == (None, WholeCardPhase.NONE)

    def test_fresh_grant_is_establishing(self) -> None:
        """Inside the establish grace the residency reads as establishing."""
        ledger = _granted_ledger(0)
        model, phase = ledger.phase(0, now=_NOW + 10.0, establish_grace_seconds=_ESTABLISH_GRACE)
        assert model == "heavy-model"
        assert phase is WholeCardPhase.ESTABLISHING

    def test_past_grace_is_holding(self) -> None:
        """Past the establish grace the residency reads as holding."""
        ledger = _granted_ledger(0)
        _model, phase = ledger.phase(
            0,
            now=_NOW + _ESTABLISH_GRACE + 1.0,
            establish_grace_seconds=_ESTABLISH_GRACE,
        )
        assert phase is WholeCardPhase.HOLDING


class TestGraceWindows:
    """Grace windows mark a held queue as intentional, bounded so a stuck setup still trips recovery."""

    def test_establishing_residency_is_in_grace(self) -> None:
        """A residency inside its establish window reports grace."""
        ledger = _granted_ledger(0)
        assert ledger.grace_active(
            now=_NOW + 10.0,
            establish_grace_seconds=_ESTABLISH_GRACE,
            restore_grace_seconds=_RESTORE_GRACE,
        )

    def test_grace_expires(self) -> None:
        """Past both windows the grace no longer applies."""
        ledger = _granted_ledger(0)
        assert not ledger.grace_active(
            now=_NOW + _ESTABLISH_GRACE + 1.0,
            establish_grace_seconds=_ESTABLISH_GRACE,
            restore_grace_seconds=_RESTORE_GRACE,
        )

    def test_restore_window_counts_even_after_model_clears(self) -> None:
        """The restore churn is covered by grace even though the model is already cleared."""
        ledger = _granted_ledger(0)
        state = ledger.state_for(0)
        state.model = None
        state.restore_at = _NOW + 100.0
        assert ledger.grace_active(
            now=_NOW + 110.0,
            establish_grace_seconds=_ESTABLISH_GRACE,
            restore_grace_seconds=_RESTORE_GRACE,
        )

    def test_drain_backstop_elapses_from_structural_completion(self) -> None:
        """The bounded drain backstop runs from the structural-completion latch, not the establishment."""
        ledger = _granted_ledger(0)
        # Established but not yet structurally complete: the backstop is not running at all.
        assert not ledger.drain_backstop_elapsed(0, now=_NOW + 500.0, settle_seconds=20.0)
        ledger.state_for(0).structural_complete_at = _NOW + 500.0
        assert not ledger.drain_backstop_elapsed(0, now=_NOW + 505.0, settle_seconds=20.0)
        assert ledger.drain_backstop_elapsed(0, now=_NOW + 525.0, settle_seconds=20.0)
        assert not ledger.drain_backstop_elapsed(1, now=_NOW + 525.0, settle_seconds=20.0)


class TestMinHold:
    """A fresh grant is immune to early release for the time it takes to amortize the actuation paid."""

    def test_min_hold_denies_then_allows(self) -> None:
        """The floor is active from the grant and lapses once it elapses."""
        ledger = _granted_ledger(0)
        assert ledger.min_hold_active(0, now=_NOW + 1.0) is True
        assert ledger.min_hold_active(0, now=_NOW + _MIN_HOLD_SECONDS - 0.1) is True
        assert ledger.min_hold_active(0, now=_NOW + _MIN_HOLD_SECONDS) is False

    def test_min_hold_is_inert_without_a_held_residency(self) -> None:
        """A card with no residency (never granted, or already restored) holds nothing."""
        ledger = WholeCardResidencyLedger()
        assert ledger.min_hold_active(0, now=_NOW) is False
        granted = _granted_ledger(0)
        granted.record_restore(0, now=_NOW + 1.0)
        assert granted.min_hold_active(0, now=_NOW + 2.0) is False

    def test_only_a_physical_establishment_opens_a_new_floor(self) -> None:
        """Re-establishing after a restore restarts the floor; a reuse re-ask leaves it where it was.

        The floor exists to amortize actuation the card has actually paid for. A second job riding the held
        residency pays none, so extending the floor on its behalf would keep a different-model head locked
        out for as long as same-model work keeps arriving.
        """
        ledger = _granted_ledger(0)
        ledger.record_grant(
            0,
            model="heavy-model",
            forecast=None,
            cooldown_until=_NOW + 300.0,
            now=_NOW + 10.0,
        )
        assert ledger.state_for(0).min_hold_until == _NOW + _MIN_HOLD_SECONDS
        ledger.record_restore(0, now=_NOW + 15.0)
        kind = ledger.record_grant(
            0,
            model="heavy-model",
            forecast=None,
            cooldown_until=_NOW + 300.0,
            now=_NOW + 20.0,
        )
        assert kind is WholeCardGrantKind.ESTABLISH
        assert ledger.state_for(0).min_hold_until == _NOW + 20.0 + _MIN_HOLD_SECONDS


class TestEstablishRateLimit:
    """Establishments per card are counted over a rolling window so churn cannot run unbounded."""

    def _establish(self, ledger: WholeCardResidencyLedger, *, at: float) -> None:
        """Record one physical establish/restore cycle on card 0 at ``at``."""
        ledger.record_grant(
            0,
            model="heavy-model",
            forecast=None,
            cooldown_until=at + 300.0,
            now=at,
        )
        ledger.record_restore(0, now=at)

    def test_unknown_card_is_never_rate_limited(self) -> None:
        """A card that has never established has no history to exceed."""
        ledger = WholeCardResidencyLedger()
        assert ledger.establish_rate_exceeded(0, now=_NOW) is False

    def test_limit_reached_within_the_window_then_replenished(self) -> None:
        """The allowance is spent inside the window and returns as the oldest establishment ages out."""
        ledger = WholeCardResidencyLedger()
        for index in range(_ESTABLISH_WINDOW_LIMIT):
            self._establish(ledger, at=_NOW + index)
            if index < _ESTABLISH_WINDOW_LIMIT - 1:
                assert ledger.establish_rate_exceeded(0, now=_NOW + index) is False
        assert ledger.establish_rate_exceeded(0, now=_NOW + _ESTABLISH_WINDOW_LIMIT) is True
        # The deferral a caller pays can never outlast the window itself.
        assert ledger.establish_rate_exceeded(0, now=_NOW + _ESTABLISH_WINDOW_SECONDS + 1.0) is False

    def test_the_limit_is_per_card(self) -> None:
        """One card exhausting its allowance leaves another card's untouched."""
        ledger = WholeCardResidencyLedger()
        for index in range(_ESTABLISH_WINDOW_LIMIT):
            self._establish(ledger, at=_NOW + index)
        assert ledger.establish_rate_exceeded(0, now=_NOW) is True
        assert ledger.establish_rate_exceeded(1, now=_NOW) is False


class TestGraceBudget:
    """Grace opened per card is capped over a rolling window, gating new establishments at admission."""

    def _cycle(self, ledger: WholeCardResidencyLedger, *, at: float) -> None:
        """Run one establish/restore cycle on card 0, charging both nominal grace windows."""
        ledger.record_grant(
            0,
            model="heavy-model",
            forecast=None,
            cooldown_until=at + 300.0,
            now=at,
            establish_grace_seconds=_ESTABLISH_GRACE,
        )
        ledger.record_restore(0, now=at + 1.0, restore_grace_seconds=_RESTORE_GRACE)

    def test_budget_accrues_gates_and_replenishes(self) -> None:
        """Repeated cycles spend the allowance, gate further establishments, then replenish as they age out."""
        ledger = WholeCardResidencyLedger()
        cycles = int(_GRACE_BUDGET_SECONDS // (_ESTABLISH_GRACE + _RESTORE_GRACE)) + 1
        for index in range(cycles):
            self._cycle(ledger, at=_NOW + index * 10.0)
            assert ledger.grace_budget_exhausted(0, now=_NOW + index * 10.0 + 2.0) is (
                (index + 1) * (_ESTABLISH_GRACE + _RESTORE_GRACE) > _GRACE_BUDGET_SECONDS
            )
        assert ledger.grace_budget_exhausted(0, now=_NOW + _GRACE_BUDGET_WINDOW_SECONDS + 100.0) is False

    def test_an_exhausted_budget_leaves_a_granted_window_intact(self) -> None:
        """A window already granted excuses the held queue for its full duration whatever the spend says.

        The establishment the window covers is a teardown the scheduler commanded. Withdrawing its excuse
        part-way would have the recovery supervisor read that deliberate action as a structural wedge, so the
        budget is answered at admission instead and never revokes an in-flight window.
        """
        ledger = WholeCardResidencyLedger()
        cycles = int(_GRACE_BUDGET_SECONDS // (_ESTABLISH_GRACE + _RESTORE_GRACE)) + 1
        for index in range(cycles):
            self._cycle(ledger, at=_NOW + index * 10.0)
        # Re-establish so a residency is held and establishing at the moment of the read.
        last = _NOW + cycles * 10.0
        ledger.record_grant(
            0,
            model="heavy-model",
            forecast=None,
            cooldown_until=last + 300.0,
            now=last,
            establish_grace_seconds=_ESTABLISH_GRACE,
        )
        assert ledger.grace_budget_exhausted(0, now=last + 1.0) is True, (
            "precondition: the card's rolling grace budget is spent"
        )
        for elapsed in (1.0, _ESTABLISH_GRACE / 2.0, _ESTABLISH_GRACE - 0.1):
            assert ledger.grace_active(
                now=last + elapsed,
                establish_grace_seconds=_ESTABLISH_GRACE,
                restore_grace_seconds=_RESTORE_GRACE,
            ), "the granted establish window holds for its whole duration despite the spent budget"
        assert not ledger.grace_active(
            now=last + _ESTABLISH_GRACE + 0.1,
            establish_grace_seconds=_ESTABLISH_GRACE,
            restore_grace_seconds=_RESTORE_GRACE,
        ), "the window's own duration is the liveness bound on a residency that never completes"

    def test_a_restore_window_is_granted_regardless_of_the_budget(self) -> None:
        """The restore churn is always excused: the budget gates establishments, not teardown reversal."""
        ledger = WholeCardResidencyLedger()
        cycles = int(_GRACE_BUDGET_SECONDS // (_ESTABLISH_GRACE + _RESTORE_GRACE)) + 1
        for index in range(cycles):
            self._cycle(ledger, at=_NOW + index * 10.0)
        last = _NOW + cycles * 10.0
        ledger.record_restore(0, now=last, restore_grace_seconds=_RESTORE_GRACE)
        assert ledger.grace_budget_exhausted(0, now=last) is True, (
            "precondition: the card's rolling grace budget is spent"
        )
        assert ledger.grace_active(
            now=last + _RESTORE_GRACE - 0.1,
            establish_grace_seconds=_ESTABLISH_GRACE,
            restore_grace_seconds=_RESTORE_GRACE,
        )
        assert not ledger.grace_active(
            now=last + _RESTORE_GRACE + 0.1,
            establish_grace_seconds=_ESTABLISH_GRACE,
            restore_grace_seconds=_RESTORE_GRACE,
        )

    def test_a_single_cycle_is_well_inside_the_budget(self) -> None:
        """The ordinary one-establish-per-rotation shape is never refused."""
        ledger = WholeCardResidencyLedger()
        self._cycle(ledger, at=_NOW)
        assert ledger.grace_budget_exhausted(0, now=_NOW + 5.0) is False


class TestGraceIsChargedPerPhysicalEpisode:
    """Grace and the establishment count are spent by teardown churn, never by reuse of a held residency.

    The cooldown exists so a burst of heavy jobs rides one residency instead of each churning a teardown and
    restore. That burst costs the card exactly one grace window, so charging it per job spends an allowance
    sized for physical churn on jobs that caused none, and the card then refuses the next genuine
    establishment while nothing has actually been torn down.
    """

    @staticmethod
    def _ask(ledger: WholeCardResidencyLedger, *, at: float, model: str = "heavy-model") -> WholeCardGrantKind:
        """Record one head's whole-card ask, exactly as the scheduler records it every cycle."""
        return ledger.record_grant(
            0,
            model=model,
            forecast=None,
            cooldown_until=at + 300.0,
            now=at,
            establish_grace_seconds=_ESTABLISH_GRACE,
        )

    def test_a_burst_reusing_one_residency_costs_one_window(self) -> None:
        """A fresh grant plus repeated same-model asks charge once and record one establishment."""
        ledger = WholeCardResidencyLedger()
        assert self._ask(ledger, at=_NOW) is WholeCardGrantKind.ESTABLISH
        for index in range(1, 12):
            assert self._ask(ledger, at=_NOW + index * 20.0) is WholeCardGrantKind.REUSE

        state = ledger.state_for(0)
        assert sum(seconds for _granted_at, seconds in state.grace_charges) == _ESTABLISH_GRACE
        assert len(state.establishments) == 1
        assert ledger.grace_budget_exhausted(0, now=_NOW + 240.0) is False
        assert ledger.establish_rate_exceeded(0, now=_NOW + 240.0) is False

    def test_a_restore_and_re_establish_costs_a_second_window(self) -> None:
        """Physically giving the card back and taking it again is a new episode, so it charges again."""
        ledger = WholeCardResidencyLedger()
        self._ask(ledger, at=_NOW)
        self._ask(ledger, at=_NOW + 10.0)
        ledger.record_restore(0, now=_NOW + 20.0, restore_grace_seconds=_RESTORE_GRACE)
        assert self._ask(ledger, at=_NOW + 30.0) is WholeCardGrantKind.ESTABLISH

        state = ledger.state_for(0)
        expected = _ESTABLISH_GRACE * 2 + _RESTORE_GRACE
        assert sum(seconds for _granted_at, seconds in state.grace_charges) == expected
        assert len(state.establishments) == 2

    def test_sustained_teardown_churn_still_exhausts_both_governors(self) -> None:
        """Repeated establish/restore cycles remain governed: the brake is on churn, not on demand."""
        ledger = WholeCardResidencyLedger()
        cycle_cost = _ESTABLISH_GRACE + _RESTORE_GRACE
        cycles = int(_GRACE_BUDGET_SECONDS // cycle_cost) + 1
        for index in range(cycles):
            at = _NOW + index * 5.0
            self._ask(ledger, at=at)
            ledger.record_restore(0, now=at + 1.0, restore_grace_seconds=_RESTORE_GRACE)

        now = _NOW + cycles * 5.0
        assert ledger.grace_budget_exhausted(0, now=now) is True
        assert ledger.establish_rate_exceeded(0, now=now) is True

    def test_a_different_model_taking_the_card_is_a_new_episode(self) -> None:
        """Swapping the resident model is a physical rotation, so it charges and counts."""
        ledger = WholeCardResidencyLedger()
        self._ask(ledger, at=_NOW)
        assert self._ask(ledger, at=_NOW + 30.0, model="other-heavy-model") is WholeCardGrantKind.ESTABLISH
        assert len(ledger.state_for(0).establishments) == 2


class TestGraceBudgetStatus:
    """The budget can say what it has spent and when the next allowance returns."""

    def test_status_reports_spend_remaining_and_the_next_replenishment(self) -> None:
        """The status quotes the rolling spend, what is left of the allowance, and the wait for more."""
        ledger = WholeCardResidencyLedger()
        state = ledger.state_for(0)
        state.grace_charges.append((_NOW, _GRACE_BUDGET_SECONDS))
        state.grace_charges.append((_NOW + 100.0, 60.0))

        status = ledger.grace_budget_status(0, now=_NOW + 200.0)
        assert status.allowance_seconds == _GRACE_BUDGET_SECONDS
        assert status.spent_seconds == _GRACE_BUDGET_SECONDS + 60.0
        assert status.remaining_seconds == 0.0
        # Only the first charge need age out for the spend to fall back inside the allowance.
        assert status.replenish_in_seconds == (_NOW + _GRACE_BUDGET_WINDOW_SECONDS) - (_NOW + 200.0)

    def test_an_unspent_card_needs_no_replenishment(self) -> None:
        """With nothing charged the whole allowance is available and no wait is quoted."""
        status = WholeCardResidencyLedger().grace_budget_status(0, now=_NOW)
        assert status.spent_seconds == 0.0
        assert status.remaining_seconds == _GRACE_BUDGET_SECONDS
        assert status.replenish_in_seconds == 0.0


class TestGovernorDeferDwell:
    """A head held off the card by a churn governor waits a bounded dwell, then stops preferring the card."""

    def test_the_dwell_runs_from_the_first_deferral_and_then_reports_exhausted(self) -> None:
        """Consecutive deferrals of one head accumulate until the dwell is spent."""
        ledger = WholeCardResidencyLedger()
        elapsed, exhausted = ledger.note_governor_defer(0, model="heavy-model", now=_NOW)
        assert (elapsed, exhausted) == (0.0, False)
        elapsed, exhausted = ledger.note_governor_defer(0, model="heavy-model", now=_NOW + 10.0)
        assert (elapsed, exhausted) == (10.0, False)
        _elapsed, exhausted = ledger.note_governor_defer(
            0,
            model="heavy-model",
            now=_NOW + _GOVERNOR_DEFER_DWELL_SECONDS,
        )
        assert exhausted is True

    def test_a_new_head_does_not_inherit_the_previous_dwell(self) -> None:
        """The dwell is one head's wait, so a different model starts its own."""
        ledger = WholeCardResidencyLedger()
        ledger.note_governor_defer(0, model="heavy-model", now=_NOW)
        elapsed, exhausted = ledger.note_governor_defer(
            0,
            model="other-heavy-model",
            now=_NOW + _GOVERNOR_DEFER_DWELL_SECONDS + 1.0,
        )
        assert (elapsed, exhausted) == (0.0, False)

    def test_clearing_the_deferral_restarts_the_next_dwell(self) -> None:
        """Once the governors let a head through, the next deferral is timed from scratch."""
        ledger = WholeCardResidencyLedger()
        ledger.note_governor_defer(0, model="heavy-model", now=_NOW)
        ledger.clear_governor_defer(0)
        elapsed, exhausted = ledger.note_governor_defer(
            0,
            model="heavy-model",
            now=_NOW + _GOVERNOR_DEFER_DWELL_SECONDS + 1.0,
        )
        assert (elapsed, exhausted) == (0.0, False)

    def test_the_dwell_is_per_card(self) -> None:
        """One card's held head says nothing about another card's."""
        ledger = WholeCardResidencyLedger()
        ledger.note_governor_defer(0, model="heavy-model", now=_NOW)
        elapsed, _exhausted = ledger.note_governor_defer(
            1,
            model="heavy-model",
            now=_NOW + _GOVERNOR_DEFER_DWELL_SECONDS,
        )
        assert elapsed == 0.0


class TestMaxCoresidentForPeak:
    """The sizing rule for how many live contexts a rejected peak can co-reside with."""

    def test_unsizable_without_total_vram(self) -> None:
        """No reported total VRAM means the depth cannot be sized."""
        assert (
            max_coresident_for_peak(
                total_vram_mb=None,
                per_process_overhead_mb=1200.0,
                marginal_overhead_mb=500.0,
                peak_mb=8000.0,
                reserve_mb=1000.0,
            )
            is None
        )

    def test_tight_budget_floors_at_one_context(self) -> None:
        """A peak that leaves less than one full context still allows the job's own context."""
        assert (
            max_coresident_for_peak(
                total_vram_mb=16000.0,
                per_process_overhead_mb=1200.0,
                marginal_overhead_mb=500.0,
                peak_mb=15000.0,
                reserve_mb=500.0,
            )
            == 1
        )

    def test_marginal_prices_additional_contexts(self) -> None:
        """Beyond the first full-cost context, each extra context costs only the marginal."""
        # Budget = 16000 - 8000 - 1000 = 7000; first context 1200, then (7000-1200)//500 = 11 more.
        assert (
            max_coresident_for_peak(
                total_vram_mb=16000.0,
                per_process_overhead_mb=1200.0,
                marginal_overhead_mb=500.0,
                peak_mb=8000.0,
                reserve_mb=1000.0,
            )
            == 12
        )

    def test_unmeasured_marginal_falls_back_to_full_cost(self) -> None:
        """An unmeasured marginal prices every context at the full first-context cost."""
        # Budget = 7000; first context 1200, then (7000-1200)//1200 = 4 more.
        assert (
            max_coresident_for_peak(
                total_vram_mb=16000.0,
                per_process_overhead_mb=1200.0,
                marginal_overhead_mb=None,
                peak_mb=8000.0,
                reserve_mb=1000.0,
            )
            == 5
        )


class TestWholeCardResidencyMachine:
    """The machine owns pure whole-card transition decisions."""

    def test_residency_demand_requires_enabled_head_and_teardown_need(self) -> None:
        """Only a head with a teardown forecast enters the residency pipeline."""
        machine = WholeCardResidencyMachine()
        forecast = StreamForecast(
            weights_mb=12_000.0,
            reserve_mb=2_000.0,
            free_now_mb=1_000.0,
            free_if_alone_mb=14_000.0,
            free_after_model_evict_mb=10_000.0,
            total_vram_mb=16_000.0,
            per_process_overhead_mb=1_000.0,
        )
        # Four live contexts against a three-context budget, so the reduction the branch proposes has
        # something to reduce and the demand is not suppressed for want of a remedy.
        assert machine.residency_demanded(forecast, enabled=True, is_head_blocker=True, live_inference_process_count=4)
        assert not machine.residency_demanded(
            forecast,
            enabled=False,
            is_head_blocker=True,
            live_inference_process_count=4,
        )
        assert not machine.residency_demanded(
            forecast,
            enabled=True,
            is_head_blocker=False,
            live_inference_process_count=4,
        )

    def test_a_reduction_with_nothing_to_reduce_is_not_a_residency_demand(self) -> None:
        """A process-count-reduction claim is refused when the budget already holds every live context.

        The branch's whole premise is that the live inference contexts over-commit the card and stopping some
        of them cures it. When the budget says more contexts fit than are running, the shortfall is made of
        charges the teardown cannot reclaim, and the residency machinery (pop monopoly, retention hold,
        sibling eviction) would only reach a topology the card is already in.
        """
        machine = WholeCardResidencyMachine()
        # Moderate weights on a 16GB card whose siblings-present floor is squeezed by charges no inference
        # teardown gives back, so the reduction branch reads True while thirteen contexts would still fit.
        forecast = StreamForecast(
            weights_mb=8_832.0,
            reserve_mb=1_519.0,
            base_reserve_mb=1_519.0,
            free_now_mb=8_000.0,
            free_if_alone_mb=15_030.0,
            free_after_model_evict_mb=8_718.0,
            total_vram_mb=16_384.0,
            per_process_overhead_mb=1_354.0,
            marginal_process_overhead_mb=384.0,
            # Safety's resident weights, the post-processing lane's context, and the image lane's decode
            # spike: the whole of the shortfall, and none of it an inference context.
            unreclaimable_charge_mb=5_928.0,
        )
        assert forecast.needs_process_count_reduction is True
        assert forecast.needs_exclusive_residency is False
        target = forecast.max_resident_processes()
        assert target is not None and target >= 2

        assert not machine.residency_demanded(
            forecast,
            enabled=True,
            is_head_blocker=True,
            live_inference_process_count=2,
        )
        # Run the pool past the budget and the same forecast demands the card again: there is now a context
        # for the reduction to take.
        assert machine.residency_demanded(
            forecast,
            enabled=True,
            is_head_blocker=True,
            live_inference_process_count=target + 1,
        )

    def test_a_deficit_larger_than_the_unreclaimable_charges_still_demands_the_card(self) -> None:
        """A card carrying no unreclaimable charges keeps its reduction demand, however roomy the budget reads.

        The suppression rests on the deficit being attributable to charges a teardown cannot reclaim. Where
        the forecast reports none, the live contexts are the only thing the shortfall can be made of and the
        reduction is exactly the right remedy.
        """
        machine = WholeCardResidencyMachine()
        forecast = StreamForecast(
            weights_mb=8_832.0,
            reserve_mb=1_519.0,
            base_reserve_mb=1_519.0,
            free_now_mb=8_000.0,
            free_if_alone_mb=15_030.0,
            free_after_model_evict_mb=8_718.0,
            total_vram_mb=16_384.0,
            per_process_overhead_mb=1_354.0,
            marginal_process_overhead_mb=384.0,
        )
        assert forecast.needs_process_count_reduction is True
        assert machine.residency_demanded(
            forecast,
            enabled=True,
            is_head_blocker=True,
            live_inference_process_count=2,
        )

    def test_the_suppression_never_reaches_a_weight_dominant_claim(self) -> None:
        """A model that needs sole residency keeps the card however many contexts the budget says fit.

        Sole residency is a statement about the weights, not about context arithmetic, so it is decided before
        the reduction branch and its suppression.
        """
        machine = WholeCardResidencyMachine()
        forecast = StreamForecast(
            weights_mb=11_500.0,
            reserve_mb=2_500.0,
            base_reserve_mb=1_519.0,
            free_now_mb=3_000.0,
            free_if_alone_mb=15_030.0,
            free_after_model_evict_mb=14_646.0,
            total_vram_mb=16_384.0,
            per_process_overhead_mb=1_354.0,
            marginal_process_overhead_mb=384.0,
            wants_whole_card=True,
        )
        assert forecast.needs_exclusive_residency is True
        assert machine.residency_demanded(
            forecast,
            enabled=True,
            is_head_blocker=True,
            live_inference_process_count=1,
        )

    def test_an_unsized_forecast_names_no_process_target(self) -> None:
        """A forecast that cannot size the card reports None, which is not the same statement as one.

        Sole residency is a measured verdict. Where the card was never sized (no reported total VRAM, or no
        forecast at all), reporting 1 would hand the caller a teardown depth nobody measured.
        """
        machine = WholeCardResidencyMachine()
        unsized = StreamForecast(
            weights_mb=12_000.0,
            reserve_mb=2_000.0,
            free_now_mb=1_000.0,
            free_if_alone_mb=14_000.0,
            free_after_model_evict_mb=10_000.0,
            total_vram_mb=None,
            per_process_overhead_mb=1_000.0,
        )
        assert machine.target_process_count(unsized) is None
        assert machine.target_process_count(None) is None

    def test_an_unsized_target_leaves_the_process_count_leg_satisfied(self) -> None:
        """With no depth to converge on, readiness turns on the other legs rather than parking forever."""
        machine = WholeCardResidencyMachine()
        unsized = StreamForecast(
            weights_mb=12_000.0,
            reserve_mb=2_000.0,
            free_now_mb=1_000.0,
            free_if_alone_mb=14_000.0,
            free_after_model_evict_mb=10_000.0,
            total_vram_mb=None,
            per_process_overhead_mb=1_000.0,
        )
        # Four live processes and no target: the count cannot gate, but the live weight fit still can.
        assert machine.teardown_complete(
            unsized,
            loaded_process_count=4,
            safety_clear_of_card=True,
            weights_fit_live=True,
            drain_backstop_elapsed=False,
        )
        # The other legs keep gating: safety has not left the card yet.
        assert not machine.teardown_complete(
            unsized,
            loaded_process_count=4,
            safety_clear_of_card=False,
            weights_fit_live=True,
            drain_backstop_elapsed=False,
        )

    def test_teardown_complete_requires_target_and_safety_then_live_fit_or_backstop(self) -> None:
        """The readiness query mirrors the scheduler's structural teardown gate."""
        machine = WholeCardResidencyMachine()
        # A measured marginal is pinned so the teardown target is sole residency (1): with an unmeasured
        # marginal the forecast now seeds a small per-context constant, under which a second 13GB-model
        # context would nominally fit and the target would read 2. The gate mechanics under test are the
        # sole-residency ones, so the marginal is fixed to a value the card-filling weights do not admit twice.
        forecast = StreamForecast(
            weights_mb=13_000.0,
            reserve_mb=1_500.0,
            free_now_mb=1_000.0,
            free_if_alone_mb=14_500.0,
            free_after_model_evict_mb=10_000.0,
            total_vram_mb=16_000.0,
            per_process_overhead_mb=1_000.0,
            marginal_process_overhead_mb=600.0,
        )
        assert not machine.teardown_complete(
            forecast,
            loaded_process_count=2,
            safety_clear_of_card=True,
            weights_fit_live=True,
            drain_backstop_elapsed=False,
        )
        assert not machine.teardown_complete(
            forecast,
            loaded_process_count=1,
            safety_clear_of_card=False,
            weights_fit_live=True,
            drain_backstop_elapsed=False,
        )
        assert machine.teardown_complete(
            forecast,
            loaded_process_count=1,
            safety_clear_of_card=True,
            weights_fit_live=False,
            drain_backstop_elapsed=True,
        )

    def test_teardown_complete_waits_for_post_process_lane_to_vacate_the_card(self) -> None:
        """A residency that must stop the post-processing lane holds the head until that lane is gone.

        The lane's CUDA context is only freed when its process exits, so admitting the head while it is still
        resident (even with the weights nominally fitting) is exactly what leaves too little room and streams
        the weights. The gate must therefore wait on the structural ``post_process_cleared`` signal.
        """
        machine = WholeCardResidencyMachine()
        forecast = StreamForecast(
            weights_mb=11_500.0,
            reserve_mb=1_500.0,
            free_now_mb=14_000.0,
            free_if_alone_mb=15_000.0,
            free_after_model_evict_mb=13_000.0,
            total_vram_mb=16_000.0,
            per_process_overhead_mb=1_000.0,
        )
        # Sole residency reached, safety off, weights read as fitting live: but the lane has not yet vacated.
        assert not machine.teardown_complete(
            forecast,
            loaded_process_count=1,
            safety_clear_of_card=True,
            post_process_pause_required=True,
            post_process_cleared=False,
            weights_fit_live=True,
            drain_backstop_elapsed=True,
        )
        # Once the lane is gone the head may load.
        assert machine.teardown_complete(
            forecast,
            loaded_process_count=1,
            safety_clear_of_card=True,
            post_process_pause_required=True,
            post_process_cleared=True,
            weights_fit_live=True,
            drain_backstop_elapsed=True,
        )
        # A residency on a card the lane does not occupy never waits on it (default: not required, cleared).
        assert machine.teardown_complete(
            forecast,
            loaded_process_count=1,
            safety_clear_of_card=True,
            weights_fit_live=True,
            drain_backstop_elapsed=False,
        )


class TestStructuralCompletionLatch:
    """The readiness query records when the teardown's structural legs first all passed."""

    @staticmethod
    def _machine() -> WholeCardResidencyMachine:
        """A machine holding one residency on card 0, granted at ``_NOW``."""
        machine = WholeCardResidencyMachine()
        machine.record_grant(
            0,
            model="heavy-model",
            forecast=None,
            cooldown_until=_NOW + 300.0,
            now=_NOW,
        )
        return machine

    @staticmethod
    def _forecast() -> StreamForecast:
        """A card-filling forecast whose teardown target is sole residency."""
        return StreamForecast(
            weights_mb=13_000.0,
            reserve_mb=1_500.0,
            free_now_mb=1_000.0,
            free_if_alone_mb=14_500.0,
            free_after_model_evict_mb=10_000.0,
            total_vram_mb=16_000.0,
            per_process_overhead_mb=1_000.0,
            marginal_process_overhead_mb=600.0,
        )

    def test_a_slow_teardown_does_not_start_the_backstop(self) -> None:
        """While siblings are still resident the latch stays unset, so the backstop is not running."""
        machine = self._machine()
        forecast = self._forecast()
        assert not machine.teardown_complete(
            forecast,
            loaded_process_count=3,
            safety_clear_of_card=True,
            weights_fit_live=False,
            drain_backstop_elapsed=False,
            device_index=0,
            now=_NOW + 200.0,
        )
        assert machine.state_for(0).structural_complete_at == 0.0
        assert not machine.drain_backstop_elapsed(0, now=_NOW + 200.0, settle_seconds=20.0)

    def test_the_latch_stamps_once_and_then_holds(self) -> None:
        """The first structurally-complete ask stamps the clock; later asks leave it alone."""
        machine = self._machine()
        forecast = self._forecast()
        machine.teardown_complete(
            forecast,
            loaded_process_count=1,
            safety_clear_of_card=True,
            weights_fit_live=False,
            drain_backstop_elapsed=False,
            device_index=0,
            now=_NOW + 200.0,
        )
        assert machine.state_for(0).structural_complete_at == _NOW + 200.0
        machine.teardown_complete(
            forecast,
            loaded_process_count=1,
            safety_clear_of_card=True,
            weights_fit_live=False,
            drain_backstop_elapsed=False,
            device_index=0,
            now=_NOW + 210.0,
        )
        assert machine.state_for(0).structural_complete_at == _NOW + 200.0
        assert not machine.drain_backstop_elapsed(0, now=_NOW + 215.0, settle_seconds=20.0)
        assert machine.drain_backstop_elapsed(0, now=_NOW + 220.0, settle_seconds=20.0)

    def test_a_re_established_grant_does_not_inherit_an_elapsed_backstop(self) -> None:
        """A retry tears the card down again, so its backstop starts from the new teardown."""
        machine = self._machine()
        forecast = self._forecast()
        machine.teardown_complete(
            forecast,
            loaded_process_count=1,
            safety_clear_of_card=True,
            weights_fit_live=False,
            drain_backstop_elapsed=False,
            device_index=0,
            now=_NOW,
        )
        assert machine.drain_backstop_elapsed(0, now=_NOW + 100.0, settle_seconds=20.0)
        machine.record_restore(0, now=_NOW + 90.0)
        machine.record_grant(
            0,
            model="heavy-model",
            forecast=None,
            cooldown_until=_NOW + 400.0,
            now=_NOW + 100.0,
        )
        assert not machine.drain_backstop_elapsed(0, now=_NOW + 100.0, settle_seconds=20.0)

    def test_a_reuse_re_ask_keeps_a_converged_residencys_backstop(self) -> None:
        """A later job for the model the card already holds does not restart the drain clock.

        The card is already converged and drained for that model; restarting the backstop on every re-ask
        would re-park a head behind a teardown that finished long ago.
        """
        machine = self._machine()
        forecast = self._forecast()
        machine.teardown_complete(
            forecast,
            loaded_process_count=1,
            safety_clear_of_card=True,
            weights_fit_live=False,
            drain_backstop_elapsed=False,
            device_index=0,
            now=_NOW,
        )
        machine.record_grant(
            0,
            model="heavy-model",
            forecast=None,
            cooldown_until=_NOW + 400.0,
            now=_NOW + 100.0,
        )
        assert machine.state_for(0).structural_complete_at == _NOW
        assert machine.drain_backstop_elapsed(0, now=_NOW + 100.0, settle_seconds=20.0)

    def test_omitting_the_clock_records_nothing(self) -> None:
        """A caller that only wants the answer (the ledger's pure query form) latches nothing."""
        machine = self._machine()
        assert machine.teardown_complete(
            self._forecast(),
            loaded_process_count=1,
            safety_clear_of_card=True,
            weights_fit_live=True,
            drain_backstop_elapsed=False,
            device_index=0,
        )
        assert machine.state_for(0).structural_complete_at == 0.0
