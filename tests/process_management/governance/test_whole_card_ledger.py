"""Unit tests for the whole-card residency ledger and the co-resident sizing rule."""

from __future__ import annotations

from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from horde_worker_regen.process_management.scheduling.governance import (
    WholeCardPhase,
    WholeCardResidencyLedger,
    WholeCardResidencyMachine,
    max_coresident_for_peak,
)

_NOW = 1_000_000.0
_ESTABLISH_GRACE = 90.0
_RESTORE_GRACE = 30.0


def _granted_ledger(device_index: int | None = None, *, model: str = "heavy-model") -> WholeCardResidencyLedger:
    """Build a ledger with one residency granted at ``_NOW``."""
    ledger = WholeCardResidencyLedger()
    ledger.record_grant(
        device_index,
        model=model,
        forecast=None,
        cooldown_until=_NOW + 300.0,
        now=_NOW,
        refresh_established=True,
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

    def test_grant_preserves_established_at_unless_refreshed(self) -> None:
        """A re-grant without refresh keeps the original establishment stamp (the grace anchor)."""
        ledger = _granted_ledger(0)
        ledger.record_grant(
            0,
            model="heavy-model",
            forecast=None,
            cooldown_until=_NOW + 600.0,
            now=_NOW + 50.0,
            refresh_established=False,
        )
        state = ledger.state_for(0)
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

    def test_drain_backstop_elapses_from_establishment(self) -> None:
        """The bounded drain backstop is measured from the establishment stamp."""
        ledger = _granted_ledger(0)
        assert not ledger.drain_backstop_elapsed(0, now=_NOW + 5.0, settle_seconds=20.0)
        assert ledger.drain_backstop_elapsed(0, now=_NOW + 25.0, settle_seconds=20.0)
        assert not ledger.drain_backstop_elapsed(1, now=_NOW + 25.0, settle_seconds=20.0)


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
        assert machine.residency_demanded(forecast, enabled=True, is_head_blocker=True)
        assert not machine.residency_demanded(forecast, enabled=False, is_head_blocker=True)
        assert not machine.residency_demanded(forecast, enabled=True, is_head_blocker=False)

    def test_teardown_complete_requires_target_and_safety_then_live_fit_or_backstop(self) -> None:
        """The readiness query mirrors the scheduler's structural teardown gate."""
        machine = WholeCardResidencyMachine()
        forecast = StreamForecast(
            weights_mb=13_000.0,
            reserve_mb=1_500.0,
            free_now_mb=1_000.0,
            free_if_alone_mb=14_500.0,
            free_after_model_evict_mb=10_000.0,
            total_vram_mb=16_000.0,
            per_process_overhead_mb=1_000.0,
        )
        assert not machine.teardown_complete(
            forecast,
            loaded_process_count=2,
            safety_pause_required=False,
            safety_paused=False,
            weights_fit_live=True,
            drain_backstop_elapsed=False,
        )
        assert not machine.teardown_complete(
            forecast,
            loaded_process_count=1,
            safety_pause_required=True,
            safety_paused=False,
            weights_fit_live=True,
            drain_backstop_elapsed=False,
        )
        assert machine.teardown_complete(
            forecast,
            loaded_process_count=1,
            safety_pause_required=True,
            safety_paused=True,
            weights_fit_live=False,
            drain_backstop_elapsed=True,
        )
