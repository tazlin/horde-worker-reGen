"""Tests for the whole-card pop-claim tracker, governor hold and pure process-set queries."""

from __future__ import annotations

from unittest.mock import patch

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.scheduling.governance import whole_card as whole_card_module
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    POP_CLAIM_RELEASE_VISIBLE_SECONDS,
    WholeCardGovernor,
    WholeCardPopClaim,
    WholeCardPopClaimRelease,
    WholeCardPopClaimTracker,
    WholeCardResidencyLedger,
    convergence_blockers,
    residency_has_holder,
)
from tests.process_management.conftest import make_mock_process_info


def _slot(
    process_id: int,
    *,
    model: str | None,
    device_index: int = 0,
    busy: bool = False,
    process_type: HordeProcessType = HordeProcessType.INFERENCE,
) -> HordeProcessInfo:
    info = make_mock_process_info(process_id, model_name=model, process_type=process_type, device_index=device_index)
    info.last_process_state = HordeProcessState.INFERENCE_STARTING if busy else HordeProcessState.WAITING_FOR_JOB
    return info


def _claim(model: str = "flux", *, held_since: float = 100.0, expires_at: float = 700.0) -> WholeCardPopClaim:
    return WholeCardPopClaim(model=model, device_index=None, held_since=held_since, expires_at=expires_at)


class TestPopClaimTracker:
    """Engage and release lines fire on edges, and the release reason stays visible for a bounded time."""

    def test_engage_is_stated_once_per_claim(self) -> None:
        """A claim held across many ticks logs one engage line."""
        tracker = WholeCardPopClaimTracker()
        with patch.object(whole_card_module.logger, "info") as info:
            tracker.disclose_edge(_claim(), now=100.0)
            tracker.disclose_edge(_claim(), now=101.0)
        assert info.call_count == 1
        assert "engaged" in str(info.call_args_list[0])

    def test_release_names_the_end_that_fired(self) -> None:
        """Empty-pop evidence, the maximum hold and a plain residency release are told apart."""
        cases = [
            (True, 200.0, WholeCardPopClaimRelease.NO_FURTHER_WORK),
            (False, 800.0, WholeCardPopClaimRelease.MAXIMUM_HOLD),
            (False, 200.0, WholeCardPopClaimRelease.RESIDENCY_RELEASED),
        ]
        for empty_release, now, expected in cases:
            tracker = WholeCardPopClaimTracker()
            tracker.disclose_edge(_claim(), now=100.0)
            if empty_release:
                tracker.note_empty_pop_release()
            with patch.object(whole_card_module.logger, "info") as info:
                tracker.disclose_edge(None, now=now)
            assert info.call_count == 1, expected
            assert tracker.recent_release(now) is expected

    def test_no_claim_and_nothing_disclosed_is_silent(self) -> None:
        """An unclaimed offer that was never claimed logs nothing and explains nothing."""
        tracker = WholeCardPopClaimTracker()
        with patch.object(whole_card_module.logger, "info") as info:
            tracker.disclose_edge(None, now=5.0)
        assert info.call_count == 0
        assert tracker.recent_release(5.0) is None

    def test_release_reason_expires(self) -> None:
        """Past the visibility window the release no longer explains the offer."""
        tracker = WholeCardPopClaimTracker()
        tracker.disclose_edge(_claim(), now=100.0)
        tracker.disclose_edge(None, now=200.0)
        assert tracker.recent_release(200.0 + POP_CLAIM_RELEASE_VISIBLE_SECONDS - 1.0) is not None
        assert tracker.recent_release(200.0 + POP_CLAIM_RELEASE_VISIBLE_SECONDS) is None

    def test_multi_gpu_skip_is_stated_once_and_only_while_held(self) -> None:
        """The inapplicability notice waits for a held residency and then speaks once."""
        tracker = WholeCardPopClaimTracker()
        with patch.object(whole_card_module.logger, "info") as info:
            tracker.disclose_skipped_on_multi_gpu(any_held=False)
            tracker.disclose_skipped_on_multi_gpu(any_held=True)
            tracker.disclose_skipped_on_multi_gpu(any_held=True)
        assert info.call_count == 1


class TestGovernorHold:
    """The ledger names the governor barring a new residency."""

    def test_free_card_has_no_hold(self) -> None:
        """A card that has not established recently is free to take a residency."""
        assert WholeCardResidencyLedger().governor_hold(None, now=1000.0) is None

    def test_rate_limit_is_named_first(self) -> None:
        """Establishing as often as the window allows names the rate governor."""
        ledger = WholeCardResidencyLedger()
        for _ in range(whole_card_module._ESTABLISH_WINDOW_LIMIT):
            ledger.record_grant(None, model="flux", forecast=None, cooldown_until=0.0, now=1000.0)
            ledger.record_restore(None, now=1000.0)
        hold = ledger.governor_hold(None, now=1001.0)
        assert hold is not None
        assert hold.governor is WholeCardGovernor.ESTABLISH_RATE
        assert hold.detail is None


class TestPureQueries:
    """Holder and blocker queries read only the process set."""

    def test_residency_has_holder_matches_model_type_and_card(self) -> None:
        """Only an inference process on the card holding the model counts."""
        holder = _slot(1, model="flux", device_index=1)
        lane = _slot(2, model="flux", process_type=HordeProcessType.COMPONENT)
        other = _slot(3, model="sdxl")
        assert residency_has_holder([holder, lane, other], "flux", None) is True
        assert residency_has_holder([holder, lane, other], "flux", 1) is True
        assert residency_has_holder([holder, lane, other], "flux", 0) is False
        assert residency_has_holder([lane, other], "flux", None) is False

    def test_convergence_blockers_are_idle_siblings_holding_queued_models(self) -> None:
        """The head's holder, busy slots, other cards and unqueued models are not blockers."""
        head = _slot(1, model="flux")
        idle_queued = _slot(2, model="sdxl")
        busy_queued = _slot(3, model="sdxl", busy=True)
        idle_unqueued = _slot(4, model="sd15")
        other_card = _slot(5, model="sdxl", device_index=1)
        blockers = convergence_blockers(
            [head, idle_queued, busy_queued, idle_unqueued, other_card],
            head_process_id=1,
            device_index=0,
            queued_models={"flux", "sdxl"},
        )
        assert blockers == [(2, "sdxl")]
