"""The whole-card dispatch gate must not admit a head against room only a departed safety context would free.

A whole-card residency's dispatch gate has two ways to release the head: the *live* free-VRAM reading confirms
the weights fit, or a bounded drain backstop elapses and the head is admitted on the forecast's ``fits_alone``
guarantee. ``fits_alone`` is sized from ``free_if_alone_mb``, which deliberately excludes the safety process's
context: it describes a card on which safety has moved off-GPU.

When ``whole_card_residency_safety_off_gpu`` is disabled the residency never asks safety to leave, so the
structural half of the gate passes with safety still holding its context on the card. The backstop fallback
then admits the head against an ``fits_alone`` figure that assumes a departure the configuration forbids, and
the head loads into a card short by roughly the safety footprint, streaming its weights.

The invariant: the backstop fallback may only stand in for a card the residency has actually cleared. Where
safety keeps its context, the gate must hold the head until the live-fit check passes (or price the resident
context into the alone figure), so a configuration corner never turns the deterministic backstop into an
over-commit.

The seam under test is :meth:`InferenceScheduler._whole_card_teardown_exhausted` rather than the pure
:meth:`WholeCardResidencyMachine.teardown_complete` it delegates to: the scheduler is where the resident-safety
fact (the configuration plus the lifecycle's pause state) is known, so the desired behavior is expressible here
without presuming the shape of the extra input the pure query will need.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import MEMORY_REPORT_INTERVAL_SECONDS
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    WHOLE_CARD_DRAIN_SETTLE_SECONDS,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.ledgers.safety_placement import SAFETY_GPU_LOAD_CHARGE_MB
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_HEAD_MODEL = "Flux.1-Schnell fp8 (Compact)"

# A tight card whose whole-card head fits alone with little to spare, which is the regime where the safety
# context is the difference between fitting and streaming.
_CARD_TOTAL_MB = 16000.0
_PER_PROCESS_OVERHEAD_MB = 1354.0
_FREE_IF_ALONE_MB = _CARD_TOTAL_MB - _PER_PROCESS_OVERHEAD_MB
_WEIGHTS_MB = 11500.0
_BASE_RESERVE_MB = 3100.0
# The live reading on a card the residency has cleared of everything except safety: the weights plus their
# bounded reserve overrun it by roughly the safety context.
_FREE_WITH_SAFETY_RESIDENT_MB = _FREE_IF_ALONE_MB - SAFETY_GPU_LOAD_CHARGE_MB


def _whole_card_forecast() -> StreamForecast:
    """An establishment forecast whose weights fit the card only once every other context has left it."""
    return StreamForecast(
        weights_mb=_WEIGHTS_MB,
        reserve_mb=4646.0,
        base_reserve_mb=_BASE_RESERVE_MB,
        free_now_mb=9129.0,
        free_if_alone_mb=_FREE_IF_ALONE_MB,
        free_after_model_evict_mb=9605.0,
        total_vram_mb=_CARD_TOTAL_MB,
        per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        marginal_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        wants_whole_card=True,
    )


def _residency_scheduler(
    *,
    safety_off_gpu_configured: bool,
    safety_paused: bool,
    measured_free_mb: float,
) -> InferenceScheduler:
    """A single-GPU scheduler holding a whole-card residency whose drain backstop has already elapsed.

    The residency is stamped into the past so the bounded drain window is spent, which is the state in which
    the gate falls back to the structural ``fits_alone`` guarantee. Only the head's own process is live, so the
    structural process-count half of the gate is satisfied.
    """
    bridge_data = make_mock_bridge_data(
        enable_vram_budget=True,
        whole_card_exclusive_residency=True,
        whole_card_residency_safety_off_gpu=safety_off_gpu_configured,
        safety_on_gpu=True,
        whole_card_residency_cooldown_seconds=45,
        image_models_to_load=[_HEAD_MODEL],
    )
    head_process = make_mock_process_info(
        0,
        model_name=_HEAD_MODEL,
        state=HordeProcessState.PRELOADED_MODEL,
        device_index=0,
    )
    # The child VRAM reports are what the live free-VRAM reading is derived from.
    head_process.total_vram_mb = int(_CARD_TOTAL_MB)
    head_process.vram_usage_mb = int(_CARD_TOTAL_MB - measured_free_mb)
    process_map = ProcessMap({0: head_process})
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        bridge_data=bridge_data,
        max_inference=4,
        device_free_mb=measured_free_mb,
    )
    lifecycle = Mock()
    lifecycle.is_safety_gpu_paused = safety_paused
    lifecycle.post_process_lane_enabled = Mock(return_value=False)
    lifecycle.component_lane_enabled = Mock(return_value=False)
    scheduler._process_lifecycle = lifecycle
    scheduler._whole_card_ledger.record_grant(
        None,
        model=_HEAD_MODEL,
        forecast=_whole_card_forecast(),
        cooldown_until=time.time() + 45.0,
        now=time.time() - (WHOLE_CARD_DRAIN_SETTLE_SECONDS + 5.0),
    )
    # The backstop runs from structural completion, so age that stamp rather than the establishment: the
    # scenario under test is a teardown that finished long ago and whose drain the live reading never confirmed.
    scheduler._whole_card_ledger.state_for(None).structural_complete_at = time.time() - (
        WHOLE_CARD_DRAIN_SETTLE_SECONDS + 5.0
    )
    return scheduler


class TestResidentSafetyBlocksTheBackstopAdmit:
    """The drain backstop stands in only for a card the residency has actually cleared."""

    def test_backstop_does_not_admit_while_safety_keeps_its_context(self) -> None:
        """With safety still on the card, the elapsed backstop must not release a head that cannot fit beside it.

        The configuration keeps safety on-GPU, but this head's residency cannot fit beside safety and the live
        reading does not hold its weights, so the residency moves safety off rather than waiting out the runtime
        placement's pressure dwell, and the head stays parked until safety has left.
        """
        scheduler = _residency_scheduler(
            safety_off_gpu_configured=False,
            safety_paused=False,
            measured_free_mb=_FREE_WITH_SAFETY_RESIDENT_MB,
        )
        forecast = _whole_card_forecast()

        assert forecast.fits_alone is True, (
            "precondition: the head fits a card it has entirely to itself, which is what the backstop leans on"
        )
        assert scheduler._whole_card_safety_off_gpu_enabled() is False, (
            "precondition: the configuration keeps safety on-GPU for residencies"
        )
        assert scheduler._residency_should_pause_safety(None) is True, (
            "a residency whose model cannot fit beside safety moves it off despite the configuration"
        )
        assert scheduler._process_lifecycle.is_safety_gpu_paused is False, (
            "precondition: safety is still holding its context on the card"
        )
        assert scheduler._whole_card_weights_fit_live(forecast) is False, (
            "precondition: the live reading does not hold the weights while safety keeps its context"
        )
        assert scheduler._whole_card_drain_settled(None) is True, (
            "precondition: the drain upper bound has been spent, so only the fallback can release the head"
        )

        assert scheduler._whole_card_teardown_exhausted(forecast) is False, (
            "the head must not be admitted on an alone-figure that assumes safety left a card it still occupies; "
            "doing so loads the weights into a card short by roughly the safety footprint"
        )

    def test_gate_completes_once_the_live_reading_holds_the_weights(self) -> None:
        """A card whose live free VRAM genuinely holds the weights releases the head regardless of safety."""
        scheduler = _residency_scheduler(
            safety_off_gpu_configured=False,
            safety_paused=False,
            measured_free_mb=_FREE_IF_ALONE_MB,
        )
        forecast = _whole_card_forecast()

        assert scheduler._whole_card_weights_fit_live(forecast) is True
        assert scheduler._residency_should_pause_safety(None) is False, (
            "the weights fit beside safety on the live reading, so safety keeps its GPU context"
        )
        assert scheduler._whole_card_teardown_exhausted(forecast) is True


class TestSafetyPauseGateStillHoldsAndReleases:
    """Where the residency does move safety off-GPU, the structural gate is unchanged."""

    def test_gate_holds_while_a_required_safety_pause_has_not_taken_effect(self) -> None:
        """A residency that must move safety off its card holds the head until safety is actually off."""
        scheduler = _residency_scheduler(
            safety_off_gpu_configured=True,
            safety_paused=False,
            measured_free_mb=_FREE_IF_ALONE_MB,
        )
        forecast = _whole_card_forecast()

        assert scheduler._residency_should_pause_safety(None) is True
        assert scheduler._whole_card_teardown_exhausted(forecast) is False

    def test_gate_completes_once_the_required_safety_pause_is_performed(self) -> None:
        """With safety off the card and the live reading holding the weights, the head is released."""
        scheduler = _residency_scheduler(
            safety_off_gpu_configured=True,
            safety_paused=True,
            measured_free_mb=_FREE_IF_ALONE_MB,
        )
        forecast = _whole_card_forecast()

        assert scheduler._whole_card_weights_fit_live(forecast) is True
        assert scheduler._whole_card_teardown_exhausted(forecast) is True


class TestSettledDrainReleasesTheHead:
    """A teardown whose free reading has stopped rising releases the head within one report interval."""

    def test_a_flat_reading_releases_the_head_long_before_the_upper_bound(self) -> None:
        """The live reading never holds the weights, yet a flat reading settles the drain and the head goes."""
        scheduler = _residency_scheduler(
            safety_off_gpu_configured=True,
            safety_paused=True,
            measured_free_mb=_FREE_WITH_SAFETY_RESIDENT_MB,
        )
        forecast = _whole_card_forecast()
        now = [time.time()]
        scheduler._clock = lambda: now[0]
        structural_at = now[0]
        scheduler._whole_card_ledger.state_for(None).structural_complete_at = structural_at

        assert scheduler._whole_card_weights_fit_live(forecast) is False, (
            "precondition: the live reading does not hold the weights, so only the drain can release the head"
        )
        assert scheduler._whole_card_teardown_exhausted(forecast) is False, (
            "the first reading after the structural teardown is the reference, so it cannot settle the drain"
        )

        now[0] += MEMORY_REPORT_INTERVAL_SECONDS
        assert scheduler._whole_card_teardown_exhausted(forecast) is True, (
            "a reading that has not risen across one report interval settles the drain"
        )
        assert now[0] - structural_at < WHOLE_CARD_DRAIN_SETTLE_SECONDS


class TestResidencySafetyMoveIsLatched:
    """A residency that moves safety off keeps it off until the residency ends."""

    def test_the_move_holds_once_safety_has_left(self) -> None:
        """After safety leaves, the live reading holds the weights, yet the decision must not bring safety back."""
        scheduler = _residency_scheduler(
            safety_off_gpu_configured=False,
            safety_paused=False,
            measured_free_mb=_FREE_WITH_SAFETY_RESIDENT_MB,
        )
        assert scheduler._residency_should_pause_safety(None) is True

        scheduler._process_lifecycle.is_safety_gpu_paused = True
        for process_info in scheduler._process_map.values():
            process_info.vram_usage_mb = int(_CARD_TOTAL_MB - _FREE_IF_ALONE_MB)
        assert scheduler._residency_should_pause_safety(None) is True, (
            "the move was undone by the room it freed, which would cycle safety on and off the card"
        )

        scheduler._whole_card_ledger.state_for(None).model = None
        assert scheduler._residency_should_pause_safety(None) is False, "the latch outlived its residency"


_LIGHT_WEIGHTS_MB = 7000.0
"""Weights whose bounded reserve fits the emptied card beside safety, in the structural frame and the live one."""


def _light_weights_forecast() -> StreamForecast:
    """A residency forecast whose weights fit beside safety, so a weight test alone keeps safety on the card."""
    return StreamForecast(
        weights_mb=_LIGHT_WEIGHTS_MB,
        reserve_mb=4646.0,
        base_reserve_mb=_BASE_RESERVE_MB,
        free_now_mb=_FREE_WITH_SAFETY_RESIDENT_MB,
        free_if_alone_mb=_FREE_IF_ALONE_MB,
        free_after_model_evict_mb=9605.0,
        total_vram_mb=_CARD_TOTAL_MB,
        per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        marginal_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        wants_whole_card=True,
    )


async def _residency_with_queued_job(
    monkeypatch: pytest.MonkeyPatch,
    *,
    job_need_mb: float,
    seat_mb: float | None = None,
) -> InferenceScheduler:
    """A residency whose weights fit beside safety, with one queued job of its model priced at ``job_need_mb``.

    ``seat_mb`` is the job's partial-load seat, None for a job clearance cannot seat partially (a LoRA job).
    """
    from horde_worker_regen.process_management.scheduling import inference_scheduler as scheduler_module
    from horde_worker_regen.process_management.scheduling.admission import pricing

    monkeypatch.setattr(scheduler_module, "predict_job_sampling_vram_mb", lambda _job, _baseline: job_need_mb)
    monkeypatch.setattr(pricing, "partial_seat_mb", lambda *_args, **_kwargs: seat_mb)
    scheduler = _residency_scheduler(
        safety_off_gpu_configured=False,
        safety_paused=False,
        measured_free_mb=_FREE_WITH_SAFETY_RESIDENT_MB,
    )
    monkeypatch.setattr(scheduler, "snapshot", Mock())
    state = scheduler._whole_card_ledger.state_for(None)
    state.forecast = _light_weights_forecast()
    await track_popped_job_async(scheduler._job_tracker, make_job_pop_response(_HEAD_MODEL))  # type: ignore[attr-defined]
    assert state.forecast.fits_alone_beside(scheduler._safety_footprint_mb()) is True, (
        "precondition: the weights fit the emptied card beside safety"
    )
    assert scheduler._whole_card_weights_fit_live(state.forecast) is True, (
        "precondition: the weights fit the live reading with safety on the card"
    )
    return scheduler


class TestResidencySafetyMoveReadsTheJobsNeed:
    """A residency moves safety off when its job's priced need, not its weights, cannot fit beside safety.

    The failure this encodes: the move tested the residency's weights against the room beside safety, while
    clearance prices the job's whole sampling need. A large model whose weights fit beside safety kept safety on
    the card, clearance then held its job for the room safety took, and the card stood idle until the runtime
    placement's sustained-pressure dwell moved safety anyway.
    """

    async def test_a_job_whose_need_cannot_fit_beside_safety_moves_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Weights that fit beside safety do not keep it there when the queued job's need does not."""
        room_mb = _FREE_IF_ALONE_MB - SAFETY_GPU_LOAD_CHARGE_MB
        scheduler = await _residency_with_queued_job(monkeypatch, job_need_mb=room_mb + 900.0)

        assert scheduler._residency_should_pause_safety(None) is True

    async def test_a_job_clearance_seats_partially_keeps_safety_on_the_gpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job whose whole need does not fit but whose partial-load seat does is admitted beside safety."""
        room_mb = _FREE_IF_ALONE_MB - SAFETY_GPU_LOAD_CHARGE_MB
        scheduler = await _residency_with_queued_job(monkeypatch, job_need_mb=room_mb + 900.0, seat_mb=room_mb - 900.0)

        assert scheduler._residency_should_pause_safety(None) is False

    async def test_a_job_that_fits_beside_safety_keeps_it_on_the_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A queued job whose need fits beside safety leaves the operator's keep-safety-on-GPU choice standing."""
        room_mb = _FREE_IF_ALONE_MB - SAFETY_GPU_LOAD_CHARGE_MB
        scheduler = await _residency_with_queued_job(monkeypatch, job_need_mb=room_mb - 900.0)

        assert scheduler._residency_should_pause_safety(None) is False

    async def test_defect_reinjection_a_weight_test_keeps_safety_beside_a_job_it_starves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Defect reinjection: with the job's need unread, the weights decide and safety stays on the card."""
        room_mb = _FREE_IF_ALONE_MB - SAFETY_GPU_LOAD_CHARGE_MB
        scheduler = await _residency_with_queued_job(monkeypatch, job_need_mb=room_mb + 900.0)
        monkeypatch.setattr(scheduler, "_residency_head_need_mb", lambda _model, **_kwargs: None)

        assert scheduler._residency_should_pause_safety(None) is False
