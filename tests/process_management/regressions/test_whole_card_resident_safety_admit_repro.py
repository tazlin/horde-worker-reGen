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

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _SAFETY_GPU_LOAD_CHARGE_MB,
    _WHOLE_CARD_DRAIN_SETTLE_SECONDS,
    InferenceScheduler,
)
from tests.process_management.conftest import make_mock_bridge_data, make_mock_process_info
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
_FREE_WITH_SAFETY_RESIDENT_MB = _FREE_IF_ALONE_MB - _SAFETY_GPU_LOAD_CHARGE_MB


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
        now=time.time() - (_WHOLE_CARD_DRAIN_SETTLE_SECONDS + 5.0),
        refresh_established=True,
    )
    return scheduler


class TestResidentSafetyBlocksTheBackstopAdmit:
    """The drain backstop stands in only for a card the residency has actually cleared."""

    def test_backstop_does_not_admit_while_safety_keeps_its_context(self) -> None:
        """With safety left on the card by configuration, the elapsed backstop must not release the head."""
        scheduler = _residency_scheduler(
            safety_off_gpu_configured=False,
            safety_paused=False,
            measured_free_mb=_FREE_WITH_SAFETY_RESIDENT_MB,
        )
        forecast = _whole_card_forecast()

        assert forecast.fits_alone is True, (
            "precondition: the head fits a card it has entirely to itself, which is what the backstop leans on"
        )
        assert scheduler._residency_should_pause_safety(None) is False, (
            "precondition: the configuration forbids moving safety off-GPU for this residency"
        )
        assert scheduler._process_lifecycle.is_safety_gpu_paused is False, (
            "precondition: safety is still holding its context on the card"
        )
        assert scheduler._whole_card_weights_fit_live(forecast) is False, (
            "precondition: the live reading does not hold the weights while safety keeps its context"
        )
        assert scheduler._whole_card_drain_backstop_elapsed(None) is True, (
            "precondition: the bounded drain window has been spent, so only the fallback can release the head"
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
