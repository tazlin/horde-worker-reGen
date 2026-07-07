"""Regression guard: a card-light model must not be driven onto the whole-card residency path.

The condition this reproduces: a large-VRAM card (24GB here) serving only SDXL checkpoints, with
``whole_card_exclusive_residency`` on, ``safety_on_gpu`` on, and two threads. The worker reserves the whole
card for an SDXL model, reduces the live process count, and cycles safety off and on the GPU; sustained, that
churn can starve a queue head long enough for the recovery supervisor to soft-reset and fault the backlog,
which the horde answers with forced maintenance.

The reservation here comes from the *verdict-driven* path, not the streaming forecast: the forecast reads
``needs_teardown=False`` (SDXL co-resides after evicting a sibling model), yet the VRAM budget verdict rejects
the head and ``_max_coresident_for_peak_mb`` sizes a process-count reduction. Both the verdict's conservative
burden peak and the reduction depth rest on the per-process context overhead, which is over-counted when it is
the unmeasured fallback: the one-time CUDA runtime/first-context cost is charged against every live context
(the inference contexts plus the on-GPU safety context), collapsing the structural free-VRAM floor below the
model's footprint. Counted with the true per-additional-context marginal (a fraction of the first context),
the model co-resides on a large card with room to spare and the residency never engages.

Two behaviors compound:

  1. *A card-light model should never reserve the card.* A checkpoint whose weights are a small fraction of
     total VRAM co-resides; the whole-card path engages for it only when the per-context overhead is an
     unmeasured fallback (no probe marginal, no clean idle baseline) that over-counts contexts.
  2. *A reservation outlives its head.* Held through the configured cooldown (and while its model stays
     queued), the reservation keeps the card for a model that is no longer the head, so a later head of a
     different model parks behind it, the queue deadlocks, and the pools soft-reset.

The fix gates both whole-card establish paths (the forecast-driven ``needs_teardown_path`` and the
verdict-driven context reduction) on the demand being trustworthy: the model is genuinely card-demanding (its
footprint dominates the device, or its baseline wants the whole card on intent) *or* the per-context cost was
actually measured (so the contention is real, not an over-count). A card-light model on a host that could not
measure the marginal falls through to ordinary model eviction instead of reserving the device.

These tests pin: the card-fraction classification (``StreamForecast.is_card_demanding``), the scheduler's
trust gate (``_whole_card_warranted``), that the verdict-driven establish is vetoed for a card-light model
with an unmeasured marginal, that the forecast-driven establish is likewise vetoed, and (when a teardown
*is* warranted) that only the head may claim the card (the ``is_head_blocker`` backstop). The dispatch
diagnostic attribution is pinned too.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

# A representative 24GB card and the figures a forecast logs for it (a first-context overhead near 3815MB,
# SDXL weights ~4900, sole-residency-free ~20259, four inference contexts plus the on-GPU safety context).
_DEVICE_TOTAL_VRAM_MB = 24074.0
_PER_PROCESS_OVERHEAD_MB = 3815.0
_SDXL_WEIGHTS_MB = 4900.0
_SDXL_BASE_RESERVE_MB = 1800.0
_FLUX_FP8_WEIGHTS_MB = 11500.0  # a genuinely card-demanding (but not whole-card-intent) checkpoint
_SMALL_CARD_TOTAL_VRAM_MB = 10240.0  # a card on which SDXL *does* contend, so a teardown is legitimate there
_TRUE_MARGINAL_MB = 600.0  # a realistic per-additional-context cost; far smaller than the first context
_LIVE_CONTEXT_COUNT = 4

# Representative model names: a starved head and a second model that the card was reserved for.
_HEAD_SDXL = "WAI-NSFW-illustrious-SDXL"
_HEAVY_SDXL = "AlbedoBase XL (SDXL)"
_OTHER_SDXL = "Juggernaut XL"


def _forecast(
    *,
    weights_mb: float | None,
    total_vram_mb: float | None,
    base_reserve_mb: float = _SDXL_BASE_RESERVE_MB,
    wants_whole_card: bool = False,
    needs_pcr_shape: bool = False,
) -> StreamForecast:
    """Build a StreamForecast for the unit/gate tests.

    ``needs_pcr_shape`` sets the free figures so the result reads ``needs_process_count_reduction`` (fits
    alone, fails the structural floor, not weight-dominant); otherwise it reads comfortably co-resident.
    """
    if needs_pcr_shape:
        free_now_mb = 3000.0
        free_after_model_evict_mb = (weights_mb or 0.0) + base_reserve_mb - 1000.0
        free_if_alone_mb = (total_vram_mb or 0.0) - _PER_PROCESS_OVERHEAD_MB
        reserve_mb = 8000.0
    else:
        free_now_mb = (total_vram_mb or 0.0) - _PER_PROCESS_OVERHEAD_MB
        free_after_model_evict_mb = free_now_mb
        free_if_alone_mb = free_now_mb
        reserve_mb = base_reserve_mb
    return StreamForecast(
        weights_mb=weights_mb,
        reserve_mb=reserve_mb,
        free_now_mb=free_now_mb,
        free_if_alone_mb=free_if_alone_mb,
        free_after_model_evict_mb=free_after_model_evict_mb,
        total_vram_mb=total_vram_mb,
        per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        marginal_process_overhead_mb=_TRUE_MARGINAL_MB,
        wants_whole_card=wants_whole_card,
        base_reserve_mb=base_reserve_mb,
    )


def _idle_context_map(num_processes: int, *, free_mb: float, safety_on_gpu: bool = False) -> ProcessMap:
    """A map of idle, model-free inference contexts reporting a device-wide free VRAM (incident topology)."""
    procs: dict[int, HordeProcessInfo] = {}
    used_mb = _DEVICE_TOTAL_VRAM_MB - free_mb
    for pid in range(1, num_processes + 1):
        proc = make_mock_process_info(pid, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
        proc.vram_usage_mb = used_mb
        procs[pid] = proc
    return ProcessMap(procs)


def _wedge_bridge_data(*, safety_on_gpu: bool = True) -> Mock:
    """Budget-on, whole-card-on, two-thread, 24GB configuration: the high-VRAM, all-SDXL case under test."""
    return make_mock_bridge_data(
        enable_vram_budget=True,
        whole_card_exclusive_residency=True,
        whole_card_residency_safety_off_gpu=safety_on_gpu,
        safety_on_gpu=safety_on_gpu,
        vram_reserve_mb=2048,
        ram_reserve_mb=4096,
        vram_per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        overbudget_exclusive_mode=True,
        whole_card_residency_cooldown_seconds=45,
        max_threads=2,
    )


def _make_scheduler(
    process_map: ProcessMap,
    job_tracker: JobTracker,
    *,
    measured_marginal_mb: float | None = None,
    safety_on_gpu: bool = True,
) -> InferenceScheduler:
    """An InferenceScheduler with the OS-level levers stubbed; optionally fed a measured per-context marginal.

    Leaving ``measured_marginal_mb`` None reproduces a host where the marginal could not be measured (no probe
    figure, no clean idle baseline), so the forecast falls back to the over-counted first-context overhead per
    context.
    """
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=_wedge_bridge_data(safety_on_gpu=safety_on_gpu),
        max_concurrent=1,
        max_inference=4,
    )
    scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=len(list(process_map.values())))
    scheduler._process_lifecycle.pause_safety_on_gpu = Mock(return_value=True)
    scheduler._process_lifecycle.is_safety_gpu_paused = False
    scheduler._measured_available_ram_mb = lambda: 48000.0  # type: ignore[method-assign]
    if measured_marginal_mb is not None:
        scheduler.set_measured_marginal_overhead_mb(measured_marginal_mb)
    return scheduler


def _install_forecast_per_model(scheduler: InferenceScheduler, forecast_for: dict[str, StreamForecast]) -> None:
    """Make the scheduler return a chosen forecast per model name (default: co-resident)."""

    def _forecast_streaming(
        job: ImageGenerateJobPopResponse,
        baseline: object,
        *,
        device_index: int | None = None,
    ) -> StreamForecast:
        if job.model in forecast_for:
            return forecast_for[job.model]
        return _forecast(weights_mb=_SDXL_WEIGHTS_MB, total_vram_mb=_DEVICE_TOTAL_VRAM_MB)

    scheduler._forecast_streaming = _forecast_streaming  # type: ignore[method-assign]


def _resident(pid: int, model: str | None, state: HordeProcessState) -> HordeProcessInfo:
    """A process pinned to the 24GB device with a low device-free reading (siblings hold the card)."""
    proc = make_mock_process_info(pid, model_name=model, state=state)
    proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
    proc.vram_usage_mb = _DEVICE_TOTAL_VRAM_MB - 2000
    if model is not None:
        # A fresh committed reservation matching the held card, so the arbiter's measured floor denies a
        # non-head candidate against the resident's footprint rather than relaxing to admit on a cold ledger.
        proc.process_reserved_mb = 16000
        proc.report_sampled_at = time.time()
    return proc


def _residency_recorded(scheduler: InferenceScheduler, model: str) -> bool:
    """Whether a whole-card residency is held for ``model`` (the scheduler reserved the card for it)."""
    found, _device = scheduler._residency_holder_for_model(model)
    return found


# --------------------------------------------------------------------------------------------------------- #
#  The card-fraction classification: SDXL is not card-demanding on a big card, but is on a small one.         #
# --------------------------------------------------------------------------------------------------------- #


class TestIsCardDemanding:
    """``StreamForecast.is_card_demanding`` decides whether reserving the whole card could ever be warranted."""

    def test_sdxl_is_not_card_demanding_on_a_24gb_card(self) -> None:
        """An ~4.9GB SDXL checkpoint is a small share of a 24GB card; it co-resides and never needs it alone."""
        forecast = _forecast(weights_mb=_SDXL_WEIGHTS_MB, total_vram_mb=_DEVICE_TOTAL_VRAM_MB)
        assert not forecast.is_card_demanding

    def test_flux_fp8_is_card_demanding_on_a_24gb_card(self) -> None:
        """A heavier checkpoint whose footprint dominates the card is teardown-eligible even off-intent."""
        forecast = _forecast(weights_mb=_FLUX_FP8_WEIGHTS_MB, total_vram_mb=_DEVICE_TOTAL_VRAM_MB)
        assert forecast.is_card_demanding

    def test_sdxl_is_card_demanding_on_a_small_card(self) -> None:
        """The same SDXL genuinely contends on a 10GB card, so a teardown there is legitimate."""
        forecast = _forecast(weights_mb=_SDXL_WEIGHTS_MB, total_vram_mb=_SMALL_CARD_TOTAL_VRAM_MB)
        assert forecast.is_card_demanding

    def test_wants_whole_card_baseline_is_always_card_demanding(self) -> None:
        """A whole-card-intent baseline qualifies even when its weight seed reads light (the tier asserts it)."""
        forecast = _forecast(weights_mb=1000.0, total_vram_mb=_DEVICE_TOTAL_VRAM_MB, wants_whole_card=True)
        assert forecast.is_card_demanding

    def test_unsized_footprint_is_conservatively_card_demanding(self) -> None:
        """When the footprint cannot be sized, stay conservative (preserve the prior eager behavior)."""
        forecast = _forecast(weights_mb=None, total_vram_mb=_DEVICE_TOTAL_VRAM_MB)
        assert forecast.is_card_demanding


# --------------------------------------------------------------------------------------------------------- #
#  The scheduler's trust gate: a teardown demand must be card-demanding or rest on a measured marginal.       #
# --------------------------------------------------------------------------------------------------------- #


class TestWholeCardWarranted:
    """``_whole_card_warranted`` is what stops an unmeasured over-count reserving the card for a light model."""

    def _scheduler(self, *, measured_marginal_mb: float | None) -> InferenceScheduler:
        process_map = _idle_context_map(_LIVE_CONTEXT_COUNT, free_mb=4709.0)
        return _make_scheduler(process_map, JobTracker(), measured_marginal_mb=measured_marginal_mb)

    def test_card_light_model_with_unmeasured_marginal_is_not_warranted(self) -> None:
        """A card-light SDXL on a host with no measured marginal must not be trusted to demand a teardown."""
        scheduler = self._scheduler(measured_marginal_mb=None)
        assert scheduler._marginal_process_overhead_mb() is None
        forecast = _forecast(weights_mb=_SDXL_WEIGHTS_MB, total_vram_mb=_DEVICE_TOTAL_VRAM_MB)
        assert not scheduler._whole_card_warranted(forecast)

    def test_card_light_model_with_measured_marginal_is_warranted(self) -> None:
        """A measured per-context cost means the contention is real (the threads=2 retention regime)."""
        scheduler = self._scheduler(measured_marginal_mb=_TRUE_MARGINAL_MB)
        assert scheduler._marginal_process_overhead_mb() is not None
        forecast = _forecast(weights_mb=_SDXL_WEIGHTS_MB, total_vram_mb=_DEVICE_TOTAL_VRAM_MB)
        assert scheduler._whole_card_warranted(forecast)

    def test_card_demanding_model_is_warranted_even_without_a_measured_marginal(self) -> None:
        """A model whose footprint dominates the card is trusted regardless of how contexts were counted."""
        scheduler = self._scheduler(measured_marginal_mb=None)
        forecast = _forecast(weights_mb=_FLUX_FP8_WEIGHTS_MB, total_vram_mb=_DEVICE_TOTAL_VRAM_MB)
        assert scheduler._whole_card_warranted(forecast)


# --------------------------------------------------------------------------------------------------------- #
#  The verdict-driven establish is vetoed for a card-light SDXL with no measured per-context cost.           #
# --------------------------------------------------------------------------------------------------------- #


class TestVerdictDrivenEstablishGatedByWarrant:
    """The verdict-driven path: a budget-rejected head sizing a context reduction from the burden peak."""

    def _run(self, *, marginal_mb: float | None) -> tuple[InferenceScheduler, ImageGenerateJobPopResponse]:
        # Four idle contexts pin device-free low; with no resident models, gentle reclaim frees nothing, so a
        # budget-rejected head reaches the context-reduction sizing: the verdict-driven establish path. The
        # per-context cost is pinned directly (rather than via captured baselines) so the test isolates the
        # warrant gate: ``None`` is the host that could not measure the marginal.
        process_map = _idle_context_map(_LIVE_CONTEXT_COUNT, free_mb=4709.0)
        job_tracker = JobTracker()
        scheduler = _make_scheduler(process_map, job_tracker, safety_on_gpu=False)
        scheduler._marginal_process_overhead_mb = lambda: marginal_mb  # type: ignore[method-assign]
        return scheduler, make_job_pop_response(_HEAD_SDXL)

    async def test_card_light_sdxl_head_unmeasured_marginal_does_not_reserve_card(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A budget-rejected, card-light SDXL head with no measured marginal must not reserve the whole card.

        Without the warrant gate the over-counted fallback sizes a sub-count reduction and establishes a
        residency for the SDXL head; the gate vetoes it, so the head is served by ordinary eviction instead of
        reserving the device on a phantom (and so cannot later starve a different head while held in cooldown).
        """
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda job, baseline: _SDXL_WEIGHTS_MB)
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: 12000.0)
        scheduler, head = self._run(marginal_mb=None)
        await track_popped_job_async(scheduler._job_tracker, head)

        scheduler.preload_models()

        assert not _residency_recorded(scheduler, _HEAD_SDXL), (
            "a card-light SDXL head on a host with no measured per-context cost must not reserve the whole card"
        )
        scheduler._process_lifecycle.pause_safety_on_gpu.assert_not_called()


# --------------------------------------------------------------------------------------------------------- #
#  The forecast-driven establish is gated by the same warrant.                                               #
# --------------------------------------------------------------------------------------------------------- #


class TestForecastDrivenEstablishGatedByWarrant:
    """The other establish path (``needs_teardown_path``) honors the same trust gate."""

    async def test_card_light_sdxl_head_unmeasured_marginal_does_not_reserve_card(self) -> None:
        """A card-light SDXL head whose per-context cost was never measured must not reserve the card.

        Even when the streaming forecast itself reads ``needs_process_count_reduction``, the warrant gate
        vetoes the establish: the demand rests on an unmeasured, over-counted per-context overhead.
        """
        head_proc = _resident(1, None, HordeProcessState.WAITING_FOR_JOB)
        spare = _resident(2, None, HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({1: head_proc, 2: spare})
        job_tracker = JobTracker()
        scheduler = _make_scheduler(process_map, job_tracker, measured_marginal_mb=None)
        pcr_forecast = _forecast(
            weights_mb=_SDXL_WEIGHTS_MB,
            total_vram_mb=_DEVICE_TOTAL_VRAM_MB,
            needs_pcr_shape=True,
        )
        assert pcr_forecast.needs_process_count_reduction
        _install_forecast_per_model(scheduler, {_HEAD_SDXL: pcr_forecast})

        await track_popped_job_async(job_tracker, make_job_pop_response(_HEAD_SDXL))

        scheduler.preload_models()

        assert not _residency_recorded(scheduler, _HEAD_SDXL)


# --------------------------------------------------------------------------------------------------------- #
#  When a teardown *is* warranted, only the head may claim the card (the is_head_blocker backstop).           #
# --------------------------------------------------------------------------------------------------------- #


class TestWarrantedTeardownClaimsCardOnlyForHead:
    """With the demand warranted (a measured marginal proving contention), queue order must still be honored."""

    def _scheduler_with_resident_head(self, job_tracker: JobTracker) -> InferenceScheduler:
        head_proc = _resident(1, _HEAD_SDXL, HordeProcessState.WAITING_FOR_JOB)
        spare = _resident(2, None, HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({1: head_proc, 2: spare})
        # A measured marginal makes the teardown trustworthy, so the only remaining gate is queue order.
        scheduler = _make_scheduler(process_map, job_tracker, measured_marginal_mb=_TRUE_MARGINAL_MB)
        pcr_forecast = _forecast(
            weights_mb=_SDXL_WEIGHTS_MB, total_vram_mb=_DEVICE_TOTAL_VRAM_MB, needs_pcr_shape=True
        )
        _install_forecast_per_model(scheduler, {_HEAVY_SDXL: pcr_forecast})
        return scheduler

    async def test_non_head_does_not_claim_card(self) -> None:
        """A warranted-teardown model behind a resident head must defer, not tear the head's process down."""
        job_tracker = JobTracker()
        scheduler = self._scheduler_with_resident_head(job_tracker)

        await track_popped_job_async(job_tracker, make_job_pop_response(_HEAD_SDXL))  # the head
        await track_popped_job_async(job_tracker, make_job_pop_response(_HEAVY_SDXL))

        scheduler.preload_models()

        assert not _residency_recorded(scheduler, _HEAVY_SDXL), (
            "a non-head model must not reserve the card even when its teardown demand is warranted"
        )

    async def test_model_claims_card_once_it_becomes_the_head(self) -> None:
        """The deferral is until the model's turn: once it is the head, the warranted teardown must fire."""
        job_tracker = JobTracker()
        scheduler = self._scheduler_with_resident_head(job_tracker)

        sdxl_head = make_job_pop_response(_HEAD_SDXL)
        await track_popped_job_async(job_tracker, sdxl_head)
        await track_popped_job_async(job_tracker, make_job_pop_response(_HEAVY_SDXL))

        scheduler.preload_models()
        assert not _residency_recorded(scheduler, _HEAVY_SDXL)

        await mark_job_in_progress_async(job_tracker, sdxl_head)
        scheduler.preload_models()

        assert _residency_recorded(scheduler, _HEAVY_SDXL), (
            "once it is the head, a warranted teardown must claim the card"
        )


# --------------------------------------------------------------------------------------------------------- #
#  The stall is attributed to the held non-head residency, not mistaken for a budget defer.                  #
# --------------------------------------------------------------------------------------------------------- #


class TestNonHeadResidencyDispatchDiagnostic:
    """``_diagnose_dispatch_stall`` must name a held non-head residency as the reason a head cannot load."""

    def _scheduler_with_no_residents(self) -> tuple[InferenceScheduler, JobTracker]:
        spare = _resident(2, None, HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({2: spare})
        job_tracker = JobTracker()
        scheduler = _make_scheduler(process_map, job_tracker)
        return scheduler, job_tracker

    async def test_head_stall_attributes_a_held_nonhead_residency(self) -> None:
        """The head's model is not resident while a residency is held for another model: name it as the cause."""
        scheduler, job_tracker = self._scheduler_with_no_residents()
        scheduler._residency_state(None).model = _HEAVY_SDXL
        head = await track_popped_job_async(job_tracker, make_job_pop_response(_HEAD_SDXL))

        reason = scheduler._diagnose_dispatch_stall(head, {})

        assert "whole-card residency is held for non-head model" in reason
        assert _HEAVY_SDXL in reason
        assert "budget defer" not in reason

    async def test_head_stall_without_residency_is_not_misattributed(self) -> None:
        """Control: with no residency held, the not-resident head falls back to the generic reason."""
        scheduler, job_tracker = self._scheduler_with_no_residents()
        head = await track_popped_job_async(job_tracker, make_job_pop_response(_HEAD_SDXL))

        reason = scheduler._diagnose_dispatch_stall(head, {})

        assert "whole-card residency is held for non-head model" not in reason
