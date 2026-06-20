"""Reproduction of the single-residency pre-stage gap: a whole-card head should load into RAM early.

The dynamics (on a 16GB device): the worker is sampling a job for one model (e.g. an SDXL checkpoint) on
one inference process while sibling processes sit idle. A heavy whole-card head (Flux) is popped. Today the
residency machine immediately tears the idle siblings down and then *waits* for the in-flight job to drain
before it preloads Flux at all, so Flux's slow disk->RAM load (~11GB) only begins once the device is free.

That ordering is backwards. ``preload_model`` loads a checkpoint into *system RAM*, not VRAM (the weights
move to VRAM at sampling time), so a spare process can pre-stage Flux into RAM concurrently with the in-flight
job's sampling, RAM permitting. When the device frees, Flux is already resident in RAM and only needs the
fast RAM->VRAM move before it samples.

These tests pin that behavior: while a live job holds the device, a whole-card head must be preloaded into a
spare process's RAM (gated by the RAM budget), and the residency must then converge to sole VRAM residency on
the process that holds the pre-staged model.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management import resource_budget
from horde_worker_regen.process_management.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeControlFlag, HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_map import ProcessMap

from .conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from .test_inference_scheduling import _make_inference_scheduler

# Representative of a 16GB-device scenario (mirrors test_whole_card_residency_repro).
_DEVICE_TOTAL_VRAM_MB = 16375
_FLUX_WEIGHTS_MB = 11500.0
_FLUX_SAMPLING_PEAK_MB = 15218.0  # activation-inclusive peak: Flux needs sole residency but fits alone
_PER_PROCESS_OVERHEAD_MB = 1288
_VRAM_RESERVE_MB = 2048
_RAM_RESERVE_MB = 4096

_FLUX_MODEL = "Flux.1-Schnell fp8 (Compact)"
_FLUX_BASELINE = "flux_schnell"
_RESIDENT_SDXL = "CyberRealistic Pony"
_SDXL_BASELINE = "stable_diffusion_xl"

_NUM_IDLE_SIBLINGS = 3  # a 3-queue (4-process) worker: one busy + three idle, the user's scenario


def _overlap_bridge_data() -> Mock:
    """Budget-on, whole-card-residency-on, 4-process, 16GB configuration."""
    return make_mock_bridge_data(
        enable_vram_budget=True,
        whole_card_exclusive_residency=True,
        vram_reserve_mb=_VRAM_RESERVE_MB,
        ram_reserve_mb=_RAM_RESERVE_MB,
        vram_per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        overbudget_exclusive_mode=True,
        safety_on_gpu=True,
        image_models_to_load=[_RESIDENT_SDXL, _FLUX_MODEL],
        max_threads=1,
    )


def _build_overlap_scheduler(
    *,
    free_mb: float,
    num_idle: int = _NUM_IDLE_SIBLINGS,
) -> tuple[InferenceScheduler, ProcessMap, JobTracker, HordeProcessInfo, list[HordeProcessInfo]]:
    """A worker sampling SDXL on one process with ``num_idle`` model-free idle siblings.

    The busy process is mid-inference (``INFERENCE_STARTING``); the idle siblings can take a preload.
    Device-free reads ``free_mb`` (low, because the SDXL job is resident on the card). ``scale_inference``
    and the safety pause are stubbed so the test observes the scheduler's *decisions* without driving real
    OS processes.
    """
    busy = make_mock_process_info(1, model_name=_RESIDENT_SDXL, state=HordeProcessState.INFERENCE_STARTING)
    idle: list[HordeProcessInfo] = [
        make_mock_process_info(pid, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        for pid in range(2, 2 + num_idle)
    ]
    for proc in (busy, *idle):
        proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
        proc.vram_usage_mb = _DEVICE_TOTAL_VRAM_MB - free_mb
    process_map = ProcessMap({proc.process_id: proc for proc in (busy, *idle)})

    job_tracker = JobTracker()
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=_overlap_bridge_data(),
        max_concurrent=1,
        max_inference=1 + num_idle,
    )
    scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=1 + num_idle)
    scheduler._process_lifecycle.pause_safety_on_gpu = Mock(return_value=True)
    scheduler._process_lifecycle.is_safety_gpu_paused = False
    return scheduler, process_map, job_tracker, busy, idle


def _seed_flux_weight_estimates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin Flux's weight + sampling-peak estimates so the forecast deterministically needs sole residency."""
    monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda job, baseline: _FLUX_WEIGHTS_MB)
    monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: _FLUX_SAMPLING_PEAK_MB)


def _idle_preloading_flux(idle: list[HordeProcessInfo]) -> list[HordeProcessInfo]:
    """The idle siblings that were handed a Flux preload this cycle."""
    return [
        proc
        for proc in idle
        if proc.last_control_flag == HordeControlFlag.PRELOAD_MODEL and proc.loaded_horde_model_name == _FLUX_MODEL
    ]


class TestEarlyRamPreStage:
    """A whole-card head must pre-stage into a spare's RAM while the in-flight job still holds the device."""

    async def test_flux_head_preloads_into_idle_ram_while_sdxl_samples(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RED on current code: the head is reserved exclusive but no idle process is given its RAM preload.

        Today the residency tears the idle siblings down and defers, so Flux's disk->RAM load does not begin
        until the SDXL job drains. The fix pre-stages Flux into a spare process's RAM concurrently with the
        SDXL sampling, so the load is already done (or well underway) when the device frees.
        """
        _seed_flux_weight_estimates(monkeypatch)
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 12000.0)

        scheduler, _process_map, job_tracker, busy, idle = _build_overlap_scheduler(free_mb=2000.0)
        # Plenty of system RAM to hold Flux alongside the in-flight SDXL job.
        scheduler._measured_available_ram_mb = lambda: 64000.0  # type: ignore[method-assign]

        sdxl_job = make_job_pop_response(_RESIDENT_SDXL)
        await track_popped_job_async(job_tracker, sdxl_job)
        await mark_job_in_progress_async(job_tracker, sdxl_job)
        flux_head = make_job_pop_response(_FLUX_MODEL, width=1216, height=1216)
        await track_popped_job_async(job_tracker, flux_head)

        admitted = scheduler.preload_models()

        preloading = _idle_preloading_flux(idle)
        assert admitted is True, "a preload should have been issued (Flux staged into RAM)"
        assert len(preloading) == 1, "exactly one idle sibling should be given Flux's RAM preload"
        assert job_tracker.is_admitted_exclusive(flux_head) is True, "the whole-card head is reserved exclusive"

    async def test_prestage_does_not_disturb_the_in_flight_job(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pre-staging into a spare must not touch the process running the live SDXL job."""
        _seed_flux_weight_estimates(monkeypatch)
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 12000.0)

        scheduler, _process_map, job_tracker, busy, _idle = _build_overlap_scheduler(free_mb=2000.0)
        scheduler._measured_available_ram_mb = lambda: 64000.0  # type: ignore[method-assign]

        sdxl_job = make_job_pop_response(_RESIDENT_SDXL)
        await track_popped_job_async(job_tracker, sdxl_job)
        await mark_job_in_progress_async(job_tracker, sdxl_job)
        flux_head = make_job_pop_response(_FLUX_MODEL, width=1216, height=1216)
        await track_popped_job_async(job_tracker, flux_head)

        scheduler.preload_models()

        assert busy.last_process_state == HordeProcessState.INFERENCE_STARTING
        assert busy.loaded_horde_model_name == _RESIDENT_SDXL
        assert busy.last_control_flag != HordeControlFlag.PRELOAD_MODEL

    async def test_no_prestage_when_ram_budget_cannot_fit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Guard: with the in-flight job already near the RAM ceiling, do not pre-stage (it would page).

        The user's premise is explicitly "assuming the RAM can support it". When the heavy head's RAM cost
        would not fit alongside the live job, the scheduler must fall back to the prior claim-the-card-and-wait
        behavior rather than force a second multi-GB checkpoint into a RAM-pressured host.
        """
        _seed_flux_weight_estimates(monkeypatch)
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 60000.0)

        scheduler, _process_map, job_tracker, _busy, idle = _build_overlap_scheduler(free_mb=2000.0)
        scheduler._measured_available_ram_mb = lambda: 8000.0  # type: ignore[method-assign]

        sdxl_job = make_job_pop_response(_RESIDENT_SDXL)
        await track_popped_job_async(job_tracker, sdxl_job)
        await mark_job_in_progress_async(job_tracker, sdxl_job)
        flux_head = make_job_pop_response(_FLUX_MODEL, width=1216, height=1216)
        await track_popped_job_async(job_tracker, flux_head)

        admitted = scheduler.preload_models()

        assert admitted is False
        assert _idle_preloading_flux(idle) == [], "no idle sibling should be given Flux's RAM preload"


class TestResidencyConvergesAfterDrain:
    """Once the head is pre-staged and the device frees, the residency must collapse to sole VRAM residency."""

    async def test_collapses_to_target_protecting_the_prestaged_holder(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RED on current code: after the live job drains, nothing drives the teardown to sole residency.

        Post-drain state: Flux is resident in RAM on a spare (``PRELOADED_MODEL``), the former busy process is
        idle but still holds its CUDA context, and no job is in progress. The residency must now scale down to
        the model's target (1) so Flux samples with the whole card; the process holding the pre-staged Flux is
        protected from the teardown (it carries the queued model).
        """
        _seed_flux_weight_estimates(monkeypatch)

        # One process holds Flux pre-staged in RAM; three siblings sit idle holding only their contexts.
        flux_holder = make_mock_process_info(2, model_name=_FLUX_MODEL, state=HordeProcessState.PRELOADED_MODEL)
        former_busy = make_mock_process_info(1, model_name=_RESIDENT_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
        spare_a = make_mock_process_info(3, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare_b = make_mock_process_info(4, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        procs = [former_busy, flux_holder, spare_a, spare_b]
        for proc in procs:
            proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
            proc.vram_usage_mb = _DEVICE_TOTAL_VRAM_MB - 15007.0  # device drained: the live job has finished
        process_map = ProcessMap({proc.process_id: proc for proc in procs})

        job_tracker = JobTracker()
        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=_overlap_bridge_data(),
            max_concurrent=1,
            max_inference=4,
        )
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=4)
        scheduler._process_lifecycle.pause_safety_on_gpu = Mock(return_value=True)
        scheduler._process_lifecycle.is_safety_gpu_paused = False

        # Flux is resident-in-RAM and a residency is being held for it (the state the pre-stage leaves).
        scheduler._horde_model_map.update_entry(
            horde_model_name=_FLUX_MODEL,
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=flux_holder.process_id,
        )
        scheduler._sibling_teardown_for_model = _FLUX_MODEL
        scheduler._whole_card_forecast = scheduler._forecast_streaming(
            make_job_pop_response(_FLUX_MODEL, width=1216, height=1216),
            _FLUX_BASELINE,
        )

        flux_head = make_job_pop_response(_FLUX_MODEL, width=1216, height=1216)
        await track_popped_job_async(job_tracker, flux_head)

        scheduler.preload_models()

        scheduler._process_lifecycle.scale_inference_processes.assert_called_with(1)


def _staged_flux_scheduler(
    *,
    num_processes: int,
    free_mb: float,
    safety_paused: bool,
) -> tuple[InferenceScheduler, JobTracker, HordeProcessInfo]:
    """A scheduler with Flux pre-staged in RAM on one process and ``num_processes`` inference slots live.

    Models the window after the heavy head was pre-staged: ``num_processes > 1`` means idle siblings have not
    yet exited (the residency is still collapsing); ``num_processes == 1`` with the card drained is the
    converged, ready-to-sample state. A whole-card residency is recorded as held for Flux.
    """
    flux_holder = make_mock_process_info(2, model_name=_FLUX_MODEL, state=HordeProcessState.PRELOADED_MODEL)
    procs = [flux_holder]
    procs += [
        make_mock_process_info(pid, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        for pid in range(3, 3 + (num_processes - 1))
    ]
    for proc in procs:
        proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
        proc.vram_usage_mb = _DEVICE_TOTAL_VRAM_MB - free_mb
    process_map = ProcessMap({proc.process_id: proc for proc in procs})

    job_tracker = JobTracker()
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=_overlap_bridge_data(),
        max_concurrent=1,
        max_inference=4,
    )
    scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=num_processes)
    scheduler._process_lifecycle.pause_safety_on_gpu = Mock(return_value=True)
    scheduler._process_lifecycle.is_safety_gpu_paused = safety_paused
    scheduler._horde_model_map.update_entry(
        horde_model_name=_FLUX_MODEL,
        load_state=ModelLoadState.LOADED_IN_RAM,
        process_id=flux_holder.process_id,
    )
    scheduler._sibling_teardown_for_model = _FLUX_MODEL
    return scheduler, job_tracker, flux_holder


class TestPrestagedHeadWaitsForSoleResidency:
    """A pre-staged head must not sample until the residency has collapsed to sole VRAM residency.

    Sampling commits the weights to VRAM, so dispatching before the idle siblings (or the just-drained busy
    process) have released their CUDA contexts would force the first step to stream over the bus -- exactly
    the streaming storm the residency exists to prevent. The head keeps its head-of-queue spot and dispatches
    the instant the card is clear.
    """

    async def test_not_dispatched_while_siblings_still_hold_contexts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RED on a naive fix: with idle siblings still up, the pre-staged head must defer, not sample."""
        _seed_flux_weight_estimates(monkeypatch)

        # Three sibling contexts still on the card and free not yet recovered: residency not converged.
        scheduler, job_tracker, flux_holder = _staged_flux_scheduler(
            num_processes=4,
            free_mb=2000.0,
            safety_paused=False,
        )
        flux_head = make_job_pop_response(_FLUX_MODEL, width=1216, height=1216)
        await track_popped_job_async(job_tracker, flux_head)
        job_tracker.mark_admitted_exclusive(flux_head)

        started = await scheduler.start_inference()

        assert started is False
        assert flux_holder.last_control_flag != HordeControlFlag.START_INFERENCE
        assert flux_head not in job_tracker.jobs_in_progress

    async def test_dispatched_once_residency_has_converged(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """At sole residency with the card drained and safety off-GPU, the pre-staged head finally samples."""
        _seed_flux_weight_estimates(monkeypatch)

        # One live process (the Flux holder), safety off-GPU, card drained: the converged, ready state.
        scheduler, job_tracker, flux_holder = _staged_flux_scheduler(
            num_processes=1,
            free_mb=15007.0,
            safety_paused=True,
        )
        flux_head = make_job_pop_response(_FLUX_MODEL, width=1216, height=1216)
        await track_popped_job_async(job_tracker, flux_head)
        job_tracker.mark_admitted_exclusive(flux_head)

        started = await scheduler.start_inference()

        assert started is True
        assert flux_holder.last_control_flag == HordeControlFlag.START_INFERENCE
        assert flux_head in job_tracker.jobs_in_progress
