"""Reproduction of the Flux weight-streaming storm and its whole-card-residency fix.

The failure mode (on a 16GB device): a heavy Flux job loaded while two SD/SDXL models stayed resident
across sibling inference processes. Device-free VRAM collapsed to near zero, ComfyUI streamed Flux's
weights from host RAM, sampling slowed by several times, and the slow job was killed as a suspected hang
(a recovery) with the job faulted. The constants below are representative of that 16GB-device scenario (a
~11.3GB resident Flux fp8 checkpoint that leaves only tens of MB free alongside the other models).

Two layers are exercised: the pure streaming forecast (does it classify avoidable vs unavoidable
streaming correctly?) and the scheduler (does it reserve the whole card for such a model before it loads,
rather than letting it stream and get hang-graded?).
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management import resource_budget
from horde_worker_regen.process_management.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.resource_budget import (
    StreamForecast,
    effective_inference_reserve_mb,
    forecast_weight_streaming,
    predict_job_weight_mb,
)

from .conftest import make_job_pop_response, make_mock_bridge_data, make_mock_process_info, track_popped_job_async
from .test_inference_scheduling import _make_inference_scheduler

# Representative of the observed 16GB-device scenario.
_DEVICE_TOTAL_VRAM_MB = 16375
_FLUX_WEIGHTS_MB = 11500.0  # the registry seed for flux baselines (~fp8 resident footprint)
_FLUX_FP16_WEIGHTS_MB = 22700.0  # fp16 Flux: streams even with the whole card to itself
_FREE_WITH_SIBLINGS_MB = 57.0  # device-free left once Flux loads alongside the resident SD/SDXL models
_VRAM_RESERVE_MB = 2048
_RAM_RESERVE_MB = 4096

_FLUX_MODEL = "Flux.1-Schnell fp8 (Compact)"
_RESIDENT_SD15 = "Deliberate"
_RESIDENT_SDXL = "CyberRealistic Pony"


class TestStreamForecastClassification:
    """The pure forecast must tell avoidable streaming (curable by exclusive residency) from unavoidable."""

    def test_effective_reserve_honors_comfy_floor_and_config(self) -> None:
        """The reserve is at least ComfyUI's inference floor, with the configured value as an extra floor."""
        # On a 16GB Windows card ComfyUI's minimum_inference_memory is ~1519MB; a 2048 config floor wins.
        assert effective_inference_reserve_mb(_DEVICE_TOTAL_VRAM_MB, 2048.0) == 2048.0
        # A small config floor is raised to the ComfyUI floor so the forecast matches the real split point.
        assert effective_inference_reserve_mb(_DEVICE_TOTAL_VRAM_MB, 256.0) >= 1519.0
        # No total VRAM (cold start) falls back to the configured floor.
        assert effective_inference_reserve_mb(None, 2048.0) == 2048.0

    def test_flux_fp8_streams_coresident_but_fits_alone(self) -> None:
        """Flux fp8 at ~57MB free streams co-resident, but with the card to itself it fits: exclusive."""
        forecast = StreamForecast(
            weights_mb=_FLUX_WEIGHTS_MB,
            reserve_mb=_VRAM_RESERVE_MB,
            free_now_mb=_FREE_WITH_SIBLINGS_MB,
            free_if_alone_mb=_DEVICE_TOTAL_VRAM_MB - 1288,  # measured per-process overhead on the live box
        )
        assert forecast.fits_coresident is False
        assert forecast.fits_alone is True
        assert forecast.needs_exclusive_residency is True
        assert forecast.streams_unavoidably is False

    def test_flux_fp16_streams_even_alone(self) -> None:
        """fp16 Flux is too big for a 16GB card even alone: unavoidable streaming, not exclusive residency."""
        forecast = StreamForecast(
            weights_mb=_FLUX_FP16_WEIGHTS_MB,
            reserve_mb=_VRAM_RESERVE_MB,
            free_now_mb=_FREE_WITH_SIBLINGS_MB,
            free_if_alone_mb=_DEVICE_TOTAL_VRAM_MB - 1288,
        )
        assert forecast.needs_exclusive_residency is False
        assert forecast.streams_unavoidably is True

    def test_unknown_or_cold_start_never_blocks(self) -> None:
        """An unknown weight estimate or absent telemetry forecasts co-resident, so it never blocks a load."""
        unknown_weights = StreamForecast(weights_mb=None, reserve_mb=2048, free_now_mb=57, free_if_alone_mb=15000)
        assert unknown_weights.fits_coresident is True
        assert unknown_weights.needs_exclusive_residency is False
        cold = StreamForecast(weights_mb=11500, reserve_mb=2048, free_now_mb=None, free_if_alone_mb=15000)
        assert cold.fits_coresident is True

    def test_predict_job_weight_uses_baseline_seed(self) -> None:
        """The weight estimate comes from the baseline's resident-weight seed, None for an unknown baseline."""
        job = make_job_pop_response(_FLUX_MODEL)
        assert predict_job_weight_mb(job, "flux_1") == _FLUX_WEIGHTS_MB
        assert predict_job_weight_mb(job, "definitely-not-a-baseline") is None
        assert predict_job_weight_mb(job, None) is None

    def test_forecast_end_to_end_with_representative_numbers(self) -> None:
        """The forecast function reproduces the expected classification from a representative device state."""
        job = make_job_pop_response(_FLUX_MODEL)
        forecast = forecast_weight_streaming(
            job,
            "flux_1",
            free_now_mb=_FREE_WITH_SIBLINGS_MB,
            total_vram_mb=_DEVICE_TOTAL_VRAM_MB,
            per_process_overhead_mb=1288.0,
            configured_reserve_floor_mb=float(_VRAM_RESERVE_MB),
        )
        assert forecast.needs_exclusive_residency is True


def _storm_bridge_data() -> Mock:
    """Bridge data mirroring the observed config: budget on, whole-card residency on, multi-model, 16GB box."""
    return make_mock_bridge_data(
        enable_vram_budget=True,
        whole_card_exclusive_residency=True,
        vram_reserve_mb=_VRAM_RESERVE_MB,
        ram_reserve_mb=_RAM_RESERVE_MB,
        vram_per_process_overhead_mb=0,
        overbudget_exclusive_mode=True,
        safety_on_gpu=True,
        image_models_to_load=[_RESIDENT_SD15, _RESIDENT_SDXL, _FLUX_MODEL],
        max_threads=1,
    )


def _build_storm_scheduler() -> tuple[InferenceScheduler, ProcessMap, JobTracker, object, object]:
    """Two idle sibling processes resident with SD/SDXL, device-free at the ~57MB a Flux load leaves."""
    proc_sd15 = make_mock_process_info(1, model_name=_RESIDENT_SD15, state=HordeProcessState.WAITING_FOR_JOB)
    proc_sdxl = make_mock_process_info(2, model_name=_RESIDENT_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
    for proc in (proc_sd15, proc_sdxl):
        proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
        proc.vram_usage_mb = _DEVICE_TOTAL_VRAM_MB - _FREE_WITH_SIBLINGS_MB
    process_map = ProcessMap({1: proc_sd15, 2: proc_sdxl})

    job_tracker = JobTracker()
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=_storm_bridge_data(),
        max_concurrent=1,
        max_inference=2,
    )
    return scheduler, process_map, job_tracker, proc_sd15, proc_sdxl


class TestWholeCardExclusiveResidency:
    """A Flux head whose weights would stream co-resident must get sole residency before it loads."""

    async def test_flux_head_reserves_whole_card_instead_of_streaming(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The fix: a forecast-streaming head is marked exclusive and evicts siblings, not admitted co-resident.

        Before the fix the scheduler let Flux load alongside the resident SD/SDXL models (device-free ~57MB),
        ComfyUI streamed its weights, and the slow sampling tripped the hang-grader. After the fix the
        scheduler forecasts the streaming, marks the job exclusive (suppressing sibling staging), evicts the
        idle resident siblings, and defers the preload until the device clears.
        """
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda job, baseline: _FLUX_WEIGHTS_MB)

        scheduler, _process_map, job_tracker, proc_sd15, proc_sdxl = _build_storm_scheduler()
        head_job = make_job_pop_response(_FLUX_MODEL)
        await track_popped_job_async(job_tracker, head_job)

        # The preload is deferred this cycle: the device is not yet clear for a full-resident Flux load.
        admitted = scheduler.preload_models()

        assert admitted is False, "Flux must not be admitted co-resident at ~57MB free (it would stream)"
        assert job_tracker.is_admitted_exclusive(head_job) is True, "the head must be reserved exclusive early"
        assert HordeControlFlag.UNLOAD_MODELS_FROM_VRAM in {
            proc_sd15.last_control_flag,
            proc_sdxl.last_control_flag,
        }, "the resident siblings must be evicted to clear the card"
        # The siblings were NOT given a competing preload while the whole-card job is reserved.
        assert HordeControlFlag.PRELOAD_MODEL not in {
            proc_sd15.last_control_flag,
            proc_sdxl.last_control_flag,
        }
