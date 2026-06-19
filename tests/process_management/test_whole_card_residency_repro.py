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

import time
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management import resource_budget
from horde_worker_regen.process_management.horde_process import HordeProcessType
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
_PER_PROCESS_OVERHEAD_MB = 1288  # measured idle torch/CUDA-context VRAM per inference process on the live box
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

    def test_flux_fp8_streams_coresident_evicting_models_suffices(self) -> None:
        """Flux fp8 at ~57MB free streams co-resident; evicting the two sibling *models* leaves room."""
        forecast = StreamForecast(
            weights_mb=_FLUX_WEIGHTS_MB,
            reserve_mb=_VRAM_RESERVE_MB,
            free_now_mb=_FREE_WITH_SIBLINGS_MB,
            free_if_alone_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,  # 15087
            free_after_model_evict_mb=_DEVICE_TOTAL_VRAM_MB - 2 * _PER_PROCESS_OVERHEAD_MB,  # 13799: 2 contexts fit
        )
        assert forecast.fits_coresident is False
        assert forecast.fits_after_model_evict is True
        assert forecast.fits_alone is True
        assert forecast.needs_exclusive_residency is True
        assert forecast.requires_sibling_teardown is False  # dropping sibling models is enough
        assert forecast.streams_unavoidably is False

    def test_flux_fp8_requires_sibling_teardown_when_contexts_overcommit(self) -> None:
        """The live blind spot: high instantaneous free yet contexts structurally over-commit the card.

        At admit the idle siblings' contexts have not allocated, so ``free_now`` reads ~15GB, but four
        process contexts mean the model needs sibling *processes* stopped, not just their models evicted.
        The structural floor must catch this even though ``free_now`` looks ample.
        """
        forecast = StreamForecast(
            weights_mb=_FLUX_WEIGHTS_MB,
            reserve_mb=_VRAM_RESERVE_MB,
            free_now_mb=15000.0,  # deceptively high: the idle siblings' contexts have not allocated yet
            free_if_alone_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,  # 15087
            free_after_model_evict_mb=_DEVICE_TOTAL_VRAM_MB
            - 4 * _PER_PROCESS_OVERHEAD_MB,  # 11223: contexts over-commit
        )
        assert forecast.fits_coresident is False  # structural floor overrides the high instantaneous reading
        assert forecast.fits_after_model_evict is False  # evicting models does not free the contexts
        assert forecast.fits_alone is True
        assert forecast.needs_exclusive_residency is True
        assert forecast.requires_sibling_teardown is True
        assert forecast.streams_unavoidably is False

    def test_flux_fp16_streams_even_alone(self) -> None:
        """fp16 Flux is too big for a 16GB card even alone: unavoidable streaming, not exclusive residency."""
        forecast = StreamForecast(
            weights_mb=_FLUX_FP16_WEIGHTS_MB,
            reserve_mb=_VRAM_RESERVE_MB,
            free_now_mb=_FREE_WITH_SIBLINGS_MB,
            free_if_alone_mb=_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB,
            free_after_model_evict_mb=_DEVICE_TOTAL_VRAM_MB - 4 * _PER_PROCESS_OVERHEAD_MB,
        )
        assert forecast.needs_exclusive_residency is False
        assert forecast.requires_sibling_teardown is False
        assert forecast.streams_unavoidably is True

    def test_unknown_or_cold_start_never_blocks(self) -> None:
        """An unknown weight estimate or absent telemetry forecasts co-resident, so it never blocks a load."""
        unknown_weights = StreamForecast(
            weights_mb=None,
            reserve_mb=2048,
            free_now_mb=57,
            free_if_alone_mb=15000,
            free_after_model_evict_mb=11223,
        )
        assert unknown_weights.fits_coresident is True
        assert unknown_weights.needs_exclusive_residency is False
        assert unknown_weights.requires_sibling_teardown is False
        cold = StreamForecast(
            weights_mb=11500,
            reserve_mb=2048,
            free_now_mb=None,
            free_if_alone_mb=15000,
            free_after_model_evict_mb=11223,
        )
        assert cold.fits_coresident is True

    def test_predict_job_weight_uses_baseline_seed(self) -> None:
        """The weight estimate comes from the baseline's resident-weight seed, None for an unknown baseline."""
        job = make_job_pop_response(_FLUX_MODEL)
        assert predict_job_weight_mb(job, "flux_1") == _FLUX_WEIGHTS_MB
        assert predict_job_weight_mb(job, "definitely-not-a-baseline") is None
        assert predict_job_weight_mb(job, None) is None

    def test_forecast_end_to_end_requires_teardown_then_converges(self) -> None:
        """End to end: four processes' contexts force a teardown; after it, the reduced count fits (converges).

        This is the property that stops the teardown looping forever: keying the structural floor off the
        *live* process count means once siblings are stopped the same model re-forecasts as co-resident.
        """
        job = make_job_pop_response(_FLUX_MODEL)
        before = forecast_weight_streaming(
            job,
            "flux_1",
            free_now_mb=15000.0,
            total_vram_mb=_DEVICE_TOTAL_VRAM_MB,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
            num_inference_processes=4,
            configured_reserve_floor_mb=float(_VRAM_RESERVE_MB),
        )
        assert before.requires_sibling_teardown is True
        # The reserve is the activation working set (peak ~14000 - weights 11500 = 2500 MB), not the flat
        # 2048 floor, so Flux needs sole residency: (16375 - 11500 - 2500) // 1288 == 1, down to one process.
        assert before.max_resident_processes() == 1

        after = forecast_weight_streaming(
            job,
            "flux_1",
            free_now_mb=float(_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB),
            total_vram_mb=_DEVICE_TOTAL_VRAM_MB,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
            num_inference_processes=1,
            configured_reserve_floor_mb=float(_VRAM_RESERVE_MB),
        )
        assert after.requires_sibling_teardown is False
        assert after.fits_coresident is True

    def test_reserve_covers_activation_working_set_not_flat_floor(self) -> None:
        """The forecast reserve is the model's activation working set, not the flat configured floor.

        Regression for the persistent 2026-06-19 overflow: Flux's per-step activations (~2.5 GB at 512^2,
        more at higher resolution) dwarf the ~2 GB reserve floor. A weights-plus-flat-reserve forecast
        judged Flux co-resident, and the sampling step then drove free VRAM to zero and spilled activations
        to host RAM. The reserve must reflect the activation-inclusive peak.
        """
        job = make_job_pop_response(_FLUX_MODEL)
        forecast = forecast_weight_streaming(
            job,
            "flux_1",
            free_now_mb=15000.0,
            total_vram_mb=_DEVICE_TOTAL_VRAM_MB,
            per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
            num_inference_processes=1,
            configured_reserve_floor_mb=float(_VRAM_RESERVE_MB),
        )
        assert forecast.reserve_mb > float(_VRAM_RESERVE_MB)

    def test_safety_on_gpu_context_lowers_current_achievable_free(self) -> None:
        """A safety-on-GPU context is charged against the *current* achievable free (free_after_model_evict).

        Stopping idle inference siblings cannot reclaim it, so it must lower that figure. It does NOT lower
        free_if_alone (the ceiling), since claiming the whole card for a heavy model moves safety off-GPU too.
        """
        job = make_job_pop_response(_FLUX_MODEL)
        kwargs = {
            "free_now_mb": 15000.0,
            "total_vram_mb": _DEVICE_TOTAL_VRAM_MB,
            "per_process_overhead_mb": _PER_PROCESS_OVERHEAD_MB,
            "num_inference_processes": 2,
            "configured_reserve_floor_mb": float(_VRAM_RESERVE_MB),
        }
        without_safety = forecast_weight_streaming(job, "flux_1", num_extra_resident_contexts=0, **kwargs)
        with_safety = forecast_weight_streaming(job, "flux_1", num_extra_resident_contexts=1, **kwargs)
        # The ceiling (sole residency, safety off) is unchanged...
        assert with_safety.free_if_alone_mb == without_safety.free_if_alone_mb
        # ...but the current floor drops by the safety context.
        assert (
            with_safety.free_after_model_evict_mb
            == without_safety.free_after_model_evict_mb - _PER_PROCESS_OVERHEAD_MB
        )


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


def _build_context_overcommit_scheduler(
    num_processes: int = 4,
    *,
    max_inference: int | None = None,
) -> tuple[InferenceScheduler, ProcessMap, JobTracker]:
    """Idle, model-free inference processes whose *contexts* over-commit the card (free still reads high).

    This is the live blind spot: at admit time the siblings are idle (no model) and free VRAM reads ample
    because their CUDA contexts have not yet allocated, but loading a heavy model would collapse free once
    those contexts materialise. The fix must read the structural floor, not the deceptively-high instant.
    """
    procs: dict[int, object] = {}
    for pid in range(1, num_processes + 1):
        proc = make_mock_process_info(pid, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
        proc.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB  # device reads ~15GB free: one context's worth used
        procs[pid] = proc
    process_map = ProcessMap(procs)

    job_tracker = JobTracker()
    bridge_data = make_mock_bridge_data(
        enable_vram_budget=True,
        whole_card_exclusive_residency=True,
        vram_reserve_mb=_VRAM_RESERVE_MB,
        ram_reserve_mb=_RAM_RESERVE_MB,
        vram_per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,  # config override so the test is deterministic
        overbudget_exclusive_mode=True,
        safety_on_gpu=True,
        image_models_to_load=[_FLUX_MODEL],
        max_threads=1,
    )
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=bridge_data,
        max_concurrent=1,
        max_inference=max_inference if max_inference is not None else num_processes,
    )
    return scheduler, process_map, job_tracker


class TestWholeCardSiblingTeardown:
    """When sibling process *contexts* (not their models) over-commit the card, stop idle siblings."""

    async def test_flux_head_stops_idle_siblings_to_reclaim_contexts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The new fix: a whole-card model whose siblings' contexts over-commit triggers a process teardown.

        Evicting sibling models cannot free a ~1GB-per-process CUDA context (only the process exiting can),
        so the scheduler reduces the live inference-process count to the largest that still fits the model's
        weights plus reserve, and marks the job exclusive. ``free_now`` reads ~15GB here (idle contexts not yet
        allocated), so a forecast keyed only on the instant would wrongly admit it co-resident.
        """
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda job, baseline: _FLUX_WEIGHTS_MB)
        # Activation-inclusive peak == weights + 2500 MB working set, so the reserve is 2500 (above the floor)
        # and Flux needs sole residency: (16375 - 11500 - 2500) // 1288 == 1.
        monkeypatch.setattr(resource_budget, "predict_job_vram_mb", lambda job, baseline: _FLUX_WEIGHTS_MB + 2500)

        scheduler, _process_map, job_tracker = _build_context_overcommit_scheduler(num_processes=4)
        # Stub the lifecycle scaler so the test asserts the call and returns an int the scheduler can log.
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=1)

        head_job = make_job_pop_response(_FLUX_MODEL)
        await track_popped_job_async(job_tracker, head_job)

        admitted = scheduler.preload_models()

        assert admitted is False, "the whole-card head must defer until the device is cleared"
        assert job_tracker.is_admitted_exclusive(head_job) is True
        # Flux needs the whole card: teardown all the way down to one inference process.
        scheduler._process_lifecycle.scale_inference_processes.assert_called_once_with(1)
        assert scheduler._sibling_teardown_for_model == _FLUX_MODEL

    async def test_siblings_restored_after_whole_card_job_drains(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Once the torn-down model is neither pending nor in progress, the process count is grown back."""
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda job, baseline: _FLUX_WEIGHTS_MB)

        # Torn-down state: only two processes live but the configured ceiling is four.
        scheduler, _process_map, job_tracker = _build_context_overcommit_scheduler(num_processes=2, max_inference=4)
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=4)
        # Simulate the state left after a teardown whose exclusive job has since completed and drained.
        scheduler._sibling_teardown_for_model = _FLUX_MODEL

        scheduler._restore_siblings_after_whole_card()

        scheduler._process_lifecycle.scale_inference_processes.assert_called_once_with(4)
        assert scheduler._sibling_teardown_for_model is None

    async def test_restore_held_while_torn_down_model_still_queued(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Back-to-back whole-card jobs must not thrash: hold the restore while the model is still queued."""
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda job, baseline: _FLUX_WEIGHTS_MB)

        scheduler, _process_map, job_tracker = _build_context_overcommit_scheduler(num_processes=4)
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=4)
        head_job = make_job_pop_response(_FLUX_MODEL)
        await track_popped_job_async(job_tracker, head_job)
        scheduler._sibling_teardown_for_model = _FLUX_MODEL

        scheduler._restore_siblings_after_whole_card()

        scheduler._process_lifecycle.scale_inference_processes.assert_not_called()
        assert scheduler._sibling_teardown_for_model == _FLUX_MODEL

    async def test_restore_held_through_cooldown_then_restores(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After the heavy job drains, the residency is held for the configured cooldown, then restored.

        The cooldown batches churn: a burst of whole-card jobs reuses one residency instead of each
        triggering a teardown/restore + safety cycle.
        """
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda job, baseline: _FLUX_WEIGHTS_MB)

        scheduler, _process_map, job_tracker = _build_context_overcommit_scheduler(num_processes=2, max_inference=4)
        scheduler._process_lifecycle.scale_inference_processes = Mock(return_value=4)
        scheduler._runtime_config.bridge_data.whole_card_residency_cooldown_seconds = 300
        scheduler._sibling_teardown_for_model = _FLUX_MODEL

        # Drained (no job queued/in progress) but still inside the cooldown -> hold the residency.
        scheduler._whole_card_cooldown_until = time.time() + 300
        scheduler._restore_siblings_after_whole_card()
        scheduler._process_lifecycle.scale_inference_processes.assert_not_called()
        assert scheduler._sibling_teardown_for_model == _FLUX_MODEL

        # Cooldown elapsed -> restore concurrency.
        scheduler._whole_card_cooldown_until = time.time() - 1
        scheduler._restore_siblings_after_whole_card()
        scheduler._process_lifecycle.scale_inference_processes.assert_called_once_with(4)
        assert scheduler._sibling_teardown_for_model is None

    def test_forecast_stops_charging_safety_context_once_paused(self) -> None:
        """Once safety is paused off-GPU, the forecast must stop charging its context.

        Regression for an observed live wedge: with safety paused but the forecast still subtracting its
        (now-freed) context, the structural floor (free_after_model_evict) stayed below Flux's demand, so
        the whole-card branch deferred Flux every tick forever and it never loaded. The forecast must read
        the live pause state, not just the safety_on_gpu config.
        """
        scheduler, process_map, _job_tracker = _build_context_overcommit_scheduler(num_processes=1)
        process_map[0] = make_mock_process_info(
            0,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        job = make_job_pop_response(_FLUX_MODEL)

        scheduler._process_lifecycle.is_safety_gpu_paused = False
        on_gpu = scheduler._forecast_streaming(job, "flux_1")
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        paused = scheduler._forecast_streaming(job, "flux_1")

        # Pausing safety frees one context, so the structural floor rises by one per-process overhead.
        assert on_gpu.free_after_model_evict_mb is not None
        assert paused.free_after_model_evict_mb == on_gpu.free_after_model_evict_mb + _PER_PROCESS_OVERHEAD_MB

    def test_grace_suppresses_structural_wedge_during_establishment(self) -> None:
        """While a whole-card residency establishes, the intentionally-held queue is not a structural wedge.

        Regression for the save-our-ship soft reset that fired mid-setup: the establishment now cycles the
        safety process off-GPU (~20s) on top of the teardown and the ~11GB load, so the heavy head is
        deferred long enough to look like a sustained queue deadlock. The grace must suppress that, bounded
        so a residency that never loads still trips the supervisor.
        """
        scheduler, _process_map, _job_tracker = _build_context_overcommit_scheduler(num_processes=4)
        assert scheduler.whole_card_residency_grace_active() is False

        scheduler._sibling_teardown_for_model = _FLUX_MODEL
        scheduler._whole_card_established_at = time.time()
        assert scheduler.whole_card_residency_grace_active() is True

        # Past the bounded grace window the suppression lifts (a genuinely-stuck residency trips the SOS).
        from horde_worker_regen.process_management import inference_scheduler as _sched_mod

        scheduler._whole_card_established_at = time.time() - (_sched_mod._WHOLE_CARD_ESTABLISH_GRACE_SECONDS + 1.0)
        assert scheduler.whole_card_residency_grace_active() is False

        # The restore window is also covered: respawning siblings + cycling safety back on-GPU is churn
        # that must not read as a wedge either.
        scheduler._sibling_teardown_for_model = None
        scheduler._whole_card_restore_at = time.time()
        assert scheduler.whole_card_residency_grace_active() is True
        scheduler._whole_card_restore_at = time.time() - (_sched_mod._WHOLE_CARD_RESTORE_GRACE_SECONDS + 1.0)
        assert scheduler.whole_card_residency_grace_active() is False


class TestWholeCardResidencyState:
    """The status accessor must surface the posture heads-up and a held residency's live detail for the TUI."""

    def test_possible_when_feature_on_and_a_teardown_can_occur(self) -> None:
        """`possible` is true when the feature and budget are on and more than one inference process can run."""
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(
                enable_vram_budget=True,
                whole_card_exclusive_residency=True,
                vram_reserve_mb=_VRAM_RESERVE_MB,
                ram_reserve_mb=_RAM_RESERVE_MB,
            ),
            max_inference=2,
        )
        scheduler._process_lifecycle.is_safety_gpu_paused = False

        state = scheduler.whole_card_residency_state()

        assert state.possible is True
        assert state.enabled is True
        assert state.active is False
        assert state.processes_max == 2

    def test_not_possible_when_feature_disabled(self) -> None:
        """With the feature off there is no heads-up, even on a multi-process worker."""
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(
                enable_vram_budget=True,
                whole_card_exclusive_residency=False,
                vram_reserve_mb=_VRAM_RESERVE_MB,
                ram_reserve_mb=_RAM_RESERVE_MB,
            ),
            max_inference=4,
        )
        scheduler._process_lifecycle.is_safety_gpu_paused = False

        assert scheduler.whole_card_residency_state().possible is False

    def test_not_possible_single_process_without_safety_teardown(self) -> None:
        """One inference process and no safety-on-GPU means nothing to tear down, so no heads-up."""
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(
                enable_vram_budget=True,
                whole_card_exclusive_residency=True,
                whole_card_residency_safety_off_gpu=False,
                safety_on_gpu=False,
                vram_reserve_mb=_VRAM_RESERVE_MB,
                ram_reserve_mb=_RAM_RESERVE_MB,
            ),
            max_inference=1,
        )
        scheduler._process_lifecycle.is_safety_gpu_paused = False

        assert scheduler.whole_card_residency_state().possible is False

    def test_possible_single_process_when_safety_can_move_off_gpu(self) -> None:
        """Even with one inference process, moving safety off-GPU is a teardown worth the heads-up."""
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(
                enable_vram_budget=True,
                whole_card_exclusive_residency=True,
                whole_card_residency_safety_off_gpu=True,
                safety_on_gpu=True,
                vram_reserve_mb=_VRAM_RESERVE_MB,
                ram_reserve_mb=_RAM_RESERVE_MB,
            ),
            max_inference=1,
        )
        scheduler._process_lifecycle.is_safety_gpu_paused = False

        assert scheduler.whole_card_residency_state().possible is True

    def test_active_state_reports_model_phase_and_forecast_numbers(self) -> None:
        """A held residency surfaces its model, establishing phase, target/ceiling, and forecast numbers."""
        scheduler = _make_inference_scheduler(bridge_data=_storm_bridge_data(), max_inference=2)
        scheduler._process_lifecycle.is_safety_gpu_paused = True
        forecast = StreamForecast(
            weights_mb=_FLUX_WEIGHTS_MB,
            reserve_mb=3700.0,
            free_now_mb=_FREE_WITH_SIBLINGS_MB,
            free_if_alone_mb=15087.0,
            free_after_model_evict_mb=12000.0,
            total_vram_mb=float(_DEVICE_TOTAL_VRAM_MB),
            per_process_overhead_mb=float(_PER_PROCESS_OVERHEAD_MB),
        )
        # Simulate the state _establish_whole_card_residency leaves: model reserved, forecast cached, grace running.
        scheduler._sibling_teardown_for_model = _FLUX_MODEL
        scheduler._whole_card_forecast = forecast
        scheduler._whole_card_established_at = time.time()
        scheduler._whole_card_cooldown_until = time.time() + 45.0

        state = scheduler.whole_card_residency_state()

        assert state.active is True
        assert state.model == _FLUX_MODEL
        assert state.phase == "establishing"
        assert state.safety_paused is True
        assert state.processes_max == 2
        # (16375 - 11500 - 3700) // 1288 -> 0 -> floored to a single surviving (loading) process.
        assert state.processes_target == 1
        assert state.max_resident_processes == 1
        assert state.weights_mb == _FLUX_WEIGHTS_MB
        assert state.free_if_alone_mb == 15087.0
        assert state.cooldown_remaining_seconds is not None
        assert state.cooldown_remaining_seconds > 0

    def test_phase_holding_after_establish_grace(self) -> None:
        """Once the establish grace elapses, an active residency reads as holding (serving), not establishing."""
        from horde_worker_regen.process_management import inference_scheduler as _sched_mod

        scheduler = _make_inference_scheduler(bridge_data=_storm_bridge_data(), max_inference=2)
        scheduler._process_lifecycle.is_safety_gpu_paused = False
        scheduler._sibling_teardown_for_model = _FLUX_MODEL
        scheduler._whole_card_forecast = None
        scheduler._whole_card_established_at = time.time() - (_sched_mod._WHOLE_CARD_ESTABLISH_GRACE_SECONDS + 1.0)

        state = scheduler.whole_card_residency_state()

        assert state.active is True
        assert state.phase == "holding"
