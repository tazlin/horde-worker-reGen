"""Reproduction of the budget-starvation queue wedge.

This encodes the failure mode where a worker with a small-VRAM device serves a wide
variety of models. When a head-of-queue job's *fresh preload* is gated by the VRAM
budget and the job's predicted peak plus the reserve exceeds the device's *achievable*
free-VRAM floor (the CUDA contexts of every process, plus safety-on-GPU, hold the
remainder and cannot be reclaimed), the budget defers the preload forever. The head is
never dispatched even though every inference process is idle, which the queue-deadlock
detector then reports as a structural wedge and save-our-ship faults the job.

The experiment isolates the budget verdict as the single active variable: the same
state with a fittable prediction dispatches normally, proving the confounders (process
health, affinity, queue contents) are not the cause.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.integration.test_deadlock_detection import _make_message_dispatcher
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

# Faithful to the observed device/job: a ~16 GB card whose free VRAM settles to a floor
# below the heavy SDXL job's predicted peak + reserve, so the verdict can never flip.
_DEVICE_TOTAL_VRAM_MB = 16375
_ACHIEVABLE_FREE_VRAM_MB = 14655.0  # floor after every idle resident model is evicted
_HEAVY_SDXL_PREDICTED_VRAM_MB = 13574.0
_VRAM_RESERVE_MB = 2048
_RAM_RESERVE_MB = 4096

_HEAD_MODEL = "WAI-NSFW-illustrious-SDXL"
_RESIDENT_SD15 = "Deliberate"
_RESIDENT_SDXL = "AlbedoBase XL 3.1"


def _wedge_bridge_data() -> Mock:
    """Bridge data mirroring the live config: budget on, frequent VRAM unload, multi-model."""
    return make_mock_bridge_data(
        enable_vram_budget=True,
        vram_reserve_mb=_VRAM_RESERVE_MB,
        ram_reserve_mb=_RAM_RESERVE_MB,
        unload_models_from_vram_often=True,
        moderate_performance_mode=True,
        safety_on_gpu=True,
        image_models_to_load=[_RESIDENT_SD15, _RESIDENT_SDXL, _HEAD_MODEL],
        max_threads=1,
    )


def _build_wedged_scheduler(
    free_vram_mb: float,
) -> tuple[InferenceScheduler, ProcessMap, JobTracker, HordeProcessInfo, HordeProcessInfo]:
    """Two idle inference processes, each resident with a *different* model than the head needs.

    The head-of-queue job needs a third model that is resident on neither process; the device
    reports ``free_vram_mb`` free. Mirrors the observed idle-pool-with-evicted-head state.
    """
    proc_sd15 = make_mock_process_info(1, model_name=_RESIDENT_SD15, state=HordeProcessState.WAITING_FOR_JOB)
    proc_sdxl = make_mock_process_info(2, model_name=_RESIDENT_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
    # The device-wide free VRAM the budget reads is min(total - usage) across inference processes.
    for proc in (proc_sd15, proc_sdxl):
        proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
        proc.vram_usage_mb = _DEVICE_TOTAL_VRAM_MB - free_vram_mb
    process_map = ProcessMap({1: proc_sd15, 2: proc_sdxl})

    job_tracker = JobTracker()
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=_wedge_bridge_data(),
        max_concurrent=1,
        max_inference=2,
    )
    return scheduler, process_map, job_tracker, proc_sd15, proc_sdxl


async def _enqueue_head_jobs(job_tracker: JobTracker, count: int = 2) -> None:
    """Queue ``count`` jobs that all need the head model (none in progress)."""
    for _ in range(count):
        await track_popped_job_async(job_tracker, make_job_pop_response(_HEAD_MODEL))


class TestBudgetStarvationWedge:
    """The active variable is the VRAM budget verdict for the head job; everything else is held fixed."""

    async def test_unfittable_head_is_admitted_best_effort_not_deferred_forever(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A head whose predicted peak + reserve exceeds the free-VRAM floor must not defer forever.

        If nothing ever admits the preload, the queue wedges and the head is faulted. Instead the budget must
        first attempt reclamation and, once that is exhausted while no live job holds the device, admit the
        head best-effort so the queue makes progress. Loading one model onto an otherwise-idle GPU cannot
        reintroduce the multi-process over-commit the budget exists to prevent.
        """
        monkeypatch.setattr(
            resource_budget,
            "predict_job_sampling_vram_mb",
            lambda job, baseline: _HEAVY_SDXL_PREDICTED_VRAM_MB,
        )
        # RAM is irrelevant to this VRAM-bound case, but the best-effort VRAM admit skips the RAM gate
        # entirely; pin a fittable RAM picture so the test is unambiguous either way.
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 1000.0)

        scheduler, _process_map, job_tracker, proc_sd15, proc_sdxl = _build_wedged_scheduler(
            free_vram_mb=_ACHIEVABLE_FREE_VRAM_MB,
        )
        monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: 64000.0)
        await _enqueue_head_jobs(job_tracker)

        # Sanity: the threshold is genuinely unreachable on this device.
        assert _HEAVY_SDXL_PREDICTED_VRAM_MB + _VRAM_RESERVE_MB > _ACHIEVABLE_FREE_VRAM_MB

        preloaded_any = False
        reclaimed_first = False
        for _ in range(20):
            if scheduler.preload_models():
                preloaded_any = True
                break
            # Until the head is admitted, the budget should be attempting reclamation, not idling.
            if HordeControlFlag.UNLOAD_MODELS_FROM_VRAM in {
                proc_sd15.last_control_flag,
                proc_sdxl.last_control_flag,
            }:
                reclaimed_first = True

        assert reclaimed_first is True, "the budget should attempt to reclaim idle VRAM before best-effort admit"
        assert preloaded_any is True, "head must eventually be admitted best-effort rather than starved forever"
        assert HordeControlFlag.PRELOAD_MODEL in {
            proc_sd15.last_control_flag,
            proc_sdxl.last_control_flag,
        }

    async def test_fittable_head_dispatches_isolating_the_budget_as_cause(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CONTROL: identical state, only the prediction lowered below the floor -> the preload proceeds.

        Holding the process pool, affinity, queue, and reserves fixed and flipping only the budget
        verdict flips the outcome, isolating the budget gate (not the confounders) as the cause.
        """
        fittable_vram = _ACHIEVABLE_FREE_VRAM_MB - _VRAM_RESERVE_MB - 100.0
        monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: fittable_vram)
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 1000.0)

        scheduler, _process_map, job_tracker, proc_sd15, proc_sdxl = _build_wedged_scheduler(
            free_vram_mb=_ACHIEVABLE_FREE_VRAM_MB,
        )
        # Ample available RAM so the (now-reached) RAM gate does not itself defer.
        monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: 64000.0)
        await _enqueue_head_jobs(job_tracker)

        preloaded_any = any(scheduler.preload_models() for _ in range(5))

        assert preloaded_any is True
        assert HordeControlFlag.PRELOAD_MODEL in {
            proc_sd15.last_control_flag,
            proc_sdxl.last_control_flag,
        }


class TestWedgeManifestsAsStructuralQueueDeadlock:
    """The budget-starved head state is exactly what the queue-deadlock detector flags as a wedge."""

    async def test_idle_pool_with_unresident_head_is_a_structural_wedge(self) -> None:
        """The permanently-deferred-head state produces a sustained 'queue deadlock' wedge signature.

        With every inference process idle, pending work whose model is resident on none of them, and
        no recent pop, the detector flags a queue deadlock. A permanently-deferred preload keeps that
        deadlock set indefinitely, so once it outlasts the normal model-load / churn window it reads as
        a structural wedge: the signal save-our-ship uses to fault the head. (A merely transient
        all-idle gap, by contrast, is not yet structural; see ``test_transient_wedge_giveup_repro``.)
        """
        proc_sd15 = make_mock_process_info(1, model_name=_RESIDENT_SD15, state=HordeProcessState.WAITING_FOR_JOB)
        proc_sdxl = make_mock_process_info(2, model_name=_RESIDENT_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({1: proc_sd15, 2: proc_sdxl})

        job_tracker = JobTracker()
        await _enqueue_head_jobs(job_tracker)

        # No recent pop, so the detector does not defer to the recent-pop grace.
        state = WorkerState(last_job_pop_time=0.0)
        dispatcher = _make_message_dispatcher(
            state=state,
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=_wedge_bridge_data(),
        )

        assert process_map.all_waiting_for_job() is True

        dispatcher.detect_deadlock()

        assert dispatcher._in_queue_deadlock is True
        # The head model is named as the deadlock model, but it is resident on no process: the
        # "without a model causing it" branch (no loaded model can explain the stall).
        assert dispatcher._queue_deadlock_model == _HEAD_MODEL
        # A permanent budget defer keeps the deadlock set indefinitely; once sustained it is structural.
        dispatcher._last_queue_deadlock_detected_time = time.time() - 60
        snapshot = dispatcher.get_deadlock_snapshot()
        assert snapshot.indicates_structural_wedge() is True
