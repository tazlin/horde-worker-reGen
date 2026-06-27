"""Lifecycle-style whole-card residency regression matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.resource_budget import StreamForecast
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import make_job_pop_response, make_mock_process_info, track_popped_job_async
from tests.process_management.regressions.test_whole_card_deadlock_fixes import (
    _DEVICE_TOTAL_VRAM_MB,
    _FLUX_MODEL,
    _FLUX_WEIGHTS_MB,
    _MARGINAL_OVERHEAD_MB,
    _OTHER_SDXL,
    _PER_PROCESS_OVERHEAD_MB,
    _RESIDENT_SDXL,
    _deadlock_bridge_data,
    _make_real_plm,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


@dataclass(frozen=True)
class WholeCardLifecycleCase:
    """A deterministic whole-card queue lifecycle shape."""

    mode: Literal["initial", "prestaged"]
    target: int
    total_processes: int
    holder_state: HordeProcessState
    holder_load_state: ModelLoadState
    sibling_models: tuple[str | None, ...]
    queue_tail: tuple[str, ...]
    max_threads: int = 1
    safety_on_gpu: bool = False
    safety_pause_required: bool = False


@dataclass(frozen=True)
class LifecycleHarness:
    """Objects shared by a lifecycle matrix case."""

    scheduler: InferenceScheduler
    process_map: ProcessMap
    horde_model_map: HordeModelMap
    job_tracker: JobTracker
    flux_job: ImageGenerateJobPopResponse


def _forecast_for_target(target: int) -> StreamForecast:
    """Return a whole-card forecast whose process target is deterministic on the 24 GB fixture card."""
    reserve_mb = 6500.0 if target == 1 else 5000.0
    forecast = StreamForecast(
        total_vram_mb=float(_DEVICE_TOTAL_VRAM_MB),
        free_now_mb=float(_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB),
        free_if_alone_mb=float(_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB),
        free_after_model_evict_mb=float(_DEVICE_TOTAL_VRAM_MB - _PER_PROCESS_OVERHEAD_MB),
        weights_mb=_FLUX_WEIGHTS_MB,
        reserve_mb=reserve_mb,
        per_process_overhead_mb=float(_PER_PROCESS_OVERHEAD_MB),
        marginal_process_overhead_mb=_MARGINAL_OVERHEAD_MB,
        wants_whole_card=True,
    )
    assert forecast.max_resident_processes() == target
    return forecast


def _record_loaded_model(
    horde_model_map: HordeModelMap,
    *,
    model_name: str | None,
    load_state: ModelLoadState,
    process_id: int,
) -> None:
    if model_name is None:
        return
    horde_model_map.update_entry(horde_model_name=model_name, load_state=load_state, process_id=process_id)


def _make_flux_head_harness(case: WholeCardLifecycleCase) -> LifecycleHarness:
    """Build a queue-head Flux lifecycle with a real PLM and mocked process pipes."""
    processes = {
        1: make_mock_process_info(1, model_name=_FLUX_MODEL, state=case.holder_state),
    }
    for offset, model_name in enumerate(case.sibling_models, start=2):
        processes[offset] = make_mock_process_info(
            offset,
            model_name=model_name,
            state=HordeProcessState.WAITING_FOR_JOB,
        )

    for process in processes.values():
        process.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
        process.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB

    process_map = ProcessMap(processes)
    horde_model_map = HordeModelMap(root={})
    _record_loaded_model(horde_model_map, model_name=_FLUX_MODEL, load_state=case.holder_load_state, process_id=1)
    for process_id, model_name in enumerate(case.sibling_models, start=2):
        _record_loaded_model(
            horde_model_map,
            model_name=model_name,
            load_state=ModelLoadState.LOADED_IN_VRAM,
            process_id=process_id,
        )

    job_tracker = JobTracker()
    bridge_data = _deadlock_bridge_data(
        max_threads=case.max_threads,
        safety_on_gpu=case.safety_on_gpu,
        whole_card_residency_safety_off_gpu=case.safety_pause_required,
    )
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        horde_model_map=horde_model_map,
        bridge_data=bridge_data,
        max_concurrent=case.max_threads,
        max_inference=case.total_processes,
    )
    scheduler._process_lifecycle = _make_real_plm(
        process_map=process_map,
        job_tracker=job_tracker,
        horde_model_map=horde_model_map,
        bridge_data=bridge_data,
        target_process_count=case.total_processes,
    )
    if case.safety_pause_required:
        scheduler._process_lifecycle._safety_gpu_paused = True

    flux_job = make_job_pop_response(_FLUX_MODEL, width=1216, height=1216)
    return LifecycleHarness(
        scheduler=scheduler,
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        flux_job=flux_job,
    )


async def _queue_flux_head_case(harness: LifecycleHarness, case: WholeCardLifecycleCase) -> None:
    await track_popped_job_async(harness.job_tracker, harness.flux_job)
    for model in case.queue_tail:
        await track_popped_job_async(harness.job_tracker, make_job_pop_response(model))


def _begin_residency_for_case(harness: LifecycleHarness, case: WholeCardLifecycleCase) -> None:
    forecast = _forecast_for_target(case.target)
    if case.mode == "initial":
        harness.scheduler._establish_whole_card_residency(harness.flux_job, forecast, announce=False)
    else:
        harness.scheduler._begin_whole_card_residency(harness.flux_job, forecast, announce=False)


def _complete_child_preload_acks(harness: LifecycleHarness) -> None:
    """Model the child-side PRELOAD_MODEL acknowledgement between scheduler ticks."""
    for process in harness.process_map.values():
        if process.last_control_flag != HordeControlFlag.PRELOAD_MODEL:
            continue
        model_name = process.loaded_horde_model_name
        if model_name is None:
            continue
        process.last_process_state = HordeProcessState.PRELOADED_MODEL
        process.last_control_flag = None
        harness.horde_model_map.update_entry(
            horde_model_name=model_name,
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=process.process_id,
        )


async def _drive_scheduler_lifecycle(
    harness: LifecycleHarness,
    *,
    expected_job: ImageGenerateJobPopResponse,
    max_cycles: int = 6,
) -> ImageGenerateJobPopResponse | None:
    """Run deterministic scheduling ticks until inference dispatches or the lifecycle wedges."""
    for _ in range(max_cycles):
        harness.scheduler._converge_whole_card_residency()
        harness.scheduler.preload_models()
        _complete_child_preload_acks(harness)
        if await harness.scheduler.start_inference():
            assert len(harness.job_tracker.jobs_in_progress) == 1
            dispatched = harness.job_tracker.jobs_in_progress[0]
            assert dispatched is expected_job
            return dispatched
    return None


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            WholeCardLifecycleCase(
                mode="initial",
                target=1,
                total_processes=3,
                holder_state=HordeProcessState.WAITING_FOR_JOB,
                holder_load_state=ModelLoadState.LOADED_IN_VRAM,
                sibling_models=(_RESIDENT_SDXL, _OTHER_SDXL),
                queue_tail=(_RESIDENT_SDXL, _OTHER_SDXL),
            ),
            id="initial-target1-queued-siblings",
        ),
        pytest.param(
            WholeCardLifecycleCase(
                mode="initial",
                target=2,
                total_processes=4,
                holder_state=HordeProcessState.WAITING_FOR_JOB,
                holder_load_state=ModelLoadState.LOADED_IN_VRAM,
                sibling_models=(_RESIDENT_SDXL, _OTHER_SDXL, None),
                queue_tail=(_OTHER_SDXL, _RESIDENT_SDXL),
                max_threads=2,
            ),
            id="initial-target2-high-vram-mixed-siblings",
        ),
        pytest.param(
            WholeCardLifecycleCase(
                mode="prestaged",
                target=1,
                total_processes=3,
                holder_state=HordeProcessState.PRELOADED_MODEL,
                holder_load_state=ModelLoadState.LOADED_IN_RAM,
                sibling_models=(_RESIDENT_SDXL, _OTHER_SDXL),
                queue_tail=(_RESIDENT_SDXL, _OTHER_SDXL),
            ),
            id="prestaged-ram-target1-post-drain",
        ),
        pytest.param(
            WholeCardLifecycleCase(
                mode="prestaged",
                target=2,
                total_processes=4,
                holder_state=HordeProcessState.PRELOADED_MODEL,
                holder_load_state=ModelLoadState.LOADED_IN_RAM,
                sibling_models=(_RESIDENT_SDXL, None, _OTHER_SDXL),
                queue_tail=(_OTHER_SDXL, _RESIDENT_SDXL),
                max_threads=2,
            ),
            id="prestaged-ram-target2-mixed-siblings",
        ),
        pytest.param(
            WholeCardLifecycleCase(
                mode="prestaged",
                target=1,
                total_processes=2,
                holder_state=HordeProcessState.PRELOADED_MODEL,
                holder_load_state=ModelLoadState.LOADED_IN_RAM,
                sibling_models=(_RESIDENT_SDXL,),
                queue_tail=(_RESIDENT_SDXL,),
                safety_on_gpu=True,
                safety_pause_required=True,
            ),
            id="prestaged-target1-safety-already-paused",
        ),
    ],
)
async def test_whole_card_head_lifecycle_matrix_converges(case: WholeCardLifecycleCase) -> None:
    """A whole-card head should converge and dispatch across representative worker lifecycle states."""
    harness = _make_flux_head_harness(case)
    await _queue_flux_head_case(harness, case)
    _begin_residency_for_case(harness, case)

    dispatched = await _drive_scheduler_lifecycle(harness, expected_job=harness.flux_job)

    assert dispatched is harness.flux_job
    assert harness.process_map.num_loaded_inference_processes() <= case.target
    holder = harness.process_map.get(1)
    assert holder is not None
    assert holder.last_control_flag == HordeControlFlag.START_INFERENCE
    assert holder.loaded_horde_model_name == _FLUX_MODEL


@pytest.mark.parametrize(
    "queue_models",
    [
        pytest.param((_RESIDENT_SDXL, _FLUX_MODEL), id="resident-head-then-flux"),
        pytest.param((_RESIDENT_SDXL, _OTHER_SDXL, _FLUX_MODEL), id="resident-head-sdxl-then-flux"),
    ],
)
async def test_whole_card_residency_waits_until_flux_reaches_queue_head(
    queue_models: tuple[str, ...],
) -> None:
    """Flux behind a ready resident head must not reserve the card before its queue turn."""
    head_process = make_mock_process_info(
        1,
        model_name=_RESIDENT_SDXL,
        state=HordeProcessState.PRELOADED_MODEL,
    )
    flux_process = make_mock_process_info(2, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    for process in (head_process, flux_process):
        process.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
        process.vram_usage_mb = _PER_PROCESS_OVERHEAD_MB

    process_map = ProcessMap({1: head_process, 2: flux_process})
    horde_model_map = HordeModelMap(root={})
    horde_model_map.update_entry(
        horde_model_name=_RESIDENT_SDXL,
        load_state=ModelLoadState.LOADED_IN_RAM,
        process_id=1,
    )
    job_tracker = JobTracker()
    jobs = [make_job_pop_response(model) for model in queue_models]
    for job in jobs:
        await track_popped_job_async(job_tracker, job)

    bridge_data = _deadlock_bridge_data(image_models_to_load=[_RESIDENT_SDXL, _OTHER_SDXL, _FLUX_MODEL])
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        horde_model_map=horde_model_map,
        bridge_data=bridge_data,
        max_concurrent=1,
        max_inference=2,
    )
    scheduler._process_lifecycle = _make_real_plm(
        process_map=process_map,
        job_tracker=job_tracker,
        horde_model_map=horde_model_map,
        bridge_data=bridge_data,
        target_process_count=2,
    )

    scheduler.preload_models()
    assert scheduler.whole_card_residency_state().active is False

    assert await scheduler.start_inference() is True
    assert jobs[0] in job_tracker.jobs_in_progress
    assert jobs[0].model == _RESIDENT_SDXL
    assert head_process.last_control_flag == HordeControlFlag.START_INFERENCE
    assert scheduler.whole_card_residency_state().active is False
    assert flux_process.last_control_flag != HordeControlFlag.START_INFERENCE
    assert all(process.process_type is HordeProcessType.INFERENCE for process in process_map.values())
