"""Forward-progress contracts between whole-card residency and post-processing admission."""

from __future__ import annotations

import time
from unittest.mock import Mock

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeImageResult, HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)
from tests.process_management.regressions.test_whole_card_grant_and_dispatch_gate import (
    _CARD16_TOTAL_MB,
    _FLUX16_ESTABLISH_FREE_NOW_MB,
    _FLUX_ESTABLISH_RESERVE_MB,
    _FLUX_MODEL,
    _FLUX_WEIGHTS_MB,
    _SDXL_A,
    _bridge_data,
    _forecast_16gb,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler
from tests.process_management.workers import test_post_process_orchestration as post_process_tests


def _scheduler_with_active_residency(
    *,
    job_tracker: JobTracker,
) -> tuple[InferenceScheduler, ImageGenerateJobPopResponse]:
    """Build a held residency with an idle holder and an idle process for another model."""
    holder = make_mock_process_info(
        1,
        model_name=_FLUX_MODEL,
        state=HordeProcessState.WAITING_FOR_JOB,
    )
    other = make_mock_process_info(
        2,
        model_name=_SDXL_A,
        state=HordeProcessState.WAITING_FOR_JOB,
    )
    for process in (holder, other):
        process.total_vram_mb = _CARD16_TOTAL_MB
        process.vram_usage_mb = 1200

    process_map = ProcessMap({1: holder, 2: other})
    model_map = HordeModelMap(root={})
    model_map.update_entry(
        horde_model_name=_FLUX_MODEL,
        load_state=ModelLoadState.LOADED_IN_VRAM,
        process_id=1,
    )
    model_map.update_entry(
        horde_model_name=_SDXL_A,
        load_state=ModelLoadState.LOADED_IN_VRAM,
        process_id=2,
    )
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        horde_model_map=model_map,
        bridge_data=_bridge_data(
            whole_card_residency_cooldown_seconds=45,
            whole_card_residency_safety_off_gpu=True,
        ),
        max_concurrent=1,
        max_inference=2,
    )
    holder_job = make_job_pop_response(_FLUX_MODEL)
    scheduler._whole_card_ledger.record_grant(
        None,
        model=_FLUX_MODEL,
        forecast=_forecast_16gb(
            weights_mb=_FLUX_WEIGHTS_MB,
            reserve_mb=_FLUX_ESTABLISH_RESERVE_MB,
            free_now_mb=_FLUX16_ESTABLISH_FREE_NOW_MB,
            wants_whole_card=True,
        ),
        cooldown_until=time.time() + 45,
        now=time.time(),
        refresh_established=True,
    )
    return scheduler, holder_job


async def test_drained_residency_yields_to_ready_different_model_head() -> None:
    """A speculative cooldown must not park a ready queue head for another resident model."""
    tracker = JobTracker()
    scheduler, _holder_job = _scheduler_with_active_residency(job_tracker=tracker)
    next_job = make_job_pop_response(_SDXL_A)
    await track_popped_job_async(tracker, next_job)

    scheduler._restore_siblings_after_whole_card()

    assert scheduler.is_whole_card_residency_active() is False
    assert await scheduler.start_inference() is True
    assert next_job in tracker.jobs_in_progress


async def test_residency_release_keeps_safety_off_until_pending_post_processing_drains() -> None:
    """Releasing a drained lease must not consume the room needed by accepted downstream work."""
    tracker = JobTracker()
    scheduler, _holder_job = _scheduler_with_active_residency(job_tracker=tracker)
    post_process_job = HordeJobInfo(
        sdk_api_job_info=make_job_pop_response(_FLUX_MODEL, post_processing=["RealESRGAN_x4plus"]),
        job_image_results=[HordeImageResult(image_bytes=b"raw-image")],
        state=GENERATION_STATE.ok,
        time_popped=time.time(),
    )
    await tracker.queue_for_post_processing(post_process_job)
    scheduler._residency_state(None).cooldown_until = time.time() - 1
    scheduler._process_lifecycle.restore_safety_on_gpu = Mock(return_value=True)

    scheduler._restore_siblings_after_whole_card()

    assert scheduler.is_whole_card_residency_active() is False
    scheduler._process_lifecycle.restore_safety_on_gpu.assert_not_called()


async def test_post_processing_admission_applies_remaining_safety_reclaim(monkeypatch: object) -> None:
    """A deferred lane head applies the arbiter's safety reclaim when no idle model remains to evict."""
    monkeypatch.setattr(  # type: ignore[attr-defined]
        post_process_tests.post_process_orchestrator_module,
        "predict_job_post_processing_vram_mb",
        lambda *_args, **_kwargs: 6429.0,
    )
    process_manager = make_testable_process_manager(
        enable_vram_budget=True,
        safety_on_gpu=True,
        whole_card_residency_safety_off_gpu=True,
        post_processing_lane_enabled=True,
        vram_reserve_mb=1596,
        ram_reserve_mb=8192,
        device_free_mb=5000.0,
    )
    lane = post_process_tests._make_lane_process()
    safety = make_mock_process_info(
        8,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.SAFETY,
    )
    idle_inference = make_mock_process_info(
        9,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
    )
    for process in (lane, safety, idle_inference):
        process.total_vram_mb = _CARD16_TOTAL_MB
        process.vram_usage_mb = 0
        process.process_reserved_mb = 0
        process.process_allocated_mb = 0
    safety.process_reserved_mb = 3000
    safety.process_allocated_mb = 3000
    process_manager._process_map.clear()
    process_manager._process_map.update({7: lane, 8: safety, 9: idle_inference})

    job_info = post_process_tests._make_pp_job_info(["RealESRGAN_x4plus"])
    await process_manager._job_tracker.queue_for_post_processing(job_info)
    process_manager._begin_vram_arbiter_cycle()

    await process_manager.start_post_processing()

    assert job_info in process_manager._job_tracker.jobs_pending_post_processing
    assert process_manager._process_lifecycle.is_safety_gpu_paused is True
