"""Reproduces a PP drain head deadlocking behind idle disaggregation-lane contexts.

The dedicated post-processing lane is admitted against truthful device-free VRAM.  On the 16 GB live shape
that motivated this regression, a two-operation upscale/face-fix chain asks for 6,429 MB while the arbiter sees
6,856 MB free and keeps its 819 MB noise margin: only 6,037 MB is admissible.  Idle-model and allocator-cache
reclaim can already be exhausted at that point; the remaining shortfall is held by idle CUDA contexts belonging
to the VAE and component service lanes.

At the same time the inference scheduler intentionally withholds the next sampler so accepted post-processing
gets a drain window.  If PP admission cannot borrow one of those *idle* service-lane contexts, both sides wait:
all processes are idle, the queue-deadlock detector reaches its structural horizon, and Save-our-ship rebuilds
the whole pool.  The reset appears to fix the problem only because it incidentally frees the contexts; the PP
job starts immediately afterward.

These tests specify the cooperative boundary before production code changes:

* after softer reclaim is exhausted, a PP drain may borrow the first idle service lane in the existing reclaim
  order, and must restore only that borrowed lane once accepted PP work drains;
* a busy service lane is never stopped;
* an active sampler is never disturbed to make PP fit; and
* disabling safety-off-GPU remains authoritative -- PP liveness cannot bypass that operator policy.

"""

from __future__ import annotations

from typing import cast
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager
from tests.process_management.workers import test_post_process_orchestration as pp_tests

_CARD_TOTAL_MB = 16_375
_DEVICE_FREE_MB = 6_856.0
_PP_PEAK_MB = 6_429.0


def _live_process(
    process_id: int,
    process_type: HordeProcessType,
    *,
    state: HordeProcessState = HordeProcessState.WAITING_FOR_JOB,
    reserved_mb: int,
) -> HordeProcessInfo:
    """Build one GPU child with the allocator figures observed in the live wedge."""
    process = make_mock_process_info(
        process_id,
        model_name=None,
        state=state,
        process_type=process_type,
    )
    process.total_vram_mb = _CARD_TOTAL_MB
    process.vram_usage_mb = reserved_mb
    process.process_reserved_mb = reserved_mb
    process.process_allocated_mb = reserved_mb
    return process


def _live_shaped_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[HordeWorkerProcessManager, HordeProcessInfo, HordeProcessInfo, HordeProcessInfo]:
    """Return a manager whose bare GPU contexts leave the PP head just below safe admission."""
    monkeypatch.setattr(
        pp_tests.post_process_orchestrator_module,
        "predict_job_post_processing_vram_mb",
        lambda *_args, **_kwargs: _PP_PEAK_MB,
    )
    manager = make_testable_process_manager(
        enable_vram_budget=True,
        enable_pipeline_disaggregation=True,
        post_processing_lane_enabled=True,
        safety_on_gpu=True,
        # This operator policy is deliberately false in the live configuration.  The PP fix must not override it.
        whole_card_residency_safety_off_gpu=False,
        vram_reserve_mb=1_596,
        device_free_mb=_DEVICE_FREE_MB,
    )
    post_process = _live_process(1, HordeProcessType.POST_PROCESS, reserved_mb=1_288)
    component = _live_process(2, HordeProcessType.COMPONENT, reserved_mb=1_350)
    vae = _live_process(3, HordeProcessType.VAE_LANE, reserved_mb=1_330)
    safety = _live_process(0, HordeProcessType.SAFETY, reserved_mb=2_430)
    inference_a = _live_process(4, HordeProcessType.INFERENCE, reserved_mb=1_288)
    inference_b = _live_process(5, HordeProcessType.INFERENCE, reserved_mb=1_362)
    manager._process_map.clear()
    manager._process_map.update(
        {
            0: safety,
            1: post_process,
            2: component,
            3: vae,
            4: inference_a,
            5: inference_b,
        },
    )
    return manager, vae, component, safety


async def test_pp_drain_borrows_idle_service_lane_before_the_sos_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    """The live-shaped non-fit starts targeted idle-context reclaim instead of waiting for a pool reset."""
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    job_info = pp_tests._make_pp_job_info(["CodeFormers", "RealESRGAN_x4plus"])
    await manager._job_tracker.queue_for_post_processing(job_info)
    manager._begin_vram_arbiter_cycle()

    await manager.start_post_processing()
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is False

    # The first rejection spends only the existing model/cache reclaim.  A fresh measurement still shows the
    # same non-fit, proving that softer rung did not yield enough; only this re-check may borrow a lane context.
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()

    assert job_info in manager._job_tracker.jobs_pending_post_processing
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is True, (
        "PP admission exhausted model/cache reclaim but did not borrow the first idle service-lane context; "
        "inference will hold for the PP drain while PP keeps waiting, and SOS will rebuild the entire pool"
    )
    assert manager._process_lifecycle.is_component_gpu_paused is False
    assert manager._process_lifecycle.is_safety_gpu_paused is False

    # An unchanged later sample must not stack a second teardown while this bounded loan is outstanding. If one
    # idle context is insufficient, the existing aging and SOS backstops remain authoritative.
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()
    assert manager._process_lifecycle.is_component_gpu_paused is False


async def test_pp_drain_restores_only_its_borrowed_lane_after_work_drains(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PP-specific context loan is bounded by accepted PP work and does not become a permanent lane shed."""
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    job_info = pp_tests._make_pp_job_info(["CodeFormers", "RealESRGAN_x4plus"])
    await manager._job_tracker.queue_for_post_processing(job_info)
    manager._begin_vram_arbiter_cycle()

    await manager.start_post_processing()
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is False
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is True

    # Model the context exit becoming visible to NVML.  The next fresh snapshot admits and dispatches PP.
    manager._last_device_free_mb_by_device[0] = 8_200.0
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()
    assert job_info in manager._job_tracker.jobs_being_post_processed
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is True

    job_id = job_info.sdk_api_job_info.id_
    assert job_id is not None
    taken = await manager._job_tracker.take_being_post_processed(job_id)
    assert taken is job_info
    await manager.start_post_processing()

    assert manager._process_lifecycle.is_vae_lane_gpu_paused is False
    assert manager._process_lifecycle.is_component_gpu_paused is False


async def test_pp_drain_never_borrows_a_busy_service_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live disaggregation work is a hard boundary: PP waits rather than tearing either producer down."""
    manager, vae, component, _safety = _live_shaped_manager(monkeypatch)
    vae.last_process_state = HordeProcessState.INFERENCE_STARTING
    component.last_process_state = HordeProcessState.INFERENCE_STARTING
    job_info = pp_tests._make_pp_job_info(["CodeFormers", "RealESRGAN_x4plus"])
    await manager._job_tracker.queue_for_post_processing(job_info)
    manager._begin_vram_arbiter_cycle()

    await manager.start_post_processing()
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()

    assert job_info in manager._job_tracker.jobs_pending_post_processing
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is False
    assert manager._process_lifecycle.is_component_gpu_paused is False


async def test_pp_drain_does_not_reclaim_service_lanes_while_sampling_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing sampling/PP mutex remains authoritative; an active sampler is allowed to finish."""
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    manager._process_map[4].last_process_state = HordeProcessState.INFERENCE_STARTING
    manager._post_process_orchestrator._sampling_coresidency_check = Mock(return_value=False)
    job_info = pp_tests._make_pp_job_info(["CodeFormers", "RealESRGAN_x4plus"])
    await manager._job_tracker.queue_for_post_processing(job_info)
    manager._begin_vram_arbiter_cycle()

    await manager.start_post_processing()
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()

    assert job_info in manager._job_tracker.jobs_pending_post_processing
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is False
    assert manager._process_lifecycle.is_component_gpu_paused is False
    assert manager._process_lifecycle.is_safety_gpu_paused is False


async def test_pp_drain_does_not_bypass_safety_off_gpu_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no idle service lane available, safety remains on GPU when the operator forbids that reclaim."""
    manager, vae, component, safety = _live_shaped_manager(monkeypatch)
    vae.last_process_state = HordeProcessState.INFERENCE_STARTING
    component.last_process_state = HordeProcessState.INFERENCE_STARTING
    job_info = pp_tests._make_pp_job_info(["CodeFormers", "RealESRGAN_x4plus"])
    await manager._job_tracker.queue_for_post_processing(job_info)
    manager._begin_vram_arbiter_cycle()

    await manager.start_post_processing()
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()

    assert safety.last_process_state is HordeProcessState.WAITING_FOR_JOB
    assert manager._process_lifecycle.is_safety_gpu_paused is False


async def test_pp_drain_does_not_skip_persistent_cache_reclaim_to_borrow_a_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A still-measured allocator cache keeps the request on the softer rung, even on later re-checks."""
    manager, _vae, _component, _safety = _live_shaped_manager(monkeypatch)
    inference = manager._process_map[4]
    inference.process_reserved_mb = 2_500
    inference.process_allocated_mb = 1_000
    job_info = pp_tests._make_pp_job_info(["CodeFormers", "RealESRGAN_x4plus"])
    await manager._job_tracker.queue_for_post_processing(job_info)

    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()

    assert cast(Mock, inference.pipe_connection.send).call_count >= 1
    assert manager._process_lifecycle.is_vae_lane_gpu_paused is False
    assert manager._process_lifecycle.is_component_gpu_paused is False


async def test_pp_drain_never_restores_a_same_owner_pause_it_did_not_acquire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A governor-owned RECLAIM_LADDER pause is not mistaken for the PP drain's own context loan."""
    manager, _vae, component, _safety = _live_shaped_manager(monkeypatch)
    assert manager._process_lifecycle.pause_vae_lane_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True
    component.last_process_state = HordeProcessState.INFERENCE_STARTING
    job_info = pp_tests._make_pp_job_info(["CodeFormers", "RealESRGAN_x4plus"])
    await manager._job_tracker.queue_for_post_processing(job_info)

    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()
    manager._begin_vram_arbiter_cycle()
    await manager.start_post_processing()
    await manager._job_tracker.abandon_pending_post_processing(job_info)
    await manager.start_post_processing()

    assert manager._process_lifecycle.is_vae_lane_gpu_paused is True
    assert manager._process_lifecycle.vae_lane_pause_owner is PauseOwner.RECLAIM_LADDER
