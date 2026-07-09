"""Scheduler-side tests for the reclaim ladder: candidate assembly (active-sampler immunity) and actuators."""

from __future__ import annotations

import time
from collections.abc import Callable

from horde_sdk.ai_horde_api import GENERATION_STATE

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeImageResult, HordeProcessState
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.reclaim_ladder import ReclaimRungKind
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import make_job_pop_response, make_mock_bridge_data, make_mock_process_info
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _idle_resident(process_id: int, *, model: str, reserved_mb: int) -> object:
    info = make_mock_process_info(process_id, model_name=model, state=HordeProcessState.WAITING_FOR_JOB)
    info.process_reserved_mb = reserved_mb
    return info


def _scheduler_with_reclaimable_lanes(
    *,
    process_map: ProcessMap,
    bridge_data: object,
    job_tracker: JobTracker | None = None,
    post_processing_lane_commitments_provider: Callable[[], int] | None = None,
) -> InferenceScheduler:
    """Build a scheduler whose lane lifecycle flags are plain booleans for reclaim-candidate tests."""
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        bridge_data=bridge_data,  # type: ignore[arg-type]
        job_tracker=job_tracker,
        post_processing_lane_commitments_provider=post_processing_lane_commitments_provider,
    )
    scheduler._process_lifecycle.is_safety_gpu_paused = False
    scheduler._process_lifecycle.is_post_process_gpu_paused = False
    scheduler._process_lifecycle.is_vae_lane_gpu_paused = False
    scheduler._process_lifecycle.is_component_gpu_paused = False
    scheduler._process_lifecycle.vae_lane_enabled.return_value = False
    scheduler._process_lifecycle.component_lane_enabled.return_value = False
    return scheduler


def _pp_job_info() -> HordeJobInfo:
    """Build a minimal generated job that requested image post-processing."""
    return HordeJobInfo(
        sdk_api_job_info=make_job_pop_response(post_processing=["RealESRGAN_x4plus"]),
        job_image_results=[HordeImageResult(image_bytes=b"raw-image")],
        state=GENERATION_STATE.ok,
        censored=False,
        time_popped=time.time(),
    )


def test_busy_inference_process_is_not_a_reclaim_candidate() -> None:
    """An actively-sampling process is excluded from the idle-resident candidates (active-sampler immunity)."""
    idle = _idle_resident(1, model="m_idle", reserved_mb=5000)
    busy = make_mock_process_info(2, model_name="m_busy", state=HordeProcessState.INFERENCE_STARTING)
    busy.process_reserved_mb = 6000  # type: ignore[attr-defined]
    scheduler = _make_inference_scheduler(process_map=ProcessMap({1: idle, 2: busy}))  # type: ignore[dict-item]

    candidates = scheduler.build_reclaim_ladder_candidates(None)

    assert {resident.process_id for resident in candidates.idle_residents} == {1}


def test_idle_resident_promises_its_measured_reservation() -> None:
    """An idle resident's promised free is its measured device reservation."""
    idle = _idle_resident(1, model="m_idle", reserved_mb=5000)
    scheduler = _make_inference_scheduler(process_map=ProcessMap({1: idle}))  # type: ignore[dict-item]

    candidates = scheduler.build_reclaim_ladder_candidates(None)

    assert len(candidates.idle_residents) == 1
    assert candidates.idle_residents[0].footprint_mb == 5000.0


def test_unload_idle_model_refuses_a_busy_process() -> None:
    """The targeted unload actuator never unloads an actively-sampling process (the second immunity guard)."""
    busy = make_mock_process_info(2, model_name="m", state=HordeProcessState.INFERENCE_STARTING)
    scheduler = _make_inference_scheduler(process_map=ProcessMap({2: busy}))

    assert scheduler.unload_idle_model(2) is False


def test_unload_idle_model_sends_the_unload_flag_for_an_idle_resident() -> None:
    """The targeted unload actuator sends UNLOAD_MODELS_FROM_VRAM to a single idle resident and reports success."""
    idle = make_mock_process_info(1, model_name="m", state=HordeProcessState.WAITING_FOR_JOB)
    scheduler = _make_inference_scheduler(process_map=ProcessMap({1: idle}))

    assert scheduler.unload_idle_model(1) is True

    sent = [call.args[0] for call in idle.pipe_connection.send.call_args_list]
    assert any(getattr(message, "control_flag", None) is HordeControlFlag.UNLOAD_MODELS_FROM_VRAM for message in sent)


def test_absent_process_is_a_no_op_unload() -> None:
    """Unloading a process not in the map returns False rather than raising."""
    scheduler = _make_inference_scheduler(process_map=ProcessMap({}))
    assert scheduler.unload_idle_model(999) is False


def test_calibration_event_increments_the_counter() -> None:
    """A recorded shortfall bumps the scheduler's calibration counter (no footprint key applies at reclaim)."""
    from horde_worker_regen.process_management.resources.reclaim_ladder import ReclaimRung

    scheduler = _make_inference_scheduler(process_map=ProcessMap({}))
    rung = ReclaimRung(
        kind=ReclaimRungKind.UNLOAD_IDLE_MODEL,
        device_index=0,
        promised_freed_mb=2000.0,
        tenant_label="m",
        target_process_id=1,
    )

    scheduler.record_calibration_event(rung, promised_mb=2000.0, realized_mb=100.0)

    assert scheduler._reclaim_calibration_events == 1


def test_safety_off_gpu_rung_respects_residency_safety_setting() -> None:
    """A safety process is not a safety-off-GPU candidate when safety teardown is disabled for residency."""
    safety = make_mock_process_info(
        10,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.SAFETY,
    )
    safety.process_reserved_mb = 3044  # type: ignore[attr-defined]
    scheduler = _scheduler_with_reclaimable_lanes(
        process_map=ProcessMap({10: safety}),
        bridge_data=make_mock_bridge_data(
            safety_on_gpu=True,
            whole_card_residency_safety_off_gpu=False,
        ),
    )

    candidates = scheduler.build_reclaim_ladder_candidates(None)

    assert candidates.safety is None


def test_safety_off_gpu_rung_is_available_when_operator_allows_safety_teardown() -> None:
    """The safety-off-GPU rung remains available when both safety GPU placement and teardown are enabled."""
    safety = make_mock_process_info(
        10,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.SAFETY,
    )
    safety.process_reserved_mb = 3044  # type: ignore[attr-defined]
    scheduler = _scheduler_with_reclaimable_lanes(
        process_map=ProcessMap({10: safety}),
        bridge_data=make_mock_bridge_data(
            safety_on_gpu=True,
            whole_card_residency_safety_off_gpu=True,
        ),
    )

    candidates = scheduler.build_reclaim_ladder_candidates(None)

    assert candidates.safety is not None
    assert candidates.safety.kind is ReclaimRungKind.SAFETY_OFF_GPU


def test_safety_off_gpu_rung_is_absent_when_safety_is_configured_cpu_side() -> None:
    """A CPU-side safety configuration must not create an on-GPU safety teardown candidate."""
    safety = make_mock_process_info(
        10,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.SAFETY,
    )
    safety.process_reserved_mb = 3044  # type: ignore[attr-defined]
    scheduler = _scheduler_with_reclaimable_lanes(
        process_map=ProcessMap({10: safety}),
        bridge_data=make_mock_bridge_data(
            safety_on_gpu=False,
            whole_card_residency_safety_off_gpu=True,
        ),
    )

    candidates = scheduler.build_reclaim_ladder_candidates(None)

    assert candidates.safety is None


def test_post_processing_lane_rung_is_absent_when_lane_is_disabled() -> None:
    """A disabled post-processing lane does not produce a lane-pause candidate from stale process state."""
    post_process = make_mock_process_info(
        1,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.POST_PROCESS,
    )
    post_process.process_reserved_mb = 1288  # type: ignore[attr-defined]
    scheduler = _scheduler_with_reclaimable_lanes(
        process_map=ProcessMap({1: post_process}),
        bridge_data=make_mock_bridge_data(
            allow_post_processing=False,
            post_processing_lane_enabled=False,
        ),
    )

    candidates = scheduler.build_reclaim_ladder_candidates(None)

    assert all(candidate.kind is not ReclaimRungKind.PAUSE_PP_LANE for candidate in candidates.lanes)


def test_post_processing_lane_rung_is_available_when_lane_is_enabled() -> None:
    """An enabled idle post-processing lane remains available as a reclaim candidate."""
    post_process = make_mock_process_info(
        1,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.POST_PROCESS,
    )
    post_process.process_reserved_mb = 1288  # type: ignore[attr-defined]
    scheduler = _scheduler_with_reclaimable_lanes(
        process_map=ProcessMap({1: post_process}),
        bridge_data=make_mock_bridge_data(
            allow_post_processing=True,
            post_processing_lane_enabled=True,
        ),
    )

    candidates = scheduler.build_reclaim_ladder_candidates(None)

    assert any(candidate.kind is ReclaimRungKind.PAUSE_PP_LANE for candidate in candidates.lanes)


def test_busy_post_processing_lane_is_not_a_pause_rung() -> None:
    """A lane actively post-processing image work is not reclaimable as an idle lane."""
    post_process = make_mock_process_info(
        1,
        model_name=None,
        state=HordeProcessState.POST_PROCESSING,
        process_type=HordeProcessType.POST_PROCESS,
    )
    post_process.process_reserved_mb = 1288  # type: ignore[attr-defined]
    scheduler = _scheduler_with_reclaimable_lanes(
        process_map=ProcessMap({1: post_process}),
        bridge_data=make_mock_bridge_data(
            allow_post_processing=True,
            post_processing_lane_enabled=True,
        ),
    )

    candidates = scheduler.build_reclaim_ladder_candidates(None)

    assert all(candidate.kind is not ReclaimRungKind.PAUSE_PP_LANE for candidate in candidates.lanes)


def test_graph_alchemy_running_on_post_processing_lane_is_not_a_pause_rung() -> None:
    """A lane running graph-backed alchemy is busy even though it is not in JobTracker."""
    post_process = make_mock_process_info(
        1,
        model_name=None,
        state=HordeProcessState.ALCHEMY_STARTING,
        process_type=HordeProcessType.POST_PROCESS,
    )
    post_process.process_reserved_mb = 1288  # type: ignore[attr-defined]
    scheduler = _scheduler_with_reclaimable_lanes(
        process_map=ProcessMap({1: post_process}),
        bridge_data=make_mock_bridge_data(
            allow_post_processing=True,
            post_processing_lane_enabled=True,
        ),
    )

    candidates = scheduler.build_reclaim_ladder_candidates(None)

    assert all(candidate.kind is not ReclaimRungKind.PAUSE_PP_LANE for candidate in candidates.lanes)


async def test_pending_image_post_processing_keeps_lane_out_of_pause_rungs() -> None:
    """Queued image post-processing owns lane liveness, so the ladder must not cycle the idle process."""
    post_process = make_mock_process_info(
        1,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.POST_PROCESS,
    )
    post_process.process_reserved_mb = 1288  # type: ignore[attr-defined]
    job_tracker = JobTracker()
    await job_tracker.queue_for_post_processing(_pp_job_info())
    scheduler = _scheduler_with_reclaimable_lanes(
        process_map=ProcessMap({1: post_process}),
        bridge_data=make_mock_bridge_data(
            allow_post_processing=True,
            post_processing_lane_enabled=True,
        ),
        job_tracker=job_tracker,
    )

    candidates = scheduler.build_reclaim_ladder_candidates(None)

    assert all(candidate.kind is not ReclaimRungKind.PAUSE_PP_LANE for candidate in candidates.lanes)


def test_graph_alchemy_commitment_keeps_idle_post_processing_lane_out_of_pause_rungs() -> None:
    """Queued graph alchemy shares the PP lane, so the lane is not idle from the reclaim ladder's view."""
    post_process = make_mock_process_info(
        1,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.POST_PROCESS,
    )
    post_process.process_reserved_mb = 1288  # type: ignore[attr-defined]
    scheduler = _scheduler_with_reclaimable_lanes(
        process_map=ProcessMap({1: post_process}),
        bridge_data=make_mock_bridge_data(
            allow_post_processing=True,
            post_processing_lane_enabled=True,
        ),
        post_processing_lane_commitments_provider=lambda: 1,
    )

    candidates = scheduler.build_reclaim_ladder_candidates(None)

    assert all(candidate.kind is not ReclaimRungKind.PAUSE_PP_LANE for candidate in candidates.lanes)
