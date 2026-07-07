"""Regression: a reclaim-ladder pause of the post-processing lane must not wedge the drain.

The reclaim ladder can stop the post-processing lane off the GPU as a saturation-relief rung. When a single
shared latch guarded that pause, only the whole-card residency completion loop cleared it, so a ladder-initiated
pause (which has no residency grant to complete) stayed set forever: the lane never restarted, and a job
stranded in ``PENDING_POST_PROCESSING`` neither dispatched nor aged out, so the harness drain gate never
cleared. These tests pin the fix from both sides: the ladder owns and restores its own pause when the card
returns HEALTHY (through the real scheduler actuator and lifecycle), and while the lane is ladder-paused a
stranded job runs its patience clock and takes the raw-image fallback rather than waiting on a lane that may
never return.
"""

from __future__ import annotations

from horde_sdk.ai_horde_api import GENERATION_STATE

from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
from horde_worker_regen.process_management.resources.reclaim_ladder import ReclaimRung, ReclaimRungKind
from horde_worker_regen.process_management.workers import post_process_orchestrator as pp_module
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager
from tests.process_management.workers.test_post_process_orchestration import _make_pp_job_info


def _pp_pause_ladder() -> tuple[ReclaimRung, ...]:
    """A one-rung ladder that pauses the post-processing lane (the saturation-relief rung under test)."""
    return (
        ReclaimRung(
            kind=ReclaimRungKind.PAUSE_PP_LANE,
            device_index=0,
            promised_freed_mb=1400.0,
            tenant_label="post_process_lane",
        ),
    )


def test_ladder_paused_pp_lane_is_restored_when_the_card_returns_healthy() -> None:
    """Driving the real ladder against the real scheduler actuator: a HEALTHY card unwinds the ladder's pause.

    This exercises the full seam the wedge lived on: engine -> scheduler.pause/restore_post_process_lane ->
    lifecycle pause/restore, with the pause tagged RECLAIM_LADDER so the ladder (not the residency) owns it.
    """
    process_manager = make_testable_process_manager(post_processing_lane_enabled=True)
    process_manager._process_map.clear()
    # A running lane so the ladder's restore start-hook is a clean no-op (the lane is still mapped); the point
    # under test is the pause latch, which gates whether the per-tick start hook may bring the lane back.
    process_manager._process_map[7] = make_mock_process_info(
        7,
        process_type=HordeProcessType.POST_PROCESS,
        model_name=None,
    )
    ladder = _pp_pause_ladder()

    # SATURATED: the ladder issues its PP-lane pause through the real scheduler actuator.
    process_manager._reclaim_ladder.on_tick(
        0,
        saturated=True,
        device_free_mb=100.0,
        actuator=process_manager._inference_scheduler,
        ladder_builder=lambda: ladder,
    )
    assert process_manager._process_lifecycle.is_post_process_gpu_paused is True
    assert process_manager._process_lifecycle.post_process_pause_owner is PauseOwner.RECLAIM_LADDER

    # The card returns HEALTHY: the ladder unwinds its own pause, so the lane is no longer held off-GPU.
    process_manager._reclaim_ladder.on_tick(
        0,
        saturated=False,
        healthy=True,
        device_free_mb=9000.0,
        actuator=process_manager._inference_scheduler,
        ladder_builder=lambda: ladder,
    )
    assert process_manager._process_lifecycle.is_post_process_gpu_paused is False
    assert process_manager._process_lifecycle.post_process_pause_owner is None


def test_ladder_pause_is_held_through_pressure_and_not_cleared_by_a_residency_restore() -> None:
    """A ladder pause survives the PRESSURE band and a whole-card residency restore; only the ladder lifts it."""
    process_manager = make_testable_process_manager(post_processing_lane_enabled=True)
    process_manager._process_map.clear()
    process_manager._process_map[7] = make_mock_process_info(
        7,
        process_type=HordeProcessType.POST_PROCESS,
        model_name=None,
    )
    ladder = _pp_pause_ladder()

    process_manager._reclaim_ladder.on_tick(
        0,
        saturated=True,
        device_free_mb=100.0,
        actuator=process_manager._inference_scheduler,
        ladder_builder=lambda: ladder,
    )
    assert process_manager._process_lifecycle.post_process_pause_owner is PauseOwner.RECLAIM_LADDER

    # A whole-card residency completion restore (a different owner) must not lift the ladder's hold.
    assert process_manager._process_lifecycle.restore_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD) is False
    assert process_manager._process_lifecycle.is_post_process_gpu_paused is True

    # Saturation only eases into PRESSURE (not HEALTHY): the ladder holds the pause rather than flapping the
    # lane's context back onto a still-tight card.
    process_manager._reclaim_ladder.on_tick(
        0,
        saturated=False,
        healthy=False,
        device_free_mb=500.0,
        actuator=process_manager._inference_scheduler,
        ladder_builder=lambda: ladder,
    )
    assert process_manager._process_lifecycle.is_post_process_gpu_paused is True


async def test_pending_pp_job_takes_the_fallback_while_the_lane_is_ladder_paused() -> None:
    """A job stranded behind a ladder-paused lane runs its patience clock and faults out, never wedging.

    With no lane process and a ladder-owned pause, the orchestrator arms the patience clock (a residency pause
    would suppress it, but a ladder pause has no bounded restore guarantee), and past the window the job takes
    the existing raw-image fallback so the drain proceeds.
    """
    process_manager = make_testable_process_manager(post_processing_lane_enabled=True)
    process_manager._process_map.clear()  # the lane was torn down by the pause; nothing serves the queue
    assert process_manager._process_lifecycle.pause_post_process_off_gpu(owner=PauseOwner.RECLAIM_LADDER) is True

    job_info = _make_pp_job_info()
    await process_manager._job_tracker.queue_for_post_processing(job_info)

    await process_manager.start_post_processing()
    orchestrator = process_manager._post_process_orchestrator
    key = str(job_info.sdk_api_job_info.id_)
    assert key in orchestrator._deferrals  # the liveness clock armed despite the pause

    orchestrator._deferrals[key].first_deferred_at -= pp_module._ADMISSION_PATIENCE_SECONDS + 1.0
    await process_manager.start_post_processing()

    assert job_info not in process_manager._job_tracker.jobs_pending_post_processing
    assert job_info in process_manager._job_tracker.jobs_pending_submit
    assert job_info.state == GENERATION_STATE.faulted
    assert job_info.job_image_results is None
