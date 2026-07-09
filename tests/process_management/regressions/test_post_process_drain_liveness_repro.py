"""Post-processing work must keep its own liveness path after inference accepts it."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
from tests.process_management.conftest import (
    make_job_pop_response,
    make_testable_process_manager,
    track_popped_job_async,
)
from tests.process_management.workers.test_post_process_orchestration import _make_pp_job_info


async def _noop_sleep(_delay: float) -> None:
    return None


async def test_pending_post_processing_starts_the_lane_when_no_lane_is_present() -> None:
    """Queued post-processing work should request a lane when the lane is enabled and absent."""
    process_manager = make_testable_process_manager(post_processing_lane_enabled=True)
    process_manager._process_map.clear()
    process_manager._process_lifecycle.start_post_process_processes = Mock(return_value=True)  # type: ignore[method-assign]

    job_info = _make_pp_job_info()
    await process_manager._job_tracker.queue_for_post_processing(job_info)

    await process_manager.start_post_processing()

    process_manager._process_lifecycle.start_post_process_processes.assert_called_once()


async def test_pending_post_processing_lifts_stale_whole_card_pause_when_no_inference_remains() -> None:
    """A stale whole-card lane pause should not hold post-processing after inference has drained."""
    process_manager = make_testable_process_manager(post_processing_lane_enabled=True)
    process_manager._process_map.clear()
    assert process_manager._process_lifecycle.pause_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD) is True
    process_manager._process_lifecycle.restore_post_process_off_gpu = Mock(return_value=True)  # type: ignore[method-assign]

    job_info = _make_pp_job_info()
    await process_manager._job_tracker.queue_for_post_processing(job_info)
    assert len(process_manager._job_tracker.jobs_pending_inference) == 0
    assert len(process_manager._job_tracker.jobs_in_progress) == 0

    await process_manager.start_post_processing()

    process_manager._process_lifecycle.restore_post_process_off_gpu.assert_called_once_with(
        owner=PauseOwner.WHOLE_CARD,
    )


async def test_shutdown_drain_drives_post_processing_queue_before_teardown() -> None:
    """Shutdown drain should keep driving accepted post-processing work after inference is empty."""
    process_manager = make_testable_process_manager(post_processing_lane_enabled=True)
    process_manager._sleep = _noop_sleep  # type: ignore[method-assign]
    process_manager._control_loop_tick = AsyncMock(return_value=False)  # type: ignore[method-assign]
    process_manager._start_timed_shutdown = Mock()  # type: ignore[method-assign]
    process_manager._process_lifecycle.start_safety_processes = Mock(return_value=True)  # type: ignore[method-assign]
    process_manager._process_lifecycle.start_inference_processes = Mock(return_value=True)  # type: ignore[method-assign]

    job_info = _make_pp_job_info()
    await process_manager._job_tracker.queue_for_post_processing(job_info)
    assert len(process_manager._job_tracker.jobs_pending_inference) == 0

    async def _drain_post_processing() -> None:
        await process_manager._job_tracker.abandon_pending_post_processing(job_info)

    process_manager.start_post_processing = AsyncMock(side_effect=_drain_post_processing)  # type: ignore[method-assign]

    await process_manager._process_control_loop()

    process_manager.start_post_processing.assert_awaited()


async def test_inference_start_does_not_clear_recovery_episode_while_post_processing_is_stuck() -> None:
    """SOS progress accounting must distinguish upstream starts from post-processing drain progress."""
    process_manager = make_testable_process_manager(post_processing_lane_enabled=True)
    coordinator = process_manager._recovery_coordinator
    tracker = process_manager._job_tracker

    pp_job_info = _make_pp_job_info()
    await tracker.queue_for_post_processing(pp_job_info)
    coordinator.episode_progress_baseline = tracker.total_num_completed_jobs
    coordinator.episode_inference_start_baseline = tracker.total_num_inference_starts
    coordinator.episode_post_processing_progress_baseline = tracker.total_num_post_processing_progress

    upstream_job = make_job_pop_response(model="stable_diffusion")
    await track_popped_job_async(tracker, upstream_job)
    await tracker.mark_inference_started(upstream_job)

    assert coordinator.made_progress_since_episode() is False

    await tracker.begin_post_processing(pp_job_info, process_id=7, process_launch_identifier=1)

    assert coordinator.made_progress_since_episode() is True
