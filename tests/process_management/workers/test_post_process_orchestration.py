"""Tests for the dedicated post-processing lane: dispatch, result handling, recovery, and chain parity."""

from __future__ import annotations

import time
from unittest.mock import Mock

from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.worker.chaining import CHAIN_NODE_STATE

from horde_worker_regen.process_management.ipc.messages import (
    HordeImageResult,
    HordePostProcessControlMessage,
    HordePostProcessResultMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.scheduling.workload_flow import POST_PROCESS_RESERVE_FLOW
from horde_worker_regen.process_management.workers import post_process_orchestrator as post_process_orchestrator_module
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)


def _make_pp_job_info(post_processing: list[str] | None = None) -> HordeJobInfo:
    """Build a HordeJobInfo for a job requesting post-processing, carrying one raw image result."""
    job = make_job_pop_response(post_processing=post_processing or ["RealESRGAN_x4plus"])
    return HordeJobInfo(
        sdk_api_job_info=job,
        job_image_results=[HordeImageResult(image_bytes=b"raw-image")],
        state=GENERATION_STATE.ok,
        censored=False,
        time_popped=time.time(),
    )


def _make_lane_process(process_id: int = 7) -> Mock:
    return make_mock_process_info(
        process_id,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.POST_PROCESS,
    )


class TestStartPostProcessing:
    """Dispatch of pending post-processing jobs to the lane."""

    async def test_no_pending_jobs_returns_early(self) -> None:
        """With nothing queued, the orchestrator does nothing."""
        process_manager = make_testable_process_manager()
        await process_manager.start_post_processing()

    async def test_no_lane_process_leaves_job_pending(self) -> None:
        """A queued job stays pending when no lane process is available."""
        process_manager = make_testable_process_manager()
        job_info = _make_pp_job_info()
        await process_manager._job_tracker.queue_for_post_processing(job_info)

        await process_manager.start_post_processing()

        assert job_info in process_manager._job_tracker.jobs_pending_post_processing
        assert job_info not in process_manager._job_tracker.jobs_being_post_processed

    async def test_absent_lane_ages_job_out_to_no_image_fault(self) -> None:
        """With no lane process (and no deliberate pause), a pending job's patience clock starts.

        Deferral records are otherwise only created against a live lane process; without this arming, a
        job queued while the lane is dead (crash, failed restart) would never age out and would wait
        forever, wedging the drain. Once the patience window elapses the job is reported faulted without
        images so the horde reissues it to another worker.
        """
        process_manager = make_testable_process_manager()
        job_info = _make_pp_job_info()
        await process_manager._job_tracker.queue_for_post_processing(job_info)

        await process_manager.start_post_processing()
        orchestrator = process_manager._post_process_orchestrator
        key = str(job_info.sdk_api_job_info.id_)
        assert key in orchestrator._deferrals

        # Push the record past the patience window: the next pass must fault the job without images.
        orchestrator._deferrals[key].first_deferred_at -= (
            post_process_orchestrator_module._ADMISSION_PATIENCE_SECONDS + 1.0
        )
        await process_manager.start_post_processing()

        assert job_info not in process_manager._job_tracker.jobs_pending_post_processing
        assert job_info in process_manager._job_tracker.jobs_pending_submit
        assert job_info.state == GENERATION_STATE.faulted
        assert job_info.job_image_results is None

    async def test_breaker_disabled_pending_job_faults_without_images(self) -> None:
        """A job already queued for post-processing when the breaker flips is not left pending."""
        process_manager = make_testable_process_manager()
        process_manager._state.post_processing_disabled_by_breaker = True
        process_manager._state.post_processing_disabled_reason = "post-processing disabled by test"
        job_info = _make_pp_job_info()
        await process_manager._job_tracker.queue_for_post_processing(job_info)

        await process_manager.start_post_processing()

        assert job_info not in process_manager._job_tracker.jobs_pending_post_processing
        assert job_info in process_manager._job_tracker.jobs_pending_submit
        assert job_info.state == GENERATION_STATE.faulted
        assert job_info.job_image_results is None

    async def test_breaker_trip_faults_pending_but_leaves_active_best_effort(self) -> None:
        """Already-popped PP work is bounded after the breaker flips.

        Pending jobs are faulted on the next orchestrator pass; an active lane job is left alone so it can
        finish best-effort. If it does not return, the post-processing orphan watchdog owns the bounded fault.
        """
        process_manager = make_testable_process_manager(
            post_processing_fault_threshold=1,
            post_processing_fault_window_seconds=1800,
        )
        pending_job = _make_pp_job_info()
        active_job = _make_pp_job_info()
        tracker = process_manager._job_tracker
        await tracker.queue_for_post_processing(pending_job)
        await tracker.queue_for_post_processing(active_job)
        await tracker.begin_post_processing(active_job)

        for _ in range(2):
            tracker.note_post_processing_overcommit_fault()
        process_manager._apply_post_processing_fault_breaker()
        await process_manager.start_post_processing()

        assert pending_job in tracker.jobs_pending_submit
        assert pending_job.state == GENERATION_STATE.faulted
        assert pending_job.job_image_results is None
        assert active_job in tracker.jobs_being_post_processed

    async def test_paused_lane_does_not_age_jobs_out(self) -> None:
        """During a deliberate whole-card pause the patience clock stays unarmed.

        The residency lifecycle restarts the lane when the card is released, so jobs wait for the real
        lane rather than being immediately faulted during the residency window.
        """
        process_manager = make_testable_process_manager()
        process_manager._process_lifecycle._post_process_gpu_paused = True
        job_info = _make_pp_job_info()
        await process_manager._job_tracker.queue_for_post_processing(job_info)

        await process_manager.start_post_processing()

        orchestrator = process_manager._post_process_orchestrator
        assert str(job_info.sdk_api_job_info.id_) not in orchestrator._deferrals
        assert job_info in process_manager._job_tracker.jobs_pending_post_processing

    async def test_chain_waits_while_sampling_holds_a_tight_card(self) -> None:
        """With sampling in progress and co-residency unaffordable, the chain is not dispatched.

        Co-running a chain against active sampling on a card that cannot hold both peaks silently
        demand-pages both sides for the whole overlap; the chain waits for the card instead. The
        patience record arms so a never-idle card still ages the job out to a no-image fault.
        """
        process_manager = make_testable_process_manager()
        lane = _make_lane_process()
        sampling_process = make_mock_process_info(
            3,
            model_name="AlbedoBase XL (SDXL)",
            state=HordeProcessState.INFERENCE_STARTING,
        )
        process_manager._process_map.clear()
        process_manager._process_map.update({7: lane, 3: sampling_process})
        orchestrator = process_manager._post_process_orchestrator
        orchestrator._sampling_coresidency_check = lambda reserve_mb: False

        sampling_job = make_job_pop_response(model="AlbedoBase XL (SDXL)")
        await track_popped_job_async(process_manager._job_tracker, sampling_job)
        await process_manager._job_tracker.mark_inference_started(sampling_job, device_index=None)

        job_info = _make_pp_job_info()
        await process_manager._job_tracker.queue_for_post_processing(job_info)

        await process_manager.start_post_processing()

        assert job_info in process_manager._job_tracker.jobs_pending_post_processing
        lane.pipe_connection.send.assert_not_called()
        assert str(job_info.sdk_api_job_info.id_) in orchestrator._deferrals

    async def test_chain_dispatches_while_in_progress_job_is_only_downloading_aux(self) -> None:
        """Aux downloads do not use the GPU, so pending post-processing drains before line-skip work."""
        process_manager = make_testable_process_manager()
        lane = _make_lane_process()
        aux_process = make_mock_process_info(
            3,
            model_name="AlbedoBase XL (SDXL)",
            state=HordeProcessState.DOWNLOADING_AUX_MODEL,
        )
        process_manager._process_map.clear()
        process_manager._process_map.update({7: lane, 3: aux_process})
        orchestrator = process_manager._post_process_orchestrator
        orchestrator._sampling_coresidency_check = lambda reserve_mb: False
        process_manager._state.wants_line_skip_candidate = True

        aux_job = make_job_pop_response(model="AlbedoBase XL (SDXL)")
        await track_popped_job_async(process_manager._job_tracker, aux_job)
        await process_manager._job_tracker.mark_inference_started(aux_job, device_index=None)

        job_info = _make_pp_job_info()
        await process_manager._job_tracker.queue_for_post_processing(job_info)

        await process_manager.start_post_processing()

        assert job_info in process_manager._job_tracker.jobs_being_post_processed
        assert str(job_info.sdk_api_job_info.id_) not in orchestrator._deferrals
        assert lane.pipe_connection.send.call_count == 1

    async def test_chain_dispatches_when_coresidency_affordable(self) -> None:
        """On a card that can hold both peaks, sampling in progress does not delay the chain."""
        process_manager = make_testable_process_manager()
        lane = _make_lane_process()
        process_manager._process_map.clear()
        process_manager._process_map.update({7: lane})
        orchestrator = process_manager._post_process_orchestrator
        orchestrator._sampling_coresidency_check = lambda reserve_mb: True

        sampling_job = make_job_pop_response(model="AlbedoBase XL (SDXL)")
        await track_popped_job_async(process_manager._job_tracker, sampling_job)
        await process_manager._job_tracker.mark_inference_started(sampling_job, device_index=None)

        job_info = _make_pp_job_info()
        await process_manager._job_tracker.queue_for_post_processing(job_info)

        await process_manager.start_post_processing()

        assert job_info in process_manager._job_tracker.jobs_being_post_processed

    async def test_successful_dispatch_moves_job_and_sends_operations(self) -> None:
        """A successful dispatch moves the job to being-post-processed and sends images plus operations."""
        process_manager = make_testable_process_manager()
        lane = _make_lane_process()
        process_manager._process_map.clear()
        process_manager._process_map.update({7: lane})

        job_info = _make_pp_job_info(["RealESRGAN_x4plus", "GFPGAN"])
        await process_manager._job_tracker.queue_for_post_processing(job_info)

        await process_manager.start_post_processing()

        assert job_info not in process_manager._job_tracker.jobs_pending_post_processing
        assert job_info in process_manager._job_tracker.jobs_being_post_processed

        sent = lane.pipe_connection.send.call_args.args[0]
        assert isinstance(sent, HordePostProcessControlMessage)
        assert sent.job_id == job_info.sdk_api_job_info.id_
        assert sent.images_bytes == [b"raw-image"]
        assert sent.post_processing == ["RealESRGAN_x4plus", "GFPGAN"]

    async def test_successful_dispatch_reserves_estimated_post_process_vram(self, monkeypatch: object) -> None:
        """Active post-processing is charged to the shared committed-VRAM ledger until its result arrives."""
        monkeypatch.setattr(  # type: ignore[attr-defined]
            post_process_orchestrator_module,
            "predict_job_post_processing_vram_mb",
            lambda *_args, **_kwargs: 1234.0,
        )
        process_manager = make_testable_process_manager()
        lane = _make_lane_process()
        process_manager._process_map.clear()
        process_manager._process_map.update({7: lane})

        job_info = _make_pp_job_info(["RealESRGAN_x4plus"])
        await process_manager._job_tracker.queue_for_post_processing(job_info)

        await process_manager.start_post_processing()

        assert process_manager._reserve_ledger.total_vram_mb() == 1234.0
        assert process_manager._reserve_ledger.total_vram_mb_excluding(POST_PROCESS_RESERVE_FLOW) == 0.0

    async def test_dispatch_defers_and_reclaims_when_post_process_peak_would_overcommit(
        self,
        monkeypatch: object,
    ) -> None:
        """A low-free-VRAM post-processing start is deferred after asking the scheduler to reclaim idle VRAM."""
        monkeypatch.setattr(  # type: ignore[attr-defined]
            post_process_orchestrator_module,
            "predict_job_post_processing_vram_mb",
            lambda *_args, **_kwargs: 4000.0,
        )
        process_manager = make_testable_process_manager(enable_vram_budget=True, vram_reserve_mb=1500)
        lane = _make_lane_process()
        lane.total_vram_mb = 16000
        lane.vram_usage_mb = 15500
        idle_inference = make_mock_process_info(
            3,
            model_name="cold-model",
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        idle_inference.total_vram_mb = 16000
        idle_inference.vram_usage_mb = 4000
        process_manager._process_map.clear()
        process_manager._process_map.update({7: lane, 3: idle_inference})

        job_info = _make_pp_job_info(["RealESRGAN_x4plus"])
        await process_manager._job_tracker.queue_for_post_processing(job_info)

        await process_manager.start_post_processing()

        assert job_info in process_manager._job_tracker.jobs_pending_post_processing
        assert job_info not in process_manager._job_tracker.jobs_being_post_processed
        assert process_manager._reserve_ledger.total_vram_mb() == 0.0
        assert idle_inference.last_control_flag is not None
        assert lane.pipe_connection.send.call_count == 0

    async def test_send_failure_on_live_lane_flags_replacement(self) -> None:
        """A failed send to a live, loaded lane process arms the lane-replacement state machine."""
        process_manager = make_testable_process_manager()
        lane = make_mock_process_info(
            7,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.POST_PROCESS,
            safe_send_returns=False,
        )
        process_manager._process_map.clear()
        process_manager._process_map.update({7: lane})

        job_info = _make_pp_job_info()
        await process_manager._job_tracker.queue_for_post_processing(job_info)

        await process_manager.start_post_processing()

        assert process_manager._process_lifecycle.post_process_processes_should_be_replaced
        assert job_info in process_manager._job_tracker.jobs_pending_post_processing


class TestPostProcessResultHandling:
    """The dispatcher's handling of lane results."""

    async def _job_being_post_processed(self, process_manager: object) -> HordeJobInfo:
        job_info = _make_pp_job_info()
        tracker = process_manager._job_tracker  # type: ignore[attr-defined]
        await tracker.queue_for_post_processing(job_info)
        await tracker.begin_post_processing(job_info)
        return job_info

    async def test_ok_result_adopts_images_and_moves_to_safety(self) -> None:
        """A successful result replaces the raw images and queues the job for safety."""
        process_manager = make_testable_process_manager()
        job_info = await self._job_being_post_processed(process_manager)

        processed = HordeImageResult(image_bytes=b"post-processed")
        message = HordePostProcessResultMessage(
            process_id=7,
            process_launch_identifier=0,
            info="done",
            time_elapsed=1.0,
            job_id=job_info.sdk_api_job_info.id_,
            job_image_results=[processed],
            state=GENERATION_STATE.ok,
        )

        await process_manager._message_dispatcher._handle_post_process_result(message)

        assert job_info in process_manager._job_tracker.jobs_pending_safety_check
        assert job_info.job_image_results == [processed]

    async def test_result_releases_post_process_vram_reserve(self) -> None:
        """The active post-processing reserve is released when the lane returns a result."""
        process_manager = make_testable_process_manager()
        job_info = await self._job_being_post_processed(process_manager)
        process_manager._reserve_ledger.set(
            POST_PROCESS_RESERVE_FLOW,
            str(job_info.sdk_api_job_info.id_),
            vram_mb=1234.0,
        )

        processed = HordeImageResult(image_bytes=b"post-processed")
        message = HordePostProcessResultMessage(
            process_id=7,
            process_launch_identifier=0,
            info="done",
            time_elapsed=1.0,
            job_id=job_info.sdk_api_job_info.id_,
            job_image_results=[processed],
            state=GENERATION_STATE.ok,
        )

        await process_manager._message_dispatcher._handle_post_process_result(message)

        assert process_manager._reserve_ledger.total_vram_mb() == 0.0

    async def test_faulted_result_reports_no_image_fault(self) -> None:
        """A faulted result clears images, feeds the breaker window, and reaches submit as a fault."""
        process_manager = make_testable_process_manager()
        job_info = await self._job_being_post_processed(process_manager)

        message = HordePostProcessResultMessage(
            process_id=7,
            process_launch_identifier=0,
            info="post-processing failed",
            time_elapsed=1.0,
            job_id=job_info.sdk_api_job_info.id_,
            job_image_results=None,
            state=GENERATION_STATE.faulted,
        )

        await process_manager._message_dispatcher._handle_post_process_result(message)

        assert job_info in process_manager._job_tracker.jobs_pending_submit
        assert job_info.state == GENERATION_STATE.faulted
        assert job_info.job_image_results is None
        assert process_manager._job_tracker.count_recent_post_processing_faults(60.0) == 1


class TestPostProcessOrphanRecovery:
    """The watchdog for jobs whose lane result will never return."""

    async def test_orphan_is_requeued_after_grace(self) -> None:
        """A job stuck being post-processed past the grace is requeued for a fresh attempt."""
        process_manager = make_testable_process_manager()
        coordinator = process_manager._recovery_coordinator
        job_info = _make_pp_job_info()
        tracker = process_manager._job_tracker
        await tracker.queue_for_post_processing(job_info)
        await tracker.begin_post_processing(job_info)

        job_id = job_info.sdk_api_job_info.id_
        process_manager._reserve_ledger.set(POST_PROCESS_RESERVE_FLOW, str(job_id), vram_mb=1234.0)
        coordinator.orphan_post_process_since[job_id] = time.time() - (
            coordinator.ORPHAN_POST_PROCESS_GRACE_SECONDS + 1
        )

        await coordinator.reconcile_orphaned_post_process_jobs()

        assert job_info in tracker.jobs_pending_post_processing
        assert coordinator.post_process_requeue_count[job_id] == 1
        assert process_manager._reserve_ledger.total_vram_mb() == 0.0

    async def test_orphan_faults_without_images_after_budget(self) -> None:
        """Once the requeue budget is spent, the job is reported faulted without images."""
        process_manager = make_testable_process_manager()
        coordinator = process_manager._recovery_coordinator
        job_info = _make_pp_job_info()
        tracker = process_manager._job_tracker
        await tracker.queue_for_post_processing(job_info)
        await tracker.begin_post_processing(job_info)

        job_id = job_info.sdk_api_job_info.id_
        process_manager._reserve_ledger.set(POST_PROCESS_RESERVE_FLOW, str(job_id), vram_mb=1234.0)
        coordinator.orphan_post_process_since[job_id] = time.time() - (
            coordinator.ORPHAN_POST_PROCESS_GRACE_SECONDS + 1
        )
        coordinator.post_process_requeue_count[job_id] = coordinator.POST_PROCESS_REQUEUE_MAX

        await coordinator.reconcile_orphaned_post_process_jobs()

        assert job_info in tracker.jobs_pending_submit
        assert job_info.state == GENERATION_STATE.faulted
        assert job_info.job_image_results is None
        assert tracker.count_recent_post_processing_faults(60.0) == 1
        assert process_manager._reserve_ledger.total_vram_mb() == 0.0


class TestLaneWatchdogReap:
    """A lane process silent mid-job is replaced and feeds the post-processing breaker."""

    def test_silent_lane_process_is_flagged_and_feeds_breaker(self) -> None:
        """The stuck-state check on a POST_PROCESS process arms replacement and counts the fault."""
        process_manager = make_testable_process_manager()
        lane = _make_lane_process()
        lane.last_process_state = HordeProcessState.POST_PROCESSING
        silent_for = 120.0
        lane.last_received_timestamp = time.time() - silent_for
        lane.last_heartbeat_timestamp = time.time() - silent_for
        process_manager._process_map.clear()
        process_manager._process_map.update({7: lane})

        replaced = process_manager._process_lifecycle._check_and_replace_process(
            lane,
            60.0,
            HordeProcessState.POST_PROCESSING,
            "seems to be stuck post processing",
        )

        assert replaced
        assert process_manager._process_lifecycle.post_process_processes_should_be_replaced
        assert process_manager._job_tracker.count_recent_post_processing_faults(60.0) == 1


class TestChainParity:
    """The chain context mirrors the job's queue-stage walk."""

    async def test_full_post_processing_walk_completes_chain(self) -> None:
        """A popped job with post-processing walks every chain stage to COMPLETED at finalize."""
        process_manager = make_testable_process_manager()
        tracker = process_manager._job_tracker

        job = make_job_pop_response(post_processing=["RealESRGAN_x4plus"])
        await track_popped_job_async(tracker, job)
        tracked = tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.chain_context is not None
        assert set(tracked.chain_context.snapshot()) == {"generate", "post_process", "safety_check", "submit"}

        await tracker.mark_inference_started(job)
        assert tracked.chain_context.snapshot()["generate"] == CHAIN_NODE_STATE.EXECUTING

        job_info = HordeJobInfo(
            sdk_api_job_info=job,
            job_image_results=[HordeImageResult(image_bytes=b"raw")],
            state=GENERATION_STATE.ok,
            censored=False,
            time_popped=time.time(),
        )
        await tracker.release_in_progress(job)
        await tracker.queue_for_post_processing(job_info)
        snapshot = tracked.chain_context.snapshot()
        assert snapshot["generate"] == CHAIN_NODE_STATE.COMPLETED
        assert snapshot["post_process"] == CHAIN_NODE_STATE.READY

        await tracker.begin_post_processing(job_info)
        assert tracked.chain_context.snapshot()["post_process"] == CHAIN_NODE_STATE.EXECUTING

        taken = await tracker.take_being_post_processed(job.id_)
        assert taken is job_info
        await tracker.queue_for_safety_post_processed(job_info)
        snapshot = tracked.chain_context.snapshot()
        assert snapshot["post_process"] == CHAIN_NODE_STATE.COMPLETED

        await tracker.begin_safety_check(job_info)
        assert tracked.chain_context.snapshot()["safety_check"] == CHAIN_NODE_STATE.EXECUTING

        await tracker.take_being_safety_checked(job.id_)
        await tracker.queue_for_submit(job_info)
        assert tracked.chain_context.snapshot()["safety_check"] == CHAIN_NODE_STATE.COMPLETED

        chain_context = tracked.chain_context
        await tracker.finalize_submitted(job_info)
        assert all(state == CHAIN_NODE_STATE.COMPLETED for state in chain_context.snapshot().values())
        assert chain_context.is_finished
        assert not chain_context.has_failed

    async def test_no_post_processing_flow_skips_pp_stage(self) -> None:
        """A job without post-processing gets a chain with no post-process node."""
        process_manager = make_testable_process_manager()
        tracker = process_manager._job_tracker

        job = make_job_pop_response()
        await track_popped_job_async(tracker, job)
        tracked = tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.chain_context is not None
        assert set(tracked.chain_context.snapshot()) == {"generate", "safety_check", "submit"}

    async def test_terminal_fault_fails_chain(self) -> None:
        """A job that finalizes faulted marks the executing chain stage failed and skips downstream."""
        process_manager = make_testable_process_manager()
        tracker = process_manager._job_tracker

        job = make_job_pop_response()
        await track_popped_job_async(tracker, job)
        tracked = tracker.get_tracked_job(job.id_)
        assert tracked is not None
        assert tracked.chain_context is not None
        chain_context = tracked.chain_context

        await tracker.mark_inference_started(job)

        job_info = HordeJobInfo(
            sdk_api_job_info=job,
            job_image_results=None,
            state=GENERATION_STATE.faulted,
            censored=False,
            time_popped=time.time(),
        )
        await tracker.release_in_progress(job)
        await tracker.queue_for_submit(job_info)
        await tracker.finalize_submitted(job_info)

        snapshot = chain_context.snapshot()
        assert snapshot["generate"] == CHAIN_NODE_STATE.FAILED
        assert snapshot["safety_check"] == CHAIN_NODE_STATE.SKIPPED
        assert snapshot["submit"] == CHAIN_NODE_STATE.SKIPPED
        assert chain_context.has_failed
