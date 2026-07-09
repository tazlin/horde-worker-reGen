"""A requeued post-processing job must survive lane replacement without vanishing.

The orphan watchdog requeues a job whose lane result never arrived; lane replacement (teardown of the
old process, a starting successor, retirement of the old launch) then unfolds around the requeued job.
Every step of that choreography must leave the job visible: pending, being post-processed, or terminal
(post-processed or raw-fallback submitted). A job silently leaving all of those states forfeits its
finished inference with no fault, no ledger event, and no log line, which is the one outcome the
recovery machinery exists to prevent.
"""

from __future__ import annotations

import time

from horde_sdk.ai_horde_api import GENERATION_STATE

from horde_worker_regen.process_management.ipc.messages import (
    HordeImageResult,
    HordePostProcessControlMessage,
    HordePostProcessResultMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
)

_OLD_LANE_ID = 7
_NEW_LANE_ID = 8


def _make_pp_job_info() -> HordeJobInfo:
    job = make_job_pop_response(post_processing=["RealESRGAN_x4plus", "GFPGAN"])
    return HordeJobInfo(
        sdk_api_job_info=job,
        job_image_results=[HordeImageResult(image_bytes=b"raw-image")],
        state=GENERATION_STATE.ok,
        censored=False,
        time_popped=time.time(),
    )


def _job_locations(process_manager: object, job_info: HordeJobInfo) -> set[str]:
    """Return every queue-visible location the job currently occupies."""
    tracker = process_manager._job_tracker  # type: ignore[attr-defined]
    locations: set[str] = set()
    if job_info in tracker.jobs_pending_post_processing:
        locations.add("pending_post_processing")
    if job_info in tracker.jobs_being_post_processed:
        locations.add("being_post_processed")
    if job_info in tracker.jobs_pending_safety_check:
        locations.add("pending_safety_check")
    if job_info in tracker.jobs_pending_submit:
        locations.add("pending_submit")
    return locations


def _deliver_one_message(process_manager: object, message: object) -> None:
    """Feed one child message through the dispatcher's normal queue-drain path."""
    dispatcher = process_manager._message_dispatcher  # type: ignore[attr-defined]
    dispatcher._process_message_queue.empty.side_effect = [False, True]
    dispatcher._process_message_queue.get.return_value = message


class TestRequeueSurvivesLaneReplacement:
    """The full orphan-requeue-then-replace choreography, asserted for visibility at every step."""

    async def test_job_remains_visible_through_replacement_and_completes(self) -> None:
        """Dispatch, silence, requeue, lane teardown, successor bring-up: the job stays accounted for."""
        process_manager = make_testable_process_manager()
        coordinator = process_manager._recovery_coordinator
        tracker = process_manager._job_tracker
        now = {"s": 1000.0}
        coordinator._clock = lambda: now["s"]

        old_lane = make_mock_process_info(
            _OLD_LANE_ID,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.POST_PROCESS,
        )
        process_manager._process_map.clear()
        process_manager._process_map.update({_OLD_LANE_ID: old_lane})

        job_info = _make_pp_job_info()
        await tracker.queue_for_post_processing(job_info)
        await process_manager.start_post_processing()
        assert _job_locations(process_manager, job_info) == {"being_post_processed"}

        # The lane goes silent past the orphan grace; the watchdog requeues the job for a fresh attempt.
        # The first reconcile pass records when the orphan was first seen; the post-grace pass acts on it.
        await coordinator.reconcile_orphaned_post_process_jobs()
        now["s"] += coordinator.ORPHAN_POST_PROCESS_GRACE_SECONDS + 1
        await coordinator.reconcile_orphaned_post_process_jobs()
        assert _job_locations(process_manager, job_info) == {"pending_post_processing"}

        # Lane replacement tears the old process down. The old lane's last-job reference still names the
        # requeued job; teardown must not fault or detach a job that is no longer in flight on it.
        process_manager._process_map.clear()
        await coordinator.reconcile_orphaned_post_process_jobs()
        assert _job_locations(process_manager, job_info) == {"pending_post_processing"}

        # A late result from the retired launch surfaces as a known-lost marker (the dispatcher records
        # the drop when it ignores the stale message). The reconcile pass that drains the marker must
        # not disturb a job that has already been requeued to pending.
        process_manager._message_dispatcher._post_process_results_known_lost.add(job_info.sdk_api_job_info.id_)
        await coordinator.reconcile_orphaned_post_process_jobs()
        assert _job_locations(process_manager, job_info) == {"pending_post_processing"}

        # The successor lane appears, first still starting (not dispatchable), then ready.
        new_lane = make_mock_process_info(
            _NEW_LANE_ID,
            model_name=None,
            state=HordeProcessState.PROCESS_STARTING,
            process_type=HordeProcessType.POST_PROCESS,
        )
        process_manager._process_map.update({_NEW_LANE_ID: new_lane})
        await process_manager.start_post_processing()
        assert _job_locations(process_manager, job_info) == {"pending_post_processing"}

        process_manager._process_map.on_process_state_change(_NEW_LANE_ID, HordeProcessState.WAITING_FOR_JOB)
        await process_manager.start_post_processing()
        assert _job_locations(process_manager, job_info) == {"being_post_processed"}

        sent = new_lane.pipe_connection.send.call_args.args[0]
        assert isinstance(sent, HordePostProcessControlMessage)
        assert sent.job_id == job_info.sdk_api_job_info.id_

        # The successor completes the pass; the job reaches the safety stage with its processed images.
        result = HordePostProcessResultMessage(
            process_id=_NEW_LANE_ID,
            process_launch_identifier=0,
            info="done",
            time_elapsed=1.0,
            job_id=job_info.sdk_api_job_info.id_,
            job_image_results=[HordeImageResult(image_bytes=b"post-processed")],
            state=GENERATION_STATE.ok,
        )
        await process_manager._message_dispatcher._handle_post_process_result(result)
        assert _job_locations(process_manager, job_info) == {"pending_safety_check"}

    async def test_lifecycle_known_lost_marker_does_not_drop_a_pending_job(self) -> None:
        """The lane-teardown known-lost path must leave a requeued (pending) job untouched.

        Teardown marks jobs it believes are in flight as known-lost so the watchdog skips the orphan
        grace. When the job was already requeued to pending before teardown captured its marker, the
        drain of that marker must not detach, fault, or requeue-count the pending job.
        """
        process_manager = make_testable_process_manager()
        coordinator = process_manager._recovery_coordinator
        tracker = process_manager._job_tracker
        now = {"s": 1000.0}
        coordinator._clock = lambda: now["s"]

        lane = make_mock_process_info(
            _OLD_LANE_ID,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.POST_PROCESS,
        )
        process_manager._process_map.clear()
        process_manager._process_map.update({_OLD_LANE_ID: lane})

        job_info = _make_pp_job_info()
        await tracker.queue_for_post_processing(job_info)
        await process_manager.start_post_processing()

        await coordinator.reconcile_orphaned_post_process_jobs()
        now["s"] += coordinator.ORPHAN_POST_PROCESS_GRACE_SECONDS + 1
        await coordinator.reconcile_orphaned_post_process_jobs()
        assert _job_locations(process_manager, job_info) == {"pending_post_processing"}

        # Teardown captures the job in its known-lost set (stale view of in-flight work), then the
        # coordinator drains it on the next pass.
        process_manager._process_lifecycle._post_process_results_known_lost.add(job_info.sdk_api_job_info.id_)
        await coordinator.reconcile_orphaned_post_process_jobs()
        await coordinator.reconcile_orphaned_post_process_jobs()

        assert _job_locations(process_manager, job_info) == {"pending_post_processing"}
        assert coordinator.post_process_requeue_count.get(job_info.sdk_api_job_info.id_) == 1


class TestRetiredPostProcessLaunchResult:
    """A completed post-processing pass should not be discarded only because its launch was retired."""

    async def test_late_successful_result_from_replaced_lane_completes_in_flight_job(self) -> None:
        """A still-tracked post-processing job adopts the images its original lane produced."""
        process_manager = make_testable_process_manager()
        tracker = process_manager._job_tracker
        launch_id = 71
        lane = make_mock_process_info(
            _OLD_LANE_ID,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.POST_PROCESS,
        )
        lane.process_launch_identifier = launch_id
        process_manager._process_map.clear()
        process_manager._process_map.update({_OLD_LANE_ID: lane})

        job_info = _make_pp_job_info()
        await tracker.queue_for_post_processing(job_info)
        await process_manager.start_post_processing()
        assert _job_locations(process_manager, job_info) == {"being_post_processed"}

        process_manager._process_map.retire_process(lane, "post-processing lane replacement")

        processed = HordeImageResult(image_bytes=b"post-processed")
        result = HordePostProcessResultMessage(
            process_id=_OLD_LANE_ID,
            process_launch_identifier=launch_id,
            info="done",
            time_elapsed=1.0,
            job_id=job_info.sdk_api_job_info.id_,
            job_image_results=[processed],
            state=GENERATION_STATE.ok,
        )
        _deliver_one_message(process_manager, result)
        await process_manager._message_dispatcher.receive_and_handle_process_messages()

        assert _job_locations(process_manager, job_info) == {"pending_safety_check"}
        assert job_info.job_image_results == [processed]

    async def test_result_from_previous_attempt_does_not_override_current_attempt(self) -> None:
        """A stale successful result is ignored after the job has been requeued to a successor lane."""
        process_manager = make_testable_process_manager()
        tracker = process_manager._job_tracker
        old_launch_id = 71
        new_launch_id = 72
        old_lane = make_mock_process_info(
            _OLD_LANE_ID,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.POST_PROCESS,
        )
        old_lane.process_launch_identifier = old_launch_id
        process_manager._process_map.clear()
        process_manager._process_map.update({_OLD_LANE_ID: old_lane})

        job_info = _make_pp_job_info()
        await tracker.queue_for_post_processing(job_info)
        await process_manager.start_post_processing()
        assert _job_locations(process_manager, job_info) == {"being_post_processed"}

        process_manager._process_map.retire_process(old_lane, "post-processing lane replacement")
        await tracker.requeue_one_being_post_processed(job_info.sdk_api_job_info.id_)

        new_lane = make_mock_process_info(
            _NEW_LANE_ID,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.POST_PROCESS,
        )
        new_lane.process_launch_identifier = new_launch_id
        process_manager._process_map.update({_NEW_LANE_ID: new_lane})
        await process_manager.start_post_processing()
        assert _job_locations(process_manager, job_info) == {"being_post_processed"}

        old_result = HordePostProcessResultMessage(
            process_id=_OLD_LANE_ID,
            process_launch_identifier=old_launch_id,
            info="done",
            time_elapsed=1.0,
            job_id=job_info.sdk_api_job_info.id_,
            job_image_results=[HordeImageResult(image_bytes=b"old-post-processed")],
            state=GENERATION_STATE.ok,
        )
        _deliver_one_message(process_manager, old_result)
        await process_manager._message_dispatcher.receive_and_handle_process_messages()

        assert _job_locations(process_manager, job_info) == {"being_post_processed"}
        assert job_info.job_image_results == [HordeImageResult(image_bytes=b"raw-image")]

    async def test_faulted_result_from_replaced_lane_remains_known_lost(self) -> None:
        """A fault from a retired lane re-enters bounded recovery instead of terminally faulting the job."""
        process_manager = make_testable_process_manager()
        tracker = process_manager._job_tracker
        launch_id = 71
        lane = make_mock_process_info(
            _OLD_LANE_ID,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.POST_PROCESS,
        )
        lane.process_launch_identifier = launch_id
        process_manager._process_map.clear()
        process_manager._process_map.update({_OLD_LANE_ID: lane})

        job_info = _make_pp_job_info()
        await tracker.queue_for_post_processing(job_info)
        await process_manager.start_post_processing()
        process_manager._process_map.retire_process(lane, "post-processing lane replacement")

        result = HordePostProcessResultMessage(
            process_id=_OLD_LANE_ID,
            process_launch_identifier=launch_id,
            info="post-processing failed",
            time_elapsed=1.0,
            job_id=job_info.sdk_api_job_info.id_,
            job_image_results=None,
            state=GENERATION_STATE.faulted,
        )
        _deliver_one_message(process_manager, result)
        await process_manager._message_dispatcher.receive_and_handle_process_messages()

        assert _job_locations(process_manager, job_info) == {"being_post_processed"}
        assert process_manager._message_dispatcher.take_post_process_results_known_lost() == {
            job_info.sdk_api_job_info.id_,
        }
