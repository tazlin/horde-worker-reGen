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
