"""Regression tests for the safety-orphan wedge and the safety-before-submit invariant.

The safety orchestrator only ever acts on ``jobs_pending_safety_check``. A job already handed to the
safety process that never returns a verdict, because the safety process was replaced or a result message
was dropped, is invisible to the orchestrator: nothing re-checks it, nothing clears it from
``jobs_being_safety_checked``. Stranded jobs pin pipeline slots; with the queue then unable to drain the
worker can latch a structural deadlock, soft-reset its pools, and give up on the pending inference
backlog: an escalating chain that ends in horde-forced maintenance.

These tests cover the watchdog fix and the overriding invariant: an image is **never** submitted unless
the safety process actually returned a verdict for it.

* ``TestSafetyOrphanReconciler`` - the watchdog recovers a job stranded in SAFETY_CHECKING (requeues for
  a fresh check), and escalates a job the safety pipeline cannot check to a no-image fault plus a
  soft-pause, instead of stranding it forever.
* ``TestNoUncheckedSubmit`` - the submit boundary faults (never uploads) a job carrying images that did
  not pass safety, and the safety-result handler is the sole writer of the ``safety_evaluated`` gate.
* ``TestSafetyResultInvariant`` - a safety *evaluation failure* drops the images and faults the job
  rather than letting the original, uncleared image reach the submit path.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

from horde_sdk.ai_horde_api import GENERATION_STATE

from horde_worker_regen.process_management.ipc.messages import (
    HordeImageResult,
    HordeProcessState,
    HordeSafetyResultMessage,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobStage
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    move_job_to_being_safety_checked_async,
)


def _safety_job_info(*, model: str = "stable_diffusion") -> HordeJobInfo:
    """A completed job carrying one image, shaped for the safety/submit stages (real, not a Mock)."""
    pop = make_job_pop_response(model=model, r2_upload="https://r2.example/upload")
    return HordeJobInfo(
        sdk_api_job_info=pop,
        job_image_results=[HordeImageResult(image_bytes=b"imgdata")],
        state=None,
        time_popped=time.time(),
    )


def _add_idle_safety_process(pm: object, process_id: int = 10) -> None:
    """Attach a ready (idle) safety process, so the safety pool reads as available but not busy.

    Represents the case where a safety process is up but idle: no verdict was returned for jobs already
    sent to it.
    """
    safety_proc = make_mock_process_info(
        process_id,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.SAFETY,
    )
    pm._process_map[process_id] = safety_proc  # type: ignore[attr-defined]


async def _strand_in_safety_checking(pm: object, job_info: HordeJobInfo) -> None:
    """Register a job and move it into SAFETY_CHECKING (sent to safety, awaiting a verdict)."""
    await move_job_to_being_safety_checked_async(pm._job_tracker, job_info)  # type: ignore[attr-defined]


class TestSafetyOrphanReconciler:
    """The watchdog that recovers jobs stranded in SAFETY_CHECKING."""

    async def test_orchestrator_alone_never_recovers_a_stranded_safety_job(self) -> None:
        """Characterises the root-cause gap: the orchestrator only acts on PENDING_SAFETY_CHECK.

        A job already in SAFETY_CHECKING whose verdict was lost is invisible to ``start_evaluate_safety``
        (it early-returns when nothing is *pending*), so without the watchdog it stays stranded forever.
        """
        pm = make_testable_process_manager()
        _add_idle_safety_process(pm)
        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)

        # Pumping the orchestrator does nothing: there is no pending safety work, only stranded work.
        await pm.start_evaluate_safety()

        assert job_info in pm._job_tracker.jobs_being_safety_checked
        assert pm._job_tracker.jobs_pending_safety_check == ()

    async def test_stranded_safety_job_requeued_after_grace(self) -> None:
        """Past the grace window, a stranded job is sent back for a fresh safety check (images preserved)."""
        pm = make_testable_process_manager()
        _add_idle_safety_process(pm)
        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None

        # Within grace: only watched, never disturbed.
        await pm._recovery_coordinator.reconcile_orphaned_safety_jobs()
        assert pm._job_tracker.get_stage(job_id) == JobStage.SAFETY_CHECKING

        # Backdate the grace clock past the window: now it is requeued for a fresh check, not stranded.
        pm._recovery_coordinator.orphan_safety_since[job_id] = time.time() - (
            pm._recovery_coordinator.ORPHAN_SAFETY_GRACE_SECONDS + 1
        )
        await pm._recovery_coordinator.reconcile_orphaned_safety_jobs()

        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SAFETY_CHECK
        assert pm._recovery_coordinator.safety_requeue_count[job_id] == 1
        # Its images are kept so they are actually re-checked, never submitted unchecked.
        assert job_info.job_image_results is not None
        assert job_info.safety_evaluated is False
        # A ready safety process is up, so the healthy pool is not needlessly torn down.
        assert pm._process_lifecycle.safety_processes_should_be_replaced is False

    async def test_multiple_stranded_jobs_all_recovered_at_once(self) -> None:
        """Multiple jobs stranded in SAFETY_CHECKING are all recovered in a single reconcile pass.

        Every stranded job is requeued once its grace elapses, so a full safety-stage backlog drains
        rather than leaking forever.
        """
        pm = make_testable_process_manager()
        _add_idle_safety_process(pm)

        job_infos = [_safety_job_info() for _ in range(5)]
        for job_info in job_infos:
            await _strand_in_safety_checking(pm, job_info)
        assert len(pm._job_tracker.jobs_being_safety_checked) == 5

        for job_info in job_infos:
            job_id = job_info.sdk_api_job_info.id_
            assert job_id is not None
            pm._recovery_coordinator.orphan_safety_since[job_id] = time.time() - (
                pm._recovery_coordinator.ORPHAN_SAFETY_GRACE_SECONDS + 1
            )

        await pm._recovery_coordinator.reconcile_orphaned_safety_jobs()

        assert pm._job_tracker.jobs_being_safety_checked == ()
        assert len(pm._job_tracker.jobs_pending_safety_check) == 5

    async def test_repeated_orphan_escalates_to_no_image_fault_and_soft_pause(self) -> None:
        """A job the pipeline keeps failing to check is faulted with NO image, and pops soft-pause.

        This is the user's hard rule: rather than loop forever (or risk submitting unchecked), the job is
        dropped (images cleared, reported faulted so the horde reissues it) and the worker stops popping.
        """
        pm = make_testable_process_manager()
        _add_idle_safety_process(pm)
        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None

        # This job has already exhausted its re-check attempts.
        pm._recovery_coordinator.safety_requeue_count[job_id] = pm._recovery_coordinator.SAFETY_REQUEUE_MAX
        pm._recovery_coordinator.orphan_safety_since[job_id] = time.time() - (
            pm._recovery_coordinator.ORPHAN_SAFETY_GRACE_SECONDS + 1
        )

        assert pm._state.self_throttle_paused is False
        await pm._recovery_coordinator.reconcile_orphaned_safety_jobs()

        # Faulted with no image (the image is never submitted unchecked) and reported for reissue.
        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SUBMIT
        assert job_info.job_image_results is None
        assert job_info.state == GENERATION_STATE.faulted
        # The worker soft-paused popping so it stops taking on work it cannot safety-check.
        assert pm._state.self_throttle_paused is True
        assert pm._state.self_throttle_paused_until > time.time()

    async def test_unrecoverable_safety_pool_faults_with_no_image(self) -> None:
        """When the safety pool is crash-looping, a stranded job is faulted (no image) immediately."""
        pm = make_testable_process_manager()
        # No ready safety process, and a recent rebuild storm: the pool is unrecoverable.
        for pid in [p.process_id for p in pm._process_map.values() if p.process_type == HordeProcessType.SAFETY]:
            del pm._process_map[pid]
        pm._process_lifecycle._safety_recovery_history = [time.time()] * 100
        assert pm._recovery_coordinator.is_safety_pool_unrecoverable() is True

        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None
        pm._recovery_coordinator.orphan_safety_since[job_id] = time.time() - (
            pm._recovery_coordinator.ORPHAN_SAFETY_GRACE_SECONDS + 1
        )

        await pm._recovery_coordinator.reconcile_orphaned_safety_jobs()

        # Even on its first orphan tick: no re-check loop, straight to a no-image fault + soft pause.
        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SUBMIT
        assert job_info.job_image_results is None
        assert pm._state.self_throttle_paused is True


class TestNoUncheckedSubmit:
    """The overriding invariant: an image is never submitted without a real safety verdict."""

    async def test_submit_faults_a_job_with_images_but_no_verdict(self) -> None:
        """A PENDING_SUBMIT job carrying images but ``safety_evaluated=False`` is faulted, not uploaded."""
        pm = make_testable_process_manager()
        submitter = pm._job_submitter
        submitter._dry_run_skip_api = True  # never touch the real API; the upload path must not be reached

        job_info = _safety_job_info()
        assert job_info.safety_evaluated is False  # never passed safety
        await pm._job_tracker.queue_for_submit(job_info)

        await submitter.api_submit_job()

        # The guard dropped the images and reported a fault rather than uploading an unchecked image.
        assert job_info.job_image_results is None
        assert job_info.state == GENERATION_STATE.faulted

    async def test_submit_uploads_a_job_only_after_a_verdict(self) -> None:
        """A job that *did* pass safety (``safety_evaluated=True``) is allowed through the guard."""
        pm = make_testable_process_manager()
        submitter = pm._job_submitter
        submitter._dry_run_skip_api = True

        job_info = _safety_job_info()
        job_info.safety_evaluated = True
        job_info.censored = False
        job_info.state = GENERATION_STATE.ok
        await pm._job_tracker.queue_for_submit(job_info)

        await submitter.api_submit_job()

        # Not punted: it kept its images and was not coerced to faulted by the safety guard.
        assert job_info.job_image_results is not None
        assert job_info.state == GENERATION_STATE.ok

    def test_safety_evaluated_is_the_checked_for_safety_gate(self) -> None:
        """``is_job_checked_for_safety`` reflects the explicit flag, not the nullable censored sentinel."""
        job_info = _safety_job_info()
        assert job_info.is_job_checked_for_safety is False
        # Setting only the outcome sentinel must not, by itself, claim the job was checked.
        job_info.censored = False
        assert job_info.is_job_checked_for_safety is False
        job_info.safety_evaluated = True
        assert job_info.is_job_checked_for_safety is True


def _safety_eval(*, failed: bool = False, is_nsfw: bool = False, is_csam: bool = False) -> Mock:
    """A single per-image safety evaluation, as the safety process reports it."""
    evaluation = Mock()
    evaluation.failed = failed
    evaluation.is_nsfw = is_nsfw
    evaluation.is_csam = is_csam
    evaluation.replacement_image_bytes = None
    return evaluation


class TestSafetyResultInvariant:
    """The safety-result handler is the sole writer of the submit gate, and never lets unchecked through."""

    async def test_clean_verdict_marks_evaluated_and_queues_submit(self) -> None:
        """A clean verdict marks the job safety-evaluated and moves it to submit."""
        pm = make_testable_process_manager()
        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)

        await pm._message_dispatcher._handle_safety_result(
            Mock(job_id=job_info.sdk_api_job_info.id_, safety_evaluations=[_safety_eval()], time_elapsed=1.0),
        )

        assert job_info.safety_evaluated is True
        assert job_info in pm._job_tracker.jobs_pending_submit

    async def test_evaluation_failure_drops_images_and_does_not_mark_evaluated(self) -> None:
        """If safety could not evaluate an image, the images are dropped and the job is faulted.

        The original, uncleared image must never survive to the submit path; ``safety_evaluated`` stays
        False to reflect that no clean verdict was obtained.
        """
        pm = make_testable_process_manager()
        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)

        await pm._message_dispatcher._handle_safety_result(
            Mock(
                job_id=job_info.sdk_api_job_info.id_,
                safety_evaluations=[_safety_eval(failed=True)],
                time_elapsed=1.0,
            ),
        )

        assert job_info.safety_evaluated is False
        assert job_info.job_image_results is None
        assert job_info.state == GENERATION_STATE.faulted
        assert job_info in pm._job_tracker.jobs_pending_submit


def _deliver_one_message(pm: object, message: object) -> None:
    """Feed a single message through the dispatcher's real receive loop (mock queue: one item, then empty)."""
    dispatcher = pm._message_dispatcher  # type: ignore[attr-defined]
    dispatcher._process_message_queue.empty.side_effect = [False, True]
    dispatcher._process_message_queue.get.return_value = message


class TestRetiredSafetyLaunchStrandsInFlightJob:
    """The trigger for the orphan wedge: replacing a safety process discards its in-flight verdict.

    Whole-card residency moves the safety process off the GPU while a card-filling model holds the device,
    then replaces it (retires the launch, starts a fresh one) when the residency lifts. A verdict produced
    by the retired launch for a job that was mid-check is dropped at the launch-identifier guard. That guard
    keeps a stale message from crashing the control loop, but on its own it leaves the job sitting in
    SAFETY_CHECKING with a verdict that will never be re-delivered: only the orphan watchdog rescues it, and
    only after its multi-second grace. When several such replacements land in quick succession the stranded
    jobs accumulate faster than the watchdog clears them and the pipeline wedges into a soft reset.

    The contract: dropping a result from a retired launch flags the job as having a verdict that is known
    lost (positive evidence, not the watchdog's timeout suspicion), so the next reconcile tick re-checks it
    at once rather than only after the orphan grace elapses. The bounded requeue/escalation bookkeeping is
    unchanged, so a job whose re-checks keep failing is still faulted rather than looping forever.
    """

    async def _strand_then_drop_retired_verdict(self, pm: object) -> HordeJobInfo:
        """Strand a job in SAFETY_CHECKING, retire its safety launch, and deliver+drop the late verdict."""
        retired_launch_id = 12
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        safety_proc.process_launch_identifier = retired_launch_id
        pm._process_map[safety_proc.process_id] = safety_proc  # type: ignore[attr-defined]

        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None
        assert pm._job_tracker.get_stage(job_id) == JobStage.SAFETY_CHECKING  # type: ignore[attr-defined]

        # Whole-card residency lifts and the safety process is restored: its launch is retired.
        pm._process_map.retire_process(  # type: ignore[attr-defined]
            safety_proc,
            "whole-card residency complete: restoring safety to GPU",
        )

        # The verdict the retired launch had in flight for this job now arrives and is dropped.
        verdict = HordeSafetyResultMessage(
            process_id=safety_proc.process_id,
            process_launch_identifier=retired_launch_id,
            info="late verdict from retired safety launch",
            job_id=job_id,
            safety_evaluations=[],
        )
        _deliver_one_message(pm, verdict)
        await pm._message_dispatcher.receive_and_handle_process_messages()  # type: ignore[attr-defined]
        return job_info

    async def test_dropped_retired_safety_verdict_requeues_without_waiting_out_the_grace(self) -> None:
        """A dropped retired-launch verdict re-checks its job on the next tick, not after the orphan grace."""
        pm = make_testable_process_manager()
        job_info = await self._strand_then_drop_retired_verdict(pm)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None

        # No grace backdating by the test: one reconcile tick after the drop must already recover the job,
        # because the dropped verdict is positive evidence the verdict is lost (not merely late).
        await pm._recovery_coordinator.reconcile_orphaned_safety_jobs()

        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SAFETY_CHECK
        assert job_info not in pm._job_tracker.jobs_being_safety_checked
        assert pm._recovery_coordinator.safety_requeue_count[job_id] == 1
        # Its images are preserved so the fresh check has something to evaluate; never submitted unchecked.
        assert job_info.job_image_results is not None
        assert job_info.safety_evaluated is False

    async def test_repeated_dropped_verdicts_still_bounded_to_a_no_image_fault(self) -> None:
        """If re-checks keep losing the verdict, the job is faulted (no image), not requeued forever."""
        pm = make_testable_process_manager()
        job_info = await self._strand_then_drop_retired_verdict(pm)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None

        # Pre-charge the requeue count to its ceiling: this drop is the one that must escalate, not loop.
        pm._recovery_coordinator.safety_requeue_count[job_id] = pm._recovery_coordinator.SAFETY_REQUEUE_MAX

        await pm._recovery_coordinator.reconcile_orphaned_safety_jobs()

        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SUBMIT
        assert job_info.job_image_results is None
        assert job_info.state == GENERATION_STATE.faulted
        assert pm._state.self_throttle_paused is True
