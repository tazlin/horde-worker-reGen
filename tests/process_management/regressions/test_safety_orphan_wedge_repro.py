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
* ``TestRetiredSafetyLaunchVerdictStillApplies`` / ``TestRetiredSafetyLaunchDropsUnownedVerdicts`` - a
  verdict is valid for the job it checked whichever launch produced it, while a verdict for a job some
  later launch now owns stays dropped.
* ``TestPlacementMoveWaitsForTheCheck`` - a placement move ends the safety process after the check in flight
  returns its verdict, bounded by ``process_timeout``, so the orphan sweep never requeues that job.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

from horde_sdk.ai_horde_api import GENERATION_STATE

from horde_worker_regen.process_management.ipc.messages import (
    AlchemyFormSpec,
    HordeAlchemyResultMessage,
    HordeControlFlag,
    HordeImageResult,
    HordeProcessState,
    HordeSafetyEvaluation,
    HordeSafetyResultMessage,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobStage
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    move_job_to_being_safety_checked_async,
    queue_job_for_safety_async,
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


async def _strand_in_safety_checking(
    pm: object,
    job_info: HordeJobInfo,
    *,
    process_id: int | None = None,
    process_launch_identifier: int | None = None,
) -> None:
    """Register a job and move it into SAFETY_CHECKING (sent to safety, awaiting a verdict)."""
    await move_job_to_being_safety_checked_async(
        pm._job_tracker,  # type: ignore[attr-defined]
        job_info,
        process_id=process_id,
        process_launch_identifier=process_launch_identifier,
    )


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
        # The rebuilds followed children that died: the evidence that makes them a crash loop rather than
        # placement churn the parent asked for.
        pm._process_lifecycle._safety_child_failure_in_streak = True
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

    async def test_late_verdict_clears_requeue_bookkeeping_on_the_following_reconcile(self) -> None:
        """A verdict for a requeued job is accepted, and its watchdog attempt state is retired that tick."""
        pm = make_testable_process_manager()
        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None

        requeued = await pm._job_tracker.requeue_one_being_safety_checked(job_id)
        assert requeued is True
        pm._recovery_coordinator.safety_requeue_count[job_id] = 1
        pm._recovery_coordinator.orphan_safety_since[job_id] = time.time()

        await pm._message_dispatcher._handle_safety_result(
            Mock(
                job_id=job_id,
                safety_evaluations=[_safety_eval()],
                time_elapsed=14.0,
            ),
        )
        await pm._recovery_coordinator.reconcile_orphaned_safety_jobs()

        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SUBMIT
        assert job_info.safety_evaluated is True
        assert job_id not in pm._recovery_coordinator.orphan_safety_since
        assert job_id not in pm._recovery_coordinator.safety_requeue_count


class TestLateVerdictForRequeuedJob:
    """A verdict that arrives after the orphan watchdog requeued its job still resolves that job.

    A slow (not lost) check means the verdict lands while the job sits in PENDING_SAFETY_CHECK awaiting a
    duplicate re-check. Discarding it stranded finished work behind another full check on an already
    overloaded safety stage: every re-check's verdict was in turn late, so jobs cycled requeue -> duplicate
    check -> late verdict until they were faulted and their finished images thrown away.
    """

    async def test_late_verdict_resolves_a_requeued_job(self) -> None:
        """The verdict is applied, the job moves to submit, and the pending re-check is cancelled."""
        pm = make_testable_process_manager()
        _add_idle_safety_process(pm)
        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None

        # The watchdog gives up waiting and requeues the job for a fresh check.
        pm._recovery_coordinator.orphan_safety_since[job_id] = time.time() - (
            pm._recovery_coordinator.ORPHAN_SAFETY_GRACE_SECONDS + 1
        )
        await pm._recovery_coordinator.reconcile_orphaned_safety_jobs()
        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SAFETY_CHECK

        # The verdict the watchdog presumed lost was only slow; it arrives now.
        await pm._message_dispatcher._handle_safety_result(
            Mock(job_id=job_id, safety_evaluations=[_safety_eval()], time_elapsed=50.0),
        )

        assert job_info.safety_evaluated is True
        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SUBMIT
        assert pm._job_tracker.jobs_pending_safety_check == ()


class TestSafetyDurationAlwaysFeedsTheAverage:
    """Every verdict's wall-clock feeds the safety-duration average, whatever became of its job.

    The orphan grace scales off this average; a safety stage slow enough that every verdict arrives after
    its job was requeued or faulted is exactly the one the scaling needs to see, so recording only
    verdicts that land on a waiting job locks the grace at the baseline and the requeue loop feeds itself.
    """

    async def test_duration_recorded_when_no_job_awaits_the_verdict(self) -> None:
        """A verdict for an already-resolved job still folds its wall-clock into the average."""
        pm = make_testable_process_manager()
        orphan_verdict_job = _safety_job_info()
        assert pm._state.avg_safety_seconds == 0.0

        await pm._message_dispatcher._handle_safety_result(
            Mock(
                job_id=orphan_verdict_job.sdk_api_job_info.id_,
                safety_evaluations=[_safety_eval()],
                time_elapsed=14.0,
            ),
        )

        assert pm._state.avg_safety_seconds == 14.0


def _deliver_one_message(pm: object, message: object) -> None:
    """Feed a single message through the dispatcher's real receive loop (mock queue: one item, then empty)."""
    dispatcher = pm._message_dispatcher  # type: ignore[attr-defined]
    dispatcher._process_message_queue.empty.side_effect = [False, True]
    dispatcher._process_message_queue.get.return_value = message


_SAFETY_PROCESS_ID = 10
"""The safety process id used by the retired-launch scenarios."""

_RETIRED_LAUNCH_ID = 12
"""The launch that was retired part-way through a check."""

_CURRENT_LAUNCH_ID = 13
"""The launch that replaced it."""


def _retire_safety_launch(pm: object, launch_identifier: int) -> None:
    """Retire a safety launch the way a residency-driven replacement does."""
    safety_proc = make_mock_process_info(
        _SAFETY_PROCESS_ID,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.SAFETY,
    )
    safety_proc.process_launch_identifier = launch_identifier
    pm._process_map[safety_proc.process_id] = safety_proc  # type: ignore[attr-defined]
    pm._process_map.retire_process(  # type: ignore[attr-defined]
        safety_proc,
        "whole-card residency complete: restoring safety to GPU",
    )


def _clean_evaluation() -> HordeSafetyEvaluation:
    """A per-image verdict that clears the image (nothing censored, nothing failed)."""
    return HordeSafetyEvaluation(is_nsfw=False, is_csam=False, replacement_image_bytes=None)


async def _deliver_safety_verdict(
    pm: object,
    job_id: object,
    launch_identifier: int,
    evaluations: list[HordeSafetyEvaluation],
) -> None:
    """Feed one safety verdict from ``launch_identifier`` through the dispatcher's real receive loop."""
    verdict = HordeSafetyResultMessage(
        process_id=_SAFETY_PROCESS_ID,
        process_launch_identifier=launch_identifier,
        info="verdict from a retired safety launch",
        job_id=job_id,  # type: ignore[arg-type]
        safety_evaluations=evaluations,
    )
    _deliver_one_message(pm, verdict)
    await pm._message_dispatcher.receive_and_handle_process_messages()  # type: ignore[attr-defined]


class TestRetiredSafetyLaunchVerdictStillApplies:
    """A verdict for a job still awaiting one is valid whichever launch produced it.

    Whole-card residency moves the safety process off the GPU while a card-filling model holds the device,
    then replaces it (retires the launch, starts a fresh one) when the residency lifts. The outgoing launch
    keeps evaluating the jobs it was already given and returns real verdicts for them. Discarding those
    leaves jobs that were fully checked sitting in SAFETY_CHECKING, drives repeated pool rebuilds, and ends
    in a job whose images were cleared being faulted for want of a check that had in fact completed.

    Validity is keyed on the job: while the retired launch still owns the job's current attempt, its verdict
    is the one that attempt is waiting for and is applied. Everything else keeps the guard's protection, as
    ``TestRetiredSafetyLaunchDropsUnownedVerdicts`` pins.
    """

    async def test_retired_launch_verdict_is_applied_to_the_job_awaiting_it(self) -> None:
        """The job it checked is marked evaluated and moves to submit, with no requeue and no fault."""
        pm = make_testable_process_manager()
        job_info = _safety_job_info()
        await _strand_in_safety_checking(
            pm,
            job_info,
            process_id=_SAFETY_PROCESS_ID,
            process_launch_identifier=_RETIRED_LAUNCH_ID,
        )
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None
        _retire_safety_launch(pm, _RETIRED_LAUNCH_ID)

        await _deliver_safety_verdict(pm, job_id, _RETIRED_LAUNCH_ID, [_clean_evaluation()])

        assert job_info.safety_evaluated is True
        assert job_info.job_image_results is not None
        assert job_info.state != GENERATION_STATE.faulted
        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SUBMIT
        assert job_info not in pm._job_tracker.jobs_being_safety_checked

    async def test_applied_verdict_leaves_nothing_for_the_orphan_watchdog_to_recover(self) -> None:
        """No known-lost flag and no requeue follow an applied verdict: the job was checked, not stranded."""
        pm = make_testable_process_manager()
        job_info = _safety_job_info()
        await _strand_in_safety_checking(
            pm,
            job_info,
            process_id=_SAFETY_PROCESS_ID,
            process_launch_identifier=_RETIRED_LAUNCH_ID,
        )
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None
        _retire_safety_launch(pm, _RETIRED_LAUNCH_ID)

        await _deliver_safety_verdict(pm, job_id, _RETIRED_LAUNCH_ID, [_clean_evaluation()])
        await pm._recovery_coordinator.reconcile_orphaned_safety_jobs()

        assert pm._recovery_coordinator.safety_requeue_count == {}
        assert pm._state.self_throttle_paused is False
        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SUBMIT


class TestRetiredSafetyLaunchDropsUnownedVerdicts:
    """A verdict from a launch that no longer owns the job's check stays dropped, and flags it known-lost.

    Once a job has been re-dispatched to a fresh launch, a verdict from the retired one is a duplicate: the
    live attempt owns the job's state and must not be overwritten by it. Dropping it is still positive
    evidence that the retired launch will not deliver, so the job is re-checked on the next reconcile tick
    rather than waiting out the orphan grace, and the bounded requeue/escalation bookkeeping is unchanged.
    """

    async def _strand_then_drop_retired_verdict(self, pm: object) -> HordeJobInfo:
        """Strand a job under the current launch, then deliver+drop a verdict from the retired one."""
        job_info = _safety_job_info()
        await _strand_in_safety_checking(
            pm,
            job_info,
            process_id=_SAFETY_PROCESS_ID,
            process_launch_identifier=_CURRENT_LAUNCH_ID,
        )
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None
        assert pm._job_tracker.get_stage(job_id) == JobStage.SAFETY_CHECKING  # type: ignore[attr-defined]

        _retire_safety_launch(pm, _RETIRED_LAUNCH_ID)
        await _deliver_safety_verdict(pm, job_id, _RETIRED_LAUNCH_ID, [])
        return job_info

    async def test_unowned_verdict_never_resolves_the_live_attempt(self) -> None:
        """The job stays in SAFETY_CHECKING under its live attempt; the stale verdict changes nothing."""
        pm = make_testable_process_manager()
        job_info = await self._strand_then_drop_retired_verdict(pm)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None

        assert pm._job_tracker.get_stage(job_id) == JobStage.SAFETY_CHECKING
        assert job_info.safety_evaluated is False

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


class TestOffGpuSafetyGraceAndPlacementRebuilds:
    """The orphan watchdog reads where safety is running before it judges a late verdict.

    Two failures this encodes. A check that legitimately takes over a minute on the CPU was requeued at the
    on-GPU grace, discarding progress that was going to finish and paying for the whole check again. And a
    requeue found the pool unready during a deliberate placement rebuild and forced another replacement, which
    restarted that rebuild from the beginning, so the worker rebuilt safety for as long as the placement kept
    flipping.
    """

    async def test_off_gpu_grace_scales_to_what_checks_are_taking(self) -> None:
        """With safety on the CPU and a live pool, the grace follows the measured check duration."""
        pm = make_testable_process_manager()
        _add_idle_safety_process(pm)
        coordinator = pm._recovery_coordinator
        pm._process_lifecycle._safety_gpu_paused = True
        pm._state.avg_safety_seconds = 40.0

        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None

        # Well past the on-GPU baseline, and well inside what checks are actually taking off-GPU.
        coordinator.orphan_safety_since[job_id] = time.time() - (coordinator.ORPHAN_SAFETY_GRACE_SECONDS + 15.0)
        await coordinator.reconcile_orphaned_safety_jobs()
        assert pm._job_tracker.get_stage(job_id) == JobStage.SAFETY_CHECKING

        # The one stranded job is itself the whole backlog, so the grace is the slack multiple plus one
        # queue slot's worth of the measured average.
        coordinator.orphan_safety_since[job_id] = time.time() - (
            40.0 * (coordinator.SAFETY_GRACE_OFF_GPU_CHECK_MULTIPLE + 1) + 1.0
        )
        await coordinator.reconcile_orphaned_safety_jobs()
        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SAFETY_CHECK

    async def test_off_gpu_grace_widens_with_the_safety_backlog(self) -> None:
        """The lane is single-serial, so each queued job adds one average check to the grace.

        A verdict's latency includes draining every check ahead of it; a grace that ignores queue wait
        requeues jobs whose checks were always going to finish, and each requeue queues a duplicate check
        that deepens the very backlog making verdicts late.
        """
        pm = make_testable_process_manager()
        _add_idle_safety_process(pm)
        coordinator = pm._recovery_coordinator
        pm._process_lifecycle._safety_gpu_paused = True
        pm._state.avg_safety_seconds = 14.0

        job_infos = [_safety_job_info() for _ in range(6)]
        for job_info in job_infos:
            await _strand_in_safety_checking(pm, job_info)

        expected = 14.0 * (coordinator.SAFETY_GRACE_OFF_GPU_CHECK_MULTIPLE + 6)
        assert coordinator._orphan_safety_grace_seconds() == expected

        # A job waiting out the queue (past the old per-check-only grace, inside the backlog-aware one)
        # is left alone rather than requeued into a duplicate check.
        job_id = job_infos[0].sdk_api_job_info.id_
        assert job_id is not None
        coordinator.orphan_safety_since[job_id] = time.time() - (expected - 10.0)
        await coordinator.reconcile_orphaned_safety_jobs()
        assert pm._job_tracker.get_stage(job_id) == JobStage.SAFETY_CHECKING

    async def test_on_gpu_safety_keeps_the_baseline_grace(self) -> None:
        """A resident safety process is held to the on-GPU baseline however long its checks average."""
        pm = make_testable_process_manager()
        _add_idle_safety_process(pm)
        coordinator = pm._recovery_coordinator
        # Safety came up on a card: the grace reads the card it occupies, not whether a pause took it.
        pm._process_lifecycle._safety_pinned_card = 0
        pm._state.avg_safety_seconds = 40.0

        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None

        coordinator.orphan_safety_since[job_id] = time.time() - (coordinator.ORPHAN_SAFETY_GRACE_SECONDS + 1.0)
        await coordinator.reconcile_orphaned_safety_jobs()

        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SAFETY_CHECK

    async def test_configured_cpu_safety_scales_the_grace_without_a_pause(self) -> None:
        """Safety configured off the GPU gets the same scaled grace as safety paused off it.

        The pause is only one of the ways safety ends up on the CPU; an operator can configure it there, and
        that check is just as slow. Keying the grace on the pause flag held a permanently CPU-bound safety
        lane to the on-GPU baseline, so every check that outran it was requeued into a duplicate.
        """
        pm = make_testable_process_manager()
        _add_idle_safety_process(pm)
        coordinator = pm._recovery_coordinator
        # Never paused, and holding no card: the configured-CPU placement.
        assert pm._process_lifecycle.is_safety_gpu_paused is False
        assert pm._process_lifecycle.safety_gpu_card_index() is None
        pm._state.avg_safety_seconds = 30.0

        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None

        assert coordinator._orphan_safety_grace_seconds() == 30.0 * (
            coordinator.SAFETY_GRACE_OFF_GPU_CHECK_MULTIPLE + 1
        )

        coordinator.orphan_safety_since[job_id] = time.time() - (coordinator.ORPHAN_SAFETY_GRACE_SECONDS + 15.0)
        await coordinator.reconcile_orphaned_safety_jobs()
        assert pm._job_tracker.get_stage(job_id) == JobStage.SAFETY_CHECKING

    async def test_the_scaled_grace_is_capped(self) -> None:
        """A pathological average cannot let a hung safety process hold a job indefinitely."""
        pm = make_testable_process_manager()
        _add_idle_safety_process(pm)
        coordinator = pm._recovery_coordinator
        pm._process_lifecycle._safety_gpu_paused = True
        pm._state.avg_safety_seconds = 10_000.0

        assert coordinator._orphan_safety_grace_seconds() == coordinator.SAFETY_GRACE_MAX_SECONDS

    async def test_an_intentional_placement_rebuild_is_not_forced_to_restart(self) -> None:
        """A requeue during a deliberate placement rebuild leaves that rebuild to finish on its own."""
        pm = make_testable_process_manager()
        coordinator = pm._recovery_coordinator
        # No safety process: the pool is unready, which is the branch that forces a replacement.
        pm._process_lifecycle._safety_replacement_intentional_until_ready = True
        assert pm._process_lifecycle.safety_placement_transition_pending is True

        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None
        coordinator.orphan_safety_since[job_id] = time.time() - (coordinator.ORPHAN_SAFETY_GRACE_SECONDS + 1.0)

        await coordinator.reconcile_orphaned_safety_jobs()

        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SAFETY_CHECK
        assert pm._process_lifecycle.safety_processes_should_be_replaced is False

    async def test_an_unready_pool_with_no_rebuild_in_flight_is_still_replaced(self) -> None:
        """Absent a deliberate rebuild, an unready pool is still rebuilt: the remedy is not weakened."""
        pm = make_testable_process_manager()
        coordinator = pm._recovery_coordinator

        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None
        coordinator.orphan_safety_since[job_id] = time.time() - (coordinator.ORPHAN_SAFETY_GRACE_SECONDS + 1.0)

        await coordinator.reconcile_orphaned_safety_jobs()

        assert pm._process_lifecycle.safety_processes_should_be_replaced is True


def _add_busy_safety_process(pm: object, process_id: int = 10) -> None:
    """Attach the only safety process, alive and mid-check, so it is progressing but cannot take a check."""
    safety_proc = make_mock_process_info(
        process_id,
        model_name=None,
        state=HordeProcessState.EVALUATING_SAFETY,
        process_type=HordeProcessType.SAFETY,
    )
    pm._process_map[process_id] = safety_proc  # type: ignore[attr-defined]


class TestBusySafetyProcessCountsAsAlive:
    """The orphan watchdog judges a late verdict by whether safety is alive, not whether it is idle.

    The safety lane is single-serial, so the one process is unable to accept a check for the whole time it is
    running one. Read as an unready pool, a healthy process is replaced mid-check and an off-GPU check is held to
    the on-GPU grace, though the check in progress is what the scaled grace is sized for.
    """

    async def test_requeue_beside_a_busy_safety_process_does_not_replace_it(self) -> None:
        """A late verdict is requeued, and the alive process checking ahead of it is left running."""
        pm = make_testable_process_manager()
        _add_busy_safety_process(pm)
        coordinator = pm._recovery_coordinator
        safety_proc = pm._process_map[10]
        assert safety_proc.can_accept_job() is False
        assert safety_proc.is_process_alive() is True
        assert pm._process_lifecycle.safety_placement_transition_pending is False

        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None
        coordinator.orphan_safety_since[job_id] = time.time() - (coordinator.ORPHAN_SAFETY_GRACE_SECONDS + 1.0)

        await coordinator.reconcile_orphaned_safety_jobs()

        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SAFETY_CHECK
        assert coordinator.safety_requeue_count[job_id] == 1
        assert pm._process_lifecycle.safety_processes_should_be_replaced is False

    async def test_a_busy_off_gpu_safety_process_keeps_the_scaled_grace(self) -> None:
        """An off-GPU process mid-check is progressing, so the grace scales to the measured check duration."""
        pm = make_testable_process_manager()
        _add_busy_safety_process(pm)
        coordinator = pm._recovery_coordinator
        pm._process_lifecycle._safety_gpu_paused = True
        pm._state.avg_safety_seconds = 40.0

        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info)

        assert coordinator._orphan_safety_grace_seconds() == 40.0 * (
            coordinator.SAFETY_GRACE_OFF_GPU_CHECK_MULTIPLE + 1
        )

    async def test_a_live_process_that_never_returns_a_verdict_is_replaced_at_the_fault(self) -> None:
        """Requeues leave a live process alone; exhausting them on one job is evidence it does not progress.

        The lifecycle hang watchdog has no safety-check state, so without this a live but hung safety process
        is never replaced and every job it holds is faulted in turn.
        """
        pm = make_testable_process_manager()
        _add_busy_safety_process(pm)
        coordinator = pm._recovery_coordinator

        job_info = _safety_job_info()
        await _strand_in_safety_checking(pm, job_info, process_id=10)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None

        for attempt in range(1, coordinator.SAFETY_REQUEUE_MAX + 1):
            coordinator.orphan_safety_since[job_id] = time.time() - (coordinator.ORPHAN_SAFETY_GRACE_SECONDS + 1.0)
            await coordinator.reconcile_orphaned_safety_jobs()
            assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SAFETY_CHECK
            assert coordinator.safety_requeue_count[job_id] == attempt
            assert pm._process_lifecycle.safety_processes_should_be_replaced is False
            # The orchestrator hands the requeued job back to the same live process.
            await pm._job_tracker.begin_safety_check(job_info, process_id=10)

        coordinator.orphan_safety_since[job_id] = time.time() - (coordinator.ORPHAN_SAFETY_GRACE_SECONDS + 1.0)
        await coordinator.reconcile_orphaned_safety_jobs()

        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SUBMIT
        assert job_info.state == GENERATION_STATE.faulted
        assert pm._process_lifecycle.safety_processes_should_be_replaced is True


def _sent_end_request(safety_proc: object) -> bool:
    """Return whether the parent sent ``END_PROCESS`` down this process's pipe."""
    send = safety_proc.pipe_connection.send  # type: ignore[attr-defined]
    return any(call.args[0].control_flag == HordeControlFlag.END_PROCESS for call in send.call_args_list)


def _pause_safety_off_gpu_for_whole_card(pm: object) -> None:
    """Put the lifecycle in the state a whole-card residency leaves, with safety paused off the GPU and rebuilt."""
    lifecycle = pm._process_lifecycle  # type: ignore[attr-defined]
    lifecycle._safety_gpu_paused = True
    lifecycle._safety_gpu_pause_owner = PauseOwner.WHOLE_CARD
    lifecycle.start_safety_processes = Mock()


_CLIP_FORM_ID = "clip-form-1"
"""A CLIP alchemy form, which runs on the safety process."""


class TestPlacementMoveWaitsForTheCheck:
    """A safety placement move ends the process only after the check or CLIP form it is running returns.

    The end of a whole-card residency restores safety to the GPU by replacing the CPU safety process. Ending it
    on the replacement's first tick would discard the check in flight, and the orphan sweep would requeue the job
    after its grace, so a job already being checked would pay a second check and the grace on its submit path.
    The end request waits for the job tracker's current safety attempt to resolve, bounded by ``process_timeout``.
    """

    async def _restore_with_a_check_in_flight(self, pm: object) -> HordeJobInfo:
        """Seat a CPU safety process mid-check on one job, then request the restore to the GPU."""
        _add_busy_safety_process(pm, _SAFETY_PROCESS_ID)
        _pause_safety_off_gpu_for_whole_card(pm)
        job_info = _safety_job_info()
        await _strand_in_safety_checking(
            pm,
            job_info,
            process_id=_SAFETY_PROCESS_ID,
            process_launch_identifier=0,
        )
        assert pm._process_lifecycle.restore_safety_on_gpu(owner=PauseOwner.WHOLE_CARD) is True  # type: ignore[attr-defined]
        return job_info

    async def test_a_restore_mid_check_ends_the_process_after_the_verdict_reaches_the_job(self) -> None:
        """The end waits across ticks, the verdict is applied to the job, and only then is the process ended."""
        pm = make_testable_process_manager()
        job_info = await self._restore_with_a_check_in_flight(pm)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None
        safety_proc = pm._process_map[_SAFETY_PROCESS_ID]
        lifecycle = pm._process_lifecycle

        lifecycle._replace_all_safety_process()
        lifecycle._replace_all_safety_process()

        assert _sent_end_request(safety_proc) is False
        assert safety_proc.end_intended is False
        assert safety_proc.last_process_state == HordeProcessState.EVALUATING_SAFETY

        await _deliver_safety_verdict(pm, job_id, 0, [_clean_evaluation()])

        assert job_info.safety_evaluated is True
        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SUBMIT

        lifecycle._replace_all_safety_process()

        assert _sent_end_request(safety_proc) is True
        assert safety_proc.end_intended is True
        assert safety_proc.last_process_state == HordeProcessState.PROCESS_ENDING

    async def test_a_restore_with_no_check_in_flight_ends_the_process_at_once(self) -> None:
        """An idle process is ended on the replacement's first tick."""
        pm = make_testable_process_manager()
        _add_idle_safety_process(pm, _SAFETY_PROCESS_ID)
        _pause_safety_off_gpu_for_whole_card(pm)
        safety_proc = pm._process_map[_SAFETY_PROCESS_ID]

        assert pm._process_lifecycle.restore_safety_on_gpu(owner=PauseOwner.WHOLE_CARD) is True
        pm._process_lifecycle._replace_all_safety_process()

        assert _sent_end_request(safety_proc) is True
        assert safety_proc.last_process_state == HordeProcessState.PROCESS_ENDING

    async def test_a_check_that_never_returns_is_ended_at_the_unresponsive_timeout(self) -> None:
        """A check older than ``process_timeout`` no longer holds the end request."""
        pm = make_testable_process_manager()
        job_info = await self._restore_with_a_check_in_flight(pm)
        job_id = job_info.sdk_api_job_info.id_
        safety_proc = pm._process_map[_SAFETY_PROCESS_ID]
        lifecycle = pm._process_lifecycle

        lifecycle._replace_all_safety_process()
        assert _sent_end_request(safety_proc) is False

        (tracked,) = [tracked for tracked in pm._job_tracker.tracked_jobs() if tracked.job_id == job_id]
        tracked.current_stage_since = time.time() - (pm.bridge_data.process_timeout + 1.0)
        lifecycle._replace_all_safety_process()

        assert _sent_end_request(safety_proc) is True
        assert safety_proc.last_process_state == HordeProcessState.PROCESS_ENDING

    async def test_a_dead_process_mid_check_is_not_waited_on(self) -> None:
        """A process that exited cannot deliver its verdict, so the replacement proceeds on the next tick."""
        pm = make_testable_process_manager()
        await self._restore_with_a_check_in_flight(pm)
        safety_proc = pm._process_map[_SAFETY_PROCESS_ID]
        safety_proc.mp_process.is_alive.return_value = False

        pm._process_lifecycle._replace_all_safety_process()

        assert _sent_end_request(safety_proc) is True

    async def test_a_new_check_is_not_sent_while_the_replacement_is_armed(self) -> None:
        """An idle process about to be ended does not take a check, so a backlog cannot hold the move open."""
        pm = make_testable_process_manager()
        _add_idle_safety_process(pm, _SAFETY_PROCESS_ID)
        _pause_safety_off_gpu_for_whole_card(pm)
        safety_proc = pm._process_map[_SAFETY_PROCESS_ID]
        job_info = _safety_job_info()
        await queue_job_for_safety_async(pm._job_tracker, job_info)
        assert pm._process_lifecycle.restore_safety_on_gpu(owner=PauseOwner.WHOLE_CARD) is True

        await pm.start_evaluate_safety()

        safety_proc.pipe_connection.send.assert_not_called()
        assert job_info in pm._job_tracker.jobs_pending_safety_check

    async def test_a_job_checked_across_a_restore_is_never_requeued_by_the_orphan_sweep(self) -> None:
        """Across the whole move, the job keeps the one check and the sweep does not requeue it."""
        pm = make_testable_process_manager()
        job_info = await self._restore_with_a_check_in_flight(pm)
        job_id = job_info.sdk_api_job_info.id_
        assert job_id is not None
        safety_proc = pm._process_map[_SAFETY_PROCESS_ID]
        lifecycle = pm._process_lifecycle
        coordinator = pm._recovery_coordinator

        verdict_tick = 3
        for tick in range(6):
            # The child finishes the check on the verdict tick unless it was sent END_PROCESS first. A child that
            # reads END_PROCESS exits, and the check it was running does not report.
            if tick == verdict_tick and not _sent_end_request(safety_proc):
                await _deliver_safety_verdict(pm, job_id, 0, [_clean_evaluation()])
            if _sent_end_request(safety_proc):
                safety_proc.mp_process.is_alive.return_value = False
            lifecycle._replace_all_safety_process()
            await coordinator.reconcile_orphaned_safety_jobs()

        # However long the move took, a job still awaiting a verdict would now be past the sweep's grace.
        if job_id in coordinator.orphan_safety_since:
            coordinator.orphan_safety_since[job_id] = time.time() - (coordinator.ORPHAN_SAFETY_GRACE_SECONDS + 1.0)
        await coordinator.reconcile_orphaned_safety_jobs()

        assert _SAFETY_PROCESS_ID not in pm._process_map
        lifecycle.start_safety_processes.assert_called_once()
        assert pm._job_tracker.get_stage(job_id) == JobStage.PENDING_SUBMIT
        assert job_info.safety_evaluated is True
        assert coordinator.safety_requeue_count == {}

    @staticmethod
    def _queue_clip_form(pm: object) -> None:
        """Queue one CLIP form, which routes to the safety process."""
        pm._alchemy_coordinator._pending_forms.append(  # type: ignore[attr-defined]
            AlchemyFormSpec(form_id=_CLIP_FORM_ID, form="caption", source_image_bytes=b"img"),
        )

    def _restore_with_a_clip_form_in_flight(self, pm: object) -> None:
        """Dispatch a CLIP form to an idle CPU safety process, then request the restore to the GPU."""
        _add_idle_safety_process(pm, _SAFETY_PROCESS_ID)
        _pause_safety_off_gpu_for_whole_card(pm)
        self._queue_clip_form(pm)
        pm._alchemy_coordinator.dispatch_pending_forms()  # type: ignore[attr-defined]
        assert _CLIP_FORM_ID in pm._alchemy_coordinator._in_flight  # type: ignore[attr-defined]
        assert pm._process_lifecycle.restore_safety_on_gpu(owner=PauseOwner.WHOLE_CARD) is True  # type: ignore[attr-defined]

    async def test_a_restore_mid_clip_form_ends_the_process_after_the_form_result(self) -> None:
        """A CLIP form on the safety process holds the end request until its result reaches the coordinator."""
        pm = make_testable_process_manager()
        self._restore_with_a_clip_form_in_flight(pm)
        safety_proc = pm._process_map[_SAFETY_PROCESS_ID]
        lifecycle = pm._process_lifecycle
        alchemy = pm._alchemy_coordinator

        lifecycle._replace_all_safety_process()
        lifecycle._replace_all_safety_process()

        assert _sent_end_request(safety_proc) is False
        assert safety_proc.end_intended is False

        alchemy.on_alchemy_result(
            HordeAlchemyResultMessage(
                process_id=_SAFETY_PROCESS_ID,
                process_launch_identifier=safety_proc.process_launch_identifier,
                info="caption done",
                form_id=_CLIP_FORM_ID,
                form="caption",
                state=GENERATION_STATE.ok,
                result_payload={"caption": "a cat"},
            ),
        )

        assert alchemy._in_flight == {}
        assert [submit.form_id for submit in alchemy._pending_submits] == [_CLIP_FORM_ID]

        lifecycle._replace_all_safety_process()

        assert _sent_end_request(safety_proc) is True
        assert safety_proc.last_process_state == HordeProcessState.PROCESS_ENDING

    async def test_a_clip_form_past_the_unresponsive_timeout_does_not_hold_the_end(self) -> None:
        """A form dispatched longer ago than ``process_timeout`` no longer holds the end request."""
        pm = make_testable_process_manager()
        self._restore_with_a_clip_form_in_flight(pm)
        safety_proc = pm._process_map[_SAFETY_PROCESS_ID]
        lifecycle = pm._process_lifecycle

        lifecycle._replace_all_safety_process()
        assert _sent_end_request(safety_proc) is False

        pm._alchemy_coordinator._in_flight_dispatched_at[_CLIP_FORM_ID] = time.monotonic() - (
            pm.bridge_data.process_timeout + 1.0
        )
        lifecycle._replace_all_safety_process()

        assert _sent_end_request(safety_proc) is True
        assert safety_proc.last_process_state == HordeProcessState.PROCESS_ENDING

    async def test_a_clip_form_is_not_sent_while_the_replacement_is_armed(self) -> None:
        """An idle process about to be ended does not take a CLIP form, so a stream of forms cannot hold the move."""
        pm = make_testable_process_manager()
        _add_idle_safety_process(pm, _SAFETY_PROCESS_ID)
        _pause_safety_off_gpu_for_whole_card(pm)
        safety_proc = pm._process_map[_SAFETY_PROCESS_ID]
        assert pm._process_lifecycle.restore_safety_on_gpu(owner=PauseOwner.WHOLE_CARD) is True
        assert pm._process_lifecycle.safety_replacement_armed is True
        self._queue_clip_form(pm)

        pm._alchemy_coordinator.dispatch_pending_forms()

        safety_proc.pipe_connection.send.assert_not_called()
        assert pm._alchemy_coordinator._in_flight == {}
        assert [spec.form_id for spec in pm._alchemy_coordinator._pending_forms] == [_CLIP_FORM_ID]
