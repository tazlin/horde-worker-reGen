"""Belt-and-suspenders for a lost inference result, plus the assumptions that shape the fix.

When a slot's inference result is lost (for example dropped by the launch-identifier guard while the
slot is being replaced), the job stays marked in progress with nothing left to move it on. The orphaned
-job watchdog is the periodic backstop; the prompt detector tested here is the second, independent
layer: the moment a slot reports it is idle again (``WAITING_FOR_JOB``) while still referencing a job
that is still in progress, the result can only have been lost, so the job is released immediately.

The investigation behind this fix found that ``last_job_referenced`` is *not* a "currently running"
flag. It is the job whose model the slot is associated with: it is cleared when the model is unloaded
or the process ends, but deliberately retained across a job's completion, and several scheduling
heuristics read it while the slot is idle. The characterization tests here lock those assumptions down
so a future "just clear it when the slot goes idle" change cannot silently break them:

* ``keep_single_inference`` keeps the worker single-process for a resident ControlNet-XL job *while the
  slot is idle*, reading the retained reference.
* the VRAM-heavy variant of that same check, by contrast, only fires while the slot is actively
  inferring.
* model unload and process end are the points that clear the reference.

That asymmetry is exactly why the lost-result fix releases the stuck *job* rather than blanket-clearing
the reference.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeInferenceResultMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_job,
    make_mock_model_reference_record,
    make_mock_process_info,
    mark_job_in_progress_async,
)
from tests.process_management.ipc.test_message_dispatch import _enqueue_many, _make_dispatcher


def _waiting_for_job_message(process_id: int, launch_identifier: int = 0) -> HordeProcessStateChangeMessage:
    return HordeProcessStateChangeMessage(
        process_id=process_id,
        process_launch_identifier=launch_identifier,
        process_state=HordeProcessState.WAITING_FOR_JOB,
        info="Waiting for job",
    )


class TestPromptLostResultRecovery:
    """A slot reporting idle while its job is still in progress means the result was lost: release it now."""

    async def test_idle_transition_releases_a_job_whose_result_was_lost(self) -> None:
        """The new mechanism: WAITING_FOR_JOB + still-in-progress referenced job -> the job is released.

        This is the incident at the message level: the result that should have ended the job never
        reached the parent, so without this the job would pin the in-progress count forever.
        """
        process_info = make_mock_process_info(
            2,
            model_name="AlbedoBase XL 3.1",
            state=HordeProcessState.INFERENCE_STARTING,
        )
        process_map = ProcessMap({2: process_info})
        job_tracker = JobTracker()
        dispatcher = _make_dispatcher(process_map=process_map, job_tracker=job_tracker)

        job_tracker.set_retry_policy(2)  # attempts remain, so a lost result requeues rather than faulting out
        job = make_job_pop_response(model="AlbedoBase XL 3.1")
        await job_tracker.record_popped_job(job)
        await mark_job_in_progress_async(job_tracker, job)
        process_info.last_job_referenced = job
        assert job in job_tracker.jobs_in_progress
        assert job.id_ is not None

        dispatcher._handle_process_state_change(_waiting_for_job_message(2))

        assert job not in job_tracker.jobs_in_progress
        # Released retryably so the horde gets it done, not silently dropped or pinned in progress.
        assert job_tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE

    async def test_lost_result_is_reported_faulted_when_no_retries_remain(self) -> None:
        """The release is bounded: with attempts exhausted the job is faulted and reported, not requeued.

        This guards against the detector turning a lost result into an endless requeue loop: once the
        retry budget is spent it routes the job to PENDING_SUBMIT so the horde reissues it and the worker
        moves on.
        """
        process_info = make_mock_process_info(2, model_name="m", state=HordeProcessState.INFERENCE_STARTING)
        process_map = ProcessMap({2: process_info})
        job_tracker = JobTracker()
        job_tracker.set_retry_policy(1)  # the single attempt is already spent by dispatching the job
        dispatcher = _make_dispatcher(process_map=process_map, job_tracker=job_tracker)

        job = make_job_pop_response(model="m")
        await job_tracker.record_popped_job(job)
        await mark_job_in_progress_async(job_tracker, job)
        process_info.last_job_referenced = job
        assert job.id_ is not None

        dispatcher._handle_process_state_change(_waiting_for_job_message(2))

        assert job not in job_tracker.jobs_in_progress
        assert job_tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT

    async def test_dropped_result_then_idle_recovers_through_the_receive_loop(self) -> None:
        """End-to-end at the message layer: a result dropped by the launch-id guard, then the idle report.

        This is the precise incident shape. A stale-launch result message is discarded (the slot was
        being replaced), so the job is never moved on; the very next idle report on the live launch lets
        the prompt detector release it. The two messages flow through the real ordered receive loop.
        """
        live_launch = 113
        process_info = make_mock_process_info(2, model_name="m", state=HordeProcessState.INFERENCE_STARTING)
        process_info.process_launch_identifier = live_launch
        process_map = ProcessMap({2: process_info})
        job_tracker = JobTracker()
        job_tracker.set_retry_policy(2)
        dispatcher = _make_dispatcher(process_map=process_map, job_tracker=job_tracker)

        job = make_job_pop_response(model="m")
        await job_tracker.record_popped_job(job)
        await mark_job_in_progress_async(job_tracker, job)
        process_info.last_job_referenced = job
        assert job.id_ is not None

        stale_result = Mock(spec=HordeInferenceResultMessage)
        stale_result.process_id = 2
        stale_result.process_launch_identifier = 106  # an older, already-replaced launch: this is dropped
        stale_result.sdk_api_job_info = job

        idle_report = HordeProcessStateChangeMessage(
            process_id=2,
            process_launch_identifier=live_launch,
            process_state=HordeProcessState.WAITING_FOR_JOB,
            info="Waiting for job",
        )

        _enqueue_many(dispatcher, [stale_result, idle_report])
        await dispatcher.receive_and_handle_process_messages()

        # The dropped result left the job stuck; the idle report recovered it instead of wedging.
        assert job not in job_tracker.jobs_in_progress
        assert job_tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE

    async def test_idle_transition_does_not_disturb_a_completed_job(self) -> None:
        """A job that already left in-progress (its result was handled) is not re-faulted on idle.

        The reference is retained after completion (by design), but the job is no longer in progress, so
        the detector must leave it exactly where the normal pipeline put it. This is the guard that the
        result-before-idle message ordering relies on.
        """
        process_info = make_mock_process_info(
            2,
            model_name="stable_diffusion",
            state=HordeProcessState.INFERENCE_COMPLETE,
        )
        process_map = ProcessMap({2: process_info})
        job_tracker = JobTracker()
        dispatcher = _make_dispatcher(process_map=process_map, job_tracker=job_tracker)

        job = make_job_pop_response(model="stable_diffusion")
        job_info = await job_tracker.record_popped_job(job)
        await mark_job_in_progress_async(job_tracker, job)
        process_info.last_job_referenced = job
        # The result arrived and moved the job into the safety tail, as it would in the normal FIFO order.
        await job_tracker.queue_for_safety(job_info)
        assert job not in job_tracker.jobs_in_progress
        assert job_info in job_tracker.jobs_pending_safety_check
        assert job.id_ is not None

        dispatcher._handle_process_state_change(_waiting_for_job_message(2))

        # Untouched: still in the safety tail, and the reference is still retained (not cleared on idle).
        assert job_info in job_tracker.jobs_pending_safety_check
        assert job_tracker.get_stage(job.id_) == JobStage.PENDING_SAFETY_CHECK
        assert process_info.last_job_referenced is job

    async def test_idle_transition_with_no_referenced_job_is_a_noop(self) -> None:
        """An ordinary idle transition on a slot with no associated job does nothing."""
        process_info = make_mock_process_info(2, model_name=None, state=HordeProcessState.INFERENCE_COMPLETE)
        process_map = ProcessMap({2: process_info})
        job_tracker = JobTracker()
        dispatcher = _make_dispatcher(process_map=process_map, job_tracker=job_tracker)

        dispatcher._handle_process_state_change(_waiting_for_job_message(2))

        assert len(job_tracker.jobs_in_progress) == 0

    async def test_dispatch_window_is_not_reaped(self) -> None:
        """A just-dispatched job (in progress, referenced by a still-preloaded slot) must not be released.

        Between marking a job in progress and the child reporting INFERENCE_STARTING, the slot is briefly
        PRELOADED_MODEL while already referencing the new in-progress job. Only the idle (WAITING_FOR_JOB)
        transition signals a finished/abandoned job, so a preload-complete transition must leave the job
        alone; otherwise the recovery path would cannibalise healthy dispatches.
        """
        process_info = make_mock_process_info(
            3,
            model_name="stable_diffusion",
            state=HordeProcessState.PRELOADING_MODEL,
        )
        process_map = ProcessMap({3: process_info})
        job_tracker = JobTracker()
        dispatcher = _make_dispatcher(process_map=process_map, job_tracker=job_tracker)

        job = make_job_pop_response(model="stable_diffusion")
        await job_tracker.record_popped_job(job)
        await mark_job_in_progress_async(job_tracker, job)
        process_info.last_job_referenced = job

        dispatcher._handle_process_state_change(
            HordeProcessStateChangeMessage(
                process_id=3,
                process_launch_identifier=0,
                process_state=HordeProcessState.PRELOADED_MODEL,
                info="Preloaded",
            ),
        )

        assert job in job_tracker.jobs_in_progress


class TestDispatchRaceIsNotReapedAsLostResult:
    """A job dispatched onto a slot that is still draining a *prior* idle must not be reaped as lost.

    The reap's correctness rests on one assumption: a ``WAITING_FOR_JOB`` transition seen while a
    referenced job is still in progress can only be a slot returning to idle *after* running that job, so
    a missing result means it was lost. That assumption breaks during the dispatch window.

    ``last_job_referenced`` and the in-progress mark are set by the scheduler the instant it *dispatches*
    a job, before the child has acknowledged (or even received) it. Meanwhile the child can still be
    emitting state messages from the moment *before* the dispatch, e.g. the ``WAITING_FOR_JOB`` it reports
    after unloading the previous model to make room. The parent reads that stale idle report only after it
    has already optimistically stamped the new job onto the slot, so the reap sees "idle + job in
    progress" and wrongly concludes the new job's result was lost, faulting a job that never ran.

    The discriminator is the state the slot is transitioning *from*: only a return to idle from an
    inference-active state means a result could have been produced and lost. A return to idle from a
    teardown/preload path is the dispatch window, and the slot is about to pick the job up. This is also
    machine-general: slower disks/model swaps widen the dispatch window, making the false positive more
    likely, not less.
    """

    async def test_idle_from_unload_path_does_not_reap_a_freshly_dispatched_job(self) -> None:
        """Incident shape: slot goes UNLOADED_MODEL_FROM_RAM -> WAITING_FOR_JOB with a just-dispatched job.

        The slot unloaded the previous model to free VRAM for the new job, reported the resulting idle,
        and only then will it preload + run the dispatched job. The idle report predates the job starting,
        so the job must stay in progress for the slot to take up; reaping it here double-faults a healthy
        job (and, in the field, races a model that is simply slow to load).
        """
        process_info = make_mock_process_info(
            1,
            model_name="Flux.1-Schnell fp8 (Compact)",
            state=HordeProcessState.UNLOADED_MODEL_FROM_RAM,
        )
        process_map = ProcessMap({1: process_info})
        job_tracker = JobTracker()
        job_tracker.set_retry_policy(2)
        dispatcher = _make_dispatcher(process_map=process_map, job_tracker=job_tracker)

        job = make_job_pop_response(model="Flux.1-Schnell fp8 (Compact)")
        await job_tracker.record_popped_job(job)
        await mark_job_in_progress_async(job_tracker, job)
        # The scheduler stamps the slot at dispatch, before the child has begun the job.
        process_info.last_job_referenced = job
        process_info.last_control_flag = HordeControlFlag.START_INFERENCE
        assert job in job_tracker.jobs_in_progress

        dispatcher._handle_process_state_change(_waiting_for_job_message(1))

        assert job in job_tracker.jobs_in_progress

    async def test_idle_from_preload_complete_does_not_reap_a_freshly_dispatched_job(self) -> None:
        """The same window, reached via a preload-complete path rather than an unload.

        A slot can settle to idle straight off a (no-op) preload before taking the dispatched job. As with
        the unload path, no inference has run, so there is no result to have lost.
        """
        process_info = make_mock_process_info(
            1,
            model_name="stable_diffusion",
            state=HordeProcessState.PRELOADED_MODEL,
        )
        process_map = ProcessMap({1: process_info})
        job_tracker = JobTracker()
        job_tracker.set_retry_policy(2)
        dispatcher = _make_dispatcher(process_map=process_map, job_tracker=job_tracker)

        job = make_job_pop_response(model="stable_diffusion")
        await job_tracker.record_popped_job(job)
        await mark_job_in_progress_async(job_tracker, job)
        process_info.last_job_referenced = job

        dispatcher._handle_process_state_change(_waiting_for_job_message(1))

        assert job in job_tracker.jobs_in_progress

    async def test_unload_then_idle_sequence_through_the_receive_loop_keeps_the_job(self) -> None:
        """The full incident ordering through the real receive loop: unload report, then idle report.

        Both messages arrive on the live launch after the scheduler has already marked the job in
        progress. The job must survive the burst and remain available for the slot to run.
        """
        process_info = make_mock_process_info(
            1,
            model_name="Flux.1-Schnell fp8 (Compact)",
            state=HordeProcessState.INFERENCE_COMPLETE,
        )
        process_map = ProcessMap({1: process_info})
        job_tracker = JobTracker()
        job_tracker.set_retry_policy(2)
        dispatcher = _make_dispatcher(process_map=process_map, job_tracker=job_tracker)

        job = make_job_pop_response(model="Flux.1-Schnell fp8 (Compact)")
        await job_tracker.record_popped_job(job)
        await mark_job_in_progress_async(job_tracker, job)
        process_info.last_job_referenced = job

        unload_report = HordeProcessStateChangeMessage(
            process_id=1,
            process_launch_identifier=0,
            process_state=HordeProcessState.UNLOADED_MODEL_FROM_RAM,
            info="Unloaded models from RAM",
        )
        idle_report = _waiting_for_job_message(1)

        _enqueue_many(dispatcher, [unload_report, idle_report])
        await dispatcher.receive_and_handle_process_messages()

        assert job in job_tracker.jobs_in_progress


class TestKeepSingleInferenceReadsRetainedReference:
    """``keep_single_inference`` is the main reader that depends on the reference surviving into idle.

    These lock that dependency: it is why the lost-result fix releases the stuck job instead of clearing
    the reference when a slot goes idle.
    """

    async def test_controlnet_xl_keeps_single_inference_while_slot_is_idle(self) -> None:
        """An idle slot still associated with a resident ControlNet-XL job keeps the worker single-process.

        ``can_accept_job()`` is true here (WAITING_FOR_JOB), so this only works because the reference is
        retained after the job stops actively running. Clearing it on idle would silently drop the guard.
        """
        controlnet_xl_model = "qr-controlnet-sdxl"
        process_info = make_mock_process_info(
            1,
            model_name=controlnet_xl_model,
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        process_info.last_job_referenced = make_mock_job(model=controlnet_xl_model, workflow="qr_code")
        process_map = ProcessMap({1: process_info})

        reference = {
            controlnet_xl_model: make_mock_model_reference_record(
                controlnet_xl_model,
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            ),
        }

        keep, reason = process_map.keep_single_inference(
            stable_diffusion_model_reference=reference,
            post_process_job_overlap=False,
        )

        assert keep is True
        assert reason == "ControlNet XL"

    async def test_card_demanding_model_does_not_hold_worker_single(self) -> None:
        """A card-demanding model actively inferring places no worker-wide single-process hold here.

        Its serialization belongs to the scheduler's size-tier overlap gate, which is derived from the
        model reference, scopes to the in-flight job's card, and relaxes against measured headroom. A
        second name-list hold in the process map would stack on that gate while being blind to devices
        and to demanding models the list never learned about.
        """
        card_demanding_model = "Flux.1-Schnell fp8 (Compact)"

        actively_inferring = make_mock_process_info(
            1,
            model_name=card_demanding_model,
            state=HordeProcessState.INFERENCE_STARTING,
        )
        actively_inferring.last_job_referenced = make_mock_job(model=card_demanding_model)
        keep_busy, _reason_busy = ProcessMap({1: actively_inferring}).keep_single_inference(
            stable_diffusion_model_reference={},
            post_process_job_overlap=True,
        )
        assert keep_busy is False


class TestReferenceClearedOnModelTeardown:
    """The reference is cleared when the model leaves the slot: that is the field's real lifecycle."""

    def test_ram_unload_clears_the_job_association(self) -> None:
        """Unloading the model from RAM drops the job association (the model it belonged to is gone)."""
        process_info = make_mock_process_info(
            1,
            model_name="stable_diffusion",
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        process_info.last_job_referenced = make_mock_job(model="stable_diffusion")
        process_map = ProcessMap({1: process_info})

        process_map.on_model_ram_clear(1)

        assert process_info.last_job_referenced is None
        assert process_info.loaded_horde_model_name is None

    def test_process_end_clears_the_job_association(self) -> None:
        """A slot reported as ending drops its job association so nothing reads a dead slot's old job."""
        process_info = make_mock_process_info(
            1,
            model_name="stable_diffusion",
            state=HordeProcessState.INFERENCE_STARTING,
        )
        process_info.last_job_referenced = make_mock_job(model="stable_diffusion")
        process_map = ProcessMap({1: process_info})

        process_map.on_process_ending(1)

        assert process_info.last_job_referenced is None
