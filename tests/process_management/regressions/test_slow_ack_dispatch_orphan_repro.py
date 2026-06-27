"""RED reproduction: the orphan watchdog punts a slot that is slow to acknowledge its dispatch.

Observed in a live session (a 16GB, ``max_threads=1`` worker carrying ~110 models, so its inference
slots constantly cycle for RAM reclaim and re-spawn under host contention). A light img2img job was
line-skipped onto an inference slot via ``START_INFERENCE``, but the child, starved of CPU/IO by a
sibling that was simultaneously downloading aux models and another that was sampling, did not drain its
control pipe and transition out of ``WAITING_FOR_JOB`` for ~74s. Meanwhile the orchestrator's
orphaned-in-progress-job watchdog saw a slot that still ``can_accept_job()`` (idle), concluded no live
slot owned the job, and punted it after the 30s grace. The bounded retry then re-dispatched the job to
the *same* still-stalled slot, which stalled again and was punted a second time, faulting the job to the
horde. The slot finally drained its pipe and produced a valid result, which was dropped with
``Job ... not found in jobs_lookup`` -- the GPU work was wasted and the requestor's job needlessly
faulted.

The root cause is that ``_inference_slot_owns_job`` decides ownership purely from ``can_accept_job()``
(i.e. process state), ignoring that the slot has a fresh ``START_INFERENCE`` in flight for exactly this
job. ``start_inference`` stamps the slot with ``last_control_flag == START_INFERENCE``,
``current_inference_started_at`` (dispatch time) and ``last_job_referenced``; a slot so stamped is the
genuine owner of the dispatched job during the brief window before it acks, and must not be treated as
an idle slot carrying a stale reference (the lost-result case the watchdog legitimately punts).

These tests assert the corrected behavior and are expected to FAIL (RED) against the current code:

* ``test_freshly_dispatched_slot_owns_its_job_despite_waiting_state`` -- ownership recognition.
* ``test_watchdog_does_not_start_orphan_clock_for_freshly_dispatched_slot`` -- the watchdog backstop.

``test_idle_slot_with_stale_reference_is_still_punted`` is the guard: it pins the watchdog's legitimate
behavior (a truly idle slot carrying only a stale reference, no dispatch in flight, is still an orphan)
so the fix stays narrow and does not reintroduce the original orphaned-job wedge.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeInferenceResultMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobStage
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)


async def test_freshly_dispatched_slot_owns_its_job_despite_waiting_state() -> None:
    """A slot with a fresh START_INFERENCE in flight owns its job even while briefly still WAITING_FOR_JOB.

    Reproduces the dispatch-ack race: the orchestrator has sent START_INFERENCE and stamped the slot, but
    the contended child has not yet transitioned state. The slot is the job's owner; ``can_accept_job()``
    being momentarily true (idle state) does not make it available for other work.
    """
    pm = make_testable_process_manager()

    slot = make_mock_process_info(3, model_name="Deliberate 3.0", state=HordeProcessState.WAITING_FOR_JOB)
    slot.last_control_flag = HordeControlFlag.START_INFERENCE
    slot.current_inference_started_at = time.time()  # dispatched just now; ack not yet received
    pm._process_map[3] = slot

    job = make_job_pop_response(model="Deliberate 3.0")
    await track_popped_job_async(pm._job_tracker, job)
    await pm._job_tracker.mark_inference_started(job)
    slot.last_job_referenced = job
    assert job.id_ is not None

    # RED: the slot holding the freshly dispatched job is its owner.
    assert pm._recovery_coordinator.inference_slot_owns_job(job.id_) is True


async def test_watchdog_does_not_start_orphan_clock_for_freshly_dispatched_slot() -> None:
    """The orphan watchdog must not begin tracking (let alone punt) a job whose slot was just dispatched.

    Because the slot is recognized as the owner, the watchdog never records the job as orphaned, so the
    grace clock never starts and the slot keeps the job long enough to actually run it -- instead of the
    observed punt-and-requeue-to-the-same-stalled-slot loop that faulted a job that was about to succeed.
    """
    pm = make_testable_process_manager()

    slot = make_mock_process_info(3, model_name="Deliberate 3.0", state=HordeProcessState.WAITING_FOR_JOB)
    slot.last_control_flag = HordeControlFlag.START_INFERENCE
    slot.current_inference_started_at = time.time()
    pm._process_map[3] = slot

    job = make_job_pop_response(model="Deliberate 3.0")
    await track_popped_job_async(pm._job_tracker, job)
    await pm._job_tracker.mark_inference_started(job)
    slot.last_job_referenced = job
    assert job.id_ is not None

    pm._recovery_coordinator.reconcile_orphaned_in_progress_jobs()

    # RED: an owned job is never marked orphaned, so it is not on the grace clock and not punted.
    assert job.id_ not in pm._recovery_coordinator.orphan_in_progress_since
    assert pm._job_tracker.get_stage(job.id_) == JobStage.INFERENCE_IN_PROGRESS
    assert pm._recovery_coordinator.orphan_punt_history == []


async def test_idle_slot_with_stale_reference_is_still_punted() -> None:
    """Guard (expected GREEN, before and after the fix): a stale reference with no dispatch is an orphan.

    A genuinely idle slot whose ``last_job_referenced`` is a long-finished job (no START_INFERENCE in
    flight, no fresh dispatch stamp) does not own the job a lost result stranded in progress. The
    watchdog must still punt it, so the narrow dispatch-aware ownership fix does not reintroduce the
    original orphaned-job wedge.
    """
    pm = make_testable_process_manager()

    slot = make_mock_process_info(3, model_name="Deliberate 3.0", state=HordeProcessState.WAITING_FOR_JOB)
    # No dispatch in flight: the control flag is not START_INFERENCE and there is no dispatch timestamp.
    slot.last_control_flag = None
    slot.current_inference_started_at = None
    pm._process_map[3] = slot

    job = make_job_pop_response(model="Deliberate 3.0")
    await track_popped_job_async(pm._job_tracker, job)
    await pm._job_tracker.mark_inference_started(job)
    slot.last_job_referenced = job  # stale reference only
    assert job.id_ is not None

    assert pm._recovery_coordinator.inference_slot_owns_job(job.id_) is False

    pm._recovery_coordinator.orphan_in_progress_since[job.id_] = time.time() - (
        pm._recovery_coordinator.ORPHAN_IN_PROGRESS_GRACE_SECONDS + 1
    )
    pm._recovery_coordinator.reconcile_orphaned_in_progress_jobs()

    # The stranded job is punted off the head (requeued by the bounded-retry policy) so the queue drains.
    assert pm._job_tracker.get_stage(job.id_) != JobStage.INFERENCE_IN_PROGRESS
    assert len(pm._recovery_coordinator.orphan_punt_history) == 1


async def test_re_dispatch_re_stamps_ownership_to_the_new_job() -> None:
    """Dispatching a second job to the same slot moves ownership: the first job is no longer owned by it.

    The dispatch stamps are per-slot, so re-dispatch overwrites them. A slot freshly stamped for job B does
    not also own a previously dispatched job A; otherwise a re-used slot would shield a stale A forever.
    """
    pm = make_testable_process_manager()

    slot = make_mock_process_info(3, model_name="m", state=HordeProcessState.WAITING_FOR_JOB)
    pm._process_map[3] = slot

    job_a = make_job_pop_response(model="m")
    job_b = make_job_pop_response(model="m")
    for job in (job_a, job_b):
        await track_popped_job_async(pm._job_tracker, job)
        await pm._job_tracker.mark_inference_started(job)
    assert job_a.id_ is not None
    assert job_b.id_ is not None

    # Dispatch A, then re-dispatch B onto the same slot (the stamps are overwritten).
    slot.last_control_flag = HordeControlFlag.START_INFERENCE
    slot.current_inference_started_at = time.time()
    slot.last_job_referenced = job_a
    assert pm._recovery_coordinator.inference_slot_owns_job(job_a.id_) is True

    slot.current_inference_started_at = time.time()
    slot.last_job_referenced = job_b

    assert pm._recovery_coordinator.inference_slot_owns_job(job_b.id_) is True
    assert pm._recovery_coordinator.inference_slot_owns_job(job_a.id_) is False


async def test_dropped_result_retires_the_slots_dispatch_stamp() -> None:
    """Hardening: a result for an untracked (dropped) job retires the slot's in-flight stamp first.

    The in-flight-timestamp clears were hoisted above the job-not-found early-return, so a dropped result
    cannot leave ``current_inference_started_at`` set. Otherwise the slot would masquerade as a
    dispatch-in-flight owner (of its stale reference) and shield that job from the orphaned-job watchdog.
    """
    pm = make_testable_process_manager()

    slot = make_mock_process_info(3, model_name="m", state=HordeProcessState.WAITING_FOR_JOB)
    slot.last_control_flag = HordeControlFlag.START_INFERENCE
    slot.current_inference_started_at = time.time()
    pm._process_map[3] = slot

    # The job the slot was dispatched is no longer tracked: its result is dropped (not in jobs_lookup).
    dropped_job = make_job_pop_response(model="m")
    slot.last_job_referenced = dropped_job
    assert dropped_job.id_ is not None

    # While the stamp stands the slot looks like the job's dispatch-in-flight owner.
    assert pm._recovery_coordinator.inference_slot_owns_job(dropped_job.id_) is True

    msg = Mock(spec=HordeInferenceResultMessage)
    msg.process_id = 3
    msg.sdk_api_job_info = dropped_job
    await pm._message_dispatcher._handle_inference_result(msg)

    # The dropped result retired the stamp, so the slot can no longer phantom-own its stale reference.
    assert slot.current_inference_started_at is None
    assert pm._recovery_coordinator.inference_slot_owns_job(dropped_job.id_) is False
