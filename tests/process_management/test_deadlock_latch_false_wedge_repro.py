"""RED reproduction: a latched queue-deadlock flag triggers a save-our-ship soft reset on a healthy worker.

Observed in a live session on a contended 16GB ``max_threads=1`` worker. The sequence:

1. A brief all-idle gap between jobs set the queue-deadlock flag (``_in_queue_deadlock``) -- the normal
   transient state while the scheduler picks the next model.
2. The scheduler then dispatched the head (a heavy SDXL job with LoRAs + post-processing) onto a healthy
   slot, which began running it. The queue condition was resolved: a job was in progress on a live slot.
3. ``detect_deadlock`` could not clear the queue-deadlock flag, though, because its clear branch also
   requires ``num_starting_processes() == 0`` -- and a sibling slot, cycled moments earlier to reclaim
   RAM, was still in ``PROCESS_STARTING`` (slow to spin up under host contention). The flag latched.
4. ~20s later ``indicates_structural_wedge()`` flipped true on the stale flag, ``_assess_wedge()``
   returned true, and the recovery supervisor performed a save-our-ship soft reset: it rebuilt the
   *entire* inference pool, tearing down the one slot that was healthily mid-inference (its last
   heartbeat 1.6s earlier) and faulting its in-flight job as "crashed or hung" (``exitcode=None``).

So a worker that was making genuine progress was soft-reset and a good job faulted, purely because the
queue-deadlock flag could not clear while an unrelated sibling was slow to start. The flag's "all-idle
queue is stuck" meaning is false the instant a job is in progress on a live slot; a starting sibling
must not keep it latched, and the worker-level wedge assessment must not fire while real inference is
advancing.

These tests assert the corrected behavior and are expected to FAIL (RED) against the current code.
``test_sustained_all_idle_queue_deadlock_is_still_a_wedge`` is the guard (expected GREEN): a genuine
sustained queue deadlock with every slot idle must still escalate, so the fix does not blind the
recovery supervisor to a real wedge.
"""

from __future__ import annotations

import time

from horde_worker_regen.process_management.messages import HordeProcessState

from .conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)

_STRUCTURAL_WEDGE_AGE = 25.0
"""Seconds the queue-deadlock flag has been set: past the 20s structural-wedge window, as in the live run."""


async def test_queue_deadlock_clears_when_job_in_progress_despite_starting_sibling() -> None:
    """detect_deadlock must clear the queue-deadlock flag once a job is in progress, even if a slot starts.

    The clear branch is currently gated on ``num_starting_processes() == 0``, so a sibling slow to spin up
    (a routine RAM-reclaim re-spawn on a contended host) latches a flag whose precondition -- an all-idle,
    stuck queue -- no longer holds.
    """
    pm = make_testable_process_manager()
    pm._state.last_job_pop_time = time.time() - 60  # the last pop is not recent; detection is live

    busy = make_mock_process_info(1, model_name="resident", state=HordeProcessState.INFERENCE_STARTING)
    pm._process_map[1] = busy
    starting = make_mock_process_info(2, model_name=None, state=HordeProcessState.PROCESS_STARTING)
    pm._process_map[2] = starting

    job = make_job_pop_response(model="resident")
    await track_popped_job_async(pm._job_tracker, job)
    await pm._job_tracker.mark_inference_started(job)
    busy.last_job_referenced = job

    # A queue deadlock was detected during an earlier all-idle gap and has not been cleared since.
    pm._message_dispatcher._in_queue_deadlock = True
    pm._message_dispatcher._last_queue_deadlock_detected_time = time.time() - _STRUCTURAL_WEDGE_AGE

    pm.detect_deadlock()

    # RED: the queue is no longer deadlocked (a job is in progress on a live slot), so the flag must clear
    # rather than stay latched by the starting sibling.
    assert pm._message_dispatcher.get_deadlock_snapshot().in_queue_deadlock is False


async def test_worker_with_job_in_progress_is_not_assessed_as_wedged() -> None:
    """A worker actively running a job on a healthy slot must not be assessed as structurally wedged.

    Even granting the latched flag, ``_assess_wedge`` must not green-light a save-our-ship soft reset
    while real inference is advancing -- the reset rebuilds the whole pool and faults the in-flight job.
    """
    pm = make_testable_process_manager()
    pm._state.last_job_pop_time = time.time() - 60

    busy = make_mock_process_info(1, model_name="resident", state=HordeProcessState.INFERENCE_STARTING)
    pm._process_map[1] = busy
    starting = make_mock_process_info(2, model_name=None, state=HordeProcessState.PROCESS_STARTING)
    pm._process_map[2] = starting

    job = make_job_pop_response(model="resident")
    await track_popped_job_async(pm._job_tracker, job)
    await pm._job_tracker.mark_inference_started(job)
    busy.last_job_referenced = job
    assert job.id_ is not None
    assert job in pm._job_tracker.jobs_in_progress

    pm._message_dispatcher._in_queue_deadlock = True
    pm._message_dispatcher._last_queue_deadlock_detected_time = time.time() - _STRUCTURAL_WEDGE_AGE
    pm.detect_deadlock()

    # RED: a job is in progress on a live, healthy slot, so the worker is not wedged.
    assert pm._assess_wedge() is False


async def test_sustained_all_idle_queue_deadlock_is_still_a_wedge() -> None:
    """Guard (expected GREEN): a real sustained all-idle queue deadlock must still escalate to a wedge.

    Nothing is in progress, every slot is idle, and the head's model cannot be placed; the flag has been
    set well past the structural-wedge window. This is the genuine wedge the recovery supervisor exists
    to break, and the dispatch-aware fix must leave it intact.
    """
    pm = make_testable_process_manager()
    pm._state.last_job_pop_time = time.time() - 60

    idle = make_mock_process_info(1, model_name="resident", state=HordeProcessState.WAITING_FOR_JOB)
    pm._process_map[1] = idle

    head = make_job_pop_response(model="unschedulable")
    await track_popped_job_async(pm._job_tracker, head)

    pm.detect_deadlock()
    pm._message_dispatcher._last_queue_deadlock_detected_time = time.time() - _STRUCTURAL_WEDGE_AGE

    assert pm._message_dispatcher.get_deadlock_snapshot().indicates_structural_wedge() is True
    assert pm._assess_wedge() is True
