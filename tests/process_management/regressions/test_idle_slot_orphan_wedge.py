"""Reproduction of a permanent wedge caused by an idle slot shielding an orphaned in-progress job.

After a burst of process recoveries the worker can stop dispatching entirely: jobs sit ``pending
start`` forever while inference processes sit in ``PRELOADED_MODEL`` holding the exact models those
pending jobs need, and the worker never recovers on its own.

The mechanism:

* A recovery storm strands one job in ``jobs_in_progress``: its inference result is dropped by the
  launch-identifier mismatch guard while its slot is being replaced, so no result ever moves it out of
  progress.
* The slot that last ran it returns to ``WAITING_FOR_JOB`` but still references it via
  ``last_job_referenced`` (that reference is not cleared when a job completes or the slot goes idle).
* ``_inference_slot_owns_job`` treats *any* alive slot whose ``last_job_referenced`` matches as the
  owner, even an idle one that will never produce a result. So the orphaned-in-progress-job watchdog
  considers the job owned and never punts it.
* With ``max_threads=1`` and the GPU sampling lease off, ``_max_jobs_in_progress_allowed`` is 1. The
  phantom in-progress job makes ``jobs_in_progress (1) >= cap (1)`` true, so
  ``get_next_job_and_process`` returns ``None`` at the cap gate on every cycle. Dispatch wedges.

The fix belongs in the watchdog's ownership test: a slot only *owns* an in-progress job while it is
actually working it (i.e. it cannot accept new work). An idle slot with a stale reference must not
shield the job. Once the watchdog punts the phantom, ``jobs_in_progress`` drops to 0 and the cap gate
re-opens, so the ready preloaded head dispatches.

This is distinct from the earlier orphaned-job wedge (``test_orphaned_job_wedge.py``), where the
replacement faulted the *wrong* job. Here the per-slot recovery and the watchdog both run correctly;
the watchdog's ownership predicate is simply too loose.
"""

from __future__ import annotations

import time

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobStage
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)


class TestInferenceSlotOwnership:
    """A slot owning a job must mean it is actively working it, not merely that it once referenced it.

    The orphan watchdog relies on this distinction to decide what to punt.
    """

    async def test_idle_waiting_slot_with_stale_reference_does_not_own_job(self) -> None:
        """An idle ``WAITING_FOR_JOB`` slot holding a stale reference is not the owner (the wedge core).

        This is process 2 in the incident: it finished/returned to idle but ``last_job_referenced`` still
        points at a job that is somehow still in progress. It will never produce a result, so it cannot be
        treated as the owner, otherwise the watchdog leaves the job pinned forever.
        """
        pm = make_testable_process_manager()
        idle_slot = make_mock_process_info(2, model_name="AlbedoBase XL 3.1", state=HordeProcessState.WAITING_FOR_JOB)
        pm._process_map[2] = idle_slot

        job = make_job_pop_response(model="AlbedoBase XL 3.1")
        await track_popped_job_async(pm._job_tracker, job)
        await pm._job_tracker.mark_inference_started(job)
        idle_slot.last_job_referenced = job
        assert job.id_ is not None

        assert pm._inference_slot_owns_job(job.id_) is False

    async def test_idle_preloaded_slot_with_stale_reference_does_not_own_job(self) -> None:
        """A ``PRELOADED_MODEL`` slot (ready for its *next* job) is idle and cannot own an in-progress job.

        Processes 1 and 3 in the incident were ``PRELOADED_MODEL``: ready to start, not running anything.
        ``PRELOADED_MODEL`` is a ``can_accept_job`` state, so such a slot must not be reported as an owner.
        """
        pm = make_testable_process_manager()
        preloaded_slot = make_mock_process_info(
            1,
            model_name="WAI-NSFW-illustrious-SDXL",
            state=HordeProcessState.PRELOADED_MODEL,
        )
        pm._process_map[1] = preloaded_slot

        job = make_job_pop_response(model="WAI-NSFW-illustrious-SDXL")
        await track_popped_job_async(pm._job_tracker, job)
        await pm._job_tracker.mark_inference_started(job)
        preloaded_slot.last_job_referenced = job
        assert job.id_ is not None

        assert pm._inference_slot_owns_job(job.id_) is False

    async def test_busy_slot_running_job_owns_it(self) -> None:
        """A slot actively sampling (``INFERENCE_STARTING``) genuinely owns its job; the fix must keep this."""
        pm = make_testable_process_manager()
        busy_slot = make_mock_process_info(
            1,
            model_name="stable_diffusion",
            state=HordeProcessState.INFERENCE_STARTING,
        )
        pm._process_map[1] = busy_slot

        job = make_job_pop_response(model="stable_diffusion")
        await track_popped_job_async(pm._job_tracker, job)
        await pm._job_tracker.mark_inference_started(job)
        busy_slot.last_job_referenced = job
        assert job.id_ is not None

        assert pm._inference_slot_owns_job(job.id_) is True


class TestWatchdogPuntsIdleShieldedOrphan:
    """The orphan watchdog must punt a job whose only "owner" is an idle slot, so the queue can drain."""

    async def test_watchdog_punts_job_shielded_only_by_idle_waiting_slot(self) -> None:
        """The exact wedge: an in-progress job referenced only by an idle ``WAITING_FOR_JOB`` slot.

        If the watchdog treats the idle slot as the owner, it deletes the job's grace-clock entry every tick
        and never punts it. Instead the job must be recognised as orphaned and punted once the grace window
        elapses, so it stops pinning the in-progress count.
        """
        pm = make_testable_process_manager()
        idle_slot = make_mock_process_info(2, model_name="AlbedoBase XL 3.1", state=HordeProcessState.WAITING_FOR_JOB)
        pm._process_map[2] = idle_slot

        job = make_job_pop_response(model="AlbedoBase XL 3.1")
        await track_popped_job_async(pm._job_tracker, job)
        await pm._job_tracker.mark_inference_started(job)
        idle_slot.last_job_referenced = job
        assert job.id_ is not None

        # Backdate the grace clock so a single reconcile pass is past the window. With the bug the entry
        # is pruned before it can be acted on (the job is wrongly "owned"), so the backdate is irrelevant
        # and the job stays pinned; that divergence is what makes this RED.
        pm._orphan_in_progress_since[job.id_] = time.time() - (pm._ORPHAN_IN_PROGRESS_GRACE_SECONDS + 1)
        pm._reconcile_orphaned_in_progress_jobs()

        assert pm._job_tracker.get_stage(job.id_) != JobStage.INFERENCE_IN_PROGRESS
        assert len(pm._orphan_punt_history) == 1

    async def test_watchdog_punts_job_shielded_only_by_preloaded_slot(self) -> None:
        """Variation: the shielding slot is ``PRELOADED_MODEL`` (processes 1 and 3 in the incident)."""
        pm = make_testable_process_manager()
        preloaded_slot = make_mock_process_info(
            1,
            model_name="ChilloutMix",
            state=HordeProcessState.PRELOADED_MODEL,
        )
        pm._process_map[1] = preloaded_slot

        job = make_job_pop_response(model="ChilloutMix")
        await track_popped_job_async(pm._job_tracker, job)
        await pm._job_tracker.mark_inference_started(job)
        preloaded_slot.last_job_referenced = job
        assert job.id_ is not None

        pm._orphan_in_progress_since[job.id_] = time.time() - (pm._ORPHAN_IN_PROGRESS_GRACE_SECONDS + 1)
        pm._reconcile_orphaned_in_progress_jobs()

        assert pm._job_tracker.get_stage(job.id_) != JobStage.INFERENCE_IN_PROGRESS

    async def test_watchdog_still_protects_running_job_with_an_extra_stale_idle_reference(self) -> None:
        """Guard: a job genuinely running on a busy slot must survive even if an idle slot also references it.

        The fix must not over-punt. When one slot is actively sampling the job (real owner) and a second,
        idle slot happens to hold a stale reference to the same job, the job is owned and must be left
        alone.
        """
        pm = make_testable_process_manager()
        busy_owner = make_mock_process_info(
            1,
            model_name="stable_diffusion",
            state=HordeProcessState.INFERENCE_STARTING,
        )
        stale_idle = make_mock_process_info(2, model_name="stable_diffusion", state=HordeProcessState.WAITING_FOR_JOB)
        pm._process_map[1] = busy_owner
        pm._process_map[2] = stale_idle

        job = make_job_pop_response(model="stable_diffusion")
        await track_popped_job_async(pm._job_tracker, job)
        await pm._job_tracker.mark_inference_started(job)
        busy_owner.last_job_referenced = job
        stale_idle.last_job_referenced = job
        assert job.id_ is not None

        pm._orphan_in_progress_since[job.id_] = time.time() - (pm._ORPHAN_IN_PROGRESS_GRACE_SECONDS + 1)
        pm._reconcile_orphaned_in_progress_jobs()

        assert pm._job_tracker.get_stage(job.id_) == JobStage.INFERENCE_IN_PROGRESS
        assert pm._orphan_punt_history == []


class TestPhantomInProgressJobWedgesDispatch:
    """End-to-end: the phantom in-progress job blocks all dispatch until the watchdog clears it.

    The process manager shares one ``job_tracker`` and ``process_map`` between the orphan watchdog and the
    inference scheduler, so these exercise the real interaction that produced the incident.
    """

    async def test_phantom_in_progress_job_wedges_dispatch_until_watchdog_clears_it(self) -> None:
        """Reconstructs the wedge and proves the watchdog fix re-opens dispatch.

        With ``max_threads=1`` the in-progress cap is 1. A phantom in-progress job (owned only by an idle
        slot) holds that single slot, so the scheduler refuses to dispatch the ready, preloaded head. Before
        the fix the watchdog cannot clear the phantom, so dispatch stays wedged; after the fix the watchdog
        punts it, the cap re-opens, and the head dispatches.
        """
        pm = make_testable_process_manager(max_threads=1)

        # The ready head: its model is resident and preloaded on process 1, exactly the
        # PRELOADED_MODEL-with-a-matching-pending-job state observed during the wedge.
        ready_head_slot = make_mock_process_info(
            1,
            model_name="WAI-NSFW-illustrious-SDXL",
            state=HordeProcessState.PRELOADED_MODEL,
        )
        # The slot stranded by the recovery storm: idle, but still referencing a job stuck in progress.
        stranded_slot = make_mock_process_info(
            2,
            model_name="AlbedoBase XL 3.1",
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        pm._process_map[1] = ready_head_slot
        pm._process_map[2] = stranded_slot

        head_job = make_job_pop_response(model="WAI-NSFW-illustrious-SDXL")
        await track_popped_job_async(pm._job_tracker, head_job)

        phantom_job = make_job_pop_response(model="AlbedoBase XL 3.1")
        await track_popped_job_async(pm._job_tracker, phantom_job)
        await pm._job_tracker.mark_inference_started(phantom_job)
        stranded_slot.last_job_referenced = phantom_job
        assert head_job.id_ is not None
        assert phantom_job.id_ is not None

        # Symptom: the in-progress cap (1) is fully consumed by the phantom, so the ready head cannot be
        # dispatched even though its process is preloaded and idle-capable.
        assert pm._inference_scheduler._max_jobs_in_progress_allowed(0) == 1
        assert await pm._inference_scheduler.get_next_job_and_process() is None

        # The watchdog runs every control-loop tick; advance its grace clock and let it act.
        pm._orphan_in_progress_since[phantom_job.id_] = time.time() - (pm._ORPHAN_IN_PROGRESS_GRACE_SECONDS + 1)
        pm._reconcile_orphaned_in_progress_jobs()

        # The phantom is no longer pinning the in-progress count...
        assert pm._job_tracker.get_stage(phantom_job.id_) != JobStage.INFERENCE_IN_PROGRESS
        assert len(pm._job_tracker.jobs_in_progress) == 0

        # ...so dispatch is unblocked: the ready, preloaded head can finally be scheduled.
        next_up = await pm._inference_scheduler.get_next_job_and_process()
        assert next_up is not None
        assert next_up.next_job.id_ == head_job.id_
        assert next_up.process_with_model.process_id == 1

    async def test_ready_preloaded_head_dispatches_when_no_phantom_present(self) -> None:
        """Control: the identical state without the phantom dispatches immediately.

        This isolates the phantom in-progress job as the active variable: the cap math, the preloaded
        process, and the queue are otherwise healthy, so a ready head is scheduled at once. It passes both
        before and after the fix, proving the wedge is caused by the un-punted phantom rather than the cap.
        """
        pm = make_testable_process_manager(max_threads=1)

        ready_head_slot = make_mock_process_info(
            1,
            model_name="WAI-NSFW-illustrious-SDXL",
            state=HordeProcessState.PRELOADED_MODEL,
        )
        pm._process_map[1] = ready_head_slot

        head_job = make_job_pop_response(model="WAI-NSFW-illustrious-SDXL")
        await track_popped_job_async(pm._job_tracker, head_job)
        assert head_job.id_ is not None

        next_up = await pm._inference_scheduler.get_next_job_and_process()
        assert next_up is not None
        assert next_up.next_job.id_ == head_job.id_
        assert next_up.process_with_model.process_id == 1
