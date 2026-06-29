"""RED reproduction: a deliberate RAM-reclaim process cycle is misread as a structural queue wedge.

Scenario: a sole-process worker (max_threads=1) finishes a heavy head and the allocator retains its
pages even after an explicit RAM unload. The failure path:

1. A heavy head (Flux.1-Schnell fp8) finishes inference, leaving its weights' allocator pages retained
   on the sole inference process even after a RAM unload.
2. The RAM budget refuses to preload the next head (a WAI-NSFW-illustrious-SDXL job) because the
   retained allocation makes "available" look too low, and (finding nothing else to reclaim) cycles
   that one idle process to return the pages to the OS (``_replace_stale_ram_unload_process``). This is
   a *deliberate, healthy* reclaim, flagged ``intentional_reclaim=True`` so it is not counted as a crash.
3. That leaves the worker momentarily with zero serving inference processes (the replacement is
   ``PROCESS_STARTING``) while pending SDXL jobs wait. The queue-deadlock detector flags the all-idle
   queue, and once the structural-wedge window (``_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS``, 20s) elapses the
   recovery supervisor declares a structural wedge, soft-resets the pools, and faults the backlog as
   "unservable (scheduler wedged with idle processes despite a healthy pool)". The horde counts those as
   dropped jobs, which can provoke forced maintenance.

The jobs were perfectly servable: they needed only the time for the freshly-cycled process to come up
and preload. The worker faulted its own backlog over a window it deliberately created.

Root cause: a worker-initiated RAM-reclaim process cycle has no recovery *grace*. The whole-card
residency establish/restore (``whole_card_residency_grace_active``) and the heavy-head load
(``heavy_head_load_grace_active``) both shield their deliberately-held all-idle windows from the
save-our-ship wedge assessment; the RAM-reclaim cycle, an equally deliberate, equally bounded hold,
does not, so its window is the one that trips the supervisor.

These tests assert the corrected behavior and are expected to FAIL (RED) against current code: while a
RAM-reclaim cycle is in its bounded grace window, ``_assess_wedge`` must not report a structural wedge,
and the still-servable backlog must not be faulted. The guard tests (``...still_escalates``) are
expected GREEN: a genuine sustained wedge with *no* recent reclaim must still escalate, so the grace
does not blind the supervisor to a real wedge.
"""

from __future__ import annotations

import time

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)

# The queue-deadlock flag has been set this long: past the 20s structural-wedge window.
_STRUCTURAL_WEDGE_AGE = 25.0

# Enough retained RAM to exceed the stale-RAM-unload replace threshold (1GB), so the budget's reclaim
# path chooses to cycle the slot rather than preload onto it.
_RETAINED_RAM_BYTES = 38 * 1024 * 1024 * 1024


def _stub_low_level_spawn(pm: HordeWorkerProcessManager) -> None:
    """Replace only the OS-level inference spawn with a state flip to a fresh ``PROCESS_STARTING`` slot.

    The scheduler's ``_replace_stale_ram_unload_process`` and the lifecycle's
    ``_replace_inference_process`` both run in full; only the real ``ctx.Pipe()`` / child launch is
    faked. This keeps the reclaim-cycle decision path (and whatever grace it records) exercised exactly
    as in production, while staying a torch/network-free unit test.
    """

    def fake_start(process_id: int, *, device_index: int = 0) -> None:
        slot = pm._process_map[process_id]
        slot.last_process_state = HordeProcessState.PROCESS_STARTING
        slot.loaded_horde_model_name = None

    pm._process_lifecycle._start_inference_process = fake_start  # type: ignore[method-assign]


def _stage_stale_ram_unload_slot(pm: HordeWorkerProcessManager, process_id: int) -> None:
    """Put an idle inference slot into the exact 'unloaded RAM but allocator kept the pages' state."""
    slot = make_mock_process_info(process_id, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    slot.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_RAM
    slot.ram_usage_bytes = _RETAINED_RAM_BYTES
    pm._process_map[process_id] = slot


async def _queue_pending_inference(pm: HordeWorkerProcessManager, models: list[str]) -> None:
    """Pop and track one pending-inference job per model (nothing marked in progress)."""
    for model in models:
        await track_popped_job_async(pm._job_tracker, make_job_pop_response(model=model))


def _latch_sustained_queue_deadlock(pm: HordeWorkerProcessManager) -> None:
    """Set the queue-deadlock flag as having persisted past the structural-wedge window."""
    pm._message_dispatcher._in_queue_deadlock = True
    pm._message_dispatcher._last_queue_deadlock_detected_time = time.time() - _STRUCTURAL_WEDGE_AGE


def _perform_ram_reclaim_cycle(pm: HordeWorkerProcessManager) -> None:
    """Drive the real RAM-budget reclaim cycle of the staged stale slot; assert it actually cycled."""
    cycled = pm._inference_scheduler._replace_stale_ram_unload_process()
    assert cycled is True, "precondition: the staged stale-RAM slot must be cycled by the reclaim path"


class TestRamReclaimCycleIsNotAStructuralWedge:
    """``_assess_wedge`` must not fire while a deliberate RAM-reclaim cycle is in its grace window."""

    async def test_two_job_backlog_after_reclaim_is_not_wedged(self) -> None:
        """Sole slot cycled for RAM, two SDXL jobs pending, sustained queue deadlock past the wedge window."""
        pm = make_testable_process_manager(max_threads=1)
        pm._state.last_job_pop_time = time.time() - 60
        _stub_low_level_spawn(pm)
        _stage_stale_ram_unload_slot(pm, process_id=1)
        await _queue_pending_inference(pm, ["WAI-NSFW-illustrious-SDXL", "WAI-NSFW-illustrious-SDXL"])

        _perform_ram_reclaim_cycle(pm)
        _latch_sustained_queue_deadlock(pm)

        # Sanity: the pool is not otherwise broken; the only signal is the deliberate reclaim window.
        assert pm._recovery_coordinator.is_inference_pool_unrecoverable() is False
        # RED: the cycled slot is mid-respawn by the worker's own deliberate reclaim, not a wedge.
        assert pm._recovery_coordinator.assess_wedge() is False

    async def test_single_head_after_reclaim_is_not_wedged(self) -> None:
        """Queue depth of one: a single pending head after a reclaim cycle is still not a wedge."""
        pm = make_testable_process_manager(max_threads=1)
        pm._state.last_job_pop_time = time.time() - 60
        _stub_low_level_spawn(pm)
        _stage_stale_ram_unload_slot(pm, process_id=1)
        await _queue_pending_inference(pm, ["AAM XL"])

        _perform_ram_reclaim_cycle(pm)
        _latch_sustained_queue_deadlock(pm)

        assert pm._recovery_coordinator.assess_wedge() is False

    async def test_mixed_model_backlog_after_reclaim_is_not_wedged(self) -> None:
        """A heterogeneous pending queue (the head plus differing models behind it) is still not a wedge."""
        pm = make_testable_process_manager(max_threads=1)
        pm._state.last_job_pop_time = time.time() - 60
        _stub_low_level_spawn(pm)
        _stage_stale_ram_unload_slot(pm, process_id=1)
        await _queue_pending_inference(pm, ["Flux.1-Schnell fp8 (Compact)", "AAM XL", "Realistic Vision"])

        _perform_ram_reclaim_cycle(pm)
        _latch_sustained_queue_deadlock(pm)

        assert pm._recovery_coordinator.assess_wedge() is False

    async def test_reclaim_with_idle_sibling_present_is_not_wedged(self) -> None:
        """Config variant: a healthy idle sibling exists alongside the cycled slot.

        The preload will land on whichever slot the scheduler picks once it is ready; the reclaim window
        is bounded and self-clearing either way, so it must not be assessed as a structural wedge.
        """
        pm = make_testable_process_manager(max_threads=2)
        pm._state.last_job_pop_time = time.time() - 60
        _stub_low_level_spawn(pm)
        _stage_stale_ram_unload_slot(pm, process_id=1)
        sibling = make_mock_process_info(2, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        pm._process_map[2] = sibling
        await _queue_pending_inference(pm, ["WAI-NSFW-illustrious-SDXL", "AAM XL"])

        _perform_ram_reclaim_cycle(pm)
        _latch_sustained_queue_deadlock(pm)

        assert pm._recovery_coordinator.assess_wedge() is False


class TestRamReclaimCycleGiveUpKeepsBacklog:
    """``_give_up_on_wedged_jobs`` must not fault a still-servable backlog during a reclaim window.

    The supervisor reaches give-up only after ``_assess_wedge`` reported a wedge; the primary fix is to
    keep ``_assess_wedge`` from firing during the reclaim grace (above). These tests pin the
    defense-in-depth at the layer that actually dropped the jobs in the live run: even if give-up is
    reached, a backlog whose only obstacle is the worker's own bounded reclaim cycle must be preserved,
    not faulted (the live run faulted both jobs here).
    """

    async def test_give_up_does_not_fault_backlog_after_reclaim(self) -> None:
        """Two pending SDXL jobs must survive give-up during the reclaim window."""
        pm = make_testable_process_manager(max_threads=1)
        pm._state.last_job_pop_time = time.time() - 60
        _stub_low_level_spawn(pm)
        _stage_stale_ram_unload_slot(pm, process_id=1)
        await _queue_pending_inference(pm, ["WAI-NSFW-illustrious-SDXL", "WAI-NSFW-illustrious-SDXL"])

        _perform_ram_reclaim_cycle(pm)
        _latch_sustained_queue_deadlock(pm)

        pending_before = len(list(pm._job_tracker.jobs_pending_inference))
        assert pending_before == 2
        pm._recovery_coordinator.give_up_on_wedged_jobs()

        # RED: the jobs are servable once the cycled slot comes up; the reclaim window must not drop them.
        assert len(list(pm._job_tracker.jobs_pending_inference)) == pending_before

    async def test_give_up_preserves_single_head_after_reclaim(self) -> None:
        """A lone pending head must likewise survive give-up during the reclaim window."""
        pm = make_testable_process_manager(max_threads=1)
        pm._state.last_job_pop_time = time.time() - 60
        _stub_low_level_spawn(pm)
        _stage_stale_ram_unload_slot(pm, process_id=1)
        await _queue_pending_inference(pm, ["AAM XL"])

        _perform_ram_reclaim_cycle(pm)
        _latch_sustained_queue_deadlock(pm)

        pm._recovery_coordinator.give_up_on_wedged_jobs()

        assert len(list(pm._job_tracker.jobs_pending_inference)) == 1


class TestRamReclaimGraceIsScoped:
    """Guards (expected GREEN): the reclaim grace must not blind the supervisor to a genuine wedge."""

    async def test_genuine_sustained_wedge_without_reclaim_still_escalates(self) -> None:
        """No reclaim in play: an all-idle slot with an unschedulable head still trips the wedge.

        This is the real wedge the supervisor exists to break. The reclaim grace must be scoped to an
        actual recent reclaim cycle, so this case is unaffected.
        """
        pm = make_testable_process_manager(max_threads=1)
        pm._state.last_job_pop_time = time.time() - 60
        idle = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        pm._process_map[1] = idle
        await _queue_pending_inference(pm, ["unschedulable"])

        _latch_sustained_queue_deadlock(pm)

        assert pm._recovery_coordinator.assess_wedge() is True

    async def test_give_up_faults_backlog_on_genuine_wedge_without_reclaim(self) -> None:
        """On a genuine wedge (no reclaim), give-up must still reissue the stuck head as before."""
        pm = make_testable_process_manager(max_threads=1)
        pm._state.last_job_pop_time = time.time() - 60
        idle = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        pm._process_map[1] = idle
        await _queue_pending_inference(pm, ["unschedulable"])

        _latch_sustained_queue_deadlock(pm)
        pm._recovery_coordinator.give_up_on_wedged_jobs()

        assert len(list(pm._job_tracker.jobs_pending_inference)) == 0

    async def test_inference_in_progress_after_reclaim_is_not_wedged(self) -> None:
        """Composes with the existing belt-and-braces: a live slot mid-inference is never a wedge.

        Here a sibling is actively running a job while the reclaim-cycled slot respawns; the worker is
        plainly making progress. This already holds via ``has_inference_in_progress`` and must keep
        holding once the reclaim grace is added.
        """
        pm = make_testable_process_manager(max_threads=2)
        pm._state.last_job_pop_time = time.time() - 60
        _stub_low_level_spawn(pm)
        _stage_stale_ram_unload_slot(pm, process_id=1)
        busy = make_mock_process_info(2, model_name="resident", state=HordeProcessState.INFERENCE_STARTING)
        pm._process_map[2] = busy

        live_job = make_job_pop_response(model="resident")
        await track_popped_job_async(pm._job_tracker, live_job)
        await pm._job_tracker.mark_inference_started(live_job)
        busy.last_job_referenced = live_job
        await _queue_pending_inference(pm, ["WAI-NSFW-illustrious-SDXL"])

        _perform_ram_reclaim_cycle(pm)
        _latch_sustained_queue_deadlock(pm)

        assert pm._recovery_coordinator.assess_wedge() is False
