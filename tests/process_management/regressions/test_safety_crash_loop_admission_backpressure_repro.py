"""Safety-process crash loop on a saturated card produces no inference-side admission backpressure.

A safety child that dies is reaped and respawned (:meth:`ProcessLifecycleManager._reap_if_crashed`). When
the card is saturated the device-free governor defers its start (:meth:`_defer_gpu_start`), so a safety child
that keeps crashing respawns straight back into a start that keeps deferring: the safety pool crash-loop
signal (:attr:`ProcessLifecycleManager.safety_pool_failing`) trips, yet nothing forces the inference side to
yield room so the safety pool can actually come up. The card stays full of inference work and safety never
starts.

RED contract: once the safety pool is crash-looping while its start is deferred on a saturated card,
inference admission must be throttled (or an equivalent card reclaim issued) so the safety pool can start.
Asserted at the admission seam: with the safety pool crash-looping and a saturated safety start pending, a
preload the VRAM arbiter would otherwise admit is instead held. This must fail today (admission ignores the
safety pool's state and proceeds).

Control: a single safety crash on a healthy card respawns cleanly, does not trip the crash-loop signal, and
leaves inference admission untouched.

Note for the reviewer: the honest seam for the backpressure is a design choice (throttle preload admission,
shed a resident inference model, or pause pops). This suite asserts it at the preload-admission seam because
that is where the scheduler already reads lifecycle state; if the fix surfaces the backpressure elsewhere, the
observable here should move with it.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.action_ledger import LedgerEventType
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import (
    SAFETY_CRASH_LOOP_MAX,
    ProcessLifecycleManager,
)
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from horde_worker_regen.process_management.resources.resource_budget import BudgetVerdict
from horde_worker_regen.process_management.resources.vram_arbiter import (
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _SAFETY_RECOVERY_HOLD_TTL_SECONDS,
    _WholeCardDemandOutcome,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)

# A saturated card's free-VRAM reading: below the hard floor, so the device-free governor defers a GPU start.
_SATURATED_FREE_MB = 100.0


def _saturate_safety_card(lifecycle: ProcessLifecycleManager) -> None:
    """Force the device-free governor to read the safety card as saturated, so a safety start defers."""
    lifecycle._device_free_mb_provider = lambda _device_index: _SATURATED_FREE_MB  # type: ignore[method-assign]
    lifecycle._device_governor_state_provider = lambda _device_index: GovernorState.SATURATED  # type: ignore[method-assign]


def _fitting_device_state() -> DeviceVramState:
    """A device state with ample device-free room, so the VRAM arbiter would admit an ordinary preload."""
    return DeviceVramState(
        total_vram_mb=24000.0,
        baseline_mb=1000.0,
        committed_vram_mb=2000.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        device_free_mb=21000.0,
    )


def _prime_admission_that_would_fit(
    pm: HordeWorkerProcessManager,
) -> tuple[ImageGenerateJobPopResponse, HordeProcessInfo]:
    """Stage a preload the scheduler would admit absent any safety backpressure; return (head, target).

    Installs a fitting VRAM cycle and clears the RAM verdict so the only remaining reason to hold the preload
    would be a deliberate safety-recovery backpressure, not resource scarcity.
    """
    scheduler = pm._inference_scheduler
    target = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    pm._process_map[1] = target

    arbiter = VramArbiter()
    arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: _fitting_device_state()}))
    scheduler._vram_arbiter = arbiter
    scheduler._decide_whole_card_demand = Mock(return_value=_WholeCardDemandOutcome.FALL_THROUGH)  # type: ignore[method-assign]
    scheduler._ram_budget.check_job = Mock(  # type: ignore[method-assign]
        return_value=BudgetVerdict(fits=True, predicted_mb=1000.0, available_mb=64000.0, reserve_mb=4096.0),
    )
    scheduler._measured_available_ram_mb = lambda: 64000.0  # type: ignore[method-assign]

    head = make_job_pop_response("head_model")
    return head, target


async def _stage_head(pm: HordeWorkerProcessManager, head: ImageGenerateJobPopResponse) -> None:
    """Track the head job as pending inference."""
    await track_popped_job_async(pm._job_tracker, head)


def _crashed_safety_process(process_id: int) -> HordeProcessInfo:
    """A safety HordeProcessInfo whose OS process has exited unexpectedly (a crash the reaper must catch)."""
    proc = make_mock_process_info(
        process_id,
        model_name=None,
        state=HordeProcessState.PROCESS_STARTING,
        process_type=HordeProcessType.SAFETY,
    )
    # ``mp_process`` is a Mock in tests; configure its liveness and exit code to model a native abort.
    proc.mp_process.is_alive.return_value = False  # pyrefly: ignore - Mock stand-in for the child handle
    proc.mp_process.exitcode = -6  # pyrefly: ignore - Mock stand-in; a native abort (SIGABRT), not a clean end
    return proc


def _drive_safety_crash_loop(lifecycle: ProcessLifecycleManager) -> None:
    """Trip the safety-pool crash-loop signal and leave a saturated safety start pending.

    A real reap of a crashed safety child exercises the recovery entry point; the recovery history is then
    advanced to the crash-loop threshold to stand in for the repeated reaps a sustained crash loop produces,
    and one real ``start_safety_processes`` under saturation records the pending, deferred safety start.
    """
    crashed = _crashed_safety_process(0)
    lifecycle._process_map[0] = crashed
    assert lifecycle._reap_if_crashed(crashed) is True

    now = time.time()
    lifecycle._safety_recovery_history = [now] * (SAFETY_CRASH_LOOP_MAX + 1)

    # A fresh start attempt under saturation records a deferred, pending safety GPU start.
    lifecycle._process_map.delete_safety_processes()
    lifecycle._safety_processes_ending = False
    lifecycle.start_safety_processes()


class TestSafetyCrashLoopBackpressure:
    """A crash-looping safety pool on a saturated card must create inference-side backpressure."""

    async def test_crash_loop_saturation_throttles_inference_admission(self) -> None:
        """A preload the arbiter would admit is held while the safety pool crash-loops on a saturated card.

        With the safety pool crash-looping and its start deferred for want of device headroom, admitting more
        inference work keeps the card full and the safety pool starved. The admission seam must hold the
        otherwise-fitting preload so the card can drain for the safety pool.
        """
        pm = make_testable_process_manager(safety_on_gpu=True, device_free_mb=_SATURATED_FREE_MB)
        lifecycle = pm._process_lifecycle
        _saturate_safety_card(lifecycle)
        head, target = _prime_admission_that_would_fit(pm)
        await _stage_head(pm, head)

        _drive_safety_crash_loop(lifecycle)

        # Preconditions: the safety pool is genuinely crash-looping and cannot start on the saturated card.
        assert lifecycle.safety_pool_failing is True
        assert lifecycle.has_pending_safety_starts() is True

        admitted = pm._inference_scheduler._admit_preload_under_budget(head, target, is_head_blocker=True)
        assert admitted is False, (
            "a preload the VRAM arbiter would admit must be held while the safety pool crash-loops on a "
            "saturated card, so the card can drain and the safety pool can start; admission proceeded instead"
        )

    async def test_single_safety_crash_on_healthy_card_respawns_without_inference_impact(self) -> None:
        """One safety crash on a healthy card respawns cleanly and leaves inference admission untouched (control).

        A lone crash is not a crash loop: the reaper recovers the safety process, the crash-loop signal stays
        clear, and a fitting preload is admitted exactly as it would be with a healthy safety pool.
        """
        pm = make_testable_process_manager(safety_on_gpu=True, device_free_mb=24000.0)
        lifecycle = pm._process_lifecycle
        head, target = _prime_admission_that_would_fit(pm)
        await _stage_head(pm, head)

        crashed = _crashed_safety_process(0)
        lifecycle._process_map[0] = crashed
        assert lifecycle._reap_if_crashed(crashed) is True

        assert lifecycle.safety_pool_failing is False
        admitted = pm._inference_scheduler._admit_preload_under_budget(head, target, is_head_blocker=True)
        assert admitted is True

    async def test_hold_releases_when_safety_pool_starts(self) -> None:
        """Once the deferred safety start is satisfied the hold releases and admission resumes (liveness).

        A permanent-hold backpressure would be a new wedge class. This proves the release edge: with the
        crash-loop hold engaged, satisfying the pending safety start (the card drained and safety came up)
        clears the pending-start signal, so the next admission is no longer held.
        """
        pm = make_testable_process_manager(safety_on_gpu=True, device_free_mb=_SATURATED_FREE_MB)
        lifecycle = pm._process_lifecycle
        _saturate_safety_card(lifecycle)
        head, target = _prime_admission_that_would_fit(pm)
        await _stage_head(pm, head)
        _drive_safety_crash_loop(lifecycle)

        scheduler = pm._inference_scheduler
        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is False
        assert scheduler._safety_recovery_hold_since != 0.0

        # The safety pool started: its deferred GPU start is satisfied, so the pending-start signal clears.
        lifecycle._pending_gpu_starts.clear()
        assert lifecycle.has_pending_safety_starts() is False

        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is True
        assert scheduler._safety_recovery_hold_since == 0.0

    async def test_hold_releases_at_ttl_when_safety_still_cannot_start(self) -> None:
        """The hold releases at its TTL even if safety never starts, so inference is not starved forever.

        If safety still cannot start after the bounded window, holding inference longer only starves the card
        without helping. The TTL cap releases the hold (logged loud elsewhere) and admission resumes; the
        crash-loop condition itself remains for the give-up machinery that already watches safety-pool health.
        """
        pm = make_testable_process_manager(safety_on_gpu=True, device_free_mb=_SATURATED_FREE_MB)
        lifecycle = pm._process_lifecycle
        _saturate_safety_card(lifecycle)
        head, target = _prime_admission_that_would_fit(pm)
        await _stage_head(pm, head)
        _drive_safety_crash_loop(lifecycle)

        scheduler = pm._inference_scheduler
        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is False

        # Age the hold past its TTL while the safety pool is still crash-looping and its start still pending.
        scheduler._safety_recovery_hold_since -= _SAFETY_RECOVERY_HOLD_TTL_SECONDS + 1.0
        assert lifecycle.safety_pool_failing is True
        assert lifecycle.has_pending_safety_starts() is True

        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is True
        assert scheduler._safety_recovery_hold_since == 0.0

    async def test_ttl_expiry_latches_so_a_persistent_failure_does_not_re_hold(self) -> None:
        """After a TTL expiry with the condition persisting, admission proceeds and does not re-hold.

        The latch guards the persistent-failure case the TTL exists for: without it the ``since == 0`` reset
        re-engages the hold on the very next evaluation, admitting at most one preload per TTL window forever
        and re-logging the CRITICAL each cycle. The expiry must fire once and then let inference run.
        """
        pm = make_testable_process_manager(safety_on_gpu=True, device_free_mb=_SATURATED_FREE_MB)
        lifecycle = pm._process_lifecycle
        _saturate_safety_card(lifecycle)
        head, target = _prime_admission_that_would_fit(pm)
        await _stage_head(pm, head)
        _drive_safety_crash_loop(lifecycle)

        scheduler = pm._inference_scheduler

        def _count(event_type: LedgerEventType) -> int:
            return sum(1 for e in lifecycle.action_ledger.recent(limit=500) if e.event_type == event_type)

        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is False
        engage_events_after_hold = _count(LedgerEventType.SAFETY_RECOVERY_HOLD_ENGAGED)
        assert engage_events_after_hold == 1

        # Age past the TTL: the hold expires once and latches the episode.
        scheduler._safety_recovery_hold_since -= _SAFETY_RECOVERY_HOLD_TTL_SECONDS + 1.0
        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is True
        assert scheduler._safety_recovery_hold_expired is True
        released_events = _count(LedgerEventType.SAFETY_RECOVERY_HOLD_RELEASED)

        # Every subsequent evaluation, with the condition still persisting, proceeds without re-holding and
        # accrues no further engage or release (CRITICAL) events.
        for _ in range(5):
            assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is True
            assert scheduler._safety_recovery_hold_since == 0.0
        assert _count(LedgerEventType.SAFETY_RECOVERY_HOLD_ENGAGED) == engage_events_after_hold
        assert _count(LedgerEventType.SAFETY_RECOVERY_HOLD_RELEASED) == released_events

    async def test_fresh_episode_holds_again_after_the_condition_clears(self) -> None:
        """Once the condition clears, a later crash-loop episode holds again (episode reset proof).

        The TTL latch must not permanently disable the hold. After the safety pool recovers (the release path
        runs and clears the latch), a fresh crash-loop on a saturated card is a new episode that engages the
        hold exactly as the first one did.
        """
        pm = make_testable_process_manager(safety_on_gpu=True, device_free_mb=_SATURATED_FREE_MB)
        lifecycle = pm._process_lifecycle
        _saturate_safety_card(lifecycle)
        head, target = _prime_admission_that_would_fit(pm)
        await _stage_head(pm, head)
        _drive_safety_crash_loop(lifecycle)

        scheduler = pm._inference_scheduler
        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is False
        scheduler._safety_recovery_hold_since -= _SAFETY_RECOVERY_HOLD_TTL_SECONDS + 1.0
        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is True
        assert scheduler._safety_recovery_hold_expired is True

        # The condition clears: the pending safety start is satisfied, so the release path runs and unlatches.
        lifecycle._pending_gpu_starts.clear()
        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is True
        assert scheduler._safety_recovery_hold_expired is False

        # A fresh crash-loop episode on the saturated card holds again, exactly like the first.
        _drive_safety_crash_loop(lifecycle)
        assert lifecycle.safety_pool_failing is True
        assert lifecycle.has_pending_safety_starts() is True
        assert scheduler._admit_preload_under_budget(head, target, is_head_blocker=True) is False
        assert scheduler._safety_recovery_hold_since != 0.0
