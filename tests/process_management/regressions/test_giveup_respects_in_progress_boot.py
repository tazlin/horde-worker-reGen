"""Give-up must not fault servable work while a recovery-initiated pool rebuild is still booting.

The save-our-ship escalation gates give-up on pool readiness so that a just-rebuilt pool's boot window
(replacement children importing torch, no lane accepting yet) is not mistaken for an unrecoverable wedge.
A bounded boot allowance exists as a backstop for a pool that never comes up. The invariant these tests
encode is that the allowance must not fire while a rebuild is *demonstrably still progressing* (replacement
processes alive and in ``PROCESS_STARTING``): give-up merely on a fixed elapsed allowance faults jobs the
finishing boot then serves. Give-up must still fire, within bounded time, when a boot is genuinely dead or
pathologically hung (that direction is covered by the controls here and by the exploratory matrix).

The coordinator seam (:meth:`run_recovery_supervisor`) is exercised throughout: it derives ``is_wedged``
and ``pool_ready`` from real process/dispatcher state, so it reproduces the defect regardless of whether
the eventual remedy lands in the wedge derivation or in the escalation policy.
"""

from __future__ import annotations

import time

import pytest

from horde_worker_regen.process_management.ipc.action_ledger import LedgerEvent, LedgerEventType
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.recovery_supervisor import RecoveryAction, RecoverySupervisor
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)


class _FakeClock:
    """A monotonic clock the test advances explicitly, so escalation timing is deterministic."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# A realistic boot allowance far smaller than the boot windows the reproductions drive, so a boot that
# outlasts the allowance (the observed real replacement boots did) crosses it inside the test timeline.
_BOOT_ALLOWANCE = 6.0
_POOL_READY_GRACE = 3.0


def _install_supervisor(
    pm: HordeWorkerProcessManager,
    clock: _FakeClock,
    **overrides: float | int,
) -> RecoverySupervisor:
    """Swap in a supervisor on the fake clock with tight, legible escalation windows."""
    params: dict[str, float | int] = {
        "wedge_grace_seconds": 1,
        "reset_interval_seconds": 1,
        "max_soft_resets": 1,
        "pool_ready_grace_seconds": _POOL_READY_GRACE,
        "boot_allowance_seconds": _BOOT_ALLOWANCE,
        "give_up_cooldown_seconds": 5,
        "max_give_up_cycles": 2,
        "clean_streak_seconds": 100,
    }
    params.update(overrides)
    supervisor = RecoverySupervisor(clock=clock, **params)  # type: ignore[arg-type]
    pm._recovery_coordinator.recovery_supervisor = supervisor
    return supervisor


async def _latch_structural_queue_wedge(pm: HordeWorkerProcessManager, *, model: str = "resident") -> None:
    """Latch a sustained queue deadlock over an idle, model-resident inference slot (a servable head)."""
    proc = make_mock_process_info(0, model_name=model, state=HordeProcessState.WAITING_FOR_JOB)
    pm._process_map[0] = proc
    await track_popped_job_async(pm._job_tracker, make_job_pop_response(model=model))

    dispatcher = pm._message_dispatcher
    dispatcher._in_queue_deadlock = True
    dispatcher._queue_deadlock_model = model
    # Backdate the deadlock so it reads as a sustained structural wedge, not the transient between-jobs gap.
    dispatcher._last_queue_deadlock_detected_time = time.time() - 60


def _abandoned_records(pm: HordeWorkerProcessManager) -> list[LedgerEvent]:
    """All RECOVERY_ABANDONED events currently in the action ledger."""
    return [
        event
        for event in pm._recovery_coordinator._action_ledger.recent(limit=1000)
        if event.event_type == LedgerEventType.RECOVERY_ABANDONED
    ]


def _disable_pp_reclaim_yield(pm: HordeWorkerProcessManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise the post-processing-reclaim deferral so a give-up is never masked as a yield.

    That remedy is a different give-up-deferral path (a wedged head parked behind resident PP weights). These
    tests isolate the boot-window gate, so any give-up the escalation reaches must act, not yield.
    """
    monkeypatch.setattr(pm._recovery_coordinator, "_give_up_yields_to_pp_reclaim", lambda: False)


def _spy_give_up(pm: HordeWorkerProcessManager, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Record every ``give_up_on_wedged_jobs`` call while still running the real implementation."""
    calls: list[dict[str, object]] = []
    real_give_up = pm._recovery_coordinator.give_up_on_wedged_jobs

    def _record(**kwargs: object) -> None:
        calls.append(kwargs)
        real_give_up(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pm._recovery_coordinator, "give_up_on_wedged_jobs", _record)
    return calls


async def test_boot_window_past_allowance_does_not_fault_servable_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rebuild whose boot outlasts the fixed allowance must not have its servable head faulted.

    A structural queue wedge soft-resets the pool; the replacement slot then sits ``PROCESS_STARTING`` and
    alive for a window longer than the boot allowance (a merely slow, still-healthy boot). Give-up must be
    held off for the whole window: the allowance elapsing over a live, still-booting pool is not evidence the
    pool is unrecoverable, and faulting here drops the very job the finishing boot is about to run.
    """
    pm = make_testable_process_manager()
    clock = _FakeClock()
    _install_supervisor(pm, clock)
    await _latch_structural_queue_wedge(pm)
    _disable_pp_reclaim_yield(pm, monkeypatch)
    # A terminal give-up would arm the real force-kill backstop thread and write an abort sentinel file,
    # both of which outlive the test and kill the interpreter later in the session; neutralise it.
    monkeypatch.setattr(pm, "_abort", lambda: None)
    proc = pm._process_map[0]

    monkeypatch.setattr(
        pm._process_lifecycle,
        "rebuild_inference_pool",
        lambda *, reason: setattr(proc, "last_process_state", HordeProcessState.PROCESS_STARTING),
    )
    monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)
    give_up_calls = _spy_give_up(pm, monkeypatch)

    assert pm._recovery_coordinator.assess_wedge() is True  # precondition: the wedge is real

    # Drive well past the boot allowance and its ready-grace with the slot alive and PROCESS_STARTING.
    for _ in range(int(_BOOT_ALLOWANCE + _POOL_READY_GRACE) * 2 + 10):
        clock.advance(1)
        pm._recovery_coordinator.run_recovery_supervisor()

    assert proc.last_process_state is HordeProcessState.PROCESS_STARTING  # still a live, booting slot
    assert proc.is_process_alive() is True  # the boot is progressing, not dead
    assert give_up_calls == []  # give-up was held off for the whole boot window
    assert _abandoned_records(pm) == []
    assert len(list(pm._job_tracker.jobs_pending_inference)) == 1  # the head was never faulted


async def test_head_survives_and_is_served_when_boot_completes_after_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A head queued behind a slow rebuild survives to be served once the boot completes past the allowance.

    The complement of the fault-prevention test: after a boot that outlasts the allowance finishes and the
    slot reaches an accepting state (the wedge clears), the queued head is still present to be dispatched,
    and the episode closes with no give-up and no abandonment record.
    """
    pm = make_testable_process_manager()
    clock = _FakeClock()
    _install_supervisor(pm, clock, clean_streak_seconds=4)
    await _latch_structural_queue_wedge(pm)
    _disable_pp_reclaim_yield(pm, monkeypatch)
    monkeypatch.setattr(pm, "_abort", lambda: None)  # keep the force-kill backstop out of the test interpreter
    proc = pm._process_map[0]

    monkeypatch.setattr(
        pm._process_lifecycle,
        "rebuild_inference_pool",
        lambda *, reason: setattr(proc, "last_process_state", HordeProcessState.PROCESS_STARTING),
    )
    monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)
    give_up_calls = _spy_give_up(pm, monkeypatch)

    # A slow boot that runs well past the allowance while still alive.
    for _ in range(int(_BOOT_ALLOWANCE) * 2 + 4):
        clock.advance(1)
        pm._recovery_coordinator.run_recovery_supervisor()

    assert len(list(pm._job_tracker.jobs_pending_inference)) == 1  # head still queued behind the boot

    # The replacement slot reaches an accepting state and the wedge clears: the head is served, not faulted.
    proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
    pm._message_dispatcher._in_queue_deadlock = False
    pm._job_tracker._total_num_completed_jobs += 1  # the freshly ready lane makes real forward progress
    for _ in range(6):
        clock.advance(1)
        pm._recovery_coordinator.run_recovery_supervisor()

    assert give_up_calls == []
    assert _abandoned_records(pm) == []
    assert len(list(pm._job_tracker.jobs_pending_inference)) == 1  # survived to be dispatched


async def test_structural_wedge_over_booting_pool_does_not_reach_give_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """An elapsed-time structural-wedge signal over a visibly booting pool must not drive give-up.

    Isolates the structural-wedge / boot-allowance interaction without a soft reset (``max_soft_resets=0``):
    the deadlock signal qualifies purely on elapsed time while the replacement slot is alive and
    ``PROCESS_STARTING`` (``num_starting_processes() > 0``). The give-up decision must account for the
    in-progress boot rather than treating the elapsed wedge as proof of an unrecoverable pool.
    """
    pm = make_testable_process_manager()
    clock = _FakeClock()
    _install_supervisor(pm, clock, max_soft_resets=0)
    await _latch_structural_queue_wedge(pm)
    _disable_pp_reclaim_yield(pm, monkeypatch)
    monkeypatch.setattr(pm, "_abort", lambda: None)  # keep the force-kill backstop out of the test interpreter

    # The replacement child is alive but still booting (no soft reset is taken with a zero reset budget).
    proc = pm._process_map[0]
    proc.last_process_state = HordeProcessState.PROCESS_STARTING
    give_up_calls = _spy_give_up(pm, monkeypatch)

    # Root cause: the structural-wedge signal is purely elapsed-time and the pool is demonstrably booting.
    snapshot = pm._message_dispatcher.get_deadlock_snapshot()
    assert snapshot.indicates_structural_wedge() is True
    assert pm._process_map.num_starting_processes() == 1
    assert pm._recovery_coordinator.is_inference_pool_ready() is False

    for _ in range(int(_BOOT_ALLOWANCE + _POOL_READY_GRACE) * 2 + 6):
        clock.advance(1)
        pm._recovery_coordinator.run_recovery_supervisor()

    assert give_up_calls == []  # a booting pool is not an unrecoverable wedge
    assert _abandoned_records(pm) == []
    assert len(list(pm._job_tracker.jobs_pending_inference)) == 1


async def test_ready_pool_sustained_wedge_still_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuinely ready pool with a wedge sustained past the grace still reissues the servable head.

    The safety valve must not be dulled by the boot-window hold: once the rebuilt pool is accepting again and
    the wedge persists past the ready-grace, exactly one give-up faults the head so the horde reissues it, and
    the pool is not abandoned on this first give-up (it has live capacity).
    """
    pm = make_testable_process_manager()
    clock = _FakeClock()
    _install_supervisor(pm, clock)
    await _latch_structural_queue_wedge(pm)
    _disable_pp_reclaim_yield(pm, monkeypatch)

    aborted = {"called": False}
    monkeypatch.setattr(pm, "_abort", lambda: aborted.__setitem__("called", True))

    proc = pm._process_map[0]
    # Emulate an instant, healthy boot: the rebuilt slot is immediately accepting again (pool ready).
    monkeypatch.setattr(
        pm._process_lifecycle,
        "rebuild_inference_pool",
        lambda *, reason: setattr(proc, "last_process_state", HordeProcessState.WAITING_FOR_JOB),
    )
    monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)

    for _ in range(40):
        clock.advance(1)
        pm._recovery_coordinator.run_recovery_supervisor()
        if _abandoned_records(pm):
            break

    records = _abandoned_records(pm)
    assert len(records) == 1  # the valve fired
    assert records[0].detail["jobs_faulted"] == 1  # the servable head was reissued
    assert records[0].detail["terminal"] is False
    assert len(list(pm._job_tracker.jobs_pending_inference)) == 0
    assert aborted["called"] is False  # a live-capacity pool is not abandoned on the first give-up


async def test_boot_completes_within_allowance_no_give_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rebuild that reaches an accepting state within the allowance recovers with no give-up.

    The healthy fast-boot path: the replacement slot boots and starts accepting inside the boot allowance, the
    wedge clears, and the episode closes with zero faulting and zero abandonment records.
    """
    pm = make_testable_process_manager()
    clock = _FakeClock()
    _install_supervisor(pm, clock, clean_streak_seconds=4)
    await _latch_structural_queue_wedge(pm)
    _disable_pp_reclaim_yield(pm, monkeypatch)
    proc = pm._process_map[0]

    monkeypatch.setattr(
        pm._process_lifecycle,
        "rebuild_inference_pool",
        lambda *, reason: setattr(proc, "last_process_state", HordeProcessState.PROCESS_STARTING),
    )
    monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)
    give_up_calls = _spy_give_up(pm, monkeypatch)

    # A short boot: a couple of ticks starting, comfortably inside the allowance.
    for _ in range(2):
        clock.advance(1)
        pm._recovery_coordinator.run_recovery_supervisor()

    proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
    pm._message_dispatcher._in_queue_deadlock = False
    pm._job_tracker._total_num_completed_jobs += 1
    for _ in range(6):
        clock.advance(1)
        pm._recovery_coordinator.run_recovery_supervisor()

    assert give_up_calls == []
    assert _abandoned_records(pm) == []
    assert len(list(pm._job_tracker.jobs_pending_inference)) == 1


async def test_recovery_granted_retry_spared_by_first_give_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """A retry a same-cycle rebuild granted is left queued through the first (non-terminal) give-up.

    A non-terminal give-up must not terminally fault the very retry recovery just granted before the rebuilt
    pool has had a chance to run it; the job stays pending after the first give-up.
    """
    pm = make_testable_process_manager()
    clock = _FakeClock()
    _install_supervisor(pm, clock)
    await _latch_structural_queue_wedge(pm)
    _disable_pp_reclaim_yield(pm, monkeypatch)
    pm._job_tracker.set_retry_policy(2)

    head = pm._job_tracker.jobs_pending_inference[0]
    assert head.id_ is not None
    pm._job_tracker.handle_job_fault_now(head, retryable=True, recovery_requeue=True)
    assert pm._job_tracker.retry_granted_by_recovery(head.id_) is True

    monkeypatch.setattr(pm, "_abort", lambda: None)
    proc = pm._process_map[0]
    monkeypatch.setattr(
        pm._process_lifecycle,
        "rebuild_inference_pool",
        lambda *, reason: setattr(proc, "last_process_state", HordeProcessState.WAITING_FOR_JOB),
    )
    monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)

    # The first (non-terminal) give-up spares the recovery-granted retry, so it faults nothing and writes no
    # ledger record; observe it through the call spy rather than the ledger.
    give_up_calls: list[dict[str, object]] = []
    real_give_up = pm._recovery_coordinator.give_up_on_wedged_jobs

    def _record_pending(**kwargs: object) -> None:
        real_give_up(**kwargs)  # type: ignore[arg-type]
        give_up_calls.append(
            {
                "terminal": kwargs.get("terminal"),
                "pending_after": len(list(pm._job_tracker.jobs_pending_inference)),
            },
        )

    monkeypatch.setattr(pm._recovery_coordinator, "give_up_on_wedged_jobs", _record_pending)

    for _ in range(40):
        clock.advance(1)
        pm._recovery_coordinator.run_recovery_supervisor()
        if give_up_calls:
            break

    assert give_up_calls, "the first give-up never fired over the ready, wedged pool"
    assert give_up_calls[0]["terminal"] is False  # a ready pool with live capacity is not abandoned first
    assert give_up_calls[0]["pending_after"] == 1  # the recovery-granted retry was spared, still queued


def test_head_recovery_in_flight_defers_give_up_then_fires_when_it_lands() -> None:
    """A ready lane whose head model is materialising defers give-up; give-up fires once that clears.

    The pure-policy control for the second in-flight-capacity window: while ``head_recovery_in_flight`` is
    True over a ready, wedged pool the give-up anchor is held, so no give-up fires past the grace; the moment
    the materialisation clears, the anchor sets and the sustained wedge escalates to give-up.
    """
    clock = _FakeClock()
    supervisor = RecoverySupervisor(
        clock=clock,
        wedge_grace_seconds=1,
        reset_interval_seconds=1,
        max_soft_resets=1,
        pool_ready_grace_seconds=_POOL_READY_GRACE,
        boot_allowance_seconds=1000,
        give_up_cooldown_seconds=1000,
        max_give_up_cycles=2,
        clean_streak_seconds=100,
    )

    # Open the episode, spend the soft reset, then hold over a ready pool with the head still materialising.
    deferred: list[RecoveryAction] = []
    for _ in range(int(_POOL_READY_GRACE) * 3 + 4):
        clock.advance(1)
        deferred.append(
            supervisor.evaluate(is_wedged=True, pool_ready=True, head_recovery_in_flight=True),
        )
    assert RecoveryAction.GIVE_UP not in deferred  # the loading head deferred give-up the whole time

    # The materialisation clears: the anchor sets and the still-persisting wedge escalates to give-up.
    fired = False
    for _ in range(int(_POOL_READY_GRACE) + 3):
        clock.advance(1)
        if supervisor.evaluate(is_wedged=True, pool_ready=True, head_recovery_in_flight=False) is (
            RecoveryAction.GIVE_UP
        ):
            fired = True
            break
    assert fired  # give-up still fires once the in-flight capacity resolves and the wedge persists
