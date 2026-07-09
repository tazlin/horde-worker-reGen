"""Reproduction: save-our-ship gave up on servable jobs while a just-rebuilt pool was still booting.

Observed behavior: a queue-deadlock wedge opened a recovery episode; the supervisor soft-reset the pools
at the wedge grace, and then, purely on episode age, returned GIVE_UP roughly five seconds later while the
replacement inference children were still PROCESS_STARTING (torch import not finished). Give-up faulted the
three servable jobs the freshly rebuilt pool was about to run. Because give-up then fired every control
tick once the episode had aged out, and recorded a RECOVERY_ABANDONED ledger event each time even with
``jobs_faulted=0`` (a live-but-starting pool made capacity look available, so the structural abort never
fired), the ledger accumulated hundreds of no-op abandonment records.

The fix makes give-up readiness-aware: the escalation clock does not advance while the rebuilt pool is
still booting (no inference lane in an accepting state), give-up fires at most once per cycle, the ledger
records only when the give-up actually faulted a job or made a terminal abort decision, and a wedge that
outlives a fresh soft-reset cycle escalates to a deliberate abort rather than faulting forever.

These tests drive the real ``run_recovery_supervisor`` / ``give_up_on_wedged_jobs`` paths with a fake
escalation clock, emulating the pool rebuild's boot window by toggling the replacement slot's process
state (no real children run in unit tests).
"""

from __future__ import annotations

import time

import pytest

from horde_worker_regen.process_management.ipc.action_ledger import LedgerEvent, LedgerEventType
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.recovery_supervisor import RecoverySupervisor
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


def _install_supervisor(pm: HordeWorkerProcessManager, clock: _FakeClock) -> RecoverySupervisor:
    """Swap in a supervisor on the fake clock with tight, legible escalation windows."""
    supervisor = RecoverySupervisor(
        clock=clock,
        wedge_grace_seconds=1,
        reset_interval_seconds=1,
        max_soft_resets=1,
        pool_ready_grace_seconds=3,
        boot_allowance_seconds=20,
        give_up_cooldown_seconds=5,
        max_give_up_cycles=2,
        clean_streak_seconds=100,
    )
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


async def test_boot_window_after_soft_reset_does_not_fault_servable_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The incident: give-up must not fault the head while the just-rebuilt pool is still booting.

    A structural queue wedge soft-resets the pool; the replacement slot then sits PROCESS_STARTING through
    the boot window. Give-up must be held off for the whole window, and when the slot reaches an accepting
    state and the wedge clears, the worker resumes with zero jobs faulted and zero abandonment records.
    """
    pm = make_testable_process_manager()
    clock = _FakeClock()
    _install_supervisor(pm, clock)
    await _latch_structural_queue_wedge(pm)
    proc = pm._process_map[0]

    # A soft reset rebuilds the pool; emulate the replacement children entering the boot window.
    monkeypatch.setattr(
        pm._process_lifecycle,
        "rebuild_inference_pool",
        lambda *, reason: setattr(proc, "last_process_state", HordeProcessState.PROCESS_STARTING),
    )
    monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)

    give_up_calls: list[dict[str, object]] = []
    real_give_up = pm._recovery_coordinator.give_up_on_wedged_jobs

    def _spy_give_up(**kwargs: object) -> None:
        give_up_calls.append(kwargs)
        real_give_up(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pm._recovery_coordinator, "give_up_on_wedged_jobs", _spy_give_up)

    assert pm._recovery_coordinator.assess_wedge() is True  # precondition: the wedge is real

    # Open the episode, soft-reset, then tick through the boot window with the slot PROCESS_STARTING.
    for _ in range(8):
        clock.advance(1)
        pm._recovery_coordinator.run_recovery_supervisor()

    assert proc.last_process_state is HordeProcessState.PROCESS_STARTING  # still booting
    assert give_up_calls == []  # give-up was held off for the whole boot window
    assert len(list(pm._job_tracker.jobs_pending_inference)) == 1  # the head was never faulted

    # The replacement slot finishes booting (accepting again) and the wedge clears: the worker resumes.
    proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
    pm._message_dispatcher._in_queue_deadlock = False
    for _ in range(4):
        clock.advance(1)
        pm._recovery_coordinator.run_recovery_supervisor()

    assert give_up_calls == []
    assert _abandoned_records(pm) == []
    assert len(list(pm._job_tracker.jobs_pending_inference)) == 1


async def _drive_to_first_give_up(
    pm: HordeWorkerProcessManager,
    clock: _FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild-keeps-ready pool + persistent wedge, ticked until the first give-up has fired."""
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
            return
    raise AssertionError("give-up never fired over a ready, persistently wedged pool")


async def test_give_up_still_protects_ready_wedged_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give-up must still reissue the head when the pool is ready and the wedge persists past the grace.

    Exactly one give-up faults the servable head; exactly one honest ledger record (jobs_faulted > 0) is
    written; the worker does not abort on this first give-up (the pool has live capacity).
    """
    pm = make_testable_process_manager()
    clock = _FakeClock()
    _install_supervisor(pm, clock)
    await _latch_structural_queue_wedge(pm)

    aborted = {"called": False}
    monkeypatch.setattr(pm, "_abort", lambda: aborted.__setitem__("called", True))

    await _drive_to_first_give_up(pm, clock, monkeypatch)

    records = _abandoned_records(pm)
    assert len(records) == 1
    assert records[0].detail["jobs_faulted"] == 1
    assert records[0].detail["terminal"] is False
    assert len(list(pm._job_tracker.jobs_pending_inference)) == 0  # the head was reissued
    assert aborted["called"] is False  # a healthy pool with capacity is not abandoned on the first give-up


async def test_latched_give_up_does_not_spam_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a give-up, wedged ticks within the cool-down produce no further faulting or ledger records."""
    pm = make_testable_process_manager()
    clock = _FakeClock()
    _install_supervisor(pm, clock)
    await _latch_structural_queue_wedge(pm)
    monkeypatch.setattr(pm, "_abort", lambda: None)

    await _drive_to_first_give_up(pm, clock, monkeypatch)
    assert len(_abandoned_records(pm)) == 1

    # Tick repeatedly while still within the give-up cool-down: the latch must suppress any repeat give-up.
    for _ in range(4):
        clock.advance(1)
        pm._recovery_coordinator.run_recovery_supervisor()

    assert len(_abandoned_records(pm)) == 1  # no spam: still exactly one record


async def test_persisting_wedge_over_ready_pool_escalates_to_deliberate_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wedge that outlives a continuation soft-reset cycle aborts deliberately, not by chance.

    The first give-up faults the head (no abort: capacity is live). The wedge persists over the ready pool,
    so after the cool-down one further soft-reset cycle runs and a second, terminal give-up abandons ship.
    Both ledger records are honest (a faulted job, then a terminal decision).
    """
    pm = make_testable_process_manager()
    clock = _FakeClock()
    _install_supervisor(pm, clock)
    await _latch_structural_queue_wedge(pm)

    aborted = {"called": False}
    monkeypatch.setattr(pm, "_abort", lambda: aborted.__setitem__("called", True))

    proc = pm._process_map[0]
    monkeypatch.setattr(
        pm._process_lifecycle,
        "rebuild_inference_pool",
        lambda *, reason: setattr(proc, "last_process_state", HordeProcessState.WAITING_FOR_JOB),
    )
    monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)

    for _ in range(60):
        clock.advance(1)
        pm._recovery_coordinator.run_recovery_supervisor()
        if aborted["called"]:
            break

    assert aborted["called"] is True  # a deliberate terminal escalation, not an infinite spin
    records = _abandoned_records(pm)
    assert len(records) == 2  # exactly one continuation: two give-ups total
    assert records[0].detail["terminal"] is False
    assert records[1].detail["terminal"] is True
    assert records[0].detail["jobs_faulted"] == 1  # the first reissued the servable head


def _install_transient_window_supervisor(pm: HordeWorkerProcessManager, clock: _FakeClock) -> RecoverySupervisor:
    """Swap in a supervisor whose clean streak is short enough that a rebuild's not-wedged window meets it."""
    supervisor = RecoverySupervisor(
        clock=clock,
        wedge_grace_seconds=1,
        reset_interval_seconds=1,
        max_soft_resets=1,
        pool_ready_grace_seconds=3,
        boot_allowance_seconds=20,
        give_up_cooldown_seconds=5,
        max_give_up_cycles=2,
        clean_streak_seconds=3,
    )
    pm._recovery_coordinator.recovery_supervisor = supervisor
    return supervisor


class TestZeroProgressSoftResetDoesNotReLogAsFirst:
    """The field signature: a rebuild's transient not-wedged window must not reset the escalation counter.

    A structural queue wedge that soft-resets, momentarily reads as not-wedged while the pool rebuilds, and
    recurs without any work moving forward must climb the escalation, not re-log every reset as the first.
    """

    async def test_clean_streak_without_progress_holds_the_escalation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A soft reset followed by a no-progress clean streak keeps the counter, then escalates to give-up."""
        pm = make_testable_process_manager()
        clock = _FakeClock()
        _install_transient_window_supervisor(pm, clock)
        await _latch_structural_queue_wedge(pm)

        aborted = {"called": False}
        monkeypatch.setattr(pm, "_abort", lambda: aborted.__setitem__("called", True))

        proc = pm._process_map[0]
        dispatcher = pm._message_dispatcher

        # The rebuild momentarily clears the deadlock (the transient not-wedged window) with the slot still
        # accepting; no job is served, so this is not a real recovery.
        def _rebuild_clears_deadlock(*, reason: str) -> None:
            proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
            dispatcher._in_queue_deadlock = False

        monkeypatch.setattr(pm._process_lifecycle, "rebuild_inference_pool", _rebuild_clears_deadlock)
        monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)

        # Drive to the first soft reset (which clears the deadlock via the rebuild above).
        for _ in range(6):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()
            if pm._recovery_coordinator.recovery_supervisor.limp_by_level == 1:
                break
        assert pm._recovery_coordinator.recovery_supervisor.limp_by_level == 1
        assert dispatcher._in_queue_deadlock is False  # the transient not-wedged rebuild window is open

        # Tick past the clean streak with the deadlock cleared and zero progress recorded.
        for _ in range(6):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()

        # The transient window must not have been mistaken for a recovery: the episode is still open and the
        # escalation counter is held (without the progress requirement it would have reset to zero here).
        assert pm._recovery_coordinator.recovery_supervisor.is_in_episode is True
        assert pm._recovery_coordinator.recovery_supervisor.limp_by_level == 1

        # The wedge recurs. With the counter held at the spent budget, the worker escalates to give-up rather
        # than taking another first soft reset.
        dispatcher._in_queue_deadlock = True
        dispatcher._last_queue_deadlock_detected_time = time.time() - 60
        for _ in range(12):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()
            if _abandoned_records(pm):
                break
        assert _abandoned_records(pm), "the held counter did not escalate: the wedge never reached give-up"

    async def test_progress_after_reset_closes_episode_and_next_wedge_starts_fresh(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Healthy counterpart: real progress after the reset closes the episode; a later wedge starts at #1."""
        pm = make_testable_process_manager()
        clock = _FakeClock()
        _install_transient_window_supervisor(pm, clock)
        await _latch_structural_queue_wedge(pm)

        monkeypatch.setattr(pm, "_abort", lambda: None)

        proc = pm._process_map[0]
        dispatcher = pm._message_dispatcher

        def _rebuild_clears_deadlock(*, reason: str) -> None:
            proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
            dispatcher._in_queue_deadlock = False

        monkeypatch.setattr(pm._process_lifecycle, "rebuild_inference_pool", _rebuild_clears_deadlock)
        monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)

        for _ in range(6):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()
            if pm._recovery_coordinator.recovery_supervisor.limp_by_level == 1:
                break
        assert pm._recovery_coordinator.recovery_supervisor.limp_by_level == 1

        # The rebuilt pool serves work: a completion past the post-reset baseline is real forward progress.
        pm._job_tracker._total_num_completed_jobs += 1

        for _ in range(6):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()

        # Progress-backed clean streak closes the episode and returns the escalation counter to baseline.
        assert pm._recovery_coordinator.recovery_supervisor.is_in_episode is False
        assert pm._recovery_coordinator.recovery_supervisor.limp_by_level == 0

        # A later, independent wedge opens a fresh episode: its first reset is #1 again, not a continuation.
        dispatcher._in_queue_deadlock = True
        dispatcher._last_queue_deadlock_detected_time = time.time() - 60
        for _ in range(6):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()
            if pm._recovery_coordinator.recovery_supervisor.limp_by_level == 1:
                break
        assert pm._recovery_coordinator.recovery_supervisor.limp_by_level == 1
