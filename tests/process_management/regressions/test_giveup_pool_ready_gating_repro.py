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
from horde_worker_regen.process_management.ipc.messages import HordeProcessState, ModelLoadState
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


def _set_head_model_loading(pm: HordeWorkerProcessManager, *, model: str = "resident") -> None:
    """Put the head-of-queue job's model into the map's LOADING state (a preload in flight over an idle lane)."""
    pm._inference_scheduler._horde_model_map.update_entry(
        model,
        load_state=ModelLoadState.LOADING,
        process_id=0,
    )


class TestHeadModelMaterialisingDefersGiveUp:
    """A ready lane whose head-of-queue model is still loading is capacity in flight, not a wedge to fault.

    The boot-window test above holds give-up off while the *pool* is booting (no lane accepting). This covers
    the incident's second window: the rebuilt pool is accepting again, but the scheduler is mid-preload of the
    head's model. Give-up must defer to that load (bounded by the preload budget) instead of faulting the very
    job the pool is loading.
    """

    async def test_loading_head_over_ready_pool_is_not_faulted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pool accepting + head model LOADING: give-up is held off; the head is never faulted."""
        pm = make_testable_process_manager()
        clock = _FakeClock()
        _install_supervisor(pm, clock)
        # The head-materialisation deferral is bounded on the coordinator's clock; drive it off the same fake
        # clock so the whole test window stays well inside the (default, large) preload budget.
        pm._recovery_coordinator._clock = clock
        await _latch_structural_queue_wedge(pm)
        _set_head_model_loading(pm)
        proc = pm._process_map[0]

        # A soft reset rebuilds the pool; the replacement lane comes back accepting (ready) while the head's
        # model is still loading. The rebuild does not disturb the model-map LOADING state.
        monkeypatch.setattr(
            pm._process_lifecycle,
            "rebuild_inference_pool",
            lambda *, reason: setattr(proc, "last_process_state", HordeProcessState.WAITING_FOR_JOB),
        )
        monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)

        give_up_calls: list[dict[str, object]] = []
        real_give_up = pm._recovery_coordinator.give_up_on_wedged_jobs

        def _spy_give_up(**kwargs: object) -> None:
            give_up_calls.append(kwargs)
            real_give_up(**kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(pm._recovery_coordinator, "give_up_on_wedged_jobs", _spy_give_up)

        assert pm._recovery_coordinator.assess_wedge() is True
        assert pm._inference_scheduler.head_model_materializing() is True  # precondition for the deferral

        # Tick well past the pool-ready grace with the head's model loading the entire time.
        for _ in range(12):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()

        assert give_up_calls == []  # the loading head deferred give-up for the whole window
        assert _abandoned_records(pm) == []
        assert len(list(pm._job_tracker.jobs_pending_inference)) == 1  # the head was never faulted

        # The load completes and the queue clears (the head would dispatch); the worker resumes cleanly.
        pm._inference_scheduler._horde_model_map.expire_entry("resident")
        pm._message_dispatcher._in_queue_deadlock = False
        for _ in range(4):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()

        assert give_up_calls == []
        assert _abandoned_records(pm) == []
        assert len(list(pm._job_tracker.jobs_pending_inference)) == 1

    async def test_stuck_head_load_still_escalates_after_the_preload_budget(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A head load that never completes only defers give-up for the preload budget, then escalates."""
        preload_budget = 8
        pm = make_testable_process_manager()
        # The gpu-config resolution re-clamps a construction-time preload_timeout, so set the deferral budget
        # directly on the live bridge data the coordinator reads.
        pm.bridge_data.preload_timeout = preload_budget
        clock = _FakeClock()
        _install_supervisor(pm, clock)
        pm._recovery_coordinator._clock = clock
        await _latch_structural_queue_wedge(pm)
        _set_head_model_loading(pm)
        proc = pm._process_map[0]

        monkeypatch.setattr(pm, "_abort", lambda: None)
        monkeypatch.setattr(
            pm._process_lifecycle,
            "rebuild_inference_pool",
            lambda *, reason: setattr(proc, "last_process_state", HordeProcessState.WAITING_FOR_JOB),
        )
        monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)

        # The head's model stays LOADING and the wedge persists for the whole run: the deferral must be
        # bounded by the preload budget rather than parking give-up forever.
        first_give_up_at: float | None = None
        for _ in range(40):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()
            if _abandoned_records(pm):
                first_give_up_at = clock.now
                break

        assert first_give_up_at is not None, "give-up never fired: the head-load deferral was unbounded"
        # It fired only after the preload budget elapsed, never inside it (that window belongs to the load).
        assert first_give_up_at > preload_budget


async def test_recovery_granted_retry_survives_first_give_up_then_faults_when_wedge_persists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry a pool rebuild granted gets a dispatch opportunity: the first give-up spares it, the terminal does not.

    The rebuild requeues an in-flight job (``recovery_requeue``), granting it another attempt. A non-terminal
    give-up in the same cycle must not terminally fault that retry before the rebuilt pool has had a chance to
    run it; the terminal give-up (the wedge outlived the continuation cycle) faults it regardless.
    """
    pm = make_testable_process_manager()
    clock = _FakeClock()
    _install_supervisor(pm, clock)
    await _latch_structural_queue_wedge(pm)
    pm._job_tracker.set_retry_policy(2)

    # Emulate the rebuild granting this in-flight job a retry: it is requeued to PENDING_INFERENCE and marked
    # as a recovery-granted retry (exactly what _replace_inference_process does during a soft-reset rebuild).
    head = pm._job_tracker.jobs_pending_inference[0]
    assert head.id_ is not None
    pm._job_tracker.handle_job_fault_now(head, retryable=True, recovery_requeue=True)
    assert pm._job_tracker.retry_granted_by_recovery(head.id_) is True

    aborted = {"called": False}
    monkeypatch.setattr(pm, "_abort", lambda: aborted.__setitem__("called", True))

    proc = pm._process_map[0]
    monkeypatch.setattr(
        pm._process_lifecycle,
        "rebuild_inference_pool",
        lambda *, reason: setattr(proc, "last_process_state", HordeProcessState.WAITING_FOR_JOB),
    )
    monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)

    give_up_calls: list[dict[str, object]] = []
    real_give_up = pm._recovery_coordinator.give_up_on_wedged_jobs

    def _spy_give_up(**kwargs: object) -> None:
        pending_before = len(list(pm._job_tracker.jobs_pending_inference))
        real_give_up(**kwargs)  # type: ignore[arg-type]
        give_up_calls.append(
            {
                "terminal": kwargs.get("terminal"),
                "pending_before": pending_before,
                "pending_after": len(list(pm._job_tracker.jobs_pending_inference)),
            },
        )

    monkeypatch.setattr(pm._recovery_coordinator, "give_up_on_wedged_jobs", _spy_give_up)

    for _ in range(60):
        clock.advance(1)
        pm._recovery_coordinator.run_recovery_supervisor()
        if aborted["called"]:
            break

    assert give_up_calls, "give-up never fired"
    # The first give-up was non-terminal and left the recovery-granted retry queued for a dispatch opportunity.
    assert give_up_calls[0]["terminal"] is False
    assert give_up_calls[0]["pending_after"] == 1
    # A later terminal give-up (the wedge outlived the continuation cycle) faults it regardless.
    terminal_calls = [call for call in give_up_calls if call["terminal"] is True]
    assert terminal_calls, "the persisting wedge never reached a terminal give-up"
    assert terminal_calls[-1]["pending_after"] == 0
    assert aborted["called"] is True


async def _fault_pending_as_generation_failure(pm: HordeWorkerProcessManager) -> None:
    """Terminally fault every pending-inference job as an ordinary generation failure (default origin)."""
    for job in list(pm._job_tracker.jobs_pending_inference):
        pm._job_tracker.handle_job_fault_now(job, retryable=False)


async def _drain_faulted_submits(pm: HordeWorkerProcessManager, count: int) -> None:
    """Run the submit accounting over ``count`` queued faulted jobs with the network stubbed out."""
    pm._job_submitter._dry_run_skip_api = True
    for _ in range(count):
        await pm._job_submitter.api_submit_job()


async def test_give_up_batch_does_not_manufacture_consecutive_failure_pause() -> None:
    """A give-up batch faulting several jobs is a recovery action, so it must not latch the pop pause."""
    pm = make_testable_process_manager()
    pm._process_map[0] = make_mock_process_info(0, model_name="resident", state=HordeProcessState.WAITING_FOR_JOB)
    for _ in range(3):
        await track_popped_job_async(pm._job_tracker, make_job_pop_response(model="resident"))

    # A structural queue wedge over a live pool is the give-up's fault trigger; the jobs are servable-looking
    # but reissued so the horde re-dispatches them. None was a generation failure.
    pm._message_dispatcher._in_queue_deadlock = True
    pm._message_dispatcher._queue_deadlock_model = "resident"
    pm._message_dispatcher._last_queue_deadlock_detected_time = time.time() - 60

    pm._recovery_coordinator.give_up_on_wedged_jobs(terminal=False)
    assert len(list(pm._job_tracker.jobs_pending_submit)) == 3  # all three faulted and queued to submit

    await _drain_faulted_submits(pm, 3)

    assert pm._state.consecutive_failed_jobs == 0  # the recovery batch did not feed the failure counter
    latched = pm._job_popper._handle_consecutive_failures(pm.bridge_data, time.time())
    assert latched is False
    assert pm._state.too_many_consecutive_failed_jobs is False


async def test_genuine_generation_failures_still_latch_the_consecutive_failure_pause() -> None:
    """The counterpart: three real generation failures still count and latch the pop pause, exactly as before."""
    pm = make_testable_process_manager()
    for _ in range(3):
        await track_popped_job_async(pm._job_tracker, make_job_pop_response(model="resident"))

    await _fault_pending_as_generation_failure(pm)
    assert len(list(pm._job_tracker.jobs_pending_submit)) == 3

    await _drain_faulted_submits(pm, 3)

    assert pm._state.consecutive_failed_jobs == 3  # real failures still count
    latched = pm._job_popper._handle_consecutive_failures(pm.bridge_data, time.time())
    assert latched is True
    assert pm._state.too_many_consecutive_failed_jobs is True
