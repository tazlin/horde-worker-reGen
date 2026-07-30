"""Reproduction: a deterministically-doomed inference pool flaps forever instead of giving up.

Observed behavior: a worker whose every inference child crashed on start (here, a CPU-only torch in
the child's env raising ``Torch not compiled with CUDA enabled`` during hordelib init) racked up 24
process recoveries in a single session and never terminated on its own; it kept respawning the doomed
pool until an operator stopped it. Sibling sessions on the same broken env *did* give up and exit
cleanly, the only difference being how fast each crash burst recurred.

Root cause: the save-our-ship abort fires only when both conditions hold *at the same tick*:

  1. the recovery episode has aged past ``give_up_after_seconds`` (the supervisor returns ``GIVE_UP``), and
  2. ``_is_inference_pool_unrecoverable()`` is True (every slot quarantined right now).

But the path to give-up runs a soft reset first, and the soft reset's ``rebuild_inference_pool`` clears
the quarantine set to respawn the slots. So while the episode ages toward give-up the pool looks
*recoverable* (slots un-quarantined, merely starting), and ``_give_up_on_wedged_jobs`` skips the abort.
Whether the abort is ever reached then hinges on a race: if the freshly respawned slots crash and
re-quarantine *before* a clean streak closes the episode, the still-open episode catches the pool fully
quarantined and aborts (the tight-loop sessions). But if the respawned slots are slow to crash again,
e.g. the lazy inference start is gated behind a failing/slow download, the not-wedged window outlasts
``clean_streak_seconds``; the episode closes and the give-up clock resets. The next crash burst opens a
*fresh* episode (age 0), the next soft reset un-quarantines before that episode can age past give-up,
and the worker flaps between soft reset and re-crash indefinitely, accumulating recoveries without ever
aborting.

These tests drive the real recovery supervisor, soft-reset, and give-up paths through that slow-restart
cycle (the pool's respawn/crash is emulated by toggling the quarantine set, since no real children run
in unit tests) and assert the flapping ends.

How it ends depends on what is watching. A supervised worker (or one whose operator set
``exit_on_unhandled_faults``) exits non-zero so a fresh process is launched in its place. An unattended
worker's exit would end its usefulness rather than restore it, so it stays up; what must be bounded there is
the *harm*, meaning the process churn and the job faulting a doomed pool produces. Each test covers both
postures.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.action_ledger import LedgerEvent, LedgerEventType
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.recovery_supervisor import (
    _DEFAULT_CLEAN_STREAK_SECONDS,
    RecoverySupervisor,
)
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager


class _FakeClock:
    """A monotonic clock the test advances explicitly, so escalation timing is deterministic."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# A respawn-to-recrash window longer than the recovery clean streak: the doomed slots are slow to crash
# again (lazy inference start gated behind a failing download), so each soft reset's un-quarantine looks
# like a recovery and closes the episode before give-up can catch the pool quarantined.
_SLOW_RESTART_SECONDS = _DEFAULT_CLEAN_STREAK_SECONDS + 6.0

_SUPERVISION_ARMS = pytest.mark.parametrize("supervised", [True, False], ids=["supervised", "unattended"])
"""Both supervision postures: a worker something would relaunch, and an unattended one."""


def _install_abort_stub(
    pm: HordeWorkerProcessManager,
    monkeypatch: pytest.MonkeyPatch,
    *,
    supervised: bool,
) -> dict[str, bool]:
    """Attach the supervision posture and capture whether the worker actually exited.

    Args:
        pm: The process manager under test.
        monkeypatch: Fixture used to replace the real abort.
        supervised: Whether a supervising frontend is attached. With one, the recovery abort is permitted and
            the stub stands in for the real exit, marking shutdown exactly as the shutdown manager does so the
            worker's own "did the abort take" reading is truthful. Without one the abort is withheld, and the
            recovery gate never reaches this stub.

    Returns:
        A one-key record whose ``called`` entry reports whether the abort fired.
    """
    aborted = {"called": False}

    def _abort() -> None:
        aborted["called"] = True
        pm._state.initiate_shutdown()

    if supervised:
        pm._supervisor = Mock()
    monkeypatch.setattr(pm, "_abort", _abort)
    return aborted


def _count_pool_rebuilds(pm: HordeWorkerProcessManager, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Replace the pool rebuilds with counting no-ops, so any resumed churn is visible without real children.

    Returns:
        A one-key record whose ``count`` entry is the number of inference-pool rebuilds requested.
    """
    counts = {"count": 0}

    def _rebuild_inference(*, reason: str) -> None:
        counts["count"] += 1

    monkeypatch.setattr(pm._process_lifecycle, "rebuild_inference_pool", _rebuild_inference)
    monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)
    return counts


def _abandoned_records(pm: HordeWorkerProcessManager) -> list[LedgerEvent]:
    """All RECOVERY_ABANDONED events currently in the action ledger."""
    return [
        event
        for event in pm._recovery_coordinator._action_ledger.recent(limit=1000)
        if event.event_type == LedgerEventType.RECOVERY_ABANDONED
    ]


class TestDoomedPoolEventuallyAborts:
    """The whole point of give-up: a pool that can never serve must stop the worker, not loop forever."""

    @_SUPERVISION_ARMS
    def test_doomed_pool_with_slow_restart_eventually_aborts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        supervised: bool,
    ) -> None:
        """A doomed pool whose respawn is slower than the clean streak must not flap forever.

        Supervised, the escalation ends in the exit that brings a fresh process back. Unattended, the worker
        stays up and the churn itself must stop: process recoveries and abandonment records both stop climbing.
        """
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle
        max_procs = pm.max_inference_processes

        # Deterministic clock so the episode/give-up/clean-streak timing is exact, shared with the coordinator
        # so the quiescent window the unattended arm measures is deterministic too.
        clock = _FakeClock()
        pm._recovery_coordinator.recovery_supervisor = RecoverySupervisor(clock=clock)
        pm._recovery_coordinator._clock = clock

        # A soft reset rebuilds the pool in place, which clears the quarantine set and respawns the
        # slots. Emulate just that effect (un-quarantine every slot) without launching real children, and
        # record the replacements as the process recoveries they are: that count is the churn under test.
        def _fake_rebuild_inference(*, reason: str) -> None:
            lifecycle._quarantined_inference_slots.clear()
            lifecycle._num_process_recoveries += max_procs

        monkeypatch.setattr(lifecycle, "rebuild_inference_pool", _fake_rebuild_inference)
        monkeypatch.setattr(lifecycle, "rebuild_safety_pool", lambda *, reason: None)

        aborted = _install_abort_stub(pm, monkeypatch, supervised=supervised)

        def _crash_burst_quarantines_pool() -> None:
            """The doomed children crashed on start: the breaker quarantines every slot."""
            lifecycle._quarantined_inference_slots = set(range(max_procs))

        def _escalation_settled() -> bool:
            """Whether the escalation has reached its end for this posture (an exit, or a quiescent worker)."""
            return aborted["called"] or pm._state.recovery_parked

        # Many crash/recover cycles: far more give-up opportunities than the 24 recoveries observed live.
        for _cycle in range(15):
            _crash_burst_quarantines_pool()

            # The supervisor notices the wedge and (after its grace) soft-resets, which un-quarantines
            # the pool. Tick through the soft reset and past the give-up age while un-quarantined.
            for _ in range(5):
                clock.advance(2.0)
                pm._recovery_coordinator.run_recovery_supervisor()
                if _escalation_settled():
                    break

            # The respawned slots are slow to crash again: a not-wedged window that outlasts the clean
            # streak, closing the episode and resetting the give-up clock.
            for _ in range(int(_SLOW_RESTART_SECONDS // 2) + 1):
                clock.advance(2.0)
                pm._recovery_coordinator.run_recovery_supervisor()

            if _escalation_settled():
                break

        if supervised:
            assert aborted["called"], (
                "A deterministically-doomed inference pool never aborted: the save-our-ship loop soft-reset "
                "the pool indefinitely (racking up unbounded process recoveries) instead of giving up, "
                "because the abort requires the pool to be fully quarantined at the exact give-up tick and "
                "the soft reset's un-quarantine keeps that from ever coinciding."
            )
            return

        assert aborted["called"] is False, (
            "The unattended worker exited: with nothing watching to launch a replacement, exiting ends the "
            "worker instead of recovering it."
        )
        churn_before = lifecycle._num_process_recoveries
        abandonments_before = len(_abandoned_records(pm))

        # Keep the doomed pool doomed and keep ticking. Nothing about the pool has changed, so nothing the
        # escalation could do would help, and the worker must stop trying rather than churn.
        for _ in range(60):
            _crash_burst_quarantines_pool()
            clock.advance(2.0)
            pm._recovery_coordinator.run_recovery_supervisor()

        assert lifecycle._num_process_recoveries == churn_before, (
            "The unattended worker kept rebuilding a pool no remaining remedy can fix: process recoveries "
            "must stop climbing once the escalation has nothing left to try."
        )
        assert len(_abandoned_records(pm)) == abandonments_before, (
            "The unattended worker kept giving up on jobs over a pool it cannot serve: the faulting must stop, "
            "not repeat for as long as the worker runs."
        )

    @_SUPERVISION_ARMS
    def test_give_up_after_soft_reset_unquarantine_does_not_abort(
        self,
        monkeypatch: pytest.MonkeyPatch,
        supervised: bool,
    ) -> None:
        """Isolated phase mismatch: GIVE_UP arriving just after a soft-reset un-quarantine skips the abort.

        This is the single tick at the heart of the loop above. The supervisor has decided the episode is
        doomed (``GIVE_UP``), but the soft reset that preceded it already cleared the quarantine set, so
        ``_give_up_on_wedged_jobs`` sees a "recoverable" pool and declines to abort. Supervised, that tick must
        end in the exit; unattended, it must end the escalation instead of handing it back another cycle.
        """
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle
        clock = _FakeClock()
        pm._recovery_coordinator._clock = clock
        pm._recovery_coordinator.recovery_supervisor = RecoverySupervisor(clock=clock)

        aborted = _install_abort_stub(pm, monkeypatch, supervised=supervised)
        rebuilds = _count_pool_rebuilds(pm, monkeypatch)

        # The pool is doomed (it will re-crash), but a soft reset has *just* un-quarantined every slot, so
        # at this instant nothing is quarantined: exactly the state the episode reaches when give-up is due.
        lifecycle._quarantined_inference_slots = set()
        assert pm._recovery_coordinator.is_inference_pool_unrecoverable() is False

        pm._recovery_coordinator.give_up_on_wedged_jobs()

        if supervised:
            assert aborted["called"], (
                "Give-up fired on a doomed pool but did not abort because the pool was transiently "
                "un-quarantined by the preceding soft reset; the worker keeps running and loops."
            )
            return

        assert aborted["called"] is False

        # The doomed pool persists, so the escalation spends the rungs it has left. Once they are spent the
        # churn and the faulting must both stop rather than repeat for as long as the worker runs.
        lifecycle._quarantined_inference_slots = set(range(pm.max_inference_processes))
        for _ in range(60):
            clock.advance(2.0)
            pm._recovery_coordinator.run_recovery_supervisor()
            if pm._state.recovery_parked:
                break

        rebuilds_when_spent = rebuilds["count"]
        abandonments_when_spent = len(_abandoned_records(pm))
        for _ in range(60):
            clock.advance(2.0)
            pm._recovery_coordinator.run_recovery_supervisor()

        assert rebuilds["count"] == rebuilds_when_spent, (
            "The unattended worker kept rebuilding the pool after the escalation had exhausted every remedy "
            "it can perform in place."
        )
        assert len(_abandoned_records(pm)) == abandonments_when_spent, (
            "The exhausted escalation repeated its give-up: the spent attempts must be the last until the "
            "worker is permitted a fresh one."
        )

    def test_exhausted_escalation_resumes_when_the_condition_clears(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The quiescent state is exitable: after its re-probe interval the unattended worker tries again.

        What dooms a pool is often outside the worker (a co-tenant process holding the card's VRAM). So a
        worker that stopped escalating must not stay stopped: once the interval elapses it re-assesses, and a
        pool that can serve again keeps the worker serving rather than quiescent.
        """
        pm = make_testable_process_manager()
        clock = _FakeClock()
        pm._recovery_coordinator._clock = clock
        aborted = _install_abort_stub(pm, monkeypatch, supervised=False)
        rebuilds = _count_pool_rebuilds(pm, monkeypatch)

        pm._process_lifecycle._quarantined_inference_slots = set()
        pm._recovery_coordinator.give_up_on_wedged_jobs(terminal=True)
        assert aborted["called"] is False
        assert pm._state.recovery_parked is True

        # The blocker clears while the worker waits: a live, idle lane can accept work again.
        pm._process_map[0] = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)
        clock.advance(pm._recovery_coordinator.RECOVERY_PARK_REPROBE_SECONDS + 1.0)
        pm._recovery_coordinator.run_recovery_supervisor()

        assert pm._state.recovery_parked is False  # the worker pops and escalates again
        assert rebuilds["count"] == 0  # a healthy pool needs no rebuild
        assert len(_abandoned_records(pm)) == 1  # and no further job was given up on


class TestGiveUpDoesNotOverAbort:
    """The doom-aware abort gate must not fire on a worker that is merely starved or has recovered."""

    def test_healthy_pool_starved_by_queue_deadlock_does_not_abort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A healthy pool (capacity available, never quarantined) must reissue work and keep running, not abort."""
        pm = make_testable_process_manager()
        aborted = {"called": False}
        monkeypatch.setattr(pm, "_abort", lambda: aborted.__setitem__("called", True))

        # A live, idle inference process: capacity is available and no doom was ever latched.
        pm._process_map[0] = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)
        assert pm._recovery_coordinator.is_inference_capacity_available() is True
        assert pm._recovery_coordinator.episode_saw_unrecoverable_pool is False

        pm._recovery_coordinator.give_up_on_wedged_jobs()

        assert aborted["called"] is False

    def test_served_progress_clears_doom_latch_so_recovered_worker_does_not_abort(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A doomed pool that recovers and serves a job clears the latch; a later give-up must not abort it."""
        pm = make_testable_process_manager()
        clock = _FakeClock()
        pm._recovery_coordinator.recovery_supervisor = RecoverySupervisor(clock=clock)
        lifecycle = pm._process_lifecycle

        aborted = {"called": False}
        monkeypatch.setattr(pm, "_abort", lambda: aborted.__setitem__("called", True))
        monkeypatch.setattr(
            lifecycle,
            "rebuild_inference_pool",
            lambda *, reason: lifecycle._quarantined_inference_slots.clear(),
        )
        monkeypatch.setattr(lifecycle, "rebuild_safety_pool", lambda *, reason: None)

        # A live idle process so capacity stays available throughout: the latch, not the capacity gate, is
        # what this exercises.
        pm._process_map[0] = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)

        # The episode opens doomed (every slot quarantined): the latch is set and the baseline captured.
        lifecycle._quarantined_inference_slots = set(range(pm.max_inference_processes))
        clock.advance(2.0)
        pm._recovery_coordinator.run_recovery_supervisor()
        assert pm._recovery_coordinator.episode_saw_unrecoverable_pool is True

        # The pool genuinely recovers and serves a job: un-quarantine and record a completion past the baseline.
        lifecycle._quarantined_inference_slots = set()
        pm._job_tracker._total_num_completed_jobs += 1

        # Tick well past the give-up age. The served progress clears the latch, so give-up declines to abort.
        for _ in range(6):
            clock.advance(2.0)
            pm._recovery_coordinator.run_recovery_supervisor()

        assert pm._recovery_coordinator.episode_saw_unrecoverable_pool is False
        assert aborted["called"] is False


class TestRunawayRecoveryBackstop:
    """An independent, clock-agnostic catch-all: flapping recoveries abandon ship even if give-up never coincides."""

    @_SUPERVISION_ARMS
    def test_flapping_recoveries_abandon_ship(self, monkeypatch: pytest.MonkeyPatch, supervised: bool) -> None:
        """Recoveries past the ceiling within the window end the flapping: by exiting, or by stopping the churn.

        The backstop's job is to stop a worker that cannot stabilise. Supervised, that is the exit. Unattended,
        the exit would not bring a fresh process back, so the rebuilding that produced this recovery rate is
        what stops: no further pool rebuild is requested even though every slot is quarantined.
        """
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle
        clock = _FakeClock()
        pm._recovery_coordinator._clock = clock
        aborted = _install_abort_stub(pm, monkeypatch, supervised=supervised)
        rebuilds = _count_pool_rebuilds(pm, monkeypatch)

        lifecycle._num_process_recoveries = pm._recovery_coordinator.RUNAWAY_RECOVERY_CEILING

        if supervised:
            assert pm._recovery_coordinator.maybe_abort_on_runaway_recoveries() is True
            assert aborted["called"] is True
            return

        assert pm._recovery_coordinator.maybe_abort_on_runaway_recoveries() is False
        assert aborted["called"] is False

        # Every slot quarantined is a wedge an escalation still running would soft-reset on every tick.
        lifecycle._quarantined_inference_slots = set(range(pm.max_inference_processes))
        churn_before = lifecycle._num_process_recoveries
        for _ in range(60):
            clock.advance(2.0)
            pm._recovery_coordinator.run_recovery_supervisor()

        assert rebuilds["count"] == 0, (
            "The flapping worker kept rebuilding its pool after the backstop found the recovery rate past its "
            "ceiling; the churn is exactly what the backstop exists to stop."
        )
        assert lifecycle._num_process_recoveries == churn_before
        assert len(_abandoned_records(pm)) == 1  # the single backstop record, not one per tick

    def test_sparse_recoveries_do_not_abandon_ship(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A recovery count just below the ceiling must not trip the backstop."""
        pm = make_testable_process_manager()
        aborted = {"called": False}
        monkeypatch.setattr(pm, "_abort", lambda: aborted.__setitem__("called", True))

        pm._process_lifecycle._num_process_recoveries = pm._recovery_coordinator.RUNAWAY_RECOVERY_CEILING - 1

        assert pm._recovery_coordinator.maybe_abort_on_runaway_recoveries() is False
        assert aborted["called"] is False

    def test_recoveries_outside_window_are_pruned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Recoveries older than the rolling window age out and cannot, alone, breach the ceiling."""
        pm = make_testable_process_manager()
        aborted = {"called": False}
        monkeypatch.setattr(pm, "_abort", lambda: aborted.__setitem__("called", True))

        # A full ceiling's worth of recoveries, but all older than the window, with no new ones since.
        stale = time.time() - pm._recovery_coordinator.RUNAWAY_RECOVERY_WINDOW_SECONDS - 10.0
        pm._recovery_coordinator.recovery_event_times = [
            stale for _ in range(pm._recovery_coordinator.RUNAWAY_RECOVERY_CEILING)
        ]
        pm._recovery_coordinator.last_seen_recovery_count = pm._process_lifecycle._num_process_recoveries

        assert pm._recovery_coordinator.maybe_abort_on_runaway_recoveries() is False
        assert aborted["called"] is False
        assert pm._recovery_coordinator.recovery_event_times == []
