"""Reproduction: a safety pool that can never start is respawned forever without ever being declared failing.

Three gaps in the safety-process recovery escalation share one consequence: the worker rebuilds a safety
child that dies on every start, indefinitely, while every signal that would escalate stays clear.

1. Laundering. :meth:`ProcessLifecycleManager._replace_all_safety_process` skips the crash-loop record while
   either intentional-replacement flag is set. ``_safety_replacement_intentional_until_ready`` is cleared only
   by :meth:`_observe_safety_pool_readiness`, which requires a *loaded* safety process. A
   child that dies while still ``PROCESS_STARTING`` never loads, so an intentional window opened by a
   whole-card pause never closes and every subsequent crash-driven rebuild is counted as part of the
   placement change. :attr:`ProcessLifecycleManager.safety_pool_failing` stays False for an unbounded
   crash-on-start loop. Contract: the suppression is bounded, so a rebuild loop that never reaches readiness
   is eventually counted and the pool is reported failing.

2. No futility classification. The reap-rebuild-start loop respawns immediately with no ceiling. The only
   breaker is the sliding window (``SAFETY_CRASH_LOOP_MAX`` rebuilds within ``CRASH_LOOP_WINDOW_SECONDS``),
   which a slow cold start outruns: rebuilds spaced wider than the window's per-rebuild budget age out before
   they can accumulate, so the pool respawns forever at a rate the breaker can never see. Contract: the
   rate-independent companion the inference side already has (``CRASH_LOOP_MAX_START_FAILURES``): consecutive
   rebuilds where the child never reaches readiness are structural regardless of spacing, so the pool is
   declared unrecoverable and the wedge assessment sees it.

3. No failure-independent backstop. Job pops can be held at a gate (``no_safety_process`` among them) for
   hours with no jobs flowing. The pop-liveness sentinel logs that, and nothing consumes it: no wedge
   assessment keys on a held gate. Contract: a pop gate held past a threshold with zero job flow is a wedge
   regardless of which gate it is, since the gate's own cause cannot satisfy a completed-work signal.

The controls pin the behavior the fixes must not break: an intentional cycle that does reach readiness stays
uncounted, one crash on a healthy pool is not a loop, the sliding-window threshold still trips without an
intentional window, and a rebuild streak broken by a successful load is not structural.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import (
    CRASH_LOOP_WINDOW_SECONDS,
    SAFETY_CRASH_LOOP_MAX,
    SAFETY_PROCESS_ID,
    SAFETY_RESPAWN_BACKOFF_MAX_SECONDS,
    ProcessLifecycleManager,
)
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager

_SLOW_RESPAWN_SECONDS = CRASH_LOOP_WINDOW_SECONDS / 2
"""Spacing between rebuilds that the sliding-window breaker can never accumulate past its threshold.

A cold start this slow lets each rebuild age out of ``CRASH_LOOP_WINDOW_SECONDS`` before enough of them
coincide, which is the condition a rate-independent futility signal has to cover.
"""

_FUTILE_RESPAWN_COUNT = SAFETY_CRASH_LOOP_MAX * 3
"""Rebuild attempts driven per futility scenario: well past any plausible bound on respawning."""

_BACKOFF_CLEARING_RESPAWN_SECONDS = SAFETY_RESPAWN_BACKOFF_MAX_SECONDS * 2
"""Spacing between rebuilds that clears the respawn backoff while staying dense inside the crash-loop window.

Consecutive failed rebuilds are deliberately spaced out by a growing backoff, so a scenario driving rebuild
after rebuild has to move the clock past it or the state machine legitimately declines to respawn yet.
"""

_GATE_HELD_SECONDS = 3600.0
"""How long a pop gate is held in the backstop scenarios: far past any threshold a fix could pick."""

_BRIEF_GATE_HELD_SECONDS = 30.0
"""A gate hold short enough that no threshold should treat it as a wedge."""


class _AdvanceableTime:
    """Stand-in for the ``time`` module whose ``time()`` the test can advance.

    Every other attribute is forwarded to the real module, so code paths that use ``monotonic`` or ``sleep``
    are unaffected. Only the wall clock the crash-loop window is measured against moves.
    """

    def __init__(self) -> None:
        self._offset = 0.0

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - forwards the real module's arbitrary members
        """Return the real ``time`` module's attribute of that name."""
        return getattr(time, name)

    def time(self) -> float:
        """Return the current wall clock, shifted by however far the test has advanced it."""
        return time.time() + self._offset

    def advance(self, seconds: float) -> None:
        """Move the wall clock forward."""
        self._offset += seconds


class _FakeClock:
    """A clock the test advances explicitly, so gate-hold durations are exact."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward."""
        self.now += seconds


def _stub_safety_start(lifecycle: ProcessLifecycleManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the rebuild state machine from spawning a child, without changing what it records.

    The replacement flow's start step is the only part that would touch the OS. Everything the scenarios read
    (the recovery history, the intentional-window flags, the pool's failing verdict) is decided around it.
    """
    monkeypatch.setattr(lifecycle, "start_safety_processes", lambda: True)


def _crashed_starting_safety_process() -> HordeProcessInfo:
    """A safety child that died while still ``PROCESS_STARTING`` (it never reached readiness)."""
    crashed = make_mock_process_info(
        SAFETY_PROCESS_ID,
        model_name=None,
        state=HordeProcessState.PROCESS_STARTING,
        process_type=HordeProcessType.SAFETY,
    )
    crashed.mp_process.is_alive.return_value = False  # pyrefly: ignore - Mock stand-in for the child handle
    crashed.mp_process.exitcode = -6  # pyrefly: ignore - Mock stand-in; a native abort, not a clean end
    return crashed


def _drive_rebuild_to_completion(lifecycle: ProcessLifecycleManager) -> None:
    """Tick the safety replacement state machine until it has started a fresh safety process."""
    for _ in range(5):
        if not lifecycle.safety_processes_should_be_replaced:
            return
        lifecycle._replace_all_safety_process()
    raise AssertionError("the safety replacement state machine never completed a rebuild cycle")


def _crash_safety_during_startup(lifecycle: ProcessLifecycleManager) -> None:
    """Run one full cycle of a safety child dying before readiness and being rebuilt."""
    crashed = _crashed_starting_safety_process()
    lifecycle._process_map[SAFETY_PROCESS_ID] = crashed
    assert lifecycle._reap_if_crashed(crashed) is True
    _drive_rebuild_to_completion(lifecycle)


def _safety_child_reaches_readiness(lifecycle: ProcessLifecycleManager) -> None:
    """Put a ready safety process in the map and let the lifecycle observe that readiness."""
    lifecycle._process_map[SAFETY_PROCESS_ID] = make_mock_process_info(
        SAFETY_PROCESS_ID,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.SAFETY,
    )
    lifecycle._observe_safety_pool_readiness()


def _hold_pop_gate(pm: HordeWorkerProcessManager, clock: _FakeClock, gate: str, held_seconds: float) -> None:
    """Report job pops as held at ``gate`` for ``held_seconds``, with no pop reaching the horde since."""
    pm._state.last_pop_gate = gate
    pm._state.last_pop_gate_since = clock.now - held_seconds
    pm._state.last_pop_attempt_completed_at = clock.now - held_seconds


class TestIntentionalWindowStopsLaunderingCrashLoops:
    """An intentional replacement window must not absorb a crash loop that never reaches readiness."""

    def test_crash_on_start_loop_inside_an_intentional_window_is_eventually_counted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A safety child crashing before readiness must trip the pool's failing signal, window open or not.

        The whole-card pause opens the window legitimately; what must not follow is an unbounded rebuild loop
        that the window keeps invisible to every consumer of ``safety_pool_failing``.
        """
        pm = make_testable_process_manager(safety_on_gpu=True)
        lifecycle = pm._process_lifecycle
        _stub_safety_start(lifecycle, monkeypatch)

        fake_time = _AdvanceableTime()
        monkeypatch.setattr(
            "horde_worker_regen.process_management.lifecycle.process_lifecycle.time",
            fake_time,
        )

        assert lifecycle.pause_safety_on_gpu() is True
        _drive_rebuild_to_completion(lifecycle)
        assert lifecycle._safety_replacement_intentional_until_ready is True

        for _ in range(_FUTILE_RESPAWN_COUNT):
            _crash_safety_during_startup(lifecycle)
            fake_time.advance(_BACKOFF_CLEARING_RESPAWN_SECONDS)

        assert lifecycle.safety_pool_failing is True, (
            f"{_FUTILE_RESPAWN_COUNT} safety rebuilds, none of which reached readiness, left the pool "
            "reported healthy: the intentional-replacement window suppressed every one of them, so no "
            "consumer of the crash-loop signal can see a pool that cannot start at all"
        )

    def test_intentional_window_closes_on_readiness_and_later_crashes_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An intentional cycle that does reach readiness closes its window, and later crashes are counted.

        The suppression exists so a burst of whole-card pauses does not read as a crash loop. That must keep
        working through the real pause entry point, and it must not outlive the placement change it covers.
        """
        pm = make_testable_process_manager(safety_on_gpu=True)
        lifecycle = pm._process_lifecycle
        _stub_safety_start(lifecycle, monkeypatch)

        fake_time = _AdvanceableTime()
        monkeypatch.setattr(
            "horde_worker_regen.process_management.lifecycle.process_lifecycle.time",
            fake_time,
        )

        assert lifecycle.pause_safety_on_gpu() is True
        _drive_rebuild_to_completion(lifecycle)
        assert lifecycle._safety_recovery_history == []

        _safety_child_reaches_readiness(lifecycle)
        assert lifecycle._safety_replacement_intentional_until_ready is False

        for _ in range(SAFETY_CRASH_LOOP_MAX + 1):
            _crash_safety_during_startup(lifecycle)
            fake_time.advance(_BACKOFF_CLEARING_RESPAWN_SECONDS)

        assert lifecycle.safety_pool_failing is True


class TestSafetyRespawnFutility:
    """Consecutive rebuilds that never reach readiness must be classified structural, whatever their spacing."""

    def test_slow_consecutive_start_failures_declare_the_pool_unrecoverable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rebuilds too slow for the sliding window must still be recognised as a pool that cannot start.

        A deterministic start failure costs a full cold start each time, so the failures arrive spaced out.
        Counting only within a window means the worker respawns such a pool forever; the streak itself is the
        evidence, and the wedge assessment has to see it.
        """
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle
        coordinator = pm._recovery_coordinator
        _stub_safety_start(lifecycle, monkeypatch)

        fake_time = _AdvanceableTime()
        monkeypatch.setattr(
            "horde_worker_regen.process_management.lifecycle.process_lifecycle.time",
            fake_time,
        )

        for _ in range(_FUTILE_RESPAWN_COUNT):
            _crash_safety_during_startup(lifecycle)
            fake_time.advance(_SLOW_RESPAWN_SECONDS)

        # No safety process ever became ready and none is deferred for headroom, so the pool's own failing
        # verdict is the only thing standing between this streak and the unrecoverable classification.
        assert coordinator.is_safety_pool_ready() is False
        assert lifecycle.has_pending_safety_starts() is False

        assert coordinator.is_safety_pool_unrecoverable() is True, (
            f"{_FUTILE_RESPAWN_COUNT} consecutive safety rebuilds, none reaching readiness, left the pool "
            "classified recoverable because they were spaced wider than the crash-loop window; the worker "
            "would respawn this pool for as long as it runs"
        )
        assert coordinator.assess_wedge() is True, (
            "a safety pool that cannot start is on every job's path, so the wedge assessment must report it "
            "and let the escalation reach its terminal rung"
        )

    def test_a_successful_load_breaks_the_start_failure_streak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rebuild streak interrupted by a safety process that does reach readiness is not structural.

        Futility is about a pool that never initialises. One that comes up, serves, and later dies is an
        ordinary crash the rebuild path is for, so the streak must not carry across the successful load.
        """
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle
        coordinator = pm._recovery_coordinator
        _stub_safety_start(lifecycle, monkeypatch)

        fake_time = _AdvanceableTime()
        monkeypatch.setattr(
            "horde_worker_regen.process_management.lifecycle.process_lifecycle.time",
            fake_time,
        )

        for _ in range(2):
            _crash_safety_during_startup(lifecycle)
            fake_time.advance(_SLOW_RESPAWN_SECONDS)

        _safety_child_reaches_readiness(lifecycle)
        fake_time.advance(_SLOW_RESPAWN_SECONDS)

        for _ in range(2):
            _crash_safety_during_startup(lifecycle)
            fake_time.advance(_SLOW_RESPAWN_SECONDS)

        assert coordinator.is_safety_pool_unrecoverable() is False

    def test_single_safety_crash_respawns_without_declaring_the_pool_failing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One safety crash is recovered in place and escalates nothing.

        The rebuild path exists for exactly this. Neither the crash-loop signal nor the wedge assessment may
        fire on a lone crash, or every transient safety fault would drive the terminal escalation.
        """
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle
        coordinator = pm._recovery_coordinator
        _stub_safety_start(lifecycle, monkeypatch)

        _crash_safety_during_startup(lifecycle)

        assert lifecycle.safety_pool_failing is False
        assert coordinator.is_safety_pool_unrecoverable() is False
        assert coordinator.assess_wedge() is False

    def test_crash_loop_window_threshold_trips_without_an_intentional_window(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rebuilds inside the crash-loop window trip the failing signal once they pass the threshold.

        The direct pin on ``SAFETY_CRASH_LOOP_MAX``: at the threshold the pool is still merely churning, and
        the rebuild past it is what makes it a loop.
        """
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle
        _stub_safety_start(lifecycle, monkeypatch)

        fake_time = _AdvanceableTime()
        monkeypatch.setattr(
            "horde_worker_regen.process_management.lifecycle.process_lifecycle.time",
            fake_time,
        )

        for _ in range(SAFETY_CRASH_LOOP_MAX):
            _crash_safety_during_startup(lifecycle)
            fake_time.advance(_BACKOFF_CLEARING_RESPAWN_SECONDS)
        assert lifecycle.safety_pool_failing is False

        _crash_safety_during_startup(lifecycle)
        assert lifecycle.safety_pool_failing is True


class TestHeldPopGateBackstop:
    """A pop gate held with no work flowing is a wedge whatever holds it."""

    def test_long_held_gate_with_no_job_flow_is_a_wedge(self) -> None:
        """Pops gated for an hour with nothing completed must register as a wedge.

        This is the failure-independent backstop: the signal is completed work, which no gate-holding failure
        can produce on its own, so it stands whether the gate is safety, inference, or anything later added.
        """
        pm = make_testable_process_manager()
        clock = _FakeClock()
        pm._recovery_coordinator._clock = clock
        _hold_pop_gate(pm, clock, "no_safety_process", _GATE_HELD_SECONDS)

        assert pm._job_tracker.total_num_completed_jobs == 0
        assert pm._recovery_coordinator.assess_wedge() is True, (
            f"job pops were held at a gate for {_GATE_HELD_SECONDS:.0f}s with nothing completed and the "
            "worker reported no wedge: the held gate is disclosed to the log and consumed by nothing, so "
            "the escalation never opens an episode over a worker that is serving nothing at all"
        )

    def test_briefly_held_gate_is_not_a_wedge(self) -> None:
        """A gate held for a short spell is ordinary backpressure and must not register as a wedge."""
        pm = make_testable_process_manager()
        clock = _FakeClock()
        pm._recovery_coordinator._clock = clock
        _hold_pop_gate(pm, clock, "no_safety_process", _BRIEF_GATE_HELD_SECONDS)

        assert pm._recovery_coordinator.assess_wedge() is False

    def test_long_held_gate_with_work_completing_is_not_a_wedge(self) -> None:
        """A gate that has held a long time while jobs still complete is not a wedge.

        The gate name alone proves nothing: a worker whose queue is full holds a gate continuously while
        serving at full rate. Only the absence of completed work makes the hold a wedge.
        """
        pm = make_testable_process_manager()
        clock = _FakeClock()
        pm._recovery_coordinator._clock = clock
        _hold_pop_gate(pm, clock, "no_inference_process", _GATE_HELD_SECONDS)

        pm._job_tracker._total_num_completed_jobs += 5
        pm._state.last_pop_attempt_completed_at = clock.now

        assert pm._recovery_coordinator.assess_wedge() is False
