"""Specification tests for the supervisor's *alive-but-wedged* worker recovery (the true last resort).

Background
----------
The observed incident: a single inference process began the GPU force-load of an over-budget model and
went uninterruptibly dark *inside* that CUDA call; the orchestrator's control loop stopped advancing at
the same instant. Every in-worker watchdog (hung-process replacement, orphaned-job reconciliation, the
save-our-ship recovery supervisor) is a sequential step *inside* that one control-loop coroutine, so when
the loop stalls none of them run. The outer supervisor never intervened either: it only relaunches on an
actual process *exit*, and a worker frozen in a driver call is still ``is_alive()``. Worse, the worker's
liveness frames keep arriving from a daemon sender thread carrying a now-frozen ``loop_alive_wall_time``,
so any "did a frame arrive recently" heuristic reads the corpse as healthy.

The crucial property is that the trigger is entirely outside the worker's control: arbitrary job mixes,
job ordering, and host VRAM/RAM pressure decide whether a given model load wedges. So the backstop has to
be *input-agnostic*: it must recover a worker that has stopped making progress regardless of why. That is
exactly what the supervisor is positioned to do, which is why these tests live at this layer.

The contract under test
------------------------
The supervisor judges progress from the *value* of the worker's reported ``loop_alive_wall_time`` (the
wall-clock stamp of the control loop's most recent tick), not from frame arrival. While the process is
alive and not intentionally stopping, if that value has not advanced for
``worker_launcher.WEDGE_LIVENESS_TIMEOUT_SECONDS`` of parent wall-clock, the worker is treated as wedged:
its whole process tree is force-killed (orphan-proof, like a crash) and it is routed through the ordinary
restart budget. Keying on loop liveness rather than snapshot freshness is deliberate: a worker legitimately
busy loading weights keeps ticking its loop, so this never cries wolf on a slow download/preload the way a
snapshot-age threshold would.

Several of these are RED until that watchdog exists: the positive-detection cases assert a kill/relaunch
that does not happen today. The false-positive guards assert the watchdog must *not* fire on healthy or
intentionally-quiet workers; they protect the eventual implementation from the restart-churn that a naive
snapshot-age trigger previously caused.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    ProcessSnapshot,
    WorkerConfigSummary,
    WorkerLivenessFrame,
    WorkerStateSnapshot,
)
from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.tui import worker_launcher
from horde_worker_regen.tui.worker_launcher import (
    SupervisorStatus,
    WorkerProcessMode,
    WorkerSupervisor,
)

# A short, test-local wedge budget so the fake clock only has to step a few seconds. Installed onto the
# module under test via monkeypatch in each test; raising=False so the suite still runs (and the positive
# cases fail loudly) before the production constant exists.
_TEST_WEDGE_TIMEOUT = 30.0


class _Clock:
    """A hand-cranked wall clock so tests advance time deterministically instead of sleeping."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class _FeedConn:
    """A pipe stand-in that hands back frames fed to it between ticks, then reports empty.

    Mirrors the subset of ``multiprocessing.connection`` the supervisor uses: ``poll`` reports whether a
    frame is queued, ``recv`` pops the oldest, ``send`` is a no-op sink (commands are not under test here).
    """

    def __init__(self) -> None:
        self._frames: list[object] = []
        self.closed = False
        self.sent: list[object] = []

    def feed(self, frame: object) -> None:
        self._frames.append(frame)

    def poll(self, timeout: float | None = None) -> bool:
        return bool(self._frames)

    def recv(self) -> object:
        return self._frames.pop(0)

    def send(self, obj: object) -> None:
        self.sent.append(obj)

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    """A controllable spawned-process stand-in (one per spawn)."""

    _next_pid = 5000

    def __init__(self) -> None:
        self._alive = False
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.exitcode: int | None = None

    def start(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def die(self, exitcode: int = 0) -> None:
        self._alive = False
        self.exitcode = exitcode

    def join(self, timeout: float | None = None) -> None:
        self._alive = False

    def terminate(self) -> None:
        self._alive = False


class _FakeCtx:
    """A multiprocessing-context stand-in that records spawns and exposes the latest process."""

    def __init__(self) -> None:
        self.process_count = 0
        self.last_process: _FakeProcess | None = None
        self.connections: list[_FeedConn] = []

    def Pipe(self, duplex: bool = True) -> tuple[_FeedConn, _FeedConn]:  # noqa: N802 - mirrors ctx API
        parent, child = _FeedConn(), _FeedConn()
        self.connections.append(parent)
        return parent, child

    def Process(self, **kwargs: object) -> _FakeProcess:  # noqa: N802 - mirrors ctx API
        self.process_count += 1
        self.last_process = _FakeProcess()
        return self.last_process


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    auto_restart: bool = True,
    max_restart_attempts: int = 5,
    wedge_timeout: float = _TEST_WEDGE_TIMEOUT,
    stall_reset: float = 1.0e9,
) -> tuple[WorkerSupervisor, _FakeCtx, _Clock, list[int]]:
    """Build a started supervisor wired to a fake clock, a fake process tree, and a tree-kill spy.

    The tree-kill spy both records the killed pid and flips the current fake process dead, so the kill's
    effect (the process exits) is visible to the following tick, the way a real ``kill_process_tree`` would
    make ``is_alive()`` go false.

    ``stall_reset`` defaults effectively-infinite so the supervisor-self-stall guard never fires for the
    wedge-accrual tests (which step the clock in large jumps to stand in for elapsed time): those tests
    isolate the wedge logic, while the dedicated self-stall test lowers it to exercise the guard.
    """
    clock = _Clock()
    monkeypatch.setattr(worker_launcher.time, "time", clock.now)
    monkeypatch.setattr(worker_launcher, "WEDGE_LIVENESS_TIMEOUT_SECONDS", wedge_timeout, raising=False)
    monkeypatch.setattr(worker_launcher, "_SUPERVISOR_STALL_RESET_SECONDS", stall_reset, raising=False)

    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(
        WorkerLaunchOptions(),
        mode=WorkerProcessMode.FAKE,
        ctx=ctx,  # type: ignore[arg-type]
        auto_restart=auto_restart,
        max_restart_attempts=max_restart_attempts,
        restart_backoff_seconds=0.0,
    )

    killed_pids: list[int] = []

    def _spy_tree_kill(pid: int, **_kwargs: object) -> list[int]:
        killed_pids.append(pid)
        if ctx.last_process is not None and ctx.last_process.pid == pid:
            ctx.last_process.die(exitcode=-9)
        return [pid]

    monkeypatch.setattr(worker_launcher, "kill_process_tree", _spy_tree_kill)

    supervisor.start()
    return supervisor, ctx, clock, killed_pids


def _feed_liveness(supervisor: WorkerSupervisor, ctx: _FakeCtx, loop_alive_wall_time: float) -> None:
    """Deliver one liveness frame to the supervisor's current pipe (as the daemon sender thread would)."""
    conn = supervisor._connection
    assert isinstance(conn, _FeedConn)
    conn.feed(WorkerLivenessFrame(loop_alive_wall_time=loop_alive_wall_time))


def _preloading_snapshot() -> WorkerStateSnapshot:
    """A snapshot showing one inference process mid-preload: legitimately busy, not wedged."""
    return WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Test", worker_version="12.0.0"),
        processes=[
            ProcessSnapshot(
                process_id=1,
                process_type="INFERENCE",
                last_process_state="PRELOADING_MODEL",
                is_alive=True,
                is_busy=True,
            ),
        ],
    )


# --------------------------------------------------------------------------------------------------------
# Positive detection: the wedge must be caught and the worker recovered.
# --------------------------------------------------------------------------------------------------------


def test_frozen_liveness_with_frames_still_arriving_triggers_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exemplar: frames keep arriving with a FROZEN loop_alive_wall_time, yet the worker is wedged.

    This is the insidious shape: the daemon sender thread is alive and shipping liveness frames, so the
    process looks like it is "still talking", but the control loop behind those frames has stopped. The
    watchdog must judge the unchanged timestamp value, not the frame traffic, and recover the worker.
    """
    supervisor, ctx, clock, killed_pids = _install(monkeypatch)
    wedged_pid = ctx.last_process.pid  # type: ignore[union-attr]

    frozen_stamp = clock.now()
    _feed_liveness(supervisor, ctx, frozen_stamp)
    supervisor.tick()  # establishes the liveness baseline

    # Time marches on in the parent; the worker keeps emitting the same stale stamp every tick.
    for _ in range(5):
        clock.advance(_TEST_WEDGE_TIMEOUT / 4)
        _feed_liveness(supervisor, ctx, frozen_stamp)
        supervisor.tick()

    assert killed_pids == [wedged_pid], "a wedged-but-alive worker was never force-killed"
    # The kill made the old process exit; a following tick relaunches within budget.
    supervisor.tick()
    assert ctx.process_count == 2, "the worker was not relaunched after the wedge kill"
    assert supervisor.is_alive(), "the relaunched worker should be running"
    assert supervisor.status is not SupervisorStatus.CRASHED, "a recoverable wedge must not read as terminal"


def test_total_silence_no_frames_triggers_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    """The literal log shape: after one frame, the worker goes fully silent (not even liveness frames).

    A loop wedged so hard the daemon sender cannot run either produces no further frames at all. The
    watchdog cannot wait for a frame that will never come; it must time out on parent wall-clock alone.
    """
    supervisor, ctx, clock, killed_pids = _install(monkeypatch)
    wedged_pid = ctx.last_process.pid  # type: ignore[union-attr]

    _feed_liveness(supervisor, ctx, clock.now())
    supervisor.tick()

    # No more frames are ever fed; only time passes.
    for _ in range(5):
        clock.advance(_TEST_WEDGE_TIMEOUT / 4)
        supervisor.tick()

    assert killed_pids == [wedged_pid], "a silent, alive worker was never force-killed"


def test_wedge_kill_targets_the_whole_process_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recovery must tree-kill by pid, not bare-terminate the direct child.

    A wedged worker still owns live inference/safety subprocesses holding GPU contexts. Terminating only
    the direct child would orphan them resident on the device with nothing left to reap them, so the kill
    has to take the whole tree by pid (the orphan-proof path used for graceful-stop overruns and crashes).
    """
    supervisor, ctx, clock, killed_pids = _install(monkeypatch)
    expected_pid = ctx.last_process.pid  # type: ignore[union-attr]

    frozen = clock.now()
    _feed_liveness(supervisor, ctx, frozen)
    supervisor.tick()
    clock.advance(_TEST_WEDGE_TIMEOUT + 1.0)
    _feed_liveness(supervisor, ctx, frozen)
    supervisor.tick()

    assert killed_pids == [expected_pid], "the wedge recovery did not tree-kill the worker by pid"


def test_repeated_wedges_are_bounded_by_the_restart_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker that wedges, relaunches, and wedges again must not churn forever.

    If the host conditions that cause the wedge persist (e.g. a job mix that always over-commits VRAM),
    naive recovery would loop relaunching into the same wall indefinitely. Wedge recovery shares the crash
    restart budget, so it gives up cleanly into CRASHED once the budget is exhausted.
    """
    supervisor, ctx, clock, _killed = _install(monkeypatch, max_restart_attempts=2)

    # Each iteration: establish a baseline, freeze, time out -> kill, then relaunch on the next tick.
    for _ in range(4):
        if not supervisor.is_alive():
            break
        frozen = clock.now()
        _feed_liveness(supervisor, ctx, frozen)
        supervisor.tick()
        clock.advance(_TEST_WEDGE_TIMEOUT + 1.0)
        _feed_liveness(supervisor, ctx, frozen)
        supervisor.tick()  # detects wedge, kills
        supervisor.tick()  # relaunches (or gives up if budget spent)

    assert ctx.process_count == 3, "expected initial + 2 relaunches before the budget was exhausted"
    assert supervisor.status is SupervisorStatus.CRASHED


def test_wedge_with_auto_restart_off_force_kills_but_does_not_relaunch(monkeypatch: pytest.MonkeyPatch) -> None:
    """With auto-restart disabled, a wedge must still free the GPU (kill the tree) but not relaunch.

    Leaving a frozen worker alive would pin the device and its orphan-prone children even when the operator
    has opted out of automatic relaunches; the terminal state is CRASHED, and no new process is spawned.
    """
    supervisor, ctx, clock, killed_pids = _install(monkeypatch, auto_restart=False)
    wedged_pid = ctx.last_process.pid  # type: ignore[union-attr]

    frozen = clock.now()
    _feed_liveness(supervisor, ctx, frozen)
    supervisor.tick()
    clock.advance(_TEST_WEDGE_TIMEOUT + 1.0)
    _feed_liveness(supervisor, ctx, frozen)
    supervisor.tick()

    assert killed_pids == [wedged_pid], "auto_restart=False should still kill a wedged worker to free the GPU"
    supervisor.tick()
    assert ctx.process_count == 1, "auto_restart=False must not relaunch after a wedge"
    assert supervisor.status is SupervisorStatus.CRASHED


# --------------------------------------------------------------------------------------------------------
# False-positive guards: the watchdog must never fire on a worker that is actually making progress or is
# intentionally quiet. These protect the implementation from the restart-churn a naive trigger caused.
# --------------------------------------------------------------------------------------------------------


def test_advancing_liveness_is_never_treated_as_wedged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loop that keeps ticking (advancing loop_alive_wall_time) is healthy, however much time passes."""
    supervisor, ctx, clock, killed_pids = _install(monkeypatch)

    for _ in range(10):
        clock.advance(_TEST_WEDGE_TIMEOUT)  # well past the budget every step...
        _feed_liveness(supervisor, ctx, clock.now())  # ...but the worker advances its stamp in lockstep
        supervisor.tick()

    assert killed_pids == [], "a healthy, advancing worker must never be wedge-killed"
    assert ctx.process_count == 1
    assert supervisor.status is SupervisorStatus.RUNNING


def test_long_model_load_with_live_loop_is_not_a_wedge(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuinely-busy preload (minutes of weight loading) is not a wedge while the loop still ticks.

    The orchestrator loop keeps advancing during a child's model load, so keying on loop liveness (not
    snapshot freshness) lets a legitimately slow load run well past any snapshot-age threshold without a
    spurious recovery. The stale snapshot here would trip a frame-age trigger; the loop stamp must not.
    """
    supervisor, ctx, clock, killed_pids = _install(monkeypatch)

    # The snapshot stays the same (a single model loading) for the whole window, but the loop keeps ticking.
    for _ in range(8):
        clock.advance(_TEST_WEDGE_TIMEOUT)
        conn = supervisor._connection
        assert isinstance(conn, _FeedConn)
        conn.feed(_preloading_snapshot())
        _feed_liveness(supervisor, ctx, clock.now())
        supervisor.tick()

    assert killed_pids == [], "a busy-but-live model load was misread as a wedge"
    assert ctx.process_count == 1


def test_transient_stall_then_recovery_is_not_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A brief stall under the budget that then resumes ticking must not be force-killed.

    Real control loops occasionally pause (a slow tick, a momentary lock wait). As long as the stamp
    advances again before the timeout, the worker is fine and the advance tracker resets.
    """
    supervisor, ctx, clock, killed_pids = _install(monkeypatch)

    frozen = clock.now()
    _feed_liveness(supervisor, ctx, frozen)
    supervisor.tick()

    # Stall for almost the whole budget, then resume.
    clock.advance(_TEST_WEDGE_TIMEOUT * 0.9)
    _feed_liveness(supervisor, ctx, frozen)  # still frozen, but under the timeout
    supervisor.tick()

    clock.advance(_TEST_WEDGE_TIMEOUT * 0.9)  # cumulatively past the budget, but the stamp is about to move
    _feed_liveness(supervisor, ctx, clock.now())  # loop resumed
    supervisor.tick()

    # Now let plenty more time pass with the loop continuing to tick.
    for _ in range(3):
        clock.advance(_TEST_WEDGE_TIMEOUT * 0.5)
        _feed_liveness(supervisor, ctx, clock.now())
        supervisor.tick()

    assert killed_pids == [], "a transient stall that recovered was wrongly treated as a wedge"
    assert ctx.process_count == 1


def test_no_liveness_yet_during_startup_is_not_a_wedge(monkeypatch: pytest.MonkeyPatch) -> None:
    """A freshly-spawned worker that has not reported its first liveness frame is not (yet) wedged.

    Before any tick stamp exists there is no baseline to measure non-advancement against; a still-starting
    worker (importing torch, enumerating the device) must be given the room to send its first frame rather
    than be killed for silence it has not had a chance to break.
    """
    supervisor, ctx, clock, killed_pids = _install(monkeypatch)

    # No liveness frame is ever delivered; only time passes while the worker "starts up".
    for _ in range(5):
        clock.advance(_TEST_WEDGE_TIMEOUT)
        supervisor.tick()

    assert killed_pids == [], "a worker that never reported should not be wedge-killed (only startup-graced)"
    assert ctx.process_count == 1


def test_intentional_graceful_stop_quiet_is_not_a_wedge(monkeypatch: pytest.MonkeyPatch) -> None:
    """During a graceful stop the loop legitimately stops ticking; that quiet must not trip the watchdog.

    A drain deliberately winds the control loop down, so loop_alive_wall_time naturally stops advancing.
    Termination of an overrunning graceful stop is owned by the graceful-stop deadline, not the wedge
    watchdog; the two must not both fire (a double-kill / status fight).
    """
    supervisor, ctx, clock, killed_pids = _install(monkeypatch)

    _feed_liveness(supervisor, ctx, clock.now())
    supervisor.tick()

    # Begin a graceful stop with a deadline far beyond the wedge budget; the worker is draining quietly.
    supervisor.request_graceful_stop(timeout=_TEST_WEDGE_TIMEOUT * 100)
    for _ in range(5):
        clock.advance(_TEST_WEDGE_TIMEOUT)  # past the wedge budget, but well within the graceful deadline
        supervisor.tick()

    assert killed_pids == [], "the wedge watchdog wrongly fired during an intentional graceful stop"
    assert supervisor.status is not SupervisorStatus.CRASHED


def test_crashed_worker_uses_the_exit_path_not_the_wedge_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker that actually exits is handled by the existing crash path; the wedge path must not double up.

    Guards the ordering: once the process is dead, ``is_alive()`` is false and the unexpected-exit handler
    owns recovery. The wedge check (which only applies to a *live* process) must not also fire and so must
    not tree-kill an already-dead pid.
    """
    supervisor, ctx, clock, killed_pids = _install(monkeypatch)

    _feed_liveness(supervisor, ctx, clock.now())
    supervisor.tick()

    clock.advance(_TEST_WEDGE_TIMEOUT + 1.0)  # stamp is now stale...
    ctx.last_process.die(exitcode=1)  # type: ignore[union-attr]  # ...but the worker also actually died
    supervisor.tick()

    assert killed_pids == [], "a process that already exited must not be tree-killed by the wedge path"
    assert ctx.process_count == 2, "the ordinary crash path should have relaunched the exited worker"


def test_supervisor_self_stall_does_not_blame_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frozen *supervisor* (host slept/resumed, descheduled, paused in a debugger) must not kill the worker.

    The supervisor observes the worker only when it ticks. If it does not tick for a long time, a huge gap
    appears between two consecutive ticks; that gap is the supervisor's own outage, not worker silence. It
    must re-grace the worker (reset the wedge baseline) rather than attribute its blackout to the worker and
    kill a process that may well have been healthy the whole time.
    """
    # Lower the stall-reset threshold so a single large tick gap is recognised as a supervisor stall.
    supervisor, ctx, clock, killed_pids = _install(monkeypatch, stall_reset=_TEST_WEDGE_TIMEOUT)

    _feed_liveness(supervisor, ctx, clock.now())
    supervisor.tick()  # establishes the baseline and the previous-tick wall time

    # The supervisor itself goes dark for far longer than both the stall threshold and the wedge timeout,
    # with no intervening ticks; on its first tick back, a fresh worker frame is waiting (the worker, on the
    # same machine, resumed too).
    clock.advance(_TEST_WEDGE_TIMEOUT * 10)
    _feed_liveness(supervisor, ctx, clock.now())
    supervisor.tick()

    # ...and it keeps ticking healthily afterwards.
    for _ in range(4):
        clock.advance(5.0)
        _feed_liveness(supervisor, ctx, clock.now())
        supervisor.tick()

    assert killed_pids == [], "the supervisor blamed its own stall on the worker and killed it"
    assert ctx.process_count == 1
    assert supervisor.status is SupervisorStatus.RUNNING


def test_long_idle_with_advancing_loop_is_not_a_wedge(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker idle for a long time (no jobs) still ticks its control loop, so it is not wedged.

    Idle is a common steady state: the loop keeps running and advancing its stamp even with an empty queue
    and no work to dispatch. Sustained quiet on the *job* side must never be confused with a stalled loop.
    """
    supervisor, ctx, clock, killed_pids = _install(monkeypatch)

    # Far more elapsed time than the wedge window, but the loop advances its stamp every step.
    for _ in range(50):
        clock.advance(5.0)
        _feed_liveness(supervisor, ctx, clock.now())
        supervisor.tick()

    assert killed_pids == [], "a long-idle worker with a live loop was misread as a wedge"
    assert ctx.process_count == 1
    assert supervisor.status is SupervisorStatus.RUNNING
