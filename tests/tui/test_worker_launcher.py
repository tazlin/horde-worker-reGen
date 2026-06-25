"""Tests for the worker supervisor: the restart state machine (fast) and a real fake-worker spawn (e2e)."""

from __future__ import annotations

import io
import time

import pytest
from loguru import logger

from horde_worker_regen.process_management.ipc.supervisor_channel import (
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


def _stub_snapshot() -> WorkerStateSnapshot:
    """A minimal snapshot to stand in for a now-dead worker's last frame."""
    return WorkerStateSnapshot(config=WorkerConfigSummary(dreamer_name="Test", worker_version="12.0.0"))


class _FakeConn:
    """A no-traffic stand-in for a pipe connection."""

    def __init__(self) -> None:
        self.closed = False
        self.sent: list[object] = []

    def poll(self, timeout: float | None = None) -> bool:
        return False

    def recv(self) -> object:
        raise EOFError

    def send(self, obj: object) -> None:
        self.sent.append(obj)

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    """A controllable stand-in for a spawned process."""

    def __init__(self) -> None:
        self._alive = False
        self.pid = 4321
        self.exitcode: int | None = None
        self.terminated = False

    def start(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def kill_it(self) -> None:
        self._alive = False
        self.exitcode = 1

    def join(self, timeout: float | None = None) -> None:
        self._alive = False

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False


class _FakeCtx:
    """A multiprocessing-context stand-in that records spawns."""

    def __init__(self) -> None:
        self.process_count = 0
        self.last_process: _FakeProcess | None = None

    def Pipe(self, duplex: bool = True) -> tuple[_FakeConn, _FakeConn]:  # noqa: N802 - mirrors ctx API
        return _FakeConn(), _FakeConn()

    def Process(self, **kwargs: object) -> _FakeProcess:  # noqa: N802 - mirrors ctx API
        self.process_count += 1
        self.last_process = _FakeProcess()
        return self.last_process


class _ScriptedConn:
    """A pipe stand-in that yields a fixed list of frames once, then stays empty."""

    def __init__(self, frames: list[object]) -> None:
        self._frames = list(frames)
        self.closed = False

    def poll(self, timeout: float | None = None) -> bool:
        return bool(self._frames)

    def recv(self) -> object:
        return self._frames.pop(0)

    def send(self, obj: object) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_drain_records_liveness_and_marks_running_without_a_snapshot() -> None:
    """A liveness frame alone refreshes last_liveness_wall_time and confirms the worker is RUNNING."""
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(WorkerLaunchOptions(), mode=WorkerProcessMode.FAKE, ctx=ctx)  # type: ignore[arg-type]
    supervisor.start()
    assert ctx.last_process is not None  # keep the process alive so is_alive() is True

    supervisor._connection = _ScriptedConn([WorkerLivenessFrame(loop_alive_wall_time=12345.0)])  # type: ignore[assignment]
    snapshots = supervisor.drain_snapshots()

    assert snapshots == []  # a liveness frame is not a snapshot
    assert supervisor.latest_snapshot is None  # content unchanged
    assert supervisor.last_liveness_wall_time == 12345.0
    assert supervisor.status is SupervisorStatus.RUNNING


def test_restart_on_crash_relaunches() -> None:
    """A crashed worker is relaunched on the next tick when auto-restart is enabled."""
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(
        WorkerLaunchOptions(),
        mode=WorkerProcessMode.FAKE,
        ctx=ctx,  # type: ignore[arg-type]
        restart_backoff_seconds=0.0,
    )
    supervisor.start()
    assert ctx.process_count == 1

    assert ctx.last_process is not None
    ctx.last_process.kill_it()
    supervisor.tick()
    assert ctx.process_count == 2
    assert supervisor.restart_attempts == 1


def test_restart_budget_is_bounded() -> None:
    """The supervisor stops relaunching after the restart budget is exhausted."""
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(
        WorkerLaunchOptions(),
        mode=WorkerProcessMode.FAKE,
        ctx=ctx,  # type: ignore[arg-type]
        restart_backoff_seconds=0.0,
        max_restart_attempts=2,
    )
    supervisor.start()

    for _ in range(4):
        assert ctx.last_process is not None
        ctx.last_process.kill_it()
        supervisor.tick()

    assert ctx.process_count == 3  # initial + 2 restarts, then budget exhausted
    assert supervisor.status is SupervisorStatus.CRASHED


def test_recoverable_crash_shows_restarting_not_crashed() -> None:
    """A crash with restarts available reads as RESTARTING immediately, never flashing red CRASHED."""
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(
        WorkerLaunchOptions(),
        mode=WorkerProcessMode.FAKE,
        ctx=ctx,  # type: ignore[arg-type]
        restart_backoff_seconds=60.0,  # large backoff so the relaunch is deferred past this tick
    )
    supervisor.start()
    assert ctx.last_process is not None
    ctx.last_process.kill_it()

    supervisor.tick()

    assert supervisor.status is SupervisorStatus.RESTARTING
    assert ctx.process_count == 1  # backoff not yet elapsed, so no relaunch happened on this tick


def test_no_auto_restart_leaves_worker_stopped() -> None:
    """With auto-restart disabled, a crash is observed but not relaunched."""
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(
        WorkerLaunchOptions(),
        mode=WorkerProcessMode.FAKE,
        ctx=ctx,  # type: ignore[arg-type]
        auto_restart=False,
    )
    supervisor.start()
    assert ctx.last_process is not None
    ctx.last_process.kill_it()
    supervisor.tick()
    assert ctx.process_count == 1
    assert supervisor.status is SupervisorStatus.CRASHED


def _capture_supervisor_logs() -> tuple[io.StringIO, int]:
    """Attach a StringIO sink to the global loguru logger and return it with its sink id.

    The supervisor logs through loguru, whose default sink binds to the real stderr fd and so dodges
    pytest's capsys/capfd; an explicit sink on the same logger captures its output deterministically.
    """
    sink = io.StringIO()
    return sink, logger.add(sink, level="DEBUG", format="{message}")


def test_early_crash_without_snapshot_points_at_startup_log() -> None:
    """A worker that dies before ever reporting is flagged as a startup crash with a log pointer."""
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(
        WorkerLaunchOptions(),
        mode=WorkerProcessMode.FAKE,
        ctx=ctx,  # type: ignore[arg-type]
        auto_restart=False,
    )
    supervisor.start()
    assert ctx.last_process is not None
    ctx.last_process.kill_it()

    sink, sink_id = _capture_supervisor_logs()
    try:
        supervisor.tick()
    finally:
        logger.remove(sink_id)

    logged = sink.getvalue()
    assert "crashed during startup" in logged
    assert "bridge_main_startup.log" in logged


def test_crash_after_reporting_omits_startup_hint() -> None:
    """A worker that reported at least one snapshot crashed mid-run, so no startup-crash pointer is given."""
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(
        WorkerLaunchOptions(),
        mode=WorkerProcessMode.FAKE,
        ctx=ctx,  # type: ignore[arg-type]
        auto_restart=False,
    )
    supervisor.start()
    # Stand in for a worker that ran and reported before dying; this is not a startup failure.
    supervisor.latest_snapshot = _stub_snapshot()
    assert ctx.last_process is not None
    ctx.last_process.kill_it()

    sink, sink_id = _capture_supervisor_logs()
    try:
        supervisor.tick()
    finally:
        logger.remove(sink_id)

    assert "crashed during startup" not in sink.getvalue()


def test_request_graceful_stop_is_non_blocking_and_finalized_by_tick() -> None:
    """request_graceful_stop returns without joining; a later tick finalizes once the worker exits."""
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(
        WorkerLaunchOptions(),
        mode=WorkerProcessMode.FAKE,
        ctx=ctx,  # type: ignore[arg-type]
    )
    supervisor.start()
    process = ctx.last_process
    assert process is not None

    supervisor.request_graceful_stop()

    # Non-blocking: the worker was asked to shut down but NOT joined, so it is still alive right after.
    # (A blocking stop() would have joined and left it not-alive before returning.)
    assert process.is_alive()
    assert supervisor.is_alive()

    # The worker finishes draining and exits on its own; the next tick finalizes the stop.
    process.kill_it()
    supervisor.tick()
    assert not supervisor.is_alive()
    assert supervisor.status is SupervisorStatus.STOPPED
    assert ctx.process_count == 1  # an intentional stop must never trigger a relaunch


def test_graceful_stop_terminates_worker_that_overruns_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker that ignores the shutdown request past its deadline is force-killed by a tick.

    The force path tree-kills by pid (so the worker's inference/safety subprocesses cannot be orphaned),
    rather than terminating only the direct child; the spy stands in for that OS-level tree kill.
    """
    ctx = _FakeCtx()
    killed_pids: list[int] = []
    monkeypatch.setattr(
        worker_launcher,
        "kill_process_tree",
        lambda pid, **_kwargs: killed_pids.append(pid) or [pid],
    )
    supervisor = WorkerSupervisor(
        WorkerLaunchOptions(),
        mode=WorkerProcessMode.FAKE,
        ctx=ctx,  # type: ignore[arg-type]
    )
    supervisor.start()
    process = ctx.last_process
    assert process is not None

    supervisor.request_graceful_stop(timeout=0.0)  # deadline is now, so the next tick must force-kill
    supervisor.tick()
    assert killed_pids == [process.pid]

    process.kill_it()  # the tree kill takes the worker down; the next tick observes the exit
    supervisor.tick()
    assert not supervisor.is_alive()
    assert supervisor.status is SupervisorStatus.STOPPED


def test_restart_clears_stale_snapshot_and_shows_restarting() -> None:
    """A user restart drops the dead worker's last frame and presents a single RESTARTING phase.

    Guards against the false UNRESPONSIVE: with the stale snapshot cleared, the dashboard derives a
    calm INITIALIZING/RESTARTING rather than ageing the old frame into the red staleness state.
    """
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(WorkerLaunchOptions(), mode=WorkerProcessMode.FAKE, ctx=ctx)  # type: ignore[arg-type]
    supervisor.start()
    supervisor.latest_snapshot = _stub_snapshot()

    supervisor.restart()

    assert supervisor.latest_snapshot is None
    assert supervisor.status is SupervisorStatus.RESTARTING
    assert ctx.process_count == 2  # the worker was relaunched


def test_stop_clears_snapshot() -> None:
    """An explicit stop drops the last frame so a stopped worker shows no stale live data."""
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(WorkerLaunchOptions(), mode=WorkerProcessMode.FAKE, ctx=ctx)  # type: ignore[arg-type]
    supervisor.start()
    supervisor.latest_snapshot = _stub_snapshot()

    supervisor.stop()

    assert supervisor.latest_snapshot is None
    assert supervisor.status is SupervisorStatus.STOPPED


@pytest.mark.e2e
def test_fake_worker_spawns_streams_and_pauses() -> None:
    """End-to-end: spawn the real fake worker, receive snapshots, pause it, and stop cleanly."""
    supervisor = WorkerSupervisor(
        WorkerLaunchOptions(worker_name="LauncherTest"),
        mode=WorkerProcessMode.FAKE,
    )
    supervisor.start()
    try:
        deadline = time.time() + 8.0
        while time.time() < deadline and supervisor.latest_snapshot is None:
            supervisor.tick()
            time.sleep(0.1)
        assert supervisor.latest_snapshot is not None, "no snapshot received from the fake worker"
        assert supervisor.latest_snapshot.processes, "snapshot carried no processes"
        assert supervisor.is_alive()

        assert supervisor.request_pause() is True
        flipped = False
        deadline = time.time() + 4.0
        while time.time() < deadline:
            supervisor.tick()
            time.sleep(0.1)
            if supervisor.latest_snapshot.maintenance_mode:
                flipped = True
                break
        assert flipped, "pause command was not reflected in a snapshot"
    finally:
        supervisor.stop(timeout=10.0)
    assert not supervisor.is_alive()
    assert supervisor.status is SupervisorStatus.STOPPED
