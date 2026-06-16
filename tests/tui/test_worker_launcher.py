"""Tests for the worker supervisor: the restart state machine (fast) and a real fake-worker spawn (e2e)."""

from __future__ import annotations

import time

import pytest

from horde_worker_regen.process_management.supervisor_channel import WorkerConfigSummary, WorkerStateSnapshot
from horde_worker_regen.run_worker import WorkerLaunchOptions
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


def test_graceful_stop_terminates_worker_that_overruns_deadline() -> None:
    """A worker that ignores the shutdown request past its deadline is force-terminated by a tick."""
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(
        WorkerLaunchOptions(),
        mode=WorkerProcessMode.FAKE,
        ctx=ctx,  # type: ignore[arg-type]
    )
    supervisor.start()
    process = ctx.last_process
    assert process is not None

    supervisor.request_graceful_stop(timeout=0.0)  # deadline is now, so the next tick must terminate
    supervisor.tick()
    assert process.terminated is True

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
