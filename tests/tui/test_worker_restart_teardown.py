"""Repro for the 'Save & Restart' deterioration: premature force-kill and orphaned subprocesses.

These tests pin two invariants the restart path must hold so that pressing 'Save + restart worker'
(or F9) never kills in-flight jobs early or leaves inference/safety subprocesses hanging:

1. Timing: the supervisor's force-kill timeout must outlast the worker's *own* drain-and-self-kill
   window. If the supervisor's deadline is shorter, it terminates the worker mid-drain, killing
   pending jobs that would have finished on their own.

2. Tree-kill: when the supervisor does force-terminate, it must kill the worker's whole process tree.
   The worker child is the parent of the inference/safety subprocesses; terminating only the direct
   child orphans those grandchildren (notably on Windows, where a child's lifetime is not bound to its
   parent), leaving them resident on the GPU with nothing left to reap them.

Both force sites are covered (the blocking ``stop()`` used by ``WorkerSupervisor.restart()`` and the
cooperative tick path used by the attached/host restart), plus the end-to-end ``restart()`` that mirrors
the user's actual action.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.lifecycle.shutdown_manager import (
    _FAULT_REPORT_GRACE_SECONDS,
    MAX_SHUTDOWN_GRACE_SECONDS,
)
from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.tui import worker_launcher
from horde_worker_regen.tui.worker_launcher import (
    GRACEFUL_STOP_TIMEOUT_SECONDS,
    WorkerProcessMode,
    WorkerSupervisor,
)

# region timing invariant


def test_stop_timeout_outlasts_worker_force_kill_ceiling() -> None:
    """The supervisor must wait longer than the worker's own hard-capped force-kill grace.

    The worker scales its drain grace with outstanding work up to ``MAX_SHUTDOWN_GRACE_SECONDS`` before
    its backstop fires. If the supervisor's timeout is below that ceiling, a worker still legitimately
    draining a queue is terminated early and its in-flight jobs are killed (the reported symptom).
    """
    assert GRACEFUL_STOP_TIMEOUT_SECONDS > MAX_SHUTDOWN_GRACE_SECONDS


def test_stop_timeout_outlasts_full_worker_self_teardown() -> None:
    """The supervisor must also outlast the worker's fault-report tail after the grace expires.

    Once the grace lapses with work outstanding, the worker spends up to ``_FAULT_REPORT_GRACE_SECONDS``
    reporting those jobs as faulted (so the horde reissues them promptly) before it self-exits. The
    supervisor's terminate must be a true last resort, firing only after that whole window, otherwise it
    pre-empts the worker's own orderly teardown.
    """
    worst_case_worker_self_teardown = MAX_SHUTDOWN_GRACE_SECONDS + _FAULT_REPORT_GRACE_SECONDS
    assert worst_case_worker_self_teardown < GRACEFUL_STOP_TIMEOUT_SECONDS


# endregion


# region tree-kill fakes


class _FakeGrandchild:
    """Stands in for an inference/safety subprocess the worker child spawned (a grandchild of the TUI)."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.alive = True


class _FakeConn:
    """A no-traffic stand-in for a pipe connection."""

    def __init__(self) -> None:
        self.closed = False

    def poll(self, timeout: float | None = None) -> bool:
        return False

    def recv(self) -> object:
        raise EOFError

    def send(self, obj: object) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeWorkerProcess:
    """A worker child that ignores the shutdown request and whose grandchildren outlive a plain terminate.

    ``terminate()`` models the real OS primitive: it ends only this direct process. The already-spawned
    grandchildren are not in the same process group / job object, so they are left alive (orphaned). Only a
    tree-aware kill should be able to reap them.
    """

    def __init__(self, pid: int, *, stubborn: bool, grandchildren: list[_FakeGrandchild]) -> None:
        self.pid = pid
        self.exitcode: int | None = None
        self.grandchildren = grandchildren
        self._alive = False
        self._stubborn = stubborn
        self.terminated = False
        self.killed = False

    def start(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        # A stubborn worker keeps draining (stays alive) regardless of the join; a cooperative one exits.
        if not self._stubborn:
            self._alive = False

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False  # the direct child dies; grandchildren are deliberately left orphaned.

    def kill(self) -> None:
        self.killed = True
        self._alive = False


class _FakeCtx:
    """A multiprocessing-context stand-in that hands out worker processes each with live grandchildren."""

    def __init__(self, *, stubborn: bool) -> None:
        self.stubborn = stubborn
        self.processes: list[_FakeWorkerProcess] = []
        self._next_pid = 1000

    def Pipe(self, duplex: bool = True) -> tuple[_FakeConn, _FakeConn]:  # noqa: N802 - mirrors ctx API
        return _FakeConn(), _FakeConn()

    def Process(self, **kwargs: object) -> _FakeWorkerProcess:  # noqa: N802 - mirrors ctx API
        self._next_pid += 1
        pid = self._next_pid
        grandchildren = [_FakeGrandchild(pid * 10 + offset) for offset in range(2)]
        process = _FakeWorkerProcess(pid, stubborn=self.stubborn, grandchildren=grandchildren)
        self.processes.append(process)
        return process


def _install_tree_kill_spy(monkeypatch: pytest.MonkeyPatch, ctx: _FakeCtx) -> list[int]:
    """Patch ``worker_launcher.kill_process_tree`` with a spy that reaps a fake process's grandchildren.

    Returns the list that records the pids the supervisor asked to tree-kill.
    """
    killed_pids: list[int] = []

    def _spy(pid: int, **kwargs: object) -> list[int]:
        killed_pids.append(pid)
        for process in ctx.processes:
            if process.pid == pid:
                process._alive = False
                process.killed = True
                for grandchild in process.grandchildren:
                    grandchild.alive = False
                return [pid, *(grandchild.pid for grandchild in process.grandchildren)]
        return [pid]

    monkeypatch.setattr(worker_launcher, "kill_process_tree", _spy, raising=False)
    return killed_pids


# endregion


# region tree-kill behaviour


def test_blocking_stop_overrun_force_kills_whole_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blocking ``stop()`` on a worker that won't exit must reap the whole tree, not just the child.

    This is the path ``WorkerSupervisor.restart()`` (F9 / Save + restart) drives. A bare ``terminate()``
    here leaves the inference/safety grandchildren orphaned on the GPU.
    """
    ctx = _FakeCtx(stubborn=True)
    killed_pids = _install_tree_kill_spy(monkeypatch, ctx)
    supervisor = WorkerSupervisor(WorkerLaunchOptions(), mode=WorkerProcessMode.FAKE, ctx=ctx)  # type: ignore[arg-type]
    supervisor.start()
    process = ctx.processes[0]

    supervisor.stop(timeout=0.0)  # join returns at once; the stubborn worker is still alive -> force path

    assert killed_pids == [process.pid], "the supervisor must tree-kill the worker by pid on overrun"
    assert all(not grandchild.alive for grandchild in process.grandchildren), (
        "inference/safety subprocesses were orphaned by the force-stop"
    )


def test_graceful_stop_overrun_force_kills_whole_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cooperative tick force path (attached/host restart) must also reap the whole tree."""
    ctx = _FakeCtx(stubborn=True)
    killed_pids = _install_tree_kill_spy(monkeypatch, ctx)
    supervisor = WorkerSupervisor(WorkerLaunchOptions(), mode=WorkerProcessMode.FAKE, ctx=ctx)  # type: ignore[arg-type]
    supervisor.start()
    process = ctx.processes[0]

    supervisor.request_graceful_stop(timeout=0.0)  # deadline is now, so the next tick must force-kill
    supervisor.tick()

    assert killed_pids == [process.pid], "the overrun tick must tree-kill the worker by pid"
    assert all(not grandchild.alive for grandchild in process.grandchildren), (
        "inference/safety subprocesses were orphaned by the overrun terminate"
    )


def test_restart_does_not_orphan_old_subprocesses(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end 'Save & Restart': the old worker's subprocesses must be dead before the new one spawns.

    Mirrors the user's action exactly. The danger is a fresh worker booting on top of the previous
    worker's still-resident inference/safety processes, double-occupying VRAM and contending for the GPU.
    """
    ctx = _FakeCtx(stubborn=True)
    killed_pids = _install_tree_kill_spy(monkeypatch, ctx)
    supervisor = WorkerSupervisor(WorkerLaunchOptions(), mode=WorkerProcessMode.FAKE, ctx=ctx)  # type: ignore[arg-type]
    supervisor.start()
    old_process = ctx.processes[0]
    old_grandchildren = old_process.grandchildren

    supervisor.restart()

    assert len(ctx.processes) == 2, "a new worker should have been spawned"
    assert old_process.pid in killed_pids, "the old worker's tree must be force-reaped during restart"
    assert all(not grandchild.alive for grandchild in old_grandchildren), (
        "restart left the previous worker's subprocesses resident alongside the new worker"
    )


# endregion
