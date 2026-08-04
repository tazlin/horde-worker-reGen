"""The terminal recovery rung, end to end from the worker's decision to the supervisor's relaunch.

The escalation's last rung is a fresh process, and the worker is only allowed to take it because something
is watching that will start a new one. Each half of that contract is unit-tested on its own: the worker
grants the rung when a supervisor is attached, and the supervisor relaunches a worker that exits
unexpectedly. Neither test observes the join between them, so the contract could break in the middle (a
disposition that never becomes a failed exit, or an exit status the supervisor treats as a deliberate stop)
with both halves still green, leaving an escalation whose endpoint is a stopped worker.

These tests wire the halves together at the seam: a real ``SupervisorChannel`` on the worker end of a real
pipe, a real ``WorkerSupervisor`` on the other, and the real entry-point conversion from disposition to
process status in between. Only the OS process spawn is a double: the worker's escalation cannot be
provoked from the outside (it is driven by GPU-side faults), so a spawning variant would have to fake the
fault anyway while paying minutes of real process startup per case.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import time
from collections.abc import Iterator
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management import main_entry_point
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    SupervisorChannel,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.process_management.lifecycle.worker_recovery_coordinator import RecoveryDisposition
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.tui import worker_launcher
from horde_worker_regen.tui.worker_launcher import SupervisorStatus, WorkerProcessMode, WorkerSupervisor
from tests.process_management.conftest import make_testable_process_manager
from tests.tui.test_supervisor_wedge_recovery import _FakeProcess

try:
    # On Windows a duplex Pipe yields PipeConnection; alias it so annotations match (see worker_launcher).
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore


class _ParentClosableChildEnd:
    """The child end of a pipe as the *parent* holds it: closing it must not disturb the child's own copy.

    After a real spawn the parent closes its copy of the child end (so a child exit shows up as EOF) while
    the child keeps using its own handle. With both ends in one process there is a single object, so the
    parent's close is absorbed here to preserve the cross-process meaning.
    """

    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self.parent_closed = False

    def close(self) -> None:
        self.parent_closed = True


class _FakeSpawnCtx:
    """A multiprocessing context that hands out real pipes but fake, controllable processes."""

    def __init__(self) -> None:
        self.process_count = 0
        self.last_process: _FakeProcess | None = None
        self.worker_ends: list[Connection] = []

    def Pipe(self, duplex: bool = True) -> tuple[Connection, _ParentClosableChildEnd]:  # noqa: N802 - ctx API
        parent_end, worker_end = multiprocessing.get_context("spawn").Pipe(duplex=duplex)
        self.worker_ends.append(worker_end)
        return parent_end, _ParentClosableChildEnd(worker_end)

    def Process(self, **kwargs: object) -> _FakeProcess:  # noqa: N802 - ctx API
        self.process_count += 1
        self.last_process = _FakeProcess()
        return self.last_process


@pytest.fixture
def supervised_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[WorkerSupervisor, HordeWorkerProcessManager, _FakeSpawnCtx]]:
    """Yield a started :class:`WorkerSupervisor` and a process manager holding the worker end of its pipe.

    The manager's ``_supervisor`` is a real :class:`SupervisorChannel` over that pipe, so the attachment the
    terminal rung tests for is the same object a supervised worker actually has, rather than a stand-in that
    is attached by definition.
    """
    monkeypatch.setattr(worker_launcher, "kill_process_tree", lambda pid, **_kwargs: [pid])
    ctx = _FakeSpawnCtx()
    supervisor = WorkerSupervisor(
        WorkerLaunchOptions(),
        mode=WorkerProcessMode.FAKE,
        ctx=ctx,  # type: ignore[arg-type]
        restart_backoff_seconds=0.0,
    )
    supervisor.start()

    process_manager = make_testable_process_manager(exit_on_unhandled_faults=False)
    channel = SupervisorChannel(ctx.worker_ends[0])  # pyrefly: ignore
    process_manager._supervisor = channel
    # The real abort writes an ``.abort`` sentinel into the working directory and hard-kills child
    # processes; neither is part of the contract under test, and both would leak out of the test run.
    process_manager._shutdown_manager = Mock()
    try:
        yield supervisor, process_manager, ctx
    finally:
        channel.close()
        supervisor._cleanup_process()


def _exit_status_for_session(
    monkeypatch: pytest.MonkeyPatch,
    process_manager: HordeWorkerProcessManager,
) -> int:
    """Run the real entry point around this manager's session and return the process exit status.

    ``start`` builds the real control-loop coroutine before handing it to asyncio; the harness closes it and
    invokes the terminal recovery the escalation would have requested from inside that loop. Everything
    after that (the retained disposition, session persistence, and the conversion to a process status) is
    the production path.
    """

    def _run_worker_loop(coroutine: object) -> None:
        assert hasattr(coroutine, "close")
        coroutine.close()  # type: ignore[union-attr]
        assert process_manager._request_terminal_recovery() is RecoveryDisposition.RESTART_PROCESS

    monkeypatch.setattr(asyncio, "run", _run_worker_loop)
    monkeypatch.setattr("atexit.register", Mock())
    monkeypatch.setattr("signal.signal", Mock())
    monkeypatch.setattr(main_entry_point, "verify_worker_identity", Mock())
    monkeypatch.setattr(main_entry_point, "coerce_bridge_data_to_capabilities", Mock())
    monkeypatch.setattr(main_entry_point, "HordeWorkerProcessManager", Mock(return_value=process_manager))
    monkeypatch.setattr(main_entry_point, "_persist_session_state", Mock())

    with pytest.raises(SystemExit) as exit_info:
        main_entry_point.start_working(Mock(), Mock(), Mock())
    code = exit_info.value.code
    assert isinstance(code, int)
    return code


def _assert_channel_reaches_the_supervisor(
    supervisor: WorkerSupervisor,
    process_manager: HordeWorkerProcessManager,
) -> None:
    """Prove the two ends are really joined before the rung is exercised.

    The rung turns on the worker having a supervisor, so a channel that happened to be inert would make the
    pin pass for the wrong reason. Pushing one snapshot through the worker end and reading it off the
    supervisor end establishes that the attachment is a working pipe.
    """
    channel = process_manager._supervisor
    assert isinstance(channel, SupervisorChannel)
    channel.send_snapshot(WorkerStateSnapshot(config=WorkerConfigSummary(dreamer_name="Test", worker_version="0")))

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if supervisor.drain_snapshots():
            return
        time.sleep(0.02)
    raise AssertionError("the worker end of the supervisor pipe never reached the supervisor")


def test_terminal_recovery_under_a_real_supervisor_ends_in_a_running_worker(
    monkeypatch: pytest.MonkeyPatch,
    supervised_worker: tuple[WorkerSupervisor, HordeWorkerProcessManager, _FakeSpawnCtx],
) -> None:
    """A worker attached to a real supervisor exits non-zero for recovery, and the supervisor replaces it.

    This is the whole rung: the attachment permits the exit, cleanup begins, the entry point turns the
    retained disposition into a failed process status, and the supervisor reads that exit as unexpected and
    launches a replacement. The rung is a recovery only if the last step happens.
    """
    supervisor, process_manager, ctx = supervised_worker
    _assert_channel_reaches_the_supervisor(supervisor, process_manager)

    exit_status = _exit_status_for_session(monkeypatch, process_manager)

    assert exit_status != 0, "terminal recovery must leave a failed process status for the supervisor to see"
    process_manager._shutdown_manager.abort.assert_called_once()  # type: ignore[attr-defined]

    # The worker process exits with that status; the supervisor is driven exactly as the TUI drives it.
    assert ctx.last_process is not None
    ctx.last_process.die(exitcode=exit_status)
    supervisor.tick()
    supervisor.tick()

    assert ctx.process_count == 2, "the supervisor did not relaunch the worker that exited for recovery"
    assert supervisor.is_alive()
    assert supervisor.status is not SupervisorStatus.CRASHED, "a recovery restart must not read as terminal"
