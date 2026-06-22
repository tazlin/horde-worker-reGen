"""Regression tests for the tray "Stop worker & exit" path: ``WorkerHost.stop()`` must halt the worker.

The tray menu wires its stop action straight to
[`WorkerHost.stop`][horde_worker_regen.tui.worker_host.WorkerHost.stop] (see ``tui/tray.py`` and
``WorkerHost._start_tray``). ``stop()`` only sets the ``serve_forever`` stop event; the worker is actually
torn down later, in the accept loop's ``finally`` (``_shutdown`` -> ``WorkerSupervisor.stop``). That tail
had no coverage: the existing host tests assert a stop via ``LIFECYCLE_STOP`` (the dashboard button's
``request_graceful_stop``), and only ever call ``host.stop()`` in teardown where success is not asserted.

These exercise the ``host.stop()`` path end to end against real child processes and assert OS-level
termination (not just the supervisor's ``is_alive()`` bookkeeping), including that the force-kill backstop
reaps a grandchild the worker would otherwise orphan: the "worker keeps running after I clicked stop"
failure mode.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from pathlib import Path

import psutil
import pytest

from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.tui import worker_launcher
from horde_worker_regen.tui.attach import AttachedWorkerSupervisor
from horde_worker_regen.tui.worker_host import WorkerHost
from horde_worker_regen.tui.worker_launcher import WorkerProcessMode, WorkerSupervisor
from tests.tui import _host_stop_workers


def _wait_for(predicate: Callable[[], bool], *, timeout: float = 20.0, interval: float = 0.1) -> bool:
    """Poll ``predicate`` until it is true or the timeout elapses; returns whether it became true."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _process_or_none(pid: int | None) -> psutil.Process | None:
    """A live :class:`psutil.Process` for ``pid``, or None if it is missing/unreadable."""
    if pid is None:
        return None
    try:
        return psutil.Process(pid)
    except psutil.Error:
        return None


@pytest.mark.e2e
def test_host_stop_terminates_owned_worker_process() -> None:
    """The tray path (``host.stop()``) terminates the worker process the host owns, at the OS level.

    Asserts the real process is gone, not merely that ``supervisor.is_alive()`` flipped: the latter would
    also read False if the handle were dropped while the OS process lived on.
    """
    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="TrayStopTest"), mode=WorkerProcessMode.FAKE)
    host = WorkerHost(supervisor, host="127.0.0.1", port=0)
    host_thread = threading.Thread(target=host.serve_forever, name="test-tray-stop-host", daemon=True)
    host_thread.start()
    worker_proc: psutil.Process | None = None
    try:
        assert _wait_for(lambda: host.port != 0), "host did not bind a port"
        client = AttachedWorkerSupervisor(("127.0.0.1", host.port), mode=WorkerProcessMode.FAKE)
        try:
            assert _wait_for(lambda: client.connected), "client never connected to the host"
            client.start()
            assert _wait_for(supervisor.is_alive), "the host never started the worker"
            assert supervisor._process is not None
            worker_proc = _process_or_none(supervisor._process.pid)
            assert worker_proc is not None and worker_proc.is_running()
        finally:
            client.close()

        # Exactly what the tray's "Stop worker & exit" menu action invokes (tray on_stop=host.stop).
        host.stop()

        assert _wait_for(lambda: not worker_proc.is_running(), timeout=60.0), (
            "the worker process was still alive after the tray stop path (host.stop())"
        )
        assert not supervisor.is_alive()
    finally:
        host.stop()
        host_thread.join(timeout=15.0)
        if worker_proc is not None:
            with contextlib.suppress(psutil.Error):
                worker_proc.kill()


@pytest.mark.e2e
def test_host_stop_force_kills_unresponsive_worker_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that ignores SHUTDOWN is force-killed *with its whole tree*: no orphaned grandchild.

    This is the orphan failure mode behind a worker that "keeps running" after the tray stop. The tray
    path ends in ``WorkerSupervisor.stop()``; a short timeout drives its force-kill backstop, which must
    reap the worker's children (the real worker's inference/safety processes), not just the worker itself.
    """
    pidfile = tmp_path / "tree_pids.txt"
    monkeypatch.setenv(_host_stop_workers.PIDFILE_ENV_VAR, str(pidfile))
    monkeypatch.setattr(
        worker_launcher,
        "_target_for_mode",
        lambda mode: _host_stop_workers.shutdown_ignoring_tree_worker,
    )

    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="TreeKillTest"), mode=WorkerProcessMode.FAKE)
    worker_proc: psutil.Process | None = None
    grandchild_proc: psutil.Process | None = None
    supervisor.start()
    try:
        assert _wait_for(lambda: pidfile.exists() and bool(pidfile.read_text()), timeout=30.0), (
            "the worker never recorded its pids"
        )
        worker_pid, grandchild_pid = (int(part) for part in pidfile.read_text().split(","))
        worker_proc = _process_or_none(worker_pid)
        grandchild_proc = _process_or_none(grandchild_pid)
        assert worker_proc is not None and worker_proc.is_running()
        assert grandchild_proc is not None and grandchild_proc.is_running()

        # Short timeout so the (otherwise 150s) graceful window collapses straight to the force-kill path.
        supervisor.stop(timeout=3.0)

        assert _wait_for(lambda: not worker_proc.is_running(), timeout=30.0), "the worker survived the force-kill"
        assert _wait_for(lambda: not grandchild_proc.is_running(), timeout=30.0), (
            "the worker's grandchild was orphaned: the stop did not reap the whole process tree"
        )
    finally:
        for process in (grandchild_proc, worker_proc):
            if process is not None:
                with contextlib.suppress(psutil.Error):
                    process.kill()
