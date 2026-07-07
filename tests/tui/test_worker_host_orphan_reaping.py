"""Tests that an abruptly-killed worker host does not leave a GPU-resident worker tree behind.

A host that dies the hard way (a closed launcher window, a taskkill, a crash) skips its clean teardown,
and on Windows a child outlives its parent, so the worker and its inference/safety children would keep a
GPU resident with nothing left to stop them. Two independent guards must hold:

* a Job Object binds the worker tree to the host's lifetime, so the OS reaps the tree the moment the host
  dies (``WorkerSupervisor`` + ``tui/job_object.py``); and
* the host records its worker pid and, on the next startup, reaps any tree a prior host orphaned
  (``OwnedProcessRegistry.reap_orphans_from_previous_run(kill_tree=True)``).
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import psutil
import pytest

from horde_worker_regen.process_management.lifecycle.owned_process_registry import OwnedProcessRegistry
from tests.tui import _host_stop_workers

# These spawn real OS process trees (a host, its worker, and a grandchild) to assert OS-level reaping, so
# the module is opt-in via -m slow (skipped in a default sweep).
pytestmark = pytest.mark.slow


def _wait_for(predicate: Callable[[], bool], *, timeout: float = 20.0, interval: float = 0.1) -> bool:
    """Poll ``predicate`` until it is true or the timeout elapses; returns whether it became true."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _kill_quietly(pid: int | None) -> None:
    """Best-effort kill of a pid, ignoring an already-gone or unreadable process (test cleanup)."""
    if pid is None:
        return
    with contextlib.suppress(psutil.Error):
        psutil.Process(pid).kill()


def _spawn_idle_tree(pidfile: Path) -> None:
    """Start a standalone process that spawns a grandchild, records both pids to ``pidfile``, then idles.

    Stands in for a worker tree a prior host orphaned: the recorded parent pid is what a host would have
    persisted, and the grandchild is the GPU-resident child that a pid-only reap would leave behind.
    """
    program = (
        "import os, sys, subprocess, time;"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']);"
        "open(sys.argv[1], 'w').write(f'{os.getpid()},{g.pid}');"
        "time.sleep(600)"
    )
    subprocess.Popen([sys.executable, "-c", program, str(pidfile)])


@pytest.mark.skipif(sys.platform != "win32", reason="the Job Object kill-on-close guard is Windows-only")
@pytest.mark.e2e
def test_abrupt_host_death_kills_worker_tree(tmp_path: Path) -> None:
    """Hard-killing the host process reaps the worker and its grandchild (the Job Object guarantee)."""
    pidfile = tmp_path / "harness_pids.txt"
    env = dict(os.environ)
    env[_host_stop_workers.PIDFILE_ENV_VAR] = str(pidfile)

    harness = subprocess.Popen([sys.executable, "-m", "tests.tui._orphan_host_harness"], env=env)
    worker_pid: int | None = None
    grandchild_pid: int | None = None
    try:
        assert _wait_for(lambda: pidfile.exists() and bool(pidfile.read_text()), timeout=40.0), (
            "the harness worker never recorded its pids"
        )
        worker_pid, grandchild_pid = (int(part) for part in pidfile.read_text().split(","))
        worker_proc = psutil.Process(worker_pid)
        grandchild_proc = psutil.Process(grandchild_pid)
        assert worker_proc.is_running() and grandchild_proc.is_running()

        # Hard-kill only the host, as a closed launcher window would; the worker tree must not survive it.
        psutil.Process(harness.pid).kill()
        harness.wait(timeout=10)

        assert _wait_for(lambda: not worker_proc.is_running(), timeout=20.0), (
            "the worker was orphaned: it outlived the host that owned it"
        )
        assert _wait_for(lambda: not grandchild_proc.is_running(), timeout=20.0), (
            "the worker's grandchild was orphaned: the host's death did not reap the whole tree"
        )
    finally:
        _kill_quietly(grandchild_pid)
        _kill_quietly(worker_pid)
        _kill_quietly(harness.pid)


@pytest.mark.e2e
def test_host_startup_reaps_orphaned_worker_tree(tmp_path: Path) -> None:
    """A host's startup sweep kills an orphaned worker *tree* a prior host recorded, not just its top pid."""
    pidfile = tmp_path / "orphan_tree_pids.txt"
    registry = OwnedProcessRegistry(path=tmp_path / "host_owned_pids.json")

    _spawn_idle_tree(pidfile)
    assert _wait_for(lambda: pidfile.exists() and bool(pidfile.read_text()), timeout=30.0), (
        "the orphan tree never recorded its pids"
    )
    worker_pid, grandchild_pid = (int(part) for part in pidfile.read_text().split(","))
    worker_proc = psutil.Process(worker_pid)
    grandchild_proc = psutil.Process(grandchild_pid)
    try:
        assert worker_proc.is_running() and grandchild_proc.is_running()

        # A prior host would have persisted just the worker pid; record it the same way, then sweep.
        registry.record(os_pid=worker_pid, launch_identifier=1, process_type="worker")
        reaped = registry.reap_orphans_from_previous_run(kill_tree=True)
        assert worker_pid in reaped

        assert _wait_for(lambda: not worker_proc.is_running(), timeout=20.0), "the orphaned worker survived the sweep"
        assert _wait_for(lambda: not grandchild_proc.is_running(), timeout=20.0), (
            "the orphaned grandchild survived: the startup sweep reaped only the top pid, not the tree"
        )
    finally:
        _kill_quietly(grandchild_pid)
        _kill_quietly(worker_pid)
