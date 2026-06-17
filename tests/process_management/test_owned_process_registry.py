"""Unit tests for the owned-process registry (orphan reaping with pid-reuse safety).

These spawn short-lived, harmless child processes (a sleeping Python interpreter) to exercise the
real psutil identity checks, rather than mocking them, so the create_time pid-reuse guard is tested
against an actual OS process. Every spawned child is force-killed in a finally block.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest

from horde_worker_regen.process_management.owned_process_registry import (
    OwnedProcessRecord,
    OwnedProcessRegistry,
    kill_process_tree,
)

_SLEEPER_CODE = "import time; time.sleep(30)"


@pytest.fixture
def sleeper() -> Iterator[subprocess.Popen[bytes]]:
    """Spawn a harmless sleeping child process and guarantee it is killed afterwards."""
    proc = subprocess.Popen([sys.executable, "-c", _SLEEPER_CODE])  # noqa: S603
    try:
        yield proc
    finally:
        with contextlib.suppress(Exception):
            proc.kill()
            proc.wait(timeout=5)


def test_record_and_forget_round_trip(tmp_path: Path, sleeper: subprocess.Popen[bytes]) -> None:
    """Recording a pid persists it; forgetting removes it; both survive a reload from disk."""
    path = tmp_path / "owned_pids.json"
    registry = OwnedProcessRegistry(path=path)

    registry.record(os_pid=sleeper.pid, launch_identifier=7, process_type="INFERENCE")
    assert path.exists()

    reloaded = OwnedProcessRegistry(path=path)
    records = reloaded._load()
    assert [r.os_pid for r in records] == [sleeper.pid]
    assert records[0].launch_identifier == 7
    assert records[0].process_type == "INFERENCE"
    assert records[0].create_time > 0

    registry.forget(sleeper.pid)
    assert OwnedProcessRegistry(path=path)._load() == []


def test_reap_kills_a_matching_orphan(tmp_path: Path, sleeper: subprocess.Popen[bytes]) -> None:
    """A recorded, still-living child with a matching identity is killed on the next startup reap."""
    path = tmp_path / "owned_pids.json"
    OwnedProcessRegistry(path=path).record(os_pid=sleeper.pid, launch_identifier=1, process_type="SAFETY")

    # A fresh registry models the next worker startup reading the prior run's file.
    killed = OwnedProcessRegistry(path=path).reap_orphans_from_previous_run()

    assert sleeper.pid in killed
    # The process is gone (give the OS a moment to reflect the kill).
    deadline = time.time() + 5
    while psutil.pid_exists(sleeper.pid) and time.time() < deadline:
        time.sleep(0.05)
    assert not _is_running(sleeper.pid)
    # The registry is reset after reaping so the same pid is never targeted twice.
    assert OwnedProcessRegistry(path=path)._load() == []


def test_reap_spares_a_reused_pid(tmp_path: Path, sleeper: subprocess.Popen[bytes]) -> None:
    """A pid whose recorded create_time no longer matches (pid reuse) must not be killed."""
    path = tmp_path / "owned_pids.json"
    registry = OwnedProcessRegistry(path=path)

    # Hand-write a record for the live pid but with a bogus create_time, modelling a pid that has
    # been recycled into an unrelated process since it was recorded.
    registry._records[sleeper.pid] = OwnedProcessRecord(
        os_pid=sleeper.pid,
        create_time=1.0,  # far from the real creation time
        launch_identifier=1,
        process_type="INFERENCE",
    )
    registry._persist()

    killed = OwnedProcessRegistry(path=path).reap_orphans_from_previous_run()

    assert sleeper.pid not in killed
    assert _is_running(sleeper.pid), "the reaper killed an innocent pid-reuse victim"


def test_reap_ignores_dead_pid(tmp_path: Path) -> None:
    """A recorded pid that no longer exists is silently skipped (no error, nothing killed)."""
    path = tmp_path / "owned_pids.json"
    proc = subprocess.Popen([sys.executable, "-c", _SLEEPER_CODE])  # noqa: S603
    pid = proc.pid
    create_time = psutil.Process(pid).create_time()
    proc.kill()
    proc.wait(timeout=5)

    registry = OwnedProcessRegistry(path=path)
    registry._records[pid] = OwnedProcessRecord(
        os_pid=pid,
        create_time=create_time,
        launch_identifier=1,
        process_type="DOWNLOAD",
    )
    registry._persist()

    assert OwnedProcessRegistry(path=path).reap_orphans_from_previous_run() == []


def test_kill_all_owned(tmp_path: Path, sleeper: subprocess.Popen[bytes]) -> None:
    """kill_all_owned terminates currently-tracked children (the atexit/signal backstop)."""
    registry = OwnedProcessRegistry(path=tmp_path / "owned_pids.json")
    registry.record(os_pid=sleeper.pid, launch_identifier=1, process_type="INFERENCE")

    killed = registry.kill_all_owned()

    assert sleeper.pid in killed
    deadline = time.time() + 5
    while psutil.pid_exists(sleeper.pid) and time.time() < deadline:
        time.sleep(0.05)
    assert not _is_running(sleeper.pid)


def test_kill_process_tree_kills_descendants(tmp_path: Path) -> None:
    """kill_process_tree reaps a parent *and* its children -- the orphan source on cancel/hung-level kill.

    A benchmark cancel that targeted only the controller left the level runner and its GPU-resident
    worker children alive (no parent-child lifetime link under spawn on Windows). This spawns a parent
    that itself spawns a child and asserts that killing the *tree* takes both down.
    """
    parent_code = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "time.sleep(60)"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code])  # noqa: S603
    child_pids: list[int] = []
    try:
        # Wait for the grandchild to come up so the tree actually has depth to orphan.
        deadline = time.time() + 10
        while time.time() < deadline:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                child_pids = [c.pid for c in psutil.Process(parent.pid).children(recursive=True)]
            if child_pids:
                break
            time.sleep(0.1)
        assert child_pids, "parent never spawned a child for the test to orphan"

        targeted = kill_process_tree(parent.pid, grace_seconds=5.0)
        assert parent.pid in targeted
        assert all(pid in targeted for pid in child_pids)

        # The parent and every descendant are gone shortly after (zombies count as dead).
        all_pids = [parent.pid, *child_pids]
        gone_deadline = time.time() + 5
        while time.time() < gone_deadline and any(_is_running(pid) for pid in all_pids):
            time.sleep(0.05)
        for pid in all_pids:
            assert not _is_running(pid), f"pid {pid} survived the tree kill"
    finally:
        with contextlib.suppress(Exception):
            parent.kill()
            parent.wait(timeout=5)
        for pid in child_pids:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                psutil.Process(pid).kill()


def _is_running(pid: int) -> bool:
    """Return whether a pid maps to a live, non-zombie process."""
    if not psutil.pid_exists(pid):
        return False
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
