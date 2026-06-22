"""Spawn targets for the worker-host stop regression tests (importable by the spawn child).

These live outside the ``test_*`` module so the ``multiprocessing`` ``spawn`` child can import them by
qualified name on Windows (pytest does not run in the child, so the target must import cleanly with only
the standard library). Each mimics one worker-shutdown shape the host's stop path must cope with.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

# A child that simply idles for far longer than any test; the worker is its parent, so it is the
# "grandchild" of the host and must be reaped when the host force-kills the worker tree.
_GRANDCHILD_IDLE = [sys.executable, "-c", "import time; time.sleep(600)"]

PIDFILE_ENV_VAR = "REGEN_TEST_TREE_PIDFILE"
"""Where the tree-spawning worker writes ``<worker_pid>,<grandchild_pid>`` for the test to read back."""


def _write_pids(worker_pid: int, grandchild_pid: int) -> None:
    """Publish the worker and grandchild pids to the file the parent test is polling for."""
    with open(os.environ[PIDFILE_ENV_VAR], "w", encoding="utf-8") as handle:
        handle.write(f"{worker_pid},{grandchild_pid}")


_GRANDCHILD_SPAWN_DELAY_SECONDS = 0.7
"""Wait before spawning the grandchild so the supervisor has bound this worker to its Job Object first.

The real worker spawns its inference/safety children only after heavy initialisation, by which point the
host has already taken ownership of the process; this delay reproduces that ordering so the grandchild is
created inside the job (and thus reaped with it) rather than racing the host's bind."""


def shutdown_ignoring_tree_worker(connection: object, options: object) -> None:
    """Spawn a grandchild, record both pids, then ignore SHUTDOWN so the host must force-kill the tree.

    Models the orphan failure mode: a worker that does not exit on its own (a wedge, or one whose own
    teardown stalls) would, without a tree-wide kill, leave its inference/safety grandchildren resident.
    """
    time.sleep(_GRANDCHILD_SPAWN_DELAY_SECONDS)
    grandchild = subprocess.Popen(_GRANDCHILD_IDLE)
    _write_pids(os.getpid(), grandchild.pid)
    while True:
        time.sleep(0.5)
