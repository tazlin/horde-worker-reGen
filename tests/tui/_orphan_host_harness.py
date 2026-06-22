"""A minimal worker host for the orphan-reaping tests: it owns a tree-spawning worker, then idles.

Run as a module in a child process so a test can hard-kill it (standing in for a launcher window closed
the hard way) and assert the worker tree it owned dies with it. It lives outside the ``test_*`` module so
the ``multiprocessing`` spawn child can import the worker target by qualified name on Windows.

The worker pid and its grandchild pid are published via ``_host_stop_workers`` (the ``PIDFILE`` env var)
so the test can watch those specific OS processes.
"""

from __future__ import annotations

import time

from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.tui import worker_launcher
from horde_worker_regen.tui.worker_launcher import WorkerProcessMode, WorkerSupervisor
from tests.tui import _host_stop_workers


def main() -> None:
    """Own a tree-spawning worker via a real supervisor, then idle until the test hard-kills this process."""
    worker_launcher._target_for_mode = lambda mode: _host_stop_workers.shutdown_ignoring_tree_worker
    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="OrphanHarness"), mode=WorkerProcessMode.FAKE)
    supervisor.start()
    while True:
        time.sleep(0.5)


if __name__ == "__main__":
    main()
