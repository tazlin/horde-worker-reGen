"""Integration tests for the worker host + attach client over a real socket, against the fake worker.

These exercise the browser-mode contract: a client attaches and receives live state, the worker survives a
client disconnecting (the whole point of the host), a second client sees the same running worker, and an
explicit stop stops it.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable

import pytest

from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.tui.attach import AttachedWorkerSupervisor
from horde_worker_regen.tui.worker_host import WorkerHost
from horde_worker_regen.tui.worker_launcher import WorkerProcessMode, WorkerSupervisor


def _wait_for(predicate: Callable[[], bool], *, timeout: float = 20.0, interval: float = 0.1) -> bool:
    """Poll ``predicate`` until it is true or the timeout elapses; returns whether it became true."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _free_port() -> int:
    """Reserve and release an ephemeral port, returning its number for a host to bind next."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


@pytest.mark.e2e
def test_attach_streams_state_survives_disconnect_and_stops() -> None:
    """A client attaches and streams state; the worker outlives a disconnect; a stop request stops it."""
    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="HostTest"), mode=WorkerProcessMode.FAKE)
    host = WorkerHost(supervisor, host="127.0.0.1", port=0)
    host_thread = threading.Thread(target=host.serve_forever, name="test-worker-host", daemon=True)
    host_thread.start()
    try:
        assert _wait_for(lambda: host.port != 0), "host did not bind a port"
        address = ("127.0.0.1", host.port)

        client = AttachedWorkerSupervisor(address, mode=WorkerProcessMode.FAKE)
        try:
            assert _wait_for(lambda: client.connected), "client did not connect to the host"
            client.start()  # ask the host to start the fake worker
            assert _wait_for(lambda: client.latest_snapshot is not None), "no snapshot reached the client"
            assert _wait_for(client.is_alive), "host never reported the worker running"
        finally:
            client.close()

        # The worker must outlive the client disconnecting (that is the reason the host exists).
        time.sleep(1.0)
        assert supervisor.is_alive(), "the worker stopped when the client disconnected"

        # A second client attaches and sees the same running worker, then stops it explicitly.
        second = AttachedWorkerSupervisor(address, mode=WorkerProcessMode.FAKE)
        try:
            assert _wait_for(lambda: second.latest_snapshot is not None), "second client received no snapshot"
            assert _wait_for(second.is_alive), "second client did not see the worker running"
            second.stop()
            assert _wait_for(lambda: not supervisor.is_alive(), timeout=60.0), "worker did not stop on request"
        finally:
            second.close()
    finally:
        host.stop()
        host_thread.join(timeout=15.0)


@pytest.mark.e2e
def test_start_buffered_while_disconnected_is_delivered_on_connect() -> None:
    """A start() issued before the host is reachable is buffered and delivered once the client connects.

    This is the auto-start / wizard path in browser mode: the supervisor's start() can fire at mount,
    before the background reader has connected, and must not be lost.
    """
    port = _free_port()
    address = ("127.0.0.1", port)
    client = AttachedWorkerSupervisor(address, mode=WorkerProcessMode.FAKE, reconnect_backoff=0.2)
    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="BufferTest"), mode=WorkerProcessMode.FAKE)
    host = WorkerHost(supervisor, host="127.0.0.1", port=port)
    host_thread = threading.Thread(target=host.serve_forever, name="test-buffer-host", daemon=True)
    try:
        client.start()  # nothing is listening yet, so this must be buffered rather than dropped
        host_thread.start()
        assert _wait_for(supervisor.is_alive), "the buffered start was not delivered after the host came up"
    finally:
        client.close()
        host.stop()
        host_thread.join(timeout=15.0)


@pytest.mark.e2e
def test_host_start_is_idempotent_no_double_spawn() -> None:
    """A second start() while the worker is already running does not respawn or duplicate it."""
    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="IdempotentTest"), mode=WorkerProcessMode.FAKE)
    host = WorkerHost(supervisor, host="127.0.0.1", port=0)
    host_thread = threading.Thread(target=host.serve_forever, name="test-idempotent-host", daemon=True)
    host_thread.start()
    try:
        assert _wait_for(lambda: host.port != 0)
        client = AttachedWorkerSupervisor(("127.0.0.1", host.port), mode=WorkerProcessMode.FAKE)
        try:
            assert _wait_for(lambda: client.connected)
            client.start()
            assert _wait_for(supervisor.is_alive)
            running_process = supervisor._process
            assert running_process is not None
            original_pid = running_process.pid

            client.start()  # second start must be a no-op while the worker is alive
            time.sleep(1.0)

            assert supervisor.is_alive()
            assert supervisor._process is not None
            assert supervisor._process.pid == original_pid, "the worker was respawned by a redundant start"
        finally:
            client.close()
    finally:
        host.stop()
        host_thread.join(timeout=15.0)


def _running_pid(supervisor: WorkerSupervisor) -> int | None:
    """Return the running worker's pid, or None when it is not running."""
    process = supervisor._process
    return process.pid if process is not None and process.is_alive() else None


@pytest.mark.e2e
def test_restart_via_client_stops_then_starts_a_fresh_worker() -> None:
    """A client restart cycles the worker: the old process exits and a new one (different pid) comes up.

    This is the case the non-blocking stop made subtle: the start must wait for the graceful stop to
    complete rather than being swallowed while the worker still drains.
    """
    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="RestartTest"), mode=WorkerProcessMode.FAKE)
    host = WorkerHost(supervisor, host="127.0.0.1", port=0)
    host_thread = threading.Thread(target=host.serve_forever, name="test-restart-host", daemon=True)
    host_thread.start()
    try:
        assert _wait_for(lambda: host.port != 0)
        client = AttachedWorkerSupervisor(("127.0.0.1", host.port), mode=WorkerProcessMode.FAKE)
        try:
            assert _wait_for(lambda: client.connected)
            client.start()
            assert _wait_for(supervisor.is_alive)
            original_pid = _running_pid(supervisor)
            assert original_pid is not None

            client.restart()
            new_pid_is_up = _wait_for(
                lambda: _running_pid(supervisor) is not None and _running_pid(supervisor) != original_pid,
                timeout=60.0,
            )
            assert new_pid_is_up, "the worker did not come back up as a fresh process after restart"
        finally:
            client.close()
    finally:
        host.stop()
        host_thread.join(timeout=15.0)
