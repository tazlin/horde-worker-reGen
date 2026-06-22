"""Tests for the ``horde-worker-web --status`` / ``--stop`` control commands.

These exercise the discover/stop affordances against a real :class:`WorkerHost` bound on an ephemeral
port. The worker itself is never started: the host serves status frames regardless, so these stay fast
and need no GPU or spawned worker process.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable

import pytest

from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.tui import web
from horde_worker_regen.tui.worker_host import WorkerHost
from horde_worker_regen.tui.worker_launcher import WorkerProcessMode, WorkerSupervisor


def _wait_for(predicate: Callable[[], bool], *, timeout: float = 10.0, interval: float = 0.05) -> bool:
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


def _running_host() -> tuple[WorkerHost, threading.Thread]:
    """Start a fake-mode worker host on an ephemeral port (worker not started); return it and its thread."""
    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="ControlTest"), mode=WorkerProcessMode.FAKE)
    host = WorkerHost(supervisor, host="127.0.0.1", port=0)
    thread = threading.Thread(target=host.serve_forever, name="test-control-host", daemon=True)
    thread.start()
    assert _wait_for(lambda: host.port != 0), "host did not bind a port"
    return host, thread


def test_status_reports_not_running_on_free_port() -> None:
    """Against a port with no host, status reports nothing running and yields a non-zero exit code."""
    address = ("127.0.0.1", _free_port())
    assert web._query_host_status(address, timeout=0.5) is None
    assert web._print_host_status(address) == 1


def test_status_reports_a_running_host() -> None:
    """A live host answers status with a frame the launcher can parse and report as success."""
    host, thread = _running_host()
    try:
        address = ("127.0.0.1", host.port)
        status = web._query_host_status(address)
        assert status is not None
        assert status["type"] == "status"
        assert status["worker_running"] is False  # the worker was never started
        assert web._print_host_status(address) == 0
    finally:
        host.stop()
        thread.join(timeout=10.0)


def test_stop_shuts_a_running_host_down() -> None:
    """``--stop`` drives a detached host to unwind serve_forever and exit cleanly."""
    host, thread = _running_host()
    try:
        address = ("127.0.0.1", host.port)
        assert web._request_host_stop(address) == 0
        assert _wait_for(lambda: not thread.is_alive(), timeout=15.0), "host did not stop on request"
    finally:
        host.stop()
        thread.join(timeout=10.0)


def test_stop_reports_when_no_host() -> None:
    """``--stop`` against a free port reports nothing to stop and returns a non-zero exit code."""
    assert web._request_host_stop(("127.0.0.1", _free_port())) == 1


def test_main_status_exits_with_query_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main(["--status"])`` resolves the host port and exits with the status helper's code."""
    seen: list[tuple[str, int]] = []
    monkeypatch.setattr(web, "_print_host_status", lambda address: seen.append(address) or 0)
    with pytest.raises(SystemExit) as exc_info:
        web.main(["--status", "--host-port", "7717"])
    assert exc_info.value.code == 0
    assert seen == [("127.0.0.1", 7717)]


def test_main_stop_exits_with_request_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main(["--stop"])`` resolves the host port and exits with the stop helper's code."""
    seen: list[tuple[str, int]] = []
    monkeypatch.setattr(web, "_request_host_stop", lambda address: seen.append(address) or 1)
    with pytest.raises(SystemExit) as exc_info:
        web.main(["--stop", "--host-port", "9999"])
    assert exc_info.value.code == 1
    assert seen == [("127.0.0.1", 9999)]
