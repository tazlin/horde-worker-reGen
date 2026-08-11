"""Slow end-to-end proof that the native page consumes the real worker-host protocol."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Awaitable, Callable

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.tui.native_dashboard import NATIVE_ACTION_PATH, NATIVE_STATE_PATH, NativeDashboardWeb
from horde_worker_regen.tui.worker_host import WorkerHost
from horde_worker_regen.tui.worker_launcher import WorkerProcessMode, WorkerSupervisor

pytestmark = [pytest.mark.slow, pytest.mark.e2e]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows has distinct selector/proactor loop capabilities")
def test_importing_native_dashboard_preserves_windows_subprocess_support() -> None:
    """Importing worker types must not replace textual-serve's subprocess-capable proactor policy."""
    probe = textwrap.dedent(
        """
        import asyncio
        import sys

        import horde_worker_regen.tui.native_dashboard  # noqa: F401
        from horde_worker_regen.run_worker import _configure_worker_event_loop_policy

        assert isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy)

        async def open_child():
            process = await asyncio.create_subprocess_exec(sys.executable, "-c", "pass")
            assert await process.wait() == 0

        asyncio.run(open_child())
        _configure_worker_event_loop_policy()
        assert isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy)
        """,
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


async def _eventually(check: Callable[[], Awaitable[bool]], *, timeout: float = 25.0) -> bool:
    """Poll an async predicate until it succeeds or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await check():
            return True
        await asyncio.sleep(0.1)
    return False


async def test_native_http_adapter_starts_and_observes_worker_over_real_host_socket() -> None:
    """HTTP action → attach protocol → host, and host snapshot → projected HTTP state, both work end to end."""
    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="NativeProtocol"), mode=WorkerProcessMode.FAKE)
    host = WorkerHost(supervisor, host="127.0.0.1", port=0)
    host_thread = threading.Thread(target=host.serve_forever, name="test-native-worker-host", daemon=True)
    host_thread.start()
    client: TestClient | None = None
    try:
        assert await _eventually(lambda: asyncio.sleep(0, result=host.port != 0)), "host did not bind"
        native = NativeDashboardWeb(("127.0.0.1", host.port))
        app = web.Application()
        native.register(app)
        client = TestClient(TestServer(app))
        await client.start_server()

        start = await client.post(NATIVE_ACTION_PATH, json={"action": "start"})
        assert start.status == 202

        async def native_sees_running_snapshot() -> bool:
            assert client is not None
            response = await client.get(NATIVE_STATE_PATH)
            payload = await response.json()
            return bool(
                payload["connected"]
                and payload["worker_running"]
                and payload["worker_name"]
                and payload["processes_total"] == len(payload["process_states"])
            )

        assert await _eventually(native_sees_running_snapshot), "native projection never received a running snapshot"

        await client.close()
        client = None
        await asyncio.sleep(0.5)
        assert supervisor.is_alive(), "closing the native browser adapter stopped the host-owned worker"
    finally:
        if client is not None:
            await client.close()
        host.stop()
        host_thread.join(timeout=15.0)
