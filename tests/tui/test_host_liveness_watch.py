"""Tests for the web launcher's host-liveness leash and the host's shutdown farewell.

The launcher (``textual-serve``) is a separate process from the worker host; when the host exits on its
own (e.g. the tray's "Stop worker && exit") the launcher must learn of it and wind down rather than linger
as an orphaned console. These exercise that detection against a real :class:`WorkerHost` and against bare
sockets, without starting a worker or the blocking ``serve()``.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from unittest.mock import Mock

from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.tui import socket_protocol as sp
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
    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="WatchTest"), mode=WorkerProcessMode.FAKE)
    host = WorkerHost(supervisor, host="127.0.0.1", port=0)
    thread = threading.Thread(target=host.serve_forever, name="test-watch-host", daemon=True)
    thread.start()
    assert _wait_for(lambda: host.port != 0), "host did not bind a port"
    return host, thread


def test_watch_fires_when_host_stops() -> None:
    """When a live host unwinds, the watcher's ``on_host_gone`` fires (the launcher would then wind down)."""
    host, thread = _running_host()
    gone = threading.Event()
    watcher = threading.Thread(
        target=web._watch_host_liveness,
        args=(("127.0.0.1", host.port), gone.set),
        daemon=True,
    )
    watcher.start()
    try:
        # Give the watcher a moment to attach before the host goes away.
        assert _wait_for(lambda: len(host._clients) >= 1, timeout=5.0), "watcher did not attach to the host"
        host.stop()
        assert gone.wait(timeout=10.0), "watcher did not detect the host going away"
    finally:
        host.stop()
        thread.join(timeout=10.0)


def test_watch_fires_on_explicit_farewell_frame() -> None:
    """An explicit ``host_shutdown`` frame ends the watch even before the socket closes."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    gone = threading.Event()

    def serve_one_then_announce() -> None:
        conn, _ = server.accept()
        sp.send_frame(conn, sp.hello_message())
        sp.send_frame(conn, sp.host_shutdown_message("test"))
        # Hold the socket open so the watch can only have broken on the frame, not on EOF.
        assert gone.wait(timeout=10.0)
        conn.close()

    announcer = threading.Thread(target=serve_one_then_announce, daemon=True)
    announcer.start()
    watcher = threading.Thread(
        target=web._watch_host_liveness,
        args=(("127.0.0.1", port), gone.set),
        daemon=True,
    )
    watcher.start()
    try:
        assert gone.wait(timeout=10.0), "watcher did not act on the host_shutdown frame"
    finally:
        announcer.join(timeout=10.0)
        server.close()


def test_watch_holds_through_a_silent_but_connected_host() -> None:
    """A host that stays connected yet sends nothing for longer than the connect timeout is not "gone".

    The watcher connects with a short timeout so an absent host is noticed quickly; that timeout must not
    carry over into the read loop, where a control loop paused by a slow tick or a stalled broadcast would
    otherwise be mistaken for a dead host and take the dashboard down under a healthy worker.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    gone = threading.Event()
    release = threading.Event()

    def serve_one_silently() -> None:
        conn, _ = server.accept()
        sp.send_frame(conn, sp.hello_message())
        release.wait(timeout=30.0)
        conn.close()

    holder = threading.Thread(target=serve_one_silently, daemon=True)
    holder.start()
    watcher = threading.Thread(
        target=web._watch_host_liveness,
        args=(("127.0.0.1", port), gone.set),
        daemon=True,
    )
    watcher.start()
    try:
        silence = web._HOST_WATCH_CONNECT_TIMEOUT_SECONDS * 3
        assert not gone.wait(timeout=silence), "watcher treated a silent, still-connected host as gone"
        release.set()
        assert gone.wait(timeout=10.0), "watcher did not notice the host closing after the silence"
    finally:
        release.set()
        holder.join(timeout=10.0)
        server.close()


def test_watch_fires_when_host_never_comes_up() -> None:
    """A host that never binds (nothing on the port) winds the launcher down once the grace elapses."""
    gone = threading.Event()
    watcher = threading.Thread(
        target=web._watch_host_liveness,
        args=(("127.0.0.1", _free_port()), gone.set),
        kwargs={"grace_seconds": 1.0},
        daemon=True,
    )
    watcher.start()
    assert gone.wait(timeout=10.0), "watcher did not give up on an absent host"


def test_host_sends_farewell_frame_on_shutdown() -> None:
    """A connected client receives a ``host_shutdown`` frame when the host tears down."""
    host, thread = _running_host()
    try:
        sock = socket.create_connection(("127.0.0.1", host.port), timeout=2.0)
        sock.settimeout(5.0)
        assert _wait_for(lambda: len(host._clients) >= 1, timeout=5.0), "client did not attach"
        host.stop()
        saw_farewell = False
        while True:
            message = sp.recv_frame(sock)
            if message is None:
                break
            if message.get("type") == sp.MSG_HOST_SHUTDOWN:
                saw_farewell = True
                break
        sock.close()
        assert saw_farewell, "host did not announce its shutdown before closing"
    finally:
        host.stop()
        thread.join(timeout=10.0)


def test_control_loop_survives_a_failing_tick() -> None:
    """One tick raising (a failed allocation, say) does not end the host's only supervising thread."""
    supervisor = Mock()
    supervisor.tick.side_effect = [MemoryError(), None, None, None, None, None]
    host = WorkerHost(supervisor, host="127.0.0.1", port=0, control_interval=0.01)

    thread = threading.Thread(target=host._control_loop, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while supervisor.tick.call_count < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    host._stop.set()
    thread.join(timeout=5.0)

    assert supervisor.tick.call_count >= 3, "the loop must keep ticking after a tick raised"
    assert not thread.is_alive()


def test_stalled_client_is_dropped_instead_of_blocking_broadcast() -> None:
    """A client that stops reading is dropped from the broadcast set; the control thread is never held.

    Without a bound on the send, one dashboard session whose browser has stopped draining its socket
    would block ``sendall`` on the control thread forever, starving the supervisor tick and every other
    client (including the launcher's liveness watcher).
    """
    from horde_worker_regen.tui import worker_host as wh

    host, thread = _running_host()
    stalled = None
    try:
        stalled = socket.create_connection(("127.0.0.1", host.port), timeout=2.0)
        assert _wait_for(lambda: len(host._clients) >= 1, timeout=5.0), "client did not attach"
        # Never read from `stalled`. Drive broadcasts directly until the send buffers fill; each call must
        # return within the client socket timeout, and the client must end up dropped.
        deadline = time.time() + 60.0
        while time.time() < deadline and host._clients:
            started = time.time()
            host._broadcast()
            assert time.time() - started < wh._CLIENT_SOCKET_TIMEOUT_SECONDS + 2.0, "broadcast blocked"
        assert not host._clients, "stalled client was never dropped"
    finally:
        if stalled is not None:
            stalled.close()
        host.stop()
        thread.join(timeout=10.0)
