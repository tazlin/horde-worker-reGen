"""Client side of browser/served mode: attach to a running worker host over a socket.

[`AttachedWorkerSupervisor`][horde_worker_regen.tui.attach.AttachedWorkerSupervisor] is a drop-in for
[`WorkerSupervisor`][horde_worker_regen.tui.worker_launcher.WorkerSupervisor] that does not own a worker.
Instead it connects to a [`WorkerHost`][horde_worker_regen.tui.worker_host.WorkerHost], reflects the
streamed snapshots/status, and turns the TUI's commands and start/stop requests into wire messages. Both
classes satisfy [`SupervisorLike`][horde_worker_regen.tui.attach.SupervisorLike], so the app is agnostic to
which one it drives.

The crucial difference from the owning supervisor is lifecycle: ``stop()`` is an explicit "stop the worker"
(the user pressed it), while ``close()`` only detaches this session. So closing a browser tab tears down the
client but leaves the worker running on the host, which is the whole point of served mode.
"""

from __future__ import annotations

import contextlib
import socket
import threading
from typing import Protocol

from loguru import logger

from horde_worker_regen.process_management.supervisor_channel import (
    SupervisorCommand,
    SupervisorControlMessage,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui import socket_protocol as sp
from horde_worker_regen.tui.worker_launcher import SupervisorStatus, WorkerProcessMode

_CONNECT_TIMEOUT_SECONDS = 5.0
_RECONNECT_BACKOFF_SECONDS = 1.0


class SupervisorLike(Protocol):
    """The supervisor surface the TUI depends on, satisfied by both the owning and attach supervisors."""

    latest_snapshot: WorkerStateSnapshot | None
    last_liveness_wall_time: float | None
    """Wall-clock time of the worker loop's last liveness signal, or None if it never reported one."""

    @property
    def status(self) -> SupervisorStatus:
        """The worker's current lifecycle status."""

    @property
    def mode(self) -> WorkerProcessMode:
        """Which worker implementation is in use (real/fake)."""

    @property
    def restart_attempts(self) -> int:
        """How many consecutive worker restarts have been attempted."""

    def is_alive(self) -> bool:
        """Whether the worker process is currently running."""

    def tick(self) -> None:
        """Advance the supervisor (drain state and handle restarts)."""

    def start(self) -> None:
        """Start the worker."""

    def stop(self, *, timeout: float = ...) -> None:
        """Stop the worker (an explicit control action)."""

    def restart(self) -> None:
        """Restart the worker (stop then start)."""

    def close(self) -> None:
        """Release the supervisor as the frontend exits (the worker's fate depends on the implementor)."""

    def request_pause(self) -> bool:
        """Ask the worker to stop popping new jobs."""

    def request_resume(self) -> bool:
        """Ask the worker to resume popping jobs."""

    def request_reload_config(self) -> bool:
        """Ask the worker to reload bridgeData.yaml."""

    def request_pause_downloads(self) -> bool:
        """Ask the worker to hold background downloads."""

    def request_resume_downloads(self) -> bool:
        """Ask the worker to resume background downloads."""

    def request_download_rate_limit(self, rate_limit_kbps: int) -> bool:
        """Ask the worker to set the download bandwidth cap in KB/s."""


class AttachedWorkerSupervisor:
    """Presents the supervisor interface while the real worker lives on a separate host process."""

    def __init__(
        self,
        address: tuple[str, int],
        *,
        mode: WorkerProcessMode = WorkerProcessMode.REAL,
        reconnect_backoff: float = _RECONNECT_BACKOFF_SECONDS,
    ) -> None:
        """Begin connecting to the host at ``address`` on a background reader thread."""
        self._address = address
        self._mode = mode
        self._reconnect_backoff = reconnect_backoff

        self.latest_snapshot: WorkerStateSnapshot | None = None
        self.last_liveness_wall_time: float | None = None
        self._status = SupervisorStatus.STOPPED
        self._restart_attempts = 0
        self._worker_running = False

        self._socket: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._pending_lifecycle: str | None = None
        """The latest start/stop/restart intent issued while disconnected, delivered on (re)connect.

        Without this, an auto-start or wizard "Start" fired at mount, before the background reader has
        connected, would be silently dropped and the worker would never start."""
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, name="worker-attach-reader", daemon=True)
        self._reader.start()

    @property
    def status(self) -> SupervisorStatus:
        """The worker status most recently reported by the host."""
        return self._status

    @property
    def mode(self) -> WorkerProcessMode:
        """The worker mode (the host's value once connected; the constructor default until then)."""
        return self._mode

    @property
    def restart_attempts(self) -> int:
        """The host's current restart-attempt count."""
        return self._restart_attempts

    @property
    def connected(self) -> bool:
        """Whether the client currently has a live socket to the host."""
        return self._socket is not None

    def is_alive(self) -> bool:
        """Whether the host reports its worker process as running."""
        return self._worker_running

    def tick(self) -> None:
        """No-op: the background reader keeps the snapshot and status current between calls."""

    def start(self) -> None:
        """Ask the host to start the worker (idempotent host-side)."""
        self._send_lifecycle(sp.LIFECYCLE_START)

    def stop(self, *, timeout: float = 0.0) -> None:
        """Ask the host to stop the worker (an explicit user/control action, not a session close)."""
        self._send_lifecycle(sp.LIFECYCLE_STOP)

    def restart(self) -> None:
        """Ask the host to restart the worker as a single intent (sent even across a brief disconnect)."""
        self._send_lifecycle(sp.LIFECYCLE_RESTART)

    def close(self) -> None:
        """Detach this session without stopping the worker (it lives on the host)."""
        self._stop.set()
        with self._send_lock:
            sock = self._socket
            self._socket = None
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                sock.close()

    def send_command(self, command: SupervisorControlMessage) -> bool:
        """Forward a worker control command to the host; False if not currently connected."""
        return self._send(sp.command_message(command))

    def request_pause(self) -> bool:
        """Ask the worker to stop popping new jobs (in-flight jobs finish)."""
        return self.send_command(SupervisorControlMessage(command=SupervisorCommand.PAUSE))

    def request_resume(self) -> bool:
        """Ask the worker to resume popping jobs."""
        return self.send_command(SupervisorControlMessage(command=SupervisorCommand.RESUME))

    def request_reload_config(self) -> bool:
        """Ask the worker to re-read bridgeData.yaml and hot-swap the runtime config."""
        return self.send_command(SupervisorControlMessage(command=SupervisorCommand.RELOAD_CONFIG))

    def request_pause_downloads(self) -> bool:
        """Ask the worker to hold background model downloads."""
        return self.send_command(SupervisorControlMessage(command=SupervisorCommand.PAUSE_DOWNLOADS))

    def request_resume_downloads(self) -> bool:
        """Ask the worker to resume background model downloads."""
        return self.send_command(SupervisorControlMessage(command=SupervisorCommand.RESUME_DOWNLOADS))

    def request_download_rate_limit(self, rate_limit_kbps: int) -> bool:
        """Ask the worker to set the background-download bandwidth cap in KB/s (0 clears the cap)."""
        return self.send_command(
            SupervisorControlMessage(
                command=SupervisorCommand.SET_DOWNLOAD_RATE_LIMIT,
                download_rate_limit_kbps=rate_limit_kbps,
            ),
        )

    # region transport

    def _send(self, message: dict[str, object]) -> bool:
        """Send a frame to the host under the send lock; False if the socket is down."""
        with self._send_lock:
            return self._send_locked(message)

    def _send_locked(self, message: dict[str, object]) -> bool:
        """Send a frame on the current socket; the caller must hold the send lock."""
        sock = self._socket
        if sock is None:
            return False
        try:
            sp.send_frame(sock, message)
            return True
        except OSError:
            return False

    def _send_lifecycle(self, action: str) -> None:
        """Send a lifecycle action, or buffer the latest intent to deliver once (re)connected.

        Unlike worker commands (which are transient and meaningfully fail when disconnected), a
        start/stop must reach the host even if issued before the connection is up, so the buffered
        intent is replayed on connect. Only the latest action is kept, so start-then-stop collapses
        to stop.
        """
        with self._send_lock:
            if not self._send_locked(sp.lifecycle_message(action)):
                self._pending_lifecycle = action
            else:
                self._pending_lifecycle = None

    def _read_loop(self) -> None:
        """Connect (retrying) and apply incoming frames until the session is closed."""
        while not self._stop.is_set():
            try:
                sock = socket.create_connection(self._address, timeout=_CONNECT_TIMEOUT_SECONDS)
            except OSError:
                self._mark_disconnected()
                if self._stop.wait(self._reconnect_backoff):
                    return
                continue
            sock.settimeout(None)
            with self._send_lock:
                self._socket = sock
                # Deliver any start/stop intent issued while we were disconnected.
                if self._pending_lifecycle is not None and self._send_locked(
                    sp.lifecycle_message(self._pending_lifecycle),
                ):
                    self._pending_lifecycle = None
            try:
                while not self._stop.is_set():
                    message = sp.recv_frame(sock)
                    if message is None:
                        break
                    self._apply(message)
            except (OSError, ValueError) as read_error:
                logger.debug(f"Worker host connection dropped: {read_error}")
            finally:
                with self._send_lock:
                    self._socket = None
                with contextlib.suppress(OSError):
                    sock.close()
            self._mark_disconnected()
            if self._stop.wait(self._reconnect_backoff):
                return

    def _mark_disconnected(self) -> None:
        """Reflect a lost/absent host connection as a stopped, not-running worker."""
        self._status = SupervisorStatus.STOPPED
        self._worker_running = False
        # Drop the last frame so a dropped connection does not age into a false UNRESPONSIVE.
        self.latest_snapshot = None
        self.last_liveness_wall_time = None

    def _apply(self, message: dict[str, object]) -> None:
        """Update local state from a host frame (snapshot or status; hello is ignored)."""
        message_type = message.get("type")
        if message_type == sp.MSG_SNAPSHOT:
            self.latest_snapshot = sp.parse_snapshot(message)
        elif message_type == sp.MSG_STATUS:
            self._apply_status(message)

    def _apply_status(self, message: dict[str, object]) -> None:
        """Apply a status frame's fields (status / restart count / running / mode)."""
        status_value = message.get("status")
        if isinstance(status_value, str):
            with contextlib.suppress(ValueError):
                self._status = SupervisorStatus(status_value)
                # The host streams no snapshots while stopped/restarting; drop the last frame so it
                # cannot age into a false UNRESPONSIVE on this attached session.
                if self._status in (SupervisorStatus.STOPPED, SupervisorStatus.RESTARTING):
                    self.latest_snapshot = None
                    self.last_liveness_wall_time = None
        restart_attempts = message.get("restart_attempts", 0)
        self._restart_attempts = restart_attempts if isinstance(restart_attempts, int) else 0
        self._worker_running = bool(message.get("worker_running", False))
        mode_value = message.get("mode")
        if isinstance(mode_value, str):
            with contextlib.suppress(ValueError):
                self._mode = WorkerProcessMode(mode_value)
        # Ignore the host's (stale) liveness while stopped/restarting, mirroring the snapshot drop above,
        # so it cannot age into a false UNRESPONSIVE on this attached session.
        if self._status not in (SupervisorStatus.STOPPED, SupervisorStatus.RESTARTING):
            liveness_wall_time = message.get("last_liveness_wall_time")
            if isinstance(liveness_wall_time, int | float):
                self.last_liveness_wall_time = float(liveness_wall_time)

    # endregion
