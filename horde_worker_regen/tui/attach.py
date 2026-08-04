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
import dataclasses
import socket
import threading
from typing import Protocol

from loguru import logger

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    SupervisorCommand,
    SupervisorControlMessage,
    WorkerFatalConfigError,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui import socket_protocol as sp
from horde_worker_regen.tui.worker_launcher import SupervisorStallStats, SupervisorStatus, WorkerProcessMode

_CONNECT_TIMEOUT_SECONDS = 5.0
_RECONNECT_BACKOFF_SECONDS = 1.0

_WORKER_STILL_UP_STATUSES = frozenset({SupervisorStatus.STARTING, SupervisorStatus.RUNNING})
"""Host statuses that mean the worker has not yet moved on from the lifecycle the client last saw.

A locally-recorded stop intent stays in force while the host reports one of these, because the host keeps
reporting the pre-stop lifecycle until its own supervisor picks the request up. Any other reported status
means the host has acted, so the local intent is spent."""

_TERMINAL_STATUSES = frozenset({SupervisorStatus.STOPPED, SupervisorStatus.CRASHED})
"""Host statuses that end the life of the worker incarnation whose frames are in flight.

Any snapshot still arriving under one of these describes a worker that is already gone, so it is refused
rather than applied."""


def _parse_stall_stats(message: dict[str, object]) -> SupervisorStallStats:
    """Read the host supervisor's stall counters out of a status frame, tolerating a host without them.

    A host predating these fields reports a quiet supervisor rather than a missing one: the display this
    feeds is an alarm, and inventing an alarm from an absent field would be worse than showing none.
    """
    forgiven_resets = message.get("stall_forgiven_resets", 0)
    forgiven_seconds = message.get("stall_forgiven_seconds", 0.0)
    refused_resets = message.get("stall_refused_resets", 0)
    resets = message.get("stall_resets_in_window", 0)
    forgiven_seconds_in_window = message.get("stall_forgiven_seconds_in_window", 0.0)
    maximum = message.get("stall_max_forgiven_resets", 0)
    largest_gap = message.get("largest_tick_gap_seconds", 0.0)
    return dataclasses.replace(
        SupervisorStallStats.quiet(),
        forgiven_resets=forgiven_resets if isinstance(forgiven_resets, int) else 0,
        forgiven_seconds=float(forgiven_seconds) if isinstance(forgiven_seconds, int | float) else 0.0,
        refused_resets=refused_resets if isinstance(refused_resets, int) else 0,
        resets_in_window=resets if isinstance(resets, int) else 0,
        forgiven_seconds_in_window=(
            float(forgiven_seconds_in_window) if isinstance(forgiven_seconds_in_window, int | float) else 0.0
        ),
        max_forgiven_resets=maximum if isinstance(maximum, int) else 0,
        budget_spent=bool(message.get("stall_budget_spent", False)),
        largest_tick_gap_seconds=float(largest_gap) if isinstance(largest_gap, int | float) else 0.0,
    )


class SupervisorLike(Protocol):
    """The supervisor surface the TUI depends on, satisfied by both the owning and attach supervisors."""

    latest_snapshot: WorkerStateSnapshot | None
    last_liveness_wall_time: float | None
    """Wall-clock time of the worker loop's last liveness signal, or None if it never reported one."""
    last_fatal_error: WorkerFatalConfigError | None
    """The worker's reported non-retryable config problem, or None. The attach client leaves this None
    (the host socket does not relay it), so served mode shows the generic crash message."""

    @property
    def status(self) -> SupervisorStatus:
        """The worker's current lifecycle status."""

    @property
    def mode(self) -> WorkerProcessMode:
        """Which worker implementation is in use (real/fake)."""

    @property
    def restart_attempts(self) -> int:
        """How many consecutive worker restarts have been attempted."""

    @property
    def stall_stats(self) -> SupervisorStallStats:
        """How much of the supervisor's own unavailability it has excused (its liveness, not the worker's)."""

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

    def request_graceful_stop(self, *, timeout: float = ...) -> None:
        """Begin a non-blocking graceful stop."""

    def request_restart(self, *, timeout: float = ...) -> None:
        """Begin a non-blocking stop-then-start transition."""

    def close(self) -> None:
        """Release the supervisor as the frontend exits (the worker's fate depends on the implementor)."""

    def request_pause(self) -> bool:
        """Ask the worker to stop popping new jobs."""

    def request_resume(self) -> bool:
        """Ask the worker to resume popping jobs."""

    def request_drain(self) -> bool:
        """Ask the worker to stop popping new jobs and let in-flight work finish (without exiting)."""

    def request_set_concurrency(
        self,
        *,
        target_processes: int | None = None,
        target_threads: int | None = None,
    ) -> bool:
        """Ask the worker to scale running inference processes and/or the concurrent-inference cap."""

    def request_reload_config(self) -> bool:
        """Ask the worker to reload bridgeData.yaml."""

    def request_pause_downloads(self) -> bool:
        """Ask the worker to hold background downloads."""

    def request_resume_downloads(self) -> bool:
        """Ask the worker to resume background downloads."""

    def request_download_rate_limit(self, rate_limit_kbps: int) -> bool:
        """Ask the worker to set the download bandwidth cap in KB/s."""

    def request_downloads_only_hold(self) -> bool:
        """Ask the worker to enter the download-only posture (pre-fetch models, GPU uncommitted)."""

    def request_go_live(self) -> bool:
        """Ask the worker to leave download-only mode and start serving jobs."""

    def request_download_models(self, model_names: list[str], *, include_aux: bool) -> bool:
        """Ask the worker to fetch a chosen set of models on demand."""

    def request_set_server_maintenance(self, enabled: bool) -> bool:
        """Ask the worker to set its server-side (horde) maintenance flag on or off."""

    def request_set_stats_export(self, enabled: bool) -> bool:
        """Ask the worker to enable or disable stats JSONL export."""

    def force_kill(self) -> None:
        """Force-kill the worker process tree immediately, without waiting for a graceful drain.

        The owning supervisor kills the local worker tree; the attach supervisor sends a stop intent
        and closes the connection (the host owns the real process tree).
        """


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
        self.last_fatal_error: WorkerFatalConfigError | None = None
        self._status = SupervisorStatus.STOPPED
        self._restart_attempts = 0
        self._stall_stats = SupervisorStallStats.quiet()
        self._worker_running = False
        self._stop_requested = False
        """Whether this session has asked the host to stop the worker and is still awaiting its verdict.

        The host owns the process and keeps reporting the worker as running until its own supervisor picks
        the request up, so without a local record the session would present a stop it initiated as a
        running worker gone silent."""

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
    def stall_stats(self) -> SupervisorStallStats:
        """The host supervisor's stall counters as last reported (quiet until a status frame carries them).

        The complete record belongs to the host process and is relayed on the wire; this client does not
        manufacture any local stall counters of its own.
        """
        return self._stall_stats

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
        self._stop_requested = False
        self._send_lifecycle(sp.LIFECYCLE_START)

    def stop(self, *, timeout: float = 0.0) -> None:
        """Ask the host to stop the worker, presenting the stop locally until the host confirms it.

        An explicit user/control action, not a session close. The host keeps streaming the draining
        worker's last frame and its now-frozen liveness stamp while its own status still reads running, so
        the intent is recorded and the frames dropped here; otherwise this session would show a stop it
        asked for as a worker that stopped responding.
        """
        self._stop_requested = True
        self.latest_snapshot = None
        self.last_liveness_wall_time = None
        self._status = SupervisorStatus.STOPPING
        self._send_lifecycle(sp.LIFECYCLE_STOP)

    def request_graceful_stop(self, *, timeout: float = 0.0) -> None:
        """Ask the host to stop the worker without blocking this client."""
        self.stop(timeout=timeout)

    def restart(self) -> None:
        """Ask the host to restart the worker as a single intent (sent even across a brief disconnect)."""
        self.request_restart()

    def request_restart(self, *, timeout: float = 0.0) -> None:
        """Ask the host to restart and present the intent locally until the host confirms it."""
        self._stop_requested = False
        self.latest_snapshot = None
        self.last_liveness_wall_time = None
        self._status = SupervisorStatus.RESTARTING
        self._send_lifecycle(sp.LIFECYCLE_RESTART)

    def force_kill(self) -> None:
        """Ask the host to stop the worker, then detach this session immediately.

        In attached mode the real process tree lives on the host, so this sends a stop lifecycle
        message and tears down the connection without waiting for confirmation.
        """
        self.stop()
        self.close()

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

    def request_drain(self) -> bool:
        """Ask the worker to stop popping new jobs and let in-flight work finish (without exiting)."""
        return self.send_command(SupervisorControlMessage(command=SupervisorCommand.DRAIN))

    def request_set_concurrency(
        self,
        *,
        target_processes: int | None = None,
        target_threads: int | None = None,
    ) -> bool:
        """Ask the worker to scale running inference processes and/or the concurrent-inference cap."""
        return self.send_command(
            SupervisorControlMessage(
                command=SupervisorCommand.SET_CONCURRENCY,
                target_processes=target_processes,
                target_threads=target_threads,
            ),
        )

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

    def request_downloads_only_hold(self) -> bool:
        """Ask the worker to enter the download-only posture (pre-fetch models, GPU uncommitted)."""
        return self.send_command(SupervisorControlMessage(command=SupervisorCommand.DOWNLOADS_ONLY_HOLD))

    def request_go_live(self) -> bool:
        """Ask the worker to leave download-only mode and start serving jobs."""
        return self.send_command(SupervisorControlMessage(command=SupervisorCommand.GO_LIVE))

    def request_download_models(self, model_names: list[str], *, include_aux: bool) -> bool:
        """Ask the worker to fetch a chosen set of models on demand (the TUI download picker)."""
        return self.send_command(
            SupervisorControlMessage(
                command=SupervisorCommand.DOWNLOAD_MODELS,
                download_model_names=list(model_names),
                download_include_aux=include_aux,
            ),
        )

    def request_set_server_maintenance(self, enabled: bool) -> bool:
        """Ask the worker to set its server-side (horde) maintenance flag on or off."""
        return self.send_command(
            SupervisorControlMessage(
                command=SupervisorCommand.SET_SERVER_MAINTENANCE,
                server_maintenance_enabled=enabled,
            ),
        )

    def request_set_stats_export(self, enabled: bool) -> bool:
        """Ask the worker to enable or disable stats JSONL export."""
        return self.send_command(
            SupervisorControlMessage(
                command=SupervisorCommand.SET_STATS_EXPORT,
                stats_export_enabled=enabled,
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
        self._stop_requested = False
        # Drop the last frame so a dropped connection does not age into a false UNRESPONSIVE.
        self.latest_snapshot = None
        self.last_liveness_wall_time = None

    def _apply(self, message: dict[str, object]) -> None:
        """Update local state from a host frame (snapshot or status; hello is ignored)."""
        message_type = message.get("type")
        if message_type == sp.MSG_SNAPSHOT:
            if self._accepts_worker_frames():
                self.latest_snapshot = sp.parse_snapshot(message)
        elif message_type == sp.MSG_STATUS:
            self._apply_status(message)

    def _accepts_worker_frames(self) -> bool:
        """Whether a streamed frame still describes the worker incarnation this session is presenting.

        One host broadcast writes its status frame and then its retained snapshot, so a snapshot of a
        worker that has just died lands immediately behind the status reporting the death, and behind the
        frames a locally-issued stop dropped. Acceptance is decided by the lifecycle this session is in
        rather than by the order the two frames happen to arrive in, which is what keeps a spent frame
        from being re-adopted and then aged into the unresponsive alarm by the next running lifecycle.
        """
        if self._stop_requested:
            return False
        return self._status not in _TERMINAL_STATUSES

    def _apply_status(self, message: dict[str, object]) -> None:
        """Apply a status frame's fields (status / restart count / running / mode / host stall counters)."""
        status_value = message.get("status")
        if isinstance(status_value, str):
            with contextlib.suppress(ValueError):
                reported = SupervisorStatus(status_value)
                if self._stop_requested and reported in _WORKER_STILL_UP_STATUSES:
                    # The host has not picked the stop up yet; keep presenting the intent this session
                    # issued rather than reverting to the lifecycle it is about to leave.
                    reported = SupervisorStatus.STOPPING
                else:
                    self._stop_requested = False
                self._status = reported
                # The host streams no snapshots while stopped/restarting; drop the last frame so it
                # cannot age into a false UNRESPONSIVE on this attached session.
                if self._status in (SupervisorStatus.STOPPED, SupervisorStatus.RESTARTING):
                    self.latest_snapshot = None
                    self.last_liveness_wall_time = None
        restart_attempts = message.get("restart_attempts", 0)
        self._restart_attempts = restart_attempts if isinstance(restart_attempts, int) else 0
        self._worker_running = bool(message.get("worker_running", False))
        self._stall_stats = _parse_stall_stats(message)
        mode_value = message.get("mode")
        if isinstance(mode_value, str):
            with contextlib.suppress(ValueError):
                self._mode = WorkerProcessMode(mode_value)
        # Ignore the host's (stale) liveness outside a running lifecycle, mirroring the snapshot drop
        # above, so it cannot age into a false UNRESPONSIVE on this attached session.
        if self._status not in (
            SupervisorStatus.STOPPED,
            SupervisorStatus.STOPPING,
            SupervisorStatus.RESTARTING,
        ):
            liveness_wall_time = message.get("last_liveness_wall_time")
            if isinstance(liveness_wall_time, int | float):
                self.last_liveness_wall_time = float(liveness_wall_time)

    # endregion
