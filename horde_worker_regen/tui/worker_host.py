"""A persistent worker host that owns one worker and serves its state over a localhost socket.

This is the server half of browser/served mode. ``textual-serve`` runs a fresh TUI subprocess per browser
session, so the worker cannot live inside any one session. The host owns a single
[`WorkerSupervisor`][horde_worker_regen.tui.worker_launcher.WorkerSupervisor] (reusing all of its spawn /
auto-restart / command logic) and lets any number of TUI clients attach over a socket: they receive a live
stream of snapshots and status, and send worker commands and process-lifecycle (start/stop/restart)
requests. The worker therefore survives a browser tab closing, and multiple viewers stay consistent.

All supervisor interaction is funnelled through one control thread (mirroring the TUI's single-UI-thread
model), so ``tick``, lifecycle changes, and command forwarding never run concurrently. Client reader
threads only ever read from their own socket and enqueue requests; the control thread is the only sender.

Single-instance is enforced by the listening socket: a second host on the same port fails to bind, which is
how the web launcher detects an already-running host and attaches to it instead.
"""

from __future__ import annotations

import argparse
import contextlib
import multiprocessing
import os
import queue
import socket
import threading
import time

from loguru import logger

from horde_worker_regen.app_state import default_app_state_dir
from horde_worker_regen.process_management.owned_process_registry import OwnedProcessRegistry
from horde_worker_regen.process_management.supervisor_channel import SupervisorControlMessage
from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.tui import socket_protocol as sp
from horde_worker_regen.tui import tray as tray_module
from horde_worker_regen.tui.logging_setup import setup_supervisor_file_logging
from horde_worker_regen.tui.worker_launcher import WorkerProcessMode, WorkerSupervisor

_ACCEPT_TIMEOUT_SECONDS = 0.5
"""How often the accept loop wakes to check for shutdown."""

HOST_OWNED_PIDS_FILENAME = "host_owned_pids.json"
"""Where the host records the worker pid it owns, kept distinct from the worker's own child registry.

A host that dies the hard way orphans its worker tree; the next host reads this on startup and reaps any
survivor. It is the host's worker pid only (a separate file from the worker's ``owned_pids.json``, which
tracks that worker's inference/safety children)."""


class WorkerHost:
    """Owns one worker via a supervisor and serves its live state to attached TUI clients."""

    def __init__(
        self,
        supervisor: WorkerSupervisor,
        *,
        host: str = sp.DEFAULT_HOST_ADDRESS,
        port: int = sp.DEFAULT_HOST_PORT,
        control_interval: float = 0.25,
    ) -> None:
        """Store the (unstarted) supervisor and the address to bind; does not bind until :meth:`serve_forever`."""
        self._supervisor = supervisor
        self._host = host
        self._port = port
        self._control_interval = control_interval

        self._server_socket: socket.socket | None = None
        self._clients: set[socket.socket] = set()
        self._clients_lock = threading.Lock()
        self._requests: queue.Queue[tuple[str, object]] = queue.Queue()
        self._stop = threading.Event()
        self._restart_after_stop = False
        """Set by a restart request: once the in-progress graceful stop completes, the worker is started."""
        self._threads: list[threading.Thread] = []
        self._tray: tray_module.WorkerTray | None = None

    @property
    def port(self) -> int:
        """The port the host listens on (after binding, the actual port when 0 was requested)."""
        return self._port

    def serve_forever(self) -> None:
        """Bind, accept clients, and run the control loop until :meth:`stop`. Blocks the caller.

        Raises OSError if the port is already in use; the caller (web launcher) treats that as "a host is
        already running" and attaches to it instead.
        """
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.bind((self._host, self._port))
        except OSError:
            server.close()
            raise
        server.listen(8)
        server.settimeout(_ACCEPT_TIMEOUT_SECONDS)
        self._server_socket = server
        self._port = server.getsockname()[1]
        logger.info(f"Worker host listening on {self._host}:{self._port} (mode={self._supervisor.mode.value}).")

        control = threading.Thread(target=self._control_loop, name="worker-host-control", daemon=True)
        control.start()
        self._threads.append(control)

        self._start_tray()

        try:
            self._accept_loop(server)
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Signal the host to stop serving and shut the worker down."""
        self._stop.set()

    # region tray icon

    def _start_tray(self) -> None:
        """Show a Windows tray icon for this host (a no-op elsewhere) so a detached worker stays visible."""
        self._tray = tray_module.WorkerTray(
            on_open_dashboard=self._open_dashboard,
            on_stop=self.stop,
            status_provider=self._tray_status_text,
        )
        self._tray.start()

    def _tray_status_text(self) -> str:
        """The live one-line status the tray menu shows (re-read each time the menu opens)."""
        running = "running" if self._supervisor.is_alive() else "stopped"
        return f"Worker {running} ({self._supervisor.mode.value})"

    def _open_dashboard(self) -> None:
        """Open the dashboard against this host, honouring a non-default web port from the environment."""
        port_env = os.getenv("HORDE_WORKER_WEB_PORT")
        tray_module.open_dashboard(int(port_env) if port_env else 8000)

    # endregion

    # region client handling

    def _accept_loop(self, server: socket.socket) -> None:
        """Accept client connections until stopped, spawning a reader thread per client."""
        while not self._stop.is_set():
            try:
                client, _address = server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            reader = threading.Thread(
                target=self._handle_client,
                args=(client,),
                name="worker-host-client",
                daemon=True,
            )
            reader.start()
            self._threads.append(reader)

    def _handle_client(self, client: socket.socket) -> None:
        """Greet a newly-connected client, register it for broadcasts, then read its requests.

        Only ``hello`` is sent here; the first status and snapshot arrive on the control thread's next
        broadcast (within one control interval). Keeping every supervisor read on the control thread
        avoids racing its start/stop/tick mutations from this per-client thread.
        """
        try:
            sp.send_frame(client, sp.hello_message())
        except OSError:
            client.close()
            return

        with self._clients_lock:
            self._clients.add(client)

        try:
            while not self._stop.is_set():
                message = sp.recv_frame(client)
                if message is None:
                    break
                self._enqueue_request(message)
        except (OSError, ValueError):
            pass
        finally:
            self._drop_client(client)

    def _enqueue_request(self, message: dict[str, object]) -> None:
        """Translate a client frame into a control-thread request (worker command or lifecycle action)."""
        message_type = message.get("type")
        if message_type == sp.MSG_COMMAND:
            self._requests.put(("command", sp.parse_command(message)))
        elif message_type == sp.MSG_LIFECYCLE:
            action = message.get("action")
            if isinstance(action, str):
                self._requests.put(("lifecycle", action))

    def _drop_client(self, client: socket.socket) -> None:
        """Remove a client from the broadcast set and close its socket (idempotent)."""
        with self._clients_lock:
            self._clients.discard(client)
        with contextlib.suppress(OSError):
            client.close()

    # endregion

    # region control thread

    def _control_loop(self) -> None:
        """The single owner of the supervisor: apply requests, tick, and broadcast, on an interval."""
        while not self._stop.is_set():
            self._drain_requests()
            self._supervisor.tick()
            self._apply_pending_restart()
            self._broadcast()
            time.sleep(self._control_interval)

    def _apply_pending_restart(self) -> None:
        """Start the worker once a restart-triggered graceful stop has fully completed.

        Restart is a stop-then-start, but the stop is non-blocking and completed across ticks, so the
        start has to wait until the worker has actually exited rather than firing while it still drains.
        """
        if self._restart_after_stop and not self._supervisor.is_alive():
            self._restart_after_stop = False
            self._supervisor.start()

    def _drain_requests(self) -> None:
        """Apply every queued client request to the supervisor (worker commands and lifecycle)."""
        while True:
            try:
                kind, payload = self._requests.get_nowait()
            except queue.Empty:
                return
            if kind == "command" and isinstance(payload, SupervisorControlMessage):
                self._supervisor.send_command(payload)
            elif kind == "lifecycle" and isinstance(payload, str):
                self._apply_lifecycle(payload)

    def _apply_lifecycle(self, action: str) -> None:
        """Start, stop, or restart the worker process in response to a client request.

        START is idempotent: with multiple attached sessions (each of which may auto-start), only the
        first actually spawns; a START while the worker is alive is ignored so a second worker is never
        spawned over the first.
        """
        if action == sp.LIFECYCLE_START:
            if not self._supervisor.is_alive():
                self._supervisor.start()
        elif action == sp.LIFECYCLE_STOP:
            self._restart_after_stop = False  # an explicit stop cancels any pending restart
            self._supervisor.request_graceful_stop()
        elif action == sp.LIFECYCLE_RESTART:
            self._restart_after_stop = True
            self._supervisor.request_graceful_stop()
        elif action == sp.LIFECYCLE_SHUTDOWN:
            # The launcher is exiting: stop serving so serve_forever unwinds and stops the worker cleanly.
            self._stop.set()

    def _status_message(self) -> dict[str, object]:
        """Build the current host/supervisor status frame."""
        return sp.status_message(
            status=self._supervisor.status.value,
            restart_attempts=self._supervisor.restart_attempts,
            mode=self._supervisor.mode.value,
            worker_running=self._supervisor.is_alive(),
            last_liveness_wall_time=self._supervisor.last_liveness_wall_time,
        )

    def _broadcast(self) -> None:
        """Send the latest status (and snapshot, if any) to every connected client; drop dead ones."""
        with self._clients_lock:
            clients = list(self._clients)
        if not clients:
            return
        status = self._status_message()
        snapshot = self._supervisor.latest_snapshot
        snapshot_frame = sp.snapshot_message(snapshot) if snapshot is not None else None
        for client in clients:
            try:
                sp.send_frame(client, status)
                if snapshot_frame is not None:
                    sp.send_frame(client, snapshot_frame)
            except OSError:
                self._drop_client(client)

    def _shutdown(self) -> None:
        """Stop accepting clients, remove the tray icon, stop the worker, and close all sockets.

        Each client is sent a farewell frame before its socket closes, so a watcher (the web launcher)
        learns the host is going away with intent. This matters most on the tray "Stop worker && exit"
        path: the host is a separate process from the launcher, so without this the launcher would keep
        serving a dead host as an invisible orphaned console.
        """
        self._stop.set()
        if self._tray is not None:
            self._tray.stop()
        with self._clients_lock:
            clients = list(self._clients)
            self._clients.clear()
        farewell = sp.host_shutdown_message()
        for client in clients:
            with contextlib.suppress(OSError):
                sp.send_frame(client, farewell)
                # Half-close first so the farewell is flushed and the peer reads it before EOF; a bare
                # close can race into an RST that discards the still-buffered frame (a Windows hazard the
                # reverse path documents too). The frame is best-effort regardless; the close is what the
                # watcher ultimately acts on.
                client.shutdown(socket.SHUT_WR)
            with contextlib.suppress(OSError):
                client.close()
        if self._server_socket is not None:
            with contextlib.suppress(OSError):
                self._server_socket.close()
        self._supervisor.stop()

    # endregion


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the worker-host command-line arguments (worker options mirror the TUI)."""
    parser = argparse.ArgumentParser(
        prog="horde-worker-host",
        description="Own the AI Horde worker and serve its state to attaching dashboards over a socket.",
    )
    parser.add_argument("--host", type=str, default=sp.DEFAULT_HOST_ADDRESS, help="Address to bind.")
    parser.add_argument("--port", type=int, default=sp.DEFAULT_HOST_PORT, help="Port to bind.")
    parser.add_argument(
        "--process-mode",
        choices=[mode.value for mode in WorkerProcessMode],
        default=WorkerProcessMode.REAL.value,
        help="'real' runs the GPU worker; 'fake' runs a synthetic worker.",
    )
    parser.add_argument("-e", "--load-config-from-env-vars", action="store_true", help="Load config from env vars.")
    parser.add_argument("--amd", "--amd-gpu", action="store_true", help="Enable AMD GPU optimisations.")
    parser.add_argument("-n", "--worker-name", type=str, default=None, help="Override the worker name.")
    parser.add_argument("--directml", type=int, default=None, help="Enable directml on the given device index.")
    parser.add_argument("--no-auto-restart", action="store_true", help="Do not relaunch the worker if it crashes.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point (``horde-worker-host``): own the worker and serve it over a socket."""
    multiprocessing.freeze_support()
    args = _parse_args(argv)

    # The host owns a worker the same way the TUI does, so give it its own on-disk log for launch and
    # restart diagnostics. Its console output is still useful to the web launcher, so keep stderr.
    setup_supervisor_file_logging("host")

    options = WorkerLaunchOptions(
        load_config_from_env_vars=args.load_config_from_env_vars,
        amd=args.amd,
        worker_name=args.worker_name,
        directml=args.directml,
    )

    # Reap a worker tree a prior host orphaned (a hard-closed launcher), then own this host's worker pid so
    # a successor can do the same for us. Skipped under test, where it would touch real OS processes and a
    # shared on-disk file. The Job Object (in the supervisor) is the first line of defence; this is the
    # backstop for when it could not apply (an old build, or a job-assignment that lost the spawn race).
    owned_registry: OwnedProcessRegistry | None = None
    if not os.environ.get("AI_HORDE_TESTING"):
        owned_registry = OwnedProcessRegistry(path=default_app_state_dir() / HOST_OWNED_PIDS_FILENAME)
        reaped = owned_registry.reap_orphans_from_previous_run(kill_tree=True)
        if reaped:
            logger.warning(f"Reaped an orphaned worker tree left by a previous host: {reaped}")

    supervisor = WorkerSupervisor(
        options,
        mode=WorkerProcessMode(args.process_mode),
        auto_restart=not args.no_auto_restart,
        owned_registry=owned_registry,
    )
    host = WorkerHost(supervisor, host=args.host, port=args.port)
    try:
        host.serve_forever()
    except KeyboardInterrupt:
        host.stop()
    finally:
        supervisor.stop()


if __name__ == "__main__":
    main()
