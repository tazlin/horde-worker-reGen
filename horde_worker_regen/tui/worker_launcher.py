"""Launch and supervise the worker as a child process for the TUI.

The TUI owns the worker: it spawns it over a duplex pipe (no on-disk state file), reads
``WorkerStateSnapshot`` frames, sends ``SupervisorControlMessage`` commands, and relaunches the
worker if it dies unexpectedly; the structured counterpart to the worker's ``.abort`` sentinel.

Transport is isolated to :meth:`WorkerSupervisor._spawn`, :meth:`WorkerSupervisor.drain_snapshots`,
and :meth:`WorkerSupervisor.send_command`. Swapping the pipe for a localhost socket (the documented
fallback for environments where nested ``spawn`` misbehaves) touches only those three methods; the
snapshot/command models and every screen are transport-agnostic.
"""

from __future__ import annotations

import contextlib
import enum
import io
import multiprocessing
import os
import sys
import time
from collections.abc import Callable, Iterator
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import TextIO

from loguru import logger

from horde_worker_regen.process_management.supervisor_channel import (
    SupervisorCommand,
    SupervisorControlMessage,
    WorkerStateSnapshot,
)
from horde_worker_regen.run_worker import WorkerLaunchOptions

try:
    # On Windows a duplex Pipe yields PipeConnection; alias it so annotations match (see process_info.py).
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore


class WorkerProcessMode(enum.StrEnum):
    """Which worker implementation the supervisor launches."""

    REAL = "real"
    """The real hordelib-backed worker (needs a GPU and the full environment)."""
    FAKE = "fake"
    """A synthetic worker emitting believable snapshots; for TUI development, tests, and web demos."""


class SupervisorStatus(enum.StrEnum):
    """The supervisor's view of the worker process lifecycle."""

    STARTING = "starting"
    RUNNING = "running"
    CRASHED = "crashed"
    RESTARTING = "restarting"
    STOPPED = "stopped"


_HEALTHY_UPTIME_SECONDS = 60.0
"""A worker alive and reporting for this long is considered stable; the restart budget resets."""

GRACEFUL_STOP_TIMEOUT_SECONDS = 95.0
"""How long :meth:`WorkerSupervisor.stop` waits for the worker to drain and exit before terminating it.

Kept above the worker's own force-kill backstop (``shutdown_manager.MAX_SHUTDOWN_GRACE_SECONDS``, 90s)
so the worker always exits on its own first; ``terminate()`` then becomes a true last resort instead
of firing mid-drain, which previously re-orphaned in-flight jobs."""


def _stream_has_real_fd(stream: TextIO | None) -> bool:
    """Whether ``stream`` maps to a usable OS file descriptor.

    Textual's screen-capture replacements for ``sys.stdout``/``sys.stderr`` return -1 from ``fileno()``
    (rather than raising), so a plain truthiness or ``hasattr`` check would accept them; require a real,
    non-negative descriptor instead.
    """
    if stream is None:
        return False
    try:
        return stream.fileno() >= 0
    except (OSError, ValueError, io.UnsupportedOperation):
        return False


@contextlib.contextmanager
def _real_std_streams_for_spawn() -> Iterator[None]:
    """Restore the interpreter's real ``stdout``/``stderr`` for the duration of a child-process spawn.

    On POSIX, multiprocessing's resource-tracker process is (re)launched lazily, and
    ``resource_tracker.ensure_running`` passes ``sys.stderr.fileno()`` to it. While the Textual app is
    running it swaps in capture streams whose ``fileno()`` is -1, which ``fork_exec`` rejects with
    ``ValueError: bad value(s) in fds_to_keep``. A one-time warm-up at startup is not enough: if the
    tracker dies mid-session, ``ensure_running`` relaunches it on the next spawn, again under the
    redirected streams. Pointing ``sys.stdout``/``sys.stderr`` back at the originals (which keep their
    real descriptors) just for the spawn makes that handshake succeed every time.

    A no-op on Windows, where multiprocessing never spawns a resource-tracker process.
    """
    if os.name != "posix":
        yield
        return

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    # __stdout__/__stderr__ can be None or closed under pythonw/detached runs; leave such a stream as-is
    # since the original is no better than the current one.
    if _stream_has_real_fd(sys.__stdout__):
        sys.stdout = sys.__stdout__
    if _stream_has_real_fd(sys.__stderr__):
        sys.stderr = sys.__stderr__
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr


def _target_for_mode(mode: WorkerProcessMode) -> Callable[..., None]:
    """Return the top-level (picklable) spawn target for the requested mode."""
    if mode is WorkerProcessMode.FAKE:
        from horde_worker_regen.tui.mock_worker import run_mock_worker

        return run_mock_worker

    from horde_worker_regen.run_worker import run_supervised

    return run_supervised


class WorkerSupervisor:
    """Owns the worker child process and the supervisor pipe; drains state and sends control.

    Drive it by calling :meth:`tick` on a timer (the TUI uses a Textual interval): ``tick`` drains
    pending snapshots and, if the worker died unexpectedly, restarts it (bounded by an attempt
    budget and a backoff). All transport errors are swallowed so the TUI never crashes with the
    worker.
    """

    def __init__(
        self,
        options: WorkerLaunchOptions,
        *,
        mode: WorkerProcessMode = WorkerProcessMode.REAL,
        auto_restart: bool = True,
        max_restart_attempts: int = 5,
        restart_backoff_seconds: float = 3.0,
        ctx: BaseContext | None = None,
    ) -> None:
        """Initialize the supervisor (does not launch; call :meth:`start`).

        Args:
            options: The worker launch options forwarded to the child.
            mode: Which worker implementation to launch.
            auto_restart: Whether to relaunch the worker after an unexpected exit.
            max_restart_attempts: Consecutive restart attempts before giving up (reset after stable uptime).
            restart_backoff_seconds: Minimum delay between an observed crash and a relaunch.
            ctx: The multiprocessing context (defaults to a fresh ``spawn`` context).
        """
        self._options = options
        self._mode = mode
        self._auto_restart = auto_restart
        self._max_restart_attempts = max_restart_attempts
        self._restart_backoff = restart_backoff_seconds
        self._ctx = ctx if ctx is not None else multiprocessing.get_context("spawn")

        self._process: BaseProcess | None = None
        self._connection: Connection | None = None
        self._status = SupervisorStatus.STOPPED
        self._intentional_stop = False
        self._graceful_stop_deadline = 0.0
        """When > 0, a non-blocking graceful stop is in progress; ``tick`` terminates the worker if it
        has not exited by this monotonic-free wall-clock deadline (see :meth:`request_graceful_stop`)."""
        self._restart_attempts = 0
        self._last_crash_time = 0.0
        self._last_spawn_time = 0.0

        self.latest_snapshot: WorkerStateSnapshot | None = None
        self.on_snapshot: Callable[[WorkerStateSnapshot], None] | None = None
        self.on_status_change: Callable[[SupervisorStatus], None] | None = None

    @property
    def status(self) -> SupervisorStatus:
        """The current supervisor status."""
        return self._status

    @property
    def mode(self) -> WorkerProcessMode:
        """The worker implementation this supervisor launches."""
        return self._mode

    @property
    def restart_attempts(self) -> int:
        """How many consecutive restarts have been attempted since the last stable run."""
        return self._restart_attempts

    def is_alive(self) -> bool:
        """Whether the worker process is currently running."""
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        """Launch the worker child process."""
        self._intentional_stop = False
        self._graceful_stop_deadline = 0.0
        self._restart_attempts = 0
        self._spawn()

    def _spawn(self, *, status: SupervisorStatus = SupervisorStatus.STARTING) -> None:
        """Create a fresh pipe and worker process (the pipe transport seam).

        Drops any retained snapshot up front: the new worker has not reported yet, so keeping the old
        worker's last frame would let it age past the staleness threshold and read as UNRESPONSIVE
        during a relaunch. ``status`` lets a restart present a single ``RESTARTING`` phase instead of
        flickering through ``STARTING``.
        """
        self.latest_snapshot = None
        parent_connection, child_connection = self._ctx.Pipe(duplex=True)
        target = _target_for_mode(self._mode)
        process = self._ctx.Process(  # type: ignore[attr-defined]
            target=target,
            args=(child_connection, self._options),
            name=f"horde-worker-{self._mode.value}",
            daemon=False,
        )
        # Restore the real std streams across the spawn so a lazy resource-tracker (re)launch under
        # Textual's redirected streams does not crash with "bad value(s) in fds_to_keep" (POSIX only).
        with _real_std_streams_for_spawn():
            process.start()
        # The parent never uses the child's end; closing it lets us detect child exit via EOF.
        child_connection.close()

        self._process = process
        self._connection = parent_connection
        self._last_spawn_time = time.time()
        self._set_status(status)
        logger.info(f"Launched worker (mode={self._mode.value}, pid={process.pid}).")

    def _set_status(self, status: SupervisorStatus) -> None:
        """Update status and notify any observer on change."""
        if status != self._status:
            self._status = status
            if self.on_status_change is not None:
                self.on_status_change(status)

    def drain_snapshots(self) -> list[WorkerStateSnapshot]:
        """Return all snapshots currently waiting on the pipe, without blocking (the read seam)."""
        snapshots: list[WorkerStateSnapshot] = []
        connection = self._connection
        if connection is None:
            return snapshots
        try:
            while connection.poll():
                message = connection.recv()
                if isinstance(message, WorkerStateSnapshot):
                    snapshots.append(message)
        except (EOFError, OSError):
            # Pipe closed (child exiting); tick() will observe the dead process and react.
            pass

        if snapshots:
            self.latest_snapshot = snapshots[-1]
            if self.is_alive():
                self._set_status(SupervisorStatus.RUNNING)
            if time.time() - self._last_spawn_time > _HEALTHY_UPTIME_SECONDS:
                self._restart_attempts = 0
            if self.on_snapshot is not None:
                self.on_snapshot(snapshots[-1])
        return snapshots

    def send_command(self, command: SupervisorControlMessage) -> bool:
        """Send a control command to the worker. Returns ``False`` if the pipe is unusable (the write seam)."""
        connection = self._connection
        if connection is None:
            return False
        try:
            connection.send(command)
            return True
        except (OSError, ValueError):
            return False

    def tick(self) -> None:
        """Drain snapshots and advance the worker lifecycle. Call this regularly (e.g. every 0.25s).

        This is the single place the lifecycle progresses, so a cooperative (non-blocking) graceful stop
        completes here too: while the worker drains, ticks keep draining snapshots and reporting status;
        once it exits, the stop is finalized. The control loop never has to block on a join.
        """
        self.drain_snapshots()

        process = self._process
        if process is None:
            return

        if process.is_alive():
            self._terminate_if_graceful_stop_overran(process)
            return

        if self._intentional_stop:
            self._complete_graceful_stop()
            return

        self._handle_unexpected_exit(process)

    def _terminate_if_graceful_stop_overran(self, process: BaseProcess) -> None:
        """Force-terminate a worker that has not exited within its graceful-stop deadline."""
        if not self._intentional_stop or self._graceful_stop_deadline == 0.0:
            return
        if time.time() < self._graceful_stop_deadline:
            return
        logger.warning("Worker did not exit within the graceful-stop window; terminating.")
        process.terminate()
        self._graceful_stop_deadline = 0.0  # terminate() is forceful; the next tick will finalize.

    def _complete_graceful_stop(self) -> None:
        """Finalize a graceful stop once the worker has exited: clean up the pipe and mark it stopped."""
        self._cleanup_process()
        self._graceful_stop_deadline = 0.0
        self._intentional_stop = False
        self._set_status(SupervisorStatus.STOPPED)

    def _startup_crash_hint(self) -> str:
        """Point the operator at the child's crash logs when a worker died young without ever reporting.

        A worker that exits before sending a single snapshot almost certainly failed during startup, where
        the crash predates hordelib's bridge.log sink; the child's own crash-capture writes the reason to
        bridge_main_startup.log / bridge_main_console.log instead. The parent only knows the exit code, so
        without this pointer the operator has the exit code but no idea where the "why" landed.
        """
        if self.latest_snapshot is not None or (time.time() - self._last_spawn_time) >= _HEALTHY_UPTIME_SECONDS:
            return ""
        return " It never reported, so it likely crashed during startup; see logs/bridge_main_startup.log"

    def _handle_unexpected_exit(self, process: BaseProcess) -> None:
        """React to a worker that exited on its own: relaunch within budget, else mark crashed."""
        # Reserve the alarming CRASHED state for the terminal case (auto-restart off, or the restart
        # budget exhausted); a recoverable relaunch should read as a calm "Restarting…" from the instant
        # the exit is observed, not flash red first.
        startup_hint = self._startup_crash_hint()
        terminal = (not self._auto_restart) or (self._restart_attempts >= self._max_restart_attempts)
        if terminal:
            self._set_status(SupervisorStatus.CRASHED)
            if not self._auto_restart:
                logger.warning(
                    f"Worker process exited unexpectedly (exitcode={process.exitcode}); not restarting.{startup_hint}",
                )
            else:
                logger.error(
                    f"Worker exceeded the restart budget ({self._max_restart_attempts}); leaving it "
                    f"stopped.{startup_hint}",
                )
            return

        self._set_status(SupervisorStatus.RESTARTING)

        now = time.time()
        if self._last_crash_time == 0.0:
            self._last_crash_time = now
            logger.warning(
                f"Worker process exited unexpectedly (exitcode={process.exitcode}); relaunching.{startup_hint}",
            )
        if now - self._last_crash_time < self._restart_backoff:
            return  # Honour the backoff; a later tick performs the relaunch.

        self._restart_attempts += 1
        self._last_crash_time = 0.0
        logger.warning(f"Restarting worker (attempt {self._restart_attempts}/{self._max_restart_attempts}).")
        self._cleanup_process()
        self._spawn()

    # region convenience commands

    def request_pause(self) -> bool:
        """Ask the worker to stop popping new jobs (in-flight jobs finish)."""
        return self.send_command(SupervisorControlMessage(command=SupervisorCommand.PAUSE))

    def request_resume(self) -> bool:
        """Ask the worker to resume popping jobs."""
        return self.send_command(SupervisorControlMessage(command=SupervisorCommand.RESUME))

    def request_drain(self) -> bool:
        """Ask the worker to drain (stop popping; finish in-flight work)."""
        return self.send_command(SupervisorControlMessage(command=SupervisorCommand.DRAIN))

    def request_reload_config(self) -> bool:
        """Ask the worker to re-read bridgeData.yaml and hot-swap the runtime config."""
        return self.send_command(SupervisorControlMessage(command=SupervisorCommand.RELOAD_CONFIG))

    def request_restart_process(self, process_id: int) -> bool:
        """Ask the worker to replace one inference process slot."""
        return self.send_command(
            SupervisorControlMessage(command=SupervisorCommand.RESTART_PROCESS, process_id=process_id),
        )

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

    # endregion

    def request_graceful_stop(self, *, timeout: float = GRACEFUL_STOP_TIMEOUT_SECONDS) -> None:
        """Begin a non-blocking graceful shutdown; successive :meth:`tick` calls complete it.

        Unlike :meth:`stop`, this returns immediately instead of joining the worker, so a single-threaded
        owner (the worker host) keeps draining snapshots and serving clients while the worker finishes its
        in-flight jobs. The worker is asked to shut down; if it has not exited within ``timeout`` a later
        tick terminates it.

        Concurrency:
            Intended to be driven by the same thread that calls :meth:`tick`; it is not safe to call
            concurrently with :meth:`tick`, :meth:`start`, or :meth:`stop`.
        """
        self._intentional_stop = True
        process = self._process
        if process is None or not process.is_alive():
            self._complete_graceful_stop()
            return
        self.send_command(SupervisorControlMessage(command=SupervisorCommand.SHUTDOWN))
        self._graceful_stop_deadline = time.time() + timeout

    def restart(self) -> None:
        """Stop the worker and start it again (blocking; for owners that drive lifecycle synchronously).

        Presented as a single ``RESTARTING`` phase from the instant it begins: the stale snapshot is
        dropped and the status is pinned to ``RESTARTING`` across the stop and relaunch (the stop is
        told not to fall back to ``STOPPED``), so the dashboard shows a calm "relaunching" rather than
        ageing the dead worker's last frame into the alarming ``UNRESPONSIVE``. The next fresh snapshot
        flips the status to ``RUNNING`` (see :meth:`drain_snapshots`).
        """
        self.latest_snapshot = None
        self._set_status(SupervisorStatus.RESTARTING)
        self.stop(set_stopped_status=False)
        # Mirror start()'s lifecycle reset, but keep the RESTARTING phase instead of STARTING.
        self._intentional_stop = False
        self._graceful_stop_deadline = 0.0
        self._restart_attempts = 0
        self._spawn(status=SupervisorStatus.RESTARTING)

    def stop(self, *, timeout: float = GRACEFUL_STOP_TIMEOUT_SECONDS, set_stopped_status: bool = True) -> None:
        """Gracefully shut the worker down (blocking), terminating it if it overruns the timeout.

        ``set_stopped_status=False`` lets :meth:`restart` keep the ``RESTARTING`` phase across the stop
        instead of briefly flashing ``STOPPED`` between the teardown and the relaunch.
        """
        self._intentional_stop = True
        process = self._process
        if process is not None and process.is_alive():
            self.send_command(SupervisorControlMessage(command=SupervisorCommand.SHUTDOWN))
            process.join(timeout)
            if process.is_alive():
                logger.warning("Worker did not exit after shutdown request; terminating.")
                process.terminate()
                process.join(5.0)
        self._cleanup_process()
        self._graceful_stop_deadline = 0.0
        if set_stopped_status:
            self._set_status(SupervisorStatus.STOPPED)

    def close(self) -> None:
        """Release the worker when the frontend exits.

        This supervisor owns the worker child directly, so releasing it means stopping it (leaving it
        orphaned would be worse). The attach client overrides this meaning: there, the worker lives on a
        separate host, so closing only detaches the session and leaves the worker running.
        """
        self.stop()

    def _cleanup_process(self) -> None:
        """Close the connection and drop the process handle (and the now-dead worker's last snapshot)."""
        if self._connection is not None:
            with contextlib.suppress(Exception):
                self._connection.close()
            self._connection = None
        self._process = None
        # The worker that produced it is gone; keeping it would age into a false UNRESPONSIVE.
        self.latest_snapshot = None
