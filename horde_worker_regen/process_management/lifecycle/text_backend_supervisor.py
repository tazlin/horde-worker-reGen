"""Launch and keep alive the external text-inference program the text-generation flow talks to.

The text workload's inference engine (koboldcpp today) is a program the worker does not control. It has no
pipe, no IPC vocabulary and no `HordeProcessInfo`; the worker can start it, ask it a few HTTP questions
through a [`TextBackend`][horde_worker_regen.text_backends.protocol.TextBackend] driver, and stop it. This
module owns exactly that, and nothing more: the seam with a black box is kept to start, ready, stop, so
there are as few places as possible where the two programs can disagree.

- :class:`TextBackendLaunchSpec`: the executable, its arguments, the port and the log file.
- :func:`koboldcpp_launch_spec`: derives the spec for koboldcpp from the worker's settings.
- :class:`TextBackendSupervisor`: runs the launch, readiness gate, liveness watch and relaunch ladder on
  one asyncio task, measures the backend's VRAM footprint as the card's free-memory delta, and keeps every
  process id in the owned-PID registry so a crashed worker's next run can reap it.

Two facts about koboldcpp shape the process handling. It is a PyInstaller one-file binary, so the process
the worker launches is a bootloader that unpacks itself and runs the real server as a child; terminating
only the bootloader leaves the server alive, listening on the port and holding VRAM. The supervisor
therefore resolves the whole process tree once the backend is ready, records every PID, and stops the
tree descendants first. And a server that survived a previous run still holds the port, so a relaunch on
that port fails until it is gone; the supervisor checks the port is free before launching and treats a
held port as a launch failure that backs off rather than a reason to guess a different port.

The relaunch ladder is deliberately one rung: the process exited, or it never answered `ready()` within
the patience window, so kill the tree and launch again with backoff. Finer detection (an alive process
that stopped answering mid-generation) is left to the flow, whose `generate()` raises
`TextBackendUnavailable` and re-runs its own readiness gate; it is added here only once the backend proves
a stable interface for it.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import time
from collections.abc import Awaitable, Callable
from enum import auto
from pathlib import Path
from typing import Protocol

import psutil
from loguru import logger
from pydantic import BaseModel, ConfigDict
from strenum import StrEnum

from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.owned_process_registry import OwnedProcessRegistry
from horde_worker_regen.text_backends import TextBackend

LOOPBACK_HOST = "127.0.0.1"
"""The backend only ever listens on loopback; the worker is its sole client."""

TEXT_BACKEND_LOG_FILE_NAME = "text_backend.log"
"""File under the run's ``logs/`` that receives the backend's own stdout and stderr."""


class LaunchedProcess(Protocol):
    """The OS-process control surface the supervisor needs; `subprocess.Popen` satisfies it."""

    @property
    def pid(self) -> int:
        """Return the OS process id."""
        ...

    def poll(self) -> int | None:
        """Return the exit code if the process has exited, else None."""
        ...

    def terminate(self) -> None:
        """Ask the process to exit."""
        ...

    def kill(self) -> None:
        """Force the process to exit."""
        ...

    def wait(self, timeout: float | None = None) -> int:
        """Block until the process exits and return its exit code.

        Raises:
            subprocess.TimeoutExpired: The process was still running when the timeout elapsed.
        """
        ...


class TextBackendLaunchSpec(BaseModel):
    """Represents everything needed to start one text backend process."""

    model_config = ConfigDict(frozen=True)

    executable: Path
    """The backend binary."""
    arguments: tuple[str, ...]
    """Arguments after the executable, already rendered as strings."""
    port: int
    """The loopback port the backend will listen on; also what `base_url` is built from."""
    log_path: Path
    """Where the backend's own stdout and stderr go."""
    device_index: int | None = None
    """The stable device index the backend was told to use, for footprint measurement; None off-GPU."""

    @property
    def command(self) -> list[str]:
        """Return the full argv."""
        return [str(self.executable), *self.arguments]

    @property
    def base_url(self) -> str:
        """Return the HTTP base URL the driver targets."""
        return f"http://{LOOPBACK_HOST}:{self.port}"


class KoboldcppArguments:
    """koboldcpp's command-line flags, as spelled by upstream's argument parser."""

    MODEL = "--model"
    PORT = "--port"
    HOST = "--host"
    USE_CUDA = "--usecuda"
    GPU_LAYERS = "--gpulayers"
    CONTEXT_SIZE = "--contextsize"
    SKIP_LAUNCHER = "--skiplauncher"
    QUIET = "--quiet"


def koboldcpp_launch_spec(
    *,
    executable: Path,
    model_path: Path,
    port: int,
    device_index: int | None,
    gpu_layers: int,
    context_length: int,
    log_path: Path,
) -> TextBackendLaunchSpec:
    """Create the launch spec for koboldcpp serving one GGUF on loopback.

    The embedded horde worker inside koboldcpp stays off (no API key is passed): the worker pops. The
    launcher GUI is skipped and output is quietened because the process runs unattended under the
    supervisor. `device_index` becomes the CUDA ordinal; the parent sets ``CUDA_DEVICE_ORDER=PCI_BUS_ID``
    before any child starts, so the worker's stable index and koboldcpp's ordinal agree. None means the
    backend runs without CUDA (a no-CUDA build, or a CPU-only host).

    Args:
        executable: The koboldcpp binary.
        model_path: The GGUF file to load.
        port: Loopback port to listen on.
        device_index: Stable device index for the CUDA backend, or None for no CUDA flag.
        gpu_layers: How many layers to offload; koboldcpp caps a large number at the model's layer count.
        context_length: The context size koboldcpp allocates its KV cache for.
        log_path: Where the backend's output goes.
    """
    arguments: list[str] = [
        KoboldcppArguments.MODEL,
        str(model_path),
        KoboldcppArguments.PORT,
        str(port),
        KoboldcppArguments.HOST,
        LOOPBACK_HOST,
    ]
    if device_index is not None:
        arguments.extend([KoboldcppArguments.USE_CUDA, str(device_index)])
    arguments.extend(
        [
            KoboldcppArguments.GPU_LAYERS,
            str(gpu_layers),
            KoboldcppArguments.CONTEXT_SIZE,
            str(context_length),
            KoboldcppArguments.SKIP_LAUNCHER,
            KoboldcppArguments.QUIET,
        ],
    )
    return TextBackendLaunchSpec(
        executable=executable,
        arguments=tuple(arguments),
        port=port,
        log_path=log_path,
        device_index=device_index,
    )


def default_launch_process(spec: TextBackendLaunchSpec) -> LaunchedProcess:
    """Start the backend with its output appended to the spec's log file."""
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    # The handle is handed to the child and released when the child exits; the supervisor never reads it.
    log_file = spec.log_path.open("ab")  # noqa: SIM115 - the child process owns this handle's lifetime
    try:
        return subprocess.Popen(spec.command, stdout=log_file, stderr=subprocess.STDOUT)
    finally:
        log_file.close()


def port_is_free(port: int, *, host: str = LOOPBACK_HOST) -> bool:
    """Return whether nothing is listening on ``host:port``.

    A bind probe rather than a connect probe: a process that has bound the port but is not yet accepting
    still owns it, and that is exactly the stale-server case this guards against.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


class TextBackendSupervisorTimings(BaseModel):
    """Represents the supervisor's waits, all in seconds."""

    model_config = ConfigDict(frozen=True)

    ready_poll_interval: float = 2.0
    """Pause between `ready()` probes while the backend starts."""
    ready_probe_deadline: float = 5.0
    """Deadline handed to each `ready()` probe."""
    ready_patience: float = 300.0
    """How long a launch may take to answer `ready()` before it is killed and relaunched.

    A PyInstaller one-file binary unpacks itself on every start and then loads the model; a 3B model on a
    warm disk took about a minute, and a large model on a cold disk takes several."""
    liveness_poll_interval: float = 1.0
    """Pause between exit-code polls while serving."""
    stop_grace: float = 15.0
    """How long a terminated process may take to exit before it is killed."""
    relaunch_backoff: tuple[float, ...] = (5.0, 15.0, 60.0, 300.0)
    """Pause before each consecutive relaunch attempt; the last value repeats."""


class TextBackendState(StrEnum):
    """Where the supervised backend is in its life."""

    STOPPED = auto()
    LAUNCHING = auto()
    SERVING = auto()
    BACKING_OFF = auto()


DeviceFreeTotalReader = Callable[[int], tuple[float, float] | None]
"""Reads a device's ``(free_mb, total_mb)``; None when the reading is unavailable."""

LaunchProcess = Callable[[TextBackendLaunchSpec], LaunchedProcess]
"""Starts the backend process described by a spec."""

Sleep = Callable[[float], Awaitable[None]]


class TextBackendSupervisor:
    """Keep one text backend process launched, ready and accounted for until asked to stop.

    The driver (`backend`) targets the spec's loopback URL, which does not change across relaunches, so
    the text-generation flow holds one driver object for the supervisor's whole life and never learns
    about relaunches except through `ready()` returning False for a while.

    Concurrency:
        `run()` is one asyncio task. `stop()` may be awaited from another task at any time; it flips the
        stop flag, tears the process tree down, and `run()` returns at its next await.
    """

    def __init__(
        self,
        *,
        launch_spec: TextBackendLaunchSpec,
        backend: TextBackend,
        owned_registry: OwnedProcessRegistry | None = None,
        launch_process: LaunchProcess = default_launch_process,
        read_device_free_total_mb: DeviceFreeTotalReader | None = None,
        timings: TextBackendSupervisorTimings | None = None,
        first_launch_identifier: int = 0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        """Wire the supervisor; nothing starts until `run()`.

        Args:
            launch_spec: What to launch and where it listens.
            backend: The driver already pointed at the spec's URL; used only for `ready()` here.
            owned_registry: Where launched PIDs are recorded for orphan reaping; None records nothing.
            launch_process: How a process is started; injected so tests never spawn anything.
            read_device_free_total_mb: Card memory reader for the footprint delta; None disables it.
            timings: Waits and patience; defaults suit koboldcpp.
            first_launch_identifier: The registry's launch identifier for the first launch; increments per
                launch so a reaped PID can be attributed to its launch.
            sleep: Awaitable sleep, injected so tests can run the ladder without wall-clock waits.
        """
        self._launch_spec = launch_spec
        self._backend = backend
        self._owned_registry = owned_registry
        self._launch_process = launch_process
        self._read_device_free_total_mb = read_device_free_total_mb
        self._timings = timings or TextBackendSupervisorTimings()
        self._next_launch_identifier = first_launch_identifier
        self._sleep = sleep

        self._process: LaunchedProcess | None = None
        self._tree_pids: tuple[int, ...] = ()
        self._state = TextBackendState.STOPPED
        self._stop_requested = False
        self._footprint_mb: float | None = None
        self._launch_count = 0
        self._consecutive_failures = 0

    @property
    def backend(self) -> TextBackend:
        """Return the driver the flow talks through."""
        return self._backend

    @property
    def state(self) -> TextBackendState:
        """Return where the backend is in its life."""
        return self._state

    @property
    def is_serving(self) -> bool:
        """Return whether the backend has answered `ready()` and its process is still alive."""
        return self._state is TextBackendState.SERVING

    @property
    def footprint_mb(self) -> float | None:
        """Return the card's free-VRAM drop between launch and ready, or None when unmeasured.

        Measured once per launch, so a backend that grows after ready is not tracked here; that is the
        arbiter's foreign-floor reading.
        """
        return self._footprint_mb

    @property
    def launch_count(self) -> int:
        """Return how many times a process has been started."""
        return self._launch_count

    @property
    def pids(self) -> tuple[int, ...]:
        """Return the PIDs of the current process tree, bootloader first; empty when stopped."""
        return self._tree_pids

    async def run(self) -> None:
        """Run the launch, gate, watch and relaunch loop until `stop()` is awaited."""
        while not self._stop_requested:
            became_ready = await self._launch_and_gate()
            if self._stop_requested:
                break
            if became_ready:
                self._consecutive_failures = 0
                await self._watch_until_exit()
                if self._stop_requested:
                    break
            else:
                self._consecutive_failures += 1
            await self._back_off()
        self._state = TextBackendState.STOPPED

    async def stop(self) -> None:
        """Stop the backend's whole process tree and end `run()`."""
        self._stop_requested = True
        await self._stop_tree()
        self._state = TextBackendState.STOPPED

    async def _launch_and_gate(self) -> bool:
        """Launch the process and wait for `ready()`; return whether it became ready.

        A launch that fails leaves no process behind: the tree is stopped before returning False.
        """
        self._state = TextBackendState.LAUNCHING
        if not port_is_free(self._launch_spec.port):
            logger.warning(
                f"Text backend port {self._launch_spec.port} is already in use; a previous backend may still "
                "be running. Launch deferred.",
            )
            return False

        free_before_mb = self._read_free_mb()
        try:
            self._process = self._launch_process(self._launch_spec)
        except OSError as error:
            logger.error(f"Text backend failed to start ({self._launch_spec.executable}): {error}")
            self._process = None
            return False
        self._launch_count += 1
        launch_identifier = self._next_launch_identifier
        self._next_launch_identifier += 1
        self._record_tree(launch_identifier)
        logger.info(
            f"Text backend launched (pid {self._process.pid}, launch {self._launch_count}): "
            f"{' '.join(self._launch_spec.command)}",
        )

        started = time.monotonic()
        while not self._stop_requested:
            exit_code = self._process.poll()
            if exit_code is not None:
                logger.error(f"Text backend exited with code {exit_code} before it became ready")
                await self._stop_tree()
                return False
            if await self._backend.ready(deadline_seconds=self._timings.ready_probe_deadline):
                self._record_tree(launch_identifier)
                self._footprint_mb = self._footprint_from(free_before_mb)
                self._state = TextBackendState.SERVING
                footprint = "unmeasured" if self._footprint_mb is None else f"{self._footprint_mb:.0f} MB"
                logger.info(
                    f"Text backend ready after {time.monotonic() - started:.1f}s; VRAM footprint {footprint}",
                )
                return True
            if time.monotonic() - started > self._timings.ready_patience:
                logger.error(
                    f"Text backend did not become ready within {self._timings.ready_patience:.0f}s; stopping it",
                )
                await self._stop_tree()
                return False
            await self._sleep(self._timings.ready_poll_interval)
        return False

    async def _watch_until_exit(self) -> None:
        """Poll the process until it exits or a stop is requested."""
        while not self._stop_requested:
            if self._process is None:
                return
            exit_code = self._process.poll()
            if exit_code is not None:
                logger.warning(f"Text backend exited with code {exit_code} while serving; it will be relaunched")
                self._forget_tree()
                self._process = None
                self._state = TextBackendState.STOPPED
                return
            await self._sleep(self._timings.liveness_poll_interval)

    async def _back_off(self) -> None:
        """Pause before the next launch, longer after each consecutive failure."""
        if self._stop_requested:
            return
        backoff = self._timings.relaunch_backoff
        index = min(self._consecutive_failures, len(backoff) - 1)
        self._state = TextBackendState.BACKING_OFF
        logger.info(f"Text backend relaunch in {backoff[index]:.0f}s")
        await self._sleep(backoff[index])

    def _read_free_mb(self) -> float | None:
        if self._read_device_free_total_mb is None or self._launch_spec.device_index is None:
            return None
        reading = self._read_device_free_total_mb(self._launch_spec.device_index)
        return None if reading is None else reading[0]

    def _footprint_from(self, free_before_mb: float | None) -> float | None:
        free_after_mb = self._read_free_mb()
        if free_before_mb is None or free_after_mb is None:
            return None
        return max(0.0, free_before_mb - free_after_mb)

    def _record_tree(self, launch_identifier: int) -> None:
        """Resolve the process tree and record every PID not yet recorded.

        Called at launch (the bootloader alone) and again at ready (by then the bootloader has spawned the
        real server), so the registry holds the PID that actually owns the port and the VRAM.
        """
        if self._process is None:
            return
        resolved = _process_tree_pids(self._process.pid)
        newly_seen = tuple(pid for pid in resolved if pid not in self._tree_pids)
        self._tree_pids = tuple(pid for pid in self._tree_pids if pid in resolved) + newly_seen
        if self._owned_registry is None:
            return
        for pid in newly_seen:
            self._owned_registry.record(
                os_pid=pid,
                launch_identifier=launch_identifier,
                process_type=HordeProcessType.TEXT_BACKEND.name,
            )

    def _forget_tree(self) -> None:
        if self._owned_registry is not None:
            for pid in self._tree_pids:
                self._owned_registry.forget(pid)
        self._tree_pids = ()

    async def _stop_tree(self) -> None:
        """Terminate the descendants first, then the launched process; kill whatever outlives the grace."""
        process = self._process
        if process is None:
            self._forget_tree()
            return
        descendants = _living_descendants(process.pid)
        for descendant in descendants:
            _terminate_quietly(descendant)
        _terminate_handle_quietly(process)
        try:
            await asyncio.to_thread(process.wait, self._timings.stop_grace)
        except subprocess.TimeoutExpired:
            logger.warning("Text backend ignored terminate; killing it")
            _kill_handle_quietly(process)
            await asyncio.to_thread(process.wait, self._timings.stop_grace)
        _, still_alive = psutil.wait_procs(descendants, timeout=self._timings.stop_grace)
        for survivor in still_alive:
            logger.warning(f"Text backend child pid {survivor.pid} ignored terminate; killing it")
            _kill_quietly(survivor)
        self._forget_tree()
        self._process = None
        self._state = TextBackendState.STOPPED


def _process_tree_pids(root_pid: int) -> tuple[int, ...]:
    """Return the root PID followed by its living descendants; just the root when it cannot be inspected."""
    return (root_pid, *(descendant.pid for descendant in _living_descendants(root_pid)))


def _living_descendants(root_pid: int) -> list[psutil.Process]:
    try:
        return psutil.Process(root_pid).children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def _terminate_quietly(process: psutil.Process) -> None:
    try:
        process.terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
        logger.debug(f"Text backend child pid {process.pid} could not be terminated: {error}")


def _kill_quietly(process: psutil.Process) -> None:
    try:
        process.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
        logger.debug(f"Text backend child pid {process.pid} could not be killed: {error}")


def _terminate_handle_quietly(process: LaunchedProcess) -> None:
    try:
        process.terminate()
    except OSError as error:
        logger.debug(f"Text backend pid {process.pid} could not be terminated: {error}")


def _kill_handle_quietly(process: LaunchedProcess) -> None:
    try:
        process.kill()
    except OSError as error:
        logger.debug(f"Text backend pid {process.pid} could not be killed: {error}")


__all__ = [
    "LOOPBACK_HOST",
    "TEXT_BACKEND_LOG_FILE_NAME",
    "DeviceFreeTotalReader",
    "KoboldcppArguments",
    "LaunchProcess",
    "LaunchedProcess",
    "TextBackendLaunchSpec",
    "TextBackendState",
    "TextBackendSupervisor",
    "TextBackendSupervisorTimings",
    "default_launch_process",
    "koboldcpp_launch_spec",
    "port_is_free",
]
