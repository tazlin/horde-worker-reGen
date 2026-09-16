"""Launch and keep alive the external text-inference program the text-generation flow talks to.

A text backend is a separate program the worker does not control (koboldcpp today; sonar and other
KoboldAI-API servers planned). It has no pipe, no IPC vocabulary and no `HordeProcessInfo`; the worker can
start it, ask it a few HTTP questions through a
[`TextBackend`][horde_worker_regen.text_backends.protocol.TextBackend] driver, and stop it. This module owns
exactly that, and nothing more: the seam with a black box is kept to start, ready, stop, so there are as
few places as possible where the two programs can disagree. Which program, and with which command line, is
decided elsewhere ([`text_backends.launch`][horde_worker_regen.text_backends.launch]); the supervisor is
handed a factory that obtains the executable and renders a
[`TextBackendLaunchSpec`][horde_worker_regen.text_backends.launch_spec.TextBackendLaunchSpec],
and treats every backend the same way.

- :class:`TextBackendSupervisor`: runs the obtaining step, launch, readiness gate, liveness watch and
  relaunch ladder on one asyncio task, measures the backend's VRAM footprint as the card's free-memory
  delta, and keeps every process id in the owned-PID registry so a crashed worker's next run can reap it.
  The whole of the backend's life is here, obtaining included, so it has one row on the dashboard from the
  moment the worker decides to have a backend at all.
- :func:`default_launch_process`: starts the spec's command with output appended to its log file.
- :func:`port_is_free`: the pre-launch check that nothing still holds the backend's port.
- :func:`parse_backend_buffer_sizes` and :class:`TextBackendBufferSizes`: the load-time buffer figures a
  llama.cpp-based backend prints once, read out of the log it writes.

The supervisor is also the backend's row on the dashboard. `to_process_snapshot()` satisfies
[`SupervisedProcessSnapshotSource`][horde_worker_regen.process_management.ipc.supervisor_channel.SupervisedProcessSnapshotSource],
so the snapshot builder can put the backend in the process table, the Live tab and the native page beside
the worker's own children without it entering `ProcessMap`, whose entries are pipe-bearing children with
torch allocator readings and an image state vocabulary a text row has none of.

Two process facts are handled generically because a backend may exhibit either. A launched program may run
the real server as a child of the launched process (koboldcpp does: it is a PyInstaller one-file binary
whose bootloader unpacks itself and spawns the server), so terminating only the launched PID can leave the
server alive, listening on the port and holding VRAM. The supervisor therefore resolves the whole process
tree once the backend is ready, records every PID, and stops the tree descendants first. And a server that
survived a previous run still holds its port, so a relaunch on that port fails until it is gone; the
supervisor checks the port is free before launching and treats a held port as a launch failure that backs
off rather than a reason to guess a different port.

The relaunch ladder is deliberately one rung: the process exited, or it never answered `ready()` within
the patience window, so kill the tree and launch again with backoff. Finer detection (an alive process
that stopped answering mid-generation) is left to the flow, whose `generate()` raises
`TextBackendUnavailable` and re-runs its own readiness gate; it is added here only once a backend proves a
stable interface for it.
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
from horde_model_reference.text_backend_names import TEXT_BACKENDS
from loguru import logger
from pydantic import BaseModel, ConfigDict
from strenum import StrEnum

from horde_worker_regen.analysis.log_signatures import pattern_for
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    TEXT_BACKEND_PROCESS_ID,
    ProcessSnapshot,
    TextBackendActivity,
    TextBackendDetail,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.owned_process_registry import OwnedProcessRegistry
from horde_worker_regen.text_backends import (
    TextBackend,
    TextBackendDescription,
    TextBackendError,
    UnsupportedTextBackendError,
)
from horde_worker_regen.text_backends.launch_spec import LOOPBACK_HOST, TextBackendLaunchSpec
from horde_worker_regen.text_backends.provision import TextBackendProvisionError

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

    Start-up is dominated by model loading, and some backends also unpack themselves first (a PyInstaller
    one-file binary does). A 3B model on a warm disk took about a minute; a large model on a cold disk
    takes several."""
    liveness_poll_interval: float = 1.0
    """Pause between exit-code polls while serving."""
    stop_grace: float = 15.0
    """How long a terminated process may take to exit before it is killed."""
    relaunch_backoff: tuple[float, ...] = (5.0, 15.0, 60.0, 300.0)
    """Pause before each consecutive relaunch attempt; the last value repeats."""


class TextBackendState(StrEnum):
    """Where the supervised backend is in its life.

    Deliberately its own vocabulary rather than a borrowed ``HordeProcessState``: the supervised backend is
    not in ``ProcessMap``, and a name disjoint from the image-and-alchemy states is what keeps every reader
    that tests a state against one of those sets from matching a text row.
    """

    STOPPED = auto()
    PROVISIONING = auto()
    LAUNCHING = auto()
    SERVING = auto()
    BACKING_OFF = auto()


class TextBackendBufferSizes(BaseModel):
    """Represents the load-time buffer sizes a llama.cpp-based backend printed, summed over its devices.

    Best-effort display detail: the backend exposes no runtime memory query, so these are read once per
    launch out of the log it writes, and a backend that prints nothing (a quiet debug level, or a program
    that is not llama.cpp-based) leaves every field None. Nothing here is used for admission or pricing;
    the launch-time free-VRAM delta stays the footprint the worker reasons with.
    """

    model_config = ConfigDict(frozen=True)

    model_mebibytes: int | None = None
    """Weights the loader placed in buffers."""
    kv_mebibytes: int | None = None
    """The KV cache the loader allocated for the configured context."""
    compute_mebibytes: int | None = None
    """The compute buffers the loader reserved for its worst-case graph."""


def parse_backend_buffer_sizes(backend_output: str) -> TextBackendBufferSizes:
    """Parse the model, KV and compute buffer sizes out of one launch's backend output.

    A llama.cpp-based backend prints one line per device per buffer kind, so each kind is summed across the
    lines that named it. A kind no line mentioned stays None rather than becoming a zero, which would read
    as "the backend allocated nothing".

    Args:
        backend_output: The text the backend wrote since its process was launched.

    Returns:
        The summed sizes, in the mebibytes the backend prints.
    """
    return TextBackendBufferSizes(
        model_mebibytes=_sum_buffer_lines(backend_output, "text_backend_model_buffer"),
        kv_mebibytes=_sum_buffer_lines(backend_output, "text_backend_kv_buffer"),
        compute_mebibytes=_sum_buffer_lines(backend_output, "text_backend_compute_buffer"),
    )


def _sum_buffer_lines(backend_output: str, signature_name: str) -> int | None:
    """Return the rounded sum of one registered buffer pattern's matches, or None when it matched nothing."""
    matches = pattern_for(signature_name).finditer(backend_output)
    mebibytes = [float(match.group("mebibytes")) for match in matches]
    return round(sum(mebibytes)) if mebibytes else None


DeviceFreeTotalReader = Callable[[int], tuple[float, float] | None]
"""Reads a device's ``(free_mb, total_mb)``; None when the reading is unavailable."""

LaunchProcess = Callable[[TextBackendLaunchSpec], LaunchedProcess]
"""Starts the backend process described by a spec."""

LaunchSpecFactory = Callable[[], TextBackendLaunchSpec]
"""Obtains the backend's executable and renders its command line.

Blocking, and on a first run for hundreds of megabytes: a backend distributed as a release download is
fetched and verified here. The supervisor calls it once, in a thread, before its first launch.

Raises:
    TextBackendProvisionError: The executable could not be obtained.
    UnsupportedTextBackendError: The worker cannot render a command line for this backend.
"""

Sleep = Callable[[float], Awaitable[None]]


class TextBackendSupervisor:
    """Keep one text backend obtained, launched, ready and accounted for until asked to stop.

    The driver (`backend`) targets the spec's loopback URL, which does not change across relaunches, so
    the text-generation flow holds one driver object for the supervisor's whole life and never learns
    about relaunches except through `ready()` returning False for a while.

    The backend's life begins at obtaining it, not at launching it, so the supervisor takes a factory
    rather than a rendered spec: the row exists from construction and a first-run download is something
    the dashboard can show, rather than a minute in which the worker appears to have no backend at all.

    Concurrency:
        `run()` is one asyncio task. `stop()` may be awaited from another task at any time; it flips the
        stop flag, tears the process tree down, and `run()` returns at its next await.
    """

    def __init__(
        self,
        *,
        launch_spec_factory: LaunchSpecFactory,
        backend: TextBackend,
        backend_kind: TEXT_BACKENDS = TEXT_BACKENDS.koboldcpp,
        owned_registry: OwnedProcessRegistry | None = None,
        launch_process: LaunchProcess = default_launch_process,
        read_device_free_total_mb: DeviceFreeTotalReader | None = None,
        timings: TextBackendSupervisorTimings | None = None,
        first_launch_identifier: int = 0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        """Wire the supervisor; nothing starts until `run()`.

        Args:
            launch_spec_factory: Obtains the executable and renders what to launch and where it listens.
                Called once, in a thread, at the start of `run()`; a failure is reported here and leaves
                the backend stopped rather than reaching the caller.
            backend: The driver already pointed at the backend's URL; used for `ready()` and, once ready,
                one `describe()` whose answer names the backend's row on the dashboard. Its address is
                decided by configuration rather than by the spec, so it can exist before the spec does.
            backend_kind: Which program the spec launches. The spec is deliberately kind-agnostic (it is
                one command line), so the kind is carried alongside it, for display only.
            owned_registry: Where launched PIDs are recorded for orphan reaping; None records nothing.
            launch_process: How a process is started; injected so tests never spawn anything.
            read_device_free_total_mb: Card memory reader for the footprint delta; None disables it.
            timings: Waits and patience.
            first_launch_identifier: The registry's launch identifier for the first launch; increments per
                launch so a reaped PID can be attributed to its launch.
            sleep: Awaitable sleep, injected so tests can run the ladder without wall-clock waits.
        """
        self._launch_spec_factory = launch_spec_factory
        self._launch_spec: TextBackendLaunchSpec | None = None
        self._backend = backend
        self._backend_kind = backend_kind
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

        # Everything below describes the current launch for the dashboard's row and is cleared when the
        # process goes away, so a stale model name or buffer figure can never outlive the process it came
        # from. None of it is read by the ladder.
        self._description: TextBackendDescription | None = None
        self._buffer_sizes = TextBackendBufferSizes()
        self._log_bytes_at_launch = 0
        self._ready_since: float | None = None
        self._last_health_ok_at: float | None = None
        # Outlives one launch deliberately, unlike the fields above: it is when the worker last *tried*,
        # which is what a reader measuring how long the backend has failed to answer needs, and a launch
        # that never started a process (a held port) has to count as an attempt too.
        self._launching_since: float | None = None
        self._provision_error: str | None = None
        """Why the backend could not be obtained, or None while that has not happened.

        The reason rather than a state: a backend that cannot be obtained is stopped, and without the
        reason on the row the dashboard would show a worker permanently "obtaining" a program that is
        never coming."""
        self._backoff_seconds: float | None = None
        self._backoff_started_monotonic: float | None = None

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
        """Return the PIDs of the current process tree, launched process first; empty when stopped."""
        return self._tree_pids

    async def run(self) -> None:
        """Run the obtain, launch, gate, watch and relaunch loop until `stop()` is awaited.

        Obtaining happens once: a backend the worker cannot obtain at all is not a backend a relaunch
        ladder can recover, so the reason is reported and the supervisor stops rather than downloading in
        a loop.
        """
        while not self._stop_requested:
            spec = self._launch_spec
            if spec is None:
                spec = await self._obtain_launch_spec()
            if spec is None or self._stop_requested:
                break
            became_ready = await self._launch_and_gate(spec)
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

    async def _obtain_launch_spec(self) -> TextBackendLaunchSpec | None:
        """Obtain the backend and render its command line, or report why that could not be done.

        Run in a thread because a first provision downloads a release. The row reads `provisioning` for as
        long as this takes, which on a first run is the difference between a dashboard that shows a
        download and one that shows nothing.
        """
        self._state = TextBackendState.PROVISIONING
        try:
            self._launch_spec = await asyncio.to_thread(self._launch_spec_factory)
        except TextBackendProvisionError as error:
            self._provision_error = f"Could not obtain the {self._backend_kind!s} text backend: {error}"
            logger.error(self._provision_error)
        except UnsupportedTextBackendError as error:
            self._provision_error = f"Cannot launch the {self._backend_kind!s} text backend: {error}"
            logger.error(self._provision_error)
        return self._launch_spec

    async def _launch_and_gate(self, spec: TextBackendLaunchSpec) -> bool:
        """Launch the process and wait for `ready()`; return whether it became ready.

        A launch that fails leaves no process behind: the tree is stopped before returning False.
        """
        self._state = TextBackendState.LAUNCHING
        self._clear_launch_detail()
        self._launching_since = time.time()
        if not port_is_free(spec.port):
            logger.warning(
                f"Text backend port {spec.port} is already in use; a previous backend may still "
                "be running. Launch deferred.",
            )
            return False

        free_before_mb = self._read_free_mb(spec)
        # The log is appended across launches, so the size now is where this launch's own output begins.
        self._log_bytes_at_launch = _log_size_bytes(spec.log_path)
        try:
            self._process = self._launch_process(spec)
        except OSError as error:
            logger.error(f"Text backend failed to start ({spec.executable}): {error}")
            self._process = None
            return False
        self._launch_count += 1
        launch_identifier = self._next_launch_identifier
        self._next_launch_identifier += 1
        self._record_tree(launch_identifier)
        logger.info(
            f"Text backend launched (pid {self._process.pid}, launch {self._launch_count}): {' '.join(spec.command)}",
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
                self._footprint_mb = self._footprint_from(spec, free_before_mb)
                self._ready_since = time.time()
                self._last_health_ok_at = self._ready_since
                self._state = TextBackendState.SERVING
                footprint = "unmeasured" if self._footprint_mb is None else f"{self._footprint_mb:.0f} MB"
                logger.info(
                    f"Text backend ready after {time.monotonic() - started:.1f}s; VRAM footprint {footprint}",
                )
                await self._collect_launch_detail(spec)
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
                self._clear_launch_detail()
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
        self._backoff_seconds = backoff[index]
        self._backoff_started_monotonic = time.monotonic()
        logger.info(f"Text backend relaunch in {backoff[index]:.0f}s")
        await self._sleep(backoff[index])

    async def _collect_launch_detail(self, spec: TextBackendLaunchSpec) -> None:
        """Read the display-only facts about the launch that has just become ready.

        Both halves are best effort and neither gates serving: the description is one HTTP question the
        driver already answers, and the buffer sizes come from the output the backend has written since it
        was launched. A backend that refuses either leaves those fields unset and keeps generating.
        """
        try:
            self._description = await self._backend.describe()
        except TextBackendError as error:
            logger.warning(f"Text backend became ready but would not describe itself: {error}")
            self._description = None
        backend_output = await asyncio.to_thread(
            _read_log_from,
            spec.log_path,
            self._log_bytes_at_launch,
        )
        self._buffer_sizes = parse_backend_buffer_sizes(backend_output)

    def _clear_launch_detail(self) -> None:
        """Drop the facts that belong to a launch that is over, so no row outlives its process."""
        self._description = None
        self._buffer_sizes = TextBackendBufferSizes()
        self._ready_since = None
        self._backoff_seconds = None
        self._backoff_started_monotonic = None

    def to_process_snapshot(self, activity: TextBackendActivity | None = None) -> ProcessSnapshot:
        """Return the backend's row for the dashboard's process surfaces.

        Satisfies
        [`SupervisedProcessSnapshotSource`][horde_worker_regen.process_management.ipc.supervisor_channel.SupervisedProcessSnapshotSource]:
        the row is marked external and carries a typed
        [`TextBackendDetail`][horde_worker_regen.process_management.ipc.supervisor_channel.TextBackendDetail]
        instead of the allocator readings and job progress a pipe-bearing child reports. Every figure the
        supervisor supplies is one it already holds, so the call neither blocks nor asks the backend
        anything.

        Args:
            activity: What the text flow currently has in flight against this backend, which the
                supervisor cannot see for itself. None when no flow is attached, which leaves the row's
                request counts at zero and the row idle rather than claiming work nothing accounts for.
        """
        description = self._description
        spec = self._launch_spec
        return ProcessSnapshot(
            process_id=TEXT_BACKEND_PROCESS_ID,
            process_type=HordeProcessType.TEXT_BACKEND.name,
            device_index=None if spec is None else spec.device_index,
            last_process_state=self._state.name,
            is_alive=self._state in (TextBackendState.LAUNCHING, TextBackendState.SERVING),
            is_busy=activity is not None and activity.active_requests > 0,
            is_external=True,
            os_pid=self._tree_pids[0] if self._tree_pids else None,
            display_state=self._display_state(),
            loaded_horde_model_name=description.model_name if description is not None else None,
            text_backend=TextBackendDetail(
                kind=str(self._backend_kind),
                model_name=description.model_name if description is not None else None,
                port=None if spec is None else spec.port,
                context_length=description.max_context_length if description is not None else None,
                max_length=description.max_length if description is not None else None,
                launch_count=self._launch_count,
                footprint_mb=None if self._footprint_mb is None else round(self._footprint_mb),
                launching_since=self._launching_since,
                provision_error=self._provision_error,
                ready_since=self._ready_since,
                last_health_ok_at=self._last_health_ok_at,
                relaunch_backoff_seconds=self._relaunch_backoff_remaining(),
                active_requests=activity.active_requests if activity is not None else 0,
                queued_requests=activity.queued_requests if activity is not None else 0,
                tokens_per_second=activity.tokens_per_second if activity is not None else None,
                model_buffer_mb=self._buffer_sizes.model_mebibytes,
                kv_buffer_mb=self._buffer_sizes.kv_mebibytes,
                compute_buffer_mb=self._buffer_sizes.compute_mebibytes,
            ),
        )

    def _display_state(self) -> str:
        """Return the operator-facing label for the backend's current state."""
        if self._state is TextBackendState.SERVING:
            return "ready"
        if self._state is TextBackendState.LAUNCHING:
            return "launching"
        if self._state is TextBackendState.PROVISIONING:
            return "provisioning"
        if self._provision_error is not None:
            return "not obtained"
        if self._state is TextBackendState.BACKING_OFF:
            remaining = self._relaunch_backoff_remaining()
            return "relaunching" if remaining is None else f"relaunching in {remaining:.0f} s"
        return "stopped"

    def _relaunch_backoff_remaining(self) -> float | None:
        """Return the seconds left of the relaunch pause, or None when the backend is not backing off."""
        if self._state is not TextBackendState.BACKING_OFF:
            return None
        if self._backoff_seconds is None or self._backoff_started_monotonic is None:
            return None
        return max(0.0, self._backoff_seconds - (time.monotonic() - self._backoff_started_monotonic))

    def _read_free_mb(self, spec: TextBackendLaunchSpec) -> float | None:
        if self._read_device_free_total_mb is None or spec.device_index is None:
            return None
        reading = self._read_device_free_total_mb(spec.device_index)
        return None if reading is None else reading[0]

    def _footprint_from(self, spec: TextBackendLaunchSpec, free_before_mb: float | None) -> float | None:
        free_after_mb = self._read_free_mb(spec)
        if free_before_mb is None or free_after_mb is None:
            return None
        return max(0.0, free_before_mb - free_after_mb)

    def _record_tree(self, launch_identifier: int) -> None:
        """Resolve the process tree and record every PID not yet recorded.

        Called at launch (the launched process alone) and again at ready (by then a bootloader-style
        program has spawned the real server), so the registry holds the PID that actually owns the port
        and the VRAM.
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
        self._clear_launch_detail()
        self._state = TextBackendState.STOPPED


def _log_size_bytes(log_path: Path) -> int:
    """Return the backend log's current size, or 0 when it does not exist yet or cannot be read."""
    try:
        return log_path.stat().st_size
    except OSError as error:
        logger.debug(f"Text backend log {log_path} could not be sized: {error}")
        return 0


def _read_log_from(log_path: Path, offset_bytes: int) -> str:
    """Return the backend log's text from ``offset_bytes`` to its end, or an empty string when unreadable.

    Undecodable bytes are replaced rather than raising: the file holds a third-party program's raw output,
    including whatever a progress spinner wrote, and every line this is read for is plain ASCII.
    """
    try:
        with log_path.open("rb") as log_file:
            log_file.seek(offset_bytes)
            return log_file.read().decode("utf-8", errors="replace")
    except OSError as error:
        logger.debug(f"Text backend log {log_path} could not be read: {error}")
        return ""


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
    "TEXT_BACKEND_LOG_FILE_NAME",
    "DeviceFreeTotalReader",
    "LaunchProcess",
    "LaunchedProcess",
    "TextBackendBufferSizes",
    "TextBackendState",
    "TextBackendSupervisor",
    "TextBackendSupervisorTimings",
    "default_launch_process",
    "parse_backend_buffer_sizes",
    "port_is_free",
]
