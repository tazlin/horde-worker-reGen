"""Contains the base class for all processes, and additional helper types used by processes."""

from __future__ import annotations

import abc
import enum
import os
import queue
import signal
import sys
import threading
import time
from abc import abstractmethod
from enum import auto

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock

import psutil
from loguru import logger

from horde_worker_regen.process_management._internal._aliased_types import ProcessQueue
from horde_worker_regen.process_management.fd_limits import descriptor_soft_limit, open_descriptor_count
from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeHeartbeatType,
    HordeProcessHeartbeatMessage,
    HordeProcessMemoryMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
    UnsupportedControlMessageError,
)


class HordeProcessType(enum.Enum):
    """The type of process. This distinguishes between inference, safety, and potentially other process types.

    Process *type* is a lifecycle concept (which entry point, restart policy); job routing is
    driven by :class:`WorkerCapability` instead.
    """

    INFERENCE = auto()
    SAFETY = auto()
    DOWNLOAD = auto()
    """A background model-downloading process; serves no jobs and lives outside the process map."""
    POST_PROCESS = auto()
    """A dedicated post-processing process that keeps the upscaler/face-fixer models resident and runs the
    post-processing phase of image jobs (and the graph-backed alchemy forms) off the inference processes."""
    COMPONENT = auto()
    """A dedicated component lane: the stable producer of shared VAE/text-encoder weights. It serves no jobs;
    it holds a hot-set of canonical components resident and publishes them so inference processes adopt one
    copy instead of each loading their own. See ``workers/component_lane_process.py``."""
    VAE_LANE = auto()
    """A dedicated VAE lane: the disaggregated pipeline's VAE-encode/decode stage. Operationally a
    post-processing-style lane (holds a hordelib backend, serves jobs with results-known-lost semantics,
    pauses/restores off-GPU during whole-card windows, reports ``POST_PROCESSING`` state), but distinct from
    ``POST_PROCESS`` so a busy post-processing lane can never block a critical-path VAE stage and its
    co-residency charge (the tiled-decode spike) stays honest. See ``workers/vae_lane_process.py``."""


ALLOCATOR_CACHE_CAPABLE_PROCESS_TYPES: frozenset[HordeProcessType] = frozenset(
    {
        HordeProcessType.INFERENCE,
        HordeProcessType.SAFETY,
        HordeProcessType.POST_PROCESS,
        HordeProcessType.COMPONENT,
        HordeProcessType.VAE_LANE,
    },
)
"""Process types whose control dispatch implements ``RELEASE_ALLOCATOR_CACHE``.

A sender that fans the flag out across the process map (rather than targeting one known lane) must filter
on this set: a control flag delivered outside its receiver's dispatch contract is a routing error the
receiver drops instead of acting on, so the send accomplishes nothing and pollutes the child's log."""


class WorkerCapability(enum.Flag):
    """What kinds of work a process can be dispatched.

    Dispatch decisions key on capabilities rather than process types, so new job kinds
    (alchemy today; audio/video later) add a flag and a handler instead of a process type.
    """

    IMAGE_GEN = enum.auto()
    """Image generation jobs (the inference hot path)."""
    SAFETY_EVAL = enum.auto()
    """Post-generation safety evaluation."""
    ALCHEMY_GRAPH = enum.auto()
    """Graph-backed alchemy forms: upscalers, facefixers, strip_background."""
    ALCHEMY_CLIP = enum.auto()
    """CLIP-stack alchemy forms: caption, interrogation, nsfw."""


DEFAULT_CAPABILITIES: dict[HordeProcessType, WorkerCapability] = {
    HordeProcessType.INFERENCE: WorkerCapability.IMAGE_GEN,
    HordeProcessType.SAFETY: WorkerCapability.SAFETY_EVAL | WorkerCapability.ALCHEMY_CLIP,
    HordeProcessType.POST_PROCESS: WorkerCapability.ALCHEMY_GRAPH,
    HordeProcessType.DOWNLOAD: WorkerCapability(0),
    HordeProcessType.COMPONENT: WorkerCapability(0),
    HordeProcessType.VAE_LANE: WorkerCapability(0),
}
"""The capabilities each process type declares by default."""


class HordeProcess(abc.ABC):
    """The base class for all sub-processes."""

    process_id: int
    """The ID of the process. This is not the same as the process PID."""
    process_type: HordeProcessType
    """The type of process. This distinguishes between inference, safety, and potentially other process types."""
    process_message_queue: ProcessQueue
    """The queue the main process uses to receive messages from all worker processes."""
    pipe_connection: Connection  # FIXME # TODO - this could be a Queue?
    """Receives `HordeControlMessage`s from the main process."""

    disk_lock: Lock
    """A lock used to prevent multiple processes from accessing disk at the same time."""

    process_launch_identifier: int
    """The unique identifier for this launch."""

    _loop_interval: float = 0.02
    """The time to sleep between each loop iteration."""

    _end_process: bool = False
    """Whether the process should end soon."""

    _memory_report_interval: float = 5.0
    """The cadence (seconds) of the dedicated reporter thread's interval memory report."""

    _periodic_report_includes_vram: bool = False
    """Whether the interval-driven memory report should sample VRAM.

    GPU-bearing processes (inference, the post-process/VAE/component lanes, and the safety process when it
    holds models on the GPU) set this True so the main process keeps a fresh device-wide free-VRAM figure and
    per-process allocator attribution to budget against. A process with no GPU allocator to read leaves it
    False. When True, the reporter thread still withholds the VRAM read until the device context is already
    initialised (see :meth:`_offthread_vram_sampling_ready`), so the thread never triggers the device init.
    """

    _last_sent_process_state: HordeProcessState = HordeProcessState.PROCESS_STARTING
    """The last process state that was sent to the main process."""

    def get_vram_usage_mb(self) -> int:
        """Return ``torch_total - torch_free`` (mem_get_info): device-wide used on Linux, per-process view on Windows.

        The platform semantics are load-bearing and differ:

        - **On Linux**, ``mem_get_info`` is device-wide from any process, so this is a true whole-device used
          reading that conflates the shared device baseline (OS/desktop/other apps), every process's fixed
          context overhead, all resident model weights, and any in-flight activation peak. Its correct
          consumer is the parent's free-VRAM computation (``total - used`` gives device-wide free).
        - **On Windows/WDDM**, ``mem_get_info`` is a *per-process view*: the process sees roughly its own
          baseline plus context plus usage and is blind to siblings' usage. This figure is therefore NOT
          device truth on Windows and must never be treated as such, nor read as a device-wide free.

        On neither platform is this a valid *per-process charge*: on Linux it includes the whole device
        baseline and every sibling's VRAM (folding the shared baseline into one context, see the
        baseline-vs-marginal contract in ``scheduling/context_overhead_model``); on Windows it still bundles
        this process's own baseline and context with its usage. The honest per-process charge is
        ``process_reserved_mb`` (see :meth:`get_process_vram_stats`) plus the platform context constant, and
        the truthful device-wide figure on Windows comes only from the parent-side NVML device-total-used read.

        Device-wide free (rather than comfy's ``get_free_memory``, which adds back this process's reclaimable
        torch cache and so under-counts usage and over-states free) is used deliberately: an inflated free
        here is exactly what lets the scheduler admit a heavy model co-resident that then streams, and only
        the device-wide figure correctly attributes VRAM held by other (including leaked) processes that the
        per-process comfy number hides.
        """
        from hordelib.api import get_torch_device_free_vram_mb, get_torch_total_vram_mb

        return get_torch_total_vram_mb() - get_torch_device_free_vram_mb()

    def get_process_vram_stats(self) -> tuple[int, int, int, int] | None:
        """Return this process's own ``(allocated_mb, reserved_mb, peak_reserved_mb, aimdo_mb)``, or None off-GPU.

        Byte-exact, platform-independent, sibling-independent per-process attribution. The first three come
        from the torch allocator (``memory_reserved`` excludes the CUDA context); reading resets the
        allocator's peak counter, so ``peak_reserved_mb`` is the high-water since the previous report.
        ``aimdo_mb`` is the disjoint complement the torch allocator cannot see: the engine's direct-IO weight
        pool captured *if* that subsystem is initialised. It is inert (always 0) in the current embedding
        because nothing calls its native init, so weights normally live in the torch caching allocator and are
        counted by ``reserved_mb``. None when there is no GPU allocator to read, so the memory report simply
        omits the fields.
        """
        from hordelib.api import get_process_vram_stats

        stats = get_process_vram_stats(reset_peak=True)
        if stats is None:
            return None
        return stats.allocated_mb, stats.reserved_mb, stats.peak_reserved_mb, stats.aimdo_mb

    def get_vram_total_mb(self) -> int:
        """Return the total MB of VRAM available on the GPU."""
        from hordelib.api import get_torch_total_vram_mb

        return get_torch_total_vram_mb()

    def __init__(
        self,
        process_id: int,
        process_message_queue: ProcessQueue,
        pipe_connection: Connection,
        disk_lock: Lock,
        process_launch_identifier: int,
        device_index: int = 0,
    ) -> None:
        """Initialise the process.

        Args:
            process_id (int): The ID of the process. This is not the same as the process PID.
            process_message_queue (ProcessQueue): The queue the main process uses to receive messages from all worker \
                processes.
            pipe_connection (Connection): Receives `HordeControlMessage`s from the main process.
            disk_lock (Lock): A lock used to prevent multiple processes from accessing disk at the same time.
            process_launch_identifier (int): The unique identifier for this launch.
            device_index (int, optional): The stable index of the GPU this process is pinned to, reported \
                back on memory messages so the parent can attribute VRAM per card. Defaults to 0.
        """
        self.process_id = process_id
        self.process_message_queue = process_message_queue
        self.pipe_connection = pipe_connection
        self.disk_lock = disk_lock
        self.process_launch_identifier = process_launch_identifier
        self.device_index = device_index

        self._control_inbox: queue.SimpleQueue[object] = queue.SimpleQueue()
        self._control_reader_stop = threading.Event()
        self._control_reader_thread: threading.Thread | None = None

        self._memory_reporter_stop = threading.Event()
        self._memory_reporter_thread: threading.Thread | None = None

        self.send_process_state_change_message(
            process_state=HordeProcessState.PROCESS_STARTING,
            info="Process starting",
        )

    def _start_control_pipe_reader(self) -> None:
        """Start a daemon thread that drains the control pipe into an in-memory inbox.

        The main loop handles a preload (and its blocking aux-model download) synchronously, so without
        this it stops calling ``recv()`` for the whole download. The parent's ``safe_send_message`` is a
        blocking ``pipe_connection.send()``; once the OS pipe buffer to a non-draining child fills, that
        send blocks the parent's *entire* control loop, which then stops publishing supervisor snapshots
        and the dashboard ages into a false "Worker Unresponsive". Draining on a dedicated thread keeps
        the pipe readable at all times so the parent can never wedge on a busy child. Messages are only
        *handled* on the main loop (see :meth:`receive_and_handle_control_messages`); this thread never
        touches model/GPU state. The download process already does this (its ``_control_loop``); this
        brings the inference and safety processes to parity.
        """
        if self._control_reader_thread is not None:
            return
        thread = threading.Thread(
            target=self._control_pipe_reader_loop,
            name=f"horde-control-reader-{self.process_id}",
            daemon=True,
        )
        self._control_reader_thread = thread
        thread.start()

    def _control_pipe_reader_loop(self) -> None:
        """Block on the control pipe and forward each message to the inbox until told to stop."""
        while not self._control_reader_stop.is_set():
            try:
                # Poll with a timeout so the thread can observe the stop event between messages rather
                # than blocking forever in recv() after the main loop has decided to exit.
                if not self.pipe_connection.poll(0.1):
                    continue
                message = self.pipe_connection.recv()
            except (EOFError, OSError):
                # Parent gone / pipe closed: ask the main loop to exit, then stop draining.
                self._end_process = True
                return
            self._control_inbox.put(message)

    def _start_memory_reporter_thread(self) -> None:
        """Start a daemon thread that emits the interval memory report independent of the main loop.

        The interval report must keep flowing while the main loop is blocked for the whole duration of a
        GPU operation (a 20-150s sample). If it rode the main loop, every report the parent held would be an
        idle-boundary snapshot taken *after* post-job cleanup (torch reserved back near the CUDA-context
        floor), so the multi-GB mid-job working set would be systematically invisible and the parent's
        committed-VRAM ledger would read far below the device's true usage. A dedicated thread samples on
        ``_memory_report_interval`` regardless of what the main thread is doing, so the parent sees the live
        working set.

        Safety contract: this thread reads *only* observational statistics and never touches model or GPU
        compute state. ``torch.cuda.memory_allocated/reserved/max_reserved`` are lock-protected allocator
        bookkeeping reads, safe from any thread; ``torch.cuda.mem_get_info`` is a CUDA runtime query, safe to
        call cross-thread *once the context is initialised* because a CUDA context is per-process (not
        per-thread), so any thread's query hits the same shared context without disturbing the main thread's
        compute stream. The thread never calls ``empty_cache`` and never mutates any model. The one thing it
        must not do is *initialise* the device: ``mem_get_info`` and ``reset_peak_memory_stats`` lazily create
        the context on the calling thread when none exists, and doing that off the main thread is the hazard.
        So VRAM sampling is gated on :meth:`_offthread_vram_sampling_ready`, which reports readiness only once
        the main thread has already initialised the device; until then the thread reports RAM/FDs and omits
        the VRAM fields.
        """
        if self._memory_reporter_thread is not None:
            return
        thread = threading.Thread(
            target=self._memory_reporter_loop,
            name=f"horde-memory-reporter-{self.process_id}",
            daemon=True,
        )
        self._memory_reporter_thread = thread
        thread.start()

    def _memory_reporter_loop(self) -> None:
        """Emit an interval memory report on the reporter thread until told to stop.

        Sends an immediate first report on start (matching the old main-loop behaviour of reporting soon
        after startup), then one per ``_memory_report_interval``. Any exception in a single sample is
        swallowed and logged so a transient failure never kills the thread and silences all future reports.
        """
        while not self._memory_reporter_stop.is_set():
            try:
                include_vram = self._periodic_report_includes_vram and self._offthread_vram_sampling_ready()
                self.send_memory_report_message(include_vram=include_vram)
            except Exception as e:
                logger.error(f"Memory reporter thread failed to send a report: {type(e).__name__} {e}")
            self._memory_reporter_stop.wait(self._memory_report_interval)

    def _offthread_vram_sampling_ready(self) -> bool:
        """Whether the reporter thread may sample device VRAM without itself triggering a device init.

        The reporter thread must never be the first caller to touch the device: ``mem_get_info`` and the
        allocator peak reset both lazily create the CUDA context on the calling thread, and doing that off
        the main thread is the hazard. This defers to hordelib's readiness predicate, which reads only the
        runtime's already-initialised flag (it never creates a context) and returns True once the main thread
        has initialised the device. Overridable so a simulation subprocess whose VRAM getters are synthetic
        (no torch) can report without the real device gate.

        Imports the predicate from ``hordelib.utils.torch_memory`` directly rather than the ``hordelib.api``
        facade: the reporter thread also runs in the safety process, which must never trigger a ComfyUI
        import, and that module is the single, ComfyUI-free source of accelerator truth.
        """
        from hordelib.utils.torch_memory import offthread_vram_sampling_ready

        return offthread_vram_sampling_ready()

    def send_process_state_change_message(
        self,
        process_state: HordeProcessState,
        info: str,
        time_elapsed: float | None = None,
    ) -> None:
        """Send a process state change message to the main process.

        Args:
            process_state (HordeProcessState): The state of the process.
            info (str): Information about the process.
            time_elapsed (float | None, optional): The time elapsed during the last operation, if applicable. \
                Defaults to None.

        """
        message = HordeProcessStateChangeMessage(
            process_state=process_state,
            process_id=self.process_id,
            process_launch_identifier=self.process_launch_identifier,
            reported_os_pid=os.getpid(),
            info=info,
            time_elapsed=time_elapsed,
        )
        self.process_message_queue.put(message)
        self._last_sent_process_state = process_state

    _heartbeat_limit_interval_seconds: float = 1.0
    _last_heartbeat_time: float = 0.0
    _last_heartbeat_type: HordeHeartbeatType = HordeHeartbeatType.OTHER

    _IDLE_HEARTBEAT_STATES: frozenset[HordeProcessState] = frozenset(
        {
            HordeProcessState.WAITING_FOR_JOB,
            HordeProcessState.INFERENCE_COMPLETE,
            HordeProcessState.ALCHEMY_COMPLETE,
            HordeProcessState.PRELOADED_MODEL,
        },
    )
    """States in which it is safe to emit a liveness heartbeat: alive and looping, but not mid-job."""

    def send_heartbeat_message(
        self,
        heartbeat_type: HordeHeartbeatType,
        *,
        process_warning: str | None = None,
        percent_complete: int | None = None,
        current_step: int | None = None,
        total_steps: int | None = None,
        iterations_per_second: float | None = None,
        nonadvancing_step_repeats: int = 0,
    ) -> None:
        """Send a heartbeat message to the main process, indicating that the process is still alive.

        Note that this will only send a heartbeat message if the last heartbeat was sent more than
        `_heartbeat_limit_interval_seconds` ago or if the heartbeat type has changed. A type change is
        always forwarded immediately (it is a meaningful transition); only repeated same-type heartbeats
        inside the window are throttled.
        """
        if (heartbeat_type == self._last_heartbeat_type) and (
            time.time() - self._last_heartbeat_time
        ) < self._heartbeat_limit_interval_seconds:
            return

        message = HordeProcessHeartbeatMessage(
            process_id=self.process_id,
            process_launch_identifier=self.process_launch_identifier,
            reported_os_pid=os.getpid(),
            info="Heartbeat",
            time_elapsed=None,
            heartbeat_type=heartbeat_type,
            process_warning=process_warning,
            percent_complete=percent_complete,
            current_step=current_step,
            total_steps=total_steps,
            iterations_per_second=iterations_per_second,
            nonadvancing_step_repeats=nonadvancing_step_repeats,
        )
        self.process_message_queue.put(message)

        self._last_heartbeat_type = heartbeat_type
        self._last_heartbeat_time = time.time()

    def _maybe_send_idle_heartbeat(self) -> None:
        """Emit a throttled liveness heartbeat while idle so the dashboard's heartbeat stays fresh.

        Without this an idle process never refreshes its heartbeat timestamp and reads as unresponsive
        in the live view; the safety process is the worst case, sitting in ``WAITING_FOR_JOB`` between
        checks that are each over in milliseconds. Restricted to idle states and gated by an explicit
        interval check (the heartbeat throttle only gates *type changes*) so it never interleaves with
        the ``INFERENCE_STEP`` stream that hung-process detection relies on.
        """
        if self._last_sent_process_state not in self._IDLE_HEARTBEAT_STATES:
            return
        if time.time() - self._last_heartbeat_time < self._heartbeat_limit_interval_seconds:
            return
        self.send_heartbeat_message(HordeHeartbeatType.OTHER)

    @abstractmethod
    def cleanup_for_exit(self) -> None:
        """Cleanup and exit the process."""

    def send_memory_report_message(
        self,
        include_vram: bool = False,
    ) -> bool:
        """Send a memory report message to the main process.

        Args:
            include_vram (bool, optional): Whether to include VRAM usage in the message. Defaults to False.
        """
        message = HordeProcessMemoryMessage(
            process_id=self.process_id,
            process_launch_identifier=self.process_launch_identifier,
            reported_os_pid=os.getpid(),
            info="Memory report",
            time_elapsed=None,
            ram_usage_bytes=psutil.Process().memory_info().rss,
            open_fds=open_descriptor_count(),
            fd_soft_limit=descriptor_soft_limit(),
            device_index=self.device_index,
            sampled_at=time.time(),
        )

        try:
            if include_vram:
                message.vram_usage_mb = self.get_vram_usage_mb()
                message.vram_total_mb = self.get_vram_total_mb()
                process_stats = self.get_process_vram_stats()
                if process_stats is not None:
                    (
                        message.process_allocated_mb,
                        message.process_reserved_mb,
                        message.process_peak_reserved_mb,
                        message.process_aimdo_mb,
                    ) = process_stats
        except Exception as e:
            logger.error(f"Failed to get VRAM usage: {e}")
            return False

        self.process_message_queue.put(message)
        return True

    @abstractmethod
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        """Receive and handle a control message from the main process.

        Args:
            message (HordeControlMessage): The message to handle.
        """

    def receive_and_handle_control_messages(self) -> None:
        """Handle any control messages the reader thread has drained from the main process."""
        while True:
            try:
                message = self._control_inbox.get_nowait()
            except queue.Empty:
                return

            if not isinstance(message, HordeControlMessage):
                logger.critical(f"Received unexpected message type: {type(message).__name__}")
                continue

            if message.control_flag == HordeControlFlag.END_PROCESS:
                self._end_process = True
                logger.info("Received end process message")
                return

            try:
                self._receive_and_handle_control_message(message)
            except UnsupportedControlMessageError as e:
                # A routing error on the parent's side: no handler ran, so this process's state is intact.
                # Dropping the message keeps the process alive; ending it would convert a sender bug into a
                # crash-restart loop for a healthy child.
                logger.error(f"Dropped a control message this process does not support: {e}")
            except Exception as e:
                logger.error(f"Failed to handle control message: {type(e).__name__} {e}")
                # A supported handler failed mid-action, so the process state is unknown: exit and let the
                # parent recover a fresh process.
                self._end_process = True

    def worker_cycle(self) -> None:
        """Do any process specific handling after messages have been received and handled.

        Override this to implement any process specific logic.
        """
        return

    def main_loop(self) -> None:
        """Start the main loop of the process."""
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self._start_control_pipe_reader()
        self._start_memory_reporter_thread()

        while not self._end_process:
            time.sleep(self._loop_interval)
            self.receive_and_handle_control_messages()
            self.worker_cycle()
            self._maybe_send_idle_heartbeat()

        self._control_reader_stop.set()
        self._memory_reporter_stop.set()

        # We escaped the loop, so the process is ending
        self.send_process_state_change_message(
            process_state=HordeProcessState.PROCESS_ENDING,
            info="Process ending",
        )

        self.cleanup_for_exit()

        logger.info("Process ended")
        self.send_process_state_change_message(
            process_state=HordeProcessState.PROCESS_ENDED,
            info="Process ended",
        )

        # We are exiting, so send a final memory report
        self.send_memory_report_message(include_vram=False)

        # Exit the process (we expect to be a child process)
        sys.exit(0)


_signals_caught = 0


def signal_handler(sig: int, frame: object) -> None:
    """Handle a signal.

    This will exit the process gracefully if the process has only received one signal,
    or exit immediately if the process has received two signals.
    """
    global _signals_caught
    if _signals_caught >= 2:
        logger.warning("Received second signal, exiting immediately")
        sys.exit(0)

    logger.info("Received signal, exiting gracefully")
    _signals_caught += 1
