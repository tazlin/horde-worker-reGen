"""Contains the base class for all processes, and additional helper types used by processes."""

from __future__ import annotations

import abc
import enum
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

from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeHeartbeatType,
    HordeProcessHeartbeatMessage,
    HordeProcessMemoryMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
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
    HordeProcessType.INFERENCE: WorkerCapability.IMAGE_GEN | WorkerCapability.ALCHEMY_GRAPH,
    HordeProcessType.SAFETY: WorkerCapability.SAFETY_EVAL | WorkerCapability.ALCHEMY_CLIP,
    HordeProcessType.DOWNLOAD: WorkerCapability(0),
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
    """The time to wait between each memory report."""

    _last_periodic_memory_report_time: float = 0.0
    """Wall-clock time of the last interval-driven memory report (0 = none sent yet)."""

    _periodic_report_includes_vram: bool = False
    """Whether the interval-driven memory report should sample VRAM.

    Inference processes set this True so the main process keeps a fresh device-wide free-VRAM
    figure to budget against; the CPU-only safety process leaves it False (it has no GPU to sample).
    """

    _last_sent_process_state: HordeProcessState = HordeProcessState.PROCESS_STARTING
    """The last process state that was sent to the main process."""

    def get_vram_usage_mb(self) -> int:
        """Return the MB of VRAM used on the GPU.

        Uses device-wide free (mem_get_info) rather than comfy's get_free_memory: the latter adds back
        this process's reclaimable torch cache, so total - comfy_free under-counts usage and over-states
        free. The parent's VRAM budget and the whole-card streaming forecast are built from these reports,
        so an inflated free here is exactly what lets the scheduler admit a heavy model co-resident that
        then streams. Device-wide free also correctly attributes VRAM held by other (including leaked)
        processes, which the per-process comfy number hides.
        """
        from hordelib.api import get_torch_device_free_vram_mb, get_torch_total_vram_mb

        return get_torch_total_vram_mb() - get_torch_device_free_vram_mb()

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
            info="Heartbeat",
            time_elapsed=None,
            heartbeat_type=heartbeat_type,
            process_warning=process_warning,
            percent_complete=percent_complete,
            current_step=current_step,
            total_steps=total_steps,
            iterations_per_second=iterations_per_second,
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

    def _maybe_send_periodic_memory_report(self) -> None:
        """Emit an interval-driven memory report so the main process's free-VRAM view stays fresh.

        The event-driven reports (model load/unload, inference failure) only fire on state
        transitions, so during a long single job or an idle stretch the main process's last
        free-VRAM figure goes stale. The worker's VRAM/RAM budget gates dispatch on that figure, so
        a stale read can both over-commit (acting on freed-but-still-counted VRAM) and under-commit.
        This adds a low-frequency floor independent of transitions; the event-driven reports remain.
        """
        now = time.time()
        if now - self._last_periodic_memory_report_time < self._memory_report_interval:
            return
        self._last_periodic_memory_report_time = now
        self.send_memory_report_message(include_vram=self._periodic_report_includes_vram)

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
            info="Memory report",
            time_elapsed=None,
            ram_usage_bytes=psutil.Process().memory_info().rss,
            device_index=self.device_index,
        )

        try:
            if include_vram:
                message.vram_usage_mb = self.get_vram_usage_mb()
                message.vram_total_mb = self.get_vram_total_mb()
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
            except Exception as e:
                logger.error(f"Failed to handle control message: {type(e).__name__} {e}")
                # This is a terminal error, so we should exit
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

        while not self._end_process:
            time.sleep(self._loop_interval)
            self.receive_and_handle_control_messages()
            self.worker_cycle()
            self._maybe_send_idle_heartbeat()
            self._maybe_send_periodic_memory_report()

        self._control_reader_stop.set()

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
