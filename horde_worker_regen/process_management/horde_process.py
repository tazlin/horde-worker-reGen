"""Contains the base class for all processes, and additional helper types used by processes."""

from __future__ import annotations

import abc
import enum
import signal
import sys
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

    _last_sent_process_state: HordeProcessState = HordeProcessState.PROCESS_STARTING
    """The last process state that was sent to the main process."""

    def get_vram_usage_mb(self) -> int:
        """Return the MB of VRAM used on the GPU."""
        from hordelib.api import get_torch_free_vram_mb, get_torch_total_vram_mb

        return get_torch_total_vram_mb() - get_torch_free_vram_mb()

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
    ) -> None:
        """Initialise the process.

        Args:
            process_id (int): The ID of the process. This is not the same as the process PID.
            process_message_queue (ProcessQueue): The queue the main process uses to receive messages from all worker \
                processes.
            pipe_connection (Connection): Receives `HordeControlMessage`s from the main process.
            disk_lock (Lock): A lock used to prevent multiple processes from accessing disk at the same time.
            process_launch_identifier (int): The unique identifier for this launch.
        """
        self.process_id = process_id
        self.process_message_queue = process_message_queue
        self.pipe_connection = pipe_connection
        self.disk_lock = disk_lock
        self.process_launch_identifier = process_launch_identifier

        self.send_process_state_change_message(
            process_state=HordeProcessState.PROCESS_STARTING,
            info="Process starting",
        )

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
        `_heartbeat_limit_interval_seconds` ago or if the heartbeat type has changed.
        """
        if (heartbeat_type != self._last_heartbeat_type) and (
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
        """Get and handle any control messages pending from the main process."""
        while self.pipe_connection.poll():
            message = self.pipe_connection.recv()

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

        while not self._end_process:
            time.sleep(self._loop_interval)
            self.receive_and_handle_control_messages()
            self.worker_cycle()

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
