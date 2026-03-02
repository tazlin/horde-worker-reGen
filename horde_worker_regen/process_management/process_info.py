"""Contains information about a horde child process."""

from __future__ import annotations

import multiprocessing
import time
from typing import TYPE_CHECKING

from horde_model_reference.meta_consts import STABLE_DIFFUSION_BASELINE_CATEGORY
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from loguru import logger

from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeHeartbeatType,
    HordeProcessState,
)

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore


class HordeProcessInfo:
    """Contains information about a horde child process."""

    mp_process: multiprocessing.Process
    """The multiprocessing.Process object for this process."""
    pipe_connection: Connection
    """The connection through which messages can be sent to this process."""
    process_id: int
    """The ID of this process. This is not an OS process ID."""
    process_type: HordeProcessType
    """The type of this process."""
    last_process_state: HordeProcessState
    """The last known state of this process."""

    last_heartbeat_timestamp: float
    """Last time we received a heartbeat from this process."""
    last_heartbeat_delta: float
    """The delta between the last two heartbeats. Used to determine if the process is stuck."""
    last_heartbeat_type: HordeHeartbeatType
    """The type of the last heartbeat received from this process."""
    heartbeats_inference_steps: int
    """The number of inference steps that have been completed since the last heartbeat."""
    last_heartbeat_percent_complete: int | None
    """The last percentage reported by the process."""

    last_received_timestamp: float
    """Last time we updated the process info. If we're regularly working, then this value should change frequently."""
    loaded_horde_model_name: str | None
    """The name of the horde model that is (supposedly) currently loaded in this process."""
    loaded_horde_model_baseline: STABLE_DIFFUSION_BASELINE_CATEGORY | str | None
    """The baseline of the horde model that is (supposedly) currently loaded in this process."""
    last_control_flag: HordeControlFlag | None
    """The last control flag sent, to avoid duplication."""

    last_job_referenced: ImageGenerateJobPopResponse | None

    ram_usage_bytes: int
    """The amount of RAM used by this process."""
    vram_usage_bytes: int
    """The amount of VRAM used by this process."""
    total_vram_bytes: int
    """The total amount of VRAM available to this process."""
    batch_amount: int
    """The total amount of batching being run by this process."""

    recently_unloaded_from_ram: bool
    """True if models were recently unloaded from RAM."""

    process_launch_identifier: int
    """The identifier for the process launch. Used to track restarting of specific process slots."""

    # TODO: VRAM usage

    def __init__(
        self,
        mp_process: multiprocessing.Process,
        pipe_connection: Connection,
        process_id: int,
        process_type: HordeProcessType,
        last_process_state: HordeProcessState,
        process_launch_identifier: int,
    ) -> None:
        """Initialize a new HordeProcessInfo object.

        Args:
            mp_process (multiprocessing.Process): The multiprocessing.Process object for this process.
            pipe_connection (Connection): The connection through which messages can be sent to this process.
            process_id (int): The ID of this process. This is not an OS process ID.
            process_type (HordeProcessType): The type of this process.
            last_process_state (HordeProcessState): The last known state of this process.
            process_launch_identifier (int): The identifier for the process launch. Used to track restarting of \
                specific process slots.
        """
        self.mp_process = mp_process
        self.pipe_connection = pipe_connection
        self.process_id = process_id
        self.process_type = process_type
        self.last_process_state = last_process_state
        self.last_received_timestamp = time.time()
        self.loaded_horde_model_name = None
        self.loaded_horde_model_baseline = None
        self.last_control_flag = None

        self.last_heartbeat_timestamp = time.time()
        self.last_heartbeat_delta = 0
        self.last_heartbeat_type = HordeHeartbeatType.OTHER
        self.heartbeats_inference_steps = 0
        self.last_heartbeat_percent_complete = None

        self.last_job_referenced = None

        self.ram_usage_bytes = 0
        self.vram_usage_bytes = 0
        self.total_vram_bytes = 0
        self.batch_amount = 1

        self.recently_unloaded_from_ram = False

        self.process_launch_identifier = process_launch_identifier

    def is_process_busy(self) -> bool:
        """Return true if the process is actively engaged in a task.

        This does not include the process starting up or shutting down.
        """
        return (
            self.last_process_state == HordeProcessState.INFERENCE_STARTING
            or self.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING
            or self.last_process_state == HordeProcessState.ALCHEMY_STARTING
            or self.last_process_state == HordeProcessState.DOWNLOADING_MODEL
            or self.last_process_state == HordeProcessState.DOWNLOADING_AUX_MODEL
            or self.last_process_state == HordeProcessState.PRELOADING_MODEL
            or self.last_process_state == HordeProcessState.PRELOADED_MODEL
            or self.last_process_state == HordeProcessState.JOB_RECEIVED
            or self.last_process_state == HordeProcessState.EVALUATING_SAFETY
            or self.last_process_state == HordeProcessState.PROCESS_STARTING
        )

    def is_process_alive(self) -> bool:
        """Return true if the process is alive."""
        if not self.mp_process.is_alive():
            return False
        return not (self.last_process_state == HordeProcessState.PROCESS_ENDING or HordeProcessState.PROCESS_ENDED)

    def safe_send_message(self, message: HordeControlMessage) -> bool:
        """Send a message to the process.

        Args:
            message (HordeControlMessage): The message to send.

        Returns:
            bool: True if the message was sent successfully, False otherwise.
        """
        try:
            self.pipe_connection.send(message)
            return True
        except Exception as e:
            from horde_worker_regen.process_management.process_manager import _caught_signal

            if not _caught_signal:
                logger.error(f"Failed to send message to process {self.process_id}: {e}")
            return False

    def __repr__(self) -> str:
        """Return a string representation of the process info."""
        return str(
            f"HordeProcessInfo(process_id={self.process_id}, last_process_state={self.last_process_state}, "
            f"loaded_horde_model_name={self.loaded_horde_model_name})",
        )

    def can_accept_job(self) -> bool:
        """Return true if the process can accept a job."""
        return (
            self.last_process_state == HordeProcessState.WAITING_FOR_JOB
            or self.last_process_state == HordeProcessState.PRELOADED_MODEL
            or self.last_process_state == HordeProcessState.INFERENCE_COMPLETE
            or self.last_process_state == HordeProcessState.ALCHEMY_COMPLETE
        )
