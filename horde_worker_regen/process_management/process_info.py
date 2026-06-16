"""Contains information about a horde child process."""

from __future__ import annotations

import multiprocessing
import time
from typing import override

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from hordelib.metrics import DownloadEvent, JobPhaseMetrics
from loguru import logger

from horde_worker_regen.process_management.horde_process import (
    DEFAULT_CAPABILITIES,
    HordeProcessType,
    WorkerCapability,
)
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
    os_pid: int | None
    """The OS process id (``mp_process.pid``), captured right after the process starts.

    Distinct from ``process_id`` (a stable logical slot 0,1,2...). Used to take ownership of the
    real process: it is logged in crash/timeout diagnostics and recorded in the owned-PID registry so
    a parent that dies hard can have its orphaned children reaped on the next startup. None until the
    process has been started.
    """
    process_type: HordeProcessType
    """The type of this process."""
    capabilities: WorkerCapability
    """The kinds of work this process can be dispatched (job routing keys on this)."""
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
    loaded_horde_model_baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None
    """The baseline of the horde model that is (supposedly) currently loaded in this process."""
    last_control_flag: HordeControlFlag | None
    """The last control flag sent, to avoid duplication."""

    last_job_referenced: ImageGenerateJobPopResponse | None

    current_inference_started_at: float | None
    """Epoch time inference was dispatched to this slot for its current job, or None when not inferring.

    Set by the scheduler at dispatch and cleared when a result (or fault) arrives, so the graded-slowdown
    monitor can measure how long the slot has actually been sampling against the expected sampling time."""
    current_job_expected_sampling_seconds: float | None
    """The performance model's expected sampling seconds for the slot's current job, or None when unknown.

    Travels with the slot from dispatch so the watchdog can grade "slower than expected" without itself
    needing the performance model. None on a cold start (no seed/calibration yet) suppresses grading."""
    current_job_slowdown_level: int
    """Highest slowdown rung already logged for the current job (0 none, 1 notice, 2 warn).

    Reset to 0 at each dispatch so the graded-slowdown monitor escalates a job's notices at most once
    per rung instead of every watchdog tick."""

    ram_usage_bytes: int
    """The amount of RAM used by this process."""
    vram_usage_mb: int
    """The amount of VRAM (MB) used by this process."""
    total_vram_mb: int
    """The total amount of VRAM (MB) available to this process."""
    batch_amount: int
    """The total amount of batching being run by this process."""

    last_iterations_per_second: float | None
    """The most recent sampling rate reported by this process (-1.0 = not yet known)."""
    last_current_step: int | None
    """The most recent sampling step reported by this process."""
    last_total_steps: int | None
    """The total steps of the most recent sampling run reported by this process."""

    last_job_metrics: JobPhaseMetrics | None
    """The per-job metrics snapshot from the most recently finished job on this process."""
    vram_used_high_water_mb: int
    """The highest in-job VRAM usage (MB) ever reported by this process (0 = none yet)."""
    ram_used_high_water_mb: int
    """The highest in-job RAM usage (MB) ever reported by this process (0 = none yet)."""
    num_jobs_completed: int
    """Count of jobs/forms this slot has finished (inference result, safety check, or alchemy form).

    Resets to 0 when the slot is replaced (a fresh ``HordeProcessInfo`` is built), so it reads as the
    work done by the *current* process. Surfaced per-process in the live view as running feedback,
    most usefully for the safety process whose checks are otherwise too fast to see.
    """
    cumulative_download_events: list[DownloadEvent]
    """All ad-hoc download events reported by this process, in arrival order."""

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
        capabilities: WorkerCapability | None = None,
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
            capabilities (WorkerCapability | None, optional): The work kinds this process serves. \
                Defaults to the process type's defaults.
        """
        self.mp_process = mp_process
        self.pipe_connection = pipe_connection
        self.process_id = process_id
        self.os_pid = mp_process.pid
        self.process_type = process_type
        self.capabilities = capabilities if capabilities is not None else DEFAULT_CAPABILITIES[process_type]
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
        self.current_inference_started_at = None
        self.current_job_expected_sampling_seconds = None
        self.current_job_slowdown_level = 0

        self.ram_usage_bytes = 0
        self.vram_usage_mb = 0
        self.total_vram_mb = 0
        self.batch_amount = 1

        self.last_iterations_per_second = None
        self.last_current_step = None
        self.last_total_steps = None

        self.last_job_metrics = None
        self.vram_used_high_water_mb = 0
        self.ram_used_high_water_mb = 0
        self.num_jobs_completed = 0
        self.cumulative_download_events = []

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
        return self.last_process_state not in (
            HordeProcessState.PROCESS_ENDING,
            HordeProcessState.PROCESS_ENDED,
        )

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

            # Pipe closure is expected when the child is ending/ended, or when the
            # worker is shutting down — log at debug level to avoid noise.
            _pipe_expected_to_close = (
                _caught_signal
                or self.last_process_state in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED)
                or not self.mp_process.is_alive()
            )
            if not _pipe_expected_to_close:
                logger.error(f"Failed to send message to process {self.process_id}: {e}")
            return False

    @override
    def __repr__(self) -> str:
        """Return a string representation of the process info."""
        return (
            f"HordeProcessInfo(process_id={self.process_id}, last_process_state={self.last_process_state}, "
            f"loaded_horde_model_name={self.loaded_horde_model_name})"
        )

    def can_accept_job(self) -> bool:
        """Return true if the process can accept a job."""
        return (
            self.last_process_state == HordeProcessState.WAITING_FOR_JOB
            or self.last_process_state == HordeProcessState.PRELOADED_MODEL
            or self.last_process_state == HordeProcessState.INFERENCE_COMPLETE
            or self.last_process_state == HordeProcessState.ALCHEMY_COMPLETE
        )
