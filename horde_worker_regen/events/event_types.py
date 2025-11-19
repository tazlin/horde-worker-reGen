"""Event type definitions for the worker event system.

All events inherit from WorkerEvent and are immutable dataclasses.
Events capture state changes and significant occurrences in the worker lifecycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from horde_model_reference.meta_consts import STABLE_DIFFUSION_BASELINE_CATEGORY
    from horde_sdk.ai_horde_api.fields import JobID

    from horde_worker_regen.process_management.messages import (
        HordeHeartbeatType,
        HordeProcessState,
        ModelLoadState,
    )


class EventPriority(Enum):
    """Priority levels for events (for potential filtering/routing)."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class WorkerEvent:
    """Base class for all worker events.

    All events are immutable and timestamped.
    """

    timestamp: float = field(default_factory=time.time)
    """The epoch timestamp when this event was created."""

    priority: EventPriority = EventPriority.NORMAL
    """The priority level of this event."""


# ============================================================================
# Process Events - Related to child process lifecycle and state
# ============================================================================


@dataclass(frozen=True)
class ProcessStartedEvent(WorkerEvent):
    """Emitted when a child process has started."""

    process_id: int
    """The ID of the process that started."""

    process_type: str
    """The type of process (inference or safety)."""

    process_launch_identifier: int
    """The launch identifier for tracking process restarts."""


@dataclass(frozen=True)
class ProcessEndedEvent(WorkerEvent):
    """Emitted when a child process has ended."""

    process_id: int
    """The ID of the process that ended."""

    process_type: str
    """The type of process (inference or safety)."""

    exit_code: int | None = None
    """The exit code of the process, if available."""


@dataclass(frozen=True)
class ProcessStateChangedEvent(WorkerEvent):
    """Emitted when a process changes state."""

    process_id: int
    """The ID of the process whose state changed."""

    old_state: HordeProcessState
    """The previous state of the process."""

    new_state: HordeProcessState
    """The new state of the process."""

    loaded_model_name: str | None = None
    """The name of the model loaded in this process, if any."""


@dataclass(frozen=True)
class ProcessHeartbeatEvent(WorkerEvent):
    """Emitted when a heartbeat is received from a child process."""

    process_id: int
    """The ID of the process sending the heartbeat."""

    heartbeat_type: HordeHeartbeatType
    """The type of heartbeat (inference step, pipeline state change, etc.)."""

    percent_complete: int | None = None
    """The percentage complete of the current operation (0-100), if applicable."""

    process_state: HordeProcessState | None = None
    """The current state of the process."""

    priority: EventPriority = EventPriority.LOW  # High frequency, low priority


@dataclass(frozen=True)
class ProcessMemoryUpdatedEvent(WorkerEvent):
    """Emitted when process memory usage is updated."""

    process_id: int
    """The ID of the process whose memory was updated."""

    ram_usage_bytes: int
    """The amount of RAM used by this process in bytes."""

    vram_usage_bytes: int
    """The amount of VRAM used by this process in bytes."""

    vram_total_bytes: int
    """The total amount of VRAM available to this process in bytes."""

    priority: EventPriority = EventPriority.LOW  # Frequent updates


# ============================================================================
# Job Events - Related to job lifecycle and processing
# ============================================================================


@dataclass(frozen=True)
class JobPoppedEvent(WorkerEvent):
    """Emitted when a new job is popped from the API."""

    job_id: JobID
    """The unique identifier for this job."""

    model_name: str
    """The name of the model requested for this job."""

    width: int
    """The width of the image to generate."""

    height: int
    """The height of the image to generate."""

    steps: int
    """The number of inference steps."""

    batch_size: int
    """The number of images in the batch."""

    estimated_megapixelsteps: int
    """The estimated complexity of this job in megapixelsteps."""

    has_source_image: bool = False
    """Whether this job includes a source image (img2img)."""

    has_loras: bool = False
    """Whether this job uses LoRA models."""

    has_controlnet: bool = False
    """Whether this job uses ControlNet."""


@dataclass(frozen=True)
class JobStartedEvent(WorkerEvent):
    """Emitted when a job starts being processed by a worker process."""

    job_id: JobID
    """The unique identifier for this job."""

    process_id: int
    """The ID of the process handling this job."""

    model_name: str
    """The name of the model being used."""


@dataclass(frozen=True)
class JobCompletedEvent(WorkerEvent):
    """Emitted when a job is successfully completed and submitted."""

    job_id: JobID
    """The unique identifier for this job."""

    process_id: int | None
    """The ID of the process that completed this job."""

    model_name: str
    """The name of the model used."""

    kudos_earned: float
    """The amount of kudos earned for this job."""

    generation_time_seconds: float
    """The total time taken to generate this job."""

    num_images: int
    """The number of images generated."""

    cumulative_kudos: float | None = None
    """The cumulative kudos earned this session, if available."""


@dataclass(frozen=True)
class JobFaultedEvent(WorkerEvent):
    """Emitted when a job fails or is faulted."""

    job_id: JobID
    """The unique identifier for this job."""

    process_id: int | None
    """The ID of the process that faulted this job, if known."""

    model_name: str
    """The name of the model being used."""

    fault_type: str
    """The type of fault that occurred."""

    fault_message: str | None = None
    """Detailed message about the fault, if available."""

    priority: EventPriority = EventPriority.HIGH


@dataclass(frozen=True)
class JobQueueChangedEvent(WorkerEvent):
    """Emitted when the job queue state changes."""

    pending_inference: int
    """Number of jobs pending inference."""

    in_progress: int
    """Number of jobs currently being processed."""

    pending_safety_check: int
    """Number of jobs pending safety evaluation."""

    being_safety_checked: int
    """Number of jobs currently being safety checked."""

    pending_submit: int
    """Number of jobs pending submission to API."""

    priority: EventPriority = EventPriority.LOW  # Frequent updates


# ============================================================================
# Model Events - Related to model downloading and loading
# ============================================================================


@dataclass(frozen=True)
class ModelDownloadStartedEvent(WorkerEvent):
    """Emitted when a model download starts."""

    model_name: str
    """The name of the model being downloaded."""

    process_id: int
    """The ID of the process downloading the model."""


@dataclass(frozen=True)
class ModelDownloadProgressEvent(WorkerEvent):
    """Emitted during model download to report progress."""

    model_name: str
    """The name of the model being downloaded."""

    process_id: int
    """The ID of the process downloading the model."""

    percent_complete: float
    """The download progress percentage (0.0-100.0)."""

    priority: EventPriority = EventPriority.LOW  # Frequent updates


@dataclass(frozen=True)
class ModelLoadingEvent(WorkerEvent):
    """Emitted when a model starts loading into memory."""

    model_name: str
    """The name of the model being loaded."""

    process_id: int
    """The ID of the process loading the model."""

    load_stage: ModelLoadState
    """The current load stage (LOADING, LOADED_IN_RAM, LOADED_IN_VRAM, etc.)."""


@dataclass(frozen=True)
class ModelLoadedEvent(WorkerEvent):
    """Emitted when a model is successfully loaded."""

    model_name: str
    """The name of the model that was loaded."""

    process_id: int
    """The ID of the process that loaded the model."""

    location: str
    """Where the model was loaded ('RAM', 'VRAM', etc.)."""

    model_baseline: STABLE_DIFFUSION_BASELINE_CATEGORY | str | None = None
    """The baseline of the loaded model."""


@dataclass(frozen=True)
class ModelUnloadedEvent(WorkerEvent):
    """Emitted when a model is unloaded from memory."""

    model_name: str
    """The name of the model that was unloaded."""

    process_id: int
    """The ID of the process that unloaded the model."""

    location: str
    """Where the model was unloaded from ('RAM', 'VRAM', etc.)."""


# ============================================================================
# Worker Events - Related to overall worker status and lifecycle
# ============================================================================


@dataclass(frozen=True)
class WorkerStartedEvent(WorkerEvent):
    """Emitted when the worker starts up."""

    worker_name: str
    """The name of this worker."""

    worker_version: str
    """The version of the worker software."""

    max_threads: int
    """Maximum number of concurrent inference processes."""

    max_power: int
    """Maximum power (complexity) this worker will accept."""

    num_models: int
    """Number of models this worker can serve."""

    priority: EventPriority = EventPriority.HIGH


@dataclass(frozen=True)
class WorkerStatusEvent(WorkerEvent):
    """Emitted periodically with complete worker status snapshot.

    This is a comprehensive event containing all status information,
    suitable for periodic UI refreshes.
    """

    # Process information
    num_processes_active: int
    """Number of active processes."""

    num_processes_idle: int
    """Number of idle processes."""

    num_processes_busy: int
    """Number of busy processes."""

    # Job information
    jobs_pending_inference: int
    """Number of jobs pending inference."""

    jobs_in_progress: int
    """Number of jobs in progress."""

    jobs_pending_safety: int
    """Number of jobs pending safety check."""

    jobs_being_safety_checked: int
    """Number of jobs being safety checked."""

    total_jobs_popped: int
    """Total number of jobs popped this session."""

    total_jobs_completed: int
    """Total number of jobs completed this session."""

    total_jobs_faulted: int
    """Total number of jobs faulted this session."""

    # Model information
    active_models: set[str]
    """Set of currently loaded model names."""

    # Performance metrics
    cumulative_kudos: float
    """Total kudos earned this session."""

    time_without_jobs_seconds: float
    """Time spent without jobs available."""

    session_uptime_seconds: float
    """Total session uptime in seconds."""

    # Memory usage
    total_ram_usage_gb: float | None = None
    """Total RAM usage across all processes in GB."""

    total_vram_usage_gb: float | None = None
    """Total VRAM usage across all processes in GB."""

    # Additional status fields
    status_message: str | None = None
    """Optional status message."""

    is_shutting_down: bool = False
    """Whether the worker is shutting down."""

    priority: EventPriority = EventPriority.LOW  # Regular status updates


@dataclass(frozen=True)
class APIMessageReceivedEvent(WorkerEvent):
    """Emitted when a message is received from the Horde API."""

    message_id: str
    """The unique identifier for this message."""

    message_text: str
    """The text content of the message."""

    message_origin: str
    """The origin of the message (e.g., 'horde', 'admin')."""

    message_expiry: str | None = None
    """When this message expires, if applicable."""

    severity: str = "info"
    """The severity level of the message."""

    priority: EventPriority = EventPriority.HIGH


@dataclass(frozen=True)
class MaintenanceModeEvent(WorkerEvent):
    """Emitted when maintenance mode status changes."""

    is_maintenance_mode: bool
    """Whether the API is in maintenance mode."""

    message: str | None = None
    """Optional maintenance message."""

    priority: EventPriority = EventPriority.HIGH


@dataclass(frozen=True)
class ShutdownInitiatedEvent(WorkerEvent):
    """Emitted when worker shutdown is initiated."""

    reason: str
    """The reason for shutdown (e.g., 'user_requested', 'signal', 'error')."""

    graceful: bool = True
    """Whether this is a graceful shutdown."""

    priority: EventPriority = EventPriority.CRITICAL


# ============================================================================
# Performance Events - Related to performance and warnings
# ============================================================================


@dataclass(frozen=True)
class KudosEarnedEvent(WorkerEvent):
    """Emitted when kudos are earned from a completed job."""

    job_id: JobID
    """The job that earned the kudos."""

    amount: float
    """The amount of kudos earned."""

    cumulative_total: float
    """The cumulative total kudos earned this session."""

    job_time_seconds: float | None = None
    """The time taken for this job, if available."""


@dataclass(frozen=True)
class PerformanceWarningEvent(WorkerEvent):
    """Emitted when a performance warning is detected."""

    warning_type: str
    """The type of warning (e.g., 'slow_job', 'high_memory', 'stuck_process')."""

    message: str
    """Detailed warning message."""

    process_id: int | None = None
    """The process ID related to this warning, if applicable."""

    severity: str = "warning"
    """The severity level ('info', 'warning', 'error', 'critical')."""

    metadata: dict[str, Any] | None = None
    """Additional metadata about the warning."""

    priority: EventPriority = EventPriority.HIGH
