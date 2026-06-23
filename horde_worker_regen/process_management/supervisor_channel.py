"""The supervisor channel: structured state + control between a TUI/supervisor and the worker.

A supervising frontend (``horde_worker_regen.tui``) launches the worker as a child process and holds
one end of a duplex pipe. The worker pushes :class:`WorkerStateSnapshot` objects at a steady cadence
and drains :class:`SupervisorControlMessage` commands each loop tick. This mirrors the worker's own
internal IPC (see ``messages.py``) and is the structured upgrade of the ``.abort``-sentinel external
supervision hook already present in the control loop.

Models here are deliberately pure-data and JSON-round-trippable: the default transport is a
``multiprocessing`` pipe (pickle), but the same models serialize cleanly for the localhost-socket
fallback the launcher can swap to without touching any screen code.
"""

from __future__ import annotations

import enum
import threading
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from horde_sdk.ai_horde_api.apimodels.generate.pop import ImageGenerateJobPopPayload

    from horde_worker_regen.process_management.process_info import HordeProcessInfo
    from horde_worker_regen.process_management.run_metrics import JobMetricsRecord
    from horde_worker_regen.process_management.system_memory import SystemMemorySummary

SUPERVISOR_PROTOCOL_VERSION = 7
"""Bumped when the snapshot/command schema changes incompatibly; the TUI checks it on connect.

v2 added per-process ``num_jobs_completed`` and the snapshot's worker-details maintenance/paused and
last-pop no-jobs/skip-reason fields.
v3 added alchemy config/runtime fields, ``pending_jobs``, ``JobFeatureSummary``, and extended
``RecentJobRecord`` with model name and feature data.
v4 added the lightweight :class:`WorkerLivenessFrame`, emitted on its own cadence so the supervisor
can judge worker liveness independently of full-snapshot production.
v5 added the resolved model ``baseline`` to ``JobQueueEntry`` and ``RecentJobRecord`` so the queue and
recent-jobs tables can show a model's baseline alongside its name.
v6 added the snapshot's ``system_memory`` field (:class:`SystemMemorySnapshot`): total/available RAM and
the worker's per-role RSS share.
v7 added ``DownloadStatusSnapshot.active`` (the full set of concurrent in-flight downloads) for
host-parallel downloading; ``current`` is retained as the primary entry for single-download consumers.
"""

RECENT_JOBS_IN_SNAPSHOT = 25
"""How many of the most recent finished-job records to carry in a snapshot (bounds payload size)."""

PENDING_JOBS_IN_SNAPSHOT = 8
"""How many pending-inference jobs to carry in a snapshot (bounds payload size)."""


class JobFeatureSummary(BaseModel):
    """The notable features of one image-generation job, for compact display."""

    loras: int = 0
    tis: int = 0
    control_type: str | None = None
    post_processing: list[str] = Field(default_factory=list)
    hires_fix: bool = False
    workflow: str | None = None

    @classmethod
    def from_payload(cls, payload: ImageGenerateJobPopPayload) -> JobFeatureSummary:
        """Summarize the notable features of an image-generation job payload."""
        return cls(
            loras=len(payload.loras) if payload.loras else 0,
            tis=len(payload.tis) if payload.tis else 0,
            control_type=str(payload.control_type) if payload.control_type else None,
            post_processing=[str(post_proc_step) for post_proc_step in payload.post_processing],
            hires_fix=payload.hires_fix,
            workflow=str(payload.workflow) if payload.workflow else None,
        )

    def as_tags(self) -> list[str]:
        """Compact display tags (e.g. ``['2×LoRA', 'canny', 'HiRes']``)."""
        tags: list[str] = []
        if self.loras:
            tags.append(f"{self.loras}×LoRA")
        if self.tis:
            tags.append(f"{self.tis}×TI")
        if self.control_type:
            tags.append(self.control_type)
        for post_proc_step in self.post_processing:
            tags.append(post_proc_step)
        if self.hires_fix:
            tags.append("HiRes")
        if self.workflow:
            tags.append(f"wf:{self.workflow}")
        return tags

    def is_empty(self) -> bool:
        """True when no notable features are present."""
        return not (
            self.loras or self.tis or self.control_type or self.post_processing or self.hires_fix or self.workflow
        )


class JobQueueEntry(BaseModel):
    """A pending-inference job, for the queue-preview on the overview screen."""

    job_id: str
    model: str
    baseline: str | None = None
    """The model's image baseline (e.g. ``stable_diffusion_xl``); None when it could not be resolved."""
    steps: int | None = None
    width: int | None = None
    height: int | None = None
    features: JobFeatureSummary | None = None


class WorkerConfigSummary(BaseModel):
    """The operationally-relevant bridge-data fields the overview/worker panels display.

    This is a compact projection of ``reGenBridgeData``; the config *editor* reads the full
    ``bridgeData.yaml`` directly, so the snapshot only carries what the dashboards render.
    """

    dreamer_name: str
    worker_version: str
    horde_username: str | None = None
    num_models: int = 0
    custom_models: bool = False
    max_power: int = 8
    max_threads: int = 1
    queue_size: int = 1
    max_batch: int = 1
    safety_on_gpu: bool = False
    allow_img2img: bool = True
    allow_lora: bool = False
    effective_allow_lora: bool | None = None
    """Whether job pops currently advertise LoRA support; None means same as ``allow_lora``."""
    allow_controlnet: bool = False
    allow_sdxl_controlnet: bool = False
    allow_post_processing: bool = True
    high_performance_mode: bool = False
    moderate_performance_mode: bool = False
    extra_slow_worker: bool = False

    alchemist: bool = False
    alchemy_concurrent: bool = True
    alchemy_max_concurrency: int = 1
    alchemy_vram_headroom_mb: int = 2000
    alchemy_caption_enabled: bool = False
    alchemy_forms: list[str] = Field(default_factory=list)


class ProcessSnapshot(BaseModel):
    """A serializable projection of one child process's live state (from ``HordeProcessInfo``)."""

    process_id: int
    process_type: str
    """The ``HordeProcessType`` name (e.g. ``INFERENCE`` / ``SAFETY``)."""
    last_process_state: str
    """The ``HordeProcessState`` name (e.g. ``WAITING_FOR_JOB`` / ``INFERENCE_STARTING``)."""
    is_alive: bool
    is_busy: bool

    loaded_horde_model_name: str | None = None
    loaded_horde_model_baseline: str | None = None
    current_job_id: str | None = None

    last_heartbeat_timestamp: float = 0.0
    last_heartbeat_delta: float = 0.0
    last_heartbeat_type: str = "OTHER"
    heartbeats_inference_steps: int = 0
    last_heartbeat_percent_complete: int | None = None

    ram_usage_bytes: int = 0
    vram_usage_mb: int = 0
    total_vram_mb: int = 0
    batch_amount: int = 1

    current_job_width: int | None = None
    current_job_height: int | None = None
    current_job_steps: int | None = None
    """The active job's resolution and step count (None when idle); surfaced as ``W×H`` / steps."""

    last_iterations_per_second: float | None = None
    last_current_step: int | None = None
    last_total_steps: int | None = None

    vram_used_high_water_mb: int = 0
    ram_used_high_water_mb: int = 0

    num_jobs_completed: int = 0
    """Jobs/forms this slot has finished (inference, safety check, or alchemy form); resets on replace."""

    current_job_features: JobFeatureSummary | None = None
    """Notable features of the active job (LoRAs, ControlNet, etc.); None when idle."""

    @classmethod
    def from_process_info(cls, info: HordeProcessInfo) -> ProcessSnapshot:
        """Build a snapshot from a live ``HordeProcessInfo`` (read-only; no import coupling)."""
        job = info.last_job_referenced
        current_job_id = str(job.id_.root) if job is not None and job.id_ is not None else None
        baseline = info.loaded_horde_model_baseline

        features: JobFeatureSummary | None = None
        current_job_width: int | None = None
        current_job_height: int | None = None
        current_job_steps: int | None = None
        if info.is_process_busy() and job is not None:
            candidate = JobFeatureSummary.from_payload(job.payload)
            if not candidate.is_empty():
                features = candidate
            current_job_width = job.payload.width
            current_job_height = job.payload.height
            current_job_steps = job.payload.ddim_steps

        return cls(
            process_id=info.process_id,
            process_type=info.process_type.name,
            last_process_state=info.last_process_state.name,
            is_alive=info.is_process_alive(),
            is_busy=info.is_process_busy(),
            loaded_horde_model_name=info.loaded_horde_model_name,
            loaded_horde_model_baseline=str(baseline) if baseline is not None else None,
            current_job_id=current_job_id,
            last_heartbeat_timestamp=info.last_heartbeat_timestamp,
            last_heartbeat_delta=info.last_heartbeat_delta,
            last_heartbeat_type=info.last_heartbeat_type.name,
            heartbeats_inference_steps=info.heartbeats_inference_steps,
            last_heartbeat_percent_complete=info.last_heartbeat_percent_complete,
            ram_usage_bytes=info.ram_usage_bytes,
            vram_usage_mb=info.vram_usage_mb,
            total_vram_mb=info.total_vram_mb,
            batch_amount=info.batch_amount,
            current_job_width=current_job_width,
            current_job_height=current_job_height,
            current_job_steps=current_job_steps,
            last_iterations_per_second=info.last_iterations_per_second,
            last_current_step=info.last_current_step,
            last_total_steps=info.last_total_steps,
            vram_used_high_water_mb=info.vram_used_high_water_mb,
            ram_used_high_water_mb=info.ram_used_high_water_mb,
            num_jobs_completed=info.num_jobs_completed,
            current_job_features=features,
        )


class RecentJobRecord(BaseModel):
    """A lean projection of one finished job, for the insights/recent-activity views.

    Deliberately a local model (not ``run_metrics.JobMetricsRecord``) so this module, imported by
    the TUI and the fake worker, stays free of the heavy ``horde_sdk``/logfire import chain.
    """

    job_id: str
    is_alchemy: bool = False
    faulted: bool = False
    queue_wait_seconds: float | None = None
    e2e_seconds: float | None = None
    safety_seconds: float | None = None
    model_name: str | None = None
    baseline: str | None = None
    """The model's image baseline (e.g. ``stable_diffusion_xl``); None when it could not be resolved."""
    steps: int | None = None
    width: int | None = None
    height: int | None = None
    features: JobFeatureSummary | None = None

    @classmethod
    def from_metrics_record(cls, record: JobMetricsRecord, baseline: str | None = None) -> RecentJobRecord:
        """Project a worker-side ``JobMetricsRecord`` into the lean wire form.

        ``baseline`` is resolved by the caller (the process manager, which owns the model metadata) since
        ``JobMetricsRecord`` does not itself carry one.
        """
        features: JobFeatureSummary | None = None
        has_features = bool(
            record.loras_count
            or record.tis_count
            or record.control_type
            or record.post_processing
            or record.hires_fix,
        )
        if has_features:
            features = JobFeatureSummary(
                loras=record.loras_count,
                tis=record.tis_count,
                control_type=record.control_type,
                post_processing=record.post_processing,
                hires_fix=record.hires_fix,
            )
        return cls(
            job_id=record.job_id,
            is_alchemy=record.is_alchemy,
            faulted=record.faulted,
            queue_wait_seconds=record.queue_wait_seconds,
            e2e_seconds=record.e2e_seconds,
            safety_seconds=record.safety_seconds,
            model_name=record.model_name,
            baseline=baseline,
            steps=record.steps,
            width=record.width,
            height=record.height,
            features=features,
        )


class DownloadPhase(enum.StrEnum):
    """What the background download process is doing right now."""

    INITIALIZING = "initializing"
    """Loading model managers / fetching the model reference (a network call on first run)."""
    SCANNING = "scanning"
    """Verifying which configured models are already on disk (may hash large files on first run)."""
    DOWNLOADING = "downloading"
    """Actively downloading one or more models."""
    IDLE = "idle"
    """Nothing queued; all requested models are present (or were skipped)."""
    PAUSED = "paused"
    """Downloads are paused by the operator; queued work is held."""
    ERROR = "error"
    """The download subsystem hit an unrecoverable error (see ``error_message``)."""


class DownloadItem(BaseModel):
    """A queued (not yet started) download, labelled with the feature that needs it."""

    model_name: str
    feature: str
    """Human label for why this downloads (e.g. 'image model', 'LoRa', 'ControlNet annotators')."""
    target_dir: str | None = None
    """Where the file(s) will be written on disk."""
    size_bytes: int | None = None


class CurrentDownloadStatus(BaseModel):
    """The download in progress right now, with live progress."""

    model_name: str
    feature: str
    target_dir: str
    host: str | None = None
    """The source hostname this is downloading from (e.g. ``civitai.com``); None when not tracked."""
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_bps: float | None = None
    eta_seconds: float | None = None

    @property
    def percent(self) -> float | None:
        """Completion as a 0-100 percentage, or None when the total size is unknown."""
        if self.total_bytes <= 0:
            return None
        return min(100.0, self.downloaded_bytes / self.total_bytes * 100.0)


class DownloadFailure(BaseModel):
    """A download that was attempted and failed, with a human-readable reason."""

    model_name: str
    feature: str
    reason: str


class DownloadStatusSnapshot(BaseModel):
    """The full state of the download subsystem, for the TUI/console downloads view."""

    phase: DownloadPhase = DownloadPhase.IDLE
    current: CurrentDownloadStatus | None = None
    """The primary in-flight download (``active[0]`` when any), kept for back-compat with single-download
    consumers; prefer :attr:`active` for the full set when downloads run in parallel."""
    active: list[CurrentDownloadStatus] = Field(default_factory=list)
    """Every download in flight right now (one per executor thread); empty when idle. Parallel downloads
    target distinct hosts by default, so this typically lists one entry per active host."""
    pending: list[DownloadItem] = Field(default_factory=list)
    failures: list[DownloadFailure] = Field(default_factory=list)
    present_model_names: list[str] = Field(default_factory=list)
    paused: bool = False
    rate_limit_kbps: int | None = None
    error_message: str | None = None


class DownloadPlanSummary(BaseModel):
    """The disk implications of the current model config (computed once via model_download_plan)."""

    present_bytes: int = 0
    to_download_bytes: int = 0
    total_bytes: int = 0
    free_disk_bytes: int | None = None
    fits: bool = True
    shortfall_bytes: int = 0
    num_present: int = 0
    num_to_download: int = 0
    sizes_complete: bool = True
    """False when some configured models lack size metadata, so the byte totals are a lower bound."""


class WorkerLivenessFrame(BaseModel):
    """A tiny liveness heartbeat, emitted on its own cadence independent of full-snapshot production.

    The supervisor judges worker responsiveness from :attr:`loop_alive_wall_time` (the wall-clock time
    the worker's control loop last started a tick) rather than from the age of the last full
    :class:`WorkerStateSnapshot`. This decouples "is the loop making progress?" from "what is the rich
    state?": a snapshot that briefly fails to build (or is coalesced away) no longer looks like a stall,
    while a genuinely wedged loop reports an accurate, growing liveness age.
    """

    protocol_version: int = SUPERVISOR_PROTOCOL_VERSION
    loop_alive_wall_time: float = 0.0
    """Worker wall-clock time (``time.time()``) the control loop last began a tick."""


class WholeCardResidencyStatus(BaseModel):
    """Whole-card exclusive-residency posture: whether it can engage, and live detail when it has.

    Whole-card residency stops idle sibling inference processes (and may move the safety process off-GPU)
    to give a very heavy model sole use of the device, so its weights stay resident instead of streaming
    from host RAM. From the dashboard that looks like processes vanishing for no reason; these fields let
    the TUI explain it. ``possible`` is the config/topology heads-up (it *could* engage); the rest is the
    live detail of a residency currently held. MB figures are device VRAM rounded to whole MB.
    """

    possible: bool = False
    """The feature could engage under the current config and process topology (operator heads-up)."""
    enabled: bool = False
    """The ``whole_card_exclusive_residency`` config flag is on."""
    safety_off_gpu_enabled: bool = False
    """A whole-card job would also move the safety process off-GPU (config + safety on-GPU)."""
    cooldown_seconds: int = 0
    """Configured seconds a residency is held after its last heavy job drains, before restoring."""
    per_process_overhead_mb: int = 0
    """Per-process CUDA-context VRAM the forecast assumes (configured override, else startup-measured)."""
    total_vram_mb: int = 0
    """Device total VRAM (MB), 0 before any process has reported."""

    active: bool = False
    """A whole-card residency is currently held."""
    model: str | None = None
    """The model holding sole residency, when active."""
    phase: str = ""
    """``establishing`` (siblings stopping, model loading) or ``holding`` (serving) while active."""
    safety_paused: bool = False
    """The safety process is currently paused off-GPU for this residency."""
    processes_now: int = 0
    """Loaded inference processes right now (after any teardown)."""
    processes_target: int = 0
    """Inference processes the residency targets (the forecast's max-resident count)."""
    processes_max: int = 0
    """The normal inference-process ceiling, so the paused count is ``processes_max - processes_now``."""
    cooldown_remaining_seconds: float | None = None
    """Seconds left before the residency restores once its jobs drained, or None when not active."""

    weights_mb: int | None = None
    """Resident weight footprint of the residency model (detail view)."""
    reserve_mb: int | None = None
    """Free-VRAM headroom the forecast required, the activation working set (detail view)."""
    free_now_mb: int | None = None
    """Measured device-wide free VRAM at establishment (detail view)."""
    free_if_alone_mb: int | None = None
    """Free VRAM achievable with sole residency (detail view)."""
    max_resident_processes: int | None = None
    """The forecast's largest co-resident process count that still avoids streaming."""


class SystemMemorySnapshot(BaseModel):
    """The system-RAM picture and the worker's per-role share of it, for the supervisor/TUI.

    The wire projection of :class:`~horde_worker_regen.process_management.system_memory.SystemMemorySummary`.
    Only the raw figures travel; the derived breakdown (used, other, fractions) is recomputed on the
    receiving side via :meth:`to_summary` so the math lives in exactly one place.
    """

    total_bytes: int = 0
    """Total physical RAM on the machine."""
    available_bytes: int = 0
    """RAM the OS reports as available without paging."""
    worker_rss_by_role: dict[str, int] = Field(default_factory=dict)
    """Per-role resident-set sizes (bytes) for the worker's own processes (see the ``ROLE_*`` keys)."""

    @classmethod
    def from_summary(cls, summary: SystemMemorySummary) -> SystemMemorySnapshot:
        """Project a worker-side :class:`SystemMemorySummary` onto the wire model."""
        return cls(
            total_bytes=summary.total_bytes,
            available_bytes=summary.available_bytes,
            worker_rss_by_role=dict(summary.worker_rss_by_role),
        )

    def to_summary(self) -> SystemMemorySummary:
        """Rebuild a :class:`SystemMemorySummary` (with its derived properties) from the wire fields."""
        from horde_worker_regen.process_management.system_memory import build_system_memory_summary

        return build_system_memory_summary(
            total_bytes=self.total_bytes,
            available_bytes=self.available_bytes,
            worker_rss_by_role=self.worker_rss_by_role,
        )


class WorkerStateSnapshot(BaseModel):
    """One frame of worker state pushed from the worker to its supervisor over the pipe.

    Carries the same headline information the console ``StatusReporter`` assembles, plus the
    per-process detail the live view renders. Payload size is bounded: ``recent_jobs`` is capped
    at :data:`RECENT_JOBS_IN_SNAPSHOT`.
    """

    protocol_version: int = SUPERVISOR_PROTOCOL_VERSION
    timestamp: float = Field(default_factory=time.time)
    session_start_time: float = 0.0

    shutting_down: bool = False
    maintenance_mode: bool = False
    """Maintenance: the local pop loop hit maintenance, the operator paused the worker locally, or the
    worker self-throttled (see ``self_throttle_paused``)."""
    self_throttle_paused: bool = False
    """The worker paused popping itself: resource/OOM faults accumulated fast enough that it backed off to
    avoid the horde forcing maintenance for "dropping too many jobs"."""
    worker_details_maintenance: bool = False
    """The horde's worker-details API reports this worker in maintenance (polled, advisory)."""
    worker_details_paused: bool = False
    """The horde's worker-details API reports this worker paused (polled, advisory)."""
    too_many_consecutive_failed_jobs: bool = False

    # Connectivity / health signals the worker already tracks (surfaced for the status monitor).
    worker_registered: bool = False
    """Whether the AI Horde API has returned this worker's details at least once (it is known)."""
    user_info_failed: bool = False
    """The most recent user-details API call failed; the clearest API/network-reachability signal."""
    user_info_failed_reason: str | None = None
    """A short description of the last user-info failure, when one occurred."""
    in_error_backoff: bool = False
    """The job-pop throttler is backing off after repeated pop failures (server/network trouble)."""
    consecutive_failed_jobs: int = 0
    """How many jobs have failed back-to-back (resets on a success)."""
    seconds_since_last_pop: float | None = None
    """Seconds since the worker last successfully popped a job (None if it never has)."""
    last_pop_no_jobs_available: bool = False
    """The most recent successful pop returned no job (a short-term 'no work right now' signal)."""
    last_pop_skipped_reasons: dict[str, int] = Field(default_factory=dict)
    """Why the last 'no job available' pop skipped work, per reason (models/nsfw/max_pixels/...)."""
    api_messages: list[str] = Field(default_factory=list)
    """Operator/maintenance messages delivered by the horde in pop responses."""

    config: WorkerConfigSummary
    processes: list[ProcessSnapshot] = Field(default_factory=list)

    # Headline job counters (from the job tracker / submitter / popper).
    num_jobs_popped: int = 0
    num_jobs_submitted: int = 0
    num_jobs_faulted: int = 0
    num_job_slowdowns: int = 0
    num_process_recoveries: int = 0
    pending_megapixelsteps: int = 0
    jobs_pending_inference: int = 0
    jobs_in_progress: int = 0
    jobs_pending_safety_check: int = 0
    jobs_being_safety_checked: int = 0
    jobs_pending_submit: int = 0
    """Jobs that have cleared safety and are queued for API submission (the pipeline's tail stage)."""
    time_spent_no_jobs_available: float = 0.0

    kudos_per_hour: float | None = None
    kudos_this_session: float | None = None

    active_models: list[str] = Field(default_factory=list)

    gpu_utilization_mean_percent: float | None = None
    gpu_utilization_busy_fraction: float | None = None
    gpu_utilization_samples: int = 0
    """How many GPU-utilization samples back the figures above (0 = unmeasured, e.g. no NVML)."""

    vram_high_water_mb_per_process: dict[int, int] = Field(default_factory=dict)
    ram_high_water_mb_per_process: dict[int, int] = Field(default_factory=dict)
    disk_free_bytes: dict[str, int] = Field(default_factory=dict)

    system_memory: SystemMemorySnapshot | None = None
    """Live system-RAM total/available and the worker's per-role RSS share (None before first sample)."""

    downloads: DownloadStatusSnapshot | None = None
    """Live download-subsystem state (None when background downloads are disabled)."""
    download_plan: DownloadPlanSummary | None = None
    """The one-time disk implications of the configured models (None when not computed)."""
    lora_pops_blocked_by_downloads: bool = False
    """Configured LoRA support is temporarily suppressed because background downloads are active."""
    lora_pops_blocked_by_disk: bool = False
    """Configured LoRA support is suppressed because the LoRA cache volume is below its free-space floor.

    Unlike the transient download block, this persists until disk space recovers; the TUI surfaces it
    prominently because, left unaddressed, it stops the worker from serving any LoRA jobs."""

    recent_jobs: list[RecentJobRecord] = Field(default_factory=list)
    """The most recent finished-job records, newest last (capped)."""

    alchemy_forms_pending: int = 0
    """Forms popped from the API but not yet dispatched to a child process."""
    alchemy_forms_in_flight: int = 0
    """Forms dispatched to a child process, awaiting a result message."""
    alchemy_forms_awaiting_submit: int = 0
    """Forms with a completed result, waiting for API submission."""
    alchemy_total_submitted: int = 0
    """Cumulative forms successfully submitted this session."""
    alchemy_total_faulted: int = 0
    """Cumulative forms that faulted (permanently failed) this session."""

    pending_jobs: list[JobQueueEntry] = Field(default_factory=list)
    """Pending-inference jobs (capped at :data:`PENDING_JOBS_IN_SNAPSHOT`), oldest first."""

    whole_card_residency: WholeCardResidencyStatus = Field(default_factory=WholeCardResidencyStatus)
    """Whole-card exclusive-residency posture: whether it can engage, and live detail when it has."""


class SupervisorCommand(enum.Enum):
    """A control action the supervisor asks the worker to take, drained each loop tick."""

    PAUSE = enum.auto()
    """Enter maintenance mode: stop popping new jobs (in-flight jobs finish)."""
    RESUME = enum.auto()
    """Leave maintenance mode and resume popping jobs."""
    DRAIN = enum.auto()
    """Stop popping and let in-flight jobs finish without exiting (alias of PAUSE for now)."""
    RESTART_PROCESS = enum.auto()
    """Replace one inference process by ``process_id`` (e.g. a stuck slot)."""
    RELOAD_CONFIG = enum.auto()
    """Re-read ``bridgeData.yaml`` from disk and hot-swap the runtime config."""
    SET_CONCURRENCY = enum.auto()
    """Adjust the live inference concurrency: thread cap and/or running process count."""
    PAUSE_DOWNLOADS = enum.auto()
    """Hold background model downloads (the current chunk loop blocks) until resumed."""
    RESUME_DOWNLOADS = enum.auto()
    """Resume held background model downloads."""
    SET_DOWNLOAD_RATE_LIMIT = enum.auto()
    """Set the background-download bandwidth cap in KB/s (0 or None clears the cap)."""
    DOWNLOADS_ONLY_HOLD = enum.auto()
    """Hold the worker in a download-only posture: keep the download process (and reference refresh)
    running but do not start inference/safety or pop jobs. Lets the operator pre-fetch models without
    committing the GPU. Lifted by :attr:`GO_LIVE`."""
    GO_LIVE = enum.auto()
    """Leave the download-only hold and bring the worker fully up (inference/safety start once a model is
    present, popping resumes). In-flight downloads continue; the present-set gate keeps serving safe."""
    DOWNLOAD_MODELS = enum.auto()
    """Fetch a chosen set of models on demand: the selected image models (and optionally the aux pass),
    enqueued into the background download process without changing config. Drives the TUI download picker."""
    SET_SERVER_MAINTENANCE = enum.auto()
    """Set the worker's *server-side* maintenance flag on the horde (``server_maintenance_enabled``).

    Distinct from :attr:`PAUSE`/:attr:`RESUME` (the local pop-pause): this calls the horde API so the
    horde itself stops (or resumes) sending the worker jobs, matching the maintenance the job-pop response
    reports."""
    SHUTDOWN = enum.auto()
    """Begin a graceful, timed shutdown of the worker."""


class SupervisorControlMessage(BaseModel):
    """A command sent from the supervisor to the worker over the pipe."""

    command: SupervisorCommand
    process_id: int | None = None
    """The target process slot, required for :attr:`SupervisorCommand.RESTART_PROCESS`."""
    target_threads: int | None = None
    """The new concurrent-inference cap, for :attr:`SupervisorCommand.SET_CONCURRENCY` (clamped to \
    the session ceiling)."""
    target_processes: int | None = None
    """The new running inference-process count, for :attr:`SupervisorCommand.SET_CONCURRENCY`."""
    download_rate_limit_kbps: int | None = None
    """The new download bandwidth cap in KB/s, for :attr:`SupervisorCommand.SET_DOWNLOAD_RATE_LIMIT`."""
    server_maintenance_enabled: bool | None = None
    """The desired server-side maintenance state, for :attr:`SupervisorCommand.SET_SERVER_MAINTENANCE`."""
    download_model_names: list[str] = Field(default_factory=list)
    """The image models to fetch on demand, for :attr:`SupervisorCommand.DOWNLOAD_MODELS`."""
    download_include_aux: bool = False
    """Whether a :attr:`SupervisorCommand.DOWNLOAD_MODELS` request should also run the aux/default pass."""


class SupervisorChannel:
    """The worker's end of the supervisor pipe, designed so the consumer can never stall the worker.

    Snapshots are sent on a daemon thread from a single latest-only slot: :meth:`send_snapshot` only
    updates the slot and returns immediately, so the worker's control loop never blocks on a slow or
    hung supervisor (it just sends the freshest state once the consumer catches up). :meth:`drain_commands`
    is non-blocking and skips a tick rather than wait if the sender thread is mid-send. A dead pipe is
    caught and marks the channel closed.

    Send and receive on the one duplex connection are serialized by a lock, so the sender thread and
    the control loop never touch the connection concurrently.
    """

    _LIVENESS_INTERVAL = 1.0
    """How often the sender thread emits a :class:`WorkerLivenessFrame` (seconds)."""

    def __init__(self, connection: Connection) -> None:
        """Wrap the worker-side pipe connection and start the snapshot sender thread."""
        self._connection = connection
        self._lock = threading.Lock()
        self._closed = False
        self._latest: WorkerStateSnapshot | None = None
        self._pending = threading.Event()
        self._stop = threading.Event()
        self._loop_alive_wall_time = time.time()
        """Updated by :meth:`note_alive` from the control loop; read by the sender thread. A single
        float shared one-writer/one-reader (atomic under the GIL), so no lock is needed; a stale read
        only means a slightly older liveness timestamp, which is harmless."""
        self._last_liveness_monotonic = 0.0
        self._sender = threading.Thread(
            target=self._send_loop,
            name="supervisor-snapshot-sender",
            daemon=True,
        )
        self._sender.start()

    def note_alive(self) -> None:
        """Record that the control loop just advanced (call once per tick). Never blocks."""
        self._loop_alive_wall_time = time.time()

    def send_snapshot(self, snapshot: WorkerStateSnapshot) -> bool:
        """Hand the latest snapshot to the sender thread. Never blocks; returns False once closed."""
        if self._closed:
            return False
        self._latest = snapshot
        self._pending.set()
        return True

    def _send_loop(self) -> None:
        """Send the freshest snapshot when one is pending and emit liveness frames on their own cadence.

        Either send may block on a slow consumer; that is fine, because this runs on a daemon thread and
        never touches the worker's control loop.
        """
        while not self._stop.is_set():
            got_pending = self._pending.wait(timeout=self._LIVENESS_INTERVAL)

            now = time.monotonic()
            if now - self._last_liveness_monotonic >= self._LIVENESS_INTERVAL:
                self._last_liveness_monotonic = now
                if not self._send_frame(WorkerLivenessFrame(loop_alive_wall_time=self._loop_alive_wall_time)):
                    return

            if not got_pending:
                continue
            self._pending.clear()
            snapshot = self._latest
            if snapshot is None:
                continue
            if not self._send_frame(snapshot):
                return

    def _send_frame(self, frame: WorkerStateSnapshot | WorkerLivenessFrame) -> bool:
        """Send one frame under the connection lock. Returns False once the channel is closed/dead."""
        with self._lock:
            if self._closed:
                return False
            try:
                self._connection.send(frame)
            except Exception:
                self._closed = True
                return False
        return True

    def drain_commands(self) -> list[SupervisorControlMessage]:
        """Return all control messages currently waiting, without blocking (skips if a send is in progress)."""
        if self._closed:
            return []
        commands: list[SupervisorControlMessage] = []
        if not self._lock.acquire(blocking=False):
            return commands
        try:
            while self._connection.poll():
                message = self._connection.recv()
                if isinstance(message, SupervisorControlMessage):
                    commands.append(message)
        except (EOFError, OSError):
            self._closed = True
        finally:
            self._lock.release()
        return commands

    def close(self) -> None:
        """Stop the sender thread (the connection itself is owned by the caller)."""
        self._stop.set()
        self._pending.set()

    @property
    def closed(self) -> bool:
        """Whether the channel has encountered an unrecoverable pipe error."""
        return self._closed
