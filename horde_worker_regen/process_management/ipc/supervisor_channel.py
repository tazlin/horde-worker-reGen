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

from horde_worker_regen.process_management.models.feature_readiness import FeatureReadiness

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from horde_sdk.ai_horde_api.apimodels.generate.pop import ImageGenerateJobPopPayload

    from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
    from horde_worker_regen.process_management.resources.run_metrics import JobMetricsRecord
    from horde_worker_regen.process_management.resources.system_memory import SystemMemorySummary

SUPERVISOR_PROTOCOL_VERSION = 18
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
v8 added per-card multi-GPU data: ``ProcessSnapshot.device_index`` (which card each slot is pinned to)
and the snapshot's ``per_card`` list of :class:`CardSnapshot` (per-card VRAM, contexts, residency, and
fault/unservable-model health). Additive: a single-GPU host reports exactly one ``CardSnapshot``.
v9 added the snapshot's ``feature_readiness`` (:class:`FeatureReadinessSummary`): the per-feature
deps+on-disk readiness the worker uses to decide which gated features it offers to the Horde.
v10 added ``orchestration_intent`` and ``work_ledger`` so the Overview can show the worker's current
decision and promote per-job state out of the process table.
v11 added worker-owned stats samples/history, model/baseline rollups, and the stats JSONL export control.
v12 added ``pop_governors`` (:class:`PopGovernorsSnapshot`): the live + session-aggregate state of every
pop/scheduling governor holding back or reshaping job pops, for the Overview strip and the stats tab.
v14 added :class:`WorkerFatalConfigError`, a one-shot frame the worker sends before exiting on a fatal,
non-retryable configuration problem (e.g. a worker name taken by another account) so the supervisor can
stop relaunching and the dashboard can show the specific reason and remedy instead of a generic crash.
v16 added post-processing lane counters and per-entry queue order so the TUI can show the image job route
through the dedicated post-processing process.
v17 added the operator-facing reason post-processing was session-disabled.
"""

RECENT_JOBS_IN_SNAPSHOT = 25
"""How many of the most recent finished-job records to carry in a snapshot (bounds payload size)."""

PENDING_JOBS_IN_SNAPSHOT = 8
"""How many pending-inference jobs to carry in a snapshot (bounds payload size)."""

WORK_LEDGER_ENTRIES_IN_SNAPSHOT = 18
"""How many active/recent job rows to carry for the Overview work ledger (bounds payload size)."""


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
    queue_order: int | None = None
    """1-based order among currently tracked image jobs by pop order; None for non-image forms."""


class WorkLedgerStage(enum.StrEnum):
    """A job's operator-facing stage in the Overview work ledger."""

    QUEUED = "queued"
    PREPARING = "preparing"
    INFERENCE = "inference"
    POST_PROCESSING = "post_processing"
    SAFETY = "safety"
    SUBMIT = "submit"
    COMPLETED = "completed"
    FAULTED = "faulted"


class WorkLedgerEntry(BaseModel):
    """One active or recently-finished job row for the Overview work ledger."""

    job_id: str
    stage: WorkLedgerStage
    model: str | None = None
    baseline: str | None = None
    process_id: int | None = None
    device_index: int | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    iterations_per_second: float | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    features: JobFeatureSummary | None = None
    queue_order: int | None = None
    """1-based order among currently tracked image jobs by pop order; None for recent/alchemy rows."""
    age_seconds: float | None = None
    queue_wait_seconds: float | None = None
    safety_seconds: float | None = None
    e2e_seconds: float | None = None
    intent: str | None = None
    raw_reason: str | None = None
    faulted: bool = False


class OrchestrationIntentSnapshot(BaseModel):
    """The scheduler/popper's current plain-English intent for the Overview Now/Next/Why strip."""

    summary: str = "Waiting for worker state."
    next_action: str | None = None
    why: str | None = None
    raw_gate: str | None = None
    target_job_id: str | None = None
    target_model: str | None = None
    target_process_id: int | None = None
    target_device_index: int | None = None
    updated_at: float = Field(default_factory=time.time)


class WorkerConfigSummary(BaseModel):
    """The operationally-relevant bridge-data fields the overview/worker panels display.

    This is a compact projection of ``reGenBridgeData``; the config *editor* reads the full
    ``bridgeData.yaml`` directly, so the snapshot only carries what the dashboards render.
    """

    dreamer_name: str
    alchemist_name: str | None = None
    """The worker's alchemist identity, shown in place of the dreamer name on an alchemist-only worker."""
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
    device_index: int = 0
    """The stable index of the GPU this slot is pinned to (0 on a single-GPU host).

    Lets the dashboard group a card's slots together and show a per-process GPU column on multi-GPU hosts."""
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
            device_index=info.device_index,
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


class StatsSample(BaseModel):
    """One lightweight, worker-owned statistics sample for trend history and JSONL export."""

    timestamp: float = Field(default_factory=time.time)
    jobs_submitted: int = 0
    jobs_faulted: int = 0
    kudos_per_hour: float | None = None
    kudos_this_session: float | None = None
    """Cumulative kudos earned this session at sample time, for windowed kudos/hr deltas."""
    eligible_seconds_total: float = 0.0
    """Cumulative productive (pipeline-occupied) seconds at sample time; the kudos/hr denominator."""
    gpu_duty_percent: float | None = None
    gpu_busy_fraction: float | None = None
    pending_megapixelsteps: int = 0
    jobs_pending_inference: int = 0
    jobs_in_progress: int = 0
    jobs_pending_safety_check: int = 0
    jobs_being_safety_checked: int = 0
    jobs_pending_post_processing: int = 0
    jobs_being_post_processed: int = 0
    jobs_pending_submit: int = 0
    time_spent_no_jobs_available: float = 0.0
    num_process_recoveries: int = 0
    num_job_slowdowns: int = 0
    alchemy_forms_pending: int = 0
    alchemy_forms_in_flight: int = 0
    alchemy_forms_awaiting_submit: int = 0
    alchemy_total_submitted: int = 0
    alchemy_total_faulted: int = 0
    process_state_summary: str = ""
    """Compact per-process state line for offline duty-cycle attribution."""
    orchestration_intent_summary: str = ""
    """The scheduler/popper's current high-level intent at sample time."""
    orchestration_next_action: str | None = None
    """The next planned orchestration action, when known."""
    orchestration_why: str | None = None
    """Human-readable reason for the current orchestration decision."""
    orchestration_raw_gate: str | None = None
    """Raw gate/blocking reason behind the orchestration decision, when available."""
    maintenance_mode: bool = False
    self_throttle_paused: bool = False
    supervisor_paused: bool = False
    last_pop_maintenance_mode: bool = False
    worker_details_maintenance: bool = False
    in_error_backoff: bool = False
    last_pop_no_jobs_available: bool = False
    last_pop_skipped_reasons: dict[str, int] = Field(default_factory=dict)
    churn_counts: dict[str, int] = Field(default_factory=dict)
    """Cumulative reload/respawn churn counts by kind at sample time."""
    slot_duty_totals: dict[str, float] = Field(default_factory=dict)
    """Cumulative slot-seconds per slot-duty bucket (sampling vs each empty-slot attribution) at sample
    time. Monotonically growing; consumers difference two samples for a window's capacity-normalized
    active/idle/gated breakdown."""
    slot_duty_capacity: int = 0
    """Configured concurrent-inference slot count the slot-duty totals are normalized against."""
    dispatch_hold_bucket: str | None = None
    """The slot-duty bucket currently holding the next dispatch (None when dispatching or no work waits)."""


class StatsRollupRow(BaseModel):
    """Incremental rollup of finalized jobs by model/baseline (image) or by form (alchemy)."""

    model: str | None = None
    baseline: str | None = None
    jobs: int = 0
    megapixelsteps: float = 0.0
    sampling_seconds: float = 0.0
    e2e_seconds: float = 0.0
    batch_gt_one_jobs: int = 0
    faulted_jobs: int = 0
    """How many of the folded jobs/forms faulted (the by-form table surfaces this per form)."""
    vram_high_water_mb: int = 0
    """Peak VRAM high-water observed across the folded jobs/forms, when a child reported it (0 otherwise)."""


class StatsExportState(BaseModel):
    """Current worker-side JSONL export state."""

    enabled: bool = False
    active_file_path: str | None = None
    bytes_in_stats_files: int = 0
    warning_over_50_mib: bool = False
    last_write_error: str | None = None


class StatsHistoryBackfill(BaseModel):
    """Recent exact samples plus a decimated all-session series for reconnecting frontends."""

    recent_samples: list[StatsSample] = Field(default_factory=list)
    all_session_samples: list[StatsSample] = Field(default_factory=list)


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


FEATURE_LORA_ADHOC = "LoRa (job)"
"""The ``feature`` label a job-driven ad-hoc LoRA prefetch download carries."""
FEATURE_TI_ADHOC = "textual inversion (job)"
"""The ``feature`` label a job-driven ad-hoc textual-inversion prefetch download carries."""
ADHOC_PREFETCH_FEATURES = frozenset({FEATURE_LORA_ADHOC, FEATURE_TI_ADHOC})
"""Download-feature labels for the job-driven ad-hoc prefetch pipeline (LoRA/TI placed on disk at job pop).

These downloads are how a LoRA job becomes dispatchable, so they are excluded from the worker-wide
LoRA-advertising suppression (which only bulk/default seeding and image/aux fetches should trigger)."""


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


class FeatureInfoRow(BaseModel):
    """A read-only readiness line for a feature the worker does not gate on disk presence (LoRA, safety).

    These keep their own existing gating (per-job ad-hoc LoRA, the startup safety-model ensure); the row
    only surfaces their state alongside the gated features so the readiness table is a complete picture.
    """

    label: str
    status: str
    """A short human status, e.g. 'present', 'enabled', 'verifying downloads'."""
    ok: bool = True
    """Whether the line reads as healthy/ready (green) versus pending/blocked (muted), for styling."""


class FeatureReadinessSummary(BaseModel):
    """The worker's per-feature readiness: what it offers to the Horde and why anything is withheld.

    ``gated`` carries the features the worker withholds until their models/annotators are on disk
    (ControlNet, SDXL-ControlNet, post-processing); ``informational`` carries read-only lines for
    features with their own gating (LoRA, safety). Built parent-side from the same readiness the pop gate
    enforces, so the table can never disagree with what the worker actually advertises.
    """

    gated: list[FeatureReadiness] = Field(default_factory=list)
    informational: list[FeatureInfoRow] = Field(default_factory=list)


class WorkerFatalConfigError(BaseModel):
    """A one-shot frame the worker sends before exiting on a fatal, non-retryable config problem.

    Some misconfigurations can never succeed on a relaunch: a worker name taken by another account, a
    name still left at its reserved default, or the dreamer/alchemist names colliding. The worker
    detects these at startup (before spawning any children) and sends this frame so the supervisor stops
    burning its restart budget on a config that cannot work, and the dashboard shows the specific reason
    and remedy rather than a generic "crashed". ``detail`` carries the full human explanation, including
    how to fix it.
    """

    protocol_version: int = SUPERVISOR_PROTOCOL_VERSION
    title: str = "Worker configuration problem"
    """A short headline for the dashboard (e.g. "Worker name problem")."""
    detail: str = ""
    """The full operator-facing explanation and remedy."""


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


class PopGovernorStatus(BaseModel):
    """One pop/scheduling governor's live spell plus session aggregates, for the dashboard and tooling.

    A governor is any condition that holds back or reshapes job pops (whole-card residency, the large-model
    switch throttle and re-entry cooldown, post-inference backpressure, the unservable-model holdback, the
    consecutive-failure pause, pop error-backoff, a LoRA download backoff, model stickiness, the megapixelstep
    wait, the self-throttle). These fields let the TUI show *which* governor is engaged and *for how long*,
    and let the stats tab compare how much of the session each one consumed.
    """

    name: str
    """Stable machine key (snake_case), e.g. ``large_model_switch``."""
    label: str
    """Short human-friendly name for the dashboard."""
    active: bool = False
    """Whether the governor is engaged right now."""
    reason: str | None = None
    """Short human-readable cause for the current engagement, when active."""
    current_spell_seconds: float = 0.0
    """How long the current spell has been engaged (0 when idle)."""
    expected_remaining_seconds: float | None = None
    """Estimated seconds until release, or None when the governor has no fixed timer."""
    triggers: int = 0
    """How many times this governor has engaged this session."""
    total_active_seconds: float = 0.0
    """Aggregate engaged time this session (completed spells plus the live one)."""
    fraction_of_session: float = 0.0
    """``total_active_seconds`` as a fraction of the session length so far (0..1)."""


class PopGovernorsSnapshot(BaseModel):
    """The set of pop/scheduling governors with history or a live spell, plus a roll-up flag."""

    governors: list[PopGovernorStatus] = Field(default_factory=list)
    """Per-governor live + aggregate state, active first (see ``PopGovernorRegistry.views``)."""
    any_active: bool = False
    """Whether any governor is currently engaged (a quick "is the worker being held back" flag)."""


class RamGovernanceSnapshot(BaseModel):
    """The RAM governor's current posture as operator-visible state.

    This is the projection of the scheduler's per-cycle host-memory snapshot after the pure governor
    policy has run: measured RAM vs the danger floor, whether intake is held, and which reclaim remedies
    are currently active. It is intentionally compact so the dashboard can explain why pops or preloads
    are being held without exposing the whole internal process map.
    """

    measured: bool = False
    """Whether the governor has produced at least one RAM verdict this session."""
    under_pressure: bool = False
    """Available RAM is below the absolute danger floor."""
    reason: str = ""
    """Short human explanation, usually the verdict's ``reason()`` string."""
    available_mb: int | None = None
    """Measured available system RAM (MB), or None when telemetry is unavailable."""
    floor_mb: int = 0
    """The active available-RAM danger floor (MB)."""
    total_mb: int | None = None
    """Total system RAM (MB), or None when unknown."""
    pop_hold_active: bool = False
    """The soft intake hold is active because RAM is pressured, near the floor, or reclaim is draining."""
    pop_pause_active: bool = False
    """The hard self-throttle pop pause is active."""
    pop_pause_remaining_seconds: float | None = None
    """Seconds until the hard pop pause lapses, when active."""
    draining_process_ids: list[int] = Field(default_factory=list)
    """Inference process ids currently draining so a RAM-heavy slot can be recycled."""
    shed_card_indices: list[int] = Field(default_factory=list)
    """GPU device indices whose inference context count was reduced for host-RAM pressure."""
    restore_headroom_mb: int = 0
    """Measured RAM headroom above reserves available for restoring a shed context."""
    per_context_ram_estimate_mb: int = 0
    """Estimated resident-RAM cost of restoring one inference context."""
    per_process_ceiling_mb: int | None = None
    """Configured resident-RAM ceiling for a single inference process, when enabled."""


class PreloadAdmissionSnapshot(BaseModel):
    """The most recent preload-admission decision the scheduler made for a queued image job."""

    decision: str = ""
    """Stable decision key from ``AdmissionDecision`` (e.g. ``admit``, ``defer_budget``)."""
    model: str | None = None
    """Model whose queued job was judged, if known."""
    process_id: int | None = None
    """Target process selected by the decision, when one was selected."""
    reason: str = ""
    """Short human explanation for the decision."""
    timestamp: float = 0.0
    """Worker wall-clock time when the decision was recorded; 0 means no decision yet."""


class SchedulingGovernanceSnapshot(BaseModel):
    """Operator-visible scheduler governance state for the Overview and Live views."""

    ram: RamGovernanceSnapshot = Field(default_factory=RamGovernanceSnapshot)
    """The RAM governor's latest measured posture and active remedies."""
    preload: PreloadAdmissionSnapshot = Field(default_factory=PreloadAdmissionSnapshot)
    """The latest preload-admission decision made while scanning the pending queue."""


_CARD_VRAM_PRESSURE_FLOOR_MB = 1024.0
"""Free VRAM below this (or below ~8% of the card) reads as VRAM pressure (a near-OOM heads-up)."""


class CardSnapshot(BaseModel):
    """A serializable per-card view of one GPU this worker drives, for the multi-GPU dashboard.

    One entry per driven card (a single-GPU host reports exactly one). Carries what an operator needs to
    judge a single card at a glance: its VRAM headroom, how many inference contexts it is running against its
    target, the whole-card residency it may be holding, and any models that have gone locally unservable on
    it. Per-card it/s and jobs/hr are derived on the receiving side (from the card's processes and successive
    ``jobs_completed`` deltas) so the math lives with the other throughput trends.
    """

    device_index: int
    """The stable (PCI-bus) index of this card."""
    device_name: str | None = None
    """The human GPU name (e.g. ``NVIDIA GeForce RTX 4090``), or None when the device map has no entry."""
    kind: str = "cuda"
    """The accelerator backend (``cuda``/``rocm``/``xpu``/...)."""

    total_vram_mb: float | None = None
    """This card's total VRAM (MB); None until a process reports and the device map lacks a capacity."""
    free_vram_mb: float | None = None
    """Measured free VRAM (MB) on this card; None until a process on it has reported memory."""

    loaded_contexts: int = 0
    """Inference processes on this card with an allocated device context right now."""
    busy_contexts: int = 0
    """This card's inference processes currently mid-inference (a duty proxy and 'card is active' signal)."""
    target_process_count: int = 0
    """How many inference processes this card aims to run (its queue/concurrency-derived target)."""
    max_concurrent_inference: int = 0
    """This card's concurrent-sampling ceiling."""

    jobs_completed: int = 0
    """Cumulative jobs this card has finished; the receiver derives jobs/hr from successive deltas."""

    residency_model: str | None = None
    """The model holding whole-card residency on this card, or None when none is held."""
    residency_phase: str = ""
    """``establishing`` / ``holding`` while a residency is active on this card, else empty."""

    unservable_models: list[str] = Field(default_factory=list)
    """Models currently treated as locally unservable on this card (its over-budget circuit-breaker tripped)."""
    worst_fault_streak: int = 0
    """The worst per-model consecutive over-budget fault streak on this card (0 when none)."""

    @property
    def vram_headroom_fraction(self) -> float | None:
        """Free VRAM as a fraction of this card's total (0.0-1.0), or None when either figure is unknown."""
        if self.free_vram_mb is None or not self.total_vram_mb:
            return None
        return max(0.0, min(self.free_vram_mb / self.total_vram_mb, 1.0))

    @property
    def is_vram_pressured(self) -> bool:
        """Whether this card is low on VRAM (a near-OOM heads-up): under a small floor or a small fraction.

        The fraction guard catches a large card with proportionally little left; the absolute floor catches
        a small card where even a healthy-looking fraction is too few MB to load another model safely.
        """
        if self.free_vram_mb is None:
            return False
        fraction = self.vram_headroom_fraction
        return self.free_vram_mb < _CARD_VRAM_PRESSURE_FLOOR_MB or (fraction is not None and fraction < 0.08)


class SystemMemorySnapshot(BaseModel):
    """The system-RAM picture and the worker's per-role share of it, for the supervisor/TUI.

    The wire projection of :class:`~horde_worker_regen.process_management.resources.system_memory.SystemMemorySummary`.
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
        from horde_worker_regen.process_management.resources.system_memory import build_system_memory_summary

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
    supervisor_paused: bool = False
    """The worker is locally paused by operator command (F2 / PAUSE); distinct from any server-side state."""
    last_pop_maintenance_mode: bool = False
    """The most recent pop response returned a maintenance-mode error (cleared on the next successful pop)."""
    worker_details_maintenance: bool = False
    """The horde's worker-details API reports this worker in maintenance (polled, advisory)."""
    worker_details_paused: bool = False
    """The horde's worker-details API reports this worker paused (polled, advisory)."""
    too_many_consecutive_failed_jobs: bool = False

    gpu_torch_incompatible: bool = False
    """The installed PyTorch has no CUDA kernels for this GPU's architecture, so the worker stopped popping.

    Reported by a torch-bearing inference child at startup (the parent and TUI never import torch). A
    build/hardware mismatch: the wheel was compiled for a different set of GPU architectures than the
    installed card. Sticky for the session; fixed by reinstalling the matching backend and restarting."""
    gpu_torch_incompatible_reason: str | None = None
    """Operator-facing detail for ``gpu_torch_incompatible`` (device + remedy); None when not tripped."""

    torch_build_cpu_only: bool = False
    """The installed PyTorch is a CPU-only build, so image generation is disabled (alchemy still runs).

    Reported by a torch-bearing inference child at startup (the parent and TUI never import torch). The
    runtime counterpart of a ``bin/backend`` 'cpu' sentinel: it makes a CPU torch build serve alchemy-only
    even when the sentinel was never set (e.g. a manual CPU install). Sticky for the session; fixed by
    installing a GPU build and restarting."""
    torch_build_cpu_only_reason: str | None = None
    """Operator-facing detail for ``torch_build_cpu_only`` (why image gen is off + remedy); None when not tripped."""

    post_processing_disabled: bool = False
    """Post-processing is session-disabled and no longer advertised to the Horde."""
    post_processing_disabled_reason: str | None = None
    """Operator-facing detail for why post-processing was disabled; None when not tripped."""

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
    jobs_pending_post_processing: int = 0
    jobs_being_post_processed: int = 0
    jobs_pending_submit: int = 0
    """Jobs that have cleared safety and are queued for API submission (the pipeline's tail stage)."""
    time_spent_no_jobs_available: float = 0.0

    kudos_per_hour: float | None = None
    kudos_this_session: float | None = None
    eligible_seconds_total: float = 0.0
    """Cumulative productive (pipeline-occupied) seconds since the first submit; the kudos/hr denominator."""

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
    feature_readiness: FeatureReadinessSummary | None = None
    """Per-feature deps+on-disk readiness driving which gated features the worker offers (None until built)."""
    lora_pops_blocked_by_downloads: bool = False
    """Configured LoRA support is temporarily suppressed because background downloads are active."""
    lora_pops_blocked_by_disk: bool = False
    """Configured LoRA support is suppressed because the LoRA cache volume is below its free-space floor.

    Unlike the transient download block, this persists until disk space recovers; the TUI surfaces it
    prominently because, left unaddressed, it stops the worker from serving any LoRA jobs."""

    recent_jobs: list[RecentJobRecord] = Field(default_factory=list)
    """The most recent finished-job records, newest last (capped)."""

    latest_stats_sample: StatsSample | None = None
    """The latest one-second worker-owned stats sample."""
    stats_model_rollups: list[StatsRollupRow] = Field(default_factory=list)
    """Finalized image-job rollups by model."""
    stats_baseline_rollups: list[StatsRollupRow] = Field(default_factory=list)
    """Finalized image-job rollups by baseline."""
    stats_form_rollups: list[StatsRollupRow] = Field(default_factory=list)
    """Finalized alchemy-form rollups by form (``model`` carries the form name); empty for a dreamer worker."""
    stats_export: StatsExportState = Field(default_factory=StatsExportState)
    """Worker-side stats JSONL export state."""
    stats_history_backfill: StatsHistoryBackfill | None = None
    """Bounded stats history for reconnecting frontends."""

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

    enabled_workloads: list[str] = Field(default_factory=list)
    """The workloads this worker serves, as ``WorkloadKind`` values (e.g. ``image_generation``,
    ``alchemy``), sorted for stable rendering.

    Carried as plain strings so this module stays free of the heavy import chain behind ``WorkloadKind``;
    consumers reconstruct the typed enum. The dashboard uses this to identify the worker's mode (an
    alchemist-only worker reshapes around alchemy) rather than inferring it from model counts. Empty only
    before the first snapshot or for a worker configured to serve nothing."""

    pending_jobs: list[JobQueueEntry] = Field(default_factory=list)
    """Pending-inference jobs (capped at :data:`PENDING_JOBS_IN_SNAPSHOT`), oldest first."""

    orchestration_intent: OrchestrationIntentSnapshot = Field(default_factory=OrchestrationIntentSnapshot)
    """Current plain-English scheduler intent for the Overview Now/Next/Why strip."""

    work_ledger: list[WorkLedgerEntry] = Field(default_factory=list)
    """Active and recent job rows for the Overview work ledger."""

    whole_card_residency: WholeCardResidencyStatus = Field(default_factory=WholeCardResidencyStatus)
    """Whole-card exclusive-residency posture: whether it can engage, and live detail when it has."""

    pop_governors: PopGovernorsSnapshot = Field(default_factory=PopGovernorsSnapshot)
    """The pop/scheduling governors holding back or reshaping job pops, with live + session-aggregate state."""

    scheduling_governance: SchedulingGovernanceSnapshot = Field(default_factory=SchedulingGovernanceSnapshot)
    """RAM-governor posture plus the latest preload-admission decision for operator diagnostics."""

    per_card: list[CardSnapshot] = Field(default_factory=list)
    """Per-card multi-GPU view, one entry per driven card (exactly one on a single-GPU host)."""


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
    SET_STATS_EXPORT = enum.auto()
    """Enable or disable worker-side stats JSONL export for this session."""


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
    stats_export_enabled: bool | None = None
    """Desired stats JSONL export state for :attr:`SupervisorCommand.SET_STATS_EXPORT`."""


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
