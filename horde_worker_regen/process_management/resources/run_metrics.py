"""Worker-wide run metrics, readable in-process without an OTel backend.

Logfire mirrors most of these numbers for observability, but the benchmark controller
(and the e2e harness) need them programmatically at the end of a run. This module
aggregates per-job stage latencies (from the job tracker's finalize observer), per-job
phase metrics and download events (from the child-process metrics messages), process
crash events, and headline counters into one :class:`RunMetricsSnapshot`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from hordelib.metrics import DownloadEvent, JobPhaseMetrics
from loguru import logger
from pydantic import BaseModel, Field

from horde_worker_regen.app_state import default_app_state_dir
from horde_worker_regen.process_management.ipc.messages import (
    HordeDownloadMetricsMessage,
    HordeJobMetricsMessage,
)
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    StatsExportState,
    StatsHistoryBackfill,
    StatsRollupRow,
    StatsSample,
)
from horde_worker_regen.telemetry_spans import (
    job_e2e_histogram,
    job_queue_wait_histogram,
    job_safety_histogram,
)

if TYPE_CHECKING:
    from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
    from horde_worker_regen.process_management.jobs.job_tracker import TrackedJob

ChurnKind = Literal["model_swap", "vram_eviction", "process_cycle"]
"""A between-jobs reload/respawn event whose frequency erodes duty cycle.

``model_swap``: a preload that displaced a *different* model already resident on that process (the
prior model's load work is thrown away). ``vram_eviction``: an idle resident model was unloaded from
VRAM to make room. ``process_cycle``: a healthy idle inference process was deliberately restarted to
reclaim allocator-stranded RAM. None are faults; their *rate* is the churn signal.
"""

_CHURN_EVENT_RETENTION_SECONDS = 3600.0
_STATS_SAMPLE_INTERVAL_SECONDS = 1.0
_STATS_RECENT_HISTORY_SECONDS = 2 * 60 * 60
_STATS_ALL_SESSION_POINTS = 720
_STATS_ROTATE_BYTES = 5 * 1024 * 1024
_STATS_WARNING_BYTES = 50 * 1024 * 1024
"""Drop churn timestamps older than this so the lists stay bounded on a long-running worker. Far wider
than the duty-cycle report window, so every report's lookback is fully covered."""


class JobMetricsRecord(BaseModel):
    """The full metrics picture of one finished job (or alchemy form)."""

    job_id: str
    is_alchemy: bool = False
    faulted: bool = False
    time_popped: float | None = None
    stage_timestamps: dict[str, float] = {}
    """Epoch time of first entry into each ``JobStage`` (plus ``FINALIZED``)."""
    queue_wait_seconds: float | None = None
    """Pop to inference start."""
    e2e_seconds: float | None = None
    """Pop to finalization (submit)."""
    safety_seconds: float | None = None
    """Safety-check queue entry to submit-ready."""
    phase_metrics: JobPhaseMetrics | None = None
    """Model-load/sampling/memory metrics reported by the child process, when correlated."""

    model_name: str | None = None
    steps: int | None = None
    width: int | None = None
    height: int | None = None
    loras_count: int = 0
    tis_count: int = 0
    control_type: str | None = None
    post_processing: list[str] = Field(default_factory=list)
    hires_fix: bool = False
    batch_count: int = 1
    megapixelsteps: float = 0.0
    sampling_seconds: float | None = None


class StatsSampleEvent(BaseModel):
    """One JSONL export event carrying a periodic stats sample."""

    event: Literal["stats_sample"] = "stats_sample"
    sample: StatsSample


class StatsJobCompletedEvent(BaseModel):
    """One JSONL export event carrying a finalized job metrics record."""

    event: Literal["job_completed"] = "job_completed"
    job: JobMetricsRecord
    baseline: str | None = None


class _StatsJsonlExporter:
    """Session-scoped, rotating JSONL writer for stats samples and finalized jobs."""

    def __init__(self, *, worker_version: str, state_dir: Path | None = None) -> None:
        self._directory = (state_dir if state_dir is not None else default_app_state_dir()) / "stats"
        self._version = worker_version.replace("/", "_").replace("\\", "_")
        self._stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self._index = 0
        self._active_path: Path | None = None
        self._last_write_error: str | None = None
        self._disabled_by_error = False

    @property
    def active_file_path(self) -> str | None:
        """Return the active JSONL path, if a file has been opened."""
        return str(self._active_path) if self._active_path is not None else None

    @property
    def last_write_error(self) -> str | None:
        """Return the last write error surfaced to the TUI."""
        return self._last_write_error

    @property
    def disabled_by_error(self) -> bool:
        """Whether export disabled itself after an IO failure."""
        return self._disabled_by_error

    def write(self, event: StatsSampleEvent | StatsJobCompletedEvent) -> bool:
        """Append one typed event. Returns False when an IO error disabled export."""
        if self._disabled_by_error:
            return False
        try:
            payload = event.model_dump_json() + "\n"
            path = self._path_for_payload(len(payload.encode("utf-8")))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
        except OSError as write_error:
            self._last_write_error = str(write_error)
            self._disabled_by_error = True
            logger.warning(f"Stats JSONL export disabled after write failure: {write_error}")
            return False
        return True

    def state(self, *, enabled: bool) -> StatsExportState:
        """Build the supervisor-visible export state."""
        total = self.total_bytes()
        return StatsExportState(
            enabled=enabled and not self._disabled_by_error,
            active_file_path=self.active_file_path,
            bytes_in_stats_files=total,
            warning_over_50_mib=total > _STATS_WARNING_BYTES,
            last_write_error=self._last_write_error,
        )

    def total_bytes(self) -> int:
        """Return the total size of retained stats JSONL files."""
        try:
            return sum(path.stat().st_size for path in self._directory.glob("stats-v*.jsonl") if path.is_file())
        except OSError:
            return 0

    def _path_for_payload(self, payload_bytes: int) -> Path:
        self._directory.mkdir(parents=True, exist_ok=True)
        if self._active_path is None:
            self._active_path = self._candidate_path()
            return self._active_path
        try:
            current_size = self._active_path.stat().st_size if self._active_path.exists() else 0
        except OSError:
            current_size = 0
        if current_size and current_size + payload_bytes > _STATS_ROTATE_BYTES:
            self._index += 1
            self._active_path = self._candidate_path()
        return self._active_path

    def _candidate_path(self) -> Path:
        return self._directory / f"stats-v{self._version}-{self._stamp}-{self._index:03d}.jsonl"


class ProcessCrashRecord(BaseModel):
    """One observed child-process crash/hang/replacement."""

    process_id: int
    process_launch_identifier: int
    last_state: str
    reason: str
    timestamp: float


class RunMetricsSnapshot(BaseModel):
    """Everything the run metrics aggregator observed, frozen at snapshot time."""

    jobs: list[JobMetricsRecord]
    downloads: list[DownloadEvent]
    vram_used_high_water_mb_per_process: dict[int, int]
    ram_used_high_water_mb_per_process: dict[int, int]
    disk_min_free_bytes: dict[str, int]
    num_process_recoveries: int
    num_job_slowdowns: int
    job_slowdown_events: int = 0
    """Count of in-flight WARN-level slowdown gradings the hung-process watchdog raised this run (a running
    job measured far past its expected sampling time). Distinct from ``num_job_slowdowns``, which counts
    *completed* jobs that finished slow: this counts the mid-flight escalations, the same signal the
    paged-slowdown watchdog joins with WDDM paging attribution to reclaim a card."""
    paging_victim_replacements: int = 0
    """Count of inference slots the paged-slowdown watchdog replaced this run: a WARN-graded job that was
    still advancing but on VRAM the WDDM driver had demoted to system memory (measured per-PID paging
    attribution). Each is a card reclaimed from a job that would otherwise have limped for minutes without
    tripping the silence-based hang timeout."""
    time_spent_no_jobs_available: float
    process_crash_events: list[ProcessCrashRecord]
    gpu_utilization_mean_percent: float | None = None
    """Average GPU core utilization (the duty cycle) sampled over the run, when measured."""
    gpu_utilization_busy_fraction: float | None = None
    """Fraction of GPU samples at or above the busy threshold, when measured."""
    gpu_utilization_samples: int = 0
    """How many GPU-utilization samples backed the figures above."""
    churn_event_times: dict[ChurnKind, list[float]] = Field(default_factory=dict)
    """Epoch timestamps of each between-jobs reload/respawn event, keyed by :data:`ChurnKind`.

    Raw timestamps (not pre-counted) so a consumer can attribute churn to the same window it measures
    duty over, exactly as it filters ``jobs`` by ``FINALIZED``. Retained for the last hour."""
    phase: str = ""
    """Human-readable description of what the worker was doing at snapshot time (e.g. "initializing
    inference process", "waiting for first job", "running inference"). Drives benchmark live progress
    so a slow cold start reads as motion rather than a hang. Empty when not populated."""
    process_state_summary: str = ""
    """Compact per-process state line (e.g. ``inf#1=PROCESS_STARTING safety#0=WAITING_FOR_JOB``)."""
    committed_vram_mb: float | None = None
    """Last reconciled committed-VRAM ledger sum (context constant + allocator-reserved per live GPU
    process) for the primary card, when the attribution reconciler has observed one. Calibration
    visibility only; scheduling never reads it from here."""
    vram_attribution_drift_mb: float | None = None
    """Last reconciled attribution drift (device used minus baseline+committed) for the primary card.
    Persistent positive drift is the only early overcommit signal that exists under WDDM."""
    admission_denials: int = 0
    """Count of admissions denied by the ledger-driven measured floor for the primary card this run.

    A measured denial is one the lying free-VRAM figure would have admitted but the committed-plus-planned
    ledger arithmetic rejected as a physical over-commit. Calibration visibility only."""
    measured_unloads_issued: int = 0
    """Count of under-pressure idle-model unloads the physical-overcommit trigger issued for the primary card.

    Each is one idle resident model evicted because ``committed + baseline > total`` held across the confirming
    streak. Calibration visibility only."""
    admission_headroom_mb: float | None = None
    """Last observed admission headroom (capacity minus committed-plus-planned demand) for the primary card,
    or None when the measured floor was not applied (cold start or a stale ledger). Calibration visibility."""
    admission_foreign_pressure_defers: int = 0
    """Count of preloads the arbiter deferred under foreign pressure this run: reclaim was exhausted and the
    worker's own committed load fit capacity, yet the candidate did not physically fit the truthful device-free
    reading, so it could not be admitted into reality. A rising count marks a card held by load the worker did
    not commit. Calibration visibility only."""
    starvation_diagnostics: int = 0
    """Count of starvation diagnostics the arbiter emitted this run: a head-of-queue preload deferred past the
    diagnostic horizon with the reclaim ladder exhausted and no verified progress. Each named the full
    admission arithmetic in a warning; the job stays queued for the structural-wedge recovery supervisor to
    reroute. Calibration visibility only."""
    device_free_mb: float | None = None
    """Latest NVML device-level free VRAM (MB) for the primary card, read by the device-free governor in the
    torch-free parent, or None before the first governor sample. On WDDM this device-level figure is the only
    truthful proximity-to-cliff signal: per-process reads lie once the driver demotes an allocator to system
    memory. Calibration visibility only."""
    governor_pressure_events: int = 0
    """Count of device-free governor transitions into PRESSURE (free below the soft floor) this run, summed
    across governed cards. Each marks a card crossing into the band where new VRAM growth is held. Calibration
    visibility only."""
    governor_saturation_events: int = 0
    """Count of device-free governor transitions into SATURATED (free below the hard floor) this run, summed
    across governed cards. Each marks a card crossing the paging cliff, where the reclaim ladder runs.
    Calibration visibility only."""
    ladder_rungs_issued: int = 0
    """Count of verified-reclaim-ladder rungs the engine issued this run (each an idle-model unload, allocator
    cache release, lane pause, or safety off-GPU that actually acted). Calibration visibility only."""
    ladder_verified_frees_mb: float = 0.0
    """Cumulative realized NVML device-free gain (MB) the reclaim ladder verified against its rungs' promises
    this run. The measured, externally-confirmed counterpart to the promised frees. Calibration visibility."""
    ladder_verification_shortfalls: int = 0
    """Count of reclaim rungs that freed less than half their promised device memory within the verification
    window this run: each named its tenant in a warning and recorded a calibration event. Calibration
    visibility only."""
    per_step_floor_triggers: int = 0
    """Count of times the per-step floor tripped this run: a sampling slot ran two consecutive steps each at
    or above the floor multiple of its expected per-step time while its card was at PRESSURE or SATURATED,
    forcing the reclaim ladder to run without waiting for the whole-job elapsed-ratio rungs. Calibration
    visibility only."""
    dispatch_reconciliation_holds: int = 0
    """Count of dispatches the residency-reconciliation gate held this run: a staged job whose VRAM
    materialisation would over-commit the card was kept queued (never faulted) while the single reclaim owner
    evicted idle residents, rather than committing its weights to VRAM into an over-commit. Calibration
    visibility only."""
    dispatch_reconciliation_conflicts: int = 0
    """Count of dispatch-time residency conflicts detected this run: every scheduling pass a staged dispatch
    was priced non-fitting counts once, so this can exceed the held-dispatch count when a single job is held
    across several passes. Calibration visibility only."""
    dispatch_reconciliation_hold_seconds: float = 0.0
    """Cumulative seconds dispatches spent held for residency reconciliation this run, summed across held jobs.
    A rising figure marks the card spending real time reconciling co-resident weights before dispatch.
    Calibration visibility only."""
    dispatch_reconciliation_released_by_reclaim: int = 0
    """Count of held dispatches released after this gate's own idle-resident eviction freed the room this run
    (device-free verified sufficient on a later pass). Its counterpart to natural free separates room this gate
    made from room the card recovered on its own. Calibration visibility only."""
    dispatch_reconciliation_released_by_natural_free: int = 0
    """Count of held dispatches released because the card freed on its own this run (a sibling finished or
    foreign pressure abated), without this gate having emitted an eviction for the held job. Calibration
    visibility only."""
    safety_placement_demotions: int = 0
    """Count of runtime safety-placement policy moves of the safety process off-GPU to CPU this run: the safety
    context did not fit beside the largest active sampling peak on its card for the consecutive-cycle streak.
    Distinct from the whole-card residency's own safety pauses. Calibration visibility only."""
    safety_placement_promotions: int = 0
    """Count of runtime safety-placement policy restores of the safety process back onto the GPU this run: the
    chosen card's measured device-free proved durable room for the safety context for the consecutive-cycle
    streak. Its counterpart to the demotions; a demotion with no matching promotion is CPU safety as the
    steady state under sustained load. Calibration visibility only."""
    safety_placement_card: int | None = None
    """The driven card the safety process currently occupies, or None when safety is off-GPU (running on CPU).
    On a box too tight to host safety beside its sampler this settles at None, with pop backpressure bounding
    intake to CPU-safety throughput. Calibration visibility only."""


class WorkerRunMetrics:
    """Aggregates run-wide metrics inside the main process.

    Wired by ``HordeWorkerProcessManager``: the message dispatcher feeds job/download
    metrics messages, the job tracker's finalize observer feeds stage latencies, and
    the process lifecycle manager records crash events.
    """

    def __init__(self, *, baseline_resolver: Callable[[str], str | None] | None = None) -> None:
        """Initialize empty aggregation state."""
        self._jobs: list[JobMetricsRecord] = []
        self._downloads: list[DownloadEvent] = []
        self._phase_metrics_by_job: dict[str, JobPhaseMetrics] = {}
        self._vram_high_water_per_process: dict[int, int] = {}
        self._ram_high_water_per_process: dict[int, int] = {}
        self._crash_events: list[ProcessCrashRecord] = []
        self._baseline_resolver = baseline_resolver
        self._stats_samples: list[StatsSample] = []
        self._all_stats_samples: list[StatsSample] = []
        self._last_stats_sample_time = 0.0
        self._model_rollups: dict[tuple[str | None, str | None], StatsRollupRow] = {}
        self._baseline_rollups: dict[str | None, StatsRollupRow] = {}
        self._form_rollups: dict[str, StatsRollupRow] = {}
        self._stats_exporter: _StatsJsonlExporter | None = None
        self._stats_export_enabled = False
        self._churn_event_times: dict[ChurnKind, list[float]] = {
            "model_swap": [],
            "vram_eviction": [],
            "process_cycle": [],
        }

    def reset(self) -> None:
        """Clear all aggregated metrics, e.g. at a benchmark level boundary on a warm worker."""
        self._jobs.clear()
        self._downloads.clear()
        self._phase_metrics_by_job.clear()
        self._vram_high_water_per_process.clear()
        self._ram_high_water_per_process.clear()
        self._crash_events.clear()
        self._stats_samples.clear()
        self._all_stats_samples.clear()
        self._last_stats_sample_time = 0.0
        self._model_rollups.clear()
        self._baseline_rollups.clear()
        self._form_rollups.clear()
        for times in self._churn_event_times.values():
            times.clear()

    def record_churn(self, kind: ChurnKind) -> None:
        """Record one between-jobs reload/respawn event (see :data:`ChurnKind`), pruning stale entries.

        Wired as the scheduler's churn observer; safe to call from the control loop. Pruning here (not
        on read) keeps the lists bounded without a separate timer.
        """
        now = time.time()
        times = self._churn_event_times[kind]
        times.append(now)
        cutoff = now - _CHURN_EVENT_RETENTION_SECONDS
        if times and times[0] < cutoff:
            self._churn_event_times[kind] = [t for t in times if t >= cutoff]

    def on_job_metrics(self, message: HordeJobMetricsMessage) -> None:
        """Handle a per-job metrics message from a child process."""
        metrics = message.phase_metrics

        if metrics.vram_used_high_water_mb is not None:
            current = self._vram_high_water_per_process.get(message.process_id, 0)
            self._vram_high_water_per_process[message.process_id] = max(current, metrics.vram_used_high_water_mb)
        if metrics.ram_used_high_water_mb is not None:
            current = self._ram_high_water_per_process.get(message.process_id, 0)
            self._ram_high_water_per_process[message.process_id] = max(current, metrics.ram_used_high_water_mb)

        # Both image jobs and alchemy forms finalize later (the alchemy coordinator records the form's
        # full record at submit, with its name and pop->submit timing), so hold the child's phase metrics
        # keyed by job/form id for correlation when that record is built.
        self._phase_metrics_by_job[message.job_id] = metrics

    def on_download_metrics(self, message: HordeDownloadMetricsMessage) -> None:
        """Handle a download-events message from a child process."""
        self._downloads.extend(message.events)

    def on_job_finalized(self, tracked: TrackedJob, completed_job_info: HordeJobInfo) -> None:
        """Fold a finalized job's stage latencies into the run metrics (tracker observer)."""
        from horde_sdk.ai_horde_api import GENERATION_STATE

        stage_timestamps = dict(tracked.stage_timestamps)
        time_popped = tracked.time_popped
        finalized_at = stage_timestamps.get("FINALIZED", time.time())

        queue_wait: float | None = None
        inference_started = stage_timestamps.get("INFERENCE_IN_PROGRESS")
        if time_popped is not None and inference_started is not None:
            queue_wait = inference_started - time_popped

        e2e: float | None = None
        if time_popped is not None:
            e2e = finalized_at - time_popped

        safety: float | None = None
        safety_started = stage_timestamps.get("PENDING_SAFETY_CHECK")
        submit_ready = stage_timestamps.get("PENDING_SUBMIT")
        if safety_started is not None and submit_ready is not None:
            safety = submit_ready - safety_started

        if queue_wait is not None:
            job_queue_wait_histogram.record(queue_wait)
        if e2e is not None:
            job_e2e_histogram.record(e2e)
        if safety is not None:
            job_safety_histogram.record(safety)

        job_id = str(tracked.job_id)
        api_job = tracked.sdk_api_job_info
        model_name: str | None = str(api_job.model) if api_job.model is not None else None
        payload = api_job.payload
        control_type: str | None = str(payload.control_type) if payload.control_type else None

        phase_metrics = self._phase_metrics_by_job.pop(job_id, None)
        sampling_seconds = (
            phase_metrics.sampling.duration_seconds if phase_metrics is not None and phase_metrics.sampling else None
        )
        batch_count = payload.n_iter if isinstance(payload.n_iter, int) and payload.n_iter > 0 else 1
        megapixelsteps = (
            (float(payload.width or 0) * float(payload.height or 0) / 1_000_000.0)
            * float(
                payload.ddim_steps or 0,
            )
            * float(batch_count)
        )
        record = JobMetricsRecord(
            job_id=job_id,
            faulted=completed_job_info.state == GENERATION_STATE.faulted,
            time_popped=time_popped,
            stage_timestamps=stage_timestamps,
            queue_wait_seconds=queue_wait,
            e2e_seconds=e2e,
            safety_seconds=safety,
            phase_metrics=phase_metrics,
            model_name=model_name,
            steps=payload.ddim_steps,
            width=payload.width,
            height=payload.height,
            loras_count=len(payload.loras) if payload.loras else 0,
            tis_count=len(payload.tis) if payload.tis else 0,
            control_type=control_type,
            post_processing=[str(post_proc_step) for post_proc_step in payload.post_processing],
            hires_fix=payload.hires_fix,
            batch_count=batch_count,
            megapixelsteps=megapixelsteps,
            sampling_seconds=sampling_seconds,
        )
        self._jobs.append(record)
        baseline = self._resolve_baseline(model_name)
        self._fold_rollup(record, baseline=baseline)
        self._write_job_event(record, baseline=baseline)

    def record_alchemy_form(
        self,
        *,
        form_id: str,
        form: str,
        e2e_seconds: float | None,
        faulted: bool,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Record one finished alchemy form, the alchemist analogue of :meth:`on_job_finalized`.

        Alchemy forms do not pass through the image job tracker, so the alchemy coordinator calls this at
        submit/fault time with the pop->submit ``e2e_seconds`` and the form's name and source resolution.
        Any phase metrics the child reported for this form (VRAM/RAM, correlated by id) are folded in. The
        record feeds the recent-jobs view, the by-form rollup, and the JSONL export, giving alchemy the
        same per-job observability image generation has.
        """
        phase_metrics = self._phase_metrics_by_job.pop(form_id, None)
        sampling_seconds = (
            phase_metrics.sampling.duration_seconds if phase_metrics is not None and phase_metrics.sampling else None
        )
        if e2e_seconds is not None:
            job_e2e_histogram.record(e2e_seconds)
        record = JobMetricsRecord(
            job_id=form_id,
            is_alchemy=True,
            faulted=faulted,
            e2e_seconds=e2e_seconds,
            phase_metrics=phase_metrics,
            model_name=form,
            width=width,
            height=height,
            sampling_seconds=sampling_seconds,
        )
        self._jobs.append(record)
        self._fold_form_rollup(record)
        self._write_job_event(record, baseline=None)

    def record_process_crash(
        self,
        *,
        process_id: int,
        process_launch_identifier: int,
        last_state: str,
        reason: str,
    ) -> None:
        """Record one child-process crash/hang/replacement event."""
        self._crash_events.append(
            ProcessCrashRecord(
                process_id=process_id,
                process_launch_identifier=process_launch_identifier,
                last_state=last_state,
                reason=reason,
                timestamp=time.time(),
            ),
        )

    def set_stats_export(self, enabled: bool, *, worker_version: str) -> None:
        """Enable or disable session-scoped stats JSONL export."""
        self._stats_export_enabled = enabled
        if enabled and self._stats_exporter is None:
            self._stats_exporter = _StatsJsonlExporter(worker_version=worker_version)

    def record_stats_sample(self, sample: StatsSample) -> StatsSample | None:
        """Append a periodic stats sample at most once per second and export it when enabled."""
        if sample.timestamp - self._last_stats_sample_time < _STATS_SAMPLE_INTERVAL_SECONDS:
            return None
        self._last_stats_sample_time = sample.timestamp
        self._stats_samples.append(sample)
        self._all_stats_samples.append(sample)
        cutoff = sample.timestamp - _STATS_RECENT_HISTORY_SECONDS
        if self._stats_samples and self._stats_samples[0].timestamp < cutoff:
            self._stats_samples = [entry for entry in self._stats_samples if entry.timestamp >= cutoff]
        self._write_sample_event(sample)
        return sample

    def latest_stats_sample(self) -> StatsSample | None:
        """Return the newest worker-owned stats sample."""
        return self._stats_samples[-1] if self._stats_samples else None

    def stats_history_backfill(self) -> StatsHistoryBackfill:
        """Return exact recent samples plus a decimated all-session sequence."""
        return StatsHistoryBackfill(
            recent_samples=list(self._stats_samples),
            all_session_samples=self._decimated_stats_samples(),
        )

    def model_rollups(self) -> list[StatsRollupRow]:
        """Return finalized image-job rollups by model."""
        return sorted(self._model_rollups.values(), key=lambda row: row.jobs, reverse=True)

    def baseline_rollups(self) -> list[StatsRollupRow]:
        """Return finalized image-job rollups by baseline."""
        return sorted(self._baseline_rollups.values(), key=lambda row: row.jobs, reverse=True)

    def form_rollups(self) -> list[StatsRollupRow]:
        """Return finalized alchemy-form rollups by form (``model`` carries the form name)."""
        return sorted(self._form_rollups.values(), key=lambda row: row.jobs, reverse=True)

    def stats_export_state(self) -> StatsExportState:
        """Return current JSONL export state for the supervisor snapshot."""
        if self._stats_exporter is None:
            return StatsExportState(enabled=False)
        state = self._stats_exporter.state(enabled=self._stats_export_enabled)
        if self._stats_exporter.disabled_by_error:
            self._stats_export_enabled = False
        return state

    def _resolve_baseline(self, model_name: str | None) -> str | None:
        if model_name is None or self._baseline_resolver is None:
            return None
        return self._baseline_resolver(model_name)

    def _fold_rollup(self, record: JobMetricsRecord, *, baseline: str | None) -> None:
        if record.is_alchemy:
            return
        model_key = (record.model_name, baseline)
        model_row = self._model_rollups.setdefault(
            model_key,
            StatsRollupRow(model=record.model_name, baseline=baseline),
        )
        self._add_to_rollup(model_row, record)

        baseline_row = self._baseline_rollups.setdefault(baseline, StatsRollupRow(baseline=baseline))
        self._add_to_rollup(baseline_row, record)

    def _fold_form_rollup(self, record: JobMetricsRecord) -> None:
        """Fold a finished alchemy form into the by-form rollup (keyed by form name).

        Reuses :class:`StatsRollupRow`: ``model`` holds the form, ``jobs`` counts forms, and
        ``e2e_seconds`` accumulates so the dashboard can show a per-form average. The sampling/megapixelstep
        columns stay zero for alchemy (forms have no diffusion steps), which the by-form table omits.
        """
        form = record.model_name or "unknown"
        row = self._form_rollups.setdefault(form, StatsRollupRow(model=form))
        self._add_to_rollup(row, record)

    @staticmethod
    def _add_to_rollup(row: StatsRollupRow, record: JobMetricsRecord) -> None:
        row.jobs += 1
        row.megapixelsteps += record.megapixelsteps
        row.sampling_seconds += record.sampling_seconds or 0.0
        row.e2e_seconds += record.e2e_seconds or 0.0
        if record.batch_count > 1:
            row.batch_gt_one_jobs += 1
        if record.faulted:
            row.faulted_jobs += 1
        if record.phase_metrics is not None and record.phase_metrics.vram_used_high_water_mb is not None:
            row.vram_high_water_mb = max(row.vram_high_water_mb, record.phase_metrics.vram_used_high_water_mb)

    def _write_sample_event(self, sample: StatsSample) -> None:
        if not self._stats_export_enabled or self._stats_exporter is None:
            return
        if not self._stats_exporter.write(StatsSampleEvent(sample=sample)):
            self._stats_export_enabled = False

    def _write_job_event(self, record: JobMetricsRecord, *, baseline: str | None) -> None:
        if not self._stats_export_enabled or self._stats_exporter is None:
            return
        if not self._stats_exporter.write(StatsJobCompletedEvent(job=record, baseline=baseline)):
            self._stats_export_enabled = False

    def _decimated_stats_samples(self) -> list[StatsSample]:
        if len(self._all_stats_samples) <= _STATS_ALL_SESSION_POINTS:
            return list(self._all_stats_samples)
        step = len(self._all_stats_samples) / _STATS_ALL_SESSION_POINTS
        return [self._all_stats_samples[int(index * step)] for index in range(_STATS_ALL_SESSION_POINTS)]

    def snapshot(
        self,
        *,
        num_process_recoveries: int = 0,
        num_job_slowdowns: int = 0,
        job_slowdown_events: int = 0,
        paging_victim_replacements: int = 0,
        time_spent_no_jobs_available: float = 0.0,
        disk_min_free_bytes: dict[str, int] | None = None,
        phase: str = "",
        process_state_summary: str = "",
        committed_vram_mb: float | None = None,
        vram_attribution_drift_mb: float | None = None,
        admission_denials: int = 0,
        measured_unloads_issued: int = 0,
        admission_headroom_mb: float | None = None,
        admission_foreign_pressure_defers: int = 0,
        starvation_diagnostics: int = 0,
        device_free_mb: float | None = None,
        governor_pressure_events: int = 0,
        governor_saturation_events: int = 0,
        ladder_rungs_issued: int = 0,
        ladder_verified_frees_mb: float = 0.0,
        ladder_verification_shortfalls: int = 0,
        per_step_floor_triggers: int = 0,
        dispatch_reconciliation_holds: int = 0,
        dispatch_reconciliation_conflicts: int = 0,
        dispatch_reconciliation_hold_seconds: float = 0.0,
        dispatch_reconciliation_released_by_reclaim: int = 0,
        dispatch_reconciliation_released_by_natural_free: int = 0,
        safety_placement_demotions: int = 0,
        safety_placement_promotions: int = 0,
        safety_placement_card: int | None = None,
    ) -> RunMetricsSnapshot:
        """Return an immutable snapshot of everything observed so far.

        The headline counters and the live phase/process-state strings live on the process manager and
        its collaborators, so the caller passes them in rather than this class duplicating them.
        """
        return RunMetricsSnapshot(
            jobs=list(self._jobs),
            downloads=list(self._downloads),
            vram_used_high_water_mb_per_process=dict(self._vram_high_water_per_process),
            ram_used_high_water_mb_per_process=dict(self._ram_high_water_per_process),
            disk_min_free_bytes=dict(disk_min_free_bytes or {}),
            num_process_recoveries=num_process_recoveries,
            num_job_slowdowns=num_job_slowdowns,
            job_slowdown_events=job_slowdown_events,
            paging_victim_replacements=paging_victim_replacements,
            time_spent_no_jobs_available=time_spent_no_jobs_available,
            process_crash_events=list(self._crash_events),
            churn_event_times={kind: list(times) for kind, times in self._churn_event_times.items()},
            phase=phase,
            process_state_summary=process_state_summary,
            committed_vram_mb=committed_vram_mb,
            vram_attribution_drift_mb=vram_attribution_drift_mb,
            admission_denials=admission_denials,
            measured_unloads_issued=measured_unloads_issued,
            admission_headroom_mb=admission_headroom_mb,
            admission_foreign_pressure_defers=admission_foreign_pressure_defers,
            starvation_diagnostics=starvation_diagnostics,
            device_free_mb=device_free_mb,
            governor_pressure_events=governor_pressure_events,
            governor_saturation_events=governor_saturation_events,
            ladder_rungs_issued=ladder_rungs_issued,
            ladder_verified_frees_mb=ladder_verified_frees_mb,
            ladder_verification_shortfalls=ladder_verification_shortfalls,
            per_step_floor_triggers=per_step_floor_triggers,
            dispatch_reconciliation_holds=dispatch_reconciliation_holds,
            dispatch_reconciliation_conflicts=dispatch_reconciliation_conflicts,
            dispatch_reconciliation_hold_seconds=dispatch_reconciliation_hold_seconds,
            dispatch_reconciliation_released_by_reclaim=dispatch_reconciliation_released_by_reclaim,
            dispatch_reconciliation_released_by_natural_free=dispatch_reconciliation_released_by_natural_free,
            safety_placement_demotions=safety_placement_demotions,
            safety_placement_promotions=safety_placement_promotions,
            safety_placement_card=safety_placement_card,
        )
