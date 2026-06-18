"""Worker-wide run metrics, readable in-process without an OTel backend.

Logfire mirrors most of these numbers for observability, but the benchmark controller
(and the e2e harness) need them programmatically at the end of a run. This module
aggregates per-job stage latencies (from the job tracker's finalize observer), per-job
phase metrics and download events (from the child-process metrics messages), process
crash events, and headline counters into one :class:`RunMetricsSnapshot`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from hordelib.metrics import DownloadEvent, JobPhaseMetrics
from pydantic import BaseModel, Field

from horde_worker_regen.process_management.messages import (
    HordeDownloadMetricsMessage,
    HordeJobMetricsMessage,
)
from horde_worker_regen.telemetry_spans import (
    job_e2e_histogram,
    job_queue_wait_histogram,
    job_safety_histogram,
)

if TYPE_CHECKING:
    from horde_worker_regen.process_management.job_models import HordeJobInfo
    from horde_worker_regen.process_management.job_tracker import TrackedJob


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
    time_spent_no_jobs_available: float
    process_crash_events: list[ProcessCrashRecord]
    gpu_utilization_mean_percent: float | None = None
    """Average GPU core utilization (the duty cycle) sampled over the run, when measured."""
    gpu_utilization_busy_fraction: float | None = None
    """Fraction of GPU samples at or above the busy threshold, when measured."""
    gpu_utilization_samples: int = 0
    """How many GPU-utilization samples backed the figures above."""
    phase: str = ""
    """Human-readable description of what the worker was doing at snapshot time (e.g. "initializing
    inference process", "waiting for first job", "running inference"). Drives benchmark live progress
    so a slow cold start reads as motion rather than a hang. Empty when not populated."""
    process_state_summary: str = ""
    """Compact per-process state line (e.g. ``inf#1=PROCESS_STARTING safety#0=WAITING_FOR_JOB``)."""


class WorkerRunMetrics:
    """Aggregates run-wide metrics inside the main process.

    Wired by ``HordeWorkerProcessManager``: the message dispatcher feeds job/download
    metrics messages, the job tracker's finalize observer feeds stage latencies, and
    the process lifecycle manager records crash events.
    """

    def __init__(self) -> None:
        """Initialize empty aggregation state."""
        self._jobs: list[JobMetricsRecord] = []
        self._downloads: list[DownloadEvent] = []
        self._phase_metrics_by_job: dict[str, JobPhaseMetrics] = {}
        self._vram_high_water_per_process: dict[int, int] = {}
        self._ram_high_water_per_process: dict[int, int] = {}
        self._crash_events: list[ProcessCrashRecord] = []

    def reset(self) -> None:
        """Clear all aggregated metrics, e.g. at a benchmark level boundary on a warm worker."""
        self._jobs.clear()
        self._downloads.clear()
        self._phase_metrics_by_job.clear()
        self._vram_high_water_per_process.clear()
        self._ram_high_water_per_process.clear()
        self._crash_events.clear()

    def on_job_metrics(self, message: HordeJobMetricsMessage) -> None:
        """Handle a per-job metrics message from a child process."""
        metrics = message.phase_metrics

        if metrics.vram_used_high_water_mb is not None:
            current = self._vram_high_water_per_process.get(message.process_id, 0)
            self._vram_high_water_per_process[message.process_id] = max(current, metrics.vram_used_high_water_mb)
        if metrics.ram_used_high_water_mb is not None:
            current = self._ram_high_water_per_process.get(message.process_id, 0)
            self._ram_high_water_per_process[message.process_id] = max(current, metrics.ram_used_high_water_mb)

        if message.is_alchemy:
            # Alchemy forms never pass through the image job tracker, so their record
            # is complete as soon as the child reports.
            self._jobs.append(
                JobMetricsRecord(
                    job_id=message.job_id,
                    is_alchemy=True,
                    phase_metrics=metrics,
                ),
            )
        else:
            # Image jobs finalize later; hold the phase metrics for correlation.
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

        self._jobs.append(
            JobMetricsRecord(
                job_id=job_id,
                faulted=completed_job_info.state == GENERATION_STATE.faulted,
                time_popped=time_popped,
                stage_timestamps=stage_timestamps,
                queue_wait_seconds=queue_wait,
                e2e_seconds=e2e,
                safety_seconds=safety,
                phase_metrics=self._phase_metrics_by_job.pop(job_id, None),
                model_name=model_name,
                steps=payload.ddim_steps,
                width=payload.width,
                height=payload.height,
                loras_count=len(payload.loras) if payload.loras else 0,
                tis_count=len(payload.tis) if payload.tis else 0,
                control_type=control_type,
                post_processing=[str(post_proc_step) for post_proc_step in payload.post_processing],
                hires_fix=payload.hires_fix,
            ),
        )

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

    def snapshot(
        self,
        *,
        num_process_recoveries: int = 0,
        num_job_slowdowns: int = 0,
        time_spent_no_jobs_available: float = 0.0,
        disk_min_free_bytes: dict[str, int] | None = None,
        phase: str = "",
        process_state_summary: str = "",
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
            time_spent_no_jobs_available=time_spent_no_jobs_available,
            process_crash_events=list(self._crash_events),
            phase=phase,
            process_state_summary=process_state_summary,
        )
