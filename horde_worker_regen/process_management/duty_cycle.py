"""Shared duty-cycle math and idle attribution, used by both the live worker and the benchmark.

"Duty cycle" is GPU core uptime: the fraction of wall-clock the GPU is actually doing work under
load. The headline figure is the measured NVML mean utilization (see ``utils/gpu_monitor.py``); this
module's job is to *explain the gap* between that and 100% by attributing the lost time to phases of
the job lifecycle, and crucially to separate idle the worker cannot help (the horde had no jobs to
hand it) from idle it can (hand-off gaps between jobs).

The same summary feeds the worker's periodic threshold logs and the benchmark soak's advisory, so an
operator reads one consistent "where did the time go" story whether they are tailing a live worker or
a benchmark report.

Kept torch-free (stdlib + pydantic only) so the benchmark planner and any orchestrator can import it
without dragging the inference stack. It operates on :class:`JobMetricsRecord`, which already lives in
``run_metrics`` and is already imported by the benchmark report.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from horde_worker_regen.process_management.run_metrics import JobMetricsRecord

PHASE_ORDER = [
    "queue_wait",
    "model_unload",
    "disk_load",
    "vram_load",
    "sampling",
    "vae",
    "encode",
    "graph_overhead",
    "other_inference",
    "safety",
    "submit",
]
"""The pipeline phases a job's wall time is attributed to, in lifecycle order."""

GPU_BUSY_PHASES = ("vram_load", "sampling", "vae", "encode")
"""Phases that put the GPU core to work. Everything else is idle hand-off the worker can shrink."""

_EFFICIENCY_GAP_PHASES = (
    "queue_wait",
    "model_unload",
    "disk_load",
    "graph_overhead",
    "other_inference",
    "safety",
    "submit",
)
"""Non-GPU phases inside a job's lifecycle, ranked to name the biggest worker-side time sinks."""

# hordelib phase_seconds keys folded into the named buckets below, peeled out of the old catch-all
# ``other_inference`` residual so the breakdown names encode and graph framing instead of hiding them.
# Absent on older engines (the keys simply do not appear), in which case the buckets are omitted and
# ``other_inference`` retains that time exactly as before.
_ENCODE_KEYS = ("clip_encode", "vae_encode")
"""GPU encode phases (prompt CLIP encode and, for img2img, VAE encode)."""

_GRAPH_OVERHEAD_KEYS = ("pipeline_setup", "pipeline_validate", "pipeline_finalize")
"""ComfyUI graph framing around execution: build/validate the prompt graph, then tear down."""

_CHURN_LABELS = {
    "model_swap": "model swaps",
    "vram_eviction": "VRAM evictions",
    "process_cycle": "process cycles",
}
"""Human-friendly names for the between-jobs reload/respawn churn counts shown on the duty line."""

_PHASE_LABELS = {
    "queue_wait": "queue wait",
    "model_unload": "model unload (eviction)",
    "disk_load": "model load (disk)",
    "vram_load": "model load (VRAM)",
    "sampling": "sampling",
    "vae": "VAE decode",
    "encode": "prompt/image encode",
    "graph_overhead": "graph build/teardown",
    "other_inference": "node/IPC overhead",
    "safety": "safety",
    "submit": "submit",
}
"""Human-friendly phase names for logs and reports (operators do not think in stage enum keys)."""


def median(values: list[float]) -> float | None:
    """Return the median of ``values``, or None when empty."""
    return statistics.median(values) if values else None


def phase_breakdown(jobs: list[JobMetricsRecord]) -> dict[str, float]:
    """Median per-job seconds in each pipeline phase, to show where a typical job's time goes.

    Built from worker stage timestamps (queue_wait/inference/safety/submit) and the child's
    phase metrics (disk->RAM, RAM->VRAM, sampling, VAE); ``other_inference`` is the inference
    time not otherwise accounted for (graph build, prompt encode, image encode, IPC).
    """
    samples: dict[str, list[float]] = {phase: [] for phase in PHASE_ORDER}

    for job in jobs:
        if job.is_alchemy:
            continue
        stamps = job.stage_timestamps or {}
        if job.queue_wait_seconds is not None:
            samples["queue_wait"].append(job.queue_wait_seconds)

        inference_start = stamps.get("INFERENCE_IN_PROGRESS")
        inference_end = stamps.get("PENDING_SAFETY_CHECK")
        submit_ready = stamps.get("PENDING_SUBMIT")
        finalized = stamps.get("FINALIZED")

        if submit_ready is not None and finalized is not None:
            samples["submit"].append(max(0.0, finalized - submit_ready))
        if inference_end is not None and submit_ready is not None:
            samples["safety"].append(max(0.0, submit_ready - inference_end))

        pm = job.phase_metrics
        if pm is None or inference_start is None or inference_end is None:
            continue
        inference_total = max(0.0, inference_end - inference_start)

        disk_load = sum(m.duration_seconds for m in pm.model_loads if m.phase == "disk_to_ram")
        vram_load = sum(m.duration_seconds for m in pm.model_loads if m.phase == "ram_to_vram")
        sampling = pm.sampling.duration_seconds if pm.sampling is not None else 0.0
        # Only the VAE-decode phase belongs in the vae bucket. Summing all of phase_seconds
        # (clip_encode plus pipeline_setup/validate/execute/finalize) folded the whole ComfyUI
        # execution into "vae" (pipeline_execute alone covers most of the job) which mislabelled
        # nearly every job as VAE-bound and zeroed other_inference. The residual subtraction below
        # already attributes that overhead (graph build, prompt encode, IPC) to other_inference.
        phase_seconds = pm.phase_seconds or {}
        vae = phase_seconds.get("vae_decode", 0.0)
        encode = sum(phase_seconds.get(key, 0.0) for key in _ENCODE_KEYS)
        graph_overhead = sum(phase_seconds.get(key, 0.0) for key in _GRAPH_OVERHEAD_KEYS)
        other = max(0.0, inference_total - disk_load - vram_load - sampling - vae - encode - graph_overhead)

        samples["disk_load"].append(disk_load)
        samples["vram_load"].append(vram_load)
        samples["sampling"].append(sampling)
        samples["vae"].append(vae)
        # Only contribute encode/graph_overhead when the engine actually reported them; otherwise the
        # bucket stays absent (median over an empty list -> dropped) and ``other_inference`` keeps the
        # time, preserving behaviour against engines that predate these phase_seconds keys.
        if any(key in phase_seconds for key in _ENCODE_KEYS):
            samples["encode"].append(encode)
        if any(key in phase_seconds for key in _GRAPH_OVERHEAD_KEYS):
            samples["graph_overhead"].append(graph_overhead)
        samples["other_inference"].append(other)
        # Between-jobs VRAM eviction cost (engine-reported). It happens outside this job's inference
        # window, so it is surfaced as its own worker-side gap, never subtracted from the residual.
        if "model_unload" in phase_seconds:
            samples["model_unload"].append(phase_seconds["model_unload"])

    return {phase: value for phase in PHASE_ORDER if (value := median(samples[phase])) is not None}


def span_derived_busy_ratio(breakdown: dict[str, float]) -> float | None:
    """GPU-touching phases (vram_load + sampling + vae + encode) / the whole per-job wall, or None.

    A phase-attributed duty-cycle proxy that needs no tracing backend: it says what share of a
    typical job's lifecycle is actual GPU work versus idle hand-off (queue_wait, disk_load,
    graph_overhead, other_inference, safety, submit). The NVML mean is still the headline; this
    explains it, and on backends with no NVML telemetry it stands in as the duty-cycle estimate.
    """
    total = sum(breakdown.values())
    if total <= 0:
        return None
    busy = sum(breakdown.get(phase, 0.0) for phase in GPU_BUSY_PHASES)
    return busy / total


def top_efficiency_gaps(breakdown: dict[str, float], *, limit: int = 2) -> list[tuple[str, float]]:
    """The largest worker-side (non-GPU) phases in ``breakdown``, biggest first, for a concise message."""
    gaps = [(phase, breakdown[phase]) for phase in _EFFICIENCY_GAP_PHASES if breakdown.get(phase, 0.0) > 0.0]
    gaps.sort(key=lambda entry: entry[1], reverse=True)
    return gaps[:limit]


def format_phase_gaps(breakdown: dict[str, float], *, limit: int = 2) -> str:
    """Render the top worker-side gaps as ``"model load (disk) 1.8s/job, safety 0.9s/job"`` (or "")."""
    return ", ".join(
        f"{_PHASE_LABELS.get(phase, phase)} {seconds:.1f}s/job"
        for phase, seconds in top_efficiency_gaps(breakdown, limit=limit)
    )


class DutyCycleSummary(BaseModel):
    """A duty-cycle reading plus the attribution that explains it, over one observation window."""

    window_seconds: float
    completed_jobs: int = 0
    nvml_mean_percent: float | None = None
    """Measured mean GPU core utilization over the window (the headline duty cycle), when measured."""
    nvml_busy_fraction: float | None = None
    """Fraction of samples at or above the busy threshold (any GPU activity at all), when measured."""
    phase_breakdown_seconds: dict[str, float] = {}
    """Median per-job seconds in each phase (the "where the time goes" view)."""
    span_derived_busy_ratio: float | None = None
    """Phase-derived duty-cycle proxy; the headline fallback where NVML cannot report."""
    no_jobs_available_seconds: float = 0.0
    """Wall time in the window the worker sat idle because the horde offered no jobs."""
    no_jobs_available_fraction: float | None = None
    """``no_jobs_available_seconds`` as a fraction of the window, or None when the window is unknown."""
    churn_counts: dict[str, int] = {}
    """Count of each between-jobs reload/respawn event in the window, keyed by churn kind. The *rate*
    of these (model swaps, VRAM evictions, process cycles) is what inflates queue_wait, so naming them
    on the duty line points the operator at the reload churn behind a low duty cycle."""

    def effective_duty_percent(self) -> float | None:
        """The duty-cycle number to compare against thresholds: NVML mean, else the phase proxy."""
        if self.nvml_mean_percent is not None:
            return self.nvml_mean_percent
        if self.span_derived_busy_ratio is not None:
            return self.span_derived_busy_ratio * 100.0
        return None

    def headline_source(self) -> str:
        """Which signal backs ``effective_duty_percent`` (so a reader knows how it was measured)."""
        if self.nvml_mean_percent is not None:
            return "nvml"
        if self.span_derived_busy_ratio is not None:
            return "phase-derived"
        return "none"

    def is_demand_limited(self, *, min_fraction: float = 0.10) -> bool:
        """Whether idle is meaningfully due to the horde having no work (not worker inefficiency)."""
        return self.no_jobs_available_fraction is not None and self.no_jobs_available_fraction >= min_fraction

    def top_efficiency_gaps(self, *, limit: int = 2) -> list[tuple[str, float]]:
        """The largest worker-side (non-GPU) per-job phases, biggest first, for a concise message."""
        return top_efficiency_gaps(self.phase_breakdown_seconds, limit=limit)

    def format_gap_summary(self, *, limit: int = 2) -> str:
        """Render the top worker-side gaps as ``"model load (disk) 1.8s/job, safety 0.9s/job"``."""
        return format_phase_gaps(self.phase_breakdown_seconds, limit=limit)

    def format_churn_summary(self) -> str:
        """Render non-zero churn as ``"23 model swaps, 18 VRAM evictions"`` (largest first), or "".

        The biggest reload/respawn driver leads, so the operator sees what to suppress (e.g. enable
        residency) first. Empty when no churn occurred in the window.
        """
        items = sorted(
            ((kind, count) for kind, count in self.churn_counts.items() if count > 0),
            key=lambda entry: entry[1],
            reverse=True,
        )
        return ", ".join(f"{count} {_CHURN_LABELS.get(kind, kind)}" for kind, count in items)


def summarize_duty_cycle(
    jobs: list[JobMetricsRecord],
    *,
    window_seconds: float,
    time_spent_no_jobs_available: float = 0.0,
    nvml_mean_percent: float | None = None,
    nvml_busy_fraction: float | None = None,
    churn_counts: dict[str, int] | None = None,
) -> DutyCycleSummary:
    """Distill duty cycle and idle attribution for ``jobs`` observed over ``window_seconds``.

    ``jobs`` are the per-job records that fall in the window (the caller filters a live worker's
    accumulating list; the benchmark passes a level's whole set). ``time_spent_no_jobs_available``
    is the demand-limited idle attributable to this window, ``nvml_*`` are the measured GPU figures
    when a backend can report them (None otherwise, in which case the phase proxy carries the
    headline), and ``churn_counts`` is the per-window count of each reload/respawn event.
    """
    breakdown = phase_breakdown(jobs)
    completed = sum(1 for job in jobs if not job.is_alchemy and not job.faulted)

    no_jobs_fraction: float | None = None
    if window_seconds > 0:
        no_jobs_fraction = max(0.0, min(1.0, time_spent_no_jobs_available / window_seconds))

    return DutyCycleSummary(
        window_seconds=window_seconds,
        completed_jobs=completed,
        nvml_mean_percent=nvml_mean_percent,
        nvml_busy_fraction=nvml_busy_fraction,
        phase_breakdown_seconds=breakdown,
        span_derived_busy_ratio=span_derived_busy_ratio(breakdown),
        no_jobs_available_seconds=time_spent_no_jobs_available,
        no_jobs_available_fraction=no_jobs_fraction,
        churn_counts=dict(churn_counts or {}),
    )


__all__ = [
    "GPU_BUSY_PHASES",
    "PHASE_ORDER",
    "DutyCycleSummary",
    "format_phase_gaps",
    "median",
    "phase_breakdown",
    "span_derived_busy_ratio",
    "summarize_duty_cycle",
    "top_efficiency_gaps",
]
