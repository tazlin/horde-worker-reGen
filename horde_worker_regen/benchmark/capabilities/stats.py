"""Distill a raw harness run into the criteria-relevant statistics.

This is the pure measurement seam shared by every driver: the warm/in-process path hands a
:class:`~horde_worker_regen.harness.HarnessResult` straight to
:func:`level_stats_from_harness_result`, and the on-disk subprocess path reconstitutes the same
counts and metrics and feeds :func:`level_stats_from_metrics`. Keeping the assembly here (rather
than in the report layer) lets the capability probes and their tests reuse it without importing
the report/recommendation machinery, and keeps it torch-free.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Protocol

from horde_worker_regen.benchmark.criteria import LevelStats
from horde_worker_regen.process_management.resources.duty_cycle import (
    phase_breakdown,
    span_derived_busy_ratio,
)

if TYPE_CHECKING:
    from horde_worker_regen.harness import HarnessResult
    from horde_worker_regen.process_management.resources.run_metrics import JobMetricsRecord, RunMetricsSnapshot


class _LevelCounts(Protocol):
    """The count fields a stats run needs, satisfied by both ``HarnessResult`` and ``HarnessSummary``.

    Both the live result dataclass and its JSON-friendly on-disk summary expose these identically, so
    a single assembly serves the warm and subprocess paths without a shared base class.
    """

    num_jobs_expected: int
    num_jobs_completed: int
    num_jobs_faulted: int
    num_alchemy_forms_expected: int
    num_alchemy_forms_completed: int
    num_alchemy_forms_faulted: int
    audit_failures: list[str]
    timed_out: bool


def _percentile(values: list[float], fraction: float) -> float | None:
    """Return the value at ``fraction`` of the sorted sample (nearest-rank), or None if empty."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _its_retention(jobs: list[JobMetricsRecord]) -> float | None:
    """Median sampling rate of the second half of completed image jobs ÷ that of the first half.

    Ordered by completion time, this catches throughput decay over a sustained run (thermal
    throttling, a VRAM/RAM leak, queue backpressure). Returns None without enough samples.
    """
    timed: list[tuple[float, float]] = []
    for job in jobs:
        if job.is_alchemy or job.phase_metrics is None or job.phase_metrics.sampling is None:
            continue
        its = job.phase_metrics.sampling.iterations_per_second
        finalized = job.stage_timestamps.get("FINALIZED")
        if its > 0 and finalized is not None:
            timed.append((finalized, its))

    if len(timed) < 4:
        return None

    timed.sort(key=lambda entry: entry[0])
    midpoint = len(timed) // 2
    first_half = statistics.median([its for _, its in timed[:midpoint]])
    second_half = statistics.median([its for _, its in timed[midpoint:]])
    if first_half <= 0:
        return None
    return second_half / first_half


def _post_warmup_vram_reloads(jobs: list[JobMetricsRecord]) -> int | None:
    """RAM->VRAM reloads across image jobs after the first completed one (warm-up excluded).

    Orders jobs by completion and drops the first (the unavoidable cold load), then counts every
    subsequent ``ram_to_vram`` model load. Under expected residency this is 0; a positive count
    means a resident model was evicted and reloaded (residency defeated / stickiness thrash).
    Returns None when no job carried phase metrics.
    """
    timed: list[tuple[float, int]] = []
    for job in jobs:
        if job.is_alchemy or job.phase_metrics is None:
            continue
        finalized = job.stage_timestamps.get("FINALIZED")
        if finalized is None:
            continue
        reloads = sum(1 for load in job.phase_metrics.model_loads if load.phase == "ram_to_vram")
        timed.append((finalized, reloads))

    if not timed:
        return None
    timed.sort(key=lambda entry: entry[0])
    return sum(reloads for _, reloads in timed[1:])


def level_stats_from_metrics(
    counts: _LevelCounts,
    metrics: RunMetricsSnapshot | None,
    *,
    total_vram_mb: int | None = None,
) -> LevelStats:
    """Distill the run counts and metrics snapshot into the criteria-relevant statistics."""
    its_values: list[float] = []
    disk_loads: list[float] = []
    vram_loads: list[float] = []
    queue_waits: list[float] = []
    e2es: list[float] = []
    download_rates: list[float] = []
    vram_high_water: int | None = None

    if metrics is not None:
        for job in metrics.jobs:
            if job.phase_metrics is not None:
                if job.phase_metrics.sampling is not None and job.phase_metrics.sampling.iterations_per_second > 0:
                    its_values.append(job.phase_metrics.sampling.iterations_per_second)
                for load in job.phase_metrics.model_loads:
                    if load.phase == "disk_to_ram":
                        disk_loads.append(load.duration_seconds)
                    else:
                        vram_loads.append(load.duration_seconds)
            if job.queue_wait_seconds is not None:
                queue_waits.append(job.queue_wait_seconds)
            if job.e2e_seconds is not None:
                e2es.append(job.e2e_seconds)
        for download in metrics.downloads:
            if download.success and download.megabytes_per_second > 0:
                download_rates.append(download.megabytes_per_second)
        if metrics.vram_used_high_water_mb_per_process:
            vram_high_water = max(metrics.vram_used_high_water_mb_per_process.values())

    disk_min_free = (
        min(metrics.disk_min_free_bytes.values()) if metrics is not None and metrics.disk_min_free_bytes else None
    )

    level_phase_breakdown = phase_breakdown(metrics.jobs) if metrics is not None else {}

    return LevelStats(
        num_jobs_expected=counts.num_jobs_expected,
        num_jobs_completed=counts.num_jobs_completed,
        num_jobs_faulted=counts.num_jobs_faulted,
        num_alchemy_forms_expected=counts.num_alchemy_forms_expected,
        num_alchemy_forms_completed=counts.num_alchemy_forms_completed,
        num_alchemy_forms_faulted=counts.num_alchemy_forms_faulted,
        num_audit_failures=len(counts.audit_failures),
        num_process_recoveries=metrics.num_process_recoveries if metrics is not None else 0,
        timed_out=counts.timed_out,
        its_p50=statistics.median(its_values) if its_values else None,
        its_min=min(its_values) if its_values else None,
        its_retention_fraction=_its_retention(metrics.jobs) if metrics is not None else None,
        gpu_utilization_mean_percent=metrics.gpu_utilization_mean_percent if metrics is not None else None,
        gpu_utilization_busy_fraction=metrics.gpu_utilization_busy_fraction if metrics is not None else None,
        span_derived_busy_ratio=span_derived_busy_ratio(level_phase_breakdown),
        post_warmup_vram_reloads=(_post_warmup_vram_reloads(metrics.jobs) if metrics is not None else None),
        vram_used_high_water_mb=vram_high_water,
        total_vram_mb=total_vram_mb,
        disk_min_free_bytes=disk_min_free,
        download_mbps_min=min(download_rates) if download_rates else None,
        model_load_disk_seconds_median=statistics.median(disk_loads) if disk_loads else None,
        model_load_vram_seconds_median=statistics.median(vram_loads) if vram_loads else None,
        queue_wait_seconds_p95=_percentile(queue_waits, 0.95),
        e2e_seconds_p95=_percentile(e2es, 0.95),
        phase_breakdown_seconds=level_phase_breakdown,
        time_spent_no_jobs_available=metrics.time_spent_no_jobs_available if metrics is not None else None,
    )


def level_stats_from_harness_result(result: HarnessResult, *, total_vram_mb: int | None = None) -> LevelStats:
    """Distill a live (in-process) harness run into the criteria-relevant statistics."""
    return level_stats_from_metrics(result, result.metrics, total_vram_mb=total_vram_mb)
