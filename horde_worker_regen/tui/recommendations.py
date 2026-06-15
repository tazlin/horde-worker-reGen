"""Turn a live worker snapshot into actionable insights.

A light, dependency-free analysis (no benchmark or hordelib import) that encodes the operational
levers learned from the duty-cycle and memory work: low GPU duty cycle, VRAM pressure, fault rate,
idle time, and configuration mismatches. The benchmark remains the authoritative capability sweep;
these are the at-a-glance, in-the-moment hints.
"""

from __future__ import annotations

import dataclasses
import enum

from horde_worker_regen.process_management.supervisor_channel import WorkerStateSnapshot


class Severity(enum.IntEnum):
    """Insight severity, ordered so the most urgent sorts first."""

    CRITICAL = 0
    WARNING = 1
    SUGGESTION = 2
    INFO = 3

    @property
    def label(self) -> str:
        """A short uppercase label."""
        return self.name

    @property
    def colour(self) -> str:
        """A Rich colour for this severity."""
        return {
            Severity.CRITICAL: "bold white on red",
            Severity.WARNING: "yellow",
            Severity.SUGGESTION: "cyan",
            Severity.INFO: "grey62",
        }[self]


@dataclasses.dataclass(frozen=True)
class Recommendation:
    """One actionable insight derived from the worker state."""

    severity: Severity
    title: str
    detail: str


_LOW_DUTY_CYCLE = 50.0
_HIGH_VRAM_FRACTION = 0.92
_HIGH_FAULT_RATE = 0.10
_BIG_GPU_VRAM_MB = 20_000


def analyze(snapshot: WorkerStateSnapshot) -> list[Recommendation]:
    """Return insights for a worker-state snapshot, most severe first."""
    recommendations: list[Recommendation] = []
    config = snapshot.config

    if snapshot.too_many_consecutive_failed_jobs:
        recommendations.append(
            Recommendation(
                Severity.CRITICAL,
                "Too many consecutive failed jobs",
                "The worker has paused after repeated failures. Check the Logs view for the root cause "
                "(often a bad model, OOM, or a misconfiguration).",
            ),
        )

    _check_fault_rate(snapshot, recommendations)
    _check_vram_pressure(snapshot, recommendations)
    _check_duty_cycle(snapshot, recommendations)
    _check_idle(snapshot, recommendations)
    _check_memory_mode(snapshot, recommendations)

    if config.extra_slow_worker and config.max_batch > 1:
        recommendations.append(
            Recommendation(
                Severity.SUGGESTION,
                "Extra-slow worker with batching",
                "extra_slow_worker is on but max_batch > 1; set max_batch to 1 to avoid long batch jobs.",
            ),
        )

    if snapshot.maintenance_mode:
        recommendations.append(
            Recommendation(
                Severity.INFO,
                "Worker is paused / in maintenance",
                "No new jobs are being popped. Resume from the worker controls to continue.",
            ),
        )

    if not recommendations:
        recommendations.append(
            Recommendation(Severity.INFO, "No issues detected", "The worker looks healthy."),
        )

    recommendations.sort(key=lambda item: item.severity)
    return recommendations


def _check_fault_rate(snapshot: WorkerStateSnapshot, out: list[Recommendation]) -> None:
    """Flag a high job fault rate."""
    total = snapshot.num_jobs_submitted + snapshot.num_jobs_faulted
    if total >= 10 and snapshot.num_jobs_faulted / total > _HIGH_FAULT_RATE:
        rate = snapshot.num_jobs_faulted / total * 100
        out.append(
            Recommendation(
                Severity.WARNING,
                f"High fault rate ({rate:.0f}%)",
                f"{snapshot.num_jobs_faulted} of {total} jobs faulted. Inspect the logs; consider dropping "
                "VRAM-heavy models or lowering max_power/max_batch.",
            ),
        )


def _check_vram_pressure(snapshot: WorkerStateSnapshot, out: list[Recommendation]) -> None:
    """Flag processes whose VRAM high-water is close to the device total."""
    for process in snapshot.processes:
        if process.total_vram_mb <= 0:
            continue
        peak = max(process.vram_used_high_water_mb, process.vram_usage_mb)
        if peak / process.total_vram_mb >= _HIGH_VRAM_FRACTION:
            out.append(
                Recommendation(
                    Severity.WARNING,
                    f"VRAM pressure on process {process.process_id}",
                    f"Peak VRAM {peak} MB of {process.total_vram_mb} MB. Reduce max_batch/max_power, or "
                    "disable safety_on_gpu to free headroom and avoid out-of-memory faults.",
                ),
            )
            return


def _check_duty_cycle(snapshot: WorkerStateSnapshot, out: list[Recommendation]) -> None:
    """Flag a low GPU duty cycle while work is available."""
    duty = snapshot.gpu_utilization_mean_percent
    work_present = snapshot.jobs_in_progress > 0 or snapshot.jobs_pending_inference > 0
    if duty is not None and work_present and duty < _LOW_DUTY_CYCLE:
        detail = (
            f"GPU duty cycle is {duty:.0f}% with work queued. The GPU is idling between jobs, usually "
            "RAM→VRAM reloads. "
        )
        if not snapshot.config.high_memory_mode:
            detail += "Enabling high_memory_mode keeps models resident and cuts reload thrash. "
        if snapshot.config.max_threads == 1 or snapshot.config.queue_size == 0:
            detail += "Raising max_threads/queue_size lets a second job stage while one samples."
        out.append(Recommendation(Severity.SUGGESTION, f"Low GPU duty cycle ({duty:.0f}%)", detail.strip()))


def _check_idle(snapshot: WorkerStateSnapshot, out: list[Recommendation]) -> None:
    """Flag substantial time spent with no jobs available."""
    if snapshot.time_spent_no_jobs_available > 600 and not snapshot.maintenance_mode:
        minutes = snapshot.time_spent_no_jobs_available / 60
        out.append(
            Recommendation(
                Severity.SUGGESTION,
                "Frequently idle (low demand)",
                f"~{minutes:.0f} minutes without jobs this session. Offering more models or raising "
                "max_power can increase the jobs you receive.",
            ),
        )


def _check_memory_mode(snapshot: WorkerStateSnapshot, out: list[Recommendation]) -> None:
    """Suggest high_memory_mode on a big GPU that has it disabled."""
    config = snapshot.config
    if config.high_memory_mode or config.extra_slow_worker:
        return
    biggest = max((process.total_vram_mb for process in snapshot.processes), default=0)
    if biggest > _BIG_GPU_VRAM_MB and config.max_threads == 1:
        out.append(
            Recommendation(
                Severity.SUGGESTION,
                "Large GPU without high_memory_mode",
                f"This GPU reports {biggest} MB VRAM but high_memory_mode is off. Enabling it keeps models "
                "resident and improves throughput.",
            ),
        )
