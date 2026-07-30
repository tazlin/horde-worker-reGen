"""Turn a live worker snapshot into actionable insights.

A light, dependency-free analysis (no benchmark or hordelib import) that encodes the operational
levers learned from the duty-cycle and memory work: low GPU duty cycle, VRAM pressure, fault rate,
idle time, and configuration mismatches. The benchmark remains the authoritative capability sweep;
these are the at-a-glance, in-the-moment hints.
"""

from __future__ import annotations

import dataclasses
import enum

from horde_worker_regen.process_management.ipc.supervisor_channel import ModelPoolSnapshot, WorkerStateSnapshot
from horde_worker_regen.tui.formatters import human_duration


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

# Pool-off: enough measured session model swaps to make the pool trade relevant.
_POOL_OFF_MODEL_SWAPS = 3
# Pool-on thresholds: a demand reading this old means the ranker is acting on a frozen signal; a seat with this
# many charged empty pops (or seated this long with no match) is not matching demand; a seat that
# matched resident work within this window is evidence that the pool is avoiding a cold load at pop time.
_POOL_STALE_DEMAND_SECONDS = 900.0
_POOL_HIGH_EMPTY_POPS = 5
_POOL_UNPRODUCTIVE_SEAT_SECONDS = 600.0
_POOL_FRESH_FULFILLED_SECONDS = 180.0


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
    _check_model_pool(snapshot, recommendations)

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


def _check_model_pool(snapshot: WorkerStateSnapshot, out: list[Recommendation]) -> None:
    """Route to the pool-off or pool-on advisors depending on whether the fixed model pool is active."""
    pool = snapshot.model_pool
    if pool is None or not pool.enabled:
        _check_pool_off_diversity(snapshot, out)
        return
    _check_pool_on(snapshot, pool, out)


def _check_pool_off_diversity(snapshot: WorkerStateSnapshot, out: list[Recommendation]) -> None:
    """With the pool off, offer the pool trade only after measured model-swap churn.

    Distinct recent models are not proof of a swap because multiple processes may keep them resident. The
    session churn counter records actual preloads that displaced another model, which makes the recommendation
    evidence-based while preserving model variety as a legitimate operator preference.
    """
    sample = snapshot.latest_stats_sample
    model_swaps = sample.churn_counts.get("model_swap", 0) if sample is not None else 0
    if model_swaps >= _POOL_OFF_MODEL_SWAPS:
        out.append(
            Recommendation(
                Severity.SUGGESTION,
                f"Model pool off after {model_swaps} model swaps",
                "This session has recorded model preloads that displaced another resident model. Enabling the "
                "model pool (or its demand-following preset) can bias pops toward a smaller seat set and reduce "
                "that churn when the seats stay resident, while trading away some model variety. This is a genuine "
                "preference: leave the pool off if serving that variety is the point.",
            ),
        )


def _check_pool_on(
    snapshot: WorkerStateSnapshot,
    pool: ModelPoolSnapshot,
    out: list[Recommendation],
) -> None:
    """Read live seats and demand age, flagging issues and noting measured resident matches."""
    before = len(out)
    _check_pool_demand_staleness(pool, out)
    _check_pool_unproductive_seats(pool, out)
    if len(out) == before:
        _note_pool_resident_matches(pool, out)


def _check_pool_demand_staleness(pool: ModelPoolSnapshot, out: list[Recommendation]) -> None:
    """Flag a demand reading old enough that the ranker is ranking against a frozen signal."""
    age = pool.demand_age_seconds
    if age is not None and age >= _POOL_STALE_DEMAND_SECONDS:
        out.append(
            Recommendation(
                Severity.WARNING,
                "Model pool demand reading is stale",
                f"The pool's demand ranking last refreshed {human_duration(age)} ago, so the ranker is flying "
                "blind and has frozen rotations until a fresh reading arrives. Check the worker's connectivity to "
                "the horde demand endpoint; while the signal is stale it holds the current seats rather than "
                "re-ranking them.",
            ),
        )


def _check_pool_unproductive_seats(pool: ModelPoolSnapshot, out: list[Recommendation]) -> None:
    """Flag seats that keep taking empty fixed-lane pops or have matched nothing since seating."""
    flagged: list[str] = []
    for seat in pool.seats:
        if seat.model is None or seat.pending_model is not None:
            continue
        seated_long_without_work = (
            seat.last_fulfilled_age_seconds is None and (seat.dwell_seconds or 0.0) >= _POOL_UNPRODUCTIVE_SEAT_SECONDS
        )
        if seated_long_without_work or seat.empty_pops >= _POOL_HIGH_EMPTY_POPS:
            flagged.append(seat.model)
    if flagged:
        out.append(
            Recommendation(
                Severity.SUGGESTION,
                "Model pool seats are not matching demand",
                f"Seat(s) {', '.join(flagged)} keep taking fixed-lane pops that come back empty (or have matched "
                "no pops since seating), which is what charges a seat toward demotion. If this persists, the "
                "horde "
                "has little demand for those models on this card: review your pins, or let the demand ranker "
                "rotate them out (model_pool.ranker_enabled) so the seats go to models with live demand.",
            ),
        )


def _note_pool_resident_matches(pool: ModelPoolSnapshot, out: list[Recommendation]) -> None:
    """Note recent pop matches that were resident when accepted."""
    resident_matches = [
        seat
        for seat in pool.seats
        if seat.model is not None
        and seat.pending_model is None
        and seat.last_fulfilled_age_seconds is not None
        and seat.last_fulfilled_age_seconds <= _POOL_FRESH_FULFILLED_SECONDS
        and seat.last_match_was_resident is True
    ]
    if resident_matches:
        out.append(
            Recommendation(
                Severity.INFO,
                "Model pool is receiving resident matches",
                f"{len(resident_matches)} seat(s) matched a pop while already resident within the last few minutes. "
                "That avoided a cold model load at acceptance time; job completion is tracked separately.",
            ),
        )
