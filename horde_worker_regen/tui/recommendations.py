"""Turn a live worker snapshot into findings.

A light, dependency-free analysis (no benchmark or hordelib import) that encodes the operational
levers learned from the duty-cycle and memory work: low GPU duty cycle, VRAM pressure, fault rate,
idle time, and configuration mismatches. The benchmark remains the authoritative capability sweep;
these are the at-a-glance, in-the-moment hints.

Each hint is a :class:`~horde_worker_regen.analysis.finding_kinds.Finding` of a declared kind, so the
dashboard shows it with the same card, badge words and copy rules as a log diagnosis, and the catalogue
documents it beside them. A healthy snapshot yields no findings at all; the view says so itself.
"""

from __future__ import annotations

from horde_worker_regen.analysis.finding_kinds import SEVERITY_ORDER, Finding, FindingKind, Severity
from horde_worker_regen.process_management.ipc.supervisor_channel import ModelPoolSnapshot, WorkerStateSnapshot
from horde_worker_regen.tui.formatters import human_duration

_LOW_DUTY_CYCLE = 50.0
_HIGH_VRAM_FRACTION = 0.92
_HIGH_FAULT_RATE = 0.10
_IDLE_SECONDS = 600.0

# Pool-off: enough measured session model swaps to make the pool trade relevant.
_POOL_OFF_MODEL_SWAPS = 3
# Pool-on thresholds: a demand reading this old means the ranker is acting on a frozen signal; a seat with this
# many charged empty pops (or seated this long with no match) is not matching demand; a seat that
# matched resident work within this window is evidence that the pool is avoiding a cold load at pop time.
_POOL_STALE_DEMAND_SECONDS = 900.0
_POOL_HIGH_EMPTY_POPS = 5
_POOL_UNPRODUCTIVE_SEAT_SECONDS = 600.0
_POOL_FRESH_FULFILLED_SECONDS = 180.0


def analyze(snapshot: WorkerStateSnapshot) -> list[Finding]:
    """Return the findings for a worker-state snapshot, most severe first; empty when nothing is worth saying."""
    findings: list[Finding] = []
    config = snapshot.config

    if snapshot.too_many_consecutive_failed_jobs:
        findings.append(
            Finding(
                kind=FindingKind.CONSECUTIVE_FAILURE_PAUSE,
                severity=Severity.CRITICAL,
                headline="The worker paused asking for work after several failed jobs in a row.",
                action_addendum="Check the Logs tab for the cause.",
            ),
        )

    _check_fault_rate(snapshot, findings)
    _check_vram_pressure(snapshot, findings)
    _check_duty_cycle(snapshot, findings)
    _check_idle(snapshot, findings)
    _check_model_pool(snapshot, findings)

    if config.extra_slow_worker and config.max_batch > 1:
        findings.append(
            Finding(
                kind=FindingKind.EXTRA_SLOW_BATCHING,
                severity=Severity.SUGGESTION,
                headline=f"`extra_slow_worker` is on and `max_batch` is {config.max_batch}.",
            ),
        )

    if snapshot.maintenance_mode:
        findings.append(
            Finding(
                kind=FindingKind.FORCED_MAINTENANCE,
                severity=Severity.INFO,
                title_override="The worker is in maintenance mode",
                headline="No new jobs are taken while maintenance is on.",
                action_addendum="Resume from the worker controls when you are ready.",
            ),
        )

    findings.sort(key=lambda finding: SEVERITY_ORDER[finding.severity])
    return findings


def _check_fault_rate(snapshot: WorkerStateSnapshot, out: list[Finding]) -> None:
    """Flag a high job fault rate."""
    total = snapshot.num_jobs_submitted + snapshot.num_jobs_faulted
    if total >= 10 and snapshot.num_jobs_faulted / total > _HIGH_FAULT_RATE:
        rate = snapshot.num_jobs_faulted / total * 100
        out.append(
            Finding(
                kind=FindingKind.FAULT_RATE,
                severity=Severity.WARNING,
                headline=f"{snapshot.num_jobs_faulted} of {total} jobs failed this session, {rate:.0f}%.",
            ),
        )


def _check_vram_pressure(snapshot: WorkerStateSnapshot, out: list[Finding]) -> None:
    """Flag the first process whose VRAM high-water is close to the device total."""
    for process in snapshot.processes:
        if process.total_vram_mb <= 0:
            continue
        peak = max(process.vram_used_high_water_mb, process.vram_usage_mb)
        if peak / process.total_vram_mb >= _HIGH_VRAM_FRACTION:
            out.append(
                Finding(
                    kind=FindingKind.VRAM_PRESSURE,
                    severity=Severity.WARNING,
                    headline=(
                        f"Process {process.process_id} peaked at {peak} MB of the card's {process.total_vram_mb} MB."
                    ),
                ),
            )
            return


def _check_duty_cycle(snapshot: WorkerStateSnapshot, out: list[Finding]) -> None:
    """Flag a low GPU duty cycle while work is available."""
    duty = snapshot.gpu_utilization_mean_percent
    work_present = snapshot.jobs_in_progress > 0 or snapshot.jobs_pending_inference > 0
    if duty is None or not work_present or duty >= _LOW_DUTY_CYCLE:
        return
    if snapshot.config.max_threads == 1 or snapshot.config.queue_size == 0:
        action = "Raise `max_threads` or `queue_size` so a second job can load while one runs."
    else:
        action = "Put models on an SSD and keep the model list short, so reloads between jobs are quick."
    out.append(
        Finding(
            kind=FindingKind.LOW_DUTY_CYCLE,
            severity=Severity.SUGGESTION,
            headline=f"The GPU was busy {duty:.0f}% of the time with work waiting.",
            action_addendum=action,
        ),
    )


def _check_idle(snapshot: WorkerStateSnapshot, out: list[Finding]) -> None:
    """Flag substantial time spent with no jobs available."""
    if snapshot.time_spent_no_jobs_available > _IDLE_SECONDS and not snapshot.maintenance_mode:
        minutes = snapshot.time_spent_no_jobs_available / 60
        out.append(
            Finding(
                kind=FindingKind.LOW_DEMAND_IDLE,
                severity=Severity.SUGGESTION,
                headline=f"About {minutes:.0f} minutes of this session had no jobs available.",
            ),
        )


def _check_model_pool(snapshot: WorkerStateSnapshot, out: list[Finding]) -> None:
    """Route to the pool-off or pool-on advisors depending on whether the fixed model pool is active."""
    pool = snapshot.model_pool
    if pool is None or not pool.enabled:
        _check_pool_off_diversity(snapshot, out)
        return
    _check_pool_on(pool, out)


def _check_pool_off_diversity(snapshot: WorkerStateSnapshot, out: list[Finding]) -> None:
    """With the pool off, offer the pool trade only after measured model-swap churn.

    Distinct recent models are not proof of a swap because multiple processes may keep them resident. The
    session churn counter records actual preloads that displaced another model, which makes the finding
    evidence-based while preserving model variety as a legitimate operator preference.
    """
    sample = snapshot.latest_stats_sample
    model_swaps = sample.churn_counts.get("model_swap", 0) if sample is not None else 0
    if model_swaps >= _POOL_OFF_MODEL_SWAPS:
        out.append(
            Finding(
                kind=FindingKind.MODEL_POOL_OFF_SWAPS,
                severity=Severity.SUGGESTION,
                headline=f"This session recorded {model_swaps} model swaps with the model pool off.",
            ),
        )


def _check_pool_on(pool: ModelPoolSnapshot, out: list[Finding]) -> None:
    """Read live seats and demand age, flagging issues and noting measured resident matches."""
    before = len(out)
    _check_pool_demand_staleness(pool, out)
    _check_pool_unproductive_seats(pool, out)
    if len(out) == before:
        _note_pool_resident_matches(pool, out)


def _check_pool_demand_staleness(pool: ModelPoolSnapshot, out: list[Finding]) -> None:
    """Flag a demand reading old enough that the ranker is ranking against a frozen signal."""
    age = pool.demand_age_seconds
    if age is not None and age >= _POOL_STALE_DEMAND_SECONDS:
        out.append(
            Finding(
                kind=FindingKind.MODEL_POOL_STALE_DEMAND,
                severity=Severity.WARNING,
                headline=f"The pool's demand reading last refreshed {human_duration(age)} ago.",
            ),
        )


def _check_pool_unproductive_seats(pool: ModelPoolSnapshot, out: list[Finding]) -> None:
    """Flag seats that keep taking empty fixed-lane pops or have matched nothing since seating."""
    flagged: list[str] = []
    for seat in pool.seats:
        if seat.model is None or seat.pending_model is not None:
            continue
        seated_long_without_work = (
            seat.last_fulfilled_age_seconds is None and (seat.dwell_seconds or 0.0) >= _POOL_UNPRODUCTIVE_SEAT_SECONDS
        )
        if seated_long_without_work or seat.empty_pops >= _POOL_HIGH_EMPTY_POPS:
            flagged.append(f'"{seat.model}"')
    if flagged:
        out.append(
            Finding(
                kind=FindingKind.MODEL_POOL_UNPRODUCTIVE_SEATS,
                severity=Severity.SUGGESTION,
                headline=(
                    f"The seats for {', '.join(flagged)} keep getting empty requests or have matched nothing "
                    "since seating."
                ),
            ),
        )


def _note_pool_resident_matches(pool: ModelPoolSnapshot, out: list[Finding]) -> None:
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
            Finding(
                kind=FindingKind.MODEL_POOL_RESIDENT_MATCHES,
                severity=Severity.INFO,
                headline=(
                    f"{len(resident_matches)} seats matched a job while their model was already loaded in the "
                    "last few minutes."
                ),
            ),
        )
