"""Turn a live worker snapshot into findings.

A light, dependency-free analysis (no benchmark or hordelib import) that encodes the operational
levers learned from the duty-cycle and memory work: low GPU duty cycle, VRAM pressure, fault rate,
idle time, configuration mismatches, and what each model earns on its card. The benchmark remains the
authoritative capability sweep; these are the at-a-glance, in-the-moment hints.

Each hint is a :class:`~horde_worker_regen.analysis.finding_kinds.Finding` of a declared kind, so the
dashboard shows it with the same card, badge words and copy rules as a log diagnosis, and the catalogue
documents it beside them. A healthy snapshot yields no findings at all; the view says so itself.

The earning findings read the worker's figures and verdicts from ``WorkerStateSnapshot.card_earning`` and do
not compute any statistic. A share is one snapshot figure over another.
"""

from __future__ import annotations

from horde_worker_regen.analysis.finding_kinds import SEVERITY_ORDER, Finding, FindingKind, Severity
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    CardEarningRow,
    CardEarningSnapshot,
    CardEarningWindowRestart,
    ModelPoolSnapshot,
    WorkerStateSnapshot,
)
from horde_worker_regen.process_management.scheduling.governance.whole_card import PopClaimValueVerdict
from horde_worker_regen.tui.formatters import human_duration
from horde_worker_regen.tui.widgets.earning import (
    join_words,
    plain_span,
    rate_text,
    setting_words,
    share_text,
    verdict_phrase,
)

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
    _check_card_earning(snapshot, findings)

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


def _check_card_earning(snapshot: WorkerStateSnapshot, out: list[Finding]) -> None:
    """Flag each model that earns below the rest of its card, then note a restart of the earning window.

    A model with claim time gets a card whatever its share, because the worker asked the horde only for it during
    that time. A model without a claim gets one only when it took more than half of the card's wall. That is more
    seconds than the rest's rate is read over, which is the wall less the model's seconds.
    """
    earning = snapshot.card_earning
    walls = {card.device_index: card.wall_seconds for card in earning.cards}
    several_cards = len(set(walls) | {row.device_index for row in earning.rows}) > 1
    below_rest = [
        row
        for row in earning.rows
        if row.verdict == PopClaimValueVerdict.BELOW_THE_REST and walls.get(row.device_index, 0.0) > 0
    ]
    claimed = sorted(
        (row for row in below_rest if row.claim_seconds > 0),
        key=lambda row: (-row.claim_seconds, row.device_index),
    )
    for row in claimed:
        out.append(
            _below_rest_finding(
                snapshot,
                row,
                wall_seconds=walls[row.device_index],
                several_cards=several_cards,
                had_claim=True,
            ),
        )
    unclaimed = sorted(
        (
            row
            for row in below_rest
            if row.claim_seconds <= 0
            and row.attributed_card_seconds > walls[row.device_index] - row.attributed_card_seconds
        ),
        key=lambda row: (-row.attributed_card_seconds, row.device_index),
    )
    for row in unclaimed:
        out.append(
            _below_rest_finding(
                snapshot,
                row,
                wall_seconds=walls[row.device_index],
                several_cards=several_cards,
                had_claim=False,
            ),
        )
    _check_earning_restart(snapshot, out)


def _below_rest_finding(
    snapshot: WorkerStateSnapshot,
    row: CardEarningRow,
    *,
    wall_seconds: float,
    several_cards: bool,
    had_claim: bool,
) -> Finding:
    """Return the finding for one model that earns below the rest of its card.

    Args:
        snapshot: The snapshot the row comes from.
        row: The model's earning row.
        wall_seconds: The card's wall seconds in the window, above 0.
        several_cards: Whether the window has more than one card, so the headline names the card.
        had_claim: Whether a whole-card pop claim for the model stood on the card in the window.
    """
    earning = snapshot.card_earning
    phrase = verdict_phrase(row.verdict, row=row, min_jobs=earning.verdict_min_jobs)
    card_words = f"card {row.device_index}" if several_cards else "its card"
    card_share = share_text(row.attributed_card_seconds / wall_seconds)
    evidence = _earning_evidence(earning, row, now=snapshot.timestamp)
    if had_claim:
        claim_share = share_text(row.claim_seconds / wall_seconds)
        evidence.insert(1, f"It took {card_share} of the card's time.")
        return Finding(
            kind=FindingKind.CLAIMED_MODEL_EARNS_BELOW_REST,
            severity=Severity.SUGGESTION,
            headline=(
                f'"{row.model}" earns {phrase} of {card_words}, and the worker asked the horde only for it for '
                f"{claim_share} of the card's time."
            ),
            evidence=evidence,
        )
    return Finding(
        kind=FindingKind.MODEL_EARNS_BELOW_REST,
        severity=Severity.INFO,
        headline=f'"{row.model}" earns {phrase} of {card_words} and took {card_share} of the card\'s time.',
        evidence=evidence,
    )


def _earning_evidence(earning: CardEarningSnapshot, row: CardEarningRow, *, now: float) -> list[str]:
    """Return the figures behind a below-the-rest finding: the two rates, their bands, the jobs and the window."""
    lines = [
        f"It earns {rate_text(row.model_kudos_per_second)} kudos per second of card time. The rest of the card "
        f"earns {rate_text(row.rest_kudos_per_second)}.",
    ]
    bands = (row.model_band_low, row.model_band_high, row.rest_band_low, row.rest_band_high)
    if all(band is not None for band in bands):
        lines.append(
            f"Its 95% band is {rate_text(row.model_band_low)} to {rate_text(row.model_band_high)}. The rest's band "
            f"is {rate_text(row.rest_band_low)} to {rate_text(row.rest_band_high)}.",
        )
    lines.append(f"Counted from {row.jobs:,} of its jobs on the card.")
    if earning.window_started_at is not None:
        lines.append(f"Measured over the last {plain_span(now - earning.window_started_at)}.")
    return lines


def _check_earning_restart(snapshot: WorkerStateSnapshot, out: list[Finding]) -> None:
    """Note that the earning window restarted, until a model's verdict is measured again.

    A worker that does not serve image jobs gets the note only once it has figures, as with the Stats tab's earning
    section.
    """
    earning = snapshot.card_earning
    restart = earning.last_restart
    if restart is None:
        return
    if not (snapshot.config.dreamer or earning.rows):
        return
    if any(row.verdict != PopClaimValueVerdict.UNMEASURED for row in earning.rows):
        return
    out.append(
        Finding(
            kind=FindingKind.CARD_EARNING_RESTARTED,
            severity=Severity.INFO,
            headline=_restart_headline(restart),
        ),
    )


def _restart_headline(restart: CardEarningWindowRestart) -> str:
    """Return the restart note's headline, naming the changed settings when the worker reported them."""
    if not restart.changed_settings:
        return "The earning figures started again because the settings differed from the last run's."
    names = [setting_words(field_name) for field_name in restart.changed_settings]
    return f"The earning figures started again when {join_words(names)} changed."
