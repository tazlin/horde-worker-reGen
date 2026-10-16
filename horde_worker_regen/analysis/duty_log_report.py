"""Epoch-aware GPU duty-cycle report over an appended ``bridge.log``.

``bridge.log`` is appended across worker restarts, so a single file holds many sessions. Comparing a
config change means reading each *session epoch* separately and tying its duty-cycle numbers to the
config that produced them. This tool splits the log on the once-per-session process-manager init
banner, then for each epoch parses the periodic ``GPU duty cycle`` lines (and the reload-churn /
per-job-gap attribution they now carry), the effective worker config, and any disk-pressure warnings,
and prints a compact per-epoch verdict plus the biggest loss buckets.

When stats JSONL is present the stats-backed report leads (:mod:`horde_worker_regen.analysis.session_duty`).
A log named beside it adds each session's clearance-hold digest and idle attribution (every idle card-second
placed on the timeline of the job that ended it) from the matching epoch, and the learned footprint store's
sampling prices follow (:mod:`horde_worker_regen.analysis.footprint_view`).

It imports without the inference stack; usable from the CLI (``horde-duty-report``, or ``horde-log duty``)
or the benchmark/TUI. The grep-friendly log format is the contract: see the regexes below, which mirror
what ``process_manager._log_duty_cycle_summary``, ``status_reporter`` and
``InferenceScheduler._log_clearance_hold`` emit.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from horde_worker_regen.analysis.footprint_view import (
    default_footprint_store_path,
    learned_sampling_footprints,
    render_learned_sampling_footprints,
)
from horde_worker_regen.analysis.governor_signatures import (
    GOVERNOR_ENTER_RE,
    GOVERNOR_EXIT_RE,
    GOVERNOR_LABELS,
)
from horde_worker_regen.analysis.log_signatures import SIGNATURES
from horde_worker_regen.analysis.session_duty import (
    _INFERENCE_LANE_LABEL,
    _LANE_STATE_RE,
    _POST_PROCESS_LANE_LABEL,
    _POST_PROCESSING_LANE_STATES,
    _SAMPLING_LANE_STATES,
    SessionDutyReport,
    _lane_states,
    analyze_stats_sessions,
    read_stats_payloads,
    render_session_duty_report,
    stats_session_local_start,
)
from horde_worker_regen.process_management.scheduling.slot_duty import SlotDutyBucket
from horde_worker_regen.stats_operations import default_stats_dir

_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")
_TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# Once-per-session boundary: the worker's process manager constructs exactly once per launch and logs
# several ``__init__`` lines in a burst at startup. The worker-identity "not yet registered" line is a
# fallback for logs where the manager lines are absent.
_EPOCH_BOUNDARY_RE = re.compile(r"process_management\.process_manager:__init__:")
_EPOCH_BOUNDARY_FALLBACK_RE = re.compile(r"worker_identity:_verify_worker_names_owned:.*is not yet registered")
_EPOCH_COLLAPSE_SECONDS = 30.0
"""Boundary lines closer together than this belong to the same startup burst, not a new epoch."""

# A multi-GPU worker states each card's duty in a parenthetical after the worker-wide headline; the
# headline keeps its shape either way, so the breakdown is an optional group here.
_DUTY_RE = re.compile(
    r"GPU duty cycle (?P<duty>\d+)%(?: \((?P<per_card>card \d+: \d+%(?:, card \d+: \d+%)*)\))? "
    r"over last (?P<window>\d+)s "
    r"\(target \d+%, source=(?P<source>[\w-]+), busy=(?P<busy>\d+)%\)",
)
# Clauses end at the next "; " part separator or the ". " before the jobs/processes context. A plain
# "[^.;]" stop is wrong because the decimal in "16.5s/job" contains a period; the lookahead only ends
# on a period *followed by a space*, which decimals never are.
_GAPS_RE = re.compile(r"biggest worker-side gaps: (?P<gaps>.+?)(?=; |\. |$)")
_CHURN_RE = re.compile(r"reload churn: (?P<churn>.+?)(?=; |\. |$)")
_NO_JOBS_RE = re.compile(r"(?P<frac>\d+)% of the window had no jobs available")
_GAP_ENTRY_RE = re.compile(r"(?P<label>.+?) (?P<seconds>[\d.]+)s/job")
_CHURN_ENTRY_RE = re.compile(r"(?P<count>\d+) (?P<label>[a-zA-Z ]+)")

_CONFIG_RE = re.compile(
    r"unload_models_from_vram_often: (?P<unload>\w+) \| high_performance_mode: (?P<hpm>\w+) \| "
    # high_memory_mode was removed from the worker; the optional group still parses older logs that carry it.
    r"moderate_performance_mode: (?P<mpm>\w+)(?: \| high_memory_mode: (?P<hmm>\w+))?",
)
_IDENTITY_RE = re.compile(
    r"dreamer_name: (?P<name>[^|]+?) \|.*?num_models: (?P<num_models>\d+).*?"
    r"max_power: (?P<max_power>\d+).*?max_threads: (?P<max_threads>\d+) \| "
    r"queue_size: (?P<queue_size>\d+) \| safety_on_gpu: (?P<safety>\w+)",
)
_DISK_RE = re.compile(r"Low disk space on .*?: (?P<free>[\d.]+) GB free")

CLEARANCE_HOLD_RE = re.compile(
    r"Clearance held for process (?P<process>\d+) \((?P<model>.+?)\): (?P<decision>[^;]+); "
    r"(?:candidate (?:(?P<candidate>-?\d+)MB|unpriced) \(child already holds (?:(?P<held>-?\d+)MB|unreported)\) "
    r"vs device free (?:(?P<free>-?\d+)MB|n/a), available (?:(?P<available>-?\d+)MB|n/a), "
    r"reserve (?P<reserve>-?\d+)MB, outstanding reservations (?P<outstanding>-?\d+)MB, "
    r"noise buffer (?P<noise>-?\d+)MB; reclaim run: (?P<reclaim>[^.;]+)(?:; outstanding by unit: [^.]*)?\.)?",
)
"""Matches the clearance-hold line ``InferenceScheduler._log_clearance_hold`` emits.

The arithmetic group is optional: a hold for the post-processing co-residency mutex carries prose in its
place. The line is edge-logged once per distinct ``(process, decision)``, so a count of these lines is a
count of distinct holds, never of held ticks."""
_NO_RECLAIM = "none"
STATS_EPOCH_MATCH_SECONDS = 120.0
"""How far a stats file's start stamp may sit from its session's manager-init banner and still match it."""

# Duty bands for the distribution, as (exclusive-upper-bound, label); the last catches the rest.
_DUTY_BANDS: tuple[tuple[float, str], ...] = ((40, "<40%"), (60, "40-60%"), (75, "60-75%"), (90, "75-90%"))
_DUTY_TARGET = 90.0


def _parse_ts(line: str) -> datetime | None:
    """Parse the loguru timestamp prefix of ``line``, or None if it has none (continuation line)."""
    match = _TS_RE.match(line)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group("ts"), _TS_FORMAT)
    except ValueError:
        return None


@dataclass
class DutyWindow:
    """One ``GPU duty cycle`` report line: the reading plus its attribution."""

    timestamp: datetime | None
    duty_percent: int
    window_seconds: int
    source: str
    busy_percent: int
    no_jobs_percent: int | None
    gaps: dict[str, float]
    """Per-job seconds by friendly phase label (e.g. ``"queue wait": 16.5``)."""
    churn: dict[str, int]
    """Counts by friendly churn label (e.g. ``"VRAM evictions": 18``)."""


@dataclass
class EpochConfig:
    """The effective worker config seen in an epoch (last status line wins)."""

    dreamer_name: str | None = None
    num_models: int | None = None
    max_threads: int | None = None
    queue_size: int | None = None
    max_power: int | None = None
    safety_on_gpu: bool | None = None
    unload_models_from_vram_often: bool | None = None
    high_performance_mode: bool | None = None
    moderate_performance_mode: bool | None = None
    high_memory_mode: bool | None = None

    def summary(self) -> str:
        """A one-line config digest for the report header."""
        parts: list[str] = []
        if self.num_models is not None:
            parts.append(f"models={self.num_models}")
        if self.max_threads is not None:
            parts.append(f"max_threads={self.max_threads}")
        if self.queue_size is not None:
            parts.append(f"queue_size={self.queue_size}")
        if self.max_power is not None:
            parts.append(f"max_power={self.max_power}")
        flags = {
            "unload_models_from_vram_often": self.unload_models_from_vram_often,
            "high_performance_mode": self.high_performance_mode,
            "high_memory_mode": self.high_memory_mode,
            "safety_on_gpu": self.safety_on_gpu,
        }
        parts.extend(name for name, value in flags.items() if value)
        return ", ".join(parts) if parts else "(not found in the log)"


@dataclass
class EpochReport:
    """The duty-cycle picture for one session epoch."""

    index: int
    start: datetime | None
    end: datetime | None
    config: EpochConfig
    windows: list[DutyWindow] = field(default_factory=list)
    min_disk_free_gb: float | None = None
    governor_seconds: dict[str, float] = field(default_factory=dict)
    """Per pop-governor engaged seconds across the epoch (reconstructed from the ENTER/EXIT spell lines)."""

    def governor_attribution(self) -> dict[str, float]:
        """Per-governor engaged time, biggest first, for the epoch's duty-loss attribution."""
        return dict(sorted(self.governor_seconds.items(), key=lambda kv: kv[1], reverse=True))

    @property
    def duration_minutes(self) -> float | None:
        """Wall-clock minutes spanned by the duty windows, or None if unknown."""
        if self.start is None or self.end is None:
            return None
        return (self.end - self.start).total_seconds() / 60.0

    def duty_values(self) -> list[int]:
        """The duty-cycle percentages observed, in order."""
        return [w.duty_percent for w in self.windows]

    def mean_duty(self) -> float | None:
        """Mean duty cycle across the epoch's windows, or None when it has none."""
        values = self.duty_values()
        return statistics.mean(values) if values else None

    def mean_busy(self) -> float | None:
        """Mean busy-fraction percent across the windows, or None when it has none."""
        values = [w.busy_percent for w in self.windows]
        return statistics.mean(values) if values else None

    def band_distribution(self) -> dict[str, int]:
        """How many windows fell in each duty band (the shape of the shortfall)."""
        counts = {label: 0 for _, label in _DUTY_BANDS}
        counts[f">={int(_DUTY_TARGET)}%"] = 0
        for value in self.duty_values():
            placed = False
            for upper, label in _DUTY_BANDS:
                if value < upper:
                    counts[label] += 1
                    placed = True
                    break
            if not placed:
                counts[f">={int(_DUTY_TARGET)}%"] += 1
        return counts

    def mean_gaps(self) -> dict[str, float]:
        """Mean per-job seconds per phase across windows that reported it, biggest first."""
        accumulated: dict[str, list[float]] = {}
        for window in self.windows:
            for label, seconds in window.gaps.items():
                accumulated.setdefault(label, []).append(seconds)
        means = {label: statistics.mean(values) for label, values in accumulated.items()}
        return dict(sorted(means.items(), key=lambda kv: kv[1], reverse=True))

    def churn_totals(self) -> dict[str, int]:
        """Total churn events of each kind across the epoch, biggest first."""
        totals: dict[str, int] = {}
        for window in self.windows:
            for label, count in window.churn.items():
                totals[label] = totals.get(label, 0) + count
        return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))


def parse_duty_window(message: str, timestamp: datetime | None) -> DutyWindow | None:
    """Parse one ``GPU duty cycle`` log message into a :class:`DutyWindow`, or None if it is not one."""
    match = _DUTY_RE.search(message)
    if match is None:
        return None

    gaps: dict[str, float] = {}
    gaps_match = _GAPS_RE.search(message)
    if gaps_match is not None:
        for entry in gaps_match.group("gaps").split(","):
            entry_match = _GAP_ENTRY_RE.search(entry.strip())
            if entry_match is not None:
                gaps[entry_match.group("label").strip()] = float(entry_match.group("seconds"))

    churn: dict[str, int] = {}
    churn_match = _CHURN_RE.search(message)
    if churn_match is not None:
        for entry in churn_match.group("churn").split(","):
            entry_match = _CHURN_ENTRY_RE.search(entry.strip())
            if entry_match is not None:
                churn[entry_match.group("label").strip()] = int(entry_match.group("count"))

    no_jobs_match = _NO_JOBS_RE.search(message)
    return DutyWindow(
        timestamp=timestamp,
        duty_percent=int(match.group("duty")),
        window_seconds=int(match.group("window")),
        source=match.group("source"),
        busy_percent=int(match.group("busy")),
        no_jobs_percent=int(no_jobs_match.group("frac")) if no_jobs_match else None,
        gaps=gaps,
        churn=churn,
    )


def split_into_epochs(lines: list[str]) -> list[list[str]]:
    """Split raw log lines into per-session epochs on the process-manager init banner.

    Lines before the first boundary (pre-session preamble) are dropped. Boundary lines within
    ``_EPOCH_COLLAPSE_SECONDS`` of the current epoch's start are the same startup burst, not a new
    epoch. Falls back to the worker-identity marker if no manager-init lines are present.
    """
    boundary_re = _EPOCH_BOUNDARY_RE
    if not any(_EPOCH_BOUNDARY_RE.search(line) for line in lines):
        boundary_re = _EPOCH_BOUNDARY_FALLBACK_RE

    epochs: list[list[str]] = []
    current_start_ts: datetime | None = None
    for line in lines:
        if boundary_re.search(line):
            ts = _parse_ts(line)
            is_new_epoch = (
                not epochs
                or current_start_ts is None
                or ts is None
                or (ts - current_start_ts).total_seconds() > _EPOCH_COLLAPSE_SECONDS
            )
            if is_new_epoch:
                epochs.append([])
                current_start_ts = ts
        if epochs:
            epochs[-1].append(line)
    return epochs


def build_epoch_report(index: int, lines: list[str]) -> EpochReport:
    """Parse one epoch's lines into an :class:`EpochReport`."""
    config = EpochConfig()
    windows: list[DutyWindow] = []
    min_disk: float | None = None
    last_ts: datetime | None = None
    # Pop-governor spell reconstruction: pair each ENTER with the next EXIT for the same governor; a spell
    # still open at the epoch's last timestamp counts to there. Attributes idle/non-pop time to the condition
    # that caused it, alongside the duty-band shortfall.
    governor_open: dict[str, datetime] = {}
    governor_seconds: dict[str, float] = {}

    for line in lines:
        ts = _parse_ts(line)
        if ts is not None:
            last_ts = ts

        enter = GOVERNOR_ENTER_RE.search(line)
        if enter is not None and ts is not None:
            governor_open[enter.group("name")] = ts
            continue
        governor_exit = GOVERNOR_EXIT_RE.search(line)
        if governor_exit is not None and ts is not None:
            opened = governor_open.pop(governor_exit.group("name"), None)
            if opened is not None:
                name = governor_exit.group("name")
                governor_seconds[name] = governor_seconds.get(name, 0.0) + max(0.0, (ts - opened).total_seconds())
            continue

        window = parse_duty_window(line, ts)
        if window is not None:
            windows.append(window)
            continue

        config_match = _CONFIG_RE.search(line)
        if config_match is not None:
            config.unload_models_from_vram_often = config_match.group("unload") == "True"
            config.high_performance_mode = config_match.group("hpm") == "True"
            config.moderate_performance_mode = config_match.group("mpm") == "True"
            # high_memory_mode was removed from the worker; only older logs carry it (None when absent).
            config.high_memory_mode = config_match.group("hmm") == "True" if config_match.group("hmm") else None
            continue

        identity_match = _IDENTITY_RE.search(line)
        if identity_match is not None:
            config.dreamer_name = identity_match.group("name").strip()
            config.num_models = int(identity_match.group("num_models"))
            config.max_power = int(identity_match.group("max_power"))
            config.max_threads = int(identity_match.group("max_threads"))
            config.queue_size = int(identity_match.group("queue_size"))
            config.safety_on_gpu = identity_match.group("safety") == "True"
            continue

        disk_match = _DISK_RE.search(line)
        if disk_match is not None:
            free = float(disk_match.group("free"))
            min_disk = free if min_disk is None else min(min_disk, free)

    # Close any governor spell still open at the epoch's end so its time is not lost.
    for name, opened in governor_open.items():
        if last_ts is not None:
            governor_seconds[name] = governor_seconds.get(name, 0.0) + max(0.0, (last_ts - opened).total_seconds())

    start = _parse_ts(lines[0]) if lines else None
    return EpochReport(
        index=index,
        start=windows[0].timestamp if windows else start,
        end=windows[-1].timestamp if windows else last_ts,
        config=config,
        windows=windows,
        min_disk_free_gb=min_disk,
        governor_seconds=governor_seconds,
    )


def analyze_log(lines: list[str]) -> list[EpochReport]:
    """Split ``lines`` into epochs and build a report for each that has at least one duty window."""
    reports = [build_epoch_report(i, epoch_lines) for i, epoch_lines in enumerate(split_into_epochs(lines))]
    return [report for report in reports if report.windows]


@dataclass
class ClearanceHold:
    """Represents one ``Clearance held for process`` line.

    The figures are None for a hold whose line carries no arithmetic (the post-processing co-residency
    mutex), and individually None where the line printed ``unpriced``, ``unreported`` or ``n/a``.
    """

    timestamp: datetime | None
    process_id: int
    model: str
    decision: str
    candidate_mb: int | None
    child_holds_mb: int | None
    device_free_mb: int | None
    available_mb: int | None
    reserve_mb: int | None
    outstanding_mb: int | None
    noise_buffer_mb: int | None
    reclaim_run: str | None
    """The reclaim commands the arbiter ran, comma-joined, ``none`` for no reclaim, None without arithmetic."""


@dataclass
class ClearanceHoldDigest:
    """Represents one model's clearance holds across a session."""

    model: str
    hold_lines: int
    """Distinct holds logged; the line is edge-triggered per process and decision, so this is not tick time."""
    decisions: dict[str, int]
    median_candidate_mb: float | None
    median_available_mb: float | None
    median_outstanding_mb: float | None
    reclaim_runs: dict[str, int]
    """Holds per reclaim-run text, ``none`` included; holds without arithmetic are not counted here."""


def parse_clearance_hold(line: str) -> ClearanceHold | None:
    """Parse one ``Clearance held for process`` log line, or return None when it is not one."""
    match = CLEARANCE_HOLD_RE.search(line)
    if match is None:
        return None
    return ClearanceHold(
        timestamp=_parse_ts(line),
        process_id=int(match.group("process")),
        model=match.group("model"),
        decision=match.group("decision").strip(),
        candidate_mb=_int_group(match, "candidate"),
        child_holds_mb=_int_group(match, "held"),
        device_free_mb=_int_group(match, "free"),
        available_mb=_int_group(match, "available"),
        reserve_mb=_int_group(match, "reserve"),
        outstanding_mb=_int_group(match, "outstanding"),
        noise_buffer_mb=_int_group(match, "noise"),
        reclaim_run=match.group("reclaim"),
    )


def summarize_clearance_holds(lines: list[str]) -> list[ClearanceHoldDigest]:
    """Group the clearance-hold lines among ``lines`` by model, most holds first."""
    by_model: dict[str, list[ClearanceHold]] = {}
    for line in lines:
        hold = parse_clearance_hold(line)
        if hold is not None:
            by_model.setdefault(hold.model, []).append(hold)
    digests = [_clearance_digest(model, holds) for model, holds in by_model.items()]
    digests.sort(key=lambda digest: (-digest.hold_lines, digest.model))
    return digests


def epoch_lines_for_stats_session(lines: list[str], started_local: datetime) -> list[str]:
    """Return the log epoch a stats session belongs to, matched by start time on the host clock.

    Args:
        lines: The whole ``bridge.log``.
        started_local: The stats session's start stamp (its filename), in the host's local time.

    Returns:
        The lines of the epoch whose first line sits closest to ``started_local``, within
        ``STATS_EPOCH_MATCH_SECONDS``; empty when no epoch starts that close (a tail-capped log, for one).
    """
    best: list[str] = []
    best_distance: float | None = None
    for epoch in split_into_epochs(lines):
        epoch_start = _parse_ts(epoch[0]) if epoch else None
        if epoch_start is None:
            continue
        distance = abs((epoch_start - started_local).total_seconds())
        if distance > STATS_EPOCH_MATCH_SECONDS:
            continue
        if best_distance is None or distance < best_distance:
            best, best_distance = epoch, distance
    return best


def render_clearance_hold_digest(digests: list[ClearanceHoldDigest]) -> list[str]:
    """Render the per-model clearance-hold digest as report lines, empty when no hold was logged."""
    if not digests:
        return []
    total = sum(digest.hold_lines for digest in digests)
    out = [f"   times a job waited for free VRAM before sampling (clearance holds): {total}"]
    for digest in digests:
        decisions = ", ".join(f"{decision} {count}" for decision, count in digest.decisions.items())
        # A run of several commands is comma-joined in the line itself, so it is rejoined with "+" here.
        reclaim = ", ".join(f"{run.replace(', ', '+')} {count}" for run, count in digest.reclaim_runs.items()) or "n/a"
        out.append(
            f"     {digest.model}: {digest.hold_lines} ({decisions}), median VRAM needed "
            f"{_mb_or_na(digest.median_candidate_mb)}, available {_mb_or_na(digest.median_available_mb)}, "
            f"reserved for other jobs {_mb_or_na(digest.median_outstanding_mb)}, "
            f"steps taken to free VRAM: {reclaim}"
        )
    return out


def _clearance_digest(model: str, holds: list[ClearanceHold]) -> ClearanceHoldDigest:
    decisions: dict[str, int] = {}
    reclaim_runs: dict[str, int] = {}
    for hold in holds:
        decisions[hold.decision] = decisions.get(hold.decision, 0) + 1
        if hold.reclaim_run is not None:
            reclaim_runs[hold.reclaim_run] = reclaim_runs.get(hold.reclaim_run, 0) + 1
    return ClearanceHoldDigest(
        model=model,
        hold_lines=len(holds),
        decisions=dict(sorted(decisions.items(), key=lambda kv: -kv[1])),
        median_candidate_mb=_median_or_none([hold.candidate_mb for hold in holds]),
        median_available_mb=_median_or_none([hold.available_mb for hold in holds]),
        median_outstanding_mb=_median_or_none([hold.outstanding_mb for hold in holds]),
        reclaim_runs=dict(sorted(reclaim_runs.items(), key=lambda kv: (kv[0] != _NO_RECLAIM, -kv[1]))),
    )


def _int_group(match: re.Match[str], group: str) -> int | None:
    value = match.group(group)
    return None if value is None else int(value)


def _median_or_none(values: list[int | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(statistics.median(present)) if present else None


def _mb_or_na(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}MB"


_IDLE_SAMPLE_WEIGHT_CAP_SECONDS = 5.0
"""The most wall clock one stats sample stands for, so a gap in the stream does not land on one sample as idle."""
_LAST_SAMPLE_WEIGHT_SECONDS = 1.0
"""The weight of the window's last sample, which has no successor to measure its gap against."""
_DISPATCH_LOOKAHEAD_SECONDS = 0.5
"""How far after the sample that first sees a lane sampling its dispatch line may be stamped."""
_GRANT_LOOKBACK_SECONDS = 0.05
"""A sample grant logged this far before its job's dispatch line still belongs to that job."""
_GRANT_LOOKAHEAD_SECONDS = 1.0
"""How far after the stretch's ending sample the ending job's sample grant may be stamped."""
_UNPRICED_LOOKAHEAD_SECONDS = 2.0
"""How far after the stretch's ending sample an unpriced-sampling line may be stamped and still count."""
_RUNG_2_RESTART_SECONDS = 120.0
"""Lanes starting within this long after a rung-2 residency ends are the siblings it tore down restarting."""
_HOST_OFFSET_STEP_SECONDS = 900.0
"""Host UTC offsets are whole quarter hours, so the measured offset is rounded to one."""
_DISPATCH_STAGE = "INFERENCE_IN_PROGRESS"
"""The ``stage_timestamps`` key the job tracker stamps at dispatch (``JobStage.INFERENCE_IN_PROGRESS.name``).

Spelled out because importing the job tracker pulls the SDK into this import-light module."""
_PROCESS_STARTING_STATE = "PROCESS_STARTING"
_PRELOADING_STATE = "PRELOADING_MODEL"
_RESIDENCY_GOVERNOR = "whole_card_residency"
_NO_LOCAL_WORK_GOVERNORS = (
    "consecutive_failure_pause",
    "self_throttle_pause",
    _RESIDENCY_GOVERNOR,
    "large_model_reentry",
    "megapixelstep_wait",
)
"""Pop governors that explain an empty local queue, in the order the first engaged one is named."""
_RUNG_2_HOLD_BUCKETS = frozenset(
    {SlotDutyBucket.WHOLE_CARD_CONVERGENCE.value, SlotDutyBucket.WHOLE_CARD_RESERVED.value},
)
_PRELOAD_HOLD_BUCKETS = frozenset({SlotDutyBucket.PRELOAD_DEFERRED.value, SlotDutyBucket.OVERLAP_HEADWAY.value})
"""Hold buckets under which a preloading lane, not the hold, is what the head waits on (an unset bucket too)."""
_NO_FIRST_STEP = "none"
"""The first-step kind of a job whose ``from dispatch to first step`` line is absent."""
_UNKNOWN_MODEL = "unknown model"
_RENDERED_MODELS_PER_CAUSE = 8


class IdleContext(StrEnum):
    """The whole-card state an idle sample fell in."""

    RUNG_2_RESIDENCY = "rung-2 residency"
    """A whole-card residency episode held the card (``Pop governor ENTER: whole_card_residency`` to its exit)."""
    RUNG_1_RUN = "rung-1 run"
    """A dominant-model run narrowed the offer, from its open line to its end line."""
    NEITHER = "neither"


class IdleSegment(StrEnum):
    """Where on the ending job's timeline an idle sample sits."""

    PRE_DISPATCH = "pre-dispatch"
    """Before the ending job was dispatched."""
    ARBITER = "arbiter"
    """Dispatch to the job's sample grant, or to its first step when no grant line was logged."""
    CHILD_LOAD = "child load after grant"
    """Sample grant to the first step: the child loading and preparing to sample."""
    WINDOW_END = "window end"
    """The stretch runs to the window's end, so no job ends it."""


class ArbiterCause(StrEnum):
    """How the arbiter segment of an idle stretch ended."""

    GRANTED = "dispatch to sample grant"
    NO_GRANT = "no sample grant line"
    NO_GRANT_UNPRICED = "no sample grant line (unpriced after timeout)"
    """The lane entered its denoise loop without a recorded grant inside the stretch."""


@dataclass
class IdleStretch:
    """Represents consecutive idle stats samples and the job whose sampling ended them."""

    start_timestamp: float
    """The first idle sample's epoch timestamp."""
    idle_seconds: float
    ending_process: int | None
    """The lane that turned sampling at the stretch's end, None for a stretch that reaches the window's end."""
    ending_job: str | None
    """The short id of the job most recently dispatched on the ending lane."""
    model: str | None
    first_step_kind: str | None
    segment_seconds: dict[str, float]


@dataclass
class IdleBucket:
    """Represents the idle seconds of one cause on the ending job's timeline."""

    segment: IdleSegment
    cause: str
    """The pre-dispatch cause, the :class:`ArbiterCause`, or the first-step kind (``cold`` or ``resident``)."""
    model: str | None
    """The ending job's model for the arbiter and child-load segments, None before dispatch."""
    seconds: float = 0.0
    stretches: int = 0
    context_seconds: dict[str, float] = field(default_factory=dict)
    """Seconds by :class:`IdleContext` value."""


@dataclass
class IdleAttribution:
    """Represents every idle card-second of a session's log epoch, placed on the next job's timeline."""

    window_seconds: float
    idle_seconds: float
    target_duty_percent: float
    rung_1_runs: int
    rung_2_episodes: int
    stretches: list[IdleStretch]
    buckets: list[IdleBucket]
    """Biggest first."""

    @property
    def seconds_above_target(self) -> float:
        """Idle seconds that would have to become sampling for the window to reach the card-busy target."""
        return self.idle_seconds - (1.0 - self.target_duty_percent / 100.0) * self.window_seconds

    def segment_seconds(self) -> dict[IdleSegment, float]:
        """Idle seconds per segment, in timeline order, segments with none left out."""
        totals = dict.fromkeys(IdleSegment, 0.0)
        for bucket in self.buckets:
            totals[bucket.segment] += bucket.seconds
        return {segment: seconds for segment, seconds in totals.items() if seconds > 0.0}

    def to_dict(self) -> dict[str, object]:
        """Return the totals and buckets as JSON-ready values; the per-stretch rows are left out for size."""
        return {
            "window_seconds": self.window_seconds,
            "idle_seconds": self.idle_seconds,
            "target_duty_percent": self.target_duty_percent,
            "seconds_above_target": self.seconds_above_target,
            "stretch_count": len(self.stretches),
            "rung_1_runs": self.rung_1_runs,
            "rung_2_episodes": self.rung_2_episodes,
            "segment_seconds": {str(segment): seconds for segment, seconds in self.segment_seconds().items()},
            "buckets": [asdict(bucket) | {"segment": str(bucket.segment)} for bucket in self.buckets],
        }


@dataclass
class _Dispatch:
    seconds: float
    job: str


@dataclass
class _Spell:
    start: float
    end: float
    name: str


@dataclass
class _EpochTimeline:
    """The epoch's lines the attribution reads, stamped in seconds of the host clock read as UTC."""

    start: float
    end: float
    dispatches: dict[int, list[_Dispatch]] = field(default_factory=dict)
    grants: dict[int, list[float]] = field(default_factory=dict)
    unpriced: dict[int, list[float]] = field(default_factory=dict)
    first_step_kind: dict[str, str] = field(default_factory=dict)
    runs: list[_Spell] = field(default_factory=list)
    residencies: list[_Spell] = field(default_factory=list)
    governor_spells: list[_Spell] = field(default_factory=list)
    pop_claims: list[_Spell] = field(default_factory=list)

    def context(self, seconds: float) -> IdleContext:
        if any(spell.start <= seconds <= spell.end for spell in self.residencies):
            return IdleContext.RUNG_2_RESIDENCY
        if any(spell.start <= seconds <= spell.end for spell in self.runs):
            return IdleContext.RUNG_1_RUN
        return IdleContext.NEITHER


_EPOCH_ORIGIN = datetime(1970, 1, 1)


def _host_seconds(stamp: datetime) -> float:
    """Seconds since the epoch of a naive host-clock stamp read as if it were UTC."""
    return (stamp - _EPOCH_ORIGIN).total_seconds()


def _read_epoch_timeline(epoch_lines: list[str]) -> _EpochTimeline | None:
    """Parse the dispatch, grant, first-step, run, residency and governor lines of one epoch."""
    stamped = [(stamp, line) for line in epoch_lines if (stamp := _parse_ts(line)) is not None]
    if not stamped:
        return None
    timeline = _EpochTimeline(start=_host_seconds(stamped[0][0]), end=_host_seconds(stamped[-1][0]))
    open_runs: dict[str, float] = {}
    open_residency: float | None = None
    open_governors: dict[str, float] = {}
    open_claim: float | None = None
    for stamp, line in stamped:
        seconds = _host_seconds(stamp)
        if "Starting inference for job" in line and (match := SIGNATURES["inference_dispatched"].pattern.search(line)):
            timeline.dispatches.setdefault(int(match.group("process")), []).append(
                _Dispatch(seconds=seconds, job=match.group("job").lower()),
            )
        elif "into its load-and-sample window" in line and (
            match := SIGNATURES["clearance_sample_granted"].pattern.search(line)
        ):
            timeline.grants.setdefault(int(match.group("process")), []).append(seconds)
        elif "from dispatch to first step" in line and (
            match := SIGNATURES["dispatch_to_first_step"].pattern.search(line)
        ):
            timeline.first_step_kind[match.group("job").lower()] = match.group("residency")
        elif "without a recorded grant" in line and (
            match := SIGNATURES["unpriced_sampling_window"].pattern.search(line)
        ):
            timeline.unpriced.setdefault(int(match.group("process")), []).append(seconds)
        elif "Dominant-model run for" in line:
            if (match := SIGNATURES["dominant_run_opened"].pattern.search(line)) is not None:
                open_runs.setdefault(match.group("model"), seconds)
            elif (match := SIGNATURES["dominant_run_ended"].pattern.search(line)) is not None:
                start = open_runs.pop(match.group("model"), timeline.start)
                timeline.runs.append(_Spell(start=start, end=seconds, name=match.group("model")))
        elif ("Pre-staging whole-card head" in line or "Whole-card residency: reserving" in line) and (
            SIGNATURES["whole_card_prestage"].pattern.search(line)
            or SIGNATURES["whole_card_establish"].pattern.search(line)
        ):
            # A rung-2 grant overwrites the whole-card ledger's state, so an open run ends here without its
            # own end line.
            for model, start in open_runs.items():
                timeline.runs.append(_Spell(start=start, end=seconds, name=model))
            open_runs.clear()
        elif "Pop governor " in line:
            if (enter := GOVERNOR_ENTER_RE.search(line)) is not None:
                open_governors[enter.group("name")] = seconds
                if SIGNATURES["residency_governor_model"].pattern.search(line):
                    open_residency = seconds
            elif (governor_exit := GOVERNOR_EXIT_RE.search(line)) is not None:
                name = governor_exit.group("name")
                if name in open_governors:
                    timeline.governor_spells.append(_Spell(start=open_governors.pop(name), end=seconds, name=name))
                if name == _RESIDENCY_GOVERNOR and open_residency is not None:
                    timeline.residencies.append(_Spell(start=open_residency, end=seconds, name=name))
                    open_residency = None
        elif "Whole-card pop claim" in line:
            if SIGNATURES["pop_claim_engaged"].pattern.search(line):
                open_claim = seconds
            elif SIGNATURES["pop_claim_released"].pattern.search(line) and open_claim is not None:
                timeline.pop_claims.append(_Spell(start=open_claim, end=seconds, name="pop_claim"))
                open_claim = None
    for model, start in open_runs.items():
        timeline.runs.append(_Spell(start=start, end=timeline.end, name=model))
    if open_residency is not None:
        timeline.residencies.append(_Spell(start=open_residency, end=timeline.end, name=_RESIDENCY_GOVERNOR))
    timeline.runs.sort(key=lambda spell: spell.start)
    timeline.residencies.sort(key=lambda spell: spell.start)
    return timeline


def _host_clock_offset(timeline: _EpochTimeline, jobs: list[dict[str, Any]]) -> float | None:
    """Return the host clock's offset from UTC in seconds, from dispatch lines paired with the jobs' stamps.

    The log is stamped on the host's clock and the stats in epoch seconds; each job's dispatch is recorded in
    both, so their median difference is the offset with no guess at the host's time zone. None when no
    dispatch line names a job the stats recorded.
    """
    dispatched_at: dict[str, float] = {}
    for job in jobs:
        stages = job.get("stage_timestamps")
        job_id = job.get("job_id")
        if isinstance(stages, dict) and isinstance(job_id, str):
            stamp = stages.get(_DISPATCH_STAGE)
            if isinstance(stamp, int | float):
                dispatched_at[job_id[:8].lower()] = float(stamp)
    differences = [
        dispatch.seconds - dispatched_at[dispatch.job]
        for dispatches in timeline.dispatches.values()
        for dispatch in dispatches
        if dispatch.job in dispatched_at
    ]
    if not differences:
        return None
    return round(statistics.median(differences) / _HOST_OFFSET_STEP_SECONDS) * _HOST_OFFSET_STEP_SECONDS


def _inference_lane_states(sample: dict[str, Any]) -> dict[int, str]:
    """Each inference lane's state in ``sample``, keyed by its process id."""
    summary = str(sample.get("process_state_summary") or "")
    return {
        int(match.group("process")): match.group("state")
        for match in _LANE_STATE_RE.finditer(summary)
        if match.group("label") == _INFERENCE_LANE_LABEL
    }


def _sampling_processes(sample: dict[str, Any]) -> list[int]:
    return sorted(
        process for process, state in _inference_lane_states(sample).items() if state in _SAMPLING_LANE_STATES
    )


def _pre_dispatch_cause(sample: dict[str, Any], seconds: float, timeline: _EpochTimeline) -> str:
    """Name what held an idle sample that falls before its ending job's dispatch, from the sample itself."""
    inference_states = set(_inference_lane_states(sample).values())
    if not sample.get("jobs_pending_inference") and not sample.get("jobs_in_progress"):
        active = {spell.name for spell in timeline.governor_spells if spell.start <= seconds <= spell.end}
        for name in _NO_LOCAL_WORK_GOVERNORS:
            if name in active:
                return f"no local work: pop governor {name}"
        if any(spell.start <= seconds <= spell.end for spell in timeline.pop_claims):
            return "no local work: pop claim narrows the offer"
        if sample.get("jobs_pending_safety_check") or sample.get("jobs_being_safety_checked"):
            return "no local work: safety outstanding"
        if sample.get("jobs_pending_submit"):
            return "no local work: submit outstanding"
        if any(
            label == _POST_PROCESS_LANE_LABEL and state in _POST_PROCESSING_LANE_STATES
            for label, state in _lane_states(sample)
        ):
            return "no local work: post-processing outstanding"
        return "no local work: nothing outstanding"
    hold = sample.get("dispatch_hold_bucket")
    if timeline.context(seconds) is IdleContext.RUNG_2_RESIDENCY or hold in _RUNG_2_HOLD_BUCKETS:
        return f"rung-2 residency ({hold})"
    if _PROCESS_STARTING_STATE in inference_states:
        residency_ended_recently = any(
            0.0 <= seconds - spell.end <= _RUNG_2_RESTART_SECONDS for spell in timeline.residencies
        )
        if residency_ended_recently:
            return "rung-2 exit: lanes restarting"
        return f"lanes starting ({hold})"
    if _PRELOADING_STATE in inference_states and (hold is None or hold in _PRELOAD_HOLD_BUCKETS):
        return f"model preloading ({hold})"
    return f"pre-dispatch hold ({hold})"


def _first_between(stamps: list[float], low: float, high: float) -> float | None:
    index = bisect.bisect_left(stamps, low)
    if index < len(stamps) and stamps[index] <= high:
        return stamps[index]
    return None


@dataclass
class _EndingJob:
    process: int
    job: str
    dispatched: float
    granted: float | None
    unpriced: bool
    model: str
    first_step_kind: str


def _ending_job(
    processes: list[int],
    ending_seconds: float,
    timeline: _EpochTimeline,
    models: dict[str, str],
) -> _EndingJob | None:
    """Pick the job most recently dispatched on a lane that turned sampling at ``ending_seconds``."""
    best: tuple[int, _Dispatch] | None = None
    for process in processes:
        dispatches = timeline.dispatches.get(process, [])
        index = bisect.bisect_right([d.seconds for d in dispatches], ending_seconds + _DISPATCH_LOOKAHEAD_SECONDS) - 1
        if index >= 0 and (best is None or dispatches[index].seconds > best[1].seconds):
            best = (process, dispatches[index])
    if best is None:
        return None
    process, dispatch = best
    granted = _first_between(
        timeline.grants.get(process, []),
        dispatch.seconds - _GRANT_LOOKBACK_SECONDS,
        ending_seconds + _GRANT_LOOKAHEAD_SECONDS,
    )
    unpriced = _first_between(
        timeline.unpriced.get(process, []),
        dispatch.seconds,
        ending_seconds + _UNPRICED_LOOKAHEAD_SECONDS,
    )
    return _EndingJob(
        process=process,
        job=dispatch.job,
        dispatched=dispatch.seconds,
        granted=granted,
        unpriced=unpriced is not None,
        model=models.get(dispatch.job, _UNKNOWN_MODEL),
        first_step_kind=timeline.first_step_kind.get(dispatch.job, _NO_FIRST_STEP),
    )


def _place_idle_sample(
    sample: dict[str, Any],
    seconds: float,
    ending: _EndingJob | None,
    timeline: _EpochTimeline,
) -> tuple[IdleSegment, str, str | None]:
    """Return the segment, cause and model an idle sample lands in on its ending job's timeline."""
    if ending is None:
        return IdleSegment.WINDOW_END, _pre_dispatch_cause(sample, seconds, timeline), None
    if seconds < ending.dispatched:
        return IdleSegment.PRE_DISPATCH, _pre_dispatch_cause(sample, seconds, timeline), None
    if ending.granted is None:
        cause = ArbiterCause.NO_GRANT_UNPRICED if ending.unpriced else ArbiterCause.NO_GRANT
        return IdleSegment.ARBITER, str(cause), ending.model
    if seconds < ending.granted:
        return IdleSegment.ARBITER, str(ArbiterCause.GRANTED), ending.model
    return IdleSegment.CHILD_LOAD, ending.first_step_kind, ending.model


def summarize_idle_attribution(
    samples: list[dict[str, Any]],
    epoch_lines: list[str],
    jobs: list[dict[str, Any]],
    *,
    target_duty_percent: float = _DUTY_TARGET,
) -> IdleAttribution | None:
    """Place every idle card-second of a stats session's log epoch on the timeline of the job that ended it.

    An idle sample is one with no inference lane in a sampling state, weighted by the gap to the next sample
    capped at :data:`_IDLE_SAMPLE_WEIGHT_CAP_SECONDS`. Consecutive idle samples form a stretch, which ends at
    the first sample where a lane samples; every idle sample of the stretch lands on the timeline of the job
    most recently dispatched on that lane: before its dispatch (named from the sample), dispatch to its sample
    grant (the arbiter), or grant to its first step (the child loading, split by the first-step kind). A
    stretch that reaches the window's end has no ending job and is named from its samples alone.

    Args:
        samples: The session's ``stats_sample`` payloads.
        epoch_lines: The log epoch the session belongs to.
        jobs: The session's ``job_completed`` payloads, which pair the log's clock with the stats' and name
            each job's model.
        target_duty_percent: The card-busy target the idle seconds are measured against.

    Returns:
        The attribution over the samples inside the epoch's span, or None when the epoch has no stamped line,
        no dispatch line names a recorded job, or no sample falls inside the span.
    """
    timeline = _read_epoch_timeline(epoch_lines)
    if timeline is None:
        return None
    offset = _host_clock_offset(timeline, jobs)
    if offset is None:
        return None
    window: list[tuple[float, float, dict[str, Any]]] = []
    for sample in samples:
        stamp = sample.get("timestamp")
        if not isinstance(stamp, int | float):
            continue
        seconds = float(stamp) + offset
        if timeline.start <= seconds <= timeline.end:
            window.append((seconds, float(stamp), sample))
    if not window:
        return None
    window.sort(key=lambda entry: entry[0])
    weights = [
        min(_IDLE_SAMPLE_WEIGHT_CAP_SECONDS, max(0.0, window[index + 1][0] - seconds))
        if index + 1 < len(window)
        else _LAST_SAMPLE_WEIGHT_SECONDS
        for index, (seconds, _, _) in enumerate(window)
    ]
    sampling = [_sampling_processes(sample) for _, _, sample in window]
    models = {
        job["job_id"][:8].lower(): str(job.get("model_name") or _UNKNOWN_MODEL)
        for job in jobs
        if isinstance(job.get("job_id"), str)
    }

    buckets: dict[tuple[IdleSegment, str, str | None], IdleBucket] = {}
    stretches: list[IdleStretch] = []
    index = 0
    while index < len(window):
        if sampling[index]:
            index += 1
            continue
        stop = index
        while stop < len(window) and not sampling[stop]:
            stop += 1
        ending = _ending_job(sampling[stop], window[stop][0], timeline, models) if stop < len(window) else None
        segment_seconds: dict[str, float] = {}
        touched: set[tuple[IdleSegment, str, str | None]] = set()
        for position in range(index, stop):
            seconds, _, sample = window[position]
            weight = weights[position]
            key = _place_idle_sample(sample, seconds, ending, timeline)
            bucket = buckets.setdefault(key, IdleBucket(segment=key[0], cause=key[1], model=key[2]))
            bucket.seconds += weight
            context = str(timeline.context(seconds))
            bucket.context_seconds[context] = bucket.context_seconds.get(context, 0.0) + weight
            segment_seconds[str(key[0])] = segment_seconds.get(str(key[0]), 0.0) + weight
            touched.add(key)
        for key in touched:
            buckets[key].stretches += 1
        stretches.append(
            IdleStretch(
                start_timestamp=window[index][1],
                idle_seconds=sum(weights[index:stop]),
                ending_process=ending.process if ending else None,
                ending_job=ending.job if ending else None,
                model=ending.model if ending else None,
                first_step_kind=ending.first_step_kind if ending else None,
                segment_seconds=segment_seconds,
            ),
        )
        index = stop

    return IdleAttribution(
        window_seconds=sum(weights),
        idle_seconds=sum(stretch.idle_seconds for stretch in stretches),
        target_duty_percent=target_duty_percent,
        rung_1_runs=len(timeline.runs),
        rung_2_episodes=len(timeline.residencies),
        stretches=stretches,
        buckets=sorted(buckets.values(), key=lambda bucket: bucket.seconds, reverse=True),
    )


def render_idle_attribution(attribution: IdleAttribution | None) -> list[str]:
    """Render the idle attribution as report lines grouped by segment, cause and model; empty without one."""
    if attribution is None or attribution.idle_seconds <= 0.0:
        return []
    idle = attribution.idle_seconds

    def share(seconds: float) -> str:
        return f"{seconds:.0f}s ({seconds / idle * 100:.1f}%)"

    def contexts(context_seconds: dict[str, float]) -> str:
        parts = [
            f"{context} {context_seconds[context]:.0f}s"
            for context in map(str, IdleContext)
            if context in context_seconds
        ]
        return "; " + ", ".join(parts) if parts else ""

    out = [
        f"   idle card-seconds by the next job's timeline: {idle:.0f}s of {attribution.window_seconds:.0f}s "
        f"({idle / attribution.window_seconds * 100:.1f}%) over {len(attribution.stretches)} stretches, "
        f"{max(0.0, attribution.seconds_above_target):.0f}s above the {attribution.target_duty_percent:.0f}% "
        f"card-busy target; rung-1 runs {attribution.rung_1_runs}, rung-2 episodes {attribution.rung_2_episodes}",
    ]
    for segment, segment_total in attribution.segment_seconds().items():
        out.append(f"     {segment}: {share(segment_total)}")
        by_cause: dict[str, list[IdleBucket]] = {}
        for bucket in attribution.buckets:
            if bucket.segment is segment:
                by_cause.setdefault(bucket.cause, []).append(bucket)
        for cause, cause_buckets in sorted(by_cause.items(), key=lambda kv: -sum(b.seconds for b in kv[1])):
            context_seconds: dict[str, float] = {}
            for bucket in cause_buckets:
                for context, seconds in bucket.context_seconds.items():
                    context_seconds[context] = context_seconds.get(context, 0.0) + seconds
            out.append(f"       {cause}: {share(sum(b.seconds for b in cause_buckets))}{contexts(context_seconds)}")
            for bucket in [b for b in cause_buckets if b.model is not None][:_RENDERED_MODELS_PER_CAUSE]:
                out.append(
                    f"         {bucket.model}: {share(bucket.seconds)} over {bucket.stretches} stretches"
                    f"{contexts(bucket.context_seconds)}"
                )
    return out


def _verdict(report: EpochReport) -> str:
    """A one-line plain-language read of the epoch's headline number."""
    mean = report.mean_duty()
    if mean is None:
        return "no GPU duty cycle readings"
    if mean >= _DUTY_TARGET:
        return f"at target ({mean:.0f}% >= {int(_DUTY_TARGET)}%)"
    busy = report.mean_busy()
    verdict = f"below the {int(_DUTY_TARGET)}% target, mean {mean:.0f}%"
    if busy is not None:
        idle = max(0.0, 100.0 - busy)
        partial = max(0.0, busy - mean)
        verdict += (
            f". The GPU was idle about {idle:.0f}% of the time and lost about {partial:.0f}% "
            "to running below full load"
        )
    return verdict


def render_report(reports: list[EpochReport]) -> str:
    """Render the per-epoch report as a human-readable block."""
    if not reports:
        return "No session with GPU duty cycle readings found in the log."

    out: list[str] = []
    for report in reports:
        start = report.start.strftime("%Y-%m-%d %H:%M") if report.start else "?"
        end = report.end.strftime("%H:%M") if report.end else "?"
        duration = report.duration_minutes
        duration_str = f"{duration:.0f} min" if duration is not None else "? min"
        out.append(
            f"== Log session {report.index} | {start}-{end} ({duration_str}) | duty readings: {len(report.windows)} =="
        )
        out.append(f"   config: {report.config.summary()}")
        out.append(f"   verdict: {_verdict(report)}")

        values = report.duty_values()
        if values:
            out.append(f"   duty: mean {statistics.mean(values):.0f}%  min {min(values)}%  max {max(values)}%")
            bands = report.band_distribution()
            out.append("   readings by duty: " + "  ".join(f"{label} {count}" for label, count in bands.items()))

        gaps = report.mean_gaps()
        if gaps:
            top = "  ".join(f"{label} {seconds:.1f}s/job" for label, seconds in list(gaps.items())[:4])
            out.append(f"   longest waits per job: {top}")

        churn = report.churn_totals()
        if churn:
            out.append("   reloads and restarts: " + "  ".join(f"{label} {count}" for label, count in churn.items()))
        else:
            out.append("   reloads and restarts: none reported (none happened, or the log is from an older worker)")

        governors = report.governor_attribution()
        if governors:
            epoch_seconds = (duration or 0.0) * 60.0
            parts = []
            for name, seconds in list(governors.items())[:4]:
                label = GOVERNOR_LABELS.get(name, name)
                share = f" ({seconds / epoch_seconds * 100:.0f}%)" if epoch_seconds > 0 else ""
                parts.append(f"{label} {seconds / 60:.1f}m{share}")
            out.append("   time not asking the horde for work: " + "  ".join(parts))

        if report.min_disk_free_gb is not None:
            out.append(
                f"   (!) low disk space: down to {report.min_disk_free_gb:.1f} GB free, which can stall downloads"
            )
        out.append("")
    return "\n".join(out).rstrip()


def _report_to_dict(report: EpochReport) -> dict[str, object]:
    """A JSON-serializable view of an epoch report for ``--json``."""
    return {
        "index": report.index,
        "start": report.start.isoformat() if report.start else None,
        "end": report.end.isoformat() if report.end else None,
        "duration_minutes": report.duration_minutes,
        "window_count": len(report.windows),
        "mean_duty_percent": report.mean_duty(),
        "mean_busy_percent": report.mean_busy(),
        "governor_seconds": report.governor_attribution(),
        "min_duty_percent": min(report.duty_values()) if report.windows else None,
        "max_duty_percent": max(report.duty_values()) if report.windows else None,
        "band_distribution": report.band_distribution(),
        "top_per_job_gaps": report.mean_gaps(),
        "churn_totals": report.churn_totals(),
        "min_disk_free_gb": report.min_disk_free_gb,
        "config": report.config.__dict__,
    }


def add_duty_report_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the duty report's options to ``parser``, shared by ``horde-duty-report`` and ``horde-log duty``."""
    parser.add_argument(
        "log",
        nargs="?",
        default=None,
        type=Path,
        help=(
            "A bridge.log to report from. Given without --stats, the report is built from this log alone. "
            "Default: logs/bridge.log, read only when no stats files are found."
        ),
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Stats directory containing stats-v*.jsonl[.gz] files (default: .horde_worker_regen/stats).",
    )
    parser.add_argument(
        "--logs",
        type=Path,
        default=None,
        help=(
            "A log file, or a directory containing bridge.log. Used like the positional log. With --stats, "
            "it adds to each session the times a job waited for free VRAM before sampling."
        ),
    )
    parser.add_argument(
        "--footprints",
        type=Path,
        default=None,
        help=(
            "The learned VRAM usage file (vram_footprints.json) to list. Default: config/vram_footprints.json "
            "beside --stats in a support bundle, otherwise the worker's file."
        ),
    )
    parser.add_argument("--last", action="store_true", help="Only the most recent session.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")


def run_duty_report(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Print the duty report for parsed ``args``: stats sessions when present, else the bridge.log epochs.

    Args:
        args: Arguments parsed by a parser built with :func:`add_duty_report_arguments`.
        parser: That parser, for reporting a missing input.

    Returns:
        The process exit code.
    """
    explicit_log = args.log is not None or args.logs is not None
    stats_dir = args.stats or default_stats_dir()
    stats_reports = []
    if args.stats is not None or not explicit_log:
        stats_reports = analyze_stats_sessions(stats_dir, last=args.last)

    log_path = _resolve_log_path(args.logs or args.log or Path("logs/bridge.log"))
    if stats_reports:
        log_lines = _read_log_lines(log_path) if explicit_log and log_path.exists() else []
        epochs = [_epoch_lines_for(report, log_lines) for report in stats_reports]
        holds = [summarize_clearance_holds(lines) for lines in epochs]
        idle = [_idle_attribution_for(report, lines) for report, lines in zip(stats_reports, epochs, strict=True)]
        if args.json:
            payload = [report.to_dict() for report in stats_reports]
            if log_lines:
                for session_payload, digests, attribution in zip(payload, holds, idle, strict=True):
                    session_payload["clearance_holds"] = [asdict(digest) for digest in digests]
                    session_payload["idle_attribution"] = attribution.to_dict() if attribution else None
            print(json.dumps(payload, indent=2))
            return 0
        blocks = [
            "\n".join(
                [
                    render_session_duty_report([report]),
                    *render_clearance_hold_digest(digests),
                    *render_idle_attribution(attribution),
                ],
            )
            for report, digests, attribution in zip(stats_reports, holds, idle, strict=True)
        ]
        print("\n\n".join(blocks))
        _print_footprints(args)
        return 0

    if not log_path.exists():
        parser.error(f"stats not found in {stats_dir} and log file not found: {log_path}")

    reports = analyze_log(_read_log_lines(log_path))
    if args.last and reports:
        reports = reports[-1:]

    if args.json:
        print(json.dumps([_report_to_dict(r) for r in reports], indent=2))
        return 0
    print(render_report(reports))
    _print_footprints(args)
    return 0


def main() -> None:
    """CLI entry point: prefer stats JSONL, then fall back to bridge.log parsing."""
    parser = argparse.ArgumentParser(
        description="GPU duty cycle report from the worker's stats files or its bridge.log."
    )
    add_duty_report_arguments(parser)
    exit_code = run_duty_report(parser.parse_args(), parser)
    if exit_code:
        raise SystemExit(exit_code)


def _read_log_lines(log_path: Path) -> list[str]:
    return log_path.read_text(encoding="utf-8", errors="replace").splitlines()


def _epoch_lines_for(report: SessionDutyReport, log_lines: list[str]) -> list[str]:
    """Return the log epoch the stats session belongs to, empty without one."""
    started_local = stats_session_local_start(report)
    if not log_lines or started_local is None:
        return []
    return epoch_lines_for_stats_session(log_lines, started_local)


def _idle_attribution_for(report: SessionDutyReport, epoch_lines: list[str]) -> IdleAttribution | None:
    """Return the session's idle attribution over its log epoch, None without an epoch."""
    if not epoch_lines:
        return None
    samples: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for payload in read_stats_payloads([Path(source) for source in report.source_files]):
        event = payload.get("event")
        if event == "stats_sample" and isinstance(payload.get("sample"), dict):
            samples.append(payload["sample"])
        elif event == "job_completed" and isinstance(payload.get("job"), dict):
            jobs.append(payload["job"])
    return summarize_idle_attribution(samples, epoch_lines, jobs, target_duty_percent=report.target_duty_percent)


def _print_footprints(args: argparse.Namespace) -> None:
    """Print the learned-footprint view; a missing default store prints nothing, a missing named one says so."""
    store_path = args.footprints or default_footprint_store_path(args.stats)
    if not store_path.is_file():
        if args.footprints is not None:
            print(f"\nLearned VRAM usage file not found: {store_path}")
        return
    print()
    print(render_learned_sampling_footprints(store_path, learned_sampling_footprints(store_path)))


def _resolve_log_path(path: Path) -> Path:
    """Resolve ``--logs`` whether it points at a directory or bridge.log itself."""
    return path / "bridge.log" if path.is_dir() else path


if __name__ == "__main__":
    main()
