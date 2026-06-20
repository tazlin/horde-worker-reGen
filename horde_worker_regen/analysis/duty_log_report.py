"""Epoch-aware GPU duty-cycle report over an appended ``bridge.log``.

``bridge.log`` is appended across worker restarts, so a single file holds many sessions. Comparing a
config change means reading each *session epoch* separately and tying its duty-cycle numbers to the
config that produced them. This tool splits the log on the once-per-session process-manager init
banner, then for each epoch parses the periodic ``GPU duty cycle`` lines (and the reload-churn /
per-job-gap attribution they now carry), the effective worker config, and any disk-pressure warnings,
and prints a compact per-epoch verdict plus the biggest loss buckets.

Pure stdlib (re + datetime + argparse) so it imports without the inference stack; usable from the CLI
(``horde-duty-report``) or the benchmark/TUI. The grep-friendly log format is the contract: see the
regexes below, which mirror what ``process_manager._log_duty_cycle_summary`` and
``status_reporter`` emit.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")
_TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# Once-per-session boundary: the worker's process manager constructs exactly once per launch and logs
# several ``__init__`` lines in a burst at startup. The worker-identity "not yet registered" line is a
# fallback for logs where the manager lines are absent.
_EPOCH_BOUNDARY_RE = re.compile(r"process_management\.process_manager:__init__:")
_EPOCH_BOUNDARY_FALLBACK_RE = re.compile(r"worker_identity:_verify_worker_names_owned:.*is not yet registered")
_EPOCH_COLLAPSE_SECONDS = 30.0
"""Boundary lines closer together than this belong to the same startup burst, not a new epoch."""

_DUTY_RE = re.compile(
    r"GPU duty cycle (?P<duty>\d+)% over last (?P<window>\d+)s "
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
    r"moderate_performance_mode: (?P<mpm>\w+) \| high_memory_mode: (?P<hmm>\w+)",
)
_IDENTITY_RE = re.compile(
    r"dreamer_name: (?P<name>[^|]+?) \|.*?num_models: (?P<num_models>\d+).*?"
    r"max_power: (?P<max_power>\d+).*?max_threads: (?P<max_threads>\d+) \| "
    r"queue_size: (?P<queue_size>\d+) \| safety_on_gpu: (?P<safety>\w+)",
)
_DISK_RE = re.compile(r"Low disk space on .*?: (?P<free>[\d.]+) GB free")

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
            parts.append(f"threads={self.max_threads}")
        if self.queue_size is not None:
            parts.append(f"queue={self.queue_size}")
        if self.max_power is not None:
            parts.append(f"max_power={self.max_power}")
        flags = {
            "unload_often": self.unload_models_from_vram_often,
            "high_perf": self.high_performance_mode,
            "high_mem": self.high_memory_mode,
            "safety_gpu": self.safety_on_gpu,
        }
        parts.extend(name for name, value in flags.items() if value)
        return ", ".join(parts) if parts else "(config not seen)"


@dataclass
class EpochReport:
    """The duty-cycle picture for one session epoch."""

    index: int
    start: datetime | None
    end: datetime | None
    config: EpochConfig
    windows: list[DutyWindow] = field(default_factory=list)
    min_disk_free_gb: float | None = None

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

    for line in lines:
        ts = _parse_ts(line)
        if ts is not None:
            last_ts = ts

        window = parse_duty_window(line, ts)
        if window is not None:
            windows.append(window)
            continue

        config_match = _CONFIG_RE.search(line)
        if config_match is not None:
            config.unload_models_from_vram_often = config_match.group("unload") == "True"
            config.high_performance_mode = config_match.group("hpm") == "True"
            config.moderate_performance_mode = config_match.group("mpm") == "True"
            config.high_memory_mode = config_match.group("hmm") == "True"
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

    start = _parse_ts(lines[0]) if lines else None
    return EpochReport(
        index=index,
        start=windows[0].timestamp if windows else start,
        end=windows[-1].timestamp if windows else last_ts,
        config=config,
        windows=windows,
        min_disk_free_gb=min_disk,
    )


def analyze_log(lines: list[str]) -> list[EpochReport]:
    """Split ``lines`` into epochs and build a report for each that has at least one duty window."""
    reports = [build_epoch_report(i, epoch_lines) for i, epoch_lines in enumerate(split_into_epochs(lines))]
    return [report for report in reports if report.windows]


def _verdict(report: EpochReport) -> str:
    """A one-line plain-language read of the epoch's headline number."""
    mean = report.mean_duty()
    if mean is None:
        return "no duty-cycle data"
    if mean >= _DUTY_TARGET:
        return f"at target ({mean:.0f}% >= {int(_DUTY_TARGET)}%)"
    busy = report.mean_busy()
    detail = f"mean {mean:.0f}%"
    if busy is not None:
        idle = max(0.0, 100.0 - busy)
        partial = max(0.0, busy - mean)
        detail += f"; ~{idle:.0f}% idle (hand-off/no-work) + ~{partial:.0f}% partial-utilization"
    return f"below {int(_DUTY_TARGET)}% target: {detail}"


def render_report(reports: list[EpochReport]) -> str:
    """Render the per-epoch report as a human-readable block."""
    if not reports:
        return "No session epochs with duty-cycle data found in the log."

    out: list[str] = []
    for report in reports:
        start = report.start.strftime("%Y-%m-%d %H:%M") if report.start else "?"
        end = report.end.strftime("%H:%M") if report.end else "?"
        duration = report.duration_minutes
        duration_str = f"{duration:.0f} min" if duration is not None else "? min"
        out.append(f"== Epoch {report.index} | {start}-{end} ({duration_str}) | {len(report.windows)} windows ==")
        out.append(f"   config: {report.config.summary()}")
        out.append(f"   verdict: {_verdict(report)}")

        values = report.duty_values()
        if values:
            out.append(f"   duty: mean {statistics.mean(values):.0f}%  min {min(values)}%  max {max(values)}%")
            bands = report.band_distribution()
            out.append("   bands: " + "  ".join(f"{label} {count}" for label, count in bands.items()))

        gaps = report.mean_gaps()
        if gaps:
            top = "  ".join(f"{label} {seconds:.1f}s/job" for label, seconds in list(gaps.items())[:4])
            out.append(f"   top per-job gaps: {top}")

        churn = report.churn_totals()
        if churn:
            out.append("   reload churn: " + "  ".join(f"{label} {count}" for label, count in churn.items()))
        else:
            out.append("   reload churn: none reported (pre-instrumentation log, or no churn)")

        if report.min_disk_free_gb is not None:
            out.append(f"   (!) disk pressure: dipped to {report.min_disk_free_gb:.1f} GB free (can stall downloads)")
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
        "min_duty_percent": min(report.duty_values()) if report.windows else None,
        "max_duty_percent": max(report.duty_values()) if report.windows else None,
        "band_distribution": report.band_distribution(),
        "top_per_job_gaps": report.mean_gaps(),
        "churn_totals": report.churn_totals(),
        "min_disk_free_gb": report.min_disk_free_gb,
        "config": report.config.__dict__,
    }


def main() -> None:
    """CLI entry point: parse a bridge.log and print per-epoch duty-cycle reports."""
    parser = argparse.ArgumentParser(description="Epoch-aware GPU duty-cycle report over a bridge.log.")
    parser.add_argument(
        "log",
        nargs="?",
        default="logs/bridge.log",
        type=Path,
        help="Path to the bridge.log to analyze (default: logs/bridge.log).",
    )
    parser.add_argument("--last", action="store_true", help="Only report the most recent epoch.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text.")
    args = parser.parse_args()

    if not args.log.exists():
        parser.error(f"log file not found: {args.log}")

    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    reports = analyze_log(lines)
    if args.last and reports:
        reports = reports[-1:]

    if args.json:
        print(json.dumps([_report_to_dict(r) for r in reports], indent=2))
    else:
        print(render_report(reports))


if __name__ == "__main__":
    main()
