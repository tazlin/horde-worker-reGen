"""Structured, durable progress events for a benchmark ramp.

A ramp is otherwise dark for the length of a level (up to many minutes of real inference). The controller
emits the events defined here through a [`ProgressSink`][horde_worker_regen.benchmark.progress_channel.ProgressSink];
the default sink appends them as newline-delimited JSON to ``progress.jsonl`` in the run directory, so the
CLI and the TUI can render a live view by tailing one file: durable, replayable, and decoupled from the
process that produced it.

This mirrors the *shape* of the worker's supervisor channel (pure-data, protocol-versioned models) but uses
a file transport instead of a pipe, because a benchmark is a bounded batch job whose progress is worth
keeping on disk. Events are deliberately lean and JSON-round-trippable.
"""

from __future__ import annotations

import abc
import enum
import json
import time
from pathlib import Path
from typing import Annotated, Literal

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

PROGRESS_FILENAME = "progress.jsonl"
"""The progress event log written into a run's output directory."""

BENCHMARK_PROGRESS_PROTOCOL_VERSION = 3
"""Bumped when the event schema changes incompatibly; stamped into every event and checked by readers.

v2 added ``RAMP_PLANNED`` (the per-level resource-requirements plan). v3 adds
``RampFinished.suggestion_decisions`` (per-setting recommendation provenance). Readers tolerate unknown
kinds and unknown fields default in, so an older reader simply ignores the additions.
"""


class ProgressEventKind(enum.StrEnum):
    """Discriminates the kind of a benchmark progress event."""

    RAMP_STARTING = "ramp_starting"
    RAMP_STARTED = "ramp_started"
    RAMP_PLANNED = "ramp_planned"
    LEVEL_STARTED = "level_started"
    LEVEL_PROGRESS = "level_progress"
    LEVEL_FINISHED = "level_finished"
    RAMP_FINISHED = "ramp_finished"


class BenchmarkProgressEvent(BaseModel):
    """Base for all progress events: the protocol version and an emission timestamp."""

    protocol_version: int = BENCHMARK_PROGRESS_PROTOCOL_VERSION
    timestamp: float = Field(default_factory=time.time)


class RampStarting(BenchmarkProgressEvent):
    """Emitted immediately when a ramp process begins, before the slow import + hardware-probe phase.

    ``RampStarted`` cannot be emitted until the machine has been detected and the ladder built, which
    requires importing torch/hordelib (tens of seconds cold) and probing the GPU. That whole window is
    otherwise dark: the progress file does not exist yet and the redirected console is buffered. This
    early heartbeat creates the progress file and gives the TUI a visible startup phase to render so a
    slow or wedged startup reads as motion rather than a hang.
    """

    kind: Literal[ProgressEventKind.RAMP_STARTING] = ProgressEventKind.RAMP_STARTING
    run_id: str = ""
    process_mode: str = "real"
    phase: str = ""
    """A human-readable description of what the starting process is doing (e.g. "detecting hardware")."""


class RampStarted(BenchmarkProgressEvent):
    """Emitted once at the start of a ramp, carrying the run identity and machine summary."""

    kind: Literal[ProgressEventKind.RAMP_STARTED] = ProgressEventKind.RAMP_STARTED
    run_id: str = ""
    num_levels: int = 0
    tiers: list[str] = Field(default_factory=list)
    process_mode: str = "real"
    gpu_name: str | None = None
    total_vram_mb: int | None = None


class LevelPlanRow(BaseModel):
    """One level's projected resource requirements and predicted run/skip verdict.

    A compact, JSON-round-trippable projection of
    [`LevelRequirements`][horde_worker_regen.benchmark.requirements.LevelRequirements] plus the verdict
    the controller's pre-flight would reach on this machine, so the preview and the actual ramp agree.
    """

    level_id: str
    stage: str = ""
    tier: str = ""
    estimated_vram_mb: int | None = None
    min_disk_free_gb: float = 0.0
    requires_network: bool = False
    requires_civitai_key: bool = False
    features: list[str] = Field(default_factory=list)
    will_run: bool = True
    verdict: str = ""
    """Empty when the level will run; otherwise the skip reason."""


_PLAN_JSON_BEGIN = "<<<HORDE_BENCHMARK_PLAN_JSON>>>"
_PLAN_JSON_END = "<<<END_HORDE_BENCHMARK_PLAN_JSON>>>"


def encode_plan_rows(rows: list[LevelPlanRow]) -> str:
    """Serialise plan rows for ``plan --json``, wrapped in sentinels so a reader can isolate the payload.

    The benchmark imports the inference stack and the project routes loguru/logfire to stdout, so a
    subprocess's stdout is *not* guaranteed to be pure JSON: it can carry log lines and library banners
    before, around, and after our output. Bracketing the JSON with unmistakable markers lets the reader
    extract exactly our payload regardless of that noise (see `decode_plan_rows`).
    """
    payload = json.dumps([row.model_dump() for row in rows])
    return f"{_PLAN_JSON_BEGIN}{payload}{_PLAN_JSON_END}"


def decode_plan_rows(raw_stdout: str) -> list[LevelPlanRow]:
    """Extract and parse the plan rows emitted by `encode_plan_rows` from a noisy subprocess stdout.

    Raises:
        ValueError: if the sentinel-delimited payload is absent or unparseable.
    """
    start = raw_stdout.find(_PLAN_JSON_BEGIN)
    end = raw_stdout.find(_PLAN_JSON_END, start + len(_PLAN_JSON_BEGIN)) if start != -1 else -1
    if start == -1 or end == -1:
        raise ValueError("plan JSON markers not found in output")
    payload = raw_stdout[start + len(_PLAN_JSON_BEGIN) : end]
    try:
        items = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValueError(f"plan JSON payload is malformed: {e}") from e
    return [LevelPlanRow.model_validate(item) for item in items]


class RampPlanned(BenchmarkProgressEvent):
    """Emitted once before the first level, carrying the resource plan for every ladder level.

    Lets a tailing ``monitor``/TUI show, up front, what each level needs and whether it will run on
    this machine, instead of learning only when a level finishes as ``SKIPPED``.
    """

    kind: Literal[ProgressEventKind.RAMP_PLANNED] = ProgressEventKind.RAMP_PLANNED
    run_id: str = ""
    rows: list[LevelPlanRow] = Field(default_factory=list)


class LevelStarted(BenchmarkProgressEvent):
    """Emitted when a level begins (including skipped levels, which finish immediately after)."""

    kind: Literal[ProgressEventKind.LEVEL_STARTED] = ProgressEventKind.LEVEL_STARTED
    level_id: str
    description: str = ""
    stage: str = ""
    tier: str = ""
    axis: str = ""
    level_index: int = 0
    num_levels: int = 0
    jobs_expected: int | None = None
    timeout_seconds: float | None = None


class LevelProgress(BenchmarkProgressEvent):
    """Emitted periodically while a level runs, carrying its latest live metrics."""

    kind: Literal[ProgressEventKind.LEVEL_PROGRESS] = ProgressEventKind.LEVEL_PROGRESS
    level_id: str
    jobs_completed: int = 0
    jobs_faulted: int = 0
    jobs_expected: int | None = None
    iterations_per_second: float | None = None
    vram_used_mb: int | None = None
    gpu_busy_percent: float | None = None
    elapsed_seconds: float = 0.0
    phase: str = ""
    """What the worker is doing right now (e.g. "initializing inference process", "running inference"),
    so a long cold start before the first job reads as motion rather than a hang."""
    process_summary: str = ""
    """Compact per-process state line, surfaced under extra verbosity."""
    num_process_recoveries: int = 0
    """Cumulative child-process restarts so far; a non-zero value during startup is the visible tell of
    a respawn storm (rather than a merely slow level)."""


class LevelFinished(BenchmarkProgressEvent):
    """Emitted when a level concludes, carrying its outcome and headline statistics."""

    kind: Literal[ProgressEventKind.LEVEL_FINISHED] = ProgressEventKind.LEVEL_FINISHED
    level_id: str
    outcome: str = ""
    reasons: list[str] = Field(default_factory=list)
    advisories: list[str] = Field(default_factory=list)
    its_p50: float | None = None
    gpu_busy_percent: float | None = None
    vram_used_high_water_mb: int | None = None
    num_findings: int = 0


class SuggestionDecisionRow(BaseModel):
    """One suggested setting's provenance, projected for the live stream.

    A lean, import-light mirror of
    [`SuggestionDecision`][horde_worker_regen.benchmark.report.SuggestionDecision] (the controller maps
    one to the other when emitting) so a tailing TUI/console can show *why* each value holds without
    importing the report/hordelib chain. ``value_text`` is pre-rendered (``on``/``off``/list) to keep
    this purely presentational.
    """

    setting: str
    value_text: str = ""
    basis: str = ""
    basis_label: str = ""
    detail: str = ""


class RampFinished(BenchmarkProgressEvent):
    """Emitted once at the end of a ramp, carrying the totals and the synthesized recommendation."""

    kind: Literal[ProgressEventKind.RAMP_FINISHED] = ProgressEventKind.RAMP_FINISHED
    run_id: str = ""
    levels_passed: int = 0
    levels_total: int = 0
    num_findings: int = 0
    report_path: str | None = None
    suggested_bridge_data_yaml: str = ""
    suggestion_decisions: list[SuggestionDecisionRow] = Field(default_factory=list)
    """Per-setting provenance for the recommendation; empty for runs from before protocol v3."""
    consistency_warnings: list[str] = Field(default_factory=list)
    """Messages from the recommendation's self-consistency check (empty when fully grounded)."""


AnyProgressEvent = Annotated[
    RampStarting | RampStarted | RampPlanned | LevelStarted | LevelProgress | LevelFinished | RampFinished,
    Field(discriminator="kind"),
]
"""The union of all concrete progress events, discriminated by ``kind``."""

_EVENT_MODEL_BY_KIND: dict[str, type[BenchmarkProgressEvent]] = {
    ProgressEventKind.RAMP_STARTING: RampStarting,
    ProgressEventKind.RAMP_STARTED: RampStarted,
    ProgressEventKind.RAMP_PLANNED: RampPlanned,
    ProgressEventKind.LEVEL_STARTED: LevelStarted,
    ProgressEventKind.LEVEL_PROGRESS: LevelProgress,
    ProgressEventKind.LEVEL_FINISHED: LevelFinished,
    ProgressEventKind.RAMP_FINISHED: RampFinished,
}


class LevelLiveSnapshot(BaseModel):
    """Latest-only live metrics for an in-progress level, written by the level runner and tailed up.

    This is the hand-off between the isolated level subprocess (which alone sees the running worker's
    metrics) and the controller (which republishes it as a :class:`LevelProgress` event).
    """

    jobs_completed: int = 0
    jobs_faulted: int = 0
    iterations_per_second: float | None = None
    vram_used_mb: int | None = None
    gpu_busy_percent: float | None = None
    elapsed_seconds: float = 0.0
    phase: str = ""
    """What the worker is doing right now; see :class:`LevelProgress.phase`."""
    process_summary: str = ""
    """Compact per-process state line; see :class:`LevelProgress.process_summary`."""
    num_process_recoveries: int = 0
    """Cumulative child-process restarts; see :class:`LevelProgress.num_process_recoveries`."""


def parse_progress_event(raw_line: str) -> BenchmarkProgressEvent | None:
    """Parse one JSONL line into its concrete event, or None if it is blank/garbage/unknown.

    Tolerant by design: a partially written or malformed line yields None rather than raising, so a
    reader tailing a file mid-write is never disturbed.
    """
    stripped = raw_line.strip()
    if not stripped:
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    kind = data.get("kind")
    if not isinstance(kind, str):
        return None
    model = _EVENT_MODEL_BY_KIND.get(kind)
    if model is None:
        return None
    try:
        return model.model_validate(data)
    except ValidationError:
        return None


def read_progress_events(path: Path) -> list[BenchmarkProgressEvent]:
    """Return every parseable event from a progress file (for replaying a finished run)."""
    if not path.exists():
        return []
    events: list[BenchmarkProgressEvent] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        event = parse_progress_event(raw_line)
        if event is not None:
            events.append(event)
    return events


class ProgressTailer:
    """Incrementally reads new events from a growing progress file, surviving partial lines.

    Tracks a byte offset and buffers an incomplete trailing line, so a writer appending mid-line never
    causes a dropped or duplicated event. :meth:`poll` returns only events appended since the last call.
    """

    def __init__(self, path: Path) -> None:
        """Wrap the progress file to tail (which need not exist yet)."""
        self._path = path
        self._offset = 0
        self._partial_line = ""

    def poll(self) -> list[BenchmarkProgressEvent]:
        """Return the events appended since the previous poll (empty if none, or the file is absent)."""
        if not self._path.exists():
            return []
        try:
            with self._path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
                self._offset = handle.tell()
        except OSError as read_error:
            logger.debug(f"Could not tail progress file {self._path}: {read_error}")
            return []

        combined = self._partial_line + chunk
        lines = combined.split("\n")
        self._partial_line = lines.pop()

        events: list[BenchmarkProgressEvent] = []
        for raw_line in lines:
            event = parse_progress_event(raw_line)
            if event is not None:
                events.append(event)
        return events


class ProgressSink(abc.ABC):
    """A destination for benchmark progress events.

    Subclass Integration:
        Implement :meth:`emit` to deliver one event. :meth:`close` is optional and defaults to a no-op.
        Emission must never raise into the controller; sinks that touch external resources should
        swallow and log their own errors.
    """

    @abc.abstractmethod
    def emit(self, event: BenchmarkProgressEvent) -> None:
        """Deliver one progress event."""

    def close(self) -> None:  # noqa: B027 - intentionally an optional no-op hook, not abstract
        """Release any resources held by the sink (no-op by default)."""


class NullProgressSink(ProgressSink):
    """A sink that discards every event (the default when no progress stream is wanted)."""

    def emit(self, event: BenchmarkProgressEvent) -> None:
        """Discard the event."""


class MultiProgressSink(ProgressSink):
    """A sink that fans one event out to several sinks (e.g. a durable file plus a live console)."""

    def __init__(self, sinks: list[ProgressSink]) -> None:
        """Wrap the ordered sinks each event is delivered to."""
        self._sinks = list(sinks)

    def emit(self, event: BenchmarkProgressEvent) -> None:
        """Deliver the event to every wrapped sink in order."""
        for sink in self._sinks:
            sink.emit(event)

    def close(self) -> None:
        """Close every wrapped sink in order."""
        for sink in self._sinks:
            sink.close()


class JsonlProgressSink(ProgressSink):
    """A sink that appends each event as one JSON line to a file (the durable default transport).

    Opens the file per event so a reader always sees a flushed, complete line and a crash loses at most
    the in-flight write. Events are infrequent (order one per second), so the per-event open is cheap.
    """

    def __init__(self, path: Path) -> None:
        """Create the sink, ensuring the parent directory of ``path`` exists."""
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """The progress file this sink writes to."""
        return self._path

    def emit(self, event: BenchmarkProgressEvent) -> None:
        """Append the event as a single JSON line, swallowing IO errors (progress must not break the run)."""
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
        except OSError as write_error:
            logger.debug(f"Could not write progress event to {self._path}: {write_error}")


__all__ = [
    "BENCHMARK_PROGRESS_PROTOCOL_VERSION",
    "PROGRESS_FILENAME",
    "AnyProgressEvent",
    "BenchmarkProgressEvent",
    "JsonlProgressSink",
    "LevelFinished",
    "LevelLiveSnapshot",
    "LevelPlanRow",
    "LevelProgress",
    "LevelStarted",
    "MultiProgressSink",
    "NullProgressSink",
    "ProgressEventKind",
    "ProgressSink",
    "ProgressTailer",
    "RampFinished",
    "RampPlanned",
    "RampStarted",
    "RampStarting",
    "SuggestionDecisionRow",
    "decode_plan_rows",
    "encode_plan_rows",
    "parse_progress_event",
    "read_progress_events",
]
