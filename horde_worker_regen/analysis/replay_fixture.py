"""Build a session-replay fixture from a recorded worker session's stats files and learned footprint store.

The dispatch-world simulator replays a live session as its inputs: the stream of image jobs the worker popped
and the learned footprint store it priced them with. Both come from what a support bundle already carries,
the session's ``job_completed`` stats rows and ``config/vram_footprints.json``. ``horde-log replay-fixture``
writes them as two files:

- :data:`SESSION_JOBS_FILENAME`, one :class:`ReplayJobRow` per image ``job_completed`` row as a JSON line,
  times in seconds from the session's ``session_start`` row, sorted by pop offset. Prompts are never read. A
  row carries job parameters, stage timings, measured device peaks and kudos.
- :data:`FOOTPRINTS_FILENAME`, the learned store with every key's platform rewritten to
  :data:`PLATFORM_PLACEHOLDER`, so a test can substitute its own host's ``sys.platform`` and price the keys.

A session is chosen as ``horde-log sessions`` numbers it, and its stats files are the group whose filename
stamp sits nearest the session's first log line, the same match the duty report makes between a stats session
and a log epoch.

- :func:`build_replay_job_rows` turns a session's stats payloads into rows.
- :func:`platform_neutral_footprint_text` rewrites a footprint store's platform keys.
- :func:`write_replay_fixture` resolves a session in a bundle or logs directory and writes both files.
- :func:`read_replay_job_rows` parses a written job stream back into rows.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from horde_worker_regen.analysis.bundle import LogBundle
from horde_worker_regen.analysis.diagnose import select_sessions
from horde_worker_regen.analysis.duty_log_report import STATS_EPOCH_MATCH_SECONDS
from horde_worker_regen.analysis.footprint_view import default_footprint_store_path
from horde_worker_regen.analysis.session_duty import (
    discover_stats_sessions,
    read_stats_payloads,
    stats_files_local_start,
)
from horde_worker_regen.analysis.sessions import WorkerSession, segment_bundle_sessions
from horde_worker_regen.process_management.jobs.job_tracker import JobStage
from horde_worker_regen.process_management.resources.run_metrics import (
    JobMetricsRecord,
    SessionStartEvent,
    StatsJobCompletedEvent,
)
from horde_worker_regen.process_management.scheduling.workload_kind import WorkloadKind
from horde_worker_regen.stats_operations import default_stats_dir

SESSION_JOBS_FILENAME = "session_jobs.jsonl"
"""The job stream the subcommand writes, one :class:`ReplayJobRow` per line."""
FOOTPRINTS_FILENAME = "footprints.json"
"""The platform-neutral learned footprint store the subcommand writes."""
PLATFORM_PLACEHOLDER = "<platform>"
"""The token a fixture's footprint keys carry where the recording host's ``sys.platform`` stood."""

_JOB_COMPLETED_EVENT: str = StatsJobCompletedEvent.model_fields["event"].default
_SESSION_START_EVENT: str = SessionStartEvent.model_fields["event"].default
_FINALIZED_STAMP = "FINALIZED"
"""The ``stage_timestamps`` entry the job tracker stamps at finalization, beside the :class:`JobStage` names."""
_STATS_DIRNAME = "stats"
"""The stats directory's name in a support bundle, beside its ``logs`` directory."""

_STORE_OBSERVATIONS_FIELD = "observations"
_STORE_KEY_FIELD = "key"
_STORE_PLATFORM_FIELD = "platform"
"""The persisted footprint store's layout: an ``observations`` list whose entries hold a ``key`` with a
``platform``, as :class:`horde_worker_regen.process_management.resources.vram_footprints.LearnedFootprintStore`
writes it."""


class _RecordedPhase(StrEnum):
    """The non-sampling phase names hordelib's metrics collector records in ``phase_seconds``."""

    CLIP_ENCODE = "clip_encode"
    VAE_DECODE = "vae_decode"
    MODEL_UNLOAD = "model_unload"
    PIPELINE_EXECUTE = "pipeline_execute"


class ReplayFixtureError(Exception):
    """Raised when a session, its stats files or its footprint store cannot be turned into a fixture."""


@dataclass(frozen=True)
class ReplayJobRow:
    """Represents one image job of a recorded session as a replay fixture row.

    Offsets are seconds from the session's ``session_start`` row, rounded to milliseconds. Values copied from
    the job's record keep the type the stats file carried, so a fixture rebuilt from the same files is the
    same bytes.
    """

    offset_s: float
    """When the job was popped."""
    job_id: str
    model: str | None
    baseline: str | None
    width: int | None
    height: int | None
    steps: int | None
    batch: int
    """The batch the child measured the job's footprint at, 1 when it reported none."""
    sampler: str | None
    hires_fix: bool
    loras: int
    tis: int
    control_type: str | None
    post_processing: tuple[str, ...]
    faulted: bool
    fault_reason: str | None
    kudos: float | None
    whole_card: bool | None
    batch_count: int
    cfg_scale: float | None
    megapixelsteps: float
    dispatch_hold_s: float | None
    model_load_s: float | None
    lora_wait_s: float | None
    post_processing_chain_s: float | None
    post_processing_encode_s: float | None
    queue_depth_at_dispatch: int | None
    queue_wait_s: float | None
    dispatch_offset_s: float | None
    """When the job entered ``INFERENCE_IN_PROGRESS``."""
    detached_offset_s: float | None
    finalized_offset_s: float | None
    sampling_s: float | None
    steps_completed: int | None
    clip_encode_s: float | None
    vae_decode_s: float | None
    model_unload_s: float | None
    pipeline_execute_s: float | None
    ram_to_vram_s: float
    """The job's host-to-device load seconds, summed over its load events. Zero when it loaded nothing."""
    disk_to_ram_s: float
    """The job's disk-to-host load seconds, summed over its load events. Zero when it loaded nothing."""
    post_processing_s: float | None
    """Seconds from ``POST_PROCESSING`` to ``PENDING_SAFETY_CHECK``, the post-processing lane's window."""
    safety_s: float | None
    peak_device_used_mb: float | None
    peak_resident_weights_mb: float | None
    vram_used_high_water_mb: int | None
    resident_weights_after_job_mb: float | None


_ROW_ADAPTER: TypeAdapter[ReplayJobRow] = TypeAdapter(ReplayJobRow)


@dataclass(frozen=True)
class ReplayFixtureSummary:
    """Represents what a fixture build read and wrote, for the command's report."""

    session_start_epoch: float
    stats_files: tuple[Path, ...]
    footprints_path: Path
    job_count: int
    faulted_count: int
    unpopped_count: int
    """Image records with no pop time, which have no offset and are left out of the stream."""
    rewritten_platforms: tuple[str, ...]
    missing_by_column: dict[str, int]
    """How many rows carry no value in each column a replay leans on."""


_SUMMARY_COLUMNS: tuple[str, ...] = ("sampling_s", "baseline", "dispatch_offset_s", "kudos")


def _stage_offset(record: JobMetricsRecord, stage_name: str, session_start: float) -> float | None:
    """Return when the record first entered ``stage_name``, in seconds from the session start, or None."""
    stamp = record.stage_timestamps.get(stage_name)
    return None if stamp is None else round(stamp - session_start, 3)


def _stage_gap(record: JobMetricsRecord, start_stage: JobStage, end_stage: JobStage) -> float | None:
    """Return the seconds between two stage stamps of the record, or None when either is missing."""
    started = record.stage_timestamps.get(start_stage.name)
    ended = record.stage_timestamps.get(end_stage.name)
    return None if started is None or ended is None else round(ended - started, 3)


def _replay_job_row(record: JobMetricsRecord, *, popped: float, session_start: float) -> ReplayJobRow:
    """Create the fixture row for one image job record."""
    metrics = record.phase_metrics
    footprint = metrics.vram_footprint if metrics is not None else None
    sampling = metrics.sampling if metrics is not None else None
    phase_seconds = metrics.phase_seconds if metrics is not None else {}
    loads = metrics.model_loads if metrics is not None else []
    return ReplayJobRow(
        offset_s=round(popped - session_start, 3),
        job_id=record.job_id,
        model=record.model_name,
        baseline=footprint.baseline if footprint is not None else None,
        width=record.width,
        height=record.height,
        steps=record.steps,
        batch=(footprint.batch_size if footprint is not None else None) or 1,
        sampler=record.sampler_name,
        hires_fix=record.hires_fix,
        loras=record.loras_count,
        tis=record.tis_count,
        control_type=record.control_type,
        post_processing=tuple(record.post_processing),
        faulted=record.faulted,
        fault_reason=record.fault_reason,
        kudos=record.kudos_reward,
        whole_card=record.whole_card,
        batch_count=record.batch_count,
        cfg_scale=record.cfg_scale,
        megapixelsteps=record.megapixelsteps,
        dispatch_hold_s=record.dispatch_hold_seconds,
        model_load_s=record.model_load_seconds,
        lora_wait_s=record.lora_wait_seconds,
        post_processing_chain_s=record.post_processing_chain_seconds,
        post_processing_encode_s=record.post_processing_encode_seconds,
        queue_depth_at_dispatch=record.queue_depth_at_dispatch,
        queue_wait_s=record.queue_wait_seconds,
        dispatch_offset_s=_stage_offset(record, JobStage.INFERENCE_IN_PROGRESS.name, session_start),
        detached_offset_s=_stage_offset(record, JobStage.DETACHED.name, session_start),
        finalized_offset_s=_stage_offset(record, _FINALIZED_STAMP, session_start),
        sampling_s=sampling.duration_seconds if sampling is not None else None,
        steps_completed=sampling.steps_completed if sampling is not None else None,
        clip_encode_s=phase_seconds.get(_RecordedPhase.CLIP_ENCODE),
        vae_decode_s=phase_seconds.get(_RecordedPhase.VAE_DECODE),
        model_unload_s=phase_seconds.get(_RecordedPhase.MODEL_UNLOAD),
        pipeline_execute_s=phase_seconds.get(_RecordedPhase.PIPELINE_EXECUTE),
        ram_to_vram_s=round(sum(load.duration_seconds for load in loads if load.phase == "ram_to_vram"), 3),
        disk_to_ram_s=round(sum(load.duration_seconds for load in loads if load.phase == "disk_to_ram"), 3),
        post_processing_s=_stage_gap(record, JobStage.POST_PROCESSING, JobStage.PENDING_SAFETY_CHECK),
        safety_s=record.safety_seconds,
        peak_device_used_mb=footprint.peak_device_used_mb if footprint is not None else None,
        peak_resident_weights_mb=footprint.peak_resident_weights_mb if footprint is not None else None,
        vram_used_high_water_mb=metrics.vram_used_high_water_mb if metrics is not None else None,
        resident_weights_after_job_mb=footprint.resident_weights_after_job_mb if footprint is not None else None,
    )


@dataclass(frozen=True)
class ReplayJobStream:
    """Represents a session's image jobs as fixture rows, with the start they are offset from."""

    session_start_epoch: float
    rows: list[ReplayJobRow]
    """The rows, sorted by pop offset."""
    unpopped_count: int
    """Image records with no pop time, left out because a row needs a pop offset."""


def build_replay_job_rows(payloads: Iterable[Mapping[str, Any]]) -> ReplayJobStream:
    """Parse a session's stats payloads into replay rows, one per image ``job_completed`` record.

    The first ``session_start`` row is the zero every offset counts from. Records of other workloads are
    skipped, since the dispatch world replays image jobs only.

    Args:
        payloads: The session's stats objects in file order, as :func:`read_stats_payloads` yields them.

    Returns:
        ReplayJobStream: The rows sorted by pop offset, and the session start.

    Raises:
        ReplayFixtureError: A ``job_completed`` row comes before any ``session_start`` row, the session has
            none, or a row of either kind does not validate.
    """
    session_start: float | None = None
    rows: list[ReplayJobRow] = []
    unpopped_count = 0
    for payload in payloads:
        event = payload.get("event")
        try:
            if event == _SESSION_START_EVENT and session_start is None:
                session_start = SessionStartEvent.model_validate(payload).timestamp
                continue
            if event != _JOB_COMPLETED_EVENT:
                continue
            record = StatsJobCompletedEvent.model_validate(payload).job
        except ValidationError as validation_error:
            raise ReplayFixtureError(f"a {event} stats row does not validate: {validation_error}") from None
        if record.workload != WorkloadKind.IMAGE_GENERATION:
            continue
        if session_start is None:
            raise ReplayFixtureError(f"job {record.job_id} completed before the session's session_start row")
        if record.time_popped is None:
            unpopped_count += 1
            continue
        rows.append(_replay_job_row(record, popped=record.time_popped, session_start=session_start))
    if session_start is None:
        raise ReplayFixtureError("the stats files carry no session_start row")
    rows.sort(key=lambda row: row.offset_s)
    return ReplayJobStream(session_start_epoch=session_start, rows=rows, unpopped_count=unpopped_count)


def render_replay_job_rows(rows: Iterable[ReplayJobRow]) -> str:
    """Return the job stream as JSON lines with sorted keys, one row per line."""
    return "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows)


def read_replay_job_rows(path: Path) -> list[ReplayJobRow]:
    """Parse a job stream written by :func:`write_replay_fixture`, in file order.

    Args:
        path: The ``session_jobs.jsonl`` file, or a committed copy of it.

    Returns:
        list[ReplayJobRow]: One row per non-blank line.

    Raises:
        ValidationError: A line is not a row.
    """
    return [_ROW_ADAPTER.validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def platform_neutral_footprint_text(store_text: str) -> tuple[str, list[str]]:
    """Return a footprint store's text with every key's platform replaced by :data:`PLATFORM_PLACEHOLDER`.

    The store prices only keys for the host's own ``sys.platform``, so a fixture recorded on one platform
    carries a placeholder a test substitutes with its own. Nothing else in the store changes.

    Args:
        store_text: A persisted ``vram_footprints.json``.

    Returns:
        tuple[str, list[str]]: The rewritten store, indented with sorted keys, and the platforms it replaced.

    Raises:
        ReplayFixtureError: The text is not a store with an observation list of keyed entries.
    """
    store = json.loads(store_text)
    entries = store.get(_STORE_OBSERVATIONS_FIELD) if isinstance(store, dict) else None
    if not isinstance(entries, list):
        raise ReplayFixtureError(f"the footprint store has no {_STORE_OBSERVATIONS_FIELD} list")
    platforms: set[str] = set()
    for entry in entries:
        key = entry.get(_STORE_KEY_FIELD) if isinstance(entry, dict) else None
        if not isinstance(key, dict) or _STORE_PLATFORM_FIELD not in key:
            raise ReplayFixtureError(f"a footprint store entry has no {_STORE_KEY_FIELD}.{_STORE_PLATFORM_FIELD}")
        platforms.add(str(key[_STORE_PLATFORM_FIELD]))
        key[_STORE_PLATFORM_FIELD] = PLATFORM_PLACEHOLDER
    return json.dumps(store, indent=1, sort_keys=True) + "\n", sorted(platforms)


def default_replay_stats_dir(path: Path) -> Path:
    """Return the stats directory for a bundle or logs directory when none is named.

    Args:
        path: A support bundle's root, its ``logs`` directory, or a worker's own logs directory.

    Returns:
        Path: The bundle's ``stats`` directory when ``path`` is the bundle or its ``logs``, else the worker's.
    """
    for candidate in (path / _STATS_DIRNAME, path.parent / _STATS_DIRNAME):
        if candidate.is_dir():
            return candidate
    return default_stats_dir()


def session_stats_files(session: WorkerSession, stats_dir: Path) -> list[Path]:
    """Return the stats files of the stats session that ``session`` wrote.

    The stats filename stamp and the log are both on the worker host's clock, so the group whose stamp sits
    nearest the session's first log line, within the duty report's epoch tolerance, is the session's.

    Args:
        session: A worker session segmented from the logs.
        stats_dir: The directory holding the stats files.

    Returns:
        list[Path]: The session's stats files, in rotation order.

    Raises:
        ReplayFixtureError: The session has no timestamped line, or no stats session starts near it.
    """
    started = session.start_ts
    if started is None:
        raise ReplayFixtureError(f"session #{session.index} has no timestamped log line")
    best_paths: list[Path] | None = None
    best_distance: float | None = None
    for _session_id, paths in discover_stats_sessions(stats_dir):
        stamp = stats_files_local_start(paths)
        if stamp is None:
            continue
        distance = abs((stamp - started).total_seconds())
        if distance > STATS_EPOCH_MATCH_SECONDS:
            continue
        if best_distance is None or distance < best_distance:
            best_paths, best_distance = paths, distance
    if best_paths is None:
        raise ReplayFixtureError(
            f"no stats session in {stats_dir} starts within {STATS_EPOCH_MATCH_SECONDS:.0f}s of session "
            f"#{session.index} ({started:%Y-%m-%d %H:%M:%S})",
        )
    return best_paths


def select_worker_session(path: Path, *, session_index: int | None, last: bool) -> WorkerSession:
    """Return the worker session ``horde-log sessions`` numbers ``session_index``, or the last one.

    Raises:
        ReplayFixtureError: No worker session matches.
    """
    sessions = segment_bundle_sessions(LogBundle.from_path(path))
    selected = select_sessions(sessions, last=last, session_index=session_index)
    if not selected:
        raise ReplayFixtureError(f"no matching worker session in {path}")
    return selected[0]


def write_replay_fixture(
    path: Path,
    out_dir: Path,
    *,
    session_index: int | None,
    last: bool = False,
    stats_dir: Path | None = None,
    footprints_path: Path | None = None,
) -> ReplayFixtureSummary:
    """Write a session's job stream and platform-neutral footprint store into ``out_dir``.

    Args:
        path: A support bundle's root, its ``logs`` directory, or a worker's logs directory.
        out_dir: Where :data:`SESSION_JOBS_FILENAME` and :data:`FOOTPRINTS_FILENAME` are written; created when
            missing.
        session_index: The worker session's number, as ``horde-log sessions`` lists it.
        last: Take the most recent worker session instead of ``session_index``.
        stats_dir: The stats directory, else :func:`default_replay_stats_dir`.
        footprints_path: The learned footprint store, else the bundle's copy beside ``stats_dir`` or the
            worker's own.

    Returns:
        ReplayFixtureSummary: What was read and written.

    Raises:
        ReplayFixtureError: The session, its stats or its footprint store cannot be resolved or parsed.
    """
    session = select_worker_session(path, session_index=session_index, last=last)
    resolved_stats_dir = stats_dir or default_replay_stats_dir(path)
    stats_files = session_stats_files(session, resolved_stats_dir)
    resolved_footprints_path = footprints_path or default_footprint_store_path(resolved_stats_dir)
    if not resolved_footprints_path.is_file():
        raise ReplayFixtureError(f"learned footprint store not found: {resolved_footprints_path}")

    stream = build_replay_job_rows(read_stats_payloads(stats_files))
    footprint_text, platforms = platform_neutral_footprint_text(
        resolved_footprints_path.read_text(encoding="utf-8"),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SESSION_JOBS_FILENAME).write_text(
        render_replay_job_rows(stream.rows),
        encoding="utf-8",
        newline="\n",
    )
    (out_dir / FOOTPRINTS_FILENAME).write_text(footprint_text, encoding="utf-8", newline="\n")
    row_columns = [asdict(row) for row in stream.rows]
    return ReplayFixtureSummary(
        session_start_epoch=stream.session_start_epoch,
        stats_files=tuple(stats_files),
        footprints_path=resolved_footprints_path,
        job_count=len(stream.rows),
        faulted_count=sum(row.faulted for row in stream.rows),
        unpopped_count=stream.unpopped_count,
        rewritten_platforms=tuple(platforms),
        missing_by_column={
            column: sum(1 for columns in row_columns if columns[column] is None) for column in _SUMMARY_COLUMNS
        },
    )


def render_replay_fixture_summary(summary: ReplayFixtureSummary, out_dir: Path) -> str:
    """Return the command's report of a fixture build."""
    missing = ", ".join(f"{column} {count}" for column, count in summary.missing_by_column.items())
    lines = [
        f"Wrote {out_dir / SESSION_JOBS_FILENAME} ({summary.job_count} jobs, {summary.faulted_count} faulted) "
        f"and {out_dir / FOOTPRINTS_FILENAME}.",
        f"Stats files: {', '.join(stats_file.name for stats_file in summary.stats_files)}; "
        f"session_start epoch {summary.session_start_epoch}.",
        f"Footprint store: {summary.footprints_path}; platforms rewritten: "
        f"{', '.join(summary.rewritten_platforms) or 'none'}.",
        f"Rows missing a value: {missing}.",
    ]
    if summary.unpopped_count:
        lines.append(f"Left out {summary.unpopped_count} image records with no pop time.")
    return "\n".join(lines)


__all__ = [
    "FOOTPRINTS_FILENAME",
    "PLATFORM_PLACEHOLDER",
    "SESSION_JOBS_FILENAME",
    "ReplayFixtureError",
    "ReplayFixtureSummary",
    "ReplayJobRow",
    "ReplayJobStream",
    "build_replay_job_rows",
    "default_replay_stats_dir",
    "platform_neutral_footprint_text",
    "read_replay_job_rows",
    "render_replay_fixture_summary",
    "render_replay_job_rows",
    "select_worker_session",
    "session_stats_files",
    "write_replay_fixture",
]
