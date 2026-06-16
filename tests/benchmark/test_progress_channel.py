"""Tests for the benchmark progress channel: event round-trips, the JSONL sink, the tailer, rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from horde_worker_regen.benchmark.progress_channel import (
    BENCHMARK_PROGRESS_PROTOCOL_VERSION,
    PROGRESS_FILENAME,
    BenchmarkProgressEvent,
    JsonlProgressSink,
    LevelFinished,
    LevelLiveSnapshot,
    LevelProgress,
    LevelStarted,
    MultiProgressSink,
    ProgressSink,
    ProgressTailer,
    RampFinished,
    RampStarted,
    parse_progress_event,
    read_progress_events,
)
from horde_worker_regen.benchmark.progress_console import format_progress_event


class _RecordingSink(ProgressSink):
    """A test sink that keeps every emitted event in order."""

    def __init__(self) -> None:
        self.events: list[BenchmarkProgressEvent] = []

    def emit(self, event: BenchmarkProgressEvent) -> None:
        self.events.append(event)


def _all_event_kinds() -> list[BenchmarkProgressEvent]:
    return [
        RampStarted(run_id="r", num_levels=2, tiers=["sd15"], process_mode="fake", gpu_name="X", total_vram_mb=24000),
        LevelStarted(
            level_id="A-sd15",
            description="baseline",
            stage="A",
            tier="sd15",
            axis="baseline",
            level_index=0,
            num_levels=2,
            jobs_expected=4,
            timeout_seconds=900.0,
        ),
        LevelProgress(
            level_id="A-sd15",
            jobs_completed=2,
            jobs_faulted=0,
            jobs_expected=4,
            iterations_per_second=8.5,
            vram_used_mb=3000,
            gpu_busy_percent=88.0,
            elapsed_seconds=12.0,
        ),
        LevelFinished(
            level_id="A-sd15",
            outcome="passed",
            advisories=["note"],
            its_p50=9.1,
            gpu_busy_percent=85.0,
            vram_used_high_water_mb=3100,
        ),
        RampFinished(
            run_id="r",
            levels_passed=2,
            levels_total=2,
            report_path="x/report.json",
            suggested_bridge_data_yaml="max_threads: 2",
        ),
    ]


def test_each_event_json_round_trips() -> None:
    """Every event serializes and parses back into the same concrete type and value."""
    for event in _all_event_kinds():
        restored = parse_progress_event(event.model_dump_json())
        assert restored is not None
        assert type(restored) is type(event)
        assert restored == event
        assert restored.protocol_version == BENCHMARK_PROGRESS_PROTOCOL_VERSION


def test_parse_tolerates_blank_garbage_and_unknown() -> None:
    """A blank, malformed, non-object, or unknown-kind line parses to None rather than raising."""
    assert parse_progress_event("") is None
    assert parse_progress_event("   ") is None
    assert parse_progress_event("{not json") is None
    assert parse_progress_event("[1, 2, 3]") is None
    assert parse_progress_event('{"kind": "not_a_real_kind"}') is None


def test_jsonl_sink_and_read_round_trip(tmp_path: Path) -> None:
    """Events written by the JSONL sink read back in order with their types intact."""
    path = tmp_path / PROGRESS_FILENAME
    sink = JsonlProgressSink(path)
    events = _all_event_kinds()
    for event in events:
        sink.emit(event)
    sink.close()

    restored = read_progress_events(path)
    assert [type(event) for event in restored] == [type(event) for event in events]


def test_multi_sink_fans_out(tmp_path: Path) -> None:
    """A multi-sink delivers each event to every wrapped sink."""
    recorder = _RecordingSink()
    jsonl = JsonlProgressSink(tmp_path / PROGRESS_FILENAME)
    multi = MultiProgressSink([recorder, jsonl])

    event = RampStarted(run_id="r", num_levels=1)
    multi.emit(event)
    multi.close()

    assert recorder.events == [event]
    assert len(read_progress_events(tmp_path / PROGRESS_FILENAME)) == 1


def test_tailer_returns_only_new_events(tmp_path: Path) -> None:
    """The tailer yields each appended event exactly once and nothing on a quiet poll."""
    path = tmp_path / PROGRESS_FILENAME
    sink = JsonlProgressSink(path)
    tailer = ProgressTailer(path)

    assert tailer.poll() == []

    sink.emit(RampStarted(run_id="r", num_levels=1))
    first = tailer.poll()
    assert len(first) == 1 and isinstance(first[0], RampStarted)

    assert tailer.poll() == []

    sink.emit(LevelStarted(level_id="A", num_levels=1))
    second = tailer.poll()
    assert len(second) == 1 and isinstance(second[0], LevelStarted)


def test_tailer_tolerates_partial_line(tmp_path: Path) -> None:
    """A line written across two polls is buffered and parsed once it is complete."""
    path = tmp_path / PROGRESS_FILENAME
    tailer = ProgressTailer(path)

    complete_line = RampStarted(run_id="r", num_levels=1).model_dump_json()
    path.write_text(complete_line + "\n" + '{"kind": "level_star', encoding="utf-8")
    first = tailer.poll()
    assert len(first) == 1 and isinstance(first[0], RampStarted)

    with path.open("a", encoding="utf-8") as handle:
        handle.write('ted", "level_id": "A"}\n')
    second = tailer.poll()
    assert len(second) == 1 and isinstance(second[0], LevelStarted)


def test_live_snapshot_round_trips() -> None:
    """The latest-only live snapshot survives a JSON round-trip (the runner-to-controller hand-off)."""
    snapshot = LevelLiveSnapshot(
        jobs_completed=3,
        jobs_faulted=1,
        iterations_per_second=7.5,
        vram_used_mb=3000,
        gpu_busy_percent=80.0,
        elapsed_seconds=20.0,
    )
    restored = LevelLiveSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored == snapshot


def test_format_renders_each_event() -> None:
    """Every event renders to a non-empty console line."""
    for event in _all_event_kinds():
        line = format_progress_event(event)
        assert line is not None and line.strip()


def test_progress_line_shows_phase_and_restarts() -> None:
    """A still-starting level (no jobs yet) renders its phase and surfaces any process restarts."""
    event = LevelProgress(
        level_id="A-sd15",
        jobs_completed=0,
        jobs_expected=4,
        elapsed_seconds=42.0,
        phase="initializing inference process",
        process_summary="inf#1=PROCESS_STARTING safety#0=PROCESS_STARTING",
        num_process_recoveries=3,
    )
    line = format_progress_event(event)
    assert line is not None
    assert "initializing inference process" in line
    assert "3 process restart" in line
    # The per-process detail is opt-in (verbose) only.
    assert "inf#1=PROCESS_STARTING" not in line
    assert "inf#1=PROCESS_STARTING" in (format_progress_event(event, verbose=True) or "")


@pytest.mark.e2e
def test_fake_ramp_emits_lifecycle_events(tmp_path: Path) -> None:
    """A fake-mode ramp writes the ramp/level lifecycle events to progress.jsonl."""
    from horde_worker_regen.benchmark.controller import BenchmarkController
    from horde_worker_regen.benchmark.ladder import LadderOptions, build_default_ladder

    ladder = build_default_ladder(
        LadderOptions(
            tiers=["sd15"],
            jobs_per_level=2,
            include_concurrency=False,
            include_features=False,
            include_alchemy=False,
        ),
    )
    sink = JsonlProgressSink(tmp_path / PROGRESS_FILENAME)
    BenchmarkController(ladder, tmp_path, process_mode="fake", progress_sink=sink).run()

    emitted_types = {type(event) for event in read_progress_events(tmp_path / PROGRESS_FILENAME)}
    assert RampStarted in emitted_types
    assert LevelStarted in emitted_types
    assert LevelFinished in emitted_types
    assert RampFinished in emitted_types
