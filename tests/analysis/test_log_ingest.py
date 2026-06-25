"""Unit tests for log ingestion: the loguru line parse and its tolerance of how files are written.

The worker writes ``bridge.log`` through a multiprocess queue that leaves NUL bytes and torn lines, and
rotates sinks into ``.zip`` archives. These tests pin that the reader survives all of it.
"""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

from horde_worker_regen.analysis.log_ingest import parse_lines, parse_ts, read_records

_GOOD = (
    "2026-06-24 18:30:38.847 | ERROR    | "
    "horde_worker_regen.process_management.lifecycle.process_lifecycle:_reap_if_crashed:962 - "
    "process 1 exited unexpectedly (exitcode=1)"
)
_STARTUP = "2026-06-24 18:09:40.795 | CRITICAL | inference_1:startup - worker child crashed before its log was ready:"


class TestLineParsing:
    """Parsing the canonical ``ts | LEVEL | name:function:line - message`` format and its variants."""

    def test_parses_full_record(self) -> None:
        """A normal orchestrator line yields all structured fields."""
        (record,) = parse_lines([_GOOD], Path("bridge.log"))
        assert record.level == "ERROR"
        assert record.name == "horde_worker_regen.process_management.lifecycle.process_lifecycle"
        assert record.function == "_reap_if_crashed"
        assert record.lineno == 962
        assert "exitcode=1" in record.message
        assert record.timestamp is not None

    def test_startup_location_without_lineno(self) -> None:
        """The pre-sink crash backstop logs ``role:startup`` with no line number; it still parses."""
        (record,) = parse_lines([_STARTUP], Path("bridge_inference_1_startup.log"))
        assert record.name == "inference_1"
        assert record.function == "startup"
        assert record.lineno is None

    def test_continuation_lines_fold_into_prior_record(self) -> None:
        """A traceback (no timestamp prefix) attaches to the record it followed, not lost."""
        lines = [_STARTUP, "Traceback (most recent call last):", '  File "x.py", line 1', "AssertionError: boom"]
        (record,) = parse_lines(lines, Path("bridge_inference_1_startup.log"))
        assert "AssertionError: boom" in record.full_text
        assert record.continuation[-1] == "AssertionError: boom"

    def test_orphan_continuation_before_any_record_is_dropped(self) -> None:
        """Leading timestamp-less noise with no record to attach to does not crash or fabricate a record."""
        assert parse_lines(["junk with no head", "more junk"], Path("x.log")) == []

    def test_parse_ts_returns_none_for_non_timestamp(self) -> None:
        """A line without a leading timestamp yields None rather than raising."""
        assert parse_ts("no timestamp here") is None


class TestReadingFiles:
    """Reading from the real on-disk shapes: NUL bytes, plain files, zip rotations, and gzip."""

    def test_nul_bytes_are_stripped(self, tmp_path: Path) -> None:
        """NUL bytes from the enqueue=True multiprocess writer do not break parsing."""
        path = tmp_path / "bridge.log"
        path.write_bytes(_GOOD.encode("utf-8") + b"\x00\x00\n")
        (record,) = read_records(path)
        assert record.lineno == 962

    def test_reads_zip_rotation(self, tmp_path: Path) -> None:
        """A loguru ``.zip`` rotation is read transparently."""
        archive = tmp_path / "bridge.2026-06-22_00-00-00.log.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("bridge.2026-06-22_00-00-00.log", _GOOD + "\n")
        (record,) = read_records(archive)
        assert record.name.endswith("process_lifecycle")

    def test_reads_gzip(self, tmp_path: Path) -> None:
        """A ``.gz`` archive is read transparently."""
        archive = tmp_path / "bridge.log.gz"
        with gzip.open(archive, "wt", encoding="utf-8") as handle:
            handle.write(_GOOD + "\n")
        (record,) = read_records(archive)
        assert record.lineno == 962

    def test_records_sorted_by_timestamp_across_sources(self, tmp_path: Path) -> None:
        """Reading the active log plus an older rotation returns one stream in time order."""
        older = "2026-06-24 18:00:00.000 | INFO     | a.b:c:1 - older"
        newer = "2026-06-24 18:30:00.000 | INFO     | a.b:c:2 - newer"
        active = tmp_path / "bridge.log"
        active.write_text(newer + "\n", encoding="utf-8")
        rotated = tmp_path / "bridge.2026-06-24_18-00-00.log"
        rotated.write_text(older + "\n", encoding="utf-8")
        records = read_records(active, rotated)
        assert [r.message for r in records] == ["older", "newer"]
