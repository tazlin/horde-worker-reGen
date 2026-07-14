"""Tests for the live watch's incremental record reader and rotation-archive exclusion.

The attach supervisor watches a running worker's ``logs/`` every interval. To stay cheap on a long-running
worker it (1) never reads rotation ``.zip``/``.gz`` archives, and (2) re-parses only the bytes each active
log gained since the previous pass, while still handing detectors the same records a whole-file parse would.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from horde_worker_regen.analysis import log_ingest
from horde_worker_regen.analysis.bundle import LogBundle
from horde_worker_regen.analysis.log_ingest import IncrementalRecordReader

_STARTUP = "2026-06-24 18:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process"
_OOM = "2026-06-24 18:00:10.000 | ERROR | x:y:1 - CUDA out of memory. Tried to allocate 2.00 GiB"


def _spy_reads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, int]]:
    """Record every ``(path, start_offset)`` the reader actually reopens the file with."""
    calls: list[tuple[Path, int]] = []
    real = log_ingest._read_region

    def _spy(path: Path, start_offset: int) -> tuple[list[str], int]:
        calls.append((path, start_offset))
        return real(path, start_offset)

    monkeypatch.setattr(log_ingest, "_read_region", _spy)
    return calls


def test_unchanged_file_is_not_reopened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second pass over a file whose size did not change reuses the cache without reopening it."""
    calls = _spy_reads(monkeypatch)
    log = tmp_path / "bridge.log"
    log.write_text(_STARTUP + "\n" + _OOM + "\n", encoding="utf-8")
    reader = IncrementalRecordReader({})

    first = reader(log)
    assert len(first) == 2
    assert len(calls) == 1

    second = reader(log)
    assert len(second) == 2
    assert len(calls) == 1  # unchanged size: the file was never reopened


def test_only_appended_bytes_are_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An appended file is read from the previous consumed offset, yet the full record set is returned."""
    calls = _spy_reads(monkeypatch)
    log = tmp_path / "bridge.log"
    log.write_text(_STARTUP + "\n", encoding="utf-8")
    first_size = log.stat().st_size
    reader = IncrementalRecordReader({})

    assert len(reader(log)) == 1
    assert calls[-1] == (log, 0)

    with log.open("a", encoding="utf-8") as handle:
        handle.write(_OOM + "\n")
    records = reader(log)

    assert len(records) == 2  # detectors see every record
    assert calls[-1] == (log, first_size)  # but only the appended region was read


def test_traceback_appended_after_a_record_folds_into_it(tmp_path: Path) -> None:
    """Continuation lines appended after their record head fold into it exactly as a whole-file parse would."""
    log = tmp_path / "bridge.log"
    head = "2026-06-24 18:00:10.000 | ERROR | x:y:1 - pipeline crashed"
    log.write_text(head + "\n", encoding="utf-8")
    reader = IncrementalRecordReader({})

    first = reader(log)
    assert len(first) == 1
    assert first[0].continuation == []

    with log.open("a", encoding="utf-8") as handle:
        handle.write("Traceback (most recent call last):\n  File script.py, line 1\nValueError: bad\n")
    records = reader(log)

    assert len(records) == 1  # the appended lines did not open a new record
    assert records[0].continuation == [
        "Traceback (most recent call last):",
        "  File script.py, line 1",
        "ValueError: bad",
    ]
    assert "ValueError: bad" in records[0].full_text


def test_shrunk_file_is_reparsed_from_zero(tmp_path: Path) -> None:
    """A file that shrank (a rotation swapped a fresh, shorter active log in) is parsed from the start again."""
    log = tmp_path / "bridge.log"
    log.write_text(_STARTUP + "\n" + _OOM + "\n", encoding="utf-8")
    reader = IncrementalRecordReader({})
    assert len(reader(log)) == 2

    log.write_text(_STARTUP + "\n", encoding="utf-8")  # rotated: smaller than before
    assert len(reader(log)) == 1


def test_live_bundle_reads_only_active_logs_never_rotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live bundle classifies only the active ``bridge.log``, skipping both zip and uncompressed rotations."""
    (tmp_path / "bridge.log").write_text(_STARTUP + "\n" + _OOM + "\n", encoding="utf-8")
    rotated_zip = tmp_path / "bridge.2026-06-22_00-55-59_013989.log.zip"
    with zipfile.ZipFile(rotated_zip, "w") as archive:
        archive.writestr("bridge.log", _STARTUP + "\n" + _OOM + "\n")
    rotated_log = tmp_path / "bridge.2026-06-23_00-55-59_013989.log"
    rotated_log.write_text(_STARTUP + "\n" + _OOM + "\n", encoding="utf-8")

    def _forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the live watch opened a rotation zip archive")

    monkeypatch.setattr(log_ingest.zipfile, "ZipFile", _forbidden)

    reader = IncrementalRecordReader({})
    bundle = LogBundle.from_path(tmp_path, active_only=True, record_reader=reader)

    # Only the active file is classified: the rotation history (zip and uncompressed alike) is dropped.
    assert bundle.orchestrator_paths == [tmp_path / "bridge.log"]
    assert rotated_zip not in bundle.orchestrator_paths
    assert rotated_log not in bundle.orchestrator_paths
    assert len(bundle.orchestrator_records()) == 2  # read without touching any rotation
