"""Tests for retained stats JSONL file operations."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import NamedTuple, Protocol

from horde_worker_regen.stats_operations import compress_old_stats_files, downsample_stats_files, main


class _CaptureResult(NamedTuple):
    """Subset of pytest's capture result used by these tests."""

    out: str
    err: str


class _CaptureFixture(Protocol):
    """Subset of pytest's capsys fixture used by these tests."""

    def readouterr(self) -> _CaptureResult:
        """Return captured stdout/stderr."""


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sample(timestamp: float) -> dict[str, object]:
    return {"event": "stats_sample", "sample": {"timestamp": timestamp, "jobs_submitted": int(timestamp)}}


def _job(job_id: str) -> dict[str, object]:
    return {"event": "job_completed", "job": {"job_id": job_id}}


def test_compress_old_stats_files_skips_latest_by_default(tmp_path: Path) -> None:
    """Compression leaves the newest uncompressed stats file alone for a running worker."""
    older = tmp_path / "stats-v1.0.0-20260101-000000-000.jsonl"
    latest = tmp_path / "stats-v1.0.0-20260101-000100-000.jsonl"
    _write_jsonl(older, [_sample(1.0)])
    _write_jsonl(latest, [_sample(2.0)])

    report = compress_old_stats_files(tmp_path)

    compressed = older.with_name(older.name + ".gz")
    assert report.files_changed == 1
    assert compressed.exists()
    assert not older.exists()
    assert latest.exists()
    assert _read_jsonl(compressed)[0]["event"] == "stats_sample"


def test_compress_can_include_latest(tmp_path: Path) -> None:
    """The caller can explicitly include the newest file when the worker is known stopped."""
    path = tmp_path / "stats-v1.0.0-20260101-000000-000.jsonl"
    _write_jsonl(path, [_sample(1.0)])

    report = compress_old_stats_files(tmp_path, include_latest=True)

    assert report.files_changed == 1
    assert path.with_name(path.name + ".gz").exists()


def test_downsample_preserves_jobs_and_skips_latest_by_default(tmp_path: Path) -> None:
    """Downsampling thins samples in older files, keeps job events, and leaves the newest file untouched."""
    older = tmp_path / "stats-v1.0.0-20260101-000000-000.jsonl"
    latest = tmp_path / "stats-v1.0.0-20260101-000100-000.jsonl"
    _write_jsonl(older, [_sample(0.0), _sample(1.0), _job("a"), _sample(5.0), _sample(6.0)])
    _write_jsonl(latest, [_sample(7.0), _sample(8.0)])

    report = downsample_stats_files(5.0, tmp_path)

    assert report.samples_dropped == 2
    assert _read_jsonl(older) == [_sample(0.0), _job("a"), _sample(5.0)]
    assert _read_jsonl(latest) == [_sample(7.0), _sample(8.0)]


def test_downsample_rewrites_compressed_files(tmp_path: Path) -> None:
    """Compressed stats files stay compressed after downsampling."""
    path = tmp_path / "stats-v1.0.0-20260101-000000-000.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for event in [_sample(0.0), _sample(1.0), _sample(10.0)]:
            handle.write(json.dumps(event) + "\n")

    report = downsample_stats_files(5.0, tmp_path, include_latest=True)

    assert report.files_changed == 1
    assert path.exists()
    assert _read_jsonl(path) == [_sample(0.0), _sample(10.0)]


def test_cli_uses_same_operations(tmp_path: Path, capsys: _CaptureFixture) -> None:
    """The CLI entry point reports the same operation summary."""
    path = tmp_path / "stats-v1.0.0-20260101-000000-000.jsonl"
    _write_jsonl(path, [_sample(1.0)])

    main(["--stats-dir", str(tmp_path), "compress", "--include-latest"])

    output = capsys.readouterr().out
    assert "Files changed: 1" in output
    assert path.with_name(path.name + ".gz").exists()
