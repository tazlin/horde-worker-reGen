"""Tests for the startup stats-directory purge (fail-closed age-out plus total-size cap)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from horde_worker_regen._guarded_purge import _BYTES_PER_GB, _SECONDS_PER_DAY
from horde_worker_regen.stats_purge import purge_stats_directory, purge_worker_stats_safely


def _write(path: Path, size_bytes: int, *, age_days: float) -> Path:
    path.write_bytes(b"\0" * size_bytes)
    mtime = time.time() - age_days * _SECONDS_PER_DAY
    os.utime(path, (mtime, mtime))
    return path


def test_missing_directory_is_a_noop(tmp_path: Path) -> None:
    """A missing stats directory is a no-op."""
    result = purge_stats_directory(tmp_path / "nope", max_age_days=30, max_total_gb=5)
    assert result.deleted_files == 0


def test_ages_out_recognized_stats_files(tmp_path: Path) -> None:
    """Recognized .jsonl and .jsonl.gz stats files past the age limit are removed; fresh ones stay."""
    old_jsonl = _write(tmp_path / "stats-v1.0.0-20240101-000000-000.jsonl", 10, age_days=90)
    old_gz = _write(tmp_path / "stats-v1.0.0-20240101-000000-001.jsonl.gz", 10, age_days=90)
    fresh = _write(tmp_path / "stats-v1.0.0-20260101-000000-000.jsonl", 10, age_days=1)

    result = purge_stats_directory(tmp_path, max_age_days=30, max_total_gb=0)

    assert not old_jsonl.exists()
    assert not old_gz.exists()
    assert fresh.exists()
    assert result.aged_out == 2


def test_foreign_files_and_subdirs_and_symlinks_are_never_touched(tmp_path: Path) -> None:
    """Fail-closed: only ``stats-v*.jsonl(.gz)`` are eligible; everything else in the dir survives."""
    foreign = _write(tmp_path / "notes.txt", _BYTES_PER_GB, age_days=365)
    leftover_tmp = _write(tmp_path / "stats-v1.0.0-20240101-000000-000.jsonl.gz.tmp", 10, age_days=365)
    wrong_prefix = _write(tmp_path / "metrics-20240101.jsonl", 10, age_days=365)

    subdir = tmp_path / "keep"
    subdir.mkdir()
    buried = _write(subdir / "stats-v1.0.0-20240101-000000-000.jsonl", 10, age_days=365)

    external = tmp_path / "external"
    external.mkdir()
    link_target = _write(external / "stats-v1.0.0-precious.jsonl", 10, age_days=365)
    link_path = tmp_path / "stats-v1.0.0-shortcut.jsonl"
    symlink_made = True
    try:
        link_path.symlink_to(link_target)
    except (OSError, NotImplementedError):
        symlink_made = False

    purge_stats_directory(tmp_path, max_age_days=1, max_total_gb=0.0000001)

    assert foreign.exists()
    assert leftover_tmp.exists()
    assert wrong_prefix.exists()
    assert buried.exists()
    assert subdir.is_dir()
    if symlink_made:
        assert link_path.is_symlink()
        assert link_target.exists()


def test_size_cap_trims_oldest_first(tmp_path: Path) -> None:
    """The size cap trims the oldest stats file first."""
    older = _write(tmp_path / "stats-v1.0.0-20250101-000000-000.jsonl", _BYTES_PER_GB // 2, age_days=5)
    newer = _write(tmp_path / "stats-v1.0.0-20260101-000000-000.jsonl", _BYTES_PER_GB // 2, age_days=1)

    purge_stats_directory(tmp_path, max_age_days=0, max_total_gb=(_BYTES_PER_GB // 2) / _BYTES_PER_GB + 0.01)

    assert not older.exists()
    assert newer.exists()


def test_safe_wrapper_swallows_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The startup wrapper must never raise, whatever the sweep does."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("horde_worker_regen.stats_purge.purge_stats_directory", _boom)

    purge_worker_stats_safely(max_age_days=30, max_total_gb=5, stats_dir=tmp_path)  # must not raise
