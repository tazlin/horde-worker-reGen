"""Tests for the shared guarded directory sweep (the delete guard behind log and stats retention)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from horde_worker_regen._guarded_purge import (
    _BYTES_PER_GB,
    _SECONDS_PER_DAY,
    PurgeResult,
    guarded_purge_directory,
)


def _recognizes_dat(filename: str) -> bool:
    """An arbitrary recognizer for the tests: only ``*.dat`` files are this sweep's own."""
    return filename.endswith(".dat")


def _write(path: Path, size_bytes: int, *, age_days: float) -> Path:
    path.write_bytes(b"\0" * size_bytes)
    mtime = time.time() - age_days * _SECONDS_PER_DAY
    os.utime(path, (mtime, mtime))
    return path


def _purge(directory: Path, *, max_age_days: float, max_total_gb: float) -> PurgeResult:
    return guarded_purge_directory(
        directory,
        recognizer=_recognizes_dat,
        max_age_days=max_age_days,
        max_total_gb=max_total_gb,
        label="Test purge",
    )


def test_missing_directory_is_a_noop(tmp_path: Path) -> None:
    """A missing directory is a no-op that removes nothing."""
    assert _purge(tmp_path / "nope", max_age_days=30, max_total_gb=5) == PurgeResult()


def test_age_out_removes_only_old_recognized_files(tmp_path: Path) -> None:
    """The age-out deletes only recognized files past the age limit, sparing fresh ones."""
    old = _write(tmp_path / "old.dat", 10, age_days=90)
    fresh = _write(tmp_path / "fresh.dat", 10, age_days=1)

    result = _purge(tmp_path, max_age_days=30, max_total_gb=0)

    assert not old.exists()
    assert fresh.exists()
    assert result.aged_out == 1
    assert result.deleted_files == 1


def test_unrecognized_files_are_never_deleted(tmp_path: Path) -> None:
    """Fail-closed: anything the recognizer does not accept is left alone, even when old and over budget."""
    foreign = _write(tmp_path / "notes.txt", _BYTES_PER_GB, age_days=365)
    other = _write(tmp_path / "keep.bin", 10, age_days=365)

    _purge(tmp_path, max_age_days=1, max_total_gb=0.001)

    assert foreign.exists()
    assert other.exists()


def test_age_out_then_size_cap_compose(tmp_path: Path) -> None:
    """Age-out runs first; the size cap then trims what remains, counted separately."""
    third_gb = _BYTES_PER_GB // 3
    _write(tmp_path / "a.dat", third_gb, age_days=100)  # aged out
    survivor_old = _write(tmp_path / "b.dat", third_gb, age_days=10)
    survivor_new = _write(tmp_path / "c.dat", third_gb, age_days=1)

    result = _purge(tmp_path, max_age_days=30, max_total_gb=third_gb / _BYTES_PER_GB + 0.01)

    assert result.aged_out == 1
    assert result.size_trimmed == 1
    assert not survivor_old.exists()
    assert survivor_new.exists()


def test_size_cap_trims_oldest_first(tmp_path: Path) -> None:
    """The size cap trims oldest-first so the newest file survives."""
    older = _write(tmp_path / "older.dat", _BYTES_PER_GB // 2, age_days=5)
    newer = _write(tmp_path / "newer.dat", _BYTES_PER_GB // 2, age_days=1)

    _purge(tmp_path, max_age_days=0, max_total_gb=(_BYTES_PER_GB // 2) / _BYTES_PER_GB + 0.01)

    assert not older.exists()
    assert newer.exists()


def test_zero_limits_disable_both_stages(tmp_path: Path) -> None:
    """Zero limits disable both stages, keeping everything."""
    kept = _write(tmp_path / "kept.dat", _BYTES_PER_GB, age_days=9999)

    result = _purge(tmp_path, max_age_days=0, max_total_gb=0)

    assert kept.exists()
    assert result.deleted_files == 0


def test_subdirectories_are_never_descended_or_deleted(tmp_path: Path) -> None:
    """Nested folders and their contents are never inspected or removed."""
    nested = tmp_path / "keep" / "inner"
    nested.mkdir(parents=True)
    buried = _write(nested / "buried.dat", 10, age_days=9999)
    top_level = _write(tmp_path / "top.dat", 10, age_days=9999)

    result = _purge(tmp_path, max_age_days=1, max_total_gb=0.000001)

    assert buried.exists()
    assert nested.is_dir()
    assert not top_level.exists()
    assert result.deleted_files == 1


def test_symlinks_are_skipped_and_never_followed(tmp_path: Path) -> None:
    """A top-level symlink is judged as itself: neither it nor its target is deleted."""
    external = tmp_path / "external"
    external.mkdir()
    target = _write(external / "precious.dat", 10, age_days=9999)

    swept = tmp_path / "swept"
    swept.mkdir()
    link_path = swept / "shortcut.dat"
    try:
        link_path.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/account")

    result = _purge(swept, max_age_days=1, max_total_gb=0)

    assert link_path.is_symlink()  # the symlink itself survived
    assert target.exists()  # and its target outside the swept dir was never reached
    assert result.deleted_files == 0
