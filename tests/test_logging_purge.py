"""Tests for the startup log-directory purge (age-out plus total-size cap)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from horde_worker_regen.logging_purge import (
    _BYTES_PER_GB,
    _SECONDS_PER_DAY,
    LogPurgeResult,
    purge_log_directory,
    purge_worker_logs_safely,
)


def _write(path: Path, size_bytes: int, *, age_days: float) -> Path:
    """Create a file of *size_bytes* whose mtime is *age_days* in the past."""
    path.write_bytes(b"\0" * size_bytes)
    mtime = time.time() - age_days * _SECONDS_PER_DAY
    os.utime(path, (mtime, mtime))
    return path


def test_missing_directory_is_a_noop(tmp_path: Path) -> None:
    """A directory that does not exist yields an empty result and no error."""
    result = purge_log_directory(tmp_path / "does_not_exist", max_age_days=30, max_total_gb=5)
    assert result == LogPurgeResult()


def test_age_out_removes_only_old_files(tmp_path: Path) -> None:
    """Files past the age limit are deleted; recent files are kept."""
    old = _write(tmp_path / "bridge.2020-01-01_00-00-00.log.zip", 100, age_days=45)
    recent = _write(tmp_path / "bridge.log", 100, age_days=1)

    result = purge_log_directory(tmp_path, max_age_days=30, max_total_gb=0)

    assert not old.exists()
    assert recent.exists()
    assert result.aged_out == 1
    assert result.deleted_files == 1
    assert result.deleted_bytes == 100


def test_age_out_boundary_keeps_files_newer_than_cutoff(tmp_path: Path) -> None:
    """A file a touch younger than the age limit survives; one a touch older is purged."""
    just_under = _write(tmp_path / "bridge_1.log", 10, age_days=29.9)
    just_over = _write(tmp_path / "bridge_2.log", 10, age_days=30.1)

    purge_log_directory(tmp_path, max_age_days=30, max_total_gb=0)

    assert just_under.exists()
    assert not just_over.exists()


def test_size_cap_trims_oldest_first_until_under_budget(tmp_path: Path) -> None:
    """Over-budget directories shed their oldest files until they fit."""
    half_gb = _BYTES_PER_GB // 2
    oldest = _write(tmp_path / "bridge.2024-01-01_00-00-00.log.zip", half_gb, age_days=10)
    middle = _write(tmp_path / "bridge.2024-06-01_00-00-00.log.zip", half_gb, age_days=5)
    newest = _write(tmp_path / "bridge.log", half_gb, age_days=1)

    # Budget of 1 GB against 1.5 GB present forces exactly the oldest file out.
    result = purge_log_directory(tmp_path, max_age_days=0, max_total_gb=1.0)

    assert not oldest.exists()
    assert middle.exists()
    assert newest.exists()
    assert result.size_trimmed == 1
    assert result.aged_out == 0
    assert result.remaining_bytes == 2 * half_gb


def test_zero_limits_disable_both_stages(tmp_path: Path) -> None:
    """Zero limits keep even an ancient, oversized file."""
    ancient = _write(tmp_path / "bridge.log", _BYTES_PER_GB, age_days=9999)

    result = purge_log_directory(tmp_path, max_age_days=0, max_total_gb=0)

    assert ancient.exists()
    assert result.deleted_files == 0


def test_unrecognized_files_are_never_deleted(tmp_path: Path) -> None:
    """Fail-closed: anything the registry does not recognize as a worker log is left alone.

    This covers both plainly-unrelated files (the action ledger, a note) and a stray ``.log`` whose name
    matches no declared family, even when it is old and the directory is over budget.
    """
    ledger = _write(tmp_path / "action_ledger.jsonl", _BYTES_PER_GB, age_days=365)
    readme = _write(tmp_path / "notes.txt", 10, age_days=365)
    stray_log = _write(tmp_path / "some_other_tool.log", 10, age_days=365)

    purge_log_directory(tmp_path, max_age_days=1, max_total_gb=0.001)

    assert ledger.exists()
    assert readme.exists()
    assert stray_log.exists()


def test_age_out_and_size_cap_compose(tmp_path: Path) -> None:
    """Age-out runs first; the size cap then trims what remains, counted separately."""
    third_gb = _BYTES_PER_GB // 3
    _write(tmp_path / "bridge_0.log", third_gb, age_days=100)  # aged out
    survivor_old = _write(tmp_path / "bridge.2024-01-01_00-00-00.log.zip", third_gb, age_days=10)
    survivor_new = _write(tmp_path / "bridge.log", third_gb, age_days=1)

    # After age-out, ~0.66 GB remains in two files; a 0.4 GB cap trims the older survivor.
    result = purge_log_directory(tmp_path, max_age_days=30, max_total_gb=third_gb / _BYTES_PER_GB + 0.01)

    assert result.aged_out == 1
    assert result.size_trimmed == 1
    assert not survivor_old.exists()
    assert survivor_new.exists()


def test_subdirectories_are_never_descended_or_deleted(tmp_path: Path) -> None:
    """Nested folders (e.g. a remote-support tree) and their contents are never touched."""
    nested = tmp_path / "remote_support" / "session"
    nested.mkdir(parents=True)
    buried = _write(nested / "old.log", 10, age_days=9999)
    top_level = _write(tmp_path / "bridge.log", 10, age_days=9999)

    result = purge_log_directory(tmp_path, max_age_days=1, max_total_gb=0.000001)

    # The buried file and its directory survive; only the top-level file is eligible.
    assert buried.exists()
    assert nested.is_dir()
    assert not top_level.exists()
    assert result.deleted_files == 1


def test_symlinks_are_skipped_and_never_followed(tmp_path: Path) -> None:
    """A symlink at the top level is judged as itself: neither it nor its target is deleted."""
    external = tmp_path / "external"
    external.mkdir()
    target = _write(external / "precious.log", 10, age_days=9999)

    link = tmp_path / "logs"
    link.mkdir()
    link_path = link / "shortcut.log"
    try:
        link_path.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/account")

    result = purge_log_directory(link, max_age_days=1, max_total_gb=0)

    assert link_path.is_symlink()  # the symlink itself was not removed
    assert target.exists()  # and the target outside logs/ was never reached
    assert result.deleted_files == 0


def test_safe_wrapper_swallows_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The startup wrapper must never raise, whatever the sweep does."""

    def _boom(*_args: object, **_kwargs: object) -> LogPurgeResult:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("horde_worker_regen.logging_purge.purge_log_directory", _boom)

    # Must not raise.
    purge_worker_logs_safely(max_age_days=30, max_total_gb=5, log_dir=tmp_path)
