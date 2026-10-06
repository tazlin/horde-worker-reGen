"""The available-commit reading the RAM admission gates price a checkpoint mapping against."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

from horde_worker_regen.process_management.scheduling import host_commit

_MB = 1024 * 1024


def _fake_kernel32(*, available_page_file_bytes: int, succeeds: bool = True) -> SimpleNamespace:
    """A stand-in for ``windll`` whose ``GlobalMemoryStatusEx`` fills the structure it is handed."""

    def global_memory_status_ex(status_reference: object) -> int:
        status = status_reference._obj  # type: ignore[attr-defined]
        assert status.dwLength == ctypes.sizeof(status), "the caller sets dwLength before the call"
        status.ullAvailPageFile = available_page_file_bytes
        return 1 if succeeds else 0

    return SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=global_memory_status_ex))


class TestMeasureAvailableCommit:
    """Windows reports ullAvailPageFile in MB; elsewhere there is no figure."""

    def test_windows_reports_available_page_file_in_mb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The figure is GlobalMemoryStatusEx's available commit, converted to MB."""
        monkeypatch.setattr(host_commit.sys, "platform", "win32")
        monkeypatch.setattr(ctypes, "windll", _fake_kernel32(available_page_file_bytes=20480 * _MB), raising=False)

        assert host_commit.measure_available_commit_mb() == 20480.0

    def test_a_failed_windows_read_reports_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed call leaves the gates on physical RAM rather than pricing a zeroed structure."""
        monkeypatch.setattr(host_commit.sys, "platform", "win32")
        monkeypatch.setattr(
            ctypes,
            "windll",
            _fake_kernel32(available_page_file_bytes=20480 * _MB, succeeds=False),
            raising=False,
        )

        assert host_commit.measure_available_commit_mb() is None

    def test_posix_reports_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A POSIX host charges no commit for a file mapping, so there is no figure to price."""
        monkeypatch.setattr(host_commit.sys, "platform", "linux")

        assert host_commit.measure_available_commit_mb() is None
