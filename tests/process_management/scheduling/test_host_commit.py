"""The available-commit reading the RAM admission gates price a checkpoint mapping against."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from horde_worker_regen.process_management.scheduling import host_commit
from horde_worker_regen.process_management.scheduling.host_commit import (
    PageFileSetting,
    PageFileUsage,
    page_file_growth_room_mb,
    parse_paging_files,
)

_MB = 1024 * 1024


def _fake_kernel32(*, available_page_file_bytes: int, succeeds: bool = True) -> SimpleNamespace:
    """A stand-in for ``windll`` whose ``GlobalMemoryStatusEx`` fills the structure it is handed."""

    def global_memory_status_ex(status_reference: object) -> int:
        status = status_reference._obj  # type: ignore[attr-defined]
        assert status.dwLength == ctypes.sizeof(status), "the caller sets dwLength before the call"
        status.ullAvailPageFile = available_page_file_bytes
        return 1 if succeeds else 0

    return SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=global_memory_status_ex))


def _free_by_drive(free_mb: dict[str, float]) -> Callable[[str], float]:
    """A drive free-space reader over fixed figures, keyed by drive root."""
    return lambda drive_root: free_mb[drive_root]


@pytest.fixture(autouse=True)
def _host_without_page_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with an unread settings cache, no failure logged and no page files on the host.

    The readers are replaced so no test reads the registry or the page files of the machine running the suite.
    """
    monkeypatch.setattr(host_commit, "_SETTINGS_CACHE", host_commit._SettingsCache())
    monkeypatch.setattr(host_commit, "_growth_room_failure_logged", False)
    monkeypatch.setattr(host_commit, "_read_paging_file_settings", lambda: ())
    monkeypatch.setattr(host_commit, "_read_page_file_usages", lambda: ())
    monkeypatch.setattr(host_commit, "_drive_free_mb", lambda _drive_root: 0.0)


def _windows_host(monkeypatch: pytest.MonkeyPatch, *, available_mb: int) -> None:
    """Report a Windows host whose current-size available commit is ``available_mb``."""
    monkeypatch.setattr(host_commit.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", _fake_kernel32(available_page_file_bytes=available_mb * _MB), raising=False)


class TestMeasureAvailableCommit:
    """Windows reports ullAvailPageFile in MB plus the page files' growth room; elsewhere there is no figure."""

    def test_windows_reports_available_page_file_in_mb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no page file able to grow, the figure is GlobalMemoryStatusEx's available commit in MB."""
        _windows_host(monkeypatch, available_mb=20480)

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

    def test_growth_room_is_added_to_the_live_figure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fixed file below its maximum and a system-managed file with free space both add their room."""
        _windows_host(monkeypatch, available_mb=8823)
        monkeypatch.setattr(
            host_commit,
            "_read_paging_file_settings",
            lambda: parse_paging_files(["s:\\pagefile.sys 4096 8192", "t:\\pagefile.sys 0 0"]),
        )
        monkeypatch.setattr(
            host_commit,
            "_read_page_file_usages",
            lambda: (PageFileUsage("s:\\pagefile.sys", 4096.0), PageFileUsage("t:\\pagefile.sys", 6144.0)),
        )
        monkeypatch.setattr(host_commit, "_drive_free_mb", _free_by_drive({"s:\\": 100_000.0, "t:\\": 28_000.0}))

        assert host_commit.measure_available_commit_mb() == pytest.approx(8823.0 + 4096.0 + 28_000.0)

    def test_the_settings_are_read_once_and_the_sizes_on_every_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The registry changes only at reboot, while the page files grow and shrink between readings."""
        _windows_host(monkeypatch, available_mb=8000)
        settings_reads: list[int] = []
        current_sizes = iter([4096.0, 6000.0])

        def read_settings() -> tuple[PageFileSetting, ...]:
            settings_reads.append(1)
            return parse_paging_files(["s:\\pagefile.sys 4096 8192"])

        monkeypatch.setattr(host_commit, "_read_paging_file_settings", read_settings)
        monkeypatch.setattr(
            host_commit,
            "_read_page_file_usages",
            lambda: (PageFileUsage("s:\\pagefile.sys", next(current_sizes)),),
        )
        monkeypatch.setattr(host_commit, "_drive_free_mb", _free_by_drive({"s:\\": 100_000.0}))

        first = host_commit.measure_available_commit_mb()
        second = host_commit.measure_available_commit_mb()

        assert (first, second) == (8000.0 + 4096.0, 8000.0 + 2192.0)
        assert len(settings_reads) == 1

    def test_an_unreadable_registry_leaves_the_plain_figure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed settings read is not retried, and the reading is ullAvailPageFile alone."""
        _windows_host(monkeypatch, available_mb=8823)
        settings_reads: list[int] = []

        def unreadable() -> tuple[PageFileSetting, ...]:
            settings_reads.append(1)
            raise FileNotFoundError("PagingFiles")

        monkeypatch.setattr(host_commit, "_read_paging_file_settings", unreadable)
        monkeypatch.setattr(
            host_commit,
            "_read_page_file_usages",
            lambda: (PageFileUsage("t:\\pagefile.sys", 6144.0),),
        )
        monkeypatch.setattr(host_commit, "_drive_free_mb", _free_by_drive({"t:\\": 90_000.0}))

        assert host_commit.measure_available_commit_mb() == 8823.0
        assert host_commit.measure_available_commit_mb() == 8823.0
        assert len(settings_reads) == 1
        assert host_commit._growth_room_failure_logged

    def test_a_failed_size_query_leaves_the_plain_figure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed page-file query falls back to ullAvailPageFile alone."""
        _windows_host(monkeypatch, available_mb=8823)
        monkeypatch.setattr(
            host_commit,
            "_read_paging_file_settings",
            lambda: parse_paging_files(["t:\\pagefile.sys 0 0"]),
        )

        def failed_query() -> tuple[PageFileUsage, ...]:
            raise OSError("NtQuerySystemInformation(SystemPageFileInformation) returned status 0xc0000001")

        monkeypatch.setattr(host_commit, "_read_page_file_usages", failed_query)

        assert host_commit.measure_available_commit_mb() == 8823.0
        assert host_commit._growth_room_failure_logged


class TestPageFileGrowthRoom:
    """Total space available for page-file growth."""

    def test_a_fixed_file_grows_to_its_maximum(self) -> None:
        """A file with a configured maximum may grow from its current size to that maximum."""
        room = page_file_growth_room_mb(
            [PageFileSetting("s:\\pagefile.sys", 8192.0)],
            [PageFileUsage("s:\\pagefile.sys", 4096.0)],
            _free_by_drive({"s:\\": 100_000.0}),
        )

        assert room == 4096.0

    def test_a_fixed_file_at_its_maximum_has_no_room(self) -> None:
        """A file already at its maximum cannot grow, and a file reported past it does not count negative room."""
        room = page_file_growth_room_mb(
            [PageFileSetting("s:\\pagefile.sys", 8192.0)],
            [PageFileUsage("s:\\pagefile.sys", 8300.0)],
            _free_by_drive({"s:\\": 100_000.0}),
        )

        assert room == 0.0

    def test_a_fixed_file_is_bounded_by_its_drive(self) -> None:
        """A drive with less free space than the file's remaining maximum bounds the growth at that space."""
        room = page_file_growth_room_mb(
            [PageFileSetting("c:\\pagefile.sys", 16_384.0)],
            [PageFileUsage("c:\\pagefile.sys", 4096.0)],
            _free_by_drive({"c:\\": 3000.0}),
        )

        assert room == 3000.0

    def test_a_system_managed_file_grows_into_its_drive_free_space(self) -> None:
        """A system-managed page file may use all free space on the drive without a worker-imposed floor."""
        room = page_file_growth_room_mb(
            parse_paging_files(["t:\\pagefile.sys 0 0"]),
            [PageFileUsage("t:\\pagefile.sys", 6144.0)],
            _free_by_drive({"t:\\": 28_000.0}),
        )

        assert room == pytest.approx(28_000.0)

    def test_a_full_drive_gives_no_room(self) -> None:
        """A drive with no free space gives a system-managed file no room, never negative room."""
        room = page_file_growth_room_mb(
            parse_paging_files(["t:\\pagefile.sys 0 0"]),
            [PageFileUsage("t:\\pagefile.sys", 6144.0)],
            _free_by_drive({"t:\\": 0.0}),
        )

        assert room == 0.0

    def test_files_are_summed(self) -> None:
        """Each live file contributes its own room, judged against its own drive."""
        room = page_file_growth_room_mb(
            parse_paging_files(["s:\\pagefile.sys 4096 8192", "t:\\pagefile.sys 0 0"]),
            [PageFileUsage("s:\\pagefile.sys", 6000.0), PageFileUsage("t:\\pagefile.sys", 6144.0)],
            _free_by_drive({"s:\\": 100_000.0, "t:\\": 1000.0}),
        )

        assert room == pytest.approx(2192.0 + 1000.0)

    def test_a_windows_placed_entry_covers_the_file_windows_placed(self) -> None:
        """A ``?:`` entry is a system-managed file on whichever drive Windows chose."""
        room = page_file_growth_room_mb(
            parse_paging_files(["?:\\pagefile.sys"]),
            [PageFileUsage("c:\\pagefile.sys", 4096.0)],
            _free_by_drive({"c:\\": 500.0}),
        )

        assert room == pytest.approx(500.0)

    def test_a_live_file_the_settings_do_not_name_cannot_grow(self) -> None:
        """Growth is counted only for a file whose maximum is known."""
        room = page_file_growth_room_mb(
            parse_paging_files(["s:\\pagefile.sys 4096 8192"]),
            [PageFileUsage("d:\\pagefile.sys", 4096.0)],
            _free_by_drive({"d:\\": 500_000.0}),
        )

        assert room == 0.0


class TestPageFileDriveRoots:
    """The live page files' drives are exposed for host-condition notes without raising."""

    def test_roots_come_from_the_live_files(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each live file contributes its drive root, lower-cased."""
        monkeypatch.setattr(host_commit.sys, "platform", "win32")
        monkeypatch.setattr(
            host_commit,
            "_read_page_file_usages",
            lambda: (PageFileUsage("T:\\pagefile.sys", 6144.0), PageFileUsage("s:\\pagefile.sys", 4096.0)),
        )

        assert host_commit.page_file_drive_roots() == frozenset({"t:\\", "s:\\"})

    def test_a_failed_read_gives_no_roots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed query returns an empty page-file list."""
        monkeypatch.setattr(host_commit.sys, "platform", "win32")

        def fail() -> tuple[PageFileUsage, ...]:
            raise OSError("query failed")

        monkeypatch.setattr(host_commit, "_read_page_file_usages", fail)

        assert host_commit.page_file_drive_roots() == frozenset()


class TestParsePagingFiles:
    """The ``PagingFiles`` registry value: a path, then the initial and maximum size in MB."""

    def test_sizes_and_system_managed_entries(self) -> None:
        """A maximum of 0, or a path with no sizes, is a file Windows sizes itself."""
        settings = parse_paging_files(["S:\\pagefile.sys 4096 8192", "t:\\pagefile.sys 0 0", "?:\\pagefile.sys", ""])

        assert settings == (
            PageFileSetting("s:\\pagefile.sys", 8192.0),
            PageFileSetting("t:\\pagefile.sys", 0.0),
            PageFileSetting("?:\\pagefile.sys", 0.0),
        )
        assert [setting.system_managed for setting in settings] == [False, True, True]

    def test_a_malformed_maximum_raises(self) -> None:
        """A maximum that is not a number is a failed read, which the reading treats as no growth room."""
        with pytest.raises(ValueError):
            parse_paging_files(["s:\\pagefile.sys 4096 lots"])
