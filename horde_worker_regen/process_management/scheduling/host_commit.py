"""Host commit available for checkpoint mappings, checked alongside physical RAM.

On Windows a child maps a checkpoint as a copy-on-write file view, and the view charges its whole file size to
system commit when it is mapped, for as long as the mapping lives. The commit charge does not consume physical RAM, so
psutil's ``available`` does not reflect it, and at the commit limit the next mapping fails inside the child. Admission
therefore prices a load that maps a checkpoint against :func:`measure_available_commit_mb` as well as against
available RAM.

The commit limit is physical RAM plus the page files at their current size, and Windows extends a page file when
a charge reaches the limit, within the same call that makes the charge. The reading adds that growth room to the
live figure (:func:`page_file_growth_room_mb`) to include commit that Windows can make available on demand.
Growth is limited by the configured maximum, if any, and the drive's free space. Windows does not apply the
worker's low-disk floor when extending a page file.
"""

from __future__ import annotations

import ctypes
import mmap
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from loguru import logger

__all__ = [
    "PageFileSetting",
    "PageFileUsage",
    "measure_available_commit_mb",
    "page_file_drive_roots",
    "page_file_growth_room_mb",
    "parse_paging_files",
]

_BYTES_PER_MB = 1024 * 1024

_MEMORY_MANAGEMENT_KEY = r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
_PAGING_FILES_VALUE = "PagingFiles"

_SYSTEM_PAGE_FILE_INFORMATION_CLASS = 18
_STATUS_INFO_LENGTH_MISMATCH = 0xC0000004
_PAGE_FILE_QUERY_INITIAL_BYTES = 4096
_PAGE_FILE_QUERY_MAX_BYTES = 64 * 1024
_NT_PATH_PREFIX = "\\??\\"
_ANY_DRIVE_PREFIX = "?:"
"""The drive a ``PagingFiles`` entry names when Windows places the file itself."""

_commit_read_failure_logged = False
_growth_room_failure_logged = False


@dataclass(frozen=True)
class PageFileSetting:
    """A page file configured in the ``PagingFiles`` registry value."""

    path: str
    """The file's path, lower-cased. A ``?:`` drive stands for whichever drive Windows places the file on."""
    maximum_mb: float
    """The configured maximum size in MB, or 0 for a file Windows sizes itself."""

    @property
    def system_managed(self) -> bool:
        """Whether Windows sizes the file itself, bounded only by its drive's free space."""
        return self.maximum_mb <= 0.0


@dataclass(frozen=True)
class PageFileUsage:
    """A live page file and its current size."""

    path: str
    """The file's drive-qualified path, lower-cased."""
    current_mb: float
    """The file's current size in MB."""


@dataclass
class _SettingsCache:
    """Holds the ``PagingFiles`` settings, which change only at reboot, so the registry is read once per process."""

    attempted: bool = False
    settings: tuple[PageFileSetting, ...] | None = None


_SETTINGS_CACHE = _SettingsCache()


class _UnicodeString(ctypes.Structure):
    """The NT ``UNICODE_STRING`` structure."""

    _fields_ = (
        ("Length", ctypes.c_ushort),
        ("MaximumLength", ctypes.c_ushort),
        ("Buffer", ctypes.c_void_p),
    )


class _SystemPageFileInformation(ctypes.Structure):
    """A ``SYSTEM_PAGEFILE_INFORMATION`` entry, with sizes in pages."""

    _fields_ = (
        ("NextEntryOffset", ctypes.c_ulong),
        ("TotalSize", ctypes.c_ulong),
        ("TotalInUse", ctypes.c_ulong),
        ("PeakUsage", ctypes.c_ulong),
        ("PageFileName", _UnicodeString),
    )


class _MemoryStatusEx(ctypes.Structure):
    """Represents the Win32 ``MEMORYSTATUSEX`` structure ``GlobalMemoryStatusEx`` fills."""

    _fields_ = (
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    )


def parse_paging_files(entries: Sequence[str]) -> tuple[PageFileSetting, ...]:
    """Return the page-file settings a ``PagingFiles`` registry value lists.

    Each entry is a path followed by its initial and maximum size in MB. A path with no sizes, or a maximum of 0,
    is a file Windows sizes itself.

    Raises:
        ValueError: An entry's maximum size is not a number.
    """
    settings: list[PageFileSetting] = []
    for entry in entries:
        parts = entry.split()
        if not parts:
            continue
        maximum_mb = float(parts[2]) if len(parts) >= 3 else 0.0
        settings.append(PageFileSetting(path=parts[0].lower(), maximum_mb=maximum_mb))
    return tuple(settings)


def _drive_root(path: str) -> str:
    r"""Return the root of the drive a drive-qualified path is on, such as ``t:\``."""
    if len(path) < 3 or path[1] != ":" or path[2] != "\\":
        raise ValueError(f"page file path {path!r} is not drive-qualified")
    return path[:3]


def page_file_growth_room_mb(
    settings: Sequence[PageFileSetting],
    usages: Sequence[PageFileUsage],
    drive_free_mb: Callable[[str], float],
) -> float:
    r"""Return the total space (MB) available for Windows to extend live page files.

    A file with a configured maximum grows to that maximum, as far as its drive's free space allows. A
    system-managed file grows into its drive's free space. Windows extends it on demand without regard to
    the worker's low-disk floor, so a floor here would only understate the room while the disk monitor and
    the download guard handle disk safety. A live file is counted as unable to grow unless the settings
    include it directly or through a ``?:`` entry.

    Args:
        settings: The configured page files (:func:`parse_paging_files`).
        usages: The live page files and their current sizes.
        drive_free_mb: Returns the free space (MB) of a drive root such as ``t:\``.

    Raises:
        ValueError: A live page file's path is not drive-qualified.
    """
    by_path = {setting.path: setting for setting in settings}
    placed_by_windows = next((setting for setting in settings if setting.path.startswith(_ANY_DRIVE_PREFIX)), None)
    growth_room_mb = 0.0
    for usage in usages:
        setting = by_path.get(usage.path, placed_by_windows)
        if setting is None:
            continue
        free_mb = drive_free_mb(_drive_root(usage.path))
        file_room_mb = free_mb if setting.system_managed else min(setting.maximum_mb - usage.current_mb, free_mb)
        growth_room_mb += max(0.0, file_room_mb)
    return growth_room_mb


def _read_paging_file_settings() -> tuple[PageFileSetting, ...]:
    """Read the ``PagingFiles`` registry value.

    Raises:
        OSError: The key or value cannot be read, or the host has no registry.
        ValueError: The value is not a string list, or an entry's maximum is not a number.
    """
    if sys.platform != "win32":
        raise OSError("the PagingFiles registry value exists only on Windows")
    import winreg

    # The ``winreg`` stub exists only for a Windows platform, so a type check configured for another host does
    # not see its attributes; the platform check above guards the read.
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _MEMORY_MANAGEMENT_KEY) as key:  # pyrefly: ignore[missing-attribute]
        entries, value_type = winreg.QueryValueEx(key, _PAGING_FILES_VALUE)  # pyrefly: ignore[missing-attribute]
    if value_type != winreg.REG_MULTI_SZ or not isinstance(entries, list):  # pyrefly: ignore[missing-attribute]
        raise ValueError(f"{_PAGING_FILES_VALUE} is registry type {value_type}, not a string list")
    return parse_paging_files([str(entry) for entry in entries])


def _read_page_file_usages() -> tuple[PageFileUsage, ...]:
    """Read each live page file's current size with one ``NtQuerySystemInformation`` call.

    Raises:
        OSError: The query fails, or its result does not fit the largest buffer offered.
    """
    # ``windll`` exists only on Windows, and the caller's platform check guards this read.
    ntdll = ctypes.windll.ntdll  # pyrefly: ignore[missing-attribute]
    buffer_size = _PAGE_FILE_QUERY_INITIAL_BYTES
    while True:
        buffer = ctypes.create_string_buffer(buffer_size)
        returned_bytes = ctypes.c_ulong(0)
        status = ntdll.NtQuerySystemInformation(
            _SYSTEM_PAGE_FILE_INFORMATION_CLASS,
            buffer,
            buffer_size,
            ctypes.byref(returned_bytes),
        )
        status &= 0xFFFFFFFF
        if status == _STATUS_INFO_LENGTH_MISMATCH and buffer_size < _PAGE_FILE_QUERY_MAX_BYTES:
            buffer_size *= 2
            continue
        if status != 0:
            raise OSError(f"NtQuerySystemInformation(SystemPageFileInformation) returned status {status:#010x}")
        break
    if returned_bytes.value == 0:
        return ()
    usages: list[PageFileUsage] = []
    offset = 0
    while True:
        entry = _SystemPageFileInformation.from_buffer(buffer, offset)
        name = ctypes.wstring_at(entry.PageFileName.Buffer, entry.PageFileName.Length // 2)
        usages.append(
            PageFileUsage(
                path=name.removeprefix(_NT_PATH_PREFIX).lower(),
                current_mb=entry.TotalSize * mmap.PAGESIZE / _BYTES_PER_MB,
            ),
        )
        if entry.NextEntryOffset == 0:
            return tuple(usages)
        offset += entry.NextEntryOffset


def _drive_free_mb(drive_root: str) -> float:
    """Return a drive's free space in MB."""
    return shutil.disk_usage(drive_root).free / _BYTES_PER_MB


def _log_growth_room_failure_once(error: BaseException) -> None:
    """Log the first growth-query failure. Failed queries use the current available commit without growth."""
    global _growth_room_failure_logged
    if _growth_room_failure_logged:
        return
    _growth_room_failure_logged = True
    logger.debug(
        f"Page-file growth room unreadable ({type(error).__name__}: {error}). Available commit is the current "
        "limit less the commit charged.",
    )


def _cached_paging_file_settings() -> tuple[PageFileSetting, ...] | None:
    """Return the page-file settings read at the first call, or None when that read failed."""
    if not _SETTINGS_CACHE.attempted:
        _SETTINGS_CACHE.attempted = True
        try:
            _SETTINGS_CACHE.settings = _read_paging_file_settings()
        except (OSError, ValueError) as error:
            _log_growth_room_failure_once(error)
    return _SETTINGS_CACHE.settings


def _page_file_growth_room_or_zero() -> float:
    """Return the live page files' growth room (MB), or 0 when any part of it cannot be read."""
    settings = _cached_paging_file_settings()
    if settings is None:
        return 0.0
    try:
        return page_file_growth_room_mb(settings, _read_page_file_usages(), _drive_free_mb)
    except Exception as error:  # noqa: BLE001 - a failed growth query leaves the current available commit usable.
        _log_growth_room_failure_once(error)
        return 0.0


def page_file_drive_roots() -> frozenset[str]:
    r"""Return the lower-cased drive roots (such as ``t:\``) of the live page files, empty off Windows.

    A page file that shares a drive with the model cache grows into the same free space the cache uses, so
    callers can report the shared drive. Return an empty tuple on failure.
    """
    if sys.platform != "win32":
        return frozenset()
    try:
        return frozenset(_drive_root(usage.path).lower() for usage in _read_page_file_usages())
    except (OSError, ValueError):
        return frozenset()


def measure_available_commit_mb() -> float | None:
    """Return the host's available commit (MB), or None where a checkpoint mapping is not charged to commit.

    On Windows this is ``GlobalMemoryStatusEx``'s ``ullAvailPageFile``, the commit limit at the page files' current
    size less the commit already charged, plus the growth room the page files still have
    (:func:`page_file_growth_room_mb`). The settings are read once per process. The current sizes and drive free
    space are read on each call, one system query and one free-space query per page file. When the growth room
    cannot be read, the reading is ``ullAvailPageFile`` alone. On POSIX hosts a private file mapping is not charged
    against a commit limit when it is mapped, so there is no figure and None leaves the RAM gates pricing physical
    RAM alone. A failed Windows read also returns None, and the first failure is logged.
    """
    global _commit_read_failure_logged
    if sys.platform != "win32":
        return None
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    # ``windll`` exists only on Windows, and the platform check above guards this read.
    kernel32 = ctypes.windll.kernel32  # pyrefly: ignore[missing-attribute]
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        if not _commit_read_failure_logged:
            _commit_read_failure_logged = True
            logger.warning(
                "GlobalMemoryStatusEx failed. RAM admission prices physical RAM only until a commit reading succeeds.",
            )
        return None
    return status.ullAvailPageFile / _BYTES_PER_MB + _page_file_growth_room_or_zero()
