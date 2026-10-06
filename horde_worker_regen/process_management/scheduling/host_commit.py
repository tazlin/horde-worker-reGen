"""The host's available commit, the figure a checkpoint mapping is charged against beside physical RAM.

On Windows a child maps a checkpoint as a copy-on-write file view, and the view charges its whole file size to
system commit when it is mapped, for as long as the mapping lives. The charge consumes no physical RAM, so
psutil's ``available`` cannot see it; at the commit limit the next mapping fails inside the child. Admission
therefore prices a load that maps a checkpoint against :func:`measure_available_commit_mb` as well as against
available RAM.
"""

from __future__ import annotations

import ctypes
import sys

from loguru import logger

_BYTES_PER_MB = 1024 * 1024

_commit_read_failure_logged = False


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


def measure_available_commit_mb() -> float | None:
    """Return the host's available commit (MB), or None where a checkpoint mapping is not charged to commit.

    On Windows this is ``GlobalMemoryStatusEx``'s ``ullAvailPageFile``: the commit limit (physical RAM plus the
    page files at their current size) less the commit already charged. On POSIX hosts a private file mapping is
    not charged against a commit limit when it is mapped, so there is no figure and None leaves the RAM gates
    pricing physical RAM alone. A failed Windows read also returns None; the first failure is logged.
    """
    global _commit_read_failure_logged
    if sys.platform != "win32":
        return None
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        if not _commit_read_failure_logged:
            _commit_read_failure_logged = True
            logger.warning(
                "GlobalMemoryStatusEx failed; RAM admission prices physical RAM only until a commit reading succeeds.",
            )
        return None
    return status.ullAvailPageFile / _BYTES_PER_MB
