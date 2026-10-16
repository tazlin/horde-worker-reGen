"""Measure process RAM without treating reclaimable mapped checkpoint pages as allocator growth.

`private_ram_usage_bytes` reads the kernel's private working-set counter (Linux ``RssAnon + RssShmem``, Windows
``PrivateWorkingSetSize``) instead of walking the process's pages, because the walk's cost scales with the working
set and on Windows holds the GIL for its whole duration. The walk remains the fallback. `PrivateRamPeakSampler`
polls the reading during a model load.
"""

import ctypes
import functools
import os
import sys
import threading
from pathlib import Path
from types import TracebackType

import psutil
from loguru import logger

_unavailable_readings: set[tuple[str, type[Exception]]] = set()

_PROC_ROOT = Path("/proc")

_LINUX_STATUS_PRIVATE_FIELDS = ("RssAnon:", "RssShmem:")


def _log_unavailable_once(reading: str, error: Exception) -> None:
    """Keep an unsupported per-second reading from flooding a child's log."""
    key = (reading, type(error))
    if key in _unavailable_readings:
        logger.trace(f"{reading} RAM reading still unavailable: {type(error).__name__}: {error}")
        return
    _unavailable_readings.add(key)
    logger.debug(f"{reading} RAM reading unavailable: {type(error).__name__}: {error}")


def linux_private_ram_bytes(smaps_rollup: str) -> int | None:
    """Return anonymous or private-dirty resident pages from Linux's aggregate mapping report.

    Unique-set size alone includes exclusively mapped clean checkpoint pages. Those pages are
    reclaimable file cache and cannot justify cycling a healthy fp8 process or pricing another context.
    Anonymous resident pages and private dirty pages overlap, so take their maximum rather than sum.
    """
    readings: dict[str, int] = {}
    for line in smaps_rollup.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in {"Anonymous:", "Private_Dirty:"}:
            readings[fields[0]] = int(fields[1]) * 1024
    return max(readings.values()) if readings else None


def _linux_status_private_ram_bytes(status: str) -> int:
    """Return ``RssAnon + RssShmem`` from a ``/proc/<pid>/status`` report.

    Shmem is counted beside anonymous pages because pinned host memory can be shmem, which the walk's
    ``Private_Dirty`` also counts.

    Raises:
        ValueError: A field is absent or not an integer.
    """
    readings: dict[str, int] = {}
    for line in status.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in _LINUX_STATUS_PRIVATE_FIELDS:
            readings[fields[0]] = int(fields[1]) * 1024
    if len(readings) != len(_LINUX_STATUS_PRIVATE_FIELDS):
        raise ValueError(f"{' or '.join(_LINUX_STATUS_PRIVATE_FIELDS)} absent from the status report")
    return sum(readings.values())


class _ProcessMemoryCountersEx2(ctypes.Structure):
    """Represents the Win32 ``PROCESS_MEMORY_COUNTERS_EX2`` structure ``GetProcessMemoryInfo`` fills."""

    _fields_ = (
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
        ("PrivateWorkingSetSize", ctypes.c_size_t),
        ("SharedCommitUsage", ctypes.c_ulonglong),
    )


@functools.cache
def _windows_memory_dlls() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    """Return kernel32 and psapi with the prototypes of the two calls used here declared."""
    # ``WinDLL`` exists only on Windows; the only caller is behind a platform check.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # pyrefly: ignore[missing-attribute]
    psapi = ctypes.WinDLL("psapi", use_last_error=True)  # pyrefly: ignore[missing-attribute]
    # HANDLE returns must be declared, or ctypes' default c_int truncates them on 64-bit Windows.
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentProcess.argtypes = ()
    psapi.GetProcessMemoryInfo.restype = ctypes.c_long
    psapi.GetProcessMemoryInfo.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCountersEx2),
        ctypes.c_ulong,
    )
    return kernel32, psapi


def _windows_private_working_set_bytes() -> int:
    """Return the current process's ``PrivateWorkingSetSize``.

    Raises:
        OSError: The call failed, or the field reads zero on a build that does not fill it.
    """
    kernel32, psapi = _windows_memory_dlls()
    counters = _ProcessMemoryCountersEx2()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        last_error = ctypes.get_last_error()  # pyrefly: ignore[missing-attribute]
        raise OSError(last_error, "GetProcessMemoryInfo failed")
    if counters.PrivateWorkingSetSize == 0:
        raise OSError("PrivateWorkingSetSize reads zero")
    return counters.PrivateWorkingSetSize


PEAK_SAMPLE_INTERVAL_SECONDS = 0.1
"""How often the load-peak sampler reads private RAM; a load's transient lasts seconds, so 100 ms resolves it.

The counter read takes a small fraction of the interval, so polling does not slow the load it measures."""


class PrivateRamPeakSampler:
    """Track the highest private RAM reading of this process while a block runs.

    Used around a model load: the settled reading after the load can sit below the transient the load
    touched, and a staging price learned from the settled figure alone would under-reserve the next load.
    """

    def __init__(self, interval_seconds: float = PEAK_SAMPLE_INTERVAL_SECONDS) -> None:
        """Prepare a sampler for the current process."""
        self._process = psutil.Process()
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes: int | None = None

    def _sample(self) -> None:
        reading = private_ram_usage_bytes(self._process)
        if reading is not None and (self.peak_bytes is None or reading > self.peak_bytes):
            self.peak_bytes = reading

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def __enter__(self) -> "PrivateRamPeakSampler":
        """Take a first reading and start polling on a daemon thread."""
        self._sample()
        self._thread = threading.Thread(target=self._run, name="private-ram-peak", daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop polling and take a last reading."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()


def private_ram_usage_bytes(process: psutil.Process) -> int | None:
    """Return the private working set in bytes, excluding reclaimable mapped checkpoint pages.

    Linux reads ``RssAnon + RssShmem`` from ``/proc/<pid>/status`` and falls back to the ``smaps_rollup`` walk.
    Windows reads ``PrivateWorkingSetSize`` and falls back to the USS walk. Other platforms, and a Linux host
    where both readings fail, use USS.

    Args:
        process: The process to read. The Windows counter is read only when this is the current process, which is
            what every caller passes; any other process takes the walk.
    """
    if sys.platform == "linux":
        process_directory = _PROC_ROOT / str(process.pid)
        try:
            return _linux_status_private_ram_bytes((process_directory / "status").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            _log_unavailable_once("Linux status private", error)
        try:
            reading = linux_private_ram_bytes((process_directory / "smaps_rollup").read_text(encoding="utf-8"))
            if reading is not None:
                return reading
        except (OSError, ValueError) as error:
            _log_unavailable_once("Linux private", error)
    elif sys.platform == "win32" and process.pid == os.getpid():
        try:
            return _windows_private_working_set_bytes()
        except OSError as error:
            _log_unavailable_once("Windows private working-set", error)
    try:
        return process.memory_full_info().uss
    except (psutil.Error, AttributeError, NotImplementedError) as error:
        _log_unavailable_once("Unique", error)
        return None
