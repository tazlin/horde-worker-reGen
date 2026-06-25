"""Bind a spawned worker (and its whole tree) to the host's lifetime on Windows via a Job Object.

A worker the host spawns outlives the host when the host dies the hard way (a closed launcher window, a
taskkill, a crash that skips teardown): on Windows a child's lifetime is not tied to its parent's, so the
worker and its own inference/safety children keep a GPU resident with nothing left to stop them. A Job
Object created with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` ties the tree to a handle the host process
holds; when the host ends for any reason the OS closes that handle and terminates every process in the
job. Children a job member spawns join the job automatically, so binding the worker covers its grandchildren.

Best-effort and Windows-only: off Windows, or if any Win32 call fails, every method is an inert no-op and
the `OwnedProcessRegistry` startup sweep remains the backstop.
"""

from __future__ import annotations

import contextlib
import sys

from loguru import logger

_WINDOWS = sys.platform == "win32"

if _WINDOWS:
    import ctypes
    from ctypes import wintypes

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectExtendedLimitInformation = 9
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = (
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        )

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = (
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        )

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = (
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        )

    def _kernel32() -> ctypes.WinDLL:
        """The kernel32 DLL with last-error tracking enabled for the handful of calls used here."""
        dll = ctypes.WinDLL("kernel32", use_last_error=True)
        # HANDLE returns must be declared, or ctypes' default c_int truncates them on 64-bit Windows.
        dll.CreateJobObjectW.restype = wintypes.HANDLE
        dll.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        dll.SetInformationJobObject.restype = wintypes.BOOL
        dll.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
        dll.OpenProcess.restype = wintypes.HANDLE
        dll.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        dll.AssignProcessToJobObject.restype = wintypes.BOOL
        dll.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        dll.CloseHandle.restype = wintypes.BOOL
        dll.CloseHandle.argtypes = (wintypes.HANDLE,)
        return dll


class WorkerJobObject:
    """A kill-on-close Job Object the host holds; assigned worker processes die when the host does.

    Inert off Windows or if the job could not be created, so callers need no platform guard of their own.
    """

    def __init__(self) -> None:
        """Create the kill-on-close job; leaves the object inert (a no-op) if that is not possible here."""
        self._handle: int | None = None
        if not _WINDOWS:
            return
        with contextlib.suppress(Exception):
            self._handle = self._create_kill_on_close_job()
        if self._handle is None:
            logger.debug("No worker Job Object; relying on the owned-pid registry to reap orphans instead.")

    @property
    def active(self) -> bool:
        """Whether a usable job exists (Windows, and creation succeeded)."""
        return self._handle is not None

    def _create_kill_on_close_job(self) -> int | None:
        """Create an unnamed job set to terminate all its members when its last handle closes."""
        dll = _kernel32()
        handle = dll.CreateJobObjectW(None, None)
        if not handle:
            return None
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not dll.SetInformationJobObject(
            handle,
            _JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            dll.CloseHandle(handle)
            return None
        return handle

    def assign(self, pid: int | None) -> bool:
        """Add the process ``pid`` (and thus the children it later spawns) to the job; False on any failure."""
        if self._handle is None or pid is None:
            return False
        dll = _kernel32()
        process_handle = dll.OpenProcess(_PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, pid)
        if not process_handle:
            logger.debug(f"Could not open worker pid {pid} to bind it to the Job Object.")
            return False
        try:
            if dll.AssignProcessToJobObject(self._handle, process_handle):
                return True
            logger.debug(f"Could not assign worker pid {pid} to the Job Object.")
            return False
        finally:
            dll.CloseHandle(process_handle)
