"""Per-process GPU shared-memory sampling: the direct WDDM demand-paging signal.

Under Windows' WDDM, over-subscribing dedicated VRAM does not fail allocations: the OS demotes them to
the system-memory-backed *shared* GPU segment and pages on demand. In that regime the usual telemetry
lies: ``mem_get_info`` keeps reporting generous free VRAM and core utilization reads high while the
device mostly waits on the bus, so throughput halves with no visible error. The OS, however, tracks the
demotion per process: the ``GPU Process Memory`` performance counters (the data behind Task Manager's
"Shared GPU memory" column) expose each process's local (dedicated) and shared (system-backed) GPU bytes
by PID. A worker child whose *shared* usage climbs while its job samples is being demand-paged, and the
PID attribution distinguishes the worker's own over-commit from external VRAM pressure (a game or
browser would show under its own PID instead).

The sampler reads those counters through PDH via ctypes on a background thread, mirroring
:class:`~horde_worker_regen.utils.gpu_monitor.GpuUtilizationSampler`'s degradation contract: it runs in
the torch-free orchestrator, assumes no CUDA (the counters are WDDM-level and vendor-neutral), and on
any non-Windows host, missing counter set, or PDH error it simply collects nothing and reports None,
never raising into the control loop.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger

_SHARED_USAGE_COUNTER_PATH = r"\GPU Process Memory(*)\Shared Usage"
_LOCAL_USAGE_COUNTER_PATH = r"\GPU Process Memory(*)\Local Usage"

_ADAPTER_DEDICATED_COUNTER_PATH = r"\GPU Adapter Memory(*)\Dedicated Usage"

_PDH_MORE_DATA = 0x800007D2
_PDH_FMT_LARGE = 0x00000400

_INSTANCE_PID_PATTERN = re.compile(r"pid_(\d+)")
_INSTANCE_LUID_PATTERN = re.compile(r"luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)")

_BYTES_PER_MB = 1024.0 * 1024.0


@dataclass(frozen=True)
class GpuProcessMemorySample:
    """One reading of per-process GPU memory usage, in MB, keyed by OS PID.

    A PID can own several counter instances (one per adapter LUID / physical segment); values are the
    per-PID sums so a reading answers "how much GPU memory does this process hold" directly.
    """

    timestamp: float
    shared_mb_by_pid: dict[int, float] = field(default_factory=dict)
    local_mb_by_pid: dict[int, float] = field(default_factory=dict)
    local_mb_by_pid_and_adapter: dict[tuple[int, str], float] = field(default_factory=dict)
    """Dedicated GPU memory per process and adapter LUID, which says which adapter a process is on."""
    adapter_dedicated_mb: dict[str, float] = field(default_factory=dict)
    """Dedicated memory in use on each adapter across every process, keyed by adapter LUID. Empty where
    the ``GPU Adapter Memory`` counter set is unavailable."""


def parse_counter_instance_pid(instance_name: str) -> int | None:
    """Extract the OS PID from a ``GPU Process Memory`` counter instance name, or None.

    Instance names look like ``pid_12345_luid_0x00000000_0x0000ABCD_phys_0``.
    """
    match = _INSTANCE_PID_PATTERN.search(instance_name)
    if match is None:
        return None
    return int(match.group(1))


def parse_counter_instance_adapter(instance_name: str) -> str | None:
    """Extract the adapter LUID from a GPU counter instance name, or None.

    Both counter sets name the adapter the same way: ``luid_0x00000000_0x0000ABCD_phys_0`` for an adapter,
    and the same text after ``pid_12345_`` for a process on it.
    """
    match = _INSTANCE_LUID_PATTERN.search(instance_name)
    if match is None:
        return None
    return match.group(1).lower()


def adapter_dedicated_used_mb(sample: GpuProcessMemorySample, pids: set[int]) -> float | None:
    """Return the dedicated memory in use on the adapter ``pids`` run on, across every process, or None.

    The adapter is the one on which the given processes hold the most dedicated memory between them, so a
    card is found from the worker processes pinned to it with no index-to-LUID table. None when the adapter
    counters are unavailable or none of the processes holds memory on any adapter yet.
    """
    if not sample.adapter_dedicated_mb:
        return None
    held_by_adapter: dict[str, float] = {}
    for (pid, adapter), local_mb in sample.local_mb_by_pid_and_adapter.items():
        if pid in pids and local_mb > 0.0:
            held_by_adapter[adapter] = held_by_adapter.get(adapter, 0.0) + local_mb
    if not held_by_adapter:
        return None
    adapter = max(held_by_adapter, key=lambda candidate: held_by_adapter[candidate])
    return sample.adapter_dedicated_mb.get(adapter)


def sum_by_pid(mb_by_instance: dict[str, float]) -> dict[int, float]:
    """Sum per-instance megabytes by the OS PID each instance names, dropping instances that name none."""
    totals: dict[int, float] = {}
    for instance_name, megabytes in mb_by_instance.items():
        pid = parse_counter_instance_pid(instance_name)
        if pid is not None:
            totals[pid] = totals.get(pid, 0.0) + megabytes
    return totals


def sum_by_pid_and_adapter(mb_by_instance: dict[str, float]) -> dict[tuple[int, str], float]:
    """Sum per-instance megabytes by the PID and adapter each instance names, dropping any that lack either."""
    totals: dict[tuple[int, str], float] = {}
    for instance_name, megabytes in mb_by_instance.items():
        pid = parse_counter_instance_pid(instance_name)
        adapter = parse_counter_instance_adapter(instance_name)
        if pid is not None and adapter is not None:
            totals[(pid, adapter)] = totals.get((pid, adapter), 0.0) + megabytes
    return totals


def sum_by_adapter(mb_by_instance: dict[str, float]) -> dict[str, float]:
    """Sum per-instance megabytes by the adapter each instance names, dropping instances that name none."""
    totals: dict[str, float] = {}
    for instance_name, megabytes in mb_by_instance.items():
        adapter = parse_counter_instance_adapter(instance_name)
        if adapter is not None:
            totals[adapter] = totals.get(adapter, 0.0) + megabytes
    return totals


def assess_worker_paging(
    sample: GpuProcessMemorySample,
    worker_pids: set[int],
    *,
    shared_threshold_mb: float,
) -> dict[int, float]:
    """Return the worker PIDs whose shared (system-backed) GPU usage exceeds the threshold.

    A small shared mapping (tens of MB) is normal runtime bookkeeping; hundreds of MB on a compute
    process means its allocations were demoted out of dedicated VRAM: the demand-paging signature.
    Restricting to ``worker_pids`` is the attribution: an elevated foreign PID is external VRAM
    pressure, not the worker's own over-commit, and must not trip the worker-side verdict.
    """
    return {
        pid: shared_mb
        for pid, shared_mb in sample.shared_mb_by_pid.items()
        if pid in worker_pids and shared_mb >= shared_threshold_mb
    }


def _make_pdh_reader() -> Callable[[], GpuProcessMemorySample | None] | None:
    """Build a PDH-backed reader for the GPU Process Memory counters, or None when unavailable.

    Windows-only by nature (PDH and the WDDM counter set do not exist elsewhere). Probes once: any
    failure to open the query or add the counters (older Windows, counter set disabled, non-WDDM
    environment) returns None so the caller skips sampling entirely.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        pdh = ctypes.windll.pdh  # type: ignore

        class _PdhFmtCounterValueLarge(ctypes.Structure):
            _fields_ = (
                ("CStatus", wintypes.DWORD),
                ("largeValue", ctypes.c_longlong),
            )

        class _PdhFmtCounterValueItem(ctypes.Structure):
            _fields_ = (
                ("szName", ctypes.c_wchar_p),
                ("FmtValue", _PdhFmtCounterValueLarge),
            )

        query_handle = ctypes.c_void_p()
        if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query_handle)) != 0:
            return None

        counter_handles: dict[str, ctypes.c_void_p] = {}
        for kind, path in (("shared", _SHARED_USAGE_COUNTER_PATH), ("local", _LOCAL_USAGE_COUNTER_PATH)):
            handle = ctypes.c_void_p()
            # The English-name variant keeps the paths valid on localized Windows installs.
            if pdh.PdhAddEnglishCounterW(query_handle, path, 0, ctypes.byref(handle)) != 0:
                pdh.PdhCloseQuery(query_handle)
                return None
            counter_handles[kind] = handle

        # The adapter counters are a second counter set. A host without it keeps the per-process reading.
        adapter_handle = ctypes.c_void_p()
        adapter_counters_available = (
            pdh.PdhAddEnglishCounterW(query_handle, _ADAPTER_DEDICATED_COUNTER_PATH, 0, ctypes.byref(adapter_handle))
            == 0
        )

        def _read_counter_array(handle: ctypes.c_void_p) -> dict[str, float] | None:
            buffer_size = wintypes.DWORD(0)
            item_count = wintypes.DWORD(0)
            # PDH statuses are unsigned 32-bit HRESULT-style codes; ctypes returns them through the
            # default signed c_int, so mask before comparing (PDH_MORE_DATA reads as a negative int
            # otherwise and the size-probe call looks like a failure).
            status = 0xFFFFFFFF & pdh.PdhGetFormattedCounterArrayW(
                handle,
                _PDH_FMT_LARGE,
                ctypes.byref(buffer_size),
                ctypes.byref(item_count),
                None,
            )
            if status != _PDH_MORE_DATA:
                return None
            buffer = (ctypes.c_byte * buffer_size.value)()
            status = 0xFFFFFFFF & pdh.PdhGetFormattedCounterArrayW(
                handle,
                _PDH_FMT_LARGE,
                ctypes.byref(buffer_size),
                ctypes.byref(item_count),
                buffer,
            )
            if status != 0:
                return None
            items = ctypes.cast(buffer, ctypes.POINTER(_PdhFmtCounterValueItem * item_count.value)).contents
            by_instance: dict[str, float] = {}
            for item in items:
                if item.szName is None:
                    continue
                by_instance[item.szName] = by_instance.get(item.szName, 0.0) + item.FmtValue.largeValue / _BYTES_PER_MB
            return by_instance

        def _read() -> GpuProcessMemorySample | None:
            try:
                if pdh.PdhCollectQueryData(query_handle) != 0:
                    return None
                shared = _read_counter_array(counter_handles["shared"])
                local = _read_counter_array(counter_handles["local"])
                if shared is None and local is None:
                    return None
                adapters = _read_counter_array(adapter_handle) if adapter_counters_available else None
                return GpuProcessMemorySample(
                    timestamp=time.time(),
                    shared_mb_by_pid=sum_by_pid(shared or {}),
                    local_mb_by_pid=sum_by_pid(local or {}),
                    local_mb_by_pid_and_adapter=sum_by_pid_and_adapter(local or {}),
                    adapter_dedicated_mb=sum_by_adapter(adapters or {}),
                )
            except Exception:  # noqa: BLE001 - telemetry must never raise into the control loop
                return None

        # Prime the query once; a first collect that fails outright means the counter set is not
        # usable here, so report the whole source unavailable rather than polling failures forever.
        if _read() is None:
            pdh.PdhCloseQuery(query_handle)
            return None
        return _read
    except Exception as probe_error:  # noqa: BLE001 - "no PDH" is an expected environment, not a crash
        logger.debug(f"WDDM paging telemetry unavailable ({probe_error})")
        return None


class WddmPagingMonitor:
    """Polls per-process GPU local/shared usage on a background thread between ``start()`` and ``stop()``.

    ``latest()`` returns the most recent reading (or None when telemetry is unavailable or not yet
    collected), so control-loop consumers pay a dict lookup, never a PDH round-trip.
    """

    def __init__(
        self,
        *,
        interval_seconds: float = 2.0,
        read_sample: Callable[[], GpuProcessMemorySample | None] | None = None,
    ) -> None:
        """Initialize the monitor.

        Args:
            interval_seconds: How often to poll the counters. The wildcard instance enumeration has a
                real (tens of ms) cost, so this stays coarser than the utilization sampler.
            read_sample: Override the counter reader (for tests). When None, a PDH reader is probed at
                ``start()``; if unavailable the monitor no-ops.
        """
        self._interval = interval_seconds
        self._read = read_sample
        self._latest: GpuProcessMemorySample | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin sampling on a background thread (a no-op when the counter source is unavailable)."""
        if self._read is None:
            self._read = _make_pdh_reader()
        if self._read is None:
            return
        self._thread = threading.Thread(target=self._loop, name="wddm-paging-monitor", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._read is not None
        while not self._stop_event.wait(self._interval):
            try:
                sample = self._read()
            except Exception:  # noqa: BLE001 - telemetry must never raise into the control loop
                sample = None
            if sample is not None:
                self._latest = sample

    def stop(self) -> None:
        """Stop sampling."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest(self) -> GpuProcessMemorySample | None:
        """The most recent per-process reading, or None when unavailable."""
        return self._latest
