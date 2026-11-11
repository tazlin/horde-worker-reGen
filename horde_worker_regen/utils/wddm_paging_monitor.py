"""Per-process GPU shared-memory sampling: the direct WDDM demand-paging signal.

Under Windows' WDDM, over-subscribing dedicated VRAM does not fail allocations: the OS demotes them to
the system-memory-backed *shared* GPU segment and pages on demand. In that regime the usual telemetry
lies: ``mem_get_info`` keeps reporting generous free VRAM and core utilization reads high while the
device mostly waits on the bus, so throughput halves with no visible error. The OS, however, tracks the
demotion per process: the ``GPU Process Memory`` performance counters (the data behind Task Manager's
"Shared GPU memory" column) expose each process's local (dedicated) and shared (system-backed) GPU bytes
by PID. A worker child whose *shared* usage climbs while the card's dedicated memory is short is being
demand-paged, and the PID attribution distinguishes the worker's own over-commit from external VRAM pressure
(a game or browser would show under its own PID instead). The shared counter also includes host memory
registered with the driver by a process, so shared usage while dedicated memory is free is not paging
(:func:`assess_worker_paging`).

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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from loguru import logger
from strenum import StrEnum

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


def _format_mb(megabytes: float | None) -> str:
    return "unread" if megabytes is None else f"{megabytes:.0f}MB"


@dataclass(frozen=True)
class WorkerPagingFigures:
    """Represents one worker process's GPU memory at a paging sample, beside its card's device reading.

    The shared figure alone does not separate allocations demoted out of dedicated VRAM from host memory
    registered with the driver by the process (ComfyUI pins offloaded weights with ``cudaHostRegister``), which
    Windows may count in the same segment. A demotion needs the card's dedicated memory to be short, so the process's
    local usage and the card's free and total are recorded with it.
    """

    pid: int
    shared_mb: float
    local_mb: float | None
    """The process's dedicated GPU memory, from the ``Local Usage`` counter. None without a reading for it."""
    adapter_dedicated_used_mb: float | None
    """Dedicated memory in use on the process's adapter across every process. None without the adapter counters."""
    device_index: int | None
    device_free_mb: float | None
    """Free memory in the parent's last device-wide reading for the process's card. None without one."""
    device_total_mb: float | None
    stale_device_reading_age_seconds: float | None = None
    """The age of a device-wide reading left out for being older than the sample interval. None otherwise."""

    def describe(self) -> str:
        """Return the figures as the text of one process in a log line."""
        stale = (
            ""
            if self.stale_device_reading_age_seconds is None
            else f" (reading {self.stale_device_reading_age_seconds:.1f}s old)"
        )
        return (
            f"pid {self.pid}: {self.shared_mb:.0f}MB shared, {_format_mb(self.local_mb)} local, "
            f"card {self.device_index} free {_format_mb(self.device_free_mb)} of {_format_mb(self.device_total_mb)}"
            f"{stale}, adapter dedicated {_format_mb(self.adapter_dedicated_used_mb)}"
        )

    def as_inputs(self) -> dict[str, int | float | None]:
        """Return the figures as flat inputs for a stats row."""
        return {
            "pid": self.pid,
            "shared_mb": round(self.shared_mb, 1),
            "local_mb": None if self.local_mb is None else round(self.local_mb, 1),
            "adapter_dedicated_used_mb": (
                None if self.adapter_dedicated_used_mb is None else round(self.adapter_dedicated_used_mb, 1)
            ),
            "device_index": self.device_index,
            "device_free_mb": None if self.device_free_mb is None else round(self.device_free_mb, 1),
            "device_total_mb": None if self.device_total_mb is None else round(self.device_total_mb, 1),
        }


class SharedUsageVerdict(StrEnum):
    """One paging sample's verdict on a worker process whose shared usage is over the threshold."""

    COUNTED_DEDICATED_SHORT = "counted, dedicated memory short"
    """The card's free dedicated memory is at or under its admission margin, so a demotion is possible."""
    COUNTED_CARD_UNREAD = "counted, card unread"
    """The process's card does not have a device-wide reading, so the shared usage counts alone."""
    NOT_COUNTED_CARD_HAS_ROOM = "not counted, card has room"
    """The card's free dedicated memory is above its admission margin, so the shared usage is not a demotion."""

    @property
    def counted(self) -> bool:
        """Whether the process counts toward the paging verdict in this sample."""
        return self is not SharedUsageVerdict.NOT_COUNTED_CARD_HAS_ROOM


@dataclass(frozen=True)
class WorkerPagingAssessment:
    """Represents one paging sample's verdict on each worker process whose shared usage is over the threshold."""

    verdict_by_pid: dict[int, SharedUsageVerdict]
    shared_mb_by_pid: dict[int, float]
    """The shared usage of each process in ``verdict_by_pid``."""

    @property
    def elevated_shared_mb_by_pid(self) -> dict[int, float]:
        """Return the shared usage of each process counted toward the paging verdict."""
        return {pid: self.shared_mb_by_pid[pid] for pid, verdict in self.verdict_by_pid.items() if verdict.counted}


def assess_worker_paging(
    figures_by_pid: Mapping[int, WorkerPagingFigures],
    *,
    shared_threshold_mb: float,
    margin_mb_by_device: Mapping[int, float],
) -> WorkerPagingAssessment:
    """Return the sample's verdict on each worker process whose shared GPU usage is over the threshold.

    A small shared mapping (tens of MB) is normal runtime bookkeeping. More than the threshold is either
    allocations demoted by the driver out of dedicated VRAM, or host memory registered with the driver by the
    process (ComfyUI registers offloaded weights this way). The driver demotes only when dedicated memory runs
    out, so a process counts only while its card's free memory is at or under the card's admission margin.
    Without a device-wide reading for the card, or a margin for it, the two cannot be told apart, and the process
    counts on its shared usage alone.

    ``figures_by_pid`` holds only the worker's own processes, which is the attribution. A foreign process with
    high shared usage is external VRAM pressure and does not count toward the worker's verdict.

    Args:
        figures_by_pid: The sample's figures for each worker process, keyed by OS pid.
        shared_threshold_mb: The shared usage at or above which a process is judged.
        margin_mb_by_device: The admission margin of each card with a device-wide reading, keyed by device index.
    """
    verdict_by_pid: dict[int, SharedUsageVerdict] = {}
    shared_mb_by_pid: dict[int, float] = {}
    for pid, figures in figures_by_pid.items():
        if figures.shared_mb < shared_threshold_mb:
            continue
        margin_mb = None if figures.device_index is None else margin_mb_by_device.get(figures.device_index)
        if figures.device_free_mb is None or margin_mb is None:
            verdict_by_pid[pid] = SharedUsageVerdict.COUNTED_CARD_UNREAD
        elif figures.device_free_mb <= margin_mb:
            verdict_by_pid[pid] = SharedUsageVerdict.COUNTED_DEDICATED_SHORT
        else:
            verdict_by_pid[pid] = SharedUsageVerdict.NOT_COUNTED_CARD_HAS_ROOM
        shared_mb_by_pid[pid] = figures.shared_mb
    return WorkerPagingAssessment(verdict_by_pid=verdict_by_pid, shared_mb_by_pid=shared_mb_by_pid)


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

    @property
    def interval(self) -> float:
        """Seconds between samples."""
        return self._interval

    def set_interval(self, interval_seconds: float) -> None:
        """Change how often the counters are polled, from the next sample on.

        The paging verdict is content with a coarse interval. A reader that prices admission from the adapter
        counter needs the figure at the pace it acts, and asks for that here.
        """
        self._interval = interval_seconds

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
