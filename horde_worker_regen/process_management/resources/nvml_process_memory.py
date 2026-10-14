"""Per-process device memory on one card, read by the parent through NVML.

The parent's foreign-VRAM reading is the card's use less the worker's own. Where the driver reports each
compute process's device memory, the worker's share is measured directly as the sum over its processes'
pids, which needs no estimate of contexts or allocator reports. NVML reports that figure on Linux. Under
WDDM it lists the processes without a usage figure, so the reader returns None there and the caller keeps
its estimate.

The reader is torch-free and goes through hordelib's NVML initialisation, so a host without NVML is latched off
once and every later call returns None at no cost.
"""

from __future__ import annotations

from loguru import logger

_BYTES_PER_MB = 1024 * 1024

_unavailable_reported = False
"""Whether the first failure to read the process list has been logged. Later failures go to TRACE, since the
read runs once per governor tick."""


def _note_unavailable(reason: str) -> None:
    """Log why the process list is unavailable, at DEBUG once and at TRACE afterwards."""
    global _unavailable_reported
    if _unavailable_reported:
        logger.trace(f"NVML per-process memory unavailable: {reason}")
        return
    _unavailable_reported = True
    logger.debug(f"NVML per-process memory unavailable: {reason}")


def read_process_device_memory_mb(device_index: int) -> dict[int, float] | None:
    """Return each compute process's device memory (MB) on a card, keyed by OS pid, or None.

    None when NVML is unavailable, the card's process list cannot be read, or the driver gives any listed
    process no usage figure (WDDM), since a partial sum would read as a smaller worker footprint than the card
    holds. An empty mapping is a valid reading, meaning no compute process holds a context on the card.

    Args:
        device_index: The NVML device index of the card.
    """
    try:
        from hordelib.utils.nvml import is_nvml_available
    except ImportError as import_error:
        _note_unavailable(f"hordelib NVML helpers did not import ({import_error})")
        return None
    if not is_nvml_available():
        return None
    try:
        import pynvml
    except ImportError as import_error:
        _note_unavailable(f"the NVML binding did not import ({import_error})")
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except pynvml.NVMLError as nvml_error:
        _note_unavailable(f"device {device_index}: {nvml_error}")
        return None
    usage_mb: dict[int, float] = {}
    for process in processes:
        used_bytes = process.usedGpuMemory
        if used_bytes is None:
            _note_unavailable(f"device {device_index} reports no per-process usage")
            return None
        pid = int(process.pid)
        usage_mb[pid] = usage_mb.get(pid, 0.0) + used_bytes / _BYTES_PER_MB
    return usage_mb
