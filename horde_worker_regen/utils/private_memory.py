"""Measure process RAM without treating reclaimable Linux checkpoint mappings as allocator growth."""

import sys
from pathlib import Path

import psutil
from loguru import logger

_unavailable_readings: set[tuple[str, type[Exception]]] = set()


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


def private_ram_usage_bytes(process: psutil.Process) -> int | None:
    """Return non-reclaimable private working-set bytes, falling back to USS off Linux or without procfs."""
    if sys.platform == "linux":
        try:
            reading = linux_private_ram_bytes(Path(f"/proc/{process.pid}/smaps_rollup").read_text(encoding="utf-8"))
            if reading is not None:
                return reading
        except (OSError, ValueError) as error:
            _log_unavailable_once("Linux private", error)
    try:
        return process.memory_full_info().uss
    except (psutil.Error, AttributeError, NotImplementedError) as error:
        _log_unavailable_once("Unique", error)
        return None
