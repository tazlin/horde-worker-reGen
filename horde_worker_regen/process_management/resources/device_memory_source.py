"""A card's device-memory reading, with the source it came from and how far it describes the device.

The parent reads a card's free and total memory through NVML, which only an NVIDIA card with a working
management library answers. ROCm, XPU and MPS cards, and an NVIDIA card whose NVML cannot initialise, have no
such read. Their inference processes still report free and total memory through the execution layer's
backend-agnostic accelerator calls, and this module turns those reports into the same reading.

A consumer takes the figure and its provenance together, because the same number means different things by
source. A device-wide figure is what the card has. A process-local figure still counts memory a sibling
process holds as free. A host-memory figure is system RAM a unified-memory card shares with everything else.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from strenum import StrEnum


class DeviceMemorySourceKind(StrEnum):
    """Where a card's memory reading was taken."""

    NVML = "nvml"
    """The parent's NVML read of the device."""
    CHILD_REPORT = "child_report"
    """The figures the card's processes report through the accelerator layer."""


class ReadingProvenance(StrEnum):
    """How far a memory reading describes the device it is attributed to."""

    DEVICE_WIDE = "device_wide"
    """Every tenant of the card is inside the figure."""
    PROCESS_LOCAL = "process_local"
    """The reporting process's view of the card, which can read memory another process holds as free."""
    HOST_MEMORY = "host_memory"
    """System RAM reported as the card's memory, as a unified-memory or CPU device does."""


@dataclass(frozen=True)
class DeviceMemoryReading:
    """One card's free and total memory at a sample, with its source and provenance."""

    free_mb: float
    total_mb: float
    source: DeviceMemorySourceKind
    provenance: ReadingProvenance

    @property
    def is_device_wide(self) -> bool:
        """Whether every tenant of the card is inside the figure."""
        return self.provenance is ReadingProvenance.DEVICE_WIDE


def nvml_reading(*, free_mb: float, total_mb: float) -> DeviceMemoryReading:
    """Return the reading for a card NVML answered for, which is device-wide by construction."""
    return DeviceMemoryReading(
        free_mb=free_mb,
        total_mb=total_mb,
        source=DeviceMemorySourceKind.NVML,
        provenance=ReadingProvenance.DEVICE_WIDE,
    )


def child_reported_reading(
    *,
    reported_free_mb: float | None,
    reported_total_mb: float | None,
    committed_mb: float,
    host_memory: bool = False,
    platform: str = sys.platform,
) -> DeviceMemoryReading | None:
    """Return a card's reading from its processes' reports, or None before any report.

    Off Windows a process reads the whole device, so the reported free figure stands as device-wide. On
    Windows each process reads a process-local view of the card, which still counts memory a sibling holds as
    free. There the figure is capped at the card's total less what the worker's processes have committed
    between them, the one device-level bound the parent can form without a device read. A unified-memory or
    CPU device reports system RAM, which is passed through and marked as host memory.

    Args:
        reported_free_mb: The most conservative free figure across the card's reporting processes.
        reported_total_mb: The card's total as its processes report it.
        committed_mb: The summed footprint of the worker's live processes on the card.
        host_memory: Whether the card's backend reports system RAM as its memory.
        platform: ``sys.platform`` by default. A test passes either shape.
    """
    if reported_free_mb is None or reported_total_mb is None:
        return None
    if host_memory:
        provenance = ReadingProvenance.HOST_MEMORY
        free_mb = reported_free_mb
    elif platform == "win32":
        provenance = ReadingProvenance.PROCESS_LOCAL
        free_mb = min(reported_free_mb, max(0.0, reported_total_mb - committed_mb))
    else:
        provenance = ReadingProvenance.DEVICE_WIDE
        free_mb = reported_free_mb
    return DeviceMemoryReading(
        free_mb=free_mb,
        total_mb=reported_total_mb,
        source=DeviceMemorySourceKind.CHILD_REPORT,
        provenance=provenance,
    )
