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
    ADAPTER_COUNTER = "adapter_counter"
    """The dedicated memory Windows counts on the card's adapter, across every process."""
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
    used_mb: float | None = None
    """Memory in use on the device across every process, where the source measures it. It is stated apart
    from ``total_mb - free_mb`` because a driver holds some memory back that is neither. None from process
    reports, which do not measure it."""

    @property
    def is_device_wide(self) -> bool:
        """Whether every tenant of the card is inside the figure."""
        return self.provenance is ReadingProvenance.DEVICE_WIDE


@dataclass(frozen=True)
class DeviceReadingShadow:
    """What each other source reads as a card's free memory at one sample, recorded beside the reading in use.

    Nothing acts on these figures. On a card NVML answers for they say how far each alternative is from the
    device figure, which is the only way to know what a card without NVML is being priced against.
    """

    child_reported_free_mb: float | None
    """The most conservative free figure the card's processes report, as reported."""
    ledger_capped_free_mb: float | None
    """The reported figure capped at the card's total less the worker's committed ledger."""
    adapter_counter_free_mb: float | None
    """The card's total less the dedicated memory Windows counts on its adapter. None off Windows."""


def shadow_free_figures(
    *,
    reported_free_mb: float | None,
    reported_total_mb: float | None,
    committed_mb: float,
    adapter_dedicated_used_mb: float | None,
) -> DeviceReadingShadow:
    """Return every alternative free figure that can be formed from what the parent holds for a card.

    The capped figure is taken from :func:`child_reported_reading` as Windows forms it, so the comparison
    records the reading a card with no device read is given and no second copy of that arithmetic exists.
    """
    windows_fallback = child_reported_reading(
        reported_free_mb=reported_free_mb,
        reported_total_mb=reported_total_mb,
        committed_mb=committed_mb,
        platform="win32",
    )
    capped_mb = None if windows_fallback is None else windows_fallback.free_mb
    from_adapter = adapter_counter_reading(dedicated_used_mb=adapter_dedicated_used_mb, total_mb=reported_total_mb)
    return DeviceReadingShadow(
        child_reported_free_mb=reported_free_mb,
        ledger_capped_free_mb=capped_mb,
        adapter_counter_free_mb=None if from_adapter is None else from_adapter.free_mb,
    )


def nvml_reading(*, free_mb: float, total_mb: float, used_mb: float | None = None) -> DeviceMemoryReading:
    """Return the reading for a card NVML answered for, which is device-wide by construction."""
    return DeviceMemoryReading(
        free_mb=free_mb,
        total_mb=total_mb,
        source=DeviceMemorySourceKind.NVML,
        provenance=ReadingProvenance.DEVICE_WIDE,
        used_mb=used_mb,
    )


def adapter_counter_reading(*, dedicated_used_mb: float | None, total_mb: float | None) -> DeviceMemoryReading | None:
    """Return a card's reading from the dedicated memory Windows counts on its adapter, or None without both.

    The counter covers every process on the adapter, so the reading is device-wide. Free is the card's total
    less that usage. A driver also holds back memory the counter does not show (a few hundred megabytes on an
    NVIDIA card), so this free figure sits above the driver's by that amount, inside the admission margin.

    Args:
        dedicated_used_mb: The adapter's dedicated usage across every process.
        total_mb: The card's total, as its processes report it.
    """
    if dedicated_used_mb is None or total_mb is None:
        return None
    return DeviceMemoryReading(
        free_mb=max(0.0, total_mb - dedicated_used_mb),
        total_mb=total_mb,
        source=DeviceMemorySourceKind.ADAPTER_COUNTER,
        provenance=ReadingProvenance.DEVICE_WIDE,
        used_mb=dedicated_used_mb,
    )


def select_device_reading(
    *,
    from_nvml: DeviceMemoryReading | None,
    nvml_has_answered: bool,
    from_adapter_counter: DeviceMemoryReading | None,
    from_child_reports: DeviceMemoryReading | None,
) -> DeviceMemoryReading | None:
    """Return the reading a card is priced against this sample, from the sources that answered.

    NVML is the reading wherever it answers. A card NVML has answered for takes no other source, so a missed
    sample there is None and the next one fills it. A card NVML has never answered for takes the Windows
    adapter counter, which is device-wide, and failing that its processes' reports.
    """
    if from_nvml is not None:
        return from_nvml
    if nvml_has_answered:
        return None
    if from_adapter_counter is not None:
        return from_adapter_counter
    return from_child_reports


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
