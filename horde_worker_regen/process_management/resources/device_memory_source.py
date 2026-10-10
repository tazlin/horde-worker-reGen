"""The device-memory reading for a card whose parent-side device read never answers.

The parent reads a card's free and total memory through NVML, which only an NVIDIA card with a working
management library answers. ROCm, XPU and MPS cards, and an NVIDIA card whose NVML cannot initialise, have no
such read. Their inference processes still report free and total memory through the execution layer's
backend-agnostic accelerator calls, and this module turns those reports into the same reading.
"""

from __future__ import annotations

import sys


def child_reported_free_total_mb(
    *,
    reported_free_mb: float | None,
    reported_total_mb: float | None,
    committed_mb: float,
    platform: str = sys.platform,
) -> tuple[float, float] | None:
    """Return a card's ``(free_mb, total_mb)`` from its processes' reports, or None before any report.

    Off Windows a process reads the whole device, so the reported free figure stands. On Windows each
    process reads a process-local view of the card, which still counts memory a sibling holds as free. There the
    figure is capped at the card's total less what the worker's processes have committed between them, the
    one device-level bound the parent can form without a device read.

    Args:
        reported_free_mb: The most conservative free figure across the card's reporting processes.
        reported_total_mb: The card's total as its processes report it.
        committed_mb: The summed footprint of the worker's live processes on the card.
        platform: ``sys.platform`` by default. A test passes either shape.
    """
    if reported_free_mb is None or reported_total_mb is None:
        return None
    if platform != "win32":
        return reported_free_mb, reported_total_mb
    ledger_free_mb = max(0.0, reported_total_mb - committed_mb)
    return min(reported_free_mb, ledger_free_mb), reported_total_mb
