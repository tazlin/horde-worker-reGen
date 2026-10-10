"""The device-memory reading formed from a card's processes where no parent-side device read answers."""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.resources.device_memory_source import child_reported_free_total_mb


@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
@pytest.mark.parametrize(
    ("free_mb", "total_mb"),
    [(None, 24000.0), (20000.0, None), (None, None)],
)
def test_no_reading_before_a_process_has_reported(
    platform: str, free_mb: float | None, total_mb: float | None
) -> None:
    """A card none of whose processes has reported both figures has no reading on any platform."""
    reading = child_reported_free_total_mb(
        reported_free_mb=free_mb,
        reported_total_mb=total_mb,
        committed_mb=0.0,
        platform=platform,
    )

    assert reading is None


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_a_device_wide_report_stands_as_reported(platform: str) -> None:
    """Off Windows a process reads the whole device, so the committed ledger does not lower its figure."""
    reading = child_reported_free_total_mb(
        reported_free_mb=18000.0,
        reported_total_mb=24000.0,
        committed_mb=15000.0,
        platform=platform,
    )

    assert reading == (18000.0, 24000.0)


def test_a_process_local_report_is_capped_by_what_the_worker_has_committed() -> None:
    """On Windows a process still counts a sibling's memory as free, so the ledger bounds the figure."""
    reading = child_reported_free_total_mb(
        reported_free_mb=15000.0,
        reported_total_mb=16000.0,
        committed_mb=9000.0,
        platform="win32",
    )

    assert reading == (7000.0, 16000.0)


def test_a_process_local_report_below_the_ledger_bound_stands() -> None:
    """The cap only lowers the figure: a process that reads less free than the ledger implies is believed."""
    reading = child_reported_free_total_mb(
        reported_free_mb=3000.0,
        reported_total_mb=16000.0,
        committed_mb=9000.0,
        platform="win32",
    )

    assert reading == (3000.0, 16000.0)


def test_a_ledger_past_the_card_total_reads_as_no_room() -> None:
    """A committed sum beyond the card's total leaves zero free, never a negative figure."""
    reading = child_reported_free_total_mb(
        reported_free_mb=15000.0,
        reported_total_mb=16000.0,
        committed_mb=17500.0,
        platform="win32",
    )

    assert reading == (0.0, 16000.0)
