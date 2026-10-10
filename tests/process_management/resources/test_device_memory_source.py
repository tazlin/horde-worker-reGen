"""The device-memory reading formed from a card's processes where no parent-side device read answers."""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.resources.device_memory_source import (
    DeviceMemorySourceKind,
    ReadingProvenance,
    child_reported_reading,
    nvml_reading,
    shadow_free_figures,
)


@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
@pytest.mark.parametrize(
    ("free_mb", "total_mb"),
    [(None, 24000.0), (20000.0, None), (None, None)],
)
def test_no_reading_before_a_process_has_reported(
    platform: str, free_mb: float | None, total_mb: float | None
) -> None:
    """A card none of whose processes has reported both figures has no reading on any platform."""
    reading = child_reported_reading(
        reported_free_mb=free_mb,
        reported_total_mb=total_mb,
        committed_mb=0.0,
        platform=platform,
    )

    assert reading is None


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_a_report_off_windows_stands_as_a_device_wide_reading(platform: str) -> None:
    """Off Windows a process reads the whole device, so the committed ledger does not lower its figure."""
    reading = child_reported_reading(
        reported_free_mb=18000.0,
        reported_total_mb=24000.0,
        committed_mb=15000.0,
        platform=platform,
    )

    assert reading is not None
    assert (reading.free_mb, reading.total_mb) == (18000.0, 24000.0)
    assert reading.source is DeviceMemorySourceKind.CHILD_REPORT
    assert reading.provenance is ReadingProvenance.DEVICE_WIDE
    assert reading.is_device_wide


def test_a_windows_report_is_process_local_and_capped_by_what_the_worker_has_committed() -> None:
    """On Windows a process still counts a sibling's memory as free, so the ledger bounds the figure."""
    reading = child_reported_reading(
        reported_free_mb=15000.0,
        reported_total_mb=16000.0,
        committed_mb=9000.0,
        platform="win32",
    )

    assert reading is not None
    assert (reading.free_mb, reading.total_mb) == (7000.0, 16000.0)
    assert reading.provenance is ReadingProvenance.PROCESS_LOCAL
    assert not reading.is_device_wide


def test_a_windows_report_below_the_ledger_bound_stands() -> None:
    """The cap only lowers the figure. A process that reads less free than the ledger implies is believed."""
    reading = child_reported_reading(
        reported_free_mb=3000.0,
        reported_total_mb=16000.0,
        committed_mb=9000.0,
        platform="win32",
    )

    assert reading is not None
    assert reading.free_mb == 3000.0


def test_a_ledger_past_the_card_total_reads_as_no_room() -> None:
    """A committed sum beyond the card's total leaves zero free, never a negative figure."""
    reading = child_reported_reading(
        reported_free_mb=15000.0,
        reported_total_mb=16000.0,
        committed_mb=17500.0,
        platform="win32",
    )

    assert reading is not None
    assert reading.free_mb == 0.0


@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
def test_a_host_memory_card_reports_system_ram_uncapped_and_marked(platform: str) -> None:
    """A unified-memory or CPU device reports system RAM, which no VRAM ledger bounds on any platform."""
    reading = child_reported_reading(
        reported_free_mb=40000.0,
        reported_total_mb=65536.0,
        committed_mb=60000.0,
        host_memory=True,
        platform=platform,
    )

    assert reading is not None
    assert reading.free_mb == 40000.0
    assert reading.provenance is ReadingProvenance.HOST_MEMORY
    assert not reading.is_device_wide


def test_the_shadow_figures_are_formed_from_what_the_parent_holds() -> None:
    """Each alternative free figure is stated beside the others, with the cap and the adapter figure applied."""
    shadow = shadow_free_figures(
        reported_free_mb=15009.0,
        reported_total_mb=16375.0,
        committed_mb=2144.0,
        adapter_dedicated_used_mb=3240.0,
    )

    assert shadow.child_reported_free_mb == 15009.0
    assert shadow.ledger_capped_free_mb == 14231.0
    assert shadow.adapter_counter_free_mb == 13135.0


def test_a_shadow_figure_with_no_input_is_none() -> None:
    """A figure whose source has not answered is absent, and the others are still formed."""
    unreported = shadow_free_figures(
        reported_free_mb=None,
        reported_total_mb=None,
        committed_mb=0.0,
        adapter_dedicated_used_mb=3240.0,
    )
    no_counter = shadow_free_figures(
        reported_free_mb=15009.0,
        reported_total_mb=16375.0,
        committed_mb=0.0,
        adapter_dedicated_used_mb=None,
    )

    assert (unreported.child_reported_free_mb, unreported.ledger_capped_free_mb) == (None, None)
    assert unreported.adapter_counter_free_mb is None
    assert no_counter.adapter_counter_free_mb is None
    assert no_counter.ledger_capped_free_mb == 15009.0


def test_an_nvml_reading_is_device_wide() -> None:
    """NVML reads the device itself, so its reading is device-wide by construction."""
    reading = nvml_reading(free_mb=9000.0, total_mb=16376.0)

    assert reading.source is DeviceMemorySourceKind.NVML
    assert reading.is_device_wide
