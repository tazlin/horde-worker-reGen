"""Tests for the console StatusReporter's downloads section and startup plan summary."""

from __future__ import annotations

from collections.abc import Callable

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadFailure,
    DownloadItem,
    DownloadPhase,
    DownloadPlanSummary,
    DownloadStatusSnapshot,
)
from horde_worker_regen.reporting.status_reporter import StatusReporter

_GB = 1024**3


def _collector() -> tuple[Callable[..., None], list[str]]:
    """Return a logging-function stand-in that records the lines it is given."""
    lines: list[str] = []

    def _record(message: str, *_args: object, **_kwargs: object) -> None:
        lines.append(message)

    return _record, lines


def test_print_downloads_returns_false_with_no_data() -> None:
    """With neither status nor plan there is nothing to show and nothing printed."""
    reporter = StatusReporter(0.0, 0.0)
    record, lines = _collector()
    assert reporter._print_downloads(record, None, None) is False
    assert lines == []


def test_plan_summary_line_reports_fit_and_over_budget() -> None:
    """The one-line plan summary states the fit verdict, with a shortfall when over budget."""
    fits = DownloadPlanSummary(
        present_bytes=_GB,
        to_download_bytes=2 * _GB,
        total_bytes=3 * _GB,
        free_disk_bytes=100 * _GB,
        fits=True,
        shortfall_bytes=0,
        num_present=1,
        num_to_download=1,
        sizes_complete=True,
    )
    over = DownloadPlanSummary(
        present_bytes=0,
        to_download_bytes=20 * _GB,
        total_bytes=20 * _GB,
        free_disk_bytes=5 * _GB,
        fits=False,
        shortfall_bytes=15 * _GB,
        num_present=0,
        num_to_download=2,
        sizes_complete=False,
    )
    assert "fits" in StatusReporter._plan_summary_line(fits)
    over_line = StatusReporter._plan_summary_line(over)
    assert "OVER BUDGET" in over_line
    assert "lower bound" in over_line  # sizes_complete is False


def test_print_downloads_renders_current_queue_and_failures() -> None:
    """The downloads section shows the current download, the queue, and any failures."""
    reporter = StatusReporter(0.0, 0.0)
    record, lines = _collector()
    status = DownloadStatusSnapshot(
        phase=DownloadPhase.DOWNLOADING,
        current=CurrentDownloadStatus(
            model_name="Flux",
            feature="image model",
            target_dir="models/compvis",
            downloaded_bytes=3 * _GB,
            total_bytes=12 * _GB,
            speed_bps=90 * 1024 * 1024,
            eta_seconds=100.0,
        ),
        pending=[DownloadItem(model_name="Albedo", feature="image model", size_bytes=6 * _GB)],
        failures=[DownloadFailure(model_name="Bad", feature="LoRa", reason="out of disk space")],
        rate_limit_kbps=5000,
    )

    assert reporter._print_downloads(record, status, None) is True
    blob = "\n".join(lines)
    assert "Flux" in blob
    assert "models/compvis" in blob
    assert "Queued (1)" in blob
    assert "out of disk space" in blob
    assert "limit 5000 KB/s" in blob


def test_print_download_status_shows_paused() -> None:
    """A paused snapshot is labelled as paused in the phase line."""
    reporter = StatusReporter(0.0, 0.0)
    record, lines = _collector()
    status = DownloadStatusSnapshot(phase=DownloadPhase.PAUSED, paused=True)
    reporter._print_downloads(record, status, None)
    assert any("PAUSED" in line for line in lines)
