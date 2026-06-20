"""Tests for the Downloads view's live 'N of M models ready' aggregation."""

from __future__ import annotations

from rich.console import Console

from horde_worker_regen.process_management.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadFailure,
    DownloadItem,
    DownloadPlanSummary,
    DownloadStatusSnapshot,
)
from horde_worker_regen.tui.widgets.downloads import DownloadsView


def _render(renderable: object) -> str:
    console = Console(width=160)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def _plan(num_present: int, num_to_download: int) -> DownloadPlanSummary:
    return DownloadPlanSummary(num_present=num_present, num_to_download=num_to_download)


def _item(name: str) -> DownloadItem:
    return DownloadItem(model_name=name, feature="image model")


def _current(name: str) -> CurrentDownloadStatus:
    return CurrentDownloadStatus(model_name=name, feature="image model", target_dir="/models")


def test_readiness_is_none_without_a_download_process() -> None:
    """No live readiness when downloads are not running (the plan alone is shown)."""
    assert DownloadsView._readiness(_plan(2, 3), None) is None


def test_readiness_is_none_when_nothing_is_configured() -> None:
    """A zero-model config has no meaningful readiness fraction."""
    assert DownloadsView._readiness(_plan(0, 0), DownloadStatusSnapshot()) is None


def test_readiness_counts_pending_and_current_as_not_ready() -> None:
    """Of 5 configured, one downloading plus two queued leaves two ready."""
    downloads = DownloadStatusSnapshot(pending=[_item("b"), _item("c")], current=_current("a"))
    assert DownloadsView._readiness(_plan(num_present=2, num_to_download=3), downloads) == (2, 5)


def test_readiness_does_not_double_count_current_still_in_queue() -> None:
    """A model that is both 'current' and still listed in the queue is subtracted once."""
    downloads = DownloadStatusSnapshot(pending=[_item("a"), _item("b")], current=_current("a"))
    assert DownloadsView._readiness(_plan(num_present=1, num_to_download=2), downloads) == (1, 3)


def test_readiness_counts_failures_as_not_ready() -> None:
    """A failed model is not ready, but is not double-subtracted if also queued."""
    downloads = DownloadStatusSnapshot(
        failures=[DownloadFailure(model_name="z", feature="image model", reason="boom")],
    )
    assert DownloadsView._readiness(_plan(num_present=1, num_to_download=2), downloads) == (2, 3)


def test_readiness_all_present() -> None:
    """When everything is on disk and nothing is queued, all models are ready."""
    assert DownloadsView._readiness(_plan(num_present=3, num_to_download=0), DownloadStatusSnapshot()) == (3, 3)


def test_compact_plan_summarizes_presence_and_fit() -> None:
    """The thin-view disk plan collapses presence, to-fetch, and fit onto one line."""
    plan = DownloadPlanSummary(num_present=38, num_to_download=6, fits=True)
    text = _render(DownloadsView()._render_plan_compact(plan))
    assert "38" in text and "on disk" in text
    assert "6" in text and "to fetch" in text
    assert "fits on disk" in text


def test_compact_plan_flags_over_budget() -> None:
    """When the plan does not fit, the compact line states the shortfall instead of 'fits'."""
    plan = DownloadPlanSummary(num_present=10, num_to_download=4, fits=False, shortfall_bytes=360 * 1024**3)
    text = _render(DownloadsView()._render_plan_compact(plan))
    assert "OVER BUDGET" in text
