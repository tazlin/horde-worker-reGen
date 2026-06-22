"""Tests for the Downloads view's live 'N of M models ready' aggregation."""

from __future__ import annotations

import random

from rich.console import Console

from horde_worker_regen.process_management.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadFailure,
    DownloadItem,
    DownloadPhase,
    DownloadPlanSummary,
    DownloadStatusSnapshot,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.widgets.downloads import DownloadsView, summarize_download_activity


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


def test_readiness_is_the_present_count_while_others_are_in_flight() -> None:
    """Of 5 configured with 2 on disk, exactly 2 are ready while the rest download/queue."""
    downloads = DownloadStatusSnapshot(pending=[_item("b"), _item("c")], current=_current("a"))
    assert DownloadsView._readiness(_plan(num_present=2, num_to_download=3), downloads) == (2, 5)


def test_readiness_unaffected_by_queue_membership() -> None:
    """The headline count is the plan's present count; queue contents never alter it."""
    downloads = DownloadStatusSnapshot(pending=[_item("a"), _item("b")], current=_current("a"))
    assert DownloadsView._readiness(_plan(num_present=1, num_to_download=2), downloads) == (1, 3)


def test_readiness_reflects_present_count_not_the_queue() -> None:
    """Readiness is the live on-disk present count; a failed/queued model simply is not present yet.

    (Single-source semantics: of 3 configured with 1 on disk, exactly 1 is ready regardless of what the
    download queue or failures list says.)
    """
    downloads = DownloadStatusSnapshot(
        failures=[DownloadFailure(model_name="z", feature="image model", reason="boom")],
    )
    assert DownloadsView._readiness(_plan(num_present=1, num_to_download=2), downloads) == (1, 3)


def test_readiness_all_present() -> None:
    """When everything is on disk and nothing is queued, all models are ready."""
    assert DownloadsView._readiness(_plan(num_present=3, num_to_download=0), DownloadStatusSnapshot()) == (3, 3)


def test_readiness_does_not_collapse_when_queue_exceeds_plan() -> None:
    """A queue larger than the plan's to-download count must not zero out an otherwise-present set.

    Regression for the reported 100/101 -> 0: the live queue derives presence by a different route than
    the plan (hordelib's sha256-gated availability vs on_disk_layout) and could drift far larger than the
    plan's missing set. Single-sourcing the count from the plan makes the queue irrelevant to the tally.
    """
    downloads = DownloadStatusSnapshot(
        phase=DownloadPhase.DOWNLOADING,
        pending=[_item(f"m{i}") for i in range(100)],
        current=_current("m100"),
    )
    # 100 present + 1 to download = 101 total; the bloated 101-entry queue must not perturb the count.
    assert DownloadsView._readiness(_plan(num_present=100, num_to_download=1), downloads) == (100, 101)


def test_readiness_invariant_holds_under_adversarial_drift() -> None:
    """Fuzz: ``ready`` is always ``num_present`` and stays within ``[num_present, total]``.

    Pins the invariant any future edit must keep: the live download queue (however it drifts) can never
    alter the headline tally, so an already-present model never reads as not-ready (``ready >=
    num_present``) and the count never exceeds the configured total (``ready <= total``).
    """
    rng = random.Random(20260622)
    for _ in range(3000):
        num_present = rng.randint(0, 200)
        num_to_download = rng.randint(0, 200)
        plan = _plan(num_present, num_to_download)
        pending = [_item(f"p{rng.randint(0, 400)}") for _ in range(rng.randint(0, 400))]
        current = _current(f"c{rng.randint(0, 400)}") if rng.random() < 0.7 else None
        failures = [
            DownloadFailure(model_name=f"f{rng.randint(0, 400)}", feature="image model", reason="x")
            for _ in range(rng.randint(0, 400))
        ]
        downloads = DownloadStatusSnapshot(pending=pending, current=current, failures=failures)

        result = DownloadsView._readiness(plan, downloads)
        total = num_present + num_to_download
        if total <= 0:
            assert result is None
            continue
        assert result is not None
        ready, reported_total = result
        assert reported_total == total
        assert num_present <= ready <= total, (num_present, num_to_download, ready, total)


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


def _snapshot(
    *,
    downloads: DownloadStatusSnapshot | None = None,
    plan: DownloadPlanSummary | None = None,
) -> WorkerStateSnapshot:
    return WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="t", worker_version="0.0.0"),
        downloads=downloads,
        download_plan=plan,
    )


def _downloading(
    name: str,
    *,
    downloaded: int,
    total: int,
    speed: float,
    paused: bool = False,
) -> DownloadStatusSnapshot:
    return DownloadStatusSnapshot(
        phase=DownloadPhase.PAUSED if paused else DownloadPhase.DOWNLOADING,
        current=CurrentDownloadStatus(
            model_name=name,
            feature="image model",
            target_dir="/models",
            downloaded_bytes=downloaded,
            total_bytes=total,
            speed_bps=speed,
        ),
        paused=paused,
    )


def test_summary_is_none_without_a_snapshot_or_downloads() -> None:
    """No activity to surface when there is no snapshot, no download process, or no current file."""
    assert summarize_download_activity(None) is None
    assert summarize_download_activity(_snapshot()) is None
    assert summarize_download_activity(_snapshot(downloads=DownloadStatusSnapshot())) is None


def test_summary_is_none_when_idle_even_with_a_present_count() -> None:
    """An IDLE process (nothing in flight) yields no activity badge, even if models are present."""
    downloads = DownloadStatusSnapshot(phase=DownloadPhase.IDLE)
    assert summarize_download_activity(_snapshot(downloads=downloads, plan=_plan(3, 0))) is None


def test_summary_reports_current_file_percent_and_ready_fraction() -> None:
    """An in-flight download surfaces the current-file percent, speed, and the ready/total fraction."""
    downloads = _downloading("Zalando", downloaded=512, total=1024, speed=2048.0)
    downloads.pending = [_item("queued")]
    snapshot = _snapshot(downloads=downloads, plan=_plan(num_present=1, num_to_download=2))
    activity = summarize_download_activity(snapshot)
    assert activity is not None
    assert activity.paused is False
    assert activity.current_name == "Zalando"
    assert activity.percent == 50.0
    assert activity.speed_bps == 2048.0
    # 3 configured (1 present + 2 to download); one downloading + one queued => 1 ready.
    assert (activity.ready, activity.total) == (1, 3)


def test_summary_count_is_none_when_plan_not_yet_loaded() -> None:
    """An active download before the plan is computed summarizes with no count, not a misleading 0/0."""
    downloads = _downloading("Zalando", downloaded=1, total=2, speed=1.0)
    activity = summarize_download_activity(_snapshot(downloads=downloads, plan=None))
    assert activity is not None
    assert activity.ready is None
    assert activity.total is None


def test_summary_flags_paused_download() -> None:
    """A paused mid-download still summarizes (so the badge can show the pause marker)."""
    downloads = _downloading("Zalando", downloaded=256, total=1024, speed=0.0, paused=True)
    activity = summarize_download_activity(_snapshot(downloads=downloads, plan=_plan(0, 1)))
    assert activity is not None
    assert activity.paused is True
