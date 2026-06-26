"""Tests for the Stats TUI view render helpers."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    StatsRollupRow,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.widgets.stats import StatsView


def _render(renderable: object) -> str:
    console_file = StringIO()
    console = Console(file=console_file, force_terminal=False, width=120)
    console.print(renderable)
    return console_file.getvalue()


def test_stats_view_renders_minimal_snapshot() -> None:
    """The headline panel renders from a snapshot with no rollups yet."""
    snapshot = WorkerStateSnapshot(config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"))

    text = _render(StatsView._render_headlines(snapshot))

    assert "Session stats" in text
    assert "0 submitted / 0 faulted" in text


def test_stats_view_renders_populated_rollups() -> None:
    """The rollup table shows model, baseline, timings, and batch>1 job count."""
    rows = [
        StatsRollupRow(
            model="Deliberate",
            baseline="stable_diffusion_1",
            jobs=2,
            megapixelsteps=15.5,
            sampling_seconds=6.0,
            e2e_seconds=12.0,
            batch_gt_one_jobs=1,
        ),
    ]

    text = _render(StatsView._render_rollups("By model totals", rows))

    assert "Deliberate" in text
    assert "SD1.5" in text
    assert "15.5" in text
    assert "1" in text
