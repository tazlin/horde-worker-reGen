"""Tests for the Stats TUI view render helpers."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    StatsRollupRow,
    WorkerConfigSummary,
    WorkerStateSnapshot,
    WorkloadTotalsSnapshot,
)
from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind
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


def test_the_workload_split_states_what_each_flow_completed_and_earned() -> None:
    """One pair of headline counters cannot say which flow earned what on a worker serving several."""
    totals = {
        WorkloadKind.IMAGE_GENERATION: WorkloadTotalsSnapshot(completed=40, faulted=2, kudos=310.0),
        WorkloadKind.TEXT_GENERATION: WorkloadTotalsSnapshot(
            completed=9,
            faulted=1,
            kudos=37.5,
            mean_generated_tokens=152.0,
        ),
    }

    text = _render(StatsView._render_workload_totals(totals))

    assert "By workload totals" in text
    assert "Image generation" in text
    assert "Text generation" in text
    assert "310.0" in text
    assert "37.5" in text
    assert "152" in text


def test_a_workload_whose_backend_counts_no_tokens_shows_no_mean() -> None:
    """A stock text backend counts nothing, and a zero there would be read as a measurement."""
    totals = {WorkloadKind.TEXT_GENERATION: WorkloadTotalsSnapshot(completed=3, faulted=0, kudos=8.0)}

    text = _render(StatsView._render_workload_totals(totals))

    assert "Text generation" in text
    assert "-" in text


def test_the_workload_split_shows_for_a_scribe_only_worker() -> None:
    """Its single row carries the token mean, which the headline counters have no room for."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", dreamer=False, worker_version="12.0.0", scribe=True),
        workload_totals={
            WorkloadKind.TEXT_GENERATION: WorkloadTotalsSnapshot(
                completed=9,
                faulted=1,
                kudos=37.5,
                mean_generated_tokens=152.0,
            ),
        },
    )

    assert StatsView._shows_workload_split(snapshot) is True


def test_the_workload_split_stays_hidden_on_an_image_only_worker() -> None:
    """One workload's totals are the headline figures already, so the table would only restate them."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        workload_totals={WorkloadKind.IMAGE_GENERATION: WorkloadTotalsSnapshot(completed=40, faulted=2, kudos=310.0)},
    )

    assert StatsView._shows_workload_split(snapshot) is False
