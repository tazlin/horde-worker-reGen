"""Tests for the Overview work-ledger's per-job pipeline-disaggregation stage line."""

from __future__ import annotations

from rich.console import Console

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    DisaggStageRow,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.widgets.overview import OverviewView


def _render(renderable: object, width: int = 160) -> str:
    console = Console(width=width)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def _snapshot(rows: list[DisaggStageRow]) -> WorkerStateSnapshot:
    config = WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0")
    return WorkerStateSnapshot(config=config, disagg_job_stages=rows)


def test_no_disagg_jobs_renders_no_line() -> None:
    """A monolithic (or idle disaggregated) worker shows no disagg stage line at all."""
    assert OverviewView._disagg_stage_line(_snapshot([])) is None


def test_stage_line_names_job_stage_and_dispatch_target() -> None:
    """Each in-flight disaggregated job shows its short id, its stage, and the process it dispatched to."""
    rows = [
        DisaggStageRow(job_id="abcdef123456", stage="sampling", process_id=3, process_launch_identifier=0),
        DisaggStageRow(job_id="99887766aabb", stage="awaiting_latent_decode"),
    ]
    line = OverviewView._disagg_stage_line(_snapshot(rows))
    assert line is not None

    text = _render(line)
    # The underscored stage value reads as spaced words, and the dispatch target shows the process.
    assert "sampling" in text
    assert "awaiting latent decode" in text
    assert "→p3" in text
    # The undispatched job shows the waiting marker rather than a process id.
    assert "→…" in text
    # Job ids are shortened to their tail so the line stays compact.
    assert "def123456" not in text
    assert "ef123456" in text


def test_stage_line_elides_beyond_the_cap() -> None:
    """More disaggregated jobs than the display cap collapse the remainder into a "+N more" marker."""
    rows = [
        DisaggStageRow(job_id=f"job{index:09d}", stage="sampling", process_id=index)
        for index in range(OverviewView._DISAGG_STAGE_LINE_CAP + 3)
    ]
    line = OverviewView._disagg_stage_line(_snapshot(rows))
    assert line is not None

    text = _render(line)
    assert "+3 more" in text
