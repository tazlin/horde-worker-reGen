"""Tests for the insights view's fixed model pool panel rendering."""

from __future__ import annotations

from rich.console import Console

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    ModelPoolBenchRow,
    ModelPoolSeatRow,
    ModelPoolSnapshot,
)
from horde_worker_regen.tui.widgets.insights import InsightsView


def _render(renderable: object) -> str:
    console = Console(width=160)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def _pool_snapshot() -> ModelPoolSnapshot:
    return ModelPoolSnapshot(
        enabled=True,
        seats=[
            ModelPoolSeatRow(
                model="Deliberate",
                source="MANUAL",
                state="ACTIVE",
                dwell_seconds=180.0,
                empty_pops=2,
                last_fulfilled_age_seconds=12.0,
            ),
            ModelPoolSeatRow(
                model="AlbedoBase XL",
                source="RANKER",
                state="PENDING_DOWNLOAD",
                dwell_seconds=45.0,
                pending_model="Flux.1-dev",
            ),
        ],
        bench=[ModelPoolBenchRow(model="OldModel", reason="EMPTY_POPS", cooldown_remaining_seconds=200.0)],
        current_lane="FIXED",
        last_fixed_seat_count=2,
        demand_age_seconds=42.0,
        download_budget_gb=10.0,
        download_bytes_charged=1024 * 1024 * 1024,
    )


def test_model_pool_panel_renders_seats_lane_and_bench() -> None:
    """The pool panel surfaces each seat, the lane/demand status line, and the benched model."""
    text = _render(InsightsView()._render_model_pool(_pool_snapshot()))

    assert "Model pool" in text
    assert "Deliberate" in text
    assert "AlbedoBase XL" in text
    # A pending seat shows the model it is downloading to swap in.
    assert "Flux.1-dev" in text
    # The lane/demand line and its budget usage.
    assert "FIXED" in text
    assert "budget" in text
    # The bench line names the demoted model and its reason.
    assert "OldModel" in text
    assert "EMPTY_POPS" in text


def test_model_pool_status_line_omits_budget_when_not_configured() -> None:
    """With no download budget the status line drops the budget field rather than showing 0."""
    pool = _pool_snapshot()
    pool.download_budget_gb = 0.0
    text = _render(InsightsView()._render_pool_status_line(pool))

    assert "budget" not in text
    assert "FIXED" in text


def test_model_pool_bench_line_summarizes_overflow() -> None:
    """Beyond the shown bench rows the line notes how many more models are benched."""
    pool = _pool_snapshot()
    pool.bench = [
        ModelPoolBenchRow(model=f"Model{index}", reason="timer_lost", cooldown_remaining_seconds=60.0)
        for index in range(5)
    ]
    text = _render(InsightsView()._render_pool_bench(pool))

    assert "+2 more" in text
