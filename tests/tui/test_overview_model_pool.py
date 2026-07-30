"""Tests for the Overview tab's compact fixed model pool panel."""

from __future__ import annotations

from rich.console import Console
from textual.app import App, ComposeResult

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    ModelPoolBenchRow,
    ModelPoolSeatRow,
    ModelPoolSnapshot,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.health import HealthReport, HealthStatus, WorkerPhase
from horde_worker_regen.tui.widgets.overview import OverviewView
from horde_worker_regen.tui.widgets.overview_layout import element_for_node


def _render(renderable: object, width: int = 160) -> str:
    console = Console(width=width)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def _pool_snapshot() -> ModelPoolSnapshot:
    return ModelPoolSnapshot(
        enabled=True,
        seats=[
            ModelPoolSeatRow(model="Deliberate", source="MANUAL", state="ACTIVE", dwell_seconds=180.0),
            ModelPoolSeatRow(
                model="AlbedoBase XL",
                source="RANKER",
                state="PENDING_DOWNLOAD",
                pending_model="Flux.1-dev",
            ),
        ],
        bench=[ModelPoolBenchRow(model="OldModel", reason="EMPTY_POPS", cooldown_remaining_seconds=200.0)],
        current_lane="FIXED",
        last_fixed_seat_count=2,
        fixed_pops=10,
        fixed_fulfilled=7,
        free_pops=4,
        free_fulfilled=1,
    )


def test_panel_shows_seats_lane_and_fixed_hit_rate() -> None:
    """The compact panel names each seated model with its source glyph, the lane, and the fixed hit rate."""
    text = _render(OverviewView._render_model_pool_panel(_pool_snapshot()))

    assert "Model pool" in text
    assert "Deliberate" in text
    assert "AlbedoBase XL" in text
    # The downloading seat reads as downloading, the served seat as active.
    assert "downloading" in text
    assert "active" in text
    # The lane line carries the fixed-lane hit rate (7 of 10 pops fulfilled) and the bench size.
    assert "FIXED" in text
    assert "7/10" in text
    assert "70%" in text
    assert "bench" in text


def test_panel_omits_hit_rate_before_any_fixed_pop() -> None:
    """With no fixed-lane pops tallied yet the panel shows the lane but no hit-rate fraction."""
    pool = _pool_snapshot()
    pool.fixed_pops = 0
    pool.fixed_fulfilled = 0
    text = _render(OverviewView._render_model_pool_panel(pool))

    assert "fixed hits" not in text
    assert "FIXED" in text


def test_panel_handles_no_seated_models() -> None:
    """A pool holding no model yet renders a clear placeholder rather than an empty seat line."""
    pool = _pool_snapshot()
    pool.seats = []
    text = _render(OverviewView._render_model_pool_panel(pool))

    assert "no seated models yet" in text


def test_model_pool_panel_is_a_hideable_layout_element() -> None:
    """The panel node is registered in the layout registry so an operator can hide it like other panels."""
    assert element_for_node("#overview-model-pool") is not None


class _OverviewHarness(App[None]):
    """Mount an OverviewView alone so update_view's show/hide guard can be exercised."""

    def compose(self) -> ComposeResult:
        yield OverviewView()


def _idle_report() -> HealthReport:
    return HealthReport(WorkerPhase.IDLE, HealthStatus.INFO, "Idle", "", [], False)


def _snapshot(pool: ModelPoolSnapshot | None) -> WorkerStateSnapshot:
    return WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        model_pool=pool,
    )


async def test_panel_shown_when_pool_enabled_and_hidden_when_absent() -> None:
    """update_view reveals the pool panel for an enabled pool and hides it entirely when the pool is off."""
    app = _OverviewHarness()
    async with app.run_test(size=(160, 60)) as pilot:
        view = app.query_one(OverviewView)

        view.update_view(_idle_report(), _snapshot(_pool_snapshot()), frame=0)
        await pilot.pause()
        assert view.query_one("#overview-model-pool").display is True

        view.update_view(_idle_report(), _snapshot(None), frame=0)
        await pilot.pause()
        assert view.query_one("#overview-model-pool").display is False
