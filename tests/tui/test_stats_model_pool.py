"""Tests for the Stats tab's fixed model pool section and its by-model seat marking."""

from __future__ import annotations

from rich.console import Console
from textual.app import App, ComposeResult

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    ModelPoolBenchRow,
    ModelPoolSeatReadiness,
    ModelPoolSeatRow,
    ModelPoolSnapshot,
    StatsRollupRow,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.widgets.stats import StatsView


def _render(renderable: object, width: int = 160) -> str:
    console = Console(width=width)
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
                readiness=ModelPoolSeatReadiness.RESIDENT,
            ),
            ModelPoolSeatRow(model=None, source=None, state="ACTIVE"),
        ],
        bench=[ModelPoolBenchRow(model="OldModel", reason="EMPTY_POPS", cooldown_remaining_seconds=200.0)],
        current_lane="FIXED",
        fixed_pops=10,
        fixed_fulfilled=6,
        fixed_resident_hits=5,
        free_pops=4,
        free_fulfilled=1,
    )


def test_pool_section_shows_per_lane_matches_residency_seats_and_bench() -> None:
    """The pool section reports pop matches, resident hits, ready seats, and bench size."""
    text = _render(StatsView._render_model_pool(_pool_snapshot()))

    assert "Model pool" in text
    assert "FIXED" in text
    assert "1 resident / 1 seated / 2 total" in text
    assert "6 / 10 matched (60%); 5 resident" in text
    assert "1 / 4 matched (25%); 0 resident" in text


def test_lane_pop_summary_dashes_before_any_pop() -> None:
    """A lane with no pops yet shows a dash rather than a divide-by-zero rate."""
    assert StatsView._lane_pop_summary(0, 0, 0) == "-"
    assert StatsView._lane_pop_summary(3, 4, 2) == "3 / 4 matched (75%); 2 resident"


def test_by_model_rollup_marks_seated_models() -> None:
    """A seated model is marked in the by-model table and the legend appears; unseated rows are unmarked."""
    rows = [
        StatsRollupRow(model="Deliberate", baseline="stable_diffusion_1", jobs=5),
        StatsRollupRow(model="SomeOtherModel", baseline="stable_diffusion_xl", jobs=2),
    ]

    text = _render(StatsView._render_rollups("By model totals", rows, seated_models=frozenset({"Deliberate"})))

    assert "◆" in text
    assert "holds a model-pool seat" in text


def test_by_model_rollup_without_seats_has_no_marker() -> None:
    """With no seated models the table carries neither the marker glyph nor its legend."""
    rows = [StatsRollupRow(model="Deliberate", baseline="stable_diffusion_1", jobs=5)]

    text = _render(StatsView._render_rollups("By model totals", rows))

    assert "◆" not in text
    assert "holds a model-pool seat" not in text


class _StatsHarness(App[None]):
    """Mount a StatsView alone so update_snapshot's show/hide guard can be exercised."""

    def compose(self) -> ComposeResult:
        yield StatsView()


def _snapshot(pool: ModelPoolSnapshot | None) -> WorkerStateSnapshot:
    return WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        model_pool=pool,
    )


async def test_pool_section_shown_only_when_enabled() -> None:
    """The pool section displays for an enabled pool and hides entirely when the pool is off."""
    app = _StatsHarness()
    async with app.run_test(size=(160, 60)) as pilot:
        view = app.query_one(StatsView)

        view.update_snapshot(_snapshot(_pool_snapshot()))
        await pilot.pause()
        assert view.query_one("#stats-model-pool").display is True

        view.update_snapshot(_snapshot(None))
        await pilot.pause()
        assert view.query_one("#stats-model-pool").display is False
