"""Tests for the Stats tab rendering helpers."""

from __future__ import annotations

from rich.console import Console

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    PopGovernorsSnapshot,
    PopGovernorStatus,
    StatsRollupRow,
)
from horde_worker_regen.tui.widgets.stats import StatsView


def _render(renderable: object, width: int = 160) -> str:
    console = Console(width=width)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def test_governor_table_shows_per_governor_aggregates() -> None:
    """Each governor row reports its engagement count, total time, and session share, marking active ones."""
    governors = PopGovernorsSnapshot(
        governors=[
            PopGovernorStatus(
                name="large_model_switch",
                label="Large-model switch throttle",
                active=True,
                current_spell_seconds=8.0,
                expected_remaining_seconds=22.0,
                triggers=4,
                total_active_seconds=120.0,
                fraction_of_session=0.2,
            ),
            PopGovernorStatus(
                name="post_inference_backpressure",
                label="Post-inference backpressure",
                active=False,
                triggers=2,
                total_active_seconds=30.0,
                fraction_of_session=0.05,
            ),
        ],
        any_active=True,
    )

    text = _render(StatsView._render_governors(governors))

    assert "Pop governors" in text
    assert "Large-model switch throttle" in text
    assert "active" in text
    assert "left" in text  # the active countdown
    assert "Post-inference backpressure" in text
    assert "idle" in text
    assert "20.0%" in text


def test_form_rollups_table_shows_form_count_and_average() -> None:
    """The by-form table lists each alchemy form with its count and a derived average e2e."""
    rows = [
        StatsRollupRow(model="RealESRGAN_x4plus", jobs=4, e2e_seconds=8.0),
        StatsRollupRow(model="caption", jobs=2, e2e_seconds=10.0),
    ]

    text = _render(StatsView._render_form_rollups(rows))

    assert "By alchemy form totals" in text
    assert "RealESRGAN_x4plus" in text
    assert "caption" in text
    # 4 forms over 8s totals a 2s average; the column derives it from the rollup.
    assert "2s" in text


def test_form_rollups_table_handles_no_forms() -> None:
    """With no finalized forms the table shows a clear placeholder rather than an empty grid."""
    text = _render(StatsView._render_form_rollups([]))
    assert "no finalized alchemy forms yet" in text
