"""The Insights tab shows its live findings as finding cards, rebuilt only when the findings change."""

from __future__ import annotations

import asyncio

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from horde_worker_regen.process_management.ipc.supervisor_channel import WorkerConfigSummary, WorkerStateSnapshot
from horde_worker_regen.tui.widgets.finding_card import FindingCard
from horde_worker_regen.tui.widgets.insights import InsightsView

pytestmark = pytest.mark.slow


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield InsightsView()


def _snapshot(**overrides: object) -> WorkerStateSnapshot:
    base: dict[str, object] = {"config": WorkerConfigSummary(dreamer_name="Insights", worker_version="12.0.0")}
    base.update(overrides)
    return WorkerStateSnapshot(**base)  # type: ignore[arg-type]


async def _wait_for_cards(pilot: object, view: InsightsView, *, count: int) -> list[FindingCard]:
    for _ in range(300):
        await pilot.pause()  # type: ignore[attr-defined]
        cards = list(view.query(FindingCard))
        if len(cards) == count and all(card.query(".finding-header") for card in cards):
            return cards
        await asyncio.sleep(0.02)
    raise AssertionError(f"expected {count} cards, have {[card.finding.id for card in view.query(FindingCard)]}")


def _message(view: InsightsView) -> Static:
    return view.query_one("#insights-findings-message", Static)


@pytest.mark.asyncio
async def test_findings_become_cards_and_a_healthy_snapshot_clears_them() -> None:
    """Cards follow the findings: mounted in severity order, kept while unchanged, cleared for a healthy snapshot."""
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(InsightsView)
        view.update_snapshot(
            _snapshot(too_many_consecutive_failed_jobs=True, num_jobs_submitted=80, num_jobs_faulted=20)
        )
        cards = await _wait_for_cards(pilot, view, count=2)
        assert [card.finding.id for card in cards] == ["consecutive_failure_pause", "fault_rate"]
        assert not _message(view).display

        view.update_snapshot(
            _snapshot(too_many_consecutive_failed_jobs=True, num_jobs_submitted=80, num_jobs_faulted=20)
        )
        await pilot.pause()
        assert [id(card) for card in view.query(FindingCard)] == [id(card) for card in cards]

        view.update_snapshot(_snapshot())
        await _wait_for_cards(pilot, view, count=0)
        assert _message(view).display
        assert "looks healthy" in str(_message(view).render())
