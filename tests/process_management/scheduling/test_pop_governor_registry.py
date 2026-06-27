"""Tests for the pop-governor registry: spell tracking, session aggregates, and boundary log lines."""

from __future__ import annotations

from horde_worker_regen.process_management.scheduling.pop_governor_registry import (
    PopGovernorReading,
    PopGovernorRegistry,
)

_SWITCH = "large_model_switch"
_LABEL = "Large-model switch throttle"


def _reading(*, active: bool, reason: str | None = None, remaining: float | None = None) -> PopGovernorReading:
    return PopGovernorReading(
        name=_SWITCH,
        label=_LABEL,
        active=active,
        reason=reason,
        expected_remaining_seconds=remaining,
    )


class TestSpellTracking:
    """A governor's active spell opens, holds, and closes across successive readings."""

    def test_engaging_opens_a_spell_and_counts_a_trigger(self) -> None:
        """The first active reading opens a spell, increments triggers, and logs an ENTER line."""
        lines: list[str] = []
        registry = PopGovernorRegistry(log=lines.append)

        registry.update([_reading(active=True, reason="churn", remaining=30.0)], now=100.0)

        views = registry.views(now=100.0, session_elapsed_seconds=100.0)
        assert len(views) == 1
        view = views[0]
        assert view.active is True
        assert view.triggers == 1
        assert view.reason == "churn"
        assert view.expected_remaining_seconds == 30.0
        assert any(line.startswith(f"Pop governor ENTER: {_SWITCH}") for line in lines)

    def test_holding_accumulates_current_spell_time(self) -> None:
        """While a spell stays open the current-spell seconds grow and no new trigger is counted."""
        registry = PopGovernorRegistry(log=lambda _line: None)
        registry.update([_reading(active=True)], now=100.0)
        registry.update([_reading(active=True)], now=130.0)

        view = registry.views(now=130.0, session_elapsed_seconds=130.0)[0]
        assert view.triggers == 1
        assert view.current_spell_seconds == 30.0
        assert view.total_active_seconds == 30.0

    def test_releasing_closes_the_spell_and_banks_total(self) -> None:
        """An inactive reading closes the spell, banks its time, and logs an EXIT line with the totals."""
        lines: list[str] = []
        registry = PopGovernorRegistry(log=lines.append)
        registry.update([_reading(active=True)], now=100.0)
        registry.update([_reading(active=False)], now=140.0)

        view = registry.views(now=200.0, session_elapsed_seconds=200.0)[0]
        assert view.active is False
        assert view.current_spell_seconds == 0.0
        assert view.total_active_seconds == 40.0
        assert view.triggers == 1
        assert any("Pop governor EXIT: large_model_switch after 40s" in line for line in lines)

    def test_second_engagement_counts_a_new_trigger_and_adds_to_total(self) -> None:
        """A re-engagement after release is a new spell: triggers increments and totals accumulate."""
        registry = PopGovernorRegistry(log=lambda _line: None)
        registry.update([_reading(active=True)], now=100.0)
        registry.update([_reading(active=False)], now=120.0)  # 20s
        registry.update([_reading(active=True)], now=200.0)
        registry.update([_reading(active=False)], now=215.0)  # 15s

        view = registry.views(now=300.0, session_elapsed_seconds=300.0)[0]
        assert view.triggers == 2
        assert view.total_active_seconds == 35.0


class TestSnapshotShape:
    """The view list omits never-engaged governors and orders active ones first."""

    def test_never_engaged_governor_is_omitted(self) -> None:
        """A governor that only ever reads inactive is not shown (keeps the dashboard uncluttered)."""
        registry = PopGovernorRegistry(log=lambda _line: None)
        registry.update([_reading(active=False)], now=100.0)
        assert registry.views(now=100.0, session_elapsed_seconds=100.0) == []

    def test_fraction_of_session_is_total_over_elapsed(self) -> None:
        """``fraction_of_session`` is aggregate active time over the session length, clamped to [0, 1]."""
        registry = PopGovernorRegistry(log=lambda _line: None)
        registry.update([_reading(active=True)], now=0.0)
        registry.update([_reading(active=False)], now=25.0)  # 25s of a 100s session

        view = registry.views(now=100.0, session_elapsed_seconds=100.0)[0]
        assert view.fraction_of_session == 0.25

    def test_active_governors_sort_before_idle(self) -> None:
        """Active governors come first so the dashboard leads with what is engaged right now."""
        registry = PopGovernorRegistry(log=lambda _line: None)
        idle = PopGovernorReading(name="a_idle", label="A", active=True)
        live = PopGovernorReading(name="b_live", label="B", active=True)
        registry.update([idle, live], now=0.0)
        registry.update([PopGovernorReading(name="a_idle", label="A", active=False), live], now=10.0)

        names = [v.name for v in registry.views(now=20.0, session_elapsed_seconds=20.0)]
        assert names[0] == "b_live"  # still active sorts ahead of the released one
