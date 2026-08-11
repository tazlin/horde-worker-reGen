"""Tests for the width-driven column selection the dashboard's Rich tables share."""

from __future__ import annotations

from horde_worker_regen.tui.responsive import (
    ColumnSpec,
    DensityTier,
    intent_ceiling,
    select_columns,
    shed_hint,
)

_COLUMNS: list[ColumnSpec[str]] = [
    ColumnSpec("Name", DensityTier.CRITICAL, lambda row: row, width=10),
    ColumnSpec("State", DensityTier.ESSENTIAL, lambda row: row, width=20),
    ColumnSpec("Rate", DensityTier.NORMAL, lambda row: row, width=20),
    ColumnSpec("Detail", DensityTier.DETAILS, lambda row: row, width=20),
]


def _headers(available_width: int | None, *, ceiling: DensityTier = DensityTier.DETAILS) -> list[str]:
    """The headers selected for a width, for readable assertions."""
    return [spec.header for spec in select_columns(_COLUMNS, ceiling=ceiling, available_width=available_width).columns]


def test_a_wide_terminal_keeps_every_permitted_column() -> None:
    """Given room for all of them, nothing is shed."""
    assert _headers(200) == ["Name", "State", "Rate", "Detail"]


def test_columns_shed_from_the_least_important_first() -> None:
    """Each tier is admitted only while the whole set still fits, so the tail goes first."""
    assert _headers(60) == ["Name", "State", "Rate"]
    assert _headers(40) == ["Name", "State"]


def test_the_essentials_shed_below_the_terminal_floor() -> None:
    """A phone-width table sheds the essential columns rather than truncating them into mush."""
    assert _headers(20) == ["Name"]


def test_the_identity_column_is_never_shed() -> None:
    """Even at an absurd width the row-identity column survives: rows must stay tellable apart."""
    assert _headers(1) == ["Name"]


def test_disabled_shedding_returns_every_permitted_column() -> None:
    """A width of None (pre-layout, or a fixed-width test console) selects on intent alone."""
    assert _headers(None, ceiling=DensityTier.ESSENTIAL) == ["Name", "State"]


def test_simple_experience_clamps_to_the_essentials() -> None:
    """The Simple intent never admits the tuning columns, however wide the terminal is."""
    assert intent_ceiling(detailed=False, simple=True) is DensityTier.ESSENTIAL
    assert _headers(200, ceiling=intent_ceiling(detailed=False, simple=True)) == ["Name", "State"]


def test_shed_hint_counts_what_the_width_removed_and_names_the_width_to_get_it_back() -> None:
    """The caption reports the hidden columns and the width at which the next tier reappears."""
    layout = select_columns(_COLUMNS, ceiling=DensityTier.DETAILS, available_width=20)
    hint = shed_hint(layout)

    assert layout.hidden_count == 3
    assert hint is not None
    assert "3 more columns" in hint
    assert layout.needed_width is not None
    assert str(layout.needed_width) in hint
    # Re-rendering at the named width really does bring the next tier back.
    assert "State" in _headers(layout.needed_width)


def test_shed_hint_is_absent_when_nothing_was_clamped() -> None:
    """A table showing everything carries no caption, so the caption's presence is the signal."""
    assert shed_hint(select_columns(_COLUMNS, ceiling=DensityTier.DETAILS, available_width=200)) is None
