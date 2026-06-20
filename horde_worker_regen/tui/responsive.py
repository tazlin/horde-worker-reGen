"""Width-aware column selection for the dashboard's Rich tables.

Rich does not reflow a table by dropping columns: when the natural width exceeds the available space it
truncates cells instead, which is what turns a wide table into mush on an 80-column terminal. So column
shedding has to happen here, in Python, before the columns are added.

Each table declares its columns as :class:`ColumnSpec`s tagged with a :class:`DensityTier`. At render time
:func:`select_columns` keeps the essentials and then admits each higher tier only while the running width
budget still fits the available space, so a narrow terminal sheds the least-important columns first and a
wide terminal shows everything. The budget is derived from the columns' own declared widths, so it stays
honest as columns are added or retimed rather than relying on hand-tuned width thresholds.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from rich.table import Table
from rich.text import Text

Cell = str | Text
"""What a column's render callable returns: a plain string or a styled Rich ``Text``."""

Justify = Literal["left", "right", "center"]


class DensityTier(enum.IntEnum):
    """How important a column is, lowest (always shown) to highest (only on wide/detailed views).

    Ordered so a single ``<=`` comparison expresses "fits within this density", which is what both the
    width budget and the F6 intent ceiling lean on.
    """

    ESSENTIAL = 0
    """Shown at every width: the few columns that answer "what is this slot doing right now"."""
    NORMAL = 1
    """The everyday columns; admitted once the terminal is wide enough to carry them."""
    WIDE = 2
    """The enriched columns that complete the picture on a roomy terminal."""
    DETAILS = 3
    """Diagnostic columns the F6 details view requests, shown only when both intent and width allow."""


_ASCENDING_OPTIONAL_TIERS: tuple[DensityTier, ...] = (
    DensityTier.NORMAL,
    DensityTier.WIDE,
    DensityTier.DETAILS,
)
"""The tiers admitted on top of ESSENTIAL, in the order the width budget considers them."""


@dataclass(frozen=True)
class ColumnSpec[T]:
    """One table column: its header, its density tier, how to render a row, and its Rich add-column args.

    ``render`` maps a single row item to a cell; ``budget`` (defaulting to the declared/estimated width)
    is what :func:`select_columns` sums to decide whether the column fits.
    """

    header: str
    tier: DensityTier
    render: Callable[[T], Cell]
    justify: Justify = "left"
    width: int | None = None
    min_width: int | None = None
    max_width: int | None = None
    no_wrap: bool = False
    budget: int = field(default=0)
    """Explicit width estimate for the fit calculation; 0 means "derive from width/min_width/header"."""

    def content_width(self) -> int:
        """The column's width contribution to the fit budget (excludes padding and borders)."""
        if self.budget:
            return self.budget
        if self.width is not None:
            return self.width
        if self.min_width is not None:
            return self.min_width
        return max(len(self.header), 6)


@dataclass(frozen=True)
class ColumnLayout[T]:
    """The outcome of a selection: the columns to draw, and what width hid was clamped away."""

    columns: list[ColumnSpec[T]]
    hidden_count: int
    needed_width: int | None
    """The width at which the next-hidden tier would be revealed, or None when nothing is hidden."""


def _budget[T](specs: Sequence[ColumnSpec[T]]) -> int:
    """Estimate the terminal width a set of columns needs: content plus Rich's padding and borders.

    Rich's default cell padding adds two cells per column and the box draws a vertical rule between and
    around the columns; the estimate is deliberately a hair generous so a borderline fit truncates rather
    than overflows.
    """
    count = len(specs)
    content = sum(spec.content_width() for spec in specs)
    return content + 2 * count + (count + 1)


def intent_ceiling(detailed: bool) -> DensityTier:
    """Map the F6 view intent to the highest tier it permits: details unlocks DETAILS, else WIDE."""
    return DensityTier.DETAILS if detailed else DensityTier.WIDE


def select_columns[T](
    specs: Sequence[ColumnSpec[T]],
    *,
    ceiling: DensityTier,
    available_width: int | None,
) -> ColumnLayout[T]:
    """Pick the columns to draw: everything ``ceiling`` permits, clamped to what ``available_width`` fits.

    ``available_width`` of None disables shedding (used before the widget has been laid out, and by tests
    that render at a fixed console width): every permitted column is returned. Otherwise the essentials
    are always kept and each higher tier is admitted only while the whole set still fits the width, so the
    least-important columns shed first.
    """
    permitted = [spec for spec in specs if spec.tier <= ceiling]
    if available_width is None:
        return ColumnLayout(columns=list(permitted), hidden_count=0, needed_width=None)

    fitted = DensityTier.ESSENTIAL
    for tier in _ASCENDING_OPTIONAL_TIERS:
        if tier > ceiling:
            break
        cohort = [spec for spec in permitted if spec.tier <= tier]
        if not any(spec.tier is tier for spec in cohort):
            # No columns at this tier; advancing is free and lets a populated higher tier still be tried.
            fitted = tier
            continue
        if _budget(cohort) <= available_width:
            fitted = tier
        else:
            break

    columns = [spec for spec in permitted if spec.tier <= fitted]
    hidden = [spec for spec in permitted if spec.tier > fitted]
    needed_width: int | None = None
    if hidden:
        next_tier = min(spec.tier for spec in hidden)
        needed_width = _budget([spec for spec in permitted if spec.tier <= next_tier])
    return ColumnLayout(columns=columns, hidden_count=len(hidden), needed_width=needed_width)


def shed_hint[T](layout: ColumnLayout[T]) -> str | None:
    """A short caption naming how many columns the width clamped away and the width to reveal them.

    Returns None when nothing was hidden, so a table only carries the hint when it is actually clamped.
    """
    if layout.hidden_count == 0 or layout.needed_width is None:
        return None
    plural = "column" if layout.hidden_count == 1 else "columns"
    return f"+{layout.hidden_count} more {plural} at ≥{layout.needed_width} cols wide"


def add_columns[T](table: Table, specs: Sequence[ColumnSpec[T]]) -> None:
    """Add the selected columns to a Rich table, carrying each column's alignment and width hints."""
    for spec in specs:
        table.add_column(
            spec.header,
            justify=spec.justify,
            width=spec.width,
            min_width=spec.min_width,
            max_width=spec.max_width,
            no_wrap=spec.no_wrap,
        )


def placeholder_row[T](specs: Sequence[ColumnSpec[T]], message_header: str, message: str) -> list[Cell]:
    """Build an empty-state row for the selected columns, placing ``message`` under ``message_header``.

    Falls back to the first column when the named column was shed, so the message is never lost on a
    narrow terminal.
    """
    cells: list[Cell] = ["-"] * len(specs)
    if not cells:
        return cells
    headers = [spec.header for spec in specs]
    index = headers.index(message_header) if message_header in headers else 0
    cells[index] = Text(message, style="grey50")
    return cells
