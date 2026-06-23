"""The GPUs screen: a per-card breakdown for multi-GPU operators (one collapsed card on a single-GPU host).

Each driven card gets one row: its VRAM headroom (with a near-OOM pressure flag), how many inference
contexts it is running against its target, its throughput (it/s summed from the card's processes, plus a
jobs/hr trend derived here from successive per-card completion counts), and -- in the F6 details view -- the
whole-card residency it may be holding and any models gone locally unservable on it. The table sheds columns
to fit the terminal exactly like the overview tables, and the thin view collapses the whole tab to a single
aggregate line.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from horde_worker_regen.app_state import OverviewViewMode
from horde_worker_regen.process_management.supervisor_channel import CardSnapshot, WorkerStateSnapshot
from horde_worker_regen.tui.formatters import (
    format_its,
    gpu_label,
    mini_bar,
    shorten,
    sparkline,
)
from horde_worker_regen.tui.responsive import (
    ColumnSpec,
    DensityTier,
    add_columns,
    intent_ceiling,
    select_columns,
    shed_hint,
)

_TREND_HISTORY = 180
"""How many per-card jobs-completed samples each card's trend buffer retains."""

_TREND_SAMPLE_INTERVAL = 1.0
"""Minimum wall-clock seconds between recorded per-card samples, so the window spans minutes not frames."""

_TREND_SPARK_WIDTH = 24
"""Maximum number of recent samples drawn in a per-card jobs/hr sparkline."""


@dataclass(frozen=True)
class _CardRow:
    """One per-card row paired with the throughput figures derived on the receiving side.

    ``its_per_second`` is summed from the card's live processes and ``jobs_per_hour`` / ``jobs_spark`` come
    from this view's per-card history, so each :class:`CardSnapshot` column can render from a single row.
    """

    card: CardSnapshot
    its_per_second: float | None = None
    jobs_per_hour: float | None = None
    jobs_spark: list[float] = field(default_factory=list)


def _gpu_cell(row: _CardRow) -> Text:
    """The GPU cell: the device index and its trimmed model name (or backend kind)."""
    card = row.card
    return Text(gpu_label(card.device_index, card.device_name, card.kind), no_wrap=True)


def _vram_cell(row: _CardRow) -> Text:
    """The VRAM cell: a used-fraction bar plus free/total in GB, reddened under VRAM pressure."""
    card = row.card
    if card.free_vram_mb is None or not card.total_vram_mb:
        return Text("-", style="grey50")
    fraction = card.vram_headroom_fraction
    used_fraction = 1.0 - fraction if fraction is not None else 0.0
    style = "red" if card.is_vram_pressured else "green"
    text = Text()
    text.append(mini_bar(used_fraction, 8), style=style)
    text.append(f" {card.free_vram_mb / 1024:.1f}/{card.total_vram_mb / 1024:.1f}G free", style="")
    if card.is_vram_pressured:
        text.append(" ⚠", style="bold red")
    return text


def _contexts_cell(row: _CardRow) -> Text:
    """The Contexts cell: busy/loaded contexts against the card's target process count."""
    card = row.card
    style = "green" if card.busy_contexts > 0 else "grey62"
    return Text(f"{card.busy_contexts}/{card.loaded_contexts}▸{card.target_process_count}", style=style)


def _its_cell(row: _CardRow) -> str:
    """The it/s cell: the card's combined sampling rate across its busy processes."""
    return format_its(row.its_per_second)


def _jobs_cell(row: _CardRow) -> Text:
    """The Jobs/hr cell: the per-card rate plus a sparkline of recent completions."""
    rate = "-" if row.jobs_per_hour is None else f"{row.jobs_per_hour:,.0f}"
    text = Text(rate)
    spark = sparkline(row.jobs_spark)
    if spark:
        text.append(" ")
        text.append(spark, style="green")
    return text


def _duty_cell(row: _CardRow) -> Text:
    """The Duty cell: a busy-context proxy bar (this card's mid-inference share of its loaded contexts).

    A proxy for hardware duty until per-card NVML sampling lands: it answers "is this card's capacity being
    used" from the contexts actually sampling, which is the actionable signal for an under-fed card.
    """
    card = row.card
    if card.loaded_contexts <= 0:
        return Text("-", style="grey50")
    fraction = card.busy_contexts / card.loaded_contexts
    return Text(mini_bar(fraction, 8), style="green")


def _residency_cell(row: _CardRow) -> Text:
    """The Residency cell: the whole-card residency model and phase held on this card, or a dash."""
    card = row.card
    if not card.residency_model:
        return Text("-", style="grey50")
    return Text(f"{shorten(card.residency_model, 16)} ({card.residency_phase})", style="#f0beff")


def _faults_cell(row: _CardRow) -> Text:
    """The Faults cell: models gone locally unservable on this card and the worst over-budget streak."""
    card = row.card
    if card.unservable_models:
        return Text(f"{len(card.unservable_models)} unservable ×{card.worst_fault_streak}", style="bold red")
    if card.worst_fault_streak > 0:
        return Text(f"streak {card.worst_fault_streak}", style="yellow")
    return Text("-", style="grey50")


_CARD_COLUMNS: list[ColumnSpec[_CardRow]] = [
    ColumnSpec("GPU", DensityTier.ESSENTIAL, _gpu_cell, min_width=10, no_wrap=True),
    ColumnSpec("VRAM", DensityTier.ESSENTIAL, _vram_cell, min_width=22, no_wrap=True),
    ColumnSpec("Contexts", DensityTier.ESSENTIAL, _contexts_cell, justify="right", width=10, no_wrap=True),
    ColumnSpec("it/s", DensityTier.NORMAL, _its_cell, justify="right", width=9),
    ColumnSpec("Jobs/hr", DensityTier.NORMAL, _jobs_cell, min_width=12, no_wrap=True),
    ColumnSpec("Conc", DensityTier.WIDE, lambda r: str(r.card.max_concurrent_inference), justify="right", width=5),
    ColumnSpec("Duty", DensityTier.WIDE, _duty_cell, width=9, no_wrap=True),
    ColumnSpec("Residency", DensityTier.DETAILS, _residency_cell, min_width=18, no_wrap=True),
    ColumnSpec("Faults", DensityTier.DETAILS, _faults_cell, min_width=14, no_wrap=True),
]
"""The per-card table's columns, tagged by the density tier at which each appears."""


class GpusView(VerticalScroll):
    """A per-card multi-GPU breakdown that rides the same F6 density cycle as the overview."""

    def __init__(self) -> None:
        """Set up the view, including the per-card jobs-completed history for the jobs/hr trend."""
        super().__init__()
        self._jobs_history: dict[int, deque[tuple[float, int]]] = {}
        self._last_sample = 0.0

    def compose(self) -> ComposeResult:
        """Lay out the single body Static the table (or thin aggregate line) renders into."""
        yield Static(id="gpus-body")

    def update_view(
        self,
        snapshot: WorkerStateSnapshot | None,
        *,
        mode: OverviewViewMode = OverviewViewMode.NORMAL,
    ) -> None:
        """Refresh the per-card view for the active ``mode`` from the snapshot's ``per_card`` section."""
        body = self.query_one("#gpus-body", Static)
        if snapshot is None or not snapshot.per_card:
            body.update(Panel(Text("waiting for first snapshot", style="grey50"), title="GPUs", title_align="left"))
            return

        self._maybe_record(snapshot)

        if mode is OverviewViewMode.THIN:
            body.update(self._render_aggregate(snapshot))
            return

        width = self.content_size.width or None
        body.update(self._render_table(snapshot, detailed=mode is OverviewViewMode.DETAILS, available_width=width))

    def _maybe_record(self, snapshot: WorkerStateSnapshot) -> None:
        """Append one per-card jobs-completed sample at most once per :data:`_TREND_SAMPLE_INTERVAL`."""
        now = time.time()
        if now - self._last_sample < _TREND_SAMPLE_INTERVAL:
            return
        self._last_sample = now
        for card in snapshot.per_card:
            history = self._jobs_history.setdefault(card.device_index, deque(maxlen=_TREND_HISTORY))
            history.append((now, card.jobs_completed))

    def _card_jobs_per_hour(self, device_index: int) -> tuple[float | None, list[float]]:
        """Derive a jobs/hr rate and per-sample completion series for one card from its history."""
        samples = list(self._jobs_history.get(device_index, ()))
        if len(samples) < 2:
            return None, []
        deltas = [float(max(0, b[1] - a[1])) for a, b in zip(samples, samples[1:], strict=False)]
        elapsed = samples[-1][0] - samples[0][0]
        completed = samples[-1][1] - samples[0][1]
        rate = (completed / elapsed * 3600.0) if elapsed > 0 else None
        return rate, deltas[-_TREND_SPARK_WIDTH:]

    def _card_its(self, snapshot: WorkerStateSnapshot, device_index: int) -> float | None:
        """Sum the live sampling rate (it/s) across this card's busy inference processes, or None."""
        rates = [
            process.last_iterations_per_second
            for process in snapshot.processes
            if process.device_index == device_index
            and process.last_iterations_per_second is not None
            and process.last_iterations_per_second > 0
        ]
        return sum(rates) if rates else None

    def _build_rows(self, snapshot: WorkerStateSnapshot) -> list[_CardRow]:
        """Pair each card snapshot with its derived it/s and jobs/hr trend for rendering."""
        rows: list[_CardRow] = []
        for card in snapshot.per_card:
            rate, deltas = self._card_jobs_per_hour(card.device_index)
            rows.append(
                _CardRow(
                    card=card,
                    its_per_second=self._card_its(snapshot, card.device_index),
                    jobs_per_hour=rate,
                    jobs_spark=deltas,
                ),
            )
        return rows

    def _render_table(
        self,
        snapshot: WorkerStateSnapshot,
        *,
        detailed: bool,
        available_width: int | None,
    ) -> RenderableType:
        """Build the per-card table whose columns shed to fit ``available_width``."""
        layout = select_columns(_CARD_COLUMNS, ceiling=intent_ceiling(detailed), available_width=available_width)
        table = Table(
            title="GPUs",
            title_style="bold",
            expand=True,
            border_style="grey37",
            header_style="bold",
        )
        add_columns(table, layout.columns)
        for row in self._build_rows(snapshot):
            table.add_row(*[spec.render(row) for spec in layout.columns])
        if (hint := shed_hint(layout)) is not None:
            table.caption = hint
            table.caption_style = "italic grey50"
        return table

    @staticmethod
    def _render_aggregate(snapshot: WorkerStateSnapshot) -> Panel:
        """Collapse every card to one line: card count, total free VRAM, loaded contexts, and active ones."""
        cards = snapshot.per_card
        free_values = [card.free_vram_mb for card in cards if card.free_vram_mb is not None]
        free_g = (sum(free_values) / 1024) if free_values else None
        loaded = sum(card.loaded_contexts for card in cards)
        busy = sum(card.busy_contexts for card in cards)
        pressured = sum(1 for card in cards if card.is_vram_pressured)

        sep = ("   ·   ", "grey37")
        line = Text.assemble(
            (f"{len(cards)} GPU{'s' if len(cards) != 1 else ''}", "bold"),
            sep,
            (f"{free_g:.1f}G free" if free_g is not None else "VRAM ?", "cyan"),
            sep,
            (f"{loaded} ctx", "grey70"),
            sep,
            (f"{busy} active", "green"),
        )
        if pressured:
            line.append_text(Text.assemble(sep, (f"⚠ {pressured} pressured", "bold red")))
        return Panel(line, title="GPUs", title_align="left", border_style="grey37", padding=(0, 1))
