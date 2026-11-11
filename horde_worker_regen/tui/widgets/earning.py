"""Card earning section: what each model earns on each card, where each card's time went, and time under claims.

Every figure, band and verdict comes from the worker's
[`CardEarningSnapshot`][horde_worker_regen.process_management.ipc.supervisor_channel.CardEarningSnapshot]. This
module turns them into words and bars and does not compute any statistic of its own.

- [`verdict_phrase`][horde_worker_regen.tui.widgets.earning.verdict_phrase]: the one wording of an earning
  verdict, read by this section and by the Overview's residency panel.
- [`claim_earning_row`][horde_worker_regen.tui.widgets.earning.claim_earning_row]: the row of a standing claim's
  model on its card.
- [`CardEarningView`][horde_worker_regen.tui.widgets.earning.CardEarningView]: the Stats tab section, with a
  plain form at Simple and a detailed form at Advanced and above.
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    CardEarningCardRow,
    CardEarningRow,
    CardEarningSnapshot,
    CardEarningWindowRestart,
)
from horde_worker_regen.process_management.scheduling.governance.whole_card import PopClaimValueVerdict
from horde_worker_regen.tui.formatters import human_duration, is_low_fidelity, shorten

__all__ = [
    "VERDICT_TEXT",
    "CardEarningView",
    "claim_earning_row",
    "render_earning_detail",
    "render_earning_simple",
    "verdict_phrase",
]

VERDICT_TEXT: dict[str, str] = {
    PopClaimValueVerdict.UNMEASURED: "not measured yet",
    PopClaimValueVerdict.PAYS: "not below the rest",
    PopClaimValueVerdict.BELOW_THE_REST: "below the rest",
}
"""How each ``PopClaimValueVerdict`` reads when the model's rates are not at hand. An unknown value is shown as it
arrives, so a worker newer than the dashboard still says something true."""

_PAYS_AT_OR_ABOVE_TEXT = "earning well"
"""A ``pays`` verdict whose model rate is at or above the rest of its card's rate."""

_PAYS_UNDER_TEXT = "about the same as the rest"
"""A ``pays`` verdict whose model rate is under the rest's rate, with the two bands overlapping."""

_BAR_WIDTH = 12
"""The longest earning bar, in characters. The best earner on a card fills it."""

_SHARE_BAR_WIDTH = 24
"""The longest time-share bar, in characters. The card's whole wall fills it, so a share of a few percent still
differs from one of ten."""

_MODEL_NAME_WIDTH = 24
"""The widest model name in the plain form, so a line fits an 80-column terminal."""


def verdict_phrase(verdict: str, *, row: CardEarningRow | None = None, min_jobs: int | None = None) -> str:
    """Return the plain words for an earning verdict.

    A ``pays`` verdict reads as "earning well" or "about the same as the rest" when ``row`` gives both rates,
    by which of the two is higher. An ``unmeasured`` verdict names the job count when that count is under the
    floor reported by the worker.

    Args:
        verdict: The verdict string from the snapshot.
        row: The model's earning row on its card, when the caller has one.
        min_jobs: The worker's job floor for a verdict, or None when the worker did not report one.
    """
    if verdict == PopClaimValueVerdict.PAYS and row is not None:
        model_rate = row.model_kudos_per_second
        rest_rate = row.rest_kudos_per_second
        if model_rate is not None and rest_rate is not None:
            return _PAYS_AT_OR_ABOVE_TEXT if model_rate >= rest_rate else _PAYS_UNDER_TEXT
    phrase = VERDICT_TEXT.get(verdict, verdict)
    if verdict == PopClaimValueVerdict.UNMEASURED and row is not None and min_jobs is not None and row.jobs < min_jobs:
        phrase += f" ({row.jobs} of {min_jobs} jobs)"
    return phrase


def claim_earning_row(
    earning: CardEarningSnapshot,
    *,
    model: str,
    device_index: int | None,
) -> CardEarningRow | None:
    """Return the earning row for ``model`` on the claim's card, or None when the snapshot has none.

    Args:
        earning: The snapshot's earning figures.
        model: The claimed model.
        device_index: The claim's card, or None when the claim did not name one.
    """
    for row in earning.rows:
        if row.model == model and (device_index is None or row.device_index == device_index):
            return row
    return None


def _bar(fraction: float, *, width: int = _BAR_WIDTH) -> str:
    """Return a bar of filled blocks for ``fraction`` of ``width``, at least one block above zero."""
    if fraction <= 0:
        return "·"
    filled = max(1, round(min(fraction, 1.0) * width))
    return ("#" if is_low_fidelity() else "█") * filled


def _share_text(fraction: float) -> str:
    """Return a share of the wall as a whole percentage, with tiny non-zero shares shown as under one."""
    if 0 < fraction < 0.005:
        return "<1%"
    return f"{fraction * 100:.0f}%"


def _plain_span(seconds: float) -> str:
    """Return a duration in words an operator reads at a glance, such as "9 hours" or "40 minutes"."""
    minutes = max(1, round(seconds / 60))
    if minutes < 90:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = round(seconds / 3600)
    return f"{hours} hour{'s' if hours != 1 else ''}"


def _join_words(words: list[str]) -> str:
    """Return ``words`` joined as "a", "a and b" or "a, b and c"."""
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def _restart_sentence(restart: CardEarningWindowRestart) -> str:
    """Return the sentence that says why the window last restarted."""
    if restart.changed_settings:
        return f"Restarted when {_join_words(restart.changed_settings)} changed."
    return "Restarted because the settings differed from the last run's."


def _footer(earning: CardEarningSnapshot, *, now: float) -> Text:
    """Return the lines under the section: the window's span, the last restart and the rows left out."""
    sentences: list[str] = []
    if earning.window_started_at is not None:
        sentences.append(f"Measured over the last {_plain_span(now - earning.window_started_at)}.")
    if earning.last_restart is not None:
        sentences.append(_restart_sentence(earning.last_restart))
    if earning.rows_omitted > 0:
        noun = "model" if earning.rows_omitted == 1 else "models"
        sentences.append(f"And {earning.rows_omitted} more {noun} with less card time.")
    return Text(" ".join(sentences), style="grey62")


def _is_empty(earning: CardEarningSnapshot) -> bool:
    """Return whether the window holds no figures at all."""
    return not earning.rows and not earning.cards


def _empty_message() -> Text:
    """Return the message for a window with no figures."""
    return Text(
        "Not measured yet. The worker measures what each model earns from the jobs it finishes, and no "
        "finished job is counted yet. The figures appear here once the first jobs finish.",
        style="grey62",
    )


def _card_rows(earning: CardEarningSnapshot, device_index: int) -> list[CardEarningRow]:
    """Return the card's rows, the ones with a bar first by model rate, then the rest in the snapshot's order."""
    card_rows = [row for row in earning.rows if row.device_index == device_index]
    with_bar = [row for row in card_rows if _has_bar(row)]
    without_bar = [row for row in card_rows if not _has_bar(row)]
    with_bar.sort(key=lambda row: -(row.model_kudos_per_second or 0.0))
    return with_bar + without_bar


def _has_bar(row: CardEarningRow) -> bool:
    """Return whether the row's rate is drawn as a bar. A model not measured yet is not drawn."""
    return row.verdict != PopClaimValueVerdict.UNMEASURED and row.model_kudos_per_second is not None


def _card_indices(earning: CardEarningSnapshot) -> list[int]:
    """Return every card in the window, by device index ascending."""
    return sorted({card.device_index for card in earning.cards} | {row.device_index for row in earning.rows})


def _card_wall(earning: CardEarningSnapshot, device_index: int) -> CardEarningCardRow:
    """Return the card's time row, or an empty one when the snapshot does not list the card."""
    for card in earning.cards:
        if card.device_index == device_index:
            return card
    return CardEarningCardRow(device_index=device_index)


def _card_title(device_index: int, *, several_cards: bool) -> str:
    """Return the block title, naming the card only on a host with more than one."""
    return f"What each model earns on card {device_index}" if several_cards else "What each model earns on this card"


def _simple_card_block(earning: CardEarningSnapshot, device_index: int, *, several_cards: bool) -> Panel:
    """Return the plain block for one card: the earning lines, the time shares and the claim sentences."""
    rows = _card_rows(earning, device_index)
    best = max((row.model_kudos_per_second or 0.0 for row in rows if _has_bar(row)), default=0.0)

    earn_grid = Table.grid(padding=(0, 2))
    earn_grid.add_column(no_wrap=True, width=_MODEL_NAME_WIDTH)
    earn_grid.add_column(no_wrap=True, width=_BAR_WIDTH)
    earn_grid.add_column()
    for row in rows:
        rate = row.model_kudos_per_second or 0.0
        bar = _bar(rate / best) if _has_bar(row) and best > 0 else "·"
        earn_grid.add_row(
            shorten(row.model, _MODEL_NAME_WIDTH),
            Text(bar, style="cyan"),
            verdict_phrase(row.verdict, row=row, min_jobs=earning.verdict_min_jobs),
        )
    if not rows:
        earn_grid.add_row(Text("No model has card time here yet.", style="grey62"), "", "")

    parts: list[RenderableType] = [earn_grid, Text(""), Text("Where the card's time went", style="bold")]
    parts.append(_simple_time_grid(earning, device_index))
    claim_lines = _simple_claim_lines(earning, device_index)
    if claim_lines:
        parts.extend([Text(""), Text("Time spent offering one model only", style="bold"), *claim_lines])
    return Panel(
        Group(*parts),
        title=_card_title(device_index, several_cards=several_cards),
        title_align="left",
        border_style="grey37",
        padding=(0, 1),
    )


def _time_shares(earning: CardEarningSnapshot, device_index: int) -> list[tuple[str, float, float]]:
    """Return each share of the card's wall as (label, fraction, seconds): its models, any left out, then idle.

    The models left out of the snapshot have the card's attributed seconds that the listed rows do not have.
    """
    card = _card_wall(earning, device_index)
    if card.wall_seconds <= 0:
        return []
    card_rows = [row for row in earning.rows if row.device_index == device_index]
    shares = [
        (row.model, row.attributed_card_seconds / card.wall_seconds, row.attributed_card_seconds)
        for row in card_rows
        if row.attributed_card_seconds > 0
    ]
    listed_seconds = sum(row.attributed_card_seconds for row in card_rows)
    other_seconds = card.attributed_seconds - listed_seconds
    if earning.rows_omitted > 0 and other_seconds > 0:
        shares.append(("other models", other_seconds / card.wall_seconds, other_seconds))
    shares.append(("idle", card.unattributed_seconds / card.wall_seconds, card.unattributed_seconds))
    return shares


def _simple_time_grid(earning: CardEarningSnapshot, device_index: int) -> RenderableType:
    """Return the plain time shares: one line per model and one for idle, each a bar and a percentage."""
    shares = _time_shares(earning, device_index)
    if not shares:
        return Text("No card time measured yet.", style="grey62")
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True, width=_MODEL_NAME_WIDTH)
    grid.add_column(no_wrap=True, width=_SHARE_BAR_WIDTH)
    grid.add_column(no_wrap=True, justify="right")
    for label, fraction, _seconds in shares:
        style = "grey62" if label == "idle" else "green"
        bar = _bar(fraction, width=_SHARE_BAR_WIDTH)
        grid.add_row(shorten(label, _MODEL_NAME_WIDTH), Text(bar, style=style), _share_text(fraction))
    return grid


def _claim_rows(earning: CardEarningSnapshot, device_index: int) -> list[CardEarningRow]:
    """Return the card's rows with time under a whole-card claim, the most claimed first."""
    rows = [row for row in earning.rows if row.device_index == device_index and row.claim_seconds > 0]
    rows.sort(key=lambda row: -row.claim_seconds)
    return rows


def _simple_claim_lines(earning: CardEarningSnapshot, device_index: int) -> list[Text]:
    """Return one sentence per model the worker offered alone on this card."""
    wall = _card_wall(earning, device_index).wall_seconds
    lines: list[Text] = []
    if wall <= 0:
        return lines
    for row in _claim_rows(earning, device_index):
        share = _share_text(row.claim_seconds / wall)
        phrase = verdict_phrase(row.verdict, row=row, min_jobs=earning.verdict_min_jobs)
        lines.append(
            Text(f"Asked the horde only for {row.model} for {share} of the card's time. It is {phrase}."),
        )
    return lines


def render_earning_simple(earning: CardEarningSnapshot, *, now: float) -> RenderableType:
    """Return the plain form: one block per card, then the footer.

    Args:
        earning: The snapshot's earning figures.
        now: The snapshot's wall clock, which the window's span is measured to.
    """
    if _is_empty(earning):
        parts: list[RenderableType] = [_empty_message()]
        if earning.last_restart is not None:
            parts.append(Text(_restart_sentence(earning.last_restart), style="grey62"))
        return Panel(
            Group(*parts),
            title="What each model earns on this card",
            title_align="left",
            border_style="grey37",
            padding=(0, 1),
        )
    cards = _card_indices(earning)
    several_cards = len(cards) > 1
    blocks: list[RenderableType] = [
        _simple_card_block(earning, device_index, several_cards=several_cards) for device_index in cards
    ]
    blocks.append(_footer(earning, now=now))
    return Group(*blocks)


def _rate_text(rate: float | None) -> str:
    """Return a kudos-per-second figure, or a dash when there is none."""
    return "-" if rate is None else f"{rate:.2f}"


def _band_text(low: float | None, high: float | None) -> str:
    """Return a 95% band as "low to high", or a dash when either end is missing."""
    if low is None or high is None:
        return "-"
    return f"{low:.2f} to {high:.2f}"


_DETAIL_LEGEND = (
    "Kudos/s is kudos per second of card time. The rest is the card's other work over the card's remaining time. "
    "A band is the range the true rate lies in with 95% confidence, judged from how much the readings vary. "
    "Readings counts the time slices behind each rate, the model's then the rest's."
)
"""The gloss under the detailed earning table, so the terms read without the docs."""


def _detail_card_block(earning: CardEarningSnapshot, device_index: int, *, several_cards: bool) -> Panel:
    """Return the detailed block for one card: the figures per model, the time split and the claim time."""
    rows = _card_rows(earning, device_index)
    table = Table(expand=True, border_style="grey37", header_style="bold")
    table.add_column("Model", no_wrap=True, max_width=32, overflow="ellipsis")
    table.add_column("Kudos/s", justify="right", no_wrap=True)
    table.add_column("95% band", justify="right", no_wrap=True)
    table.add_column("Rest kudos/s", justify="right", no_wrap=True)
    table.add_column("Rest band", justify="right", no_wrap=True)
    table.add_column("Jobs", justify="right", no_wrap=True)
    table.add_column("Readings", justify="right", no_wrap=True)
    table.add_column("Verdict")
    for row in rows:
        table.add_row(
            row.model,
            _rate_text(row.model_kudos_per_second),
            _band_text(row.model_band_low, row.model_band_high),
            _rate_text(row.rest_kudos_per_second),
            _band_text(row.rest_band_low, row.rest_band_high),
            f"{row.jobs:,}",
            f"{row.model_buckets}/{row.rest_buckets}",
            verdict_phrase(row.verdict, row=row, min_jobs=earning.verdict_min_jobs),
        )
    if not rows:
        table.add_row("no model has card time here yet", "", "", "", "", "", "", "")

    parts: list[RenderableType] = [table, Text(_DETAIL_LEGEND, style="grey62"), Text("")]
    parts.append(_detail_time_table(earning, device_index))
    claim_table = _detail_claim_table(earning, device_index)
    if claim_table is not None:
        parts.extend([Text(""), claim_table])
    return Panel(
        Group(*parts),
        title=_card_title(device_index, several_cards=several_cards),
        title_align="left",
        border_style="grey37",
        padding=(0, 1),
    )


def _detail_time_table(earning: CardEarningSnapshot, device_index: int) -> RenderableType:
    """Return the card's time split with the seconds behind each share."""
    card = _card_wall(earning, device_index)
    heading = Text(f"Where the card's time went, of {human_duration(card.wall_seconds)} on the card", style="bold")
    table = Table(border_style="grey37", header_style="bold")
    table.add_column("Model", no_wrap=True, max_width=32, overflow="ellipsis")
    table.add_column("Share", justify="right", no_wrap=True)
    table.add_column("Card time", justify="right", no_wrap=True)
    shares = _time_shares(earning, device_index)
    for label, fraction, seconds in shares:
        table.add_row(label, _share_text(fraction), human_duration(seconds))
    if not shares:
        table.add_row("no card time measured yet", "", "")
    return Group(heading, table)


def _detail_claim_table(earning: CardEarningSnapshot, device_index: int) -> RenderableType | None:
    """Return the time under whole-card claims per model with its seconds, or None when there was none."""
    rows = _claim_rows(earning, device_index)
    wall = _card_wall(earning, device_index).wall_seconds
    if not rows or wall <= 0:
        return None
    table = Table(border_style="grey37", header_style="bold")
    table.add_column("Model", no_wrap=True, max_width=32, overflow="ellipsis")
    table.add_column("Share", justify="right", no_wrap=True)
    table.add_column("Claim time", justify="right", no_wrap=True)
    table.add_column("Verdict")
    for row in rows:
        table.add_row(
            row.model,
            _share_text(row.claim_seconds / wall),
            human_duration(row.claim_seconds),
            verdict_phrase(row.verdict, row=row, min_jobs=earning.verdict_min_jobs),
        )
    return Group(Text("Time spent offering one model only", style="bold"), table)


def render_earning_detail(earning: CardEarningSnapshot, *, now: float) -> RenderableType:
    """Return the detailed form: one block per card with every figure, then the footer.

    Args:
        earning: The snapshot's earning figures.
        now: The snapshot's wall clock, which the window's span is measured to.
    """
    if _is_empty(earning):
        parts: list[RenderableType] = [_empty_message()]
        if earning.last_restart is not None:
            parts.append(Text(_restart_sentence(earning.last_restart), style="grey62"))
        return Panel(Group(*parts), title="Card earning", title_align="left", border_style="grey37", padding=(0, 1))
    cards = _card_indices(earning)
    several_cards = len(cards) > 1
    blocks: list[RenderableType] = [
        _detail_card_block(earning, device_index, several_cards=several_cards) for device_index in cards
    ]
    blocks.append(_footer(earning, now=now))
    return Group(*blocks)


class CardEarningView(Vertical):
    """Stats tab section that shows the worker's card earning figures in a plain and a detailed form.

    Both forms stay mounted. The screen's level class shows one of them through the stylesheet's
    ``level-simple-only`` and ``level-operator-only`` rules, so a level change does not rebuild anything.
    """

    DEFAULT_CSS = """
    CardEarningView {
        height: auto;
    }
    CardEarningView > Static {
        height: auto;
    }
    """

    def __init__(self) -> None:
        """Initialize the section with nothing shown yet."""
        super().__init__(id="stats-card-earning")
        self._shown: CardEarningSnapshot | None = None

    def compose(self) -> ComposeResult:
        """Lay out the plain form and the detailed form."""
        yield Static(id="card-earning-simple", classes="level-simple-only")
        yield Static(id="card-earning-detail", classes="level-operator-only")

    def update_earning(self, earning: CardEarningSnapshot, *, now: float) -> None:
        """Mutate both forms to show ``earning``, skipping the rebuild when the figures are unchanged.

        The worker refreshes the figures at most every ten seconds while the snapshot arrives about twice a
        second, so most calls change nothing.

        Args:
            earning: The snapshot's earning figures.
            now: The snapshot's wall clock, which the window's span is measured to.
        """
        if self._shown is not None and earning == self._shown:
            return
        self._shown = earning
        self.query_one("#card-earning-simple", Static).update(render_earning_simple(earning, now=now))
        self.query_one("#card-earning-detail", Static).update(render_earning_detail(earning, now=now))
