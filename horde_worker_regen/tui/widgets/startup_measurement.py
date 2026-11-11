"""Startup measurement section: what a GPU process uses before it loads a model, as the worker measured it per card.

Every figure and verdict comes from the worker's
[`StartupContextReading`][horde_worker_regen.process_management.ipc.supervisor_channel.StartupContextReading] list.
This module turns them into words and does not judge a reading itself.

- [`verdict_sentence`][horde_worker_regen.tui.widgets.startup_measurement.verdict_sentence]: the dashboard's one
  wording of a measurement verdict, read by this section and by the Insights tab's finding. The startup log line
  has its own technical wording in ``utils/accelerator_probe.py``.
- [`figure_in_use_sentence`][horde_worker_regen.tui.widgets.startup_measurement.figure_in_use_sentence]: which
  figure the worker uses for a card.
- [`StartupMeasurementView`][horde_worker_regen.tui.widgets.startup_measurement.StartupMeasurementView]: the Stats
  tab section, with a plain form at Simple and a detailed form at Advanced and above.
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from horde_worker_regen.process_management.ipc.supervisor_channel import StartupContextReading, StartupContextRun
from horde_worker_regen.tui.widgets.earning import setting_words
from horde_worker_regen.utils.accelerator_probe import ContextReadingVerdict

__all__ = [
    "MEASUREMENT_GLOSS",
    "OVERRIDE_FIELD",
    "StartupMeasurementView",
    "figure_in_use_sentence",
    "render_measurement_detail",
    "render_measurement_simple",
    "verdict_sentence",
]

OVERRIDE_FIELD = "vram_per_process_overhead_mb"
"""The config field whose value above 0 replaces the measured first-process figure on every card. It is named
through ``setting_words``, so the dashboard shows it as the config file writes it."""

MEASUREMENT_GLOSS = (
    "Before it loads a model, each GPU process uses some of the card's memory. At startup the worker measures "
    "this for a first and a second process, and the two figures should agree."
)
"""The plain explanation shown once above the per-card lines."""

_DETAIL_LEGEND = (
    "First is the memory used by the first GPU process, and further is the memory used by each further process. "
    "Before, after first and after second are the card's used memory before any process, after the first and after "
    "a second. Each figure is the difference between two of them."
)
"""The gloss under the detailed table, so the terms read without the docs."""

_SECTION_TITLE = "Startup memory measurement"


def verdict_sentence(reading: StartupContextReading) -> str:
    """Return the plain sentence for a card's measurement verdict.

    An unknown verdict is quoted as it arrives, so a worker newer than the dashboard still shows something true.

    Args:
        reading: The card's startup measurement.
    """
    if reading.verdict == ContextReadingVerdict.CONSISTENT:
        return "The two figures agreed."
    if reading.verdict == ContextReadingVerdict.INCONSISTENT:
        tries = "try" if reading.probe_runs == 1 else f"of the {reading.probe_runs} tries"
        return f"The two figures did not agree on any {tries}."
    if reading.verdict == ContextReadingVerdict.UNCHECKED:
        return "The worker could not compare the two figures, so the reading was not checked."
    return f'Check result: "{reading.verdict}".'


def figure_in_use_sentence(reading: StartupContextReading) -> str:
    """Return the plain sentence that says which first-process figure the worker uses for a card.

    A config key is in backticks, as a finding shows it. The Stats section is plain text and drops the marks.

    Args:
        reading: The card's startup measurement.
    """
    if reading.first_context_override_mb is not None:
        return f"The worker uses {reading.first_context_override_mb} MB, set in {setting_words(OVERRIDE_FIELD)}."
    if reading.verdict == ContextReadingVerdict.INCONSISTENT:
        return f"The worker uses {reading.chosen.first_context_mb} MB, the middle figure of the tries."
    return f"The worker uses {reading.chosen.first_context_mb} MB."


def _card_prefix(reading: StartupContextReading, *, several_cards: bool) -> str:
    """Return the line's lead, naming the card only on a host with more than one."""
    return f"Card {reading.device_index}: " if several_cards else ""


def render_measurement_simple(readings: list[StartupContextReading]) -> RenderableType:
    """Return the plain form: the gloss, then one line per card.

    Args:
        readings: Each card's startup measurement.
    """
    several_cards = len(readings) > 1
    lines: list[RenderableType] = [Text(MEASUREMENT_GLOSS, style="grey62"), Text("")]
    for reading in readings:
        style = "yellow" if reading.verdict == ContextReadingVerdict.INCONSISTENT else ""
        lines.append(
            Text(
                f"{_card_prefix(reading, several_cards=several_cards)}{verdict_sentence(reading)} "
                f"{figure_in_use_sentence(reading).replace('`', '')}",
                style=style,
            ),
        )
    if any(reading.verdict == ContextReadingVerdict.INCONSISTENT for reading in readings):
        lines.append(Text("The Insights tab says what to do about figures that did not agree.", style="grey62"))
    return Panel(Group(*lines), title=_SECTION_TITLE, title_align="left", border_style="grey37", padding=(0, 1))


def _reading_text(reading_mb: int | None) -> str:
    """Return one raw reading in MB, or "not read" when the probe did not take it."""
    return "not read" if reading_mb is None else f"{reading_mb:,}"


def _check_words(reading: StartupContextReading) -> str:
    """Return the short check result for the detailed table."""
    if reading.verdict == ContextReadingVerdict.CONSISTENT:
        return "agreed"
    if reading.verdict == ContextReadingVerdict.INCONSISTENT:
        return "did not agree"
    if reading.verdict == ContextReadingVerdict.UNCHECKED:
        return "not checked"
    return reading.verdict


def _run_cells(run: StartupContextRun) -> tuple[str, str, str, str, str]:
    """Return a run's two figures and three readings as table cells.

    A further-process figure of 0 without the reading after the second process was not measured, so it does not
    read as a measured 0.
    """
    marginal_measured = run.marginal_mb > 0 or run.after_holder_mb is not None
    return (
        f"{run.first_context_mb:,}",
        f"{run.marginal_mb:,}" if marginal_measured else "not measured",
        _reading_text(run.baseline_mb),
        _reading_text(run.after_context_mb),
        _reading_text(run.after_holder_mb),
    )


def _figure_columns(table: Table) -> None:
    """Mutate ``table`` to add the columns for two figures and three readings, all in MB."""
    for header in ("First MB", "Further MB", "Before MB", "After first MB", "After second MB"):
        table.add_column(header, justify="right", no_wrap=True)


def _runs_table(reading: StartupContextReading) -> RenderableType:
    """Return every try's figures for a card whose figures never agreed, marking the try in use."""
    table = Table(border_style="grey37", header_style="bold")
    table.add_column("Try", justify="right", no_wrap=True)
    _figure_columns(table)
    table.add_column("In use", no_wrap=True)
    for position, run in enumerate(reading.runs, start=1):
        table.add_row(str(position), *_run_cells(run), "yes" if position == reading.chosen_run else "")
    return Group(Text(f"Every try on card {reading.device_index}", style="bold"), table)


def render_measurement_detail(readings: list[StartupContextReading]) -> RenderableType:
    """Return the detailed form: each card's figures and readings, then every try of a card that disagreed.

    Args:
        readings: Each card's startup measurement.
    """
    table = Table(expand=True, border_style="grey37", header_style="bold")
    table.add_column("Card", justify="right", no_wrap=True)
    _figure_columns(table)
    table.add_column("Check", no_wrap=True)
    table.add_column("Try used", justify="right", no_wrap=True)
    for reading in readings:
        table.add_row(
            str(reading.device_index),
            *_run_cells(reading.chosen),
            _check_words(reading),
            f"{reading.chosen_run} of {reading.probe_runs}",
        )
    parts: list[RenderableType] = [table, Text(_DETAIL_LEGEND, style="grey62")]
    for reading in readings:
        if reading.runs:
            parts.extend([Text(""), _runs_table(reading)])
    override_mb = next(
        (reading.first_context_override_mb for reading in readings if reading.first_context_override_mb is not None),
        None,
    )
    if override_mb is not None:
        parts.extend(
            [
                Text(""),
                Text(
                    f"{setting_words(OVERRIDE_FIELD).replace('`', '')} is {override_mb}, so the worker uses "
                    f"{override_mb} MB as the first figure on every card.",
                ),
            ],
        )
    return Panel(Group(*parts), title=_SECTION_TITLE, title_align="left", border_style="grey37", padding=(0, 1))


class StartupMeasurementView(Vertical):
    """Stats tab section that shows the worker's startup memory measurement in a plain and a detailed form.

    Both forms stay mounted. The screen's level class shows one of them through the stylesheet's
    ``level-simple-only`` and ``level-operator-only`` rules, so a level change does not rebuild anything.
    """

    DEFAULT_CSS = """
    StartupMeasurementView {
        height: auto;
    }
    StartupMeasurementView > Static {
        height: auto;
    }
    """

    def __init__(self) -> None:
        """Initialize the section with nothing shown yet."""
        super().__init__(id="stats-startup-measurement")
        self._shown: list[StartupContextReading] | None = None

    def compose(self) -> ComposeResult:
        """Lay out the plain form and the detailed form."""
        yield Static(id="startup-measurement-simple", classes="level-simple-only")
        yield Static(id="startup-measurement-detail", classes="level-operator-only")

    def update_readings(self, readings: list[StartupContextReading]) -> None:
        """Mutate both forms to show ``readings``, skipping the rebuild when they are unchanged.

        The readings are fixed for the session, so after the first call only a change to the override setting
        rebuilds anything.

        Args:
            readings: Each card's startup measurement from the snapshot.
        """
        if self._shown is not None and readings == self._shown:
            return
        self._shown = readings
        self.query_one("#startup-measurement-simple", Static).update(render_measurement_simple(readings))
        self.query_one("#startup-measurement-detail", Static).update(render_measurement_detail(readings))
