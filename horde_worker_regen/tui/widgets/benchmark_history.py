"""A modal for browsing and comparing past benchmark runs.

Every ramp leaves a ``report.json`` on disk, but the live Benchmark tab only ever shows the run in
progress (or the last one recorded in app state). This modal turns that pile of reports into something
navigable: a list of past runs, a full rendered report for any one of them, and a diff of a run against
the one before it so an operator can see at a glance whether a run regressed.

It owns no worker and touches only the filesystem, so it is pushed straight from the Benchmark view
(no app coordination) and the heavy report/history imports are deferred to first use, keeping them off
the TUI's hot path.
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import RenderableType
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Static

from horde_worker_regen.benchmark.capabilities.report_render import render_markdown
from horde_worker_regen.benchmark.history import (
    ReportComparison,
    RunSummary,
    compare_reports,
    list_runs,
    load_report,
)

_RESULTS_ROOT = Path("benchmark_results")


class BenchmarkHistoryModal(ModalScreen[None]):
    """Browse past benchmark runs: view any run's report, or diff it against the previous run."""

    DEFAULT_CSS = """
    BenchmarkHistoryModal {
        align: center middle;
    }
    BenchmarkHistoryModal #history-dialog {
        width: 90%;
        height: 90%;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    BenchmarkHistoryModal #history-table {
        height: 11;
        margin-bottom: 1;
    }
    BenchmarkHistoryModal #history-actions {
        height: 3;
    }
    BenchmarkHistoryModal #history-actions Button {
        margin-right: 1;
    }
    BenchmarkHistoryModal #history-detail-scroll {
        height: 1fr;
        border: round $panel;
        padding: 0 1;
    }
    """

    BINDINGS = [("escape", "dismiss_modal", "Close")]

    def __init__(self, *, results_root: Path = _RESULTS_ROOT) -> None:
        """Store where to look for run directories (overridable for tests)."""
        super().__init__()
        self._results_root = results_root
        self._summaries: list[RunSummary] = []
        self._current_run_id: str | None = None
        self._last_detail: RenderableType | None = None
        """The renderable last shown in the detail pane (retained so tests can inspect it)."""

    def compose(self) -> ComposeResult:
        """Lay out the run table, the action buttons, and the scrollable detail pane."""
        with Vertical(id="history-dialog"):
            yield Static(Text("Past benchmark runs", style="bold"), id="history-title")
            table: DataTable[str] = DataTable(id="history-table", cursor_type="row", zebra_stripes=True)
            yield table
            with Horizontal(id="history-actions"):
                yield Button("View report", id="history-view", variant="primary")
                yield Button("Compare with previous", id="history-compare", variant="default")
                yield Button("Close", id="history-close", variant="warning")
            with VerticalScroll(id="history-detail-scroll"):
                yield Static(id="history-detail")

    def on_mount(self) -> None:
        """Load the run list and render the initial selection (or an empty state)."""
        from horde_worker_regen.app_state import AppStateStore

        self._summaries = list_runs(self._results_root)
        last_benchmark = AppStateStore().load().last_benchmark
        self._current_run_id = last_benchmark.run_id if last_benchmark is not None else None

        table = self.query_one("#history-table", DataTable)
        table.add_columns("Date", "GPU", "Passed", "Findings", "Run")
        for summary in self._summaries:
            current = "  *current*" if summary.run_id == self._current_run_id else ""
            table.add_row(
                _format_when(summary.created_at),
                summary.gpu_name or "-",
                f"{summary.levels_passed}/{summary.levels_total}",
                str(summary.num_findings),
                f"{summary.run_id}{current}",
            )

        if not self._summaries:
            self._set_detail(Text("No past benchmark runs found under benchmark_results/.", style="yellow"))
        else:
            self._set_detail(
                Text("Select a run, then View report or Compare with previous.", style="grey70"),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route the action buttons to view, compare, or close."""
        if event.button.id == "history-view":
            self._view_selected()
        elif event.button.id == "history-compare":
            self._compare_selected()
        elif event.button.id == "history-close":
            self.dismiss(None)

    def action_dismiss_modal(self) -> None:
        """Close the modal (Escape)."""
        self.dismiss(None)

    def _selected_index(self) -> int | None:
        """The highlighted run's index, or None when the list is empty."""
        if not self._summaries:
            return None
        row = self.query_one("#history-table", DataTable).cursor_row
        if row is None or row < 0 or row >= len(self._summaries):
            return None
        return row

    def _view_selected(self) -> None:
        """Render the highlighted run's full markdown report into the detail pane."""
        index = self._selected_index()
        if index is None:
            return
        summary = self._summaries[index]
        report = load_report(Path(summary.run_dir))
        if report is None:
            self._set_detail(Text(f"Could not load report for {summary.run_id}.", style="red"))
            return
        self._set_detail(Markdown(render_markdown(report)))

    def _compare_selected(self) -> None:
        """Diff the highlighted run against the next-older run and render the differences."""
        index = self._selected_index()
        if index is None:
            return
        if index + 1 >= len(self._summaries):
            self._set_detail(Text("No earlier run to compare against (this is the oldest run).", style="yellow"))
            return
        newer = load_report(Path(self._summaries[index].run_dir))
        older = load_report(Path(self._summaries[index + 1].run_dir))
        if newer is None or older is None:
            self._set_detail(Text("Could not load one of the runs to compare.", style="red"))
            return
        self._set_detail(_render_comparison(compare_reports(older, newer)))

    def _set_detail(self, renderable: RenderableType) -> None:
        """Replace the detail pane's contents (and retain them for inspection)."""
        self._last_detail = renderable
        self.query_one("#history-detail", Static).update(renderable)


def _format_when(created_at: float) -> str:
    """Render a run's creation time as a local ``YYYY-MM-DD HH:MM`` stamp (or ``-`` when unset)."""
    if not created_at:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(created_at))


def _render_comparison(comparison: ReportComparison) -> Table:
    """Render a :class:`ReportComparison` as a grouped diff table (older -> newer)."""
    table = Table(title=f"{comparison.older_run_id}  ->  {comparison.newer_run_id}", expand=True)
    table.add_column("Group")
    table.add_column("Field")
    table.add_column("Was")
    table.add_column("Now")

    if not comparison.has_changes:
        table.add_row("-", "no differences", "-", "-")
        return table

    for change in comparison.outcome_changes:
        table.add_row("outcome", change.level_id, change.old, _outcome_now(change.old, change.new))
    for change in comparison.capability_changes:
        table.add_row("capability", change.field, change.old, change.new)
    for change in comparison.suggested_changes:
        table.add_row("suggested", change.field, change.old, change.new)
    for change in comparison.baseline_its_changes:
        table.add_row("baseline it/s", change.field, change.old, change.new)

    delta = comparison.findings_delta
    if delta:
        style = "red" if delta > 0 else "green"
        table.add_row(
            "findings",
            "robustness findings",
            str(comparison.older_findings),
            Text(f"{comparison.newer_findings} ({delta:+d})", style=style),
        )
    return table


def _outcome_now(old: str, new: str) -> Text:
    """Colour an outcome's new value: green when it recovered to passing, red when it regressed."""
    if new == "passed" and old != "passed":
        return Text(new, style="green")
    if old == "passed" and new != "passed":
        return Text(new, style="red")
    return Text(new)


__all__ = ["BenchmarkHistoryModal"]
