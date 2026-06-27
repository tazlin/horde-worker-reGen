"""The logs view: tail the main bridge log or any subprocess log, with level and text filters."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, RichLog, Select, Static

from horde_worker_regen.app_state import OverviewViewMode
from horde_worker_regen.tui.log_tailer import LOG_DIR, BridgeLog, LogFollower, discover_bridge_logs_grouped

_LEVEL_RANK: dict[str, int] = {
    "TRACE": 5,
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

_LEVEL_STYLE: dict[str, str] = {
    "TRACE": "grey50",
    "DEBUG": "grey62",
    "INFO": "white",
    "SUCCESS": "green",
    "WARNING": "yellow",
    "ERROR": "bold red",
    "CRITICAL": "bold white on red",
}

_LEVEL_CHOICES = ["ALL", "DEBUG", "INFO", "WARNING", "ERROR"]


class LogsView(Vertical):
    """Follow a selected bridge log file with level and substring filtering."""

    BINDINGS = [
        Binding("end", "jump_to_latest", "Jump to latest"),
        Binding("ctrl+b", "support_bundle", "Support bundle"),
    ]

    DEFAULT_CSS = """
    LogsView #log-controls {
        height: 3;
        padding: 0 1;
    }
    LogsView #log-process {
        width: 26;
        margin-right: 1;
    }
    LogsView #log-history {
        width: 22;
        margin-right: 1;
    }
    LogsView #log-level {
        width: 14;
        margin-right: 1;
    }
    LogsView #log-search {
        width: 1fr;
    }
    LogsView #log-bundle {
        width: auto;
        margin-left: 1;
    }
    LogsView #log-output {
        border: round $foreground 20%;
    }
    LogsView #log-tally {
        height: 1;
        padding: 0 1;
    }
    LogsView #log-scroll-hint {
        height: 1;
        padding: 0 1;
        color: $warning;
        text-style: bold;
    }
    """

    _LEVEL_TALLY_STYLE: dict[str, str] = {
        "INFO": "white",
        "SUCCESS": "green",
        "DEBUG": "grey62",
        "WARNING": "yellow",
        "ERROR": "bold red",
        "CRITICAL": "bold white on red",
    }
    """The levels summarized in the detailed-view tally, in display order."""

    def __init__(self) -> None:
        """Initialize follower and filter state."""
        super().__init__()
        self._follower: LogFollower | None = None
        self._current_path: Path | None = None
        self._grouped: dict[str, list[BridgeLog]] = {}
        self._process_keys: list[str] = []
        self._history_labels: list[str] = []
        self._process_key: str | None = None
        self._history_label = "current"
        self._updating = False
        """Set while mutating the selects programmatically so their Changed events are ignored."""
        self._min_rank = 0
        self._search = ""
        self._mode = OverviewViewMode.NORMAL
        self._level_counts: dict[str, int] = {}
        """Running per-level line counts for the detailed-view tally (the shape of the run)."""
        self._unseen_below = 0
        """Lines appended while the user was scrolled up; cleared when they return to the bottom."""
        self._poll_in_flight = False
        """Guards against overlapping off-loop reads: a slow read on a large file must not let the
        0.5s interval stack a second concurrent read of the same follower."""

    def compose(self) -> ComposeResult:
        """Lay out the process/history/level/search controls and the log output."""
        with Horizontal(id="log-controls"):
            yield Select((), prompt="waiting for logs…", id="log-process", allow_blank=True)
            yield Select((("current", "current"),), value="current", id="log-history", allow_blank=True)
            yield Select(
                ((level, level) for level in _LEVEL_CHOICES),
                value="ALL",
                allow_blank=False,
                id="log-level",
            )
            yield Input(placeholder="filter text…", id="log-search")
            yield Button("Support bundle", id="log-bundle")
        yield Static(id="log-tally")
        yield RichLog(id="log-output", highlight=False, markup=False, wrap=False, max_lines=5000)
        yield Static(id="log-scroll-hint")

    def on_mount(self) -> None:
        """Discover log files and begin polling."""
        self.query_one("#log-tally", Static).display = False
        self.query_one("#log-scroll-hint", Static).display = False
        self._refresh(select_first=True)
        self.set_interval(0.5, self._poll)

    def _refresh(self, *, select_first: bool) -> None:
        """Re-scan the logs directory, refreshing the process/history selectors only when they change."""
        self._grouped = discover_bridge_logs_grouped()
        keys = list(self._grouped.keys())
        if keys != self._process_keys:
            self._process_keys = keys
            self._rebuild_process_options()

        if self._process_key is None and select_first and keys:
            self._select_process(keys[0], programmatic=True)
        elif self._process_key is not None and self._process_key in self._grouped:
            labels = [entry.history_label for entry in self._grouped[self._process_key]]
            if labels != self._history_labels:
                self._rebuild_history_options(reset=False)
            # The live file is followed across rotations by LogFollower, but pick it up if it only
            # appeared after the process was first selected.
            if self._history_label == "current":
                self._follow(self._process_key, "current")

    def _rebuild_process_options(self) -> None:
        """Repopulate the process selector from the current process keys (one entry per process)."""
        select = self.query_one("#log-process", Select)
        options = [(self._grouped[key][0].process_label, key) for key in self._process_keys]
        self._updating = True
        try:
            select.set_options(options)
            if self._process_key in self._grouped:
                select.value = self._process_key
        finally:
            self._updating = False

    def _rebuild_history_options(self, *, reset: bool) -> None:
        """Repopulate the history selector for the selected process (live file first, then dates)."""
        if self._process_key is None:
            return
        entries = self._grouped.get(self._process_key, [])
        labels = [entry.history_label for entry in entries]
        self._history_labels = labels
        target = "current" if reset else self._history_label
        if target not in labels:
            target = labels[0] if labels else "current"
        self._history_label = target
        select = self.query_one("#log-history", Select)
        self._updating = True
        try:
            select.set_options([(label, label) for label in labels])
            if labels:
                select.value = target
        finally:
            self._updating = False

    def _select_process(self, key: str, *, programmatic: bool) -> None:
        """Switch the active process: reset history to its live file and begin following it."""
        if programmatic:
            select = self.query_one("#log-process", Select)
            self._updating = True
            try:
                select.value = key
            finally:
                self._updating = False
        self._process_key = key
        self._rebuild_history_options(reset=True)
        self._follow(key, self._history_label)

    def _entry_for(self, key: str, history_label: str) -> BridgeLog | None:
        """Resolve a (process, history) pair to a log file, falling back to the newest available."""
        entries = self._grouped.get(key, [])
        for entry in entries:
            if entry.history_label == history_label:
                return entry
        return entries[0] if entries else None

    def _follow(self, key: str, history_label: str) -> None:
        """Begin following the file for a (process, history) pair if it is not already current."""
        entry = self._entry_for(key, history_label)
        if entry is not None and entry.path != self._current_path:
            self._switch_file(entry.path)

    def _switch_file(self, path: Path) -> None:
        """Begin following a new file, clearing the view and re-priming the tail."""
        self._current_path = path
        self._follower = LogFollower(path)
        self._level_counts = {}
        self._unseen_below = 0
        log = self.query_one("#log-output", RichLog)
        log.auto_scroll = True
        log.clear()
        log.write(Text(f"── following {path} ──", style="italic grey62"))
        with contextlib.suppress(Exception):
            self._render_scroll_hint()

    def set_view_mode(self, mode: OverviewViewMode) -> None:
        """Apply the shared F6 density contract to the log view.

        Thin is the bare stream (the filter controls are hidden so the whole pane is log). Normal keeps
        the full filter set. Detailed adds a per-level tally above the stream so the shape of the run is
        legible before reading a line; the controls stay visible (detailed never shows less than normal).
        """
        if mode is self._mode:
            return
        self._mode = mode
        self.query_one("#log-controls", Horizontal).display = mode is not OverviewViewMode.THIN
        self.query_one("#log-tally", Static).display = mode is OverviewViewMode.DETAILS
        if mode is OverviewViewMode.DETAILS:
            self._render_tally()

    def _render_tally(self) -> None:
        """Refresh the per-level line-count tally shown in the detailed view."""
        tally = self.query_one("#log-tally", Static)
        if not self._level_counts:
            tally.update(Text("no lines yet", style="grey50"))
            return
        text = Text()
        first = True
        for level, style in self._LEVEL_TALLY_STYLE.items():
            count = self._level_counts.get(level, 0)
            if not count:
                continue
            if not first:
                text.append("   ·   ", style="grey37")
            first = False
            text.append(f"{count:,}", style=f"bold {style}")
            text.append(f" {level}", style="grey50")
        tally.update(text if text.plain else Text("no levelled lines yet", style="grey50"))

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle process/history/level selection changes (ignoring programmatic updates)."""
        if self._updating:
            return
        if event.select.id == "log-process":
            if event.value is not Select.BLANK and str(event.value) != self._process_key:
                self._select_process(str(event.value), programmatic=False)
        elif event.select.id == "log-history":
            if event.value is not Select.BLANK and str(event.value) != self._history_label:
                self._history_label = str(event.value)
                if self._process_key is not None:
                    self._follow(self._process_key, self._history_label)
        elif event.select.id == "log-level":
            self._min_rank = _LEVEL_RANK.get(str(event.value), 0) if event.value != "ALL" else 0
            self._reprime()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update the substring filter and re-prime so it applies to the visible tail."""
        if event.input.id == "log-search":
            self._search = event.value.strip().lower()
            self._reprime()

    def _reprime(self) -> None:
        """Re-read the current file's tail so filter changes apply to existing lines."""
        if self._current_path is not None:
            self._switch_file(self._current_path)

    async def _poll(self) -> None:
        """Append any new (filtered) lines and keep the file list current.

        The follower's file read can block on a large log, so it runs off the event loop; an
        in-flight guard prevents the recurring interval from stacking overlapping reads, and lines
        from a follower that was swapped out mid-read (a file switch) are discarded as stale.
        """
        self._refresh(select_first=True)
        follower = self._follower
        if follower is None or self._poll_in_flight:
            return
        self._poll_in_flight = True
        try:
            new_lines = await asyncio.to_thread(follower.poll)
        finally:
            self._poll_in_flight = False
        if not new_lines or follower is not self._follower:
            return
        log = self.query_one("#log-output", RichLog)
        # Follow the tail only while the user is already at the bottom. When they have scrolled up to
        # read, pause auto-scroll so new lines never yank the viewport, and tally what they have not
        # seen so a hint can offer a one-key jump back to the latest.
        at_bottom = log.is_vertical_scroll_end
        log.auto_scroll = at_bottom
        written = 0
        for line in new_lines:
            styled = self._style_line(line)
            if styled is not None:
                log.write(styled)
                written += 1
        # Tally what the user has not seen while scrolled up; clear it once they are back at the bottom
        # (whether they scrolled there by hand or jumped with End).
        if at_bottom:
            self._unseen_below = 0
        else:
            self._unseen_below += written
        self._render_scroll_hint()
        if self._mode is OverviewViewMode.DETAILS:
            self._render_tally()

    def _render_scroll_hint(self) -> None:
        """Show or hide the 'new lines below' hint based on whether the user has scrolled up."""
        hint = self.query_one("#log-scroll-hint", Static)
        if self._unseen_below > 0:
            hint.update(Text(f"↓ {self._unseen_below:,} new line(s) below; press End to jump to latest"))
            hint.display = True
        else:
            hint.display = False

    def action_jump_to_latest(self) -> None:
        """Scroll to the newest line and resume following the tail."""
        log = self.query_one("#log-output", RichLog)
        log.auto_scroll = True
        log.scroll_end(animate=False)
        self._unseen_below = 0
        self._render_scroll_hint()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dispatch the controls-row buttons."""
        if event.button.id == "log-bundle":
            self.action_support_bundle()

    def action_support_bundle(self) -> None:
        """Generate a redacted support bundle for a maintainer, off the UI thread."""
        button = self.query_one("#log-bundle", Button)
        if button.disabled:
            return
        button.disabled = True
        self.notify("Generating support bundle… this can take ~15s.", title="Support bundle")
        self.run_worker(self._build_support_bundle, thread=True, exclusive=True, group="support-bundle")

    def _build_support_bundle(self) -> None:
        """Build the bundle on a worker thread, then notify the result on the UI thread.

        Imported lazily so the (torch-free but heavier) analysis package is only pulled in when an
        operator actually asks for a bundle, not on every TUI start.
        """
        from horde_worker_regen.analysis.support_bundle import build_support_bundle

        out = Path(f"horde_support_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
        try:
            result = build_support_bundle(LOG_DIR, out)
        except Exception as error:  # noqa: BLE001 - report any failure to the operator, never crash the TUI
            self.app.call_from_thread(self._notify_bundle_error, error)
            return
        self.app.call_from_thread(self._notify_bundle_done, result.out_path, result.redaction_count)

    def _notify_bundle_done(self, out_path: Path, redaction_count: int) -> None:
        """UI-thread callback: announce the written bundle and re-enable the button."""
        self.query_one("#log-bundle", Button).disabled = False
        self.notify(
            f"Wrote {out_path.name}. Redacted {redaction_count} secret/identifier occurrence(s); "
            "skim it before sending.",
            title="Support bundle",
            timeout=12,
        )

    def _notify_bundle_error(self, error: Exception) -> None:
        """UI-thread callback: report a bundle failure and re-enable the button."""
        self.query_one("#log-bundle", Button).disabled = False
        self.notify(
            f"Support bundle failed: {type(error).__name__}: {error}",
            title="Support bundle",
            severity="error",
        )

    def _style_line(self, line: str) -> Text | None:
        """Apply level and search filters, returning a styled line or None to drop it.

        Every parsed level is tallied (before the filters are applied) so the detailed-view level tally
        reflects the whole run's shape, not just the lines the current filter happens to show.
        """
        level = self._parse_level(line)
        if level is not None:
            self._level_counts[level] = self._level_counts.get(level, 0) + 1
        if self._min_rank and level is not None and _LEVEL_RANK.get(level, 0) < self._min_rank:
            return None
        if self._search and self._search not in line.lower():
            return None
        style = _LEVEL_STYLE.get(level or "", "white")
        return Text(line, style=style)

    @staticmethod
    def _parse_level(line: str) -> str | None:
        """Extract the loguru level token from a formatted line, if present."""
        parts = line.split(" | ", 2)
        if len(parts) >= 2:
            candidate = parts[1].strip()
            if candidate in _LEVEL_RANK:
                return candidate
        return None
