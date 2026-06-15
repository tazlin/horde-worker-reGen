"""The logs view: tail the main bridge log or any subprocess log, with level and text filters."""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, RichLog, Select

from horde_worker_regen.tui.log_tailer import LogFollower, discover_bridge_logs

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

    DEFAULT_CSS = """
    LogsView #log-controls {
        height: 3;
        padding: 0 1;
    }
    LogsView #log-controls Select {
        width: 32;
        margin-right: 1;
    }
    LogsView #log-search {
        width: 1fr;
    }
    LogsView #log-output {
        border: round $foreground 20%;
    }
    """

    def __init__(self) -> None:
        """Initialize follower and filter state."""
        super().__init__()
        self._follower: LogFollower | None = None
        self._current_path: Path | None = None
        self._known_files: list[Path] = []
        self._min_rank = 0
        self._search = ""

    def compose(self) -> ComposeResult:
        """Lay out the file/level/search controls and the log output."""
        with Horizontal(id="log-controls"):
            yield Select((), prompt="No logs yet", id="log-file", allow_blank=True)
            yield Select(
                ((level, level) for level in _LEVEL_CHOICES),
                value="ALL",
                allow_blank=False,
                id="log-level",
            )
            yield Input(placeholder="filter text…", id="log-search")
        yield RichLog(id="log-output", highlight=False, markup=False, wrap=False, max_lines=5000)

    def on_mount(self) -> None:
        """Discover log files and begin polling."""
        self._refresh_file_options(select_first=True)
        self.set_interval(0.5, self._poll)

    def _refresh_file_options(self, *, select_first: bool) -> None:
        """Re-scan the logs directory and update the file selector if the set changed."""
        files = discover_bridge_logs()
        if files == self._known_files:
            return
        self._known_files = files
        select = self.query_one("#log-file", Select)
        select.set_options((self._label_for(path), str(path)) for path in files)
        if select_first and files and self._current_path is None:
            select.value = str(files[0])
            self._switch_file(files[0])

    @staticmethod
    def _label_for(path: Path) -> str:
        """A friendly selector label for a bridge log path."""
        if path.name == "bridge.log":
            return "bridge (main)"
        return path.stem.replace("bridge_", "subprocess ")

    def _switch_file(self, path: Path) -> None:
        """Begin following a new file, clearing the view and re-priming the tail."""
        self._current_path = path
        self._follower = LogFollower(path)
        log = self.query_one("#log-output", RichLog)
        log.clear()
        log.write(Text(f"── following {path} ──", style="italic grey62"))

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle file/level selection changes."""
        if event.select.id == "log-file":
            if event.value is Select.BLANK:
                return
            path = Path(str(event.value))
            if path != self._current_path:
                self._switch_file(path)
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

    def _poll(self) -> None:
        """Append any new (filtered) lines and keep the file list current."""
        self._refresh_file_options(select_first=True)
        if self._follower is None:
            return
        new_lines = self._follower.poll()
        if not new_lines:
            return
        log = self.query_one("#log-output", RichLog)
        for line in new_lines:
            styled = self._style_line(line)
            if styled is not None:
                log.write(styled)

    def _style_line(self, line: str) -> Text | None:
        """Apply level and search filters, returning a styled line or None to drop it."""
        level = self._parse_level(line)
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
