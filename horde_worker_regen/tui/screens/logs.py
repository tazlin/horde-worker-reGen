"""Log viewer screen for the Horde Worker TUI."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Header, Footer, Label, RichLog

if TYPE_CHECKING:
    from horde_worker_regen.tui.data_provider import TUIDataProvider


class LogsScreen(Screen):
    """Screen showing real-time log messages."""

    BINDINGS = [
        ("1", "switch_screen('dashboard')", "Dashboard"),
        ("2", "switch_screen('processes')", "Processes"),
        ("3", "switch_screen('jobs')", "Jobs"),
        ("4", "switch_screen('stats')", "Stats"),
        ("5", "switch_screen('config')", "Config"),
        ("6", "switch_screen('logs')", "Logs"),
        ("q", "quit_app", "Quit"),
        ("r", "refresh", "Refresh"),
        ("c", "clear_logs", "Clear"),
    ]

    CSS = """
    LogsScreen {
        background: $surface;
    }

    #logs-header {
        dock: top;
        height: auto;
        background: $primary;
        color: $text;
        padding: 1 2;
        text-style: bold;
    }

    #logs-container {
        height: 1fr;
        margin: 1;
        background: $panel;
        border: solid $accent;
        padding: 1;
    }

    RichLog {
        background: $surface-darken-1;
        border: solid $primary;
    }
    """

    def __init__(
        self,
        data_provider: TUIDataProvider,
        *,
        name: str | None = None,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
    ) -> None:
        """Initialize the logs screen.

        Args:
            data_provider: Data provider instance
            name: Screen name
            id: Screen ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self.data_provider = data_provider
        self._last_log_count = 0

    def compose(self) -> ComposeResult:
        """Compose the logs screen layout."""
        yield Header(show_clock=True)
        yield Label("Live Logs - Real-time Worker Activity", id="logs-header")
        with VerticalScroll(id="logs-container"):
            yield RichLog(highlight=True, markup=True, auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        """Handle screen mount."""
        self.set_interval(0.5, self.refresh_logs)
        self.refresh_logs()

    def refresh_logs(self) -> None:
        """Refresh log display."""
        try:
            logs = self.data_provider.get_logs(limit=500)
            rich_log = self.query_one(RichLog)

            # Only update if new logs are available
            if len(logs) > self._last_log_count:
                new_logs = logs[self._last_log_count :]
                for timestamp, level, message in new_logs:
                    # Format timestamp
                    time_str = timestamp.strftime("%H:%M:%S")

                    # Color code by level
                    if level == "ERROR":
                        color = "red"
                    elif level == "WARNING":
                        color = "yellow"
                    elif level == "INFO":
                        color = "green"
                    elif level == "DEBUG":
                        color = "blue"
                    else:
                        color = "white"

                    # Add to log
                    rich_log.write(f"[dim]{time_str}[/dim] [{color}]{level:8}[/{color}] {message}")

                self._last_log_count = len(logs)

        except Exception as e:
            pass

    def action_switch_screen(self, screen_name: str) -> None:
        """Switch to a different screen.

        Args:
            screen_name: Name of the screen to switch to
        """
        self.app.switch_screen(screen_name)

    def action_quit_app(self) -> None:
        """Quit the application."""
        self.app.exit()

    def action_refresh(self) -> None:
        """Manually refresh the logs."""
        self.refresh_logs()

    def action_clear_logs(self) -> None:
        """Clear the log display."""
        rich_log = self.query_one(RichLog)
        rich_log.clear()
        self._last_log_count = 0
