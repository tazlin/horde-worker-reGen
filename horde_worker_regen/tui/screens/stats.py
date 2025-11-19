"""Statistics and kudos screen for the Horde Worker TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Grid, Horizontal
from textual.screen import Screen
from textual.widgets import Header, Footer, Label, Digits

if TYPE_CHECKING:
    from horde_worker_regen.tui.data_provider import TUIDataProvider


class StatsScreen(Screen):
    """Screen showing detailed kudos and performance statistics."""

    BINDINGS = [
        ("1", "switch_screen('dashboard')", "Dashboard"),
        ("2", "switch_screen('processes')", "Processes"),
        ("3", "switch_screen('jobs')", "Jobs"),
        ("4", "switch_screen('stats')", "Stats"),
        ("5", "switch_screen('config')", "Config"),
        ("6", "switch_screen('logs')", "Logs"),
        ("q", "quit_app", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    CSS = """
    StatsScreen {
        background: $surface;
    }

    #stats-header {
        dock: top;
        height: auto;
        background: $primary;
        color: $text;
        padding: 1 2;
        text-style: bold;
    }

    #stats-grid {
        grid-size: 1 2;
        grid-gutter: 1;
        margin: 1;
        height: auto;
    }

    .stats-panel {
        background: $panel;
        border: solid $accent;
        padding: 1;
        height: auto;
        min-height: 20;
    }

    .panel-title {
        text-style: bold;
        color: $accent;
        background: $primary-darken-2;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    .stat-row {
        height: auto;
        padding: 0 2;
        margin: 0 0 1 0;
    }

    .stat-label {
        color: $text-muted;
        width: 30;
    }

    .stat-value {
        color: $text;
        text-style: bold;
    }

    .highlight-stat {
        text-align: center;
        color: $success;
        width: 100%;
        height: 5;
        content-align: center middle;
        background: $primary-darken-3;
        margin: 1 0;
        text-style: bold;
    }

    Digits {
        width: 100%;
        text-align: center;
        margin: 1 0;
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
        """Initialize the stats screen.

        Args:
            data_provider: Data provider instance
            name: Screen name
            id: Screen ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self.data_provider = data_provider

    def compose(self) -> ComposeResult:
        """Compose the stats screen layout."""
        yield Header(show_clock=True)
        yield Label("Statistics & Kudos - Performance Tracking", id="stats-header")

        with Grid(id="stats-grid"):
            # Kudos Panel
            with Container(classes="stats-panel"):
                yield Label("Kudos Statistics", classes="panel-title")
                yield Digits("0.00", id="kudos-display")
                with Horizontal(classes="stat-row"):
                    yield Label("Session Kudos Earned:", classes="stat-label")
                    yield Label("0.00", id="stat-session-kudos", classes="stat-value")
                with Horizontal(classes="stat-row"):
                    yield Label("Kudos Per Hour:", classes="stat-label")
                    yield Label("0.00", id="stat-kudos-hour", classes="stat-value")
                with Horizontal(classes="stat-row"):
                    yield Label("Total User Kudos:", classes="stat-label")
                    yield Label("0.00", id="stat-user-kudos", classes="stat-value")
                with Horizontal(classes="stat-row"):
                    yield Label("Username:", classes="stat-label")
                    yield Label("Unknown", id="stat-username", classes="stat-value")
                yield Label("Session Performance", classes="panel-title")
                with Horizontal(classes="stat-row"):
                    yield Label("Session Uptime:", classes="stat-label")
                    yield Label("00:00:00", id="stat-uptime", classes="stat-value")
                with Horizontal(classes="stat-row"):
                    yield Label("Uptime Percentage:", classes="stat-label")
                    yield Label("0%", id="stat-uptime-pct", classes="stat-value")
                with Horizontal(classes="stat-row"):
                    yield Label("Total Idle Time:", classes="stat-label")
                    yield Label("00:00:00", id="stat-idle", classes="stat-value")

            # Performance Panel
            with Container(classes="stats-panel"):
                yield Label("Performance Metrics", classes="panel-title")
                with Horizontal(classes="stat-row"):
                    yield Label("Jobs Faulted:", classes="stat-label")
                    yield Label("0", id="perf-faulted", classes="stat-value")
                with Horizontal(classes="stat-row"):
                    yield Label("Job Slowdowns:", classes="stat-label")
                    yield Label("0", id="perf-slowdowns", classes="stat-value")
                with Horizontal(classes="stat-row"):
                    yield Label("Consecutive Failures:", classes="stat-label")
                    yield Label("0", id="perf-failures", classes="stat-value")
                with Horizontal(classes="stat-row"):
                    yield Label("Process Recoveries:", classes="stat-label")
                    yield Label("0", id="perf-recoveries", classes="stat-value")
                yield Label("System Health", classes="panel-title")
                with Horizontal(classes="stat-row"):
                    yield Label("Worker Status:", classes="stat-label")
                    yield Label("Unknown", id="health-status", classes="stat-value")
                with Horizontal(classes="stat-row"):
                    yield Label("Active Processes:", classes="stat-label")
                    yield Label("0", id="health-processes", classes="stat-value")
                with Horizontal(classes="stat-row"):
                    yield Label("Queue Utilization:", classes="stat-label")
                    yield Label("0%", id="health-queue", classes="stat-value")
                with Horizontal(classes="stat-row"):
                    yield Label("Session Start:", classes="stat-label")
                    yield Label("Unknown", id="health-session-start", classes="stat-value")

        yield Footer()

    def on_mount(self) -> None:
        """Handle screen mount."""
        self.set_interval(1.0, self.refresh_data)
        self.refresh_data()

    def refresh_data(self) -> None:
        """Refresh statistics data."""
        try:
            stats = self.data_provider.get_kudos_stats()
            worker = self.data_provider.get_worker_status()
            jobs = self.data_provider.get_job_queue_info()
            processes = self.data_provider.get_process_info()

            # Update kudos display
            session_kudos = stats.get("session_kudos", 0)
            self.query_one("#kudos-display", Digits).update(f"{session_kudos:,.2f}")

            # Update kudos stats
            self.query_one("#stat-session-kudos", Label).update(f"{session_kudos:,.2f}")
            self.query_one("#stat-kudos-hour", Label).update(f"{stats.get('kudos_per_hour', 0):,.2f}")
            self.query_one("#stat-user-kudos", Label).update(f"{stats.get('user_kudos', 0):,.2f}")
            self.query_one("#stat-username", Label).update(stats.get("user_name", "Unknown"))

            # Update session performance
            self.query_one("#stat-uptime", Label).update(worker.get("uptime", "00:00:00"))
            self.query_one("#stat-uptime-pct", Label).update(f"{worker.get('uptime_percentage', 0)}%")
            self.query_one("#stat-idle", Label).update(stats.get("idle_time", "00:00:00"))

            # Update performance metrics
            self.query_one("#perf-faulted", Label).update(str(stats.get("jobs_faulted", 0)))
            self.query_one("#perf-slowdowns", Label).update(str(stats.get("job_slowdowns", 0)))
            self.query_one("#perf-failures", Label).update(str(stats.get("consecutive_failures", 0)))
            self.query_one("#perf-recoveries", Label).update(str(stats.get("process_recoveries", 0)))

            # Update system health
            self.query_one("#health-status", Label).update(worker.get("status", "Unknown"))
            active_processes = sum(
                1 for p in processes if p.get("state") not in ["WAITING_FOR_JOB", "PROCESS_STARTING"]
            )
            self.query_one("#health-processes", Label).update(f"{active_processes}/{len(processes)}")
            self.query_one("#health-queue", Label).update(f"{jobs.get('utilization', 0)}%")
            self.query_one("#health-session-start", Label).update(worker.get("session_start", "Unknown"))

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
        """Manually refresh the data."""
        self.refresh_data()
