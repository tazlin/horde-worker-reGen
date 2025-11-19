"""Job queue screen for the Horde Worker TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Grid, Horizontal
from textual.screen import Screen
from textual.widgets import Header, Footer, Label, ProgressBar

if TYPE_CHECKING:
    from horde_worker_regen.tui.data_provider import TUIDataProvider


class JobsScreen(Screen):
    """Screen showing job queue information and status."""

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
    JobsScreen {
        background: $surface;
    }

    #jobs-header {
        dock: top;
        height: auto;
        background: $primary;
        color: $text;
        padding: 1 2;
        text-style: bold;
    }

    #jobs-grid {
        grid-size: 1 3;
        grid-gutter: 1;
        margin: 1;
        height: auto;
    }

    .queue-panel {
        background: $panel;
        border: solid $accent;
        padding: 1;
        height: auto;
        min-height: 15;
    }

    .panel-title {
        text-style: bold;
        color: $accent;
        background: $primary-darken-2;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    .queue-stat {
        height: auto;
        padding: 0 2;
        margin: 0 0 1 0;
    }

    .stat-label {
        color: $text-muted;
        width: 25;
    }

    .stat-value {
        color: $text;
        text-style: bold;
    }

    .large-stat {
        text-align: center;
        text-style: bold;
        color: $success;
        width: 100%;
        height: 3;
        content-align: center middle;
        background: $primary-darken-3;
        margin: 1 0;
    }

    ProgressBar {
        margin: 1 2;
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
        """Initialize the jobs screen.

        Args:
            data_provider: Data provider instance
            name: Screen name
            id: Screen ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self.data_provider = data_provider

    def compose(self) -> ComposeResult:
        """Compose the jobs screen layout."""
        yield Header(show_clock=True)
        yield Label("Job Queues - Status & Throughput", id="jobs-header")

        with Grid(id="jobs-grid"):
            # Queue Overview Panel
            with Container(classes="queue-panel"):
                yield Label("Queue Overview", classes="panel-title")
                yield Label("0", id="total-jobs-large", classes="large-stat")
                with Horizontal(classes="queue-stat"):
                    yield Label("Pending Inference:", classes="stat-label")
                    yield Label("0", id="queue-pending", classes="stat-value")
                with Horizontal(classes="queue-stat"):
                    yield Label("In Progress:", classes="stat-label")
                    yield Label("0", id="queue-progress", classes="stat-value")
                with Horizontal(classes="queue-stat"):
                    yield Label("Pending Safety:", classes="stat-label")
                    yield Label("0", id="queue-safety", classes="stat-value")
                with Horizontal(classes="queue-stat"):
                    yield Label("Being Checked:", classes="stat-label")
                    yield Label("0", id="queue-checking", classes="stat-value")
                with Horizontal(classes="queue-stat"):
                    yield Label("Pending Submit:", classes="stat-label")
                    yield Label("0", id="queue-submit", classes="stat-value")

            # Queue Utilization Panel
            with Container(classes="queue-panel"):
                yield Label("Queue Utilization", classes="panel-title")
                yield Label("0%", id="utilization-large", classes="large-stat")
                with Horizontal(classes="queue-stat"):
                    yield Label("Max Queue Size:", classes="stat-label")
                    yield Label("0", id="max-queue-size", classes="stat-value")
                with Horizontal(classes="queue-stat"):
                    yield Label("Current Load:", classes="stat-label")
                    yield Label("0/0", id="current-load", classes="stat-value")
                yield ProgressBar(total=100, id="utilization-bar", show_eta=False)
                with Horizontal(classes="queue-stat"):
                    yield Label("Status:", classes="stat-label")
                    yield Label("Idle", id="queue-status", classes="stat-value")

            # Performance Metrics Panel
            with Container(classes="queue-panel"):
                yield Label("Performance Metrics", classes="panel-title")
                with Horizontal(classes="queue-stat"):
                    yield Label("Jobs Faulted:", classes="stat-label")
                    yield Label("0", id="perf-faulted", classes="stat-value")
                with Horizontal(classes="queue-stat"):
                    yield Label("Job Slowdowns:", classes="stat-label")
                    yield Label("0", id="perf-slowdowns", classes="stat-value")
                with Horizontal(classes="queue-stat"):
                    yield Label("Consecutive Fails:", classes="stat-label")
                    yield Label("0", id="perf-failures", classes="stat-value")
                with Horizontal(classes="queue-stat"):
                    yield Label("Process Recoveries:", classes="stat-label")
                    yield Label("0", id="perf-recoveries", classes="stat-value")

        yield Footer()

    def on_mount(self) -> None:
        """Handle screen mount."""
        self.set_interval(1.0, self.refresh_data)
        self.refresh_data()

    def refresh_data(self) -> None:
        """Refresh job queue data."""
        try:
            jobs = self.data_provider.get_job_queue_info()
            stats = self.data_provider.get_kudos_stats()

            # Update queue overview
            total = jobs.get("total", 0)
            self.query_one("#total-jobs-large", Label).update(f"Total: {total}")
            self.query_one("#queue-pending", Label).update(str(jobs.get("pending_inference", 0)))
            self.query_one("#queue-progress", Label).update(str(jobs.get("in_progress", 0)))
            self.query_one("#queue-safety", Label).update(str(jobs.get("pending_safety_check", 0)))
            self.query_one("#queue-checking", Label).update(str(jobs.get("being_safety_checked", 0)))
            self.query_one("#queue-submit", Label).update(str(jobs.get("pending_submit", 0)))

            # Update utilization
            utilization = jobs.get("utilization", 0)
            max_size = jobs.get("max_queue_size", 1)
            self.query_one("#utilization-large", Label).update(f"{utilization}%")
            self.query_one("#max-queue-size", Label).update(str(max_size))
            self.query_one("#current-load", Label).update(f"{total}/{max_size}")
            self.query_one("#utilization-bar", ProgressBar).update(progress=utilization)

            # Determine queue status
            if total == 0:
                queue_status = "Idle - No Jobs"
            elif utilization > 90:
                queue_status = "Critical - Queue Full"
            elif utilization > 70:
                queue_status = "High Load"
            elif utilization > 30:
                queue_status = "Normal Load"
            else:
                queue_status = "Low Load"
            self.query_one("#queue-status", Label).update(queue_status)

            # Update performance metrics
            self.query_one("#perf-faulted", Label).update(str(stats.get("jobs_faulted", 0)))
            self.query_one("#perf-slowdowns", Label).update(str(stats.get("job_slowdowns", 0)))
            self.query_one("#perf-failures", Label).update(str(stats.get("consecutive_failures", 0)))
            self.query_one("#perf-recoveries", Label).update(str(stats.get("process_recoveries", 0)))

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
