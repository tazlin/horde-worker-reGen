"""Main dashboard screen for the Horde Worker TUI."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Header, Footer, Label, Static

from horde_worker_regen.tui.widgets import StatBox

if TYPE_CHECKING:
    from horde_worker_regen.tui.data_provider import TUIDataProvider


class DashboardScreen(Screen):
    """Main dashboard screen showing overall worker status."""

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
    DashboardScreen {
        background: $surface;
    }

    #status-header {
        dock: top;
        height: auto;
        background: $primary;
        color: $text;
        padding: 1 2;
        text-style: bold;
    }

    #main-grid {
        grid-size: 2 2;
        grid-gutter: 1;
        margin: 1;
        height: auto;
    }

    #worker-info {
        background: $panel;
        border: solid $accent;
        padding: 1;
        height: auto;
    }

    #process-info {
        background: $panel;
        border: solid $accent;
        padding: 1;
        height: auto;
    }

    #job-info {
        background: $panel;
        border: solid $accent;
        padding: 1;
        height: auto;
    }

    #kudos-info {
        background: $panel;
        border: solid $accent;
        padding: 1;
        height: auto;
    }

    .panel-title {
        text-style: bold;
        color: $accent;
        background: $primary-darken-2;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    .info-row {
        height: auto;
        padding: 0 1;
    }

    .info-label {
        color: $text-muted;
        width: 20;
    }

    .info-value {
        color: $text;
        text-style: bold;
    }

    #stat-boxes {
        grid-size: 4;
        grid-gutter: 1;
        margin: 1;
        height: 6;
    }

    #alerts-panel {
        background: $panel;
        border: solid $warning;
        padding: 1;
        margin: 1;
        height: auto;
        min-height: 5;
    }

    .alert-message {
        color: $warning;
        padding: 0 1;
    }

    #last-update {
        dock: bottom;
        height: 1;
        background: $primary-darken-2;
        color: $text-muted;
        padding: 0 2;
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
        """Initialize the dashboard screen.

        Args:
            data_provider: Data provider instance
            name: Screen name
            id: Screen ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self.data_provider = data_provider

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        yield Header(show_clock=True)

        yield Label("Horde Worker reGen - Dashboard", id="status-header")

        # Main grid with 4 info panels
        with Grid(id="main-grid"):
            # Worker Info Panel
            with Container(id="worker-info"):
                yield Label("Worker Status", classes="panel-title")
                with Horizontal(classes="info-row"):
                    yield Label("Status:", classes="info-label")
                    yield Label("Starting...", id="worker-status", classes="info-value")
                with Horizontal(classes="info-row"):
                    yield Label("Worker Name:", classes="info-label")
                    yield Label("Loading...", id="worker-name", classes="info-value")
                with Horizontal(classes="info-row"):
                    yield Label("Uptime:", classes="info-label")
                    yield Label("00:00:00", id="worker-uptime", classes="info-value")
                with Horizontal(classes="info-row"):
                    yield Label("Uptime %:", classes="info-label")
                    yield Label("0%", id="worker-uptime-pct", classes="info-value")

            # Process Info Panel
            with Container(id="process-info"):
                yield Label("Process Overview", classes="panel-title")
                with Horizontal(classes="info-row"):
                    yield Label("Active Processes:", classes="info-label")
                    yield Label("0", id="active-processes", classes="info-value")
                with Horizontal(classes="info-row"):
                    yield Label("Total Processes:", classes="info-label")
                    yield Label("0", id="total-processes", classes="info-value")
                with Horizontal(classes="info-row"):
                    yield Label("Safety Process:", classes="info-label")
                    yield Label("Unknown", id="safety-status", classes="info-value")
                with Horizontal(classes="info-row"):
                    yield Label("Recoveries:", classes="info-label")
                    yield Label("0", id="process-recoveries", classes="info-value")

            # Job Queue Info Panel
            with Container(id="job-info"):
                yield Label("Job Queues", classes="panel-title")
                with Horizontal(classes="info-row"):
                    yield Label("Pending:", classes="info-label")
                    yield Label("0", id="jobs-pending", classes="info-value")
                with Horizontal(classes="info-row"):
                    yield Label("In Progress:", classes="info-label")
                    yield Label("0", id="jobs-progress", classes="info-value")
                with Horizontal(classes="info-row"):
                    yield Label("Total Jobs:", classes="info-label")
                    yield Label("0", id="jobs-total", classes="info-value")
                with Horizontal(classes="info-row"):
                    yield Label("Queue Usage:", classes="info-label")
                    yield Label("0%", id="queue-usage", classes="info-value")

            # Kudos Info Panel
            with Container(id="kudos-info"):
                yield Label("Kudos & Performance", classes="panel-title")
                with Horizontal(classes="info-row"):
                    yield Label("Session Kudos:", classes="info-label")
                    yield Label("0.00", id="session-kudos", classes="info-value")
                with Horizontal(classes="info-row"):
                    yield Label("Kudos/Hour:", classes="info-label")
                    yield Label("0.00", id="kudos-hour", classes="info-value")
                with Horizontal(classes="info-row"):
                    yield Label("Total Kudos:", classes="info-label")
                    yield Label("0.00", id="total-kudos", classes="info-value")
                with Horizontal(classes="info-row"):
                    yield Label("Jobs Faulted:", classes="info-label")
                    yield Label("0", id="jobs-faulted", classes="info-value")

        # Stat boxes
        with Grid(id="stat-boxes"):
            yield StatBox("Session Start", "Loading...", id="stat-session-start")
            yield StatBox("Idle Time", "00:00:00", id="stat-idle-time")
            yield StatBox("Job Slowdowns", "0", id="stat-slowdowns")
            yield StatBox("Consecutive Fails", "0", id="stat-failures")

        # Alerts panel
        with Container(id="alerts-panel"):
            yield Label("Alerts & Warnings", classes="panel-title")
            yield Label("No alerts", id="alert-content", classes="alert-message")

        yield Label("Last update: Never", id="last-update")

        yield Footer()

    def on_mount(self) -> None:
        """Handle screen mount."""
        self.set_interval(2.0, self.refresh_data)
        self.refresh_data()

    def refresh_data(self) -> None:
        """Refresh dashboard data."""
        try:
            data = self.data_provider.get_all_data()

            # Update worker status
            worker = data.get("worker_status", {})
            self.query_one("#worker-status", Label).update(worker.get("status", "Unknown"))
            self.query_one("#worker-name", Label).update(worker.get("worker_name", "Unknown"))
            self.query_one("#worker-uptime", Label).update(worker.get("uptime", "00:00:00"))
            self.query_one("#worker-uptime-pct", Label).update(f"{worker.get('uptime_percentage', 0)}%")

            # Update process info
            processes = data.get("processes", [])
            active_count = sum(
                1 for p in processes if p.get("state") not in ["WAITING_FOR_JOB", "PROCESS_STARTING"]
            )
            total_count = len(processes)
            safety_active = any(p.get("is_safety_process") for p in processes)

            self.query_one("#active-processes", Label).update(str(active_count))
            self.query_one("#total-processes", Label).update(str(total_count))
            self.query_one("#safety-status", Label).update("Active" if safety_active else "Inactive")

            # Update job queues
            jobs = data.get("job_queues", {})
            self.query_one("#jobs-pending", Label).update(str(jobs.get("pending_inference", 0)))
            self.query_one("#jobs-progress", Label).update(str(jobs.get("in_progress", 0)))
            self.query_one("#jobs-total", Label).update(str(jobs.get("total", 0)))
            self.query_one("#queue-usage", Label).update(f"{jobs.get('utilization', 0)}%")

            # Update kudos stats
            stats = data.get("kudos_stats", {})
            self.query_one("#session-kudos", Label).update(str(stats.get("session_kudos", 0)))
            self.query_one("#kudos-hour", Label).update(str(stats.get("kudos_per_hour", 0)))
            self.query_one("#total-kudos", Label).update(str(stats.get("user_kudos", 0)))
            self.query_one("#jobs-faulted", Label).update(str(stats.get("jobs_faulted", 0)))
            self.query_one("#process-recoveries", Label).update(str(stats.get("process_recoveries", 0)))

            # Update stat boxes
            self.query_one("#stat-session-start", StatBox).update_stat(
                worker.get("session_start", "Unknown")
            )
            self.query_one("#stat-idle-time", StatBox).update_stat(stats.get("idle_time", "00:00:00"))
            self.query_one("#stat-slowdowns", StatBox).update_stat(str(stats.get("job_slowdowns", 0)))
            self.query_one("#stat-failures", StatBox).update_stat(
                str(stats.get("consecutive_failures", 0))
            )

            # Update alerts
            alerts = []
            if worker.get("shutting_down"):
                alerts.append("Worker is shutting down")
            if worker.get("maintenance_mode"):
                alerts.append("Maintenance mode active")
            if stats.get("consecutive_failures", 0) > 3:
                alerts.append(f"High failure rate: {stats.get('consecutive_failures')} consecutive failures")
            if jobs.get("utilization", 0) > 90:
                alerts.append(f"Queue nearly full: {jobs.get('utilization')}%")

            alert_text = "\n".join(alerts) if alerts else "No alerts"
            self.query_one("#alert-content", Label).update(alert_text)

            # Update last update time
            last_update = data.get("last_update", datetime.now())
            self.query_one("#last-update", Label).update(
                f"Last update: {last_update.strftime('%H:%M:%S')}"
            )

        except Exception as e:
            self.query_one("#alert-content", Label).update(f"Error updating dashboard: {e}")

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
