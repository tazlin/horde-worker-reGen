"""The Control tab: less-common worker commands without global keybinding clutter."""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Static

from horde_worker_regen.process_management.ipc.supervisor_channel import WorkerStateSnapshot
from horde_worker_regen.tui.worker_launcher import SupervisorStatus


class ControlView(VerticalScroll):
    """Operator controls that are important but not worth occupying global keybindings."""

    DEFAULT_CSS = """
    ControlView #control-actions {
        height: auto;
        margin-bottom: 1;
    }
    ControlView #control-actions Button {
        margin: 0 1 1 0;
        min-width: 16;
    }
    ControlView #control-summary {
        height: auto;
    }
    """

    class TogglePauseRequested(Message):
        """Posted when the operator asks to pause or resume local job popping."""

    class ToggleAutoStartRequested(Message):
        """Posted when the operator toggles launch-time auto-start."""

    class StartStopRequested(Message):
        """Posted when the operator asks to start or gracefully stop the worker."""

    class RestartRequested(Message):
        """Posted when the operator asks to restart the worker."""

    class ToggleServerMaintenanceRequested(Message):
        """Posted when the operator toggles horde-side maintenance."""

    def __init__(self) -> None:
        """Initialize cached state used for button labels before the first worker snapshot arrives."""
        super().__init__()
        self._paused = False
        self._running = False
        self._auto_start = False
        self._server_maintenance = False

    def compose(self) -> ComposeResult:
        """Lay out the action row and current control-state summary."""
        with Horizontal(id="control-actions"):
            yield Button("Start worker", id="control-start-stop", variant="primary")
            yield Button("Pause worker", id="control-pause")
            yield Button("Auto-start: off", id="control-autostart")
            yield Button("Restart", id="control-restart", variant="warning")
            yield Button("Horde maintenance", id="control-maintenance")
        yield Static(id="control-summary")

    def update_view(
        self,
        snapshot: WorkerStateSnapshot | None,
        *,
        supervisor_status: SupervisorStatus,
        is_alive: bool,
        restart_attempts: int,
        auto_start: bool,
    ) -> None:
        """Refresh button labels and the state summary from the latest app/worker state."""
        self._running = is_alive and supervisor_status is not SupervisorStatus.STOPPED
        self._paused = bool(snapshot is not None and snapshot.supervisor_paused)
        self._auto_start = auto_start
        self._server_maintenance = bool(snapshot is not None and snapshot.worker_details_maintenance)

        self.query_one("#control-start-stop", Button).label = "Stop worker" if self._running else "Start worker"
        self.query_one("#control-pause", Button).label = "Resume worker" if self._paused else "Pause worker"
        self.query_one("#control-autostart", Button).label = f"Auto-start: {'on' if auto_start else 'off'}"
        self.query_one("#control-maintenance", Button).label = (
            "Horde maintenance: on" if self._server_maintenance else "Horde maintenance: off"
        )
        self.query_one("#control-summary", Static).update(
            self._render_summary(snapshot, supervisor_status, restart_attempts),
        )

    def _render_summary(
        self,
        snapshot: WorkerStateSnapshot | None,
        supervisor_status: SupervisorStatus,
        restart_attempts: int,
    ) -> Panel:
        """Render the current control posture as a compact key/value panel."""
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold cyan", no_wrap=True)
        table.add_column()
        table.add_column(justify="right", style="bold cyan", no_wrap=True)
        table.add_column()

        local_pause = "paused" if self._paused else "serving allowed"
        server_maintenance = "on" if self._server_maintenance else "off"
        auto_start = "on" if self._auto_start else "off"
        worker = snapshot.config.dreamer_name if snapshot is not None else "-"
        table.add_row("Worker", worker, "Supervisor", supervisor_status.value)
        table.add_row("Local pause", local_pause, "Horde maintenance", server_maintenance)
        table.add_row("Auto-start", auto_start, "Restart attempts", str(restart_attempts))

        help_text = Text(
            "Pause only stops new pops; in-flight jobs finish. Horde maintenance asks the API to stop sending work.",
            style="grey62",
        )
        return Panel(Group(table, help_text), title="Control state", title_align="left", border_style="grey37")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Translate button presses into app-level control messages."""
        if event.button.id == "control-pause":
            self.post_message(self.TogglePauseRequested())
        elif event.button.id == "control-autostart":
            self.post_message(self.ToggleAutoStartRequested())
        elif event.button.id == "control-start-stop":
            self.post_message(self.StartStopRequested())
        elif event.button.id == "control-restart":
            self.post_message(self.RestartRequested())
        elif event.button.id == "control-maintenance":
            self.post_message(self.ToggleServerMaintenanceRequested())
