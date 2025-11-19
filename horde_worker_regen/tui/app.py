"""Main Textual TUI application for horde-worker-reGen."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Footer,
    Header,
    Label,
    RichLog,
    Static,
)

if TYPE_CHECKING:
    from horde_sdk.ai_horde_api.apimodels import UserDetailsResponse

    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.process_management.process_manager import (
        APIWorkerMessage,
        HordeWorkerProcessManager,
        TorchDeviceMap,
    )

import horde_worker_regen


class WorkerHeader(Static):
    """Header showing worker basic information."""

    def __init__(self, bridge_data: reGenBridgeData, version: str) -> None:
        """Initialize the header."""
        super().__init__()
        self.bridge_data = bridge_data
        self.version = version
        self.uptime_start = time.time()

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static(self._get_header_text(), id="header-text")

    def _get_header_text(self) -> str:
        """Get header text."""
        uptime = timedelta(seconds=int(time.time() - self.uptime_start))
        return (
            f"[bold cyan]horde-worker-reGen v{self.version}[/bold cyan]\n"
            f"Worker: [bold]{self.bridge_data.dreamer_name}[/bold] | "
            f"Uptime: {uptime}"
        )

    def update_display(self, user_info: UserDetailsResponse | None) -> None:
        """Update the header display."""
        uptime = timedelta(seconds=int(time.time() - self.uptime_start))
        kudos_info = ""
        if user_info:
            kudos_info = f" | Kudos: {user_info.kudos:,.0f}"

        text = (
            f"[bold cyan]horde-worker-reGen v{self.version}[/bold cyan]\n"
            f"Worker: [bold]{self.bridge_data.dreamer_name}[/bold] | "
            f"Uptime: {uptime}{kudos_info}"
        )
        self.query_one("#header-text", Static).update(text)


class ConfigPanel(Static):
    """Panel showing worker configuration."""

    def __init__(self, bridge_data: reGenBridgeData) -> None:
        """Initialize the config panel."""
        super().__init__()
        self.bridge_data = bridge_data

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        # Calculate pixel dimensions from max_power
        # Formula: 64 * 64 * 8 * max_power
        pixels = 64 * 64 * 8 * self.bridge_data.max_power
        dimension = int(pixels ** 0.5)

        performance_mode = "Normal"
        if self.bridge_data.high_performance_mode:
            performance_mode = "High"
        elif self.bridge_data.moderate_performance_mode:
            performance_mode = "Moderate"

        config_text = (
            f"[bold]Configuration[/bold]\n"
            f"Max Threads: {self.bridge_data.max_threads} | "
            f"Queue Size: {self.bridge_data.queue_size} | "
            f"Max Batch: {self.bridge_data.max_batch}\n"
            f"Max Power: {self.bridge_data.max_power} ({dimension}x{dimension}px) | "
            f"Performance: {performance_mode} | "
            f"Safety on GPU: {'Yes' if self.bridge_data.safety_on_gpu else 'No'}"
        )
        yield Static(config_text, id="config-text")


class ProcessStatusPanel(Static):
    """Panel showing process status information."""

    process_info = reactive("")

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("[bold]Process Status[/bold]", id="process-status-title")
        yield Static(self.process_info, id="process-status-content")

    def watch_process_info(self, new_info: str) -> None:
        """Update process info when it changes."""
        self.query_one("#process-status-content", Static).update(new_info)


class JobQueuePanel(Static):
    """Panel showing job queue information."""

    queue_info = reactive("")

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("[bold]Job Queue[/bold]", id="queue-title")
        yield Static(self.queue_info, id="queue-content")

    def watch_queue_info(self, new_info: str) -> None:
        """Update queue info when it changes."""
        self.query_one("#queue-content", Static).update(new_info)


class StatisticsPanel(Static):
    """Panel showing session statistics."""

    stats_info = reactive("")

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("[bold]Session Statistics[/bold]", id="stats-title")
        yield Static(self.stats_info, id="stats-content")

    def watch_stats_info(self, new_info: str) -> None:
        """Update stats info when it changes."""
        self.query_one("#stats-content", Static).update(new_info)


class ResourcesPanel(Static):
    """Panel showing system resources."""

    resources_info = reactive("")

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("[bold]System Resources[/bold]", id="resources-title")
        yield Static(self.resources_info, id="resources-content")

    def watch_resources_info(self, new_info: str) -> None:
        """Update resources info when it changes."""
        self.query_one("#resources-content", Static).update(new_info)


class FeaturesPanel(Static):
    """Panel showing enabled features."""

    def __init__(self, bridge_data: reGenBridgeData) -> None:
        """Initialize the features panel."""
        super().__init__()
        self.bridge_data = bridge_data

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        def check(value: bool) -> str:
            return "[green]✓[/green]" if value else "[red]✗[/red]"

        features_text = (
            f"[bold]Active Features[/bold]\n"
            f"{check(self.bridge_data.allow_img2img)} img2img | "
            f"{check(self.bridge_data.allow_lora)} LoRA | "
            f"{check(self.bridge_data.allow_controlnet)} ControlNet\n"
            f"{check(self.bridge_data.allow_sdxl_controlnet)} SDXL ControlNet | "
            f"{check(self.bridge_data.allow_post_processing)} Post-Processing | "
            f"{check(self.bridge_data.post_process_job_overlap)} PP Overlap"
        )
        yield Static(features_text, id="features-text")


class HordeWorkerTUI(App):
    """Textual TUI application for horde-worker-reGen."""

    CSS = """
    Screen {
        background: $surface;
    }

    #header-text {
        background: $primary;
        color: $text;
        padding: 1;
        border: solid $accent;
    }

    #config-text, #features-text {
        background: $surface-darken-1;
        padding: 1;
        border: solid $accent;
        margin: 1;
    }

    #process-status-title, #queue-title, #stats-title, #resources-title {
        background: $accent;
        color: $text;
        padding: 0 1;
    }

    #process-status-content, #queue-content, #stats-content, #resources-content {
        background: $surface-darken-1;
        padding: 1;
        min-height: 5;
        border: solid $accent;
    }

    #activity-log {
        background: $surface-darken-1;
        border: solid $accent;
        padding: 1;
        height: 10;
    }

    .panel {
        background: $surface-darken-1;
        border: solid $accent;
        margin: 1;
        padding: 1;
    }

    Horizontal {
        height: auto;
    }

    Vertical {
        height: auto;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle Dark Mode"),
    ]

    def __init__(
        self,
        bridge_data: reGenBridgeData,
        process_manager: HordeWorkerProcessManager | None = None,
    ) -> None:
        """Initialize the TUI application."""
        super().__init__()
        self.bridge_data = bridge_data
        self.process_manager = process_manager
        self.session_start_time = time.time()
        self.version = horde_worker_regen.__version__

    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header()
        yield WorkerHeader(self.bridge_data, self.version)
        yield ConfigPanel(self.bridge_data)

        with Horizontal():
            with Vertical():
                yield ProcessStatusPanel()
                yield JobQueuePanel()
            with Vertical():
                yield StatisticsPanel()
                yield ResourcesPanel()

        yield FeaturesPanel(self.bridge_data)
        yield RichLog(id="activity-log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        """Start the update timer when app mounts."""
        self.set_interval(2.0, self.update_status)
        self.log_message("TUI started successfully")

    def log_message(self, message: str) -> None:
        """Add a message to the activity log."""
        log = self.query_one("#activity-log", RichLog)
        timestamp = datetime.now().strftime("%H:%M:%S")
        log.write(f"[dim]{timestamp}[/dim] {message}")

    def update_status(self) -> None:
        """Update all status panels with current information."""
        if not self.process_manager:
            return

        try:
            # Update worker header
            header = self.query_one(WorkerHeader)
            header.update_display(self.process_manager.user_info)

            # Update process status
            process_panel = self.query_one(ProcessStatusPanel)
            process_info_strings = self.process_manager.process_map.get_process_info_strings()
            process_text = "\n".join(process_info_strings) if process_info_strings else "No active processes"
            process_panel.process_info = process_text

            # Update job queue
            queue_panel = self.query_one(JobQueuePanel)
            jobs_pending = len(self.process_manager.jobs_pending_inference)
            jobs_safety = self.process_manager.jobs_pending_safety_check
            jobs_in_progress = self.process_manager.jobs_in_progress
            pending_mps = self.process_manager.pending_megapixelsteps

            queue_text = (
                f"Jobs pending inference: {jobs_pending}\n"
                f"Jobs in progress: {jobs_in_progress}\n"
                f"Jobs pending safety check: {jobs_safety}\n"
                f"Pending megapixelsteps: {pending_mps}"
            )
            queue_panel.queue_info = queue_text

            # Update statistics
            stats_panel = self.query_one(StatisticsPanel)
            session_time = time.time() - self.session_start_time
            kudos_per_hour = 0
            if session_time > 0:
                # Calculate kudos per hour if we have user info
                if hasattr(self.process_manager, 'session_kudos_earned'):
                    kudos_per_hour = (self.process_manager.session_kudos_earned / session_time) * 3600

            stats_text = (
                f"Jobs popped: {self.process_manager.num_jobs_total}\n"
                f"Jobs submitted: {self.process_manager.total_num_completed_jobs}\n"
                f"Jobs faulted: {self.process_manager.num_jobs_faulted}\n"
                f"Process recoveries: {self.process_manager.num_process_recoveries}\n"
                f"Slow jobs: {self.process_manager.num_job_slowdowns}\n"
                f"Time without jobs: {self.process_manager.time_spent_no_jobs_available:.2f}s"
            )
            if kudos_per_hour > 0:
                stats_text += f"\nEstimated kudos/hour: {kudos_per_hour:.2f}"
            stats_panel.stats_info = stats_text

            # Update resources
            resources_panel = self.query_one(ResourcesPanel)
            device_map = self.process_manager.device_map
            total_ram = self.process_manager.total_ram_gigabytes

            resources_text = f"System RAM: {total_ram}GB\n"

            if device_map:
                for device_id, device_info in device_map.items():
                    resources_text += (
                        f"\nGPU {device_id}: {device_info.get('name', 'Unknown')}\n"
                        f"  VRAM: {device_info.get('vram_used', 'N/A')}/{device_info.get('vram_total', 'N/A')}"
                    )
            else:
                resources_text += "GPU info not available"

            resources_panel.resources_info = resources_text

        except Exception as e:
            self.log_message(f"[red]Error updating status: {e}[/red]")

    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.dark = not self.dark

    def set_process_manager(self, process_manager: HordeWorkerProcessManager) -> None:
        """Set the process manager after initialization."""
        self.process_manager = process_manager
