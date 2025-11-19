"""Configuration screen for the Horde Worker TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Grid, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Header, Footer, Label

if TYPE_CHECKING:
    from horde_worker_regen.tui.data_provider import TUIDataProvider


class ConfigScreen(Screen):
    """Screen showing worker configuration settings."""

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
    ConfigScreen {
        background: $surface;
    }

    #config-header {
        dock: top;
        height: auto;
        background: $primary;
        color: $text;
        padding: 1 2;
        text-style: bold;
    }

    #config-scroll {
        height: 1fr;
        margin: 1;
    }

    .config-panel {
        background: $panel;
        border: solid $accent;
        padding: 1;
        margin: 0 0 1 0;
        height: auto;
    }

    .panel-title {
        text-style: bold;
        color: $accent;
        background: $primary-darken-2;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    .config-row {
        height: auto;
        padding: 0 2;
        margin: 0 0 1 0;
    }

    .config-label {
        color: $text-muted;
        width: 30;
    }

    .config-value {
        color: $text;
        text-style: bold;
    }

    #models-list {
        background: $primary-darken-3;
        border: solid $primary;
        padding: 1;
        margin: 1 2;
        height: auto;
        min-height: 3;
    }

    .model-item {
        color: $text;
        padding: 0 1;
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
        """Initialize the config screen.

        Args:
            data_provider: Data provider instance
            name: Screen name
            id: Screen ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self.data_provider = data_provider

    def compose(self) -> ComposeResult:
        """Compose the config screen layout."""
        yield Header(show_clock=True)
        yield Label("Worker Configuration - Settings Overview", id="config-header")

        with VerticalScroll(id="config-scroll"):
            # General Settings Panel
            with Container(classes="config-panel"):
                yield Label("General Settings", classes="panel-title")
                with Horizontal(classes="config-row"):
                    yield Label("Worker Name:", classes="config-label")
                    yield Label("Loading...", id="cfg-worker-name", classes="config-value")
                with Horizontal(classes="config-row"):
                    yield Label("Max Threads:", classes="config-label")
                    yield Label("0", id="cfg-max-threads", classes="config-value")
                with Horizontal(classes="config-row"):
                    yield Label("Queue Size:", classes="config-label")
                    yield Label("0", id="cfg-queue-size", classes="config-value")
                with Horizontal(classes="config-row"):
                    yield Label("Max Power:", classes="config-label")
                    yield Label("0", id="cfg-max-power", classes="config-value")

            # Memory Settings Panel
            with Container(classes="config-panel"):
                yield Label("Memory Settings", classes="panel-title")
                with Horizontal(classes="config-row"):
                    yield Label("RAM to Leave Free:", classes="config-label")
                    yield Label("Default", id="cfg-ram-free", classes="config-value")
                with Horizontal(classes="config-row"):
                    yield Label("VRAM to Leave Free:", classes="config-label")
                    yield Label("Default", id="cfg-vram-free", classes="config-value")
                with Horizontal(classes="config-row"):
                    yield Label("High Performance Mode:", classes="config-label")
                    yield Label("No", id="cfg-high-perf", classes="config-value")
                with Horizontal(classes="config-row"):
                    yield Label("Moderate Performance Mode:", classes="config-label")
                    yield Label("No", id="cfg-mod-perf", classes="config-value")
                with Horizontal(classes="config-row"):
                    yield Label("Low Memory Mode:", classes="config-label")
                    yield Label("No", id="cfg-low-mem", classes="config-value")
                with Horizontal(classes="config-row"):
                    yield Label("Very Low Memory Mode:", classes="config-label")
                    yield Label("No", id="cfg-very-low-mem", classes="config-value")

            # Safety & Security Panel
            with Container(classes="config-panel"):
                yield Label("Safety & Security", classes="panel-title")
                with Horizontal(classes="config-row"):
                    yield Label("Safety on GPU:", classes="config-label")
                    yield Label("No", id="cfg-safety-gpu", classes="config-value")
                with Horizontal(classes="config-row"):
                    yield Label("Allow Unsafe IP:", classes="config-label")
                    yield Label("No", id="cfg-unsafe-ip", classes="config-value")
                with Horizontal(classes="config-row"):
                    yield Label("Require Upfront Kudos:", classes="config-label")
                    yield Label("No", id="cfg-upfront-kudos", classes="config-value")

            # Models Panel
            with Container(classes="config-panel"):
                yield Label("Loaded Models", classes="panel-title")
                with Container(id="models-list"):
                    yield Label("Loading models...", classes="model-item")

        yield Footer()

    def on_mount(self) -> None:
        """Handle screen mount."""
        self.set_interval(5.0, self.refresh_data)
        self.refresh_data()

    def refresh_data(self) -> None:
        """Refresh configuration data."""
        try:
            config = self.data_provider.get_config_info()

            if "error" in config:
                return

            # Update general settings
            self.query_one("#cfg-worker-name", Label).update(config.get("worker_name", "Unknown"))
            self.query_one("#cfg-max-threads", Label).update(str(config.get("max_threads", 0)))
            self.query_one("#cfg-queue-size", Label).update(str(config.get("queue_size", 0)))
            self.query_one("#cfg-max-power", Label).update(str(config.get("max_power", 0)))

            # Update memory settings
            self.query_one("#cfg-ram-free", Label).update(str(config.get("ram_to_leave_free", "Default")))
            self.query_one("#cfg-vram-free", Label).update(str(config.get("vram_to_leave_free", "Default")))
            self.query_one("#cfg-high-perf", Label).update(
                "Yes" if config.get("high_performance_mode") else "No"
            )
            self.query_one("#cfg-mod-perf", Label).update(
                "Yes" if config.get("moderate_performance_mode") else "No"
            )
            self.query_one("#cfg-low-mem", Label).update("Yes" if config.get("low_memory_mode") else "No")
            self.query_one("#cfg-very-low-mem", Label).update(
                "Yes" if config.get("very_low_memory_mode") else "No"
            )

            # Update safety settings
            self.query_one("#cfg-safety-gpu", Label).update("Yes" if config.get("safety_on_gpu") else "No")
            self.query_one("#cfg-unsafe-ip", Label).update(
                "Yes" if config.get("allow_unsafe_ip") else "No"
            )
            self.query_one("#cfg-upfront-kudos", Label).update(
                "Yes" if config.get("require_upfront_kudos") else "No"
            )

            # Update models list
            models_container = self.query_one("#models-list", Container)
            # Clear existing items
            models_container.remove_children()

            models = config.get("models", [])
            if models:
                for model in models:
                    models_container.mount(Label(f"• {model}", classes="model-item"))
            else:
                models_container.mount(Label("No models configured", classes="model-item"))

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
