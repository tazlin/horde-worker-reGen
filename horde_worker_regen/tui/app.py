"""Main Textual application for Horde Worker reGen TUI."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.app import App
from textual.driver import Driver

from horde_worker_regen.tui.data_provider import TUIDataProvider
from horde_worker_regen.tui.screens import (
    ConfigScreen,
    DashboardScreen,
    JobsScreen,
    LogsScreen,
    ProcessesScreen,
    StatsScreen,
)

if TYPE_CHECKING:
    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager


class HordeWorkerTUI(App):
    """Main Textual UI application for Horde Worker reGen."""

    CSS = """
    App {
        background: $surface;
    }
    """

    SCREENS = {
        "dashboard": DashboardScreen,
        "processes": ProcessesScreen,
        "jobs": JobsScreen,
        "stats": StatsScreen,
        "config": ConfigScreen,
        "logs": LogsScreen,
    }

    def __init__(
        self,
        process_manager: HordeWorkerProcessManager,
        driver_class: type[Driver] | None = None,
        css_path: str | None = None,
        watch_css: bool = False,
    ):
        """Initialize the TUI application.

        Args:
            process_manager: The HordeWorkerProcessManager instance to monitor
            driver_class: Optional driver class override
            css_path: Optional CSS file path
            watch_css: Whether to watch CSS for changes
        """
        super().__init__(driver_class=driver_class, css_path=css_path, watch_css=watch_css)
        self.process_manager = process_manager
        self.data_provider = TUIDataProvider(process_manager)

    def on_mount(self) -> None:
        """Handle app mount."""
        # Install all screens with data provider
        self.install_screen(DashboardScreen(self.data_provider), name="dashboard")
        self.install_screen(ProcessesScreen(self.data_provider), name="processes")
        self.install_screen(JobsScreen(self.data_provider), name="jobs")
        self.install_screen(StatsScreen(self.data_provider), name="stats")
        self.install_screen(ConfigScreen(self.data_provider), name="config")
        self.install_screen(LogsScreen(self.data_provider), name="logs")

        # Start with dashboard
        self.push_screen("dashboard")

    async def run_async_with_worker(self) -> None:
        """Run the TUI alongside the worker process manager.

        This method runs the TUI and process manager concurrently.
        """
        # Create tasks for both the TUI and the process manager
        tui_task = asyncio.create_task(self.run_async())
        worker_task = asyncio.create_task(self.process_manager.start())

        # Wait for either to complete (likely the worker will run indefinitely)
        done, pending = await asyncio.wait(
            [tui_task, worker_task], return_when=asyncio.FIRST_COMPLETED
        )

        # If TUI exits, we might want to stop the worker
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def integrate_with_loguru(self) -> None:
        """Integrate with loguru to capture log messages.

        Call this method after initializing the app to capture
        loguru messages and display them in the TUI logs screen.
        """
        try:
            from loguru import logger

            # Add a custom sink that forwards to the TUI
            def tui_sink(message: str) -> None:
                """Custom loguru sink for TUI."""
                # Parse the loguru message format
                # This is a simplified version - you may need to adjust based on actual format
                try:
                    # Extract level and message from loguru's formatted output
                    # loguru format: "2024-01-01 12:00:00 | LEVEL    | message"
                    parts = message.split("|")
                    if len(parts) >= 3:
                        level = parts[1].strip()
                        msg = "|".join(parts[2:]).strip()
                        self.data_provider.add_log(level, msg)
                    else:
                        self.data_provider.add_log("INFO", message.strip())
                except Exception:
                    # Fallback
                    self.data_provider.add_log("INFO", message.strip())

            # Add the sink
            logger.add(tui_sink, format="{time} | {level: <8} | {message}", level="INFO")

        except ImportError:
            # loguru not available
            pass


def run_tui_with_worker(process_manager: HordeWorkerProcessManager) -> None:
    """Run the TUI application with the worker process manager.

    Args:
        process_manager: The HordeWorkerProcessManager instance to monitor
    """
    app = HordeWorkerTUI(process_manager)

    # Integrate with loguru if available
    app.integrate_with_loguru()

    # Run the app
    asyncio.run(app.run_async_with_worker())
