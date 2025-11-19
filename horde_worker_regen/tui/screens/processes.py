"""Process details screen for the Horde Worker TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Header, Footer, Label

from horde_worker_regen.tui.widgets import ProcessCard

if TYPE_CHECKING:
    from horde_worker_regen.tui.data_provider import TUIDataProvider


class ProcessesScreen(Screen):
    """Screen showing detailed information about all worker processes."""

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
    ProcessesScreen {
        background: $surface;
    }

    #processes-header {
        dock: top;
        height: auto;
        background: $primary;
        color: $text;
        padding: 1 2;
        text-style: bold;
    }

    #processes-scroll {
        height: 1fr;
        margin: 1;
    }

    #no-processes {
        height: 100%;
        content-align: center middle;
        color: $text-muted;
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
        """Initialize the processes screen.

        Args:
            data_provider: Data provider instance
            name: Screen name
            id: Screen ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self.data_provider = data_provider
        self._process_cards: dict[int, ProcessCard] = {}

    def compose(self) -> ComposeResult:
        """Compose the processes screen layout."""
        yield Header(show_clock=True)
        yield Label("Worker Processes - Detailed View", id="processes-header")
        with VerticalScroll(id="processes-scroll"):
            yield Label("Loading processes...", id="no-processes")
        yield Footer()

    def on_mount(self) -> None:
        """Handle screen mount."""
        self.set_interval(1.0, self.refresh_data)
        self.refresh_data()

    def refresh_data(self) -> None:
        """Refresh process data."""
        try:
            processes = self.data_provider.get_process_info()

            # Remove the "no processes" message if it exists
            if processes:
                try:
                    no_proc_label = self.query_one("#no-processes", Label)
                    no_proc_label.remove()
                except Exception:
                    pass

            scroll_container = self.query_one("#processes-scroll", VerticalScroll)

            # Get existing process IDs
            existing_ids = set(self._process_cards.keys())
            current_ids = {p["id"] for p in processes}

            # Remove cards for processes that no longer exist
            for proc_id in existing_ids - current_ids:
                card = self._process_cards.pop(proc_id)
                card.remove()

            # Add or update cards
            for process_data in processes:
                proc_id = process_data["id"]
                if proc_id not in self._process_cards:
                    # Create new card
                    card = ProcessCard(proc_id)
                    self._process_cards[proc_id] = card
                    scroll_container.mount(card)

                # Update card data
                self._process_cards[proc_id].update_process(process_data)

            # If no processes, show message
            if not processes and "no-processes" not in [w.id for w in scroll_container.children]:
                scroll_container.mount(Label("No processes running", id="no-processes"))

        except Exception as e:
            # Log error but don't crash
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
