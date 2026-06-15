"""Small shared widgets and styling helpers for the TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label

from horde_worker_regen.tui.worker_launcher import SupervisorStatus

_STATUS_STYLES: dict[SupervisorStatus, tuple[str, str]] = {
    SupervisorStatus.STARTING: ("starting", "yellow"),
    SupervisorStatus.RUNNING: ("running", "green"),
    SupervisorStatus.CRASHED: ("crashed", "red"),
    SupervisorStatus.RESTARTING: ("restarting", "yellow"),
    SupervisorStatus.STOPPED: ("stopped", "grey50"),
}


def status_markup(status: SupervisorStatus, *, paused: bool = False) -> str:
    """Return a Rich-markup badge for a supervisor status (paused overrides the running label)."""
    if paused and status is SupervisorStatus.RUNNING:
        return "[black on yellow] PAUSED [/]"
    label, colour = _STATUS_STYLES.get(status, (status.value, "white"))
    return f"[black on {colour}] {label.upper()} [/]"


class StatCard(Vertical):
    """A compact headline metric: a muted title above a bold value."""

    DEFAULT_CSS = """
    StatCard {
        width: 1fr;
        height: 5;
        border: round $foreground 20%;
        padding: 0 1;
        margin: 0 1 1 0;
    }
    StatCard .stat-title {
        color: $text-muted;
        text-style: bold;
    }
    StatCard .stat-value {
        text-style: bold;
        padding-top: 1;
    }
    """

    def __init__(self, title: str, value: str = "-", *, card_id: str | None = None) -> None:
        """Create a stat card.

        Args:
            title: The muted caption shown above the value.
            value: The initial value text.
            card_id: An optional DOM id for the card itself (used to target it for updates).
        """
        super().__init__(id=card_id)
        self._title = title
        self._initial_value = value

    def compose(self) -> ComposeResult:
        """Lay out the title and value labels."""
        yield Label(self._title, classes="stat-title")
        yield Label(self._initial_value, classes="stat-value")

    def update_value(self, value: str) -> None:
        """Replace the displayed value."""
        self.query_one(".stat-value", Label).update(value)
