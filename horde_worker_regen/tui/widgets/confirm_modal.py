"""Small confirmation modal for destructive TUI actions."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ConfirmModal(ModalScreen[bool]):
    """Ask the operator to confirm or cancel a destructive action."""

    DEFAULT_CSS = """
    ConfirmModal {
        align: center middle;
    }
    ConfirmModal #confirm-dialog {
        width: 72;
        max-width: 90%;
        height: auto;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }
    ConfirmModal #confirm-message {
        height: auto;
    }
    ConfirmModal #confirm-actions {
        height: auto;
        padding-top: 1;
    }
    ConfirmModal #confirm-actions Button {
        margin-right: 1;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, message: str, *, confirm_label: str, cancel_label: str) -> None:
        """Store the confirmation message and button labels."""
        super().__init__()
        self._message = message
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label

    def compose(self) -> ComposeResult:
        """Lay out the warning text and yes/no buttons."""
        with Vertical(id="confirm-dialog"):
            yield Static(self._message, id="confirm-message")
            with Horizontal(id="confirm-actions"):
                yield Button(self._confirm_label, id="confirm-yes", variant="error")
                yield Button(self._cancel_label, id="confirm-no", variant="success")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with True for confirm, False for cancel."""
        self.dismiss(event.button.id == "confirm-yes")

    def action_cancel(self) -> None:
        """Treat Escape as the non-destructive answer."""
        self.dismiss(False)
