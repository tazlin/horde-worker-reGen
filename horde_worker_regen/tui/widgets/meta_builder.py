"""A modal that builds a meta model-load command (top N, all sdxl, …) with inline guidance."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from horde_worker_regen.tui.model_catalog import (
    GENERAL_META_GUIDANCE,
    META_OPTIONS,
    META_OPTIONS_BY_KIND,
    MetaKind,
    build_meta_instruction,
)


class MetaBuilderModal(ModalScreen[str | None]):
    """Compose one meta instruction; dismisses with the instruction string or None if cancelled."""

    DEFAULT_CSS = """
    MetaBuilderModal {
        align: center middle;
    }
    MetaBuilderModal #meta-dialog {
        width: 72;
        height: auto;
        max-height: 90%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    MetaBuilderModal .dialog-title {
        text-style: bold;
    }
    MetaBuilderModal #meta-kind, MetaBuilderModal #meta-count {
        margin: 1 0;
    }
    MetaBuilderModal #meta-guidance {
        color: $text-muted;
        padding: 1 0;
    }
    MetaBuilderModal #meta-general {
        color: $text-disabled;
    }
    MetaBuilderModal .dialog-buttons {
        height: auto;
        padding-top: 1;
    }
    MetaBuilderModal .dialog-buttons Button {
        margin-right: 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Lay out the kind selector, count input, guidance, and buttons."""
        with Vertical(id="meta-dialog"):
            yield Label("Build a meta model-load command", classes="dialog-title")
            yield Select(
                [(option.label, option.kind.value) for option in META_OPTIONS],
                value=META_OPTIONS[0].kind.value,
                allow_blank=False,
                id="meta-kind",
            )
            yield Input(placeholder="how many (e.g. 5)", value="5", type="integer", id="meta-count")
            yield Static(id="meta-guidance")
            yield Static(GENERAL_META_GUIDANCE, id="meta-general")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Add", variant="primary", id="meta-add")
                yield Button("Cancel", id="meta-cancel")

    def on_mount(self) -> None:
        """Show the guidance for the initial selection."""
        self._refresh(META_OPTIONS[0].kind)

    def _current_kind(self) -> MetaKind:
        """The currently selected meta kind."""
        return MetaKind(str(self.query_one("#meta-kind", Select).value))

    def _refresh(self, kind: MetaKind) -> None:
        """Update the guidance text and count-input visibility for a kind."""
        option = META_OPTIONS_BY_KIND[kind]
        self.query_one("#meta-guidance", Static).update(option.guidance)
        self.query_one("#meta-count", Input).display = option.needs_count

    def on_select_changed(self, event: Select.Changed) -> None:
        """React to a kind change."""
        if event.select.id == "meta-kind" and event.value is not Select.BLANK:
            self._refresh(MetaKind(str(event.value)))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Build-and-return on Add, or cancel."""
        if event.button.id == "meta-cancel":
            self.dismiss(None)
            return
        if event.button.id == "meta-add":
            kind = self._current_kind()
            count = 1
            if META_OPTIONS_BY_KIND[kind].needs_count:
                try:
                    count = int(self.query_one("#meta-count", Input).value)
                except ValueError:
                    count = 1
            self.dismiss(build_meta_instruction(kind, count))
