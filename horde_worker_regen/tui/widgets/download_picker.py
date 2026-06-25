"""A modal to choose which models to pre-fetch, defaulted to what the current config would download.

The operator opens this from the Downloads tab to fetch models *without* committing the GPU. The list is
seeded from the configured model set (the same resolution the Config tab shows), with the not-yet-present
models pre-selected, so the default action is exactly "download what this configuration is missing". The
operator may narrow or broaden the selection and optionally include the auxiliary models (CLIP/BLIP,
ControlNet, post-processors, default LoRas). Confirming dismisses with the chosen names; the app turns
that into a download request (entering the download-only hold first when the worker is not yet running).
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, SelectionList, Static
from textual.widgets.selection_list import Selection

from horde_worker_regen.tui.formatters import human_bytes


@dataclass(frozen=True)
class DownloadPickerRow:
    """One selectable model: its name, baseline, declared size, and whether it is already on disk."""

    name: str
    baseline: str
    size_bytes: int | None
    on_disk: bool


@dataclass(frozen=True)
class DownloadSelection:
    """The picker's result: the chosen model names and whether to also fetch the auxiliary models."""

    model_names: list[str]
    include_aux: bool


class DownloadPickerModal(ModalScreen["DownloadSelection | None"]):
    """Choose models to download (defaulted to the config's missing set); dismiss with the selection or None."""

    DEFAULT_CSS = """
    DownloadPickerModal {
        align: center middle;
    }
    DownloadPickerModal #download-picker-dialog {
        width: 90%;
        height: 90%;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    DownloadPickerModal #download-picker-intro {
        height: auto;
        margin-bottom: 1;
    }
    DownloadPickerModal #download-picker-list {
        height: 1fr;
        border: round $panel;
        margin-bottom: 1;
    }
    DownloadPickerModal #download-picker-aux {
        height: auto;
        margin-bottom: 1;
    }
    DownloadPickerModal #download-picker-actions {
        height: 3;
    }
    DownloadPickerModal #download-picker-actions Button {
        margin-right: 1;
    }
    """

    BINDINGS = [("escape", "cancel", "Close")]

    def __init__(self, rows: list[DownloadPickerRow], *, include_aux_default: bool = False) -> None:
        """Store the candidate rows (pre-selecting the missing ones) and the default aux choice."""
        super().__init__()
        self._rows = rows
        self._include_aux_default = include_aux_default

    def compose(self) -> ComposeResult:
        """Lay out the intro, the multi-select model list, the aux toggle, and the action buttons."""
        with Vertical(id="download-picker-dialog"):
            yield Static(self._intro(), id="download-picker-intro")
            if self._rows:
                yield SelectionList[str](
                    *[Selection(self._prompt(row), row.name, initial_state=not row.on_disk) for row in self._rows],
                    id="download-picker-list",
                )
            else:
                yield Static(
                    Text(
                        "No configured models resolved yet. Open the Config → Models tab and press Resolve, "
                        "then reopen this picker.",
                        style="yellow",
                    ),
                    id="download-picker-list",
                )
            yield Checkbox(
                "Also download auxiliary models (CLIP/BLIP, ControlNet, post-processing, default LoRas)",
                value=self._include_aux_default,
                id="download-picker-aux",
            )
            with Horizontal(id="download-picker-actions"):
                yield Button("Download selected", id="download-picker-confirm", variant="success")
                yield Button("Cancel", id="download-picker-cancel", variant="default")

    def _intro(self) -> Text:
        """Explain that this only downloads (no GPU work) and that missing models are pre-selected."""
        missing = sum(1 for row in self._rows if not row.on_disk)
        return Text.assemble(
            ("Choose models to download\n\n", "bold"),
            (
                "These are the models this configuration would load. The "
                f"{missing} not yet on disk are pre-selected. Downloading does not start inference or use "
                "the GPU; the worker stays in a download-only hold until you press Go live.",
                "grey70",
            ),
        )

    @staticmethod
    def _prompt(row: DownloadPickerRow) -> Text:
        """Render one model's label: name, baseline, size, and on-disk vs missing."""
        status = Text("on disk", style="green") if row.on_disk else Text("missing", style="yellow")
        return Text.assemble(
            (row.name, "bold"),
            ("  ", ""),
            (row.baseline, "cyan"),
            ("  ", ""),
            (human_bytes(row.size_bytes), "grey62"),
            ("  ", ""),
            status,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Confirm with the current selection, or cancel."""
        if event.button.id == "download-picker-confirm":
            self._confirm()
        elif event.button.id == "download-picker-cancel":
            self.dismiss(None)
        event.stop()

    def _confirm(self) -> None:
        """Gather the selected names and aux choice, then dismiss with the selection."""
        include_aux = self.query_one("#download-picker-aux", Checkbox).value
        selected: list[str] = []
        if self._rows:
            selected = list(self.query_one("#download-picker-list", SelectionList).selected)
        if not selected and not include_aux:
            # Nothing to do; treat an empty confirm as a cancel rather than a no-op request.
            self.dismiss(None)
            return
        self.dismiss(DownloadSelection(model_names=sorted(selected), include_aux=include_aux))

    def action_cancel(self) -> None:
        """Close the modal without requesting any download (Escape)."""
        self.dismiss(None)


__all__ = ["DownloadPickerModal", "DownloadPickerRow", "DownloadSelection"]
