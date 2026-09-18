"""A modal that browses text-generation models in a table and dismisses with the one chosen.

The scribe configures a single model, so this picker chooses one name rather than editing a list: a
click selects a row, Choose dismisses with its name, and the config editor writes it into the field's
input. Typing a path into that same input is the other way to answer the field, which is why the modal
never has to express "a file of my own".

Models the reference knows but declares no file for are listed under a divider and cannot be chosen:
the worker would have no artefact to load for them.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Footer, Input, Label, Static
from textual.widgets.data_table import ColumnKey

from horde_worker_regen.tui.formatters import human_bytes, human_mb
from horde_worker_regen.tui.model_catalog import TextModelInfo, load_text_models
from horde_worker_regen.tui.responsive import PHONE_BAND_MAX_WIDTH, ResponsiveModalScreen

_NAME_WIDTH = 52

_DIVIDER_NAME = "— known to the reference, no file on offer —"
"""The separator row between choosable models and the ones with nothing to load."""

# (header, width) for every table column, in table order.
_COLUMNS: tuple[tuple[str, int], ...] = (
    ("Model", _NAME_WIDTH),
    ("Quant", 9),
    ("File", 10),
    ("VRAM", 10),
    ("Context", 9),
    ("On disk", 9),
)

_PHONE_COLUMN_WIDTHS: tuple[int, ...] = (24, 7, 8, 8, 7, 8)
"""Column widths used below the terminal floor, in the same order as ``_COLUMNS``.

A ``DataTable`` sheds no columns, so a phone-width picker would otherwise open with the model name cut
off mid-word. The widths are chosen when the columns are created, so a picker left open across a resize
keeps the ones it opened with.
"""


def _column_headers() -> tuple[str, ...]:
    """The picker table's column headers, in table order."""
    return tuple(header for header, _width in _COLUMNS)


def _column_widths(app: App[object]) -> tuple[int, ...]:
    """The picker table's column widths for the app's current width band, in table order."""
    if app.size.width < PHONE_BAND_MAX_WIDTH:
        return _PHONE_COLUMN_WIDTHS
    return tuple(width for _header, width in _COLUMNS)


class TextModelPickerModal(ResponsiveModalScreen[str | None]):
    """Browse the text models the reference knows; dismisses with one name, or None when cancelled."""

    BINDINGS = [
        Binding("enter", "choose", "Choose"),
        Binding("o", "open_homepage", "Open homepage"),
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    TextModelPickerModal {
        align: center middle;
    }
    TextModelPickerModal #text-picker-dialog {
        width: 90%;
        max-width: 200;
        height: 90%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    TextModelPickerModal .dialog-title {
        text-style: bold;
    }
    TextModelPickerModal #text-picker-search {
        margin: 1 0;
    }
    TextModelPickerModal #text-picker-body {
        height: 1fr;
    }
    TextModelPickerModal #text-picker-table {
        width: 2fr;
    }
    TextModelPickerModal #text-picker-detail {
        width: 1fr;
        margin-left: 1;
        border: round $foreground 20%;
        padding: 0 1;
    }
    TextModelPickerModal #text-picker-status {
        color: $text-muted;
    }
    TextModelPickerModal .dialog-buttons {
        height: auto;
        padding-top: 1;
    }
    TextModelPickerModal .dialog-buttons Button {
        margin-right: 1;
    }
    """

    def __init__(self, text_models_dir: str = "") -> None:
        """Create the picker.

        Args:
            text_models_dir: The operator's configured text models folder, so the on-disk column answers
                for where this worker would actually look. Blank uses the worker's own default.
        """
        super().__init__()
        self._text_models_dir = text_models_dir.strip()
        self._all_models: list[TextModelInfo] = []
        self._visible: list[TextModelInfo | None] = []
        """One entry per table row; None is the divider row, which is not a model."""
        self._current: TextModelInfo | None = None
        self._loaded = False
        self._col_keys: list[ColumnKey] = []

    def compose(self) -> ComposeResult:
        """Lay out the search box, the model table with a detail panel, and the buttons."""
        with Vertical(id="text-picker-dialog"):
            yield Label(
                "Click a row to inspect  ·  Choose sets it as the scribe's model",
                classes="dialog-title",
            )
            yield Input(placeholder="search name…", id="text-picker-search")
            with Horizontal(id="text-picker-body"):
                yield DataTable(id="text-picker-table", cursor_type="row", zebra_stripes=True)
                with VerticalScroll(id="text-picker-detail"):
                    yield Static("Loading model reference…", id="text-picker-detail-body")
            yield Static("Loading model reference…", id="text-picker-status")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Choose", variant="success", id="text-picker-choose")
                yield Button("Cancel", id="text-picker-cancel")
            yield Footer()

    def on_mount(self) -> None:
        """Set up the table columns and load the text reference off the UI thread."""
        table = self.query_one("#text-picker-table", DataTable)
        self._col_keys = [
            table.add_column(header, width=width)
            for header, width in zip(_column_headers(), _column_widths(self.app), strict=True)
        ]
        self.run_worker(self._load_models, thread=True, exclusive=True)

    def _load_models(self) -> None:
        """Read the merged text reference (runs in a worker thread)."""
        try:
            models = load_text_models(Path(self._text_models_dir) if self._text_models_dir else None)
        except Exception as error:  # noqa: BLE001 - surface any loader failure to the user
            self.app.call_from_thread(self._on_load_error, f"{type(error).__name__}: {error}")
            return
        self.app.call_from_thread(self._on_models_loaded, models)

    def _on_models_loaded(self, models: list[TextModelInfo]) -> None:
        """Populate the table once the reference arrives."""
        self._all_models = models
        self._loaded = True
        self._apply_filters()

    def _on_load_error(self, message: str) -> None:
        """Show a clear error when the reference cannot be loaded."""
        self.query_one("#text-picker-status", Static).update(
            f"[red]Could not load the model reference ({message}). "
            "Run the worker once to download it, then reopen this picker.[/]",
        )
        self.query_one("#text-picker-detail-body", Static).update("")

    @staticmethod
    def _matches_search(model: TextModelInfo, search: str) -> bool:
        """Case-insensitive substring match on the model's name."""
        return not search or search in model.name.lower()

    @staticmethod
    def cells_for(model: TextModelInfo) -> tuple[str, str, Text, Text, str, Text]:
        """The six table cells for a model: name, quant, file size, footprint, context, on-disk."""
        name = model.name if len(model.name) <= _NAME_WIDTH else model.name[: _NAME_WIDTH - 1] + "…"
        return (
            name,
            model.quant or "-",
            Text(human_bytes(model.size_bytes) if model.size_bytes else "-"),
            Text(human_mb(model.footprint_mb), style="" if model.footprint_mb else "grey50"),
            str(model.context) if model.context else "-",
            Text("on disk", style="green") if model.on_disk else Text("-", style="grey50"),
        )

    @staticmethod
    def _divider_cells() -> tuple[str, str, Text, Text, str, Text]:
        """The cells of the row separating choosable models from the ones with no file on offer."""
        return (_DIVIDER_NAME, "", Text(""), Text(""), "", Text(""))

    def rows_for(self, models: list[TextModelInfo]) -> list[TextModelInfo | None]:
        """Order *models* for display, inserting the divider before the first one with no file.

        Returns:
            One entry per table row, with None standing for the divider row.
        """
        rows: list[TextModelInfo | None] = []
        divider_placed = False
        for model in models:
            if not model.has_file and not divider_placed:
                rows.append(None)
                divider_placed = True
            rows.append(model)
        return rows

    def _apply_filters(self) -> None:
        """Rebuild the table from the current search text."""
        if not self._loaded:
            return
        search = self.query_one("#text-picker-search", Input).value.strip().lower()
        self._visible = self.rows_for([model for model in self._all_models if self._matches_search(model, search)])

        table = self.query_one("#text-picker-table", DataTable)
        table.clear()
        for row in self._visible:
            table.add_row(*(self._divider_cells() if row is None else self.cells_for(row)))

        self._refresh_status()
        self._show_detail(0)

    def _refresh_status(self) -> None:
        """Update the status line with the shown, choosable, and total counts."""
        shown = [row for row in self._visible if row is not None]
        choosable = [model for model in shown if model.has_file]
        text = Text.assemble(
            (f"{len(shown)} shown (of {len(self._all_models)})", "grey70"),
            ("  ·  ", "grey50"),
            (f"{len(choosable)} with a file on offer", "green" if choosable else "grey70"),
        )
        self.query_one("#text-picker-status", Static).update(text)

    def _show_detail(self, row: int) -> None:
        """Show the model at table ``row`` in the detail panel."""
        body = self.query_one("#text-picker-detail-body", Static)
        model = self._visible[row] if 0 <= row < len(self._visible) else None
        self._current = model
        if model is None:
            body.update(
                "The models below this divider are names the reference knows without a file to load, so "
                "the worker has no artefact to obtain for them."
                if self._visible
                else "No models match the search.",
            )
            return
        body.update(self._detail_for(model))

    @staticmethod
    def _detail_for(model: TextModelInfo) -> RenderableType:
        """The record for the detail panel, including a clickable homepage link."""
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="bold cyan", justify="right", no_wrap=True)
        grid.add_column()
        if not model.has_file:
            grid.add_row("File", Text("none on offer", style="yellow"))
        else:
            grid.add_row(
                "File",
                Text("on disk", style="green") if model.on_disk else Text("will be fetched", style="yellow"),
            )
            if model.size_bytes:
                grid.add_row("Size", human_bytes(model.size_bytes))
            if model.target_path:
                grid.add_row("Path", Text(model.target_path, style="grey70"))
        if model.quant:
            grid.add_row("Quant", model.quant)
        if model.parameters_count:
            grid.add_row("Parameters", f"{model.parameters_count:,}")
        if model.footprint_mb is not None:
            grid.add_row("Measured VRAM", human_mb(model.footprint_mb))
        if model.context is not None:
            grid.add_row("Measured at context", str(model.context))
        if model.tokens_per_second is not None:
            grid.add_row("Measured rate", f"{model.tokens_per_second:.0f} tokens/s")
        if model.source_card:
            grid.add_row("Measured on", model.source_card)

        parts: list[RenderableType] = [Text(model.name, style="bold"), grid]
        if model.description:
            parts.append(Text(""))
            parts.append(Text(model.description, style="grey70"))
        if model.footprint_mb is None:
            parts.append(Text(""))
            parts.append(
                Text(
                    "Nobody has measured this model on this worker, so its VRAM cost is unknown rather than small.",
                    style="grey70",
                ),
            )
        if model.homepage:
            parts.append(Text(""))
            parts.append(
                Text.assemble(
                    ("Homepage: ", "bold cyan"),
                    (model.homepage, f"underline blue link {model.homepage}"),
                ),
            )
            parts.append(Text("press 'o' to open in your browser", style="grey50"))
        return Group(*parts)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Update the detail panel as the cursor moves (click or keyboard)."""
        self._show_detail(event.cursor_row)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Re-filter as the search text changes."""
        if event.input.id == "text-picker-search":
            self._apply_filters()

    def action_choose(self) -> None:
        """Dismiss with the highlighted model's name, refusing one with no file to load."""
        model = self._current
        if model is None:
            self.notify("Select a model row first.", severity="warning")
            return
        if not model.has_file:
            self.notify(
                f"{model.name} has no file on offer, so the worker has nothing to load for it.",
                severity="warning",
            )
            return
        self.dismiss(model.name)

    def action_open_homepage(self) -> None:
        """Open the highlighted model's homepage in the browser, if it has one."""
        if self._current is not None and self._current.homepage:
            webbrowser.open(self._current.homepage)
            self.notify(f"Opening {self._current.homepage}")
        else:
            self.notify("This model has no homepage.", severity="warning")

    def action_cancel(self) -> None:
        """Dismiss without choosing anything."""
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Choose the highlighted model, or cancel."""
        if event.button.id == "text-picker-cancel":
            self.dismiss(None)
        elif event.button.id == "text-picker-choose":
            self.action_choose()
