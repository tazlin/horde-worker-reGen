"""A modal that browses image models in a table: click a row to inspect, click ＋ to add."""

from __future__ import annotations

import webbrowser

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.coordinate import Coordinate
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, DataTable, Input, Label, Select, Static

from horde_worker_regen.model_download_plan import free_model_bytes
from horde_worker_regen.tui.formatters import human_bytes
from horde_worker_regen.tui.model_catalog import ModelInfo, friendly_baseline, load_image_models

_ADD = "＋"
_REMOVE = "✕"
_NAME_WIDTH = 40


class ModelPickerModal(ModalScreen[list[str] | None]):
    """Browse and mark image models to add; dismisses with the chosen names or None."""

    BINDINGS = [
        Binding("space", "toggle_add", "Add / remove"),
        Binding("o", "open_homepage", "Open homepage"),
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    ModelPickerModal {
        align: center middle;
    }
    ModelPickerModal #picker-dialog {
        width: 120;
        height: 88%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    ModelPickerModal .dialog-title {
        text-style: bold;
    }
    ModelPickerModal #picker-filters {
        height: 3;
    }
    ModelPickerModal #picker-baseline {
        width: 34;
        margin-right: 1;
    }
    ModelPickerModal #picker-search {
        margin: 1 0;
    }
    ModelPickerModal #picker-body {
        height: 1fr;
    }
    ModelPickerModal #picker-table {
        width: 2fr;
    }
    ModelPickerModal #picker-detail {
        width: 1fr;
        margin-left: 1;
        border: round $foreground 20%;
        padding: 0 1;
    }
    ModelPickerModal #picker-status {
        color: $text-muted;
    }
    ModelPickerModal #picker-disk {
        height: auto;
    }
    ModelPickerModal .dialog-buttons {
        height: auto;
        padding-top: 1;
    }
    ModelPickerModal .dialog-buttons Button {
        margin-right: 1;
    }
    """

    def __init__(self, exclude: set[str] | None = None) -> None:
        """Create the picker, hiding models already in the list (``exclude``)."""
        super().__init__()
        self._exclude = exclude or set()
        self._all_models: list[ModelInfo] = []
        self._by_name: dict[str, ModelInfo] = {}
        self._visible: list[ModelInfo] = []
        self._chosen: set[str] = set()
        self._current: ModelInfo | None = None
        self._loaded = False
        self._free_disk_bytes: int | None = None

    def compose(self) -> ComposeResult:
        """Lay out the filters, search, model table with a detail panel, and buttons."""
        with Vertical(id="picker-dialog"):
            yield Label(
                "Click ＋ to add a model (✕ to remove)  ·  click a row to see its full record",
                classes="dialog-title",
            )
            with Horizontal(id="picker-filters"):
                yield Select([("All baselines", "")], value="", allow_blank=False, id="picker-baseline")
                yield Checkbox("Show NSFW", value=True, id="picker-nsfw")
                yield Checkbox("Inpainting only", value=False, id="picker-inpaint")
            yield Input(placeholder="search models…", id="picker-search")
            with Horizontal(id="picker-body"):
                yield DataTable(id="picker-table", cursor_type="cell", zebra_stripes=True)
                with VerticalScroll(id="picker-detail"):
                    yield Static("Loading model reference…", id="picker-detail-body")
            yield Static("Loading model reference…", id="picker-status")
            yield Static("", id="picker-disk")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Add marked", variant="primary", id="picker-add")
                yield Button("Cancel", id="picker-cancel")

    def on_mount(self) -> None:
        """Set up the table columns and load the model reference off the UI thread."""
        table = self.query_one("#picker-table", DataTable)
        table.add_column(" ", width=3)
        table.add_column("Model", width=_NAME_WIDTH)
        table.add_column("Baseline", width=10)
        table.add_column("Disk", width=10)
        table.add_column("Flags", width=12)
        self.run_worker(self._load_models, thread=True, exclusive=True)

    def _load_models(self) -> None:
        """Blocking load of the model reference (runs in a worker thread)."""
        try:
            models = load_image_models()
        except Exception as error:  # noqa: BLE001 - surface any loader failure to the user
            self.app.call_from_thread(self._on_load_error, f"{type(error).__name__}: {error}")
            return
        self.app.call_from_thread(self._on_models_loaded, models)

    def _on_models_loaded(self, models: list[ModelInfo]) -> None:
        """Populate the baseline filter and the table once models arrive."""
        self._all_models = models
        self._by_name = {model.name: model for model in models}
        self._free_disk_bytes = free_model_bytes()
        self._loaded = True
        baselines = sorted({model.baseline for model in models if model.baseline})
        self.query_one("#picker-baseline", Select).set_options(
            [("All baselines", ""), *[(friendly_baseline(baseline), baseline) for baseline in baselines]],
        )
        self._apply_filters()

    def _on_load_error(self, message: str) -> None:
        """Show a clear error when the reference can't be loaded."""
        self.query_one("#picker-status", Static).update(
            f"[red]Could not load the model reference ({message}). "
            "Run the worker once to download it, then reopen this picker.[/]",
        )
        self.query_one("#picker-detail-body", Static).update("")

    @staticmethod
    def _flags_for(model: ModelInfo) -> str:
        """A compact flags string for a model."""
        flags = []
        if model.nsfw:
            flags.append("nsfw")
        if model.inpainting:
            flags.append("inpaint")
        return " ".join(flags)

    @staticmethod
    def _disk_cell(model: ModelInfo) -> Text:
        """A compact on-disk badge: present, or the download size, or unknown."""
        if model.on_disk:
            return Text("on disk", style="green")
        if model.size_on_disk_bytes:
            return Text(human_bytes(model.size_on_disk_bytes), style="yellow")
        return Text("-", style="grey50")

    def cells_for(self, model: ModelInfo) -> tuple[str, str, str, Text, str]:
        """The five table cells for a model: state, name, baseline, disk badge, flags."""
        state = _REMOVE if model.name in self._chosen else _ADD
        name = model.name if len(model.name) <= _NAME_WIDTH else model.name[: _NAME_WIDTH - 1] + "…"
        return state, name, friendly_baseline(model.baseline) or "-", self._disk_cell(model), self._flags_for(model)

    def _apply_filters(self) -> None:
        """Rebuild the table (and the parallel visible list) from the current filters."""
        if not self._loaded:
            return
        baseline = str(self.query_one("#picker-baseline", Select).value)
        show_nsfw = self.query_one("#picker-nsfw", Checkbox).value
        inpaint_only = self.query_one("#picker-inpaint", Checkbox).value
        search = self.query_one("#picker-search", Input).value.strip().lower()

        self._visible = [
            model
            for model in self._all_models
            if model.name not in self._exclude
            and (not baseline or model.baseline == baseline)
            and (show_nsfw or not model.nsfw)
            and (not inpaint_only or model.inpainting)
            and (not search or search in model.name.lower())
        ]

        table = self.query_one("#picker-table", DataTable)
        table.clear()
        for model in self._visible:
            table.add_row(*self.cells_for(model))

        self._refresh_status()
        self._show_detail(0)

    def _refresh_status(self) -> None:
        """Update the status line with the visible and marked counts, and the disk footer."""
        self.query_one("#picker-status", Static).update(
            f"{len(self._visible)} shown (of {len(self._all_models)})  ·  {len(self._chosen)} marked to add",
        )
        self.query_one("#picker-disk", Static).update(self._disk_footer())

    def _disk_footer(self) -> Text:
        """A disk-budget line for the resulting config (already-in-list models plus the marked ones)."""
        present_bytes = 0
        to_download_bytes = 0
        num_present = 0
        num_to_download = 0
        for name in self._exclude | self._chosen:
            model = self._by_name.get(name)
            if model is None:
                continue  # A meta command or a name absent from the reference; it cannot be sized.
            if model.on_disk:
                num_present += 1
                present_bytes += model.size_on_disk_bytes or 0
            else:
                num_to_download += 1
                to_download_bytes += model.size_on_disk_bytes or 0

        total_bytes = present_bytes + to_download_bytes
        fits = self._free_disk_bytes is None or to_download_bytes <= self._free_disk_bytes
        footer = Text.assemble(
            ("config disk  ", "grey50"),
            (f"on disk {num_present} ({human_bytes(present_bytes)})", "grey70"),
            ("  ·  ", "grey50"),
            (f"to download {num_to_download} ({human_bytes(to_download_bytes)})", "grey70"),
            ("  ·  ", "grey50"),
            (f"total {human_bytes(total_bytes)}", "grey70"),
            ("  ·  ", "grey50"),
            (f"free {human_bytes(self._free_disk_bytes)}", "grey70"),
            ("  ·  ", "grey50"),
        )
        if fits:
            footer.append("fits", style="green")
        else:
            shortfall = to_download_bytes - (self._free_disk_bytes or 0)
            footer.append(f"OVER BUDGET by {human_bytes(shortfall)}", style="bold red")
        return footer

    def _show_detail(self, row: int) -> None:
        """Show the model at table ``row`` in the detail panel."""
        body = self.query_one("#picker-detail-body", Static)
        if 0 <= row < len(self._visible):
            self._current = self._visible[row]
            body.update(self._detail_for(self._current))
        else:
            self._current = None
            body.update("No models match the current filters.")

    @staticmethod
    def _detail_for(model: ModelInfo) -> RenderableType:
        """The full record for the detail panel, including a clickable homepage link."""
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="bold cyan", justify="right", no_wrap=True)
        grid.add_column()
        grid.add_row("Baseline", model.baseline or "-")
        disk_text = (
            Text("on disk (unverified)", style="green")
            if model.on_disk
            else Text(f"will download · {human_bytes(model.size_on_disk_bytes)}", style="yellow")
        )
        grid.add_row("Disk", disk_text)
        if model.size_on_disk_bytes:
            grid.add_row("Size", human_bytes(model.size_on_disk_bytes))
        if model.target_path:
            grid.add_row("Path", Text(model.target_path, style="grey70"))
        if model.style:
            grid.add_row("Style", model.style)
        if model.version:
            grid.add_row("Version", model.version)
        grid.add_row("NSFW", "yes" if model.nsfw else "no")
        grid.add_row("Inpainting", "yes" if model.inpainting else "no")
        if model.tags:
            grid.add_row("Tags", ", ".join(model.tags))
        if model.trigger:
            grid.add_row("Triggers", ", ".join(model.trigger))

        parts: list[RenderableType] = [Text(model.name, style="bold"), grid]
        if model.description:
            parts.append(Text(""))
            parts.append(Text(model.description, style="grey70"))
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

    def _toggle(self, row: int) -> None:
        """Mark/unmark the model at table ``row`` for adding, updating its state cell."""
        if not 0 <= row < len(self._visible):
            return
        model = self._visible[row]
        if model.name in self._chosen:
            self._chosen.discard(model.name)
            state = _ADD
        else:
            self._chosen.add(model.name)
            state = _REMOVE
        self.query_one("#picker-table", DataTable).update_cell_at(Coordinate(row, 0), state)
        self._refresh_status()

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        """Update the detail panel as the cursor moves (click or keyboard)."""
        self._show_detail(event.coordinate.row)

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        """Clicking the leading ＋/✕ cell adds or removes the model; other cells only inspect."""
        if event.coordinate.column == 0:
            self._toggle(event.coordinate.row)

    def action_toggle_add(self) -> None:
        """Add/remove the highlighted model (keyboard)."""
        self._toggle(self.query_one("#picker-table", DataTable).cursor_coordinate.row)

    def action_open_homepage(self) -> None:
        """Open the highlighted model's homepage in the browser, if it has one."""
        if self._current is not None and self._current.homepage:
            webbrowser.open(self._current.homepage)
            self.notify(f"Opening {self._current.homepage}")
        else:
            self.notify("This model has no homepage.", severity="warning")

    def action_cancel(self) -> None:
        """Dismiss without adding anything."""
        self.dismiss(None)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Re-filter when the baseline changes."""
        if event.select.id == "picker-baseline":
            self._apply_filters()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Re-filter when a toggle changes."""
        self._apply_filters()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Re-filter as the search text changes."""
        if event.input.id == "picker-search":
            self._apply_filters()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Return the marked names on Add, or cancel."""
        if event.button.id == "picker-cancel":
            self.dismiss(None)
        elif event.button.id == "picker-add":
            self.dismiss(sorted(self._chosen))
