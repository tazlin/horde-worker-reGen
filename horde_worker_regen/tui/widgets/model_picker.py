"""A modal that browses image models in a table: click a row to mark/unmark it for adding."""

from __future__ import annotations

import webbrowser
from typing import TYPE_CHECKING

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.coordinate import Coordinate
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, DataTable, Footer, Input, Label, Select, Static
from textual.widgets.data_table import ColumnKey

from horde_worker_regen.tui.catalog_cache import CATALOG_CACHE
from horde_worker_regen.tui.formatters import human_bytes
from horde_worker_regen.tui.model_catalog import ModelInfo, friendly_baseline

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

_ADD = "＋ Mark"
_REMOVE = "✕ Unmark"
_NAME_WIDTH = 56

# (header, width) for every table column; the index also drives sorting (see ``_sort_value``).
_MARKER_COL = 0
_COLUMNS: tuple[tuple[str, int], ...] = (
    (" ", 10),  # mark state: ＋ Mark (addable) / ✕ Unmark (marked) / ✓ in <list> (already present)
    ("Model", _NAME_WIDTH),
    ("In config", 10),
    ("Baseline", 10),
    ("Disk", 10),
    ("Flags", 12),
)

_DISK_ALL = ""
_DISK_ON = "on"
_DISK_OFF = "off"


class _PickerTable(DataTable):
    """The picker's model table, emitting a single-click toggle the modal can act on.

    ``DataTable._on_click`` calls ``event.stop()`` on every cell-cursor click, so a ``Click`` handler
    on an ancestor never sees the event; and its built-in ``CellSelected`` only fires on a click of the
    *already-highlighted* cell (a de-facto double click for any other row). To get a true single-click
    toggle we read the clicked row from the event's render metadata, defer to the base handler so the
    cursor, scroll, and highlight all behave normally, then post our own message.
    """

    class RowToggled(Message):
        """Posted when a single click lands on a model row (not the header)."""

        def __init__(self, row: int) -> None:
            self.row = row
            super().__init__()

    async def _on_click(self, event: events.Click) -> None:
        meta = event.style.meta
        row = meta.get("row", -1)
        column = meta.get("column", -1)
        await super()._on_click(event)
        if row >= 0 and column >= 0:
            self.post_message(self.RowToggled(row))


class ModelPickerModal(ModalScreen[list[str] | None]):
    """Browse and mark image models to add; dismisses with the chosen names or None."""

    BINDINGS = [
        Binding("space", "toggle_add", "Mark / unmark"),
        Binding("o", "open_homepage", "Open homepage"),
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    ModelPickerModal {
        align: center middle;
    }
    ModelPickerModal #picker-dialog {
        width: 90%;
        max-width: 200;
        height: 90%;
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
        width: 28;
        margin-right: 1;
    }
    ModelPickerModal #picker-disk-filter {
        width: 24;
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

    def __init__(
        self,
        exclude: set[str] | None = None,
        *,
        in_target: set[str] | None = None,
        in_other: set[str] | None = None,
        target_label: str = "load",
        other_label: str = "skip",
    ) -> None:
        """Create the picker.

        Two display modes share this modal:

        - **Membership** (``in_target`` / ``in_other`` given): every model stays visible and an
          "In config" column shows whether it is already in the list being edited (``in_target``) or
          the sibling list (``in_other``). Models already in ``in_target`` are shown as present and are
          not re-addable; marking any other model adds it (the delta). This is what the config editor
          uses so the resulting set is obvious at a glance.
        - **Hide** (``exclude`` given): the supplied names are hidden from the table. The first-run
          wizard uses this, since it has a single list and replaces the selection wholesale.
        """
        super().__init__()
        self._exclude = exclude or set()
        self._in_target = in_target or set()
        self._in_other = in_other or set()
        self._target_label = target_label
        self._other_label = other_label
        self._all_models: list[ModelInfo] = []
        self._by_name: dict[str, ModelInfo] = {}
        self._visible: list[ModelInfo] = []
        self._chosen: set[str] = set()
        self._current: ModelInfo | None = None
        self._loaded = False
        self._free_disk_bytes: int | None = None
        self._weights_root: str | None = None
        self._col_keys: list[ColumnKey] = []
        self._sort_index = 1  # Model name, ascending, by default.
        self._sort_reverse = False

    def compose(self) -> ComposeResult:
        """Lay out the filters, search, model table with a detail panel, and buttons."""
        with Vertical(id="picker-dialog"):
            yield Label(
                "Click a row to mark/unmark it  ·  click a header to sort",
                classes="dialog-title",
            )
            with Horizontal(id="picker-filters"):
                yield Select([("All baselines", "")], value="", allow_blank=False, id="picker-baseline")
                yield Select(
                    [("Disk: all", _DISK_ALL), ("On disk", _DISK_ON), ("Not on disk", _DISK_OFF)],
                    value=_DISK_ALL,
                    allow_blank=False,
                    id="picker-disk-filter",
                )
                yield Checkbox("Show NSFW", value=True, id="picker-nsfw")
                yield Checkbox("Inpainting only", value=False, id="picker-inpaint")
                yield Checkbox("Marked only", value=False, id="picker-marked-only")
            yield Input(placeholder="search name, description, tags…", id="picker-search")
            with Horizontal(id="picker-body"):
                yield _PickerTable(id="picker-table", cursor_type="cell", zebra_stripes=True)
                with VerticalScroll(id="picker-detail"):
                    yield Static("Loading model reference…", id="picker-detail-body")
            yield Static("Loading model reference…", id="picker-status")
            yield Static("", id="picker-disk")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Add marked", variant="primary", id="picker-add")
                yield Button("Mark all shown", id="picker-mark-all")
                yield Button("Clear marks", id="picker-clear-marks")
                yield Button("Refresh", id="picker-refresh")
                yield Button("Cancel", id="picker-cancel")
            yield Footer()

    def on_mount(self) -> None:
        """Set up the table columns and load the model reference off the UI thread."""
        table = self.query_one("#picker-table", DataTable)
        self._col_keys = [table.add_column(header, width=width) for header, width in _COLUMNS]
        self.run_worker(self._load_models, thread=True, exclusive=True)

    def _load_models(self, *, force: bool = False) -> None:
        """Load the model reference from the shared cache (runs in a worker thread).

        Reads the warm cache, so unless ``force`` is set this usually returns instantly with what the
        startup warm (or another view) already loaded, rather than re-fetching over the network.
        """
        try:
            snapshot = CATALOG_CACHE.ensure_loaded(force=force)
        except Exception as error:  # noqa: BLE001 - surface any loader failure to the user
            self.app.call_from_thread(self._on_load_error, f"{type(error).__name__}: {error}")
            return
        self.app.call_from_thread(
            self._on_models_loaded, snapshot.catalog or [], snapshot.free_disk_bytes, snapshot.weights_root
        )

    def _on_models_loaded(
        self, models: list[ModelInfo], free_disk_bytes: int | None, weights_root: str | None
    ) -> None:
        """Populate the baseline filter and the table once models arrive."""
        self._all_models = models
        self._by_name = {model.name: model for model in models}
        self._free_disk_bytes = free_disk_bytes
        self._weights_root = weights_root
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
        if model.is_beta:
            flags.append("beta")
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

    def _marker_for(self, model: ModelInfo) -> str:
        """The leading-cell marker: already in the target list, marked to add, or addable."""
        if model.name in self._in_target:
            return f"✓ in {self._target_label}"
        return _REMOVE if model.name in self._chosen else _ADD

    def _in_config_cell(self, model: ModelInfo) -> Text:
        """Whether the model is already in the list being edited or in the sibling list."""
        if model.name in self._in_target:
            return Text(self._target_label, style="green")
        if model.name in self._in_other:
            return Text(self._other_label, style="red dim")
        return Text("-", style="grey50")

    def cells_for(self, model: ModelInfo) -> tuple[str, str, Text, str, Text, str]:
        """The six table cells for a model: marker, name, in-config, baseline, disk, flags."""
        name = model.name if len(model.name) <= _NAME_WIDTH else model.name[: _NAME_WIDTH - 1] + "…"
        return (
            self._marker_for(model),
            name,
            self._in_config_cell(model),
            friendly_baseline(model.baseline) or "-",
            self._disk_cell(model),
            self._flags_for(model),
        )

    def _matches_search(self, model: ModelInfo, search: str) -> bool:
        """Case-insensitive substring match across name, description, tags, and triggers."""
        if not search:
            return True
        haystack = " ".join(
            (model.name, model.description, " ".join(model.tags), " ".join(model.trigger)),
        ).lower()
        return search in haystack

    def _sort_value(self, model: ModelInfo) -> SupportsRichComparison:
        """The sort key for a model under the active column (index ``self._sort_index``)."""
        name = model.name.lower()
        if self._sort_index == _MARKER_COL:
            chosen = model.name in self._chosen or model.name in self._in_target
            return (not chosen, name)
        if self._sort_index == 2:  # In config
            rank = 0 if model.name in self._in_target else 1 if model.name in self._in_other else 2
            return (rank, name)
        if self._sort_index == 3:  # Baseline
            return (friendly_baseline(model.baseline).lower(), name)
        if self._sort_index == 4:  # Disk: on-disk first, then larger first.
            return (0 if model.on_disk else 1, -(model.size_on_disk_bytes or 0), name)
        if self._sort_index == 5:  # Flags
            return (self._flags_for(model), name)
        return name  # Model name (default).

    def _apply_filters(self) -> None:
        """Rebuild the table (and the parallel visible list) from the current filters and sort."""
        if not self._loaded:
            return
        baseline = str(self.query_one("#picker-baseline", Select).value)
        disk_filter = str(self.query_one("#picker-disk-filter", Select).value)
        show_nsfw = self.query_one("#picker-nsfw", Checkbox).value
        inpaint_only = self.query_one("#picker-inpaint", Checkbox).value
        marked_only = self.query_one("#picker-marked-only", Checkbox).value
        search = self.query_one("#picker-search", Input).value.strip().lower()

        self._visible = [
            model
            for model in self._all_models
            if model.name not in self._exclude
            and (not baseline or model.baseline == baseline)
            and (disk_filter == _DISK_ALL or model.on_disk == (disk_filter == _DISK_ON))
            and (show_nsfw or not model.nsfw)
            and (not inpaint_only or model.inpainting)
            and (not marked_only or model.name in self._chosen)
            and self._matches_search(model, search)
        ]
        self._visible.sort(key=self._sort_value, reverse=self._sort_reverse)

        table = self.query_one("#picker-table", DataTable)
        table.clear()
        for model in self._visible:
            table.add_row(*self.cells_for(model))
        self._update_header_labels()

        self._refresh_status()
        self._show_detail(0)

    def _update_header_labels(self) -> None:
        """Append a ▲/▼ caret to the active sort column's header (and clear the others)."""
        table = self.query_one("#picker-table", DataTable)
        caret = " ▼" if self._sort_reverse else " ▲"
        for index, key in enumerate(self._col_keys):
            base = _COLUMNS[index][0]
            label = f"{base}{caret}" if index == self._sort_index else base
            table.columns[key].label = Text(label)
        table.refresh()

    def _refresh_status(self) -> None:
        """Update the status line with the visible, marked, and already-in-list counts."""
        text = Text.assemble(
            (f"{len(self._visible)} shown (of {len(self._all_models)})", "grey70"),
            ("  ·  ", "grey50"),
            (f"{len(self._chosen)} marked to add", "grey70"),
        )
        if self._in_target:
            text.append("  ·  ", style="grey50")
            text.append(f"{len(self._in_target)} already in {self._target_label}", style="grey62")
        self.query_one("#picker-status", Static).update(text)
        self.query_one("#picker-disk", Static).update(self._disk_footer())

    def _disk_footer(self) -> Text:
        """A disk-budget line for the resulting list (its current members plus the marked additions)."""
        base = self._in_target if self._in_target else self._exclude
        present_bytes = 0
        to_download_bytes = 0
        num_present = 0
        num_to_download = 0
        for name in base | self._chosen:
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
        if self._weights_root:
            footer.append("\nmodels dir: ", style="grey50")
            footer.append(self._weights_root, style="grey62")
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
        if model.is_beta:
            grid.add_row("Source", Text("beta (PRIMARY pending queue)", style="magenta"))
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
        """Mark/unmark the model at table ``row`` for adding, updating its marker cell.

        A model already in the target list is shown as present and cannot be re-added here, so it is
        left untouched.
        """
        if not 0 <= row < len(self._visible):
            return
        model = self._visible[row]
        if model.name in self._in_target:
            return
        if model.name in self._chosen:
            self._chosen.discard(model.name)
        else:
            self._chosen.add(model.name)
        self.query_one("#picker-table", DataTable).update_cell_at(
            Coordinate(row, _MARKER_COL), self._marker_for(model)
        )
        self._refresh_status()

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        """Update the detail panel as the cursor moves (click or keyboard)."""
        self._show_detail(event.coordinate.row)

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Sort by the clicked column; clicking the active column flips the direction."""
        if event.column_index == self._sort_index:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_index = event.column_index
            self._sort_reverse = False
        self._apply_filters()

    @on(_PickerTable.RowToggled)
    def _on_row_toggled(self, event: _PickerTable.RowToggled) -> None:
        """Toggle the clicked model (single click), via the table's own toggle message."""
        self._toggle(event.row)

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
        """Re-filter when the baseline or disk filter changes."""
        self._apply_filters()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Re-filter when a toggle changes."""
        self._apply_filters()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Re-filter as the search text changes."""
        if event.input.id == "picker-search":
            self._apply_filters()

    def _mark_all_shown(self) -> None:
        """Mark every currently-visible, still-addable model for adding."""
        for model in self._visible:
            if model.name not in self._in_target:
                self._chosen.add(model.name)
        self._apply_filters()

    def _clear_marks(self) -> None:
        """Unmark every model marked for adding."""
        self._chosen.clear()
        self._apply_filters()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Return the marked names on Add, bulk-mark/clear, refresh the catalog, or cancel."""
        if event.button.id == "picker-cancel":
            self.dismiss(None)
        elif event.button.id == "picker-add":
            self.dismiss(sorted(self._chosen))
        elif event.button.id == "picker-mark-all":
            self._mark_all_shown()
        elif event.button.id == "picker-clear-marks":
            self._clear_marks()
        elif event.button.id == "picker-refresh":
            self.query_one("#picker-status", Static).update("Refreshing model reference…")
            self.run_worker(lambda: self._load_models(force=True), thread=True, exclusive=True)
