"""An editor for a models_to_load / models_to_skip list: entries plus picker and meta-builder buttons."""

from __future__ import annotations

from collections.abc import Callable

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Static

from horde_worker_regen.tui.formatters import human_bytes
from horde_worker_regen.tui.model_catalog import DiskSummary, disk_summary, is_meta_instruction
from horde_worker_regen.tui.widgets.confirm_modal import ConfirmModal
from horde_worker_regen.tui.widgets.meta_builder import MetaBuilderModal
from horde_worker_regen.tui.widgets.model_picker import ModelPickerModal, ModelPickerResult


class ModelListEditor(Vertical):
    """Manage a list of model names and meta instructions with add/remove and the picker/meta modals."""

    class Changed(Message):
        """Posted whenever the entry list changes, so a parent can recompute a derived view."""

        def __init__(self, key: str) -> None:
            """Record which list (``models_to_load`` / ``models_to_skip``) changed."""
            super().__init__()
            self.key = key

    DEFAULT_CSS = """
    ModelListEditor {
        height: auto;
        border: round $foreground 20%;
        padding: 0 1;
        margin-bottom: 1;
    }
    ModelListEditor .mle-entries {
        height: auto;
        max-height: 10;
        overflow-y: auto;
    }
    ModelListEditor .mle-row {
        height: 1;
        width: auto;
    }
    ModelListEditor .mle-remove {
        min-width: 3;
        width: 3;
        height: 1;
        border: none;
        color: $error;
        background: transparent;
    }
    ModelListEditor .mle-entry {
        width: auto;
        height: 1;
        padding-left: 1;
        content-align: left middle;
    }
    ModelListEditor .mle-empty {
        color: $text-disabled;
    }
    ModelListEditor .mle-disk {
        height: auto;
        color: $text-muted;
        padding-top: 1;
    }
    ModelListEditor .mle-controls {
        height: 3;
    }
    ModelListEditor .mle-controls Input {
        width: 1fr;
    }
    ModelListEditor .mle-controls Button {
        margin-left: 1;
    }
    ModelListEditor .mle-controls-secondary Button {
        margin-right: 1;
    }
    """

    def __init__(
        self,
        key: str,
        values: list[str],
        *,
        show_disk_total: bool = False,
        sibling_values: Callable[[], list[str]] | None = None,
    ) -> None:
        """Create the editor for ``key`` pre-filled with ``values``.

        When ``show_disk_total`` is set (the models-to-load list), a disk-budget line is shown once the
        model reference has been loaded (by opening the picker); it never forces a network fetch itself.

        ``sibling_values`` (when given) returns the other list's current entries (load vs skip), so the
        picker can show membership of both lists and make the resulting delta obvious. Without it the
        picker falls back to simply hiding this list's entries.
        """
        super().__init__(id=f"mle-root-{key}")
        self._key = key
        self._values = list(values)
        self._show_disk_total = show_disk_total
        self._sibling_values = sibling_values

    def compose(self) -> ComposeResult:
        """Lay out the entries container and the (two-row) control area."""
        yield Vertical(classes="mle-entries", id=f"mle-entries-{self._key}")
        with Horizontal(classes="mle-controls"):
            yield Input(placeholder="type a model name or meta command…", id=f"mle-input-{self._key}")
            yield Button("Add", id=f"mle-add-{self._key}")
        with Horizontal(classes="mle-controls mle-controls-secondary"):
            yield Button("Pick models…", id=f"mle-pick-{self._key}", variant="default")
            yield Button("Build meta…", id=f"mle-meta-{self._key}", variant="default")
            yield Button("Clear", id=f"mle-clear-{self._key}", variant="warning")
        if self._show_disk_total:
            yield Static("", classes="mle-disk", id=f"mle-disk-{self._key}")

    def on_mount(self) -> None:
        """Render the initial entries."""
        self._rebuild()

    def values(self) -> list[str]:
        """The current list of entries."""
        return list(self._values)

    def set_values(self, values: list[str]) -> None:
        """Replace all entries (used on reload-from-disk)."""
        self._values = list(values)
        self._rebuild()

    def add_entry(self, entry: str) -> None:
        """Append one entry if non-empty and not already present."""
        candidate = entry.strip()
        if candidate and candidate not in self._values:
            self._values.append(candidate)
            self._rebuild()

    def add_entries(self, entries: list[str]) -> None:
        """Append several entries, de-duplicating."""
        changed = False
        for entry in entries:
            candidate = entry.strip()
            if candidate and candidate not in self._values:
                self._values.append(candidate)
                changed = True
        if changed:
            self._rebuild()

    def remove_entries(self, entries: list[str]) -> None:
        """Remove exact entries from the list."""
        remove_set = set(entries)
        if not remove_set:
            return
        new_values = [entry for entry in self._values if entry not in remove_set]
        if new_values != self._values:
            self._values = new_values
            self._rebuild()

    def _rebuild(self) -> None:
        """Re-render the entry rows from the current values."""
        container = self.query_one(f"#mle-entries-{self._key}", Vertical)
        container.remove_children()
        if not self._values:
            container.mount(
                Static("(none; add models, pick from the reference, or build a meta command)", classes="mle-empty")
            )
        else:
            for index, entry in enumerate(self._values):
                meta = is_meta_instruction(entry)
                label = Text.assemble(("⚙ " if meta else ""), (entry, "cyan" if meta else "white"))
                row = Horizontal(
                    Button("✕", id=f"mle-rm-{self._key}-{index}", classes="mle-remove"),
                    Static(label, classes="mle-entry"),
                    classes="mle-row",
                )
                container.mount(row)
        self._update_disk_total()
        self.post_message(self.Changed(self._key))

    def _update_disk_total(self) -> None:
        """Recompute the disk-budget line off the UI thread (only when enabled)."""
        if not self._show_disk_total:
            return
        self.run_worker(self._compute_disk_total, thread=True, exclusive=True, group=f"mle-disk-{self._key}")

    def _compute_disk_total(self) -> None:
        """Compute the disk summary in a worker thread, then render it on the UI thread."""
        summary = disk_summary(list(self._values))
        self.app.call_from_thread(self._render_disk_total, summary)

    def _render_disk_total(self, summary: DiskSummary | None) -> None:
        """Render the disk-budget line, or a hint when the reference is not loaded yet."""
        try:
            line = self.query_one(f"#mle-disk-{self._key}", Static)
        except Exception:  # noqa: BLE001 - the widget may have been torn down mid-compute
            return
        if summary is None:
            line.update(Text("disk total: open the picker once to load model sizes", style="grey50"))
            return
        text = Text.assemble(
            ("disk  ", "grey50"),
            (f"on disk {summary.num_present} ({human_bytes(summary.present_bytes)})", "grey70"),
            ("  ·  ", "grey50"),
            (f"to download {summary.num_to_download} ({human_bytes(summary.to_download_bytes)})", "grey70"),
            ("  ·  ", "grey50"),
            (f"total {human_bytes(summary.total_bytes)}", "grey70"),
            ("  ·  ", "grey50"),
            (f"free {human_bytes(summary.free_disk_bytes)}", "grey70"),
            ("  ·  ", "grey50"),
        )
        text.append(
            "fits" if summary.fits else f"OVER BUDGET by {human_bytes(summary.shortfall_bytes)}",
            style="green" if summary.fits else "bold red",
        )
        if summary.num_unsized:
            text.append(f"  ·  {summary.num_unsized} entries not sized (meta/unknown)", style="yellow")
        line.update(text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle this editor's buttons (add/pick/meta/clear/remove); never bubble to the form."""
        button_id = event.button.id or ""
        if button_id == f"mle-add-{self._key}":
            field_input = self.query_one(f"#mle-input-{self._key}", Input)
            self.add_entry(field_input.value)
            field_input.value = ""
        elif button_id == f"mle-pick-{self._key}":
            self._open_picker()
        elif button_id == f"mle-meta-{self._key}":
            self.app.push_screen(MetaBuilderModal(), self._on_meta_result)
        elif button_id == f"mle-clear-{self._key}":
            self._confirm_clear()
        elif button_id.startswith(f"mle-rm-{self._key}-"):
            index = int(button_id.rsplit("-", 1)[1])
            if 0 <= index < len(self._values):
                self._values.pop(index)
                self._rebuild()
        event.stop()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Add the typed entry when Enter is pressed in the input."""
        if event.input.id == f"mle-input-{self._key}":
            self.add_entry(event.value)
            event.input.value = ""
            event.stop()

    def _open_picker(self) -> None:
        """Open the model picker, showing both lists' membership when a sibling provider is set."""
        if self._sibling_values is None:
            self.app.push_screen(ModelPickerModal(set(self._values)), self._on_picker_result)
            return
        target_label, other_label = ("skip", "load") if self._key == "models_to_skip" else ("load", "skip")
        in_target = {entry for entry in self._values if not is_meta_instruction(entry)}
        in_other = {entry for entry in self._sibling_values() if not is_meta_instruction(entry)}
        self.app.push_screen(
            ModelPickerModal(
                in_target=in_target,
                in_other=in_other,
                target_label=target_label,
                other_label=other_label,
            ),
            self._on_picker_result,
        )

    def _on_picker_result(self, result: ModelPickerResult | list[str] | None) -> None:
        """Apply model-list edits chosen in the picker modal."""
        if isinstance(result, ModelPickerResult):
            self.remove_entries(result.remove)
            self.add_entries(result.add)
        elif result:
            self.add_entries(result)

    def _confirm_clear(self) -> None:
        """Ask before clearing all entries."""
        if not self._values:
            return
        self.app.push_screen(
            ConfirmModal(
                f"Clear every entry from {self._display_name()}? This removes unsaved model-rule edits in this list.",
                confirm_label="Yes, clear list",
                cancel_label="No, keep list",
            ),
            self._on_clear_confirmed,
        )

    def _on_clear_confirmed(self, confirmed: bool) -> None:
        """Clear only after confirmation."""
        if confirmed:
            self.set_values([])

    def _display_name(self) -> str:
        """Human label for confirmation messages."""
        return "Offer list" if self._key == "models_to_load" else "Exclusion list"

    def _on_meta_result(self, result: str | None) -> None:
        """Add a meta instruction built in the meta-builder modal."""
        if result:
            self.add_entry(result)
