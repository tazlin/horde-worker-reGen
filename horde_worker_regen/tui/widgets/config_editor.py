"""The config editor: a full form over bridgeData.yaml with model controls and bounds validation."""

from __future__ import annotations

import contextlib
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Checkbox, Input, Label, Rule, Static, Switch, TabbedContent, TabPane, TextArea

from horde_worker_regen.tui.config_form import (
    CONFIG_FIELDS,
    CONFIG_SUBTABS,
    DEFAULT_CONFIG_PATH,
    MODELS_TO_LOAD_KEY,
    MODELS_TO_SKIP_KEY,
    SECTION_GUIDANCE,
    ConfigField,
    FieldKind,
    coerce_value,
    current_value,
    load_config,
    save_config,
)
from horde_worker_regen.tui.widgets.model_list_editor import ModelListEditor
from horde_worker_regen.tui.widgets.model_manager import ModelManagerView

_FIELD_BY_KEY = {field.key: field for field in CONFIG_FIELDS}
_MODELS_SECTION = "Models"


def _subtab_id(label: str) -> str:
    """A DOM-safe TabPane id derived from a sub-tab label (e.g. ``LoRA/Alchemy`` -> ``cfgtab-lora-alchemy``)."""
    slug = "".join(char if char.isalnum() else "-" for char in label.lower())
    return f"cfgtab-{slug.strip('-')}"


def _normalise_form(value: str) -> str:
    """Normalise an alchemy form for comparison (the worker lowercases and underscores forms)."""
    return value.replace("-", "_").lower()


class ConfigEditorView(Vertical):
    """Edit bridgeData.yaml through typed widgets; save and (optionally) reload/restart the worker."""

    DEFAULT_CSS = """
    ConfigEditorView #config-subtabs {
        height: 1fr;
    }
    ConfigEditorView .config-subtab-scroll {
        height: 1fr;
        padding: 0 1;
    }
    ConfigEditorView #config-actions {
        height: 3;
        padding: 0 1;
    }
    ConfigEditorView #config-actions Button {
        margin-right: 1;
    }
    ConfigEditorView .config-field {
        height: auto;
        padding: 0 1;
    }
    ConfigEditorView .config-row {
        height: 3;
    }
    ConfigEditorView .config-row Label {
        width: 34;
        content-align: left middle;
        height: 3;
    }
    ConfigEditorView .config-help {
        color: $text-disabled;
        padding-left: 1;
    }
    ConfigEditorView .config-section {
        color: $accent;
        text-style: bold;
        padding: 1 1 0 1;
    }
    ConfigEditorView .config-guidance {
        color: $text-muted;
        padding: 0 1;
    }
    ConfigEditorView TextArea {
        height: 5;
    }
    ConfigEditorView #config-status {
        padding: 0 1;
    }
    """

    class ApplyRequested(Message):
        """Posted after a successful save when the user wants the worker to pick up the change."""

        def __init__(self, *, restart: bool) -> None:
            """Record whether the worker should restart (vs. hot-reload)."""
            super().__init__()
            self.restart = restart

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        """Load the config file so widgets can be pre-filled."""
        super().__init__()
        self._config_path = config_path
        self._data = load_config(config_path)

    def compose(self) -> ComposeResult:
        """Lay out the pinned action bar and status line, then the grouped fields in sub-tabs."""
        with Horizontal(id="config-actions"):
            yield Button("Reload from disk", id="config-reload", variant="default")
            yield Button("Save", id="config-save", variant="primary")
            yield Button("Save + apply", id="config-apply", variant="success")
            yield Button("Save + restart worker", id="config-restart", variant="warning")
        yield Static(f"Editing {self._config_path}  ·  ⟳ marks fields that need a worker restart", id="config-status")

        with TabbedContent(id="config-subtabs"):
            for label, sections in CONFIG_SUBTABS:
                with TabPane(label, id=_subtab_id(label)), VerticalScroll(classes="config-subtab-scroll"):
                    for section in sections:
                        yield from self._compose_section(section)

    def _compose_section(self, section: str) -> ComposeResult:
        """Yield the heading, guidance, and field widgets for one section (the model panel for Models)."""
        if section == _MODELS_SECTION:
            yield Label(section, classes="config-section")
            yield Rule()
            if section in SECTION_GUIDANCE:
                yield Static(SECTION_GUIDANCE[section], classes="config-guidance")
            yield ModelManagerView(
                [str(item) for item in current_value(_FIELD_BY_KEY[MODELS_TO_LOAD_KEY], self._data)],
                [str(item) for item in current_value(_FIELD_BY_KEY[MODELS_TO_SKIP_KEY], self._data)],
                load_large_models=bool(current_value(_FIELD_BY_KEY["load_large_models"], self._data)),
            )
            return

        section_fields = [field for field in CONFIG_FIELDS if field.section == section]
        if not section_fields:
            return
        yield Label(section, classes="config-section")
        yield Rule()
        if section in SECTION_GUIDANCE:
            yield Static(SECTION_GUIDANCE[section], classes="config-guidance")
        for field in section_fields:
            yield self._compose_field(field)

    def _label_text(self, field: ConfigField) -> str:
        """A field label annotated with bounds/unit and the restart marker."""
        bounds = ""
        if field.minimum is not None and field.maximum is not None:
            unit = f" {field.unit}" if field.unit else ""
            bounds = f"  ({field.minimum}–{field.maximum}{unit})"
        elif field.unit:
            bounds = f"  ({field.unit})"
        marker = "  ⟳" if field.requires_restart else ""
        return f"{field.label}{bounds}{marker}"

    def _compose_field(self, field: ConfigField) -> Vertical:
        """Build the widget group for one field, pre-filled from the loaded config."""
        value = current_value(field, self._data)
        label_text = self._label_text(field)
        widget_id = f"cfg-{field.key}"

        if field.kind is FieldKind.BOOL:
            control: Vertical = Vertical(
                Horizontal(Label(label_text), Switch(value=bool(value), id=widget_id), classes="config-row"),
                Static(field.help, classes="config-help"),
                classes="config-field",
            )
        elif field.kind is FieldKind.MODEL_LIST:
            entries = [str(item) for item in value]
            control = Vertical(
                Label(label_text),
                Static(field.help, classes="config-help"),
                ModelListEditor(field.key, entries, show_disk_total=field.key == MODELS_TO_LOAD_KEY),
                classes="config-field",
            )
        elif field.kind is FieldKind.SELECT_MULTI:
            chosen = {_normalise_form(str(item)) for item in value}
            checkboxes = [
                Checkbox(choice, value=_normalise_form(choice) in chosen, id=f"cfg-{field.key}-{choice}")
                for choice in field.choices
            ]
            control = Vertical(
                Label(label_text),
                Static(field.help, classes="config-help"),
                Horizontal(*checkboxes, classes="config-row"),
                classes="config-field",
            )
        elif field.kind is FieldKind.STR_LIST:
            text = "\n".join(str(item) for item in value)
            control = Vertical(
                Label(label_text),
                Static(field.help, classes="config-help"),
                TextArea(text=text, id=widget_id),
                classes="config-field",
            )
        else:
            input_type = "integer" if field.kind is FieldKind.INT else "text"
            field_input = Input(
                value=str(value),
                id=widget_id,
                type=input_type,  # type: ignore[arg-type]
                password=field.secret,
            )
            control = Vertical(
                Horizontal(Label(label_text), field_input, classes="config-row"),
                Static(field.help, classes="config-help"),
                classes="config-field",
            )
        return control

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dispatch the action-bar buttons (model-list buttons are handled by their own editor)."""
        if event.button.id == "config-reload":
            self._reload_from_disk()
        elif event.button.id == "config-save":
            self._save()
        elif event.button.id in ("config-apply", "config-restart") and self._save():
            self.post_message(self.ApplyRequested(restart=event.button.id == "config-restart"))

    def update_worker_models(self, active_models: list[str]) -> None:
        """Forward the running worker's currently-loaded model list to the model panel (best-effort)."""
        with contextlib.suppress(Exception):
            self.query_one(ModelManagerView).update_worker_models(active_models)

    def reload_from_disk(self) -> None:
        """Re-read the config file and refresh every widget (e.g. after the setup wizard writes it)."""
        self._reload_from_disk()

    def _reload_from_disk(self) -> None:
        """Re-read the file and push values back into the widgets."""
        self._data = load_config(self._config_path)
        for field in CONFIG_FIELDS:
            value = current_value(field, self._data)
            if field.kind is FieldKind.MODEL_LIST:
                self.query_one(f"#mle-root-{field.key}", ModelListEditor).set_values([str(item) for item in value])
            elif field.kind is FieldKind.SELECT_MULTI:
                chosen = {_normalise_form(str(item)) for item in value}
                for choice in field.choices:
                    self.query_one(f"#cfg-{field.key}-{choice}", Checkbox).value = _normalise_form(choice) in chosen
            else:
                widget = self.query_one(f"#cfg-{field.key}")
                if isinstance(widget, Switch):
                    widget.value = bool(value)
                elif isinstance(widget, TextArea):
                    widget.text = "\n".join(str(item) for item in value)
                elif isinstance(widget, Input):
                    widget.value = str(value)
        self._set_status("Reloaded from disk.", "green")

    def _save(self) -> bool:
        """Collect widget values into the YAML and write it. Returns False on a validation error."""
        try:
            updates = self._collect_values()
        except ValueError as error:
            self._set_status(str(error), "red")
            return False

        for field, value in updates:
            present = self._key_present(field.key)
            if present or value != field.default():
                self._data[field.key] = value

        try:
            save_config(self._data, self._config_path)
        except OSError as error:
            self._set_status(f"Failed to write {self._config_path}: {error}", "red")
            return False
        self._set_status(f"Saved {self._config_path}.", "green")
        return True

    def _collect_values(self) -> list[tuple[ConfigField, object]]:
        """Read and coerce every field's widget value (raises ValueError on bad input)."""
        collected: list[tuple[ConfigField, object]] = []
        for field in CONFIG_FIELDS:
            if field.kind is FieldKind.MODEL_LIST:
                editor = self.query_one(f"#mle-root-{field.key}", ModelListEditor)
                collected.append((field, coerce_value(field, editor.values())))
                continue
            if field.kind is FieldKind.SELECT_MULTI:
                chosen = [
                    choice for choice in field.choices if self.query_one(f"#cfg-{field.key}-{choice}", Checkbox).value
                ]
                collected.append((field, coerce_value(field, chosen)))
                continue

            widget = self.query_one(f"#cfg-{field.key}")
            if isinstance(widget, Switch):
                raw: object = widget.value
            elif isinstance(widget, TextArea):
                raw = widget.text
            elif isinstance(widget, Input):
                raw = widget.value
            else:
                continue
            collected.append((field, coerce_value(field, raw)))
        return collected

    def _key_present(self, key: str) -> bool:
        """Whether a key already exists in the loaded YAML mapping."""
        try:
            return key in self._data
        except TypeError:
            return False

    def _set_status(self, message: str, colour: str) -> None:
        """Update the status line with a coloured message."""
        self.query_one("#config-status", Static).update(f"[{colour}]{message}[/]")
