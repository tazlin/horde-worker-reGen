"""The config editor: a full form over bridgeData.yaml with model controls and bounds validation."""

from __future__ import annotations

import contextlib
import enum
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    Rule,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
    TextArea,
)

from horde_worker_regen.app_state import OverviewViewMode
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
    describe_process_plan,
    format_number,
    format_yaml_value,
    load_config,
    read_gpu_device_indices,
    save_config,
    validate_identity_names,
)
from horde_worker_regen.tui.config_presets import BUILT_IN_PRESETS, ConfigPreset, PresetChange, diff_preset
from horde_worker_regen.tui.config_validation import ConfigValidationSeverity, validate_config_interlocks
from horde_worker_regen.tui.widgets.custom_model_builder import CustomModelBuilderModal, CustomModelBuilderResult
from horde_worker_regen.tui.widgets.gpu_overrides_editor import GpuOverridesEditor
from horde_worker_regen.tui.widgets.model_list_editor import ModelListEditor
from horde_worker_regen.tui.widgets.model_manager import ModelManagerView

if TYPE_CHECKING:
    from horde_worker_regen.process_management.ipc.supervisor_channel import CardSnapshot

_FIELD_BY_KEY = {field.key: field for field in CONFIG_FIELDS}
_MODELS_SECTION = "Models"
# The section whose fields drive the (non-obvious) inference-process count; a live preview is appended here.
_THROUGHPUT_SECTION = "Throughput"
_PROCESS_PREVIEW_ID = "config-process-preview"

# The per-card multi-GPU editor is its own sub-tab, mounted after the catalog-driven tabs rather than
# composed from flat ConfigFields (it edits the nested gpu_overrides block, not top-level keys).
_GPU_SUBTAB_LABEL = "Per-GPU"
_GPU_SUBTAB_ID = "cfgtab-per-gpu"

# Which sub-tab a section lives on, so a validation error can name (and jump to) the right page.
_SECTION_TO_SUBTAB: dict[str, str] = {section: label for label, sections in CONFIG_SUBTABS for section in sections}
_ADVANCED_SECTIONS: set[str] = {
    section for label, sections in CONFIG_SUBTABS if label == "Advanced" for section in sections
}

# Only fields whose section is bundled into a sub-tab get widgets in ``compose``. A section that is
# absent from ``CONFIG_SUBTABS`` (e.g. the developer-only "Dry-run" flags, which stay editable via YAML
# but are deliberately kept out of the operator UI) is never laid out, so the loops that walk every
# field must iterate this filtered view; querying an uncomposed field raises ``NoMatches``.
_RENDERED_FIELDS: tuple[ConfigField, ...] = tuple(
    field for field in CONFIG_FIELDS if field.section in _SECTION_TO_SUBTAB and not field.hidden
)


def _subtab_id(label: str) -> str:
    """A DOM-safe TabPane id derived from a sub-tab label (e.g. ``Alchemy`` -> ``cfgtab-alchemy``)."""
    slug = "".join(char if char.isalnum() else "-" for char in label.lower())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return f"cfgtab-{slug.strip('-')}"


def _normalise_form(value: str) -> str:
    """Normalise an alchemy form for comparison (the worker lowercases and underscores forms)."""
    return value.replace("-", "_").lower()


class ConfigLeaveChoice(enum.StrEnum):
    """The user's response to the "you have unsaved config edits" warning when leaving the Config tab."""

    LEAVE = "leave"
    """Navigate away but keep the unsaved edits live in the form (nothing is written to disk)."""
    DISCARD = "discard"
    """Revert the form to the values on disk, then navigate away."""
    STAY = "stay"
    """Cancel the navigation and stay on the Config tab."""
    NEVER = "never"
    """Navigate away and stop warning for the rest of this session."""


class ConfigLeaveModal(ModalScreen[ConfigLeaveChoice]):
    """Warns that the Config tab has unsaved edits before the user navigates away."""

    DEFAULT_CSS = """
    ConfigLeaveModal {
        align: center middle;
    }
    ConfigLeaveModal #config-leave-dialog {
        width: 72;
        height: auto;
        padding: 1 2;
        border: thick $warning;
        background: $surface;
    }
    ConfigLeaveModal #config-leave-dialog Button {
        width: 100%;
        margin-top: 1;
    }
    """

    BINDINGS = [("escape", "stay", "Stay")]

    _BUTTON_CHOICES: dict[str, ConfigLeaveChoice] = {
        "config-leave-keep": ConfigLeaveChoice.LEAVE,
        "config-leave-discard": ConfigLeaveChoice.DISCARD,
        "config-leave-stay": ConfigLeaveChoice.STAY,
        "config-leave-never": ConfigLeaveChoice.NEVER,
    }

    def compose(self) -> ComposeResult:
        """Lay out the warning and the four choices."""
        with Vertical(id="config-leave-dialog"):
            yield Static(self._message(), id="config-leave-message")
            yield Button("Leave (keep my edits in the form)", id="config-leave-keep", variant="primary")
            yield Button("Discard my edits and leave", id="config-leave-discard", variant="warning")
            yield Button("Stay on Config", id="config-leave-stay", variant="default")
            yield Button("Leave and don't warn me again this session", id="config-leave-never", variant="default")

    @staticmethod
    def _message() -> Text:
        """Explain that the edits are not yet saved to disk."""
        return Text.assemble(
            ("Unsaved config edits\n\n", "bold"),
            (
                "You have changes on the Config tab that have not been saved to bridgeData.yaml. "
                "Leaving keeps them in the form for now, but they will be lost if you reload from disk "
                "or close the app. Save them first to apply them to the worker.",
                "grey70",
            ),
        )

    def action_stay(self) -> None:
        """Dismiss as 'stay' when escape is pressed."""
        self.dismiss(ConfigLeaveChoice.STAY)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with the choice the pressed button represents."""
        choice = self._BUTTON_CHOICES.get(event.button.id or "")
        if choice is not None:
            self.dismiss(choice)


class ConfigPresetModal(ModalScreen[dict[str, object] | None]):
    """Preview one built-in preset and let the operator opt out per setting."""

    DEFAULT_CSS = """
    ConfigPresetModal {
        align: center middle;
    }
    ConfigPresetModal #config-preset-dialog {
        width: 90%;
        max-width: 160;
        height: 85%;
        max-height: 46;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    ConfigPresetModal #config-preset-actions {
        height: 3;
    }
    ConfigPresetModal #config-preset-actions Button {
        margin-right: 1;
    }
    ConfigPresetModal .preset-body {
        height: 1fr;
        overflow-x: auto;
    }
    ConfigPresetModal .preset-group {
        color: $accent;
        text-style: bold;
        padding: 1 1 0 1;
        min-width: 140;
    }
    ConfigPresetModal .preset-row {
        height: auto;
        min-width: 140;
        padding: 0 1;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, current: dict[str, object]) -> None:
        """Store the current config values used for the diff preview."""
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        """Build the preset picker, all preset bodies, and actions."""
        first = BUILT_IN_PRESETS[0]
        with Vertical(id="config-preset-dialog"):
            yield Label("Apply configuration preset", classes="config-section")
            yield Select(
                [(preset.label, preset.preset_id) for preset in BUILT_IN_PRESETS],
                value=first.preset_id,
                id="config-preset-select",
            )
            yield Static(first.description, id="config-preset-description", classes="config-help")
            for preset in BUILT_IN_PRESETS:
                with VerticalScroll(id=f"config-preset-body-{preset.preset_id}", classes="preset-body") as body:
                    body.display = preset is first
                    if preset.warnings:
                        yield Static("  ".join(preset.warnings), classes="config-guidance")
                    seen_categories: set[str] = set()
                    for change in preset.changes:
                        category = change.category.value
                        if category not in seen_categories:
                            seen_categories.add(category)
                            yield Label(category, classes="preset-group")
                        yield self._preset_checkbox(preset, change)
            with Horizontal(id="config-preset-actions"):
                yield Button("Apply checked changes", id="config-preset-apply", variant="success")
                yield Button("Cancel", id="config-preset-cancel", variant="default")

    def _preset_checkbox(self, preset: ConfigPreset, change: PresetChange) -> Checkbox:
        """Create one selectable row for a preset change."""
        current = self._current.get(change.key)
        selected = change.default_selected
        if (
            change.key == "allow_lora"
            and change.value is True
            and not str(self._current.get("civitai_api_token") or "")
        ):
            selected = False
        restart = " restart" if change.requires_restart else ""
        field = _FIELD_BY_KEY.get(change.key)
        label_key = field.label if field is not None else change.key
        label = f"{label_key}: {current!r} -> {change.value!r}{restart} - {change.rationale}"
        checkbox = Checkbox(
            label,
            value=selected,
            id=f"config-preset-{preset.preset_id}-{change.key}",
            classes="preset-row",
        )
        checkbox.disabled = current == change.value
        return checkbox

    def on_select_changed(self, event: Select.Changed) -> None:
        """Show the selected preset body."""
        preset_id = str(event.value)
        for preset in BUILT_IN_PRESETS:
            with contextlib.suppress(Exception):
                self.query_one(f"#config-preset-body-{preset.preset_id}").display = preset.preset_id == preset_id
        with contextlib.suppress(Exception):
            self.query_one("#config-preset-description", Static).update(_preset_by_id(preset_id).description)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Apply the checked changes or cancel."""
        if event.button.id == "config-preset-cancel":
            self.dismiss(None)
            return
        if event.button.id != "config-preset-apply":
            return
        preset_id = str(self.query_one("#config-preset-select", Select).value)
        preset = _preset_by_id(preset_id)
        changes: dict[str, object] = {}
        for diff in diff_preset(preset, self._current):
            checkbox = self.query_one(f"#config-preset-{preset.preset_id}-{diff.change.key}", Checkbox)
            if checkbox.value and diff.changed:
                changes[diff.change.key] = diff.change.value
        self.dismiss(changes)

    def action_cancel(self) -> None:
        """Dismiss without applying changes."""
        self.dismiss(None)


def _preset_by_id(preset_id: str) -> ConfigPreset:
    """Return a built-in preset by id."""
    for preset in BUILT_IN_PRESETS:
        if preset.preset_id == preset_id:
            return preset
    return BUILT_IN_PRESETS[0]


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
    ConfigEditorView #config-actions-spacer {
        width: 1fr;
    }
    ConfigEditorView #config-actions-separator {
        width: 3;
        height: 3;
        content-align: center middle;
        color: $text-muted;
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
    ConfigEditorView .config-process-preview {
        color: $accent;
        text-style: italic;
        padding: 1 1 0 1;
    }
    ConfigEditorView #config-effective-summary {
        padding: 0 1;
        color: $text-muted;
    }
    ConfigEditorView #config-change-summary {
        padding: 0 1;
        color: $accent;
    }
    ConfigEditorView #config-live-warnings {
        padding: 0 1;
        color: $warning;
    }
    ConfigEditorView .config-advanced-banner {
        margin: 0 1 1 1;
        padding: 0 1;
        border-left: thick $warning;
        color: $warning;
        height: auto;
    }
    ConfigEditorView TextArea {
        height: 5;
    }
    ConfigEditorView .config-inline-actions {
        height: 3;
        padding-top: 1;
    }
    ConfigEditorView .config-inline-actions Button {
        margin-right: 1;
    }
    ConfigEditorView #config-status {
        padding: 0 1;
    }
    """

    class ApplyRequested(Message):
        """Posted after a successful save when the worker must be restarted to pick up the change.

        A plain Save does not post this: the worker watches bridgeData.yaml's mtime and hot-reloads on
        its own, so the only reason to message the app is a restart-locked field the watch cannot apply.
        """

        def __init__(self, *, restart: bool) -> None:
            """Record that the worker should restart (only restart is routed through this message)."""
            super().__init__()
            self.restart = restart

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        """Load the config file so widgets can be pre-filled."""
        super().__init__()
        self._config_path = config_path
        self._data = load_config(config_path)
        self._mode = OverviewViewMode.NORMAL
        self._clean_state: dict[str, object] | None = None

    def compose(self) -> ComposeResult:
        """Lay out the pinned action bar and status line, then the grouped fields in sub-tabs."""
        with Horizontal(id="config-actions"):
            yield Button("Reload from disk", id="config-reload", variant="default")
            yield Button("Save", id="config-save", variant="success")
            yield Button("Save + restart worker", id="config-restart", variant="default")
            yield Static("", id="config-actions-spacer")
            yield Static("|", id="config-actions-separator")
            yield Button("Apply preset", id="config-preset", variant="default")
        yield Static(
            f"Editing {self._config_path}  ·  Save applies automatically (the worker watches the file)  ·  "
            "⟳ marks fields that only take effect on restart",
            id="config-status",
        )
        yield Static("", id="config-effective-summary")
        yield Static("", id="config-change-summary")
        yield Static("", id="config-live-warnings")

        with TabbedContent(id="config-subtabs"):
            for label, sections in CONFIG_SUBTABS:
                with TabPane(label, id=_subtab_id(label)), VerticalScroll(classes="config-subtab-scroll"):
                    for section in sections:
                        yield from self._compose_section(section)
            with TabPane(_GPU_SUBTAB_LABEL, id=_GPU_SUBTAB_ID), VerticalScroll(classes="config-subtab-scroll"):
                yield GpuOverridesEditor(self._data)

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
                only_on_disk=bool(current_value(_FIELD_BY_KEY["only_models_on_disk"], self._data)),
            )
            return

        section_fields = [field for field in CONFIG_FIELDS if field.section == section and not field.hidden]
        if not section_fields:
            return
        yield Label(section, classes="config-section")
        yield Rule()
        if section in SECTION_GUIDANCE:
            yield Static(SECTION_GUIDANCE[section], classes="config-guidance")
        if section in _ADVANCED_SECTIONS:
            yield Static(
                "Advanced: change these only when a log message, benchmark, or support instruction points here.",
                classes="config-advanced-banner",
            )
        for field in section_fields:
            yield self._compose_field(field)
        if section == _THROUGHPUT_SECTION:
            # A live estimate of the resulting process count, which is a non-obvious function of these
            # fields (see describe_process_plan). Filled in on mount and refreshed on every edit.
            yield Static("", id=_PROCESS_PREVIEW_ID, classes="config-process-preview")

    def _label_text(self, field: ConfigField) -> str:
        """A field label annotated with bounds/unit and the restart marker."""
        bounds = ""
        if field.minimum is not None and field.maximum is not None:
            unit = f" {field.unit}" if field.unit else ""
            bounds = f"  ({format_number(field.minimum)}–{format_number(field.maximum)}{unit})"
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
        elif field.kind is FieldKind.SELECT:
            selected = str(value)
            if selected not in field.choices and field.choices:
                selected = field.choices[0]
            control = Vertical(
                Horizontal(
                    Label(label_text),
                    Select([(choice, choice) for choice in field.choices], value=selected, id=widget_id),
                    classes="config-row",
                ),
                Static(field.help, classes="config-help"),
                classes="config-field",
            )
        elif field.kind in (FieldKind.STR_LIST, FieldKind.YAML):
            text = format_yaml_value(value) if field.kind is FieldKind.YAML else "\n".join(str(item) for item in value)
            children: list[Widget] = [
                Label(label_text),
                Static(field.help, classes="config-help"),
                TextArea(text=text, id=widget_id),
            ]
            if field.key == "custom_models":
                children.append(
                    Horizontal(
                        Button("Add custom model...", id="cfg-custom_models-add", variant="default"),
                        classes="config-inline-actions",
                    )
                )
            control = Vertical(
                *children,
                classes="config-field",
            )
        else:
            input_type = (
                "text"
                if field.optional
                else {FieldKind.INT: "integer", FieldKind.FLOAT: "number"}.get(
                    field.kind,
                    "text",
                )
            )
            field_input = Input(
                value="" if value is None else str(value),
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

    def on_mount(self) -> None:
        """Capture the loaded values as the clean baseline for unsaved-change detection."""
        with contextlib.suppress(Exception):
            self._clean_state = self._widget_state()
        self._update_process_preview()
        self._refresh_action_variants()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Refresh the process-count preview when any numeric/text field changes."""
        self._update_process_preview()
        self._refresh_action_variants()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Refresh the process-count preview when any toggle changes."""
        self._update_process_preview()
        self._refresh_action_variants()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Refresh restart-action emphasis when a select value changes."""
        self._refresh_action_variants()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Refresh restart-action emphasis when a checkbox changes."""
        self._refresh_action_variants()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Refresh restart-action emphasis when a text area changes."""
        self._refresh_action_variants()

    def on_model_list_editor_changed(self, message: ModelListEditor.Changed) -> None:
        """Refresh restart-action emphasis when a model rule list changes."""
        self._refresh_action_variants()

    def _update_process_preview(self) -> None:
        """Recompute and display the live inference/safety process-count estimate (best-effort).

        Reads the concurrency fields straight from their widgets so an unsaved edit is reflected, and
        pins card count / model list from the loaded config (the multi-GPU block is restart-locked, so
        its on-disk value is authoritative for a preview). Never raises: the preview is advisory.
        """
        try:
            preview = self.query_one(f"#{_PROCESS_PREVIEW_ID}", Static)
        except Exception:  # noqa: BLE001 - queried before mount / on tabs without the field
            return
        max_threads = self._int_widget_value("max_threads")
        queue_size = self._int_widget_value("queue_size")
        extra_slow = self._bool_widget_value("extra_slow_worker")
        dreamer = self._bool_widget_value("dreamer")
        load_entries = self._model_load_entries()
        device_indices = read_gpu_device_indices(self._data)
        with contextlib.suppress(Exception):
            preview.update(
                describe_process_plan(
                    max_threads=max_threads,
                    queue_size=queue_size,
                    load_entries=load_entries,
                    serves_image_generation=dreamer,
                    extra_slow_worker=extra_slow,
                    device_indices=device_indices,
                ),
            )

    def _int_widget_value(self, key: str) -> int:
        """The current integer value of a field's input, falling back to its configured default."""
        default = int(current_value(_FIELD_BY_KEY[key], self._data))
        try:
            return int(str(self.query_one(f"#cfg-{key}", Input).value).strip())
        except Exception:  # noqa: BLE001 - a blank/in-progress entry falls back to the last good value
            return default

    def _model_load_entries(self) -> list[str]:
        """The current models-to-load entries from the live editor, falling back to the loaded config."""
        try:
            return [str(item) for item in self.query_one(f"#mle-root-{MODELS_TO_LOAD_KEY}", ModelListEditor).values()]
        except Exception:  # noqa: BLE001 - the model panel may not be mounted yet
            return [str(item) for item in current_value(_FIELD_BY_KEY[MODELS_TO_LOAD_KEY], self._data)]

    def is_dirty(self) -> bool:
        """Whether any field differs from the last loaded/saved state (best-effort; never raises).

        Compares raw widget values against the baseline captured on mount, save, or reload, so it does
        not run validation and a malformed in-progress entry cannot make this throw. A failure to read
        the widgets is reported as clean, so a glitch here can never trap the user on the Config tab.
        """
        gpu_editor = self._gpu_editor()
        gpu_dirty = gpu_editor.is_dirty() if gpu_editor is not None else False
        try:
            current = self._widget_state()
        except Exception:  # noqa: BLE001 - dirty detection must never block navigation
            return gpu_dirty
        if self._clean_state is None:
            self._clean_state = current
            return gpu_dirty
        return gpu_dirty or current != self._clean_state

    def _gpu_editor(self) -> GpuOverridesEditor | None:
        """The per-card override editor, or None if it has not mounted yet."""
        try:
            return self.query_one(GpuOverridesEditor)
        except Exception:  # noqa: BLE001 - queried before mount during early dirty checks
            return None

    def _widget_state(self) -> dict[str, object]:
        """A raw, uncoerced snapshot of every editable widget keyed by field, for change detection."""
        state: dict[str, object] = {}
        for field in _RENDERED_FIELDS:
            if field.kind is FieldKind.MODEL_LIST:
                state[field.key] = tuple(self.query_one(f"#mle-root-{field.key}", ModelListEditor).values())
            elif field.kind is FieldKind.SELECT_MULTI:
                state[field.key] = tuple(
                    self.query_one(f"#cfg-{field.key}-{choice}", Checkbox).value for choice in field.choices
                )
            else:
                widget = self.query_one(f"#cfg-{field.key}")
                if isinstance(widget, Switch | Select):
                    state[field.key] = widget.value
                elif isinstance(widget, TextArea):
                    state[field.key] = widget.text
                elif isinstance(widget, Input):
                    state[field.key] = widget.value
        return state

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dispatch the action-bar buttons (model-list buttons are handled by their own editor)."""
        if event.button.id == "config-reload":
            self._reload_from_disk()
        elif event.button.id == "config-preset":
            self._open_preset_modal()
        elif event.button.id == "cfg-custom_models-add":
            self._open_custom_model_builder()
        elif event.button.id == "config-save":
            self._save()
        elif event.button.id == "config-restart" and self._save():
            self.post_message(self.ApplyRequested(restart=True))

    def set_view_mode(self, mode: OverviewViewMode) -> None:
        """Apply the shared F6 density contract to the config sub-tabs.

        Thin narrows the editor to Essentials (the four fields a worker needs to register), so a quick
        glance is not buried under every advanced sub-tab. Normal and detailed expose all sub-tabs
        (detailed never hides anything normal shows); the action bar and restart markers stay pinned.
        """
        if mode is self._mode:
            return
        self._mode = mode
        thin = mode is OverviewViewMode.THIN
        essentials_id = _subtab_id(CONFIG_SUBTABS[0][0])
        with contextlib.suppress(Exception):
            tabs = self.query_one("#config-subtabs", TabbedContent)
            for label, _sections in CONFIG_SUBTABS:
                tab_id = _subtab_id(label)
                if thin and tab_id != essentials_id:
                    tabs.hide_tab(tab_id)
                else:
                    tabs.show_tab(tab_id)
            if thin:
                tabs.hide_tab(_GPU_SUBTAB_ID)
                tabs.active = essentials_id
            else:
                tabs.show_tab(_GPU_SUBTAB_ID)

    def update_worker_models(self, active_models: list[str]) -> None:
        """Forward the running worker's currently-loaded model list to the model panel (best-effort)."""
        with contextlib.suppress(Exception):
            self.query_one(ModelManagerView).update_worker_models(active_models)

    def update_cards(self, per_card: list[CardSnapshot]) -> None:
        """Forward the live per-card snapshot to the multi-GPU editor (best-effort)."""
        gpu_editor = self._gpu_editor()
        if gpu_editor is not None:
            with contextlib.suppress(Exception):
                gpu_editor.update_cards(per_card)

    def reload_from_disk(self) -> None:
        """Re-read the config file and refresh every widget (e.g. after the setup wizard writes it)."""
        self._reload_from_disk()

    def _reload_from_disk(self) -> None:
        """Re-read the file and push values back into the widgets."""
        self._data = load_config(self._config_path)
        for field in _RENDERED_FIELDS:
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
                elif isinstance(widget, Select):
                    selected = str(value)
                    widget.value = selected if selected in field.choices else field.choices[0]
                elif isinstance(widget, TextArea):
                    widget.text = (
                        format_yaml_value(value)
                        if field.kind is FieldKind.YAML
                        else "\n".join(str(item) for item in value)
                    )
                elif isinstance(widget, Input):
                    widget.value = "" if value is None else str(value)
        gpu_editor = self._gpu_editor()
        if gpu_editor is not None:
            gpu_editor.reload(self._data)
        with contextlib.suppress(Exception):
            self._clean_state = self._widget_state()
        self._update_process_preview()
        self._refresh_action_variants()
        self._set_status("Reloaded from disk.", "green")

    def _save(self) -> bool:
        """Validate and write only the fields the operator changed. Returns False on a validation error.

        Saving only changed fields is a deliberate anti-frustration choice: a value that is already
        out of bounds on disk (or that an older config simply omits) cannot block an unrelated edit,
        a no-op Save leaves the file and its mtime untouched so the running worker is not needlessly
        hot-reloaded, and any validation error can only ever concern a field the operator just touched.
        All errors are surfaced together and the editor jumps to the first one, so nothing is hidden on
        another sub-tab.
        """
        try:
            current = self._widget_state()
        except Exception:  # noqa: BLE001 - a DOM read glitch must not strand the operator on an unsaveable form
            current = None
        baseline = self._clean_state
        if current is None or baseline is None:
            # No reliable baseline to diff against: coerce every field rather than risk dropping edits.
            changed_keys = {field.key for field in _RENDERED_FIELDS}
        else:
            changed_keys = {key for key, value in current.items() if baseline.get(key) != value}

        errors: list[tuple[ConfigField, str]] = []
        coerced: list[tuple[ConfigField, object]] = []
        for field in _RENDERED_FIELDS:
            if field.key not in changed_keys:
                continue
            try:
                coerced.append((field, self._coerce_field(field)))
            except ValueError as error:
                errors.append((field, str(error)))

        # The worker identity names are validated unconditionally (not only when edited): a blank,
        # still-default, or colliding name is exactly the config that aborts the worker at startup, so
        # the save is blocked until it is fixed even if the operator was editing something else.
        identity_errors = self._identity_name_errors()

        gpu_editor = self._gpu_editor()
        gpu_dirty = gpu_editor.is_dirty() if gpu_editor is not None else False
        # The per-card editor validates and writes its own nested block; an error here aborts the save
        # alongside any flat-field error so the operator sees both at once.
        gpu_errors = gpu_editor.apply_to(self._data) if gpu_editor is not None else []
        validation_issues = validate_config_interlocks(self._merged_config_state(coerced))
        interlock_errors = [
            (_FIELD_BY_KEY.get(issue.field_key, _FIELD_BY_KEY["dreamer"]), issue.message)
            for issue in validation_issues
            if issue.severity is ConfigValidationSeverity.ERROR
        ]
        validation_warnings = [
            issue.message for issue in validation_issues if issue.severity is ConfigValidationSeverity.WARNING
        ]

        if errors or identity_errors or gpu_errors or interlock_errors:
            self._report_save_errors(
                errors + identity_errors + gpu_errors + interlock_errors,
                gpu_error_field_ids={id(field) for field, _message in gpu_errors},
            )
            return False
        if not coerced and not gpu_dirty:
            if validation_warnings:
                self._set_status(f"No changes to save. Warnings: {'; '.join(validation_warnings)}", "yellow")
            else:
                self._set_status("No changes to save.", "green")
            return True

        for field, value in coerced:
            if self._key_present(field.key) or value != field.default():
                self._data[field.key] = value

        try:
            save_config(self._data, self._config_path)
        except OSError as error:
            self._set_status(f"Failed to write {self._config_path}: {error}", "red")
            return False
        with contextlib.suppress(Exception):
            self._clean_state = self._widget_state()
        if gpu_editor is not None:
            gpu_editor.mark_saved()
        self._refresh_action_variants()
        if validation_warnings:
            self._set_status(
                f"Saved {self._config_path} with warnings: {'; '.join(validation_warnings)}",
                "yellow",
            )
        else:
            self._set_status(
                f"Saved {self._config_path}. A running worker reloads it automatically; restart only for ⟳ fields.",
                "green",
            )
        return True

    def _coerce_field(self, field: ConfigField) -> object:
        """Read one field's widget value and coerce it to its typed YAML value (raises ValueError)."""
        if field.kind is FieldKind.MODEL_LIST:
            return coerce_value(field, self.query_one(f"#mle-root-{field.key}", ModelListEditor).values())
        if field.kind is FieldKind.SELECT_MULTI:
            chosen = [
                choice for choice in field.choices if self.query_one(f"#cfg-{field.key}-{choice}", Checkbox).value
            ]
            return coerce_value(field, chosen)

        widget = self.query_one(f"#cfg-{field.key}")
        if isinstance(widget, Switch):
            raw: object = widget.value
        elif isinstance(widget, Select):
            raw = widget.value
        elif isinstance(widget, TextArea):
            raw = widget.text
        elif isinstance(widget, Input):
            raw = widget.value
        else:
            raw = ""
        return coerce_value(field, raw)

    def _merged_config_state(self, coerced_changes: list[tuple[ConfigField, object]]) -> dict[str, object]:
        """Return current config state plus the already-coerced pending flat-field changes."""
        state: dict[str, object] = {}
        for field in _RENDERED_FIELDS:
            try:
                state[field.key] = current_value(field, self._data)
            except Exception:  # noqa: BLE001 - validation should be best-effort for untouched values
                state[field.key] = field.default()
        for field, value in coerced_changes:
            state[field.key] = value
        return state

    def _current_config_state_for_preset(self) -> dict[str, object]:
        """Best-effort live config state for preset previews."""
        state = self._merged_config_state([])
        for field in _RENDERED_FIELDS:
            with contextlib.suppress(Exception):
                state[field.key] = self._coerce_field(field)
        return state

    def _open_preset_modal(self) -> None:
        """Open the built-in preset preview and apply selected changes to live widgets."""
        self.app.push_screen(ConfigPresetModal(self._current_config_state_for_preset()), self._apply_preset_changes)

    def _open_custom_model_builder(self) -> None:
        """Open the custom-model builder and append its result to the YAML field."""
        self.app.push_screen(CustomModelBuilderModal(), self._apply_custom_model_builder_result)

    def _apply_custom_model_builder_result(self, result: CustomModelBuilderResult | None) -> None:
        """Append a custom model record and optionally add it to the offer list."""
        if result is None:
            return
        field = _FIELD_BY_KEY["custom_models"]
        try:
            current = coerce_value(field, self.query_one("#cfg-custom_models", TextArea).text)
        except ValueError as error:
            self._set_status(f"Fix Custom models YAML before adding another entry: {error}", "red")
            return
        entries = list(current) if isinstance(current, list) else []
        entries.append(result.record)
        self.query_one("#cfg-custom_models", TextArea).text = format_yaml_value(entries)
        if result.add_to_models_to_load:
            with contextlib.suppress(Exception):
                self.query_one(f"#mle-root-{MODELS_TO_LOAD_KEY}", ModelListEditor).add_entry(result.record["name"])
        self._refresh_action_variants()
        self._set_status(
            f"Added custom model {result.record['name']!r} to the form; save + restart to use it.",
            "yellow",
        )

    def _apply_preset_changes(self, changes: dict[str, object] | None) -> None:
        """Apply selected preset values to widgets without saving to disk."""
        if not changes:
            return
        for key, value in changes.items():
            self._set_widget_value(key, value)
        self._update_process_preview()
        self._refresh_action_variants()
        self._set_status(
            f"Applied preset changes to the form ({len(changes)} setting(s)); save to write them.", "yellow"
        )

    def _set_widget_value(self, key: str, value: object) -> None:
        """Set one config widget from a preset value."""
        field = _FIELD_BY_KEY.get(key)
        if field is None:
            return
        if field.kind is FieldKind.MODEL_LIST:
            items = value if isinstance(value, list) else []
            with contextlib.suppress(Exception):
                self.query_one(f"#mle-root-{key}", ModelListEditor).set_values([str(item) for item in items])
            return
        if field.kind is FieldKind.SELECT_MULTI:
            selected = {str(item) for item in value} if isinstance(value, list) else {str(value)}
            for choice in field.choices:
                with contextlib.suppress(Exception):
                    self.query_one(f"#cfg-{field.key}-{choice}", Checkbox).value = choice in selected
            return
        widget = self.query_one(f"#cfg-{key}")
        if isinstance(widget, Switch):
            widget.value = bool(value)
        elif isinstance(widget, Select):
            widget.value = str(value)
        elif isinstance(widget, TextArea):
            if field.kind is FieldKind.YAML:
                widget.text = format_yaml_value(value)
            else:
                items = value if isinstance(value, list) else []
                widget.text = "\n".join(str(item) for item in items)
        elif isinstance(widget, Input):
            widget.value = "" if value is None else str(value)

    def _identity_name_errors(self) -> list[tuple[ConfigField, str]]:
        """Validate the worker identity names from the live widgets, mapped onto their ConfigFields.

        Reads the dreamer name, the alchemist toggle, and the alchemist name straight from their
        widgets so an unsaved edit is judged, falling back to the on-disk value if a widget cannot be
        read. Delegates the actual rules to ``validate_identity_names`` and keys each error to its
        ConfigField so ``_report_save_errors`` can jump to the right sub-tab.
        """
        dreamer = self._string_widget_value("dreamer_name")
        alchemist_enabled = self._bool_widget_value("alchemist")
        alchemist = self._string_widget_value("alchemist_name")
        return [
            (_FIELD_BY_KEY[key], message)
            for key, message in validate_identity_names(
                dreamer,
                alchemist_enabled=alchemist_enabled,
                alchemist_name=alchemist,
            )
        ]

    def _string_widget_value(self, key: str) -> str:
        """The current text of a string field's input, falling back to the loaded config on a read error."""
        try:
            widget = self.query_one(f"#cfg-{key}", Input)
            return widget.value
        except Exception:  # noqa: BLE001 - a DOM read glitch must fall back, not crash the save
            return str(current_value(_FIELD_BY_KEY[key], self._data))

    def _bool_widget_value(self, key: str) -> bool:
        """The current state of a boolean field's switch, falling back to the loaded config on a read error."""
        try:
            widget = self.query_one(f"#cfg-{key}", Switch)
            return widget.value
        except Exception:  # noqa: BLE001 - a DOM read glitch must fall back, not crash the save
            return bool(current_value(_FIELD_BY_KEY[key], self._data))

    def _report_save_errors(
        self,
        errors: list[tuple[ConfigField, str]],
        *,
        gpu_error_field_ids: set[int] | None = None,
    ) -> None:
        """Surface every validation error at once and jump to the first offending field.

        The operator may have edited fields across several sub-tabs, so a terse message that silently
        referred to a field on a hidden page is exactly the trap this avoids: switch to the first
        offending field's sub-tab, focus it, and list every problem (each message already names its
        field).
        """
        first_field = errors[0][0]
        if gpu_error_field_ids is not None and id(first_field) in gpu_error_field_ids:
            tab_label: str | None = _GPU_SUBTAB_LABEL
            with contextlib.suppress(Exception):
                self.query_one("#config-subtabs", TabbedContent).active = _GPU_SUBTAB_ID
        else:
            tab_label = _SECTION_TO_SUBTAB.get(first_field.section)
            with contextlib.suppress(Exception):
                if tab_label is not None:
                    self.query_one("#config-subtabs", TabbedContent).active = _subtab_id(tab_label)
                self.query_one(f"#cfg-{first_field.key}").focus()
        details = "; ".join(message for _field, message in errors)
        location = f"  See the {tab_label} tab." if tab_label else ""
        self._set_status(f"Couldn't save: {details}.{location}", "red")

    def _key_present(self, key: str) -> bool:
        """Whether a key already exists in the loaded YAML mapping."""
        try:
            return key in self._data
        except TypeError:
            return False

    def _set_status(self, message: str, colour: str) -> None:
        """Update the status line with a coloured message."""
        self.query_one("#config-status", Static).update(f"[{colour}]{message}[/]")

    def _restart_locked_dirty(self) -> bool:
        """Whether unsaved edits include a restart-only field or per-GPU structure."""
        gpu_editor = self._gpu_editor()
        if gpu_editor is not None and gpu_editor.is_dirty():
            return True
        try:
            current = self._widget_state()
        except Exception:  # noqa: BLE001 - button colouring is advisory only
            return False
        baseline = self._clean_state
        if baseline is None:
            return False
        restart_keys = {field.key for field in _RENDERED_FIELDS if field.requires_restart}
        return any(key in restart_keys and baseline.get(key) != value for key, value in current.items())

    def _refresh_action_variants(self) -> None:
        """Keep save/restart button emphasis aligned with the current unsaved edit set."""
        with contextlib.suppress(Exception):
            self.query_one("#config-restart", Button).variant = (
                "warning" if self._restart_locked_dirty() else "default"
            )
        self._refresh_config_summaries()

    def _refresh_config_summaries(self) -> None:
        """Refresh persistent operator guidance above the config tabs."""
        with contextlib.suppress(Exception):
            state = self._current_config_state_for_preset()
            self.query_one("#config-effective-summary", Static).update(self._effective_summary(state))
            self.query_one("#config-change-summary", Static).update(self._change_summary())
            self._refresh_live_warnings(state)

    def _effective_summary(self, state: dict[str, object]) -> Text:
        """A compact, plain-language summary of what this config currently serves."""
        dreamer = bool(state.get("dreamer"))
        alchemist = bool(state.get("alchemist"))
        if dreamer and alchemist:
            role = "image + alchemy"
        elif dreamer:
            role = "image generation"
        elif alchemist:
            role = "alchemy only"
        else:
            role = "nothing enabled"

        raw_load = state.get(MODELS_TO_LOAD_KEY)
        load_rules = [str(item) for item in raw_load] if isinstance(raw_load, list) else []
        model_text = "default TOP 2" if not load_rules else ", ".join(load_rules[:2])
        if len(load_rules) > 2:
            model_text += f" +{len(load_rules) - 2}"

        threads = self._state_int(state, "max_threads", 1)
        queue = self._state_int(state, "queue_size", 0)
        process_text = (
            "extra-slow: 1 inference context"
            if bool(state.get("extra_slow_worker"))
            else (f"up to {threads + queue} inference context(s)/GPU")
        )
        feature_bits = [
            f"LoRA {'on' if bool(state.get('allow_lora')) else 'off'}",
            f"ControlNet {'on' if bool(state.get('allow_controlnet')) else 'off'}",
            f"post-processing {'on' if bool(state.get('allow_post_processing')) else 'off'}",
        ]
        if bool(state.get("allow_sdxl_controlnet")):
            feature_bits.append("SDXL add-ons on")
        if bool(state.get("load_large_models")):
            feature_bits.append("large models allowed")

        text = Text("Current: ", style="bold")
        text.append(role, style="cyan" if dreamer or alchemist else "red")
        text.append(f"  |  {process_text}  |  models: {model_text}  |  ")
        text.append(" / ".join(feature_bits))
        return text

    def _change_summary(self) -> Text:
        """Show dirty-field names, and separately call out restart-required edits."""
        dirty_labels, restart_labels = self._dirty_field_labels()
        if not dirty_labels:
            return Text("No unsaved changes.", style="grey62")
        text = Text("Unsaved: ", style="bold yellow")
        text.append(self._limited_labels(dirty_labels), style="yellow")
        if restart_labels:
            text.append("  |  restart required for: ", style="bold dark_orange")
            text.append(self._limited_labels(restart_labels), style="dark_orange")
        else:
            text.append("  |  no restart-only fields changed", style="grey62")
        return text

    def _dirty_field_labels(self) -> tuple[list[str], list[str]]:
        """Return all dirty field labels and the subset that require restart."""
        try:
            current = self._widget_state()
        except Exception:  # noqa: BLE001 - summary is advisory only
            return [], []
        baseline = self._clean_state
        if baseline is None:
            return [], []
        by_key = {field.key: field for field in _RENDERED_FIELDS}
        dirty: list[str] = []
        restart: list[str] = []
        for key, value in current.items():
            if baseline.get(key) == value:
                continue
            field = by_key.get(key)
            label = field.label if field is not None else key
            dirty.append(label)
            if field is not None and field.requires_restart:
                restart.append(label)
        gpu_editor = self._gpu_editor()
        if gpu_editor is not None and gpu_editor.is_dirty():
            dirty.append("Per-GPU settings")
            restart.append("Per-GPU settings")
        return dirty, restart

    @staticmethod
    def _limited_labels(labels: list[str], *, limit: int = 4) -> str:
        """Compact a label list for single-line summaries."""
        if len(labels) <= limit:
            return ", ".join(labels)
        return f"{', '.join(labels[:limit])} +{len(labels) - limit}"

    @staticmethod
    def _state_int(state: dict[str, object], key: str, default: int) -> int:
        """Read an int-ish value from live state."""
        raw = state.get(key, default) or default
        if not isinstance(raw, int | float | str):
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    def _refresh_live_warnings(self, state: dict[str, object]) -> None:
        """Render persistent validation warnings/errors without blocking in-progress edits."""
        issues = validate_config_interlocks(state)
        target = self.query_one("#config-live-warnings", Static)
        if not issues:
            target.display = False
            target.update("")
            return
        target.display = True
        errors = [issue.message for issue in issues if issue.severity is ConfigValidationSeverity.ERROR]
        warnings = [issue.message for issue in issues if issue.severity is ConfigValidationSeverity.WARNING]
        shown = errors[:3] if errors else warnings[:3]
        prefix = "Blocking config issue" if len(shown) == 1 else "Blocking config issues"
        style = "bold red"
        if not errors:
            prefix = "Config warning" if len(shown) == 1 else "Config warnings"
            style = "yellow"
        extra = len(errors or warnings) - len(shown)
        text = Text(f"{prefix}: ", style=style)
        text.append("; ".join(shown), style=style)
        if extra > 0:
            text.append(f" (+{extra} more)", style=style)
        target.update(text)
