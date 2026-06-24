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
from textual.widgets import Button, Checkbox, Input, Label, Rule, Static, Switch, TabbedContent, TabPane, TextArea

from horde_worker_regen.app_state import OverviewViewMode
from horde_worker_regen.tui.config_form import (
    CONFIG_FIELDS,
    CONFIG_SUBTABS,
    DEFAULT_CONFIG_PATH,
    GPU_GLOBAL_FIELDS,
    GPU_OVERRIDE_FIELDS,
    MODELS_TO_LOAD_KEY,
    MODELS_TO_SKIP_KEY,
    SECTION_GUIDANCE,
    ConfigField,
    FieldKind,
    coerce_value,
    current_value,
    format_number,
    load_config,
    save_config,
)
from horde_worker_regen.tui.widgets.gpu_overrides_editor import GpuOverridesEditor
from horde_worker_regen.tui.widgets.model_list_editor import ModelListEditor
from horde_worker_regen.tui.widgets.model_manager import ModelManagerView

if TYPE_CHECKING:
    from horde_worker_regen.process_management.supervisor_channel import CardSnapshot

_FIELD_BY_KEY = {field.key: field for field in CONFIG_FIELDS}
_MODELS_SECTION = "Models"

# The per-card multi-GPU editor is its own sub-tab, mounted after the catalog-driven tabs rather than
# composed from flat ConfigFields (it edits the nested gpu_overrides block, not top-level keys).
_GPU_SUBTAB_LABEL = "Per-GPU"
_GPU_SUBTAB_ID = "cfgtab-per-gpu"
# Field keys owned by the Per-GPU sub-tab, so a validation error routes there rather than to a flat tab
# that happens to share a section name (e.g. "Features").
_GPU_FIELD_KEYS = {field.key for field in (*GPU_GLOBAL_FIELDS, *GPU_OVERRIDE_FIELDS)}

# Which sub-tab a section lives on, so a validation error can name (and jump to) the right page.
_SECTION_TO_SUBTAB: dict[str, str] = {section: label for label, sections in CONFIG_SUBTABS for section in sections}

# Only fields whose section is bundled into a sub-tab get widgets in ``compose``. A section that is
# absent from ``CONFIG_SUBTABS`` (e.g. the developer-only "Dry-run" flags, which stay editable via YAML
# but are deliberately kept out of the operator UI) is never laid out, so the loops that walk every
# field must iterate this filtered view; querying an uncomposed field raises ``NoMatches``.
_RENDERED_FIELDS: tuple[ConfigField, ...] = tuple(
    field for field in CONFIG_FIELDS if field.section in _SECTION_TO_SUBTAB
)


def _subtab_id(label: str) -> str:
    """A DOM-safe TabPane id derived from a sub-tab label (e.g. ``Alchemy`` -> ``cfgtab-alchemy``)."""
    slug = "".join(char if char.isalnum() else "-" for char in label.lower())
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
            yield Button("Save", id="config-save", variant="primary")
            yield Button("Save + restart worker", id="config-restart", variant="warning")
        yield Static(
            f"Editing {self._config_path}  ·  Save applies automatically (the worker watches the file)  ·  "
            "⟳ marks fields that only take effect on restart",
            id="config-status",
        )

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
        elif field.kind is FieldKind.STR_LIST:
            text = "\n".join(str(item) for item in value)
            control = Vertical(
                Label(label_text),
                Static(field.help, classes="config-help"),
                TextArea(text=text, id=widget_id),
                classes="config-field",
            )
        else:
            input_type = {FieldKind.INT: "integer", FieldKind.FLOAT: "number"}.get(field.kind, "text")
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

    def on_mount(self) -> None:
        """Capture the loaded values as the clean baseline for unsaved-change detection."""
        with contextlib.suppress(Exception):
            self._clean_state = self._widget_state()

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
                if isinstance(widget, Switch):
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
                elif isinstance(widget, TextArea):
                    widget.text = "\n".join(str(item) for item in value)
                elif isinstance(widget, Input):
                    widget.value = str(value)
        gpu_editor = self._gpu_editor()
        if gpu_editor is not None:
            gpu_editor.reload(self._data)
        with contextlib.suppress(Exception):
            self._clean_state = self._widget_state()
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

        gpu_editor = self._gpu_editor()
        gpu_dirty = gpu_editor.is_dirty() if gpu_editor is not None else False
        # The per-card editor validates and writes its own nested block; an error here aborts the save
        # alongside any flat-field error so the operator sees both at once.
        gpu_errors = gpu_editor.apply_to(self._data) if gpu_editor is not None else []
        if errors or gpu_errors:
            self._report_save_errors(errors + gpu_errors)
            return False
        if not coerced and not gpu_dirty:
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
        elif isinstance(widget, TextArea):
            raw = widget.text
        elif isinstance(widget, Input):
            raw = widget.value
        else:
            raw = ""
        return coerce_value(field, raw)

    def _report_save_errors(self, errors: list[tuple[ConfigField, str]]) -> None:
        """Surface every validation error at once and jump to the first offending field.

        The operator may have edited fields across several sub-tabs, so a terse message that silently
        referred to a field on a hidden page is exactly the trap this avoids: switch to the first
        offending field's sub-tab, focus it, and list every problem (each message already names its
        field).
        """
        first_field = errors[0][0]
        if first_field.key in _GPU_FIELD_KEYS:
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
