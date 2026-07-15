"""The per-card (multi-GPU) override editor: a card-chip strip plus an inherit-or-override form per card.

One worker can drive every GPU on the machine, with each card taking a delta over the global config
(``gpu_overrides`` in bridgeData.yaml). This widget edits that nested block.

The top of the tab is a single chip strip standing in for both index pickers the operator used to face:
``[All GPUs (auto)]`` is the empty-``gpu_device_indices`` "drive everything" state made visible, and the
numbered chips (pre-populated 0-3, ``+`` to go further) are the explicit drive set. A chip is green when
the running worker actually detected that card and blue when it is explicitly driven. Picking specific
chips writes ``gpu_device_indices``; leaving Auto selected omits it. Either way, every detected,
configured, or selected card gets a collapsible section below where each field is OFF (inherits the global
value, shown in the disabled control) until its Override toggle is flipped, so a single-GPU or homogeneous
box never grows a ``gpu_overrides`` block. Two cards lay out side by side when the terminal is wide enough.

The widget operates on the same ruamel ``CommentedMap`` the parent :class:`ConfigEditorView` holds, and
exposes ``is_dirty``/``apply_to``/``reload`` so the parent can drive save, dirty-detection, and reload
uniformly with the flat fields.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Collapsible, Input, Label, Static, Switch, TextArea

from horde_worker_regen.tui.config_form import (
    CONFIG_FIELDS,
    GPU_GLOBAL_FIELDS,
    GPU_OVERRIDE_FIELDS,
    GPU_OVERRIDE_SECTIONS,
    GPU_POP_BALANCE_THRESHOLD_DEFAULT,
    GPU_POP_BALANCE_THRESHOLD_KEY,
    ConfigField,
    FieldKind,
    apply_gpu_config,
    coerce_value,
    current_value,
    read_gpu_device_indices,
    read_gpu_overrides,
    read_gpu_pop_balance_threshold,
)
from horde_worker_regen.tui.formatters import gpu_label

if TYPE_CHECKING:
    from horde_worker_regen.process_management.ipc.supervisor_channel import CardSnapshot

_GLOBAL_FIELD_BY_KEY: dict[str, ConfigField] = {field.key: field for field in CONFIG_FIELDS}

# Chips always offered even before any card is detected, so the common 1-4 GPU box needs no typing.
_PREPOPULATED_CHIPS = (0, 1, 2, 3)

# Editor content width (cells) at or above which two cards are laid out side by side. Below it a single
# card's fields would be squeezed, so the grid drops to one column. Sized for two ~70-cell card forms.
_TWO_COLUMN_MIN_WIDTH = 140

# The machine-wide knob that stays a plain field (the device-indices field is replaced by the chip strip).
_POP_BALANCE_FIELD = next(field for field in GPU_GLOBAL_FIELDS if field.key == GPU_POP_BALANCE_THRESHOLD_KEY)


class GpuOverridesEditor(Vertical):
    """Edit gpu_device_indices (via the chip strip), gpu_pop_balance_threshold, and the gpu_overrides block."""

    DEFAULT_CSS = """
    GpuOverridesEditor {
        height: auto;
    }
    GpuOverridesEditor #gpu-banner {
        padding: 0 1;
        margin-bottom: 1;
        color: $text-muted;
    }
    GpuOverridesEditor .gpu-heading {
        color: $accent;
        text-style: bold;
        padding: 1 1 0 1;
    }
    GpuOverridesEditor #gpu-strip {
        height: auto;
        padding: 0 1;
        /* Rare 8+ GPU box: let the strip scroll sideways rather than overflow the page. */
        overflow-x: auto;
    }
    GpuOverridesEditor #gpu-strip Button {
        margin: 0 1 0 0;
        min-width: 9;
    }
    GpuOverridesEditor #gpu-drive-summary {
        color: $text-muted;
        padding: 0 1;
        margin-bottom: 1;
    }
    GpuOverridesEditor .gpu-field {
        height: auto;
        padding: 0 1;
    }
    GpuOverridesEditor .gpu-row {
        height: 3;
    }
    GpuOverridesEditor .gpu-row Label {
        height: 3;
        content-align: left middle;
    }
    GpuOverridesEditor .gpu-section {
        color: $accent;
        text-style: bold;
        padding: 1 1 0 1;
    }
    /* The card grid's column count is set from on_resize (a widget's DEFAULT_CSS is auto-scoped to its
       own subtree, so an ancestor breakpoint selector like Screen.-wide cannot reach it). */
    GpuOverridesEditor #gpu-cards {
        height: auto;
        layout: grid;
        grid-size: 1;
        grid-rows: auto;
        grid-gutter: 0 2;
    }
    GpuOverridesEditor .gpu-ovr-toggle {
        width: 8;
    }
    GpuOverridesEditor .gpu-ovr-label {
        width: 32;
        height: 3;
        content-align: left middle;
    }
    GpuOverridesEditor .gpu-ovr-input {
        width: 12;
    }
    GpuOverridesEditor .gpu-state-tag {
        width: 10;
        height: 3;
        content-align: left middle;
        color: $text-disabled;
    }
    GpuOverridesEditor .gpu-state-tag.-custom {
        color: $accent;
        text-style: bold;
    }
    GpuOverridesEditor #gpu-pop-field Input {
        width: 18;
    }
    GpuOverridesEditor .gpu-hint {
        color: $text-disabled;
        padding-left: 1;
    }
    GpuOverridesEditor TextArea {
        height: 5;
    }
    GpuOverridesEditor Collapsible {
        margin: 0 0 1 0;
    }
    """

    def __init__(self, data: Any) -> None:  # noqa: ANN401 - ruamel CommentedMap shared with the parent
        """Bind the shared YAML mapping and seed the drive/override sets from what it already configures."""
        super().__init__()
        self._data = data
        self._card_names: dict[int, str] = {}
        self._card_kinds: dict[int, str] = {}
        self._detected_count = 0
        # The explicit drive set (gpu_device_indices). Empty means Auto: drive every detected GPU.
        self._driven: set[int] = set(read_gpu_device_indices(data))
        # Cards already configured on disk, which always get a section so an existing override is editable.
        self._configured: set[int] = set(read_gpu_overrides(data))
        # Cards the live worker actually reported, which always get a section so a present card is tweakable.
        self._detected: set[int] = set()
        # The mounted per-card sections and numbered chips, kept as live handles so a drive-set change
        # reconciles just the deltas (mount/remove the affected ones) and never tears down and rebuilds
        # another card's in-progress edits. Rebuilding via remove_children would also race the async
        # removal against the same-id remount (DuplicateIds), which incremental reconciliation avoids.
        self._card_widgets: dict[int, Collapsible] = {}
        self._chip_buttons: dict[int, Button] = {}
        self._clean_state: dict[str, object] | None = None

    # -- Card-set bookkeeping ------------------------------------------------------------------------

    def _section_indices(self) -> list[int]:
        """Every card that gets an override section: driven, configured on disk, or live-detected."""
        return sorted(self._driven | self._configured | self._detected)

    def _chip_indices(self) -> list[int]:
        """The numbered chips to show: the pre-populated 0-3 plus any card otherwise in play."""
        return sorted(set(_PREPOPULATED_CHIPS) | self._driven | self._configured | self._detected)

    # -- Composition ---------------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Lay out the banner, the card-chip strip, the machine-wide knob, and the per-card sections."""
        yield Static(self._banner_text(), id="gpu-banner")
        yield Label("Cards this worker drives", classes="gpu-heading")
        # The Auto and add chips are always present; numbered chips are inserted between them as the
        # drive/detected sets change (see _sync_strip), so the strip never needs a racy full rebuild.
        with Horizontal(id="gpu-strip"):
            yield Button("All GPUs (auto)", id="gpu-chip-auto", variant="primary")
            yield Button("+ card", id="gpu-chip-add", variant="default")
        yield Static(self._drive_summary(), id="gpu-drive-summary")
        yield Label("Multi-GPU balancing", classes="gpu-heading")
        yield self._compose_pop_balance_field()
        yield Label("Per-card settings", classes="gpu-heading")
        yield Static(
            "Each card inherits the global config. Open a card and flip a field's Override toggle to give "
            "that one card a different value; everything else keeps the global setting.",
            classes="gpu-hint",
        )
        yield Vertical(id="gpu-cards")

    def on_mount(self) -> None:
        """Build the chip strip and the per-card sections, then capture the clean baseline."""
        self._sync_strip()
        self._sync_cards()
        self._apply_card_columns()
        self._capture_clean()

    def on_resize(self) -> None:
        """Re-pick the per-card grid column count for the new width (two cards side by side if it fits)."""
        self._apply_card_columns()

    def _apply_card_columns(self) -> None:
        """Lay the cards out two-up when there is room, single-column otherwise."""
        with contextlib.suppress(Exception):
            cards = self.query_one("#gpu-cards", Vertical)
            cards.styles.grid_size_columns = 2 if self.size.width >= _TWO_COLUMN_MIN_WIDTH else 1

    def _banner_text(self) -> str:
        """A card-count-aware note on when per-card overrides actually apply."""
        if self._detected_count == 0:
            return (
                "[b]Per-card overrides[/b] apply only when this one worker drives more than one GPU. "
                "No running worker is detected yet, so cards will be enumerated on start. You can still "
                "pre-configure a card below by its stable PCI index."
            )
        if self._detected_count == 1:
            return (
                "[b]This machine has 1 GPU detected.[/b] Per-card overrides are IGNORED on a single-GPU "
                "worker: the global config is used as-is. These settings only take effect once this worker "
                "drives multiple cards. You may still pre-configure for a planned second card below."
            )
        return (
            f"[b]{self._detected_count} GPUs detected.[/b] Each section below overrides the global config for "
            "one physical card, keyed by its stable PCI index. Fields you do not toggle on inherit the global "
            "value."
        )

    def _drive_summary(self) -> str:
        """A one-line, mode-aware explainer of what the chip selection means right now."""
        if not self._driven:
            return "Auto: driving every detected GPU. Open a card below to give just that card custom settings."
        listed = ", ".join(str(index) for index in sorted(self._driven))
        return f"Driving only the selected card(s): {listed}. Select [b]All GPUs[/b] to drive every card instead."

    # -- Chip strip ----------------------------------------------------------------------------------

    def _sync_strip(self) -> None:
        """Reconcile the numbered chips with the current drive/detected sets, in place (no full rebuild).

        The Auto and add chips persist from compose; numbered chips are added/removed for only the indices
        that changed and the rest just have their selected state and detected dot refreshed, so a chip tap
        never races an async removal against a same-id remount.
        """
        try:
            strip = self.query_one("#gpu-strip", Horizontal)
            add_button = self.query_one("#gpu-chip-add", Button)
        except Exception:  # noqa: BLE001 - compose may not have run yet
            return
        with contextlib.suppress(Exception):
            self.query_one("#gpu-chip-auto", Button).variant = "primary" if not self._driven else "default"

        desired = self._chip_indices()
        for index in sorted(set(self._chip_buttons) - set(desired)):
            button = self._chip_buttons.pop(index)
            with contextlib.suppress(Exception):
                button.remove()
        for index in desired:
            existing = self._chip_buttons.get(index)
            if existing is not None:
                existing.variant = self._chip_variant(index)
                continue
            button = Button(
                f"GPU {index}",
                id=f"gpu-chip-{index}",
                variant=self._chip_variant(index),
            )
            self._chip_buttons[index] = button
            # Keep numbered chips ordered and always ahead of the trailing add button.
            later = [other for other in sorted(self._chip_buttons) if other > index and other in self._chip_buttons]
            before = self._chip_buttons[later[0]] if later and later[0] in self._chip_buttons else add_button
            with contextlib.suppress(Exception):
                strip.mount(button, before=before)
        with contextlib.suppress(Exception):
            self.query_one("#gpu-drive-summary", Static).update(self._drive_summary())

    def _chip_variant(self, index: int) -> Literal["primary", "success", "default"]:
        """A chip's colour: blue when explicitly driven, green when the worker detected the card, else grey.

        Colour (not a glyph) carries the detected state because the marker characters that read as "present"
        are East-Asian-ambiguous width and clip the button label.
        """
        if index in self._driven:
            return "primary"
        if index in self._detected:
            return "success"
        return "default"

    def _next_chip_index(self) -> int:
        """The next index the ``+`` button adds: one past the highest card already in play (min 4)."""
        return max([*_PREPOPULATED_CHIPS, *self._driven, *self._configured, *self._detected]) + 1

    # -- Machine-wide knob ---------------------------------------------------------------------------

    def _compose_pop_balance_field(self) -> Vertical:
        """The pop-balance-threshold field (the only machine-wide knob still rendered as a plain control)."""
        field = _POP_BALANCE_FIELD
        value = read_gpu_pop_balance_threshold(self._data)
        return Vertical(
            Horizontal(
                Label(self._label_text(field), classes="gpu-ovr-label"),
                Input(value=str(value), id=f"gpucfg-{field.key}", type="number", classes="gpu-ovr-input"),
                classes="gpu-row",
            ),
            Static(field.help, classes="gpu-hint"),
            classes="gpu-field",
            id="gpu-pop-field",
        )

    def _label_text(self, field: ConfigField) -> str:
        """A field label annotated with bounds/unit and the restart marker."""
        bounds = ""
        if field.minimum is not None and field.maximum is not None:
            unit = f" {field.unit}" if field.unit else ""
            bounds = f"  ({field.minimum:g}-{field.maximum:g}{unit})"
        elif field.unit:
            bounds = f"  ({field.unit})"
        marker = "  ⟳" if field.requires_restart else ""
        return f"{field.label}{bounds}{marker}"

    # -- Per-card sections ---------------------------------------------------------------------------

    def _sync_cards(self) -> None:
        """Reconcile the mounted card sections with the section set, mounting/removing only the deltas."""
        wanted = set(self._section_indices())
        for index in sorted(set(self._card_widgets) - wanted):
            self._remove_card(index)
        for index in sorted(wanted - set(self._card_widgets)):
            self._mount_card(index)

    def _capture_clean(self) -> None:
        """Capture the current widget state as the clean (unchanged) baseline for dirty-detection."""
        with contextlib.suppress(Exception):
            self._clean_state = self._widget_state()

    def _mount_card(self, index: int) -> None:
        """Mount one card section in sorted position, leaving any other card's live edits untouched."""
        if index in self._card_widgets:
            return
        try:
            container = self.query_one("#gpu-cards", Vertical)
        except Exception:  # noqa: BLE001 - not yet mounted
            return
        overrides = read_gpu_overrides(self._data)
        card = self._compose_card(index, overrides.get(index, {}))
        self._card_widgets[index] = card
        later = [other for other in sorted(self._card_widgets) if other > index]
        if later:
            container.mount(card, before=self._card_widgets[later[0]])
        else:
            container.mount(card)
        with contextlib.suppress(Exception):
            card.scroll_visible()

    def _remove_card(self, index: int) -> None:
        """Drop one card section (used when a chip the worker never detected is deselected)."""
        card = self._card_widgets.pop(index, None)
        if card is not None:
            with contextlib.suppress(Exception):
                card.remove()

    def _card_title(self, index: int) -> str:
        """A card's collapsible title: its index with the detected name, or a not-yet-detected note."""
        name = self._card_names.get(index)
        kind = self._card_kinds.get(index, "cuda")
        if index in self._detected:
            return f"GPU {gpu_label(index, name, kind)}"
        return f"GPU {index} (not detected yet)"

    def _compose_card(self, index: int, card: dict[str, Any]) -> Collapsible:
        """One card's collapsible block: a toggled override row per field, grouped by subsection."""
        rows: list[Vertical] = []
        for section in GPU_OVERRIDE_SECTIONS:
            section_fields = [field for field in GPU_OVERRIDE_FIELDS if field.section == section]
            if not section_fields:
                continue
            rows.append(Vertical(Label(section, classes="gpu-section"), classes="gpu-field"))
            for field in section_fields:
                rows.append(self._compose_override_row(index, field, card))
        # Collapse a card that has no overrides yet when several cards are listed, so the page is not a
        # wall of mostly-inherited fields; a single card (or one already overridden) opens expanded.
        collapsed = (not card) and len(self._section_indices()) > 1
        return Collapsible(*rows, title=self._card_title(index), collapsed=collapsed)

    def _compose_override_row(self, index: int, field: ConfigField, card: dict[str, Any]) -> Vertical:
        """A single inherit-or-override field: an Override toggle plus the (disabled-until-on) control.

        When the toggle is off the control still shows the inherited global value but is disabled, so the
        row reads as "inheriting X"; the right-hand tag spells that out (inherited / custom).
        """
        override_on = field.key in card
        value: Any = card[field.key] if override_on else self._inherited_value(field)
        toggle_id = f"gpuovr-{index}-{field.key}"
        control_id = f"gpuval-{index}-{field.key}"
        tag = Static(
            self._tag_text(override_on),
            id=f"gputag-{index}-{field.key}",
            classes=self._tag_classes(override_on),
        )

        if field.kind is FieldKind.STR_LIST:
            entries = value if isinstance(value, list) else [str(value)] if value else []
            control: Any = TextArea(
                text="\n".join(str(item) for item in entries),
                id=control_id,
                disabled=not override_on,
            )
            return Vertical(
                Horizontal(
                    Switch(value=override_on, id=toggle_id, classes="gpu-ovr-toggle"),
                    Label(field.label, classes="gpu-ovr-label"),
                    tag,
                    classes="gpu-row",
                ),
                control,
                classes="gpu-field",
            )

        if field.kind is FieldKind.BOOL:
            control = Switch(value=bool(value), id=control_id, disabled=not override_on)
        else:
            input_type = {FieldKind.INT: "integer", FieldKind.FLOAT: "number"}.get(field.kind, "text")
            control = Input(
                value="" if value is None else str(value),
                id=control_id,
                type=input_type,  # type: ignore[arg-type]
                disabled=not override_on,
                classes="gpu-ovr-input",
            )

        return Vertical(
            Horizontal(
                Switch(value=override_on, id=toggle_id, classes="gpu-ovr-toggle"),
                Label(self._label_text(field), classes="gpu-ovr-label"),
                control,
                tag,
                classes="gpu-row",
            ),
            classes="gpu-field",
        )

    @staticmethod
    def _tag_text(override_on: bool) -> str:
        """The state-tag text for a row (custom when overridden, inherited otherwise)."""
        return "custom" if override_on else "inherited"

    @staticmethod
    def _tag_classes(override_on: bool) -> str:
        """The state-tag classes, adding ``-custom`` so the tag picks up the accent colour when overridden."""
        return "gpu-state-tag -custom" if override_on else "gpu-state-tag"

    def _inherited_value(self, field: ConfigField) -> Any:  # noqa: ANN401 - kind-dependent
        """The global value a card inherits when the field is not overridden."""
        global_field = _GLOBAL_FIELD_BY_KEY.get(field.key)
        if global_field is not None:
            return current_value(global_field, self._data)
        try:
            return self._data.get(field.key)
        except AttributeError:
            return None

    # -- Events --------------------------------------------------------------------------------------

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Enable/disable a field's control and flip its state tag when its Override toggle flips."""
        switch_id = event.switch.id or ""
        if not switch_id.startswith("gpuovr-"):
            return
        suffix = switch_id[len("gpuovr-") :]
        with contextlib.suppress(Exception):
            self.query_one(f"#gpuval-{suffix}").disabled = not event.value
        with contextlib.suppress(Exception):
            tag = self.query_one(f"#gputag-{suffix}", Static)
            tag.update(self._tag_text(event.value))
            tag.set_class(event.value, "-custom")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the card-chip strip: Auto, a numbered card, or the add-a-card button."""
        button_id = event.button.id or ""
        if button_id == "gpu-chip-auto":
            self._driven.clear()
            self._refresh_after_chip_change()
        elif button_id == "gpu-chip-add":
            self._driven.add(self._next_chip_index())
            self._refresh_after_chip_change()
        elif button_id.startswith("gpu-chip-"):
            with contextlib.suppress(ValueError):
                index = int(button_id[len("gpu-chip-") :])
                self._toggle_driven(index)

    def _toggle_driven(self, index: int) -> None:
        """Flip whether a card is in the explicit drive set, adding/removing its section to match."""
        if index in self._driven:
            self._driven.discard(index)
        else:
            self._driven.add(index)
        self._refresh_after_chip_change()

    def _refresh_after_chip_change(self) -> None:
        """Re-stamp the chips and reconcile the mounted card sections after a drive-set change.

        Both are reconciled incrementally (only the chips/sections whose presence actually changed), so a
        chip tap never tears down and rebuilds another card's in-progress edits.
        """
        self._sync_strip()
        self._sync_cards()

    # -- Parent-driven lifecycle ---------------------------------------------------------------------

    def update_cards(self, per_card: list[CardSnapshot]) -> None:
        """Learn detected card indices/names from a live snapshot, mounting any newly-seen cards.

        Only a genuinely new index triggers a mount, and titles refresh in place, so the per-tick refresh
        never clobbers in-progress edits to existing card sections.
        """
        self._detected_count = len(per_card)
        with contextlib.suppress(Exception):
            self.query_one("#gpu-banner", Static).update(self._banner_text())
        new_index = False
        for card in per_card:
            self._card_names[card.device_index] = card.device_name or ""
            self._card_kinds[card.device_index] = card.kind
            if card.device_index not in self._detected:
                self._detected.add(card.device_index)
                new_index = True
        if new_index:
            self._sync_strip()
            self._sync_cards()
        # Refresh card titles and chip colours for cards whose detected state only just arrived.
        for index, card_widget in self._card_widgets.items():
            with contextlib.suppress(Exception):
                card_widget.title = self._card_title(index)
        for index, chip in self._chip_buttons.items():
            with contextlib.suppress(Exception):
                chip.variant = self._chip_variant(index)

    def is_dirty(self) -> bool:
        """Whether any multi-GPU widget differs from the last loaded/saved state (never raises)."""
        try:
            current = self._widget_state()
        except Exception:  # noqa: BLE001 - dirty detection must never block navigation
            return False
        if self._clean_state is None:
            self._clean_state = current
            return False
        return current != self._clean_state

    def _widget_state(self) -> dict[str, object]:
        """Snapshot config-owned GPU state, ignoring live-only cards that still inherit everything.

        The running worker can report detected cards after the Config page opens. Those cards are useful
        editing affordances, but their mere presence is live state, not a user config change. A detected
        card enters the dirty snapshot only once it is explicitly driven, already configured on disk, or
        has at least one Override toggle enabled in the current form.
        """
        state: dict[str, object] = {
            "gpu-driven": tuple(sorted(self._driven)),
            "gpucfg-gpu_pop_balance_threshold": self.query_one("#gpucfg-gpu_pop_balance_threshold", Input).value,
        }
        for index in self._section_indices():
            if index not in self._driven and index not in self._configured and not self._card_has_override(index):
                continue
            for field in GPU_OVERRIDE_FIELDS:
                with contextlib.suppress(Exception):
                    toggle = self.query_one(f"#gpuovr-{index}-{field.key}", Switch)
                    state[f"gpuovr-{index}-{field.key}"] = toggle.value
                    control = self.query_one(f"#gpuval-{index}-{field.key}")
                    state[f"gpuval-{index}-{field.key}"] = (
                        control.text if isinstance(control, TextArea) else control.value  # type: ignore[union-attr]
                    )
        return state

    def _card_has_override(self, index: int) -> bool:
        """Whether a mounted card has any enabled per-field override toggle."""
        for field in GPU_OVERRIDE_FIELDS:
            with contextlib.suppress(Exception):
                if self.query_one(f"#gpuovr-{index}-{field.key}", Switch).value:
                    return True
        return False

    def apply_to(self, data: Any) -> list[tuple[ConfigField, str]]:  # noqa: ANN401 - ruamel CommentedMap
        """Validate and write the multi-GPU block into ``data``; return per-field validation errors.

        Mirrors the flat editor: only toggled-on fields are coerced and written, bounds errors are
        collected (not raised), and nothing is written when no card is overridden. The drive set comes
        straight from the chip selection (always valid ints), so it never raises.
        """
        errors: list[tuple[ConfigField, str]] = []

        threshold = GPU_POP_BALANCE_THRESHOLD_DEFAULT
        with contextlib.suppress(Exception):
            raw_threshold = self.query_one("#gpucfg-gpu_pop_balance_threshold", Input).value
            try:
                coerced_threshold = coerce_value(_POP_BALANCE_FIELD, raw_threshold)
                if isinstance(coerced_threshold, int | float):
                    threshold = float(coerced_threshold)
            except ValueError as error:
                errors.append((_POP_BALANCE_FIELD, str(error)))

        overrides: dict[int, dict[str, Any]] = {}
        for index in self._section_indices():
            card_fields: dict[str, Any] = {}
            for field in GPU_OVERRIDE_FIELDS:
                try:
                    toggle = self.query_one(f"#gpuovr-{index}-{field.key}", Switch)
                except Exception:  # noqa: BLE001 - section may not be mounted
                    continue
                if not toggle.value:
                    continue
                try:
                    card_fields[field.key] = self._coerce_override(index, field)
                except ValueError as error:
                    errors.append((field, f"GPU {index}: {error}"))
            if card_fields:
                overrides[index] = card_fields

        if errors:
            return errors

        apply_gpu_config(data, device_indices=sorted(self._driven), pop_threshold=threshold, overrides=overrides)
        return []

    def _coerce_override(self, index: int, field: ConfigField) -> Any:  # noqa: ANN401 - kind-dependent
        """Read one card's field control and coerce it to its typed YAML value (raises ValueError)."""
        control = self.query_one(f"#gpuval-{index}-{field.key}")
        if isinstance(control, Switch):
            raw: object = control.value
        elif isinstance(control, TextArea):
            raw = control.text
        elif isinstance(control, Input):
            raw = control.value
        else:
            raw = ""
        return coerce_value(field, raw)

    def reload(self, data: Any) -> None:  # noqa: ANN401 - ruamel CommentedMap
        """Re-read the file: refresh the drive set, the machine-wide knob, and rebuild the sections.

        A reload is an explicit discard-to-disk, so every card is torn down and rebuilt from the file. The
        re-mount is deferred to after the next refresh so the async removals complete first and cannot
        collide with the same-id remounts.
        """
        self._data = data
        self._driven = set(read_gpu_device_indices(data))
        self._configured = set(read_gpu_overrides(data))
        with contextlib.suppress(Exception):
            self.query_one("#gpucfg-gpu_pop_balance_threshold", Input).value = str(
                read_gpu_pop_balance_threshold(data)
            )
        self._sync_strip()
        for index in list(self._card_widgets):
            self._remove_card(index)
        self.call_after_refresh(self._sync_cards)
        self.call_after_refresh(self._capture_clean)

    def mark_saved(self) -> None:
        """Reset the clean baseline after the parent persists the file."""
        self._capture_clean()
