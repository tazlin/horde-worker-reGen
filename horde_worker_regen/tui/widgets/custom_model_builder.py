"""A modal for building one ``custom_models`` YAML entry without hand-writing YAML."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from horde_worker_regen.tui.model_catalog import friendly_baseline

CUSTOM_MODEL_BASELINES: tuple[str, ...] = (
    "stable_diffusion_1",
    "stable_diffusion_2_512",
    "stable_diffusion_2_768",
    "stable_diffusion_xl",
    "stable_cascade",
    "flux_1",
)
"""Baselines accepted by the worker's custom-model path."""


@dataclass(frozen=True)
class CustomModelBuilderResult:
    """One custom model entry plus whether to add it to the served model list."""

    record: dict[str, str]
    add_to_models_to_load: bool


class CustomModelBuilderModal(ModalScreen[CustomModelBuilderResult | None]):
    """Build one custom model entry from labelled fields."""

    DEFAULT_CSS = """
    CustomModelBuilderModal {
        align: center middle;
    }
    CustomModelBuilderModal #custom-model-dialog {
        width: 86;
        max-width: 92%;
        height: auto;
        max-height: 90%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    CustomModelBuilderModal .dialog-title {
        text-style: bold;
    }
    CustomModelBuilderModal .builder-field {
        height: auto;
        padding-top: 1;
    }
    CustomModelBuilderModal .builder-field Label {
        height: 1;
    }
    CustomModelBuilderModal .builder-help {
        color: $text-muted;
        height: auto;
    }
    CustomModelBuilderModal #custom-model-error {
        color: $error;
        height: auto;
        padding-top: 1;
    }
    CustomModelBuilderModal .dialog-buttons {
        height: auto;
        padding-top: 1;
    }
    CustomModelBuilderModal .dialog-buttons Button {
        margin-right: 1;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        """Lay out the builder fields and action buttons."""
        baseline_options = [(friendly_baseline(baseline), baseline) for baseline in CUSTOM_MODEL_BASELINES]
        with Vertical(id="custom-model-dialog"):
            yield Label("Add a custom model", classes="dialog-title")
            yield Static(
                "Custom models require the horde customizer role. The model file must already exist on this "
                "machine and be readable by the worker.",
                classes="builder-help",
            )
            with Vertical(classes="builder-field"):
                yield Label("Model name")
                yield Input(placeholder="Name requesters will see, e.g. My Custom SDXL", id="custom-model-name")
                yield Static("Must not conflict with an existing horde model name.", classes="builder-help")
            with Vertical(classes="builder-field"):
                yield Label("Baseline")
                yield Select(
                    baseline_options, value="stable_diffusion_1", allow_blank=False, id="custom-model-baseline"
                )
                yield Static(
                    "Choose the closest architecture family; it controls scheduling and compatibility.",
                    classes="builder-help",
                )
            with Vertical(classes="builder-field"):
                yield Label("Model file path")
                yield Input(
                    placeholder="Full local path, e.g. D:\\models\\my_model.safetensors",
                    id="custom-model-filepath",
                )
                yield Static("Use the path as seen by this worker process.", classes="builder-help")
            yield Checkbox(
                "Also add this model name to the Offer list",
                value=True,
                id="custom-model-add-to-load",
            )
            yield Static("", id="custom-model-error")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Add custom model", variant="success", id="custom-model-add")
                yield Button("Cancel", id="custom-model-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Validate and dismiss with a record, or cancel."""
        if event.button.id == "custom-model-cancel":
            self.dismiss(None)
            return
        if event.button.id != "custom-model-add":
            return
        name = self.query_one("#custom-model-name", Input).value.strip()
        baseline = str(self.query_one("#custom-model-baseline", Select).value).strip()
        filepath = self.query_one("#custom-model-filepath", Input).value.strip()
        error = self.query_one("#custom-model-error", Static)
        if not name:
            error.update("Model name is required.")
            return
        if not baseline:
            error.update("Baseline is required.")
            return
        if not filepath:
            error.update("Model file path is required.")
            return
        self.dismiss(
            CustomModelBuilderResult(
                record={"name": name, "baseline": baseline, "filepath": filepath},
                add_to_models_to_load=self.query_one("#custom-model-add-to-load", Checkbox).value,
            ),
        )

    def action_cancel(self) -> None:
        """Dismiss without adding an entry."""
        self.dismiss(None)
