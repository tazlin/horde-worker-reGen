"""The guided first-run setup wizard.

A linear, can't-get-lost flow shown on first launch when ``bridgeData.yaml`` is unconfigured (see
[`is_setup_incomplete`][horde_worker_regen.tui.wizard.is_setup_incomplete]). It collects the only two
things a worker cannot run without (an API key and a unique worker name) plus an initial model
selection, writes them with the same light YAML path the config editor uses
([`save_config`][horde_worker_regen.tui.config_form.save_config]), and then hands off to the existing
benchmark / start flow. Every step reuses an existing control: model browsing is the same
[`ModelPickerModal`][horde_worker_regen.tui.widgets.model_picker.ModelPickerModal] the Config tab uses,
and starting the worker downloads the chosen models exactly as a normal run does.

The wizard never blocks the dashboard: cancelling leaves the worker stopped and the tabs available, so
a power user can configure by hand instead.
"""

from __future__ import annotations

import asyncio
import enum
import os
import warnings
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from horde_worker_regen.tui.config_form import DEFAULT_CONFIG_PATH, load_config, save_config
from horde_worker_regen.tui.horde_validation import AdvisoryStatus, check_worker_name_available, verify_api_key
from horde_worker_regen.tui.model_catalog import MetaKind, build_meta_instruction, is_meta_instruction
from horde_worker_regen.tui.widgets.model_picker import ModelPickerModal

DEFAULT_API_KEY = "0000000000"
"""The placeholder key shipped in bridgeData_template.yaml; treated as "not set"."""

DEFAULT_DREAMER_NAME = "An Awesome Dreamer"
"""The placeholder worker name shipped in the template; the horde rejects duplicates, so it must change."""

REGISTER_URL = "https://aihorde.net/register"


class WizardOutcome(enum.StrEnum):
    """What the user chose to do once setup was saved."""

    BENCHMARK = "benchmark"
    """Tune settings with a benchmark before serving (it starts the worker afterwards)."""
    START = "start"
    """Start the worker now; selected models download on demand."""
    STAY_STOPPED = "stay_stopped"
    """Save the configuration but leave the worker stopped (start later with F3)."""


def _config_str(data: Any, key: str) -> str:  # noqa: ANN401 - ruamel CommentedMap / dict
    """Read a string value from loaded YAML data, returning an empty string when absent."""
    try:
        value = data.get(key)
    except AttributeError:
        return ""
    return "" if value is None else str(value).strip()


def is_setup_incomplete(config_path: Path = DEFAULT_CONFIG_PATH) -> bool:
    """Return whether bridgeData lacks a real API key or still uses the default worker name.

    This is the trigger for showing the wizard. A missing or unreadable file counts as incomplete, and
    so do the template placeholders, so a freshly seeded config (the installer copies the template)
    leads straight into setup.
    """
    try:
        data = load_config(config_path)
    except Exception:  # noqa: BLE001 - an unreadable config is, for our purposes, an unconfigured one
        return True
    api_key = _config_str(data, "api_key")
    dreamer_name = _config_str(data, "dreamer_name")
    if not api_key or api_key == DEFAULT_API_KEY:
        return True
    return not dreamer_name or dreamer_name == DEFAULT_DREAMER_NAME


def _detect_total_vram_mb() -> int | None:
    """Best-effort read of the primary GPU's total VRAM in MB via NVML, or None when unavailable.

    Mirrors the graceful-degradation pattern in ``utils.gpu_monitor``: any NVML failure (no NVIDIA GPU,
    no pynvml, AMD/CPU) yields None so the caller falls back to a safe default rather than erroring.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence pynvml's deprecation FutureWarning
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            total_bytes = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - "no GPU telemetry" is expected, not a crash
        return None
    return total_bytes // (1024 * 1024)


def _top_n_for_vram(total_mb: int | None) -> int:
    """The default ``top N`` size for a card with *total_mb* of VRAM (None when unknown).

    Bigger cards can comfortably hold more resident models, so they get a larger N; without NVML we
    cannot tell, so we pick a conservative middle ground the user can change in the model step.
    """
    if total_mb is None:
        return 3
    if total_mb >= 20_000:
        return 5
    if total_mb >= 10_000:
        return 3
    return 1


def suggested_default_models() -> list[str]:
    """A VRAM-aware initial model selection, defaulting to a safe ``top N`` popularity meta."""
    return [build_meta_instruction(MetaKind.TOP_N, _top_n_for_vram(_detect_total_vram_mb()))]


def _detect_installed_torch_build() -> str | None:
    """The local build tag of the installed torch wheel (e.g. ``cu128``, ``cpu``, ``rocm6.4``).

    Read from package metadata so we never import torch itself, which is slow and would pull a CUDA
    context into the lightweight TUI process. Returns None when torch or its version is not findable.
    """
    try:
        from importlib.metadata import version

        raw = version("torch")
    except Exception:  # noqa: BLE001 - "cannot tell which build" is expected, not a crash
        return None
    _, _, local = raw.partition("+")
    return local or None


def _testing_mode() -> bool:
    """Whether we are running under the test harness, where advisory network calls are skipped."""
    return bool(os.environ.get("AI_HORDE_TESTING"))


_STEP_TITLES = (
    "Welcome",
    "Step 1 of 4  ·  API key",
    "Step 2 of 4  ·  Worker name",
    "Step 3 of 4  ·  Models",
    "Step 4 of 4  ·  Ready",
)
_FINAL_STEP = len(_STEP_TITLES) - 1


class SetupWizardModal(ModalScreen["WizardOutcome | None"]):
    """A stepped first-run wizard that writes bridgeData and returns the user's next action."""

    BINDINGS = [Binding("escape", "cancel", "Cancel setup")]

    DEFAULT_CSS = """
    SetupWizardModal {
        align: center middle;
    }
    SetupWizardModal #wizard-dialog {
        width: 72;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    SetupWizardModal #wizard-title {
        text-style: bold;
        padding-bottom: 1;
    }
    SetupWizardModal #wizard-body {
        height: auto;
        max-height: 20;
    }
    SetupWizardModal .wiz-step Input {
        margin-top: 1;
    }
    SetupWizardModal .wiz-error {
        color: $error;
        padding-top: 1;
    }
    SetupWizardModal .wiz-warn {
        color: $warning;
        padding-top: 1;
    }
    SetupWizardModal #wiz-models-summary {
        padding: 1 0;
    }
    SetupWizardModal #wiz-model-buttons Button {
        margin-right: 1;
    }
    SetupWizardModal #wiz-step-4 Button {
        width: 100%;
        margin-top: 1;
    }
    SetupWizardModal #wizard-nav {
        height: auto;
        padding-top: 1;
    }
    SetupWizardModal #wizard-nav Button {
        margin-right: 1;
    }
    SetupWizardModal #wizard-progress {
        width: 1fr;
        content-align: right middle;
        color: $text-muted;
    }
    """

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        """Store the config path and seed a VRAM-aware default model selection."""
        super().__init__()
        self._config_path = config_path
        self._step = 0
        # Detect once at construction so the UI can show what we found without re-probing the GPU.
        self._detected_vram_mb = _detect_total_vram_mb()
        self._default_top_n = _top_n_for_vram(self._detected_vram_mb)
        self._models: list[str] = [build_meta_instruction(MetaKind.TOP_N, self._default_top_n)]

    def compose(self) -> ComposeResult:
        """Lay out the title, the per-step bodies (only one shown at a time), and the nav bar."""
        with Vertical(id="wizard-dialog"):
            yield Static(id="wizard-title")
            with VerticalScroll(id="wizard-body"):
                yield Vertical(
                    Static(self._welcome_text()),
                    Static("", id="wiz-backend-warning", classes="wiz-error"),
                    id="wiz-step-0",
                    classes="wiz-step",
                )
                yield Vertical(
                    Static(
                        Text.assemble(
                            ("Paste your AI Horde API key. ", ""),
                            (f"Register free at {REGISTER_URL}.", "grey70"),
                        ),
                    ),
                    Input(placeholder="API key", password=True, id="wiz-api-key"),
                    Static("", id="wiz-api-key-error", classes="wiz-error"),
                    id="wiz-step-1",
                    classes="wiz-step",
                )
                yield Vertical(
                    Static(
                        "Choose a unique name for your worker (shown publicly on the horde). "
                        "It cannot be the default and cannot clash with an existing worker.",
                    ),
                    Input(placeholder="Worker name", id="wiz-name"),
                    Static("", id="wiz-name-error", classes="wiz-error"),
                    id="wiz-step-2",
                    classes="wiz-step",
                )
                yield Vertical(
                    Static(
                        "Which models will you serve? The default loads the most popular models for your "
                        "card; they download when the worker starts. You can change this any time on the "
                        "Config tab.",
                    ),
                    Static(id="wiz-models-detected"),
                    Static(id="wiz-models-summary"),
                    Horizontal(
                        Button("Top 1", id="wiz-models-top1"),
                        Button("Top 3", id="wiz-models-top3"),
                        Button("Top 5", id="wiz-models-top5"),
                        Button("All SD 1.5", id="wiz-models-sd15"),
                        Button("Browse all…", id="wiz-models-browse"),
                        id="wiz-model-buttons",
                    ),
                    Static(id="wiz-civitai-help"),
                    Input(placeholder="Civitai API token", password=True, id="wiz-civitai-token"),
                    Static("", id="wiz-civitai-warning", classes="wiz-warn"),
                    id="wiz-step-3",
                    classes="wiz-step",
                )
                yield Vertical(
                    Static(
                        "Setup is ready to save. A benchmark briefly tests your card and tunes settings "
                        "for you (recommended on first run); or start straight away.",
                    ),
                    Button("Run benchmark first (recommended)", id="wiz-finish-benchmark", variant="success"),
                    Button("Start worker now", id="wiz-finish-start", variant="primary"),
                    Button("Save and stay stopped", id="wiz-finish-stay", variant="default"),
                    Static("", id="wiz-finish-error", classes="wiz-error"),
                    id="wiz-step-4",
                    classes="wiz-step",
                )
            with Horizontal(id="wizard-nav"):
                yield Button("Back", id="wizard-back")
                yield Button("Next", id="wizard-next", variant="primary")
                yield Button("Cancel", id="wizard-cancel")
                yield Static(id="wizard-progress")

    def on_mount(self) -> None:
        """Render the first step and the hardware-aware hints."""
        self._update_backend_warning()
        self.query_one("#wiz-models-detected", Static).update(self._vram_detection_text())
        self._update_models_summary()
        self._show_step(0)

    def _update_backend_warning(self) -> None:
        """Show a warning on the welcome step when an NVIDIA GPU is paired with the CPU torch build."""
        self.query_one("#wiz-backend-warning", Static).update(self._backend_mismatch_warning())

    def _backend_mismatch_warning(self) -> str:
        """Text warning of an installed CPU build despite a detected NVIDIA GPU, else ``""``.

        NVML talks to the driver, not torch, so a present GPU still reports its VRAM even when the CPU
        wheel is installed; pairing that with a ``+cpu`` build is the classic "why is it so slow" trap.
        When NVML reports nothing (no NVIDIA telemetry) there is nothing to compare against.
        """
        if self._detected_vram_mb is None:
            return ""
        build = _detect_installed_torch_build()
        if build is not None and build.startswith("cpu"):
            return (
                "An NVIDIA GPU was detected, but the CPU build of PyTorch is installed, so the worker "
                "would run roughly 100x slower. Re-run the installer with HORDE_WORKER_BACKEND=cu128 "
                "to fix this."
            )
        return ""

    def _vram_detection_text(self) -> Text:
        """A grey hint stating the detected VRAM and the default model tier it implies."""
        if self._detected_vram_mb is None:
            return Text(
                "Could not detect GPU memory; defaulting to a conservative Top 3 (change it below).",
                "grey62",
            )
        gb = self._detected_vram_mb / 1024
        return Text(
            f"Detected ~{gb:.0f} GB VRAM; defaulting to Top {self._default_top_n} (change it below).",
            "grey62",
        )

    @staticmethod
    def _welcome_text() -> Text:
        """The opening explanation."""
        return Text.assemble(
            ("Let's get your worker earning kudos.\n\n", "bold"),
            (
                "This quick setup asks for your API key and a worker name, lets you pick which models to "
                "serve, and then starts the worker. It takes about a minute.",
                "grey70",
            ),
        )

    def _show_step(self, index: int) -> None:
        """Display step ``index`` only, and update the title, nav buttons, and progress text."""
        self._step = index
        for step in range(len(_STEP_TITLES)):
            self.query_one(f"#wiz-step-{step}", Vertical).display = step == index
        self.query_one("#wizard-title", Static).update(_STEP_TITLES[index])
        self.query_one("#wizard-back", Button).display = index > 0
        # The final step's own buttons commit the choice, so the generic Next is hidden there.
        self.query_one("#wizard-next", Button).display = index < _FINAL_STEP
        self.query_one("#wizard-progress", Static).update(f"{index + 1} / {len(_STEP_TITLES)}")
        if index == 1:
            self.query_one("#wiz-api-key", Input).focus()
        elif index == 2:
            self.query_one("#wiz-name", Input).focus()

    def _update_models_summary(self) -> None:
        """Reflect the current model selection in the step-3 summary line and Civitai guidance."""
        summary = ", ".join(self._models) if self._models else "(none selected)"
        self.query_one("#wiz-models-summary", Static).update(
            Text.assemble(("Selected: ", "grey62"), (summary, "bold")),
        )
        self._update_civitai_help()

    def _has_meta_selection(self) -> bool:
        """Whether the current selection contains a meta instruction (``top N`` and friends)."""
        return any(is_meta_instruction(name) for name in self._models)

    def _update_civitai_help(self) -> None:
        """Make the Civitai token read as recommended (not merely optional) for meta selections.

        ``top N`` and ``all`` style selections routinely pull popular models and LoRAs that the horde
        cannot fetch without a token, and that failure otherwise surfaces deep into the download.
        """
        help_static = self.query_one("#wiz-civitai-help", Static)
        warn_static = self.query_one("#wiz-civitai-warning", Static)
        if self._has_meta_selection():
            help_static.update(
                Text.assemble(
                    ("Civitai API token ", ""),
                    ("(recommended). ", "yellow"),
                    ("Top-N selections usually need it to download popular models and LoRAs.", "grey70"),
                ),
            )
            warn_static.update(
                ""
                if self._civitai_token()
                else "Heads up: without a Civitai token, some of these models may fail to download.",
            )
        else:
            help_static.update(
                Text.assemble(
                    ("Civitai API token ", ""),
                    ("(optional). ", "grey70"),
                    ("Some models and LoRAs need it to download.", "grey70"),
                ),
            )
            warn_static.update("")

    # region step validation / navigation

    def _api_key(self) -> str:
        """The trimmed API key currently entered."""
        return self.query_one("#wiz-api-key", Input).value.strip()

    def _worker_name(self) -> str:
        """The trimmed worker name currently entered."""
        return self.query_one("#wiz-name", Input).value.strip()

    def _civitai_token(self) -> str:
        """The trimmed Civitai API token currently entered (empty when the user left it blank)."""
        return self.query_one("#wiz-civitai-token", Input).value.strip()

    def _validate_step(self, index: int) -> bool:
        """Validate the step the user is leaving, surfacing an inline error and blocking on failure."""
        if index == 1:
            return self._validate_api_key()
        if index == 2:
            return self._validate_name()
        return True

    def _validate_api_key(self) -> bool:
        """Require a non-empty, non-placeholder API key."""
        error = self.query_one("#wiz-api-key-error", Static)
        key = self._api_key()
        if not key or key == DEFAULT_API_KEY:
            error.update("Enter your API key (the default placeholder will not work).")
            return False
        error.update("")
        return True

    def _validate_name(self) -> bool:
        """Require a non-empty worker name that is not the default placeholder."""
        error = self.query_one("#wiz-name-error", Static)
        name = self._worker_name()
        if not name or name == DEFAULT_DREAMER_NAME:
            error.update("Choose a unique worker name (not the default).")
            return False
        error.update("")
        return True

    def _advance(self) -> None:
        """Move to the next step if the current one validates, kicking off advisory horde checks."""
        leaving = self._step
        if not self._validate_step(leaving):
            return
        self._show_step(min(leaving + 1, _FINAL_STEP))
        if leaving == 1:
            self._run_advisory(self._advisory_key_check(self._api_key()), group="key-check")
        elif leaving == 2:
            self._run_advisory(self._advisory_name_check(self._worker_name()), group="name-check")

    def _retreat(self) -> None:
        """Move to the previous step."""
        self._show_step(max(self._step - 1, 0))

    def _run_advisory(self, coro: Coroutine[Any, Any, None], *, group: str) -> None:
        """Run an advisory coroutine off the UI thread, or discard it under the test harness."""
        if _testing_mode():
            coro.close()
            return
        self.run_worker(coro, group=group, exclusive=True)

    async def _advisory_key_check(self, api_key: str) -> None:
        """Validate the API key against the horde and toast a hint if it was rejected."""
        result = await asyncio.to_thread(verify_api_key, api_key)
        if result.status is AdvisoryStatus.PROBLEM:
            self.app.notify(
                f"That API key did not validate with the horde ({result.detail or 'rejected'}). "
                "Double-check it before you start the worker.",
                title="API key",
                severity="warning",
                timeout=8,
            )
        elif result.status is AdvisoryStatus.OK and result.detail:
            self.app.notify(f"API key validated for user '{result.detail}'.", title="API key", timeout=4)

    async def _advisory_name_check(self, worker_name: str) -> None:
        """Warn if the chosen worker name is already taken on the horde."""
        result = await asyncio.to_thread(check_worker_name_available, worker_name)
        if result.status is AdvisoryStatus.PROBLEM:
            self.app.notify(
                f"A worker named '{worker_name}' already exists on the horde. If it is not yours, pick "
                "a different name or you will get a 'wrong credentials' error.",
                title="Worker name",
                severity="warning",
                timeout=8,
            )

    # endregion

    # region model selection

    def _set_meta(self, kind: MetaKind, count: int = 1) -> None:
        """Replace the selection with a single meta instruction."""
        self._models = [build_meta_instruction(kind, count)]
        self._update_models_summary()

    def _browse_models(self) -> None:
        """Open the full model picker; chosen literal models replace the current selection."""
        exclude = {name for name in self._models if not is_meta_instruction(name)}
        self.app.push_screen(ModelPickerModal(exclude=exclude), self._on_models_chosen)

    def _on_models_chosen(self, chosen: list[str] | None) -> None:
        """Apply the picker result, keeping any existing meta commands alongside picked models."""
        if not chosen:
            return
        metas = [name for name in self._models if is_meta_instruction(name)]
        self._models = metas + chosen
        self._update_models_summary()

    # endregion

    def _finish(self, outcome: WizardOutcome) -> None:
        """Persist the collected settings to bridgeData and dismiss with the chosen next action."""
        error = self.query_one("#wiz-finish-error", Static)
        try:
            data = load_config(self._config_path)
            data["api_key"] = self._api_key()
            data["dreamer_name"] = self._worker_name()
            data["models_to_load"] = self._models
            civitai_token = self._civitai_token()
            if civitai_token:
                data["civitai_api_token"] = civitai_token
            save_config(data, self._config_path)
        except OSError as write_error:
            error.update(f"Could not write {self._config_path}: {write_error}")
            return
        self.dismiss(outcome)

    def action_cancel(self) -> None:
        """Abandon setup without saving; the dashboard stays usable and the worker stopped."""
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route nav, model-quick-pick, and finish buttons."""
        button_id = event.button.id or ""
        if button_id == "wizard-next":
            self._advance()
        elif button_id == "wizard-back":
            self._retreat()
        elif button_id == "wizard-cancel":
            self.action_cancel()
        elif button_id == "wiz-models-top1":
            self._set_meta(MetaKind.TOP_N, 1)
        elif button_id == "wiz-models-top3":
            self._set_meta(MetaKind.TOP_N, 3)
        elif button_id == "wiz-models-top5":
            self._set_meta(MetaKind.TOP_N, 5)
        elif button_id == "wiz-models-sd15":
            self._set_meta(MetaKind.ALL_SD15)
        elif button_id == "wiz-models-browse":
            self._browse_models()
        elif button_id == "wiz-finish-benchmark":
            self._finish(WizardOutcome.BENCHMARK)
        elif button_id == "wiz-finish-start":
            self._finish(WizardOutcome.START)
        elif button_id == "wiz-finish-stay":
            self._finish(WizardOutcome.STAY_STOPPED)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Refresh the Civitai guidance live as the token field is typed."""
        if event.input.id == "wiz-civitai-token":
            self._update_civitai_help()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Pressing Enter in a step's input advances, matching the Next button."""
        if self._step in (1, 2):
            self._advance()


__all__ = [
    "DEFAULT_API_KEY",
    "DEFAULT_DREAMER_NAME",
    "SetupWizardModal",
    "WizardOutcome",
    "is_setup_incomplete",
    "suggested_default_models",
]
