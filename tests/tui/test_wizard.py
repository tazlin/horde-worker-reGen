"""Tests for the guided first-run wizard: incomplete-setup detection and the stepped flow."""

from __future__ import annotations

import types
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input

from horde_worker_regen.tui import horde_validation, wizard
from horde_worker_regen.tui.config_form import load_config, save_config
from horde_worker_regen.tui.horde_validation import AdvisoryStatus, check_worker_name_available, verify_api_key
from horde_worker_regen.tui.wizard import (
    DEFAULT_API_KEY,
    DEFAULT_DREAMER_NAME,
    SetupWizardModal,
    WizardOutcome,
    _top_n_for_vram,
    is_setup_incomplete,
    suggested_default_models,
)


def _write_config(path: Path, *, api_key: str, dreamer_name: str) -> None:
    """Write a minimal bridgeData with the given identity fields using the editor's YAML path."""
    data = load_config(path)
    data["api_key"] = api_key
    data["dreamer_name"] = dreamer_name
    save_config(data, path)


def test_is_setup_incomplete_when_file_missing(tmp_path: Path) -> None:
    """A missing config counts as incomplete, so the wizard runs on a fresh install."""
    assert is_setup_incomplete(tmp_path / "absent.yaml") is True


def test_is_setup_incomplete_with_placeholder_key(tmp_path: Path) -> None:
    """The template's placeholder API key is treated as not set."""
    path = tmp_path / "bridgeData.yaml"
    _write_config(path, api_key=DEFAULT_API_KEY, dreamer_name="A Real Name")
    assert is_setup_incomplete(path) is True


def test_is_setup_incomplete_with_placeholder_name(tmp_path: Path) -> None:
    """The template's placeholder worker name is treated as not set."""
    path = tmp_path / "bridgeData.yaml"
    _write_config(path, api_key="a-real-key", dreamer_name=DEFAULT_DREAMER_NAME)
    assert is_setup_incomplete(path) is True


def test_setup_complete_when_identity_configured(tmp_path: Path) -> None:
    """A real key and a non-default name make setup complete (no wizard)."""
    path = tmp_path / "bridgeData.yaml"
    _write_config(path, api_key="a-real-key", dreamer_name="My Worker")
    assert is_setup_incomplete(path) is False


def test_suggested_default_models_is_a_top_n_meta() -> None:
    """The default selection is a non-empty popularity meta, regardless of GPU detection."""
    models = suggested_default_models()
    assert len(models) == 1
    assert models[0].lower().startswith("top ")


@pytest.mark.parametrize(
    ("total_mb", "expected_top_n"),
    [(None, 3), (6_000, 1), (8_000, 1), (12_000, 3), (16_000, 3), (24_000, 5)],
)
def test_top_n_for_vram_tiers(total_mb: int | None, expected_top_n: int) -> None:
    """VRAM maps to a sensible default tier, with a conservative middle ground when unknown."""
    assert _top_n_for_vram(total_mb) == expected_top_n


def test_vram_detection_text_reports_detected_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """When NVML reports VRAM, the wizard surfaces it and the tier it chose (P1.1)."""
    monkeypatch.setattr(wizard, "_detect_total_vram_mb", lambda: 8_192)
    modal = SetupWizardModal()
    text = modal._vram_detection_text().plain
    assert "8 GB" in text
    assert "Top 1" in text


def test_vram_detection_text_when_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no GPU telemetry the wizard says so and falls back to the conservative tier (P1.1)."""
    monkeypatch.setattr(wizard, "_detect_total_vram_mb", lambda: None)
    modal = SetupWizardModal()
    text = modal._vram_detection_text().plain
    assert "Could not detect" in text
    assert "Top 3" in text


def test_backend_mismatch_warns_on_cpu_build_with_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    """A detected NVIDIA card paired with the CPU torch build triggers a loud warning (P0.2)."""
    monkeypatch.setattr(wizard, "_detect_total_vram_mb", lambda: 12_000)
    monkeypatch.setattr(wizard, "_detect_installed_torch_build", lambda: "cpu")
    modal = SetupWizardModal()
    warning = modal._backend_mismatch_warning()
    assert "cu128" in warning
    assert "CPU build" in warning


def test_backend_mismatch_silent_on_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """No warning when the CUDA build is installed alongside the NVIDIA card (P0.2)."""
    monkeypatch.setattr(wizard, "_detect_total_vram_mb", lambda: 12_000)
    monkeypatch.setattr(wizard, "_detect_installed_torch_build", lambda: "cu128")
    assert SetupWizardModal()._backend_mismatch_warning() == ""


def test_backend_mismatch_silent_without_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without NVML telemetry there is nothing to compare, so we stay silent (P0.2)."""
    monkeypatch.setattr(wizard, "_detect_total_vram_mb", lambda: None)
    monkeypatch.setattr(wizard, "_detect_installed_torch_build", lambda: "cpu")
    assert SetupWizardModal()._backend_mismatch_warning() == ""


def test_has_meta_selection_tracks_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default top-N selection counts as a meta; a literal model list does not (P0.4)."""
    monkeypatch.setattr(wizard, "_detect_total_vram_mb", lambda: None)
    modal = SetupWizardModal()
    assert modal._has_meta_selection() is True
    modal._models = ["Deliberate"]
    assert modal._has_meta_selection() is False


def test_verify_api_key_status_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """verify_api_key maps a user hit to OK, a rejection to PROBLEM, and an error to UNKNOWN (P0.5)."""
    from horde_sdk.generic_api.apimodels import RequestErrorResponse

    monkeypatch.setattr(horde_validation, "_submit_find_user", lambda key: types.SimpleNamespace(username="alice"))
    ok = verify_api_key("good")
    assert ok.status is AdvisoryStatus.OK
    assert ok.detail == "alice"

    monkeypatch.setattr(
        horde_validation,
        "_submit_find_user",
        lambda key: RequestErrorResponse.model_construct(message="not found"),
    )
    assert verify_api_key("bad").status is AdvisoryStatus.PROBLEM

    def _boom(key: str) -> object:
        raise ConnectionError("offline")

    monkeypatch.setattr(horde_validation, "_submit_find_user", _boom)
    assert verify_api_key("anything").status is AdvisoryStatus.UNKNOWN


def test_check_worker_name_available_status_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing name is OK, an existing worker is PROBLEM, and a lookup error is UNKNOWN (P0.5)."""
    monkeypatch.setattr(horde_validation, "_fetch_worker_details", lambda name: None)
    assert check_worker_name_available("free").status is AdvisoryStatus.OK

    monkeypatch.setattr(horde_validation, "_fetch_worker_details", lambda name: types.SimpleNamespace(id_="w-1"))
    taken = check_worker_name_available("taken")
    assert taken.status is AdvisoryStatus.PROBLEM
    assert taken.detail == "w-1"

    def _boom(name: str) -> object:
        raise TimeoutError("slow")

    monkeypatch.setattr(horde_validation, "_fetch_worker_details", _boom)
    assert check_worker_name_available("x").status is AdvisoryStatus.UNKNOWN


class _WizardHarness(App[None]):
    """A minimal app that pushes the wizard and records its dismissal value."""

    def __init__(self, config_path: Path) -> None:
        super().__init__()
        self._config_path = config_path
        self.outcome: WizardOutcome | None | str = "unset"

    def compose(self) -> ComposeResult:
        yield Button("host", id="host")

    def on_mount(self) -> None:
        self.push_screen(SetupWizardModal(config_path=self._config_path), self._record)

    def _record(self, outcome: WizardOutcome | None) -> None:
        self.outcome = outcome


async def test_wizard_collects_identity_and_starts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stepping through the wizard writes bridgeData and dismisses with the chosen action."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "bridgeData.yaml"
    app = _WizardHarness(config_path)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#wizard-next")  # welcome -> api key
        await pilot.pause()
        app.screen.query_one("#wiz-api-key", Input).value = "my-real-key"
        await pilot.pause()
        await pilot.press("enter")  # submit advances the api-key step
        await pilot.pause()
        app.screen.query_one("#wiz-name", Input).value = "MyWorker"
        await pilot.pause()
        await pilot.press("enter")  # submit advances the name step
        await pilot.pause()
        await pilot.click("#wiz-models-top1")  # narrow the default selection
        await pilot.pause()
        await pilot.click("#wizard-next")  # models -> ready
        await pilot.pause()
        await pilot.click("#wiz-finish-start")
        await pilot.pause()

    assert app.outcome is WizardOutcome.START
    data = load_config(config_path)
    assert data["api_key"] == "my-real-key"
    assert data["dreamer_name"] == "MyWorker"
    assert data["models_to_load"] == ["top 1"]
    assert "civitai_api_token" not in data  # left blank, so no spurious key is written


async def test_wizard_writes_civitai_token_when_provided(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Civitai token entered on the models step is saved to bridgeData."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "bridgeData.yaml"
    app = _WizardHarness(config_path)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#wizard-next")  # welcome -> api key
        await pilot.pause()
        app.screen.query_one("#wiz-api-key", Input).value = "my-real-key"
        await pilot.pause()
        await pilot.press("enter")  # -> name
        await pilot.pause()
        app.screen.query_one("#wiz-name", Input).value = "MyWorker"
        await pilot.pause()
        await pilot.press("enter")  # -> models
        await pilot.pause()
        app.screen.query_one("#wiz-civitai-token", Input).value = "civ-token-123"
        await pilot.pause()
        await pilot.click("#wizard-next")  # models -> ready
        await pilot.pause()
        await pilot.click("#wiz-finish-stay")
        await pilot.pause()

    assert app.outcome is WizardOutcome.STAY_STOPPED
    data = load_config(config_path)
    assert data["civitai_api_token"] == "civ-token-123"


async def test_wizard_blocks_advance_on_missing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The wizard will not advance past the API-key step while it is empty/placeholder."""
    monkeypatch.chdir(tmp_path)
    app = _WizardHarness(tmp_path / "bridgeData.yaml")
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#wizard-next")  # welcome -> api key
        await pilot.pause()
        app.screen.query_one("#wiz-api-key", Input).value = DEFAULT_API_KEY  # still the placeholder
        await pilot.pause()
        await pilot.press("enter")  # submit should be blocked by validation
        await pilot.pause()
        # We never advanced: the api-key step is still showing and the name step is hidden.
        assert app.screen.query_one("#wiz-step-1").display is True
        assert app.screen.query_one("#wiz-step-2").display is False


async def test_wizard_cancel_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling the wizard leaves no config behind and returns None."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "bridgeData.yaml"
    app = _WizardHarness(config_path)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#wizard-cancel")
        await pilot.pause()

    assert app.outcome is None
    assert not config_path.exists()
