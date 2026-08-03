"""Behavioural cover for the Getting started page.

These assert the properties the page rests on: setup detection, presets that price themselves honestly
against the disk (and refuse to be chosen when they will not fit), a page that renders with no model
catalog at all, and a save that writes the keys it owns without disturbing anything else in the config.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Input, Static, Switch

from horde_worker_regen.app_state import AppStateStore, ExperienceLevel, OnboardingChoice
from horde_worker_regen.model_download_plan import DownloadPlan
from horde_worker_regen.tui import horde_validation, wizard
from horde_worker_regen.tui.app import HordeWorkerTUI
from horde_worker_regen.tui.config_form import load_config, save_config
from horde_worker_regen.tui.horde_validation import AdvisoryStatus, check_worker_name_available, verify_api_key
from horde_worker_regen.tui.model_catalog import ModelInfo
from horde_worker_regen.tui.widgets.simple import SimpleHomeView
from horde_worker_regen.tui.wizard import (
    DEFAULT_API_KEY,
    DEFAULT_DREAMER_NAME,
    GettingStartedScreen,
    PresetKind,
    _top_n_for_vram,
    build_preset_plans,
    is_setup_incomplete,
    preset_entries,
    suggested_default_models,
)
from tests.tui._fake_supervisor import FakeSupervisor

pytestmark = pytest.mark.slow

_GB = 1024**3

_RECORDS: dict[str, Any] = {}
"""A stand-in for the model reference; the tests replace the pricing function that would read it."""


def _plain(static: Static) -> str:
    """Return visible Static content without Rich styling."""
    renderable = static.render()
    return renderable.plain if isinstance(renderable, Text) else str(renderable)


def _write_config(path: Path, *, api_key: str, dreamer_name: str) -> None:
    """Write a minimal bridgeData with the given identity fields using the editor's YAML path."""
    data = load_config(path)
    data["api_key"] = api_key
    data["dreamer_name"] = dreamer_name
    save_config(data, path)


def _model(name: str, baseline: str) -> ModelInfo:
    """A catalog entry with just the attributes preset resolution reads."""
    return ModelInfo(name=name, baseline=baseline, nsfw=False, inpainting=False)


_CATALOG = [
    _model("Deliberate", "stable_diffusion_1"),
    _model("ICBINP", "stable_diffusion_1"),
    _model("AlbedoBase XL", "stable_diffusion_xl"),
    _model("Juggernaut XL", "stable_diffusion_xl"),
]

_POPULARITY = {"Deliberate": 900, "ICBINP": 700, "AlbedoBase XL": 500, "Juggernaut XL": 300}


def _plan(*, to_download: int, free: int | None, unknown: list[str] | None = None) -> DownloadPlan:
    """A download plan with the aggregate figures the preset cards read."""
    fits = free is None or to_download <= free
    return DownloadPlan(
        models=[],
        present_bytes=0,
        to_download_bytes=to_download,
        total_bytes=to_download,
        free_disk_bytes=free,
        fits=fits,
        shortfall_bytes=0 if fits or free is None else to_download - free,
        unknown_size_models=unknown or [],
    )


def test_is_setup_incomplete_when_file_missing(tmp_path: Path) -> None:
    """A missing config counts as incomplete, so setup opens on a fresh install."""
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
    """A real key and a non-default name make setup complete."""
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


def test_backend_mismatch_warns_on_cpu_build_with_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    """A detected NVIDIA card paired with the CPU torch build triggers a loud warning."""
    monkeypatch.setattr(wizard, "_detect_total_vram_mb", lambda: 12_000)
    monkeypatch.setattr(wizard, "_detect_installed_torch_build", lambda: "cpu")
    warning = GettingStartedScreen()._backend_mismatch_warning()
    assert "cu128" in warning
    assert "CPU build" in warning


def test_backend_mismatch_silent_on_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """No warning when the CUDA build is installed alongside the NVIDIA card."""
    monkeypatch.setattr(wizard, "_detect_total_vram_mb", lambda: 12_000)
    monkeypatch.setattr(wizard, "_detect_installed_torch_build", lambda: "cu128")
    assert GettingStartedScreen()._backend_mismatch_warning() == ""


def test_backend_mismatch_silent_without_device_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without device telemetry there is nothing to compare, so we stay silent."""
    monkeypatch.setattr(wizard, "_detect_total_vram_mb", lambda: None)
    monkeypatch.setattr(wizard, "_detect_installed_torch_build", lambda: "cpu")
    assert GettingStartedScreen()._backend_mismatch_warning() == ""


def test_preset_entries_scale_with_the_card() -> None:
    """Essentials is always a single model; the other presets follow the card's tier."""
    assert preset_entries(PresetKind.ESSENTIALS, _CATALOG, _POPULARITY, 24_000) == ["top 1"]
    assert preset_entries(PresetKind.RECOMMENDED, _CATALOG, _POPULARITY, 12_000) == ["top 3"]
    showcase = preset_entries(PresetKind.SHOWCASE, _CATALOG, _POPULARITY, 12_000)
    assert showcase == ["top 3", "AlbedoBase XL"]
    big_card = preset_entries(PresetKind.SHOWCASE, _CATALOG, _POPULARITY, 24_000)
    assert big_card == ["top 5", "AlbedoBase XL", "Juggernaut XL"]


def test_showcase_is_undeterminable_without_usage_stats() -> None:
    """Showcase names its SDXL models literally, so it cannot be built without the stats."""
    assert preset_entries(PresetKind.SHOWCASE, _CATALOG, None, 12_000) is None
    assert preset_entries(PresetKind.RECOMMENDED, None, None, 12_000) == ["top 3"]


def test_preset_plans_price_and_gate_on_free_space(monkeypatch: pytest.MonkeyPatch) -> None:
    """A preset that will not fit is not selectable and states the shortfall; a fitting one is."""

    def _fake_plan(names: list[str], records: object) -> DownloadPlan:
        # Showcase resolves to more models than the others, which is what makes it the one that overruns.
        return _plan(to_download=len(names) * 4 * _GB, free=9 * _GB)

    monkeypatch.setattr(wizard, "_download_plan", _fake_plan)
    plans = build_preset_plans(_CATALOG, _POPULARITY, _RECORDS, total_vram_mb=12_000)

    essentials = plans[PresetKind.ESSENTIALS]
    assert essentials.model_names == ["Deliberate"]
    assert essentials.selectable is True
    assert "4.0 GB downloaded" in essentials.disk_sentence()
    assert "9.0 GB free" in essentials.disk_sentence()

    showcase = plans[PresetKind.SHOWCASE]
    assert showcase.selectable is False
    assert "more than this computer has free" in showcase.disk_sentence()


def test_preset_plan_says_at_least_when_sizes_are_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model with no declared size makes the figure a lower bound, and it says so."""
    monkeypatch.setattr(
        wizard,
        "_download_plan",
        lambda names, records: _plan(to_download=2 * _GB, free=99 * _GB, unknown=["Deliberate"]),
    )
    plans = build_preset_plans(_CATALOG, _POPULARITY, _RECORDS, total_vram_mb=12_000)
    assert "at least 2.0 GB" in plans[PresetKind.ESSENTIALS].disk_sentence()


def test_preset_plans_degrade_without_a_catalog() -> None:
    """With nothing loaded the presets still exist; only their sizes are missing."""
    plans = build_preset_plans(None, None, None, total_vram_mb=12_000)
    assert plans[PresetKind.RECOMMENDED].resolved is False
    assert "not available yet" in plans[PresetKind.RECOMMENDED].disk_sentence()
    assert plans[PresetKind.SHOWCASE].resolved is False


class _PageHarness(App[None]):
    """A minimal app that pushes the Getting started page and records its dismissal value."""

    def __init__(self, config_path: Path) -> None:
        super().__init__()
        self._config_path = config_path
        self.saved: bool | None | str = "unset"

    def compose(self) -> ComposeResult:
        yield Button("host", id="host")

    def on_mount(self) -> None:
        self.push_screen(GettingStartedScreen(config_path=self._config_path), self._record)

    def _record(self, saved: bool | None) -> None:
        self.saved = saved


def _stub_catalog(monkeypatch: pytest.MonkeyPatch, *, loaded: bool = True) -> None:
    """Point the page at an in-memory catalog and a fixed disk plan, never the network."""
    if not loaded:
        monkeypatch.setattr(wizard, "_catalog_state", lambda: (None, None))
        monkeypatch.setattr(wizard, "_catalog_records", lambda: None)
        return
    monkeypatch.setattr(wizard, "_catalog_state", lambda: (_CATALOG, _POPULARITY))
    monkeypatch.setattr(wizard, "_catalog_records", lambda: _RECORDS)
    monkeypatch.setattr(wizard, "_download_plan", lambda names, records: _plan(to_download=2 * _GB, free=99 * _GB))


async def test_page_saves_only_the_keys_it_owns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Saving writes identity, models, NSFW and the preset's features, and leaves other keys alone."""
    monkeypatch.chdir(tmp_path)
    _stub_catalog(monkeypatch)
    config_path = tmp_path / "bridgeData.yaml"
    data = load_config(config_path)
    data["max_threads"] = 3  # an unrelated setting the page must not disturb
    save_config(data, config_path)

    app = _PageHarness(config_path)
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        app.screen.query_one("#gs-name", Input).value = "MyWorker"
        app.screen.query_one("#gs-api-key", Input).value = "my-real-key"
        app.screen.query_one("#gs-nsfw", Switch).value = True
        await pilot.pause()
        await pilot.click("#gs-choose-essentials")
        await pilot.pause()
        await pilot.click("#gs-save")
        await pilot.pause()

    assert app.saved is True
    written = load_config(config_path)
    assert written["api_key"] == "my-real-key"
    assert written["dreamer_name"] == "MyWorker"
    assert written["models_to_load"] == ["top 1"]
    assert written["nsfw"] is True
    assert written["allow_post_processing"] is True
    assert written["allow_lora"] is False
    assert written["allow_controlnet"] is False
    assert written["max_threads"] == 3


async def test_page_writes_the_recommended_stance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The suggested preset is pre-chosen, offers LoRA work, and writes its own model instruction."""
    monkeypatch.chdir(tmp_path)
    _stub_catalog(monkeypatch)
    monkeypatch.setattr(wizard, "_detect_total_vram_mb", lambda: 12_000)
    config_path = tmp_path / "bridgeData.yaml"

    app = _PageHarness(config_path)
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        app.screen.query_one("#gs-name", Input).value = "MyWorker"
        app.screen.query_one("#gs-api-key", Input).value = "my-real-key"
        await pilot.pause()
        await pilot.click("#gs-save")
        await pilot.pause()

    written = load_config(config_path)
    assert written["models_to_load"] == ["top 3"]
    assert written["allow_lora"] is True
    assert written["nsfw"] is False
    assert "civitai_api_token" not in written  # left blank, so no spurious key is written


async def test_civitai_token_is_saved_with_a_lora_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A token entered against a LoRA-accepting preset is written under the key the worker reads."""
    monkeypatch.chdir(tmp_path)
    _stub_catalog(monkeypatch)
    config_path = tmp_path / "bridgeData.yaml"

    app = _PageHarness(config_path)
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        app.screen.query_one("#gs-name", Input).value = "MyWorker"
        app.screen.query_one("#gs-api-key", Input).value = "my-real-key"
        await pilot.pause()
        await pilot.click("#gs-choose-showcase")
        await pilot.pause()
        app.screen.query_one("#gs-civitai-token", Input).value = "civ-token-123"
        await pilot.pause()
        await pilot.click("#gs-save")
        await pilot.pause()

    written = load_config(config_path)
    assert written["civitai_api_token"] == "civ-token-123"
    assert written["allow_lora"] is True


async def test_civitai_token_is_hidden_without_lora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The token is only asked for by the presets whose work downloads LoRA files."""
    monkeypatch.chdir(tmp_path)
    _stub_catalog(monkeypatch)

    app = _PageHarness(tmp_path / "bridgeData.yaml")
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        page = app.screen
        assert isinstance(page, GettingStartedScreen)
        assert page.query_one("#gs-civitai", Vertical).display is True  # recommended is pre-chosen
        # Choosing hides the block, which moves everything below it; pressing the buttons through the
        # screen's own handler keeps the assertions off that moving layout.
        page.on_button_pressed(Button.Pressed(page.query_one("#gs-choose-essentials", Button)))
        await pilot.pause()
        assert page.query_one("#gs-civitai", Vertical).display is False
        page.on_button_pressed(Button.Pressed(page.query_one("#gs-choose-recommended", Button)))
        await pilot.pause()
        assert page.query_one("#gs-civitai", Vertical).display is True


async def test_blank_civitai_field_keeps_an_existing_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Leaving the field empty is not a removal: a token already in the config survives the save."""
    monkeypatch.chdir(tmp_path)
    _stub_catalog(monkeypatch)
    config_path = tmp_path / "bridgeData.yaml"
    data = load_config(config_path)
    data["civitai_api_token"] = "already-set"
    save_config(data, config_path)

    app = _PageHarness(config_path)
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        app.screen.query_one("#gs-name", Input).value = "MyWorker"
        app.screen.query_one("#gs-api-key", Input).value = "my-real-key"
        app.screen.query_one("#gs-civitai-token", Input).value = ""
        await pilot.pause()
        await pilot.click("#gs-save")
        await pilot.pause()

    assert load_config(config_path)["civitai_api_token"] == "already-set"


async def test_page_blocks_saving_without_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The placeholder key and an empty name are refused, and nothing is written."""
    monkeypatch.chdir(tmp_path)
    _stub_catalog(monkeypatch)
    config_path = tmp_path / "bridgeData.yaml"

    app = _PageHarness(config_path)
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        app.screen.query_one("#gs-api-key", Input).value = DEFAULT_API_KEY
        await pilot.pause()
        await pilot.click("#gs-save")
        await pilot.pause()
        assert "unique worker name" in _plain(app.screen.query_one("#gs-name-error", Static))
        assert "placeholder" in _plain(app.screen.query_one("#gs-api-key-error", Static))

    assert app.saved == "unset"
    assert not config_path.exists()


async def test_unfitting_preset_is_disabled_on_the_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A preset the volume cannot hold stays visible, greyed, with the shortfall stated."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wizard, "_catalog_state", lambda: (_CATALOG, _POPULARITY))
    monkeypatch.setattr(wizard, "_catalog_records", lambda: _RECORDS)
    monkeypatch.setattr(
        wizard,
        "_download_plan",
        lambda names, records: _plan(to_download=len(names) * 4 * _GB, free=9 * _GB),
    )
    monkeypatch.setattr(wizard, "_detect_total_vram_mb", lambda: 12_000)

    app = _PageHarness(tmp_path / "bridgeData.yaml")
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        assert app.screen.query_one("#gs-choose-essentials", Button).disabled is False
        assert app.screen.query_one("#gs-choose-showcase", Button).disabled is True
        body = _plain(app.screen.query_one("#gs-preset-showcase-body", Static))
        assert "more than this computer has free" in body


async def test_page_renders_without_a_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no model reference available the page still opens and says sizes are unavailable."""
    monkeypatch.chdir(tmp_path)
    _stub_catalog(monkeypatch, loaded=False)

    app = _PageHarness(tmp_path / "bridgeData.yaml")
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        body = _plain(app.screen.query_one("#gs-preset-recommended-body", Static))
        assert "not available yet" in body
        # The recommended selection is still writable, because its instruction needs no catalog.
        page = app.screen
        assert isinstance(page, GettingStartedScreen)
        assert page._model_entries() == suggested_default_models()
        # The meta-command presets stay choosable with no catalog: only the preview is lost, never the
        # choice. Showcase names literal models, so it alone is withheld until the catalog arrives.
        assert not app.screen.query_one("#gs-choose-essentials", Button).disabled
        assert not app.screen.query_one("#gs-choose-recommended", Button).disabled
        assert app.screen.query_one("#gs-choose-showcase", Button).disabled


def _make_app(tmp_path: Path, *, configured: bool) -> HordeWorkerTUI:
    """Build the dashboard over a fake worker, with bridgeData either set up or untouched."""
    config_path = tmp_path / "bridgeData.yaml"
    if configured:
        config_path.write_text("api_key: real-key\ndreamer_name: TestWorker\n", encoding="utf-8")
    store = AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")
    store.record_onboarding_choice(OnboardingChoice.DECLINED)
    store.set_auto_start_worker(True)  # otherwise the start prompt covers the home actions
    store.set_experience_level(ExperienceLevel.SIMPLE)
    return HordeWorkerTUI(FakeSupervisor(), config_path=config_path, app_state_store=store)


async def test_home_action_opens_the_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Simple home keeps a Getting started action on a configured worker, and it opens the page."""
    monkeypatch.chdir(tmp_path)
    _stub_catalog(monkeypatch)
    app = _make_app(tmp_path, configured=True)
    async with app.run_test(size=(140, 60)) as pilot:
        await pilot.pause()
        home = app.query_one(SimpleHomeView)
        action = home.query_one("#simple-setup-action", Button)
        assert action.display is True
        assert str(action.label) == "Getting started"
        # The home view repaints every frame, which moves the row under a simulated pointer; pressing
        # the real button through its own handler exercises the same routing without racing the layout.
        home.on_button_pressed(Button.Pressed(action))
        await pilot.pause()
        assert isinstance(app.screen, GettingStartedScreen)


async def test_unconfigured_worker_is_sent_to_the_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unconfigured worker reaches the same page through the app's setup route."""
    monkeypatch.chdir(tmp_path)
    _stub_catalog(monkeypatch)
    app = _make_app(tmp_path, configured=False)
    async with app.run_test(size=(140, 60)) as pilot:
        await pilot.pause()
        app._open_getting_started()
        await pilot.pause()
        assert isinstance(app.screen, GettingStartedScreen)


def test_verify_api_key_status_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """verify_api_key maps a user hit to OK, a rejection to PROBLEM, and an error to UNKNOWN."""
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
    """A missing name is OK, an existing worker is PROBLEM, and a lookup error is UNKNOWN."""
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
