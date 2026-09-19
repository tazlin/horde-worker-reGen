"""Guards for the per-card multi-GPU override editor: catalog parity, nested YAML round-trip, banner.

The per-card catalog (``config_form.GPU_OVERRIDE_FIELDS``) is import-light and does not load the
``GpuOverride`` model at runtime, so it can silently drift from the real overridable field set. The
parity test below is the single enforcement point keeping the two in lockstep; it may import the heavy
model because tests are not the import-light TUI parent. The remaining tests cover the nested-YAML
write/read helpers and the editor's save/inherit behaviour end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Switch, TabbedContent

from horde_worker_regen.app_state import (
    AppStateStore,
    ExperienceLevel,
    KnownGpu,
    KnownGpuInventory,
    OnboardingChoice,
)
from horde_worker_regen.bridge_data.data_model import GpuOverride
from horde_worker_regen.tui.app import HordeWorkerTUI
from horde_worker_regen.tui.config_form import (
    GPU_OVERRIDE_FIELDS,
    apply_gpu_config,
    load_config,
    read_gpu_device_indices,
    read_gpu_overrides,
    read_gpu_pop_balance_threshold,
)
from horde_worker_regen.tui.widgets.config_editor import ConfigEditorView
from horde_worker_regen.tui.widgets.gpu_overrides_editor import GpuOverridesEditor
from tests.tui._fake_supervisor import FakeSupervisor

pytestmark = pytest.mark.slow


def _model_keys() -> set[str]:
    """Every GpuOverride field name plus its alias (the spellings the YAML may legitimately use)."""
    keys: set[str] = set()
    for name, field in GpuOverride.model_fields.items():
        keys.add(name)
        if field.alias:
            keys.add(field.alias)
    return keys


def test_catalog_matches_gpu_override_model() -> None:
    """The editor catalog covers exactly the GpuOverride fields, each by a real name or alias."""
    model_keys = _model_keys()
    catalog_keys = [field.key for field in GPU_OVERRIDE_FIELDS]

    assert len(catalog_keys) == len(set(catalog_keys)), "duplicate keys in GPU_OVERRIDE_FIELDS"

    unknown = [key for key in catalog_keys if key not in model_keys]
    assert not unknown, f"GPU_OVERRIDE_FIELDS keys not on GpuOverride: {unknown}"

    catalog_set = set(catalog_keys)
    missing: list[str] = []
    for name, field in GpuOverride.model_fields.items():
        accepted = {name} | ({field.alias} if field.alias else set())
        if not (catalog_set & accepted):
            missing.append(name)
    assert not missing, f"GpuOverride fields absent from GPU_OVERRIDE_FIELDS: {missing}"


def test_apply_gpu_config_writes_nested_block(tmp_path: Path) -> None:
    """A card with set fields produces a sorted, int-keyed gpu_overrides block plus the driven list."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text('api_key: "x"\n', encoding="utf-8")
    data = load_config(path)

    apply_gpu_config(
        data,
        device_indices=[0, 1],
        pop_threshold=0.25,
        overrides={1: {"max_threads": 2}, 0: {"allow_lora": True}},
    )

    assert read_gpu_device_indices(data) == [0, 1]
    assert read_gpu_pop_balance_threshold(data) == 0.25
    overrides = read_gpu_overrides(data)
    assert overrides == {0: {"allow_lora": True}, 1: {"max_threads": 2}}


def test_apply_gpu_config_omits_empty_pieces(tmp_path: Path) -> None:
    """Empty per-card dicts, an empty driven list, and a default threshold are not written."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text(
        "gpu_device_indices:\n  - 0\ngpu_pop_balance_threshold: 0.9\ngpu_overrides:\n  0:\n    max_threads: 4\n",
        encoding="utf-8",
    )
    data = load_config(path)

    # Nothing meaningful set: every multi-GPU key should be removed from the mapping.
    apply_gpu_config(data, device_indices=[], pop_threshold=0.5, overrides={0: {}})

    assert "gpu_device_indices" not in data
    assert "gpu_pop_balance_threshold" not in data
    assert "gpu_overrides" not in data


def test_banner_reflects_detected_card_count(tmp_path: Path) -> None:
    """The banner text states the single-GPU caveat, the no-worker case, and the multi-GPU case."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text('api_key: "x"\n', encoding="utf-8")
    editor = GpuOverridesEditor(load_config(path))

    editor._detected_count = 0
    assert "No running worker" in editor._banner_text()
    editor._detected_count = 1
    assert "1 GPU detected" in editor._banner_text() and "IGNORED" in editor._banner_text()
    editor._detected_count = 2
    assert "2 GPUs detected" in editor._banner_text()


@pytest.mark.e2e
async def test_toggling_an_override_writes_only_that_field(tmp_path: Path) -> None:
    """Flipping one card's Override toggle writes just that field; untoggled fields stay inherited."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text('api_key: "x"\ndreamer_name: "n"\ngpu_device_indices:\n  - 0\n  - 1\n', encoding="utf-8")

    class _Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield ConfigEditorView(config_path=path)

    app = _Harness()
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        editor.query_one("#gpuovr-0-max_threads", Switch).value = True
        editor.query_one("#gpuval-0-max_threads", Input).value = "2"
        await pilot.pause()
        assert editor._save() is True
        await pilot.pause()

    overrides = read_gpu_overrides(load_config(path))
    assert overrides == {0: {"max_threads": 2}}
    # Card 1 was listed but never overridden, so it must not appear in the written block.
    assert 1 not in overrides


@pytest.mark.e2e
async def test_clearing_a_toggle_removes_the_override(tmp_path: Path) -> None:
    """Turning an existing override off drops its key (and the whole block when nothing remains)."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text(
        'api_key: "x"\ndreamer_name: "n"\ngpu_overrides:\n  0:\n    allow_lora: true\n',
        encoding="utf-8",
    )

    class _Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield ConfigEditorView(config_path=path)

    app = _Harness()
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        assert editor.query_one("#gpuovr-0-allow_lora", Switch).value is True
        editor.query_one("#gpuovr-0-allow_lora", Switch).value = False
        await pilot.pause()
        assert editor._save() is True
        await pilot.pause()

    assert "gpu_overrides" not in load_config(path)


@pytest.mark.e2e
async def test_update_cards_mounts_a_detected_card(tmp_path: Path) -> None:
    """A newly-detected card index from the live snapshot gains an editable section."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text('api_key: "x"\ndreamer_name: "n"\n', encoding="utf-8")

    class _Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield ConfigEditorView(config_path=path)

    app = _Harness()
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        # No cards configured in the file, so no card section exists yet.
        assert not editor.query("#gpuovr-0-max_threads")
        editor.update_cards([SimpleNamespace(device_index=0, device_name="RTX 4090", kind="cuda")])
        await pilot.pause()
        assert editor.query("#gpuovr-0-max_threads")


def test_next_chip_index_walks_past_known_cards(tmp_path: Path) -> None:
    """The add-a-card button targets one past the highest card in play (and at least index 4)."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text('api_key: "x"\n', encoding="utf-8")
    editor = GpuOverridesEditor(load_config(path))
    assert editor._next_chip_index() == 4  # only the pre-populated 0-3 exist

    editor._driven = {0, 5}
    assert editor._next_chip_index() == 6


def test_chip_variant_encodes_driven_then_detected(tmp_path: Path) -> None:
    """A chip is primary when explicitly driven, success when only detected, default otherwise."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text('api_key: "x"\n', encoding="utf-8")
    editor = GpuOverridesEditor(load_config(path))
    editor._driven = {0}
    editor._detected = {0, 1}
    assert editor._chip_variant(0) == "primary"  # driven wins over detected
    assert editor._chip_variant(1) == "success"  # detected only
    assert editor._chip_variant(2) == "default"  # neither


@pytest.mark.e2e
async def test_chip_selection_writes_the_drive_set(tmp_path: Path) -> None:
    """Selecting a numbered chip puts that card in gpu_device_indices on save (leaving Auto omits it)."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text('api_key: "x"\ndreamer_name: "n"\n', encoding="utf-8")

    class _Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield ConfigEditorView(config_path=path)

    app = _Harness()
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        gpu = editor.query_one(GpuOverridesEditor)
        await pilot.pause()
        # Empty file => Auto mode, so nothing is written for the drive set.
        gpu.on_button_pressed(Button.Pressed(gpu.query_one("#gpu-chip-2", Button)))
        await pilot.pause()
        assert editor._save() is True

    assert read_gpu_device_indices(load_config(path)) == [2]


@pytest.mark.e2e
async def test_auto_chip_clears_the_drive_set(tmp_path: Path) -> None:
    """Pressing the Auto chip drops an explicit gpu_device_indices list back to drive-everything."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text('api_key: "x"\ndreamer_name: "n"\ngpu_device_indices:\n  - 0\n  - 1\n', encoding="utf-8")

    class _Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield ConfigEditorView(config_path=path)

    app = _Harness()
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        gpu = editor.query_one(GpuOverridesEditor)
        await pilot.pause()
        gpu.on_button_pressed(Button.Pressed(gpu.query_one("#gpu-chip-auto", Button)))
        await pilot.pause()
        assert editor._save() is True

    assert "gpu_device_indices" not in load_config(path)


@pytest.mark.e2e
async def test_add_card_button_mounts_the_next_card(tmp_path: Path) -> None:
    """The add-a-card button provisions the next index without any typing, mounting its section."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text('api_key: "x"\ndreamer_name: "n"\n', encoding="utf-8")

    class _Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield ConfigEditorView(config_path=path)

    app = _Harness()
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        gpu = editor.query_one(GpuOverridesEditor)
        await pilot.pause()
        assert not editor.query("#gpuovr-4-max_threads")
        gpu.on_button_pressed(Button.Pressed(gpu.query_one("#gpu-chip-add", Button)))
        await pilot.pause()
        assert editor.query("#gpuovr-4-max_threads")
        assert editor._save() is True

    assert read_gpu_device_indices(load_config(path)) == [4]


def _harness(path: Path) -> App[None]:
    """A bare app hosting one ConfigEditorView over ``path``."""

    class _Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield ConfigEditorView(config_path=path)

    return _Harness()


def _card(index: int, name: str = "RTX 4090") -> SimpleNamespace:
    return SimpleNamespace(device_index=index, device_name=name, kind="cuda")


@pytest.mark.e2e
async def test_a_missing_snapshot_keeps_the_known_cards(tmp_path: Path) -> None:
    """A tick with no snapshot (a worker between spawns) neither drops sections nor resets the banner count."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text('api_key: "x"\ndreamer_name: "n"\n', encoding="utf-8")
    app = _harness(path)
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        gpu = editor.query_one(GpuOverridesEditor)
        await pilot.pause()
        editor.update_cards([_card(0), _card(1)])
        await pilot.pause()
        editor.update_cards([])
        await pilot.pause()
        assert editor.query("#gpuovr-0-max_threads") and editor.query("#gpuovr-1-max_threads")
        assert "2 GPUs detected" in gpu._banner_text()


@pytest.mark.e2e
async def test_saved_card_list_shows_sections_with_a_disclaimer(tmp_path: Path) -> None:
    """A previous session's card list mounts sections before any live source, and says it is unconfirmed."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text('api_key: "x"\ndreamer_name: "n"\n', encoding="utf-8")
    app = _harness(path)
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        gpu = editor.query_one(GpuOverridesEditor)
        await pilot.pause()
        editor.set_remembered_cards(
            KnownGpuInventory(gpus=[KnownGpu(index=0, name="Card A"), KnownGpu(index=1)], recorded_at=0.0),
        )
        await pilot.pause()
        assert editor.query("#gpuovr-1-max_threads")
        assert "Showing the GPU list saved" in gpu._banner_text()
        assert "2 GPUs detected" in gpu._banner_text()
        assert "from saved list, not yet confirmed" in gpu._card_title(1)
        assert gpu._chip_variant(1) == "default"

        # The probe confirms card 0 only: the disclaimer goes, and card 1 reads as missing, not unconfirmed.
        editor.set_installed_cards([KnownGpu(index=0, name="Card A")])
        await pilot.pause()
        assert "Showing the GPU list saved" not in gpu._banner_text()
        assert gpu._chip_variant(0) == "success"
        assert "not found this session" in gpu._card_title(1)
        # Sections are never removed by a source update, so card 1's section (and any edit in it) remains.
        assert editor.query("#gpuovr-1-max_threads")


@pytest.mark.e2e
async def test_probe_lists_cards_the_worker_does_not_drive(tmp_path: Path) -> None:
    """An installed card the worker leaves undriven still gets a section, and the banner says why it is idle."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text('api_key: "x"\ndreamer_name: "n"\n', encoding="utf-8")
    app = _harness(path)
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        gpu = editor.query_one(GpuOverridesEditor)
        await pilot.pause()
        editor.update_cards([_card(0)])
        editor.set_installed_cards([KnownGpu(index=0), KnownGpu(index=1, name="Text card")])
        await pilot.pause()
        assert editor.query("#gpuovr-1-max_threads")
        assert "1 GPU detected" in gpu._banner_text()
        assert "2 are installed" in gpu._banner_text()


@pytest.mark.e2e
async def test_re_adding_a_just_dropped_chip_keeps_it_mounted(tmp_path: Path) -> None:
    """Deselecting an added card and adding it again in one burst leaves a mounted, selected chip.

    A chip removed and re-added before its async removal completes would collide on its id; the chip must
    end up both recorded and actually mounted, never recorded while absent from the strip.
    """
    path = tmp_path / "bridgeData.yaml"
    path.write_text('api_key: "x"\ndreamer_name: "n"\n', encoding="utf-8")
    app = _harness(path)
    async with app.run_test() as pilot:
        gpu = app.query_one(GpuOverridesEditor)
        await pilot.pause()
        add = gpu.query_one("#gpu-chip-add", Button)
        gpu.on_button_pressed(Button.Pressed(add))
        gpu.on_button_pressed(Button.Pressed(gpu._chip_buttons[4]))
        gpu.on_button_pressed(Button.Pressed(add))
        await pilot.pause()
        assert gpu._driven == {4}
        chip = gpu._chip_buttons[4]
        assert chip.is_mounted and chip.parent is gpu.query_one("#gpu-strip")
        assert list(gpu.query("#gpu-chip-4")) == [chip]
        assert chip.variant == "primary"


@pytest.mark.e2e
async def test_opening_the_per_card_tab_probes_once_and_saves_the_list(tmp_path: Path) -> None:
    """The first open of the per-card sub-tab enumerates the cards, shows them, and saves them for next time."""
    config_path = tmp_path / "bridgeData.yaml"
    config_path.write_text("api_key: test\ndreamer_name: TestWorker\n", encoding="utf-8")
    store = AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")
    store.record_onboarding_choice(OnboardingChoice.DECLINED)
    store.set_experience_level(ExperienceLevel.ADVANCED)
    calls: list[int] = []

    def _probe() -> list[KnownGpu]:
        calls.append(1)
        return [KnownGpu(index=0, name="Card A"), KnownGpu(index=1, name="Card B")]

    app = HordeWorkerTUI(FakeSupervisor(), config_path=config_path, app_state_store=store, gpu_probe=_probe)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        app.query_one("#main-tabs", TabbedContent).active = "tab-config"
        await pilot.pause()
        editor = app.query_one(ConfigEditorView)
        subtabs = editor.query_one("#config-subtabs", TabbedContent)
        subtabs.active = "cfgtab-per-gpu"
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert editor.query("#gpuovr-1-max_threads")

        subtabs.active = "cfgtab-dashboard"
        await pilot.pause()
        subtabs.active = "cfgtab-per-gpu"
        await pilot.pause()
        await app.workers.wait_for_complete()

    assert calls == [1]
    saved = store.load().known_gpus
    assert saved is not None and [gpu.index for gpu in saved.gpus] == [0, 1]
