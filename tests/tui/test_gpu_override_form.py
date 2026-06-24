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
from textual.widgets import Button, Input, Switch

from horde_worker_regen.bridge_data.data_model import GpuOverride
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
