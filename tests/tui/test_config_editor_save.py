"""Headless tests for the config editor's save safeties (changed-fields-only, no soft-locks)."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Select, Switch, TabbedContent

from horde_worker_regen.tui.config_form import load_config
from horde_worker_regen.tui.widgets.config_editor import ConfigEditorView


class _EditorHarness(App[None]):
    """A minimal app that mounts only the config editor over a given file."""

    def __init__(self, config_path: Path) -> None:
        super().__init__()
        self._config_path = config_path

    def compose(self) -> ComposeResult:
        yield ConfigEditorView(config_path=self._config_path)


async def _mount(tmp_path: Path, yaml_text: str) -> tuple[_EditorHarness, Path]:
    """Write *yaml_text* to a temp config and return the harness plus its path (not yet run)."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return _EditorHarness(path), path


@pytest.mark.e2e
async def test_save_writes_only_changed_fields(tmp_path: Path) -> None:
    """Saving touches only edited keys: it never floods the file with unedited defaults."""
    app, path = await _mount(tmp_path, 'api_key: "x"\ndreamer_name: "n"\nmax_threads: 1\n')
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        editor.query_one("#cfg-max_threads", Input).value = "3"
        assert editor._save() is True
        await pilot.pause()

    written = path.read_text(encoding="utf-8")
    reloaded = load_config(path)
    assert reloaded["max_threads"] == 3
    # No unedited absent key (a default the editor merely displayed) leaked into the file.
    assert "min_lora_disk_free_gb" not in written
    assert "queue_size" not in written
    assert "max_power" not in written


@pytest.mark.e2e
async def test_absent_float_field_does_not_soft_lock_save(tmp_path: Path) -> None:
    """An absent float field (shown as its 1.0 default) never blocks saving an unrelated change.

    This is the original soft-lock: the field was typed INT, so its float default failed integer
    coercion and aborted every save. It must now save cleanly without the operator touching it.
    """
    app, path = await _mount(tmp_path, 'api_key: "x"\ndreamer_name: "n"\nmax_threads: 1\n')
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        # The field shows its float default but is left untouched.
        assert editor.query_one("#cfg-min_lora_disk_free_gb", Input).value == "1.0"
        editor.query_one("#cfg-max_threads", Input).value = "2"
        assert editor._save() is True


@pytest.mark.e2e
async def test_editing_float_field_round_trips(tmp_path: Path) -> None:
    """A fractional value entered into the float field is saved as-is."""
    app, path = await _mount(tmp_path, 'api_key: "x"\ndreamer_name: "n"\n')
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        editor.query_one("#cfg-min_lora_disk_free_gb", Input).value = "1.5"
        assert editor._save() is True
        await pilot.pause()

    assert load_config(path)["min_lora_disk_free_gb"] == 1.5


@pytest.mark.e2e
async def test_select_field_round_trips(tmp_path: Path) -> None:
    """A single-choice config field renders as a Select and saves the chosen value."""
    app, path = await _mount(tmp_path, 'api_key: "x"\ndreamer_name: "n"\n')
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        selector = editor.query_one("#cfg-dedicated_post_processing", Select)
        selector.value = "off"
        assert editor._save() is True
        await pilot.pause()

    assert load_config(path)["dedicated_post_processing"] == "off"


@pytest.mark.e2e
async def test_preexisting_invalid_value_does_not_block_unrelated_save(tmp_path: Path) -> None:
    """A value already out of bounds on disk (and untouched) cannot block an unrelated edit."""
    app, path = await _mount(
        tmp_path,
        'api_key: "x"\ndreamer_name: "n"\nmax_threads: 1\nmax_lora_cache_size: 1\n',
    )
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        editor.query_one("#cfg-max_threads", Input).value = "2"
        assert editor._save() is True
        await pilot.pause()

    reloaded = load_config(path)
    assert reloaded["max_threads"] == 2
    # The untouched, already-invalid value is left exactly as the operator had it.
    assert reloaded["max_lora_cache_size"] == 1


@pytest.mark.e2e
async def test_save_succeeds_with_unrendered_dry_run_field_present(tmp_path: Path) -> None:
    """A field whose section has no sub-tab (the hidden Dry-run flags) must not break save/reload.

    These developer-only flags exist in ``CONFIG_FIELDS`` but are intentionally left out of
    ``CONFIG_SUBTABS``, so they are never composed. The field-walking loops used to query the
    uncomposed widget and raise ``NoMatches``, soft-locking Save/Reload entirely. Saving an
    unrelated edit must now succeed and leave the YAML-only flag untouched.
    """
    app, path = await _mount(
        tmp_path,
        'api_key: "x"\ndreamer_name: "n"\nmax_threads: 1\ndry_run_skip_inference: true\n',
    )
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        # The hidden flag has no widget to query.
        assert not editor.query("#cfg-dry_run_skip_inference")
        editor.query_one("#cfg-max_threads", Input).value = "4"
        assert editor._save() is True
        await pilot.pause()
        # Reload also walks every field and must not trip over the uncomposed flag.
        editor.reload_from_disk()
        await pilot.pause()

    reloaded = load_config(path)
    assert reloaded["max_threads"] == 4
    # The YAML-only developer flag is preserved verbatim, neither dropped nor coerced.
    assert reloaded["dry_run_skip_inference"] is True


@pytest.mark.e2e
async def test_save_succeeds_with_hidden_advanced_field_present(tmp_path: Path) -> None:
    """A hidden catalogued field in a rendered section is preserved without a widget."""
    app, path = await _mount(
        tmp_path,
        'api_key: "x"\ndreamer_name: "n"\nmax_threads: 1\nenable_pipeline_disaggregation: true\n',
    )
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        assert not editor.query("#cfg-enable_pipeline_disaggregation")
        editor.query_one("#cfg-max_threads", Input).value = "4"
        assert editor._save() is True
        await pilot.pause()

    reloaded = load_config(path)
    assert reloaded["max_threads"] == 4
    assert reloaded["enable_pipeline_disaggregation"] is True


@pytest.mark.e2e
async def test_invalid_edit_reports_error_and_jumps_to_offending_tab(tmp_path: Path) -> None:
    """Editing a field to an invalid value blocks the save and surfaces it on that field's sub-tab."""
    app, path = await _mount(tmp_path, 'api_key: "x"\ndreamer_name: "n"\n')
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        # max_lora_cache_size lives on the LoRA & Downloads sub-tab; 5 is below its minimum of 10.
        editor.query_one("#cfg-max_lora_cache_size", Input).value = "5"
        assert editor._save() is False
        await pilot.pause()
        assert editor.query_one("#config-subtabs", TabbedContent).active == "cfgtab-lora-downloads"


@pytest.mark.e2e
async def test_default_dreamer_name_blocks_save_and_jumps_to_essentials(tmp_path: Path) -> None:
    """The reserved-default dreamer name cannot be saved; the editor jumps to its Essentials tab.

    Identity names are validated unconditionally (not only when edited), so the still-default name
    blocks the save even with no other change, which is exactly the config that aborts the worker.
    """
    app, path = await _mount(tmp_path, 'api_key: "x"\ndreamer_name: "An Awesome Dreamer"\n')
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        assert editor._save() is False
        await pilot.pause()
        assert editor.query_one("#config-subtabs", TabbedContent).active == "cfgtab-essentials"


@pytest.mark.e2e
async def test_empty_dreamer_name_blocks_save(tmp_path: Path) -> None:
    """A blank dreamer name blocks the save (it is the worker's required horde-wide identity)."""
    app, path = await _mount(tmp_path, 'api_key: "x"\ndreamer_name: "Good Name"\n')
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        editor.query_one("#cfg-dreamer_name", Input).value = "   "
        assert editor._save() is False


@pytest.mark.e2e
async def test_alchemy_default_name_blocks_save_and_jumps_to_alchemy(tmp_path: Path) -> None:
    """With alchemy enabled, the reserved-default alchemist name blocks the save and jumps to Alchemy."""
    app, path = await _mount(
        tmp_path,
        'api_key: "x"\ndreamer_name: "Good Name"\nalchemist: true\nalchemist_name: "An Awesome Alchemist"\n',
    )
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        assert editor._save() is False
        await pilot.pause()
        assert editor.query_one("#config-subtabs", TabbedContent).active == "cfgtab-alchemy"


@pytest.mark.e2e
async def test_alchemy_name_equal_to_dreamer_blocks_save(tmp_path: Path) -> None:
    """With alchemy enabled, an alchemist name equal to the dreamer name blocks the save."""
    app, path = await _mount(
        tmp_path,
        'api_key: "x"\ndreamer_name: "Same Name"\nalchemist: true\nalchemist_name: "Same Name"\n',
    )
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        assert editor._save() is False


@pytest.mark.e2e
async def test_alchemy_name_only_required_when_enabled(tmp_path: Path) -> None:
    """A default/blank alchemist name does not block the save while alchemy is disabled."""
    app, path = await _mount(
        tmp_path,
        'api_key: "x"\ndreamer_name: "Good Name"\nalchemist: false\nalchemist_name: "An Awesome Alchemist"\n',
    )
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        editor.query_one("#cfg-max_threads", Input).value = "2"
        assert editor._save() is True


@pytest.mark.e2e
async def test_valid_identity_names_save(tmp_path: Path) -> None:
    """Replacing the default dreamer name with a unique one saves it to disk."""
    app, path = await _mount(tmp_path, 'api_key: "x"\ndreamer_name: "An Awesome Dreamer"\n')
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        editor.query_one("#cfg-dreamer_name", Input).value = "My Unique Worker"
        assert editor._save() is True
        await pilot.pause()

    assert load_config(path)["dreamer_name"] == "My Unique Worker"


@pytest.mark.e2e
async def test_interlock_validation_blocks_save_without_worker(tmp_path: Path) -> None:
    """Interlocked settings are rejected by the editor before the worker validates the file."""
    app, path = await _mount(
        tmp_path,
        'api_key: "x"\ndreamer_name: "Good Name"\nallow_img2img: true\nallow_painting: true\n',
    )
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        editor.query_one("#cfg-allow_img2img", Switch).value = False
        assert editor._save() is False
        await pilot.pause()
        assert editor.query_one("#config-subtabs", TabbedContent).active == "cfgtab-features"


@pytest.mark.e2e
async def test_preset_changes_apply_to_live_form_before_save(tmp_path: Path) -> None:
    """Preset application updates widgets but still requires an explicit save."""
    app, path = await _mount(tmp_path, 'api_key: "x"\ndreamer_name: "Good Name"\nmax_threads: 1\nqueue_size: 1\n')
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        editor._apply_preset_changes(
            {
                "queue_size": 0,
                "max_power": 32,
                "models_to_load": ["Deliberate"],
                "load_large_models": False,
            }
        )
        await pilot.pause()
        assert editor.query_one("#cfg-queue_size", Input).value == "0"
        assert editor.query_one("#cfg-max_power", Input).value == "32"
        assert load_config(path)["queue_size"] == 1
        assert editor._save() is True

    reloaded = load_config(path)
    assert reloaded["queue_size"] == 0
    assert reloaded["max_power"] == 32
    assert list(reloaded["models_to_load"]) == ["Deliberate"]


@pytest.mark.e2e
async def test_optional_sampling_lease_slots_can_be_cleared(tmp_path: Path) -> None:
    """The form accepts a blank sampling-lease slot count, meaning it tracks max_threads."""
    app, path = await _mount(
        tmp_path,
        'api_key: "x"\ndreamer_name: "Good Name"\ngpu_sampling_lease_slots: 2\n',
    )
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        field = editor.query_one("#cfg-gpu_sampling_lease_slots", Input)
        assert field.type == "text"
        field.value = ""
        assert editor._save() is True
        await pilot.pause()

    assert load_config(path)["gpu_sampling_lease_slots"] is None


@pytest.mark.e2e
async def test_action_bar_separates_presets_and_highlights_restart_edits(tmp_path: Path) -> None:
    """Apply preset is visually neutral/separate; restart is emphasized only for restart-locked edits."""
    app, path = await _mount(
        tmp_path,
        'api_key: "x"\ndreamer_name: "Good Name"\nmax_threads: 1\nmax_batch: 1\n',
    )
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        assert editor.query_one("#config-save", Button).variant == "success"
        assert editor.query_one("#config-preset", Button).variant == "default"
        assert editor.query_one("#config-actions-separator") is not None
        assert editor.query_one("#config-restart", Button).variant == "default"

        editor.query_one("#cfg-max_batch", Input).value = "2"
        await pilot.pause()
        assert editor.query_one("#config-restart", Button).variant == "default"

        editor.query_one("#cfg-max_threads", Input).value = "2"
        await pilot.pause()
        assert editor.query_one("#config-restart", Button).variant == "warning"


@pytest.mark.e2e
async def test_alchemy_toggle_via_widget_drives_name_requirement(tmp_path: Path) -> None:
    """Turning alchemy on in the form (without saving it first) makes the alchemist name required."""
    app, path = await _mount(tmp_path, 'api_key: "x"\ndreamer_name: "Good Name"\n')
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        editor.query_one("#cfg-alchemist", Switch).value = True
        assert editor._save() is False
        await pilot.pause()
        assert editor.query_one("#config-subtabs", TabbedContent).active == "cfgtab-alchemy"
