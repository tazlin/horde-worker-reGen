"""Headless tests for the config editor's save safeties (changed-fields-only, no soft-locks)."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, TabbedContent

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
async def test_invalid_edit_reports_error_and_jumps_to_offending_tab(tmp_path: Path) -> None:
    """Editing a field to an invalid value blocks the save and surfaces it on that field's sub-tab."""
    app, path = await _mount(tmp_path, 'api_key: "x"\ndreamer_name: "n"\n')
    async with app.run_test() as pilot:
        editor = app.query_one(ConfigEditorView)
        await pilot.pause()
        # max_lora_cache_size lives on the LoRA sub-tab; 5 is below its minimum of 10.
        editor.query_one("#cfg-max_lora_cache_size", Input).value = "5"
        assert editor._save() is False
        await pilot.pause()
        assert editor.query_one("#config-subtabs", TabbedContent).active == "cfgtab-lora"
