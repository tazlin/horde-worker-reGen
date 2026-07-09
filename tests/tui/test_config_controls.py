"""End-to-end (headless) tests for the interactive model-list controls."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from horde_worker_regen.tui.widgets.meta_builder import MetaBuilderModal
from horde_worker_regen.tui.widgets.model_list_editor import ModelListEditor
from horde_worker_regen.tui.widgets.model_picker import ModelPickerResult


class _EditorHost(App[None]):
    """A minimal app that hosts a single ModelListEditor for testing."""

    def __init__(self, editor: ModelListEditor) -> None:
        super().__init__()
        self._editor = editor

    def compose(self) -> ComposeResult:
        yield self._editor


@pytest.mark.e2e
async def test_model_list_editor_add_remove_dedup() -> None:
    """The list editor mounts pre-filled and supports add (with de-dup), set, and clear."""
    editor = ModelListEditor("models_to_load", ["top 2", "Deliberate"])
    app = _EditorHost(editor)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert editor.values() == ["top 2", "Deliberate"]

        editor.add_entry("AlbedoBase XL (SDXL)")
        await pilot.pause()
        assert "AlbedoBase XL (SDXL)" in editor.values()

        editor.add_entry("AlbedoBase XL (SDXL)")  # duplicate ignored
        await pilot.pause()
        assert editor.values().count("AlbedoBase XL (SDXL)") == 1

        editor.set_values([])
        await pilot.pause()
        assert editor.values() == []


@pytest.mark.e2e
async def test_model_list_clear_requires_confirmation() -> None:
    """The Clear button asks before removing every in-progress entry."""
    editor = ModelListEditor("models_to_load", ["top 2", "Deliberate"])
    app = _EditorHost(editor)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#mle-clear-models_to_load")
        await pilot.pause()

        confirm = app.screen
        assert confirm.query_one("#confirm-yes", Button).variant == "error"
        assert confirm.query_one("#confirm-no", Button).variant == "success"
        assert editor.values() == ["top 2", "Deliberate"]

        await pilot.click("#confirm-no")
        await pilot.pause()
        assert editor.values() == ["top 2", "Deliberate"]

        await pilot.click("#mle-clear-models_to_load")
        await pilot.pause()
        await pilot.click("#confirm-yes")
        await pilot.pause()
        assert editor.values() == []


@pytest.mark.e2e
async def test_model_list_applies_picker_add_and_remove_marks() -> None:
    """Picker membership results can add new models and remove existing exact model entries."""
    editor = ModelListEditor("models_to_load", ["Deliberate", "Spicy Model", "top 2"])
    app = _EditorHost(editor)
    async with app.run_test() as pilot:
        await pilot.pause()
        editor._on_picker_result(ModelPickerResult(add=["AlbedoBase XL (SDXL)"], remove=["Spicy Model"]))
        await pilot.pause()

    assert editor.values() == ["Deliberate", "top 2", "AlbedoBase XL (SDXL)"]


class _MetaHost(App[None]):
    """A minimal app that opens the meta-builder modal and records its result."""

    def __init__(self) -> None:
        super().__init__()
        self.result: str | None = None
        self.result_set = False

    def on_mount(self) -> None:
        self.push_screen(MetaBuilderModal(), self._store)

    def _store(self, value: str | None) -> None:
        self.result = value
        self.result_set = True


@pytest.mark.e2e
async def test_meta_builder_modal_builds_top_n() -> None:
    """The meta-builder modal returns the composed instruction for the default selection."""
    app = _MetaHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#meta-add")
        await pilot.pause()
    assert app.result_set
    assert app.result == "top 5"
