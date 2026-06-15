"""End-to-end (headless) tests for the interactive model-list controls."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from horde_worker_regen.tui.widgets.meta_builder import MetaBuilderModal
from horde_worker_regen.tui.widgets.model_list_editor import ModelListEditor


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
