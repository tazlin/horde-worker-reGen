"""Tests for the model picker: detail/flag formatting (unit) and table filters/marking (e2e)."""

from __future__ import annotations

import asyncio

import pytest
from rich.console import Console
from textual.app import App
from textual.coordinate import Coordinate
from textual.widgets import Checkbox, DataTable, Input

from horde_worker_regen.tui.model_catalog import ModelInfo
from horde_worker_regen.tui.widgets.model_picker import ModelPickerModal

_MODELS = [
    ModelInfo(
        "Deliberate",
        "stable_diffusion_1",
        nsfw=False,
        inpainting=False,
        style="generalist",
        description="A versatile SD1.5 model.",
        version="3.0",
        homepage="https://example.com/deliberate",
        tags=("general", "anime"),
    ),
    ModelInfo("AlbedoBase XL (SDXL)", "stable_diffusion_xl", nsfw=False, inpainting=False),
    ModelInfo("Spicy Model", "stable_diffusion_1", nsfw=True, inpainting=False),
]


def _render(renderable: object) -> str:
    """Render a Rich renderable to plain text."""
    console = Console(width=60)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def test_flags_for() -> None:
    """Flags reflect the model's nsfw/inpainting attributes."""
    assert ModelPickerModal._flags_for(_MODELS[2]) == "nsfw"
    assert ModelPickerModal._flags_for(_MODELS[1]) == ""


def test_detail_for_shows_full_record_with_homepage() -> None:
    """The detail panel renders the full record including a homepage link."""
    detail = _render(ModelPickerModal._detail_for(_MODELS[0]))
    assert "Deliberate" in detail
    assert "stable_diffusion_1" in detail
    assert "versatile" in detail
    assert "anime" in detail
    assert "3.0" in detail
    assert "example.com/deliberate" in detail


class _PickerHost(App[None]):
    """Hosts the picker and records the dismissed result."""

    def __init__(self) -> None:
        super().__init__()
        self.result: list[str] | None = None
        self.result_set = False

    def on_mount(self) -> None:
        self.push_screen(ModelPickerModal(exclude=set()), self._store)

    def _store(self, value: list[str] | None) -> None:
        self.result = value
        self.result_set = True


async def _wait_for_rows(pilot: object, table: DataTable, count: int) -> None:
    """Pump the event loop until the table reaches the expected row count."""
    for _ in range(40):
        await pilot.pause()  # type: ignore[attr-defined]
        if table.row_count == count:
            return
        await asyncio.sleep(0.05)
    assert table.row_count == count


@pytest.mark.e2e
async def test_model_picker_filters_and_marking(monkeypatch: pytest.MonkeyPatch) -> None:
    """The (patched) reference populates the table; filters narrow it; marking returns the names."""
    monkeypatch.setattr(
        "horde_worker_regen.tui.widgets.model_picker.load_image_models",
        lambda: list(_MODELS),
    )
    app = _PickerHost()
    async with app.run_test(size=(130, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ModelPickerModal)
        table = modal.query_one("#picker-table", DataTable)
        await _wait_for_rows(pilot, table, 3)

        modal.query_one("#picker-nsfw", Checkbox).value = False
        await _wait_for_rows(pilot, table, 2)

        modal.query_one("#picker-search", Input).value = "albedo"
        await _wait_for_rows(pilot, table, 1)

        # Marking the single visible row flips its state cell to the remove glyph.
        modal._toggle(0)
        await pilot.pause()
        assert str(table.get_cell_at(Coordinate(0, 0))) == "✕"

        await pilot.click("#picker-add")
        await pilot.pause()

    assert app.result_set
    assert app.result == ["AlbedoBase XL (SDXL)"]
