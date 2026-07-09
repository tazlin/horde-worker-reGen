"""Tests for the model picker: detail/flag formatting (unit) and table filters/marking (e2e)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from rich.console import Console
from rich.text import Text
from textual.app import App
from textual.coordinate import Coordinate
from textual.screen import Screen
from textual.widgets import Button, Checkbox, DataTable, Input, Select

from horde_worker_regen.tui.model_catalog import ModelInfo
from horde_worker_regen.tui.widgets.model_picker import _MARKER_COL, ModelPickerModal, ModelPickerResult

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
    if isinstance(renderable, Text):
        return renderable.plain
    console = Console(width=60, color_system=None)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def test_flags_for() -> None:
    """Flags reflect the model's nsfw/inpainting attributes."""
    assert ModelPickerModal._flags_for(_MODELS[2]) == "nsfw"
    assert ModelPickerModal._flags_for(_MODELS[1]) == ""


def test_flags_for_beta_model() -> None:
    """A beta (pending-queue) model is flagged, leading the flag string."""
    beta_model = ModelInfo("Qwen-Image", "qwen_image", nsfw=False, inpainting=False, is_beta=True)
    assert ModelPickerModal._flags_for(beta_model) == "beta"
    assert (
        ModelPickerModal._flags_for(
            ModelInfo("Qwen NSFW", "qwen_image", nsfw=True, inpainting=False, is_beta=True),
        )
        == "beta nsfw"
    )


def test_detail_for_shows_beta_source() -> None:
    """The detail panel labels a beta model's provenance."""
    beta_model = ModelInfo("Qwen-Image", "qwen_image", nsfw=False, inpainting=False, is_beta=True)
    detail = _render(ModelPickerModal._detail_for(beta_model))
    assert "beta" in detail
    assert "pending queue" in detail


def test_detail_for_shows_full_record_with_homepage() -> None:
    """The detail panel renders the full record including a homepage link."""
    detail = _render(ModelPickerModal._detail_for(_MODELS[0]))
    assert "Deliberate" in detail
    assert "stable_diffusion_1" in detail
    assert "versatile" in detail
    assert "anime" in detail
    assert "3.0" in detail
    assert "example.com/deliberate" in detail


def test_marker_and_in_config_membership() -> None:
    """In membership mode the marker and In-config cells reflect both lists."""
    modal = ModelPickerModal(in_target={"Deliberate"}, in_other={"Spicy Model"})
    # A model already in the target list can be marked for removal.
    assert modal._marker_for(_MODELS[0]) == "− Remove"
    assert _render(modal._in_config_cell(_MODELS[0])).strip() == "load"
    # A model in the sibling list is still addable, and shows its membership.
    assert modal._marker_for(_MODELS[2]) == "＋ Add"
    assert _render(modal._in_config_cell(_MODELS[2])).strip() == "skip"
    # An uninvolved model is plain-addable with no membership.
    assert modal._marker_for(_MODELS[1]) == "＋ Add"
    assert _render(modal._in_config_cell(_MODELS[1])).strip() == "-"
    modal._marked_add.add(_MODELS[1].name)
    assert modal._marker_for(_MODELS[1]) == "✕ Clear"
    assert _render(modal._in_config_cell(_MODELS[1])).strip() == "add -> load"
    modal._marked_remove.add(_MODELS[0].name)
    assert modal._marker_for(_MODELS[0]) == "✕ Clear"
    assert _render(modal._in_config_cell(_MODELS[0])).strip() == "remove -> load"


def test_matches_search_spans_metadata() -> None:
    """Search matches name, description, and tags (not just the name)."""
    modal = ModelPickerModal()
    assert modal._matches_search(_MODELS[0], "albedo") is False
    assert modal._matches_search(_MODELS[0], "deliberate") is True
    assert modal._matches_search(_MODELS[0], "versatile") is True  # description
    assert modal._matches_search(_MODELS[0], "anime") is True  # tag
    assert modal._matches_search(_MODELS[1], "albedo") is True


def test_sort_value_orders_by_active_column() -> None:
    """The sort key follows the active column; baseline groups SD 1.5 before SDXL."""
    modal = ModelPickerModal()
    modal._sort_index = 1  # Model name (default).
    by_name = sorted(_MODELS, key=modal._sort_value)
    assert [model.name for model in by_name] == ["AlbedoBase XL (SDXL)", "Deliberate", "Spicy Model"]
    modal._sort_index = 3  # Baseline.
    by_baseline = [model.name for model in sorted(_MODELS, key=modal._sort_value)]
    assert by_baseline.index("Deliberate") < by_baseline.index("AlbedoBase XL (SDXL)")
    assert by_baseline.index("Spicy Model") < by_baseline.index("AlbedoBase XL (SDXL)")


def test_cells_for_has_one_cell_per_column() -> None:
    """Each row supplies exactly one cell per declared column."""
    from horde_worker_regen.tui.widgets.model_picker import _COLUMNS

    assert len(ModelPickerModal().cells_for(_MODELS[0])) == len(_COLUMNS)


class _PickerHost(App[None]):
    """Hosts the picker and records the dismissed result."""

    def __init__(self, screen_factory: Callable[[], Screen] | None = None) -> None:
        super().__init__()
        self.result: ModelPickerResult | list[str] | None = None
        self.result_set = False
        self._screen_factory = screen_factory or (lambda: ModelPickerModal(exclude=set()))

    def on_mount(self) -> None:
        self.push_screen(self._screen_factory(), self._store)

    def _store(self, value: ModelPickerResult | list[str] | None) -> None:
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
    from horde_worker_regen.tui.catalog_cache import CATALOG_CACHE

    CATALOG_CACHE.reset()
    monkeypatch.setattr(
        "horde_worker_regen.tui.catalog_cache.load_image_models",
        lambda: list(_MODELS),
    )
    monkeypatch.setattr(
        "horde_worker_regen.tui.catalog_cache.free_model_bytes",
        lambda: None,
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

        # Marking the single visible row flips its marker cell to the clear-mark label.
        modal._toggle(0)
        await pilot.pause()
        assert str(table.get_cell_at(Coordinate(0, 0))) == "✕ Clear"

        await pilot.click("#picker-add")
        await pilot.pause()

    assert app.result_set
    assert app.result == ["AlbedoBase XL (SDXL)"]


@pytest.mark.e2e
async def test_model_picker_membership_shows_and_blocks_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """Members of the target list stay visible and can be marked for removal."""
    from horde_worker_regen.tui.catalog_cache import CATALOG_CACHE

    CATALOG_CACHE.reset()
    monkeypatch.setattr("horde_worker_regen.tui.catalog_cache.load_image_models", lambda: list(_MODELS))
    monkeypatch.setattr("horde_worker_regen.tui.catalog_cache.free_model_bytes", lambda: None)

    app = _PickerHost(lambda: ModelPickerModal(in_target={"Deliberate"}, in_other={"Spicy Model"}))
    async with app.run_test(size=(150, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ModelPickerModal)
        table = modal.query_one("#picker-table", DataTable)
        await _wait_for_rows(pilot, table, 3)  # nothing is hidden in membership mode

        # Default sort is by name: AlbedoBase, Deliberate, Spicy Model.
        assert str(table.get_cell_at(Coordinate(1, 0))) == "− Remove"
        modal._toggle(1)  # mark Deliberate for removal
        await pilot.pause()
        assert str(table.get_cell_at(Coordinate(1, 0))) == "✕ Clear"
        assert _render(table.get_cell_at(Coordinate(1, 2))).strip() == "remove -> load"

        modal._toggle(0)  # mark AlbedoBase (not in either list)
        await pilot.pause()
        assert _render(table.get_cell_at(Coordinate(0, 2))).strip() == "add -> load"
        await pilot.click("#picker-add")
        await pilot.pause()

    assert app.result == ModelPickerResult(add=["AlbedoBase XL (SDXL)"], remove=["Deliberate"])


@pytest.mark.e2e
async def test_model_picker_single_click_marker_cell_toggles_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single real mouse click on the marker cell marks the row (and clicking it again unmarks it).

    Guards the click path itself: ``DataTable`` stops the ``Click`` event on a cell-cursor click, so a
    handler on the modal never sees it. The other e2e tests call ``_toggle`` directly and so cannot
    catch a dead click handler; this one drives the click through the widget.
    """
    from textual.geometry import Offset

    from horde_worker_regen.tui.catalog_cache import CATALOG_CACHE

    CATALOG_CACHE.reset()
    monkeypatch.setattr("horde_worker_regen.tui.catalog_cache.load_image_models", lambda: list(_MODELS))
    monkeypatch.setattr("horde_worker_regen.tui.catalog_cache.free_model_bytes", lambda: None)

    app = _PickerHost()
    async with app.run_test(size=(150, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ModelPickerModal)
        table = modal.query_one("#picker-table", DataTable)
        await _wait_for_rows(pilot, table, 3)

        # Offset (3, 1): column 0 (marker) of the first data row, below the header at y=0.
        await pilot.click(table, offset=Offset(3, 1))
        await pilot.pause()
        assert str(table.get_cell_at(Coordinate(0, 0))) == "✕ Clear"
        assert modal._visible[0].name in modal._marked_add

        await pilot.click(table, offset=Offset(3, 1))
        await pilot.pause()
        assert str(table.get_cell_at(Coordinate(0, 0))) == "＋ Add"
        assert modal._visible[0].name not in modal._marked_add


@pytest.mark.e2e
async def test_model_picker_single_click_outside_marker_inspects_without_marking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clicking a non-marker cell moves the cursor (showing the detail panel) but never marks the row.

    This is the whole point of the marker-cell gate: a user can click into any row to read its details
    without that click flipping the row's mark state. Only the ＋/✕ cell toggles.
    """
    from textual.geometry import Offset

    from horde_worker_regen.tui.catalog_cache import CATALOG_CACHE

    CATALOG_CACHE.reset()
    monkeypatch.setattr("horde_worker_regen.tui.catalog_cache.load_image_models", lambda: list(_MODELS))
    monkeypatch.setattr("horde_worker_regen.tui.catalog_cache.free_model_bytes", lambda: None)

    app = _PickerHost()
    async with app.run_test(size=(150, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ModelPickerModal)
        table = modal.query_one("#picker-table", DataTable)
        await _wait_for_rows(pilot, table, 3)

        # Offset (25, 2): the Model column (past the 10-wide marker column) of the second data row.
        await pilot.click(table, offset=Offset(25, 2))
        await pilot.pause()
        assert table.cursor_coordinate.column != _MARKER_COL
        assert table.cursor_coordinate.row == 1
        assert modal._current is modal._visible[1]
        assert str(table.get_cell_at(Coordinate(1, 0))) == "＋ Add"
        assert modal._marked_add == set()
        assert modal._marked_remove == set()


@pytest.mark.e2e
async def test_model_picker_disk_and_marked_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    """The on-disk filter and the marked-only filter narrow the table."""
    from horde_worker_regen.tui.catalog_cache import CATALOG_CACHE

    disk_models = [
        ModelInfo("OnDiskModel", "stable_diffusion_1", nsfw=False, inpainting=False, on_disk=True),
        ModelInfo("ToDownloadModel", "stable_diffusion_xl", nsfw=False, inpainting=False, on_disk=False),
    ]
    CATALOG_CACHE.reset()
    monkeypatch.setattr("horde_worker_regen.tui.catalog_cache.load_image_models", lambda: list(disk_models))
    monkeypatch.setattr("horde_worker_regen.tui.catalog_cache.free_model_bytes", lambda: None)

    app = _PickerHost()
    async with app.run_test(size=(150, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ModelPickerModal)
        table = modal.query_one("#picker-table", DataTable)
        await _wait_for_rows(pilot, table, 2)

        modal.query_one("#picker-disk-filter", Select).value = "on"
        await _wait_for_rows(pilot, table, 1)
        assert str(table.get_cell_at(Coordinate(0, 1))) == "OnDiskModel"

        modal.query_one("#picker-disk-filter", Select).value = "off"
        await _wait_for_rows(pilot, table, 1)
        assert str(table.get_cell_at(Coordinate(0, 1))) == "ToDownloadModel"

        modal.query_one("#picker-disk-filter", Select).value = ""
        await _wait_for_rows(pilot, table, 2)

        # Mark one model, then the marked-only filter should leave just it.
        modal._toggle(0)
        await pilot.pause()
        marked_name = modal._visible[0].name
        modal.query_one("#picker-marked-only", Checkbox).value = True
        await _wait_for_rows(pilot, table, 1)
        assert str(table.get_cell_at(Coordinate(0, 1))) == marked_name


@pytest.mark.e2e
async def test_model_picker_clear_marks_requires_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear marks opens a red/green confirmation before dropping staged add/remove marks."""
    from horde_worker_regen.tui.catalog_cache import CATALOG_CACHE

    CATALOG_CACHE.reset()
    monkeypatch.setattr("horde_worker_regen.tui.catalog_cache.load_image_models", lambda: list(_MODELS))
    monkeypatch.setattr("horde_worker_regen.tui.catalog_cache.free_model_bytes", lambda: None)

    app = _PickerHost(lambda: ModelPickerModal(in_target={"Deliberate"}))
    async with app.run_test(size=(150, 44)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ModelPickerModal)
        table = modal.query_one("#picker-table", DataTable)
        await _wait_for_rows(pilot, table, 3)

        modal._toggle(0)  # mark add
        modal._toggle(1)  # mark remove (Deliberate)
        await pilot.pause()
        assert modal._marked_add
        assert modal._marked_remove

        await pilot.click("#picker-clear-marks")
        await pilot.pause()
        confirm = app.screen
        assert confirm.query_one("#confirm-yes", Button).variant == "error"
        assert confirm.query_one("#confirm-no", Button).variant == "success"
        assert modal._has_marks()

        await pilot.click("#confirm-no")
        await pilot.pause()
        assert modal._has_marks()

        await pilot.click("#picker-clear-marks")
        await pilot.pause()
        await pilot.click("#confirm-yes")
        await pilot.pause()
        assert not modal._has_marks()
