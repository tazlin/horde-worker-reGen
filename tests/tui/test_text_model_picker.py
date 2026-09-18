"""Tests for the text model picker: row/detail formatting (unit) and the table, search and choice (e2e)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from rich.console import Console
from rich.text import Text
from textual.app import App
from textual.widgets import Button, DataTable, Input

from horde_worker_regen.tui.model_catalog import TextModelInfo
from horde_worker_regen.tui.widgets.text_model_picker import _DIVIDER_NAME, TextModelPickerModal

_MODELS = [
    TextModelInfo(
        name="meta-llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M",
        has_file=True,
        quant="Q4_K_M",
        size_bytes=4_920_739_232,
        footprint_mb=5954,
        context=8192,
        tokens_per_second=106.0,
        source_card="NVIDIA GeForce RTX 4070 Ti SUPER (16 GB)",
        parameters_count=8_000_000_000,
        homepage="https://example.com/llama",
        description="An instruction-tuned model.",
        on_disk=True,
        target_path="T:/text-models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    ),
    TextModelInfo(
        name="mistralai/Mistral-Nemo-Instruct-2407-Q4_K_M",
        has_file=True,
        quant="Q4_K_M",
        size_bytes=7_477_208_192,
        footprint_mb=8633,
        context=8192,
        target_path="T:/text-models/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
    ),
    TextModelInfo(name="some-author/Unmeasured-Model", has_file=False),
]


def _render(renderable: object) -> str:
    """Render a Rich renderable to plain text."""
    if isinstance(renderable, Text):
        return renderable.plain
    if isinstance(renderable, str):
        return renderable
    console = Console(width=60, color_system=None)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def test_cells_carry_the_facts_a_choice_is_made_on() -> None:
    """A row states the quant, the file size, the measured footprint, the context and on-disk presence."""
    cells = [_render(cell) for cell in TextModelPickerModal.cells_for(_MODELS[0])]

    assert cells[0] == _MODELS[0].name
    assert cells[1] == "Q4_K_M"
    assert "GB" in cells[2]
    assert "GB" in cells[3]
    assert cells[4] == "8192"
    assert cells[5] == "on disk"


def test_an_unmeasured_model_reads_as_unknown_not_as_free() -> None:
    """A record nobody has run has no footprint, which the row must not render as a zero cost."""
    cells = [_render(cell) for cell in TextModelPickerModal.cells_for(_MODELS[2])]

    assert cells[1] == "-"
    assert cells[3] == "-"
    assert cells[4] == "-"


def test_rows_place_the_divider_before_the_first_model_with_no_file() -> None:
    """Models with nothing to load are separated from the choosable ones by exactly one divider."""
    modal = TextModelPickerModal()

    rows = modal.rows_for(_MODELS)

    assert [row.name if row is not None else _DIVIDER_NAME for row in rows] == [
        _MODELS[0].name,
        _MODELS[1].name,
        _DIVIDER_NAME,
        _MODELS[2].name,
    ]
    assert modal.rows_for(_MODELS[:2]) == list(_MODELS[:2])


def test_detail_names_the_measurement_and_where_it_came_from() -> None:
    """A footprint is only comparable beside the card and context it was measured at, so both are shown."""
    detail = _render(TextModelPickerModal._detail_for(_MODELS[0]))

    assert "Q4_K_M" in detail
    assert "8192" in detail
    assert "4070" in detail
    assert "106" in detail
    assert "example.com/llama" in detail


def test_detail_says_an_unmeasured_model_is_unknown_and_has_nothing_to_load() -> None:
    """Both absences are stated, since either one silently read as "fine" would mislead a choice."""
    detail = _render(TextModelPickerModal._detail_for(_MODELS[2]))

    assert "none on offer" in detail
    assert "unknown" in detail


def test_search_matches_the_name_only() -> None:
    """The scribe chooses by name, so search is over the name and nothing else."""
    assert TextModelPickerModal._matches_search(_MODELS[0], "") is True
    assert TextModelPickerModal._matches_search(_MODELS[0], "llama") is True
    assert TextModelPickerModal._matches_search(_MODELS[0], "mistral") is False
    assert TextModelPickerModal._matches_search(_MODELS[0], "instruction") is False


class _PickerHost(App[None]):
    """Hosts the picker and records the dismissed result."""

    def __init__(self) -> None:
        super().__init__()
        self.chosen: str | None = None
        self.dismissed = False

    def on_mount(self) -> None:
        self.push_screen(TextModelPickerModal(), self._store)

    def _store(self, result: str | None) -> None:
        self.dismissed = True
        self.chosen = result


@pytest.fixture
def stubbed_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the picker's reference read from the fixtures above, so no catalogue is loaded."""
    monkeypatch.setattr(
        "horde_worker_regen.tui.widgets.text_model_picker.load_text_models",
        lambda text_models_dir=None: list(_MODELS),
    )


async def _wait_until(pilot: object, condition: Callable[[], bool], *, what: str) -> None:
    """Drive the app until ``condition`` holds.

    The reference read happens on a worker thread, so how many pauses its result takes to reach the
    table depends on the host's load, not on the code under test.
    """
    for _ in range(200):
        await pilot.pause()  # type: ignore[attr-defined]
        if condition():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


async def _picker_of(pilot: object, app: _PickerHost) -> tuple[TextModelPickerModal, DataTable]:
    """Return the pushed picker and its table, once the rows are in place."""
    modal = app.screen
    assert isinstance(modal, TextModelPickerModal)
    table = modal.query_one("#text-picker-table", DataTable)
    await _wait_until(pilot, lambda: table.row_count > 0, what="the picker table to fill")
    return modal, table


@pytest.mark.e2e
@pytest.mark.usefixtures("stubbed_catalogue")
async def test_the_table_lists_every_model_and_search_narrows_it() -> None:
    """Every known model is a row, and typing in the search box filters to the matching names."""
    app = _PickerHost()
    async with app.run_test(size=(140, 40)) as pilot:
        modal, table = await _picker_of(pilot, app)

        assert table.row_count == len(_MODELS) + 1  # the divider is a row of its own

        modal.query_one("#text-picker-search", Input).value = "mistral"
        await _wait_until(pilot, lambda: table.row_count == 1, what="the search to narrow the table")


@pytest.mark.e2e
@pytest.mark.usefixtures("stubbed_catalogue")
async def test_choosing_a_row_dismisses_with_its_name() -> None:
    """Choose answers the config field with the highlighted model's reference name."""
    app = _PickerHost()
    async with app.run_test(size=(140, 40)) as pilot:
        modal, table = await _picker_of(pilot, app)

        table.move_cursor(row=1)
        await pilot.pause()
        modal.query_one("#text-picker-choose", Button).press()
        await _wait_until(pilot, lambda: app.dismissed, what="the picker to dismiss")

    assert app.chosen == _MODELS[1].name


@pytest.mark.e2e
@pytest.mark.usefixtures("stubbed_catalogue")
async def test_a_model_with_no_file_cannot_be_chosen() -> None:
    """There is no artefact to load for it, so choosing it would configure a worker that cannot start."""
    app = _PickerHost()
    async with app.run_test(size=(140, 40)) as pilot:
        modal, table = await _picker_of(pilot, app)

        table.move_cursor(row=3)  # past the divider
        await pilot.pause()
        modal.query_one("#text-picker-choose", Button).press()
        for _ in range(5):
            await pilot.pause()

        assert app.dismissed is False


@pytest.mark.e2e
@pytest.mark.usefixtures("stubbed_catalogue")
async def test_cancel_answers_nothing() -> None:
    """A cancelled picker must leave the field as the operator typed it."""
    app = _PickerHost()
    async with app.run_test(size=(140, 40)) as pilot:
        modal, _table = await _picker_of(pilot, app)

        modal.query_one("#text-picker-cancel", Button).press()
        await _wait_until(pilot, lambda: app.dismissed, what="the picker to dismiss")

    assert app.chosen is None
