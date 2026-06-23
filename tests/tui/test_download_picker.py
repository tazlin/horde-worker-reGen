"""Tests for the download picker modal: default selection, confirm payload, and empty-confirm handling."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Checkbox, SelectionList

from horde_worker_regen.tui.widgets.download_picker import (
    DownloadPickerModal,
    DownloadPickerRow,
    DownloadSelection,
)

_ROWS = [
    DownloadPickerRow(name="Present", baseline="stable_diffusion_xl", size_bytes=6_000_000_000, on_disk=True),
    DownloadPickerRow(name="Missing A", baseline="stable_diffusion_1", size_bytes=2_000_000_000, on_disk=False),
    DownloadPickerRow(name="Missing B", baseline="flux_1", size_bytes=None, on_disk=False),
]


def test_intro_counts_missing() -> None:
    """The intro line states how many configured models are not yet on disk (the pre-selected default)."""
    intro = DownloadPickerModal(_ROWS)._intro().plain
    assert "2 not yet on disk" in intro


def test_prompt_labels_presence() -> None:
    """Each row's label carries its name, baseline, and on-disk vs missing state."""
    present = DownloadPickerModal._prompt(_ROWS[0]).plain
    missing = DownloadPickerModal._prompt(_ROWS[1]).plain
    assert "Present" in present and "on disk" in present
    assert "Missing A" in missing and "missing" in missing


class _PickerHost(App[None]):
    """Hosts the download picker and records the dismissed selection."""

    def __init__(self, rows: list[DownloadPickerRow]) -> None:
        super().__init__()
        self.result: DownloadSelection | None = None
        self.result_set = False
        self._rows = rows

    def on_mount(self) -> None:
        self.push_screen(DownloadPickerModal(self._rows), self._store)

    def _store(self, value: DownloadSelection | None) -> None:
        self.result = value
        self.result_set = True


@pytest.mark.e2e
async def test_default_selection_is_the_missing_models() -> None:
    """The picker pre-selects exactly the not-on-disk models, and confirm returns them sorted."""
    app = _PickerHost(_ROWS)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, DownloadPickerModal)
        selection_list = modal.query_one("#download-picker-list", SelectionList)
        assert set(selection_list.selected) == {"Missing A", "Missing B"}

        await pilot.click("#download-picker-confirm")
        await pilot.pause()

    assert app.result_set
    assert app.result is not None
    assert app.result.model_names == ["Missing A", "Missing B"]
    assert app.result.include_aux is False


@pytest.mark.e2e
async def test_include_aux_toggle_is_carried() -> None:
    """Ticking the aux checkbox is reflected in the returned selection."""
    app = _PickerHost(_ROWS)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, DownloadPickerModal)
        modal.query_one("#download-picker-aux", Checkbox).value = True

        await pilot.click("#download-picker-confirm")
        await pilot.pause()

    assert app.result is not None
    assert app.result.include_aux is True


@pytest.mark.e2e
async def test_empty_confirm_dismisses_as_none() -> None:
    """Deselecting everything with no aux makes confirm a cancel (no empty download request)."""
    app = _PickerHost(_ROWS)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, DownloadPickerModal)
        modal.query_one("#download-picker-list", SelectionList).deselect_all()

        await pilot.click("#download-picker-confirm")
        await pilot.pause()

    assert app.result_set
    assert app.result is None
