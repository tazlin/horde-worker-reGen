"""Headless smoke test for the unified model-manager widget's preview rendering."""

from __future__ import annotations

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.widgets import Static, Switch

from horde_worker_regen.tui.model_catalog import ModelInfo
from horde_worker_regen.tui.widgets.model_manager import ModelManagerView

_CATALOG = [
    ModelInfo(
        name="AlbedoBase XL",
        baseline="stable_diffusion_xl",
        nsfw=False,
        inpainting=False,
        size_on_disk_bytes=6_000_000_000,
        on_disk=True,
    ),
    ModelInfo(
        name="Juggernaut XL",
        baseline="stable_diffusion_xl",
        nsfw=False,
        inpainting=False,
        size_on_disk_bytes=6_500_000_000,
        on_disk=False,
    ),
]


class _Host(App[None]):
    """Minimal app hosting a single ModelManagerView."""

    def __init__(self, view: ModelManagerView) -> None:
        super().__init__()
        self._view = view

    def compose(self) -> ComposeResult:
        yield self._view


def _plain(renderable: object) -> str:
    return renderable.plain if isinstance(renderable, Text) else str(renderable)


@pytest.mark.e2e
async def test_manager_renders_effective_preview_when_catalog_injected() -> None:
    """Once a catalog is available, the headline reports the effective count and the rows render."""
    view = ModelManagerView(["all sdxl"], [], load_large_models=True)
    app = _Host(view)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        # Inject a cached catalog as if the reference were already loaded, then recompute (no network).
        view._catalog = _CATALOG
        view._free_disk_bytes = 10**12
        view._recompute()
        await pilot.pause()

        assert view._last_result is not None
        assert {model.name for model in view._last_result.included} == {"AlbedoBase XL", "Juggernaut XL"}

        headline = _plain(view.query_one("#mm-headline", Static).render())
        assert "EFFECTIVE" in headline
        assert "2 will load" in headline


@pytest.mark.e2e
async def test_resolution_controls_stack_on_80_column_layout() -> None:
    """The Resolve/policy control band remains usable on a narrow terminal."""
    view = ModelManagerView(["top 2"], [], load_large_models=False, only_on_disk=True)
    app = _Host(view)
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()

        controls = view.query_one("#mm-resolution-controls", Grid)
        assert controls.styles.grid_size_columns == 1
        assert view.query_one("#cfg-load_large_models", Switch).value is False
        assert view.query_one("#cfg-only_models_on_disk", Switch).value is True
