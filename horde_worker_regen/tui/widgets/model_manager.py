"""The unified model-config panel: load/skip rules plus a live preview of the models that will load.

Replaces the two opaque ``models_to_load`` / ``models_to_skip`` list editors with one view whose headline
answers the only question most users have ("what will actually load, how big is it, will it fit?"). The
effective list is computed from the already-loaded reference for literal picks and ``all *`` commands;
``top N`` / ``bottom N`` need usage stats, fetched on demand when the user presses Resolve. The two rule
editors and the large-models switch feed the preview, so cause and effect sit on one screen.
"""

from __future__ import annotations

import contextlib

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import Button, Label, Static, Switch

from horde_worker_regen.tui.catalog_cache import CATALOG_CACHE
from horde_worker_regen.tui.formatters import human_bytes
from horde_worker_regen.tui.model_catalog import (
    ModelInfo,
    friendly_baseline,
    has_popularity_meta,
)
from horde_worker_regen.tui.model_resolution import (
    EffectiveModel,
    EffectiveStatus,
    ResolutionResult,
    resolve_effective_models,
)
from horde_worker_regen.tui.widgets.model_list_editor import ModelListEditor

_MAX_PREVIEW_ROWS = 250

_STATUS_GLYPH: dict[EffectiveStatus, tuple[str, str]] = {
    EffectiveStatus.ON_DISK: ("✓", "green"),
    EffectiveStatus.TO_DOWNLOAD: ("⬇", "yellow"),
    EffectiveStatus.SKIPPED: ("✗", "red dim"),
    EffectiveStatus.EXCLUDED_LARGE: ("⚠", "dark_orange"),
    EffectiveStatus.EXCLUDED_OFFLINE: ("⤫", "dark_orange"),
    EffectiveStatus.UNKNOWN: ("⚠", "red"),
}


class ModelManagerView(Vertical):
    """One panel that edits the load/skip rules and previews the resulting effective model set."""

    DEFAULT_CSS = """
    ModelManagerView {
        height: auto;
    }
    ModelManagerView #mm-headline {
        height: auto;
        padding: 0 1;
    }
    ModelManagerView #mm-effective {
        height: auto;
        max-height: 16;
        overflow-y: auto;
        border: round $foreground 20%;
        padding: 0 1;
        margin-bottom: 1;
    }
    ModelManagerView #mm-warnings {
        height: auto;
        padding: 0 1;
    }
    ModelManagerView #mm-resolve-row {
        height: 3;
    }
    ModelManagerView #mm-resolve-row Button {
        margin-right: 1;
    }
    ModelManagerView #mm-resolve-status {
        width: 1fr;
        content-align: left middle;
        height: 3;
    }
    ModelManagerView .mm-rules-label {
        color: $accent;
        text-style: bold;
        padding: 1 1 0 1;
    }
    ModelManagerView #mm-large-row, ModelManagerView #mm-ondisk-row {
        height: 3;
        padding: 0 1;
    }
    ModelManagerView #mm-large-row Label, ModelManagerView #mm-ondisk-row Label {
        width: 1fr;
        content-align: left middle;
        height: 3;
    }
    """

    def __init__(
        self,
        load_values: list[str],
        skip_values: list[str],
        *,
        load_large_models: bool,
        only_on_disk: bool = False,
    ) -> None:
        """Pre-fill the rule editors and the toggles from the loaded config."""
        super().__init__(id="mm-root")
        self._load_values = list(load_values)
        self._skip_values = list(skip_values)
        self._load_large = load_large_models
        self._only_on_disk = only_on_disk
        self._catalog: list[ModelInfo] | None = None
        self._popularity: dict[str, int] | None = None
        self._free_disk_bytes: int | None = None
        self._weights_root: str | None = None
        self._worker_loaded_count: int | None = None
        self._last_result: ResolutionResult | None = None
        self._catalog_poll: Timer | None = None
        """Polls the shared cache until the startup warm finishes, then adopts the catalog and stops."""

    def compose(self) -> ComposeResult:
        """Lay out the preview, the Resolve control, the two rule editors, and the large-models switch."""
        yield Static(id="mm-headline")
        yield VerticalScroll(Static(id="mm-effective-body"), id="mm-effective")
        yield Static(id="mm-warnings")
        with Horizontal(id="mm-resolve-row"):
            yield Button("Resolve ⟳", id="mm-resolve", variant="primary")
            yield Static(id="mm-resolve-status")
        yield Label("LOAD RULES  ·  models and meta commands to offer", classes="mm-rules-label")
        yield ModelListEditor("models_to_load", self._load_values, sibling_values=self._skip_editor_values)
        yield Label("SKIP RULES  ·  only remove from the set above (never add back)", classes="mm-rules-label")
        yield ModelListEditor("models_to_skip", self._skip_values, sibling_values=self._load_editor_values)
        with Horizontal(id="mm-large-row"):
            yield Label("Include large models (Flux, Cascade) in 'all' / 'top' commands")
            yield Switch(value=self._load_large, id="cfg-load_large_models")
        with Horizontal(id="mm-ondisk-row"):
            yield Label("Only models already on disk (never download; drop the rest)")
            yield Switch(value=self._only_on_disk, id="cfg-only_models_on_disk")

    def on_mount(self) -> None:
        """Render the initial hint, then adopt the catalog from the shared cache (warmed at startup)."""
        self._recompute()
        self._adopt_from_cache()
        if self._catalog is None:
            # The startup warm may still be in flight; poll the cache and adopt it the moment it lands.
            self._catalog_poll = self.set_interval(0.5, self._adopt_from_cache)

    def update_worker_models(self, active_models: list[str]) -> None:
        """Note how many models a running worker currently has loaded (None/empty clears the note)."""
        count = len(active_models) if active_models else None
        if count == self._worker_loaded_count:
            return
        self._worker_loaded_count = count
        if self._last_result is not None:
            self._render_headline(self._last_result)

    def on_model_list_editor_changed(self, message: ModelListEditor.Changed) -> None:
        """Recompute the preview when either rule list changes."""
        self._sync_values()
        self._recompute()
        if self._catalog is None:
            # The picker (or the startup warm) may have loaded the reference since mount; adopt it.
            self._adopt_from_cache()

    def on_switch_changed(self, message: Switch.Changed) -> None:
        """Recompute when the large-models or only-on-disk switch toggles."""
        if message.switch.id == "cfg-load_large_models":
            self._load_large = message.value
            self._recompute()
        elif message.switch.id == "cfg-only_models_on_disk":
            self._only_on_disk = message.value
            self._recompute()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Trigger an on-demand resolve (loads the reference and, if needed, usage stats)."""
        if event.button.id != "mm-resolve":
            return
        event.stop()
        self._sync_values()
        self.query_one("#mm-resolve-status", Static).update(Text("Resolving…", style="grey62"))
        self.run_worker(self._resolve, thread=True, exclusive=True, group="mm-catalog")

    def _load_editor_values(self) -> list[str]:
        """The models-to-load editor's current entries (for the skip picker's membership view)."""
        with contextlib.suppress(Exception):  # the editor may not be mounted yet
            return self.query_one("#mle-root-models_to_load", ModelListEditor).values()
        return []

    def _skip_editor_values(self) -> list[str]:
        """The models-to-skip editor's current entries (for the load picker's membership view)."""
        with contextlib.suppress(Exception):  # the editor may not be mounted yet
            return self.query_one("#mle-root-models_to_skip", ModelListEditor).values()
        return []

    def _sync_values(self) -> None:
        """Copy the live editor/switch values into local state (so background work sees them)."""
        # Widgets may not be mounted yet during early compose; a partial sync is fine.
        with contextlib.suppress(Exception):
            self._load_values = self.query_one("#mle-root-models_to_load", ModelListEditor).values()
            self._skip_values = self.query_one("#mle-root-models_to_skip", ModelListEditor).values()
            self._load_large = self.query_one("#cfg-load_large_models", Switch).value

    def _adopt_from_cache(self) -> None:
        """Adopt the catalog from the shared cache if it is loaded; stop polling once it is (UI thread)."""
        snapshot = CATALOG_CACHE.snapshot()
        if not snapshot.is_loaded:
            return
        if self._catalog_poll is not None:
            self._catalog_poll.stop()
            self._catalog_poll = None
        self._adopt_catalog(snapshot.catalog, snapshot.free_disk_bytes, snapshot.popularity, snapshot.weights_root)

    def _resolve(self) -> None:
        """Load the reference from the cache (fetching if needed) and usage stats for top/bottom rules."""
        want_popularity = has_popularity_meta(self._load_values + self._skip_values) or not self._load_values
        try:
            snapshot = CATALOG_CACHE.ensure_loaded(want_popularity=want_popularity)
        except Exception as error:  # noqa: BLE001 - surface any loader failure to the user
            # A stats failure should not lose an otherwise-good catalog load; retry catalog-only.
            try:
                snapshot = CATALOG_CACHE.ensure_loaded()
            except Exception:  # noqa: BLE001 - now the reference itself could not be loaded
                self.app.call_from_thread(self._on_resolve_error, f"{type(error).__name__}: {error}")
                return
            self.app.call_from_thread(
                self._adopt_catalog,
                snapshot.catalog,
                snapshot.free_disk_bytes,
                snapshot.popularity,
                snapshot.weights_root,
            )
            self.app.call_from_thread(
                self._set_resolve_status,
                Text(f"Loaded reference; usage stats unavailable ({error}).", style="yellow"),
            )
            return
        self.app.call_from_thread(
            self._adopt_catalog,
            snapshot.catalog,
            snapshot.free_disk_bytes,
            snapshot.popularity,
            snapshot.weights_root,
        )
        self.app.call_from_thread(self._set_resolve_status, Text("Resolved.", style="green"))

    def _adopt_catalog(
        self,
        catalog: list[ModelInfo] | None,
        free_disk_bytes: int | None,
        popularity: dict[str, int] | None,
        weights_root: str | None = None,
    ) -> None:
        """Store the loaded catalog/stats on the UI thread and refresh the preview."""
        self._catalog = catalog
        self._free_disk_bytes = free_disk_bytes
        self._weights_root = weights_root
        if popularity is not None:
            self._popularity = popularity
        self._recompute()

    def _on_resolve_error(self, message: str) -> None:
        """Show a clear error when the reference cannot be loaded."""
        self._set_resolve_status(
            Text(f"Could not load the model reference ({message}). Run the worker once, then retry.", style="red"),
        )

    def _set_resolve_status(self, text: Text) -> None:
        """Update the Resolve status line."""
        with contextlib.suppress(Exception):  # the widget may have been torn down
            self.query_one("#mm-resolve-status", Static).update(text)

    def _recompute(self) -> None:
        """Resolve the current rules into an effective set and render the preview."""
        result = resolve_effective_models(
            self._load_values,
            self._skip_values,
            self._catalog,
            load_large_models=self._load_large,
            only_on_disk=self._only_on_disk,
            popularity=self._popularity,
        )
        self._last_result = result
        self._render_headline(result)
        self._render_body(result)
        self._render_warnings(result)

    def configured_included_models(self) -> list[EffectiveModel]:
        """The currently-resolved configured models (the set the worker would load), for the download picker.

        Empty when the catalog has not resolved yet (the reference loads in the background via
        ``_adopt_from_cache``); the caller then nudges the operator to Resolve in this tab first.
        """
        self._sync_values()
        if self._last_result is None:
            return []
        return self._last_result.included

    def _disk_totals(self, included: list[EffectiveModel]) -> tuple[int, int, int, bool, int]:
        """Sum present/to-download bytes from the included rows (sizes are baked into the catalog)."""
        present = sum(model.size_bytes or 0 for model in included if model.on_disk)
        to_download = sum(model.size_bytes or 0 for model in included if not model.on_disk)
        fits = self._free_disk_bytes is None or to_download <= self._free_disk_bytes
        shortfall = 0 if fits or self._free_disk_bytes is None else to_download - self._free_disk_bytes
        return present, to_download, present + to_download, fits, shortfall

    def _render_headline(self, result: ResolutionResult) -> None:
        """Render the one-line answer: count, disk budget, and fit verdict."""
        headline = self.query_one("#mm-headline", Static)
        if not result.catalog_loaded:
            hint = Text("EFFECTIVE MODELS  ", style="bold")
            hint.append("press Resolve to load the reference and compute what will load.", style="grey62")
            headline.update(hint)
            return

        included = result.included
        present, to_download, total, fits, shortfall = self._disk_totals(included)
        text = Text("EFFECTIVE  ", style="bold")
        text.append(f"{len(included)} will load", style="bold cyan")
        text.append("  ·  ", style="grey50")
        text.append(f"{human_bytes(total)} total ", style="grey70")
        text.append(f"({human_bytes(present)} on disk + {human_bytes(to_download)} to download)", style="grey62")
        text.append("  ·  ", style="grey50")
        text.append(f"{human_bytes(self._free_disk_bytes)} free", style="grey70")
        text.append("  ·  ", style="grey50")
        verdict = "✓ fits" if fits else f"✗ OVER BUDGET by {human_bytes(shortfall)}"
        text.append(verdict, style="green" if fits else "bold red")
        unsized = sum(1 for model in included if model.size_bytes is None)
        if unsized:
            text.append(f"  ·  {unsized} unsized", style="yellow")
        if self._weights_root:
            text.append(f"\nmodels dir: {self._weights_root}", style="grey50")
        if self._worker_loaded_count is not None:
            text.append(f"\nWorker currently has {self._worker_loaded_count} model(s) loaded.", style="grey50")
        headline.update(text)

    def _render_body(self, result: ResolutionResult) -> None:
        """Render the per-model rows (or a guiding message when nothing is resolved yet)."""
        body = self.query_one("#mm-effective-body", Static)
        if not result.catalog_loaded:
            lines = Text()
            if result.default_applied:
                lines.append("No load rules set — the worker would default to 'top 2'.\n", style="grey62")
            lines.append("The effective list appears here once the reference is loaded.", style="grey50")
            body.update(lines)
            return
        if not result.rows:
            body.update(Text("Nothing resolved. Add a model or a meta command below.", style="grey50"))
            return

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="center", width=1)
        table.add_column(no_wrap=True)
        table.add_column(no_wrap=True, style="grey62")
        table.add_column(no_wrap=True)
        table.add_column(style="grey50")
        for model in result.rows[:_MAX_PREVIEW_ROWS]:
            glyph, colour = _STATUS_GLYPH[model.status]
            included = model.status in (EffectiveStatus.ON_DISK, EffectiveStatus.TO_DOWNLOAD)
            table.add_row(
                Text(glyph, style=colour),
                Text(model.name, style="white" if included else "grey58"),
                friendly_baseline(model.baseline) or "-",
                self._disk_cell(model),
                model.reason,
            )
        renderables: list[RenderableType] = [table]
        if len(result.rows) > _MAX_PREVIEW_ROWS:
            renderables.append(Text(f"…and {len(result.rows) - _MAX_PREVIEW_ROWS} more", style="grey50"))
        body.update(Group(*renderables))

    @staticmethod
    def _disk_cell(model: EffectiveModel) -> Text:
        """A compact on-disk badge for a row."""
        if model.status is EffectiveStatus.ON_DISK:
            return Text("on disk", style="green")
        if model.status is EffectiveStatus.TO_DOWNLOAD:
            if model.size_bytes:
                return Text(f"{human_bytes(model.size_bytes)} ↓", style="yellow")
            return Text("download", style="yellow")
        return Text("-", style="grey50")

    def _render_warnings(self, result: ResolutionResult) -> None:
        """Render the warnings and any unexpanded top/bottom commands beneath the list."""
        warnings = self.query_one("#mm-warnings", Static)
        text = Text()
        for entry in result.needs_resolve:
            text.append(f"⟳ '{entry}' not expanded yet — press Resolve.\n", style="dark_orange")
        for warning in result.warnings:
            text.append(f"⚠ {warning}\n", style="yellow")
        warnings.update(text)
