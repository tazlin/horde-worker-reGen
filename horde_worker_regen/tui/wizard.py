"""The Getting started page: the single surface that explains and performs worker setup.

Nothing else in the dashboard tells a newcomer what a worker needs or why, so this page carries the
whole story: what the AI Horde is, which three things a worker cannot run without, and why models have
to be downloaded before requests can be served. It then collects those three things inline, so the
Config tab is somewhere to refine a working worker rather than somewhere setup happens.

The page is shown automatically while ``bridgeData.yaml`` is unconfigured (see
[`is_setup_incomplete`][horde_worker_regen.tui.wizard.is_setup_incomplete]) and stays reachable from the
Simple home afterwards, because the presets and the explanations remain useful once a worker runs.

Opening it costs nothing: construction and mount only paint, and every slow answer (the device probe,
the model catalog, the disk pricing) is gathered on a worker thread and applied to the live page as it
arrives, so the options fill in under the reader rather than delaying the first frame.

What it offers is one pick-one choice between three presets, each a model selection plus a feature
stance, priced against the disk the models will actually need
([`compute_download_plan`][horde_worker_regen.model_download_plan.compute_download_plan] over the models
the selection resolves to). A preset the volume cannot hold is shown disabled with the shortfall rather
than hidden, so the constraint is legible. Model browsing is the same
[`ModelPickerModal`][horde_worker_regen.tui.widgets.model_picker.ModelPickerModal] the Config tab uses.

Saving writes only the keys this page owns through
[`save_config`][horde_worker_regen.tui.config_form.save_config], so every other setting in an existing
config survives untouched. Everything that needs the network (key validation, name-taken checks, the
model catalog and its usage stats) degrades to a quieter page rather than an error: without a catalog
the presets state that sizes are unavailable and remain choosable where they can be honoured.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import enum
import os
from collections.abc import Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Button, Input, Static, Switch

from horde_worker_regen.tui.config_form import DEFAULT_CONFIG_PATH, load_config, save_config
from horde_worker_regen.tui.formatters import human_bytes
from horde_worker_regen.tui.horde_validation import AdvisoryStatus, check_worker_name_available, verify_api_key
from horde_worker_regen.tui.model_catalog import MetaKind, build_meta_instruction, is_meta_instruction
from horde_worker_regen.tui.model_resolution import resolve_effective_models
from horde_worker_regen.tui.widgets.model_picker import ModelPickerModal, ModelPickerResult

if TYPE_CHECKING:
    from collections.abc import Mapping

    from horde_model_reference.model_reference_records import GenericModelRecord

    from horde_worker_regen.model_download_plan import DownloadPlan
    from horde_worker_regen.tui.model_catalog import ModelInfo

DEFAULT_API_KEY = "0000000000"
"""The placeholder key shipped in bridgeData_template.yaml; treated as "not set"."""

DEFAULT_DREAMER_NAME = "An Awesome Dreamer"
"""The placeholder worker name shipped in the template; the horde rejects duplicates, so it must change."""

REGISTER_URL = "https://aihorde.net/register"

_SHOWCASE_SDXL_VRAM_MB = 20_000
"""Above this much VRAM the showcase preset offers a second SDXL model rather than one."""


def _config_str(data: Any, key: str) -> str:  # noqa: ANN401 - ruamel CommentedMap / dict
    """Read a string value from loaded YAML data, returning an empty string when absent."""
    try:
        value = data.get(key)
    except AttributeError:
        return ""
    return "" if value is None else str(value).strip()


def is_setup_incomplete(config_path: Path = DEFAULT_CONFIG_PATH) -> bool:
    """Return whether bridgeData lacks a real API key or still uses the default worker name.

    This is the trigger for showing the page automatically. A missing or unreadable file counts as
    incomplete, and so do the template placeholders, so a freshly seeded config (the installer copies
    the template) leads straight into setup.
    """
    try:
        data = load_config(config_path)
    except Exception:  # noqa: BLE001 - an unreadable config is, for our purposes, an unconfigured one
        return True
    api_key = _config_str(data, "api_key")
    dreamer_name = _config_str(data, "dreamer_name")
    if not api_key or api_key == DEFAULT_API_KEY:
        return True
    return not dreamer_name or dreamer_name == DEFAULT_DREAMER_NAME


_probed_vram_mb: tuple[int | None] | None = None
"""The probe's answer once it has been asked, wrapped so that a ``None`` reading is still "asked"."""


def _detect_total_vram_mb() -> int | None:
    """Best-effort read of the primary device's total VRAM in MB, or None when unavailable.

    Routes through the out-of-process accelerator probe (backend-agnostic across CUDA/ROCm/XPU/MPS/CPU,
    not NVML) so non-NVIDIA cards size the presets correctly. The probe runs the torch-loading
    enumeration in a subprocess, keeping the TUI process itself torch-free. Any failure or a zero
    reading yields None so the caller falls back to a safe default rather than erroring.

    Spawning that subprocess costs seconds, and the installed VRAM does not change while the dashboard
    runs, so the answer is kept for the life of the process. Callers must still treat this as blocking:
    the first ask pays the full cost. Under the test harness the probe is skipped entirely, matching how
    the other out-of-process work here behaves.
    """
    global _probed_vram_mb
    if _testing_mode():
        return None
    if _probed_vram_mb is not None:
        return _probed_vram_mb[0]
    total_mb: int | None = None
    try:
        from horde_worker_regen.utils.accelerator_probe import probe_accelerators

        accelerators = probe_accelerators()
    except Exception:  # noqa: BLE001 - "no GPU telemetry" is expected, not a crash
        accelerators = []
    if accelerators:
        total_mb = accelerators[0].total_vram_mb or None
    _probed_vram_mb = (total_mb,)
    return total_mb


def _top_n_for_vram(total_mb: int | None) -> int:
    """The default ``top N`` size for a card with *total_mb* of VRAM (None when unknown).

    Bigger cards can comfortably hold more resident models, so they get a larger N; without telemetry we
    cannot tell, so we pick a conservative middle ground the user can change by picking their own models.
    """
    if total_mb is None:
        return 3
    if total_mb >= 20_000:
        return 5
    if total_mb >= 10_000:
        return 3
    return 1


def suggested_default_models() -> list[str]:
    """A VRAM-aware initial model selection, defaulting to a safe ``top N`` popularity meta."""
    return [build_meta_instruction(MetaKind.TOP_N, _top_n_for_vram(_detect_total_vram_mb()))]


def _detect_installed_torch_build() -> str | None:
    """The local build tag of the installed torch wheel (e.g. ``cu128``, ``cpu``, ``rocm6.4``).

    Read from package metadata so we never import torch itself, which is slow and would pull a CUDA
    context into the lightweight TUI process. Returns None when torch or its version is not findable.
    """
    try:
        from importlib.metadata import version

        raw = version("torch")
    except Exception:  # noqa: BLE001 - "cannot tell which build" is expected, not a crash
        return None
    _, _, local = raw.partition("+")
    return local or None


def _backend_mismatch_warning(total_vram_mb: int | None) -> str:
    """Text warning of an installed CPU build despite a detected NVIDIA GPU, else ``""``.

    The device probe talks to the driver, not torch, so a present GPU still reports its VRAM even when
    the CPU wheel is installed; pairing that with a ``+cpu`` build is the classic "why is it so slow"
    trap. When no device telemetry is available there is nothing to compare against.
    """
    if total_vram_mb is None:
        return ""
    build = _detect_installed_torch_build()
    if build is not None and build.startswith("cpu"):
        return (
            "An NVIDIA GPU was detected, but the CPU build of PyTorch is installed, so the worker "
            "would run roughly 100x slower. Re-run the installer with HORDE_WORKER_BACKEND=cu128 "
            "to fix this."
        )
    return ""


def _testing_mode() -> bool:
    """Whether we are running under the test harness, where network-touching work is skipped."""
    return bool(os.environ.get("AI_HORDE_TESTING"))


class PresetKind(enum.StrEnum):
    """The three offered starting points, from smallest footprint to largest."""

    ESSENTIALS = "essentials"
    RECOMMENDED = "recommended"
    SHOWCASE = "showcase"


@dataclasses.dataclass(frozen=True)
class Preset:
    """One offered starting point: what it serves, and the feature stance that goes with it."""

    kind: PresetKind
    title: str
    offer: str
    """One sentence on what choosing this gives the people asking for images."""
    stance: str
    """Plain wording for the kinds of work this preset accepts."""
    features: dict[str, bool]
    """bridgeData feature keys this preset writes."""


PRESETS: tuple[Preset, ...] = (
    Preset(
        kind=PresetKind.ESSENTIALS,
        title="Essentials",
        offer="Serves the single most commonly requested model, which is the fastest way to be useful.",
        stance="Accepts plain image requests and the finishing touches (upscaling, face fixing).",
        features={"allow_post_processing": True, "allow_lora": False, "allow_controlnet": False},
    ),
    Preset(
        kind=PresetKind.RECOMMENDED,
        title="Recommended",
        offer="Serves the most-requested models your graphics card can hold, so few requests pass you by.",
        stance=(
            "Accepts plain image requests, finishing touches, and LoRA styles. LoRA files download on "
            "demand and build up their own cache over time."
        ),
        features={"allow_post_processing": True, "allow_lora": True, "allow_controlnet": False},
    ),
    Preset(
        kind=PresetKind.SHOWCASE,
        title="Showcase",
        offer="Adds the larger SDXL models and the guided (ControlNet) requests that fewer workers accept.",
        stance="Accepts everything the recommended preset does, plus ControlNet work. The largest download.",
        features={"allow_post_processing": True, "allow_lora": True, "allow_controlnet": True},
    ),
)

PRESETS_BY_KIND: dict[PresetKind, Preset] = {preset.kind: preset for preset in PRESETS}

SUGGESTED_PRESET = PresetKind.RECOMMENDED
"""The preset offered as the default choice."""

FEATURE_KEYS: tuple[str, ...] = ("allow_post_processing", "allow_lora", "allow_controlnet")
"""Every feature key a preset can write, so an unselected stance is written as False rather than left."""


@dataclasses.dataclass(frozen=True)
class PresetPlan:
    """A preset resolved against the catalog and the disk it would need."""

    kind: PresetKind
    entries: list[str]
    """What would be written to ``models_to_load`` (literal names and/or meta commands)."""
    model_names: list[str]
    """The concrete models the entries resolve to, empty when they could not be resolved."""
    resolved: bool
    """Whether the entries could be turned into a concrete model list at all."""
    plan: DownloadPlan | None
    """The disk picture for the resolved models, or None when sizes are unavailable."""

    @property
    def selectable(self) -> bool:
        """Whether this preset can be chosen: its entries are known and nothing proves it will not fit.

        A preset whose preview could not be resolved is still choosable as long as its entries are
        known: the meta-command presets are expanded by the worker itself at startup, so a missing
        catalog only costs the preview, never the choice. Only a preset whose entries are unknown, or
        whose computed download does not fit the volume, is withheld.
        """
        if not self.entries:
            return False
        return self.plan is None or self.plan.fits

    def disk_sentence(self) -> str:
        """A one-line statement of what this preset costs on disk, honest about what is unknown."""
        if not self.resolved:
            return "The model list for this preset is not available yet."
        if self.plan is None:
            return "Download sizes are unavailable until the model list loads."
        free = "an unknown amount of" if self.plan.free_disk_bytes is None else human_bytes(self.plan.free_disk_bytes)
        qualifier = "at least " if not self.plan.sizes_complete else "about "
        sentence = (
            f"Needs {qualifier}{human_bytes(self.plan.to_download_bytes)} downloaded, {free} free where models go."
        )
        if not self.plan.fits:
            sentence += f" That is {human_bytes(self.plan.shortfall_bytes)} more than this computer has free."
        return sentence


def _catalog_state() -> tuple[list[ModelInfo] | None, dict[str, int] | None]:
    """Return the cached model catalog and usage stats without ever touching the network."""
    from horde_worker_regen.tui.catalog_cache import CATALOG_CACHE

    snapshot = CATALOG_CACHE.snapshot()
    return snapshot.catalog, snapshot.popularity


def _catalog_records() -> Mapping[str, GenericModelRecord] | None:
    """Return the loaded model reference records, or None when nothing is loaded yet."""
    from horde_worker_regen.tui.model_catalog import cached_image_records

    try:
        return cached_image_records()
    except Exception:  # noqa: BLE001 - sizing is enrichment; a page without it is still usable
        return None


def _download_plan(model_names: list[str], records: Mapping[str, GenericModelRecord]) -> DownloadPlan:
    """Price a concrete model list against the model volume."""
    from horde_worker_regen.model_download_plan import compute_download_plan

    return compute_download_plan(model_names, records)


def _most_used_sdxl(catalog: list[ModelInfo], popularity: dict[str, int], count: int) -> list[str]:
    """The ``count`` most-requested SDXL models in the catalog, in popularity order."""
    sdxl = {model.name for model in catalog if model.baseline == "stable_diffusion_xl"}
    ranked = sorted(popularity.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _uses in ranked if name in sdxl][:count]


def preset_entries(
    kind: PresetKind,
    catalog: list[ModelInfo] | None,
    popularity: dict[str, int] | None,
    total_vram_mb: int | None,
) -> list[str] | None:
    """The ``models_to_load`` entries a preset would write, or None when they cannot be determined.

    Essentials and Recommended are popularity meta commands, the same instruction the worker expands at
    startup, so they stay correct as the horde's model usage moves. Showcase names its SDXL models
    literally because no meta command expresses "the most-requested SDXL models"; that literal choice
    needs the catalog and the usage stats, so it is the one preset that can be undeterminable.
    """
    top_n = _top_n_for_vram(total_vram_mb)
    if kind is PresetKind.ESSENTIALS:
        return [build_meta_instruction(MetaKind.TOP_N, 1)]
    if kind is PresetKind.RECOMMENDED:
        return [build_meta_instruction(MetaKind.TOP_N, top_n)]
    if catalog is None or popularity is None:
        return None
    wanted = 2 if (total_vram_mb or 0) >= _SHOWCASE_SDXL_VRAM_MB else 1
    sdxl = _most_used_sdxl(catalog, popularity, wanted)
    if not sdxl:
        return None
    return [build_meta_instruction(MetaKind.TOP_N, top_n), *sdxl]


def build_preset_plans(
    catalog: list[ModelInfo] | None,
    popularity: dict[str, int] | None,
    records: Mapping[str, GenericModelRecord] | None,
    *,
    total_vram_mb: int | None,
) -> dict[PresetKind, PresetPlan]:
    """Resolve every preset and price it, degrading to "unknown" rather than failing.

    Args:
        catalog: The image-model catalog, or None when it has not loaded.
        popularity: Model-name to last-month usage count, or None when the stats have not loaded.
        records: The model reference records used for sizing, or None when unavailable.
        total_vram_mb: The detected VRAM, which sizes the recommended and showcase selections.

    Returns:
        A plan per preset. A plan with ``resolved`` False could not name its models; one with a None
        ``plan`` named them but could not price them.
    """
    plans: dict[PresetKind, PresetPlan] = {}
    for preset in PRESETS:
        entries = preset_entries(preset.kind, catalog, popularity, total_vram_mb)
        if entries is None:
            plans[preset.kind] = PresetPlan(preset.kind, [], [], resolved=False, plan=None)
            continue
        names = resolve_preset_models(entries, catalog, popularity)
        if names is None:
            plans[preset.kind] = PresetPlan(preset.kind, entries, [], resolved=False, plan=None)
            continue
        download_plan = None
        if records is not None:
            try:
                download_plan = _download_plan(names, records)
            except Exception:  # noqa: BLE001 - an unpriceable preset is still a choosable one
                download_plan = None
        plans[preset.kind] = PresetPlan(preset.kind, entries, names, resolved=True, plan=download_plan)
    return plans


def resolve_preset_models(
    entries: list[str],
    catalog: list[ModelInfo] | None,
    popularity: dict[str, int] | None,
) -> list[str] | None:
    """Expand preset entries to the concrete models the worker would load, or None when it cannot.

    Uses the same expansion the Config tab shows, so the page's model list and its disk figure describe
    what the worker will really do with the entries being written.
    """
    result = resolve_effective_models(entries, [], catalog, load_large_models=False, popularity=popularity)
    if not result.catalog_loaded or result.needs_resolve:
        return None
    return [row.name for row in result.included]


_WHAT_IS_THE_HORDE = (
    "The AI Horde is a crowdsourced image service. People donate time on their graphics cards, anyone "
    "can ask for images without paying, and the contributors earn kudos for the work they do. This "
    "computer becomes one of those contributors."
)

_WHAT_IS_NEEDED = (
    "Three things get a worker running:\n\n"
    "  A worker name. This is how the horde tells your computer apart from everyone else's, and it is "
    "shown publicly, so it has to be one nobody else has taken.\n\n"
    "  An API key. This is what makes the kudos you earn land on your account. Anonymous contribution "
    "works, but nothing you earn is kept. A key is free from " + REGISTER_URL + " (type that address "
    "into a browser).\n\n"
    "  Models on disk. A request can only be served by a computer that already holds the model it asks "
    "for, so the models you offer are downloaded first. That is what the sizes below are: real files, "
    "on your drive, before the first request arrives."
)


class GettingStartedScreen(Screen[bool]):
    """The full-page setup surface; dismisses True once bridgeData has been written."""

    BINDINGS = [Binding("escape", "close", "Close")]

    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (110, "-normal"), (150, "-wide")]
    """Three readable option columns need about 110 columns; below that they stack."""

    DEFAULT_CSS = """
    GettingStartedScreen {
        background: $surface;
    }
    GettingStartedScreen #gs-page {
        padding: 1 2;
    }
    /* Containers default to filling their parent, which on a scrolling page squeezes the cards inside
       them to nothing and pads the details card with dead rows. Every container here sizes to content. */
    GettingStartedScreen #gs-presets,
    GettingStartedScreen #gs-preset-row,
    GettingStartedScreen #gs-civitai,
    GettingStartedScreen .gs-preset {
        height: auto;
    }
    /* One frame around the whole choice, so the three options read as one pick-one control rather than
       three unrelated ideas. The options themselves carry no border, only a selection background. */
    GettingStartedScreen #gs-presets {
        border: round $primary;
        padding: 1 1;
        margin-bottom: 1;
    }
    GettingStartedScreen #gs-preset-row {
        layout: vertical;
    }
    GettingStartedScreen.-normal #gs-preset-row,
    GettingStartedScreen.-wide #gs-preset-row {
        layout: horizontal;
    }
    GettingStartedScreen .gs-preset {
        width: 1fr;
        padding: 0 1;
    }
    GettingStartedScreen .gs-preset.-chosen {
        background: $accent 20%;
    }
    GettingStartedScreen .gs-preset Button {
        width: 100%;
    }
    GettingStartedScreen #gs-stance {
        padding-top: 1;
        color: $text-muted;
    }
    GettingStartedScreen .gs-error {
        color: $error;
    }
    GettingStartedScreen .gs-field-row {
        height: auto;
    }
    GettingStartedScreen .gs-field-row Static {
        width: 1fr;
        height: 3;
        padding-left: 2;
        content-align: left middle;
    }
    GettingStartedScreen Input {
        margin-bottom: 1;
    }
    """

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        """Store the config path only.

        Construction happens inside the caller's click handler, before the screen is even pushed, so
        nothing here may touch the device probe, the disk or the catalog: all of that runs later, off
        the UI thread, and lands on an already-painted page.
        """
        super().__init__()
        self._config_path = config_path
        self._detected_vram_mb: int | None = None
        self._chosen = SUGGESTED_PRESET
        self._plans: dict[PresetKind, PresetPlan] = {}
        self._custom_models: list[str] | None = None
        """A hand-picked model list that overrides the chosen preset's entries, when the user made one."""

    def compose(self) -> ComposeResult:
        """Lay out the explanation, the presets, the fields that must be filled, and the save action."""
        with VerticalScroll(id="gs-page"):
            yield Static("Getting started", classes="horde-hero horde-card-title")
            yield Static(_WHAT_IS_THE_HORDE, id="gs-what-is", classes="horde-card")
            yield Static(_WHAT_IS_NEEDED, id="gs-what-needed", classes="horde-card")
            yield Static("", id="gs-backend-warning", classes="gs-error")
            yield Static("What will this computer offer?", classes="horde-card-title")
            with Vertical(id="gs-presets"):
                with Horizontal(id="gs-preset-row"):
                    for preset in PRESETS:
                        with Vertical(id=f"gs-preset-{preset.kind.value}", classes="gs-preset"):
                            yield Static(id=f"gs-preset-{preset.kind.value}-body")
                            yield Button("Choose this", id=f"gs-choose-{preset.kind.value}")
                yield Static(id="gs-stance")
            yield Static(id="gs-selection", classes="horde-card")
            with Horizontal(classes="horde-actions"):
                yield Button("Choose my own models instead", id="gs-browse")
            yield Static("Your details", classes="horde-card-title")
            with Vertical(classes="horde-card"):
                yield Static("Worker name (public, and unique on the horde)")
                yield Input(placeholder="Worker name", id="gs-name")
                yield Static("", id="gs-name-error", classes="gs-error")
                yield Static(f"API key (free from {REGISTER_URL}; without one, kudos are not kept)")
                yield Input(placeholder="API key", password=True, id="gs-api-key")
                yield Static("", id="gs-api-key-error", classes="gs-error")
                with Vertical(id="gs-civitai"):
                    yield Static(
                        "Civitai token (optional). LoRA files download from Civitai as jobs ask for "
                        "them, and some refuse anonymous downloads; a token from a free Civitai "
                        "account gets those.",
                    )
                    yield Input(placeholder="Civitai API token", password=True, id="gs-civitai-token")
                with Horizontal(classes="gs-field-row"):
                    yield Switch(id="gs-nsfw")
                    yield Static(
                        "Accept adult requests. With this off, this computer is only asked for safe-for-work images.",
                    )
            with Horizontal(classes="horde-actions"):
                yield Button("Save and close", id="gs-save", variant="success")
                yield Button("Close without saving", id="gs-close")
            yield Static("", id="gs-save-error", classes="gs-error")

    def on_mount(self) -> None:
        """Initialise once the composed children are fully mounted.

        The screen's own mount fires while its nested compose results can still be mounting, so a query
        from here can miss a widget inside a nested container. Deferring one refresh removes the race.
        """
        self.call_after_refresh(self._initialise)

    def _initialise(self) -> None:
        """Paint the page from what is already known, then gather the rest in the background."""
        self._seed_from_config()
        self.query_one("#gs-presets", Vertical).border_title = "Pick one"
        self._render_presets()
        self._start_background_load()
        # The page opens on its explanation, which is the reason it exists. Focusing a field would
        # scroll the reader past it, so the first field is reached by Tab or by clicking it.
        self.query_one("#gs-page", VerticalScroll).scroll_home(animate=False)

    def _seed_from_config(self) -> None:
        """Pre-fill the fields with what bridgeData already holds, ignoring the template placeholders."""
        try:
            data = load_config(self._config_path)
        except Exception:  # noqa: BLE001 - an unreadable config just means nothing to pre-fill
            return
        name = _config_str(data, "dreamer_name")
        if name and name != DEFAULT_DREAMER_NAME:
            self.query_one("#gs-name", Input).value = name
        api_key = _config_str(data, "api_key")
        if api_key and api_key != DEFAULT_API_KEY:
            self.query_one("#gs-api-key", Input).value = api_key
        with_nsfw = data.get("nsfw") if hasattr(data, "get") else None
        self.query_one("#gs-nsfw", Switch).value = bool(with_nsfw)
        civitai_token = _config_str(data, "civitai_api_token")
        if civitai_token:
            self.query_one("#gs-civitai-token", Input).value = civitai_token

    # region presets

    def _start_background_load(self) -> None:
        """Gather everything the presets need off the UI thread, on an already-painted page.

        One worker covers the whole open path: the device probe (a subprocess that takes seconds), the
        catalog and usage-stat fetch, and the planning that prices each preset against the disk. Results
        arrive in two passes so the page fills in as soon as each is known rather than at the end.
        """
        self.run_worker(self._load_page_data_blocking, thread=True, exclusive=True, group="getting-started-load")

    def _load_page_data_blocking(self) -> None:
        """Probe, load and plan for the worker thread; every step degrades to "unknown" on failure."""
        total_vram_mb = _detect_total_vram_mb()
        warning = _backend_mismatch_warning(total_vram_mb)
        self._publish_page_data(total_vram_mb, warning)
        if _testing_mode():
            return
        from horde_worker_regen.tui.catalog_cache import CATALOG_CACHE

        try:
            CATALOG_CACHE.ensure_loaded(want_popularity=True)
        except Exception:  # noqa: BLE001 - offline is a quieter page, never an error state
            return
        self._publish_page_data(total_vram_mb, warning)

    def _publish_page_data(self, total_vram_mb: int | None, warning: str) -> None:
        """Plan against whatever is cached now and hand the result to the UI thread (worker thread)."""
        catalog, popularity = _catalog_state()
        plans = build_preset_plans(catalog, popularity, _catalog_records(), total_vram_mb=total_vram_mb)
        self.app.call_from_thread(self._apply_page_data, total_vram_mb, warning, plans)

    def _apply_page_data(
        self,
        total_vram_mb: int | None,
        warning: str,
        plans: dict[PresetKind, PresetPlan],
    ) -> None:
        """Adopt background results and redraw. A late arrival on a closed page is dropped.

        The card the operator picked is never taken away by data landing afterwards: only the entries
        and the pricing behind each option are replaced, so a bigger card arriving late can widen the
        recommended selection without moving the choice.
        """
        if not self.is_mounted:
            return
        self._detected_vram_mb = total_vram_mb
        self._plans = plans
        with contextlib.suppress(NoMatches):
            self.query_one("#gs-backend-warning", Static).update(warning)
            self._render_presets()

    def _render_presets(self) -> None:
        """Draw the options from what is currently known, including "still checking"."""
        for preset in PRESETS:
            plan = self._plans.get(preset.kind)
            self.query_one(f"#gs-preset-{preset.kind.value}-body", Static).update(self._preset_text(preset, plan))
            card = self.query_one(f"#gs-preset-{preset.kind.value}", Vertical)
            card.set_class(preset.kind is self._chosen, "-chosen")
            button = self.query_one(f"#gs-choose-{preset.kind.value}", Button)
            # An option is only refused once it is priced and known not to fit; while the figures are
            # still being gathered every option stays live.
            button.disabled = plan is not None and not plan.selectable
            button.label = "Chosen" if preset.kind is self._chosen else "Choose this"
        self.query_one("#gs-stance", Static).update(
            Text.assemble(("This choice accepts: ", "grey62"), (PRESETS_BY_KIND[self._chosen].stance, "")),
        )
        self._update_selection_text()
        self._update_civitai_visibility()

    def _update_civitai_visibility(self) -> None:
        """Show the Civitai token only while the chosen stance accepts LoRA work, which is what needs it."""
        wants_lora = PRESETS_BY_KIND[self._chosen].features.get("allow_lora", False)
        self.query_one("#gs-civitai", Vertical).display = wants_lora

    def _preset_text(self, preset: Preset, plan: PresetPlan | None) -> Text:
        """One option: its marker and title, what it offers, and what it costs (or that we are looking).

        The radio marker carries the mutual exclusivity, so the three options read as one choice before
        anything is clicked. ``plan`` is None while the figures are still being gathered.
        """
        chosen = preset.kind is self._chosen
        heading = f"{'◉' if chosen else '○'} {preset.title}"
        if preset.kind is SUGGESTED_PRESET:
            heading += "  (suggested)"
        title_style = "bold" if chosen else "bold grey70"
        body = Text.assemble((f"{heading}\n", title_style), (f"{preset.offer}\n", ""))
        if plan is None:
            body.append("Checking what this needs on disk…", "grey62")
            return body
        body.append(plan.disk_sentence(), "grey70" if plan.selectable else "yellow")
        return body

    def _update_selection_text(self) -> None:
        """List the models the current choice will load, so the download is never a surprise."""
        static = self.query_one("#gs-selection", Static)
        if self._custom_models is not None:
            names = ", ".join(self._custom_models) or "(none)"
            static.update(Text.assemble(("Your own selection: ", "grey62"), (names, "bold")))
            return
        plan = self._plans.get(self._chosen)
        title = PRESETS_BY_KIND[self._chosen].title
        if plan is None:
            static.update(Text.assemble((f"{title}: ", "grey62"), ("working out what it will load…", "grey70")))
            return
        if not plan.model_names:
            detail = "the model list loads when the catalog is available"
            static.update(Text.assemble((f"{title}: ", "grey62"), (detail, "grey70")))
            return
        static.update(Text.assemble((f"{title} loads: ", "grey62"), (", ".join(plan.model_names), "bold")))

    def _choose(self, kind: PresetKind) -> None:
        """Adopt a preset, discarding any hand-picked list so the two cannot silently disagree."""
        self._chosen = kind
        self._custom_models = None
        self._render_presets()

    def _browse_models(self) -> None:
        """Open the model picker seeded with the current selection."""
        exclude = {entry for entry in self._model_entries() if not is_meta_instruction(entry)}
        self.app.push_screen(ModelPickerModal(exclude=exclude), self._on_models_chosen)

    def _on_models_chosen(self, chosen: ModelPickerResult | list[str] | None) -> None:
        """Adopt a hand-picked model list, which replaces the chosen preset's model entries."""
        if isinstance(chosen, ModelPickerResult):
            chosen = chosen.add
        if not chosen:
            return
        self._custom_models = list(chosen)
        self._update_selection_text()

    def _model_entries(self) -> list[str]:
        """The ``models_to_load`` entries the current choice would write."""
        if self._custom_models is not None:
            return list(self._custom_models)
        plan = self._plans.get(self._chosen)
        if plan is not None and plan.entries:
            return list(plan.entries)
        return suggested_default_models()

    # endregion

    # region validation

    def _api_key(self) -> str:
        """The trimmed API key currently entered."""
        return self.query_one("#gs-api-key", Input).value.strip()

    def _worker_name(self) -> str:
        """The trimmed worker name currently entered."""
        return self.query_one("#gs-name", Input).value.strip()

    def _validate_api_key(self) -> bool:
        """Require a non-empty, non-placeholder API key."""
        error = self.query_one("#gs-api-key-error", Static)
        key = self._api_key()
        if not key or key == DEFAULT_API_KEY:
            error.update("Enter your API key (the default placeholder will not work).")
            return False
        error.update("")
        return True

    def _validate_name(self) -> bool:
        """Require a non-empty worker name that is not the default placeholder."""
        error = self.query_one("#gs-name-error", Static)
        name = self._worker_name()
        if not name or name == DEFAULT_DREAMER_NAME:
            error.update("Choose a unique worker name (not the default).")
            return False
        error.update("")
        return True

    def _run_advisory(self, coro: Coroutine[Any, Any, None], *, group: str) -> None:
        """Run an advisory coroutine off the UI thread, or discard it under the test harness."""
        if _testing_mode():
            coro.close()
            return
        self.run_worker(coro, group=group, exclusive=True)

    async def _advisory_key_check(self, api_key: str) -> None:
        """Validate the API key against the horde and toast a hint if it was rejected."""
        result = await asyncio.to_thread(verify_api_key, api_key)
        if result.status is AdvisoryStatus.PROBLEM:
            self.app.notify(
                f"That API key did not validate with the horde ({result.detail or 'rejected'}). "
                "Double-check it before you start the worker.",
                title="API key",
                severity="warning",
                timeout=8,
            )
        elif result.status is AdvisoryStatus.OK and result.detail:
            self.app.notify(f"API key validated for user '{result.detail}'.", title="API key", timeout=4)

    async def _advisory_name_check(self, worker_name: str) -> None:
        """Warn if the chosen worker name is already taken on the horde."""
        result = await asyncio.to_thread(check_worker_name_available, worker_name)
        if result.status is AdvisoryStatus.PROBLEM:
            self.app.notify(
                f"A worker named '{worker_name}' already exists on the horde. If it is not yours, pick "
                "a different name or you will get a 'wrong credentials' error.",
                title="Worker name",
                severity="warning",
                timeout=8,
            )

    # endregion

    def _save(self) -> None:
        """Write the keys this page owns into bridgeData and dismiss, leaving every other key alone."""
        name_ok = self._validate_name()
        key_ok = self._validate_api_key()
        if not (name_ok and key_ok):
            return
        error = self.query_one("#gs-save-error", Static)
        features = PRESETS_BY_KIND[self._chosen].features
        try:
            data = load_config(self._config_path)
            data["api_key"] = self._api_key()
            data["dreamer_name"] = self._worker_name()
            data["models_to_load"] = self._model_entries()
            data["nsfw"] = self.query_one("#gs-nsfw", Switch).value
            for key in FEATURE_KEYS:
                data[key] = features.get(key, False)
            civitai_token = self.query_one("#gs-civitai-token", Input).value.strip()
            # An empty field means "leave whatever is there": the token is also settable on the Config
            # tab, and a preset without LoRA hides this field entirely, so a blank is never a removal.
            if civitai_token:
                data["civitai_api_token"] = civitai_token
            save_config(data, self._config_path)
        except OSError as write_error:
            error.update(f"Could not write {self._config_path}: {write_error}")
            return
        self.dismiss(True)

    def action_close(self) -> None:
        """Leave the page without writing anything; the dashboard stays usable."""
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route the preset choices, the picker, and the save/close actions."""
        button_id = event.button.id or ""
        if button_id.startswith("gs-choose-"):
            self._choose(PresetKind(button_id.removeprefix("gs-choose-")))
        elif button_id == "gs-browse":
            self._browse_models()
        elif button_id == "gs-save":
            self._save()
        elif button_id == "gs-close":
            self.action_close()

    def on_input_blurred(self, event: Input.Blurred) -> None:
        """Ask the horde about a completed field, without ever blocking the page on the answer."""
        value = event.input.value.strip()
        if not value:
            return
        if event.input.id == "gs-name" and value != DEFAULT_DREAMER_NAME:
            self._run_advisory(self._advisory_name_check(value), group="name-check")
        elif event.input.id == "gs-api-key" and value != DEFAULT_API_KEY:
            self._run_advisory(self._advisory_key_check(value), group="key-check")


__all__ = [
    "DEFAULT_API_KEY",
    "DEFAULT_DREAMER_NAME",
    "FEATURE_KEYS",
    "PRESETS",
    "PRESETS_BY_KIND",
    "SUGGESTED_PRESET",
    "GettingStartedScreen",
    "Preset",
    "PresetKind",
    "PresetPlan",
    "build_preset_plans",
    "is_setup_incomplete",
    "preset_entries",
    "resolve_preset_models",
    "suggested_default_models",
]
