"""Image-model catalog and meta-instruction helpers for the config editor's model controls.

The model list is loaded lazily from ``horde_model_reference`` (only when the picker is opened, off the
UI thread), so the TUI stays light. The meta-instruction logic mirrors ``horde_sdk``'s ``MetaInstruction``
regexes locally (without importing the heavy SDK) so the editor can build, classify, and explain the
``top N`` / ``all sdxl models`` style commands documented in bridgeData_template.yaml.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
    from horde_model_reference.model_reference_manager import ModelReferenceManager
    from horde_model_reference.model_reference_records import GenericModelRecord

    from horde_worker_regen.model_download_plan import ModelDiskInfo


@dataclasses.dataclass(frozen=True)
class ModelInfo:
    """A single image model from the reference, with the attributes the picker shows and filters on."""

    name: str
    baseline: str
    nsfw: bool
    inpainting: bool
    style: str = ""
    description: str = ""
    version: str = ""
    homepage: str = ""
    tags: tuple[str, ...] = ()
    trigger: tuple[str, ...] = ()
    size_on_disk_bytes: int | None = None
    """Declared on-disk size from the record, or None when the record omits one."""
    on_disk: bool = False
    """Whether the model's files already exist on disk (existence-only, not integrity-checked)."""
    target_path: str = ""
    """Where the model's primary file lives (or would be downloaded to)."""
    is_beta: bool = False
    """Whether this model comes from a PRIMARY's pending (beta) queue rather than the canonical reference."""


def load_image_models() -> list[ModelInfo]:
    """Load the image-generation model reference (blocking; call from a worker thread).

    Imports ``horde_model_reference`` lazily so the TUI does not pay for it at startup. Returns an
    alphabetical list; raises on failure so the caller can surface a clear message.
    """
    from horde_model_reference.model_reference_manager import ModelReferenceManager, PrefetchStrategy

    async def ensure_model_reference_manager_initialized() -> ModelReferenceManager:
        """Asynchronously ensure the model reference manager is initialized and return the instance."""
        if ModelReferenceManager.has_instance():
            return ModelReferenceManager.get_instance()

        horde_model_reference_manager = ModelReferenceManager(
            prefetch_strategy=PrefetchStrategy.ASYNC,
        )

        prefetch_handle = horde_model_reference_manager.deferred_prefetch_handle

        if prefetch_handle is None:
            raise RuntimeError("Failed to get prefetch handle for model reference manager")

        await prefetch_handle

        return horde_model_reference_manager

    manager = asyncio.run(ensure_model_reference_manager_initialized())

    records, beta_names = _image_records_with_beta(manager)

    disk_by_name = _disk_info_by_name(records)

    models = [
        ModelInfo(
            name=name,
            baseline=str(getattr(record, "baseline", "") or ""),
            nsfw=bool(getattr(record, "nsfw", False)),
            inpainting=bool(getattr(record, "inpainting", False)),
            style=str(getattr(record, "style", "") or ""),
            description=str(getattr(record, "description", "") or ""),
            version=str(getattr(record, "version", "") or ""),
            homepage=str(getattr(record, "homepage", "") or ""),
            tags=tuple(str(tag) for tag in (getattr(record, "tags", None) or [])),
            trigger=tuple(str(item) for item in (getattr(record, "trigger", None) or [])),
            size_on_disk_bytes=disk_by_name[name].size_bytes if name in disk_by_name else None,
            on_disk=disk_by_name[name].on_disk if name in disk_by_name else False,
            target_path=disk_by_name[name].target_path if name in disk_by_name else "",
            is_beta=name in beta_names,
        )
        for name, record in records.items()
    ]
    return sorted(models, key=lambda model: model.name.lower())


# Mirrors hordelib.beta_models.BETA_CATEGORIES_ENV_VAR. Duplicated here as a literal so the cheap
# "is beta even opted into?" gate below never has to import hordelib (a heavy import) just to read it.
_BETA_CATEGORIES_ENV_VAR = "HORDELIB_BETA_MODEL_CATEGORIES"


def _image_records_with_beta(
    manager: ModelReferenceManager,
) -> tuple[Mapping[str, GenericModelRecord], set[str]]:
    """Return the image-gen records to show, plus the names that are beta (pending-queue) models.

    The canonical reference (``get_all_model_references``) never includes a PRIMARY's pending-queue
    (beta) models, so a promotion candidate like qwen stays invisible in the picker until it lands in
    the canonical data. When the operator has opted into the image-generation beta (the same env-var
    contract the worker subprocesses use), register the pending provider and merge it via ``query`` so
    those models surface and can be flagged. Beta is best-effort: any failure degrades to the canonical
    catalog rather than breaking the picker.
    """
    from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

    category = MODEL_REFERENCE_CATEGORY.image_generation
    canonical = manager.get_all_model_references().get(category) or {}

    sourced = _query_beta_records(manager, category)
    if sourced is None:
        return canonical, set()

    from horde_model_reference import PENDING_SOURCE_ID

    records: dict[str, GenericModelRecord] = {}
    beta_names: set[str] = set()
    for record, source in sourced:
        records[record.name] = record
        if source == PENDING_SOURCE_ID:
            beta_names.add(record.name)
    return records, beta_names


def _query_beta_records(
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
) -> list[tuple[GenericModelRecord, str]] | None:
    """Register the pending (beta) provider if opted in, and return ``(record, source)`` pairs, or None.

    Returns None (falling back to canonical-only) when beta is not opted into or not fully configured
    (missing API key / PRIMARY URL); ``hordelib.beta_models`` logs the specific reason in that case.
    """
    import os

    if not os.environ.get(_BETA_CATEGORIES_ENV_VAR, "").strip():
        return None

    from loguru import logger

    try:
        from horde_model_reference import PENDING_SOURCE_ID
        from hordelib.beta_models import beta_source_for, build_pending_provider

        if manager.get_provider(PENDING_SOURCE_ID) is None:
            provider = build_pending_provider()
            if provider is None:
                return None
            manager.register_provider(provider, replace=True)

        return manager.query(category, source=beta_source_for(category, manager)).to_list_with_source()
    except Exception as error:  # noqa: BLE001 - beta is best-effort enrichment, never fatal to the picker
        logger.warning(f"Could not load beta models for the picker: {type(error).__name__}: {error}")
        return None


@dataclasses.dataclass(frozen=True)
class DiskSummary:
    """The disk implications of a models-to-load list, for the editor/picker footers."""

    present_bytes: int
    to_download_bytes: int
    total_bytes: int
    free_disk_bytes: int | None
    fits: bool
    shortfall_bytes: int
    num_present: int
    num_to_download: int
    num_unsized: int
    """Entries that contribute no size: meta commands, or names not found in the reference."""
    sizes_complete: bool


def cached_image_records() -> Mapping[str, GenericModelRecord] | None:
    """Return the image-gen reference records if already loaded, else None (never forces a fetch).

    Lets the editor compute a disk total only when the picker has already loaded the catalog, so the
    Config tab never triggers the model-reference network prefetch just by being opened.
    """
    from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
    from horde_model_reference.model_reference_manager import ModelReferenceManager

    if not ModelReferenceManager.has_instance():
        return None
    references = ModelReferenceManager.get_instance().get_all_model_references()
    return references.get(MODEL_REFERENCE_CATEGORY.image_generation) or {}


def disk_summary(model_names: list[str]) -> DiskSummary | None:
    """Summarize the disk cost of a models-to-load list, or None if the reference is not loaded yet.

    Meta commands (``top 5`` etc.) and names absent from the reference cannot be sized; they are
    counted in ``num_unsized`` and excluded from the byte totals.
    """
    records = cached_image_records()
    if records is None:
        return None

    from horde_worker_regen.model_download_plan import compute_download_plan

    recognized = [name for name in model_names if not is_meta_instruction(name) and name in records]
    num_unsized = len(model_names) - len(recognized)
    plan = compute_download_plan(recognized, records)
    return DiskSummary(
        present_bytes=plan.present_bytes,
        to_download_bytes=plan.to_download_bytes,
        total_bytes=plan.total_bytes,
        free_disk_bytes=plan.free_disk_bytes,
        fits=plan.fits,
        shortfall_bytes=plan.shortfall_bytes,
        num_present=plan.num_present,
        num_to_download=plan.num_to_download,
        num_unsized=num_unsized,
        sizes_complete=plan.sizes_complete and num_unsized == 0,
    )


def _disk_info_by_name(records: Mapping[str, GenericModelRecord]) -> dict[str, ModelDiskInfo]:
    """Map each model to its on-disk picture via the torch-free download planner (existence-only).

    Resolves the model directory once and existence-checks every record, so the picker can label
    presence and size without importing hordelib. Failures are swallowed (the catalog still loads).
    """
    try:
        from horde_worker_regen.model_download_plan import compute_download_plan

        plan = compute_download_plan(list(records.keys()), records)
    except Exception:  # noqa: BLE001 - disk planning is best-effort enrichment, never fatal to the catalog
        return {}
    return {info.name: info for info in plan.models}


class MetaKind(enum.StrEnum):
    """A kind of meta model-load instruction the builder can produce."""

    TOP_N = "top_n"
    BOTTOM_N = "bottom_n"
    ALL = "all"
    ALL_SDXL = "all_sdxl"
    ALL_SD15 = "all_sd15"
    ALL_SD21 = "all_sd21"
    ALL_SFW = "all_sfw"
    ALL_NSFW = "all_nsfw"
    ALL_INPAINTING = "all_inpainting"


@dataclasses.dataclass(frozen=True)
class MetaOption:
    """A selectable meta-instruction kind: its label, whether it needs a count, and guidance."""

    kind: MetaKind
    label: str
    needs_count: bool
    guidance: str


META_OPTIONS: list[MetaOption] = [
    MetaOption(
        MetaKind.TOP_N,
        "Top N most popular",
        True,
        "Loads the N most-used models (by the last month's stats). A civitai_api_token is "
        "recommended, as popular models often require it to download.",
    ),
    MetaOption(
        MetaKind.BOTTOM_N, "Bottom N least popular", True, "Loads the N least-used models. Mostly useful for testing."
    ),
    MetaOption(
        MetaKind.ALL,
        "All models",
        False,
        "Loads every model in the reference (over 1 TB of downloads). Almost never what you want.",
    ),
    MetaOption(MetaKind.ALL_SDXL, "All SDXL models", False, "All Stable Diffusion XL models (~7 GB each)."),
    MetaOption(MetaKind.ALL_SD15, "All SD 1.5 models", False, "All Stable Diffusion 1.5 models (~1–2 GB each)."),
    MetaOption(MetaKind.ALL_SD21, "All SD 2.x models", False, "All Stable Diffusion 2.0/2.1 models (larger)."),
    MetaOption(MetaKind.ALL_SFW, "All SFW models", False, "Every model marked safe-for-work."),
    MetaOption(MetaKind.ALL_NSFW, "All NSFW models", False, "Every model marked not-safe-for-work."),
    MetaOption(
        MetaKind.ALL_INPAINTING,
        "All inpainting models",
        False,
        "Every inpainting model (needs allow_painting). Inpainting is heavy for small GPUs.",
    ),
]

META_OPTIONS_BY_KIND = {option.kind: option for option in META_OPTIONS}

GENERAL_META_GUIDANCE = (
    "Meta commands expand to real models when the worker starts. Large models (Flux, Cascade) are "
    "excluded from ALL/TOP unless load_large_models is on. Note: models_to_skip only removes models; "
    "it never adds them back, so 'top 10' minus a skip yields 9 models."
)

# Mirrors horde_sdk.worker.dispatch.ai_horde.bridge_data.MetaInstruction (kept local to stay import-light).
_META_REGEXES: tuple[str, ...] = (
    r"all$|all models?$",
    r"all sdxl$|all sdxl models?$",
    r"all sd15$|all sd15 models?$",
    r"all sd21$|all sd21 models?$",
    r"all sfw$|all sfw models?$",
    r"all nsfw$|all nsfw models?$",
    r"all inpainting$|all inpainting models?$",
    r"top (\d+)$",
    r"bottom (\d+)$",
)


def build_meta_instruction(kind: MetaKind, count: int = 1) -> str:
    """Compose the bridgeData string for a meta kind (e.g. ``top 5`` / ``ALL SDXL MODELS``)."""
    if kind is MetaKind.TOP_N:
        return f"top {max(count, 1)}"
    if kind is MetaKind.BOTTOM_N:
        return f"bottom {max(count, 1)}"
    suffix = {
        MetaKind.ALL: "ALL MODELS",
        MetaKind.ALL_SDXL: "ALL SDXL MODELS",
        MetaKind.ALL_SD15: "ALL SD15 MODELS",
        MetaKind.ALL_SD21: "ALL SD21 MODELS",
        MetaKind.ALL_SFW: "ALL SFW MODELS",
        MetaKind.ALL_NSFW: "ALL NSFW MODELS",
        MetaKind.ALL_INPAINTING: "ALL INPAINTING MODELS",
    }
    return suffix[kind]


def is_meta_instruction(entry: str) -> bool:
    """Whether a models-list entry is a meta instruction rather than a literal model name."""
    candidate = entry.strip()
    return any(re.match(pattern, candidate, re.IGNORECASE) for pattern in _META_REGEXES)


# Ordered most-specific-first so e.g. ``all sdxl`` is not swallowed by the bare ``all`` pattern.
_META_PARSERS: tuple[tuple[MetaKind, str], ...] = (
    (MetaKind.TOP_N, r"top (\d+)$"),
    (MetaKind.BOTTOM_N, r"bottom (\d+)$"),
    (MetaKind.ALL_SDXL, r"all sdxl( models?)?$"),
    (MetaKind.ALL_SD15, r"all sd15( models?)?$"),
    (MetaKind.ALL_SD21, r"all sd21( models?)?$"),
    (MetaKind.ALL_SFW, r"all sfw( models?)?$"),
    (MetaKind.ALL_NSFW, r"all nsfw( models?)?$"),
    (MetaKind.ALL_INPAINTING, r"all inpainting( models?)?$"),
    (MetaKind.ALL, r"all( models?)?$"),
)


def parse_meta_instruction(entry: str) -> tuple[MetaKind, int | None] | None:
    """Classify a models-list entry as a meta instruction, returning its kind and N (for top/bottom).

    Mirrors the worker-side ``MetaInstruction`` matching so the editor can expand the same commands the
    worker will. Returns ``None`` for a literal model name.
    """
    candidate = entry.strip()
    for kind, pattern in _META_PARSERS:
        match = re.match(pattern, candidate, re.IGNORECASE)
        if match:
            count = int(match.group(1)) if kind in (MetaKind.TOP_N, MetaKind.BOTTOM_N) else None
            return kind, count
    return None


def has_popularity_meta(entries: list[str]) -> bool:
    """Whether any entry is a ``top N`` / ``bottom N`` command (the only kinds that need usage stats)."""
    return any(
        (parsed := parse_meta_instruction(entry)) is not None and parsed[0] in (MetaKind.TOP_N, MetaKind.BOTTOM_N)
        for entry in entries
    )


def fetch_model_popularity() -> dict[str, int]:
    """Fetch each image model's last-month usage count from the horde stats API (blocking; off-thread).

    Imports ``horde_sdk`` lazily so the parent process stays light until the user asks to resolve a
    ``top N`` / ``bottom N`` command. Raises on API/transport failure so the caller can surface it.
    """
    from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPIManualClient
    from horde_sdk.ai_horde_api.apimodels import (
        ImageStatsModelsRequest,
        ImageStatsModelsResponse,
        StatsModelsTimeframe,
    )
    from horde_sdk.ai_horde_api.consts import MODEL_STATE
    from horde_sdk.generic_api.apimodels import RequestErrorResponse

    response = AIHordeAPIManualClient().submit_request(
        ImageStatsModelsRequest(model_state=MODEL_STATE.known),
        ImageStatsModelsResponse,
    )
    if isinstance(response, RequestErrorResponse):
        raise RuntimeError(f"Stats request failed: {response.message}")
    timeframe = response.get_timeframe(StatsModelsTimeframe.month)
    return {name: uses for name, uses in timeframe.items() if name and uses is not None}


def describe_entry(entry: str) -> str:
    """A short human label for a models-list entry (annotating meta instructions)."""
    return f"⚙ {entry}  (meta)" if is_meta_instruction(entry) else entry


_FRIENDLY_BASELINES: dict[str, str] = {
    "stable_diffusion_1": "SD 1.5",
    "stable_diffusion_2_512": "SD 2.0",
    "stable_diffusion_2_768": "SD 2.1",
    "stable_diffusion_xl": "SDXL",
    "stable_cascade": "Cascade",
    "flux_1": "Flux",
    "flux_schnell": "Flux Schnell",
    "flux_dev": "Flux Dev",
    "qwen_image": "Qwen",
    "z_image_turbo": "Z-Image Turbo",
}


def friendly_baseline(baseline: str) -> str:
    """A short, readable label for a model baseline (e.g. ``stable_diffusion_xl`` → ``SDXL``)."""
    return _FRIENDLY_BASELINES.get(baseline, baseline)
