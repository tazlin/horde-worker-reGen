"""The merged canonical-plus-beta model reference view for every torch-free worker context.

Beta models are records in a PRIMARY's pending queue. A reader gets them only through ``query`` with the
pending source listed first, because ``get_all_model_references`` reads the canonical source alone. The
parent's ``stable_diffusion_reference``, the config-load filters, the TUI catalog and the benchmark planners
all read the merged view here, so every other context sees a model that one context can serve, as do
the inference children, which resolve the same two sources in the same order.

- :func:`merged_category_records` returns one category's merged records and the names the pending queue
  supplied.
- :func:`beta_aware_image_records` returns the image-generation records as a typed mapping.
- :func:`invalidate_merged_records` drops the per-process cache. The parent's periodic reference refresh
  calls it.
- :func:`share_pending_records_with_children` writes the pending records the parent holds to
  :func:`pending_records_directory` and points the processes the parent spawns at them, so the children
  never fetch the pending queue over the network. :func:`publish_pending_records` writes the files, and
  :func:`pending_publish_retry_seconds` paces the parent's retry of a publish that left a category unwritten.

The opt-in (which categories are in beta, how the pending provider is built, and the source order) is defined
in ``hordelib.beta_models``. It is torch-free and imported lazily, so a process that never reads the view does
not pay for it. The opt-in is not written here, so only a process that already declared beta reaches the
network.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import GenericModelRecord, ImageGenerationModelRecord
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from horde_model_reference.model_reference_manager import ModelReferenceManager

BOUNDED_PENDING_TIMEOUT_SECONDS = 5.0
"""Per-request timeout of the pending provider a bounded read registers."""

BOUNDED_PENDING_RETRY_MAX_ATTEMPTS = 1
"""Attempts of the pending provider a bounded read registers."""

BETA_MODELS_DIR_ENV_VAR = "HORDELIB_BETA_MODELS_DIR"
"""The variable ``hordelib.beta_models`` reads to serve pending records from files instead of the PRIMARY.

hordelib defines the name. It is spelled here because horde-engine releases without the file-backed provider do not
define it, and on those releases the variable is inert.
"""

PENDING_RECORDS_DIR_NAME = "pending"
"""The folder under the model reference base path that holds the published pending records."""


@dataclass(frozen=True)
class MergedCategoryRecords:
    """Represents one category's canonical records with the pending (beta) records laid over them by name."""

    records: Mapping[str, GenericModelRecord]
    """Every record by name. A pending record replaces a canonical record of the same name."""
    beta_names: frozenset[str] = frozenset()
    """The names whose record came from the pending queue."""
    pending_read_ok: bool = True
    """False when the category is opted into beta but the pending half failed open to the canonical records."""


class _MergedViewCache:
    """The per-process cache of merged views, valid for one manager and one opt-in set."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.manager: object | None = None
        self.opted_in: frozenset[MODEL_REFERENCE_CATEGORY] = frozenset()
        self.generation = 0
        self.views: dict[MODEL_REFERENCE_CATEGORY, MergedCategoryRecords] = {}


_CACHE = _MergedViewCache()


def invalidate_merged_records() -> None:
    """Drop every cached merged view so the next read queries the sources again.

    A read already in flight when this runs does not store its result.
    """
    with _CACHE.lock:
        _CACHE.generation += 1
        _CACHE.manager = None
        _CACHE.opted_in = frozenset()
        _CACHE.views = {}


def merged_category_records(
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
    *,
    bounded: bool = False,
) -> MergedCategoryRecords:
    """Return ``category``'s canonical records merged with the pending (beta) queue when the category is opted in.

    The first read for a manager registers hordelib's pending provider on it. A category outside the
    environment's opt-in, or any failure to register or read the pending source, returns the canonical
    records alone, since beta only adds to the canonical view, and marks the result ``pending_read_ok=False``.

    A non-empty result is cached per process for the manager and opt-in set it was read under, until
    :func:`invalidate_merged_records`, with one exception. In the default mode a result whose pending half
    failed open is not cached, so the next reader retries the pending source. The parent reads the view at
    config load and at startup and invalidates it only on the periodic reference refresh, so a cached
    failure would drop every beta model for most of a session. In bounded mode the failed-open result is
    cached like any other, since bounded readers are short-lived and a stale view costs them less than a
    repeated timeout.

    Args:
        manager: The reference manager to read and to register the pending provider on.
        category: The model reference category to read.
        bounded: Register a provider with a single short attempt when none is registered yet. Foreground
            callers in short-lived processes pass this so an unreachable PRIMARY cannot stall them for the
            provider's full retry budget. A provider already registered is reused whatever its bounds.

    Returns:
        The merged records and the names the pending queue supplied.

    Raises:
        Exception: Whatever the canonical read raises. Only the pending half fails open.
    """
    opted_in = _opted_in_categories()
    with _CACHE.lock:
        cache_matches = _CACHE.manager is manager and _CACHE.opted_in == opted_in
        cached = _CACHE.views.get(category) if cache_matches else None
        generation = _CACHE.generation
    if cached is not None:
        return cached

    view = _read_merged(manager, category, opted_in=opted_in, bounded=bounded)
    if not view.records:
        # An empty category is a reference that has not loaded yet. Callers such as the parent's startup
        # retry until it fills, so it must be read again rather than served from the cache.
        return view
    if not view.pending_read_ok and not bounded:
        return view

    with _CACHE.lock:
        if _CACHE.generation == generation:
            if _CACHE.manager is not manager or _CACHE.opted_in != opted_in:
                _CACHE.manager = manager
                _CACHE.opted_in = opted_in
                _CACHE.views = {}
            _CACHE.views[category] = view
    return view


def beta_aware_image_records(
    manager: ModelReferenceManager,
    *,
    bounded: bool = False,
) -> dict[str, ImageGenerationModelRecord]:
    """Return the merged image-generation records by name, as :func:`merged_category_records` reads them.

    Args:
        manager: The reference manager to read.
        bounded: As for :func:`merged_category_records`.

    Returns:
        A new mapping the caller may change freely.
    """
    merged = merged_category_records(manager, MODEL_REFERENCE_CATEGORY.image_generation, bounded=bounded)
    return {name: record for name, record in merged.records.items() if isinstance(record, ImageGenerationModelRecord)}


def pending_records_directory() -> Path:
    """Return the folder the parent publishes pending records to, under the model reference base path."""
    from horde_model_reference import horde_model_reference_paths

    return horde_model_reference_paths.base_path / PENDING_RECORDS_DIR_NAME


def publish_pending_records(
    manager: ModelReferenceManager,
    directory: Path,
) -> frozenset[MODEL_REFERENCE_CATEGORY]:
    """Write each opted-in category's pending records to ``{directory}/{category}.json``.

    The records are the pending half of :func:`merged_category_records`, so they come through the provider
    that read registered on ``manager``. Each file maps model name to ``model_dump(mode="json")``, the shape
    ``StaticModelProvider.from_raw`` reads, and is replaced atomically. A category whose pending read failed
    open keeps its previous file, so a PRIMARY outage does not empty the children's beta records.

    Args:
        manager: The reference manager whose pending provider is read.
        directory: The folder to write the files into. It is created when missing.

    Returns:
        The categories whose file was written.
    """
    written: set[MODEL_REFERENCE_CATEGORY] = set()
    for category in sorted(_opted_in_categories(), key=lambda opted: opted.value):
        view = merged_category_records(manager, category)
        if not view.pending_read_ok:
            logger.warning(f"Kept the previous beta {category.value} file because the pending queue was not read.")
            continue
        pending = {name: view.records[name].model_dump(mode="json") for name in sorted(view.beta_names)}
        _write_json_atomically(directory / f"{category.value}.json", pending)
        written.add(category)
    return frozenset(written)


def share_pending_records_with_children(manager: ModelReferenceManager) -> bool:
    """Publish the pending records and set :data:`BETA_MODELS_DIR_ENV_VAR` for the processes spawned after this.

    Spawned processes inherit the environment, so their pending provider serves these files and does not make a
    network request. The calling process keeps the provider it already registered. A failed write is logged,
    and the children read whatever files the last successful write left. The variable is set whatever the
    outcome.

    Args:
        manager: The reference manager whose pending provider is read.

    Returns:
        True when every opted-in category's file was written, including when no category is opted in.
    """
    directory = pending_records_directory()
    complete = False
    try:
        complete = publish_pending_records(manager, directory) == _opted_in_categories()
    except Exception as publish_error:  # noqa: BLE001 - beta only adds to the canonical view
        logger.warning(
            f"Could not publish the beta records for subprocesses: {type(publish_error).__name__}: {publish_error}"
        )
    os.environ[BETA_MODELS_DIR_ENV_VAR] = str(directory)
    return complete


def pending_publish_retry_seconds(manager: ModelReferenceManager) -> float | None:
    """Return the registered pending provider's cache TTL, the shortest useful wait before a publish is retried.

    Returns:
        The TTL in seconds, or None when no pending provider is registered or it does not state a TTL.
    """
    from horde_model_reference import PENDING_SOURCE_ID

    provider = manager.get_provider(PENDING_SOURCE_ID)
    if provider is None:
        return None
    ttl_seconds = provider.cache_ttl_seconds()
    return None if ttl_seconds is None else float(ttl_seconds)


def _write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
    """Write ``payload`` to ``path`` through a temporary file in the same folder and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temp_file:
            json.dump(payload, temp_file)
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(temp_name).unlink()
        raise


def _opted_in_categories() -> frozenset[MODEL_REFERENCE_CATEGORY]:
    """Return the categories the environment opted into beta, empty when hordelib cannot be imported."""
    try:
        from hordelib.beta_models import beta_model_categories
    except ImportError as import_error:
        logger.warning(f"Beta models are unavailable because hordelib could not be imported: {import_error}")
        return frozenset()
    return frozenset(beta_model_categories())


def _read_merged(
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
    *,
    opted_in: frozenset[MODEL_REFERENCE_CATEGORY],
    bounded: bool,
) -> MergedCategoryRecords:
    """Read one category from both sources, or from the canonical source alone when beta does not apply."""
    pending_read_ok = True
    if category in opted_in:
        try:
            sourced = _read_pending_first(manager, category, bounded=bounded)
        except Exception as pending_error:  # noqa: BLE001 - beta only adds to the canonical view
            logger.warning(
                f"Could not read beta {category.value} models from the pending queue, using the canonical "
                f"reference alone: {type(pending_error).__name__}: {pending_error}",
            )
            sourced = None
        if sourced is not None:
            return sourced
        pending_read_ok = False

    canonical = manager.get_all_model_references().get(category) or {}
    return MergedCategoryRecords(records=MappingProxyType(dict(canonical)), pending_read_ok=pending_read_ok)


def _read_pending_first(
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
    *,
    bounded: bool,
) -> MergedCategoryRecords | None:
    """Register the pending provider once and query ``category`` pending-first, or None without a pending half.

    ``hordelib.beta_models.build_pending_provider`` returns None, and logs why, when the PRIMARY URL or the
    API key is missing. The network provider reports a failed fetch as None instead of raising, and the
    manager's query reads that the same as an empty queue, so the provider is queried first and a None makes
    the read fail open. The query that follows is served from the provider's cache.
    """
    from horde_model_reference import PENDING_SOURCE_ID
    from hordelib.beta_models import beta_source_for, build_pending_provider

    provider = manager.get_provider(PENDING_SOURCE_ID)
    if provider is None:
        provider = (
            build_pending_provider(
                timeout_seconds=BOUNDED_PENDING_TIMEOUT_SECONDS,
                retry_max_attempts=BOUNDED_PENDING_RETRY_MAX_ATTEMPTS,
            )
            if bounded
            else build_pending_provider()
        )
        if provider is None:
            return None
        manager.register_provider(provider, replace=True)

    source = beta_source_for(category, manager)
    if isinstance(source, str):
        return None
    if provider.fetch_category(category) is None:
        return None

    records: dict[str, GenericModelRecord] = {}
    beta_names: set[str] = set()
    for record, source_id in manager.query(category, source=source).to_list_with_source():
        records[record.name] = record
        if source_id == PENDING_SOURCE_ID:
            beta_names.add(record.name)
    return MergedCategoryRecords(records=MappingProxyType(records), beta_names=frozenset(beta_names))
