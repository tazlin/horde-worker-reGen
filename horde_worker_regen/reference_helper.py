import asyncio
from collections.abc import Mapping

from horde_model_reference import MODEL_REFERENCE_CATEGORY, ModelReferenceManager, PrefetchStrategy, SourceSelector
from horde_model_reference.model_reference_records import GenericModelRecord
from horde_model_reference.source_consts import HORDE_SOURCE_ID
from loguru import logger


def ensure_offline_reference_manager() -> ModelReferenceManager:
    """Return an offline (read-only, never-download) reference manager for a worker subprocess.

    Subprocesses must never download references: the parent process owns all downloading and writes
    the converted files to disk, and each subprocess simply reads them. Any inherited non-offline
    singleton (possible under a ``fork`` start method, where the child inherits the parent's REPLICA
    downloader) is reset so it cannot trigger a network fetch.
    """
    if ModelReferenceManager.has_instance():
        existing = ModelReferenceManager.get_instance()
        if existing.offline:
            return existing
        logger.debug("Resetting inherited non-offline ModelReferenceManager so this subprocess stays offline")
        ModelReferenceManager.reset()

    manager = ModelReferenceManager(offline=True, prefetch_strategy=PrefetchStrategy.NONE)
    _retry_transient_category_miss(manager)
    return manager


def _retry_transient_category_miss(manager: ModelReferenceManager) -> None:
    """Make a category read that comes back empty re-ask once before it is reported missing.

    The manager decides which categories to reload before it consults the backend, and the backend's own
    time-based expiry runs during that consultation, so a category whose cache expires between the two steps
    is invalidated but not reloaded in that call and reads as absent until the next one, milliseconds later.
    A subprocess reads references on the hot path of every stage, so one such miss faults a job whose sample
    is already paid for. The re-ask closes that window; a category that is genuinely absent still reads as
    absent on the second call.
    """
    original = manager.get_model_reference_or_none

    def get_model_reference_or_none(
        category: MODEL_REFERENCE_CATEGORY,
        overwrite_existing: bool = False,
        *,
        source: SourceSelector = HORDE_SOURCE_ID,
    ) -> Mapping[str, GenericModelRecord] | None:
        found = original(category, overwrite_existing, source=source)
        if found is not None:
            return found
        logger.debug(f"Model reference for {category} read as absent; re-reading once before reporting it missing")
        return original(category, overwrite_existing, source=source)

    manager.get_model_reference_or_none = get_model_reference_or_none  # type: ignore[method-assign]


async def initialize_model_reference_manager() -> ModelReferenceManager:
    """Asynchronously initialize the model reference manager."""
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


def ensure_model_reference_manager_initialized() -> ModelReferenceManager:
    """Ensure that the model reference manager is initialized and return the instance."""
    if ModelReferenceManager.has_instance():
        return ModelReferenceManager.get_instance()

    return asyncio.run(initialize_model_reference_manager())
