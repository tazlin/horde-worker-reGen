import asyncio

from horde_model_reference import ModelReferenceManager, PrefetchStrategy
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

    return ModelReferenceManager(offline=True, prefetch_strategy=PrefetchStrategy.NONE)


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
