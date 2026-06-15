import asyncio

from horde_model_reference import ModelReferenceManager, PrefetchStrategy


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
