"""Shared beta (PRIMARY pending-queue) image-reference helpers for the torch-free worker contexts.

Three torch-free contexts must agree on which image-generation models exist: the orchestrator's
in-memory ``stable_diffusion_reference``, the config-load model filter (``load_config``), and the TUI
picker. Beta models live only in a PRIMARY's pending queue and are merged in via ``query(source=...)``;
they never appear in ``get_all_model_references`` (that path reads only the canonical backend/cache).
A model one context can serve but another has never heard of is silently dropped, so the
"register the pending provider, pick the beta source" dance is centralised here to keep them in step.

``hordelib.beta_models`` is torch-free, so importing it from here does not violate the torch-free
orchestrator invariant.
"""

from __future__ import annotations

from horde_model_reference import SourceSelector
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from loguru import logger


def beta_aware_image_source(manager: ModelReferenceManager) -> SourceSelector:
    """Return the image-generation source selector, registering the beta (pending) provider if opted in.

    A beta model the inference subprocesses can load (hordelib auto-registers the pending provider from
    ``HORDELIB_BETA_MODEL_CATEGORIES``) but the orchestrator/config-load filter have never heard of is
    never offered, scheduled, or advertised, and a job for it would ``KeyError`` on the reference
    lookups. Mirroring the subprocess contract here keeps the contexts in agreement. Beta is
    best-effort: any failure degrades to the canonical source rather than blocking the caller.
    """
    try:
        from horde_model_reference import PENDING_SOURCE_ID
        from hordelib.beta_models import beta_source_for, build_pending_provider

        if manager.get_provider(PENDING_SOURCE_ID) is None:
            provider = build_pending_provider()
            if provider is not None:
                manager.register_provider(provider, replace=True)

        return beta_source_for(MODEL_REFERENCE_CATEGORY.image_generation, manager)
    except Exception as e:  # noqa: BLE001 - beta is best-effort; never block the caller
        from horde_model_reference import HORDE_SOURCE_ID

        logger.warning(f"Could not enable beta models for the image reference: {type(e).__name__}: {e}")
        return HORDE_SOURCE_ID


def beta_aware_image_records(manager: ModelReferenceManager) -> dict[str, ImageGenerationModelRecord]:
    """Return the image-generation records merged with beta (pending) models when opted in.

    ``query()`` keeps the per-category record type through the source-bearing overload (the
    ``image_generation`` overload returns an ``ImageGenerationQuery``), so ``to_list()`` is typed as
    ``list[ImageGenerationModelRecord]`` with no cast, unlike ``get_model_reference`` + source.
    """
    source = beta_aware_image_source(manager)
    records = manager.query(MODEL_REFERENCE_CATEGORY.image_generation, source=source).to_list()
    return {record.name: record for record in records}
