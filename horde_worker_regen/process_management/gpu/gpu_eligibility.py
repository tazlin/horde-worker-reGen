"""Which cards can serve a given job: the shared heterogeneous-eligibility primitive.

A multi-GPU worker pops one job stream for its single horde identity, so per-card capability gating is
worker-side routing: when the horde returns a job, the worker must decide which cards could actually run
it (the small card may lack the VRAM for the model, a card may not offer that model, a card may have
controlnet disabled, the resolution may exceed a card's ``max_power``) and dispatch only there. This
module is that decision, factored out as a pure function so the three consumers (preload placement,
dispatch routing, and pop shaping) all agree on the same notion of "eligible".

It is deliberately torch-free: it runs in the orchestrator process, reasons only about already-derived
facts (each card's effective :class:`~horde_worker_regen.bridge_data.data_model.reGenBridgeData`, its total
VRAM, and a job's requirements extracted once via :func:`describe_job_requirements`), and reaches hordelib
only for the torch-free weight-budget primitive (:func:`hordelib.vram_planning.compute_weight_budget_mb`),
lazily and best-effort. An unknown fact never *excludes* a card: a missing weight estimate or VRAM figure
means the weight check abstains rather than wrongly ruling a card out.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from loguru import logger

if TYPE_CHECKING:
    from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime

# Source-processing values (from horde_sdk.generation_parameters.image.consts.KNOWN_IMAGE_SOURCE_PROCESSING)
# that imply an img2img-class job; ``inpainting`` is gated separately on allow_inpainting.
_IMG2IMG_SOURCE_PROCESSING = frozenset({"img2img", "remix"})
_INPAINTING_SOURCE_PROCESSING = "inpainting"


@dataclass(frozen=True)
class CardProfile:
    """The static and configured facts about one card needed to judge job eligibility.

    ``served_models`` is the card's resolved served model set (its ``image_models_to_load`` with
    ``image_models_to_skip`` and any meta-instructions already expanded by the caller); ``None`` means the
    card applies no per-card model restriction (e.g. before the model set is resolved), in which case the
    model-offered check abstains. ``total_vram_mb`` is the card's total VRAM for the weight-budget check,
    or ``None`` when unknown (the check then abstains).
    """

    device_index: int
    total_vram_mb: float | None
    config: reGenBridgeData
    served_models: frozenset[str] | None = None


@dataclass(frozen=True)
class JobRequirements:
    """What a popped job needs from a card, extracted once (torch-free) from the job and reference.

    Computed by :func:`describe_job_requirements` so :func:`card_can_serve` does no payload digging and the
    requirement extraction is unit-testable on its own. ``weight_mb`` is the resident weight footprint (from
    :func:`~horde_worker_regen.process_management.resources.resource_budget.predict_job_weight_mb`); ``None`` when it
    cannot be estimated, which makes the weight-budget check abstain.
    """

    model: str | None
    baseline: str | None
    weight_mb: float | None
    is_sdxl: bool
    needs_controlnet: bool
    needs_lora: bool
    needs_post_processing: bool
    needs_img2img: bool
    needs_inpainting: bool
    needs_nsfw: bool
    pixels: int


def describe_job_requirements(
    job: ImageGenerateJobPopResponse,
    baseline: str | None,
    weight_mb: float | None,
) -> JobRequirements:
    """Extract the per-card eligibility requirements of ``job`` from its payload and resolved metadata.

    Args:
        job: The popped job to inspect.
        baseline: The job model's baseline (from ``ModelMetadata.get_baseline``), or None if unknown.
        weight_mb: The job's resident weight footprint in MB (from ``predict_job_weight_mb``), or None.

    Returns:
        The :class:`JobRequirements` for the job.
    """
    payload = job.payload

    source_processing = str(job.source_processing) if job.source_processing is not None else ""
    needs_img2img = job.source_image is not None or source_processing in _IMG2IMG_SOURCE_PROCESSING
    needs_inpainting = source_processing == _INPAINTING_SOURCE_PROCESSING

    return JobRequirements(
        model=job.model,
        baseline=baseline,
        weight_mb=weight_mb,
        is_sdxl=baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl.value,
        needs_controlnet=bool(payload.control_type),
        needs_lora=bool(payload.loras),
        needs_post_processing=bool(payload.post_processing),
        needs_img2img=needs_img2img,
        needs_inpainting=needs_inpainting,
        # A job the requester allows to be served uncensored (use_nsfw_censor False) was only routed to NSFW
        # workers by the server; only a card with nsfw enabled may run it. A censored job any card can serve.
        needs_nsfw=not payload.use_nsfw_censor,
        pixels=int(payload.width) * int(payload.height),
    )


def _weights_fit_card(total_vram_mb: float | None, weight_mb: float | None) -> bool:
    """Whether a model's resident weights fit a card's per-device weight budget (abstains when unknown).

    Uses hordelib's torch-free :func:`hordelib.vram_planning.compute_weight_budget_mb`, the same per-device
    budget the heterogeneous placement reasons about. Returns True (do not exclude) when either figure is
    unknown or the budget lookup fails, so a missing estimate never wrongly rules a card out.
    """
    if weight_mb is None or total_vram_mb is None or total_vram_mb <= 0:
        return True
    try:
        from hordelib.vram_planning import compute_weight_budget_mb

        budget_mb = compute_weight_budget_mb(int(total_vram_mb))
    except Exception as e:  # noqa: BLE001 - a budget lookup failure must never crash routing
        logger.debug(f"Weight-budget lookup failed for {total_vram_mb} MB: {type(e).__name__} {e}")
        return True
    return weight_mb <= budget_mb


def card_can_serve(card: CardProfile, req: JobRequirements) -> bool:
    """Return whether ``card`` can serve a job with the given requirements.

    A card is eligible only if every requirement holds: the model's weights fit the card's weight budget,
    the card offers the model, the card's effective config enables every feature the job needs, and the
    resolution is within the card's ``max_power``. Any single failure rules the card out. Unknown facts
    (no weight estimate, no VRAM figure, no resolved model set) abstain rather than exclude.
    """
    config = card.config

    if not _weights_fit_card(card.total_vram_mb, req.weight_mb):
        return False

    if card.served_models is not None and req.model is not None and req.model not in card.served_models:
        return False

    if req.needs_controlnet and not config.allow_controlnet:
        return False
    if req.needs_controlnet and req.is_sdxl and not config.allow_sdxl_controlnet:
        return False
    if req.needs_lora and not config.allow_lora:
        return False
    if req.needs_post_processing and not config.allow_post_processing:
        return False
    if req.needs_img2img and not config.allow_img2img:
        return False
    if req.needs_inpainting and not config.allow_inpainting:
        return False
    if req.needs_nsfw and not config.nsfw:
        return False

    return req.pixels <= config.max_pixels


def eligible_cards(cards: Iterable[CardProfile], req: JobRequirements) -> set[int]:
    """Return the device indices of every card that can serve the job (see :func:`card_can_serve`)."""
    return {card.device_index for card in cards if card_can_serve(card, req)}


def eligible_card_indices_for(
    job: ImageGenerateJobPopResponse,
    card_runtimes: Mapping[int, CardRuntime],
    *,
    baseline: str | None,
    weight_mb: float | None,
) -> set[int]:
    """Return the device indices of the cards whose effective config can serve ``job``.

    The one place that turns a per-card runtime plan into :class:`CardProfile` profiles and runs the
    eligibility check, so preload placement, dispatch routing, and pop shaping all build the profiles the
    same way. The caller supplies the job's already-derived ``baseline`` (``ModelMetadata.get_baseline``) and
    ``weight_mb`` (``predict_job_weight_mb``) so this stays free of those dependencies.

    Args:
        job: The popped job to route.
        card_runtimes: The driven cards keyed by stable device index.
        baseline: The job model's baseline, or None if unknown.
        weight_mb: The job's resident weight footprint in MB, or None if it cannot be estimated.

    Returns:
        The device indices of every eligible card (empty when no card can serve the job).
    """
    requirements = describe_job_requirements(job, baseline, weight_mb)
    profiles = [
        CardProfile(
            device_index=card.device_index,
            total_vram_mb=card.total_vram_mb,
            config=card.config,
            served_models=frozenset(card.config.image_models_to_load),
        )
        for card in card_runtimes.values()
    ]
    return eligible_cards(profiles, requirements)
