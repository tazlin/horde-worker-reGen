"""Compose portable image-feature compatibility with card-local eligibility constraints.

The SDK owns the canonical request vocabulary, wire extraction, and directional compatibility predicate.
This module adapts a card's effective worker configuration to that seam, then applies constraints that are
necessarily local: model assignment, weight fit, and resolution. Keeping those layers separate
lets pop placement and dispatch routing use one eligibility decision without copying feature taxonomies.

The module is torch-free. Backend vocabularies come from ``hordelib.pipeline.constants``, whose values are
the execution layer's accepted names and do not initialize ComfyUI or query a device.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import auto
from functools import cache
from typing import TYPE_CHECKING

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.generation_parameters.alchemy.consts import (
    KNOWN_FACEFIXERS,
    KNOWN_MISC_POST_PROCESSORS,
    KNOWN_UPSCALERS,
)
from horde_sdk.generation_parameters.generic.consts import KNOWN_AUX_MODEL_SOURCE
from horde_sdk.generation_parameters.image.constraints import SAMPLER_SOLVER_KNOB, SCHEDULER_BASELINE_APPLICABILITY
from horde_sdk.generation_parameters.image.consts import (
    CLIP_SKIP_REPRESENTATION,
    KNOWN_IMAGE_CONTROLNETS,
    KNOWN_IMAGE_SAMPLERS,
    KNOWN_IMAGE_SCHEDULERS,
    KNOWN_IMAGE_SOURCE_PROCESSING,
    KNOWN_IMAGE_WORKFLOWS,
)
from horde_sdk.generation_parameters.image.object_models import ControlnetFeatureFlags, ImageGenerationFeatureFlags
from horde_sdk.worker.dispatch.ai_horde.image.convert import (
    image_job_pop_response_to_feature_flags,
    image_worker_bridge_data_to_feature_flags,
)
from horde_sdk.worker.feature_flags import (
    IMAGE_WORKER_NOT_CAPABLE_REASON,
    ImageWorkerFeatureFlags,
    PerBaselineFeatureFlags,
)
from loguru import logger
from strenum import StrEnum

if TYPE_CHECKING:
    from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime


class CARD_NOT_CAPABLE_REASON(StrEnum):
    """Card-local reasons that supplement SDK image-feature incompatibility reasons."""

    model_weights = auto()
    model_not_served = auto()
    max_pixels = auto()
    max_batch = auto()


type CardNotCapableReason = IMAGE_WORKER_NOT_CAPABLE_REASON | CARD_NOT_CAPABLE_REASON


@cache
def _implementation_image_features() -> ImageWorkerFeatureFlags:
    """Describe exact backend implementation support before operator choices narrow it.

    Backend value collections are read from hordelib's execution constants, while the SDK supplies the
    portable model. Unknown future baselines are deliberately absent until the backend exposes explicit
    support for them; a model-reference value alone is not evidence that every flat feature works there.

    Returns:
        The backend's portable static image feature profile.
    """
    from hordelib.pipeline.constants import (
        CONTROLNET_IMAGE_PREPROCESSOR_MAP,
        SAMPLERS_MAP,
        SCHEDULERS,
    )
    from hordelib.pipeline.patches import LAYERDIFFUSE_BASELINES

    baselines: list[KNOWN_IMAGE_GENERATION_BASELINE | str] = list(KNOWN_IMAGE_GENERATION_BASELINE)
    samplers = [sampler for sampler in KNOWN_IMAGE_SAMPLERS if sampler.value.lower() in SAMPLERS_MAP]
    backend_control_types = set(CONTROLNET_IMAGE_PREPROCESSOR_MAP)
    controlnet_features = ControlnetFeatureFlags(
        controlnets=[
            control_type for control_type in KNOWN_IMAGE_CONTROLNETS if control_type in backend_control_types
        ],
        image_is_control=True,
        return_control_map=True,
    )
    post_processing = list(
        dict.fromkeys(
            [
                *(processor.value for processor in KNOWN_UPSCALERS),
                *(processor.value for processor in KNOWN_FACEFIXERS),
                *(processor.value for processor in KNOWN_MISC_POST_PROCESSORS),
            ],
        ),
    )
    known_schedulers_by_value = {scheduler.value: scheduler for scheduler in KNOWN_IMAGE_SCHEDULERS}
    schedulers = [
        known_schedulers_by_value[scheduler_name]
        for scheduler_name in SCHEDULERS
        if scheduler_name in known_schedulers_by_value
    ]
    schedulers_by_baseline: dict[
        KNOWN_IMAGE_GENERATION_BASELINE | str,
        list[KNOWN_IMAGE_SCHEDULERS],
    ] = {}
    for baseline in baselines:
        schedulers_by_baseline[baseline] = []
        for scheduler in schedulers:
            applicable_baselines = SCHEDULER_BASELINE_APPLICABILITY.get(scheduler)
            if applicable_baselines is None or baseline in applicable_baselines:
                schedulers_by_baseline[baseline].append(scheduler)

    generation_features = ImageGenerationFeatureFlags(
        extra_texts=True,
        extra_source_images=True,
        baselines=baselines,
        clip_skip=True,
        hires_fix=True,
        tiling=True,
        schedulers=schedulers,
        samplers=samplers,
        sampler_solver_knobs=list(SAMPLER_SOLVER_KNOB),
        flow_shift=True,
        transparent=True,
        controlnets_feature_flags=controlnet_features,
        post_processing=post_processing,
        source_processing=list(KNOWN_IMAGE_SOURCE_PROCESSING),
        workflows=[KNOWN_IMAGE_WORKFLOWS.qr_code],
        tis=[KNOWN_AUX_MODEL_SOURCE.HORDELING],
        loras=[KNOWN_AUX_MODEL_SOURCE.CIVITAI],
    )
    return ImageWorkerFeatureFlags(
        image_generation_feature_flags=generation_features,
        per_baseline_feature_flags=PerBaselineFeatureFlags(
            schedulers_map=schedulers_by_baseline,
            transparent_map={baseline: baseline in LAYERDIFFUSE_BASELINES for baseline in baselines},
        ),
        backend_clip_skip_representation=CLIP_SKIP_REPRESENTATION.NEGATIVE_OFFSET,
    )


def image_worker_feature_flags(
    config: reGenBridgeData,
) -> ImageWorkerFeatureFlags:
    """Return canonical portable image support for one effective card configuration.

    This public boundary adapter intentionally returns the SDK model rather than a worker-owned feature
    record. Runtime readiness and resource constraints are applied by their owning subsystems after this
    static profile is built.
    """
    return image_worker_bridge_data_to_feature_flags(config, _implementation_image_features())


@dataclass(frozen=True)
class CardProfile:
    """Static and configured local facts needed in addition to portable feature support."""

    device_index: int
    total_vram_mb: float | None
    config: reGenBridgeData
    served_models: frozenset[str] | None = None


@dataclass(frozen=True)
class JobRequirements:
    """Canonical render requirements plus facts that only the local card can evaluate."""

    model: str | None
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None
    weight_mb: float | None
    image_features: ImageGenerationFeatureFlags
    pixels: int
    batch: int


@dataclass(frozen=True)
class CardEligibilityVerdict:
    """Per-card eligibility reasons for one accepted job."""

    requirements: JobRequirements
    reasons_by_card: Mapping[int, tuple[CardNotCapableReason, ...]]

    @property
    def eligible_card_indices(self) -> set[int]:
        """Return cards with no attributable incompatibility."""
        return {device_index for device_index, reasons in self.reasons_by_card.items() if not reasons}

    def reason_summary(self) -> str:
        """Render a stable diagnostic for a job no configured card can serve."""
        return "; ".join(
            f"device {device_index}: {', '.join(reason.value for reason in reasons) or 'eligible'}"
            for device_index, reasons in sorted(self.reasons_by_card.items())
        )


def describe_job_requirements(
    job: ImageGenerateJobPopResponse,
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
    weight_mb: float | None,
) -> JobRequirements:
    """Extract canonical portable and local requirements from an accepted image job.

    NSFW is deliberately absent. A popped job carries no NSFW signal: ``use_nsfw_censor`` states whether the
    requester wants accidental NSFW output censored, not that the job is NSFW. The horde server filters NSFW
    work by the ``nsfw`` flag on the offer, so the policy is enforced at pop time (see
    [`gpu_pop_shaping`][horde_worker_regen.process_management.gpu.gpu_pop_shaping]) rather than per card.
    """
    return JobRequirements(
        model=job.model,
        baseline=baseline,
        weight_mb=weight_mb,
        image_features=image_job_pop_response_to_feature_flags(job, resolved_baseline=baseline),
        pixels=int(job.payload.width) * int(job.payload.height),
        batch=int(job.payload.n_iter),
    )


def _weights_fit_card(total_vram_mb: float | None, weight_mb: float | None) -> bool:
    """Return whether weights fit the card budget, abstaining when either fact is unavailable."""
    if weight_mb is None or total_vram_mb is None or total_vram_mb <= 0:
        return True
    try:
        from hordelib.vram_planning import compute_weight_budget_mb

        budget_mb = compute_weight_budget_mb(int(total_vram_mb))
    except Exception as error:  # noqa: BLE001 - an unavailable estimate must not crash routing
        logger.debug(f"Weight-budget lookup failed for {total_vram_mb} MB: {type(error).__name__} {error}")
        return True
    return weight_mb <= budget_mb


def reasons_card_cannot_serve(card: CardProfile, requirements: JobRequirements) -> tuple[CardNotCapableReason, ...]:
    """Return every attributable portable or card-local incompatibility reason."""
    worker_features = image_worker_feature_flags(card.config)
    reasons: list[CardNotCapableReason] = []
    reasons.extend(worker_features.reasons_not_capable_of_features(requirements.image_features) or [])

    if not _weights_fit_card(card.total_vram_mb, requirements.weight_mb):
        reasons.append(CARD_NOT_CAPABLE_REASON.model_weights)
    if (
        card.served_models is not None
        and requirements.model is not None
        and requirements.model not in card.served_models
    ):
        reasons.append(CARD_NOT_CAPABLE_REASON.model_not_served)
    configured_max_pixels = card.config.max_pixels
    if (
        isinstance(configured_max_pixels, int)
        and not isinstance(configured_max_pixels, bool)
        and requirements.pixels > configured_max_pixels
    ):
        reasons.append(CARD_NOT_CAPABLE_REASON.max_pixels)
    configured_max_batch = card.config.max_batch
    if (
        isinstance(configured_max_batch, int)
        and not isinstance(configured_max_batch, bool)
        and requirements.batch > configured_max_batch
    ):
        reasons.append(CARD_NOT_CAPABLE_REASON.max_batch)
    return tuple(reasons)


def card_can_serve(card: CardProfile, requirements: JobRequirements) -> bool:
    """Return whether portable support and every card-local constraint cover the job."""
    return not reasons_card_cannot_serve(card, requirements)


def eligible_cards(cards: Iterable[CardProfile], requirements: JobRequirements) -> set[int]:
    """Return the device indices of every card that can serve the job."""
    return {card.device_index for card in cards if card_can_serve(card, requirements)}


def eligible_card_indices_for(
    job: ImageGenerateJobPopResponse,
    card_runtimes: Mapping[int, CardRuntime],
    *,
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
    weight_mb: float | None,
) -> set[int]:
    """Return cards whose canonical features and local constraints cover an accepted job."""
    return card_eligibility_for(
        job,
        card_runtimes,
        baseline=baseline,
        weight_mb=weight_mb,
    ).eligible_card_indices


def card_eligibility_for(
    job: ImageGenerateJobPopResponse,
    card_runtimes: Mapping[int, CardRuntime],
    *,
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
    weight_mb: float | None,
) -> CardEligibilityVerdict:
    """Return exact per-card compatibility reasons for an accepted job."""
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
    return CardEligibilityVerdict(
        requirements=requirements,
        reasons_by_card={
            profile.device_index: reasons_card_cannot_serve(profile, requirements) for profile in profiles
        },
    )
