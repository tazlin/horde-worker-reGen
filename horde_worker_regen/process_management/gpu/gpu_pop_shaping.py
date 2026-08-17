"""Multi-GPU pop shaping for a worker that presents one identity and one pop stream.

Cards may differ in VRAM, models, feature support, and policy. Independent unions of those axes erase their
correlations and can describe jobs no card serves. Production therefore rotates complete card-scoped offers
for heterogeneous cards and combines only externally equivalent card offers. An under-fed card may be selected
ahead of the fair rotation so the horde returns work that card can actually run.

Public:

- [`AdvertisedCapabilities`][horde_worker_regen.process_management.gpu.gpu_pop_shaping.AdvertisedCapabilities]:
  the aggregate capability envelope for a selected set of cards.
- [`advertised_capabilities`][horde_worker_regen.process_management.gpu.gpu_pop_shaping.advertised_capabilities]:
  build that envelope from the per-card runtime plan.
- [`requires_card_scoped_pops`][horde_worker_regen.process_management.gpu.gpu_pop_shaping.requires_card_scoped_pops]:
  identify plans whose independent capability unions would erase card correlations.

Pure and torch-free: it combines SDK profiles derived from per-card config and hordelib's execution constants,
without initializing the backend. A single-card envelope is that card's exact profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopRequest
from horde_sdk.generation_parameters.image.consts import (
    KNOWN_IMAGE_SOURCE_PROCESSING,
    KNOWN_IMAGE_WORKFLOWS,
)
from horde_sdk.generation_parameters.image.object_models import ImageGenerationFeatureFlags
from horde_sdk.worker.feature_flags import ImageWorkerFeatureFlags, union_image_worker_feature_flags

from horde_worker_regen.consts import EXTENDED_CONTROL_TYPES
from horde_worker_regen.process_management.gpu.gpu_eligibility import image_worker_feature_flags

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime


@dataclass(frozen=True)
class AdvertisedCapabilities:
    """The aggregate capability envelope for a selected set of cards.

    Each field is combined independently. This is an exact pop offer for one card or a set of cards with
    equivalent externally visible job fields. For heterogeneous cards it is useful for analysis, but must be
    narrowed to one card before emission because the aggregate can contain cross-card combinations. Production
    enforces that boundary with :func:`requires_card_scoped_pops`.
    """

    models: frozenset[str]
    """The union of every card's configured image models (the candidate set before stickiness/holdback)."""
    nsfw: bool
    """True only if every card serves NSFW work; a popped job cannot be attributed to one card's policy."""
    image_worker_features: ImageWorkerFeatureFlags
    """The canonical union of portable image features supported by at least one card."""
    max_power: int
    """The largest ``max_power`` across cards (the biggest resolution any card will accept)."""
    max_batch: int
    """The largest ``max_batch`` across cards (the biggest batch any card will accept)."""
    threads: int
    """The summed concurrent-inference ceiling across cards (the worker's total advertised thread count)."""

    @property
    def allow_img2img(self) -> bool:
        """Whether the union supports an image-to-image source mode."""
        source_modes = self.image_worker_features.image_generation_feature_flags.source_processing
        return KNOWN_IMAGE_SOURCE_PROCESSING.img2img in source_modes

    @property
    def allow_inpainting(self) -> bool:
        """Whether the union supports inpainting."""
        source_modes = self.image_worker_features.image_generation_feature_flags.source_processing
        return KNOWN_IMAGE_SOURCE_PROCESSING.inpainting in source_modes

    @property
    def allow_post_processing(self) -> bool:
        """Whether the union supports at least one embedded post-processing operation."""
        return bool(self.image_worker_features.image_generation_feature_flags.post_processing)

    @property
    def allow_controlnet(self) -> bool:
        """Whether the union supports at least one ControlNet type."""
        return bool(self.image_worker_features.image_generation_feature_flags.controlnets_feature_flags)

    @property
    def allow_sdxl_controlnet(self) -> bool:
        """Whether the union supports ControlNet on the SDXL baseline."""
        per_baseline = self.image_worker_features.per_baseline_feature_flags
        return bool(
            per_baseline
            and per_baseline.controlnet_map
            and per_baseline.controlnet_map.get(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl, False)
        )

    @property
    def allow_extended_controlnet(self) -> bool:
        """Whether the union covers every control type behind the AI Horde's all-extended offer."""
        controlnet_features = self.image_worker_features.image_generation_feature_flags.controlnets_feature_flags
        if controlnet_features is None:
            return False
        return EXTENDED_CONTROL_TYPES.issubset(set(controlnet_features.controlnets))

    @property
    def allow_lora(self) -> bool:
        """Whether the union supports at least one LoRA source."""
        return bool(self.image_worker_features.image_generation_feature_flags.loras)


def advertised_capabilities(card_runtimes: Mapping[int, CardRuntime]) -> AdvertisedCapabilities:
    """Build an independently aggregated capability envelope from a card runtime plan.

    Portable feature values are unioned; ``max_power`` and ``max_batch`` are max-ed, threads are summed, and
    models are unioned. The result is directly safe to emit only for a singleton or externally equivalent
    cards. Empty input is rejected because the SDK image profile requires at least one supported baseline;
    the caller handles an absent card plan before invoking this function.

    NSFW is AND-ed rather than unioned. A returned job carries no NSFW signal, so a job popped under an NSFW
    offer cannot be attributed to the cards that permit NSFW; a mixed fleet must therefore offer SFW work
    only; the popper applies the fleet-wide value to card-scoped offers as well.

    Args:
        card_runtimes: The driven cards keyed by stable device index.

    Returns:
        The union envelope as an
        [`AdvertisedCapabilities`][horde_worker_regen.process_management.gpu.gpu_pop_shaping.AdvertisedCapabilities].
    """
    models: set[str] = set()
    nsfw = True
    feature_profiles: list[ImageWorkerFeatureFlags] = []
    max_power = 0
    max_batch = 0
    threads = 0

    for card in card_runtimes.values():
        config = card.config
        models.update(config.image_models_to_load)
        nsfw = nsfw and bool(config.nsfw)
        feature_profiles.append(image_worker_feature_flags(config))
        max_power = max(max_power, int(config.max_power))
        max_batch = max(max_batch, int(config.max_batch))
        threads += int(card.max_concurrent_inference)

    if not feature_profiles:
        raise ValueError("At least one card runtime is required to advertise capabilities.")

    return AdvertisedCapabilities(
        models=frozenset(models),
        nsfw=nsfw,
        image_worker_features=union_image_worker_feature_flags(feature_profiles),
        max_power=max_power,
        max_batch=max_batch,
        threads=threads,
    )


def requires_card_scoped_pops(card_runtimes: Mapping[int, CardRuntime]) -> bool:
    """Return whether one union offer could describe a combination no single card can serve.

    The AI Horde pop shape carries independent model, feature, policy and resolution fields. Unioning
    heterogeneous cards loses the correlations between those fields: a model from one card can be combined
    with a feature or resolution contributed by another. A union remains rectangular and safe only when every
    card exposes the same models, feature profile, NSFW policy, power ceiling and batch ceiling. Batch size
    is a per-job field, so a union that raised it above a card's own ceiling would describe jobs that card
    must refuse. Thread counts may differ because they affect capacity rather than the shape of an individual
    returned job.

    Args:
        card_runtimes: The driven cards keyed by stable device index.

    Returns:
        True when production should rotate card-scoped offers instead of emitting one combined offer.
    """
    if len(card_runtimes) <= 1:
        return False
    per_card = [advertised_capabilities({device_index: card}) for device_index, card in card_runtimes.items()]
    first = per_card[0]
    return any(
        capability.models != first.models
        or capability.nsfw != first.nsfw
        or capability.image_worker_features != first.image_worker_features
        or capability.max_power != first.max_power
        or capability.max_batch != first.max_batch
        for capability in per_card[1:]
    )


def pop_request_supports_image_features(
    pop_request: ImageGenerateJobPopRequest,
    features: ImageGenerationFeatureFlags,
) -> bool:
    """Return whether a coarse AI Horde pop offer covers canonical image requirements.

    The pop protocol exposes broad booleans rather than the SDK's exact value sets. This projection is kept
    at that wire boundary so simulations and production shaping share the same interpretation and neither
    defines another feature record.

    Args:
        pop_request: The outgoing coarse capability offer.
        features: Canonical requirements for a candidate job.

    Returns:
        Whether the service may assign the candidate to this offer.
    """
    source_modes = set(features.source_processing)
    needs_painting = bool(
        source_modes
        & {
            KNOWN_IMAGE_SOURCE_PROCESSING.inpainting,
            KNOWN_IMAGE_SOURCE_PROCESSING.outpainting,
        }
    )
    needs_source_image = bool(
        features.extra_source_images
        or features.controlnets_feature_flags
        or features.workflows
        or source_modes - {KNOWN_IMAGE_SOURCE_PROCESSING.txt2img}
    )
    if needs_source_image and not pop_request.allow_img2img:
        return False
    if needs_painting and not pop_request.allow_painting:
        return False
    # The wire offer names LoRA specifically. Textual inversions have no corresponding
    # pop capability bit and remain independently supported by the worker profile.
    if features.loras and not pop_request.allow_lora:
        return False
    if features.post_processing and not pop_request.allow_post_processing:
        return False

    controlnet_types = set(
        features.controlnets_feature_flags.controlnets if features.controlnets_feature_flags else [],
    )
    qr_workflow = bool(features.workflows and KNOWN_IMAGE_WORKFLOWS.qr_code in features.workflows)
    needs_controlnet = bool(controlnet_types or qr_workflow)
    if needs_controlnet and not pop_request.allow_controlnet:
        return False
    if controlnet_types & EXTENDED_CONTROL_TYPES and not pop_request.allow_extended_controlnet:
        return False

    needs_sdxl_controlnet = bool(
        qr_workflow or (needs_controlnet and KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl in features.baselines)
    )
    return not (needs_sdxl_controlnet and not pop_request.allow_sdxl_controlnet)


def under_fed_card(
    eligible_card_sets: Sequence[frozenset[int] | set[int]],
    card_indices: Iterable[int],
    *,
    balance_threshold: float,
) -> int | None:
    """Return the card most starved by the current local queue, when the imbalance crosses the threshold.

    For each driven card, this measures the fraction of held jobs that card *cannot* serve. A card that
    cannot serve at least ``balance_threshold`` of the held work is under-fed: the local queue is dominated
    by work only other cards can run, so the next pop should be scoped to this card's capabilities to draw
    work it can actually run. Returns the most under-fed such card, or None when the queue is empty, there is
    only one card, or no card is starved past the threshold (in which case the worker uses the topology's
    normal safe-union or card-rotation strategy).

    Args:
        eligible_card_sets: One eligible-card set per held job (from
            :func:`~horde_worker_regen.process_management.gpu.gpu_eligibility.eligible_card_indices_for`).
        card_indices: The driven cards' stable device indices.
        balance_threshold: The fraction of held work a card must be unable to serve to count as under-fed.

    Returns:
        The device index of the most under-fed card, or None.
    """
    cards = list(card_indices)
    total = len(eligible_card_sets)
    if total == 0 or len(cards) <= 1:
        return None

    worst_card: int | None = None
    worst_unservable_fraction = 0.0
    for card in cards:
        unservable_fraction = sum(1 for eligible in eligible_card_sets if card not in eligible) / total
        if unservable_fraction >= balance_threshold and unservable_fraction > worst_unservable_fraction:
            worst_unservable_fraction = unservable_fraction
            worst_card = card
    return worst_card
