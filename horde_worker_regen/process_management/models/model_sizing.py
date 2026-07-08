"""Shared classification of how much of a device a model's inference is expected to want.

Two subsystems must agree on which models are "very large": the scheduler's concurrency/whole-card residency
machinery (a large model never co-samples and may claim the card) and the job popper's large-model pop
limiters (throttle switching to a *different* large model, and a re-entry cooldown after they drain). Keeping
the classification in one torch-free module is what stops the two halves from drifting apart: a model the
scheduler treats as whole-card but the popper offers freely would defeat the limiter, and vice versa.

The classification is by baseline (and the named VRAM-heavy checkpoints), not by a live VRAM measurement, so
it is a stable, hardware-independent fact about a model that both the orchestrator and the popper can read
without touching torch or the device.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import IntEnum

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from horde_worker_regen.consts import VRAM_HEAVY_MODELS


class ModelSizeTier(IntEnum):
    """How much of the device a model's inference is expected to want, for concurrency decisions.

    Ordered so heavier tiers compare greater. ``LIGHT`` jobs (SD1.5/SD2) are cheap enough to sample side by
    side; ``HEAVY`` (SDXL) jobs need the job they would overlap to be well underway first; and ``EXTRA_LARGE``
    (Cascade/Flux/Qwen/Z-Image and the named VRAM-heavy checkpoints) effectively want the whole card and never
    share it.
    """

    LIGHT = 0
    HEAVY = 1
    EXTRA_LARGE = 2


LIGHT_BASELINE_VALUES = frozenset(
    {
        KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1.value,
        KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_512.value,
        KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_768.value,
    },
)
"""Baselines treated as light enough to thread together without headway."""

HEAVY_BASELINE_VALUES = frozenset(
    {
        KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl.value,
    },
)
"""Baselines that need the job they overlap to be well underway before joining the card."""

EXTRA_LARGE_BASELINE_VALUES = frozenset(
    {
        KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade.value,
        KNOWN_IMAGE_GENERATION_BASELINE.flux_1.value,
        KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell.value,
        KNOWN_IMAGE_GENERATION_BASELINE.flux_dev.value,
        KNOWN_IMAGE_GENERATION_BASELINE.qwen_image.value,
        KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo.value,
    },
)
"""Baselines that effectively want the whole card and never share it with a concurrent job."""


def model_size_tier(model_name: str | None, baseline_value: str | None) -> ModelSizeTier:
    """Classify a model by how much of the device its inference is expected to want.

    A model named in the VRAM-heavy list, or carrying an extra-large baseline, wants the whole card. SDXL is
    heavy; SD1.5/SD2 are light. An unknown baseline (no loaded reference for the model) falls back to light so
    the overlap gate stays permissive rather than starving dispatch on missing metadata; a genuinely large
    model is always classified by its known baseline (or its presence in the VRAM-heavy list).
    """
    if model_name is not None and model_name in VRAM_HEAVY_MODELS:
        return ModelSizeTier.EXTRA_LARGE
    if baseline_value in EXTRA_LARGE_BASELINE_VALUES:
        return ModelSizeTier.EXTRA_LARGE
    if baseline_value in HEAVY_BASELINE_VALUES:
        return ModelSizeTier.HEAVY
    return ModelSizeTier.LIGHT


def is_extra_large_model(model_name: str | None, baseline_value: str | None) -> bool:
    """Whether a model is in the EXTRA_LARGE tier: a "very large" model that wants the whole card.

    The single predicate the popper's large-model limiters and the scheduler's whole-card intent share, so the
    set of models the limiter throttles is exactly the set the residency machinery would give the card to.
    """
    return model_size_tier(model_name, baseline_value) >= ModelSizeTier.EXTRA_LARGE


def any_offered_model_wants_whole_card(
    model_names: Iterable[str],
    baseline_lookup: Callable[[str], str | None] | None = None,
) -> bool:
    """Whether any model in an offered set is EXTRA_LARGE: a "very large" model that wants the whole card.

    The single question the worker's spawn-time sizing and process-launch paths ask about a configured model
    set, routed through the same :func:`is_extra_large_model` predicate the scheduler and popper use so "very
    large" means one thing across the worker. ``baseline_lookup`` maps a model name to its baseline value when
    a loaded reference is available; without it (before any reference is loaded, as at spawn-time sizing) the
    classification falls back to the named-checkpoint branch of :func:`model_size_tier`, which still recognises
    the named VRAM-heavy checkpoints by name alone.
    """
    for model_name in model_names:
        baseline_value = baseline_lookup(model_name) if baseline_lookup is not None else None
        if is_extra_large_model(model_name, baseline_value):
            return True
    return False
