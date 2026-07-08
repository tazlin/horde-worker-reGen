"""Tests for the shared "very large" model classification (:mod:`model_sizing`).

The tier lookup is the single authority for whether a model wants the whole card. Both the named legacy
VRAM-heavy checkpoints (classified by name, no baseline needed) and the extra-large baselines (Cascade, Flux,
Qwen, Z-Image) must resolve to :class:`ModelSizeTier.EXTRA_LARGE`, and the worker-wide
:func:`any_offered_model_wants_whole_card` predicate must agree with the per-model :func:`is_extra_large_model`.
"""

from __future__ import annotations

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from horde_worker_regen.consts import VRAM_HEAVY_MODELS
from horde_worker_regen.process_management.models.model_sizing import (
    ModelSizeTier,
    any_offered_model_wants_whole_card,
    is_extra_large_model,
    model_size_tier,
)


class TestNamedCheckpointsAreExtraLarge:
    """The legacy VRAM-heavy checkpoints classify as EXTRA_LARGE by name, without a baseline."""

    def test_every_legacy_name_is_extra_large_without_a_baseline(self) -> None:
        """Each named VRAM-heavy checkpoint resolves to EXTRA_LARGE from its name alone."""
        for model_name in VRAM_HEAVY_MODELS:
            assert model_size_tier(model_name, None) is ModelSizeTier.EXTRA_LARGE
            assert is_extra_large_model(model_name, None)


class TestExtraLargeBaselines:
    """The extra-large baselines classify as EXTRA_LARGE, including qwen_image and z_image_turbo."""

    def test_qwen_and_z_image_baselines_are_extra_large(self) -> None:
        """A qwen_image or z_image_turbo baseline resolves to EXTRA_LARGE even when the name is not listed."""
        assert is_extra_large_model("some_qwen_checkpoint", KNOWN_IMAGE_GENERATION_BASELINE.qwen_image.value)
        assert is_extra_large_model("some_z_image_checkpoint", KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo.value)

    def test_sdxl_is_not_extra_large(self) -> None:
        """An SDXL model is heavy but shares the card, so it is not EXTRA_LARGE."""
        assert not is_extra_large_model("an_sdxl_model", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl.value)

    def test_sd15_is_not_extra_large(self) -> None:
        """An SD1.5 model is light, so it is not EXTRA_LARGE."""
        assert not is_extra_large_model("an_sd15_model", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1.value)


class TestAnyOfferedModelWantsWholeCard:
    """The worker-wide predicate agrees with the per-model one over an offered set."""

    def test_name_only_detects_a_legacy_checkpoint(self) -> None:
        """Without a baseline lookup, an offered legacy VRAM-heavy checkpoint is still recognised by name."""
        assert any_offered_model_wants_whole_card(["an_sdxl_model", VRAM_HEAVY_MODELS[0]])

    def test_no_large_model_offered_is_false_without_a_lookup(self) -> None:
        """A set of ordinary names carries no whole-card model when no baseline lookup is supplied."""
        assert not any_offered_model_wants_whole_card(["model_a", "model_b"])

    def test_baseline_lookup_promotes_qwen_and_z_image(self) -> None:
        """With a baseline lookup, an offered qwen/z-image model is recognised even though it is not named."""
        baselines = {
            "a_qwen_model": KNOWN_IMAGE_GENERATION_BASELINE.qwen_image.value,
            "an_sdxl_model": KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl.value,
        }
        assert any_offered_model_wants_whole_card(
            ["an_sdxl_model", "a_qwen_model"],
            baseline_lookup=baselines.get,
        )
        assert not any_offered_model_wants_whole_card(
            ["an_sdxl_model"],
            baseline_lookup=baselines.get,
        )

    def test_empty_offered_set_is_false(self) -> None:
        """An empty offered set carries no whole-card model."""
        assert not any_offered_model_wants_whole_card([])
