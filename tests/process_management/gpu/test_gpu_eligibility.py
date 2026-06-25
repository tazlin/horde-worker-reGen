"""Tests for A4/A8 heterogeneous eligibility: which cards can serve a given job.

A card is eligible only if every requirement holds at once: the model's weights fit the card's per-device
weight budget (the heterogeneous check -- a big model fits the 24GB card but not the 8GB one), the card
offers the model, its config enables every feature the job needs, and the resolution is within its
max_power. Any single failure rules the card out; an unknown fact abstains rather than excludes.
"""

from __future__ import annotations

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from horde_worker_regen.process_management.gpu.gpu_eligibility import (
    CardProfile,
    JobRequirements,
    card_can_serve,
    describe_job_requirements,
    eligible_cards,
)
from tests.process_management.conftest import make_job_pop_response, make_mock_bridge_data

_SDXL = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl.value


def _profile(
    *,
    device_index: int,
    total_vram_mb: float | None,
    served: set[str] | None,
    **config: object,
) -> CardProfile:
    """A CardProfile with a config carrying the given feature/resolution overrides.

    Defaults the card to NSFW-enabled and roomy max_pixels so a single axis under test is the differentiator
    (the default job reads as needing NSFW, which would otherwise confound feature/model/weight assertions).
    """
    bridge = make_mock_bridge_data(**config)
    bridge.nsfw = config.get("nsfw", True)
    bridge.max_pixels = config.get("max_pixels", 5_000_000)
    return CardProfile(
        device_index=device_index,
        total_vram_mb=total_vram_mb,
        config=bridge,
        served_models=frozenset(served) if served is not None else None,
    )


class TestCardCanServe:
    """Each requirement rules a card out on its own; unknowns abstain."""

    def test_model_not_offered_excludes_card(self) -> None:
        """A card whose served set lacks the job's model cannot serve it."""
        profile = _profile(device_index=0, total_vram_mb=24576, served={"other_model"})
        requirements = describe_job_requirements(make_job_pop_response(model="wanted"), None, None)
        assert card_can_serve(profile, requirements) is False

    def test_resolution_beyond_max_pixels_excludes_card(self) -> None:
        """A job larger than the card's max_pixels is ineligible there."""
        profile = _profile(device_index=0, total_vram_mb=24576, served={"m"}, max_pixels=1000)
        job = make_job_pop_response(model="m", width=512, height=512)  # 262144 px > 1000
        assert card_can_serve(profile, describe_job_requirements(job, None, None)) is False

    def test_feature_not_allowed_excludes_card(self) -> None:
        """A ControlNet job is ineligible on a card that disables ControlNet."""
        profile = _profile(device_index=0, total_vram_mb=24576, served={"m"}, allow_controlnet=False)
        # The pop payload is frozen, so build the requirements directly to flag a ControlNet need.
        requirements = JobRequirements(
            model="m",
            baseline=None,
            weight_mb=None,
            is_sdxl=False,
            needs_controlnet=True,
            needs_lora=False,
            needs_post_processing=False,
            needs_img2img=False,
            needs_inpainting=False,
            needs_nsfw=False,
            pixels=262144,
        )
        assert card_can_serve(profile, requirements) is False

    def test_unknown_facts_abstain(self) -> None:
        """With no served set and no weight estimate, only feature/resolution can exclude (none do here)."""
        profile = _profile(device_index=0, total_vram_mb=None, served=None)
        job = make_job_pop_response(model="m", width=512, height=512)
        assert card_can_serve(profile, describe_job_requirements(job, None, None)) is True


class TestHeterogeneousWeightFit:
    """A model too large for the small card's weight budget is eligible only on the big card."""

    def test_big_model_fits_only_the_big_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With a budget of 80% of total VRAM, a 10GB-weight model fits the 24GB card but not the 8GB one."""
        import hordelib.vram_planning as vram_planning

        monkeypatch.setattr(vram_planning, "compute_weight_budget_mb", lambda total_vram_mb: total_vram_mb * 0.8)

        big_card = _profile(device_index=0, total_vram_mb=24576, served={"big"})
        small_card = _profile(device_index=1, total_vram_mb=8192, served={"big"})
        job = make_job_pop_response(model="big", width=512, height=512)
        requirements = describe_job_requirements(job, _SDXL, weight_mb=10240.0)  # ~10 GB of weights

        # 24576 * 0.8 = 19660 >= 10240 (fits); 8192 * 0.8 = 6553 < 10240 (does not fit).
        assert card_can_serve(big_card, requirements) is True
        assert card_can_serve(small_card, requirements) is False
        assert eligible_cards([big_card, small_card], requirements) == {0}

    def test_small_model_fits_both_cards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A light model fits both cards' budgets, so both are eligible."""
        import hordelib.vram_planning as vram_planning

        monkeypatch.setattr(vram_planning, "compute_weight_budget_mb", lambda total_vram_mb: total_vram_mb * 0.8)

        big_card = _profile(device_index=0, total_vram_mb=24576, served={"sd15"})
        small_card = _profile(device_index=1, total_vram_mb=8192, served={"sd15"})
        job = make_job_pop_response(model="sd15", width=512, height=512)
        requirements = describe_job_requirements(job, None, weight_mb=2048.0)  # ~2 GB

        assert eligible_cards([big_card, small_card], requirements) == {0, 1}
