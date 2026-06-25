"""Tests for the heterogeneous per-card eligibility primitive (Phase A4 of multi-GPU)."""

from __future__ import annotations

import uuid

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from pydantic import JsonValue

from horde_worker_regen.bridge_data.data_model import GpuOverride, reGenBridgeData
from horde_worker_regen.bridge_data.gpu_config import resolve_effective_gpu_config
from horde_worker_regen.process_management.gpu.gpu_eligibility import (
    CardProfile,
    card_can_serve,
    describe_job_requirements,
    eligible_cards,
)

_API_KEY = "0" * 22


def _make_job(
    *,
    model: str = "modelA",
    width: int = 512,
    height: int = 512,
    control_type: str | None = None,
    loras: list[dict] | None = None,
    post_processing: list[str] | None = None,
    source_processing: str = "txt2img",
    source_image: str | None = None,
    use_nsfw_censor: bool = True,
) -> ImageGenerateJobPopResponse:
    """Build a real job pop response with the feature flags a test needs."""
    job_id = uuid.uuid4()
    payload: dict[str, JsonValue] = {
        "prompt": "test",
        "width": width,
        "height": height,
        "ddim_steps": 20,
        "n_iter": 1,
        "seed": "1",
        "sampler_name": "k_euler",
        "use_nsfw_censor": use_nsfw_censor,
    }
    if control_type is not None:
        payload["control_type"] = control_type
    if loras is not None:
        payload["loras"] = loras  # pyrefly: ignore
    if post_processing is not None:
        payload["post_processing"] = post_processing
    data: dict[str, JsonValue] = {
        "id": str(job_id),
        "ids": [str(job_id)],
        "model": model,
        "payload": payload,
        "skipped": {},
        "source_processing": source_processing,
    }
    if source_image is not None:
        data["source_image"] = source_image
    return ImageGenerateJobPopResponse(**data)  # pyrefly: ignore


def _config(**overrides: object) -> reGenBridgeData:
    """Resolve an effective per-card config from a global base plus a single-card override."""
    base = reGenBridgeData.model_validate({"api_key": _API_KEY, "models_to_load": ["modelA"]})
    return resolve_effective_gpu_config(base, GpuOverride.model_validate(overrides) if overrides else None)


class TestDescribeJobRequirements:
    """The requirement extraction reads the right payload fields."""

    def test_plain_txt2img(self) -> None:
        """A bare txt2img job needs no features and carries its pixel count."""
        req = describe_job_requirements(
            _make_job(width=640, height=512), baseline="stable_diffusion_xl", weight_mb=None
        )
        assert req.needs_controlnet is False
        assert req.needs_lora is False
        assert req.needs_img2img is False
        assert req.needs_inpainting is False
        assert req.needs_nsfw is False
        assert req.is_sdxl is True
        assert req.pixels == 640 * 512

    def test_feature_flags_detected(self) -> None:
        """Controlnet, lora, post-processing, img2img, inpainting, and nsfw are each detected."""
        cn = describe_job_requirements(_make_job(control_type="canny"), None, None)
        assert cn.needs_controlnet is True
        lora = describe_job_requirements(_make_job(loras=[{"name": "x"}]), None, None)
        assert lora.needs_lora is True
        pp = describe_job_requirements(_make_job(post_processing=["RealESRGAN_x4plus"]), None, None)
        assert pp.needs_post_processing is True
        img = describe_job_requirements(_make_job(source_image="data", source_processing="img2img"), None, None)
        assert img.needs_img2img is True
        inpaint = describe_job_requirements(_make_job(source_image="data", source_processing="inpainting"), None, None)
        assert inpaint.needs_inpainting is True
        nsfw = describe_job_requirements(_make_job(use_nsfw_censor=False), None, None)
        assert nsfw.needs_nsfw is True


class TestCardCanServe:
    """A card serves a job only when every requirement holds."""

    def test_plain_job_served_by_default_card(self) -> None:
        """A default card serves a plain job."""
        card = CardProfile(device_index=0, total_vram_mb=24576, config=_config(), served_models=frozenset({"modelA"}))
        req = describe_job_requirements(_make_job(), baseline="stable_diffusion_1", weight_mb=2000)
        assert card_can_serve(card, req) is True

    def test_model_not_offered_excludes_card(self) -> None:
        """A card that does not offer the job's model cannot serve it."""
        card = CardProfile(device_index=0, total_vram_mb=24576, config=_config(), served_models=frozenset({"other"}))
        req = describe_job_requirements(_make_job(model="modelA"), None, None)
        assert card_can_serve(card, req) is False

    def test_none_served_models_abstains(self) -> None:
        """A card with no resolved model restriction does not exclude on the model check."""
        card = CardProfile(device_index=0, total_vram_mb=24576, config=_config(), served_models=None)
        req = describe_job_requirements(_make_job(model="anything"), None, None)
        assert card_can_serve(card, req) is True

    def test_controlnet_requires_flag(self) -> None:
        """A controlnet job is refused by a card with controlnet disabled."""
        no_cn = CardProfile(0, 24576, _config(allow_controlnet=False), frozenset({"modelA"}))
        req = describe_job_requirements(_make_job(control_type="canny"), "stable_diffusion_1", None)
        assert card_can_serve(no_cn, req) is False

    def test_sdxl_controlnet_requires_sdxl_flag(self) -> None:
        """An SDXL controlnet job needs allow_sdxl_controlnet, not just allow_controlnet."""
        cn_only = CardProfile(
            0,
            24576,
            _config(allow_img2img=True, allow_controlnet=True, allow_sdxl_controlnet=False),
            frozenset({"modelA"}),
        )
        req = describe_job_requirements(_make_job(control_type="canny"), "stable_diffusion_xl", None)
        assert card_can_serve(cn_only, req) is False

    def test_resolution_over_max_power_excludes(self) -> None:
        """A job above the card's max_pixels is refused (resolution gating)."""
        small = CardProfile(0, 24576, _config(max_power=2), frozenset({"modelA"}))  # 2 * 8*64*64 = 65536 px
        req = describe_job_requirements(_make_job(width=512, height=512), None, None)  # 262144 px
        assert card_can_serve(small, req) is False

    def test_nsfw_job_needs_nsfw_card(self) -> None:
        """An uncensored (nsfw) job is refused by an SFW card and served by an NSFW card."""
        sfw = CardProfile(0, 24576, _config(nsfw=False), frozenset({"modelA"}))
        nsfw = CardProfile(1, 24576, _config(nsfw=True), frozenset({"modelA"}))
        req = describe_job_requirements(_make_job(use_nsfw_censor=False), None, None)
        assert card_can_serve(sfw, req) is False
        assert card_can_serve(nsfw, req) is True


class TestWeightBudgetHeterogeneous:
    """The weight-budget check is the heterogeneous discriminator: a too-big model fits only the big card."""

    def test_big_model_only_on_big_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With a stubbed per-device budget, a heavy model is eligible only on the card that can hold it."""

        def fake_budget(total_vram_mb: int, **_kwargs: object) -> float:
            return float(total_vram_mb) * 0.7  # crude but monotonic in card size

        monkeypatch.setattr("hordelib.vram_planning.compute_weight_budget_mb", fake_budget)

        big = CardProfile(0, 24576, _config(), served_models=frozenset({"modelA"}))
        small = CardProfile(1, 8192, _config(), served_models=frozenset({"modelA"}))
        req = describe_job_requirements(_make_job(), baseline="flux_1", weight_mb=12000)  # > 8192*0.7, < 24576*0.7
        assert eligible_cards([big, small], req) == {0}

    def test_unknown_weight_abstains(self) -> None:
        """A missing weight estimate never excludes a card on the budget check."""
        small = CardProfile(1, 8192, _config(), served_models=frozenset({"modelA"}))
        req = describe_job_requirements(_make_job(), baseline="flux_1", weight_mb=None)
        assert eligible_cards([small], req) == {1}
