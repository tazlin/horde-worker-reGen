"""Beta (PRIMARY pending-queue) models must survive config-load model resolution.

Regression coverage for the bug where an opted-in beta model (e.g. Z-Image) that the operator listed in
``image_models_to_load`` was silently dropped: the config-load filter intersected the list with the SDK
resolver's canonical-only ``resolve_all_model_names``, which never includes pending-queue models.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from horde_worker_regen.bridge_data import load_config
from horde_worker_regen.bridge_data.load_config import (
    BridgeDataLoader,
    _beta_models_for_meta_instructions,
    _meta_instruction_matches_record,
)


def _rec(
    name: str,
    baseline: KNOWN_IMAGE_GENERATION_BASELINE = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
    *,
    nsfw: bool = False,
    inpainting: bool = False,
) -> SimpleNamespace:
    """A duck-typed stand-in for ImageGenerationModelRecord (only the fields the predicate reads)."""
    return SimpleNamespace(name=name, baseline=baseline, nsfw=nsfw, inpainting=inpainting)


def _make_bridge(
    *,
    image_models_to_load: list[str],
    meta_load_instructions: list[str] | None = None,
    meta_skip_instructions: list[str] | None = None,
    image_models_to_skip: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        meta_load_instructions=meta_load_instructions,
        meta_skip_instructions=meta_skip_instructions,
        image_models_to_load=image_models_to_load,
        image_models_to_skip=image_models_to_skip if image_models_to_skip is not None else [],
        only_models_on_disk=False,
    )


class TestMetaInstructionMatchesRecord:
    def test_all_matches_non_large_baselines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AI_HORDE_MODEL_META_LARGE_MODELS", raising=False)
        assert _meta_instruction_matches_record("all", _rec("a", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1))
        assert _meta_instruction_matches_record("all models", _rec("b"))

    def test_all_excludes_large_baselines_unless_opted_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        flux = _rec("flux", KNOWN_IMAGE_GENERATION_BASELINE.flux_1)
        cascade = _rec("cascade", KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade)

        monkeypatch.delenv("AI_HORDE_MODEL_META_LARGE_MODELS", raising=False)
        assert not _meta_instruction_matches_record("all", flux)
        assert not _meta_instruction_matches_record("all", cascade)

        monkeypatch.setenv("AI_HORDE_MODEL_META_LARGE_MODELS", "1")
        assert _meta_instruction_matches_record("all", flux)
        assert _meta_instruction_matches_record("all", cascade)

    def test_baseline_families(self) -> None:
        sdxl = _rec("x", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl)
        sd15 = _rec("y", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1)
        assert _meta_instruction_matches_record("all sdxl", sdxl)
        assert not _meta_instruction_matches_record("all sdxl", sd15)
        assert _meta_instruction_matches_record("all sd15", sd15)
        assert not _meta_instruction_matches_record("all sd15", sdxl)

    def test_sfw_nsfw_and_inpainting(self) -> None:
        nsfw = _rec("n", nsfw=True)
        sfw = _rec("s", nsfw=False)
        paint = _rec("p", inpainting=True)
        assert _meta_instruction_matches_record("all nsfw", nsfw)
        assert not _meta_instruction_matches_record("all nsfw", sfw)
        assert _meta_instruction_matches_record("all sfw", sfw)
        assert _meta_instruction_matches_record("all inpainting", paint)
        assert not _meta_instruction_matches_record("all inpainting", sfw)

    def test_unknown_instruction_matches_nothing(self) -> None:
        assert not _meta_instruction_matches_record("TOP 5", _rec("a"))


def test_beta_models_for_meta_instructions_selects_by_family() -> None:
    records = {
        "sdxl": _rec("sdxl", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl),
        "sd15": _rec("sd15", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1),
    }
    assert _beta_models_for_meta_instructions(["all sdxl"], records) == {"sdxl"}
    assert _beta_models_for_meta_instructions(["all"], records) == {"sdxl", "sd15"}


def test_explicit_beta_model_survives_filter() -> None:
    """A beta model the operator listed explicitly is kept; an unknown model is dropped."""
    bridge = _make_bridge(image_models_to_load=["Z-Image-Turbo", "Bogus-Model"])
    # beta_aware_image_records returns canonical + beta merged; here it includes the beta model.
    beta_records = {
        "Z-Image-Turbo": _rec("Z-Image-Turbo"),
        "Canonical-Model": _rec("Canonical-Model", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1),
    }

    with (
        patch.object(load_config, "beta_aware_image_records", return_value=beta_records),
        patch.object(load_config, "_make_image_model_load_resolver", return_value=Mock()),
    ):
        result = BridgeDataLoader._resolve_meta_instructions(bridge, Mock())

    assert "Z-Image-Turbo" in result
    assert "Bogus-Model" not in result


def test_meta_all_instruction_includes_beta_model() -> None:
    """``load: all`` picks up an opted-in beta model the canonical SDK resolver would have missed."""
    bridge = _make_bridge(image_models_to_load=[], meta_load_instructions=["all"])
    beta_records = {
        "Z-Image-Turbo": _rec("Z-Image-Turbo"),
        "Canonical-Model": _rec("Canonical-Model", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1),
    }
    fake_resolver = Mock()
    fake_resolver.resolve_meta_instructions.return_value = {"Canonical-Model"}

    with (
        patch.object(load_config, "beta_aware_image_records", return_value=beta_records),
        patch.object(load_config, "_make_image_model_load_resolver", return_value=fake_resolver),
        patch.object(load_config, "AIHordeAPIManualClient", Mock()),
    ):
        result = BridgeDataLoader._resolve_meta_instructions(bridge, Mock())

    assert "Z-Image-Turbo" in result
    assert "Canonical-Model" in result
