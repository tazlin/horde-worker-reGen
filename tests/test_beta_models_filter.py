"""Beta (PRIMARY pending-queue) models must survive config-load model resolution.

Regression coverage for the bug where an opted-in beta model (e.g. Z-Image) that the operator listed in
``image_models_to_load`` was silently dropped: the config-load filter intersected the list with the SDK
resolver's canonical-only ``resolve_all_model_names``, which never includes pending-queue models.
"""

from __future__ import annotations

import os
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
    load_large_models: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        meta_load_instructions=meta_load_instructions,
        meta_skip_instructions=meta_skip_instructions,
        image_models_to_load=image_models_to_load,
        image_models_to_skip=image_models_to_skip if image_models_to_skip is not None else [],
        only_models_on_disk=False,
        load_large_models=load_large_models,
    )


class TestMetaInstructionMatchesRecord:
    """Verify the SDK resolver's ``resolve_meta_instructions``."""

    def test_all_matches_non_large_baselines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``all`` and ``all models`` match the canonical baseline records, but not the large ones."""
        monkeypatch.delenv("AI_HORDE_MODEL_META_LARGE_MODELS", raising=False)
        assert _meta_instruction_matches_record("all", _rec("a", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1))
        assert _meta_instruction_matches_record("all models", _rec("b"))

    def test_all_excludes_large_baselines_unless_opted_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``all`` and ``all models`` do not match the large baselines unless the env var is set."""
        flux = _rec("flux", KNOWN_IMAGE_GENERATION_BASELINE.flux_1)
        cascade = _rec("cascade", KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade)

        monkeypatch.delenv("AI_HORDE_MODEL_META_LARGE_MODELS", raising=False)
        assert not _meta_instruction_matches_record("all", flux)
        assert not _meta_instruction_matches_record("all", cascade)

        monkeypatch.setenv("AI_HORDE_MODEL_META_LARGE_MODELS", "1")
        assert _meta_instruction_matches_record("all", flux)
        assert _meta_instruction_matches_record("all", cascade)

    def test_baseline_families(self) -> None:
        """``all sdxl`` matches only the sdxl record; ``all sd15`` matches only the sd15 record."""
        sdxl = _rec("x", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl)
        sd15 = _rec("y", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1)
        assert _meta_instruction_matches_record("all sdxl", sdxl)
        assert not _meta_instruction_matches_record("all sdxl", sd15)
        assert _meta_instruction_matches_record("all sd15", sd15)
        assert not _meta_instruction_matches_record("all sd15", sdxl)

    def test_sfw_nsfw_and_inpainting(self) -> None:
        """``all nsfw`` matches only the nsfw record; ``all sfw`` matches only the sfw record."""
        nsfw = _rec("n", nsfw=True)
        sfw = _rec("s", nsfw=False)
        paint = _rec("p", inpainting=True)
        assert _meta_instruction_matches_record("all nsfw", nsfw)
        assert not _meta_instruction_matches_record("all nsfw", sfw)
        assert _meta_instruction_matches_record("all sfw", sfw)
        assert _meta_instruction_matches_record("all inpainting", paint)
        assert not _meta_instruction_matches_record("all inpainting", sfw)

    def test_unknown_instruction_matches_nothing(self) -> None:
        """A meta instruction that is not recognized matches no records."""
        assert not _meta_instruction_matches_record("TOP 5", _rec("a"))


def test_beta_models_for_meta_instructions_selects_by_family() -> None:
    """``all sdxl`` selects only the sdxl record; ``all`` selects both records.

    Tthe canonical SDK resolver would not include the beta sdxl.
    """
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


def test_meta_instructions_skipped_when_image_generation_reference_absent() -> None:
    """An absent image_generation reference must not crash config load via the SDK resolver.

    horde_sdk's ``ImageModelLoadResolver`` indexes ``all_model_references[image_generation]`` without
    guarding a missing key, so when the image reference is unavailable (a failed reference load, or an
    alchemist-only/CPU worker) every meta-instruction raised KeyError and crashed ``BridgeDataLoader.load``.
    The guard skips the SDK expansion instead.
    """
    bridge = _make_bridge(image_models_to_load=[], meta_load_instructions=["top 2"])

    # Primed to explode exactly like the real SDK would when image_generation is missing.
    exploding_resolver = Mock()
    exploding_resolver.resolve_meta_instructions.side_effect = KeyError("image_generation")

    # A manager whose references lack the image_generation category entirely (the real failure shape).
    manager = Mock()
    manager.get_all_model_references.return_value = {}

    with (
        patch.object(load_config, "beta_aware_image_records", return_value={}),
        patch.object(load_config, "_make_image_model_load_resolver", return_value=exploding_resolver),
        patch.object(load_config, "AIHordeAPIManualClient", Mock()),
    ):
        result = BridgeDataLoader._resolve_meta_instructions(bridge, manager)

    exploding_resolver.resolve_meta_instructions.assert_not_called()
    assert result == []


def _resolve_with_stub_resolver(bridge: SimpleNamespace) -> Mock:
    """Run ``_resolve_meta_instructions`` over ``bridge`` with the SDK resolver/beta records stubbed out.

    Returns the stub resolver so callers can assert how it was invoked.
    """
    fake_resolver = Mock()
    fake_resolver.resolve_meta_instructions.return_value = set()
    with (
        patch.object(load_config, "beta_aware_image_records", return_value={}),
        patch.object(load_config, "_make_image_model_load_resolver", return_value=fake_resolver),
        patch.object(load_config, "AIHordeAPIManualClient", Mock()),
    ):
        BridgeDataLoader._resolve_meta_instructions(bridge, Mock())
    return fake_resolver


def test_load_large_models_false_clears_stale_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reload (e.g. from the TUI) to ``load_large_models: false`` clears a previously-set env var.

    The SDK reads ``AI_HORDE_MODEL_META_LARGE_MODELS`` and it is never otherwise cleared, so without this
    reconciliation a value left by an earlier large-models-on run would keep loading Flux/Stable Cascade.
    """
    monkeypatch.setenv("AI_HORDE_MODEL_META_LARGE_MODELS", "1")
    bridge = _make_bridge(image_models_to_load=[], meta_load_instructions=["all"], load_large_models=False)

    _resolve_with_stub_resolver(bridge)

    assert "AI_HORDE_MODEL_META_LARGE_MODELS" not in os.environ


def test_load_large_models_true_sets_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """``load_large_models: true`` sets the env var the SDK resolver reads."""
    monkeypatch.delenv("AI_HORDE_MODEL_META_LARGE_MODELS", raising=False)
    bridge = _make_bridge(image_models_to_load=[], meta_load_instructions=["all"], load_large_models=True)

    _resolve_with_stub_resolver(bridge)

    assert os.environ["AI_HORDE_MODEL_META_LARGE_MODELS"] == "1"


def test_load_large_models_passed_through_to_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """The config flag is forwarded to the SDK resolver so non-``all`` families honor it too."""
    monkeypatch.delenv("AI_HORDE_MODEL_META_LARGE_MODELS", raising=False)
    bridge = _make_bridge(image_models_to_load=[], meta_load_instructions=["top 5"], load_large_models=False)

    resolver = _resolve_with_stub_resolver(bridge)

    _, kwargs = resolver.resolve_meta_instructions.call_args
    assert kwargs["load_large_models"] is False
