"""Custom-model configuration must become one truthful child/parent registry before startup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from hordelib.pipeline.families.image_gen.baselines import CASCADE_BASELINES, UNET_LOADER_BASELINES
from pydantic import ValidationError

from horde_worker_regen.bridge_data.custom_models import (
    CUSTOM_MODEL_BASELINES,
    CustomModelDefinition,
    prepare_custom_models,
)
from horde_worker_regen.bridge_data.data_model import reGenBridgeData


def _definition(path: Path, *, name: str = "Local SDXL") -> CustomModelDefinition:
    return CustomModelDefinition(
        name=name,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
        filepath=str(path),
    )


def test_baseline_choices_follow_dependency_capabilities() -> None:
    """The surface follows HMR/hordelib and cannot regress to a copied tuple of names."""
    assert KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl in CUSTOM_MODEL_BASELINES
    assert KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell in CUSTOM_MODEL_BASELINES
    assert KNOWN_IMAGE_GENERATION_BASELINE.infer not in CUSTOM_MODEL_BASELINES
    assert not set(CUSTOM_MODEL_BASELINES).intersection(UNET_LOADER_BASELINES)
    assert not set(CUSTOM_MODEL_BASELINES).intersection(CASCADE_BASELINES)


@pytest.mark.parametrize(
    "baseline",
    [
        KNOWN_IMAGE_GENERATION_BASELINE.infer,
        KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade,
        KNOWN_IMAGE_GENERATION_BASELINE.qwen_image,
        KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo,
        KNOWN_IMAGE_GENERATION_BASELINE.flux_dev,
    ],
)
def test_one_file_schema_rejects_incompatible_or_forbidden_baselines(
    baseline: KNOWN_IMAGE_GENERATION_BASELINE,
) -> None:
    """A one-file definition cannot promise split/two-stage weights or a forbidden service baseline."""
    with pytest.raises(ValidationError, match="supported single-checkpoint baseline"):
        CustomModelDefinition(name="unsafe", baseline=baseline, filepath="model.safetensors")


def test_preparation_atomically_materializes_hordelib_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A readable custom checkpoint becomes the child registry and matching parent record."""
    monkeypatch.delenv("HORDELIB_CUSTOM_MODELS", raising=False)
    checkpoint = tmp_path / "local.safetensors"
    checkpoint.write_bytes(b"checkpoint")

    result = prepare_custom_models([_definition(checkpoint)], known_model_names=set(), working_directory=tmp_path)

    registry_path = tmp_path / ".horde_worker_regen" / "custom_models.json"
    assert result.registry_path == registry_path
    assert result.ready_names == {"Local SDXL"}
    assert result.issues == ()
    assert result.records["Local SDXL"].baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl
    assert json.loads(registry_path.read_text(encoding="utf-8")) == {
        "Local SDXL": {
            "name": "Local SDXL",
            "baseline": "stable_diffusion_xl",
            "config": {"files": [{"path": str(checkpoint.resolve())}]},
        },
    }


def test_unreadable_definition_is_withheld_from_registry_and_offer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad path remains visible as an issue and is never advertised."""
    monkeypatch.delenv("HORDELIB_CUSTOM_MODELS", raising=False)
    missing = tmp_path / "missing.safetensors"
    bridge = reGenBridgeData.model_validate(
        {
            "api_key": "0" * 22,
            "dreamer_name": "custom-model-test",
            "models_to_load": ["Local SDXL"],
            "custom_models": [
                {
                    "name": "Local SDXL",
                    "baseline": "stable_diffusion_xl",
                    "filepath": str(missing),
                },
            ],
        },
    )

    result = bridge.prepare_custom_models(working_directory=tmp_path)

    assert result.ready_names == set()
    assert "Local SDXL" not in bridge.image_models_to_load
    assert bridge.custom_model_configured_count == 1
    assert bridge.custom_model_issue_summaries == (f"Local SDXL: file does not exist: {missing}",)
    assert json.loads((tmp_path / ".horde_worker_regen" / "custom_models.json").read_text(encoding="utf-8")) == {}


def test_external_legacy_registry_is_validated_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A historical root registry remains operator-owned and must agree with bridgeData."""
    checkpoint = tmp_path / "local.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    registry_path = tmp_path / "custom_models.json"
    original = {
        "Local SDXL": {
            "name": "Local SDXL",
            "baseline": "stable_diffusion_1",
            "config": {"files": [{"path": str(checkpoint)}]},
        },
    }
    registry_path.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setenv("HORDELIB_CUSTOM_MODELS", str(registry_path))

    result = prepare_custom_models([_definition(checkpoint)], known_model_names=set(), working_directory=tmp_path)

    assert result.ready_names == set()
    assert (
        result.issues[0].reason
        == "external HORDELIB_CUSTOM_MODELS entry is missing or does not match baseline/filepath"
    )
    assert json.loads(registry_path.read_text(encoding="utf-8")) == original
    assert not (tmp_path / ".horde_worker_regen" / "custom_models.json").exists()


def test_definition_does_not_implicitly_join_the_offer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defining a model and choosing to offer it remain separate operator actions."""
    monkeypatch.delenv("HORDELIB_CUSTOM_MODELS", raising=False)
    checkpoint = tmp_path / "local.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    bridge = reGenBridgeData.model_validate(
        {
            "api_key": "0" * 22,
            "dreamer_name": "custom-model-test",
            "models_to_load": [],
            "custom_models": [_definition(checkpoint).model_dump(mode="json")],
        },
    )

    bridge.prepare_custom_models(working_directory=tmp_path)

    assert bridge.custom_model_ready_names == {"Local SDXL"}
    assert "Local SDXL" not in bridge.image_models_to_load
