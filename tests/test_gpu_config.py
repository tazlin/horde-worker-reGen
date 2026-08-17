"""Tests for the per-GPU override model and the effective-config resolver (Phase A1 of multi-GPU)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from horde_worker_regen.bridge_data.data_model import GpuOverride, reGenBridgeData
from horde_worker_regen.bridge_data.gpu_config import (
    resolve_all_effective_gpu_configs,
    resolve_effective_gpu_config,
)
from horde_worker_regen.bridge_data.load_config import BridgeDataLoader, ConfigFormat

_API_KEY = "0" * 22


def _make_base(**overrides: object) -> reGenBridgeData:
    """Build a global config with explicit literal models so meta-instruction stripping does not interfere."""
    data: dict[str, object] = {"api_key": _API_KEY, "models_to_load": ["modelA", "modelB"]}
    data.update(overrides)
    return reGenBridgeData.model_validate(data)


class TestGpuOverrideModel:
    """The override model accepts only the whitelisted per-card fields."""

    def test_empty_override_is_all_none(self) -> None:
        """An empty override inherits everything (every field is None)."""
        override = GpuOverride()
        assert override.max_threads is None
        assert override.allow_lora is None
        assert override.image_models_to_load is None

    def test_global_only_field_is_rejected(self) -> None:
        """A global-only field (or a typo) is a loud validation error, never a silent no-op."""
        with pytest.raises(ValidationError):
            GpuOverride.model_validate({"api_key": "x"})

    def test_aliases_are_accepted(self) -> None:
        """The bridge-data aliases (models_to_load / allow_painting) resolve to their fields."""
        override = GpuOverride.model_validate({"models_to_load": ["m"], "allow_painting": True})
        assert override.image_models_to_load == ["m"]
        assert override.allow_inpainting is True

    def test_constraints_mirror_parent(self) -> None:
        """max_power stays within the same 1..512 bound as the global field."""
        with pytest.raises(ValidationError):
            GpuOverride(max_power=0)

    def test_max_batch_constraints_mirror_parent(self) -> None:
        """max_batch stays within the same 1..20 bound as the global field."""
        with pytest.raises(ValidationError):
            GpuOverride(max_batch=0)
        with pytest.raises(ValidationError):
            GpuOverride(max_batch=21)
        assert GpuOverride(max_batch=8).max_batch == 8

    def test_safety_on_gpu_is_overridable(self) -> None:
        """safety_on_gpu is a per-card permission, so a card may withhold it from a permissive global."""
        assert GpuOverride().safety_on_gpu is None
        assert GpuOverride(safety_on_gpu=False).safety_on_gpu is False


class TestResolveInheritance:
    """A card with no override resolves to the global config object itself."""

    def test_none_override_returns_base_unchanged(self) -> None:
        """The single-GPU / no-override case is the global config object, untouched."""
        base = _make_base(max_threads=2, allow_lora=True, max_power=16, nsfw=False)
        resolved = resolve_effective_gpu_config(base, None)
        assert resolved is base
        assert resolved.max_pixels == 16 * 8 * 64 * 64

    def test_empty_override_returns_base_unchanged(self) -> None:
        """An override that sets nothing also inherits the global config wholesale."""
        base = _make_base(max_threads=2)
        assert resolve_effective_gpu_config(base, GpuOverride()) is base

    def test_partial_override_only_changes_named_fields(self) -> None:
        """A delta touches only the fields it sets; the rest still inherit the global, base untouched."""
        base = _make_base(max_threads=4, allow_lora=True, nsfw=True)
        resolved = resolve_effective_gpu_config(base, GpuOverride(max_threads=1, nsfw=False))
        assert resolved is not base
        assert resolved.max_threads == 1
        assert resolved.nsfw is False
        assert resolved.allow_lora is True  # inherited
        assert base.max_threads == 4  # global object is not mutated by resolving a card

    def test_yaml_loaded_base_with_override_does_not_copy_loader(self, tmp_path: Path) -> None:
        """Resolving a YAML-loaded config must not deepcopy ruamel's unpicklable open-stream state."""
        config_path = tmp_path / "bridgeData.yaml"
        config_path.write_text(
            f'api_key: "{_API_KEY}"\nmodels_to_load: [modelA]\ngpu_overrides:\n  0:\n    queue_size: 4\n',
            encoding="utf-8",
        )
        base = BridgeDataLoader.load(config_path, file_format=ConfigFormat.yaml)

        resolved = resolve_all_effective_gpu_configs(base, [0])[0]

        assert resolved is not base
        assert resolved.queue_size == 4
        assert resolved._yaml_loader is None
        assert base._yaml_loader is not None


class TestResolvePerCardLimits:
    """The per-card batch ceiling and safety permission resolve as ordinary deltas."""

    def test_max_batch_resolves_per_card(self) -> None:
        """A card's max_batch overrides the global one and leaves the other cards on it."""
        base = _make_base(max_batch=8, gpu_overrides={1: {"max_batch": 2}})
        resolved = resolve_all_effective_gpu_configs(base, [0, 1])
        assert resolved[0].max_batch == 8
        assert resolved[1].max_batch == 2

    def test_safety_on_gpu_resolves_per_card(self) -> None:
        """A card may withhold the permission to host safety while its sibling keeps the global grant."""
        base = _make_base(safety_on_gpu=True, gpu_overrides={1: {"safety_on_gpu": False}})
        resolved = resolve_all_effective_gpu_configs(base, [0, 1])
        assert resolved[0].safety_on_gpu is True
        assert resolved[1].safety_on_gpu is False

    def test_unknown_key_is_still_rejected(self) -> None:
        """extra="forbid" survives the new fields: a near-miss spelling is a validation error."""
        with pytest.raises(ValidationError):
            GpuOverride.model_validate({"max_batches": 4})
        with pytest.raises(ValidationError):
            GpuOverride.model_validate({"safety_on_the_gpu": True})


class TestBridgeDataTimeoutDefaults:
    """Default watchdog windows should suit average machines without needless reaping."""

    def test_preload_and_post_process_defaults_are_conservative(self) -> None:
        """Startup and post-processing get enough silence budget for normal host churn."""
        base = _make_base()
        assert base.preload_timeout == 150
        assert base.post_process_timeout == 120


class TestResolveCrossFieldRules:
    """The per-card combination is normalised by the same passes the global config runs."""

    def test_queue_size_capped_per_card(self) -> None:
        """A card with max_threads >= 2 has its queue_size capped to 3, like the global validator."""
        base = _make_base()
        resolved = resolve_effective_gpu_config(base, GpuOverride(max_threads=2, queue_size=4))
        assert resolved.queue_size == 3

    def test_controlnet_requires_img2img_per_card(self) -> None:
        """A card enabling controlnet but disabling img2img has controlnet forced off."""
        base = _make_base()
        resolved = resolve_effective_gpu_config(base, GpuOverride(allow_controlnet=True, allow_img2img=False))
        assert resolved.allow_controlnet is False
        assert resolved.allow_sdxl_controlnet is False

    def test_sdxl_controlnet_requires_controlnet_per_card(self) -> None:
        """SDXL controlnet is forced off when plain controlnet is off for the card."""
        base = _make_base()
        resolved = resolve_effective_gpu_config(
            base,
            GpuOverride(allow_sdxl_controlnet=True, allow_controlnet=False, allow_img2img=True),
        )
        assert resolved.allow_sdxl_controlnet is False

    def test_extra_slow_clamps_per_card(self) -> None:
        """A card marked extra-slow has its concurrency clamped exactly as the global override does."""
        base = _make_base()
        resolved = resolve_effective_gpu_config(base, GpuOverride(extra_slow_worker=True, max_threads=4, queue_size=4))
        assert resolved.max_threads == 1
        assert resolved.queue_size == 0
        assert resolved.preload_timeout >= 150

    def test_high_performance_scales_timeout_per_card(self) -> None:
        """A high-performance card gets the 1/3 process_timeout; a default card keeps the full value."""
        base = _make_base()
        fast = resolve_effective_gpu_config(base, GpuOverride(high_performance_mode=True))
        assert fast.process_timeout == 300 // 3
        assert resolve_effective_gpu_config(base, None).process_timeout == 300

    def test_meta_instructions_extracted_for_overridden_models(self) -> None:
        """A per-card model list containing a meta instruction has it pulled into meta_load_instructions."""
        base = _make_base()
        resolved = resolve_effective_gpu_config(base, GpuOverride(image_models_to_load=["top 5", "flux"]))
        assert resolved.image_models_to_load == ["flux"]
        assert resolved.meta_load_instructions == ["top 5"]


class TestResolveAll:
    """The orchestrator resolves the whole configured card set at once."""

    def test_resolve_all_keys_by_index(self) -> None:
        """Each index resolves to its own card; an index without an override inherits the global."""
        base = _make_base(gpu_overrides={0: {"max_threads": 3}, 1: {"nsfw": False}})
        resolved = resolve_all_effective_gpu_configs(base, [0, 1, 2])
        assert resolved[0].max_threads == 3
        assert resolved[1].nsfw is False
        assert resolved[2] is base  # index 2 has no override, inherits the global object


class TestBridgeDataFields:
    """The three new top-level fields parse and validate from raw config."""

    def test_defaults(self) -> None:
        """Absent multi-GPU config means auto-all (None indices), no overrides, 0.5 balance threshold."""
        base = _make_base()
        assert base.gpu_device_indices is None
        assert base.gpu_overrides == {}
        assert base.gpu_pop_balance_threshold == 0.5

    def test_overrides_keyed_by_int(self) -> None:
        """gpu_overrides parses string-or-int YAML keys into int-keyed GpuOverride values."""
        base = _make_base(gpu_device_indices=[0, 1], gpu_overrides={0: {"max_threads": 2}})
        assert base.gpu_device_indices == [0, 1]
        assert isinstance(base.gpu_overrides[0], GpuOverride)
        assert base.gpu_overrides[0].max_threads == 2

    def test_balance_threshold_bounds(self) -> None:
        """The balance threshold is constrained to the unit interval."""
        with pytest.raises(ValidationError):
            _make_base(gpu_pop_balance_threshold=1.5)
