"""Curated hardware presets for the TUI config editor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class PresetCategory(StrEnum):
    """High-level grouping for preset changes."""

    THROUGHPUT = "Throughput"
    WORKLOAD = "Models"
    FEATURES = "Features"
    DOWNLOADS = "LoRA & Downloads"
    SAFETY = "Safety"
    ALCHEMY = "Alchemy"
    ADVANCED = "Advanced"


@dataclass(frozen=True)
class PresetChange:
    """One proposed setting change inside a preset."""

    key: str
    value: Any
    rationale: str
    category: PresetCategory
    default_selected: bool = True
    requires_restart: bool = False


@dataclass(frozen=True)
class ConfigPreset:
    """A named collection of bridgeData changes."""

    preset_id: str
    label: str
    description: str
    changes: tuple[PresetChange, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PresetDiff:
    """A preset change compared with the current config state."""

    change: PresetChange
    current_value: Any
    changed: bool


def _change(
    key: str,
    value: Any,  # noqa: ANN401 - heterogeneous bridgeData values
    rationale: str,
    category: PresetCategory,
    *,
    default_selected: bool = True,
    requires_restart: bool = False,
) -> PresetChange:
    """Short helper for declaring preset changes."""
    return PresetChange(
        key=key,
        value=value,
        rationale=rationale,
        category=category,
        default_selected=default_selected,
        requires_restart=requires_restart,
    )


BUILT_IN_PRESETS: tuple[ConfigPreset, ...] = (
    ConfigPreset(
        preset_id="rtx4090_64gb_sdxl_balanced",
        label="RTX 4090 / 64 GB RAM - SDXL balanced",
        description="Roomy single-card preset for SD 1.5 + SDXL with common features enabled.",
        changes=(
            _change("max_threads", 1, "Keep one active sampler for stable throughput.", PresetCategory.THROUGHPUT),
            _change("queue_size", 2, "Use system RAM to keep follow-up work staged.", PresetCategory.THROUGHPUT),
            _change("max_batch", 8, "4090-class cards can handle larger batches.", PresetCategory.THROUGHPUT),
            _change("max_power", 64, "Offer up to roughly 1024-class requests.", PresetCategory.THROUGHPUT),
            _change("safety_on_gpu", True, "Use GPU safety for faster submission.", PresetCategory.SAFETY),
            _change(
                "high_performance_mode",
                True,
                "Pop more aggressively on high-end hardware.",
                PresetCategory.THROUGHPUT,
            ),
            _change(
                "moderate_performance_mode",
                False,
                "High-performance mode owns the performance setting.",
                PresetCategory.THROUGHPUT,
            ),
            _change(
                "unload_models_from_vram_often",
                False,
                "Keep model residency for better reuse on roomy VRAM.",
                PresetCategory.THROUGHPUT,
            ),
            _change("allow_post_processing", True, "Offer upscale and face-fix work.", PresetCategory.FEATURES),
            _change(
                "dedicated_post_processing",
                "auto",
                "Run the lane only when served work needs it.",
                PresetCategory.FEATURES,
                requires_restart=True,
            ),
            _change("allow_controlnet", True, "Enable classic SD1.5 ControlNet.", PresetCategory.FEATURES),
            _change("allow_sdxl_controlnet", True, "Enable SDXL QR/transparency workflows.", PresetCategory.FEATURES),
            _change(
                "allow_lora",
                True,
                "Offer LoRA jobs when a CivitAI token is configured.",
                PresetCategory.DOWNLOADS,
                default_selected=False,
            ),
            _change(
                "load_large_models",
                False,
                "Keep TOP/ALL selectors away from Flux/Qwen/Z-Image.",
                PresetCategory.WORKLOAD,
            ),
            _change(
                "models_to_load",
                ["Deliberate", "AlbedoBase XL (SDXL)"],
                "Serve representative SD 1.5 and SDXL models.",
                PresetCategory.WORKLOAD,
                requires_restart=True,
            ),
        ),
    ),
    ConfigPreset(
        preset_id="rtx4090_64gb_large_models",
        label="RTX 4090 / 64 GB RAM - large models",
        description="Conservative whole-card setup for Flux/Qwen/Z-Image style models.",
        warnings=("Large models are disruptive: this preset disables most optional feature load.",),
        changes=(
            _change("max_threads", 1, "Large models should run one at a time.", PresetCategory.THROUGHPUT),
            _change(
                "queue_size", 0, "Avoid aging jobs while a large model holds the card.", PresetCategory.THROUGHPUT
            ),
            _change("max_batch", 1, "Avoid extra activation pressure.", PresetCategory.THROUGHPUT),
            _change("max_power", 64, "Keep request size bounded.", PresetCategory.THROUGHPUT),
            _change(
                "safety_on_gpu", False, "Free the safety CUDA context for model residency.", PresetCategory.SAFETY
            ),
            _change(
                "high_performance_mode",
                False,
                "Avoid aggressive pop timing for slow heavy jobs.",
                PresetCategory.THROUGHPUT,
            ),
            _change(
                "moderate_performance_mode",
                False,
                "Avoid aggressive pop timing for slow heavy jobs.",
                PresetCategory.THROUGHPUT,
            ),
            _change(
                "unload_models_from_vram_often",
                False,
                "Keep the active large model resident.",
                PresetCategory.THROUGHPUT,
            ),
            _change(
                "allow_post_processing", False, "Avoid extra post-generation VRAM spikes.", PresetCategory.FEATURES
            ),
            _change(
                "dedicated_post_processing",
                "off",
                "Do not keep post-processing context resident.",
                PresetCategory.FEATURES,
                requires_restart=True,
            ),
            _change("allow_controlnet", False, "Avoid ControlNet memory overhead.", PresetCategory.FEATURES),
            _change("allow_sdxl_controlnet", False, "Avoid ControlNet memory overhead.", PresetCategory.FEATURES),
            _change("allow_lora", False, "Avoid ad-hoc load/download overhead.", PresetCategory.DOWNLOADS),
            _change("load_large_models", True, "Allow very large models in selections.", PresetCategory.WORKLOAD),
            _change(
                "models_to_load",
                ["Flux.1-Schnell fp8 (Compact)", "Qwen-Image", "Z-Image-Turbo"],
                "Offer the large-model family explicitly.",
                PresetCategory.WORKLOAD,
                requires_restart=True,
            ),
        ),
    ),
    ConfigPreset(
        preset_id="rtx2080_32gb_sd15_safe",
        label="RTX 2080 / 32 GB RAM - SD 1.5 safe",
        description="Low-pressure SD 1.5-only setup for 8-10 GB VRAM cards.",
        changes=(
            _change("max_threads", 1, "Keep one active sampler.", PresetCategory.THROUGHPUT),
            _change("queue_size", 0, "Avoid extra resident inference contexts.", PresetCategory.THROUGHPUT),
            _change("max_batch", 4, "Bound batch pressure on small VRAM.", PresetCategory.THROUGHPUT),
            _change("max_power", 32, "Cap at roughly 1024 square pixels.", PresetCategory.THROUGHPUT),
            _change("safety_on_gpu", False, "Keep the GPU free for inference.", PresetCategory.SAFETY),
            _change("high_performance_mode", False, "Avoid aggressive pop timing.", PresetCategory.THROUGHPUT),
            _change("moderate_performance_mode", False, "Avoid aggressive pop timing.", PresetCategory.THROUGHPUT),
            _change(
                "unload_models_from_vram_often",
                True,
                "Aggressively free VRAM between jobs.",
                PresetCategory.THROUGHPUT,
            ),
            _change("allow_post_processing", False, "Avoid upscaler/face-fix VRAM spikes.", PresetCategory.FEATURES),
            _change(
                "dedicated_post_processing",
                "off",
                "Do not keep a post-processing context resident.",
                PresetCategory.FEATURES,
                requires_restart=True,
            ),
            _change("allow_controlnet", False, "Avoid ControlNet memory overhead.", PresetCategory.FEATURES),
            _change("allow_sdxl_controlnet", False, "Avoid SDXL ControlNet memory overhead.", PresetCategory.FEATURES),
            _change("allow_lora", False, "Avoid ad-hoc LoRA overhead.", PresetCategory.DOWNLOADS),
            _change("load_large_models", False, "Exclude very large models.", PresetCategory.WORKLOAD),
            _change(
                "models_to_load",
                ["Deliberate"],
                "Serve a proven SD 1.5 model.",
                PresetCategory.WORKLOAD,
                requires_restart=True,
            ),
        ),
    ),
    ConfigPreset(
        preset_id="midrange_12_16gb_32gb_balanced",
        label="12-16 GB VRAM / 32 GB RAM - balanced",
        description="Moderate SD 1.5 + SDXL starting point for modern midrange cards.",
        changes=(
            _change("max_threads", 1, "Keep one active sampler.", PresetCategory.THROUGHPUT),
            _change("queue_size", 1, "Stage one follow-up job.", PresetCategory.THROUGHPUT),
            _change("max_batch", 4, "Moderate batch size.", PresetCategory.THROUGHPUT),
            _change("max_power", 50, "Offer larger than 1024-class jobs with a cap.", PresetCategory.THROUGHPUT),
            _change("safety_on_gpu", True, "Use GPU safety on 12 GB+ cards.", PresetCategory.SAFETY),
            _change(
                "moderate_performance_mode",
                True,
                "Pop somewhat faster on capable midrange cards.",
                PresetCategory.THROUGHPUT,
            ),
            _change("high_performance_mode", False, "Reserve high mode for larger cards.", PresetCategory.THROUGHPUT),
            _change("unload_models_from_vram_often", False, "Keep residency for reuse.", PresetCategory.THROUGHPUT),
            _change(
                "allow_post_processing", True, "Offer post-processing when budget allows.", PresetCategory.FEATURES
            ),
            _change(
                "dedicated_post_processing",
                "auto",
                "Run the lane only when needed.",
                PresetCategory.FEATURES,
                requires_restart=True,
            ),
            _change("allow_controlnet", False, "Leave heavier features opt-in.", PresetCategory.FEATURES),
            _change("allow_sdxl_controlnet", False, "Leave heavier features opt-in.", PresetCategory.FEATURES),
            _change(
                "allow_lora",
                True,
                "Offer LoRA jobs when a CivitAI token is configured.",
                PresetCategory.DOWNLOADS,
                default_selected=False,
            ),
            _change("load_large_models", False, "Exclude very large models.", PresetCategory.WORKLOAD),
            _change(
                "models_to_load",
                ["Deliberate", "AlbedoBase XL (SDXL)"],
                "Serve representative SD 1.5 and SDXL models.",
                PresetCategory.WORKLOAD,
                requires_restart=True,
            ),
        ),
    ),
)


def preset_by_id(preset_id: str) -> ConfigPreset:
    """Return a built-in preset by id, raising KeyError when unknown."""
    for preset in BUILT_IN_PRESETS:
        if preset.preset_id == preset_id:
            return preset
    raise KeyError(preset_id)


def diff_preset(preset: ConfigPreset, current: dict[str, Any]) -> list[PresetDiff]:
    """Compare a preset with current config state."""
    return [
        PresetDiff(
            change=change, current_value=current.get(change.key), changed=current.get(change.key) != change.value
        )
        for change in preset.changes
    ]
