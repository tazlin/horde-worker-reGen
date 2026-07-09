"""Import-light save-time validation for the TUI config editor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ConfigValidationSeverity(StrEnum):
    """How strongly a config validation finding should affect save."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ConfigValidationIssue:
    """One config validation finding tied to a field key."""

    field_key: str
    message: str
    severity: ConfigValidationSeverity


_META_PREFIXES = ("top", "bottom", "all")


def _bool(config: dict[str, Any], key: str) -> bool:
    """Return a config boolean, treating absent/None as false."""
    return bool(config.get(key))


def _int(config: dict[str, Any], key: str, default: int = 0) -> int:
    """Return a config integer, falling back when the value is not parseable."""
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _str(config: dict[str, Any], key: str) -> str:
    """Return a config string with whitespace trimmed."""
    return str(config.get(key) or "").strip()


def _list(config: dict[str, Any], key: str) -> list[str]:
    """Return a normalized list of strings from a list-like config value."""
    value = config.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _uses_meta_command(entries: list[str]) -> bool:
    """Whether any model entry uses a TOP/BOTTOM/ALL meta selector."""
    for entry in entries:
        lowered = entry.strip().lower()
        if any(lowered == prefix or lowered.startswith(prefix + " ") for prefix in _META_PREFIXES):
            return True
    return False


def validate_config_interlocks(config: dict[str, Any]) -> list[ConfigValidationIssue]:
    """Return config interlock errors and warnings for the raw TUI state."""
    issues: list[ConfigValidationIssue] = []

    def error(field_key: str, message: str) -> None:
        issues.append(ConfigValidationIssue(field_key, message, ConfigValidationSeverity.ERROR))

    def warning(field_key: str, message: str) -> None:
        issues.append(ConfigValidationIssue(field_key, message, ConfigValidationSeverity.WARNING))

    if _bool(config, "allow_painting") and not _bool(config, "allow_img2img"):
        error("allow_painting", "Allow inpainting requires Allow img2img.")

    if _bool(config, "allow_sdxl_controlnet") and not _bool(config, "allow_controlnet"):
        error("allow_sdxl_controlnet", "Allow SDXL ControlNet requires Allow ControlNet.")

    if _str(config, "dedicated_post_processing") == "off" and _bool(config, "allow_post_processing"):
        error("dedicated_post_processing", "Post-processing lane mode 'off' also disables offered post-processing.")

    civitai_token = _str(config, "civitai_api_token")
    if _bool(config, "allow_lora") and not civitai_token:
        error("allow_lora", "Offering LoRA jobs requires a CivitAI API token.")

    models_to_load = _list(config, "models_to_load")
    if _uses_meta_command(models_to_load) and not civitai_token:
        error("models_to_load", "TOP/BOTTOM/ALL model load rules require a CivitAI API token.")

    if not _bool(config, "dreamer") and not _bool(config, "alchemist"):
        error("dreamer", "Enable either Dreamer image generation or Alchemist; both off serves nothing.")

    if _bool(config, "extra_slow_worker"):
        if _bool(config, "high_performance_mode"):
            error("extra_slow_worker", "Extra slow worker forces High performance mode off.")
        if _bool(config, "moderate_performance_mode"):
            error("extra_slow_worker", "Extra slow worker forces Moderate performance mode off.")
        if _int(config, "queue_size") > 0:
            error("extra_slow_worker", "Extra slow worker forces Preload queue slots to 0.")
        if _int(config, "max_threads", 1) > 1:
            error("extra_slow_worker", "Extra slow worker forces Concurrent image jobs to 1.")
        if _int(config, "preload_timeout", 150) < 150:
            error("extra_slow_worker", "Extra slow worker forces Preload timeout to at least 150 seconds.")

    skipped = {entry.lower() for entry in _list(config, "models_to_skip")}
    duplicated = [entry for entry in models_to_load if entry.lower() in skipped]
    if duplicated:
        error("models_to_skip", f"Models to load also appears in Models to skip: {', '.join(duplicated)}.")

    if _bool(config, "gpu_sampling_lease_enabled") and _bool(config, "unload_models_from_vram_often"):
        warning(
            "gpu_sampling_lease_enabled",
            "GPU sampling lease is counterproductive with Unload VRAM often because there is no staged residency.",
        )

    slots = config.get("gpu_sampling_lease_slots")
    if _bool(config, "gpu_sampling_lease_enabled") and slots is not None:
        try:
            if int(slots) < _int(config, "max_threads", 1):
                warning(
                    "gpu_sampling_lease_slots",
                    "Sampling lease slots below concurrent jobs leaves configured concurrency unused.",
                )
        except (TypeError, ValueError):
            pass

    if not _bool(config, "enable_vram_budget"):
        budget_defaults = {
            "vram_reserve_mb": 2048,
            "ram_reserve_mb": 4096,
            "whole_card_exclusive_residency": True,
            "post_processing_fault_breaker_enabled": True,
        }
        for key, default in budget_defaults.items():
            if key in config and config.get(key) != default:
                warning(key, "This setting has no effect while Enable VRAM budget is off.")

    if not _bool(config, "alchemist"):
        alchemy_defaults = {
            "forms": [],
            "alchemy_caption_enabled": False,
            "alchemy_allow_concurrent": True,
            "alchemy_max_concurrency": 1,
            "alchemy_vram_headroom_mb": 2000,
            "alchemy_ram_headroom_mb": 2048,
        }
        for key, default in alchemy_defaults.items():
            if key in config and config.get(key) != default:
                warning(key, "This alchemy setting has no effect while Enable alchemist is off.")

    if _bool(config, "load_large_models") and _bool(config, "safety_on_gpu"):
        warning(
            "safety_on_gpu",
            "Very large models often need Safety on GPU disabled to free its CUDA context.",
        )

    return issues
