"""Tests for import-light config interlock validation."""

from __future__ import annotations

from horde_worker_regen.tui.config_validation import ConfigValidationSeverity, validate_config_interlocks


def _messages(config: dict[str, object], severity: ConfigValidationSeverity) -> list[str]:
    """Return messages of one severity."""
    return [issue.message for issue in validate_config_interlocks(config) if issue.severity is severity]


def test_hard_feature_interlocks_block_save() -> None:
    """Clearly ineffective feature combinations are errors."""
    errors = _messages(
        {
            "allow_img2img": False,
            "allow_painting": True,
            "allow_controlnet": False,
            "allow_sdxl_controlnet": True,
            "allow_post_processing": True,
            "dedicated_post_processing": "off",
            "dreamer": True,
        },
        ConfigValidationSeverity.ERROR,
    )

    assert any("inpainting requires" in message for message in errors)
    assert any("SDXL ControlNet requires" in message for message in errors)
    assert any("lane mode 'off'" in message for message in errors)


def test_lora_and_meta_commands_require_civitai_token() -> None:
    """LoRA and TOP/ALL model selectors are blocked without a CivitAI token."""
    errors = _messages(
        {"allow_lora": True, "models_to_load": ["top 2"], "dreamer": True},
        ConfigValidationSeverity.ERROR,
    )

    assert any("LoRA jobs requires" in message for message in errors)
    assert any("model load rules require" in message for message in errors)


def test_extra_slow_conflicts_are_errors() -> None:
    """Extra-slow mode's forced clamps are surfaced before save."""
    errors = _messages(
        {
            "extra_slow_worker": True,
            "high_performance_mode": True,
            "moderate_performance_mode": True,
            "queue_size": 1,
            "max_threads": 2,
            "preload_timeout": 80,
            "dreamer": True,
        },
        ConfigValidationSeverity.ERROR,
    )

    assert len(errors) == 5


def test_subtle_interactions_warn_without_error() -> None:
    """Risky but valid combinations are warnings, not blockers."""
    issues = validate_config_interlocks(
        {
            "dreamer": True,
            "gpu_sampling_lease_enabled": True,
            "unload_models_from_vram_often": True,
            "gpu_sampling_lease_slots": 1,
            "max_threads": 2,
            "load_large_models": True,
            "safety_on_gpu": True,
        },
    )

    assert not [issue for issue in issues if issue.severity is ConfigValidationSeverity.ERROR]
    assert len([issue for issue in issues if issue.severity is ConfigValidationSeverity.WARNING]) == 3


def test_model_pool_pins_must_be_eligible_and_not_skipped() -> None:
    """A manual pin that cannot be advertised is blocked before save."""
    errors = _messages(
        {
            "dreamer": True,
            "model_pool_enabled": True,
            "model_pool_pinned": [{"name": "Pinned", "affinity": 1.0}],
            "models_to_load": ["Other"],
            "models_to_skip": ["Pinned"],
        },
        ConfigValidationSeverity.ERROR,
    )

    assert any("Models to skip" in message for message in errors)
    assert any("explicit Models to load" in message for message in errors)


def test_model_pool_configuration_contradictions_warn() -> None:
    """Valid but ineffective pool combinations are called out without blocking save."""
    warnings = _messages(
        {
            "dreamer": True,
            "max_throughput_mode": True,
            "model_pool_enabled": False,
            "model_pool_ranker_enabled": False,
            "model_pool_seats": 1,
            "model_pool_pinned": [{"name": "A"}, {"name": "B"}],
            "models_to_load": ["all"],
        },
        ConfigValidationSeverity.WARNING,
    )

    assert any("preset is on" in message for message in warnings)
    assert any("no effect" in message for message in warnings)
    assert any("2 pinned models for 1 seat" in message for message in warnings)


def test_enabled_pins_only_pool_without_pins_warns() -> None:
    """A pool with both seat sources disabled cannot populate a seat."""
    warnings = _messages(
        {"dreamer": True, "model_pool_enabled": True, "model_pool_ranker_enabled": False},
        ConfigValidationSeverity.WARNING,
    )

    assert any("nothing to seat" in message for message in warnings)
