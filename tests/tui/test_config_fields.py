"""Tests for the expanded config-form field catalog and bounds-aware coercion."""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.jobs.alchemy_popper import DEFAULT_ALCHEMY_FORMS
from horde_worker_regen.tui.config_form import CONFIG_FIELDS, FieldKind, coerce_value

_BY_KEY = {field.key: field for field in CONFIG_FIELDS}


def test_core_fields_present_with_expected_kinds() -> None:
    """The model-list and multi-select fields use the dedicated kinds; core keys are present."""
    assert _BY_KEY["models_to_load"].kind is FieldKind.MODEL_LIST
    assert _BY_KEY["models_to_skip"].kind is FieldKind.MODEL_LIST
    assert _BY_KEY["forms"].kind is FieldKind.SELECT_MULTI
    assert _BY_KEY["dedicated_post_processing"].kind is FieldKind.SELECT
    for key in (
        "api_key",
        "horde_url",
        "dreamer_name",
        "max_threads",
        "queue_size",
        "max_batch",
        "max_power",
        "allow_lora",
        "civitai_api_token",
        "nsfw",
        "load_large_models",
        "alchemist",
        "cache_home",
        "comfy_smart_memory",
        "custom_models",
        "log_purge_max_age_days",
        "log_purge_max_total_gb",
        "enable_pipeline_disaggregation",
    ):
        assert key in _BY_KEY, key


def test_major_operator_labels_reflect_semantics() -> None:
    """The biggest levers use labels that describe what the worker actually does."""
    assert _BY_KEY["max_threads"].label == "Concurrent image jobs"
    assert _BY_KEY["queue_size"].label == "Preload queue slots"
    assert _BY_KEY["max_power"].label == "Max resolution"
    assert _BY_KEY["allow_post_processing"].label == "Offer post-processing jobs"
    assert _BY_KEY["dedicated_post_processing"].label == "Post-processing lane mode"
    assert _BY_KEY["allow_lora"].label == "Offer LoRA jobs"
    assert _BY_KEY["load_large_models"].label == "Include very large models in TOP/ALL"


def test_custom_models_yaml_validation() -> None:
    """The custom model editor accepts a list of mappings with required keys."""
    value = coerce_value(
        _BY_KEY["custom_models"],
        '- name: "my model"\n  baseline: "stable_diffusion"\n  filepath: "/models/my.safetensors"\n',
    )

    assert value == [{"name": "my model", "baseline": "stable_diffusion", "filepath": "/models/my.safetensors"}]
    with pytest.raises(ValueError, match="must include filepath"):
        coerce_value(_BY_KEY["custom_models"], '- name: "bad"\n  baseline: "stable_diffusion"\n')


def test_pipeline_disaggregation_field_is_hidden_advanced() -> None:
    """The dormant disaggregation option stays catalogued under Advanced but is not rendered."""
    field = _BY_KEY["enable_pipeline_disaggregation"]

    assert field.section == "Other"
    assert field.hidden is True


def test_alchemy_forms_include_worker_default_forms() -> None:
    """The TUI exposes every top-level alchemy form the worker offers by default."""
    assert set(DEFAULT_ALCHEMY_FORMS) <= set(_BY_KEY["forms"].choices)


def test_int_bounds_enforced() -> None:
    """Integer coercion enforces the field's min/max with clear messages."""
    threads = _BY_KEY["max_threads"]
    with pytest.raises(ValueError, match="at most 16"):
        coerce_value(threads, "20")
    with pytest.raises(ValueError, match="at least 1"):
        coerce_value(threads, "0")
    assert coerce_value(threads, "2") == 2


def test_list_kinds_accept_lists() -> None:
    """Model-list and multi-select fields coerce a list of values, trimming blanks."""
    assert coerce_value(_BY_KEY["forms"], ["caption", " nsfw "]) == ["caption", "nsfw"]
    assert coerce_value(_BY_KEY["models_to_load"], ["top 2", "Deliberate", ""]) == ["top 2", "Deliberate"]


def test_select_kind_rejects_unknown_choice() -> None:
    """Single-select fields only save one of their declared choices."""
    dedicated = _BY_KEY["dedicated_post_processing"]
    assert coerce_value(dedicated, "auto") == "auto"
    with pytest.raises(ValueError, match="auto, on, off"):
        coerce_value(dedicated, "sometimes")


def test_secret_fields_flagged() -> None:
    """Sensitive fields are marked secret so the editor masks them."""
    assert _BY_KEY["api_key"].secret
    assert _BY_KEY["civitai_api_token"].secret


def test_no_obsolete_or_scribe_fields() -> None:
    """Obsolete / unused-in-reGen / Scribe keys are intentionally excluded."""
    for key in ("dynamic_models", "scribe_name", "kai_url", "vram_to_leave_free", "disable_disk_cache"):
        assert key not in _BY_KEY, key
