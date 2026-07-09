"""Tests for built-in TUI configuration presets."""

from __future__ import annotations

from horde_worker_regen.tui.config_presets import BUILT_IN_PRESETS, diff_preset, preset_by_id


def test_expected_builtin_presets_present() -> None:
    """The v1 curated hardware presets are registered by stable id."""
    ids = {preset.preset_id for preset in BUILT_IN_PRESETS}

    assert {
        "rtx4090_64gb_sdxl_balanced",
        "rtx4090_64gb_large_models",
        "rtx2080_32gb_sd15_safe",
        "midrange_12_16gb_32gb_balanced",
    } <= ids


def test_2080_preset_is_sd15_safe() -> None:
    """The 2080 preset keeps the workload conservative."""
    preset = preset_by_id("rtx2080_32gb_sd15_safe")
    values = {change.key: change.value for change in preset.changes}

    assert values["queue_size"] == 0
    assert values["max_power"] == 32
    assert values["allow_post_processing"] is False
    assert values["allow_lora"] is False
    assert values["models_to_load"] == ["Deliberate"]


def test_preset_diff_marks_unchanged_values() -> None:
    """The diff helper reports which preset values would actually change."""
    preset = preset_by_id("rtx2080_32gb_sd15_safe")
    diffs = diff_preset(preset, {"max_threads": 1, "queue_size": 2})
    by_key = {diff.change.key: diff for diff in diffs}

    assert by_key["max_threads"].changed is False
    assert by_key["queue_size"].changed is True
