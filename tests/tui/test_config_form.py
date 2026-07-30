"""Unit tests for the config-form YAML read/write and value coercion."""

from __future__ import annotations

from pathlib import Path

import pytest

from horde_worker_regen.tui.config_form import (
    CONFIG_FIELDS,
    CONFIG_SUBTABS,
    GPU_OVERRIDE_FIELDS,
    ConfigField,
    FieldKind,
    coerce_value,
    current_value,
    field_key_present,
    field_yaml_path,
    load_config,
    save_config,
    set_field_value,
)

_SAMPLE_YAML = """\
# a comment that must survive a round-trip
api_key: "secret123"
max_threads: 2
allow_lora: true
models_to_load:
  - "Deliberate"
  - "AlbedoBase XL (SDXL)"
"""


def _field(key: str, kind: FieldKind) -> ConfigField:
    return ConfigField(key=key, label=key, kind=kind, section="x")


def test_load_and_current_value(tmp_path: Path) -> None:
    """Existing keys are read back with their typed values; missing keys fall back to defaults."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text(_SAMPLE_YAML, encoding="utf-8")
    data = load_config(path)

    assert current_value(_field("max_threads", FieldKind.INT), data) == 2
    assert current_value(_field("allow_lora", FieldKind.BOOL), data) is True
    assert current_value(_field("models_to_load", FieldKind.STR_LIST), data) == [
        "Deliberate",
        "AlbedoBase XL (SDXL)",
    ]
    assert current_value(_field("queue_size", FieldKind.INT), data) == 0  # absent -> default


def test_coerce_value() -> None:
    """Coercion converts widget values, and rejects a non-numeric integer."""
    assert coerce_value(_field("max_threads", FieldKind.INT), "3") == 3
    assert coerce_value(_field("allow_lora", FieldKind.BOOL), True) is True
    assert coerce_value(_field("models_to_load", FieldKind.STR_LIST), "a\n b \n\nc") == ["a", "b", "c"]
    with pytest.raises(ValueError, match="whole number"):
        coerce_value(_field("max_threads", FieldKind.INT), "not-a-number")


def test_coerce_float_field() -> None:
    """A float field accepts fractions, normalises whole numbers to int, and rejects non-numbers."""
    field = ConfigField(key="min_lora_disk_free_gb", label="Min LoRA disk free", kind=FieldKind.FLOAT, section="x")
    assert coerce_value(field, "1.5") == 1.5
    # A whole number is written back as an int (2, not 2.0) to keep the YAML tidy.
    assert coerce_value(field, "2") == 2
    assert isinstance(coerce_value(field, "2"), int)
    with pytest.raises(ValueError, match="must be a number"):
        coerce_value(field, "not-a-number")


def test_coerce_float_field_respects_bounds() -> None:
    """Float bounds are enforced and reported without a spurious trailing ``.0``."""
    field = ConfigField(
        key="min_lora_disk_free_gb",
        label="Min LoRA disk free",
        kind=FieldKind.FLOAT,
        section="x",
        minimum=0,
        maximum=512,
    )
    assert coerce_value(field, "0") == 0
    with pytest.raises(ValueError, match="at most 512"):
        coerce_value(field, "513")


def test_float_default_round_trips_through_coercion() -> None:
    """A float field's absent-key default must itself coerce cleanly (guards the INT-vs-float trap).

    The original soft-lock was an INT-typed field whose float default (1.0) the integer coercion then
    rejected, blocking every save. Defaults must survive their own field's coercion.
    """
    field = ConfigField(
        key="min_lora_disk_free_gb",
        label="Min LoRA disk free",
        kind=FieldKind.FLOAT,
        section="x",
        minimum=0,
        explicit_default=1.0,
    )
    assert coerce_value(field, str(field.default())) == 1.0


def test_every_field_default_survives_its_own_coercion() -> None:
    """No catalogued field may have a default the same field's coercion rejects.

    This is the general guard against the INT-vs-float soft-lock: if a field's displayed default
    cannot be saved without edits, an operator with an absent key is trapped on an unsaveable form.
    """
    trapped: list[str] = []
    for field in CONFIG_FIELDS:
        default = field.default()
        raw: object = default if isinstance(default, (bool, list)) else str(default)
        try:
            coerce_value(field, raw)
        except ValueError as error:
            trapped.append(f"{field.key}: default {default!r} fails its own coercion ({error})")
    assert not trapped, "config fields whose default cannot be saved unedited:\n" + "\n".join(trapped)


def test_save_preserves_comments_and_untouched_keys(tmp_path: Path) -> None:
    """Saving a changed value keeps comments and unrelated keys intact."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text(_SAMPLE_YAML, encoding="utf-8")
    data = load_config(path)

    data["max_threads"] = 4
    save_config(data, path)

    written = path.read_text(encoding="utf-8")
    assert "# a comment that must survive a round-trip" in written
    assert 'api_key: "secret123"' in written

    reloaded = load_config(path)
    assert current_value(_field("max_threads", FieldKind.INT), reloaded) == 4


def test_load_missing_file_returns_empty_mapping(tmp_path: Path) -> None:
    """Loading an absent file yields an empty mapping rather than raising."""
    data = load_config(tmp_path / "does_not_exist.yaml")
    assert current_value(_field("max_threads", FieldKind.INT), data) == 0


def _nested_field(key: str, kind: FieldKind, **kwargs: object) -> ConfigField:
    return ConfigField(key=key, label=key, kind=kind, section="Model pool", yaml_parent="model_pool", **kwargs)  # type: ignore[arg-type]


def test_nested_field_yaml_path_derives_leaf() -> None:
    """A nested field's YAML path is (parent, leaf), with the leaf stripped of the parent prefix."""
    assert field_yaml_path(_nested_field("model_pool_enabled", FieldKind.BOOL)) == ("model_pool", "enabled")
    assert field_yaml_path(_field("max_threads", FieldKind.INT)) == ("max_threads",)


def test_nested_field_round_trips_under_parent(tmp_path: Path) -> None:
    """A nested field defaults when absent, writes under its parent, and survives a save/reload."""
    field = _nested_field("model_pool_enabled", FieldKind.BOOL)
    data = load_config(tmp_path / "absent.yaml")

    assert field_key_present(field, data) is False
    assert current_value(field, data) is False  # kind default when the whole block is absent

    set_field_value(field, data, True)
    assert field_key_present(field, data) is True
    assert current_value(field, data) is True

    path = tmp_path / "bridgeData.yaml"
    save_config(data, path)
    reloaded = load_config(path)
    assert current_value(field, reloaded) is True
    assert reloaded["model_pool"]["enabled"] is True


def test_nested_write_preserves_sibling_keys_and_comments(tmp_path: Path) -> None:
    """Writing one nested leaf keeps the parent's other keys and comments intact."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text(
        "model_pool:\n  enabled: true  # keep this comment\n  seats: 2\n",
        encoding="utf-8",
    )
    data = load_config(path)

    seats = _nested_field("model_pool_seats", FieldKind.INT, minimum=0, maximum=64)
    set_field_value(seats, data, coerce_value(seats, "4"))
    save_config(data, path)

    written = path.read_text(encoding="utf-8")
    assert "# keep this comment" in written

    reloaded = load_config(path)
    assert current_value(seats, reloaded) == 4
    assert current_value(_nested_field("model_pool_enabled", FieldKind.BOOL), reloaded) is True


def test_nested_presence_returns_false_for_non_mapping_parent() -> None:
    """A malformed scalar parent is reported absent instead of raising while the editor tries to save."""
    field = _nested_field("model_pool_enabled", FieldKind.BOOL)

    assert field_key_present(field, {"model_pool": "invalid"}) is False


def test_max_throughput_mode_shows_effective_absent_pool_values() -> None:
    """The raw editor reflects the runtime preset while preserving explicit nested overrides."""
    enabled = _nested_field("model_pool_enabled", FieldKind.BOOL)
    ranker = _nested_field("model_pool_ranker_enabled", FieldKind.BOOL, explicit_default=True)
    budget = _nested_field("model_pool_download_budget_gb", FieldKind.FLOAT, explicit_default=0.0)
    inherited = {"max_throughput_mode": True}

    assert current_value(enabled, inherited) is True
    assert current_value(ranker, inherited) is True
    assert current_value(budget, inherited) == 50.0

    explicit = {"max_throughput_mode": True, "model_pool": {"enabled": False, "download_budget_gb": 5.0}}
    assert current_value(enabled, explicit) is False
    assert current_value(budget, explicit) == 5.0


@pytest.mark.parametrize(
    "field_key",
    [
        "model_pool_rotation_minutes",
        "model_pool_rescue_eta_seconds",
        "model_pool_rescue_window_minutes",
    ],
)
def test_strictly_positive_model_pool_fields_reject_zero(field_key: str) -> None:
    """The editor enforces the runtime model's strict-positive bounds for pool timing fields."""
    field = next(candidate for candidate in CONFIG_FIELDS if candidate.key == field_key)

    with pytest.raises(ValueError, match="must be greater than 0"):
        coerce_value(field, "0")


def test_model_pool_section_is_configurable() -> None:
    """The editor exposes a Model pool sub-tab with the throughput switch and the nested pool scalars."""
    subtab_sections = {section for _label, sections in CONFIG_SUBTABS for section in sections}
    assert "Model pool" in subtab_sections

    pool_fields = [field for field in CONFIG_FIELDS if field.section == "Model pool"]
    keys = {field.key for field in pool_fields}
    assert "max_throughput_mode" in keys
    assert "model_pool_enabled" in keys
    assert "model_pool_download_budget_gb" in keys
    assert "model_pool_pinned" in keys
    assert next(field for field in pool_fields if field.key == "model_pool_pinned").hidden is False

    for field in pool_fields:
        # max_throughput_mode is a top-level flag; every other pool field nests under model_pool.
        if field.key == "max_throughput_mode":
            assert field.yaml_parent == ""
        else:
            assert field.yaml_parent == "model_pool"


def test_deprecated_field_help_names_the_pool() -> None:
    """The deprecated stickiness and per-card dynamic_models help must say so and point at the pool."""
    stickiness = next(field for field in CONFIG_FIELDS if field.key == "model_stickiness")
    assert "deprecated" in stickiness.help.lower()
    assert "model pool" in stickiness.help.lower()

    dynamic_models = next(field for field in GPU_OVERRIDE_FIELDS if field.key == "dynamic_models")
    assert "inert" in dynamic_models.help.lower()
    assert "model pool" in dynamic_models.help.lower()
