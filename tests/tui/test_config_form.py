"""Unit tests for the config-form YAML read/write and value coercion."""

from __future__ import annotations

from pathlib import Path

import pytest

from horde_worker_regen.tui.config_form import (
    ConfigField,
    FieldKind,
    coerce_value,
    current_value,
    load_config,
    save_config,
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
