"""Guard: the config editor's absent-key defaults must match the worker's real model defaults.

The TUI catalog (``config_form.CONFIG_FIELDS``) is deliberately import-light and does not load
``reGenBridgeData`` at runtime, so its per-field "value when the key is absent" can silently drift
from what the worker actually uses when the key is omitted from ``bridgeData.yaml``. That drift is a
real operator trap: the editor would show ``unload_models_from_vram_often: False`` (or
``nsfw: False``) while the worker defaults the field to ``True``. This test is the single enforcement
point that keeps the two in sync; it is allowed to import the heavy model because tests are not the
import-light TUI parent.
"""

from __future__ import annotations

from typing import Any

from pydantic_core import PydanticUndefined

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.tui.config_form import (
    ALCHEMIST_NAME_RESERVED_DEFAULT,
    CONFIG_FIELDS,
    DREAMER_NAME_RESERVED_DEFAULT,
    FieldKind,
    validate_identity_names,
)

# Keys whose editor display is intentionally blank to prompt the operator to set their own value,
# even though the model carries a placeholder/default. Aligning these to the SDK placeholder
# ("An Awesome Dreamer", "0000000000", "./") would be worse UX, so they are exempt from parity.
_INTENTIONALLY_BLANK = frozenset(
    {
        "api_key",
        "civitai_api_token",
        "dreamer_name",
        "alchemist_name",
        "cache_home",
        # These fields have model default=None (meaning "unset"); the editor shows a blank/zero
        # placeholder that the operator is expected to fill in before enabling the feature.
        "kudos_training_data_file",
        "download_rate_limit_kbps",
    }
)


def _model_default_by_field_key() -> dict[str, Any]:
    """Map every model field name and alias to its concrete default (skipping required fields)."""
    defaults: dict[str, Any] = {}
    for name, field in reGenBridgeData.model_fields.items():
        default = field.get_default()
        if default is PydanticUndefined:
            continue
        defaults[name] = default
        if field.alias:
            defaults[field.alias] = default
    return defaults


def test_editor_defaults_match_worker_defaults() -> None:
    """Every editor field with a concrete model default shows that exact default when the key is absent."""
    model_defaults = _model_default_by_field_key()
    mismatches: list[str] = []

    for field in CONFIG_FIELDS:
        if field.key in _INTENTIONALLY_BLANK:
            continue
        if field.key not in model_defaults:
            continue
        # List-kind fields normalise an absent value to [] regardless of model shape; the editor's
        # empty-list display is the correct "nothing configured" state, so do not compare them.
        if field.kind in (FieldKind.STR_LIST, FieldKind.MODEL_LIST, FieldKind.SELECT_MULTI):
            continue

        expected = model_defaults[field.key]
        actual = field.default()
        if actual != expected:
            mismatches.append(
                f"{field.key}: editor shows {actual!r} but worker defaults to {expected!r} "
                f"(add explicit_default={expected!r} to its ConfigField)",
            )

    assert not mismatches, "config editor defaults drift from reGenBridgeData:\n" + "\n".join(mismatches)


def test_reserved_name_constants_match_model_defaults() -> None:
    """The editor's hardcoded reserved-name placeholders must equal the model's real field defaults.

    The config editor is import-light and cannot load reGenBridgeData at runtime, so it carries the
    reserved placeholder names as literals; this guard keeps them from drifting from the template the
    horde actually rejects.
    """
    assert reGenBridgeData.model_fields["dreamer_worker_name"].default == DREAMER_NAME_RESERVED_DEFAULT
    assert reGenBridgeData.model_fields["alchemist_name"].default == ALCHEMIST_NAME_RESERVED_DEFAULT


def test_validate_identity_names_rules() -> None:
    """The identity-name validator flags blank/default/colliding names and accepts valid ones."""
    # Valid: unique dreamer, alchemy off (alchemist name not checked).
    assert validate_identity_names("My Dreamer", alchemist_enabled=False, alchemist_name="") == []

    blank = validate_identity_names("  ", alchemist_enabled=False, alchemist_name="")
    assert [key for key, _ in blank] == ["dreamer_name"]

    default = validate_identity_names(DREAMER_NAME_RESERVED_DEFAULT, alchemist_enabled=False, alchemist_name="")
    assert [key for key, _ in default] == ["dreamer_name"]

    # Alchemy on: a blank alchemist name is required, the default is rejected, and it must differ.
    assert [key for key, _ in validate_identity_names("D", alchemist_enabled=True, alchemist_name="")] == [
        "alchemist_name",
    ]
    assert [
        key
        for key, _ in validate_identity_names(
            "D", alchemist_enabled=True, alchemist_name=ALCHEMIST_NAME_RESERVED_DEFAULT
        )
    ] == ["alchemist_name"]
    assert [
        key for key, _ in validate_identity_names("Dreamer", alchemist_enabled=True, alchemist_name="dreamer")
    ] == [
        "alchemist_name",
    ]
    assert validate_identity_names("Dreamer", alchemist_enabled=True, alchemist_name="Alchemist") == []
