"""Unit tests for meta model-load instruction building and classification."""

from __future__ import annotations

from horde_worker_regen.tui.model_catalog import (
    META_OPTIONS,
    META_OPTIONS_BY_KIND,
    MetaKind,
    build_meta_instruction,
    describe_entry,
    is_meta_instruction,
)


def test_build_meta_instruction() -> None:
    """Building produces the bridgeData strings the worker recognises."""
    assert build_meta_instruction(MetaKind.TOP_N, 5) == "top 5"
    assert build_meta_instruction(MetaKind.BOTTOM_N, 3) == "bottom 3"
    assert build_meta_instruction(MetaKind.ALL) == "ALL MODELS"
    assert build_meta_instruction(MetaKind.ALL_SDXL) == "ALL SDXL MODELS"
    assert build_meta_instruction(MetaKind.ALL_SFW) == "ALL SFW MODELS"


def test_build_meta_instruction_clamps_count() -> None:
    """A non-positive count is clamped to 1."""
    assert build_meta_instruction(MetaKind.TOP_N, 0) == "top 1"


def test_is_meta_instruction_recognises_commands() -> None:
    """All documented meta forms are recognised, and plain model names are not."""
    for command in [
        "top 5",
        "TOP 10",
        "bottom 3",
        "all",
        "ALL MODELS",
        "all sdxl",
        "ALL SD15 MODELS",
        "all sfw models",
        "all nsfw",
        "all inpainting models",
    ]:
        assert is_meta_instruction(command), command
    for name in ["Deliberate", "AlbedoBase XL (SDXL)", "stable_diffusion", "topmodel", "all of the above"]:
        assert not is_meta_instruction(name), name


def test_built_instructions_are_recognised() -> None:
    """Every builder output round-trips through the classifier."""
    for option in META_OPTIONS:
        built = build_meta_instruction(option.kind, 3)
        assert is_meta_instruction(built), built


def test_describe_entry_annotates_meta() -> None:
    """describe_entry marks meta instructions and leaves plain names alone."""
    assert "(meta)" in describe_entry("top 5")
    assert describe_entry("Deliberate") == "Deliberate"


def test_all_kinds_have_options() -> None:
    """Every MetaKind is offered by the builder."""
    assert set(META_OPTIONS_BY_KIND) == set(MetaKind)
