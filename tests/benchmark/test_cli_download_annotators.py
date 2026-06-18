"""Tests for the controlnet-annotator awareness in the `horde-benchmark download` subcommand."""

from __future__ import annotations

from horde_worker_regen.benchmark.cli import _controlnet_annotator_row, _ladder_control_types
from horde_worker_regen.benchmark.ladder import LadderOptions, build_default_ladder


def test_ladder_control_types_lists_the_controlnet_sweep() -> None:
    """An sd15 feature ladder exposes its classic controlnet preprocessor sweep as distinct control types."""
    ladder = build_default_ladder(
        LadderOptions(
            tiers=["sd15"],
            include_concurrency=False,
            include_features=True,
            include_alchemy=False,
        ),
    )
    control_types = _ladder_control_types(ladder)
    assert "canny" in control_types
    assert "depth" in control_types


def test_no_controlnet_means_no_annotator_row() -> None:
    """A ladder with no controlnet level produces no synthetic annotator plan row."""
    assert _controlnet_annotator_row([]) is None


def test_controlnet_yields_an_annotator_plan_row() -> None:
    """A controlnet ladder adds one labelled annotator row (size is ROM, may be None on an older engine)."""
    row = _controlnet_annotator_row(["depth"])
    assert row is not None
    assert row.name == "ControlNet annotators"
    assert row.on_disk is False
