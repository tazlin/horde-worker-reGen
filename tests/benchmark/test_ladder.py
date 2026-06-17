"""Tests for the default ramp ladder construction."""

from __future__ import annotations

import pytest

from horde_worker_regen.benchmark.ladder import LadderOptions, build_default_ladder


class TestDefaultLadder:
    """Shape and ordering of the default ladder."""

    def test_default_tiers_and_stage_order(self) -> None:
        """Each tier's stage-A baseline comes before its B/C levels; D follows all tiers."""
        ladder = build_default_ladder()
        ids = [level.id for level in ladder]
        assert ids[0].startswith("A-sd15"), "the ladder must start at the most conservative level"

        for tier in ("sd15", "sdxl"):
            stage_a_index = next(i for i, level in enumerate(ladder) if level.tier == tier and level.stage == "A")
            dependent_indices = [
                i for i, level in enumerate(ladder) if level.tier == tier and level.stage in ("B", "C")
            ]
            assert all(stage_a_index < i for i in dependent_indices)

        alchemy_levels = [level for level in ladder if level.stage == "D"]
        # The two alchemy lanes (CLIP, graph) plus the concurrent-with-image rung, each its own axis.
        assert {level.axis for level in alchemy_levels} == {"alchemy_clip", "alchemy_graph", "alchemy_concurrent"}

    def test_stage_a_establishes_baseline(self) -> None:
        """Only stage-A levels establish tier baselines."""
        ladder = build_default_ladder()
        for level in ladder:
            assert level.establishes_tier_baseline == (level.stage == "A")

    def test_flux_not_included_by_default(self) -> None:
        """Flux is opt-in (large download and VRAM footprint)."""
        assert not any(level.tier == "flux" for level in build_default_ladder())

    def test_flux_opt_in(self) -> None:
        """Requesting flux adds its levels with the right hordelib baseline."""
        ladder = build_default_ladder(LadderOptions(tiers=["flux"], include_alchemy=False))
        assert ladder[0].tier == "flux"
        assert ladder[0].baseline_hordelib == "flux_1"

    def test_downloads_opt_in_and_marked_networked(self) -> None:
        """Download levels appear only on request and are marked as needing network."""
        assert not any(level.stage == "E" for level in build_default_ladder())
        ladder = build_default_ladder(LadderOptions(include_downloads=True))
        download_levels = [level for level in ladder if level.stage == "E"]
        assert len(download_levels) == 1
        assert download_levels[0].requires_network
        assert download_levels[0].scenario.image_jobs[0].lora_names

    def test_controlnet_only_for_sd_tiers(self) -> None:
        """Flux gets no controlnet level."""
        ladder = build_default_ladder(LadderOptions(tiers=["flux"]))
        assert not any(level.axis == "controlnet" for level in ladder)

    def test_excluded_axis_drops_only_that_axis(self) -> None:
        """Excluding a single axis removes its levels while leaving its stage siblings in place."""
        from horde_worker_regen.benchmark.enums import BenchAxis

        full = build_default_ladder(LadderOptions(tiers=["sd15"]))
        assert any(level.axis == "controlnet" for level in full)

        pruned = build_default_ladder(LadderOptions(tiers=["sd15"], excluded_axes={BenchAxis.CONTROLNET}))
        assert not any(level.axis == "controlnet" for level in pruned)
        # A sibling feature axis (post-processing) is untouched by excluding controlnet.
        assert any(level.axis == "post_processing" for level in pruned)

    def test_unknown_tier_rejected(self) -> None:
        """An unknown tier name is rejected when the options are validated (it is not a BenchTier)."""
        with pytest.raises(ValueError):  # noqa: PT011 - pydantic raises a plain ValidationError (a ValueError)
            LadderOptions(tiers=["sd99"])

    def test_level_ids_unique(self) -> None:
        """Level IDs are unique (they key result files on disk)."""
        ladder = build_default_ladder(LadderOptions(include_downloads=True))
        ids = [level.id for level in ladder]
        assert len(ids) == len(set(ids))

    def test_levels_serialize_round_trip(self) -> None:
        """Levels survive the JSON round trip used between controller and level runner."""
        from horde_worker_regen.benchmark.ladder import RampLevel

        for level in build_default_ladder(LadderOptions(include_downloads=True)):
            assert RampLevel.model_validate_json(level.model_dump_json()) == level
