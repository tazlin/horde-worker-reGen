"""Tests for the redesigned tier/feature matrix, sizing, and conservative recommendation.

These are pure/table tests (no GPU, no network): they assert the ladder is grounded in real worker
capabilities (controlnet vs the qr_code workflow, the full post-processor sweep, the two alchemy
lanes, the flux/qwen tiers) and that the recommendation is conservative and headroom-aware.
"""

from __future__ import annotations

from horde_sdk.generation_parameters.alchemy.consts import (
    KNOWN_CLIP_BLIP_TYPES,
    KNOWN_FACEFIXERS,
    KNOWN_UPSCALERS,
)

from horde_worker_regen.benchmark.criteria import LevelCriteria, LevelStats, evaluate_level
from horde_worker_regen.benchmark.enums import BenchAxis, BenchTier, FindingKind, LevelOutcome
from horde_worker_regen.benchmark.ladder import (
    BENCH_TIER_MODELS,
    BETA_TIERS,
    HUGE_TIERS,
    LadderOptions,
    RampLevel,
    build_default_ladder,
)
from horde_worker_regen.benchmark.report import (
    Finding,
    LevelReport,
    synthesize_bridge_data,
    synthesize_capabilities,
)
from horde_worker_regen.benchmark.sizing import max_post_processing_resolution
from horde_worker_regen.benchmark.soak import build_validation_level

_ALL_TIERS = [BenchTier.SD15, BenchTier.SDXL, BenchTier.FLUX, BenchTier.QWEN]


def _levels_on_axis(ladder: list[RampLevel], axis: BenchAxis) -> list[RampLevel]:
    return [level for level in ladder if level.axis == axis]


class TestTierMatrix:
    """The ladder offers the right tiers and tier metadata."""

    def test_all_tiers_build_with_baselines(self) -> None:
        """Every requested tier gets a stage-A baseline level."""
        ladder = build_default_ladder(LadderOptions(tiers=_ALL_TIERS))
        baseline_tiers = {level.tier for level in ladder if level.establishes_tier_baseline}
        assert baseline_tiers == set(_ALL_TIERS)

    def test_flux_and_qwen_are_huge_and_qwen_is_beta(self) -> None:
        """flux/qwen are flagged huge (warn + auto-skip) and qwen is sourced from the beta reference."""
        assert frozenset({BenchTier.FLUX, BenchTier.QWEN}) == HUGE_TIERS
        assert frozenset({BenchTier.QWEN}) == BETA_TIERS

    def test_flux_model_name_exists_in_reference_form(self) -> None:
        """The flux model name is the published compact checkpoint, not the old placeholder."""
        assert BENCH_TIER_MODELS[BenchTier.FLUX] == "Flux.1-Schnell fp8 (Compact)"


class TestControlnetVsQrCode:
    """Classic controlnet is SD1.5-only; the qr_code workflow is the SDXL controlnet capability."""

    def test_controlnet_axis_is_sd15_only(self) -> None:
        """Controlnet axis is sd15 only."""
        ladder = build_default_ladder(LadderOptions(tiers=_ALL_TIERS))
        controlnet_tiers = {level.tier for level in _levels_on_axis(ladder, BenchAxis.CONTROLNET)}
        assert controlnet_tiers == {BenchTier.SD15}

    def test_qr_code_runs_on_sd15_and_sdxl_only(self) -> None:
        """Qr code runs on sd15 and sdxl only."""
        ladder = build_default_ladder(LadderOptions(tiers=_ALL_TIERS))
        qr_tiers = {level.tier for level in _levels_on_axis(ladder, BenchAxis.QR_CODE)}
        assert qr_tiers == {BenchTier.SD15, BenchTier.SDXL}

    def test_qr_code_job_sets_workflow_and_sdxl_flag(self) -> None:
        """Qr code job sets workflow and sdxl flag."""
        ladder = build_default_ladder(LadderOptions(tiers=[BenchTier.SD15, BenchTier.SDXL]))
        for level in _levels_on_axis(ladder, BenchAxis.QR_CODE):
            job = level.scenario.expand_image_jobs()[0]
            assert job.payload.workflow == "qr_code"
            assert job.payload.control_type is None
            if level.tier is BenchTier.SDXL:
                assert level.bridge_data_overrides.get("allow_sdxl_controlnet") is True
            else:
                assert "allow_sdxl_controlnet" not in level.bridge_data_overrides


class TestPostProcessingSweep:
    """The post-processing sweep covers every upscaler/face-fixer and probes resolution scaling."""

    def test_sweep_covers_every_post_processor(self) -> None:
        """Sweep covers every post processor."""
        ladder = build_default_ladder(LadderOptions(tiers=[BenchTier.SD15]))
        sweep = next(level for level in ladder if level.id.endswith("pp-sweep"))
        swept = {pp for job in sweep.scenario.image_jobs for pp in job.post_processing}
        expected = {member.value for member in KNOWN_UPSCALERS if member is not KNOWN_UPSCALERS.BACKEND_DEFAULT}
        expected |= {member.value for member in KNOWN_FACEFIXERS if member is not KNOWN_FACEFIXERS.BACKEND_DEFAULT}
        assert swept == expected

    def test_resolutions_probe_includes_512_and_1024(self) -> None:
        """Resolutions probe includes 512 and 1024."""
        ladder = build_default_ladder(LadderOptions(tiers=[BenchTier.SDXL], total_vram_mb=24000))
        probe = next(level for level in ladder if level.id.endswith("pp-resolutions"))
        widths = sorted({job.width for job in probe.scenario.image_jobs})
        assert 512 in widths
        assert 1024 in widths
        assert max(widths) >= 1024  # the VRAM-derived ceiling is at least native


class TestAlchemyLanes:
    """Alchemy is exercised on both lanes, plus concurrently with image jobs."""

    def test_clip_lane_offers_caption_interrogation_nsfw(self) -> None:
        """Clip lane offers caption interrogation nsfw."""
        ladder = build_default_ladder(LadderOptions(tiers=[BenchTier.SD15]))
        clip = next(level for level in _levels_on_axis(ladder, BenchAxis.ALCHEMY_CLIP))
        forms = {form.form for form in clip.scenario.alchemy_forms}
        assert forms == {member.value for member in KNOWN_CLIP_BLIP_TYPES}
        assert clip.bridge_data_overrides.get("alchemy_caption_enabled") is True

    def test_graph_lane_includes_all_upscalers_and_facefixers(self) -> None:
        """Graph lane includes all upscalers and facefixers."""
        ladder = build_default_ladder(LadderOptions(tiers=[BenchTier.SD15]))
        graph = next(level for level in _levels_on_axis(ladder, BenchAxis.ALCHEMY_GRAPH))
        forms = {form.form for form in graph.scenario.alchemy_forms}
        post_processors = {member.value for member in KNOWN_UPSCALERS if member is not KNOWN_UPSCALERS.BACKEND_DEFAULT}
        post_processors |= {
            member.value for member in KNOWN_FACEFIXERS if member is not KNOWN_FACEFIXERS.BACKEND_DEFAULT
        }
        assert post_processors <= forms

    def test_concurrent_level_mixes_lanes_and_sets_concurrency(self) -> None:
        """Concurrent level mixes lanes and sets concurrency."""
        ladder = build_default_ladder(LadderOptions(tiers=[BenchTier.SD15]))
        concurrent = next(level for level in _levels_on_axis(ladder, BenchAxis.ALCHEMY_CONCURRENT))
        assert concurrent.scenario.total_image_jobs >= 1
        assert concurrent.scenario.total_alchemy_forms >= 1
        assert concurrent.bridge_data_overrides.get("alchemy_allow_concurrent") is True
        assert concurrent.bridge_data_overrides.get("alchemy_max_concurrency") == 2


class TestMaxPostProcessingResolution:
    """The post-processing resolution ceiling tracks VRAM and the burden registry."""

    def test_unknown_vram_falls_back_to_bounded_native_multiple(self) -> None:
        """Unknown vram falls back to bounded native multiple."""
        assert max_post_processing_resolution(baseline="stable_diffusion_1", total_vram_mb=None) <= 1024

    def test_more_vram_allows_larger_or_equal_resolution(self) -> None:
        """More vram allows larger or equal resolution."""
        small = max_post_processing_resolution(baseline="stable_diffusion_xl", total_vram_mb=12000)
        large = max_post_processing_resolution(baseline="stable_diffusion_xl", total_vram_mb=48000)
        assert large >= small


def _baseline_report(tier: BenchTier, *, peak_vram_mb: int | None, its: float = 5.0) -> LevelReport:
    """A passing stage-A baseline report for *tier* with a given peak VRAM."""
    level = next(
        level
        for level in build_default_ladder(LadderOptions(tiers=[tier]))
        if level.establishes_tier_baseline and level.tier == tier
    )
    return LevelReport(
        level=level,
        outcome=LevelOutcome.PASSED,
        stats=LevelStats(its_p50=its, vram_used_high_water_mb=peak_vram_mb),
    )


class TestConservativeRecommendation:
    """Synthesis keeps headroom, prefers stability, and folds in the soak outcome."""

    def test_model_excluded_when_no_vram_headroom(self) -> None:
        """A tier whose peak VRAM leaves under the reserve free is not recommended for loading."""
        # Peak 15800 MB of 16000 MB leaves 200 MB, far under the 1500 MB reserve.
        reports = [_baseline_report(BenchTier.SD15, peak_vram_mb=15800)]
        suggestion = synthesize_bridge_data(reports, total_vram_mb=16000)
        assert suggestion.models_to_load == []
        assert any("headroom" in note for note in suggestion.notes)

    def test_model_kept_when_headroom_is_ample(self) -> None:
        """Model kept when headroom is ample."""
        reports = [_baseline_report(BenchTier.SD15, peak_vram_mb=3000)]
        suggestion = synthesize_bridge_data(reports, total_vram_mb=16000)
        assert suggestion.models_to_load == [BENCH_TIER_MODELS[BenchTier.SD15]]

    def test_failed_soak_drops_the_tier_model(self) -> None:
        """A tier whose stage-V soak failed is not recommended for loading."""
        suggestion_input = synthesize_bridge_data([_baseline_report(BenchTier.SD15, peak_vram_mb=3000)])
        soak_level = build_validation_level(suggestion_input, BenchTier.SD15, soak_seconds=60.0)
        reports = [
            _baseline_report(BenchTier.SD15, peak_vram_mb=3000),
            LevelReport(level=soak_level, outcome=LevelOutcome.FAILED, stats=LevelStats()),
        ]
        suggestion = synthesize_bridge_data(reports, total_vram_mb=16000)
        assert BENCH_TIER_MODELS[BenchTier.SD15] not in suggestion.models_to_load
        assert any("soak" in note for note in suggestion.notes)

    def test_unstable_soak_disables_concurrent_alchemy(self) -> None:
        """A hard soak finding downgrades the recommendation by disabling concurrent alchemy."""
        ladder = build_default_ladder(LadderOptions(tiers=[BenchTier.SD15]))
        reports: list[LevelReport] = [
            LevelReport(
                level=level, outcome=LevelOutcome.PASSED, stats=LevelStats(its_p50=5.0, vram_used_high_water_mb=3000)
            )
            for level in ladder
        ]
        soak_level = build_validation_level(synthesize_bridge_data(reports), BenchTier.SD15, soak_seconds=60.0)
        reports.append(
            LevelReport(
                level=soak_level,
                outcome=LevelOutcome.FAILED,
                stats=LevelStats(),
                findings=[Finding(kind=FindingKind.PROCESS_RECOVERY, level_id=soak_level.id, evidence="crash")],
            ),
        )
        suggestion = synthesize_bridge_data(reports, total_vram_mb=16000)
        assert suggestion.alchemy_allow_concurrent is False
        assert any("oncurrent alchemy disabled" in note for note in suggestion.notes)

    def test_max_batch_is_largest_clean_pass(self) -> None:
        """max_batch ignores a batch level that passed but carried a robustness finding."""
        ladder = build_default_ladder(LadderOptions(tiers=[BenchTier.SD15]))
        reports: list[LevelReport] = []
        for level in ladder:
            findings = []
            # batch4 "passes" but with a finding -> must not be chosen as the stable batch.
            if level.axis == BenchAxis.BATCH and level.rung == 2:
                findings = [Finding(kind=FindingKind.OOM, level_id=level.id, evidence="oom")]
            reports.append(
                LevelReport(level=level, outcome=LevelOutcome.PASSED, stats=LevelStats(), findings=findings)
            )
        suggestion = synthesize_bridge_data(reports)
        assert suggestion.max_batch == 2

    def test_capabilities_report_qr_and_lanes(self) -> None:
        """Capabilities reflect every axis that passed, independent of the recommendation."""
        ladder = build_default_ladder(LadderOptions(tiers=[BenchTier.SD15, BenchTier.SDXL], include_downloads=True))
        reports = [
            LevelReport(
                level=level, outcome=LevelOutcome.PASSED, stats=LevelStats(its_p50=5.0, vram_used_high_water_mb=3000)
            )
            for level in ladder
        ]
        caps = synthesize_capabilities(reports, total_vram_mb=16000)
        assert caps.supports_qr_code
        assert caps.supports_controlnet
        assert caps.supports_alchemy_clip
        assert caps.supports_alchemy_graph
        assert caps.supports_post_processing
        assert {tier_cap.tier for tier_cap in caps.tiers} >= {BenchTier.SD15, BenchTier.SDXL}


class TestAlchemyCompletionGate:
    """An alchemy-only level must actually complete its forms to pass."""

    def test_incomplete_alchemy_forms_fail_the_level(self) -> None:
        """Incomplete alchemy forms fail the level."""
        stats = LevelStats(num_alchemy_forms_expected=3, num_alchemy_forms_completed=1)
        verdict = evaluate_level(stats, LevelCriteria(max_faulted_alchemy_forms=0))
        assert not verdict.passed
        assert any("alchemy forms completed" in reason for reason in verdict.reasons)

    def test_complete_alchemy_forms_pass(self) -> None:
        """Complete alchemy forms pass."""
        stats = LevelStats(num_alchemy_forms_expected=3, num_alchemy_forms_completed=3)
        verdict = evaluate_level(stats, LevelCriteria(max_faulted_alchemy_forms=0))
        assert verdict.passed
