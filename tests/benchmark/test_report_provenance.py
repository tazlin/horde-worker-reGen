"""Tests for recommendation provenance: a suggested setting must tell proof from untested absence.

These guard the "consistent with reality" contract: a capability left off because its level was
*skipped* must be reported differently from one that *ran and failed*, and the recommendation must
never enable something on anything weaker than a real pass.
"""

from __future__ import annotations

from horde_worker_regen.benchmark.criteria import LevelStats
from horde_worker_regen.benchmark.enums import BenchAxis, BenchTier, FindingKind, LevelOutcome
from horde_worker_regen.benchmark.ladder import BENCH_TIER_MODELS, LadderOptions, RampLevel, build_default_ladder
from horde_worker_regen.benchmark.report import (
    BenchmarkReport,
    Finding,
    LevelReport,
    SuggestedBridgeData,
    SuggestionBasis,
    SuggestionDecision,
    synthesize_bridge_data,
    verify_suggestion_consistency,
)
from horde_worker_regen.benchmark.soak import build_validation_level

_MODEL = BENCH_TIER_MODELS[BenchTier.SD15]


def _ladder(**options: object) -> list[RampLevel]:
    """An sd15 ladder, with stage inclusion overridable per test."""
    return build_default_ladder(LadderOptions(tiers=[BenchTier.SD15], **options))  # type: ignore[arg-type]


def _reports(
    ladder: list[RampLevel],
    outcomes: dict[BenchAxis, LevelOutcome] | None = None,
    *,
    vram: int = 3000,
) -> list[LevelReport]:
    """Build a level report per ladder level, defaulting to PASSED but overriding by axis.

    A SKIPPED override carries a representative skip reason so the provenance detail is populated.
    """
    outcomes = outcomes or {}
    reports: list[LevelReport] = []
    for level in ladder:
        outcome = outcomes.get(level.axis, LevelOutcome.PASSED)
        reasons = ["insufficient disk: needs 10 GB free"] if outcome == LevelOutcome.SKIPPED else []
        reports.append(
            LevelReport(
                level=level,
                outcome=outcome,
                reasons=reasons,
                stats=LevelStats(its_p50=5.0, vram_used_high_water_mb=vram),
            ),
        )
    return reports


def _decision(suggestion: SuggestedBridgeData, setting: str) -> SuggestionDecision:
    return next(decision for decision in suggestion.decisions if decision.setting == setting)


def test_skipped_lora_level_is_untested_not_proven_off() -> None:
    """A skipped DOWNLOADS level leaves allow_lora off, but on an 'untested' basis, not a failure."""
    ladder = _ladder(include_downloads=True)
    suggestion = synthesize_bridge_data(_reports(ladder, {BenchAxis.DOWNLOADS: LevelOutcome.SKIPPED}))
    decision = _decision(suggestion, "allow_lora")
    assert suggestion.allow_lora is False
    assert decision.basis == SuggestionBasis.UNTESTED_SKIPPED
    assert "disk" in decision.detail  # the skip reason is carried through


def test_failed_feature_level_is_disabled_failed() -> None:
    """A post-processing level that ran and failed yields a 'disabled: failed' basis (not 'untested')."""
    ladder = _ladder()
    suggestion = synthesize_bridge_data(_reports(ladder, {BenchAxis.POST_PROCESSING: LevelOutcome.FAILED}))
    decision = _decision(suggestion, "allow_post_processing")
    assert suggestion.allow_post_processing is False
    assert decision.basis == SuggestionBasis.DISABLED_FAILED


def test_passed_feature_level_is_proven_on() -> None:
    """A passing post-processing level proves the capability and turns it on."""
    suggestion = synthesize_bridge_data(_reports(_ladder()))
    decision = _decision(suggestion, "allow_post_processing")
    assert suggestion.allow_post_processing is True
    assert decision.basis == SuggestionBasis.PROVEN


def test_axis_absent_from_ladder_is_not_in_ladder() -> None:
    """A capability whose axis was excluded (--no-features) reads as 'not in this run', not failed."""
    ladder = _ladder(include_features=False)
    suggestion = synthesize_bridge_data(_reports(ladder))
    decision = _decision(suggestion, "allow_post_processing")
    assert suggestion.allow_post_processing is False
    assert decision.basis == SuggestionBasis.NOT_IN_LADDER


def test_allow_controlnet_gated_on_classic_axis_only() -> None:
    """allow_controlnet follows the CONTROLNET axis alone; a passing QR-code level must not enable it.

    The QR-code workflow proves SDXL controlnet, not classic preprocessor controlnet. Enabling
    allow_controlnet off a QR pass made the worker accept canny/depth/openpose jobs that the
    CONTROLNET level had shown to crash (the 4090 regression: allow_controlnet on while the
    capability table said controlnet was unsupported).
    """
    ladder = _ladder()
    # CONTROLNET ran and failed; QR_CODE still passes (default).
    suggestion = synthesize_bridge_data(_reports(ladder, {BenchAxis.CONTROLNET: LevelOutcome.FAILED}))
    assert suggestion.allow_controlnet is False
    assert _decision(suggestion, "allow_controlnet").basis == SuggestionBasis.DISABLED_FAILED


def _soak_report(
    outcome: LevelOutcome,
    *,
    baseline_vram: int = 3000,
    stats: LevelStats | None = None,
    findings: list[Finding] | None = None,
) -> list[LevelReport]:
    """A passing sd15 baseline plus a stage-V soak with the given outcome, stats, and findings."""
    baseline_reports = _reports(_ladder(), vram=baseline_vram)
    soak_level = build_validation_level(synthesize_bridge_data(baseline_reports), BenchTier.SD15, soak_seconds=60.0)
    reasons = ["soak pre-flight could not assemble the resident pool"] if outcome == LevelOutcome.SKIPPED else []
    return [
        *baseline_reports,
        LevelReport(
            level=soak_level,
            outcome=outcome,
            reasons=reasons,
            stats=stats if stats is not None else LevelStats(),
            findings=findings or [],
        ),
    ]


def test_inconclusive_soak_keeps_the_model_with_a_caveat() -> None:
    """A soak that failed without proving load instability (0 jobs, no hard finding) keeps the model.

    The isolated baseline passed and fits with headroom, so a soak that merely could-not-run (a
    startup/supervisor wedge) must not zero out models_to_load — that was the 4090 regression.
    """
    failed = synthesize_bridge_data(_soak_report(LevelOutcome.CRASHED_HANG), total_vram_mb=16000)
    assert _MODEL in failed.models_to_load
    assert any("inconclusive soak" in note for note in failed.notes)
    # The model's provenance still reads as proven (its baseline passed); the caveat lives in notes.
    assert _decision(failed, "models_to_load").basis == SuggestionBasis.PROVEN


def test_unstable_soak_that_ran_meaningfully_drops_the_model() -> None:
    """A soak that completed real work and then showed a hard finding is genuine load instability."""
    reports = _soak_report(
        LevelOutcome.FAILED,
        stats=LevelStats(num_jobs_completed=20),
        findings=[Finding(kind=FindingKind.PROCESS_RECOVERY, level_id="V-sd15-soak", evidence="crash under load")],
    )
    suggestion = synthesize_bridge_data(reports, total_vram_mb=16000)
    assert _MODEL not in suggestion.models_to_load
    assert any("ran and did not hold up" in note for note in suggestion.notes)


def test_skipped_soak_omits_the_model() -> None:
    """A soak that never ran omits the model, with wording distinct from a real failure."""
    skipped = synthesize_bridge_data(_soak_report(LevelOutcome.SKIPPED), total_vram_mb=16000)
    assert _MODEL not in skipped.models_to_load
    assert any("was skipped" in note for note in skipped.notes)
    assert not any("did not pass" in note for note in skipped.notes)


def test_zero_activity_level_reads_as_untested_not_failed() -> None:
    """An alchemy lane that timed out having dispatched no forms is untested, not 'tested and failed'."""
    ladder = _ladder(include_alchemy=True)
    alchemy_axes = {BenchAxis.ALCHEMY_CLIP, BenchAxis.ALCHEMY_GRAPH, BenchAxis.ALCHEMY_CONCURRENT}
    reports = _reports(ladder, dict.fromkeys(alchemy_axes, LevelOutcome.FAILED))
    # Every alchemy lane timed out having dispatched no forms (expected forms, zero completed/faulted).
    for report in reports:
        if report.level.axis in alchemy_axes:
            report.stats = LevelStats(num_alchemy_forms_expected=8)
    suggestion = synthesize_bridge_data(reports)
    decision = _decision(suggestion, "alchemist")
    assert suggestion.alchemist is False
    assert decision.basis == SuggestionBasis.UNTESTED_SKIPPED
    assert "never exercised" in decision.detail


def test_consistency_check_passes_for_a_grounded_recommendation() -> None:
    """A fully-passing run produces a recommendation with no consistency complaints."""
    reports = _reports(_ladder(include_downloads=True))
    report = BenchmarkReport(
        levels=reports, suggested_bridge_data=synthesize_bridge_data(reports, total_vram_mb=16000)
    )
    assert verify_suggestion_consistency(report) == []


def test_every_enabled_flag_has_a_proven_basis() -> None:
    """Invariant: nothing the recommendation turns on rests on a skipped/failed/absent level."""
    reports = _reports(_ladder(include_downloads=True))
    suggestion = synthesize_bridge_data(reports, total_vram_mb=16000)
    by_setting = {decision.setting: decision for decision in suggestion.decisions}
    enabled = {
        "allow_controlnet": suggestion.allow_controlnet,
        "allow_post_processing": suggestion.allow_post_processing,
        "allow_lora": suggestion.allow_lora,
        "alchemist": suggestion.alchemist,
        "alchemy_allow_concurrent": suggestion.alchemy_allow_concurrent,
    }
    for setting, value in enabled.items():
        if value:
            assert by_setting[setting].basis == SuggestionBasis.PROVEN, setting


def test_consistency_check_flags_an_ungrounded_enable() -> None:
    """A hand-built recommendation that enables a flag on a skipped basis is flagged."""
    reports = _reports(_ladder(include_downloads=True), {BenchAxis.DOWNLOADS: LevelOutcome.SKIPPED})
    suggestion = synthesize_bridge_data(reports, total_vram_mb=16000)
    # Simulate a drift bug: the flag is on but its provenance says it was never tested.
    suggestion.allow_lora = True
    report = BenchmarkReport(levels=reports, suggested_bridge_data=suggestion)
    problems = verify_suggestion_consistency(report)
    assert any("allow_lora" in problem for problem in problems)


def test_unstable_soak_records_a_capped_soak_basis() -> None:
    """Concurrent alchemy downgraded by an unstable soak is attributed to the soak, not the axis."""
    ladder = _ladder()
    reports = _reports(ladder)
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
    assert _decision(suggestion, "alchemy_allow_concurrent").basis == SuggestionBasis.CAPPED_SOAK
