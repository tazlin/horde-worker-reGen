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


def test_controlnet_basis_spans_both_workflows() -> None:
    """allow_controlnet is only DISABLED_FAILED when both the classic and qr_code axes are non-passing."""
    ladder = _ladder()
    failed_both = {BenchAxis.CONTROLNET: LevelOutcome.FAILED, BenchAxis.QR_CODE: LevelOutcome.FAILED}
    suggestion = synthesize_bridge_data(_reports(ladder, failed_both))
    assert suggestion.allow_controlnet is False
    assert _decision(suggestion, "allow_controlnet").basis == SuggestionBasis.DISABLED_FAILED


def _soak_report(outcome: LevelOutcome, *, baseline_vram: int = 3000) -> list[LevelReport]:
    """A passing sd15 baseline plus a stage-V soak with the given outcome."""
    baseline_reports = _reports(_ladder(), vram=baseline_vram)
    soak_level = build_validation_level(synthesize_bridge_data(baseline_reports), BenchTier.SD15, soak_seconds=60.0)
    reasons = ["soak pre-flight could not assemble the resident pool"] if outcome == LevelOutcome.SKIPPED else []
    return [*baseline_reports, LevelReport(level=soak_level, outcome=outcome, reasons=reasons, stats=LevelStats())]


def test_failed_soak_and_skipped_soak_read_differently() -> None:
    """A soak that failed and one that was skipped both omit the model, but with distinct wording."""
    failed = synthesize_bridge_data(_soak_report(LevelOutcome.FAILED), total_vram_mb=16000)
    skipped = synthesize_bridge_data(_soak_report(LevelOutcome.SKIPPED), total_vram_mb=16000)

    assert _MODEL not in failed.models_to_load
    assert _MODEL not in skipped.models_to_load
    assert any("ran and did not" in note for note in failed.notes)
    assert any("was skipped" in note for note in skipped.notes)
    # The misleading "soak did not pass" wording for a never-run soak is gone.
    assert not any("did not pass" in note for note in skipped.notes)


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
