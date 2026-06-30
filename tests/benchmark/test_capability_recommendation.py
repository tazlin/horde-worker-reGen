"""Recommendation provenance over the capability model: proof must be distinguishable from absence.

The capability-engine port of ``test_report_provenance``: it guards the same "consistent with reality"
contract (a capability left off because its probe was *skipped* must read differently from one that
*ran and failed*, and nothing is enabled on weaker than a proven pass), but builds
:class:`CapabilityProbeResult` objects keyed by capability kind rather than ``LevelReport`` objects.
"""

from __future__ import annotations

from horde_worker_regen.benchmark.capabilities.capability import CapabilityKind, CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.catalog import (
    CatalogOptions,
    build_capability_catalog,
    build_sustained_probe,
)
from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
from horde_worker_regen.benchmark.capabilities.recommendation import (
    synthesize_bridge_data,
    verify_suggestion_consistency,
)
from horde_worker_regen.benchmark.capabilities.result import (
    CapabilityProbeResult,
    CapabilityReport,
    Finding,
    SuggestedBridgeData,
    SuggestionBasis,
    SuggestionDecision,
)
from horde_worker_regen.benchmark.criteria import LevelStats
from horde_worker_regen.benchmark.enums import BenchTier, FindingKind
from horde_worker_regen.benchmark.ladder import BENCH_TIER_MODELS

_MODEL = BENCH_TIER_MODELS[BenchTier.SD15]


def _catalog(**options: object) -> list[CapabilityProbe]:
    """An sd15 capability catalog, with stage inclusion overridable per test."""
    return build_capability_catalog(CatalogOptions(tiers=[BenchTier.SD15], **options))  # type: ignore[arg-type]


def _results(
    probes: list[CapabilityProbe],
    verdicts: dict[CapabilityKind, CapabilityVerdict] | None = None,
    *,
    vram: int = 3000,
) -> list[CapabilityProbeResult]:
    """A probe result per catalog probe, defaulting to PROVEN but overriding by capability kind.

    A SKIPPED override carries a representative skip reason so the provenance detail is populated.
    """
    verdicts = verdicts or {}
    results: list[CapabilityProbeResult] = []
    for probe in probes:
        verdict = verdicts.get(probe.capability.kind, CapabilityVerdict.PROVEN)
        reasons = ["insufficient disk: needs 10 GB free"] if verdict is CapabilityVerdict.SKIPPED else []
        results.append(
            CapabilityProbeResult(
                capability=probe.capability,
                verdict=verdict,
                reasons=reasons,
                stats=LevelStats(its_p50=5.0, vram_used_high_water_mb=vram),
            ),
        )
    return results


def _decision(suggestion: SuggestedBridgeData, setting: str) -> SuggestionDecision:
    return next(decision for decision in suggestion.decisions if decision.setting == setting)


def test_skipped_lora_probe_is_untested_not_proven_off() -> None:
    """A skipped lora-download probe leaves allow_lora off, but on an 'untested' basis, not a failure."""
    catalog = _catalog(include_downloads=True)
    suggestion = synthesize_bridge_data(_results(catalog, {CapabilityKind.LORA_DOWNLOAD: CapabilityVerdict.SKIPPED}))
    decision = _decision(suggestion, "allow_lora")
    assert suggestion.allow_lora is False
    assert decision.basis == SuggestionBasis.UNTESTED_SKIPPED
    assert "disk" in decision.detail  # the skip reason is carried through


def test_failed_feature_probe_is_disabled_failed() -> None:
    """A post-processing probe that ran and failed yields a 'disabled: failed' basis (not 'untested')."""
    suggestion = synthesize_bridge_data(
        _results(_catalog(), {CapabilityKind.POST_PROCESSING: CapabilityVerdict.DISPROVEN})
    )
    decision = _decision(suggestion, "allow_post_processing")
    assert suggestion.allow_post_processing is False
    assert decision.basis == SuggestionBasis.DISABLED_FAILED


def test_passed_feature_probe_is_proven_on() -> None:
    """A passing post-processing probe proves the capability and turns it on."""
    suggestion = synthesize_bridge_data(_results(_catalog()))
    decision = _decision(suggestion, "allow_post_processing")
    assert suggestion.allow_post_processing is True
    assert decision.basis == SuggestionBasis.PROVEN


def test_capability_absent_from_catalog_is_not_in_ladder() -> None:
    """A capability whose probes were excluded (--no-features) reads as 'not in this run', not failed."""
    suggestion = synthesize_bridge_data(_results(_catalog(include_features=False)))
    decision = _decision(suggestion, "allow_post_processing")
    assert suggestion.allow_post_processing is False
    assert decision.basis == SuggestionBasis.NOT_IN_LADDER


def test_allow_controlnet_gated_on_classic_capability_only() -> None:
    """allow_controlnet follows the CONTROLNET capability alone; a passing QR-code probe must not enable it.

    The QR-code workflow proves SDXL controlnet, not classic preprocessor controlnet. Enabling
    allow_controlnet off a QR pass made the worker accept canny/depth/openpose jobs the CONTROLNET probe
    had shown to crash.
    """
    # CONTROLNET ran and failed; QR_CODE still passes (default).
    suggestion = synthesize_bridge_data(_results(_catalog(), {CapabilityKind.CONTROLNET: CapabilityVerdict.DISPROVEN}))
    assert suggestion.allow_controlnet is False
    assert _decision(suggestion, "allow_controlnet").basis == SuggestionBasis.DISABLED_FAILED


def _soak_results(
    verdict: CapabilityVerdict,
    *,
    baseline_vram: int = 3000,
    stats: LevelStats | None = None,
    findings: list[Finding] | None = None,
) -> list[CapabilityProbeResult]:
    """A passing sd15 baseline plus a SUSTAINED soak with the given verdict, stats, and findings."""
    baseline_results = _results(_catalog(), vram=baseline_vram)
    sustained = build_sustained_probe(synthesize_bridge_data(baseline_results), BenchTier.SD15, soak_seconds=60.0)
    reasons = ["soak pre-flight could not assemble the resident pool"] if verdict is CapabilityVerdict.SKIPPED else []
    return [
        *baseline_results,
        CapabilityProbeResult(
            capability=sustained.capability,
            verdict=verdict,
            reasons=reasons,
            stats=stats if stats is not None else LevelStats(),
            findings=findings or [],
        ),
    ]


def test_inconclusive_soak_keeps_the_model_with_a_caveat() -> None:
    """A soak that failed without proving load instability (0 jobs, no hard finding) keeps the model."""
    failed = synthesize_bridge_data(_soak_results(CapabilityVerdict.CRASHED), total_vram_mb=16000)
    assert _MODEL in failed.models_to_load
    assert any("inconclusive soak" in note for note in failed.notes)
    # The model's provenance still reads as proven (its baseline passed); the caveat lives in notes.
    assert _decision(failed, "models_to_load").basis == SuggestionBasis.PROVEN


def test_unstable_soak_that_ran_meaningfully_drops_the_model() -> None:
    """A soak that completed real work and then showed a hard finding is genuine load instability."""
    results = _soak_results(
        CapabilityVerdict.DISPROVEN,
        stats=LevelStats(num_jobs_completed=20),
        findings=[Finding(kind=FindingKind.PROCESS_RECOVERY, level_id="sd15-sustained", evidence="crash under load")],
    )
    suggestion = synthesize_bridge_data(results, total_vram_mb=16000)
    assert _MODEL not in suggestion.models_to_load
    assert any("ran and did not hold up" in note for note in suggestion.notes)


def test_skipped_soak_omits_the_model() -> None:
    """A soak that never ran omits the model, with wording distinct from a real failure."""
    skipped = synthesize_bridge_data(_soak_results(CapabilityVerdict.SKIPPED), total_vram_mb=16000)
    assert _MODEL not in skipped.models_to_load
    assert any("was skipped" in note for note in skipped.notes)
    assert not any("did not hold up" in note for note in skipped.notes)


def test_zero_activity_probe_reads_as_untested_not_failed() -> None:
    """An alchemy lane that timed out having dispatched no forms is untested, not 'tested and failed'."""
    catalog = _catalog(include_alchemy=True)
    alchemy_kinds = {CapabilityKind.ALCHEMY_CLIP, CapabilityKind.ALCHEMY_GRAPH, CapabilityKind.ALCHEMY_CONCURRENT}
    results = _results(catalog, dict.fromkeys(alchemy_kinds, CapabilityVerdict.DISPROVEN))
    # Every alchemy lane timed out having dispatched no forms (expected forms, zero completed/faulted).
    for result in results:
        if result.capability.kind in alchemy_kinds:
            result.stats = LevelStats(num_alchemy_forms_expected=8)
    suggestion = synthesize_bridge_data(results)
    decision = _decision(suggestion, "alchemist")
    assert suggestion.alchemist is False
    assert decision.basis == SuggestionBasis.UNTESTED_SKIPPED
    assert "never exercised" in decision.detail


def test_consistency_check_passes_for_a_grounded_recommendation() -> None:
    """A fully-passing run produces a recommendation with no consistency complaints."""
    results = _results(_catalog(include_downloads=True))
    report = CapabilityReport(
        probes=results, suggested_bridge_data=synthesize_bridge_data(results, total_vram_mb=16000)
    )
    assert verify_suggestion_consistency(report) == []


def test_every_enabled_flag_has_a_proven_basis() -> None:
    """Invariant: nothing the recommendation turns on rests on a skipped/failed/absent probe."""
    suggestion = synthesize_bridge_data(_results(_catalog(include_downloads=True)), total_vram_mb=16000)
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
    results = _results(_catalog(include_downloads=True), {CapabilityKind.LORA_DOWNLOAD: CapabilityVerdict.SKIPPED})
    suggestion = synthesize_bridge_data(results, total_vram_mb=16000)
    # Simulate a drift bug: the flag is on but its provenance says it was never tested.
    suggestion.allow_lora = True
    report = CapabilityReport(probes=results, suggested_bridge_data=suggestion)
    problems = verify_suggestion_consistency(report)
    assert any("allow_lora" in problem for problem in problems)


def test_unstable_soak_records_a_capped_soak_basis() -> None:
    """Concurrent alchemy downgraded by an unstable soak is attributed to the soak, not the capability."""
    results = _results(_catalog())
    sustained = build_sustained_probe(synthesize_bridge_data(results), BenchTier.SD15, soak_seconds=60.0)
    results.append(
        CapabilityProbeResult(
            capability=sustained.capability,
            verdict=CapabilityVerdict.DISPROVEN,
            stats=LevelStats(),
            findings=[Finding(kind=FindingKind.PROCESS_RECOVERY, level_id="sd15-sustained", evidence="crash")],
        ),
    )
    suggestion = synthesize_bridge_data(results, total_vram_mb=16000)
    assert suggestion.alchemy_allow_concurrent is False
    assert _decision(suggestion, "alchemy_allow_concurrent").basis == SuggestionBasis.CAPPED_SOAK
