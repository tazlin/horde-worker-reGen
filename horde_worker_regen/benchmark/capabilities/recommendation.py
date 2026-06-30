"""Synthesize the capability surface and the conservative recommendation from a run's probe results.

This is the capability-engine replacement for the synthesis half of the legacy ``report.py``: the same
two outputs derived from a run, but read off :class:`CapabilityProbeResult` (capability + verdict)
instead of the old ``LevelReport`` (axis + outcome). It carries two distinct things:

- a :class:`WorkerCapabilities` summary of *everything the worker proved it can do* (every tier and
  feature whose probe passed in isolation), and
- a :class:`SuggestedBridgeData` *conservative recommendation* of what to actually turn on by default,
  which keeps VRAM/disk headroom, prefers stability over peak throughput, and is downgraded when the
  sustained-load soak shows the combined config does not hold up.

Separating the two means an operator sees the full capability surface without being pushed into an
aggressive default that only just fit or only just held together. Pure and torch-free: it reads only
the result models plus the (torch-free) per-tier model table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from horde_worker_regen.benchmark.capabilities.capability import CapabilityKind, CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.result import (
    SuggestedBridgeData,
    SuggestionBasis,
    SuggestionDecision,
    TierCapability,
    WorkerCapabilities,
)
from horde_worker_regen.benchmark.enums import BenchTier, FindingKind
from horde_worker_regen.benchmark.ladder import BENCH_TIER_MODELS

if TYPE_CHECKING:
    from horde_worker_regen.benchmark.capabilities.result import CapabilityProbeResult, CapabilityReport

_FAILED_VERDICTS = frozenset({CapabilityVerdict.DISPROVEN, CapabilityVerdict.CRASHED})
"""Verdicts that mean a probe *ran and did not prove* its capability (as opposed to never running)."""

_HARD_SOAK_FINDINGS = frozenset(
    {FindingKind.OOM, FindingKind.HANG, FindingKind.CRASH, FindingKind.PROCESS_RECOVERY, FindingKind.LOST_JOB},
)
"""Findings that make a soak failure genuine sustained-load instability rather than a soft/startup one."""

_MIN_VRAM_HEADROOM_MB = 1500
"""A tier's model is only recommended for ``models_to_load`` if its peak VRAM leaves this much free."""

_HEADROOM_HINT_MB = 4000
"""When the busiest probe still left at least this much VRAM unused, the recommendation adds an advisory
that higher concurrency/batch than the probed values is likely safe: the catalog probes only to a fixed
ceiling, so on a large card the proven values under-provision the hardware."""

_SOAK_MEANINGFUL_MIN_JOBS = 4
"""A failed soak is only proof of *sustained-load* instability if it completed at least this many jobs.
Below it the soak never exercised sustained load (it wedged at startup or crashed early), a worker bug
to fix rather than grounds to stop loading a model whose isolated baseline passed cleanly."""


def _passed(result: CapabilityProbeResult) -> bool:
    """Whether a probe proved its capability cleanly (PROVEN with no robustness findings)."""
    return result.verdict is CapabilityVerdict.PROVEN and not result.findings


def _proven(results: list[CapabilityProbeResult], *kinds: CapabilityKind) -> bool:
    """Whether any probe of one of ``kinds`` was proven."""
    return any(result.verdict is CapabilityVerdict.PROVEN for result in results if result.capability.kind in kinds)


def _did_nothing(result: CapabilityProbeResult) -> bool:
    """True when a probe ran to its end without completing or faulting any job or alchemy form.

    A probe that timed out having dispatched no work proves nothing about the capability: it is closer
    to 'never exercised' than to 'tested and failed'. Treating it as untested keeps the recommendation
    from reporting a capability that never actually ran as a hard, proven-off failure.
    """
    stats = result.stats
    if stats is None:
        return False
    activity = (
        stats.num_jobs_completed
        + stats.num_jobs_faulted
        + stats.num_alchemy_forms_completed
        + stats.num_alchemy_forms_faulted
    )
    expected = stats.num_jobs_expected + stats.num_alchemy_forms_expected
    return activity == 0 and expected > 0


def _capability_status(results: list[CapabilityProbeResult], *kinds: CapabilityKind) -> tuple[SuggestionBasis, str]:
    """Classify how one or more capability kinds fared, so a suggestion can record proof versus absence.

    Accepts several kinds for capabilities any of which can establish the setting (alchemy is proven by
    either lane). Precedence is proven > failed > skipped/untested > not-probed: a single pass proves
    the capability regardless of other rungs, and a probe that ran-and-failed *with real activity* is a
    stronger signal than one never reached. A failed probe that dispatched no work (see
    :func:`_did_nothing`) is downgraded to untested rather than reported as a failure.
    """
    relevant = [result for result in results if result.capability.kind in kinds]
    if not relevant:
        return SuggestionBasis.NOT_IN_LADDER, ""
    if any(result.verdict is CapabilityVerdict.PROVEN for result in relevant):
        return SuggestionBasis.PROVEN, ""
    failed = [result for result in relevant if result.verdict in _FAILED_VERDICTS]
    real_failures = [result for result in failed if not _did_nothing(result)]
    if real_failures:
        result = real_failures[0]
        return SuggestionBasis.DISABLED_FAILED, "; ".join(result.reasons) or result.verdict.value
    if failed:
        # Every failure dispatched no work, so the capability was never actually exercised.
        result = failed[0]
        detail = "; ".join(result.reasons) or "probe ran but dispatched no work"
        return SuggestionBasis.UNTESTED_SKIPPED, f"never exercised: {detail}"
    for result in relevant:
        if result.verdict is CapabilityVerdict.SKIPPED:
            return SuggestionBasis.UNTESTED_SKIPPED, "; ".join(result.reasons) or "probe was skipped"
    return SuggestionBasis.NOT_IN_LADDER, ""


def _max_stable_batch(results: list[CapabilityProbeResult], tier: BenchTier) -> int:
    """Return the largest batch size that passed for ``tier`` with no robustness findings.

    The batch size is carried on the capability's magnitude (the quantity a batch probe proves), so the
    answer reads straight off the proven results without needing the original scenario.
    """
    stable = 1
    for result in results:
        capability = result.capability
        if capability.tier != tier or capability.kind is not CapabilityKind.BATCH or not _passed(result):
            continue
        stable = max(stable, capability.magnitude)
    return stable


def _fits_with_headroom(*, baseline_passed: bool, peak_vram_mb: int | None, total_vram_mb: int | None) -> bool:
    """Whether a tier's model is safe to keep resident: it passed and leaves the VRAM reserve free.

    When the machine VRAM or the observed peak is unknown, a passing baseline is taken to fit (the probe
    ran, so it physically fit); the headroom guard only *excludes* a model when it can prove the reserve
    would be breached.
    """
    if not baseline_passed:
        return False
    if total_vram_mb is None or peak_vram_mb is None:
        return True
    return (total_vram_mb - peak_vram_mb) >= _MIN_VRAM_HEADROOM_MB


def _baseline_result(results: list[CapabilityProbeResult], tier: BenchTier) -> CapabilityProbeResult | None:
    """Return the tier's baseline probe result, or None when it is absent."""
    for result in results:
        if result.capability.tier == tier and result.capability.kind is CapabilityKind.BASELINE:
            return result
    return None


def synthesize_capabilities(
    results: list[CapabilityProbeResult],
    *,
    total_vram_mb: int | None = None,
) -> WorkerCapabilities:
    """Summarize everything the worker proved it can do across the run.

    This is the unfiltered capability surface: every tier whose baseline passed and every feature whose
    probe passed in isolation, independent of whether the conservative recommendation turns it on.
    """
    capabilities = WorkerCapabilities(
        supports_hires_fix=_proven(results, CapabilityKind.HIRES_FIX),
        supports_post_processing=_proven(results, CapabilityKind.POST_PROCESSING),
        supports_controlnet=_proven(results, CapabilityKind.CONTROLNET),
        supports_qr_code=_proven(results, CapabilityKind.QR_CODE),
        supports_alchemy_clip=_proven(results, CapabilityKind.ALCHEMY_CLIP),
        supports_alchemy_graph=_proven(results, CapabilityKind.ALCHEMY_GRAPH),
        supports_alchemy_concurrent=_proven(results, CapabilityKind.ALCHEMY_CONCURRENT),
        supports_lora=_proven(results, CapabilityKind.LORA_DOWNLOAD),
    )

    for tier, model_name in BENCH_TIER_MODELS.items():
        baseline = _baseline_result(results, tier)
        if baseline is None:
            continue
        baseline_passed = baseline.verdict is CapabilityVerdict.PROVEN
        peak_vram = baseline.stats.vram_used_high_water_mb if baseline.stats is not None else None
        capabilities.tiers.append(
            TierCapability(
                tier=tier,
                model_name=model_name,
                baseline_passed=baseline_passed,
                observed_its_p50=baseline.stats.its_p50 if baseline.stats is not None else None,
                max_stable_batch=_max_stable_batch(results, tier),
                peak_vram_mb=peak_vram,
                fits_with_headroom=_fits_with_headroom(
                    baseline_passed=baseline_passed,
                    peak_vram_mb=peak_vram,
                    total_vram_mb=total_vram_mb,
                ),
            ),
        )

    return capabilities


def _vram_headroom_note(
    results: list[CapabilityProbeResult],
    models_to_load: list[str],
    total_vram_mb: int | None,
) -> str | None:
    """Advise when the GPU has substantial VRAM the probed values never touched.

    The proven max_threads/queue_size/max_batch are the catalog's fixed ceiling, not the hardware's. On
    a large card they can badly under-provision it. Rather than fabricate untested numbers (which would
    break the proven-only contract), surface the unused headroom so the operator knows higher is likely
    safe. Returns None when VRAM is unknown, nothing is loaded, or the busiest probe already used most
    of the card.
    """
    if total_vram_mb is None or not models_to_load:
        return None
    observed_peaks = [
        result.stats.vram_used_high_water_mb
        for result in results
        if result.stats is not None and result.stats.vram_used_high_water_mb is not None
    ]
    if not observed_peaks:
        return None
    spare = total_vram_mb - max(observed_peaks)
    if spare < _HEADROOM_HINT_MB:
        return None
    return (
        f"This GPU never used more than {max(observed_peaks)} of {total_vram_mb} MB VRAM across the "
        f"whole run (~{spare} MB free at the busiest point). max_threads, queue_size, and max_batch "
        "above are the highest values the catalog probed, not the highest this card can sustain; higher "
        "values are likely safe but were not benchmarked."
    )


class _SoakOutcome(BaseModel):
    """How the sustained-load soak fared, split by what its failure actually proves.

    The conservative recommendation must not drop a model whose isolated baseline passed unless the soak
    gives *real* evidence the model is unstable under load. So a soak failure is graded:

    - ``unstable_tiers``: the soak ran a meaningful number of jobs and then showed hard trouble (faults,
      recoveries, OOM, hang). Genuine evidence against keeping the model resident -> dropped.
    - ``inconclusive_tiers``: the soak failed for a reason that is *not* load instability (too few jobs
      to test sustained load, or a soft gate like the duty-cycle floor). The baseline still passed, so
      the model is kept resident with a caveat rather than silently dropped.
    - ``skipped_tiers``: the soak existed but never ran (an unmet prerequisite/machine gate); no evidence.
    """

    unstable_tiers: set[BenchTier] = Field(default_factory=set)
    inconclusive_tiers: set[BenchTier] = Field(default_factory=set)
    inconclusive_reasons: dict[BenchTier, str] = Field(default_factory=dict)
    skipped_tiers: set[BenchTier] = Field(default_factory=set)
    any_unstable: bool = False
    """Whether any soak failed with a hard robustness finding. Gates concurrent alchemy regardless of
    job count: a crash is a crash, even if it happened early."""


def _inconclusive_soak_reason(result: CapabilityProbeResult, completed: int) -> str:
    """Explain why a failed-but-inconclusive soak does not count against keeping the model resident."""
    if completed == 0:
        return (
            "its soak completed no jobs (a worker/startup issue, not sustained-load instability), so "
            "load stability was neither proven nor disproven"
        )
    detail = "; ".join(result.reasons)
    if detail:
        return f"its soak failed without a sustained-load instability finding ({detail})"
    return "its soak failed on a non-stability gate"


def _soak_outcome(results: list[CapabilityProbeResult]) -> _SoakOutcome:
    """Summarize the sustained probe, grading each failure by what it actually proves about the model."""
    outcome = _SoakOutcome()
    for result in results:
        if result.capability.kind is not CapabilityKind.SUSTAINED or result.verdict is CapabilityVerdict.PROVEN:
            continue
        tier = result.capability.tier
        if result.verdict is CapabilityVerdict.SKIPPED:
            outcome.skipped_tiers.add(tier)
            continue
        has_hard_finding = any(finding.kind in _HARD_SOAK_FINDINGS for finding in result.findings)
        if has_hard_finding:
            outcome.any_unstable = True
        completed = result.stats.num_jobs_completed if result.stats is not None else 0
        if has_hard_finding and completed >= _SOAK_MEANINGFUL_MIN_JOBS:
            outcome.unstable_tiers.add(tier)
        else:
            outcome.inconclusive_tiers.add(tier)
            outcome.inconclusive_reasons[tier] = _inconclusive_soak_reason(result, completed)
    return outcome


def synthesize_bridge_data(
    results: list[CapabilityProbeResult],
    *,
    total_vram_mb: int | None = None,
) -> SuggestedBridgeData:
    """Derive a conservative bridgeData recommendation from the proven probes.

    Conservative means: only models that passed and fit with VRAM headroom are loaded; ``max_batch`` is
    the largest batch that passed without a robustness finding (not merely the highest attempted); and
    concurrent alchemy is recommended only if it passed and the sustained-load soak held up. When the
    soak is present and a tier's soak failed, that tier's model is dropped; a hard soak failure also
    disables concurrent alchemy. ``notes`` records each downgrade.
    """
    suggestion = SuggestedBridgeData()
    notes: list[str] = []
    passing_tiers: list[BenchTier] = []

    for result in results:
        if result.verdict is not CapabilityVerdict.PROVEN:
            continue
        capability = result.capability
        kind = capability.kind
        if kind is CapabilityKind.BASELINE:
            passing_tiers.append(capability.tier)
        elif kind is CapabilityKind.QUEUE_SIZE:
            suggestion.queue_size = max(suggestion.queue_size, capability.magnitude or 1)
        elif kind is CapabilityKind.THREADS:
            suggestion.max_threads = max(suggestion.max_threads, capability.magnitude or 1)
        elif kind is CapabilityKind.CONTROLNET:
            suggestion.allow_controlnet = True
        elif kind is CapabilityKind.QR_CODE:
            # The QR-code workflow proves the *SDXL-controlnet* path, not classic preprocessor
            # controlnet. allow_controlnet (canny/depth/openpose) is therefore left to the CONTROLNET
            # capability alone: enabling it off a QR-code pass made the worker advertise and accept
            # classic controlnet jobs the CONTROLNET probe had just shown to crash.
            if capability.tier is BenchTier.SDXL:
                suggestion.allow_sdxl_controlnet = True
        elif kind is CapabilityKind.POST_PROCESSING:
            suggestion.allow_post_processing = True
        elif kind is CapabilityKind.LORA_DOWNLOAD:
            suggestion.allow_lora = True
        elif kind in (CapabilityKind.ALCHEMY_CLIP, CapabilityKind.ALCHEMY_GRAPH):
            suggestion.alchemist = True
        elif kind is CapabilityKind.ALCHEMY_CONCURRENT:
            suggestion.alchemist = True
            suggestion.alchemy_allow_concurrent = True
            suggestion.alchemy_max_concurrency = 2

    # max_batch: the largest batch that passed cleanly across any tier (a per-job payload cap).
    suggestion.max_batch = max((_max_stable_batch(results, tier) for tier in passing_tiers), default=1)

    soak = _soak_outcome(results)

    models_to_load: list[str] = []
    for tier in passing_tiers:
        baseline = _baseline_result(results, tier)
        peak_vram = baseline.stats.vram_used_high_water_mb if baseline is not None and baseline.stats else None
        if not _fits_with_headroom(baseline_passed=True, peak_vram_mb=peak_vram, total_vram_mb=total_vram_mb):
            notes.append(
                f"{BENCH_TIER_MODELS[tier]} omitted from models_to_load: its peak VRAM leaves under "
                f"{_MIN_VRAM_HEADROOM_MB} MB headroom on this GPU.",
            )
            continue
        if tier in soak.unstable_tiers:
            notes.append(
                f"{BENCH_TIER_MODELS[tier]} omitted from models_to_load: its sustained-load soak ran and did "
                "not hold up.",
            )
            continue
        if tier in soak.skipped_tiers:
            notes.append(
                f"{BENCH_TIER_MODELS[tier]} omitted from models_to_load: its sustained-load soak was skipped, so "
                "stability under combined load was never validated.",
            )
            continue
        if tier in soak.inconclusive_tiers:
            # The baseline passed and fits with headroom; the soak failure was not load instability, so
            # keep the model resident rather than zeroing out the recommendation, but say so plainly.
            notes.append(
                f"{BENCH_TIER_MODELS[tier]} kept in models_to_load despite an inconclusive soak: "
                f"{soak.inconclusive_reasons[tier]}.",
            )
        models_to_load.append(BENCH_TIER_MODELS[tier])
    suggestion.models_to_load = models_to_load

    headroom_note = _vram_headroom_note(results, models_to_load, total_vram_mb)
    if headroom_note is not None:
        notes.append(headroom_note)

    concurrent_capped_by_soak = False
    if suggestion.alchemy_allow_concurrent and soak.any_unstable:
        suggestion.alchemy_allow_concurrent = False
        suggestion.alchemy_max_concurrency = 1
        concurrent_capped_by_soak = True
        notes.append("Concurrent alchemy disabled: the sustained-load soak showed instability under combined load.")

    suggestion.notes = notes
    suggestion.decisions = _build_decisions(
        results,
        suggestion,
        passing_tiers=passing_tiers,
        soak=soak,
        concurrent_capped_by_soak=concurrent_capped_by_soak,
    )
    return suggestion


def _build_decisions(
    results: list[CapabilityProbeResult],
    suggestion: SuggestedBridgeData,
    *,
    passing_tiers: list[BenchTier],
    soak: _SoakOutcome,
    concurrent_capped_by_soak: bool,
) -> list[SuggestionDecision]:
    """Record the basis behind every suggested value, distinguishing proof from untested absence.

    The values are already decided; this only annotates *why* each holds, by re-reading how the governing
    capability fared. The single subtlety is concurrent alchemy, whose value may have been downgraded
    from a real pass by an unstable soak: that is reported as a soak cap, not a capability basis.
    """

    def capability_decision(setting: str, value: bool | int, *kinds: CapabilityKind) -> SuggestionDecision:
        basis, detail = _capability_status(results, *kinds)
        return SuggestionDecision(setting=setting, value=value, basis=basis, detail=detail)

    decisions = [
        capability_decision("max_threads", suggestion.max_threads, CapabilityKind.THREADS),
        capability_decision("queue_size", suggestion.queue_size, CapabilityKind.QUEUE_SIZE),
        capability_decision("max_batch", suggestion.max_batch, CapabilityKind.BATCH),
        capability_decision("allow_controlnet", suggestion.allow_controlnet, CapabilityKind.CONTROLNET),
        capability_decision("allow_sdxl_controlnet", suggestion.allow_sdxl_controlnet, CapabilityKind.QR_CODE),
        capability_decision("allow_post_processing", suggestion.allow_post_processing, CapabilityKind.POST_PROCESSING),
        capability_decision("allow_lora", suggestion.allow_lora, CapabilityKind.LORA_DOWNLOAD),
        capability_decision(
            "alchemist",
            suggestion.alchemist,
            CapabilityKind.ALCHEMY_CLIP,
            CapabilityKind.ALCHEMY_GRAPH,
        ),
    ]

    if concurrent_capped_by_soak:
        decisions.append(
            SuggestionDecision(
                setting="alchemy_allow_concurrent",
                value=suggestion.alchemy_allow_concurrent,
                basis=SuggestionBasis.CAPPED_SOAK,
                detail="proved out in isolation but the sustained-load soak was unstable under combined load",
            ),
        )
    else:
        decisions.append(
            capability_decision(
                "alchemy_allow_concurrent",
                suggestion.alchemy_allow_concurrent,
                CapabilityKind.ALCHEMY_CONCURRENT,
            ),
        )

    decisions.append(_models_to_load_decision(results, suggestion, passing_tiers=passing_tiers, soak=soak))
    return decisions


def _models_to_load_decision(
    results: list[CapabilityProbeResult],
    suggestion: SuggestedBridgeData,
    *,
    passing_tiers: list[BenchTier],
    soak: _SoakOutcome,
) -> SuggestionDecision:
    """Classify the models_to_load recommendation: proven, capped by VRAM/soak, or never proven."""
    if suggestion.models_to_load:
        return SuggestionDecision(
            setting="models_to_load",
            value=list(suggestion.models_to_load),
            basis=SuggestionBasis.PROVEN,
            detail="tier baselines passed; loaded with VRAM headroom",
        )
    if passing_tiers:
        # Every passing tier was held back; report the dominant reason so it does not read as a failure.
        # (Inconclusive soaks keep their model, so they never reach this empty-list branch.)
        basis = (
            SuggestionBasis.CAPPED_SOAK if soak.unstable_tiers or soak.skipped_tiers else SuggestionBasis.CAPPED_VRAM
        )
        return SuggestionDecision(
            setting="models_to_load",
            value=[],
            basis=basis,
            detail="all passing tiers were held back; see notes",
        )
    basis, detail = _capability_status(results, CapabilityKind.BASELINE)
    return SuggestionDecision(setting="models_to_load", value=[], basis=basis, detail=detail)


def verify_suggestion_consistency(report: CapabilityReport) -> list[str]:
    """Return a message for any enabled capability or loaded model not grounded in a PROVEN basis.

    This is the literal "is the suggestion consistent with what actually ran" guard: a recommendation
    should never turn something *on* (or keep a model resident) on the strength of a probe that was
    skipped, failed, or absent. Returns an empty list when the recommendation is fully grounded.
    """
    suggestion = report.suggested_bridge_data
    by_setting = {decision.setting: decision for decision in suggestion.decisions}
    enabled: dict[str, bool | list[str]] = {
        "allow_controlnet": suggestion.allow_controlnet,
        "allow_sdxl_controlnet": suggestion.allow_sdxl_controlnet,
        "allow_post_processing": suggestion.allow_post_processing,
        "allow_lora": suggestion.allow_lora,
        "alchemist": suggestion.alchemist,
        "alchemy_allow_concurrent": suggestion.alchemy_allow_concurrent,
        "models_to_load": list(suggestion.models_to_load),
    }
    problems: list[str] = []
    for setting, value in enabled.items():
        if not value:
            continue
        decision = by_setting.get(setting)
        if decision is None:
            problems.append(f"{setting} is set but carries no provenance decision.")
        elif decision.basis != SuggestionBasis.PROVEN:
            problems.append(f"{setting} is enabled on a '{decision.basis}' basis, not a proven pass.")
    return problems


__all__ = [
    "synthesize_bridge_data",
    "synthesize_capabilities",
    "verify_suggestion_consistency",
]
