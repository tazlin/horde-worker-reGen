"""Run one capability probe end to end: drive the harness, distill stats, judge, classify findings.

This is the single place a probe turns into a :class:`CapabilityProbeResult`, shared by every driver:
the benchmark executor (warm session or per-probe subprocess), ``pytest -m e2e`` (fake mode), and
``pytest -m gpu`` (real hardware). Centralising it means a benchmark probe and its test run the exact
same code path, so "does this capability hold?" has one answer regardless of who asked.

It is the boundary between the package's import-light core and the heavy harness: the harness is
imported lazily inside the run functions, so importing this module (for its signatures) still does not
drag torch, while actually running a probe does.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from horde_worker_regen.benchmark.capabilities.capability import CapabilityKind, CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.result import CapabilityProbeResult, classify_findings
from horde_worker_regen.benchmark.capabilities.stats import level_stats_from_harness_result
from horde_worker_regen.benchmark.capabilities.timing import probe_timing
from horde_worker_regen.benchmark.criteria import evaluate_level
from horde_worker_regen.benchmark.report import HarnessSummary

if TYPE_CHECKING:
    from collections.abc import Callable

    from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
    from horde_worker_regen.benchmark.criteria import TierBaseline
    from horde_worker_regen.harness import HarnessProcessMode, HarnessResult, WarmHarnessSession
    from horde_worker_regen.process_management.resources.run_metrics import RunMetricsSnapshot

_WARMUP_KINDS: frozenset[CapabilityKind] = frozenset(
    {
        CapabilityKind.HIRES_FIX,
        CapabilityKind.POST_PROCESSING,
        CapabilityKind.CONTROLNET,
        CapabilityKind.QR_CODE,
        CapabilityKind.ALCHEMY_CLIP,
        CapabilityKind.ALCHEMY_GRAPH,
        CapabilityKind.ALCHEMY_CONCURRENT,
    },
)
"""Feature/alchemy kinds whose warm-session run pre-warms its model before the measured pass.

These are the first to touch a feature-specific weight (controlnet/QR checkpoints, upscaler/face-fixer/
BLIP models) on a reused worker, so a warm run pre-loads to absorb the one-time cold-load recovery,
exactly as the old controller warmed FEATURES/ALCHEMY levels. Baseline/concurrency probes reuse the
already-warm base checkpoint and need no warmup."""


def _harness_summary(result: HarnessResult) -> HarnessSummary:
    """Project a live :class:`HarnessResult` onto the JSON-friendly count summary the result carries."""
    return HarnessSummary(
        num_jobs_expected=result.num_jobs_expected,
        num_jobs_completed=result.num_jobs_completed,
        num_jobs_faulted=result.num_jobs_faulted,
        num_alchemy_forms_expected=result.num_alchemy_forms_expected,
        num_alchemy_forms_completed=result.num_alchemy_forms_completed,
        num_alchemy_forms_faulted=result.num_alchemy_forms_faulted,
        elapsed_seconds=result.elapsed_seconds,
        timed_out=result.timed_out,
        audit_failures=list(result.audit_failures),
        exit_reason=result.exit_reason,
        diagnostics=list(result.diagnostics),
    )


def _max_threads(probe: CapabilityProbe) -> int:
    """The probe's requested ``max_threads`` override (warm path), defaulting to 1."""
    value = probe.bridge_data_overrides.get("max_threads", 1)
    return value if isinstance(value, int) else 1


async def run_capability_probe_async(
    probe: CapabilityProbe,
    *,
    process_mode: HarnessProcessMode,
    total_vram_mb: int | None = None,
    baseline: TierBaseline | None = None,
    warm_session: WarmHarnessSession | None = None,
    on_progress: Callable[[RunMetricsSnapshot, float], None] | None = None,
) -> CapabilityProbeResult:
    """Run ``probe`` and return its complete result (verdict, stats, findings).

    A fixed-scenario probe runs on ``warm_session`` when one is supplied (reusing a booted worker);
    otherwise, and always for a soak, it runs in its own harness via
    :meth:`HarnessConfig.from_scenario`. ``baseline`` is the tier's reference it/s for the criteria
    comparison; ``total_vram_mb`` sizes the VRAM-headroom check.
    """
    from horde_worker_regen.harness import HarnessConfig, run_harness_async

    is_soak = probe.scenario.soak_seconds is not None
    if warm_session is not None and not is_soak:
        result = await warm_session.run_level(
            jobs=probe.scenario.expand_image_jobs(),
            alchemy_forms=probe.scenario.expand_alchemy_forms() or None,
            threads=_max_threads(probe),
            timeout_seconds=probe.timeout_seconds,
            warmup=probe.capability.kind in _WARMUP_KINDS,
            on_progress=on_progress,
        )
    else:
        config = HarnessConfig.from_scenario(
            probe.scenario,
            process_mode=process_mode,
            timeout_seconds=probe.timeout_seconds,
            bridge_data_overrides=dict(probe.bridge_data_overrides),
            audit=True,
            on_progress=on_progress,
        )
        result = await run_harness_async(config)

    return _result_from_harness(probe, result, total_vram_mb=total_vram_mb, baseline=baseline)


def _result_from_harness(
    probe: CapabilityProbe,
    result: HarnessResult,
    *,
    total_vram_mb: int | None,
    baseline: TierBaseline | None,
) -> CapabilityProbeResult:
    """Distill a completed harness run into a probe result: stats, verdict, timing, findings (no crash)."""
    stats = level_stats_from_harness_result(result, total_vram_mb=total_vram_mb)
    verdict = evaluate_level(stats, probe.criteria, baseline)
    findings = classify_findings(
        probe,
        audit_failures=list(result.audit_failures),
        metrics=result.metrics,
        log_tail=[],
        crashed=False,
    )
    timing = probe_timing(
        started_at_epoch=result.started_at_epoch,
        elapsed_seconds=result.elapsed_seconds,
        jobs=list(result.metrics.jobs) if result.metrics is not None else [],
    )
    logger.info(f"probe {probe.probe_id} timing: {timing.summary()}")
    return CapabilityProbeResult(
        capability=probe.capability,
        verdict=CapabilityVerdict.PROVEN if verdict.passed else CapabilityVerdict.DISPROVEN,
        reasons=verdict.reasons,
        advisories=verdict.advisories,
        stats=stats,
        harness=_harness_summary(result),
        timing=timing,
        findings=findings,
    )


def run_capability_probe(
    probe: CapabilityProbe,
    *,
    process_mode: HarnessProcessMode,
    total_vram_mb: int | None = None,
    baseline: TierBaseline | None = None,
) -> CapabilityProbeResult:
    """Synchronous wrapper around :func:`run_capability_probe_async` for non-async callers.

    Runs its own event loop, so it must not be called from within a running loop (use the async form
    there). The warm-session path is async-only and therefore not exposed here.
    """
    return asyncio.run(
        run_capability_probe_async(
            probe,
            process_mode=process_mode,
            total_vram_mb=total_vram_mb,
            baseline=baseline,
        ),
    )


__all__ = [
    "run_capability_probe",
    "run_capability_probe_async",
]
