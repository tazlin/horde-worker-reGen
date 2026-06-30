"""Drive a capability run: walk the plan on one warm worker, then synthesize the report.

This is the thin imperative remainder of the old ``BenchmarkController``, rebuilt on the pure engine.
The cascade logic that bloated the controller is gone: ordering is the topological :func:`build_plan`,
the skip/abort policy is the pure :class:`CapabilitySupervisor`, and a probe becomes a result through
the one shared :func:`run_capability_probe_async`. What remains here is genuinely imperative: detect the
machine, boot one warm worker so the rampup is paid once (not per probe, the way the per-probe cold
tests are), run each probe the supervisor admits, establish the per-tier it/s reference for the criteria
comparison, then soak the synthesized recommendation and assemble the :class:`CapabilityReport`.

Like :mod:`probe_runner`, this module is the heavy boundary: it imports the harness lazily inside the
run so the type is importable (and ``detect_machine_info`` usable) without dragging torch.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from horde_worker_regen.benchmark.capabilities.capability import Capability, CapabilityKind, CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.catalog import (
    CatalogOptions,
    build_capability_catalog,
    build_sustained_probe,
)
from horde_worker_regen.benchmark.capabilities.plan import build_plan
from horde_worker_regen.benchmark.capabilities.probe_runner import run_capability_probe_async
from horde_worker_regen.benchmark.capabilities.recommendation import synthesize_bridge_data, synthesize_capabilities
from horde_worker_regen.benchmark.capabilities.report_render import render_markdown
from horde_worker_regen.benchmark.capabilities.result import (
    CapabilityProbeResult,
    CapabilityReport,
    MachineInfo,
    SuggestedBridgeData,
)
from horde_worker_regen.benchmark.capabilities.supervisor import CapabilitySupervisor, ProbeAction
from horde_worker_regen.benchmark.criteria import TierBaseline
from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.ladder import BENCH_TIER_MODELS
from horde_worker_regen.benchmark.requirements import (
    civitai_token_available,
    compute_probe_requirements,
    requirement_skip_reason,
)

if TYPE_CHECKING:
    from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
    from horde_worker_regen.harness import HarnessProcessMode, WarmHarnessSession


def detect_machine_info(*, probe_devices: bool = True) -> MachineInfo:
    """Best-effort hardware detection (no-op on machines without torch/CUDA).

    RAM is always read (cheap, via psutil). GPU enumeration is skipped when ``probe_devices`` is False:
    it runs out-of-process (see :func:`accelerator_probe.probe_accelerators`) to keep this caller
    torch-free, but that subprocess loads torch and is pointless in fake/dry-run mode, which never
    touches the GPU. Real runs leave it on so the report and fit verdicts see the actual device.
    """
    info = MachineInfo()
    try:
        import psutil

        info.total_ram_bytes = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001 - purely informational
        pass
    if not probe_devices:
        return info
    try:
        from horde_worker_regen.utils.accelerator_probe import probe_accelerators

        accelerators = probe_accelerators()
        if accelerators:
            primary = accelerators[0]
            info.gpu_name = primary.name
            info.total_vram_mb = primary.total_vram_mb
    except Exception:  # noqa: BLE001 - purely informational
        pass
    return info


def _probe_threads(probe: CapabilityProbe) -> int:
    """The probe's requested ``max_threads`` override (1 when unset), to size the warm worker ceiling."""
    value = probe.bridge_data_overrides.get("max_threads", 1)
    return value if isinstance(value, int) else 1


class ProbeExecutor:
    """Runs a capability catalog on one warm worker and assembles the report.

    The static probes run against a single reused worker so the boot rampup is amortized across every
    check; the supervisor decides per probe whether to run, skip (machine cannot host it, or a
    prerequisite is unproven), or stop (a catastrophe broke the shared stack). After the static run the
    recommendation is synthesized and soaked per passing tier, then re-synthesized so an unstable soak
    can downgrade it.
    """

    def __init__(
        self,
        *,
        catalog_options: CatalogOptions | None = None,
        process_mode: HarnessProcessMode = "fake",
        machine: MachineInfo | None = None,
        out_dir: Path | None = None,
        run_soak: bool = True,
        soak_seconds: float = 120.0,
        strict_duty_cycle: bool = False,
        only_probe: str | None = None,
    ) -> None:
        """Configure a run; ``machine`` is detected at run time when None, ``only_probe`` narrows to one slug."""
        self._catalog_options = catalog_options if catalog_options is not None else CatalogOptions()
        self._process_mode = process_mode
        self._machine = machine
        self._out_dir = out_dir
        self._run_soak = run_soak
        self._soak_seconds = soak_seconds
        self._strict_duty_cycle = strict_duty_cycle
        self._only_probe = only_probe

    async def run_async(self) -> CapabilityReport:
        """Run the catalog and return the assembled (and, if ``out_dir`` is set, persisted) report."""
        machine = self._machine or detect_machine_info(probe_devices=self._process_mode == "real")

        probes = [
            probe
            for probe in build_capability_catalog(self._catalog_options)
            if self._only_probe is None or probe.probe_id == self._only_probe
        ]
        plan = build_plan(probes)
        supervisor = CapabilitySupervisor()
        tier_baselines: dict[BenchTier, TierBaseline] = {}
        results: list[CapabilityProbeResult] = []

        from horde_worker_regen.harness import WarmHarnessSession

        async with WarmHarnessSession(
            process_mode=self._process_mode,
            model_names=self._warm_model_names(plan.probes),
            max_threads_ceiling=max((_probe_threads(probe) for probe in plan.probes), default=1),
        ) as session:
            for probe in plan.probes:
                result = await self._resolve_probe(
                    probe,
                    machine=machine,
                    supervisor=supervisor,
                    baseline=tier_baselines.get(probe.capability.tier),
                    warm_session=session,
                )
                supervisor.record(probe, result)
                results.append(result)
                self._record_baseline(probe, result, tier_baselines)

        suggested = synthesize_bridge_data(results, total_vram_mb=machine.total_vram_mb)
        if self._run_soak and self._only_probe is None and tier_baselines:
            results.extend(
                await self._run_soaks(
                    suggested,
                    machine=machine,
                    supervisor=supervisor,
                    tier_baselines=tier_baselines,
                ),
            )
            suggested = synthesize_bridge_data(results, total_vram_mb=machine.total_vram_mb)

        report = CapabilityReport(
            machine=machine,
            probes=results,
            capabilities=synthesize_capabilities(results, total_vram_mb=machine.total_vram_mb),
            suggested_bridge_data=suggested,
            tier_baselines_its={str(tier): baseline.its_p50 for tier, baseline in tier_baselines.items()},
        )

        if self._out_dir is not None:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            (self._out_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
            (self._out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")

        return report

    def run(self) -> CapabilityReport:
        """Synchronous wrapper around :meth:`run_async` (runs its own event loop)."""
        return asyncio.run(self.run_async())

    async def _resolve_probe(
        self,
        probe: CapabilityProbe,
        *,
        machine: MachineInfo,
        supervisor: CapabilitySupervisor,
        baseline: TierBaseline | None,
        warm_session: WarmHarnessSession,
    ) -> CapabilityProbeResult:
        """Ask the supervisor whether to run ``probe``, then run it on the warm worker or record a skip."""
        skip_reason = requirement_skip_reason(
            compute_probe_requirements(probe),
            machine=machine,
            process_mode=self._process_mode,
            civitai_available=civitai_token_available(),
        )
        decision = supervisor.evaluate(probe, machine_skip_reason=skip_reason)
        if decision.action is not ProbeAction.RUN:
            logger.info(f"Skipping probe {probe.probe_id}: {decision.reason}")
            return _skipped_result(probe, decision.reason)

        logger.info(f"Running probe {probe.probe_id}")
        return await run_capability_probe_async(
            probe,
            process_mode=self._process_mode,
            total_vram_mb=machine.total_vram_mb,
            baseline=baseline,
            warm_session=warm_session,
        )

    async def _run_soaks(
        self,
        suggested: SuggestedBridgeData,
        *,
        machine: MachineInfo,
        supervisor: CapabilitySupervisor,
        tier_baselines: dict[BenchTier, TierBaseline],
    ) -> list[CapabilityProbeResult]:
        """Soak the recommendation under sustained load, one SUSTAINED probe per passing tier.

        Each soak runs in its own harness (a soak always does, even with a warm session available), so
        these run after the warm session has closed. The probe requires its tier baseline, so a tier
        whose baseline never proved out is skipped by the supervisor rather than soaked blindly.
        """
        soak_results: list[CapabilityProbeResult] = []
        for tier in tier_baselines:
            sustained = build_sustained_probe(
                suggested,
                tier,
                soak_seconds=self._soak_seconds,
                requires=(Capability(tier=tier, kind=CapabilityKind.BASELINE),),
                strict_duty_cycle=self._strict_duty_cycle,
            )
            skip_reason = requirement_skip_reason(
                compute_probe_requirements(sustained),
                machine=machine,
                process_mode=self._process_mode,
                civitai_available=civitai_token_available(),
            )
            decision = supervisor.evaluate(sustained, machine_skip_reason=skip_reason)
            if decision.action is not ProbeAction.RUN:
                logger.info(f"Skipping soak {sustained.probe_id}: {decision.reason}")
                result = _skipped_result(sustained, decision.reason)
            else:
                logger.info(f"Soaking {sustained.probe_id} for {self._soak_seconds:.0f}s")
                result = await run_capability_probe_async(
                    sustained,
                    process_mode=self._process_mode,
                    total_vram_mb=machine.total_vram_mb,
                    baseline=tier_baselines[tier],
                )
            supervisor.record(sustained, result)
            soak_results.append(result)
        return soak_results

    @staticmethod
    def _record_baseline(
        probe: CapabilityProbe,
        result: CapabilityProbeResult,
        tier_baselines: dict[BenchTier, TierBaseline],
    ) -> None:
        """Record a proven baseline's it/s p50 as the tier reference the criteria gate later compares to."""
        if not probe.establishes_baseline or result.verdict is not CapabilityVerdict.PROVEN:
            return
        if result.stats is not None and result.stats.its_p50 is not None:
            tier_baselines[probe.capability.tier] = TierBaseline(
                tier=str(probe.capability.tier),
                its_p50=result.stats.its_p50,
            )

    def _warm_model_names(self, probes: list[CapabilityProbe]) -> list[str]:
        """The union of every probe's referenced models plus the catalog tiers' base models.

        The warm worker is booted once with a reference covering everything the run might touch; an
        individual probe installs only its own scenario, so listing a model it never uses is harmless.
        """
        names: set[str] = set()
        for probe in probes:
            names.update(probe.scenario.models_referenced())
        for tier in self._catalog_options.tiers:
            if tier in BENCH_TIER_MODELS:
                names.add(BENCH_TIER_MODELS[tier])
        return sorted(names)


def _skipped_result(probe: CapabilityProbe, reason: str) -> CapabilityProbeResult:
    """A synthesized result for a probe the supervisor declined to run."""
    return CapabilityProbeResult(
        capability=probe.capability,
        verdict=CapabilityVerdict.SKIPPED,
        reasons=[reason] if reason else [],
    )


__all__ = ["ProbeExecutor", "detect_machine_info"]
