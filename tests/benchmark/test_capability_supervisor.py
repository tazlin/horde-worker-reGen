"""Unit tests for the pure capability escalation policy (RUN / SKIP / STOP)."""

from __future__ import annotations

from horde_worker_regen.benchmark.capabilities.capability import (
    Capability,
    CapabilityKind,
    CapabilityVerdict,
)
from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
from horde_worker_regen.benchmark.capabilities.result import CapabilityProbeResult
from horde_worker_regen.benchmark.capabilities.supervisor import (
    CapabilitySupervisor,
    ProbeAction,
)
from horde_worker_regen.benchmark.enums import BenchTier, FindingKind
from horde_worker_regen.benchmark.report import Finding, HarnessSummary
from horde_worker_regen.benchmark.scenarios import CannedImageJobSpec, Scenario

_BASELINE = Capability(tier=BenchTier.SD15, kind=CapabilityKind.BASELINE)
_THREADS = Capability(tier=BenchTier.SD15, kind=CapabilityKind.THREADS)


def _probe(
    capability: Capability,
    *,
    requires: tuple[Capability, ...] = (),
    establishes_baseline: bool = False,
) -> CapabilityProbe:
    """A minimal probe carrying a one-job scenario, for policy tests that never actually run it."""
    return CapabilityProbe(
        capability=capability,
        scenario=Scenario(name=capability.slug, image_jobs=[CannedImageJobSpec(count=1)]),
        requires=requires,
        establishes_baseline=establishes_baseline,
    )


def _result(
    capability: Capability,
    verdict: CapabilityVerdict,
    *,
    jobs_completed: int = 1,
    findings: list[Finding] | None = None,
) -> CapabilityProbeResult:
    """A synthetic probe result with a given verdict and (optionally) findings."""
    return CapabilityProbeResult(
        capability=capability,
        verdict=verdict,
        harness=HarnessSummary(num_jobs_completed=jobs_completed),
        findings=findings or [],
    )


def test_runs_a_probe_with_no_prerequisites() -> None:
    """A probe that requires nothing runs when the machine can host it."""
    supervisor = CapabilitySupervisor()
    decision = supervisor.evaluate(_probe(_BASELINE), machine_skip_reason=None)
    assert decision.action is ProbeAction.RUN


def test_skips_a_probe_whose_prerequisite_is_unproven() -> None:
    """A probe is skipped while a required capability has not been proven, naming the gap."""
    supervisor = CapabilitySupervisor()
    decision = supervisor.evaluate(_probe(_THREADS, requires=(_BASELINE,)), machine_skip_reason=None)
    assert decision.action is ProbeAction.SKIP
    assert _BASELINE.slug in decision.reason


def test_runs_a_probe_once_its_prerequisite_is_proven() -> None:
    """Recording a proven prerequisite unblocks its dependents."""
    supervisor = CapabilitySupervisor()
    supervisor.record(_probe(_BASELINE), _result(_BASELINE, CapabilityVerdict.PROVEN))
    decision = supervisor.evaluate(_probe(_THREADS, requires=(_BASELINE,)), machine_skip_reason=None)
    assert decision.action is ProbeAction.RUN


def test_disproven_prerequisite_keeps_dependents_skipped() -> None:
    """A failed prerequisite is not in the proven set, so dependents stay skipped (the cascade)."""
    supervisor = CapabilitySupervisor()
    supervisor.record(_probe(_BASELINE), _result(_BASELINE, CapabilityVerdict.DISPROVEN, jobs_completed=1))
    decision = supervisor.evaluate(_probe(_THREADS, requires=(_BASELINE,)), machine_skip_reason=None)
    assert decision.action is ProbeAction.SKIP


def test_machine_skip_reason_skips_before_checking_requirements() -> None:
    """A machine-fit reason skips the probe and is surfaced verbatim."""
    supervisor = CapabilitySupervisor()
    decision = supervisor.evaluate(_probe(_BASELINE), machine_skip_reason="insufficient VRAM")
    assert decision.action is ProbeAction.SKIP
    assert decision.reason == "insufficient VRAM"


def test_crash_aborts_the_run_and_stops_later_probes() -> None:
    """A crashed probe latches the abort, so every later probe is stopped (broken shared stack)."""
    supervisor = CapabilitySupervisor()
    supervisor.record(_probe(_BASELINE), _result(_BASELINE, CapabilityVerdict.CRASHED))
    assert supervisor.is_aborted
    decision = supervisor.evaluate(_probe(_THREADS, requires=(_BASELINE,)), machine_skip_reason=None)
    assert decision.action is ProbeAction.STOP
    assert "aborted" in decision.reason


def test_baseline_zero_work_with_crash_finding_is_catastrophic() -> None:
    """A baseline that completed no jobs while its processes fell over aborts the whole run."""
    supervisor = CapabilitySupervisor()
    finding = Finding(kind=FindingKind.PROCESS_RECOVERY, level_id=_BASELINE.slug, evidence="replaced")
    supervisor.record(
        _probe(_BASELINE, establishes_baseline=True),
        _result(_BASELINE, CapabilityVerdict.DISPROVEN, jobs_completed=0, findings=[finding]),
    )
    assert supervisor.is_aborted


def test_baseline_slow_failure_is_not_catastrophic() -> None:
    """A baseline that did work but missed criteria only disproves itself; the run continues."""
    supervisor = CapabilitySupervisor()
    supervisor.record(
        _probe(_BASELINE, establishes_baseline=True),
        _result(_BASELINE, CapabilityVerdict.DISPROVEN, jobs_completed=2),
    )
    assert not supervisor.is_aborted


def test_abort_can_be_disabled() -> None:
    """With the abort disabled, a crash does not stop later probes."""
    supervisor = CapabilitySupervisor(abort_on_catastrophe=False)
    supervisor.record(_probe(_BASELINE), _result(_BASELINE, CapabilityVerdict.CRASHED))
    assert not supervisor.is_aborted
    # The crashed baseline is not proven, so a dependent is skipped (not stopped).
    decision = supervisor.evaluate(_probe(_THREADS, requires=(_BASELINE,)), machine_skip_reason=None)
    assert decision.action is ProbeAction.SKIP
