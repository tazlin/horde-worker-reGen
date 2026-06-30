"""The pure escalation policy for a capability run: per probe, decide RUN / SKIP / STOP.

This mirrors the worker's ``RecoverySupervisor`` (``process_management.lifecycle.recovery_supervisor``):
an enum-returning :meth:`evaluate`, all state injected or accumulated through :meth:`record`, and no
I/O, so the policy is unit-testable against a synthetic catalog with no harness. It absorbs the old
controller's imperative skip cascade (failed-tier-baseline / failed-axis sets) and catastrophe abort
into one declarative rule set:

- a probe is **skipped** when the machine cannot host it (the resource gate, computed elsewhere and
  passed in) or when any capability it ``requires`` has not been proven;
- a probe is **run** otherwise;
- once a probe outcome is *catastrophic* (a crash/hang, or the simplest baseline failing to do any
  work while its processes fell over), the shared worker stack is presumed broken and every later
  probe is **stopped**, since it would only repeat the failure on the same stack.

The executor owns the side effects (booting workers, persisting results, emitting progress); this
class only tracks what has been proven and answers "what should happen to this probe?".
"""

from __future__ import annotations

from enum import StrEnum, auto

from pydantic import BaseModel

from horde_worker_regen.benchmark.capabilities.capability import Capability, CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
from horde_worker_regen.benchmark.capabilities.result import CapabilityProbeResult
from horde_worker_regen.benchmark.enums import FindingKind

_CATASTROPHIC_FINDING_KINDS: frozenset[FindingKind] = frozenset(
    {FindingKind.CRASH, FindingKind.OOM, FindingKind.HANG, FindingKind.PROCESS_RECOVERY},
)
"""Finding kinds that mark a baseline's zero-work failure as a broken stack rather than a slow one."""


class ProbeAction(StrEnum):
    """What the supervisor wants the executor to do with a probe this turn."""

    RUN = auto()
    """The machine can host it and every prerequisite is proven: run it."""
    SKIP = auto()
    """Record a skipped result without running: a prerequisite is unproven or the machine cannot host it."""
    STOP = auto()
    """The run has aborted on a catastrophe: record this and every later probe as skipped, run nothing."""


class ProbeDecision(BaseModel):
    """The action to take for one probe, with the reason (surfaced into the skipped result)."""

    action: ProbeAction
    reason: str = ""


class CapabilitySupervisor:
    """Tracks proven/disproven/crashed capabilities and decides RUN / SKIP / STOP per probe.

    Drive it once per probe in plan order: :meth:`evaluate` (with the machine-fit reason, or None) to
    get the action, run or skip accordingly, then :meth:`record` the result so later probes see what
    this one proved. The plan's topological order guarantees a probe's prerequisites are recorded
    before it is evaluated.
    """

    def __init__(self, *, abort_on_catastrophe: bool = True) -> None:
        """Initialize an empty run; ``abort_on_catastrophe`` toggles the broken-stack short-circuit."""
        self._abort_on_catastrophe = abort_on_catastrophe
        self._proven: set[Capability] = set()
        self._disproven: set[Capability] = set()
        self._crashed: set[Capability] = set()
        self._aborted_reason: str | None = None

    @property
    def proven(self) -> frozenset[Capability]:
        """The capabilities proven so far (a probe runs only when all of its ``requires`` are in here)."""
        return frozenset(self._proven)

    @property
    def is_aborted(self) -> bool:
        """Whether a catastrophe has aborted the run (every subsequent probe is stopped)."""
        return self._aborted_reason is not None

    @property
    def aborted_reason(self) -> str | None:
        """Why the run aborted, or None if it has not."""
        return self._aborted_reason

    def evaluate(self, probe: CapabilityProbe, *, machine_skip_reason: str | None) -> ProbeDecision:
        """Decide what to do with ``probe`` given the proven set and whether the machine can host it.

        Args:
            probe: The probe under consideration.
            machine_skip_reason: A non-None reason the machine cannot host the probe (insufficient VRAM
                or disk, missing model, ``--skip-downloads`` on a network probe, ``--only`` deselection),
                computed by the resource gate; None when the machine can host it.
        """
        if self._aborted_reason is not None:
            return ProbeDecision(action=ProbeAction.STOP, reason=self._aborted_reason)

        if machine_skip_reason is not None:
            return ProbeDecision(action=ProbeAction.SKIP, reason=machine_skip_reason)

        unmet = [required for required in probe.requires if required not in self._proven]
        if unmet:
            unmet_labels = ", ".join(required.slug for required in unmet)
            return ProbeDecision(
                action=ProbeAction.SKIP,
                reason=f"requires {unmet_labels}, not proven",
            )

        return ProbeDecision(action=ProbeAction.RUN)

    def record(self, probe: CapabilityProbe, result: CapabilityProbeResult) -> None:
        """Fold a probe's result into the proven/disproven/crashed sets and the catastrophe latch."""
        if result.verdict is CapabilityVerdict.PROVEN:
            self._proven.add(probe.capability)
        elif result.verdict is CapabilityVerdict.DISPROVEN:
            self._disproven.add(probe.capability)
        elif result.verdict is CapabilityVerdict.CRASHED:
            self._crashed.add(probe.capability)

        if self._abort_on_catastrophe and self._aborted_reason is None and self._is_catastrophic(probe, result):
            self._aborted_reason = (
                f"run aborted after catastrophic failure in {probe.probe_id} ({result.verdict}): "
                "the worker stack is shared across all probes, so a fundamental failure repeats; "
                "skipping all remaining probes"
            )

    def _is_catastrophic(self, probe: CapabilityProbe, result: CapabilityProbeResult) -> bool:
        """Whether an outcome means the shared worker stack is broken (so the run should abort).

        A crash or hang is always catastrophic (the worker process itself fell over). A baseline probe
        that completed zero jobs while its processes crashed or had to be recovered is too: the simplest
        possible workload could not be served, so nothing heavier will be. A merely slow failure
        (``DISPROVEN`` with work done) is not: it only disproves its own capability.
        """
        if result.verdict is CapabilityVerdict.CRASHED:
            return True
        if result.verdict is CapabilityVerdict.DISPROVEN and probe.establishes_baseline:
            completed = result.harness.num_jobs_completed if result.harness is not None else 0
            fell_over = any(finding.kind in _CATASTROPHIC_FINDING_KINDS for finding in result.findings)
            return completed == 0 and fell_over
        return False


__all__ = [
    "CapabilitySupervisor",
    "ProbeAction",
    "ProbeDecision",
]
