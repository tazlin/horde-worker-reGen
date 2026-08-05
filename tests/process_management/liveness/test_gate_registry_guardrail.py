"""The gate registry is complete, well-formed, and tied to the runtime surfaces gates actually use.

A gate that holds work without a declared release path is how a wedge ships: the branch is correct, the
decision is right, and nothing ever asked how the work gets out again. The registry is the declaration; this
module is what keeps it honest.

The completeness check is deliberately structural rather than a source grep. Every gate the worker can be
holding at names itself on one of a small number of enumerable runtime surfaces: the preload pass's
``AdmissionDecision``, the pop coroutine's ``PopGate`` stamp, the whole-card churn governors, the dispatch
attribution buckets, the service-lane ``PauseOwner``, the self-throttle ``PopPauseOwner``, and
``RecoveryParkReason``, and the scheduler-side ``SchedulerBudget``. Each of those is a closed ``StrEnum`` in
production code, so a new gate has to take a member, and a member without a registry entry fails here. A
heuristic that scanned for defer-shaped branches would miss the ones spelled differently and flag the ones
that are not gates; enumerating the surfaces the gates already have to use cannot.

The scheduler budgets are the surface that had to be built rather than found: the affinity line-skip window
and the whole-card minimum hold hold work on a clock without returning any admission verdict, so each stamps
its ``SchedulerBudget`` member on the line disclosing its engagement. A third budget therefore has to take a
member, which brings it here.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.config.worker_state import (
    PopGate,
    PopPauseOwner,
    RecoveryParkReason,
)
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
from horde_worker_regen.process_management.liveness.gate_registry import (
    GATE_REGISTRY,
    GateEntry,
    GateKind,
    GateSurface,
    entry_for,
    registered_keys,
)
from horde_worker_regen.process_management.scheduling.dispatch_affinity import (
    _AFFINITY_BUDGET_MAX_SECONDS,
    AffinitySkipState,
    affinity_skip_disclosure,
    record_affinity_skip,
)
from horde_worker_regen.process_management.scheduling.governance.preload_admission import AdmissionDecision
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    _ESTABLISH_WINDOW_SECONDS,
    _GRACE_BUDGET_WINDOW_SECONDS,
    _MIN_HOLD_SECONDS,
    WholeCardGovernor,
    WholeCardResidencyLedger,
)
from horde_worker_regen.process_management.scheduling.scheduler_budget import SchedulerBudget
from horde_worker_regen.process_management.scheduling.slot_duty import SlotDutyBucket

_HOLDS: tuple[GateEntry, ...] = tuple(entry for entry in GATE_REGISTRY if entry.kind is GateKind.HOLD)
"""Every entry that defers work still expected to be served."""

_OUTCOMES: tuple[GateEntry, ...] = tuple(entry for entry in GATE_REGISTRY if entry.kind is GateKind.OUTCOME)
"""Every entry that concludes work rather than holding it."""


def _entry_id(entry: GateEntry) -> str:
    """Return the parametrisation id naming one entry's surface and key."""
    return f"{entry.surface.value}.{entry.key}"


def _surface_members() -> list[tuple[GateSurface, frozenset[str]]]:
    """Return each enumerable surface with the set of key values production code can produce for it."""
    return [
        (GateSurface.PRELOAD_ADMISSION, frozenset(member.value for member in AdmissionDecision)),
        (GateSurface.POP_GATE, frozenset(member.value for member in PopGate)),
        (GateSurface.WHOLE_CARD_GOVERNOR, frozenset(member.value for member in WholeCardGovernor)),
        (GateSurface.DISPATCH_STALL, frozenset(member.value for member in SlotDutyBucket)),
        (GateSurface.LANE_PAUSE, frozenset(member.value for member in PauseOwner)),
        (GateSurface.POP_PAUSE, frozenset(member.value for member in PopPauseOwner)),
        (GateSurface.RECOVERY_PARK, frozenset(member.value for member in RecoveryParkReason)),
        (GateSurface.SCHEDULER_BUDGET, frozenset(member.value for member in SchedulerBudget)),
    ]


class TestEveryGateIsRegistered:
    """No gate reaches production without declaring how the work it holds gets out again."""

    @pytest.mark.parametrize(
        ("surface", "runtime_keys"),
        _surface_members(),
        ids=[surface.value for surface, _keys in _surface_members()],
    )
    def test_every_runtime_key_on_a_surface_has_an_entry(
        self,
        surface: GateSurface,
        runtime_keys: frozenset[str],
    ) -> None:
        """Adding a member to a gate-bearing enum without registering its release path fails here."""
        missing = sorted(runtime_keys - registered_keys(surface))
        assert not missing, (
            f"{surface.value} gates with no registry entry: {missing}. Each holds or concludes work, so each "
            "must declare what engages it, what releases it, and the bound or backstop that covers it, in "
            "horde_worker_regen/process_management/liveness/gate_registry.py."
        )

    @pytest.mark.parametrize(
        ("surface", "runtime_keys"),
        _surface_members(),
        ids=[surface.value for surface, _keys in _surface_members()],
    )
    def test_no_entry_names_a_key_production_cannot_produce(
        self,
        surface: GateSurface,
        runtime_keys: frozenset[str],
    ) -> None:
        """A stale entry is as misleading as a missing one: it promises a release path for nothing."""
        orphaned = sorted(registered_keys(surface) - runtime_keys)
        assert not orphaned, f"{surface.value} registry entries whose key no longer exists in production: {orphaned}."

    def test_every_surface_is_enumerated(self) -> None:
        """The completeness guarantee covers every surface: none is hand-listed and so unprotected."""
        enumerated = {surface for surface, _keys in _surface_members()}
        assert set(GateSurface) - enumerated == set(), (
            "a new gate surface must be enumerated here, against the closed runtime set its keys come from, "
            "or the guardrail cannot tell when a gate ships without a declared release path."
        )

    def test_keys_are_unique_within_a_surface(self) -> None:
        """Two entries for one key would let a stale declaration hide behind a current one."""
        seen: set[tuple[GateSurface, str]] = set()
        duplicates: list[tuple[GateSurface, str]] = []
        for entry in GATE_REGISTRY:
            identity = (entry.surface, entry.key)
            if identity in seen:
                duplicates.append(identity)
            seen.add(identity)
        assert not duplicates, f"duplicate registry entries: {duplicates}"


class TestEveryHoldDeclaresAReleasePath:
    """A declaration that omits the release or the bound is not a declaration."""

    @pytest.mark.parametrize("entry", GATE_REGISTRY, ids=[_entry_id(e) for e in GATE_REGISTRY])
    def test_every_entry_names_its_owner_engagement_and_observability(self, entry: GateEntry) -> None:
        """An entry nobody can trace to code, or confirm at runtime, cannot be attacked or acted on."""
        assert entry.subsystem, f"{entry.key} does not name the module that owns the decision"
        assert entry.engaged_by, f"{entry.key} does not say what puts it in force"
        assert entry.observable_at, f"{entry.key} does not say where an engagement is visible at runtime"
        assert entry.released_by, f"{entry.key} does not say what takes it out of force"

    @pytest.mark.parametrize("entry", _HOLDS, ids=[_entry_id(e) for e in _HOLDS])
    def test_every_hold_declares_a_bound_or_a_backstop(self, entry: GateEntry) -> None:
        """A hold with neither a clock nor an escalation is a wedge waiting for the state to compose."""
        assert entry.bound_seconds is not None or entry.backstop, (
            f"{entry.surface.value}.{entry.key} holds work with no declared bound and no named backstop. Either "
            "it resolves on a clock, or something else escalates when it does not."
        )

    @pytest.mark.parametrize("entry", GATE_REGISTRY, ids=[_entry_id(e) for e in GATE_REGISTRY])
    def test_a_numeric_bound_names_the_constant_it_came_from(self, entry: GateEntry) -> None:
        """An unattributed figure drifts from the constant it copied without anything noticing."""
        if entry.bound_seconds is None:
            return
        assert entry.bound_source, f"{entry.key} states a bound of {entry.bound_seconds}s without naming its source"
        assert entry.bound_seconds > 0, f"{entry.key} states a non-positive bound"

    @pytest.mark.parametrize("entry", _OUTCOMES, ids=[_entry_id(e) for e in _OUTCOMES])
    def test_an_outcome_says_why_it_cannot_hold_work(self, entry: GateEntry) -> None:
        """An entry claiming not to be a hold has to say so, or it is a hold with the fields left blank."""
        assert "not a hold" in entry.released_by, (
            f"{entry.surface.value}.{entry.key} is registered as an outcome but does not state why it cannot "
            "hold work; if it can, register it as a hold with a release path."
        )


class TestEachSchedulerBudgetStampsItsMember:
    """The budgets are only enumerable because each puts its member where a reader can see it.

    Every other surface is a decision the code already had to name; these two hold work on a clock and would
    otherwise stamp nothing. If a disclosure stops carrying its member, the enumeration above still passes
    while the runtime surface it claims to describe has gone, so the stamps are pinned here.
    """

    def test_the_line_skip_disclosure_names_its_budget(self) -> None:
        """The committed-skip line attributes the head's spent queue position to the line-skip budget."""
        state = record_affinity_skip(AffinitySkipState(), "head", 100.0)
        disclosure = affinity_skip_disclosure(state, now=110.0, budget_seconds=45.0, max_skips=6)
        assert SchedulerBudget.AFFINITY_LINE_SKIP.value in disclosure
        assert "1/6" in disclosure, "the count bound is half of what ends the bypass"
        assert "35s" in disclosure, "the wall-clock bound is the other half"

    def test_the_min_hold_disclosure_names_its_budget_only_while_it_holds(self) -> None:
        """The floor speaks while it is in force and says nothing once it has lapsed."""
        ledger = WholeCardResidencyLedger()
        ledger.record_grant(
            None,
            model="heavy",
            forecast=None,
            cooldown_until=0.0,
            now=0.0,
        )
        disclosure = ledger.min_hold_disclosure(None, now=_MIN_HOLD_SECONDS - 1.0)
        assert disclosure is not None
        assert SchedulerBudget.WHOLE_CARD_MIN_HOLD.value in disclosure
        assert "heavy" in disclosure
        assert ledger.min_hold_disclosure(None, now=_MIN_HOLD_SECONDS) is None


class TestDeclaredBoundsMatchTheProductionConstants:
    """A declared figure that has drifted from the constant it describes is worse than none."""

    @pytest.mark.parametrize(
        ("surface", "key", "constant"),
        [
            (GateSurface.WHOLE_CARD_GOVERNOR, "establish_rate", _ESTABLISH_WINDOW_SECONDS),
            (GateSurface.WHOLE_CARD_GOVERNOR, "grace_budget", _GRACE_BUDGET_WINDOW_SECONDS),
            (GateSurface.SCHEDULER_BUDGET, "whole_card_min_hold", _MIN_HOLD_SECONDS),
            (GateSurface.SCHEDULER_BUDGET, "affinity_line_skip", _AFFINITY_BUDGET_MAX_SECONDS),
        ],
    )
    def test_the_registered_bound_equals_the_constant(
        self,
        surface: GateSurface,
        key: str,
        constant: float,
    ) -> None:
        """Retuning a window updates the declaration with it."""
        entry = entry_for(surface, key)
        assert entry is not None
        assert entry.bound_seconds == constant, (
            f"{key} declares {entry.bound_seconds}s but {entry.bound_source} is now {constant}s"
        )
