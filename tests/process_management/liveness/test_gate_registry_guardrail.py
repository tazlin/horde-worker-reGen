"""The gate registry is complete, well-formed, and tied to the runtime surfaces gates actually use.

A gate that holds work without a declared release path is how a wedge ships: the branch is correct, the
decision is right, and nothing ever asked how the work gets out again. The registry is the declaration; this
module is what keeps it honest.

The completeness check is deliberately structural rather than a source grep. Every gate the worker can be
holding at names itself on one of a small number of enumerable runtime surfaces: the preload pass's
``AdmissionDecision``, the pop coroutine's ``PopGate`` stamp, the whole-card churn governors, the dispatch
attribution buckets, the service-lane ``PauseOwner``, the self-throttle ``PopPauseOwner``, and
``RecoveryParkReason``. Each of those is a closed ``StrEnum`` in production code, so a new gate has to take a
member, and a member without a registry entry fails here. A heuristic that scanned for defer-shaped branches
would miss the ones spelled differently and flag the ones that are not gates; enumerating the surfaces the
gates already have to use cannot.

The residual gap is stated rather than papered over: ``SCHEDULER_BUDGET`` carries the gates that hold work
without stamping a name on any enumerable surface (the affinity line-skip window, the whole-card minimum
hold). Those are hand-listed, and nothing here can detect a further one. Closing that gap means giving those
budgets a named runtime stamp of their own.
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
from horde_worker_regen.process_management.scheduling.dispatch_affinity import _AFFINITY_BUDGET_MAX_SECONDS
from horde_worker_regen.process_management.scheduling.governance.preload_admission import AdmissionDecision
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    _ESTABLISH_WINDOW_SECONDS,
    _GRACE_BUDGET_WINDOW_SECONDS,
    _MIN_HOLD_SECONDS,
    WholeCardGovernor,
)
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

    def test_every_surface_is_either_enumerated_or_declared_unenumerable(self) -> None:
        """The completeness guarantee covers every surface except the one documented as hand-listed."""
        enumerated = {surface for surface, _keys in _surface_members()}
        unenumerated = set(GateSurface) - enumerated
        assert unenumerated == {GateSurface.SCHEDULER_BUDGET}, (
            "a new gate surface must either be enumerated here (so the guardrail covers it) or be a "
            "deliberate hand-listed exception this test names."
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
