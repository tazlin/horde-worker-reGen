"""Unit tests for the pure RAM-governance decision functions.

Every test builds a :class:`HostMemorySnapshot` directly and asserts on the returned actions: no
scheduler, no process map, no monkeypatching. The scheduler-integrated behavior (executing these actions
against live processes) is covered by the regression suites under ``tests/process_management/regressions``.
"""

from __future__ import annotations

from horde_worker_regen.process_management.resources.resource_budget import assess_ram_pressure
from horde_worker_regen.process_management.scheduling.governance import (
    CardProcessSnapshot,
    ClearProcessDraining,
    EvictIdleModels,
    GovernanceAction,
    HostMemorySnapshot,
    InferenceSlotSnapshot,
    MarkProcessDraining,
    PausePops,
    RecycleProcess,
    ReduceCardProcesses,
    ReduceWorkerProcesses,
    RestoreCardProcess,
    SetPopHold,
    StopTrackingShedCard,
    decide_degrade_response,
    decide_over_ceiling_reclaim,
    decide_pop_hold,
    decide_pressure_governance,
    decide_process_reduction,
    decide_shed_card_restore,
)

_TOTAL_RAM_MB = 64000.0
_HEALTHY_AVAILABLE_MB = 40000.0
_CRITICAL_AVAILABLE_MB = 500.0
_CEILING_MB = 18432.0
_NOW = 1_000_000.0


def _snapshot(
    *,
    available_mb: float = _HEALTHY_AVAILABLE_MB,
    pop_pause_active: bool = False,
    pop_pause_until: float = 0.0,
    pop_hold_margin_mb: float = 4096.0,
    per_process_ceiling_mb: float | None = _CEILING_MB,
    multi_gpu_routing_active: bool = False,
    in_flight_job_count: int = 0,
    loaded_worker_process_count: int = 0,
    inference_slots: tuple[InferenceSlotSnapshot, ...] = (),
    cards: tuple[CardProcessSnapshot, ...] = (),
    draining_process_ids: frozenset[int] = frozenset(),
    shed_card_indices: frozenset[int] = frozenset(),
    restore_headroom_mb: float = 30000.0,
    per_context_ram_estimate_mb: float = 4096.0,
) -> HostMemorySnapshot:
    """Build a snapshot with healthy defaults so each test states only what it varies."""
    return HostMemorySnapshot(
        verdict=assess_ram_pressure(available_mb, _TOTAL_RAM_MB),
        now=_NOW,
        pop_pause_active=pop_pause_active,
        pop_pause_until=pop_pause_until,
        pop_hold_margin_mb=pop_hold_margin_mb,
        per_process_ceiling_mb=per_process_ceiling_mb,
        multi_gpu_routing_active=multi_gpu_routing_active,
        in_flight_job_count=in_flight_job_count,
        loaded_worker_process_count=loaded_worker_process_count,
        inference_slots=inference_slots,
        cards=cards,
        draining_process_ids=draining_process_ids,
        shed_card_indices=shed_card_indices,
        restore_headroom_mb=restore_headroom_mb,
        per_context_ram_estimate_mb=per_context_ram_estimate_mb,
    )


def _slot(
    process_id: int,
    *,
    resident_ram_mb: float,
    is_busy: bool = False,
    device_index: int = 0,
) -> InferenceSlotSnapshot:
    """Build one inference-slot snapshot with the given resident footprint."""
    return InferenceSlotSnapshot(
        process_id=process_id,
        device_index=device_index,
        resident_ram_mb=resident_ram_mb,
        is_busy=is_busy,
    )


def _only(actions: list[GovernanceAction], kind: type) -> list[GovernanceAction]:
    """Filter ``actions`` to instances of ``kind`` (order preserved)."""
    return [action for action in actions if isinstance(action, kind)]


class TestPopHold:
    """The soft pop hold engages before the hard floor and while a drain is in flight."""

    def test_hold_clear_on_a_roomy_host(self) -> None:
        """A host with ample available RAM keeps the pop hold clear."""
        assert decide_pop_hold(_snapshot()) == SetPopHold(active=False)

    def test_hold_engages_within_the_margin_of_the_floor(self) -> None:
        """Available RAM inside the margin above the floor engages the hold before the floor trips."""
        floor_mb = assess_ram_pressure(None, _TOTAL_RAM_MB).floor_mb
        assert decide_pop_hold(_snapshot(available_mb=floor_mb + 100.0)) == SetPopHold(active=True)

    def test_hold_engages_under_the_floor(self) -> None:
        """A host under the danger floor always holds pops."""
        assert decide_pop_hold(_snapshot(available_mb=_CRITICAL_AVAILABLE_MB)) == SetPopHold(active=True)

    def test_hold_engages_while_a_process_is_draining(self) -> None:
        """A drain in flight holds pops even when RAM itself reads healthy."""
        snapshot = _snapshot(draining_process_ids=frozenset({3}))
        assert decide_pop_hold(snapshot) == SetPopHold(active=True)


class TestDegradeResponse:
    """A host under its danger floor pauses pops, evicts idle models, and sheds footprint."""

    def test_healthy_host_gets_no_degrade_actions(self) -> None:
        """A host above its floor is not degraded."""
        assert decide_degrade_response(_snapshot()) == []

    def test_pressured_host_pauses_evicts_and_reduces(self) -> None:
        """The full degrade response arms the pop pause, evicts idle models, and sheds processes."""
        snapshot = _snapshot(available_mb=_CRITICAL_AVAILABLE_MB, loaded_worker_process_count=3)
        actions = decide_degrade_response(snapshot)
        pauses = _only(actions, PausePops)
        assert len(pauses) == 1
        assert isinstance(pauses[0], PausePops)
        assert pauses[0].until_time > _NOW
        assert _only(actions, EvictIdleModels) == [EvictIdleModels()]
        assert _only(actions, ReduceWorkerProcesses) == [ReduceWorkerProcesses(target_count=1)]

    def test_existing_longer_pause_is_not_rearmed(self) -> None:
        """An already-armed pause that outlasts the new window is left alone (no duplicate announcements)."""
        snapshot = _snapshot(
            available_mb=_CRITICAL_AVAILABLE_MB,
            pop_pause_active=True,
            pop_pause_until=_NOW + 3600.0,
        )
        assert _only(decide_degrade_response(snapshot), PausePops) == []

    def test_pressure_governance_always_includes_the_pop_hold(self) -> None:
        """The per-tick entry point sets the pop hold on both healthy and pressured hosts."""
        healthy = decide_pressure_governance(_snapshot())
        assert healthy == [SetPopHold(active=False)]
        pressured = decide_pressure_governance(_snapshot(available_mb=_CRITICAL_AVAILABLE_MB))
        assert pressured[0] == SetPopHold(active=True)
        assert len(pressured) > 1


class TestProcessReduction:
    """The reduction sheds toward in-flight need: per card on multi-GPU, worker-wide otherwise."""

    def test_single_gpu_reduces_worker_wide_toward_in_flight_need(self) -> None:
        """The worker-wide pool reduces toward the in-flight job count, at least one below current."""
        snapshot = _snapshot(loaded_worker_process_count=3, in_flight_job_count=2)
        assert decide_process_reduction(snapshot) == [ReduceWorkerProcesses(target_count=2)]

    def test_single_context_pool_is_never_reduced(self) -> None:
        """A pool already at one context has nothing to shed."""
        assert decide_process_reduction(_snapshot(loaded_worker_process_count=1)) == []

    def test_multi_gpu_reduces_each_card_but_never_below_one(self) -> None:
        """Each driven card reduces toward its own busy count, keeping at least one context per card."""
        cards = (
            CardProcessSnapshot(device_index=0, loaded_process_count=2, busy_process_count=0, planned_process_count=2),
            CardProcessSnapshot(device_index=1, loaded_process_count=2, busy_process_count=1, planned_process_count=2),
        )
        snapshot = _snapshot(multi_gpu_routing_active=True, cards=cards)
        assert decide_process_reduction(snapshot) == [
            ReduceCardProcesses(device_index=0, target_count=1),
            ReduceCardProcesses(device_index=1, target_count=1),
        ]

    def test_multi_gpu_spares_a_card_already_at_one_context(self) -> None:
        """A card already at one context is never reduced further."""
        cards = (
            CardProcessSnapshot(device_index=0, loaded_process_count=1, busy_process_count=0, planned_process_count=2),
        )
        snapshot = _snapshot(multi_gpu_routing_active=True, cards=cards)
        assert decide_process_reduction(snapshot) == []


class TestOverCeilingReclaim:
    """One over-ceiling process per tick: recycled when idle, drained when busy."""

    def test_disabled_ceiling_clears_every_draining_mark(self) -> None:
        """Disabling the ceiling clears all draining bookkeeping."""
        snapshot = _snapshot(per_process_ceiling_mb=None, draining_process_ids=frozenset({2, 5}))
        assert decide_over_ceiling_reclaim(snapshot) == [
            ClearProcessDraining(process_id=2),
            ClearProcessDraining(process_id=5),
        ]

    def test_idle_over_ceiling_process_is_recycled(self) -> None:
        """An idle process over the ceiling is recycled immediately."""
        slots = (_slot(1, resident_ram_mb=30000.0), _slot(2, resident_ram_mb=5000.0))
        actions = decide_over_ceiling_reclaim(_snapshot(inference_slots=slots))
        assert actions == [RecycleProcess(process_id=1, resident_ram_mb=30000.0, ceiling_mb=_CEILING_MB)]

    def test_busy_over_ceiling_process_is_drained_not_recycled(self) -> None:
        """A busy process over the ceiling is marked draining so its in-flight job finishes."""
        slots = (_slot(1, resident_ram_mb=30000.0, is_busy=True),)
        actions = decide_over_ceiling_reclaim(_snapshot(inference_slots=slots))
        assert actions == [MarkProcessDraining(process_id=1, resident_ram_mb=30000.0, ceiling_mb=_CEILING_MB)]

    def test_already_draining_process_is_not_remarked(self) -> None:
        """A process already draining is not marked again (the announcement fires once)."""
        slots = (_slot(1, resident_ram_mb=30000.0, is_busy=True),)
        snapshot = _snapshot(inference_slots=slots, draining_process_ids=frozenset({1}))
        assert decide_over_ceiling_reclaim(snapshot) == []

    def test_only_the_largest_offender_is_acted_on(self) -> None:
        """With several offenders, only the largest is reclaimed this tick (never empty every card at once)."""
        slots = (_slot(1, resident_ram_mb=20000.0), _slot(2, resident_ram_mb=30000.0))
        actions = decide_over_ceiling_reclaim(_snapshot(inference_slots=slots))
        assert actions == [RecycleProcess(process_id=2, resident_ram_mb=30000.0, ceiling_mb=_CEILING_MB)]

    def test_fallen_under_ceiling_process_is_cleared_from_draining(self) -> None:
        """A process that shrank back under the ceiling has its draining mark cleared."""
        slots = (_slot(1, resident_ram_mb=5000.0),)
        snapshot = _snapshot(inference_slots=slots, draining_process_ids=frozenset({1}))
        assert decide_over_ceiling_reclaim(snapshot) == [ClearProcessDraining(process_id=1)]


class TestShedCardRestore:
    """Shed cards grow back one context per tick, RAM-gated, once the host recovers."""

    def _card(
        self, device_index: int, *, loaded: int = 1, planned: int = 2, held: bool = False
    ) -> CardProcessSnapshot:
        """Build a card snapshot with the given counts."""
        return CardProcessSnapshot(
            device_index=device_index,
            loaded_process_count=loaded,
            busy_process_count=0,
            planned_process_count=planned,
            held_by_whole_card_residency=held,
        )

    def test_no_shed_cards_means_no_actions(self) -> None:
        """Without a pressure episode there is nothing to restore."""
        assert decide_shed_card_restore(_snapshot()) == []

    def test_no_restore_while_under_pressure(self) -> None:
        """Shed cards are not grown back while the host is still below its floor."""
        snapshot = _snapshot(
            available_mb=_CRITICAL_AVAILABLE_MB,
            shed_card_indices=frozenset({0}),
            cards=(self._card(0),),
        )
        assert decide_shed_card_restore(snapshot) == []

    def test_no_restore_while_the_pop_pause_is_still_armed(self) -> None:
        """The restore waits for the self-throttle pop pause to lapse."""
        snapshot = _snapshot(
            shed_card_indices=frozenset({0}),
            cards=(self._card(0),),
            pop_pause_active=True,
            pop_pause_until=_NOW + 10.0,
        )
        assert decide_shed_card_restore(snapshot) == []

    def test_recovered_host_restores_one_context_toward_plan(self) -> None:
        """A recovered host grows a shed card back by one context toward its plan."""
        snapshot = _snapshot(shed_card_indices=frozenset({0}), cards=(self._card(0),))
        assert decide_shed_card_restore(snapshot) == [
            RestoreCardProcess(device_index=0, target_count=2, planned_count=2),
        ]

    def test_residency_held_card_is_left_to_its_own_restore(self) -> None:
        """A card a whole-card residency holds down is untracked here, not regrown."""
        snapshot = _snapshot(shed_card_indices=frozenset({0}), cards=(self._card(0, held=True),))
        assert decide_shed_card_restore(snapshot) == [StopTrackingShedCard(device_index=0)]

    def test_card_already_at_plan_stops_being_tracked(self) -> None:
        """A card back at its planned count leaves the episode."""
        snapshot = _snapshot(shed_card_indices=frozenset({0}), cards=(self._card(0, loaded=2),))
        assert decide_shed_card_restore(snapshot) == [StopTrackingShedCard(device_index=0)]

    def test_unknown_card_stops_being_tracked(self) -> None:
        """A shed record for a card the plan no longer drives is dropped."""
        snapshot = _snapshot(shed_card_indices=frozenset({7}), cards=(self._card(0),))
        assert decide_shed_card_restore(snapshot) == [StopTrackingShedCard(device_index=7)]

    def test_restore_defers_without_headroom_for_another_context(self) -> None:
        """A card stays pending while measured RAM cannot hold another resident working set."""
        snapshot = _snapshot(
            shed_card_indices=frozenset({0}),
            cards=(self._card(0),),
            restore_headroom_mb=1000.0,
            per_context_ram_estimate_mb=22000.0,
        )
        assert decide_shed_card_restore(snapshot) == []

    def test_two_cards_cannot_both_spend_headroom_that_exists_once(self) -> None:
        """Each grant charges the estimate against remaining headroom, so grants never double-spend it."""
        snapshot = _snapshot(
            shed_card_indices=frozenset({0, 1}),
            cards=(self._card(0), self._card(1)),
            restore_headroom_mb=5000.0,
            per_context_ram_estimate_mb=4096.0,
        )
        assert decide_shed_card_restore(snapshot) == [
            RestoreCardProcess(device_index=0, target_count=2, planned_count=2),
        ]
