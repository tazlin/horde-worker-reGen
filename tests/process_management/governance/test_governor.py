"""Unit tests for the ResourceGovernor tick against a fake host.

The governor is pure orchestration: measure a verdict, build a snapshot, decide, execute. A fake host
records what was executed, so these tests pin the tick contract without a scheduler.
"""

from __future__ import annotations

from horde_worker_regen.process_management.resources.resource_budget import (
    RamPressureVerdict,
    assess_ram_pressure,
)
from horde_worker_regen.process_management.scheduling.governance import (
    CardProcessSnapshot,
    ClearProcessDraining,
    EvictIdleModels,
    GovernanceAction,
    HostMemorySnapshot,
    MarkProcessDraining,
    RecycleProcess,
    ResourceGovernor,
    RestoreCardProcess,
    RestoreWorkerProcess,
    SetPopHold,
)
from tests.process_management.governance.test_ram_governor import _CEILING_MB, _slot, _snapshot

_TOTAL_RAM_MB = 64000.0


class _FakeHost:
    """A governance host that returns a canned snapshot and records executed actions."""

    def __init__(self, snapshot: HostMemorySnapshot) -> None:
        self.snapshot = snapshot
        self.executed: list[GovernanceAction] = []

    def _ram_pressure_verdict(self) -> RamPressureVerdict:
        return self.snapshot.verdict

    def _build_host_memory_snapshot(self, verdict: RamPressureVerdict) -> HostMemorySnapshot:
        assert verdict is self.snapshot.verdict, "the tick must snapshot with the verdict it measured"
        return self.snapshot

    def _execute_governance_actions(self, actions: list[GovernanceAction]) -> None:
        self.executed.extend(actions)


class TestResourceGovernorTick:
    """One tick measures once, decides both regimes, and executes through the host."""

    def test_healthy_tick_clears_the_hold_and_reports_no_pressure(self) -> None:
        """A healthy host gets exactly the cleared pop hold and no degrade actions."""
        host = _FakeHost(_snapshot())
        governor = ResourceGovernor(host=host)

        under_pressure = governor.tick()

        assert under_pressure is False
        assert host.executed == [SetPopHold(active=False)]
        assert governor.last_ram_verdict is host.snapshot.verdict

    def test_pressured_tick_executes_the_degrade_response(self) -> None:
        """A pressured host reports pressure and executes the degrade actions."""
        host = _FakeHost(_snapshot(available_mb=500.0))
        governor = ResourceGovernor(host=host)

        under_pressure = governor.tick()

        assert under_pressure is True
        assert host.executed[0] == SetPopHold(active=True)
        assert EvictIdleModels() in host.executed

    def test_recovered_tick_restores_shed_cards_in_the_same_pass(self) -> None:
        """A recovered host's tick also carries the shed-card restore, so no separate call site exists."""
        snapshot = _snapshot(
            shed_card_indices=frozenset({0}),
            cards=(
                # One shed card below its plan, with ample headroom to grow back.
                CardProcessSnapshot(
                    device_index=0,
                    loaded_process_count=1,
                    busy_process_count=0,
                    planned_process_count=2,
                ),
            ),
        )
        host = _FakeHost(snapshot)
        governor = ResourceGovernor(host=host)

        governor.tick()

        assert RestoreCardProcess(device_index=0, target_count=2, planned_count=2) in host.executed

    def test_recovered_tick_restores_worker_wide_shed_in_the_same_pass(self) -> None:
        """A recovered single-GPU host's tick grows back worker-wide RAM-pressure shedding."""
        snapshot = _snapshot(
            loaded_worker_process_count=2,
            worker_shed_planned_process_count=4,
            worker_shed_process_count=2,
        )
        host = _FakeHost(snapshot)
        governor = ResourceGovernor(host=host)

        governor.tick()

        assert RestoreWorkerProcess(target_count=3, planned_count=4) in host.executed

    def test_drain_marked_under_pressure_resolves_after_recovery(self) -> None:
        """A drain that outlives its pressure episode is recycled, unmarked, and the pop hold released.

        Sequence: the floor trips while a ballooned process is busy, so it is marked draining rather than
        recycled. The degrade response then lifts the pressure before that job finishes. On the recovered
        host the drained process (now idle, still over the ceiling) must still be recycled and unmarked;
        otherwise the mark keeps the pop hold engaged and blocks shed restore for the rest of the session.
        """

        class _StatefulHost(_FakeHost):
            """A fake host whose executed Mark/Clear/Recycle actions mutate the governor's drain set."""

            governor: ResourceGovernor

            def _execute_governance_actions(self, actions: list[GovernanceAction]) -> None:
                super()._execute_governance_actions(actions)
                draining = self.governor.ram_state.draining_process_ids
                for action in actions:
                    if isinstance(action, MarkProcessDraining):
                        draining.add(action.process_id)
                    elif isinstance(action, (ClearProcessDraining, RecycleProcess)):
                        draining.discard(action.process_id)

        busy_balloon = _slot(4, resident_ram_mb=19000.0, is_busy=True)
        host = _StatefulHost(_snapshot(available_mb=500.0, inference_slots=(busy_balloon,), in_flight_job_count=1))
        governor = ResourceGovernor(host=host)
        host.governor = governor

        assert governor.tick() is True
        assert governor.ram_state.draining_process_ids == {4}

        # RAM recovered above the floor and the drained process finished its job, idling over the ceiling.
        idle_balloon = _slot(4, resident_ram_mb=19000.0)
        host.snapshot = _snapshot(
            inference_slots=(idle_balloon,),
            draining_process_ids=frozenset(governor.ram_state.draining_process_ids),
        )
        host.executed.clear()
        assert governor.tick() is False
        assert host.executed[0] == SetPopHold(active=True), "the hold protects the drain until it resolves"
        assert RecycleProcess(process_id=4, resident_ram_mb=19000.0, ceiling_mb=_CEILING_MB) in host.executed
        assert governor.ram_state.draining_process_ids == set()

        # With the drain resolved, the next healthy tick releases the pop hold and intake reopens.
        host.snapshot = _snapshot(inference_slots=(_slot(4, resident_ram_mb=2000.0),))
        host.executed.clear()
        governor.tick()
        assert host.executed == [SetPopHold(active=False)]

    def test_verdict_is_cached_for_within_cycle_readers(self) -> None:
        """The tick's verdict is retained so per-job gates in the same cycle act on one reading."""
        verdict = assess_ram_pressure(40000.0, _TOTAL_RAM_MB)
        host = _FakeHost(_snapshot())
        governor = ResourceGovernor(host=host)
        assert governor.last_ram_verdict is None

        governor.tick()

        assert governor.last_ram_verdict is not None
        assert governor.last_ram_verdict.floor_mb == verdict.floor_mb
