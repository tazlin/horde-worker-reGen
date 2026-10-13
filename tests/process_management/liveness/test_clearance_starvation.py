"""A staged waiter's starvation clock under reclaim, which only realised room resets.

A staged job waits at clearance until its weights fit the measured room, and the arbiter grants it one measured
load attempt once it has been starved for the probe delay. The clock measures how long the waiter has been held
with nothing left that reclaim can give, so a rung the actuator accepts and that frees nothing leaves it running,
and a rung whose room arrives restarts it from that moment. Each row re-sends the idle lanes' allocator-cache
release on every pass, standing for any rung the ladder keeps offering. One row's release returns nothing, and
the other's starts returning room partway through the hold.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.resources.run_metrics import DecisionKind, DecisionVerdict
from horde_worker_regen.process_management.scheduling.clearance_lease import CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import (
    _CARD_24GB,
    _FLUX,
    _SD15,
    _SDXL,
    _DecisionRecord,
    _DispatchWorld,
)

pytestmark = pytest.mark.closed_loop

_TICK_SECONDS = 0.5
_MAX_TICKS = 400
"""Ample for the head's lease-acquire timeout and the sample that follows it."""
_IDLE_LANE_MODELS = {1: _SDXL, 2: _SD15}
"""The idle lanes and the model each holds in RAM. The Flux head stages on lane 0."""
_UNRETAINED_COMPONENT_MB = 300.0
"""Split text-encoder and VAE allocations (MB) each idle lane keeps on the device, just over the release threshold,
so each lane stays a release target while the two together cannot close the head's deficit."""
_FOREIGN_MB = 14000.0
"""Device memory other programs hold, sized so the staged Flux head is short by more than the idle lanes hold."""
_ROOM_ARRIVES_AFTER_SECONDS = 6.0
"""How far into the hold the realising row's release starts returning the components. It is inside the probe delay,
so a clock left running from the first send would come due before one restarted when the room arrived."""
_MEASURED_ATTEMPT = "granted a measured load attempt"


def _resend_every_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every idle lane a release target on every pass, whatever its previous release returned."""

    def never_withheld(self: InferenceScheduler, process_info: HordeProcessInfo) -> bool:
        return False

    monkeypatch.setattr(InferenceScheduler, "_release_cache_withheld", never_withheld)


async def _run_flux_head(*, room_arrives_after_seconds: float | None) -> tuple[_DispatchWorld, str]:
    """Stage one Flux job on a 24 GB card beside two idle lanes holding unretained components, and run it out.

    The components are pinned, so the release returns nothing, until ``room_arrives_after_seconds`` past the
    head's first clearance hold, from which the next release returns them. None keeps them pinned throughout.
    """
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=1 + len(_IDLE_LANE_MODELS),
        max_threads=1 + len(_IDLE_LANE_MODELS),
        queue_depth=1 + len(_IDLE_LANE_MODELS),
        whole_card_enabled=False,
        closed_loop=True,
        clearance_lease=True,
        tick_seconds=_TICK_SECONDS,
        foreign_vram_mb_by_tick={1: _FOREIGN_MB},
        unretained_components_pinned=True,
    )
    world.seed_resident(0, _FLUX, in_vram=False)
    world.scheduler._process_map[0].last_process_state = HordeProcessState.WAITING_FOR_JOB
    for lane_id, model in _IDLE_LANE_MODELS.items():
        world.seed_resident(lane_id, model, in_vram=False)
        world.seed_served_vram_unload(lane_id)
        world.seed_unretained_components(lane_id, _UNRETAINED_COMPONENT_MB)
    head = make_job_pop_response(_FLUX.name, width=1024, height=1024, ddim_steps=8)
    head_id = str(head.id_)
    await world.pop(head)
    for _ in range(_MAX_TICKS):
        await world.step()
        first_hold = _first_hold_tick(world, head_id)
        if (
            room_arrives_after_seconds is not None
            and first_hold is not None
            and (world.tick - first_hold) * _TICK_SECONDS >= room_arrives_after_seconds
        ):
            world.unretained_components_pinned = False
        if world.completed_jobs >= 1:
            break
    return world, head_id


async def _drive_flux_head(*, room_arrives_after_seconds: float | None) -> tuple[_DispatchWorld, str]:
    """Run :func:`_run_flux_head` and check the shape each row depends on before it reads the outcome."""
    world, head_id = await _run_flux_head(room_arrives_after_seconds=room_arrives_after_seconds)
    assert world.completed_jobs == 1, world.state_dump()
    window = world.job_windows.get(head_id)
    assert window is not None and window.lane_id not in _IDLE_LANE_MODELS, (
        f"precondition: the head must stage on the lane holding no components. {world.state_dump()}"
    )
    first_hold = _first_hold_tick(world, head_id)
    assert first_hold is not None, f"precondition: the head must be held at clearance. {world.state_dump()}"
    first_deficit_mb = _holds(world, head_id)[0].inputs.get("room_deficit_mb")
    assert isinstance(first_deficit_mb, float), f"precondition: the first hold carries no room. {world.state_dump()}"
    assert first_deficit_mb > _UNRETAINED_COMPONENT_MB * len(_IDLE_LANE_MODELS), (
        f"precondition: the head must be short by more than the idle lanes hold, so the release can never close "
        f"the deficit. It was short {first_deficit_mb:.0f}MB"
    )
    resent = [record for record in _holds(world, head_id) if record.inputs.get("reclaim_applied_kinds")]
    assert len(resent) >= 2, f"precondition: the release must be re-sent across the hold, was sent {len(resent)}x"
    return world, head_id


def _holds(world: _DispatchWorld, job_id: str) -> list[_DecisionRecord]:
    return [
        record
        for record in world.decision_records
        if record.decision_kind is DecisionKind.CLEARANCE
        and record.subject == job_id
        and record.verdict in (DecisionVerdict.DEFER, DecisionVerdict.DENY)
    ]


def _first_hold_tick(world: _DispatchWorld, job_id: str) -> int | None:
    holds = _holds(world, job_id)
    return holds[0].tick if holds else None


def _measured_attempt_tick(world: _DispatchWorld, job_id: str) -> int | None:
    grants = [
        record.tick
        for record in world.decision_records
        if record.decision_kind is DecisionKind.CLEARANCE
        and record.subject == job_id
        and record.verdict is DecisionVerdict.ADMIT
        and record.reason == _MEASURED_ATTEMPT
    ]
    return grants[0] if grants else None


def _probe_delay_seconds(world: _DispatchWorld) -> float:
    return world.scheduler._probe_after_seconds(0)


async def test_a_rung_that_realises_nothing_leaves_the_clock_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """A release re-sent on every pass that returns nothing does not hold the head to its lease-acquire timeout.

    An accepted release that frees nothing leaves the head's starvation clock running. Were it to restart the
    clock, the measured attempt would never come due and the head would sample unpriced at its timeout. The
    head takes its measured attempt at the probe delay.
    """
    _resend_every_pass(monkeypatch)
    world, head_id = await _drive_flux_head(room_arrives_after_seconds=None)

    context = "a Flux head beside a release re-sent every pass that returns nothing"
    assert not world.clearance_timeouts, (
        f"{context}: lane(s) {world.clearance_timeouts} sampled through the "
        f"{CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS:.0f}s lease-acquire timeout. {world.state_dump()}"
    )
    first_hold = _first_hold_tick(world, head_id)
    granted = _measured_attempt_tick(world, head_id)
    assert first_hold is not None and granted is not None, f"{context}: no measured attempt. {world.state_dump()}"
    held_seconds = (granted - first_hold) * _TICK_SECONDS
    bound_seconds = _probe_delay_seconds(world) + 2 * _TICK_SECONDS
    assert held_seconds <= bound_seconds, (
        f"{context}: the measured attempt came {held_seconds:.1f}s after the first hold, past the "
        f"{bound_seconds:.1f}s the probe delay allows. {world.state_dump()}"
    )


async def test_a_rung_whose_room_arrives_restarts_the_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A release that starts returning room restarts the head's clock from that moment, so it is not probed early.

    The release returns nothing for the first seconds of the hold and then returns the idle lanes' components.
    That is reclaim working, so the measured attempt waits a whole probe delay from the room's arrival, never
    one counted from the first send.
    """
    _resend_every_pass(monkeypatch)
    world, head_id = await _drive_flux_head(room_arrives_after_seconds=_ROOM_ARRIVES_AFTER_SECONDS)

    context = "a Flux head beside a release whose room arrives partway through the hold"
    assert not world.clearance_timeouts, (
        f"{context}: lane(s) {world.clearance_timeouts} sampled through the lease-acquire timeout. "
        f"{world.state_dump()}"
    )
    first_hold = _first_hold_tick(world, head_id)
    granted = _measured_attempt_tick(world, head_id)
    assert first_hold is not None and granted is not None, f"{context}: no measured attempt. {world.state_dump()}"
    arrived_after_seconds = _ROOM_ARRIVES_AFTER_SECONDS + _TICK_SECONDS
    held_seconds = (granted - first_hold) * _TICK_SECONDS
    assert held_seconds >= arrived_after_seconds + _probe_delay_seconds(world), (
        f"{context}: the measured attempt came {held_seconds:.1f}s after the first hold, under a probe delay "
        f"after the room arrived at about {arrived_after_seconds:.1f}s. {world.state_dump()}"
    )
