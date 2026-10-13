"""The dispatch world's inference pool and post-processing lane across their lifecycle.

A lane the pool regrows after a shrink is a new child, and under the clearance lease it must be clearable like
the first pool's lanes. A modelled post-processing chain waits out its lane's pause and finishes once the lane is
restored. Each row reads the world's own records: the job windows, the chains and the ticks a chain held.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
from horde_worker_regen.process_management.scheduling.admission import pricing
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import (
    _CARD_16GB,
    _CARD_24GB,
    _FLUX,
    _SD15,
    _SDXL,
    _SDXL_OTHER,
    _DispatchWorld,
)
from tests.process_management.liveness._host_ram import HostRamLedger

pytestmark = pytest.mark.closed_loop

_TICK_SECONDS = 1.0
"""Fine enough that a lane's lease wait and a chain's hold each span several ticks."""
_MAX_TICKS = 200
"""Ample for three one-megapixel jobs served one after another, or one job and its chain across a pause."""


async def test_a_lane_the_pool_regrows_is_cleared_under_the_lease() -> None:
    """Lanes added back after a shrink take their clearance like the first pool's lanes, and are never parked.

    The pool is cut to one lane and grown back to three through the lifecycle hook a whole-card residency's
    teardown and regrow call, so two of the lanes are children the first pool never had. Three jobs of three
    models then need all three lanes, and each must reach sampling.
    """
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=3,
        max_threads=3,
        queue_depth=3,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
        clearance_lease=True,
    )
    first_pool = set(world.inference_lane_ids())
    world.scale_inference_processes(1)
    world.scale_inference_processes(3)
    regrown = set(world.inference_lane_ids()) - first_pool
    assert regrown, f"precondition: the regrow reused every retired lane's id. {world.state_dump()}"
    jobs = [
        make_job_pop_response(model.name, width=1024, height=1024, ddim_steps=20)
        for model in (_SDXL, _SDXL_OTHER, _SD15)
    ]
    for job in jobs:
        await world.pop(job)

    for _ in range(_MAX_TICKS):
        await world.step()
        if world.completed_jobs == len(jobs):
            break

    windows = [world.job_windows.get(str(job.id_)) for job in jobs]
    unsampled = [
        job.model for job, window in zip(jobs, windows, strict=True) if window is None or window.sample_from is None
    ]
    assert not unsampled, f"jobs {unsampled} never reached sampling. {world.state_dump()}"
    served_by_regrown = {window.lane_id for window in windows if window is not None} & regrown
    assert served_by_regrown, (
        f"no regrown lane served a job, so the row proves nothing about them. {world.state_dump()}"
    )
    assert not world.clearance_timeouts, f"a lane sampled only at its lease timeout: {world.clearance_timeouts}"


async def test_a_chain_waits_out_its_lanes_pause_and_completes_after_the_restore() -> None:
    """A chain pending when its lane is paused holds, without error, and finishes once the lane is back."""
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=1,
        max_threads=1,
        queue_depth=2,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
        service_contexts=True,
        service_lane_exit_seconds=_TICK_SECONDS,
        post_process_chain_peak_mb=2000.0,
        post_process_chain_hold_ticks=3,
    )
    job = make_job_pop_response(_SDXL.name, width=1024, height=1024, ddim_steps=20, post_processing=["GFPGAN"])
    await world.pop(job)
    for _ in range(_MAX_TICKS):
        if world.job_tracker.jobs_pending_post_processing or world.post_process_chains:
            break
        await world.step()
    assert world.job_tracker.jobs_pending_post_processing or world.post_process_chains, (
        f"the job never reached post-processing. {world.state_dump()}"
    )
    assert world._lifecycle.pause_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD)
    paused_ticks = 10
    for _ in range(paused_ticks):
        await world.step()
    assert not any(chain.released_tick is not None for chain in world.post_process_chains), (
        "a chain finished while its lane was paused"
    )
    assert world._lifecycle.restore_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD)
    for _ in range(_MAX_TICKS):
        if any(chain.released_tick is not None for chain in world.post_process_chains):
            break
        await world.step()

    finished = [chain for chain in world.post_process_chains if chain.released_tick is not None]
    assert finished, f"the chain never finished after the lane was restored. {world.state_dump()}"
    assert len(world.post_process_lane_absent_ticks) >= paused_ticks, (
        f"the world recorded {len(world.post_process_lane_absent_ticks)} held ticks over a {paused_ticks}-tick pause"
    )


async def test_an_idle_lane_unloaded_to_ram_keeps_its_unload_and_is_not_evicted_again() -> None:
    """A lane whose model was unloaded to RAM for a head is evicted once, and its slot stops reading as evictable.

    The parent retires an outstanding unload only when the slot materialises VRAM again, so the flag stands while
    the model sits in RAM and keeps the lane out of the idle-eviction mirror. A world that dropped it would have
    the mirror count the lane again on the next pass and re-send an eviction that frees nothing.
    """
    world = _DispatchWorld(
        card=_CARD_16GB,
        lane_count=2,
        max_threads=2,
        queue_depth=2,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
        clearance_lease=True,
        whole_card_enabled=False,
        host_ram=HostRamLedger(65_536.0, 8192.0),
    )
    world.seed_resident(0, _SDXL, in_vram=True)
    head = make_job_pop_response(_FLUX.name, width=1024, height=1024, ddim_steps=30)
    await world.pop(head)
    for _ in range(_MAX_TICKS):
        await world.step()
        if world.completed_jobs == 1:
            break

    releases = [tick for tick, lane_id in world.unload_releases if lane_id == 0]
    assert len(releases) == 1, f"the idle lane was unloaded {len(releases)} times, at ticks {releases[:8]}"
    assert world.completed_jobs == 1, f"the head never finished. {world.state_dump()}"
    idle_lane = world._process_map[0]
    assert idle_lane.loaded_horde_model_name == _SDXL.name, "the unloaded lane no longer holds its model in RAM"
    assert idle_lane.last_control_flag is HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
    assert not pricing.has_reclaimable_idle_model(
        world.scheduler.snapshot(), 1, for_head_of_queue=True, device_index=0
    ), "the RAM-held lane still reads as a reclaimable idle model"
