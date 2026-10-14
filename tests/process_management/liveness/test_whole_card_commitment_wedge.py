"""A whole-card head admitted behind a lane parked on a preload for another pending job must not park the card.

A lane that has finished a preload reports ``PRELOADED_MODEL``, which the busy predicate counts as busy, so no
generic scale-down takes it and the safety move would wait on it as a sibling with work in hand. A whole-card head
admitted exclusively behind such a lane needs that lane gone before its residency can establish, and its exclusive
claim withholds dispatch from every other job, the preloaded lane's own job included. While the parked lane
stands, neither job can move, and the card idles with both pending until something outside the dispatch path
faults them.

The escape is any one of: the preloaded lane yields to the residency's teardown, the preloaded lane's job is
dispatched before the head claims the card, or the head is declared unserviceable and the claim dropped. Each
leaves one of the two jobs sampling well inside the lease-acquire timeout. When neither job has moved, the row
first checks the teardown's part (the parked lane stopped and safety off the card), so a head parked by
another gate after the teardown is told apart from the parked sibling.

Under the clearance lease, a head whose full charge does not fit the card its teardown left takes the partial-load
seat once the card reads converged-empty. That read needs the post-processing lane's served VRAM unload to stand.
A lane whose unload flag is cleared after it is served reads reclaimable again on the next pass, so the card never
converges and the head waits out its lease-acquire timeout to sample unpriced.
"""

from __future__ import annotations

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.scheduling.admission import pricing
from horde_worker_regen.process_management.scheduling.clearance_lease import CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS
from tests.process_management.conftest import make_job_pop_response, track_popped_job_async
from tests.process_management.liveness._dispatch_world import (
    _CARD_24GB,
    _POST_PROCESS_LANE_ID,
    _SDXL,
    _SDXL_OTHER,
    _DispatchWorld,
    _ModelClass,
)

pytestmark = pytest.mark.closed_loop

_TICK_SECONDS = 1.0
_SETUP_TICKS = 60
"""Ample for the live job to dispatch and for the head to be admitted behind it."""
_HELD_AUX_TICKS = 10
"""How long after the head's admission the preloaded lane's job stays held on its auxiliary download."""
_ESCAPE_BOUND_SECONDS = CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS
"""Neither job may wait longer than one lease-acquire timeout once the auxiliary file has landed."""
_FOREIGN_VRAM_MB = 3300.0
"""Device memory another program holds on the card, which no inference teardown reclaims."""
_HEAD_WEIGHTS_MB = 16_000.0
"""What the world charges the card for a Qwen-Image-class checkpoint, which fits beside the foreign tenant and the
admission noise once the teardown is done.

The scheduler does not price the head from this figure. It prices the baseline's static seed (weights plus the
sampling reserve), so that seed against the room the tenant leaves decides whether the cleared card admits the
head.
"""
_QWEN = _ModelClass("qwen", "qwen-image-checkpoint", KNOWN_IMAGE_GENERATION_BASELINE.qwen_image, _HEAD_WEIGHTS_MB)
_SEAT_HEAD_WEIGHTS_MB = 19_500.0
"""What the world charges the card for the head in the partial-seat row, equal to the weights the scheduler's seed
prices, so the child's load leaves part of them in host RAM against the room the tenant leaves."""
_SEAT_QWEN = _ModelClass(
    "qwen", "qwen-image-checkpoint", KNOWN_IMAGE_GENERATION_BASELINE.qwen_image, _SEAT_HEAD_WEIGHTS_MB
)
_PARKED_LANE_ID = 1
_PARTIAL_SEAT_GRANT = "granted at its partial-load seat"
"""The clearance decision's reason when the arbiter admits a staged job at its partial-load seat."""


def _wedge_world(*, head: _ModelClass = _QWEN, clearance_lease: bool = False) -> _DispatchWorld:
    """Three lanes on a 24 GB card with safety and the post-processing lane on it, and a foreign tenant.

    With ``clearance_lease`` the row runs two-phase clearance and models partial loads, the configuration a
    partial-load seat is priced under, since only a clearance grant carries the free figure and activation room a
    child sizes its partial load against.
    """
    return _DispatchWorld(
        card=_CARD_24GB,
        lane_count=3,
        max_threads=2,
        queue_depth=4,
        whole_card_enabled=True,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        service_contexts=True,
        safety_off_gpu_allowed=True,
        foreign_vram_mb_by_tick={1: _FOREIGN_VRAM_MB},
        extra_model_classes=(head,),
        clearance_lease=clearance_lease,
        two_phase=clearance_lease,
        partial_loads=clearance_lease,
    )


async def _pop_aux_held(world: _DispatchWorld, job: ImageGenerateJobPopResponse) -> None:
    """Track ``job`` without marking its auxiliary files present, so dispatch holds it until they are."""
    world.popped_at[str(job.id_)] = world.now
    await track_popped_job_async(world._job_tracker, job, time_popped=world.now)


def _land_aux(world: _DispatchWorld, job: ImageGenerateJobPopResponse) -> None:
    """Mark ``job``'s auxiliary files present and the job prepared, as the prefetch completion does."""
    for lora in job.payload.loras or []:
        world._job_tracker.mark_aux_prefetched(lora.name, is_version=bool(lora.is_version), is_ti=False)
    assert job.id_ is not None
    assert world._job_tracker.mark_job_aux_prepared_if_ready(job.id_), "precondition: the held job must be released"


def _parked_lane_state(world: _DispatchWorld) -> HordeProcessState | None:
    """Return the parked lane's reported state, or None once the lane has left the pool."""
    lane = world.scheduler._process_map.get(_PARKED_LANE_ID)
    return None if lane is None else lane.last_process_state


def _parked_lane_state_at_idle_start(world: _DispatchWorld, since: float) -> str:
    """Return the parked lane's state on the first idle tick at or after ``since``, for an assertion message."""
    for observation in world.tick_observations:
        if observation.now < since or not observation.idle:
            continue
        for lane in observation.lane_states:
            if lane.lane_id == _PARKED_LANE_ID:
                return f"{lane.state.name} at tick {observation.tick}"
        return f"gone from the pool at tick {observation.tick}"
    return "never idle"


def _first_idle_tick_since(world: _DispatchWorld, since: float) -> tuple[int, float] | None:
    """Return the tick and world time of the first idle tick at or after ``since``, or None if the card never idled."""
    for observation in world.tick_observations:
        if observation.now >= since and observation.idle:
            return observation.tick, observation.now
    return None


def _tick_time(world: _DispatchWorld, tick: int | None) -> float | None:
    """Return the world time of ``tick``, or None when it was not observed."""
    if tick is None:
        return None
    for observation in world.tick_observations:
        if observation.tick == tick:
            return observation.now
    return None


def _post_process_unloads_sent(world: _DispatchWorld) -> int:
    """Return how many VRAM unloads the parent has sent the post-processing lane."""
    lane = world.scheduler._process_map[_POST_PROCESS_LANE_ID]
    return sum(
        1
        for call in lane.pipe_connection.send.call_args_list
        if call.args
        and isinstance(call.args[0], HordeControlMessage)
        and call.args[0].control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
    )


async def _seed_parked_sibling_and_admit_head(
    world: _DispatchWorld,
    head: ImageGenerateJobPopResponse,
) -> tuple[ImageGenerateJobPopResponse, float]:
    """Run a live job on lane 0 with lane 1 parked on a preload, admit ``head`` exclusively, and return the held job.

    Returns the aux-held job and the world time the live job dispatched at.
    """
    world.seed_resident(0, _SDXL_OTHER, in_vram=True)
    world.seed_resident(_PARKED_LANE_ID, _SDXL, in_vram=False)
    live = make_job_pop_response(_SDXL_OTHER.name, width=1024, height=1024, ddim_steps=40)
    held = make_job_pop_response(
        _SDXL.name,
        width=1024,
        height=1024,
        ddim_steps=20,
        loras=[LorasPayloadEntry(name="a_lora_still_downloading", model=1.0, clip=1.0)],
    )
    await world.pop(live)
    await _pop_aux_held(world, held)
    for _ in range(_SETUP_TICKS):
        await world.step()
        if world.dispatch_tick(live) is not None:
            break
    assert world.dispatch_tick(live) is not None, f"precondition: the live job never dispatched. {world.state_dump()}"
    assert _parked_lane_state(world) is HordeProcessState.PRELOADED_MODEL, (
        f"precondition: lane {_PARKED_LANE_ID} must still be parked on its SDXL preload when the head arrives, "
        f"found {_parked_lane_state(world)}. {world.state_dump()}"
    )
    assert world.dispatch_tick(held) is None, "precondition: the aux-held job must not dispatch before its file lands"
    live_dispatched_at = world.now
    await world.pop(head)

    tracker = world._job_tracker
    for _ in range(_SETUP_TICKS):
        await world.step()
        if tracker.is_admitted_exclusive(head):
            break
    assert tracker.is_admitted_exclusive(head), (
        f"precondition: the head was never admitted to the card exclusively. {world.state_dump()}"
    )
    return held, live_dispatched_at


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Without the clearance lease the head is held on its full charge with its starvation clock at zero. The "
        "aux-held job is the queue head, the head's exclusive claim holds that job, and nothing escalates. The "
        "teardown half of the row (the parked lane stopped, safety off the card) holds. The liveness half is an "
        "open defect of the no-lease path."
    ),
)
async def test_a_lane_parked_on_a_preload_does_not_wedge_a_whole_card_head_behind_it() -> None:
    """One of the preloaded lane's job and the whole-card head samples within a lease timeout of the file landing.

    Lane 1 holds an SDXL checkpoint in host RAM, staged for a pending job held on its LoRA download. The job is
    queued before the first tick so the lane stays ``PRELOADED_MODEL``, since a preload whose model no pending job
    names is cleared back to waiting at the next preload pass. A live job samples on lane 0 when a Qwen-sized
    head arrives, so the head is admitted exclusively and pre-staged into the spare lane 2 while the live job
    finishes. The live job ends and the residency stops lane 0, which leaves lane 1 between the head and sole
    residency.
    """
    world = _wedge_world()
    head = make_job_pop_response(_QWEN.name, width=1024, height=1024, ddim_steps=20)
    held, live_dispatched_at = await _seed_parked_sibling_and_admit_head(world, head)

    for _ in range(_HELD_AUX_TICKS):
        await world.step()
    _land_aux(world, held)
    landed_at = world.now

    while world.now - landed_at < _ESCAPE_BOUND_SECONDS:
        await world.step()
        if world.dispatch_tick(held) is not None or world.dispatch_tick(head) is not None:
            break

    escaped = world.dispatch_tick(held) is not None or world.dispatch_tick(head) is not None
    parked_now = _parked_lane_state(world)
    parked_now_text = "gone from the pool" if parked_now is None else parked_now.name
    parked_at_idle_text = _parked_lane_state_at_idle_start(world, live_dispatched_at)
    if not escaped:
        # Whatever else parks the head, the residency's own teardown must have taken the parked lane and moved
        # safety off the card, since neither has a job the residency could ever let run.
        assert parked_now is None and world.safety_is_off_gpu(), (
            f"the residency's teardown left lane {_PARKED_LANE_ID} parked on its SDXL preload "
            f"({parked_at_idle_text} when the idle span began, {parked_now_text} now) or safety on the card "
            f"(off the card: {world.safety_is_off_gpu()}). {world.state_dump()}"
        )
    idle_ticks = sum(1 for observation in world.tick_observations if observation.now >= landed_at and observation.idle)
    assert escaped, (
        f"neither the preloaded lane's job nor the whole-card head dispatched within "
        f"{_ESCAPE_BOUND_SECONDS:.0f}s of the auxiliary file landing: the card was idle for {idle_ticks} ticks, "
        f"lane {_PARKED_LANE_ID} was {parked_at_idle_text} when the idle span began and is {parked_now_text} now, "
        f"and the head's exclusive claim held dispatch. {world.state_dump()}"
    )


async def test_a_whole_card_head_that_does_not_fit_alone_takes_the_partial_seat_after_the_teardown() -> None:
    """A Qwen-class head too large for the card its teardown leaves samples at its seat within a lease timeout.

    The same pool as the parked-sibling row, under two-phase clearance with partial loads. The scheduler prices
    the head's full charge from its baseline seed (about 22 GB), which the 24 GB card cannot give beside a 3.3 GB
    tenant whatever the teardown stops. Once the live job ends and the teardown has taken the parked lane and
    moved safety off, nothing on the card is left to reclaim, so the staged head is granted at its partial-load
    seat. Waiting for anything else leaves it to sample unpriced at its lease-acquire timeout.
    """
    world = _wedge_world(head=_SEAT_QWEN, clearance_lease=True)
    head = make_job_pop_response(_SEAT_QWEN.name, width=1024, height=1024, ddim_steps=20)
    held, live_dispatched_at = await _seed_parked_sibling_and_admit_head(world, head)

    for _ in range(_HELD_AUX_TICKS):
        await world.step()
    _land_aux(world, held)
    while world.dispatch_tick(head) is None and world.now - live_dispatched_at < 3 * _ESCAPE_BOUND_SECONDS:
        await world.step()

    idle_start = _first_idle_tick_since(world, live_dispatched_at)
    assert idle_start is not None, f"precondition: the live job never ended. {world.state_dump()}"
    idle_tick, idle_since = idle_start
    head_sampled_tick = world.dispatch_tick(head)
    assert _parked_lane_state(world) is None and world.safety_is_off_gpu(), (
        f"the residency's teardown left lane {_PARKED_LANE_ID} as {_parked_lane_state(world)} or safety on the "
        f"card (off the card: {world.safety_is_off_gpu()}). {world.state_dump()}"
    )
    sampled_at = _tick_time(world, head_sampled_tick)
    assert sampled_at is not None and sampled_at - idle_since < _ESCAPE_BOUND_SECONDS, (
        f"the head did not sample within {_ESCAPE_BOUND_SECONDS:.0f}s of the card idling at tick {idle_tick} "
        f"(sampled at tick {head_sampled_tick}); post-processing unloads sent {_post_process_unloads_sent(world)}. "
        f"{world.state_dump()}"
    )
    head_id = str(head.id_)
    assert not any(job_id == head_id for _, _, job_id in world.clearance_timeouts), (
        f"the head sampled unpriced at its lease-acquire timeout: {world.clearance_timeouts}"
    )
    head_decisions = [record for record in world.decision_records if record.subject == head_id]
    assert any(record.reason == _PARTIAL_SEAT_GRANT for record in head_decisions), (
        f"the head was not granted at its partial-load seat; its last decisions were "
        f"{[(record.tick, record.verdict.value, record.reason) for record in head_decisions[-5:]]}"
    )
    assert _post_process_unloads_sent(world) == 1, (
        f"the post-processing lane's VRAM unload was sent {_post_process_unloads_sent(world)} times. A served "
        "unload stands until the lane materialises VRAM again"
    )


async def test_a_served_post_processing_unload_stands_and_leaves_the_card_converged() -> None:
    """An idle post-processing lane is sent its VRAM unload once, and once served it is nothing left to reclaim.

    The parent retires an outstanding unload only when the slot reports VRAM materialised again, which a
    post-processing lane holding no model never does. Its flag therefore stands after the unload is served, the
    sender skips the lane, and the eviction mirror the converged-empty read is built from does not count it.
    """
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=1,
        max_threads=1,
        queue_depth=1,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        service_contexts=True,
    )
    scheduler = world.scheduler
    assert scheduler.unload_post_process_models_from_vram(), "precondition: the idle lane takes the unload"
    for _ in range(5):
        await world.step()
        scheduler.unload_post_process_models_from_vram()

    lane = scheduler._process_map[_POST_PROCESS_LANE_ID]
    assert _post_process_unloads_sent(world) == 1
    assert lane.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
    assert not pricing.has_reclaimable_idle_model(
        scheduler.snapshot(),
        0,
        for_head_of_queue=True,
        device_index=lane.device_index,
    ), "a post-processing lane that has served its unload must not keep the card from reading converged-empty"
