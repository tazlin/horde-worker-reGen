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

A card can be granted to a whole-card head queued behind jobs waiting for auxiliary files, since preload and
dispatch both step past such jobs. When the files land during the teardown, those jobs come first in the preload
pass and are refused for the head's exclusive claim. The pass goes on over the held card for the head alone, so
the head is preloaded and samples, and the jobs ahead of it run once the residency drains.

When the job waiting for files is of the same model as the granted job behind it, it becomes the dispatch head once
the files land and takes the pending claim, since it serves the same residency. A job of another model, and an
over-budget admit's claim, are left as they were.
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
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
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
    assert job.id_ is not None
    for lora in job.payload.loras or []:
        world._job_tracker.record_aux_present(job.id_, lora.name, is_version=bool(lora.is_version), is_ti=False)
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


_SAFETY_ROW_LONG_SAMPLE_SECONDS = 25.0
"""How long the card's running SDXL job samples before the Qwen-class head can take the card."""
_SAFETY_ROW_TICKS = 90
"""Ticks the safety row runs: the long job, the teardown and the head's load and sample."""
_SAFETY_ROW_HEAD_BOUND_SECONDS = 15.0
"""How long the head may take to reach sampling once the card frees, inside the queue-deadlock horizon."""


async def test_a_whole_card_head_whose_weights_do_not_fit_beside_safety_moves_safety_off_the_card() -> None:
    """A Qwen-class head whose whole weights do not fit beside safety on a 24 GB card has the residency move safety.

    The teardown completes only when the whole weights and the reserve fit, and 19.5 GB of weights do not fit
    beside safety's context. The head's partial seat would fit beside safety, so a check on the seat alone keeps
    safety on the card and leaves the head waiting for the runtime safety placement's dwell.
    """
    long_job = make_job_pop_response(_SDXL.name, width=1024, height=1024, ddim_steps=30)
    head = make_job_pop_response(_SEAT_QWEN.name, width=1024, height=1024, ddim_steps=4)
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=2,
        max_threads=1,
        queue_depth=5,
        whole_card_enabled=True,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        service_contexts=True,
        sample_seconds_by_job={str(long_job.id_): _SAFETY_ROW_LONG_SAMPLE_SECONDS},
        extra_model_classes=(_SEAT_QWEN,),
    )
    world.seed_resident(0, _SDXL, in_vram=True)
    await world.pop(long_job)
    for _ in range(2):
        await world.step()
    await world.pop(head)
    for _ in range(_SAFETY_ROW_TICKS):
        await world.step()

    long_window = world.job_windows.get(str(long_job.id_))
    assert long_window is not None and long_window.sample_until is not None, (
        f"precondition: the long job must finish. {world.state_dump()}"
    )
    head_sampled_tick = world.dispatch_tick(head)
    sampled_at = _tick_time(world, head_sampled_tick)
    assert sampled_at is not None and sampled_at - long_window.sample_until < _SAFETY_ROW_HEAD_BOUND_SECONDS, (
        f"the head did not sample within {_SAFETY_ROW_HEAD_BOUND_SECONDS:.0f}s of the card freeing (sampled at "
        f"tick {head_sampled_tick}); safety pause events {world.safety_pause_events}. {world.state_dump()}"
    )
    assert any(owner is PauseOwner.WHOLE_CARD for _, _, owner in world.safety_pause_events), (
        f"the residency did not move safety off the card; pause events {world.safety_pause_events}. "
        f"{world.state_dump()}"
    )
    assert not world.structural_queue_wedges, (
        f"the queue-deadlock detector reported a wedge on tick(s) {world.structural_queue_wedges}. "
        f"{world.state_dump()}"
    )


_AUX_HELD_JOB_COUNT = 3
"""How many jobs waiting for LoRA files are queued ahead of the whole-card head."""
_BENEFICIARY_ROW_TICKS = 150
"""Ticks the beneficiary row runs after the files land: the head's load and sample, then the three jobs."""
_ARRIVAL_ROW_TICKS = 420
"""Ticks the arrivals row runs after the files land: past the maximum hold, the drain and the three jobs."""
_MAX_HOLD_SECONDS = 180
"""The world's default maximum hold for one residency, the bound on how long the head's model owns the card."""


def _beneficiary_world() -> _DispatchWorld:
    """Three lanes on a 24 GB card under two-phase clearance with a foreign tenant, safety and the service lanes."""
    return _DispatchWorld(
        card=_CARD_24GB,
        lane_count=3,
        max_threads=2,
        queue_depth=6,
        whole_card_enabled=True,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        service_contexts=True,
        safety_off_gpu_allowed=True,
        foreign_vram_mb_by_tick={1: _FOREIGN_VRAM_MB},
        extra_model_classes=(_QWEN,),
        clearance_lease=True,
        two_phase=True,
        partial_loads=True,
        cooldown_seconds=45,
        max_hold_seconds=_MAX_HOLD_SECONDS,
    )


async def _grant_head_behind_aux_held_jobs(
    world: _DispatchWorld,
) -> tuple[list[ImageGenerateJobPopResponse], ImageGenerateJobPopResponse]:
    """Queue three LoRA jobs held on LoRA files ahead of a Qwen-class head and step until the head is granted.

    Returns the held jobs and the head. On return the head holds the exclusive claim and has not been preloaded,
    and the held jobs' files have just landed.
    """
    held = [
        make_job_pop_response(
            _SDXL.name,
            width=1024,
            height=1024,
            ddim_steps=20,
            loras=[LorasPayloadEntry(name=f"a_lora_still_downloading_{index}", model=1.0, clip=1.0)],
        )
        for index in range(_AUX_HELD_JOB_COUNT)
    ]
    for job in held:
        await _pop_aux_held(world, job)
    head = make_job_pop_response(_QWEN.name, width=1024, height=1024, ddim_steps=8)
    await world.pop(head)
    for _ in range(_SETUP_TICKS):
        await world.step()
        if world._job_tracker.is_admitted_exclusive(head):
            break
    assert world._job_tracker.is_admitted_exclusive(head), (
        f"precondition: the head was never admitted to the card exclusively. {world.state_dump()}"
    )
    assert not any(lane.loaded_horde_model_name == _QWEN.name for lane in world.scheduler._process_map.values()), (
        f"precondition: the head must not be preloaded before the held jobs' files land. {world.state_dump()}"
    )
    for job in held:
        _land_aux(world, job)
    return held, head


async def test_a_whole_card_head_behind_jobs_whose_files_land_during_the_teardown_preloads_and_completes() -> None:
    """The head is preloaded and samples although the jobs ahead of it are refused for the head's claim.

    The card is granted to the Qwen-class head while three SDXL jobs ahead of it wait for LoRA files. The
    files land before the head is preloaded, so the SDXL jobs come first in every preload pass and are refused with
    ``EXCLUSIVE_IN_PROGRESS``. A pass that stops at that refusal does not reach the head, and the card idles with
    all four jobs pending until the residency is released as unable to converge, past the 120 s establish grace.
    The SDXL jobs run once the head's residency drains, and no job faults.
    """
    world = _beneficiary_world()
    held, head = await _grant_head_behind_aux_held_jobs(world)
    landed_at = world.now
    for _ in range(_BENEFICIARY_ROW_TICKS):
        await world.step()
        if world.submitted_jobs == len(held) + 1:
            break

    head_sampled_at = _tick_time(world, world.dispatch_tick(head))
    assert head_sampled_at is not None and head_sampled_at - landed_at < _ESCAPE_BOUND_SECONDS, (
        f"the head did not sample within {_ESCAPE_BOUND_SECONDS:.0f}s of the held jobs' files landing (sampled at "
        f"tick {world.dispatch_tick(head)}). {world.state_dump()}"
    )
    first_held_sampled_at = _tick_time(world, world.dispatch_tick(held[0]))
    assert first_held_sampled_at is not None and first_held_sampled_at - landed_at < _ESCAPE_BOUND_SECONDS, (
        f"the first held job did not sample within {_ESCAPE_BOUND_SECONDS:.0f}s of the file landing (sampled at "
        f"tick {world.dispatch_tick(held[0])}). {world.state_dump()}"
    )
    assert world.submitted_jobs == len(held) + 1, (
        f"{world.submitted_jobs} of {len(held) + 1} jobs were finalized. {world.state_dump()}"
    )
    assert world._job_tracker.num_jobs_faulted == 0, f"a job faulted. {world.state_dump()}"
    assert not world.structural_queue_wedges, (
        f"the queue-deadlock detector reported a wedge on tick(s) {world.structural_queue_wedges}. "
        f"{world.state_dump()}"
    )
    assert not world._job_tracker.has_exclusive_job_in_progress(), (
        f"an exclusive admit outlived the job. {world.state_dump()}"
    )


async def test_arriving_head_model_work_does_not_keep_the_jobs_ahead_of_it_off_the_card() -> None:
    """Qwen jobs offered every tick do not hold the SDXL jobs ahead of them past the residency's maximum hold.

    On the held card the pass preloads only the card's exclusive admit. The other Qwen jobs ride the residency,
    and the maximum hold ends the claim over the offer, so the residency drains and the SDXL jobs run. The
    row counts completed jobs and accepted-job faults.
    """
    world = _beneficiary_world()
    held, head = await _grant_head_behind_aux_held_jobs(world)
    granted_at = world.now
    offered_head_model_jobs = 0
    for _ in range(_ARRIVAL_ROW_TICKS):
        arrival = make_job_pop_response(_QWEN.name, width=1024, height=1024, ddim_steps=8)
        if await world.offer_job(arrival):
            offered_head_model_jobs += 1
        await world.step()
        if all(job_id in world.completed_job_ids for job_id in (str(job.id_) for job in held)):
            break

    held_ids = [str(job.id_) for job in held]
    assert offered_head_model_jobs > 0, f"precondition: a Qwen job must arrive after the grant. {world.state_dump()}"
    assert world.dispatch_tick(head) is not None, f"the head never sampled. {world.state_dump()}"
    assert all(job_id in world.completed_job_ids for job_id in held_ids), (
        f"{sum(job_id in world.completed_job_ids for job_id in held_ids)} of {len(held)} held jobs completed while "
        f"{offered_head_model_jobs} Qwen jobs arrived. {world.state_dump()}"
    )
    bound_seconds = _MAX_HOLD_SECONDS + _ESCAPE_BOUND_SECONDS
    for job in held:
        sampled_at = _tick_time(world, world.dispatch_tick(job))
        assert sampled_at is not None and sampled_at - granted_at < bound_seconds, (
            f"a held job sampled {sampled_at}s, past {bound_seconds:.0f}s after the grant at {granted_at}. "
            f"{world.state_dump()}"
        )
    assert world._job_tracker.num_jobs_faulted == 0, f"an accepted job faulted. {world.state_dump()}"


async def test_a_whole_card_head_that_is_the_undispatched_head_is_preloaded_and_completes() -> None:
    """The granted job is the queue's first job, and the jobs behind it run once the head's residency drains.

    The undispatched head is the residency's model, so the residency never reads as unable to converge. The
    pass is the only route to running the head, and it reaches the head on every cycle.
    """
    world = _beneficiary_world()
    head = make_job_pop_response(_QWEN.name, width=1024, height=1024, ddim_steps=8)
    await world.pop(head)
    held = [
        make_job_pop_response(
            _SDXL.name,
            width=1024,
            height=1024,
            ddim_steps=20,
            loras=[LorasPayloadEntry(name=f"a_lora_still_downloading_{index}", model=1.0, clip=1.0)],
        )
        for index in range(_AUX_HELD_JOB_COUNT)
    ]
    for job in held:
        await _pop_aux_held(world, job)
    tracker = world._job_tracker
    for _ in range(_SETUP_TICKS):
        await world.step()
        if tracker.is_admitted_exclusive(head):
            break
    assert tracker.is_admitted_exclusive(head), f"precondition: the head was never admitted. {world.state_dump()}"
    assert tracker.undispatched_head() == head, (
        f"precondition: the granted job must be the undispatched head. {world.state_dump()}"
    )
    for job in held:
        _land_aux(world, job)
    landed_at = world.now
    for _ in range(_BENEFICIARY_ROW_TICKS):
        await world.step()
        if world.submitted_jobs == len(held) + 1:
            break

    head_sampled_at = _tick_time(world, world.dispatch_tick(head))
    assert head_sampled_at is not None and head_sampled_at - landed_at < _ESCAPE_BOUND_SECONDS, (
        f"the head did not sample within {_ESCAPE_BOUND_SECONDS:.0f}s of the held jobs' files landing (sampled at "
        f"tick {world.dispatch_tick(head)}). {world.state_dump()}"
    )
    assert world.submitted_jobs == len(held) + 1, (
        f"{world.submitted_jobs} of {len(held) + 1} jobs were finalized. {world.state_dump()}"
    )
    assert world._job_tracker.num_jobs_faulted == 0, f"a job faulted. {world.state_dump()}"
    assert not world.structural_queue_wedges, (
        f"the queue-deadlock detector reported a wedge on tick(s) {world.structural_queue_wedges}. "
        f"{world.state_dump()}"
    )


_ORPHAN_ROW_TICKS = 90
"""Ticks the orphaned-mark row runs before the marked job's file lands, and again after it."""


async def test_an_exclusive_mark_left_without_a_residency_is_released_and_the_job_is_admitted_again() -> None:
    """A pending exclusive mark whose card does not hold a residency does not keep the card from other jobs.

    A soft reset rebuilds the lanes and leaves the ledger and the marks as they were, so a Qwen job can stay marked
    for a card whose residency has ended. The row starts from that state, with the Qwen job still waiting for the
    LoRA file, so the pass cannot reach it through the held card. While the mark stands, the SDXL job's preload is
    refused on every pass. Released, the SDXL job runs at once. When the Qwen job's file lands, the demand path
    grants it a new residency and marks it again, and it completes.
    """
    world = _beneficiary_world()
    tracker = world._job_tracker
    marked = make_job_pop_response(
        _QWEN.name,
        width=1024,
        height=1024,
        ddim_steps=8,
        loras=[LorasPayloadEntry(name="a_lora_still_downloading", model=1.0, clip=1.0)],
    )
    await _pop_aux_held(world, marked)
    tracker.mark_admitted_exclusive(marked, device_index=None)
    other = make_job_pop_response(_SDXL.name, width=1024, height=1024, ddim_steps=20)
    await world.pop(other)
    assert not world.scheduler._whole_card_ledger.held(), "precondition: the card must not hold a residency"

    for _ in range(_ORPHAN_ROW_TICKS):
        await world.step()
        if str(other.id_) in world.completed_job_ids:
            break
    assert str(other.id_) in world.completed_job_ids, (
        f"the SDXL job did not complete while a mark without a residency stood. {world.state_dump()}"
    )
    assert world.dispatch_tick(marked) is None, "precondition: the marked job must still wait for the file"

    _land_aux(world, marked)
    for _ in range(_ORPHAN_ROW_TICKS):
        await world.step()
        if world.submitted_jobs == 2:
            break
    assert str(marked.id_) in world.completed_job_ids, (
        f"the released job did not complete once the file landed. {world.state_dump()}"
    )
    assert world.submitted_jobs == 2, f"{world.submitted_jobs} of 2 jobs were finalized. {world.state_dump()}"
    assert world._job_tracker.num_jobs_faulted == 0, f"a job faulted. {world.state_dump()}"
    assert not world.structural_queue_wedges, (
        f"the queue-deadlock detector reported a wedge on tick(s) {world.structural_queue_wedges}. "
        f"{world.state_dump()}"
    )


_SAME_MODEL_ROW_TICKS = 150
"""Ticks the same-model rows run after the head's file lands: the head's load and sample, then the follower's."""


async def _mark_follower_behind_aux_held_head(
    world: _DispatchWorld,
) -> tuple[ImageGenerateJobPopResponse, ImageGenerateJobPopResponse]:
    """Queue a Qwen head held on a LoRA file and a Qwen follower, and step until the follower holds the claim.

    Returns the head and the follower. On return the card holds a residency for Qwen granted to the follower, the
    follower has not sampled, and the head's file has not landed.
    """
    head = make_job_pop_response(
        _QWEN.name,
        width=1024,
        height=1024,
        ddim_steps=8,
        loras=[LorasPayloadEntry(name="a_lora_still_downloading", model=1.0, clip=1.0)],
    )
    await _pop_aux_held(world, head)
    follower = make_job_pop_response(_QWEN.name, width=1024, height=1024, ddim_steps=8)
    await world.pop(follower)
    tracker = world._job_tracker
    for _ in range(_SETUP_TICKS):
        await world.step()
        if tracker.is_admitted_exclusive(follower):
            break
    assert tracker.is_admitted_exclusive(follower), (
        f"precondition: the follower was never admitted to the card exclusively. {world.state_dump()}"
    )
    assert not tracker.is_admitted_exclusive(head), (
        f"precondition: the held head must be unmarked. {world.state_dump()}"
    )
    assert world.dispatch_tick(follower) is None, (
        f"precondition: the follower must not sample before the head's file lands. {world.state_dump()}"
    )
    assert world.scheduler._whole_card_ledger.holder_for_model(_QWEN.name)[0], (
        f"precondition: the card must hold a residency for the follower's model. {world.state_dump()}"
    )
    return head, follower


def _exclusive_jobs_running(world: _DispatchWorld) -> int:
    """Return how many exclusively admitted jobs are sampling."""
    tracker = world._job_tracker
    return sum(1 for job in tracker.jobs_in_progress if tracker.is_admitted_exclusive(job))


async def test_a_same_model_head_whose_file_lands_takes_the_claim_from_the_marked_follower() -> None:
    """A Qwen head whose LoRA lands after a Qwen follower was granted the card dispatches with the card to itself.

    Both demand paths step past a job waiting for auxiliary files, so the follower is granted the residency and
    marked while the head waits. Once the file lands the head is the dispatch head again. It is unmarked, and the
    follower's pending claim with the residency held suppresses every unmarked job on the card. The dispatch walk
    returns nothing for the head and never reaches the follower, and the release pass keeps the follower's mark
    while the card holds a residency for its model. Taking the claim lets the head sample, and the follower runs
    after it. At most one exclusive job samples at a time.
    """
    world = _beneficiary_world()
    head, follower = await _mark_follower_behind_aux_held_head(world)
    _land_aux(world, head)
    landed_at = world.now
    most_exclusive_running = 0
    for _ in range(_SAME_MODEL_ROW_TICKS):
        await world.step()
        most_exclusive_running = max(most_exclusive_running, _exclusive_jobs_running(world))
        if world.submitted_jobs == 2:
            break

    head_sampled_at = _tick_time(world, world.dispatch_tick(head))
    assert head_sampled_at is not None and head_sampled_at - landed_at < _ESCAPE_BOUND_SECONDS, (
        f"the head did not sample within {_ESCAPE_BOUND_SECONDS:.0f}s of its file landing (sampled at tick "
        f"{world.dispatch_tick(head)}). {world.state_dump()}"
    )
    assert world.submitted_jobs == 2, f"{world.submitted_jobs} of 2 jobs were finalized. {world.state_dump()}"
    assert most_exclusive_running <= 1, (
        f"{most_exclusive_running} exclusive jobs sampled on the card at once. {world.state_dump()}"
    )
    assert world._job_tracker.num_jobs_faulted == 0, f"a job faulted. {world.state_dump()}"
    assert not world.structural_queue_wedges, (
        f"the queue-deadlock detector reported a wedge on tick(s) {world.structural_queue_wedges}. "
        f"{world.state_dump()}"
    )
    assert not world._job_tracker.has_exclusive_job_in_progress(), (
        f"an exclusive admit outlived the job. {world.state_dump()}"
    )


_HELD_CLAIM_TICKS = 30
"""Ticks the over-budget row watches the claim after the head's file lands."""


async def test_a_job_of_another_model_does_not_take_the_claim_from_the_marked_follower() -> None:
    """A dispatch head of a model other than the residency's stays suppressed by the follower's pending claim.

    The claim is the card's residency for Qwen, which an SDXL job does not serve. The row reads the rule on the
    card's live state, since a second inference lane resident with SDXL does not survive the teardown.
    """
    world = _beneficiary_world()
    head, follower = await _mark_follower_behind_aux_held_head(world)
    other = make_job_pop_response(_SDXL.name, width=1024, height=1024, ddim_steps=20)
    await world.pop(other)
    scheduler = world.scheduler

    assert scheduler._exclusive_dispatch_suppression_active(None), (
        f"precondition: the follower's claim must suppress unmarked jobs on the card. {world.state_dump()}"
    )
    assert scheduler._head_takes_exclusive_claim(head, None), (
        f"precondition: a Qwen head must be able to take the claim on the same state. {world.state_dump()}"
    )
    assert not scheduler._head_takes_exclusive_claim(other, None), (
        f"an SDXL head took the claim from a Qwen admit while the Qwen residency is held. {world.state_dump()}"
    )
    assert world._job_tracker.is_admitted_exclusive(follower), f"the follower lost its claim. {world.state_dump()}"
    assert not world._job_tracker.is_admitted_exclusive(other), f"the SDXL job was marked. {world.state_dump()}"


async def test_an_over_budget_admit_keeps_its_claim_when_a_same_model_head_is_ready() -> None:
    """A same-model head does not take the claim of an over-budget admit, whose mark isolates that job for its life.

    The row marks the follower over budget before the head's file lands, then watches the claim. On every tick while
    the follower is pending it keeps its mark and the head stays unmarked.
    """
    world = _beneficiary_world()
    head, follower = await _mark_follower_behind_aux_held_head(world)
    tracker = world._job_tracker
    assert world.scheduler._head_takes_exclusive_claim(head, None), (
        f"precondition: the head must be able to take a whole-card admit's claim. {world.state_dump()}"
    )
    tracker.mark_admitted_over_budget(follower)
    _land_aux(world, head)
    assert not world.scheduler._head_takes_exclusive_claim(head, None), (
        f"the head may take an over-budget admit's claim. {world.state_dump()}"
    )
    for _ in range(_HELD_CLAIM_TICKS):
        await world.step()
        if follower not in tracker.jobs_pending_inference:
            break
        assert tracker.is_admitted_exclusive(follower), (
            f"the over-budget follower lost its claim while pending. {world.state_dump()}"
        )
        assert not tracker.is_admitted_exclusive(head), (
            f"the head took the over-budget follower's claim. {world.state_dump()}"
        )
