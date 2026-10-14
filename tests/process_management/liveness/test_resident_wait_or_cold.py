"""A queue head whose model is resident on a busy lane either waits for that lane or starts cold on an idle one.

On one card the second copy is loaded only when the busy lane's remaining sampling plus the resident dispatch
overhead is longer than the cold dispatch overhead, every figure known, and never for a model that cannot share
the card with its own copy. The rows run the real scheduler over the dispatch world with the clearance lease and
the step heartbeat, and seed the performance model's overhead windows directly, since the world does not learn
them.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeControlMessage
from horde_worker_regen.process_management.scheduling.performance_model import signature_from_job
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import (
    _CARD_24GB,
    _FLUX,
    _SDXL,
    _SDXL_OTHER,
    _DispatchWorld,
    _JobWindow,
    _ModelClass,
)

pytestmark = pytest.mark.closed_loop

_TICK_SECONDS = 0.5
"""Half a second per tick, so a few seconds of dispatch overhead span several scheduling cycles."""
_STEPS = 30
_LONG_SAMPLE_SECONDS = 60.0
"""A running sample long enough that waiting for it costs far more than a cold start."""
_SHORT_SAMPLE_SECONDS = 6.0
"""A running sample short enough that waiting for it is cheaper than the cold start the row seeds."""
_MAX_TICKS = 400
_OVERHEAD_SAMPLES = 5
"""Observations per window, the performance model's default trust threshold."""


def _job(model: _ModelClass) -> ImageGenerateJobPopResponse:
    return make_job_pop_response(model.name, width=1024, height=1024, ddim_steps=_STEPS)


def _world(model: _ModelClass, *, running: ImageGenerateJobPopResponse, sample_seconds: float) -> _DispatchWorld:
    """Return a 24 GB card with two sampling slots and two lanes, the first holding ``model`` and the second empty."""
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=2,
        max_threads=2,
        queue_depth=2,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
        clearance_lease=True,
        tail_overlap=True,
        performance_model=True,
        sample_seconds_by_job={str(running.id_): sample_seconds},
    )
    world.seed_resident(world.inference_lane_ids()[0], model, in_vram=True)
    return world


def _seed_overheads(
    world: _DispatchWorld,
    job: ImageGenerateJobPopResponse,
    *,
    resident_seconds: float,
    cold_seconds: float,
) -> None:
    """Teach the scheduler's performance model the job's dispatch overheads, resident and cold."""
    performance_model = world.performance_model
    assert performance_model is not None and job.model is not None
    baseline = world.scheduler._model_metadata.get_baseline(job.model)
    signature = signature_from_job(job, str(baseline) if baseline is not None else None)
    assert signature is not None
    for _ in range(_OVERHEAD_SAMPLES):
        performance_model.observe_overhead(signature, resident_seconds, resident=True)
        performance_model.observe_overhead(signature, cold_seconds, resident=False)


async def _run(world: _DispatchWorld, running: ImageGenerateJobPopResponse, head: ImageGenerateJobPopResponse) -> int:
    """Start ``running``, queue ``head`` once it samples, and run until both finish.

    Returns:
        The most lanes that held the head's model at once.
    """
    await world.pop(running)
    for _ in range(_MAX_TICKS):
        await world.step()
        window = world.job_windows.get(str(running.id_))
        if window is not None and window.sample_from is not None:
            break
    await world.pop(head)
    process_map = world.scheduler._process_map
    most_copies = 0
    for _ in range(_MAX_TICKS):
        await world.step()
        most_copies = max(
            most_copies,
            sum(
                1
                for lane_id in world.inference_lane_ids()
                if process_map[lane_id].loaded_horde_model_name == head.model
            ),
        )
        if world.completed_jobs >= 2:
            break
    assert world.completed_jobs >= 2, f"both jobs must finish. {world.state_dump()}"
    return most_copies


def _windows(world: _DispatchWorld, *jobs: ImageGenerateJobPopResponse) -> list[_JobWindow]:
    return [world.job_windows[str(job.id_)] for job in jobs]


async def test_a_long_remaining_sample_sends_the_head_cold_to_the_idle_lane() -> None:
    """With a minute of sampling left and a cheap cold start, the head samples on the idle lane beside it."""
    running, head = _job(_SDXL), _job(_SDXL)
    world = _world(_SDXL, running=running, sample_seconds=_LONG_SAMPLE_SECONDS)
    _seed_overheads(world, head, resident_seconds=2.0, cold_seconds=4.0)

    most_copies = await _run(world, running, head)

    running_window, head_window = _windows(world, running, head)
    assert head_window.lane_id != running_window.lane_id, f"the head waited for the busy lane. {world.state_dump()}"
    assert most_copies == 2
    assert head_window.sample_from is not None and running_window.sample_until is not None
    assert head_window.sample_from < running_window.sample_until, (
        f"the head started at {head_window.sample_from:.1f}s, after the busy lane's sample ended at "
        f"{running_window.sample_until:.1f}s. {world.state_dump()}"
    )


async def test_a_short_remaining_sample_keeps_the_head_on_its_resident_lane() -> None:
    """With a few seconds left and a costly cold start, the head waits and runs on the lane holding its model."""
    running, head = _job(_SDXL), _job(_SDXL)
    world = _world(_SDXL, running=running, sample_seconds=_SHORT_SAMPLE_SECONDS)
    _seed_overheads(world, head, resident_seconds=1.0, cold_seconds=30.0)

    most_copies = await _run(world, running, head)

    running_window, head_window = _windows(world, running, head)
    assert head_window.lane_id == running_window.lane_id, f"a copy was loaded for the head. {world.state_dump()}"
    assert most_copies == 1


async def test_an_unknown_overhead_keeps_the_head_on_its_resident_lane() -> None:
    """With nothing learned about the signature's overheads, the head waits as the single-copy rule has it."""
    running, head = _job(_SDXL), _job(_SDXL)
    world = _world(_SDXL, running=running, sample_seconds=_LONG_SAMPLE_SECONDS)

    most_copies = await _run(world, running, head)

    running_window, head_window = _windows(world, running, head)
    assert head_window.lane_id == running_window.lane_id, f"a copy was loaded for the head. {world.state_dump()}"
    assert most_copies == 1


async def test_a_cold_copy_never_evicts_an_idle_lanes_resident() -> None:
    """With the only idle lane holding another model on the card, the head waits and that lane is left alone.

    The copy would start sooner, but seating it means evicting the idle lane's resident. The copy is an
    optimisation of a head that already has a resident lane, so it loads only where the card has room as it
    stands. No preload, unload or eviction reaches the idle lane.
    """
    running, head = _job(_SDXL), _job(_SDXL)
    world = _world(_SDXL, running=running, sample_seconds=_LONG_SAMPLE_SECONDS)
    idle_lane = world.inference_lane_ids()[1]
    world.seed_resident(idle_lane, _SDXL_OTHER, in_vram=True)
    _seed_overheads(world, head, resident_seconds=2.0, cold_seconds=4.0)

    most_copies = await _run(world, running, head)

    running_window, head_window = _windows(world, running, head)
    assert head_window.lane_id == running_window.lane_id, f"a copy was loaded for the head. {world.state_dump()}"
    assert most_copies == 1
    send: Mock = world.scheduler._process_map[idle_lane].pipe_connection.send  # type: ignore[assignment]
    sent_flags = [
        call.args[0].control_flag
        for call in send.call_args_list
        if call.args and isinstance(call.args[0], HordeControlMessage)
    ]
    displacing = {
        HordeControlFlag.PRELOAD_MODEL,
        HordeControlFlag.UNLOAD_MODELS_FROM_VRAM,
        HordeControlFlag.UNLOAD_MODELS_FROM_RAM,
    }
    assert not displacing.intersection(sent_flags), f"the idle lane was sent {sent_flags}. {world.state_dump()}"
    assert world.scheduler._process_map[idle_lane].loaded_horde_model_name == _SDXL_OTHER.name


async def test_an_extra_large_model_is_never_duplicated() -> None:
    """A model that cannot sample beside its own copy waits, however long the sample and cheap the cold start."""
    running, head = _job(_FLUX), _job(_FLUX)
    world = _world(_FLUX, running=running, sample_seconds=_LONG_SAMPLE_SECONDS)
    _seed_overheads(world, head, resident_seconds=2.0, cold_seconds=4.0)

    most_copies = await _run(world, running, head)

    assert most_copies == 1, f"a second copy of the extra-large model was loaded. {world.state_dump()}"
