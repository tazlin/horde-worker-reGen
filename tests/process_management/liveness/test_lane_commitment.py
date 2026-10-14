"""A lane committed to work it has not begun receives no command for other work ahead of it.

A child reads its control messages one at a time, so a command the parent sends a lane queues behind every
message already in the pipe. The parent books a preload at the send, and until the child's first report the
lane reads idle and holding the model it was sent. A preload pass that re-plans in that window can aim a
second model at the same lane, expire the first model's entry as displaced, and aim the first model back, so
the lane's pipe fills with alternating loads ahead of the job the first one was for. A lane that owns a
dispatched job is committed to that job until its first step, so no reclaim rung may take it either.

The world serves only the newest command a lane holds and does not queue a dispatch behind the loads ahead of
it, so these rows read the commands the parent sent, never the delay those commands cost a real child.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeControlModelMessage,
    HordeInferenceControlMessage,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.resources.reclaim_ladder import (
    ReclaimRung,
    ReclaimRungKind,
    build_reclaim_ladder,
    execute_reclaim_rung,
)
from horde_worker_regen.process_management.resources.run_metrics import DecisionKind
from horde_worker_regen.process_management.scheduling.admission import pricing
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import (
    _CARD_24GB,
    _FLUX,
    _SD15,
    _SD15_OTHER,
    _SDXL,
    _SDXL_OTHER,
    _DispatchWorld,
)

pytestmark = pytest.mark.closed_loop

_TICK_SECONDS = 0.5
_MAX_TICKS = 120
"""Ample for the long job on lane 0 and the two short jobs that follow it."""
_PRELOAD_REPORT_LATENCY_SECONDS = 3.0
"""How long a lane reads idle after a preload is sent: several preload passes run inside it."""
_FOREIGN_VRAM_MB = 6000.0
"""Device memory other programs hold, so a staged Flux job does not fit beside both idle residents."""
_UNRETAINED_COMPONENT_MB = 600.0
"""Split encoder and VAE allocations each idle lane keeps on the card, above the release threshold, so each lane
is a cache-release target while it idles with its model in host RAM."""


@dataclass
class _SentCommand:
    """One control message the parent sent a lane, with the tick it went out on."""

    tick: int
    flag: HordeControlFlag
    model: str | None
    job_id: str | None


@dataclass
class _CommandLog:
    """Every control message each lane was sent, and what the parent's map said each lane was loading."""

    sent: dict[int, list[_SentCommand]] = field(default_factory=dict)
    loading_after_tick: dict[int, dict[int, str]] = field(default_factory=dict)
    """Per tick, each lane's model the parent's map recorded as ``LOADING`` once the tick had run."""
    now_after_tick: dict[int, float] = field(default_factory=dict)
    _read: dict[int, int] = field(default_factory=dict)

    def record(self, world: _DispatchWorld) -> None:
        """Read the messages each lane was sent this tick and the map's loading entries after it."""
        for lane in world.scheduler._process_map.values():
            for message in _pipe_messages(lane)[self._read.get(lane.process_id, 0) :]:
                self.sent.setdefault(lane.process_id, []).append(_sent_command(world.tick, message))
            self._read[lane.process_id] = len(_pipe_messages(lane))
        self.loading_after_tick[world.tick] = {
            entry.process_id: name
            for name, entry in world.scheduler._horde_model_map.root.items()
            if entry.horde_model_load_state == ModelLoadState.LOADING and entry.process_id is not None
        }
        self.now_after_tick[world.tick] = world.now

    def first_tick_at_or_after(self, instant: float) -> int | None:
        """The first tick whose end reached ``instant`` on the world clock."""
        return next((tick for tick, now in sorted(self.now_after_tick.items()) if now >= instant), None)


def _pipe_messages(lane: HordeProcessInfo) -> list[HordeControlMessage]:
    pipe = lane.pipe_connection
    sends = pipe.send.call_args_list  # type: ignore[attr-defined]
    return [call.args[0] for call in sends if call.args and isinstance(call.args[0], HordeControlMessage)]


def _sent_command(tick: int, message: HordeControlMessage) -> _SentCommand:
    model = message.horde_model_name if isinstance(message, HordeControlModelMessage) else None
    job_id = (
        str(message.sdk_api_job_info.id_)
        if isinstance(message, HordeInferenceControlMessage) and message.sdk_api_job_info.id_ is not None
        else None
    )
    return _SentCommand(tick=tick, flag=message.control_flag, model=model, job_id=job_id)


def _preloads_over_an_unresolved_load(log: _CommandLog) -> list[str]:
    """Each preload of one model sent to a lane the parent's map still recorded as loading another."""
    violations: list[str] = []
    for lane_id, commands in log.sent.items():
        for command in commands:
            if command.flag is not HordeControlFlag.PRELOAD_MODEL:
                continue
            loading = log.loading_after_tick.get(command.tick - 1, {}).get(lane_id)
            if loading is not None and loading != command.model:
                violations.append(
                    f"tick {command.tick}: lane {lane_id} sent a preload of {command.model} while still loading "
                    f"{loading}"
                )
    return violations


def _commands_ahead_of_first_step(world: _DispatchWorld, log: _CommandLog) -> list[str]:
    """Each preload of another model or unload sent to a lane between its job's dispatch and that job's first step."""
    violations: list[str] = []
    for lane_id, commands in log.sent.items():
        for index, start in enumerate(commands):
            if start.flag is not HordeControlFlag.START_INFERENCE or start.job_id is None:
                continue
            window = world.job_windows.get(start.job_id)
            first_step_tick = (
                log.first_tick_at_or_after(window.sample_from)
                if window is not None and window.sample_from is not None
                else None
            )
            for later in commands[index + 1 :]:
                if first_step_tick is not None and later.tick > first_step_tick:
                    break
                if later.flag is HordeControlFlag.START_INFERENCE:
                    break
                if later.flag is HordeControlFlag.UNLOAD_MODELS_FROM_VRAM or (
                    later.flag is HordeControlFlag.PRELOAD_MODEL and later.model != start.model
                ):
                    violations.append(
                        f"tick {later.tick}: lane {lane_id} sent {later.flag.name} {later.model} after job "
                        f"{start.job_id[:8]} ({start.model}) was dispatched at tick {start.tick}, before its first "
                        f"step (tick {first_step_tick})"
                    )
    return violations


async def _run_two_models_for_one_free_lane(
    *, preload_report_latency_seconds: float
) -> tuple[_DispatchWorld, _CommandLog, list[str]]:
    """Queue two jobs for two models while one lane samples a long job and the other is the only free lane.

    The card runs one job at a time, so neither queued job can start while lane 0 samples, and the preload pass
    walks past the first job (its load booked on lane 1) to the second. Returns the world, the command log, and
    the queued jobs' ids in pop order.
    """
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=2,
        max_threads=1,
        queue_depth=4,
        whole_card_enabled=False,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        preload_report_latency_seconds=preload_report_latency_seconds,
        preload_latency_seconds=1.0,
    )
    world.seed_resident(0, _SDXL, in_vram=True)
    world.seed_resident(1, _SDXL_OTHER, in_vram=False)
    await world.pop(make_job_pop_response(_SDXL.name, width=1024, height=1024, ddim_steps=60))
    queued = [make_job_pop_response(_SD15.name), make_job_pop_response(_SD15_OTHER.name)]
    for job in queued:
        await world.pop(job)
    log = _CommandLog()
    for _ in range(_MAX_TICKS):
        await world.step()
        log.record(world)
        if world.completed_jobs >= 1 + len(queued):
            break
    return world, log, [str(job.id_) for job in queued]


async def test_a_lane_loading_one_model_is_sent_no_preload_for_another() -> None:
    """A lane whose first report is still due is sent one preload, and its job starts behind that load alone.

    The first queued job's load is booked on the only free lane. Before its child reports, the lane reads idle
    and holding that model. A second queued job's preload aimed at it would displace the first model's map
    entry and re-plan the first model onto the same lane, and the two would alternate for as long as the report
    took, each send queueing ahead of the first job's start.
    """
    world, log, queued_ids = await _run_two_models_for_one_free_lane(
        preload_report_latency_seconds=_PRELOAD_REPORT_LATENCY_SECONDS
    )

    context = f"two queued models and one free lane whose first report takes {_PRELOAD_REPORT_LATENCY_SECONDS:.0f}s"
    assert world.completed_jobs == 3, (
        f"{context}: only {world.completed_jobs} of 3 jobs completed. {world.state_dump()}"
    )
    violations = _preloads_over_an_unresolved_load(log)
    assert not violations, f"{context}: " + "; ".join(violations)
    violations = _commands_ahead_of_first_step(world, log)
    assert not violations, f"{context}: " + "; ".join(violations)
    first_window = world.job_windows[queued_ids[0]]
    lane_commands = log.sent.get(first_window.lane_id, [])
    start_index = next(
        index
        for index, command in enumerate(lane_commands)
        if command.flag is HordeControlFlag.START_INFERENCE and command.job_id == queued_ids[0]
    )
    ahead_of_start = [
        command for command in lane_commands[:start_index] if command.flag is HordeControlFlag.PRELOAD_MODEL
    ]
    assert [command.model for command in ahead_of_start] == [_SD15.name], (
        f"{context}: lane {first_window.lane_id} was sent "
        f"{[(command.tick, command.model) for command in ahead_of_start]} ahead of the first job's start"
    )


@pytest.mark.parametrize("preload_report_latency_seconds", [0.0, _PRELOAD_REPORT_LATENCY_SECONDS])
async def test_an_idle_lane_still_receives_the_preload(preload_report_latency_seconds: float) -> None:
    """The commitment refuses only a lane already committed, so the free lane takes the first job's load at once.

    The load goes out on the first cycle that sees the job, whether the child reports at once or seconds later,
    and the second model is loaded once a lane is free for it.
    """
    world, log, queued_ids = await _run_two_models_for_one_free_lane(
        preload_report_latency_seconds=preload_report_latency_seconds
    )

    context = f"one free lane with a {preload_report_latency_seconds:.0f}s first report"
    assert world.completed_jobs == 3, (
        f"{context}: only {world.completed_jobs} of 3 jobs completed. {world.state_dump()}"
    )
    preloads = sorted(
        (command.tick, lane_id, command.model)
        for lane_id, commands in log.sent.items()
        for command in commands
        if command.flag is HordeControlFlag.PRELOAD_MODEL
    )
    busy_start_tick = next(
        command.tick for command in log.sent.get(0, []) if command.flag is HordeControlFlag.START_INFERENCE
    )
    assert preloads and preloads[0] == (busy_start_tick + 1, 1, _SD15.name), (
        f"{context}: the free lane was not sent the first job's preload on the cycle after lane 0 started; "
        f"sent {preloads}"
    )
    assert any(model == _SD15_OTHER.name for _tick, _lane, model in preloads), (
        f"{context}: the second model was never preloaded; sent {sorted(preloads)}"
    )
    assert queued_ids[1] in world.job_windows, f"{context}: the second job never dispatched. {world.state_dump()}"


@pytest.mark.parametrize("weights_on_device", [False, True], ids=["model_in_host_ram", "model_on_device"])
async def test_a_reclaim_rung_aimed_at_a_dispatched_lane_is_declined_and_the_next_rung_runs(
    weights_on_device: bool,
) -> None:
    """A lane that owns a dispatched job is no rung's target, and a rung aimed at it acts on nothing.

    A child's report from before the dispatch can land after it, so the lane reads idle. With its model in host RAM
    and its split components on the card it has the shape of a cache-release target, and with its model on the
    card that of an eviction. The ladder offers the next lane instead. A rung aimed at the dispatched lane is
    declined without a message, and the room lists the lane's tenancy as a rung that is not permitted.
    """
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=2,
        max_threads=2,
        queue_depth=2,
        whole_card_enabled=False,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        unretained_components_pinned=True,
    )
    world.seed_resident(0, _SD15, in_vram=weights_on_device)
    if not weights_on_device:
        world.seed_served_vram_unload(0)
        world.seed_unretained_components(0, _UNRETAINED_COMPONENT_MB)
    world.seed_resident(1, _SDXL, in_vram=False)
    world.seed_served_vram_unload(1)
    world.seed_unretained_components(1, _UNRETAINED_COMPONENT_MB)
    await world.step()
    job = make_job_pop_response(_SD15.name)
    await world.pop(job)
    scheduler = world.scheduler
    # Dispatched outside the world's tick, so no child effect follows. The row reads what the parent decides about
    # the lane in the window before its child reports the job, and the world never runs the job.
    scheduler.begin_scheduling_cycle()
    assert await scheduler.start_inference(), f"precondition: the job must dispatch. {world.state_dump()}"
    dispatched_lane = scheduler._process_map[0]
    owned = dispatched_lane.current_inference_job()
    assert owned is not None and owned.id_ == job.id_, "precondition: the job must be dispatched onto lane 0"
    # The child's last report from before the dispatch, landing after it.
    scheduler._process_map.on_process_state_change(process_id=0, new_state=HordeProcessState.WAITING_FOR_JOB)
    sent_before = {lane_id: len(_pipe_messages(scheduler._process_map[lane_id])) for lane_id in (0, 1)}

    rungs = build_reclaim_ladder(scheduler.build_reclaim_ladder_candidates(None))
    assert rungs, "precondition: the idle lane must leave the ladder a rung"
    assert all(rung.target_process_id != 0 for rung in rungs), (
        f"the dispatched lane is a ladder target: {[(rung.kind.value, rung.target_process_id) for rung in rungs]}"
    )
    for kind in (ReclaimRungKind.UNLOAD_IDLE_MODEL, ReclaimRungKind.RELEASE_IDLE_CACHE):
        aimed = ReclaimRung(
            kind=kind,
            device_index=None,
            promised_freed_mb=_UNRETAINED_COMPONENT_MB,
            tenant_label="the dispatched lane",
            target_process_id=0,
        )
        assert not execute_reclaim_rung(aimed, scheduler), f"a {kind.value} rung aimed at the dispatched lane acted"
    assert len(_pipe_messages(dispatched_lane)) == sent_before[0], (
        f"the dispatched lane was sent {_pipe_messages(dispatched_lane)[sent_before[0] :]}"
    )
    assert execute_reclaim_rung(rungs[0], scheduler), f"the next rung did not act: {rungs[0]}"
    assert len(_pipe_messages(scheduler._process_map[1])) > sent_before[1], "the next rung sent the idle lane nothing"

    state = scheduler.build_vram_arbiter_device_state(None)
    assert 0 not in state.idle_process_ids, "the arbiter may release the dispatched lane's cache"
    assert state.held_idle_returns_mb.get(0) == 0.0, (
        f"the room counts the dispatched lane's tenancy as returnable: {state.held_idle_returns_mb}"
    )
    teardownable = pricing.teardownable_idle_contexts(scheduler.snapshot(), 1, device_index=None)
    assert all(slot.process_id != 0 for slot in teardownable), "the dispatched lane's context counts as teardownable"


async def test_a_clearance_sweep_spares_the_queue_heads_ready_seat() -> None:
    """A clearance hold's sweep takes room from another idle resident and leaves the head's warm copy in place.

    A staged Flux job on lane 0 does not fit, and its hold runs the eviction sweep while lane 1 holds the queue
    head's SD15 on the card and lane 2 an SDXL model nothing queued wants. The card has a free sampling slot, so
    lane 1 is the head's ready seat. The sweep unloads only lane 2. The head starts on lane 1 with no load, and
    the hold is still admitted. Both jobs are queued before the first tick, since the Flux job's first clearance
    pass runs in the tick that stages it, and a sweep run before the head arrives has no ready seat to spare.
    """
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=3,
        max_threads=2,
        queue_depth=3,
        whole_card_enabled=False,
        closed_loop=True,
        clearance_lease=True,
        tick_seconds=_TICK_SECONDS,
        foreign_vram_mb_by_tick={1: _FOREIGN_VRAM_MB},
        partial_loads=True,
    )
    world.seed_resident(0, _FLUX, in_vram=False)
    world.seed_resident(1, _SD15, in_vram=True)
    world.seed_resident(2, _SDXL_OTHER, in_vram=True)
    staged = make_job_pop_response(_FLUX.name, width=1024, height=1024, ddim_steps=8)
    head = make_job_pop_response(_SD15.name)
    staged_id, head_id = str(staged.id_), str(head.id_)
    await world.pop(staged)
    await world.pop(head)
    log = _CommandLog()
    for _ in range(_MAX_TICKS):
        await world.step()
        log.record(world)
        if world.completed_jobs >= 2:
            break

    context = "a staged Flux hold sweeping beside the head's idle SD15 copy"
    assert world.completed_jobs == 2, (
        f"{context}: only {world.completed_jobs} of 2 jobs completed. {world.state_dump()}"
    )
    assert not world.clearance_timeouts, f"{context}: lane(s) {world.clearance_timeouts} timed out at clearance"
    sweeps = [
        record.tick
        for record in world.decision_records
        if record.decision_kind is DecisionKind.CLEARANCE
        and record.subject == staged_id
        and "evict_idle_model" in str(record.inputs.get("reclaim_applied_kinds") or "")
    ]
    assert sweeps, f"precondition: the Flux hold must run the eviction sweep. {world.state_dump()}"
    head_commands = log.sent.get(1, [])
    head_start = next(
        index
        for index, command in enumerate(head_commands)
        if command.flag is HordeControlFlag.START_INFERENCE and command.job_id == head_id
    )
    assert head_commands[head_start].tick > sweeps[0], (
        f"precondition: the head must still be queued when the hold sweeps at tick {sweeps[0]}, yet it started at "
        f"tick {head_commands[head_start].tick}. {world.state_dump()}"
    )
    unloads_ahead = [
        (command.tick, command.model)
        for command in head_commands[:head_start]
        if command.flag is HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
    ]
    assert not unloads_ahead, f"{context}: the head's ready seat was sent {unloads_ahead} ahead of the head's start"
    window = world.job_windows[head_id]
    assert window.lane_id == 1, f"{context}: the head started on lane {window.lane_id}, not on its warm copy"
    assert window.load_moved_mb == 0.0 and not window.loaded_now, (
        f"{context}: the head paid a load of {window.load_moved_mb:.0f} MB on its own warm copy"
    )
    assert any(
        command.flag is HordeControlFlag.UNLOAD_MODELS_FROM_VRAM and command.tick in sweeps
        for command in log.sent.get(2, [])
    ), f"{context}: the sweep took no room from the unwanted resident on lane 2. {world.state_dump()}"
