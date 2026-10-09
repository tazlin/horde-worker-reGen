"""The idle-lane warm: which idle lanes re-read their held checkpoint ahead of a pending job, and the pass sending it.

The selection rows build the snapshot's slots, queue and model map directly over a scheduler's base snapshot, so
each row states exactly the lane states and sizes its rule reads. The pass rows drive the scheduler's
``warm_idle_lanes`` over a real process map and model map.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeProcessState,
    HordeWarmInferenceModelMessage,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType, WorkerCapability
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling.admission.preload import (
    IdleLaneWarm,
    select_idle_lane_warms,
)
from horde_worker_regen.process_management.scheduling.admission.snapshot import (
    JobSnapshot,
    ModelMapEntry,
    QueueSnapshot,
    SchedulingSnapshot,
    SlotSnapshot,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_CHECKPOINT_MB = 6500.0


def _slot(
    process_id: int,
    *,
    model: str | None,
    state: HordeProcessState = HordeProcessState.WAITING_FOR_JOB,
    busy: bool = False,
    current_job_id: str | None = None,
) -> SlotSnapshot:
    return SlotSnapshot(
        process_id=process_id,
        os_pid=100000 + process_id,
        device_index=0,
        process_type=HordeProcessType.INFERENCE,
        capabilities=WorkerCapability.IMAGE_GEN,
        state=state,
        model=model,
        baseline=None,
        last_control_flag=None,
        reserved_mb=None,
        allocated_mb=None,
        ram_usage_bytes=0,
        retained_resident_model=None,
        retained_resident_since=None,
        retention_granted_model=None,
        is_busy=busy,
        can_accept_job=current_job_id is None,
        is_alive=True,
        is_unoccupied=current_job_id is None,
        current_job_id=current_job_id,
        held_component_count=0,
        resident_weight_models=frozenset(),
        aimdo_mb=None,
        parked_preload=False,
        reserved_for_disaggregation=False,
        reuse_credit_mb=0.0,
        checkpoint_models_held=frozenset(),
    )


def _job(job_id: str, model: str, *, in_progress: bool = False, size_mb: float | None = _CHECKPOINT_MB) -> JobSnapshot:
    return JobSnapshot(
        job_id=job_id,
        model=model,
        baseline=None,
        in_progress=in_progress,
        admitted_exclusive=False,
        admitted_over_budget=False,
        aux_models_prepared=True,
        degraded_dispatch_pending=False,
        disaggregation_class_eligible=False,
        tracked=True,
        measured_attempt_devices=frozenset(),
        measured_attempt_spent_devices=frozenset(),
        eligible_cards=frozenset({0}),
        requires_aux_preparation=False,
        popped_at=None,
        ttl=None,
        unserviceable_reason=None,
        component_charge_mb=None,
        staging_charge_mb=size_mb,
    )


def _held(slots: list[SlotSnapshot]) -> dict[str, ModelMapEntry]:
    """The model map of lanes holding their seated model in RAM."""
    return {
        slot.model: ModelMapEntry(
            model=slot.model, load_state=ModelLoadState.LOADED_IN_RAM, process_id=slot.process_id
        )
        for slot in slots
        if slot.model is not None and slot.current_job_id is None
    }


@pytest.fixture
def base_snapshot() -> SchedulingSnapshot:
    """A scheduler's snapshot over an empty pool, whose every other field the selection does not read."""
    return _make_inference_scheduler(process_map=ProcessMap({}), job_tracker=JobTracker()).snapshot()


def _snapshot(
    base: SchedulingSnapshot,
    *,
    slots: list[SlotSnapshot],
    pending: list[JobSnapshot],
    in_progress: tuple[JobSnapshot, ...] = (),
    model_map: dict[str, ModelMapEntry] | None = None,
    available_mb: float = 65536.0,
) -> SchedulingSnapshot:
    jobs = {job.job_id: job for job in (*in_progress, *pending)}
    queue = QueueSnapshot(
        pending_in_placement_order=tuple(job.job_id for job in pending),
        pending_in_pop_order=tuple(job.job_id for job in pending),
        in_progress=tuple(job.job_id for job in in_progress),
        jobs=jobs,
        payloads={},
    )
    return replace(
        base,
        slots={slot.process_id: slot for slot in slots},
        queue=queue,
        model_map=_held(slots) if model_map is None else model_map,
        host_ram=replace(base.host_ram, available_mb=available_mb),
    )


class TestSelection:
    """The walk over the pending queue, its protected set, and its room bound."""

    def test_idle_lanes_are_warmed_in_queue_order(self, base_snapshot: SchedulingSnapshot) -> None:
        """Each pending job's idle holder is warmed, in the order the queue holds the jobs."""
        slots = [_slot(0, model="b"), _slot(1, model="a")]
        snapshot = _snapshot(base_snapshot, slots=slots, pending=[_job("j1", "a"), _job("j2", "b")])

        warms = select_idle_lane_warms(snapshot, warmed=frozenset())

        assert [(warm.process_id, warm.model, warm.job_id) for warm in warms] == [(1, "a", "j1"), (0, "b", "j2")]

    def test_room_excludes_staged_lanes_and_earlier_pending_models(self, base_snapshot: SchedulingSnapshot) -> None:
        """A staged lane's model and every earlier pending model are priced out of the room; a job's own is not."""
        slots = [
            _slot(0, model="staged", state=HordeProcessState.INFERENCE_PRIMED, busy=True, current_job_id="s1"),
            _slot(1, model="a"),
            _slot(2, model="b"),
        ]
        snapshot = _snapshot(
            base_snapshot,
            slots=slots,
            pending=[_job("j1", "a"), _job("j2", "b")],
            in_progress=(_job("s1", "staged", in_progress=True),),
            available_mb=20000.0,
        )

        warms = select_idle_lane_warms(snapshot, warmed=frozenset())

        assert warms == [
            IdleLaneWarm(process_id=1, model="a", job_id="j1", room_mb=20000.0 - _CHECKPOINT_MB),
            IdleLaneWarm(process_id=2, model="b", job_id="j2", room_mb=20000.0 - 2 * _CHECKPOINT_MB),
        ]

    def test_a_sampling_lane_is_not_protected(self, base_snapshot: SchedulingSnapshot) -> None:
        """A lane past its first step holds its weights on the device, so its pages are not protected."""
        slots = [
            _slot(0, model="sampling", state=HordeProcessState.INFERENCE_STARTING, busy=True, current_job_id="s1"),
            _slot(1, model="a"),
        ]
        snapshot = _snapshot(
            base_snapshot,
            slots=slots,
            pending=[_job("j1", "a")],
            in_progress=(_job("s1", "sampling", in_progress=True),),
            available_mb=_CHECKPOINT_MB,
        )

        warms = select_idle_lane_warms(snapshot, warmed=frozenset())

        assert [(warm.process_id, warm.room_mb) for warm in warms] == [(1, _CHECKPOINT_MB)]

    def test_the_first_warm_that_does_not_fit_stops_the_walk(self, base_snapshot: SchedulingSnapshot) -> None:
        """An earlier warm that fits is kept; the first that does not ends the walk, later jobs included."""
        slots = [_slot(0, model="a"), _slot(1, model="b"), _slot(2, model="c")]
        snapshot = _snapshot(
            base_snapshot,
            slots=slots,
            pending=[_job("j1", "a"), _job("j2", "b"), _job("j3", "c", size_mb=1.0)],
            available_mb=_CHECKPOINT_MB * 1.5,
        )

        warms = select_idle_lane_warms(snapshot, warmed=frozenset())

        assert [warm.job_id for warm in warms] == ["j1"]

    def test_a_pending_job_without_a_holder_still_protects_its_model(self, base_snapshot: SchedulingSnapshot) -> None:
        """A job whose model no lane holds is not warmed, but a later warm may not evict its pages."""
        slots = [_slot(0, model="b")]
        snapshot = _snapshot(
            base_snapshot,
            slots=slots,
            pending=[_job("j1", "cold"), _job("j2", "b")],
            available_mb=_CHECKPOINT_MB * 1.5,
        )

        assert select_idle_lane_warms(snapshot, warmed=frozenset()) == []

    def test_a_model_of_unknown_size_is_not_warmed(self, base_snapshot: SchedulingSnapshot) -> None:
        """Without a checkpoint size the warm cannot be bounded, so it is not sent."""
        snapshot = _snapshot(base_snapshot, slots=[_slot(0, model="a")], pending=[_job("j1", "a", size_mb=None)])

        assert select_idle_lane_warms(snapshot, warmed=frozenset()) == []


class TestCandidates:
    """Which lanes may take a warm."""

    def test_a_pair_already_warmed_is_not_warmed_again(self, base_snapshot: SchedulingSnapshot) -> None:
        """Once per (lane, job): the scheduler's record of sent pairs suppresses a repeat."""
        snapshot = _snapshot(base_snapshot, slots=[_slot(0, model="a")], pending=[_job("j1", "a")])

        assert select_idle_lane_warms(snapshot, warmed=frozenset({(0, "j1")})) == []

    def test_a_lane_with_a_warm_outstanding_is_not_warmed_for_another_job(
        self,
        base_snapshot: SchedulingSnapshot,
    ) -> None:
        """A lane warmed for a still-pending job, or earlier in this walk, takes no second warm."""
        snapshot = _snapshot(base_snapshot, slots=[_slot(0, model="a")], pending=[_job("j1", "a"), _job("j2", "a")])

        assert [warm.job_id for warm in select_idle_lane_warms(snapshot, warmed=frozenset())] == ["j1"]
        assert select_idle_lane_warms(snapshot, warmed=frozenset({(0, "j1")})) == []

    def test_a_parked_preload_of_the_model_is_a_candidate(self, base_snapshot: SchedulingSnapshot) -> None:
        """A lane parked on a finished preload holds the model in RAM with no job, though dispatch counts it busy."""
        snapshot = _snapshot(
            base_snapshot,
            slots=[_slot(0, model="a", state=HordeProcessState.PRELOADED_MODEL, busy=True)],
            pending=[_job("j1", "a")],
        )

        assert [warm.process_id for warm in select_idle_lane_warms(snapshot, warmed=frozenset())] == [0]

    @pytest.mark.parametrize(
        "state",
        [HordeProcessState.PRELOADING_MODEL, HordeProcessState.DOWNLOADING_MODEL],
        ids=["preloading", "busy"],
    )
    def test_a_busy_lane_is_skipped(self, base_snapshot: SchedulingSnapshot, state: HordeProcessState) -> None:
        """A lane at work with no job, a load or a download, is not idle; busy is derived from the state."""
        snapshot = _snapshot(
            base_snapshot,
            slots=[_slot(0, model="a", state=state, busy=True)],
            pending=[_job("j1", "a")],
        )

        assert select_idle_lane_warms(snapshot, warmed=frozenset()) == []

    def test_a_lane_seated_with_another_model_is_skipped(self, base_snapshot: SchedulingSnapshot) -> None:
        """A lane holding a different model has nothing of the job's to read."""
        snapshot = _snapshot(base_snapshot, slots=[_slot(0, model="b")], pending=[_job("j1", "a")])

        assert select_idle_lane_warms(snapshot, warmed=frozenset()) == []

    @pytest.mark.parametrize(
        "model_map",
        [
            {"a": ModelMapEntry(model="a", load_state=ModelLoadState.LOADED_IN_VRAM, process_id=0)},
            {"a": ModelMapEntry(model="a", load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)},
            {
                "a": ModelMapEntry(model="a", load_state=ModelLoadState.LOADED_IN_RAM, process_id=0),
                "b": ModelMapEntry(model="b", load_state=ModelLoadState.LOADING, process_id=0),
            },
        ],
        ids=["held-in-vram", "another-holder", "loading-another-model"],
    )
    def test_the_model_map_must_book_the_lane_as_holding_the_model_in_ram(
        self,
        base_snapshot: SchedulingSnapshot,
        model_map: dict[str, ModelMapEntry],
    ) -> None:
        """The lane is the map's ``LOADED_IN_RAM`` holder of the model, with no load of another model coming."""
        snapshot = _snapshot(
            base_snapshot,
            slots=[_slot(0, model="a")],
            pending=[_job("j1", "a")],
            model_map=model_map,
        )

        assert select_idle_lane_warms(snapshot, warmed=frozenset()) == []

    def test_dispatched_and_in_progress_jobs_are_skipped(self, base_snapshot: SchedulingSnapshot) -> None:
        """Only jobs still waiting for a lane are warmed for."""
        slots = [
            _slot(0, model="a"),
            _slot(1, model="a", state=HordeProcessState.INFERENCE_PRIMED, current_job_id="j2"),
        ]
        snapshot = _snapshot(
            base_snapshot,
            slots=slots,
            pending=[_job("j1", "a", in_progress=True), _job("j2", "a")],
            model_map={"a": ModelMapEntry(model="a", load_state=ModelLoadState.LOADED_IN_RAM, process_id=0)},
        )

        assert select_idle_lane_warms(snapshot, warmed=frozenset()) == []


async def _pass_worker(*, lease: bool) -> tuple[InferenceScheduler, str]:
    """A scheduler with one idle lane holding the one pending job's model in RAM."""
    bridge_data = make_mock_bridge_data()
    bridge_data.gpu_sampling_lease_enabled = lease
    tracker = JobTracker()
    scheduler = _make_inference_scheduler(
        process_map=ProcessMap({0: make_mock_process_info(0, model_name="a")}),
        job_tracker=tracker,
        bridge_data=bridge_data,
    )
    scheduler._horde_model_map.update_entry(  # type: ignore[attr-defined]
        horde_model_name="a",
        load_state=ModelLoadState.LOADED_IN_RAM,
        process_id=0,
    )
    scheduler._checkpoint_staging_charge_mb = Mock(return_value=_CHECKPOINT_MB)  # type: ignore[method-assign]
    job = make_job_pop_response(model="a")
    await track_popped_job_async(tracker, job)
    return scheduler, str(job.id_)


def _sent_warms(scheduler: InferenceScheduler) -> list[HordeWarmInferenceModelMessage]:
    pipe = scheduler._process_map[0].pipe_connection
    assert isinstance(pipe, Mock)
    return [
        call.args[0] for call in pipe.send.call_args_list if isinstance(call.args[0], HordeWarmInferenceModelMessage)
    ]


class TestWarmPass:
    """The scheduler's pass: a message only, once per pair, under the lease."""

    async def test_the_warm_is_a_message_and_books_nothing(self) -> None:
        """The lane is sent the warm; its control flag, the model map and the planned charges are untouched."""
        scheduler, job_id = await _pass_worker(lease=True)
        lane = scheduler._process_map[0]

        assert scheduler.warm_idle_lanes() == 1

        assert [message.horde_model_name for message in _sent_warms(scheduler)] == ["a"]
        assert _sent_warms(scheduler)[0].control_flag is HordeControlFlag.WARM_MODEL
        assert lane.last_control_flag is None
        assert scheduler._horde_model_map.root["a"].horde_model_load_state is ModelLoadState.LOADED_IN_RAM
        assert scheduler._idle_lane_warmed == {(0, job_id)}
        assert scheduler._reserve_ledger.total_ram_mb() == 0.0

    async def test_a_pair_is_warmed_once(self) -> None:
        """A second pass while the job stays pending sends nothing."""
        scheduler, _job_id = await _pass_worker(lease=True)

        scheduler.warm_idle_lanes()

        assert scheduler.warm_idle_lanes() == 0
        assert len(_sent_warms(scheduler)) == 1

    async def test_a_record_whose_job_left_the_queue_is_dropped(self) -> None:
        """The record of sent pairs keeps only jobs still pending, so it does not grow."""
        scheduler, _job_id = await _pass_worker(lease=True)
        scheduler._idle_lane_warmed.add((0, "gone"))

        scheduler.warm_idle_lanes()

        assert (0, "gone") not in scheduler._idle_lane_warmed

    async def test_without_the_lease_nothing_is_sent(self) -> None:
        """Without the lease a dispatch is the load, and the job-start prefetch covers it."""
        scheduler, _job_id = await _pass_worker(lease=False)

        assert scheduler.warm_idle_lanes() == 0
        assert _sent_warms(scheduler) == []


class TestWarmUnderCommitBoundHost:
    """No warm is sent while available commit, not physical RAM, bounds the host."""

    async def test_a_commit_bound_host_sends_no_warm(self) -> None:
        """A warm the cache cannot serve maps the checkpoint, the one charge a commit-bound host refuses."""
        scheduler, _job_id = await _pass_worker(lease=True)
        scheduler.set_available_ram_mb_provider(lambda: 60000.0)
        scheduler.set_available_commit_mb_provider(lambda: 8000.0)

        assert scheduler.warm_idle_lanes() == 0
        assert _sent_warms(scheduler) == []

        scheduler.set_available_commit_mb_provider(lambda: None)
        assert scheduler.warm_idle_lanes() == 1
