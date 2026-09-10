"""The per-cycle scheduling snapshot: every field mirrors what the pipelines read, and building it changes nothing."""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api import GENERATION_STATE

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeImageResult,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from horde_worker_regen.process_management.scheduling import inference_scheduler as _sched_mod
from horde_worker_regen.process_management.scheduling.admission.snapshot import SchedulingSnapshot, snapshot_slot
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_test_card_runtimes,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _sent_control_flags(process_map: ProcessMap) -> list[HordeControlFlag]:
    flags: list[HordeControlFlag] = []
    for process_info in process_map.values():
        send: Mock = process_info.pipe_connection.send  # type: ignore[assignment]
        flags.extend(call.args[0].control_flag for call in send.call_args_list)
    return flags


class TestSlotSnapshot:
    """A slot snapshot mirrors the process record's gate-facing fields."""

    def test_mirrors_the_process_record(self) -> None:
        """Identity, memory, residency and the derived predicates match the record they were taken from."""
        process_info = make_mock_process_info(3, model_name="sd", state=HordeProcessState.WAITING_FOR_JOB)
        process_info.process_reserved_mb = 4000
        process_info.process_allocated_mb = 3500
        process_info.ram_usage_bytes = 5 * 1024 * 1024
        process_info.retained_resident_model = "sd"
        process_info.retained_resident_since = 990.0
        process_info.last_control_flag = HordeControlFlag.PRELOAD_MODEL
        model_map = HordeModelMap(root={})
        model_map.update_entry(horde_model_name="sd", load_state=ModelLoadState.LOADED_IN_VRAM, process_id=3)
        slot = snapshot_slot(process_info, model_map, now=1_000.0)
        assert (slot.process_id, slot.model, slot.state) == (3, "sd", HordeProcessState.WAITING_FOR_JOB)
        assert (slot.reserved_mb, slot.allocated_mb, slot.ram_usage_bytes) == (4000, 3500, 5 * 1024 * 1024)
        assert slot.retained_resident_model == "sd"
        assert slot.retained_resident_since == 990.0
        assert slot.last_control_flag is HordeControlFlag.PRELOAD_MODEL
        assert slot.is_busy == process_info.is_process_busy()
        assert slot.can_accept_job == process_info.can_accept_job()
        assert slot.is_unoccupied == process_info.is_unoccupied()
        assert slot.current_job_id is None
        assert slot.resident_weight_models == frozenset({"sd"})
        assert slot.capabilities == process_info.capabilities
        assert slot.reserved_for_disaggregation is False
        assert slot.reuse_credit_mb == 0.0 and slot.checkpoint_models_held == frozenset()

    def test_a_pinned_sampler_is_marked_reserved(self) -> None:
        """The disaggregation pin is read from the process map, not the record."""
        process_info = make_mock_process_info(3, model_name="sd")
        slot = snapshot_slot(process_info, HordeModelMap(root={}), now=1_000.0, reserved_for_disaggregation=True)
        assert slot.reserved_for_disaggregation is True


class TestSchedulingSnapshot:
    """Building a snapshot reads every collaborator once and mutates none of them."""

    async def _scheduler(self) -> tuple[SchedulingSnapshot, object]:
        idle = make_mock_process_info(0, model_name="sd", state=HordeProcessState.WAITING_FOR_JOB)
        busy = make_mock_process_info(1, model_name="xl", state=HordeProcessState.INFERENCE_STARTING)
        process_map = ProcessMap({0: idle, 1: busy})
        job_tracker = JobTracker()
        model_map = HordeModelMap(root={})
        model_map.update_entry(horde_model_name="sd", load_state=ModelLoadState.LOADED_IN_VRAM, process_id=0)
        clock = _Clock()
        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            horde_model_map=model_map,
            clock=clock,
            device_free_mb=9000.0,
        )
        head = make_job_pop_response(model="sd")
        second = make_job_pop_response(model="xl")
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, second)
        # The worker-wide key resolves to device 0, and measured free is the children's reported figure capped
        # by the parent's device reading, exactly as the scheduler's own accessors resolve them.
        scheduler.set_governor_state(0, GovernorState.PRESSURE)
        scheduler._process_map.get_free_vram_mb = Mock(return_value=12000.0)  # type: ignore[method-assign]
        return scheduler.snapshot(), scheduler

    async def test_cards_queue_slots_and_ledgers_mirror_the_scheduler(self) -> None:
        """The worker-wide card, the queue in placement order, both slots and the ledger views are all present."""
        snapshot, scheduler = await self._scheduler()
        assert snapshot.multi_gpu_routing_active is False
        assert set(snapshot.cards) == {None}
        card = snapshot.card(0)
        assert card.measured_free_mb == 9000.0, "the parent's device reading caps the children's 12000"
        assert card.measured_free_mb == scheduler._measured_free_vram_mb(device_index=None)  # type: ignore[attr-defined]
        assert card.governor_state is GovernorState.PRESSURE
        assert card.growth_held is False
        assert card.whole_card_model is None and card.whole_card_phase is None

        assert set(snapshot.slots) == {0, 1}
        assert snapshot.slots[1].is_busy is True and snapshot.slots[0].is_busy is False
        assert snapshot.routing_device_index(snapshot.slots[0]) is None

        assert len(snapshot.queue.pending_in_placement_order) == 2
        assert snapshot.queue.in_progress == ()
        head_id = snapshot.queue.head()
        assert head_id is not None and snapshot.queue.jobs[head_id].model == "sd"
        assert snapshot.queue.payloads[head_id].model == "sd"
        assert snapshot.model_loaded("sd") is True and snapshot.model_loaded("xl") is False
        assert snapshot.model_map["sd"].process_id == 0
        assert snapshot.slots[0].resident_weight_models == frozenset({"sd"})
        assert snapshot.queue.jobs[head_id].tracked is True
        assert snapshot.queue.jobs[head_id].measured_attempt_devices == frozenset()
        assert snapshot.queue.jobs[head_id].requires_aux_preparation is False
        assert snapshot.queue.jobs[head_id].popped_at is not None
        assert snapshot.queue.jobs[head_id].unserviceable_reason is None
        assert snapshot.queue.jobs[head_id].component_charge_mb is None
        assert snapshot.services.ram_budget is scheduler._ram_budget  # type: ignore[attr-defined]
        assert card.exclusive_job_in_progress is False
        assert snapshot.draining_process_ids == frozenset()
        assert snapshot.shutting_down is False
        assert snapshot.max_concurrent_inference_processes == 1
        assert snapshot.host_ram.pressure.under_pressure is False, "a live reading stands in before the first tick"
        assert card.config is scheduler._runtime_config.bridge_data  # type: ignore[attr-defined]
        assert snapshot.config_for(None) is card.config
        assert snapshot.safety_footprint_mb > 0.0

        assert snapshot.ledgers.retention.wddm_paging_active is False
        assert snapshot.ledgers.dispatch_holds.hold_since == {}
        assert snapshot.ledgers.head_admission.barrier_job_id is None
        assert snapshot.host_ram.total_mb > 0.0
        assert snapshot.now == 1_000.0
        assert snapshot.services.model_metadata is scheduler._model_metadata  # type: ignore[attr-defined]

    async def test_building_twice_is_pure(self) -> None:
        """Two snapshots of an unchanged worker are equal, and no child received a message."""
        snapshot, scheduler = await self._scheduler()
        again = scheduler.snapshot()  # type: ignore[attr-defined]
        assert again.cards == snapshot.cards
        assert again.slots == snapshot.slots
        assert again.queue.pending_in_placement_order == snapshot.queue.pending_in_placement_order
        assert again.ledgers == snapshot.ledgers
        assert _sent_control_flags(scheduler._process_map) == []  # type: ignore[attr-defined]

    async def test_multi_gpu_cards_carry_their_own_config_and_the_worker_view(self) -> None:
        """A two-card host snapshots each card plus the worker-wide view, and scopes slots to their card."""
        card_runtimes = make_test_card_runtimes(device_indices=(0, 1))
        slot = make_mock_process_info(0, model_name="sd", state=HordeProcessState.WAITING_FOR_JOB)
        slot.device_index = 1
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: slot}),
            card_runtimes=card_runtimes,
            device_free_mb=6000.0,
        )
        snapshot = scheduler.snapshot()
        assert snapshot.multi_gpu_routing_active is True
        assert set(snapshot.cards) == {None, 0, 1}
        assert snapshot.cards[1].config is card_runtimes[1].config
        assert snapshot.cards[None].config is scheduler._runtime_config.bridge_data  # type: ignore[attr-defined]
        assert snapshot.config_for(1) is card_runtimes[1].config
        assert snapshot.routing_device_index(snapshot.slots[0]) == 1
        assert snapshot.card(1) is snapshot.cards[1]
        assert snapshot.slots_of_type(HordeProcessType.INFERENCE) == (snapshot.slots[0],)

    async def test_the_card_count_excludes_the_worker_wide_view(self) -> None:
        """``card_count`` counts real cards only, so a bound expressed in cards is not inflated by one.

        The worker-wide entry is keyed ``None`` alongside the per-card ones, and a copy bound that counted it
        would permit one more copy of a model than there are cards to sample it on.
        """
        slot = make_mock_process_info(0, model_name="sd", state=HordeProcessState.WAITING_FOR_JOB)
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: slot}),
            card_runtimes=make_test_card_runtimes(device_indices=(0, 1)),
        )

        snapshot = scheduler.snapshot()

        assert set(snapshot.cards) == {None, 0, 1}
        assert snapshot.card_count == 2

    async def test_one_card_counts_as_one_card(self) -> None:
        """Card-agnostic routing still reports a card, so a per-card bound stays a number rather than zero."""
        snapshot, _ = await self._scheduler()

        assert snapshot.multi_gpu_routing_active is False
        assert snapshot.card_count == 1


class TestPendingPostProcessingFacts:
    """The preload gate's two post-processing facts: which card the lane sits on, and what a chain will cost."""

    async def _scheduler_with_lane(self, *, lane_card: int) -> SchedulingSnapshot:
        """A two-card worker whose post-processing lane sits on ``lane_card`` with one chain pending."""
        inference_lane = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        post_process_lane = make_mock_process_info(
            7,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.POST_PROCESS,
            device_index=lane_card,
        )
        job_tracker = JobTracker()
        pending_chain = make_job_pop_response("sd", post_processing=["RealESRGAN_x4plus"])
        await job_tracker.queue_for_post_processing(
            HordeJobInfo(
                sdk_api_job_info=pending_chain,
                job_image_results=[HordeImageResult(image_bytes=b"raw")],
                state=GENERATION_STATE.ok,
                censored=False,
                time_popped=time.time(),
            ),
        )
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: inference_lane, 7: post_process_lane}),
            job_tracker=job_tracker,
            card_runtimes=make_test_card_runtimes(device_indices=(0, 1)),
        )
        return scheduler.snapshot()

    async def test_the_lane_card_and_the_smallest_pending_peak_are_pinned(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The card carrying the idle lane, and the chain estimate the hold weighs, are both on the snapshot."""
        monkeypatch.setattr(_sched_mod, "predict_job_post_processing_vram_mb", lambda _job, _baseline: 4000.0)

        snapshot = await self._scheduler_with_lane(lane_card=1)

        assert snapshot.post_processing_lane_card_index == 1
        assert snapshot.pending_post_processing_reserve_mb == 4000.0

    async def test_an_unknown_chain_estimate_reads_as_nothing_pending(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no estimate for the pending chain the reserve is zero: the gate restricts only on evidence."""
        monkeypatch.setattr(_sched_mod, "predict_job_post_processing_vram_mb", lambda _job, _baseline: None)

        snapshot = await self._scheduler_with_lane(lane_card=0)

        assert snapshot.post_processing_lane_card_index == 0
        assert snapshot.pending_post_processing_reserve_mb == 0.0

    async def test_no_lane_names_no_card(self) -> None:
        """A worker with no post-processing lane names no card for the hold to apply to."""
        scheduler = _make_inference_scheduler(
            process_map=ProcessMap({0: make_mock_process_info(0, model_name=None)}),
            card_runtimes=make_test_card_runtimes(device_indices=(0, 1)),
        )

        snapshot = scheduler.snapshot()

        assert snapshot.post_processing_lane_card_index is None
        assert snapshot.pending_post_processing_reserve_mb == 0.0
