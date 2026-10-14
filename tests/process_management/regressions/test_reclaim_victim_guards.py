"""Victim-selection guards for the VRAM eviction and the stale-RAM slot recycle.

Both reclaim mechanisms pick a victim from the live pool, and both sit on paths where a wrong pick
manufactures the very stall they exist to relieve:

- `InferenceScheduler.unload_models_from_vram` frees room for an incoming load. Which idle copy it leaves in
  place is decided by `pricing.sweep_spares_idle_copy`. A ready seat (an idle lane holding a model a pending
  job can start on this cycle) is spared, because dispatch seats that job there later in the same cycle and
  unloading it only buys a reload. The head-of-queue escalation keeps only the head's own seat and overrides
  every other sparing, because by then the head has priority for the room.
- `InferenceScheduler._replace_stale_ram_unload_process` cycles an idle slot whose allocator retained
  a freed model's pages, the only way to return that RAM to the OS. Its preconditions (idle,
  model-less, most recently told to unload from RAM, above the retention threshold) are each a guard
  against recycling a slot that is doing or about to do useful work; in particular a slot whose most
  recent control message is a preload must never be reaped mid-stage, since the queue head is counting
  on it. Retention itself is an evidence question: a child samples its RSS on a fixed interval, so
  "the unload returned nothing" may only be read off a report sampled after the unload, against a
  threshold no freshly spawned child reaches, and one slot may not answer every reclaim attempt.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import MEMORY_REPORT_INTERVAL_SECONDS
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap, ModelLoadState
from horde_worker_regen.process_management.scheduling.admission import pricing
from horde_worker_regen.process_management.scheduling.ledgers.ram_reclaim import FRESH_INFERENCE_CHILD_BASELINE_MB
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    mark_ram_unload_settled,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_NEXT_MODEL = "CyberRealistic Pony"
_OTHER_MODEL = "WAI-NSFW-illustrious-SDXL"

_RETAINED_BYTES = 16 * 1024 * 1024 * 1024


class TestVramUnloadNextModelGuard:
    """The VRAM sweep spares a ready seat for the head, and the escalation overrides a follower's seat."""

    async def _scheduler_with_resident(self, queued_models: list[str]):  # noqa: ANN202
        """Return a requester slot and a sibling holding the next model, over a real queue of ``queued_models``."""
        requester = make_mock_process_info(1, model_name=None)
        sibling = make_mock_process_info(2, model_name=_NEXT_MODEL)
        process_map = ProcessMap({1: requester, 2: sibling})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name=_NEXT_MODEL,
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=2,
        )
        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=JobTracker(),
            bridge_data=make_mock_bridge_data(max_threads=2),
            max_concurrent=2,
            max_inference=2,
        )
        for model in queued_models:
            await track_popped_job_async(scheduler._job_tracker, make_job_pop_response(model))
        return scheduler, requester, sibling

    async def test_gentle_reclaim_spares_the_next_model(self) -> None:
        """A lane holding the queue head's model is spared as the head's ready seat, by both sweeps.

        Dispatch seats the head on that lane later in the same cycle with no load. Evicting the copy trades one
        deferred load for two: the incoming model gains room by unloading weights the very next dispatch must
        immediately reload. The head escalation keeps this seat too, since it is the head's own.
        """
        scheduler, requester, sibling = await self._scheduler_with_resident([_NEXT_MODEL])
        assert sibling.process_id in pricing.ready_seat_process_ids(scheduler.snapshot())

        freed = scheduler.unload_models_from_vram(requester, under_pressure=True)
        freed_for_head = scheduler.unload_models_from_vram(requester, under_pressure=True, for_head_of_queue=True)

        assert freed is False
        assert freed_for_head is False
        assert sibling.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM

    async def test_head_escalation_may_evict_the_next_model(self) -> None:
        """CONTROL: the head-of-queue escalation overrides a follower's ready seat, since the head has priority."""
        scheduler, requester, sibling = await self._scheduler_with_resident([_OTHER_MODEL, _NEXT_MODEL])
        assert sibling.process_id in pricing.ready_seat_process_ids(scheduler.snapshot())

        freed = scheduler.unload_models_from_vram(requester, under_pressure=True, for_head_of_queue=True)

        assert freed is True
        assert sibling.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM


class TestVramUnloadSparesEveryLookaheadModel:
    """Every queued model with a ready seat keeps its resident copy under gentle reclaim.

    The guard is set membership, not a single element: with several distinct models queued, sparing
    only one of them (as an order-dependent pick would) evicts weights another queued job must
    immediately reload. Only a resident copy no queued job wants is fair game.
    """

    _UNQUEUED_MODEL = "AlbedoBase XL (SDXL)"

    async def _scheduler_with_three_residents(self):  # noqa: ANN202
        """Return lanes holding the head's model, a follower's model and an unqueued model, with both jobs queued."""
        requester = make_mock_process_info(1, model_name=None)
        head_holder = make_mock_process_info(2, model_name=_NEXT_MODEL)
        tail_holder = make_mock_process_info(3, model_name=_OTHER_MODEL)
        idle_holder = make_mock_process_info(4, model_name=self._UNQUEUED_MODEL)
        process_map = ProcessMap({1: requester, 2: head_holder, 3: tail_holder, 4: idle_holder})
        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=HordeModelMap(root={}),
            job_tracker=JobTracker(),
            bridge_data=make_mock_bridge_data(max_threads=2),
            max_concurrent=2,
            max_inference=4,
        )
        head_job = make_job_pop_response(_NEXT_MODEL)
        tail_job = make_job_pop_response(_OTHER_MODEL)
        await track_popped_job_async(scheduler._job_tracker, head_job)
        await track_popped_job_async(scheduler._job_tracker, tail_job)
        return scheduler, str(tail_job.id_), requester, head_holder, tail_holder, idle_holder

    async def test_lookahead_models_spared_unqueued_model_evicted(self) -> None:
        """Gentle reclaim leaves both queued models' ready seats resident and takes the copy nothing queued wants."""
        scheduler, _, requester, head_holder, tail_holder, idle_holder = await self._scheduler_with_three_residents()
        assert pricing.ready_seat_process_ids(scheduler.snapshot()) == frozenset(
            {head_holder.process_id, tail_holder.process_id}
        )

        freed = scheduler.unload_models_from_vram(requester, under_pressure=True)

        assert freed is True
        assert head_holder.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
        assert tail_holder.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
        assert idle_holder.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM

    async def test_lookahead_protection_dropped_when_card_cannot_afford_coresidency(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On a card that cannot hold a queued copy beside the head's sampling, only a seatless copy is evicted.

        A lookahead copy with no ready seat (its job held by a dispatch gate) is spared only while it fits beside
        the head's sampling peak. Sparing it on a card that cannot hold both forces driver demand-paging during
        the head's sampling (silent on WDDM, which keeps reporting free VRAM while saturated), costing far more
        than the one reload the protection saves. The affordability judgment is static: reported total minus
        reserve minus the head job's sampling peak minus the resident's weight footprint.

        The head's own ready seat is spared whatever that judgment says. Its copy is the weights the head samples
        with once dispatch seats it there, so it is no co-resident beside the head's peak, and unloading it only
        buys the head a reload.
        """
        (
            scheduler,
            tail_job_id,
            requester,
            head_holder,
            tail_holder,
            idle_holder,
        ) = await self._scheduler_with_three_residents()
        # A 16 GB card: peak 8258 + footprint 6600 + reserve leaves no room for co-residency.
        for process_info in (requester, head_holder, tail_holder, idle_holder):
            process_info.total_vram_mb = 16376
        monkeypatch.setattr(pricing, "predict_job_sampling_vram_mb", lambda job, baseline: 8258.0)
        monkeypatch.setattr(pricing, "predict_job_footprint_mb", lambda job, baseline: 6600.0)
        scheduler._dispatch_holds.note_pp_defer(tail_job_id, deferred=True)
        snapshot = scheduler.snapshot()
        assert pricing.ready_seat_process_ids(snapshot) == frozenset({head_holder.process_id})
        assert pricing.coresident_lookahead_affordable(snapshot, _OTHER_MODEL, device_index=None) is False

        freed = scheduler.unload_models_from_vram(requester, under_pressure=True)

        assert freed is True
        assert head_holder.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
        assert tail_holder.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
        assert idle_holder.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM


class TestStaleRamUnloadRecycleScope:
    """The stale-RAM recycle fires only on a provably idle, drained, allocator-stuck slot."""

    def _make_scheduler(self, process_info):  # noqa: ANN001, ANN202
        return _make_inference_scheduler(
            process_map=ProcessMap({0: process_info}),
            job_tracker=JobTracker(),
            bridge_data=make_mock_bridge_data(max_threads=2),
            max_concurrent=2,
            max_inference=2,
        )

    def _stale_slot(self):  # noqa: ANN202
        """An idle, model-less slot that kept multi-GB RAM after its unload: the intended victim."""
        process_info = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_RAM
        process_info.ram_usage_bytes = _RETAINED_BYTES
        mark_ram_unload_settled(process_info)
        return process_info

    def test_eligible_slot_is_recycled_as_intentional_reclaim(self) -> None:
        """The intended victim is cycled deliberately (no crash bookkeeping, see intentional_reclaim)."""
        process_info = self._stale_slot()
        scheduler = self._make_scheduler(process_info)

        cycled = scheduler._replace_stale_ram_unload_process()

        assert cycled is True
        scheduler._process_lifecycle._replace_inference_process.assert_called_once()
        _, kwargs = scheduler._process_lifecycle._replace_inference_process.call_args
        assert kwargs.get("intentional_reclaim") is True

    def test_slot_routed_a_preload_is_not_recycled(self) -> None:
        """A slot whose latest control message is a preload is off-limits.

        The admission pipeline just chose this slot for the queue head; reaping it mid-stage would
        fault the head's load and re-enter the deferral it was admitted to escape.
        """
        process_info = self._stale_slot()
        process_info.last_control_flag = HordeControlFlag.PRELOAD_MODEL
        scheduler = self._make_scheduler(process_info)

        assert scheduler._replace_stale_ram_unload_process() is False
        scheduler._process_lifecycle._replace_inference_process.assert_not_called()

    def test_busy_slot_is_not_recycled(self) -> None:
        """A slot actively preloading is never reaped, whatever its RAM footprint reads."""
        process_info = self._stale_slot()
        process_info.last_process_state = HordeProcessState.PRELOADING_MODEL
        scheduler = self._make_scheduler(process_info)

        assert scheduler._replace_stale_ram_unload_process() is False
        scheduler._process_lifecycle._replace_inference_process.assert_not_called()

    def test_slot_still_holding_a_model_is_not_recycled(self) -> None:
        """A slot with a resident model is a warm cache, not an allocator-stuck husk."""
        process_info = self._stale_slot()
        process_info.loaded_horde_model_name = _OTHER_MODEL
        scheduler = self._make_scheduler(process_info)

        assert scheduler._replace_stale_ram_unload_process() is False
        scheduler._process_lifecycle._replace_inference_process.assert_not_called()

    def test_slot_below_retention_threshold_is_not_recycled(self) -> None:
        """A slot that actually released its RAM has nothing to reclaim; respawning it is pure churn."""
        process_info = self._stale_slot()
        process_info.ram_usage_bytes = 256 * 1024 * 1024
        scheduler = self._make_scheduler(process_info)

        assert scheduler._replace_stale_ram_unload_process() is False
        scheduler._process_lifecycle._replace_inference_process.assert_not_called()

    def test_reading_older_than_the_unload_is_not_evidence(self) -> None:
        """A slot whose only RSS reading predates its unload is not a candidate.

        That reading describes the footprint the unload was sent to release, so treating it as retention
        cycles slots that did release, paying a respawn and a cold load for nothing. The reclaim waits for a
        reading that can actually answer the question.
        """
        process_info = self._stale_slot()
        assert process_info.last_ram_unload_requested_at is not None
        process_info.report_sampled_at = process_info.last_ram_unload_requested_at - 1.0
        scheduler = self._make_scheduler(process_info)

        assert scheduler._replace_stale_ram_unload_process() is False
        scheduler._process_lifecycle._replace_inference_process.assert_not_called()

    def test_reading_taken_a_full_interval_after_the_unload_is_evidence(self) -> None:
        """CONTROL: the same slot, with a reading one report interval past the unload, is cycled."""
        process_info = self._stale_slot()
        assert process_info.last_ram_unload_requested_at is not None
        process_info.report_sampled_at = process_info.last_ram_unload_requested_at + MEMORY_REPORT_INTERVAL_SECONDS
        scheduler = self._make_scheduler(process_info)

        assert scheduler._replace_stale_ram_unload_process() is True
        scheduler._process_lifecycle._replace_inference_process.assert_called_once()

    def test_slot_at_the_fresh_child_baseline_is_not_recycled(self) -> None:
        """A post-unload reading at a cold child's baseline RSS is a released slot, not a stuck one.

        The threshold sits above what any freshly spawned child carries, so a slot at the baseline can never
        qualify; were it able to, every cycle's own successor would qualify and the pool would cycle forever.
        """
        process_info = self._stale_slot()
        process_info.ram_usage_bytes = int(FRESH_INFERENCE_CHILD_BASELINE_MB * 1024 * 1024)
        scheduler = self._make_scheduler(process_info)

        assert scheduler._replace_stale_ram_unload_process() is False
        scheduler._process_lifecycle._replace_inference_process.assert_not_called()

    def test_same_slot_is_not_cycled_twice_inside_the_minimum_interval(self) -> None:
        """One slot answers at most one stale-unload cycle per window.

        Its successor respawns and cold-loads, so cycling the same slot again on the next reclaim attempt
        spends the pool faster than the returned RAM is worth.
        """
        process_info = self._stale_slot()
        scheduler = self._make_scheduler(process_info)

        assert scheduler._replace_stale_ram_unload_process() is True
        assert scheduler._replace_stale_ram_unload_process() is False
        scheduler._process_lifecycle._replace_inference_process.assert_called_once()
