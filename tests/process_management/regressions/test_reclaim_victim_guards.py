"""Victim-selection guards for the VRAM eviction and the stale-RAM slot recycle.

Both reclaim mechanisms pick a victim from the live pool, and both sit on paths where a wrong pick
manufactures the very stall they exist to relieve:

- `InferenceScheduler.unload_models_from_vram` frees room for an incoming load. Its ``next_model``
  guard is what keeps it from evicting the model the queue is about to dispatch, so the head's own
  weights are never dropped out from under it by the reclaim running on its behalf. The head-of-queue
  escalation deliberately overrides that guard, because by then the head has priority for the room.
- `InferenceScheduler._replace_stale_ram_unload_process` cycles an idle slot whose allocator retained
  a freed model's pages, the only way to return that RAM to the OS. Its preconditions (idle,
  model-less, most recently told to unload from RAM, above the retention threshold) are each a guard
  against recycling a slot that is doing or about to do useful work; in particular a slot whose most
  recent control message is a preload must never be reaped mid-stage, since the queue head is counting
  on it.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap, ModelLoadState
from horde_worker_regen.process_management.scheduling import inference_scheduler as inference_scheduler_module
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_NEXT_MODEL = "CyberRealistic Pony"
_OTHER_MODEL = "WAI-NSFW-illustrious-SDXL"

_RETAINED_BYTES = 16 * 1024 * 1024 * 1024


class TestVramUnloadNextModelGuard:
    """The gentle VRAM reclaim spares the model the queue is about to need; the escalation may not."""

    def _scheduler_with_resident(self, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN202
        """A requester slot plus a sibling holding the queue's next model resident."""
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
        monkeypatch.setattr(scheduler, "get_next_n_models", lambda n: [_NEXT_MODEL])
        return scheduler, requester, sibling

    def test_gentle_reclaim_spares_the_next_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the head escalation, the resident copy of the next queued model is not evicted.

        Evicting it would trade one deferred load for two: the incoming model gains room by unloading
        weights the very next dispatch must immediately reload.
        """
        scheduler, requester, sibling = self._scheduler_with_resident(monkeypatch)

        freed = scheduler.unload_models_from_vram(requester, under_pressure=True)

        assert freed is False
        assert sibling.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM

    def test_head_escalation_may_evict_the_next_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CONTROL: the head-of-queue escalation overrides the guard; the head has priority for room."""
        scheduler, requester, sibling = self._scheduler_with_resident(monkeypatch)

        freed = scheduler.unload_models_from_vram(requester, under_pressure=True, for_head_of_queue=True)

        assert freed is True
        assert sibling.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM


class TestVramUnloadSparesEveryLookaheadModel:
    """Every model in the queue lookahead keeps its resident copy under gentle reclaim.

    The guard is set membership, not a single element: with several distinct models queued, sparing
    only one of them (as an order-dependent pick would) evicts weights another queued job must
    immediately reload. Only a resident copy no queued job wants is fair game.
    """

    _UNQUEUED_MODEL = "AlbedoBase XL (SDXL)"

    def _scheduler_with_three_residents(self, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN202
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
        monkeypatch.setattr(scheduler, "get_next_n_models", lambda n: [_NEXT_MODEL, _OTHER_MODEL])
        return scheduler, requester, head_holder, tail_holder, idle_holder

    def test_lookahead_models_spared_unqueued_model_evicted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Gentle reclaim leaves both queued models resident and takes the copy nothing queued wants."""
        scheduler, requester, head_holder, tail_holder, idle_holder = self._scheduler_with_three_residents(
            monkeypatch,
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
        """On a card that cannot hold a queued resident alongside the head's sampling peak, evict it.

        Sparing the copy would force driver demand-paging during the head's sampling (silent on WDDM,
        which keeps reporting free VRAM while saturated), costing far more than the one reload the
        protection saves. The affordability judgment is static: reported total minus reserve minus the
        head job's sampling peak minus the resident's weight footprint.
        """
        scheduler, requester, head_holder, tail_holder, idle_holder = self._scheduler_with_three_residents(
            monkeypatch,
        )
        # A 16 GB card: peak 8258 + footprint 6600 + reserve leaves no room for co-residency.
        for process_info in (requester, head_holder, tail_holder, idle_holder):
            process_info.total_vram_mb = 16376
        await track_popped_job_async(scheduler._job_tracker, make_job_pop_response(_NEXT_MODEL))
        monkeypatch.setattr(
            inference_scheduler_module,
            "predict_job_sampling_vram_mb",
            lambda job, baseline: 8258.0,
        )
        monkeypatch.setattr(
            inference_scheduler_module,
            "predict_job_footprint_mb",
            lambda job, baseline: 6600.0,
        )

        freed = scheduler.unload_models_from_vram(requester, under_pressure=True)

        assert freed is True
        assert head_holder.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
        assert tail_holder.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM


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
