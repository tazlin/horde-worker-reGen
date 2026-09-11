"""Tests for A5.1 GPU-aware dispatch routing: per-card residency lookup + eligible-card selection.

A multi-GPU scheduler must dispatch a job only to a card whose effective config can serve it, and among
several eligible resident cards prefer one ready now then the least-loaded (the "sticky, then least-loaded"
policy). A single-GPU scheduler keeps the original card-agnostic lookup, so routing is a strict no-op there.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeProcessState,
    ModelInfo,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.lru_cache import LRUCache
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.scheduling.admission.preload import (
    duplicate_copies_permitted,
    duplicate_copy_may_serve,
    select_preload_target,
)
from horde_worker_regen.process_management.scheduling.dispatch_affinity import (
    _AFFINITY_MAX_SKIPS,
    record_affinity_skip,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.ledgers.dispatch_holds import DispatchDecline
from horde_worker_regen.process_management.scheduling.slot_duty import SlotDutyBucket
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_card_runtimes,
    make_test_model_metadata,
    make_test_runtime_config,
    make_testable_process_manager,
    mark_job_in_progress_async,
    track_popped_job_async,
)


def _card_config(*, models: list[str], max_pixels: int) -> Mock:
    """A per-card effective config mock with the fields the eligibility check reads."""
    cfg = make_mock_bridge_data()
    cfg.image_models_to_load = models
    cfg.max_pixels = max_pixels
    cfg.nsfw = True  # keep the nsfw axis from excluding either card; resolution is the differentiator
    return cfg


def _make_scheduler(
    *,
    process_map: ProcessMap,
    card_runtimes: dict | None,
    max_threads: int = 2,
) -> InferenceScheduler:
    """Build an InferenceScheduler with a given process map and per-card runtime plan."""
    bridge_data = make_mock_bridge_data()
    bridge_data.max_threads = max_threads
    return InferenceScheduler(
        state=WorkerState(),
        process_map=process_map,
        horde_model_map=HordeModelMap(root={}),
        job_tracker=JobTracker(),
        process_lifecycle=Mock(
            is_model_load_quarantined=Mock(return_value=False),
            get_processes_with_model_for_queued_job=Mock(return_value=[]),
        ),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        model_metadata=make_test_model_metadata(),
        card_runtimes=card_runtimes,
        max_concurrent_inference_processes=2,
        max_inference_processes=4,
        lru=LRUCache(4),
    )


async def _preload_target(scheduler: InferenceScheduler, job: ImageGenerateJobPopResponse) -> HordeProcessInfo | None:
    """The slot the preload gates choose for a freshly tracked job, resolved back to its process record."""
    await track_popped_job_async(scheduler._job_tracker, job)
    process_id = select_preload_target(scheduler.snapshot(), str(job.id_), ())
    return scheduler._process_map[process_id] if process_id is not None else None


async def _duplicate_may_serve(scheduler: InferenceScheduler, job: ImageGenerateJobPopResponse) -> bool:
    await track_popped_job_async(scheduler._job_tracker, job)
    return duplicate_copy_may_serve(scheduler.snapshot(), str(job.id_))


def _two_cards(*, card0_max_pixels: int, card1_max_pixels: int, max_threads: int = 1) -> dict:
    """A 24GB card 0 and an 8GB card 1, each serving stable_diffusion, differing only in max_pixels.

    ``max_threads`` is each card's sampling cap, which placement reads to decide whether a card already
    serving the model could run another copy of it.
    """
    rt0 = make_test_card_runtimes(
        device_indices=(0,),
        config=_card_config(models=["stable_diffusion"], max_pixels=card0_max_pixels),
        total_vram_mb=24576.0,
        max_concurrent_inference=max_threads,
    )
    rt1 = make_test_card_runtimes(
        device_indices=(1,),
        config=_card_config(models=["stable_diffusion"], max_pixels=card1_max_pixels),
        total_vram_mb=8192.0,
        max_concurrent_inference=max_threads,
    )
    return {0: rt0[0], 1: rt1[1]}


class TestEligibilityRouting:
    """A job is routed only to a card whose effective config can serve it."""

    def test_resolution_gating_excludes_the_small_max_power_card(self) -> None:
        """A 512x512 job (262144 px) is ineligible on a card whose max_pixels is below that."""
        process_map = ProcessMap(
            {
                0: make_mock_process_info(0, model_name="stable_diffusion", device_index=0),
                1: make_mock_process_info(1, model_name="stable_diffusion", device_index=1),
            },
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_two_cards(card0_max_pixels=5_000_000, card1_max_pixels=1000),
        )
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        assert scheduler._eligible_card_indices(job) == {0}
        chosen = scheduler.resident_process_for_job(job)
        assert chosen is not None
        assert chosen.device_index == 0

    def test_model_resident_only_on_ineligible_card_yields_no_process(self) -> None:
        """If the only resident copy is on a card that cannot serve the job, dispatch finds nothing."""
        process_map = ProcessMap(
            {1: make_mock_process_info(1, model_name="stable_diffusion", device_index=1)},
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_two_cards(card0_max_pixels=5_000_000, card1_max_pixels=1000),
        )
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        assert scheduler._eligible_card_indices(job) == {0}
        assert scheduler.resident_process_for_job(job) is None


class TestStickyLeastLoaded:
    """Among several eligible resident cards, prefer ready-now then the least-loaded card."""

    def test_prefers_the_less_loaded_eligible_card(self) -> None:
        """With both cards eligible and ready, the card running fewer inference jobs wins the dispatch."""
        process_map = ProcessMap(
            {
                0: make_mock_process_info(0, model_name="stable_diffusion", device_index=0),
                # A second, busy process on card 0 makes it the more-loaded card.
                2: make_mock_process_info(
                    2,
                    model_name="other_model",
                    device_index=0,
                    state=HordeProcessState.INFERENCE_STARTING,
                ),
                1: make_mock_process_info(1, model_name="stable_diffusion", device_index=1),
            },
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_two_cards(card0_max_pixels=5_000_000, card1_max_pixels=5_000_000),
        )
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        assert scheduler._eligible_card_indices(job) == {0, 1}
        chosen = scheduler.resident_process_for_job(job)
        assert chosen is not None
        assert chosen.device_index == 1

    def test_prefers_a_ready_process_over_a_busy_one(self) -> None:
        """A resident process mid-inference is passed over for one that can take work now."""
        process_map = ProcessMap(
            {
                0: make_mock_process_info(
                    0,
                    model_name="stable_diffusion",
                    device_index=0,
                    state=HordeProcessState.INFERENCE_STARTING,
                ),
                1: make_mock_process_info(1, model_name="stable_diffusion", device_index=1),
            },
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_two_cards(card0_max_pixels=5_000_000, card1_max_pixels=5_000_000),
        )
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        chosen = scheduler.resident_process_for_job(job)
        assert chosen is not None
        assert chosen.device_index == 1


class TestSingleGpuNoop:
    """A single-GPU host keeps the original first-resident lookup, untouched by routing."""

    def test_single_card_uses_first_resident_process(self) -> None:
        """With one card, routing is inactive and dispatch returns the first resident process."""
        process_map = ProcessMap(
            {0: make_mock_process_info(0, model_name="stable_diffusion", device_index=0)},
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=make_test_card_runtimes(device_indices=(0,)),
        )
        assert scheduler._multi_gpu_routing_active is False
        job = make_job_pop_response(model="stable_diffusion")
        chosen = scheduler.resident_process_for_job(job)
        assert chosen is not None
        assert chosen.device_index == 0

    def test_no_card_runtimes_is_card_agnostic(self) -> None:
        """The card-agnostic default (no runtimes injected) leaves routing inactive."""
        process_map = ProcessMap(
            {0: make_mock_process_info(0, model_name="stable_diffusion", device_index=0)},
        )
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=None)
        assert scheduler._multi_gpu_routing_active is False
        job = make_job_pop_response(model="stable_diffusion")
        assert scheduler.resident_process_for_job(job) is not None

    async def test_single_card_faults_exactly_ineligible_head_and_reaches_follower(self) -> None:
        """Exact incompatibility is bounded on one card even though placement retains its legacy fast path."""
        process_map = ProcessMap(
            {0: make_mock_process_info(0, model_name="stable_diffusion", device_index=0)},
        )
        card_runtimes = make_test_card_runtimes(
            device_indices=(0,),
            config=_card_config(models=["stable_diffusion"], max_pixels=5_000_000),
        )
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=card_runtimes)
        unsupported_head = make_job_pop_response(model="stable_diffusion", workflow="future_workflow")
        runnable_follower = make_job_pop_response(model="stable_diffusion")
        await track_popped_job_async(scheduler._job_tracker, unsupported_head)
        await track_popped_job_async(scheduler._job_tracker, runnable_follower)

        assert await scheduler.get_next_job_and_process(information_only=False) is None
        assert unsupported_head.id_ is not None
        tracked_head = scheduler._job_tracker.get_tracked_job(unsupported_head.id_)
        assert tracked_head is not None
        assert tracked_head.stage is JobStage.PENDING_SUBMIT
        assert tracked_head.fault_reason is not None
        assert "workflows" in tracked_head.fault_reason

        selected = await scheduler.get_next_job_and_process(information_only=False)
        assert selected is not None
        assert selected.next_job == runnable_follower


def _empty_slot(process_id: int, *, device_index: int) -> Mock:
    """An idle inference process holding no model (a free slot a preload can target)."""
    return make_mock_process_info(
        process_id,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        device_index=device_index,
    )


def _empty_slot_with_vram(
    process_id: int,
    *,
    device_index: int,
    total_vram_mb: int,
    vram_usage_mb: int,
) -> Mock:
    """An idle inference process with a recent device-free VRAM report."""
    process = _empty_slot(process_id, device_index=device_index)
    process.total_vram_mb = total_vram_mb
    process.vram_usage_mb = vram_usage_mb
    return process


class TestPreloadCardPlacement:
    """A5.4: a fresh preload picks which eligible card to load onto (sticky, then least-loaded)."""

    async def test_places_on_the_least_loaded_eligible_card(self) -> None:
        """With the model resident nowhere, the free slot on the card running fewer jobs wins."""
        process_map = ProcessMap(
            {
                # Card 0 is busier (a running job) than card 1; both have a free slot.
                0: make_mock_process_info(
                    0,
                    model_name="other_model",
                    device_index=0,
                    state=HordeProcessState.INFERENCE_STARTING,
                ),
                1: _empty_slot(1, device_index=0),
                2: _empty_slot(2, device_index=1),
            },
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_two_cards(card0_max_pixels=5_000_000, card1_max_pixels=5_000_000),
        )
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        chosen = await _preload_target(scheduler, job)
        assert chosen is not None
        assert chosen.device_index == 1

    async def test_sticky_prefers_a_serving_card_that_can_still_run_another_copy(self) -> None:
        """A card already serving the model wins the preload while it has spare sampling capacity.

        Its second lane can start the copy the moment it lands, so the sticky preference costs nothing and
        keeps the model's copies together.
        """
        process_map = ProcessMap(
            {
                # Card 0 already holds the model on a busy slot (sticky) and is the more-loaded card.
                0: make_mock_process_info(
                    0,
                    model_name="stable_diffusion",
                    device_index=0,
                    state=HordeProcessState.INFERENCE_STARTING,
                ),
                1: _empty_slot(1, device_index=0),
                2: _empty_slot(2, device_index=1),
            },
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_two_cards(card0_max_pixels=5_000_000, card1_max_pixels=5_000_000, max_threads=2),
        )
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        chosen = await _preload_target(scheduler, job)
        assert chosen is not None
        assert chosen.device_index == 0

    async def test_a_serving_card_at_its_cap_ranks_last_for_a_second_copy(self) -> None:
        """With one sampling slot per card, the copy goes to a card that can run it, not beside the busy one.

        The eight-card serialization: sticky placement sent every duplicate to the card already running the
        model, whose sibling lane could not sample under a cap of one, so the hot model's queue drained
        through a single lane while other cards idled.
        """
        process_map = ProcessMap(
            {
                0: make_mock_process_info(
                    0,
                    model_name="stable_diffusion",
                    device_index=0,
                    state=HordeProcessState.INFERENCE_STARTING,
                ),
                1: _empty_slot(1, device_index=0),
                2: _empty_slot(2, device_index=1),
            },
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_two_cards(card0_max_pixels=5_000_000, card1_max_pixels=5_000_000, max_threads=1),
        )
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)

        chosen = await _preload_target(scheduler, job)

        assert chosen is not None
        assert chosen.device_index == 1
        assert chosen.process_id == 2

    async def test_equal_load_prefers_card_with_more_measured_free_vram(self) -> None:
        """A safety/post-process context on card 0 should not attract a tied fresh preload."""
        process_map = ProcessMap(
            {
                1: _empty_slot_with_vram(1, device_index=0, total_vram_mb=7786, vram_usage_mb=2564),
                2: _empty_slot_with_vram(2, device_index=1, total_vram_mb=8107, vram_usage_mb=173),
            },
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_two_cards(card0_max_pixels=5_000_000, card1_max_pixels=5_000_000),
        )
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)

        chosen = await _preload_target(scheduler, job)

        assert chosen is not None
        assert chosen.device_index == 1

    async def test_never_places_on_an_ineligible_card(self) -> None:
        """A card whose resolution limit excludes the job is never chosen, even when it is the only idle one."""
        process_map = ProcessMap(
            {
                # Card 0 (eligible) is busy with a job but has a free slot; card 1 is idle but ineligible.
                0: make_mock_process_info(
                    0,
                    model_name="other_model",
                    device_index=0,
                    state=HordeProcessState.INFERENCE_STARTING,
                ),
                1: _empty_slot(1, device_index=0),
                2: _empty_slot(2, device_index=1),
            },
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_two_cards(card0_max_pixels=5_000_000, card1_max_pixels=1000),
        )
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        assert scheduler._eligible_card_indices(job) == {0}
        chosen = await _preload_target(scheduler, job)
        assert chosen is not None
        assert chosen.device_index == 0

    async def test_single_gpu_uses_first_available_slot(self) -> None:
        """With one card, placement is inactive and the first available slot is returned, as before."""
        process_map = ProcessMap({0: _empty_slot(0, device_index=0)})
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=make_test_card_runtimes(device_indices=(0,)),
        )
        assert scheduler._multi_gpu_routing_active is False
        job = make_job_pop_response(model="stable_diffusion")
        chosen = await _preload_target(scheduler, job)
        assert chosen is not None
        assert chosen.device_index == 0


def _preloading_slot(process_id: int, *, device_index: int, model_name: str) -> Mock:
    """A process mid-preload of ``model_name`` on ``device_index`` (busy, not available for a new load)."""
    return make_mock_process_info(
        process_id,
        model_name=model_name,
        state=HordeProcessState.PRELOADING_MODEL,
        device_index=device_index,
    )


class TestPerCardPreloadSerialization:
    """The preload-serialization gate is per-card: one card mid-load must not starve an idle other card.

    Reproduction of the live two-1070 starvation: the gate exists so two checkpoints do not load onto the
    *same* device at once, but it counted preloading processes worker-wide. The busy card was almost always
    mid-preload, so every attempt to stage a model onto the idle second card was deferred and that card never
    received its first model; it sat ``WAITING_FOR_JOB`` forever while the other card did all the work.
    """

    async def test_idle_card_preloads_while_other_card_is_mid_preload(self) -> None:
        """With card 0 mid-preload and card 1 idle, the pending job stages onto card 1 (not deferred)."""
        idle_card1 = _empty_slot(2, device_index=1)
        process_map = ProcessMap(
            {
                # Card 0's only slot is busy loading another model; card 1 has a free slot.
                0: _preloading_slot(0, device_index=0, model_name="other_model"),
                2: idle_card1,
            },
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_two_cards(card0_max_pixels=5_000_000, card1_max_pixels=5_000_000),
        )
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        await track_popped_job_async(scheduler._job_tracker, job)

        admitted = scheduler.preload_models()

        assert admitted is True, "the idle card must get the preload, not be blocked by the busy card"
        assert idle_card1.last_control_flag == HordeControlFlag.PRELOAD_MODEL
        assert idle_card1.loaded_horde_model_name == "stable_diffusion"

    async def test_same_card_preload_still_serialized(self) -> None:
        """Guard: two slots on the *same* card keep one-at-a-time serialization (the gate's real purpose)."""
        idle_same_card = _empty_slot(1, device_index=0)
        process_map = ProcessMap(
            {
                0: _preloading_slot(0, device_index=0, model_name="other_model"),
                1: idle_same_card,
            },
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            # One card only: routing inactive, so the count is worker-wide exactly as before.
            card_runtimes=make_test_card_runtimes(
                device_indices=(0,),
                config=_card_config(models=["stable_diffusion"], max_pixels=5_000_000),
            ),
        )
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        await track_popped_job_async(scheduler._job_tracker, job)

        admitted = scheduler.preload_models()

        assert admitted is False, "a same-card preload must wait for the in-flight one to finish"
        assert idle_same_card.last_control_flag != HordeControlFlag.PRELOAD_MODEL


class TestCardConfigTracksConfigReload:
    """A card's effective config follows a live config reload, so routing and advertising stay in step.

    The popper advertises from live bridge data while eligibility is decided against each card's effective
    config. If the card configs kept their startup values, a reload that raised a limit or enabled a feature
    would have the worker accept work it then faults as ineligible.
    """

    @staticmethod
    def _manager(**bridge_overrides: object) -> HordeWorkerProcessManager:
        """A manager whose reload path can be driven without a live download process."""
        manager = make_testable_process_manager(**bridge_overrides)
        manager._process_lifecycle = Mock()  # pyrefly: ignore - a stub stands in for the lifecycle manager
        manager._download_coordinator._process_lifecycle = manager._process_lifecycle
        manager._download_coordinator.initial_download_requested = True
        return manager

    def test_raised_resolution_limit_makes_a_refused_job_eligible(self) -> None:
        """A job above the configured max_pixels becomes servable once the reload raises the limit."""
        manager = self._manager(max_pixels=1000, nsfw=True)
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)  # 262144 px
        assert manager._inference_scheduler._eligible_card_indices(job) == set()

        manager._apply_reloaded_bridge_data(
            make_mock_bridge_data(max_pixels=5_000_000, nsfw=True, dry_run_skip_inference=True),
        )

        assert manager._inference_scheduler._eligible_card_indices(job) == {0}

    def test_lowered_resolution_limit_makes_a_servable_job_ineligible(self) -> None:
        """The refresh applies in both directions: a tightened limit stops offering the card for the job."""
        manager = self._manager(max_pixels=5_000_000, nsfw=True)
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        assert manager._inference_scheduler._eligible_card_indices(job) == {0}

        manager._apply_reloaded_bridge_data(
            make_mock_bridge_data(max_pixels=1000, nsfw=True, dry_run_skip_inference=True),
        )

        assert manager._inference_scheduler._eligible_card_indices(job) == set()

    def test_sfw_card_admits_an_ordinary_job(self) -> None:
        """NSFW is an offer-level policy: a card configured SFW still routes the jobs the horde returns."""
        manager = self._manager(max_pixels=5_000_000, nsfw=False)
        job = make_job_pop_response(model="stable_diffusion")
        assert manager._inference_scheduler._eligible_card_indices(job) == {0}

    def test_reload_leaves_the_card_plan_otherwise_untouched(self) -> None:
        """Only the config moves: the session's concurrency primitives and sizes are not re-derived."""
        manager = self._manager(max_pixels=1000, nsfw=True)
        before = manager._card_runtimes[0]

        manager._apply_reloaded_bridge_data(
            make_mock_bridge_data(max_pixels=5_000_000, nsfw=True, dry_run_skip_inference=True),
        )

        after = manager._card_runtimes[0]
        assert after.config is not before.config
        assert after.inference_semaphore is before.inference_semaphore
        assert after.vae_decode_semaphore is before.vae_decode_semaphore
        assert after.target_process_count == before.target_process_count
        assert after.max_concurrent_inference == before.max_concurrent_inference
        assert after.total_vram_mb == before.total_vram_mb
        assert after.mask_kind == before.mask_kind


class TestDuplicateCopyEscape:
    """A model whose every resident copy is busy may earn a second copy on another card.

    The single-copy rule (a resident or loading model is never preloaded again) is load and VRAM economy;
    across cards it starves: a pending queue dominated by models resident on busy processes leaves idle
    cards doing nothing. The escape only *considers* a duplicate; the preload pipeline's guards still
    decide whether it lands.
    """

    def _scheduler(self, process_map: ProcessMap) -> InferenceScheduler:
        return _make_scheduler(
            process_map=process_map,
            card_runtimes=_two_cards(card0_max_pixels=5_000_000, card1_max_pixels=5_000_000),
        )

    async def _seat_running_copy(self, scheduler: InferenceScheduler, model: str) -> None:
        """Track the job the busy copy is sampling, so the queue accounts for the work that copy holds.

        The duplicate bound is read off the queue: a lane sampling work the tracker does not know about would
        understate its model's demand and refuse the second copy the queued job is waiting for.
        """
        running = make_job_pop_response(model=model, width=512, height=512)
        await track_popped_job_async(scheduler._job_tracker, running)
        await mark_job_in_progress_async(scheduler._job_tracker, running)

    async def test_all_copies_busy_allows_a_duplicate(self) -> None:
        """Every eligible copy busy sampling: the queued job may seek a second copy on the idle card."""
        process_map = ProcessMap(
            {
                0: make_mock_process_info(
                    0, model_name="stable_diffusion", device_index=0, state=HordeProcessState.INFERENCE_STARTING
                ),
                1: make_mock_process_info(1, model_name=None, device_index=1),
            },
        )
        scheduler = self._scheduler(process_map)
        await self._seat_running_copy(scheduler, "stable_diffusion")
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        assert await _duplicate_may_serve(scheduler, job) is True

    async def test_a_free_copy_forbids_a_duplicate(self) -> None:
        """An accepting resident copy serves the job without any load; the single-copy rule holds."""
        process_map = ProcessMap(
            {
                0: make_mock_process_info(
                    0, model_name="stable_diffusion", device_index=0, state=HordeProcessState.WAITING_FOR_JOB
                ),
                1: make_mock_process_info(1, model_name=None, device_index=1),
            },
        )
        scheduler = self._scheduler(process_map)
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        assert await _duplicate_may_serve(scheduler, job) is False

    async def test_a_loading_copy_forbids_a_duplicate(self) -> None:
        """A load in flight is about to provide a serving copy; doubling it is the waste the rule stops."""
        process_map = ProcessMap(
            {
                0: make_mock_process_info(
                    0, model_name="stable_diffusion", device_index=0, state=HordeProcessState.INFERENCE_STARTING
                ),
                1: make_mock_process_info(1, model_name=None, device_index=1),
            },
        )
        scheduler = self._scheduler(process_map)
        await self._seat_running_copy(scheduler, "stable_diffusion")
        scheduler._horde_model_map.root["stable_diffusion"] = ModelInfo(
            horde_model_name="stable_diffusion",
            horde_model_load_state=ModelLoadState.LOADING,
            process_id=0,
        )
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        assert await _duplicate_may_serve(scheduler, job) is False

    async def test_single_gpu_never_duplicates(self) -> None:
        """On one card a duplicate is pure waste; the escape is multi-GPU only."""
        process_map = ProcessMap(
            {
                0: make_mock_process_info(
                    0, model_name="stable_diffusion", device_index=0, state=HordeProcessState.INFERENCE_STARTING
                ),
            },
        )
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=None)
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        assert await _duplicate_may_serve(scheduler, job) is False

    async def test_no_resident_copy_defers_to_the_ordinary_preload(self) -> None:
        """With no copy anywhere the ordinary preload path owns the job; the escape stays out of it."""
        process_map = ProcessMap(
            {
                0: make_mock_process_info(0, model_name=None, device_index=0),
                1: make_mock_process_info(1, model_name=None, device_index=1),
            },
        )
        scheduler = self._scheduler(process_map)
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        assert await _duplicate_may_serve(scheduler, job) is False


class TestDuplicateCopiesAreBounded:
    """A burst of one model earns copies up to its own demand, and never more than one per card.

    Without a bound the escape is the mirror of the defect it fixes: a queue dominated by one model would take
    a lane on every card for it and leave the pool holding one model, so the next model's job pays a full disk
    reload wherever it lands.
    """

    def _four_card_scheduler(self, process_map: ProcessMap) -> InferenceScheduler:
        return _make_scheduler(process_map=process_map, card_runtimes=_uniform_cards(4))

    async def test_the_bound_is_the_smaller_of_the_queues_demand_and_the_card_count(self) -> None:
        """Two jobs on a four-card host earn two copies; six jobs earn four, one per card."""
        process_map = ProcessMap({index: _lane(index, device_index=index, model=None) for index in range(4)})
        scheduler = self._four_card_scheduler(process_map)
        await _queue(scheduler, "stable_diffusion", "stable_diffusion")

        snapshot = scheduler.snapshot()
        assert duplicate_copies_permitted(snapshot, "stable_diffusion") == 2, "the queue's demand binds first"

        await _queue(scheduler, *(["stable_diffusion"] * 4))
        assert duplicate_copies_permitted(scheduler.snapshot(), "stable_diffusion") == 4, "the card count binds"

    async def test_a_copy_per_card_is_the_ceiling(self) -> None:
        """With every card already holding a busy copy, a deeper queue earns no further copy.

        A fifth copy would have to share a card with one of the four, where it cannot sample under a cap of
        one; the job waits for a running copy instead of paying for weights that would idle.
        """
        process_map = ProcessMap(
            {
                index: make_mock_process_info(
                    index,
                    model_name="stable_diffusion",
                    device_index=index,
                    state=HordeProcessState.INFERENCE_STARTING,
                )
                for index in range(4)
            },
        )
        scheduler = self._four_card_scheduler(process_map)
        for _running in range(4):
            running = make_job_pop_response(model="stable_diffusion", width=512, height=512)
            await track_popped_job_async(scheduler._job_tracker, running)
            await mark_job_in_progress_async(scheduler._job_tracker, running)
        waiting = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        await track_popped_job_async(scheduler._job_tracker, waiting)

        snapshot = scheduler.snapshot()
        assert duplicate_copies_permitted(snapshot, "stable_diffusion") == 4
        assert duplicate_copy_may_serve(snapshot, str(waiting.id_)) is False

    async def test_a_free_card_still_earns_the_next_copy(self) -> None:
        """The control: below the bound the escape still fires, so the ceiling is what refused above."""
        process_map = ProcessMap(
            {
                0: make_mock_process_info(
                    0,
                    model_name="stable_diffusion",
                    device_index=0,
                    state=HordeProcessState.INFERENCE_STARTING,
                ),
                1: _lane(1, device_index=1, model=None),
                2: _lane(2, device_index=2, model=None),
                3: _lane(3, device_index=3, model=None),
            },
        )
        scheduler = self._four_card_scheduler(process_map)
        running = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        await track_popped_job_async(scheduler._job_tracker, running)
        await mark_job_in_progress_async(scheduler._job_tracker, running)
        waiting = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        await track_popped_job_async(scheduler._job_tracker, waiting)

        snapshot = scheduler.snapshot()
        assert duplicate_copy_may_serve(snapshot, str(waiting.id_)) is True
        assert select_preload_target(snapshot, str(waiting.id_), ()) in {1, 2, 3}, "onto a card that can run it"


# ----------------------------------------------------------------------------------------------------------
# Cross-card dispatch: a head that cannot be seated must not withhold seating on the worker's other cards
# ----------------------------------------------------------------------------------------------------------

_ALL_MODELS = ["stable_diffusion", "model_a", "model_b", "model_c", "model_d", "model_e"]


def _uniform_cards(count: int, *, max_concurrent_inference: int = 1) -> dict:
    """``count`` identical cards, each serving every model in ``_ALL_MODELS`` under its own concurrency cap."""
    cards: dict = {}
    for index in range(count):
        config = _card_config(models=_ALL_MODELS, max_pixels=5_000_000)
        # The card's effective config carries the same thread count its semaphores were sized for, as the
        # real per-card plan derives both from one figure.
        config.max_threads = max_concurrent_inference
        cards.update(
            make_test_card_runtimes(
                device_indices=(index,),
                config=config,
                max_concurrent_inference=max_concurrent_inference,
                target_process_count=2,
            ),
        )
    return cards


def _lane(process_id: int, *, device_index: int, model: str | None) -> HordeProcessInfo:
    """An idle inference lane on ``device_index`` holding ``model``'s weights."""
    return make_mock_process_info(process_id, model_name=model, device_index=device_index)


async def _queue(scheduler: InferenceScheduler, *models: str) -> list[ImageGenerateJobPopResponse]:
    """Pop one job per model, in order, so the first is the dispatch head."""
    jobs = []
    for model in models:
        job = make_job_pop_response(model=model, width=512, height=512)
        await track_popped_job_async(scheduler._job_tracker, job)
        jobs.append(job)
    return jobs


async def _seat(
    scheduler: InferenceScheduler,
    job: ImageGenerateJobPopResponse,
    process: HordeProcessInfo,
) -> None:
    """Put ``job`` in progress on ``process``, as a committed dispatch leaves the world."""
    process.record_inference_ownership(job, attempt_ordinal=0)
    process.last_process_state = HordeProcessState.INFERENCE_STARTING
    await scheduler._job_tracker.mark_inference_started(job, device_index=process.device_index)


async def _drain_one_cycle(scheduler: InferenceScheduler) -> list[tuple[ImageGenerateJobPopResponse, int]]:
    """Every (job, card) selection one scheduling cycle's dispatch loop would commit, in order.

    Stands in for ``while await start_inference()``: each selection is seated exactly as a committed dispatch
    leaves the world, so the next call sees the card it filled. Bounded by the queue length, so a selector
    that never stops handing out the same job fails as a hang rather than passing.
    """
    scheduler.begin_scheduling_cycle()
    seated: list[tuple[ImageGenerateJobPopResponse, int]] = []
    for _attempt in range(len(scheduler._job_tracker.jobs_pending_inference) + 1):
        selection = await scheduler.get_next_job_and_process()
        if selection is None:
            return seated
        await _seat(scheduler, selection.next_job, selection.process_with_model)
        seated.append((selection.next_job, selection.process_with_model.device_index))
        scheduler._pending_line_skip = None
    raise AssertionError("the dispatch loop kept selecting past the whole queue")


class TestWorkerWideConcurrencyCeiling:
    """The worker's concurrent-sampling ceiling is the sum of its cards', not one card's figure."""

    def test_single_card_ceiling_is_the_live_thread_cap(self) -> None:
        """One card keeps the pre-multi-GPU answer: the live effective max_threads."""
        scheduler = _make_scheduler(
            process_map=ProcessMap({0: _lane(0, device_index=0, model="stable_diffusion")}),
            card_runtimes=_uniform_cards(1, max_concurrent_inference=2),
            max_threads=2,
        )
        assert scheduler.worker_wide_concurrency_ceiling() == 2

    def test_ceiling_sums_every_driven_card(self) -> None:
        """Eight cards at one thread each admit eight concurrent jobs, not one."""
        scheduler = _make_scheduler(
            process_map=ProcessMap({0: _lane(0, device_index=0, model="stable_diffusion")}),
            card_runtimes=_uniform_cards(8, max_concurrent_inference=1),
            max_threads=1,
        )
        assert scheduler.worker_wide_concurrency_ceiling() == 8
        assert scheduler._max_jobs_in_progress_allowed() == 8
        assert scheduler._max_jobs_in_progress_allowed(card=scheduler._card_runtimes[3]) == 1

    def test_a_lowered_live_cap_lowers_every_card(self) -> None:
        """A runtime thread reduction reaches the sum: four cards at two threads fall to four."""
        scheduler = _make_scheduler(
            process_map=ProcessMap({0: _lane(0, device_index=0, model="stable_diffusion")}),
            card_runtimes=_uniform_cards(4, max_concurrent_inference=2),
            max_threads=2,
        )
        assert scheduler.worker_wide_concurrency_ceiling() == 8

        scheduler._runtime_config.set_effective_max_threads(1)

        assert scheduler.worker_wide_concurrency_ceiling() == 4

    def test_the_lane_pool_is_summed_across_cards(self) -> None:
        """The scheduler's lane count is every card's target process count, not one card's.

        It is what sizes ``affinity_active``: a model's home is a lane anywhere on the host, so a worker whose
        lane pool read as one card's would call itself the overflow regime while it had homes to give out.
        """
        manager = make_testable_process_manager()

        assert manager.max_inference_processes == sum(
            card.target_process_count for card in manager._card_runtimes.values()
        )
        assert manager._inference_scheduler.snapshot().max_inference_processes == manager.max_inference_processes


class TestCrossCardDispatch:
    """A head parked on one card leaves every other card free to take work from further down the queue."""

    async def test_every_free_card_is_filled_in_one_cycle(self) -> None:
        """Best case: four idle cards each holding a queued model take one job apiece in a single cycle."""
        process_map = ProcessMap(
            {index: _lane(index, device_index=index, model=f"model_{letter}") for index, letter in enumerate("abcd")},
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_uniform_cards(4),
            max_threads=1,
        )
        await _queue(scheduler, "model_a", "model_b", "model_c", "model_d")

        seated = await _drain_one_cycle(scheduler)

        assert [card for _job, card in seated] == [0, 1, 2, 3]

    async def test_a_busy_head_lane_does_not_hold_the_other_cards(self) -> None:
        """Worst case: the head's only copy is on a busy lane; every other card still takes its own job."""
        head_lane = _lane(0, device_index=0, model="model_a")
        process_map = ProcessMap(
            {
                0: head_lane,
                1: _lane(1, device_index=1, model="model_b"),
                2: _lane(2, device_index=2, model="model_c"),
                3: _lane(3, device_index=3, model="model_d"),
            },
        )
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=_uniform_cards(4), max_threads=1)
        blocking_job, head, *_followers = await _queue(scheduler, "model_a", "model_a", "model_b", "model_c")
        await _seat(scheduler, blocking_job, head_lane)

        seated = await _drain_one_cycle(scheduler)

        assert [card for _job, card in seated] == [1, 2]
        assert head in scheduler._job_tracker.jobs_pending_inference
        assert scheduler._affinity_skip_state.skip_count == 0

    async def test_a_cold_head_does_not_hold_the_other_cards(self) -> None:
        """With the head's model resident nowhere, more than one card's worth of resident work still runs.

        A cold head is counted against the whole worker's ceiling rather than one card's, so this is what the
        per-card sum buys: the first bypass fills one card, and the second is only reachable because the
        ceiling is the pool's rather than a single card's thread count.
        """
        process_map = ProcessMap(
            {
                0: _lane(0, device_index=0, model=None),
                1: _lane(1, device_index=1, model="model_b"),
                2: _lane(2, device_index=2, model="model_c"),
            },
        )
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=_uniform_cards(3), max_threads=1)
        cold_head, *_followers = await _queue(scheduler, "model_a", "model_b", "model_c")

        seated = await _drain_one_cycle(scheduler)

        assert [card for _job, card in seated] == [1, 2]
        assert cold_head in scheduler._job_tracker.jobs_pending_inference
        assert scheduler._affinity_skip_state.skip_count == 0

    async def test_the_head_card_is_left_to_the_head(self) -> None:
        """A job resident on an idle lane of the head's own card is not seated ahead of the head.

        The head's card carries a sibling that has taken its one sampling slot, so the head waits for that
        slot rather than for its own lane. Both the card's cap and the head's claim on it refuse the
        candidate, and the second card's job is what runs instead.
        """
        head_lane = _lane(0, device_index=0, model="model_a")
        sibling_lane = _lane(1, device_index=0, model="model_b")
        process_map = ProcessMap(
            {
                0: head_lane,
                1: sibling_lane,
                2: _lane(2, device_index=1, model="model_c"),
            },
        )
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=_uniform_cards(2), max_threads=1)
        sibling_job, head, same_card_candidate, other_card_candidate = await _queue(
            scheduler,
            "model_b",
            "model_a",
            "model_b",
            "model_c",
        )
        await _seat(scheduler, sibling_job, sibling_lane)
        # The sibling's lane is busy, but the head's own lane is idle and holds the head's model.
        assert head_lane.can_accept_job()

        selection = await scheduler.get_next_job_and_process()

        assert selection is not None
        assert selection.next_job is other_card_candidate
        assert selection.process_with_model.device_index == 1
        assert same_card_candidate in scheduler._job_tracker.jobs_pending_inference
        assert head in scheduler._job_tracker.jobs_pending_inference

    async def test_every_card_at_cap_dispatches_nothing(self) -> None:
        """With every card's sampling slot taken, selection returns nothing and the dispatch loop ends."""
        lanes = {index: _lane(index, device_index=index, model=f"model_{letter}") for index, letter in enumerate("ab")}
        process_map = ProcessMap(lanes)
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=_uniform_cards(2), max_threads=1)
        running_a, running_b, *_pending = await _queue(scheduler, "model_a", "model_b", "model_a", "model_b")
        await _seat(scheduler, running_a, lanes[0])
        await _seat(scheduler, running_b, lanes[1])

        assert await scheduler.get_next_job_and_process() is None
        assert await scheduler.start_inference() is False

    async def test_a_candidate_resident_only_on_an_ineligible_card_is_skipped(self) -> None:
        """Cross-card seating obeys card eligibility: an unservable candidate is passed for the next one."""
        head_lane = _lane(0, device_index=0, model="model_a")
        process_map = ProcessMap(
            {
                0: head_lane,
                1: _lane(1, device_index=1, model="model_b"),
                2: _lane(2, device_index=2, model="model_c"),
            },
        )
        card_runtimes = _uniform_cards(3)
        # Card 1 cannot serve any job of this size, so its resident copy of model_b is unreachable.
        card_runtimes[1] = dataclasses.replace(
            card_runtimes[1],
            config=_card_config(models=_ALL_MODELS, max_pixels=1000),
        )
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=card_runtimes, max_threads=1)
        blocking_job, _head, ineligible_candidate, servable_candidate = await _queue(
            scheduler,
            "model_a",
            "model_a",
            "model_b",
            "model_c",
        )
        await _seat(scheduler, blocking_job, head_lane)

        selection = await scheduler.get_next_job_and_process()

        assert selection is not None
        assert selection.next_job is servable_candidate
        assert selection.process_with_model.device_index == 2
        assert ineligible_candidate in scheduler._job_tracker.jobs_pending_inference

    async def test_an_exclusive_job_suppresses_only_its_own_card(self) -> None:
        """An exclusively-admitted job holds its card to itself; the worker's other cards keep dispatching.

        Card 1 has room under its cap and an idle lane already holding the candidate's model, so only the
        exclusive hold can refuse it; card 2 shows the suppression did not spread to the rest of the pool.
        """
        head_lane = _lane(0, device_index=0, model="model_a")
        exclusive_lane = _lane(1, device_index=1, model="model_b")
        process_map = ProcessMap(
            {
                0: head_lane,
                1: exclusive_lane,
                2: _lane(2, device_index=2, model="model_c"),
                3: _lane(3, device_index=1, model="model_b"),
            },
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_uniform_cards(3, max_concurrent_inference=2),
            max_threads=2,
        )
        blocking_job, _head, exclusive_job, suppressed_candidate, free_card_candidate = await _queue(
            scheduler,
            "model_a",
            "model_a",
            "model_b",
            "model_b",
            "model_c",
        )
        await _seat(scheduler, blocking_job, head_lane)
        await _seat(scheduler, exclusive_job, exclusive_lane)
        scheduler._job_tracker.mark_admitted_exclusive(exclusive_job, device_index=1)

        selection = await scheduler.get_next_job_and_process()

        assert selection is not None
        assert selection.next_job is free_card_candidate
        assert selection.process_with_model.device_index == 2
        assert suppressed_candidate in scheduler._job_tracker.jobs_pending_inference

    async def test_the_look_ahead_and_the_dispatch_call_agree(self) -> None:
        """The cache contract: both calls in a cycle name the same pair, and the peek changes nothing."""
        head_lane = _lane(0, device_index=0, model="model_a")
        process_map = ProcessMap(
            {
                0: head_lane,
                1: _lane(1, device_index=1, model="model_b"),
            },
        )
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=_uniform_cards(2), max_threads=1)
        blocking_job, head, candidate = await _queue(scheduler, "model_a", "model_a", "model_b")
        await _seat(scheduler, blocking_job, head_lane)

        peeked = await scheduler.get_next_job_and_process(information_only=True)
        launched = await scheduler.get_next_job_and_process()

        assert peeked is not None and launched is not None
        assert peeked.next_job is candidate
        assert launched.next_job is candidate
        assert launched.process_with_model.device_index == 1
        assert launched.line_skip is not None
        assert launched.line_skip.reason == "cross_card"
        assert launched.line_skip.displaced_job is head
        assert candidate in scheduler._job_tracker.jobs_pending_inference

    async def test_a_single_card_head_at_its_cap_still_dispatches_nothing(self) -> None:
        """Single-GPU parity: one card at its cap withholds dispatch exactly as it always has."""
        head_lane = _lane(0, device_index=0, model="model_a")
        process_map = ProcessMap(
            {
                0: head_lane,
                1: _lane(1, device_index=0, model="model_b"),
            },
        )
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=_uniform_cards(1), max_threads=1)
        blocking_job, _head, _candidate = await _queue(scheduler, "model_a", "model_a", "model_b")
        await _seat(scheduler, blocking_job, head_lane)

        assert scheduler._multi_gpu_routing_active is False
        assert await scheduler.get_next_job_and_process() is None


class TestColdHeadWithALoadInFlight:
    """A cold head whose load is already staging keeps the capacity on that card and no more.

    Reached once the head has spent its resident-bypass window: past that the bypass loop yields (so the
    head's own load can run) and cross-card seating is what still fills the rest of the pool. A cycle
    preloads *and* dispatches, so the head's in-flight load and a cross-card seat now happen together.
    """

    def _spend_the_bypass_window(self, scheduler: InferenceScheduler, head: ImageGenerateJobPopResponse) -> None:
        """Age the head past its resident-bypass skip ceiling, as a queue of resident work does."""
        head_id = str(head.id_)
        for _skip in range(_AFFINITY_MAX_SKIPS):
            scheduler._affinity_skip_state = record_affinity_skip(
                scheduler._affinity_skip_state,
                head_id,
                scheduler._clock(),
            )

    async def test_cross_card_seating_spares_the_loading_cards_other_lane(self) -> None:
        """Cards with no copy of the head's model take queued work; the loading card's spare lane does not.

        Card 0 is staging the head's model and has a second idle lane holding another queued model: seating
        that would put card 0 at its cap just as the head's weights land.
        """
        loading_lane = _lane(0, device_index=0, model=None)
        loading_lane.last_process_state = HordeProcessState.PRELOADING_MODEL
        process_map = ProcessMap(
            {
                0: loading_lane,
                1: _lane(1, device_index=0, model="model_b"),
                2: _lane(2, device_index=1, model="model_c"),
                3: _lane(3, device_index=2, model="model_d"),
            },
        )
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=_uniform_cards(3), max_threads=1)
        scheduler._horde_model_map.update_entry(
            horde_model_name="model_a",
            load_state=ModelLoadState.LOADING,
            process_id=0,
        )
        cold_head, card_zero_candidate, *_followers = await _queue(
            scheduler,
            "model_a",
            "model_b",
            "model_c",
            "model_d",
        )
        self._spend_the_bypass_window(scheduler, cold_head)

        seated = await _drain_one_cycle(scheduler)

        assert [card for _job, card in seated] == [1, 2]
        assert card_zero_candidate in scheduler._job_tracker.jobs_pending_inference
        assert cold_head in scheduler._job_tracker.jobs_pending_inference

    async def test_a_load_on_another_card_does_not_exclude_it(self) -> None:
        """Only the card carrying the head's own load is spared; a load for someone else withholds nothing."""
        loading_lane = _lane(0, device_index=0, model=None)
        loading_lane.last_process_state = HordeProcessState.PRELOADING_MODEL
        process_map = ProcessMap(
            {
                0: loading_lane,
                1: _lane(1, device_index=0, model="model_b"),
                2: _lane(2, device_index=1, model="model_c"),
            },
        )
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=_uniform_cards(2), max_threads=1)
        # The load in flight is for model_e, which no queued job here is the head for.
        scheduler._horde_model_map.update_entry(
            horde_model_name="model_e",
            load_state=ModelLoadState.LOADING,
            process_id=0,
        )
        cold_head, card_zero_candidate, other_card_candidate = await _queue(
            scheduler,
            "model_a",
            "model_b",
            "model_c",
        )
        self._spend_the_bypass_window(scheduler, cold_head)

        selection = await scheduler.get_next_job_and_process()

        assert selection is not None
        assert selection.line_skip is not None and selection.line_skip.reason == "cross_card"
        assert selection.next_job is card_zero_candidate
        assert selection.process_with_model.device_index == 0
        assert other_card_candidate in scheduler._job_tracker.jobs_pending_inference


class TestHeadPriorityBarrierScope:
    """The head-priority barrier drains the card the barred head's load is aimed at, not the whole fleet."""

    async def test_another_cards_resident_work_still_dispatches(self) -> None:
        """The barred head waits on card 0; card 1's queued job runs, since card 1 never has to drain for it."""
        head_lane = _lane(0, device_index=0, model="model_a")
        process_map = ProcessMap({0: head_lane, 1: _lane(1, device_index=1, model="model_b")})
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=_uniform_cards(2), max_threads=1)
        blocking_job, head, other_card_candidate = await _queue(scheduler, "model_a", "model_a", "model_b")
        await _seat(scheduler, blocking_job, head_lane)
        scheduler._head_admission.engage_barrier(str(head.id_))

        assert scheduler._barred_head_card_index() == 0
        assert scheduler._barrier_covers_card(1) is False

        started = await scheduler.start_inference()

        assert started is True
        assert other_card_candidate in scheduler._job_tracker.jobs_in_progress

    async def test_the_heads_own_card_is_still_withheld(self) -> None:
        """A second job on the barred head's own card is held: that card draining is the head's remedy."""
        head_lane = _lane(0, device_index=0, model="model_a")
        process_map = ProcessMap(
            {
                0: head_lane,
                1: _lane(1, device_index=0, model="model_c"),
                2: _lane(2, device_index=1, model="model_d"),
            },
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_uniform_cards(2, max_concurrent_inference=2),
            max_threads=2,
        )
        blocking_job, head, same_card_candidate = await _queue(scheduler, "model_a", "model_a", "model_c")
        await _seat(scheduler, blocking_job, head_lane)
        scheduler._head_admission.engage_barrier(str(head.id_))

        assert scheduler._barrier_covers_card(0) is True

        started = await scheduler.start_inference()

        assert started is False
        assert same_card_candidate in scheduler._job_tracker.jobs_pending_inference

    async def test_a_single_card_barrier_withholds_everything(self) -> None:
        """Single-GPU parity: with one card the barrier holds every dispatch but the head's, as it always has."""
        head_lane = _lane(0, device_index=0, model="model_a")
        process_map = ProcessMap({0: head_lane, 1: _lane(1, device_index=0, model="model_c")})
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_uniform_cards(1, max_concurrent_inference=2),
            max_threads=2,
        )
        blocking_job, head, same_card_candidate = await _queue(scheduler, "model_a", "model_a", "model_c")
        await _seat(scheduler, blocking_job, head_lane)
        scheduler._head_admission.engage_barrier(str(head.id_))

        assert scheduler._multi_gpu_routing_active is False
        assert scheduler._barrier_covers_card(0) is True

        assert await scheduler.start_inference() is False
        assert same_card_candidate in scheduler._job_tracker.jobs_pending_inference


class TestDegradedRetryIsolation:
    """A degraded retry waits for the card it would land on to empty, not for the whole worker."""

    def _mark_degraded(self, scheduler: InferenceScheduler, job: ImageGenerateJobPopResponse) -> None:
        """Flag ``job`` for the isolated retry the tracker grants after a resource failure."""
        tracked = scheduler._job_tracker.get_tracked_job(job.id_)
        assert tracked is not None
        tracked.needs_degraded_dispatch = True

    async def test_it_waits_only_for_its_own_card(self) -> None:
        """Card 1 is sampling; the retry's own card is empty, so it runs rather than waiting for the fleet."""
        retry_lane = _lane(0, device_index=0, model="model_a")
        busy_lane = _lane(1, device_index=1, model="model_b")
        scheduler = _make_scheduler(
            process_map=ProcessMap({0: retry_lane, 1: busy_lane}),
            card_runtimes=_uniform_cards(2),
            max_threads=1,
        )
        retry_job, other_card_job = await _queue(scheduler, "model_a", "model_b")
        await _seat(scheduler, other_card_job, busy_lane)
        self._mark_degraded(scheduler, retry_job)

        started = await scheduler.start_inference()

        assert started is True
        assert retry_job in scheduler._job_tracker.jobs_in_progress

    async def test_it_still_waits_for_work_on_its_own_card(self) -> None:
        """A sibling sampling on the retry's own card holds it: that is the VRAM pressure isolation is for."""
        retry_lane = _lane(0, device_index=0, model="model_a")
        sibling_lane = _lane(1, device_index=0, model="model_b")
        scheduler = _make_scheduler(
            process_map=ProcessMap({0: retry_lane, 1: sibling_lane, 2: _lane(2, device_index=1, model="model_c")}),
            card_runtimes=_uniform_cards(2, max_concurrent_inference=2),
            max_threads=2,
        )
        retry_job, sibling_job = await _queue(scheduler, "model_a", "model_b")
        await _seat(scheduler, sibling_job, sibling_lane)
        self._mark_degraded(scheduler, retry_job)

        started = await scheduler.start_inference()

        assert started is False
        assert retry_job in scheduler._job_tracker.jobs_pending_inference

    async def test_one_card_waits_for_the_whole_worker(self) -> None:
        """Single-GPU parity: with one card the scope is the worker's in-progress count, exactly as before."""
        retry_lane = _lane(0, device_index=0, model="model_a")
        sibling_lane = _lane(1, device_index=0, model="model_b")
        scheduler = _make_scheduler(
            process_map=ProcessMap({0: retry_lane, 1: sibling_lane}),
            card_runtimes=_uniform_cards(1, max_concurrent_inference=2),
            max_threads=2,
        )
        retry_job, sibling_job = await _queue(scheduler, "model_a", "model_b")
        await _seat(scheduler, sibling_job, sibling_lane)
        self._mark_degraded(scheduler, retry_job)

        assert scheduler._multi_gpu_routing_active is False
        assert await scheduler.start_inference() is False
        assert retry_job in scheduler._job_tracker.jobs_pending_inference


class TestHeadPriorityBarrierForAColdHead:
    """A barred head with no copy anywhere has not chosen a card, so the barrier follows where it may load."""

    async def test_an_idle_eligible_card_means_nothing_is_withheld(self) -> None:
        """Card 1 is eligible for the head and already free of live work, so the drain is already satisfied.

        The escape the barrier waits for is a card with no live consumer where the head's load will land.
        One exists, so withholding card 0's resident work would cost duty and bring the admit no closer.
        """
        busy_lane = _lane(0, device_index=0, model="model_b")
        process_map = ProcessMap(
            {
                0: busy_lane,
                1: _lane(1, device_index=0, model="model_c"),
                2: _lane(2, device_index=1, model=None),
            },
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_uniform_cards(2, max_concurrent_inference=2),
            max_threads=2,
        )
        cold_head, running_job, same_card_candidate = await _queue(scheduler, "model_a", "model_b", "model_c")
        await _seat(scheduler, running_job, busy_lane)
        scheduler._head_admission.engage_barrier(str(cold_head.id_))

        assert scheduler._barred_head_card_index() is None
        assert scheduler._barrier_covers_card(0) is False

        started = await scheduler.start_inference()

        assert started is True
        assert same_card_candidate in scheduler._job_tracker.jobs_in_progress

    async def _every_eligible_card_busy(self) -> tuple[InferenceScheduler, ImageGenerateJobPopResponse]:
        """Cards 0 and 1 busy and eligible for a cold barred head, card 2 unable to serve it at all.

        Returns:
            The scheduler and the candidate resident on card 2.
        """
        card_zero_lane = _lane(0, device_index=0, model="model_b")
        card_one_lane = _lane(1, device_index=1, model="model_c")
        process_map = ProcessMap(
            {
                0: card_zero_lane,
                1: card_one_lane,
                2: _lane(2, device_index=2, model="model_e"),
            },
        )
        card_runtimes = _uniform_cards(3, max_concurrent_inference=2)
        # Card 2 offers only model_e, so the head's model can never be placed there while its own resident
        # candidate still can be dispatched onto it.
        card_runtimes[2] = dataclasses.replace(
            card_runtimes[2],
            config=_card_config(models=["model_e"], max_pixels=5_000_000),
        )
        scheduler = _make_scheduler(process_map=process_map, card_runtimes=card_runtimes, max_threads=2)
        cold_head, running_zero, running_one, ineligible_card_candidate = await _queue(
            scheduler,
            "model_a",
            "model_b",
            "model_c",
            "model_e",
        )
        await _seat(scheduler, running_zero, card_zero_lane)
        await _seat(scheduler, running_one, card_one_lane)
        scheduler._head_admission.engage_barrier(str(cold_head.id_))
        return scheduler, ineligible_card_candidate

    async def test_a_card_that_could_never_host_the_head_keeps_dispatching(self) -> None:
        """Card 2's config cannot serve the head, so no amount of draining it admits the head; it keeps serving.

        This is the card whose work the fleet-wide answer used to withhold for nothing.
        """
        scheduler, ineligible_card_candidate = await self._every_eligible_card_busy()

        assert scheduler._barred_head_card_index() is None
        assert scheduler._barred_head_eligible_cards() == {0, 1}
        assert scheduler._barrier_covers_card(2) is False

        started = await scheduler.start_inference()

        assert started is True
        assert ineligible_card_candidate in scheduler._job_tracker.jobs_in_progress

    async def test_the_eligible_busy_cards_are_the_ones_that_drain(self) -> None:
        """A job that would land on an eligible card is withheld, so those cards empty for the head's load.

        Both cards can serve the head and both are sampling, so its load has nowhere to go until one frees.
        Card 0's spare lane holds a queued model and is the only thing that could take its second slot.
        """
        card_zero_lane = _lane(0, device_index=0, model="model_b")
        card_one_lane = _lane(1, device_index=1, model="model_c")
        process_map = ProcessMap(
            {
                0: card_zero_lane,
                1: card_one_lane,
                2: _lane(2, device_index=0, model="model_d"),
            },
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=_uniform_cards(2, max_concurrent_inference=2),
            max_threads=2,
        )
        cold_head, running_zero, running_one, eligible_card_candidate = await _queue(
            scheduler,
            "model_a",
            "model_b",
            "model_c",
            "model_d",
        )
        await _seat(scheduler, running_zero, card_zero_lane)
        await _seat(scheduler, running_one, card_one_lane)
        scheduler._head_admission.engage_barrier(str(cold_head.id_))

        assert scheduler._barred_head_card_index() is None
        assert scheduler._barrier_covers_card(0) is True
        assert scheduler._barrier_covers_card(1) is True

        started = await scheduler.start_inference()

        assert started is False
        assert eligible_card_candidate in scheduler._job_tracker.jobs_pending_inference


def _withhold_for_residency_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
    scheduler: InferenceScheduler,
    *jobs: ImageGenerateJobPopResponse,
) -> None:
    """Make the dispatch residency-reconciliation gate withhold exactly ``jobs``.

    The gate's own arithmetic (the arbiter verdict, the eviction it issues, the room it prices a displaced
    head at) has tests of its own; what is under test here is what the dispatch loop does once a gate has said
    no, so the verdict is supplied directly and the pricing is kept inert.
    """
    held = {str(job.id_) for job in jobs}
    monkeypatch.setattr(scheduler, "_budget_active", lambda: True)
    monkeypatch.setattr(scheduler, "_displaced_head_outstanding_mb", lambda _job, **_kwargs: None)
    monkeypatch.setattr(
        scheduler,
        "_dispatch_residency_reconciliation_holds",
        lambda next_job, _process, **_kwargs: str(next_job.id_) in held,
    )


def _withhold_for_post_processing(
    monkeypatch: pytest.MonkeyPatch,
    scheduler: InferenceScheduler,
    *jobs: ImageGenerateJobPopResponse,
) -> None:
    """Make the post-processing co-residency defer gate withhold exactly ``jobs``."""
    held = {str(job.id_) for job in jobs}
    monkeypatch.setattr(
        scheduler,
        "_should_defer_dispatch_for_post_processing",
        lambda next_job, *, process_with_model: str(next_job.id_) in held,
    )


def _withhold_for_retained_residents(
    monkeypatch: pytest.MonkeyPatch,
    scheduler: InferenceScheduler,
    *jobs: ImageGenerateJobPopResponse,
) -> None:
    """Make the retained-resident dispatch gate withhold exactly ``jobs``."""
    held = {str(job.id_) for job in jobs}
    monkeypatch.setattr(
        scheduler,
        "_retained_resident_dispatch_holds",
        lambda next_job, _process: str(next_job.id_) in held,
    )


class TestAWithheldDispatchDoesNotEndTheCycle:
    """A gate that withholds the selected job speaks for one card; the rest of the fleet keeps dispatching."""

    async def test_one_card_still_ends_the_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single-GPU parity: with one card a withheld head ends the dispatch loop, exactly as it always has.

        There is nowhere else for the resumed walk to look, so the fix is inert here by construction rather
        than by a flag: the cross-card walk is the only re-selection, and it has no other card to offer.
        """
        scheduler = _make_scheduler(
            process_map=ProcessMap(
                {
                    0: _lane(0, device_index=0, model="model_a"),
                    1: _lane(1, device_index=0, model="model_b"),
                },
            ),
            card_runtimes=_uniform_cards(1, max_concurrent_inference=2),
            max_threads=2,
        )
        head, follower = await _queue(scheduler, "model_a", "model_b")
        _withhold_for_residency_reconciliation(monkeypatch, scheduler, head)

        assert scheduler._multi_gpu_routing_active is False
        assert await scheduler.start_inference() is False
        assert head in scheduler._job_tracker.jobs_pending_inference
        assert follower in scheduler._job_tracker.jobs_pending_inference

    async def test_another_cards_resident_work_runs_in_the_same_cycle(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Best case: the head is withheld on card 0 and card 1's resident job dispatches on the same pass."""
        scheduler = _make_scheduler(
            process_map=ProcessMap(
                {
                    0: _lane(0, device_index=0, model="model_a"),
                    1: _lane(1, device_index=1, model="model_b"),
                    2: _lane(2, device_index=2, model="model_c"),
                },
            ),
            card_runtimes=_uniform_cards(3),
            max_threads=1,
        )
        head, follower, _third = await _queue(scheduler, "model_a", "model_b", "model_c")
        _withhold_for_residency_reconciliation(monkeypatch, scheduler, head)

        started = await scheduler.start_inference()

        assert started is True
        assert follower in scheduler._job_tracker.jobs_in_progress
        assert head in scheduler._job_tracker.jobs_pending_inference

    async def test_work_only_on_the_withheld_card_dispatches_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Worst case: every seatable job sits on the withheld head's own card, so the loop ends empty.

        That card's capacity is the withheld head's to claim the moment its gate releases, so seating a
        follower there would take exactly what the wait is for. The call returning at all is the no-spin half
        of the claim: a re-selection that kept offering the same job would hang rather than fail.
        """
        scheduler = _make_scheduler(
            process_map=ProcessMap(
                {
                    0: _lane(0, device_index=0, model="model_a"),
                    1: _lane(1, device_index=0, model="model_b"),
                    2: _lane(2, device_index=1, model="model_c"),
                },
            ),
            card_runtimes=_uniform_cards(2, max_concurrent_inference=2),
            max_threads=2,
        )
        head, same_card_follower = await _queue(scheduler, "model_a", "model_b")
        _withhold_for_residency_reconciliation(monkeypatch, scheduler, head)

        assert await scheduler.start_inference() is False
        assert same_card_follower in scheduler._job_tracker.jobs_pending_inference

    async def test_a_post_processing_defer_leaves_the_other_cards_alone(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Corner: the head is deferred behind a chain on its own card; another card's job still runs."""
        scheduler = _make_scheduler(
            process_map=ProcessMap(
                {
                    0: _lane(0, device_index=0, model="model_a"),
                    1: _lane(1, device_index=1, model="model_b"),
                },
            ),
            card_runtimes=_uniform_cards(2),
            max_threads=1,
        )
        head, follower = await _queue(scheduler, "model_a", "model_b")
        _withhold_for_post_processing(monkeypatch, scheduler, head)

        assert await scheduler.start_inference() is True
        assert follower in scheduler._job_tracker.jobs_in_progress
        assert head in scheduler._job_tracker.jobs_pending_inference

    async def test_a_retained_resident_hold_leaves_the_other_cards_alone(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Corner: the head waits for a sibling's retained weights on its card; another card's job runs."""
        scheduler = _make_scheduler(
            process_map=ProcessMap(
                {
                    0: _lane(0, device_index=0, model="model_a"),
                    1: _lane(1, device_index=1, model="model_b"),
                },
            ),
            card_runtimes=_uniform_cards(2),
            max_threads=1,
        )
        head, follower = await _queue(scheduler, "model_a", "model_b")
        _withhold_for_retained_residents(monkeypatch, scheduler, head)

        assert await scheduler.start_inference() is True
        assert follower in scheduler._job_tracker.jobs_in_progress
        assert head in scheduler._job_tracker.jobs_pending_inference

    async def test_two_withheld_jobs_in_one_cycle_reach_the_third(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Corner: two gates fire in sequence within one pass; both are excluded and the third job runs."""
        scheduler = _make_scheduler(
            process_map=ProcessMap(
                {
                    0: _lane(0, device_index=0, model="model_a"),
                    1: _lane(1, device_index=1, model="model_b"),
                    2: _lane(2, device_index=2, model="model_c"),
                },
            ),
            card_runtimes=_uniform_cards(3),
            max_threads=1,
        )
        scheduler.begin_scheduling_cycle()
        head, second, third = await _queue(scheduler, "model_a", "model_b", "model_c")
        _withhold_for_residency_reconciliation(monkeypatch, scheduler, head, second)

        assert await scheduler.start_inference() is True
        assert third in scheduler._job_tracker.jobs_in_progress
        assert set(scheduler.dispatch_holds.cycle_declines) == {str(head.id_), str(second.id_)}
        # Nothing is left that another card can take, so the loop ends rather than re-offering a job whose
        # gate has already answered for this cycle.
        assert await scheduler.start_inference() is False

    async def test_the_look_ahead_and_the_dispatch_call_agree_after_an_exclusion(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Edge: once jobs are excluded, both selection calls of the cycle still give one answer."""
        scheduler = _make_scheduler(
            process_map=ProcessMap(
                {
                    0: _lane(0, device_index=0, model="model_a"),
                    1: _lane(1, device_index=1, model="model_b"),
                },
            ),
            card_runtimes=_uniform_cards(2),
            max_threads=1,
        )
        scheduler.begin_scheduling_cycle()
        head, follower = await _queue(scheduler, "model_a", "model_b")
        _withhold_for_residency_reconciliation(monkeypatch, scheduler, head, follower)

        assert await scheduler.start_inference() is False

        peeked = await scheduler.get_next_job_and_process(information_only=True)
        selected = await scheduler.get_next_job_and_process()

        assert peeked is None
        assert selected is None

    async def test_the_exclusion_does_not_survive_the_cycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Edge: the next cycle re-asks the gate, so a released hold dispatches rather than staying excluded."""
        scheduler = _make_scheduler(
            process_map=ProcessMap(
                {
                    0: _lane(0, device_index=0, model="model_a"),
                    1: _lane(1, device_index=1, model="model_b"),
                },
            ),
            card_runtimes=_uniform_cards(2),
            max_threads=1,
        )
        scheduler.begin_scheduling_cycle()
        head, _follower = await _queue(scheduler, "model_a", "model_b")
        _withhold_for_residency_reconciliation(monkeypatch, scheduler, head)

        await scheduler.start_inference()
        assert str(head.id_) in scheduler.dispatch_holds.cycle_declines

        scheduler.begin_scheduling_cycle()

        assert scheduler.dispatch_holds.cycle_declines == {}
        selection = await scheduler.get_next_job_and_process()
        assert selection is not None
        assert selection.next_job is head


class TestEveryWithheldDispatchIsRecorded:
    """A dispatch declined with no record is a stall nothing can explain, so each gate names itself."""

    async def _decline_for(
        self,
        scheduler: InferenceScheduler,
        job: ImageGenerateJobPopResponse,
    ) -> DispatchDecline:
        """Run one dispatch pass that must dispatch nothing, and return the record it made for ``job``."""
        assert await scheduler.start_inference() is False
        decline = scheduler.dispatch_holds.cycle_declines.get(str(job.id_))
        assert decline is not None, f"no decline recorded; declines: {scheduler.dispatch_holds.cycle_declines}"
        return decline

    def _one_card_scheduler(self, *lanes: tuple[int, str]) -> InferenceScheduler:
        """A single-card worker whose every lane is idle and holds the named model."""
        return _make_scheduler(
            process_map=ProcessMap(
                {process_id: _lane(process_id, device_index=0, model=model) for process_id, model in lanes},
            ),
            card_runtimes=_uniform_cards(1, max_concurrent_inference=2),
            max_threads=2,
        )

    async def test_the_head_priority_barrier_names_itself(self) -> None:
        """A dispatch withheld to drain the card for a starved head is recorded as the barrier's."""
        head_lane = _lane(0, device_index=0, model="model_a")
        scheduler = _make_scheduler(
            process_map=ProcessMap({0: head_lane, 1: _lane(1, device_index=0, model="model_c")}),
            card_runtimes=_uniform_cards(1, max_concurrent_inference=2),
            max_threads=2,
        )
        scheduler.begin_scheduling_cycle()
        blocking_job, head, same_card_candidate = await _queue(scheduler, "model_a", "model_a", "model_c")
        await _seat(scheduler, blocking_job, head_lane)
        scheduler._head_admission.engage_barrier(str(head.id_))

        decline = await self._decline_for(scheduler, same_card_candidate)

        assert decline.bucket is SlotDutyBucket.HEAD_PRIORITY_BARRIER
        assert decline.device_index == 0

    async def test_an_unprepared_aux_job_names_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The terminal aux gate records its refusal, so a lane held for a prefetch is never unexplained.

        Selection reads the same preparation flag and skips an unprepared job, so the gate is reachable only
        by a job whose files went missing between the walk that chose it and the gate that asks again. The
        pair is therefore selected first and the flag set afterwards, and the gate ladder is asked directly:
        a whole pass cannot be driven into this state, because selection would refuse the job before the
        terminal gate ever saw it.
        """
        scheduler = self._one_card_scheduler((0, "model_a"))
        scheduler.begin_scheduling_cycle()
        (head,) = await _queue(scheduler, "model_a")
        selection = await scheduler.get_next_job_and_process()
        assert selection is not None
        assert selection.next_job is head
        assert selection.line_skip is None
        monkeypatch.setattr(scheduler._job_tracker, "job_requires_aux_preparation", lambda job: job is head)

        decline = scheduler._dispatch_withheld_by(selection)

        assert decline is not None
        assert decline.bucket is SlotDutyBucket.AUX_PREPARATION

    async def test_a_degraded_retry_names_itself(self) -> None:
        """An isolated retry waiting for its card to clear is recorded as the isolation wait it is."""
        retry_lane = _lane(0, device_index=0, model="model_a")
        sibling_lane = _lane(1, device_index=0, model="model_b")
        scheduler = _make_scheduler(
            process_map=ProcessMap({0: retry_lane, 1: sibling_lane}),
            card_runtimes=_uniform_cards(1, max_concurrent_inference=2),
            max_threads=2,
        )
        scheduler.begin_scheduling_cycle()
        retry_job, sibling_job = await _queue(scheduler, "model_a", "model_b")
        await _seat(scheduler, sibling_job, sibling_lane)
        tracked = scheduler._job_tracker.get_tracked_job(retry_job.id_)
        assert tracked is not None
        tracked.needs_degraded_dispatch = True

        decline = await self._decline_for(scheduler, retry_job)

        assert decline.bucket is SlotDutyBucket.DEGRADED_ISOLATION_PENDING

    async def test_a_whole_card_prestage_names_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pre-staged head waiting for sole residency is recorded as the convergence wait."""
        scheduler = self._one_card_scheduler((0, "model_a"))
        scheduler.begin_scheduling_cycle()
        (head,) = await _queue(scheduler, "model_a")
        monkeypatch.setattr(scheduler, "_prestaged_whole_card_not_ready", lambda _job: True)

        decline = await self._decline_for(scheduler, head)

        assert decline.bucket is SlotDutyBucket.WHOLE_CARD_CONVERGENCE

    async def test_a_post_processing_defer_names_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A head deferred behind a post-processing chain is recorded as that defer."""
        scheduler = self._one_card_scheduler((0, "model_a"))
        scheduler.begin_scheduling_cycle()
        (head,) = await _queue(scheduler, "model_a")
        _withhold_for_post_processing(monkeypatch, scheduler, head)

        decline = await self._decline_for(scheduler, head)

        assert decline.bucket is SlotDutyBucket.POST_PROCESSING_DEFER

    async def test_a_residency_reconciliation_hold_names_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A head whose materialisation would over-commit the card is recorded as the reconciliation wait."""
        scheduler = self._one_card_scheduler((0, "model_a"))
        scheduler.begin_scheduling_cycle()
        (head,) = await _queue(scheduler, "model_a")
        _withhold_for_residency_reconciliation(monkeypatch, scheduler, head)

        decline = await self._decline_for(scheduler, head)

        assert decline.bucket is SlotDutyBucket.RESIDENCY_RECONCILIATION

    async def test_a_retained_resident_hold_names_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A head waiting for a sibling's retained weights is recorded in the same reconciliation bucket."""
        scheduler = self._one_card_scheduler((0, "model_a"))
        scheduler.begin_scheduling_cycle()
        (head,) = await _queue(scheduler, "model_a")
        _withhold_for_retained_residents(monkeypatch, scheduler, head)

        decline = await self._decline_for(scheduler, head)

        assert decline.bucket is SlotDutyBucket.RESIDENCY_RECONCILIATION

    async def test_the_stall_explainer_names_the_gate_that_withheld_the_head(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The explainer reads the pass's own record, so a withheld head is never reported as unexplained."""
        scheduler = self._one_card_scheduler((0, "model_a"))
        scheduler.begin_scheduling_cycle()
        (head,) = await _queue(scheduler, "model_a")
        _withhold_for_residency_reconciliation(monkeypatch, scheduler, head)

        await self._decline_for(scheduler, head)
        bucket, reason = scheduler._classify_dispatch_stall(head, {})

        assert bucket is SlotDutyBucket.RESIDENCY_RECONCILIATION
        assert "over-commit the card" in reason
        assert "no matching gate" not in reason
