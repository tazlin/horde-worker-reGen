"""Tests for A5.1 GPU-aware dispatch routing: per-card residency lookup + eligible-card selection.

A multi-GPU scheduler must dispatch a job only to a card whose effective config can serve it, and among
several eligible resident cards prefer one ready now then the least-loaded (the "sticky, then least-loaded"
policy). A single-GPU scheduler keeps the original card-agnostic lookup, so routing is a strict no-op there.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.lru_cache import LRUCache
from horde_worker_regen.process_management.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_card_runtimes,
    make_test_model_metadata,
    make_test_runtime_config,
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
) -> InferenceScheduler:
    """Build an InferenceScheduler with a given process map and per-card runtime plan."""
    bridge_data = make_mock_bridge_data()
    bridge_data.max_threads = 2
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


def _two_cards(*, card0_max_pixels: int, card1_max_pixels: int) -> dict:
    """A 24GB card 0 and an 8GB card 1, each serving stable_diffusion, differing only in max_pixels."""
    rt0 = make_test_card_runtimes(
        device_indices=(0,),
        config=_card_config(models=["stable_diffusion"], max_pixels=card0_max_pixels),
        total_vram_mb=24576.0,
    )
    rt1 = make_test_card_runtimes(
        device_indices=(1,),
        config=_card_config(models=["stable_diffusion"], max_pixels=card1_max_pixels),
        total_vram_mb=8192.0,
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
        chosen = scheduler._resident_process_for_job(job)
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
        assert scheduler._resident_process_for_job(job) is None


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
        chosen = scheduler._resident_process_for_job(job)
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
        chosen = scheduler._resident_process_for_job(job)
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
        chosen = scheduler._resident_process_for_job(job)
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
        assert scheduler._resident_process_for_job(job) is not None


def _empty_slot(process_id: int, *, device_index: int) -> Mock:
    """An idle inference process holding no model (a free slot a preload can target)."""
    return make_mock_process_info(
        process_id,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        device_index=device_index,
    )


class TestPreloadCardPlacement:
    """A5.4: a fresh preload picks which eligible card to load onto (sticky, then least-loaded)."""

    def test_places_on_the_least_loaded_eligible_card(self) -> None:
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
        chosen = scheduler._select_preload_process(job, [])
        assert chosen is not None
        assert chosen.device_index == 1

    def test_sticky_prefers_a_card_already_holding_the_model(self) -> None:
        """A card already serving the model is preferred for the preload even if it is the busier card."""
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
            card_runtimes=_two_cards(card0_max_pixels=5_000_000, card1_max_pixels=5_000_000),
        )
        job = make_job_pop_response(model="stable_diffusion", width=512, height=512)
        chosen = scheduler._select_preload_process(job, [])
        assert chosen is not None
        assert chosen.device_index == 0

    def test_never_places_on_an_ineligible_card(self) -> None:
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
        chosen = scheduler._select_preload_process(job, [])
        assert chosen is not None
        assert chosen.device_index == 0

    def test_single_gpu_uses_first_available_slot(self) -> None:
        """With one card, placement is inactive and the first available slot is returned, as before."""
        process_map = ProcessMap({0: _empty_slot(0, device_index=0)})
        scheduler = _make_scheduler(
            process_map=process_map,
            card_runtimes=make_test_card_runtimes(device_indices=(0,)),
        )
        assert scheduler._multi_gpu_routing_active is False
        job = make_job_pop_response(model="stable_diffusion")
        chosen = scheduler._select_preload_process(job, [])
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
    received its first model -- it sat ``WAITING_FOR_JOB`` forever while the other card did all the work.
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
