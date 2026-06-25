"""Tests for A5.2 per-card concurrency: the in-progress count cap and the overlap-headway gate per card.

Cards are independent sampling/VRAM domains, so on a multi-GPU host the concurrency cap must count a
candidate against *its own card's* in-progress jobs and ceilings (a small card with ``max_threads=1`` must
not borrow the big card's headroom), and the overlap-headway gate must compare a candidate only against the
jobs already sampling on the same card (two heavy jobs on different cards do not contend). A single-GPU host
keeps the worker-wide comparison, so both gates are a strict no-op there.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.lru_cache import LRUCache
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler

from .conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_card_runtimes,
    make_test_model_metadata,
    make_test_runtime_config,
    mark_job_in_progress_async,
    track_popped_job_async,
)

_SDXL = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl


def _two_cards(*, card0_max_concurrent: int, card1_max_concurrent: int) -> dict[int, CardRuntime]:
    """A 24GB card 0 and an 8GB card 1, each with its own concurrent-sampling ceiling.

    ``target_process_count`` is one above the concurrency ceiling per card (a spare staging slot), so a
    lease-on cap can be told apart from the lease-off ceiling.
    """
    rt0 = make_test_card_runtimes(
        device_indices=(0,),
        max_concurrent_inference=card0_max_concurrent,
        target_process_count=card0_max_concurrent + 1,
        total_vram_mb=24576.0,
    )
    rt1 = make_test_card_runtimes(
        device_indices=(1,),
        max_concurrent_inference=card1_max_concurrent,
        target_process_count=card1_max_concurrent + 1,
        total_vram_mb=8192.0,
    )
    return {0: rt0[0], 1: rt1[1]}


def _make_scheduler(
    *,
    card_runtimes: dict[int, CardRuntime] | None,
    process_map: ProcessMap | None = None,
    job_tracker: JobTracker | None = None,
    reference: dict[str, object] | None = None,
) -> InferenceScheduler:
    """Build an InferenceScheduler with a per-card runtime plan and an optional model reference."""
    bridge_data = make_mock_bridge_data()
    bridge_data.max_threads = 2
    return InferenceScheduler(
        state=WorkerState(),
        process_map=process_map if process_map is not None else ProcessMap({}),
        horde_model_map=HordeModelMap(root={}),
        job_tracker=job_tracker if job_tracker is not None else JobTracker(),
        process_lifecycle=Mock(is_model_load_quarantined=Mock(return_value=False)),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        model_metadata=make_test_model_metadata(reference),
        card_runtimes=card_runtimes,
        max_concurrent_inference_processes=2,
        max_inference_processes=4,
        lru=LRUCache(4),
    )


async def _attach_in_flight_job(
    *,
    process_map: ProcessMap,
    job_tracker: JobTracker,
    reference: dict[str, object],
    process_id: int,
    device_index: int,
    steps_done: int,
    total_steps: int = 20,
) -> None:
    """Register a heavy (SDXL) job in-flight on ``process_id`` pinned to ``device_index``."""
    model = f"m{process_id}_{_SDXL.value}"
    proc = make_mock_process_info(
        process_id,
        model_name=model,
        state=HordeProcessState.INFERENCE_STARTING,
        device_index=device_index,
    )
    proc.last_total_steps = total_steps
    proc.last_current_step = steps_done
    process_map[process_id] = proc
    reference[model] = make_mock_model_reference_record(model, baseline=_SDXL)
    job = make_job_pop_response(model, ddim_steps=total_steps)
    proc.last_job_referenced = job
    await track_popped_job_async(job_tracker, job)
    await mark_job_in_progress_async(job_tracker, job)
    return job


class TestPerCardConcurrencyCap:
    """The in-progress count cap reads each card's own ceiling, not a worker-wide sum."""

    def test_per_card_ceiling_is_that_cards_max_concurrent(self) -> None:
        """With the lease off, the cap for a card is that card's concurrent-sampling ceiling."""
        cards = _two_cards(card0_max_concurrent=2, card1_max_concurrent=1)
        scheduler = _make_scheduler(card_runtimes=cards)
        # The big card admits two concurrent jobs; the small card only one, even though they share a worker.
        assert scheduler._max_jobs_in_progress_allowed(0, card=cards[0]) == 2
        assert scheduler._max_jobs_in_progress_allowed(0, card=cards[1]) == 1

    def test_global_path_matches_worker_wide_ceiling(self) -> None:
        """Passing no card keeps the worker-wide ceiling: the byte-identical single-GPU path."""
        cards = _two_cards(card0_max_concurrent=2, card1_max_concurrent=1)
        scheduler = _make_scheduler(card_runtimes=cards)
        assert scheduler._max_jobs_in_progress_allowed(0) == scheduler._max_concurrent_inference_processes


class TestJobsInProgressByCard:
    """In-progress jobs are attributed to the card their live process is pinned to."""

    async def test_attributes_in_flight_jobs_to_their_card(self) -> None:
        """Each card's in-progress view contains only the jobs whose running slot sits on that card."""
        process_map = ProcessMap({})
        job_tracker = JobTracker()
        reference: dict[str, object] = {}
        job_on_card0 = await _attach_in_flight_job(
            process_map=process_map,
            job_tracker=job_tracker,
            reference=reference,
            process_id=0,
            device_index=0,
            steps_done=5,
        )
        job_on_card1 = await _attach_in_flight_job(
            process_map=process_map,
            job_tracker=job_tracker,
            reference=reference,
            process_id=1,
            device_index=1,
            steps_done=5,
        )
        scheduler = _make_scheduler(
            card_runtimes=_two_cards(card0_max_concurrent=2, card1_max_concurrent=2),
            process_map=process_map,
            job_tracker=job_tracker,
            reference=reference,
        )
        assert scheduler._jobs_in_progress_on_card(0) == [job_on_card0]
        assert scheduler._jobs_in_progress_on_card(1) == [job_on_card1]


class TestPerCardOverlapGate:
    """The overlap-headway gate compares a candidate only against jobs sampling on the same card."""

    async def test_heavy_overlap_blocked_on_same_card_but_allowed_on_an_idle_one(self) -> None:
        """A fresh heavy job is held behind a 0%-progress heavy sibling on its card, but free on an idle card."""
        process_map = ProcessMap({})
        job_tracker = JobTracker()
        reference: dict[str, object] = {}
        await _attach_in_flight_job(
            process_map=process_map,
            job_tracker=job_tracker,
            reference=reference,
            process_id=0,
            device_index=0,
            steps_done=0,
        )
        reference["cand"] = make_mock_model_reference_record("cand", baseline=_SDXL)
        scheduler = _make_scheduler(
            card_runtimes=_two_cards(card0_max_concurrent=2, card1_max_concurrent=2),
            process_map=process_map,
            job_tracker=job_tracker,
            reference=reference,
        )
        candidate = make_job_pop_response("cand")
        # Card 0 already runs a just-started heavy job: two heavy loads must not stack with no headway.
        assert scheduler._concurrent_overlap_allowed(candidate, target_device_index=0) is False
        # Card 1 is an independent, idle domain: the candidate may take it.
        assert scheduler._concurrent_overlap_allowed(candidate, target_device_index=1) is True

    async def test_single_gpu_ignores_target_index_and_compares_worker_wide(self) -> None:
        """One card means routing is inactive, so the gate is worker-wide regardless of any target index."""
        process_map = ProcessMap({})
        job_tracker = JobTracker()
        reference: dict[str, object] = {}
        await _attach_in_flight_job(
            process_map=process_map,
            job_tracker=job_tracker,
            reference=reference,
            process_id=0,
            device_index=0,
            steps_done=0,
        )
        reference["cand"] = make_mock_model_reference_record("cand", baseline=_SDXL)
        scheduler = _make_scheduler(
            card_runtimes=make_test_card_runtimes(device_indices=(0,)),
            process_map=process_map,
            job_tracker=job_tracker,
            reference=reference,
        )
        assert scheduler._multi_gpu_routing_active is False
        candidate = make_job_pop_response("cand")
        # Even pointing at a non-existent "other" card, the single-GPU gate still sees the in-flight job.
        assert scheduler._concurrent_overlap_allowed(candidate, target_device_index=1) is False
