"""A LoRA-carrying mono-model queue must not starve the GPU while a head awaits auxiliary preparation.

A job whose base model is resident may still need its LoRAs or textual inversions placed on disk. Those
files are fetched by the dedicated download process at pop time, off the sampling lanes, so the job stays
pending and invisible to dispatch until its preparation gate clears. It holds no lane and no reservation in
the meantime, so a fitting sibling flows through ordinary admission and keeps the card sampling.

These tests pin that liveness: an unprepared LoRA candidate is not fed onto an idle lane (it would still be
missing its files), a prepared candidate whose files are already on disk dispatches immediately, a prepared
follower sharing the head's model is fed while the still-unprepared head waits, partial cache coverage does
not clear the gate, and the concurrency cap still bounds a prepared backfill because the gated head reserves
no capacity to bypass.
"""

from __future__ import annotations

from collections.abc import Iterable
from unittest.mock import Mock

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_models import NextJobAndProcess
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.lru_cache import LRUCache
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_model_metadata,
    make_test_runtime_config,
    mark_job_aux_prepared,
    mark_job_in_progress_async,
    track_popped_job_async,
)

_SDXL = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl

# Model names mirror the production mono-model SDXL/illustrious/pony mix that exhibits the wedge.
_ILLUSTRIOUS = "WAI-NSFW-illustrious-SDXL"
_PONY = "Nova Furry Pony"
_CYBER = "CyberRealistic Pony"


def _lora_entries(names: Iterable[str]) -> list[LorasPayloadEntry]:
    """LoRA payload entries for the given references (all version-pinned, as production LoRAs usually are)."""
    return [LorasPayloadEntry(name=name, model=1.0, clip=1.0, is_version=True) for name in names]


def _small_job(model: str, *, lora_names: Iterable[str] = ()) -> ImageGenerateJobPopResponse:
    """A small (512x512, 20-step) job, light enough that no size gate interferes with these scenarios."""
    entries = _lora_entries(lora_names)
    return make_job_pop_response(model, width=512, height=512, ddim_steps=20, loras=entries or None)


def _make_scheduler(
    *,
    process_map: ProcessMap,
    job_tracker: JobTracker,
    reference: dict[str, object],
    max_concurrent: int = 1,
    high_performance_mode: bool = False,
    moderate_performance_mode: bool = False,
    purge_loras_on_download: bool = False,
    state: WorkerState | None = None,
) -> InferenceScheduler:
    """Build a scheduler over an explicit process map, model reference, and job tracker.

    The device-free provider returns an ample truthful reading so measured-truth admission never withholds
    a dispatch for a missing reading: these scenarios isolate the auxiliary-preparation gate, not the VRAM
    gate.
    """
    bridge_data = make_mock_bridge_data(
        max_threads=max_concurrent,
        high_performance_mode=high_performance_mode,
        moderate_performance_mode=moderate_performance_mode,
        purge_loras_on_download=purge_loras_on_download,
    )
    scheduler = InferenceScheduler(
        state=state if state is not None else WorkerState(),
        process_map=process_map,
        horde_model_map=HordeModelMap(root={}),
        job_tracker=job_tracker,
        process_lifecycle=Mock(
            get_processes_with_model_for_queued_job=Mock(return_value=[]),
            is_model_load_quarantined=Mock(return_value=False),
        ),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        model_metadata=make_test_model_metadata(reference),
        max_concurrent_inference_processes=max_concurrent,
        max_inference_processes=max(2, len(process_map)),
        lru=LRUCache(max(2, len(process_map))),
    )
    scheduler.set_device_free_mb_provider(lambda _device_index: 24000.0)
    return scheduler


async def _dispatch(scheduler: InferenceScheduler) -> NextJobAndProcess | None:
    """One real dispatch decision (the value ``start_inference`` acts on)."""
    return await scheduler.get_next_job_and_process()


class TestLoraAuxPreparationLaneLiveness:
    """A gated head must yield its lane only to work that needs no fresh download to sample now."""

    async def test_nonlora_resident_backfill_feeds_idle_lane(self) -> None:
        """A no-LoRA candidate whose model is resident on the idle lane is fed while the LoRA head waits.

        The LoRA head is invisible to dispatch until its pop-time prefetch clears its gate, so the resident
        no-LoRA candidate simply becomes the dispatch head on the idle lane; no line-skip is needed.
        """
        head = _small_job(_ILLUSTRIOUS, lora_names=["head-lora"])
        backfill = _small_job(_PONY)

        head_lane = make_mock_process_info(0, model_name=_ILLUSTRIOUS, state=HordeProcessState.PRELOADED_MODEL)
        head_lane.last_job_referenced = head
        idle = make_mock_process_info(1, model_name=_PONY, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, backfill)

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: head_lane, 1: idle}),
            job_tracker=job_tracker,
            reference={
                _ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL),
                _PONY: make_mock_model_reference_record(_PONY, baseline=_SDXL),
            },
            max_concurrent=2,
        )

        result = await _dispatch(scheduler)
        assert result is not None
        assert result.next_job is backfill
        assert result.process_with_model.process_id == 1
        assert result.line_skip is None
        assert result.next_job is not head

    async def test_unprepared_lora_backfill_is_not_dispatched(self) -> None:
        """A candidate whose LoRAs are not yet prepared is not fed onto the idle lane; it waits its turn.

        Both the head and the candidate still need their files placed on disk, so both are invisible to
        dispatch and the card holds rather than trading one pending download for another.
        """
        head = _small_job(_ILLUSTRIOUS, lora_names=["head-lora"])
        backfill = _small_job(_PONY, lora_names=["uncached-lora"])

        head_lane = make_mock_process_info(0, model_name=_ILLUSTRIOUS, state=HordeProcessState.PRELOADED_MODEL)
        head_lane.last_job_referenced = head
        idle = make_mock_process_info(1, model_name=_PONY, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, backfill)

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: head_lane, 1: idle}),
            job_tracker=job_tracker,
            reference={
                _ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL),
                _PONY: make_mock_model_reference_record(_PONY, baseline=_SDXL),
            },
            max_concurrent=2,
        )

        assert await _dispatch(scheduler) is None

    async def test_prepared_lora_backfill_feeds_idle_lane(self) -> None:
        """A prepared LoRA candidate (its files cached) feeds the idle lane while the unprepared head waits."""
        head = _small_job(_ILLUSTRIOUS, lora_names=["head-lora"])
        backfill = _small_job(_PONY, lora_names=["cached-lora"])

        head_lane = make_mock_process_info(0, model_name=_ILLUSTRIOUS, state=HordeProcessState.PRELOADED_MODEL)
        head_lane.last_job_referenced = head
        idle = make_mock_process_info(1, model_name=_PONY, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, backfill)
        mark_job_aux_prepared(job_tracker, backfill)

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: head_lane, 1: idle}),
            job_tracker=job_tracker,
            reference={
                _ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL),
                _PONY: make_mock_model_reference_record(_PONY, baseline=_SDXL),
            },
            max_concurrent=2,
        )

        result = await _dispatch(scheduler)
        assert result is not None, "a prepared candidate whose LoRAs are already on disk should feed the idle lane"
        assert result.next_job is backfill
        assert result.process_with_model.process_id == 1
        assert result.next_job is not head

    async def test_prepared_follower_feeds_idle_lane_while_unprepared_head_waits(self) -> None:
        """A follower sharing the head's model is fed once prepared, while the still-unprepared head waits.

        Both jobs' base model is resident (head on its own lane, follower on an idle sibling). While the
        follower is unprepared it is invisible like the head, so nothing dispatches. Once its LoRAs are
        prepared it dispatches on the idle sibling; the head stays invisible until its own preparation
        completes.
        """
        head = _small_job(_ILLUSTRIOUS, lora_names=["shared-lora"])
        follower = _small_job(_ILLUSTRIOUS, lora_names=["shared-lora"])

        head_lane = make_mock_process_info(0, model_name=_ILLUSTRIOUS, state=HordeProcessState.PRELOADED_MODEL)
        head_lane.last_job_referenced = head
        idle = make_mock_process_info(1, model_name=_ILLUSTRIOUS, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, follower)

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: head_lane, 1: idle}),
            job_tracker=job_tracker,
            reference={_ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL)},
            max_concurrent=2,
        )

        assert await _dispatch(scheduler) is None, "both jobs are unprepared, so neither is dispatchable yet"

        # The follower's preparation completes (its files are on disk and its gate clears).
        mark_job_aux_prepared(job_tracker, follower)

        result = await _dispatch(scheduler)
        assert result is not None, "the prepared follower should be fed onto a resident lane"
        assert result.next_job is follower
        assert result.next_job is not head


class TestAuxPreparationGateFenceGuards:
    """The gate clears only when a job's whole auxiliary set is prepared, and never reserves capacity."""

    async def test_partial_lora_coverage_does_not_clear_the_gate(self) -> None:
        """A job needing two LoRAs, only one prepared, stays gated: the other is still missing from disk."""
        head = _small_job(_ILLUSTRIOUS, lora_names=["head-lora"])
        partial = _small_job(_PONY, lora_names=["cached-one", "still-missing"])
        cached_only = _small_job(_PONY, lora_names=["cached-one"])

        head_lane = make_mock_process_info(0, model_name=_ILLUSTRIOUS, state=HordeProcessState.PRELOADED_MODEL)
        head_lane.last_job_referenced = head
        idle = make_mock_process_info(1, model_name=_PONY, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, partial)
        mark_job_aux_prepared(job_tracker, cached_only)  # seeds only "cached-one" on disk

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: head_lane, 1: idle}),
            job_tracker=job_tracker,
            reference={
                _ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL),
                _PONY: make_mock_model_reference_record(_PONY, baseline=_SDXL),
            },
        )

        assert await _dispatch(scheduler) is None, "a candidate with any unprepared LoRA must stay gated"

    async def test_cap_reached_holds_a_prepared_backfill_behind_the_concurrency_limit(self) -> None:
        """At the concurrency cap, a prepared backfill waits: the gated head holds no lane to bypass.

        With one real sampler already occupying the single concurrency lane, a prepared backfill cannot run
        yet. The unprepared LoRA head holds nothing, so there is no lane to borrow the slot from; the backfill
        simply waits for the cap to free, the honest behaviour once a gated job reserves no capacity.
        """
        sampling_job = _small_job(_CYBER)
        head = _small_job(_ILLUSTRIOUS, lora_names=["head-lora"])
        backfill = _small_job(_PONY, lora_names=["cached-lora"])

        sampling = make_mock_process_info(0, model_name=_CYBER, state=HordeProcessState.INFERENCE_STARTING)
        sampling.last_job_referenced = sampling_job
        head_lane = make_mock_process_info(1, model_name=_ILLUSTRIOUS, state=HordeProcessState.PRELOADED_MODEL)
        head_lane.last_job_referenced = head
        idle = make_mock_process_info(2, model_name=_PONY, state=HordeProcessState.WAITING_FOR_JOB)

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, sampling_job)
        await mark_job_in_progress_async(job_tracker, sampling_job)
        await track_popped_job_async(job_tracker, head)
        await track_popped_job_async(job_tracker, backfill)
        mark_job_aux_prepared(job_tracker, backfill)

        scheduler = _make_scheduler(
            process_map=ProcessMap({0: sampling, 1: head_lane, 2: idle}),
            job_tracker=job_tracker,
            reference={
                _CYBER: make_mock_model_reference_record(_CYBER, baseline=_SDXL),
                _ILLUSTRIOUS: make_mock_model_reference_record(_ILLUSTRIOUS, baseline=_SDXL),
                _PONY: make_mock_model_reference_record(_PONY, baseline=_SDXL),
            },
            max_concurrent=1,
        )

        assert await _dispatch(scheduler) is None
