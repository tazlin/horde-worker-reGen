"""An auxiliary-gated job must hold no sampling slot before it can sample.

A resident-model job may still need its LoRAs/TIs placed on disk. The pop-time prefetch pipeline does that
while the job stays pending, so the job is invisible to dispatch (and preload) until its files land and its
preparation gate clears. It reserves no future activation peak in the meantime, so a fitting sibling for an
idle lane is dispatched instead of being withheld behind a phantom reservation.

These tests hold that contract: a gated head keeps its queue position without owning a sampling reservation;
a fitting backfill sibling is fed while it waits; and once its files are cached it competes at the ordinary
dispatch gate, which continues to account for any backfill already sampling (only one incompatible peak is
admitted at a time). The controls retain the direct path for jobs that need no auxiliary files and for jobs
whose files are already known cached.
"""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import Mock

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
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
    track_popped_job_async,
)

_SDXL = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl
_HEAD_MODEL = "queued-lora-model"
_BACKFILL_MODEL = "ready-backfill-model"


def _job(model: str, *, lora: str | None = None) -> ImageGenerateJobPopResponse:
    loras = None if lora is None else [LorasPayloadEntry(name=lora, model=1.0, clip=1.0, is_version=True)]
    return make_job_pop_response(model, width=512, height=512, ddim_steps=20, loras=loras)


def _scheduler(
    process_map: ProcessMap,
    tracker: JobTracker,
    model_map: HordeModelMap,
    *,
    max_threads: int = 1,
    high_performance_mode: bool = False,
    moderate_performance_mode: bool = False,
    device_free_mb: float = 12_348.0,
) -> InferenceScheduler:
    bridge = make_mock_bridge_data(
        max_threads=max_threads,
        high_performance_mode=high_performance_mode,
        moderate_performance_mode=moderate_performance_mode,
        image_models_to_load=[_HEAD_MODEL, _BACKFILL_MODEL],
    )
    reference = {
        _HEAD_MODEL: make_mock_model_reference_record(_HEAD_MODEL, baseline=_SDXL),
        _BACKFILL_MODEL: make_mock_model_reference_record(_BACKFILL_MODEL, baseline=_SDXL),
    }
    scheduler = InferenceScheduler(
        state=WorkerState(),
        process_map=process_map,
        horde_model_map=model_map,
        job_tracker=tracker,
        process_lifecycle=Mock(
            get_processes_with_model_for_queued_job=Mock(return_value=[]),
            is_model_load_quarantined=Mock(return_value=False),
        ),
        runtime_config=make_test_runtime_config(bridge_data=bridge),
        model_metadata=make_test_model_metadata(reference),
        max_concurrent_inference_processes=max_threads,
        max_inference_processes=2,
        lru=LRUCache(2),
    )
    scheduler.set_device_free_mb_provider(lambda _device_index: device_free_mb)
    return scheduler


async def _resident_head_setup(
    *,
    followers: tuple[ImageGenerateJobPopResponse, ...] = (),
    max_threads: int = 1,
    high_performance_mode: bool = False,
    moderate_performance_mode: bool = False,
    device_free_mb: float = 12_348.0,
) -> tuple[InferenceScheduler, JobTracker, ImageGenerateJobPopResponse, HordeProcessInfo]:
    head = _job(_HEAD_MODEL, lora="uncached-head-lora")
    head_process = make_mock_process_info(0, model_name=_HEAD_MODEL, state=HordeProcessState.PRELOADED_MODEL)
    head_process.total_vram_mb = 16_375
    head_process.process_reserved_mb = 1_372
    process_map = ProcessMap({0: head_process})
    model_map = HordeModelMap(root={})
    model_map.update_entry(_HEAD_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=0)
    tracker = JobTracker()
    await track_popped_job_async(tracker, head)
    for follower in followers:
        await track_popped_job_async(tracker, follower)
    return (
        _scheduler(
            process_map,
            tracker,
            model_map,
            max_threads=max_threads,
            high_performance_mode=high_performance_mode,
            moderate_performance_mode=moderate_performance_mode,
            device_free_mb=device_free_mb,
        ),
        tracker,
        head,
        head_process,
    )


class TestAuxPreparationRegression:
    """Preparation-only downloads leave sampling capacity available to eligible queued work."""

    async def test_uncached_resident_lora_head_prepares_without_sampling_reservation(self) -> None:
        """A LoRA-blocked head stays pending and holds no sampling reservation while its files are placed.

        Preparation is the pop-time prefetch pipeline's job, off the sampling lanes. The head is invisible to
        dispatch until its prefetch clears its gate, so it neither samples nor reserves any activation peak;
        the only follower here needs a non-resident model, so nothing else runs this pass either.
        """
        backfill = _job(_BACKFILL_MODEL)
        scheduler, tracker, head, head_process = await _resident_head_setup(followers=(backfill,))

        started = await scheduler.start_inference()

        assert started is False
        assert head in tracker.jobs_pending_inference
        assert head not in tracker.jobs_in_progress
        assert head_process.last_control_flag is not HordeControlFlag.START_INFERENCE
        assert scheduler._reserve_ledger.effective_planned_vram_mb({0: 1_372.0}) == 0.0

    async def test_prepared_head_allows_ready_backfill_without_weakening_later_admission(self) -> None:
        """Prepared pending work frees the idle lane, while its eventual sampling still re-enters admission."""
        head = _job(_HEAD_MODEL, lora="uncached-head-lora")
        second_lora = _job(_HEAD_MODEL, lora="another-uncached-lora")
        backfill = _job(_BACKFILL_MODEL)
        tail = _job(_HEAD_MODEL)
        head_process = make_mock_process_info(0, model_name=_HEAD_MODEL, state=HordeProcessState.PRELOADED_MODEL)
        backfill_process = make_mock_process_info(
            1,
            model_name=_BACKFILL_MODEL,
            state=HordeProcessState.PRELOADED_MODEL,
        )
        for process in (head_process, backfill_process):
            process.total_vram_mb = 16_375
            process.process_reserved_mb = 1_372
        model_map = HordeModelMap(root={})
        model_map.update_entry(_HEAD_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=0)
        model_map.update_entry(_BACKFILL_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)
        tracker = JobTracker()
        for job in (head, second_lora, backfill, tail):
            await track_popped_job_async(tracker, job)
        scheduler = _scheduler(
            ProcessMap({0: head_process, 1: backfill_process}),
            tracker,
            model_map,
            max_threads=2,
            device_free_mb=11_000.0,
        )

        # The gated head is invisible to dispatch, so the resident backfill sibling is fed on the first pass.
        assert await scheduler.start_inference() is True
        assert backfill in tracker.jobs_in_progress
        assert head in tracker.jobs_pending_inference
        assert head not in tracker.jobs_in_progress
        assert backfill_process.last_control_flag is HordeControlFlag.START_INFERENCE

        # Completion only makes the head eligible; the backfill's live reservation remains authoritative.
        mark_job_aux_prepared(tracker, head)
        head_process.last_process_state = HordeProcessState.PRELOADED_MODEL
        model_map.update_entry(_HEAD_MODEL, load_state=ModelLoadState.LOADED_IN_VRAM, process_id=0)
        scheduler._concurrent_overlap_allowed = Mock(return_value=True)  # type: ignore[method-assign]
        scheduler._vram_arbiter = None
        assert await scheduler.start_inference() is False
        assert head not in tracker.jobs_in_progress


@pytest.mark.parametrize(
    "event_order",
    [
        ("schedule", "complete"),
        ("complete", "schedule"),
    ],
)
async def test_download_completion_and_backfill_dispatch_orderings_admit_only_one_sampler(
    event_order: Sequence[str],
) -> None:
    """Either event ordering makes progress without admitting incompatible sampling peaks together.

    With a single concurrency lane, the gated LoRA head is invisible while its files are placed, so the
    resident backfill is fed first regardless of when the download completes or a schedule pass runs. The head
    then waits behind the one admitted sampler rather than stacking a second incompatible peak.
    """
    head = _job(_HEAD_MODEL, lora="head-lora")
    backfill = _job(_BACKFILL_MODEL)
    head_process = make_mock_process_info(0, model_name=_HEAD_MODEL, state=HordeProcessState.PRELOADED_MODEL)
    backfill_process = make_mock_process_info(1, model_name=_BACKFILL_MODEL, state=HordeProcessState.PRELOADED_MODEL)
    for process in (head_process, backfill_process):
        process.total_vram_mb = 16_375
        process.process_reserved_mb = 1_372
    tracker = JobTracker()
    await track_popped_job_async(tracker, head)
    await track_popped_job_async(tracker, backfill)
    model_map = HordeModelMap(root={})
    model_map.update_entry(_HEAD_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=0)
    model_map.update_entry(_BACKFILL_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)
    scheduler = _scheduler(ProcessMap({0: head_process, 1: backfill_process}), tracker, model_map)

    # The gated head yields the first lane to the resident backfill.
    assert await scheduler.start_inference() is True
    for event in event_order:
        if event == "complete":
            mark_job_aux_prepared(tracker, head)
            head_process.last_process_state = HordeProcessState.PRELOADED_MODEL
        else:
            await scheduler.start_inference()

    await scheduler.start_inference()

    assert len(tracker.jobs_in_progress) == 1
    assert tracker.jobs_in_progress[0] is backfill


async def test_backfill_retries_after_measured_vram_recovers() -> None:
    """A temporary physical-memory hold defers safely and dispatches once a later reading fits."""
    head = _job(_HEAD_MODEL, lora="head-lora")
    backfill = _job(_BACKFILL_MODEL)
    head_process = make_mock_process_info(0, model_name=_HEAD_MODEL, state=HordeProcessState.PRELOADED_MODEL)
    backfill_process = make_mock_process_info(1, model_name=_BACKFILL_MODEL, state=HordeProcessState.PRELOADED_MODEL)
    for process in (head_process, backfill_process):
        process.total_vram_mb = 16_375
        process.process_reserved_mb = 1_372
    tracker = JobTracker()
    await track_popped_job_async(tracker, head)
    await track_popped_job_async(tracker, backfill)
    model_map = HordeModelMap(root={})
    model_map.update_entry(_HEAD_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=0)
    model_map.update_entry(_BACKFILL_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)
    scheduler = _scheduler(ProcessMap({0: head_process, 1: backfill_process}), tracker, model_map)
    free_mb = {"value": 8_500.0}
    scheduler.set_device_free_mb_provider(lambda _device_index: free_mb["value"])

    # The gated head holds nothing; the backfill is the dispatch head but its sampling peak does not fit the
    # temporarily-held card, so it defers (holding no reservation) rather than over-committing.
    assert await scheduler.start_inference() is False
    assert await scheduler.start_inference() is False
    assert tracker.jobs_in_progress == ()
    assert head_process.last_control_flag is not HordeControlFlag.START_INFERENCE

    free_mb["value"] = 12_348.0
    scheduler._vram_arbiter = None
    assert await scheduler.start_inference() is True
    assert tracker.jobs_in_progress == (backfill,)


@pytest.mark.parametrize(
    ("queue_shape", "max_threads", "high_performance", "moderate_performance", "device_free_mb"),
    [
        ("head-only", 1, False, False, 12_348.0),
        ("non-lora-follower", 1, False, False, 12_348.0),
        ("lora-run-then-backfill", 1, True, False, 12_348.0),
        ("mixed-four-deep", 2, False, True, 9_500.0),
        ("ample-card", 2, False, False, 24_000.0),
    ],
)
async def test_aux_preparation_is_independent_of_queue_and_worker_shape(
    queue_shape: str,
    max_threads: int,
    high_performance: bool,
    moderate_performance: bool,
    device_free_mb: float,
) -> None:
    """Queue depth, performance mode, concurrency, and free VRAM do not turn downloads into sampling."""
    shapes = {
        "head-only": (),
        "non-lora-follower": (_job(_BACKFILL_MODEL),),
        "lora-run-then-backfill": (_job(_HEAD_MODEL, lora="second-lora"), _job(_BACKFILL_MODEL)),
        "mixed-four-deep": (
            _job(_HEAD_MODEL, lora="second-lora"),
            _job(_BACKFILL_MODEL),
            _job(_HEAD_MODEL),
        ),
        "ample-card": (_job(_BACKFILL_MODEL),),
    }
    scheduler, tracker, head, head_process = await _resident_head_setup(
        followers=shapes[queue_shape],
        max_threads=max_threads,
        high_performance_mode=high_performance,
        moderate_performance_mode=moderate_performance,
        device_free_mb=device_free_mb,
    )

    await scheduler.start_inference()
    assert head in tracker.jobs_pending_inference
    assert head not in tracker.jobs_in_progress
    assert head_process.last_control_flag is not HordeControlFlag.START_INFERENCE


class TestAuxPreparationControls:
    """Only jobs with unresolved LoRAs take the preparation-only path."""

    async def test_non_lora_head_dispatches_normally(self) -> None:
        """A job without LoRAs proceeds directly through ordinary inference dispatch."""
        head = _job(_HEAD_MODEL)
        process = make_mock_process_info(0, model_name=_HEAD_MODEL, state=HordeProcessState.PRELOADED_MODEL)
        process.total_vram_mb = 16_375
        model_map = HordeModelMap(root={})
        model_map.update_entry(_HEAD_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=0)
        tracker = JobTracker()
        await track_popped_job_async(tracker, head)
        scheduler = _scheduler(ProcessMap({0: process}), tracker, model_map, device_free_mb=24_000.0)

        assert await scheduler.start_inference() is True
        assert head in tracker.jobs_in_progress
        assert process.last_control_flag is HordeControlFlag.START_INFERENCE

    async def test_cached_lora_head_dispatches_normally(self) -> None:
        """A job already known to have its LoRAs ready does not repeat preparation."""
        scheduler, tracker, head, head_process = await _resident_head_setup(device_free_mb=24_000.0)
        mark_job_aux_prepared(tracker, head)

        assert await scheduler.start_inference() is True
        assert head in tracker.jobs_in_progress
        assert head_process.last_control_flag is HordeControlFlag.START_INFERENCE

    async def test_unprepared_lora_head_holds_no_lane_and_never_samples(self) -> None:
        """An unprepared LoRA head neither claims its lane nor faults; it simply waits, holding nothing."""
        scheduler, tracker, head, head_process = await _resident_head_setup()

        assert await scheduler.start_inference() is False
        assert head in tracker.jobs_pending_inference
        assert head not in tracker.jobs_in_progress
        assert head_process.last_control_flag is not HordeControlFlag.START_INFERENCE
        assert scheduler._reserve_ledger.effective_planned_vram_mb({0: 1_372.0}) == 0.0

        # Once its files are known cached, the same head dispatches through ordinary admission.
        mark_job_aux_prepared(tracker, head)
        head_process.total_vram_mb = 16_375
        assert await scheduler.start_inference() is True
        assert head in tracker.jobs_in_progress
        assert head_process.last_control_flag is HordeControlFlag.START_INFERENCE
