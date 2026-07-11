"""Auxiliary preparation must not reserve a sampling slot before the job can sample.

A resident-model job may still need a long LoRA download.  Sending the ordinary inference command for
that work marks the job in progress and reserves its future activation peak before the download begins.
The line-skip selector can then find a bounded, auxiliary-ready job for an idle sibling, but dispatch-time
admission charges the downloading job's reservation and withholds the selected work.  Both lanes remain
non-sampling even though the later job fits the card by itself.

These tests require auxiliary resolution to remain a pending-queue operation.  The downloading job keeps
its queue position without owning a sampling reservation; once its auxiliaries are ready it competes at the
normal dispatch gate, which continues to account for any backfill already sampling.  The controls retain the
direct path for jobs that need no download and for LoRA jobs whose files are already known to be cached.
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
            aux_download_deadline_for_dispatch=Mock(return_value=120.0),
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
        """A LoRA-blocked head stays pending while its auxiliary files are resolved."""
        backfill = _job(_BACKFILL_MODEL)
        scheduler, tracker, head, head_process = await _resident_head_setup(followers=(backfill,))

        started = await scheduler.start_inference()

        assert started is False
        assert head in tracker.jobs_pending_inference
        assert head not in tracker.jobs_in_progress
        assert head_process.last_control_flag is HordeControlFlag.PREPARE_AUX_MODELS
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

        assert await scheduler.start_inference() is False
        assert head_process.last_process_state is HordeProcessState.DOWNLOADING_AUX_MODEL
        assert head not in tracker.jobs_in_progress

        assert await scheduler.start_inference() is True
        assert backfill in tracker.jobs_in_progress
        assert head in tracker.jobs_pending_inference
        assert head not in tracker.jobs_in_progress
        assert backfill_process.last_control_flag is HordeControlFlag.START_INFERENCE

        # Completion only makes the head eligible; the backfill's live reservation remains authoritative.
        tracker.mark_job_loras_cached(head)
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
    """Either event ordering makes progress without admitting incompatible sampling peaks together."""
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

    assert await scheduler.start_inference() is False
    for event in event_order:
        if event == "complete":
            tracker.mark_job_loras_cached(head)
            head_process.last_process_state = HordeProcessState.PRELOADED_MODEL
        else:
            await scheduler.start_inference()

    await scheduler.start_inference()

    assert len(tracker.jobs_in_progress) == 1
    expected = backfill if event_order[0] == "schedule" else head
    assert tracker.jobs_in_progress[0] is expected


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

    assert await scheduler.start_inference() is False
    assert await scheduler.start_inference() is False
    assert tracker.jobs_in_progress == ()
    assert head_process.last_control_flag is HordeControlFlag.PREPARE_AUX_MODELS

    free_mb["value"] = 12_348.0
    scheduler._vram_arbiter = None
    assert await scheduler.start_inference() is True
    assert tracker.jobs_in_progress == (backfill,)


async def test_process_replacement_during_preparation_retries_without_claiming_the_job() -> None:
    """Losing the preparation lane leaves no orphan reservation and a replacement can retry the command."""
    scheduler, tracker, head, old_process = await _resident_head_setup()

    assert await scheduler.start_inference() is False
    replacement = make_mock_process_info(0, model_name=_HEAD_MODEL, state=HordeProcessState.PRELOADED_MODEL)
    replacement.total_vram_mb = 16_375
    replacement.process_reserved_mb = 1_372
    scheduler._process_map[0] = replacement

    assert await scheduler.start_inference() is False
    assert old_process.last_control_flag is HordeControlFlag.PREPARE_AUX_MODELS
    assert replacement.last_control_flag is HordeControlFlag.PREPARE_AUX_MODELS
    assert head in tracker.jobs_pending_inference
    assert head not in tracker.jobs_in_progress
    assert scheduler._reserve_ledger.effective_planned_vram_mb({0: 1_372.0}) == 0.0


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

    assert await scheduler.start_inference() is False
    assert head in tracker.jobs_pending_inference
    assert head not in tracker.jobs_in_progress
    assert head_process.last_control_flag is HordeControlFlag.PREPARE_AUX_MODELS


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
        tracker.mark_job_loras_cached(head)

        assert await scheduler.start_inference() is True
        assert head in tracker.jobs_in_progress
        assert head_process.last_control_flag is HordeControlFlag.START_INFERENCE

    async def test_failed_prepare_send_leaves_job_pending_and_retryable(self) -> None:
        """A failed preparation send neither claims the job nor converts it into an inference fault."""
        scheduler, tracker, head, head_process = await _resident_head_setup()
        head_process.pipe_connection.send.side_effect = Exception("closed test pipe")

        assert await scheduler.start_inference() is False
        assert head in tracker.jobs_pending_inference
        assert head not in tracker.jobs_in_progress
        assert head_process.last_control_flag is not HordeControlFlag.START_INFERENCE

        head_process.pipe_connection.send.side_effect = None
        assert await scheduler.start_inference() is False
        assert head_process.last_control_flag is HordeControlFlag.PREPARE_AUX_MODELS
        assert head not in tracker.jobs_in_progress
