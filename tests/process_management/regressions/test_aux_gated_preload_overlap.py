"""An auxiliary-gated job may stage its checkpoint into a slot nothing else wants.

A job whose LoRAs/TIs are still downloading cannot sample, so it must never take capacity from a job that
can. It does not follow that its checkpoint must sit unloaded: when no other pending job is competing and
the target slot is empty, staging the weights costs nobody anything and runs the load alongside the
auxiliary download instead of serializing it afterwards. On a cold start that difference is the whole
checkpoint load added to the wait.

These tests hold both halves. The overlap half: an aux-gated job alone in the queue preloads into an empty
slot. The restraint half, which is what the gate was protecting: a dispatchable sibling outranks it, an
occupied slot is never displaced for it, and it still cannot sample while gated. The dispatch gate is
unchanged and covered by ``test_aux_preparation_line_skip_liveness``; this file is about preload only.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordePreloadInferenceModelMessage,
    HordeProcessState,
    ModelLoadState,
)
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
_GATED_MODEL = "gated-lora-model"
_READY_MODEL = "ready-model"
_OTHER_MODEL = "unrelated-resident-model"


def _job(model: str, *, lora: str | None = None) -> ImageGenerateJobPopResponse:
    loras = None if lora is None else [LorasPayloadEntry(name=lora, model=1.0, clip=1.0, is_version=True)]
    return make_job_pop_response(model, width=512, height=512, ddim_steps=20, loras=loras)


def _scheduler(
    process_map: ProcessMap,
    tracker: JobTracker,
    model_map: HordeModelMap,
    *,
    device_free_mb: float = 24_000.0,
) -> InferenceScheduler:
    bridge = make_mock_bridge_data(
        max_threads=1,
        image_models_to_load=[_GATED_MODEL, _READY_MODEL, _OTHER_MODEL],
    )
    reference = {
        name: make_mock_model_reference_record(name, baseline=_SDXL)
        for name in (_GATED_MODEL, _READY_MODEL, _OTHER_MODEL)
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
        max_concurrent_inference_processes=1,
        max_inference_processes=2,
        lru=LRUCache(2),
    )
    scheduler.set_device_free_mb_provider(lambda _device_index: device_free_mb)
    return scheduler


def _empty_slot(process_id: int) -> HordeProcessInfo:
    """An inference slot holding no model and waiting for work."""
    process = make_mock_process_info(process_id, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    process.total_vram_mb = 16_375
    process.process_reserved_mb = 1_372
    return process


def _preloaded_model_name(process: HordeProcessInfo) -> str | None:
    """The model named by the last preload message sent to *process*, or None if none was sent."""
    for call in process.pipe_connection.send.call_args_list:  # type: ignore[attr-defined]  # Mock pipe in tests
        message = call.args[0]
        if isinstance(message, HordePreloadInferenceModelMessage):
            return message.horde_model_name
    return None


class TestAuxGatedPreloadOverlaps:
    """The checkpoint load runs alongside the auxiliary download when the slot is free."""

    async def test_lone_gated_job_preloads_into_empty_slot(self) -> None:
        """With nothing else queued, a gated job's model is staged rather than left until its files land."""
        gated = _job(_GATED_MODEL, lora="still-downloading")
        slot = _empty_slot(0)
        tracker = JobTracker()
        await track_popped_job_async(tracker, gated)
        scheduler = _scheduler(ProcessMap({0: slot}), tracker, HordeModelMap(root={}))

        assert scheduler.preload_models() is True
        assert _preloaded_model_name(slot) == _GATED_MODEL

    async def test_gated_job_still_cannot_sample_while_its_files_download(self) -> None:
        """Staging weights is not dispatch: the job holds no lane and never starts inference while gated."""
        gated = _job(_GATED_MODEL, lora="still-downloading")
        slot = _empty_slot(0)
        tracker = JobTracker()
        await track_popped_job_async(tracker, gated)
        scheduler = _scheduler(ProcessMap({0: slot}), tracker, HordeModelMap(root={}))

        scheduler.preload_models()
        slot.last_process_state = HordeProcessState.PRELOADED_MODEL
        slot.loaded_horde_model_name = _GATED_MODEL

        assert await scheduler.start_inference() is False
        assert gated in tracker.jobs_pending_inference
        assert gated not in tracker.jobs_in_progress
        assert slot.last_control_flag is not HordeControlFlag.START_INFERENCE

    async def test_preloaded_gated_job_dispatches_once_its_files_land(self) -> None:
        """The staged model is the one the job then samples on, so the overlap is not wasted work."""
        gated = _job(_GATED_MODEL, lora="still-downloading")
        slot = _empty_slot(0)
        tracker = JobTracker()
        await track_popped_job_async(tracker, gated)
        scheduler = _scheduler(ProcessMap({0: slot}), tracker, HordeModelMap(root={}))

        assert scheduler.preload_models() is True
        slot.last_process_state = HordeProcessState.PRELOADED_MODEL
        slot.loaded_horde_model_name = _GATED_MODEL
        mark_job_aux_prepared(tracker, gated)

        assert await scheduler.start_inference() is True
        assert gated in tracker.jobs_in_progress
        assert slot.last_control_flag is HordeControlFlag.START_INFERENCE


class TestAuxGatedPreloadRestraint:
    """A gated job never takes capacity from work that can actually run."""

    async def test_dispatchable_sibling_gets_the_slot_first(self) -> None:
        """A pending job that can sample outranks a gated one, so the gated model is not staged."""
        gated = _job(_GATED_MODEL, lora="still-downloading")
        ready = _job(_READY_MODEL)
        slot = _empty_slot(0)
        tracker = JobTracker()
        await track_popped_job_async(tracker, gated)
        await track_popped_job_async(tracker, ready)
        scheduler = _scheduler(ProcessMap({0: slot}), tracker, HordeModelMap(root={}))

        assert scheduler.preload_models() is True
        assert _preloaded_model_name(slot) == _READY_MODEL

    async def test_occupied_slot_is_never_displaced_for_a_gated_job(self) -> None:
        """A resident model is not thrown away to stage a job that cannot sample yet."""
        gated = _job(_GATED_MODEL, lora="still-downloading")
        slot = make_mock_process_info(0, model_name=_OTHER_MODEL, state=HordeProcessState.WAITING_FOR_JOB)
        slot.total_vram_mb = 16_375
        slot.process_reserved_mb = 1_372
        model_map = HordeModelMap(root={})
        model_map.update_entry(_OTHER_MODEL, load_state=ModelLoadState.LOADED_IN_RAM, process_id=0)
        tracker = JobTracker()
        await track_popped_job_async(tracker, gated)
        scheduler = _scheduler(ProcessMap({0: slot}), tracker, model_map)

        assert scheduler.preload_models() is False
        assert _preloaded_model_name(slot) is None
        assert slot.loaded_horde_model_name == _OTHER_MODEL

    async def test_ungated_job_preloads_exactly_as_before(self) -> None:
        """The control: a job needing no auxiliary files is unaffected by the overlap allowance."""
        ready = _job(_READY_MODEL)
        slot = _empty_slot(0)
        tracker = JobTracker()
        await track_popped_job_async(tracker, ready)
        scheduler = _scheduler(ProcessMap({0: slot}), tracker, HordeModelMap(root={}))

        assert scheduler.preload_models() is True
        assert _preloaded_model_name(slot) == _READY_MODEL
