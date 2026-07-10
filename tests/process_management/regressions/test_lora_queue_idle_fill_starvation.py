"""Regression coverage for LoRA-heavy queues that have idle inference capacity.

Auxiliary downloads occupy job-concurrency slots without using the sampler.  When the local queue contains
only LoRA work whose auxiliary files are not cached, the worker must request a small non-LoRA job and admit it
through the normal queue-depth limit so an idle inference process can keep the GPU active.  The same liveness
rule applies below the job-concurrency cap: download-only work must not suppress the idle-fill starvation clock.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import cast
from unittest.mock import AsyncMock, Mock

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopRequest, ImageGenerateJobPopResponse, LorasPayloadEntry

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.utils.job_utils import line_skip_pop_max_power
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    mark_job_in_progress_async,
    track_popped_job_async,
)
from tests.process_management.jobs.test_job_popping import _make_popper
from tests.process_management.regressions.test_lora_aux_download_lane_starvation_repro import _make_scheduler

_SD15 = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1
_SDXL = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl


def _lora_job(model: str, names: Iterable[str]) -> ImageGenerateJobPopResponse:
    """Build a modest image job whose listed LoRAs are not assumed to be cached."""
    loras = [LorasPayloadEntry(name=name, model=1.0, clip=1.0, is_version=True) for name in names]
    return make_job_pop_response(model, width=512, height=512, ddim_steps=20, loras=loras)


async def _download_saturated_queue() -> tuple[
    WorkerState,
    JobTracker,
    ProcessMap,
    dict[str, ImageGenerationModelRecord],
]:
    """Build two active auxiliary downloads, one queued LoRA job, and one idle inference process."""
    first = _lora_job("download-a", ["lora-a"])
    second = _lora_job("download-b", ["lora-b"])
    queued = _lora_job("queued-c", ["lora-c"])

    tracker = JobTracker()
    for job in (first, second):
        await track_popped_job_async(tracker, job)
        await mark_job_in_progress_async(tracker, job)
    await track_popped_job_async(tracker, queued)
    await tracker.increment_jobs_completed()

    first_process = make_mock_process_info(
        0,
        model_name="download-a",
        state=HordeProcessState.DOWNLOADING_AUX_MODEL,
    )
    first_process.last_job_referenced = first
    first_process.last_received_timestamp = time.time() - 10.0
    second_process = make_mock_process_info(
        1,
        model_name="download-b",
        state=HordeProcessState.DOWNLOADING_AUX_MODEL,
    )
    second_process.last_job_referenced = second
    second_process.last_received_timestamp = time.time() - 10.0
    idle_process = make_mock_process_info(2, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    safety_process = make_mock_process_info(
        10,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.SAFETY,
    )
    process_map = ProcessMap({0: first_process, 1: second_process, 2: idle_process, 10: safety_process})
    reference = {
        model: make_mock_model_reference_record(model, baseline=_SDXL)
        for model in ("download-a", "download-b", "queued-c", "fill-d")
    }
    return WorkerState(), tracker, process_map, reference


class TestLoraIntakeCeiling:
    """The LoRA intake ceiling reserves local capacity independently of the process-pool width."""

    async def test_two_lora_jobs_reach_the_ceiling_on_a_wide_process_pool(self) -> None:
        """Two accepted LoRA jobs stop further LoRA intake even when more inference processes exist."""
        tracker = JobTracker()
        popper = _make_popper(job_tracker=tracker, max_inference_processes=6)
        await popper._enqueue_popped_job(_lora_job("model-a", ["lora-a"]))
        await popper._enqueue_popped_job(_lora_job("model-b", ["lora-b"]))

        assert popper._lora_queue_cap_reached() is True

    async def test_small_process_pool_retains_one_nonlora_slot(self) -> None:
        """The absolute ceiling does not weaken the existing N-1 reserve on a two-process pool."""
        tracker = JobTracker()
        popper = _make_popper(job_tracker=tracker, max_inference_processes=2)
        await popper._enqueue_popped_job(_lora_job("model-a", ["lora-a"]))

        assert popper._lora_queue_cap_reached() is True


class TestDownloadOnlyConcurrencyLiveness:
    """Download-only jobs must trigger an urgent non-LoRA pop while an inference process is idle."""

    async def test_active_download_blocker_arms_line_skip_when_queued_loras_cannot_run(self) -> None:
        """A saturated concurrency budget arms line-skip when every queued candidate needs a download."""
        state, tracker, process_map, reference = await _download_saturated_queue()
        scheduler = _make_scheduler(
            process_map=process_map,
            job_tracker=tracker,
            reference=cast(dict[str, object], reference),
            max_concurrent=2,
            state=state,
        )

        selected = await scheduler.get_next_job_and_process()

        assert selected is None
        assert state.wants_line_skip_candidate is True

    async def test_active_download_blocker_respects_the_patience_threshold(self) -> None:
        """Fresh auxiliary downloads do not trigger urgent intake before the configured patience elapses."""
        state, tracker, process_map, reference = await _download_saturated_queue()
        process_map[0].last_received_timestamp = time.time()
        process_map[1].last_received_timestamp = time.time()
        scheduler = _make_scheduler(
            process_map=process_map,
            job_tracker=tracker,
            reference=cast(dict[str, object], reference),
            max_concurrent=2,
            state=state,
        )

        assert await scheduler.get_next_job_and_process() is None
        assert state.wants_line_skip_candidate is False

    async def test_active_download_blocker_respects_a_disabled_line_skip_breaker(self) -> None:
        """A disabled auxiliary line-skip threshold never arms the urgent-pop state."""
        state, tracker, process_map, reference = await _download_saturated_queue()
        scheduler = _make_scheduler(
            process_map=process_map,
            job_tracker=tracker,
            reference=cast(dict[str, object], reference),
            max_concurrent=2,
            state=state,
        )
        scheduler._runtime_config.bridge_data.aux_model_download_line_skip_threshold_seconds = None

        assert await scheduler.get_next_job_and_process() is None
        assert state.wants_line_skip_candidate is False

    async def test_saturated_lora_queue_overpops_a_nonlora_fill_job(self) -> None:
        """The scheduler and popper together request runnable work through a full LoRA queue."""
        state, tracker, process_map, reference = await _download_saturated_queue()
        bridge = make_mock_bridge_data(
            allow_lora=True,
            max_threads=2,
            queue_size=1,
            image_models_to_load=["download-a", "download-b", "queued-c", "fill-d"],
        )
        scheduler = _make_scheduler(
            process_map=process_map,
            job_tracker=tracker,
            reference=cast(dict[str, object], reference),
            max_concurrent=2,
            state=state,
        )
        assert await scheduler.get_next_job_and_process() is None

        session = Mock()
        session.submit_request = AsyncMock(return_value=ImageGenerateJobPopResponse(id=None, ids=[], payload={}))
        popper = _make_popper(
            state=state,
            process_map=process_map,
            job_tracker=tracker,
            bridge_data=bridge,
            horde_client_session=session,
            max_inference_processes=3,
            max_concurrent_inference_processes=2,
        )

        await popper.api_job_pop()

        session.submit_request.assert_awaited_once()
        request: ImageGenerateJobPopRequest = session.submit_request.call_args.args[0]
        assert request.allow_lora is False
        expected_max_power = min(
            bridge.max_power,
            line_skip_pop_max_power(
                high_performance_mode=False,
                moderate_performance_mode=False,
            ),
        )
        assert request.max_pixels == expected_max_power * 8 * 64 * 64

    async def test_download_only_work_does_not_suppress_idle_fill_below_concurrency_cap(self) -> None:
        """Below the concurrency cap, an auxiliary download still allows the idle-fill clock to mature."""
        active = _lora_job("download-a", ["lora-a"])
        queued = _lora_job("queued-b", ["lora-b"])
        tracker = JobTracker()
        await track_popped_job_async(tracker, active)
        await mark_job_in_progress_async(tracker, active)
        await track_popped_job_async(tracker, queued)

        downloading = make_mock_process_info(
            0,
            model_name="download-a",
            state=HordeProcessState.DOWNLOADING_AUX_MODEL,
        )
        downloading.last_job_referenced = active
        idle = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        state = WorkerState()
        scheduler = _make_scheduler(
            process_map=ProcessMap({0: downloading, 1: idle}),
            job_tracker=tracker,
            reference={
                "download-a": make_mock_model_reference_record("download-a", baseline=_SDXL),
                "queued-b": make_mock_model_reference_record("queued-b", baseline=_SDXL),
            },
            max_concurrent=2,
            state=state,
        )
        bridge = scheduler._runtime_config.bridge_data
        bridge.idle_fill_threshold_seconds = 1
        scheduler._head_starvation_job_id = str(queued.id_)
        scheduler._head_starvation_since = time.time() - 10.0

        scheduler._update_head_starvation_timer(queued)
        scheduler._update_idle_fill_arm(bridge)

        assert state.wants_idle_fill_candidate is True

    async def test_sampling_work_still_suppresses_idle_fill_alongside_an_aux_download(self) -> None:
        """A real sampler keeps the overlap safety fence closed even when another job is download-only."""
        downloading_job = _lora_job("download-a", ["lora-a"])
        sampling_job = make_job_pop_response("sampling-b")
        queued_job = make_job_pop_response("queued-c")
        tracker = JobTracker()
        for active_job in (downloading_job, sampling_job):
            await track_popped_job_async(tracker, active_job)
            await mark_job_in_progress_async(tracker, active_job)
        await track_popped_job_async(tracker, queued_job)

        downloading_process = make_mock_process_info(
            0,
            model_name="download-a",
            state=HordeProcessState.DOWNLOADING_AUX_MODEL,
        )
        downloading_process.last_job_referenced = downloading_job
        sampling_process = make_mock_process_info(
            1,
            model_name="sampling-b",
            state=HordeProcessState.INFERENCE_STARTING,
        )
        sampling_process.last_job_referenced = sampling_job
        idle_process = make_mock_process_info(2, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        state = WorkerState()
        scheduler = _make_scheduler(
            process_map=ProcessMap({0: downloading_process, 1: sampling_process, 2: idle_process}),
            job_tracker=tracker,
            reference={
                model: make_mock_model_reference_record(model, baseline=_SDXL)
                for model in ("download-a", "sampling-b", "queued-c")
            },
            max_concurrent=3,
            state=state,
        )
        scheduler._head_starvation_job_id = str(queued_job.id_)
        scheduler._head_starvation_since = time.time() - 10.0

        scheduler._update_head_starvation_timer(queued_job)
        scheduler._update_idle_fill_arm(scheduler._runtime_config.bridge_data)

        assert scheduler._head_starvation_since == 0.0
        assert state.wants_idle_fill_candidate is False

    async def test_prearmed_line_skip_is_an_effective_liveness_control(self) -> None:
        """Control: once armed, the urgent pop bypasses queue depth and excludes fresh LoRA work."""
        state, tracker, process_map, _reference = await _download_saturated_queue()
        state.wants_line_skip_candidate = True
        bridge = make_mock_bridge_data(
            allow_lora=True,
            max_threads=2,
            queue_size=1,
            image_models_to_load=["download-a", "download-b", "queued-c", "fill-d"],
        )
        session = Mock()
        session.submit_request = AsyncMock(return_value=ImageGenerateJobPopResponse(id=None, ids=[], payload={}))
        popper = _make_popper(
            state=state,
            process_map=process_map,
            job_tracker=tracker,
            bridge_data=bridge,
            horde_client_session=session,
            max_inference_processes=3,
            max_concurrent_inference_processes=2,
        )

        await popper.api_job_pop()

        request: ImageGenerateJobPopRequest = session.submit_request.call_args.args[0]
        assert request.allow_lora is False


class _ModelMetadata:
    """Minimal baseline lookup for an end-to-end idle-fill ladder control."""

    def __init__(self) -> None:
        self._baselines = {"light-model": _SD15, "heavy-model": _SDXL}

    def get_baseline(self, model_name: str) -> KNOWN_IMAGE_GENERATION_BASELINE | None:
        return self._baselines.get(model_name)


class TestIdleFillLadderProgression:
    """An armed fill request broadens promptly until eligible work is found."""

    async def test_consecutive_empty_responses_offer_every_rung_without_pacing_delay(self) -> None:
        """Four urgent attempts progress from light-small through heavy-large on consecutive ticks."""
        state = WorkerState(wants_idle_fill_candidate=True)
        bridge = make_mock_bridge_data(
            allow_lora=True,
            max_power=128,
            image_models_to_load=["light-model", "heavy-model"],
        )
        session = Mock()
        session.submit_request = AsyncMock(
            side_effect=[ImageGenerateJobPopResponse(id=None, ids=[], payload={}) for _ in range(4)],
        )
        process_map = ProcessMap(
            {
                0: make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB),
                10: make_mock_process_info(
                    10,
                    model_name=None,
                    state=HordeProcessState.WAITING_FOR_JOB,
                    process_type=HordeProcessType.SAFETY,
                ),
            },
        )
        popper = _make_popper(
            state=state,
            process_map=process_map,
            bridge_data=bridge,
            horde_client_session=session,
        )
        popper._model_metadata = _ModelMetadata()  # type: ignore[assignment]

        for _ in range(4):
            await popper.api_job_pop()

        requests = [call.args[0] for call in session.submit_request.await_args_list]
        small = line_skip_pop_max_power(high_performance_mode=False, moderate_performance_mode=False)
        assert [(set(request.models), request.max_pixels) for request in requests] == [
            ({"light-model"}, small * 8 * 64 * 64),
            ({"light-model"}, 128 * 8 * 64 * 64),
            ({"heavy-model"}, small * 8 * 64 * 64),
            ({"heavy-model"}, 128 * 8 * 64 * 64),
        ]
        assert all(request.allow_lora is False for request in requests)
