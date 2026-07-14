"""Regression coverage for LoRA-heavy queues that have idle inference capacity.

When the local queue contains only LoRA work, the worker must request a small non-LoRA job and admit it
through the normal queue-depth limit so an idle inference process can keep the GPU active. The idle-fill
ladder broadens that request across attempts until eligible work is found.
"""

from __future__ import annotations

from collections.abc import Iterable
from unittest.mock import AsyncMock, Mock

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.utils.job_utils import small_pop_max_power
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
)
from tests.process_management.jobs.test_job_popping import _make_popper

_SD15 = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1
_SDXL = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl


def _lora_job(model: str, names: Iterable[str]) -> ImageGenerateJobPopResponse:
    """Build a modest image job whose listed LoRAs are not assumed to be cached."""
    loras = [LorasPayloadEntry(name=name, model=1.0, clip=1.0, is_version=True) for name in names]
    return make_job_pop_response(model, width=512, height=512, ddim_steps=20, loras=loras)


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
        small = small_pop_max_power(high_performance_mode=False, moderate_performance_mode=False)
        assert [(set(request.models), request.max_pixels) for request in requests] == [
            ({"light-model"}, small * 8 * 64 * 64),
            ({"light-model"}, 128 * 8 * 64 * 64),
            ({"heavy-model"}, small * 8 * 64 * 64),
            ({"heavy-model"}, 128 * 8 * 64 * 64),
        ]
        assert all(request.allow_lora is False for request in requests)
