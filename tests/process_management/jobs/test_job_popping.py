"""Tests for JobPopper orchestration logic.

Tests for individual extracted components (PopThrottler, SourceImageDownloader,
_select_models_for_pop, APIWorkerMessage) live in their own test modules.
These tests focus on how JobPopper coordinates those components and the
higher-level api_job_pop flow.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from unittest.mock import AsyncMock, Mock, patch

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateJobPopRequest,
    ImageGenerateJobPopResponse,
    LorasPayloadEntry,
)
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from loguru import logger

from horde_worker_regen.bridge_data.data_model import ModelPoolConfig
from horde_worker_regen.process_management.config.worker_state import RecoveryParkReason, WorkerState
from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadPhase,
    DownloadStatusSnapshot,
)
from horde_worker_regen.process_management.jobs.job_popper import JobPopper
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.jobs.pool_lanes import PoolLaneState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.model_availability import ModelAvailability
from horde_worker_regen.process_management.process_manager import POST_PROCESSING_GATE_OPEN_REQUIREMENT_MB
from horde_worker_regen.process_management.scheduling.model_pool import PopLane
from horde_worker_regen.process_management.scheduling.pop_throttler import CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS
from horde_worker_regen.process_management.simulation._canned_scenarios import CannedJobSource, make_empty_pop_response
from horde_worker_regen.utils.job_utils import small_pop_max_power
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_job,
    make_mock_process_info,
    make_test_api_sessions,
    make_test_card_runtimes,
    make_test_runtime_config,
    make_testable_process_manager,
    track_popped_job_async,
)


def _make_popper(
    *,
    state: WorkerState | None = None,
    process_map: ProcessMap | None = None,
    job_tracker: JobTracker | None = None,
    bridge_data: Mock | None = None,
    horde_client_session: object | None = None,
    aiohttp_session: object | None = None,
    shutdown_manager: Mock | None = None,
    max_inference_processes: int = 2,
    max_concurrent_inference_processes: int = 1,
    image_models_to_load: list[str] | None = None,
    dry_run_skip_api: bool = False,
    model_availability: ModelAvailability | None = None,
    post_processing_lane_commitments_provider: Callable[[], int] | None = None,
    extended_controlnet_ready_provider: Callable[[], bool] | None = None,
    post_processing_lane_paused_provider: Callable[[], bool] | None = None,
    vram_pressure_provider: Callable[[], bool] | None = None,
    pool_active_seats_provider: Callable[[], frozenset[str]] | None = None,
    pool_pop_outcome_sink: Callable[..., None] | None = None,
    card_runtimes: dict[int, CardRuntime] | None = None,
) -> JobPopper:
    """Build a JobPopper with mostly-mocked dependencies."""
    if state is None:
        state = WorkerState()
    if process_map is None:
        process_map = ProcessMap({})
    if job_tracker is None:
        job_tracker = JobTracker()
    if bridge_data is None:
        kwargs: dict = {}  # pyrefly: ignore - type inference is not useful in this test
        if image_models_to_load is not None:
            kwargs["image_models_to_load"] = image_models_to_load
        bridge_data = make_mock_bridge_data(**kwargs)
    if horde_client_session is None:
        horde_client_session = Mock()
    if aiohttp_session is None:
        aiohttp_session = Mock()
    if shutdown_manager is None:
        shutdown_manager = Mock()

    return JobPopper(
        state=state,
        process_map=process_map,
        job_tracker=job_tracker,
        shutdown_manager=shutdown_manager,
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        api_sessions=make_test_api_sessions(
            horde_client_session=horde_client_session,
            aiohttp_session=aiohttp_session,
        ),
        max_inference_processes=max_inference_processes,
        max_concurrent_inference_processes=max_concurrent_inference_processes,
        dry_run_skip_api=dry_run_skip_api,
        model_availability=model_availability,
        post_processing_lane_commitments_provider=post_processing_lane_commitments_provider,
        extended_controlnet_ready_provider=extended_controlnet_ready_provider,
        post_processing_lane_paused_provider=post_processing_lane_paused_provider,
        vram_pressure_provider=vram_pressure_provider,
        pool_active_seats_provider=pool_active_seats_provider,
        pool_pop_outcome_sink=pool_pop_outcome_sink,
        card_runtimes=card_runtimes,
    )


def _make_process_map_with_available_processes(*, num_safety: int = 1) -> ProcessMap:
    """Create a process map with an available inference process and ``num_safety`` safety processes."""
    procs: dict[int, object] = {
        0: make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.WAITING_FOR_JOB),
    }
    for i in range(num_safety):
        procs[10 + i] = make_mock_process_info(
            10 + i,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
    return ProcessMap(procs)  # type: ignore[arg-type]


async def _queue_n_jobs_for_safety(job_tracker: JobTracker, n: int) -> None:
    """Place ``n`` jobs into the post-inference safety backlog (PENDING_SAFETY_CHECK)."""
    for _ in range(n):
        job = Mock()
        job.id_ = uuid.uuid4()
        job.model = "stable_diffusion"
        job_info = Mock()
        job_info.sdk_api_job_info = job
        await job_tracker.queue_for_safety(job_info)


async def _queue_n_jobs_for_post_processing(job_tracker: JobTracker, n: int) -> None:
    """Place ``n`` jobs into the pending post-processing stage."""
    for _ in range(n):
        job = Mock()
        job.id_ = uuid.uuid4()
        job.model = "stable_diffusion"
        job.payload.post_processing = ["RealESRGAN_x4plus"]
        job_info = Mock()
        job_info.sdk_api_job_info = job
        await job_tracker.queue_for_post_processing(job_info)


async def _queue_n_popped_jobs_requesting_post_processing(job_tracker: JobTracker, n: int) -> None:
    """Place ``n`` accepted jobs that have requested post-processing but have not reached the lane."""
    for index in range(n):
        await job_tracker.record_popped_job(
            make_job_pop_response(model=f"accepted_pp_{index}", post_processing=["RealESRGAN_x4plus"], n_iter=8),
        )


async def _queue_n_popped_jobs_without_post_processing(job_tracker: JobTracker, n: int) -> None:
    """Place ``n`` accepted ordinary image jobs that do not need the post-processing lane."""
    for index in range(n):
        await job_tracker.record_popped_job(make_job_pop_response(model=f"accepted_plain_{index}"))


class TestApiJobPopGuardClauses:
    """Each guard clause in api_job_pop should short-circuit cleanly."""

    async def test_shutting_down_returns_early_and_clears_flag(self) -> None:
        """When shutting_down is True, pop exits immediately and clears last_pop_no_jobs."""
        state = WorkerState(shutting_down=True, last_pop_no_jobs_available=True)
        popper = _make_popper(state=state)

        await popper.api_job_pop()

        assert state.last_pop_no_jobs_available is False

    async def test_gpu_torch_incompatible_blocks_pop(self) -> None:
        """The sticky torch/GPU-incompatible flag stops popping even with a fully available process pool."""
        state = WorkerState(gpu_torch_incompatible=True, last_pop_no_jobs_available=True)
        popper = _make_popper(state=state, process_map=_make_process_map_with_available_processes())

        await popper.api_job_pop()

        assert state.last_pop_no_jobs_available is False
        assert state.gpu_torch_incompatible is True

    async def test_cpu_only_torch_build_blocks_image_pop(self) -> None:
        """A CPU-only torch build stops the image popper even with a fully available process pool.

        This is the runtime equivalent of a 'cpu' install sentinel: image generation is disabled while
        alchemy (a separate loop) keeps running.
        """
        state = WorkerState(torch_build_cpu_only=True, last_pop_no_jobs_available=True)
        popper = _make_popper(state=state, process_map=_make_process_map_with_available_processes())

        await popper.api_job_pop()

        assert state.last_pop_no_jobs_available is False
        assert state.torch_build_cpu_only is True

    async def test_recovery_park_blocks_pop(self) -> None:
        """A worker whose recovery escalation is parked takes no new work, even with an available pool.

        The park means every remedy the worker can apply in place is spent over a pool that cannot serve, so
        work accepted now would only be faulted back to the horde.
        """
        state = WorkerState(
            recovery_parked=True,
            recovery_park_reason=RecoveryParkReason.UNRECOVERABLE_POOL,
            last_pop_no_jobs_available=True,
        )
        session = Mock()
        session.submit_request = AsyncMock()
        popper = _make_popper(
            state=state,
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
        )

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()
        assert state.last_pop_no_jobs_available is False

    async def test_too_many_consecutive_failures_blocks_pop(self) -> None:
        """Active failure pause prevents any pop attempt."""
        state = WorkerState(
            too_many_consecutive_failed_jobs=True,
            too_many_consecutive_failed_jobs_time=time.time(),
        )
        popper = _make_popper(state=state)

        await popper.api_job_pop()

        # Still in failure state
        assert state.too_many_consecutive_failed_jobs is True

    async def test_consecutive_failure_pause_expires_and_resets(self) -> None:
        """After CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS, the pause should lift."""
        state = WorkerState(
            too_many_consecutive_failed_jobs=True,
            too_many_consecutive_failed_jobs_time=time.time() - CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS - 1,
            consecutive_failed_jobs=5,
        )
        popper = _make_popper(state=state)

        await popper.api_job_pop()

        assert state.too_many_consecutive_failed_jobs is False
        assert state.consecutive_failed_jobs == 0

    async def test_reaching_failure_threshold_activates_pause(self) -> None:
        """When consecutive_failed_jobs hits 3, pause should activate."""
        state = WorkerState(consecutive_failed_jobs=3)
        popper = _make_popper(state=state)

        await popper.api_job_pop()

        assert state.too_many_consecutive_failed_jobs is True
        assert state.too_many_consecutive_failed_jobs_time > 0

    async def test_failure_threshold_with_exit_on_faults_shuts_down(self) -> None:
        """When exit_on_unhandled_faults is True, reaching threshold triggers shutdown."""
        state = WorkerState(consecutive_failed_jobs=3)
        bd = make_mock_bridge_data(exit_on_unhandled_faults=True)
        shutdown_mgr = Mock()
        popper = _make_popper(state=state, bridge_data=bd, shutdown_manager=shutdown_mgr)

        await popper.api_job_pop()

        shutdown_mgr.shutdown.assert_called_once()

    async def test_full_queue_returns_early(self) -> None:
        """Queue at capacity should prevent further pops."""
        job_tracker = JobTracker()
        # Default bridge data: queue_size=1, max_threads=1 → max_jobs_in_queue = 2
        for _ in range(10):
            await track_popped_job_async(job_tracker, make_mock_job())

        popper = _make_popper(job_tracker=job_tracker)
        await popper.api_job_pop()

    async def test_no_safety_process_returns_early(self) -> None:
        """Without an available safety process, pop should not proceed."""
        popper = _make_popper(process_map=ProcessMap({}))
        await popper.api_job_pop()

    async def test_no_inference_process_returns_early(self) -> None:
        """With safety but no inference process, pop should not proceed."""
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        popper = _make_popper(process_map=ProcessMap({10: safety_proc}))
        await popper.api_job_pop()

    async def test_no_models_configured_returns_early(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty model list should prevent pops (with a sleep penalty)."""
        sleep = AsyncMock()
        monkeypatch.setattr("horde_worker_regen.process_management.jobs.job_popper.asyncio.sleep", sleep)
        process_map = _make_process_map_with_available_processes()
        popper = _make_popper(process_map=process_map, image_models_to_load=[])
        await popper.api_job_pop()
        sleep.assert_awaited_once_with(3)

    async def test_too_frequent_pop_returns_early(self) -> None:
        """Popping again within the throttle window should be skipped."""
        state = WorkerState(last_job_pop_time=time.time())
        process_map = _make_process_map_with_available_processes()
        popper = _make_popper(state=state, process_map=process_map)
        await popper.api_job_pop()

    async def test_no_completed_session_jobs_blocks_queue_ahead(self) -> None:
        """Until the first job of the session completes, a second pop must not happen.

        This is the warm-up rule: if we're doomed to fail with 1 job, we're
        doomed to fail with 2 jobs.
        """
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())
        assert job_tracker.total_num_completed_jobs == 0

        session = Mock()
        session.submit_request = AsyncMock()
        popper = _make_popper(
            job_tracker=job_tracker,
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
        )

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_completed_session_job_allows_queue_ahead(self) -> None:
        """Once any job has completed this session, queue-ahead pops are allowed.

        Regression test: the warm-up gate must not block whenever nothing
        happens to be pending submit; it only applies before the first
        completion of the session.
        """
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())
        await job_tracker.increment_jobs_completed()

        session = Mock()
        session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="test error"))
        popper = _make_popper(
            job_tracker=job_tracker,
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
        )

        await popper.api_job_pop()

        session.submit_request.assert_awaited_once()


class TestFeatureReadinessGate:
    """A pop withholds a gated feature until its models/annotators are on disk (first-class readiness)."""

    @staticmethod
    async def _pop_and_capture_request(availability: ModelAvailability, **bridge_overrides: object) -> object:
        """Drive one full pop with the given availability and return the built job-pop request."""
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())
        await job_tracker.increment_jobs_completed()  # clear the session warm-up gate so a pop happens
        session = Mock()
        session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
        popper = _make_popper(
            job_tracker=job_tracker,
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
            bridge_data=make_mock_bridge_data(**bridge_overrides),
            model_availability=availability,
        )

        await popper.api_job_pop()

        session.submit_request.assert_awaited_once()
        return session.submit_request.call_args.args[0]

    async def test_controlnet_withheld_while_its_models_download(self) -> None:
        """ControlNet is enabled but its models are not yet on disk, so the pop must not advertise it.

        Post-processing, whose models are present, is still advertised in the same pop, proving the gate
        is per-feature rather than an all-or-nothing switch.
        """
        availability = ModelAvailability()
        availability.update(
            present={"stable_diffusion"},
            currently_downloading=None,
            pending=(),
            failed=(),
            controlnet_present=False,
            post_processing_present=True,
        )

        request = await self._pop_and_capture_request(
            availability,
            allow_controlnet=True,
            allow_post_processing=True,
            allow_sdxl_controlnet=False,
        )

        assert request.allow_controlnet is False
        assert request.allow_post_processing is True

    async def test_controlnet_offered_once_its_models_are_present(self) -> None:
        """Once ControlNet's models are reported on disk, the pop advertises it again."""
        availability = ModelAvailability()
        availability.update(
            present={"stable_diffusion"},
            currently_downloading=None,
            pending=(),
            failed=(),
            controlnet_present=True,
            post_processing_present=True,
        )

        request = await self._pop_and_capture_request(availability, allow_controlnet=True)

        assert request.allow_controlnet is True

    async def test_unknown_presence_does_not_withhold(self) -> None:
        """With no presence reported yet (None), an enabled feature is advertised, as before readiness."""
        availability = ModelAvailability()
        availability.update(
            present={"stable_diffusion"},
            currently_downloading=None,
            pending=(),
            failed=(),
            controlnet_present=None,
        )

        request = await self._pop_and_capture_request(availability, allow_controlnet=True)

        assert request.allow_controlnet is True


class TestHeterogeneousCardPopRotation:
    """Each heterogeneous offer stays within one card's routeable capability rectangle."""

    async def test_successive_pops_rotate_complete_card_scoped_offers(self) -> None:
        """Model, feature, resolution, and capacity fields rotate together by card; nsfw stays fleet-wide."""
        plain_config = make_mock_bridge_data(
            image_models_to_load=["plain-model"],
            allow_controlnet=False,
            allow_post_processing=False,
            allow_lora=False,
            max_power=2,
            max_threads=1,
            nsfw=False,
        )
        feature_config = make_mock_bridge_data(
            image_models_to_load=["feature-model"],
            allow_controlnet=True,
            allow_post_processing=True,
            allow_lora=True,
            max_power=8,
            max_threads=2,
            nsfw=True,
        )
        card_runtimes = {
            0: make_test_card_runtimes(
                device_indices=(0,),
                max_concurrent_inference=1,
                config=plain_config,
            )[0],
            1: make_test_card_runtimes(
                device_indices=(1,),
                max_concurrent_inference=2,
                config=feature_config,
            )[1],
        }
        session = Mock()
        session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
        popper = _make_popper(
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
            bridge_data=make_mock_bridge_data(image_models_to_load=["plain-model", "feature-model"]),
            card_runtimes=card_runtimes,
        )

        for _ in range(4):
            popper._state.last_pop_no_jobs_available = False
            await popper.api_job_pop(urgent=True)

        requests = [call.args[0] for call in session.submit_request.call_args_list]
        assert [set(request.models) for request in requests] == [
            {"plain-model"},
            {"feature-model"},
            {"plain-model"},
            {"feature-model"},
        ]
        assert [request.allow_controlnet for request in requests] == [False, True, False, True]
        assert [request.allow_post_processing for request in requests] == [False, True, False, True]
        assert [request.allow_lora for request in requests] == [False, True, False, True]
        # nsfw does not rotate: a popped job is not pinned to the offering card and carries no NSFW marker,
        # so a fleet with any SFW card offers SFW work only.
        assert [request.nsfw for request in requests] == [False, False, False, False]
        assert [request.max_pixels for request in requests] == [2 * 8 * 64 * 64, 8 * 8 * 64 * 64] * 2
        assert [request.threads for request in requests] == [1, 2, 1, 2]

    async def test_requested_amount_follows_the_scoped_card(self) -> None:
        """The requested batch is the offered card's own ceiling, never the global or the other card's."""
        small_batch = make_mock_bridge_data(image_models_to_load=["plain-model"], max_batch=1)
        large_batch = make_mock_bridge_data(image_models_to_load=["feature-model"], max_batch=6)
        card_runtimes = {
            0: make_test_card_runtimes(device_indices=(0,), config=small_batch)[0],
            1: make_test_card_runtimes(device_indices=(1,), config=large_batch)[1],
        }
        session = Mock()
        session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
        popper = _make_popper(
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
            bridge_data=make_mock_bridge_data(
                image_models_to_load=["plain-model", "feature-model"],
                max_batch=3,
            ),
            card_runtimes=card_runtimes,
        )

        for _ in range(2):
            popper._state.last_pop_no_jobs_available = False
            await popper.api_job_pop(urgent=True)

        requests = [call.args[0] for call in session.submit_request.call_args_list]
        assert [request.amount for request in requests] == [1, 6]


_SERVER_SUPPORTS_EXTENDED_CONTROLNET = (
    "horde_worker_regen.process_management.jobs.job_popper.server_supports_extended_controlnet"
)


class TestExtendedControlnetOffer:
    """allow_extended_controlnet ANDs the operator flag, live annotator readiness, and the server probe."""

    @staticmethod
    def _ready_availability() -> ModelAvailability:
        availability = ModelAvailability()
        availability.update(
            present={"stable_diffusion"},
            currently_downloading=None,
            pending=(),
            failed=(),
            controlnet_present=True,
            post_processing_present=True,
        )
        return availability

    @classmethod
    async def _pop_and_capture_request(
        cls,
        *,
        extended_controlnet: bool,
        ready: bool,
        server_supports: bool = True,
    ) -> object:
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())
        await job_tracker.increment_jobs_completed()
        session = Mock()
        session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
        popper = _make_popper(
            job_tracker=job_tracker,
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
            bridge_data=make_mock_bridge_data(
                extended_controlnet=extended_controlnet,
                allow_controlnet=True,
            ),
            model_availability=cls._ready_availability(),
            extended_controlnet_ready_provider=lambda: ready,
        )
        with patch(_SERVER_SUPPORTS_EXTENDED_CONTROLNET, return_value=server_supports):
            await popper.api_job_pop()
        session.submit_request.assert_awaited_once()
        return session.submit_request.call_args.args[0]

    async def test_flag_off_sends_false_even_when_ready(self) -> None:
        """The operator opt-out withholds extended regardless of annotator readiness."""
        request = await self._pop_and_capture_request(extended_controlnet=False, ready=True)
        assert request.allow_extended_controlnet is False

    async def test_flag_on_but_not_ready_sends_false(self) -> None:
        """Opted in but the annotators are not yet servable: the offer stays fail-closed."""
        request = await self._pop_and_capture_request(extended_controlnet=True, ready=False)
        assert request.allow_extended_controlnet is False

    async def test_flag_on_and_ready_sends_true(self) -> None:
        """Opted in with servable annotators and a server that proves support: extended is advertised."""
        request = await self._pop_and_capture_request(extended_controlnet=True, ready=True)
        assert request.allow_extended_controlnet is True
        # Extended implies the classic set: a worker offering extended always also offers classic controlnet.
        assert request.allow_controlnet is True

    async def test_server_without_support_clamps_even_when_flagged_and_ready(self) -> None:
        """The server-capability gate withholds extended even with the flag on and annotators servable."""
        request = await self._pop_and_capture_request(
            extended_controlnet=True,
            ready=True,
            server_supports=False,
        )
        assert request.allow_extended_controlnet is False
        # The classic offer is unaffected by the extended-controlnet server gate.
        assert request.allow_controlnet is True

    async def test_readiness_flip_between_pops_changes_offer_without_restart(self) -> None:
        """A False->True annotator-readiness flip changes the sent value on the next pop, no restart."""
        ready = {"value": False}
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())
        await job_tracker.increment_jobs_completed()
        session = Mock()
        popper = _make_popper(
            job_tracker=job_tracker,
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
            bridge_data=make_mock_bridge_data(extended_controlnet=True, allow_controlnet=True),
            model_availability=self._ready_availability(),
            extended_controlnet_ready_provider=lambda: ready["value"],
        )

        with patch(_SERVER_SUPPORTS_EXTENDED_CONTROLNET, return_value=True):
            session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
            popper._state.last_pop_no_jobs_available = False
            await popper.api_job_pop(urgent=True)
            first_request = session.submit_request.call_args.args[0]
            assert first_request.allow_extended_controlnet is False

            ready["value"] = True
            session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
            popper._state.last_pop_no_jobs_available = False
            await popper.api_job_pop(urgent=True)
            second_request = session.submit_request.call_args.args[0]
            assert second_request.allow_extended_controlnet is True


class TestPostProcessingBreakerSuppression:
    """A latched post-processing fault breaker withholds post-processing from the pop request."""

    @staticmethod
    async def _pop_and_capture_request(*, state: WorkerState) -> object:
        """Drive one full pop with the given worker state and return the built job-pop request."""
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())
        await job_tracker.increment_jobs_completed()  # clear the session warm-up gate so a pop happens
        session = Mock()
        session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
        popper = _make_popper(
            state=state,
            job_tracker=job_tracker,
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
        )

        await popper.api_job_pop()

        session.submit_request.assert_awaited_once()
        return session.submit_request.call_args.args[0]

    async def test_latched_breaker_withholds_post_processing(self) -> None:
        """With the breaker latched, the pop advertises ``allow_post_processing=False``."""
        state = WorkerState()
        state.post_processing_disabled_by_breaker = True

        request = await self._pop_and_capture_request(state=state)

        assert request.allow_post_processing is False

    async def test_unlatched_breaker_advertises_post_processing(self) -> None:
        """With the breaker not latched, post-processing is advertised as configured (the default path)."""
        request = await self._pop_and_capture_request(state=WorkerState())

        assert request.allow_post_processing is True

    async def test_headroom_gate_withholds_post_processing(self) -> None:
        """With the proactive headroom gate closed, the pop advertises ``allow_post_processing=False``."""
        state = WorkerState()
        state.post_processing_withheld_for_headroom = True

        request = await self._pop_and_capture_request(state=state)

        assert request.allow_post_processing is False


class TestPausedPostProcessingLaneSuppression:
    """A post-processing lane held off the GPU stops being advertised, and resumes when it returns."""

    @staticmethod
    async def _pop_and_capture_request(*, lane_paused: bool) -> object:
        """Drive one full pop with the lane paused or up and return the built job-pop request."""
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())
        await job_tracker.increment_jobs_completed()  # clear the session warm-up gate so a pop happens
        session = Mock()
        session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
        popper = _make_popper(
            job_tracker=job_tracker,
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
            post_processing_lane_paused_provider=lambda: lane_paused,
        )

        await popper.api_job_pop()

        session.submit_request.assert_awaited_once()
        return session.submit_request.call_args.args[0]

    async def test_paused_lane_withholds_post_processing(self) -> None:
        """A lane held off the GPU cannot run an upscale, so the pop stops advertising post-processing."""
        request = await self._pop_and_capture_request(lane_paused=True)

        assert request.allow_post_processing is False

    async def test_restored_lane_advertises_post_processing_again(self) -> None:
        """Once the lane is back on the GPU the capability is advertised again, with no latch to clear."""
        request = await self._pop_and_capture_request(lane_paused=False)

        assert request.allow_post_processing is True

    def test_pause_alone_withholds_with_the_headroom_gate_open(self) -> None:
        """The pause itself withholds the offer, not either self-protection latch.

        Both latches are proven clear first: the headroom gate is driven open by free VRAM that holds the open
        requirement past the sustain window, and the fault breaker never trips. The reclaim-ladder lane pause is
        then the only remaining reason the offer can be withheld.
        """
        manager = make_testable_process_manager(
            post_processing_fault_breaker_enabled=True,
            device_free_mb=POST_PROCESSING_GATE_OPEN_REQUIREMENT_MB + 5000.0,
        )
        manager._apply_post_processing_headroom_gate()
        manager._pp_headroom_sustained_since = (
            time.monotonic() - manager._POST_PROCESSING_HEADROOM_SUSTAIN_SECONDS - 1.0
        )
        manager._apply_post_processing_headroom_gate()
        assert manager._state.post_processing_withheld_for_headroom is False
        assert manager._state.post_processing_disabled_by_breaker is False
        assert manager._job_popper._post_processing_offer_withheld() is False

        assert manager._inference_scheduler.pause_post_process_lane(None) is True
        assert manager._process_lifecycle.is_post_process_gpu_paused is True

        # Neither latch moved: the pause is the sole cause of the withheld offer.
        assert manager._state.post_processing_withheld_for_headroom is False
        assert manager._state.post_processing_disabled_by_breaker is False
        assert manager._job_popper._post_processing_offer_withheld() is True


class TestVramPressureModelNarrowing:
    """Under sustained VRAM pressure the whole-card models come off the offer, floored so it never empties."""

    _FLUX = "Flux.1-Schnell fp8 (Compact)"
    _CASCADE = "Stable Cascade 1.0"
    _SMALL = "stable_diffusion"

    async def _popper_holding_one_job(
        self,
        *,
        under_pressure: bool,
        model: str = _SMALL,
    ) -> JobPopper:
        """A popper holding one queued job (so the idle rail does not fire) with the given pressure reading."""
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_job_pop_response(model))
        return _make_popper(
            job_tracker=job_tracker,
            vram_pressure_provider=lambda: under_pressure,
        )

    async def test_pressure_drops_extra_large_models(self) -> None:
        """With every card under pressure the whole-card models are withheld and the rest stay offered."""
        popper = await self._popper_holding_one_job(under_pressure=True)

        offered = popper._apply_vram_pressure_model_narrowing({self._SMALL, self._FLUX, self._CASCADE})

        assert offered == {self._SMALL}

    async def test_healthy_worker_offers_everything(self) -> None:
        """With no pressure the offer is untouched, whole-card models included."""
        popper = await self._popper_holding_one_job(under_pressure=False)

        offered = popper._apply_vram_pressure_model_narrowing({self._SMALL, self._FLUX, self._CASCADE})

        assert offered == {self._SMALL, self._FLUX, self._CASCADE}

    async def test_whole_card_only_worker_keeps_its_offer(self) -> None:
        """A worker configured only with whole-card models keeps offering them: an empty offer never recovers."""
        popper = await self._popper_holding_one_job(under_pressure=True, model=self._FLUX)

        offered = popper._apply_vram_pressure_model_narrowing({self._FLUX, self._CASCADE})

        assert offered == {self._FLUX, self._CASCADE}

    def test_idle_queue_is_never_narrowed(self) -> None:
        """Holding no work, the worker offers everything: narrowing now could never be relieved."""
        popper = _make_popper(job_tracker=JobTracker(), vram_pressure_provider=lambda: True)

        offered = popper._apply_vram_pressure_model_narrowing({self._SMALL, self._FLUX})

        assert offered == {self._SMALL, self._FLUX}

    async def test_pop_under_pressure_still_advertises_a_non_empty_model_set(self) -> None:
        """End to end: a whole-card-only worker under pressure still sends a real, non-empty offer."""
        job_tracker = JobTracker()
        job_tracker.set_performance_mode_thresholds(1000)
        await track_popped_job_async(job_tracker, make_job_pop_response(self._FLUX))
        await job_tracker.increment_jobs_completed()
        session = Mock()
        session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
        process_map = ProcessMap(
            {
                0: make_mock_process_info(0, model_name=self._FLUX, state=HordeProcessState.WAITING_FOR_JOB),
                10: make_mock_process_info(
                    10,
                    model_name=None,
                    state=HordeProcessState.WAITING_FOR_JOB,
                    process_type=HordeProcessType.SAFETY,
                ),
            },  # type: ignore[arg-type]
        )
        popper = _make_popper(
            job_tracker=job_tracker,
            process_map=process_map,
            horde_client_session=session,
            bridge_data=make_mock_bridge_data(image_models_to_load=[self._FLUX], queue_size=8),
            vram_pressure_provider=lambda: True,
        )

        await popper.api_job_pop()

        session.submit_request.assert_awaited_once()
        request = session.submit_request.call_args.args[0]
        assert request.models == [self._FLUX]


class TestPostProcessingBacklogOfferShaping:
    """Post-processing pressure narrows the advertised feature set without stopping unrelated intake."""

    @staticmethod
    async def _pop_and_capture_request(
        *,
        post_processing_backlog_depth: int,
        accepted_post_processing_depth: int = 0,
        accepted_non_post_processing_depth: int = 0,
        post_processing_lane_enabled: bool = True,
        allow_post_processing: bool = True,
        availability: ModelAvailability | None = None,
        urgent: bool = False,
        shared_lane_commitments: int = 0,
    ) -> tuple[object, WorkerState]:
        """Drive one pop with configured post-processing commitments and return the request and state."""
        job_tracker = JobTracker()
        job_tracker.set_performance_mode_thresholds(1000)
        await job_tracker.increment_jobs_completed()
        await _queue_n_popped_jobs_requesting_post_processing(job_tracker, accepted_post_processing_depth)
        await _queue_n_popped_jobs_without_post_processing(job_tracker, accepted_non_post_processing_depth)
        await _queue_n_jobs_for_post_processing(job_tracker, post_processing_backlog_depth)

        state = WorkerState(last_job_pop_time=time.time() if urgent else 0.0)
        session = Mock()
        session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
        popper = _make_popper(
            state=state,
            job_tracker=job_tracker,
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
            bridge_data=make_mock_bridge_data(
                allow_post_processing=allow_post_processing,
                post_processing_lane_enabled=post_processing_lane_enabled,
                queue_size=8,
            ),
            model_availability=availability,
            post_processing_lane_commitments_provider=lambda: shared_lane_commitments,
        )

        await popper.api_job_pop(urgent=urgent)

        session.submit_request.assert_awaited_once()
        return session.submit_request.call_args.args[0], state

    async def test_deep_post_processing_backlog_suppresses_only_post_processing_offer(self) -> None:
        """A saturated post-processing tail should not stop normal image pops."""
        request, state = await self._pop_and_capture_request(post_processing_backlog_depth=2)

        assert request.allow_post_processing is False
        assert state.last_pop_skipped_reasons.get("post_processing_backlog") is None

    async def test_shallow_post_processing_backlog_keeps_post_processing_offer(self) -> None:
        """A single waiting post-processing job is within the lane's ordinary overlap budget."""
        request, _state = await self._pop_and_capture_request(post_processing_backlog_depth=1)

        assert request.allow_post_processing is True

    async def test_accepted_post_processing_jobs_suppress_next_post_processing_offer(self) -> None:
        """Already-popped post-processing jobs count before they reach the post-processing lane."""
        request, state = await self._pop_and_capture_request(
            post_processing_backlog_depth=0,
            accepted_post_processing_depth=2,
        )

        assert request.allow_post_processing is False
        assert state.last_pop_skipped_reasons.get("post_processing_backlog") is None

    async def test_mixed_accepted_and_lane_post_processing_pressure_suppresses_offer(self) -> None:
        """Accepted PP commitments and lane backlog share one pressure counter."""
        request, _state = await self._pop_and_capture_request(
            post_processing_backlog_depth=1,
            accepted_post_processing_depth=1,
        )

        assert request.allow_post_processing is False

    async def test_ordinary_accepted_jobs_do_not_suppress_post_processing_offer(self) -> None:
        """Only jobs that requested post-processing count toward post-processing offer shaping."""
        request, _state = await self._pop_and_capture_request(
            post_processing_backlog_depth=0,
            accepted_non_post_processing_depth=2,
        )

        assert request.allow_post_processing is True

    async def test_graph_alchemy_lane_commitments_suppress_post_processing_offer(self) -> None:
        """Graph alchemy occupies the same post-processing lane and counts toward offer shaping."""
        request, _state = await self._pop_and_capture_request(
            post_processing_backlog_depth=0,
            shared_lane_commitments=2,
        )

        assert request.allow_post_processing is False

    async def test_disabled_lane_does_not_apply_backlog_offer_shaping(self) -> None:
        """A stale pending-post-processing count is ignored when no dedicated lane is active."""
        request, _state = await self._pop_and_capture_request(
            post_processing_backlog_depth=2,
            post_processing_lane_enabled=False,
        )

        assert request.allow_post_processing is True

    async def test_backlog_offer_shaping_runs_after_readiness_allows_feature(self) -> None:
        """Readiness may allow post-processing while backlog pressure still suppresses the next offer."""
        availability = ModelAvailability()
        availability.update(
            present={"stable_diffusion"},
            currently_downloading=None,
            pending=(),
            failed=(),
            post_processing_present=True,
        )

        request, _state = await self._pop_and_capture_request(
            post_processing_backlog_depth=2,
            availability=availability,
        )

        assert request.allow_post_processing is False

    async def test_urgent_pop_still_suppresses_post_processing_offer(self) -> None:
        """The idle-fill/urgent path may bypass cadence, but not the post-processing offer shape."""
        request, _state = await self._pop_and_capture_request(
            post_processing_backlog_depth=2,
            urgent=True,
        )

        assert request.allow_post_processing is False

    async def test_configured_post_processing_off_stays_off_with_or_without_backlog(self) -> None:
        """Backlog pressure must not re-enable a feature disabled by configuration."""
        request, _state = await self._pop_and_capture_request(
            post_processing_backlog_depth=0,
            allow_post_processing=False,
        )

        assert request.allow_post_processing is False


class TestPopAhead:
    """Tests for hunger detection and the urgent (throttle-bypassing) pop path."""

    def test_hungry_when_work_flowing_slot_free_and_room(self) -> None:
        """Flowing work + a free inference process + queue room + no backoff => hungry."""
        popper = _make_popper(process_map=_make_process_map_with_available_processes())
        assert popper._is_hungry(popper._runtime_config.bridge_data) is True

    def test_not_hungry_when_no_jobs_available(self) -> None:
        """If the last pop reported no work, do not fast-pop (stay polite)."""
        state = WorkerState(last_pop_no_jobs_available=True)
        popper = _make_popper(state=state, process_map=_make_process_map_with_available_processes())
        assert popper._is_hungry(popper._runtime_config.bridge_data) is False

    async def test_not_hungry_when_queue_full(self) -> None:
        """A full local queue means no need to fast-pop."""
        job_tracker = JobTracker()
        for _ in range(10):
            await track_popped_job_async(job_tracker, make_mock_job())
        popper = _make_popper(job_tracker=job_tracker, process_map=_make_process_map_with_available_processes())
        assert popper._is_hungry(popper._runtime_config.bridge_data) is False

    def test_not_hungry_when_no_free_inference_process(self) -> None:
        """No process able to take a job means fast-popping would just over-fill."""
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        busy_inf = make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.INFERENCE_STARTING)
        popper = _make_popper(process_map=ProcessMap({10: safety_proc, 0: busy_inf}))
        assert popper._is_hungry(popper._runtime_config.bridge_data) is False

    def test_not_hungry_when_in_error_backoff(self) -> None:
        """While backing off after a pop error, do not bypass the throttle."""
        popper = _make_popper(process_map=_make_process_map_with_available_processes())
        popper._pop_throttler.on_pop_error()
        assert popper._is_hungry(popper._runtime_config.bridge_data) is False

    async def test_urgent_bypasses_frequency_throttle(self) -> None:
        """An urgent pop proceeds even within the inter-pop frequency window."""
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())
        await job_tracker.increment_jobs_completed()

        session = Mock()
        session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="test error"))
        popper = _make_popper(
            state=WorkerState(last_job_pop_time=time.time()),  # throttle window would normally block
            job_tracker=job_tracker,
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
        )

        await popper.api_job_pop(urgent=True)

        session.submit_request.assert_awaited_once()

    async def test_non_urgent_respects_frequency_throttle(self) -> None:
        """Without urgency, a pop inside the frequency window is skipped (no request sent)."""
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())
        await job_tracker.increment_jobs_completed()

        session = Mock()
        session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="test error"))
        popper = _make_popper(
            state=WorkerState(last_job_pop_time=time.time()),
            job_tracker=job_tracker,
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
        )

        await popper.api_job_pop(urgent=False)

        session.submit_request.assert_not_awaited()


class TestPostInferenceBackpressure:
    """Backpressure from the post-inference (safety) stage onto the popper.

    When the safety stage is slower than inference, the unbounded post-inference queue grows until jobs
    age past their horde ttl and are server-aborted as too slow (which the horde answers with forced
    maintenance). The popper must stop popping once the safety backlog can no longer clear within the
    deadline, sized from the measured safety cost and the ttl so it self-tunes instead of needing an
    operator knob.
    """

    async def test_deep_safety_backlog_blocks_pop(self) -> None:
        """A safety backlog past the deadline-derived cap suppresses popping (the core self-heal)."""
        job_tracker = JobTracker()
        # avg_safety 10s, ttl 60s -> budget 30s -> cap int(30/10)=3.
        state = WorkerState(avg_safety_seconds=10.0, recent_job_ttl=60.0)
        await _queue_n_jobs_for_safety(job_tracker, 3)
        popper = _make_popper(state=state, job_tracker=job_tracker)
        assert popper._is_post_inference_backlogged() is True

    async def test_shallow_safety_backlog_does_not_block(self) -> None:
        """A backlog below the cap leaves popping unthrottled."""
        job_tracker = JobTracker()
        state = WorkerState(avg_safety_seconds=10.0, recent_job_ttl=60.0)  # cap 3
        await _queue_n_jobs_for_safety(job_tracker, 2)
        popper = _make_popper(state=state, job_tracker=job_tracker)
        assert popper._is_post_inference_backlogged() is False

    async def test_empty_backlog_never_blocks(self) -> None:
        """With nothing waiting for safety the gate is inert regardless of timings."""
        popper = _make_popper(state=WorkerState(avg_safety_seconds=99.0, recent_job_ttl=1.0))
        assert popper._is_post_inference_backlogged() is False

    async def test_cap_rises_when_safety_is_faster(self) -> None:
        """Faster measured safety raises the tolerated backlog (self-tuning, no knob)."""
        job_tracker = JobTracker()
        await _queue_n_jobs_for_safety(job_tracker, 5)
        slow = _make_popper(state=WorkerState(avg_safety_seconds=10.0, recent_job_ttl=60.0), job_tracker=job_tracker)
        fast = _make_popper(state=WorkerState(avg_safety_seconds=2.0, recent_job_ttl=60.0), job_tracker=job_tracker)
        # cap_slow = int(30/10)=3 -> 5 blocks; cap_fast = int(30/2)=15 -> 5 is fine.
        assert slow._is_post_inference_backlogged() is True
        assert fast._is_post_inference_backlogged() is False

    async def test_cap_tightens_for_shorter_ttl(self) -> None:
        """A shorter horde deadline lowers the cap so jobs still clear in time."""
        job_tracker = JobTracker()
        await _queue_n_jobs_for_safety(job_tracker, 4)
        long_state = WorkerState(avg_safety_seconds=5.0, recent_job_ttl=300.0)
        short_state = WorkerState(avg_safety_seconds=5.0, recent_job_ttl=30.0)
        long_ttl = _make_popper(state=long_state, job_tracker=job_tracker)
        short_ttl = _make_popper(state=short_state, job_tracker=job_tracker)
        # long: int(150/5)=30 -> 4 fine; short: int(15/5)=3 -> 4 blocks.
        assert long_ttl._is_post_inference_backlogged() is False
        assert short_ttl._is_post_inference_backlogged() is True

    async def test_falls_back_to_defaults_without_measurements(self) -> None:
        """Before any safety sample or ttl, a conservative default cap still bounds the backlog."""
        job_tracker = JobTracker()
        # defaults: 8s safety, 150s ttl -> budget 75s -> cap int(75/8)=9.
        await _queue_n_jobs_for_safety(job_tracker, 9)
        popper = _make_popper(state=WorkerState(), job_tracker=job_tracker)
        assert popper._is_post_inference_backlogged() is True

    async def test_cap_scales_with_safety_process_count(self) -> None:
        """Two safety processes clear the backlog twice as fast, so the cap doubles."""
        job_tracker = JobTracker()
        await _queue_n_jobs_for_safety(job_tracker, 5)
        state = WorkerState(avg_safety_seconds=10.0, recent_job_ttl=60.0)  # per-process cap 3
        one = _make_popper(state=state, job_tracker=job_tracker, process_map=ProcessMap({}))
        two = _make_popper(
            state=state,
            job_tracker=job_tracker,
            process_map=_make_process_map_with_available_processes(num_safety=2),
        )
        assert one._is_post_inference_backlogged() is True  # cap 3 < 5
        assert two._is_post_inference_backlogged() is False  # cap 6 >= 5

    async def test_api_job_pop_records_skip_reason_when_backlogged(self) -> None:
        """A pop suppressed by backpressure records the reason and sends no request."""
        job_tracker = JobTracker()
        state = WorkerState(avg_safety_seconds=10.0, recent_job_ttl=60.0)  # cap 3
        await _queue_n_jobs_for_safety(job_tracker, 5)
        session = Mock()
        session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="should not be called"))
        popper = _make_popper(
            state=state,
            job_tracker=job_tracker,
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
        )

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()
        assert state.last_pop_skipped_reasons.get("safety_backlog", 0) >= 1

    def test_hungry_is_false_when_backlogged(self) -> None:
        """The fast-pop path also yields to backpressure so it cannot bypass the gate."""

        async def _setup() -> JobPopper:
            job_tracker = JobTracker()
            await _queue_n_jobs_for_safety(job_tracker, 5)
            return _make_popper(
                state=WorkerState(avg_safety_seconds=10.0, recent_job_ttl=60.0),
                job_tracker=job_tracker,
                process_map=_make_process_map_with_available_processes(),
            )

        import asyncio

        popper = asyncio.run(_setup())
        assert popper._is_hungry(popper._runtime_config.bridge_data) is False


class _FakeBacklogTracker:
    """A minimal stand-in exposing the backlog surfaces the backpressure gate reads.

    The pending-submit list is held empty so these cases isolate the safety-backlog hysteresis; the submit
    backlog is exercised separately (see the submit-stall reproduction suite).
    """

    def __init__(self) -> None:
        self.jobs_pending_safety_check: list[object] = []
        self.jobs_being_safety_checked: list[object] = []
        self.jobs_pending_submit: list[object] = []

    def set_backlog(self, depth: int) -> None:
        """Set the pending-safety backlog to exactly ``depth`` placeholder entries."""
        self.jobs_pending_safety_check = [object() for _ in range(depth)]
        self.jobs_being_safety_checked = []


class TestBackpressureReleaseHysteresis:
    """The backpressure gate engages at the cap and releases only below a lower bound (hysteresis)."""

    def _popper_with_fixed_cap(self, cap: int) -> JobPopper:
        popper = _make_popper()
        popper._job_tracker = _FakeBacklogTracker()  # pyrefly: ignore - a stub stands in for the job tracker
        popper._max_safe_safety_backlog = lambda: cap  # pyrefly: ignore - fixing the cap isolates the hysteresis
        return popper

    def test_engages_at_cap_and_holds_until_lower_bound(self) -> None:
        """Once engaged at the cap the gate stays engaged until the backlog drains below half the cap."""
        popper = self._popper_with_fixed_cap(4)  # release bound = 4 * 0.5 = 2
        tracker = popper._job_tracker

        tracker.set_backlog(4)
        assert popper._is_post_inference_backlogged() is True  # engage at cap

        tracker.set_backlog(3)
        assert popper._is_post_inference_backlogged() is True  # above the lower bound: still engaged

        tracker.set_backlog(2)
        assert popper._is_post_inference_backlogged() is False  # at/below the lower bound: released

    def test_released_gate_does_not_reengage_until_cap_again(self) -> None:
        """After release, a backlog between the bounds does not re-engage until it reaches the cap again."""
        popper = self._popper_with_fixed_cap(4)
        tracker = popper._job_tracker

        tracker.set_backlog(4)
        assert popper._is_post_inference_backlogged() is True
        tracker.set_backlog(1)
        assert popper._is_post_inference_backlogged() is False  # released

        tracker.set_backlog(3)  # between bounds, but coming from released state
        assert popper._is_post_inference_backlogged() is False
        tracker.set_backlog(4)  # back at the cap
        assert popper._is_post_inference_backlogged() is True

    def test_small_backlog_is_a_noop(self) -> None:
        """A backlog that never reaches the cap never engages the gate."""
        popper = self._popper_with_fixed_cap(4)
        tracker = popper._job_tracker

        for depth in (0, 1, 2, 3, 3, 2, 1, 0):
            tracker.set_backlog(depth)
            assert popper._is_post_inference_backlogged() is False


class TestHandleConsecutiveFailures:
    """Tests for _handle_consecutive_failures directly."""

    def test_below_threshold_returns_false(self) -> None:
        """2 failures should not trigger pause."""
        state = WorkerState(consecutive_failed_jobs=2)
        popper = _make_popper(state=state)
        bd = make_mock_bridge_data()

        assert popper._handle_consecutive_failures(bd, time.time()) is False
        assert state.too_many_consecutive_failed_jobs is False

    def test_zero_failures_returns_false(self) -> None:
        """0 failures should not trigger pause."""
        state = WorkerState(consecutive_failed_jobs=0)
        popper = _make_popper(state=state)
        bd = make_mock_bridge_data()

        assert popper._handle_consecutive_failures(bd, time.time()) is False

    def test_exactly_three_failures_triggers(self) -> None:
        """Exactly 3 failures should trigger pause."""
        state = WorkerState(consecutive_failed_jobs=3)
        popper = _make_popper(state=state)
        bd = make_mock_bridge_data()

        result = popper._handle_consecutive_failures(bd, time.time())

        assert result is True
        assert state.too_many_consecutive_failed_jobs is True

    def test_arming_the_pause_increments_the_run_counter(self) -> None:
        """Arming the pause bumps the monotonic run counter the soak result reads to detect any backoff."""
        state = WorkerState(consecutive_failed_jobs=3)
        popper = _make_popper(state=state)
        bd = make_mock_bridge_data()

        assert state.consecutive_failed_jobs_pause_count == 0
        popper._handle_consecutive_failures(bd, time.time())
        assert state.consecutive_failed_jobs_pause_count == 1

        # A still-active pause on a later cycle must not double-count the same episode.
        popper._handle_consecutive_failures(bd, time.time())
        assert state.consecutive_failed_jobs_pause_count == 1

    def test_active_pause_returns_true_within_window(self) -> None:
        """When already in a failure pause, the method should return True to indicate the pause is still active."""
        state = WorkerState(
            too_many_consecutive_failed_jobs=True,
            too_many_consecutive_failed_jobs_time=time.time(),
        )
        popper = _make_popper(state=state)
        bd = make_mock_bridge_data()

        assert popper._handle_consecutive_failures(bd, time.time()) is True

    def test_active_pause_resets_after_wait_window(self) -> None:
        """After the wait window, the method should return True once to indicate reset, then clear the failure."""
        expired_time = time.time() - CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS - 1
        state = WorkerState(
            too_many_consecutive_failed_jobs=True,
            too_many_consecutive_failed_jobs_time=expired_time,
            consecutive_failed_jobs=5,
        )
        popper = _make_popper(state=state)
        bd = make_mock_bridge_data()

        result = popper._handle_consecutive_failures(bd, time.time())

        # Returns True for the current cycle (resetting) but state is cleared
        assert result is True
        assert state.too_many_consecutive_failed_jobs is False
        assert state.consecutive_failed_jobs == 0


class TestIsQueueFull:
    """Tests for _is_queue_full."""

    def test_empty_queue_not_full(self) -> None:
        """With no pending jobs, the queue should not be considered full."""
        popper = _make_popper()
        bd = make_mock_bridge_data(queue_size=1, max_threads=1)

        assert popper._is_queue_full(bd) is False

    async def test_queue_at_capacity_is_full(self) -> None:
        """When pending jobs reach the max allowed, the queue should be considered full."""
        job_tracker = JobTracker()
        # queue_size=1, max_threads=1 → max_jobs_in_queue = 2
        for _ in range(2):
            await track_popped_job_async(job_tracker, make_mock_job())

        popper = _make_popper(job_tracker=job_tracker)
        bd = make_mock_bridge_data(queue_size=1, max_threads=1)

        assert popper._is_queue_full(bd) is True

    async def test_multi_thread_increases_capacity(self) -> None:
        """max_threads > 1 should increase allowed queue depth."""
        job_tracker = JobTracker()
        for _ in range(2):
            await track_popped_job_async(job_tracker, make_mock_job())

        popper = _make_popper(job_tracker=job_tracker)
        bd = make_mock_bridge_data(queue_size=1, max_threads=2)

        # max_jobs_in_queue = queue_size + 1 + (max_threads - 1) = 1 + 1 + 1 = 3
        assert popper._is_queue_full(bd) is False

    async def test_queue_one_below_capacity_not_full(self) -> None:
        """When pending jobs are one below the max allowed, the queue should not be considered full."""
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_mock_job())

        popper = _make_popper(job_tracker=job_tracker)
        bd = make_mock_bridge_data(queue_size=1, max_threads=1)

        # max_jobs_in_queue = 2, current = 1 → not full
        assert popper._is_queue_full(bd) is False

    async def test_large_queue_size(self) -> None:
        """With a larger queue_size, the method should calculate capacity accordingly."""
        job_tracker = JobTracker()
        for _ in range(5):
            await track_popped_job_async(job_tracker, make_mock_job())

        popper = _make_popper(job_tracker=job_tracker)
        bd = make_mock_bridge_data(queue_size=10, max_threads=1)

        # max_jobs_in_queue = 10 + 1 = 11, current = 5 → not full
        assert popper._is_queue_full(bd) is False


class TestProcessApiMessages:
    """Tests for _process_api_messages."""

    def test_no_messages_attribute(self) -> None:
        """Response without messages attr should not raise."""
        popper = _make_popper()
        response = Mock(spec=[])  # no attributes at all

        popper._process_api_messages(response)

        assert len(popper._api_messages_received) == 0

    def test_none_messages(self) -> None:
        """If messages is None, it should be treated the same as an empty list (no messages)."""
        popper = _make_popper()
        response = Mock()
        response.messages = None

        popper._process_api_messages(response)

        assert len(popper._api_messages_received) == 0

    def test_empty_messages(self) -> None:
        """If messages is an empty list, it should simply result in no messages being processed."""
        popper = _make_popper()
        response = Mock()
        response.messages = []  # pyrefly: ignore - we don't need type inference for an empty list in this test

        popper._process_api_messages(response)

        assert len(popper._api_messages_received) == 0

    def test_new_message_stored(self) -> None:
        """A message with a new ID should be stored in _api_messages_received."""
        popper = _make_popper()
        response = Mock()
        response.messages = [
            {"id": "msg-1", "message": "hello", "origin": "system", "expiry": "2026-12-31"},
        ]

        popper._process_api_messages(response)

        assert "msg-1" in popper._api_messages_received
        assert popper._api_messages_received["msg-1"].message_text == "hello"

    def test_duplicate_message_not_overwritten(self) -> None:
        """Same message ID should not overwrite an already-received message."""
        popper = _make_popper()
        response1 = Mock()
        response1.messages = [
            {
                "id": "msg-1",
                "message": "first",
                "origin": "system",
                "expiry": None,  # pyrefly: ignore - we don't need type inference for this test
            },
        ]
        response2 = Mock()
        response2.messages = [
            {
                "id": "msg-1",
                "message": "second",
                "origin": "system",
                "expiry": None,  # pyrefly: ignore - we don't need type inference for this test
            },
        ]

        popper._process_api_messages(response1)
        popper._process_api_messages(response2)

        assert popper._api_messages_received["msg-1"].message_text == "first"

    def test_multiple_messages_in_one_response(self) -> None:
        """Multiple messages in a single response should all be processed."""
        popper = _make_popper()
        response = Mock()
        response.messages = [
            {
                "id": "msg-1",
                "message": "a",
                "origin": "system",
                "expiry": None,  # pyrefly: ignore - we don't need type inference for this test
            },
            {
                "id": "msg-2",
                "message": "b",
                "origin": "admin",
                "expiry": None,  # pyrefly: ignore - we don't need type inference for this test
            },
        ]

        popper._process_api_messages(response)

        assert len(popper._api_messages_received) == 2

    def test_malformed_message_does_not_crash(self) -> None:
        """An exception during message parsing should be caught, not propagated."""
        popper = _make_popper()
        response = Mock()
        # A non-dict message should cause from_raw_dict to fail
        response.messages = [42]

        # Should not raise
        popper._process_api_messages(response)


class TestHandlePopErrorResponse:
    """Tests for _handle_pop_error_response."""

    def _make_error_response(self, message: str) -> RequestErrorResponse:
        resp = Mock(spec=RequestErrorResponse)
        resp.message = message
        return resp

    def test_maintenance_mode_sets_state_flag(self) -> None:
        """Maintenance mode messages cause last_pop_maintenance_mode and last_pop_no_jobs_available to be True."""
        state = WorkerState()
        popper = _make_popper(state=state)

        resp = self._make_error_response("Server is in maintenance mode")

        popper._handle_pop_error_response(resp)

        assert state.last_pop_maintenance_mode is True
        assert state.last_pop_no_jobs_available is True

    def test_maintenance_mode_only_warns_first_time(self) -> None:
        """Second maintenance mode response should not re-log the warning."""
        state = WorkerState(last_pop_maintenance_mode=True)
        popper = _make_popper(state=state)

        resp = self._make_error_response("Server is in maintenance mode")

        # Should not raise; just quietly update
        popper._handle_pop_error_response(resp)

    def test_every_rejection_is_counted_even_though_only_the_first_warns(self) -> None:
        """The warning is edge-triggered, so the count is what measures a long maintenance episode."""
        state = WorkerState()
        popper = _make_popper(state=state)

        for _ in range(3):
            popper._handle_pop_error_response(self._make_error_response("Server is in maintenance mode"))

        assert state.server_maintenance_pop_rejections == 3
        assert state.server_maintenance_latched_at > 0.0

    def test_the_hordes_own_reason_marks_the_pause_as_forced(self) -> None:
        """The horde writes this reason itself when it pauses a worker for dropping too many jobs."""
        state = WorkerState()
        popper = _make_popper(state=state)

        popper._handle_pop_error_response(
            self._make_error_response("Maintenance mode activated because worker is dropping too many jobs"),
        )

        assert state.server_maintenance_forced_by_server is True

    def test_any_other_reason_is_treated_as_deliberate(self) -> None:
        """A pause the horde did not impose belongs to whoever set it, so the worker leaves it alone."""
        state = WorkerState()
        popper = _make_popper(state=state)

        popper._handle_pop_error_response(
            self._make_error_response("This worker has been put into maintenance mode by its owner"),
        )

        assert state.server_maintenance_forced_by_server is False

    def test_wrong_credentials_message(self) -> None:
        """Wrong credentials messages cause last_pop_no_jobs_available to be True."""
        state = WorkerState()
        popper = _make_popper(state=state)

        resp = self._make_error_response("Wrong credentials provided")

        popper._handle_pop_error_response(resp)

        assert state.last_pop_no_jobs_available is True

    def test_unrecognized_model_message(self) -> None:
        """Unrecognized model messages cause last_pop_no_jobs_available to be True."""
        state = WorkerState()
        popper = _make_popper(state=state)

        resp = self._make_error_response("We cannot accept workers serving this model")

        popper._handle_pop_error_response(resp)

        assert state.last_pop_no_jobs_available is True

    def test_generic_error_message(self) -> None:
        """Generic error messages should not set any state flags."""
        state = WorkerState()
        popper = _make_popper(state=state)

        resp = self._make_error_response("Something unexpected went wrong")

        popper._handle_pop_error_response(resp)

        assert state.last_pop_no_jobs_available is True

    def test_error_response_slows_throttler(self) -> None:
        """Any error response should cause the pop frequency to slow down."""
        popper = _make_popper()
        original_frequency = popper._pop_throttler.current_pop_frequency

        resp = self._make_error_response("Server error")

        popper._handle_pop_error_response(resp)

        assert popper._pop_throttler.current_pop_frequency > original_frequency


class TestApplySdkWorkarounds:
    """Tests for _apply_sdk_workarounds."""

    def test_missing_seed_gets_assigned(self) -> None:
        """Jobs without a seed should receive a random integer seed."""
        job = make_job_pop_response(seed="42")
        # Simulate SDK returning None seed
        dumped = job.model_dump(by_alias=True)
        dumped["payload"]["seed"] = None
        from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

        job_no_seed = ImageGenerateJobPopResponse(**dumped)

        result = JobPopper._apply_sdk_workarounds(job_no_seed)

        assert result.payload.seed is not None

    def test_denoising_strength_cleared_without_source_image(self) -> None:
        """Denoising strength should be None when there's no source image (txt2img)."""
        job = make_job_pop_response()
        dumped = job.model_dump(by_alias=True)
        dumped["payload"]["denoising_strength"] = 0.75
        dumped["source_image"] = None
        from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

        job_with_denoise = ImageGenerateJobPopResponse(**dumped)

        result = JobPopper._apply_sdk_workarounds(job_with_denoise)

        assert result.payload.denoising_strength is None

    def test_denoising_strength_preserved_with_source_image(self) -> None:
        """Denoising strength should be kept when source image exists."""
        job = make_job_pop_response()
        dumped = job.model_dump(by_alias=True)
        dumped["payload"]["denoising_strength"] = 0.75
        dumped["source_image"] = "base64imagedata"
        from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

        job_with_img = ImageGenerateJobPopResponse(**dumped)

        result = JobPopper._apply_sdk_workarounds(job_with_img)

        assert result.payload.denoising_strength == 0.75

    def test_no_workarounds_needed_returns_same_data(self) -> None:
        """When neither workaround applies, the response should be equivalent."""
        job = make_job_pop_response(seed="42")

        result = JobPopper._apply_sdk_workarounds(job)

        assert result.payload.seed is not None
        assert result.id_ == job.id_


class TestEnqueuePoppedJob:
    """Tests for _enqueue_popped_job."""

    async def test_job_added_to_pending_inference(self) -> None:
        """When a job is enqueued, it should be added to the jobs_pending_inference list."""
        job_tracker = JobTracker()
        popper = _make_popper(job_tracker=job_tracker)
        job = make_job_pop_response()

        await popper._enqueue_popped_job(job)

        assert len(job_tracker.jobs_pending_inference) == 1
        assert job_tracker.jobs_pending_inference[0] is job

    async def test_pop_timestamp_recorded(self) -> None:
        """When a job is enqueued, the current time should be recorded in job_pop_timestamps for that job."""
        job_tracker = JobTracker()
        popper = _make_popper(job_tracker=job_tracker)
        job = make_job_pop_response()

        await popper._enqueue_popped_job(job)

        assert job in job_tracker.job_pop_timestamps
        assert job_tracker.job_pop_timestamps[job] > 0

    async def test_jobs_lookup_entry_created(self) -> None:
        """When a job is enqueued, an entry should be created in jobs_lookup with the correct info."""
        job_tracker = JobTracker()
        popper = _make_popper(job_tracker=job_tracker)
        job = make_job_pop_response()

        await popper._enqueue_popped_job(job)

        assert job in job_tracker.jobs_lookup
        info = job_tracker.jobs_lookup[job]
        assert info.sdk_api_job_info is job
        assert info.state is None
        assert info.time_popped > 0

    async def test_multiple_jobs_enqueued_in_order(self) -> None:
        """When multiple jobs are enqueued, they should be added to the pending list in the order enqueued."""
        job_tracker = JobTracker()
        popper = _make_popper(job_tracker=job_tracker)
        job1 = make_job_pop_response(model="model_a")
        job2 = make_job_pop_response(model="model_b")

        await popper._enqueue_popped_job(job1)
        await popper._enqueue_popped_job(job2)

        assert len(job_tracker.jobs_pending_inference) == 2
        assert job_tracker.jobs_pending_inference[0] is job1
        assert job_tracker.jobs_pending_inference[1] is job2


class TestDryRunPopRequestHandoff:
    """Dry-run sources receive the same validated request that the live API would receive."""

    async def test_constructed_request_is_passed_to_canned_source(self) -> None:
        """The simulated API boundary receives dynamic model, size, and feature shaping."""
        source = Mock(spec=CannedJobSource)
        source.next_pop_response.return_value = make_empty_pop_response()
        bridge_data = make_mock_bridge_data(
            image_models_to_load=["stable_diffusion"],
            max_power=32,
            allow_lora=False,
        )
        popper = _make_popper(
            state=WorkerState(last_job_pop_time=0.0),
            process_map=_make_process_map_with_available_processes(),
            bridge_data=bridge_data,
            dry_run_skip_api=True,
        )
        popper.set_canned_job_source(source)

        await popper.api_job_pop()

        source.next_pop_response.assert_called_once()
        request = source.next_pop_response.call_args.args[0]
        assert isinstance(request, ImageGenerateJobPopRequest)
        assert request.models == ["stable_diffusion"]
        assert request.max_pixels == 32 * 8 * 64 * 64
        assert request.allow_lora is False


# Patch paths in the module under test to bypass SDK and telemetry dependencies
_POP_REQUEST_PATH = "horde_worker_regen.process_management.jobs.job_popper.ImageGenerateJobPopRequest"
_SPAN_POP_PATH = "horde_worker_regen.process_management.jobs.job_popper.span_job_pop"
_VERSION_PATH = "horde_worker_regen.__version__"


def _noop_span(**_kwargs: object):  # noqa: ANN202
    """No-op replacement for span_job_pop."""
    import contextlib

    return contextlib.nullcontext()


# Stack all three patches needed for full-flow tests
# (untyped because this is a decorator factory, not a regular function, so type inference isn't helpful here)
def _full_flow_patches(  # noqa: ANN202
    fn,  # noqa: ANN001
):
    """Apply all patches needed to run api_job_pop through the full flow."""
    fn = patch(_SPAN_POP_PATH, _noop_span)(fn)
    fn = patch(_POP_REQUEST_PATH)(fn)
    fn = patch(_VERSION_PATH, "0.0.0-test", create=True)(fn)
    return fn  # noqa: RET504


class TestApiJobPopFullFlow:
    """End-to-end tests for api_job_pop with mocked API responses.

    We patch ImageGenerateJobPopRequest and span_job_pop so we don't depend on
    SDK validation or telemetry; the tests focus on how JobPopper orchestrates
    the response.
    """

    def _make_ready_popper(
        self,
        *,
        api_response: object | None = None,
        state: WorkerState | None = None,
        job_tracker: JobTracker | None = None,
        bridge_data: Mock | None = None,
        model_availability: ModelAvailability | None = None,
    ) -> JobPopper:
        """Create a popper in a state where all guard clauses pass."""
        if state is None:
            state = WorkerState(last_job_pop_time=0.0)
        if job_tracker is None:
            job_tracker = JobTracker()

        pm = _make_process_map_with_available_processes()

        horde_session = AsyncMock()
        if api_response is not None:
            horde_session.submit_request = AsyncMock(return_value=api_response)

        return _make_popper(
            state=state,
            process_map=pm,
            job_tracker=job_tracker,
            bridge_data=bridge_data,
            horde_client_session=horde_session,
            model_availability=model_availability,
        )

    @_full_flow_patches
    async def test_successful_pop_enqueues_job(self, _mock_req_cls: Mock) -> None:
        """A successful pop with a valid job should add it to the queue."""
        job_response = make_job_pop_response()
        popper = self._make_ready_popper(api_response=job_response)

        await popper.api_job_pop()

        assert len(popper._job_tracker.jobs_pending_inference) == 1
        assert popper._state.last_pop_no_jobs_available is False

    @_full_flow_patches
    async def test_successful_pop_resets_maintenance_flag(self, _mock_req_cls: Mock) -> None:
        """After a successful pop, last_pop_maintenance_mode should be reset to False.

        This is so that future maintenance mode responses will log a warning again.
        """
        state = WorkerState(last_job_pop_time=0.0)
        popper = self._make_ready_popper(
            api_response=make_job_pop_response(),
            state=state,
        )
        popper._state.last_pop_maintenance_mode = True
        popper._state.server_maintenance_latched_at = time.time()
        popper._state.server_maintenance_forced_by_server = True
        popper._state.server_maintenance_pop_rejections = 12

        await popper.api_job_pop()

        assert popper._state.last_pop_maintenance_mode is False  # pyrefly: ignore - "always true" is wrong, api_job_pop() should mutate
        assert popper._state.server_maintenance_cleared_by_job_pop is True
        # Work arriving is the end-to-end proof the episode is over, so it retires the whole episode.
        assert popper._state.server_maintenance_latched_at == 0.0
        assert popper._state.server_maintenance_forced_by_server is False
        assert popper._state.server_maintenance_pop_rejections == 0

    @_full_flow_patches
    async def test_successful_pop_resets_throttler_to_default(self, _mock_req_cls: Mock) -> None:
        """After a successful pop, the throttler should reset to default frequency."""
        popper = self._make_ready_popper(api_response=make_job_pop_response())
        popper._pop_throttler.on_pop_error()  # put throttler in error state

        await popper.api_job_pop()

        assert popper._pop_throttler.current_pop_frequency == popper._pop_throttler._default_pop_frequency

    @_full_flow_patches
    async def test_no_job_available_sets_flag(self, _mock_req_cls: Mock) -> None:
        """When API returns a response with id_ = None (no job), flag should be set."""
        # Build a mock that quacks like an empty ImageGenerateJobPopResponse
        empty_response = Mock()
        empty_response.id_ = None
        empty_response.skipped = Mock()
        empty_response.skipped.model_dump.return_value = {}  #  pyrefly: ignore - we just need to ensure this doesn't raise, the actual content isn't important for this test
        empty_response.skipped.model_extra = None
        empty_response.messages = None

        popper = self._make_ready_popper(api_response=empty_response)

        await popper.api_job_pop()

        assert popper._state.last_pop_no_jobs_available is True
        assert len(popper._job_tracker.jobs_pending_inference) == 0

    @_full_flow_patches
    async def test_no_job_available_does_not_clear_maintenance_latch(self, _mock_req_cls: Mock) -> None:
        """Only a real popped job proves horde maintenance is off; an empty response does not."""
        empty_response = Mock()
        empty_response.id_ = None
        empty_response.skipped = Mock()
        empty_response.skipped.model_dump.return_value = {}
        empty_response.skipped.model_extra = None
        empty_response.messages = None
        state = WorkerState(last_job_pop_time=0.0, last_pop_maintenance_mode=True)
        popper = self._make_ready_popper(api_response=empty_response, state=state)

        await popper.api_job_pop()

        assert popper._state.last_pop_maintenance_mode is True
        assert popper._state.server_maintenance_cleared_by_job_pop is False

    @_full_flow_patches
    async def test_api_exception_slows_throttler(self, _mock_req_cls: Mock) -> None:
        """When the API call raises, the throttler should switch to error frequency."""
        horde_session = AsyncMock()
        horde_session.submit_request = AsyncMock(side_effect=ConnectionError("network down"))

        job_tracker = JobTracker()

        popper = _make_popper(
            state=WorkerState(last_job_pop_time=0.0),
            process_map=_make_process_map_with_available_processes(),
            job_tracker=job_tracker,
            horde_client_session=horde_session,
        )

        await popper.api_job_pop()

        assert popper._pop_throttler.current_pop_frequency == popper._pop_throttler._error_pop_frequency

    @_full_flow_patches
    async def test_error_response_handled(self, _mock_req_cls: Mock) -> None:
        """RequestErrorResponse should be handled by _handle_pop_error_response."""
        error_resp = Mock(spec=RequestErrorResponse)
        error_resp.message = "Server is in maintenance mode"
        error_resp.__class__ = RequestErrorResponse

        horde_session = AsyncMock()
        horde_session.submit_request = AsyncMock(return_value=error_resp)

        job_tracker = JobTracker()

        state = WorkerState(last_job_pop_time=0.0)
        popper = _make_popper(
            state=state,
            process_map=_make_process_map_with_available_processes(),
            job_tracker=job_tracker,
            horde_client_session=horde_session,
        )

        await popper.api_job_pop()

    @_full_flow_patches
    async def test_job_faults_initialized_for_popped_job(self, _mock_req_cls: Mock) -> None:
        """When a job is popped, its fault list should be initialized."""
        job_response = make_job_pop_response()
        popper = self._make_ready_popper(api_response=job_response)

        await popper.api_job_pop()

        assert job_response.id_ in popper._job_tracker.job_faults
        assert popper._job_tracker.job_faults[job_response.id_] == []

    @_full_flow_patches
    async def test_pop_updates_last_pop_time(self, _mock_req_cls: Mock) -> None:
        """Successful or not, api_job_pop should update last_job_pop_time."""
        state = WorkerState(last_job_pop_time=0.0)
        popper = self._make_ready_popper(api_response=make_job_pop_response(), state=state)

        await popper.api_job_pop()

        assert state.last_job_pop_time > 0

    @_full_flow_patches
    async def test_allow_lora_true_when_configured_and_downloads_idle(self, mock_req_cls: Mock) -> None:
        """Configured LoRA support is advertised while background downloads are idle."""
        availability = ModelAvailability()
        availability.update(
            present={"stable_diffusion"},
            currently_downloading=None,
            pending=(),
            failed=(),
            status=DownloadStatusSnapshot(phase=DownloadPhase.IDLE),
        )
        popper = self._make_ready_popper(
            api_response=make_job_pop_response(),
            model_availability=availability,
        )

        await popper.api_job_pop()

        assert mock_req_cls.call_args.kwargs["allow_lora"] is True

    @_full_flow_patches
    async def test_allow_lora_false_while_background_download_active(self, mock_req_cls: Mock) -> None:
        """Active background downloads suppress LoRA advertisement for new pops."""
        availability = ModelAvailability()
        availability.update(
            present={"stable_diffusion"},
            currently_downloading="Flux",
            pending=(),
            failed=(),
            status=DownloadStatusSnapshot(
                phase=DownloadPhase.DOWNLOADING,
                current=CurrentDownloadStatus(model_name="Flux", feature="image model", target_dir="models/compvis"),
            ),
        )
        popper = self._make_ready_popper(
            api_response=make_job_pop_response(),
            model_availability=availability,
        )

        await popper.api_job_pop()

        assert mock_req_cls.call_args.kwargs["allow_lora"] is False

    @_full_flow_patches
    async def test_allow_lora_false_when_disk_exhausted(self, mock_req_cls: Mock) -> None:
        """An unrecoverable LoRA-cache disk shortfall suppresses LoRA advertisement."""
        availability = ModelAvailability()
        availability.update(
            present={"stable_diffusion"},
            currently_downloading=None,
            pending=(),
            failed=(),
            status=DownloadStatusSnapshot(phase=DownloadPhase.IDLE),
        )
        popper = self._make_ready_popper(
            api_response=make_job_pop_response(),
            state=WorkerState(last_job_pop_time=0.0, lora_disk_exhausted=True),
            model_availability=availability,
        )

        await popper.api_job_pop()

        assert mock_req_cls.call_args.kwargs["allow_lora"] is False

    @_full_flow_patches
    async def test_allow_lora_false_when_disabled_in_config(self, mock_req_cls: Mock) -> None:
        """The temporary gate cannot enable LoRA when the user disabled it."""
        availability = ModelAvailability()
        availability.update(
            present={"stable_diffusion"},
            currently_downloading=None,
            pending=(),
            failed=(),
            status=DownloadStatusSnapshot(phase=DownloadPhase.IDLE),
        )
        popper = self._make_ready_popper(
            api_response=make_job_pop_response(),
            bridge_data=make_mock_bridge_data(allow_lora=False),
            model_availability=availability,
        )

        await popper.api_job_pop()

        assert mock_req_cls.call_args.kwargs["allow_lora"] is False

    @_full_flow_patches
    async def test_allow_lora_false_while_download_backoff_active(self, mock_req_cls: Mock) -> None:
        """A live LoRA-download backoff withholds LoRA advertisement from new pops."""
        state = WorkerState(last_job_pop_time=0.0)
        state.lora_download_backoff.register_timeout(now=time.time())
        popper = self._make_ready_popper(api_response=make_job_pop_response(), state=state)

        await popper.api_job_pop()

        assert mock_req_cls.call_args.kwargs["allow_lora"] is False

    @_full_flow_patches
    async def test_allow_lora_true_after_backoff_window_elapses(self, mock_req_cls: Mock) -> None:
        """Once the backoff window has passed, LoRA advertisement resumes."""
        state = WorkerState(last_job_pop_time=0.0)
        # A strike far in the past: its window has long since elapsed.
        state.lora_download_backoff.register_timeout(now=time.time() - 10_000)
        popper = self._make_ready_popper(api_response=make_job_pop_response(), state=state)

        await popper.api_job_pop()

        assert mock_req_cls.call_args.kwargs["allow_lora"] is True


class TestLoraQueueCap:
    """Direct tests for the N-1 LoRA-queue cap helper.

    The full pop flow's queue-full guard makes pre-enqueuing jobs an awkward fixture, so the cap logic
    is exercised here; its one-line wiring into ``pop_allow_lora`` is straightforward.
    """

    def _lora_job(self) -> object:
        """Build a minimal job carrying a single LoRA."""
        return make_job_pop_response(loras=[LorasPayloadEntry(name="123", is_version=False)])

    async def test_non_lora_jobs_do_not_count(self) -> None:
        """A queue of non-LoRA jobs never reaches the LoRA cap."""
        job_tracker = JobTracker()
        popper = _make_popper(job_tracker=job_tracker, max_inference_processes=2)
        await popper._enqueue_popped_job(make_job_pop_response())
        await popper._enqueue_popped_job(make_job_pop_response())
        assert popper._lora_queue_cap_reached() is False

    async def test_cap_is_processes_minus_one(self) -> None:
        """Three inference processes allow two queued LoRA jobs before the cap is reached."""
        job_tracker = JobTracker()
        popper = _make_popper(job_tracker=job_tracker, max_inference_processes=3)

        await popper._enqueue_popped_job(self._lora_job())
        assert popper._lora_queue_cap_reached() is False

        await popper._enqueue_popped_job(self._lora_job())
        assert popper._lora_queue_cap_reached() is True

    async def test_cap_floors_at_one(self) -> None:
        """A single-inference-process worker still allows one LoRA job, capping at the next."""
        job_tracker = JobTracker()
        popper = _make_popper(job_tracker=job_tracker, max_inference_processes=1)

        assert popper._lora_queue_cap_reached() is False
        await popper._enqueue_popped_job(self._lora_job())
        assert popper._lora_queue_cap_reached() is True


class TestJobPopFrequency:
    """Tests for pop frequency state management."""

    def test_last_pop_recently_true(self) -> None:
        """When last_job_pop_time is very recent, last_pop_recently should return True."""
        state = WorkerState(last_job_pop_time=time.time())
        assert state.last_pop_recently() is True

    def test_last_pop_recently_false(self) -> None:
        """When last_job_pop_time is not recent, last_pop_recently should return False."""
        state = WorkerState(last_job_pop_time=time.time() - 20)
        assert state.last_pop_recently() is False

    def test_default_pop_frequency(self) -> None:
        """Default pop frequency should be 1.0 seconds."""
        popper = _make_popper()
        assert popper._pop_throttler._current_pop_frequency == 1.0

    def test_error_pop_frequency(self) -> None:
        """Error pop frequency should be 5.0 seconds."""
        popper = _make_popper()
        assert popper._pop_throttler._error_pop_frequency == 5.0


# region Idle-fill ladder pop shaping


async def _seed_completed_session(job_tracker: JobTracker) -> None:
    """Put one completed job behind us so the per-session warm-up gate no longer blocks a queue-ahead pop."""
    await track_popped_job_async(job_tracker, make_mock_job())
    await job_tracker.increment_jobs_completed()


async def _queue_head_model_jobs(job_tracker: JobTracker, n: int) -> None:
    """Queue ``n`` pending-inference jobs for the head model.

    Two or more of the same model trip the "one running plus one queued" per-model cap in
    ``_select_models_for_pop``, standing in for a queue saturated with jobs that all share (and so all
    block behind) the same head model.
    """
    for _ in range(n):
        await track_popped_job_async(job_tracker, make_mock_job(model="stable_diffusion"))


def _make_fill_popper(
    *,
    state: WorkerState,
    job_tracker: JobTracker,
    process_map: ProcessMap | None = None,
    bridge_data: Mock | None = None,
) -> tuple[JobPopper, Mock]:
    """Build a popper wired for idle-fill pop scenarios plus the API session mock it will call.

    Unless a test overrides them, two models are configured (the head model plus a ``sibling_model``
    standing in for the model resident on the idle sibling process) and the process pool has a free
    inference and safety process. That way the only reason a given test's pop is withheld is the single
    gate that test is exercising, not incidental ineligibility.
    """
    session = Mock()
    session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
    if bridge_data is None:
        popper = _make_popper(
            state=state,
            job_tracker=job_tracker,
            process_map=process_map if process_map is not None else _make_process_map_with_available_processes(),
            horde_client_session=session,
            image_models_to_load=["stable_diffusion", "sibling_model"],
        )
    else:
        popper = _make_popper(
            state=state,
            job_tracker=job_tracker,
            process_map=process_map if process_map is not None else _make_process_map_with_available_processes(),
            horde_client_session=session,
            bridge_data=bridge_data,
        )
    return popper, session


_SD15_BASELINE = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1
_SDXL_BASELINE = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl
_FLUX_BASELINE = KNOWN_IMAGE_GENERATION_BASELINE.flux_1
_SMALL_CAP = small_pop_max_power(high_performance_mode=False, moderate_performance_mode=False)


class _FakeModelMetadata:
    """Minimal metadata stub mapping model names to baselines for idle-fill ladder tests."""

    def __init__(self, baselines: dict[str, KNOWN_IMAGE_GENERATION_BASELINE]) -> None:
        self._baselines = baselines

    def get_baseline(self, model_name: str) -> KNOWN_IMAGE_GENERATION_BASELINE | None:
        return self._baselines.get(model_name)


def _idle_fill_state(rung: int = 0) -> WorkerState:
    """A worker state with the idle-fill breaker armed at the given ladder rung."""
    state = WorkerState()
    state.wants_idle_fill_candidate = True
    state.idle_fill_rung = rung
    return state


class TestIdleFillLadderShaping:
    """``_apply_idle_fill_ladder`` offers a smallest-fastest-first, size-narrowed no-LoRA slice per rung.

    The rungs are (light=sd15, small), (light, large), (heavy=sdxl, small), (heavy, large); rungs whose
    baseline the worker has no model for are skipped, and the whole-card EXTRA_LARGE tier is never a fill.
    """

    @staticmethod
    def _ladder(
        baselines: dict[str, KNOWN_IMAGE_GENERATION_BASELINE],
        rung: int,
        *,
        max_power: int = _SMALL_CAP * 4,
    ) -> tuple[set[str], int]:
        popper = _make_popper(state=_idle_fill_state(rung), image_models_to_load=list(baselines))
        popper._model_metadata = _FakeModelMetadata(baselines)  # type: ignore[assignment]
        return popper._apply_idle_fill_ladder(set(baselines), max_power, make_mock_bridge_data())

    def test_rung0_offers_light_at_small_cap(self) -> None:
        """Rung 0 is the first rung, offering light at small cap."""
        models, cap = self._ladder({"sd15a": _SD15_BASELINE, "sdxlA": _SDXL_BASELINE}, 0)
        assert models == {"sd15a"}
        assert cap == _SMALL_CAP

    def test_rung1_offers_light_at_large_cap(self) -> None:
        """Rung 1 is the last light rung, offering light at large cap."""
        large = _SMALL_CAP * 4
        models, cap = self._ladder({"sd15a": _SD15_BASELINE, "sdxlA": _SDXL_BASELINE}, 1, max_power=large)
        assert models == {"sd15a"}
        assert cap == large

    def test_rung2_offers_heavy_at_small_cap(self) -> None:
        """Rung 2 is the first heavy rung, offering heavy at small cap."""
        models, cap = self._ladder({"sd15a": _SD15_BASELINE, "sdxlA": _SDXL_BASELINE}, 2)
        assert models == {"sdxlA"}
        assert cap == _SMALL_CAP

    def test_rung3_offers_heavy_at_large_cap(self) -> None:
        """Rung 3 is the last rung, offering heavy at large cap."""
        large = _SMALL_CAP * 4
        models, cap = self._ladder({"sd15a": _SD15_BASELINE, "sdxlA": _SDXL_BASELINE}, 3, max_power=large)
        assert models == {"sdxlA"}
        assert cap == large

    def test_absent_light_baseline_skips_to_heavy(self) -> None:
        """Rung 0 is light, but if the worker has no light models, it should skip to the heavy rung."""
        # A worker with only SDXL models has no sd15 rungs, so rung 0 is the sdxl-small rung.
        models, cap = self._ladder({"sdxlA": _SDXL_BASELINE, "sdxlB": _SDXL_BASELINE}, 0)
        assert models == {"sdxlA", "sdxlB"}
        assert cap == _SMALL_CAP

    def test_extra_large_never_offered_as_fill(self) -> None:
        """The EXTRA_LARGE tier is never offered as a fill rung, even if the worker has a model for it."""
        baselines = {"sd15a": _SD15_BASELINE, "fluxA": _FLUX_BASELINE}
        low, _ = self._ladder(baselines, 0)
        high, _ = self._ladder(baselines, 3)  # clamps to the last existing (light) rung
        assert low == {"sd15a"}
        assert "fluxA" not in high

    def test_metadata_none_falls_back_to_flat_small(self) -> None:
        """When the popper has no model metadata, the rung logic cannot narrow by baseline.

        It should fall back to the small-cap slice of all models.
        """
        large = _SMALL_CAP * 4
        popper = _make_popper(state=_idle_fill_state(0), image_models_to_load=["m1", "m2"])
        popper._model_metadata = None  # type: ignore[assignment]
        models, cap = popper._apply_idle_fill_ladder({"m1", "m2"}, large, make_mock_bridge_data())
        assert models == {"m1", "m2"}  # unchanged: no baseline info to narrow by
        assert cap == _SMALL_CAP


class TestIdleFillEscalationAndGates:
    """Full-pop behaviour when the scheduler arms ``wants_idle_fill_candidate``."""

    async def test_no_lora_offered_on_fill(self) -> None:
        """A fill pop never advertises LoRA support (a LoRA job would itself block on a download)."""
        popper, session = _make_fill_popper(
            state=_idle_fill_state(0),
            job_tracker=JobTracker(),
            bridge_data=make_mock_bridge_data(allow_lora=True),
        )
        await popper.api_job_pop()
        request: ImageGenerateJobPopRequest = session.submit_request.call_args.args[0]
        assert request.allow_lora is False

    async def test_no_job_advances_the_rung(self) -> None:
        """When the horde has no job at this rung, the ladder climbs one rung for the next fill tick."""
        state = _idle_fill_state(0)
        popper, session = _make_fill_popper(state=state, job_tracker=JobTracker())
        # The horde's "no jobs available" is a success response with id_=None (not an error response).
        session.submit_request = AsyncMock(return_value=ImageGenerateJobPopResponse(id=None, ids=[], payload={}))
        await popper.api_job_pop()
        assert state.idle_fill_rung == 1

    async def test_rung_clamps_at_the_max(self) -> None:
        """The rung counter never runs away past the last ladder index on repeated no-jobs."""
        state = _idle_fill_state(3)
        popper, session = _make_fill_popper(state=state, job_tracker=JobTracker())
        session.submit_request = AsyncMock(return_value=ImageGenerateJobPopResponse(id=None, ids=[], payload={}))
        await popper.api_job_pop()
        assert state.idle_fill_rung == 3

    @_full_flow_patches
    async def test_fill_job_resets_the_rung(self, _mock_req_cls: Mock) -> None:
        """Obtaining a fill job restarts the ladder at the smallest rung for the next idle episode."""
        state = _idle_fill_state(2)
        horde_session = AsyncMock()
        horde_session.submit_request = AsyncMock(return_value=make_job_pop_response())
        popper = _make_popper(
            state=state,
            process_map=_make_process_map_with_available_processes(),
            job_tracker=JobTracker(),
            horde_client_session=horde_session,
        )

        await popper.api_job_pop()

        assert state.idle_fill_rung == 0

    async def test_armed_pops_through_full_queue(self) -> None:
        """Idle-fill admits one extra job past the configured depth (the permitted over-pop)."""
        job_tracker = JobTracker()
        await _queue_head_model_jobs(job_tracker, 2)  # queue_size=1, max_threads=1 -> depth cap is 2
        await job_tracker.increment_jobs_completed()
        popper, session = _make_fill_popper(state=_idle_fill_state(0), job_tracker=job_tracker)

        await popper.api_job_pop()

        session.submit_request.assert_awaited_once()

    async def test_full_queue_blocks_pop_when_unarmed(self) -> None:
        """The same full queue blocks the pop when idle-fill is not armed (depth cap otherwise honoured)."""
        job_tracker = JobTracker()
        await _queue_head_model_jobs(job_tracker, 2)
        await job_tracker.increment_jobs_completed()
        popper, session = _make_fill_popper(state=WorkerState(), job_tracker=job_tracker)

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_unarmed_leaves_pop_unbiased(self) -> None:
        """With idle-fill not armed, LoRA and max_power are advertised unchanged."""
        popper, session = _make_fill_popper(
            state=WorkerState(),
            job_tracker=JobTracker(),
            bridge_data=make_mock_bridge_data(allow_lora=True, max_power=64),
        )

        await popper.api_job_pop()

        request: ImageGenerateJobPopRequest = session.submit_request.call_args.args[0]
        assert request.allow_lora is True
        assert request.max_pixels == 64 * 8 * 64 * 64


# endregion


# region Fixed model pool lane routing


class TestPoolLaneAdvertising:
    """A pool-routed pop narrows its offer to the chosen lane and gates the residency-bias call by lane."""

    @staticmethod
    def _make_pool_popper(
        *,
        seats: frozenset[str],
        eligible: list[str],
        initial_lane_state: PoolLaneState,
        outcome_sink: Callable[..., None] | None = None,
    ) -> tuple[JobPopper, Mock, Mock]:
        """Build a pool-enabled popper primed to pop once, returning it, its session, and the bias spy."""
        session = Mock()
        session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
        popper = _make_popper(
            state=WorkerState(last_job_pop_time=0.0),
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
            bridge_data=make_mock_bridge_data(
                image_models_to_load=eligible,
                model_pool=ModelPoolConfig(enabled=True),
            ),
            pool_active_seats_provider=lambda: seats,
            pool_pop_outcome_sink=outcome_sink,
        )
        popper._pool_lane_state = initial_lane_state
        bias_spy = Mock(side_effect=lambda models: models)
        popper._apply_residency_advertising_bias = bias_spy  # type: ignore[method-assign]
        return popper, session, bias_spy

    async def test_fixed_lane_advertises_seated_intersect_eligible_and_skips_bias(self) -> None:
        """The fixed lane narrows the offer to the seated-and-eligible models and skips the residency-bias call."""
        popper, session, bias_spy = self._make_pool_popper(
            seats=frozenset({"model_a", "model_b"}),
            eligible=["model_a", "model_b", "model_c"],
            initial_lane_state=PoolLaneState(fixed_credit=1000),
        )

        await popper.api_job_pop()

        request = session.submit_request.call_args.args[0]
        assert set(request.models) == {"model_a", "model_b"}
        assert popper.latest_pool_lane() is PopLane.FIXED
        bias_spy.assert_not_called()

    async def test_free_lane_advertises_eligible_minus_seats_and_keeps_bias(self) -> None:
        """The free lane offers the eligible models the pool does not seat and still runs the residency-bias call."""
        popper, session, bias_spy = self._make_pool_popper(
            seats=frozenset({"model_a"}),
            eligible=["model_a", "model_b", "model_c"],
            initial_lane_state=PoolLaneState(free_credit=1000),
        )

        await popper.api_job_pop()

        request = session.submit_request.call_args.args[0]
        assert set(request.models) == {"model_b", "model_c"}
        assert popper.latest_pool_lane() is PopLane.FREE
        bias_spy.assert_called_once()


class TestPoolLaneGating:
    """The pool routes a pop only for a non-idle-fill pop with a provider, an enabled pool, and seats."""

    @staticmethod
    def _popper(*, provider: Callable[[], frozenset[str]] | None, enabled: bool) -> JobPopper:
        """Build a pool-configured popper with the given seat provider and enabled flag."""
        return _make_popper(
            bridge_data=make_mock_bridge_data(model_pool=ModelPoolConfig(enabled=enabled)),
            pool_active_seats_provider=provider,
        )

    def test_idle_fill_pop_bypasses_the_pool(self) -> None:
        """An idle-fill pop is never routed through the pool, even with seats available."""
        popper = self._popper(provider=lambda: frozenset({"model_a"}), enabled=True)
        decision = popper._apply_pool_lane(
            {"model_a", "model_b"},
            popper._runtime_config.bridge_data,
            idle_fill_wanted=True,
        )
        assert decision is None
        assert popper.latest_pool_lane() is None

    def test_unrouted_cycle_retains_last_routed_lane_for_status(self) -> None:
        """An idle-fill cycle does not erase the most recent pool lane shown in status."""
        popper = self._popper(provider=lambda: frozenset({"model_a"}), enabled=True)
        first = popper._apply_pool_lane(
            {"model_a", "model_b"},
            popper._runtime_config.bridge_data,
            idle_fill_wanted=False,
        )
        assert first is not None

        second = popper._apply_pool_lane(
            {"model_a", "model_b"},
            popper._runtime_config.bridge_data,
            idle_fill_wanted=True,
        )

        assert second is None
        assert popper.latest_pool_lane() is first.lane

    def test_disabled_pool_does_not_route(self) -> None:
        """A disabled pool never routes a pop, leaving the offer to the existing shaping."""
        popper = self._popper(provider=lambda: frozenset({"model_a"}), enabled=False)
        decision = popper._apply_pool_lane(
            {"model_a", "model_b"},
            popper._runtime_config.bridge_data,
            idle_fill_wanted=False,
        )
        assert decision is None

    def test_absent_provider_does_not_route(self) -> None:
        """A popper wired without a seat provider never routes a pop through the pool."""
        popper = self._popper(provider=None, enabled=True)
        decision = popper._apply_pool_lane(
            {"model_a", "model_b"},
            popper._runtime_config.bridge_data,
            idle_fill_wanted=False,
        )
        assert decision is None

    def test_empty_seats_do_not_route(self) -> None:
        """An enabled pool that currently seats nothing does not route a pop."""
        popper = self._popper(provider=frozenset, enabled=True)
        decision = popper._apply_pool_lane(
            {"model_a", "model_b"},
            popper._runtime_config.bridge_data,
            idle_fill_wanted=False,
        )
        assert decision is None


class TestPoolProviderAbsentRegression:
    """With the pool feature off, the advertised model set is byte-identical to a plain popper's."""

    @staticmethod
    async def _capture_models(
        *,
        provider: Callable[[], frozenset[str]] | None,
        enabled: bool,
    ) -> set[str]:
        session = Mock()
        session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
        popper = _make_popper(
            state=WorkerState(last_job_pop_time=0.0),
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
            bridge_data=make_mock_bridge_data(
                image_models_to_load=["model_a", "model_b", "model_c"],
                model_pool=ModelPoolConfig(enabled=enabled),
            ),
            pool_active_seats_provider=provider,
        )
        await popper.api_job_pop()
        return set(session.submit_request.call_args.args[0].models)

    async def test_provider_absent_matches_baseline(self) -> None:
        """With no provider the advertised set is the full eligible set, as a plain popper would send."""
        baseline = await self._capture_models(provider=None, enabled=False)
        assert baseline == {"model_a", "model_b", "model_c"}

    async def test_provider_present_but_disabled_matches_baseline(self) -> None:
        """A wired-but-disabled pool advertises the same set as the provider-absent baseline."""
        baseline = await self._capture_models(provider=None, enabled=False)
        with_disabled_pool = await self._capture_models(
            provider=lambda: frozenset({"model_a"}),
            enabled=False,
        )
        assert with_disabled_pool == baseline


class TestPoolOutcomeSink:
    """A pool-routed pop reports its outcome (lane, advertised set, popped model or None) to the sink."""

    @_full_flow_patches
    async def test_empty_fixed_lane_pop_reports_none(self, _mock_req_cls: Mock) -> None:
        """An empty fixed-lane pop reports its lane and advertised set to the sink with a None popped model."""
        empty_response = Mock()
        empty_response.id_ = None
        empty_response.skipped = Mock()
        empty_response.skipped.model_dump.return_value = {}
        empty_response.skipped.model_extra = None
        empty_response.messages = None

        sink = Mock()
        session = AsyncMock()
        session.submit_request = AsyncMock(return_value=empty_response)
        popper = _make_popper(
            state=WorkerState(last_job_pop_time=0.0),
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
            bridge_data=make_mock_bridge_data(
                image_models_to_load=["model_a", "model_b"],
                model_pool=ModelPoolConfig(enabled=True),
            ),
            pool_active_seats_provider=lambda: frozenset({"model_a", "model_b"}),
            pool_pop_outcome_sink=sink,
        )

        await popper.api_job_pop()

        sink.assert_called_once()
        outcome = sink.call_args.kwargs
        assert outcome["lane"] is PopLane.FIXED
        assert outcome["advertised"] == frozenset({"model_a", "model_b"})
        assert outcome["popped_model"] is None
        assert outcome["popped_model_was_resident"] is False

    @_full_flow_patches
    async def test_popped_fixed_lane_pop_reports_model(self, _mock_req_cls: Mock) -> None:
        """A fulfilled fixed-lane pop reports the popped model name to the sink."""
        job_response = make_job_pop_response(model="model_a")
        sink = Mock()
        session = AsyncMock()
        session.submit_request = AsyncMock(return_value=job_response)
        popper = _make_popper(
            state=WorkerState(last_job_pop_time=0.0),
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
            bridge_data=make_mock_bridge_data(
                image_models_to_load=["model_a", "model_b"],
                model_pool=ModelPoolConfig(enabled=True),
            ),
            pool_active_seats_provider=lambda: frozenset({"model_a", "model_b"}),
            pool_pop_outcome_sink=sink,
        )

        await popper.api_job_pop()

        sink.assert_called_once()
        outcome = sink.call_args.kwargs
        assert outcome["lane"] is PopLane.FIXED
        assert outcome["popped_model"] == "model_a"
        assert outcome["popped_model_was_resident"] is False

    @_full_flow_patches
    async def test_popped_resident_model_reports_resident_hit(self, _mock_req_cls: Mock) -> None:
        """A model already loaded by a live inference process is reported as a resident match."""
        job_response = make_job_pop_response(model="model_a")
        sink = Mock()
        session = AsyncMock()
        session.submit_request = AsyncMock(return_value=job_response)
        process_map = _make_process_map_with_available_processes()
        process_map[0].loaded_horde_model_name = "model_a"
        popper = _make_popper(
            state=WorkerState(last_job_pop_time=0.0),
            process_map=process_map,
            horde_client_session=session,
            bridge_data=make_mock_bridge_data(
                image_models_to_load=["model_a", "model_b"],
                model_pool=ModelPoolConfig(enabled=True),
            ),
            pool_active_seats_provider=lambda: frozenset({"model_a", "model_b"}),
            pool_pop_outcome_sink=sink,
        )

        await popper.api_job_pop()

        assert sink.call_args.kwargs["popped_model_was_resident"] is True
        tally = popper.latest_pool_lane_tally()
        assert tally.fixed_resident_hits == 1


def _empty_pop_response() -> Mock:
    """A stand-in job-pop response with no job (``id_`` None) that the full-flow parse accepts."""
    response = Mock()
    response.id_ = None
    response.skipped = Mock()
    response.skipped.model_dump.return_value = {}
    response.skipped.model_extra = None
    response.messages = None
    return response


class TestPoolLaneTally:
    """A pool-routed pop advances the cumulative per-lane pop/fulfillment tally the status snapshot reads."""

    @_full_flow_patches
    async def test_fulfilled_fixed_pop_counts_pop_and_fulfillment(self, _mock_req_cls: Mock) -> None:
        """A fixed-lane pop that returns a job counts one fixed pop and one fixed fulfillment."""
        session = AsyncMock()
        session.submit_request = AsyncMock(return_value=make_job_pop_response(model="model_a"))
        popper = _make_popper(
            state=WorkerState(last_job_pop_time=0.0),
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
            bridge_data=make_mock_bridge_data(
                image_models_to_load=["model_a", "model_b"],
                model_pool=ModelPoolConfig(enabled=True),
            ),
            pool_active_seats_provider=lambda: frozenset({"model_a", "model_b"}),
        )

        await popper.api_job_pop()

        tally = popper.latest_pool_lane_tally()
        assert (tally.fixed_pops, tally.fixed_fulfilled) == (1, 1)
        assert (tally.free_pops, tally.free_fulfilled) == (0, 0)

    @_full_flow_patches
    async def test_empty_fixed_pop_counts_pop_without_fulfillment(self, _mock_req_cls: Mock) -> None:
        """A fixed-lane pop that comes back empty counts the pop but not a fulfillment."""
        session = AsyncMock()
        session.submit_request = AsyncMock(return_value=_empty_pop_response())
        popper = _make_popper(
            state=WorkerState(last_job_pop_time=0.0),
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
            bridge_data=make_mock_bridge_data(
                image_models_to_load=["model_a", "model_b"],
                model_pool=ModelPoolConfig(enabled=True),
            ),
            pool_active_seats_provider=lambda: frozenset({"model_a", "model_b"}),
        )

        await popper.api_job_pop()

        tally = popper.latest_pool_lane_tally()
        assert (tally.fixed_pops, tally.fixed_fulfilled) == (1, 0)
        assert (tally.free_pops, tally.free_fulfilled) == (0, 0)

    @_full_flow_patches
    async def test_free_lane_pop_counts_under_free_lane(self, _mock_req_cls: Mock) -> None:
        """A free-lane pop is tallied under the free lane, leaving the fixed-lane counts untouched."""
        session = AsyncMock()
        session.submit_request = AsyncMock(return_value=_empty_pop_response())
        popper = _make_popper(
            state=WorkerState(last_job_pop_time=0.0),
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
            bridge_data=make_mock_bridge_data(
                image_models_to_load=["model_a", "model_b", "model_c"],
                model_pool=ModelPoolConfig(enabled=True),
            ),
            pool_active_seats_provider=lambda: frozenset({"model_a"}),
        )
        # A seated model that is a strict subset of the eligible set leaves a non-empty free offer; the
        # weighted round-robin then picks the free lane when its credit dominates.
        popper._pool_lane_state = PoolLaneState(free_credit=1000)

        await popper.api_job_pop()

        assert popper.latest_pool_lane() is PopLane.FREE
        tally = popper.latest_pool_lane_tally()
        assert (tally.free_pops, tally.free_fulfilled) == (1, 0)
        assert (tally.fixed_pops, tally.fixed_fulfilled) == (0, 0)


# endregion

# region Bounded awaits on the pop path


_POP_TIMEOUT_CONST = "horde_worker_regen.process_management.jobs.job_popper.POP_REQUEST_TIMEOUT_SECONDS"
_SOURCE_IMAGE_TIMEOUT_CONST = (
    "horde_worker_regen.process_management.jobs.job_popper.SOURCE_IMAGE_DOWNLOAD_TIMEOUT_SECONDS"
)

# The single pop coroutine must never block indefinitely: every await it performs is bounded, so one
# unresponsive server cannot silence the worker's only intake path.
_TEST_AWAIT_BOUND_SECONDS = 5.0
"""Outer bound the tests hold api_job_pop to. Generous relative to the patched-in ceilings, so a failure
here means the await was unbounded rather than merely slow."""


async def _hang_forever(*_args: object, **_kwargs: object) -> object:
    """Await that never completes, standing in for an unresponsive peer."""
    await asyncio.Event().wait()
    raise AssertionError("unreachable")  # pragma: no cover


def _capture_logs(level: str = "WARNING") -> tuple[list[str], int]:
    """Attach a loguru sink collecting messages at or above ``level``."""
    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(message.record["message"]), level=level)
    return lines, sink_id


class TestPopPathBoundedAwaits:
    """The pop coroutine's network awaits are individually bounded."""

    def _make_popper_with_session(self, session: AsyncMock, job_tracker: JobTracker | None = None) -> JobPopper:
        """Create a popper past every guard clause, backed by ``session``."""
        return _make_popper(
            state=WorkerState(last_job_pop_time=0.0),
            process_map=_make_process_map_with_available_processes(),
            job_tracker=job_tracker if job_tracker is not None else JobTracker(),
            horde_client_session=session,
        )

    @_full_flow_patches
    async def test_hanging_pop_request_is_bounded(self, _mock_req_cls: Mock) -> None:
        """A pop request that never answers is abandoned and handled as a pop error."""
        session = AsyncMock()
        session.submit_request = _hang_forever
        popper = self._make_popper_with_session(session)

        lines, sink_id = _capture_logs()
        try:
            with patch(_POP_TIMEOUT_CONST, 0.05):
                await asyncio.wait_for(popper.api_job_pop(), timeout=_TEST_AWAIT_BOUND_SECONDS)
        finally:
            logger.remove(sink_id)

        assert popper._pop_throttler.current_pop_frequency == popper._pop_throttler._error_pop_frequency
        assert any("TimeoutError" in line for line in lines), lines

    @_full_flow_patches
    async def test_bare_timeout_error_logs_its_type(self, _mock_req_cls: Mock) -> None:
        """``str(TimeoutError())`` is empty, so the failure log must name the exception type."""
        session = AsyncMock()
        session.submit_request = AsyncMock(side_effect=TimeoutError())
        popper = self._make_popper_with_session(session)

        lines, sink_id = _capture_logs()
        try:
            await popper.api_job_pop()
        finally:
            logger.remove(sink_id)

        assert any("Failed to pop job" in line and "TimeoutError" in line for line in lines), lines

    @_full_flow_patches
    async def test_non_timeout_exception_logs_type_and_message(self, _mock_req_cls: Mock) -> None:
        """Ordinary pop failures keep their message and gain the exception type."""
        session = AsyncMock()
        session.submit_request = AsyncMock(side_effect=ConnectionError("network down"))
        popper = self._make_popper_with_session(session)

        lines, sink_id = _capture_logs()
        try:
            await popper.api_job_pop()
        finally:
            logger.remove(sink_id)

        assert any("ConnectionError" in line and "network down" in line for line in lines), lines

    @_full_flow_patches
    async def test_fast_successful_pop_is_unaffected(self, _mock_req_cls: Mock) -> None:
        """The bound is inert for a prompt response."""
        session = AsyncMock()
        session.submit_request = AsyncMock(return_value=make_job_pop_response())
        popper = self._make_popper_with_session(session)

        await asyncio.wait_for(popper.api_job_pop(), timeout=_TEST_AWAIT_BOUND_SECONDS)

        assert len(popper._job_tracker.jobs_pending_inference) == 1
        assert popper._pop_throttler.current_pop_frequency == popper._pop_throttler._default_pop_frequency

    @_full_flow_patches
    async def test_slow_pop_under_the_bound_succeeds(self, _mock_req_cls: Mock) -> None:
        """A response that arrives inside the ceiling is served normally."""
        response = make_job_pop_response()

        async def _slow_submit(*_args: object, **_kwargs: object) -> ImageGenerateJobPopResponse:
            await asyncio.sleep(0.05)
            return response

        session = AsyncMock()
        session.submit_request = _slow_submit
        popper = self._make_popper_with_session(session)

        with patch(_POP_TIMEOUT_CONST, 2.0):
            await asyncio.wait_for(popper.api_job_pop(), timeout=_TEST_AWAIT_BOUND_SECONDS)

        assert len(popper._job_tracker.jobs_pending_inference) == 1
        assert popper._pop_throttler.current_pop_frequency == popper._pop_throttler._default_pop_frequency

    @_full_flow_patches
    async def test_hanging_source_image_download_faults_the_job(self, _mock_req_cls: Mock) -> None:
        """An unanswered source-image download is abandoned and faulted like an exhausted retry loop."""
        response = make_job_pop_response().model_copy(
            update={"source_image": "https://example.invalid/source.webp"},
        )
        session = AsyncMock()
        session.submit_request = AsyncMock(return_value=response)
        job_tracker = JobTracker()
        popper = self._make_popper_with_session(session, job_tracker=job_tracker)
        popper._source_image_downloader.download_source_images = _hang_forever  # pyrefly: ignore - test double

        with patch(_SOURCE_IMAGE_TIMEOUT_CONST, 0.05):
            await asyncio.wait_for(popper.api_job_pop(), timeout=_TEST_AWAIT_BOUND_SECONDS)

        assert response.id_ is not None
        faults = await job_tracker.get_faults_for_job(response.id_)
        assert [fault.type_ for fault in faults] == [METADATA_TYPE.source_image]
        assert faults[0].value == METADATA_VALUE.download_failed
        # The job is still queued: a missing source image is reported as a fault at submit time, exactly as
        # it is when the downloader's own retries are exhausted.
        assert len(job_tracker.jobs_pending_inference) == 1


# endregion


# region Pop gate disclosure

_SELECT_MODELS_PATH = "horde_worker_regen.process_management.jobs.job_popper._select_models_for_pop"


class TestPopGateStamping:
    """Every silent early return in the pop coroutine names the gate that is holding pops.

    The coroutine has many gates that can hold indefinitely with no log line of their own, so a worker held
    at one of them is indistinguishable from a worker the horde simply has no work for. Recording the gate
    (and when it took hold) is what lets the liveness sentinel and the operator surfaces say which condition
    the worker is waiting on.
    """

    def _popper_past_the_early_gates(
        self,
        *,
        process_map: ProcessMap | None = None,
        job_tracker: JobTracker | None = None,
    ) -> JobPopper:
        """Build a popper whose posture, resources, and throttler let the flow reach the later gates."""
        return _make_popper(
            state=WorkerState(last_job_pop_time=0.0),
            process_map=process_map if process_map is not None else _make_process_map_with_available_processes(),
            job_tracker=job_tracker if job_tracker is not None else JobTracker(),
        )

    async def test_no_inference_process_is_stamped(self) -> None:
        """The gate that wedges a pool reloading every slot at once."""
        process_map = ProcessMap(
            {
                10: make_mock_process_info(
                    10,
                    model_name=None,
                    state=HordeProcessState.WAITING_FOR_JOB,
                    process_type=HordeProcessType.SAFETY,
                ),
                0: make_mock_process_info(0, model_name=None, state=HordeProcessState.PROCESS_STARTING),
            },  # type: ignore[arg-type]
        )
        popper = self._popper_past_the_early_gates(process_map=process_map)

        await popper.api_job_pop()

        assert popper._state.last_pop_gate == "no_inference_process"
        assert popper._state.last_pop_gate_since > 0.0

    async def test_no_safety_process_is_stamped(self) -> None:
        """A pool with no safety process to hand results to holds pops just as silently."""
        process_map = ProcessMap(
            {0: make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)},  # type: ignore[arg-type]
        )
        popper = self._popper_past_the_early_gates(process_map=process_map)

        await popper.api_job_pop()

        assert popper._state.last_pop_gate == "no_safety_process"

    async def test_queue_full_is_stamped(self) -> None:
        """A full local queue is a normal, healthy hold, and still has to be attributable."""
        job_tracker = JobTracker()
        for index in range(4):
            await job_tracker.record_popped_job(make_job_pop_response(model=f"queued_{index}"))
        popper = self._popper_past_the_early_gates(job_tracker=job_tracker)

        await popper.api_job_pop()

        assert popper._state.last_pop_gate == "queue_full"

    async def test_warmup_first_job_is_stamped(self) -> None:
        """Queueing ahead is withheld until the session first job completes."""
        job_tracker = JobTracker()
        await job_tracker.record_popped_job(make_job_pop_response(model="first"))
        popper = self._popper_past_the_early_gates(job_tracker=job_tracker)

        await popper.api_job_pop()

        assert popper._state.last_pop_gate == "warmup_first_job"

    async def test_megapixelstep_wait_is_stamped(self) -> None:
        """The megapixelstep governor holds pops while large in-flight work drains."""
        popper = self._popper_past_the_early_gates()
        popper._pop_throttler.should_wait_for_megapixelsteps = Mock(return_value=True)  # type: ignore[method-assign]

        await popper.api_job_pop()

        assert popper._state.last_pop_gate == "megapixelstep_wait"

    async def test_no_eligible_models_is_stamped(self) -> None:
        """Model selection returning nothing servable is a hold, not an attempt."""
        popper = self._popper_past_the_early_gates()

        with patch(_SELECT_MODELS_PATH, return_value=None):
            await popper.api_job_pop()

        assert popper._state.last_pop_gate == "no_eligible_models"

    async def test_large_model_limits_emptying_the_offer_is_stamped(self) -> None:
        """The large-model switch and cooldown limits can empty the offer, ending the cycle silently."""
        popper = self._popper_past_the_early_gates()
        popper._apply_large_model_pop_limits = Mock(return_value=set())  # type: ignore[method-assign]

        await popper.api_job_pop()

        assert popper._state.last_pop_gate == "large_model_limits"

    async def test_an_offer_emptied_by_narrowing_is_stamped_and_never_sent(self) -> None:
        """An empty offer must end the cycle at the floor, stamped and unsent.

        The server matches an empty model list to unconstrained requests and can answer with a job carrying
        no model name, which this worker would then reject and fault.
        """
        popper = self._popper_past_the_early_gates()
        popper._apply_residency_advertising_bias = Mock(return_value=set())  # type: ignore[method-assign]

        await popper.api_job_pop()

        assert popper._state.last_pop_gate == "empty_offer"
        assert popper._state.last_pop_attempt_completed_at == 0.0

    async def test_the_gate_stamp_is_only_moved_when_the_gate_changes(self) -> None:
        """The stamp measures how long this gate has held, so a repeat tick must not refresh it."""
        popper = self._popper_past_the_early_gates()
        popper._pop_throttler.should_wait_for_megapixelsteps = Mock(return_value=True)  # type: ignore[method-assign]

        await popper.api_job_pop()
        first_stamp = popper._state.last_pop_gate_since
        await popper.api_job_pop()

        assert popper._state.last_pop_gate_since == first_stamp

    @_full_flow_patches
    async def test_a_completed_pop_clears_the_gate_and_stamps_the_attempt(self, _mock_req_cls: Mock) -> None:
        """Reaching the horde is the all-clear: no gate holds, and the attempt time is recorded."""
        state = WorkerState(last_job_pop_time=0.0, last_pop_gate="no_inference_process")
        session = AsyncMock()
        session.submit_request = AsyncMock(return_value=make_job_pop_response())
        popper = _make_popper(
            state=state,
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
        )

        await popper.api_job_pop()

        assert state.last_pop_gate is None
        assert state.last_pop_attempt_completed_at > 0.0

    @_full_flow_patches
    async def test_a_failed_pop_still_counts_as_a_completed_attempt(self, _mock_req_cls: Mock) -> None:
        """A request that raised still proves the popper is reaching (and hearing from) the network."""
        session = AsyncMock()
        session.submit_request = AsyncMock(side_effect=RuntimeError("boom"))
        popper = _make_popper(
            state=WorkerState(last_job_pop_time=0.0),
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
        )

        await popper.api_job_pop()

        assert popper._state.last_pop_attempt_completed_at > 0.0

    @_full_flow_patches
    async def test_an_error_response_counts_as_a_completed_attempt(self, _mock_req_cls: Mock) -> None:
        """An API error response is an answer from the horde, so the attempt concluded."""
        session = AsyncMock()
        session.submit_request = AsyncMock(
            return_value=RequestErrorResponse(message="Wrong credentials"),
        )
        popper = _make_popper(
            state=WorkerState(last_job_pop_time=0.0),
            process_map=_make_process_map_with_available_processes(),
            horde_client_session=session,
        )

        await popper.api_job_pop()

        assert popper._state.last_pop_gate is None
        assert popper._state.last_pop_attempt_completed_at > 0.0

    async def test_the_dry_run_path_also_stamps_a_completed_attempt(self) -> None:
        """The simulated boundary stands in for the horde, so a dry run is not a silent worker."""
        source = Mock(spec=CannedJobSource)
        source.next_pop_response.return_value = make_empty_pop_response()
        popper = _make_popper(
            state=WorkerState(last_job_pop_time=0.0),
            process_map=_make_process_map_with_available_processes(),
            dry_run_skip_api=True,
        )
        popper.set_canned_job_source(source)

        await popper.api_job_pop()

        assert popper._state.last_pop_gate is None
        assert popper._state.last_pop_attempt_completed_at > 0.0

    async def test_the_popper_start_seeds_the_attempt_stamp(self) -> None:
        """A worker that never completes an attempt measures its silence from when the loop started."""
        state = WorkerState()
        shutdown_manager = Mock()
        shutdown_manager.is_time_for_shutdown.return_value = True
        popper = _make_popper(state=state, shutdown_manager=shutdown_manager)

        await popper.run()

        assert state.last_pop_attempt_completed_at > 0.0


# endregion


class TestIntakeBudget:
    """The intake cap (running plus buffered jobs) is per card, summed across every driven card.

    ``jobs_pending_inference`` includes in-progress jobs, so a worker-wide cap sized for one card lets a
    single running job stop the intake that feeds its siblings: on a 4-card host the original flat
    ``queue_size + max_threads`` paused pops with two jobs in flight while two cards idled.
    """

    def test_no_card_plan_keeps_the_single_card_formula(self) -> None:
        """Without a card plan, the budget is exactly the original queue_size + max_threads."""
        for queue_size, max_threads in [(0, 1), (1, 1), (1, 2), (2, 2)]:
            bridge_data = make_mock_bridge_data(queue_size=queue_size, max_threads=max_threads)
            popper = _make_popper(bridge_data=bridge_data)
            assert popper._intake_budget(bridge_data) == queue_size + max_threads

    def test_single_card_plan_matches_the_no_plan_formula(self) -> None:
        """One driven card with no overrides is byte-identical to the legacy worker-wide cap."""
        bridge_data = make_mock_bridge_data(queue_size=1, max_threads=1)
        popper = _make_popper(
            bridge_data=bridge_data,
            card_runtimes=make_test_card_runtimes(device_indices=(0,), config=bridge_data),
        )
        assert popper._intake_budget(bridge_data) == 2

    def test_four_cards_sum_their_budgets(self) -> None:
        """Four cards at queue_size 1 / max_threads 1 hold 8 jobs, not 2."""
        bridge_data = make_mock_bridge_data(queue_size=1, max_threads=1)
        popper = _make_popper(
            bridge_data=bridge_data,
            card_runtimes=make_test_card_runtimes(device_indices=(0, 1, 2, 3), config=bridge_data),
        )
        assert popper._intake_budget(bridge_data) == 8

    def test_per_card_effective_configs_shape_each_contribution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each card contributes its own effective (override-applied) budget, not the global one."""
        import horde_worker_regen.process_management.jobs.job_popper as job_popper_module

        bridge_data = make_mock_bridge_data(queue_size=1, max_threads=1)
        card0 = make_mock_bridge_data(queue_size=1, max_threads=1)
        card1 = make_mock_bridge_data(queue_size=2, max_threads=2)
        monkeypatch.setattr(
            job_popper_module,
            "resolve_all_effective_gpu_configs",
            lambda _base, _indices: {0: card0, 1: card1},
        )
        popper = _make_popper(
            bridge_data=bridge_data,
            card_runtimes=make_test_card_runtimes(device_indices=(0, 1), config=bridge_data),
        )
        assert popper._intake_budget(bridge_data) == 6

    def test_budget_follows_a_config_snapshot_swap(self) -> None:
        """A new config snapshot recomputes the budget; the cache is per snapshot, not forever."""
        first = make_mock_bridge_data(queue_size=1, max_threads=1)
        popper = _make_popper(
            bridge_data=first,
            card_runtimes=make_test_card_runtimes(device_indices=(0, 1), config=first),
        )
        assert popper._intake_budget(first) == 4
        second = make_mock_bridge_data(queue_size=2, max_threads=1)
        assert popper._intake_budget(second) == 6

    async def test_queue_full_opens_one_slot_per_card_budget(self) -> None:
        """_is_queue_full trips at the summed budget: 7 held jobs fit a 4-card budget of 8, 8 do not."""
        bridge_data = make_mock_bridge_data(queue_size=1, max_threads=1)
        job_tracker = JobTracker()
        popper = _make_popper(
            bridge_data=bridge_data,
            job_tracker=job_tracker,
            card_runtimes=make_test_card_runtimes(device_indices=(0, 1, 2, 3), config=bridge_data),
        )
        for index in range(7):
            await job_tracker.record_popped_job(make_job_pop_response(model=f"held_{index}"))
        assert popper._is_queue_full(bridge_data) is False
        await job_tracker.record_popped_job(make_job_pop_response(model="held_7"))
        assert popper._is_queue_full(bridge_data) is True
