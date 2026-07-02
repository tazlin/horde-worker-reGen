"""Tests for JobPopper orchestration logic.

Tests for individual extracted components (PopThrottler, SourceImageDownloader,
_select_models_for_pop, APIWorkerMessage) live in their own test modules.
These tests focus on how JobPopper coordinates those components and the
higher-level api_job_pop flow.
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import AsyncMock, Mock, patch

from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopRequest, LorasPayloadEntry

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadPhase,
    DownloadStatusSnapshot,
)
from horde_worker_regen.process_management.jobs.job_popper import JobPopper
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.model_availability import ModelAvailability
from horde_worker_regen.process_management.scheduling.pop_throttler import CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS
from horde_worker_regen.utils.job_utils import line_skip_pop_max_power
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_job,
    make_mock_process_info,
    make_test_api_sessions,
    make_test_runtime_config,
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

    async def test_no_models_configured_returns_early(self) -> None:
        """Empty model list should prevent pops (with a sleep penalty)."""
        process_map = _make_process_map_with_available_processes()
        popper = _make_popper(process_map=process_map, image_models_to_load=[])
        await popper.api_job_pop()

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

        await popper.api_job_pop()

        assert popper._state.last_pop_maintenance_mode is False  # pyrefly: ignore - "always true" is wrong, api_job_pop() should mutate
        assert popper._state.server_maintenance_cleared_by_job_pop is True

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


# region Aux-download line-skip aggressive pop


async def _seed_completed_session(job_tracker: JobTracker) -> None:
    """Put one completed job behind us so the per-session warm-up gate no longer blocks a queue-ahead pop."""
    await track_popped_job_async(job_tracker, make_mock_job())
    await job_tracker.increment_jobs_completed()


async def _queue_head_model_jobs(job_tracker: JobTracker, n: int) -> None:
    """Queue ``n`` pending-inference jobs for the head model.

    Two or more of the same model trip the "one running plus one queued" per-model cap in
    ``_select_models_for_pop``, standing in for the real line-skip situation where the queue is saturated
    with jobs that all share (and so all block behind) the aux-download-stalled head model.
    """
    for _ in range(n):
        await track_popped_job_async(job_tracker, make_mock_job(model="stable_diffusion"))


def _make_line_skip_popper(
    *,
    state: WorkerState,
    job_tracker: JobTracker,
    process_map: ProcessMap | None = None,
    bridge_data: Mock | None = None,
) -> tuple[JobPopper, Mock]:
    """Build a popper wired for line-skip pop scenarios plus the API session mock it will call.

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


def _armed_state() -> WorkerState:
    """A worker state with the aux-download line-skip breaker armed."""
    state = WorkerState()
    state.wants_line_skip_candidate = True
    return state


class TestAuxDownloadLineSkipPopBias:
    """When the scheduler arms ``wants_line_skip_candidate`` the pop biases toward a small non-LoRA job.

    The scheduler sets the flag when a slot is blocked downloading auxiliary models past
    ``aux_model_download_line_skip_threshold_seconds`` and nothing already queued can slip past the blocked
    head. Biasing the next pop small and non-LoRA lets an idle sibling process pick up skippable work so
    the GPU keeps sampling while the download finishes.
    """

    @staticmethod
    async def _pop_and_capture_request(
        *,
        state: WorkerState,
        **bridge_overrides: object,
    ) -> ImageGenerateJobPopRequest:
        """Drive one full pop with the given state and bridge overrides, returning the built pop request."""
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
            bridge_data=make_mock_bridge_data(**bridge_overrides),
        )

        await popper.api_job_pop()

        session.submit_request.assert_awaited_once()
        request: ImageGenerateJobPopRequest = session.submit_request.call_args.args[0]
        return request

    async def test_armed_flag_withholds_lora(self) -> None:
        """An armed flag stops the pop advertising LoRA support even when the config allows it."""
        request = await self._pop_and_capture_request(state=_armed_state(), allow_lora=True)

        assert request.allow_lora is False

    async def test_armed_flag_caps_oversized_max_power(self) -> None:
        """An armed flag caps a large configured max_power down to the line-skip resolution ceiling."""
        request = await self._pop_and_capture_request(state=_armed_state(), max_power=64)

        ceiling = line_skip_pop_max_power(high_performance_mode=False, moderate_performance_mode=False)
        assert request.max_pixels == ceiling * 8 * 64 * 64

    async def test_armed_flag_caps_extreme_max_power(self) -> None:
        """Even a pathologically large configured max_power is pulled down to the default-mode ceiling."""
        request = await self._pop_and_capture_request(state=_armed_state(), max_power=512)

        ceiling = line_skip_pop_max_power(high_performance_mode=False, moderate_performance_mode=False)
        assert request.max_pixels == ceiling * 8 * 64 * 64

    async def test_armed_flag_ceiling_widens_in_high_performance_mode(self) -> None:
        """The cap tracks the perf mode: high mode admits a larger skip job than the default mode."""
        request = await self._pop_and_capture_request(
            state=_armed_state(),
            max_power=512,
            high_performance_mode=True,
        )

        ceiling = line_skip_pop_max_power(high_performance_mode=True, moderate_performance_mode=False)
        default_ceiling = line_skip_pop_max_power(high_performance_mode=False, moderate_performance_mode=False)
        assert ceiling > default_ceiling  # guards the premise: high mode really is a wider ceiling
        assert request.max_pixels == ceiling * 8 * 64 * 64

    async def test_armed_flag_ceiling_widens_in_moderate_performance_mode(self) -> None:
        """Moderate mode sits between default and high: its ceiling is wider than default, narrower than high."""
        request = await self._pop_and_capture_request(
            state=_armed_state(),
            max_power=512,
            moderate_performance_mode=True,
        )

        ceiling = line_skip_pop_max_power(high_performance_mode=False, moderate_performance_mode=True)
        assert request.max_pixels == ceiling * 8 * 64 * 64

    async def test_armed_flag_does_not_enlarge_a_small_max_power(self) -> None:
        """A worker already configured for small jobs keeps its own (smaller) max_power under the bias."""
        request = await self._pop_and_capture_request(state=_armed_state(), max_power=8)

        assert request.max_pixels == 8 * 8 * 64 * 64

    async def test_armed_flag_withholds_lora_even_with_room_in_the_lora_queue(self) -> None:
        """LoRA is withheld purely because the flag is armed, independent of the per-queue LoRA cap.

        The queue here is empty, so the LoRA-cap suppression path is not what clears LoRA; the armed flag is.
        """
        request = await self._pop_and_capture_request(state=_armed_state(), allow_lora=True)

        assert request.allow_lora is False

    async def test_unarmed_flag_leaves_pop_unbiased(self) -> None:
        """With the flag clear (the default), LoRA support and max_power are advertised unchanged."""
        request = await self._pop_and_capture_request(state=WorkerState(), allow_lora=True, max_power=64)

        assert request.allow_lora is True
        assert request.max_pixels == 64 * 8 * 64 * 64


class TestAuxDownloadLineSkipGateRelaxation:
    """An armed flag relaxes the steady-state throughput/pacing governors for that one pop.

    The situation that arms the flag is a *full* local queue whose head is stalled, so the popper's normal
    depth and cadence gates would refuse the pop before the bias could take effect. Because a skip job is
    expected to dispatch onto the idle sibling immediately rather than buffer, the flag lets the pop slip
    past those gates; each relaxation is paired with an unarmed control proving the gate is otherwise honoured.
    """

    async def test_armed_pops_through_full_queue(self) -> None:
        """A queue at the configured depth does not block the skip pop (one extra slot is admitted)."""
        job_tracker = JobTracker()
        await _queue_head_model_jobs(job_tracker, 2)  # queue_size=1, max_threads=1 -> depth cap is 2
        await job_tracker.increment_jobs_completed()
        popper, session = _make_line_skip_popper(state=_armed_state(), job_tracker=job_tracker)

        await popper.api_job_pop()

        session.submit_request.assert_awaited_once()

    async def test_full_queue_blocks_pop_when_unarmed(self) -> None:
        """The same full queue blocks the pop when the flag is clear (the depth cap is otherwise honoured)."""
        job_tracker = JobTracker()
        await _queue_head_model_jobs(job_tracker, 2)
        await job_tracker.increment_jobs_completed()
        popper, session = _make_line_skip_popper(state=WorkerState(), job_tracker=job_tracker)

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_armed_full_queue_pop_is_still_biased(self) -> None:
        """The pop that slips past a full queue is itself the biased one: LoRA withheld and max_power capped."""
        job_tracker = JobTracker()
        await _queue_head_model_jobs(job_tracker, 2)
        await job_tracker.increment_jobs_completed()
        bridge = make_mock_bridge_data(
            image_models_to_load=["stable_diffusion", "sibling_model"],
            allow_lora=True,
            max_power=64,
        )
        popper, session = _make_line_skip_popper(state=_armed_state(), job_tracker=job_tracker, bridge_data=bridge)

        await popper.api_job_pop()

        session.submit_request.assert_awaited_once()
        request: ImageGenerateJobPopRequest = session.submit_request.call_args.args[0]
        ceiling = line_skip_pop_max_power(high_performance_mode=False, moderate_performance_mode=False)
        assert request.allow_lora is False
        assert request.max_pixels == ceiling * 8 * 64 * 64

    async def test_armed_pops_through_megapixelstep_wait(self) -> None:
        """An armed flag bypasses the megapixelstep governor so the small skip job is not held behind it."""
        job_tracker = JobTracker()
        await _seed_completed_session(job_tracker)
        popper, session = _make_line_skip_popper(state=_armed_state(), job_tracker=job_tracker)
        popper._pop_throttler.should_wait_for_megapixelsteps = Mock(return_value=True)  # type: ignore[method-assign]

        await popper.api_job_pop()

        session.submit_request.assert_awaited_once()

    async def test_megapixelstep_wait_blocks_pop_when_unarmed(self) -> None:
        """With the flag clear, the megapixelstep governor still holds the pop."""
        job_tracker = JobTracker()
        await _seed_completed_session(job_tracker)
        popper, session = _make_line_skip_popper(state=WorkerState(), job_tracker=job_tracker)
        popper._pop_throttler.should_wait_for_megapixelsteps = Mock(return_value=True)  # type: ignore[method-assign]

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_armed_pops_through_frequency_window(self) -> None:
        """An armed flag makes the pop urgent, so the inter-pop cadence window does not hold it back."""
        job_tracker = JobTracker()
        await _seed_completed_session(job_tracker)
        state = _armed_state()
        state.last_job_pop_time = time.time()  # a non-urgent pop would be gated as "too soon"
        popper, session = _make_line_skip_popper(state=state, job_tracker=job_tracker)

        await popper.api_job_pop(urgent=False)  # caller does not mark it urgent; the armed flag must

        session.submit_request.assert_awaited_once()

    async def test_frequency_window_blocks_pop_when_unarmed(self) -> None:
        """With the flag clear, a non-urgent pop inside the cadence window is skipped."""
        job_tracker = JobTracker()
        await _seed_completed_session(job_tracker)
        state = WorkerState(last_job_pop_time=time.time())
        popper, session = _make_line_skip_popper(state=state, job_tracker=job_tracker)

        await popper.api_job_pop(urgent=False)

        session.submit_request.assert_not_awaited()


class TestAuxDownloadLineSkipProtectiveGatesStillApply:
    """The aggressive pop relaxes throughput/pacing governors only; genuinely protective gates still block.

    Each case arms the flag on a popper that would otherwise pop, then engages one protective gate and
    asserts no pop is attempted. This locks in that the fix does not blanket-override every guard: it must
    not start a job the degraded worker cannot promptly serve, nor pop during a hard stop.
    """

    async def _armed_popper(
        self,
        *,
        state: WorkerState | None = None,
        process_map: ProcessMap | None = None,
    ) -> tuple[JobPopper, Mock]:
        """An armed popper seeded so that, absent the gate under test, a pop would proceed."""
        if state is None:
            state = _armed_state()
        else:
            state.wants_line_skip_candidate = True
        job_tracker = JobTracker()
        await _seed_completed_session(job_tracker)
        return _make_line_skip_popper(state=state, job_tracker=job_tracker, process_map=process_map)

    async def test_shutting_down_still_blocks(self) -> None:
        """A shutdown in progress blocks even an armed pop."""
        popper, session = await self._armed_popper(state=WorkerState(shutting_down=True))

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_supervisor_pause_still_blocks(self) -> None:
        """An operator/supervisor pause blocks even an armed pop."""
        popper, session = await self._armed_popper(state=WorkerState(supervisor_paused=True))

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_self_throttle_pause_still_blocks(self) -> None:
        """The worker's own self-throttle pause blocks even an armed pop."""
        popper, session = await self._armed_popper(state=WorkerState(self_throttle_paused=True))

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_ram_pressure_hold_still_blocks(self) -> None:
        """The pre-floor RAM-pressure hold blocks even an armed pop: no new ttl clock under RAM danger."""
        popper, session = await self._armed_popper(state=WorkerState(ram_pressure_pop_hold=True))

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_downloads_only_hold_still_blocks(self) -> None:
        """The pre-GO_LIVE download-only posture blocks even an armed pop."""
        popper, session = await self._armed_popper(state=WorkerState(downloads_only_hold=True))

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_gpu_torch_incompatible_still_blocks(self) -> None:
        """A torch/GPU mismatch (every job would fail at kernel launch) blocks even an armed pop."""
        popper, session = await self._armed_popper(state=WorkerState(gpu_torch_incompatible=True))

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_cpu_only_torch_build_still_blocks(self) -> None:
        """A CPU-only torch build (image generation disabled) blocks even an armed pop."""
        popper, session = await self._armed_popper(state=WorkerState(torch_build_cpu_only=True))

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_consecutive_failure_pause_still_blocks(self) -> None:
        """An active consecutive-failure pause blocks even an armed pop."""
        state = WorkerState(
            too_many_consecutive_failed_jobs=True,
            too_many_consecutive_failed_jobs_time=time.time(),
        )
        popper, session = await self._armed_popper(state=state)

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_post_inference_backlog_still_blocks(self) -> None:
        """A safety-stage backlog blocks even an armed pop: the skip job would only age out behind it."""
        popper, session = await self._armed_popper()
        popper._is_post_inference_backlogged = Mock(return_value=True)  # type: ignore[method-assign]

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_no_free_inference_process_still_blocks(self) -> None:
        """With no inference process free to take a job, an armed pop does not proceed."""
        busy_inference = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.INFERENCE_STARTING,
        )
        safety = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        popper, session = await self._armed_popper(process_map=ProcessMap({0: busy_inference, 10: safety}))

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_no_free_safety_process_still_blocks(self) -> None:
        """With no safety process available, an armed pop does not proceed."""
        popper, session = await self._armed_popper(
            process_map=_make_process_map_with_available_processes(num_safety=0),
        )

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()


class TestAuxDownloadLineSkipRelaxationBounds:
    """The relaxation is bounded: it admits one extra job and never conjures a pop from nothing."""

    async def test_queue_allowance_is_exactly_one_slot(self) -> None:
        """One over the cap still blocks even when armed, so intake cannot run away tick after tick."""
        job_tracker = JobTracker()
        await _queue_head_model_jobs(job_tracker, 3)  # cap is 2; one extra is allowed, so 3 is over the line
        await job_tracker.increment_jobs_completed()
        popper, session = _make_line_skip_popper(state=_armed_state(), job_tracker=job_tracker)

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_allowance_rides_on_the_threads_adjusted_cap(self) -> None:
        """The extra slot is relative to the true depth cap, which widens with max_threads (not an absolute)."""
        # queue_size=1, max_threads=3 -> depth cap is 1 + 1 + (3 - 1) = 4; the armed allowance makes 5.
        bridge = make_mock_bridge_data(
            image_models_to_load=["stable_diffusion", "sibling_model"],
            max_threads=3,
        )
        job_tracker = JobTracker()
        await _queue_head_model_jobs(job_tracker, 4)  # exactly at the (threads-adjusted) cap
        await job_tracker.increment_jobs_completed()
        popper, session = _make_line_skip_popper(state=_armed_state(), job_tracker=job_tracker, bridge_data=bridge)

        await popper.api_job_pop()

        session.submit_request.assert_awaited_once()

    async def test_allowance_still_bounded_on_a_wide_cap(self) -> None:
        """One past the threads-adjusted cap blocks even when armed, matching the one-slot allowance."""
        bridge = make_mock_bridge_data(
            image_models_to_load=["stable_diffusion", "sibling_model"],
            max_threads=3,
        )
        job_tracker = JobTracker()
        await _queue_head_model_jobs(job_tracker, 5)  # cap 4 + one allowed slot -> 5 is over
        await job_tracker.increment_jobs_completed()
        popper, session = _make_line_skip_popper(state=_armed_state(), job_tracker=job_tracker, bridge_data=bridge)

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()

    async def test_no_eligible_model_still_blocks_when_armed(self) -> None:
        """An armed flag does not force a pop when nothing is offerable: it cannot invent a skippable job.

        Only the head model is configured and it is over-queued (so ``_select_models_for_pop`` drops it),
        while a deep queue keeps the depth gate from firing first. With no other model to offer, the pop is
        correctly withheld despite the armed flag.
        """
        bridge = make_mock_bridge_data(
            image_models_to_load=["stable_diffusion"],  # no sibling model to fall back to
            queue_size=6,  # keep the depth gate from being the thing that blocks
        )
        job_tracker = JobTracker()
        await _queue_head_model_jobs(job_tracker, 2)  # trips the per-model "one running plus one queued" cap
        await job_tracker.increment_jobs_completed()
        popper, session = _make_line_skip_popper(state=_armed_state(), job_tracker=job_tracker, bridge_data=bridge)

        await popper.api_job_pop()

        session.submit_request.assert_not_awaited()


# endregion
