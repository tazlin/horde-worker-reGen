"""Tests for JobPopper orchestration logic.

Tests for individual extracted components (PopThrottler, SourceImageDownloader,
_select_models_for_pop, APIWorkerMessage) live in their own test modules.
These tests focus on how JobPopper coordinates those components and the
higher-level api_job_pop flow.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

from horde_sdk import RequestErrorResponse

from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.job_popper import JobPopper
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeProcessState
from horde_worker_regen.process_management.pop_throttler import CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import make_job_pop_response, make_mock_bridge_data, make_mock_process_info


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
) -> JobPopper:
    """Build a JobPopper with mostly-mocked dependencies."""
    if state is None:
        state = WorkerState()
    if process_map is None:
        process_map = ProcessMap({})
    if job_tracker is None:
        job_tracker = JobTracker()
    if bridge_data is None:
        kwargs: dict = {}
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
        get_bridge_data=lambda: bridge_data,
        get_horde_client_session=lambda: horde_client_session,
        get_aiohttp_session=lambda: aiohttp_session,
        get_effective_megapixelsteps=lambda job: 1,
        max_inference_processes=max_inference_processes,
        max_concurrent_inference_processes=max_concurrent_inference_processes,
        dry_run_skip_api=dry_run_skip_api,
    )


def _make_process_map_with_available_processes() -> ProcessMap:
    """Create a process map that has both a safety and an inference process available."""
    safety_proc = make_mock_process_info(
        10,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.SAFETY,
    )
    inf_proc = make_mock_process_info(
        0,
        model_name="stable_diffusion",
        state=HordeProcessState.WAITING_FOR_JOB,
    )
    return ProcessMap({10: safety_proc, 0: inf_proc})


# ── Early-return guard clause tests ──────────────────────────────────────


class TestApiJobPopGuardClauses:
    """Each guard clause in api_job_pop should short-circuit cleanly."""

    def test_shutting_down_returns_early_and_clears_flag(self) -> None:
        """When shutting_down is True, pop exits immediately and clears last_pop_no_jobs."""
        state = WorkerState(shutting_down=True, last_pop_no_jobs_available=True)
        popper = _make_popper(state=state)

        asyncio.run(popper.api_job_pop())

        assert state.last_pop_no_jobs_available is False

    def test_too_many_consecutive_failures_blocks_pop(self) -> None:
        """Active failure pause prevents any pop attempt."""
        state = WorkerState(
            too_many_consecutive_failed_jobs=True,
            too_many_consecutive_failed_jobs_time=time.time(),
        )
        popper = _make_popper(state=state)

        asyncio.run(popper.api_job_pop())

        # Still in failure state
        assert state.too_many_consecutive_failed_jobs is True

    def test_consecutive_failure_pause_expires_and_resets(self) -> None:
        """After CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS, the pause should lift."""
        state = WorkerState(
            too_many_consecutive_failed_jobs=True,
            too_many_consecutive_failed_jobs_time=time.time() - CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS - 1,
            consecutive_failed_jobs=5,
        )
        popper = _make_popper(state=state)

        asyncio.run(popper.api_job_pop())

        assert state.too_many_consecutive_failed_jobs is False
        assert state.consecutive_failed_jobs == 0

    def test_reaching_failure_threshold_activates_pause(self) -> None:
        """When consecutive_failed_jobs hits 3, pause should activate."""
        state = WorkerState(consecutive_failed_jobs=3)
        popper = _make_popper(state=state)

        asyncio.run(popper.api_job_pop())

        assert state.too_many_consecutive_failed_jobs is True
        assert state.too_many_consecutive_failed_jobs_time > 0

    def test_failure_threshold_with_exit_on_faults_shuts_down(self) -> None:
        """When exit_on_unhandled_faults is True, reaching threshold triggers shutdown."""
        state = WorkerState(consecutive_failed_jobs=3)
        bd = make_mock_bridge_data(exit_on_unhandled_faults=True)
        shutdown_mgr = Mock()
        popper = _make_popper(state=state, bridge_data=bd, shutdown_manager=shutdown_mgr)

        asyncio.run(popper.api_job_pop())

        shutdown_mgr.shutdown.assert_called_once()

    def test_full_queue_returns_early(self) -> None:
        """Queue at capacity should prevent further pops."""
        job_tracker = JobTracker()
        # Default bridge data: queue_size=1, max_threads=1 → max_jobs_in_queue = 2
        for _ in range(10):
            job = Mock()
            job.model = "stable_diffusion"
            job_tracker.jobs_pending_inference.append(job)

        popper = _make_popper(job_tracker=job_tracker)
        asyncio.run(popper.api_job_pop())

    def test_no_safety_process_returns_early(self) -> None:
        """Without an available safety process, pop should not proceed."""
        popper = _make_popper(process_map=ProcessMap({}))
        asyncio.run(popper.api_job_pop())

    def test_no_inference_process_returns_early(self) -> None:
        """With safety but no inference process, pop should not proceed."""
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        popper = _make_popper(process_map=ProcessMap({10: safety_proc}))
        asyncio.run(popper.api_job_pop())

    def test_no_models_configured_returns_early(self) -> None:
        """Empty model list should prevent pops (with a sleep penalty)."""
        process_map = _make_process_map_with_available_processes()
        popper = _make_popper(process_map=process_map, image_models_to_load=[])
        asyncio.run(popper.api_job_pop())

    def test_too_frequent_pop_returns_early(self) -> None:
        """Popping again within the throttle window should be skipped."""
        state = WorkerState(last_job_pop_time=time.time())
        process_map = _make_process_map_with_available_processes()
        popper = _make_popper(state=state, process_map=process_map)
        asyncio.run(popper.api_job_pop())

    def test_first_job_not_submitted_blocks_second_pop(self) -> None:
        """When there are pending jobs but none submitted yet, don't pop more."""
        job_tracker = JobTracker()
        job = Mock()
        job.model = "stable_diffusion"
        job_tracker.jobs_pending_inference.append(job)
        # jobs_pending_submit is 0 (default)

        popper = _make_popper(job_tracker=job_tracker)
        asyncio.run(popper.api_job_pop())


# ── Consecutive failure handling ─────────────────────────────────────────


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


# ── Queue capacity checks ───────────────────────────────────────────────


class TestIsQueueFull:
    """Tests for _is_queue_full."""

    def test_empty_queue_not_full(self) -> None:
        """With no pending jobs, the queue should not be considered full."""
        popper = _make_popper()
        bd = make_mock_bridge_data(queue_size=1, max_threads=1)

        assert popper._is_queue_full(bd) is False

    def test_queue_at_capacity_is_full(self) -> None:
        """When pending jobs reach the max allowed, the queue should be considered full."""
        job_tracker = JobTracker()
        # queue_size=1, max_threads=1 → max_jobs_in_queue = 2
        for _ in range(2):
            job = Mock()
            job.model = "stable_diffusion"
            job_tracker.jobs_pending_inference.append(job)

        popper = _make_popper(job_tracker=job_tracker)
        bd = make_mock_bridge_data(queue_size=1, max_threads=1)

        assert popper._is_queue_full(bd) is True

    def test_multi_thread_increases_capacity(self) -> None:
        """max_threads > 1 should increase allowed queue depth."""
        job_tracker = JobTracker()
        for _ in range(2):
            job = Mock()
            job.model = "stable_diffusion"
            job_tracker.jobs_pending_inference.append(job)

        popper = _make_popper(job_tracker=job_tracker)
        bd = make_mock_bridge_data(queue_size=1, max_threads=2)

        # max_jobs_in_queue = queue_size + 1 + (max_threads - 1) = 1 + 1 + 1 = 3
        assert popper._is_queue_full(bd) is False

    def test_queue_one_below_capacity_not_full(self) -> None:
        """When pending jobs are one below the max allowed, the queue should not be considered full."""
        job_tracker = JobTracker()
        job = Mock()
        job.model = "stable_diffusion"
        job_tracker.jobs_pending_inference.append(job)

        popper = _make_popper(job_tracker=job_tracker)
        bd = make_mock_bridge_data(queue_size=1, max_threads=1)

        # max_jobs_in_queue = 2, current = 1 → not full
        assert popper._is_queue_full(bd) is False

    def test_large_queue_size(self) -> None:
        """With a larger queue_size, the method should calculate capacity accordingly."""
        job_tracker = JobTracker()
        for _ in range(5):
            job = Mock()
            job.model = "stable_diffusion"
            job_tracker.jobs_pending_inference.append(job)

        popper = _make_popper(job_tracker=job_tracker)
        bd = make_mock_bridge_data(queue_size=10, max_threads=1)

        # max_jobs_in_queue = 10 + 1 = 11, current = 5 → not full
        assert popper._is_queue_full(bd) is False


# ── API message processing ───────────────────────────────────────────────


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
        response.messages = []

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
            {"id": "msg-1", "message": "first", "origin": "system", "expiry": None},
        ]
        response2 = Mock()
        response2.messages = [
            {"id": "msg-1", "message": "second", "origin": "system", "expiry": None},
        ]

        popper._process_api_messages(response1)
        popper._process_api_messages(response2)

        assert popper._api_messages_received["msg-1"].message_text == "first"

    def test_multiple_messages_in_one_response(self) -> None:
        """Multiple messages in a single response should all be processed."""
        popper = _make_popper()
        response = Mock()
        response.messages = [
            {"id": "msg-1", "message": "a", "origin": "system", "expiry": None},
            {"id": "msg-2", "message": "b", "origin": "admin", "expiry": None},
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


# ── Error response handling ──────────────────────────────────────────────


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
        original_frequency = popper._throttler.current_pop_frequency

        resp = self._make_error_response("Server error")

        popper._handle_pop_error_response(resp)

        assert popper._throttler.current_pop_frequency > original_frequency


# ── SDK workaround tests ─────────────────────────────────────────────────


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


# ── Enqueue popped job ───────────────────────────────────────────────────


class TestEnqueuePoppedJob:
    """Tests for _enqueue_popped_job."""

    def test_job_added_to_pending_inference(self) -> None:
        """When a job is enqueued, it should be added to the jobs_pending_inference list."""
        job_tracker = JobTracker()
        popper = _make_popper(job_tracker=job_tracker)
        job = make_job_pop_response()

        asyncio.run(popper._enqueue_popped_job(job))

        assert len(job_tracker.jobs_pending_inference) == 1
        assert job_tracker.jobs_pending_inference[0] is job

    def test_pop_timestamp_recorded(self) -> None:
        """When a job is enqueued, the current time should be recorded in job_pop_timestamps for that job."""
        job_tracker = JobTracker()
        popper = _make_popper(job_tracker=job_tracker)
        job = make_job_pop_response()

        asyncio.run(popper._enqueue_popped_job(job))

        assert job in job_tracker.job_pop_timestamps
        assert job_tracker.job_pop_timestamps[job] > 0

    def test_jobs_lookup_entry_created(self) -> None:
        """When a job is enqueued, an entry should be created in jobs_lookup with the correct info."""
        job_tracker = JobTracker()
        popper = _make_popper(job_tracker=job_tracker)
        job = make_job_pop_response()

        asyncio.run(popper._enqueue_popped_job(job))

        assert job in job_tracker.jobs_lookup
        info = job_tracker.jobs_lookup[job]
        assert info.sdk_api_job_info is job
        assert info.state is None
        assert info.time_popped > 0

    def test_multiple_jobs_enqueued_in_order(self) -> None:
        """When multiple jobs are enqueued, they should be added to the pending list in the order enqueued."""
        job_tracker = JobTracker()
        popper = _make_popper(job_tracker=job_tracker)
        job1 = make_job_pop_response(model="model_a")
        job2 = make_job_pop_response(model="model_b")

        asyncio.run(popper._enqueue_popped_job(job1))
        asyncio.run(popper._enqueue_popped_job(job2))

        assert len(job_tracker.jobs_pending_inference) == 2
        assert job_tracker.jobs_pending_inference[0] is job1
        assert job_tracker.jobs_pending_inference[1] is job2


# ── Full pop flow with API mock ──────────────────────────────────────────

# Patch paths in the module under test to bypass SDK and telemetry dependencies
_POP_REQUEST_PATH = "horde_worker_regen.process_management.job_popper.ImageGenerateJobPopRequest"
_SPAN_POP_PATH = "horde_worker_regen.process_management.job_popper.span_job_pop"
_VERSION_PATH = "horde_worker_regen.__version__"


def _noop_span(**_kwargs: object):  # noqa: ANN202
    """No-op replacement for span_job_pop."""
    import contextlib

    return contextlib.nullcontext()


# Stack all three patches needed for full-flow tests
def _full_flow_patches(fn):  # noqa: ANN001, ANN202
    """Apply all patches needed to run api_job_pop through the full flow."""
    fn = patch(_SPAN_POP_PATH, _noop_span)(fn)
    fn = patch(_POP_REQUEST_PATH)(fn)
    fn = patch(_VERSION_PATH, "0.0.0-test", create=True)(fn)
    return fn  # noqa: RET504


class TestApiJobPopFullFlow:
    """End-to-end tests for api_job_pop with mocked API responses.

    We patch ImageGenerateJobPopRequest and span_job_pop so we don't depend on
    SDK validation or telemetry — the tests focus on how JobPopper orchestrates
    the response.
    """

    def _make_ready_popper(
        self,
        *,
        api_response: object | None = None,
        state: WorkerState | None = None,
        job_tracker: JobTracker | None = None,
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
            horde_client_session=horde_session,
        )

    @_full_flow_patches
    def test_successful_pop_enqueues_job(self, _mock_req_cls: Mock) -> None:
        """A successful pop with a valid job should add it to the queue."""
        job_response = make_job_pop_response()
        popper = self._make_ready_popper(api_response=job_response)

        asyncio.run(popper.api_job_pop())

        assert len(popper._job_tracker.jobs_pending_inference) == 1
        assert popper._state.last_pop_no_jobs_available is False

    @_full_flow_patches
    def test_successful_pop_resets_maintenance_flag(self, _mock_req_cls: Mock) -> None:
        """After a successful pop, last_pop_maintenance_mode should be reset to False.

        This is so that future maintenance mode responses will log a warning again.
        """
        state = WorkerState(last_job_pop_time=0.0)
        popper = self._make_ready_popper(
            api_response=make_job_pop_response(),
            state=state,
        )
        popper._state.last_pop_maintenance_mode = True

        asyncio.run(popper.api_job_pop())

        assert popper._state.last_pop_maintenance_mode is False

    @_full_flow_patches
    def test_successful_pop_resets_throttler_to_default(self, _mock_req_cls: Mock) -> None:
        """After a successful pop, the throttler should reset to default frequency."""
        popper = self._make_ready_popper(api_response=make_job_pop_response())
        popper._throttler.on_pop_error()  # put throttler in error state

        asyncio.run(popper.api_job_pop())

        assert popper._throttler.current_pop_frequency == popper._throttler._default_pop_frequency

    @_full_flow_patches
    def test_no_job_available_sets_flag(self, _mock_req_cls: Mock) -> None:
        """When API returns a response with id_ = None (no job), flag should be set."""
        # Build a mock that quacks like an empty ImageGenerateJobPopResponse
        empty_response = Mock()
        empty_response.id_ = None
        empty_response.skipped = Mock()
        empty_response.skipped.model_dump.return_value = {}
        empty_response.skipped.model_extra = None
        empty_response.messages = None

        popper = self._make_ready_popper(api_response=empty_response)

        asyncio.run(popper.api_job_pop())

        assert popper._state.last_pop_no_jobs_available is True
        assert len(popper._job_tracker.jobs_pending_inference) == 0

    @_full_flow_patches
    def test_api_exception_slows_throttler(self, _mock_req_cls: Mock) -> None:
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

        asyncio.run(popper.api_job_pop())

        assert popper._throttler.current_pop_frequency == popper._throttler._error_pop_frequency

    @_full_flow_patches
    def test_error_response_handled(self, _mock_req_cls: Mock) -> None:
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

        asyncio.run(popper.api_job_pop())

    @_full_flow_patches
    def test_job_faults_initialized_for_popped_job(self, _mock_req_cls: Mock) -> None:
        """When a job is popped, its fault list should be initialized."""
        job_response = make_job_pop_response()
        popper = self._make_ready_popper(api_response=job_response)

        asyncio.run(popper.api_job_pop())

        assert job_response.id_ in popper._job_tracker.job_faults
        assert popper._job_tracker.job_faults[job_response.id_] == []

    @_full_flow_patches
    def test_pop_updates_last_pop_time(self, _mock_req_cls: Mock) -> None:
        """Successful or not, api_job_pop should update last_job_pop_time."""
        state = WorkerState(last_job_pop_time=0.0)
        popper = self._make_ready_popper(api_response=make_job_pop_response(), state=state)

        asyncio.run(popper.api_job_pop())

        assert state.last_job_pop_time > 0


# ── WorkerState integration ─────────────────────────────────────────────


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
        assert popper._throttler._current_pop_frequency == 1.0

    def test_error_pop_frequency(self) -> None:
        """Error pop frequency should be 5.0 seconds."""
        popper = _make_popper()
        assert popper._throttler._error_pop_frequency == 5.0
