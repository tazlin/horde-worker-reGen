"""Tests for JobSubmitter."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

from horde_sdk.ai_horde_api import GENERATION_STATE

from horde_worker_regen.process_management.job_models import HordeJobInfo, PendingSubmitJob
from horde_worker_regen.process_management.job_submitter import JobSubmitter
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeImageResult
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_test_api_sessions,
    make_test_model_metadata,
    make_test_runtime_config,
    queue_job_for_submit_async,
    track_popped_job_async,
)


def _make_submitter(
    *,
    state: WorkerState | None = None,
    job_tracker: JobTracker | None = None,
    bridge_data: Mock | None = None,
    horde_client_session: object | None = None,
    aiohttp_session: object | None = None,
) -> JobSubmitter:
    """Build a JobSubmitter with mostly-mocked dependencies."""
    if state is None:
        state = WorkerState()
    if job_tracker is None:
        job_tracker = JobTracker()
    if bridge_data is None:
        bridge_data = make_mock_bridge_data()
    if horde_client_session is None:
        horde_client_session = Mock()
    if aiohttp_session is None:
        aiohttp_session = Mock()

    return JobSubmitter(
        state=state,
        job_tracker=job_tracker,
        shutdown_manager=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        api_sessions=make_test_api_sessions(
            horde_client_session=horde_client_session,
            aiohttp_session=aiohttp_session,
        ),
        model_metadata=make_test_model_metadata(),
    )


class TestSubmitSingleGeneration:
    """Tests for submit_single_generation."""

    async def test_no_image_result_faults(self) -> None:
        """If there is no image result and the job is not already faulted, the method faults the job."""
        submitter = _make_submitter()

        new_submit = Mock(spec=PendingSubmitJob)
        new_submit.job_id = "test-id"
        new_submit.image_result = None
        new_submit.is_faulted = False
        # A real PendingSubmitJob always carries completed_job_info; a non-faulted state keeps this on
        # the "no image is a local error" path rather than the fault-report path.
        new_submit.completed_job_info = Mock(state=None)

        await submitter.submit_single_generation(new_submit)
        new_submit.fault.assert_called_once()

    async def test_already_faulted_no_image_proceeds_to_submit(self) -> None:
        """When already faulted and no image, the code skips upload and submits fault metadata."""
        submitter = _make_submitter(horde_client_session=AsyncMock())

        completed_info = Mock()
        completed_info.sdk_api_job_info = Mock()
        completed_info.sdk_api_job_info.payload = Mock()
        completed_info.sdk_api_job_info.payload.seed = 42
        completed_info.sdk_api_job_info.get_follow_up_default_request_type.return_value = Mock
        completed_info.state = None

        new_submit = Mock(spec=PendingSubmitJob)
        new_submit.job_id = "test-id"
        new_submit.image_result = None
        new_submit.is_faulted = True
        new_submit.completed_job_info = completed_info

        await submitter.submit_single_generation(new_submit)
        new_submit.fault.assert_not_called()

    async def test_generation_faulted_no_image_is_reported_to_api(self) -> None:
        """A job whose generation faulted (images cleared, state=faulted) is still reported to the horde.

        The submit used to early-return for any imageless job whose per-submit-task ``is_faulted`` flag
        was unset (it never is on a fresh task), so a generation fault was never reported and the horde
        only reissued the job after its own timeout. The job's own faulted state must drive the report.
        """
        horde_session = AsyncMock()
        # Returning None short-circuits before the kudos math; we only need to prove the API was called.
        horde_session.submit_request = AsyncMock(return_value=None)
        submitter = _make_submitter(horde_client_session=horde_session)

        completed_info = Mock()
        completed_info.sdk_api_job_info = Mock()
        completed_info.sdk_api_job_info.payload = Mock()
        completed_info.sdk_api_job_info.payload.seed = 42
        completed_info.sdk_api_job_info.get_follow_up_default_request_type.return_value = Mock
        completed_info.state = GENERATION_STATE.faulted

        new_submit = Mock(spec=PendingSubmitJob)
        new_submit.job_id = "test-id"
        new_submit.image_result = None
        new_submit.is_faulted = False
        new_submit.completed_job_info = completed_info

        await submitter.submit_single_generation(new_submit)

        new_submit.fault.assert_not_called()
        horde_session.submit_request.assert_awaited_once()


class TestApiSubmitJob:
    """Tests for api_submit_job."""

    async def test_no_pending_submits_returns_early(self) -> None:
        """If there are no pending submits, the method should return early without doing anything."""
        submitter = _make_submitter()
        await submitter.api_submit_job()

    async def test_faulted_job_increments_consecutive_failures(self) -> None:
        """If a job submission results in a faulted job, the consecutive_failed_jobs counter should be incremented."""
        from horde_sdk.ai_horde_api import GENERATION_STATE

        from horde_worker_regen.process_management.job_models import HordeJobInfo

        state = WorkerState()
        job_tracker = JobTracker()
        submitter = _make_submitter(state=state, job_tracker=job_tracker)

        job = make_job_pop_response("stable_diffusion", r2_upload="https://example.com/upload")

        job_info = HordeJobInfo(
            sdk_api_job_info=job,
            state=GENERATION_STATE.faulted,
            time_popped=0.0,
        )
        job_info.job_image_results = None
        job_info.censored = False
        job_info.time_to_generate = 1.0

        await track_popped_job_async(job_tracker, job, time_popped=0.0)
        await queue_job_for_submit_async(job_tracker, job_info)

        faulted_submit = Mock(spec=PendingSubmitJob)
        faulted_submit.is_finished = True
        faulted_submit.is_faulted = True
        faulted_submit.kudos_reward = 0
        faulted_submit.kudos_per_second = 0

        with patch.object(submitter, "submit_single_generation", new_callable=AsyncMock) as mock_submit:
            mock_submit.return_value = faulted_submit
            await submitter.api_submit_job()

        assert state.consecutive_failed_jobs >= 1

    async def test_unsafetychecked_job_is_punted_not_raised(self) -> None:
        """A PENDING_SUBMIT job with images but no safety verdict must be punted, not crash the loop.

        Regression for the shutdown crash loop: such a "poison" job used to make api_submit_job raise
        ``ValueError("censored is None")`` every iteration without ever removing the job, so the submit
        queue never drained and shutdown wedged. It must instead be faulted, reported, and removed.
        """
        state = WorkerState()
        job_tracker = JobTracker()

        horde_session = AsyncMock()
        horde_session.submit_request = AsyncMock(return_value=Mock(reward=6.65))
        submitter = _make_submitter(
            state=state,
            job_tracker=job_tracker,
            horde_client_session=horde_session,
            aiohttp_session=AsyncMock(),
        )

        job = make_job_pop_response("stable_diffusion", r2_upload="https://example.com/upload")
        poison = HordeJobInfo(
            sdk_api_job_info=job,
            state=GENERATION_STATE.ok,
            time_popped=0.0,
            job_image_results=[HordeImageResult(image_base64="data")],
            censored=None,
        )
        await track_popped_job_async(job_tracker, job, time_popped=0.0)
        await queue_job_for_submit_async(job_tracker, poison)
        assert len(job_tracker.jobs_pending_submit) == 1

        # Must not raise, and must drain the job (reported as a fault) so the loop can make progress.
        await submitter.api_submit_job()

        assert len(job_tracker.jobs_pending_submit) == 0
        horde_session.submit_request.assert_awaited()
        assert state.consecutive_failed_jobs >= 1

    async def test_repeated_head_failures_drop_the_stuck_job(self) -> None:
        """The submit loop's backstop drops a head-of-queue job that keeps raising, so the queue drains.

        api_submit_job punts known-bad jobs itself; this guards against an *unforeseen* exception
        recurring on the same head job forever (the shape that produced thousands of identical
        tracebacks and blocked shutdown).
        """
        state = WorkerState()
        job_tracker = JobTracker()
        submitter = _make_submitter(state=state, job_tracker=job_tracker)

        job = make_job_pop_response("stable_diffusion")
        job_info = HordeJobInfo(
            sdk_api_job_info=job,
            state=GENERATION_STATE.ok,
            censored=False,
            time_popped=0.0,
            job_image_results=[HordeImageResult(image_base64="data")],
        )
        await track_popped_job_async(job_tracker, job, time_popped=0.0)
        await queue_job_for_submit_async(job_tracker, job_info)

        error = RuntimeError("unexpected submit failure")
        for _ in range(submitter._MAX_CONSECUTIVE_HEAD_SUBMIT_FAILURES - 1):
            await submitter._handle_unexpected_submit_failure(error)
            assert len(job_tracker.jobs_pending_submit) == 1, "job dropped too early"

        await submitter._handle_unexpected_submit_failure(error)
        assert len(job_tracker.jobs_pending_submit) == 0
        assert state.consecutive_failed_jobs >= 1
