"""Tests for JobSubmitter."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

from horde_worker_regen.process_management.job_models import PendingSubmitJob
from horde_worker_regen.process_management.job_submitter import JobSubmitter
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import make_job_pop_response, make_mock_bridge_data


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
        get_bridge_data=lambda: bridge_data,
        get_stable_diffusion_reference=lambda: None,
        get_horde_client_session=lambda: horde_client_session,
        get_aiohttp_session=lambda: aiohttp_session,
    )


class TestSubmitSingleGeneration:
    """Tests for submit_single_generation."""

    def test_no_image_result_faults(self) -> None:
        submitter = _make_submitter()

        new_submit = Mock(spec=PendingSubmitJob)
        new_submit.job_id = "test-id"
        new_submit.image_result = None
        new_submit.is_faulted = False

        asyncio.run(submitter.submit_single_generation(new_submit))
        new_submit.fault.assert_called_once()

    def test_already_faulted_no_image_proceeds_to_submit(self) -> None:
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

        result = asyncio.run(submitter.submit_single_generation(new_submit))
        new_submit.fault.assert_not_called()


class TestApiSubmitJob:
    """Tests for api_submit_job."""

    def test_no_pending_submits_returns_early(self) -> None:
        submitter = _make_submitter()
        asyncio.run(submitter.api_submit_job())

    def test_faulted_job_increments_consecutive_failures(self) -> None:
        from horde_sdk.ai_horde_api import GENERATION_STATE

        from horde_worker_regen.process_management.job_models import HordeJobInfo

        state = WorkerState()
        jt = JobTracker()
        submitter = _make_submitter(state=state, job_tracker=jt)

        job = make_job_pop_response("stable_diffusion", r2_upload="https://example.com/upload")

        job_info = HordeJobInfo(
            sdk_api_job_info=job,
            state=GENERATION_STATE.faulted,
            time_popped=0.0,
        )
        job_info.job_image_results = None
        job_info.censored = False
        job_info.time_to_generate = 1.0

        jt.jobs_pending_submit.append(job_info)
        jt.jobs_lookup[job] = job_info
        jt.job_pop_timestamps[job] = 0.0

        faulted_submit = Mock(spec=PendingSubmitJob)
        faulted_submit.is_finished = True
        faulted_submit.is_faulted = True
        faulted_submit.kudos_reward = 0
        faulted_submit.kudos_per_second = 0

        with patch.object(submitter, "submit_single_generation", new_callable=AsyncMock) as mock_submit:
            mock_submit.return_value = faulted_submit
            asyncio.run(submitter.api_submit_job())

        assert state.consecutive_failed_jobs >= 1
