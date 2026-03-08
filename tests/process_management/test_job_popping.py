"""Tests for JobPopper."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import Mock

from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.job_popper import JobPopper
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeProcessState
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import make_mock_bridge_data, make_mock_process_info


def _make_popper(
    *,
    state: WorkerState | None = None,
    process_map: ProcessMap | None = None,
    job_tracker: JobTracker | None = None,
    bridge_data: Mock | None = None,
    horde_client_session: object | None = None,
    aiohttp_session: object | None = None,
    max_inference_processes: int = 2,
    max_concurrent_inference_processes: int = 1,
    image_models_to_load: list[str] | None = None,
) -> JobPopper:
    """Build a JobPopper with mostly-mocked dependencies."""
    if state is None:
        state = WorkerState()
    if process_map is None:
        process_map = ProcessMap({})
    if job_tracker is None:
        job_tracker = JobTracker()
    if bridge_data is None:
        kwargs = {}
        if image_models_to_load is not None:
            kwargs["image_models_to_load"] = image_models_to_load
        bridge_data = make_mock_bridge_data(**kwargs)
    if horde_client_session is None:
        horde_client_session = Mock()
    if aiohttp_session is None:
        aiohttp_session = Mock()

    return JobPopper(
        state=state,
        process_map=process_map,
        job_tracker=job_tracker,
        shutdown_manager=Mock(),
        get_bridge_data=lambda: bridge_data,
        get_horde_client_session=lambda: horde_client_session,
        get_aiohttp_session=lambda: aiohttp_session,
        get_effective_megapixelsteps=lambda job: 1,
        max_inference_processes=max_inference_processes,
        max_concurrent_inference_processes=max_concurrent_inference_processes,
    )


class TestApiJobPop:
    """Tests for api_job_pop."""

    def test_shutting_down_returns_early(self) -> None:
        state = WorkerState(shutting_down=True)
        popper = _make_popper(state=state)

        asyncio.run(popper.api_job_pop())
        assert state.last_pop_no_jobs_available is False

    def test_too_many_consecutive_failures_pauses(self) -> None:
        state = WorkerState(
            too_many_consecutive_failed_jobs=True,
            too_many_consecutive_failed_jobs_time=time.time(),
        )
        popper = _make_popper(state=state)

        asyncio.run(popper.api_job_pop())

    def test_consecutive_failures_threshold_triggers_pause(self) -> None:
        state = WorkerState(consecutive_failed_jobs=3)
        popper = _make_popper(state=state)

        asyncio.run(popper.api_job_pop())
        assert state.too_many_consecutive_failed_jobs is True

    def test_full_queue_returns_early(self) -> None:
        jt = JobTracker()
        for i in range(10):
            job = Mock()
            job.model = "stable_diffusion"
            jt.jobs_pending_inference.append(job)

        popper = _make_popper(job_tracker=jt)
        asyncio.run(popper.api_job_pop())

    def test_no_safety_process_returns_early(self) -> None:
        state = WorkerState(last_job_pop_time=0.0)
        popper = _make_popper(state=state)

        asyncio.run(popper.api_job_pop())

    def test_no_inference_process_returns_early(self) -> None:
        state = WorkerState(last_job_pop_time=0.0)
        safety_proc = make_mock_process_info(
            10, model_name=None, state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        pm = ProcessMap({10: safety_proc})

        popper = _make_popper(state=state, process_map=pm)
        asyncio.run(popper.api_job_pop())

    def test_no_models_configured_returns_early(self) -> None:
        state = WorkerState(last_job_pop_time=0.0)
        safety_proc = make_mock_process_info(
            10, model_name=None, state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        inf_proc = make_mock_process_info(
            0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB,
        )
        pm = ProcessMap({10: safety_proc, 0: inf_proc})

        popper = _make_popper(state=state, process_map=pm, image_models_to_load=[])
        asyncio.run(popper.api_job_pop())

    def test_too_frequent_pop_returns_early(self) -> None:
        state = WorkerState(last_job_pop_time=time.time())
        safety_proc = make_mock_process_info(
            10, model_name=None, state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        inf_proc = make_mock_process_info(
            0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB,
        )
        pm = ProcessMap({10: safety_proc, 0: inf_proc})

        popper = _make_popper(state=state, process_map=pm)
        asyncio.run(popper.api_job_pop())


class TestJobPopFrequency:
    """Tests for pop frequency state management."""

    def test_last_pop_recently_true(self) -> None:
        state = WorkerState(last_job_pop_time=time.time())
        assert state.last_pop_recently() is True

    def test_last_pop_recently_false(self) -> None:
        state = WorkerState(last_job_pop_time=time.time() - 20)
        assert state.last_pop_recently() is False

    def test_default_pop_frequency(self) -> None:
        popper = _make_popper()
        assert popper._job_pop_frequency == 1.0

    def test_error_pop_frequency(self) -> None:
        popper = _make_popper()
        assert popper._error_job_pop_frequency == 5.0


class TestGetSourceImages:
    """Tests for _get_source_images."""

    def test_no_source_images(self) -> None:
        popper = _make_popper()

        job = Mock()
        job.id_ = "test-id"
        job.source_image = None
        job.source_mask = None
        job.extra_source_images = None

        result = asyncio.run(popper._get_source_images(job))
        assert result is job

    def test_none_id_returns_early(self) -> None:
        popper = _make_popper()

        job = Mock()
        job.id_ = None

        result = asyncio.run(popper._get_source_images(job))
        assert result is job
