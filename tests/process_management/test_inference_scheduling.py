"""Tests for InferenceScheduler."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.lru_cache import LRUCache
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import make_job_pop_response, make_mock_bridge_data, make_mock_process_info


def _make_inference_scheduler(
    *,
    state: WorkerState | None = None,
    process_map: ProcessMap | None = None,
    horde_model_map: HordeModelMap | None = None,
    job_tracker: JobTracker | None = None,
    bridge_data: Mock | None = None,
    max_concurrent: int = 1,
    max_inference: int = 2,
) -> InferenceScheduler:
    """Build an InferenceScheduler with mostly-mocked dependencies."""
    if state is None:
        state = WorkerState()
    if process_map is None:
        process_map = ProcessMap({})
    if horde_model_map is None:
        horde_model_map = HordeModelMap(root={})
    if job_tracker is None:
        job_tracker = JobTracker()
    if bridge_data is None:
        bridge_data = make_mock_bridge_data()

    return InferenceScheduler(
        state=state,
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        process_lifecycle=Mock(get_processes_with_model_for_queued_job=Mock(return_value=[])),
        get_bridge_data=lambda: bridge_data,
        get_model_baseline=lambda name: None,
        get_stable_diffusion_reference=lambda: None,
        max_concurrent_inference_processes=max_concurrent,
        max_inference_processes=max_inference,
        lru=LRUCache(max_inference),
    )


class TestPreloadModels:
    """Tests for preload_models."""

    def test_no_pending_jobs_returns_false(self) -> None:
        """Preload should return False if there are no pending inference jobs."""
        inference_scheduler = _make_inference_scheduler()
        assert inference_scheduler.preload_models() is False

    def test_model_already_loaded_returns_false(self) -> None:
        """Preload should return False if the needed model is already loaded in a process."""
        process_info = make_mock_process_info(0, model_name="stable_diffusion")
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        job_tracker.jobs_pending_inference.append(job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        assert inference_scheduler.preload_models() is False

    def test_preload_sends_message_when_process_available(self) -> None:
        """Preload should send a message to a process to load the model if it's not already loaded."""
        process_info = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job = make_job_pop_response("new_model")
        job_tracker.jobs_pending_inference.append(job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        result = inference_scheduler.preload_models()
        assert result is True
        assert process_info.last_control_flag == HordeControlFlag.PRELOAD_MODEL

    def test_no_available_process_returns_false(self) -> None:
        """Preload should return False if there are no available processes to load the model."""
        process_info = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job = make_job_pop_response("new_model")
        job_tracker.jobs_pending_inference.append(job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        assert inference_scheduler.preload_models() is False

    def test_clears_preloaded_model_no_longer_needed(self) -> None:
        """If a model was preloaded for a job but that job is no longer pending, the model should be cleared."""
        process_info = make_mock_process_info(0, model_name="old_model", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({0: process_info})
        process_map.on_process_state_change = Mock()  # type: ignore
        job_tracker = JobTracker()

        job = make_job_pop_response("different_model")
        job_tracker.jobs_pending_inference.append(job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        inference_scheduler.preload_models()

        process_map.on_process_state_change.assert_called_with(  # type: ignore
            process_id=0,
            new_state=HordeProcessState.WAITING_FOR_JOB,
        )


class TestGetNextJobAndProcess:
    """Tests for get_next_job_and_process."""

    def test_no_pending_jobs_returns_none(self) -> None:
        """get_next_job_and_process should return None if there are no pending jobs."""
        inference_scheduler = _make_inference_scheduler()
        assert inference_scheduler.get_next_job_and_process() is None

    def test_returns_job_with_matching_process(self) -> None:
        """get_next_job_and_process should return a job and process that match if one is available."""
        process_info = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.PRELOADED_MODEL,
        )
        process_map = ProcessMap({0: process_info})
        hmm = HordeModelMap(root={})
        job_tracker = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        job_tracker.jobs_pending_inference.append(job)

        hmm.update_entry(
            horde_model_name="stable_diffusion",
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=0,
        )

        sched = _make_inference_scheduler(process_map=process_map, horde_model_map=hmm, job_tracker=job_tracker)
        result = sched.get_next_job_and_process()
        assert result is not None
        assert result.next_job is job
        assert result.process_with_model is process_info

    def test_no_process_with_model_returns_none(self) -> None:
        """get_next_job_and_process should return None if there is no process with the required model."""
        process_info = make_mock_process_info(0, model_name="other_model")
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        job_tracker.jobs_pending_inference.append(job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        assert inference_scheduler.get_next_job_and_process() is None

    def test_max_concurrent_reached_returns_none(self) -> None:
        """get_next_job_and_process should return None if the maximum number of concurrent jobs is reached."""
        process_info = make_mock_process_info(
            0, model_name="stable_diffusion", state=HordeProcessState.PRELOADED_MODEL
        )
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job_in_progress = make_job_pop_response("stable_diffusion")
        job_tracker.jobs_in_progress.append(job_in_progress)

        job = make_job_pop_response("stable_diffusion")
        job_tracker.jobs_pending_inference.append(job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        assert inference_scheduler.get_next_job_and_process() is None

    def test_skipped_line_is_returned_on_second_call(self) -> None:
        """get_next_job_and_process should return the skipped line on the second call."""
        inference_scheduler = _make_inference_scheduler()
        next_job_process = Mock()
        inference_scheduler._job_tracker._skipped_line_next_job_and_process = next_job_process

        assert inference_scheduler.get_next_job_and_process() is next_job_process

    def test_job_in_progress_is_skipped(self) -> None:
        """get_next_job_and_process should skip jobs that are already in progress."""
        process_info = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.PRELOADED_MODEL,
        )
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        job_tracker.jobs_pending_inference.append(job)
        job_tracker.jobs_in_progress.append(job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        assert inference_scheduler.get_next_job_and_process() is None


class TestStartInference:
    """Tests for start_inference."""

    def test_no_next_job_returns_false(self) -> None:
        """start_inference should return False if there is no next job."""
        inference_scheduler = _make_inference_scheduler()
        assert inference_scheduler.start_inference() is False

    def test_successful_start_adds_to_in_progress(self) -> None:
        """start_inference should add the job to in_progress if it starts successfully."""
        process_info = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.PRELOADED_MODEL,
        )
        process_map = ProcessMap({0: process_info})
        horde_model_map = HordeModelMap(root={})
        job_tracker = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        job_tracker.jobs_pending_inference.append(job)

        horde_model_map.update_entry(
            horde_model_name="stable_diffusion",
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=0,
        )

        inference_scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
        )
        result = inference_scheduler.start_inference()
        assert result is True
        assert job in job_tracker.jobs_in_progress
        assert process_info.last_control_flag == HordeControlFlag.START_INFERENCE

    def test_failed_send_faults_job(self) -> None:
        """If sending the message to start inference fails, the job should be faulted and not added to in_progress."""
        process_info = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.PRELOADED_MODEL,
            safe_send_returns=False,
        )
        process_map = ProcessMap({0: process_info})
        horde_model_map = HordeModelMap(root={})
        job_tracker = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        job_tracker.jobs_pending_inference.append(job)

        horde_model_map.update_entry(
            horde_model_name="stable_diffusion",
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=0,
        )

        inference_scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
        )
        result = inference_scheduler.start_inference()
        assert result is True
        assert job not in job_tracker.jobs_in_progress


class TestUnloadModels:
    """Tests for unload_models and related methods."""

    def test_unload_models_no_pending_returns_false(self) -> None:
        """unload_models should return False if there are no pending inference jobs."""
        inference_scheduler = _make_inference_scheduler()
        assert inference_scheduler.unload_models() is False

    def test_unload_models_pending_job_returns_false_for_needed_model(self) -> None:
        """unload_models should return False if there is a pending job for the model."""
        job_tracker = JobTracker()
        job = make_job_pop_response("stable_diffusion")
        job_tracker.jobs_pending_inference.append(job)

        process_info = make_mock_process_info(
            0, model_name="stable_diffusion", state=HordeProcessState.PRELOADED_MODEL
        )
        process_map = ProcessMap({0: process_info})

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        assert inference_scheduler.unload_models() is False
        assert process_info.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_RAM

    def test_unload_models_single_thread_single_model_returns_false(self) -> None:
        """If there is only one inference process and one model, unload_models should return False.

        This is because we don't want to unload the only model we have if we only have one process,
        as there is no benefit and it could cause issues if the process is slow to unload/load.
        """
        job_tracker = JobTracker()
        job = make_job_pop_response("stable_diffusion")
        job_tracker.jobs_pending_inference.append(job)

        inference_scheduler = _make_inference_scheduler(job_tracker=job_tracker)
        assert inference_scheduler.unload_models() is False

    def test_get_next_n_models_returns_correct(self) -> None:
        """get_next_n_models should return a list of unique model names for the next n pending inference jobs."""
        job_tracker = JobTracker()
        job1 = make_job_pop_response("model_a")
        job2 = make_job_pop_response("model_b")
        job3 = make_job_pop_response("model_a")
        job_tracker.jobs_pending_inference.extend([job1, job2, job3])

        inference_scheduler = _make_inference_scheduler(job_tracker=job_tracker)
        result = inference_scheduler.get_next_n_models(3)
        assert result == ["model_a", "model_b"]

    def test_unload_from_ram_invalid_process_raises(self) -> None:
        """unload_from_ram should raise an error if the process ID is not in the process map."""
        inference_scheduler = _make_inference_scheduler()
        with pytest.raises(ValueError, match="not in the process map"):
            inference_scheduler.unload_from_ram(99)

    def test_unload_from_ram_non_inference_warns(self) -> None:
        """unload_from_ram should log a warning if the process is not an inference process."""
        process_info = make_mock_process_info(0, process_type=HordeProcessType.SAFETY)
        process_map = ProcessMap({0: process_info})

        inference_scheduler = _make_inference_scheduler(process_map=process_map)
        inference_scheduler.unload_from_ram(0)

    def test_unload_from_ram_recently_unloaded_skips(self) -> None:
        """unload_from_ram should skip sending the unload message if the process was recently unloaded from RAM."""
        process_info = make_mock_process_info(0)
        process_info.recently_unloaded_from_ram = True
        process_map = ProcessMap({0: process_info})

        inference_scheduler = _make_inference_scheduler(process_map=process_map)
        old_control_flag = process_info.last_control_flag
        inference_scheduler.unload_from_ram(0)
        assert process_info.last_control_flag == old_control_flag


class TestGetSingleJobEffectiveMegapixelsteps:
    """Tests for get_single_job_effective_megapixelsteps."""

    def test_returns_value(self) -> None:
        """get_single_job_effective_megapixelsteps should return an integer value for a valid job."""
        inference_scheduler = _make_inference_scheduler()
        job = make_job_pop_response("stable_diffusion")

        result = inference_scheduler.get_single_job_effective_megapixelsteps(job)
        assert isinstance(result, int)
        assert result > 0
