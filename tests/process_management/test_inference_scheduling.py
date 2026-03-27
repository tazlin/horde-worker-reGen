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


def _make_scheduler(
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
        sched = _make_scheduler()
        assert sched.preload_models() is False

    def test_model_already_loaded_returns_false(self) -> None:
        proc = make_mock_process_info(0, model_name="stable_diffusion")
        process_map = ProcessMap({0: proc})
        jt = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        jt.jobs_pending_inference.append(job)

        sched = _make_scheduler(process_map=process_map, job_tracker=jt)
        assert sched.preload_models() is False

    def test_preload_sends_message_when_process_available(self) -> None:
        proc = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: proc})
        jt = JobTracker()

        job = make_job_pop_response("new_model")
        jt.jobs_pending_inference.append(job)

        sched = _make_scheduler(process_map=process_map, job_tracker=jt)
        result = sched.preload_models()
        assert result is True
        assert proc.last_control_flag == HordeControlFlag.PRELOAD_MODEL

    def test_no_available_process_returns_false(self) -> None:
        proc = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)
        process_map = ProcessMap({0: proc})
        jt = JobTracker()

        job = make_job_pop_response("new_model")
        jt.jobs_pending_inference.append(job)

        sched = _make_scheduler(process_map=process_map, job_tracker=jt)
        assert sched.preload_models() is False

    def test_clears_preloaded_model_no_longer_needed(self) -> None:
        proc = make_mock_process_info(0, model_name="old_model", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({0: proc})
        process_map.on_process_state_change = Mock()  # type: ignore
        jt = JobTracker()

        job = make_job_pop_response("different_model")
        jt.jobs_pending_inference.append(job)

        sched = _make_scheduler(process_map=process_map, job_tracker=jt)
        sched.preload_models()

        process_map.on_process_state_change.assert_called_with(  # type: ignore
            process_id=0,
            new_state=HordeProcessState.WAITING_FOR_JOB,
        )


class TestGetNextJobAndProcess:
    """Tests for get_next_job_and_process."""

    def test_no_pending_jobs_returns_none(self) -> None:
        sched = _make_scheduler()
        assert sched.get_next_job_and_process() is None

    def test_returns_job_with_matching_process(self) -> None:
        proc = make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({0: proc})
        hmm = HordeModelMap(root={})
        jt = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        jt.jobs_pending_inference.append(job)

        hmm.update_entry(
            horde_model_name="stable_diffusion",
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=0,
        )

        sched = _make_scheduler(process_map=process_map, horde_model_map=hmm, job_tracker=jt)
        result = sched.get_next_job_and_process()
        assert result is not None
        assert result.next_job is job
        assert result.process_with_model is proc

    def test_no_process_with_model_returns_none(self) -> None:
        proc = make_mock_process_info(0, model_name="other_model")
        process_map = ProcessMap({0: proc})
        jt = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        jt.jobs_pending_inference.append(job)

        sched = _make_scheduler(process_map=process_map, job_tracker=jt)
        assert sched.get_next_job_and_process() is None

    def test_max_concurrent_reached_returns_none(self) -> None:
        proc = make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({0: proc})
        jt = JobTracker()

        job_in_progress = make_job_pop_response("stable_diffusion")
        jt.jobs_in_progress.append(job_in_progress)

        job = make_job_pop_response("stable_diffusion")
        jt.jobs_pending_inference.append(job)

        sched = _make_scheduler(process_map=process_map, job_tracker=jt)
        assert sched.get_next_job_and_process() is None

    def test_skipped_line_is_returned_on_second_call(self) -> None:
        sched = _make_scheduler()
        next_job_process = Mock()
        sched._job_tracker._skipped_line_next_job_and_process = next_job_process

        assert sched.get_next_job_and_process() is next_job_process

    def test_job_in_progress_is_skipped(self) -> None:
        proc = make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({0: proc})
        jt = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        jt.jobs_pending_inference.append(job)
        jt.jobs_in_progress.append(job)

        sched = _make_scheduler(process_map=process_map, job_tracker=jt)
        assert sched.get_next_job_and_process() is None


class TestStartInference:
    """Tests for start_inference."""

    def test_no_next_job_returns_false(self) -> None:
        sched = _make_scheduler()
        assert sched.start_inference() is False

    def test_successful_start_adds_to_in_progress(self) -> None:
        proc = make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({0: proc})
        hmm = HordeModelMap(root={})
        jt = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        jt.jobs_pending_inference.append(job)

        hmm.update_entry(
            horde_model_name="stable_diffusion",
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=0,
        )

        sched = _make_scheduler(process_map=process_map, horde_model_map=hmm, job_tracker=jt)
        result = sched.start_inference()
        assert result is True
        assert job in jt.jobs_in_progress
        assert proc.last_control_flag == HordeControlFlag.START_INFERENCE

    def test_failed_send_faults_job(self) -> None:
        proc = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.PRELOADED_MODEL,
            safe_send_returns=False,
        )
        process_map = ProcessMap({0: proc})
        hmm = HordeModelMap(root={})
        jt = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        jt.jobs_pending_inference.append(job)

        hmm.update_entry(
            horde_model_name="stable_diffusion",
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=0,
        )

        sched = _make_scheduler(process_map=process_map, horde_model_map=hmm, job_tracker=jt)
        result = sched.start_inference()
        assert result is True
        assert job not in jt.jobs_in_progress


class TestUnloadModels:
    """Tests for unload_models and related methods."""

    def test_unload_models_no_pending_returns_false(self) -> None:
        sched = _make_scheduler()
        assert sched.unload_models() is False

    def test_unload_models_pending_job_returns_false_for_needed_model(self) -> None:
        jt = JobTracker()
        job = make_job_pop_response("stable_diffusion")
        jt.jobs_pending_inference.append(job)

        proc = make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({0: proc})

        sched = _make_scheduler(process_map=process_map, job_tracker=jt)
        assert sched.unload_models() is False
        assert proc.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_RAM

    def test_unload_models_single_thread_single_model_returns_false(self) -> None:
        jt = JobTracker()
        job = make_job_pop_response("stable_diffusion")
        jt.jobs_pending_inference.append(job)

        sched = _make_scheduler(job_tracker=jt)
        assert sched.unload_models() is False

    def test_get_next_n_models_returns_correct(self) -> None:
        jt = JobTracker()
        job1 = make_job_pop_response("model_a")
        job2 = make_job_pop_response("model_b")
        job3 = make_job_pop_response("model_a")
        jt.jobs_pending_inference.extend([job1, job2, job3])

        sched = _make_scheduler(job_tracker=jt)
        result = sched.get_next_n_models(3)
        assert result == ["model_a", "model_b"]

    def test_unload_from_ram_invalid_process_raises(self) -> None:
        sched = _make_scheduler()
        with pytest.raises(ValueError, match="not in the process map"):
            sched.unload_from_ram(99)

    def test_unload_from_ram_non_inference_warns(self) -> None:
        proc = make_mock_process_info(0, process_type=HordeProcessType.SAFETY)
        process_map = ProcessMap({0: proc})

        sched = _make_scheduler(process_map=process_map)
        sched.unload_from_ram(0)

    def test_unload_from_ram_recently_unloaded_skips(self) -> None:
        proc = make_mock_process_info(0)
        proc.recently_unloaded_from_ram = True
        process_map = ProcessMap({0: proc})

        sched = _make_scheduler(process_map=process_map)
        old_control_flag = proc.last_control_flag
        sched.unload_from_ram(0)
        assert proc.last_control_flag == old_control_flag


class TestGetSingleJobEffectiveMegapixelsteps:
    """Tests for get_single_job_effective_megapixelsteps."""

    def test_returns_value(self) -> None:
        sched = _make_scheduler()
        job = make_job_pop_response("stable_diffusion")

        result = sched.get_single_job_effective_megapixelsteps(job)
        assert isinstance(result, int)
        assert result > 0
