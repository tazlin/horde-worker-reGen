"""Tests for InferenceScheduler."""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.inference_scheduler import (
    _RESIDENCY_GRACE_SECONDS,
    InferenceScheduler,
)
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.lru_cache import LRUCache
from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_test_model_metadata,
    make_test_runtime_config,
    mark_job_in_progress_async,
    track_popped_job_async,
)


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
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        model_metadata=make_test_model_metadata(),
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

    async def test_model_already_loaded_returns_false(self) -> None:
        """Preload should return False if the needed model is already loaded in a process."""
        process_info = make_mock_process_info(0, model_name="stable_diffusion")
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        await track_popped_job_async(job_tracker, job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        assert inference_scheduler.preload_models() is False

    async def test_preload_sends_message_when_process_available(self) -> None:
        """Preload should send a message to a process to load the model if it's not already loaded."""
        process_info = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job = make_job_pop_response("new_model")
        await track_popped_job_async(job_tracker, job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        result = inference_scheduler.preload_models()
        assert result is True
        assert process_info.last_control_flag == HordeControlFlag.PRELOAD_MODEL

    async def test_no_available_process_returns_false(self) -> None:
        """Preload should return False if there are no available processes to load the model."""
        process_info = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job = make_job_pop_response("new_model")
        await track_popped_job_async(job_tracker, job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        assert inference_scheduler.preload_models() is False

    async def test_clears_preloaded_model_no_longer_needed(self) -> None:
        """If a model was preloaded for a job but that job is no longer pending, the model should be cleared."""
        process_info = make_mock_process_info(0, model_name="old_model", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({0: process_info})
        process_map.on_process_state_change = Mock()  # type: ignore
        job_tracker = JobTracker()

        job = make_job_pop_response("different_model")
        await track_popped_job_async(job_tracker, job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        inference_scheduler.preload_models()

        process_map.on_process_state_change.assert_called_with(  # type: ignore
            process_id=0,
            new_state=HordeProcessState.WAITING_FOR_JOB,
        )


class TestGetNextJobAndProcess:
    """Tests for get_next_job_and_process."""

    async def test_no_pending_jobs_returns_none(self) -> None:
        """get_next_job_and_process should return None if there are no pending jobs."""
        inference_scheduler = _make_inference_scheduler()
        assert await inference_scheduler.get_next_job_and_process() is None

    async def test_returns_job_with_matching_process(self) -> None:
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
        await track_popped_job_async(job_tracker, job)

        hmm.update_entry(
            horde_model_name="stable_diffusion",
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=0,
        )

        sched = _make_inference_scheduler(process_map=process_map, horde_model_map=hmm, job_tracker=job_tracker)
        result = await sched.get_next_job_and_process()
        assert result is not None
        assert result.next_job is job
        assert result.process_with_model is process_info

    async def test_no_process_with_model_returns_none(self) -> None:
        """get_next_job_and_process should return None if there is no process with the required model."""
        process_info = make_mock_process_info(0, model_name="other_model")
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        await track_popped_job_async(job_tracker, job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        assert await inference_scheduler.get_next_job_and_process() is None

    async def test_max_concurrent_reached_returns_none(self) -> None:
        """get_next_job_and_process should return None if the maximum number of concurrent jobs is reached."""
        process_info = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.PRELOADED_MODEL,
        )
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job_in_progress = make_job_pop_response("stable_diffusion")
        await mark_job_in_progress_async(job_tracker, job_in_progress)

        job = make_job_pop_response("stable_diffusion")
        await track_popped_job_async(job_tracker, job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        assert await inference_scheduler.get_next_job_and_process() is None

    async def test_skipped_line_is_returned_on_second_call(self) -> None:
        """A cached line-skip decision should be returned on the next call within the same cycle."""
        process_info = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.PRELOADED_MODEL,
        )
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        await track_popped_job_async(job_tracker, job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        cached = Mock()
        cached.next_job = job
        cached.process_with_model = process_info
        process_info.can_accept_job = Mock(return_value=True)  # type: ignore[method-assign]
        inference_scheduler._pending_line_skip = cached

        assert await inference_scheduler.get_next_job_and_process() is cached

    async def test_job_in_progress_is_skipped(self) -> None:
        """get_next_job_and_process should skip jobs that are already in progress."""
        process_info = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.PRELOADED_MODEL,
        )
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job = make_job_pop_response("stable_diffusion")
        await track_popped_job_async(job_tracker, job)
        await mark_job_in_progress_async(job_tracker, job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        assert await inference_scheduler.get_next_job_and_process() is None


class TestStartInference:
    """Tests for start_inference."""

    async def test_no_next_job_returns_false(self) -> None:
        """start_inference should return False if there is no next job."""
        inference_scheduler = _make_inference_scheduler()
        assert await inference_scheduler.start_inference() is False

    async def test_successful_start_adds_to_in_progress(self) -> None:
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
        await track_popped_job_async(job_tracker, job)

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
        result = await inference_scheduler.start_inference()
        assert result is True
        assert job in job_tracker.jobs_in_progress
        assert process_info.last_control_flag == HordeControlFlag.START_INFERENCE

    async def test_failed_send_faults_job(self) -> None:
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
        await track_popped_job_async(job_tracker, job)

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
        result = await inference_scheduler.start_inference()
        assert result is True
        assert job not in job_tracker.jobs_in_progress


class TestUnloadModels:
    """Tests for unload_models and related methods."""

    def test_unload_models_no_pending_returns_false(self) -> None:
        """unload_models should return False if there are no pending inference jobs."""
        inference_scheduler = _make_inference_scheduler()
        assert inference_scheduler.unload_models() is False

    async def test_unload_models_pending_job_returns_false_for_needed_model(self) -> None:
        """unload_models should return False if there is a pending job for the model."""
        job_tracker = JobTracker()
        job = make_job_pop_response("stable_diffusion")
        await track_popped_job_async(job_tracker, job)

        process_info = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.PRELOADED_MODEL,
        )
        process_map = ProcessMap({0: process_info})

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        assert inference_scheduler.unload_models() is False
        assert process_info.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_RAM

    async def test_unload_models_single_thread_single_model_returns_false(self) -> None:
        """If there is only one inference process and one model, unload_models should return False.

        This is because we don't want to unload the only model we have if we only have one process,
        as there is no benefit and it could cause issues if the process is slow to unload/load.
        """
        job_tracker = JobTracker()
        job = make_job_pop_response("stable_diffusion")
        await track_popped_job_async(job_tracker, job)

        inference_scheduler = _make_inference_scheduler(job_tracker=job_tracker)
        assert inference_scheduler.unload_models() is False

    async def test_get_next_n_models_returns_correct(self) -> None:
        """get_next_n_models should return a list of unique model names for the next n pending inference jobs."""
        job_tracker = JobTracker()
        job1 = make_job_pop_response("model_a")
        job2 = make_job_pop_response("model_b")
        job3 = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job1)
        await track_popped_job_async(job_tracker, job2)
        await track_popped_job_async(job_tracker, job3)

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


class TestSpeculativeDispatchCap:
    """Tests for _max_jobs_in_progress_allowed (lease-gated speculative pre-staging)."""

    def _vram_process_map(self, free_mb: int) -> ProcessMap:
        proc = make_mock_process_info(0)
        proc.total_vram_mb = 16000
        proc.vram_usage_mb = 16000 - free_mb
        return ProcessMap({0: proc})

    def test_base_cap_when_lease_disabled(self) -> None:
        """Without the lease the cap is the concurrent-sampling count (no pre-staging)."""
        scheduler = _make_inference_scheduler(
            process_map=self._vram_process_map(12000),
            max_concurrent=2,
            max_inference=4,
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=False),
        )
        assert scheduler._max_jobs_in_progress_allowed(0) == 2

    def test_cap_raised_to_all_processes_when_lease_on_and_vram_ample(self) -> None:
        """With the lease and free VRAM, spare processes may stage ahead up to the process count."""
        scheduler = _make_inference_scheduler(
            process_map=self._vram_process_map(12000),
            max_concurrent=2,
            max_inference=4,
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
        )
        assert scheduler._max_jobs_in_progress_allowed(0) == 4

    def test_cap_falls_back_when_vram_low(self) -> None:
        """Pre-staging is withheld when free VRAM is below the headroom threshold."""
        scheduler = _make_inference_scheduler(
            process_map=self._vram_process_map(1000),
            max_concurrent=2,
            max_inference=4,
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
        )
        assert scheduler._max_jobs_in_progress_allowed(0) == 2

    def test_cap_falls_back_when_vram_unknown(self) -> None:
        """With no VRAM report yet (cold start), do not speculate."""
        scheduler = _make_inference_scheduler(
            max_concurrent=2,
            max_inference=4,
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
        )
        assert scheduler._max_jobs_in_progress_allowed(0) == 2

    def test_post_processing_count_added(self) -> None:
        """Post-processing overlap slots extend the cap additively."""
        scheduler = _make_inference_scheduler(
            max_concurrent=2,
            max_inference=4,
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=False),
        )
        assert scheduler._max_jobs_in_progress_allowed(1) == 3


class TestGetSingleJobEffectiveMegapixelsteps:
    """Tests for get_single_job_effective_megapixelsteps."""

    def test_returns_value(self) -> None:
        """get_single_job_effective_megapixelsteps should return an integer value for a valid job."""
        inference_scheduler = _make_inference_scheduler()
        job = make_job_pop_response("stable_diffusion")

        result = inference_scheduler.get_single_job_effective_megapixelsteps(job)
        assert isinstance(result, int)
        assert result > 0


class TestWorkingSetResidency:
    """Tests for the working-set residency policy that prevents inter-job unload thrash."""

    async def test_compute_wanted_models_unions_live_state(self) -> None:
        """The wanted set is the union of resident models, pending-job models, and in-progress models."""
        job_tracker = JobTracker()
        pending = make_job_pop_response("model_pending")
        in_progress = make_job_pop_response("model_in_progress")
        await track_popped_job_async(job_tracker, pending)
        await track_popped_job_async(job_tracker, in_progress)
        await mark_job_in_progress_async(job_tracker, in_progress)

        process_map = ProcessMap(
            {
                0: make_mock_process_info(0, model_name="model_resident"),
                1: make_mock_process_info(1, model_name=None),
            },
        )

        scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        assert scheduler._compute_wanted_models() == {"model_resident", "model_pending", "model_in_progress"}

    async def test_refresh_and_recent_demand(self) -> None:
        """Refreshing demand stamps pending/in-progress models; recency reflects the grace window."""
        job_tracker = JobTracker()
        job = make_job_pop_response("hot_model")
        await track_popped_job_async(job_tracker, job)

        scheduler = _make_inference_scheduler(job_tracker=job_tracker)
        scheduler._refresh_model_demand()
        assert scheduler._is_recently_demanded("hot_model") is True
        assert scheduler._is_recently_demanded("never_seen") is False

    def test_recent_demand_expires_after_grace(self) -> None:
        """A model whose last demand predates the grace window is no longer recently demanded."""
        scheduler = _make_inference_scheduler()
        scheduler._model_last_in_demand["cold_model"] = time.time() - (_RESIDENCY_GRACE_SECONDS + 5)
        assert scheduler._is_recently_demanded("cold_model") is False

    def test_refresh_prunes_far_stale_entries(self) -> None:
        """Entries far past the grace window are pruned so the demand map cannot grow unbounded."""
        scheduler = _make_inference_scheduler()
        scheduler._model_last_in_demand["ancient"] = time.time() - (_RESIDENCY_GRACE_SECONDS * 10)
        scheduler._refresh_model_demand()
        assert "ancient" not in scheduler._model_last_in_demand

    def test_fits_regime_protects_ram_and_vram(self) -> None:
        """When the wanted set fits the process count, a wanted model is protected from both unloads."""
        scheduler = _make_inference_scheduler(max_inference=2)
        wanted = {"model_a", "model_b"}
        assert scheduler._residency_protects_from_unload("model_a", wanted, vram=False) is True
        assert scheduler._residency_protects_from_unload("model_a", wanted, vram=True) is True

    def test_overflow_regime_protects_only_ram_via_grace(self) -> None:
        """When models exceed processes, only recently-demanded RAM residency is protected, not VRAM."""
        scheduler = _make_inference_scheduler(max_inference=2)
        wanted = {"model_a", "model_b", "model_c"}  # 3 > 2 processes => affinity inactive
        scheduler._model_last_in_demand["model_a"] = time.time()
        assert scheduler._residency_protects_from_unload("model_a", wanted, vram=False) is True
        assert scheduler._residency_protects_from_unload("model_a", wanted, vram=True) is False

    def test_overflow_regime_evicts_stale_model(self) -> None:
        """An overflow model not recently demanded is not protected from RAM eviction."""
        scheduler = _make_inference_scheduler(max_inference=2)
        wanted = {"model_a", "model_b", "model_c"}
        assert scheduler._residency_protects_from_unload("model_a", wanted, vram=False) is False

    def test_none_model_never_protected(self) -> None:
        """A process with no loaded model is never protected."""
        scheduler = _make_inference_scheduler(max_inference=2)
        assert scheduler._residency_protects_from_unload(None, {"model_a"}, vram=False) is False

    async def test_unload_models_keeps_resident_working_set(self) -> None:
        """unload_models must not evict a resident working-set model just because its queue is momentarily empty."""
        job_tracker = JobTracker()
        # A pending job for model_b keeps the unload path active; model_a has no pending job but is resident.
        pending = make_job_pop_response("model_b")
        await track_popped_job_async(job_tracker, pending)

        process_map = ProcessMap(
            {
                0: make_mock_process_info(0, model_name="model_a"),
                1: make_mock_process_info(1, model_name="model_b"),
            },
        )
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry("model_a", load_state=ModelLoadState.LOADED_IN_RAM, process_id=0)
        horde_model_map.update_entry("model_b", load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
            max_concurrent=2,
            max_inference=2,
            bridge_data=make_mock_bridge_data(image_models_to_load=["model_a", "model_b"]),
        )

        assert scheduler.unload_models() is False
        assert process_map[0].last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_RAM
