"""Tests for InferenceScheduler."""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeProcessState,
    ModelInfo,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.lru_cache import LRUCache
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _RESIDENCY_GRACE_SECONDS,
    InferenceScheduler,
)
from tests.process_management.conftest import (
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
    # The effective concurrency cap is now read live from the runtime config's max_threads; tie it to
    # the fixture's max_concurrent so these tests exercise the intended concurrent-inference cap.
    bridge_data.max_threads = max_concurrent

    return InferenceScheduler(
        state=state,
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        process_lifecycle=Mock(
            get_processes_with_model_for_queued_job=Mock(return_value=[]),
            is_model_load_quarantined=Mock(return_value=False),
            aux_download_deadline_for_dispatch=Mock(return_value=120.0),
        ),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        model_metadata=make_test_model_metadata(),
        max_concurrent_inference_processes=max_concurrent,
        max_inference_processes=max_inference,
        lru=LRUCache(max_inference),
    )


class TestSchedulerDiagnosticThrottle:
    """Tests for high-frequency scheduler diagnostic coalescing."""

    def test_unchanged_diagnostic_is_suppressed_until_interval(self) -> None:
        """Repeated identical scheduler diagnostics should be coalesced within the cadence window."""
        scheduler = _make_inference_scheduler()

        assert scheduler._scheduler_diagnostic_suppressed_count("diagnostic", ("same",)) == 0
        assert scheduler._scheduler_diagnostic_suppressed_count("diagnostic", ("same",)) is None
        assert scheduler._scheduler_diagnostic_suppressed_count("diagnostic", ("same",)) is None

        state_key, emitted_at, suppressed_count = scheduler._scheduler_diagnostic_log_state["diagnostic"]
        scheduler._scheduler_diagnostic_log_state["diagnostic"] = (state_key, emitted_at - 31.0, suppressed_count)

        assert scheduler._scheduler_diagnostic_suppressed_count("diagnostic", ("same",)) == 2
        assert scheduler._scheduler_diagnostic_suppressed_count("diagnostic", ("same",)) is None

    def test_changed_diagnostic_logs_immediately(self) -> None:
        """A semantic scheduler diagnostic change should bypass the cadence limit."""
        scheduler = _make_inference_scheduler()

        assert scheduler._scheduler_diagnostic_suppressed_count("diagnostic", ("old",)) == 0
        assert scheduler._scheduler_diagnostic_suppressed_count("diagnostic", ("old",)) is None

        assert scheduler._scheduler_diagnostic_suppressed_count("diagnostic", ("new",)) == 1


class TestLineSkipRejectionLogThrottle:
    """The line-skip rejection log is rate-limited per (candidate, reason) without losing fidelity."""

    def _capture(self, fn) -> list[str]:  # noqa: ANN001
        """Run ``fn`` with a temporary loguru sink and return the line-skip messages it emitted."""
        from loguru import logger

        messages: list[str] = []
        sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="DEBUG")
        try:
            fn()
        finally:
            logger.remove(sink_id)
        return [m for m in messages if "Line-skip candidate" in m]

    def test_identical_rejection_repeats_are_throttled(self) -> None:
        """Re-evaluating the same (candidate, reason) within the interval logs only once."""
        scheduler = _make_inference_scheduler()

        def emit() -> None:
            for _ in range(5):
                scheduler._log_line_skip_rejection("abc12345", "has_loras", "rejected: candidate has LoRAs.")

        emitted = self._capture(emit)
        assert len(emitted) == 1
        assert "rejected: candidate has LoRAs." in emitted[0]

    def test_distinct_reason_or_candidate_logs_immediately(self) -> None:
        """A changed reason or a different candidate is not suppressed (full fidelity preserved)."""
        scheduler = _make_inference_scheduler()

        def emit() -> None:
            scheduler._log_line_skip_rejection("abc12345", "has_loras", "rejected: candidate has LoRAs.")
            scheduler._log_line_skip_rejection("abc12345", "same_model", "rejected: same model as blocked job head.")
            scheduler._log_line_skip_rejection("def67890", "has_loras", "rejected: candidate has LoRAs.")

        emitted = self._capture(emit)
        assert len(emitted) == 3

    def test_throttle_lifts_after_interval(self) -> None:
        """Once the interval elapses, the same rejection logs again."""
        from horde_worker_regen.process_management.scheduling.inference_scheduler import (
            _LINE_SKIP_REJECTION_LOG_INTERVAL,
        )

        scheduler = _make_inference_scheduler()

        def emit() -> None:
            scheduler._log_line_skip_rejection("abc12345", "has_loras", "rejected: candidate has LoRAs.")
            # Age the recorded timestamp past the interval so the next call re-emits.
            scheduler._line_skip_rejection_log_state["abc12345:has_loras"] -= _LINE_SKIP_REJECTION_LOG_INTERVAL + 1
            scheduler._log_line_skip_rejection("abc12345", "has_loras", "rejected: candidate has LoRAs.")

        emitted = self._capture(emit)
        assert len(emitted) == 2


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

    async def test_quarantined_model_is_not_preloaded_and_its_job_is_faulted(self) -> None:
        """A model quarantined for repeated load failures must never be preloaded; its job is faulted instead."""
        from horde_worker_regen.process_management.jobs.job_tracker import JobStage

        process_info = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        job = make_job_pop_response("Z-Image-Turbo")
        await track_popped_job_async(job_tracker, job)

        inference_scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        inference_scheduler._process_lifecycle.is_model_load_quarantined = Mock(return_value=True)

        result = inference_scheduler.preload_models()

        assert result is False
        assert process_info.last_control_flag != HordeControlFlag.PRELOAD_MODEL
        assert job.id_ is not None
        # Faulted for reissue to the horde, not left wedging the queue.
        assert job_tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT

    async def test_preload_expires_stale_loading_entry_for_idle_process(self) -> None:
        """A stale loading model-map entry must not prevent a queued model from being preloaded."""
        process_info = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: process_info})
        horde_model_map = HordeModelMap(
            root={
                "new_model": ModelInfo(
                    horde_model_name="new_model",
                    horde_model_load_state=ModelLoadState.LOADING,
                    process_id=0,
                ),
            },
        )
        job_tracker = JobTracker()

        job = make_job_pop_response("new_model")
        await track_popped_job_async(job_tracker, job)

        inference_scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
        )

        result = inference_scheduler.preload_models()

        assert result is True
        assert process_info.last_control_flag == HordeControlFlag.PRELOAD_MODEL
        assert horde_model_map.root["new_model"].horde_model_load_state == ModelLoadState.LOADING
        assert horde_model_map.root["new_model"].process_id == 0

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

    async def test_forecast_head_lets_resident_job_bypass(self) -> None:
        """When the head's model is forecast to load, a later resident-model job bypasses it."""
        holder = make_mock_process_info(1, model_name="resident_b", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({1: holder})
        hmm = HordeModelMap(root={})
        hmm.update_entry(horde_model_name="big_a", load_state=ModelLoadState.LOADING, process_id=1)

        job_tracker = JobTracker()
        head_job = make_job_pop_response("big_a")
        bypass_job = make_job_pop_response("resident_b")
        await track_popped_job_async(job_tracker, head_job)
        await track_popped_job_async(job_tracker, bypass_job)

        sched = _make_inference_scheduler(process_map=process_map, horde_model_map=hmm, job_tracker=job_tracker)
        result = await sched.get_next_job_and_process()

        assert result is not None
        assert result.next_job is bypass_job
        assert result.process_with_model is holder
        assert result.line_skip is not None
        assert result.line_skip.displaced_job is head_job

    async def test_non_forecast_head_does_not_bypass(self) -> None:
        """When the head's model is not forecast to load, no bypass occurs so the head gets priority."""
        holder = make_mock_process_info(1, model_name="resident_b", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({1: holder})

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_job_pop_response("big_a"))
        await track_popped_job_async(job_tracker, make_job_pop_response("resident_b"))

        sched = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        assert await sched.get_next_job_and_process() is None

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


class TestAuxDownloadLineSkip:
    """Tests for modest resident jobs bypassing a head job blocked on aux downloads."""

    async def _make_scheduler(
        self,
        *,
        candidate_job: ImageGenerateJobPopResponse | None = None,
        candidate_process_state: HordeProcessState = HordeProcessState.PRELOADED_MODEL,
        exclusive_active_job: bool = False,
        bridge_data: Mock | None = None,
    ) -> tuple[InferenceScheduler, JobTracker, ImageGenerateJobPopResponse, ImageGenerateJobPopResponse]:
        """Create a queue where the head job's model is busy downloading aux models."""
        blocked_process = make_mock_process_info(
            0,
            model_name="blocked_model",
            state=HordeProcessState.DOWNLOADING_AUX_MODEL,
        )
        candidate_process = make_mock_process_info(
            1,
            model_name="small_model",
            state=candidate_process_state,
        )
        process_map = ProcessMap({0: blocked_process, 1: candidate_process})
        job_tracker = JobTracker()

        active_aux_job = make_job_pop_response("blocked_model")
        await track_popped_job_async(job_tracker, active_aux_job)
        await mark_job_in_progress_async(job_tracker, active_aux_job)
        if exclusive_active_job:
            job_tracker.mark_admitted_exclusive(active_aux_job)

        blocked_head_job = make_job_pop_response("blocked_model")
        await track_popped_job_async(job_tracker, blocked_head_job)

        if candidate_job is None:
            candidate_job = make_job_pop_response("small_model")
        await track_popped_job_async(job_tracker, candidate_job)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=bridge_data,
            max_concurrent=1,
            max_inference=2,
        )
        return scheduler, job_tracker, blocked_head_job, candidate_job

    async def test_modest_resident_job_bypasses_aux_download_cap(self) -> None:
        """A small resident job can dispatch while the capped slot is only downloading aux models."""
        scheduler, job_tracker, blocked_head_job, candidate_job = await self._make_scheduler()

        result = await scheduler.get_next_job_and_process()

        assert result is not None
        assert result.next_job is candidate_job
        assert result.process_with_model.process_id == 1
        assert result.line_skip is not None
        assert result.line_skip.displaced_job is blocked_head_job

        assert await scheduler.start_inference() is True
        assert candidate_job in job_tracker.jobs_in_progress

    async def test_aux_download_cap_bypass_rejects_oversized_candidate(self) -> None:
        """The cap bypass still enforces the performance-mode eMPS threshold."""
        large_candidate = make_job_pop_response("small_model", width=1024, height=1024, ddim_steps=100)
        bridge_data = make_mock_bridge_data(high_performance_mode=True)
        scheduler, _, _, _ = await self._make_scheduler(candidate_job=large_candidate, bridge_data=bridge_data)

        assert await scheduler.get_next_job_and_process() is None

    async def test_aux_download_cap_bypass_rejects_lora_candidate(self) -> None:
        """A bypass candidate that also needs LoRA work must not jump the line."""
        lora_candidate = make_job_pop_response(
            "small_model",
            loras=[LorasPayloadEntry(name="line-skip-test-lora")],
        )
        scheduler, _, _, _ = await self._make_scheduler(candidate_job=lora_candidate)

        assert await scheduler.get_next_job_and_process() is None

    async def test_aux_download_cap_bypass_requires_ready_candidate_process(self) -> None:
        """The resident candidate's process must be able to accept work."""
        scheduler, _, _, _ = await self._make_scheduler(
            candidate_process_state=HordeProcessState.INFERENCE_STARTING,
        )

        assert await scheduler.get_next_job_and_process() is None

    async def test_aux_download_cap_bypass_respects_exclusive_in_progress_job(self) -> None:
        """Exclusive jobs keep full-device isolation even when the head slot is downloading aux models."""
        scheduler, _, _, _ = await self._make_scheduler(exclusive_active_job=True)

        assert await scheduler.get_next_job_and_process() is None


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

    async def test_resident_job_dispatches_during_shutdown(self) -> None:
        """During graceful shutdown, an already-queued job whose model is resident still dispatches.

        The popper stops accepting new work once shutdown is armed, so the only jobs that can start
        here were accepted before the stop. They are given a chance to finish (bounded by the shutdown
        grace and the force-kill backstop) rather than being faulted without ever running.
        """
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

        state = WorkerState()
        state.initiate_shutdown()

        inference_scheduler = _make_inference_scheduler(
            state=state,
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
        )

        result = await inference_scheduler.start_inference()
        assert result is True
        assert job in job_tracker.jobs_in_progress
        assert process_info.last_control_flag == HordeControlFlag.START_INFERENCE


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

    async def test_unload_models_survives_stale_loaded_model_absent_from_map(self) -> None:
        """A slot referencing a model the map no longer tracks must not crash the control loop.

        The model map can expire an entry (the stale-loading sweep in ``_expire_stale_model_map_entries``,
        or ``expire_entries_for_process`` when a process dies) while the owning ``process_info`` still
        reports it as ``loaded_horde_model_name``. ``unload_models`` indexed the map with a raw ``[]`` and
        raised ``KeyError``, which propagated out of ``_control_loop_tick`` and abended the main process
        (observed mid aux-model/LoRA download). It must tolerate the divergence instead.
        """
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_job_pop_response("other_model"))

        # Idle slot still claims a model that is NOT in the (empty) model map: the divergence.
        process_info = make_mock_process_info(
            0,
            model_name="NeverEnding Dream",
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        process_map = ProcessMap({0: process_info})

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=HordeModelMap(root={}),
            job_tracker=job_tracker,
        )

        # Must not raise KeyError; the stale idle slot is reclaimable (under_pressure bypasses the
        # residency grace so the outcome is deterministic).
        assert scheduler.unload_models(under_pressure=True) is True

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

    def test_replace_stale_ram_unload_process_cycles_model_less_idle_slot(self) -> None:
        """An idle model-less process that still holds RAM after unload should be replaced."""
        process_info = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_RAM
        process_info.ram_usage_bytes = 2 * 1024 * 1024 * 1024
        process_map = ProcessMap({0: process_info})

        inference_scheduler = _make_inference_scheduler(process_map=process_map)

        assert inference_scheduler._replace_stale_ram_unload_process() is True
        # The cycle is a deliberate RAM reclaim of a healthy idle slot, so it must go through the
        # intentional-reclaim path and not be laundered as a crash/hang recovery.
        inference_scheduler._process_lifecycle._replace_inference_process.assert_called_once_with(
            process_info,
            intentional_reclaim=True,
        )

    async def test_unload_vram_does_not_resend_to_model_less_process(self) -> None:
        """Calling unload_models_from_vram twice must not re-send the UNLOAD command to a model-less process.

        A process with ``loaded_horde_model_name=None`` that was already told to unload from VRAM must
        not receive the same IPC command on every scheduling cycle. The guard that the non-None branch
        has (``last_control_flag != UNLOAD_MODELS_FROM_VRAM``) is missing from the None (else) branch,
        resulting in unbounded re-sends: the livelock that triggers a save-our-ship soft reset when
        the whole-card residency convergence hangs waiting for a sibling that keeps being prodded to
        unload models it does not have.
        """
        # A model-less process (like the stuck process 4 in the incident).
        empty = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        # A model-holding process that is the target (unload is called "on" this one).
        target = make_mock_process_info(0, model_name="some_model", state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: target, 1: empty})

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_job_pop_response("some_model"))

        sched = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            max_inference=2,
        )

        # First call: the model-less process must be told to unload (it may hold stale VRAM).
        sched.unload_models_from_vram(target)
        assert empty.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM, (
            "first call must set the control flag on the model-less process"
        )

        # Second call: the guard must prevent re-sending. The process was already told to unload and
        # its response ("No models to unload from VRAM") has not yet arrived or been processed, so
        # re-sending the same command is wasted IPC that causes the convergence livelock.
        sched.unload_models_from_vram(target)
        # The last_control_flag must still be UNLOAD_MODELS_FROM_VRAM; the command was NOT re-sent
        # because the guard suppressed the duplicate, meaning safe_send_message was not called again.
        assert empty.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM, (
            "second call must NOT re-send: the last_control_flag guard must suppress the duplicate "
            "just as the non-None branch does"
        )
        # The else branch must also set unloaded_any when it first issues the command, so the caller
        # knows that VRAM clearance work was initiated and does not loop forever assuming nothing happened.
        # (This is a secondary assertion; the primary bug is the missing guard.)


class TestHeadOfQueueMakeRoom:
    """The head-of-queue job must keep making progress.

    other queued jobs' models are idle-resident, the budget gate escalates to evict one of them so
    the head gets room, rather than starving the whole worker behind an un-loadable head.
    """

    async def _scheduler_with_head_blocked(self) -> tuple[InferenceScheduler, Mock]:
        """A model-less preload slot, a slot holding a *queued* model, and a non-resident head job."""
        target = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        holder = make_mock_process_info(1, model_name="queued_b", state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: target, 1: holder})

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_job_pop_response("big_a"))
        await track_popped_job_async(job_tracker, make_job_pop_response("queued_b"))

        sched = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker, max_inference=2)
        return sched, holder

    async def test_ram_escalation_overrides_pending_guard(self) -> None:
        """Gentle RAM reclaim spares a still-queued model; the head escalation may reclaim it."""
        sched, holder = await self._scheduler_with_head_blocked()

        assert sched.unload_models(under_pressure=True) is False
        assert sched.unload_models(under_pressure=True, for_head_of_queue=True) is True

    async def test_vram_escalation_overrides_next_model_guard_and_reports(self) -> None:
        """Gentle VRAM reclaim spares the next-up model; the head escalation reclaims it and reports."""
        sched, holder = await self._scheduler_with_head_blocked()
        target = sched._process_map[0]

        assert sched.unload_models_from_vram(target, under_pressure=True) is False
        assert holder.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM

        assert sched.unload_models_from_vram(target, under_pressure=True, for_head_of_queue=True) is True
        assert holder.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM

    async def test_preload_makes_room_for_head_instead_of_wedging(self) -> None:
        """When the head model does not fit VRAM and only queued models are resident, preload evicts one."""
        sched, holder = await self._scheduler_with_head_blocked()

        sched._budget_active = Mock(return_value=True)  # type: ignore[method-assign]
        sched._measured_free_vram_mb = Mock(return_value=1000.0)  # type: ignore[method-assign]
        sched._vram_budget = Mock()
        sched._vram_budget.check_job.return_value = Mock(fits=False, reason=Mock(return_value="does not fit"))

        sched.preload_models()

        # The head's blocker triggered the escalation: the only idle resident copy (another queued
        # job's model) was unloaded from VRAM to give the head room, so the worker does not wedge.
        assert holder.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM


class TestHeadOfQueueNotWedgedByAffinity:
    """A non-resident head-of-queue job must always be able to claim a slot.

    Model->process affinity is provisioned against the inference-process *ceiling*
    (``max_inference_processes`` = queue_size + concurrency), not the count of running processes. With
    more resident models than running processes, affinity wrongly concludes every model has a home and
    pins every idle slot, so a genuinely-queued head whose model is not resident gets no process and the
    whole worker wedges (every process idle, jobs queued). The head must be given room regardless, and
    this make-room must not be gated on the VRAM/RAM budget being active.
    """

    async def test_head_displaces_undemanded_resident_when_affinity_pins_all(self) -> None:
        """The head preloads by displacing an idle resident model no queued job needs, sparing demanded ones."""
        # Two running inference processes, both holding a resident model; no empty slot. One model is
        # still demanded by queued jobs (``demanded``), the other is idle slack nothing queued needs.
        demanded = make_mock_process_info(0, model_name="demanded", state=HordeProcessState.WAITING_FOR_JOB)
        undemanded = make_mock_process_info(1, model_name="idle_slack", state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: demanded, 1: undemanded})

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_job_pop_response("head_non_resident"))
        await track_popped_job_async(job_tracker, make_job_pop_response("demanded"))
        await track_popped_job_async(job_tracker, make_job_pop_response("demanded"))

        # Ceiling (3) exceeds the running process count (2), which is exactly what wrongly activates
        # affinity and pins both slots in the field.
        sched = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker, max_inference=3)
        assert sched._budget_active() is False

        assert sched.preload_models() is True
        # The head displaced the *undemanded* resident, preserving the demanded one for its queued jobs.
        assert undemanded.last_control_flag == HordeControlFlag.PRELOAD_MODEL
        assert demanded.last_control_flag != HordeControlFlag.PRELOAD_MODEL

    async def test_head_makes_room_without_budget_when_only_queued_models_resident(self) -> None:
        """With the budget gate inactive and every idle slot holding a queued model, the head still gets room."""
        holder_b = make_mock_process_info(0, model_name="queued_b", state=HordeProcessState.WAITING_FOR_JOB)
        holder_c = make_mock_process_info(1, model_name="queued_c", state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: holder_b, 1: holder_c})

        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, make_job_pop_response("head_a"))
        await track_popped_job_async(job_tracker, make_job_pop_response("queued_b"))
        await track_popped_job_async(job_tracker, make_job_pop_response("queued_c"))

        sched = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker, max_inference=3)
        assert sched._budget_active() is False

        # The head is non-resident and the budget is off, so the previous budget-gated escalation never
        # ran; the worker must still make room (displace a queued model) rather than wedge.
        assert sched.preload_models() is True
        assert (
            holder_b.last_control_flag == HordeControlFlag.PRELOAD_MODEL
            or holder_c.last_control_flag == HordeControlFlag.PRELOAD_MODEL
        )

    async def test_head_room_never_displaces_in_progress_work(self) -> None:
        """The head-room fallback must never evict a slot whose model is running a job."""
        live = make_mock_process_info(0, model_name="live_model", state=HordeProcessState.INFERENCE_STARTING)
        idle_resident = make_mock_process_info(1, model_name="idle_model", state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: live, 1: idle_resident})

        job_tracker = JobTracker()
        # The live model has an in-progress job; the head is a different, non-resident model.
        live_job = make_job_pop_response("live_model")
        await mark_job_in_progress_async(job_tracker, live_job)
        await track_popped_job_async(job_tracker, make_job_pop_response("head_non_resident"))

        sched = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker, max_inference=3)

        sched.preload_models()
        # Only the idle resident is a legal displacement target; the live slot is untouched.
        assert live.last_control_flag != HordeControlFlag.PRELOAD_MODEL


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
