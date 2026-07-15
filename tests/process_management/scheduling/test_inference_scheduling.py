"""Tests for InferenceScheduler."""

from __future__ import annotations

import time
from collections.abc import Callable
from unittest.mock import Mock

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry, TIPayloadEntry

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime
from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeImageResult,
    HordeProcessState,
    ModelInfo,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.lru_cache import LRUCache
from horde_worker_regen.process_management.models.model_metadata import ModelMetadata
from horde_worker_regen.process_management.scheduling import inference_scheduler as _sched_mod
from horde_worker_regen.process_management.scheduling.dispatch_affinity import (
    _AFFINITY_MAX_SKIPS,
    AffinitySkipState,
    record_affinity_skip,
)
from horde_worker_regen.process_management.scheduling.governance import AdmissionDecision
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _PRELOAD_FIRST_REPORT_GRACE_SECONDS,
    _RESIDENCY_GRACE_SECONDS,
    InferenceScheduler,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_card_runtimes,
    make_test_model_metadata,
    make_test_runtime_config,
    mark_job_aux_prepared,
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
    card_runtimes: dict[int, CardRuntime] | None = None,
    model_metadata: ModelMetadata | None = None,
    post_processing_lane_commitments_provider: Callable[[], int] | None = None,
    device_free_mb: float | None = 24000.0,
) -> InferenceScheduler:
    """Build an InferenceScheduler with mostly-mocked dependencies.

    ``device_free_mb`` models the fake card's truthful device-free reading, the measured-truth admission
    identity's primary input: the default is an ample card so admission-neutral tests never defer on a
    missing reading. Tests exercising VRAM pressure pass a small figure (or install a crafted arbiter
    cycle), and tests exercising the missing-reading contract pass None.
    """
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

    scheduler = InferenceScheduler(
        state=state,
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=job_tracker,
        process_lifecycle=Mock(
            get_processes_with_model_for_queued_job=Mock(return_value=[]),
            is_model_load_quarantined=Mock(return_value=False),
        ),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        model_metadata=model_metadata if model_metadata is not None else make_test_model_metadata(),
        max_concurrent_inference_processes=max_concurrent,
        max_inference_processes=max_inference,
        lru=LRUCache(max_inference),
        post_processing_lane_commitments_provider=post_processing_lane_commitments_provider,
        card_runtimes=card_runtimes,
    )
    if device_free_mb is not None:
        scheduler.set_device_free_mb_provider(lambda _device_index: device_free_mb)
    return scheduler


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


class TestModelServiceabilityAdmission:
    """Scheduler guards stale model offers before child VRAM work starts."""

    async def test_unserviceable_late_arrival_faults_before_preload(self) -> None:
        """A stale SDXL job on an 8GB card is faulted without sending a child preload."""
        model = "sdxl_model"
        bridge_data = make_mock_bridge_data(image_models_to_load=[model])
        reference = {
            model: make_mock_model_reference_record(
                model,
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            ),
        }
        metadata = make_test_model_metadata(reference)
        process_map = ProcessMap({0: make_mock_process_info(0, model_name=None)})
        scheduler = _make_inference_scheduler(
            bridge_data=bridge_data,
            model_metadata=metadata,
            process_map=process_map,
            card_runtimes=make_test_card_runtimes(config=bridge_data, total_vram_mb=8192.0),
        )
        scheduler.set_admission_baseline_provider(lambda _device: 1024.0)
        job = make_job_pop_response(model)
        assert job.id_ is not None
        await track_popped_job_async(scheduler._job_tracker, job)

        outcome = scheduler._attempt_preload_for_job(job, head_job=job, loaded_models=set())

        assert outcome.name == "NEXT_JOB"
        latest = scheduler.latest_preload_admission()
        assert latest is not None
        assert latest.decision is AdmissionDecision.UNSERVICEABLE
        # The doomed job is faulted terminally through the existing fault machinery before any child preload:
        # its stage moves to PENDING_SUBMIT and the tracked job info carries the faulted generation state.
        assert scheduler._job_tracker.get_stage(job.id_) is JobStage.PENDING_SUBMIT
        assert scheduler._job_tracker.jobs_lookup[job].state is GENERATION_STATE.faulted
        assert (
            "model minimum footprint cannot fit any serving card" in scheduler._job_tracker.job_faults[job.id_][0].ref
        )
        assert all(not process.pipe_connection.send.called for process in scheduler._process_map.values())


class TestWddmPagingVictimAccessor:
    """The paging-victim accessor exposes a fresh victim map and ages a stale one out."""

    def test_active_verdict_exposes_victim_shared_mb_map(self) -> None:
        """An active verdict makes the per-PID shared-MB map readable while it is fresh."""
        scheduler = _make_inference_scheduler()

        scheduler.note_wddm_paging({100001: 512.0, 100002: 300.0}, active=True)

        assert scheduler.wddm_paging_victim_shared_mb_by_pid(5.0) == {100001: 512.0, 100002: 300.0}

    def test_stale_verdict_ages_out_to_empty(self) -> None:
        """A verdict older than the freshness window yields no victims, so a caller cannot act on it."""
        scheduler = _make_inference_scheduler()
        scheduler.note_wddm_paging({100001: 512.0}, active=True)

        # Backdate the recording stamp past any sane freshness window.
        scheduler._wddm_paging_victims_updated_monotonic -= 100.0

        assert scheduler.wddm_paging_victim_shared_mb_by_pid(5.0) == {}

    def test_cleared_verdict_yields_no_victims(self) -> None:
        """When paging clears (active=False), the victim map is emptied immediately."""
        scheduler = _make_inference_scheduler()
        scheduler.note_wddm_paging({100001: 512.0}, active=True)

        scheduler.note_wddm_paging({}, active=False)

        assert scheduler.wddm_paging_victim_shared_mb_by_pid(5.0) == {}


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

    async def test_resident_head_is_attempted_before_later_preload(self) -> None:
        """A resident queue head gets the scheduling cycle before later models are preloaded."""
        head_process = make_mock_process_info(
            0,
            model_name="head_model",
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        later_process = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: head_process, 1: later_process})
        job_tracker = JobTracker()

        await track_popped_job_async(job_tracker, make_job_pop_response("head_model"))
        await track_popped_job_async(job_tracker, make_job_pop_response("later_model"))

        inference_scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            max_concurrent=1,
            max_inference=2,
        )

        result = inference_scheduler.preload_models()

        assert result is False
        assert later_process.last_control_flag != HordeControlFlag.PRELOAD_MODEL

    async def test_pending_post_processing_holds_speculative_preload(self, monkeypatch: object) -> None:
        """A pending chain gets a drain window before another model is staged."""
        monkeypatch.setattr(  # type: ignore[attr-defined]
            _sched_mod,
            "predict_job_post_processing_vram_mb",
            lambda _job, _baseline_name: 4000.0,
        )
        inference_process = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        post_process_lane = make_mock_process_info(
            7,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.POST_PROCESS,
        )
        process_map = ProcessMap({0: inference_process, 7: post_process_lane})
        job_tracker = JobTracker()

        await track_popped_job_async(job_tracker, make_job_pop_response("new_model"))
        pp_job = make_job_pop_response("stable_diffusion", post_processing=["RealESRGAN_x4plus"])
        pp_job_info = HordeJobInfo(
            sdk_api_job_info=pp_job,
            job_image_results=[HordeImageResult(image_bytes=b"raw-image")],
            state=GENERATION_STATE.ok,
            censored=False,
            time_popped=time.time(),
        )
        await job_tracker.queue_for_post_processing(pp_job_info)

        inference_scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            max_concurrent=1,
            max_inference=1,
        )

        result = inference_scheduler.preload_models()

        assert result is False
        assert inference_process.last_control_flag != HordeControlFlag.PRELOAD_MODEL

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

    @pytest.mark.parametrize(
        "process_state",
        [
            HordeProcessState.DOWNLOADING_MODEL,
            HordeProcessState.PRELOADING_MODEL,
            HordeProcessState.UNLOADED_MODEL_FROM_RAM,
        ],
    )
    def test_loading_entry_survives_valid_loading_owner_states(self, process_state: HordeProcessState) -> None:
        """A process mid-load (or freshly RAM-unloaded) is a valid owner of a LOADING model-map entry."""
        process_info = make_mock_process_info(0, model_name="new_model", state=process_state)
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
        inference_scheduler = _make_inference_scheduler(process_map=process_map, horde_model_map=horde_model_map)

        expired = inference_scheduler._expire_stale_model_map_entries()

        assert expired == []
        assert "new_model" in horde_model_map.root

    async def test_recent_preload_request_graces_waiting_for_first_state_report(self) -> None:
        """A just-sent PRELOAD_MODEL must not expire before the child publishes its first preload state."""
        process_info = make_mock_process_info(0, model_name="new_model", state=HordeProcessState.WAITING_FOR_JOB)
        process_info.last_control_flag = HordeControlFlag.PRELOAD_MODEL
        process_info.last_preload_requested_at = time.time()
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

        assert result is False
        assert "new_model" in horde_model_map.root
        assert horde_model_map.root["new_model"].horde_model_load_state == ModelLoadState.LOADING

    def test_old_preload_request_does_not_grace_stale_idle_loading_entry(self) -> None:
        """The first-report grace is bounded so abandoned loading entries can still expire."""
        process_info = make_mock_process_info(0, model_name="new_model", state=HordeProcessState.WAITING_FOR_JOB)
        process_info.last_control_flag = HordeControlFlag.PRELOAD_MODEL
        process_info.last_preload_requested_at = time.time() - _PRELOAD_FIRST_REPORT_GRACE_SECONDS - 1.0
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
        inference_scheduler = _make_inference_scheduler(process_map=process_map, horde_model_map=horde_model_map)

        expired = inference_scheduler._expire_stale_model_map_entries()

        assert expired == ["new_model"]
        assert "new_model" not in horde_model_map.root

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

    async def test_non_forecast_head_bypassed_within_budget(self) -> None:
        """Within the affinity budget a resident-model job now passes a cold non-forecast head (card stays fed).

        Previously a non-forecast head was never bypassed; the head's preload-defer window left a resident job
        idle behind it. The affinity skip budget inverts that inside the window: the resident job dispatches and
        the head keeps its queue position via a ``resident_bypass`` line-skip.
        """
        holder = make_mock_process_info(1, model_name="resident_b", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({1: holder})

        job_tracker = JobTracker()
        head_job = make_job_pop_response("big_a")
        bypass_job = make_job_pop_response("resident_b")
        await track_popped_job_async(job_tracker, head_job)
        await track_popped_job_async(job_tracker, bypass_job)

        sched = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        result = await sched.get_next_job_and_process()

        assert result is not None
        assert result.next_job is bypass_job
        assert result.process_with_model is holder
        assert result.line_skip is not None
        assert result.line_skip.displaced_job is head_job
        assert result.line_skip.reason == "resident_bypass"

    async def test_non_forecast_head_reclaims_after_skip_budget_exhausted(self) -> None:
        """Once the skip ceiling is hit the cold head is no longer bypassed and reclaims the fall-through path.

        This is the old non-forecast protection, its precondition moved: the head is protected once the bound is
        spent, not unconditionally. With no bypass and its model neither resident nor loading, dispatch defers so
        the room-making machinery runs for the head.
        """
        holder = make_mock_process_info(1, model_name="resident_b", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({1: holder})

        job_tracker = JobTracker()
        head_job = make_job_pop_response("big_a")
        bypass_job = make_job_pop_response("resident_b")
        await track_popped_job_async(job_tracker, head_job)
        await track_popped_job_async(job_tracker, bypass_job)

        sched = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        sched._affinity_skip_state = AffinitySkipState(
            head_job_id=str(head_job.id_),
            first_skip_time=time.time(),
            skip_count=_AFFINITY_MAX_SKIPS,
        )
        assert await sched.get_next_job_and_process() is None

    async def test_non_forecast_head_reclaims_after_budget_window_elapses(self) -> None:
        """Once the wall-clock window elapses the cold head reclaims, even under the skip ceiling."""
        holder = make_mock_process_info(1, model_name="resident_b", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({1: holder})

        job_tracker = JobTracker()
        head_job = make_job_pop_response("big_a")
        bypass_job = make_job_pop_response("resident_b")
        await track_popped_job_async(job_tracker, head_job)
        await track_popped_job_async(job_tracker, bypass_job)

        sched = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        sched._affinity_skip_state = AffinitySkipState(
            head_job_id=str(head_job.id_),
            first_skip_time=time.time() - 3600.0,
            skip_count=1,
        )
        assert await sched.get_next_job_and_process() is None

    async def test_forecast_head_bypass_is_bounded_by_budget(self) -> None:
        """The forecast-to-load bypass is now bounded too: an exhausted budget stops it (was unbounded before).

        This closes the latent ttl-aging hole where a forecast head could be bypassed forever. With the budget
        spent and the head's model still loading, no bypass occurs and the head keeps its position.
        """
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
        sched._affinity_skip_state = AffinitySkipState(
            head_job_id=str(head_job.id_),
            first_skip_time=time.time(),
            skip_count=_AFFINITY_MAX_SKIPS,
        )
        assert await sched.get_next_job_and_process() is None

    async def test_pin_wait_head_bypasses_regardless_of_budget(self) -> None:
        """Control: a pin-waiting head bypasses even with the affinity budget spent (it funds no fresh copy)."""
        holder = make_mock_process_info(1, model_name="resident_b", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({1: holder})

        job_tracker = JobTracker()
        head_job = make_job_pop_response("big_a")
        bypass_job = make_job_pop_response("resident_b")
        await track_popped_job_async(job_tracker, head_job)
        await track_popped_job_async(job_tracker, bypass_job)

        sched = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        sched._affinity_skip_state = AffinitySkipState(
            head_job_id=str(head_job.id_),
            first_skip_time=time.time(),
            skip_count=_AFFINITY_MAX_SKIPS,
        )
        sched._pinned_lane_resident_for_job = Mock(return_value=Mock())  # type: ignore[method-assign]

        result = await sched.get_next_job_and_process()
        assert result is not None
        assert result.next_job is bypass_job
        assert result.line_skip is not None
        assert result.line_skip.reason == "resident_bypass"

    async def test_information_only_does_not_advance_affinity_window(self) -> None:
        """The look-ahead call surfaces the bypass but must not advance the skip window (only dispatch does)."""
        holder = make_mock_process_info(1, model_name="resident_b", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({1: holder})

        job_tracker = JobTracker()
        head_job = make_job_pop_response("big_a")
        bypass_job = make_job_pop_response("resident_b")
        await track_popped_job_async(job_tracker, head_job)
        await track_popped_job_async(job_tracker, bypass_job)

        sched = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        before = sched._affinity_skip_state

        result = await sched.get_next_job_and_process(information_only=True)
        assert result is not None
        assert result.line_skip is not None
        assert sched._affinity_skip_state is before
        assert sched.latest_affinity_skips() == 0

    async def test_resident_head_dispatches_immediately_even_mid_budget(self) -> None:
        """A head whose model is resident dispatches directly, regardless of an in-progress affinity window."""
        head_holder = make_mock_process_info(0, model_name="big_a", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({0: head_holder})

        job_tracker = JobTracker()
        head_job = make_job_pop_response("big_a")
        await track_popped_job_async(job_tracker, head_job)

        sched = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        sched._affinity_skip_state = AffinitySkipState(
            head_job_id=str(head_job.id_),
            first_skip_time=time.time(),
            skip_count=3,
        )

        result = await sched.get_next_job_and_process()
        assert result is not None
        assert result.next_job is head_job
        assert result.line_skip is None

    async def test_cold_head_reclaims_against_a_steady_resident_stream(self) -> None:
        """Liveness: a cold head passed by a steady resident stream still reclaims the slot within the bound.

        An unbounded bypass would starve the cold head forever behind never-ending resident work. Each cycle the
        resident job would bypass the head; a committed bypass advances the skip window exactly as
        ``start_inference`` does. After the bound is spent the stream can no longer bypass, and once the head's
        model is made resident the head itself dispatches. The ``_pending_line_skip`` cache is cleared each
        iteration to model the per-cycle re-evaluation ``run_scheduling_cycle`` performs.
        """
        holder = make_mock_process_info(1, model_name="resident_b", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({1: holder})

        job_tracker = JobTracker()
        head_job = make_job_pop_response("big_a")
        bypass_job = make_job_pop_response("resident_b")
        await track_popped_job_async(job_tracker, head_job)
        await track_popped_job_async(job_tracker, bypass_job)

        sched = _make_inference_scheduler(process_map=process_map, job_tracker=job_tracker)
        head_id = str(head_job.id_)
        now = time.time()

        for _ in range(_AFFINITY_MAX_SKIPS):
            sched._pending_line_skip = None
            result = await sched.get_next_job_and_process()
            assert result is not None
            assert result.next_job is bypass_job
            assert result.line_skip is not None
            assert result.line_skip.reason == "resident_bypass"
            sched._affinity_skip_state = record_affinity_skip(sched._affinity_skip_state, head_id, now)
            now += 1.0

        # The bound is spent: the stream can no longer bypass, so the head reclaims the fall-through path.
        sched._pending_line_skip = None
        assert await sched.get_next_job_and_process() is None

        # Once room is made and the head's model becomes resident, the head itself dispatches (not the stream).
        head_holder = make_mock_process_info(0, model_name="big_a", state=HordeProcessState.PRELOADED_MODEL)
        sched._process_map = ProcessMap({0: head_holder, 1: holder})
        sched._pending_line_skip = None
        reclaimed = await sched.get_next_job_and_process()
        assert reclaimed is not None
        assert reclaimed.next_job is head_job
        assert reclaimed.line_skip is None

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


class TestAuxGatedSiblingDispatch:
    """A gated auxiliary head holds no lane, so a fitting same-model sibling becomes the dispatch head."""

    async def test_same_model_non_aux_sibling_is_fed_while_lora_head_gated(self) -> None:
        """A same-model non-aux job is fed onto an idle sibling while an unprepared LoRA head stays gated.

        A mono-model queue whose head carries not-yet-prepared LoRAs would otherwise starve the GPU. The LoRA
        head is invisible to dispatch (it holds no lane until its prefetch clears its gate), so a same-model
        non-LoRA job simply becomes the dispatch head and is fed onto the idle sibling lane, keeping the card
        sampling. The gated LoRA head is never selected.
        """
        idle_sibling = make_mock_process_info(2, model_name="blocked_model", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({2: idle_sibling})
        job_tracker = JobTracker()

        lora_head = make_job_pop_response("blocked_model", loras=[LorasPayloadEntry(name="lora-b")])
        await track_popped_job_async(job_tracker, lora_head)

        same_model_candidate = make_job_pop_response("blocked_model")
        await track_popped_job_async(job_tracker, same_model_candidate)

        scheduler = _make_inference_scheduler(
            process_map=process_map, job_tracker=job_tracker, max_concurrent=2, max_inference=2
        )

        result = await scheduler.get_next_job_and_process()

        assert result is not None
        assert result.next_job is same_model_candidate
        assert result.process_with_model.process_id == 2
        # No line-skip is needed: the gated LoRA head is not the dispatch head, so the sibling is the head.
        assert result.line_skip is None
        assert result.next_job is not lora_head


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

    async def test_pending_post_processing_holds_next_non_coresident_sampler(self, monkeypatch: object) -> None:
        """Pending post-processing gets a drain window before another sampler that cannot share the card."""
        monkeypatch.setattr(  # type: ignore[attr-defined]
            _sched_mod,
            "predict_job_sampling_vram_mb",
            lambda _job, _baseline: 8000.0,
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            _sched_mod,
            "predict_job_post_processing_vram_mb",
            lambda _job, _baseline_name: 4000.0,
        )
        target_process = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.PRELOADED_MODEL,
        )
        sampling_process = make_mock_process_info(
            1,
            model_name="stable_diffusion",
            state=HordeProcessState.INFERENCE_STARTING,
        )
        post_process_lane = make_mock_process_info(
            7,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.POST_PROCESS,
        )
        process_map = ProcessMap({0: target_process, 1: sampling_process, 7: post_process_lane})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name="stable_diffusion",
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=0,
        )
        job_tracker = JobTracker()
        active_job = make_job_pop_response("stable_diffusion")
        next_job = make_job_pop_response("stable_diffusion")
        await track_popped_job_async(job_tracker, active_job)
        await job_tracker.mark_inference_started(active_job, device_index=None)
        await track_popped_job_async(job_tracker, next_job)

        pp_job = make_job_pop_response("stable_diffusion", post_processing=["RealESRGAN_x4plus"])
        pp_job_info = HordeJobInfo(
            sdk_api_job_info=pp_job,
            job_image_results=[HordeImageResult(image_bytes=b"raw-image")],
            state=GENERATION_STATE.ok,
            censored=False,
            time_popped=time.time(),
        )
        await job_tracker.queue_for_post_processing(pp_job_info)

        inference_scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
            max_concurrent=2,
            # A pressured card: the measured-truth second admission path also withholds, so the hold is
            # genuine (both the static reported-total gate and the measured reading refuse the overlap).
            device_free_mb=10000.0,
        )
        inference_scheduler.pp_sampling_coresidency_affordable = Mock(return_value=False)  # type: ignore[method-assign]

        result = await inference_scheduler.start_inference()

        assert result is False
        assert next_job not in job_tracker.jobs_in_progress
        assert target_process.last_control_flag != HordeControlFlag.START_INFERENCE

    async def test_pending_post_processing_holds_next_sampler_after_card_goes_idle(self, monkeypatch: object) -> None:
        """A feasible pending chain gets the next card turn after sampling drains."""
        monkeypatch.setattr(  # type: ignore[attr-defined]
            _sched_mod,
            "predict_job_sampling_vram_mb",
            lambda _job, _baseline: 8000.0,
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            _sched_mod,
            "predict_job_post_processing_vram_mb",
            lambda _job, _baseline_name: 4000.0,
        )
        target_process = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.PRELOADED_MODEL,
        )
        post_process_lane = make_mock_process_info(
            7,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.POST_PROCESS,
        )
        process_map = ProcessMap({0: target_process, 7: post_process_lane})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name="stable_diffusion",
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=0,
        )
        job_tracker = JobTracker()
        next_job = make_job_pop_response("stable_diffusion")
        await track_popped_job_async(job_tracker, next_job)

        pp_job = make_job_pop_response("stable_diffusion", post_processing=["RealESRGAN_x4plus"])
        pp_job_info = HordeJobInfo(
            sdk_api_job_info=pp_job,
            job_image_results=[HordeImageResult(image_bytes=b"raw-image")],
            state=GENERATION_STATE.ok,
            censored=False,
            time_popped=time.time(),
        )
        await job_tracker.queue_for_post_processing(pp_job_info)

        inference_scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
            max_concurrent=2,
            # A pressured card: the measured-truth second admission path also withholds, so the hold is
            # genuine (both the static reported-total gate and the measured reading refuse the overlap).
            device_free_mb=10000.0,
        )
        inference_scheduler.pp_sampling_coresidency_affordable = Mock(return_value=False)  # type: ignore[method-assign]

        result = await inference_scheduler.start_inference()

        assert result is False
        assert next_job not in job_tracker.jobs_in_progress
        assert target_process.last_control_flag != HordeControlFlag.START_INFERENCE
        inference_scheduler.pp_sampling_coresidency_affordable.assert_called_once_with(
            sampling_peak_mb=8000.0,
            pp_reserve_mb=4000.0,
            device_index=0,
        )


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
        # A fresh committed reservation that over-commits the card, so the arbiter's measured floor denies the
        # head and describes evicting this idle resident (another queued job's model).
        holder.total_vram_mb = 16000
        holder.process_reserved_mb = 16000
        holder.report_sampled_at = time.time()
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
        # The truthful device-free reading models the same pressure the predictive gate reports, so admission
        # genuinely does not fit and the head's blocker escalates to the idle-resident eviction.
        sched.set_device_free_mb_provider(lambda _device_index: 1000.0)
        sched._vram_budget = Mock()
        sched._vram_budget.check_job.return_value = Mock(
            fits=False,
            predicted_mb=None,
            reserve_mb=2000.0,
            reason=Mock(return_value="does not fit"),
        )

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
        assert scheduler._max_jobs_in_progress_allowed() == 2

    def test_cap_raised_to_all_processes_when_lease_on_and_vram_ample(self) -> None:
        """With the lease and free VRAM, spare processes may stage ahead up to the process count."""
        scheduler = _make_inference_scheduler(
            process_map=self._vram_process_map(12000),
            max_concurrent=2,
            max_inference=4,
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
        )
        assert scheduler._max_jobs_in_progress_allowed() == 4

    def test_cap_falls_back_when_vram_low(self) -> None:
        """Pre-staging is withheld when free VRAM is below the headroom threshold."""
        scheduler = _make_inference_scheduler(
            process_map=self._vram_process_map(1000),
            max_concurrent=2,
            max_inference=4,
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
        )
        assert scheduler._max_jobs_in_progress_allowed() == 2

    def test_cap_falls_back_when_vram_unknown(self) -> None:
        """With no VRAM report yet (cold start), do not speculate."""
        scheduler = _make_inference_scheduler(
            max_concurrent=2,
            max_inference=4,
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
        )
        assert scheduler._max_jobs_in_progress_allowed() == 2


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


class TestAuxPreparationGate:
    """The dispatch gate for jobs carrying auxiliary files (LoRAs and textual inversions).

    A job whose auxiliary files are not yet prepared must not dispatch: it holds no lane and no reservation
    (start_inference returns without adding it to jobs_in_progress) and instead has its files prepared while
    it stays pending. Once its prepared flag is set, ordinary admission proceeds and it is GPU-fed. The gate
    covers textual inversions the same way it covers LoRAs.
    """

    def _lora(self, name: str = "styleA") -> LorasPayloadEntry:
        return LorasPayloadEntry(name=name, model=1.0, clip=1.0, is_version=False)

    async def _resident(
        self,
        job: ImageGenerateJobPopResponse,
    ) -> tuple[InferenceScheduler, ProcessMap, JobTracker]:
        """Build a scheduler where ``job``'s model is resident on an idle preloaded process."""
        process_info = make_mock_process_info(0, model_name=job.model, state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({0: process_info})
        horde_model_map = HordeModelMap(root={})
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, job)
        horde_model_map.update_entry(
            horde_model_name=job.model,
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=0,
        )
        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
        )
        return scheduler, process_map, job_tracker

    async def test_unprepared_lora_head_holds_no_lane(self) -> None:
        """An unprepared LoRA head does not dispatch: it stays pending, unclaimed, and holds no lane.

        Preparation is the pop-time prefetch pipeline's job; the scheduler never dispatches the gated head nor
        sends it any control message, so it holds no lane and no reservation while its files are placed on disk.
        """
        job = make_job_pop_response("stable_diffusion", loras=[self._lora()])
        scheduler, process_map, job_tracker = await self._resident(job)

        assert await scheduler.start_inference() is False
        assert job not in job_tracker.jobs_in_progress
        assert job_tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
        assert process_map[0].last_control_flag != HordeControlFlag.START_INFERENCE

    async def test_prepared_lora_head_is_gpu_fed(self) -> None:
        """Once its aux-prepared flag is set, the LoRA head dispatches and enters progress."""
        job = make_job_pop_response("stable_diffusion", loras=[self._lora()])
        scheduler, process_map, job_tracker = await self._resident(job)
        mark_job_aux_prepared(job_tracker, job)

        assert await scheduler.start_inference() is True
        assert job in job_tracker.jobs_in_progress
        assert process_map[0].last_control_flag == HordeControlFlag.START_INFERENCE

    async def test_unprepared_ti_head_holds_no_lane(self) -> None:
        """A textual-inversion job gates exactly like a LoRA job: unprepared, it does not dispatch."""
        job = make_job_pop_response("stable_diffusion", tis=[TIPayloadEntry(name="emb-1")])
        scheduler, process_map, job_tracker = await self._resident(job)

        assert await scheduler.start_inference() is False
        assert job not in job_tracker.jobs_in_progress
        assert process_map[0].last_control_flag != HordeControlFlag.START_INFERENCE

    async def test_prepared_ti_head_is_gpu_fed(self) -> None:
        """A prepared textual-inversion job dispatches like any other resident job."""
        job = make_job_pop_response("stable_diffusion", tis=[TIPayloadEntry(name="emb-1")])
        scheduler, _process_map, job_tracker = await self._resident(job)
        mark_job_aux_prepared(job_tracker, job)

        assert await scheduler.start_inference() is True
        assert job in job_tracker.jobs_in_progress

    async def test_non_aux_sibling_dispatches_while_a_lora_job_is_unprepared(self) -> None:
        """A job with no auxiliary files is never gated: it is GPU-fed even while a LoRA job awaits preparation.

        Both jobs' models are resident on their own idle lanes. The plain job carries no reservation-holding
        dependency on the LoRA job's preparation, so the scheduler feeds it across successive ticks.
        """
        plain = make_job_pop_response("plain_model")
        lora_job = make_job_pop_response("lora_model", loras=[self._lora()])
        plain_proc = make_mock_process_info(0, model_name="plain_model", state=HordeProcessState.PRELOADED_MODEL)
        lora_proc = make_mock_process_info(1, model_name="lora_model", state=HordeProcessState.PRELOADED_MODEL)
        process_map = ProcessMap({0: plain_proc, 1: lora_proc})
        horde_model_map = HordeModelMap(root={})
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, lora_job)
        await track_popped_job_async(job_tracker, plain)
        horde_model_map.update_entry("plain_model", load_state=ModelLoadState.LOADED_IN_RAM, process_id=0)
        horde_model_map.update_entry("lora_model", load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
            max_concurrent=2,
            max_inference=2,
            bridge_data=make_mock_bridge_data(image_models_to_load=["plain_model", "lora_model"]),
        )

        # Drive the scheduler until the plain job is fed or a bounded number of ticks elapse; the unprepared
        # LoRA job must never seize a lane in the meantime.
        for _ in range(4):
            await scheduler.start_inference()
            if plain in job_tracker.jobs_in_progress:
                break
        assert plain in job_tracker.jobs_in_progress
        assert lora_job not in job_tracker.jobs_in_progress

    async def test_cold_model_sibling_preloads_and_samples_while_gated_head_waits(self) -> None:
        """A cold-model sibling is preloaded and sampled while the gated LoRA head reserves nothing.

        The gated head's base model is resident, but the head cannot sample until its prefetch clears its
        gate: it holds no lane and no VRAM reservation. A sibling for a not-yet-resident model must therefore
        be selectable for preload and then dispatch, proving the gated head never reserves capacity that would
        block the sibling's cold load. This encodes the no-reservation-during-download contract.
        """
        resident = "resident_lora_model"
        cold = "cold_sibling_model"
        lora_head = make_job_pop_response(resident, loras=[self._lora()])
        cold_sibling = make_job_pop_response(cold)

        head_lane = make_mock_process_info(0, model_name=resident, state=HordeProcessState.PRELOADED_MODEL)
        idle_lane = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: head_lane, 1: idle_lane})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(resident, load_state=ModelLoadState.LOADED_IN_RAM, process_id=0)
        job_tracker = JobTracker()
        await track_popped_job_async(job_tracker, lora_head)
        await track_popped_job_async(job_tracker, cold_sibling)

        reference = {
            resident: make_mock_model_reference_record(resident),
            cold: make_mock_model_reference_record(cold),
        }
        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
            max_concurrent=2,
            max_inference=2,
            model_metadata=make_test_model_metadata(reference),
            bridge_data=make_mock_bridge_data(image_models_to_load=[resident, cold]),
        )

        # The cold sibling is selected for preload (the gated head is invisible to preload and holds nothing).
        assert scheduler.preload_models() is True
        assert idle_lane.loaded_horde_model_name == cold
        assert lora_head not in job_tracker.jobs_in_progress

        # Once the sibling's model materialises, it dispatches and samples while the head still waits.
        idle_lane.last_process_state = HordeProcessState.PRELOADED_MODEL
        idle_lane.loaded_horde_model_name = cold
        horde_model_map.update_entry(cold, load_state=ModelLoadState.LOADED_IN_RAM, process_id=1)

        for _ in range(4):
            await scheduler.start_inference()
            if cold_sibling in job_tracker.jobs_in_progress:
                break
        assert cold_sibling in job_tracker.jobs_in_progress
        assert lora_head not in job_tracker.jobs_in_progress
        assert job_tracker.get_stage(lora_head.id_) == JobStage.PENDING_INFERENCE
