"""RED repro: concurrent inference dispatch ignores in-flight progress and model size.

The concurrency cap (``max_threads``) is a pure count: as soon as fewer than ``max_threads`` jobs
are in progress and a process holding the next job's model can accept work, the scheduler dispatches
that job, no matter what the already-running job(s) are doing. Two consequences fall out of that:

* The second job can start while the first is still at zero step progress (or even still loading its
  weights). When both jobs are heavy (SDXL) this stacks two weight loads and two activation peaks onto
  the device at the same instant. The card thrashes, the first job's sampling stalls mid-loop, and the
  step-timeout watchdog tears the slot down and faults its job: a process "recovery" that the worker
  should never have provoked.
* Model size is not considered at all. SD1.5 jobs are cheap enough to sample together, but two SDXL
  jobs (or anything extra-large, or a batched job) need the running job to be well underway, or the
  whole card, before another sampler joins.

These tests assert the *desired* policy: overlap is gated on the running job's progress, scaled by the
baseline/size of both the running and the candidate job. They fail against the current count-only cap
(the second job is admitted immediately) and should pass once a progress-and-size-aware overlap gate
is in place.
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.consts import VRAM_HEAVY_MODELS
from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.inference_scheduler import (
    InferenceScheduler,
    _ModelSizeTier,
)
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.lru_cache import LRUCache
from horde_worker_regen.process_management.messages import HordeProcessState
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_model_metadata,
    make_test_runtime_config,
    mark_job_in_progress_async,
    track_popped_job_async,
)

_SD15 = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1
_SD2 = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_768
_SDXL = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl
_FLUX = KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell
_CASCADE = KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade
_QWEN = KNOWN_IMAGE_GENERATION_BASELINE.qwen_image


def _make_scheduler(
    *,
    process_map: ProcessMap | None = None,
    job_tracker: JobTracker | None = None,
    reference: dict[str, object] | None = None,
    max_concurrent: int = 2,
) -> InferenceScheduler:
    """Build a scheduler with mostly-mocked dependencies and an optional model reference."""
    bridge_data = make_mock_bridge_data()
    bridge_data.max_threads = max_concurrent
    return InferenceScheduler(
        state=WorkerState(),
        process_map=process_map if process_map is not None else ProcessMap({}),
        horde_model_map=HordeModelMap(root={}),
        job_tracker=job_tracker if job_tracker is not None else JobTracker(),
        process_lifecycle=Mock(
            get_processes_with_model_for_queued_job=Mock(return_value=[]),
            is_model_load_quarantined=Mock(return_value=False),
            aux_download_deadline_for_dispatch=Mock(return_value=120.0),
        ),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        model_metadata=make_test_model_metadata(reference),
        max_concurrent_inference_processes=max_concurrent,
        max_inference_processes=2,
        lru=LRUCache(2),
    )


async def _make_overlap_scenario(
    *,
    running_baseline: KNOWN_IMAGE_GENERATION_BASELINE,
    candidate_baseline: KNOWN_IMAGE_GENERATION_BASELINE,
    running_steps_done: int,
    running_total_steps: int = 20,
    running_n_iter: int = 1,
    candidate_n_iter: int = 1,
    max_concurrent: int = 2,
) -> tuple[InferenceScheduler, ImageGenerateJobPopResponse]:
    """Build a scheduler with one in-flight job and one resident candidate head.

    The running job sits on process 0 with the given step progress and batch size; the candidate head
    sits resident on process 1, ready to accept work. ``max_concurrent`` is 2 so the count cap alone
    would always admit the candidate, isolating the progress/size overlap gate as the only thing that
    can hold it back.

    Returns:
        ``(scheduler, candidate_job)``.
    """
    running_model = f"running_{running_baseline.value}"
    candidate_model = f"candidate_{candidate_baseline.value}"

    running_proc = make_mock_process_info(
        0,
        model_name=running_model,
        state=HordeProcessState.INFERENCE_STARTING,
    )
    candidate_proc = make_mock_process_info(
        1,
        model_name=candidate_model,
        state=HordeProcessState.PRELOADED_MODEL,
    )
    process_map = ProcessMap({0: running_proc, 1: candidate_proc})
    job_tracker = JobTracker()

    reference: dict[str, object] = {
        running_model: make_mock_model_reference_record(running_model, baseline=running_baseline),
        candidate_model: make_mock_model_reference_record(candidate_model, baseline=candidate_baseline),
    }

    running_job = make_job_pop_response(
        running_model,
        ddim_steps=running_total_steps,
        n_iter=running_n_iter,
    )
    await track_popped_job_async(job_tracker, running_job)
    await mark_job_in_progress_async(job_tracker, running_job)

    # Mirror what the dispatcher stamps onto the slot so the overlap gate can find the running job and
    # read its live step progress.
    running_proc.last_job_referenced = running_job
    running_proc.loaded_horde_model_baseline = running_baseline
    running_proc.batch_amount = running_n_iter
    running_proc.last_total_steps = running_total_steps
    running_proc.last_current_step = running_steps_done
    running_proc.last_heartbeat_percent_complete = int(100 * running_steps_done / running_total_steps)

    candidate_job = make_job_pop_response(candidate_model, ddim_steps=20, n_iter=candidate_n_iter)
    await track_popped_job_async(job_tracker, candidate_job)

    scheduler = _make_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        reference=reference,
        max_concurrent=max_concurrent,
    )
    return scheduler, candidate_job


async def _overlap_admitted(scheduler: InferenceScheduler, candidate_job: ImageGenerateJobPopResponse) -> bool:
    """Whether the scheduler would dispatch the candidate head as a concurrent overlap right now."""
    result = await scheduler.get_next_job_and_process()
    if result is None:
        return False
    return result.next_job is candidate_job


class TestConcurrentOverlapHeadway:
    """The second concurrent inference must wait for the first to make size-appropriate headway."""

    async def test_two_sd15_thread_together_immediately(self) -> None:
        """Two SD1.5 jobs are cheap enough to sample together with no headway required (control)."""
        scheduler, candidate_job = await _make_overlap_scenario(
            running_baseline=_SD15,
            candidate_baseline=_SD15,
            running_steps_done=0,
        )
        assert await _overlap_admitted(scheduler, candidate_job) is True

    async def test_second_sdxl_blocked_until_first_sdxl_has_headway(self) -> None:
        """Two SDXL jobs must not overlap until the first is well underway (considerable headway).

        A second SDXL joining the card at the start of the first stacks two weight loads and activation
        peaks, which stalls the first sampler into a step-timeout teardown.
        """
        scheduler, candidate_job = await _make_overlap_scenario(
            running_baseline=_SDXL,
            candidate_baseline=_SDXL,
            running_steps_done=0,
        )
        assert await _overlap_admitted(scheduler, candidate_job) is False

    async def test_second_sdxl_admitted_once_first_sdxl_is_well_underway(self) -> None:
        """Once the running SDXL job is far enough along, a second SDXL may join."""
        scheduler, candidate_job = await _make_overlap_scenario(
            running_baseline=_SDXL,
            candidate_baseline=_SDXL,
            running_steps_done=18,
        )
        assert await _overlap_admitted(scheduler, candidate_job) is True

    async def test_sd15_candidate_waits_for_running_sdxl_to_progress(self) -> None:
        """An SDXL that started first gets some breathing room before a cheaper SD1.5 joins it."""
        scheduler, candidate_job = await _make_overlap_scenario(
            running_baseline=_SDXL,
            candidate_baseline=_SD15,
            running_steps_done=0,
        )
        assert await _overlap_admitted(scheduler, candidate_job) is False

    async def test_sd15_candidate_admitted_after_sdxl_has_modest_headway(self) -> None:
        """The cheaper SD1.5 candidate joins once the running SDXL has a modest amount of progress."""
        scheduler, candidate_job = await _make_overlap_scenario(
            running_baseline=_SDXL,
            candidate_baseline=_SD15,
            running_steps_done=18,
        )
        assert await _overlap_admitted(scheduler, candidate_job) is True

    async def test_extra_large_running_never_overlaps_even_near_completion(self) -> None:
        """An extra-large (Flux) job in flight never shares the card, regardless of its progress."""
        scheduler, candidate_job = await _make_overlap_scenario(
            running_baseline=_FLUX,
            candidate_baseline=_SD15,
            running_steps_done=19,
        )
        assert await _overlap_admitted(scheduler, candidate_job) is False

    async def test_extra_large_candidate_never_joins_a_busy_card(self) -> None:
        """An extra-large candidate never threads onto a card that is already sampling another job."""
        scheduler, candidate_job = await _make_overlap_scenario(
            running_baseline=_SD15,
            candidate_baseline=_FLUX,
            running_steps_done=19,
        )
        assert await _overlap_admitted(scheduler, candidate_job) is False

    async def test_batched_running_job_never_overlaps(self) -> None:
        """A batched job holds the card to itself: no second thread starts while it samples."""
        scheduler, candidate_job = await _make_overlap_scenario(
            running_baseline=_SD15,
            candidate_baseline=_SD15,
            running_steps_done=19,
            running_n_iter=2,
        )
        assert await _overlap_admitted(scheduler, candidate_job) is False

    async def test_batched_candidate_never_overlaps(self) -> None:
        """A batched candidate wants the card to itself and never joins another running job."""
        scheduler, candidate_job = await _make_overlap_scenario(
            running_baseline=_SD15,
            candidate_baseline=_SD15,
            running_steps_done=19,
            candidate_n_iter=2,
        )
        assert await _overlap_admitted(scheduler, candidate_job) is False

    async def test_both_heavy_admitted_exactly_at_threshold(self) -> None:
        """Two SDXL jobs may overlap the moment the running job reaches the required headway (15/20)."""
        scheduler, candidate_job = await _make_overlap_scenario(
            running_baseline=_SDXL,
            candidate_baseline=_SDXL,
            running_steps_done=15,
        )
        assert await _overlap_admitted(scheduler, candidate_job) is True

    async def test_both_heavy_blocked_just_below_threshold(self) -> None:
        """Two SDXL jobs stay apart while the running job is just short of the required headway (14/20)."""
        scheduler, candidate_job = await _make_overlap_scenario(
            running_baseline=_SDXL,
            candidate_baseline=_SDXL,
            running_steps_done=14,
        )
        assert await _overlap_admitted(scheduler, candidate_job) is False

    async def test_information_only_lookahead_is_not_gated(self) -> None:
        """The look-ahead view still surfaces the next job even when the overlap gate would hold it.

        ``run_scheduling_cycle`` uses the ``information_only`` call to make heavy-model/batch decisions;
        the hold is enforced on the real dispatch path, not by hiding the job from look-ahead.
        """
        scheduler, candidate_job = await _make_overlap_scenario(
            running_baseline=_SDXL,
            candidate_baseline=_SDXL,
            running_steps_done=0,
        )
        result = await scheduler.get_next_job_and_process(information_only=True)
        assert result is not None
        assert result.next_job is candidate_job

    async def test_start_inference_holds_the_gated_overlap(self) -> None:
        """The real dispatch entry point declines to start a gated overlap."""
        scheduler, _ = await _make_overlap_scenario(
            running_baseline=_SDXL,
            candidate_baseline=_SDXL,
            running_steps_done=0,
        )
        assert await scheduler.start_inference() is False

    async def test_start_inference_dispatches_once_headway_is_met(self) -> None:
        """The real dispatch entry point starts the overlap once the running job is far enough along."""
        scheduler, candidate_job = await _make_overlap_scenario(
            running_baseline=_SDXL,
            candidate_baseline=_SDXL,
            running_steps_done=18,
        )
        assert await scheduler.start_inference() is True
        assert candidate_job in scheduler._job_tracker.jobs_in_progress


def _attach_in_flight_job(
    *,
    process_map: ProcessMap,
    job_tracker: JobTracker,
    reference: dict[str, object],
    process_id: int,
    baseline: KNOWN_IMAGE_GENERATION_BASELINE,
    steps_done: int,
    total_steps: int = 20,
) -> ImageGenerateJobPopResponse:
    """Register a model in the reference and return a job to be marked in-flight on ``process_id``.

    The caller is responsible for awaiting ``track_popped_job_async``/``mark_job_in_progress_async``;
    this builds the process, stamps its progress, and wires the reference entry.
    """
    model = f"m{process_id}_{baseline.value}"
    proc = make_mock_process_info(process_id, model_name=model, state=HordeProcessState.INFERENCE_STARTING)
    proc.last_total_steps = total_steps
    proc.last_current_step = steps_done
    process_map[process_id] = proc
    reference[model] = make_mock_model_reference_record(model, baseline=baseline)
    job = make_job_pop_response(model, ddim_steps=total_steps)
    proc.last_job_referenced = job
    return job


class TestConcurrentOverlapAllowedUnit:
    """Direct coverage of the overlap-admission policy, including multi-job in-flight state."""

    def test_first_job_always_allowed(self) -> None:
        """With nothing in flight, any candidate (even extra-large) may start."""
        reference: dict[str, object] = {"big": make_mock_model_reference_record("big", baseline=_FLUX)}
        scheduler = _make_scheduler(reference=reference)
        candidate = make_job_pop_response("big")
        assert scheduler._concurrent_overlap_allowed(candidate) is True

    async def test_blocks_when_any_in_flight_job_lacks_headway(self) -> None:
        """The gate considers every in-flight job: one well-progressed sibling does not unblock a fresh one."""
        process_map = ProcessMap({})
        job_tracker = JobTracker()
        reference: dict[str, object] = {}

        done_job = _attach_in_flight_job(
            process_map=process_map,
            job_tracker=job_tracker,
            reference=reference,
            process_id=0,
            baseline=_SDXL,
            steps_done=20,
        )
        fresh_job = _attach_in_flight_job(
            process_map=process_map,
            job_tracker=job_tracker,
            reference=reference,
            process_id=1,
            baseline=_SDXL,
            steps_done=0,
        )
        for job in (done_job, fresh_job):
            await track_popped_job_async(job_tracker, job)
            await mark_job_in_progress_async(job_tracker, job)

        candidate_proc = make_mock_process_info(2, model_name="cand", state=HordeProcessState.PRELOADED_MODEL)
        process_map[2] = candidate_proc
        reference["cand"] = make_mock_model_reference_record("cand", baseline=_SD15)

        scheduler = _make_scheduler(process_map=process_map, job_tracker=job_tracker, reference=reference)
        candidate = make_job_pop_response("cand")
        # The freshly-started SDXL (0% progress) requires headway a cheaper joiner has not earned yet.
        assert scheduler._concurrent_overlap_allowed(candidate) is False


class TestModelSizeTier:
    """The size classification that scales the overlap headway."""

    def test_sd15_and_sd2_are_light(self) -> None:
        """SD1.5 and SD2 baselines classify as light."""
        scheduler = _make_scheduler(
            reference={
                "a": make_mock_model_reference_record("a", baseline=_SD15),
                "b": make_mock_model_reference_record("b", baseline=_SD2),
            },
        )
        assert scheduler._model_size_tier("a") is _ModelSizeTier.LIGHT
        assert scheduler._model_size_tier("b") is _ModelSizeTier.LIGHT

    def test_sdxl_is_heavy(self) -> None:
        """The SDXL baseline classifies as heavy."""
        scheduler = _make_scheduler(reference={"a": make_mock_model_reference_record("a", baseline=_SDXL)})
        assert scheduler._model_size_tier("a") is _ModelSizeTier.HEAVY

    def test_flux_cascade_qwen_are_extra_large(self) -> None:
        """Flux, Cascade, and Qwen baselines classify as extra-large."""
        scheduler = _make_scheduler(
            reference={
                "f": make_mock_model_reference_record("f", baseline=_FLUX),
                "c": make_mock_model_reference_record("c", baseline=_CASCADE),
                "q": make_mock_model_reference_record("q", baseline=_QWEN),
            },
        )
        assert scheduler._model_size_tier("f") is _ModelSizeTier.EXTRA_LARGE
        assert scheduler._model_size_tier("c") is _ModelSizeTier.EXTRA_LARGE
        assert scheduler._model_size_tier("q") is _ModelSizeTier.EXTRA_LARGE

    def test_named_vram_heavy_model_is_extra_large_without_a_baseline(self) -> None:
        """A model named in the VRAM-heavy list is extra-large even when its baseline is unknown."""
        scheduler = _make_scheduler(reference={})
        assert scheduler._model_size_tier(VRAM_HEAVY_MODELS[0]) is _ModelSizeTier.EXTRA_LARGE

    def test_unknown_or_missing_model_defaults_to_light(self) -> None:
        """Missing metadata stays permissive (light) rather than starving dispatch."""
        scheduler = _make_scheduler(reference={})
        assert scheduler._model_size_tier("never-heard-of-it") is _ModelSizeTier.LIGHT
        assert scheduler._model_size_tier(None) is _ModelSizeTier.LIGHT


class TestRequiredOverlapHeadway:
    """The headway fraction required for each running/candidate size pairing."""

    def test_two_light_jobs_need_no_headway(self) -> None:
        """Two light jobs may overlap immediately."""
        assert InferenceScheduler._required_overlap_headway(_ModelSizeTier.LIGHT, _ModelSizeTier.LIGHT) == 0.0

    def test_one_heavy_side_is_modest(self) -> None:
        """A pairing with exactly one heavy side needs modest, symmetric headway."""
        modest = InferenceScheduler._required_overlap_headway(_ModelSizeTier.HEAVY, _ModelSizeTier.LIGHT)
        assert modest == 0.5
        assert InferenceScheduler._required_overlap_headway(_ModelSizeTier.LIGHT, _ModelSizeTier.HEAVY) == modest

    def test_two_heavy_jobs_need_considerable_headway(self) -> None:
        """Two heavy jobs need more headway than a pairing with a single heavy side."""
        considerable = InferenceScheduler._required_overlap_headway(_ModelSizeTier.HEAVY, _ModelSizeTier.HEAVY)
        assert considerable == 0.75
        assert considerable > InferenceScheduler._required_overlap_headway(_ModelSizeTier.HEAVY, _ModelSizeTier.LIGHT)


class TestInFlightProgressFraction:
    """How the gate reads a running job's progress off its slot."""

    def _scheduler_with_running_job(
        self,
        *,
        last_total_steps: int | None,
        last_current_step: int | None,
        last_percent: int | None,
    ) -> tuple[InferenceScheduler, ImageGenerateJobPopResponse]:
        proc = make_mock_process_info(0, model_name="m", state=HordeProcessState.INFERENCE_STARTING)
        proc.last_total_steps = last_total_steps
        proc.last_current_step = last_current_step
        proc.last_heartbeat_percent_complete = last_percent
        job = make_job_pop_response("m")
        proc.last_job_referenced = job
        scheduler = _make_scheduler(process_map=ProcessMap({0: proc}))
        return scheduler, job

    def test_uses_step_progress_when_available(self) -> None:
        """Step counts are the primary progress signal."""
        scheduler, job = self._scheduler_with_running_job(
            last_total_steps=20,
            last_current_step=5,
            last_percent=99,
        )
        assert scheduler._in_flight_progress_fraction(job) == 0.25

    def test_falls_back_to_percent_when_steps_missing(self) -> None:
        """Percent-complete is used when step counts are not yet reported."""
        scheduler, job = self._scheduler_with_running_job(
            last_total_steps=None,
            last_current_step=None,
            last_percent=40,
        )
        assert scheduler._in_flight_progress_fraction(job) == 0.4

    def test_zero_when_nothing_reported_yet(self) -> None:
        """A freshly dispatched job that has reported neither steps nor percent reads as zero progress."""
        scheduler, job = self._scheduler_with_running_job(
            last_total_steps=None,
            last_current_step=None,
            last_percent=None,
        )
        assert scheduler._in_flight_progress_fraction(job) == 0.0

    def test_zero_when_no_process_owns_the_job(self) -> None:
        """A job not referenced by any slot reads as zero progress rather than erroring."""
        scheduler = _make_scheduler(process_map=ProcessMap({}))
        orphan_job = make_job_pop_response("m")
        assert scheduler._in_flight_progress_fraction(orphan_job) == 0.0
