"""Reproduction and fix for the sampler-hang storm that a model can drive across every slot.

Failure mode:
    One checkpoint deterministically wedged at its last sampling step. The stuck-step watchdog killed the
    slot each time, and the kills alternated between two slots, so the per-slot crash-loop breaker (keyed on
    the slot) never accumulated enough replacements to trip. The per-model quarantine could not help either:
    it only counted load failures a *child* reported, and a model that loads fine and then hangs the sampler
    reports nothing. With no counter watching the model, the worker kept popping its jobs, faulting them, and
    the horde force-set maintenance for "dropping too many jobs".

What is pinned here:
    - The stuck-step watchdog attributes the hang to the model that was sampling, and a second hang within
      the incident window quarantines it.
    - Quarantine faults the model's remaining queued jobs non-retryably (the horde reissues them elsewhere)
      through the one handler every feed site shares.
    - A quarantined model comes off the pop offer, which is what stops the horde assigning more of its jobs,
      with a floor so the exclusion can never leave the worker advertising nothing.
    - A native crash while a model is preloading (a child that dies before it can report the failure) counts
      as a load failure for that model, as does a live child reaped for outstaying its preload window.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, Mock

from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api import GENERATION_STATE

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_popper import JobPopper
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import (
    CRASH_LOOP_MAX_REPLACEMENTS,
    MODEL_INCIDENT_WINDOW_SECONDS,
    MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD,
    MODEL_SAMPLER_HANG_QUARANTINE_THRESHOLD,
    ModelIncidentKind,
    ProcessLifecycleManager,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_job,
    make_mock_process_info,
    make_test_api_sessions,
    make_test_runtime_config,
    make_testable_process_manager,
    track_popped_job_async,
)

_HANGING_MODEL = "WAI-NSFW-illustrious-SDXL"
_HEALTHY_MODEL = "stable_diffusion"
_STUCK_STEP_LIMIT = 20


def _hung_slot(process_id: int, model: str) -> HordeProcessInfo:
    """A sampling slot looping on one step: heart-beating (so silence cannot reap it) but not advancing."""
    slot = make_mock_process_info(process_id, model_name=model, state=HordeProcessState.INFERENCE_STARTING)
    now = time.time()
    slot.last_heartbeat_timestamp = now
    slot.last_received_timestamp = now
    slot.last_process_state_started_at = now
    slot.last_current_step = 24
    slot.last_total_steps = 25
    slot.nonadvancing_step_repeats = _STUCK_STEP_LIMIT + 1
    return slot


def _healthy_slot(process_id: int, model: str) -> HordeProcessInfo:
    """An idle slot that the watchdog has no reason to touch."""
    slot = make_mock_process_info(process_id, model_name=model, state=HordeProcessState.WAITING_FOR_JOB)
    now = time.time()
    slot.last_heartbeat_timestamp = now
    slot.last_received_timestamp = now
    slot.last_process_state_started_at = now
    return slot


def _make_storm_manager() -> HordeWorkerProcessManager:
    """A process manager whose lifecycle and popper are real, with the OS-touching calls stubbed out."""
    manager = make_testable_process_manager(
        image_models_to_load=[_HANGING_MODEL, _HEALTHY_MODEL],
        inference_stuck_step_repeat_limit=_STUCK_STEP_LIMIT,
    )
    lifecycle = manager._process_lifecycle
    lifecycle._end_inference_process = Mock()  # type: ignore[method-assign]
    lifecycle._request_inference_process_start = Mock()  # type: ignore[method-assign]
    return manager


def _run_watchdog(lifecycle: ProcessLifecycleManager) -> None:
    """Run one hung-process sweep, clearing the post-recovery debounce that spaces real sweeps apart."""
    lifecycle._recently_recovered = False
    lifecycle.replace_hung_processes()


class TestSamplerHangStorm:
    """Kills that round-robin across slots must still converge on the model that causes them."""

    async def test_alternating_slot_hangs_quarantine_the_model_and_free_its_queue(self) -> None:
        """Two hangs of one model on two different slots quarantine it and hand its backlog back.

        The storm's signature: no single slot is ever replaced twice, so only a model-keyed counter can see
        the pattern at all.
        """
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle
        job_tracker = manager._job_tracker

        first_slot = _hung_slot(1, _HANGING_MODEL)
        second_slot = _healthy_slot(2, _HANGING_MODEL)
        manager._process_map[1] = first_slot
        manager._process_map[2] = second_slot

        queued_for_hanging_model = await track_popped_job_async(
            job_tracker,
            make_job_pop_response(model=_HANGING_MODEL),
        )
        queued_for_healthy_model = await track_popped_job_async(
            job_tracker,
            make_job_pop_response(model=_HEALTHY_MODEL),
        )

        _run_watchdog(lifecycle)

        assert lifecycle.is_model_load_quarantined(_HANGING_MODEL) is False, "one hang is not yet a verdict"
        assert queued_for_hanging_model in job_tracker.jobs_pending_inference

        # The horde re-dispatches the model, so the next hang lands on the *other* slot.
        manager._process_map[2] = _hung_slot(2, _HANGING_MODEL)

        _run_watchdog(lifecycle)

        assert lifecycle.is_model_load_quarantined(_HANGING_MODEL) is True
        assert _HANGING_MODEL in lifecycle.quarantined_models()
        # Neither slot was replaced often enough to trip its own breaker: without the model-keyed counter
        # nothing at all would have fired.
        assert lifecycle._quarantined_inference_slots == set()

        assert queued_for_hanging_model not in job_tracker.jobs_pending_inference
        # Terminally faulted, not requeued: the horde reissues it to a worker that can run the model.
        assert job_tracker.get_stage(queued_for_hanging_model.id_) is JobStage.PENDING_SUBMIT
        pending_submit = job_tracker.jobs_pending_submit
        assert [info.state for info in pending_submit] == [GENERATION_STATE.faulted]
        # A different model's queued work is untouched: the quarantine is per model, not a worker-wide stop.
        assert queued_for_healthy_model in job_tracker.jobs_pending_inference
        assert lifecycle.is_model_load_quarantined(_HEALTHY_MODEL) is False

    async def test_a_single_hang_does_not_quarantine(self) -> None:
        """One hang can be a driver hiccup or a paging stall, so it is not enough to de-list the model."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle
        manager._process_map[1] = _hung_slot(1, _HANGING_MODEL)
        queued = await track_popped_job_async(manager._job_tracker, make_job_pop_response(model=_HANGING_MODEL))

        _run_watchdog(lifecycle)

        assert lifecycle.is_model_load_quarantined(_HANGING_MODEL) is False
        assert queued in manager._job_tracker.jobs_pending_inference

    def test_hangs_outside_the_window_do_not_accumulate(self) -> None:
        """An old hang ages out, so a slow trickle over hours never adds up to a quarantine."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle

        lifecycle.record_model_incident(_HANGING_MODEL, ModelIncidentKind.SAMPLER_HANG, job_id="job-old")
        aged = [
            incident._replace(at=incident.at - MODEL_INCIDENT_WINDOW_SECONDS - 1.0)
            for incident in lifecycle._model_incident_history[_HANGING_MODEL]
        ]
        lifecycle._model_incident_history[_HANGING_MODEL] = aged

        assert (
            lifecycle.record_model_incident(_HANGING_MODEL, ModelIncidentKind.SAMPLER_HANG, job_id="job-new") is False
        )
        assert lifecycle.is_model_load_quarantined(_HANGING_MODEL) is False

    def test_kinds_are_counted_against_their_own_thresholds(self) -> None:
        """A load failure and a hang are different evidence, so they do not pool into one count."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle

        lifecycle.record_model_incident(_HANGING_MODEL, ModelIncidentKind.LOAD_FAILURE)
        assert lifecycle.record_model_incident(_HANGING_MODEL, ModelIncidentKind.SAMPLER_HANG, job_id="a") is False
        assert lifecycle.is_model_load_quarantined(_HANGING_MODEL) is False
        # The hang threshold is the lower of the two, so it is the one the second hang crosses.
        assert MODEL_SAMPLER_HANG_QUARANTINE_THRESHOLD < MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD
        assert lifecycle.record_model_incident(_HANGING_MODEL, ModelIncidentKind.SAMPLER_HANG, job_id="b") is True

    async def test_hang_attribution_leaves_the_slot_breaker_alone(self) -> None:
        """Attributing a hang to the model must not change how the slot's own crash-loop breaker counts.

        The slot really was killed, so its replacement still counts; the model counter is additional
        evidence, never a substitute for the per-slot one.
        """
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle

        for _ in range(CRASH_LOOP_MAX_REPLACEMENTS + 1):
            manager._process_map[1] = _hung_slot(1, _HANGING_MODEL)
            _run_watchdog(lifecycle)

        assert len(lifecycle._slot_recovery_history[1]) == CRASH_LOOP_MAX_REPLACEMENTS + 1
        assert 1 in lifecycle._quarantined_inference_slots


class TestHangsAreCountedPerJob:
    """The hang threshold asks whether the model fails *jobs*, not how many kills one job cost.

    A job that hangs is requeued when its slot is replaced, so it can reach the watchdog again on the next
    slot. Counting those kills would let a single unlucky generation quarantine a model that runs
    everything else fine, which is the opposite of what the counter is for.
    """

    async def test_the_watchdog_records_the_job_the_slot_was_holding(self) -> None:
        """The reap carries the hung job's id into the incident, which is what the dedupe reads."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle
        job = await track_popped_job_async(manager._job_tracker, make_job_pop_response(model=_HANGING_MODEL))
        slot = _hung_slot(1, _HANGING_MODEL)
        slot.last_job_referenced = job
        manager._process_map[1] = slot

        _run_watchdog(lifecycle)

        assert [incident.job_id for incident in lifecycle._model_incident_history[_HANGING_MODEL]] == [str(job.id_)]

    def test_one_job_hanging_twice_is_one_piece_of_evidence(self) -> None:
        """The same job reaching the watchdog again after its requeue must not cross the threshold."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle

        for _ in range(MODEL_SAMPLER_HANG_QUARANTINE_THRESHOLD + 1):
            assert (
                lifecycle.record_model_incident(_HANGING_MODEL, ModelIncidentKind.SAMPLER_HANG, job_id="job-1")
                is False
            )

        assert lifecycle.is_model_load_quarantined(_HANGING_MODEL) is False

    def test_distinct_jobs_still_quarantine(self) -> None:
        """Different jobs hanging on the same model is the pattern the quarantine exists for."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle

        assert lifecycle.record_model_incident(_HANGING_MODEL, ModelIncidentKind.SAMPLER_HANG, job_id="job-1") is False
        assert lifecycle.record_model_incident(_HANGING_MODEL, ModelIncidentKind.SAMPLER_HANG, job_id="job-2") is True

    def test_a_repeat_and_a_new_job_count_as_two(self) -> None:
        """A mix of a repeat and a fresh job counts the jobs, so the fresh one is what crosses."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle

        assert lifecycle.record_model_incident(_HANGING_MODEL, ModelIncidentKind.SAMPLER_HANG, job_id="job-1") is False
        assert lifecycle.record_model_incident(_HANGING_MODEL, ModelIncidentKind.SAMPLER_HANG, job_id="job-1") is False
        assert lifecycle.record_model_incident(_HANGING_MODEL, ModelIncidentKind.SAMPLER_HANG, job_id="job-2") is True

    def test_an_unknown_job_id_stands_on_its_own(self) -> None:
        """An unrecorded id identifies nothing, so it can never be assumed to be a repeat of another."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle

        assert lifecycle.record_model_incident(_HANGING_MODEL, ModelIncidentKind.SAMPLER_HANG) is False
        assert lifecycle.record_model_incident(_HANGING_MODEL, ModelIncidentKind.SAMPLER_HANG) is True

    def test_load_failures_are_unaffected_by_the_job_dedupe(self) -> None:
        """A load failure happens before any job-specific work, so it is counted per occurrence."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle

        for _ in range(MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD - 1):
            assert (
                lifecycle.record_model_incident(_HANGING_MODEL, ModelIncidentKind.LOAD_FAILURE, job_id="job-1")
                is False
            )

        assert lifecycle.record_model_incident(_HANGING_MODEL, ModelIncidentKind.LOAD_FAILURE, job_id="job-1") is True


class TestPreloadCrashAttribution:
    """A child that dies natively mid-preload never reports the failure, so the parent must attribute it."""

    @staticmethod
    def _dead_preloading_slot(process_id: int, model: str) -> HordeProcessInfo:
        """A slot whose OS process exited while it was preloading ``model``."""
        slot = make_mock_process_info(process_id, model_name=model, state=HordeProcessState.PRELOADING_MODEL)
        slot.mp_process = Mock(is_alive=Mock(return_value=False), exitcode=-1073741819, pid=slot.os_pid)
        return slot

    def test_native_preload_crash_counts_as_a_load_failure(self) -> None:
        """Repeated native deaths during one model's preload quarantine it, as reported failures do."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle

        for attempt in range(MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD):
            slot = self._dead_preloading_slot(1 + attempt, _HANGING_MODEL)
            manager._process_map[slot.process_id] = slot
            lifecycle._replace_inference_process(slot)

        assert lifecycle.is_model_load_quarantined(_HANGING_MODEL) is True

    def test_a_crash_in_another_state_is_not_attributed_to_the_model(self) -> None:
        """A slot that dies while sampling or idle says nothing about the model's loadability."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle

        for attempt in range(MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD):
            slot = self._dead_preloading_slot(1 + attempt, _HANGING_MODEL)
            slot.last_process_state = HordeProcessState.INFERENCE_STARTING
            manager._process_map[slot.process_id] = slot
            lifecycle._replace_inference_process(slot)

        assert lifecycle.is_model_load_quarantined(_HANGING_MODEL) is False

    def test_a_reported_load_failure_is_not_double_counted_by_the_crash_path(self) -> None:
        """One failure the child reported and the parent then reaps must count once, not twice."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle

        for attempt in range(MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD - 1):
            slot = self._dead_preloading_slot(1 + attempt, _HANGING_MODEL)
            manager._process_map[slot.process_id] = slot
            manager._on_model_load_failure(slot.process_id, _HANGING_MODEL)
            lifecycle._replace_inference_process(slot)

        # Two reported-and-reaped failures: double counting would already have crossed the threshold of 3.
        assert lifecycle.is_model_load_quarantined(_HANGING_MODEL) is False
        assert len(lifecycle._model_incident_history[_HANGING_MODEL]) == MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD - 1


class TestStuckPreloadAttribution:
    """A slot still alive but wedged loading a model is the same evidence against it as one that died."""

    @staticmethod
    def _stuck_preloading_slot(process_id: int, model: str) -> HordeProcessInfo:
        """A live slot silent past ``preload_timeout`` while still in ``PRELOADING_MODEL``."""
        slot = make_mock_process_info(process_id, model_name=model, state=HordeProcessState.PRELOADING_MODEL)
        stale = time.time() - 10_000
        slot.last_heartbeat_timestamp = stale
        slot.last_received_timestamp = stale
        slot.last_process_state_started_at = stale
        return slot

    def test_the_preload_watchdog_attributes_the_reap_to_the_model(self) -> None:
        """Reaping one wedged preload records a load-failure incident for the model being loaded."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle

        slot = self._stuck_preloading_slot(1, _HANGING_MODEL)
        manager._process_map[slot.process_id] = slot
        _run_watchdog(lifecycle)

        assert slot.process_id not in manager._process_map, "the wedged slot was reaped"
        assert len(lifecycle._model_incident_history[_HANGING_MODEL]) == 1

    def test_repeated_stuck_preloads_quarantine_the_model(self) -> None:
        """A model that wedges the loader on slot after slot crosses the load-failure threshold."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle

        for attempt in range(MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD):
            slot = self._stuck_preloading_slot(1 + attempt, _HANGING_MODEL)
            manager._process_map[slot.process_id] = slot
            _run_watchdog(lifecycle)

        assert lifecycle.is_model_load_quarantined(_HANGING_MODEL) is True

    def test_a_stuck_preload_says_nothing_about_another_model(self) -> None:
        """Incidents stay keyed to the model that was loading, so a healthy model is never dragged in."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle

        for attempt in range(MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD):
            slot = self._stuck_preloading_slot(1 + attempt, _HANGING_MODEL)
            manager._process_map[slot.process_id] = slot
            _run_watchdog(lifecycle)

        assert lifecycle.is_model_load_quarantined(_HEALTHY_MODEL) is False
        assert _HEALTHY_MODEL not in lifecycle._model_incident_history

    def test_a_bulk_replacement_mid_preload_is_not_attributed_to_the_model(self) -> None:
        """Only the preload-window watchdog charges a live slot's model; a bulk cycle carries no evidence."""
        manager = _make_storm_manager()
        lifecycle = manager._process_lifecycle

        for attempt in range(MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD):
            slot = self._stuck_preloading_slot(1 + attempt, _HANGING_MODEL)
            manager._process_map[slot.process_id] = slot
            lifecycle._replace_inference_process(slot)

        assert lifecycle.is_model_load_quarantined(_HANGING_MODEL) is False
        assert _HANGING_MODEL not in lifecycle._model_incident_history


def _make_popper(
    *,
    quarantined: frozenset[str],
    image_models_to_load: list[str],
    job_tracker: JobTracker,
    horde_client_session: object,
) -> JobPopper:
    """A JobPopper wired to a fixed quarantine set, otherwise mocked like the popper's own tests."""
    # No model is resident on the free slot, so the residency-bias narrowing (a separate lane) stays out of
    # the way and the offer under test is the configured set minus the quarantine exclusion.
    process_map = ProcessMap(
        {
            0: make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB),
            10: make_mock_process_info(
                10,
                model_name=None,
                state=HordeProcessState.WAITING_FOR_JOB,
                process_type=HordeProcessType.SAFETY,
            ),
        },
    )
    return JobPopper(
        state=WorkerState(),
        process_map=process_map,
        job_tracker=job_tracker,
        shutdown_manager=Mock(),
        runtime_config=make_test_runtime_config(
            bridge_data=make_mock_bridge_data(image_models_to_load=image_models_to_load),
        ),
        api_sessions=make_test_api_sessions(
            horde_client_session=horde_client_session,
            aiohttp_session=Mock(),
        ),
        max_inference_processes=2,
        max_concurrent_inference_processes=1,
        quarantined_models_provider=lambda: quarantined,
    )


async def _offered_models(*, quarantined: frozenset[str], image_models_to_load: list[str]) -> set[str]:
    """Drive one real pop and return the model set it advertised to the horde."""
    job_tracker = JobTracker()
    await track_popped_job_async(job_tracker, make_mock_job(model="warm_up"))
    await job_tracker.increment_jobs_completed()  # clear the session warm-up gate so a pop happens
    session = Mock()
    session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
    popper = _make_popper(
        quarantined=quarantined,
        image_models_to_load=image_models_to_load,
        job_tracker=job_tracker,
        horde_client_session=session,
    )

    await popper.api_job_pop()

    session.submit_request.assert_awaited_once()
    return set(session.submit_request.call_args.args[0].models)


class TestQuarantinedModelPopOffer:
    """Quarantine has to reach the offer: an advertised model keeps being assigned, and keeps faulting."""

    async def test_quarantined_model_is_not_advertised(self) -> None:
        """The model comes off the offer while its healthy sibling stays on it."""
        offered = await _offered_models(
            quarantined=frozenset({_HANGING_MODEL}),
            image_models_to_load=[_HANGING_MODEL, _HEALTHY_MODEL],
        )

        assert _HANGING_MODEL not in offered
        assert _HEALTHY_MODEL in offered

    async def test_exclusion_never_empties_the_offer(self) -> None:
        """A worker whose only model is quarantined keeps offering it rather than going silent.

        An empty offer is sent nothing, so it can never produce the activity that would clear the
        quarantine; taking the faults is the recoverable failure, advertising nothing is not.
        """
        offered = await _offered_models(
            quarantined=frozenset({_HANGING_MODEL}),
            image_models_to_load=[_HANGING_MODEL],
        )

        assert offered == {_HANGING_MODEL}

    async def test_nothing_quarantined_leaves_the_offer_untouched(self) -> None:
        """The exclusion is inert for a worker that has never quarantined anything."""
        offered = await _offered_models(
            quarantined=frozenset(),
            image_models_to_load=[_HANGING_MODEL, _HEALTHY_MODEL],
        )

        assert offered == {_HANGING_MODEL, _HEALTHY_MODEL}
