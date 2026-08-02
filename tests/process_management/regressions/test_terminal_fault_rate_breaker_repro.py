"""Reproduction and fix for a terminal-fault stream the worker never reacted to.

Failure mode:
    A poison model double-faulted five jobs over roughly half an hour. At the fourth and fifth faulted
    submission the horde force-set the worker into maintenance for "dropping too many jobs". The worker had
    no reading of its own terminal-fault rate at all: the session faulted counter only moves at submit time,
    the per-model quarantine only sees faults attributable to one checkpoint, and the resource/OOM
    self-throttle only sees faults a card could not serve. Nothing watched the plain rate, so the server's
    breaker fired before any local one could.

What is pinned here:
    - A burst of terminal generation faults inside the window pauses new job pops, and the pause is armed
      from the tracker's fault decision rather than from the submit-time counter.
    - Every pause lifts on its deadline alone, including while the faults continue.
    - A re-trip inside the decay window buys a longer cooldown, up to a cap; a quiet period resets it.
    - Only genuine generation faults count: retryable failures that requeue, and faults a scheduling
      recovery issued, leave the breaker alone.
    - The non-retryable faults a model quarantine's backlog sweep emits do count, because the horde counts
      them as drops too.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, Mock

import pytest
from horde_sdk import RequestErrorResponse

from horde_worker_regen.process_management.config.worker_state import PopPauseOwner
from horde_worker_regen.process_management.ipc.action_ledger import LedgerEventType
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_popper import JobPopper
from horde_worker_regen.process_management.jobs.job_tracker import (
    InferenceFailureResolution,
    JobFaultOrigin,
    JobTracker,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.process_manager import (
    TERMINAL_FAULT_BREAKER_COOLDOWN_SECONDS,
    TERMINAL_FAULT_BREAKER_ESCALATION_DECAY_SECONDS,
    TERMINAL_FAULT_BREAKER_MAX_COOLDOWN_SECONDS,
    TERMINAL_FAULT_BREAKER_THRESHOLD,
    TERMINAL_FAULT_BREAKER_WINDOW_SECONDS,
    HordeWorkerProcessManager,
)
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

_FAULTING_MODEL = "WAI-NSFW-illustrious-SDXL"


def _make_manager() -> HordeWorkerProcessManager:
    """A process manager whose fault-rate breaker is wired exactly as the live worker's is."""
    return make_testable_process_manager(image_models_to_load=[_FAULTING_MODEL])


async def _fault_terminally(
    manager: HordeWorkerProcessManager,
    count: int,
    *,
    model: str = _FAULTING_MODEL,
    fault_origin: JobFaultOrigin = JobFaultOrigin.GENERATION,
) -> None:
    """Drive ``count`` jobs through the tracker to a terminal fault of ``fault_origin``."""
    for _ in range(count):
        job = await track_popped_job_async(manager._job_tracker, make_job_pop_response(model=model))
        manager._job_tracker.handle_job_fault_now(job, retryable=False, fault_origin=fault_origin)


def _age_fault_history(manager: HordeWorkerProcessManager, seconds: float) -> None:
    """Push every recorded terminal fault ``seconds`` further into the past."""
    manager._terminal_fault_history = [(when - seconds, model) for when, model in manager._terminal_fault_history]


def _lapse_pause(manager: HordeWorkerProcessManager) -> None:
    """Bring the standing pause's deadline forward and run the tick that lapses it."""
    manager._state.self_throttle_paused_until = time.time() - 1.0
    manager._apply_self_maintenance_throttle()
    assert manager._state.self_throttle_paused is False


def _standing_cooldown(manager: HordeWorkerProcessManager) -> float:
    """How long the standing pause still has to run, from now."""
    return manager._state.self_throttle_paused_until - time.time()


async def _attempt_pop(manager: HordeWorkerProcessManager) -> Mock:
    """Drive one real pop over the manager's own worker state; return the API session it used.

    The popper carries a tracker of its own so the only thing under test is the shared pop gate: a queue
    full of the jobs the breaker was fed would otherwise suppress the pop for an unrelated reason.
    """
    job_tracker = JobTracker()
    await track_popped_job_async(job_tracker, make_mock_job(model="warm_up"))
    await job_tracker.increment_jobs_completed()  # clear the session warm-up gate so a pop can happen
    session = Mock()
    session.submit_request = AsyncMock(return_value=RequestErrorResponse(message="no jobs"))
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
    popper = JobPopper(
        state=manager._state,
        process_map=process_map,
        job_tracker=job_tracker,
        shutdown_manager=Mock(),
        runtime_config=make_test_runtime_config(
            bridge_data=make_mock_bridge_data(image_models_to_load=[_FAULTING_MODEL]),
        ),
        api_sessions=make_test_api_sessions(horde_client_session=session, aiohttp_session=Mock()),
        max_inference_processes=2,
        max_concurrent_inference_processes=1,
    )

    await popper.api_job_pop()

    return session


class TestTerminalFaultRateBreaker:
    """The worker has to see its own drop rate before the horde does."""

    async def test_a_burst_of_terminal_faults_stops_new_pops(self) -> None:
        """Reaching the threshold inside the window arms a fault-owned pause and pops stop being built."""
        manager = _make_manager()

        assert (await _attempt_pop(manager)).submit_request.await_count == 1, "a healthy worker pops"

        await _fault_terminally(manager, TERMINAL_FAULT_BREAKER_THRESHOLD)
        manager._apply_self_maintenance_throttle()

        assert manager._state.self_throttle_paused is True
        assert manager._state.self_throttle_pause_owner is PopPauseOwner.FAULT_THROTTLE
        assert manager._state.workload_intake_paused is True
        assert _standing_cooldown(manager) == pytest.approx(TERMINAL_FAULT_BREAKER_COOLDOWN_SECONDS, abs=2.0)

        (await _attempt_pop(manager)).submit_request.assert_not_awaited()

    async def test_the_trip_is_recorded_on_the_action_ledger(self) -> None:
        """The arm carries the count, the threshold, the duration, and the models that produced it."""
        manager = _make_manager()
        await _fault_terminally(manager, TERMINAL_FAULT_BREAKER_THRESHOLD)

        manager._apply_self_maintenance_throttle()

        armed = [
            event
            for event in manager._action_ledger.recent(limit=100)
            if event.event_type == LedgerEventType.POP_PAUSE_ARMED
        ]
        assert len(armed) == 1
        detail = armed[0].detail
        assert detail["owner"] == PopPauseOwner.FAULT_THROTTLE.value
        assert detail["terminal_faults"] == TERMINAL_FAULT_BREAKER_THRESHOLD
        assert detail["threshold"] == TERMINAL_FAULT_BREAKER_THRESHOLD
        assert detail["duration_seconds"] == TERMINAL_FAULT_BREAKER_COOLDOWN_SECONDS
        assert detail["models"] == _FAULTING_MODEL

    async def test_pops_resume_once_the_cooldown_elapses(self) -> None:
        """The pause lifts on its deadline and the worker builds pop requests again."""
        manager = _make_manager()
        await _fault_terminally(manager, TERMINAL_FAULT_BREAKER_THRESHOLD)
        manager._apply_self_maintenance_throttle()
        assert manager._state.self_throttle_paused is True

        _lapse_pause(manager)

        assert manager._state.self_throttle_pause_owner is None
        assert manager._state.workload_intake_paused is False
        (await _attempt_pop(manager)).submit_request.assert_awaited_once()

    async def test_the_pause_lifts_even_while_faults_keep_arriving(self) -> None:
        """Nothing the failure itself does can extend a standing pause: only the deadline lifts it.

        The worker re-trips afterwards if the condition persists, so the outage is a bounded, repeating
        pause an operator can read, never an open-ended silence.
        """
        manager = _make_manager()
        await _fault_terminally(manager, TERMINAL_FAULT_BREAKER_THRESHOLD)
        manager._apply_self_maintenance_throttle()

        await _fault_terminally(manager, TERMINAL_FAULT_BREAKER_THRESHOLD * 2)
        _lapse_pause(manager)

        manager._apply_self_maintenance_throttle()
        assert manager._state.self_throttle_paused is True
        assert _standing_cooldown(manager) <= TERMINAL_FAULT_BREAKER_MAX_COOLDOWN_SECONDS

    async def test_the_spent_evidence_does_not_immediately_re_trip(self) -> None:
        """The faults a trip answered are consumed by it; a fresh burst is needed for the next trip."""
        manager = _make_manager()
        await _fault_terminally(manager, TERMINAL_FAULT_BREAKER_THRESHOLD)
        manager._apply_self_maintenance_throttle()
        assert manager._state.self_throttle_paused is True

        _lapse_pause(manager)
        manager._apply_self_maintenance_throttle()

        assert manager._state.self_throttle_paused is False


class TestBreakerControls:
    """Conditions that must never pause a worker's intake."""

    async def test_a_healthy_worker_never_pauses(self) -> None:
        """A worker with no terminal faults at all is left alone by the breaker."""
        manager = _make_manager()

        manager._apply_self_maintenance_throttle()

        assert manager._state.self_throttle_paused is False
        (await _attempt_pop(manager)).submit_request.assert_awaited_once()

    async def test_faults_below_the_threshold_do_not_trip(self) -> None:
        """A worker dropping the occasional job keeps working."""
        manager = _make_manager()

        await _fault_terminally(manager, TERMINAL_FAULT_BREAKER_THRESHOLD - 1)
        manager._apply_self_maintenance_throttle()

        assert manager._state.self_throttle_paused is False

    async def test_faults_spread_wider_than_the_window_do_not_trip(self) -> None:
        """A slow trickle over hours ages out instead of accumulating into a pause."""
        manager = _make_manager()

        await _fault_terminally(manager, TERMINAL_FAULT_BREAKER_THRESHOLD)
        _age_fault_history(manager, TERMINAL_FAULT_BREAKER_WINDOW_SECONDS + 1.0)
        manager._apply_self_maintenance_throttle()

        assert manager._state.self_throttle_paused is False

    async def test_retryable_failures_do_not_count(self) -> None:
        """A failure the worker requeues dropped nothing, so the horde counts nothing against it."""
        manager = _make_manager()
        manager._job_tracker.set_retry_policy(2)

        for _ in range(TERMINAL_FAULT_BREAKER_THRESHOLD):
            job = await track_popped_job_async(manager._job_tracker, make_job_pop_response(model=_FAULTING_MODEL))
            resolution = manager._job_tracker.handle_job_fault_now(job, retryable=True)
            assert resolution is InferenceFailureResolution.RETRY

        assert manager._terminal_fault_history == []
        manager._apply_self_maintenance_throttle()
        assert manager._state.self_throttle_paused is False

    async def test_scheduling_recovery_faults_do_not_count(self) -> None:
        """A recovery action faulting a wedged backlog is not a verdict on generating, so it cannot pause."""
        manager = _make_manager()

        await _fault_terminally(
            manager,
            TERMINAL_FAULT_BREAKER_THRESHOLD,
            fault_origin=JobFaultOrigin.SCHEDULING_RECOVERY,
        )
        manager._apply_self_maintenance_throttle()

        assert manager._terminal_fault_history == []
        assert manager._state.self_throttle_paused is False

    async def test_the_shutdown_drain_does_not_count(self) -> None:
        """A worker on its way out faults its backlog deliberately; pausing its intake means nothing."""
        manager = _make_manager()
        manager._state.shutting_down = True

        await _fault_terminally(manager, TERMINAL_FAULT_BREAKER_THRESHOLD)

        assert manager._terminal_fault_history == []


class TestBreakerEscalation:
    """A condition one pause did not resolve earns a longer one, bounded and self-resetting."""

    async def test_a_re_trip_inside_the_decay_window_doubles_the_cooldown(self) -> None:
        """The second trip pauses for twice as long, because the first cooldown did not settle anything."""
        manager = _make_manager()
        await _fault_terminally(manager, TERMINAL_FAULT_BREAKER_THRESHOLD)
        manager._apply_self_maintenance_throttle()
        _lapse_pause(manager)

        await _fault_terminally(manager, TERMINAL_FAULT_BREAKER_THRESHOLD)
        manager._apply_self_maintenance_throttle()

        assert manager._state.self_throttle_paused is True
        assert _standing_cooldown(manager) == pytest.approx(2 * TERMINAL_FAULT_BREAKER_COOLDOWN_SECONDS, abs=2.0)

    async def test_a_quiet_period_resets_the_escalation(self) -> None:
        """A worker that goes the decay window without tripping starts its next trip from the base."""
        manager = _make_manager()
        await _fault_terminally(manager, TERMINAL_FAULT_BREAKER_THRESHOLD)
        manager._apply_self_maintenance_throttle()
        _lapse_pause(manager)

        manager._terminal_fault_breaker_last_trip_at -= TERMINAL_FAULT_BREAKER_ESCALATION_DECAY_SECONDS + 1.0
        await _fault_terminally(manager, TERMINAL_FAULT_BREAKER_THRESHOLD)
        manager._apply_self_maintenance_throttle()

        assert manager._terminal_fault_breaker_escalation == 0
        assert _standing_cooldown(manager) == pytest.approx(TERMINAL_FAULT_BREAKER_COOLDOWN_SECONDS, abs=2.0)

    async def test_the_escalated_cooldown_is_capped(self) -> None:
        """However long a condition persists, the pause it buys stays bounded."""
        manager = _make_manager()
        manager._terminal_fault_breaker_last_trip_at = time.time()
        manager._terminal_fault_breaker_escalation = 12

        await _fault_terminally(manager, TERMINAL_FAULT_BREAKER_THRESHOLD)
        manager._apply_self_maintenance_throttle()

        assert _standing_cooldown(manager) == pytest.approx(TERMINAL_FAULT_BREAKER_MAX_COOLDOWN_SECONDS, abs=2.0)


class TestQuarantineSweepInteraction:
    """The sweep's own faults are real drops, so they belong on the breaker's count."""

    async def test_a_quarantine_backlog_sweep_trips_the_breaker(self) -> None:
        """Handing a deep backlog back to the horde pauses intake while it drains, which is the intent."""
        manager = _make_manager()
        for _ in range(TERMINAL_FAULT_BREAKER_THRESHOLD):
            await track_popped_job_async(manager._job_tracker, make_job_pop_response(model=_FAULTING_MODEL))

        manager._on_model_quarantined(_FAULTING_MODEL)
        manager._apply_self_maintenance_throttle()

        assert len(manager._terminal_fault_history) == 0, "the trip consumed the sweep's faults"
        assert manager._state.self_throttle_paused is True
        assert manager._state.self_throttle_pause_owner is PopPauseOwner.FAULT_THROTTLE
        (await _attempt_pop(manager)).submit_request.assert_not_awaited()
