"""The worker discloses a pop loop that has stopped reaching the horde.

Silence on the intake path is the one failure the worker cannot distinguish from a quiet horde by looking
at its own logs: the pop coroutine returns early at any of a dozen gates without saying so, so a wedged
pool and an empty queue look identical. The sentinel turns that silence into a line naming the gate, and
stays quiet whenever the silence is explained (the worker is deliberately not taking work, or a tracked
governor is holding pops on purpose).
"""

from __future__ import annotations

import asyncio
import time

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from loguru import logger

from horde_worker_regen.process_management.config.worker_state import PopGate
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    SupervisorCommand,
    SupervisorControlMessage,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessState, HordeProcessType
from horde_worker_regen.process_management.process_manager import (
    POP_LIVENESS_ERROR_SECONDS,
    POP_LIVENESS_FROZEN_QUEUE_SECONDS,
    POP_LIVENESS_NON_EXPLAINING_GOVERNORS,
    POP_LIVENESS_WARN_SECONDS,
    HordeWorkerProcessManager,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)


@pytest.fixture(autouse=True)
def _strictly_increasing_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``time.time`` injective for this module's tests.

    The pop path stamps ``last_pop_attempt_completed_at`` with ``time.time()``, and these tests prove a
    completed attempt was recorded by asserting the stamp strictly advanced. The OS clock's resolution can
    be coarser than a whole stubbed pop cycle, in which case two legitimate recordings carry the same stamp
    and the strict proof fails on timing alone. Nudging any repeated reading forward by a microsecond keeps
    readings real while making strict advancement a sound proxy for "the recording happened".
    """
    real_time = time.time
    last_reading = [0.0]

    def _injective_time() -> float:
        now = real_time()
        if now <= last_reading[0]:
            now = last_reading[0] + 1e-6
        last_reading[0] = now
        return now

    monkeypatch.setattr(time, "time", _injective_time)


_SAFETY_SLOT = 0
"""The process-map slot holding the safety lane the pop ladder requires before it will offer work."""

_INFERENCE_SLOT = 1
"""The process-map slot holding the inference lane the pop ladder requires before it will offer work."""


def _capture(level: str) -> tuple[list[str], int]:
    """Attach a loguru sink collecting messages at exactly ``level``."""
    lines: list[str] = []
    sink_id = logger.add(
        lambda message: lines.append(message.record["message"]),
        level=level,
        filter=lambda record: record["level"].name == level,
    )
    return lines, sink_id


class _StubHordeSession:
    """Stand in for the horde client session at the network boundary, and nowhere else.

    Everything the pop cycle does either side of the request (the gate ladder, the offer composition, the
    attempt-completion stamp, the throttler's success/error accounting) runs as it does in production.
    """

    def __init__(self, outcome: ImageGenerateJobPopResponse | Exception) -> None:
        """Answer every request with ``outcome``, returning it or raising it."""
        self._outcome = outcome
        self.request_count = 0

    async def submit_request(self, request: object, expected_response_type: object) -> ImageGenerateJobPopResponse:
        """Record that the request went out, then deliver the configured outcome."""
        self.request_count += 1
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class _HangingHordeSession:
    """A horde client session whose request never comes back, modelling a pop that hangs on the wire."""

    def __init__(self) -> None:
        """Start with the request unreached and the caller blocked once it arrives."""
        self.reached = asyncio.Event()
        self._released = asyncio.Event()
        self.request_count = 0

    async def submit_request(self, request: object, expected_response_type: object) -> ImageGenerateJobPopResponse:
        """Block indefinitely, so the pop cycle sits past the point where it cleared its gate."""
        self.request_count += 1
        self.reached.set()
        await self._released.wait()
        raise AssertionError("the hanging pop request was never meant to return")


def _no_job_answer() -> ImageGenerateJobPopResponse:
    """Build the horde's "nothing for you" answer, which is a completed attempt like any other."""
    return ImageGenerateJobPopResponse(id=None, ids=[], payload={}, skipped={})


def _pooled_manager(**bridge_overrides: object) -> HordeWorkerProcessManager:
    """Build a manager whose pool satisfies the pop ladder's process preconditions.

    Without both lanes present the ladder stops at ``no_safety_process`` or ``no_inference_process`` and no
    test could ever drive a pop attempt to the request.
    """
    manager = make_testable_process_manager(bridge_data=make_mock_bridge_data(**bridge_overrides))
    manager._process_map[_SAFETY_SLOT] = make_mock_process_info(
        _SAFETY_SLOT,
        process_type=HordeProcessType.SAFETY,
        model_name=None,
    )
    manager._process_map[_INFERENCE_SLOT] = make_mock_process_info(_INFERENCE_SLOT, model_name=None)
    return manager


async def _complete_a_pop_attempt(
    manager: HordeWorkerProcessManager,
    *,
    outcome: ImageGenerateJobPopResponse | Exception | None = None,
) -> float:
    """Drive one whole pop cycle to the horde and return the completion stamp the popper recorded.

    Returns:
        The wall-clock stamp the pop path wrote to ``last_pop_attempt_completed_at``.
    """
    session = _StubHordeSession(_no_job_answer() if outcome is None else outcome)
    manager._api_sessions.set_horde_client_session(session)  # pyrefly: ignore - a stub stands in for the session
    before = manager._state.last_pop_attempt_completed_at
    await manager._job_popper.api_job_pop(urgent=True)

    assert session.request_count == 1, "the pop cycle never reached the request"
    completed_at = manager._state.last_pop_attempt_completed_at
    assert completed_at > before, "the pop path did not record the completed attempt"
    return completed_at


async def _hold_pops_at_no_inference_process(manager: HordeWorkerProcessManager) -> None:
    """Take the inference lane out of service and drive a pop cycle onto that gate.

    The gate name, the hold stamp, and the absence of a further attempt all come from the pop path itself,
    so a pop loop that stopped maintaining them would leave the caller's preconditions unmet.
    """
    manager._process_map[_INFERENCE_SLOT].last_process_state = HordeProcessState.INFERENCE_STARTING
    assert manager._process_map.get_first_available_inference_process() is None

    attempt_before = manager._state.last_pop_attempt_completed_at
    await manager._job_popper.api_job_pop(urgent=True)

    assert manager._state.last_pop_gate == PopGate.NO_INFERENCE_PROCESS
    assert manager._state.last_pop_attempt_completed_at == attempt_before, "a held cycle must not count as an attempt"


async def _silenced_at_the_pool_gate(**bridge_overrides: object) -> tuple[HordeWorkerProcessManager, float]:
    """Build a manager that reached the horde once and has been held at the pool gate ever since.

    Returns:
        The manager and the stamp of its one completed attempt, from which its silence is measured.
    """
    manager = _pooled_manager(**bridge_overrides)
    completed_at = await _complete_a_pop_attempt(manager)
    await _hold_pops_at_no_inference_process(manager)
    return manager, completed_at


class TestPopLivenessSentinel:
    """The sentinel fires once per silence episode, and only when the silence is unexplained.

    Every scenario reaches its preconditions through the pop path itself: the attempt clock is the one the
    pop cycle stamps when it hears back from the horde, and the gate is the one its ladder chose. A pop loop
    that stopped maintaining either would fail these tests at their setup rather than passing them.
    """

    async def test_silence_past_the_warn_threshold_names_the_gate(self) -> None:
        """The disclosure has to carry the gate: the gate is what an operator can act on."""
        manager, completed_at = await _silenced_at_the_pool_gate()

        lines, sink_id = _capture("WARNING")
        try:
            manager._check_pop_liveness(completed_at + POP_LIVENESS_WARN_SECONDS + 3.0)
        finally:
            logger.remove(sink_id)

        assert len(lines) == 1, lines
        assert str(PopGate.NO_INFERENCE_PROCESS) in lines[0]

    async def test_the_warning_is_edge_triggered(self) -> None:
        """The control loop ticks many times a second, so a level-triggered line would drown the log."""
        manager, completed_at = await _silenced_at_the_pool_gate()
        now = completed_at + POP_LIVENESS_WARN_SECONDS + 3.0

        lines, sink_id = _capture("WARNING")
        try:
            for tick in range(5):
                manager._check_pop_liveness(now + tick)
        finally:
            logger.remove(sink_id)

        assert len(lines) == 1, lines

    async def test_prolonged_silence_escalates_to_an_error(self) -> None:
        """Minutes without reaching the horde is a wedge, not a slow patch."""
        manager, completed_at = await _silenced_at_the_pool_gate()
        now = completed_at + POP_LIVENESS_ERROR_SECONDS + 5.0

        lines, sink_id = _capture("ERROR")
        try:
            manager._check_pop_liveness(now)
            manager._check_pop_liveness(now + 1.0)
        finally:
            logger.remove(sink_id)

        assert len(lines) == 1, lines
        assert str(PopGate.NO_INFERENCE_PROCESS) in lines[0]

    async def test_the_error_repeat_is_time_boxed_not_per_tick(self) -> None:
        """The control loop ticks constantly, so a persisting wedge restates itself on a clock, not a tick."""
        manager, completed_at = await _silenced_at_the_pool_gate()
        now = completed_at + POP_LIVENESS_ERROR_SECONDS + 5.0

        lines, sink_id = _capture("ERROR")
        try:
            manager._check_pop_liveness(now)
            for tick in range(1, 60):
                manager._check_pop_liveness(now + tick)
            assert len(lines) == 1, lines
            manager._check_pop_liveness(now + POP_LIVENESS_ERROR_SECONDS)
        finally:
            logger.remove(sink_id)

        assert len(lines) == 2, lines

    async def test_an_active_pop_governor_explains_the_silence(self) -> None:
        """A tracked governor already logs its own spell boundaries; the sentinel must not double up.

        The governor here is latched by the popper's own consecutive-failure handling and read into the
        registry by the production collector, so the sentinel is being asked to honour a spell the worker
        genuinely opened rather than one the test declared.
        """
        manager = _pooled_manager()
        completed_at = await _complete_a_pop_attempt(manager)

        manager._state.consecutive_failed_jobs = 3
        await manager._job_popper.api_job_pop(urgent=True)
        manager._update_pop_governors()

        assert manager._state.too_many_consecutive_failed_jobs is True
        assert manager._state.last_pop_gate == PopGate.CONSECUTIVE_FAILURE_PAUSE
        assert manager._state.workload_intake_paused is False, "the quiet must come from the governor, not a pause"
        assert manager._pop_governor_registry.any_active(ignore=POP_LIVENESS_NON_EXPLAINING_GOVERNORS) is True

        warnings, warn_sink = _capture("WARNING")
        errors, error_sink = _capture("ERROR")
        try:
            manager._check_pop_liveness(completed_at + POP_LIVENESS_ERROR_SECONDS + 5.0)
        finally:
            logger.remove(warn_sink)
            logger.remove(error_sink)

        assert warnings == []
        assert errors == []

    async def test_error_backoff_does_not_explain_the_silence(self) -> None:
        """Backoff never stops pops, and its spell closes only when an attempt completes.

        A failed pop followed by a latched gate would hold the spell open forever; accepting it as an
        explanation would mute the sentinel for exactly that wedge. The backoff is opened by a real failed
        attempt here, so the spell under test is the one production would be carrying.
        """
        manager = _pooled_manager()
        completed_at = await _complete_a_pop_attempt(manager, outcome=RuntimeError("the horde could not be reached"))
        await _hold_pops_at_no_inference_process(manager)
        manager._update_pop_governors()

        assert manager._job_popper.is_in_error_backoff is True
        assert manager._pop_governor_registry.any_active() is True
        assert manager._pop_governor_registry.any_active(ignore=POP_LIVENESS_NON_EXPLAINING_GOVERNORS) is False

        lines, sink_id = _capture("WARNING")
        try:
            manager._check_pop_liveness(completed_at + POP_LIVENESS_WARN_SECONDS + 3.0)
        finally:
            logger.remove(sink_id)

        assert len(lines) == 1, lines
        assert str(PopGate.NO_INFERENCE_PROCESS) in lines[0]

    async def test_a_paused_worker_is_not_a_wedged_worker(self) -> None:
        """Intake being held worker-wide is the operator's own doing, and needs no alarm."""
        manager = _pooled_manager()
        completed_at = await _complete_a_pop_attempt(manager)

        manager._apply_supervisor_command(SupervisorControlMessage(command=SupervisorCommand.PAUSE))
        await manager._job_popper.api_job_pop(urgent=True)

        assert manager._state.workload_intake_paused is True
        assert manager._state.last_pop_gate == PopGate.INTAKE_PAUSED

        warnings, warn_sink = _capture("WARNING")
        errors, error_sink = _capture("ERROR")
        try:
            manager._check_pop_liveness(completed_at + POP_LIVENESS_ERROR_SECONDS + 5.0)
        finally:
            logger.remove(warn_sink)
            logger.remove(error_sink)

        assert warnings == []
        assert errors == []

    async def test_a_completed_attempt_rearms_the_sentinel(self) -> None:
        """Each silence episode gets its own disclosure, or a recurring wedge is reported only once."""
        manager, first_completed_at = await _silenced_at_the_pool_gate()

        lines, sink_id = _capture("WARNING")
        try:
            manager._check_pop_liveness(first_completed_at + POP_LIVENESS_WARN_SECONDS + 3.0)
            assert len(lines) == 1, lines

            manager._process_map[_INFERENCE_SLOT].last_process_state = HordeProcessState.WAITING_FOR_JOB
            second_completed_at = await _complete_a_pop_attempt(manager)
            assert second_completed_at > first_completed_at
            manager._check_pop_liveness(second_completed_at + 1.0)

            # A different gate on the second episode, so the re-armed disclosure is provably the new one.
            manager._process_map.pop(_SAFETY_SLOT)
            await manager._job_popper.api_job_pop(urgent=True)
            assert manager._state.last_pop_gate == PopGate.NO_SAFETY_PROCESS
            manager._check_pop_liveness(second_completed_at + POP_LIVENESS_WARN_SECONDS + 3.0)
        finally:
            logger.remove(sink_id)

        assert len(lines) == 2, lines
        assert str(PopGate.NO_SAFETY_PROCESS) in lines[1]

    async def test_silence_with_no_gate_reads_as_an_outstanding_request(self) -> None:
        """No gate plus no attempt is the fingerprint of a request (or a loop) that never came back.

        The pop path clears its gate immediately before the request goes out and stamps the attempt only
        once an answer arrives, so this pairing is reachable only while a request is genuinely in flight.
        That is the state the test holds the worker in.
        """
        manager = _pooled_manager()
        completed_at = await _complete_a_pop_attempt(manager)

        hanging = _HangingHordeSession()
        manager._api_sessions.set_horde_client_session(hanging)  # pyrefly: ignore - a stub stands in for the session
        in_flight = asyncio.create_task(manager._job_popper.api_job_pop(urgent=True))
        try:
            await asyncio.wait_for(hanging.reached.wait(), timeout=5.0)

            assert manager._state.last_pop_gate is None
            assert manager._state.last_pop_attempt_completed_at == completed_at

            lines, sink_id = _capture("WARNING")
            try:
                manager._check_pop_liveness(completed_at + POP_LIVENESS_WARN_SECONDS + 3.0)
            finally:
                logger.remove(sink_id)
        finally:
            in_flight.cancel()
            with pytest.raises(asyncio.CancelledError):
                await in_flight

        assert len(lines) == 1, lines
        assert "gate" not in lines[0] or "no gate" in lines[0]
        assert "outstanding" in lines[0] or "no longer running" in lines[0]

    async def test_a_recent_attempt_is_silent(self) -> None:
        """A worker that just reached the horde is healthy however little work it is getting."""
        manager, completed_at = await _silenced_at_the_pool_gate()

        warnings, warn_sink = _capture("WARNING")
        try:
            manager._check_pop_liveness(completed_at + 1.0)
        finally:
            logger.remove(warn_sink)

        assert warnings == []


_FROZEN_QUEUE_PHRASE = "full and not draining"
"""The phrase identifying the full-but-frozen escalation, distinct from the ordinary silence disclosure."""


async def _full_queue_manager(head_model: str = "head_model") -> tuple[HordeWorkerProcessManager, float]:
    """Build a manager whose local queue is genuinely full and whose pops are held at that gate.

    The queue depth is the popper's own, and the gate is recorded through the popper's own gate-noting
    entry point, so the held condition is the one the pop loop would produce rather than a hand-set field.

    Returns:
        The manager and the wall-clock time the gate began holding.
    """
    manager = make_testable_process_manager()
    await track_popped_job_async(manager._job_tracker, make_job_pop_response(head_model))
    while not manager._job_popper._is_queue_full(manager.bridge_data):
        await track_popped_job_async(manager._job_tracker, make_job_pop_response("trailing_model"))
    manager._job_popper._note_pop_gate(PopGate.QUEUE_FULL)
    held_since = manager._state.last_pop_gate_since
    manager._state.last_pop_attempt_completed_at = held_since
    return manager, held_since


class TestFullButFrozenQueue:
    """A full local queue is healthy only while it drains; a full queue that stops moving is a wedge."""

    async def test_a_full_queue_that_stops_draining_escalates_and_names_the_head(self) -> None:
        """Pops held at the full-queue gate with nothing dispatched or completed is not backpressure.

        The full-queue gate explains why no pop reaches the horde, and taking that as the whole explanation
        leaves the worker's most complete stall (queue full, every slot idle, nothing moving) reported as
        ordinary capacity management. The escalation has to name the head, because the head is the job whose
        block is holding everything behind it.
        """
        manager, held_since = await _full_queue_manager()
        assert manager._job_popper._is_queue_full(manager.bridge_data) is True
        assert manager._state.last_pop_gate == "queue_full"

        errors, sink_id = _capture("ERROR")
        try:
            for tick in range(0, int(POP_LIVENESS_FROZEN_QUEUE_SECONDS) * 2, 10):
                manager._check_pop_liveness(held_since + tick)
        finally:
            logger.remove(sink_id)

        frozen = [line for line in errors if _FROZEN_QUEUE_PHRASE in line]
        assert len(frozen) == 1, errors
        assert "head_model" in frozen[0]

    async def test_a_full_queue_that_keeps_draining_is_quiet(self) -> None:
        """Backpressure on a worker that is serving must not be reported as a stall.

        A worker at its queue depth holds this gate continuously and attempts no pops while it does, so the
        only thing separating capacity management from a wedge is whether work is still moving.
        """
        manager, held_since = await _full_queue_manager()

        errors, error_sink = _capture("ERROR")
        warnings, warn_sink = _capture("WARNING")
        try:
            for tick in range(0, int(POP_LIVENESS_FROZEN_QUEUE_SECONDS) * 2, 10):
                await manager._job_tracker.increment_jobs_completed()
                manager._check_pop_liveness(held_since + tick)
        finally:
            logger.remove(warn_sink)
            logger.remove(error_sink)

        assert [line for line in errors if _FROZEN_QUEUE_PHRASE in line] == []
        # The sentinel really did run past the thresholds that would have reached the escalation: the
        # ordinary silence disclosure fired on the same ticks.
        assert [line for line in warnings if "queue_full" in line] != []

    async def test_the_frozen_escalation_is_time_boxed_not_per_tick(self) -> None:
        """A stall the operator has already been told about restates itself on a clock, not on every tick."""
        manager, held_since = await _full_queue_manager()

        errors, sink_id = _capture("ERROR")
        try:
            span = int(POP_LIVENESS_FROZEN_QUEUE_SECONDS) * 3
            for tick in range(0, span, 5):
                manager._check_pop_liveness(held_since + tick)
        finally:
            logger.remove(sink_id)

        frozen = [line for line in errors if _FROZEN_QUEUE_PHRASE in line]
        assert len(frozen) == 2, frozen
