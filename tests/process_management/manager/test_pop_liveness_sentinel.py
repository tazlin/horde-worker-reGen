"""The worker discloses a pop loop that has stopped reaching the horde.

Silence on the intake path is the one failure the worker cannot distinguish from a quiet horde by looking
at its own logs: the pop coroutine returns early at any of a dozen gates without saying so, so a wedged
pool and an empty queue look identical. The sentinel turns that silence into a line naming the gate, and
stays quiet whenever the silence is explained (the worker is deliberately not taking work, or a tracked
governor is holding pops on purpose).
"""

from __future__ import annotations

import time

from loguru import logger

from horde_worker_regen.process_management.process_manager import (
    POP_LIVENESS_ERROR_SECONDS,
    POP_LIVENESS_FROZEN_QUEUE_SECONDS,
    POP_LIVENESS_WARN_SECONDS,
    HordeWorkerProcessManager,
)
from horde_worker_regen.process_management.scheduling.pop_governor_registry import PopGovernorReading
from tests.process_management.conftest import (
    make_job_pop_response,
    make_testable_process_manager,
    track_popped_job_async,
)


def _capture(level: str) -> tuple[list[str], int]:
    """Attach a loguru sink collecting messages at exactly ``level``."""
    lines: list[str] = []
    sink_id = logger.add(
        lambda message: lines.append(message.record["message"]),
        level=level,
        filter=lambda record: record["level"].name == level,
    )
    return lines, sink_id


def _silent_manager(
    *,
    silent_for: float,
    gate: str | None = "no_inference_process",
    gate_held_for: float | None = None,
) -> tuple[HordeWorkerProcessManager, float]:
    """Build a manager whose popper has not completed an attempt for ``silent_for`` seconds."""
    manager = make_testable_process_manager()
    now = time.time()
    manager._state.last_pop_attempt_completed_at = now - silent_for
    manager._state.last_pop_gate = gate
    manager._state.last_pop_gate_since = now - (gate_held_for if gate_held_for is not None else silent_for)
    return manager, now


class TestPopLivenessSentinel:
    """The sentinel fires once per silence episode, and only when the silence is unexplained."""

    def test_silence_past_the_warn_threshold_names_the_gate(self) -> None:
        """The disclosure has to carry the gate: the gate is what an operator can act on."""
        manager, now = _silent_manager(silent_for=POP_LIVENESS_WARN_SECONDS + 3.0)

        lines, sink_id = _capture("WARNING")
        try:
            manager._check_pop_liveness(now)
        finally:
            logger.remove(sink_id)

        assert len(lines) == 1, lines
        assert "no_inference_process" in lines[0]

    def test_the_warning_is_edge_triggered(self) -> None:
        """The control loop ticks many times a second, so a level-triggered line would drown the log."""
        manager, now = _silent_manager(silent_for=POP_LIVENESS_WARN_SECONDS + 3.0)

        lines, sink_id = _capture("WARNING")
        try:
            for tick in range(5):
                manager._check_pop_liveness(now + tick)
        finally:
            logger.remove(sink_id)

        assert len(lines) == 1, lines

    def test_prolonged_silence_escalates_to_an_error(self) -> None:
        """Minutes without reaching the horde is a wedge, not a slow patch."""
        manager, now = _silent_manager(silent_for=POP_LIVENESS_ERROR_SECONDS + 5.0)

        lines, sink_id = _capture("ERROR")
        try:
            manager._check_pop_liveness(now)
            manager._check_pop_liveness(now + 1.0)
        finally:
            logger.remove(sink_id)

        assert len(lines) == 1, lines
        assert "no_inference_process" in lines[0]

    def test_the_error_repeat_is_time_boxed_not_per_tick(self) -> None:
        """The control loop ticks constantly, so a persisting wedge restates itself on a clock, not a tick."""
        manager, now = _silent_manager(silent_for=POP_LIVENESS_ERROR_SECONDS + 5.0)

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

    def test_an_active_pop_governor_explains_the_silence(self) -> None:
        """A tracked governor already logs its own spell boundaries; the sentinel must not double up."""
        manager, now = _silent_manager(silent_for=POP_LIVENESS_ERROR_SECONDS + 5.0)
        manager._pop_governor_registry.update(
            [PopGovernorReading(name="large_model_switch", label="Large-model switch", active=True)],
            now=now,
        )

        warnings, warn_sink = _capture("WARNING")
        errors, error_sink = _capture("ERROR")
        try:
            manager._check_pop_liveness(now)
        finally:
            logger.remove(warn_sink)
            logger.remove(error_sink)

        assert warnings == []
        assert errors == []

    def test_error_backoff_does_not_explain_the_silence(self) -> None:
        """Backoff never stops pops, and its spell closes only when an attempt completes.

        A failed pop followed by a latched gate would hold the spell open forever; accepting it as an
        explanation would mute the sentinel for exactly that wedge.
        """
        manager, now = _silent_manager(silent_for=POP_LIVENESS_WARN_SECONDS + 3.0)
        manager._pop_governor_registry.update(
            [PopGovernorReading(name="pop_error_backoff", label="Pop error-backoff", active=True)],
            now=now,
        )

        lines, sink_id = _capture("WARNING")
        try:
            manager._check_pop_liveness(now)
        finally:
            logger.remove(sink_id)

        assert len(lines) == 1, lines
        assert "no_inference_process" in lines[0]

    def test_a_paused_worker_is_not_a_wedged_worker(self) -> None:
        """Intake being held worker-wide is the operator's own doing, and needs no alarm."""
        manager, now = _silent_manager(silent_for=POP_LIVENESS_ERROR_SECONDS + 5.0)
        manager._state.supervisor_paused = True

        warnings, warn_sink = _capture("WARNING")
        errors, error_sink = _capture("ERROR")
        try:
            manager._check_pop_liveness(now)
        finally:
            logger.remove(warn_sink)
            logger.remove(error_sink)

        assert warnings == []
        assert errors == []

    def test_a_completed_attempt_rearms_the_sentinel(self) -> None:
        """Each silence episode gets its own disclosure, or a recurring wedge is reported only once."""
        manager, now = _silent_manager(silent_for=POP_LIVENESS_WARN_SECONDS + 3.0)

        lines, sink_id = _capture("WARNING")
        try:
            manager._check_pop_liveness(now)
            manager._state.last_pop_attempt_completed_at = now + 1.0
            manager._check_pop_liveness(now + 2.0)
            manager._state.last_pop_gate = "queue_full"
            manager._state.last_pop_gate_since = now + 2.0
            manager._check_pop_liveness(now + 2.0 + POP_LIVENESS_WARN_SECONDS + 3.0)
        finally:
            logger.remove(sink_id)

        assert len(lines) == 2, lines
        assert "queue_full" in lines[1]

    def test_silence_with_no_gate_reads_as_an_outstanding_request(self) -> None:
        """No gate plus no attempt is the fingerprint of a request (or a loop) that never came back."""
        manager, now = _silent_manager(silent_for=POP_LIVENESS_WARN_SECONDS + 3.0, gate=None)

        lines, sink_id = _capture("WARNING")
        try:
            manager._check_pop_liveness(now)
        finally:
            logger.remove(sink_id)

        assert len(lines) == 1, lines
        assert "gate" not in lines[0] or "no gate" in lines[0]
        assert "outstanding" in lines[0] or "no longer running" in lines[0]

    def test_a_recent_attempt_is_silent(self) -> None:
        """A worker that just reached the horde is healthy however little work it is getting."""
        manager, now = _silent_manager(silent_for=1.0)

        warnings, warn_sink = _capture("WARNING")
        try:
            manager._check_pop_liveness(now)
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
    manager._job_popper._note_pop_gate("queue_full")
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
