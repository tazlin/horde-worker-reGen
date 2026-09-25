"""The alchemy flow discloses its own intake silence, and names the gate holding it.

An alchemist-only worker's alchemy pop is its whole intake path. Every early return in the pop policy is
silent, so before this disclosure a worker held at a gate looked exactly like a worker the horde had no forms
for. These tests pin the two halves: the policy names its hold, and the disclosure fires on silence and stays
quiet for the holds that are the operator's own doing.
"""

from __future__ import annotations

from loguru import logger

from horde_worker_regen.process_management.config.worker_state import PopGate
from horde_worker_regen.process_management.jobs.alchemy_popper import (
    ALCHEMY_POP_LIVENESS_ERROR_SECONDS,
    ALCHEMY_POP_LIVENESS_WARN_SECONDS,
    AlchemyCoordinator,
)
from tests.process_management.conftest import make_testable_process_manager

_BASE = 1_000_000.0


def _coordinator(*, alchemist: bool = True) -> AlchemyCoordinator:
    """Build a real coordinator over the testable manager, with alchemy on or off."""
    return make_testable_process_manager(alchemist=alchemist)._alchemy_coordinator


def _capture_liveness_lines() -> tuple[list[str], int]:
    """Attach a loguru sink that keeps only the alchemy liveness disclosures."""
    lines: list[str] = []

    def _sink(message: object) -> None:
        text = message.record["message"]  # type: ignore[attr-defined]
        if text.startswith("Alchemy pop liveness:"):
            lines.append(f"{message.record['level'].name}: {text}")  # type: ignore[attr-defined]

    return lines, logger.add(_sink, level="WARNING")


def test_the_pop_policy_names_the_role_hold_and_should_pop_stamps_it() -> None:
    """With the role off the policy returns the role gate, and the boolean seam records it for the disclosure."""
    coordinator = _coordinator(alchemist=False)
    coordinator.bridge_data.alchemist = False

    assert coordinator._pop_gate() is PopGate.ALCHEMY_NOT_SERVED
    assert coordinator._should_pop() is False
    assert coordinator._state.alchemy_last_pop_gate == str(PopGate.ALCHEMY_NOT_SERVED)
    assert coordinator._state.alchemy_last_pop_gate_since > 0.0


def test_silence_past_the_notice_window_names_the_gate_then_escalates() -> None:
    """A held pop discloses once as a warning, then again as an error past the escalation window."""
    coordinator = _coordinator()
    coordinator._state.alchemy_last_pop_attempt_completed_at = _BASE
    coordinator._state.alchemy_last_pop_gate = str(PopGate.ALCHEMY_VRAM_HEADROOM)
    coordinator._state.alchemy_last_pop_gate_since = _BASE

    lines, sink = _capture_liveness_lines()
    try:
        coordinator._check_pop_liveness(_BASE + ALCHEMY_POP_LIVENESS_WARN_SECONDS - 1.0)
        assert lines == [], "the notice must not fire inside its own window"

        coordinator._check_pop_liveness(_BASE + ALCHEMY_POP_LIVENESS_WARN_SECONDS + 1.0)
        coordinator._check_pop_liveness(_BASE + ALCHEMY_POP_LIVENESS_WARN_SECONDS + 2.0)
        assert len(lines) == 1, lines
        assert "WARNING" in lines[0] and str(PopGate.ALCHEMY_VRAM_HEADROOM) in lines[0]

        coordinator._check_pop_liveness(_BASE + ALCHEMY_POP_LIVENESS_ERROR_SECONDS + 1.0)
        assert len(lines) == 2, lines
        assert "ERROR" in lines[1]
    finally:
        logger.remove(sink)


def test_a_completed_attempt_rearms_the_episode() -> None:
    """A pop that reaches the horde ends the spell, so the next silence starts its own clock."""
    coordinator = _coordinator()
    coordinator._state.alchemy_last_pop_attempt_completed_at = _BASE
    coordinator._state.alchemy_last_pop_gate = str(PopGate.QUEUE_FULL)
    coordinator._state.alchemy_last_pop_gate_since = _BASE

    lines, sink = _capture_liveness_lines()
    try:
        coordinator._check_pop_liveness(_BASE + ALCHEMY_POP_LIVENESS_WARN_SECONDS + 1.0)
        assert len(lines) == 1

        resumed_at = _BASE + ALCHEMY_POP_LIVENESS_WARN_SECONDS + 2.0
        coordinator._note_pop_attempt_completed(resumed_at)
        coordinator._check_pop_liveness(resumed_at + 1.0)
        assert len(lines) == 1, "a completed attempt must re-arm the episode rather than re-warn"

        coordinator._check_pop_liveness(resumed_at + ALCHEMY_POP_LIVENESS_WARN_SECONDS + 1.0)
        assert len(lines) == 2
    finally:
        logger.remove(sink)


def test_the_operators_own_holds_stay_silent() -> None:
    """The role choice and a worker-wide intake pause are intended, so neither is reported as a wedge."""
    coordinator = _coordinator()
    coordinator._state.alchemy_last_pop_attempt_completed_at = _BASE

    lines, sink = _capture_liveness_lines()
    try:
        coordinator._state.alchemy_last_pop_gate = str(PopGate.ALCHEMY_NOT_SERVED)
        coordinator._state.alchemy_last_pop_gate_since = _BASE
        coordinator._check_pop_liveness(_BASE + ALCHEMY_POP_LIVENESS_ERROR_SECONDS + 1.0)
        assert lines == [], "the operator turning the role off is not a wedge"

        coordinator._state.alchemy_last_pop_gate = str(PopGate.RAM_PRESSURE)
        coordinator._state.self_throttle_paused = True
        coordinator._check_pop_liveness(_BASE + ALCHEMY_POP_LIVENESS_ERROR_SECONDS + 2.0)
        assert lines == [], "a worker-wide intake pause is accounted for elsewhere"
    finally:
        logger.remove(sink)
