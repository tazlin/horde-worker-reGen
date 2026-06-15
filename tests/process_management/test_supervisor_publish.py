"""Tests for the dirty-gated, floor-bounded supervisor snapshot publishing cadence.

Publishing must surface display-relevant change promptly (a changed state signature publishes on the
next tick) while staying quiet when nothing changes (a periodic floor still emits a heartbeat frame so
the TUI can tell a live worker from a hung one). The cadence is isolated here from the full snapshot
build, which is exercised elsewhere.
"""

from __future__ import annotations

from horde_worker_regen.process_management.messages import HordeProcessState

from .conftest import make_mock_process_info, make_testable_process_manager


class _Recorder:
    """A stand-in supervisor channel that counts the snapshots it is handed."""

    def __init__(self) -> None:
        self.count = 0
        self.closed = False

    def send_snapshot(self, snapshot: object) -> bool:
        """Record a send and report success (the worker keeps the channel)."""
        self.count += 1
        return True


def test_publish_is_dirty_gated_with_a_floor() -> None:
    """Snapshots publish on signature change and at the floor, and are suppressed when unchanged."""
    manager = make_testable_process_manager()
    recorder = _Recorder()
    manager._supervisor = recorder  # type: ignore[assignment]
    manager._build_worker_state_snapshot = lambda: object()  # type: ignore[assignment,method-assign,return-value]
    manager._process_map[0] = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)

    # First publish: the signature changes from None to a value, so a frame goes out.
    manager._publish_supervisor_snapshot()
    assert recorder.count == 1

    # No change and within the floor: suppressed.
    manager._publish_supervisor_snapshot()
    assert recorder.count == 1

    # A state change makes the signature differ: it publishes on the next tick (~2 Hz responsiveness).
    manager._process_map[0].last_process_state = HordeProcessState.INFERENCE_STARTING
    manager._publish_supervisor_snapshot()
    assert recorder.count == 2

    # Still no further change, still within the floor: suppressed.
    manager._publish_supervisor_snapshot()
    assert recorder.count == 2

    # Simulate the floor elapsing: a heartbeat frame goes out even with no state change.
    manager._last_supervisor_publish_time -= manager._supervisor_publish_floor_interval + 1.0
    manager._publish_supervisor_snapshot()
    assert recorder.count == 3
