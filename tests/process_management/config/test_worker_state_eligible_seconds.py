"""Tests for the eligible-seconds (productive-time) clock that denominates the kudos/hr rate."""

from __future__ import annotations

from horde_worker_regen.process_management.config.worker_state import WorkerState


def test_eligible_seconds_does_not_run_before_first_submit() -> None:
    """No productive time is credited until the first submit, so the cold-start lead-in is excluded."""
    state = WorkerState()

    state.tick_eligible_seconds(100.0, has_pipeline_work=True, max_dt=5.0)
    state.tick_eligible_seconds(101.0, has_pipeline_work=True, max_dt=5.0)

    assert state.first_kudos_event_time is None
    assert state.eligible_seconds_total == 0.0


def test_eligible_seconds_accumulates_only_while_pipeline_has_work() -> None:
    """After the first submit, only ticks with pipeline work advance the clock; idle ticks are skipped."""
    state = WorkerState()

    # Establish a tick baseline, then record the first submit.
    state.tick_eligible_seconds(100.0, has_pipeline_work=True, max_dt=5.0)
    state.note_first_kudos_event(100.5)

    # A productive second is counted.
    state.tick_eligible_seconds(101.0, has_pipeline_work=True, max_dt=5.0)
    assert state.eligible_seconds_total == 1.0

    # An empty-pipeline stretch (idle, maintenance, or a drained pause) is not.
    state.tick_eligible_seconds(105.0, has_pipeline_work=False, max_dt=5.0)
    assert state.eligible_seconds_total == 1.0

    # Work resumes: only the productive interval since the last tick counts.
    state.tick_eligible_seconds(106.0, has_pipeline_work=True, max_dt=5.0)
    assert state.eligible_seconds_total == 2.0


def test_eligible_seconds_clamps_a_stalled_gap() -> None:
    """A gap far larger than a tick (a stalled loop) is clamped so it cannot dump spurious productive time."""
    state = WorkerState()

    state.tick_eligible_seconds(100.0, has_pipeline_work=True, max_dt=5.0)
    state.note_first_kudos_event(100.0)

    state.tick_eligible_seconds(160.0, has_pipeline_work=True, max_dt=5.0)

    assert state.eligible_seconds_total == 5.0


def test_note_first_kudos_event_is_idempotent() -> None:
    """The first-submit timestamp latches once and is not overwritten by later submits."""
    state = WorkerState()

    state.note_first_kudos_event(100.0)
    state.note_first_kudos_event(200.0)

    assert state.first_kudos_event_time == 100.0
