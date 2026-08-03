"""The "the horde had no jobs" flag is evidence about one attempt, so it must stale without attempts.

``last_pop_no_jobs_available`` is set by the last pop attempt that reached the horde. Consumers that
suppress recovery while it holds have to know whether that evidence still describes the present: a worker
whose pop loop stopped attempting keeps the flag at its last value indefinitely, which would otherwise
suppress recovery forever.
"""

from __future__ import annotations

from horde_worker_regen.process_management.config.worker_state import (
    POP_NO_JOBS_EVIDENCE_WINDOW_SECONDS,
    WorkerState,
)


def test_evidence_is_fresh_within_the_window() -> None:
    """A no-jobs verdict from an attempt inside the window still describes the present."""
    now = 1_000.0
    state = WorkerState(
        last_pop_no_jobs_available=True,
        last_pop_attempt_completed_at=now - (POP_NO_JOBS_EVIDENCE_WINDOW_SECONDS / 2),
    )

    assert state.pop_no_jobs_evidence_fresh(now) is True


def test_evidence_stales_once_no_attempt_completes() -> None:
    """With no attempt for longer than the window the flag is a stale reading, not a live verdict."""
    now = 1_000.0
    state = WorkerState(
        last_pop_no_jobs_available=True,
        last_pop_attempt_completed_at=now - (POP_NO_JOBS_EVIDENCE_WINDOW_SECONDS + 1.0),
    )

    assert state.pop_no_jobs_evidence_fresh(now) is False


def test_window_boundary_is_inclusive() -> None:
    """An attempt exactly at the window edge still counts as fresh."""
    now = 1_000.0
    state = WorkerState(
        last_pop_no_jobs_available=True,
        last_pop_attempt_completed_at=now - POP_NO_JOBS_EVIDENCE_WINDOW_SECONDS,
    )

    assert state.pop_no_jobs_evidence_fresh(now) is True


def test_flag_unset_is_never_fresh_evidence() -> None:
    """Without the flag there is no no-jobs evidence to be fresh, however recent the attempt."""
    state = WorkerState(last_pop_no_jobs_available=False, last_pop_attempt_completed_at=1_000.0)

    assert state.pop_no_jobs_evidence_fresh(1_000.0) is False


def test_a_worker_that_never_popped_has_no_fresh_evidence() -> None:
    """The default zero stamp must not read as a recent attempt."""
    state = WorkerState(last_pop_no_jobs_available=True)

    assert state.pop_no_jobs_evidence_fresh(1_000.0) is False
