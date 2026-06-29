"""The post-processing fault breaker: repeated unhostable post-processing peaks disable the feature.

A post-processing peak that cannot be hosted faults the job and the horde reissues it, but a worker that
keeps faulting trips the horde's forced-maintenance (the spiral this guards against). These pin the worker's
self-protective breaker: a rolling-window counter fed by both fault sources (the planner's unhostable-peak
faults and watchdog-reaped post-processing stalls), a trip when the count *exceeds* the threshold, and a
session latch that is never auto-cleared (the over-commit is structural; only a restart clears it).
"""

from __future__ import annotations

from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from tests.process_management.conftest import make_testable_process_manager


class TestPostProcessingFaultCounter:
    """The rolling-window counter the breaker reads, fed by both fault sources."""

    def test_counts_within_window_and_excludes_older(self) -> None:
        """Each recorded fault is counted within the window; a zero-width window counts none."""
        job_tracker = JobTracker()
        assert job_tracker.count_recent_post_processing_faults(600) == 0

        job_tracker.note_post_processing_overcommit_fault()
        job_tracker.note_post_processing_overcommit_fault()

        assert job_tracker.count_recent_post_processing_faults(600) == 2
        # A zero-width window excludes faults recorded a moment ago (the boundary is strict-enough to prune).
        assert job_tracker.count_recent_post_processing_faults(-1) == 0


class TestPostProcessingFaultBreaker:
    """The trip/latch behaviour the control loop drives via ``_apply_post_processing_fault_breaker``."""

    def test_trips_only_after_exceeding_threshold_and_latches(self) -> None:
        """The breaker tolerates exactly the threshold and trips on the next fault, then latches."""
        manager = make_testable_process_manager(
            post_processing_fault_breaker_enabled=True,
            post_processing_fault_threshold=4,
            post_processing_fault_window_seconds=1800,
        )

        # Exactly the threshold is tolerated (the trip is strictly greater-than).
        for _ in range(4):
            manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is False

        # One more crosses it: the breaker trips and stamps the time.
        manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True
        assert manager._state.post_processing_breaker_tripped_at > 0

    def test_session_latched_never_auto_clears(self) -> None:
        """Once tripped the latch persists across later checks (and a soft reset, which never rebuilds state)."""
        manager = make_testable_process_manager(
            post_processing_fault_threshold=1,
            post_processing_fault_window_seconds=600,
        )
        for _ in range(2):
            manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True

        # A subsequent check (even with the faults aged out of a tiny window) leaves the latch set: the
        # over-commit is structural, so the breaker clears only on a process restart, not on its own.
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True

    def test_disabled_flag_prevents_trip(self) -> None:
        """With the breaker disabled, no number of faults latches it off."""
        manager = make_testable_process_manager(
            post_processing_fault_breaker_enabled=False,
            post_processing_fault_threshold=1,
            post_processing_fault_window_seconds=1800,
        )
        for _ in range(10):
            manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is False
