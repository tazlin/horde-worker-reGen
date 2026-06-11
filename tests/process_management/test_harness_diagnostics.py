"""Unit tests for harness diagnostic utilities and failure-mode detection.

These verify that the harness helpers (``_cleanup_stale_abort_file``,
``_determine_exit_reason``, ``_collect_run_diagnostics``, ``HarnessResult.failure_summary``)
produce correct and actionable information when the harness misbehaves — without
spawning any child processes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from horde_worker_regen.harness import (
    HarnessResult,
    _cleanup_stale_abort_file,
    _collect_run_diagnostics,
    _determine_exit_reason,
)
from horde_worker_regen.process_management.horde_process import HordeProcessType
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
)


class TestCleanupStaleAbortFile:
    """Tests for `_cleanup_stale_abort_file`."""

    def test_removes_file_when_present(self, tmp_path: Path) -> None:
        """When a .abort file exists, it should be removed."""
        abort_file = tmp_path / ".abort"
        abort_file.write_text("")

        with patch("os.getcwd", return_value=str(tmp_path)):
            _cleanup_stale_abort_file()

        assert not abort_file.exists(), ".abort file should have been removed"

    def test_noop_when_file_absent(self, tmp_path: Path) -> None:
        """When no .abort file exists, the call should be a no-op."""
        abort_file = tmp_path / ".abort"
        assert not abort_file.exists()

        with patch("os.getcwd", return_value=str(tmp_path)):
            _cleanup_stale_abort_file()

        assert not abort_file.exists()

    def test_noop_when_os_remove_raises(self, tmp_path: Path) -> None:
        """If os.remove fails the exception should propagate so the caller knows cleanup couldn't proceed."""
        abort_file = tmp_path / ".abort"
        abort_file.write_text("")

        with (
            patch("os.getcwd", return_value=str(tmp_path)),
            patch("os.remove", side_effect=PermissionError("access denied")),
            pytest.raises(PermissionError),
        ):
            _cleanup_stale_abort_file()


class TestDetermineExitReason:
    """Tests for `_determine_exit_reason`."""

    def test_completed_when_all_jobs_done(self) -> None:
        """When all jobs are accounted for, return 'completed'."""
        manager = make_testable_process_manager()
        # Simulate 3 completed jobs.
        manager._job_tracker._total_num_completed_jobs = 3
        manager._job_tracker._num_jobs_faulted = 0

        reason = _determine_exit_reason(
            manager=manager,
            num_jobs_expected=3,
            timed_out=False,
            exception_raised=None,
        )
        assert reason == "completed"

    def test_timed_out(self) -> None:
        """When the run timed out, return 'timed_out'."""
        manager = make_testable_process_manager()

        reason = _determine_exit_reason(
            manager=manager,
            num_jobs_expected=3,
            timed_out=True,
            exception_raised=None,
        )
        assert reason == "timed_out"

    def test_exception_raised(self) -> None:
        """When an exception was raised, return it in the reason."""
        manager = make_testable_process_manager()
        exc = ValueError("something went wrong")

        reason = _determine_exit_reason(
            manager=manager,
            num_jobs_expected=3,
            timed_out=False,
            exception_raised=exc,
        )
        assert reason == "exception: ValueError: something went wrong"

    def test_shut_down_before_completion(self) -> None:
        """When shut_down is set but jobs aren't done, report it."""
        manager = make_testable_process_manager()
        manager._state.shut_down = True

        reason = _determine_exit_reason(
            manager=manager,
            num_jobs_expected=3,
            timed_out=False,
            exception_raised=None,
        )
        assert reason == "shut_down_before_completion"

    def test_shut_down_not_set_falls_through_to_unknown(self) -> None:
        """When nothing specific is detected, return 'unknown'."""
        manager = make_testable_process_manager()

        reason = _determine_exit_reason(
            manager=manager,
            num_jobs_expected=3,
            timed_out=False,
            exception_raised=None,
        )
        assert reason == "unknown"

    def test_completed_takes_priority_over_shut_down(self) -> None:
        """When jobs are complete, 'completed' wins even if shut_down is set."""
        manager = make_testable_process_manager()
        manager._state.shut_down = True
        manager._job_tracker._total_num_completed_jobs = 3

        reason = _determine_exit_reason(
            manager=manager,
            num_jobs_expected=3,
            timed_out=False,
            exception_raised=None,
        )
        assert reason == "completed"


class TestCollectRunDiagnostics:
    """Tests for `_collect_run_diagnostics`."""

    def test_no_warnings_for_healthy_run(self) -> None:
        """A run with processes and completed jobs produces no diagnostics."""
        manager = make_testable_process_manager()
        manager._job_tracker._total_num_completed_jobs = 3
        # Populate process map with mock inference and safety processes so
        # the "no processes" diagnostic is not triggered.
        manager._process_map[0] = make_mock_process_info(process_id=0)
        manager._process_map[1] = make_mock_process_info(
            process_id=1,
            process_type=HordeProcessType.SAFETY,
        )
        # Populate the tracker's internal lookup so "no jobs popped" diagnostic
        # is not triggered.  `jobs_lookup` filters to entries with non-None job_info.

        from horde_worker_regen.process_management.job_tracker import JobStage, TrackedJob

        dummy_job = make_job_pop_response()
        tracked = TrackedJob(
            job_id=dummy_job.id_,  # pyrefly: ignore
            sdk_api_job_info=dummy_job,
            stage=JobStage.PENDING_SUBMIT,
            job_info=Mock(),  # non-None so jobs_lookup includes this entry
        )
        manager._job_tracker._jobs[dummy_job.id_] = tracked  # pyrefly: ignore

        diags = _collect_run_diagnostics(
            manager=manager,
            num_jobs_expected=3,
            elapsed=10.0,
        )
        assert diags == []

    def test_warns_when_zero_processes(self) -> None:
        """When no inference or safety processes are started, warn."""
        manager = make_testable_process_manager()

        diags = _collect_run_diagnostics(
            manager=manager,
            num_jobs_expected=3,
            elapsed=5.0,
        )
        assert any("No inference processes" in d for d in diags)
        assert any("No safety processes" in d for d in diags)

    def test_warns_when_no_jobs_processed(self) -> None:
        """When no jobs completed or faulted after a meaningful time, warn."""
        manager = make_testable_process_manager()

        diags = _collect_run_diagnostics(
            manager=manager,
            num_jobs_expected=3,
            elapsed=5.0,
        )
        assert any("No jobs completed or faulted" in d for d in diags)

    def test_no_jobs_warning_when_elapsed_short(self) -> None:
        """Don't warn about no jobs when the run was very short (<2s)."""
        manager = make_testable_process_manager()

        diags = _collect_run_diagnostics(
            manager=manager,
            num_jobs_expected=3,
            elapsed=1.0,
        )
        assert not any("No jobs completed or faulted" in d for d in diags)

    def test_warns_when_no_jobs_popped(self) -> None:
        """When no jobs appear in the tracker lookup, flag it."""
        manager = make_testable_process_manager()

        diags = _collect_run_diagnostics(
            manager=manager,
            num_jobs_expected=3,
            elapsed=5.0,
        )
        assert any("No jobs were ever popped" in d for d in diags)


class TestHarnessResultFailureSummary:
    """Tests for `HarnessResult.failure_summary()`."""

    def test_empty_for_success(self) -> None:
        """A successful result returns 'no issues detected'."""
        result = HarnessResult(
            num_jobs_expected=3,
            num_jobs_completed=3,
            num_jobs_faulted=0,
            elapsed_seconds=5.0,
            timed_out=False,
            exit_reason="completed",
        )
        assert result.failure_summary() == "no issues detected"

    def test_includes_exit_reason(self) -> None:
        """The exit reason is always included when set."""
        result = HarnessResult(
            num_jobs_expected=3,
            num_jobs_completed=3,
            num_jobs_faulted=0,
            elapsed_seconds=5.0,
            timed_out=False,
            exit_reason="shut_down_before_completion",
        )
        summary = result.failure_summary()
        assert "exit_reason=shut_down_before_completion" in summary

    def test_includes_timed_out(self) -> None:
        """Timed out flag is surfaced."""
        result = HarnessResult(
            num_jobs_expected=3,
            num_jobs_completed=0,
            num_jobs_faulted=0,
            elapsed_seconds=120.0,
            timed_out=True,
            exit_reason="timed_out",
        )
        summary = result.failure_summary()
        assert "timed_out=True" in summary

    def test_includes_jobs_completed_count(self) -> None:
        """When fewer jobs than expected completed, include the count."""
        result = HarnessResult(
            num_jobs_expected=3,
            num_jobs_completed=0,
            num_jobs_faulted=0,
            elapsed_seconds=5.0,
            timed_out=False,
        )
        summary = result.failure_summary()
        assert "jobs_completed=0/3" in summary

    def test_includes_jobs_faulted_when_nonzero(self) -> None:
        """Faulted jobs are reported when > 0."""
        result = HarnessResult(
            num_jobs_expected=3,
            num_jobs_completed=0,
            num_jobs_faulted=2,
            elapsed_seconds=5.0,
            timed_out=False,
        )
        summary = result.failure_summary()
        assert "jobs_faulted=2" in summary

    def test_includes_audit_failures(self) -> None:
        """Audit failure count is shown when non-empty."""
        result = HarnessResult(
            num_jobs_expected=3,
            num_jobs_completed=3,
            num_jobs_faulted=0,
            elapsed_seconds=5.0,
            timed_out=False,
            audit_failures=["Job X double submit"],
        )
        summary = result.failure_summary()
        assert "audit_failures=1" in summary

    def test_includes_diagnostics(self) -> None:
        """Diagnostic messages are included in the summary."""
        result = HarnessResult(
            num_jobs_expected=3,
            num_jobs_completed=0,
            num_jobs_faulted=0,
            elapsed_seconds=5.0,
            timed_out=False,
            diagnostics=["No jobs were ever popped"],
        )
        summary = result.failure_summary()
        assert "diagnostics=['No jobs were ever popped']" in summary
