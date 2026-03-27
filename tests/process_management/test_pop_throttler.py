"""Tests for PopThrottler."""

from __future__ import annotations

import time
from unittest.mock import Mock, patch

import pytest

from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.pop_throttler import (
    CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS,
    PopThrottler,
)

from .conftest import make_mock_bridge_data


def _make_throttler(
    *,
    job_tracker: JobTracker | None = None,
    default_pop_frequency: float = 1.0,
    error_pop_frequency: float = 5.0,
) -> PopThrottler:
    if job_tracker is None:
        job_tracker = JobTracker()
    return PopThrottler(
        job_tracker=job_tracker,
        default_pop_frequency=default_pop_frequency,
        error_pop_frequency=error_pop_frequency,
    )


class TestPopFrequencyDefaults:
    """Constructor sets frequencies correctly."""

    def test_default_frequency_is_one_second(self) -> None:
        throttler = _make_throttler()
        assert throttler.current_pop_frequency == 1.0

    def test_custom_default_frequency(self) -> None:
        throttler = _make_throttler(default_pop_frequency=2.5)
        assert throttler.current_pop_frequency == 2.5

    def test_custom_error_frequency(self) -> None:
        throttler = _make_throttler(error_pop_frequency=10.0)
        throttler.on_pop_error()
        assert throttler.current_pop_frequency == 10.0


class TestIsPopTooSoon:
    """Timing gate: ensures minimum interval between pops."""

    def test_pop_immediately_after_last_is_too_soon(self) -> None:
        throttler = _make_throttler(default_pop_frequency=1.0)
        assert throttler.is_pop_too_soon(time.time()) is True

    def test_pop_after_delay_is_not_too_soon(self) -> None:
        throttler = _make_throttler(default_pop_frequency=1.0)
        past = time.time() - 2.0
        assert throttler.is_pop_too_soon(past) is False

    def test_pop_exactly_at_boundary_is_not_too_soon(self) -> None:
        """When elapsed == frequency, the pop should be allowed."""
        throttler = _make_throttler(default_pop_frequency=1.0)
        past = time.time() - 1.0
        # At boundary or just past, should not block
        assert throttler.is_pop_too_soon(past) is False

    def test_error_frequency_raises_the_gate(self) -> None:
        throttler = _make_throttler(error_pop_frequency=5.0)
        throttler.on_pop_error()

        two_seconds_ago = time.time() - 2.0
        assert throttler.is_pop_too_soon(two_seconds_ago) is True

        six_seconds_ago = time.time() - 6.0
        assert throttler.is_pop_too_soon(six_seconds_ago) is False

    def test_zero_last_pop_time_always_allows(self) -> None:
        """First ever pop (last_pop_time == 0) should never be blocked."""
        throttler = _make_throttler()
        assert throttler.is_pop_too_soon(0.0) is False


class TestOnPopSuccess:
    """Success resets frequency to default."""

    def test_resets_to_default_after_error(self) -> None:
        throttler = _make_throttler(default_pop_frequency=1.0, error_pop_frequency=5.0)
        throttler.on_pop_error()
        assert throttler.current_pop_frequency == 5.0

        throttler.on_pop_success()
        assert throttler.current_pop_frequency == 1.0

    def test_idempotent_when_already_at_default(self) -> None:
        throttler = _make_throttler()
        throttler.on_pop_success()
        assert throttler.current_pop_frequency == 1.0


class TestOnPopError:
    """Error slows down pop frequency."""

    def test_switches_to_error_frequency(self) -> None:
        throttler = _make_throttler(error_pop_frequency=5.0)
        throttler.on_pop_error()
        assert throttler.current_pop_frequency == 5.0

    def test_multiple_errors_stay_at_error_frequency(self) -> None:
        throttler = _make_throttler(error_pop_frequency=5.0)
        throttler.on_pop_error()
        throttler.on_pop_error()
        throttler.on_pop_error()
        assert throttler.current_pop_frequency == 5.0


class TestOnNoJobsAvailable:
    """Idle time tracking when the queue is empty."""

    def test_first_empty_poll_starts_tracking(self) -> None:
        throttler = _make_throttler()
        now = time.time()
        throttler.on_no_jobs_available(now, queue_empty=True)

        assert throttler._last_pop_no_jobs_available_time == now
        assert throttler._time_spent_no_jobs_available == 0.0  # first call: elapsed from self is 0

    def test_subsequent_empty_polls_accumulate_idle_time(self) -> None:
        throttler = _make_throttler()
        t1 = 1000.0
        throttler.on_no_jobs_available(t1, queue_empty=True)
        assert throttler._last_pop_no_jobs_available_time == t1

        t2 = 1003.0
        throttler.on_no_jobs_available(t2, queue_empty=True)
        assert throttler._time_spent_no_jobs_available == pytest.approx(3.0)
        assert throttler._last_pop_no_jobs_available_time == t2

    def test_non_empty_queue_does_not_track(self) -> None:
        """When queue has jobs, no idle time should accumulate."""
        throttler = _make_throttler()
        throttler.on_no_jobs_available(1000.0, queue_empty=False)

        assert throttler._last_pop_no_jobs_available_time == 0.0
        assert throttler._time_spent_no_jobs_available == 0.0

    def test_mixed_empty_and_nonempty_polls(self) -> None:
        throttler = _make_throttler()

        # Empty poll starts tracking
        throttler.on_no_jobs_available(100.0, queue_empty=True)
        throttler.on_no_jobs_available(102.0, queue_empty=True)
        assert throttler._time_spent_no_jobs_available == pytest.approx(2.0)

        # Non-empty poll does NOT reset the tracker (by design)
        throttler.on_no_jobs_available(105.0, queue_empty=False)
        assert throttler._time_spent_no_jobs_available == pytest.approx(2.0)


class TestOnJobPopped:
    """Job popped resets idle tracking."""

    def test_resets_idle_time_tracker(self) -> None:
        throttler = _make_throttler()
        throttler.on_no_jobs_available(100.0, queue_empty=True)
        throttler.on_no_jobs_available(103.0, queue_empty=True)
        assert throttler._last_pop_no_jobs_available_time == 103.0

        throttler.on_job_popped()
        assert throttler._last_pop_no_jobs_available_time == 0.0


class TestShouldWaitForMegapixelsteps:
    """Megapixelstep wait logic — the most complex throttle path."""

    def _make_bridge_data(self, **overrides: object) -> Mock:
        return make_mock_bridge_data(**overrides)

    def test_no_wait_when_pending_below_threshold(self) -> None:
        """When pending MPS is low enough, no waiting should occur."""
        jt = JobTracker()
        # No jobs in queue → pending MPS = 0
        throttler = _make_throttler(job_tracker=jt)
        bd = self._make_bridge_data()

        assert throttler.should_wait_for_megapixelsteps(bd) is False

    def test_no_wait_resets_trigger_flag(self) -> None:
        """When we drop below threshold, the trigger flag must be cleared."""
        jt = JobTracker()
        jt._triggered_max_pending_megapixelsteps = True
        throttler = _make_throttler(job_tracker=jt)
        bd = self._make_bridge_data()

        throttler.should_wait_for_megapixelsteps(bd)
        assert jt._triggered_max_pending_megapixelsteps is False

    def test_first_call_above_threshold_sets_trigger(self) -> None:
        """First time we're above max pending MPS, the trigger flag and time should be set."""
        jt = JobTracker()
        jt._max_pending_megapixelsteps = 5
        throttler = _make_throttler(job_tracker=jt)
        bd = self._make_bridge_data()

        # Add many large jobs to push pending MPS above threshold
        for _ in range(10):
            job = Mock()
            job.payload.width = 1024
            job.payload.height = 1024
            job.payload.ddim_steps = 50
            job.payload.n_iter = 1
            job.payload.post_processing = []
            job.payload.loras = []
            job.payload.control_type = None
            job.payload.hires_fix = False
            jt.jobs_pending_inference.append(job)

        assert jt.should_wait_for_pending_megapixelsteps() is True

        result = throttler.should_wait_for_megapixelsteps(bd)
        assert result is True
        assert jt._triggered_max_pending_megapixelsteps is True
        assert jt._triggered_max_pending_megapixelsteps_time > 0

    def test_returns_true_while_within_wait_period(self) -> None:
        """While within the calculated wait window, should keep returning True."""
        jt = JobTracker()
        jt._max_pending_megapixelsteps = 5
        throttler = _make_throttler(job_tracker=jt)
        bd = self._make_bridge_data()

        # Add jobs above threshold
        for _ in range(10):
            job = Mock()
            job.payload.width = 1024
            job.payload.height = 1024
            job.payload.ddim_steps = 50
            job.payload.n_iter = 1
            job.payload.post_processing = []
            job.payload.loras = []
            job.payload.control_type = None
            job.payload.hires_fix = False
            jt.jobs_pending_inference.append(job)

        # First call sets the trigger
        throttler.should_wait_for_megapixelsteps(bd)
        # Second call should still wait (we just started)
        assert throttler.should_wait_for_megapixelsteps(bd) is True

    def test_returns_false_after_wait_time_elapses(self) -> None:
        """After the calculated wait period, waiting should end."""
        jt = JobTracker()
        jt._max_pending_megapixelsteps = 5
        throttler = _make_throttler(job_tracker=jt)
        bd = self._make_bridge_data()

        for _ in range(10):
            job = Mock()
            job.payload.width = 1024
            job.payload.height = 1024
            job.payload.ddim_steps = 50
            job.payload.n_iter = 1
            job.payload.post_processing = []
            job.payload.loras = []
            job.payload.control_type = None
            job.payload.hires_fix = False
            jt.jobs_pending_inference.append(job)

        # Trigger the wait
        throttler.should_wait_for_megapixelsteps(bd)

        # Fast forward past wait time by backdating the trigger
        jt._triggered_max_pending_megapixelsteps_time = time.time() - 1000

        assert throttler.should_wait_for_megapixelsteps(bd) is False
        assert jt._triggered_max_pending_megapixelsteps is False

    def test_high_performance_mode_reduces_wait(self) -> None:
        """High performance mode should give a much shorter wait time than normal."""
        jt = JobTracker()
        throttler = _make_throttler(job_tracker=jt)

        bd_normal = self._make_bridge_data(high_performance_mode=False, moderate_performance_mode=False)
        bd_high = self._make_bridge_data(high_performance_mode=True)

        normal_wait = throttler._calculate_megapixelstep_wait(bd_normal)
        high_wait = throttler._calculate_megapixelstep_wait(bd_high)

        # With 0 pending, both should be 0 or very small
        # Add jobs to make the difference meaningful
        for _ in range(5):
            job = Mock()
            job.payload.width = 1024
            job.payload.height = 1024
            job.payload.ddim_steps = 50
            job.payload.n_iter = 1
            job.payload.post_processing = []
            job.payload.loras = []
            job.payload.control_type = None
            job.payload.hires_fix = False
            jt.jobs_pending_inference.append(job)

        normal_wait = throttler._calculate_megapixelstep_wait(bd_normal)
        high_wait = throttler._calculate_megapixelstep_wait(bd_high)

        assert high_wait < normal_wait

    def test_moderate_performance_mode_reduces_wait(self) -> None:
        jt = JobTracker()
        throttler = _make_throttler(job_tracker=jt)

        for _ in range(5):
            job = Mock()
            job.payload.width = 1024
            job.payload.height = 1024
            job.payload.ddim_steps = 50
            job.payload.n_iter = 1
            job.payload.post_processing = []
            job.payload.loras = []
            job.payload.control_type = None
            job.payload.hires_fix = False
            jt.jobs_pending_inference.append(job)

        bd_normal = self._make_bridge_data(high_performance_mode=False, moderate_performance_mode=False)
        bd_moderate = self._make_bridge_data(moderate_performance_mode=True, high_performance_mode=False)

        normal_wait = throttler._calculate_megapixelstep_wait(bd_normal)
        moderate_wait = throttler._calculate_megapixelstep_wait(bd_moderate)

        assert moderate_wait < normal_wait

    def test_multi_thread_reduces_wait(self) -> None:
        jt = JobTracker()
        throttler = _make_throttler(job_tracker=jt)

        for _ in range(5):
            job = Mock()
            job.payload.width = 1024
            job.payload.height = 1024
            job.payload.ddim_steps = 50
            job.payload.n_iter = 1
            job.payload.post_processing = []
            job.payload.loras = []
            job.payload.control_type = None
            job.payload.hires_fix = False
            jt.jobs_pending_inference.append(job)

        bd_single = self._make_bridge_data(max_threads=1)
        bd_multi = self._make_bridge_data(max_threads=2)

        single_wait = throttler._calculate_megapixelstep_wait(bd_single)
        multi_wait = throttler._calculate_megapixelstep_wait(bd_multi)

        assert multi_wait < single_wait


class TestCalculateMegapixelstepWait:
    """Unit tests for the wait time calculation tiers."""

    def _make_throttler_with_pending(self, pending_mps: int) -> PopThrottler:
        """Make a throttler whose JobTracker reports the given pending MPS."""
        jt = JobTracker()
        throttler = _make_throttler(job_tracker=jt)
        # Patch get_pending_megapixelsteps to return exactly what we want
        jt.get_pending_megapixelsteps = Mock(return_value=pending_mps)  # type: ignore[method-assign]
        return throttler

    def test_low_pending_tier(self) -> None:
        """pending < 40 → factor 0.5."""
        throttler = self._make_throttler_with_pending(20)
        bd = make_mock_bridge_data(max_threads=1)
        assert throttler._calculate_megapixelstep_wait(bd) == pytest.approx(10.0)

    def test_medium_pending_tier(self) -> None:
        """40 <= pending < 80 → factor 0.7."""
        throttler = self._make_throttler_with_pending(60)
        bd = make_mock_bridge_data(max_threads=1)
        assert throttler._calculate_megapixelstep_wait(bd) == pytest.approx(42.0)

    def test_high_pending_tier(self) -> None:
        """pending >= 80 → factor 0.8."""
        throttler = self._make_throttler_with_pending(100)
        bd = make_mock_bridge_data(max_threads=1)
        assert throttler._calculate_megapixelstep_wait(bd) == pytest.approx(80.0)

    def test_zero_pending_gives_zero_wait(self) -> None:
        throttler = self._make_throttler_with_pending(0)
        bd = make_mock_bridge_data(max_threads=1)
        assert throttler._calculate_megapixelstep_wait(bd) == 0.0

    def test_high_perf_clamps_small_wait_to_one(self) -> None:
        """High performance mode: if wait < 35 after scaling, clamp to 1."""
        throttler = self._make_throttler_with_pending(20)
        bd = make_mock_bridge_data(high_performance_mode=True, max_threads=1)
        wait = throttler._calculate_megapixelstep_wait(bd)
        # 20 * 0.5 * 0.2 = 2.0 < 35 → clamped to 1
        assert wait == 1.0

    def test_moderate_perf_clamps_small_wait_to_one(self) -> None:
        """Moderate performance mode: if wait < 20 after scaling, clamp to 1."""
        throttler = self._make_throttler_with_pending(20)
        bd = make_mock_bridge_data(moderate_performance_mode=True, high_performance_mode=False, max_threads=1)
        wait = throttler._calculate_megapixelstep_wait(bd)
        # 20 * 0.5 * 0.4 = 4.0 < 20 → clamped to 1
        assert wait == 1.0

    def test_boundary_at_40_uses_medium_tier(self) -> None:
        """Exactly 40 pending → medium tier (0.7)."""
        throttler = self._make_throttler_with_pending(40)
        bd = make_mock_bridge_data(max_threads=1)
        assert throttler._calculate_megapixelstep_wait(bd) == pytest.approx(28.0)

    def test_boundary_at_80_uses_high_tier(self) -> None:
        """Exactly 80 pending → high tier (0.8)."""
        throttler = self._make_throttler_with_pending(80)
        bd = make_mock_bridge_data(max_threads=1)
        assert throttler._calculate_megapixelstep_wait(bd) == pytest.approx(64.0)


class TestConstantsIntegrity:
    """Guard against accidental changes to sentinel values."""

    def test_consecutive_failed_jobs_wait_is_positive(self) -> None:
        assert CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS > 0

    def test_consecutive_failed_jobs_wait_value(self) -> None:
        assert CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS == 180
