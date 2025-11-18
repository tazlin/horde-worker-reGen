"""Tests for kudos calculator."""

import time
from collections import deque

import pytest

from horde_worker_regen.utils.kudos_calculator import KudosCalculator


def test_calculate_kudos_per_hour_normal():
    """Test kudos per hour calculation with normal values."""
    kudos_generated = 100.0
    time_elapsed = 1800  # 30 minutes

    result = KudosCalculator.calculate_kudos_per_hour(kudos_generated, time_elapsed)

    # 100 kudos in 30 minutes = 200 kudos per hour
    assert result == 200.0


def test_calculate_kudos_per_hour_zero_time():
    """Test kudos per hour calculation with zero time elapsed."""
    kudos_generated = 100.0
    time_elapsed = 0

    result = KudosCalculator.calculate_kudos_per_hour(kudos_generated, time_elapsed)

    assert result == 0.0


def test_calculate_kudos_per_hour_one_hour():
    """Test kudos per hour calculation with exactly one hour."""
    kudos_generated = 150.0
    time_elapsed = 3600  # 1 hour

    result = KudosCalculator.calculate_kudos_per_hour(kudos_generated, time_elapsed)

    assert result == 150.0


def test_calculate_active_kudos_per_hour_normal():
    """Test active kudos per hour calculation with normal values."""
    kudos_generated = 100.0
    time_elapsed = 1800  # 30 minutes
    time_spent_idle = 300  # 5 minutes idle

    result = KudosCalculator.calculate_active_kudos_per_hour(
        kudos_generated,
        time_elapsed,
        time_spent_idle,
    )

    # 100 kudos in 25 minutes active = 240 kudos per hour
    assert result == 240.0


def test_calculate_active_kudos_per_hour_no_idle_time():
    """Test active kudos per hour calculation with no idle time."""
    kudos_generated = 100.0
    time_elapsed = 1800  # 30 minutes
    time_spent_idle = 0

    result = KudosCalculator.calculate_active_kudos_per_hour(
        kudos_generated,
        time_elapsed,
        time_spent_idle,
    )

    # Same as regular kudos per hour
    assert result == 200.0


def test_calculate_active_kudos_per_hour_all_idle():
    """Test active kudos per hour calculation when all time is idle."""
    kudos_generated = 100.0
    time_elapsed = 1800  # 30 minutes
    time_spent_idle = 1800  # All idle

    result = KudosCalculator.calculate_active_kudos_per_hour(
        kudos_generated,
        time_elapsed,
        time_spent_idle,
    )

    # No active time = 0
    assert result == 0.0


def test_calculate_active_kudos_per_hour_more_idle_than_elapsed():
    """Test active kudos per hour calculation when idle time exceeds elapsed time."""
    kudos_generated = 100.0
    time_elapsed = 1800  # 30 minutes
    time_spent_idle = 2000  # More than elapsed (edge case)

    result = KudosCalculator.calculate_active_kudos_per_hour(
        kudos_generated,
        time_elapsed,
        time_spent_idle,
    )

    # Negative active time = 0
    assert result == 0.0


def test_calculate_kudos_totals_past_hour_all_recent():
    """Test calculating kudos totals when all events are within past hour."""
    current_time = time.time()
    kudos_events = deque(
        [
            (current_time - 1800, 50.0),  # 30 minutes ago
            (current_time - 900, 30.0),  # 15 minutes ago
            (current_time - 300, 20.0),  # 5 minutes ago
        ],
        maxlen=100,
    )

    total, cleaned = KudosCalculator.calculate_kudos_totals_past_hour(kudos_events)

    assert total == 100.0
    assert len(cleaned) == 3


def test_calculate_kudos_totals_past_hour_some_old():
    """Test calculating kudos totals when some events are older than 1 hour."""
    current_time = time.time()
    kudos_events = deque(
        [
            (current_time - 7200, 100.0),  # 2 hours ago (should be excluded)
            (current_time - 3700, 75.0),  # 1 hour 2 minutes ago (should be excluded)
            (current_time - 1800, 50.0),  # 30 minutes ago
            (current_time - 900, 30.0),  # 15 minutes ago
        ],
        maxlen=100,
    )

    total, cleaned = KudosCalculator.calculate_kudos_totals_past_hour(kudos_events)

    assert total == 80.0  # Only the last two events
    assert len(cleaned) == 2


def test_calculate_kudos_totals_past_hour_all_old():
    """Test calculating kudos totals when all events are older than 1 hour."""
    current_time = time.time()
    kudos_events = deque(
        [
            (current_time - 7200, 100.0),  # 2 hours ago
            (current_time - 5400, 75.0),  # 1.5 hours ago
        ],
        maxlen=100,
    )

    total, cleaned = KudosCalculator.calculate_kudos_totals_past_hour(kudos_events)

    assert total == 0.0
    assert len(cleaned) == 0


def test_calculate_kudos_totals_past_hour_empty():
    """Test calculating kudos totals with empty events."""
    kudos_events = deque([], maxlen=100)

    total, cleaned = KudosCalculator.calculate_kudos_totals_past_hour(kudos_events)

    assert total == 0.0
    assert len(cleaned) == 0


def test_calculate_kudos_totals_past_hour_maxlen_preserved():
    """Test that maxlen is preserved in cleaned deque."""
    current_time = time.time()
    original_maxlen = 50
    kudos_events = deque(
        [
            (current_time - 1800, 50.0),
            (current_time - 900, 30.0),
        ],
        maxlen=original_maxlen,
    )

    total, cleaned = KudosCalculator.calculate_kudos_totals_past_hour(kudos_events)

    assert cleaned.maxlen == original_maxlen


def test_calculate_all_metrics_integration():
    """Test calculate_all_metrics returns all expected values."""
    current_time = time.time()
    kudos_generated_this_session = 200.0
    session_start_time = current_time - 3600  # Started 1 hour ago
    time_spent_no_jobs_available = 600  # 10 minutes idle
    kudos_events = deque(
        [
            (current_time - 1800, 100.0),  # 30 minutes ago
            (current_time - 900, 100.0),  # 15 minutes ago
        ],
        maxlen=100,
    )

    (
        time_since_session_start,
        kudos_per_hour_session,
        kudos_total_past_hour,
        active_kudos_per_hour,
        cleaned_events,
    ) = KudosCalculator.calculate_all_metrics(
        kudos_generated_this_session,
        session_start_time,
        time_spent_no_jobs_available,
        kudos_events,
    )

    # Check time_since_session_start is approximately 1 hour (allow small delta for test execution time)
    assert abs(time_since_session_start - 3600) < 1

    # Check kudos_per_hour_session: 200 kudos in 1 hour = 200/hr
    assert abs(kudos_per_hour_session - 200.0) < 1

    # Check kudos_total_past_hour: both events are within the hour
    assert kudos_total_past_hour == 200.0

    # Check active_kudos_per_hour: 200 kudos in 50 minutes active = 240/hr
    expected_active = 200.0 / (3600 - 600) * 3600
    assert abs(active_kudos_per_hour - expected_active) < 1

    # Check cleaned_events has both events
    assert len(cleaned_events) == 2


def test_calculate_all_metrics_with_old_events():
    """Test calculate_all_metrics properly cleans old events."""
    current_time = time.time()
    kudos_generated_this_session = 300.0
    session_start_time = current_time - 7200  # Started 2 hours ago
    time_spent_no_jobs_available = 1200  # 20 minutes idle
    kudos_events = deque(
        [
            (current_time - 7000, 50.0),  # ~2 hours ago (should be excluded)
            (current_time - 5000, 50.0),  # ~1.4 hours ago (should be excluded)
            (current_time - 1800, 100.0),  # 30 minutes ago
            (current_time - 900, 100.0),  # 15 minutes ago
        ],
        maxlen=100,
    )

    (
        time_since_session_start,
        kudos_per_hour_session,
        kudos_total_past_hour,
        active_kudos_per_hour,
        cleaned_events,
    ) = KudosCalculator.calculate_all_metrics(
        kudos_generated_this_session,
        session_start_time,
        time_spent_no_jobs_available,
        kudos_events,
    )

    # Check kudos_total_past_hour only includes recent events
    assert kudos_total_past_hour == 200.0

    # Check cleaned_events only has recent events
    assert len(cleaned_events) == 2


def test_calculate_all_metrics_no_idle_time():
    """Test calculate_all_metrics with no idle time."""
    current_time = time.time()
    kudos_generated_this_session = 100.0
    session_start_time = current_time - 1800  # Started 30 minutes ago
    time_spent_no_jobs_available = 0  # No idle time
    kudos_events = deque(
        [
            (current_time - 900, 100.0),  # 15 minutes ago
        ],
        maxlen=100,
    )

    (
        time_since_session_start,
        kudos_per_hour_session,
        kudos_total_past_hour,
        active_kudos_per_hour,
        cleaned_events,
    ) = KudosCalculator.calculate_all_metrics(
        kudos_generated_this_session,
        session_start_time,
        time_spent_no_jobs_available,
        kudos_events,
    )

    # When there's no idle time, active and regular kudos per hour should be the same
    assert abs(kudos_per_hour_session - active_kudos_per_hour) < 0.01
