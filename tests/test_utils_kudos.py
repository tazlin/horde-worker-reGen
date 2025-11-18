"""Tests for kudos utility functions."""

from horde_worker_regen.utils.kudos_utils import generate_kudos_info_string


def test_generate_kudos_info_string_short_session() -> None:
    """Test kudos info string generation for a session under 1 hour."""
    result = generate_kudos_info_string(
        kudos_generated_this_session=100.0,
        time_since_session_start=1800.0,  # 30 minutes
        kudos_per_hour_session=200.0,
        kudos_total_past_hour=100.0,
        active_kudos_per_hour=250.0,
        time_spent_no_jobs_available=0.0,
        max_time_spent_no_jobs_available=300.0,
    )

    # Should show minutes for short sessions
    assert "30.00 minutes" in result
    assert "100.00" in result  # kudos generated
    assert "(extrapolated)" in result  # should be extrapolated for short sessions
    assert "200.00" in result  # kudos per hour


def test_generate_kudos_info_string_long_session() -> None:
    """Test kudos info string generation for a session over 1 hour."""
    result = generate_kudos_info_string(
        kudos_generated_this_session=500.0,
        time_since_session_start=7200.0,  # 2 hours
        kudos_per_hour_session=250.0,
        kudos_total_past_hour=300.0,
        active_kudos_per_hour=280.0,
        time_spent_no_jobs_available=0.0,
        max_time_spent_no_jobs_available=300.0,
    )

    # Should show hours for long sessions
    assert "2.00 hours" in result
    assert "500.00" in result  # kudos generated
    assert "(actual)" in result  # should be actual for long sessions
    assert "250.00" in result  # kudos per hour


def test_generate_kudos_info_string_with_downtime() -> None:
    """Test kudos info string generation when there's significant downtime."""
    result = generate_kudos_info_string(
        kudos_generated_this_session=500.0,
        time_since_session_start=7200.0,  # 2 hours
        kudos_per_hour_session=250.0,
        kudos_total_past_hour=300.0,
        active_kudos_per_hour=350.0,
        time_spent_no_jobs_available=400.0,  # More than max
        max_time_spent_no_jobs_available=300.0,
    )

    # Should include active kudos per hour when downtime exceeds max
    assert "Active (jobs available):" in result
    assert "350.00 kudos/hr" in result


def test_generate_kudos_info_string_no_downtime() -> None:
    """Test kudos info string generation with minimal downtime."""
    result = generate_kudos_info_string(
        kudos_generated_this_session=500.0,
        time_since_session_start=7200.0,
        kudos_per_hour_session=250.0,
        kudos_total_past_hour=300.0,
        active_kudos_per_hour=260.0,
        time_spent_no_jobs_available=100.0,  # Less than max
        max_time_spent_no_jobs_available=300.0,
    )

    # Should NOT include active kudos per hour when downtime is below max
    assert "Active (jobs available):" not in result


def test_generate_kudos_info_string_exactly_one_hour() -> None:
    """Test kudos info string generation at exactly 1 hour."""
    result = generate_kudos_info_string(
        kudos_generated_this_session=250.0,
        time_since_session_start=3600.0,  # Exactly 1 hour
        kudos_per_hour_session=250.0,
        kudos_total_past_hour=250.0,
        active_kudos_per_hour=250.0,
        time_spent_no_jobs_available=0.0,
        max_time_spent_no_jobs_available=300.0,
    )

    # At exactly 3600 seconds, should NOT be > 3600, so should still show minutes
    assert "60.00 minutes" in result
    assert "(extrapolated)" in result


def test_generate_kudos_info_string_just_over_one_hour() -> None:
    """Test kudos info string generation just over 1 hour."""
    result = generate_kudos_info_string(
        kudos_generated_this_session=260.0,
        time_since_session_start=3601.0,  # Just over 1 hour
        kudos_per_hour_session=260.0,
        kudos_total_past_hour=260.0,
        active_kudos_per_hour=260.0,
        time_spent_no_jobs_available=0.0,
        max_time_spent_no_jobs_available=300.0,
    )

    # Just over 1 hour should show hours and actual
    assert "hours" in result
    assert "(actual)" in result


def test_generate_kudos_info_string_zero_kudos() -> None:
    """Test kudos info string generation with zero kudos."""
    result = generate_kudos_info_string(
        kudos_generated_this_session=0.0,
        time_since_session_start=1800.0,
        kudos_per_hour_session=0.0,
        kudos_total_past_hour=0.0,
        active_kudos_per_hour=0.0,
        time_spent_no_jobs_available=0.0,
        max_time_spent_no_jobs_available=300.0,
    )

    # Should still generate a valid string
    assert "0.00" in result
    assert "minutes" in result


def test_generate_kudos_info_string_format() -> None:
    """Test that the kudos info string uses pipe separator."""
    result = generate_kudos_info_string(
        kudos_generated_this_session=100.0,
        time_since_session_start=1800.0,
        kudos_per_hour_session=200.0,
        kudos_total_past_hour=100.0,
        active_kudos_per_hour=250.0,
        time_spent_no_jobs_available=0.0,
        max_time_spent_no_jobs_available=300.0,
    )

    # Should use pipe separator
    assert "|" in result
    # Should have at least 2 parts
    parts = result.split("|")
    assert len(parts) >= 2


def test_generate_kudos_info_string_large_values() -> None:
    """Test kudos info string generation with large values."""
    result = generate_kudos_info_string(
        kudos_generated_this_session=1_234_567.89,
        time_since_session_start=36000.0,  # 10 hours
        kudos_per_hour_session=123_456.78,
        kudos_total_past_hour=120_000.0,
        active_kudos_per_hour=125_000.0,
        time_spent_no_jobs_available=0.0,
        max_time_spent_no_jobs_available=300.0,
    )

    # Should include comma formatting for large numbers
    assert "1,234,567.89" in result
    assert "123,456.78" in result
