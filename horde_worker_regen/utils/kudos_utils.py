"""Kudos calculation and formatting utility functions."""

from __future__ import annotations


def generate_kudos_info_string(
    kudos_generated_this_session: float,
    time_since_session_start: float,
    kudos_per_hour_session: float,
    kudos_total_past_hour: float,
    active_kudos_per_hour: float,
    time_spent_no_jobs_available: float,
    max_time_spent_no_jobs_available: float,
) -> str:
    """Generate a string with information about the kudos generated in the current session.

    Args:
        kudos_generated_this_session: The kudos generated in the current session.
        time_since_session_start: The time since the session started.
        kudos_per_hour_session: The kudos per hour generated in the current session.
        kudos_total_past_hour: The total kudos generated in the past hour.
        active_kudos_per_hour: The kudos per hour generated while active (jobs available).
        time_spent_no_jobs_available: The time spent with no jobs available.
        max_time_spent_no_jobs_available: The maximum time allowed with no jobs available.

    Returns:
        A string with information about the kudos generated in the current session.
    """
    kudos_info_string_elements: list[str] = []
    if time_since_session_start <= 3600:
        kudos_info_string_elements = [
            f"Total Session Kudos: {kudos_generated_this_session:,.2f} over "
            f"{time_since_session_start / 60:.2f} minutes",
        ]
    else:
        kudos_info_string_elements = [
            f"Total Session Kudos: {kudos_generated_this_session:,.2f} over "
            f"{time_since_session_start / 3600:.2f} hours",
        ]

    if time_since_session_start > 3600:
        kudos_info_string_elements.append(
            f"Session: {kudos_per_hour_session:,.2f} (actual) kudos/hr",
        )
    else:
        kudos_info_string_elements.append(
            f"Session: {kudos_per_hour_session:,.2f} (extrapolated) kudos/hr",
        )

    if time_spent_no_jobs_available > max_time_spent_no_jobs_available:
        kudos_info_string_elements.append(
            f"Active (jobs available): {active_kudos_per_hour:,.2f} kudos/hr",
        )

    return " | ".join(kudos_info_string_elements)
