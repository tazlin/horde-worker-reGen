"""Kudos calculation and analysis."""

from __future__ import annotations

import time
from collections import deque


class KudosCalculator:
    """Handles kudos calculations and metrics."""

    @staticmethod
    def calculate_kudos_per_hour(kudos_generated: float, time_elapsed: float) -> float:
        """Calculate kudos per hour from total kudos and elapsed time.

        Args:
            kudos_generated: Total kudos generated.
            time_elapsed: Time elapsed in seconds.

        Returns:
            Kudos per hour rate.
        """
        if time_elapsed == 0:
            return 0.0
        return kudos_generated / time_elapsed * 3600

    @staticmethod
    def calculate_active_kudos_per_hour(
        kudos_generated: float,
        time_elapsed: float,
        time_spent_idle: float,
    ) -> float:
        """Calculate kudos per hour excluding idle time.

        Args:
            kudos_generated: Total kudos generated.
            time_elapsed: Total time elapsed in seconds.
            time_spent_idle: Time spent without jobs available in seconds.

        Returns:
            Kudos per hour rate when actively processing jobs.
        """
        active_time = time_elapsed - time_spent_idle
        if active_time <= 0:
            return 0.0
        return kudos_generated / active_time * 3600

    @staticmethod
    def calculate_kudos_totals_past_hour(
        kudos_events: deque[tuple[float, float]],
    ) -> tuple[float, deque[tuple[float, float]]]:
        """Calculate the total kudos generated in the past hour and clean up old events.

        Args:
            kudos_events: Deque of (timestamp, kudos) tuples.

        Returns:
            Tuple of (total kudos in past hour, cleaned kudos events deque).
        """
        kudos_total_past_hour = 0.0
        num_events_found = 0
        current_time = time.time()

        for event_time, kudos in reversed(kudos_events):
            if current_time - event_time > 3600:
                break

            num_events_found += 1
            kudos_total_past_hour += kudos

        # Remove events older than 1 hour
        elements_to_remove = len(kudos_events) - num_events_found
        if elements_to_remove > 0:
            # Create new deque with only recent events
            if num_events_found > 0:
                cleaned_events = deque(list(kudos_events)[-num_events_found:], maxlen=kudos_events.maxlen)
            else:
                # All events are old, return empty deque
                cleaned_events = deque([], maxlen=kudos_events.maxlen)
        else:
            cleaned_events = kudos_events

        return kudos_total_past_hour, cleaned_events

    @staticmethod
    def calculate_all_metrics(
        kudos_generated_this_session: float,
        session_start_time: float,
        time_spent_no_jobs_available: float,
        kudos_events: deque[tuple[float, float]],
    ) -> tuple[float, float, float, float, deque[tuple[float, float]]]:
        """Calculate all kudos metrics.

        Args:
            kudos_generated_this_session: Total kudos generated in this session.
            session_start_time: Session start timestamp.
            time_spent_no_jobs_available: Time spent idle in seconds.
            kudos_events: Deque of (timestamp, kudos) tuples.

        Returns:
            Tuple of:
            - time_since_session_start (seconds)
            - kudos_per_hour_session
            - kudos_total_past_hour
            - active_kudos_per_hour
            - cleaned kudos_events deque
        """
        time_since_session_start = time.time() - session_start_time

        kudos_per_hour_session = KudosCalculator.calculate_kudos_per_hour(
            kudos_generated_this_session,
            time_since_session_start,
        )

        active_kudos_per_hour = KudosCalculator.calculate_active_kudos_per_hour(
            kudos_generated_this_session,
            time_since_session_start,
            time_spent_no_jobs_available,
        )

        kudos_total_past_hour, cleaned_events = KudosCalculator.calculate_kudos_totals_past_hour(
            kudos_events,
        )

        return (
            time_since_session_start,
            kudos_per_hour_session,
            kudos_total_past_hour,
            active_kudos_per_hour,
            cleaned_events,
        )
