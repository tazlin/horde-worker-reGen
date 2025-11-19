"""Example event listeners demonstrating event system usage.

These listeners can be used for testing, debugging, or as templates
for creating custom listeners.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from loguru import logger

from horde_worker_regen.events.event_listener import EventListener, FilteredEventListener
from horde_worker_regen.events.event_types import (
    EventPriority,
    JobCompletedEvent,
    JobFaultedEvent,
    JobPoppedEvent,
    KudosEarnedEvent,
    ProcessHeartbeatEvent,
    WorkerEvent,
)

if TYPE_CHECKING:
    pass


class LoggingEventListener(EventListener):
    """Simple event listener that logs all events to the console.

    Useful for debugging and understanding event flow.

    Example:
        dispatcher = EventDispatcher()
        listener = LoggingEventListener()
        dispatcher.subscribe_to_all(listener)
    """

    def __init__(self, log_level: str = "DEBUG", include_heartbeats: bool = False) -> None:
        """Initialize the logging listener.

        Args:
            log_level: The log level to use (DEBUG, INFO, WARNING, ERROR).
            include_heartbeats: If False, skip logging heartbeat events (they're very frequent).
        """
        self.log_level = log_level.upper()
        self.include_heartbeats = include_heartbeats

    def on_event(self, event: WorkerEvent) -> None:
        """Log the event.

        Args:
            event: The event to log.
        """
        # Skip heartbeats if configured to do so
        if not self.include_heartbeats and isinstance(event, ProcessHeartbeatEvent):
            return

        event_name = event.__class__.__name__
        timestamp = event.timestamp

        # Format the event details
        details = self._format_event_details(event)

        log_message = f"[Event] {event_name} @ {timestamp:.2f}: {details}"

        # Log at appropriate level
        if self.log_level == "DEBUG":
            logger.debug(log_message)
        elif self.log_level == "INFO":
            logger.info(log_message)
        elif self.log_level == "WARNING":
            logger.warning(log_message)
        elif self.log_level == "ERROR":
            logger.error(log_message)

    def _format_event_details(self, event: WorkerEvent) -> str:
        """Format event details for logging.

        Args:
            event: The event to format.

        Returns:
            A string representation of the event details.
        """
        if isinstance(event, JobPoppedEvent):
            return f"job_id={str(event.job_id)[:8]}, model={event.model_name}, {event.width}x{event.height}"
        elif isinstance(event, JobCompletedEvent):
            return f"job_id={str(event.job_id)[:8]}, kudos={event.kudos_earned:.2f}, time={event.generation_time_seconds:.1f}s"
        elif isinstance(event, JobFaultedEvent):
            return f"job_id={str(event.job_id)[:8]}, fault={event.fault_type}"
        elif isinstance(event, KudosEarnedEvent):
            return f"amount={event.amount:.2f}, total={event.cumulative_total:.2f}"
        else:
            # Generic format: show all non-private attributes
            attrs = {k: v for k, v in event.__dict__.items() if not k.startswith("_")}
            return ", ".join(f"{k}={v}" for k, v in list(attrs.items())[:5])  # Limit to first 5


class EventStatisticsListener(EventListener):
    """Event listener that tracks statistics about events.

    Collects counts of different event types and can produce summary reports.

    Example:
        dispatcher = EventDispatcher()
        stats = EventStatisticsListener()
        dispatcher.subscribe_to_all(stats)

        # Later, get statistics
        print(stats.get_summary())
    """

    def __init__(self) -> None:
        """Initialize the statistics listener."""
        self.event_counts: dict[str, int] = defaultdict(int)
        """Count of each event type."""

        self.total_events = 0
        """Total number of events received."""

        self.total_kudos = 0.0
        """Total kudos earned (from KudosEarnedEvent)."""

        self.jobs_completed = 0
        """Number of jobs completed (from JobCompletedEvent)."""

        self.jobs_faulted = 0
        """Number of jobs faulted (from JobFaultedEvent)."""

        self.priority_counts: dict[EventPriority, int] = defaultdict(int)
        """Count of events by priority."""

    def on_event(self, event: WorkerEvent) -> None:
        """Track statistics for the event.

        Args:
            event: The event to track.
        """
        event_type_name = event.__class__.__name__
        self.event_counts[event_type_name] += 1
        self.total_events += 1
        self.priority_counts[event.priority] += 1

        # Track specific metrics
        if isinstance(event, KudosEarnedEvent):
            self.total_kudos = event.cumulative_total  # Use cumulative from event
        elif isinstance(event, JobCompletedEvent):
            self.jobs_completed += 1
        elif isinstance(event, JobFaultedEvent):
            self.jobs_faulted += 1

    def get_summary(self) -> dict:
        """Get a summary of event statistics.

        Returns:
            A dictionary containing event statistics.
        """
        return {
            "total_events": self.total_events,
            "event_counts": dict(self.event_counts),
            "priority_counts": {k.name: v for k, v in self.priority_counts.items()},
            "total_kudos": self.total_kudos,
            "jobs_completed": self.jobs_completed,
            "jobs_faulted": self.jobs_faulted,
        }

    def print_summary(self) -> None:
        """Print a formatted summary of statistics."""
        summary = self.get_summary()

        logger.info("=" * 60)
        logger.info("Event Statistics Summary")
        logger.info("=" * 60)
        logger.info(f"Total Events: {summary['total_events']}")
        logger.info(f"Total Kudos: {summary['total_kudos']:.2f}")
        logger.info(f"Jobs Completed: {summary['jobs_completed']}")
        logger.info(f"Jobs Faulted: {summary['jobs_faulted']}")
        logger.info("-" * 60)
        logger.info("Event Counts by Type:")
        for event_type, count in sorted(summary["event_counts"].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {event_type}: {count}")
        logger.info("-" * 60)
        logger.info("Event Counts by Priority:")
        for priority, count in summary["priority_counts"].items():
            logger.info(f"  {priority}: {count}")
        logger.info("=" * 60)

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.event_counts.clear()
        self.priority_counts.clear()
        self.total_events = 0
        self.total_kudos = 0.0
        self.jobs_completed = 0
        self.jobs_faulted = 0


class HighPriorityEventListener(FilteredEventListener):
    """Example listener that only handles high priority events.

    Demonstrates the FilteredEventListener base class.

    Example:
        dispatcher = EventDispatcher()
        listener = HighPriorityEventListener()
        dispatcher.subscribe_to_all(listener)
    """

    def should_handle_event(self, event: WorkerEvent) -> bool:
        """Only handle high and critical priority events.

        Args:
            event: The event to check.

        Returns:
            True if the event is high or critical priority.
        """
        return event.priority in (EventPriority.HIGH, EventPriority.CRITICAL)

    def handle_event(self, event: WorkerEvent) -> None:
        """Handle a high priority event.

        Args:
            event: The high priority event.
        """
        logger.warning(f"[HIGH PRIORITY] {event.__class__.__name__}: {event}")
