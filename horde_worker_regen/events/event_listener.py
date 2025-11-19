"""Event listener base class for subscribing to worker events.

Event listeners can subscribe to specific event types and receive notifications
when those events are emitted. This enables decoupled observation of worker state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from horde_worker_regen.events.event_types import WorkerEvent


class EventListener(ABC):
    """Abstract base class for event listeners.

    Subclasses should implement the `on_event` method to handle events.
    Listeners can subscribe to specific event types using the EventDispatcher.
    """

    @abstractmethod
    def on_event(self, event: WorkerEvent) -> None:
        """Handle an event.

        This method is called when an event of a subscribed type is emitted.
        Implementations should handle the event quickly to avoid blocking
        the event dispatcher. For long-running operations, consider using
        a queue or async processing.

        Args:
            event: The event that was emitted.
        """
        pass

    def on_error(self, event: WorkerEvent, exception: Exception) -> None:
        """Handle an error that occurred while processing an event.

        This method is called if `on_event` raises an exception. The default
        implementation logs the error. Subclasses can override this to provide
        custom error handling.

        Args:
            event: The event that was being processed when the error occurred.
            exception: The exception that was raised.
        """
        from loguru import logger

        logger.error(
            f"Error in {self.__class__.__name__}.on_event for {event.__class__.__name__}: {exception}",
        )


class FilteredEventListener(EventListener):
    """Base class for listeners that filter events by type.

    This is a convenience class that provides filtering functionality.
    Subclasses should override `should_handle_event` and `handle_event`.
    """

    def on_event(self, event: WorkerEvent) -> None:
        """Handle an event if it passes the filter.

        Args:
            event: The event that was emitted.
        """
        if self.should_handle_event(event):
            self.handle_event(event)

    def should_handle_event(self, event: WorkerEvent) -> bool:
        """Determine if this event should be handled.

        The default implementation returns True for all events. Subclasses
        can override this to filter events based on type, priority, or other criteria.

        Args:
            event: The event to check.

        Returns:
            True if the event should be handled, False otherwise.
        """
        return True

    @abstractmethod
    def handle_event(self, event: WorkerEvent) -> None:
        """Handle a filtered event.

        This method is only called for events that pass the filter.

        Args:
            event: The event to handle.
        """
        pass


class CallbackEventListener(EventListener):
    """Simple event listener that calls a callback function.

    This is useful for quick event subscriptions without creating a full class.

    Example:
        def my_callback(event):
            print(f"Received event: {event}")

        listener = CallbackEventListener(my_callback)
        dispatcher.subscribe(JobCompletedEvent, listener)
    """

    def __init__(self, callback: callable) -> None:
        """Initialize the callback listener.

        Args:
            callback: A callable that accepts a WorkerEvent as its only argument.
        """
        self.callback = callback

    def on_event(self, event: WorkerEvent) -> None:
        """Call the callback with the event.

        Args:
            event: The event to pass to the callback.
        """
        self.callback(event)
