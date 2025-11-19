"""Event dispatcher for managing event subscriptions and emissions.

The EventDispatcher is the central hub for the event system. It manages
subscriptions from event listeners and emits events to all subscribed listeners.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Type

from loguru import logger

if TYPE_CHECKING:
    from horde_worker_regen.events.event_listener import EventListener
    from horde_worker_regen.events.event_types import WorkerEvent


class EventDispatcher:
    """Central event dispatcher for the worker event system.

    The dispatcher maintains subscriptions from event listeners and emits
    events to all subscribed listeners. It is thread-safe and can be used
    from multiple threads or async contexts.

    Example:
        # Create dispatcher
        dispatcher = EventDispatcher()

        # Subscribe to events
        dispatcher.subscribe(JobCompletedEvent, my_listener)

        # Emit events
        event = JobCompletedEvent(job_id="123", kudos_earned=10.5)
        dispatcher.emit(event)

        # Unsubscribe
        dispatcher.unsubscribe(JobCompletedEvent, my_listener)
    """

    def __init__(self, enable_logging: bool = False) -> None:
        """Initialize the event dispatcher.

        Args:
            enable_logging: If True, log all event emissions and subscriptions.
                           Useful for debugging but can be verbose.
        """
        self._subscriptions: dict[Type[WorkerEvent], list[EventListener]] = defaultdict(list)
        """Mapping of event types to lists of subscribed listeners."""

        self._global_listeners: list[EventListener] = []
        """Listeners that receive all events regardless of type."""

        self._lock = threading.RLock()
        """Thread lock for thread-safe subscription management."""

        self._enable_logging = enable_logging
        """Whether to log event operations."""

        self._event_count = 0
        """Total number of events emitted (for statistics)."""

        self._error_count = 0
        """Total number of errors during event handling (for statistics)."""

    def subscribe(self, event_type: Type[WorkerEvent], listener: EventListener) -> None:
        """Subscribe a listener to a specific event type.

        The listener's `on_event` method will be called whenever an event
        of the specified type is emitted.

        Args:
            event_type: The type of event to subscribe to (e.g., JobCompletedEvent).
            listener: The listener that should receive events of this type.
        """
        with self._lock:
            if listener not in self._subscriptions[event_type]:
                self._subscriptions[event_type].append(listener)
                if self._enable_logging:
                    logger.debug(
                        f"Subscribed {listener.__class__.__name__} to {event_type.__name__}",
                    )
            else:
                logger.warning(
                    f"Listener {listener.__class__.__name__} is already subscribed to {event_type.__name__}",
                )

    def subscribe_to_all(self, listener: EventListener) -> None:
        """Subscribe a listener to all event types.

        The listener will receive every event emitted, regardless of type.
        This is useful for logging or monitoring all worker activity.

        Args:
            listener: The listener that should receive all events.
        """
        with self._lock:
            if listener not in self._global_listeners:
                self._global_listeners.append(listener)
                if self._enable_logging:
                    logger.debug(
                        f"Subscribed {listener.__class__.__name__} to all events",
                    )
            else:
                logger.warning(
                    f"Listener {listener.__class__.__name__} is already subscribed to all events",
                )

    def unsubscribe(self, event_type: Type[WorkerEvent], listener: EventListener) -> bool:
        """Unsubscribe a listener from a specific event type.

        Args:
            event_type: The event type to unsubscribe from.
            listener: The listener to unsubscribe.

        Returns:
            True if the listener was successfully unsubscribed, False if it wasn't subscribed.
        """
        with self._lock:
            if listener in self._subscriptions[event_type]:
                self._subscriptions[event_type].remove(listener)
                if self._enable_logging:
                    logger.debug(
                        f"Unsubscribed {listener.__class__.__name__} from {event_type.__name__}",
                    )
                return True
            return False

    def unsubscribe_from_all(self, listener: EventListener) -> bool:
        """Unsubscribe a listener from all events.

        Args:
            listener: The listener to unsubscribe.

        Returns:
            True if the listener was successfully unsubscribed, False if it wasn't subscribed.
        """
        with self._lock:
            if listener in self._global_listeners:
                self._global_listeners.remove(listener)
                if self._enable_logging:
                    logger.debug(
                        f"Unsubscribed {listener.__class__.__name__} from all events",
                    )
                return True
            return False

    def unsubscribe_all_for_listener(self, listener: EventListener) -> int:
        """Unsubscribe a listener from all event types it's subscribed to.

        This is useful for cleanup when a listener is no longer needed.

        Args:
            listener: The listener to unsubscribe from everything.

        Returns:
            The number of subscriptions that were removed.
        """
        removed_count = 0

        with self._lock:
            # Remove from specific event type subscriptions
            for event_type, listeners in list(self._subscriptions.items()):
                if listener in listeners:
                    listeners.remove(listener)
                    removed_count += 1

            # Remove from global listeners
            if listener in self._global_listeners:
                self._global_listeners.remove(listener)
                removed_count += 1

        if self._enable_logging and removed_count > 0:
            logger.debug(
                f"Unsubscribed {listener.__class__.__name__} from {removed_count} subscription(s)",
            )

        return removed_count

    def emit(self, event: WorkerEvent) -> None:
        """Emit an event to all subscribed listeners.

        This method calls the `on_event` method of all listeners subscribed
        to this event type, as well as all global listeners. If a listener
        raises an exception, it is caught and the listener's `on_error` method
        is called.

        Args:
            event: The event to emit.
        """
        event_type = type(event)
        self._event_count += 1

        if self._enable_logging:
            logger.debug(f"Emitting event: {event_type.__name__}")

        # Get all listeners that should receive this event
        # We copy the lists to avoid issues if listeners modify subscriptions
        with self._lock:
            specific_listeners = list(self._subscriptions.get(event_type, []))
            global_listeners = list(self._global_listeners)

        all_listeners = specific_listeners + global_listeners

        # Deliver event to all listeners
        for listener in all_listeners:
            try:
                listener.on_event(event)
            except Exception as e:
                self._error_count += 1
                try:
                    listener.on_error(event, e)
                except Exception as error_handler_exception:
                    logger.error(
                        f"Error in {listener.__class__.__name__}.on_error: {error_handler_exception}",
                    )

    def get_listener_count(self, event_type: Type[WorkerEvent] | None = None) -> int:
        """Get the number of listeners subscribed to a specific event type.

        Args:
            event_type: The event type to check. If None, returns the total number
                       of unique listeners across all subscriptions.

        Returns:
            The number of listeners.
        """
        with self._lock:
            if event_type is None:
                # Count unique listeners across all subscriptions
                all_listeners = set(self._global_listeners)
                for listeners in self._subscriptions.values():
                    all_listeners.update(listeners)
                return len(all_listeners)
            else:
                # Count listeners for specific event type (including global listeners)
                specific_count = len(self._subscriptions.get(event_type, []))
                global_count = len(self._global_listeners)
                return specific_count + global_count

    def get_statistics(self) -> dict[str, int]:
        """Get statistics about event dispatcher usage.

        Returns:
            A dictionary containing:
                - total_events_emitted: Total number of events emitted
                - total_errors: Total number of errors during event handling
                - total_subscriptions: Total number of active subscriptions
                - global_listeners: Number of global listeners
        """
        with self._lock:
            total_subscriptions = sum(len(listeners) for listeners in self._subscriptions.values())
            return {
                "total_events_emitted": self._event_count,
                "total_errors": self._error_count,
                "total_subscriptions": total_subscriptions,
                "global_listeners": len(self._global_listeners),
            }

    def clear_all_subscriptions(self) -> None:
        """Remove all subscriptions.

        This is useful for cleanup or testing. Use with caution.
        """
        with self._lock:
            self._subscriptions.clear()
            self._global_listeners.clear()
            if self._enable_logging:
                logger.debug("Cleared all event subscriptions")

    def __repr__(self) -> str:
        """Return a string representation of the dispatcher."""
        stats = self.get_statistics()
        return (
            f"EventDispatcher(subscriptions={stats['total_subscriptions']}, "
            f"global_listeners={stats['global_listeners']}, "
            f"events_emitted={stats['total_events_emitted']}, "
            f"errors={stats['total_errors']})"
        )
