"""Simple tests for the event system.

These tests verify that the event dispatcher and listeners work correctly.
Run with: python -m pytest horde_worker_regen/events/test_events.py
"""

from __future__ import annotations

import time

from horde_worker_regen.events import (
    EventDispatcher,
    EventListener,
    JobCompletedEvent,
    JobPoppedEvent,
    KudosEarnedEvent,
    ProcessHeartbeatEvent,
)
from horde_worker_regen.events.event_listener import CallbackEventListener
from horde_worker_regen.events.event_types import EventPriority
from horde_worker_regen.events.example_listeners import EventStatisticsListener, HighPriorityEventListener
from horde_worker_regen.process_management.messages import HordeHeartbeatType


def test_basic_event_emission():
    """Test that events can be emitted and received."""
    dispatcher = EventDispatcher()
    events_received = []

    listener = CallbackEventListener(lambda e: events_received.append(e))
    dispatcher.subscribe(JobCompletedEvent, listener)

    # Emit event
    event = JobCompletedEvent(
        job_id="test123",
        process_id=1,
        model_name="Test Model",
        kudos_earned=10.5,
        generation_time_seconds=5.0,
        num_images=1,
    )
    dispatcher.emit(event)

    # Verify
    assert len(events_received) == 1
    assert events_received[0].job_id == "test123"
    assert events_received[0].kudos_earned == 10.5


def test_multiple_listeners():
    """Test that multiple listeners can subscribe to the same event."""
    dispatcher = EventDispatcher()
    listener1_events = []
    listener2_events = []

    listener1 = CallbackEventListener(lambda e: listener1_events.append(e))
    listener2 = CallbackEventListener(lambda e: listener2_events.append(e))

    dispatcher.subscribe(JobCompletedEvent, listener1)
    dispatcher.subscribe(JobCompletedEvent, listener2)

    # Emit event
    event = JobCompletedEvent(
        job_id="test",
        process_id=1,
        model_name="Test",
        kudos_earned=5.0,
        generation_time_seconds=2.0,
        num_images=1,
    )
    dispatcher.emit(event)

    # Both listeners should receive the event
    assert len(listener1_events) == 1
    assert len(listener2_events) == 1


def test_unsubscribe():
    """Test that unsubscribing works correctly."""
    dispatcher = EventDispatcher()
    events_received = []

    listener = CallbackEventListener(lambda e: events_received.append(e))
    dispatcher.subscribe(JobCompletedEvent, listener)

    # Emit first event
    event1 = JobCompletedEvent(
        job_id="test1",
        process_id=1,
        model_name="Test",
        kudos_earned=5.0,
        generation_time_seconds=2.0,
        num_images=1,
    )
    dispatcher.emit(event1)

    # Unsubscribe
    dispatcher.unsubscribe(JobCompletedEvent, listener)

    # Emit second event
    event2 = JobCompletedEvent(
        job_id="test2",
        process_id=1,
        model_name="Test",
        kudos_earned=5.0,
        generation_time_seconds=2.0,
        num_images=1,
    )
    dispatcher.emit(event2)

    # Should only have received first event
    assert len(events_received) == 1
    assert events_received[0].job_id == "test1"


def test_global_listener():
    """Test that global listeners receive all events."""
    dispatcher = EventDispatcher()
    events_received = []

    listener = CallbackEventListener(lambda e: events_received.append(e))
    dispatcher.subscribe_to_all(listener)

    # Emit different event types
    event1 = JobCompletedEvent(
        job_id="test",
        process_id=1,
        model_name="Test",
        kudos_earned=5.0,
        generation_time_seconds=2.0,
        num_images=1,
    )
    event2 = JobPoppedEvent(
        job_id="test2",
        model_name="Test Model",
        width=512,
        height=512,
        steps=20,
        batch_size=1,
        estimated_megapixelsteps=100,
    )
    event3 = ProcessHeartbeatEvent(
        process_id=1,
        heartbeat_type=HordeHeartbeatType.INFERENCE_STEP,
        percent_complete=50,
    )

    dispatcher.emit(event1)
    dispatcher.emit(event2)
    dispatcher.emit(event3)

    # Should have received all three events
    assert len(events_received) == 3
    assert isinstance(events_received[0], JobCompletedEvent)
    assert isinstance(events_received[1], JobPoppedEvent)
    assert isinstance(events_received[2], ProcessHeartbeatEvent)


def test_statistics_listener():
    """Test the statistics listener."""
    dispatcher = EventDispatcher()
    stats_listener = EventStatisticsListener()
    dispatcher.subscribe_to_all(stats_listener)

    # Emit various events
    for i in range(5):
        event = JobCompletedEvent(
            job_id=f"test{i}",
            process_id=1,
            model_name="Test",
            kudos_earned=10.0,
            generation_time_seconds=2.0,
            num_images=1,
        )
        dispatcher.emit(event)

    for i in range(3):
        event = KudosEarnedEvent(
            job_id=f"test{i}",
            amount=10.0,
            cumulative_total=10.0 * (i + 1),
        )
        dispatcher.emit(event)

    # Check statistics
    summary = stats_listener.get_summary()
    assert summary["total_events"] == 8  # 5 job completed + 3 kudos
    assert summary["jobs_completed"] == 5
    assert summary["total_kudos"] == 30.0  # Last cumulative total


def test_high_priority_filter():
    """Test the high priority event filter."""
    dispatcher = EventDispatcher()
    high_priority_events = []

    class TestHighPriorityListener(HighPriorityEventListener):
        def handle_event(self, event):
            high_priority_events.append(event)

    listener = TestHighPriorityListener()
    dispatcher.subscribe_to_all(listener)

    # Emit events with different priorities
    low_event = ProcessHeartbeatEvent(
        process_id=1,
        heartbeat_type=HordeHeartbeatType.OTHER,
        priority=EventPriority.LOW,
    )
    normal_event = JobCompletedEvent(
        job_id="test",
        process_id=1,
        model_name="Test",
        kudos_earned=5.0,
        generation_time_seconds=2.0,
        num_images=1,
        priority=EventPriority.NORMAL,
    )
    high_event = JobCompletedEvent(
        job_id="test",
        process_id=1,
        model_name="Test",
        kudos_earned=5.0,
        generation_time_seconds=2.0,
        num_images=1,
        priority=EventPriority.HIGH,
    )

    dispatcher.emit(low_event)
    dispatcher.emit(normal_event)
    dispatcher.emit(high_event)

    # Should only have received the high priority event
    assert len(high_priority_events) == 1
    assert high_priority_events[0].priority == EventPriority.HIGH


def test_event_timestamp():
    """Test that events have timestamps."""
    before = time.time()
    event = JobCompletedEvent(
        job_id="test",
        process_id=1,
        model_name="Test",
        kudos_earned=5.0,
        generation_time_seconds=2.0,
        num_images=1,
    )
    after = time.time()

    # Timestamp should be between before and after
    assert before <= event.timestamp <= after


def test_dispatcher_statistics():
    """Test that dispatcher statistics are tracked correctly."""
    dispatcher = EventDispatcher()
    listener = CallbackEventListener(lambda e: None)

    dispatcher.subscribe(JobCompletedEvent, listener)
    dispatcher.subscribe(JobPoppedEvent, listener)

    # Emit some events
    for i in range(10):
        event = JobCompletedEvent(
            job_id=f"test{i}",
            process_id=1,
            model_name="Test",
            kudos_earned=5.0,
            generation_time_seconds=2.0,
            num_images=1,
        )
        dispatcher.emit(event)

    stats = dispatcher.get_statistics()
    assert stats["total_events_emitted"] == 10
    assert stats["total_subscriptions"] == 2
    assert stats["total_errors"] == 0


def test_error_handling():
    """Test that errors in listeners are handled gracefully."""

    class FailingListener(EventListener):
        def __init__(self):
            self.error_count = 0

        def on_event(self, event):
            raise ValueError("Test error")

        def on_error(self, event, exception):
            self.error_count += 1

    dispatcher = EventDispatcher()
    listener = FailingListener()
    dispatcher.subscribe(JobCompletedEvent, listener)

    # Emit event (should not raise)
    event = JobCompletedEvent(
        job_id="test",
        process_id=1,
        model_name="Test",
        kudos_earned=5.0,
        generation_time_seconds=2.0,
        num_images=1,
    )
    dispatcher.emit(event)

    # Error should have been caught and on_error called
    assert listener.error_count == 1
    stats = dispatcher.get_statistics()
    assert stats["total_errors"] == 1


if __name__ == "__main__":
    # Run basic tests
    print("Running event system tests...")
    test_basic_event_emission()
    print("✓ Basic event emission")
    test_multiple_listeners()
    print("✓ Multiple listeners")
    test_unsubscribe()
    print("✓ Unsubscribe")
    test_global_listener()
    print("✓ Global listener")
    test_statistics_listener()
    print("✓ Statistics listener")
    test_high_priority_filter()
    print("✓ High priority filter")
    test_event_timestamp()
    print("✓ Event timestamp")
    test_dispatcher_statistics()
    print("✓ Dispatcher statistics")
    test_error_handling()
    print("✓ Error handling")
    print("\nAll tests passed! ✓")
