# Worker Event System

## Overview

The worker event system provides an event-driven architecture for observing worker state changes, job progress, and other significant occurrences. This enables clean separation between business logic and presentation layers (UI, monitoring, metrics, etc.).

## Architecture

```
┌─────────────────────┐
│  Business Logic     │
│  (ProcessManager,   │
│   StatusReporter)   │
└──────────┬──────────┘
           │ emits events
           ▼
┌─────────────────────┐
│  EventDispatcher    │◄─── subscribe ──┬── ConsoleUI
│                     │                 ├── TerminalUI
│  (Thread-safe hub)  │                 ├── MetricsExporter
└─────────────────────┘                 └── Custom Listeners
```

## Key Concepts

### Events

Events are immutable dataclasses that represent something that happened in the worker. All events inherit from `WorkerEvent` and include:

- **Timestamp**: When the event occurred (epoch time)
- **Priority**: Event priority (LOW, NORMAL, HIGH, CRITICAL)
- **Type-specific data**: Fields relevant to that event type

### Event Dispatcher

The `EventDispatcher` is the central hub that:
- Manages subscriptions from listeners
- Emits events to all subscribed listeners
- Is thread-safe and can be used from async contexts
- Handles errors gracefully (calls `on_error` if listener raises exception)

### Event Listeners

Event listeners are objects that receive and handle events. Listeners implement the `EventListener` interface:

```python
class MyListener(EventListener):
    def on_event(self, event: WorkerEvent) -> None:
        # Handle the event
        pass

    def on_error(self, event: WorkerEvent, exception: Exception) -> None:
        # Handle errors (optional, has default implementation)
        pass
```

## Event Types

### Process Events
- `ProcessStartedEvent` - Child process started
- `ProcessEndedEvent` - Child process ended
- `ProcessStateChangedEvent` - Process changed state (downloading, inferencing, etc.)
- `ProcessHeartbeatEvent` - Heartbeat received (includes progress %)
- `ProcessMemoryUpdatedEvent` - Memory usage updated

### Job Events
- `JobPoppedEvent` - New job received from API
- `JobStartedEvent` - Job started processing
- `JobCompletedEvent` - Job completed successfully
- `JobFaultedEvent` - Job failed or faulted
- `JobQueueChangedEvent` - Job queue state changed

### Model Events
- `ModelDownloadStartedEvent` - Model download started
- `ModelDownloadProgressEvent` - Model download progress update
- `ModelLoadingEvent` - Model loading into memory
- `ModelLoadedEvent` - Model loaded successfully
- `ModelUnloadedEvent` - Model unloaded from memory

### Worker Events
- `WorkerStartedEvent` - Worker started up
- `WorkerStatusEvent` - Periodic status snapshot (all worker state)
- `APIMessageReceivedEvent` - Message received from Horde API
- `MaintenanceModeEvent` - Maintenance mode status changed
- `ShutdownInitiatedEvent` - Worker shutdown initiated

### Performance Events
- `KudosEarnedEvent` - Kudos earned from job
- `PerformanceWarningEvent` - Performance warning detected

## Usage Examples

### Basic Event Emission

```python
from horde_worker_regen.events import EventDispatcher, JobCompletedEvent

# Create dispatcher
dispatcher = EventDispatcher()

# Emit an event
event = JobCompletedEvent(
    job_id="abc123",
    process_id=1,
    model_name="SDXL",
    kudos_earned=12.5,
    generation_time_seconds=5.2,
    num_images=1,
)
dispatcher.emit(event)
```

### Creating a Listener

```python
from horde_worker_regen.events import EventListener, JobCompletedEvent

class KudosTracker(EventListener):
    def __init__(self):
        self.total_kudos = 0.0

    def on_event(self, event: WorkerEvent) -> None:
        if isinstance(event, JobCompletedEvent):
            self.total_kudos += event.kudos_earned
            print(f"Total kudos: {self.total_kudos}")

# Subscribe the listener
tracker = KudosTracker()
dispatcher.subscribe(JobCompletedEvent, tracker)
```

### Subscribing to All Events

```python
from horde_worker_regen.events import LoggingEventListener

# This listener will receive ALL events
logger_listener = LoggingEventListener(log_level="INFO", include_heartbeats=False)
dispatcher.subscribe_to_all(logger_listener)
```

### Using Callback Listeners

For simple cases, you can use a callback function:

```python
from horde_worker_regen.events import CallbackEventListener, JobPoppedEvent

def on_job_popped(event):
    print(f"New job: {event.model_name}")

listener = CallbackEventListener(on_job_popped)
dispatcher.subscribe(JobPoppedEvent, listener)
```

### Filtered Listeners

For more control, extend `FilteredEventListener`:

```python
from horde_worker_regen.events import FilteredEventListener, EventPriority

class CriticalEventLogger(FilteredEventListener):
    def should_handle_event(self, event: WorkerEvent) -> bool:
        return event.priority == EventPriority.CRITICAL

    def handle_event(self, event: WorkerEvent) -> None:
        print(f"CRITICAL: {event}")

listener = CriticalEventLogger()
dispatcher.subscribe_to_all(listener)  # Gets all events but filters to CRITICAL
```

## Integration Guide

### Phase 1: Add EventDispatcher to ProcessManager (CURRENT)

The event system is created but not yet integrated into the worker. Future phases will:

1. Add `EventDispatcher` instance to `HordeWorkerProcessManager`
2. Emit events at key points in worker lifecycle
3. Create UI implementations that subscribe to events

### Phase 2: Emit Events from Business Logic (TODO)

Modify worker code to emit events:

```python
# In process_manager.py
class HordeWorkerProcessManager:
    def __init__(self, ..., event_dispatcher: EventDispatcher):
        self.event_dispatcher = event_dispatcher

        # Emit worker started event
        self.event_dispatcher.emit(WorkerStartedEvent(
            worker_name=bridge_data.dreamer_worker_name,
            worker_version=horde_worker_regen.__version__,
            max_threads=max_threads,
            ...
        ))

    def _on_job_completed(self, job):
        # Emit job completed event
        self.event_dispatcher.emit(JobCompletedEvent(
            job_id=job.id_,
            kudos_earned=kudos,
            ...
        ))
```

### Phase 3: Create UI Implementations (TODO)

```python
# In ui/console_ui.py
class ConsoleUI(BaseWorkerUI):
    def initialize(self, event_dispatcher: EventDispatcher):
        # Subscribe to status events only (current behavior)
        event_dispatcher.subscribe(WorkerStatusEvent, self.status_listener)

# In ui/terminal_ui.py
class TerminalUI(BaseWorkerUI):
    def initialize(self, event_dispatcher: EventDispatcher):
        # Subscribe to all events for live UI
        event_dispatcher.subscribe(ProcessStateChangedEvent, self.process_listener)
        event_dispatcher.subscribe(JobPoppedEvent, self.job_listener)
        event_dispatcher.subscribe(ProcessHeartbeatEvent, self.progress_listener)
        # etc...
```

## Best Practices

### Event Emission
1. **Emit events immediately after state changes** - Don't batch or delay
2. **Keep events immutable** - Use frozen dataclasses
3. **Include relevant context** - All data needed to understand the event
4. **Use appropriate priority** - Don't mark everything as CRITICAL

### Listener Implementation
1. **Keep `on_event` fast** - Offload long operations to queues/threads
2. **Handle errors gracefully** - Implement `on_error` if needed
3. **Don't modify worker state** - Listeners should observe, not mutate
4. **Unsubscribe when done** - Clean up listeners on shutdown

### Performance
1. **High-frequency events use LOW priority** - Heartbeats, memory updates
2. **Filter events early** - Use `FilteredEventListener` or type-specific subscriptions
3. **Limit global listeners** - Subscribe to specific types when possible
4. **Monitor statistics** - Use `dispatcher.get_statistics()` to track usage

## Testing

```python
# Test that events are emitted correctly
def test_event_emission():
    dispatcher = EventDispatcher()
    events_received = []

    listener = CallbackEventListener(lambda e: events_received.append(e))
    dispatcher.subscribe(JobCompletedEvent, listener)

    # Emit event
    event = JobCompletedEvent(job_id="test", kudos_earned=5.0, ...)
    dispatcher.emit(event)

    # Verify
    assert len(events_received) == 1
    assert events_received[0].job_id == "test"
```

## Thread Safety

The `EventDispatcher` is thread-safe and uses a reentrant lock (`threading.RLock`) to protect subscription management. This means:

- ✅ Safe to emit events from multiple threads
- ✅ Safe to subscribe/unsubscribe from multiple threads
- ✅ Safe to use in async contexts (but listeners should be async-aware)
- ⚠️ Listeners are called synchronously - keep them fast!

## Migration Path

### Current State (Phase 1)
- ✅ Event system infrastructure created
- ✅ Event types defined
- ✅ EventDispatcher implemented
- ✅ Example listeners provided

### Next Steps (Phase 2)
- [ ] Add EventDispatcher to HordeWorkerProcessManager
- [ ] Emit events from process state changes
- [ ] Emit events from job lifecycle
- [ ] Emit events from model operations
- [ ] Test event emissions

### Future (Phase 3+)
- [ ] Create ConsoleUI wrapper (maintains current behavior)
- [ ] Create TerminalUI with rich/textual
- [ ] Add configuration for UI mode selection
- [ ] Add metrics exporter listener
- [ ] Add web UI event stream

## Example: Complete Integration

```python
# In run_worker.py
from horde_worker_regen.events import EventDispatcher
from horde_worker_regen.ui import ConsoleUI, TerminalUI

def main():
    # Create event dispatcher
    event_dispatcher = EventDispatcher(enable_logging=False)

    # Choose UI based on config
    if bridge_data.disable_terminal_ui:
        ui = ConsoleUI()
    else:
        ui = TerminalUI()

    # Create process manager with event dispatcher
    process_manager = HordeWorkerProcessManager(
        bridge_data=bridge_data,
        event_dispatcher=event_dispatcher,
        ...
    )

    # Initialize UI (subscribes to events)
    ui.initialize(event_dispatcher)

    # Start worker (events will flow to UI)
    process_manager.start()
```

## Statistics and Monitoring

Get dispatcher statistics:

```python
stats = dispatcher.get_statistics()
print(stats)
# {
#     'total_events_emitted': 1523,
#     'total_errors': 0,
#     'total_subscriptions': 5,
#     'global_listeners': 1
# }
```

Use the built-in statistics listener:

```python
from horde_worker_regen.events.example_listeners import EventStatisticsListener

stats_listener = EventStatisticsListener()
dispatcher.subscribe_to_all(stats_listener)

# Later...
stats_listener.print_summary()
```

## Questions?

For more information or questions about the event system:
- See type definitions in `event_types.py`
- See example implementations in `example_listeners.py`
- Check the implementation in `event_dispatcher.py`
