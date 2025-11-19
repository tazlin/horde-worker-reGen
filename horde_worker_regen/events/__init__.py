"""Event system for worker UI and monitoring.

This package provides an event-driven architecture for observing worker state changes,
job progress, and other significant events. This allows decoupling of business logic
from UI presentation and enables multiple observers (terminal UI, metrics exporters, etc.).
"""

from __future__ import annotations

from horde_worker_regen.events.event_dispatcher import EventDispatcher
from horde_worker_regen.events.event_listener import EventListener
from horde_worker_regen.events.event_types import (
    APIMessageReceivedEvent,
    JobCompletedEvent,
    JobFaultedEvent,
    JobPoppedEvent,
    JobQueueChangedEvent,
    JobStartedEvent,
    KudosEarnedEvent,
    MaintenanceModeEvent,
    ModelDownloadProgressEvent,
    ModelDownloadStartedEvent,
    ModelLoadedEvent,
    ModelLoadingEvent,
    ModelUnloadedEvent,
    PerformanceWarningEvent,
    ProcessEndedEvent,
    ProcessHeartbeatEvent,
    ProcessMemoryUpdatedEvent,
    ProcessStartedEvent,
    ProcessStateChangedEvent,
    ShutdownInitiatedEvent,
    WorkerEvent,
    WorkerStartedEvent,
    WorkerStatusEvent,
)

__all__ = [
    # Core classes
    "EventDispatcher",
    "EventListener",
    # Base event type
    "WorkerEvent",
    # Process events
    "ProcessStateChangedEvent",
    "ProcessHeartbeatEvent",
    "ProcessMemoryUpdatedEvent",
    "ProcessStartedEvent",
    "ProcessEndedEvent",
    # Job events
    "JobPoppedEvent",
    "JobStartedEvent",
    "JobCompletedEvent",
    "JobFaultedEvent",
    "JobQueueChangedEvent",
    # Model events
    "ModelDownloadStartedEvent",
    "ModelDownloadProgressEvent",
    "ModelLoadingEvent",
    "ModelLoadedEvent",
    "ModelUnloadedEvent",
    # Worker events
    "WorkerStartedEvent",
    "WorkerStatusEvent",
    "APIMessageReceivedEvent",
    "MaintenanceModeEvent",
    "ShutdownInitiatedEvent",
    # Performance events
    "KudosEarnedEvent",
    "PerformanceWarningEvent",
]
