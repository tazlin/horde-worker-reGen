"""A self-audited, append-only record of the lifecycle actions the worker takes on its children.

When a subprocess hangs, crashes, or is replaced, the single most useful thing for diagnosis is a
truthful, ordered account of what the parent *did* and *observed* leading up to it: when each slot
was spawned (and its OS pid), when inference was dispatched to it, when a held semaphore was released
on its behalf, when a timeout fired and why it was replaced. Without that, a post-mortem is guesswork
from scattered log lines.

The ledger keeps a bounded in-memory ring (always on, cheap, queryable for the timeout diagnostics
dump) and optionally mirrors each event to a size-rotated JSONL file so the record survives a restart
for offline analysis. It never raises: a file IO error is logged once and degrades to in-memory only,
so auditing can never itself wedge the worker.
"""

from __future__ import annotations

import enum
import json
import os
from collections import deque
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field


class LedgerEventType(enum.StrEnum):
    """The kinds of lifecycle action/observation the worker records about its children."""

    PROCESS_SPAWNED = "process_spawned"
    PROCESS_REPLACED = "process_replaced"
    PROCESS_QUARANTINED = "process_quarantined"
    PROCESS_ENDED = "process_ended"
    PROCESS_START_DEFERRED = "process_start_deferred"
    INFERENCE_DISPATCHED = "inference_dispatched"
    INFERENCE_RETRIED = "inference_retried"
    INFERENCE_FAULTED = "inference_faulted"
    POST_PROCESS_FAULTED = "post_process_faulted"
    PRELOAD_REQUESTED = "preload_requested"
    SEMAPHORE_RELEASED = "semaphore_released"
    TIMEOUT_DETECTED = "timeout_detected"
    SLOWDOWN_DETECTED = "slowdown_detected"
    SOFT_RESET = "soft_reset"
    RECOVERY_ABANDONED = "recovery_abandoned"
    ORPHAN_REAPED = "orphan_reaped"
    GOVERNANCE_RESET = "governance_reset"
    POP_PAUSE_ARMED = "pop_pause_armed"
    POP_PAUSE_LAPSED = "pop_pause_lapsed"
    HEAD_PRIORITY_BARRIER_ENGAGED = "head_priority_barrier_engaged"
    HEAD_PRIORITY_BARRIER_RELEASED = "head_priority_barrier_released"
    SAFETY_RECOVERY_HOLD_ENGAGED = "safety_recovery_hold_engaged"
    SAFETY_RECOVERY_HOLD_RELEASED = "safety_recovery_hold_released"
    RESIDENCY_ADVERTISING_NARROWED = "residency_advertising_narrowed"
    RESIDENCY_ADVERTISING_RELEASED = "residency_advertising_released"


class LedgerEvent(BaseModel):
    """One recorded lifecycle action or observation, keyed to a process slot when applicable."""

    timestamp: float
    event_type: LedgerEventType
    process_id: int | None = None
    """The logical slot id (0,1,2...), not the OS pid."""
    os_pid: int | None = None
    launch_identifier: int | None = None
    job_id: str | None = None
    reason: str = ""
    detail: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


_DEFAULT_MAX_IN_MEMORY = 500
_DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024


class ActionLedger:
    """Bounded in-memory ring of lifecycle events, optionally mirrored to a rotated JSONL file."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        max_in_memory: int = _DEFAULT_MAX_IN_MEMORY,
        max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        """Initialize the ledger.

        Args:
            path: JSONL file to mirror events to, or None for in-memory only (e.g. under test).
            max_in_memory: How many recent events to keep queryable in memory.
            max_file_bytes: Rotate the JSONL file (to ``<name>.1``) once it exceeds this size.
        """
        self._events: deque[LedgerEvent] = deque(maxlen=max_in_memory)
        self._path = path
        self._max_file_bytes = max_file_bytes
        self._file_disabled = False

    def record(
        self,
        event_type: LedgerEventType,
        *,
        process_id: int | None = None,
        os_pid: int | None = None,
        launch_identifier: int | None = None,
        job_id: str | None = None,
        reason: str = "",
        detail: dict[str, str | int | float | bool | None] | None = None,
    ) -> LedgerEvent:
        """Append a lifecycle event to the ring (and the file, if configured). Never raises."""
        import time

        event = LedgerEvent(
            timestamp=time.time(),
            event_type=event_type,
            process_id=process_id,
            os_pid=os_pid,
            launch_identifier=launch_identifier,
            job_id=job_id,
            reason=reason,
            detail=detail or {},
        )
        self._events.append(event)
        self._append_to_file(event)
        return event

    def recent(self, *, process_id: int | None = None, limit: int = 20) -> list[LedgerEvent]:
        """Return up to ``limit`` most-recent events, optionally filtered to one slot (oldest first)."""
        if process_id is None:
            selected = list(self._events)
        else:
            selected = [e for e in self._events if e.process_id == process_id]
        return selected[-limit:]

    def _append_to_file(self, event: LedgerEvent) -> None:
        if self._path is None or self._file_disabled:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed()
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.model_dump()) + "\n")
        except Exception as e:
            # Degrade to in-memory only rather than letting an audit IO error disrupt the worker.
            logger.warning(f"Action ledger file at {self._path} is unwritable ({type(e).__name__}); in-memory only")
            self._file_disabled = True

    def _rotate_if_needed(self) -> None:
        try:
            if self._path is not None and self._path.exists() and self._path.stat().st_size >= self._max_file_bytes:
                backup = self._path.with_suffix(self._path.suffix + ".1")
                os.replace(self._path, backup)
        except OSError as e:
            logger.debug(f"Action ledger rotation skipped: {type(e).__name__} {e}")
