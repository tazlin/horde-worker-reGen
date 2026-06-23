"""Host-aware admission control for parallel model downloads.

The download process can fetch several models at once, but should parallelize across *distinct hosts*
(``civitai.com`` ‖ ``huggingface.co`` ‖ ``github.com``) rather than open many connections to one server.
This module is the pure, torch/hordelib-free policy core: it owns the pending queue and the in-flight
accounting and answers "which task may start now?" under two live limits:

- ``per_host_concurrency`` (default 1): how many downloads to the *same* host may run at once. Raising
  it is the "thread to a single domain too" toggle.
- ``max_parallel_downloads``: the global ceiling across all hosts (1 restores fully-sequential behaviour).

Execution (which manager method actually fetches the bytes, progress reporting, pausing) lives in the
download process; this module never imports hordelib or touches disk, so the admission policy can be
unit-tested directly. Coordination uses a :class:`threading.Condition` so executor threads block when
nothing is admissible and wake the instant a task is enqueued, a slot frees, or a limit changes.
"""

from __future__ import annotations

import enum
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = [
    "DownloadKind",
    "DownloadTask",
    "HostAwareDownloadScheduler",
]


class DownloadKind(enum.Enum):
    """How the executor should fetch a task (the scheduler itself only schedules; it never runs them)."""

    IMAGE_MODEL = enum.auto()
    """A configured image-generation checkpoint (validate + retry-once via ``download_one_model``)."""
    AUX_MODEL = enum.auto()
    """A single auxiliary model fetched through its manager's ``download_model`` (clip/controlnet/...)."""
    SAFETY = enum.auto()
    """The required safety models (DeepDanbooru + CLIP); a one-shot ensure, not a per-file fetch."""
    DEFAULT_LORAS = enum.auto()
    """The curated default-LoRa set (CivitAI ad-hoc engine, coarse progress)."""
    ANNOTATORS = enum.auto()
    """ControlNet annotators (need a ComfyUI init; no clean per-file URL)."""


@dataclass(frozen=True)
class DownloadTask:
    """One unit of download work, tagged with the host it targets for per-host scheduling."""

    kind: DownloadKind
    model_name: str
    """The reference key (image/aux) or a synthetic label for coarse kinds (safety/annotators/loras)."""
    host: str
    """The source hostname (see :func:`model_download_core.download_host_for_url`)."""
    feature: str
    """Human label for why this downloads (e.g. 'image model', 'ControlNet')."""
    manager_key: str = ""
    """Which model manager fetches an ``AUX_MODEL`` (e.g. 'controlnet', 'gfpgan'); unused otherwise."""
    target_dir: str = ""
    size_bytes: int | None = None
    exclusive: bool = False
    """Run alone: admitted only when nothing else is in flight, and blocks others while it runs.

    Used for work that mutates global process state rather than fetching one clean file (the ControlNet
    annotators, which do a full ComfyUI/torch init); running it concurrently with other downloads sharing
    the model managers would race that global setup.
    """

    @property
    def dedup_key(self) -> tuple[DownloadKind, str, str]:
        """Identity for de-duplication: the same model under the same manager/kind is one task."""
        return (self.kind, self.manager_key, self.model_name)


@dataclass
class _State:
    """Mutable scheduler state, guarded by the scheduler's condition lock."""

    pending: list[DownloadTask] = field(default_factory=list)
    in_flight_by_host: Counter[str] = field(default_factory=Counter)
    in_flight_keys: set[tuple[DownloadKind, str, str]] = field(default_factory=set)
    active_count: int = 0
    exclusive_in_flight: int = 0


class HostAwareDownloadScheduler:
    """Admits download tasks subject to a global and a per-host concurrency limit.

    Thread-safe. Executor threads call :meth:`acquire` (blocks until a task is admissible or the
    scheduler is closed) and :meth:`release` when done. Producers call :meth:`enqueue` / :meth:`prune`
    and may retune the limits live via :meth:`set_limits`.
    """

    def __init__(self, *, max_parallel_downloads: int = 4, per_host_concurrency: int = 1) -> None:
        """Initialise with the global and per-host concurrency ceilings (each clamped to >= 1)."""
        self._cond = threading.Condition()
        self._state = _State()
        self._max_parallel = max(1, max_parallel_downloads)
        self._per_host = max(1, per_host_concurrency)
        self._closed = False

    def set_limits(
        self,
        *,
        max_parallel_downloads: int | None = None,
        per_host_concurrency: int | None = None,
    ) -> None:
        """Retune the ceilings live; wakes waiters since a raised limit may make a task admissible now."""
        with self._cond:
            if max_parallel_downloads is not None:
                self._max_parallel = max(1, max_parallel_downloads)
            if per_host_concurrency is not None:
                self._per_host = max(1, per_host_concurrency)
            self._cond.notify_all()

    def enqueue(self, task: DownloadTask) -> bool:
        """Queue *task* unless an identical one is already pending or in flight. Returns whether added."""
        with self._cond:
            if task.dedup_key in self._state.in_flight_keys:
                return False
            if any(pending.dedup_key == task.dedup_key for pending in self._state.pending):
                return False
            self._state.pending.append(task)
            self._cond.notify()
            return True

    def enqueue_many(self, tasks: list[DownloadTask]) -> int:
        """Queue each of *tasks* (deduped); return how many were newly added."""
        added = 0
        with self._cond:
            for task in tasks:
                if task.dedup_key in self._state.in_flight_keys:
                    continue
                if any(pending.dedup_key == task.dedup_key for pending in self._state.pending):
                    continue
                self._state.pending.append(task)
                added += 1
            if added:
                self._cond.notify_all()
        return added

    def prune(self, keep: Callable[[DownloadTask], bool]) -> list[DownloadTask]:
        """Drop pending tasks for which *keep* is false (e.g. a model removed from config).

        Only the *pending* queue is pruned; an in-flight task is the executor's to cancel. Returns the
        removed tasks so the caller can decide whether any in-flight download must also be aborted.
        """
        with self._cond:
            removed = [task for task in self._state.pending if not keep(task)]
            if removed:
                self._state.pending = [task for task in self._state.pending if keep(task)]
            return removed

    def acquire(self, *, timeout: float = 0.2) -> DownloadTask | None:
        """Block until a task is admissible, then claim it; return None on timeout or when closed.

        Admissible = some pending task whose host is below ``per_host_concurrency`` while the global
        ``max_parallel_downloads`` is not yet reached. Iterates the queue in order and returns the first
        admissible task, so a download to an idle host can start ahead of one blocked behind a busy host.
        """
        with self._cond:
            if self._closed:
                return None
            task = self._find_admissible()
            if task is None:
                self._cond.wait(timeout)
                if self._closed:
                    return None
                task = self._find_admissible()
                if task is None:
                    return None
            self._state.pending.remove(task)
            self._state.in_flight_by_host[task.host] += 1
            self._state.in_flight_keys.add(task.dedup_key)
            self._state.active_count += 1
            if task.exclusive:
                self._state.exclusive_in_flight += 1
            return task

    def _find_admissible(self) -> DownloadTask | None:
        """Return the first pending task that fits the limits and exclusivity rules (caller holds the lock).

        An exclusive task runs alone: it is admissible only once nothing is in flight, and while it runs no
        other task is admitted. A non-exclusive task is held while any exclusive task is in flight.
        """
        if self._state.active_count >= self._max_parallel:
            return None
        for task in self._state.pending:
            if task.exclusive:
                if self._state.active_count == 0:
                    return task
                continue
            if self._state.exclusive_in_flight > 0:
                continue
            if self._state.in_flight_by_host[task.host] < self._per_host:
                return task
        return None

    def release(self, task: DownloadTask) -> None:
        """Mark *task* finished, freeing its host and global slots, and wake any waiting executor."""
        with self._cond:
            self._state.active_count = max(0, self._state.active_count - 1)
            if task.exclusive:
                self._state.exclusive_in_flight = max(0, self._state.exclusive_in_flight - 1)
            self._state.in_flight_keys.discard(task.dedup_key)
            remaining = self._state.in_flight_by_host[task.host] - 1
            if remaining > 0:
                self._state.in_flight_by_host[task.host] = remaining
            else:
                self._state.in_flight_by_host.pop(task.host, None)
            self._cond.notify_all()

    @property
    def active_count(self) -> int:
        """How many downloads are currently in flight."""
        with self._cond:
            return self._state.active_count

    def pending_snapshot(self) -> list[DownloadTask]:
        """A copy of the pending queue, for status reporting."""
        with self._cond:
            return list(self._state.pending)

    def has_work(self) -> bool:
        """Whether any task is pending or in flight (the process is not idle)."""
        with self._cond:
            return bool(self._state.pending) or self._state.active_count > 0

    def close(self) -> None:
        """Unblock every waiting :meth:`acquire` (returns None) for shutdown."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
