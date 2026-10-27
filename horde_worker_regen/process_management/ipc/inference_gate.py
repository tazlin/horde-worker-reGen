"""Bound inference permits across spawned children, with capacity changes while drained."""

from __future__ import annotations

import errno
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.context import BaseContext
from multiprocessing.synchronize import Semaphore
from multiprocessing.util import Finalize
from types import TracebackType

_WAKE_RECHECK_SECONDS = 0.1
"""Maximum wait before rechecking capacity if a process dies before forwarding a wake hint."""


class ResizableInferenceGate:
    """Preserve bounded release semantics while sharing capacity with existing children.

    The parent releases a dead child's possibly held permit during recovery. Rejecting an
    over-release is essential: recovery must never increase the configured concurrency.
    Capacity changes require every permit to be returned and intake to be paused by the caller.
    """

    def __init__(self, capacity: int, *, ctx: BaseContext) -> None:
        """Create a spawn-compatible gate with ``capacity`` available permits."""
        if capacity < 1:
            raise ValueError("inference capacity must be positive")
        handle, self._lock_path = tempfile.mkstemp(prefix="horde-inference-gate-")
        os.write(handle, b"0")
        os.close(handle)
        Finalize(None, os.unlink, args=(self._lock_path,), exitpriority=-100)
        self._wake = ctx.BoundedSemaphore(1)
        self._wake.acquire()
        self._capacity = ctx.Value("i", capacity, lock=False)
        self._available = ctx.Value("i", capacity, lock=False)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        # A child can die during a counter update. The OS releases a file lock on process death;
        # an inherited Python lock can remain acquired forever and block the parent's recovery.
        with open(self._lock_path, "r+b", buffering=0) as stream:
            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as error:
                        if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                            raise
                        time.sleep(0.001)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _notify(self) -> None:
        try:
            self._wake.release()
        except ValueError:
            # A wake is already pending. Counter ownership remains protected by the OS lock.
            return

    def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
        """Acquire a permit, returning False when a nonblocking or timed wait cannot proceed."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            with self._locked():
                if self._available.value:
                    self._available.value -= 1
                    if self._available.value:
                        self._notify()
                    return True
            if not block:
                return False
            remaining = _WAKE_RECHECK_SECONDS if deadline is None else deadline - time.monotonic()
            if remaining <= 0:
                return False
            # The semaphore is only a hint: a releaser can die after updating the counter, or a
            # waiter after consuming the hint. Neither may strand another waiter indefinitely.
            self._wake.acquire(timeout=min(remaining, _WAKE_RECHECK_SECONDS))

    def release(self) -> None:
        """Return a permit, rejecting an over-release during crash recovery."""
        with self._locked():
            if self._available.value >= self._capacity.value:
                self._notify()
                raise ValueError("inference gate released too many times")
            self._available.value += 1
            self._notify()

    def resize(self, capacity: int) -> None:
        """Change capacity only after all holders have returned their permits."""
        if capacity < 1:
            raise ValueError("inference capacity must be positive")
        with self._locked():
            if self._available.value != self._capacity.value:
                raise RuntimeError("cannot resize inference capacity while permits are held")
            self._capacity.value = capacity
            self._available.value = capacity
            self._notify()

    def get_value(self) -> int:
        """Return the number of currently available permits."""
        with self._locked():
            return self._available.value

    @property
    def capacity(self) -> int:
        """Return the configured bound shared by every child."""
        with self._locked():
            return self._capacity.value

    def __enter__(self) -> bool:
        """Acquire a permit for a context-managed inference."""
        return self.acquire()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        """Return the context's permit."""
        self.release()


InferenceGate = Semaphore | ResizableInferenceGate
"""Inference permit interface, including externally supplied fixed gates in tests."""
