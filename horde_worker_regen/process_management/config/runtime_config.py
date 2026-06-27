"""Live, hot-reloadable bridge configuration.

Public members:
    ``RuntimeConfig``: single-writer/many-reader holder for ``reGenBridgeData``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData


class RuntimeConfig:
    """Holds the current bridge configuration snapshot and the live effective concurrency.

    The reload loop is the only writer of ``bridge_data``; components read via ``bridge_data``.
    Reads return whatever snapshot is current at the moment of the read; no atomicity is promised
    across multiple reads.

    ``effective_max_threads`` is the *live* concurrent-inference cap the scheduler and popper read
    each decision. It is bounded by ``max_threads_ceiling`` (the value the IPC semaphores were
    provisioned for at construction, which cannot grow at runtime). A config reload re-derives it
    from ``bridge_data.max_threads``; a supervisor command can set it directly. This split lets the
    worker change its thread count at runtime without resizing multiprocessing primitives.
    """

    _bridge_data: reGenBridgeData
    _max_threads_ceiling: int
    _effective_max_threads: int

    def __init__(self, initial: reGenBridgeData, *, max_threads_ceiling: int | None = None) -> None:
        """Initialize with the first bridge configuration and concurrency ceiling."""
        self._bridge_data = initial
        ceiling = max(1, max_threads_ceiling if max_threads_ceiling is not None else initial.max_threads)
        self._max_threads_ceiling = ceiling
        self._effective_max_threads = max(1, min(initial.max_threads, ceiling))

    @property
    def bridge_data(self) -> reGenBridgeData:
        """Return the current bridge configuration snapshot."""
        return self._bridge_data

    @property
    def max_threads_ceiling(self) -> int:
        """The maximum concurrent-inference count the IPC primitives were provisioned for."""
        return self._max_threads_ceiling

    @property
    def effective_max_threads(self) -> int:
        """The live concurrent-inference cap the scheduler and popper enforce."""
        return self._effective_max_threads

    def set_effective_max_threads(self, value: int) -> int:
        """Set the live concurrent-inference cap (clamped to ``[1, max_threads_ceiling]``).

        Returns:
            The value actually applied after clamping.
        """
        self._effective_max_threads = max(1, min(value, self._max_threads_ceiling))
        return self._effective_max_threads

    def update(self, new: reGenBridgeData) -> None:
        """Replace the current configuration with ``new`` and re-derive the effective thread cap."""
        self._bridge_data = new
        self._effective_max_threads = max(1, min(new.max_threads, self._max_threads_ceiling))
