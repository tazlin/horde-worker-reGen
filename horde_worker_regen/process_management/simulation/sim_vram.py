"""A simulated device-VRAM ledger for deterministic post-processing-pressure tests.

The worker faults jobs and burns process recoveries when a job's post-processing peak (an
upscaler/face-fixer that runs *after* sampling) overflows a card already committed to sibling models
and process contexts. Reproducing that without a GPU needs a faithful model of one fact established by
reading ComfyUI's ``model_management.free_memory``: it iterates a *per-process* ``current_loaded_models``
global, so a child process can free its **own** resident model weights to make room for its upscaler, but
it **cannot** free a sibling process's resident model or CUDA context. Only the orchestrator (by sending
that sibling an unload) can reclaim cross-process VRAM.

This module provides:

* :class:`SimVramLedger`: a small, ``multiprocessing.Manager``-backed ledger of per-process VRAM
  contributions on each simulated card, mutable and readable across the spawn boundary (orchestrator and
  every fake child share one ledger).
* :func:`simulate_post_processing_allocation`: the decision a fake child makes when its job reaches
  post-processing: free its own models, then report whether the device has room for the peak. This
  encodes the cross-process rule, so a sibling's weights stay charged until the orchestrator evicts them.

The orchestrator needs no changes to see the simulated device: it derives free VRAM from each child's
reported ``total_vram_mb - vram_usage_mb``, so a fake child that reports ledger-derived figures feeds the
real budget/forecast seams directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing.managers import DictProxy, SyncManager
from threading import Lock as ThreadLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.synchronize import Lock as ProcessLock

# The kinds of VRAM a process contributes on a card. Kept as plain string suffixes (not an enum) so the
# flat ledger keys stay trivially picklable and human-readable in a proxy dict dump.
_KIND_WEIGHTS = "weights"
"""Resident model weights a process holds (freeable by that process itself: ComfyUI's free_memory)."""
_KIND_CONTEXT = "context"
"""Fixed per-process CUDA-context overhead (reclaimed only by stopping the process)."""
_KIND_TRANSIENT = "transient"
"""A transient allocation in flight (a sampling or post-processing activation peak)."""
_TOTAL_KEY = "__total__"
"""Per-device sentinel key holding the card's total VRAM (MB)."""


def _key(device_index: int, process_id: int | str, kind: str) -> str:
    """Build the flat ledger key for one process's contribution of ``kind`` on ``device_index``."""
    return f"{device_index}|{process_id}|{kind}"


def _total_key(device_index: int) -> str:
    """Build the flat ledger key holding ``device_index``'s total VRAM."""
    return f"{device_index}|{_TOTAL_KEY}"


@dataclass(frozen=True)
class SimProcessVram:
    """The VRAM a single (fake) inference process occupies on a simulated card, in MB.

    Attributes:
        process_id: The inference slot this describes.
        weights_mb: Resident model-weight footprint (the process can free this for its own upscaler).
        context_mb: Fixed CUDA-context overhead (only a process teardown reclaims it).
    """

    process_id: int
    weights_mb: float
    context_mb: float


@dataclass
class SimVramSpec:
    """A declarative seed for a simulated card's residency, for a test to express an over/under-commit.

    Attributes:
        device_index: The card this describes (0 on a single-GPU host).
        total_vram_mb: The card's total VRAM.
        processes: Per-process resident weights and context overhead already on the card.
    """

    device_index: int = 0
    total_vram_mb: float = 16375.0
    processes: list[SimProcessVram] = field(default_factory=list)

    def seed(self, ledger: SimVramLedger) -> None:
        """Write this card's total and per-process residency into ``ledger``."""
        ledger.set_total(self.device_index, self.total_vram_mb)
        for proc in self.processes:
            ledger.set_resident_weights(self.device_index, proc.process_id, proc.weights_mb)
            ledger.set_context_overhead(self.device_index, proc.process_id, proc.context_mb)


class SimVramLedger:
    """A cross-process VRAM ledger backed by a ``multiprocessing.Manager`` proxy dict.

    Holds only a manager proxy dict and a lock, so a :class:`SimVramLedger` instance pickles across the
    spawn boundary: the orchestrator and every fake child operate on the same underlying ledger. All
    figures are MB. Construct via :meth:`from_manager` (in the process that owns the manager) and pass the
    instance to the fake processes as a normal kwarg.

    The cross-process rule lives in the API surface: :meth:`free_own_models` frees only the named
    process's weights and transient, never a sibling's, mirroring ComfyUI's per-process ``free_memory``.
    """

    def __init__(self, entries: DictProxy[str, float] | dict[str, float], lock: ProcessLock | ThreadLock) -> None:
        """Wrap a (proxy or plain) dict of ledger entries and a guarding lock.

        Args:
            entries: The shared mapping of flat key -> MB. A manager ``DictProxy`` for the cross-process
                case, or a plain ``dict`` for in-process unit tests.
            lock: A lock guarding compound read-modify-write sequences. A manager lock across processes,
                or a ``threading.Lock`` in-process.
        """
        self._entries = entries
        self._lock = lock

    @classmethod
    def from_manager(cls, manager: SyncManager) -> SimVramLedger:
        """Build a ledger backed by ``manager`` (shareable with spawned processes)."""
        return cls(manager.dict(), manager.Lock())

    @classmethod
    def in_process(cls) -> SimVramLedger:
        """Build a single-process ledger (a plain dict + thread lock), for unit tests without a spawn."""
        return cls({}, ThreadLock())

    def set_total(self, device_index: int, total_mb: float) -> None:
        """Set ``device_index``'s total VRAM (MB)."""
        with self._lock:
            self._entries[_total_key(device_index)] = float(total_mb)

    def set_resident_weights(self, device_index: int, process_id: int, weights_mb: float) -> None:
        """Set the resident model-weight footprint (MB) a process holds on a card."""
        with self._lock:
            self._entries[_key(device_index, process_id, _KIND_WEIGHTS)] = float(weights_mb)

    def set_context_overhead(self, device_index: int, process_id: int, context_mb: float) -> None:
        """Set a process's fixed CUDA-context overhead (MB) on a card."""
        with self._lock:
            self._entries[_key(device_index, process_id, _KIND_CONTEXT)] = float(context_mb)

    def set_transient(self, device_index: int, process_id: int, transient_mb: float) -> None:
        """Set a process's in-flight transient allocation (MB), a sampling or post-processing peak."""
        with self._lock:
            self._entries[_key(device_index, process_id, _KIND_TRANSIENT)] = float(transient_mb)

    def clear_transient(self, device_index: int, process_id: int) -> None:
        """Drop a process's transient allocation once its sampling/post-processing peak is released."""
        self.set_transient(device_index, process_id, 0.0)

    def free_own_models(self, device_index: int, process_id: int) -> None:
        """Free *only* this process's resident weights and transient (ComfyUI's per-process free_memory).

        A sibling's weights and context are left untouched: a child cannot reclaim cross-process VRAM,
        only the orchestrator can (by sending that sibling an unload, which the sibling enacts via this
        same call on its own slot).
        """
        with self._lock:
            self._entries[_key(device_index, process_id, _KIND_WEIGHTS)] = 0.0
            self._entries[_key(device_index, process_id, _KIND_TRANSIENT)] = 0.0

    def total_mb(self, device_index: int) -> float:
        """Return ``device_index``'s total VRAM (MB); 0.0 when the card was never seeded."""
        with self._lock:
            return float(self._entries.get(_total_key(device_index), 0.0))

    def device_used_mb(self, device_index: int) -> float:
        """Return device-wide used VRAM (MB): every process's weights + context + transient on the card.

        This is the device-wide figure each child reports as ``vram_usage_mb`` (the worker treats it as
        ``torch_total - torch_free``), so all children on a card agree on the free figure the orchestrator
        derives.
        """
        prefix = f"{device_index}|"
        total_marker = _total_key(device_index)
        with self._lock:
            return float(
                sum(mb for key, mb in self._entries.items() if key.startswith(prefix) and key != total_marker),
            )

    def device_free_mb(self, device_index: int) -> float:
        """Return device-wide free VRAM (MB): ``total - used`` (floored at 0)."""
        return max(0.0, self.total_mb(device_index) - self.device_used_mb(device_index))


def simulate_post_processing_allocation(
    ledger: SimVramLedger,
    *,
    device_index: int,
    process_id: int,
    post_processing_peak_mb: float,
) -> bool:
    """Resolve whether a process's post-processing peak fits, applying the cross-process free rule.

    Mirrors what happens inside a real inference child when its job reaches post-processing: ComfyUI calls
    ``free_memory`` for the upscaler, which can evict only this process's own model. So this frees the
    process's own weights, then checks whether the device now has room for ``post_processing_peak_mb``.
    Returns ``True`` when the peak fits (the upscaler allocates and the job completes), ``False`` when it
    does not (the upscaler thrashes or stalls, and the caller should then hang so the watchdog reaps it).

    Because a sibling's weights and context are *not* freed here, an over-commit caused by sibling
    residency or contexts is only curable by the orchestrator evicting those siblings beforehand; this is
    exactly the dynamic the harness exists to make observable.
    """
    ledger.free_own_models(device_index, process_id)
    if ledger.device_free_mb(device_index) < post_processing_peak_mb:
        return False
    # The peak fits: charge it transiently so a concurrent sibling decision sees the card as occupied
    # while this upscaler runs (the caller clears it when post-processing completes).
    ledger.set_transient(device_index, process_id, post_processing_peak_mb)
    return True
