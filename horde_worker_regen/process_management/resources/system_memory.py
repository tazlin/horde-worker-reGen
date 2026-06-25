"""System-RAM accounting: how much total RAM exists, how much is in use, and the worker's own share.

The worker spawns several processes that each keep model weights resident in system RAM for fast
reload, so its memory footprint is spread across the orchestrator, the inference processes, the
safety process, and the background download process. Neither the console status block nor the TUI
made the *total* system-RAM picture, or that per-role split, legible: an operator could see free VRAM
but had no clear read on whether the machine was close to paging, nor which part of the worker was
holding the RAM. This module turns a few cheap psutil reads plus the per-process RSS the worker
already tracks into one :class:`SystemMemorySummary` that the console reporter and the TUI both render.

RSS is what is summed per role: the resident footprint each process actually occupies. It double-counts
memory pages shared between processes (copy-on-write fork pages, shared libraries, any shared-memory
weight buffers), so the per-role figures are an upper bound on the worker's true cost to the machine,
not an exact partition. The system-wide *used* figure (total minus available) is the authoritative
whole-machine number and is derived independently of the per-role sum; ``other_bytes`` (used minus the
worker subtotal) is therefore clamped at zero so the over-counting can never produce a negative.

Torch-free by construction (stdlib + the caller's psutil reads only), so it is safe to import in the
orchestrator process.
"""

from __future__ import annotations

from dataclasses import dataclass

ROLE_ORCHESTRATOR = "orchestrator"
"""The main worker process: job orchestration, scheduling, the TUI/supervisor channel."""
ROLE_INFERENCE = "inference"
"""All inference processes combined; this is where resident model weights dominate the footprint."""
ROLE_SAFETY = "safety"
"""The safety-checker process."""
ROLE_DOWNLOAD = "download"
"""The background model-download process (lives outside the process map)."""

WORKER_ROLE_ORDER: tuple[str, ...] = (
    ROLE_INFERENCE,
    ROLE_SAFETY,
    ROLE_ORCHESTRATOR,
    ROLE_DOWNLOAD,
)
"""Display order for the per-role breakdown: heaviest-first (inference), with download last."""

ROLE_LABELS: dict[str, str] = {
    ROLE_ORCHESTRATOR: "orchestrator",
    ROLE_INFERENCE: "inference",
    ROLE_SAFETY: "safety",
    ROLE_DOWNLOAD: "download",
}
"""Human-readable labels for each role, for the console/TUI breakdown."""


@dataclass(frozen=True)
class SystemMemorySummary:
    """A point-in-time view of system RAM and the worker's per-role share of it.

    ``worker_rss_by_role`` maps a role key (see the ``ROLE_*`` constants) to that role's summed RSS in
    bytes. All byte figures are non-negative; the builder clamps them.
    """

    total_bytes: int
    """Total physical RAM on the machine."""
    available_bytes: int
    """RAM the OS reports as available for new allocations without paging (psutil's ``available``)."""
    worker_rss_by_role: dict[str, int]
    """Per-role resident-set sizes (bytes) for the worker's own processes."""

    @property
    def used_bytes(self) -> int:
        """System-wide RAM in use (total minus available): the authoritative whole-machine figure."""
        return max(0, self.total_bytes - self.available_bytes)

    @property
    def worker_total_bytes(self) -> int:
        """The worker's own resident footprint, summed across every role (an upper bound; see module docs)."""
        return sum(max(0, value) for value in self.worker_rss_by_role.values())

    @property
    def other_bytes(self) -> int:
        """Used RAM not attributable to the worker (the OS and other applications).

        Clamped at zero because RSS over-counts shared pages, so the worker subtotal can exceed the
        independently-derived system used figure.
        """
        return max(0, self.used_bytes - self.worker_total_bytes)

    @property
    def used_fraction(self) -> float | None:
        """Fraction of total RAM in use system-wide, or None when total is unknown."""
        if self.total_bytes <= 0:
            return None
        return self.used_bytes / self.total_bytes

    @property
    def worker_fraction(self) -> float | None:
        """Fraction of total RAM held by the worker's own processes, or None when total is unknown."""
        if self.total_bytes <= 0:
            return None
        return min(1.0, self.worker_total_bytes / self.total_bytes)

    def nonzero_role_items(self) -> list[tuple[str, int]]:
        """Return ``(role, bytes)`` pairs with a non-zero footprint, in :data:`WORKER_ROLE_ORDER`.

        Roles that contributed nothing (e.g. no download process running) are omitted so the breakdown
        stays terse.
        """
        items: list[tuple[str, int]] = []
        for role in WORKER_ROLE_ORDER:
            value = self.worker_rss_by_role.get(role, 0)
            if value > 0:
                items.append((role, value))
        return items


def build_system_memory_summary(
    *,
    total_bytes: int,
    available_bytes: int,
    worker_rss_by_role: dict[str, int],
) -> SystemMemorySummary:
    """Build a :class:`SystemMemorySummary`, clamping every byte figure to be non-negative.

    ``worker_rss_by_role`` is copied (so the caller's dict is never aliased) and any negative or missing
    entries are normalised to zero.
    """
    normalised = {role: max(0, int(value)) for role, value in worker_rss_by_role.items()}
    return SystemMemorySummary(
        total_bytes=max(0, int(total_bytes)),
        available_bytes=max(0, int(available_bytes)),
        worker_rss_by_role=normalised,
    )
