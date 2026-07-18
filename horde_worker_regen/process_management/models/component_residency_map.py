"""Parent-side visibility of which model components each child holds resident in its RAM component cache.

A GPU-bearing child (an inference process or a VAE/component lane) keeps an MB-budgeted cache of loaded
model components and reports its resident entries on every memory report. This map is the parent's decoded
view of those reports, keyed by process id: it answers which checkpoints are staged in RAM (the residency-
bias floor for pop advertising) and which cached components a RAM-pressure rung may evict without disturbing
a queued or in-flight job.

The parent is torch-free and never imports hordelib, so this consumes only the plain
:class:`~horde_worker_regen.process_management.ipc.messages.HeldComponentSnapshot` transport type, never
hordelib's cache. It mirrors :class:`~horde_worker_regen.process_management.models.horde_model_map.HordeModelMap`:
one shared structure the message dispatcher updates and the process lifecycle expires on death/recycle.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from loguru import logger

from horde_worker_regen.process_management.ipc.messages import HeldComponentSnapshot

_CHECKPOINT_KIND = "checkpoint"
"""The component kind whose identity is the bare horde model name, so it is the staged-model set."""


@dataclass(frozen=True)
class _ProcessResidency:
    """One process's most recent residency snapshot, tagged with the launch that produced it.

    ``launch_identifier`` is stored so a late report from a since-replaced generation of the same process id
    (a lower launch identifier, since the parent's launch counter only increases) is rejected rather than
    overwriting the live generation's residency.
    """

    launch_identifier: int
    held: tuple[HeldComponentSnapshot, ...]


class ComponentResidencyMap:
    """A mapping of process id to the component-cache entries that process last reported holding resident.

    Updated from memory reports (:meth:`update_from_report`) and expired on process death/recycle
    (:meth:`expire_process`), exactly as the whole-model :class:`HordeModelMap` is. Query it for the staged
    checkpoint set (:meth:`checkpoint_models_held_on`) and for the identities held across the worker
    (:meth:`identities_held`).
    """

    def __init__(self) -> None:
        """Initialise an empty map."""
        self._by_process: dict[int, _ProcessResidency] = {}

    def update_from_report(
        self,
        process_id: int,
        launch_identifier: int,
        held: list[HeldComponentSnapshot] | None,
    ) -> None:
        """Record the residency snapshot from one process's memory report, rejecting stale launches.

        ``held`` is None when the reporting process carries no residency data (an older child, or a process
        with no loaded component cache); such a report leaves any existing entry untouched rather than
        clearing it, since None means "no data", not "nothing resident". A real cache-bearing child always
        reports a list (empty when its cache is empty), which replaces the stored snapshot. A report whose
        ``launch_identifier`` is older than the one already recorded for this process id is a late message
        from a replaced generation and is dropped.

        Args:
            process_id: The reporting process's id.
            launch_identifier: The launch identifier the report carried (the parent's monotonic launch
                counter at that process's spawn).
            held: The reported resident component snapshots, or None when the process reports no data.
        """
        if held is None:
            return

        existing = self._by_process.get(process_id)
        if existing is not None and launch_identifier < existing.launch_identifier:
            logger.debug(
                f"Dropping a stale component-residency report from process {process_id} (launch "
                f"{launch_identifier} < {existing.launch_identifier}); it was replaced.",
            )
            return

        self._by_process[process_id] = _ProcessResidency(
            launch_identifier=launch_identifier,
            held=tuple(held),
        )

    def expire_process(self, process_id: int) -> None:
        """Forget a process's residency (called when the process dies or is recycled)."""
        self._by_process.pop(process_id, None)

    def identities_held(self, kind: str | None = None) -> frozenset[str]:
        """Return every component identity held across all processes, optionally filtered to one ``kind``."""
        return frozenset(
            snapshot.identity
            for residency in self._by_process.values()
            for snapshot in residency.held
            if kind is None or snapshot.kind == kind
        )

    def checkpoint_models_held_on(self, process_ids: Collection[int]) -> frozenset[str]:
        """Return the checkpoint identities (bare horde model names) held on the given processes.

        A checkpoint entry's identity is the bare horde model name, so its checkpoint-kind identities are the
        worker's RAM-staged model set with no sidecar lookup. Restricting to the given processes lets a caller
        ask only about the slots it cares about (for example the live, healthy inference processes eligible to
        sample).
        """
        wanted = set(process_ids)
        return frozenset(
            snapshot.identity
            for process_id, residency in self._by_process.items()
            if process_id in wanted
            for snapshot in residency.held
            if snapshot.kind == _CHECKPOINT_KIND
        )
