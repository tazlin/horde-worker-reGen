"""Single source of truth for which image models the worker wants on disk, and why.

Public members:
    ``DesiredState`` -- parent-side authority for the desired on-disk image-model set.
    ``ReconcilePlan`` -- the immutable diff a reconcile produces (what to fetch, what to cancel).

The worker had several independent download triggers, only one of which reconciled: the config path sent an
authoritative set that pruned anything not configured, while the on-demand picker sent an additive request
with no authoritative set, so a picker-added model was silently cancelled by the next config reconcile.
Funnelling every trigger through one desired set keeps the picker and config from diverging. This module is
parent-side and deliberately torch-free (it is covered by the orchestrator torch-free tripwire).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ReconcilePlan:
    """The diff between the desired on-disk image-model set and what is present or in flight."""

    desired: frozenset[str]
    """The full set the worker wants on disk: configured models unioned with picker additions."""
    to_fetch: tuple[str, ...]
    """Desired models not yet present on disk, sorted. Sent to the download process to enqueue."""
    to_cancel: tuple[str, ...]
    """Queued/in-flight downloads no longer desired, sorted. The download process prunes these from its
    queue and aborts an in-flight one; on-disk files are never deleted, since pruning is queue-only."""

    @property
    def has_work(self) -> bool:
        """Whether the plan asks the download process to do anything (fetch or cancel)."""
        return bool(self.to_fetch or self.to_cancel)


class DesiredState:
    """Parent-side authority for which image models should be on disk, and why.

    The desired set is the resolved configured set (``bridge_data.image_models_to_load``, passed in fresh on
    every reconcile so it is never stale) unioned with the operator's transient picker additions (held here).
    The picker edits this one set rather than sending a parallel, un-reconciled request, so config and picker
    can never diverge. Picker additions live only in memory: a downloaded one stays on disk across a restart,
    but the worker then reverts to the configured-only desired set, which is the intended transient pre-fetch.
    """

    _picker_additions: set[str]

    def __init__(self) -> None:
        """Initialise with no picker additions; the desired set is whatever config resolves to."""
        self._picker_additions = set()

    @property
    def picker_additions(self) -> frozenset[str]:
        """The models the operator has added via the picker, on top of the configured set."""
        return frozenset(self._picker_additions)

    def add_picker_models(self, model_names: Iterable[str]) -> None:
        """Add operator-chosen models to the desired set (the picker's "download now")."""
        self._picker_additions.update(model_names)

    def clear_picker_models(self, model_names: Iterable[str] | None = None) -> None:
        """Drop picker additions: a specific subset, or all of them when ``model_names`` is None."""
        if model_names is None:
            self._picker_additions.clear()
        else:
            self._picker_additions.difference_update(model_names)

    def reconcile(
        self,
        *,
        configured: Iterable[str],
        present: Iterable[str],
        in_flight: Iterable[str] = (),
    ) -> ReconcilePlan:
        """Diff the desired set (configured + picker additions) against disk and the download queue.

        ``present`` is what the download process reports on disk; ``in_flight`` is what it has queued or is
        downloading. ``to_cancel`` is the in-flight subset no longer desired, so a removal prunes the queue
        without touching files.
        """
        desired = frozenset(configured) | frozenset(self._picker_additions)
        present_set = frozenset(present)
        in_flight_set = frozenset(in_flight)
        to_fetch = tuple(sorted(desired - present_set))
        to_cancel = tuple(sorted(in_flight_set - desired))
        return ReconcilePlan(desired=desired, to_fetch=to_fetch, to_cancel=to_cancel)
