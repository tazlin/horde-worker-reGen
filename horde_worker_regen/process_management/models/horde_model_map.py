"""A mapping of horde model names to ModelInfo objects."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import PrivateAttr, RootModel

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeProcessState,
    ModelInfo,
    ModelLoadState,
)

if TYPE_CHECKING:
    from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo

PRELOAD_FIRST_REPORT_GRACE_SECONDS = 5.0
"""How long a just-sent preload may still look idle before its first child state report arrives.

The parent records the model as ``LOADING`` immediately after sending ``PRELOAD_MODEL``, but the child
may still read as ``WAITING_FOR_JOB`` until it drains the control pipe and publishes its first preload
state. This short grace keeps stale-entry cleanup from expiring a healthy, just-sent preload while still
letting genuinely abandoned loading entries clear promptly.
"""

_LOADING_OWNER_STATES = frozenset(
    {
        HordeProcessState.PROCESS_STARTING,
        HordeProcessState.DOWNLOADING_MODEL,
        HordeProcessState.PRELOADING_MODEL,
        HordeProcessState.UNLOADED_MODEL_FROM_RAM,
    },
)
"""The process states in which a ``LOADING`` entry can truthfully still be loading."""


class StaleEntryReason(enum.Enum):
    """Why a model-map entry no longer describes weights any process holds or is loading."""

    PROCESS_GONE = enum.auto()
    """The owning process left the pool."""
    LOADING_ABANDONED = enum.auto()
    """The entry says loading, but the owner is in a state that cannot be loading and no preload was just sent."""
    DISPLACED = enum.auto()
    """The owning slot now holds a different model."""


@dataclass(frozen=True)
class ExpiredEntry:
    """One entry the stale sweep removed, with what the sweep saw."""

    model: str
    process_id: int
    reason: StaleEntryReason
    process_state: HordeProcessState | None
    """The owner's state when the entry was judged, or None when the owner was gone."""
    holder_model: str | None
    """The model the owning slot holds instead, for a displaced entry."""


class HordeModelMap(RootModel[dict[str, ModelInfo]]):
    """A mapping of horde model names to `ModelInfo` objects. Contains some helper methods."""

    _blank_identity_warned: bool = PrivateAttr(default=False)
    """Whether the refusal to key an entry on a blank model name has already been surfaced."""

    def weights_resident_on_process(self, model_name: str | None, process_info: HordeProcessInfo | None) -> bool:
        """Whether ``model_name``'s weights are in VRAM on ``process_info``'s slot right now.

        Every consumer answers this question here: the admission resident-weight credit, the arbiter's
        ``candidate_already_resident`` no-op admit, and the resident-footprint observer. The parent's retention
        record (``retained_resident_model``) is checked first because it is the only record that covers weights
        held between jobs: a slot that finished under a retention grant reports its weights back in system
        RAM, and a disaggregated sampler reports no transition at all.

        Otherwise the map's load state decides, for the matching process. The committed floor charges a slot's
        weights by the process map's ``loaded_horde_model_name``, and the map's process pointer can briefly lag
        that record, so a process that itself reports the model loaded is also credited when the map says the
        model is in VRAM. One map state is not evidence: a lane in ``INFERENCE_PRIMED`` still has its job's
        weights in system RAM until the clearance lease admits it, but the child marks the model ``IN_USE`` as
        soon as it takes the job. Crediting that would price the clearance as activation only and admit a
        whole-card model into a slot that cannot hold it. So a primed lane's job model is never resident
        unless retention holds it.
        """
        if model_name is None or process_info is None:
            return False
        if process_info.retained_resident_model == model_name:
            return True
        if process_info.last_process_state is HordeProcessState.INFERENCE_PRIMED:
            return False
        model_info = self.root.get(model_name)
        if model_info is None or model_info.horde_model_load_state not in (
            ModelLoadState.LOADED_IN_VRAM,
            ModelLoadState.IN_USE,
        ):
            return False
        if model_info.process_id == process_info.process_id:
            return True
        return process_info.loaded_horde_model_name == model_name

    def update_entry(
        self,
        horde_model_name: str,
        *,
        load_state: ModelLoadState | None = None,
        process_id: int | None = None,
    ) -> None:
        """Update the entry for the given model name. If the model does not exist, it will be created.

        A blank (empty or whitespace-only) name is refused without mutating the map. Such a name identifies no
        model, so an entry keyed on it can never be matched by a real load or cleared by a real unload; it would
        persist as a residency this map reports for a model that does not exist. The refusal is surfaced once,
        since the condition repeats for as long as whatever produced the name keeps producing it.

        Args:
            horde_model_name (str): The (horde) name of the model to update.
            load_state (ModelLoadState | None, optional): The load state of the model. Defaults to None.
            process_id (int | None, optional): The process ID of the process that has this model loaded. \
                Defaults to None.

        Raises:
            ValueError: If the process_id is None and the model does not exist.
            ValueError: If the load_state is None and the model does not exist.
        """
        if not horde_model_name or not horde_model_name.strip():
            if not self._blank_identity_warned:
                self._blank_identity_warned = True
                logger.warning(
                    f"Refusing to record a model load state against a blank model name (got "
                    f"{horde_model_name!r}, process {process_id}); the model map is left unchanged.",
                )
            return

        if horde_model_name not in self.root:
            if process_id is None:
                raise ValueError("process_id must be provided when adding a new model to the map")
            if load_state is None:
                raise ValueError("model_load_state must be provided when adding a new model to the map")

            self.root[horde_model_name] = ModelInfo(
                horde_model_name=horde_model_name,
                horde_model_load_state=load_state,
                process_id=process_id,
            )

        if load_state is not None:
            self.root[horde_model_name].horde_model_load_state = load_state
            logger.debug(f"Updated load state for {horde_model_name} to {load_state}")

        if process_id is not None:
            self.root[horde_model_name].process_id = process_id
            logger.debug(f"Updated process ID for {horde_model_name} to {process_id}")

    def expire_entry(self, horde_model_name: str) -> ModelInfo | None:
        """Removes information about a horde model.

        :param horde_model_name: Name of model to remove
        :return: model name if removed; 'none' string otherwise
        """
        return self.root.pop(horde_model_name, None)

    def expire_entries_for_process(self, process_id: int) -> list[str]:
        """Remove every model entry that is loaded on (or loading into) the given process.

        Used when a process dies: a model the scheduler believes is ``LOADING`` (or loaded) on a now-dead
        slot is otherwise treated as resident forever (``preload_models`` skips any model already in the
        loaded/loading set), so the pending job that wanted it is never re-preloaded onto a fresh slot.
        Keying off ``process_id`` rather than the dead slot's ``loaded_horde_model_name`` is essential
        because that name is cleared the moment the child reports ``PROCESS_ENDING``, leaving the stale
        map entry as the only remaining record of the wedge.

        Returns:
            The names of the models whose entries were removed.
        """
        expired = [name for name, info in self.root.items() if info.process_id == process_id]
        for name in expired:
            self.root.pop(name, None)
        return expired

    def expire_stale_entries(self, processes: Mapping[int, HordeProcessInfo], *, now: float) -> list[ExpiredEntry]:
        """Remove every entry whose owning process can no longer be holding or loading that model.

        A slot holds one model at a time, so an entry naming a slot that now names a different model is a
        record of weights nothing holds. It is not inert: the preload pass reads the map as part of the
        already-loaded set, so a displaced model's surviving entry makes its pending job look served and the
        job is never staged onto a free slot. A ``LOADING`` entry whose owner is in a state that cannot be
        loading is likewise stale, except inside :data:`PRELOAD_FIRST_REPORT_GRACE_SECONDS` of a preload sent
        to that owner for that model, when the child has simply not reported yet.

        Returns:
            What was removed and why, in map order, for the caller to log and act on.
        """
        expired: list[ExpiredEntry] = []
        for model_name, model_info in list(self.root.items()):
            process_info = processes.get(model_info.process_id)
            if process_info is None:
                self.root.pop(model_name, None)
                expired.append(
                    ExpiredEntry(model_name, model_info.process_id, StaleEntryReason.PROCESS_GONE, None, None),
                )
                continue
            recent_preload_request = (
                model_info.horde_model_load_state == ModelLoadState.LOADING
                and process_info.last_control_flag == HordeControlFlag.PRELOAD_MODEL
                and process_info.loaded_horde_model_name == model_name
                and (now - process_info.last_preload_requested_at) <= PRELOAD_FIRST_REPORT_GRACE_SECONDS
            )
            if (
                model_info.horde_model_load_state == ModelLoadState.LOADING
                and process_info.last_process_state not in _LOADING_OWNER_STATES
                and not recent_preload_request
            ):
                self.root.pop(model_name, None)
                expired.append(
                    ExpiredEntry(
                        model_name,
                        process_info.process_id,
                        StaleEntryReason.LOADING_ABANDONED,
                        process_info.last_process_state,
                        None,
                    ),
                )
                continue
            if process_info.holds_different_model(model_name):
                self.root.pop(model_name, None)
                expired.append(
                    ExpiredEntry(
                        model_name,
                        process_info.process_id,
                        StaleEntryReason.DISPLACED,
                        process_info.last_process_state,
                        process_info.loaded_horde_model_name,
                    ),
                )
        return expired

    def is_model_loaded(self, horde_model_name: str) -> bool:
        """Return true if the given model is loaded in any process."""
        if horde_model_name not in self.root:
            return False
        return self.root[horde_model_name].horde_model_load_state.is_loaded()

    def is_model_loading(self, horde_model_name: str) -> bool:
        """Return true if the given model is currently being loaded in any process."""
        if horde_model_name not in self.root:
            return False
        return self.root[horde_model_name].horde_model_load_state == ModelLoadState.LOADING
