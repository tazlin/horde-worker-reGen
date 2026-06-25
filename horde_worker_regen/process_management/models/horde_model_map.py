"""A mapping of horde model names to ModelInfo objects."""

from __future__ import annotations

from loguru import logger
from pydantic import RootModel

from horde_worker_regen.process_management.ipc.messages import ModelInfo, ModelLoadState


class HordeModelMap(RootModel[dict[str, ModelInfo]]):
    """A mapping of horde model names to `ModelInfo` objects. Contains some helper methods."""

    def update_entry(
        self,
        horde_model_name: str,
        *,
        load_state: ModelLoadState | None = None,
        process_id: int | None = None,
    ) -> None:
        """Update the entry for the given model name. If the model does not exist, it will be created.

        Args:
            horde_model_name (str): The (horde) name of the model to update.
            load_state (ModelLoadState | None, optional): The load state of the model. Defaults to None.
            process_id (int | None, optional): The process ID of the process that has this model loaded. \
                Defaults to None.

        Raises:
            ValueError: If the process_id is None and the model does not exist.
            ValueError: If the load_state is None and the model does not exist.
        """
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
