"""Reload live bridge data without stalling the worker event loop."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable

from horde_model_reference.model_reference_manager import ModelReferenceManager
from loguru import logger
from pydantic import ValidationError

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.bridge_data.load_config import BridgeDataLoader
from horde_worker_regen.consts import BRIDGE_CONFIG_FILENAME
from horde_worker_regen.process_management.config.worker_state import WorkerState


class BridgeDataReloader:
    """Reload bridge data from disk and apply it on the event loop."""

    def __init__(
        self,
        *,
        state: WorkerState,
        bridge_data_provider: Callable[[], reGenBridgeData],
        model_reference_manager_provider: Callable[[], ModelReferenceManager | None],
        apply_bridge_data: Callable[[reGenBridgeData], None],
        enable_performance_mode: Callable[[], None],
        shutdown_callback: Callable[[], None],
        loop_interval: float = 1.0,
    ) -> None:
        """Initialize the bridge-data reloader.

        Args:
            state: Shared worker state used to stop the watcher during shutdown.
            bridge_data_provider: Return the current live bridge data.
            model_reference_manager_provider: Return the model reference manager used by the loader.
            apply_bridge_data: Apply a freshly loaded bridge-data object on the event loop.
            enable_performance_mode: Re-apply performance-mode thresholds after reload.
            shutdown_callback: Start worker shutdown after cancellation.
            loop_interval: Seconds between file watcher checks.
        """
        self._state = state
        self._bridge_data_provider = bridge_data_provider
        self._model_reference_manager_provider = model_reference_manager_provider
        self._apply_bridge_data = apply_bridge_data
        self._enable_performance_mode = enable_performance_mode
        self._shutdown_callback = shutdown_callback
        self._loop_interval = loop_interval

        self._last_reload_time = 0.0
        self._last_modified_time = 0.0
        self._reload_lock: asyncio.Lock | None = None
        self._reload_tasks: set[asyncio.Task[None]] = set()

    def load_bridge_data_blocking(self) -> reGenBridgeData | None:
        """Read and resolve bridge data from disk, returning the new model when available."""
        if self._bridge_data_provider()._loaded_from_env_vars:
            return None

        model_reference_manager = self._model_reference_manager_provider()
        if model_reference_manager is None:
            logger.debug("No model reference manager available; skipping bridge data reload")
            return None

        try:
            return BridgeDataLoader.load(
                file_path=BRIDGE_CONFIG_FILENAME,
                horde_model_reference_manager=model_reference_manager,
            )
        except Exception as reload_error:
            logger.debug(reload_error)

            if "No such file or directory" in str(reload_error):
                logger.error(f"Could not find {BRIDGE_CONFIG_FILENAME}. Please create it and try again.")

            if isinstance(reload_error, ValidationError):
                logger.error(f"The following fields in {BRIDGE_CONFIG_FILENAME} failed validation:")
                for validation_error in reload_error.errors():
                    logger.error(f"{validation_error['loc'][0]}: {validation_error['msg']}")

            return None

    def get_bridge_data_from_disk(self) -> None:
        """Load bridge data from disk synchronously and apply it when valid."""
        bridge_data = self.load_bridge_data_blocking()
        if bridge_data is not None:
            self._apply_bridge_data(bridge_data)

    async def reload_bridge_data_off_loop(self) -> None:
        """Reload bridge data without stalling the event loop."""
        if self._reload_lock is None:
            self._reload_lock = asyncio.Lock()
        async with self._reload_lock:
            bridge_data = await asyncio.to_thread(self.load_bridge_data_blocking)
            if bridge_data is not None:
                self._apply_bridge_data(bridge_data)

    def schedule_config_reload(self) -> None:
        """Kick off an off-loop config reload from synchronous on-loop code."""
        task = asyncio.create_task(self.reload_bridge_data_off_loop())
        self._reload_tasks.add(task)
        task.add_done_callback(self._reload_tasks.discard)

    async def bridge_data_loop(self) -> None:
        """Watch bridgeData.yaml and reload it when the file changes."""
        while True:
            try:
                if self._state.shutting_down:
                    break

                self._last_modified_time = os.path.getmtime(BRIDGE_CONFIG_FILENAME)

                if self._last_reload_time < self._last_modified_time:
                    logger.info(f"Reloading {BRIDGE_CONFIG_FILENAME}")
                    await self.reload_bridge_data_off_loop()
                    self._last_reload_time = os.path.getmtime(BRIDGE_CONFIG_FILENAME)
                    logger.success(f"Reloaded {BRIDGE_CONFIG_FILENAME}")
                    self._enable_performance_mode()
                await asyncio.sleep(self._loop_interval)
            except asyncio.CancelledError as cancel_error:
                self._shutdown_callback()
                logger.debug(f"CancelledError: {cancel_error}")
            except Exception as reload_error:
                logger.warning(f"Error while watching {BRIDGE_CONFIG_FILENAME} for changes: {reload_error}")
                await asyncio.sleep(self._loop_interval)
