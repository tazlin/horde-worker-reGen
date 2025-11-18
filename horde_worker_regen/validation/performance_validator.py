"""Performance mode validation logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData


class PerformanceModeValidator:
    """Validates and adjusts performance mode settings."""

    @staticmethod
    def validate_and_adjust_performance_modes(bridge_data: reGenBridgeData) -> reGenBridgeData:
        """Validate the performance modes and set the appropriate values.

        Args:
            bridge_data: The bridge data configuration to validate and adjust.

        Returns:
            The bridge data with performance modes adjusted as needed.
        """
        # Validate max_threads and queue_size compatibility
        if bridge_data.max_threads >= 2 and bridge_data.queue_size > 3:
            bridge_data.queue_size = 3
            logger.warning(
                "The queue_size value has been set to 3 because the max_threads value is 2.",
            )

        # Adjust process timeout for high performance mode
        if bridge_data.high_performance_mode:
            PerformanceModeValidator._adjust_high_performance_mode(bridge_data)
        elif bridge_data.moderate_performance_mode:
            PerformanceModeValidator._adjust_moderate_performance_mode(bridge_data)

        # Handle extra slow worker mode
        if bridge_data.extra_slow_worker:
            PerformanceModeValidator._adjust_extra_slow_worker(bridge_data)

        # Handle memory modes
        if bridge_data.very_high_memory_mode and not bridge_data.high_memory_mode:
            bridge_data.high_memory_mode = True
            logger.debug(
                "Very high memory mode is enabled, so the high_memory_mode value has been set to True.",
            )

        if bridge_data.high_memory_mode:
            PerformanceModeValidator._validate_high_memory_mode(bridge_data)

        return bridge_data

    @staticmethod
    def _adjust_high_performance_mode(bridge_data: reGenBridgeData) -> None:
        """Adjust settings for high performance mode."""
        process_timeout_changed_message = (
            "High performance mode is enabled, so the process_timeout value has "
            f"been set to 1/3 of the default value. The new value is {bridge_data.process_timeout}."
        )
        default_process_timeout = bridge_data.model_fields["process_timeout"].default

        if bridge_data.process_timeout == default_process_timeout:
            logger.debug(process_timeout_changed_message)
        else:
            logger.warning(process_timeout_changed_message)

        bridge_data.process_timeout = default_process_timeout // 3

    @staticmethod
    def _adjust_moderate_performance_mode(bridge_data: reGenBridgeData) -> None:
        """Adjust settings for moderate performance mode."""
        process_timeout_changed_message = (
            "Moderate performance mode is enabled, so the process_timeout value has "
            f"been set to 1/2 of the default value. The new value is {bridge_data.process_timeout}."
        )
        default_process_timeout = bridge_data.model_fields["process_timeout"].default

        if bridge_data.process_timeout == default_process_timeout:
            logger.debug(process_timeout_changed_message)
        else:
            logger.warning(process_timeout_changed_message)

        bridge_data.process_timeout = default_process_timeout // 2

    @staticmethod
    def _adjust_extra_slow_worker(bridge_data: reGenBridgeData) -> None:
        """Adjust settings for extra slow worker mode."""
        if bridge_data.high_performance_mode:
            bridge_data.high_performance_mode = False
            logger.warning(
                "Extra slow worker is enabled, so the high_performance_mode value has been set to False.",
            )
        if bridge_data.moderate_performance_mode:
            bridge_data.moderate_performance_mode = False
            logger.warning(
                "Extra slow worker is enabled, so the moderate_performance_mode value has been set to False.",
            )
        if bridge_data.high_memory_mode:
            bridge_data.high_memory_mode = False
            logger.warning(
                "Extra slow worker is enabled, so the high_memory_mode value has been set to False.",
            )
        if bridge_data.very_high_memory_mode:
            bridge_data.very_high_memory_mode = False
            logger.warning(
                "Extra slow worker is enabled, so the very_high_memory_mode value has been set to False.",
            )
        if bridge_data.queue_size > 0:
            bridge_data.queue_size = 0
            logger.warning(
                "Extra slow worker is enabled, so the queue_size value has been set to 0. "
                "This behavior may change in the future.",
            )
        if bridge_data.max_threads > 1:
            bridge_data.max_threads = 1
            logger.warning(
                "Extra slow worker is enabled, so the max_threads value has been set to 1. "
                "This behavior may change in the future.",
            )
        if bridge_data.preload_timeout < 120:
            bridge_data.preload_timeout = 120
            logger.warning(
                "Extra slow worker is enabled, so the preload_timeout value has been set to 120. "
                "This behavior may change in the future.",
            )

    @staticmethod
    def _validate_high_memory_mode(bridge_data: reGenBridgeData) -> None:
        """Validate and warn about high memory mode settings."""
        if bridge_data.queue_size == 0:
            logger.warning(
                "High memory mode is enabled, you should consider setting queue_size to 1 or higher. "
                "Increasing this value increases system memory usage. See the bridgeData_template.yaml for more "
                "information.",
            )

        if bridge_data.unload_models_from_vram_often:
            logger.warning(
                "High memory mode is enabled, you should consider setting unload_models_from_vram_often to False.",
            )

        if bridge_data.cycle_process_on_model_change:
            bridge_data.cycle_process_on_model_change = False
            logger.warning(
                "High memory mode is enabled, so the cycle_process_on_model_change value has been set to False.",
            )
