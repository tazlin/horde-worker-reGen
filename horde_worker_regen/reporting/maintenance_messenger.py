"""Maintenance mode messaging."""

from __future__ import annotations

from loguru import logger


class MaintenanceModeMessenger:
    """Handles display of maintenance mode messages to users."""

    @staticmethod
    def print_maintenance_mode_messages() -> None:
        """Print the information about maintenance mode to the user."""

        def warning_function_no_format(x: str) -> None:
            """Print a warning message with consistent formatting."""
            logger.opt(ansi=True, raw=True).warning(
                "<fg #f1c40f>" + x + "</>\n",
            )

        warning_function_no_format(
            "Your worker is in maintenance mode. Set your API key at https://tinybots.net/artbot/settings, "
            "click save, then click unpause on https://tinybots.net/artbot/settings?panel=workers while the worker "
            "is running to clear this message.",
        )
        warning_function_no_format(
            "If you didn't expect seeing this message, its probable that the worker "
            "dropped too many jobs, and the server stepped in to prevent further jobs from being "
            "dropped. Please check the logs above, and possibly your logs/ folder as well.",
        )
        warning_function_no_format("Common reasons for forced maintenance mode are: ")
        warning_function_no_format("  - `max_threads` is too high.")
        warning_function_no_format("  - `queue_size` is too high.")
        warning_function_no_format("  - `max_batch` is too high.")
        warning_function_no_format("  - `max_power` is too high.")
        warning_function_no_format("  - The worker can't handle, SDXL, Cascade, or Flux models.")
        warning_function_no_format(
            "  - If you have the equivalent GPU of a 1070 or less, set"
            " limit_max_steps or extra_slow_worker. "
            "This should only be done as a last resort.",
        )

        warning_function_no_format(
            "If you continue to see this message, come to the official discord (https://discord.gg/3DxrhksKzn).",
        )
