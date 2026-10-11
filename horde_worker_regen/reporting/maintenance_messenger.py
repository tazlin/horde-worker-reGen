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
            "Your worker is in maintenance mode, so the horde gives it no jobs except your own. To leave "
            "maintenance mode, enter your API key at https://tinybots.net/artbot/settings and click save, then "
            "click unpause at https://tinybots.net/artbot/settings?panel=workers while the worker is running.",
        )
        warning_function_no_format(
            "If you did not turn maintenance mode on, the horde most likely did because the worker dropped too "
            "many jobs. Check the messages above and the logs/ folder for the cause.",
        )
        warning_function_no_format("Common reasons the horde forces maintenance mode:")
        warning_function_no_format("  - `max_threads` is too high.")
        warning_function_no_format("  - `queue_size` is too high.")
        warning_function_no_format("  - `max_batch` is too high.")
        warning_function_no_format("  - `max_power` is too high.")
        warning_function_no_format("  - This machine cannot run the SDXL, Cascade or Flux models the worker offers.")
        warning_function_no_format(
            "  - The GPU is a GTX 1070 or slower. As a last resort, set `limit_max_steps` or `extra_slow_worker`.",
        )

        warning_function_no_format(
            "If this message keeps appearing, ask for help on the official Discord (https://discord.gg/3DxrhksKzn).",
        )
