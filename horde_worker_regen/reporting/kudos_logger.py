"""Kudos reporting and logging."""

from __future__ import annotations

from horde_sdk.ai_horde_api.apimodels import UserDetailsResponse
from loguru import logger


class KudosLogger:
    """Handles kudos logging and display."""

    @staticmethod
    def log_kudos_info(
        kudos_info_string: str,
        kudos_generated_this_session: float,
        user_info: UserDetailsResponse | None,
        limited_console_messages: bool,
    ) -> None:
        """Log the kudos information string.

        Args:
            kudos_info_string: The kudos information string to log.
            kudos_generated_this_session: The kudos generated in the current session.
            user_info: The user information from the API.
            limited_console_messages: Whether to use limited console messages (success level).
        """
        log_function = logger.opt(ansi=True).info

        if limited_console_messages:
            log_function = logger.opt(ansi=True).success

        if kudos_generated_this_session > 0:
            log_function(
                f"<fg #7dcea0>{kudos_info_string}</>",
            )

        if user_info is not None and user_info.kudos_details is not None:
            log_function(
                "<fg #7dcea0>"
                f"Total Kudos Accumulated: {user_info.kudos_details.accumulated:,.2f} "
                f"(all workers for {user_info.username})"
                "</>",
            )
            if user_info.kudos_details.accumulated is not None and user_info.kudos_details.accumulated < 0:
                log_function(
                    "<fg #7dcea0>"
                    "Negative kudos means you've requested more than you've earned. This can be normal."
                    "</>",
                )
