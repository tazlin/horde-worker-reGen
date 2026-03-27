"""Central telemetry configuration using Pydantic Logfire.

Logfire works without any cloud setup (``send_to_logfire=False`` is default).
Users opt in to cloud export by setting the ``LOGFIRE_TOKEN`` environment variable.
"""

from __future__ import annotations

import logfire
from loguru import logger

_telemetry_configured: bool = False


def configure_telemetry() -> None:
    """Configure telemetry in the main (driver) process.

    Called once from ``run_worker.py`` after logging is set up.
    """
    global _telemetry_configured  # noqa: PLW0603
    if _telemetry_configured:
        return

    logfire.configure(
        send_to_logfire="if-token-present",
        service_name="horde-worker-regen",
    )
    logfire.instrument_system_metrics()

    # Bridge loguru → logfire so spans can capture structured logs.
    logger.add(logfire.loguru_handler(), format="{message}")

    _telemetry_configured = True


def configure_child_telemetry(process_id: int) -> None:
    """Configure telemetry in a spawned child process.

    Args:
        process_id: The logical process id assigned by the worker.
    """
    logfire.configure(
        send_to_logfire="if-token-present",
        service_name=f"horde-worker-regen-child-{process_id}",
    )
