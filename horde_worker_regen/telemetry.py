"""Central telemetry configuration using Pydantic Logfire.

Logfire works without any cloud setup (``send_to_logfire=False`` is default).
Users opt in to cloud export by setting the ``LOGFIRE_TOKEN`` environment variable.
"""

from __future__ import annotations

import os

import logfire
from loguru import logger

_telemetry_configured: bool = False


def claim_logfire_ownership() -> None:
    """Tell hordelib the worker owns logfire/loguru configuration in this process.

    Must be called before hordelib is imported: hordelib's import-time
    ``initialize_logfire()`` would otherwise re-configure logfire and replace our
    loguru handlers.
    """
    os.environ["HORDELIB_EXTERNAL_LOGFIRE"] = "1"


def configure_telemetry() -> None:
    """Configure telemetry in the main (driver) process.

    Called once from ``run_worker.py`` after logging is set up.
    """
    global _telemetry_configured  # noqa: PLW0603
    if _telemetry_configured:
        return

    claim_logfire_ownership()

    logfire.configure(
        send_to_logfire="if-token-present",
        service_name="horde-worker-regen",
    )

    try:
        logfire.instrument_system_metrics()
    except RuntimeError as e:
        # The system-metrics instrumentation extras are optional; telemetry still works without them.
        logger.debug(f"System metrics instrumentation unavailable: {e}")

    # Bridge loguru → logfire so spans can capture structured logs.
    # loguru_handler() returns the kwargs for logger.add (sink, format, etc).
    logger.add(**logfire.loguru_handler())

    _telemetry_configured = True


def instrument_aiohttp_client() -> None:
    """Instrument aiohttp client sessions, tolerating missing optional instrumentation extras."""
    try:
        logfire.instrument_aiohttp_client()
    except RuntimeError as e:
        logger.debug(f"aiohttp client instrumentation unavailable: {e}")


def configure_child_telemetry(process_id: int) -> None:
    """Configure telemetry in a spawned child process.

    Args:
        process_id: The logical process id assigned by the worker.
    """
    claim_logfire_ownership()

    logfire.configure(
        send_to_logfire="if-token-present",
        service_name=f"horde-worker-regen-child-{process_id}",
    )
