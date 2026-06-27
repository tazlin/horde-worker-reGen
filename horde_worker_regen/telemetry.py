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


TELEMETRY_OPT_IN_ENV_VAR = "AIWORKER_REGEN_ENABLE_TELEMETRY"
"""Set to a truthy value to opt this worker into OpenTelemetry tracing (off by default)."""


def telemetry_enabled() -> bool:
    """Whether OpenTelemetry tracing is explicitly opted in for this worker.

    Tracing is OFF by default. It is enabled ONLY by this worker's own flag
    (:data:`TELEMETRY_OPT_IN_ENV_VAR`); never implicitly by ambient ``OTEL_*`` / ``LOGFIRE_*``
    settings a developer may carry in their shell or system environment. Opt in only when a
    collector (Jaeger/Prometheus) or the Logfire cloud is actually running to consume the spans.
    """
    return os.getenv(TELEMETRY_OPT_IN_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")


def enforce_telemetry_default_off() -> None:
    """Force the OpenTelemetry SDK off unless tracing is explicitly opted in.

    hordelib instruments every ComfyUI internal op, creating hundreds of spans per job. With no
    collector the SDK still builds and processes those spans on threads that contend for the GIL,
    which measurably starves the inference loop and depresses GPU duty cycle (≈1s/job of stall was
    measured on an sd15 soak; disabling it raised throughput ~20% and duty-cycle coverage from
    ~0.86 to ~0.93). The shipped worker therefore disables tracing *explicitly*, hard-overriding
    any ambient ``OTEL_SDK_DISABLED=false`` or ``OTEL_EXPORTER_OTLP_*`` a developer has set
    system-wide, rather than hoping the environment is clean.

    Call as early as possible (before logfire/hordelib import) so the kill switch is read when the
    OTel SDK initialises; the env var is inherited by spawned child processes.
    """
    if telemetry_enabled():
        return
    os.environ["OTEL_SDK_DISABLED"] = "true"


def configure_telemetry() -> None:
    """Configure telemetry in the main (driver) process.

    Called once from ``run_worker.py`` after logging is set up.
    """
    global _telemetry_configured  # noqa: PLW0603
    if _telemetry_configured:
        return

    claim_logfire_ownership()
    enforce_telemetry_default_off()

    logfire.configure(
        send_to_logfire="if-token-present",
        service_name="horde-worker-regen",
        console=False,
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
    enforce_telemetry_default_off()

    logfire.configure(
        send_to_logfire="if-token-present",
        service_name=f"horde-worker-regen-child-{process_id}",
        console=False,
    )
