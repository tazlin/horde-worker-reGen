"""Opt-in debug attach helpers for worker child processes."""

from __future__ import annotations

import os

_DEBUG_ENABLED_ENV_VARS = ("DEBUG_HORDE_WORKER_PROCESSES", "DEBUG_HARNESS_WORKERS")
_DEBUG_BASE_PORT_ENV_VARS = ("DEBUG_HORDE_WORKER_PROCESS_PORT", "DEBUG_HARNESS_WORKER_PORT")


def _env_var_enabled(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def _get_debug_base_port() -> int:
    for env_var_name in _DEBUG_BASE_PORT_ENV_VARS:
        env_var_value = os.getenv(env_var_name)
        if env_var_value is not None:
            return int(env_var_value)

    return 5680


def maybe_wait_for_process_debugger(process_id: int, process_kind: str) -> None:
    """Open an opt-in debugpy attach point for worker child processes."""
    if not any(_env_var_enabled(env_var_name) for env_var_name in _DEBUG_ENABLED_ENV_VARS):
        return

    import debugpy

    port = _get_debug_base_port() + process_id

    print(f"{process_kind} process {process_id} waiting for debugger on 127.0.0.1:{port}", flush=True)
    debugpy.listen(("127.0.0.1", port))
    debugpy.wait_for_client()
