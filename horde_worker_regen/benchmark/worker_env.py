"""Apply the worker's ``bridgeData.yaml`` environment for real-mode benchmarking.

The harness builds *synthetic* bridge data and forces ``_loaded_from_env_vars`` so it never reads
``bridgeData.yaml`` (see ``horde_worker_regen.harness.build_harness_bridge_data``). A normal worker
run, by contrast, sets ``AIWORKER_CACHE_HOME`` from the config's ``cache_home`` at startup
(``horde_worker_regen.load_env_vars``). Without that, the real inference children fall back to
hordelib's CWD-relative ``./models`` weights root (``UserSettings.get_model_directory``), find no
checkpoints, and exit with "No models available" -- wedging the first level until its timeout.

Calling :func:`ensure_worker_env` once at the top of the benchmark process tree (CLI and the
isolated level runner) restores parity with a real worker run, so the benchmark measures the user's
actual model directory (and honours their civitai token / lora cache for the download levels).
"""

from __future__ import annotations

import os

from loguru import logger


def ensure_worker_env(process_mode: str) -> None:
    """Best-effort: apply the worker's ``bridgeData.yaml`` env (``cache_home`` etc.) to this process.

    A missing or unreadable config must not break ``fake``/``dry_run``/CI runs, so failures are
    swallowed. In ``real`` mode, an unresolved model directory is warned about loudly because it
    guarantees the "No models available" crash.

    Args:
        process_mode: The benchmark process mode (``fake``, ``dry_run``, or ``real``).
    """
    from horde_worker_regen.load_env_vars import load_env_vars_from_config

    try:
        load_env_vars_from_config()
    except FileNotFoundError:
        pass  # No bridgeData.yaml: handled by the real-mode warning below.
    except Exception as e:  # noqa: BLE001 - benchmark env setup must never hard-fail the run
        logger.warning(f"Could not load worker env from bridgeData.yaml: {type(e).__name__}: {e}")

    if process_mode == "real" and not os.getenv("AIWORKER_CACHE_HOME"):
        logger.warning(
            "Real-mode benchmark: AIWORKER_CACHE_HOME is unset and no bridgeData.yaml `cache_home` was "
            "found, so hordelib will look for models under ./models and the worker may crash with "
            "'No models available'. Set `cache_home` in bridgeData.yaml or export AIWORKER_CACHE_HOME.",
        )
