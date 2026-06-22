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
from collections.abc import Iterable

from loguru import logger

from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.ladder import BETA_TIERS

_ANON_AI_HORDE_API_KEY = "0000000000"
"""The AI-Horde anonymous reader key, sufficient to read the PRIMARY pending (beta) model queue."""


def ensure_worker_env(process_mode: str, tiers: Iterable[BenchTier] | None = None) -> None:
    """Best-effort: apply the worker's ``bridgeData.yaml`` env (``cache_home`` etc.) to this process.

    A missing or unreadable config must not break ``fake``/``dry_run``/CI runs, so failures are
    swallowed. In ``real`` mode, an unresolved model directory is warned about loudly because it
    guarantees the "No models available" crash. When a beta tier (e.g. qwen) is requested, the
    hordelib beta opt-in env is set so its pending-reference model surfaces (see
    :func:`_enable_beta_models`).

    Args:
        process_mode: The benchmark process mode (``fake``, ``dry_run``, or ``real``).
        tiers: The tiers the ramp will attempt; used to decide whether to opt into beta models.
    """
    from horde_worker_regen.load_env_vars import load_env_vars_from_config

    try:
        load_env_vars_from_config()
    except FileNotFoundError:
        pass  # No bridgeData.yaml: handled by the real-mode warning below.
    except Exception as e:  # noqa: BLE001 - benchmark env setup must never hard-fail the run
        logger.warning(f"Could not load worker env from bridgeData.yaml: {type(e).__name__}: {e}")

    if tiers is not None and any(tier in BETA_TIERS for tier in tiers):
        _enable_beta_models()

    if process_mode == "real" and not os.getenv("AIWORKER_CACHE_HOME"):
        logger.warning(
            "Real-mode benchmark: AIWORKER_CACHE_HOME is unset and no bridgeData.yaml `cache_home` was "
            "found, so hordelib will look for models under ./models and the worker may crash with "
            "'No models available'. Set `cache_home` in bridgeData.yaml or export AIWORKER_CACHE_HOME.",
        )


def _enable_beta_models() -> None:
    """Opt into the image-generation beta (pending) model category for this process tree.

    Sets hordelib's beta env vars (without overriding values the operator already set) so a beta-only
    checkpoint such as qwen surfaces from the PRIMARY pending queue. Beta also requires a PRIMARY URL
    (``HORDE_MODEL_REFERENCE_PRIMARY_API_URL``); its absence is warned about, not fatal.
    """
    from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
    from hordelib.beta_models import BETA_API_KEY_ENV_VAR, BETA_CATEGORIES_ENV_VAR

    os.environ.setdefault(BETA_CATEGORIES_ENV_VAR, MODEL_REFERENCE_CATEGORY.image_generation.value)
    os.environ.setdefault(BETA_API_KEY_ENV_VAR, _ANON_AI_HORDE_API_KEY)
    logger.info(
        f"Beta models enabled for beta tier(s) via {BETA_CATEGORIES_ENV_VAR}={os.environ[BETA_CATEGORIES_ENV_VAR]}.",
    )
    if not os.getenv("HORDE_MODEL_REFERENCE_PRIMARY_API_URL"):
        logger.warning(
            "A beta tier was requested but HORDE_MODEL_REFERENCE_PRIMARY_API_URL is unset; beta models "
            "are served from a PRIMARY pending queue and will not load without it, so the tier will skip.",
        )


__all__ = ["ensure_worker_env"]
