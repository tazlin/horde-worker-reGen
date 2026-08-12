"""Apply the worker's ``bridgeData.yaml`` environment for real-mode benchmarking.

The harness builds *synthetic* bridge data and forces ``_loaded_from_env_vars`` so it never reads
``bridgeData.yaml`` (see ``horde_worker_regen.harness.build_harness_bridge_data``). A normal worker
run, by contrast, sets ``AIWORKER_CACHE_HOME`` from the config's ``cache_home`` at startup
(``horde_worker_regen.load_env_vars``). Without that, the real inference children fall back to
hordelib's CWD-relative ``./models`` weights root (``UserSettings.get_model_directory``), find no
checkpoints, and exit with "No models available", wedging the first level until its timeout.

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


def ensure_worker_env(process_mode: str, tiers: Iterable[BenchTier] | None = None) -> None:
    """Best-effort: apply the worker's ``bridgeData.yaml`` env (``cache_home`` etc.) to this process.

    A missing or unreadable config must not break ``fake``/``dry_run``/CI runs, so failures are
    swallowed. In ``real`` mode, an unresolved model directory is warned about loudly because it
    guarantees the "No models available" crash, and the hordelib beta opt-in is filled in so the run
    measures the same model surface the worker it is benchmarking would boot with.

    The beta opt-in is confined to ``real`` mode because it is what lets a surface read the PRIMARY's
    pending queue over the network. A fake or dry-run run boots no children that could load those
    weights, so the network round-trip would buy nothing and would make an offline box's results depend
    on its connectivity. Applied before the caller builds its catalog, so the offered forms and the
    booted children agree.

    Args:
        process_mode: The benchmark process mode (``fake``, ``dry_run``, or ``real``).
        tiers: The tiers the ramp will attempt; used to decide whether to opt into beta models.
    """
    from horde_worker_regen.load_env_vars import apply_beta_model_env_defaults, load_env_vars_from_config

    try:
        load_env_vars_from_config()
    except FileNotFoundError:
        pass  # No bridgeData.yaml: handled by the real-mode warning below.
    except Exception as e:  # noqa: BLE001 - benchmark env setup must never hard-fail the run
        logger.warning(f"Could not load worker env from bridgeData.yaml: {type(e).__name__}: {e}")

    if process_mode == "real":
        # A worker run gets this from its config load; a benchmark run with no readable bridgeData.yaml
        # would otherwise measure a narrower model surface than the worker it is benchmarking, skipping
        # every beta-only form and tier. Setdefault semantics, so an operator's own values (including an
        # explicit opt-out) still win.
        apply_beta_model_env_defaults()

        if tiers is not None and any(tier in BETA_TIERS for tier in tiers):
            _enable_beta_models()

    if process_mode == "real" and not os.getenv("AIWORKER_CACHE_HOME"):
        logger.warning(
            "Real-mode benchmark: AIWORKER_CACHE_HOME is unset and no bridgeData.yaml `cache_home` was "
            "found, so hordelib will look for models under ./models and the worker may crash with "
            "'No models available'. Set `cache_home` in bridgeData.yaml or export AIWORKER_CACHE_HOME.",
        )


def _enable_beta_models() -> None:
    """Report the beta (pending) model opt-in for this process tree, filling it in if nothing set it.

    The opt-in itself is the worker's (:func:`apply_beta_model_env_defaults`), applied by
    :func:`ensure_worker_env` before this runs; calling it again is a no-op under its setdefault
    semantics and keeps a beta tier from depending on the order the env was assembled in. Spawned
    children inherit the environment, so setting it here covers the whole process tree. Beta also
    requires a PRIMARY URL (``HORDE_MODEL_REFERENCE_PRIMARY_API_URL``); its absence is warned about,
    not fatal.
    """
    from horde_worker_regen.load_env_vars import apply_beta_model_env_defaults

    apply_beta_model_env_defaults()
    logger.info(
        f"Beta models enabled for beta tier(s) via "
        f"HORDELIB_BETA_MODEL_CATEGORIES={os.environ.get('HORDELIB_BETA_MODEL_CATEGORIES', '')!r}.",
    )
    if not os.getenv("HORDE_MODEL_REFERENCE_PRIMARY_API_URL"):
        logger.warning(
            "A beta tier was requested but HORDE_MODEL_REFERENCE_PRIMARY_API_URL is unset; beta models "
            "are served from a PRIMARY pending queue and will not load without it, so the tier will skip.",
        )


__all__ = ["ensure_worker_env"]
