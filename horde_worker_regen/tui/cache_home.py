"""Resolve the effective model cache directory for the TUI's disk-space figures.

Disk figures (free space, on-disk presence, download totals) all resolve against the *weights root*:
``$AIWORKER_CACHE_HOME`` if set, else the relative ``models`` directory. The worker derives that env var
from bridgeData's ``cache_home`` at startup (see ``load_env_vars``), but the TUI parent process never did.
So when an operator pointed ``cache_home`` at another disk without also exporting ``AIWORKER_CACHE_HOME``,
the TUI computed presence against an empty ``./models`` and free space against the current-directory volume:
every model looked like it needed downloading and the free-space number was for the wrong drive.

This applies the same ``cache_home`` precedence the worker uses to the TUI's own process, once, so every
downstream disk computation (``free_model_bytes``, ``compute_download_plan``) lands on the configured
volume. The TUI is also the process that spawns the worker, so the worker inherits the same resolved value.
"""

from __future__ import annotations

import os
from pathlib import Path

from horde_worker_regen.tui.config_form import DEFAULT_CONFIG_PATH, load_config


def apply_cache_home_env(config_path: Path = DEFAULT_CONFIG_PATH) -> str | None:
    """Mirror the worker's ``cache_home`` precedence into this process's ``AIWORKER_CACHE_HOME`` (if unset).

    Precedence matches ``load_env_vars``: an already-set ``AIWORKER_CACHE_HOME`` wins (a shell export or a
    ``.env`` always takes precedence); otherwise bridgeData's ``cache_home``; otherwise the installer's
    ``<HORDE_WORKER_DATA_DIR>/models`` fallback. Returns the effective value, or None when nothing supplied
    one (leaving :func:`resolve_weights_root`'s own ``models`` default to apply).
    """
    existing = os.environ.get("AIWORKER_CACHE_HOME")
    if existing:
        return existing

    cache_home = _config_cache_home(config_path)
    if cache_home:
        os.environ["AIWORKER_CACHE_HOME"] = cache_home
        return cache_home

    data_dir = os.environ.get("HORDE_WORKER_DATA_DIR")
    if data_dir:
        value = os.path.join(data_dir, "models")
        os.environ["AIWORKER_CACHE_HOME"] = value
        return value

    return None


def _config_cache_home(config_path: Path) -> str | None:
    """Read ``cache_home`` from bridgeData, or None when the file is absent, unreadable, or omits it."""
    try:
        data = load_config(config_path)
        value = data.get("cache_home")
    except Exception:  # noqa: BLE001 - a missing/garbled config must never block the TUI starting
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None
