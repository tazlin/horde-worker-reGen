"""Apply the worker's beta-model opt-in to the TUI host process so the picker can surface beta models.

The worker opts every install into the image-generation beta at startup (``load_env_vars``), but the
TUI host process never runs that path. Without this, the model picker's beta gate
(``HORDELIB_BETA_MODEL_CATEGORIES``) is unset in the TUI, so pending-queue models like qwen stay hidden
even though the worker would happily load them. Mirroring the opt-in here, reading ``api_key`` from the
same bridgeData, keeps the picker consistent with what the worker will actually load. The TUI is also the
process that spawns the worker, so the worker inherits these env vars too.
"""

from __future__ import annotations

from pathlib import Path

from horde_worker_regen.load_env_vars import apply_beta_model_env_defaults
from horde_worker_regen.tui.config_form import DEFAULT_CONFIG_PATH, load_config


def apply_beta_model_env(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Set the beta opt-in env defaults in this process, using bridgeData's ``api_key`` when readable.

    A missing or garbled config must never block the TUI starting, so a read failure simply falls back to
    the anonymous key (sufficient for pending-queue reads). ``setdefault`` semantics in
    :func:`apply_beta_model_env_defaults` mean any value the operator set explicitly still wins.
    """
    api_key: str | None
    try:
        value = load_config(config_path).get("api_key")
        api_key = str(value) if value else None
    except Exception:  # noqa: BLE001 - a missing/garbled config must never block the TUI starting
        api_key = None
    apply_beta_model_env_defaults(api_key)
