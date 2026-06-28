"""Back-compat re-export of the update check, which now lives at the package root.

The implementation moved to :mod:`horde_worker_regen.update_check` so the headless worker can use it
without importing the TUI package. This module keeps the original import path working for the dashboard
and existing tests.
"""

from __future__ import annotations

from horde_worker_regen.update_check import (
    DISABLE_ENV_VAR,
    NEWER_RELEASE_ENV_VAR,
    RELEASES_URL,
    UPDATE_CHECK_INTERVAL_SECONDS,
    UpdateInfo,
    apply_update_check_result,
    check_for_update,
    current_version,
    update_check_disabled,
)

__all__ = [
    "DISABLE_ENV_VAR",
    "NEWER_RELEASE_ENV_VAR",
    "RELEASES_URL",
    "UPDATE_CHECK_INTERVAL_SECONDS",
    "UpdateInfo",
    "apply_update_check_result",
    "check_for_update",
    "current_version",
    "update_check_disabled",
]
