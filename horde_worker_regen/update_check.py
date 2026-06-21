"""A best-effort 'is there a newer release?' check, shared by the dashboard and the headless worker.

Compares the running version against the latest GitHub release and, when newer, lets a caller surface a
non-blocking notice telling the user how to update (``winget upgrade`` or re-running the installer). It
deliberately does not apply updates itself: self-replacing a running install is platform-specific and
error-prone (file locks on Windows), so we point the user at the same idempotent installer they already
have rather than hand-roll an updater.

The network call is isolated in ``_fetch_latest_release`` so tests can stub it, uses only the standard
library (no extra dependency, no heavy import such as torch), and degrades to "no update" on any error so
an offline launch is silent. Living at the package root keeps it importable from the headless worker
(``run_worker``/``status_reporter``) without dragging in the TUI package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from horde_worker_regen import __version__

_API_URL = "https://api.github.com/repos/Haidra-Org/horde-worker-reGen/releases/latest"
RELEASES_URL = "https://github.com/Haidra-Org/horde-worker-reGen/releases/latest"
_TIMEOUT_SECONDS = 6.0

NEWER_RELEASE_ENV_VAR = "AIWORKER_NEWER_RELEASE_AVAILABLE"
"""Set to the newer release version once a startup check finds one, so periodic logs can re-nag."""

DISABLE_ENV_VAR = "HORDE_WORKER_NO_UPDATE_CHECK"
"""When set, every update check is skipped (offline/air-gapped installs, or to silence the nag)."""


@dataclass(frozen=True)
class UpdateInfo:
    """A newer release than the one running."""

    latest_version: str
    html_url: str


def current_version() -> str:
    """The version of the running worker package."""
    return __version__


def update_check_disabled() -> bool:
    """Whether update checks should be skipped (explicitly disabled, or running under the test suite)."""
    return bool(os.environ.get(DISABLE_ENV_VAR) or os.environ.get("AI_HORDE_TESTING"))


def _version_tuple(value: str) -> tuple[int, ...]:
    """Parse a version like ``v12.0.1`` into ``(12, 0, 1)``, tolerating a leading v and odd suffixes."""
    parts: list[int] = []
    for chunk in value.strip().lstrip("vV").split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _is_newer(latest: str, current: str) -> bool:
    """Whether *latest* is a strictly higher version than *current*."""
    return _version_tuple(latest) > _version_tuple(current)


def _fetch_latest_release() -> dict[str, Any]:
    """Fetch the latest release metadata from the GitHub API (blocking; isolated for tests)."""
    import json
    import urllib.request

    request = urllib.request.Request(  # noqa: S310 - fixed https GitHub API URL, not user input
        _API_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "horde-worker-reGen-update-check"},
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310 - see above
        return json.loads(response.read().decode("utf-8"))


def check_for_update(current: str | None = None) -> UpdateInfo | None:
    """Return info about a newer release than *current*, or None when up to date or unreachable."""
    current = current or current_version()
    try:
        data = _fetch_latest_release()
    except Exception:  # noqa: BLE001 - an unreachable API means "no update to report", not a crash
        return None
    tag = str(data.get("tag_name") or "").strip()
    if not tag or not _is_newer(tag, current):
        return None
    return UpdateInfo(latest_version=tag.lstrip("vV"), html_url=str(data.get("html_url") or RELEASES_URL))


__all__ = [
    "DISABLE_ENV_VAR",
    "NEWER_RELEASE_ENV_VAR",
    "RELEASES_URL",
    "UpdateInfo",
    "check_for_update",
    "current_version",
    "update_check_disabled",
]
