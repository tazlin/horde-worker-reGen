"""A best-effort 'is there a newer release?' check, shared by the dashboard and the headless worker.

Compares the running version against the latest GitHub release and, when newer, lets a caller surface a
non-blocking notice telling the user how to update (``update.cmd``/``update.sh`` or re-running the
installer). It deliberately does not apply updates itself: the in-place applier lives in the
``worker_bootstrap`` package, which runs before the venv exists; this notifier only informs.

To keep the notice and the applier from disagreeing about what "newer" means, which channel applies, and
which repo to look at, this module reuses ``worker_bootstrap.updater`` (pure standard library, shipped in
the same bundle) when it is importable, and degrades to a local stable-channel check when it is not (a user
who installed only the worker into their own virtualenv). It uses only the standard library (no heavy
import such as torch) and degrades to "no update" on any error so an offline launch is silent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from horde_worker_regen import __version__

_DEFAULT_REPO = "Haidra-Org/horde-worker-reGen"
RELEASES_URL = "https://github.com/Haidra-Org/horde-worker-reGen/releases/latest"
_TIMEOUT_SECONDS = 6.0

UPDATE_CHECK_INTERVAL_SECONDS = 1800.0
"""How often (in seconds) the worker re-checks for a newer release (30 minutes)."""

NEWER_RELEASE_ENV_VAR = "AIWORKER_NEWER_RELEASE_AVAILABLE"
"""Set to the newer release version once a startup check finds one, so periodic logs can re-nag."""

DISABLE_ENV_VAR = "HORDE_WORKER_NO_UPDATE_CHECK"
"""When set, every update check is skipped (offline/air-gapped installs, or to silence the nag)."""

# The in-place updater carries the canonical version/channel/origin logic. Reuse it when present so the
# notifier never reports a different verdict than the applier; fall back to a local check when the bundle's
# bootstrap package is not importable (e.g. a worker installed standalone into a user-managed venv).
_bootstrap_updater: ModuleType | None
try:
    from worker_bootstrap import updater as _bootstrap_updater
except Exception:  # noqa: BLE001 - the notifier must work even without the bootstrap package present
    _bootstrap_updater = None


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


def _resolve_repo() -> str:
    """The ``owner/repo`` to query, following the recorded install origin when the bootstrap is available."""
    if _bootstrap_updater is not None:
        try:
            return _bootstrap_updater.resolve_update_repo()
        except Exception:  # noqa: BLE001 - any failure falls back to the production default
            pass
    return _DEFAULT_REPO


def _resolve_channel() -> str:
    """The release channel (``stable``/``beta``); a beta build is inferred onto the beta channel."""
    if _bootstrap_updater is not None:
        try:
            return _bootstrap_updater.update_channel()
        except Exception:  # noqa: BLE001 - any failure falls back to the stable channel
            pass
    return "stable"


def _version_tuple(value: str) -> tuple[int, ...]:
    """Parse a version like ``v12.0.1`` into ``(12, 0, 1)`` (local fallback comparator, core only)."""
    parts: list[int] = []
    for chunk in value.strip().lstrip("vV").split("+", 1)[0].split("-", 1)[0].split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _is_newer(latest: str, current: str) -> bool:
    """Whether *latest* is a strictly higher version than *current* (semver precedence when available)."""
    if _bootstrap_updater is not None:
        return bool(_bootstrap_updater.is_newer(latest, current))
    return _version_tuple(latest) > _version_tuple(current)


def _fetch_latest_release() -> dict[str, Any]:
    """Fetch the latest stable release metadata from the GitHub API (blocking; isolated for tests)."""
    import json
    import urllib.request

    request = urllib.request.Request(  # noqa: S310 - fixed https GitHub API URL, not user input
        f"https://api.github.com/repos/{_resolve_repo()}/releases/latest",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "horde-worker-reGen-update-check"},
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310 - see above
        return json.loads(response.read().decode("utf-8"))


def _fetch_releases() -> list[dict[str, Any]]:
    """Fetch the recent releases list (includes pre-releases) for the beta channel; isolated for tests."""
    import json
    import urllib.request

    request = urllib.request.Request(  # noqa: S310 - fixed https GitHub API URL, not user input
        f"https://api.github.com/repos/{_resolve_repo()}/releases?per_page=30",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "horde-worker-reGen-update-check"},
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310 - see above
        data = json.loads(response.read().decode("utf-8"))
    return [release for release in data if isinstance(release, dict)] if isinstance(data, list) else []


def _pick_latest_release(releases: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the highest-precedence non-draft release from a releases list (beta channel)."""
    best: dict[str, Any] | None = None
    best_tag = ""
    for release in releases:
        if release.get("draft"):
            continue
        tag = str(release.get("tag_name") or "").strip()
        if not tag:
            continue
        if best is None or _is_newer(tag, best_tag):
            best, best_tag = release, tag.lstrip("vV")
    return best


def check_for_update(current: str | None = None) -> UpdateInfo | None:
    """Return info about a newer release than *current*, or None when up to date or unreachable.

    Honours the resolved channel: the stable channel reads ``/releases/latest`` (which hides pre-releases);
    the beta channel reads the recent releases list and picks the newest by precedence, so a beta user is
    notified of newer betas and of the matching stable when it graduates.
    """
    current = current or current_version()
    try:
        release = _pick_latest_release(_fetch_releases()) if _resolve_channel() == "beta" else _fetch_latest_release()
    except Exception:  # noqa: BLE001 - an unreachable API means "no update to report", not a crash
        return None
    if not release:
        return None
    tag = str(release.get("tag_name") or "").strip()
    if not tag or not _is_newer(tag, current):
        return None
    return UpdateInfo(latest_version=tag.lstrip("vV"), html_url=str(release.get("html_url") or RELEASES_URL))


def apply_update_check_result(info: UpdateInfo | None) -> None:
    """Set or clear ``NEWER_RELEASE_ENV_VAR`` to match an update check result.

    Callers (the headless worker's periodic loop, the TUI's periodic check) pass the result of
    :func:`check_for_update` here so the env-var-driven log nag in
    :class:`~horde_worker_regen.reporting.status_reporter.StatusReporter` stays in sync with the
    latest verdict.

    Args:
        info: The result of :func:`check_for_update`, or None when up to date or unreachable.
    """
    if info is None:
        os.environ.pop(NEWER_RELEASE_ENV_VAR, None)
    else:
        os.environ[NEWER_RELEASE_ENV_VAR] = info.latest_version


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
