"""The version string the worker reports about itself at runtime.

The release version itself lives in exactly one place: ``__version__`` in ``horde_worker_regen``
(read by hatchling at build time, see ``[tool.hatch.version]`` in ``pyproject.toml``). This module adds
a thin, best-effort annotation on top of it: when the worker is being run straight out of a git checkout
that is *not* sitting exactly on the matching release tag, the reported version gains a
``+dev.g<shortsha>`` (and ``.dirty`` when there are uncommitted changes) suffix.

The point is to make a developer's local run distinguishable from a real release on the AI Horde (the
``bridge_agent`` header) and in logs, without ever touching the clean ``__version__`` literal that hatch
and semver parse. Everything here is best-effort: no git, a missing ``git`` binary, or any failure all
degrade silently to the plain ``__version__``. The result is computed once and cached, so the subprocess
calls happen at most once per process.
"""

from __future__ import annotations

import subprocess
from functools import cache
from pathlib import Path

from horde_worker_regen import __version__

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    """Run a git command in the repo root, returning stripped stdout or None on any failure."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed git args, no user input
            ["git", *args],  # noqa: S607 - rely on PATH; git absence is handled below
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _dev_suffix() -> str:
    """Compute a ``+dev.g<sha>[.dirty]`` suffix, or an empty string for a release-equivalent checkout.

    Returns an empty string when there is no usable git checkout, or when HEAD is exactly on the
    ``v{__version__}`` tag with a clean working tree (i.e. this *is* the release that ``__version__``
    describes). A dirty tree on the tag still earns a ``.dirty`` marker, since it is no longer the
    released bits.
    """
    if not (_REPO_ROOT / ".git").exists():
        return ""

    on_release_tag = _git("describe", "--tags", "--exact-match") == f"v{__version__}"
    dirty = bool(_git("status", "--porcelain"))

    # Exact release tag with a clean tree -> indistinguishable from a real release, so no suffix.
    if on_release_tag and not dirty:
        return ""

    short_sha = _git("rev-parse", "--short", "HEAD")
    if not short_sha:
        return ""

    return f"+dev.g{short_sha}{'.dirty' if dirty else ''}"


@cache
def runtime_version() -> str:
    """The worker version to report at runtime, annotated for non-release git checkouts.

    Returns the clean ``__version__`` for installed/release runs, or ``__version__`` plus a
    ``+dev.g<sha>`` suffix when run from a git checkout that is not on the matching release tag.
    """
    return f"{__version__}{_dev_suffix()}"


__all__ = ["runtime_version"]
