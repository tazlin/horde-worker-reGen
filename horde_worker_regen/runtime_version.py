"""The version string the worker reports about itself at runtime.

The release version itself lives in exactly one place: ``__version__`` in ``horde_worker_regen``
(read by hatchling at build time, see ``[tool.hatch.version]`` in ``pyproject.toml``). This module adds
a thin, best-effort annotation on top of it: when the worker is being run straight out of a git checkout
that is *not* sitting exactly on the matching release tag, the reported version gains a
``+dev.g<shortsha>`` (and ``.modified`` when there are uncommitted changes) suffix.

The point is to make a developer's local run distinguishable from a real release on the AI Horde (the
``bridge_agent`` header) and in logs, without ever touching the clean ``__version__`` literal that hatch
and semver parse. Everything here is best-effort: no git, a missing ``git`` binary, or any failure all
degrade silently to the plain ``__version__``. The result is computed once and cached, so the subprocess
calls happen at most once per process.
"""

from __future__ import annotations

import json
import subprocess
from functools import cache
from importlib import metadata
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from horde_worker_regen import __version__

_REPO_ROOT = Path(__file__).resolve().parent.parent

HORDELIB_DISTRIBUTION = "horde-engine"
"""The distribution name hordelib is installed under."""


def _git(*args: str, cwd: Path | None = None) -> str | None:
    """Run a git command in *cwd* (the repo root by default), returning stripped stdout or None on any failure."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed git args, no user input
            ["git", *args],  # noqa: S607 - rely on PATH; git absence is handled below
            cwd=_REPO_ROOT if cwd is None else cwd,
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
    """Compute a ``+dev.g<sha>[.modified]`` suffix, or an empty string for a release-equivalent checkout.

    Returns an empty string when there is no usable git checkout, or when HEAD is exactly on the
    ``v{__version__}`` tag with a clean working tree (i.e. this *is* the release that ``__version__``
    describes). A dirty tree on the tag still earns a ``.modified`` marker, since it is no longer the
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

    return f"+dev.g{short_sha}{'.modified' if dirty else ''}"


@cache
def runtime_version() -> str:
    """The worker version to report at runtime, annotated for non-release git checkouts.

    Returns the clean ``__version__`` for installed/release runs, or ``__version__`` plus a
    ``+dev.g<sha>`` suffix when run from a git checkout that is not on the matching release tag.
    """
    return f"{__version__}{_dev_suffix()}"


def _direct_url_checkout(distribution: metadata.Distribution) -> Path | None:
    """The local directory a distribution was installed from (PEP 610 ``direct_url.json``), or None."""
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    url = data.get("url") if isinstance(data, dict) else None
    if not isinstance(url, str) or urlparse(url).scheme != "file":
        return None
    return Path(url2pathname(urlparse(url).path))


@cache
def hordelib_identity() -> str:
    """The installed hordelib version and source, as ``<version> (<source>)``.

    An editable checkout reports the same version number for every commit, so a ``file:`` direct-URL
    install adds the checkout's commit and a ``.modified`` marker for a dirty tree. The package itself is
    never imported: the orchestrator process stays torch-free. Never raises.
    """
    try:
        distribution = metadata.distribution(HORDELIB_DISTRIBUTION)
        version = distribution.version
        checkout = _direct_url_checkout(distribution)
    except Exception:  # noqa: BLE001 - identity is best-effort and must never fail startup
        return "unknown (not installed)"
    if checkout is None:
        return f"{version} (site-packages)"
    short_sha = _git("rev-parse", "--short", "HEAD", cwd=checkout)
    if not short_sha:
        return f"{version} (editable {checkout.name} unknown)"
    dirty = bool(_git("status", "--porcelain", cwd=checkout))
    return f"{version} (editable {checkout.name} g{short_sha}{'.modified' if dirty else ''})"


__all__ = ["HORDELIB_DISTRIBUTION", "hordelib_identity", "runtime_version"]
