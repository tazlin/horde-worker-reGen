"""Self-update: discover, verify, and apply a newer release bundle in place.

Standard-library only, like the rest of ``worker_bootstrap``: this can run before the project venv exists
and must not import the worker or any third-party package. The release bundle is files-only (source,
shims, ``pyproject.toml``, ``uv.lock``); the venv, ``bin/``, models, and ``bridgeData.yaml`` live outside
it and are preserved. An update is therefore a *verified* download followed by a files overlay, after
which the lock-aware sync installs whatever the new lock requires.

Every network and filesystem step fails closed and non-fatally: a launch-time check must never raise into
the launch path, and a partial or unverifiable download must never overlay the install.

The updater is channel- and origin-aware. It pulls from the repo the user actually installed from (the
front-end records it in ``bin/install-info``; an env var overrides), follows a stable or beta channel using
semantic-versioning precedence so a running beta is never reverted to an older stable, and bows out
entirely for installs whose updates are owned elsewhere (winget, a git checkout).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from worker_bootstrap import paths

_DEFAULT_REPO = "Haidra-Org/horde-worker-reGen"
"""Release origin used only when neither an explicit override nor an install-time marker names one (e.g. a
hand-extracted zip). A fork or staging channel redirects it via the marker or the env override without a
code change; Haidra-Org is the canonical production origin."""

_BUNDLE_ASSET = "horde-worker-reGen.zip"
_CHECKSUMS_ASSET = "SHA256SUMS"
_HTTP_TIMEOUT = 15.0
_USER_AGENT = "horde-worker-updater"

_BOM = "﻿"
"""A UTF-8 byte-order mark, which PowerShell/sh may prepend when writing ``bin/install-info``; stripped
before parsing so the first key is not read as a ``<BOM>method`` key."""

_INNO_APP_ID = "{2A6B13F1-1070-4369-A40F-4132AE735E60}"
"""The graphical installer's stable Inno Setup AppId (see ``packaging/inno/HordeWorker.iss``). Inno records
the uninstall entry under ``<AppId>_is1``; the updater rewrites that entry's ``DisplayVersion`` after a
self-update so Add/Remove Programs does not drift from the files actually on disk."""

_CHECK_THROTTLE_SECONDS = 6 * 60 * 60
"""How long a launch-time check is trusted before re-checking, so a plain launch is not gated on the GitHub
API every time. The explicit ``update`` command ignores this and always checks."""

UpdateChannel = Literal["stable", "beta"]

InstallMethod = Literal["one-line", "exe", "zip", "winget", "dev", "unknown"]
"""How the install was performed. The marker carries the first three; ``winget`` and ``dev`` are detected
from the environment (a winget portable path; a git working tree or an in-repo dev pin). ``unknown`` is an
unmarked but otherwise ordinary install (a hand-extracted zip), which is allowed to self-update."""

_PRESERVE_NAMES = frozenset({"bridgeData.yaml", ".venv", "bin", "logs", ".horde-sync-stamp"})
"""Names an overlay must never clobber even if a future bundle mistakenly contained them: user state and the
heavy, separately-managed runtime. The bundle does not ship these today, so this is defence in depth."""

_SHIM_SUFFIXES = frozenset({".cmd", ".sh", ".ps1", ".bat"})
"""Top-level shell launcher shims that are NOT overlaid. A running ``.cmd``/``.sh`` is read incrementally by
its interpreter, so overwriting the script driving the update corrupts it mid-run (the reason the old updater
piped a remote installer from memory). The shims are stable and refreshed by the full installer; the
self-update carries the Python source and lockfile, which is what changes release to release. Python modules
are safe to overlay: the interpreter reads each file fully and closes it."""

_MIRROR_DIRS = frozenset({"horde_worker_regen", "worker_bootstrap"})
"""Bundle directories the overlay mirrors rather than merges: the worker's Python import roots, where a
module deleted upstream must not linger and shadow the new code. These hold only bundled source (no user
state), so pruning files the new bundle no longer ships is safe, and it clears stale ``__pycache__``
bytecode too. Every other bundle directory is merged (never pruned), so an unknown future layout is left
intact."""


def auto_update_policy() -> str:
    """Resolve the update policy from ``HORDE_WORKER_AUTO_UPDATE``: ``prompt`` (default), ``auto``, ``off``."""
    value = os.environ.get("HORDE_WORKER_AUTO_UPDATE", "").strip().lower()
    return value if value in ("prompt", "auto", "off") else "prompt"


def default_root() -> Path:
    """The install root the updater operates on (the bundled worker folder)."""
    return paths.install_root()


def _read_install_info(root: Path) -> dict[str, str]:
    """Parse ``bin/install-info`` into a ``key=value`` mapping, tolerating a BOM and comments.

    The file is written by a front-end (PowerShell/sh/Inno), any of which may emit a UTF-8 BOM, so the BOM
    is stripped before parsing. A missing or malformed file yields an empty mapping rather than raising.
    """
    info: dict[str, str] = {}
    try:
        text = paths.install_info_file(root).read_text(encoding="utf-8")
    except OSError:
        return info
    for line in text.replace(_BOM, "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        info[key.strip().lower()] = value.strip()
    return info


def resolve_update_repo(root: Path | None = None) -> str:
    """Return the ``owner/repo`` to pull releases from (env override > install marker > default).

    Following the recorded install origin is what lets a fork or staging install update itself from where
    it actually came from instead of a hardcoded production repo.
    """
    override = os.environ.get("HORDE_WORKER_UPDATE_REPO", "").strip()
    if override:
        return override
    recorded = _read_install_info(root or default_root()).get("repo", "")
    return recorded or _DEFAULT_REPO


def _has_local_path_source(pyproject: Path) -> bool:
    """Whether ``pyproject.toml`` still pins a dependency to a local ``path =`` source (an in-repo dev pin).

    Mirrors the release workflow's guard: a released bundle never carries one, so its presence marks a
    developer checkout the self-updater must not overlay.
    """
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return False
    return re.search(r"^\s*[A-Za-z0-9_.-]+\s*=\s*\{[^}]*\bpath\s*=", text, re.MULTILINE) is not None


def resolve_install_method(root: Path | None = None) -> InstallMethod:
    """Classify how this worker was installed, to decide whether the self-updater owns updates.

    A winget portable install lives under a WinGet ``Packages`` directory; a developer checkout is a git
    working tree or a path-pinned ``pyproject.toml``. Both are detected from the environment because the
    front-end could not (or must not) record them. Otherwise the marker's ``method`` is trusted, falling
    back to ``unknown`` for an unmarked but ordinary install.
    """
    root = root or default_root()
    normalized = str(root).replace("\\", "/").lower()
    if "/winget/packages/" in normalized:
        return "winget"
    if (root / ".git").exists() or _has_local_path_source(paths.pyproject_path(root)):
        return "dev"
    method = _read_install_info(root).get("method", "").lower()
    if method in ("one-line", "exe", "zip"):
        return method  # type: ignore[return-value]
    return "unknown"


def self_update_allowed(root: Path | None = None) -> tuple[bool, str]:
    """Whether the in-place self-updater may run here, with a pointer to the right path when it may not.

    winget tracks the installed version itself, so a silent self-update would drift from winget's database
    and a later ``winget upgrade`` would reinstall the older manifest version over the updated files; a git
    checkout's files are the user's working tree. In both cases updates are owned elsewhere.
    """
    method = resolve_install_method(root)
    if method == "winget":
        return False, "This is a winget-managed install; update it with `winget upgrade Haidra.HordeWorker`."
    if method == "dev":
        return False, "This looks like a development checkout; update it with `git pull` then `update-runtime`."
    return True, ""


def installed_version(root: Path) -> str | None:
    """Read ``__version__`` from the bundled worker package without importing it (stdlib-only).

    Importing ``horde_worker_regen`` is not possible here (its dependencies may not be installed yet), so
    the literal is parsed straight from the source file, mirroring how the release workflow reads it.
    """
    init = root / "horde_worker_regen" / "__init__.py"
    try:
        text = init.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    return match.group(1) if match else None


def _has_prerelease(version: str) -> bool:
    """Whether *version* carries a semver pre-release suffix (a ``-`` in its core, ignoring build metadata)."""
    core = version.strip().lstrip("vV").split("+", 1)[0]
    return "-" in core


def update_channel(root: Path | None = None) -> UpdateChannel:
    """Resolve the release channel: ``HORDE_WORKER_UPDATE_CHANNEL`` env, else inferred from the install.

    A worker running a pre-release build is inferred onto the ``beta`` channel so it keeps receiving betas
    and eventually graduates to the matching stable, rather than being told it is "up to date" against the
    latest stable. Everything else defaults to ``stable``.
    """
    value = os.environ.get("HORDE_WORKER_UPDATE_CHANNEL", "").strip().lower()
    if value in ("stable", "beta"):
        return value  # type: ignore[return-value]
    installed = installed_version(root or default_root())
    if installed and _has_prerelease(installed):
        return "beta"
    return "stable"


def _parse_version(version: str) -> tuple[tuple[int, ...], tuple[tuple[int, int, str], ...]]:
    """Parse a version/tag into a semver (release-core, pre-release-identifiers) pair for precedence.

    A leading ``v`` and any ``+build`` metadata are ignored. Each pre-release identifier is normalised to a
    homogeneous, directly-comparable triple: numeric identifiers as ``(0, value, "")`` and alphanumeric as
    ``(1, 0, text)``, so a numeric identifier always sorts below an alphanumeric one (semver rule 11.4.3)
    and the tuples never mix ``int`` and ``str`` in the same slot. Tags are expected to be dash-delimited
    semver; a non-dash suffix (e.g. ``1.2.3rc1``) is treated as part of the release core, not a pre-release.
    """
    raw = version.strip().lstrip("vV").split("+", 1)[0]
    core_part, _, pre_part = raw.partition("-")
    core: list[int] = []
    for chunk in core_part.split("."):
        match = re.match(r"\d+", chunk)
        core.append(int(match.group()) if match else 0)
    pre: list[tuple[int, int, str]] = []
    for identifier in pre_part.split(".") if pre_part else []:
        if identifier.isdigit():
            pre.append((0, int(identifier), ""))
        else:
            pre.append((1, 0, identifier))
    return tuple(core), tuple(pre)


def compare_versions(left: str, right: str) -> int:
    """Compare two versions by semantic-versioning precedence; return -1, 0, or 1.

    Release cores are compared component-wise (zero-padded to equal length). When cores are equal, a
    version with no pre-release outranks one with a pre-release; two pre-releases compare identifier-wise,
    and a longer identifier list outranks its prefix (semver rule 11.4.4).
    """
    core_left, pre_left = _parse_version(left)
    core_right, pre_right = _parse_version(right)
    length = max(len(core_left), len(core_right))
    padded_left = core_left + (0,) * (length - len(core_left))
    padded_right = core_right + (0,) * (length - len(core_right))
    if padded_left != padded_right:
        return -1 if padded_left < padded_right else 1
    if not pre_left and not pre_right:
        return 0
    if not pre_left:
        return 1
    if not pre_right:
        return -1
    if pre_left == pre_right:
        return 0
    return -1 if pre_left < pre_right else 1


def is_newer(candidate: str, installed: str) -> bool:
    """Whether release tag *candidate* has strictly higher precedence than the *installed* version."""
    return compare_versions(candidate, installed) > 0


@dataclass(frozen=True)
class UpdateInfo:
    """The outcome of a check: the versions involved and the assets needed to apply an update."""

    current: str | None
    latest: str | None
    available: bool
    bundle_url: str | None
    checksums_url: str | None
    channel: UpdateChannel = "stable"
    is_prerelease: bool = False


@dataclass(frozen=True)
class UpdateResult:
    """The outcome of applying (or declining to apply) an update."""

    ok: bool
    message: str
    from_version: str | None
    to_version: str | None


def _http_get(url: str) -> bytes:
    """GET *url* and return the body bytes (https GitHub URLs only)."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/octet-stream"})
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:  # noqa: S310 - fixed https GitHub URL
        return response.read()


def _fetch_json(url: str) -> object:
    """GET a GitHub API URL and return the decoded JSON body (https GitHub API URLs only)."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:  # noqa: S310 - fixed https API URL
        return json.loads(response.read().decode("utf-8"))


def _release_from_obj(obj: object) -> tuple[str, dict[str, str], bool] | None:
    """Extract ``(tag, {asset_name: url}, is_prerelease)`` from a GitHub release object, or None."""
    if not isinstance(obj, dict):
        return None
    tag = obj.get("tag_name")
    if not isinstance(tag, str) or not tag:
        return None
    assets = {
        asset["name"]: asset["browser_download_url"]
        for asset in obj.get("assets", [])
        if isinstance(asset, dict) and "name" in asset and "browser_download_url" in asset
    }
    return tag, assets, bool(obj.get("prerelease"))


def _latest_stable(repo: str) -> tuple[str, dict[str, str], bool] | None:
    """The latest stable release (``/releases/latest`` excludes drafts and pre-releases)."""
    try:
        data = _fetch_json(f"https://api.github.com/repos/{repo}/releases/latest")
    except Exception:  # noqa: BLE001 - a check must never raise into the launch path
        return None
    return _release_from_obj(data)


def _latest_including_prereleases(repo: str) -> tuple[str, dict[str, str], bool] | None:
    """The highest-precedence non-draft release across the recent list, including pre-releases.

    ``/releases/latest`` hides pre-releases, so the beta channel reads the full list and picks the newest by
    semver precedence. This naturally graduates a beta user to a newer final (a final outranks its own
    pre-releases) and never offers something older than what is installed.
    """
    try:
        data = _fetch_json(f"https://api.github.com/repos/{repo}/releases?per_page=30")
    except Exception:  # noqa: BLE001 - a check must never raise into the launch path
        return None
    if not isinstance(data, list):
        return None
    best: tuple[str, dict[str, str], bool] | None = None
    for obj in data:
        if isinstance(obj, dict) and obj.get("draft"):
            continue
        release = _release_from_obj(obj)
        if release is None:
            continue
        if best is None or compare_versions(release[0], best[0]) > 0:
            best = release
    return best


def latest_release(repo: str, channel: UpdateChannel = "stable") -> tuple[str, dict[str, str], bool] | None:
    """Return ``(tag, {asset_name: url}, is_prerelease)`` for the channel's newest release, or None."""
    if channel == "beta":
        return _latest_including_prereleases(repo)
    return _latest_stable(repo)


def check_for_update(
    root: Path,
    repo: str | None = None,
    channel: UpdateChannel | None = None,
) -> UpdateInfo:
    """Compare the installed version against the channel's newest release. Never raises."""
    current = installed_version(root)
    repo = repo or resolve_update_repo(root)
    channel = channel or update_channel(root)
    latest = latest_release(repo, channel)
    if latest is None:
        return UpdateInfo(
            current=current,
            latest=None,
            available=False,
            bundle_url=None,
            checksums_url=None,
            channel=channel,
            is_prerelease=False,
        )
    tag, assets, is_prerelease = latest
    available = current is not None and is_newer(tag, current)
    return UpdateInfo(
        current=current,
        latest=tag,
        available=available,
        bundle_url=assets.get(_BUNDLE_ASSET),
        checksums_url=assets.get(_CHECKSUMS_ASSET),
        channel=channel,
        is_prerelease=is_prerelease,
    )


def _read_state(root: Path) -> dict[str, object]:
    """Return the persisted update state (skip + throttle), or an empty mapping when absent/corrupt."""
    try:
        data = json.loads(paths.update_state_file(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(root: Path, state: dict[str, object]) -> None:
    """Persist the update state (best-effort; a write failure only costs a re-offer or an early re-check)."""
    path = paths.update_state_file(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass


def is_version_skipped(root: Path, version: str) -> bool:
    """Whether the user previously chose to skip exactly this version (so it is not re-offered)."""
    return _read_state(root).get("skip_version") == version


def mark_version_skipped(root: Path, version: str) -> None:
    """Record that this version was skipped; it is re-offered only once a newer version is released."""
    state = _read_state(root)
    state["skip_version"] = version
    _write_state(root, state)


def clear_skip(root: Path) -> None:
    """Forget any skipped version (called after a successful update, so a future offer is not suppressed)."""
    state = _read_state(root)
    if state.pop("skip_version", None) is not None:
        _write_state(root, state)


def should_check_now(root: Path) -> bool:
    """Whether enough time has passed since the last launch-time check to check again."""
    last = _read_state(root).get("last_check_ts")
    if not isinstance(last, (int, float)):
        return True
    return (time.time() - float(last)) >= _CHECK_THROTTLE_SECONDS


def record_check(root: Path) -> None:
    """Stamp the time of a launch-time check so the next launches within the window skip the network call."""
    state = _read_state(root)
    state["last_check_ts"] = time.time()
    _write_state(root, state)


def sync_arp_version(root: Path, version: str | None) -> None:
    """After a self-update of a ``.exe`` install, align its Add/Remove Programs DisplayVersion (win32 only).

    Best-effort and silent: the install still works if the registry entry is missing or unwritable. Only an
    ``exe`` install on Windows has an Inno uninstall entry to keep honest; every other case returns early.
    """
    if version is None or sys.platform != "win32" or resolve_install_method(root) != "exe":
        return
    try:
        import winreg
    except ImportError:
        return
    key_path = rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{_INNO_APP_ID}_is1"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, version.lstrip("vV"))
    except OSError:
        pass


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in blocks so a large bundle does not load into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_hash(checksums_text: str, asset_name: str) -> str | None:
    """Find the sha256 for *asset_name* in a ``<sha256>  <name>`` SHA256SUMS body.

    The asset is matched on its exact basename (tolerating a leading binary ``*`` marker and any path
    prefix), so an unrelated asset whose name merely ends with the same text is never mistaken for it.
    """
    for line in checksums_text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and Path(parts[-1].lstrip("*")).name == asset_name:
            return parts[0]
    return None


def _bundle_root(extracted: Path) -> Path:
    """Return the directory holding the bundle files within *extracted*.

    The release zip is created from the staging dir's contents, so the files sit at the archive root; a
    single wrapping directory (some zip tools add one) is transparently descended into.
    """
    if (extracted / "pyproject.toml").exists():
        return extracted
    children = list(extracted.iterdir())
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extracted


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy *src* onto *dst* via a temp file in the destination directory, then atomically replace.

    A crash mid-copy leaves the previous *dst* intact rather than a half-written file. This matters most for
    ``uv.lock`` and ``pyproject.toml``, which the post-update sync reads to reconcile the venv.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.horde-update-tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _remove_empty_dirs(root: Path) -> None:
    """Remove directories under *root* left empty after a mirror prune (deepest first; non-empty are kept)."""
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        # rmdir only succeeds on an empty dir; one still holding kept/preserved files is left in place.
        with contextlib.suppress(OSError):
            directory.rmdir()


def _mirror_dir(src: Path, dst: Path) -> None:
    """Copy *src* onto *dst* and delete files under *dst* that *src* no longer ships (a true mirror).

    Used only for the worker's import roots (see ``_MIRROR_DIRS``) so a module dropped upstream cannot
    linger and shadow the new code. Preserved names and the running shims are never deleted (defensive;
    these directories do not contain them).
    """
    shutil.copytree(src, dst, dirs_exist_ok=True)
    expected = {p.relative_to(src) for p in src.rglob("*") if p.is_file()}
    for existing in (p for p in dst.rglob("*") if p.is_file()):
        if existing.relative_to(dst) in expected:
            continue
        if existing.name in _PRESERVE_NAMES or existing.suffix.lower() in _SHIM_SUFFIXES:
            continue
        existing.unlink(missing_ok=True)
    _remove_empty_dirs(dst)


def _overlay(bundle_root: Path, install_root: Path) -> None:
    """Copy the bundle files over the install, never touching preserved state or the running shims.

    The worker's import roots are mirrored (files absent from the new bundle are pruned) so a deleted module
    cannot persist; every other directory is merged. Top-level files are written atomically so an
    interrupted overlay leaves either the old or the new file, never a truncated one.
    """
    for item in bundle_root.iterdir():
        if item.name in _PRESERVE_NAMES:
            continue
        if item.is_file() and item.suffix.lower() in _SHIM_SUFFIXES:
            continue
        target = install_root / item.name
        if item.is_dir():
            if item.name in _MIRROR_DIRS:
                _mirror_dir(item, target)
            else:
                shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            _atomic_copy(item, target)


def _invalidate_sync_stamp(root: Path) -> None:
    """Delete the venv's lock fingerprint so the next launch re-syncs, even if the overlay is interrupted.

    The overlay replaces ``uv.lock`` but preserves the venv; if it stops partway, the install can be left
    with new code but the still-old lock and a stamp that matches it, so a plain launch would skip the sync
    and run the new code against the old dependencies. Clearing the stamp before any file is overlaid makes
    every outcome (partial or complete) reconcile on the next launch. Best-effort: a failed delete only
    forgoes this safety net, never the update.
    """
    with contextlib.suppress(OSError):
        paths.sync_stamp_file(root).unlink(missing_ok=True)


def perform_update(root: Path, info: UpdateInfo) -> UpdateResult:
    """Download, verify, and overlay the release named by *info*. Returns a result; never raises.

    Integrity is mandatory: the bundle's SHA-256 must be present in the published ``SHA256SUMS`` and match
    the download before any file is written, so an unverifiable or corrupt download can never overlay the
    install. The download and extraction happen in a temp dir; only the final overlay touches the install.
    """
    if not info.available or not info.bundle_url:
        return UpdateResult(False, "No update available.", info.current, info.latest)
    if not info.checksums_url:
        return UpdateResult(
            False, "No checksums were published for this release; refusing to apply.", info.current, info.latest
        )

    with tempfile.TemporaryDirectory(prefix="horde-update-") as tmp:
        tmpdir = Path(tmp)
        zip_path = tmpdir / _BUNDLE_ASSET
        try:
            zip_path.write_bytes(_http_get(info.bundle_url))
            checksums = _http_get(info.checksums_url).decode("utf-8")
        except Exception as error:  # noqa: BLE001 - surface the failure as a skipped update, never a crash
            return UpdateResult(False, f"Download failed: {error}", info.current, info.latest)

        expected = _expected_hash(checksums, _BUNDLE_ASSET)
        if expected is None:
            return UpdateResult(
                False, "The bundle's checksum was not found; refusing to apply.", info.current, info.latest
            )
        if _sha256(zip_path).lower() != expected.lower():
            return UpdateResult(False, "Checksum mismatch; refusing to apply the download.", info.current, info.latest)

        extracted = tmpdir / "extracted"
        try:
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extracted)
            # Invalidate the sync stamp before touching the install, so a crash mid-overlay still reconciles.
            _invalidate_sync_stamp(root)
            _overlay(_bundle_root(extracted), root)
        except (OSError, zipfile.BadZipFile) as error:
            return UpdateResult(False, f"Could not apply the update: {error}", info.current, info.latest)

    return UpdateResult(True, f"Updated to {info.latest}.", info.current, installed_version(root))
