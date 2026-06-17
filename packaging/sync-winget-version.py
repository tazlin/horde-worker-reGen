#!/usr/bin/env python3
"""Rewrite the winget manifests' version (and download URL / hash) to a given release version.

The winget manifests under ``packaging/winget/`` carry the release version in three places that must all
agree with ``horde_worker_regen.__version__``: ``PackageVersion`` in each of the three files and the
tagged ``InstallerUrl`` in the installer manifest. This script is the single writer of those fields so a
release bump never has to hand-edit (and drift) them; ``tests/test_packaging_versions.py`` guards the
result.

Usage (run from the repo root)::

    python packaging/sync-winget-version.py 12.5.13
    python packaging/sync-winget-version.py 12.5.13 --sha256 <hex>

With no ``--version`` argument the version is read from ``horde_worker_regen/__init__.py`` so CI can call
it without restating the number. ``--sha256`` is optional (wingetcreate/komac usually fill the real hash
from the published asset); when given it replaces the installer hash too.

Stdlib only, so it can run in the release workflow without installing the project.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_WINGET_DIR = Path(__file__).resolve().parent / "winget"
_INIT_FILE = Path(__file__).resolve().parent.parent / "horde_worker_regen" / "__init__.py"

_INSTALLER = _WINGET_DIR / "Haidra.HordeWorker.installer.yaml"
_VERSION_FILE = _WINGET_DIR / "Haidra.HordeWorker.yaml"
_LOCALE = _WINGET_DIR / "Haidra.HordeWorker.locale.en-US.yaml"

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+([.-][0-9A-Za-z.-]+)?$")


def read_init_version() -> str:
    """Read ``__version__`` from ``horde_worker_regen/__init__.py`` without importing the package."""
    match = re.search(r'__version__\s*=\s*"([^"]+)"', _INIT_FILE.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"Could not find __version__ in {_INIT_FILE}")
    return match.group(1)


def _sub_once(path: Path, pattern: str, replacement: str) -> None:
    """Replace the first match of *pattern* in *path*, erroring if it is not present exactly once."""
    text = path.read_text(encoding="utf-8")
    new_text, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise SystemExit(f"Expected exactly one match for {pattern!r} in {path.name}, found {count}.")
    path.write_text(new_text, encoding="utf-8")


def sync(version: str, sha256: str | None = None) -> None:
    """Rewrite PackageVersion in all manifests and the tagged InstallerUrl (and hash if given)."""
    if not _SEMVER_RE.match(version):
        raise SystemExit(f"Version {version!r} does not look like a release version (e.g. 12.5.13).")

    for path in (_INSTALLER, _VERSION_FILE, _LOCALE):
        _sub_once(path, r"(?m)^PackageVersion:.*$", f"PackageVersion: {version}")

    _sub_once(
        _INSTALLER,
        r"(?m)^(\s*InstallerUrl:\s*https://github\.com/Haidra-Org/horde-worker-reGen/releases/download/)v[^/]+(/.*)$",
        rf"\g<1>v{version}\g<2>",
    )

    if sha256 is not None:
        _sub_once(_INSTALLER, r"(?m)^(\s*InstallerSha256:\s*).*$", rf"\g<1>{sha256.lower()}")

    print(f"Synced winget manifests to v{version}" + (f" (sha256 {sha256.lower()})" if sha256 else ""))


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and sync the manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?", help="Release version (defaults to __init__.py's __version__).")
    parser.add_argument("--sha256", help="Installer zip SHA-256 (optional).")
    args = parser.parse_args(argv)

    version = args.version or read_init_version()
    sync(version, args.sha256)


if __name__ == "__main__":
    main(sys.argv[1:])
