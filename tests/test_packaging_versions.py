"""Guards that every hand-maintained copy of the release version agrees with ``__version__``.

The release version lives in one place (``horde_worker_regen.__version__``, which ``pyproject.toml``
sources via ``[tool.hatch.version]``). The winget manifests under ``packaging/winget/`` are the only
other files that must restate it. The two winget guards below are skipped while winget publishing is
paused (see ``docs/how-to/enable-winget-publishing.md``) so a release bump need not also sync the dormant
manifests; un-skip them when re-enabling. ``packaging/sync-winget-version.py`` is the intended writer.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from horde_worker_regen import __version__

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WINGET_PAUSED = "winget publishing is paused; see docs/how-to/enable-winget-publishing.md"
_WINGET_DIR = _REPO_ROOT / "packaging" / "winget"
_WINGET_MANIFESTS = (
    _WINGET_DIR / "Haidra.HordeWorker.installer.yaml",
    _WINGET_DIR / "Haidra.HordeWorker.yaml",
    _WINGET_DIR / "Haidra.HordeWorker.locale.en-US.yaml",
)


def _package_version(path: Path) -> str:
    match = re.search(r"(?m)^PackageVersion:\s*(\S+)\s*$", path.read_text(encoding="utf-8"))
    assert match is not None, f"No PackageVersion line in {path.name}"
    return match.group(1)


def test_pyproject_sources_version_from_init() -> None:
    """Pyproject derives the version from __init__.py rather than restating it (single source).

    This is the structural guard that keeps the two files from drifting again: ``version`` must be
    declared dynamic, never pinned statically, and ``[tool.hatch.version].path`` must point at the file
    that defines ``__version__``.
    """
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "version" in pyproject["project"].get("dynamic", []), "project.version must be dynamic"
    assert "version" not in pyproject["project"], "project.version must not be statically pinned"
    assert pyproject["tool"]["hatch"]["version"]["path"] == "horde_worker_regen/__init__.py"


@pytest.mark.skip(reason=_WINGET_PAUSED)
def test_winget_package_versions_match_version() -> None:
    """Every winget manifest's PackageVersion equals the release version."""
    for manifest in _WINGET_MANIFESTS:
        assert _package_version(manifest) == __version__, f"{manifest.name} PackageVersion drifted"


@pytest.mark.skip(reason=_WINGET_PAUSED)
def test_winget_installer_url_tag_matches_version() -> None:
    """The installer download URL points at the matching ``v{version}`` release tag."""
    installer = _WINGET_DIR / "Haidra.HordeWorker.installer.yaml"
    match = re.search(r"/releases/download/v([^/]+)/", installer.read_text(encoding="utf-8"))
    assert match is not None, "No tagged InstallerUrl in the winget installer manifest"
    assert match.group(1) == __version__, "Installer URL tag drifted from the release version"
