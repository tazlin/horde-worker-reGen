"""Guard: no shipped source file may be silently gitignored.

The worker ships as a files-only release bundle built from git-tracked files: the release workflow runs
``actions/checkout`` (tracked files only) then ``cp -r horde_worker_regen`` / ``cp -r worker_bootstrap``.
A gitignore pattern that accidentally matches a source directory therefore drops that module from the
bundle, so a fresh install crashes with ``ModuleNotFoundError`` while the developer's working tree (which
still has the untracked files on disk) runs fine and hides the gap.

That is exactly how a bare ``testing`` / ``dummy`` line once swallowed ``process_management/simulation``.
This test fails the moment any importable source file under a bundled package is ignored, so the class of
bug is caught in CI instead of on a user's machine.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# The packages the release workflow copies recursively into the bundle (see packaging/bundle-include.txt
# and .github/workflows/release.yml). Anything importable under these must be tracked to actually ship.
BUNDLED_PACKAGES = ("horde_worker_regen", "worker_bootstrap")


def test_no_bundled_source_file_is_gitignored() -> None:
    """Every ``.py`` under a bundled package must be tracked (not matched by any gitignore rule)."""
    git = shutil.which("git")
    if git is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("git not available; cannot check ignore status")

    sources = [
        path
        for package in BUNDLED_PACKAGES
        for path in (REPO_ROOT / package).rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    assert sources, "found no source files to check; the package layout moved"

    rel_paths = "\n".join(str(path.relative_to(REPO_ROOT).as_posix()) for path in sources)
    # check-ignore --stdin prints (only) the paths that match an ignore rule; empty output means all clear.
    result = subprocess.run(
        [git, "-C", str(REPO_ROOT), "check-ignore", "--stdin"],
        input=rel_paths,
        capture_output=True,
        text=True,
        check=False,
    )
    ignored = [line for line in result.stdout.splitlines() if line.strip()]
    assert not ignored, (
        "these shipped source files are gitignored and would be missing from the release bundle "
        f"(fresh installs would crash on import): {ignored}"
    )
