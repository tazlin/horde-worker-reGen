"""Resolve a usable ``git``, the one runtime dependency the worker cannot install with uv.

``hordelib.initialise()`` shells out to a system ``git`` on the first job to clone and pin ComfyUI plus a
couple of custom nodes, so a machine with no git would fail mid-job with an opaque error. This module makes
git a satisfied dependency: an existing git on PATH is always preferred (no download), and only when none
is found do we fall back to a portable MinGit on Windows. On Linux/macOS, where git is a one-line package
install, we surface clear guidance instead of bundling a second copy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from worker_bootstrap import paths

# MinGit (git-for-windows) is the portable, redistributable git build. Pinned for reproducibility; bump
# here or override per-run with HORDE_WORKER_MINGIT_VERSION. Only x64 is published as MinGit; on Windows
# ARM64 it runs under x64 emulation (this is a fallback path, the common case has or installs system git).
_DEFAULT_MINGIT_VERSION = "2.47.1"


@dataclass(frozen=True)
class GitResolution:
    """The outcome of resolving git for this install."""

    git_path: str | None
    """Path to the resolved git executable (system or bundled), or None when git is unavailable."""
    source: str
    """Where git came from: ``"system"``, ``"mingit"`` (bundled), or ``"missing"``."""
    message: str
    """A human-readable status line (or actionable error guidance when git is missing)."""

    @property
    def ok(self) -> bool:
        """True when a usable git was resolved."""
        return self.git_path is not None


def _can_provision() -> bool:
    """Return True where we bundle a portable git as a fallback (Windows only)."""
    return os.name == "nt"


def _mingit_version() -> str:
    """Return the MinGit version to fetch (pinned, overridable via env)."""
    return os.environ.get("HORDE_WORKER_MINGIT_VERSION", _DEFAULT_MINGIT_VERSION)


def mingit_url() -> str:
    """Return the download URL for the pinned MinGit archive."""
    version = _mingit_version()
    return f"https://github.com/git-for-windows/git/releases/download/v{version}.windows.1/MinGit-{version}-64-bit.zip"


def mingit_git_exe(root: Path | None = None) -> Path:
    """Return the path where the bundled MinGit's ``git.exe`` lives once unpacked."""
    return paths.git_cmd_dir(root) / "git.exe"


def find_system_git() -> str | None:
    """Return the path to a working ``git`` on PATH, or None.

    Verified with a ``git --version`` probe so a broken shim (a stub that resolves but cannot run) is
    treated as absent rather than trusted.
    """
    exe = shutil.which("git")
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0 and "git version" in (result.stdout or "").lower():
        return exe
    return None


def notice_line(system_git: str | None) -> str:
    """Return the install-notice line describing how git will be handled this run."""
    if system_git:
        return f"  - git: using the git already on your PATH ({system_git})."
    if _can_provision():
        return (
            f"  - git: none found on your PATH; a portable MinGit ({_mingit_version()}) will be downloaded "
            "from github.com/git-for-windows/git (GPLv2) for the worker's own use."
        )
    return (
        "  - git: none found on your PATH. The worker needs git to fetch ComfyUI on first run; install it "
        "(e.g. 'sudo apt install git', 'sudo dnf install git', or 'brew install git') before continuing."
    )


def provision_mingit(root: Path | None = None) -> Path:
    """Download and unpack the pinned MinGit into ``bin/git``; return the unpacked ``git.exe`` path.

    Raises:
        OSError: If the download fails or the archive does not contain git where expected.
        zipfile.BadZipFile: If the downloaded archive is corrupt.
    """
    target = paths.git_dir(root)
    target.mkdir(parents=True, exist_ok=True)
    url = mingit_url()
    print(f"Downloading a portable git (MinGit {_mingit_version()}) from {url} ...")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "mingit.zip"
        _download(url, archive)
        # The MinGit archive's root is the git tree itself (cmd/, mingw64/, ...), so extracting into
        # bin/git lands bin/git/cmd/git.exe directly.
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    git_exe = mingit_git_exe(root)
    if not git_exe.exists():
        raise OSError(f"the MinGit archive did not contain git at the expected location ({git_exe})")
    print("Portable git is ready.")
    return git_exe


def _download(url: str, dest: Path, *, retries: int = 3, timeout: int = 60) -> None:
    """Download ``url`` to ``dest`` with a few retries (URLError is an OSError subclass)."""
    last_error: OSError | None = None
    for _ in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "horde-worker-bootstrap"})
            with urllib.request.urlopen(request, timeout=timeout) as response, dest.open("wb") as handle:
                shutil.copyfileobj(response, handle)  # type: ignore
            return
        except OSError as exc:
            last_error = exc
    raise OSError(f"download failed after {retries} attempts: {last_error}")


def ensure_git(root: Path | None = None, *, download: bool = True) -> GitResolution:
    """Resolve git for the install: prefer system git, else (Windows) provision MinGit, else fail clearly."""
    system_git = find_system_git()
    if system_git:
        return GitResolution(system_git, "system", f"Using the git already on PATH: {system_git}")

    if not _can_provision():
        return GitResolution(
            None,
            "missing",
            "ERROR: git was not found on your PATH, and the worker needs it to fetch ComfyUI on first run. "
            "Install git (e.g. 'sudo apt install git', 'sudo dnf install git', or 'brew install git') and "
            "re-run.",
        )

    git_exe = mingit_git_exe(root)
    if git_exe.exists():
        return GitResolution(str(git_exe), "mingit", f"Using the bundled portable git at {git_exe}")
    if not download:
        return GitResolution(
            None,
            "missing",
            "git is not on PATH and the bundled portable git has not been downloaded yet.",
        )
    try:
        git_exe = provision_mingit(root)
    except (OSError, zipfile.BadZipFile) as exc:
        return GitResolution(
            None,
            "missing",
            f"ERROR: could not provision a portable git ({exc}). Install git from https://git-scm.com/ "
            "(or git-for-windows.github.io) and re-run.",
        )
    return GitResolution(str(git_exe), "mingit", f"Downloaded a portable git to {git_exe}")
