"""Locate and repair the private uv executable used by the bootstrap.

Managed installs deliberately preserve their top-level shell shims and ``bin/`` directory during an
in-place update.  A release that raises ``[tool.uv] required-version`` can therefore overlay the new
``pyproject.toml`` while leaving the previous private uv binary in place.  The shell shim can still run
this stdlib-only bootstrap with ``--no-project``; this module then upgrades a *copy* of that known-working
binary and uses the verified, versioned sidecar for all project operations.

Updating a copy is important on Windows, where the uv process that launched ``bootstrap.py`` still has
``bin/uv.exe`` open.  It also makes repair interruption-safe: the old bootstrap binary remains runnable
until the replacement has downloaded and passed a version check.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

from worker_bootstrap import paths

_VERSION_CHECK_TIMEOUT_SECONDS = 15
_SELF_UPDATE_TIMEOUT_SECONDS = 180


class UvCompatibilityError(RuntimeError):
    """Raised when the private uv cannot be reconciled with the project's required version."""


def required_uv_version(root: Path | None = None) -> str | None:
    """Return the exact uv version pinned by ``pyproject.toml``, or ``None`` when it has no pin.

    The worker intentionally uses an exact pin: its release shims and container images are updated in
    lockstep with this value.  uv accepts broader requirement specifiers, but those provide no concrete
    target for an automatic ``uv self update`` and are therefore rejected with an actionable error.
    """
    project_file = paths.install_root() / "pyproject.toml" if root is None else root / "pyproject.toml"
    try:
        project = tomllib.loads(project_file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    raw = project.get("tool", {}).get("uv", {}).get("required-version")
    if not isinstance(raw, str) or not raw.strip():
        return None
    required = raw.strip()
    if required.startswith("=="):
        required = required[2:].strip()
    if not required or any(character in required for character in "<>=!~, *"):
        raise UvCompatibilityError(
            f"[tool.uv] required-version must be an exact version for automatic bootstrap repair; got {raw!r}."
        )
    return required


def _reported_version(executable: str, *, cwd: Path) -> str | None:
    """Return the version reported by *executable*, keeping the probe outside the uv project."""
    try:
        result = subprocess.run(  # noqa: S603 - executable is the bundled uv or an explicit PATH resolution
            [executable, "--version"],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=_VERSION_CHECK_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    fields = result.stdout.strip().split()
    return fields[1] if len(fields) >= 2 and fields[0] == "uv" else None


def _versioned_uv_path(root: Path, version: str) -> Path:
    """Return the stable sidecar path for *version*."""
    suffix = ".exe" if os.name == "nt" else ""
    return paths.bin_dir(root) / f"uv-{version}{suffix}"


def _bundled_uv_path(root: Path) -> Path:
    """Return the ordinary private uv path laid down by the runtime shim."""
    name = "uv.exe" if os.name == "nt" else "uv"
    return paths.bin_dir(root) / name


def ensure_compatible_uv(root: Path | None = None, executable: str | None = None) -> str:
    """Return a uv satisfying the project's exact pin, self-repairing a managed install when necessary.

    Repair never mutates the executable that is currently hosting the bootstrap.  Instead it copies that
    standalone binary to a temporary sibling, asks the copy to self-update, verifies its reported version,
    and atomically publishes it as ``bin/uv-<version>[.exe]``.  A future bootstrap reuses the sidecar, so a
    permanently old preserved shell shim is harmless.  PATH-provided uv in a developer checkout is never
    copied or modified; developers retain ownership of their tool installation.

    Raises:
        UvCompatibilityError: If an exact version is required but no verified compatible private uv can
            be found or produced.
    """
    install_root = paths.install_root() if root is None else root
    selected = uv_executable(install_root) if executable is None else executable
    required = required_uv_version(install_root)
    if required is None:
        return selected

    bin_dir = paths.bin_dir(install_root)
    versioned = _versioned_uv_path(install_root, required)
    if versioned.exists() and _reported_version(str(versioned), cwd=bin_dir) == required:
        return str(versioned)
    if _reported_version(selected, cwd=bin_dir) == required:
        return selected

    bundled = _bundled_uv_path(install_root)
    selected_unresolved = Path(selected).resolve(strict=False)
    bundled_unresolved = bundled.resolve(strict=False)
    if not bundled.exists():
        if selected_unresolved != bundled_unresolved:
            raise UvCompatibilityError(
                f"uv {required} is required, but the PATH-provided uv is incompatible. "
                f"Update it with `uv self update {required}`."
            )
        raise UvCompatibilityError(
            f"uv {required} is required, but the worker's private uv binary is missing. Re-run the worker installer."
        )
    bin_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if os.name == "nt" else ""
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".uv-{required}-repair-", suffix=suffix, dir=bin_dir)
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(bundled, temporary)
        print(f"Updating the worker's private uv to {required} ...", flush=True)
        try:
            result = subprocess.run(  # noqa: S603 - temporary is our private copy of the bundled executable
                [str(temporary), "self", "update", required],
                cwd=str(bin_dir),
                check=False,
                capture_output=True,
                text=True,
                timeout=_SELF_UPDATE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise UvCompatibilityError(f"Could not update the worker's private uv to {required}: {error}") from error
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            suffix_text = f" ({detail})" if detail else ""
            raise UvCompatibilityError(
                f"Could not update the worker's private uv to {required}{suffix_text}. "
                f"Run `./bin/uv self update {required}` and retry."
            )
        actual = _reported_version(str(temporary), cwd=bin_dir)
        if actual != required:
            raise UvCompatibilityError(
                f"The repaired uv reported {actual or 'an unreadable version'}; expected {required}."
            )
        os.replace(temporary, versioned)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"The worker's private uv is now {required}.")
    return str(versioned)


def uv_executable(root: Path | None = None) -> str:
    """Return the path to uv: bundled ``bin/uv[.exe]`` if present, else ``uv`` from PATH.

    Mirrors ``runtime.cmd``'s precedence so a packaged install uses its pinned uv while a dev checkout uses
    whatever uv is on PATH.
    """
    install_root = paths.install_root() if root is None else root
    name = "uv.exe" if os.name == "nt" else "uv"
    try:
        required = required_uv_version(install_root)
    except UvCompatibilityError:
        required = None
    if required is not None:
        versioned = _versioned_uv_path(install_root, required)
        if versioned.exists():
            return str(versioned)
    bundled = _bundled_uv_path(install_root)
    if bundled.exists():
        return str(bundled)
    return shutil.which("uv") or name
