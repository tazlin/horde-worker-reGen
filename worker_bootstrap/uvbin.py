"""Locate and repair the private uv executable used by the bootstrap.

Managed installs deliberately preserve their top-level shell shims and ``bin/`` directory during an
in-place update.  A release that raises ``[tool.uv] required-version`` can therefore overlay the new
``pyproject.toml`` while leaving the previous private uv binary in place.  The shell shim can still run
this stdlib-only bootstrap with ``--no-project``; this module then downloads uv's exact release artifact,
verifies its published SHA-256, and uses the verified, versioned sidecar for all project operations.

Downloading a sidecar instead of invoking ``uv self update`` is important because standalone uv can be
installed in unmanaged mode, which deliberately disables self-update.  It also avoids replacing the uv
process that launched ``bootstrap.py`` (locked on Windows) and makes repair interruption-safe.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from tarfile import TarError, TarFile

from worker_bootstrap import paths

_VERSION_CHECK_TIMEOUT_SECONDS = 15
_DOWNLOAD_TIMEOUT_SECONDS = 180
_UV_RELEASE_BASE_URL = "https://github.com/astral-sh/uv/releases/download"
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class UvCompatibilityError(RuntimeError):
    """Raised when the private uv cannot be reconciled with the project's required version."""


def required_uv_version(root: Path | None = None) -> str | None:
    """Return the exact uv version pinned by ``pyproject.toml``, or ``None`` when it has no pin.

    The worker intentionally uses an exact pin: its release shims and container images are updated in
    lockstep with this value.  uv accepts broader requirement specifiers, but those provide no concrete
    target for an automatic exact-artifact repair and are therefore rejected with an actionable error.
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


def _release_archive_name() -> str:
    """Return uv's release-archive name for the current supported platform."""
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "aarch64"
    else:
        raise UvCompatibilityError(f"Automatic uv repair does not support architecture {machine or 'unknown'!r}.")

    if os.name == "nt":
        return f"uv-{arch}-pc-windows-msvc.zip"
    if sys.platform == "darwin":
        return f"uv-{arch}-apple-darwin.tar.gz"
    if sys.platform.startswith("linux"):
        return f"uv-{arch}-unknown-linux-gnu.tar.gz"
    raise UvCompatibilityError(f"Automatic uv repair does not support platform {sys.platform!r}.")


def _download(url: str) -> bytes:
    """Download one small bootstrap artifact from uv's fixed HTTPS release origin."""
    request = urllib.request.Request(url, headers={"User-Agent": "horde-worker-regen-bootstrap"})
    with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310
        return response.read()


def _published_sha256(checksum_payload: bytes, archive_name: str) -> str:
    """Parse uv's ``<archive>.sha256`` sidecar and return its validated digest."""
    try:
        fields = checksum_payload.decode("ascii").strip().split()
    except UnicodeDecodeError as error:
        raise UvCompatibilityError("uv's published checksum was not ASCII text.") from error
    if not fields or not _SHA256_PATTERN.fullmatch(fields[0]):
        raise UvCompatibilityError("uv's published checksum was malformed.")
    if len(fields) >= 2 and Path(fields[-1].lstrip("*")).name != archive_name:
        raise UvCompatibilityError(
            f"uv's published checksum named {fields[-1]!r}, but the requested artifact is {archive_name!r}."
        )
    return fields[0].lower()


def _uv_payload(archive_payload: bytes, archive_name: str) -> bytes:
    """Extract only the uv executable from a verified release archive."""
    executable_name = "uv.exe" if os.name == "nt" else "uv"
    try:
        if archive_name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
                matches = [name for name in archive.namelist() if Path(name).name == executable_name]
                if len(matches) != 1:
                    raise UvCompatibilityError(f"uv archive contained {len(matches)} {executable_name!r} entries.")
                return archive.read(matches[0])
        with TarFile.open(fileobj=io.BytesIO(archive_payload), mode="r:gz") as archive:
            matches = [
                member
                for member in archive.getmembers()
                if member.isfile() and Path(member.name).name == executable_name
            ]
            if len(matches) != 1:
                raise UvCompatibilityError(f"uv archive contained {len(matches)} {executable_name!r} entries.")
            extracted = archive.extractfile(matches[0])
            if extracted is None:
                raise UvCompatibilityError(f"Could not read {executable_name!r} from the uv archive.")
            return extracted.read()
    except (OSError, TarError, ValueError, zipfile.BadZipFile) as error:
        raise UvCompatibilityError(f"Could not unpack uv's release archive: {error}") from error


def _download_verified_uv(version: str, target: Path, *, probe_directory: Path) -> None:
    """Download, checksum, version-probe, and atomically publish an exact uv release."""
    archive_name = _release_archive_name()
    artifact_url = f"{_UV_RELEASE_BASE_URL}/{version}/{archive_name}"
    print(f"Downloading the worker's private uv {version} ...", flush=True)
    try:
        checksum = _published_sha256(_download(f"{artifact_url}.sha256"), archive_name)
        archive_payload = _download(artifact_url)
    except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
        raise UvCompatibilityError(f"Could not download uv {version}: {error}") from error
    actual_checksum = hashlib.sha256(archive_payload).hexdigest()
    if actual_checksum != checksum:
        raise UvCompatibilityError(
            f"Downloaded uv {version} failed SHA-256 verification (expected {checksum}, got {actual_checksum})."
        )

    temporary: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = ".exe" if os.name == "nt" else ""
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".uv-{version}-download-", suffix=suffix, dir=target.parent
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)
        temporary.write_bytes(_uv_payload(archive_payload, archive_name))
        if os.name != "nt":
            temporary.chmod(0o755)
        actual_version = _reported_version(str(temporary), cwd=probe_directory)
        if actual_version != version:
            raise UvCompatibilityError(
                f"The downloaded uv reported {actual_version or 'an unreadable version'}; expected {version}."
            )
        os.replace(temporary, target)
    except UvCompatibilityError:
        raise
    except OSError as error:
        raise UvCompatibilityError(f"Could not publish the worker's private uv {version}: {error}") from error
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)


def ensure_compatible_uv(root: Path | None = None, executable: str | None = None) -> str:
    """Return a uv satisfying the project's exact pin, self-repairing a managed install when necessary.

    Repair never mutates the executable that is currently hosting the bootstrap or a PATH-provided uv.
    Instead it downloads uv's exact platform archive, verifies the adjacent checksum published by uv,
    probes the extracted executable, and atomically publishes it as ``bin/uv-<version>[.exe]``.  A future
    bootstrap reuses the sidecar, so an old preserved shell shim and an unmanaged uv are harmless.

    Raises:
        UvCompatibilityError: If an exact version is required but no verified compatible private uv can
            be found or produced.
    """
    install_root = paths.install_root() if root is None else root
    selected = uv_executable(install_root) if executable is None else executable
    required = required_uv_version(install_root)
    if required is None:
        return selected

    versioned = _versioned_uv_path(install_root, required)
    # uv checks [tool.uv] required-version even for some non-project commands. Run every compatibility
    # probe outside the install, otherwise the stale executable can reject the new pin before we learn its
    # actual version.
    try:
        with tempfile.TemporaryDirectory(prefix="horde-worker-uv-command-") as command_directory_name:
            command_directory = Path(command_directory_name)
            if versioned.exists() and _reported_version(str(versioned), cwd=command_directory) == required:
                return str(versioned)
            if _reported_version(selected, cwd=command_directory) == required:
                return selected

            _download_verified_uv(required, versioned, probe_directory=command_directory)
    except UvCompatibilityError:
        raise
    except OSError as error:
        raise UvCompatibilityError(f"Could not prepare uv {required} for bootstrap: {error}") from error
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
