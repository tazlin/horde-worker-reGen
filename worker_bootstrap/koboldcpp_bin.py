"""Provision the koboldcpp executable the worker's text-generation workload drives as a child process.

Operators cannot be asked to build koboldcpp: a CUDA build needs Visual Studio or a full GCC toolchain plus
the CUDA toolkit, and the result is a native library the worker would then have to locate. Upstream instead
publishes a single-file PyInstaller binary per release and per accelerator variant, so this module downloads
one, exactly as :mod:`worker_bootstrap.uvbin` downloads uv's exact release artifact: a pinned version, a
pinned SHA-256 per supported asset, verification before anything is published, and an atomic publish to a
stable path.

Public surface:

- :data:`KOBOLDCPP_VERSION`: the pinned upstream release tag; a bump changes the pin table in one diff.
- :class:`KoboldcppVariant`: CUDA or no-CUDA (Vulkan plus CPU) asset selection.
- :class:`KoboldcppAsset`: one published asset with its pinned digest.
- :func:`koboldcpp_executable`: the published binary if this install has one.
- :func:`ensure_koboldcpp`: idempotently download, verify, publish and probe the pinned binary.

Nothing here is wired into a launch path yet; provisioning is on demand and the caller decides when.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.client
import os
import platform
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

from worker_bootstrap import detect, paths

KOBOLDCPP_VERSION = "v1.121"
"""The pinned upstream release tag. Bumping it also replaces every digest in the asset table below."""

_RELEASE_BASE_URL = "https://github.com/LostRuins/koboldcpp/releases/download"
# urllib applies this per blocking socket operation rather than to the whole transfer, so it bounds a
# stalled connection without capping how long a 600 MB body may legitimately take.
_DOWNLOAD_TIMEOUT_SECONDS = 900
# A one-file PyInstaller binary unpacks itself into a temporary directory on every run; for a 600 MB
# artifact on a cold disk that alone can take a couple of minutes before the process prints anything.
_VERSION_PROBE_TIMEOUT_SECONDS = 240
_DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024
_PROGRESS_STEP_FRACTION = 0.1
_USER_AGENT = "horde-worker-regen-bootstrap"


class KoboldcppProvisionError(RuntimeError):
    """Raised when a verified koboldcpp binary cannot be produced for this install."""


class UnsupportedKoboldcppPlatformError(KoboldcppProvisionError):
    """Raised when upstream publishes no asset this module is willing to run on the host platform."""


class KoboldcppVariant(StrEnum):
    """Which accelerator backends a koboldcpp asset carries."""

    CUDA = auto()
    """The full build, carrying CUDA alongside the Vulkan and CPU backends."""
    NOCUDA = auto()
    """The smaller build, carrying Vulkan and CPU only: the right choice for AMD, Intel and CPU-only hosts."""


class KoboldcppPlatform(StrEnum):
    """A host platform upstream publishes a supported single-file binary for."""

    WINDOWS_X64 = auto()
    LINUX_X64 = auto()


@dataclass(frozen=True)
class KoboldcppAsset:
    """Represents one published release binary, identified by host platform and accelerator variant."""

    host_platform: KoboldcppPlatform
    """The platform the asset runs on."""
    variant: KoboldcppVariant
    """The accelerator backends the asset carries."""
    asset_name: str
    """The file name upstream publishes the asset under."""
    sha256: str
    """The pinned digest of the published asset, checked before the binary can reach the stable path."""

    @property
    def download_url(self) -> str:
        """Return the pinned release's download URL for this asset."""
        return f"{_RELEASE_BASE_URL}/{KOBOLDCPP_VERSION}/{self.asset_name}"


# Upstream ships no checksum files, so each digest is pinned here from the release API's asset `digest`
# field. Verifying against a pin rather than against whatever the download claims is the point: a bump is a
# reviewed diff, and an intercepted or truncated body cannot publish itself.
_SUPPORTED_ASSETS: tuple[KoboldcppAsset, ...] = (
    KoboldcppAsset(
        host_platform=KoboldcppPlatform.WINDOWS_X64,
        variant=KoboldcppVariant.CUDA,
        asset_name="koboldcpp.exe",
        sha256="90b0d74ec01e5ef72efb6d45e6f10bee649458920ec951f48d58794c366b1639",
    ),
    KoboldcppAsset(
        host_platform=KoboldcppPlatform.WINDOWS_X64,
        variant=KoboldcppVariant.NOCUDA,
        asset_name="koboldcpp-nocuda.exe",
        sha256="ce94894823b19b32efe5db8b6cc8bbc459db23fa98fda16da978fbec960473b7",
    ),
    KoboldcppAsset(
        host_platform=KoboldcppPlatform.LINUX_X64,
        variant=KoboldcppVariant.CUDA,
        asset_name="koboldcpp-linux-x64",
        sha256="463a5eb1392f0c40e6b5a9031ff77eeb9ae4fe8db4018c9108eb1f6fab659024",
    ),
    KoboldcppAsset(
        host_platform=KoboldcppPlatform.LINUX_X64,
        variant=KoboldcppVariant.NOCUDA,
        asset_name="koboldcpp-linux-x64-nocuda",
        sha256="5939cb137d382a7095b82c330bb62579740066604861f0b57e59e49a20597666",
    ),
)

_ASSETS_BY_TARGET: dict[tuple[KoboldcppPlatform, KoboldcppVariant], KoboldcppAsset] = {
    (asset.host_platform, asset.variant): asset for asset in _SUPPORTED_ASSETS
}

_X64_MACHINES = ("amd64", "x86_64")


def pinned_version_number() -> str:
    """Return the pinned version as koboldcpp itself reports it, without the release tag's leading ``v``."""
    return KOBOLDCPP_VERSION.removeprefix("v")


def _platform_for(*, os_name: str, sys_platform: str, machine: str) -> KoboldcppPlatform:
    """Return the supported platform for the given interpreter and machine identifiers.

    Args:
        os_name: The value of ``os.name`` (``"nt"`` on Windows).
        sys_platform: The value of ``sys.platform``.
        machine: The value of ``platform.machine()``.

    Returns:
        The matching :class:`KoboldcppPlatform`.

    Raises:
        UnsupportedKoboldcppPlatformError: If upstream publishes no asset this module supports for that
            platform. Upstream's ``-oldpc`` and ``-mac-arm64`` builds are deliberately not wired up, so an
            Apple Silicon or 32-bit host is reported as unsupported rather than served a guessed asset.
    """
    reported_machine = machine or "an unknown machine"
    is_x64 = machine.lower() in _X64_MACHINES
    if os_name == "nt":
        if not is_x64:
            raise UnsupportedKoboldcppPlatformError(
                f"koboldcpp is only provisioned for Windows x64; this host reports {reported_machine!r}."
            )
        return KoboldcppPlatform.WINDOWS_X64
    if sys_platform.startswith("linux"):
        if not is_x64:
            raise UnsupportedKoboldcppPlatformError(
                f"koboldcpp is only provisioned for Linux x64; this host reports {reported_machine!r}."
            )
        return KoboldcppPlatform.LINUX_X64
    raise UnsupportedKoboldcppPlatformError(
        f"koboldcpp is only provisioned for Windows x64 and Linux x64; this host reports "
        f"{sys_platform!r} on {reported_machine!r}."
    )


def host_platform() -> KoboldcppPlatform:
    """Return the supported platform this interpreter is running on.

    Raises:
        UnsupportedKoboldcppPlatformError: If the host platform has no supported published asset.
    """
    return _platform_for(os_name=os.name, sys_platform=sys.platform, machine=platform.machine())


def default_variant() -> KoboldcppVariant:
    """Return the variant to install when the caller does not name one.

    An NVIDIA card gets the CUDA build; everything else gets the no-CUDA build, which still carries the
    Vulkan and CPU backends and is therefore the working choice on AMD, Intel and CPU-only hosts. The ROCm
    fork is a separate upstream project and is not a variant here.
    """
    return KoboldcppVariant.CUDA if detect._nvidia_present() else KoboldcppVariant.NOCUDA


def koboldcpp_asset(*, variant: KoboldcppVariant | None = None) -> KoboldcppAsset:
    """Return the pinned asset for this host, defaulting the variant to what the hardware wants.

    Args:
        variant: Force a variant instead of deriving it from the detected hardware.

    Returns:
        The :class:`KoboldcppAsset` to download, with its pinned digest.

    Raises:
        UnsupportedKoboldcppPlatformError: If the host platform has no supported published asset.
    """
    return _ASSETS_BY_TARGET[(host_platform(), variant or default_variant())]


def _executable_suffix() -> str:
    """Return the host's executable suffix, so a published path is runnable as-is."""
    return ".exe" if os.name == "nt" else ""


def _stable_path(root: Path) -> Path:
    """Return the stable published path callers and the child-process launch read."""
    return paths.bin_dir(root) / f"koboldcpp{_executable_suffix()}"


def _versioned_path(root: Path) -> Path:
    """Return the staging path a download is verified and probed at before the stable path changes."""
    return paths.bin_dir(root) / f"koboldcpp-{KOBOLDCPP_VERSION}{_executable_suffix()}"


def _version_stamp_path(root: Path) -> Path:
    """Return the sidecar recording which release the stable path currently holds.

    The stable path cannot carry the version in its name and still be the one name a launcher knows, and a
    600 MB binary is not worth storing twice, so the published version is recorded beside it instead.
    """
    return paths.bin_dir(root) / "koboldcpp-version"


def _published_version(root: Path) -> str | None:
    """Return the release tag recorded for the published binary, or None when nothing is recorded."""
    try:
        recorded = _version_stamp_path(root).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return recorded or None


def _reported_version(executable: Path) -> str | None:
    """Return what the binary prints for ``--version``, or None when the probe could not produce a version."""
    try:
        result = subprocess.run(  # noqa: S603 - the argument is a path this module just verified and published
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=_VERSION_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _stream_download_to(url: str, destination: Path) -> str:
    """Stream a release asset to *destination*, returning the SHA-256 of exactly what was written.

    The asset is hundreds of megabytes, so it is hashed chunk by chunk while it lands on disk; it is never
    held in memory the way uv's small archive is.

    Returns:
        The lowercase hex SHA-256 digest of the written bytes.

    Raises:
        OSError: If the transfer or the write fails.
        http.client.HTTPException: If the HTTP response is malformed or truncated.
    """
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    written = 0
    with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310
        raw_length = response.headers.get("Content-Length")
        total = int(raw_length) if raw_length is not None and raw_length.isdigit() else 0
        progress_step = int(total * _PROGRESS_STEP_FRACTION)
        next_progress = progress_step
        with destination.open("wb") as handle:
            while chunk := response.read(_DOWNLOAD_CHUNK_BYTES):
                digest.update(chunk)
                handle.write(chunk)
                written += len(chunk)
                if progress_step and written >= next_progress:
                    print(f"  ... {written // (1024 * 1024)} MiB of {total // (1024 * 1024)} MiB", flush=True)
                    next_progress = written + progress_step
    return digest.hexdigest()


def _download_verified_koboldcpp(asset: KoboldcppAsset, root: Path) -> Path:
    """Download, checksum, publish and probe *asset*, returning the stable published path.

    The download lands on a temporary sibling, is compared to the pinned digest, and only then becomes the
    versioned staging file. The stable path changes last, after the staged binary has reported the pinned
    version, so a failure at any step leaves whatever was already published runnable.

    Raises:
        KoboldcppProvisionError: If the download, the digest comparison, the publish or the version probe
            fails.
    """
    stable = _stable_path(root)
    versioned = _versioned_path(root)
    print(f"Downloading koboldcpp {KOBOLDCPP_VERSION} ({asset.asset_name}, {asset.variant}) ...", flush=True)

    temporary: Path | None = None
    try:
        paths.bin_dir(root).mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".koboldcpp-{KOBOLDCPP_VERSION}-download-",
            suffix=_executable_suffix(),
            dir=paths.bin_dir(root),
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)
        try:
            actual_sha256 = _stream_download_to(asset.download_url, temporary)
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            raise KoboldcppProvisionError(f"Could not download koboldcpp {KOBOLDCPP_VERSION}: {error}") from error
        if actual_sha256 != asset.sha256:
            raise KoboldcppProvisionError(
                f"Downloaded {asset.asset_name} failed SHA-256 verification "
                f"(expected {asset.sha256}, got {actual_sha256})."
            )
        if os.name != "nt":
            temporary.chmod(0o755)
        os.replace(temporary, versioned)
        temporary = None
    except KoboldcppProvisionError:
        raise
    except OSError as error:
        raise KoboldcppProvisionError(f"Could not stage koboldcpp {KOBOLDCPP_VERSION}: {error}") from error
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)

    print("Verifying the downloaded koboldcpp (a one-file binary unpacks itself first) ...", flush=True)
    reported = _reported_version(versioned)
    if reported is None or pinned_version_number() not in reported:
        with contextlib.suppress(OSError):
            versioned.unlink(missing_ok=True)
        raise KoboldcppProvisionError(
            f"The downloaded koboldcpp reported {reported or 'no readable version'}; "
            f"expected {pinned_version_number()}. Nothing was published."
        )

    try:
        # Clear the stamp first: a crash between the rename and the stamp write costs one redundant
        # download, while a crash the other way round would claim the old binary is the new release.
        _version_stamp_path(root).unlink(missing_ok=True)
        os.replace(versioned, stable)
        _version_stamp_path(root).write_text(f"{KOBOLDCPP_VERSION}\n", encoding="utf-8")
    except OSError as error:
        raise KoboldcppProvisionError(
            f"Could not publish koboldcpp {KOBOLDCPP_VERSION} to {stable}: {error}"
        ) from error
    print(f"koboldcpp {KOBOLDCPP_VERSION} is ready at {stable}.", flush=True)
    return stable


def koboldcpp_executable(root: Path | None = None) -> Path | None:
    """Return the published koboldcpp binary for this install, or None when it has not been provisioned.

    Args:
        root: The install root to look under; defaults to this bundle's own install root.

    Returns:
        The stable published path when it exists, otherwise None. The path is returned without probing the
        binary, so a caller that needs a guaranteed-runnable executable calls :func:`ensure_koboldcpp`.
    """
    stable = _stable_path(paths.install_root() if root is None else root)
    return stable if stable.exists() else None


def ensure_koboldcpp(root: Path | None = None, *, variant: KoboldcppVariant | None = None) -> Path:
    """Return a verified koboldcpp binary for this install, downloading the pinned release if needed.

    Idempotent: when the stable path already holds the pinned release it is returned without any network
    access or subprocess probe. Otherwise the pinned asset is downloaded, compared to its pinned SHA-256,
    staged under a versioned name, probed with ``--version``, and only then published to the stable path.

    Args:
        root: The install root to provision into; defaults to this bundle's own install root.
        variant: Force a variant instead of deriving it from the detected hardware.

    Returns:
        The stable published path of the verified binary.

    Raises:
        UnsupportedKoboldcppPlatformError: If the host platform has no supported published asset.
        KoboldcppProvisionError: If the download, the digest comparison, the publish or the version probe
            fails. An already published binary is left in place and runnable.

    Side Effects:
        Writes into ``bin/``: the published binary and the sidecar recording its release.
    """
    install_root = paths.install_root() if root is None else root
    published = koboldcpp_executable(install_root)
    if published is not None and _published_version(install_root) == KOBOLDCPP_VERSION:
        return published
    return _download_verified_koboldcpp(koboldcpp_asset(variant=variant), install_root)


def _main(argv: list[str] | None = None) -> int:
    """Provision koboldcpp into the current install root and print the resulting path."""
    parser = argparse.ArgumentParser(
        prog="python -m worker_bootstrap.koboldcpp_bin",
        description=f"Download and verify the pinned koboldcpp {KOBOLDCPP_VERSION} binary for this install.",
    )
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in KoboldcppVariant],
        default=None,
        help="Force an accelerator variant instead of choosing one from the detected hardware.",
    )
    arguments = parser.parse_args(argv)
    selected = KoboldcppVariant(arguments.variant) if arguments.variant is not None else None
    try:
        print(ensure_koboldcpp(variant=selected))
    except KoboldcppProvisionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
