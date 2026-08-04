"""Explicit legacy runtime for NVIDIA JetPack 5 (L4T R35 / CUDA 11.4).

The current worker requires Python 3.12 and a modern PyTorch stack. JetPack 5 cannot load those CUDA
wheels, and PyPI's ``cu118`` index does not publish the required aarch64 builds. This profile therefore
keeps the modern installer as the control plane while provisioning a pinned, patched v9.0.7 worker in the
preserved data directory with NVIDIA's local JetPack wheels.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from worker_bootstrap import paths, updater

LEGACY_COMMIT = "da894e8b63d22fe3ca7fbbafb0280f69134a60e5"
LEGACY_ARCHIVE_SHA256 = "2dcb22bd7a351eda0d9cdcb756396c2817f194ac07b721d4a5e07b97e1bbe696"
PYTHON_VERSION = "3.10.20"
REQUIRED_WHEEL_NAMES: tuple[str, ...] = (
    "torch-2.1.0a0+git7bcf7da-cp310-cp310-linux_aarch64.whl",
    "torchvision-0.16.0+fbb4cc5-cp310-cp310-linux_aarch64.whl",
    "torchaudio-2.1.0+6ea1133-cp310-cp310-linux_aarch64.whl",
)
REQUIRED_WHEEL_SHA256: dict[str, str] = {
    REQUIRED_WHEEL_NAMES[0]: "1079d7eaf5e6be486a534bc1546ce355e53fa63494eab9bb72611ca48d3f9cdf",
    REQUIRED_WHEEL_NAMES[1]: "0ddb4140f0827ec89349a61d47c0d6c5319c469686da501c6113b48c3faba3b1",
    REQUIRED_WHEEL_NAMES[2]: "00204769e28772eb12bb53a3cc33b61e8d1c7be640c44edfd23b587dff4f3566",
}
XFORMERS_WHEEL_SHA256 = "496de486460855325ba10afb81b560f30cddd8aa592077efa877f6d7bb884672"
_XFORMERS_GLOB = "xformers-0.0.23+e1b36f7.d*-cp310-cp310-linux_aarch64.whl"
_L4T_RELEASE = Path("/etc/nv_tegra_release")
_L4T_MAJOR_RE = re.compile(r"^#\s*R(\d+)\b")
_SOURCE_STAMP = ".horde-jetpack5-source"
_INSTALL_STAMP = ".horde-jetpack5-install"


class JetPack5Error(RuntimeError):
    """A clear, operator-actionable JetPack 5 profile failure."""


def parse_l4t_major_release(text: str) -> int | None:
    """Return the L4T major release from NVIDIA's ``/etc/nv_tegra_release`` marker."""
    match = _L4T_MAJOR_RE.search(text.strip())
    return int(match.group(1)) if match else None


def read_l4t_release(path: Path = _L4T_RELEASE) -> str:
    """Read the NVIDIA L4T release marker, returning an empty string when it is absent."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def validate_host(*, system: str | None = None, machine: str | None = None, l4t_release: str | None = None) -> None:
    """Require the one platform this compatibility runtime is built and tested for."""
    actual_system = system or platform.system()
    actual_machine = machine or platform.machine()
    release = read_l4t_release() if l4t_release is None else l4t_release
    linux_arm64 = actual_system == "Linux" and actual_machine in ("aarch64", "arm64")
    supported = linux_arm64 and parse_l4t_major_release(release) == 35
    if not supported:
        raise JetPack5Error(
            "The jetpack5 backend only supports NVIDIA JetPack 5 (L4T R35) on Linux aarch64; "
            f"detected {actual_system} {actual_machine} with {release.strip() or 'no L4T release marker'}."
        )


def runtime_root(root: Path) -> Path:
    """Return the preserved directory containing the pinned legacy worker runtime."""
    return paths.data_root(root) / "jetpack5-runtime"


def wheel_dir() -> Path:
    """Return the operator-provided directory containing NVIDIA's JetPack 5 wheels."""
    return Path(os.environ.get("HORDE_WORKER_JETPACK5_WHEELS", str(Path.home() / "jetson"))).expanduser()


def _verify_checksum(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected.lower():
        raise JetPack5Error(f"JetPack 5 wheel checksum mismatch for {path.name}: expected {expected}, got {actual}.")


def _xformers_checksum(wheel: Path) -> str:
    checksum_file = wheel.with_suffix(f"{wheel.suffix}.sha256")
    try:
        checksum = checksum_file.read_text(encoding="ascii").split()[0].lower()
    except (OSError, IndexError) as error:
        raise JetPack5Error(
            f"Missing or invalid xFormers checksum sidecar: {checksum_file}. "
            "Rebuild the wheel with build-xformers-jetson-jp5.sh."
        ) from error
    if re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise JetPack5Error(f"Invalid SHA256 value in xFormers checksum sidecar: {checksum_file}")
    return checksum


def resolve_wheels(
    directory: Path,
    *,
    required_hashes: dict[str, str] | None = None,
    xformers_hash: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Resolve and verify the exact tested CUDA 11.4 wheel set."""
    hashes = REQUIRED_WHEEL_SHA256 if required_hashes is None else required_hashes
    required = tuple(directory / name for name in REQUIRED_WHEEL_NAMES)
    for wheel in required:
        if not wheel.is_file():
            raise JetPack5Error(
                f"Required JetPack 5 wheel not found: {wheel}. Generic cu118 wheels are not a substitute "
                "for NVIDIA's aarch64 JetPack CUDA 11.4 builds. Set HORDE_WORKER_JETPACK5_WHEELS to the "
                "directory containing the tested wheel set."
            )
        expected = hashes.get(wheel.name)
        if expected is None:
            raise JetPack5Error(f"No pinned checksum is configured for required JetPack 5 wheel: {wheel.name}")
        _verify_checksum(wheel, expected)
    xformers = sorted(directory.glob(_XFORMERS_GLOB))
    if len(xformers) != 1:
        raise JetPack5Error(
            f"Expected exactly one {_XFORMERS_GLOB} wheel in {directory}, found {len(xformers)}. "
            "Build it with build-xformers-jetson-jp5.sh on the Jetson, using one compiler worker."
        )
    expected_xformers = (
        xformers_hash
        or os.environ.get("HORDE_WORKER_JETPACK5_XFORMERS_SHA256", "").strip().lower()
        or XFORMERS_WHEEL_SHA256
    )
    if re.fullmatch(r"[0-9a-f]{64}", expected_xformers) is None:
        raise JetPack5Error("HORDE_WORKER_JETPACK5_XFORMERS_SHA256 must be a 64-character lowercase SHA256.")
    sidecar_xformers = _xformers_checksum(xformers[0])
    if sidecar_xformers != expected_xformers:
        raise JetPack5Error(
            f"JetPack 5 wheel checksum mismatch for {xformers[0].name}: "
            f"trusted {expected_xformers}, sidecar contains {sidecar_xformers}."
        )
    _verify_checksum(xformers[0], expected_xformers)
    return required[0], required[1], required[2], xformers[0]


def _patch_file(root: Path) -> Path:
    return root / "requirements" / "jetpack5" / "runtime.patch"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_id(root: Path) -> str:
    patch = _patch_file(root)
    if not patch.is_file():
        raise JetPack5Error(f"JetPack 5 compatibility patch is missing from the release bundle: {patch}")
    return f"{LEGACY_COMMIT}:{_sha256(patch)}"


def _safe_extract(archive: Path, destination: Path) -> Path:
    """Extract a single-root tar archive while rejecting path traversal and link escapes."""
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if target != base and base not in target.parents:
                raise JetPack5Error(f"Legacy runtime archive contains an unsafe path: {member.name}")
            if member.issym() or member.islnk():
                raise JetPack5Error(f"Legacy runtime archive contains an unsupported link: {member.name}")
        bundle.extractall(destination, members=members, filter="data")
    roots = [entry for entry in destination.iterdir() if entry.is_dir()]
    if len(roots) != 1:
        raise JetPack5Error(f"Legacy runtime archive should contain one root directory, found {len(roots)}")
    return roots[0]


def _source_archive_url(root: Path) -> str:
    override = os.environ.get("HORDE_WORKER_JETPACK5_SOURCE_URL", "").strip()
    if override:
        return override
    repo = updater.resolve_update_repo(root)
    return f"https://github.com/{repo}/archive/{LEGACY_COMMIT}.tar.gz"


def _download_archive(destination: Path, root: Path) -> None:
    url = _source_archive_url(root)
    expected = os.environ.get("HORDE_WORKER_JETPACK5_SOURCE_SHA256", LEGACY_ARCHIVE_SHA256)
    print(f"Downloading the pinned JetPack 5 worker runtime ({LEGACY_COMMIT[:12]})...", flush=True)
    try:
        urllib.request.urlretrieve(url, destination)  # noqa: S310 - URL is pinned or explicitly operator-provided
    except OSError as error:
        raise JetPack5Error(f"Could not download the JetPack 5 legacy runtime from {url}: {error}") from error
    actual = _sha256(destination)
    if actual != expected:
        raise JetPack5Error(
            f"JetPack 5 runtime archive checksum mismatch: expected {expected}, got {actual}. "
            "Refusing to execute unverified source."
        )


def _materialize_runtime(root: Path) -> Path:
    """Download the pinned source and apply the bundled compatibility patch exactly once."""
    destination = runtime_root(root)
    profile_id = _profile_id(root)
    source_stamp = destination / _SOURCE_STAMP
    if destination.is_dir() and source_stamp.is_file():
        if source_stamp.read_text(encoding="utf-8").strip() != profile_id:
            raise JetPack5Error(
                f"The JetPack 5 profile changed while a legacy runtime already exists at {destination}. "
                "Move that directory aside and re-run the installer so the old working runtime remains backed up."
            )
        return destination
    if destination.exists():
        raise JetPack5Error(
            f"Refusing to overwrite the unrecognized JetPack 5 runtime path {destination}; move it aside and retry."
        )
    git = shutil.which("git")
    if not git:
        raise JetPack5Error("git is required to apply the JetPack 5 compatibility patch.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jetpack5-", dir=destination.parent) as temporary:
        temp = Path(temporary)
        archive = temp / "legacy-runtime.tar.gz"
        _download_archive(archive, root)
        extracted = _safe_extract(archive, temp / "source")
        result = subprocess.run(
            [git, "apply", "--whitespace=nowarn", str(_patch_file(root))],
            cwd=extracted,
            check=False,
        )
        if result.returncode != 0:
            raise JetPack5Error("The JetPack 5 compatibility patch did not apply to the pinned legacy source.")
        (extracted / _SOURCE_STAMP).write_text(profile_id, encoding="utf-8")
        shutil.move(str(extracted), destination)
    return destination


def _validate_python310(python: Path) -> Path:
    if not python.is_file():
        raise JetPack5Error(f"Python 3.10 executable not found: {python}")
    result = subprocess.run(
        [str(python), "-c", "import sys; raise SystemExit(sys.version_info[:3] != (3, 10, 20))"],
        check=False,
    )
    if result.returncode != 0:
        raise JetPack5Error(f"JetPack 5 requires Python {PYTHON_VERSION}, but {python} is a different version.")
    return python


def _python310(uv: str, root: Path) -> Path:
    override = os.environ.get("HORDE_WORKER_JETPACK5_PYTHON")
    if override:
        return _validate_python310(Path(override).expanduser())
    if found := shutil.which("python3.10"):
        return _validate_python310(Path(found))
    pyenv = Path.home() / ".pyenv" / "versions" / PYTHON_VERSION / "bin" / "python"
    if pyenv.is_file():
        return _validate_python310(pyenv)

    print(f"Installing managed Python {PYTHON_VERSION} for the JetPack 5 runtime...", flush=True)
    install = subprocess.run([uv, "python", "install", PYTHON_VERSION], cwd=root, check=False)
    if install.returncode != 0:
        raise JetPack5Error(
            f"Could not provision Python {PYTHON_VERSION}. Set HORDE_WORKER_JETPACK5_PYTHON to that exact version."
        )
    located = subprocess.run(
        [uv, "python", "find", PYTHON_VERSION],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if located.returncode != 0 or not located.stdout.strip():
        raise JetPack5Error(f"uv installed Python {PYTHON_VERSION} but could not locate its executable.")
    return _validate_python310(Path(located.stdout.strip().splitlines()[-1]))


def _replace_with_symlink(link: Path, target: Path) -> None:
    """Create a profile-owned link without overwriting a user-owned regular file or directory."""
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        return
    link.symlink_to(target)


def _share_operator_files(root: Path, legacy: Path) -> None:
    config = root / "bridgeData.yaml"
    if not config.exists():
        shutil.copy2(legacy / "bridgeData_template.yaml", config)
    _replace_with_symlink(legacy / "bridgeData.yaml", config)
    dotenv = root / ".env"
    if dotenv.exists():
        _replace_with_symlink(legacy / ".env", dotenv)
    _replace_with_symlink(root / ".venv", legacy / ".venv")


def _install_runtime(uv: str, root: Path, legacy: Path) -> None:
    profile_id = _profile_id(root)
    install_stamp = legacy / _INSTALL_STAMP
    python = legacy / ".venv" / "bin" / "python"
    installed_profile = install_stamp.read_text(encoding="utf-8").strip() if install_stamp.is_file() else None
    if installed_profile == profile_id and python.is_file():
        _share_operator_files(root, legacy)
        return

    wheels = resolve_wheels(wheel_dir())
    python310 = _python310(uv, root)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": str(python310),
            "JETSON_WHEEL_DIR": str(wheels[0].parent),
            "TORCH_WHEEL": str(wheels[0]),
            "TORCH_WHEEL_SHA256": REQUIRED_WHEEL_SHA256[wheels[0].name],
            "TORCHVISION_WHEEL": str(wheels[1]),
            "TORCHVISION_WHEEL_SHA256": REQUIRED_WHEEL_SHA256[wheels[1].name],
            "TORCHAUDIO_WHEEL": str(wheels[2]),
            "TORCHAUDIO_WHEEL_SHA256": REQUIRED_WHEEL_SHA256[wheels[2].name],
            "XFORMERS_WHEEL": str(wheels[3]),
            "XFORMERS_WHEEL_SHA256": (
                os.environ.get("HORDE_WORKER_JETPACK5_XFORMERS_SHA256", "").strip().lower() or XFORMERS_WHEEL_SHA256
            ),
            "CMAKE_BUILD_PARALLEL_LEVEL": "1",
            "MAKEFLAGS": "-j1",
            "MAX_JOBS": "1",
            "NINJAFLAGS": "-j1",
        }
    )
    print("Installing the pinned JetPack 5 worker runtime with one build thread...", flush=True)
    result = subprocess.run(["sh", str(legacy / "install-jetson-jp5.sh")], cwd=legacy, env=environment, check=False)
    if result.returncode != 0:
        raise JetPack5Error(f"JetPack 5 runtime installation failed with exit code {result.returncode}.")
    install_stamp.write_text(profile_id, encoding="utf-8")
    _share_operator_files(root, legacy)


def sync_jetpack5(uv: str, *, root: Path) -> int:
    """Provision or verify the isolated legacy runtime; return a CLI-style status code."""
    try:
        validate_host()
        legacy = _materialize_runtime(root)
        _install_runtime(uv, root, legacy)
    except JetPack5Error as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"JetPack 5 legacy runtime ready at {legacy}")
    return 0


def _clean_rest(rest: list[str]) -> list[str]:
    return rest[1:] if rest[:1] == ["--"] else rest


def launch_jetpack5(*, root: Path, mode: str, rest: list[str]) -> int:
    """Launch the legacy worker. Its v9 runtime is headless, regardless of the modern UI selector."""
    legacy = runtime_root(root)
    if mode == "benchmark":
        print("ERROR: the JetPack 5 legacy runtime does not provide the modern benchmark command.", file=sys.stderr)
        return 2
    if mode != "bridge":
        print(
            f"JetPack 5 uses the legacy headless worker; '{mode}' UI mode is unavailable. Starting headless.",
            file=sys.stderr,
        )
    return subprocess.run(
        ["sh", str(legacy / "start-jetson-jp5.sh"), *_clean_rest(rest)],
        cwd=legacy,
        check=False,
    ).returncode


def preload_jetpack5(*, root: Path) -> int:
    """Download and verify models using the legacy runtime without starting the worker."""
    legacy = runtime_root(root)
    python = legacy / ".venv" / "bin" / "python"
    command = 'set -a; if [ -f ./.env ]; then . ./.env; fi; set +a; exec "$1" -s "$2"'
    return subprocess.run(
        ["sh", "-c", command, "jetpack5-preload", str(python), str(legacy / "download_models.py")],
        cwd=legacy,
        check=False,
    ).returncode
