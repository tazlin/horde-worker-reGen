"""Guards that the uv version in runtime scripts matches pyproject.toml's required-version.

The uv version is specified in three places:
- pyproject.toml: [tool.uv] required-version (what uv checks at runtime)
- runtime.sh: HORDE_WORKER_UV_VERSION default (what Linux/macOS downloads)
- runtime.cmd: UV_VERSION default (what Windows downloads)

When these drift, users running update scripts get "Required uv version does not match" errors.
This test ensures they stay in sync.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
_RUNTIME_SH_PATH = _REPO_ROOT / "runtime.sh"
_RUNTIME_CMD_PATH = _REPO_ROOT / "runtime.cmd"


def _extract_required_uv_version() -> str:
    """Extract the required uv version from pyproject.toml."""
    pyproject = tomllib.loads(_PYPROJECT_PATH.read_text(encoding="utf-8"))
    version = pyproject.get("tool", {}).get("uv", {}).get("required-version")
    assert version is not None, "No [tool.uv] required-version in pyproject.toml"
    return version


def _extract_runtime_sh_uv_version() -> str:
    """Extract the default uv version from runtime.sh."""
    content = _RUNTIME_SH_PATH.read_text(encoding="utf-8")
    match = re.search(r'version="\$\{HORDE_WORKER_UV_VERSION:-([^}]+)\}"', content)
    assert match is not None, "Could not find HORDE_WORKER_UV_VERSION default in runtime.sh"
    return match.group(1)


def _extract_runtime_cmd_uv_version() -> str:
    """Extract the default uv version from runtime.cmd."""
    content = _RUNTIME_CMD_PATH.read_text(encoding="utf-8")
    match = re.search(r'set "UV_VERSION=([^"]+)"', content)
    assert match is not None, "Could not find UV_VERSION in runtime.cmd"
    return match.group(1)


def test_runtime_sh_uv_version_matches_required_version() -> None:
    """The uv version downloaded by runtime.sh matches pyproject.toml's required-version."""
    required = _extract_required_uv_version()
    runtime_sh = _extract_runtime_sh_uv_version()
    assert runtime_sh == required, (
        f"runtime.sh downloads uv {runtime_sh} but pyproject.toml requires {required}. "
        f"Update the HORDE_WORKER_UV_VERSION default in runtime.sh to match."
    )


def test_runtime_cmd_uv_version_matches_required_version() -> None:
    """The uv version downloaded by runtime.cmd matches pyproject.toml's required-version."""
    required = _extract_required_uv_version()
    runtime_cmd = _extract_runtime_cmd_uv_version()
    assert runtime_cmd == required, (
        f"runtime.cmd downloads uv {runtime_cmd} but pyproject.toml requires {required}. "
        f"Update the UV_VERSION default in runtime.cmd to match."
    )


def test_all_uv_versions_are_consistent() -> None:
    """All three uv version declarations (pyproject.toml, runtime.sh, runtime.cmd) agree."""
    required = _extract_required_uv_version()
    runtime_sh = _extract_runtime_sh_uv_version()
    runtime_cmd = _extract_runtime_cmd_uv_version()

    versions = {
        "pyproject.toml": required,
        "runtime.sh": runtime_sh,
        "runtime.cmd": runtime_cmd,
    }

    unique_versions = set(versions.values())
    assert len(unique_versions) == 1, (
        f"uv versions are inconsistent across files: {versions}. All three must declare the same version."
    )


def test_runtime_sh_replaces_an_existing_mismatched_uv() -> None:
    """The POSIX bootstrap must version-check an existing private uv instead of trusting its presence."""
    content = _RUNTIME_SH_PATH.read_text(encoding="utf-8")

    assert '"$SCRIPT_DIR/bin/uv" --version' in content
    assert 'if [ "$existing_version" = "$version" ]' in content
    assert '[ -x "$SCRIPT_DIR/bin/uv" ] && return 0' not in content


def test_runtime_cmd_replaces_an_existing_mismatched_uv() -> None:
    """The Windows bootstrap must version-check an existing private uv instead of trusting its presence."""
    content = _RUNTIME_CMD_PATH.read_text(encoding="utf-8")
    pre_download_check = content.split("\n:ensure_uv\n", maxsplit=1)[1].split("\n:download_uv\n", maxsplit=1)[0]

    assert '"%~dp0bin\\uv.exe" --version' in content
    assert 'if "%UV_ACTUAL%"=="%UV_VERSION%" exit /b 0' in content
    assert 'if exist "%~dp0bin\\uv.exe" exit /b 0' not in pre_download_check


def test_runtime_sh_verifies_and_stages_uv_before_replacement() -> None:
    """The POSIX launcher never executes or publishes an unverified uv download."""
    content = _RUNTIME_SH_PATH.read_text(encoding="utf-8")

    assert '"$url.sha256"' in content
    assert 'actual="$(sha256sum' in content
    assert 'candidate="$tmp_dir/uv-${triple}/uv"' in content
    assert '"$candidate" --version' in content
    assert 'mv -f "$candidate" "$SCRIPT_DIR/bin/uv"' in content
    assert "astral.sh/uv/install.sh" not in content


def test_runtime_cmd_verifies_and_stages_uv_before_replacement() -> None:
    """Both Windows download paths verify checksum and version before replacing private uv."""
    content = _RUNTIME_CMD_PATH.read_text(encoding="utf-8")

    assert '"%UV_URL%.sha256"' in content
    assert "certutil.exe" in content
    assert '"%UV_EXPECTED:~63,1%"' in content
    assert 'findstr /L /I /X /C:"%UV_EXPECTED%"' in content
    assert '"%UV_CANDIDATE%" --version' in content
    assert "Get-FileHash -Algorithm SHA256" in content
    assert "uv version mismatch" in content
    assert "astral.sh/uv/install.ps1" not in content
