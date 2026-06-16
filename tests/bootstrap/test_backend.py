"""Unit tests for backend precedence, the legacy remap, and locked-extra validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from worker_bootstrap import backend


@pytest.mark.parametrize(
    ("token", "expected"),
    [("cu128", "cu126"), ("CU128", "cu126"), (" cu128 ", "cu126"), ("cu130", "cu130"), (" cpu ", "cpu")],
)
def test_remap_legacy(token: str, expected: str) -> None:
    """A retired cu128 token maps to cu126; everything else passes through (trimmed)."""
    assert backend.remap_legacy(token) == expected


def test_resolve_precedence() -> None:
    """CLI flag > env > file > detection > default, with the cu128 remap applied to the winner."""
    assert backend.resolve_backend(cli_flag="cu130", env_value="cpu", file_value="cu126") == "cu130"
    assert backend.resolve_backend(env_value="cpu", file_value="cu126", detected="cu130") == "cpu"
    assert backend.resolve_backend(file_value="cu126", detected="cu130") == "cu126"
    assert backend.resolve_backend(detected="cu130") == "cu130"
    assert backend.resolve_backend() == "cu126"
    assert backend.resolve_backend(file_value="cu128") == "cu126"  # stale persisted token is remapped


def test_resolve_skips_blank_sources() -> None:
    """Empty/whitespace sources are ignored so a blank env var does not win over a real file value."""
    assert backend.resolve_backend(cli_flag="", env_value="   ", file_value="cu130") == "cu130"


def test_backend_file_roundtrip(tmp_path: Path) -> None:
    """write_backend_file then read_backend_file returns the token; a missing file reads as None."""
    path = tmp_path / "bin" / "backend"
    assert backend.read_backend_file(path) is None
    backend.write_backend_file(path, "cu130")
    assert path.read_text(encoding="utf-8") == "cu130"  # no trailing newline
    assert backend.read_backend_file(path) == "cu130"


def test_read_backend_file_empty_is_none(tmp_path: Path) -> None:
    """An empty bin/backend reads as None so resolution falls through to the next source."""
    path = tmp_path / "backend"
    path.write_text("   \n", encoding="utf-8")
    assert backend.read_backend_file(path) is None


def _write_pyproject(tmp_path: Path) -> Path:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project.optional-dependencies]\ncu126 = ["torch"]\ncu130 = ["torch"]\ncu132 = ["torch"]\ncpu = ["torch"]\n',
        encoding="utf-8",
    )
    return pyproject


def test_locked_extras_reads_pyproject(tmp_path: Path) -> None:
    """locked_extras returns the optional-dependency keys (sorted)."""
    assert backend.locked_extras(_write_pyproject(tmp_path)) == ["cpu", "cu126", "cu130", "cu132"]


def test_validate_accepts_locked(tmp_path: Path) -> None:
    """A locked build extra validates without raising."""
    backend.validate_locked_extra("cu126", _write_pyproject(tmp_path))


@pytest.mark.parametrize("token", ["rocm", "amd-unsupported", "bogus"])
def test_validate_rejects_unlocked(tmp_path: Path, token: str) -> None:
    """A non-locked build raises ValueError with actionable guidance."""
    with pytest.raises(ValueError, match="not a locked build"):
        backend.validate_locked_extra(token, _write_pyproject(tmp_path))
