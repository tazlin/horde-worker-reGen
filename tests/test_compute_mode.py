"""Unit tests for the torch-free compute-mode sentinel reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from horde_worker_regen.compute_mode import (
    ComputeMode,
    compute_mode_display_label,
    intended_compute_mode,
    is_cpu_only_install,
    read_backend_token,
    reconcile_with_probe,
)

_BACKEND_ENV = "HORDE_WORKER_BACKEND"


@pytest.fixture(autouse=True)
def _clear_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure a developer's HORDE_WORKER_BACKEND never leaks into these tests."""
    monkeypatch.delenv(_BACKEND_ENV, raising=False)


def _write_backend(tmp_path: Path, token: str) -> Path:
    backend_file = tmp_path / "bin" / "backend"
    backend_file.parent.mkdir(parents=True, exist_ok=True)
    backend_file.write_text(token, encoding="utf-8")
    return backend_file


def test_cpu_token_is_cpu_mode(tmp_path: Path) -> None:
    """A 'cpu' token classifies as CPU mode / alchemist-only."""
    backend_file = _write_backend(tmp_path, "cpu")
    assert intended_compute_mode(backend_file=backend_file) is ComputeMode.CPU
    assert is_cpu_only_install(backend_file=backend_file) is True


@pytest.mark.parametrize("token", ["cu126", "cu130", "cu132", "rocm", "rocm-windows", "xpu"])
def test_gpu_tokens_are_accelerated(tmp_path: Path, token: str) -> None:
    """Every GPU/accelerator build token classifies as accelerated (not CPU-only)."""
    backend_file = _write_backend(tmp_path, token)
    assert intended_compute_mode(backend_file=backend_file) is ComputeMode.ACCELERATED
    assert is_cpu_only_install(backend_file=backend_file) is False


def test_missing_sentinel_is_unknown(tmp_path: Path) -> None:
    """A missing sentinel yields unknown intent (None), not a forced CPU assumption."""
    # No file written, and the package-relative fallback should not exist under tmp.
    backend_file = tmp_path / "bin" / "backend"
    assert intended_compute_mode(backend_file=backend_file) is None
    assert is_cpu_only_install(backend_file=backend_file) is False


def test_env_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HORDE_WORKER_BACKEND takes precedence over the persisted file."""
    backend_file = _write_backend(tmp_path, "cu132")
    monkeypatch.setenv(_BACKEND_ENV, "cpu")
    assert read_backend_token(backend_file=backend_file) == "cpu"
    assert is_cpu_only_install(backend_file=backend_file) is True


def test_legacy_cu128_normalised(tmp_path: Path) -> None:
    """A retired cu128 token folds onto cu126 and still classifies as accelerated."""
    backend_file = _write_backend(tmp_path, "cu128")
    assert read_backend_token(backend_file=backend_file) == "cu126"
    assert intended_compute_mode(backend_file=backend_file) is ComputeMode.ACCELERATED


def test_token_is_case_insensitive(tmp_path: Path) -> None:
    """An upper-case token still classifies correctly."""
    backend_file = _write_backend(tmp_path, "CPU")
    assert is_cpu_only_install(backend_file=backend_file) is True


def test_reconcile_gpu_intent_no_accelerator_warns(tmp_path: Path) -> None:
    """A GPU intent with a CPU-only probe result warns about a missing/broken accelerator."""
    backend_file = _write_backend(tmp_path, "cu132")
    message = reconcile_with_probe(["cpu"], backend_file=backend_file)
    assert message is not None
    assert "no accelerator" in message.lower()


def test_reconcile_cpu_intent_with_accelerator_warns(tmp_path: Path) -> None:
    """A CPU intent with a real accelerator present warns that the GPU is idle."""
    backend_file = _write_backend(tmp_path, "cpu")
    message = reconcile_with_probe(["cuda"], backend_file=backend_file)
    assert message is not None
    assert "alchemist" in message.lower()


def test_reconcile_matching_is_silent(tmp_path: Path) -> None:
    """Matching intent and hardware produce no warning."""
    backend_file = _write_backend(tmp_path, "cu132")
    assert reconcile_with_probe(["cuda"], backend_file=backend_file) is None
    cpu_file = _write_backend(tmp_path, "cpu")
    assert reconcile_with_probe(["cpu"], backend_file=cpu_file) is None


def test_reconcile_unknown_intent_is_silent(tmp_path: Path) -> None:
    """Unknown intent (no sentinel) never warns regardless of probe result."""
    backend_file = tmp_path / "bin" / "backend"
    assert reconcile_with_probe(["cpu"], backend_file=backend_file) is None


def test_display_label_cpu(tmp_path: Path) -> None:
    """A CPU install gets a UI label calling out alchemist-only mode."""
    backend_file = _write_backend(tmp_path, "cpu")
    assert compute_mode_display_label(backend_file=backend_file) == "CPU (alchemist-only)"


def test_display_label_gpu_is_none(tmp_path: Path) -> None:
    """A GPU install adds no label (the dashboard is unchanged)."""
    backend_file = _write_backend(tmp_path, "cu132")
    assert compute_mode_display_label(backend_file=backend_file) is None
