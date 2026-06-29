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


def test_offer_cpu_returns_cpu_on_c_answer() -> None:
    """Choosing CPU at the prompt overrides an auto-detected GPU build."""
    token = backend.choose_backend_interactively(
        "cu132",
        explicitly_chosen=False,
        interactive=True,
        prompt=lambda _: "c",
        emit=lambda _: None,
    )
    assert token == "cpu"


def test_offer_cpu_default_keeps_gpu() -> None:
    """Pressing enter (or anything but C) keeps the detected GPU build."""
    token = backend.choose_backend_interactively(
        "cu132",
        explicitly_chosen=False,
        interactive=True,
        prompt=lambda _: "",
        emit=lambda _: None,
    )
    assert token == "cu132"


def test_offer_cpu_respects_explicit_choice() -> None:
    """An explicit --backend / env choice is never second-guessed (no prompt)."""

    def _boom(_: str) -> str:
        raise AssertionError("must not prompt when the backend was chosen explicitly")

    token = backend.choose_backend_interactively(
        "cu132",
        explicitly_chosen=True,
        interactive=True,
        prompt=_boom,
        emit=lambda _: None,
    )
    assert token == "cu132"


def test_offer_cpu_no_prompt_when_non_interactive() -> None:
    """A non-interactive run never prompts and keeps the detected build."""

    def _boom(_: str) -> str:
        raise AssertionError("must not prompt in a non-interactive run")

    token = backend.choose_backend_interactively(
        "cu132",
        explicitly_chosen=False,
        interactive=False,
        prompt=_boom,
        emit=lambda _: None,
    )
    assert token == "cu132"


def test_offer_cpu_no_prompt_when_already_cpu() -> None:
    """When no GPU was detected (token already cpu) there is nothing to choose, so no prompt."""

    def _boom(_: str) -> str:
        raise AssertionError("must not prompt when the detected build is already cpu")

    token = backend.choose_backend_interactively(
        "cpu",
        explicitly_chosen=False,
        interactive=True,
        prompt=_boom,
        emit=lambda _: None,
    )
    assert token == "cpu"


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


def _write_pyproject_with_features(tmp_path: Path) -> Path:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project.optional-dependencies]\ncu126 = ["torch"]\ncpu = ["torch"]\n'
        'controlnet = ["horde-engine[controlnet]"]\npost-processing = ["horde-engine[rembg]"]\n',
        encoding="utf-8",
    )
    return pyproject


def test_locked_extras_reads_pyproject(tmp_path: Path) -> None:
    """locked_extras returns the optional-dependency keys (sorted)."""
    assert backend.locked_extras(_write_pyproject(tmp_path)) == ["cpu", "cu126", "cu130", "cu132"]


def test_locked_extras_excludes_feature_extras(tmp_path: Path) -> None:
    """Feature extras are not torch builds, so locked_extras filters them out."""
    assert backend.locked_extras(_write_pyproject_with_features(tmp_path)) == ["cpu", "cu126"]


def test_validate_accepts_locked(tmp_path: Path) -> None:
    """A locked build extra validates without raising."""
    backend.validate_locked_extra("cu126", _write_pyproject(tmp_path))


@pytest.mark.parametrize("token", ["rocm", "amd-unsupported", "bogus"])
def test_validate_rejects_unlocked(tmp_path: Path, token: str) -> None:
    """A non-locked build raises ValueError with actionable guidance."""
    with pytest.raises(ValueError, match="not a locked build"):
        backend.validate_locked_extra(token, _write_pyproject(tmp_path))


def test_validate_rejects_feature_extra_as_build(tmp_path: Path) -> None:
    """A feature extra is not a valid backend token even though it is a real extra."""
    with pytest.raises(ValueError, match="not a locked build"):
        backend.validate_locked_extra("post-processing", _write_pyproject_with_features(tmp_path))


@pytest.mark.parametrize("token", ["cu126", "cu130", "cu132", "cpu"])
def test_full_builds_default_to_all_feature_extras(token: str) -> None:
    """NVIDIA/CPU builds keep the full feature set by default (zero-change UX)."""
    assert backend.desired_feature_extras(token) == backend.FEATURE_EXTRAS


@pytest.mark.parametrize("token", ["rocm", "rocm-windows", "xpu", "mps"])
def test_lean_builds_default_to_no_feature_extras(token: str) -> None:
    """Non-NVIDIA backends default lean: no feature extras unless opted in."""
    assert backend.desired_feature_extras(token) == ()


def test_feature_extras_env_override_opts_in_on_lean_build() -> None:
    """HORDE_WORKER_FEATURES opts a lean backend into specific extras."""
    assert backend.desired_feature_extras("rocm", env_value="controlnet") == ("controlnet",)
    assert backend.desired_feature_extras("rocm", env_value="controlnet, post-processing") == (
        "controlnet",
        "post-processing",
    )


def test_feature_extras_env_none_forces_lean_on_full_build() -> None:
    """An explicit 'none' (or empty) override strips features even from a full build."""
    assert backend.desired_feature_extras("cu132", env_value="none") == ()
    assert backend.desired_feature_extras("cu132", env_value="   ") == backend.FEATURE_EXTRAS


def test_feature_extras_env_rejects_unknown() -> None:
    """An override naming a non-feature extra raises with guidance."""
    with pytest.raises(ValueError, match="unknown feature extra"):
        backend.desired_feature_extras("cu126", env_value="cu126")
