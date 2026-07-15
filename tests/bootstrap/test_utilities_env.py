"""Unit tests for the utilities-venv provisioning plan, stamp bookkeeping, and want/need resolution."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from worker_bootstrap import backend, paths, utilities_env


def _write_requirements(root: Path, token: str, *, hashed: bool = False) -> Path:
    """Write a requirements pin for *token* under *root*, optionally carrying a hash line."""
    path = utilities_env.utilities_requirements_file(token=token, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "horde-image-utilities==1.2.3"
    if hashed:
        body += " --hash=sha256:" + "0" * 64
    path.write_text(body + "\n", encoding="utf-8")
    return path


def test_requirements_file_naming(tmp_path: Path) -> None:
    """The pin path follows requirements/utilities.<token>.txt at the install root."""
    assert utilities_env.utilities_requirements_file(token="cu132", root=tmp_path) == (
        tmp_path / "requirements" / "utilities.cu132.txt"
    )


@pytest.mark.parametrize("token", ["cu126", "cu130", "cu132", "cpu"])
def test_repo_has_a_bootstrap_seed_for_every_locked_full_backend(token: str) -> None:
    """Every full build that wants utilities by default ships a provisionable requirements seed."""
    root = Path(__file__).parents[2]
    assert utilities_env.utilities_requirements_file(token=token, root=root).is_file()


@pytest.mark.parametrize("token", ["cu126", "cu130", "cu132", "cpu"])
@pytest.mark.parametrize("package", ["torch", "torchvision"])
def test_seed_torch_build_matches_lock(token: str, package: str) -> None:
    """Each seed pins the exact torch/torchvision build uv.lock resolves for that backend.

    The utilities venv installs from the seed and hardlinks its wheels from the same peered uv cache the
    main ``.venv`` uses, so it only avoids re-downloading the multi-GB torch stack when it requests the
    *identical* build the main sync already fetched. If a lock bump moved torch and a seed did not follow
    (or vice versa), the utilities install would resolve a divergent build the cache cannot serve. Pinning
    this equality keeps the two environments cache-shareable.
    """
    root = Path(__file__).parents[2]
    seed = utilities_env.utilities_requirements_file(token=token, root=root).read_text(encoding="utf-8")
    match = re.search(rf"^{package}==(\S+)$", seed, re.MULTILINE)
    assert match, f"seed for {token} does not pin {package}=="
    build = match.group(1)  # e.g. 2.12.1+cu132
    lock = (root / "uv.lock").read_text(encoding="utf-8")
    assert f'version = "{build}"' in lock, (
        f"seed utilities.{token}.txt pins {package}=={build}, which uv.lock does not resolve; the utilities "
        f"venv would download a {package} build the main .venv's cache cannot serve"
    )


def test_plan_creates_then_installs_without_hashes(tmp_path: Path) -> None:
    """A plain (un-hashed) pin yields a clean venv-create then a pip install with no --require-hashes."""
    _write_requirements(tmp_path, "cu126", hashed=False)
    commands = utilities_env.plan_utilities_provision(uv="UV", backend_token="cu126", root=tmp_path)

    assert commands == [
        [
            "UV",
            "venv",
            "--clear",
            str(paths.utilities_venv_dir(tmp_path)),
            "--python",
            utilities_env.UTILITIES_PYTHON_VERSION,
        ],
        [
            "UV",
            "pip",
            "install",
            "--python",
            str(paths.utilities_python(tmp_path)),
            "-r",
            str(utilities_env.utilities_requirements_file(token="cu126", root=tmp_path)),
        ],
    ]


def test_plan_venv_create_is_non_interactive(tmp_path: Path) -> None:
    """The venv-create step passes --clear so a reprovision never hangs on uv's 'replace it?' prompt.

    uv >=0.8 prompts before replacing an existing venv (and uv >=0.10 requires --clear to replace one), so
    without this flag a stale-venv reprovision would block on stdin. The utilities lane must stay fully
    managed.
    """
    _write_requirements(tmp_path, "cu126")
    create, _install = utilities_env.plan_utilities_provision(uv="UV", backend_token="cu126", root=tmp_path)
    assert "--clear" in create


def test_plan_adds_require_hashes_when_pin_is_hashed(tmp_path: Path) -> None:
    """A hashed pin makes the install step pass --require-hashes (auto-detected from file content)."""
    _write_requirements(tmp_path, "cu132", hashed=True)
    _create, install = utilities_env.plan_utilities_provision(uv="UV", backend_token="cu132", root=tmp_path)
    assert "--require-hashes" in install
    assert install.index("--require-hashes") < install.index("-r")


def test_plan_require_hashes_flag_overrides_detection(tmp_path: Path) -> None:
    """An explicit require_hashes flag wins over auto-detection in both directions."""
    _write_requirements(tmp_path, "cpu", hashed=False)
    _c1, forced_on = utilities_env.plan_utilities_provision(
        uv="UV", backend_token="cpu", root=tmp_path, require_hashes=True
    )
    assert "--require-hashes" in forced_on

    _write_requirements(tmp_path, "cpu", hashed=True)
    _c2, forced_off = utilities_env.plan_utilities_provision(
        uv="UV", backend_token="cpu", root=tmp_path, require_hashes=False
    )
    assert "--require-hashes" not in forced_off


def test_plan_uses_platform_interpreter_path(tmp_path: Path) -> None:
    """The install step targets the utilities venv interpreter (Scripts/ on Windows, bin/ elsewhere)."""
    _write_requirements(tmp_path, "cu126")
    _create, install = utilities_env.plan_utilities_provision(uv="UV", backend_token="cu126", root=tmp_path)
    assert str(paths.utilities_python(tmp_path)) in install


@pytest.mark.parametrize("token", ["cu126", "cu130", "cu132", "cpu"])
def test_provision_wanted_full_builds_default_on(token: str) -> None:
    """Full NVIDIA/CPU builds want the utilities venv by default (their feature set is non-empty)."""
    assert utilities_env.utilities_provision_wanted(token=token) is True


@pytest.mark.parametrize("token", ["rocm", "rocm-windows", "xpu", "mps"])
def test_provision_wanted_lean_builds_default_off(token: str) -> None:
    """Lean backends do not want the utilities venv unless features are opted in."""
    assert utilities_env.utilities_provision_wanted(token=token) is False


def test_provision_wanted_env_override_opts_lean_in() -> None:
    """HORDE_WORKER_FEATURES opting a lean backend into a feature makes the utilities venv wanted."""
    assert utilities_env.utilities_provision_wanted(token="rocm", env_value="controlnet") is True


def test_provision_wanted_env_none_opts_full_out() -> None:
    """An explicit 'none' strips features and so makes the utilities venv unwanted even on a full build."""
    assert utilities_env.utilities_provision_wanted(token="cu132", env_value="none") is False


def test_provision_wanted_rejects_unknown_extra() -> None:
    """An override naming a non-feature extra propagates the backend resolution's ValueError."""
    with pytest.raises(ValueError, match="unknown feature extra"):
        utilities_env.utilities_provision_wanted(token="cu126", env_value="not-a-feature")


def test_provision_wanted_matches_backend_resolution() -> None:
    """Want is exactly 'the resolved feature set is non-empty' (ties to backend.desired_feature_extras)."""
    for token in ("cu126", "cpu", "rocm", "mps"):
        expected = bool(backend.desired_feature_extras(token))
        assert utilities_env.utilities_provision_wanted(token=token) is expected


def test_needs_provision_false_when_no_requirements_file(tmp_path: Path) -> None:
    """With no committed pin for the token, there is nothing to provision yet (stays a no-op)."""
    assert utilities_env.needs_provision(backend_token="cu126", root=tmp_path) is False


def test_needs_provision_true_when_venv_interpreter_missing(tmp_path: Path) -> None:
    """A committed pin but no venv interpreter means the venv must be provisioned."""
    _write_requirements(tmp_path, "cu126")
    assert utilities_env.needs_provision(backend_token="cu126", root=tmp_path) is True


def test_needs_provision_true_when_no_stamp(tmp_path: Path) -> None:
    """An existing interpreter but no stamp is treated as stale (provision once)."""
    _write_requirements(tmp_path, "cu126")
    paths.utilities_python(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    paths.utilities_python(tmp_path).write_text("", encoding="utf-8")
    assert utilities_env.needs_provision(backend_token="cu126", root=tmp_path) is True


def test_needs_provision_false_when_stamp_matches(tmp_path: Path) -> None:
    """A matching stamp (same token + same requirements digest) proves the venv is current."""
    _write_requirements(tmp_path, "cu126")
    paths.utilities_python(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    paths.utilities_python(tmp_path).write_text("", encoding="utf-8")
    utilities_env.write_utilities_stamp(backend_token="cu126", root=tmp_path)
    assert utilities_env.needs_provision(backend_token="cu126", root=tmp_path) is False


def test_needs_provision_true_when_requirements_changed(tmp_path: Path) -> None:
    """A changed requirements pin (different digest) invalidates the stamp and forces reprovision."""
    _write_requirements(tmp_path, "cu126")
    paths.utilities_python(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    paths.utilities_python(tmp_path).write_text("", encoding="utf-8")
    utilities_env.write_utilities_stamp(backend_token="cu126", root=tmp_path)

    utilities_env.utilities_requirements_file(token="cu126", root=tmp_path).write_text(
        "horde-image-utilities==9.9.9\n", encoding="utf-8"
    )
    assert utilities_env.needs_provision(backend_token="cu126", root=tmp_path) is True


def test_needs_provision_true_when_backend_token_changed(tmp_path: Path) -> None:
    """A stamp recorded for a different backend token does not satisfy a provision for this one."""
    _write_requirements(tmp_path, "cu126")
    _write_requirements(tmp_path, "cu132")
    paths.utilities_python(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    paths.utilities_python(tmp_path).write_text("", encoding="utf-8")
    utilities_env.write_utilities_stamp(backend_token="cu126", root=tmp_path)
    assert utilities_env.needs_provision(backend_token="cu132", root=tmp_path) is True


def test_stamp_roundtrip_and_content(tmp_path: Path) -> None:
    """The stamp persists the backend token and the requirements SHA256, and reads back equal."""
    requirements = _write_requirements(tmp_path, "cu130")
    utilities_env.write_utilities_stamp(backend_token="cu130", root=tmp_path)

    stamp = utilities_env.read_utilities_stamp(tmp_path)
    assert stamp is not None
    assert stamp.backend_token == "cu130"
    assert stamp.requirements_sha256 == utilities_env._sha256_file(requirements)


def test_read_stamp_returns_none_when_absent(tmp_path: Path) -> None:
    """A missing stamp file reads back as None."""
    assert utilities_env.read_utilities_stamp(tmp_path) is None


def test_read_stamp_returns_none_on_malformed(tmp_path: Path) -> None:
    """A malformed or partial stamp reads back as None rather than raising."""
    stamp_path = paths.utilities_stamp_file(tmp_path)
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps({"backend_token": "cu126"}), encoding="utf-8")
    assert utilities_env.read_utilities_stamp(tmp_path) is None

    stamp_path.write_text("{not json", encoding="utf-8")
    assert utilities_env.read_utilities_stamp(tmp_path) is None
