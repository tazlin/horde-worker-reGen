"""Unit tests for the utilities-venv provisioning plan, stamp bookkeeping, and want/need resolution."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from worker_bootstrap import backend, paths, utilities_env

REPO_ROOT = Path(__file__).parents[2]


def _write_lock(root: Path, *, body: str = "utilities-lock-contents\n") -> Path:
    """Write a stand-in utilities uv.lock under *root* so provisioning has something to sync from."""
    lock = paths.utilities_lock_file(root)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(body, encoding="utf-8")
    return lock


def _make_interpreter(root: Path) -> None:
    """Create a stand-in utilities venv interpreter file so the interpreter-present branch is exercised."""
    interpreter = paths.utilities_python(root)
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("", encoding="utf-8")


# --- the committed utilities project ----------------------------------------------------------------


def test_repo_ships_utilities_project_and_lock() -> None:
    """The utilities uv project (pyproject + committed lock) is present so a fresh install can provision."""
    assert (paths.utilities_project_dir(REPO_ROOT) / "pyproject.toml").is_file()
    assert paths.utilities_lock_file(REPO_ROOT).is_file()


@pytest.mark.parametrize("token", ["cu126", "cu130", "cu132", "cpu"])
def test_utilities_project_declares_each_full_backend_extra(token: str) -> None:
    """Every full build the lane defaults on for is a build extra of the utilities project."""
    pyproject = (paths.utilities_project_dir(REPO_ROOT) / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(rf"^{token} = \[", pyproject, re.MULTILINE), f"utilities pyproject has no {token} extra"


@pytest.mark.parametrize("package", ["torch", "torchvision"])
def test_utilities_lock_torch_build_matches_root_lock(package: str) -> None:
    """The utilities lock pins the same torch/torchvision builds the root uv.lock resolves.

    The utilities venv only avoids re-downloading the multi-GB torch stack when it installs the *identical*
    build the main sync already fetched. Pinning this equality means a torch bump in the root lock that is
    not mirrored here (or vice versa) fails CI instead of silently reintroducing a divergent build that the
    shared cache cannot serve.
    """

    def builds(text: str) -> set[str]:
        # Capture the "<version>+<build>" strings uv records on dependency-array lines, e.g.
        #   { name = "torch", version = "2.12.1+cu132", source = ... }
        return set(re.findall(rf'name = "{re.escape(package)}", version = "([^"]+\+(?:cu\d+|cpu))"', text))

    util = builds(paths.utilities_lock_file(REPO_ROOT).read_text(encoding="utf-8"))
    root = builds((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    assert util, f"no pinned {package} build found in the utilities lock"
    assert util == root, (
        f"utilities lock {package} builds {sorted(util)} != root lock {sorted(root)}; re-pin "
        f"requirements/utilities/pyproject.toml and re-run `uv lock` there"
    )


# --- the provisioning plan --------------------------------------------------------------------------


def test_plan_syncs_from_lock(tmp_path: Path) -> None:
    """The plan is a single locked sync of the build extra, targeting the utilities project."""
    commands = utilities_env.plan_utilities_provision(uv="UV", backend_token="cu126", root=tmp_path)
    assert commands == [
        [
            "UV",
            "sync",
            "--locked",
            "--reinstall",
            "--project",
            str(paths.utilities_project_dir(tmp_path)),
            "--extra",
            "cu126",
            "--python",
            utilities_env.UTILITIES_PYTHON_VERSION,
        ],
    ]


def test_plan_is_locked_and_non_interactive(tmp_path: Path) -> None:
    """The sync passes --locked (install strictly from the lock; never re-resolve, never prompt)."""
    (command,) = utilities_env.plan_utilities_provision(uv="UV", backend_token="cu132", root=tmp_path)
    assert "--locked" in command
    assert command[1] == "sync"


def test_plan_forces_reinstall(tmp_path: Path) -> None:
    """The sync passes --reinstall so a re-provision reconciles actual wheels, not just name+version.

    Provisioning fires only on a stale venv, and a plain ``uv sync`` leaves an already-installed distribution
    in place when the lock repoints the same version at a different source (as the utilities lock does for
    onnxruntime-gpu's per-build CUDA index). Without --reinstall the old wheel would survive and a success
    stamp would wedge the venv until it was deleted by hand.
    """
    (command,) = utilities_env.plan_utilities_provision(uv="UV", backend_token="cu126", root=tmp_path)
    assert "--reinstall" in command


def test_plan_routes_the_requested_build_extra(tmp_path: Path) -> None:
    """The backend token becomes the synced build extra, so the right torch build is installed."""
    (command,) = utilities_env.plan_utilities_provision(uv="UV", backend_token="cpu", root=tmp_path)
    assert command[command.index("--extra") + 1] == "cpu"


# --- want resolution (ties to backend.desired_feature_extras) ---------------------------------------


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


def test_expects_utilities_lock_only_for_full_builds() -> None:
    """A missing lock is a broken install only for the builds that ship the utilities project."""
    assert all(backend.expects_utilities_lock(t) for t in ("cu126", "cu130", "cu132", "cpu"))
    assert not any(backend.expects_utilities_lock(t) for t in ("rocm", "rocm-windows", "xpu", "mps"))


# --- needs_provision (stamp vs the committed lock) --------------------------------------------------


def test_needs_provision_false_when_no_lock(tmp_path: Path) -> None:
    """With no committed utilities lock, there is nothing to sync from (stays a no-op)."""
    assert utilities_env.needs_provision(backend_token="cu126", root=tmp_path) is False


def test_needs_provision_true_when_venv_interpreter_missing(tmp_path: Path) -> None:
    """A committed lock but no venv interpreter means the venv must be provisioned."""
    _write_lock(tmp_path)
    assert utilities_env.needs_provision(backend_token="cu126", root=tmp_path) is True


def test_needs_provision_true_when_no_stamp(tmp_path: Path) -> None:
    """An existing interpreter but no stamp is treated as stale (provision once)."""
    _write_lock(tmp_path)
    _make_interpreter(tmp_path)
    assert utilities_env.needs_provision(backend_token="cu126", root=tmp_path) is True


def test_needs_provision_false_when_stamp_matches(tmp_path: Path) -> None:
    """A matching stamp (same token + same lock digest) proves the venv is current."""
    _write_lock(tmp_path)
    _make_interpreter(tmp_path)
    utilities_env.write_utilities_stamp(backend_token="cu126", root=tmp_path)
    assert utilities_env.needs_provision(backend_token="cu126", root=tmp_path) is False


def test_needs_provision_true_when_lock_changed(tmp_path: Path) -> None:
    """A changed utilities lock (different digest) invalidates the stamp and forces re-sync."""
    _write_lock(tmp_path)
    _make_interpreter(tmp_path)
    utilities_env.write_utilities_stamp(backend_token="cu126", root=tmp_path)
    _write_lock(tmp_path, body="a-different-lock\n")
    assert utilities_env.needs_provision(backend_token="cu126", root=tmp_path) is True


def test_needs_provision_true_when_backend_token_changed(tmp_path: Path) -> None:
    """A stamp recorded for a different backend token does not satisfy a provision for this one."""
    _write_lock(tmp_path)
    _make_interpreter(tmp_path)
    utilities_env.write_utilities_stamp(backend_token="cu126", root=tmp_path)
    assert utilities_env.needs_provision(backend_token="cu132", root=tmp_path) is True


# --- stamp round-trip -------------------------------------------------------------------------------


def test_stamp_roundtrip_and_content(tmp_path: Path) -> None:
    """The stamp persists the backend token and the utilities-lock SHA256, and reads back equal."""
    lock = _write_lock(tmp_path)
    utilities_env.write_utilities_stamp(backend_token="cu130", root=tmp_path)

    stamp = utilities_env.read_utilities_stamp(tmp_path)
    assert stamp is not None
    assert stamp.backend_token == "cu130"
    assert stamp.lock_sha256 == utilities_env._sha256_file(lock)


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
