"""Ad-hoc ROCm torch install, ported from update-runtime-rocm.sh plus ComfyUI's AMD Windows guidance.

ROCm is intentionally not part of ``uv.lock`` (torch 2.12.0 has no locked ROCm wheel matching the worker's
supported targets), so it is installed out-of-band: a base sync of the cpu extra, then a torch stack from
the right ROCm wheel source. Linux defaults are stable PyTorch ROCm wheels. Windows AMD follows the
official AMD ROCm Windows PyTorch wheels used by the current ComfyUI guidance. Overrides
``HORDE_WORKER_ROCM_TORCH`` / ``HORDE_WORKER_ROCM_INDEX`` mirror the previous shell script for Linux;
Windows can override ``HORDE_WORKER_ROCM_WINDOWS_VERSION`` / ``HORDE_WORKER_ROCM_WINDOWS_BASE``.
Imported lazily by the CLI so the common path never pays for it.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from worker_bootstrap import backend, detect, paths, runner, sync_plan

_DEFAULT_TORCH = "2.9.1"
_DEFAULT_INDEX = "https://download.pytorch.org/whl/rocm6.4"
_DEFAULT_WINDOWS_ROCM = "7.2.1"
_DEFAULT_WINDOWS_BASE = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1"
_WINDOWS_ROCM_TOKENS = {
    detect.ROCM_WINDOWS,
    detect.ROCM_GFX110X,
    detect.ROCM_GFX1151,
    detect.ROCM_GFX120X,
}
_LIBHSA = "libhsa-runtime64.so"
_SYSTEM_LIBHSA = Path("/opt/rocm/lib") / _LIBHSA


@dataclass(frozen=True)
class _RocmInstallPlan:
    """The ad-hoc torch install command shape for a ROCm backend token."""

    token: str
    index: str
    index_option: str
    torch_specs: tuple[str, ...]
    packages: tuple[str, ...]
    pre: bool
    reinstall: bool


def is_rocm_token(token: str) -> bool:
    """Return whether *token* is one of the ad-hoc ROCm install profiles."""
    return token == detect.ROCM or token in _WINDOWS_ROCM_TOKENS


def _windows_rocm_url(package: str, version: str, wheel_tag: str, *, base: str, local: str = "") -> str:
    """Return an AMD ROCm Windows wheel URL for *package*."""
    return f"{base}{package}-{version}{local}-{wheel_tag}-win_amd64.whl"


def _install_plan(token: str) -> _RocmInstallPlan:
    """Return the torch package/index plan for a ROCm backend token."""
    override_index = os.environ.get("HORDE_WORKER_ROCM_INDEX")
    if token in _WINDOWS_ROCM_TOKENS:
        rocm_version = os.environ.get("HORDE_WORKER_ROCM_WINDOWS_VERSION", _DEFAULT_WINDOWS_ROCM)
        base = os.environ.get("HORDE_WORKER_ROCM_WINDOWS_BASE", _DEFAULT_WINDOWS_BASE)
        if not base.endswith("/"):
            base = f"{base}/"
        rocm_local = f"%2Brocm{rocm_version}"
        return _RocmInstallPlan(
            token=token,
            index=base,
            index_option="",
            torch_specs=(
                _windows_rocm_url("amd_smi", rocm_version, "py3-none", base=base),
                _windows_rocm_url("hip_sdk", rocm_version, "py3-none", base=base),
                _windows_rocm_url("torch", "2.9.1", "cp312-cp312", base=base, local=rocm_local),
                _windows_rocm_url("torchvision", "0.24.1", "cp312-cp312", base=base, local=rocm_local),
                _windows_rocm_url("torchaudio", "2.9.1", "cp312-cp312", base=base, local=rocm_local),
            ),
            packages=(),
            pre=False,
            reinstall=True,
        )
    torch_version = os.environ.get("HORDE_WORKER_ROCM_TORCH", _DEFAULT_TORCH)
    return _RocmInstallPlan(
        token=token,
        index=override_index or _DEFAULT_INDEX,
        index_option="--extra-index-url",
        torch_specs=(f"torch=={torch_version}",),
        packages=("torchvision", "torchaudio", "pytorch-triton-rocm"),
        pre=False,
        reinstall=True,
    )


def _is_wsl() -> bool:
    """Return True when running under WSL2 (its kernel string contains WSL2)."""
    try:
        out = subprocess.run(["uname", "-a"], capture_output=True, text=True, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "WSL2" in out


def _patch_wsl_libhsa(root: Path) -> None:
    """Under WSL, overwrite any bundled libhsa-runtime64.so with the host ROCm one (mirrors the script)."""
    if not _is_wsl() or not _SYSTEM_LIBHSA.exists():
        return
    print("WSL environment detected. Patching ROCm libhsa-runtime64.so")
    for found in root.rglob(_LIBHSA):
        with contextlib.suppress(OSError):
            shutil.copyfile(_SYSTEM_LIBHSA, found)


def _rocm_torch_already_satisfies(root: Path, target: str) -> bool:
    """Return whether the installed ROCm torch already meets *target* (so a reinstall is optional)."""
    installed = sync_plan.installed_versions(paths.venv_dir(root)).get("torch")
    if installed is None:
        return False
    return sync_plan.version_at_least(installed, target)


def sync_rocm(uv: str, *, root: Path | None = None, hold: bool = False, token: str = detect.ROCM) -> int:
    """Install the base environment then the ad-hoc ROCm torch stack; return the final exit code.

    With ``hold=True`` (the user opted to limp along), the ad-hoc torch reinstall is skipped when the
    installed torch already satisfies the target version: the ROCm stack is not in the lock, so "holding"
    simply means not re-pulling the ~GB torch wheel when nothing requires it.
    """
    if not is_rocm_token(token):
        raise ValueError(f"not a ROCm backend token: {token}")
    root = root or paths.install_root()
    # ROCm is a lean "others" backend: the cpu extra provides only the universal deps. Feature extras
    # (rembg/onnxruntime) are off by default and opt-in via HORDE_WORKER_FEATURES (their CPU wheels do
    # exist on x86), resolved against the ROCm token so the cpu base does not auto-pull them.
    feature_extras = backend.desired_feature_extras(token, env_value=os.environ.get("HORDE_WORKER_FEATURES"))
    features_note = ", ".join(feature_extras) if feature_extras else "none (lean base)"
    print(f"Installing the base environment (everything except the GPU torch build; features: {features_note})...")
    base_rc = runner.uv_sync(uv, "cpu", extras=feature_extras, root=root)
    if base_rc != 0:
        return base_rc

    plan = _install_plan(token)
    first_torch_spec = next((spec for spec in plan.torch_specs if "torch-" in spec or spec.startswith("torch")), "")
    held_target = first_torch_spec.removeprefix("torch==")
    if hold and first_torch_spec.startswith("torch==") and _rocm_torch_already_satisfies(root, held_target):
        print(f"Limping along: keeping the installed ROCm torch (target {held_target} already satisfied).")
        return 0
    print(f"Installing the ROCm PyTorch stack ad-hoc ({token} from {plan.index})...")
    args = ["pip", "install"]
    if plan.pre:
        args.append("--pre")
    if plan.reinstall:
        args.append("--reinstall")
    args += [*plan.torch_specs, *plan.packages]
    if plan.index_option:
        args += [plan.index_option, plan.index]
    torch_rc = runner.run_uv(uv, args, root=root)
    if torch_rc != 0:
        print(
            f"ERROR: ad-hoc ROCm PyTorch install failed. Confirm {plan.index} publishes the requested "
            "torch stack, or set HORDE_WORKER_ROCM_TORCH / HORDE_WORKER_ROCM_INDEX.",
        )
        return torch_rc

    # Ensure no NVIDIA-only helpers leaked into the ROCm environment (best effort).
    runner.run_uv(uv, ["pip", "uninstall", "pynvml", "nvidia-ml-py"], root=root)
    _patch_wsl_libhsa(root)

    amd_go_fast = root / "horde_worker_regen" / "amd_go_fast" / "install_amd_go_fast.sh"
    if amd_go_fast.exists():
        return runner.uv_run(uv, [str(amd_go_fast)], root=root)
    return 0
