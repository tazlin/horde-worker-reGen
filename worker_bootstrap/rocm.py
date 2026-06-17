"""Ad-hoc ROCm torch install (Linux), ported from update-runtime-rocm.sh.

ROCm is intentionally not part of ``uv.lock`` (torch 2.12.0 has no ROCm wheel, and PyTorch has not
published the pytorch-triton-rocm its rocm7.x wheels depend on), so it is installed out-of-band: a base
sync of the cpu extra, then a pinned torch stack from the ROCm wheel index. Overrides
``HORDE_WORKER_ROCM_TORCH`` / ``HORDE_WORKER_ROCM_INDEX`` mirror the previous shell script. Imported lazily
by the CLI so the common path never pays for it.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

from worker_bootstrap import backend, paths, runner, sync_plan

_DEFAULT_TORCH = "2.9.1"
_DEFAULT_INDEX = "https://download.pytorch.org/whl/rocm6.4"
_LIBHSA = "libhsa-runtime64.so"
_SYSTEM_LIBHSA = Path("/opt/rocm/lib") / _LIBHSA


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


def sync_rocm(uv: str, *, root: Path | None = None, hold: bool = False) -> int:
    """Install the base environment then the ad-hoc ROCm torch stack; return the final exit code.

    With ``hold=True`` (the user opted to limp along), the ad-hoc torch reinstall is skipped when the
    installed torch already satisfies the target version: the ROCm stack is not in the lock, so "holding"
    simply means not re-pulling the ~GB torch wheel when nothing requires it.
    """
    root = root or paths.install_root()
    # ROCm is a lean "others" backend: the cpu extra provides only the universal deps. Feature extras
    # (rembg/onnxruntime) are off by default and opt-in via HORDE_WORKER_FEATURES (their CPU wheels do
    # exist on x86 Linux), resolved against the "rocm" token so the cpu base does not auto-pull them.
    feature_extras = backend.desired_feature_extras("rocm", env_value=os.environ.get("HORDE_WORKER_FEATURES"))
    features_note = ", ".join(feature_extras) if feature_extras else "none (lean base)"
    print(f"Installing the base environment (everything except the GPU torch build; features: {features_note})...")
    base_rc = runner.uv_sync(uv, "cpu", extras=feature_extras, root=root)
    if base_rc != 0:
        return base_rc

    torch_version = os.environ.get("HORDE_WORKER_ROCM_TORCH", _DEFAULT_TORCH)
    index = os.environ.get("HORDE_WORKER_ROCM_INDEX", _DEFAULT_INDEX)
    if hold and _rocm_torch_already_satisfies(root, torch_version):
        print(f"Limping along: keeping the installed ROCm torch (target {torch_version} already satisfied).")
        return 0
    print(f"Installing the ROCm PyTorch stack ad-hoc (torch {torch_version} from {index})...")
    torch_rc = runner.run_uv(
        uv,
        [
            "pip",
            "install",
            "--reinstall",
            f"torch=={torch_version}",
            "torchvision",
            "torchaudio",
            "pytorch-triton-rocm",
            "--extra-index-url",
            index,
        ],
        root=root,
    )
    if torch_rc != 0:
        print(
            f"ERROR: ad-hoc ROCm PyTorch install failed. Confirm {index} publishes torch=={torch_version} "
            "(and its pytorch-triton-rocm), or set HORDE_WORKER_ROCM_TORCH / HORDE_WORKER_ROCM_INDEX.",
        )
        return torch_rc

    # Ensure no NVIDIA-only helpers leaked into the ROCm environment (best effort).
    runner.run_uv(uv, ["pip", "uninstall", "pynvml", "nvidia-ml-py"], root=root)
    _patch_wsl_libhsa(root)

    amd_go_fast = root / "horde_worker_regen" / "amd_go_fast" / "install_amd_go_fast.sh"
    if amd_go_fast.exists():
        return runner.uv_run(uv, [str(amd_go_fast)], root=root)
    return 0
