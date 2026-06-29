"""Cross-platform GPU / torch-build detection.

Ported from ``packaging/detect-backend.ps1`` and the ``install.sh`` detection block so a single
standard-library implementation backs every install channel and platform. ``detect_backend`` returns a
build token that the locked uv extras (``cu126``/``cu130``/``cu132``/``cpu``) or an ad-hoc ROCm path
consume. For NVIDIA it selects the newest CUDA build the driver's reported max CUDA version can run
(13.2+ -> ``cu132``, 13.0/13.1 -> ``cu130``, anything older or unreadable -> the safe ``cu126``), then
applies an architecture floor: a GPU whose compute capability exceeds what the ``cu126`` wheel carries
kernels for (Hopper sm_90) gets at least ``cu130`` even on an older driver, because ``cu126`` has no
kernel image for it and would die at the first kernel launch. See :func:`_cuda_build`. AMD Windows
follows the current ComfyUI/AMD ROCm support matrices and routes recognized Radeon/Ryzen AI devices to
the official ROCm Windows PyTorch stack.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

CU126 = "cu126"
CU130 = "cu130"
CU132 = "cu132"
AMD_UNSUPPORTED = "amd-unsupported"
ROCM = "rocm"
ROCM_WINDOWS = "rocm-windows"
ROCM_GFX110X = "rocm-gfx110x"
ROCM_GFX1151 = "rocm-gfx1151"
ROCM_GFX120X = "rocm-gfx120x"
CPU = "cpu"

_CUDA_VERSION_RE = re.compile(r"CUDA Version:\s*(\d+)\.(\d+)")
_COMPUTE_CAP_RE = re.compile(r"(\d+)\.(\d+)")
# The locked wheels cover overlapping but different architecture windows (verified against PyTorch's
# CUDA build matrix and the NVIDIA CUDA 13 release notes):
#   cu126 (CUDA 12.6): sm_50..sm_90  (Maxwell through Hopper; no Blackwell)
#   cu130/cu132 (CUDA 13.x): sm_75..sm_120  (Turing through Blackwell; pre-Turing dropped in CUDA 13)
# A build has no kernel image for a card outside its window and dies at the first kernel launch
# (cudaErrorNoKernelImageForDevice), so the build must be clamped into the card's valid window in
# both directions, not just upward. See _cuda_build.
_CU126_MAX_COMPUTE_CAP = (9, 0)  # above this (Blackwell sm_100/sm_120), cu126 has no kernels -> floor cu130
_CUDA13_MIN_COMPUTE_CAP = (7, 5)  # below this (pre-Turing), cu130/cu132 have no kernels -> ceil cu126
# Windows "Display adapters" device class; its subkeys carry a DriverDesc per adapter.
_DISPLAY_CLASS_KEY = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def _is_windows() -> bool:
    """Return True when running on Windows."""
    return os.name == "nt"


def _nvidia_smi_path() -> str | None:
    """Locate nvidia-smi: PATH first, then the driver's default Windows location."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if _is_windows():
        candidate = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        if candidate.exists():
            return str(candidate)
    return None


def _windows_display_adapters() -> list[str]:
    """Return display-adapter names from the registry (avoids the deprecated/removed ``wmic``)."""
    if not _is_windows():
        return []
    import winreg  # Windows-only stdlib; imported lazily so this module stays importable on POSIX.

    names: list[str] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS_KEY) as root:
            index = 0
            while True:
                try:
                    sub = winreg.EnumKey(root, index)
                except OSError:
                    break
                index += 1
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, sub) as adapter:
                        desc, _ = winreg.QueryValueEx(adapter, "DriverDesc")
                except OSError:
                    continue
                if isinstance(desc, str):
                    names.append(desc)
    except OSError:
        return []
    return names


def _linux_lspci_match(needles: tuple[str, ...]) -> bool:
    """Best-effort display-controller match via ``lspci``; False when lspci is unavailable."""
    lspci = shutil.which("lspci")
    if not lspci:
        return False
    try:
        out = subprocess.run([lspci], capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    relevant = [ln for ln in out.splitlines() if ("VGA" in ln or "3D controller" in ln or "Display" in ln)]
    haystack = "\n".join(relevant).upper()
    return any(needle.upper() in haystack for needle in needles)


def _nvidia_present() -> bool:
    """Detect an NVIDIA GPU via nvidia-smi, Linux driver nodes, or the Windows adapter list."""
    if _nvidia_smi_path():
        return True
    if _is_windows():
        return any("NVIDIA" in name.upper() for name in _windows_display_adapters())
    if Path("/proc/driver/nvidia/version").exists() or Path("/dev/nvidia0").exists():
        return True
    return _linux_lspci_match(("NVIDIA",))


def _amd_present() -> bool:
    """Detect an AMD/Radeon GPU on Windows (registry) or Linux (kfd / lspci)."""
    if _is_windows():
        return any(("AMD" in name.upper() or "RADEON" in name.upper()) for name in _windows_display_adapters())
    if Path("/dev/kfd").exists():
        return True
    return _linux_lspci_match(("AMD", "Radeon", "Advanced Micro Devices"))


def _windows_amd_rocm_backend() -> str | None:
    """Return the AMD Windows ROCm token when the adapter is in the current support matrix.

    Keep the matcher intentionally conservative: unknown AMD cards stay ``amd-unsupported`` instead of
    getting a ROCm stack that may import but fail at runtime.
    """
    if not _is_windows():
        return None
    for raw_name in _windows_display_adapters():
        name = raw_name.upper()
        compact = re.sub(r"[\s\-_()]+", "", name)
        if "AMD" not in name and "RADEON" not in name:
            continue
        if re.search(r"\bRX\s*(9060|9070|7700|7900)\b", name) or re.search(r"\b(PRO\s+)?W7900\b", name):
            return ROCM_WINDOWS
        if re.search(r"\bAI\s+PRO\s+R9700\b", name):
            return ROCM_WINDOWS
        if "RYZENAIMAX" in compact or "STRIXHALO" in compact or "RADEON8050S" in compact or "RADEON8060S" in compact:
            return ROCM_WINDOWS
        if re.search(r"RYZENAI9(HX)?(365|370|375|465|470|475)", compact):
            return ROCM_WINDOWS
    return None


def _rocm_runtime_present() -> bool:
    """Return True when a usable ROCm runtime appears installed (Linux only)."""
    if _is_windows():
        return False
    return Path("/dev/kfd").exists() or shutil.which("rocminfo") is not None


def parse_cuda_version(smi_output: str) -> tuple[int, int]:
    """Return the (major, minor) CUDA version parsed from nvidia-smi output, or (0, 0) when unreadable."""
    match = _CUDA_VERSION_RE.search(smi_output or "")
    if not match:
        return (0, 0)
    return int(match.group(1)), int(match.group(2))


def _nvidia_cuda_version() -> tuple[int, int]:
    """Run nvidia-smi and parse the driver's max CUDA (major, minor); (0, 0) when unreadable.

    (0, 0) (nvidia-smi absent, or a freshly installed driver that has not added it to PATH) makes
    :func:`_cuda_build` fall back to the safe cu126 build.
    """
    exe = _nvidia_smi_path()
    if not exe:
        return (0, 0)
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return (0, 0)
    return parse_cuda_version(out)


def parse_compute_cap(smi_output: str) -> tuple[int, int]:
    """Return the highest (major, minor) compute capability in --query-gpu=compute_cap output.

    nvidia-smi prints one ``major.minor`` per GPU (e.g. ``12.0`` for Blackwell); the highest wins so a
    mixed-GPU box is floored onto a build that can run its newest card. (0, 0) when nothing parses.
    """
    caps = [(int(m.group(1)), int(m.group(2))) for m in _COMPUTE_CAP_RE.finditer(smi_output or "")]
    return max(caps) if caps else (0, 0)


def _nvidia_compute_cap() -> tuple[int, int]:
    """Query the GPU's compute capability via nvidia-smi; (0, 0) when unreadable.

    This is the GPU's architecture (sm_<major><minor>), distinct from the driver's CUDA ceiling: it is
    what decides whether a given wheel actually contains a kernel image for the card. (0, 0) (nvidia-smi
    absent, an old driver without the query field, or a parse miss) leaves :func:`_cuda_build` to decide
    purely from the driver's CUDA version, as before.
    """
    exe = _nvidia_smi_path()
    if not exe:
        return (0, 0)
    try:
        out = subprocess.run(
            [exe, "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return (0, 0)
    return parse_compute_cap(out)


def _cuda_build(version: tuple[int, int], compute_cap: tuple[int, int] = (0, 0)) -> str:
    """Pick the newest locked CUDA build the driver allows, clamped to the GPU's valid arch window.

    Two independent constraints decide the build, and the highest *valid* build is the newest one that
    satisfies both:

    * Driver ceiling: a torch wheel built against CUDA toolkit T needs a driver whose max CUDA version
      is >= T. The newest build the driver covers wins: 13.2+ -> cu132, 13.0/13.1 -> cu130, anything
      older or an unreadable (0, 0) version -> cu126. Tuple ordering keeps this version-aware
      (so e.g. (13, 1) < (13, 2)).
    * Architecture window: the driver version is only a ceiling, not proof the wheel has kernels for the
      card. cu126 carries sm_50..sm_90; the CUDA 13 wheels carry sm_75..sm_120 (pre-Turing was dropped
      in CUDA 13). A build outside the card's window has no kernel image and dies at the first kernel
      launch (cudaErrorNoKernelImageForDevice), so the driver-based pick is clamped both ways:
        - a Blackwell+ card (> ``_CU126_MAX_COMPUTE_CAP``) is floored onto cu130 even on a CUDA 12.x
          driver; cu126 can never run it, whereas cu130 runs once the (separately warned) driver is
          updated;
        - a pre-Turing card (< ``_CUDA13_MIN_COMPUTE_CAP``) is held at cu126 even on a CUDA 13 driver --
          the CUDA 13 wheels dropped it, and cu126 still runs on the newer driver.

    ``compute_cap`` (0, 0) (unreadable: nvidia-smi absent or an old driver without the field) skips the
    window clamp and keeps the driver-only pick, preserving the prior behaviour.
    """
    if version >= (13, 2):
        build = CU132
    elif version >= (13, 0):
        build = CU130
    else:
        build = CU126

    if compute_cap == (0, 0):
        return build
    if compute_cap > _CU126_MAX_COMPUTE_CAP:
        return CU130 if build == CU126 else build
    if compute_cap < _CUDA13_MIN_COMPUTE_CAP:
        return CU126
    return build


def detect_backend() -> str:
    """Return the torch build token for this machine.

    Returns:
        ``cu132``/``cu130``/``cu126`` (NVIDIA: the newest build the driver's max CUDA version supports,
        clamped to the GPU's valid architecture window, see :func:`_cuda_build`), ``rocm`` (AMD with a
        ROCm runtime on Linux), ``rocm-windows`` for
        supported AMD Windows Radeon/Ryzen AI devices, ``amd-unsupported`` for an AMD card with no known
        installable backend, or ``cpu``.
    """
    if _nvidia_present():
        return _cuda_build(_nvidia_cuda_version(), _nvidia_compute_cap())
    if _amd_present():
        if windows_backend := _windows_amd_rocm_backend():
            return windows_backend
        return ROCM if _rocm_runtime_present() else AMD_UNSUPPORTED
    return CPU
