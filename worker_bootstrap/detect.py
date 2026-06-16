"""Cross-platform GPU / torch-build detection.

Ported from ``packaging/detect-backend.ps1`` and the ``install.sh`` detection block so a single
standard-library implementation backs every install channel and platform. ``detect_backend`` returns a
build token that the locked uv extras (``cu126``/``cu130``/``cu132``/``cpu``) or the ad-hoc ROCm path
consume. For NVIDIA it selects the newest CUDA build the driver's reported max CUDA version can run
(13.2+ -> ``cu132``, 13.0/13.1 -> ``cu130``, anything older or unreadable -> the safe ``cu126``); see
:func:`_cuda_build`.
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
CPU = "cpu"

_CUDA_VERSION_RE = re.compile(r"CUDA Version:\s*(\d+)\.(\d+)")
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


def _cuda_build(version: tuple[int, int]) -> str:
    """Pick the newest locked CUDA build the driver's max CUDA version can run.

    A torch wheel built against CUDA toolkit T needs a driver whose max CUDA version is >= T, so the
    newest build the driver covers wins: 13.2+ -> cu132, 13.0/13.1 -> cu130, anything older or an
    unreadable (0, 0) version -> the safe cu126 (the only CUDA-12 build of torch 2.12.0, which also runs
    on CUDA 13 drivers). Tuple ordering makes the comparison version-aware (so e.g. (13, 1) < (13, 2)).
    """
    if version >= (13, 2):
        return CU132
    if version >= (13, 0):
        return CU130
    return CU126


def detect_backend() -> str:
    """Return the torch build token for this machine.

    Returns:
        ``cu132``/``cu130``/``cu126`` (NVIDIA: the newest build the driver's max CUDA version supports,
        see :func:`_cuda_build`), ``rocm`` (AMD with a ROCm runtime on Linux), ``amd-unsupported`` (AMD
        without a usable backend, e.g. Windows), or ``cpu``.
    """
    if _nvidia_present():
        return _cuda_build(_nvidia_cuda_version())
    if _amd_present():
        return ROCM if _rocm_runtime_present() else AMD_UNSUPPORTED
    return CPU
