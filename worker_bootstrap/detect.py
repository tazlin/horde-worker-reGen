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
from dataclasses import dataclass
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


def gpu_arch_supported(arch_list: list[str], capability: tuple[int, int]) -> bool:
    """Whether a CUDA torch build compiled for ``arch_list`` has a usable kernel/PTX for ``capability``.

    ``arch_list`` is what ``torch.cuda.get_arch_list()`` reports for the *installed* wheel: ``sm_<n>``
    binary cubins and/or ``compute_<n>`` PTX. A device of compute capability (major, minor) can run:

    * a binary ``sm_<n>`` kernel of the *same major* whose minor is <= the device minor (cubins are
      forward-compatible only within a major), or
    * any ``compute_<n>`` PTX whose (major, minor) <= the device, JIT-compiled at load time.

    If neither exists, every kernel launch raises ``cudaErrorNoKernelImageForDevice``. This is the
    authoritative *post-install* compatibility test (the wheel describing itself), as opposed to the
    *pre-install* prediction :func:`_cuda_build` makes from a hardcoded table. The worker's inference
    process keeps its own copy of this logic because ``worker_bootstrap`` is not importable from the
    packaged worker; a guard test pins the two implementations together.
    """
    dev_major, dev_minor = capability
    for entry in arch_list:
        kind, _, ver = entry.partition("_")
        if not ver.isdigit() or len(ver) < 2:
            continue
        major, minor = int(ver[:-1]), int(ver[-1])
        if kind == "sm" and major == dev_major and minor <= dev_minor:
            return True
        if kind == "compute" and (major, minor) <= (dev_major, dev_minor):
            return True
    return False


def _clamp_build_to_arch(build: str, compute_cap: tuple[int, int]) -> str:
    """Clamp a CUDA build token into the GPU's valid architecture window.

    A build token names a CUDA toolkit line but is not itself proof the wheel carries a kernel image for
    the installed card, whether the token came from the driver-version pick, a persisted ``bin/backend``,
    or a forced override. cu126 carries sm_50..sm_90; the CUDA 13 wheels carry sm_75..sm_120 (pre-Turing
    was dropped in CUDA 13). A build outside the card's window has no kernel image and dies at the first
    kernel launch (cudaErrorNoKernelImageForDevice), so the token is clamped both ways:

    * a Blackwell+ card (> ``_CU126_MAX_COMPUTE_CAP``) is lifted off cu126 onto cu130; cu126 can never run
      it, whereas cu130 runs once the (separately warned) driver is updated;
    * a pre-Turing card (< ``_CUDA13_MIN_COMPUTE_CAP``) is held at cu126; the CUDA 13 wheels dropped it,
      and cu126 still runs on the newer driver.

    A non-CUDA token (cpu/rocm) and an unreadable ``compute_cap`` of (0, 0) are returned unchanged.
    """
    if compute_cap == (0, 0) or build not in (CU126, CU130, CU132):
        return build
    if compute_cap > _CU126_MAX_COMPUTE_CAP:
        return CU130 if build == CU126 else build
    if compute_cap < _CUDA13_MIN_COMPUTE_CAP:
        return CU126
    return build


def live_compute_capability() -> tuple[int, int]:
    """Return the live GPU's highest compute capability via nvidia-smi, or (0, 0) when unreadable.

    A public accessor for the post-install architecture check: it answers "what card is actually here?"
    without importing torch, so the bootstrap can compare it against the installed wheel's arch list.
    """
    return _nvidia_compute_cap()


# clamp_action values recorded on a BackendDecision, naming what the architecture clamp did to the
# driver-ceiling (or persisted) build. These are stable strings a support bundle / test can key on.
CLAMP_NONE = "in_window"  # the build already carries kernels for the card; unchanged
CLAMP_FLOORED = "floored_to_cu130"  # a Blackwell+ card lifted off cu126 (which has no kernel image for it)
CLAMP_CEILED = "ceiled_to_cu126"  # a pre-Turing card held on cu126 (the CUDA 13 wheels dropped it)
CLAMP_SKIPPED = "skipped_unreadable_cap"  # compute capability could not be read, so no clamp was applied
CLAMP_NOT_CUDA = "not_cuda"  # a cpu/rocm token, never clamped


@dataclass(frozen=True)
class BackendDecision:
    """The full audit trail behind a torch-build choice: the chosen token plus every input and step.

    A build token on its own ("cu126") cannot explain *why* it was picked, which is exactly what a support
    bundle needs when the installed wheel turns out to have no kernels for the card. This records the
    hardware signals read (driver CUDA ceiling, GPU compute capability), the intermediate driver-only pick,
    what the architecture clamp did to it, and a human-readable reason, so a maintainer can tell a stale
    persisted token apart from an unreadable ``nvidia-smi`` apart from an out-of-date selection table.

    ``stage`` is ``"detect"`` (a fresh hardware probe) or ``"reconcile"`` (re-clamping an already-resolved
    token against the live GPU). Fields that do not apply to a stage (a reconcile never reads the driver
    ceiling; a non-NVIDIA detect never reads a compute capability) stay ``None``.
    """

    stage: str
    final_token: str
    reason: str
    nvidia_present: bool | None = None
    amd_present: bool | None = None
    nvidia_smi_path: str | None = None
    driver_cuda_version: tuple[int, int] | None = None
    compute_capability: tuple[int, int] | None = None
    driver_ceiling_build: str | None = None
    input_token: str | None = None
    clamp_action: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view, rendering version tuples as ``"major.minor"`` strings."""

        def _cap(value: tuple[int, int] | None) -> str | None:
            return None if value is None else f"{value[0]}.{value[1]}"

        return {
            "stage": self.stage,
            "final_token": self.final_token,
            "reason": self.reason,
            "nvidia_present": self.nvidia_present,
            "amd_present": self.amd_present,
            "nvidia_smi_path": self.nvidia_smi_path,
            "driver_cuda_version": _cap(self.driver_cuda_version),
            "compute_capability": _cap(self.compute_capability),
            "driver_ceiling_build": self.driver_ceiling_build,
            "input_token": self.input_token,
            "clamp_action": self.clamp_action,
        }


def _explain_clamp(pre_clamp: str, post_clamp: str, compute_cap: tuple[int, int]) -> tuple[str, str]:
    """Return the ``(clamp_action, reason)`` describing what :func:`_clamp_build_to_arch` did and why."""
    if pre_clamp not in (CU126, CU130, CU132):
        return CLAMP_NOT_CUDA, f"{pre_clamp} is not a CUDA build; no architecture clamp applies"
    if compute_cap == (0, 0):
        return (
            CLAMP_SKIPPED,
            f"kept the driver-ceiling build {pre_clamp}; the GPU compute capability could not be read, so "
            "no architecture clamp was applied",
        )
    cap_tag = f"{compute_cap[0]}.{compute_cap[1]}"
    if post_clamp == pre_clamp:
        return CLAMP_NONE, f"{pre_clamp} carries kernels for compute capability {cap_tag}"
    if post_clamp == CU130 and pre_clamp == CU126:
        return (
            CLAMP_FLOORED,
            f"cu126 has no kernel image for compute capability {cap_tag} (Blackwell or newer); floored to "
            "cu130 (a driver update may be required before it can load)",
        )
    return (
        CLAMP_CEILED,
        f"the CUDA 13 wheels dropped compute capability {cap_tag} (pre-Turing); held at cu126",
    )


def _driver_ceiling_build(version: tuple[int, int]) -> str:
    """Return the newest locked CUDA build the driver's max CUDA version can load (before any arch clamp).

    A torch wheel built against CUDA toolkit T needs a driver whose max CUDA version is >= T. The newest
    build the driver covers wins: 13.2+ -> cu132, 13.0/13.1 -> cu130, anything older or an unreadable
    (0, 0) version -> cu126. Tuple ordering keeps this version-aware (so e.g. (13, 1) < (13, 2)).
    """
    if version >= (13, 2):
        return CU132
    if version >= (13, 0):
        return CU130
    return CU126


def _cuda_build(version: tuple[int, int], compute_cap: tuple[int, int] = (0, 0)) -> str:
    """Pick the newest locked CUDA build the driver allows, clamped to the GPU's valid arch window.

    Two independent constraints decide the build, and the highest *valid* build is the newest one that
    satisfies both:

    * Driver ceiling: see :func:`_driver_ceiling_build`.
    * Architecture window: the driver version is only a ceiling, not proof the wheel has kernels for the
      card, so the driver-based pick is clamped into the card's valid window (see
      :func:`_clamp_build_to_arch`). ``compute_cap`` (0, 0) skips the clamp and keeps the driver-only
      pick, preserving the prior behaviour.
    """
    return _clamp_build_to_arch(_driver_ceiling_build(version), compute_cap)


def describe_reconcile(token: str) -> BackendDecision:
    """Re-clamp an already-resolved token to the live GPU's arch window, recording the full decision.

    See :func:`reconcile_backend_for_gpu` for the behaviour; this variant returns a :class:`BackendDecision`
    so an install path can persist *why* the token was (or was not) changed, not just the resulting token.
    """
    if token not in (CU126, CU130, CU132):
        action, reason = _explain_clamp(token, token, (0, 0))
        return BackendDecision(
            stage="reconcile", final_token=token, input_token=token, clamp_action=action, reason=reason
        )
    compute_cap = _nvidia_compute_cap()
    final = _clamp_build_to_arch(token, compute_cap)
    action, reason = _explain_clamp(token, final, compute_cap)
    return BackendDecision(
        stage="reconcile",
        final_token=final,
        input_token=token,
        compute_capability=compute_cap,
        clamp_action=action,
        reason=reason,
    )


def reconcile_backend_for_gpu(token: str) -> str:
    """Re-clamp an already-resolved backend token to the live GPU's architecture window.

    A persisted ``bin/backend`` token (or the cu126 default) can name a torch build with no kernel image
    for the installed GPU: a cu126 token kept from before a Blackwell card was installed, say, which the
    sync path would otherwise reinstall on every update, leaving torch unable to launch a single kernel
    ("no CUDA kernels for this GPU"). Re-reading the live compute capability and re-applying the same
    arch-window clamp :func:`_cuda_build` uses lets such a token self-heal at install time.

    A non-CUDA token (cpu/rocm) is returned untouched and never triggers an nvidia-smi probe; an
    unreadable capability likewise returns the token unchanged.
    """
    return describe_reconcile(token).final_token


def describe_backend_selection() -> BackendDecision:
    """Return the torch build token for this machine together with the full decision trail.

    See :func:`detect_backend` for the resolution rules; this variant returns a :class:`BackendDecision`
    so the install path can persist the hardware signals and the reasoning behind the pick (the breadcrumb
    a support bundle reads to diagnose a wrong-build install).
    """
    if _nvidia_present():
        smi = _nvidia_smi_path()
        driver = _nvidia_cuda_version()
        compute_cap = _nvidia_compute_cap()
        ceiling = _driver_ceiling_build(driver)
        final = _clamp_build_to_arch(ceiling, compute_cap)
        action, clamp_reason = _explain_clamp(ceiling, final, compute_cap)
        driver_tag = f"{driver[0]}.{driver[1]}" if driver != (0, 0) else "unreadable"
        return BackendDecision(
            stage="detect",
            final_token=final,
            nvidia_present=True,
            nvidia_smi_path=smi,
            driver_cuda_version=driver,
            compute_capability=compute_cap,
            driver_ceiling_build=ceiling,
            clamp_action=action,
            reason=f"NVIDIA GPU; driver CUDA {driver_tag} -> {ceiling}; {clamp_reason}",
        )
    if _amd_present():
        if windows_backend := _windows_amd_rocm_backend():
            return BackendDecision(
                stage="detect",
                final_token=windows_backend,
                nvidia_present=False,
                amd_present=True,
                reason="AMD GPU matched a supported ROCm Windows profile",
            )
        if _rocm_runtime_present():
            return BackendDecision(
                stage="detect",
                final_token=ROCM,
                nvidia_present=False,
                amd_present=True,
                reason="AMD GPU with a ROCm runtime present",
            )
        return BackendDecision(
            stage="detect",
            final_token=AMD_UNSUPPORTED,
            nvidia_present=False,
            amd_present=True,
            reason="AMD GPU with no known installable backend",
        )
    return BackendDecision(
        stage="detect",
        final_token=CPU,
        nvidia_present=False,
        amd_present=False,
        reason="no NVIDIA or AMD GPU detected",
    )


def detect_backend() -> str:
    """Return the torch build token for this machine.

    Returns:
        ``cu132``/``cu130``/``cu126`` (NVIDIA: the newest build the driver's max CUDA version supports,
        clamped to the GPU's valid architecture window, see :func:`_cuda_build`), ``rocm`` (AMD with a
        ROCm runtime on Linux), ``rocm-windows`` for
        supported AMD Windows Radeon/Ryzen AI devices, ``amd-unsupported`` for an AMD card with no known
        installable backend, or ``cpu``.
    """
    return describe_backend_selection().final_token
