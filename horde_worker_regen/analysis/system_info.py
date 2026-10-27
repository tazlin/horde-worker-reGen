"""Collect system information for diagnosing worker problems.

Include the worker version and Git state, OS/Python versions, CPU/RAM, free disk space on the working
and model-cache volumes, and an optional GPU inventory. The accelerator probe runs in a separate process
and requires explicit opt-in, so generating a bundle does not load torch into the orchestrator or CLI.

``rotational_storage`` reports whether a path is on spinning or solid-state media, for advice that depends on
how expensive a model reload from disk is.

Read ``bridgeData.yaml`` directly when available to avoid importing the config module and its Textual
dependency. This module can be used without the TUI.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
from loguru import logger

from horde_worker_regen.runtime_version import runtime_version

# Defined locally (not imported from tui.config_form) to keep this module free of the Textual import.
_DEFAULT_CONFIG_PATH = Path("bridgeData.yaml")


def _load_config_yaml(config_path: Path) -> dict[str, Any]:
    """Best-effort parse of the worker config to a dict; empty on any error (the redactor backstops it)."""
    if not config_path.is_file():
        return {}
    try:
        from ruamel.yaml import YAML

        data = YAML(typ="safe").load(config_path.read_text(encoding="utf-8", errors="replace"))
        return dict(data) if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - a malformed/unreadable config must not break info collection
        return {}


def resolve_cache_home(config_path: Path = _DEFAULT_CONFIG_PATH) -> str | None:
    """Resolve the model-cache directory the same way the worker does: env var, then config, else None.

    Mirrors ``load_env_vars`` precedence: an explicit ``AIWORKER_CACHE_HOME`` wins, otherwise the
    config's ``cache_home`` key. Returns None when neither is set (the installer default is unknown here).
    """
    env = os.environ.get("AIWORKER_CACHE_HOME")
    if env:
        return env
    value = _load_config_yaml(config_path).get("cache_home")
    return str(value) if value else None


def config_secret_values(config_path: Path = _DEFAULT_CONFIG_PATH) -> list[str | None]:
    """The secret values to scrub, read from the config: the horde api_key and the CivitAI token."""
    data = _load_config_yaml(config_path)
    return [data.get("api_key"), data.get("civitai_api_token"), data.get("CIVIT_API_TOKEN")]


def config_worker_name(config_path: Path = _DEFAULT_CONFIG_PATH) -> str | None:
    """The worker's registered name from the config (``dreamer_name``), for identifier redaction."""
    data = _load_config_yaml(config_path)
    name = data.get("dreamer_name") or data.get("worker_name")
    return str(name) if name else None


def config_stats_export_enabled(config_path: Path = _DEFAULT_CONFIG_PATH) -> bool:
    """Whether the config turns on the per-session stats JSONL export (off when the key is absent).

    The bundle records it so a maintainer reading an archive with no ``stats/`` member knows the stream was
    never written rather than lost, since the stats rows are the per-second scheduling evidence.
    """
    return bool(_load_config_yaml(config_path).get("stats_export_enabled", False))


def _disk_free(path: Path) -> dict[str, int] | None:
    """Free/total bytes for the volume holding ``path``, or None if it cannot be read."""
    try:
        usage = shutil.disk_usage(path)
        return {"free_bytes": usage.free, "total_bytes": usage.total}
    except OSError:
        return None


_SYSFS_ROOT = Path("/sys")
# Get-PhysicalDisk and its Storage module take several seconds to load on a cold PowerShell.
_WINDOWS_STORAGE_PROBE_TIMEOUT_SECONDS = 10.0
_WINDOWS_MEDIA_TYPE_ROTATIONAL: dict[str, bool] = {"HDD": True, "SSD": False, "SCM": False}


def rotational_storage(path: Path) -> bool | None:
    """Return whether ``path`` is on rotational media: True for a spinning disk, False for solid state.

    Linux reads the ``queue/rotational`` flag of the block device behind the path, walking a partition to its
    disk. Windows asks ``Get-PhysicalDisk`` for the media type of the disk behind the path's drive letter, in a
    bounded PowerShell subprocess. Every other platform, and any failure, returns None.

    Args:
        path: A file or directory on the volume to probe.

    Returns:
        True for rotational media, False for solid state, None when the type could not be determined.

    Performance:
        The Windows probe spawns PowerShell and takes a few seconds, so call it once per path and off any
        event loop.
    """
    if sys.platform == "linux":
        try:
            device_number = os.stat(path).st_dev
        except OSError as error:
            logger.debug(f"Storage type of {path} not determined: {error}")
            return None
        return _linux_rotational(major=os.major(device_number), minor=os.minor(device_number))
    if sys.platform == "win32":
        return _windows_rotational(path)
    return None


def _block_device_entry(*, major: int, minor: int) -> Path:
    """Return the sysfs directory of block device ``major:minor``."""
    return _SYSFS_ROOT / "dev" / "block" / f"{major}:{minor}"


def _linux_rotational(*, major: int, minor: int) -> bool | None:
    """Return the rotational flag of the disk holding block device ``major:minor``, or None when unreadable."""
    try:
        uevent = (_block_device_entry(major=major, minor=minor) / "uevent").read_text(encoding="utf-8")
    except OSError as error:
        # Filesystems with anonymous devices (btrfs subvolumes, overlay, tmpfs) have no block device entry.
        logger.debug(f"Storage type of block device {major}:{minor} not determined: {error}")
        return None
    fields = dict(line.split("=", 1) for line in uevent.splitlines() if "=" in line)
    device_name = fields.get("DEVNAME")
    if not device_name:
        return None
    disk_name = device_name
    if fields.get("DEVTYPE") == "partition":
        parents = [disk.name for disk in (_SYSFS_ROOT / "block").iterdir() if (disk / device_name).is_dir()]
        if not parents:
            return None
        disk_name = parents[0]
    try:
        flag = (_SYSFS_ROOT / "block" / disk_name / "queue" / "rotational").read_text(encoding="utf-8").strip()
    except OSError as error:
        logger.debug(f"Storage type of {disk_name} not determined: {error}")
        return None
    if flag not in ("0", "1"):
        return None
    return flag == "1"


def _windows_rotational(path: Path) -> bool | None:
    """Return the media type of the physical disk behind the path's drive letter, or None when unknown."""
    try:
        drive = path.resolve().drive
    except (OSError, RuntimeError) as error:
        logger.debug(f"Storage type of {path} not determined: {error}")
        return None
    if len(drive) != 2 or not drive.endswith(":") or not drive[0].isalpha():
        # A UNC share or a volume mounted without a letter cannot be mapped through Get-Partition.
        return None
    script = (
        "$ErrorActionPreference = 'Stop'; "
        f"$partition = Get-Partition -DriveLetter {drive[0]}; "
        "(Get-PhysicalDisk | Where-Object { $_.DeviceId -eq [string]$partition.DiskNumber }).MediaType"
    )
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=_WINDOWS_STORAGE_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.debug(f"Storage type of {path} not determined: {error}")
        return None
    if completed.returncode != 0:
        logger.debug(f"Storage type of {path} not determined: {completed.stderr.strip()[:300]}")
        return None
    # A Storage Spaces or RAID volume may span multiple physical disks, and some drives report "Unspecified".
    return _WINDOWS_MEDIA_TYPE_ROTATIONAL.get(completed.stdout.strip())


_SMI_CUDA_RE = re.compile(r"CUDA Version:\s*(\d+\.\d+)")


def nvidia_smi_summary() -> dict[str, Any] | None:
    """Return the NVIDIA driver's version, its CUDA ceiling, and per-GPU name + compute capability.

    Torch-free and cheap: it shells ``nvidia-smi`` (never imports torch), so a bundle captures the two
    numbers a wrong-CUDA-build incident turns on -- the driver's max CUDA version (the build *ceiling*) and
    each card's compute capability (which decides whether a wheel has kernels for it) -- even on an install
    whose backend-decision breadcrumb predates that feature. Returns None when nvidia-smi is absent or
    unreadable (no NVIDIA GPU, or the driver has not added it to PATH), so a non-NVIDIA host simply omits
    the block.
    """
    exe = shutil.which("nvidia-smi")
    if not exe and os.name == "nt":
        candidate = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        exe = str(candidate) if candidate.exists() else None
    if not exe:
        return None

    summary: dict[str, Any] = {}
    try:
        header = subprocess.run([exe], capture_output=True, text=True, timeout=20, check=False).stdout
        match = _SMI_CUDA_RE.search(header or "")
        summary["driver_max_cuda_version"] = match.group(1) if match else None
        gpus = subprocess.run(
            [exe, "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return summary or None

    cards: list[dict[str, str]] = []
    for line in (gpus or "").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[0]:
            card = {"name": fields[0], "compute_cap": fields[1]}
            if len(fields) >= 3:
                card["driver_version"] = fields[2]
            cards.append(card)
    if cards:
        summary["gpus"] = cards
    return summary or None


def collect_system_info(*, cache_home: str | None = None, probe_gpu: bool = False) -> dict[str, Any]:
    """Gather worker version, OS/python, CPU/RAM, disk, and (optionally) a GPU inventory.

    Args:
        cache_home: The resolved model-cache directory (for its disk-free figure), or None.
        probe_gpu: When True, run the out-of-process accelerator probe to inventory GPUs. Off by default
            because it spawns a torch subprocess; the worker's startup log already records the GPUs.
    """
    memory = psutil.virtual_memory()
    info: dict[str, Any] = {
        "worker_version": runtime_version(),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram": {"total_bytes": memory.total, "available_bytes": memory.available, "percent_used": memory.percent},
        "disk": {
            "working_dir": _disk_free(Path.cwd()),
            "cache_home": _disk_free(Path(cache_home)) if cache_home else None,
        },
        "cache_home": cache_home,
    }
    driver = nvidia_smi_summary()
    if driver is not None:
        info["nvidia_smi"] = driver
    if probe_gpu:
        info["accelerators"] = _probe_gpus()
    return info


def _probe_gpus() -> list[dict[str, Any]]:
    """Inventory GPUs via the out-of-process probe; empty list on any failure (best-effort)."""
    try:
        from horde_worker_regen.utils.accelerator_probe import probe_accelerators

        return [accelerator.model_dump() for accelerator in probe_accelerators(timeout_seconds=30.0)]
    except Exception:  # noqa: BLE001 - the GPU probe is best-effort context, never required
        return []
