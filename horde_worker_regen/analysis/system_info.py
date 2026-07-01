"""Collect the host context a maintainer needs to reason about a worker incident.

The logs say *what* happened; this says *where*. Worker version (with git state), OS/python, CPU/RAM,
free disk on the working volume and the model-cache volume, and optionally a GPU inventory. All of it is
torch-free in-process: the only torch-touching collector, the accelerator probe, runs out of process and
is opt-in, so generating a bundle never drags the inference stack into the orchestrator/CLI.

Reads ``bridgeData.yaml`` directly (best-effort) rather than importing the Textual-laden config module,
so this stays importable without the TUI.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

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


def _disk_free(path: Path) -> dict[str, int] | None:
    """Free/total bytes for the volume holding ``path``, or None if it cannot be read."""
    try:
        usage = shutil.disk_usage(path)
        return {"free_bytes": usage.free, "total_bytes": usage.total}
    except OSError:
        return None


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
