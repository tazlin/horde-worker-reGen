"""Out-of-process accelerator inventory, so torch-free callers can read device info without loading torch.

Reading the accelerator inventory (each device's index, name, and total VRAM) goes through hordelib's
backend-agnostic :func:`hordelib.utils.torch_memory.enumerate_accelerators`. That call has to interrogate
the active GPU backend, which loads torch (~500MB RSS) into whatever process makes it. The worker's
long-lived orchestrator process and the interactive config wizard are deliberately torch-free (see
``AGENTS.md`` -- only the inference/safety children, which need torch for their core function, should pay
that cost), so they must not enumerate accelerators in-process.

This module runs the enumeration in a short-lived subprocess and returns the result as plain validated
data. The subprocess pays the torch cost and frees it on exit, leaving the caller torch-free. It stays
backend-agnostic: the subprocess uses the same hordelib helper, so it reports whatever backend ComfyUI
supports (CUDA/ROCm, Intel XPU, Apple MPS, DirectML, CPU), not just NVIDIA.
"""

from __future__ import annotations

import json
import subprocess
import sys

from loguru import logger
from pydantic import BaseModel

# The child only imports the torch-free ``torch_memory`` submodule, then calls the enumeration (which
# loads torch in *this* short-lived process). The result is emitted on stdout behind a sentinel prefix so
# parsing is robust against any stray stdout (logging/telemetry banners) the import might produce.
_RESULT_PREFIX = "ACCEL_PROBE_JSON:"

_PROBE_SOURCE = f"""
import json
import sys

try:
    from hordelib.utils.torch_memory import (
        enumerate_accelerators,
        get_torch_free_vram_mb,
        get_torch_total_vram_mb,
    )

    _accelerators = enumerate_accelerators()
    # Right after enumeration (which initialises the backend) the only device allocation is this fresh
    # process's runtime/CUDA context, with no model loaded, so total-free approximates the per-process
    # VRAM overhead on an idle device -- the term the streaming forecast subtracts from total VRAM to get
    # the free achievable under sole residency.
    try:
        _overhead_mb = max(0, int(get_torch_total_vram_mb()) - int(get_torch_free_vram_mb()))
    except BaseException:
        _overhead_mb = 0
    _payload = [
        {{
            "index": int(a.index),
            "name": str(a.name),
            "total_vram_mb": int(a.total_vram_mb),
            "runtime_overhead_mb": _overhead_mb,
        }}
        for a in _accelerators
    ]
except BaseException as exc:  # noqa: BLE001 - any failure means "no devices"; report and exit non-zero
    print("ACCEL_PROBE_ERR:" + repr(exc), file=sys.stderr)
    sys.exit(3)

print({_RESULT_PREFIX!r} + json.dumps(_payload))
"""


class ProbedAccelerator(BaseModel):
    """One accelerator's identity and capacity, as returned by the out-of-process probe."""

    index: int
    name: str
    total_vram_mb: int
    runtime_overhead_mb: int = 0
    """Approx. per-process VRAM (MB) a fresh torch process consumes for its context, measured on the idle
    device at probe time. Defaults to 0 for probes/serialisations that predate this field."""


def probe_accelerators(*, timeout_seconds: float = 120.0) -> list[ProbedAccelerator]:
    """Return the machine's accelerators by enumerating them in a short-lived subprocess.

    Keeps the calling (orchestrator/wizard) process torch-free: the subprocess loads torch, answers, and
    exits. Never raises -- any failure (no backend, subprocess error or timeout, malformed output) is
    logged at debug and yields an empty list, so the caller degrades to "no devices detected" rather than
    crashing. The subprocess reuses this interpreter (``sys.executable``), so it sees the same hordelib.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE_SOURCE],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as run_error:
        logger.debug(f"Accelerator probe subprocess could not run: {type(run_error).__name__} {run_error}")
        return []

    if completed.returncode != 0:
        logger.debug(f"Accelerator probe exited {completed.returncode}: {completed.stderr.strip()}")
        return []

    for line in completed.stdout.splitlines():
        if not line.startswith(_RESULT_PREFIX):
            continue
        try:
            raw_entries = json.loads(line[len(_RESULT_PREFIX) :])
            return [ProbedAccelerator.model_validate(entry) for entry in raw_entries]
        except (ValueError, TypeError) as parse_error:
            logger.debug(f"Could not parse accelerator probe output: {parse_error}")
            return []

    logger.debug("Accelerator probe produced no result line")
    return []
