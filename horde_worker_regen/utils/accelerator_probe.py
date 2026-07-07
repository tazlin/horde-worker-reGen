"""Out-of-process accelerator inventory, so torch-free callers can read device info without loading torch.

Reading the accelerator inventory (each device's index, name, and total VRAM) goes through hordelib's
backend-agnostic :func:`hordelib.utils.torch_memory.enumerate_accelerators`. That call has to interrogate
the active GPU backend, which loads torch (~500MB RSS) into whatever process makes it. The worker's
long-lived orchestrator process and the interactive config wizard are deliberately torch-free (see
``AGENTS.md``; only the inference/safety children, which need torch for their core function, should pay
that cost), so they must not enumerate accelerators in-process.

This module runs the enumeration in a short-lived subprocess and returns the result as plain validated
data. The subprocess pays the torch cost and frees it on exit, leaving the caller torch-free. It stays
backend-agnostic: the subprocess uses the same hordelib helper, so it reports whatever backend ComfyUI
supports (CUDA/ROCm, Intel XPU, Apple MPS, DirectML, CPU), not just NVIDIA.

Beyond the inventory, the subprocess also measures two VRAM figures the streaming forecast needs, which
correspond to two distinct terms of the device's VRAM decomposition (device baseline / per-process
marginal overhead / model weights / activation peaks; see ``scheduling/context_overhead_model``):

- the *first/sole* process's context cost: the one-time, device-wide CUDA runtime allocation plus one
  context (and any fixed device baseline the reading happens to include). This is paid once per device and
  sizes ``free_if_alone``; it is never the cost of an additional context; and
- the *marginal* cost of each additional sibling context: measured directly by bringing up a second
  context-holding process and reading the device-wide used *delta*. Because the one-time runtime and the
  device baseline are already counted in the first figure, the delta isolates term (2) alone, so the
  forecast can size ``free_after_model_evict`` from the real per-context cost instead of charging the whole
  one-time-inclusive overhead per process (which over-counts badly on a big card and is what wedged a 24GB
  worker).

The delta is only visible cross-process where the platform reports true device-wide VRAM; Linux does,
Windows WDDM does not, so there the marginal reads 0 and the worker seeds a conservative
per-additional-context constant (``resource_budget._SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB``) rather than
re-charging the first-context overhead per context.
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

# A minimal second process that materialises *its own* backend context on the same device (a real kernel
# launch, so the runtime/context fully allocates (enumeration alone does not), announces it, then idles
# until the probe kills it. Run via ``python -c`` from the probe (below), so it stays a plain source string.
_HOLDER_SOURCE = """
import sys

try:
    import torch

    if torch.cuda.is_available():
        _dev = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        _dev = "xpu"
    else:
        _dev = None
    if _dev is not None:
        # A matmul loads cuBLAS, so the context materialises the way a real inference process's does (a bare
        # elementwise kernel under-counts it slightly); .item() forces the sync.
        _block = torch.ones((512, 512), device=_dev)
        float((_block @ _block).sum().item())
    sys.stdout.write("HOLDER_READY\\n")
    sys.stdout.flush()
    sys.stdin.readline()  # idle until the probe signals shutdown (or kills us)
except BaseException as exc:  # noqa: BLE001
    sys.stderr.write("HOLDER_ERR:" + repr(exc))
    sys.exit(4)
"""

_PROBE_SOURCE = f"""
import json
import subprocess
import sys
import threading

try:
    from hordelib.utils.torch_memory import enumerate_accelerators
    # Device-wide free (mem_get_info), NOT comfy's per-process view (torch_memory.get_torch_free_vram_mb):
    # only the device-wide figure sees a *sibling* process's context, which is the whole point of the
    # second-context measurement below. (On Windows WDDM even this is per-process, so the marginal reads 0
    # there and the worker falls back; on the Linux servers the worker targets it is true device-wide.)
    from hordelib.api import get_torch_device_free_vram_mb, get_torch_total_vram_mb

    _accelerators = enumerate_accelerators()

    def _device_used_mb():
        return max(0, int(get_torch_total_vram_mb()) - int(get_torch_device_free_vram_mb()))

    def _materialize_context():
        import torch
        if torch.cuda.is_available():
            _dev = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            _dev = "xpu"
        else:
            return
        # Match the holder: a matmul loads cuBLAS so this process's context materialises like a real
        # inference process's, and the figures (first-context overhead and the marginal) stay comparable.
        _block = torch.ones((512, 512), device=_dev)
        float((_block @ _block).sum().item())

    # First/sole context: materialise this process's context and read device-wide used (the one-time runtime
    # cost plus one context (plus any fixed device baseline, e.g. a desktop compositor), the figure that
    # survives at sole residency, which the forecast subtracts from total VRAM for free_if_alone.
    try:
        _materialize_context()
        _overhead_mb = _device_used_mb()
    except BaseException:
        _overhead_mb = 0

    # Marginal cost of an *additional* sibling context: bring up a second process that materialises its own
    # context, then measure the device-wide used delta. The one-time runtime (and any device baseline) is
    # already counted, so the delta is what each extra inference process really costs (the per-context figure
    # the forecast multiplies by (process count - 1) for free_after_model_evict, instead of charging the whole
    # one-time-inclusive overhead per process. Best-effort: any failure leaves it 0 and the worker falls back.
    _marginal_mb = 0
    _holder = None
    try:
        _holder = subprocess.Popen(
            [sys.executable, "-c", {_HOLDER_SOURCE!r}],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        _ready = {{"ok": False}}

        def _await_ready():
            # Scan the holder's stdout for the sentinel rather than reading one line: importing hordelib
            # prints telemetry banners to stdout first. An empty read is EOF (the holder died).
            while True:
                line = _holder.stdout.readline()
                if not line:
                    return
                if line.strip() == "HOLDER_READY":
                    _ready["ok"] = True
                    return

        _waiter = threading.Thread(target=_await_ready, daemon=True)
        _waiter.start()
        _waiter.join(timeout=90)  # bound a hung holder so it never costs the basic device inventory
        if _ready["ok"]:
            _marginal_mb = max(0, _device_used_mb() - _overhead_mb)
    except BaseException:
        _marginal_mb = 0
    finally:
        if _holder is not None:
            try:
                _holder.kill()
            except BaseException:
                pass

    _payload = [
        {{
            "index": int(a.index),
            "name": str(a.name),
            "total_vram_mb": int(a.total_vram_mb),
            "kind": str(a.kind),
            "runtime_overhead_mb": _overhead_mb,
            "marginal_overhead_mb": _marginal_mb,
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
    kind: str = "cuda"
    """The accelerator backend reported by the probe (``cuda``/``rocm``/``xpu``/``directml``/...). Used to
    pin each inference process to its device with the right backend mask. Defaults to ``cuda`` for probes/
    serialisations that predate this field."""
    runtime_overhead_mb: int = 0
    """Approx. VRAM (MB) the *first/sole* fresh torch process consumes on the idle device: the one-time
    CUDA-runtime/kernel allocation plus one context. Sizes free-if-alone. Defaults to 0 for probes/
    serialisations that predate this field."""
    marginal_overhead_mb: int = 0
    """Approx. VRAM (MB) each *additional* sibling process's context costs once the first has paid the shared
    one-time runtime cost, measured by bringing up a second context and taking the device-wide used delta.
    On one GPU this is several times smaller than ``runtime_overhead_mb``. Sizes free-after-model-evict.
    0 when it could not be measured (single-context backends, probe failure), where the worker seeds a
    conservative per-additional-context constant (``resource_budget._SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB``)
    rather than re-charging the first-context ``runtime_overhead_mb`` against every context."""


def probe_accelerators(*, timeout_seconds: float = 120.0) -> list[ProbedAccelerator]:
    """Return the machine's accelerators by enumerating them in a short-lived subprocess.

    Keeps the calling (orchestrator/wizard) process torch-free: the subprocess loads torch, answers, and
    exits. Never raises: any failure (no backend, subprocess error or timeout, malformed output) is
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
