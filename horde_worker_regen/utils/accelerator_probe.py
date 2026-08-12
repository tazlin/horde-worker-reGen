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
  context. This is paid once per device and sizes ``free_if_alone``; it is never the cost of an additional
  context. Like the marginal below it is a before/after delta, so whatever the device already held when the
  probe started (a desktop compositor, another tenant's process) cancels instead of being charged to the
  worker: a device-wide *used* reading taken after the context materialises includes every other tenant on
  the card, and that figure applied as a per-process overhead prices a small card out of models it serves;
  and
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
    def _nvml_used_mb():
        # NVML queries the driver without creating a CUDA context, and reports true device-wide usage on
        # every platform (including Windows WDDM, where the torch reading below is only this process's view).
        # Both properties are load-bearing for the before/after pair: a "before" reading taken through torch
        # would initialise CUDA and so already contain the context being measured, and a pair read on
        # different bases would not subtract.
        try:
            from hordelib.utils.nvml import get_device_memory_mb

            _memory = get_device_memory_mb(0)
            return None if _memory is None else int(_memory.used_mb)
        except BaseException:
            return None

    # Read the device's pre-existing baseline FIRST, before anything can touch the GPU: enumeration and every
    # torch memory helper initialise CUDA as a side effect.
    _baseline_mb = _nvml_used_mb()

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

    # First/sole context: materialise this process's context, then read used again. The pair is reported raw
    # and the parent takes the delta (see _first_context_overhead_mb): the subtraction is plain arithmetic that
    # must stay directly testable, while this child only ever runs on real hardware. The NVML pair is used when
    # both readings landed, so the two figures share a basis; otherwise the baseline is dropped and the torch
    # reading stands alone (baseline-inclusive, over-counting in the safe direction, as before).
    try:
        _materialize_context()
        _overhead_mb = _device_used_mb()
        _context_nvml_mb = _nvml_used_mb()
    except BaseException:
        _overhead_mb = 0
        _context_nvml_mb = None
    if _baseline_mb is None or _context_nvml_mb is None:
        _context_used_mb = _overhead_mb
        _baseline_mb = None
    else:
        _context_used_mb = _context_nvml_mb

    # Marginal cost of an *additional* sibling context: bring up a second process that materialises its own
    # context, then measure the device-wide used delta. The one-time runtime (and any device baseline) is
    # already counted, so the delta is what each extra inference process really costs (the per-context figure
    # the forecast multiplies by (process count - 1) for free_after_model_evict, instead of charging the whole
    # one-time-inclusive overhead per process. Best-effort: any failure leaves it 0 and the worker falls back.
    _marginal_mb = 0
    _marginal_note = "not attempted"
    _holder = None
    try:
        _holder = subprocess.Popen(
            [sys.executable, "-c", {_HOLDER_SOURCE!r}],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
        if not _ready["ok"]:
            _marginal_note = "holder never signalled ready (timeout or early exit)"
        else:
            # Take the delta on the NVML basis, not the torch one. torch's device-free reading is
            # per-process on Windows WDDM, so it cannot see the holder's context at all and the
            # subtraction is structurally zero there; NVML queries the driver and is device-wide on
            # every platform. Falls back to the torch pair only where NVML is unavailable, which is
            # the reading that at least works on the platforms where it is device-wide.
            _after_nvml_mb = _nvml_used_mb()
            if _after_nvml_mb is not None and _context_nvml_mb is not None:
                _marginal_mb = max(0, _after_nvml_mb - _context_nvml_mb)
                _marginal_note = f"nvml delta {{_after_nvml_mb}}-{{_context_nvml_mb}}"
            else:
                _marginal_mb = max(0, _device_used_mb() - _overhead_mb)
                _marginal_note = "torch delta (nvml unavailable); per-process on WDDM, may read 0"
            if _marginal_mb == 0:
                _marginal_note += "; measured zero"
    except BaseException as _marginal_exc:
        _marginal_mb = 0
        _marginal_note = "raised: " + repr(_marginal_exc)
    finally:
        if _holder is not None:
            try:
                _holder.kill()
                # Drain both pipes after killing. stderr is a pipe so a holder failure is reportable
                # rather than discarded, and an undrained pipe can block the writer once its buffer
                # fills, which would turn a noisy holder into the hang the ready-timeout exists to bound.
                _, _holder_err = _holder.communicate(timeout=10)
                if _holder_err and _marginal_mb == 0:
                    _marginal_note += " | holder stderr: " + _holder_err.strip().replace("\\n", " ")[-300:]
            except BaseException:
                pass

    _payload = [
        {{
            "index": int(a.index),
            "name": str(a.name),
            "total_vram_mb": int(a.total_vram_mb),
            "kind": str(a.kind),
            "context_device_used_mb": _context_used_mb,
            "device_baseline_mb": _baseline_mb,
            "marginal_overhead_mb": _marginal_mb,
            "marginal_note": _marginal_note,
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
    CUDA-runtime/kernel allocation plus one context. Sizes free-if-alone. Derived by the parent as the
    before/after delta over :attr:`device_baseline_mb`, so VRAM other tenants already held is not charged to
    the worker. Defaults to 0 for probes/serialisations that predate this field."""
    context_device_used_mb: int = 0
    """Device-wide VRAM used (MB) measured after the probe materialised its context: the raw reading behind
    :attr:`runtime_overhead_mb`, before the baseline is netted out. Carried so the derivation is inspectable
    and the raw figures stay available for diagnostics."""
    device_baseline_mb: int | None = None
    """Device-wide VRAM used (MB) on this card *before* the probe created any context, read through NVML so
    the reading itself allocates nothing. None when unmeasurable (a non-NVIDIA backend, no driver), where the
    first-context overhead degrades to the baseline-inclusive raw reading."""
    marginal_overhead_mb: int = 0
    """Approx. VRAM (MB) each *additional* sibling process's context costs once the first has paid the shared
    one-time runtime cost, measured by bringing up a second context and taking the device-wide used delta.
    On one GPU this is several times smaller than ``runtime_overhead_mb``. Sizes free-after-model-evict.
    0 when it could not be measured (single-context backends, probe failure), where the worker seeds a
    conservative per-additional-context constant (``resource_budget._SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB``)
    rather than re-charging the first-context ``runtime_overhead_mb`` against every context."""
    marginal_note: str = ""
    """Why :attr:`marginal_overhead_mb` reads what it does, for diagnosis when it reads 0.

    A zero marginal is indistinguishable at the value alone between a genuine measurement, a holder that
    never started, a raised exception, and a platform where the reading cannot see a sibling process at
    all. This carries which of those happened, including the holder's stderr where one failed, so the
    seeded fallback is a visible decision rather than a silent one. Empty for probes that predate it."""


def _first_context_overhead_mb(*, context_device_used_mb: int, device_baseline_mb: int | None) -> int:
    """Return the first/sole context's own VRAM cost (MB): the reading net of the device's prior baseline.

    The raw reading is device-wide, so it contains every tenant already on the card (a desktop compositor,
    another application, a second worker) as well as the context just created. Charged whole as a per-process
    overhead it is not a measurement of the worker at all, and the error scales with how busy the *host*
    is rather than with anything the worker does: a desktop machine's several-GB baseline, applied to a small
    card, removes more of that card's budget than the models it is being asked to serve. Subtracting the
    pre-context baseline leaves the context's own cost, which is what the term means.

    Clamped at 0: the baseline and the post-context reading are separate samples, so an unrelated tenant
    releasing memory in between can invert them, and a negative overhead would credit the worker VRAM it never
    freed. An unmeasurable baseline (None) leaves the raw reading, which over-counts in the safe direction.

    Args:
        context_device_used_mb (int): Device-wide VRAM used (MB) after the probe materialised its context.
        device_baseline_mb (int | None): Device-wide VRAM used (MB) before it did, or None when unmeasurable.
    """
    if device_baseline_mb is None:
        return max(0, context_device_used_mb)
    return max(0, context_device_used_mb - device_baseline_mb)


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
            accelerators = [ProbedAccelerator.model_validate(entry) for entry in raw_entries]
            # Reported once per probe, at info when the marginal is unmeasured. An unmeasured marginal is
            # not a detail: the forecast then prices every additional context from a seeded constant for
            # the whole session, which sizes free-after-model-evict and so the residency and process-count
            # decisions built on it. Left at debug when a real figure landed.
            for accelerator in accelerators:
                if accelerator.marginal_overhead_mb > 0:
                    logger.debug(
                        f"Accelerator {accelerator.index} marginal context cost: "
                        f"{accelerator.marginal_overhead_mb}MB ({accelerator.marginal_note})",
                    )
                else:
                    logger.info(
                        f"Accelerator {accelerator.index} marginal context cost unmeasured; the stream "
                        f"forecast will price additional contexts from its seeded constant. "
                        f"Reason: {accelerator.marginal_note or 'unreported'}",
                    )
            # The child reports the raw before/after pair; the first-context overhead is derived here. An entry
            # carrying no post-context reading (a serialisation that predates the pair) keeps whatever
            # ``runtime_overhead_mb`` it already had rather than being zeroed.
            return [
                (
                    accelerator
                    if accelerator.context_device_used_mb <= 0
                    else accelerator.model_copy(
                        update={
                            "runtime_overhead_mb": _first_context_overhead_mb(
                                context_device_used_mb=accelerator.context_device_used_mb,
                                device_baseline_mb=accelerator.device_baseline_mb,
                            ),
                        },
                    )
                )
                for accelerator in accelerators
            ]
        except (ValueError, TypeError) as parse_error:
            logger.debug(f"Could not parse accelerator probe output: {parse_error}")
            return []

    logger.debug("Accelerator probe produced no result line")
    return []
