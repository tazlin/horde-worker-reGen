"""Classify a faulted inference result so the retry policy can decide *how* to retry it.

A job can fault for two broadly different reasons, and the worker should treat them differently:

- A **resource failure** (CUDA/HIP out-of-memory, an allocator failure) is often transient: the same
  job may succeed if it is retried with less concurrent VRAM pressure. Such a failure earns one
  *degraded* (isolated) retry before being reported faulted.
- Any **other** fault (malformed input, a model/graph error, an unexpected exception) is unlikely to be
  fixed by simply re-running, so it only gets the ordinary bounded retry, if any.

The signal we have to classify on is the faulted result's ``info`` string. Real inference tags it with
the originating exception summary (see ``inference_process.send_inference_result_message``); the chaos
harness tags an injected out-of-memory fault with its own marker. This module recognizes both. It is
deliberately dependency-free and substring-based so it cannot itself raise on a surprising message.

A subtlety the multi-process worker exposed: under VRAM over-commit the *driver* OOM is often caught and
swallowed by ComfyUI ("Got an OOM, unloading all loaded models"), so the pipeline yields no output node
and hordelib raises a generic ``Pipeline failed to run - no images were produced`` instead. The torch
"CUDA out of memory" wording never reaches us; the only surface signal is "no images were produced".
That phrasing is therefore treated as a resource-class failure too: on a worker running concurrent
inference processes, a pipeline that produced nothing is, in practice, the visible end of an OOM that was
handled out of view. The cost of a rare false positive is bounded (one device-clearing isolated retry,
and the per-model breaker only trips when a model produces no images on *every* attempt, since a single
success resets its streak), while the benefit is that the over-budget breaker and self-throttle backstops
finally see the storm instead of leaving it to the disruptive save-our-ship soft reset.
"""

from __future__ import annotations

_RESOURCE_FAILURE_MARKERS: tuple[str, ...] = (
    "out of memory",
    "outofmemoryerror",
    "cuda out of memory",
    "cuda error",
    "hip out of memory",
    "cublas",
    "injected-fault:oom",
    "no images were produced",
)
"""Lower-cased substrings that mark a faulted result as a recoverable resource (VRAM/RAM) failure.

Covers torch CUDA/cuBLAS allocator messages, the AMD/HIP equivalent, the chaos harness's injected
out-of-memory marker (``FAULT_INFO_PREFIX`` + ``FaultKind.OOM`` in ``fault_injection``), and the
swallowed-OOM surface form (hordelib's ``no images were produced`` when ComfyUI caught the driver OOM and
the pipeline yielded no output node) -- see the module docstring for why the empty-result case is folded
in on a multi-process worker."""


def is_resource_failure(info: str | None) -> bool:
    """Whether a faulted inference result's ``info`` indicates a recoverable resource (VRAM/RAM) failure.

    Returns False for an empty/unknown ``info`` so that, absent any positive signal, a fault is treated as
    a generic (non-resource) failure rather than over-promising a degraded retry that would not help.
    """
    if not info:
        return False
    lowered = info.lower()
    return any(marker in lowered for marker in _RESOURCE_FAILURE_MARKERS)
