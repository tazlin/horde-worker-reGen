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
)
"""Lower-cased substrings that mark a faulted result as a recoverable resource (VRAM/RAM) failure.

Covers torch CUDA/cuBLAS allocator messages, the AMD/HIP equivalent, and the chaos harness's injected
out-of-memory marker (``FAULT_INFO_PREFIX`` + ``FaultKind.OOM`` in ``fault_injection``)."""


def is_resource_failure(info: str | None) -> bool:
    """Whether a faulted inference result's ``info`` indicates a recoverable resource (VRAM/RAM) failure.

    Returns False for an empty/unknown ``info`` so that, absent any positive signal, a fault is treated as
    a generic (non-resource) failure rather than over-promising a degraded retry that would not help.
    """
    if not info:
        return False
    lowered = info.lower()
    return any(marker in lowered for marker in _RESOURCE_FAILURE_MARKERS)
