"""Recognize a CUDA out-of-memory failure from its text.

A CUDA OOM does not always reach the worker as a ``torch.cuda.OutOfMemoryError``: ComfyUI catches the
error inside node execution and the lane sees a generic ``RuntimeError`` whose message wraps the original
"CUDA out of memory" text. Both the live worker (to decide whether to reclaim VRAM and retry a chain) and
the post-hoc log analyzer (to classify a session's faults) need the same signal, so the signature lives
here once rather than being re-spelled in each place.
"""

from __future__ import annotations

import re
import sys

OOM_TEXT_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError|RuntimeError: .*out of memory",
)
"""Matches the CUDA out-of-memory fingerprints, whether raised as a typed error or wrapped in a message."""


def is_out_of_memory_text(text: str) -> bool:
    """Return whether ``text`` bears a CUDA out-of-memory fingerprint."""
    return OOM_TEXT_RE.search(text) is not None


def is_resource_class_exception(exc: BaseException) -> bool:
    """Return whether an exception is a CUDA out-of-memory (device resource) failure. Never raises.

    A CUDA OOM may arrive typed as ``torch.cuda.OutOfMemoryError`` or, when ComfyUI swallows it inside node
    execution, as a generic error whose message wraps the out-of-memory text. Both classify as resource-class
    so a disaggregated stage can defer-then-retry (or the job be re-routed monolithically) rather than being
    forfeited. torch is consulted only if already imported, so this stays torch-free where torch was never
    loaded; any failure of the classification itself is swallowed and reported as not-resource-class.
    """
    try:
        torch_module = sys.modules.get("torch")
        if torch_module is not None:
            oom_error = getattr(getattr(torch_module, "cuda", None), "OutOfMemoryError", None)
            if isinstance(oom_error, type) and isinstance(exc, oom_error):
                return True
        return is_out_of_memory_text(f"{type(exc).__name__}: {exc}")
    except Exception:
        return False
