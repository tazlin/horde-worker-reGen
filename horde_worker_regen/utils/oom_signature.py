"""Recognize a CUDA out-of-memory failure from its text.

A CUDA OOM does not always reach the worker as a ``torch.cuda.OutOfMemoryError``: ComfyUI catches the
error inside node execution and the lane sees a generic ``RuntimeError`` whose message wraps the original
"CUDA out of memory" text. Both the live worker (to decide whether to reclaim VRAM and retry a chain) and
the post-hoc log analyzer (to classify a session's faults) need the same signal, so the signature lives
here once rather than being re-spelled in each place.
"""

from __future__ import annotations

import re

OOM_TEXT_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError|RuntimeError: .*out of memory",
)
"""Matches the CUDA out-of-memory fingerprints, whether raised as a typed error or wrapped in a message."""


def is_out_of_memory_text(text: str) -> bool:
    """Return whether ``text`` bears a CUDA out-of-memory fingerprint."""
    return OOM_TEXT_RE.search(text) is not None
