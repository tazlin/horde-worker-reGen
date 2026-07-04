"""Tests for the single-sourced CUDA out-of-memory fingerprint.

Both the live post-processing lane (to reclaim VRAM and retry) and the log analyzer (to classify faults)
match on this signal, so the cases here pin the shapes it must recognize, including the wrapped
``RuntimeError`` the lane actually sees when ComfyUI swallows the typed error.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.utils.oom_signature import is_out_of_memory_text

# The message the lane observes: ComfyUI catches the CUDA OOM and it re-emerges wrapped in the pipeline's
# "produced no results" RuntimeError, carrying the original allocator text.
_WRAPPED_LANE_OOM = (
    "RuntimeError: Pipeline failed to run - declared output node(s) {'output_image'} produced no results. "
    "Model: RealESRGAN_x4plus. Error: CUDA out of memory. Tried to allocate 1024.00 MiB. GPU 0 has a total "
    "capacity of 23.51 GiB of which 11.32 GiB is free. 4.00 GiB allowed;"
)


@pytest.mark.parametrize(
    "text",
    [
        "CUDA out of memory. Tried to allocate 1024.00 MiB.",
        "torch.cuda.OutOfMemoryError: CUDA out of memory.",
        "OutOfMemoryError: CUDA out of memory",
        _WRAPPED_LANE_OOM,
    ],
)
def test_recognizes_oom_shapes(text: str) -> None:
    """Typed OOM errors and the wrapped-RuntimeError shape the lane sees are all recognized."""
    assert is_out_of_memory_text(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "ValueError: Unknown alchemy form for post-processing process: bogus",
        "RuntimeError: Alchemy form produced no image",
        "post-processing produced no output image",
        "Too many open files",
    ],
)
def test_non_oom_failures_are_not_matched(text: str) -> None:
    """A non-OOM failure must not be mistaken for an out-of-memory (no reclaim-and-retry for those)."""
    assert is_out_of_memory_text(text) is False
