"""The post-processing lane reclaims its own VRAM and retries once on a CUDA OOM before faulting.

The lane sees a CUDA out-of-memory as a generic error wrapping the allocator text (ComfyUI swallows the
typed error), so the retry is keyed on the fingerprint. Only a genuine OOM earns a reclaim-and-retry; a
non-OOM failure, or a second OOM after reclaiming, propagates to the caller's fault handling unchanged.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.workers.post_process_process import HordePostProcessProcess

# The wrapped RuntimeError shape the lane actually observes when a chain OOMs inside ComfyUI.
_LANE_OOM_MESSAGE = (
    "Pipeline failed to run - declared output node(s) produced no results. "
    "Model: RealESRGAN_x4plus. Error: CUDA out of memory. Tried to allocate 1024.00 MiB."
)


def _bare_lane() -> HordePostProcessProcess:
    """A lane instance with just the state the retry helper touches (no hordelib/torch bring-up)."""
    inst = object.__new__(HordePostProcessProcess)
    inst._reclaim_own_vram_for_retry = Mock()  # type: ignore[method-assign]
    return inst


def test_oom_triggers_single_reclaim_and_retry() -> None:
    """A first-attempt OOM reclaims the lane's VRAM once and the retry's result is returned."""
    inst = _bare_lane()
    attempts: list[int] = []

    def run() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError(_LANE_OOM_MESSAGE)
        return "post-processed"

    assert inst._run_with_oom_retry(run, context="job A") == "post-processed"
    assert len(attempts) == 2
    inst._reclaim_own_vram_for_retry.assert_called_once()  # type: ignore[attr-defined]


def test_non_oom_error_propagates_without_reclaim() -> None:
    """A non-OOM failure is not retried; reclaim is never issued and the error reaches the caller."""
    inst = _bare_lane()

    def run() -> str:
        raise ValueError("Alchemy form produced no image")

    with pytest.raises(ValueError):
        inst._run_with_oom_retry(run, context="job B")
    inst._reclaim_own_vram_for_retry.assert_not_called()  # type: ignore[attr-defined]


def test_second_oom_after_reclaim_propagates() -> None:
    """When even the reclaimed retry OOMs, the error propagates so the job faults without images."""
    inst = _bare_lane()

    def run() -> str:
        raise RuntimeError(_LANE_OOM_MESSAGE)

    with pytest.raises(RuntimeError):
        inst._run_with_oom_retry(run, context="job C")
    inst._reclaim_own_vram_for_retry.assert_called_once()  # type: ignore[attr-defined]
