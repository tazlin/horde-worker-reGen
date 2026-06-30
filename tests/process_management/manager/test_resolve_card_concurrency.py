"""Unit tests for ``resolve_card_concurrency``, focused on the alchemist-only fleet-sizing collapse.

A worker that does not serve image generation (an alchemist-only worker, whether by CPU install or a
deliberate ``dreamer: false`` opt-out) must spawn a single inference process per card regardless of the
configured threads/queue, since graph alchemy forms serialize through one process. A dreamer/mixed
worker keeps the existing sizing untouched.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.process_manager import resolve_card_concurrency


@pytest.mark.parametrize("max_threads", [1, 2, 4])
@pytest.mark.parametrize("queue_size", [0, 1, 2])
def test_alchemist_only_forces_single_inference_process(max_threads: int, queue_size: int) -> None:
    """With image generation not served, the process count collapses to one for any threads/queue."""
    result = resolve_card_concurrency(
        max_threads=max_threads,
        queue_size=queue_size,
        num_models_to_load=0,
        gpu_sampling_lease_enabled=False,
        gpu_sampling_lease_slots=None,
        max_threads_ceiling=max_threads,
        serves_image_generation=False,
    )
    assert result.target_process_count == 1


def test_dreamer_sizing_unchanged_by_default() -> None:
    """Serving image generation (the default) keeps the queue_size + ceiling process count."""
    result = resolve_card_concurrency(
        max_threads=2,
        queue_size=2,
        num_models_to_load=5,
        gpu_sampling_lease_enabled=False,
        gpu_sampling_lease_slots=None,
        max_threads_ceiling=2,
    )
    assert result.target_process_count == 4  # queue_size (2) + ceiling (2)


def test_dreamer_single_model_single_thread_still_collapses() -> None:
    """The pre-existing single-model/single-thread collapse is preserved for the image path."""
    result = resolve_card_concurrency(
        max_threads=1,
        queue_size=2,
        num_models_to_load=1,
        gpu_sampling_lease_enabled=False,
        gpu_sampling_lease_slots=None,
        max_threads_ceiling=1,
    )
    assert result.target_process_count == 1
