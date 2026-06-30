"""Tests for the runtime alchemist-only inference-fleet collapse.

When an inference child reports a CPU-only torch build at runtime (the sentinel-less manual CPU install),
the worker, which came up sized for image generation, must collapse to one inference process per card:
image generation is disabled and a single process serves the graph alchemy forms. This mirrors the
startup ``serves_image_generation=False`` sizing the install sentinel would have produced.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

from tests.process_management.conftest import make_testable_process_manager


def _set_card_target(manager: object, target: int) -> int:
    """Set the single test card's target_process_count, mutating the shared dict, and return its index."""
    index = next(iter(manager._card_runtimes))  # type: ignore[attr-defined]
    manager._card_runtimes[index] = replace(manager._card_runtimes[index], target_process_count=target)  # type: ignore[attr-defined]
    return index


def test_scale_down_lowers_target_and_reaps_when_cpu_build_detected() -> None:
    """With the CPU-build flag set, each card's target drops to one and the extra contexts are reaped."""
    manager = make_testable_process_manager()
    index = _set_card_target(manager, 3)

    manager._process_map.num_loaded_inference_processes = Mock(return_value=3)  # type: ignore[method-assign]
    manager._process_lifecycle.scale_inference_processes = Mock(return_value=1)  # type: ignore[method-assign]
    manager._process_lifecycle.refresh_max_inference_processes = Mock()  # type: ignore[method-assign]

    manager._state.torch_build_cpu_only = True
    manager._enforce_alchemist_only_scale_down()

    assert manager._card_runtimes[index].target_process_count == 1
    assert manager.max_inference_processes == 1
    manager._process_lifecycle.refresh_max_inference_processes.assert_called_once()
    manager._process_lifecycle.scale_inference_processes.assert_called_once_with(1, device_index=index)


def test_scale_down_is_a_noop_without_cpu_build_flag() -> None:
    """A normal (GPU/dreamer) worker is untouched: target is preserved and nothing is reaped."""
    manager = make_testable_process_manager()
    index = _set_card_target(manager, 3)

    manager._process_lifecycle.scale_inference_processes = Mock(return_value=3)  # type: ignore[method-assign]

    assert manager._state.torch_build_cpu_only is False
    manager._enforce_alchemist_only_scale_down()

    assert manager._card_runtimes[index].target_process_count == 3
    manager._process_lifecycle.scale_inference_processes.assert_not_called()


def test_scale_down_is_idempotent_once_collapsed() -> None:
    """Re-running while already at one neither re-lowers the target nor reaps again."""
    manager = make_testable_process_manager()
    index = _set_card_target(manager, 1)

    manager._process_map.num_loaded_inference_processes = Mock(return_value=1)  # type: ignore[method-assign]
    manager._process_lifecycle.scale_inference_processes = Mock(return_value=1)  # type: ignore[method-assign]
    manager._process_lifecycle.refresh_max_inference_processes = Mock()  # type: ignore[method-assign]

    manager._state.torch_build_cpu_only = True
    manager._enforce_alchemist_only_scale_down()

    assert manager._card_runtimes[index].target_process_count == 1
    # Already at one: no target change (so no refresh) and nothing to reap.
    manager._process_lifecycle.refresh_max_inference_processes.assert_not_called()
    manager._process_lifecycle.scale_inference_processes.assert_not_called()
