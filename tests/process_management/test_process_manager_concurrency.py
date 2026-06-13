"""Semantic tests for GPU concurrency resolution (sampling-lease wiring).

These pin the *policy* — how enabling the lease decouples the denoise gate from the whole-job
inference semaphore, and how the slot count is clamped — rather than the literal arithmetic.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.process_manager import _resolve_inference_concurrency


class TestResolveInferenceConcurrency:
    """Policy for sizing the inference semaphore and lease slots as the lease is toggled."""

    def test_lease_disabled_keeps_inference_semaphore_at_concurrent_count(self) -> None:
        """Lease off: the whole-job semaphore stays the only GPU gate (concurrent-sampling count)."""
        semaphore_size, _ = _resolve_inference_concurrency(
            gpu_sampling_lease_enabled=False,
            configured_lease_slots=1,
            max_concurrent_inference_processes=2,
            max_inference_processes=4,
        )
        assert semaphore_size == 2

    def test_lease_enabled_opens_inference_semaphore_to_all_processes(self) -> None:
        """Lease on: the whole-job semaphore opens to every process so spares can stage ahead."""
        semaphore_size, _ = _resolve_inference_concurrency(
            gpu_sampling_lease_enabled=True,
            configured_lease_slots=1,
            max_concurrent_inference_processes=2,
            max_inference_processes=4,
        )
        assert semaphore_size == 4

    def test_default_one_slot_serializes_denoise(self) -> None:
        """A single configured slot serializes the denoise loop."""
        _, slots = _resolve_inference_concurrency(
            gpu_sampling_lease_enabled=True,
            configured_lease_slots=1,
            max_concurrent_inference_processes=2,
            max_inference_processes=4,
        )
        assert slots == 1

    def test_slots_independent_of_concurrent_process_count(self) -> None:
        """The slot count is the configured value, not bolted to the concurrent-process count."""
        _, slots = _resolve_inference_concurrency(
            gpu_sampling_lease_enabled=True,
            configured_lease_slots=2,
            max_concurrent_inference_processes=4,
            max_inference_processes=6,
        )
        assert slots == 2

    def test_slots_clamped_to_process_count(self) -> None:
        """Slots can never exceed the number of inference processes."""
        _, slots = _resolve_inference_concurrency(
            gpu_sampling_lease_enabled=True,
            configured_lease_slots=99,
            max_concurrent_inference_processes=2,
            max_inference_processes=4,
        )
        assert slots == 4

    @pytest.mark.parametrize("configured", [0, -5])
    def test_slots_floored_to_one(self, configured: int) -> None:
        """At least one denoise loop may always run, even if misconfigured below one."""
        _, slots = _resolve_inference_concurrency(
            gpu_sampling_lease_enabled=True,
            configured_lease_slots=configured,
            max_concurrent_inference_processes=2,
            max_inference_processes=4,
        )
        assert slots == 1
