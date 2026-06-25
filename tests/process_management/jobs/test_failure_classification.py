"""Tests for the inference-failure classifier that drives degraded-retry decisions."""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.jobs.failure_classification import is_resource_failure
from horde_worker_regen.process_management.simulation.fault_injection import FAULT_INFO_PREFIX, FaultKind


@pytest.mark.parametrize(
    "info",
    [
        "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "torch.cuda.OutOfMemoryError: CUDA out of memory",
        "HIP out of memory",
        "cuBLAS error",
        f"{FAULT_INFO_PREFIX}{FaultKind.OOM}",
    ],
)
def test_resource_failures_are_recognized(info: str) -> None:
    """Real torch/CUDA/HIP out-of-memory wording and the injected OOM marker all classify as resource."""
    assert is_resource_failure(info) is True


@pytest.mark.parametrize(
    "info",
    [
        None,
        "",
        "fake inference",
        "12.3 it/s",
        "ValueError: bad prompt",
        f"{FAULT_INFO_PREFIX}fail_every_n",
    ],
)
def test_non_resource_failures_are_not_misclassified(info: str | None) -> None:
    """An empty/unknown reason, a normal rate string, and non-OOM faults are not treated as resource."""
    assert is_resource_failure(info) is False
