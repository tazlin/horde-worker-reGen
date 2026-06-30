"""Tests for the safety process's device resolution.

The parent decides ``cpu_only`` from config plus the torch-free install sentinel, which can disagree
with the actual torch build (a manually installed CPU torch whose ``bin/backend`` sentinel was never
written reports ``cpu_only=False``). The torch-bearing safety child must still load its models on CPU
when CUDA is genuinely unavailable, or horde_safety raises during deserialization.
"""

from __future__ import annotations

from horde_worker_regen.process_management.workers.safety_process import resolve_safety_device


def test_cpu_only_always_resolves_to_cpu() -> None:
    """An explicit cpu_only request uses CPU whether or not CUDA is present."""
    assert resolve_safety_device(cpu_only=True, cuda_available=True) == "cpu"
    assert resolve_safety_device(cpu_only=True, cuda_available=False) == "cpu"


def test_falls_back_to_cpu_when_cuda_unavailable() -> None:
    """The regression: cpu_only=False but no real CUDA must not try to load on 'cuda'."""
    assert resolve_safety_device(cpu_only=False, cuda_available=False) == "cpu"


def test_uses_cuda_when_requested_and_available() -> None:
    """A normal GPU worker is unaffected: CUDA is used when requested and present."""
    assert resolve_safety_device(cpu_only=False, cuda_available=True) == "cuda"
