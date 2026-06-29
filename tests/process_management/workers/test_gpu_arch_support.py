"""Unit tests for the torch-build-vs-GPU architecture compatibility check."""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.workers.inference_process import _gpu_arch_supported

# The compiled architecture set a stable cu126 wheel reports (matches the worker's real install logs):
# binary cubins through Hopper, no Blackwell and no forward PTX.
_CU126_ARCH = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
# A CUDA 13 wheel: Turing through Blackwell, pre-Turing dropped.
_CU130_ARCH = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


@pytest.mark.parametrize(
    ("arch_list", "capability", "expected"),
    [
        # cu126 covers Ampere/Hopper but has no kernel image for Blackwell sm_120 -> unsupported.
        (_CU126_ARCH, (8, 6), True),
        (_CU126_ARCH, (9, 0), True),
        (_CU126_ARCH, (12, 0), False),  # the RTX 50-series failure from the logs
        (_CU126_ARCH, (10, 0), False),  # datacenter Blackwell sm_100
        # cu130 covers Blackwell but dropped pre-Turing, so an old Pascal/Maxwell card is unsupported.
        (_CU130_ARCH, (12, 0), True),
        (_CU130_ARCH, (7, 5), True),
        (_CU130_ARCH, (6, 1), False),  # Pascal GTX 10-series
        (_CU130_ARCH, (5, 2), False),  # Maxwell
        # Binary cubins are forward-compatible only within a major: sm_86 cubin runs sm_8.9 (Ada)...
        (["sm_80", "sm_86"], (8, 9), True),
        # ...but never across a major boundary (no sm_9x cubin here).
        (["sm_80", "sm_86"], (9, 0), False),
        # PTX (compute_*) JIT-forwards across majors, so compute_90 PTX can target Blackwell sm_120.
        (["sm_90", "compute_90"], (12, 0), True),
    ],
)
def test_gpu_arch_supported(arch_list: list[str], capability: tuple[int, int], expected: bool) -> None:
    """_gpu_arch_supported mirrors CUDA's cubin (within-major) and PTX (forward) compatibility rules."""
    assert _gpu_arch_supported(arch_list, capability) is expected


def test_gpu_arch_supported_ignores_malformed_entries() -> None:
    """Non sm_/compute_ tags (e.g. a ROCm gfx target) are skipped rather than crashing the check."""
    assert _gpu_arch_supported(["gfx1100", "sm_90"], (9, 0)) is True
    assert _gpu_arch_supported(["gfx1100"], (9, 0)) is False
