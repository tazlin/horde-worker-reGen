"""Unit tests for the container torch-vs-GPU preflight."""

from __future__ import annotations

import pytest

from horde_worker_regen import torch_gpu_preflight


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((12, 0), "cu130"),  # Blackwell sm_120 (RTX 50-series)
        ((10, 0), "cu130"),  # datacenter Blackwell sm_100
        ((9, 0), "cu126"),  # Hopper is within the cu126 window
        ((7, 0), "cu126"),  # Volta pre-Turing, held on cu126
        ((6, 1), "cu126"),  # Pascal
    ],
)
def test_recommended_backend(capability: tuple[int, int], expected: str) -> None:
    """A card above the cu126 arch ceiling is steered to cu130; anything else to cu126."""
    assert torch_gpu_preflight.recommended_backend(capability) == expected


def test_incompatibility_message_is_actionable() -> None:
    """The fatal message names the card, its arch tag, the build, and the exact rebuild command."""
    message = torch_gpu_preflight.incompatibility_message(
        device_name="NVIDIA GeForce RTX 5090",
        capability=(12, 0),
        arch_list=["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"],
        torch_version="2.12.1+cu126",
        torch_build="12.6",
    )
    assert "RTX 5090" in message
    assert "sm_120" in message
    assert "TORCH_BACKEND=cu130" in message
    assert "immutable" in message
    assert "2.12.1+cu126" in message


def test_incompatibility_message_handles_unknown_build_tag() -> None:
    """A missing torch CUDA build tag still yields a coherent message (no 'None' leaking in)."""
    message = torch_gpu_preflight.incompatibility_message(
        device_name="Some GPU",
        capability=(12, 0),
        arch_list=["sm_90"],
        torch_version="2.12.1",
        torch_build=None,
    )
    assert "CUDA None" not in message
    assert "2.12.1" in message
