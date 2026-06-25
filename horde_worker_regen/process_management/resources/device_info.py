"""Torch device information models."""

from __future__ import annotations

from pydantic import BaseModel, RootModel


class TorchDeviceInfo(BaseModel):
    """Contains information about a torch device."""

    device_name: str
    device_index: int
    total_memory: int
    kind: str = "cuda"
    """The accelerator backend (``cuda``/``rocm``/``xpu``/``directml``/...), from the accelerator probe.

    Drives per-process device pinning on a multi-GPU host (the mask env var differs by backend, e.g.
    ``CUDA_VISIBLE_DEVICES`` vs ``HIP_VISIBLE_DEVICES``). Defaults to ``cuda`` so device maps built before
    this field existed (and older serialisations) stay valid."""


class TorchDeviceMap(RootModel[dict[int, TorchDeviceInfo]]):  # TODO
    """A mapping of device IDs to TorchDeviceInfo objects. Contains some helper methods."""
