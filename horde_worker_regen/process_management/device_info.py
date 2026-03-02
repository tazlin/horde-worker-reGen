"""Torch device information models."""

from __future__ import annotations

from pydantic import BaseModel, RootModel


class TorchDeviceInfo(BaseModel):
    """Contains information about a torch device."""

    device_name: str
    device_index: int
    total_memory: int


class TorchDeviceMap(RootModel[dict[int, TorchDeviceInfo]]):  # TODO
    """A mapping of device IDs to TorchDeviceInfo objects. Contains some helper methods."""
