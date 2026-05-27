"""Live, hot-reloadable bridge configuration.

Public members:
    ``RuntimeConfig`` — single-writer/many-reader holder for ``reGenBridgeData``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData


class RuntimeConfig:
    """Holds the current bridge configuration snapshot.

    The reload loop is the only writer; components read via ``bridge_data``.
    Reads return whatever snapshot is current at the moment of the read — no
    atomicity is promised across multiple reads.
    """

    _bridge_data: reGenBridgeData

    def __init__(self, initial: reGenBridgeData) -> None:
        """Initialize with the first bridge configuration."""
        self._bridge_data = initial

    @property
    def bridge_data(self) -> reGenBridgeData:
        """Return the current bridge configuration snapshot."""
        return self._bridge_data

    def update(self, new: reGenBridgeData) -> None:
        """Replace the current configuration with ``new``."""
        self._bridge_data = new
