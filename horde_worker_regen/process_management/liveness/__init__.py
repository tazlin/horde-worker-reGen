"""Cross-subsystem liveness declarations for the worker's control plane.

Gates that can hold or defer work are spread across popping, scheduling, governance and lifecycle. This
package is where the *declarations* about them live: statements that are true of a gate regardless of which
subsystem implements it, and that no single subsystem can own without implying it decides for the others.

Nothing here runs on a control-loop path. The declarations are consulted by tests and by people.
"""

from horde_worker_regen.process_management.liveness.gate_registry import (
    GATE_REGISTRY,
    GateEntry,
    GateKind,
    GateSurface,
    entry_for,
    registered_keys,
)

__all__ = [
    "GATE_REGISTRY",
    "GateEntry",
    "GateKind",
    "GateSurface",
    "entry_for",
    "registered_keys",
]
