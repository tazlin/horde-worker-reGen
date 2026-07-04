"""Thin, lazily-importing bridge to hordelib's cross-process component-sharing fabric.

The orchestrator parent must stay torch-free, and sharing is opt-in (HORDE_SHARED_COMPONENTS=1), so
the hordelib import happens only inside these functions, only when the feature is enabled. The bus
itself is plain ``multiprocessing`` primitives; tensor payloads never transit the parent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import multiprocessing


def build_component_bus(ctx: multiprocessing.context.BaseContext, process_ids: list[int]) -> Any:  # noqa: ANN401
    """Construct the sharing bus for the given child process ids (see SharedComponentBus)."""
    from hordelib.execution.shared_components import SharedComponentBus

    return SharedComponentBus(ctx, process_ids)


def endpoint_for(bus: Any, process_id: int) -> Any | None:  # noqa: ANN401
    """The endpoint to hand a child, or None when the bus is absent or the pid was not declared."""
    if bus is None:
        return None
    try:
        return bus.endpoint_for(process_id)
    except Exception:  # noqa: BLE001 - an undeclared pid (unexpected growth) just runs unshared
        return None
