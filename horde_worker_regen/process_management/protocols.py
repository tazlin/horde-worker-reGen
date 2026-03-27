"""Runtime dependency protocols for process management components.

Components that need access to mutable, runtime-changeable resources (bridge data,
HTTP sessions) accept providers that satisfy these protocols rather than bare
``Callable[[], T]`` annotations. This gives static analysis a single source of
truth for the contract instead of N divergent ``Callable`` signatures.

Existing lambdas (``lambda: self.bridge_data``) already satisfy these protocols
with zero changes at the call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiohttp import ClientSession
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPIAsyncClientSession

if TYPE_CHECKING:
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData


@runtime_checkable
class BridgeDataProvider(Protocol):
    """Callable that returns the current bridge configuration snapshot."""

    def __call__(self) -> reGenBridgeData: ...


@runtime_checkable
class HordeClientSessionProvider(Protocol):
    """Callable that returns the active AI Horde async client session."""

    def __call__(self) -> AIHordeAPIAsyncClientSession: ...


@runtime_checkable
class AiohttpSessionProvider(Protocol):
    """Callable that returns the shared aiohttp client session."""

    def __call__(self) -> ClientSession: ...
