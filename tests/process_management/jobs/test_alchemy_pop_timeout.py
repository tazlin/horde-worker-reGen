"""The alchemy pop loop owns a short request bound so it remains shutdown-responsive."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest

from horde_worker_regen.process_management.jobs import alchemy_popper
from tests.process_management.conftest import make_testable_process_manager


async def test_alchemy_pop_times_out_and_enters_error_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """An API request that never answers returns control instead of pinning the gathered main loop."""
    manager = make_testable_process_manager(alchemist=True)
    coordinator = manager._alchemy_coordinator
    monkeypatch.setattr(coordinator, "_should_pop", lambda: True)
    monkeypatch.setattr(alchemy_popper, "ALCHEMY_POP_REQUEST_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(alchemy_popper, "expand_offered_forms", lambda *_args, **_kwargs: ["nsfw"])
    coordinator.bridge_data.priority_usernames = []

    request_cancelled = asyncio.Event()

    async def _never_returns(*_args: object, **_kwargs: object) -> object:
        try:
            await asyncio.Event().wait()
        finally:
            request_cancelled.set()

    session = Mock()
    session.submit_request = AsyncMock(side_effect=_never_returns)
    api_sessions = Mock()
    api_sessions.require_horde_client_session.return_value = session
    coordinator._api_sessions = api_sessions

    started_at = time.time()
    await asyncio.wait_for(coordinator.api_alchemy_pop(), timeout=1.0)

    session.submit_request.assert_awaited_once()
    assert request_cancelled.is_set(), "wait_for did not cancel the timed-out SDK request"
    assert coordinator._last_pop_time >= started_at + (coordinator._error_pop_frequency - coordinator._pop_frequency)
