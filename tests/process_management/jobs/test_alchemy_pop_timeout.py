"""The alchemy pop loop's request bound and the offer bookkeeping it does around each pop."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from loguru import logger

from horde_worker_regen.process_management.jobs import alchemy_popper
from horde_worker_regen.process_management.jobs.alchemy_popper import lane_bound_post_processor_candidates
from horde_worker_regen.process_management.models.model_availability import PostProcessorPresence
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
    # The hold is its own instant: the pop time itself is what the worker's "last pop" figure reads, so a
    # backoff written into it would report a pop that has not happened yet.
    assert coordinator._pop_hold_until >= started_at + coordinator._error_pop_frequency
    assert coordinator.last_pop_time <= time.time()


async def test_withheld_post_processors_are_logged_only_when_the_set_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The withheld-post-processor line is logged when the withheld set changes and once when it clears.

    Alchemy pops on a short interval, so a line emitted per pop would bury the log while a first-run
    worker downloads its weights.
    """
    manager = make_testable_process_manager(alchemist=True)
    coordinator = manager._alchemy_coordinator
    monkeypatch.setattr(coordinator, "_should_pop", lambda: True)
    coordinator.bridge_data.priority_usernames = []

    session = Mock()
    session.submit_request = AsyncMock(return_value=SimpleNamespace(forms=[], skipped=None))
    api_sessions = Mock()
    api_sessions.require_horde_client_session.return_value = session
    coordinator._api_sessions = api_sessions

    def _report_present(lane_bound: frozenset[str]) -> None:
        manager._model_availability.update(
            present=set(),
            currently_downloading=None,
            pending=(),
            failed=(),
            post_processor_presence=PostProcessorPresence(lane_bound=lane_bound),
        )

    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="INFO")
    try:
        _report_present(frozenset())
        await coordinator.api_alchemy_pop()
        await coordinator.api_alchemy_pop()
        withheld_lines = [line for line in lines if "withholding" in line]

        _report_present(frozenset(lane_bound_post_processor_candidates()))
        await coordinator.api_alchemy_pop()
        await coordinator.api_alchemy_pop()
    finally:
        logger.remove(sink_id)

    assert len(withheld_lines) == 1, lines
    assert "GFPGAN" in withheld_lines[0]
    assert [line for line in lines if "withholding" in line] == withheld_lines, "a later pop re-logged the same set"
    assert sum("all are offered" in line for line in lines) == 1, lines
