"""Tests that the worker's config reload runs off the event loop so a slow resolve never wedges it."""

from __future__ import annotations

import asyncio
import threading

from tests.process_management.conftest import make_testable_process_manager


async def test_reload_runs_blocking_load_off_loop_and_applies_on_loop() -> None:
    """The network-bound load runs in a worker thread; the swap-in happens back on the event loop."""
    process_manager = make_testable_process_manager()
    main_thread = threading.get_ident()
    sentinel = object()
    load_thread: dict[str, int] = {}
    applied: dict[str, object] = {}

    def fake_load() -> object:
        load_thread["id"] = threading.get_ident()
        return sentinel

    def fake_apply(bridge_data: object) -> None:
        applied["bridge_data"] = bridge_data
        applied["thread"] = threading.get_ident()

    process_manager._bridge_data_reloader.load_bridge_data_blocking = fake_load  # type: ignore[assignment,method-assign]
    process_manager._bridge_data_reloader._apply_bridge_data = fake_apply  # type: ignore[assignment,method-assign]

    await process_manager._bridge_data_reloader.reload_bridge_data_off_loop()

    assert applied["bridge_data"] is sentinel
    assert load_thread["id"] != main_thread, "the blocking load must not run on the event loop thread"
    assert applied["thread"] == main_thread, "the apply must run on the event loop thread"


async def test_reload_skips_apply_when_nothing_to_load() -> None:
    """A failed/skipped load (None) leaves the current config untouched."""
    process_manager = make_testable_process_manager()
    applied: dict[str, bool] = {}

    process_manager._bridge_data_reloader.load_bridge_data_blocking = lambda: None  # type: ignore[assignment,method-assign]
    process_manager._bridge_data_reloader._apply_bridge_data = lambda bridge_data: applied.setdefault("called", True)  # type: ignore[assignment,method-assign]

    await process_manager._bridge_data_reloader.reload_bridge_data_off_loop()

    assert "called" not in applied


async def test_schedule_config_reload_runs_the_reload() -> None:
    """A supervisor-triggered reload schedules the off-loop reload and the task completes."""
    process_manager = make_testable_process_manager()
    done = asyncio.Event()

    async def fake_reload() -> None:
        done.set()

    process_manager._bridge_data_reloader.reload_bridge_data_off_loop = fake_reload  # type: ignore[assignment,method-assign]

    process_manager._bridge_data_reloader.schedule_config_reload()
    await asyncio.wait_for(done.wait(), timeout=1.0)
