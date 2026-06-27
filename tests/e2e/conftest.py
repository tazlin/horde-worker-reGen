"""Shared fixtures for end-to-end dry-run tests."""

from __future__ import annotations

import asyncio
import multiprocessing
import sys
import time
from collections.abc import Generator
from unittest.mock import Mock

import pytest
from loguru import logger

from tests.process_management.conftest import (
    make_mock_bridge_data,
    make_testable_process_manager,
)

# On Windows the default ProactorEventLoop (IOCP-based) teardown can race with
# VS Code's named-pipe server that vscode-pytest relies on, causing the pipe to
# disappear before the final test report is sent.  Switching to the selector-based
# event loop avoids IOCP altogether; the e2e tests here do not need IOCP features
# (they use ``multiprocessing``, not ``asyncio.subprocess``).
if sys.platform == "win32":
    _E2E_EVENT_LOOP_POLICY = asyncio.WindowsSelectorEventLoopPolicy()
else:
    _E2E_EVENT_LOOP_POLICY = asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    """Override the event loop policy for all e2e tests in this directory.

    pytest-asyncio uses this fixture (when defined) to determine which policy
    to use when creating event loops for async test functions.
    """
    return _E2E_EVENT_LOOP_POLICY


@pytest.fixture(autouse=True)
def _reap_multiprocessing_children() -> Generator[None, None, None]:
    """Reap lingering multiprocessing children after each e2e test.

    The e2e tests spawn real OS child processes via ``multiprocessing.Process`` (spawn
    context).  The primary fix for the vscode-pytest named-pipe error is making the
    tests async so they reuse pytest-asyncio's managed event loop instead of spawning
    a nested ``asyncio.run()`` loop whose IOCP teardown can race with the pipe server.

    This fixture acts as a secondary safeguard, ensuring any child processes that
    escaped the shutdown sequence are reaped before the next test runs.
    """
    yield  # let the test run
    deadline = time.time() + 5.0
    while time.time() < deadline:
        children = multiprocessing.active_children()
        if not children:
            break
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=1.0)
            else:
                child.join(timeout=0.1)
        time.sleep(0.1)
    else:
        remaining = [c for c in multiprocessing.active_children() if c.is_alive()]
        if remaining:
            logger.warning(f"Could not reap {len(remaining)} child process(es) after 5 s")


@pytest.fixture()
def dry_run_bridge_data() -> Mock:
    """Bridge data with all three dry-run flags enabled."""
    return make_mock_bridge_data(
        dry_run_skip_inference=True,
        dry_run_skip_safety=True,
        dry_run_skip_api=True,
        dry_run_inference_delay=0.0,
    )


@pytest.fixture()
def dry_run_process_manager(dry_run_bridge_data: Mock) -> object:
    """A process manager wired for full dry-run."""
    return make_testable_process_manager(bridge_data=dry_run_bridge_data)
