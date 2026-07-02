"""The test harness must never reach across to real OS processes or the shared on-disk registry.

Building a real ``HordeWorkerProcessManager`` (as ``make_testable_process_manager`` does) runs the worker's
startup, which otherwise reaps orphaned child pids recorded in ``.horde_worker_regen/owned_pids.json`` and
kills any still-alive match. Run in the same working directory as a live worker, that would terminate the
worker's inference/safety children. These tests pin the two guards that prevent it: the ``AI_HORDE_TESTING``
flag (which stops the registry being built at all) and the session fixture that redirects the registry's
state dir to a throwaway path.
"""

from __future__ import annotations

import os
from pathlib import Path

from horde_worker_regen.process_management.lifecycle import owned_process_registry
from tests.process_management.conftest import make_testable_process_manager


def test_testing_flag_is_set_during_the_suite() -> None:
    """The suite marks itself as under test so the manager's process-reaping guard is active."""
    assert os.environ.get("AI_HORDE_TESTING"), "AI_HORDE_TESTING must be set so the worker skips orphan reaping"


def test_testable_manager_does_not_build_a_reaping_registry() -> None:
    """A manager built in tests never constructs the owned-pid registry, so it can reap nothing."""
    process_manager = make_testable_process_manager()

    assert process_manager._owned_registry is None


def test_owned_registry_state_dir_is_isolated_from_the_repo() -> None:
    """The registry's state dir is redirected off the real ``.horde_worker_regen`` in the working dir."""
    isolated = owned_process_registry.default_app_state_dir()

    assert isolated != Path.cwd() / ".horde_worker_regen"
    # A default-constructed registry honours the redirected dir, so its file cannot be the live worker's.
    assert owned_process_registry.OwnedProcessRegistry().path.parent == isolated
