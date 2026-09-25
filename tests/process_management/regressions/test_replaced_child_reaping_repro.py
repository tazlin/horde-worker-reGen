"""Reproduction: a service-lane child replaced while still starting outlives its replacement.

The safety, post-process, utilities, component and VAE replacement machines end the old child only when it
counts as loaded, and a child in ``PROCESS_STARTING`` does not. The next step retires the map entry, which
forgets the child without touching its OS process. Every teardown path walks the process map, so a replaced
child that was still starting is never ended at all: it survives the parent, holding whatever it had loaded, and
each stuck-starting replacement adds one more.

Contracts:

1. Replacing a child of any of those lanes, in any state, ends its OS process before the map entry is retired:
   a starting child is terminated, a straggler is killed, and a child confirmed dead leaves the owned-PID
   registry with a ``PROCESS_ENDED`` ledger record.
2. A child that already exited is reaped without a signal or an error.
3. A child that survives the kill stays in the owned-PID registry, and the replacement still proceeds.
4. The shutdown reap kills whatever the owned-PID registry still holds outside the map before clearing it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from loguru import logger

from horde_worker_regen.process_management.ipc.action_ledger import LedgerEventType
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.owned_process_registry import OwnedProcessRegistry
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import (
    SAFETY_PROCESS_ID,
    ProcessLifecycleManager,
)
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager

_POST_PROCESS_ID = 7
"""Slot id given to the post-processing child in these scenarios."""

_UTILITIES_ID = 8
"""Slot id given to the image-utilities child in these scenarios."""

_COMPONENT_ID = 9
"""Slot id given to the component-lane child in these scenarios."""

_VAE_LANE_ID = 10
"""Slot id given to the VAE-lane child in these scenarios."""

_LANE_MACHINES: dict[HordeProcessType, tuple[str, str, str]] = {
    HordeProcessType.SAFETY: (
        "_initiate_safety_replacement",
        "_replace_all_safety_process",
        "start_safety_processes",
    ),
    HordeProcessType.POST_PROCESS: (
        "_initiate_post_process_replacement",
        "_replace_all_post_process_process",
        "start_post_process_processes",
    ),
    HordeProcessType.UTILITIES: (
        "_initiate_utilities_replacement",
        "_replace_all_utilities_process",
        "start_utilities_processes",
    ),
    HordeProcessType.COMPONENT: (
        "_initiate_component_replacement",
        "_replace_all_component_process",
        "start_component_processes",
    ),
    HordeProcessType.VAE_LANE: (
        "_initiate_vae_lane_replacement",
        "_replace_all_vae_lane_process",
        "start_vae_lane_processes",
    ),
}
"""Each lane's replacement flag setter, its end -> delete -> start machine, and its start hook."""

_REPLACEMENT_FLAGS: dict[HordeProcessType, str] = {
    HordeProcessType.SAFETY: "_safety_processes_should_be_replaced",
    HordeProcessType.POST_PROCESS: "_post_process_processes_should_be_replaced",
    HordeProcessType.UTILITIES: "_utilities_processes_should_be_replaced",
    HordeProcessType.COMPONENT: "_component_processes_should_be_replaced",
    HordeProcessType.VAE_LANE: "_vae_lane_processes_should_be_replaced",
}
"""The flag each lane's machine clears once it has started the replacement."""

_TIMEOUT_REPLACED_LANES = frozenset(
    {HordeProcessType.SAFETY, HordeProcessType.POST_PROCESS, HordeProcessType.UTILITIES}
)
"""Lanes whose startup timeout drives the replacement directly; the others are replaced by their crash reaper
or a recycle, which flag the lane the same way."""

_STARTUP_TIMEOUT_SECONDS = 60.0
"""The startup timeout a stuck child is judged against."""


class _ChildHandle:
    """A stand-in for a child's ``multiprocessing`` handle that dies only when the parent's signal lands."""

    def __init__(self, *, alive: bool, dies_on_terminate: bool = True, dies_on_kill: bool = True) -> None:
        self.alive = alive
        self.pid = 0
        self.exitcode: int | None = None if alive else 1
        self._dies_on_terminate = dies_on_terminate
        self._dies_on_kill = dies_on_kill
        self.terminate = Mock(side_effect=self._on_terminate)
        self.kill = Mock(side_effect=self._on_kill)
        self.join = Mock()

    def is_alive(self) -> bool:
        return self.alive

    def _on_terminate(self) -> None:
        if self._dies_on_terminate:
            self.alive = False

    def _on_kill(self) -> None:
        if self._dies_on_kill:
            self.alive = False


def _stuck_child(process_id: int, process_type: HordeProcessType, handle: _ChildHandle) -> HordeProcessInfo:
    """A child that has sat in ``PROCESS_STARTING`` far past its startup timeout."""
    child = make_mock_process_info(
        process_id,
        model_name=None,
        state=HordeProcessState.PROCESS_STARTING,
        process_type=process_type,
    )
    handle.pid = child.os_pid or 0
    child.mp_process = handle  # pyrefly: ignore[bad-assignment] - a handle double for the child
    stale = time.time() - _STARTUP_TIMEOUT_SECONDS * 10
    child.last_received_timestamp = stale
    child.last_heartbeat_timestamp = stale
    child.last_process_state_started_at = stale
    return child


def _lifecycle_with_registry(monkeypatch: pytest.MonkeyPatch) -> tuple[ProcessLifecycleManager, Mock]:
    """Build a lifecycle whose child starts are stubbed and whose owned-PID registry is observable."""
    lifecycle = make_testable_process_manager()._process_lifecycle
    registry = Mock(spec=OwnedProcessRegistry)
    registry.kill_all_owned.return_value = []
    lifecycle._owned_registry = registry
    for _initiate, _replace, start in _LANE_MACHINES.values():
        monkeypatch.setattr(lifecycle, start, Mock(return_value=True))
    return lifecycle, registry


def _replace_as_stuck_starting(lifecycle: ProcessLifecycleManager, child: HordeProcessInfo) -> None:
    """Flag the child's lane for replacement and drive its replacement machine to completion.

    A lane the startup timeout covers is timed out of its startup; the others are flagged as their crash reaper
    and recycle paths flag them.
    """
    lifecycle._process_map[child.process_id] = child
    initiate, replace_all, _start = _LANE_MACHINES[child.process_type]
    if child.process_type in _TIMEOUT_REPLACED_LANES:
        assert (
            lifecycle._check_and_replace_process(
                child,
                _STARTUP_TIMEOUT_SECONDS,
                HordeProcessState.PROCESS_STARTING,
                "seems to be stuck starting",
            )
            is True
        )
    else:
        getattr(lifecycle, initiate)()
    for _ in range(5):
        if not getattr(lifecycle, _REPLACEMENT_FLAGS[child.process_type]):
            return
        getattr(lifecycle, replace_all)()
    raise AssertionError("the replacement state machine never completed a rebuild cycle")


def _ended_records(lifecycle: ProcessLifecycleManager, child: HordeProcessInfo) -> list[object]:
    """Return the ledger's ``PROCESS_ENDED`` records for this launch of the child."""
    return [
        event
        for event in lifecycle.action_ledger.recent(limit=500)
        if event.event_type is LedgerEventType.PROCESS_ENDED
        and event.launch_identifier == child.process_launch_identifier
        and event.process_id == child.process_id
    ]


@pytest.fixture
def captured_errors() -> Iterator[list[str]]:
    """Every ERROR-and-above parent log message emitted during the test."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="ERROR")
    yield messages
    logger.remove(sink_id)


_LANES = pytest.mark.parametrize(
    ("process_id", "process_type"),
    [
        (SAFETY_PROCESS_ID, HordeProcessType.SAFETY),
        (_POST_PROCESS_ID, HordeProcessType.POST_PROCESS),
        (_UTILITIES_ID, HordeProcessType.UTILITIES),
        (_COMPONENT_ID, HordeProcessType.COMPONENT),
        (_VAE_LANE_ID, HordeProcessType.VAE_LANE),
    ],
    ids=["safety", "post-process", "utilities", "component", "vae-lane"],
)


class TestReplacedStartingChildIsEnded:
    """A child replaced while still starting is ended, not forgotten."""

    @_LANES
    def test_a_stuck_starting_child_is_terminated_and_leaves_the_registry(
        self,
        monkeypatch: pytest.MonkeyPatch,
        process_id: int,
        process_type: HordeProcessType,
    ) -> None:
        """The child cannot read END_PROCESS before its control loop runs, so it is terminated."""
        lifecycle, registry = _lifecycle_with_registry(monkeypatch)
        handle = _ChildHandle(alive=True)
        child = _stuck_child(process_id, process_type, handle)

        _replace_as_stuck_starting(lifecycle, child)

        handle.terminate.assert_called_once()
        handle.kill.assert_not_called()
        assert handle.alive is False, "the replaced child is still running after its replacement completed"
        registry.forget.assert_called_once_with(child.os_pid)
        assert len(_ended_records(lifecycle, child)) == 1
        assert child.end_intended is True
        assert lifecycle._process_map.get(process_id) is not child

    @_LANES
    def test_a_child_that_ignores_terminate_is_killed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        process_id: int,
        process_type: HordeProcessType,
    ) -> None:
        """A straggler past the end grace is killed, the discipline an inference slot's end uses."""
        lifecycle, registry = _lifecycle_with_registry(monkeypatch)
        handle = _ChildHandle(alive=True, dies_on_terminate=False)
        child = _stuck_child(process_id, process_type, handle)

        _replace_as_stuck_starting(lifecycle, child)

        handle.terminate.assert_called_once()
        handle.kill.assert_called_once()
        assert handle.alive is False
        registry.forget.assert_called_once_with(child.os_pid)

    @_LANES
    def test_an_already_exited_child_is_reaped_cleanly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        captured_errors: list[str],
        process_id: int,
        process_type: HordeProcessType,
    ) -> None:
        """A child that died on its own is joined and forgotten with no signal and no error."""
        lifecycle, registry = _lifecycle_with_registry(monkeypatch)
        handle = _ChildHandle(alive=False)
        child = _stuck_child(process_id, process_type, handle)

        _replace_as_stuck_starting(lifecycle, child)

        handle.terminate.assert_not_called()
        handle.kill.assert_not_called()
        handle.join.assert_called_with(timeout=0)
        registry.forget.assert_called_once_with(child.os_pid)
        assert len(_ended_records(lifecycle, child)) == 1
        assert [message for message in captured_errors if "still alive after a kill" in message] == []

    @_LANES
    def test_a_child_that_survives_the_kill_stays_owned_and_the_replacement_proceeds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        captured_errors: list[str],
        process_id: int,
        process_type: HordeProcessType,
    ) -> None:
        """The registry keeps the only record of a child the kill could not end, for the exit backstop."""
        lifecycle, registry = _lifecycle_with_registry(monkeypatch)
        handle = _ChildHandle(alive=True, dies_on_terminate=False, dies_on_kill=False)
        child = _stuck_child(process_id, process_type, handle)

        _replace_as_stuck_starting(lifecycle, child)

        registry.forget.assert_not_called()
        assert _ended_records(lifecycle, child) == []
        assert len([message for message in captured_errors if "still alive after a kill" in message]) == 1
        start = getattr(lifecycle, _LANE_MACHINES[process_type][2])
        start.assert_called_once()

    def test_a_loaded_safety_child_is_given_its_end_grace_before_a_kill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A child past startup was sent END_PROCESS, so it is joined rather than terminated."""
        lifecycle, registry = _lifecycle_with_registry(monkeypatch)
        handle = _ChildHandle(alive=True)
        handle.join.side_effect = lambda timeout=None: setattr(handle, "alive", False)
        child = _stuck_child(SAFETY_PROCESS_ID, HordeProcessType.SAFETY, handle)
        child.last_process_state = HordeProcessState.WAITING_FOR_JOB
        lifecycle._process_map[SAFETY_PROCESS_ID] = child
        lifecycle.safety_processes_should_be_replaced = True

        lifecycle._replace_all_safety_process()
        child.pipe_connection.send.assert_called()  # type: ignore[attr-defined]
        lifecycle._replace_all_safety_process()

        handle.terminate.assert_not_called()
        handle.kill.assert_not_called()
        registry.forget.assert_called_once_with(child.os_pid)
        lifecycle.start_safety_processes.assert_called_once()  # pyrefly: ignore[missing-attribute]


class TestShutdownReapKillsOwnedChildrenOutsideTheMap:
    """The final reap does not clear the registry's record of a child it did not reap."""

    def test_the_registry_is_killed_before_it_is_cleared(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A record left once the map is reaped is a child the map lost track of; it is killed, then cleared."""
        lifecycle, registry = _lifecycle_with_registry(monkeypatch)
        registry.kill_all_owned.return_value = [4242]

        assert lifecycle.reap_all_processes_for_shutdown() is True

        called = [call[0] for call in registry.mock_calls]
        assert "kill_all_owned" in called
        assert "clear" in called
        assert called.index("kill_all_owned") < called.index("clear")

    def test_the_hard_kill_also_kills_the_registry_before_clearing_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The terminal backstop covers the same children."""
        lifecycle, registry = _lifecycle_with_registry(monkeypatch)

        lifecycle._hard_kill_processes()

        called = [call[0] for call in registry.mock_calls]
        assert called.index("kill_all_owned") < called.index("clear")
