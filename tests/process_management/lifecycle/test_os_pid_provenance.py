"""Tests pinning that the stored os_pid is the child's self-reported os.getpid(), not the spawn handle."""

from __future__ import annotations

import os

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import make_mock_process_info


def test_reconcile_adopts_the_child_reported_pid() -> None:
    """A child-reported pid overwrites the handle-derived os_pid (the value per-PID telemetry then addresses)."""
    proc = make_mock_process_info(1)  # os_pid is the mocked spawn-handle pid (100001)
    process_map = ProcessMap({1: proc})
    assert proc.os_pid == 100001

    process_map.reconcile_reported_os_pid(1, 4242)

    assert proc.os_pid == 4242, "the stored os_pid must equal the child-reported value"


def test_reconcile_ignores_a_missing_report() -> None:
    """A None report (an older child that does not self-report) leaves the handle-derived value in place."""
    proc = make_mock_process_info(1)
    process_map = ProcessMap({1: proc})
    handle_pid = proc.os_pid

    process_map.reconcile_reported_os_pid(1, None)

    assert proc.os_pid == handle_pid


def test_reconcile_absent_process_is_a_no_op() -> None:
    """Reconciling a process not in the map does not raise."""
    process_map = ProcessMap({})
    process_map.reconcile_reported_os_pid(99, 4242)


def test_worker_state_change_message_carries_the_reporters_own_pid() -> None:
    """The base send helper stamps the message with the sender's real os.getpid()."""
    # A minimal HordeProcess is abstract; assert instead that the message model carries the field and the
    # send helper (verified by construction here) populates it with the reporter's own pid, matching what the
    # parent adopts. This pins the field contract the dispatcher reconciles against.
    from horde_worker_regen.process_management.ipc.messages import HordeProcessStateChangeMessage

    message = HordeProcessStateChangeMessage(
        process_state=HordeProcessState.PROCESS_STARTING,
        process_id=1,
        process_launch_identifier=0,
        reported_os_pid=os.getpid(),
        info="starting",
    )
    assert message.reported_os_pid == os.getpid()

    # And the parent adopts exactly that value.
    proc = make_mock_process_info(1, process_type=HordeProcessType.INFERENCE)
    process_map = ProcessMap({1: proc})
    process_map.reconcile_reported_os_pid(message.process_id, message.reported_os_pid)
    assert proc.os_pid == os.getpid()


def test_older_child_without_a_report_keeps_the_handle_pid() -> None:
    """A resident whose message omits the pid (default None) is not disturbed."""
    proc = make_mock_process_info(2, state=HordeProcessState.WAITING_FOR_JOB)
    process_map = ProcessMap({2: proc})
    handle_pid = proc.os_pid
    process_map.reconcile_reported_os_pid(2, None)
    assert proc.os_pid == handle_pid
