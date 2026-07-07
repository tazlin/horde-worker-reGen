"""Liveness tests for control-message routing across process types.

A control flag delivered outside its receiver's dispatch contract is a parent-side routing error, not a
child-side failure: the child must drop it loudly and stay alive. The terminal-on-exception rule in the
base receive loop exists for a *supported* handler failing mid-action (state unknown); applying it to a
routing mismatch converts a sender bug into a crash-restart loop for a healthy child. The safety process
is the canonical victim: it is a GPU committed-ledger tenant, so ledger-wide fan-outs reach it, and it
historically had no handler for the cache-release flag those fan-outs carry.
"""

from __future__ import annotations

import queue
from typing import override
from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessMemoryMessage,
    HordeProcessState,
    UnsupportedControlMessageError,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import (
    ALLOCATOR_CACHE_CAPABLE_PROCESS_TYPES,
    HordeProcess,
    HordeProcessType,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.workers.safety_process import HordeSafetyProcess
from tests.process_management.conftest import make_mock_process_info


class _RoutingErrorProcess(HordeProcess):
    """A stub whose dispatch rejects every message as outside its contract."""

    @override
    def cleanup_for_exit(self) -> None:
        return

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        raise UnsupportedControlMessageError(f"stub does not handle {message.control_flag}")


class _FailingHandlerProcess(HordeProcess):
    """A stub whose (supported) handler fails mid-action."""

    @override
    def cleanup_for_exit(self) -> None:
        return

    @override
    def _receive_and_handle_control_message(self, message: HordeControlMessage) -> None:
        raise RuntimeError("handler blew up mid-action")


def _make_process(cls: type[HordeProcess]) -> HordeProcess:
    return cls(
        process_id=3,
        process_message_queue=Mock(spec=queue.Queue),
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
    )


class TestBaseLoopRoutingFence:
    """The base receive loop drops routing mismatches and stays terminal on real handler failures."""

    def test_unsupported_message_is_dropped_and_the_process_lives(self) -> None:
        """An unsupported control message is consumed without ending the process."""
        proc = _make_process(_RoutingErrorProcess)
        proc._control_inbox.put(HordeControlMessage(control_flag=HordeControlFlag.RELEASE_ALLOCATOR_CACHE))

        proc.receive_and_handle_control_messages()

        assert proc._end_process is False
        assert proc._control_inbox.qsize() == 0

    def test_repeated_unsupported_messages_never_accumulate_into_an_exit(self) -> None:
        """A sender bug that keeps fanning the flag out must not eventually kill the child."""
        proc = _make_process(_RoutingErrorProcess)
        for _ in range(5):
            proc._control_inbox.put(HordeControlMessage(control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM))

        proc.receive_and_handle_control_messages()

        assert proc._end_process is False

    def test_supported_handler_failure_remains_terminal(self) -> None:
        """An exception escaping a supported handler still ends the process (state unknown)."""
        proc = _make_process(_FailingHandlerProcess)
        proc._control_inbox.put(HordeControlMessage(control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM))

        proc.receive_and_handle_control_messages()

        assert proc._end_process is True


def _make_safety_process(message_queue: Mock) -> HordeSafetyProcess:
    return HordeSafetyProcess(
        process_id=0,
        process_message_queue=message_queue,
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
        dry_run_skip_safety=True,
    )


class TestSafetyProcessControlRouting:
    """The safety lane handles the ledger fan-out's cache-release flag and survives foreign flags."""

    def test_release_allocator_cache_is_handled_with_a_memory_report(self) -> None:
        """The safety process answers RELEASE_ALLOCATOR_CACHE with a fresh memory report, no exception."""

        class _FakeQueue:
            def __init__(self) -> None:
                self.messages: list[object] = []

            def put(self, message: object) -> None:
                self.messages.append(message)

        fake_queue = _FakeQueue()
        proc = _make_safety_process(fake_queue)  # type: ignore[arg-type]
        before = len([m for m in fake_queue.messages if isinstance(m, HordeProcessMemoryMessage)])

        proc._receive_and_handle_control_message(
            HordeControlMessage(control_flag=HordeControlFlag.RELEASE_ALLOCATOR_CACHE),
        )

        reports = [m for m in fake_queue.messages if isinstance(m, HordeProcessMemoryMessage)]
        assert len(reports) == before + 1

    def test_foreign_control_message_is_dropped_without_ending_the_process(self) -> None:
        """A control flag the safety lane does not implement is dropped by the base loop; the lane lives.

        This is the crash-loop class end to end: the parent fans a lane-wide flag out, the safety child
        has no handler, and the child must survive it rather than exiting on the first fan-out after
        every restart.
        """
        proc = _make_safety_process(Mock(spec=queue.Queue))
        proc._control_inbox.put(HordeControlMessage(control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_RAM))

        proc.receive_and_handle_control_messages()

        assert proc._end_process is False


class TestRecalibrationFanOutCapability:
    """The ledger recalibration fan-out targets only process types that implement the flag."""

    def test_capability_set_covers_every_gpu_ledger_tenant_type(self) -> None:
        """Every process type that can hold a GPU context is in the capability set; DOWNLOAD is not."""
        gpu_tenant_types = {
            HordeProcessType.INFERENCE,
            HordeProcessType.SAFETY,
            HordeProcessType.POST_PROCESS,
            HordeProcessType.COMPONENT,
            HordeProcessType.VAE_LANE,
        }
        assert gpu_tenant_types <= ALLOCATOR_CACHE_CAPABLE_PROCESS_TYPES
        assert HordeProcessType.DOWNLOAD not in ALLOCATOR_CACHE_CAPABLE_PROCESS_TYPES

    def test_fan_out_asks_capable_idle_tenants_and_skips_the_rest(self) -> None:
        """A hostile map (a routing-incapable process reporting a reservation) is never asked to release.

        The safety process, a first-class ledger tenant, IS asked; a DOWNLOAD-type process that somehow
        reports a reservation is skipped, whatever it claims to hold.
        """
        from horde_worker_regen.process_management.config.worker_state import WorkerState
        from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
        from horde_worker_regen.process_management.models.lru_cache import LRUCache
        from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
        from tests.process_management.conftest import (
            make_mock_bridge_data,
            make_test_model_metadata,
            make_test_runtime_config,
        )

        process_map = ProcessMap({})
        specs = [
            (0, HordeProcessType.SAFETY),
            (1, HordeProcessType.POST_PROCESS),
            (2, HordeProcessType.DOWNLOAD),
            (4, HordeProcessType.INFERENCE),
        ]
        for pid, process_type in specs:
            proc = make_mock_process_info(
                process_id=pid,
                model_name=None,
                state=HordeProcessState.WAITING_FOR_JOB,
                process_type=process_type,
            )
            proc.process_reserved_mb = 512.0
            process_map[pid] = proc

        scheduler = InferenceScheduler(
            state=WorkerState(),
            process_map=process_map,
            horde_model_map=HordeModelMap(root={}),
            job_tracker=JobTracker(),
            process_lifecycle=Mock(
                get_processes_with_model_for_queued_job=Mock(return_value=[]),
                is_model_load_quarantined=Mock(return_value=False),
            ),
            runtime_config=make_test_runtime_config(bridge_data=make_mock_bridge_data()),
            model_metadata=make_test_model_metadata(),
            max_concurrent_inference_processes=1,
            max_inference_processes=1,
            lru=LRUCache(2),
        )

        lanes_asked = scheduler.recalibrate_committed_ledger()

        assert lanes_asked == 3
        asked_pids = {pid for pid, process_type in specs if process_map[pid].pipe_connection.send.call_count > 0}
        assert asked_pids == {0, 1, 4}
