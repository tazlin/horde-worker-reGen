"""Parent-side sender for RELEASE_ALLOCATOR_CACHE (Stage 0 actuation scaffolding).

The scheduler exposes a typed ``release_allocator_cache(process_id)`` that mirrors the unload senders'
safe-send path. Nothing dispatches it in production yet; these tests confirm it delivers the flag to the
addressed process and that the fake inference process answers the flag with a memory report (the harness
seam through which the future arbiter's actuation can be driven).
"""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessMemoryMessage,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import make_mock_process_info
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def test_sender_delivers_the_flag_to_the_addressed_process() -> None:
    """release_allocator_cache sends exactly the RELEASE_ALLOCATOR_CACHE flag to that process's pipe."""
    process_info = make_mock_process_info(7)
    scheduler = _make_inference_scheduler(process_map=ProcessMap({7: process_info}))

    assert scheduler.release_allocator_cache(7) is True

    sent = [call.args[0] for call in process_info.pipe_connection.send.call_args_list]
    assert any(
        isinstance(m, HordeControlMessage) and m.control_flag is HordeControlFlag.RELEASE_ALLOCATOR_CACHE for m in sent
    )


def test_sender_reports_failure_for_an_absent_process() -> None:
    """Addressing a process not in the map returns False rather than raising."""
    scheduler = _make_inference_scheduler(process_map=ProcessMap({}))
    assert scheduler.release_allocator_cache(999) is False


def test_fake_inference_process_answers_the_flag_with_a_memory_report() -> None:
    """The fake inference process handles the flag by emitting a fresh memory report, no model unload."""
    from multiprocessing import Semaphore
    from unittest.mock import Mock

    from horde_worker_regen.process_management.ipc.messages import HordeModelStateChangeMessage
    from horde_worker_regen.process_management.simulation.fake_worker_processes import FakeInferenceProcess

    class _FakeQueue:
        def __init__(self) -> None:
            self.messages: list[object] = []

        def put(self, message: object) -> None:
            self.messages.append(message)

    queue = _FakeQueue()
    fake = FakeInferenceProcess(
        process_id=1,
        process_message_queue=queue,  # type: ignore[arg-type]
        pipe_connection=Mock(),
        inference_semaphore=Semaphore(1),
        disk_lock=Mock(),
        process_launch_identifier=0,
    )
    queue.messages.clear()

    fake._receive_and_handle_control_message(
        HordeControlMessage(control_flag=HordeControlFlag.RELEASE_ALLOCATOR_CACHE),
    )

    assert any(isinstance(m, HordeProcessMemoryMessage) for m in queue.messages)
    assert [m for m in queue.messages if isinstance(m, HordeModelStateChangeMessage)] == []
