"""Parent-side sender for RELEASE_ALLOCATOR_CACHE and the release-target memory it leaves.

The scheduler's ``release_allocator_cache(process_id)`` mirrors the unload senders' safe-send path. These tests
confirm it delivers the flag to the addressed process, that the fake inference process answers the flag with a
memory report, and that a lane whose release returned nothing stops being a release target until it loads or
unloads a model.
"""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessMemoryMessage,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.reclaim_ladder import teardown_verification_settle_seconds
from horde_worker_regen.process_management.scheduling.inference_scheduler import _RELEASE_CACHE_MIN_RECLAIMABLE_MB
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


class _Clock:
    """A scheduler clock the rows advance by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _past_the_settle(clock: _Clock) -> None:
    clock.now += teardown_verification_settle_seconds()


class TestReleaseTargetMemory:
    """A lane whose release left its reservation in place is not asked again until its tenancy changes.

    Every reclaim step is graded on what it freed, after the teardown settle the ladder gives a lane rung. Inside
    the settle the release is outstanding: the lane is not asked again and the room still counts what the release
    may return. A lane the child cannot release (live references still pin the blocks) would otherwise be
    re-targeted every cycle, and each send would count as an acting rung.
    """

    def test_cache_holder_is_not_targeted_again_until_an_unload(self) -> None:
        """A cache-only lane is withheld through the settle, refused after it, and restored by an unload."""
        process_info = make_mock_process_info(7, model_name=None)
        process_info.process_reserved_mb = 4000
        process_info.process_allocated_mb = 500
        process_map = ProcessMap({7: process_info})
        clock = _Clock()
        scheduler = _make_inference_scheduler(process_map=process_map, clock=clock)

        assert 7 in scheduler._gpu_process_activity_ids(None)[0]
        assert scheduler.release_allocator_cache(7) is True
        assert 7 not in scheduler._gpu_process_activity_ids(None)[0]
        assert process_info.release_cache_refusal_logged is False

        _past_the_settle(clock)
        assert 7 not in scheduler._gpu_process_activity_ids(None)[0]
        assert process_info.release_cache_refusal_logged is True

        process_map.on_model_vram_clear(7)
        assert 7 in scheduler._gpu_process_activity_ids(None)[0]

    def test_a_release_that_returned_its_threshold_keeps_the_lane_a_target(self) -> None:
        """A reservation that fell by the release threshold is graded as a working release without the settle."""
        process_info = make_mock_process_info(7, model_name=None)
        process_info.process_reserved_mb = 4000
        process_info.process_allocated_mb = 500
        scheduler = _make_inference_scheduler(process_map=ProcessMap({7: process_info}), clock=_Clock())

        assert scheduler.release_allocator_cache(7) is True
        process_info.process_reserved_mb = 4000 - int(_RELEASE_CACHE_MIN_RECLAIMABLE_MB)

        assert 7 in scheduler._gpu_process_activity_ids(None)[0]
        assert process_info.release_cache_requested_at_reserved_mb is None

    def test_lane_with_model_in_ram_returns_nothing_after_the_settle_until_a_load(self) -> None:
        """The room counts the lane's components through the settle, nothing after it, and again after a load."""
        process_info = make_mock_process_info(7, model_name="stable_diffusion")
        process_info.process_reserved_mb = 1800
        process_info.process_allocated_mb = 1800
        process_map = ProcessMap({7: process_info})
        clock = _Clock()
        scheduler = _make_inference_scheduler(process_map=process_map, clock=clock)

        targets = scheduler._gpu_process_activity_ids(None)[0]
        assert 7 in targets
        assert scheduler._held_idle_returns_mb(None, targets) == {7: 1800.0}

        assert scheduler.release_allocator_cache(7) is True
        targets = scheduler._gpu_process_activity_ids(None)[0]
        assert 7 not in targets
        assert scheduler._held_idle_returns_mb(None, targets) == {7: 1800.0}

        _past_the_settle(clock)
        targets = scheduler._gpu_process_activity_ids(None)[0]
        assert 7 not in targets
        assert scheduler._held_idle_returns_mb(None, targets) == {7: 0.0}

        process_map.on_model_load_state_change(7, horde_model_name="stable_diffusion")
        assert 7 in scheduler._gpu_process_activity_ids(None)[0]

    def test_lane_whose_unload_the_device_refused_returns_nothing_until_it_materialises(self) -> None:
        """A refused unload leaves the room nothing permitted on the lane until it next materialises weights."""
        process_info = make_mock_process_info(7, model_name="stable_diffusion")
        process_info.retained_resident_model = "stable_diffusion"
        process_info.process_reserved_mb = 6000
        process_info.process_allocated_mb = 6000
        process_map = ProcessMap({7: process_info})
        scheduler = _make_inference_scheduler(process_map=process_map, clock=_Clock())

        assert scheduler._held_idle_returns_mb(None, frozenset()) == {7: 6000.0}

        process_map.on_vram_unload_refused(7)
        assert scheduler._held_idle_returns_mb(None, frozenset()) == {7: 0.0}

        process_map.note_vram_materialized(7)
        assert scheduler._held_idle_returns_mb(None, frozenset()) == {7: 6000.0}
