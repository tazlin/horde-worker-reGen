"""Per-lane handler tests for RELEASE_ALLOCATOR_CACHE (Stage 0 actuation scaffolding).

Releasing the allocator cache must reclaim the torch caching allocator's free blocks WITHOUT unloading
any model: the observable contract is that the child sends a fresh memory report and emits no
model-unload state change. The component lane, which previously handled no unload/cache messages at all,
additionally gains a real model-unload handler; these tests pin both behaviours on the dry-run lanes
(their ML paths are rig-only, so a dry-run instance exercises the message contract without a backend).
"""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeModelStateChangeMessage,
    HordeProcessMemoryMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
)
from horde_worker_regen.process_management.workers.component_lane_process import HordeComponentLaneProcess
from horde_worker_regen.process_management.workers.post_process_process import HordePostProcessProcess
from horde_worker_regen.process_management.workers.vae_lane_process import HordeVaeLaneProcess


class _FakeQueue:
    """A minimal stand-in for the process message queue that records what a lane sends."""

    def __init__(self) -> None:
        self.messages: list[object] = []

    def put(self, message: object) -> None:
        """Record a message the lane sent to the parent."""
        self.messages.append(message)


def _release_cache_message() -> HordeControlMessage:
    return HordeControlMessage(control_flag=HordeControlFlag.RELEASE_ALLOCATOR_CACHE)


def _unload_vram_message() -> HordeControlMessage:
    return HordeControlMessage(control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM)


def _memory_reports(queue: _FakeQueue) -> list[HordeProcessMemoryMessage]:
    return [m for m in queue.messages if isinstance(m, HordeProcessMemoryMessage)]


def _unload_state_changes(queue: _FakeQueue) -> list[HordeProcessStateChangeMessage]:
    unload_states = {
        HordeProcessState.UNLOADED_MODEL_FROM_VRAM,
        HordeProcessState.UNLOADED_MODEL_FROM_RAM,
    }
    return [
        m for m in queue.messages if isinstance(m, HordeProcessStateChangeMessage) and m.process_state in unload_states
    ]


def _make_vae_lane(queue: _FakeQueue) -> HordeVaeLaneProcess:
    return HordeVaeLaneProcess(
        process_id=1,
        process_message_queue=queue,  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
        dry_run=True,
    )


def _make_post_process(queue: _FakeQueue) -> HordePostProcessProcess:
    return HordePostProcessProcess(
        process_id=2,
        process_message_queue=queue,  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
        dry_run_skip_post_processing=True,
    )


def _make_component_lane(queue: _FakeQueue) -> HordeComponentLaneProcess:
    return HordeComponentLaneProcess(
        process_id=3,
        process_message_queue=queue,  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=0,
        dry_run=True,
    )


class TestCacheReleaseEmitsReportWithoutUnload:
    """Each GPU-holding lane answers RELEASE_ALLOCATOR_CACHE with a memory report and no model unload."""

    def test_vae_lane(self) -> None:
        """The VAE lane reports memory and does not signal any model unload."""
        queue = _FakeQueue()
        lane = _make_vae_lane(queue)
        before = len(_memory_reports(queue))

        lane._receive_and_handle_control_message(_release_cache_message())

        assert len(_memory_reports(queue)) == before + 1
        assert _unload_state_changes(queue) == []

    def test_post_process_lane(self) -> None:
        """The post-processing lane reports memory and does not signal any model unload."""
        queue = _FakeQueue()
        lane = _make_post_process(queue)
        before = len(_memory_reports(queue))

        lane._receive_and_handle_control_message(_release_cache_message())

        assert len(_memory_reports(queue)) == before + 1
        assert _unload_state_changes(queue) == []

    def test_component_lane(self) -> None:
        """The text-encode service reports memory and does not signal any model unload."""
        queue = _FakeQueue()
        lane = _make_component_lane(queue)
        before = len(_memory_reports(queue))

        lane._receive_and_handle_control_message(_release_cache_message())

        assert len(_memory_reports(queue)) == before + 1
        assert _unload_state_changes(queue) == []


class TestComponentLaneUnloadHandler:
    """The component lane gains a real model-unload handler that reports its cleared state."""

    def test_unload_from_vram_reports_state_and_memory(self) -> None:
        """UNLOAD_MODELS_FROM_VRAM emits the unloaded-from-VRAM state change plus a memory report."""
        queue = _FakeQueue()
        lane = _make_component_lane(queue)
        queue.messages.clear()

        lane._receive_and_handle_control_message(_unload_vram_message())

        unload_states = [m.process_state for m in _unload_state_changes(queue)]
        assert HordeProcessState.UNLOADED_MODEL_FROM_VRAM in unload_states
        assert len(_memory_reports(queue)) == 1

    def test_unload_does_not_report_a_named_model_state_change(self) -> None:
        """The service holds no whole-model bookkeeping, so it never emits a per-model state change."""
        queue = _FakeQueue()
        lane = _make_component_lane(queue)
        queue.messages.clear()

        lane._receive_and_handle_control_message(_unload_vram_message())

        assert [m for m in queue.messages if isinstance(m, HordeModelStateChangeMessage)] == []
