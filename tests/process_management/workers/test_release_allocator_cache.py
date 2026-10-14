"""Per-lane handler tests for RELEASE_ALLOCATOR_CACHE.

Releasing the allocator cache must reclaim the torch caching allocator's free blocks WITHOUT unloading
any model: the observable contract is that the child sends a fresh memory report and emits no
model-unload state change. The component lane also has a model-unload handler. The service lanes are pinned
on their dry-run instances (their ML paths are rig-only, so a dry-run instance exercises the message contract
without a backend). The inference child is pinned on its actuator alone. It moves its cached components'
device weights to host RAM before the cache release only while it idles with its model in host RAM, and reports
after both.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeModelStateChangeMessage,
    HordeProcessMemoryMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
    ModelLoadState,
)
from horde_worker_regen.process_management.workers.component_lane_process import HordeComponentLaneProcess
from horde_worker_regen.process_management.workers.inference_process import HordeInferenceProcess
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


def _inference_child(
    calls: list[str],
    *,
    busy: bool,
    load_state: ModelLoadState | None,
) -> HordeInferenceProcess:
    """An inference child wired only for the cache release, recording each step it takes in order.

    The cache empty and the memory report read the device, so each records a marker instead. The component
    release stays real down to hordelib's module function, which records too.
    """
    child = object.__new__(HordeInferenceProcess)
    child._dry_run_skip_inference = False
    child._reports_held_components = True
    child._is_busy = busy
    child._active_model_name = "sdxl-checkpoint" if load_state is not None else None
    child._active_model_load_state = load_state
    child.clear_gc_and_torch_cache = Mock(side_effect=lambda: calls.append("empty"))  # type: ignore[method-assign]

    def _record_report(*_args: object, **_kwargs: object) -> bool:
        calls.append("report")
        return True

    child.send_memory_report_message = Mock(side_effect=_record_report)  # type: ignore[method-assign]
    return child


@pytest.fixture
def component_release_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every call into hordelib's component device-weight release on a shared list."""
    from hordelib.execution import component_cache

    calls: list[str] = []

    def _record_release() -> int:
        calls.append("release")
        return 2

    monkeypatch.setattr(component_cache, "release_component_device_weights", _record_release)
    return calls


class TestInferenceCacheRelease:
    """The inference child moves its components' device weights only while it idles with its model in host RAM."""

    def test_idle_with_model_in_ram_releases_then_empties_then_reports(
        self,
        component_release_calls: list[str],
    ) -> None:
        """The component weights go first, so the cache empty returns their blocks and the report shows it."""
        child = _inference_child(component_release_calls, busy=False, load_state=ModelLoadState.LOADED_IN_RAM)

        child.release_allocator_cache()

        assert component_release_calls == ["release", "empty", "report"]

    def test_busy_lane_only_empties_and_reports(self, component_release_calls: list[str]) -> None:
        """A lane serving a job keeps its components on the device."""
        child = _inference_child(component_release_calls, busy=True, load_state=ModelLoadState.LOADED_IN_RAM)

        child.release_allocator_cache()

        assert component_release_calls == ["empty", "report"]

    def test_model_on_the_device_only_empties_and_reports(self, component_release_calls: list[str]) -> None:
        """A lane whose weights are on the device is the eviction rung's, and keeps its components there."""
        child = _inference_child(component_release_calls, busy=False, load_state=ModelLoadState.LOADED_IN_VRAM)

        child.release_allocator_cache()

        assert component_release_calls == ["empty", "report"]
