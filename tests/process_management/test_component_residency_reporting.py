"""Child-side tests for component-residency reporting and targeted eviction.

A GPU-bearing child attaches its component-cache residency to each memory report and evicts named entries on
request, both best-effort and never faulting. The real VAE lane runs ML-free under ``dry_run``, so it stands
in for any cache-bearing child here (its component-cache access lives in the shared ``HordeProcess`` base).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import (
    HeldComponentSnapshot,
    HordeControlFlag,
    HordeControlMessage,
    HordeEvictComponentsControlMessage,
    HordeProcessMemoryMessage,
)
from horde_worker_regen.process_management.workers.vae_lane_process import HordeVaeLaneProcess


class _FakeQueue:
    """A minimal stand-in for the process message queue that records what the lane sends."""

    def __init__(self) -> None:
        self.messages: list[object] = []

    def put(self, message: object) -> None:
        """Record a message the lane sent to the parent."""
        self.messages.append(message)


def _make_dry_run_lane(queue: _FakeQueue) -> HordeVaeLaneProcess:
    return HordeVaeLaneProcess(
        process_id=7,
        process_message_queue=queue,  # type: ignore[arg-type]
        pipe_connection=Mock(),
        disk_lock=Mock(),
        process_launch_identifier=3,
        dry_run=True,
    )


def _patch_shared_manager(monkeypatch: pytest.MonkeyPatch, cache: object) -> None:
    """Point hordelib's ``SharedModelManager.manager._models_in_ram`` at ``cache``."""
    fake_manager = SimpleNamespace(manager=SimpleNamespace(_models_in_ram=cache))
    monkeypatch.setattr("hordelib.api.SharedModelManager", fake_manager)


def _sole_memory_message(queue: _FakeQueue) -> HordeProcessMemoryMessage:
    memory = [message for message in queue.messages if isinstance(message, HordeProcessMemoryMessage)]
    assert len(memory) == 1
    return memory[0]


class TestHeldComponentReporting:
    """A cache-bearing child attaches its residency; a child without a cache reports None."""

    def test_report_carries_converted_held_components(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The memory report carries each cache entry converted to the worker-side snapshot type."""
        held = [
            SimpleNamespace(kind="checkpoint", identity="ModelA", approx_ram_mb=7000.0),
            SimpleNamespace(kind="vae", identity="vae@abc", approx_ram_mb=512.0),
        ]
        cache = SimpleNamespace(held_report=lambda: held)
        _patch_shared_manager(monkeypatch, cache)

        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        # A real (non-dry-run) cache-bearing lane sets this; force it on so the dry-run stand-in reports.
        lane._reports_held_components = True
        queue.messages.clear()

        lane.send_memory_report_message(include_vram=False)

        message = _sole_memory_message(queue)
        assert message.held_components == [
            HeldComponentSnapshot(kind="checkpoint", identity="ModelA", approx_ram_mb=7000.0),
            HeldComponentSnapshot(kind="vae", identity="vae@abc", approx_ram_mb=512.0),
        ]

    def test_dry_run_lane_reports_none(self) -> None:
        """A dry-run lane has no loaded backend, so it reports None without importing hordelib."""
        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        assert lane._reports_held_components is False
        queue.messages.clear()

        lane.send_memory_report_message(include_vram=False)

        assert _sole_memory_message(queue).held_components is None

    def test_read_failure_reports_none_and_logs_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A residency read that raises degrades to None rather than disturbing the report."""

        def _boom() -> list[object]:
            raise RuntimeError("cache unavailable")

        cache = SimpleNamespace(held_report=_boom)
        _patch_shared_manager(monkeypatch, cache)

        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        lane._reports_held_components = True
        queue.messages.clear()

        lane.send_memory_report_message(include_vram=False)

        assert _sole_memory_message(queue).held_components is None
        assert lane._held_components_read_failed_logged is True


class TestEvictHandler:
    """The evict handler drops named entries, tolerates unknown identities, and never faults."""

    def test_evict_forwards_identities_to_the_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cache-bearing lane forwards the requested identities to the cache's eviction."""
        cache = Mock()
        cache.evict_identities.return_value = 2
        _patch_shared_manager(monkeypatch, cache)

        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        lane._reports_held_components = True

        lane._receive_and_handle_control_message(
            HordeEvictComponentsControlMessage(identities=["ModelA", "ModelC"]),
        )

        cache.evict_identities.assert_called_once_with(["ModelA", "ModelC"])

    def test_evict_unknown_identities_does_not_fault(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Evicting an identity the cache does not hold is a no-op (zero evicted), never an error."""
        cache = Mock()
        cache.evict_identities.return_value = 0
        _patch_shared_manager(monkeypatch, cache)

        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        lane._reports_held_components = True

        # No exception raised even though nothing matched.
        lane._receive_and_handle_control_message(
            HordeEvictComponentsControlMessage(identities=["does-not-exist"]),
        )
        cache.evict_identities.assert_called_once_with(["does-not-exist"])

    def test_evict_without_a_cache_is_a_noop(self) -> None:
        """A dry-run lane (no backend) ignores the request without importing hordelib."""
        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        assert lane._reports_held_components is False

        # Would raise if it tried to reach a cache; the no-cache guard keeps it silent.
        lane.evict_held_components(["ModelA"])


class TestOldParentDispatchContract:
    """A control flag outside a lane's dispatch contract is dropped loudly, keeping the lane alive."""

    def test_unsupported_control_flag_is_reported(self) -> None:
        """An unrelated flag (a routing error) raises the dispatch-contract error rather than acting."""
        from horde_worker_regen.process_management.ipc.messages import UnsupportedControlMessageError

        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)

        with pytest.raises(UnsupportedControlMessageError):
            lane._receive_and_handle_control_message(
                HordeControlMessage(control_flag=HordeControlFlag.PRELOAD_MODEL),
            )
