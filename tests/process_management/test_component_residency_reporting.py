"""Child-side tests for component-residency reporting and targeted restore or eviction.

A GPU-bearing child attaches its component-cache residency to each memory report, and restores or evicts
named entries on request, all best-effort and never faulting. Restoring is the cheaper of the two reclaim
actions: it hands back a component's device-memory claim while its pristine weights stay resident in host
RAM, so the report carries which entries hold residue worth restoring.

The real VAE lane runs ML-free under ``dry_run``, so it stands in for any cache-bearing child here; the
component-cache access it exercises lives in the shared ``HordeProcess`` base.
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
    HordeRestoreComponentsControlMessage,
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


def _patch_residency_api(monkeypatch: pytest.MonkeyPatch, **replacements: object) -> None:
    """Replace the declared ``hordelib.api`` residency helpers the child calls.

    The child reaches these by name rather than through the cache object, so the stub sits on the
    published surface instead of on ``SharedModelManager``'s private cache attribute.
    """
    for name, replacement in replacements.items():
        monkeypatch.setattr(f"hordelib.api.{name}", replacement, raising=False)


def _sole_memory_message(queue: _FakeQueue) -> HordeProcessMemoryMessage:
    memory = [message for message in queue.messages if isinstance(message, HordeProcessMemoryMessage)]
    assert len(memory) == 1
    return memory[0]


class TestHeldComponentReporting:
    """A cache-bearing child attaches its residency; a child without a cache reports None."""

    def test_report_carries_converted_held_components(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The memory report carries each cache entry converted to the worker-side snapshot type.

        The stubbed snapshots carry no residue indicator, standing in for an installed hordelib older than
        that field: the conversion reads it defensively, so the whole report survives the omission.
        """
        held = [
            SimpleNamespace(kind="checkpoint", identity="ModelA", approx_ram_mb=7000.0),
            SimpleNamespace(kind="vae", identity="vae@abc", approx_ram_mb=512.0),
        ]
        _patch_residency_api(monkeypatch, held_components=lambda: held)

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

    def test_report_carries_the_residue_indicator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Whether an entry holds patch residue rides the report, so the parent can pick restore over evict."""
        held = [
            SimpleNamespace(kind="checkpoint", identity="Patched", approx_ram_mb=7000.0, mutated=True),
            SimpleNamespace(kind="checkpoint", identity="Clean", approx_ram_mb=7000.0, mutated=False),
        ]
        _patch_residency_api(monkeypatch, held_components=lambda: held)

        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        lane._reports_held_components = True
        queue.messages.clear()

        lane.send_memory_report_message(include_vram=False)

        reported = _sole_memory_message(queue).held_components
        assert reported is not None
        assert {snapshot.identity: snapshot.mutated for snapshot in reported} == {"Patched": True, "Clean": False}

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

        _patch_residency_api(monkeypatch, held_components=_boom)

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
        evict = Mock(return_value=2)
        _patch_residency_api(monkeypatch, evict_components=evict)

        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        lane._reports_held_components = True

        lane._receive_and_handle_control_message(
            HordeEvictComponentsControlMessage(identities=["ModelA", "ModelC"]),
        )

        evict.assert_called_once_with(["ModelA", "ModelC"])

    def test_evict_unknown_identities_does_not_fault(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Evicting an identity the cache does not hold is a no-op (zero evicted), never an error."""
        evict = Mock(return_value=0)
        _patch_residency_api(monkeypatch, evict_components=evict)

        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        lane._reports_held_components = True

        # No exception raised even though nothing matched.
        lane._receive_and_handle_control_message(
            HordeEvictComponentsControlMessage(identities=["does-not-exist"]),
        )
        evict.assert_called_once_with(["does-not-exist"])

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


class TestRestoreHandler:
    """The restore handler returns named entries to their loaded state and never faults."""

    def test_restore_forwards_identities_to_the_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cache-bearing lane forwards the requested identities to the declared restore helper."""
        restore = Mock(return_value=2)
        _patch_residency_api(monkeypatch, restore_components=restore)

        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        lane._reports_held_components = True

        lane._receive_and_handle_control_message(
            HordeRestoreComponentsControlMessage(identities=["ModelA", "ModelC"]),
        )

        restore.assert_called_once_with(["ModelA", "ModelC"])

    def test_restore_unknown_identities_does_not_fault(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Restoring an identity the cache does not hold is a no-op, never an error."""
        restore = Mock(return_value=0)
        _patch_residency_api(monkeypatch, restore_components=restore)

        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        lane._reports_held_components = True

        lane._receive_and_handle_control_message(
            HordeRestoreComponentsControlMessage(identities=["does-not-exist"]),
        )
        restore.assert_called_once_with(["does-not-exist"])

    def test_restore_failure_does_not_fault_the_lane(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A restore that raises is swallowed: a reclaim action must never take down the lane."""

        def _boom(identities: list[str]) -> int:
            raise RuntimeError("restore unavailable")

        _patch_residency_api(monkeypatch, restore_components=_boom)

        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        lane._reports_held_components = True

        lane._receive_and_handle_control_message(
            HordeRestoreComponentsControlMessage(identities=["ModelA"]),
        )

    def test_restore_without_a_cache_is_a_noop(self) -> None:
        """A dry-run lane (no backend) ignores the request without importing hordelib."""
        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        assert lane._reports_held_components is False

        lane._receive_and_handle_control_message(
            HordeRestoreComponentsControlMessage(identities=["ModelA"]),
        )


class TestRestoreStatsReporting:
    """The memory report carries cumulative restore counters, so the parent can trend the contract."""

    def test_report_carries_restore_counters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Marked, restored, and released megabytes ride the report the parent already receives."""
        stats = SimpleNamespace(marked=12, restored=5, restored_bytes=3 * 1024 * 1024)
        _patch_residency_api(monkeypatch, held_components=list, component_restore_stats=lambda: stats)

        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        lane._reports_held_components = True
        queue.messages.clear()

        lane.send_memory_report_message(include_vram=False)

        message = _sole_memory_message(queue)
        assert message.components_marked == 12
        assert message.components_restored == 5
        assert message.components_restored_mb == pytest.approx(3.0)

    def test_a_stats_read_failure_leaves_the_memory_report_intact(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A counter read that raises must not cost the report the memory figures it was sent for."""

        def _boom() -> object:
            raise RuntimeError("stats unavailable")

        _patch_residency_api(monkeypatch, held_components=list, component_restore_stats=_boom)

        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        lane._reports_held_components = True
        queue.messages.clear()

        lane.send_memory_report_message(include_vram=False)

        message = _sole_memory_message(queue)
        assert message.components_marked is None
        assert message.ram_usage_bytes is not None

    def test_a_lane_without_a_cache_reports_no_counters(self) -> None:
        """A dry-run lane never imports hordelib, so the counters stay None rather than zero."""
        queue = _FakeQueue()
        lane = _make_dry_run_lane(queue)
        queue.messages.clear()

        lane.send_memory_report_message(include_vram=False)

        message = _sole_memory_message(queue)
        assert message.components_marked is None
        assert message.components_restored is None
        assert message.components_restored_mb is None
