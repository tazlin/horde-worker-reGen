"""Manager-level tests for the fixed-pool control-loop tick.

These verify the wiring between the manager and the pure pool engine: an enabled pool ticks against live
demand and ledgers the seat transitions it produces, while a disabled pool ticks are a no-op.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

from horde_worker_regen.bridge_data.data_model import ModelPoolConfig
from horde_worker_regen.process_management.ipc.action_ledger import LedgerEventType
from horde_worker_regen.process_management.models.download_coordinator import ModelDownloadCoordinator
from horde_worker_regen.process_management.models.model_availability import ModelAvailability
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.scheduling.model_demand_poller import DemandSnapshot, ModelDemandRecord
from tests.process_management.conftest import make_mock_bridge_data, make_testable_process_manager

_BYTES_PER_GIGABYTE = 1024**3


def _seed_demand(manager: HordeWorkerProcessManager, model_name: str) -> None:
    """Inject a fresh, non-stale demand snapshot carrying one queued model into the manager's poller."""
    _seed_demand_multi(manager, {model_name: 100.0})


def _seed_demand_multi(manager: HordeWorkerProcessManager, queued_by_model: dict[str, float]) -> None:
    """Inject a fresh, non-stale demand snapshot carrying several queued models into the manager's poller."""
    snapshot = DemandSnapshot(
        records={
            model_name: ModelDemandRecord(
                queued=queued,
                jobs=5,
                eta_seconds=50,
                worker_count=1,
                performance=1.0,
            )
            for model_name, queued in queued_by_model.items()
        },
        fetched_at=time.monotonic(),
    )
    manager._model_demand_poller._latest = snapshot


def _set_availability(
    manager: HordeWorkerProcessManager,
    *,
    present: set[str],
    failed: tuple[str, ...] = (),
) -> None:
    """Publish an authoritative on-disk availability report into the manager's availability holder."""
    manager._model_availability.update(
        present=present,
        currently_downloading=None,
        pending=(),
        failed=failed,
        scan_complete=True,
    )


def _mock_record(size_bytes: int) -> Mock:
    """Build a stand-in image-model record carrying only the declared download size the budget reads."""
    record = Mock()
    record.declared_total_size_bytes = size_bytes
    return record


def _ledger_event_types(manager: HordeWorkerProcessManager) -> set[LedgerEventType]:
    return {event.event_type for event in manager._action_ledger.recent(limit=100)}


class TestModelPoolTick:
    """The control-loop pool tick advances seats and ledgers transitions when the pool is enabled."""

    def test_enabled_pool_seats_a_candidate_and_ledgers_it(self) -> None:
        """Test enabled pool seats a candidate and ledgers it."""
        manager = make_testable_process_manager(
            model_pool=ModelPoolConfig(enabled=True, seats=1, min_dwell_minutes=0.0),
        )
        _seed_demand(manager, "stable_diffusion")

        manager._maybe_tick_model_pool()

        assert "stable_diffusion" in manager._model_pool.active_seat_models()
        assert LedgerEventType.MODEL_POOL_SEATED in _ledger_event_types(manager)

    def test_disabled_pool_tick_is_a_no_op(self) -> None:
        """Test disabled pool tick is a no op."""
        manager = make_testable_process_manager(model_pool=ModelPoolConfig(enabled=False))
        _seed_demand(manager, "stable_diffusion")

        manager._maybe_tick_model_pool()

        assert manager._model_pool.active_seat_models() == frozenset()
        assert LedgerEventType.MODEL_POOL_SEATED not in _ledger_event_types(manager)


class TestModelPoolDownloadBudget:
    """The manager gates off-disk pool candidates by the operator's download budget before the engine sees them."""

    def test_zero_budget_filters_off_disk_candidates(self) -> None:
        """With a zero budget an off-disk candidate never reaches the engine, so only present models seat."""
        manager = make_testable_process_manager(
            model_pool=ModelPoolConfig(enabled=True, seats=2, min_dwell_minutes=0.0, download_budget_gb=0.0),
            image_models_to_load=["present_a", "offdisk_b"],
        )
        manager.stable_diffusion_reference = {
            "present_a": _mock_record(10 * _BYTES_PER_GIGABYTE),
            "offdisk_b": _mock_record(10 * _BYTES_PER_GIGABYTE),
        }
        manager._model_pool_cluster_sizes = {}
        _set_availability(manager, present={"present_a"})
        _seed_demand_multi(manager, {"present_a": 100.0, "offdisk_b": 200.0})

        manager._maybe_tick_model_pool()

        assert "present_a" in manager._model_pool.active_seat_models()
        assert "offdisk_b" not in manager._model_pool.active_seat_models()
        assert manager._model_pool.pending_download_models() == frozenset()

    def test_positive_budget_admits_off_disk_until_exhausted(self) -> None:
        """A positive budget admits off-disk candidates in rank order until the next one no longer fits."""
        manager = make_testable_process_manager(
            model_pool=ModelPoolConfig(enabled=True, seats=2, min_dwell_minutes=0.0, download_budget_gb=40.0),
            image_models_to_load=["offdisk_a", "offdisk_b"],
        )
        manager.stable_diffusion_reference = {
            "offdisk_a": _mock_record(30 * _BYTES_PER_GIGABYTE),
            "offdisk_b": _mock_record(30 * _BYTES_PER_GIGABYTE),
        }
        manager._model_pool_cluster_sizes = {}
        _set_availability(manager, present=set())
        _seed_demand_multi(manager, {"offdisk_a": 200.0, "offdisk_b": 100.0})

        manager._maybe_tick_model_pool()

        assert manager._model_pool.pending_download_models() == frozenset({"offdisk_a"})
        assert manager._model_pool_download_bytes_charged == 30 * _BYTES_PER_GIGABYTE

    def test_failed_download_is_not_refunded(self) -> None:
        """A failed pool download resolves the seat but keeps its charged bytes, since the bandwidth was spent."""
        manager = make_testable_process_manager(
            model_pool=ModelPoolConfig(enabled=True, seats=1, min_dwell_minutes=0.0, download_budget_gb=40.0),
            image_models_to_load=["offdisk_a"],
        )
        manager.stable_diffusion_reference = {"offdisk_a": _mock_record(30 * _BYTES_PER_GIGABYTE)}
        manager._model_pool_cluster_sizes = {}
        _set_availability(manager, present=set())
        _seed_demand_multi(manager, {"offdisk_a": 100.0})
        manager._maybe_tick_model_pool()
        assert manager._model_pool.pending_download_models() == frozenset({"offdisk_a"})
        assert manager._model_pool_download_bytes_charged == 30 * _BYTES_PER_GIGABYTE

        _set_availability(manager, present=set(), failed=("offdisk_a",))
        manager._observe_pool_downloads(time.monotonic())

        assert manager._model_pool.pending_download_models() == frozenset()
        assert manager._model_pool_download_bytes_charged == 30 * _BYTES_PER_GIGABYTE
        assert LedgerEventType.MODEL_POOL_DEMOTED in _ledger_event_types(manager)


class TestModelPoolDownloadWiring:
    """The manager commands pool downloads on a pending transition and observes their completion or failure."""

    def test_download_pending_sends_the_download_request(self) -> None:
        """A pending-download transition asks the download coordinator to fetch the chosen off-disk model."""
        manager = make_testable_process_manager(
            model_pool=ModelPoolConfig(enabled=True, seats=1, min_dwell_minutes=0.0, download_budget_gb=40.0),
            image_models_to_load=["offdisk_a"],
        )
        manager.stable_diffusion_reference = {"offdisk_a": _mock_record(30 * _BYTES_PER_GIGABYTE)}
        manager._model_pool_cluster_sizes = {}
        manager._download_coordinator.request_pool_model_download = Mock()
        _set_availability(manager, present=set())
        _seed_demand_multi(manager, {"offdisk_a": 100.0})

        manager._maybe_tick_model_pool()

        manager._download_coordinator.request_pool_model_download.assert_called_once_with("offdisk_a")

    def test_completed_download_reaches_the_engine_and_ledgers_it(self) -> None:
        """A pool model landing on disk resolves its pending seat and records the ready event."""
        manager = make_testable_process_manager(
            model_pool=ModelPoolConfig(enabled=True, seats=1, min_dwell_minutes=0.0, download_budget_gb=40.0),
            image_models_to_load=["offdisk_a"],
        )
        manager.stable_diffusion_reference = {"offdisk_a": _mock_record(30 * _BYTES_PER_GIGABYTE)}
        manager._model_pool_cluster_sizes = {}
        _set_availability(manager, present=set())
        _seed_demand_multi(manager, {"offdisk_a": 100.0})
        manager._maybe_tick_model_pool()
        assert manager._model_pool.pending_download_models() == frozenset({"offdisk_a"})

        _set_availability(manager, present={"offdisk_a"})
        manager._observe_pool_downloads(time.monotonic())

        assert "offdisk_a" in manager._model_pool.active_seat_models()
        assert manager._model_pool.pending_download_models() == frozenset()
        assert LedgerEventType.MODEL_POOL_DOWNLOAD_READY in _ledger_event_types(manager)


class TestModelPoolDesiredSetUnion:
    """A config reconcile send unions the pool's pending downloads into the authoritative desired set."""

    def test_reconcile_desired_set_includes_pool_pending(self) -> None:
        """The desired-image-model set on a reconcile send carries the pool-pending model so it is not pruned."""
        process_lifecycle = Mock()
        desired_state = Mock()
        plan = Mock()
        plan.desired = {"configured_x"}
        plan.to_fetch = ["configured_x"]
        plan.has_work = True
        desired_state.reconcile.return_value = plan

        availability = ModelAvailability()
        availability.update(present=set(), currently_downloading=None, pending=(), failed=())
        bridge = make_mock_bridge_data(image_models_to_load=["configured_x"])

        coordinator = ModelDownloadCoordinator(
            state=Mock(),
            process_map=Mock(),
            process_lifecycle=process_lifecycle,
            model_availability=availability,
            desired_state=desired_state,
            bridge_data_provider=lambda: bridge,
            stable_diffusion_reference_provider=lambda: None,
            enable_background_downloads=True,
            pool_pending_models_provider=lambda: frozenset({"pool_pending_y"}),
        )

        coordinator.reconcile_downloads()

        _args, kwargs = process_lifecycle.request_downloads.call_args
        assert "pool_pending_y" in kwargs["desired_image_models"]
        assert "configured_x" in kwargs["desired_image_models"]
