"""The parent publishes the pending (beta) records for its children at startup and on each reference refresh."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from horde_model_reference import PENDING_SOURCE_ID, StaticModelProvider

import horde_worker_regen.bridge_data.beta_source as beta_source
import horde_worker_regen.process_management.process_manager as process_manager_module
from horde_worker_regen.bridge_data.beta_source import BETA_MODELS_DIR_ENV_VAR
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from tests.process_management.conftest import make_testable_process_manager

_PENDING_TTL_SECONDS = 60


def _join_refresh_thread() -> None:
    for thread in threading.enumerate():
        if thread.name in {"reference-refresh", "pending-publish-retry"}:
            thread.join(timeout=10)


def _manager_with_publish_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[bool],
) -> tuple[HordeWorkerProcessManager, list[object], Mock]:
    """Return a manager whose pending provider has a TTL, the publish calls, and the broadcast mock.

    Each publish returns the next of ``outcomes``. The first full refresh is due on the first tick.
    """
    process_manager = make_testable_process_manager()
    reference_manager = Mock()
    reference_manager.get_provider.return_value = StaticModelProvider(
        PENDING_SOURCE_ID,
        {},
        cache_ttl_seconds=_PENDING_TTL_SECONDS,
    )
    process_manager.horde_model_reference_manager = reference_manager
    remaining = list(outcomes)
    published: list[object] = []

    def _share(manager: object) -> bool:
        published.append(manager)
        return remaining.pop(0)

    monkeypatch.setattr(process_manager_module, "invalidate_merged_records", lambda: None)
    monkeypatch.setattr(process_manager_module, "share_pending_records_with_children", _share)
    broadcast = Mock()
    monkeypatch.setattr(process_manager._process_lifecycle, "broadcast_reload_model_database", broadcast)
    process_manager._last_reference_refresh = 0.0
    return process_manager, published, broadcast


def _tick(process_manager: HordeWorkerProcessManager) -> None:
    process_manager._maybe_refresh_references()
    _join_refresh_thread()


def test_incomplete_publish_is_retried_after_the_provider_ttl_and_broadcasts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A publish that left a category unwritten is retried between refreshes, and children reload on success."""
    process_manager, published, broadcast = _manager_with_publish_outcomes(monkeypatch, [False, True])
    _tick(process_manager)
    _tick(process_manager)
    assert broadcast.call_count == 1

    process_manager._last_pending_publish -= _PENDING_TTL_SECONDS + 1
    _tick(process_manager)
    _tick(process_manager)

    assert len(published) == 2
    assert not process_manager._pending_publish_incomplete
    assert broadcast.call_count == 2


def test_incomplete_publish_is_not_retried_before_the_provider_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ticks inside the provider's TTL leave the incomplete publish waiting."""
    process_manager, published, broadcast = _manager_with_publish_outcomes(monkeypatch, [False])
    _tick(process_manager)

    _tick(process_manager)
    _tick(process_manager)

    assert len(published) == 1
    assert process_manager._pending_publish_incomplete
    assert broadcast.call_count == 1


def test_complete_publish_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A publish that wrote every category waits for the next full refresh."""
    process_manager, published, broadcast = _manager_with_publish_outcomes(monkeypatch, [True])
    _tick(process_manager)

    process_manager._last_pending_publish -= _PENDING_TTL_SECONDS + 1
    _tick(process_manager)
    _tick(process_manager)

    assert len(published) == 1
    assert broadcast.call_count == 1


def test_init_publishes_pending_records_and_points_children_at_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The parent reads on the network provider, then publishes and sets the directory for later spawns."""
    monkeypatch.setenv(BETA_MODELS_DIR_ENV_VAR, "inherited-directory")
    manager = Mock()
    monkeypatch.setattr(process_manager_module.ModelReferenceManager, "get_instance", lambda: manager)
    directory_during_parent_read: list[str | None] = []

    def _records(_manager: object) -> dict[str, Mock]:
        directory_during_parent_read.append(os.environ.get(BETA_MODELS_DIR_ENV_VAR))
        return {"Some-Model": Mock()}

    published: list[tuple[object, Path]] = []
    monkeypatch.setattr(process_manager_module, "beta_aware_image_records", _records)
    monkeypatch.setattr(beta_source, "pending_records_directory", lambda: tmp_path)
    monkeypatch.setattr(
        beta_source,
        "publish_pending_records",
        lambda publish_manager, directory: published.append((publish_manager, directory)) or frozenset(),
    )
    process_manager = make_testable_process_manager()
    process_manager.stable_diffusion_reference = None

    process_manager._init_model_reference()

    assert directory_during_parent_read == [None]
    assert published == [(manager, tmp_path)]
    assert os.environ[BETA_MODELS_DIR_ENV_VAR] == str(tmp_path)


def test_refresh_publishes_after_invalidation_and_before_the_broadcast(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reloading children read files written from the refreshed view."""
    events: list[str] = []
    process_manager = make_testable_process_manager()
    monkeypatch.setattr(process_manager_module, "invalidate_merged_records", lambda: events.append("invalidate"))

    def _share(_manager: object) -> bool:
        events.append("publish")
        return True

    monkeypatch.setattr(process_manager_module, "share_pending_records_with_children", _share)
    broadcast = Mock(side_effect=lambda: events.append("broadcast"))
    monkeypatch.setattr(process_manager._process_lifecycle, "broadcast_reload_model_database", broadcast)
    process_manager._last_reference_refresh = 0.0

    process_manager._maybe_refresh_references()
    _join_refresh_thread()
    process_manager._maybe_refresh_references()

    assert events == ["invalidate", "publish", "broadcast"]


def test_failed_refresh_does_not_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refresh whose canonical download fails leaves the published files as they were."""
    shared: list[object] = []
    process_manager = make_testable_process_manager()
    reference_manager = Mock()
    reference_manager.get_all_model_references_or_none.side_effect = ConnectionError("PRIMARY unreachable")
    process_manager.horde_model_reference_manager = reference_manager
    monkeypatch.setattr(process_manager_module, "share_pending_records_with_children", shared.append)
    process_manager._last_reference_refresh = 0.0

    process_manager._maybe_refresh_references()
    _join_refresh_thread()

    assert shared == []
    assert not process_manager._pending_reference_reload_broadcast
