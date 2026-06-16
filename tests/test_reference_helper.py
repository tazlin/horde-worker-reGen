"""Tests for reference_helper's offline subprocess guarantee."""

from __future__ import annotations

from collections.abc import Generator

import horde_model_reference as _hmr
import pytest
from horde_model_reference import ModelReferenceManager, PrefetchStrategy

from horde_worker_regen.reference_helper import ensure_offline_reference_manager

# The offline subprocess guarantee requires a horde_model_reference that supports offline mode.
# Until the worker's pinned dependency is bumped to that (unreleased) version, skip rather than fail.
pytestmark = pytest.mark.skipif(
    "offline" not in _hmr.HordeModelReferenceSettings.model_fields,
    reason="installed horde_model_reference predates offline support",
)


@pytest.fixture
def reset_reference_singleton() -> Generator[None]:
    """Reset the ModelReferenceManager singleton around a test."""
    previous = ModelReferenceManager._instance
    ModelReferenceManager._instance = None
    try:
        yield
    finally:
        ModelReferenceManager._instance = previous


def test_ensure_offline_creates_offline_manager(
    reset_reference_singleton: None,
    tmp_path: object,
) -> None:
    """The helper builds an offline, write-incapable reference manager."""
    manager = ensure_offline_reference_manager()
    assert manager.offline is True
    assert manager.backend.supports_writes() is False


def test_ensure_offline_reuses_existing_offline_manager(reset_reference_singleton: None) -> None:
    """A second call returns the same already-offline singleton."""
    first = ensure_offline_reference_manager()
    second = ensure_offline_reference_manager()
    assert first is second


def test_ensure_offline_resets_inherited_non_offline_manager(
    reset_reference_singleton: None,
    tmp_path: object,
) -> None:
    """A non-offline manager inherited under fork is replaced so the subprocess cannot download."""
    online = ModelReferenceManager(
        base_path=tmp_path,  # type: ignore[arg-type]
        offline=False,
        prefetch_strategy=PrefetchStrategy.NONE,
    )
    assert online.offline is False

    offline = ensure_offline_reference_manager()
    assert offline is not online
    assert offline.offline is True
