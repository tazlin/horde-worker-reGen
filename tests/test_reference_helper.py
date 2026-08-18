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


def test_offline_manager_re_asks_once_when_a_category_reads_as_absent(reset_reference_singleton: None) -> None:
    """A category that reads as absent on the first call is re-read once before it is reported missing."""
    manager = ensure_offline_reference_manager()
    calls: list[str] = []
    answers = iter([None, {"model": object()}])

    def flaky(category: object, overwrite_existing: bool = False, **kwargs: object) -> object:
        calls.append(str(category))
        return next(answers)

    manager.get_model_reference_or_none = flaky  # type: ignore[method-assign]
    # Re-install the re-ask around the flaky read, as the helper does around the real one.
    from horde_worker_regen.reference_helper import _retry_transient_category_miss

    _retry_transient_category_miss(manager)

    found = manager.get_model_reference_or_none("image_generation")
    assert found is not None and "model" in found
    assert calls == ["image_generation", "image_generation"]


def test_offline_manager_reports_a_genuinely_missing_category_after_two_reads(
    reset_reference_singleton: None,
) -> None:
    """A category absent on both reads is still reported missing; the re-ask never invents data."""
    manager = ensure_offline_reference_manager()
    calls: list[str] = []

    def absent(category: object, overwrite_existing: bool = False, **kwargs: object) -> object:
        calls.append(str(category))
        return None

    manager.get_model_reference_or_none = absent  # type: ignore[method-assign]
    from horde_worker_regen.reference_helper import _retry_transient_category_miss

    _retry_transient_category_miss(manager)

    assert manager.get_model_reference_or_none("image_generation") is None
    assert calls == ["image_generation", "image_generation"]
