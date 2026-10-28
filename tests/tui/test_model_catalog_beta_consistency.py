"""The editor's cached records and the picker read the same merged canonical-plus-beta view.

A beta (pending-queue) model is absent from the canonical reference, so the editor's disk footer would
miscount an on-disk beta model as "to download" if it read the canonical-only ``get_all_model_references``.
Both ``cached_image_records`` and ``load_image_models`` read the merged view in
``horde_worker_regen.bridge_data.beta_source``, so both surfaces agree with the worker.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

from horde_worker_regen.bridge_data.beta_source import invalidate_merged_records
from horde_worker_regen.tui import model_catalog
from tests.test_beta_source import PendingAwareManagerDouble, image_record, serve_pending

_BETA_MODEL = "Beta-Model"
_CANONICAL_MODEL = "Canonical-Model"


@pytest.fixture(autouse=True)
def _fresh_merged_view() -> Iterator[None]:
    invalidate_merged_records()
    yield
    invalidate_merged_records()


_BOUNDED_BUILD = {"timeout_seconds": 5.0, "retry_max_attempts": 1}


def _manager_with_beta(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PendingAwareManagerDouble, list[dict[str, object]]]:
    """Return a manager with one canonical and one beta model, and the pending provider build calls."""
    build_calls = serve_pending(
        monkeypatch,
        {MODEL_REFERENCE_CATEGORY.image_generation: {_BETA_MODEL: image_record(_BETA_MODEL)}},
    )
    manager = PendingAwareManagerDouble(
        {MODEL_REFERENCE_CATEGORY.image_generation: {_CANONICAL_MODEL: image_record(_CANONICAL_MODEL)}},
    )
    return manager, build_calls


def test_cached_image_records_includes_beta_via_the_merged_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """cached_image_records returns the merged view's records through the bounded provider."""
    from horde_model_reference.model_reference_manager import ModelReferenceManager

    manager, build_calls = _manager_with_beta(monkeypatch)
    monkeypatch.setattr(ModelReferenceManager, "has_instance", classmethod(lambda _cls: True))
    monkeypatch.setattr(ModelReferenceManager, "get_instance", classmethod(lambda _cls: manager))

    records = model_catalog.cached_image_records()

    assert records is not None
    assert set(records) == {_BETA_MODEL, _CANONICAL_MODEL}
    assert build_calls == [_BOUNDED_BUILD]


def test_picker_flags_beta_models_from_the_merged_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_image_models lists the beta model, flags only it as beta, and reads through the bounded provider."""
    manager, build_calls = _manager_with_beta(monkeypatch)
    monkeypatch.setattr(model_catalog, "_loaded_model_reference_manager", lambda: manager)

    models = {model.name: model for model in model_catalog.load_image_models()}

    assert set(models) == {_BETA_MODEL, _CANONICAL_MODEL}
    assert models[_BETA_MODEL].is_beta
    assert not models[_CANONICAL_MODEL].is_beta
    assert build_calls == [_BOUNDED_BUILD]


def test_cached_image_records_none_without_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Still non-forcing: no manager instance means no records (never triggers a fetch)."""
    from horde_model_reference.model_reference_manager import ModelReferenceManager

    monkeypatch.setattr(ModelReferenceManager, "has_instance", classmethod(lambda _cls: False))
    assert model_catalog.cached_image_records() is None
