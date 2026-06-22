"""The editor's cached records resolve the same beta-aware set as the picker.

A beta (pending-queue) model such as Z-Image-Turbo is absent from the canonical reference, so the
editor's disk footer would miscount an on-disk beta model as "to download" if it read the canonical-only
``get_all_model_references``. ``cached_image_records`` must instead go through the beta-aware loader the
picker uses, so both surfaces agree.
"""

from __future__ import annotations

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import (
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)

from horde_worker_regen.tui import model_catalog


def _zimage_record() -> ImageGenerationModelRecord:
    return ImageGenerationModelRecord(
        name="Z-Image-Turbo",
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo,
        nsfw=True,
        description="beta record",
        config=GenericModelRecordConfig(download=[]),
    )


def test_cached_image_records_includes_beta_via_shared_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """cached_image_records returns the beta-aware loader's records, not the canonical-only reference."""
    from horde_model_reference.model_reference_manager import ModelReferenceManager

    monkeypatch.setattr(ModelReferenceManager, "has_instance", classmethod(lambda _cls: True))
    monkeypatch.setattr(ModelReferenceManager, "get_instance", classmethod(lambda _cls: object()))

    beta_records = {"Z-Image-Turbo": _zimage_record()}
    monkeypatch.setattr(
        model_catalog,
        "_image_records_with_beta",
        lambda _manager: (beta_records, {"Z-Image-Turbo"}),
    )

    records = model_catalog.cached_image_records()
    assert records is not None
    assert "Z-Image-Turbo" in records


def test_cached_image_records_none_without_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Still non-forcing: no manager instance means no records (never triggers a fetch)."""
    from horde_model_reference.model_reference_manager import ModelReferenceManager

    monkeypatch.setattr(ModelReferenceManager, "has_instance", classmethod(lambda _cls: False))
    assert model_catalog.cached_image_records() is None
