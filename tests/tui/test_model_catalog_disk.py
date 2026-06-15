"""Tests for the model picker's disk-summary helper (config disk budget for the editor footer)."""

from __future__ import annotations

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)

from horde_worker_regen.tui import model_catalog


def _record(name: str, file_name: str, size: int | None) -> ImageGenerationModelRecord:
    return ImageGenerationModelRecord(
        name=name,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
        nsfw=False,
        description="test",
        size_on_disk_bytes=size,
        config=GenericModelRecordConfig(
            download=[DownloadRecord(file_name=file_name, file_url=f"https://example/{file_name}")],
        ),
    )


def test_disk_summary_none_when_reference_not_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no cached reference, the summary is None so the editor shows a hint rather than forcing a load."""
    monkeypatch.setattr(model_catalog, "cached_image_records", lambda: None)
    assert model_catalog.disk_summary(["anything"]) is None


def test_disk_summary_counts_unsized_meta_and_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Meta commands and names absent from the reference are counted as unsized, not sized into totals."""
    reference = {"Known": _record("Known", "known.safetensors", 1000)}
    monkeypatch.setattr(model_catalog, "cached_image_records", lambda: reference)

    summary = model_catalog.disk_summary(["Known", "top 5", "NotInReference"])
    assert summary is not None
    # "top 5" is a meta command and "NotInReference" is absent; both are unsized.
    assert summary.num_unsized == 2
    assert summary.sizes_complete is False
