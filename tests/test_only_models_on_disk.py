"""Tests for the only_models_on_disk resolution filter on the worker."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from horde_model_reference import MODEL_REFERENCE_CATEGORY

from horde_worker_regen.bridge_data.load_config import BridgeDataLoader


def test_filter_to_models_on_disk_keeps_only_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only models whose files are present survive; the rest are dropped (never downloaded)."""
    records = {"A": object(), "B": object(), "C": object()}
    manager = Mock()
    manager.get_all_model_references.return_value = {MODEL_REFERENCE_CATEGORY.image_generation: records}

    present = {"A", "C"}
    monkeypatch.setattr(
        "horde_worker_regen.model_download_plan.is_model_present",
        lambda name, reference: name in present,
    )

    result = BridgeDataLoader._filter_to_models_on_disk(["A", "B", "C"], manager)

    assert sorted(result) == ["A", "C"]


def test_filter_to_models_on_disk_empty_when_none_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing on disk, the served set is empty rather than a download list."""
    manager = Mock()
    manager.get_all_model_references.return_value = {MODEL_REFERENCE_CATEGORY.image_generation: {"A": object()}}
    monkeypatch.setattr(
        "horde_worker_regen.model_download_plan.is_model_present",
        lambda name, reference: False,
    )

    assert BridgeDataLoader._filter_to_models_on_disk(["A"], manager) == []
