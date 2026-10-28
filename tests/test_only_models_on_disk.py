"""The only_models_on_disk filter prices presence against the merged canonical-plus-beta reference."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from loguru import logger

from horde_worker_regen.bridge_data.load_config import BridgeDataLoader
from tests.test_beta_source import PendingAwareManagerDouble, image_record, lay_checkpoint, serve_pending

_CANONICAL_PRESENT = "Canonical-Present"
_CANONICAL_ABSENT = "Canonical-Absent"
_BETA_PRESENT = "Beta-Present"
_BETA_ABSENT = "Beta-Absent"


@pytest.fixture
def cache_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the weights root at an empty temporary cache home."""
    monkeypatch.setenv("AIWORKER_CACHE_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def info_lines() -> Iterator[list[str]]:
    """Collect the INFO-and-above log messages emitted during the test."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    yield messages
    logger.remove(sink_id)


def _manager_with_beta(cache_home: Path, monkeypatch: pytest.MonkeyPatch) -> PendingAwareManagerDouble:
    """Build a manager whose canonical and pending sources each hold one present and one absent model."""
    canonical_present = image_record(_CANONICAL_PRESENT)
    beta_present = image_record(_BETA_PRESENT)
    lay_checkpoint(cache_home, canonical_present)
    lay_checkpoint(cache_home, beta_present)
    serve_pending(
        monkeypatch,
        {
            MODEL_REFERENCE_CATEGORY.image_generation: {
                _BETA_PRESENT: beta_present,
                _BETA_ABSENT: image_record(_BETA_ABSENT),
            }
        },
    )
    return PendingAwareManagerDouble(
        {
            MODEL_REFERENCE_CATEGORY.image_generation: {
                _CANONICAL_PRESENT: canonical_present,
                _CANONICAL_ABSENT: image_record(_CANONICAL_ABSENT),
            },
        },
    )


def test_beta_model_with_its_file_present_survives(cache_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A beta model whose checkpoint is on disk is kept, though the canonical reference lacks its record."""
    manager = _manager_with_beta(cache_home, monkeypatch)

    result = BridgeDataLoader._filter_to_models_on_disk([_BETA_PRESENT], manager)  # type: ignore[arg-type]

    assert result == [_BETA_PRESENT]


def test_presence_decides_for_beta_and_canonical_alike(cache_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Present models of either source survive and absent models of either source are dropped."""
    manager = _manager_with_beta(cache_home, monkeypatch)

    result = BridgeDataLoader._filter_to_models_on_disk(
        [_CANONICAL_PRESENT, _CANONICAL_ABSENT, _BETA_PRESENT, _BETA_ABSENT],
        manager,  # type: ignore[arg-type]
    )

    assert sorted(result) == sorted([_CANONICAL_PRESENT, _BETA_PRESENT])


def test_drop_line_names_the_dropped_models_sorted(
    cache_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    info_lines: list[str],
) -> None:
    """The INFO drop line lists every dropped model in sorted order."""
    manager = _manager_with_beta(cache_home, monkeypatch)

    BridgeDataLoader._filter_to_models_on_disk(
        [_BETA_ABSENT, _CANONICAL_PRESENT, _CANONICAL_ABSENT],
        manager,  # type: ignore[arg-type]
    )

    drop_lines = [line for line in info_lines if "only_models_on_disk" in line]
    assert len(drop_lines) == 1
    assert str(sorted([_BETA_ABSENT, _CANONICAL_ABSENT])) in drop_lines[0]


def test_nothing_on_disk_serves_nothing(cache_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no checkpoint on disk the served set is empty, never a download list."""
    monkeypatch.delenv("HORDELIB_BETA_MODEL_CATEGORIES", raising=False)
    manager = PendingAwareManagerDouble(
        {MODEL_REFERENCE_CATEGORY.image_generation: {_CANONICAL_ABSENT: image_record(_CANONICAL_ABSENT)}},
    )

    assert BridgeDataLoader._filter_to_models_on_disk([_CANONICAL_ABSENT], manager) == []  # type: ignore[arg-type]
