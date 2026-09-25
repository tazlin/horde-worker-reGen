"""The missing image model list: the predicate every surface calls and the warning the config loader writes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from loguru import logger

from horde_worker_regen import compute_mode
from horde_worker_regen.analysis.log_signatures import SIGNATURES
from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.bridge_data.load_config import BridgeDataLoader
from horde_worker_regen.capabilities import (
    IMAGE_MODELS_UNCONFIGURED_MESSAGE,
    image_models_unconfigured,
    image_models_unconfigured_in,
)

_API_KEY = "0123456789abcdef012345"
_TEXT_ROLE: dict[str, object] = {"scribe": True, "text_backend_managed": False}


@pytest.fixture(autouse=True)
def _gpu_install_and_fresh_warning_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the install to a GPU build and start every test before any config has been loaded."""
    monkeypatch.setattr(compute_mode, "is_cpu_only_install", lambda **_: False)
    monkeypatch.setattr(BridgeDataLoader, "_image_models_unconfigured_reported", False)


def _bridge_data(**config: object) -> reGenBridgeData:
    return reGenBridgeData.model_validate({"api_key": _API_KEY, **config})


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param({}, True, id="image role with the key absent"),
        pytest.param({"alchemist": True}, True, id="image and alchemy roles with the key absent"),
        pytest.param({"models_to_load": []}, False, id="explicit empty list means top 2"),
        pytest.param({"models_to_load": ["top 2"]}, False, id="explicit list"),
        pytest.param({"dreamer": False, "alchemist": True}, False, id="alchemy only"),
        pytest.param({"dreamer": False, **_TEXT_ROLE}, False, id="text only"),
        pytest.param({"dreamer": False, "alchemist": True, **_TEXT_ROLE}, False, id="alchemy and text"),
        pytest.param(
            {"gpu_overrides": {0: {"models_to_load": ["top 3"]}}}, False, id="a card configures its own list"
        ),
    ],
)
def test_the_predicate_fires_only_for_an_image_worker_with_no_model_list(
    config: dict[str, object],
    expected: bool,
) -> None:
    """Only the image role with ``models_to_load`` absent everywhere satisfies the predicate."""
    assert image_models_unconfigured_in(_bridge_data(**config)) is expected


def test_a_cpu_only_install_serves_no_image_work_so_never_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    """``dreamer`` left on over a CPU install is not the image role."""
    monkeypatch.setattr(compute_mode, "is_cpu_only_install", lambda **_: True)

    assert image_models_unconfigured_in(_bridge_data()) is False
    assert image_models_unconfigured(dreamer=True, models_to_load_configured=False) is False


def test_the_key_presence_survives_later_assignment_of_the_model_list() -> None:
    """Resolution assigns the resolved list after validation; the recorded presence does not change."""
    absent = _bridge_data()
    absent.image_models_to_load = ["Deliberate"]
    configured = _bridge_data(models_to_load=[])
    configured.image_models_to_load = []

    assert image_models_unconfigured_in(absent) is True
    assert image_models_unconfigured_in(absent.model_copy()) is True
    assert image_models_unconfigured_in(configured) is False


def test_a_mock_bridge_data_counts_as_configured() -> None:
    """A ``Mock`` reads every attribute as truthy, which must not read as a missing list."""
    assert image_models_unconfigured_in(Mock()) is False


def _load(tmp_path: Path, yaml_text: str) -> list[str]:
    """Load *yaml_text* through the config loader and return the missing-list warnings it wrote."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text(f'api_key: "{_API_KEY}"\n{yaml_text}', encoding="utf-8")
    warnings: list[str] = []
    sink_id = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING")
    try:
        BridgeDataLoader.load(path)
    finally:
        logger.remove(sink_id)
    return [line for line in warnings if SIGNATURES["image_models_unconfigured"].pattern.search(line)]


def test_startup_warns_once_with_the_registered_line(tmp_path: Path) -> None:
    """The first load of an image config with no list warns, in the wording the log registry pins."""
    assert _load(tmp_path, "") == [IMAGE_MODELS_UNCONFIGURED_MESSAGE]


def test_a_reload_with_the_key_still_absent_does_not_repeat_the_warning(tmp_path: Path) -> None:
    """Every hot reload goes through the loader, so the warning is edge-triggered."""
    _load(tmp_path, "")

    assert _load(tmp_path, "") == []


def test_the_warning_returns_after_a_reload_that_cleared_it(tmp_path: Path) -> None:
    """Removing the key again after setting it is a new occurrence and warns again."""
    _load(tmp_path, "")
    _load(tmp_path, "models_to_load:\n  - top 2\n")

    assert _load(tmp_path, "") == [IMAGE_MODELS_UNCONFIGURED_MESSAGE]


@pytest.mark.parametrize(
    "yaml_text",
    [
        pytest.param("models_to_load: []\n", id="explicit empty list"),
        pytest.param("dreamer: false\nalchemist: true\n", id="alchemy only"),
        pytest.param("dreamer: false\nscribe: true\ntext_backend_managed: false\n", id="text only"),
        pytest.param(
            "dreamer: false\nalchemist: true\nscribe: true\ntext_backend_managed: false\n", id="alchemy and text"
        ),
    ],
)
def test_no_warning_without_the_hazard(tmp_path: Path, yaml_text: str) -> None:
    """An explicit empty list, or a worker without the image role, loads without the warning."""
    assert _load(tmp_path, yaml_text) == []
