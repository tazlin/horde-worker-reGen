"""The default image model list: the predicate that gates it and the config loader that applies it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from loguru import logger

from horde_worker_regen import compute_mode
from horde_worker_regen.analysis.log_signatures import SIGNATURES
from horde_worker_regen.bridge_data import AIWORKER_ENV_PREFIXES
from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.bridge_data.load_config import DEFAULT_IMAGE_MODELS_MESSAGE, BridgeDataLoader
from horde_worker_regen.capabilities import DEFAULT_IMAGE_MODELS_TO_LOAD, image_models_unconfigured

_API_KEY = "0123456789abcdef012345"
_TEXT_ROLE: dict[str, object] = {"scribe": True, "text_backend_managed": False}


@pytest.fixture(autouse=True)
def _gpu_install_and_fresh_default_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the install to a GPU build and start every test before any config has been loaded."""
    monkeypatch.setattr(compute_mode, "is_cpu_only_install", lambda **_: False)
    monkeypatch.setattr(BridgeDataLoader, "_default_image_models_applied", False)


def _bridge_data(**config: object) -> reGenBridgeData:
    return reGenBridgeData.model_validate({"api_key": _API_KEY, **config})


def _unconfigured(bridge_data: reGenBridgeData) -> bool:
    """Apply the predicate to a validated config, as the loader does."""
    return image_models_unconfigured(
        dreamer=bridge_data.dreamer,
        models_to_load_configured=bridge_data.models_to_load_configured,
    )


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
    assert _unconfigured(_bridge_data(**config)) is expected


def test_a_cpu_only_install_serves_no_image_work_so_never_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    """``dreamer`` left on over a CPU install is not the image role."""
    monkeypatch.setattr(compute_mode, "is_cpu_only_install", lambda **_: True)

    assert _unconfigured(_bridge_data()) is False
    assert image_models_unconfigured(dreamer=True, models_to_load_configured=False) is False


def test_the_key_presence_survives_later_assignment_of_the_model_list() -> None:
    """Resolution assigns the resolved list after validation; the recorded presence does not change."""
    absent = _bridge_data()
    absent.image_models_to_load = ["Deliberate"]
    configured = _bridge_data(models_to_load=[])
    configured.image_models_to_load = []

    assert _unconfigured(absent) is True
    assert _unconfigured(absent.model_copy()) is True
    assert _unconfigured(configured) is False


def _capture_default_lines() -> tuple[list[str], int]:
    """Attach a WARNING sink and return its buffer and id."""
    warnings: list[str] = []
    sink_id = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING")
    return warnings, sink_id


def _default_lines(warnings: list[str]) -> list[str]:
    return [line for line in warnings if SIGNATURES["image_models_unconfigured"].pattern.search(line)]


def _load(tmp_path: Path, yaml_text: str) -> tuple[reGenBridgeData, list[str]]:
    """Load *yaml_text* through the config loader; return the config and the default-list lines it logged."""
    path = tmp_path / "bridgeData.yaml"
    path.write_text(f'api_key: "{_API_KEY}"\n{yaml_text}', encoding="utf-8")
    warnings, sink_id = _capture_default_lines()
    try:
        bridge_data = BridgeDataLoader.load(path)
    finally:
        logger.remove(sink_id)
    return bridge_data, _default_lines(warnings)


def test_the_default_matches_the_sdk_substitute_for_an_empty_list() -> None:
    """The loader's default is what the SDK validator writes for an explicit empty list."""
    explicit_empty = _bridge_data(models_to_load=[])

    assert explicit_empty.image_models_to_load == []
    assert explicit_empty.meta_load_instructions == list(DEFAULT_IMAGE_MODELS_TO_LOAD)


def test_an_image_worker_with_the_key_absent_loads_top_2_and_says_so(tmp_path: Path) -> None:
    """The meta instruction is recorded as the SDK's validation would, with one line in the registered wording."""
    bridge_data, lines = _load(tmp_path, "")

    assert bridge_data.image_models_to_load == []
    assert bridge_data.meta_load_instructions == list(DEFAULT_IMAGE_MODELS_TO_LOAD)
    assert lines == [DEFAULT_IMAGE_MODELS_MESSAGE]


def test_an_explicit_empty_list_gets_top_2_from_the_sdk_without_a_second_line(tmp_path: Path) -> None:
    """The SDK validator substitutes and logs its own line; the loader adds nothing."""
    bridge_data, lines = _load(tmp_path, "models_to_load: []\n")

    assert bridge_data.meta_load_instructions == list(DEFAULT_IMAGE_MODELS_TO_LOAD)
    assert lines == []


@pytest.mark.parametrize(
    ("yaml_text", "expected_models", "expected_meta"),
    [
        pytest.param("models_to_load:\n  - Deliberate\n", ["Deliberate"], None, id="model name"),
        pytest.param("models_to_load:\n  - top 5\n", [], ["top 5"], id="meta instruction"),
    ],
)
def test_an_explicit_list_is_untouched(
    tmp_path: Path,
    yaml_text: str,
    expected_models: list[str],
    expected_meta: list[str] | None,
) -> None:
    """An operator's list is the one that loads."""
    bridge_data, lines = _load(tmp_path, yaml_text)

    assert bridge_data.image_models_to_load == expected_models
    assert bridge_data.meta_load_instructions == expected_meta
    assert lines == []


@pytest.mark.parametrize(
    "yaml_text",
    [
        pytest.param("dreamer: false\nalchemist: true\n", id="alchemy only"),
        pytest.param("dreamer: false\nscribe: true\ntext_backend_managed: false\n", id="text only"),
        pytest.param(
            "dreamer: false\nalchemist: true\nscribe: true\ntext_backend_managed: false\n", id="alchemy and text"
        ),
        pytest.param(
            "gpu_overrides:\n  0:\n    models_to_load:\n      - top 3\n", id="a card configures its own list"
        ),
    ],
)
def test_no_default_without_the_image_role_or_with_a_card_list(tmp_path: Path, yaml_text: str) -> None:
    """A worker without the image role, or with a card's own list, keeps an empty global list and logs nothing."""
    bridge_data, lines = _load(tmp_path, yaml_text)

    assert bridge_data.image_models_to_load == []
    assert bridge_data.meta_load_instructions is None
    assert lines == []


def test_a_cpu_only_install_keeps_an_empty_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``dreamer`` on over a CPU install does not serve image generation, so no default applies."""
    monkeypatch.setattr(compute_mode, "is_cpu_only_install", lambda **_: True)

    bridge_data, lines = _load(tmp_path, "")

    assert bridge_data.image_models_to_load == []
    assert bridge_data.meta_load_instructions is None
    assert lines == []


def test_a_hot_reload_applies_the_default_without_repeating_the_line(tmp_path: Path) -> None:
    """Every hot reload goes through the loader, so reloads that keep applying the default stay silent."""
    _load(tmp_path, "")

    reloaded, lines = _load(tmp_path, "")

    assert reloaded.meta_load_instructions == list(DEFAULT_IMAGE_MODELS_TO_LOAD)
    assert lines == []


def test_the_line_returns_after_a_reload_that_set_the_key(tmp_path: Path) -> None:
    """Removing the key again after a reload that set it is a new occurrence and logs again."""
    _load(tmp_path, "")
    _, set_lines = _load(tmp_path, "models_to_load:\n  - top 2\n")

    reloaded, lines = _load(tmp_path, "")

    assert set_lines == []
    assert reloaded.meta_load_instructions == list(DEFAULT_IMAGE_MODELS_TO_LOAD)
    assert lines == [DEFAULT_IMAGE_MODELS_MESSAGE]


def test_an_env_var_config_with_the_key_absent_loads_top_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env-var loader applies the same default."""
    for key in list(os.environ):
        if key.startswith(AIWORKER_ENV_PREFIXES):
            monkeypatch.delenv(key)
    monkeypatch.setenv("AIWORKER_REGEN_API_KEY", _API_KEY)
    # load_env_vars exports config values into os.environ, which would outlive the test.
    monkeypatch.setattr(reGenBridgeData, "load_env_vars", lambda self: None)

    warnings, sink_id = _capture_default_lines()
    try:
        bridge_data = BridgeDataLoader.load_from_env_vars()
    finally:
        logger.remove(sink_id)

    assert bridge_data.meta_load_instructions == list(DEFAULT_IMAGE_MODELS_TO_LOAD)
    assert _default_lines(warnings) == [DEFAULT_IMAGE_MODELS_MESSAGE]
