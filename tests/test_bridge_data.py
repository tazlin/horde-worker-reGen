# import yaml
import pathlib

import pytest
from horde_model_reference.model_reference_manager import GitHubBackend, ModelReferenceManager, PrefetchStrategy
from horde_sdk.generic_api.consts import ANON_API_KEY
from pydantic import JsonValue
from ruamel.yaml import YAML

from horde_worker_regen.bridge_data.data_model import _warn_lease_without_residency, reGenBridgeData
from horde_worker_regen.bridge_data.load_config import BridgeDataLoader, ConfigFormat


class TestLeaseResidencyWarning:
    """The lease only helps under residency, so enabling it without residency must be detected."""

    def test_disabled_lease_is_never_flagged(self) -> None:
        """A disabled lease is never flagged, regardless of residency settings."""
        assert (
            _warn_lease_without_residency(
                gpu_sampling_lease_enabled=False,
                high_memory_mode=False,
                unload_models_from_vram_often=True,
            )
            is False
        )

    def test_resident_lease_is_not_flagged(self) -> None:
        """high_memory_mode with no frequent unload is the resident config the lease needs."""
        assert (
            _warn_lease_without_residency(
                gpu_sampling_lease_enabled=True,
                high_memory_mode=True,
                unload_models_from_vram_often=False,
            )
            is False
        )

    @pytest.mark.parametrize(
        ("high_memory_mode", "unload_often"),
        [
            (False, False),  # no high-memory pin: model is not held in VRAM between jobs
            (True, True),  # high-memory but actively unloads: not resident in practice
            (False, True),
        ],
    )
    def test_non_resident_lease_is_flagged(self, high_memory_mode: bool, unload_often: bool) -> None:
        """A lease enabled without true residency is flagged as counterproductive."""
        assert (
            _warn_lease_without_residency(
                gpu_sampling_lease_enabled=True,
                high_memory_mode=high_memory_mode,
                unload_models_from_vram_often=unload_often,
            )
            is True
        )

    def test_validator_exercises_warning_path(self) -> None:
        """A lease-on, residency-off config still validates (the validator's warning path runs)."""
        yaml = YAML(typ="safe")
        with open("bridgeData_template.yaml", encoding="utf-8") as f:
            raw = yaml.load(f)
        raw["gpu_sampling_lease_enabled"] = True
        raw["unload_models_from_vram_often"] = True
        raw["high_memory_mode"] = False

        bridge_data = reGenBridgeData.model_validate(raw)

        assert bridge_data.gpu_sampling_lease_enabled is True
        assert bridge_data.gpu_sampling_lease_slots == 1


def test_bridge_data_yaml() -> None:
    """Test that the bridge data template file can be loaded and parsed as YAML."""
    # bridge_data_filename = "bridgeData.yaml"
    bridge_data_filename = "bridgeData_template.yaml"
    bridge_data_raw: dict[str, JsonValue] | None = None

    yaml = YAML(typ="safe")

    with open(bridge_data_filename, encoding="utf-8") as f:
        bridge_data_raw = yaml.load(f)

    assert bridge_data_raw is not None

    parsed_bridge_data = reGenBridgeData.model_validate(bridge_data_raw)

    assert parsed_bridge_data is not None
    assert parsed_bridge_data.disable_terminal_ui is False
    assert parsed_bridge_data.api_key == ANON_API_KEY

    assert parsed_bridge_data.meta_load_instructions is not None
    assert len(parsed_bridge_data.meta_load_instructions) == 1


async def test_bridge_data_loader_yaml_template() -> None:
    """Test that the bridge data template file can be loaded and parsed by a BridgeDataLoader."""
    bridge_data_loader = BridgeDataLoader()

    if not ModelReferenceManager.has_instance():
        horde_model_reference_manager = ModelReferenceManager()
    else:
        horde_model_reference_manager = ModelReferenceManager.get_instance()

    bridge_data = bridge_data_loader.load(
        file_path="bridgeData_template.yaml",
        file_format=ConfigFormat.yaml,
        horde_model_reference_manager=horde_model_reference_manager,
    )

    assert bridge_data is not None
    assert bridge_data.disable_terminal_ui is False
    assert bridge_data.api_key == ANON_API_KEY


async def test_bridge_data_loader_yaml_local_if_present() -> None:
    """Test that the bridge data file can be loaded and parsed by a BridgeDataLoader (if present)."""
    bridge_data_loader = BridgeDataLoader()

    if not ModelReferenceManager.has_instance():
        horde_model_reference_manager = ModelReferenceManager(
            backend=GitHubBackend(),
            prefetch_strategy=PrefetchStrategy.DEFERRED,
        )
        assert horde_model_reference_manager.deferred_prefetch_handle is not None
        await horde_model_reference_manager.deferred_prefetch_handle
    else:
        horde_model_reference_manager = ModelReferenceManager.get_instance()

    if not pathlib.Path("bridgeData.yaml").is_file():
        pytest.skip("bridgeData.yaml not found")

    bridge_data = bridge_data_loader.load(
        file_path="bridgeData.yaml",
        file_format=ConfigFormat.yaml,
        horde_model_reference_manager=horde_model_reference_manager,
    )

    assert bridge_data is not None
    assert bridge_data.api_key != ANON_API_KEY
    assert len(bridge_data.image_models_to_load) > 0


async def test_bridge_data_load_from_env_vars() -> None:
    """Test that the bridge data can be loaded from environment variables."""
    import os

    os.environ["AIWORKER_REGEN_HORDE_URL"] = "https://localhost:8080"
    os.environ["AIWORKER_REGEN_MODELS_TO_LOAD"] = "['model1', 'model2']"

    if not ModelReferenceManager.has_instance():
        horde_model_reference_manager = ModelReferenceManager(
            backend=GitHubBackend(),
            prefetch_strategy=PrefetchStrategy.DEFERRED,
        )
        assert horde_model_reference_manager.deferred_prefetch_handle is not None
        await horde_model_reference_manager.deferred_prefetch_handle
    else:
        horde_model_reference_manager = ModelReferenceManager.get_instance()

    bridge_data = BridgeDataLoader.load_from_env_vars(
        horde_model_reference_manager=horde_model_reference_manager,
    )
    assert bridge_data is not None
    assert bridge_data._loaded_from_env_vars is True


def test_bridge_data_to_dot_env_file() -> None:
    """Test that the bridge data can be written to a .env file."""
    bridge_data = reGenBridgeData.model_validate({})

    bridge_data.horde_url = "https://localhost:8080"
    bridge_data.image_models_to_load = ["model1", "model2"]

    BridgeDataLoader.write_bridge_data_as_dot_env_file(bridge_data, "bridgeData.env")
    assert pathlib.Path("bridgeData.env").is_file()
