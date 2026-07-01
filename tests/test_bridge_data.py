# import yaml
import asyncio
import pathlib

import pytest
from horde_model_reference.model_reference_manager import GitHubBackend, ModelReferenceManager, PrefetchStrategy
from horde_sdk.generic_api.consts import ANON_API_KEY
from pydantic import JsonValue
from ruamel.yaml import YAML

from horde_worker_regen.bridge_data.data_model import (
    _warn_lease_slots_below_threads,
    _warn_lease_without_residency,
    reGenBridgeData,
)
from horde_worker_regen.bridge_data.load_config import BridgeDataLoader, ConfigFormat


class TestLeaseResidencyWarning:
    """The lease only helps under residency, so enabling it with frequent unloads must be detected."""

    def test_disabled_lease_is_never_flagged(self) -> None:
        """A disabled lease is never flagged, regardless of unload settings."""
        assert (
            _warn_lease_without_residency(
                gpu_sampling_lease_enabled=False,
                unload_models_from_vram_often=True,
            )
            is False
        )

    def test_resident_lease_is_not_flagged(self) -> None:
        """No frequent unload is the resident config the lease needs, so it is not flagged."""
        assert (
            _warn_lease_without_residency(
                gpu_sampling_lease_enabled=True,
                unload_models_from_vram_often=False,
            )
            is False
        )

    def test_non_resident_lease_is_flagged(self) -> None:
        """A lease enabled alongside frequent unloads (no residency to overlap) is flagged."""
        assert (
            _warn_lease_without_residency(
                gpu_sampling_lease_enabled=True,
                unload_models_from_vram_often=True,
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

        bridge_data = reGenBridgeData.model_validate(raw)

        assert bridge_data.gpu_sampling_lease_enabled is True
        # The template leaves the slot count unset so it tracks max_threads at runtime.
        assert bridge_data.gpu_sampling_lease_slots is None


class TestLeaseSlotsBelowThreadsWarning:
    """An explicit slot count under max_threads under-serializes denoise, so it must be detected."""

    def test_disabled_lease_is_never_flagged(self) -> None:
        """A disabled lease is never flagged, regardless of the slot/threads relationship."""
        assert (
            _warn_lease_slots_below_threads(
                gpu_sampling_lease_enabled=False,
                gpu_sampling_lease_slots=1,
                max_threads=4,
            )
            is False
        )

    def test_auto_slots_are_never_flagged(self) -> None:
        """Unset slots (None) track max_threads, so the under-serialization case cannot arise."""
        assert (
            _warn_lease_slots_below_threads(
                gpu_sampling_lease_enabled=True,
                gpu_sampling_lease_slots=None,
                max_threads=4,
            )
            is False
        )

    def test_explicit_slots_at_or_above_threads_not_flagged(self) -> None:
        """An explicit count that meets the concurrency cap leaves no admitted concurrency unused."""
        assert (
            _warn_lease_slots_below_threads(
                gpu_sampling_lease_enabled=True,
                gpu_sampling_lease_slots=4,
                max_threads=4,
            )
            is False
        )

    def test_explicit_slots_below_threads_is_flagged(self) -> None:
        """An explicit count under max_threads samples fewer jobs at once than the worker admits."""
        assert (
            _warn_lease_slots_below_threads(
                gpu_sampling_lease_enabled=True,
                gpu_sampling_lease_slots=1,
                max_threads=2,
            )
            is True
        )


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


def test_ram_safety_defaults() -> None:
    """The RAM-safety knobs default to the conservative values that hedge against an OS OOM kill.

    A resident inference process can allocate several GB in one step, so the danger floor keeps ~15% of RAM
    free (not 10%), and a per-process ceiling bounds a single process's resident balloon so the summed
    footprint across processes and co-tenants cannot drive the host into a kernel OOM kill.
    """
    assert reGenBridgeData.model_fields["ram_pressure_pause_percent"].default == 85.0
    assert reGenBridgeData.model_fields["ram_per_process_max_mb"].default == 18432


def test_template_matches_ram_safety_defaults() -> None:
    """The shipped template must carry the same RAM-safety defaults as the model (no silent drift)."""
    yaml = YAML(typ="safe")
    with open("bridgeData_template.yaml", encoding="utf-8") as f:
        raw = yaml.load(f)
    parsed = reGenBridgeData.model_validate(raw)
    assert parsed.ram_pressure_pause_percent == 85.0
    assert parsed.ram_per_process_max_mb == 18432


def test_bridge_data_loader_yaml_template() -> None:
    """Test that the bridge data template file can be loaded and parsed by a BridgeDataLoader.

    Synchronous on purpose: ``BridgeDataLoader.load`` initializes the model reference manager via
    ``asyncio.run`` internally, which cannot run inside a pytest-asyncio event loop.
    """
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

    # `load` initializes the model reference manager via `asyncio.run`; run it off this event loop.
    bridge_data = await asyncio.to_thread(
        bridge_data_loader.load,
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

    # `load_from_env_vars` initializes the model reference manager via `asyncio.run`; run it off-loop.
    bridge_data = await asyncio.to_thread(
        BridgeDataLoader.load_from_env_vars,
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
