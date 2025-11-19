"""The config model and initializers for the reGen configuration model."""

from __future__ import annotations

import json
import os

from horde_sdk.ai_horde_worker.bridge_data import CombinedHordeBridgeData
from loguru import logger
from pydantic import Field, field_validator, model_validator
from ruamel.yaml import YAML

from horde_worker_regen.consts import TOTAL_LORA_DOWNLOAD_TIMEOUT
from horde_worker_regen.locale_info.regen_bridge_data_fields import BRIDGE_DATA_FIELD_DESCRIPTIONS
from horde_worker_regen.validation.performance_validator import PerformanceModeValidator


class reGenBridgeData(CombinedHordeBridgeData):
    """The config model for reGen. Extra fields added here are specific to this worker implementation.

    See `CombinedHordeBridgeData` from the SDK for more information..
    """

    _loaded_from_env_vars: bool = False

    disable_terminal_ui: bool = Field(
        default=True,
    )

    safety_on_gpu: bool = Field(
        default=False,
    )
    """If true, the safety model will be run on the GPU."""

    _yaml_loader: YAML | None = None

    cycle_process_on_model_change: bool = Field(
        default=False,
    )
    """If true, the process will stop and restart when the model loaded changes.

    Warning: This can cause substantial delays in processing.
    """

    CIVIT_API_TOKEN: str | None = Field(
        default=None,
        alias="civitai_api_token",
    )
    """The API token for CivitAI, used for downloading LoRas and login-required models."""

    unload_models_from_vram_often: bool = Field(default=True)
    """If true, models will be unloaded from VRAM more often."""

    process_timeout: int = Field(default=300)
    """The maximum amount of time to allow a job to run before it is killed"""

    post_process_timeout: int = Field(default=60, ge=15)

    download_timeout: int = Field(default=TOTAL_LORA_DOWNLOAD_TIMEOUT + 1)
    """The maximum amount of time to allow an aux model to download before it is killed"""
    preload_timeout: int = Field(default=80, ge=15)
    """The maximum amount of time to allow a model to load before it is killed"""
    inference_step_timeout: int = Field(default=15, ge=15, le=30)
    """The maximum amount of time to allow a single inference step to run before the process is killed"""

    minutes_allowed_without_jobs: int = Field(default=30, ge=0, lt=60 * 60)

    horde_model_stickiness: float = Field(default=0.0, le=1.0, ge=0.0, alias="model_stickiness")
    """
    A percent chance (expressed as a decimal between 0 and 1) that the currently loaded models will
    be favored when popping a job.
    """

    high_memory_mode: bool = Field(default=False)
    """Indicates that the worker should consume more memory to improve performance."""

    very_high_memory_mode: bool = Field(default=False)
    """Indicates that the worker should consume even more memory to improve performance.

    This has data-center grade cards in mind, and is not recommended for consumer grade cards.
    """

    high_performance_mode: bool = Field(default=False)
    """If you have a 4090 or better, set this to true to enable high performance mode."""

    moderate_performance_mode: bool = Field(default=False)
    """If you have a 3080 or better, set this to true to enable moderate performance mode."""

    very_fast_disk_mode: bool = Field(default=False)
    """If you have a very fast disk, set this to true to concurrently load more models at a time from disk."""

    post_process_job_overlap: bool = Field(default=False)
    """High and moderate performance modes will skip post processing if this is set to true."""

    capture_kudos_training_data: bool = Field(default=False)

    kudos_training_data_file: str | None = Field(default=None)

    exit_on_unhandled_faults: bool = Field(default=False)
    """If true, the worker will exit if an unhandled fault occurs instead of attempting to recover."""

    purge_loras_on_download: bool = Field(default=False)

    remove_maintenance_on_init: bool = Field(default=False)

    load_large_models: bool = Field(default=False)

    custom_models: list[dict] = Field(
        default_factory=list,
    )

    limited_console_messages: bool = Field(default=False)
    """If true, the worker will only log for submit and the status message.

    Set stats_output_frequency (in seconds) for control over the status message.
    """

    # Mock process configuration (for testing/development)
    enable_mock_processes: bool = Field(default=False)
    """If true, use mock processes instead of real GPU processes (for testing/development).

    ⚠️ WARNING: Mock processes generate fake images and should NEVER be used in production!
    This mode is intended for testing the terminal UI, event system, and worker orchestration
    without requiring GPU hardware.
    """

    mock_speed_multiplier: float = Field(default=1.0, ge=0.1, le=1000.0)
    """Speed multiplier for mock processes (e.g., 10.0 = 10x faster, 0.1 = 10x slower).

    Higher values speed up testing and development. Only applies when enable_mock_processes=True.
    """

    mock_enable_failures: bool = Field(default=False)
    """Enable random job failures in mock processes for testing error handling."""

    mock_failure_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    """Probability of job failure in mock processes (0.0-1.0). Default is 5% failure rate."""

    mock_enable_slowdowns: bool = Field(default=False)
    """Enable random job slowdowns in mock processes for testing timeout handling."""

    mock_slowdown_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    """Probability of slow job in mock processes (0.0-1.0). Default is 10% slowdown rate."""

    mock_slowdown_multiplier: float = Field(default=3.0, ge=1.0, le=100.0)
    """How much slower a slow job should be in mock processes. Default is 3x slower."""

    mock_vram_usage_mb: int = Field(default=8192, ge=0)
    """Simulated VRAM usage for mock processes in megabytes. Default is 8GB."""

    mock_ram_usage_mb: int = Field(default=4096, ge=0)
    """Simulated RAM usage for mock processes in megabytes. Default is 4GB."""

    mock_scenario: str | None = Field(default=None)
    """Predefined mock scenario name for testing specific behaviors.

    Options: "HAPPY_PATH", "RANDOM_FAILURES", "SLOW_INFERENCE", "STUCK_PROCESS",
    "DOWNLOAD_FAILURES", "MEMORY_PRESSURE", "RAPID_FIRE"
    """

    @model_validator(mode="after")
    def validate_performance_modes(self) -> reGenBridgeData:
        """Validate the performance modes and set the appropriate values.

        Returns:
            The config model with the performance modes set appropriately.
        """
        return PerformanceModeValidator.validate_and_adjust_performance_modes(self)

    @field_validator("dreamer_worker_name", mode="after")
    def validate_dreamer_worker_name(cls, value: str) -> str:
        """Apply the environment variable override for the `dreamer_worker_name` field."""
        AIWORKER_DREAMER_WORKER_NAME = os.getenv("AIWORKER_DREAMER_WORKER_NAME")
        if AIWORKER_DREAMER_WORKER_NAME:
            logger.warning(
                "AIWORKER_DREAMER_WORKER_NAME environment variable is set. This will override the value for "
                "`dreamer_worker_name` in the config file.",
            )
            return AIWORKER_DREAMER_WORKER_NAME

        return value

    @model_validator(mode="after")
    def validate_mock_configuration(self) -> reGenBridgeData:
        """Validate and warn about mock process configuration."""
        if self.enable_mock_processes:
            logger.warning("=" * 80)
            logger.warning("⚠️  MOCK MODE ENABLED ⚠️")
            logger.warning("=" * 80)
            logger.warning("Mock processes will be used instead of real GPU processes!")
            logger.warning("Generated images will be FAKE placeholders, NOT real images.")
            logger.warning("This mode is for TESTING AND DEVELOPMENT ONLY.")
            logger.warning("DO NOT use mock mode in production or for real work!")
            logger.warning("=" * 80)

            if self.mock_scenario:
                logger.info(f"Mock scenario: {self.mock_scenario}")
            if self.mock_speed_multiplier != 1.0:
                logger.info(f"Mock speed multiplier: {self.mock_speed_multiplier}x")
            if self.mock_enable_failures:
                logger.info(f"Mock failure simulation enabled ({self.mock_failure_rate * 100:.1f}% rate)")
            if self.mock_enable_slowdowns:
                logger.info(
                    f"Mock slowdown simulation enabled ({self.mock_slowdown_rate * 100:.1f}% rate, "
                    f"{self.mock_slowdown_multiplier}x slower)",
                )

        return self

    def prepare_custom_models(self) -> None:
        """Prepare the custom models."""
        if os.getenv("HORDELIB_CUSTOM_MODELS"):
            logger.info(
                f"HORDELIB_CUSTOM_MODELS already set to '{os.getenv('HORDELIB_CUSTOM_MODELS')}. "
                "Doing nothing for custom models.",
            )
            return
        custom_models_dict = {}
        for model in self.custom_models:
            if not model.get("name"):
                logger.warning(f"Model name not specified for custom model entry {model}. Skipping")
                continue
            if not model.get("baseline"):
                logger.warning(f"Model baseline not specified for custom model entry {model}. Skipping")
                continue
            if not model.get("filepath"):
                logger.warning(f"Model filepath not specified for custom model entry {model}. Skipping")
                continue
            # TODO: Handle Stable Cascade models
            custom_models_dict[model["name"]] = {
                "name": model["name"],
                "baseline": model["baseline"],
                "type": "ckpt",
                "config": {"files": [{"path": model["filepath"]}]},
            }
        cwd = os.getcwd()
        if len(custom_models_dict) > 0:
            with open(f"{cwd}/custom_models.json", "w") as f:
                json.dump(custom_models_dict, f, indent=4)
        else:
            if os.path.exists(f"{cwd}/custom_models.json"):
                os.remove(f"{cwd}/custom_models.json")
        os.environ["HORDELIB_CUSTOM_MODELS"] = f"{cwd}/custom_models.json"

    @staticmethod
    def load_custom_models() -> None:
        """Load the custom models from the `custom_models.json` file."""
        cwd = os.getcwd()
        if not os.getenv("HORDELIB_CUSTOM_MODELS") and os.path.exists(f"{cwd}/custom_models.json"):
            os.environ["HORDELIB_CUSTOM_MODELS"] = f"{cwd}/custom_models.json"
            logger.debug(f"HORDELIB_CUSTOM_MODELS: {cwd}/custom_models.json")

    def load_env_vars(self) -> None:
        """Load the environment variables into the config model."""
        # See load_env_vars.py's `def load_env_vars(self) -> None:`
        if self.models_folder_parent and os.getenv("AIWORKER_CACHE_HOME") is None:
            os.environ["AIWORKER_CACHE_HOME"] = self.models_folder_parent
        if self.horde_url:
            if os.environ.get("AI_HORDE_URL"):
                logger.warning(
                    "AI_HORDE_URL environment variable already set. This will override the value for `horde_url` in "
                    "the config file.",
                )
            else:
                if os.environ.get("AI_HORDE_DEV_URL"):
                    logger.warning(
                        "AI_HORDE_DEV_URL environment variable already set. This will override the value for "
                        "`horde_url` in the config file.",
                    )
                if os.environ.get("AI_HORDE_URL") is None:
                    os.environ["AI_HORDE_URL"] = self.horde_url
                else:
                    logger.warning(
                        "AI_HORDE_URL environment variable already set. This will override the value for `horde_url` "
                        "in the config file.",
                    )

        if self.CIVIT_API_TOKEN is not None:
            os.environ["CIVIT_API_TOKEN"] = self.CIVIT_API_TOKEN

        if self.max_lora_cache_size and os.getenv("AIWORKER_LORA_CACHE_SIZE") is None:
            os.environ["AIWORKER_LORA_CACHE_SIZE"] = str(self.max_lora_cache_size * 1024)

        if self.load_large_models:
            os.environ["AI_HORDE_MODEL_META_LARGE_MODELS"] = "1"

    def save(self, file_path: str) -> None:
        """Save the config model to a file.

        Args:
            file_path (str): The path to the file to save the config model to.
        """
        if self._yaml_loader is None:
            self._yaml_loader = YAML()

        with open(file_path, "w", encoding="utf-8") as f:
            self._yaml_loader.dump(self.model_dump(), f)


# Dynamically add descriptions to the fields of the model
for field_name, field in reGenBridgeData.model_fields.items():
    if field_name in BRIDGE_DATA_FIELD_DESCRIPTIONS:
        field.description = BRIDGE_DATA_FIELD_DESCRIPTIONS[field_name]
