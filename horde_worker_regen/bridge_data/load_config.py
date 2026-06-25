"""Contains methods for loading the config file."""

import json
import os
import re
from collections.abc import Iterable, Mapping
from enum import auto
from pathlib import Path

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPIManualClient
from horde_sdk.worker.dispatch.ai_horde.bridge_data import MetaInstruction
from horde_sdk.worker.model_meta import ImageModelLoadResolver
from loguru import logger
from ruamel.yaml import YAML
from strenum import StrEnum

from horde_worker_regen.bridge_data import AIWORKER_REGEN_PREFIX
from horde_worker_regen.bridge_data.beta_source import beta_aware_image_records
from horde_worker_regen.bridge_data.data_model import reGenBridgeData


def _make_image_model_load_resolver(
    horde_model_reference_manager: ModelReferenceManager,
) -> ImageModelLoadResolver:
    """Construct an ``ImageModelLoadResolver`` over the worker's own reference manager.

    Injecting the already-initialised manager makes the SDK read the reference the parent already holds
    instead of building (and network-prefetching) its own. It also bypasses the SDK's internal
    ``asyncio.run()``, so this is safe to call from inside the worker's running reload loop with no
    worker-thread workaround: a config *reload* used to need a throwaway thread only because the SDK
    constructor could not call ``asyncio.run()`` under a running loop, which injection sidesteps.

    Injection requires ``horde_sdk`` new enough to accept the manager; that floor is enforced by the
    dependency pin, not tolerated at runtime, so a skewed install fails loudly rather than silently
    degrading to a network fetch.
    """
    return ImageModelLoadResolver(horde_model_reference_manager)


def _meta_instruction_matches_record(instruction: str, record: ImageGenerationModelRecord) -> bool:
    """Whether a single meta instruction selects ``record`` from the beta (pending-queue) records.

    Mirrors the reference-driven instruction families of ``horde_sdk``'s ``ImageModelLoadResolver`` so
    opted-in beta models are picked up by the same meta instructions the canonical resolver applies.
    The usage-stats families (``TOP N`` / ``BOTTOM N``) are not mirrored: those resolve against the
    horde's stats API, which already returns names for any served model regardless of beta status.
    """

    def matches(pattern: str) -> bool:
        return re.match(pattern, instruction, re.IGNORECASE) is not None

    if matches(MetaInstruction.ALL_REGEX):
        # "all" excludes the heavy baselines unless the operator opted into large models, matching the
        # SDK's remove_large_models so a beta flux/cascade is not silently bulk-loaded.
        if os.getenv("AI_HORDE_MODEL_META_LARGE_MODELS"):
            return True
        return record.baseline not in (
            KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade,
            KNOWN_IMAGE_GENERATION_BASELINE.flux_1,
        )
    if matches(MetaInstruction.ALL_SDXL_REGEX):
        return record.baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl
    if matches(MetaInstruction.ALL_SD15_REGEX):
        return record.baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1
    if matches(MetaInstruction.ALL_SD21_REGEX):
        return record.baseline in (
            KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_512,
            KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_768,
        )
    if matches(MetaInstruction.ALL_INPAINTING_REGEX):
        return bool(record.inpainting)
    if matches(MetaInstruction.ALL_SFW_REGEX):
        return record.nsfw is False
    if matches(MetaInstruction.ALL_NSFW_REGEX):
        return record.nsfw is True
    return False


def _beta_models_for_meta_instructions(
    instructions: Iterable[str],
    beta_records: Mapping[str, ImageGenerationModelRecord],
) -> set[str]:
    """Resolve meta instructions against beta (pending-queue) image records.

    The SDK resolver only sees the canonical reference, so a meta instruction like ``all`` would
    silently exclude beta models such as Z-Image. Apply the same instruction families to the
    beta-merged records so opted-in beta models are selected alongside the canonical ones.
    """
    matched: set[str] = set()
    for instruction in instructions:
        for name, record in beta_records.items():
            if _meta_instruction_matches_record(instruction, record):
                matched.add(name)
    return matched


class UnsupportedConfigFormat(Exception):
    """The config file format is not supported."""

    def __init__(self, file_path: str | Path, file_format: str) -> None:
        """Initialise the exception."""
        super().__init__(f"Unsupported config file format: {file_format} ({file_path})")

    @staticmethod
    def load_from_env_vars(
        *,
        horde_model_reference_manager: ModelReferenceManager,
    ) -> reGenBridgeData:
        """Checks for AIWORKER_REGEN_* format environment variables and loads the config from them."""
        raw_config: dict[str, str] = {}
        for key, value in os.environ.items():
            if key.startswith(AIWORKER_REGEN_PREFIX):
                # Coverts the env var name to the attr name found in the reGenBridgeData model
                raw_config[key[len(AIWORKER_REGEN_PREFIX) :].lower()] = value

        config: dict[str, object] = {}

        for key, value in raw_config.items():
            attr_name = key[len(AIWORKER_REGEN_PREFIX) :].lower()
            if value.lower() in {"true", "false"}:
                config[attr_name] = value.lower() == "true"
            elif any(delimiter in value for delimiter in ["[", ",", ";"]):
                if "[" in value and "]" not in value:
                    raise ValueError(f"Invalid list format for {attr_name}. Missing closing bracket.")
                value_as_list = re.split(r"[\[\],;]", value.strip("[]"))
                config[attr_name] = [item.strip().strip("'").strip('"') for item in value_as_list]
                logger.debug(f"Converted {attr_name} to list: {config[attr_name]} from {value}")
            else:
                config[attr_name] = value

        bridge_data = reGenBridgeData.model_validate(config)

        for set_field in bridge_data.model_fields_set:
            logger.warning(f"AIWORKER_REGEN_{set_field} environment variable set.")

        bridge_data.image_models_to_load = BridgeDataLoader._resolve_meta_instructions(
            bridge_data,
            horde_model_reference_manager,
        )

        return bridge_data

    @staticmethod
    def write_bridge_data_as_dot_env_file(bridge_data: reGenBridgeData, file_path: str | Path) -> None:
        """Write the bridge data to a .env file.

        Args:
            bridge_data (reGenBridgeData): The bridge data to write to the .env file.
            file_path (str | Path): The path to the .env file to write the bridge data to.
        """
        file_path = Path(file_path)

        with open(file_path, "w", encoding="utf-8") as f:
            for field_name, _ in reGenBridgeData.model_fields.items():
                if field_name in bridge_data.model_fields_set:
                    f.write(f"AIWORKER_REGEN_{field_name.upper()}={getattr(bridge_data, field_name)}\n")


class ConfigFormat(StrEnum):
    """The format of the config file."""

    yaml = auto()
    json = auto()


class BridgeDataLoader:
    """Contains methods for loading the config file."""

    @staticmethod
    def _infer_format(file_path: str | Path) -> ConfigFormat:
        """Infer the config file format from the file extension.

        Args:
            file_path (str | Path): The path to the config file.

        Returns:
            ConfigFormat: The config file format.

        Raises:
            UnsupportedConfigFormat: If the config file format is not supported.
        """
        file_path = Path(file_path)

        if file_path.suffix == ".yaml" or file_path.suffix == ".yml":
            return ConfigFormat.yaml

        if file_path.suffix == ".json":
            return ConfigFormat.json

        raise UnsupportedConfigFormat(file_path, file_path.suffix)

    @staticmethod
    def load(
        file_path: str | Path,
        *,
        file_format: ConfigFormat | None = None,
        horde_model_reference_manager: ModelReferenceManager | None = None,
    ) -> reGenBridgeData:
        """Load the config file and validate it.

        Args:
            file_path (str | Path): The path to the config file.
            file_format (ConfigFormat | None, optional): The config file format. Defaults to None. \
            The file format will be inferred from the file extension if not provided.
            horde_model_reference_manager (ModelReferenceManager | None, optional): The model reference manager. \
            Used to resolve meta instructions. Defaults to None.

        Returns:
            reGenBridgeData: The validated config file.

        Raises:
            ValidationError: If the config file is invalid.
            UnsupportedConfigFormat: If the config file format is not supported.

        """
        file_path = Path(file_path)
        # Infer the file format if not provided
        if not file_format:
            file_format = BridgeDataLoader._infer_format(file_path)

        bridge_data: reGenBridgeData | None = None

        if file_format == ConfigFormat.yaml:
            yaml = YAML()
            with open(file_path, encoding="utf-8") as f:
                config = yaml.load(f)

            bridge_data = reGenBridgeData.model_validate(config)
            if bridge_data is not None:
                bridge_data._yaml_loader = yaml

        if file_format == ConfigFormat.json:
            with open(file_path, encoding="utf-8") as f:
                config = json.load(f)

            bridge_data = reGenBridgeData.model_validate(config)

        if not bridge_data:
            raise UnsupportedConfigFormat(file_path, file_format)

        if not horde_model_reference_manager:
            logger.warning(
                "No model reference manager provided. The config file will not be able to resolve meta instructions.",
            )
            return bridge_data

        bridge_data.image_models_to_load = BridgeDataLoader._resolve_meta_instructions(
            bridge_data,
            horde_model_reference_manager,
        )

        reGenBridgeData.load_custom_models()

        return bridge_data

    @staticmethod
    def load_from_env_vars(
        *,
        horde_model_reference_manager: ModelReferenceManager | None = None,
    ) -> reGenBridgeData:
        """Checks for AIWORKER_REGEN_* format environment variables and loads the config from them."""
        config: dict[str, object] = {}

        for key, value in os.environ.items():
            if key.startswith(AIWORKER_REGEN_PREFIX):
                # Coverts the env var name to the attr name found in the reGenBridgeData model
                attr_name = key[len(AIWORKER_REGEN_PREFIX) :].lower()
                if value.lower() in ("true", "false"):
                    config[attr_name] = value.lower() == "true"
                elif any(delimiter in value for delimiter in ["[", ",", ";"]):
                    if "[" in value and "]" not in value:
                        raise ValueError(f"Invalid list format for {attr_name}. Missing closing bracket.")
                    value_as_list = re.split(r"[\[\],;]", value.strip("[]"))
                    config[attr_name] = [item.strip().strip("'").strip('"') for item in value_as_list]
                    logger.debug(f"Converted {attr_name} to list: {config[attr_name]} from {value}")
                else:
                    config[attr_name] = value

        for field_name, field_info in reGenBridgeData.model_fields.items():
            if field_info.alias is not None and field_name in config:
                config[field_info.alias] = config.pop(field_name)
                logger.warning(
                    f"Config `{field_name}` was set by an environment variable. "
                    f"However, it is an aliased field in the config file. "
                    f"Renaming to `{field_info.alias}`.",
                )

        # Load the config
        bridge_data = reGenBridgeData.model_validate(config)

        for set_field in bridge_data.model_fields_set:
            if bridge_data.model_extra is not None and set_field in bridge_data.model_extra:
                logger.warning(
                    f"Config `{set_field}` was set by an environment variable. "
                    f"However, it is not a valid field in the config file.",
                )
            logger.info(f"Config `{set_field}` was set by an environment variable.")

        if horde_model_reference_manager is not None:
            bridge_data.image_models_to_load = BridgeDataLoader._resolve_meta_instructions(
                bridge_data,
                horde_model_reference_manager,
            )

        bridge_data.load_env_vars()
        bridge_data._loaded_from_env_vars = True
        return bridge_data

    @staticmethod
    def write_bridge_data_as_dot_env_file(bridge_data: reGenBridgeData, file_path: str | Path) -> None:
        """Write the bridge data to a .env file.

        Args:
            bridge_data (reGenBridgeData): The bridge data to write to the .env file.
            file_path (str | Path): The path to the .env file to write the bridge data to.
        """
        file_path = Path(file_path)

        with open(file_path, "w", encoding="utf-8") as f:
            for field_name, _ in reGenBridgeData.model_fields.items():
                if field_name in bridge_data.model_fields_set:
                    field_info = reGenBridgeData.model_fields[field_name]
                    config_field_value = getattr(bridge_data, field_name)
                    if config_field_value == field_info.default:
                        continue

                    field_alias = field_info.alias

                    f.write(
                        f"{AIWORKER_REGEN_PREFIX}{field_name.upper() if field_alias is None else field_alias.upper()}"
                        f"={config_field_value}\n",
                    )

    @staticmethod
    def _resolve_meta_instructions(  # FIXME: This should be moved into the SDK
        bridge_data: reGenBridgeData,
        horde_model_reference_manager: ModelReferenceManager,
    ) -> list[str]:
        """Resolve the meta instructions in the bridge data. Note that this modifies the bridge data in place.

        Args:
            bridge_data (reGenBridgeData): The bridge data.
            horde_model_reference_manager (ModelReferenceManager): The model reference manager.

        Returns:
            list[str]: The image models that will be loaded.
        """
        load_resolver = _make_image_model_load_resolver(horde_model_reference_manager)

        # Reconcile the SDK's env-var transport to the config here, the one chokepoint every resolution path
        # flows through (startup, env-var config, and config reload). The SDK's remove_large_models gates the
        # `all` instruction on AI_HORDE_MODEL_META_LARGE_MODELS being unset regardless of the param below, and
        # that env var is never otherwise cleared, so a reload from large-models-on to off (e.g. via the TUI)
        # would keep the stale value and keep loading Flux/Stable Cascade. Make the config authoritative every
        # time models are resolved.
        if bridge_data.load_large_models:
            os.environ["AI_HORDE_MODEL_META_LARGE_MODELS"] = "1"
        elif os.environ.pop("AI_HORDE_MODEL_META_LARGE_MODELS", None) is not None:
            logger.warning(
                "AI_HORDE_MODEL_META_LARGE_MODELS was set but `load_large_models` is false; clearing it so "
                "large models (e.g. Flux, Stable Cascade) are not loaded.",
            )

        # The SDK resolver only sees the canonical reference (get_all_model_references never includes a
        # PRIMARY's pending-queue / beta models), so anything beta would be silently dropped both from the
        # meta-instruction expansion below and the known-models filter further down. Build the beta-merged
        # records once and use them to keep opted-in beta models (e.g. qwen, Z-Image) advertised.
        beta_records = beta_aware_image_records(horde_model_reference_manager)

        # Pass the config flag through so the resolver strips large baselines (Flux, Stable Cascade) for the
        # stats-based families (`top N`/`bottom N`) too, not only the `all` instruction. Without this the SDK
        # param defaults to True and those families could surface a large model even with the flag off.
        resolved_models = None
        if bridge_data.meta_load_instructions is not None:
            resolved_models = load_resolver.resolve_meta_instructions(
                list(bridge_data.meta_load_instructions),
                AIHordeAPIManualClient(),
                load_large_models=bridge_data.load_large_models,
            )
            resolved_models |= _beta_models_for_meta_instructions(bridge_data.meta_load_instructions, beta_records)

        if bridge_data.meta_skip_instructions is not None:
            skip_models: set[str] = load_resolver.resolve_meta_instructions(
                list(bridge_data.meta_skip_instructions),
                AIHordeAPIManualClient(),
                load_large_models=bridge_data.load_large_models,
            )
            skip_models |= _beta_models_for_meta_instructions(bridge_data.meta_skip_instructions, beta_records)
            existing_skip_models = set(bridge_data.image_models_to_skip)
            bridge_data.image_models_to_skip = list(existing_skip_models.union(skip_models))

        if resolved_models is not None:
            bridge_data.image_models_to_load = list(set(bridge_data.image_models_to_load + list(resolved_models)))

        if bridge_data.image_models_to_skip is not None and len(bridge_data.image_models_to_skip) > 0:
            bridge_data.image_models_to_load = list(
                set(bridge_data.image_models_to_load) - set(bridge_data.image_models_to_skip),
            )

        # Remove models not in the model reference manager (canonical + opted-in beta)
        known_models = set(beta_records)

        total_resolved_models = len(bridge_data.image_models_to_load)

        bridge_data.image_models_to_load = list(set(bridge_data.image_models_to_load) & known_models)

        used_models = len(bridge_data.image_models_to_load)

        if total_resolved_models != used_models:
            logger.debug(
                f"Resolved {total_resolved_models} models, but only {used_models} "
                "are available in the model reference manager.",
            )

        if bridge_data.only_models_on_disk:
            bridge_data.image_models_to_load = BridgeDataLoader._filter_to_models_on_disk(
                bridge_data.image_models_to_load,
                horde_model_reference_manager,
            )

        return bridge_data.image_models_to_load

    @staticmethod
    def _filter_to_models_on_disk(
        model_names: list[str],
        horde_model_reference_manager: ModelReferenceManager,
    ) -> list[str]:
        """Drop any resolved model whose files are not already present on disk (no download is queued).

        Honors ``only_models_on_disk``: presence is an existence-only check against the same weights
        roots the download planner uses, so the served set is pinned to what the operator already has.
        """
        from horde_model_reference import MODEL_REFERENCE_CATEGORY

        from horde_worker_regen.model_download_plan import is_model_present

        references = horde_model_reference_manager.get_all_model_references()
        image_records = references.get(MODEL_REFERENCE_CATEGORY.image_generation) or {}

        on_disk = [name for name in model_names if is_model_present(name, image_records)]

        dropped = len(model_names) - len(on_disk)
        if dropped:
            logger.info(
                f"only_models_on_disk is set: dropped {dropped} model(s) not present on disk; "
                f"keeping {len(on_disk)} already-downloaded model(s).",
            )

        return on_disk
