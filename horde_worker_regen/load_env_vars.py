"""Contains the functions to load the environment variables from the config file."""

import os
import pathlib

from dotenv import load_dotenv
from loguru import logger
from ruamel.yaml import YAML

load_dotenv()


def load_env_vars_from_config() -> None:  # FIXME: there is a dynamic way to do this
    """Load the environment variables from the config file."""
    yaml = YAML()
    config_file = "bridgeData.yaml"
    template_file = "bridgeData_template.yaml"

    if not pathlib.Path(config_file).exists():
        if pathlib.Path(template_file).exists():
            raise FileNotFoundError(f"{template_file} found. Please set variables and rename it to {config_file}.")
        raise FileNotFoundError(f"{config_file} not found")

    # Users on windows occasionally use backslashes in their paths, which causes issues on loading.
    # We're going to load the file as text and print the lines with backslashes to the user, and instruct them to
    # replace them with forward slashes.

    with open(config_file, encoding="utf-8") as f:
        lines = f.readlines()
        found_backslashes = False
        for line in lines:
            if "\\" in line:
                print(f"Backslashes found in the following line:\n{line}")
                found_backslashes = True

                print(
                    "Please replace backslashes with forward slashes in the config file, "
                    "as backslashes are not supported.",
                )

                corrected_line = line.replace("\\", "/")
                print(f"Corrected line:\n{corrected_line}")

    if found_backslashes:
        import sys

        sys.exit(1)

    with open(config_file, encoding="utf-8") as f:
        config = yaml.load(f)

    # See data_model.py's `def load_env_vars(self) -> None:`
    if "cache_home" in config:
        if os.getenv("AIWORKER_CACHE_HOME") is None:
            os.environ["AIWORKER_CACHE_HOME"] = config["cache_home"]
        else:
            print(
                "AIWORKER_CACHE_HOME environment variable already set. "
                "This will override the value for `cache_home` in the config file.",
            )

    # Peered-data fallback, applied at the LOWEST precedence: the scripted installers run the worker from a
    # runtime shim that exports HORDE_WORKER_DATA_DIR (the sibling <worker>-data folder preserved across
    # reinstalls) but deliberately do NOT pre-set AIWORKER_CACHE_HOME, so a user-set env var and a config
    # `cache_home` both win over this. Only when neither supplied a model location do we default models into
    # <data>/models so a fresh install reuses previously downloaded weights instead of re-downloading them.
    if os.getenv("AIWORKER_CACHE_HOME") is None:
        data_dir = os.getenv("HORDE_WORKER_DATA_DIR")
        if data_dir:
            os.environ["AIWORKER_CACHE_HOME"] = os.path.join(data_dir, "models")

    if "max_lora_cache_size" in config:
        if os.getenv("AIWORKER_LORA_CACHE_SIZE") is None:
            try:
                cache_size_gb = int(config["max_lora_cache_size"])
            except ValueError as e:
                raise ValueError(
                    "max_lora_cache_size must be an integer, but is not.",
                ) from e
            # max_lora_cache_size is gigabytes; hordelib reads AIWORKER_LORA_CACHE_SIZE as megabytes.
            # This must match data_model.py's load_env_vars conversion so the two paths agree.
            os.environ["AIWORKER_LORA_CACHE_SIZE"] = str(cache_size_gb * 1024)
        else:
            print(
                "AIWORKER_LORA_CACHE_SIZE environment variable already set. "
                "This will override the value for `max_lora_cache_size` in the config file.",
            )
    if "min_lora_disk_free_gb" in config and os.getenv("AIWORKER_LORA_MIN_DISK_FREE_MB") is None:
        try:
            min_free_gb = float(config["min_lora_disk_free_gb"])
        except (ValueError, TypeError) as e:
            raise ValueError(
                "min_lora_disk_free_gb must be a number, but is not.",
            ) from e
        os.environ["AIWORKER_LORA_MIN_DISK_FREE_MB"] = str(round(min_free_gb * 1024))
    if "civitai_api_token" in config:
        if os.getenv("CIVIT_API_TOKEN") is None:
            os.environ["CIVIT_API_TOKEN"] = config["civitai_api_token"]
        else:
            print(
                "CIVIT_API_TOKEN environment variable already set. "
                "This will override the value for `civitai_api_token` in the config file.",
            )

    if "horde_url" in config:
        known_ai_horde_urls = [
            "stablehorde.net",
            "aihorde.net",
        ]

        custom_horde_url = config["horde_url"]
        AI_HORDE_URL = os.getenv("AI_HORDE_URL")
        if custom_horde_url and any(url in custom_horde_url for url in known_ai_horde_urls):
            if AI_HORDE_URL is None or not AI_HORDE_URL:
                logger.debug("Using default AI Horde URL.")
        else:
            logger.warning(
                f"Using custom AI Horde URL `{custom_horde_url}`. Make sure this is correct and ends in `/api/`.",
            )
            os.environ["AI_HORDE_URL"] = custom_horde_url

    if "load_large_models" in config and os.getenv("AI_HORDE_MODEL_META_LARGE_MODELS") is None:
        config_value = config["load_large_models"]
        if config_value is True:
            os.environ["AI_HORDE_MODEL_META_LARGE_MODELS"] = "1"

    if "limited_console_messages" in config and os.getenv("AIWORKER_LIMITED_CONSOLE_MESSAGES") is None:
        config_value = config["limited_console_messages"]
        if config_value is True:
            os.environ["AIWORKER_LIMITED_CONSOLE_MESSAGES"] = "1"

    _enable_beta_models(config)


def _enable_beta_models(config: dict) -> None:
    """Opt every worker into the image-generation beta (e.g. qwen) by default.

    Beta models live in the model-reference PRIMARY's pending queue rather than the canonical
    reference, so surfacing one (such as qwen) requires both hordelib's beta opt-in env vars and a
    PRIMARY URL to read the pending queue from (see ``hordelib.beta_models``). Reading the pending
    queue only needs a reader-level key, which any AI-Horde key satisfies, including the anonymous
    ``"0000000000"``; we reuse the worker's own ``api_key`` when one is configured.

    Every value is applied with ``setdefault`` so an operator who set any of these explicitly wins,
    including opting back out by exporting ``HORDELIB_BETA_MODEL_CATEGORIES=""`` (an empty value is
    still "set", so the default below does not clobber it, and hordelib treats empty as disabled).

    The env-var names are mirrored as literals rather than imported from ``hordelib.beta_models``
    because this runs in the torch-free orchestrator before any subprocess spawns, and importing
    hordelib here would eagerly drag in torch.
    """
    # Mirrors hordelib.beta_models.BETA_CATEGORIES_ENV_VAR / BETA_API_KEY_ENV_VAR.
    os.environ.setdefault("HORDELIB_BETA_MODEL_CATEGORIES", "image_generation")
    os.environ.setdefault("HORDELIB_BETA_MODELS_API_KEY", config.get("api_key") or "0000000000")
    os.environ.setdefault("HORDE_MODEL_REFERENCE_PRIMARY_API_URL", "https://models.aihorde.net/api")


if __name__ == "__main__":
    load_env_vars_from_config()
    logger.info("Environment variables loaded.")
