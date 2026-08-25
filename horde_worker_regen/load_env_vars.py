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

    with open(config_file, encoding="utf-8") as f:
        # Backslashes are ordinary characters in YAML plain scalars and are native path separators on
        # Windows. The former raw-text preflight rejected every backslash before YAML parsing, including
        # valid custom-model filepaths written by the dashboard, and terminated startup via SystemExit.
        # Let the YAML parser distinguish valid plain scalars from malformed quoted escape sequences.
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
    apply_huggingface_cache_isolation()

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

    # Expose the worker's key to the model-download path so hordelib can fetch hostable models from the gated
    # R2 mirror. The download subprocess inherits this env; the anonymous key cannot be trusted, so it is left
    # unset (the engine then downloads straight from each model's origin host).
    configured_api_key = config.get("api_key")
    if configured_api_key and configured_api_key != "0000000000":
        if os.getenv("AIHORDE_API_KEY") is None:
            os.environ["AIHORDE_API_KEY"] = configured_api_key
        else:
            print(
                "AIHORDE_API_KEY environment variable already set. "
                "This will override the value for `api_key` in the config file.",
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

    # The config field is authoritative for large-model loading (see data_model.py's load_env_vars). Set the
    # env var when the config opts in, and clear any pre-existing value when it opts out, so a stale env var
    # (a prior True run, or an exported shell/Docker value) cannot silently defeat `load_large_models: false`.
    if "load_large_models" in config:
        if config["load_large_models"] is True:
            os.environ["AI_HORDE_MODEL_META_LARGE_MODELS"] = "1"
        elif os.environ.pop("AI_HORDE_MODEL_META_LARGE_MODELS", None) is not None:
            logger.warning(
                "AI_HORDE_MODEL_META_LARGE_MODELS was set but `load_large_models` is false; clearing it so "
                "large models (e.g. Flux, Stable Cascade) are not loaded.",
            )

    if "limited_console_messages" in config and os.getenv("AIWORKER_LIMITED_CONSOLE_MESSAGES") is None:
        config_value = config["limited_console_messages"]
        if config_value is True:
            os.environ["AIWORKER_LIMITED_CONSOLE_MESSAGES"] = "1"

    apply_beta_model_env_defaults(config.get("api_key"))


LEGACY_HUB_CACHES_ENV_VAR = "AIWORKER_HF_LEGACY_HUB_CACHES"
"""Where the HuggingFace hub cache resolved before isolation took over, ``os.pathsep``-joined.

Set by :func:`apply_huggingface_cache_isolation` for the download process, which moves the annotator
entries a worker fetched into one of these locations before the cache was isolated."""


def _legacy_hub_cache_dirs(*, target_hub_dir: str) -> list[str]:
    """The hub cache directories a pre-isolation worker may have populated, excluding the target itself.

    An ambient ``HF_HUB_CACHE``/``HUGGINGFACE_HUB_CACHE`` names the hub directory outright; an ambient
    ``HF_HOME`` holds it under ``hub``; and with none of them set the hub used its own default under the
    user's cache directory (``XDG_CACHE_HOME`` or ``~/.cache``).
    """
    candidates: list[str] = []
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(name)
        if value:
            candidates.append(value)
    ambient_home = os.environ.get("HF_HOME")
    if ambient_home:
        candidates.append(os.path.join(ambient_home, "hub"))
    user_cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    candidates.append(os.path.join(user_cache, "huggingface", "hub"))
    target = os.path.normcase(os.path.normpath(target_hub_dir))
    seen: set[str] = set()
    legacy: list[str] = []
    for candidate in candidates:
        key = os.path.normcase(os.path.normpath(candidate))
        if key == target or key in seen:
            continue
        seen.add(key)
        legacy.append(candidate)
    return legacy


HUGGING_FACE_CACHE_ENV_VARS: tuple[str, ...] = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
"""The variables the HuggingFace stack reads its cache location from; the later two outrank ``HF_HOME``."""

HUGGINGFACE_HOME_DIRNAME = "hf_transformers"
"""The worker-wide HuggingFace home under ``AIWORKER_CACHE_HOME``.

The name is ``horde_safety``'s: it has always put the safety models' transformers cache there when it found a
cache root and no ``HF_HOME``, so every existing worker already holds those weights at this location. Sharing
it, rather than naming a second isolated directory, is what lets the whole worker use one hub cache without
any process fetching again what another already has."""


def huggingface_home_for(cache_home: str) -> str:
    """The worker-wide ``HF_HOME`` for a cache root."""
    return os.path.join(cache_home, HUGGINGFACE_HOME_DIRNAME)


def apply_huggingface_cache_isolation() -> None:
    """Point the HuggingFace stack at the worker-wide cache under ``AIWORKER_CACHE_HOME``, process-wide.

    ``AIWORKER_CACHE_HOME`` is the operator's promise that every model file the worker fetches lands under one
    root. The transformers-backed ControlNet annotators (MiDaS, ZoeDepth, depth-anything, OneFormer) fetch
    through the HuggingFace hub cache, which defaults to the home drive and is outranked by an ambient
    ``HF_HUB_CACHE``/``HUGGINGFACE_HUB_CACHE``, so those variables are replaced rather than merely warned
    about. Applied here, before any child spawns, so the download, inference and safety processes share one
    cache (:data:`HUGGINGFACE_HOME_DIRNAME`); a process that resolved its own would keep a second cache on
    another volume, and an annotator verified in one process would then be fetched again by the next. The
    utilities process keeps its own isolated location by its own policy.

    A no-op without ``AIWORKER_CACHE_HOME``, in which case the hub stack keeps its own defaults.
    """
    cache_home = os.environ.get("AIWORKER_CACHE_HOME")
    if not cache_home:
        return
    huggingface_home = huggingface_home_for(cache_home)
    overridden = [name for name in HUGGING_FACE_CACHE_ENV_VARS if os.environ.get(name) not in (None, huggingface_home)]
    legacy_hub_dirs = _legacy_hub_cache_dirs(target_hub_dir=os.path.join(huggingface_home, "hub"))
    for name in HUGGING_FACE_CACHE_ENV_VARS:
        os.environ.pop(name, None)
    os.environ["HF_HOME"] = huggingface_home
    os.environ[LEGACY_HUB_CACHES_ENV_VAR] = os.pathsep.join(legacy_hub_dirs)
    if overridden:
        logger.warning(
            f"Overriding ambient {', '.join(overridden)}: AIWORKER_CACHE_HOME isolation owns the HuggingFace "
            f"cache location ({huggingface_home}).",
        )


def apply_beta_model_env_defaults(api_key: str | None = None) -> None:
    """Opt every worker (and the TUI host, which reuses this) into the image-generation, esrgan and gfpgan beta.

    Beta models (e.g. qwen, the modern upscalers, and the modern face restorers) live in the
    model-reference PRIMARY's pending queue rather than the canonical reference, so surfacing one requires
    both hordelib's beta opt-in env vars and a PRIMARY URL to read the pending queue from (see
    ``hordelib.beta_models``). The esrgan and gfpgan categories are opted in alongside image_generation so a
    worker can serve the new upscalers and face restorers the moment the AI-Horde server advertises their
    names (the worker withholds offering them until then). The gfpgan category also carries RestoreFormer,
    which shares the face-restoration on-disk folder. Reading the pending queue only
    needs a reader-level key, which any AI-Horde key satisfies, including the anonymous ``"0000000000"``;
    callers pass the worker's own ``api_key`` when one is configured, otherwise the anonymous key is used.

    Every value is applied with ``setdefault`` so an operator who set any of these explicitly wins,
    including opting back out by exporting ``HORDELIB_BETA_MODEL_CATEGORIES=""`` (an empty value is
    still "set", so the default below does not clobber it, and hordelib treats empty as disabled).

    The env-var names are mirrored as literals rather than imported from ``hordelib.beta_models``
    because this runs in the torch-free orchestrator before any subprocess spawns, and importing
    hordelib here would eagerly drag in torch.

    Args:
        api_key: A reader-level AI-Horde key for the pending-queue reads. Falls back to the anonymous key.
    """
    # Mirrors hordelib.beta_models.BETA_CATEGORIES_ENV_VAR / BETA_API_KEY_ENV_VAR.
    os.environ.setdefault("HORDELIB_BETA_MODEL_CATEGORIES", "image_generation,esrgan,gfpgan")
    os.environ.setdefault("HORDELIB_BETA_MODELS_API_KEY", api_key or "0000000000")
    os.environ.setdefault("HORDE_MODEL_REFERENCE_PRIMARY_API_URL", "https://models.aihorde.net/api")


if __name__ == "__main__":
    load_env_vars_from_config()
    logger.info("Environment variables loaded.")
