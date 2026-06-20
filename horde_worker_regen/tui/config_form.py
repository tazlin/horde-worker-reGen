"""Curated bridgeData.yaml field catalog plus light (ruamel) read/write for the config editor.

The editor works on the raw YAML, not ``reGenBridgeData``, so the TUI parent stays free of the heavy
``horde_sdk`` import chain. Comments and untouched keys are preserved; authoritative schema validation
happens worker-side when the worker reloads the file (errors surface in the Logs view). Field help,
bounds, and grouping come from bridgeData_template.yaml and the SDK's field constraints.

Fields that are obsolete or marked "Currently unused in reGen" (and the Scribe worker fields) are
intentionally omitted: showing controls that do nothing would mislead, not help.
"""

from __future__ import annotations

import dataclasses
import enum
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

DEFAULT_CONFIG_PATH = Path("bridgeData.yaml")

# YAML key names for the two model-list fields the dedicated editor handles.
MODELS_TO_LOAD_KEY = "models_to_load"
MODELS_TO_SKIP_KEY = "models_to_skip"

# Alchemy forms a worker may offer (template spelling, hyphenated).
ALCHEMY_FORMS = ("caption", "nsfw", "interrogation", "post-process")


class FieldKind(enum.StrEnum):
    """How a config field is edited and coerced."""

    BOOL = "bool"
    INT = "int"
    STR = "str"
    STR_LIST = "str_list"
    MODEL_LIST = "model_list"
    """A models_to_load/skip list, edited via the dedicated model-list control."""
    SELECT_MULTI = "select_multi"
    """A fixed set of multi-selectable string choices (e.g. alchemy forms)."""


# Sentinel for "no explicit default declared" so that a legitimate falsy explicit default
# (False, 0, "") is still honored. The kind-based fallback only applies when this is unchanged.
_UNSET: Any = object()


@dataclasses.dataclass(frozen=True)
class ConfigField:
    """One editable bridgeData field: its YAML key, presentation, and edit semantics."""

    key: str
    label: str
    kind: FieldKind
    section: str
    help: str = ""
    requires_restart: bool = False
    secret: bool = False
    minimum: int | None = None
    maximum: int | None = None
    unit: str = ""
    choices: tuple[str, ...] = ()
    explicit_default: Any = _UNSET
    """The worker's real default when the key is absent, when it differs from the kind-based fallback.

    The editor shows this as the field's value when the YAML omits the key, so the displayed value
    matches what the worker (``reGenBridgeData``) would actually use. Required because the kind-based
    fallback (BOOL->False, INT->minimum) silently disagrees with model fields that default True or to a
    non-minimum number, which would mislead the operator. Enforced against the model by
    ``tests/tui/test_config_form_defaults.py``.
    """

    def default(self) -> Any:  # noqa: ANN401 - heterogeneous defaults by kind
        """The value used when the key is absent from the file."""
        if self.explicit_default is not _UNSET:
            return self.explicit_default
        if self.kind is FieldKind.BOOL:
            return False
        if self.kind is FieldKind.INT:
            return self.minimum if self.minimum is not None else 0
        if self.kind in (FieldKind.STR_LIST, FieldKind.MODEL_LIST, FieldKind.SELECT_MULTI):
            return []
        return ""


# Section order is the display order.
SECTIONS = (
    "Connection",
    "Identity",
    "Throughput",
    "Memory & performance",
    "Content & safety",
    "Features",
    "LoRA",
    "Models",
    "Alchemist",
    "Other",
)

# Sub-tab grouping for the config editor: each tab bundles related sections so no single page
# requires long scrolling. Order is the tab order; "Models" is its own tab (the unified panel).
CONFIG_SUBTABS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Essentials", ("Connection", "Identity")),
    ("Models", ("Models",)),
    ("Performance", ("Throughput", "Memory & performance")),
    ("Content", ("Content & safety", "Features")),
    # LoRA and Alchemy are logically distinct concerns (one is an image-job feature, the other a separate
    # worker role), so each gets its own sub-tab rather than sharing a crowded combined page.
    ("LoRA", ("LoRA",)),
    ("Alchemy", ("Alchemist",)),
    ("Advanced", ("Other",)),
)

SECTION_GUIDANCE: dict[str, str] = {
    "Throughput": "Bounds are enforced (max_threads 1–16, queue_size 0–4, max_batch 1–20, max_power 1–512). "
    "See the suggested values per GPU tier in the README.",
    "Models": "Edit the load/skip rules below; the panel previews exactly which models will load and "
    "their disk cost. Press Resolve to expand 'top N' / 'bottom N' commands (needs usage stats).",
    "LoRA": "Allowing LoRA downloads them on demand; set a civitai_api_token for resources that require it.",
    "Alchemist": "Alchemy is a separate worker role (interrogation / post-processing), distinct from LoRA. "
    "Enabling it serves alchemy jobs alongside (or instead of) image generation.",
}

# Curated against bridgeData_template.yaml key names (note: dreamer_name, allow_painting, cache_home).
CONFIG_FIELDS: list[ConfigField] = [
    # Connection
    ConfigField(
        "api_key",
        "API key",
        FieldKind.STR,
        "Connection",
        "Your AI Horde API key (register at aihorde.net/register).",
        requires_restart=True,
        secret=True,
    ),
    ConfigField(
        "horde_url",
        "Horde URL",
        FieldKind.STR,
        "Connection",
        "The horde API base URL. Leave default unless using a custom horde.",
        requires_restart=True,
    ),
    ConfigField(
        "priority_usernames",
        "Priority usernames",
        FieldKind.STR_LIST,
        "Connection",
        "Usernames whose requests to prioritise (one per line). Your own is always included.",
    ),
    # Identity
    ConfigField(
        "dreamer_name",
        "Dreamer name",
        FieldKind.STR,
        "Identity",
        "Unique horde-wide name for the image worker. Do not use the default.",
        requires_restart=True,
    ),
    # Throughput
    ConfigField(
        "max_threads",
        "Max threads",
        FieldKind.INT,
        "Throughput",
        "Parallel jobs. Only high-end cards benefit; keep at 1 for xx60/xx70 or 20xx and older.",
        requires_restart=True,
        minimum=1,
        maximum=16,
    ),
    ConfigField(
        "queue_size",
        "Queue size",
        FieldKind.INT,
        "Throughput",
        "Extra jobs buffered. Increases system RAM use significantly; 0–1 for ≤32 GB RAM.",
        requires_restart=True,
        minimum=0,
        maximum=4,
        explicit_default=1,
    ),
    ConfigField(
        "max_batch",
        "Max batch",
        FieldKind.INT,
        "Throughput",
        "Images per batched request. Ensure you can make max_batch at half your max_power.",
        minimum=1,
        maximum=20,
    ),
    ConfigField(
        "max_power",
        "Max power",
        FieldKind.INT,
        "Throughput",
        "Max resolution = 64*64*8*max_power px (8=512², 32=1024²). Higher needs more VRAM.",
        minimum=1,
        maximum=512,
        explicit_default=8,
    ),
    # Memory & performance
    ConfigField(
        "high_memory_mode",
        "High memory mode",
        FieldKind.BOOL,
        "Memory & performance",
        "Use more VRAM to cut model reload time. Recommended for 24 GB+ cards.",
    ),
    ConfigField(
        "high_performance_mode",
        "High performance mode",
        FieldKind.BOOL,
        "Memory & performance",
        "Fill the local queue much faster (24 GB+ cards).",
    ),
    ConfigField(
        "moderate_performance_mode",
        "Moderate performance mode",
        FieldKind.BOOL,
        "Memory & performance",
        "Fill the queue somewhat faster (12–16 GB cards). Overridden by high_performance_mode.",
    ),
    ConfigField(
        "post_process_job_overlap",
        "Post-process overlap",
        FieldKind.BOOL,
        "Memory & performance",
        "Start the next job before the current finishes post-processing (24 GB+ cards).",
    ),
    ConfigField(
        "unload_models_from_vram_often",
        "Unload VRAM often",
        FieldKind.BOOL,
        "Memory & performance",
        "Aggressively free VRAM between jobs. Recommended for cards under 16 GB.",
        explicit_default=True,
    ),
    ConfigField(
        "very_fast_disk_mode",
        "Very fast disk mode",
        FieldKind.BOOL,
        "Memory & performance",
        "Load multiple models off disk at once. Needs a very fast disk; high disk usage.",
    ),
    ConfigField(
        "extra_slow_worker",
        "Extra slow worker",
        FieldKind.BOOL,
        "Memory & performance",
        "For very slow cards (<0.1 mps/s). Triples timeouts; users may opt out.",
    ),
    ConfigField(
        "limit_max_steps",
        "Limit max steps",
        FieldKind.BOOL,
        "Memory & performance",
        "Only take jobs below the model's average step count (good for slow workers).",
    ),
    # Content & safety
    ConfigField(
        "nsfw", "Allow NSFW", FieldKind.BOOL, "Content & safety", "Serve NSFW requests.", explicit_default=True
    ),
    ConfigField(
        "censor_nsfw", "Censor NSFW", FieldKind.BOOL, "Content & safety", "Censor NSFW images even when nsfw is true."
    ),
    ConfigField(
        "blacklist",
        "Prompt blacklist",
        FieldKind.STR_LIST,
        "Content & safety",
        "Reject jobs whose prompt contains any of these words (one per line).",
    ),
    ConfigField(
        "censorlist",
        "Censor list",
        FieldKind.STR_LIST,
        "Content & safety",
        "Always censor these words, even if nsfw is true (one per line).",
    ),
    ConfigField(
        "allow_unsafe_ip",
        "Allow unsafe IPs",
        FieldKind.BOOL,
        "Content & safety",
        "Allow requests from behind VPNs/proxies.",
        explicit_default=True,
    ),
    ConfigField(
        "require_upfront_kudos",
        "Require upfront kudos",
        FieldKind.BOOL,
        "Content & safety",
        "Only serve users who can pay the kudos upfront (excludes anonymous).",
    ),
    # Features
    ConfigField(
        "safety_on_gpu",
        "Safety on GPU",
        FieldKind.BOOL,
        "Features",
        "Run the CSAM/NSFW CLIP check on GPU (~1.2 GB VRAM). Recommended for 12 GB+.",
        requires_restart=True,
    ),
    ConfigField(
        "allow_img2img",
        "Allow img2img",
        FieldKind.BOOL,
        "Features",
        "Accept jobs that supply a source image.",
        explicit_default=True,
    ),
    ConfigField(
        "allow_painting",
        "Allow inpainting",
        FieldKind.BOOL,
        "Features",
        "Accept inpainting jobs (forced off if img2img is off).",
    ),
    ConfigField(
        "allow_post_processing",
        "Allow post-processing",
        FieldKind.BOOL,
        "Features",
        "Accept upscaling / face-fixing / other post-gen features.",
    ),
    ConfigField(
        "allow_controlnet",
        "Allow ControlNet",
        FieldKind.BOOL,
        "Features",
        "Accept ControlNet jobs (extra RAM/VRAM; needs img2img).",
    ),
    ConfigField(
        "allow_sdxl_controlnet",
        "Allow SDXL ControlNet",
        FieldKind.BOOL,
        "Features",
        "Accept SDXL ControlNet/transparency jobs (heavy; requires allow_controlnet).",
    ),
    # LoRA
    ConfigField(
        "allow_lora",
        "Allow LoRA",
        FieldKind.BOOL,
        "LoRA",
        "Accept LoRA jobs. Downloads on demand; set a civitai_api_token. Needs fast internet.",
    ),
    ConfigField(
        "civitai_api_token",
        "Civitai API token",
        FieldKind.STR,
        "LoRA",
        "Token for downloading civitai resources (LoRAs/TIs, and many popular models).",
        secret=True,
    ),
    ConfigField(
        "max_lora_cache_size",
        "LoRA cache size",
        FieldKind.INT,
        "LoRA",
        "Gigabytes of LoRAs to keep cached (minimum 10).",
        minimum=10,
        maximum=2048,
        unit="GB",
    ),
    ConfigField(
        "min_lora_disk_free_gb",
        "Min LoRA disk free",
        FieldKind.INT,
        "LoRA",
        "Keep at least this many GB free on the LoRA cache disk. Below it, the worker evicts old "
        "LoRAs to make room and stops offering LoRAs if it still can't clear the floor. 0 disables.",
        minimum=0,
        maximum=512,
        unit="GB",
        explicit_default=1.0,
    ),
    ConfigField(
        "purge_loras_on_download",
        "Purge unknown LoRAs",
        FieldKind.BOOL,
        "LoRA",
        "Delete LoRAs not in the reference when download_models runs (also removes custom ones).",
    ),
    # Models
    ConfigField(
        "models_to_load",
        "Models to load",
        FieldKind.MODEL_LIST,
        "Models",
        "Models to offer: concrete names and/or meta commands like 'top 5'.",
    ),
    ConfigField(
        "models_to_skip",
        "Models to skip",
        FieldKind.MODEL_LIST,
        "Models",
        "Models to exclude from a meta selection (only removes; never adds).",
    ),
    ConfigField(
        "load_large_models",
        "Load large models",
        FieldKind.BOOL,
        "Models",
        "Include Flux/Cascade in ALL/TOP meta commands (otherwise excluded by size).",
    ),
    # Alchemist
    ConfigField(
        "alchemist",
        "Enable alchemist",
        FieldKind.BOOL,
        "Alchemist",
        "Also serve alchemy (interrogation/post-process) jobs alongside image generation.",
    ),
    ConfigField(
        "alchemist_name",
        "Alchemist name",
        FieldKind.STR,
        "Alchemist",
        "Unique horde-wide name for the alchemist worker.",
        requires_restart=True,
    ),
    ConfigField(
        "forms",
        "Alchemy forms",
        FieldKind.SELECT_MULTI,
        "Alchemist",
        "Which alchemy forms to offer (defaults to all when empty).",
        choices=ALCHEMY_FORMS,
    ),
    ConfigField(
        "alchemy_caption_enabled",
        "Enable captioning",
        FieldKind.BOOL,
        "Alchemist",
        "Allow BLIP captioning (loads BLIP; extra RAM/VRAM).",
    ),
    ConfigField(
        "alchemy_allow_concurrent",
        "Alchemy concurrent",
        FieldKind.BOOL,
        "Alchemist",
        "Allow alchemy alongside image jobs (vs only when the image queue is empty).",
        explicit_default=True,
    ),
    ConfigField(
        "alchemy_max_concurrency",
        "Alchemy concurrency",
        FieldKind.INT,
        "Alchemist",
        "Max alchemy forms in flight at once.",
        minimum=1,
        maximum=16,
    ),
    ConfigField(
        "alchemy_vram_headroom_mb",
        "Alchemy VRAM floor",
        FieldKind.INT,
        "Alchemist",
        "Minimum free VRAM (MB) before popping a concurrent graph alchemy form.",
        minimum=0,
        maximum=49152,
        unit="MB",
        explicit_default=2000,
    ),
    # Other
    ConfigField(
        "remove_maintenance_on_init",
        "Clear maintenance on start",
        FieldKind.BOOL,
        "Other",
        "Clear maintenance mode at startup. Maintenance is a safety feature; investigate if frequent.",
    ),
    ConfigField(
        "limited_console_messages",
        "Limited console",
        FieldKind.BOOL,
        "Other",
        "Fewer console messages (for headless/cloud). Not recommended for most users.",
    ),
    ConfigField(
        "suppress_speed_warnings",
        "Suppress speed warnings",
        FieldKind.BOOL,
        "Other",
        "Hide warnings about slow generations (you are likely serving slower than ideal).",
    ),
    ConfigField(
        "exit_on_unhandled_faults",
        "Exit on faults",
        FieldKind.BOOL,
        "Other",
        "Exit on an unhandled fault (useful when run as a system service).",
    ),
    ConfigField(
        "stats_output_frequency",
        "Stats frequency",
        FieldKind.INT,
        "Other",
        "Seconds between worker stat summaries (0 disables).",
        minimum=0,
        maximum=3600,
        unit="s",
        explicit_default=30,
    ),
    ConfigField(
        "cache_home", "Models folder", FieldKind.STR, "Other", "Where models are stored.", requires_restart=True
    ),
]


def _yaml() -> YAML:
    """A ruamel YAML instance configured to preserve quotes and structure."""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    return yaml


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Any:  # noqa: ANN401 - ruamel CommentedMap
    """Load the YAML config (preserving comments). Returns an empty mapping if the file is absent."""
    if not path.exists():
        return _yaml().load("{}\n")
    with path.open("r", encoding="utf-8") as handle:
        data = _yaml().load(handle)
    return data if data is not None else _yaml().load("{}\n")


def save_config(data: Any, path: Path = DEFAULT_CONFIG_PATH) -> None:  # noqa: ANN401 - ruamel CommentedMap
    """Write the YAML config back to ``path`` (atomic replace), preserving comments."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        _yaml().dump(data, handle)
    tmp.replace(path)


def coerce_value(field: ConfigField, raw: object) -> Any:  # noqa: ANN401 - kind-dependent
    """Convert a widget value into the typed value for the YAML, raising ValueError on bad input."""
    if field.kind is FieldKind.BOOL:
        return bool(raw)
    if field.kind is FieldKind.INT:
        text = str(raw).strip()
        try:
            value = int(text)
        except ValueError as error:
            raise ValueError(f"{field.label} must be a whole number") from error
        if field.minimum is not None and value < field.minimum:
            raise ValueError(f"{field.label} must be at least {field.minimum}")
        if field.maximum is not None and value > field.maximum:
            raise ValueError(f"{field.label} must be at most {field.maximum}")
        return value
    if field.kind in (FieldKind.STR_LIST, FieldKind.MODEL_LIST, FieldKind.SELECT_MULTI):
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        return [line.strip() for line in str(raw).splitlines() if line.strip()]
    return str(raw)


def current_value(field: ConfigField, data: Any) -> Any:  # noqa: ANN401 - kind-dependent
    """Read a field's current value from loaded YAML data, falling back to its default."""
    try:
        value = data.get(field.key)
    except AttributeError:
        value = None
    if value is None:
        return field.default()
    if field.kind in (FieldKind.STR_LIST, FieldKind.MODEL_LIST, FieldKind.SELECT_MULTI) and not isinstance(
        value, list
    ):
        return [str(value)]
    return value
