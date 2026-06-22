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
    FLOAT = "float"
    STR = "str"
    STR_LIST = "str_list"
    MODEL_LIST = "model_list"
    """A models_to_load/skip list, edited via the dedicated model-list control."""
    SELECT_MULTI = "select_multi"
    """A fixed set of multi-selectable string choices (e.g. alchemy forms)."""


# Sentinel for "no explicit default declared" so that a legitimate falsy explicit default
# (False, 0, "") is still honored. The kind-based fallback only applies when this is unchanged.
_UNSET: Any = object()


def format_number(value: float) -> str:
    """Render a numeric bound without a trailing ``.0`` (so a float field shows ``512`` not ``512.0``)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


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
    minimum: float | None = None
    maximum: float | None = None
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
        if self.kind is FieldKind.FLOAT:
            return float(self.minimum) if self.minimum is not None else 0.0
        if self.kind is FieldKind.INT:
            return int(self.minimum) if self.minimum is not None else 0
        if self.kind in (FieldKind.STR_LIST, FieldKind.MODEL_LIST, FieldKind.SELECT_MULTI):
            return []
        return ""


# Section order is the display order within each subtab.
SECTIONS = (
    "Connection",
    "Identity",
    "Throughput",
    "Memory & performance",
    "Content & safety",
    "Features",
    "LoRA",
    "Models",
    "Model downloads",
    "Alchemist",
    "Timeouts",
    "Retry & scheduling",
    "VRAM budget",
    "Exclusive residency",
    "Unservable model breaker",
    "Self-maintenance",
    "GPU sampling lease",
    "Other",
    "Dry-run",
)

# Sub-tab grouping for the config editor: each tab bundles related sections so no single page
# requires long scrolling. Order is the tab order; "Models" is its own tab (the unified panel).
CONFIG_SUBTABS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Essentials", ("Connection", "Identity")),
    ("Models", ("Models", "Model downloads")),
    ("Performance", ("Throughput", "Memory & performance")),
    ("Content", ("Content & safety", "Features")),
    # LoRA and Alchemy are logically distinct concerns (one is an image-job feature, the other a separate
    # worker role), so each gets its own sub-tab rather than sharing a crowded combined page.
    ("LoRA", ("LoRA",)),
    ("Alchemy", ("Alchemist",)),
    ("Timeouts", ("Timeouts", "Retry & scheduling")),
    (
        "Budget",
        (
            "VRAM budget",
            "Exclusive residency",
            "Unservable model breaker",
            "Self-maintenance",
            "GPU sampling lease",
        ),
    ),
    ("Advanced", ("Other",)),
    ("Developer", ("Dry-run",)),
)

SECTION_GUIDANCE: dict[str, str] = {
    "Throughput": "Bounds are enforced (max_threads 1–16, queue_size 0–4, max_batch 1–20, max_power 1–512). "
    "See the suggested values per GPU tier in the README.",
    "Models": "Edit the load/skip rules below; the panel previews exactly which models will load and "
    "their disk cost. Press Resolve to expand 'top N' / 'bottom N' commands (needs usage stats).",
    "Model downloads": "Controls background download behaviour. The Downloads tab provides a live pause/resume "
    "toggle; downloads_paused here sets the default at worker startup.",
    "LoRA": "Allowing LoRA downloads them on demand; set a civitai_api_token for resources that require it.",
    "Alchemist": "Alchemy is a separate worker role (interrogation / post-processing), distinct from LoRA. "
    "Enabling it serves alchemy jobs alongside (or instead of) image generation.",
    "Timeouts": "All timeouts are in seconds. Raise first-step and contended timeouts if Flux or SDXL "
    "jobs are being false-killed during their initial step or under co-residence load.",
    "VRAM budget": "Controls how the scheduler gates preloads and dispatch against measured free VRAM. "
    "Disabling the master switch (enable_vram_budget) restores availability-only behavior.",
    "GPU sampling lease": "The lease serializes denoising loops so spare processes can stage their next pipeline "
    "in parallel. Counterproductive with unload_models_from_vram_often (no staged residency to overlap). "
    "Changes to these fields require a worker restart.",
    "Dry-run": "Testing flags that skip real GPU work. All fields here require a worker restart. "
    "Do not enable these on a production worker.",
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
        "cycle_process_on_model_change",
        "Cycle process on model change",
        FieldKind.BOOL,
        "Memory & performance",
        "Restart the inference process when the loaded model changes. Reduces inter-run drift "
        "at the cost of a full process-restart delay on every model switch.",
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
        FieldKind.FLOAT,
        "LoRA",
        "Keep at least this many GB free on the LoRA cache disk (fractions allowed). Below it, the "
        "worker evicts old LoRAs to make room and stops offering LoRAs if it still can't clear the "
        "floor. 0 disables.",
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
    # Models (rendered as ModelManagerView by config_editor.py)
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
    ConfigField(
        "only_models_on_disk",
        "Only models on disk",
        FieldKind.BOOL,
        "Models",
        "Only offer models already downloaded; any resolved model not on disk is dropped, never fetched.",
    ),
    # Model downloads
    ConfigField(
        "downloads_paused",
        "Pause downloads",
        FieldKind.BOOL,
        "Model downloads",
        "Hold background model downloads at startup. Overridable live from the Downloads tab.",
    ),
    ConfigField(
        "download_rate_limit_kbps",
        "Download rate limit",
        FieldKind.INT,
        "Model downloads",
        "Cap background downloads to this many KB/s (0 = unlimited). Enforced at 16 MB granularity.",
        minimum=0,
        maximum=100000,
        unit="KB/s",
        explicit_default=0,
    ),
    ConfigField(
        "extra_model_directories",
        "Extra model directories",
        FieldKind.STR_LIST,
        "Model downloads",
        "Additional directories to search for already-downloaded models (one path per line). "
        "Each must be laid out like the primary model folder. New downloads always go to the primary root.",
        requires_restart=True,
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
    ConfigField(
        "alchemy_ram_headroom_mb",
        "Alchemy RAM floor",
        FieldKind.INT,
        "Alchemist",
        "Minimum free RAM (MB) before popping an alchemy form. Analogous to the VRAM floor, keeps "
        "alchemy from pushing a memory-resident worker into paging.",
        minimum=0,
        maximum=49152,
        unit="MB",
        explicit_default=2048,
    ),
    # Timeouts
    ConfigField(
        "process_timeout",
        "Job timeout",
        FieldKind.INT,
        "Timeouts",
        "Max seconds a job may run before being killed. High-performance mode divides by 3; moderate by 2.",
        minimum=15,
        maximum=3600,
        unit="s",
        explicit_default=300,
    ),
    ConfigField(
        "post_process_timeout",
        "Post-process timeout",
        FieldKind.INT,
        "Timeouts",
        "Max seconds for upscaling / face-fixing before the job is killed.",
        minimum=15,
        maximum=600,
        unit="s",
        explicit_default=60,
    ),
    ConfigField(
        "preload_timeout",
        "Preload timeout",
        FieldKind.INT,
        "Timeouts",
        "Max seconds to load a model from disk into VRAM before the process is killed.",
        minimum=15,
        maximum=600,
        unit="s",
        explicit_default=80,
    ),
    ConfigField(
        "inference_step_timeout",
        "Step timeout",
        FieldKind.INT,
        "Timeouts",
        "Max seconds a single sampling step may make no progress before the slot is killed as hung.",
        minimum=15,
        maximum=60,
        unit="s",
        explicit_default=20,
    ),
    ConfigField(
        "inference_first_step_timeout",
        "First step timeout",
        FieldKind.INT,
        "Timeouts",
        "Wider grace for the first sampling step, which also covers the cold work before it "
        "(streaming a large model, prompt encoding). Raise if Flux is killed on its first step.",
        minimum=15,
        maximum=600,
        unit="s",
        explicit_default=90,
    ),
    ConfigField(
        "contended_step_timeout",
        "Contended step timeout",
        FieldKind.INT,
        "Timeouts",
        "Wider per-step grace for legitimate but heartbeat-silent heavy work: co-residence contention, "
        "hires-fix second pass, VAE decode, ControlNet graph.",
        minimum=15,
        maximum=600,
        unit="s",
        explicit_default=120,
    ),
    ConfigField(
        "download_timeout",
        "Aux download timeout",
        FieldKind.INT,
        "Timeouts",
        "Max seconds to allow an auxiliary model (LoRA, etc.) to download.",
        minimum=15,
        maximum=3600,
        unit="s",
        explicit_default=211,
    ),
    # Retry & scheduling
    ConfigField(
        "max_inference_attempts",
        "Max inference attempts",
        FieldKind.INT,
        "Retry & scheduling",
        "How many times a job may be dispatched before being faulted. 1 = no retry; 2 (default) = one retry.",
        minimum=1,
        maximum=5,
        explicit_default=2,
    ),
    ConfigField(
        "minutes_allowed_without_jobs",
        "Idle exit timeout",
        FieldKind.INT,
        "Retry & scheduling",
        "Minutes to stay alive with no jobs before exiting. 0 = run indefinitely.",
        minimum=0,
        maximum=3600,
        unit="min",
        explicit_default=30,
    ),
    ConfigField(
        "model_stickiness",
        "Model stickiness",
        FieldKind.FLOAT,
        "Retry & scheduling",
        "Probability (0.0–1.0) that the currently-loaded model is favored when popping a job. "
        "Higher values reduce model switches at the cost of throughput diversity.",
        minimum=0.0,
        maximum=1.0,
    ),
    # VRAM budget
    ConfigField(
        "enable_vram_budget",
        "Enable VRAM budget",
        FieldKind.BOOL,
        "VRAM budget",
        "Gate preloads and dispatch on measured VRAM. When off, uses availability-only behavior "
        "(not recommended on a shared/consumer GPU).",
        explicit_default=True,
    ),
    ConfigField(
        "vram_reserve_mb",
        "VRAM reserve",
        FieldKind.INT,
        "VRAM budget",
        "Free VRAM (MB) kept in reserve above a job's estimated peak. Larger = safer, lower throughput.",
        minimum=0,
        maximum=49152,
        unit="MB",
        explicit_default=2048,
    ),
    ConfigField(
        "ram_reserve_mb",
        "RAM reserve",
        FieldKind.INT,
        "VRAM budget",
        "System RAM (MB) kept in reserve so resident-in-RAM models do not force paging.",
        minimum=0,
        maximum=131072,
        unit="MB",
        explicit_default=4096,
    ),
    # Exclusive residency
    ConfigField(
        "overbudget_exclusive_mode",
        "Overbudget exclusive mode",
        FieldKind.BOOL,
        "Exclusive residency",
        "When a model is admitted over budget (best-effort head-of-queue), evict all other residents "
        "and suppress concurrent dispatch so it runs on an uncontended device.",
        explicit_default=True,
    ),
    ConfigField(
        "whole_card_exclusive_residency",
        "Whole-card exclusive residency",
        FieldKind.BOOL,
        "Exclusive residency",
        "Proactively give a model that needs most of the card sole residency before it streams, "
        "rather than reacting after a fault.",
        explicit_default=True,
    ),
    ConfigField(
        "whole_card_residency_safety_off_gpu",
        "Move safety off-GPU during whole-card",
        FieldKind.BOOL,
        "Exclusive residency",
        "Move the safety process off-GPU while a whole-card model holds the device, freeing its "
        "~1 GB CUDA context. Only applies when both enable_vram_budget and safety_on_gpu are true.",
        explicit_default=True,
    ),
    ConfigField(
        "whole_card_residency_cooldown_seconds",
        "Whole-card cooldown",
        FieldKind.INT,
        "Exclusive residency",
        "Seconds to hold single-residency mode after the last whole-card job finishes, so back-to-back "
        "heavy jobs share one teardown/restore cycle instead of each churning it.",
        minimum=0,
        maximum=600,
        unit="s",
        explicit_default=45,
    ),
    ConfigField(
        "overbudget_step_timeout",
        "Overbudget step timeout",
        FieldKind.INT,
        "Exclusive residency",
        "Per-step grace (seconds) for a job admitted over budget. Heavy models may stream weights "
        "each step and are legitimately slower than inference_step_timeout.",
        minimum=15,
        maximum=600,
        unit="s",
        explicit_default=120,
    ),
    # Unservable model breaker
    ConfigField(
        "unservable_model_fault_threshold",
        "Unservable fault threshold",
        FieldKind.INT,
        "Unservable model breaker",
        "Consecutive OOM/over-budget faults for one model before it is held back. 0 disables. "
        "A successful generation resets the counter.",
        minimum=0,
        maximum=20,
        explicit_default=3,
    ),
    ConfigField(
        "unservable_model_cooldown_seconds",
        "Unservable cooldown",
        FieldKind.INT,
        "Unservable model breaker",
        "How long (seconds) a model flagged locally unservable is suppressed before the worker retries it.",
        minimum=0,
        maximum=86400,
        unit="s",
        explicit_default=900,
    ),
    # Self-maintenance
    ConfigField(
        "self_maintenance_fault_threshold",
        "Self-maintenance fault threshold",
        FieldKind.INT,
        "Self-maintenance",
        "Cross-model OOM faults within the window before the worker self-pauses popping. 0 disables.",
        minimum=0,
        maximum=100,
        explicit_default=6,
    ),
    ConfigField(
        "self_maintenance_window_seconds",
        "Self-maintenance window",
        FieldKind.INT,
        "Self-maintenance",
        "Rolling window (seconds) over which OOM faults are counted for the self-throttle.",
        minimum=1,
        maximum=3600,
        unit="s",
        explicit_default=600,
    ),
    ConfigField(
        "self_maintenance_cooldown_seconds",
        "Self-maintenance cooldown",
        FieldKind.INT,
        "Self-maintenance",
        "How long (seconds) the worker holds its self-imposed pop-pause before resuming.",
        minimum=0,
        maximum=3600,
        unit="s",
        explicit_default=300,
    ),
    # GPU sampling lease
    ConfigField(
        "gpu_sampling_lease_enabled",
        "GPU sampling lease",
        FieldKind.BOOL,
        "GPU sampling lease",
        "Serialize GPU denoising loops so spare processes stage their next pipeline while one samples. "
        "Counterproductive with unload_models_from_vram_often (no staged residency to overlap).",
        requires_restart=True,
    ),
    ConfigField(
        "gpu_sampling_lease_slots",
        "Sampling lease slots",
        FieldKind.INT,
        "GPU sampling lease",
        "How many processes may run the denoising loop at once when gpu_sampling_lease_enabled is true. "
        "1 serializes; values > 1 permit concurrent loops (time-sliced on Windows WDDM).",
        requires_restart=True,
        minimum=1,
        maximum=16,
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
        "capture_kudos_training_data",
        "Capture kudos training data",
        FieldKind.BOOL,
        "Other",
        "Opt in to telemetry capture for kudos model training.",
    ),
    ConfigField(
        "kudos_training_data_file",
        "Kudos training data file",
        FieldKind.STR,
        "Other",
        "File path to write kudos training data (only used when capture is enabled).",
    ),
    ConfigField(
        "cache_home", "Models folder", FieldKind.STR, "Other", "Where models are stored.", requires_restart=True
    ),
    # Dry-run
    ConfigField(
        "dry_run_skip_inference",
        "Skip inference",
        FieldKind.BOOL,
        "Dry-run",
        "Skip real GPU inference and return a dummy 1x1 image instead.",
        requires_restart=True,
    ),
    ConfigField(
        "dry_run_skip_safety",
        "Skip safety",
        FieldKind.BOOL,
        "Dry-run",
        "Skip the NSFW/CSAM safety evaluation model.",
        requires_restart=True,
    ),
    ConfigField(
        "dry_run_skip_api",
        "Skip API calls",
        FieldKind.BOOL,
        "Dry-run",
        "Skip job pop and submit; use canned scenarios instead.",
        requires_restart=True,
    ),
    ConfigField(
        "dry_run_inference_delay",
        "Inference delay",
        FieldKind.FLOAT,
        "Dry-run",
        "Seconds to sleep when skip-inference is active, simulating GPU work.",
        minimum=0.0,
        maximum=60.0,
        unit="s",
        explicit_default=1.0,
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
    if field.kind in (FieldKind.INT, FieldKind.FLOAT):
        text = str(raw).strip()
        is_int = field.kind is FieldKind.INT
        try:
            value: float = int(text) if is_int else float(text)
        except ValueError as error:
            noun = "a whole number" if is_int else "a number"
            raise ValueError(f"{field.label} must be {noun}") from error
        if field.minimum is not None and value < field.minimum:
            raise ValueError(f"{field.label} must be at least {format_number(field.minimum)}")
        if field.maximum is not None and value > field.maximum:
            raise ValueError(f"{field.label} must be at most {format_number(field.maximum)}")
        # Write a clean integer to the YAML when a float field holds a whole number (2, not 2.0); the
        # worker accepts either and this keeps the file tidy and matches what the user typed.
        if not is_int and isinstance(value, float) and value.is_integer():
            return int(value)
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
