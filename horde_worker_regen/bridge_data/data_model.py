"""The config model and initializers for the reGen configuration model."""

from __future__ import annotations

import json
import os
from typing import Self

from horde_sdk.worker.dispatch.ai_horde.bridge_data import CombinedHordeBridgeData
from loguru import logger
from pydantic import Field, field_validator, model_validator
from ruamel.yaml import YAML

from horde_worker_regen.consts import TOTAL_LORA_DOWNLOAD_TIMEOUT
from horde_worker_regen.locale_info.regen_bridge_data_fields import BRIDGE_DATA_FIELD_DESCRIPTIONS


def _compute_extra_slow_overrides(
    *,
    high_performance_mode: bool,
    moderate_performance_mode: bool,
    high_memory_mode: bool,
    very_high_memory_mode: bool,
    queue_size: int,
    max_threads: int,
    preload_timeout: int,
    log: bool = False,
) -> dict[str, bool | int]:
    """Compute field overrides required when extra_slow_worker is enabled.

    Returns:
        A dict of field names to their overridden values.
    """
    overrides: dict[str, bool | int] = {}

    if high_performance_mode:
        overrides["high_performance_mode"] = False
        if log:
            logger.warning("Extra slow worker is enabled, so high_performance_mode has been set to False.")
    if moderate_performance_mode:
        overrides["moderate_performance_mode"] = False
        if log:
            logger.warning("Extra slow worker is enabled, so moderate_performance_mode has been set to False.")
    if high_memory_mode:
        overrides["high_memory_mode"] = False
        if log:
            logger.warning("Extra slow worker is enabled, so high_memory_mode has been set to False.")
    if very_high_memory_mode:
        overrides["very_high_memory_mode"] = False
        if log:
            logger.warning("Extra slow worker is enabled, so very_high_memory_mode has been set to False.")
    if queue_size > 0:
        overrides["queue_size"] = 0
        if log:
            logger.warning(
                "Extra slow worker is enabled, so queue_size has been set to 0. "
                "This behavior may change in the future.",
            )
    if max_threads > 1:
        overrides["max_threads"] = 1
        if log:
            logger.warning(
                "Extra slow worker is enabled, so max_threads has been set to 1. "
                "This behavior may change in the future.",
            )
    if preload_timeout < 120:
        overrides["preload_timeout"] = 120
        if log:
            logger.warning(
                "Extra slow worker is enabled, so preload_timeout has been set to 120. "
                "This behavior may change in the future.",
            )

    return overrides


def compute_performance_timeout(
    *,
    high_performance_mode: bool,
    moderate_performance_mode: bool,
    default_timeout: int,
    current_timeout: int,
    log: bool = False,
) -> int:
    """Compute process_timeout based on the active performance mode.

    Returns:
        The adjusted process timeout value.
    """
    if high_performance_mode:
        adjusted = default_timeout // 3
        if log:
            msg = f"High performance mode: process_timeout set to {adjusted} (1/3 of default)."
            if current_timeout == default_timeout:
                logger.debug(msg)
            else:
                logger.warning(msg)
        return adjusted

    if moderate_performance_mode:
        adjusted = default_timeout // 2
        if log:
            msg = f"Moderate performance mode: process_timeout set to {adjusted} (1/2 of default)."
            if current_timeout == default_timeout:
                logger.debug(msg)
            else:
                logger.warning(msg)
        return adjusted

    return current_timeout


def cap_queue_size(*, max_threads: int, queue_size: int, log: bool = False) -> int:
    """Cap queue_size to 3 when max_threads >= 2.

    Returns:
        The (possibly capped) queue_size.
    """
    if max_threads >= 2 and queue_size > 3:
        if log:
            logger.warning("queue_size has been set to 3 because max_threads is >= 2.")
        return 3
    return queue_size


def _resolve_high_memory_from_very_high(
    *,
    very_high_memory_mode: bool,
    high_memory_mode: bool,
    log: bool = False,
) -> bool:
    """Ensure very_high_memory_mode implies high_memory_mode.

    Returns:
        The resolved high_memory_mode value.
    """
    if very_high_memory_mode and not high_memory_mode:
        if log:
            logger.debug("very_high_memory_mode is enabled, so high_memory_mode has been set to True.")
        return True
    return high_memory_mode


def _apply_high_memory_constraints(
    *,
    high_memory_mode: bool,
    queue_size: int,
    unload_models_from_vram_often: bool,
    cycle_process_on_model_change: bool,
    log: bool = False,
) -> bool:
    """Apply constraints and emit warnings for high_memory_mode.

    Returns:
        The adjusted cycle_process_on_model_change value.
    """
    if not high_memory_mode:
        return cycle_process_on_model_change

    if log:
        if queue_size == 0:
            logger.warning(
                "High memory mode is enabled, you should consider setting queue_size to 1 or higher. "
                "Increasing this value increases system memory usage. See the bridgeData_template.yaml for more "
                "information.",
            )
        if unload_models_from_vram_often:
            logger.warning(
                "High memory mode is enabled, you should consider setting unload_models_from_vram_often to False.",
            )

    if cycle_process_on_model_change:
        if log:
            logger.warning(
                "High memory mode is enabled, so cycle_process_on_model_change has been set to False.",
            )
        return False

    return cycle_process_on_model_change


def _warn_lease_without_residency(
    *,
    gpu_sampling_lease_enabled: bool,
    high_memory_mode: bool,
    unload_models_from_vram_often: bool,
    log: bool = False,
) -> bool:
    """Detect (and optionally warn about) enabling the GPU sampling lease without residency.

    The lease brackets the diffusion model's VRAM load together with the denoise loop, so without
    residency each process's RAM->VRAM transfer is serialized behind sampling instead of
    overlapping it — usually a throughput loss rather than a gain.

    Returns:
        True if the lease is enabled in a non-resident (counterproductive) configuration.
    """
    if not gpu_sampling_lease_enabled:
        return False
    non_resident = unload_models_from_vram_often or not high_memory_mode
    if non_resident and log:
        logger.warning(
            "gpu_sampling_lease_enabled is set without VRAM residency (high_memory_mode=true + "
            "unload_models_from_vram_often=false). The lease brackets the model's VRAM load as "
            "well as the denoise loop, so without residency it serializes RAM->VRAM transfers "
            "behind sampling and typically reduces throughput.",
        )
    return non_resident


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

    download_rate_limit_kbps: int | None = Field(default=None, ge=0)
    """Cap background model downloads to this many KB/s (None or 0 means unlimited).

    Applied by the background download process and honored on config reload. Approximate: enforced at
    16MB-chunk granularity, so very low limits are coarse."""
    downloads_paused: bool = Field(default=False)
    """If true, background model downloads are held (the current chunk loop blocks) until resumed.

    Honored on config reload and overridable live from the TUI; the worker keeps serving models that
    are already on disk while downloads are paused."""

    extra_model_directories: list[str] = Field(default_factory=list)
    """Additional model weights-root directories to search for already-downloaded files.

    Each entry is a directory laid out like the primary model folder (containing ``compvis``, ``lora``,
    etc.); presence checks search the primary root and then these, so model files can be spread across
    disks. New downloads always target the primary root. Also settable via the
    ``AIWORKER_EXTRA_MODEL_DIRECTORIES`` environment variable (``os.pathsep``-separated)."""

    preload_timeout: int = Field(default=80, ge=15)
    """The maximum amount of time to allow a model to load before it is killed"""
    inference_step_timeout: int = Field(default=20, ge=15, le=60)
    """The maximum wall-clock time a single sampling step may make no progress before the slot is killed
    as hung. Kept short so a true hang is caught quickly, but not so short that a momentarily slow step
    on a busy device is mistaken for one."""
    inference_first_step_timeout: int = Field(default=90, ge=15, le=600)
    """The grace for a job's *first* sampling step, which also covers the cold work that precedes it.

    Before the first step a slot may be streaming a large combined checkpoint's components through VRAM
    (loading and running the text encoder, then loading the diffusion weights) and doing the initial
    prompt encode, none of which emit a step. That one-time work is legitimately far longer than a
    steady-state step, so the pre-first-step window uses this generous timeout and only falls back to the
    tighter ``inference_step_timeout`` once sampling progress has been observed. The watchdog floors the
    effective first-step grace at ``inference_step_timeout``, so a value below it has no effect."""

    max_inference_attempts: int = Field(default=2, ge=1, le=5)
    """How many times a single job may be dispatched to inference before it is reported faulted.

    1 disables retry (one shot, then fault: the pre-resiliency behaviour). The default of 2 grants one
    bounded retry, so a job whose slot crashed, hung, or failed to receive its dispatch is requeued for a
    fresh attempt rather than faulted outright. A resource (out-of-memory) failure spends its retry in a
    degraded, isolated dispatch. Once attempts are exhausted the job is reported faulted with diagnostics."""

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

    gpu_sampling_lease_enabled: bool = Field(default=False)
    """Coordinate the GPU denoising loop across inference processes with a shared lease.

    When true, at most `gpu_sampling_lease_slots` inference processes run the denoising loop at
    once; any extra processes stage their next pipeline (model load, prompt encode) concurrently,
    so when a sampling process finishes the next starts immediately — keeping the GPU busy
    between jobs instead of idling during per-job warm-up. Trades extra resident VRAM (each
    process keeps its model loaded) for higher GPU utilization, and **requires** residency
    (`high_memory_mode: true` + `unload_models_from_vram_often: false`) to help: the lease
    brackets the diffusion model's VRAM load as well as the denoise loop, so without residency it
    serializes the RAM->VRAM transfer behind sampling rather than overlapping it."""

    gpu_sampling_lease_slots: int = Field(default=1, ge=1)
    """How many inference processes may run the GPU denoising loop at once when
    `gpu_sampling_lease_enabled` is true (clamped to the inference-process count at runtime).

    1 (the default) serializes denoising — one process samples while the rest stage their next
    pipeline — the cleanest way to keep a single GPU busy back-to-back. Values > 1 permit that
    many concurrent denoise loops; on hardware without CUDA MPS (e.g. Windows WDDM) concurrent
    loops time-slice the GPU rather than truly parallelizing, so this can raise the
    coverage-based duty-cycle metric without improving throughput. No effect unless
    `gpu_sampling_lease_enabled` is true."""

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

    alchemist: bool = Field(default=False)
    """If true, this worker also pops and processes alchemy jobs (/v2/interrogate/pop).

    Graph forms (upscalers, facefixers, strip_background) run on the inference processes,
    which they share with image generation; CLIP forms (interrogation, nsfw) run on the
    safety process. Image jobs always win contention for those processes — alchemy only
    uses a lane image work does not currently need (see `alchemy_allow_concurrent`). The
    forms offered are controlled by the `forms` field (see `CombinedHordeBridgeData`).
    """

    alchemy_caption_enabled: bool = Field(default=False)
    """Opt in to caption alchemy forms.

    Captioning loads BLIP into the safety process on first use, which costs significant
    additional RAM/VRAM, so it is off by default.
    """

    alchemy_allow_concurrent: bool = Field(default=True)
    """Allow alchemy to run alongside image generation instead of only as backfill.

    When true, alchemy may pop while image jobs are queued, but only when a process lane is
    spare (no waiting image job needs it) and there is VRAM headroom for a typical alchemy
    form. When false, the legacy behavior applies: alchemy pops only when the image queue is
    empty.
    """

    alchemy_max_concurrency: int = Field(default=1, ge=1)
    """Maximum number of alchemy forms allowed in flight (dispatched, awaiting result) at once.

    Bounds how much of the worker alchemy can occupy concurrently with image work. Raise it
    on machines with spare compute/VRAM headroom.
    """

    alchemy_vram_headroom_mb: int = Field(default=2000, ge=0)
    """Minimum free VRAM (MB) required before popping a graph alchemy form in concurrent mode.

    Acts as the floor for the headroom estimator, which raises the requirement toward the
    observed median cost of recent alchemy forms. Free VRAM is read from worker memory
    reports; when unavailable, alchemy falls back to backfill-only behavior.
    """

    enable_vram_budget: bool = Field(default=True)
    """Gate model preloads and concurrent dispatch on a measured VRAM budget.

    When true (the default), the scheduler refuses to preload a model or stage another concurrent
    job unless the device's measured free VRAM covers the job's estimated peak plus
    `vram_reserve_mb`, and it evicts the coldest idle resident model under pressure. This is the
    proactive guard against the multi-process over-commit that OOMs a shared GPU. Set false to
    restore the prior availability-only behavior (not recommended on a shared/consumer GPU)."""

    vram_reserve_mb: int = Field(default=2048, ge=0)
    """Free VRAM (MB) the budget keeps in reserve on top of a job's estimated peak.

    Covers transient spikes the steady-state estimate misses, most notably tiled VAE decode (the
    phase that produced the observed live OOM). Larger values trade throughput for safety. Only used
    when `enable_vram_budget` is true."""

    ram_reserve_mb: int = Field(default=4096, ge=0)
    """Available system RAM (MB) the budget keeps in reserve so resident-in-RAM models do not force
    the OS to page to disk. Only used when `enable_vram_budget` is true."""

    vram_per_process_overhead_mb: int = Field(default=0, ge=0)
    """Per-process VRAM (MB) one inference process consumes for its torch/CUDA context with no model loaded.

    The streaming forecast subtracts this from total VRAM to estimate the free achievable under sole
    residency (so it can tell a model that only streams because of co-resident siblings from one that
    streams even alone). 0 (the default) auto-detects the value via the startup accelerator probe; set a
    positive value to override the measurement. Only used when `enable_vram_budget` is true."""

    overbudget_exclusive_mode: bool = Field(default=True)
    """Run a model admitted *against* the VRAM budget (a best-effort head-of-queue admit) with the device
    to itself.

    When the budget cannot fit a head-of-queue model even after reclaiming every idle resident copy, the
    scheduler admits it best-effort rather than wedging the queue. On a small-VRAM card a heavy model
    (e.g. a Flux combined checkpoint) admitted this way only fits if nothing else is resident: a second
    process loading another model concurrently pushes free VRAM to ~0 and the heavy model's weights spill
    to system RAM over PCIe, collapsing its step rate (~50-80x) until the step-timeout watchdog kills it.
    When true (the default), such a job evicts every other resident model first and suppresses concurrent
    pre-staging/dispatch for its duration, so it runs on an un-contended device. Only used when
    `enable_vram_budget` is true."""

    whole_card_exclusive_residency: bool = Field(default=True)
    """Give a model whose weights need most of the device sole residency *before* it streams, not after a fault.

    The streaming forecast (weights vs ComfyUI's inference reserve) flags a model that would offload weights
    to host RAM if loaded alongside the currently-resident models but fits cleanly with the card to itself.
    When true (the default), the scheduler proactively evicts the other processes' resident models, returns
    their freed VRAM to the driver, and suppresses prefetch into sibling slots for its duration, so the model
    loads fully resident and samples at full speed instead of streaming weights and being hang-graded. This
    is the preventative form of `overbudget_exclusive_mode`, which only reacts once a model has already been
    admitted over budget. Only used when `enable_vram_budget` is true."""

    whole_card_residency_safety_off_gpu: bool = Field(default=True)
    """Move the safety process off-GPU while a whole-card model holds the device.

    A model that needs near-sole residency (e.g. Flux on a 16GB card) only fits without streaming if the
    safety process's CUDA context (~1GB, reclaimable only by the process exiting) is also freed. When true
    (the default), the scheduler pauses safety-on-GPU for the duration of a whole-card job and restores it
    after. Costs a brief safety-process restart at each end of a whole-card residency burst (batched by
    `whole_card_residency_cooldown_seconds`). Only used when `enable_vram_budget` and `safety_on_gpu` are
    both true."""

    whole_card_residency_cooldown_seconds: int = Field(default=45, ge=0, le=600)
    """How long to hold a whole-card residency in place after its last heavy job drains, before restoring
    the torn-down sibling processes and the safety process to the GPU.

    A whole-card residency event is disruptive (it stops idle inference processes and cycles the safety
    process), so back-to-back heavy jobs should reuse one residency rather than each triggering a fresh
    teardown/restore. During the cooldown the worker stays in single-residency mode, so a subsequent
    whole-card job runs immediately with no churn and refreshes the cooldown. Only after this many seconds
    with no whole-card job pending or in progress does the worker restore full concurrency. 0 restores
    immediately (maximum responsiveness, maximum churn). Only used when `enable_vram_budget` is true."""

    overbudget_step_timeout: int = Field(default=120, ge=15, le=600)
    """Per-step hang timeout (seconds) granted to a job admitted *against* the VRAM budget.

    A best-effort over-budget admit of a heavy model can stream weights through VRAM each step, so even
    on an un-contended device its steps are legitimately far slower than `inference_step_timeout`. Killing
    such a slow-but-progressing job and dropping it is what produced the live drop storm; this generous
    per-step grace lets it complete (slowly) instead. Applied to every step of an over-budget job (not
    only its first), floored at `inference_step_timeout`. Only used when `enable_vram_budget` is true."""

    unservable_model_fault_threshold: int = Field(default=3, ge=0, le=20)
    """Consecutive over-budget terminal faults for one model before it is treated as locally unservable.

    A model the device genuinely cannot run will fault every attempt no matter how it is isolated; left
    unchecked the worker keeps popping and dropping it, and the horde server forces the worker into
    maintenance for "dropping too many jobs". After this many consecutive over-budget faults for a model,
    the worker stops popping and best-effort-admitting it for `unservable_model_cooldown_seconds`, so the
    bleeding stops. A successful generation of the model resets its counter. 0 disables the breaker."""

    unservable_model_cooldown_seconds: int = Field(default=900, ge=0)
    """How long a model flagged locally unservable is held back before the worker tries it again."""

    self_maintenance_fault_threshold: int = Field(default=6, ge=0)
    """Terminal resource/OOM faults within `self_maintenance_window_seconds` before the worker
    self-throttles.

    A backstop above the per-model breaker: if resource faults across *all* models accumulate fast enough
    to risk the horde's server-side "dropping too many jobs" guard, the worker proactively enters a local
    pop-pause for `self_maintenance_cooldown_seconds` (in-flight jobs finish) so it stops the bleeding on
    its own terms before the server forces maintenance. 0 disables the backstop."""

    self_maintenance_window_seconds: int = Field(default=600, ge=1)
    """Rolling window (seconds) over which resource/OOM faults are counted for the self-throttle backstop."""

    self_maintenance_cooldown_seconds: int = Field(default=300, ge=0)
    """How long the worker holds its self-imposed local pop-pause before resuming after a self-throttle."""

    dry_run_skip_inference: bool = Field(default=False)
    """Skip real GPU inference and return a dummy 1x1 image instead."""

    dry_run_skip_safety: bool = Field(default=False)
    """Skip the safety (NSFW/CSAM) evaluation model."""

    dry_run_skip_api: bool = Field(default=False)
    """Skip API calls (job pop and submit) and use canned scenarios."""

    dry_run_inference_delay: float = Field(default=1.0, ge=0.0)
    """Seconds to sleep when dry-run inference is active, simulating work."""

    @model_validator(mode="after")
    def validate_performance_modes(self) -> Self:
        """Validate and adjust performance mode settings based on cross-field constraints."""
        # Extra slow worker takes priority over all performance/memory settings
        if self.extra_slow_worker:
            for field_name, value in _compute_extra_slow_overrides(
                high_performance_mode=self.high_performance_mode,
                moderate_performance_mode=self.moderate_performance_mode,
                high_memory_mode=self.high_memory_mode,
                very_high_memory_mode=self.very_high_memory_mode,
                queue_size=self.queue_size,
                max_threads=self.max_threads,
                preload_timeout=self.preload_timeout,
                log=True,
            ).items():
                setattr(self, field_name, value)

        self.process_timeout = compute_performance_timeout(
            high_performance_mode=self.high_performance_mode,
            moderate_performance_mode=self.moderate_performance_mode,
            default_timeout=self.model_fields["process_timeout"].default,
            current_timeout=self.process_timeout,
            log=True,
        )

        self.queue_size = cap_queue_size(
            max_threads=self.max_threads,
            queue_size=self.queue_size,
            log=True,
        )

        self.high_memory_mode = _resolve_high_memory_from_very_high(
            very_high_memory_mode=self.very_high_memory_mode,
            high_memory_mode=self.high_memory_mode,
            log=True,
        )

        self.cycle_process_on_model_change = _apply_high_memory_constraints(
            high_memory_mode=self.high_memory_mode,
            queue_size=self.queue_size,
            unload_models_from_vram_often=self.unload_models_from_vram_often,
            cycle_process_on_model_change=self.cycle_process_on_model_change,
            log=True,
        )

        _warn_lease_without_residency(
            gpu_sampling_lease_enabled=self.gpu_sampling_lease_enabled,
            high_memory_mode=self.high_memory_mode,
            unload_models_from_vram_often=self.unload_models_from_vram_often,
            log=True,
        )

        return self

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
        if self.extra_model_directories and os.getenv("AIWORKER_EXTRA_MODEL_DIRECTORIES") is None:
            os.environ["AIWORKER_EXTRA_MODEL_DIRECTORIES"] = os.pathsep.join(self.extra_model_directories)
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
