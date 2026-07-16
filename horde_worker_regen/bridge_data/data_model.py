"""The config model and initializers for the reGen configuration model."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal, Self

from horde_sdk.generation_parameters.alchemy.consts import KNOWN_ALCHEMY_FORMS
from horde_sdk.worker.dispatch.ai_horde.bridge_data import CombinedHordeBridgeData
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from ruamel.yaml import YAML

from horde_worker_regen.consts import TOTAL_LORA_DOWNLOAD_TIMEOUT, WORKER_KNOWN_EXTRA_ALCHEMY_FORMS
from horde_worker_regen.locale_info.regen_bridge_data_fields import BRIDGE_DATA_FIELD_DESCRIPTIONS


@dataclass(frozen=True)
class ExtraSlowClamps:
    """Represents the concurrency/timeout clamps an extra-slow worker forces, each None when unchanged.

    Produced by :func:`compute_extra_slow_clamps` and applied by :func:`apply_extra_slow_clamps`. A field
    is non-None only when extra-slow mode must override the operator's value, so the apply step touches
    exactly the changed fields (and the producer logs exactly those). A typed view rather than a
    field-name->value dict, so both the global validator and the per-card resolver apply the clamps through
    attributes a type checker can follow, not string key literals.
    """

    high_performance_mode: bool | None = None
    moderate_performance_mode: bool | None = None
    queue_size: int | None = None
    max_threads: int | None = None
    preload_timeout: int | None = None


def compute_extra_slow_clamps(
    *,
    high_performance_mode: bool,
    moderate_performance_mode: bool,
    queue_size: int,
    max_threads: int,
    preload_timeout: int,
    log: bool = False,
) -> ExtraSlowClamps:
    """Return the clamps an extra-slow worker forces on the given concurrency/timeout values.

    Each returned field is None unless extra-slow mode requires changing it (so an unchanged value is never
    re-applied or re-logged). When ``log`` is True, a warning is emitted for each clamp actually applied.
    """
    high_clamp: bool | None = None
    moderate_clamp: bool | None = None
    queue_clamp: int | None = None
    threads_clamp: int | None = None
    preload_clamp: int | None = None

    if high_performance_mode:
        high_clamp = False
        if log:
            logger.warning("Extra slow worker is enabled, so high_performance_mode has been set to False.")
    if moderate_performance_mode:
        moderate_clamp = False
        if log:
            logger.warning("Extra slow worker is enabled, so moderate_performance_mode has been set to False.")
    if queue_size > 0:
        queue_clamp = 0
        if log:
            logger.warning(
                "Extra slow worker is enabled, so queue_size has been set to 0. "
                "This behavior may change in the future.",
            )
    if max_threads > 1:
        threads_clamp = 1
        if log:
            logger.warning(
                "Extra slow worker is enabled, so max_threads has been set to 1. "
                "This behavior may change in the future.",
            )
    if preload_timeout < 150:
        preload_clamp = 150
        if log:
            logger.warning(
                "Extra slow worker is enabled, so preload_timeout has been set to 150. "
                "This behavior may change in the future.",
            )

    return ExtraSlowClamps(
        high_performance_mode=high_clamp,
        moderate_performance_mode=moderate_clamp,
        queue_size=queue_clamp,
        max_threads=threads_clamp,
        preload_timeout=preload_clamp,
    )


def apply_extra_slow_clamps(config: reGenBridgeData, clamps: ExtraSlowClamps) -> None:
    """Mutate *config* in place, applying each clamp that is set (None fields are left untouched)."""
    if clamps.high_performance_mode is not None:
        config.high_performance_mode = clamps.high_performance_mode
    if clamps.moderate_performance_mode is not None:
        config.moderate_performance_mode = clamps.moderate_performance_mode
    if clamps.queue_size is not None:
        config.queue_size = clamps.queue_size
    if clamps.max_threads is not None:
        config.max_threads = clamps.max_threads
    if clamps.preload_timeout is not None:
        config.preload_timeout = clamps.preload_timeout


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


def _warn_lease_without_residency(
    *,
    gpu_sampling_lease_enabled: bool,
    unload_models_from_vram_often: bool,
    log: bool = False,
) -> bool:
    """Detect (and optionally warn about) enabling the GPU sampling lease without model residency.

    The lease brackets the diffusion model's RAM->VRAM load together with the denoise loop so a spare
    process can stage its next pipeline while another samples. With ``unload_models_from_vram_often`` the
    model is fully evicted between jobs, so there is no staged residency to overlap and the lease serializes
    the reload behind sampling instead, which is usually a throughput loss rather than a gain.

    Returns:
        True if the lease is enabled in a non-resident (counterproductive) configuration.
    """
    if not gpu_sampling_lease_enabled:
        return False
    non_resident = unload_models_from_vram_often
    if non_resident and log:
        logger.warning(
            "gpu_sampling_lease_enabled is set with unload_models_from_vram_often=true, which fully evicts "
            "the model between jobs. The lease brackets the model's RAM->VRAM load as well as the denoise "
            "loop, so without residency it serializes reloads behind sampling and typically reduces throughput.",
        )
    return non_resident


def _warn_lease_slots_below_threads(
    *,
    gpu_sampling_lease_enabled: bool,
    gpu_sampling_lease_slots: int | None,
    max_threads: int,
    log: bool = False,
) -> bool:
    """Detect (and optionally warn about) pinning fewer denoise slots than the concurrency cap.

    With the lease enabled the slot count is the active-denoise gate, so an explicit value below
    ``max_threads`` samples fewer jobs at once than the worker admits. That is the correct trade only
    where concurrent denoise loops cannot truly parallelize (no CUDA MPS, e.g. Windows WDDM), where it
    overlaps the next job's staging behind the current denoise without paying for time-sliced sampling;
    elsewhere it leaves requested concurrency unused. The default (``None``) tracks ``max_threads`` and
    never trips this.

    Returns:
        True if an explicit slot count sits below the concurrency cap.
    """
    if not gpu_sampling_lease_enabled or gpu_sampling_lease_slots is None:
        return False
    below = gpu_sampling_lease_slots < max_threads
    if below and log:
        logger.warning(
            f"gpu_sampling_lease_slots={gpu_sampling_lease_slots} is below max_threads={max_threads}, so the "
            "lease will sample fewer jobs at once than the worker runs concurrently. Leave it unset to track "
            "max_threads; an explicit lower value only helps where concurrent denoise loops time-slice rather "
            "than parallelize (no CUDA MPS, e.g. Windows WDDM).",
        )
    return below


class GpuOverride(BaseModel):
    """A per-card delta over the global config: every field optional, ``None`` meaning inherit.

    Lives on :attr:`reGenBridgeData.gpu_overrides`, keyed by stable (PCI-bus) device index. Only the fields
    meaningful to gate per card are present (concurrency, the served model set, feature flags, and the VRAM
    budget); ``extra="forbid"`` makes a typo or an attempt to override a global-only field (e.g. ``api_key``)
    a loud validation error rather than a silently ignored key. Constraints mirror the corresponding fields
    on :class:`reGenBridgeData`. The orchestrator resolves these into a per-card :class:`reGenBridgeData`
    via :func:`horde_worker_regen.bridge_data.gpu_config.resolve_effective_gpu_config`.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # -- Concurrency --
    max_threads: int | None = Field(default=None, ge=1, le=16)
    queue_size: int | None = Field(default=None, ge=0, le=4)
    high_performance_mode: bool | None = None
    moderate_performance_mode: bool | None = None
    extra_slow_worker: bool | None = None
    preload_timeout: int | None = Field(default=None, ge=15)

    # -- Models & baselines --
    image_models_to_load: list[str] | None = Field(default=None, alias="models_to_load")
    image_models_to_skip: list[str] | None = Field(default=None, alias="models_to_skip")
    dynamic_models: bool | None = None

    # -- Feature flags --
    allow_lora: bool | None = None
    allow_controlnet: bool | None = None
    allow_sdxl_controlnet: bool | None = None
    allow_post_processing: bool | None = None
    allow_inpainting: bool | None = Field(default=None, alias="allow_painting")
    allow_img2img: bool | None = None
    nsfw: bool | None = None
    max_power: int | None = Field(default=None, ge=1, le=512)

    # -- VRAM / memory budget --
    enable_vram_budget: bool | None = None
    vram_reserve_mb: int | None = Field(default=None, ge=0)
    vram_to_leave_free: str | None = Field(default=None, pattern=r"^\d+%$|^\d+$")
    whole_card_exclusive_residency: bool | None = None


class reGenBridgeData(CombinedHordeBridgeData):
    """The config model for reGen. Extra fields added here are specific to this worker implementation.

    See `CombinedHordeBridgeData` from the SDK for more information..
    """

    _loaded_from_env_vars: bool = False

    gpu_device_indices: list[int] | None = Field(default=None)
    """Which accelerator indices (stable PCI-bus order) this one worker drives.

    ``None`` (the default) auto-detects and uses *every* accelerator-kind device on the machine under a
    single horde identity. Set an explicit list (e.g. ``[0]`` or ``[0, 2]``) to opt a multi-GPU box out of
    driving all cards, or to pin which cards this worker owns. Indices key into :attr:`gpu_overrides` and
    are resolved against ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` so they map to fixed physical slots across reboots.
    A single-GPU box can ignore this entirely.
    """

    gpu_overrides: dict[int, GpuOverride] = Field(default_factory=dict)
    """Optional per-card config deltas, keyed by stable (PCI-bus) device index.

    Each value overrides only the fields it sets (everything else inherits the global config), letting a
    heterogeneous box give each card its own concurrency, served models, feature flags, and VRAM budget
    without standing up separate worker instances. A device with no entry here inherits the global config
    wholesale, so the single-GPU and homogeneous cases need no entries at all.
    """

    gpu_pop_balance_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Local-queue imbalance fraction that switches the next pop from union to a single under-fed card.

    With multiple cards the worker pops the *union* of every card's capabilities by default, then routes
    each returned job to an eligible card. When at least this fraction of the held/in-flight work is
    servable by only a subset of cards, the next pop is instead scoped to the under-fed card's capability
    set, so the horde returns work that card can actually run. 0 always targets the most under-fed card;
    1 effectively disables targeting (pure union pops). No effect on a single-GPU worker.
    """

    disable_terminal_ui: bool = Field(
        default=True,
    )

    safety_on_gpu: bool = Field(
        default=False,
    )
    """If true, the safety model will be run on the GPU."""

    dedicated_post_processing: Literal["auto", "on", "off"] = Field(
        default="auto",
    )
    """Whether to run a dedicated post-processing process.

    The dedicated process keeps the post-processing models (upscalers, face-fixers, background removal)
    resident and runs every post-processing phase off the inference processes, converting the transient
    per-job post-processing VRAM spike into one fixed, budgetable footprint.

    - "auto": run the lane whenever any of its work is served (post-processing allowed, or an
      alchemist worker whose graph forms run on the lane).
    - "on": always run the lane.
    - "off": do not run the lane; the worker will not offer post-processing at all.
    """

    @property
    def post_processing_lane_enabled(self) -> bool:
        """Whether the dedicated post-processing lane should be running.

        The lane is the only place post-processing and graph alchemy forms (upscale/facefix/
        strip_background) run; "off" therefore also implies the worker does not offer post-processing.
        "auto" ties the lane to whether any of its work is served: embedded job post-processing
        (``allow_post_processing``) or graph alchemy forms (``alchemist``).
        """
        if self.dedicated_post_processing == "off":
            return False
        if self.dedicated_post_processing == "on":
            return True
        return bool(self.allow_post_processing) or bool(self.alchemist)

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

    min_lora_disk_free_gb: float = Field(default=1.0, ge=0.0)
    """Keep at least this many gigabytes free on the LoRA cache volume.

    The ad-hoc LoRA cache is downloaded on demand and bounded by ``max_lora_cache_size``, but a
    mis-set budget or a volume shared with other data can still drive it toward a full disk (where
    every weight write fails with ENOSPC). Below this free-space floor the worker shrinks the
    effective cache to fit and evicts least-recently-used ad-hoc LoRAs to make room; if even that
    cannot clear the floor it stops offering LoRAs (see ``effective_allow_lora``). Set to 0 to
    disable the floor. Propagated to the LoRA manager as ``AIWORKER_LORA_MIN_DISK_FREE_MB``."""

    unload_models_from_vram_often: bool = Field(default=True)
    """If true, models will be unloaded from VRAM more often."""

    process_timeout: int = Field(default=300)
    """The maximum amount of time to allow a job to run before it is killed"""

    post_process_timeout: int = Field(default=120, ge=15)

    download_timeout: int = Field(default=TOTAL_LORA_DOWNLOAD_TIMEOUT + 1)
    """The maximum amount of time to allow an aux model to download before it is killed"""

    download_rate_limit_kbps: int | None = Field(default=None, ge=0)
    """Cap background model downloads to this many KB/s (None or 0 means unlimited).

    Applied by the background download process and honored on config reload. Approximate: enforced at
    16MB-chunk granularity, so very low limits are coarse. With parallel downloads the cap is the
    aggregate: it is divided across the downloads in flight."""
    download_max_parallel_downloads: int = Field(default=4, ge=1)
    """How many model downloads may run at once across all source hosts (1 = fully sequential).

    Downloads are parallelized across distinct hosts (e.g. civitai.com / huggingface.co), so a fresh
    install fetches generation, clip/blip, controlnet and post-processing models concurrently rather than
    one at a time. Honored at startup and on config reload."""
    download_per_host_concurrency: int = Field(default=1, ge=1)
    """How many downloads to the *same* host may run at once (default 1 = one connection per host).

    Raise above 1 to also allow several concurrent downloads from a single host; left at 1, a host is
    never hit by more than one download at a time. Honored at startup and on config reload."""
    download_connections_per_file: int = Field(default=4, ge=1)
    """How many concurrent connections to use for a single large model file (default 4; 1 = single stream).

    A single TCP stream to a CDN is often window/RTT-limited well below the link, so a large checkpoint is
    fetched over this many ranged connections at once to raise its download rate. Only large files on
    range-capable hosts are segmented; small files and servers that ignore Range fall back to one stream.
    Honored at startup and on config reload.

    TRADE-OFF: a segmented (multi-connection) download CANNOT be resumed. If it is interrupted (worker
    restart, network drop, crash), the partial file is discarded and the whole file is re-fetched from the
    start on the next attempt. Only ``1`` keeps the resumable single-stream behaviour, where an interrupted
    download continues from where it left off. Set this to ``1`` on a slow or unreliable connection where
    re-fetching a large checkpoint from scratch is worse than a lower steady-state rate."""
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

    preload_timeout: int = Field(default=150, ge=15)
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
    inference_stuck_step_repeat_limit: int = Field(default=20, ge=3, le=100)
    """How many times a slot may report the *same* sampling step without advancing before it is reaped.

    Guards a wedge the time-based ``inference_step_timeout`` cannot see: when the underlying ComfyUI
    generation loops on a single step (in practice the final step, after a corrupt model+LoRA combination
    or a pipeline fault), the child keeps receiving identical progress callbacks and keeps emitting
    heartbeats, so the slot never goes silent and the hang watchdog never fires. It would otherwise sit in
    ``INFERENCE_STARTING`` indefinitely, holding VRAM and a queue slot while never returning a result. A
    healthy job reports each step (including the last) exactly once, so any sustained repeat is anomalous;
    the default leaves headroom above a stray duplicate yet reaps a genuine wedge within seconds (progress
    reports arrive roughly once a second). Lower it for faster recovery, raise it if a legitimate pipeline
    on your hardware re-reports a step a handful of times before advancing."""
    contended_step_timeout: int = Field(default=120, ge=15, le=600)
    """Per-step hang timeout (seconds) for a slot doing legitimate but heartbeat-silent heavy work.

    The flat ``inference_step_timeout`` suits a light job on an uncontended device. On a multi-process
    worker two healthy cases legitimately exceed it with no step heartbeat: a single sampling step
    stretched by co-residence contention, and a feature phase that emits no sampling step for its
    duration (the hires-fix second pass, VAE decode, post-processing setup, a ControlNet graph). Neither
    is a hang. The watchdog widens the per-step grace up to this value (floored at
    ``inference_step_timeout``) when there is positive evidence of such work: a non-step pipeline phase is
    running, the slot has been graded contention-slowed, or the job's features (ControlNet, hires-fix,
    batching, large resolution) make it heavy; otherwise the grace scales with the job's expected sampling
    time. A genuinely wedged slot is still reaped once it has been continuously silent past this bound, so
    raise it if heavy/contended jobs are still being false-killed and lower it for faster hang recovery."""

    max_inference_attempts: int = Field(default=2, ge=1, le=5)
    """How many times a single job may be dispatched to inference before it is reported faulted.

    1 disables retry (one shot, then fault: the pre-resiliency behaviour). The default of 2 grants one
    bounded retry, so a job whose slot crashed, hung, or failed to receive its dispatch is requeued for a
    fresh attempt rather than faulted outright. A resource (out-of-memory) failure spends its retry in a
    degraded, isolated dispatch. Once attempts are exhausted the job is reported faulted with diagnostics."""

    minutes_allowed_without_jobs: int = Field(default=0, ge=0, lt=60 * 60)
    """After this many minutes of accumulated idle time in a session, log a low-demand advisory.

    This is purely an advisory threshold: the worker never exits or shuts down when idle, it simply
    notes that it has spent a while without jobs (usually low demand) and suggests offering more models
    or raising max_power. 0 (the default) disables the advisory, which suits a worker deliberately left
    running to wait for demand. Consumed only by the status reporter.
    """

    horde_model_stickiness: float = Field(default=0.0, le=1.0, ge=0.0, alias="model_stickiness")
    """
    A percent chance (expressed as a decimal between 0 and 1) that the currently loaded models will
    be favored when popping a job.
    """

    high_performance_mode: bool = Field(default=False)
    """If you have a 4090 or better, set this to true to enable high performance mode."""

    moderate_performance_mode: bool = Field(default=False)
    """If you have a 3080 or better, set this to true to enable moderate performance mode."""

    very_fast_disk_mode: bool = Field(default=False)
    """If you have a very fast disk, set this to true to concurrently load more models at a time from disk."""

    gpu_sampling_lease_enabled: bool = Field(default=False)
    """Coordinate the GPU denoising loop across inference processes with a shared lease.

    When true, at most `gpu_sampling_lease_slots` inference processes run the denoising loop at
    once; any extra processes stage their next pipeline (model load, prompt encode) concurrently,
    so when a sampling process finishes the next starts immediately, keeping the GPU busy
    between jobs instead of idling during per-job warm-up. Trades extra resident memory (each
    process keeps its model staged) for higher GPU utilization, and is counterproductive with
    `unload_models_from_vram_often: true`: the lease brackets the diffusion model's RAM->VRAM load
    as well as the denoise loop, so when the model is fully evicted between jobs it serializes the
    reload behind sampling rather than overlapping it."""

    gpu_sampling_lease_slots: int | None = Field(default=None, ge=1)
    """How many inference processes may run the GPU denoising loop at once when
    `gpu_sampling_lease_enabled` is true (clamped to the inference-process count at runtime).

    Leave unset (the default) to track `max_threads`: the lease is the denoise gate, so without this
    the slot count would default to 1 and enabling the lease on a `max_threads > 1` worker would
    silently sample fewer jobs at once than the worker admits. Auto-tracking keeps the
    concurrent-denoise count your concurrency settings already imply.

    Set an explicit value to override. 1 serializes denoising, so one process samples while the rest
    stage their next pipeline; on hardware without CUDA MPS (e.g. Windows WDDM) concurrent denoise
    loops time-slice the GPU rather than truly parallelizing, so an explicit value below `max_threads`
    is the efficient choice there (it overlaps the next job's RAM->VRAM load and prompt encode behind
    the current denoise without paying for time-sliced concurrent sampling). Values above 1 permit
    that many concurrent denoise loops. No effect unless `gpu_sampling_lease_enabled` is true."""

    gpu_sampling_lease_tail_overlap: bool = Field(default=True)
    """Admit the next job's sampling window during the current one's tail so the GPU never idles between
    sampling windows. Only meaningful when `gpu_sampling_lease_enabled` is true.

    The lease brackets each job's diffusion-model VRAM load and its denoise loop, so at one slot the
    incoming job's VRAM load serializes behind the outgoing job's denoise and the GPU idles through the
    handoff. With tail overlap the lease carries one extra permit the parent holds, so processes still see
    exactly `gpu_sampling_lease_slots` denoise slots in steady state. When the outgoing sampler nears the end
    of its denoise loop and a staged sibling is primed and waiting, and the card measurably has room with the
    driver not demand-paging, the parent hands the extra permit over for one handoff window: the sibling
    begins sampling while the outgoing job finishes, then the parent takes the permit back once the outgoing
    sampler leaves its loop. Leave enabled to keep the GPU busy across job boundaries; disable to restore the
    strict per-slot denoise gate with no handoff overlap."""

    capture_kudos_training_data: bool = Field(default=False)

    kudos_training_data_file: str | None = Field(default=None)

    exit_on_unhandled_faults: bool = Field(default=False)
    """If true, the worker will exit if an unhandled fault occurs instead of attempting to recover."""

    purge_loras_on_download: bool = Field(default=False)

    remove_maintenance_on_init: bool = Field(default=False)

    load_large_models: bool = Field(default=False)

    only_models_on_disk: bool = Field(default=False)
    """If true, the worker only offers models whose files are already on disk.

    Any model the load rules resolve to (a literal name or a meta command like ``top 5``) that is not
    already present is dropped from ``image_models_to_load`` rather than downloaded. Lets an operator
    pin the served set to what they have without curating an explicit list, and guarantees a config
    change never kicks off a large download.
    """

    custom_models: list[dict] = Field(
        default_factory=list,
    )

    limited_console_messages: bool = Field(default=False)
    """If true, the worker will only log for submit and the status message.

    Set stats_output_frequency (in seconds) for control over the status message.
    """

    log_purge_max_age_days: float = Field(default=30.0, ge=0.0)
    """Delete on-disk worker logs older than this many days at startup (0 disables the age-out).

    The worker's ``logs/`` directory accumulates rotated, zipped ``bridge*.log`` and ``trace*.log``
    archives plus one-per-run ``stdout``/``stderr``, startup-crash, console-redirect and ``faulthandler``
    files. The loguru sinks bound the *count* of each rotated family, but nothing ages files out or bounds
    the directory as a whole, so a long-lived install grows it without limit. At startup the worker deletes
    any log file last modified more than this many days ago. Generous by default so recent history stays
    available for investigating an incident; set 0 to keep logs until only ``log_purge_max_total_gb`` trims
    them."""

    log_purge_max_total_gb: float = Field(default=5.0, ge=0.0)
    """Cap the total size of the ``logs/`` directory in gigabytes (0 disables the size cap).

    After the age-out (``log_purge_max_age_days``), if the directory still exceeds this many gigabytes the
    worker deletes the oldest log files first until it fits, so a burst of churn cannot fill the disk
    between age-outs. The currently active sinks are the newest files and are trimmed last. Set 0 to rely
    only on the age-out."""

    stats_export_enabled: bool = Field(default=False)
    """If true, the worker exports structured per-session stats JSONL for every run without a manual toggle.

    The worker can write a machine-readable ``stats-v*.jsonl`` stream (session boundaries, per-job records,
    periodic samples, and resource/decision events) under ``.horde_worker_regen/stats`` for offline
    analysis. That export is normally opt-in per session from the dashboard; set this true to have it
    enabled automatically on every start. Off by default so upgrading never begins writing extra files
    unprompted. The dashboard toggle still works and overrides this for the running session."""

    stats_purge_max_age_days: float = Field(default=30.0, ge=0.0)
    """Delete exported stats files older than this many days at startup (0 disables the age-out).

    Mirrors ``log_purge_max_age_days`` for the ``.horde_worker_regen/stats`` directory: at startup the
    worker deletes any recognized ``stats-v*.jsonl``/``.jsonl.gz`` file last modified more than this many
    days ago. Only files the exporter itself writes are ever eligible; anything else in the directory is
    left untouched. Set 0 to keep stats until only ``stats_purge_max_total_gb`` trims them."""

    stats_purge_max_total_gb: float = Field(default=5.0, ge=0.0)
    """Cap the total size of the ``.horde_worker_regen/stats`` directory in gigabytes (0 disables the cap).

    After the age-out (``stats_purge_max_age_days``), if the directory still exceeds this many gigabytes the
    worker deletes the oldest recognized stats files first until it fits. The active session file is the
    newest and is trimmed last. Set 0 to rely only on the age-out."""

    stats_autozip_enabled: bool = Field(default=True)
    """If true, compress inactive exported stats files to ``.jsonl.gz`` at startup before purging.

    At startup the worker gzip-compresses every retained ``stats-v*.jsonl`` except the newest (the file it
    is about to write to), so prior-session exports occupy far less disk before the age-out and size-cap
    run over them. Only recognized stats files are touched, via an atomic temp-then-replace compression."""

    dreamer: bool = Field(default=True)
    """If true, this worker pops and processes image-generation jobs (the dreamer role).

    Defaults on, so a plain worker is a dreamer. Set it false to deliberately run an alchemist-only
    worker on a GPU box (deselect image generation while keeping `alchemist: true`); the role matrix is:

    - ``dreamer: true,  alchemist: false`` -> image generation only (the historical default).
    - ``dreamer: true,  alchemist: true``  -> both image generation and alchemy.
    - ``dreamer: false, alchemist: true``  -> alchemy only (no image generation).
    - ``dreamer: false, alchemist: false`` -> nothing to serve (a warning is logged).

    A CPU-only install cannot serve image generation regardless of this flag, so it is treated as
    alchemist-only there. The single source of truth deriving served workloads from these flags is
    :func:`horde_worker_regen.capabilities.enabled_workloads`.
    """

    alchemist: bool = Field(default=False)
    """If true, this worker also pops and processes alchemy jobs (/v2/interrogate/pop).

    Graph forms (upscalers, facefixers, strip_background) run on the inference processes,
    which they share with image generation; CLIP forms (interrogation, nsfw) run on the
    safety process. Image jobs always win contention for those processes; alchemy only
    uses a lane image work does not currently need (see `alchemy_allow_concurrent`). The
    forms offered are controlled by the `forms` field (see `CombinedHordeBridgeData`).
    """

    alchemy_caption_enabled: bool = Field(default=False)
    """Opt in to caption alchemy forms.

    Captioning loads BLIP into the safety process on first use, which costs significant
    additional RAM/VRAM, so it is off by default.
    """

    aesthetic_scoring_enabled: bool = Field(default=True)
    """Attach a LAION aesthetic score to every image generation as ``gen_metadata``.

    The safety pass already embeds each generated image with CLIP, which is exactly the input the
    aesthetic head consumes, so the score is near-free to produce. On by default; set false to skip the
    scoring (and the one-time predictor-weight download). This is independent of offering the
    ``aesthetic`` alchemy form, which is controlled by ``forms``.
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
    """Cold-start floor (MB) for an alchemy form's predicted VRAM cost.

    Alchemy admission shares the same committed-reserve accounting as image generation: a form is
    popped only when the device's *effective* free VRAM (measured free minus VRAM already committed by
    in-flight image and alchemy work) covers the form's predicted cost. That prediction comes from the
    headroom estimator, which raises it toward the observed median cost of recent forms; until real
    measurements accumulate it falls back to this floor. Free VRAM is read from worker memory reports;
    when unavailable, alchemy falls back to backfill-only behavior.
    """

    alchemy_ram_headroom_mb: int = Field(default=2048, ge=0)
    """Minimum effective available system RAM (MB) required before popping an alchemy form.

    The RAM analogue of `alchemy_vram_headroom_mb`: graph alchemy forms load weights into system RAM too,
    so alchemy is held back when available RAM (minus RAM already committed by in-flight work) falls below
    this floor, keeping it from pushing a memory-resident worker into paging. Read from the live system RAM
    figure; when unavailable, alchemy does not gate on RAM.
    """

    enable_vram_budget: bool = Field(default=True)
    """Gate model preloads and concurrent dispatch on a measured VRAM budget.

    When true (the default), the scheduler refuses to preload a model or stage another concurrent
    job unless the device's measured free VRAM covers the job's estimated peak plus
    `vram_reserve_mb`, and it evicts the coldest idle resident model under pressure. This is the
    proactive guard against the multi-process over-commit that OOMs a shared GPU. Set false to
    restore the prior availability-only behavior (not recommended on a shared/consumer GPU)."""

    enable_pipeline_disaggregation: bool = Field(default=False)
    """Route eligible jobs through the disaggregated stage pipeline instead of the monolithic path.

    Experimental. When true, an eligible job is split into stages that run in separate processes exchanging
    small activations rather than one process running the whole job: a dedicated text-encode service process
    produces the conditioning, the inference processes run UNet-only sampling to produce latents, a dedicated
    VAE lane handles the encode and decode, and post-processing is forced onto the dedicated post-processing
    lane. Because a sampler holds only the UNet, its VRAM peak is a fraction of the whole job's, which keeps
    two samplers co-resident on a card the whole-job footprint would collapse to one, and moves the VAE spike
    off the sampler.

    A job is disaggregated only when every condition holds: this flag is on; its source processing is
    txt2img, img2img, or remix; it requests no control_type; its model's baseline is an SD1.5 or SDXL family
    (resolved from the loaded model reference, the only v1 families supported); and both the encode-service
    (component) process and the image lane are live and healthy. Any condition failing leaves the job on the
    monolithic path, where it is charged and dispatched whole; that IS the fallback (there is no
    re-queue-on-stall), so a stage-service outage, an ineligible family, or a control/inpaint job simply runs
    monolithically. A role dying after a job is claimed is backstopped by the pipeline's per-stage patience,
    which faults the job for horde reissue rather than parking it forever."""

    pp_overlap_margin_mb_disaggregated: float | None = Field(default=None, ge=0)
    """Override (MB) for the post-processing co-residency measured second-say margin, disaggregation jobs only.

    Experimental tuning lever. The post-processing/sampling co-residency gate, on a static-accounting miss,
    gives the parent's measured device-free reading a second say and admits only when the free (net of the
    reserve, the sampling peak, and any pending chain reserve) clears a fixed margin (1024MB by default). A
    disaggregation-class-eligible job holds only its UNet-only sampler peak, so its true overlap headroom
    differs from a monolithic whole-job dispatch; setting a positive value here applies that margin in place of
    the default *only* when the candidate job is disaggregation-class-eligible. Monolithic-path jobs always
    keep the 1024MB default regardless of this setting. None (the default) leaves the behavior unchanged. Only
    used when `enable_pipeline_disaggregation` and `enable_vram_budget` are true."""

    enable_image_utilities: bool = Field(default=True)
    """Run the dedicated image-utilities lane (the ``horde_image_utilities`` capability service).

    The lane runs from its own virtual environment as a loopback HTTP service, so the native,
    accelerator-gated stack behind ControlNet annotation and background removal never enters the worker's
    main environment. When true, the worker starts and supervises that subprocess as an ordinary child.
    Defaults to true; startup remains guarded until the utilities venv is provisioned."""

    extended_controlnet: bool = Field(default=True)
    """Operator opt-in to the extended controlnet control types (everything beyond the classic set).

    Advertising extended controlnet is a per-pop decision, not a static capability: the worker only offers
    it once the annotators for the extended types are actually servable (the image-utilities lane reports
    them, or their weight files are on disk for in-graph annotation). This flag is the operator's disk-space
    opt-out; setting it false keeps the worker on the classic controlnet set even after the extended
    annotators are available. The effective per-pop value is this flag AND that readiness, so a fresh
    install advertises extended only after its annotators finish downloading, without a restart."""

    vram_reserve_mb: int = Field(default=2048, ge=0)
    """Free VRAM (MB) the budget keeps in reserve on top of a job's estimated peak.

    Covers transient spikes the steady-state estimate misses, most notably tiled VAE decode (the
    phase that produced the observed live OOM). Larger values trade throughput for safety. Only used
    when `enable_vram_budget` is true."""

    ram_reserve_mb: int = Field(default=4096, ge=0)
    """Available system RAM (MB) the budget keeps in reserve so resident-in-RAM models do not force
    the OS to page to disk. Only used when `enable_vram_budget` is true."""

    ram_pressure_pause_percent: float = Field(default=85.0, ge=0, le=100)
    """System-RAM usage percentage at or above which the worker degrades to protect against an OS OOM kill.

    Distinct from `ram_reserve_mb` (a marginal, per-job admission reserve): this is an *absolute* danger
    floor on the whole host. When system RAM usage reaches this percentage (equivalently, available RAM
    falls below `100 - this` percent of total) the worker stops admitting new model loads, sheds idle
    resident inference processes, and pauses job pops until RAM recovers, rather than loading a model's
    weights through an out-of-RAM host and being killed by the kernel OOM-killer. The effective floor is
    the *more conservative* of this percentage and `ram_pressure_min_free_mb`, so a large-RAM host is
    protected by the percentage and a small-RAM host by the absolute floor. A resident inference process
    can allocate several GB in a single step (a batch decode, or a fresh checkpoint routed through RAM),
    so the headroom this leaves must exceed one such step; the default keeps ~15% of RAM free for that
    reason. Only used when `enable_vram_budget` is true."""

    ram_pressure_min_free_mb: int = Field(default=1024, ge=0)
    """Minimum free system RAM (MB) below which the worker degrades, regardless of `ram_pressure_pause_percent`.

    The absolute companion to the percentage floor: on a small-RAM host `100 - ram_pressure_pause_percent`
    percent of total can still be too few megabytes to load safely, so the worker also degrades whenever
    free RAM drops below this many MB. The effective danger floor is `max((100 -
    ram_pressure_pause_percent)% of total RAM, this)`. Only used when `enable_vram_budget` is true."""

    ram_per_process_max_mb: int = Field(default=18432, ge=0)
    """Resident system RAM (MB) one inference process may hold before it is a reclaim candidate under pressure.

    A worker that keeps model weights resident for fast reload accumulates them in each process's address
    space, and the allocator does not return those pages to the OS without a respawn. On a multi-process /
    multi-GPU host a single process's footprint can balloon past what the shared RAM pool can hold beside its
    siblings and any co-tenants (an alchemist, a scribe), which is the shape that drives an OS OOM kill.

    When the host is below its RAM danger floor, a process whose resident RAM is at or above this ceiling is
    reclaimed: if idle it is recycled immediately (returning its retained pages), and if busy it is drained
    (fed no new work) and recycled once its in-flight job finishes. This bounds the per-process balloon so the
    summed resident set stays within RAM, rather than relying only on shedding idle siblings (which cannot help
    when every process is busy). The ceiling is only consulted while the host is under the danger floor, so a
    roomy host never recycles needlessly. Set to 0 to disable. Only used when `enable_vram_budget` is true."""

    post_processing_fault_breaker_enabled: bool = Field(default=True)
    """Disable post-processing on this worker after repeated post-processing VRAM-over-commit faults.

    A post-processing peak that cannot be hosted (a single-process worker on a tiny card, or a card a job
    over-commits) faults the job; the horde reassigns it, but a worker that keeps faulting trips the horde's
    forced-maintenance, the very spiral this guards against. When true (the default), the worker counts
    post-processing-over-commit faults (both the planner's unhostable-peak faults and watchdog-reaped
    post-processing stalls) in a rolling window and, once they exceed `post_processing_fault_threshold`
    within `post_processing_fault_window_seconds`, stops *popping* post-processing-requesting jobs and logs an
    operator advisory to downgrade settings. The suppression is session-latched (it clears only on restart),
    since the over-commit is structural and auto-recovery would simply re-trip it."""

    post_processing_fault_threshold: int = Field(default=4, ge=1)
    """The breaker trips when *more than* this many post-processing-over-commit faults occur within
    `post_processing_fault_window_seconds` (so the default tolerates 4 and trips on the 5th).

    Only used when `post_processing_fault_breaker_enabled` is true."""

    post_processing_fault_window_seconds: int = Field(default=1800, ge=60)
    """Rolling window (seconds) over which `post_processing_fault_threshold` is counted.

    Only used when `post_processing_fault_breaker_enabled` is true."""

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
    pre-staging/dispatch for its duration, so it runs on an un-contended device. The isolation only
    attaches when the streaming forecast shows the model's footprint actually dominates the card: a
    card-light model that reaches the best-effort admit through reserve arithmetic alone (free VRAM
    depressed by retained sibling contexts) shares the device, since isolating it would cap a
    multi-thread card at one job for the admit's whole lifetime. Only used when `enable_vram_budget`
    is true."""

    whole_card_exclusive_residency: bool = Field(default=True)
    """Give a model whose weights need most of the device sole residency *before* it streams, not after a fault.

    The streaming forecast (weights vs ComfyUI's inference reserve) flags a model that would offload weights
    to host RAM if loaded alongside the currently-resident models but fits cleanly with the card to itself.
    When true (the default), the scheduler proactively evicts the other processes' resident models, returns
    their freed VRAM to the driver, and suppresses prefetch into sibling slots for its duration, so the model
    loads fully resident and samples at full speed instead of streaming weights and being hang-graded. This
    is the preventative form of `overbudget_exclusive_mode`, which only reacts once a model has already been
    admitted over budget. Only used when `enable_vram_budget` is true.

    This flag governs steady-state preference only. It does not gate the emergency starvation teardown: a
    weight-dominant head starved behind the worker's own idle sibling contexts still tears them down to admit
    even when this is false. See the VRAM arbiter explanation doc for that unconditional liveness path."""

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

    large_model_switch_min_seconds: int = Field(default=0, ge=0, le=600)
    """Minimum seconds between accepting jobs for *different* very-large models (the switch throttle).

    A queue that alternates distinct very-large models (Flux -> Z-Image -> Flux) forces a fresh whole-card
    teardown and a multi-GB checkpoint reload on every switch. When set above 0, once a very-large model (the
    EXTRA_LARGE tier: Flux/Cascade/Qwen/Z-Image and the named VRAM-heavy checkpoints) is loaded or queued, the
    worker stops *offering* a different very-large model in its horde pop request until this many seconds have
    elapsed since the last distinct large model was introduced. Jobs for the large model already in play are
    unaffected; only churning to a new one is throttled, and no job is dropped (it is simply not requested). If
    the worker goes fully idle with an empty local queue it offers large models again regardless, so this never
    leaves the worker idle. 0 disables the throttle (the default)."""

    large_model_reentry_cooldown_seconds: int = Field(default=-1, ge=-1, le=600)
    """Seconds to avoid *any* very-large model after the last one drains (the re-entry cooldown).

    Once the whole-card residency lease is up and no very-large model remains loaded or queued, the worker
    stops offering *any* very-large model for this long, so it does ordinary work for a beat instead of
    immediately re-entering large-model territory and re-thrashing. -1 (the default) inherits
    `whole_card_residency_cooldown_seconds`, tying it to the residency lease it complements; 0 disables it; a
    positive value overrides. As with the switch throttle, an idle worker with an empty local queue offers
    large models again regardless, so it never sits idle. Independent of `whole_card_exclusive_residency`."""

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

    aux_model_download_line_skip_threshold_seconds: int | None = Field(default=3, ge=0)
    """Accepted for configuration compatibility and has no effect.

    Auxiliary (LoRa/TI) files are placed on disk by the dedicated download process before a job is
    dispatchable, so inference children never block on an auxiliary download and there is no aux-download
    line-skip breaker to tune. Existing bridgeData files that set this key continue to load unchanged."""

    idle_fill_threshold_seconds: int | None = Field(default=5, ge=0)
    """Seconds the queue head may sit on an idle device (its model still downloading/loading) with a free
    inference sibling before the worker over-pops a small no-LoRA "fill" job to keep the GPU busy.

    The fill is drawn up a smallest-fastest-first ladder (small sd15 -> larger sd15 -> small sdxl -> larger
    sdxl, skipping baselines the worker has no model for), advancing a rung each time the horde has no job at
    the current rung, so the card is fed the quickest-to-start work available instead of idling while a
    download finishes. LoRA jobs are never offered for the fill (they would themselves block on a download).
    Only ever fires when a head is genuinely starved with a free sibling, so it is inert in steady state.
    Keep unset to disable."""

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

    comfy_smart_memory: bool = Field(default=False)
    """Keep ComfyUI's smart memory management on so inference children hold model weights resident in VRAM
    across jobs.

    With this on, a back-to-back same-model job reuses the resident UNet/CLIP/VAE instead of re-uploading
    them from RAM, eliminating the per-job RAM->VRAM transfers that dominate small-job wall-clock. It is
    OFF by default because cross-process residency is not yet reconciled at dispatch time: a sampling peak
    landing beside an idle sibling's resident weights overcommits a tight card faster than the device-free
    governor's reclaim ladder can evict, and the driver then demotes VRAM to system memory (a card-wide
    slowdown far costlier than the transfers this saves). Enable only for experimentation on cards with
    headroom well above one sampling peak plus one resident model."""

    dry_run_skip_inference: bool = Field(default=False)
    """Skip real GPU inference and return a dummy 1x1 image instead."""

    dry_run_skip_safety: bool = Field(default=False)
    """Skip the safety (NSFW/CSAM) evaluation model."""

    dry_run_skip_post_processing: bool = Field(default=False)
    """Skip real post-processing on the dedicated post-processing process and echo images back."""

    dry_run_skip_api: bool = Field(default=False)
    """Skip API calls (job pop and submit) and use canned scenarios."""

    dry_run_inference_delay: float = Field(default=1.0, ge=0.0)
    """Seconds to sleep when dry-run inference is active, simulating work."""

    @model_validator(mode="after")
    def validate_performance_modes(self) -> Self:
        """Validate and adjust performance mode settings based on cross-field constraints."""
        # Extra slow worker takes priority over all performance/memory settings
        if self.extra_slow_worker:
            apply_extra_slow_clamps(
                self,
                compute_extra_slow_clamps(
                    high_performance_mode=self.high_performance_mode,
                    moderate_performance_mode=self.moderate_performance_mode,
                    queue_size=self.queue_size,
                    max_threads=self.max_threads,
                    preload_timeout=self.preload_timeout,
                    log=True,
                ),
            )

        self.process_timeout = compute_performance_timeout(
            high_performance_mode=self.high_performance_mode,
            moderate_performance_mode=self.moderate_performance_mode,
            default_timeout=reGenBridgeData.model_fields["process_timeout"].default,
            current_timeout=self.process_timeout,
            log=True,
        )

        self.queue_size = cap_queue_size(
            max_threads=self.max_threads,
            queue_size=self.queue_size,
            log=True,
        )

        _warn_lease_without_residency(
            gpu_sampling_lease_enabled=self.gpu_sampling_lease_enabled,
            unload_models_from_vram_often=self.unload_models_from_vram_often,
            log=True,
        )

        _warn_lease_slots_below_threads(
            gpu_sampling_lease_enabled=self.gpu_sampling_lease_enabled,
            gpu_sampling_lease_slots=self.gpu_sampling_lease_slots,
            max_threads=self.max_threads,
            log=True,
        )

        return self

    @model_validator(mode="after")
    def validate_workload_roles(self) -> Self:
        """Warn when no role is selected, so a worker that would serve nothing is not silent.

        A CPU-only install still has alchemy to serve, so this only fires for the genuinely empty
        ``dreamer: false, alchemist: false`` combination. The authoritative derivation of served
        workloads (which also accounts for the CPU install) lives in
        :func:`horde_worker_regen.capabilities.enabled_workloads`.
        """
        if not self.dreamer and not self.alchemist:
            logger.warning(
                "Both `dreamer` and `alchemist` are false, so this worker has nothing to serve. Enable "
                "`dreamer` for image generation or `alchemist` for alchemy forms in bridgeData.yaml.",
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

    @field_validator("forms")
    def validate_alchemy_forms(cls, v: list[str]) -> list[str]:
        """Validate alchemy forms against the SDK enum plus this worker's extra known forms.

        Overrides the SDK's stricter validator (which rejects any form not in its ``KNOWN_ALCHEMY_FORMS``
        enum) so the worker can offer forms it serves ahead of an SDK release: the SDK's pop/async wire
        models already accept unknown form names as plain strings, but its bridge-data validator does not,
        which would otherwise make the config un-loadable. Real typos still raise; only the explicitly
        worker-known extras (see :data:`WORKER_KNOWN_EXTRA_ALCHEMY_FORMS`) are additionally accepted.
        """
        if not isinstance(v, list):
            raise ValueError("forms must be a list")
        validated_forms: list[str] = []
        known_forms = set(KNOWN_ALCHEMY_FORMS.__members__) | set(WORKER_KNOWN_EXTRA_ALCHEMY_FORMS)
        for form in v:
            normalized = str(form).lower().replace("-", "_")
            if normalized not in known_forms:
                raise ValueError(f"Invalid form: {normalized}")
            validated_forms.append(normalized)
        return validated_forms

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

        # AIWORKER_LORA_CACHE_SIZE is consumed by hordelib in megabytes; max_lora_cache_size is gigabytes.
        if self.max_lora_cache_size and os.getenv("AIWORKER_LORA_CACHE_SIZE") is None:
            os.environ["AIWORKER_LORA_CACHE_SIZE"] = str(self.max_lora_cache_size * 1024)

        # The LoRA disk-space floor reaches the manager (in any worker subprocess) in megabytes.
        if os.getenv("AIWORKER_LORA_MIN_DISK_FREE_MB") is None:
            os.environ["AIWORKER_LORA_MIN_DISK_FREE_MB"] = str(round(self.min_lora_disk_free_gb * 1024))

        # The config field is authoritative for large-model loading: the env var is only the transport to
        # the SDK resolver. Because the env var is read process-wide and is never otherwise cleared, setting
        # it only on True would let a stale value (an earlier True run in this process, a TUI reload after
        # True, or an exported shell/Docker value) silently defeat a later `load_large_models: false`. Clear
        # it on False so the config always wins, and surface when a pre-existing value is being overridden.
        if self.load_large_models:
            os.environ["AI_HORDE_MODEL_META_LARGE_MODELS"] = "1"
        elif os.environ.pop("AI_HORDE_MODEL_META_LARGE_MODELS", None) is not None:
            logger.warning(
                "AI_HORDE_MODEL_META_LARGE_MODELS was set but `load_large_models` is false; clearing it so "
                "large models (e.g. Flux, Stable Cascade) are not loaded.",
            )

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
