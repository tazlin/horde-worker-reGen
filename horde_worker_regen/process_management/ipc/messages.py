"""Contains messages (and associated helper types) used for communication between the main process and the child processes."""  # noqa: E501

from __future__ import annotations

import enum
from enum import auto
from typing import ClassVar

from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import (
    GenMetadataEntry,
    ImageGenerateJobPopResponse,
)
from horde_sdk.ai_horde_api.consts import METADATA_TYPE
from horde_sdk.ai_horde_api.fields import GenerationID
from hordelib.metrics import DownloadEvent, JobPhaseMetrics
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from horde_worker_regen.process_management.ipc.supervisor_channel import DownloadStatusSnapshot

AUX_DOWNLOAD_FAILED_INFO = "aux-download-deadline-exceeded"
"""Marker placed in a faulted inference result's ``info`` when a child aborted its own stalled aux download.

The child sets it when an aux (LoRa/TI) download blows the dispatch deadline carried on its control message;
the parent's message dispatcher recognises it to register a LoRA-download backoff strike and apply the
backoff-aware retry policy: the same response as a watchdog teardown, but without tearing the process down."""


class ModelLoadState(enum.Enum):
    """The state of a model.

    e.g., if a model is `IN_USE` or `LOADED_IN_VRAM`
    """

    DOWNLOADING = auto()
    """The model is being downloaded."""
    ON_DISK = auto()
    """The model is on disk. It may or may not be loaded in RAM."""  # TODO: this caveat is subject to change
    LOADING = auto()
    """The model is being loaded into RAM."""
    LOADED_IN_RAM = auto()
    """The model is loaded in RAM."""
    LOADED_IN_VRAM = auto()
    """The model is loaded in VRAM."""
    IN_USE = auto()
    """The model is in use by a process."""
    FAILED = auto()
    """The model could not be loaded (the backend raised while loading it). Reported so the parent can
    track per-model load failures and quarantine a deterministically-unloadable model."""

    def is_loaded(self) -> bool:
        """Return if the model is loaded in RAM, VRAM, or in use."""
        return (
            self == ModelLoadState.LOADED_IN_RAM
            or self == ModelLoadState.LOADED_IN_VRAM
            or self == ModelLoadState.IN_USE
        )

    def is_active(self) -> bool:
        """Return if the model is loaded in VRAM or in use."""
        return self != ModelLoadState.ON_DISK and self != ModelLoadState.FAILED


class ModelInfo(BaseModel):
    """Information about a model loaded or used by a process."""

    horde_model_name: str
    """The name of the model as defined in the horde model reference."""
    horde_model_load_state: ModelLoadState
    """The state of the model."""
    process_id: int
    """The ID of the process that is using the model."""


class HordeProcessState(enum.Enum):
    """The state of a process.

    e.g., if a process is `INFERENCE_STARTING` or `WAITING_FOR_JOB`
    """

    PROCESS_STARTING = auto()
    """The process is starting."""

    PROCESS_ENDING = auto()
    """The process is ending."""
    PROCESS_ENDED = auto()
    """The process has ended."""

    WAITING_FOR_JOB = auto()
    """The process is waiting for a job."""
    JOB_RECEIVED = auto()
    """The process has received a job."""

    DOWNLOADING_MODEL = auto()
    """The process is downloading a model."""
    DOWNLOAD_COMPLETE = auto()
    """The process has finished downloading a model."""

    DOWNLOADING_AUX_MODEL = auto()
    """The process is downloading an auxiliary model. (e.g., LORA)"""
    DOWNLOAD_AUX_COMPLETE = auto()
    """The process has finished downloading an auxiliary model. (e.g., LORA)"""

    PRELOADING_MODEL = auto()
    """The process is preloading a model."""
    PRELOADED_MODEL = auto()
    """The process has finished preloading a model."""
    PRELOADING_FAILED = auto()
    """The process failed to load a model (e.g. an unsupported/corrupt checkpoint the backend cannot load).

    Distinct from a crash: the child names the offending model so the parent can quarantine *that model*
    after repeated failures, instead of mistaking a deterministically-unloadable model for a sick process and
    burning the slot down in an unbounded recovery loop."""

    UNLOADED_MODEL_FROM_VRAM = auto()
    """The process has unloaded a model from VRAM."""
    UNLOADED_MODEL_FROM_RAM = auto()
    """The process has unloaded a model from RAM."""

    INFERENCE_STARTING = auto()
    """The process is starting inference."""
    INFERENCE_COMPLETE = auto()
    """The process has finished inference."""
    INFERENCE_FAILED = auto()
    """The process has failed inference."""

    ALCHEMY_STARTING = auto()
    """The process is starting performing alchemy jobs."""
    ALCHEMY_COMPLETE = auto()
    """The process has finished performing alchemy jobs."""
    ALCHEMY_FAILED = auto()
    """The process has failed performing alchemy jobs."""

    POST_PROCESSING = auto()
    """The dedicated post-processing process is running the post-processing phase of an image job."""
    POST_PROCESSING_COMPLETE = auto()
    """The dedicated post-processing process has finished an image job's post-processing phase."""
    POST_PROCESSING_FAILED = auto()
    """The dedicated post-processing process failed an image job's post-processing phase."""

    EVALUATING_SAFETY = auto()
    """The process is evaluating safety."""
    SAFETY_FAILED = auto()
    """The process has failed evaluating safety."""

    TORCH_GPU_INCOMPATIBLE = auto()
    """The installed PyTorch has no CUDA kernels for this GPU's architecture (reported by an inference child).

    A non-retryable hardware/build mismatch detected at inference-process start, once torch is loaded: the
    wheel's compiled architectures do not include the device's compute capability, so every job would die at
    the first kernel launch with ``no kernel image is available for execution on the device`` (which ComfyUI
    hides behind a generic "no images produced" fault). The child carries the device/build detail in ``info``.
    The orchestrator latches a worker-state flag from this and stops popping jobs, and the TUI surfaces it
    prominently, all without importing torch (the invariant is that only the torch-bearing inference child
    ever touches torch; the parent and TUI learn of the problem through this torch-free signal)."""

    TORCH_BUILD_CPU_ONLY = auto()
    """The installed PyTorch is a CPU-only build, so image generation is disabled (reported by an inference child).

    Distinct from ``TORCH_GPU_INCOMPATIBLE``: nothing is broken and the child does not exit. The torch wheel
    simply has no GPU backend (no CUDA/HIP/XPU/MPS), so image generation would run ~100x slower and is
    disabled, while the CPU-friendly alchemy forms keep running on this same process. The orchestrator
    latches a worker-state flag from this and stops popping *image* jobs (alchemy is unaffected), all
    without importing torch. This is build-based (it would not fire for a merely masked or broken GPU on a
    GPU build), so it is the runtime counterpart to the ``bin/backend`` 'cpu' sentinel: it makes a CPU torch
    build prevent image generation even when the install sentinel was never set (e.g. a manual CPU install).
    A transient signal: the child reports it once at startup, then proceeds to its normal idle state."""


class HordeProcessMessage(BaseModel):
    """Process messages are sent from the child processes to the main process."""

    process_id: int
    """The ID of the process that sent the message."""
    process_launch_identifier: int
    """The identifier of the process launch."""
    info: str
    """Information about this operation sent the process."""
    time_elapsed: float | None = None
    """The time elapsed since the process started."""


class HordeProcessMemoryMessage(HordeProcessMessage):
    """Memory messages that are sent from the child processes to the main process."""

    ram_usage_bytes: int
    """The number of bytes of RAM used by the process."""
    open_fds: int | None = None
    """Open file descriptors/handles held by the process, or None if the platform metric is unavailable.

    Reported so the parent can watch descriptor headroom and surface a leak (which ends in EMFILE, "Too
    many open files") as it grows, rather than only after it has poisoned the slot."""
    fd_soft_limit: int | None = None
    """The process's soft ``RLIMIT_NOFILE`` ceiling, or None where there is no finite limit (e.g. Windows)."""
    vram_usage_mb: int | None = None
    """The MB of VRAM used on the GPU."""
    vram_total_mb: int | None = None
    """The total MB of VRAM available on the GPU."""
    device_index: int = 0
    """The stable index of the GPU this process is pinned to (0 on a single-GPU host).

    A device-pinned child measures its own (masked) device and reports under its global index, so the
    parent can aggregate VRAM usage/total per card. Defaults to 0 for single-GPU and older children."""


class HordeHeartbeatType(enum.Enum):
    """The state of the heartbeat."""

    OTHER = auto()
    PIPELINE_STATE_CHANGE = auto()
    INFERENCE_STEP = auto()


class HordeProcessHeartbeatMessage(HordeProcessMessage):
    """Heartbeat messages that are sent from the child processes to the main process."""

    heartbeat_type: HordeHeartbeatType
    """The type of the heartbeat."""
    process_warning: str | None = None
    """A warning message from the process."""

    percent_complete: int | None = None
    """The percentage (int) of the current operation that is complete, if applicable."""

    current_step: int | None = None
    """The current sampling step of the running inference, if applicable."""
    total_steps: int | None = None
    """The total sampling steps of the running inference, if applicable."""
    iterations_per_second: float | None = None
    """The instantaneous sampling rate (-1.0 when not yet known), if applicable."""

    nonadvancing_step_repeats: int = 0
    """Consecutive progress reports the child has received at the *same* sampling step without advancing.

    A healthy job reports each step (including the final one) exactly once, so this stays 0. When the
    underlying ComfyUI generation loops on a single step (in practice the final step), the child keeps
    receiving identical progress callbacks and keeps emitting heartbeats, so the parent's silence-based
    hang watchdog never fires. The child counts those non-advancing reports and forwards the running
    count here so the parent can reap the wedged slot even though it is not silent."""


class HordeProcessStateChangeMessage(HordeProcessMessage):
    """State change messages that are sent from the child processes to the main process."""

    process_state: HordeProcessState
    """The state of the process."""


class HordeModelStateChangeMessage(HordeProcessStateChangeMessage):
    """Model state change messages that are sent from the child processes to the main process.

    See also `ModelLoadState`.
    """

    horde_model_name: str
    """The name of the model as defined in the horde model reference."""
    horde_model_state: ModelLoadState
    """The state of the model."""


class HordeAuxModelStateChangeMessage(HordeProcessStateChangeMessage):
    """Auxiliary model state change messages that are sent from the child processes to the main process.

    See also `ModelLoadState`.
    """

    sdk_api_job_info: ImageGenerateJobPopResponse | None = None
    """If the model state change is related to a job, the job as sent by the API."""


class HordeDownloadProgressMessage(HordeModelStateChangeMessage):
    """Download progress messages that are sent from the child processes to the main process."""

    total_downloaded_bytes: int
    """The total number of bytes downloaded so far."""
    total_bytes: int
    """The total number of bytes that will be downloaded."""

    @property
    def progress_percent(self) -> float:
        """The progress of the download as a percentage."""
        return self.total_downloaded_bytes / self.total_bytes * 100


class HordeDownloadCompleteMessage(HordeModelStateChangeMessage):
    """Download complete messages that are sent from the child processes to the main process."""


class HordeDownloadAvailabilityMessage(HordeProcessMessage):
    """A full snapshot of on-disk image-model availability, sent by the dedicated download process.

    Unlike the per-process model-state messages, this is not tied to an entry in the process map:
    the download process lives outside it. The snapshot is sent once after the process loads its
    model managers and again after each model finishes (or fails) downloading, so the main process
    can keep its advertised-models set in sync without per-event bookkeeping.
    """

    available_model_names: list[str]
    """Every configured image model currently present on disk."""
    currently_downloading: str | None = None
    """The model being downloaded right now, if any."""
    pending_downloads: list[str] = Field(default_factory=list)
    """Models still queued to download (excludes the one in progress)."""
    failed_downloads: list[str] = Field(default_factory=list)
    """Models whose download was attempted and failed (will not be retried automatically)."""
    scan_complete: bool = True
    """False for early initializing/scanning reports whose on-disk set is not yet authoritative."""
    safety_models_present: bool = False
    """True once the required safety models (DeepDanbooru + CLIP) are confirmed on disk. The main
    process defers the safety-process launch until this is set, so the safety process finds them
    already downloaded instead of fetching ~2.3GB synchronously (and invisibly) in its constructor."""
    safety_models_attempted: bool = False
    """True once the download process has finished its one-shot ensure of the safety models, whether it
    succeeded or failed. Lets the main process tell 'not tried yet' (a transient post-scan idle report,
    keep waiting) apart from 'tried and could not provide them' (start the safety process so it self-fetches
    and surfaces the real error). Always True alongside ``safety_models_present``."""
    controlnet_present: bool | None = None
    """On-disk readiness of the ControlNet feature (its models plus the annotators), or None until probed.
    None means undeterminable (the manager is not loaded, e.g. the feature is not opted in), which the
    parent treats as "do not gate", mirroring image-model availability."""
    sdxl_controlnet_present: bool | None = None
    """On-disk readiness of the SDXL-ControlNet feature (its ControlNet models, the annotators, and the
    auxiliary miscellaneous models), or None when undeterminable."""
    post_processing_present: bool | None = None
    """On-disk readiness of the post-processing feature (the GFPGAN/ESRGAN/CodeFormer models), or None
    when undeterminable."""
    controlnet_failed: bool = False
    """True once the ControlNet annotator verify has permanently failed (the detector checkpoints download
    but do not run, even after one re-fetch). ControlNet is then withheld and the operator is notified;
    distinct from ``controlnet_present=False`` (still downloading), which recovers on its own."""
    status: DownloadStatusSnapshot | None = None
    """Rich, display-oriented status (phase, current download, queue, failures) for the TUI/console."""
    reference_changed: bool = False
    """True on the snapshot following a completed download that altered on-disk references (a new image
    model, or the LoRa/TI/aux pass). The main process broadcasts a reload so inference subprocesses
    re-read the updated reference (notably lora.json/ti.json) without a restart."""


class HordeImageResult(BaseModel):
    """Contains information about a single image that has been generated in a job."""

    image_bytes: bytes
    """The encoded bytes (PNG) of one image generated by the job."""
    generation_faults: list[GenMetadataEntry] = Field(default_factory=list)
    """The generation faults recorded for that image."""


class HordeInferenceResultMessage(HordeProcessMessage):
    """Inference result messages that are sent from the child processes to the main process."""

    job_image_results: list[HordeImageResult] | None = None
    """The per-image results (encoded image bytes plus faults) generated by the job."""
    state: GENERATION_STATE
    """The state of the job to be sent to the API."""
    sdk_api_job_info: ImageGenerateJobPopResponse

    non_reportable_faults: ClassVar[list[METADATA_TYPE]] = [METADATA_TYPE.aesthetic_score, METADATA_TYPE.information]

    @property
    def faults_count(self) -> int:
        """Return a count of all generation faults."""
        if self.job_image_results is None:
            return 0
        total = 0
        for f in self.job_image_results:
            if f.generation_faults is not None:
                total += sum(
                    1
                    for fault in f.generation_faults
                    if fault not in HordeInferenceResultMessage.non_reportable_faults
                )
        return total


class HordeJobMetricsMessage(HordeProcessMessage):
    """Per-job performance metrics, sent by a child right after a job (or alchemy form) finishes.

    Carries the snapshot from hordelib's in-process metrics collector: model-load phase
    timings (disk->RAM, RAM->VRAM), sampling stats (steps, iterations/second), and
    memory high-water marks observed during the job.
    """

    job_id: str
    """The generation ID of the image job, or the form ID of the alchemy form."""
    is_alchemy: bool = False
    """Whether these metrics belong to an alchemy form rather than an image job."""
    phase_metrics: JobPhaseMetrics
    """The per-job metrics snapshot from hordelib."""


class HordeDownloadMetricsMessage(HordeProcessMessage):
    """Completed ad-hoc download events (lora/ti), drained from hordelib's metrics collector.

    Downloads run on background threads with no job affinity, so they are reported
    separately from per-job metrics whenever the child finds drained events.
    """

    events: list[DownloadEvent]
    """The download events observed since the last drain."""


class HordeSafetyEvaluation(BaseModel):
    """The result of a safety evaluation."""

    is_nsfw: bool
    """If the image is NSFW."""
    is_csam: bool
    """If the image is CSAM."""
    replacement_image_bytes: bytes | None
    """The encoded bytes of the replacement image if it was censored."""
    failed: bool = False
    """If the safety evaluation failed."""
    aesthetic_score: float | None = None
    """The LAION 0-10 aesthetic score for the image, computed for free from the CLIP embedding the
    safety pass already produces. ``None`` when scoring is disabled or the predictor is unavailable."""


class HordeSafetyResultMessage(HordeProcessMessage):
    """Safety result messages that are sent from the child processes to the main process."""

    job_id: GenerationID
    """The ID of the job that was evaluated."""
    safety_evaluations: list[HordeSafetyEvaluation]
    """A list of safety evaluations for each image in the job."""


class HordeControlFlag(enum.Enum):
    """Control flags are sent from the main process to the child processes."""

    DOWNLOAD_MODEL = auto()
    """Signal the child process to download a model."""
    PRELOAD_MODEL = auto()
    """Signal the child process to preload a model."""
    START_INFERENCE = auto()
    """Signal the child process to start inference."""
    START_ALCHEMY = auto()
    """Signal the child process to run an alchemy form (upscale, caption, etc.)."""
    START_POST_PROCESS = auto()
    """Signal the dedicated post-processing process to run an image job's post-processing phase."""
    EVALUATE_SAFETY = auto()
    """Signal the child process to evaluate safety of images from inference."""
    UNLOAD_MODELS_FROM_VRAM = auto()
    """Signal the child process to unload models from VRAM."""
    UNLOAD_MODELS_FROM_RAM = auto()
    """Signal the child process to unload models from RAM."""
    DOWNLOAD_MODELS = auto()
    """Signal the dedicated download process to ensure a set of models are present on disk."""
    RELOAD_MODEL_DATABASE = auto()
    """Signal a child process to reload its model managers' references from disk (no download).

    Sent after the parent refreshes the on-disk reference, or after the download process reports new
    LoRa/TI availability, so subprocesses pick up the changes without restarting."""
    END_PROCESS = auto()
    """Signal the child process to end."""


class HordeControlMessage(BaseModel):
    """Control messages are sent from the main process to the child processes."""

    control_flag: HordeControlFlag
    """The control flag signaling the child process to perform an action."""


class HordeControlModelMessage(HordeControlMessage):
    """Control messages that are sent from the main process to the child processes that involve models."""

    horde_model_name: str
    """The name of the model as defined in the horde model reference."""

    aux_download_deadline_seconds: float | None = None
    """Wall-clock budget for this job's auxiliary (LoRa/TI) downloads before the child gives up.

    Set by the parent to its own (backoff-aware) stuck-aux watchdog timeout minus a margin, so the child
    cancels a stalled download and faults the job *itself*, keeping the inference process alive, a beat
    before the watchdog would otherwise tear the whole process down. ``None`` means no child-side deadline
    (the watchdog remains the only backstop), preserving behaviour for any sender that does not set it."""


class HordeDownloadControlMessage(HordeControlMessage):
    """Ask the dedicated download process to ensure a set of image models are present on disk.

    The process downloads any not already present, sequentially in the background, reporting
    a fresh :class:`HordeDownloadAvailabilityMessage` snapshot as each completes.
    """

    control_flag: HordeControlFlag = HordeControlFlag.DOWNLOAD_MODELS
    model_names: list[str] = Field(default_factory=list)
    """The horde image-model names to ensure are downloaded."""
    desired_image_models: list[str] | None = None
    """When set, the authoritative configured image-model set. The download process prunes any pending
    download not in this set and aborts the in-flight download if it is an image model not in this set,
    so a config edit that removes a model stops it downloading. ``None`` means no reconciliation (an
    additive-only message), preserving callers that only add work or set pause/rate controls."""
    download_aux: bool = False
    """If True, also run the one-time auxiliary/default downloads (LoRa defaults, controlnet,
    post-processing, safety helpers) permitted by the worker config."""
    set_paused: bool | None = None
    """If not None, pause (True) or resume (False) downloads; applied live, mid-download."""
    set_rate_limit_kbps: int | None = None
    """If not None, set the bandwidth cap in kB/s; 0 or negative clears the limit."""
    set_max_parallel_downloads: int | None = None
    """If not None, retune the global concurrent-download ceiling (across all hosts), applied live."""
    set_per_host_concurrency: int | None = None
    """If not None, retune how many concurrent downloads to a single host are allowed, applied live."""
    set_connections_per_file: int | None = None
    """If not None, retune the max concurrent connections used to fetch a single large file, applied live."""
    set_nsfw: bool | None = None
    """If not None, retune nsfw filtering of the default-LoRa pass live (replaces a download-process restart)."""
    set_allow_lora: bool | None = None
    """If not None, enable/disable the LoRa aux category live. Enabling re-arms the one-shot aux pass so the
    newly-permitted category downloads without restarting the process; disabling just stops future enqueues."""
    set_allow_controlnet: bool | None = None
    """If not None, enable/disable the ControlNet aux category live (re-arms the aux pass when enabling)."""
    set_allow_sdxl_controlnet: bool | None = None
    """If not None, enable/disable the SDXL-ControlNet aux category live (re-arms the aux pass when enabling)."""
    set_allow_post_processing: bool | None = None
    """If not None, enable/disable the post-processing aux category live (re-arms the aux pass when enabling)."""
    set_purge_loras: bool | None = None
    """If not None, retune whether the default-LoRa pass purges unused LoRas, applied live."""


class HordePreloadInferenceModelMessage(HordeControlModelMessage):
    """Preload model (for image generation) messages that are sent from the main process to the child processes."""

    will_load_loras: bool
    """If the model will be patched with LoRa(s)."""
    seamless_tiling_enabled: bool
    """If seamless tiling will be enabled."""

    sdk_api_job_info: ImageGenerateJobPopResponse

    trace_context: str | None = None
    """W3C traceparent string for cross-process span correlation."""


class HordeInferenceControlMessage(HordeControlModelMessage):
    """Inference control messages that are sent from the main process to the child processes."""

    sdk_api_job_info: ImageGenerateJobPopResponse
    """The job as sent by the API."""

    keep_model_resident_after: bool = False
    """Keep this job's model resident in VRAM after the run instead of evicting it.

    Set by the scheduler only when the next pending-inference job reuses the same model and the VRAM
    budget confirms it can stay resident across the live process set, so the back-to-back force-reload
    (the dominant non-sampling cost on small jobs) is skipped. Defaults to False, preserving the
    aggressive per-job eviction that keeps sibling GPU instances from over-committing."""

    trace_context: str | None = None
    """W3C traceparent string for cross-process span correlation."""


class AlchemyFormSpec(BaseModel):
    """A single alchemy form (one unit of alchemy work) ready for a child process.

    The source image is already downloaded and decoded to raw bytes by the main process, so
    child processes never perform network IO for job inputs.
    """

    form_id: str
    """The generation ID of this form, used for submit."""
    form: str
    """The form name (a `KNOWN_ALCHEMY_TYPES` value; kept as str to survive unknown forms)."""
    source_image_bytes: bytes
    """The encoded source image bytes to process."""
    r2_upload: str | None = None
    """The R2 URL to upload image-form results to, when applicable."""


class HordeAlchemyControlMessage(HordeControlMessage):
    """Dispatch one alchemy form to a child process (control_flag == START_ALCHEMY)."""

    form: AlchemyFormSpec
    """The alchemy form to process."""

    trace_context: str | None = None
    """W3C traceparent string for cross-process span correlation."""


class HordeAlchemyResultMessage(HordeProcessMessage):
    """The result of one alchemy form, sent from a child process to the main process.

    Matches the legacy alchemist submit protocol: text forms carry ``result_payload``
    (e.g. ``{"caption": "..."}``); image forms carry ``image_bytes`` (WebP, ready for R2)
    and submit ``{"<form>": "R2"}``.
    """

    form_id: str
    """The generation ID of the form that was processed."""
    form: str
    """The form name that was processed."""
    state: GENERATION_STATE
    """The state of the form to send to the API."""
    result_payload: dict[str, str | bool | dict | list] | None = None
    """The inline result for text forms (caption/interrogation/nsfw)."""
    image_bytes: bytes | None = None
    """The WebP-encoded result image bytes for graph forms, to be uploaded to R2."""


class HordeSafetyControlMessage(HordeControlMessage):
    """Message with images and other information to be evaluated for safety."""

    job_id: GenerationID
    """The ID of the job that was evaluated."""
    prompt: str
    """The prompt used to generate the images."""
    censor_nsfw: bool
    """If NSFW images should be censored."""
    sfw_worker: bool
    """If the worker is SFW."""
    images_bytes: list[bytes]
    """The encoded bytes of the images generated by the job."""
    horde_model_info: ImageGenerationModelRecord | None = None
    """The model info as defined in the horde model reference."""
    include_aesthetic_score: bool = False
    """If set, the safety pass also scores each image with the LAION aesthetic head and returns the
    score on each evaluation, to be attached as per-generation ``gen_metadata`` (worker-configurable)."""

    @model_validator(mode="after")
    def validate_censor_flags_logical(self) -> HordeSafetyControlMessage:
        """Validate that the censor flags are logical (reasonable)."""
        if not self.censor_nsfw and self.sfw_worker:
            logger.warning("HordeSafetyControlMessage: sfw_worker is True but censor_nsfw is False")
            self.censor_nsfw = True

        return self


class HordePostProcessControlMessage(HordeControlMessage):
    """Dispatch an image job's post-processing phase to the dedicated post-processing process.

    Carries the raw (pre-post-processing) images and the requested post-processor list, so the
    post-processing process runs the same per-image, per-operation loop the inference process used
    to run inline, but on models it keeps resident.
    """

    control_flag: HordeControlFlag = HordeControlFlag.START_POST_PROCESS
    job_id: GenerationID
    """The ID of the job whose images are to be post-processed."""
    images_bytes: list[bytes]
    """The encoded bytes of the raw images to post-process, in generation order."""
    post_processing: list[str]
    """The requested post-processor names (upscalers/face-fixers/strip-background), in request order."""
    trace_context: str | None = None
    """W3C traceparent string for cross-process span correlation."""


class HordePostProcessResultMessage(HordeProcessMessage):
    """The result of an image job's post-processing phase, sent from the post-processing process.

    Mirrors :class:`HordeInferenceResultMessage`: the processed images (and any faults recorded during
    post-processing) replace the raw images before the job proceeds to the safety stage.
    """

    job_id: GenerationID
    """The ID of the job whose images were post-processed."""
    job_image_results: list[HordeImageResult] | None = None
    """The post-processed per-image results, or None if post-processing faulted with no usable output."""
    state: GENERATION_STATE
    """The state of the job to be sent to the API (``ok`` or ``faulted``)."""
