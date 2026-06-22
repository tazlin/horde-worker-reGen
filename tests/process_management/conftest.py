"""Shared fixtures for process management tests."""

from __future__ import annotations

import uuid
from unittest.mock import Mock

import pytest
from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry
from pydantic import JsonValue

from horde_worker_regen.process_management.api_sessions import ApiSessions
from horde_worker_regen.process_management.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeProcessState
from horde_worker_regen.process_management.model_metadata import ModelMetadata
from horde_worker_regen.process_management.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_manager import (
    HordeWorkerProcessManager,
    MultiprocessingPrimitives,
    SystemResources,
)
from horde_worker_regen.process_management.runtime_config import RuntimeConfig


async def track_popped_job_async(
    job_tracker: JobTracker,
    job: object,
    *,
    time_popped: float | None = None,
) -> ImageGenerateJobPopResponse:
    """Record a popped job in JobTracker using the async mutation API.

    If ``job`` is not already an ``ImageGenerateJobPopResponse``, this helper
    will coerce it into one using any compatible attributes found on the object.
    """
    if isinstance(job, ImageGenerateJobPopResponse):
        pop_response = job
    else:
        payload = getattr(job, "payload", None)

        def _coerce_int(value: object, default: int) -> int:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return default
            return default

        model = getattr(job, "model", "stable_diffusion") or "stable_diffusion"
        width = _coerce_int(getattr(payload, "width", 512), 512)
        height = _coerce_int(getattr(payload, "height", 512), 512)
        ddim_steps = _coerce_int(getattr(payload, "ddim_steps", 30), 30)
        n_iter = _coerce_int(getattr(payload, "n_iter", 1), 1)
        seed = getattr(payload, "seed", "42")
        prompt = getattr(payload, "prompt", "test prompt")

        loras = getattr(payload, "loras", None)
        if not isinstance(loras, list):
            loras = None

        r2_upload = getattr(job, "r2_upload", None)
        if r2_upload is not None and not isinstance(r2_upload, str):
            r2_upload = None

        pop_response = make_job_pop_response(
            model=str(model),
            width=width,
            height=height,
            ddim_steps=ddim_steps,
            n_iter=n_iter,
            seed=str(seed) if seed is not None else "42",
            prompt=str(prompt) if prompt is not None else "test prompt",
            loras=loras,
            r2_upload=r2_upload,
        )

    await job_tracker.record_popped_job(pop_response, time_popped=time_popped)
    return pop_response


def make_mock_job(
    *,
    model: str = "stable_diffusion",
    width: int = 512,
    height: int = 512,
    ddim_steps: int = 30,
    n_iter: int = 1,
    post_processing: list[str] | None = None,
    loras: list[LorasPayloadEntry] | None = None,
    control_type: str | None = None,
    hires_fix: bool = False,
    workflow: object | None = None,
) -> Mock:
    """Create a mock job with sensible defaults for testing.

    The returned mock has a ``.payload`` sub-mock with standard image-generation
    fields sufficient for megapixelstep calculation and job tracking. Callers
    can override or add any attribute after creation.
    """
    job = Mock()
    job.model = model
    job.payload.width = width
    job.payload.height = height
    job.payload.ddim_steps = ddim_steps
    job.payload.n_iter = n_iter
    default_post_processing: list[str] = []
    job.payload.post_processing = post_processing if post_processing is not None else default_post_processing
    default_loras: list[LorasPayloadEntry] = []
    job.payload.loras = loras if loras is not None else default_loras
    job.payload.control_type = control_type
    job.payload.hires_fix = hires_fix
    job.payload.workflow = workflow
    return job


async def mark_job_in_progress_async(job_tracker: JobTracker, job: object) -> None:
    """Mark a job as in-progress using the async mutation API."""
    await job_tracker.mark_inference_started(job)  # type: ignore[arg-type]


async def queue_job_for_safety_async(job_tracker: JobTracker, job_info: object) -> None:
    """Queue a completed job for safety checking via JobTracker."""
    await job_tracker.queue_for_safety(job_info)  # type: ignore[arg-type]


async def queue_job_for_submit_async(job_tracker: JobTracker, job_info: object) -> None:
    """Queue a completed job for submission via JobTracker."""
    await job_tracker.queue_for_submit(job_info)  # type: ignore[arg-type]


async def move_job_to_being_safety_checked_async(job_tracker: JobTracker, job_info: object) -> None:
    """Move a job into the being-safety-checked collection via JobTracker."""
    await queue_job_for_safety_async(job_tracker, job_info)
    await job_tracker.begin_safety_check(job_info)  # type: ignore[arg-type]


async def add_job_fault_async(job_tracker: JobTracker, job_id: object, fault_entry: object) -> None:
    """Append a fault entry for a job via JobTracker."""
    await job_tracker.record_source_image_fault(job_id, fault_entry)  # type: ignore[arg-type]


def make_mock_bridge_data(**overrides: object) -> Mock:
    """Create a mock reGenBridgeData with sensible defaults.

    Any keyword argument overrides the corresponding attribute on the mock.
    """
    bd = Mock()
    bd.image_models_to_load = ["stable_diffusion"]
    bd.custom_models = []  # pyrefly: ignore - this field is required but not relevant to our tests, so we can just set it to an empty list
    bd.extra_model_directories = []
    bd.max_threads = 1
    bd.queue_size = 1
    bd.high_performance_mode = False
    bd.moderate_performance_mode = False
    bd.safety_on_gpu = False
    bd.nsfw = False
    bd.post_process_job_overlap = False
    bd.unload_models_from_vram_often = False
    bd.gpu_sampling_lease_enabled = False
    bd.gpu_sampling_lease_slots = 1
    bd.cycle_process_on_model_change = False
    bd.very_fast_disk_mode = False
    bd.remove_maintenance_on_init = False
    bd.stats_output_frequency = 20.0
    bd.process_timeout = 120
    bd.inference_step_timeout = 60
    bd.inference_first_step_timeout = 120
    bd.contended_step_timeout = 120
    bd.overbudget_step_timeout = 120
    bd.overbudget_exclusive_mode = True
    bd.whole_card_residency_safety_off_gpu = True
    bd.whole_card_residency_cooldown_seconds = 0
    bd.unservable_model_fault_threshold = 3
    bd.unservable_model_cooldown_seconds = 900
    bd.self_maintenance_fault_threshold = 6
    bd.self_maintenance_window_seconds = 600
    bd.self_maintenance_cooldown_seconds = 300
    bd.max_inference_attempts = 2
    bd.preload_timeout = 120
    bd.download_timeout = 120
    bd.post_process_timeout = 60
    bd.max_batch = 1
    bd.max_power = 8
    bd.exit_on_unhandled_faults = False
    bd.suppress_speed_warnings = True
    bd.capture_kudos_training_data = False
    bd.limited_console_messages = False
    # The SDK validates API keys as exactly 22 characters
    bd.api_key = "T" * 22
    bd.dreamer_worker_name = "test-worker"
    bd.horde_model_stickiness = 0
    bd.blacklist = []  # pyrefly: ignore - this field is required but not relevant to our tests, so we can just set it to an empty list
    bd.require_upfront_kudos = False
    bd.allow_img2img = True
    bd.allow_inpainting = True
    bd.allow_unsafe_ip = False
    bd.allow_post_processing = True
    bd.allow_controlnet = True
    bd.allow_sdxl_controlnet = True
    bd.extra_slow_worker = False
    bd.limit_max_steps = False
    bd.allow_lora = True
    bd.min_lora_disk_free_gb = 1.0
    bd.alchemist = False
    bd.alchemy_allow_concurrent = True
    bd.alchemy_max_concurrency = 1
    bd.alchemy_vram_headroom_mb = 2000
    bd.alchemy_ram_headroom_mb = 2048
    bd.alchemy_caption_enabled = False
    bd.forms = []
    bd._loaded_from_env_vars = False
    bd.dry_run_skip_inference = False
    bd.dry_run_skip_safety = False
    bd.dry_run_skip_api = False
    bd.dry_run_inference_delay = 1.0
    for k, v in overrides.items():
        setattr(bd, k, v)
    return bd


def make_mock_sd_reference() -> dict[str, Mock]:
    """Create an empty stand-in for the stable diffusion model reference dict."""
    return {}


def make_test_runtime_config(
    bridge_data: Mock | None = None,
    **bridge_overrides: object,
) -> RuntimeConfig:
    """Create a RuntimeConfig wrapping a mock bridge data for testing."""
    if bridge_data is None:
        bridge_data = make_mock_bridge_data(**bridge_overrides)
    return RuntimeConfig(initial=bridge_data)  # type: ignore[arg-type]


def make_test_api_sessions(
    *,
    horde_client_session: object | None = None,
    aiohttp_session: object | None = None,
) -> ApiSessions:
    """Create an ApiSessions with optional mocked session handles."""
    sessions = ApiSessions()
    if horde_client_session is not None:
        sessions.set_horde_client_session(horde_client_session)  # type: ignore[arg-type]
    if aiohttp_session is not None:
        sessions.set_aiohttp_session(aiohttp_session)  # type: ignore[arg-type]
    return sessions


def make_test_model_metadata(
    reference: object | None = None,
) -> ModelMetadata:
    """Create a ModelMetadata, optionally pre-loaded with a reference."""
    metadata = ModelMetadata()
    if reference is not None:
        metadata.set_reference(reference)  # type: ignore[arg-type]
    return metadata


def make_job_pop_response(
    model: str = "stable_diffusion",
    *,
    width: int = 512,
    height: int = 512,
    ddim_steps: int = 30,
    n_iter: int = 1,
    seed: str = "42",
    prompt: str = "test prompt",
    loras: list[LorasPayloadEntry] | None = None,
    r2_upload: str | None = None,
) -> ImageGenerateJobPopResponse:
    """Create a real ImageGenerateJobPopResponse for testing."""
    job_id = uuid.uuid4()
    data: dict[str, JsonValue] = {
        "id": str(job_id),
        "ids": [str(job_id)],
        "model": model,
        "payload": {
            "prompt": prompt,
            "width": width,
            "height": height,
            "ddim_steps": ddim_steps,
            "n_iter": n_iter,
            "seed": seed,
            "sampler_name": "k_euler",
        },
        "skipped": {},
        "source_processing": "txt2img",
    }
    if loras is not None:
        data["payload"]["loras"] = loras  # pyrefly: ignore - type safety doesn't matter here; violations will be caught elsewhere
    if r2_upload is not None:
        data["r2_upload"] = r2_upload
    return ImageGenerateJobPopResponse(**data)  # pyrefly: ignore - type violations will be caught by pydantic


def make_test_system_resources(
    total_ram_bytes: int = 32 * 1024 * 1024 * 1024,
) -> SystemResources:
    """Create a SystemResources with fake hardware info."""
    device_map = TorchDeviceMap(
        root={
            0: TorchDeviceInfo(device_name="TestGPU", device_index=0, total_memory=8 * 1024 * 1024 * 1024),
        },
    )
    return SystemResources(total_ram_bytes=total_ram_bytes, device_map=device_map)


def make_test_mp_primitives() -> MultiprocessingPrimitives:
    """Create MultiprocessingPrimitives with mocks instead of real OS primitives."""
    return MultiprocessingPrimitives(
        process_message_queue=Mock(),
        inference_semaphore=Mock(),
        disk_lock=Mock(),
        aux_model_lock=Mock(),
        vae_decode_semaphore=Mock(),
        gpu_sampling_lease=Mock(),
        download_bandwidth_semaphore=Mock(),
    )


def make_testable_process_manager(
    bridge_data: Mock | None = None,
    stable_diffusion_reference: dict[str, Mock] | None = None,
    system_resources: SystemResources | None = None,
    mp_primitives: MultiprocessingPrimitives | None = None,
    **bridge_overrides: object,
) -> HordeWorkerProcessManager:
    """Build a HordeWorkerProcessManager that can be constructed without torch/psutil/network.

    All external dependencies are replaced with mocks or test doubles.
    """
    if bridge_data is None:
        bridge_data = make_mock_bridge_data(**bridge_overrides)
    if stable_diffusion_reference is None:
        stable_diffusion_reference = make_mock_sd_reference()
    if system_resources is None:
        system_resources = make_test_system_resources()
    if mp_primitives is None:
        mp_primitives = make_test_mp_primitives()

    mock_ctx = Mock()
    mock_model_ref_manager = Mock()

    return HordeWorkerProcessManager(
        ctx=mock_ctx,
        bridge_data=bridge_data,
        horde_model_reference_manager=mock_model_ref_manager,
        max_safety_processes=1,
        system_resources=system_resources,
        mp_primitives=mp_primitives,
        skip_api_init=True,
        stable_diffusion_reference=stable_diffusion_reference,
    )


def make_mock_process_info(
    process_id: int = 0,
    *,
    model_name: str | None = "stable_diffusion",
    state: HordeProcessState = HordeProcessState.WAITING_FOR_JOB,
    process_type: HordeProcessType = HordeProcessType.INFERENCE,
    safe_send_returns: bool = True,
) -> HordeProcessInfo:
    """Create a real HordeProcessInfo with a mocked mp_process and pipe_connection.

    Uses real HordeProcessInfo so Pydantic models that require it (e.g. NextJobAndProcess)
    pass validation.
    """
    mp_process = Mock()
    mp_process.is_alive.return_value = True
    # A started process has a real integer OS pid (and no exitcode yet); mirror that so HordeProcessInfo
    # captures an int os_pid rather than a Mock (the action ledger validates os_pid as int | None).
    mp_process.pid = 100000 + process_id
    mp_process.exitcode = None
    pipe_connection = Mock()
    if not safe_send_returns:
        pipe_connection.send.side_effect = Exception("mock send failure")

    proc = HordeProcessInfo(
        mp_process=mp_process,
        pipe_connection=pipe_connection,
        process_id=process_id,
        process_type=process_type,
        last_process_state=state,
        process_launch_identifier=0,
    )
    proc.loaded_horde_model_name = model_name
    return proc


def make_mock_model_reference_record(
    model_name: str = "stable_diffusion",
    description: str = "A test model",
    baseline: KNOWN_IMAGE_GENERATION_BASELINE = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
    nsfw: bool = False,
) -> ImageGenerationModelRecord:
    """Create a mock ImageGenerationModelRecord with the given name."""
    return ImageGenerationModelRecord(
        name=model_name,
        baseline=baseline,
        nsfw=nsfw,
        description=description,
    )


@pytest.fixture()
def mock_job_pop_response() -> ImageGenerateJobPopResponse:
    """Create a concrete ImageGenerateJobPopResponse."""
    return make_job_pop_response(model="stable_diffusion")


@pytest.fixture()
def mock_horde_job_info(mock_job_pop_response: Mock) -> Mock:
    """Create a mock HordeJobInfo wrapping a pop response."""
    job_info = Mock()
    job_info.sdk_api_job_info = mock_job_pop_response
    job_info.state = None
    job_info.time_popped = 0.0
    job_info.time_to_generate = None
    job_info.censored = None
    job_info.job_image_results = None
    return job_info


@pytest.fixture()
def job_tracker() -> JobTracker:
    """Create a fresh JobTracker instance."""
    return JobTracker()


@pytest.fixture()
def mock_process_info() -> HordeProcessInfo:
    """Create a mock HordeProcessInfo."""
    return make_mock_process_info()


@pytest.fixture()
def process_manager() -> HordeWorkerProcessManager:
    """Create a testable HordeWorkerProcessManager with all external deps mocked."""
    return make_testable_process_manager()
