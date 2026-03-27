"""Shared fixtures for process management tests."""

from __future__ import annotations

import uuid
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeProcessState
from horde_worker_regen.process_management.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_manager import (
    HordeWorkerProcessManager,
    MultiprocessingPrimitives,
    SystemResources,
)


def make_mock_bridge_data(**overrides: object) -> Mock:
    """Create a mock reGenBridgeData with sensible defaults.

    Any keyword argument overrides the corresponding attribute on the mock.
    """
    bd = Mock()
    bd.image_models_to_load = ["stable_diffusion"]
    bd.custom_models = []
    bd.max_threads = 1
    bd.queue_size = 1
    bd.high_memory_mode = False
    bd.high_performance_mode = False
    bd.moderate_performance_mode = False
    bd.safety_on_gpu = False
    bd.nsfw = False
    bd.post_process_job_overlap = False
    bd.unload_models_from_vram_often = False
    bd.cycle_process_on_model_change = False
    bd.very_fast_disk_mode = False
    bd.remove_maintenance_on_init = False
    bd.stats_output_frequency = 20.0
    bd.process_timeout = 120
    bd.inference_step_timeout = 60
    bd.preload_timeout = 120
    bd.download_timeout = 120
    bd.post_process_timeout = 60
    bd.max_batch = 1
    bd.max_power = 8
    bd.exit_on_unhandled_faults = False
    bd.suppress_speed_warnings = True
    bd.capture_kudos_training_data = False
    bd.limited_console_messages = False
    bd.api_key = "test-api-key"
    bd.dreamer_worker_name = "test-worker"
    bd.horde_model_stickiness = 0
    bd.blacklist = []
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
    bd._loaded_from_env_vars = False
    bd.dry_run_skip_inference = False
    bd.dry_run_skip_safety = False
    bd.dry_run_skip_api = False
    bd.dry_run_inference_delay = 1.0
    for k, v in overrides.items():
        setattr(bd, k, v)
    return bd


def make_mock_sd_reference() -> Mock:
    """Create a mock StableDiffusion_ModelReference."""
    ref = Mock()
    ref.root = {}
    return ref


def make_job_pop_response(
    model: str = "stable_diffusion",
    *,
    width: int = 512,
    height: int = 512,
    ddim_steps: int = 30,
    n_iter: int = 1,
    seed: str = "42",
    prompt: str = "test prompt",
    loras: list[object] | None = None,
    r2_upload: str | None = None,
) -> ImageGenerateJobPopResponse:
    """Create a real ImageGenerateJobPopResponse for testing."""
    job_id = uuid.uuid4()
    data: dict = {
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
        data["payload"]["loras"] = loras
    if r2_upload is not None:
        data["r2_upload"] = r2_upload
    return ImageGenerateJobPopResponse(**data)


def make_test_system_resources(
    total_ram_bytes: int = 32 * 1024 * 1024 * 1024,
) -> SystemResources:
    """Create a SystemResources with fake hardware info."""
    device_map = TorchDeviceMap(root={
        0: TorchDeviceInfo(device_name="TestGPU", device_index=0, total_memory=8 * 1024 * 1024 * 1024),
    })
    return SystemResources(total_ram_bytes=total_ram_bytes, device_map=device_map)


def make_test_mp_primitives() -> MultiprocessingPrimitives:
    """Create MultiprocessingPrimitives with mocks instead of real OS primitives."""
    return MultiprocessingPrimitives(
        process_message_queue=Mock(),
        inference_semaphore=Mock(),
        disk_lock=Mock(),
        aux_model_lock=Mock(),
        vae_decode_semaphore=Mock(),
    )


def make_testable_process_manager(
    bridge_data: Mock | None = None,
    stable_diffusion_reference: Mock | None = None,
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
        max_download_processes=1,
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


@pytest.fixture()
def mock_job_pop_response() -> Mock:
    """Create a mock ImageGenerateJobPopResponse."""
    job = Mock()
    job.id_ = "test-job-id-1234"
    job.ids = ["test-job-id-1234"]
    job.model = "stable_diffusion"
    job.payload = Mock()
    job.payload.width = 512
    job.payload.height = 512
    job.payload.ddim_steps = 30
    job.payload.n_iter = 1
    job.payload.post_processing = []
    job.payload.loras = []
    job.payload.hires_fix = False
    job.payload.control_type = None
    job.payload.seed = 42
    job.payload.tiling = None
    job.payload.tis = []
    job.payload.sampler_name = "k_euler"
    job.payload.workflow = None
    job.payload.use_nsfw_censor = False
    job.payload.prompt = "a test prompt"
    job.source_image = None
    job.source_mask = None
    job.extra_source_images = None
    return job


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
def mock_process_info() -> Mock:
    """Create a mock HordeProcessInfo."""
    return make_mock_process_info()


@pytest.fixture()
def process_manager() -> HordeWorkerProcessManager:
    """Create a testable HordeWorkerProcessManager with all external deps mocked."""
    return make_testable_process_manager()
