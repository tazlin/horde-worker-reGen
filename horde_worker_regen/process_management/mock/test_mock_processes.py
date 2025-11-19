"""End-to-end tests for mock processes.

These tests verify that mock processes behave correctly and send proper message
sequences that match real process behavior.
"""

from __future__ import annotations

import multiprocessing
import time
from multiprocessing import Queue
from typing import TYPE_CHECKING

import pytest

from horde_worker_regen.process_management.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeHeartbeatType,
    HordeInferenceControlMessage,
    HordeModelStateChangeMessage,
    HordePreloadInferenceModelMessage,
    HordeProcessHeartbeatMessage,
    HordeProcessMemoryMessage,
    HordeProcessStateChangeMessage,
    HordeProcessState,
    HordeSafetyControlMessage,
    ModelLoadState,
)
from horde_worker_regen.process_management.mock import (
    MockConfig,
    MockScenario,
    start_mock_inference_process,
    start_mock_safety_process,
)

if TYPE_CHECKING:
    from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse


def create_fake_job(
    job_id: str = "test-job-123",
    model: str = "SDXL",
    width: int = 512,
    height: int = 512,
    steps: int = 20,
) -> ImageGenerateJobPopResponse:
    """Create a fake job for testing.

    Args:
        job_id: Job ID.
        model: Model name.
        width: Image width.
        height: Image height.
        steps: Number of steps.

    Returns:
        Fake job object.
    """
    from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, ImageGenerateJobPopPayload
    from horde_sdk.ai_horde_api.fields import JobID

    return ImageGenerateJobPopResponse(
        id_=JobID(root=job_id),
        model=model,
        payload=ImageGenerateJobPopPayload(
            width=width,
            height=height,
            ddim_steps=steps,
            n=1,
        ),
    )


def test_mock_inference_process_lifecycle():
    """Test that mock inference process starts, runs, and exits correctly."""
    ctx = multiprocessing.get_context("spawn")
    message_queue = ctx.Queue()
    parent_conn, child_conn = ctx.Pipe()

    inference_semaphore = ctx.Semaphore(1)
    vae_semaphore = ctx.Semaphore(1)
    aux_lock = ctx.Lock()
    disk_lock = ctx.Lock()

    config = MockConfig(speed_multiplier=100.0)  # 100x faster for testing

    # Start process
    process = ctx.Process(
        target=start_mock_inference_process,
        args=(
            1,  # process_id
            message_queue,
            child_conn,
            inference_semaphore,
            disk_lock,
            aux_lock,
            vae_semaphore,
            1,  # launch_identifier
            config,
        ),
    )

    process.start()

    # Wait for startup
    time.sleep(0.1)

    # Should receive PROCESS_STARTING and WAITING_FOR_JOB messages
    messages = []
    while not message_queue.empty():
        messages.append(message_queue.get(timeout=1))

    assert len(messages) >= 2
    assert isinstance(messages[0], HordeProcessStateChangeMessage)
    assert messages[0].process_state == HordeProcessState.PROCESS_STARTING

    # Find WAITING_FOR_JOB message
    waiting_msg = None
    for msg in messages:
        if isinstance(msg, HordeProcessStateChangeMessage):
            if msg.process_state == HordeProcessState.WAITING_FOR_JOB:
                waiting_msg = msg
                break

    assert waiting_msg is not None, "Should receive WAITING_FOR_JOB message"

    # Send END_PROCESS message
    parent_conn.send(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))

    # Wait for process to exit
    process.join(timeout=5)

    assert not process.is_alive(), "Process should have exited"


def test_mock_inference_model_download():
    """Test that mock process correctly simulates model download with progress."""
    ctx = multiprocessing.get_context("spawn")
    message_queue = ctx.Queue()
    parent_conn, child_conn = ctx.Pipe()

    config = MockConfig(speed_multiplier=50.0)  # 50x faster

    process = ctx.Process(
        target=start_mock_inference_process,
        args=(
            1,
            message_queue,
            child_conn,
            ctx.Semaphore(1),
            ctx.Lock(),
            ctx.Lock(),
            ctx.Semaphore(1),
            1,
            config,
        ),
    )

    process.start()
    time.sleep(0.1)

    # Clear startup messages
    while not message_queue.empty():
        message_queue.get()

    # Send download model message
    download_msg = HordeControlMessage(
        control_flag=HordeControlFlag.DOWNLOAD_MODEL,
        horde_model_name="stable_diffusion_xl",
    )
    parent_conn.send(download_msg)

    # Wait for download
    time.sleep(1.0)

    # Collect messages
    messages = []
    while not message_queue.empty():
        messages.append(message_queue.get())

    # Should have:
    # - DOWNLOADING_MODEL state change
    # - Multiple ModelStateChange with DOWNLOADING (progress updates)
    # - ModelStateChange with ON_DISK (complete)
    # - WAITING_FOR_JOB state change

    state_changes = [m for m in messages if isinstance(m, HordeProcessStateChangeMessage)]
    model_changes = [m for m in messages if isinstance(m, HordeModelStateChangeMessage)]

    # Find DOWNLOADING_MODEL state
    downloading_state = None
    for msg in state_changes:
        if msg.process_state == HordeProcessState.DOWNLOADING_MODEL:
            downloading_state = msg
            break

    assert downloading_state is not None, "Should enter DOWNLOADING_MODEL state"

    # Should have progress updates
    download_progress = [m for m in model_changes if m.horde_model_state == ModelLoadState.DOWNLOADING]
    assert len(download_progress) > 0, "Should send download progress updates"

    # Should complete with ON_DISK
    on_disk = [m for m in model_changes if m.horde_model_state == ModelLoadState.ON_DISK]
    assert len(on_disk) > 0, "Should complete download with ON_DISK state"

    # Cleanup
    parent_conn.send(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))
    process.join(timeout=5)


def test_mock_inference_model_preload():
    """Test that mock process correctly simulates model preloading."""
    ctx = multiprocessing.get_context("spawn")
    message_queue = ctx.Queue()
    parent_conn, child_conn = ctx.Pipe()

    config = MockConfig(speed_multiplier=50.0)

    process = ctx.Process(
        target=start_mock_inference_process,
        args=(1, message_queue, child_conn, ctx.Semaphore(1), ctx.Lock(), ctx.Lock(), ctx.Semaphore(1), 1, config),
    )

    process.start()
    time.sleep(0.1)

    # Clear startup messages
    while not message_queue.empty():
        message_queue.get()

    # Send preload message
    preload_msg = HordePreloadInferenceModelMessage(
        control_flag=HordeControlFlag.PRELOAD_MODEL,
        horde_model_name="stable_diffusion_xl",
    )
    parent_conn.send(preload_msg)

    # Wait for preload
    time.sleep(0.5)

    # Collect messages
    messages = []
    while not message_queue.empty():
        messages.append(message_queue.get())

    state_changes = [m for m in messages if isinstance(m, HordeProcessStateChangeMessage)]
    model_changes = [m for m in messages if isinstance(m, HordeModelStateChangeMessage)]

    # Should go through: PRELOADING_MODEL -> PRELOADED_MODEL
    preloading = None
    preloaded = None
    for msg in state_changes:
        if msg.process_state == HordeProcessState.PRELOADING_MODEL:
            preloading = msg
        elif msg.process_state == HordeProcessState.PRELOADED_MODEL:
            preloaded = msg

    assert preloading is not None, "Should enter PRELOADING_MODEL state"
    assert preloaded is not None, "Should enter PRELOADED_MODEL state"

    # Should load to RAM then VRAM
    loaded_ram = [m for m in model_changes if m.horde_model_state == ModelLoadState.LOADED_IN_RAM]
    loaded_vram = [m for m in model_changes if m.horde_model_state == ModelLoadState.LOADED_IN_VRAM]

    assert len(loaded_ram) > 0, "Should load to RAM"
    assert len(loaded_vram) > 0, "Should load to VRAM"

    # Cleanup
    parent_conn.send(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))
    process.join(timeout=5)


def test_mock_inference_job_execution():
    """Test that mock process correctly simulates full inference job."""
    ctx = multiprocessing.get_context("spawn")
    message_queue = ctx.Queue()
    parent_conn, child_conn = ctx.Pipe()

    config = MockConfig(speed_multiplier=100.0)  # Very fast for testing

    process = ctx.Process(
        target=start_mock_inference_process,
        args=(1, message_queue, child_conn, ctx.Semaphore(1), ctx.Lock(), ctx.Lock(), ctx.Semaphore(1), 1, config),
    )

    process.start()
    time.sleep(0.1)

    # Clear startup messages
    while not message_queue.empty():
        message_queue.get()

    # Create and send inference job
    fake_job = create_fake_job(steps=10)  # Only 10 steps for faster test
    inference_msg = HordeInferenceControlMessage(
        control_flag=HordeControlFlag.START_INFERENCE,
        job=fake_job,
    )
    parent_conn.send(inference_msg)

    # Wait for inference to complete
    time.sleep(1.0)

    # Collect messages
    messages = []
    while not message_queue.empty():
        messages.append(message_queue.get())

    state_changes = [m for m in messages if isinstance(m, HordeProcessStateChangeMessage)]
    heartbeats = [m for m in messages if isinstance(m, HordeProcessHeartbeatMessage)]

    # Should have state sequence: INFERENCE_STARTING -> INFERENCE_POST_PROCESSING -> INFERENCE_COMPLETE
    states_found = {msg.process_state for msg in state_changes}

    assert HordeProcessState.INFERENCE_STARTING in states_found, "Should start inference"
    assert HordeProcessState.INFERENCE_POST_PROCESSING in states_found, "Should post-process"
    assert HordeProcessState.INFERENCE_COMPLETE in states_found, "Should complete inference"

    # Should send heartbeats during inference
    inference_heartbeats = [h for h in heartbeats if h.heartbeat_type == HordeHeartbeatType.INFERENCE_STEP]
    assert len(inference_heartbeats) > 0, "Should send inference step heartbeats"

    # At least one heartbeat should have progress
    with_progress = [h for h in inference_heartbeats if h.percent_complete is not None]
    assert len(with_progress) > 0, "Should report progress percentage"

    # Should eventually return to WAITING_FOR_JOB
    final_states = [msg.process_state for msg in state_changes[-3:]]
    assert HordeProcessState.WAITING_FOR_JOB in final_states, "Should return to waiting state"

    # Cleanup
    parent_conn.send(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))
    process.join(timeout=5)


def test_mock_inference_memory_reporting():
    """Test that mock process sends memory reports."""
    ctx = multiprocessing.get_context("spawn")
    message_queue = ctx.Queue()
    parent_conn, child_conn = ctx.Pipe()

    config = MockConfig(
        speed_multiplier=10.0,
        mock_ram_usage_mb=4096,
        mock_vram_usage_mb=8192,
    )

    process = ctx.Process(
        target=start_mock_inference_process,
        args=(1, message_queue, child_conn, ctx.Semaphore(1), ctx.Lock(), ctx.Lock(), ctx.Semaphore(1), 1, config),
    )

    process.start()

    # Wait for a few memory reports (sent every 5 seconds by default)
    time.sleep(1.0)  # Should get at least one quickly

    # Collect messages
    messages = []
    while not message_queue.empty():
        messages.append(message_queue.get())

    memory_reports = [m for m in messages if isinstance(m, HordeProcessMemoryMessage)]

    assert len(memory_reports) > 0, "Should send memory reports"

    # Check memory values are reasonable
    for report in memory_reports:
        assert report.ram_usage_bytes > 0, "RAM usage should be positive"
        # VRAM might be 0 if not explicitly requested

    # Cleanup
    parent_conn.send(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))
    process.join(timeout=5)


def test_mock_safety_process_evaluation():
    """Test that mock safety process correctly evaluates images."""
    ctx = multiprocessing.get_context("spawn")
    message_queue = ctx.Queue()
    parent_conn, child_conn = ctx.Pipe()

    config = MockConfig(speed_multiplier=50.0, safety_check_time_seconds=0.1)

    process = ctx.Process(
        target=start_mock_safety_process,
        args=(1, message_queue, child_conn, ctx.Lock(), 1, config, True),
    )

    process.start()
    time.sleep(0.1)

    # Clear startup messages
    while not message_queue.empty():
        message_queue.get()

    # Create fake job with images (we need the structure but not real images)
    from horde_worker_regen.process_management.messages import HordeImageResult, CompletedJobLookupInfo

    fake_job_info = create_fake_job()
    fake_images = [
        HordeImageResult(image_base64="fake_image_1", seed=1, generation_faults=[]),
        HordeImageResult(image_base64="fake_image_2", seed=2, generation_faults=[]),
    ]

    lookup_info = CompletedJobLookupInfo(
        sdk_api_job_info=fake_job_info,
        job_image_results=fake_images,
    )

    # Send safety evaluation request
    safety_msg = HordeSafetyControlMessage(
        control_flag=HordeControlFlag.EVALUATE_SAFETY,
        job=lookup_info,
    )
    parent_conn.send(safety_msg)

    # Wait for evaluation
    time.sleep(0.5)

    # Collect messages
    messages = []
    while not message_queue.empty():
        messages.append(message_queue.get())

    state_changes = [m for m in messages if isinstance(m, HordeProcessStateChangeMessage)]

    # Should enter EVALUATING_SAFETY state
    evaluating = None
    for msg in state_changes:
        if msg.process_state == HordeProcessState.EVALUATING_SAFETY:
            evaluating = msg
            break

    assert evaluating is not None, "Should enter EVALUATING_SAFETY state"

    # Should send result (check for HordeSafetyResultMessage)
    from horde_worker_regen.process_management.messages import HordeSafetyResultMessage

    results = [m for m in messages if isinstance(m, HordeSafetyResultMessage)]
    assert len(results) > 0, "Should send safety evaluation result"

    # Check result has evaluations for each image
    result = results[0]
    assert len(result.safety_evaluations) == 2, "Should have evaluation for each image"

    # Check evaluations have scores
    for eval in result.safety_evaluations:
        assert 0.0 <= eval.nsfw_score <= 1.0, "NSFW score should be in valid range"
        assert 0.0 <= eval.csam_score <= 1.0, "CSAM score should be in valid range"

    # Cleanup
    parent_conn.send(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))
    process.join(timeout=5)


def test_mock_scenario_application():
    """Test that scenarios correctly modify config."""
    config = MockConfig()

    # Test RAPID_FIRE scenario
    config.apply_scenario(MockScenario.RAPID_FIRE)
    assert config.speed_multiplier == 100.0
    assert not config.enable_failures

    # Test RANDOM_FAILURES scenario
    config.apply_scenario(MockScenario.RANDOM_FAILURES)
    assert config.enable_failures
    assert config.failure_rate > 0.0

    # Test SLOW_INFERENCE scenario
    config.apply_scenario(MockScenario.SLOW_INFERENCE)
    assert config.enable_slowdowns
    assert config.slowdown_multiplier > 1.0


if __name__ == "__main__":
    # Run tests manually
    print("Running mock process tests...")

    print("\n1. Testing process lifecycle...")
    test_mock_inference_process_lifecycle()
    print("✓ Process lifecycle test passed")

    print("\n2. Testing model download...")
    test_mock_inference_model_download()
    print("✓ Model download test passed")

    print("\n3. Testing model preload...")
    test_mock_inference_model_preload()
    print("✓ Model preload test passed")

    print("\n4. Testing job execution...")
    test_mock_inference_job_execution()
    print("✓ Job execution test passed")

    print("\n5. Testing memory reporting...")
    test_mock_inference_memory_reporting()
    print("✓ Memory reporting test passed")

    print("\n6. Testing safety evaluation...")
    test_mock_safety_process_evaluation()
    print("✓ Safety evaluation test passed")

    print("\n7. Testing scenario application...")
    test_mock_scenario_application()
    print("✓ Scenario application test passed")

    print("\n✅ All tests passed!")
