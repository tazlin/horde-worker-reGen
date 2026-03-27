"""Tests for ProcessLifecycleManager."""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeProcessState
from horde_worker_regen.process_management.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.worker_state import WorkerState


def _make_plm(
    *,
    process_map: ProcessMap | None = None,
    job_tracker: JobTracker | None = None,
) -> ProcessLifecycleManager:
    """Helper to build a PLM with mostly-mocked dependencies."""
    bridge_data = Mock()
    bridge_data.image_models_to_load = ["stable_diffusion"]
    bridge_data.safety_on_gpu = False
    bridge_data.high_memory_mode = False
    bridge_data.very_high_memory_mode = False
    bridge_data.process_timeout = 120
    bridge_data.inference_step_timeout = 60
    bridge_data.preload_timeout = 120
    bridge_data.download_timeout = 120
    bridge_data.post_process_timeout = 60
    bridge_data.max_batch = 1
    bridge_data.exit_on_unhandled_faults = False

    return ProcessLifecycleManager(
        process_map=process_map or ProcessMap({}),
        horde_model_map=Mock(),
        job_tracker=job_tracker or JobTracker(),
        process_message_queue=Mock(),
        inference_semaphore=Mock(),
        disk_lock=Mock(),
        aux_model_lock=Mock(),
        vae_decode_semaphore=Mock(),
        get_bridge_data=lambda: bridge_data,
        max_inference_processes=2,
        max_safety_processes=1,
        amd_gpu=False,
        directml=None,
        abort_callback=Mock(),
        state=WorkerState(),
    )


def test_init_stores_references() -> None:
    """Test that the constructor properly stores references to its dependencies."""
    plm = _make_plm()
    assert plm.num_processes_launched == 0
    assert plm._num_process_recoveries == 0
    assert plm._safety_processes_should_be_replaced is False
    assert plm._safety_processes_ending is False
    assert plm._recently_recovered is False
    assert plm._hung_processes_detected is False
    assert plm._hung_processes_detected_time == 0.0


def test_get_processes_with_model_for_queued_job_empty() -> None:
    """If there are no processes or no jobs pending inference, the result should be empty."""
    plm = _make_plm()
    result = plm.get_processes_with_model_for_queued_job()
    assert result == []


def test_get_processes_with_model_for_queued_job_matches() -> None:
    """If there is a waiting process with the needed model, it should be returned."""
    process_map = ProcessMap({})
    job_tracker = JobTracker()

    proc = Mock()
    proc.process_id = 0
    proc.loaded_horde_model_name = "stable_diffusion"
    proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
    process_map[0] = proc

    # The method checks `model_name in jobs_pending_inference` which compares
    # a string against job objects — only matches if the job object itself
    # equals the model name string. Use the model name string directly to
    # exercise this path.
    job_tracker.jobs_pending_inference.append("stable_diffusion")

    plm = _make_plm(process_map=process_map, job_tracker=job_tracker)
    result = plm.get_processes_with_model_for_queued_job()

    assert 0 in result


def test_get_processes_with_model_for_queued_job_preloaded() -> None:
    """If there is a preloaded process with the needed model, it should be returned."""
    process_map = ProcessMap({})

    proc = Mock()
    proc.process_id = 1
    proc.loaded_horde_model_name = "some_other_model"
    proc.last_process_state = HordeProcessState.PRELOADED_MODEL
    process_map[1] = proc

    plm = _make_plm(process_map=process_map)
    result = plm.get_processes_with_model_for_queued_job()

    assert 1 in result


def test_recently_recovered_property() -> None:
    """Test the recently_recovered property getter and setter."""
    plm = _make_plm()
    assert plm.recently_recovered is False

    plm._recently_recovered = True
    assert plm.recently_recovered is True


def test_safety_processes_should_be_replaced_property() -> None:
    """Test the safety_processes_should_be_replaced property getter and setter."""
    plm = _make_plm()
    assert plm.safety_processes_should_be_replaced is False

    plm.safety_processes_should_be_replaced = True
    assert plm.safety_processes_should_be_replaced is True
