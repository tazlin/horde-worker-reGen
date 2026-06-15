"""Tests for ProcessLifecycleManager."""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeProcessState
from horde_worker_regen.process_management.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import make_mock_process_info, make_test_runtime_config, track_popped_job_async


def _make_plm(
    *,
    process_map: ProcessMap | None = None,
    job_tracker: JobTracker | None = None,
) -> ProcessLifecycleManager:
    """Helper to build a PLM with mostly-mocked dependencies."""
    bridge_data = Mock()
    bridge_data.image_models_to_load = ["stable_diffusion"]
    bridge_data.max_threads = 1
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
        gpu_sampling_lease=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
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


async def test_get_processes_with_model_for_queued_job_matches() -> None:
    """If there is a waiting process with the needed model, it should be returned."""
    process_map = ProcessMap({})
    job_tracker = JobTracker()

    proc = Mock()
    proc.process_id = 0
    proc.loaded_horde_model_name = "stable_diffusion"
    proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
    process_map[0] = proc

    queued_job = Mock()
    queued_job.id_ = "queued-job"
    queued_job.model = "stable_diffusion"
    await track_popped_job_async(job_tracker, queued_job)

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


def _patch_spawn_with_stub(plm: ProcessLifecycleManager) -> None:
    """Replace real process spawning with a stub that adds an idle mock process to the map."""

    def _fake_start(pid: int) -> HordeProcessInfo:
        info = make_mock_process_info(pid, model_name=None, process_type=HordeProcessType.INFERENCE)
        plm._process_map[pid] = info
        plm.num_processes_launched += 1
        return info

    plm._start_inference_process = _fake_start  # type: ignore[method-assign]


def test_allocate_inference_pid_picks_lowest_free() -> None:
    """The pid allocator returns the lowest unused slot id, reusing freed ones."""
    process_map = ProcessMap(
        {
            0: make_mock_process_info(0, process_type=HordeProcessType.SAFETY),
            1: make_mock_process_info(1, model_name=None),
        },
    )
    plm = _make_plm(process_map=process_map)
    assert plm._allocate_inference_pid() == 2

    process_map.pop(1)
    assert plm._allocate_inference_pid() == 1


def test_scale_up_starts_processes_up_to_ceiling() -> None:
    """Scaling up spawns processes, bounded by max_inference_processes."""
    plm = _make_plm()  # max_inference_processes=2
    _patch_spawn_with_stub(plm)

    assert plm.scale_inference_processes(2) == 2
    assert plm._process_map.num_inference_processes() == 2
    assert sorted(plm._process_map.keys()) == [0, 1]

    # Requests beyond the launched ceiling are capped.
    assert plm.scale_inference_processes(5) == 2


def test_scale_down_stops_idle_processes() -> None:
    """Scaling down ends idle inference processes and removes them from the map."""
    plm = _make_plm()
    _patch_spawn_with_stub(plm)
    plm.scale_inference_processes(2)

    assert plm.scale_inference_processes(1) == 1
    assert plm._process_map.num_inference_processes() == 1


def test_scale_down_never_kills_busy_processes() -> None:
    """A busy (mid-inference) process is retained even when scaling toward zero."""
    busy = make_mock_process_info(0, model_name="m", state=HordeProcessState.INFERENCE_STARTING)
    idle = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    plm = _make_plm(process_map=ProcessMap({0: busy, 1: idle}))

    plm.scale_inference_processes(0)

    remaining = list(plm._process_map.keys())
    assert remaining == [0]  # the busy process survives


def test_pid_reused_after_scale_down_then_up() -> None:
    """A slot freed by scaling down is reused on the next scale up (no collision)."""
    plm = _make_plm()
    _patch_spawn_with_stub(plm)
    plm.scale_inference_processes(2)  # pids 0, 1
    plm.scale_inference_processes(1)  # removes the first idle slot (pid 0)
    plm.scale_inference_processes(2)  # should re-allocate pid 0

    assert sorted(plm._process_map.keys()) == [0, 1]
