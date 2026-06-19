"""Regression probes for the an observed soak wedge: process death during PRELOAD is never recovered.

A fellow dev's RTX 4090 ``horde-benchmark ramp`` soak level completed 0 jobs in 656s then was killed
by the level watchdog. An inference child exited *during model preload* and reported ``PROCESS_ENDED``
(its main loop escaped after a ``PRELOAD_MODEL`` handler raised, taking the same graceful shutdown path
as an intended end). The supervisor's crash reaper bailed on the ``PROCESS_ENDED`` state and never
replaced the slot, while the single popped job stayed pinned at pending-start forever
(``num_process_recoveries: 0``).

These tests cover both halves of the fix:

* ``_reap_if_crashed`` keys recovery off whether the supervisor *intended* the end, not the last
  reported state, so an unexpected ``PROCESS_ENDED`` is recovered while a deliberate end is left alone.
* a dead slot's stale model-map entry is expired by process id (its ``loaded_horde_model_name`` is
  already nulled by the ``PROCESS_ENDING`` message), so the stranded pending job can be re-preloaded
  onto the fresh slot instead of being treated as "already resident" forever.
"""

from __future__ import annotations

import multiprocessing
from unittest.mock import Mock

from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.messages import HordeProcessState, ModelInfo, ModelLoadState
from horde_worker_regen.process_management.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_test_runtime_config,
    track_popped_job_async,
)


def _make_plm(*, process_map: ProcessMap, horde_model_map: HordeModelMap) -> ProcessLifecycleManager:
    bridge_data = Mock()
    bridge_data.image_models_to_load = ["stable_diffusion"]
    bridge_data.max_threads = 1
    bridge_data.safety_on_gpu = False
    bridge_data.high_memory_mode = False
    bridge_data.very_high_memory_mode = False
    bridge_data.process_timeout = 300
    bridge_data.inference_step_timeout = 15
    bridge_data.preload_timeout = 80
    bridge_data.download_timeout = 120
    bridge_data.post_process_timeout = 60
    bridge_data.max_batch = 1
    bridge_data.exit_on_unhandled_faults = False

    return ProcessLifecycleManager(
        ctx=multiprocessing.get_context("spawn"),
        process_map=process_map,
        horde_model_map=horde_model_map,
        job_tracker=JobTracker(),
        process_message_queue=Mock(),
        inference_semaphore=Mock(),
        disk_lock=Mock(),
        aux_model_lock=Mock(),
        vae_decode_semaphore=Mock(),
        gpu_sampling_lease=Mock(),
        download_bandwidth_semaphore=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        max_inference_processes=4,
        max_safety_processes=1,
        amd_gpu=False,
        directml=None,
        abort_callback=Mock(),
        state=WorkerState(),
    )


def _kill_os_process(process_info: object) -> None:
    """Make the slot's mocked OS process look like it has already exited."""
    process_info.mp_process.is_alive.return_value = False  # type: ignore[attr-defined]
    process_info.mp_process.exitcode = 0  # type: ignore[attr-defined]


def test_model_map_expires_entries_by_process_id() -> None:
    """The process-scoped expiry removes only the dead slot's entries, leaving peers intact."""
    model_map = HordeModelMap(
        root={
            "stable_diffusion": ModelInfo(
                horde_model_name="stable_diffusion",
                horde_model_load_state=ModelLoadState.LOADING,
                process_id=1,
            ),
            "other_model": ModelInfo(
                horde_model_name="other_model",
                horde_model_load_state=ModelLoadState.LOADED_IN_VRAM,
                process_id=2,
            ),
        },
    )

    expired = model_map.expire_entries_for_process(1)

    assert expired == ["stable_diffusion"]
    assert "stable_diffusion" not in model_map.root
    assert "other_model" in model_map.root


async def test_unexpected_process_ended_during_preload_is_recovered() -> None:
    """A slot that self-reports PROCESS_ENDED mid-preload (not an intended end) must be recovered.

    Reproduces the soak wedge: the child exited during PRELOAD and reported PROCESS_ENDED, its
    loaded_horde_model_name was already cleared by the PROCESS_ENDING message, and the popped job sat in
    PENDING_INFERENCE with a stale LOADING map entry pinning the model as resident. The reaper must
    rebuild the slot, count the recovery, and clear the stale map entry so the job can re-preload.
    """
    model_map = HordeModelMap(
        root={
            "stable_diffusion": ModelInfo(
                horde_model_name="stable_diffusion",
                horde_model_load_state=ModelLoadState.LOADING,
                process_id=1,
            ),
        },
    )
    dead = make_mock_process_info(1, model_name=None, state=HordeProcessState.PROCESS_ENDED)
    _kill_os_process(dead)
    process_map = ProcessMap({1: dead})

    plm = _make_plm(process_map=process_map, horde_model_map=model_map)
    # Don't touch real OS processes; only the recovery decision and bookkeeping are under test.
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]

    job = make_job_pop_response(model="stable_diffusion")
    await track_popped_job_async(plm._job_tracker, job)
    # The PROCESS_ENDING message nulls last_job_referenced before we ever reap, so the job is only
    # discoverable through the queue (still PENDING_INFERENCE) and the stale model-map entry.
    dead.last_job_referenced = None
    assert job.id_ is not None

    recovered = plm._reap_if_crashed(dead)

    assert recovered is True
    plm._start_inference_process.assert_called_once_with(1)
    assert plm._num_process_recoveries == 1
    # The stale entry is gone, so preload_models no longer treats the model as resident on the dead slot.
    assert "stable_diffusion" not in model_map.root
    # The job was never moved out of the queue, so it re-dispatches normally once a slot frees up.
    assert plm._job_tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE


async def test_hard_crash_during_preload_requeues_referenced_job() -> None:
    """A hard crash (no PROCESS_ENDING message) retains last_job_referenced; the job is requeued."""
    model_map = HordeModelMap(
        root={
            "stable_diffusion": ModelInfo(
                horde_model_name="stable_diffusion",
                horde_model_load_state=ModelLoadState.LOADING,
                process_id=1,
            ),
        },
    )
    dead = make_mock_process_info(1, model_name="stable_diffusion", state=HordeProcessState.PRELOADING_MODEL)
    _kill_os_process(dead)
    process_map = ProcessMap({1: dead})

    plm = _make_plm(process_map=process_map, horde_model_map=model_map)
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]

    job = make_job_pop_response(model="stable_diffusion")
    await track_popped_job_async(plm._job_tracker, job)
    dead.last_job_referenced = job
    assert job.id_ is not None

    recovered = plm._reap_if_crashed(dead)

    assert recovered is True
    assert plm._num_process_recoveries == 1
    assert "stable_diffusion" not in model_map.root
    # The crash faults the slot's referenced job (retryable, but the default single-attempt policy makes
    # it terminal). Either way it leaves pending-start/in-progress and the queue drains; it is not pinned.
    assert plm._job_tracker.get_stage(job.id_) == JobStage.PENDING_SUBMIT


async def test_intended_end_is_not_reaped_or_recovered() -> None:
    """A slot the supervisor deliberately ended must not be treated as a crash to recover."""
    model_map = HordeModelMap(root={})
    ended = make_mock_process_info(1, model_name=None, state=HordeProcessState.PROCESS_ENDED)
    _kill_os_process(ended)
    ended.end_intended = True
    process_map = ProcessMap({1: ended})

    plm = _make_plm(process_map=process_map, horde_model_map=model_map)
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]

    recovered = plm._reap_if_crashed(ended)

    assert recovered is False
    plm._start_inference_process.assert_not_called()
    assert plm._num_process_recoveries == 0


async def test_inference_crash_reported_as_process_ended_is_recovered() -> None:
    """Bug 3: an inference crash surfaces as an unintended PROCESS_ENDED and must be recovered.

    The SDXL-controlnet fault raises inside ``start_inference``; ``@logger.catch(reraise=True)`` re-raises
    it into the child's control-message loop, which sets the end flag and exits via the same graceful
    PROCESS_ENDED path as a preload death. Before the fix the slot was never replaced
    (``num_process_recoveries: 0`` in the report) and the level idled to timeout. The recovery here, plus
    the manager's orphaned-in-progress-job watchdog, is what lets the level continue after the fault.
    """
    model_map = HordeModelMap(
        root={
            "stable_diffusion_xl": ModelInfo(
                horde_model_name="stable_diffusion_xl",
                horde_model_load_state=ModelLoadState.IN_USE,
                process_id=1,
            ),
        },
    )
    dead = make_mock_process_info(1, model_name=None, state=HordeProcessState.PROCESS_ENDED)
    _kill_os_process(dead)
    process_map = ProcessMap({1: dead})

    plm = _make_plm(process_map=process_map, horde_model_map=model_map)
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]

    job = make_job_pop_response(model="stable_diffusion_xl")
    await track_popped_job_async(plm._job_tracker, job)
    await plm._job_tracker.mark_inference_started(job)
    # The PROCESS_ENDING message nulled last_job_referenced; the in-progress job is left for the manager's
    # orphan watchdog to punt, which only runs once a live slot no longer owns it.
    dead.last_job_referenced = None
    assert job.id_ is not None

    recovered = plm._reap_if_crashed(dead)

    assert recovered is True
    plm._start_inference_process.assert_called_once_with(1)
    assert plm._num_process_recoveries == 1
    assert "stable_diffusion_xl" not in model_map.root


def test_ending_process_sets_intent_flag() -> None:
    """_end_inference_process records supervisor intent so the reaper leaves the exit alone."""
    model_map = HordeModelMap(root={})
    slot = make_mock_process_info(1, model_name="stable_diffusion", state=HordeProcessState.WAITING_FOR_JOB)
    process_map = ProcessMap({1: slot})

    plm = _make_plm(process_map=process_map, horde_model_map=model_map)
    assert slot.end_intended is False

    plm._end_inference_process(slot)

    assert slot.end_intended is True
