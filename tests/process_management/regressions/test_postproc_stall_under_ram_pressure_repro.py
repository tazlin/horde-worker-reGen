"""Post-processing silence watchdog versus a host driven below its RAM danger floor.

A single-GPU worker under sustained system-RAM exhaustion (the host paged down to a few MB of free RAM,
well under the danger floor, while the card kept ~15 GB of free VRAM) reaped six inference slots and
faulted one job. Every reap carried the identical fingerprint:

* the slot was in ``INFERENCE_POST_PROCESSING`` (its expensive sampling had already completed);
* the OS process was still alive (``exitcode is None``) but had emitted no message or heartbeat for
  ``post_process_timeout + 3 * max_batch`` seconds, so the silence watchdog declared it "stuck post
  processing" and force-replaced it;
* the replacement is attributed to a post-processing *VRAM over-commit* and feeds that feature-level
  breaker, even though free VRAM was ample and the actual cause was host-RAM swap thrash.

One job was requeued after its first slot was reaped, then stalled the same way on its replacement slot,
exhausting the two-attempt budget into a terminal fault.

The behaviour under test is the interaction between that watchdog and the RAM-pressure governor. The
governor already publishes that the host is below its danger floor (``WorkerState.ram_pressure_pop_hold``)
and is actively pausing pops and shedding footprint to recover; a *live* slot that has merely gone silent
finishing an already-inferred job during that window is starved, not hung, and reaping it discards a
finished image, burns a retry, churns the pool, and misreports the cause as a VRAM over-commit.

The tests split into three groups:

* characterization of the silence arithmetic that produced the 84 s window (no RAM pressure);
* the proposed contract for a live post-processing slot while the host is under RAM danger-floor
  pressure (reproduces the wasteful churn and the misattribution: currently red);
* boundaries the RAM-pressure grace must not erode (a genuine crash, a non-post-processing stall, and
  an idle worker), which must stay green.
"""

from __future__ import annotations

import multiprocessing
import time
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_test_card_runtimes,
    make_test_runtime_config,
    track_popped_job_async,
)

# The live worker ran the stock post-processing timeout with a batch of 8, so the silence threshold was
# 60 + 3*8 = 84 s, matching the ``since_last_heartbeat=84.x s`` on every reap in the session.
LIVE_POST_PROCESS_TIMEOUT = 60
LIVE_MAX_BATCH = 8
LIVE_SILENCE_THRESHOLD = LIVE_POST_PROCESS_TIMEOUT + 3 * LIVE_MAX_BATCH


def _make_plm(
    *,
    process_map: ProcessMap,
    post_process_timeout: int = LIVE_POST_PROCESS_TIMEOUT,
    max_batch: int = LIVE_MAX_BATCH,
    state: WorkerState | None = None,
) -> ProcessLifecycleManager:
    """Build a PLM with a numeric bridge config and the two-attempt retry policy the worker runs with."""
    bridge_data = Mock()
    bridge_data.image_models_to_load = ["stable_diffusion"]
    bridge_data.max_threads = 2
    bridge_data.safety_on_gpu = False
    bridge_data.process_timeout = 300
    # Kept small so the non-daemon debounce thread a crash-reap starts (it sleeps this long before clearing
    # ``_recently_recovered``) does not linger past the test; the post-processing slots under test are never
    # in INFERENCE_STARTING, so this per-step value gates nothing they assert on.
    bridge_data.inference_step_timeout = 1
    bridge_data.inference_first_step_timeout = 60
    bridge_data.inference_stuck_step_repeat_limit = 20
    bridge_data.contended_step_timeout = 120
    bridge_data.overbudget_step_timeout = 180
    bridge_data.preload_timeout = 80
    bridge_data.download_timeout = 120
    bridge_data.post_process_timeout = post_process_timeout
    bridge_data.max_batch = max_batch
    bridge_data.exit_on_unhandled_faults = False
    bridge_data.ram_pressure_pause_percent = 85.0
    bridge_data.ram_pressure_min_free_mb = 1024.0

    job_tracker = JobTracker()
    # The worker applies max_inference_attempts (default 2) at startup; a bare tracker would fault on the
    # first stall and never reproduce the observed requeue-then-exhaust sequence.
    job_tracker.set_retry_policy(2)

    return ProcessLifecycleManager(
        ctx=multiprocessing.get_context("spawn"),
        process_map=process_map,
        horde_model_map=Mock(),
        job_tracker=job_tracker,
        process_message_queue=Mock(),
        card_runtimes=make_test_card_runtimes(target_process_count=4),
        disk_lock=Mock(),
        aux_model_lock=Mock(),
        download_bandwidth_semaphore=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        max_safety_processes=1,
        amd_gpu=False,
        directml=None,
        abort_callback=Mock(),
        state=state if state is not None else WorkerState(),
    )


def _silent_post_processing_slot(
    process_id: int,
    *,
    silent_for: float,
    model_name: str = "CyberRealistic Pony",
) -> HordeProcessInfo:
    """A live inference slot in post-processing that has emitted nothing for ``silent_for`` seconds.

    The OS process is alive with no exit code (a starved, not crashed, child): both the message and
    heartbeat clocks the silence watchdog reads are aged, and no OS process is touched.
    """
    proc = make_mock_process_info(
        process_id,
        model_name=model_name,
        state=HordeProcessState.INFERENCE_POST_PROCESSING,
    )
    proc.mp_process.is_alive.return_value = True
    proc.mp_process.exitcode = None  # type: ignore[read-only]
    now = time.time()
    proc.last_received_timestamp = now - silent_for
    proc.last_heartbeat_timestamp = now - silent_for
    proc.last_process_state_started_at = now - silent_for
    return proc


def _quiet_os_lifecycle(plm: ProcessLifecycleManager) -> None:
    """Stub the OS-touching halves of a replacement so the recovery decision runs without spawning."""
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]


async def _attach_in_flight_job(
    plm: ProcessLifecycleManager,
    slot: HordeProcessInfo,
    *,
    model: str = "CyberRealistic Pony",
) -> ImageGenerateJobPopResponse:
    """Pop, dispatch, and pin a job to ``slot`` as its in-flight (referenced) job."""
    job = make_job_pop_response(model=model)
    await track_popped_job_async(plm._job_tracker, job)
    await plm._job_tracker.mark_inference_started(job)
    slot.last_job_referenced = job
    return job


# --------------------------------------------------------------------------------------------------
# Characterization: the silence arithmetic that produced the 84 s window, with no RAM pressure.
# A genuinely-silent post-processing slot on a healthy host is a real hang and SHOULD be reaped.
# --------------------------------------------------------------------------------------------------


async def test_silent_post_processing_slot_is_reaped_when_ram_is_healthy() -> None:
    """On a healthy host, a slot silent past the post-processing window is a hang and is replaced.

    This anchors the harness: the same wiring the pressure tests use does produce the reap, the recovery
    count, the retryable requeue, and the breaker feed, so a later "not reaped" assertion fails because of
    the pressure guard, not a mis-built fixture.
    """
    slot = _silent_post_processing_slot(3, silent_for=LIVE_SILENCE_THRESHOLD + 1)
    plm = _make_plm(process_map=ProcessMap({3: slot}))
    _quiet_os_lifecycle(plm)
    job = await _attach_in_flight_job(plm, slot)
    assert job.id_ is not None

    replaced = plm.replace_hung_processes()

    assert replaced is True
    assert plm._num_process_recoveries == 1
    # First of two attempts: the slot crash requeues the job rather than faulting it out.
    assert plm._job_tracker.get_stage(job.id_) is JobStage.PENDING_INFERENCE
    assert plm._job_tracker.count_recent_post_processing_faults(3600.0) == 1


async def test_silent_post_processing_slot_below_threshold_is_left_alone() -> None:
    """Silence just under ``post_process_timeout + 3*max_batch`` is within the window; no reap."""
    slot = _silent_post_processing_slot(3, silent_for=LIVE_SILENCE_THRESHOLD - 5)
    plm = _make_plm(process_map=ProcessMap({3: slot}))
    _quiet_os_lifecycle(plm)
    await _attach_in_flight_job(plm, slot)

    replaced = plm.replace_hung_processes()

    assert replaced is False
    assert plm._num_process_recoveries == 0


@pytest.mark.parametrize(
    ("post_process_timeout", "max_batch"),
    [
        (60, 1),  # 63 s
        (60, 8),  # 84 s (the live session)
        (30, 4),  # 42 s
        (90, 16),  # 138 s
    ],
)
async def test_post_processing_reap_threshold_scales_with_batch(
    post_process_timeout: int,
    max_batch: int,
) -> None:
    """The reap boundary tracks ``post_process_timeout + 3*max_batch`` for every batch/timeout pairing.

    Silence one second under the derived threshold must not reap; one second over must. Confirms the
    arithmetic that made the live threshold 84 s rather than the bare 60 s configured value.
    """
    threshold = post_process_timeout + 3 * max_batch

    under = _silent_post_processing_slot(3, silent_for=threshold - 1)
    plm_under = _make_plm(
        process_map=ProcessMap({3: under}),
        post_process_timeout=post_process_timeout,
        max_batch=max_batch,
    )
    _quiet_os_lifecycle(plm_under)
    await _attach_in_flight_job(plm_under, under)
    assert plm_under.replace_hung_processes() is False, f"reaped early at {threshold - 1}s (<{threshold}s)"

    over = _silent_post_processing_slot(3, silent_for=threshold + 1)
    plm_over = _make_plm(
        process_map=ProcessMap({3: over}),
        post_process_timeout=post_process_timeout,
        max_batch=max_batch,
    )
    _quiet_os_lifecycle(plm_over)
    await _attach_in_flight_job(plm_over, over)
    assert plm_over.replace_hung_processes() is True, f"failed to reap at {threshold + 1}s (>{threshold}s)"


# --------------------------------------------------------------------------------------------------
# Proposed contract: a live post-processing slot while the host is under the RAM danger floor.
# These reproduce the observed churn and misattribution and are red until the watchdog consults the
# RAM-pressure state the governor already publishes.
# --------------------------------------------------------------------------------------------------


async def test_live_post_processing_slot_is_spared_under_ram_danger_floor() -> None:
    """A starved-but-alive post-processing slot must not be reaped while the RAM governor holds pops.

    Faithful single-slot reproduction: the governor has flagged the host below its danger floor
    (``ram_pressure_pop_hold``), the card has ample free VRAM, and a slot finishing an already-inferred
    job has gone silent past the 84 s window because the host is thrashing swap. The governor owns this
    window; the finished job should keep its slot until pressure clears, not be discarded as a hang.
    """
    state = WorkerState()
    state.ram_pressure_pop_hold = True
    slot = _silent_post_processing_slot(3, silent_for=LIVE_SILENCE_THRESHOLD + 1)
    plm = _make_plm(process_map=ProcessMap({3: slot}), state=state)
    _quiet_os_lifecycle(plm)
    job = await _attach_in_flight_job(plm, slot)
    assert job.id_ is not None

    replaced = plm.replace_hung_processes()

    assert replaced is False, "a live, RAM-starved post-processing slot was reaped during a pop hold"
    assert plm._num_process_recoveries == 0
    # The already-inferred job keeps its slot; it is neither requeued nor faulted.
    assert plm._job_tracker.get_stage(job.id_) is JobStage.INFERENCE_IN_PROGRESS


async def test_ram_starved_reap_is_not_charged_as_postproc_vram_overcommit() -> None:
    """Even if a slot is reaped under RAM pressure, it must not feed the post-processing VRAM breaker.

    The feature-level breaker exists to disable post-processing when its *VRAM* peak cannot be hosted. A
    reap caused by host-RAM swap thrash (ample free VRAM) is not that condition; charging it there can
    disable post-processing for a cause unloading models cannot fix. Isolated from the grace decision:
    whatever the watchdog does about the slot, the over-commit tally must stay clean.
    """
    state = WorkerState()
    state.ram_pressure_pop_hold = True
    slot = _silent_post_processing_slot(3, silent_for=LIVE_SILENCE_THRESHOLD + 1)
    plm = _make_plm(process_map=ProcessMap({3: slot}), state=state)
    _quiet_os_lifecycle(plm)
    await _attach_in_flight_job(plm, slot)

    plm.replace_hung_processes()

    assert plm._job_tracker.count_recent_post_processing_faults(3600.0) == 0


async def test_same_job_stalling_on_two_slots_under_pressure_does_not_fault_out() -> None:
    """The observed terminal fault: one job reaped off two successive slots exhausts its two attempts.

    Job 690bc0bd stalled post-processing on slot 3, was requeued, then stalled the same way on slot 4,
    and the second reap spent its last attempt into a terminal fault. Under the pressure grace neither
    reap should happen, so the job should still be in flight, not counted as a completed (faulted) job.
    """
    state = WorkerState()
    state.ram_pressure_pop_hold = True

    first = _silent_post_processing_slot(3, silent_for=LIVE_SILENCE_THRESHOLD + 1)
    plm = _make_plm(process_map=ProcessMap({3: first}), state=state)
    _quiet_os_lifecycle(plm)
    job = await _attach_in_flight_job(plm, first)
    assert job.id_ is not None

    plm.replace_hung_processes()

    # Model the requeue-onto-a-fresh-slot the current watchdog performs: the same job lands on slot 4 and
    # stalls identically. The debounce that follows a real reap is cleared so the second tick is evaluated.
    plm._recently_recovered = False
    second = _silent_post_processing_slot(4, silent_for=LIVE_SILENCE_THRESHOLD + 1)
    plm._process_map[4] = second
    del plm._process_map[3]
    second.last_job_referenced = job
    if plm._job_tracker.get_stage(job.id_) is JobStage.PENDING_INFERENCE:
        await plm._job_tracker.mark_inference_started(job)

    plm.replace_hung_processes()

    assert plm._job_tracker.get_stage(job.id_) is not JobStage.PENDING_SUBMIT, (
        "the job was faulted out by two RAM-starvation reaps"
    )
    assert plm._job_tracker.total_num_completed_jobs == 0


# --------------------------------------------------------------------------------------------------
# Boundaries the RAM-pressure grace must not erode.
# --------------------------------------------------------------------------------------------------


async def test_dead_post_processing_slot_is_reaped_even_under_ram_pressure() -> None:
    """A genuinely exited slot is reaped immediately regardless of RAM pressure.

    The grace shields *live* starved slots; a child whose OS process has exited sends nothing further and
    must still be recovered promptly, or the worker would wedge on a corpse for the whole pressure window.
    """
    state = WorkerState()
    state.ram_pressure_pop_hold = True
    slot = _silent_post_processing_slot(3, silent_for=LIVE_SILENCE_THRESHOLD + 1)
    # The process actually died (a plain crash exit, not the OOM SIGKILL path).
    slot.mp_process.is_alive.return_value = False
    slot.mp_process.exitcode = 1  # type: ignore[read-only]
    plm = _make_plm(process_map=ProcessMap({3: slot}), state=state)
    _quiet_os_lifecycle(plm)
    await _attach_in_flight_job(plm, slot)

    replaced = plm.replace_hung_processes()

    assert replaced is True
    assert plm._num_process_recoveries == 1


async def test_preload_stall_under_ram_pressure_is_still_reaped() -> None:
    """RAM-pressure grace is scoped to post-processing; a stuck preload is a different failure and reaps.

    A slot silent past ``preload_timeout`` in ``PRELOADING_MODEL`` is not an already-inferred job worth
    protecting, so the pressure state must not widen its grace.
    """
    state = WorkerState()
    state.ram_pressure_pop_hold = True
    slot = make_mock_process_info(3, model_name="CyberRealistic Pony", state=HordeProcessState.PRELOADING_MODEL)
    slot.mp_process.is_alive.return_value = True
    slot.mp_process.exitcode = None  # type: ignore[read-only]
    now = time.time()
    # preload_timeout is 80 in the fixture; 200 s of silence is unambiguously past it.
    slot.last_received_timestamp = now - 200
    slot.last_heartbeat_timestamp = now - 200
    slot.last_process_state_started_at = now - 200
    plm = _make_plm(process_map=ProcessMap({3: slot}), state=state)
    _quiet_os_lifecycle(plm)
    await _attach_in_flight_job(plm, slot)

    replaced = plm.replace_hung_processes()

    assert replaced is True
    assert plm._num_process_recoveries == 1


async def test_idle_worker_never_runs_the_post_processing_watchdog() -> None:
    """When the last pop found no work, the whole silence-timeout block is skipped for every slot.

    A slot resident in post-processing state while the queue is empty is not evidence of a hang, so an
    idle worker must not reap it even at extreme silence, with or without RAM pressure.
    """
    state = WorkerState()
    state.last_pop_no_jobs_available = True
    slot = _silent_post_processing_slot(3, silent_for=LIVE_SILENCE_THRESHOLD * 4)
    plm = _make_plm(process_map=ProcessMap({3: slot}), state=state)
    _quiet_os_lifecycle(plm)
    await _attach_in_flight_job(plm, slot)

    replaced = plm.replace_hung_processes()

    assert replaced is False
    assert plm._num_process_recoveries == 0
