"""Regression probes for an observed wedge.

An inference slot hung mid-job and was replaced, but the replacement faulted the *wrong* job (the
first process in the map holding any in-flight job, not the job the dying slot was actually running).
The slot's real job was left ``INFERENCE_IN_PROGRESS`` with no owner and pinned the head of the queue
for 6.5 hours, during which no image inference ran at all.

These tests cover both halves of the fix:

* ``_replace_inference_process`` must fault the job belonging to the replaced process.
* the manager's orphaned-in-progress-job watchdog must punt any in-progress job that no live slot
  owns, and a storm of such punts must escalate into the save-our-ship wedge path.
"""

from __future__ import annotations

import multiprocessing
import time
from unittest.mock import Mock

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.process_lifecycle import ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_test_card_runtimes,
    make_test_runtime_config,
    make_testable_process_manager,
    track_popped_job_async,
)


def _make_plm(*, process_map: ProcessMap) -> ProcessLifecycleManager:
    bridge_data = Mock()
    bridge_data.image_models_to_load = ["stable_diffusion"]
    bridge_data.max_threads = 1
    bridge_data.safety_on_gpu = False
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
        horde_model_map=Mock(),
        job_tracker=JobTracker(),
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
        state=WorkerState(),
    )


async def test_replacement_faults_the_replaced_slots_own_job_not_a_peers() -> None:
    """Replacing a hung slot must fault *its* in-flight job, not another live slot's job.

    Reproduces the overnight wedge: a healthy peer (lower process id, so first in map-iteration order)
    holds one in-flight job while a *different*, higher-id slot hangs on its own job. The buggy
    selection scanned the map and faulted the peer's job, orphaning the hung slot's job forever. The
    fix faults the job belonging to the slot being replaced and leaves the peer's job untouched.
    """
    peer = make_mock_process_info(1, model_name="stable_diffusion", state=HordeProcessState.INFERENCE_STARTING)
    hung = make_mock_process_info(2, model_name="stable_diffusion", state=HordeProcessState.INFERENCE_STARTING)
    process_map = ProcessMap({1: peer, 2: hung})

    plm = _make_plm(process_map=process_map)
    # Don't touch real OS processes; only the job-selection/fault logic is under test.
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]

    peer_job = make_job_pop_response(model="stable_diffusion")
    hung_job = make_job_pop_response(model="stable_diffusion")
    await track_popped_job_async(plm._job_tracker, peer_job)
    await track_popped_job_async(plm._job_tracker, hung_job)
    await plm._job_tracker.mark_inference_started(peer_job)
    await plm._job_tracker.mark_inference_started(hung_job)
    peer.last_job_referenced = peer_job
    hung.last_job_referenced = hung_job

    plm._replace_inference_process(hung)

    assert peer_job.id_ is not None
    assert hung_job.id_ is not None
    # The hung slot's job was resolved (terminal fault at the default single-attempt policy)...
    assert plm._job_tracker.get_stage(hung_job.id_) == JobStage.PENDING_SUBMIT
    # ...and the peer's job is left exactly as it was, still in progress, not stolen and faulted.
    assert plm._job_tracker.get_stage(peer_job.id_) == JobStage.INFERENCE_IN_PROGRESS


async def test_watchdog_punts_orphaned_in_progress_job_after_grace() -> None:
    """A job in progress with no owning live slot is punted once the grace window elapses."""
    pm = make_testable_process_manager()
    job = make_job_pop_response(model="stable_diffusion")
    await track_popped_job_async(pm._job_tracker, job)
    await pm._job_tracker.mark_inference_started(job)
    assert job.id_ is not None

    # No inference process exists in the map, so nothing owns this in-progress job: it is orphaned.
    pm._reconcile_orphaned_in_progress_jobs()
    # Within the grace window it is only being watched, not yet punted.
    assert pm._job_tracker.get_stage(job.id_) == JobStage.INFERENCE_IN_PROGRESS

    # Backdate the first-seen time past the grace window and re-run: now it is punted. With the
    # worker's bounded-retry policy the punt requeues the job (a fresh dispatch attempt) rather than
    # faulting it outright; either way it is no longer pinned in progress, and the punt is recorded so
    # a recurring storm can escalate.
    pm._orphan_in_progress_since[job.id_] = time.time() - (pm._ORPHAN_IN_PROGRESS_GRACE_SECONDS + 1)
    pm._reconcile_orphaned_in_progress_jobs()

    assert pm._job_tracker.get_stage(job.id_) != JobStage.INFERENCE_IN_PROGRESS
    assert pm._job_tracker.get_stage(job.id_) == JobStage.PENDING_INFERENCE
    assert len(pm._orphan_punt_history) == 1


async def test_watchdog_leaves_an_owned_in_progress_job_alone() -> None:
    """A job a live inference slot is working on must never be punted by the orphan watchdog."""
    pm = make_testable_process_manager()
    owner = make_mock_process_info(1, model_name="stable_diffusion", state=HordeProcessState.INFERENCE_STARTING)
    pm._process_map[1] = owner

    job = make_job_pop_response(model="stable_diffusion")
    await track_popped_job_async(pm._job_tracker, job)
    await pm._job_tracker.mark_inference_started(job)
    owner.last_job_referenced = job
    assert job.id_ is not None

    # Even with the clock backdated, an owned job is not an orphan and must be left in progress.
    pm._orphan_in_progress_since[job.id_] = time.time() - (pm._ORPHAN_IN_PROGRESS_GRACE_SECONDS + 1)
    pm._reconcile_orphaned_in_progress_jobs()

    assert pm._job_tracker.get_stage(job.id_) == JobStage.INFERENCE_IN_PROGRESS
    assert pm._orphan_punt_history == []


def test_repeated_orphan_punts_escalate_to_wedge() -> None:
    """A storm of orphan punts surfaces as a wedge so the recovery supervisor can limp the worker by."""
    pm = make_testable_process_manager()

    assert pm._orphan_wedge_active() is False
    assert pm._assess_wedge() is False

    now = time.time()
    pm._orphan_punt_history = [now] * pm._ORPHAN_PUNT_WEDGE_THRESHOLD

    assert pm._orphan_wedge_active() is True
    assert pm._assess_wedge() is True


def test_download_only_hold_suppresses_the_wedge_verdict() -> None:
    """A worker held for downloads is never assessed as wedged, so no watchdog reaps it for lack of inference.

    Even with a genuine wedge condition present (an orphan-punt storm), the explicit download-only hold
    short-circuits the verdict: a worker pre-fetching models by design runs no inference and pops no jobs, so
    it must not be soft-reset or aborted (which would also reap its download process). The guard lifts when the
    worker goes live or starts, restoring the normal wedge detection.
    """
    pm = make_testable_process_manager()
    pm._orphan_punt_history = [time.time()] * pm._ORPHAN_PUNT_WEDGE_THRESHOLD
    assert pm._assess_wedge() is True  # the wedge condition is genuinely present

    pm._state.downloads_only_hold = True
    assert pm._assess_wedge() is False  # ...but the download-only hold suppresses any wedge-driven reap

    pm._state.downloads_only_hold = False  # cleared on go-live / start
    assert pm._assess_wedge() is True  # ...and normal wedge detection resumes


async def test_detected_deadlock_escalates_to_recovery_wedge() -> None:
    """A detected scheduler deadlock must be visible to the recovery supervisor."""
    pm = make_testable_process_manager()
    pm._state.last_job_pop_time = time.time() - 60

    job = make_job_pop_response(model="stable_diffusion")
    await track_popped_job_async(pm._job_tracker, job)

    pm.detect_deadlock()
    # The deadlock has persisted past the structural-wedge window (a genuine stuck queue, not the
    # transient all-idle gap between jobs), so it is visible to the recovery supervisor.
    pm._message_dispatcher._last_queue_deadlock_detected_time = time.time() - 60

    assert pm._message_dispatcher.get_deadlock_snapshot().has_active_deadlock() is True
    assert pm._assess_wedge() is True


async def test_safety_tail_during_lull_is_not_a_wedge() -> None:
    """A job draining through the safety/submit tail during a queue lull must not trip the wedge.

    With the inference queue empty and the last job sitting in PENDING_SAFETY_CHECK, every process is idle
    so the *general* deadlock detector fires (a tracked job exists, nothing is busy). That benign state
    must not feed the wedge assessment (which would cycle healthy processes into a reduced-concurrency
    limp-by). It is not a *queue* deadlock (no pending inference work), so it must read as not-wedged.
    """
    pm = make_testable_process_manager()
    pm._state.last_job_pop_time = time.time() - 60

    idle_inference = make_mock_process_info(1, state=HordeProcessState.WAITING_FOR_JOB)
    pm._process_map[1] = idle_inference

    safety_pending = Mock()
    safety_pending.sdk_api_job_info = make_job_pop_response(model="stable_diffusion")
    await pm._job_tracker.queue_for_safety(safety_pending)

    assert len(pm._job_tracker.jobs_pending_safety_check) == 1
    assert len(pm._job_tracker.jobs_pending_inference) == 0

    pm.detect_deadlock()

    snapshot = pm._message_dispatcher.get_deadlock_snapshot()
    assert snapshot.has_active_deadlock() is True  # the loose detector still flags it (diagnostics)
    assert snapshot.indicates_structural_wedge() is False  # but it is not a structural wedge
    assert pm._assess_wedge() is False


def test_stale_orphan_punts_age_out_of_the_wedge_window() -> None:
    """Punts older than the window do not count, so a long-ago blip is not treated as an active wedge."""
    pm = make_testable_process_manager()
    old = time.time() - (pm._ORPHAN_PUNT_WINDOW_SECONDS + 1)
    pm._orphan_punt_history = [old] * (pm._ORPHAN_PUNT_WEDGE_THRESHOLD + 2)

    assert pm._orphan_wedge_active() is False
    assert pm._orphan_punt_history == []  # pruned as a side effect


async def test_give_up_reissues_head_when_pool_healthy_but_queue_wedged() -> None:
    """Give-up must reissue a head the scheduler structurally cannot serve, even with a healthy pool.

    The save-our-ship give-up only faulted pending jobs when inference capacity was *unavailable*. A
    worker whose pool is perfectly healthy (idle processes) but whose scheduler is structurally wedged
    (pending inference work, every process idle, no progress) therefore faulted nothing and spun
    forever, until a manual shutdown faulted the stuck jobs. A sustained *queue* deadlock with capacity
    available must still reissue the unservable head so the horde reassigns it and the queue unblocks.
    """
    pm = make_testable_process_manager()
    pm._state.last_job_pop_time = time.time() - 60

    # A healthy, idle inference process: capacity IS available (so the old give-up faulted nothing).
    idle_inference = make_mock_process_info(0, model_name="resident", state=HordeProcessState.WAITING_FOR_JOB)
    pm._process_map[0] = idle_inference

    # A pending head whose model is not resident and that the scheduler cannot place; nothing in flight.
    head_job = make_job_pop_response(model="unschedulable")
    await track_popped_job_async(pm._job_tracker, head_job)

    pm.detect_deadlock()
    # Sustain the queue deadlock past the structural-wedge window: this head is genuinely unservable,
    # not the transient all-idle gap while the scheduler preloads the next model.
    pm._message_dispatcher._last_queue_deadlock_detected_time = time.time() - 60
    assert pm._message_dispatcher.get_deadlock_snapshot().indicates_structural_wedge() is True
    assert pm._is_inference_capacity_available() is True
    assert len(pm._job_tracker.jobs_pending_inference) == 1

    pm._give_up_on_wedged_jobs()

    # The head was reissued (faulted to PENDING_SUBMIT) rather than left to spin behind a healthy pool.
    assert len(pm._job_tracker.jobs_pending_inference) == 0
