"""Reproduction: a whole-card head starves behind a post-processing lane's un-reclaimed resident VRAM.

On a 24 GB card a whole-card model (Flux fp8, ~11.5 GB weights / ~16 GB resident trio, wanting sole
residency) can reach the head of the queue while the dedicated post-processing lane holds several GB of
resident upscaler/face-fixer weights. The measured device-free then sits below the head's demand. The
remedy is NOT a sibling-context teardown: an idle inference context on this platform costs only a few
hundred MB (the probe-measured marginal), so tearing contexts down frees almost nothing. The room the head
needs is the post-processing lane's resident model weights, reclaimed by unloading that lane's modules (its
cheap CUDA context may stay) so free VRAM rises toward the after-model-evict figure the head fits into.

Two failure surfaces are exercised:

* The residency path must actually claw the post-processing lane's resident modules back for the head. While
  the lane is busy it is spared (its in-flight job must finish), but once it is idle its modules must be
  unloaded so the head can fit. If the reclaim never lands, the head is never served.
* The structural-queue-wedge recovery must not permanently fault the head's backlog while a post-processing
  reclaim remedy is still reachable. Faulting a servable head whose only blocker is reclaimable lane VRAM
  drops work the card could run once the lane yields.

The scheduler cases drive the real ``preload_models`` admission path; the recovery cases drive the real
``run_recovery_supervisor`` / ``give_up_on_wedged_jobs`` paths on a fake escalation clock. Contexts are
cheap here by construction (a probe-measured marginal), so any teardown behavior is out of scope: this
module is about the post-processing lane's resident weights, not context count.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.action_ledger import LedgerEvent, LedgerEventType
from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.lifecycle.recovery_supervisor import RecoverySupervisor
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

# --- Corrected incident constants (24 GB RTX 4090, Linux). Contexts are cheap; the blocker is PP VRAM. ---
_DEVICE_TOTAL_VRAM_MB = 24074.0
_DEVICE_FREE_AT_WEDGE_MB = 11050.0
"""Measured device-free with the PP lane's ~7 GB of modules still resident: below the head's ~14 GB need."""
_DEVICE_FREE_AFTER_PP_EVICT_MB = 18456.0
"""The after-model-evict figure the head fits into once the PP lane's modules are unloaded."""

_FLUX_MODEL = "Flux.1-Schnell fp8 (Compact)"
_FLUX_WEIGHTS_MB = 11500.0
_FLUX_FOOTPRINT_MB = 16000.0  # the fp8 trio: no room for a co-resident model on a 24 GB card
_FLUX_SAMPLING_PEAK_MB = 14000.0  # weights + ~2500 MB working set

_PP_RESIDENT_LARGE_MB = 6429.0  # the upscaler/face-fixer peak the lane holds resident
_PP_RESIDENT_SMALL_MB = 3000.0

_MARGINAL_CONTEXT_MB = 487.0  # probe-measured per-additional-context cost: cheap
_PER_PROCESS_OVERHEAD_MB = 3183.0  # first/sole context: baseline + one-time CUDA runtime + one context

_CARD_16GB_MB = 16375.0

_NUM_INFERENCE_PROCESSES = 4


def _install_flux_forecast_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the head model's weight/footprint/peak so it wants the whole card and does not co-reside."""
    monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda job, baseline: _FLUX_WEIGHTS_MB)
    monkeypatch.setattr(resource_budget, "predict_job_footprint_mb", lambda job, baseline: _FLUX_FOOTPRINT_MB)
    monkeypatch.setattr(resource_budget, "predict_job_sampling_vram_mb", lambda job, baseline: _FLUX_SAMPLING_PEAK_MB)


def _build_pp_contended_scheduler(
    *,
    device_total_mb: float = _DEVICE_TOTAL_VRAM_MB,
    device_free_mb: float = _DEVICE_FREE_AT_WEDGE_MB,
    pp_lane_state: HordeProcessState = HordeProcessState.WAITING_FOR_JOB,
    whole_card_exclusive_residency: bool = True,
    include_pp_lane: bool = True,
) -> tuple[InferenceScheduler, ProcessMap, JobTracker, HordeProcessInfo | None]:
    """Build a scheduler whose card holds four idle inference contexts, safety, and a resident PP lane.

    The inference processes are model-free and idle, so the room the head needs is held by the
    post-processing lane's resident modules, not by any inference model. ``device_free_mb`` models the
    measured device-free the arbiter prices against; ``pp_lane_state`` lets a case model the lane busy
    (mid-job, must be spared) or idle (its modules are reclaimable).
    """
    procs: dict[int, HordeProcessInfo] = {}
    for pid in range(1, _NUM_INFERENCE_PROCESSES + 1):
        proc = make_mock_process_info(pid, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        proc.total_vram_mb = device_total_mb
        proc.vram_usage_mb = device_total_mb - device_free_mb
        procs[pid] = proc
    # A safety context on the card, as configured in the incident (safety_on_gpu).
    safety = make_mock_process_info(
        0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, process_type=HordeProcessType.SAFETY
    )
    safety.total_vram_mb = device_total_mb
    procs[0] = safety

    pp_lane: HordeProcessInfo | None = None
    if include_pp_lane:
        pp_lane = make_mock_process_info(
            9, model_name=None, state=pp_lane_state, process_type=HordeProcessType.POST_PROCESS
        )
        pp_lane.total_vram_mb = device_total_mb
        procs[9] = pp_lane

    process_map = ProcessMap(procs)
    job_tracker = JobTracker()
    bridge_data = make_mock_bridge_data(
        enable_vram_budget=True,
        whole_card_exclusive_residency=whole_card_exclusive_residency,
        vram_reserve_mb=2048,
        ram_reserve_mb=4096,
        overbudget_exclusive_mode=True,
        safety_on_gpu=True,
        post_process_job_overlap=True,
        image_models_to_load=[_FLUX_MODEL],
        max_threads=1,
    )
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=bridge_data,
        max_concurrent=1,
        max_inference=_NUM_INFERENCE_PROCESSES,
        device_free_mb=device_free_mb,
    )
    # Cheap, probe-measured contexts: this scenario is explicitly not a context-teardown case.
    scheduler.set_measured_marginal_overhead_mb(_MARGINAL_CONTEXT_MB)

    lifecycle = scheduler._process_lifecycle
    lifecycle.post_process_lane_enabled = Mock(return_value=include_pp_lane)
    lifecycle.is_post_process_gpu_paused = False
    lifecycle.is_safety_gpu_paused = False
    lifecycle.vae_lane_enabled = Mock(return_value=False)
    lifecycle.component_lane_enabled = Mock(return_value=False)
    lifecycle.scale_inference_processes = Mock(return_value=_NUM_INFERENCE_PROCESSES)
    lifecycle.pause_post_process_off_gpu = Mock(return_value=True)
    lifecycle.pause_safety_on_gpu = Mock(return_value=True)
    return scheduler, process_map, job_tracker, pp_lane


async def _queue_flux_head(job_tracker: JobTracker) -> object:
    """Put the whole-card head at the front of the inference queue."""
    head_job = make_job_pop_response(_FLUX_MODEL)
    await track_popped_job_async(job_tracker, head_job)
    return head_job


def _mark_head_starved(scheduler: InferenceScheduler, head_job: object, *, seconds: float = 45.0) -> None:
    """Backdate the head's starvation clock so the arbiter's first-party context escalation is eligible.

    With whole-card exclusive residency off, the residency path is reached only through the arbiter's
    starvation escalation, which requires the head to have been the undispatched head of an idle device
    past its short grace. This models the incident's long-starved head.
    """
    job_id = head_job.id_  # pyrefly: ignore - the pop response always carries an id in these tests.
    scheduler._head_starvation_job_id = str(job_id)
    scheduler._head_starvation_since = time.time() - seconds


# --------------------------------------------------------------------------------------------------------
# Part A: the residency path must reclaim the post-processing lane's resident modules for the head.
# --------------------------------------------------------------------------------------------------------


class TestResidencyReclaimsPostProcessLane:
    """A whole-card head on a card the PP lane over-commits must unload the idle lane's modules."""

    async def test_remedy_is_model_eviction_not_context_teardown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On a 24 GB card the head fits with the PP lane's cheap context alive: only its models must go.

        This pins the physical shape: the deficit is the lane's resident model weights (reclaimable by an
        unload), not its CUDA context (which a teardown would reclaim). If this read flipped, the whole
        premise (reclaim the lane's modules, keep its context) would be wrong.
        """
        _install_flux_forecast_inputs(monkeypatch)
        scheduler, _process_map, _job_tracker, _pp = _build_pp_contended_scheduler()
        forecast = scheduler._forecast_streaming(make_job_pop_response(_FLUX_MODEL), "flux_1")

        assert forecast.needs_exclusive_residency is True
        assert forecast.requires_sibling_teardown is False
        assert scheduler._post_process_context_fits_with_residency(forecast, device_index=None) is True

    async def test_idle_pp_lane_modules_are_unloaded_for_the_head(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An idle post-processing lane holding resident modules is unloaded so the head can fit.

        The head cannot be admitted while the lane holds its ~7 GB; the residency reclaim must ask the idle
        lane to unload its modules. The head still defers this tick (the freed VRAM has not materialised),
        but the reclaim it needs must have been issued.
        """
        _install_flux_forecast_inputs(monkeypatch)
        scheduler, _process_map, job_tracker, pp_lane = _build_pp_contended_scheduler()
        assert pp_lane is not None
        await _queue_flux_head(job_tracker)

        admitted = scheduler.preload_models()

        assert admitted is False, "the head must not co-reside into a card the PP lane over-commits"
        assert pp_lane.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM, (
            "the idle post-processing lane's modules were never reclaimed, so the head's room never frees"
        )

    async def test_flag_off_preventative_lane_reclaim_does_not_run_at_preload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With exclusive residency off (the incident config), no preventative lane reclaim runs at preload.

        Preventative whole-card residency (which unloads the lane's modules before the head loads) is gated
        on the exclusive-residency flag. With it off, the head's weights RAM-preload without reclaiming the
        lane: its resident VRAM contention is deferred to the dispatch/sampling stage rather than being
        resolved up front. This characterizes the incident's contributing condition; the resulting harm is
        pinned by the recovery-path case below, where the deferred contention wedges and the backlog is
        faulted.
        """
        _install_flux_forecast_inputs(monkeypatch)
        scheduler, _process_map, job_tracker, pp_lane = _build_pp_contended_scheduler(
            whole_card_exclusive_residency=False
        )
        assert pp_lane is not None
        head_job = await _queue_flux_head(job_tracker)
        _mark_head_starved(scheduler, head_job)

        scheduler.preload_models()

        # No preventative reclaim fired: the lane keeps its resident modules while the head stages into RAM.
        assert pp_lane.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
        assert scheduler._process_lifecycle.pause_post_process_off_gpu.called is False

    async def test_busy_pp_lane_is_spared_then_reclaimed_once_idle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A lane mid-job is never torn down; once it goes idle its modules must be reclaimed for the head.

        This is the incident's start (the lane was post-processing when the head demanded the card). The
        busy lane must be left to finish, but the re-drive after it idles must claw its modules back. A
        reclaim that only ever fires while the lane is busy (and so never lands) is the wedge.
        """
        _install_flux_forecast_inputs(monkeypatch)
        scheduler, _process_map, job_tracker, pp_lane = _build_pp_contended_scheduler(
            pp_lane_state=HordeProcessState.POST_PROCESSING
        )
        assert pp_lane is not None
        await _queue_flux_head(job_tracker)

        scheduler.preload_models()
        assert pp_lane.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM, (
            "a busy post-processing lane must not be unloaded mid-job"
        )

        # The lane's job drains and it goes idle; the head's next admission attempt must reclaim it.
        pp_lane.last_process_state = HordeProcessState.WAITING_FOR_JOB
        scheduler.preload_models()
        assert pp_lane.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM, (
            "the now-idle post-processing lane's modules were never reclaimed for the starved head"
        )


# --------------------------------------------------------------------------------------------------------
# Part B: the structural-queue-wedge give-up must not fault a head whose PP reclaim is still reachable.
# --------------------------------------------------------------------------------------------------------


class _FakeClock:
    """A monotonic clock the test advances explicitly, so escalation timing is deterministic."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _install_supervisor(pm: HordeWorkerProcessManager, clock: _FakeClock) -> RecoverySupervisor:
    """Swap in a supervisor on the fake clock with tight, legible escalation windows."""
    supervisor = RecoverySupervisor(
        clock=clock,
        wedge_grace_seconds=1,
        reset_interval_seconds=1,
        max_soft_resets=1,
        pool_ready_grace_seconds=3,
        boot_allowance_seconds=20,
        give_up_cooldown_seconds=5,
        max_give_up_cycles=2,
        clean_streak_seconds=100,
    )
    pm._recovery_coordinator.recovery_supervisor = supervisor
    return supervisor


def _abandoned_records(pm: HordeWorkerProcessManager) -> list[LedgerEvent]:
    """All RECOVERY_ABANDONED events currently in the action ledger."""
    return [
        event
        for event in pm._recovery_coordinator._action_ledger.recent(limit=1000)
        if event.event_type == LedgerEventType.RECOVERY_ABANDONED
    ]


async def _latch_wedge_with_reclaimable_pp_lane(pm: HordeWorkerProcessManager, *, model: str) -> HordeProcessInfo:
    """Latch a structural queue wedge over an idle inference slot beside a resident, idle PP lane.

    The head is servable the moment the post-processing lane yields its resident VRAM, so this is a wedge
    whose remedy (reclaim the lane) is reachable, not a structurally doomed pool.
    """
    proc = make_mock_process_info(0, model_name=model, state=HordeProcessState.WAITING_FOR_JOB)
    pm._process_map[0] = proc
    pp_lane = make_mock_process_info(
        9, model_name=None, state=HordeProcessState.WAITING_FOR_JOB, process_type=HordeProcessType.POST_PROCESS
    )
    pm._process_map[9] = pp_lane
    await track_popped_job_async(pm._job_tracker, make_job_pop_response(model=model))

    dispatcher = pm._message_dispatcher
    dispatcher._in_queue_deadlock = True
    dispatcher._queue_deadlock_model = model
    dispatcher._last_queue_deadlock_detected_time = time.time() - 60
    return pp_lane


async def _drive_to_first_give_up(
    pm: HordeWorkerProcessManager, clock: _FakeClock, monkeypatch: pytest.MonkeyPatch
) -> bool:
    """Tick a ready, persistently wedged pool until the first give-up fires; True if it did."""
    proc = pm._process_map[0]
    monkeypatch.setattr(
        pm._process_lifecycle,
        "rebuild_inference_pool",
        lambda *, reason: setattr(proc, "last_process_state", HordeProcessState.WAITING_FOR_JOB),
    )
    monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)

    for _ in range(40):
        clock.advance(1)
        pm._recovery_coordinator.run_recovery_supervisor()
        if _abandoned_records(pm):
            return True
    return False


class TestGiveUpGuardsReclaimableHead:
    """The give-up must not permanently fault a head whose only blocker is a reclaimable PP lane."""

    async def test_head_with_reachable_pp_reclaim_is_not_faulted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A whole-card head wedged only by a reclaimable PP lane must not be dropped by the give-up.

        The remedy is to yield the lane's resident VRAM so the head fits, not to fault the head. This
        asserts the contract the recovery path should honor: a head whose blocker is reclaimable lane VRAM
        is not abandoned.
        """
        pm = make_testable_process_manager(device_free_mb=_DEVICE_FREE_AT_WEDGE_MB)
        clock = _FakeClock()
        _install_supervisor(pm, clock)
        await _latch_wedge_with_reclaimable_pp_lane(pm, model=_FLUX_MODEL)
        monkeypatch.setattr(pm, "_abort", lambda: None)

        assert pm._recovery_coordinator.assess_wedge() is True

        await _drive_to_first_give_up(pm, clock, monkeypatch)

        faulting_records = [record for record in _abandoned_records(pm) if record.detail.get("jobs_faulted", 0) > 0]
        assert faulting_records == [], (
            "the head was faulted while a post-processing reclaim remedy was still reachable"
        )
        assert len(list(pm._job_tracker.jobs_pending_inference)) == 1, "the servable head was dropped"

    async def test_control_doomed_pool_head_is_still_reissued(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Control: with no reclaim remedy (no PP lane), a persistently wedged head is still reissued.

        This preserves the existing safety valve: a genuinely stuck queue-deadlock over a ready pool still
        reissues its head so the horde can hand it to another worker. The contrast with the case above is
        the reachable reclaim remedy, not the give-up machinery itself.
        """
        pm = make_testable_process_manager()
        clock = _FakeClock()
        _install_supervisor(pm, clock)
        # No PP lane: a resident, servable head over a ready pool with nothing left to reclaim.
        proc = make_mock_process_info(0, model_name="Deliberate", state=HordeProcessState.WAITING_FOR_JOB)
        pm._process_map[0] = proc
        await track_popped_job_async(pm._job_tracker, make_job_pop_response(model="Deliberate"))
        dispatcher = pm._message_dispatcher
        dispatcher._in_queue_deadlock = True
        dispatcher._queue_deadlock_model = "Deliberate"
        dispatcher._last_queue_deadlock_detected_time = time.time() - 60
        monkeypatch.setattr(pm, "_abort", lambda: None)

        fired = await _drive_to_first_give_up(pm, clock, monkeypatch)

        assert fired is True
        records = _abandoned_records(pm)
        assert len(records) == 1
        assert records[0].detail["jobs_faulted"] == 1
        assert len(list(pm._job_tracker.jobs_pending_inference)) == 0


# --------------------------------------------------------------------------------------------------------
# Part C: varied circumstances. Disposition is unknown until run: each asserts the liveness property that a
# card unable to host both the head and the PP lane's resident VRAM must issue a lane reclaim for the head.
# --------------------------------------------------------------------------------------------------------


_MATRIX = [
    (
        "24gb_large_pp_idle_exclusive",
        _DEVICE_TOTAL_VRAM_MB,
        _PP_RESIDENT_LARGE_MB,
        HordeProcessState.WAITING_FOR_JOB,
        True,
    ),
    (
        "24gb_large_pp_idle_no_flag",
        _DEVICE_TOTAL_VRAM_MB,
        _PP_RESIDENT_LARGE_MB,
        HordeProcessState.WAITING_FOR_JOB,
        False,
    ),
    (
        "24gb_small_pp_idle_exclusive",
        _DEVICE_TOTAL_VRAM_MB,
        _PP_RESIDENT_SMALL_MB,
        HordeProcessState.WAITING_FOR_JOB,
        True,
    ),
    ("16gb_large_pp_idle_exclusive", _CARD_16GB_MB, _PP_RESIDENT_LARGE_MB, HordeProcessState.WAITING_FOR_JOB, True),
]


class TestVariedContentionLiveness:
    """A card that cannot host both the head and the lane's resident VRAM must reclaim the lane for the head."""

    @pytest.mark.parametrize(
        ("label", "device_total_mb", "pp_resident_mb", "pp_state", "exclusive_flag"),
        _MATRIX,
        ids=[row[0] for row in _MATRIX],
    )
    async def test_idle_lane_reclaimed_when_head_cannot_coreside(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        device_total_mb: float,
        pp_resident_mb: float,
        pp_state: HordeProcessState,
        exclusive_flag: bool,
    ) -> None:
        """When the head cannot co-reside with the lane's resident VRAM, the idle lane must be reclaimed.

        The device-free models the lane's modules still resident (total minus the contexts and the lane's
        resident weights). The reclaim is either an unload of the lane's modules (a card that keeps the
        cheap context) or a full lane pause off the GPU (a card too tight even for the context). Either
        outcome frees the lane's VRAM for the head; leaving the modules resident with the head un-served is
        the wedge this asserts against.
        """
        _install_flux_forecast_inputs(monkeypatch)
        contexts_mb = _PER_PROCESS_OVERHEAD_MB + _MARGINAL_CONTEXT_MB * (_NUM_INFERENCE_PROCESSES + 1)
        device_free_mb = max(0.0, device_total_mb - contexts_mb - pp_resident_mb)
        scheduler, _process_map, job_tracker, pp_lane = _build_pp_contended_scheduler(
            device_total_mb=device_total_mb,
            device_free_mb=device_free_mb,
            pp_lane_state=pp_state,
            whole_card_exclusive_residency=exclusive_flag,
        )
        assert pp_lane is not None
        head_job = await _queue_flux_head(job_tracker)
        _mark_head_starved(scheduler, head_job)

        admitted = scheduler.preload_models()

        # Liveness: the head must make progress. Either the measured room already held it (a small resident
        # lane the head co-resides beside), or the lane's resident VRAM was reclaimed for it (an unload of the
        # lane's modules, or a full lane pause on a card too tight even for the context). Leaving the head
        # un-served with the lane's modules resident is the wedge.
        lane_unloaded = pp_lane.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
        lane_paused = scheduler._process_lifecycle.pause_post_process_off_gpu.called
        assert admitted or lane_unloaded or lane_paused, (
            f"[{label}] the head neither admitted nor reclaimed the lane's {pp_resident_mb:.0f} MB: it "
            "starves with the lane's modules resident"
        )


# --------------------------------------------------------------------------------------------------------
# Part D: controls. These pin the boundaries so the reclaim does not over-reach.
# --------------------------------------------------------------------------------------------------------


class TestReclaimBoundaries:
    """The lane reclaim fires only when the head genuinely needs the room, and spares live lane work."""

    async def test_no_pp_lane_no_spurious_reclaim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no post-processing lane present, the head drives no post-processing action.

        A card with the head and idle inference contexts but no lane exercises the ordinary whole-card path.
        No pause_post_process action may be issued: there is nothing to reclaim.
        """
        _install_flux_forecast_inputs(monkeypatch)
        scheduler, _process_map, job_tracker, pp_lane = _build_pp_contended_scheduler(include_pp_lane=False)
        assert pp_lane is None
        await _queue_flux_head(job_tracker)

        scheduler.preload_models()

        scheduler._process_lifecycle.pause_post_process_off_gpu.assert_not_called()

    async def test_head_admits_once_lane_vram_has_freed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After the lane's modules unload and device-free reflects the freed room, the head admits.

        This closes the loop: the reclaim is not busywork. Once the measured device-free rises to the
        after-model-evict figure the head fits into, the same head admits rather than deferring forever.
        """
        _install_flux_forecast_inputs(monkeypatch)
        scheduler, _process_map, job_tracker, pp_lane = _build_pp_contended_scheduler()
        assert pp_lane is not None
        head_job = await _queue_flux_head(job_tracker)

        # First pass reclaims the lane and defers.
        assert scheduler.preload_models() is False
        assert pp_lane.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM

        # The lane's modules have freed: the device now reports the after-evict room, and the lane is empty.
        scheduler.set_device_free_mb_provider(lambda _device_index: _DEVICE_FREE_AFTER_PP_EVICT_MB)
        pp_lane.vram_usage_mb = 0.0
        for pid in range(1, _NUM_INFERENCE_PROCESSES + 1):
            scheduler._process_map[pid].vram_usage_mb = _DEVICE_TOTAL_VRAM_MB - _DEVICE_FREE_AFTER_PP_EVICT_MB

        admitted = scheduler.preload_models()
        assert admitted is True, "the head still did not admit after the lane's VRAM was reclaimed"
        assert scheduler._job_tracker.is_admitted_exclusive(head_job) is True

    async def test_busy_lane_is_never_unloaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A post-processing lane running a job is never unloaded, preserving its in-flight work.

        Reclaim liveness must not trade a starved head for a faulted post-processing job: the lane finishes
        first. This is the boundary the idle-reclaim path must respect.
        """
        _install_flux_forecast_inputs(monkeypatch)
        scheduler, _process_map, job_tracker, pp_lane = _build_pp_contended_scheduler(
            pp_lane_state=HordeProcessState.POST_PROCESSING
        )
        assert pp_lane is not None
        await _queue_flux_head(job_tracker)

        scheduler.preload_models()

        assert pp_lane.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM


class TestGiveUpRemedyGraceIsBounded:
    """The reclaim yield is a bounded grace, not a new way to park forever."""

    async def test_reclaim_that_never_lands_still_faults_after_the_grace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hostile self-infliction: the lane accepts the unload but its VRAM never frees; give-up still fires.

        The guard must drive the lane unload and yield, but a remedy that provably fails to land within
        :data:`WorkerRecoveryCoordinator.PP_RECLAIM_REMEDY_GRACE_SECONDS` may not suppress the safety valve:
        past the window the ordinary give-up faults the backlog so the horde reissues it. Without this bound
        the guard would convert the faulted-backlog harm into an unbounded park, the worse wedge class.
        """
        pm = make_testable_process_manager(device_free_mb=_DEVICE_FREE_AT_WEDGE_MB)
        clock = _FakeClock()
        _install_supervisor(pm, clock)
        coordinator = pm._recovery_coordinator
        coordinator._clock = clock  # the remedy grace must run on the same driven clock
        pp_lane = await _latch_wedge_with_reclaimable_pp_lane(pm, model=_FLUX_MODEL)
        monkeypatch.setattr(pm, "_abort", lambda: None)

        proc = pm._process_map[0]
        monkeypatch.setattr(
            pm._process_lifecycle,
            "rebuild_inference_pool",
            lambda *, reason: setattr(proc, "last_process_state", HordeProcessState.WAITING_FOR_JOB),
        )
        monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)

        grace = coordinator.PP_RECLAIM_REMEDY_GRACE_SECONDS
        first_fault_at: float | None = None
        for _ in range(int(grace) + 60):
            clock.advance(1)
            coordinator.run_recovery_supervisor()
            if any(record.detail.get("jobs_faulted", 0) > 0 for record in _abandoned_records(pm)):
                first_fault_at = clock.now
                break

        # The remedy was genuinely driven: the idle lane was asked to unload its modules.
        assert pp_lane.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
        # And the yield stayed bounded: the give-up faulted the backlog once the grace expired, not before.
        assert first_fault_at is not None, "the guard parked the wedge forever: the give-up never fired"
        assert first_fault_at > grace, "the give-up faulted inside the remedy grace it was meant to yield"
        assert len(list(pm._job_tracker.jobs_pending_inference)) == 0
