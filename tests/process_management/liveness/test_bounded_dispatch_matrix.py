"""Bounded dispatch of a physically servable head, as a property over the whole scheduling loop.

A worker is wedged when it holds work it could serve and stops progressing it. The admission matrix in
``tests/process_management/resources/test_admission_liveness_matrix.py`` proves that property at the arbiter
surface, over a card model it prices directly. This module lifts the same property to the pipeline: a real
:class:`InferenceScheduler` driven across scheduling ticks against a card whose free VRAM is derived from what
is actually resident on it, so preload admission, whole-card residency, the churn governors, reclaim, and the
dispatch gate all compose the way they do in the running worker.

The property, stated once:

    For any head of queue that is physically servable (its weights fit the card outright, fit once the reclaim
    it is entitled to has run, or fit co-resident), the head reaches dispatch within a bounded number of
    scheduling ticks, and every obligation opened on the way there is closed on the way out.

"Obligation" is concrete: a planned reserve-ledger charge booked for an admitted preload, a whole-card
residency claim over the card, and the safety / post-processing GPU pauses a residency takes out. Each is
asserted released once the row's queue has drained, because a charge that outlives the work it was booked for
is exactly the double-count that wedges the next head.

Every assertion is a positive outcome read through real state: a job entering progress through the dispatch
path, a lane carrying the START_INFERENCE flag, a model named on the ceiling hold, a cycle-frozen measurement
carrying no planned charge. None of them can be satisfied by the failure they guard against.

Not every head is servable, and the table says which. A model whose sampling peak prices above the card's
achievable ceiling takes the worker's other disclosed exit: the model goes on the ceiling hold and its queued
job is faulted onward for reissue. Rows carrying that expectation are what stop the property from degenerating
into "everything asked of the worker is admitted".

Axes and their values are enumerated in ``_ROWS``; combinations deliberately left out are listed with their
reason in ``_PRUNED_COMBINATIONS`` so the table's coverage is never silently truncated.

Scope. This drives the scheduler's own loop (governance tick, preload pass, dispatch pass) with a hand-run
world standing in for the children: a preload materialises on the following tick, an unload command frees the
model's weights, a dispatched job completes on the following tick. It does not drive the process manager's
message pump, the recovery supervisor, or the pop path; those are covered by the manager and recovery suites.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from unittest.mock import Mock

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeImageResult,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.lru_cache import LRUCache
from horde_worker_regen.process_management.resources.resource_budget import CommittedReserveLedger
from horde_worker_regen.process_management.resources.vram_arbiter import MeasuredVramSnapshot
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    _ESTABLISH_WINDOW_LIMIT,
    _GRACE_BUDGET_SECONDS,
    offer_under_pop_claim,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
    _WHOLE_CARD_RESTORE_GRACE_SECONDS,
    InferenceScheduler,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_model_metadata,
    make_test_runtime_config,
    track_popped_job_async,
)

# --------------------------------------------------------------------------------------------------------
# Hardware and model classes
# --------------------------------------------------------------------------------------------------------

# The first (sole) CUDA context costs the one-time runtime allocation; every additional context costs the
# marginal only. Pinned through config so a row's arithmetic does not depend on a measured host.
_FIRST_CONTEXT_MB = 1354.0
_MARGINAL_CONTEXT_MB = 384.0
"""Matches the seeded marginal the forecast falls back to when no probe measurement exists, so the world's
own accounting and the scheduler's forecast charge sibling contexts identically."""

_AMPLE_RAM_MB = 65_536.0
"""The host RAM reading every row runs against. These rows vary VRAM; a live psutil reading would make a
heavy row's outcome depend on the machine running it."""

_TICK_SECONDS = 30.0
"""How much of the world's clock one scheduling tick advances.

The scheduler's governance windows are sized in tens of seconds (a churn governor holds a head off the card
for a four-minute dwell, a residency restore takes a one-minute wedge grace, a drain settles inside twenty
seconds). A tick worth a single second would put every one of those bounds outside any tick budget a row
could state, so the table could only ever assert what happens before them. Advancing the shared clock by a
scheduling interval per tick makes them reachable, which is what lets a row assert that a governed head is
eventually served rather than only that it is currently held."""


@dataclass(frozen=True)
class _CardClass:
    """One device-free profile: total VRAM, from which the world derives free VRAM as tenants come and go."""

    label: str
    total_mb: float


_CARD_8GB = _CardClass("8gb", 8192.0)
_CARD_16GB = _CardClass("16gb", 16384.0)
_CARD_24GB = _CardClass("24gb", 24576.0)


@dataclass(frozen=True)
class _ModelClass:
    """One model class: the reference record the scheduler prices from, and the world's residency charge.

    ``weights_mb`` is the world's own bookkeeping figure (what a resident copy costs the card), kept equal to
    hordelib's per-baseline resident weight seed so the scheduler's forecast and the world's free-VRAM
    derivation agree about what residency costs.
    """

    label: str
    name: str
    baseline: KNOWN_IMAGE_GENERATION_BASELINE
    weights_mb: float


# stable_diffusion_1: 3200 MB weights. The small class that co-resides almost anywhere.
_SD15 = _ModelClass("sd15", "sd15-checkpoint", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1, 3200.0)
_SD15_OTHER = _ModelClass("sd15_b", "sd15-checkpoint-b", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1, 3200.0)
# stable_diffusion_xl: 4900 MB core weights (6600 MB with its support components).
_SDXL = _ModelClass("sdxl", "sdxl-checkpoint", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl, 4900.0)
_SDXL_OTHER = _ModelClass("sdxl_b", "sdxl-checkpoint-b", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl, 4900.0)
# flux_1: 11500 MB core weights, and EXTRA_LARGE by tier, so it takes the whole-card residency path.
_FLUX = _ModelClass("flux", "Flux.1-Schnell fp8 (Compact)", KNOWN_IMAGE_GENERATION_BASELINE.flux_1, 11500.0)

_MODEL_CLASSES = (_SD15, _SD15_OTHER, _SDXL, _SDXL_OTHER, _FLUX)

_SAME_CLASS_PARTNER: dict[str, _ModelClass] = {
    _SD15.name: _SD15_OTHER,
    _SD15_OTHER.name: _SD15,
    _SDXL.name: _SDXL_OTHER,
    _SDXL_OTHER.name: _SDXL,
    _FLUX.name: _SDXL,
}
"""The second checkpoint an interleaved queue alternates with. Same weight class as the head wherever one
exists, so the interleave axis varies residency rotation without also changing what the card can serve."""


# --------------------------------------------------------------------------------------------------------
# Axis values
# --------------------------------------------------------------------------------------------------------


class _Residency(Enum):
    """Where the head's checkpoint sits when the row starts."""

    ABSENT = "absent"
    """Nothing holds it; a lane must preload it."""
    RESIDENT_IDLE_TARGET = "resident_idle_target"
    """A lane already holds it in VRAM and is idle, so dispatching moves nothing."""
    RESIDENT_ON_SIBLING = "resident_on_sibling"
    """A lane holds it, but that lane is running another job, so the head waits for it or lands elsewhere."""
    RAM_STAGED = "ram_staged"
    """A lane has it staged in RAM (preloaded, not yet committed to VRAM)."""


class _QueueShape(Enum):
    """The structure of the queue the head sits at the front of."""

    SINGLE_HEAD = "single_head"
    HEAD_PLUS_SKIPPERS = "head_plus_skippers"
    """A heavy head followed by lighter jobs whose model is already resident (line-skip candidates)."""
    MULTI_MODEL_INTERLEAVE = "multi_model_interleave"
    """Alternating models, so residency must rotate to drain the queue."""
    SAME_MODEL_BURST = "same_model_burst"
    """Several jobs for the head's model, all riding one residency."""
    AT_DEPTH = "at_depth"
    """The queue at its configured depth, so no further pop room exists."""


class _GovernorState(Enum):
    """The whole-card churn governors' state when the row starts, driven through the real ledger."""

    FRESH = "fresh"
    GRACE_EXHAUSTED = "grace_exhausted"
    """The card's rolling grace allowance is spent, so a new establishment is refused for its dwell."""
    ESTABLISH_RATE_EXCEEDED = "rate_exceeded"
    """The per-card establishment allowance is spent, so a new establishment is refused for its window."""


class _MidSequenceEvent(Enum):
    """A disturbance injected part-way through the row's run."""

    NONE = "none"
    TARGET_DEATH_RESPAWN = "target_death"
    """The lane holding (or loading) the head's model dies and is replaced by a fresh empty lane."""
    EXTERNAL_RECLAIM = "external_reclaim"
    """An idle sibling's resident model is evicted by an actor other than this scheduler."""


class _ClaimScenario(Enum):
    """What a row asserts about the residency's claim over the worker's pop offer.

    A residency governs the card; the claim is the same commitment applied to intake, so that foreign work
    stops arriving to push the resident weights back out. Each value names one end of that arrangement: the
    burst it exists to serve, and the three ways it gives the pool back.
    """

    NONE = "none"
    """The row says nothing about intake; the claim is exercised only as far as the other properties reach."""
    SERVES_THE_BURST = "serves_the_burst"
    """Work for the resident model keeps arriving and is served, while foreign work is not asked for."""
    CAP_RETURNS_THE_POOL = "cap_returns_the_pool"
    """The maximum hold elapses over a still-wanted residency, returning the full offer and draining the
    foreign work the claim had been holding back."""
    EMPTY_POPS_RELEASE = "empty_pops_release"
    """The resident model's demand dries up, so the claim releases on that evidence well inside its cap."""
    FOREIGN_QUEUED_FIRST = "foreign_queued_first"
    """Foreign jobs accepted before the residency existed wait for the claim to end, then drain."""


class _Expected(Enum):
    """What the row asserts about the head."""

    DISPATCHED = "dispatched"
    """The head reaches sampling within the row's tick bound."""
    UNSERVABLE_HELD = "unservable_held"
    """The head's model prices above the card's achievable ceiling, so the worker holds the model off its
    offer and hands the queued job back for reissue rather than keeping it. The row asserts that disclosed,
    bounded exit: the model is on the ceiling hold and the job did not linger in the queue."""


# --------------------------------------------------------------------------------------------------------
# The world: a scheduler driven over a card whose free VRAM follows from what is resident on it
# --------------------------------------------------------------------------------------------------------

_DEATH_TICK = 3
"""The tick a mid-sequence disturbance fires on: late enough that the row has committed to a plan, early
enough that the remaining ticks can still prove recovery within the bound."""


class _DispatchWorld:
    """Drives a real scheduler over a modelled card, advancing children by hand one tick at a time.

    The card's free VRAM is derived, never dictated: it is the total less every live context and every
    checkpoint committed to VRAM. A dispatch therefore costs the card exactly what its model weighs and an
    unload gives it back, so the reclaim actuations the scheduler orders have a real effect on the next
    tick's admission arithmetic. A staged load costs nothing until the dispatch that commits it, which is the
    gap the admitted-but-unmaterialised planned overlay covers.
    """

    def __init__(
        self,
        *,
        card: _CardClass,
        lane_count: int,
        max_threads: int,
        queue_depth: int,
        whole_card_enabled: bool = True,
        cooldown_seconds: int = 0,
        max_hold_seconds: int = 180,
        tick_seconds: float = _TICK_SECONDS,
    ) -> None:
        """Build the process pool, the model map, and the scheduler for one row.

        Args:
            card: The device-free profile the row runs on.
            lane_count: How many inference lanes the pool holds.
            max_threads: The concurrent-sampling cap (the ``max_threads`` config axis).
            queue_depth: The configured queue size, so an at-depth row can express a full queue.
            whole_card_enabled: Whether preventative whole-card exclusive residency is on.
            cooldown_seconds: How long a drained residency is retained for a follow-on heavy job. Raised by
                the rows that need the residency to still be standing when they make their assertion.
            max_hold_seconds: The operator ceiling on one residency episode, which is also what bounds its
                claim over the offer.
            tick_seconds: How much of the world's clock one tick advances. The default reaches the governance
                windows these rows turn on; a caller whose scenarios turn on the scheduler's short budgets
                (the affinity line-skip window has a fifteen-second floor) passes a shorter tick, so those
                budgets are sampled several times rather than stepped over in one advance.
        """
        self.card = card
        self.tick_seconds = tick_seconds
        self.tick = 0
        self.now = 10_000.0
        """The world's clock, shared with the tracker and the scheduler so every window they gate on is
        measured on this timeline rather than on the seconds a test run actually spends."""
        self._resident_mb: dict[int, float] = {}
        """Per-lane weights committed to VRAM. A staged load is not here: it costs the card nothing until
        the job that needs it is dispatched, which is what the admitted-but-unmaterialised planned overlay
        exists to cover."""
        self._staged_mb: dict[int, float] = {}
        """Per-lane weights held in the child's RAM cache, awaiting the dispatch that commits them."""
        self._loading: dict[int, tuple[str, float]] = {}
        self.reclaim_commands = 0
        self.snapshot: MeasuredVramSnapshot | None = None
        """The most recent cycle-frozen device measurement, the surface a row reads obligations back from."""

        processes: dict[int, HordeProcessInfo] = {}
        for lane_id in range(lane_count):
            lane = make_mock_process_info(lane_id, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
            processes[lane_id] = lane
        self._process_map = ProcessMap(processes)
        self._model_map = HordeModelMap(root={})
        self._job_tracker = JobTracker(clock=lambda: self.now)
        self._reserve_ledger = CommittedReserveLedger()

        reference: dict[str, object] = {
            model.name: make_mock_model_reference_record(model.name, baseline=model.baseline)
            for model in _MODEL_CLASSES
        }
        bridge_data = make_mock_bridge_data(
            max_threads=max_threads,
            queue_size=queue_depth,
            enable_vram_budget=True,
            whole_card_exclusive_residency=whole_card_enabled,
            whole_card_residency_safety_off_gpu=False,
            safety_on_gpu=False,
            vram_reserve_mb=0,
            ram_reserve_mb=8192.0,
            vram_per_process_overhead_mb=_FIRST_CONTEXT_MB,
            whole_card_residency_cooldown_seconds=cooldown_seconds,
            whole_card_residency_max_hold_seconds=max_hold_seconds,
            image_models_to_load=[model.name for model in _MODEL_CLASSES],
        )
        self.offers: dict[int, frozenset[str]] = {}
        """What the worker would have advertised at the end of each tick, through the real claim seam."""
        self.claim_ticks: list[int] = []
        """The ticks a whole-card residency was claiming the offer, so a row can order events against it."""
        self.claim_expires_at = 0.0
        """When the standing claim's maximum hold runs out, as the claim itself reported it."""
        self.claim_released_at = 0.0
        """The world's clock when the claim first stopped standing; 0.0 while one has never ended."""
        self._lane_ceiling = lane_count
        self._lifecycle = _make_mock_lifecycle(self)
        self._scheduler = InferenceScheduler(
            state=WorkerState(),
            process_map=self._process_map,
            horde_model_map=self._model_map,
            job_tracker=self._job_tracker,
            process_lifecycle=self._lifecycle,
            runtime_config=make_test_runtime_config(bridge_data=bridge_data),
            model_metadata=make_test_model_metadata(reference),
            max_concurrent_inference_processes=max_threads,
            max_inference_processes=lane_count,
            lru=LRUCache(max(2, lane_count)),
            reserve_ledger=self._reserve_ledger,
            clock=lambda: self.now,
        )
        self._scheduler.set_device_free_mb_provider(lambda _device_index: self.device_free_mb())
        # The rows vary VRAM, never host RAM: pinning an ample reading keeps the RAM admission gates out of
        # the variation and stops a row's outcome depending on how much memory the machine running it has.
        self._scheduler.set_available_ram_mb_provider(lambda: _AMPLE_RAM_MB)
        self._sync_reported_vram()

        self.first_dispatch: dict[str, int] = {}
        self._dispatched_at: dict[str, int] = {}
        self._lane_of: dict[str, int] = {}

    # -- card model ---------------------------------------------------------------------------------------

    def _context_charge_mb(self) -> float:
        """The card's total context cost: the one-time runtime plus one context for every live lane."""
        lanes = max(1, len(self._process_map))
        return _FIRST_CONTEXT_MB + _MARGINAL_CONTEXT_MB * (lanes - 1)

    def device_free_mb(self) -> float:
        """The truthful device-free reading: the card total less its live contexts and its committed weights."""
        held = sum(self._resident_mb.values())
        return max(0.0, self.card.total_mb - self._context_charge_mb() - held)

    def _sync_reported_vram(self) -> None:
        """Publish the derived card state through the children's VRAM reports, as a live worker would."""
        used_mb = self.card.total_mb - self.device_free_mb()
        for lane in self._process_map.values():
            lane.total_vram_mb = int(self.card.total_mb)
            lane.vram_usage_mb = int(used_mb)
            lane.process_reserved_mb = int(self._resident_mb.get(lane.process_id, 0.0))

    # -- seeding ------------------------------------------------------------------------------------------

    def seed_resident(self, lane_id: int, model: _ModelClass, *, in_vram: bool) -> None:
        """Place ``model`` on ``lane_id`` as a resident (VRAM) or staged (RAM) copy."""
        lane = self._process_map[lane_id]
        lane.loaded_horde_model_name = model.name
        lane.last_process_state = HordeProcessState.PRELOADED_MODEL
        self._model_map.update_entry(
            model.name,
            load_state=ModelLoadState.LOADED_IN_VRAM if in_vram else ModelLoadState.LOADED_IN_RAM,
            process_id=lane_id,
        )
        if in_vram:
            self._resident_mb[lane_id] = model.weights_mb
        else:
            self._staged_mb[lane_id] = model.weights_mb
        self._sync_reported_vram()

    async def pop(self, job: ImageGenerateJobPopResponse) -> None:
        """Record a popped job, exactly as the pop path hands one to the tracker."""
        await track_popped_job_async(self._job_tracker, job, time_popped=self.now)

    # -- intake -------------------------------------------------------------------------------------------

    def advertised_models(self) -> frozenset[str]:
        """The models the worker would ask the horde for right now.

        Only the residency's claim over the offer is modelled here (through the same pure stage the popper
        runs, over the same scheduler accessor it reads), because that is the only offer shaping these rows
        vary. The other narrowings are the pop suite's subject.
        """
        return offer_under_pop_claim(
            frozenset(model.name for model in _MODEL_CLASSES),
            claim=self._scheduler.whole_card_pop_claim(),
        )

    async def offer_job(self, job: ImageGenerateJobPopResponse) -> bool:
        """Take a job only when its model is one the worker is currently asking for.

        Stands in for the horde answering a pop with a job: work the worker did not advertise never arrives,
        so a job for an unadvertised model is refused and the queue never sees it.
        """
        if job.model not in self.advertised_models():
            return False
        await self.pop(job)
        return True

    def report_empty_pop(self) -> None:
        """Report that a pop taken under the standing claim came back with no work."""
        self._scheduler.note_whole_card_pop_outcome(served=False)

    # -- child-side effects -------------------------------------------------------------------------------

    def _apply_control_flags(self) -> None:
        """Honour the commands the scheduler sent last tick: an unload gives the card back its VRAM."""
        for lane in self._process_map.values():
            flag = lane.last_control_flag
            if flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                self.reclaim_commands += 1
                name = lane.loaded_horde_model_name
                self._resident_mb.pop(lane.process_id, None)
                self._staged_mb.pop(lane.process_id, None)
                self._loading.pop(lane.process_id, None)
                lane.loaded_horde_model_name = None
                lane.last_control_flag = None
                lane.last_process_state = HordeProcessState.WAITING_FOR_JOB
                if name is not None:
                    entry = self._model_map.root.get(name)
                    if entry is not None and entry.process_id == lane.process_id:
                        self._model_map.root.pop(name, None)
        self._sync_reported_vram()

    def _begin_started_preloads(self) -> None:
        """Start the load of any model the scheduler has just told a lane to bring in."""
        for name, info in list(self._model_map.root.items()):
            if info.horde_model_load_state != ModelLoadState.LOADING or info.process_id is None:
                continue
            if info.process_id in self._loading or info.process_id in self._staged_mb:
                continue
            model = _model_by_name(name)
            if model is None:
                continue
            self._loading[info.process_id] = (name, model.weights_mb)
            lane = self._process_map.get(info.process_id)
            if lane is not None and lane.last_process_state != HordeProcessState.PRELOADING_MODEL:
                lane.last_process_state = HordeProcessState.PRELOADING_MODEL
        self._sync_reported_vram()

    def _materialise_preloads(self) -> None:
        """Complete last tick's loads: the weights are staged and the lane can accept a job."""
        for lane_id, (name, weights_mb) in list(self._loading.items()):
            lane = self._process_map.get(lane_id)
            if lane is None:
                self._loading.pop(lane_id, None)
                continue
            self._loading.pop(lane_id, None)
            self._staged_mb[lane_id] = weights_mb
            lane.loaded_horde_model_name = name
            lane.last_process_state = HordeProcessState.PRELOADED_MODEL
            lane.last_control_flag = None
            self._model_map.update_entry(name, load_state=ModelLoadState.LOADED_IN_RAM, process_id=lane_id)
        self._sync_reported_vram()

    async def _complete_finished_samplers(self) -> None:
        """Return each lane that sampled on an earlier tick to an idle, still-resident state."""
        for job in list(self._job_tracker.jobs_in_progress):
            job_id = job.id_
            if job_id is None or self._dispatched_at.get(str(job_id), self.tick) >= self.tick:
                continue
            job_info = HordeJobInfo(
                sdk_api_job_info=job,
                job_image_results=[HordeImageResult(image_bytes=b"raw")],
                state=GENERATION_STATE.ok,
                censored=False,
                time_popped=self.now,
            )
            await self._job_tracker.queue_for_safety(job_info)
            self._dispatched_at.pop(str(job_id), None)
            lane_id = self._lane_of.pop(str(job_id), None)
            lane = self._process_map.get(lane_id) if lane_id is not None else None
            if lane is not None and lane.loaded_horde_model_name is not None:
                lane.last_process_state = HordeProcessState.PRELOADED_MODEL
                # The weights stay resident on the freed lane, so the next same-model job needs no reload.
                self._model_map.update_entry(
                    lane.loaded_horde_model_name,
                    load_state=ModelLoadState.LOADED_IN_VRAM,
                    process_id=lane.process_id,
                )

    async def _drain_safety(self) -> None:
        """Walk each finished job through safety and submit so it leaves the tracker as a completed job.

        Draining is what keeps a finished job from reading as a safety backlog, which the residency
        convergence treats as work not to disturb; leaving it queued would make the queue's own completions
        look like a reason to hold the card.
        """
        for job_info in list(self._job_tracker.jobs_pending_safety_check):
            await self._job_tracker.begin_safety_check(job_info)
            await self._job_tracker.queue_for_submit(job_info)
            await self._job_tracker.finalize_submitted(job_info)

    async def _dispatch_until_full(self) -> None:
        """Dispatch onto free lanes, recording the tick each job first reached sampling."""
        for _attempt in range(max(1, int(self._scheduler._runtime_config.bridge_data.max_threads))):
            before = {str(job.id_) for job in self._job_tracker.jobs_in_progress}
            started = await self._scheduler.start_inference()
            newly = [job for job in self._job_tracker.jobs_in_progress if str(job.id_) not in before]
            if not started:
                assert newly == [], "start_inference declined yet a job entered progress"
                break
            assert len(newly) == 1, "a successful dispatch must admit exactly one job"
            admitted = newly[0]
            job_id = str(admitted.id_)
            lanes = [
                lane.process_id
                for lane in self._process_map.values()
                if lane.loaded_horde_model_name == admitted.model
                and lane.last_control_flag == HordeControlFlag.START_INFERENCE
            ]
            assert lanes, "an admitted job must have been dispatched onto a lane holding its model"
            self._dispatched_at[job_id] = self.tick
            self._lane_of[job_id] = lanes[0]
            self.first_dispatch.setdefault(job_id, self.tick)
            # The child reports the model IN_USE and its slot busy the moment it starts sampling: the first
            # takes the load out of the in-flight-admitted set (releasing its planned charge), the second
            # keeps the sampling lane out of the idle pool a shrink or a second dispatch could take.
            self._process_map[lanes[0]].last_process_state = HordeProcessState.INFERENCE_STARTING
            if admitted.model is not None:
                self._model_map.update_entry(admitted.model, load_state=ModelLoadState.IN_USE, process_id=lanes[0])
            # Dispatch is the moment staged weights commit to VRAM, so this is where the card is charged.
            staged_mb = self._staged_mb.pop(lanes[0], None)
            if staged_mb is not None:
                self._resident_mb[lanes[0]] = staged_mb
                self._sync_reported_vram()

    def scale_inference_processes(
        self,
        target_count: int,
        *,
        device_index: int | None = None,
        protected_model: str | None = None,
        pressure_shortfall_mb: float | None = None,
        spared_process_id: int | None = None,
    ) -> int:
        """Grow or shrink the lane pool toward ``target_count``, returning the count after scaling.

        Mirrors the lifecycle's contract at the grain these rows turn on: a shrink ends idle lanes only (a
        busy lane is never killed, so the count may not reach the target in one call) and gives their VRAM
        back to the card; a residency's shrink names its holder as ``protected_model`` and spares it, and the
        slot the caller is about to load onto is spared by id through ``spared_process_id`` (a head not staged
        anywhere carries its model on no lane, so the name-based protection cannot reach its target). Growth
        restores empty lanes up to the pool's provisioned ceiling.
        """
        del device_index, pressure_shortfall_mb
        while len(self._process_map) > target_count:
            victim = next(
                (
                    lane
                    for lane in self._process_map.values()
                    if lane.can_accept_job()
                    and lane.process_id != spared_process_id
                    and (protected_model is None or lane.loaded_horde_model_name != protected_model)
                ),
                None,
            )
            if victim is None:
                break
            self._retire_lane(victim.process_id)
        while len(self._process_map) < min(target_count, self._lane_ceiling):
            lane_id = max(self._process_map, default=-1) + 1
            self._process_map[lane_id] = make_mock_process_info(
                lane_id,
                model_name=None,
                state=HordeProcessState.WAITING_FOR_JOB,
            )
        self._sync_reported_vram()
        return len(self._process_map)

    def _retire_lane(self, lane_id: int) -> None:
        """Remove a lane from the pool, releasing its context, its resident weights, and its map entry."""
        lane = self._process_map.get(lane_id)
        if lane is not None:
            self._process_map.retire_process(lane, reason="whole-card residency teardown")
        self._resident_mb.pop(lane_id, None)
        self._staged_mb.pop(lane_id, None)
        self._loading.pop(lane_id, None)
        if lane is None:
            return
        name = lane.loaded_horde_model_name
        if name is not None:
            entry = self._model_map.root.get(name)
            if entry is not None and entry.process_id == lane_id:
                self._model_map.root.pop(name, None)

    # -- disturbances -------------------------------------------------------------------------------------

    def kill_lane_holding(self, model: _ModelClass) -> None:
        """Kill the lane holding (or loading) ``model`` and replace it with a fresh empty lane.

        Mirrors the lifecycle's replacement of a dead inference process: the dead lane leaves the map (taking
        its model-map entry and its VRAM with it) and a new, empty lane of the same id takes its place.
        """
        victim: int | None = None
        for lane in self._process_map.values():
            if lane.loaded_horde_model_name == model.name:
                victim = lane.process_id
                break
        if victim is None:
            for lane_id, (name, _weights) in self._loading.items():
                if name == model.name:
                    victim = lane_id
                    break
        if victim is None:
            return
        self._resident_mb.pop(victim, None)
        self._staged_mb.pop(victim, None)
        self._loading.pop(victim, None)
        entry = self._model_map.root.get(model.name)
        if entry is not None and entry.process_id == victim:
            self._model_map.root.pop(model.name, None)
        replacement = make_mock_process_info(victim, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        self._process_map[victim] = replacement
        self._sync_reported_vram()

    def evict_idle_resident_sibling(self, *, except_model: _ModelClass) -> None:
        """Evict one idle resident model other than ``except_model``, as an outside reclaim actor would."""
        for lane in self._process_map.values():
            name = lane.loaded_horde_model_name
            if (
                name is None
                or name == except_model.name
                or lane.last_process_state == HordeProcessState.INFERENCE_STARTING
            ):
                continue
            self._resident_mb.pop(lane.process_id, None)
            self._staged_mb.pop(lane.process_id, None)
            lane.loaded_horde_model_name = None
            lane.last_process_state = HordeProcessState.WAITING_FOR_JOB
            entry = self._model_map.root.get(name)
            if entry is not None and entry.process_id == lane.process_id:
                self._model_map.root.pop(name, None)
            self._sync_reported_vram()
            return

    # -- the loop -----------------------------------------------------------------------------------------

    def _begin_arbiter_cycle(self) -> None:
        """Freeze this tick's device measurement, exactly as the control loop does before governance.

        Building the snapshot is also what reconciles the admission-reservation flows by omission, so a
        planned charge whose load has materialised (or whose target has gone) is released here rather than
        only when the next admission happens to ask. Driving it every tick is what makes the end-of-row
        obligation readback a statement about the running worker instead of about this harness.
        """
        self.snapshot = self._scheduler.build_vram_arbiter_snapshot(
            device_free_mb_by_device={0: self.device_free_mb()},
        )

    def _discharge_context_reductions(self) -> None:
        """Grow the pool back after a pressure reduction, the obligation the scheduler records but never closes.

        A head that does not fit may collapse the card's live inference contexts to make room; the scheduler
        takes that reduction and records the restore obligation, but discharging it belongs to the control
        loop, which calls ``restore_live_contexts`` once the card recovers. Driving only the scheduler's half
        would leave a card permanently one lane short after any pressure episode, and a pool shrunk to the
        single lane that holds an idle resident model cannot reclaim it: that lane is then the next head's own
        preload target, which every eviction path deliberately spares. The actuator is the production one, so
        it stands down under a held whole-card residency (whose own restore owns the regrowth) and no-ops once
        the pool is back at its configured size. Both of the control loop's restore paths hold while the head
        of queue is parked, because regrowing underneath a parked head re-adds the context footprint the head
        cannot be admitted over and the pair then oscillates at one cold start per tick; that gate is honoured
        here through the same predicate they read.
        """
        if self._scheduler.head_of_queue_is_parked():
            return
        self._scheduler.restore_live_contexts(None)

    async def step(self) -> None:
        """Advance one scheduling tick, in the control loop's order."""
        self.tick += 1
        self.now += self.tick_seconds
        self._apply_control_flags()
        self._materialise_preloads()
        await self._complete_finished_samplers()
        await self._drain_safety()
        self._begin_arbiter_cycle()
        self._scheduler.run_governance_tick()
        self._discharge_context_reductions()
        self._scheduler.preload_models()
        self._begin_started_preloads()
        await self._dispatch_until_full()
        self.offers[self.tick] = self.advertised_models()
        claim = self._scheduler.whole_card_pop_claim()
        if claim is not None:
            self.claim_ticks.append(self.tick)
            self.claim_expires_at = claim.expires_at
        elif self.claim_ticks and self.claim_released_at == 0.0:
            self.claim_released_at = self.now

    async def run(self, ticks: int) -> None:
        """Advance ``ticks`` scheduling ticks."""
        for _ in range(ticks):
            await self.step()

    # -- readback -----------------------------------------------------------------------------------------

    @property
    def scheduler(self) -> InferenceScheduler:
        """The scheduler under test."""
        return self._scheduler

    @property
    def job_tracker(self) -> JobTracker:
        """The tracker the scheduler shares."""
        return self._job_tracker

    @property
    def reserve_ledger(self) -> CommittedReserveLedger:
        """The shared committed/planned reserve ledger the scheduler books admissions against."""
        return self._reserve_ledger

    @property
    def lifecycle(self) -> Mock:
        """The stand-in lifecycle manager, so a row can read back the pauses a residency took out."""
        return self._lifecycle

    def planned_overlay_mb(self) -> float:
        """The planned (admitted but unmaterialised) VRAM the last frozen measurement still carries.

        Read from the cycle snapshot rather than the raw ledger so the figure is the one admission actually
        prices against: the ledger's anchors are consumed against each target's live allocator reservation
        during the snapshot build, which is exactly the release path an outlived charge would defeat.
        """
        if self.snapshot is None:
            return 0.0
        return sum(state.planned_unmaterialized_mb for state in self.snapshot.devices.values())

    def dispatch_tick(self, job: ImageGenerateJobPopResponse) -> int | None:
        """The tick ``job`` first reached sampling, or None if it never did."""
        return self.first_dispatch.get(str(job.id_))

    def stage(self, job: ImageGenerateJobPopResponse) -> JobStage | None:
        """The tracker's current stage for ``job``."""
        assert job.id_ is not None
        return self._job_tracker.get_stage(job.id_)

    def state_dump(self) -> str:
        """A one-line description of the card, the pool, and the queue, for a failing row's message."""
        lanes = ", ".join(
            f"{lane.process_id}:{lane.loaded_horde_model_name or '-'}/{lane.last_process_state.name}"
            for lane in self._process_map.values()
        )
        pending = ", ".join(str(job.model) for job in self._job_tracker.jobs_pending_inference)
        return (
            f"tick={self.tick} free={self.device_free_mb():.0f}/{self.card.total_mb:.0f}MB "
            f"lanes=[{lanes}] pending=[{pending}] in_progress={len(self._job_tracker.jobs_in_progress)} "
            f"residency_active={self._scheduler.is_whole_card_residency_active()} "
            f"planned={self.planned_overlay_mb():.0f}MB "
            f"reclaims={self.reclaim_commands} defer_reason={self._scheduler._last_budget_defer_reason!r}"
        )


def _make_mock_lifecycle(world: _DispatchWorld) -> Mock:
    """A lifecycle stand-in whose every predicate the scheduler reads answers concretely.

    A bare Mock hands back truthy Mocks for unset predicates, which silently arms gates a row never intended;
    each flag the scheduler consults is therefore pinned to its inert value here. ``scale_inference_processes``
    is wired to the world's pool so a whole-card residency's teardown genuinely reduces the live contexts and
    the residency can converge, which is the difference between exercising the residency path and watching it
    spin against a lifecycle that never acts.
    """
    lifecycle = Mock()
    lifecycle.scale_inference_processes = world.scale_inference_processes
    lifecycle.get_processes_with_model_for_queued_job = Mock(return_value=[])
    lifecycle.is_model_load_quarantined = Mock(return_value=False)
    lifecycle.is_safety_gpu_paused = False
    lifecycle.is_post_process_gpu_paused = False
    lifecycle.post_process_processes_should_be_replaced = False
    lifecycle.post_process_lane_enabled = Mock(return_value=False)
    lifecycle.component_lane_enabled = Mock(return_value=False)
    lifecycle.vae_lane_enabled = Mock(return_value=False)
    lifecycle.has_pending_inference_starts = Mock(return_value=False)
    lifecycle.pending_gpu_starts_backing_off = Mock(return_value=False)
    lifecycle.has_pending_safety_starts = Mock(return_value=False)
    lifecycle.quarantined_inference_slots = frozenset()
    lifecycle.safety_pool_failing = False
    lifecycle.safety_pool_start_failing = False
    return lifecycle


def _model_by_name(name: str) -> _ModelClass | None:
    """Resolve a model class from its reference name."""
    for model in _MODEL_CLASSES:
        if model.name == name:
            return model
    return None


# --------------------------------------------------------------------------------------------------------
# The scenario table
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    """One point in the bounded-dispatch matrix.

    Attributes:
        label: The parametrized id.
        card: The device-free profile.
        head_model: The head-of-queue job's model class.
        residency: Where the head's checkpoint starts out.
        queue: The queue structure behind the head.
        governor: The whole-card churn governors' starting state.
        max_threads: The concurrent-sampling cap.
        lanes: How many inference lanes the pool holds.
        event: A disturbance injected part-way through the run.
        expected: What the row asserts about the head.
        tick_bound: The number of scheduling ticks the head must reach dispatch within.
        run_ticks: How many ticks to drive in total (at least the bound, plus drain ticks for the
            obligation-closure assertions).
        drain_all: Whether every queued job (not only the head) must drain within ``run_ticks``.
        claim: What the row asserts about the residency's claim over the pop offer.
        cooldown_seconds: How long a drained residency is retained for a follow-on heavy job.
        max_hold_seconds: The ceiling on one residency episode, which also bounds its claim.
    """

    label: str
    card: _CardClass
    head_model: _ModelClass
    residency: _Residency
    queue: _QueueShape
    governor: _GovernorState = _GovernorState.FRESH
    max_threads: int = 1
    lanes: int = 2
    event: _MidSequenceEvent = _MidSequenceEvent.NONE
    expected: _Expected = _Expected.DISPATCHED
    tick_bound: int = 6
    run_ticks: int = 14
    drain_all: bool = True
    whole_card_enabled: bool = True
    claim: _ClaimScenario = _ClaimScenario.NONE
    cooldown_seconds: int = 0
    max_hold_seconds: int = 180


_PRUNED_COMBINATIONS: tuple[tuple[str, str], ...] = (
    (
        "8 GB card x SDXL or flux head, beyond one representative cell each",
        "neither model's sampling peak fits an 8 GB card however much is reclaimed, so every such cell takes "
        "the same ceiling-hold exit; one representative cell of each is driven (plus one with servable work "
        "queued behind it) and the rest are dropped as repeats of a single verdict.",
    ),
    (
        "whole-card governor states x non-EXTRA_LARGE head models",
        "the grace budget and the establishment rate limiter gate whole-card residency only; an SD15 or SDXL "
        "head never reaches that path, so charging its ledger varies nothing about that head's admission.",
    ),
    (
        "RESIDENT_ON_SIBLING x SINGLE_HEAD",
        "a sibling lane is only meaningful when a second job occupies it; with a single head the shape "
        "collapses onto RESIDENT_IDLE_TARGET.",
    ),
    (
        "TARGET_DEATH_RESPAWN x RESIDENT_ON_SIBLING",
        "the disturbance is defined against the lane holding the head's own copy; with the copy held by a busy "
        "sibling the kill is a live-job kill, which is the recovery suite's subject, not bounded dispatch.",
    ),
    (
        "max_threads=2 x SINGLE_HEAD",
        "a second sampling slot cannot change a one-job queue's outcome; the concurrency axis is varied only "
        "against queue shapes that hold more than one dispatchable job.",
    ),
    (
        "AT_DEPTH x max_threads=2 x every card class",
        "queue depth interacts with the pop gate, not with dispatch capacity; one representative at-depth cell "
        "per card class is driven and the concurrency cross-product is dropped as redundant.",
    ),
    (
        "EXTERNAL_RECLAIM x rows whose card is not under pressure",
        "an eviction that frees room nothing was waiting for varies nothing; the event is applied only where "
        "the head's admission actually turns on the freed VRAM.",
    ),
    (
        "pop-claim scenarios x every card, model, governor and event value",
        "the claim is a property of a held residency and of the clock, not of the card it is held on: the "
        "16 GB flux cell is the one where a residency is genuinely warranted, so the four claim scenarios are "
        "driven there and the cross-product with hardware that would either never establish a residency or "
        "never need one is dropped.",
    ),
    (
        "24 GB card x SD15 head x every governor and event value",
        "the roomy card with the smallest model admits on the first tick under every one of them; one "
        "representative cell is kept and the rest dropped as trivially-passing.",
    ),
)
"""Combinations enumerated by the axes but deliberately not driven, each with why. Listed so the table's
coverage is explicit: nothing is truncated silently."""


def _rows() -> tuple[_Row, ...]:
    """The driven matrix."""
    return (
        # -- Card class x model class, single head, nothing resident: the base servability sweep. ------------
        _Row("sd15_absent_8gb", _CARD_8GB, _SD15, _Residency.ABSENT, _QueueShape.SINGLE_HEAD),
        _Row("sd15_absent_16gb", _CARD_16GB, _SD15, _Residency.ABSENT, _QueueShape.SINGLE_HEAD),
        _Row("sd15_absent_24gb", _CARD_24GB, _SD15, _Residency.ABSENT, _QueueShape.SINGLE_HEAD),
        # SDXL on an 8 GB card is servable: its priced peak comes from the sampling estimate rather than an
        # unmeasured recommendation floor, and the modest remaining shortfall against the live reading is
        # decided by a measured attempt instead of a static decline. The card provably holds the model, so
        # the worker serves it.
        _Row(
            "sdxl_absent_8gb",
            _CARD_8GB,
            _SDXL,
            _Residency.ABSENT,
            _QueueShape.SINGLE_HEAD,
            tick_bound=8,
        ),
        _Row("sdxl_absent_16gb", _CARD_16GB, _SDXL, _Residency.ABSENT, _QueueShape.SINGLE_HEAD),
        _Row("sdxl_absent_24gb", _CARD_24GB, _SDXL, _Residency.ABSENT, _QueueShape.SINGLE_HEAD),
        _Row("flux_absent_16gb", _CARD_16GB, _FLUX, _Residency.ABSENT, _QueueShape.SINGLE_HEAD, tick_bound=8),
        _Row("flux_absent_24gb", _CARD_24GB, _FLUX, _Residency.ABSENT, _QueueShape.SINGLE_HEAD, tick_bound=8),
        # A head that cannot fit even an emptied card: the discriminating negative row.
        _Row(
            "flux_absent_8gb_unservable",
            _CARD_8GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SINGLE_HEAD,
            expected=_Expected.UNSERVABLE_HELD,
            drain_all=False,
            run_ticks=10,
        ),
        # -- Residency axis. ---------------------------------------------------------------------------------
        _Row("sd15_resident_idle_8gb", _CARD_8GB, _SD15, _Residency.RESIDENT_IDLE_TARGET, _QueueShape.SINGLE_HEAD),
        _Row("sdxl_resident_idle_16gb", _CARD_16GB, _SDXL, _Residency.RESIDENT_IDLE_TARGET, _QueueShape.SINGLE_HEAD),
        _Row("flux_resident_idle_24gb", _CARD_24GB, _FLUX, _Residency.RESIDENT_IDLE_TARGET, _QueueShape.SINGLE_HEAD),
        # A resident whole-card head on a card that demands exclusive residency. Its weights are already on
        # the device, so the residency's live-fit question is answered before it is asked and the head is not
        # left to the drain-settle backstop.
        _Row(
            "flux_resident_idle_16gb",
            _CARD_16GB,
            _FLUX,
            _Residency.RESIDENT_IDLE_TARGET,
            _QueueShape.SINGLE_HEAD,
            tick_bound=8,
        ),
        _Row("sdxl_ram_staged_16gb", _CARD_16GB, _SDXL, _Residency.RAM_STAGED, _QueueShape.SINGLE_HEAD),
        _Row("flux_ram_staged_24gb", _CARD_24GB, _FLUX, _Residency.RAM_STAGED, _QueueShape.SINGLE_HEAD),
        _Row(
            "sdxl_resident_on_busy_sibling_16gb",
            _CARD_16GB,
            _SDXL,
            _Residency.RESIDENT_ON_SIBLING,
            _QueueShape.SAME_MODEL_BURST,
            tick_bound=8,
        ),
        # -- Queue structure. --------------------------------------------------------------------------------
        _Row(
            "flux_head_plus_skippers_16gb",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.HEAD_PLUS_SKIPPERS,
            tick_bound=10,
            run_ticks=22,
        ),
        # An SDXL head with lighter work behind it on the smallest card that holds it: the head dispatches
        # via measured admission and the skippers drain with it.
        _Row(
            "sdxl_head_plus_skippers_8gb",
            _CARD_8GB,
            _SDXL,
            _Residency.ABSENT,
            _QueueShape.HEAD_PLUS_SKIPPERS,
            tick_bound=10,
            run_ticks=24,
        ),
        _Row(
            "sdxl_multi_model_interleave_16gb",
            _CARD_16GB,
            _SDXL,
            _Residency.ABSENT,
            _QueueShape.MULTI_MODEL_INTERLEAVE,
            tick_bound=8,
            run_ticks=26,
        ),
        _Row(
            "sd15_multi_model_interleave_8gb",
            _CARD_8GB,
            _SD15,
            _Residency.ABSENT,
            _QueueShape.MULTI_MODEL_INTERLEAVE,
            tick_bound=8,
            run_ticks=30,
        ),
        _Row(
            "flux_same_model_burst_24gb",
            _CARD_24GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SAME_MODEL_BURST,
            tick_bound=8,
            run_ticks=22,
        ),
        _Row(
            "flux_same_model_burst_16gb",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SAME_MODEL_BURST,
            tick_bound=8,
            run_ticks=22,
        ),
        _Row(
            "sdxl_queue_at_depth_16gb",
            _CARD_16GB,
            _SDXL,
            _Residency.ABSENT,
            _QueueShape.AT_DEPTH,
            tick_bound=8,
            run_ticks=24,
        ),
        _Row(
            "sd15_queue_at_depth_8gb",
            _CARD_8GB,
            _SD15,
            _Residency.ABSENT,
            _QueueShape.AT_DEPTH,
            tick_bound=8,
            run_ticks=26,
        ),
        # -- Concurrency. ------------------------------------------------------------------------------------
        _Row(
            "sdxl_interleave_two_threads_24gb",
            _CARD_24GB,
            _SDXL,
            _Residency.ABSENT,
            _QueueShape.MULTI_MODEL_INTERLEAVE,
            max_threads=2,
            lanes=3,
            tick_bound=8,
            run_ticks=24,
        ),
        _Row(
            "sd15_burst_two_threads_16gb",
            _CARD_16GB,
            _SD15,
            _Residency.ABSENT,
            _QueueShape.SAME_MODEL_BURST,
            max_threads=2,
            lanes=3,
            tick_bound=6,
            run_ticks=20,
        ),
        _Row(
            "flux_skippers_two_threads_24gb",
            _CARD_24GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.HEAD_PLUS_SKIPPERS,
            max_threads=2,
            lanes=3,
            tick_bound=10,
            run_ticks=24,
        ),
        # -- Governor / budget state (whole-card head only). -------------------------------------------------
        _Row(
            "flux_grace_exhausted_16gb",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SINGLE_HEAD,
            governor=_GovernorState.GRACE_EXHAUSTED,
            tick_bound=10,
            run_ticks=18,
        ),
        _Row(
            "flux_grace_exhausted_24gb",
            _CARD_24GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SINGLE_HEAD,
            governor=_GovernorState.GRACE_EXHAUSTED,
            tick_bound=10,
            run_ticks=18,
        ),
        _Row(
            "flux_rate_exceeded_16gb",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SINGLE_HEAD,
            governor=_GovernorState.ESTABLISH_RATE_EXCEEDED,
            tick_bound=10,
            run_ticks=18,
        ),
        _Row(
            "flux_rate_exceeded_burst_24gb",
            _CARD_24GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SAME_MODEL_BURST,
            governor=_GovernorState.ESTABLISH_RATE_EXCEEDED,
            tick_bound=10,
            run_ticks=24,
        ),
        _Row(
            "flux_grace_exhausted_resident_24gb",
            _CARD_24GB,
            _FLUX,
            _Residency.RESIDENT_IDLE_TARGET,
            _QueueShape.SINGLE_HEAD,
            governor=_GovernorState.GRACE_EXHAUSTED,
            tick_bound=10,
            run_ticks=18,
        ),
        # -- Mid-sequence events. ----------------------------------------------------------------------------
        _Row(
            "sdxl_target_death_respawn_16gb",
            _CARD_16GB,
            _SDXL,
            _Residency.RESIDENT_IDLE_TARGET,
            _QueueShape.SAME_MODEL_BURST,
            event=_MidSequenceEvent.TARGET_DEATH_RESPAWN,
            tick_bound=10,
            run_ticks=24,
        ),
        _Row(
            "flux_target_death_respawn_24gb",
            _CARD_24GB,
            _FLUX,
            _Residency.RESIDENT_IDLE_TARGET,
            _QueueShape.SAME_MODEL_BURST,
            event=_MidSequenceEvent.TARGET_DEATH_RESPAWN,
            tick_bound=12,
            run_ticks=26,
        ),
        _Row(
            "sd15_external_reclaim_8gb",
            _CARD_8GB,
            _SD15,
            _Residency.ABSENT,
            _QueueShape.HEAD_PLUS_SKIPPERS,
            event=_MidSequenceEvent.EXTERNAL_RECLAIM,
            tick_bound=10,
            run_ticks=24,
        ),
        _Row(
            "flux_external_reclaim_16gb",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.HEAD_PLUS_SKIPPERS,
            event=_MidSequenceEvent.EXTERNAL_RECLAIM,
            tick_bound=10,
            run_ticks=24,
        ),
        # -- Whole-card residency disabled: the same heads must still be served. -----------------------------
        _Row(
            "flux_absent_16gb_residency_off",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SINGLE_HEAD,
            whole_card_enabled=False,
            tick_bound=8,
        ),
        # -- The residency's claim over intake. --------------------------------------------------------------
        # Each row raises the cooldown so the residency is still standing where its assertion is made: with
        # the cooldown at zero every residency releases the instant its work drains, which would leave the
        # claim's own ends untested.
        _Row(
            "flux_16gb_claim_serves_the_burst",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SAME_MODEL_BURST,
            tick_bound=8,
            run_ticks=16,
            claim=_ClaimScenario.SERVES_THE_BURST,
            cooldown_seconds=600,
        ),
        _Row(
            "flux_16gb_claim_cap_returns_the_pool",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.HEAD_PLUS_SKIPPERS,
            tick_bound=8,
            run_ticks=16,
            claim=_ClaimScenario.CAP_RETURNS_THE_POOL,
            # The cooldown alone would retain this residency for the whole run, so the pool coming back is
            # attributable to the cap and to nothing else.
            cooldown_seconds=600,
            max_hold_seconds=120,
        ),
        _Row(
            "flux_16gb_claim_empty_pops_release",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SINGLE_HEAD,
            tick_bound=8,
            run_ticks=24,
            claim=_ClaimScenario.EMPTY_POPS_RELEASE,
            # Both clocks are set past the run's own reach so the early release is the only thing that can
            # end the claim inside it; the run is long enough that the cap still terminates the row.
            cooldown_seconds=600,
            max_hold_seconds=600,
        ),
        _Row(
            "flux_16gb_claim_foreign_jobs_queued_first",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.HEAD_PLUS_SKIPPERS,
            tick_bound=8,
            run_ticks=16,
            claim=_ClaimScenario.FOREIGN_QUEUED_FIRST,
        ),
    )


_ROWS = _rows()


_QUEUE_LENGTHS: dict[_QueueShape, int] = {
    _QueueShape.SINGLE_HEAD: 1,
    _QueueShape.HEAD_PLUS_SKIPPERS: 3,
    _QueueShape.MULTI_MODEL_INTERLEAVE: 4,
    _QueueShape.SAME_MODEL_BURST: 3,
    _QueueShape.AT_DEPTH: 4,
}


def _queue_models(row: _Row) -> list[_ModelClass]:
    """The models of the jobs the row queues, head first."""
    head = row.head_model
    light = _SD15 if head is not _SD15 else _SD15_OTHER
    other = _SAME_CLASS_PARTNER[head.name]
    if row.queue is _QueueShape.SINGLE_HEAD:
        return [head]
    if row.queue is _QueueShape.HEAD_PLUS_SKIPPERS:
        return [head, light, light]
    if row.queue is _QueueShape.MULTI_MODEL_INTERLEAVE:
        return [head, other, head, other]
    if row.queue is _QueueShape.SAME_MODEL_BURST:
        return [head] * 3
    return [head] * 4


def _seed_governor_state(world: _DispatchWorld, row: _Row) -> None:
    """Drive the card's whole-card ledger into the row's governor state through its real charge APIs."""
    ledger = world.scheduler._whole_card_ledger
    if row.governor is _GovernorState.FRESH:
        return
    now = world.now
    state = ledger.state_for(None)
    if row.governor is _GovernorState.ESTABLISH_RATE_EXCEEDED:
        state.establishments.extend([now - index for index in range(_ESTABLISH_WINDOW_LIMIT)])
        assert ledger.establish_rate_exceeded(None, now=now) is True, (
            f"{row.label}: precondition, the card's establishment allowance is spent"
        )
        return
    cycle_cost = _WHOLE_CARD_ESTABLISH_GRACE_SECONDS + _WHOLE_CARD_RESTORE_GRACE_SECONDS
    for index in range(int(_GRACE_BUDGET_SECONDS // cycle_cost) + 1):
        state.grace_charges.append((now - index, _WHOLE_CARD_ESTABLISH_GRACE_SECONDS))
        state.grace_charges.append((now - index, _WHOLE_CARD_RESTORE_GRACE_SECONDS))
    assert ledger.grace_budget_exhausted(None, now=now) is True, (
        f"{row.label}: precondition, the card's rolling grace allowance is spent"
    )


async def _build_world(row: _Row) -> tuple[_DispatchWorld, list[ImageGenerateJobPopResponse]]:
    """Build the world for a row and pop its queue in order."""
    world = _DispatchWorld(
        card=row.card,
        lane_count=row.lanes,
        max_threads=row.max_threads,
        queue_depth=_QUEUE_LENGTHS[row.queue],
        whole_card_enabled=row.whole_card_enabled,
        cooldown_seconds=row.cooldown_seconds,
        max_hold_seconds=row.max_hold_seconds,
    )

    if row.residency is _Residency.RESIDENT_IDLE_TARGET:
        world.seed_resident(0, row.head_model, in_vram=True)
    elif row.residency is _Residency.RAM_STAGED:
        world.seed_resident(0, row.head_model, in_vram=False)
    elif row.residency is _Residency.RESIDENT_ON_SIBLING:
        # The copy sits on the last lane rather than the first, so dispatch must route to the lane that
        # already holds the weights instead of staging a second copy into the empty lane it would otherwise
        # reach for.
        world.seed_resident(row.lanes - 1, row.head_model, in_vram=True)

    _seed_governor_state(world, row)

    jobs: list[ImageGenerateJobPopResponse] = []
    for model in _queue_models(row):
        job = make_job_pop_response(model.name, width=512, height=512, ddim_steps=8)
        await world.pop(job)
        jobs.append(job)
    return world, jobs


def _fire_event(world: _DispatchWorld, row: _Row) -> None:
    """Apply the row's mid-sequence disturbance."""
    if row.event is _MidSequenceEvent.TARGET_DEATH_RESPAWN:
        world.kill_lane_holding(row.head_model)
    elif row.event is _MidSequenceEvent.EXTERNAL_RECLAIM:
        world.evict_idle_resident_sibling(except_model=row.head_model)


async def _drive(world: _DispatchWorld, row: _Row) -> None:
    """Run the row's ticks, firing its disturbance at the fixed disturbance tick."""
    for _ in range(row.run_ticks):
        await world.step()
        if world.tick == _DEATH_TICK:
            _fire_event(world, row)


# --------------------------------------------------------------------------------------------------------
# The property
# --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("row", _ROWS, ids=[row.label for row in _ROWS])
async def test_servable_head_reaches_dispatch_within_its_bound(row: _Row) -> None:
    """A physically servable head reaches sampling within the row's tick bound.

    A head whose model prices above the card's achievable ceiling takes the other disclosed exit: the model
    goes on the ceiling hold and the job is handed back for reissue. Keeping both outcomes in one table is
    what makes it a discriminating property rather than a claim that everything asked of the worker is
    admitted.
    """
    world, jobs = await _build_world(row)
    head = jobs[0]

    await _drive(world, row)

    if row.expected is _Expected.UNSERVABLE_HELD:
        assert world.dispatch_tick(head) is None, (
            f"{row.label}: a head the card cannot serve must not have been dispatched. {world.state_dump()}"
        )
        assert world.job_tracker.is_model_held_by_ceiling(row.head_model.name), (
            f"{row.label}: an unservable head must put its model on the disclosed ceiling hold rather than "
            f"leaving the queue quietly. {world.state_dump()}"
        )
        assert world.stage(head) is JobStage.PENDING_SUBMIT, (
            f"{row.label}: the held model's job must be faulted onward for reissue rather than kept in a "
            f"queue that cannot serve it. {world.state_dump()}"
        )
        return

    dispatched_at = world.dispatch_tick(head)
    assert dispatched_at is not None, (
        f"{row.label}: the servable head never reached sampling in {row.run_ticks} ticks. {world.state_dump()}"
    )
    assert dispatched_at <= row.tick_bound, (
        f"{row.label}: the head reached sampling at tick {dispatched_at}, past its bound of {row.tick_bound}. "
        f"{world.state_dump()}"
    )


@pytest.mark.parametrize("row", _ROWS, ids=[row.label for row in _ROWS])
async def test_whole_queue_drains_and_obligations_close(row: _Row) -> None:
    """Every queued job the row can serve drains, and every obligation opened along the way is released.

    Three obligations are checked at the far end of the run: the planned reserve-ledger charges booked for
    admitted preloads, the whole-card residency claim over the device, and the safety / post-processing GPU
    pauses a residency takes out. Each must be back at its released value once the queue has drained, because
    a charge or a pause that outlives the work it was taken for is what wedges the next head.
    """
    world, jobs = await _build_world(row)

    await _drive(world, row)

    for index, job in enumerate(jobs):
        # An unservable head's own jobs take the ceiling-hold exit; everything behind it whose model the card
        # can serve must still drain, or the unservable head has wedged the work it was standing in front of.
        if not row.drain_all and job.model == row.head_model.name:
            continue
        assert world.dispatch_tick(job) is not None, (
            f"{row.label}: queued job {index} ({job.model}) never reached sampling in {row.run_ticks} "
            f"ticks. {world.state_dump()}"
        )

    planned_mb = world.planned_overlay_mb()
    assert planned_mb == pytest.approx(0.0), (
        f"{row.label}: {planned_mb:.0f} MB of planned preload charge outlived the work it was booked for. "
        f"{world.state_dump()}"
    )
    assert world.scheduler.is_whole_card_residency_active() is False, (
        f"{row.label}: the card is still claimed for an exclusive residency after the queue drained. "
        f"{world.state_dump()}"
    )
    assert world.scheduler.whole_card_residency_grace_active() is False, (
        f"{row.label}: a residency grace window is still open after the queue drained. {world.state_dump()}"
    )
    assert world.lifecycle.is_safety_gpu_paused is False, (
        f"{row.label}: safety was left paused off the GPU after the queue drained. {world.state_dump()}"
    )
    assert world.lifecycle.is_post_process_gpu_paused is False, (
        f"{row.label}: the post-processing lane was left paused after the queue drained. {world.state_dump()}"
    )


@pytest.mark.parametrize(
    "row",
    [row for row in _ROWS if row.expected is _Expected.DISPATCHED],
    ids=[row.label for row in _ROWS if row.expected is _Expected.DISPATCHED],
)
async def test_servable_work_is_never_faulted_out_of_the_queue(row: _Row) -> None:
    """No job the row queues is faulted while it waits: every one leaves the queue by being sampled.

    The wedge-class failure this guards is a servable job being given up on rather than served. Read
    positively: the count of jobs that reached sampling equals the count queued, so none left the queue by
    any other door.
    """
    world, jobs = await _build_world(row)

    await _drive(world, row)

    sampled = [job for job in jobs if world.dispatch_tick(job) is not None]
    still_queued = [job for job in jobs if world.stage(job) is JobStage.PENDING_INFERENCE]
    assert len(sampled) + len(still_queued) == len(jobs), (
        f"{row.label}: {len(jobs) - len(sampled) - len(still_queued)} queued job(s) left the queue without "
        f"being sampled. {world.state_dump()}"
    )
    assert still_queued == [], (
        f"{row.label}: {len(still_queued)} job(s) were still waiting at the end of the run. {world.state_dump()}"
    )


@pytest.mark.parametrize(
    "row",
    [row for row in _ROWS if row.governor is _GovernorState.GRACE_EXHAUSTED],
    ids=[row.label for row in _ROWS if row.governor is _GovernorState.GRACE_EXHAUSTED],
)
async def test_grace_governed_head_is_served_without_claiming_the_card(row: _Row) -> None:
    """A head the grace budget refuses the card is still served, co-resident, without ever claiming it.

    The budget's window (1200s) far outlasts the governor dwell, so the brake cannot release before the
    head stops asking for the card: the head must fall through to ordinary measured admission and sample
    without spending the establishment the governor withheld.
    """
    world, jobs = await _build_world(row)
    head = jobs[0]
    establishments_before = len(world.scheduler._whole_card_ledger.state_for(None).establishments)

    await _drive(world, row)

    assert world.dispatch_tick(head) is not None, (
        f"{row.label}: a governed head that the card can hold must still be served. {world.state_dump()}"
    )
    assert len(world.scheduler._whole_card_ledger.state_for(None).establishments) <= establishments_before, (
        f"{row.label}: the governed head claimed the card the governor refused it. {world.state_dump()}"
    )


@pytest.mark.parametrize(
    "row",
    [row for row in _ROWS if row.governor is _GovernorState.ESTABLISH_RATE_EXCEEDED],
    ids=[row.label for row in _ROWS if row.governor is _GovernorState.ESTABLISH_RATE_EXCEEDED],
)
async def test_rate_governed_head_claims_the_card_once_the_brake_lifts(row: _Row) -> None:
    """A head the rate limiter defers is served by claiming the card after the brake's own window lapses.

    The rate window and the governor dwell are the same span by design: a rate deferral always releases on
    its own before the whole-card preference is abandoned, so the head takes the residency it is entitled
    to (sole residency beats coerced co-resident streaming) rather than being downgraded. The limiter's
    ceiling is still honored: the seeded establishments age out before the new one is counted.
    """
    world, jobs = await _build_world(row)
    head = jobs[0]

    await _drive(world, row)

    assert world.dispatch_tick(head) is not None, (
        f"{row.label}: a rate-deferred head must be served once the brake lifts. {world.state_dump()}"
    )
    state = world.scheduler._whole_card_ledger.state_for(None)
    assert 1 <= len(state.establishments) <= 2, (
        f"{row.label}: the head's claim must be a single establishment inside the limiter's ceiling, found "
        f"{len(state.establishments)}. {world.state_dump()}"
    )


_CLAIM_ROWS = tuple(row for row in _ROWS if row.claim is not _ClaimScenario.NONE)

_CLAIM_INTAKE_JOBS = 3
"""How many further jobs the horde offers a row that models continuing demand.

Enough that the burst outlives the establishment and the first dispatch, and few enough that the queue the
worker accepts still drains inside the row's ticks."""


async def _drive_claim_row(world: _DispatchWorld, row: _Row) -> list[ImageGenerateJobPopResponse]:
    """Run a claim row, offering the horde's answers through whatever the worker is currently asking for.

    Returns the resident-model jobs the worker accepted mid-run, which is what a row asserting the burst is
    served reads its outcome from. A foreign job is offered on the same ticks and is expected to be refused
    for as long as the claim stands: work the worker never advertised is work the horde never sends it.
    """
    accepted: list[ImageGenerateJobPopResponse] = []
    offered_resident = 0
    for _ in range(row.run_ticks):
        await world.step()
        if row.claim is _ClaimScenario.EMPTY_POPS_RELEASE:
            world.report_empty_pop()
            continue
        if row.claim is not _ClaimScenario.SERVES_THE_BURST or offered_resident >= _CLAIM_INTAKE_JOBS:
            continue
        offered_resident += 1
        resident_job = make_job_pop_response(row.head_model.name, width=512, height=512, ddim_steps=8)
        if await world.offer_job(resident_job):
            accepted.append(resident_job)
        foreign_job = make_job_pop_response(_SD15.name, width=512, height=512, ddim_steps=8)
        foreign_taken = await world.offer_job(foreign_job)
        if world.claim_ticks and world.claim_ticks[-1] == world.tick:
            assert foreign_taken is False, (
                f"{row.label}: a foreign job arrived while the residency claimed the offer, which is the "
                f"intake the claim exists to stop. {world.state_dump()}"
            )
    return accepted


@pytest.mark.parametrize("row", _CLAIM_ROWS, ids=[row.label for row in _CLAIM_ROWS])
async def test_the_residency_claims_intake_and_gives_it_back(row: _Row) -> None:
    """A held residency asks the horde for its own model alone, and every way that claim ends returns the pool.

    The claim is what makes a residency a burst-serving window rather than a card the worker keeps evicting
    itself from: while it stands, no foreign job is asked for, so nothing arrives to push the resident weights
    back to host RAM. Each row drives one of its ends and asserts the pool comes back, because a claim with no
    reachable end is a worker that has advertised itself down to one model permanently.
    """
    world, jobs = await _build_world(row)

    accepted = await _drive_claim_row(world, row)

    assert world.claim_ticks, (
        f"{row.label}: the residency never claimed the offer, so the row proves nothing about intake. "
        f"{world.state_dump()}"
    )
    for tick in world.claim_ticks:
        assert world.offers[tick] == frozenset({row.head_model.name}), (
            f"{row.label}: at tick {tick} the claim stood but the worker was still advertising "
            f"{sorted(world.offers[tick])}. {world.state_dump()}"
        )

    if row.claim is _ClaimScenario.SERVES_THE_BURST:
        assert accepted, f"{row.label}: the claim must keep taking work for the model it holds the card for"
        for job in jobs + accepted:
            assert world.dispatch_tick(job) is not None, (
                f"{row.label}: a job for the resident model was not served inside the window the residency "
                f"was held for. {world.state_dump()}"
            )

    if row.claim is _ClaimScenario.CAP_RETURNS_THE_POOL:
        assert row.cooldown_seconds > row.run_ticks * _TICK_SECONDS, (
            f"{row.label}: precondition, the cooldown outlasts the run, so the pool returning is attributable "
            "to the maximum hold and to nothing else"
        )
        assert world.claim_released_at > 0.0, (
            f"{row.label}: the maximum hold elapsed and the claim still stands. {world.state_dump()}"
        )
        assert world.claim_released_at >= world.claim_expires_at, (
            f"{row.label}: the claim ended before its cap, so this row is not measuring the cap"
        )

    if row.claim is _ClaimScenario.EMPTY_POPS_RELEASE:
        assert world.claim_released_at > 0.0, (
            f"{row.label}: a resident model the horde has no work for held the offer for the whole run. "
            f"{world.state_dump()}"
        )
        assert world.claim_released_at < world.claim_expires_at, (
            f"{row.label}: the claim was only ended by its cap; the empty answers must release it sooner, or a "
            "worker nobody is sending work to sits on its own claim"
        )

    if row.claim is _ClaimScenario.FOREIGN_QUEUED_FIRST:
        last_claimed_tick = world.claim_ticks[-1]
        for job in jobs:
            if job.model == row.head_model.name:
                continue
            dispatched_at = world.dispatch_tick(job)
            assert dispatched_at is not None, (
                f"{row.label}: a foreign job queued before the residency existed never drained after the "
                f"claim ended. {world.state_dump()}"
            )
            assert dispatched_at > last_claimed_tick, (
                f"{row.label}: a foreign job was pulled onto the card at tick {dispatched_at}, while the "
                f"residency still claimed it. {world.state_dump()}"
            )

    # However the claim ended, the worker is asking for its whole pool again.
    assert world.offers[world.tick] == frozenset(model.name for model in _MODEL_CLASSES), (
        f"{row.label}: the full model pool never came back to the offer. {world.state_dump()}"
    )


def test_the_matrix_states_its_own_coverage() -> None:
    """The table's axes are fully enumerated: every driven row is unique and every omission is explained."""
    labels = [row.label for row in _ROWS]
    assert len(labels) == len(set(labels)), "row labels must be unique so a failure names exactly one cell"
    assert len(_PRUNED_COMBINATIONS) > 0, "omissions from the cross-product must be listed, never silent"
    for combination, reason in _PRUNED_COMBINATIONS:
        assert combination and reason, "each pruned combination must carry its reason"
    driven_cards = {row.card.label for row in _ROWS}
    driven_models = {row.head_model.label for row in _ROWS}
    driven_queues = {row.queue for row in _ROWS}
    driven_residencies = {row.residency for row in _ROWS}
    driven_governors = {row.governor for row in _ROWS}
    driven_events = {row.event for row in _ROWS}
    assert driven_cards == {"8gb", "16gb", "24gb"}
    assert {"sd15", "sdxl", "flux"} <= driven_models
    assert driven_queues == set(_QueueShape)
    assert driven_residencies == set(_Residency)
    assert driven_governors == set(_GovernorState)
    assert driven_events == set(_MidSequenceEvent)
    assert {row.max_threads for row in _ROWS} == {1, 2}
    assert {row.claim for row in _ROWS} == set(_ClaimScenario)
