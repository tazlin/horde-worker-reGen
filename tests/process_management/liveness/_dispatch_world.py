"""The virtual-clock world the liveness suites drive the real scheduler over.

A modelled card whose free VRAM is *derived* from what is resident on it, a real
:class:`~horde_worker_regen.process_management.scheduling.inference_scheduler.InferenceScheduler`,
:class:`~horde_worker_regen.process_management.jobs.job_tracker.JobTracker`,
:class:`~horde_worker_regen.process_management.lifecycle.process_map.ProcessMap` and reserve ledger, and a
hand-run set of children advanced one scheduling tick at a time on an injected clock. Everything the
scheduler decides is decided by production code; the world only supplies the physics.

Two fidelities live here, selected by the ``closed_loop`` constructor flag.

The default is the *scheduling* fidelity the bounded-dispatch matrix and the generated chaos suite are
written against: a dispatched job occupies its lane for exactly one tick, weights stay on the card once
committed, and the device-free governor is never sampled (so every card reads HEALTHY). It answers questions
about admission, routing and residency ordering, and it deliberately says nothing about time.

``closed_loop=True`` closes the loop between policy and the card. The parent's device-free governor and the
verified reclaim ladder join the tick, a dispatched job occupies its lane for derived load, sample and decode
phases, sampling charges a transient peak to the card for its window, and the scheduler's retention grants
decide whether a finished job leaves its weights behind. That is the fidelity a free-VRAM crater, a governor
trajectory and a duty figure are representable at, and it is what the incident scenarios in
``test_incident_scenarios.py`` are driven at.

Two tenants of the card are not owned by any job and are opt-in per row: ``held_component_mb`` (weights an
idle lane holds device-warm between jobs, returned only by an unload the parent actuates) and
``child_evicts_granted_resident`` (the child freeing the whole of a lane's committed copy to fund a charge,
leaving the parent's retained-resident record standing over an empty device).

Assertion helpers over a completed run live in ``_world_assertions.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import Mock

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import (
    HeldComponentSnapshot,
    HordeControlFlag,
    HordeImageResult,
    HordeInferenceControlMessage,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.jobs.job_tracker import JobStage, JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.lru_cache import LRUCache
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.resources.device_free_governor import DeviceFreeGovernor, GovernorState
from horde_worker_regen.process_management.resources.reclaim_ladder import (
    ReclaimRung,
    ReclaimRungKind,
    VerifiedReclaimLadder,
    build_reclaim_ladder,
)
from horde_worker_regen.process_management.resources.resource_budget import (
    CommittedReserveLedger,
    predict_job_sampling_vram_mb,
)
from horde_worker_regen.process_management.resources.vram_arbiter import MeasuredVramSnapshot
from horde_worker_regen.process_management.scheduling.governance.whole_card import offer_under_pop_claim
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _SAFETY_GPU_LOAD_CHARGE_MB,
    InferenceScheduler,
)
from tests.process_management.conftest import (
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

_SAFETY_PROCESS_ID = 100
_POST_PROCESS_LANE_ID = 200
"""Ids for the service processes, kept well clear of the inference lane ids so pool growth (which allocates
the next free inference id) can never collide with one."""


@dataclass(frozen=True)
class _JobShape:
    """The generation size a row's jobs ask for, which is what scales sampling activation.

    Resolution and batch move the activation working set by several gigabytes while leaving the resident
    weights untouched, so the shape axis is what separates a model's persistent residency cost from its
    transient sampling peak.
    """

    label: str
    width: int
    height: int
    batch: int


_SHAPE_SMALL = _JobShape("small", 512, 512, 1)
_SHAPE_HIRES_BATCH = _JobShape("hires_batch", 1280, 1280, 2)
"""A high-resolution batch: an SDXL sampler's activation-inclusive peak roughly doubles against its ~4.9 GB
of weights, which is the regime where an activation spike can be mistaken for permanent residency."""


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
# Closed-loop time and memory shape
# --------------------------------------------------------------------------------------------------------

_LOAD_SECONDS_PER_GB = 0.96
"""Seconds a lane spends bringing one gigabyte of checkpoint weights from host RAM onto the card.

Calibrated so an SDXL checkpoint (4900 MB) takes roughly the 4.6 s a host-to-device weight upload costs over
a consumer PCIe link, which is the cost retention exists to remove: it is paid once per job for weights that
were still on the card when the previous job ended."""

_SAMPLE_SECONDS_PER_STEP_PER_MEGAPIXEL: dict[str, float] = {
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1.value: 0.06,
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl.value: 0.175,
    KNOWN_IMAGE_GENERATION_BASELINE.flux_1.value: 0.5,
}
"""Seconds per sampling step per megapixel of requested output, per model class.

Sampling cost scales with steps and with the activation area the sampler walks, and it is the only phase the
card is actually earning kudos during, so it is what a duty figure is measured against. The SDXL figure puts
a forty-step megapixel job at seven seconds of sampling."""

_DEFAULT_SAMPLE_SECONDS_PER_STEP_PER_MEGAPIXEL = 0.175
"""Sampling pace for a baseline with no entry above; the mid class, so an unlisted model is not free."""

_CHILD_FREE_MARGIN_MB = 1024.0
"""Free VRAM (MB) a child's executor keeps clear of the allocation it is about to make.

Its shortfall arithmetic frees for the requirement plus a margin rather than for the requirement exactly, so a
load that only just fits does not leave the device at zero and the next allocation of any kind does not page.
The margin is therefore also the floor a child that computes its shortfall against device truth leaves
standing, and a card sitting below it is evidence the shortfall was computed against something else."""

_OFFLOAD_SAMPLING_PENALTY = 1.0
"""How much slower sampling runs when the whole of a checkpoint is streamed rather than resident.

Weights left in host RAM are fetched across the bus for every step they are needed on, so a fully streamed
model doubles its sampling window and a partially streamed one pays in proportion. This is the price of the
child relieving a shortfall out of its own footprint, and it is why fitting by offload is a cost rather than a
free win."""

_DECODE_SECONDS_PER_MEGAPIXEL: dict[str, float] = {
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1.value: 1.2,
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl.value: 2.0,
    KNOWN_IMAGE_GENERATION_BASELINE.flux_1.value: 2.5,
}
"""Seconds of VAE decode per megapixel of output, per model class: the tail a successor arrives during."""

_DEFAULT_DECODE_SECONDS_PER_MEGAPIXEL = 2.0
"""Decode pace for a baseline with no entry above."""


@dataclass
class _SlotOccupancy:
    """One dispatched job's hold on its lane, in world seconds, and the transient it charges the card.

    A job occupies its lane from dispatch until its decode finishes. The phases are what make the hold
    structured rather than a single opaque interval: the load phase is the cost retention removes, the sample
    phase is the only phase that counts toward duty and the one that charges an activation peak, and the
    decode phase is the tail a same-model successor arrives during while the retainer still holds its weights.
    """

    job_id: str
    lane_id: int
    model: str
    weights_mb: float
    transient_mb: float
    """Activation-only peak (MB) the sampling window adds on top of the weights already charged."""
    sample_from: float
    sample_until: float
    decode_until: float
    transient_charged: bool = False


# --------------------------------------------------------------------------------------------------------
# The world: a scheduler driven over a card whose free VRAM follows from what is resident on it
# --------------------------------------------------------------------------------------------------------


class _LadderActuation:
    """One rung the reclaim ladder actually performed, with the tick and target it acted on."""

    def __init__(self, tick: int, kind: ReclaimRungKind, tenant_label: str, target_process_id: int | None) -> None:
        """Record one performed rung."""
        self.tick = tick
        self.kind = kind
        self.tenant_label = tenant_label
        self.target_process_id = target_process_id

    def __repr__(self) -> str:
        """Describe the actuation for a failing assertion's message."""
        return f"tick {self.tick}: {self.kind.value} on {self.tenant_label}"


class _RecordingActuator:
    """The scheduler as the ladder's actuator, with every performed rung recorded on the world.

    The engine dispatches rungs through :func:`~horde_worker_regen.process_management.resources.reclaim_ladder
    .execute_reclaim_rung`, which reaches the actuator by method name and never reports which rung it ran. A
    thin pass-through is therefore the seam that lets a run assert how far down the ladder the card was taken
    without the world second-guessing the engine's ordering.
    """

    def __init__(self, world: _DispatchWorld) -> None:
        """Wrap ``world``'s scheduler, recording onto the world's actuation series."""
        self._world = world
        self._scheduler = world.scheduler

    def _record(self, kind: ReclaimRungKind, tenant_label: str, target_process_id: int | None, acted: bool) -> bool:
        if acted:
            self._world.ladder_actuations.append(
                _LadderActuation(self._world.tick, kind, tenant_label, target_process_id),
            )
        return acted

    def unload_idle_model(self, process_id: int, device_index: int | None) -> bool:
        """Unload the resident model on ``process_id``, recording the rung when the scheduler acted."""
        acted = self._scheduler.unload_idle_model(process_id, device_index)
        return self._record(ReclaimRungKind.UNLOAD_IDLE_MODEL, f"inference#{process_id}", process_id, acted)

    def release_idle_cache(self, process_id: int) -> bool:
        """Release ``process_id``'s allocator cache, recording the rung when the scheduler acted."""
        acted = self._scheduler.release_idle_cache(process_id)
        return self._record(ReclaimRungKind.RELEASE_IDLE_CACHE, f"process#{process_id}", process_id, acted)

    def pause_post_process_lane(self, device_index: int | None) -> bool:
        """Pause the post-processing lane, recording the rung when the scheduler acted."""
        acted = self._scheduler.pause_post_process_lane(device_index)
        return self._record(ReclaimRungKind.PAUSE_PP_LANE, "post_process_lane", None, acted)

    def pause_vae_lane(self, device_index: int | None) -> bool:
        """Pause the VAE/image lane, recording the rung when the scheduler acted."""
        acted = self._scheduler.pause_vae_lane(device_index)
        return self._record(ReclaimRungKind.PAUSE_VAE_LANE, "vae_lane", None, acted)

    def pause_component_lane(self, device_index: int | None) -> bool:
        """Pause the component/text-encode lane, recording the rung when the scheduler acted."""
        acted = self._scheduler.pause_component_lane(device_index)
        return self._record(ReclaimRungKind.PAUSE_COMPONENT_LANE, "component_lane", None, acted)

    def safety_off_gpu(self, device_index: int | None) -> bool:
        """Move safety off the card, recording the rung when the scheduler acted."""
        acted = self._scheduler.safety_off_gpu(device_index)
        return self._record(ReclaimRungKind.SAFETY_OFF_GPU, "safety", None, acted)

    def restore_post_process_lane(self, device_index: int | None) -> bool:
        """Restore the post-processing lane the ladder paused."""
        return self._scheduler.restore_post_process_lane(device_index)

    def restore_vae_lane(self, device_index: int | None) -> bool:
        """Restore the VAE/image lane the ladder paused."""
        return self._scheduler.restore_vae_lane(device_index)

    def restore_component_lane(self, device_index: int | None) -> bool:
        """Restore the component/text-encode lane the ladder paused."""
        return self._scheduler.restore_component_lane(device_index)

    def restore_live_contexts(self, device_index: int | None) -> bool:
        """Regrow the inference pool a reclaim reduction shrank."""
        return self._scheduler.restore_live_contexts(device_index)

    def record_calibration_event(self, rung: ReclaimRung, *, promised_mb: float, realized_mb: float) -> None:
        """Fold a verified shortfall back into the worker's calibration."""
        self._scheduler.record_calibration_event(rung, promised_mb=promised_mb, realized_mb=realized_mb)


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
        enable_vram_budget: bool = True,
        high_performance_mode: bool = False,
        moderate_performance_mode: bool = False,
        unload_models_from_vram_often: bool = False,
        cooldown_seconds: int = 0,
        max_hold_seconds: int = 180,
        tick_seconds: float = _TICK_SECONDS,
        service_contexts: bool = False,
        disaggregated: bool = False,
        closed_loop: bool = False,
        legacy_comfy_vram_unload: bool = False,
        child_free_view_lie_mb: float = 0.0,
        footprint_undershoot: float = 1.0,
        safety_off_gpu_allowed: bool = False,
        unload_release_delay_seconds: float = 0.0,
        child_evicts_granted_resident: bool = False,
        child_unload_leaks_mb: float = 0.0,
    ) -> None:
        """Build the process pool, the model map, and the scheduler for one row.

        Args:
            card: The device-free profile the row runs on.
            lane_count: How many inference lanes the pool holds.
            max_threads: The concurrent-sampling cap (the ``max_threads`` config axis).
            queue_depth: The configured queue size, so an at-depth row can express a full queue.
            whole_card_enabled: Whether preventative whole-card exclusive residency is on.
            enable_vram_budget: Whether measured VRAM admission and its recovery actions are enabled.
            high_performance_mode: Whether the high-throughput overlap posture is enabled.
            moderate_performance_mode: Whether the moderate-throughput overlap posture is enabled.
            unload_models_from_vram_often: Whether idle model residency is released eagerly.
            cooldown_seconds: How long a drained residency is retained for a follow-on heavy job. Raised by
                the rows that need the residency to still be standing when they make their assertion.
            max_hold_seconds: The operator ceiling on one residency episode, which is also what bounds its
                claim over the offer.
            tick_seconds: How much of the world's clock one tick advances. The default reaches the governance
                windows these rows turn on; a caller whose scenarios turn on the scheduler's short budgets
                (the affinity line-skip window has a fifteen-second floor) passes a shorter tick, so those
                budgets are sampled several times rather than stepped over in one advance.
            service_contexts: Whether the safety process sits on the card and the post-processing lane holds
                a context. Both are device commitments that stopping idle *inference* siblings cannot
                reclaim, so they are what a card's structural floor is squeezed by without any inference
                context being the cause.
            disaggregated: Whether the row's jobs are disaggregation-class, so a sampler is priced for the
                UNet it holds alone rather than for a whole job.
            closed_loop: Whether to close the loop between policy and the card: the device-free governor and
                the verified reclaim ladder join the tick, dispatched jobs occupy their lane for derived
                load/sample/decode phases charging a sampling transient, and the scheduler's retention grants
                decide whether a finished job leaves its weights on the device. Off, a dispatch occupies its
                lane for one tick and weights stay put, which is the fidelity the admission-ordering suites
                state their tick bounds against.
            legacy_comfy_vram_unload: Whether the escape hatch that restores the old flag-based child regime
                is configured. Under it the child's executor returns the card at the end of every prompt below
                anything a grant can suppress, so a closed-loop run evicts on every completion whatever the
                dispatch asked for.
            child_free_view_lie_mb: How much more free VRAM a child believes it has than the card really has.
                A child sees only its own allocations and, under WDDM, memory the driver has not returned still
                reads as free, so its view runs ahead of the device. The gap is what makes the child's own
                shortfall freeing under-free. Zero is a truthful child.
            footprint_undershoot: Multiplier from what the scheduler's static fit predicts a job costs the card
                to what it actually costs. Above one, admission passes on a figure the load then exceeds, which
                is the regime where the parent's own defenses have already been satisfied and only the child's
                freeing stands between the load and the card.
            safety_off_gpu_allowed: Whether the operator permits safety to be moved off the card to reclaim it.
                Off, the ladder has no safety rung at all, so a run says nothing about what reaching one costs;
                on, the safety pause and its restore are modelled as the process cycle they are.
            unload_release_delay_seconds: How long after the parent sends an unload the card actually gets the
                memory back. An unload is an IPC the child services between its own allocations and the driver
                then returns the block, so a multi-gigabyte release lands seconds after the command; at zero the
                release is instantaneous, which no worker's card ever is.
            child_evicts_granted_resident: Whether the child's executor may free the whole of a lane's
                committed weights, retention grant included, when its own shortfall arithmetic runs out of
                anything else to give. ComfyUI frees memory on the device to fund an allocation and its
                requirement is unbounded, so the grant suppresses only the worker's own end-of-job evictor and
                never ComfyUI's; the parent's record of what the slot holds is a prediction made at dispatch
                and nothing in the parent can measure the difference. Off, every lane's device copy survives
                exactly as the record says.
            child_unload_leaks_mb: How much of a lane's committed weights an unload leaves on the card. A full
                free is a request the child's backend answers by dropping what it can, and a model a live
                reference still pins is skipped: the command returns, the card keeps the weights, and the lane
                goes on reporting them as used. Zero is a backend that gives everything back, which is what
                every unload elsewhere in this world does.
        """
        self.card = card
        self.tick_seconds = tick_seconds
        self.closed_loop = closed_loop
        self.child_free_view_lie_mb = child_free_view_lie_mb
        self.footprint_undershoot = footprint_undershoot
        self.unload_release_delay_seconds = unload_release_delay_seconds
        self.child_evicts_granted_resident = child_evicts_granted_resident
        self.child_unload_leaks_mb = child_unload_leaks_mb
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
        self._resident_model: dict[int, str] = {}
        """The model whose weights each lane most recently committed to the device."""
        self._loading: dict[int, tuple[str, float]] = {}
        self._transient_mb: dict[int, float] = {}
        """Per-lane sampling activation charged to the card for the window a lane is sampling in."""
        self._offloaded_mb: dict[int, float] = {}
        """Per-lane weights the child kept in host RAM rather than commit, to relieve its own shortfall."""
        self._held_component_mb: dict[int, float] = {}
        """Per-lane component weights the child holds device-warm between jobs.

        A tenant of the card that no job owns: it survives every job boundary, is not part of any dispatch's
        footprint, and the only thing that returns it is an unload the parent actuates on that lane. A card
        packed with these is therefore squeezed by lanes that are idle and hold no running work, which is the
        regime a fit computed from live contexts and dispatched jobs alone cannot see."""
        self._granted_resident_evicted: set[int] = set()
        """Lanes whose committed weights the child freed mid-job while the parent still records them.

        Cleared as each job settles, which is where the divergence either reaches the parent's record or
        becomes the phantom the rest of the session is priced against."""
        self.child_granted_resident_evictions: list[str] = []
        """Every time the child freed a lane's granted weights to fund a charge it could not otherwise make."""
        self.unload_leaks: list[str] = []
        """Every unload the child could not complete, with what the card kept."""
        self._unload_leaked: set[int] = set()
        """Lanes whose standing unload command has already been served and refused.

        One command is served once. The parent's record of it stands until something retires it, and reading
        that standing record as a fresh command every tick would manufacture a re-issue the parent never
        made, which is exactly the behaviour a scenario here has to be able to tell apart from the real one."""
        self.retained_resident_divergences: list[tuple[int, int, str]] = []
        """Every tick an idle slot's retained-resident record named weights the card was not holding.

        As (tick, lane, model). Sampled on idle slots only: a slot mid-job can lose its weights to the child
        at any moment and the parent cannot know until the job reports, so the record is only a claim about
        the card between jobs. That is also where it is acted on, by the retention fit, the dispatch
        admission gate and same-model routing alike."""
        self._occupancy: dict[str, _SlotOccupancy] = {}
        """Closed-loop only: the jobs currently holding a lane, keyed by job id."""
        self._dispatch_device_truth_mb: dict[str, float | None] = {}
        """Per dispatched job, the device-free figure its own control message carried, or None when it carried
        none. Read off the message the scheduler actually sent, so a run with that field neutered is a faithful
        reinjection of a worker whose children are never told the truth."""
        self._lane_charge_at_dispatch: dict[str, float] = {}
        """Per dispatched job, what its lane already held when the dispatch was sent, so the dispatch-time
        device figure can be aged by the lane's own growth since rather than read as still current."""
        self.child_shortfall_frees: list[str] = []
        """Every time a child's own shortfall freeing returned weights to make room for a charge."""
        self.child_overcommits: list[str] = []
        """Every charge a child made that its believed free did not cover and its own freeing could not
        relieve: the moment the card is committed beyond what it has, which is what a crater is made of."""
        self.reclaim_commands = 0
        self._unload_due_at: dict[int, float] = {}
        """Per lane, the world-clock instant an issued unload's memory comes back to the card."""
        self.unload_releases: list[tuple[int, int]] = []
        """Every unload the world booked as complete, as (tick, lane): the tick the lane's charge came off the
        card ledger, which is when a release the parent ordered stops being in flight. What the parent's own
        readings then show still trails it by the reporting path, so this is the world's bookkeeping instant
        rather than an observed device-free figure."""
        self.safety_pause_events: list[tuple[int, float, PauseOwner]] = []
        """Every time the safety context left the card, as (tick, world clock, the owner that asked)."""
        self.safety_restore_events: list[tuple[int, float]] = []
        """Every time the safety context came back onto the card, as (tick, world clock)."""
        self.snapshot: MeasuredVramSnapshot | None = None
        """The most recent cycle-frozen device measurement, the surface a row reads obligations back from."""

        self._service_contexts = service_contexts
        processes: dict[int, HordeProcessInfo] = {}
        for lane_id in range(lane_count):
            lane = make_mock_process_info(lane_id, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
            processes[lane_id] = lane
        if service_contexts:
            processes[_SAFETY_PROCESS_ID] = make_mock_process_info(
                _SAFETY_PROCESS_ID,
                model_name=None,
                process_type=HordeProcessType.SAFETY,
                state=HordeProcessState.WAITING_FOR_JOB,
            )
            processes[_POST_PROCESS_LANE_ID] = make_mock_process_info(
                _POST_PROCESS_LANE_ID,
                model_name=None,
                process_type=HordeProcessType.POST_PROCESS,
                state=HordeProcessState.WAITING_FOR_JOB,
            )
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
            enable_vram_budget=enable_vram_budget,
            whole_card_exclusive_residency=whole_card_enabled,
            whole_card_residency_safety_off_gpu=safety_off_gpu_allowed,
            safety_on_gpu=service_contexts,
            vram_reserve_mb=0,
            ram_reserve_mb=8192.0,
            vram_per_process_overhead_mb=_FIRST_CONTEXT_MB,
            whole_card_residency_cooldown_seconds=cooldown_seconds,
            whole_card_residency_max_hold_seconds=max_hold_seconds,
            high_performance_mode=high_performance_mode,
            moderate_performance_mode=moderate_performance_mode,
            unload_models_from_vram_often=unload_models_from_vram_often,
            legacy_comfy_vram_unload=legacy_comfy_vram_unload,
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
        if disaggregated:
            # Class-eligibility is the scheduler's own seam for "this job will run as a UNet-only sampler".
            # Pinning it is what makes a row's jobs priced, admitted, and dispatched on the disaggregated
            # path without standing up the orchestrator's lanes, which these rows do not vary.
            self._scheduler._is_disaggregation_class_eligible = lambda _job: True  # type: ignore[method-assign]
        self._scheduler.set_device_free_mb_provider(lambda _device_index: self.device_free_mb())
        # The rows vary VRAM, never host RAM: pinning an ample reading keeps the RAM admission gates out of
        # the variation and stops a row's outcome depending on how much memory the machine running it has.
        self._scheduler.set_available_ram_mb_provider(lambda: _AMPLE_RAM_MB)
        self._sync_reported_vram()

        self.first_dispatch: dict[str, int] = {}
        self._dispatched_at: dict[str, int] = {}
        self._lane_of: dict[str, int] = {}

        self._governor = DeviceFreeGovernor()
        self._reclaim_ladder = VerifiedReclaimLadder()
        self._actuator = _RecordingActuator(self)
        self._healthy_since: float | None = None
        self.governor_states: list[GovernorState] = []
        """The governor's committed state at every tick of a closed-loop run, in order."""
        self.ladder_actuations: list[_LadderActuation] = []
        """Every reclaim rung the ladder actually performed, in the order it performed them."""
        self.min_device_free_mb = self.device_free_mb()
        """The lowest device-free reading the card ever showed, sampled once per tick after the card moves."""
        self.sampling_slot_seconds = 0.0
        """Slot-seconds spent sampling: the numerator of the run's duty figure."""
        self.started_at = self.now
        self.completed_jobs = 0
        self.weight_uploads = 0
        """Dispatches that had to bring their model's weights onto the card, the cost retention removes.

        A streak served on retained weights uploads once; one that re-seats each successor beside or in place
        of the retained copy uploads once per job, and the difference is entirely GPU time not spent sampling."""
        self.dispatch_lanes: list[tuple[str, int]] = []
        """Every dispatch as (model, lane), so a scenario can say where a streak's successors were seated."""
        if closed_loop:
            # The engine is the single owner of every reclaim restore obligation, including the live-context
            # reductions the scheduler's own admission path books. Sharing one instance is what makes a
            # reduction unwound on the same debounced-HEALTHY signal the ladder's lane pauses are.
            self._scheduler.set_reclaim_ladder(self._reclaim_ladder)

    # -- card model ---------------------------------------------------------------------------------------

    def _inference_lanes(self) -> list[HordeProcessInfo]:
        """The pool's inference lanes, which is what the row's residency and teardown bookkeeping is about."""
        return [lane for lane in self._process_map.values() if lane.process_type == HordeProcessType.INFERENCE]

    def _context_charge_mb(self) -> float:
        """The card's total context cost: the one-time runtime, each further context, and safety's weights.

        The safety process carries resident classifier weights on top of its context, so it is charged its
        whole-process figure (the one the scheduler also prices it at) rather than a bare context; the
        post-processing lane holds a context and no at-rest model, so it costs the marginal. Safety paused off
        the card costs it nothing, which is the whole of what the reclaim rung that moves it buys.
        """
        lanes = max(1, len(self._inference_lanes()))
        charge = _FIRST_CONTEXT_MB + _MARGINAL_CONTEXT_MB * (lanes - 1)
        if self._service_contexts:
            charge += _MARGINAL_CONTEXT_MB
            if not self.safety_is_off_gpu():
                charge += _SAFETY_GPU_LOAD_CHARGE_MB
        return charge

    def safety_is_off_gpu(self) -> bool:
        """Whether the safety context is currently paused off the card."""
        return bool(self._lifecycle.is_safety_gpu_paused)

    def pause_safety_on_gpu(self, *, owner: PauseOwner) -> bool:
        """Take the safety context off the card for ``owner``, returning whether this call did it.

        Mirrors the lifecycle's single-owner pause: a context already off the card is not paused twice, and the
        owner is recorded so only that owner's restore can bring it back.
        """
        if self.safety_is_off_gpu():
            return False
        self._lifecycle.is_safety_gpu_paused = True
        self._lifecycle.safety_pause_owner = owner
        self.safety_pause_events.append((self.tick, self.now, owner))
        self._sync_reported_vram()
        return True

    def restore_safety_on_gpu(self, *, owner: PauseOwner) -> bool:
        """Put the safety context back on the card for ``owner``, returning whether this call did it."""
        if not self.safety_is_off_gpu() or self._lifecycle.safety_pause_owner is not owner:
            return False
        self._lifecycle.is_safety_gpu_paused = False
        self._lifecycle.safety_pause_owner = None
        self.safety_restore_events.append((self.tick, self.now))
        self._sync_reported_vram()
        return True

    def device_free_mb(self) -> float:
        """The truthful device-free reading: the card total less its contexts, weights, and live activation.

        The sampling activation term is what makes a crater representable: weights are a persistent tenant a
        static fit can price ahead of time, while a sampling window adds gigabytes for its own duration only,
        and it is the sum of the two across concurrent slots that reaches a paging cliff.
        """
        held = (
            sum(self._resident_mb.values()) + sum(self._transient_mb.values()) + sum(self._held_component_mb.values())
        )
        return max(0.0, self.card.total_mb - self._context_charge_mb() - held)

    def _actual_charge_mb(self, predicted_mb: float) -> float:
        """What a charge the scheduler priced at ``predicted_mb`` really costs the card.

        The undershoot factor is the whole of the difference, so a world left at its default charges exactly
        what the forecast said and every existing row's arithmetic is unchanged.
        """
        return predicted_mb * self.footprint_undershoot

    @staticmethod
    def _dispatched_device_truth_mb(lane: HordeProcessInfo) -> float | None:
        """The device-free figure the last inference dispatch to ``lane`` carried, or None when it carried none.

        Taken from the control message the scheduler handed the lane's pipe, which is the same object a real
        child unpickles, so the world's clamp is driven by the field production populates.
        """
        pipe: object = lane.pipe_connection
        assert isinstance(pipe, Mock), "the world's lanes send through a recording pipe"
        for call in reversed(pipe.send.call_args_list):
            message = call.args[0] if call.args else None
            if isinstance(message, HordeInferenceControlMessage):
                return message.device_free_mb
        return None

    def _lane_charge_mb(self, lane_id: int) -> float:
        """What one lane holds on the card: committed weights, live activation, and device-warm components."""
        return (
            self._resident_mb.get(lane_id, 0.0)
            + self._transient_mb.get(lane_id, 0.0)
            + self._held_component_mb.get(lane_id, 0.0)
        )

    def _child_believed_free_mb(self, lane_id: int, job_id: str) -> float:
        """How much free VRAM the child on ``lane_id`` believes the card has while serving ``job_id``.

        Its own view is the truth plus whatever the process-local reading overstates. When its dispatch
        carried the parent's device reading, that reading caps the view: the child ages it by its own growth
        since the dispatch (the only allocations it can account for) and takes the lower of the two, which is
        what keeps a shortfall computed against the device rather than against the process.
        """
        believed = self.device_free_mb() + self.child_free_view_lie_mb
        dispatch_truth = self._dispatch_device_truth_mb.get(job_id)
        if dispatch_truth is not None:
            own_growth = self._lane_charge_mb(lane_id) - self._lane_charge_at_dispatch.get(job_id, 0.0)
            believed = min(believed, dispatch_truth - own_growth)
        return max(0.0, believed)

    def _child_admit_charge(
        self,
        *,
        lane_id: int,
        job_id: str,
        model: str,
        charge_mb: float,
        weight_load: bool,
    ) -> float:
        """Let the child on ``lane_id`` relieve a charge its believed free does not cover, and say what it commits.

        The child's executor allocates by shortfall: before a load or a sampling window it compares what the
        allocation needs (plus a device margin, so a load that only just fits does not leave the card at zero)
        against what it believes is free, and gives back as much of what *it* holds as the comparison asks for.
        Two things are ever available to give: a checkpoint it is still carrying for an earlier job, and part of
        the running job's own weights, which it can leave in host RAM and stream at a sampling cost. A child
        whose believed free runs ahead of the card computes no shortfall at all: it gives nothing back and
        allocates anyway, and the card is then committed past what it has.

        Returns:
            The megabytes the child actually commits for this charge, which is short of ``charge_mb`` by
            whatever of a weight load it decided to leave in host RAM.
        """
        if not self.closed_loop:
            return charge_mb
        committed_mb = self._child_relieve_charge(
            lane_id=lane_id,
            job_id=job_id,
            model=model,
            charge_mb=charge_mb,
            weight_load=weight_load,
        )
        if committed_mb > self.device_free_mb():
            self.child_overcommits.append(
                f"tick {self.tick}: lane {lane_id} committed {committed_mb:.0f}MB to a card with "
                f"{self.device_free_mb():.0f}MB really free, believing it had "
                f"{self._child_believed_free_mb(lane_id, job_id):.0f}MB",
            )
        return committed_mb

    def _child_relieve_charge(
        self,
        *,
        lane_id: int,
        job_id: str,
        model: str,
        charge_mb: float,
        weight_load: bool,
    ) -> float:
        """Run the child's shortfall relief for one charge and return what it commits after relieving."""
        shortfall_mb = (charge_mb + _CHILD_FREE_MARGIN_MB) - self._child_believed_free_mb(lane_id, job_id)
        if shortfall_mb <= 0.0:
            return charge_mb

        stale_model = self._resident_model.get(lane_id)
        if stale_model is not None and stale_model != model:
            freed_mb = self._resident_mb.pop(lane_id, 0.0)
            self._resident_model.pop(lane_id, None)
            entry = self._model_map.root.get(stale_model)
            if entry is not None and entry.process_id == lane_id:
                self._model_map.root.pop(stale_model, None)
            self._sync_reported_vram()
            self.child_shortfall_frees.append(
                f"tick {self.tick}: lane {lane_id} returned {freed_mb:.0f}MB of {stale_model} toward a "
                f"{shortfall_mb:.0f}MB shortfall",
            )
            shortfall_mb = (charge_mb + _CHILD_FREE_MARGIN_MB) - self._child_believed_free_mb(lane_id, job_id)
            if shortfall_mb <= 0.0:
                return charge_mb

        # Nothing else on this lane belongs to another job, so what is left to give is part of the running
        # job's own weights: a weight load brings less of the checkpoint onto the card, and an activation
        # (which cannot be streamed) is funded by returning weights the load already committed.
        if weight_load:
            offloaded_mb = min(shortfall_mb, charge_mb)
            self._offloaded_mb[lane_id] = self._offloaded_mb.get(lane_id, 0.0) + offloaded_mb
            self.child_shortfall_frees.append(
                f"tick {self.tick}: lane {lane_id} left {offloaded_mb:.0f}MB of {model} in host RAM rather than "
                f"charge a {shortfall_mb:.0f}MB shortfall to the card",
            )
            return charge_mb - offloaded_mb
        if self._evict_granted_resident(lane_id, shortfall_mb, model=model):
            return charge_mb
        offloaded_mb = min(shortfall_mb, self._resident_mb.get(lane_id, 0.0))
        if offloaded_mb > 0.0:
            self._resident_mb[lane_id] = self._resident_mb.get(lane_id, 0.0) - offloaded_mb
            self._offloaded_mb[lane_id] = self._offloaded_mb.get(lane_id, 0.0) + offloaded_mb
            self._sync_reported_vram()
            self.child_shortfall_frees.append(
                f"tick {self.tick}: lane {lane_id} returned {offloaded_mb:.0f}MB of {model}'s own weights to "
                f"fund a {charge_mb:.0f}MB activation",
            )
        return charge_mb

    def _evict_granted_resident(self, lane_id: int, shortfall_mb: float, *, model: str) -> bool:
        """Free the whole of ``lane_id``'s device copy when nothing but its own weights can cover a shortfall.

        The point the lane has nothing left to give but the checkpoint it is running on, which is where the
        grant stops meaning anything: the grant suppresses the worker's own end-of-job evictor and nothing
        else, and the requirement ComfyUI frees against here is unbounded, so what comes back is the whole
        copy rather than the shortfall's worth of it. The parent's record is deliberately left exactly as it
        was, which is the whole subject: nothing the parent measures separates a slot whose weights are still
        there from one whose are not, so the record stands until the child says otherwise.

        Returns:
            True when the copy was freed, so the caller skips the partial offload it would otherwise do.
        """
        if not self.child_evicts_granted_resident or shortfall_mb <= 0.0:
            return False
        freed_mb = self._resident_mb.pop(lane_id, 0.0)
        if freed_mb <= 0.0:
            return False
        self._resident_model.pop(lane_id, None)
        self._offloaded_mb[lane_id] = self._offloaded_mb.get(lane_id, 0.0) + freed_mb
        self._granted_resident_evicted.add(lane_id)
        self.child_granted_resident_evictions.append(
            f"tick {self.tick}: lane {lane_id} freed its whole {freed_mb:.0f}MB copy of {model} against a "
            f"{shortfall_mb:.0f}MB shortfall, keeping nothing the grant covered",
        )
        self._sync_reported_vram()
        return True

    def _sync_reported_vram(self) -> None:
        """Publish the derived card state through the children's VRAM reports, as a live worker would."""
        used_mb = self.card.total_mb - self.device_free_mb()
        for lane in self._process_map.values():
            lane.total_vram_mb = int(self.card.total_mb)
            lane.vram_usage_mb = int(used_mb)
            lane.process_reserved_mb = int(self._lane_charge_mb(lane.process_id))

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
            self._resident_mb[lane_id] = self._actual_charge_mb(model.weights_mb)
            self._resident_model[lane_id] = model.name
        else:
            self._staged_mb[lane_id] = self._actual_charge_mb(model.weights_mb)
        self._sync_reported_vram()

    def seed_held_components(self, lane_id: int, held_component_mb: float) -> None:
        """Give ``lane_id`` device-warm component weights it holds between jobs.

        The tenancy an idle lane carries when the component cache keeps its entries on the device rather than
        paging them to host RAM. It belongs to no job, so no dispatch's footprint includes it and no job
        boundary returns it; only an unload the parent actuates on this lane does. The lane also reports it,
        which is the parent's only view of what a slot holds beyond its checkpoint.
        """
        self._held_component_mb[lane_id] = held_component_mb
        lane = self._process_map[lane_id]
        lane.held_components = [
            HeldComponentSnapshot(
                kind="unet",
                identity=f"held-component-lane-{lane_id}",
                approx_ram_mb=held_component_mb,
            ),
        ]
        self._sync_reported_vram()

    async def pop(self, job: ImageGenerateJobPopResponse) -> None:
        """Record a popped job, exactly as the pop path hands one to the tracker."""
        await track_popped_job_async(self._job_tracker, job, time_popped=self.now)
        for lora in job.payload.loras or []:
            self._job_tracker.mark_aux_prefetched(lora.name, is_version=bool(lora.is_version), is_ti=False)
        for ti in job.payload.tis or []:
            self._job_tracker.mark_aux_prefetched(ti.name, is_version=False, is_ti=True)
        if job.id_ is not None:
            self._job_tracker.mark_job_aux_prepared_if_ready(job.id_)

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
        """Honour the commands the scheduler sent last tick: an unload gives the card back its VRAM.

        The give-back is not instantaneous where the row asks for a delay: the command is booked against the
        world clock when it is first seen and the memory returns once that instant passes, which is the shape a
        real release has (an IPC the child services between allocations, then a driver-side release of the
        block). A lane keeps its outstanding-unload flag until the memory is actually back, exactly as the
        parent's own record of the command stands until the child reports the model's new state.
        """
        for lane in self._process_map.values():
            if lane.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                self._unload_leaked.discard(lane.process_id)
            if (
                lane.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
                and lane.process_id not in self._unload_due_at
                and lane.process_id not in self._unload_leaked
            ):
                self.reclaim_commands += 1
                self._unload_due_at[lane.process_id] = self.now + self.unload_release_delay_seconds
            due_at = self._unload_due_at.get(lane.process_id)
            # A booked release lands whatever the parent's flag says by then: the command is with the child, so
            # a later flag the parent stamps on the slot does not recall the memory the driver is returning.
            if due_at is not None and self.now >= due_at:
                self._release_unloaded_lane(lane)
        self._sync_reported_vram()

    def _release_unloaded_lane(self, lane: HordeProcessInfo) -> None:
        """Give the card back what an unloading lane was holding, less whatever the child could not free."""
        self._unload_due_at.pop(lane.process_id, None)
        self.unload_releases.append((self.tick, lane.process_id))
        name = lane.loaded_horde_model_name
        if self._leak_unloaded_lane(lane):
            return
        self._resident_mb.pop(lane.process_id, None)
        self._resident_model.pop(lane.process_id, None)
        self._staged_mb.pop(lane.process_id, None)
        self._loading.pop(lane.process_id, None)
        self._transient_mb.pop(lane.process_id, None)
        self._offloaded_mb.pop(lane.process_id, None)
        # Device-warm components are returned by the same actuation and by nothing else: they outlive every
        # job boundary, so an unload the parent ordered is the only thing that gives the card them back.
        self._held_component_mb.pop(lane.process_id, None)
        lane.held_components = None
        self._granted_resident_evicted.discard(lane.process_id)
        lane.loaded_horde_model_name = None
        lane.last_control_flag = None
        lane.last_process_state = HordeProcessState.WAITING_FOR_JOB
        # A model gone from the device can no longer be a retained resident, exactly as the process
        # map's own eviction bookkeeping records it.
        lane.clear_retained_resident()
        if name is not None:
            entry = self._model_map.root.get(name)
            if entry is not None and entry.process_id == lane.process_id:
                self._model_map.root.pop(name, None)

    def _leak_unloaded_lane(self, lane: HordeProcessInfo) -> bool:
        """Keep the part of ``lane``'s copy the child could not free on the card, and say whether any stayed.

        A full free is a request: the backend drops what it can and skips a model a live reference still pins,
        and the command reports nothing about the difference. The lane goes on holding those weights and goes
        on reporting them as used, so the card is short by exactly as much as the parent believes it gained.
        The child is what closes it, by judging the unload on what the device still holds and naming the
        refusal; the parent's own refusal bookkeeping then keeps the slot recorded as VRAM-resident and out of
        a reclaim that would ask the same question again next tick.
        """
        leaked_mb = min(self.child_unload_leaks_mb, self._resident_mb.get(lane.process_id, 0.0))
        if leaked_mb <= 0.0:
            return False
        self._resident_mb[lane.process_id] = leaked_mb
        self._staged_mb.pop(lane.process_id, None)
        self._loading.pop(lane.process_id, None)
        self._transient_mb.pop(lane.process_id, None)
        self._offloaded_mb.pop(lane.process_id, None)
        self._held_component_mb.pop(lane.process_id, None)
        lane.held_components = None
        self._granted_resident_evicted.discard(lane.process_id)
        lane.last_process_state = HordeProcessState.WAITING_FOR_JOB
        self._unload_leaked.add(lane.process_id)
        self.unload_leaks.append(
            f"tick {self.tick}: lane {lane.process_id} kept {leaked_mb:.0f}MB of "
            f"{lane.loaded_horde_model_name} on the card through an unload",
        )
        self._process_map.on_vram_unload_refused(lane.process_id)
        name = lane.loaded_horde_model_name
        if name is not None and not lane.vram_unload_refused:
            # A parent that does not record the refusal has nothing to hold the slot VRAM-resident with, so
            # its map follows the command it issued rather than the device it issued it to.
            self._model_map.update_entry(
                name,
                load_state=ModelLoadState.LOADED_IN_RAM,
                process_id=lane.process_id,
            )
        self._sync_reported_vram()
        return True

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
            self._loading[info.process_id] = (name, self._actual_charge_mb(model.weights_mb))
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
            if job.payload.post_processing:
                await self._job_tracker.queue_for_post_processing(job_info)
            else:
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

    async def _drain_post_processing(self) -> None:
        """Complete pending post-processing on the hand-driven lane and pass its result to safety."""
        for job_info in list(self._job_tracker.jobs_pending_post_processing):
            await self._job_tracker.begin_post_processing(job_info, process_id=-1, process_launch_identifier=1)
            await self._job_tracker.queue_for_safety_post_processed(job_info)

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
            # The dispatch the scheduler just sent is the world's only source for what the child was told about
            # the device, so the message itself is read back off the lane's pipe. A run that neuters the field
            # is then indistinguishable from a worker that never populated it.
            self._dispatch_device_truth_mb[job_id] = self._dispatched_device_truth_mb(self._process_map[lanes[0]])
            self._lane_charge_at_dispatch[job_id] = self._lane_charge_mb(lanes[0])
            # Dispatch is the moment staged weights commit to VRAM, so this is where the card is charged.
            staged_mb = self._staged_mb.pop(lanes[0], None)
            loaded_now = staged_mb is not None
            if staged_mb is not None:
                if admitted.model is not None:
                    # The load is the child's first allocation for this job, so it passes the child's own
                    # shortfall arithmetic and commits only what that arithmetic let it bring onto the card.
                    staged_mb = self._child_admit_charge(
                        lane_id=lanes[0],
                        job_id=job_id,
                        model=admitted.model,
                        charge_mb=staged_mb,
                        weight_load=True,
                    )
                if self.closed_loop and self._resident_model.get(lanes[0]) not in (None, admitted.model):
                    # Weights the slot is still holding for a previous model are not displaced by a new load:
                    # the card carries both until something explicitly returns the old ones. Modelling the
                    # load as a replacement would make the eviction that runs ahead of a cross-model dispatch
                    # unfalsifiable, since the double residency it exists to prevent could never occur.
                    self._resident_mb[lanes[0]] = self._resident_mb.get(lanes[0], 0.0) + staged_mb
                else:
                    self._resident_mb[lanes[0]] = staged_mb
                self._resident_model[lanes[0]] = admitted.model
                self._sync_reported_vram()
            if self.closed_loop:
                self._begin_occupancy(admitted, lane_id=lanes[0], loaded_now=loaded_now)

    # -- closed-loop occupancy ----------------------------------------------------------------------------

    def _begin_occupancy(
        self,
        job: ImageGenerateJobPopResponse,
        *,
        lane_id: int,
        loaded_now: bool,
    ) -> None:
        """Give a dispatched job its load, sample and decode phases and charge its sampling transient.

        A job that landed on weights already committed to the device pays no load phase, which is the whole of
        what retention and same-model routing buy. The transient is charged from dispatch rather than from the
        first tick that observes the sampling window, because a window shorter than one tick would otherwise
        never be seen on the card at all; it is released when sampling ends, so the decode tail carries only
        the weights.
        """
        job_id = str(job.id_)
        model = job.model
        assert model is not None
        self.dispatch_lanes.append((model, lane_id))
        if loaded_now:
            self.weight_uploads += 1
        model_class = _model_by_name(model)
        predicted_weights_mb = model_class.weights_mb if model_class is not None else 0.0
        weights_mb = self._actual_charge_mb(predicted_weights_mb)
        baseline = self._scheduler._model_metadata.get_baseline(model)
        # Both halves of the footprint are priced off what the scheduler's own forecast predicted, so an
        # undershoot is the single factor between that forecast and what the card is really asked to hold.
        predicted_peak_mb = predict_job_sampling_vram_mb(job, baseline) or predicted_weights_mb
        transient_mb = self._actual_charge_mb(max(0.0, predicted_peak_mb - predicted_weights_mb))
        megapixels = float(job.payload.width * job.payload.height * (job.payload.n_iter or 1)) / (1024.0 * 1024.0)
        load_seconds = 0.0 if not loaded_now else (weights_mb / 1024.0) * _LOAD_SECONDS_PER_GB
        per_step = _SAMPLE_SECONDS_PER_STEP_PER_MEGAPIXEL.get(
            str(baseline),
            _DEFAULT_SAMPLE_SECONDS_PER_STEP_PER_MEGAPIXEL,
        )
        sample_seconds = max(0.0, float(job.payload.ddim_steps or 0)) * per_step * megapixels
        decode_seconds = (
            _DECODE_SECONDS_PER_MEGAPIXEL.get(str(baseline), _DEFAULT_DECODE_SECONDS_PER_MEGAPIXEL) * megapixels
        )
        # The activation is the child's second allocation for this job, so it passes the same shortfall
        # arithmetic the load did, against a believed free the load has already eaten into. It cannot be
        # streamed, so what relief it gets comes out of weights, and every megabyte of weights left off the
        # card is fetched across the bus on each step that needs it.
        transient_mb = self._child_admit_charge(
            lane_id=lane_id,
            job_id=job_id,
            model=model,
            charge_mb=transient_mb,
            weight_load=False,
        )
        offloaded_mb = self._offloaded_mb.get(lane_id, 0.0)
        if offloaded_mb > 0.0 and weights_mb > 0.0:
            sample_seconds *= 1.0 + _OFFLOAD_SAMPLING_PENALTY * min(1.0, offloaded_mb / weights_mb)
        sample_from = self.now + load_seconds
        sample_until = sample_from + sample_seconds
        occupancy = _SlotOccupancy(
            job_id=job_id,
            lane_id=lane_id,
            model=model,
            weights_mb=self._resident_mb.get(lane_id, weights_mb),
            transient_mb=transient_mb,
            sample_from=sample_from,
            sample_until=sample_until,
            decode_until=sample_until + decode_seconds,
            transient_charged=True,
        )
        self._occupancy[job_id] = occupancy
        self._transient_mb[lane_id] = self._transient_mb.get(lane_id, 0.0) + occupancy.transient_mb
        self._sync_reported_vram()

    async def _advance_occupancy(self) -> None:
        """Advance each held lane through its phases, crediting duty and completing the jobs that finished."""
        window_start = self.now - self.tick_seconds
        for occupancy in list(self._occupancy.values()):
            sampled_until = min(self.now, occupancy.sample_until)
            self.sampling_slot_seconds += max(0.0, sampled_until - max(window_start, occupancy.sample_from))
            if occupancy.transient_charged and self.now >= occupancy.sample_until:
                self._release_transient(occupancy)
            if self.now >= occupancy.decode_until:
                await self._complete_occupancy(occupancy)
        self._sync_reported_vram()

    def _release_transient(self, occupancy: _SlotOccupancy) -> None:
        """Give the card back the activation a finished sampling window was holding."""
        if not occupancy.transient_charged:
            return
        occupancy.transient_charged = False
        remaining = self._transient_mb.get(occupancy.lane_id, 0.0) - occupancy.transient_mb
        if remaining > 1.0:
            self._transient_mb[occupancy.lane_id] = remaining
        else:
            self._transient_mb.pop(occupancy.lane_id, None)

    async def _complete_occupancy(self, occupancy: _SlotOccupancy) -> None:
        """Finish one job: hand it onward, settle its retention grant, and apply the grant to the card.

        The settle mirrors the parent's own result path, which resolves the grant the dispatch stamped on the
        slot into its retained-resident record. What the card then does follows from that record: a granted
        model's weights stay committed to the device for the next job to reuse, and an ungranted one is
        returned by the child's explicit end-of-job evictor, leaving the checkpoint in the child's RAM cache
        for a reload. Under the legacy unload regime the child returns the card regardless of what was asked,
        so the device loses the weights whatever the record says.
        """
        job_id = occupancy.job_id
        self._occupancy.pop(job_id, None)
        self._dispatch_device_truth_mb.pop(job_id, None)
        self._lane_charge_at_dispatch.pop(job_id, None)
        # A streaming decision belongs to the job that made it: the next job on this lane runs the shortfall
        # arithmetic afresh against whatever the card then holds.
        self._offloaded_mb.pop(occupancy.lane_id, None)
        self._release_transient(occupancy)
        job = next((tracked for tracked in self._job_tracker.jobs_in_progress if str(tracked.id_) == job_id), None)
        if job is None:
            return
        job_info = HordeJobInfo(
            sdk_api_job_info=job,
            job_image_results=[HordeImageResult(image_bytes=b"raw")],
            state=GENERATION_STATE.ok,
            censored=False,
            time_popped=self.now,
        )
        if job.payload.post_processing:
            await self._job_tracker.queue_for_post_processing(job_info)
        else:
            await self._job_tracker.queue_for_safety(job_info)
        self.completed_jobs += 1
        self._dispatched_at.pop(job_id, None)
        self._lane_of.pop(job_id, None)
        lane = self._process_map.get(occupancy.lane_id)
        if lane is None:
            return
        # The parent's own result path, in its order: a result for a job the slot still owns resolves that
        # job's grant, the slot's in-flight sampling stamps retire, and the slot gives up its execution
        # ownership. A slot that kept the ownership record would keep reading as the executor of a job that
        # has finished, and dispatch selection would pass it over for the rest of the run.
        owned = lane.current_inference_job()
        if owned is None or str(owned.id_) == job_id:
            lane.settle_retention_after_job()
        lane.current_inference_started_at = None
        lane.current_first_step_at = None
        lane.current_job_expected_sampling_seconds = None
        lane.retire_inference_ownership(job)
        # A finished child returns to WAITING_FOR_JOB whether or not its model stayed resident. That is not
        # cosmetic: PRELOADED_MODEL reads as busy, and a busy slot is refused by every eviction actuator, so a
        # retaining slot parked in it would hold weights nothing could ever ask back.
        lane.last_process_state = HordeProcessState.WAITING_FOR_JOB
        lane.last_control_flag = None
        retained = lane.retained_resident_model == occupancy.model and not self._legacy_unload_regime()
        child_evicted = occupancy.lane_id in self._granted_resident_evicted
        self._granted_resident_evicted.discard(occupancy.lane_id)
        if retained and child_evicted:
            # The grant was settled onto the slot for weights the child had already freed. The child is what
            # closes that gap: it sees the device empty at the end of a run it was told to keep resident and
            # reports the model out of VRAM, which is the same state change a parent-commanded unload sends
            # and is handled by the same production reconciliation. Without it the record stands over an
            # empty device and the rest of the session is priced, held and routed against a phantom.
            self._resident_mb.pop(occupancy.lane_id, None)
            self._resident_model.pop(occupancy.lane_id, None)
            self._staged_mb[occupancy.lane_id] = occupancy.weights_mb
            self._process_map.on_model_vram_clear(occupancy.lane_id)
            if lane.retained_resident_model is None:
                self._model_map.update_entry(
                    occupancy.model,
                    load_state=ModelLoadState.LOADED_IN_RAM,
                    process_id=occupancy.lane_id,
                )
            else:
                self._model_map.update_entry(
                    occupancy.model,
                    load_state=ModelLoadState.LOADED_IN_VRAM,
                    process_id=occupancy.lane_id,
                )
            return
        if retained:
            # The explicit end-of-job evictor returns everything the grant does not cover, so a slot that
            # loaded beside stale weights is back to carrying exactly the model it retains.
            self._resident_mb[occupancy.lane_id] = occupancy.weights_mb
            self._resident_model[occupancy.lane_id] = occupancy.model
            self._model_map.update_entry(
                occupancy.model,
                load_state=ModelLoadState.LOADED_IN_VRAM,
                process_id=occupancy.lane_id,
            )
            return
        self._resident_mb.pop(occupancy.lane_id, None)
        self._resident_model.pop(occupancy.lane_id, None)
        self._staged_mb[occupancy.lane_id] = occupancy.weights_mb
        self._model_map.update_entry(
            occupancy.model,
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=occupancy.lane_id,
        )

    def _legacy_unload_regime(self) -> bool:
        """Whether the escape hatch has the child returning the card at the end of every prompt."""
        return self._scheduler._runtime_config.bridge_data.legacy_comfy_vram_unload is True

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
        while len(self._inference_lanes()) > target_count:
            victim = next(
                (
                    lane
                    for lane in self._inference_lanes()
                    if lane.can_accept_job()
                    and lane.process_id != spared_process_id
                    and (protected_model is None or lane.loaded_horde_model_name != protected_model)
                ),
                None,
            )
            if victim is None:
                break
            self._retire_lane(victim.process_id)
        while len(self._inference_lanes()) < min(target_count, self._lane_ceiling):
            lane_id = max((lane.process_id for lane in self._inference_lanes()), default=-1) + 1
            self._process_map[lane_id] = make_mock_process_info(
                lane_id,
                model_name=None,
                state=HordeProcessState.WAITING_FOR_JOB,
            )
        self._sync_reported_vram()
        return len(self._inference_lanes())

    def _retire_lane(self, lane_id: int) -> None:
        """Remove a lane from the pool, releasing its context, its resident weights, and its map entry."""
        lane = self._process_map.get(lane_id)
        if lane is not None:
            self._process_map.retire_process(lane, reason="whole-card residency teardown")
        self._resident_mb.pop(lane_id, None)
        self._resident_model.pop(lane_id, None)
        self._staged_mb.pop(lane_id, None)
        self._loading.pop(lane_id, None)
        self._transient_mb.pop(lane_id, None)
        self._offloaded_mb.pop(lane_id, None)
        self._held_component_mb.pop(lane_id, None)
        self._granted_resident_evicted.discard(lane_id)
        self._unload_due_at.pop(lane_id, None)
        self._unload_leaked.discard(lane_id)
        self._drop_occupancy_on(lane_id)
        if lane is None:
            return
        name = lane.loaded_horde_model_name
        if name is not None:
            entry = self._model_map.root.get(name)
            if entry is not None and entry.process_id == lane_id:
                self._model_map.root.pop(name, None)

    def _drop_occupancy_on(self, lane_id: int) -> None:
        """Forget the hold a lane that has gone away was carrying, so no later tick completes its job."""
        self._granted_resident_evicted.discard(lane_id)
        for job_id, occupancy in list(self._occupancy.items()):
            if occupancy.lane_id == lane_id:
                self._occupancy.pop(job_id, None)
                self._dispatch_device_truth_mb.pop(job_id, None)
                self._lane_charge_at_dispatch.pop(job_id, None)

    # -- disturbances -------------------------------------------------------------------------------------

    def kill_lane_holding(self, model: _ModelClass) -> bool:
        """Kill the lane holding (or loading) ``model`` and replace it with a fresh empty lane.

        Mirrors the lifecycle's replacement of a dead inference process: the dead lane leaves the map (taking
        its model-map entry and its VRAM with it) and a new, empty lane of the same id takes its place.

        Returns:
            True when a matching lane was replaced; False when the requested disturbance had no target.
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
            return False
        self._resident_mb.pop(victim, None)
        self._resident_model.pop(victim, None)
        self._staged_mb.pop(victim, None)
        self._loading.pop(victim, None)
        self._transient_mb.pop(victim, None)
        self._offloaded_mb.pop(victim, None)
        self._held_component_mb.pop(victim, None)
        self._granted_resident_evicted.discard(victim)
        self._unload_due_at.pop(victim, None)
        self._unload_leaked.discard(victim)
        self._drop_occupancy_on(victim)
        entry = self._model_map.root.get(model.name)
        if entry is not None and entry.process_id == victim:
            self._model_map.root.pop(model.name, None)
        replacement = make_mock_process_info(victim, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        self._process_map[victim] = replacement
        self._sync_reported_vram()
        return True

    def evict_idle_resident_sibling(self, *, except_model: _ModelClass) -> bool:
        """Evict an idle sibling and report whether the requested disturbance changed card state."""
        for lane in self._process_map.values():
            name = lane.loaded_horde_model_name
            if (
                name is None
                or name == except_model.name
                or lane.last_process_state == HordeProcessState.INFERENCE_STARTING
            ):
                continue
            self._resident_mb.pop(lane.process_id, None)
            self._resident_model.pop(lane.process_id, None)
            self._staged_mb.pop(lane.process_id, None)
            lane.loaded_horde_model_name = None
            lane.last_process_state = HordeProcessState.WAITING_FOR_JOB
            lane.clear_retained_resident()
            entry = self._model_map.root.get(name)
            if entry is not None and entry.process_id == lane.process_id:
                self._model_map.root.pop(name, None)
            self._sync_reported_vram()
            return True
        return False

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

    def _evaluate_device_free_governor(self) -> None:
        """Sample the card, debounce the governor, and drive the verified reclaim ladder for one tick.

        Mirrors ``HordeWorkerProcessManager._evaluate_device_free_governor``, which is where the parent folds
        the truthful device-free figure through the per-device state machine, pushes the resulting growth hold
        and committed state into the scheduler, and advances the card's reclaim episode. Kept in the same drive
        order (debounce, state commit, ladder tick) so a change to the manager's sequence is findable from
        here; the manager's own metrics recording and its multi-card loop are the parts left out, since this
        world models one card and reads its verdicts from state rather than from records.

        The per-step floor latch the manager can force the ladder with is absent: it is set from a sampling
        slot's crawling heartbeats, which this world does not model, so the ladder here is SATURATED-driven
        only. The stranded-lane and stranded-reduction backstops are likewise the manager's, not the
        governor's, and are left to the suites that drive a manager.
        """
        device_free_mb = self.device_free_mb()
        sample = self._governor.observe(0, device_free_mb=device_free_mb, total_vram_mb=self.card.total_mb)
        self.governor_states.append(sample.state)
        self._scheduler.set_vram_growth_hold(0, sample.state in (GovernorState.PRESSURE, GovernorState.SATURATED))
        self._scheduler.set_governor_state(0, sample.state)
        if sample.state is GovernorState.HEALTHY:
            if self._healthy_since is None:
                self._healthy_since = self.now
        else:
            self._healthy_since = None
        self._reclaim_ladder.on_tick(
            0,
            saturated=sample.state is GovernorState.SATURATED,
            healthy=sample.state is GovernorState.HEALTHY,
            device_free_mb=device_free_mb,
            actuator=self._actuator,
            ladder_builder=lambda: build_reclaim_ladder(self._scheduler.build_reclaim_ladder_candidates(0)),
            context_restore_ready=self._context_restore_ready(),
            now=self.now,
        )

    def _context_restore_ready(self) -> bool:
        """Whether a reclaim episode may regrow the pool a context reduction shrank.

        The manager's ``_context_restore_ready`` on this world's clock: a card continuously HEALTHY for the
        dwell, and no head of queue still parked on the demand the reduction was made for.
        """
        if self._healthy_since is None:
            return False
        if (self.now - self._healthy_since) < HordeWorkerProcessManager._CONTEXT_RESTORE_DWELL_SECONDS:
            return False
        return not self._scheduler.head_of_queue_is_parked()

    async def step(self) -> None:
        """Advance one scheduling tick, in the control loop's order."""
        self.tick += 1
        self.now += self.tick_seconds
        self._apply_control_flags()
        self._materialise_preloads()
        if self.closed_loop:
            await self._advance_occupancy()
        else:
            await self._complete_finished_samplers()
        await self._drain_post_processing()
        await self._drain_safety()
        self._begin_arbiter_cycle()
        if self.closed_loop:
            # The parent samples the governor on its resource-monitor cadence; driven here between the frozen
            # measurement and governance, which is the single-threaded equivalent of the hold being in place
            # before the pass that would grow the card acts on it.
            self._evaluate_device_free_governor()
        self._scheduler.run_governance_tick()
        self._discharge_context_reductions()
        # The tick drives the cycle's stages directly rather than through run_scheduling_cycle, so the cycle
        # boundary is opened here: selection state scoped to one cycle must not survive the child reports this
        # tick applied, or the world presents the scheduler a staleness its own control loop never can.
        self._scheduler.begin_scheduling_cycle()
        self._scheduler.preload_models()
        self._begin_started_preloads()
        await self._dispatch_until_full()
        self.min_device_free_mb = min(self.min_device_free_mb, self.device_free_mb())
        self._sample_retained_resident_divergence()
        self.offers[self.tick] = self.advertised_models()
        claim = self._scheduler.whole_card_pop_claim()
        if claim is not None:
            self.claim_ticks.append(self.tick)
            self.claim_expires_at = claim.expires_at
        elif self.claim_ticks and self.claim_released_at == 0.0:
            self.claim_released_at = self.now

    def _sample_retained_resident_divergence(self) -> None:
        """Record any idle slot whose retained-resident record does not match what the card holds."""
        busy_lanes = {occupancy.lane_id for occupancy in self._occupancy.values()}
        for lane in self._process_map.values():
            model = lane.retained_resident_model
            if model is None or lane.process_id in busy_lanes:
                continue
            if self._resident_model.get(lane.process_id) != model:
                self.retained_resident_divergences.append((self.tick, lane.process_id, model))

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
    def reclaim_ladder(self) -> VerifiedReclaimLadder:
        """The verified reclaim engine a closed-loop tick drives, so a row can read its counters back."""
        return self._reclaim_ladder

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

    def retained_residents(self) -> dict[int, str]:
        """The parent's authoritative record of which slot holds which model's weights between jobs."""
        return {
            lane.process_id: lane.retained_resident_model
            for lane in self._process_map.values()
            if lane.retained_resident_model is not None
        }

    def phantom_model_records(self) -> list[str]:
        """Models the parent's model map records on a lane that has since been loaded over.

        A lane holds one model, so a record naming a lane whose loaded model is a different one describes
        weights nothing holds. The preload pass reads that map as part of its already-loaded set, so such a
        record is what makes a displaced model's pending job look served.
        """
        return [
            f"{name} recorded on lane {info.process_id}, which holds {lane.loaded_horde_model_name}"
            for name, info in self._model_map.root.items()
            if (lane := self._process_map.get(info.process_id)) is not None
            and lane.loaded_horde_model_name is not None
            and lane.loaded_horde_model_name != name
        ]

    def unload_refused_lanes(self) -> list[int]:
        """Lanes the parent has recorded as having refused an unload, so reclaim passes them over."""
        return [lane.process_id for lane in self._process_map.values() if lane.vram_unload_refused]

    def ram_recorded_over_resident_weights(self) -> list[str]:
        """Models the parent records in host RAM while the card is still holding weights charged to their lane.

        The ledger error a refused unload produces: the parent counts the freed room, admits against it, and
        the card pages. Every entry here is room the worker believes it has and does not.
        """
        return [
            f"{name} recorded in RAM on lane {info.process_id}, which still holds "
            f"{self._resident_mb.get(info.process_id, 0.0):.0f}MB on the card"
            for name, info in self._model_map.root.items()
            if info.horde_model_load_state is ModelLoadState.LOADED_IN_RAM
            and info.process_id is not None
            and self._resident_mb.get(info.process_id, 0.0) > 0.0
        ]

    def vram_resident_lanes(self, model: str) -> list[int]:
        """The lanes currently carrying a committed VRAM copy of ``model``'s weights."""
        return [lane_id for lane_id, name in self._resident_model.items() if name == model]

    def ram_staged_lanes(self, model: str) -> list[int]:
        """The lanes holding ``model`` staged in the child's RAM cache, awaiting a dispatch to commit it."""
        return [
            lane_id
            for lane_id in self._staged_mb
            if (lane := self._process_map.get(lane_id)) is not None and lane.loaded_horde_model_name == model
        ]

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
    # The safety placement is stateful rather than pinned: a pause that never takes effect makes the ladder's
    # deepest rung a one-shot no-op, which would hide what repeatedly reaching it costs.
    lifecycle.is_safety_gpu_paused = False
    lifecycle.safety_pause_owner = None
    lifecycle.safety_placement_transition_pending = False
    lifecycle.pause_safety_on_gpu = world.pause_safety_on_gpu
    lifecycle.restore_safety_on_gpu = world.restore_safety_on_gpu
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
