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
from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime
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
from horde_worker_regen.process_management.lifecycle.process_lifecycle import (
    SAFETY_READINESS_LATENCY_FLOOR_SECONDS,
    PauseOwner,
    ProcessLifecycleManager,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.lru_cache import LRUCache
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb
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
from horde_worker_regen.process_management.resources.run_metrics import (
    DecisionEvent,
    DecisionKind,
    DecisionVerdict,
    FlatScalarMap,
)
from horde_worker_regen.process_management.resources.vram_arbiter import MeasuredVramSnapshot
from horde_worker_regen.process_management.scheduling.governance.whole_card import offer_under_pop_claim
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _SAFETY_GPU_LOAD_CHARGE_MB,
    InferenceScheduler,
)
from horde_worker_regen.process_management.scheduling.slot_duty import SlotDutyBucket
from tests.process_management.conftest import (
    make_mock_bridge_data,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_card_runtimes,
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

_CLOCK_EPSILON = 1e-6
"""Slack on a world-clock comparison, so an instant reached by accumulating tick advances still compares
equal to the same instant computed as a multiple of the tick."""

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
_CARD_10GB = _CardClass("10gb", 10240.0)
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
class _PendingPreload:
    """One preload the parent has sent, between the command and the staged weights.

    The two instants are separate because the parent's record and the slot's own state are separate: the send
    is optimistic, so the model map reads as loading while the lane still reports itself idle and empty, and
    only once the child reports does the lane read as busy. A shrink that runs in the first window sees a free
    slot; one that runs in the second sees a busy one.
    """

    model: str
    weights_mb: float
    report_at: float
    """When the child first reports ``PRELOADING_MODEL``, ending the window the slot reads idle in."""
    ready_at: float
    """When the weights are staged and the lane reports ``PRELOADED_MODEL``."""


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
# Per-tick dispatch observation: what the world could see about why nothing was dispatched
# --------------------------------------------------------------------------------------------------------

_CYCLING_LANE_STATES = frozenset(
    {
        HordeProcessState.PROCESS_STARTING,
        HordeProcessState.PROCESS_ENDING,
        HordeProcessState.PROCESS_ENDED,
        HordeProcessState.DOWNLOADING_MODEL,
        HordeProcessState.PRELOADING_MODEL,
    },
)
"""Lane states that mean the pool is mid-transition rather than sitting on its hands.

A lane in one of these is on its way to being able to take work (or on its way out), so a tick where one
stands is a tick the card is legitimately between things rather than idle against work it could seat."""


@dataclass(frozen=True)
class _FittingJob:
    """One pending job the card has the room to seat right now, priced the way admission prices it."""

    job_id: str
    model: str
    priced_mb: float

    def __str__(self) -> str:
        """Name the job, its model and its price, for a failing oracle's message."""
        return f"{self.job_id[:8]} ({self.model}, {self.priced_mb:.0f}MB)"


@dataclass(frozen=True)
class _ProtectedDispatchHold:
    """One dispatch declined on behalf of some other entity, with what the world can see of that entity.

    The hold classes the scheduler discloses are read from its own records (the dispatch decision sink for
    head protection, the stall classifier's bucket for the rest), never re-derived here. What this adds is
    the other half the disclosure does not carry: whether the entity the hold is being kept for is itself
    moving toward being served, and the name of the bounded grace it sits inside when it is not.
    """

    kind: str
    """Which disclosed hold class this is."""
    held_subject: str
    """The ready job whose dispatch was declined."""
    protected: str
    """The job or model the hold is being kept for."""
    progressing: bool
    """Whether the protected entity is itself making progress toward being served."""
    grace: str | None
    """The bounded, disclosed grace the protected entity sits inside, or None when it is inside none."""
    detail: str
    """What the world saw of the protected entity, for a failing oracle's message."""

    def __str__(self) -> str:
        """Name both ends of the hold, for a failing oracle's message."""
        return f"{self.kind}: held {self.held_subject[:8]} for {self.protected} ({self.detail})"


@dataclass(frozen=True)
class _TickObservation:
    """What one tick looked like to everything that judges whether the card was earning.

    Recorded at the close of every tick so a verdict over a run can ask about a span of ticks rather than
    about the run's average, which is what lets a hundred seconds of idle be distinguished from a duty figure
    it barely moves.
    """

    tick: int
    now: float
    device_free_mb: float
    jobs_in_progress: int
    fitting_pending: tuple[_FittingJob, ...]
    """Pending jobs the card's free VRAM covers at the price admission would charge them."""
    grace_reasons: tuple[str, ...]
    """Named, bounded reasons the world itself recognises for a tick with no dispatch on it."""
    head_job_id: str | None
    head_model: str | None
    head_stall_bucket: str | None
    """The scheduler's own duty-bucket attribution for the parked head, as its stall classifier named it."""
    head_stall_reason: str | None
    protected_holds: tuple[_ProtectedDispatchHold, ...]

    @property
    def idle(self) -> bool:
        """Whether the card held no dispatched work at the close of this tick."""
        return self.jobs_in_progress == 0

    def hold_summary(self) -> str:
        """Every hold reason recorded on this tick, as one line."""
        parts = [f"stall={self.head_stall_bucket}" if self.head_stall_bucket is not None else "stall=none"]
        parts.extend(str(hold) for hold in self.protected_holds)
        if self.grace_reasons:
            parts.append("grace=" + "|".join(self.grace_reasons))
        return "; ".join(parts)


@dataclass(frozen=True)
class _DecisionRecord:
    """One verdict the scheduler recorded through its decision sink, stamped with the tick it landed on."""

    tick: int
    decision_kind: DecisionKind
    subject: str
    verdict: DecisionVerdict
    reason: str
    inputs: dict[str, object]


_GOVERNOR_SEVERITY = {
    GovernorState.HEALTHY: 0,
    GovernorState.PRESSURE: 1,
    GovernorState.SATURATED: 2,
}
"""How bad each governor state is, so a pool-wide verdict can name its worst card's state."""

_HEAD_PROTECTION_REASON_MARKER = "held for the head"
"""The arbiter's own words for a non-head request declined so the head keeps the room it needs.

Matched rather than re-derived so the oracle reads the same verdict the dispatch path acted on; the arbiter
composes this text in one place (:meth:`VramArbiter._head_protection_defer`)."""


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
        safety_readiness_seconds: float = 0.0,
        safety_load_transient_mb: float = 0.0,
        unload_release_delay_seconds: float = 0.0,
        child_evicts_granted_resident: bool = False,
        child_unload_leaks_mb: float = 0.0,
        preload_report_latency_seconds: float = 0.0,
        preload_latency_seconds: float = 0.0,
        disaggregated_encode_seconds: float = 0.0,
        card_max_pixels: dict[int, int] | None = None,
        safety_card_index: int = 0,
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
            safety_readiness_seconds: How long a safety placement flip takes to reach readiness in world time.
                A flip ends the safety process and brings a replacement up, so the card is mid-rebuild for that
                whole window and the lifecycle reports a placement transition pending throughout it. Zero
                collapses the flip to an instant, which no worker's safety process ever does, and leaves the
                placement policy free to decide its next flip on the cycle after the last one.
            safety_load_transient_mb: How much more than its at-rest footprint safety charges the card while a
                restore is materialising. The classifier weights are read and copied before the process settles,
                so the peak a restore imposes is above the figure a fit priced, and a restore onto a card with
                barely enough room re-trips the pressure it just left. Zero prices the load at its at-rest
                footprint.
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
            preload_report_latency_seconds: How long a lane goes on reporting itself idle and empty after the
                parent has sent it a preload. The send is optimistic: the parent's model map records the load
                immediately, while the slot itself does not move until the child wakes and reports
                ``PRELOADING_MODEL``. In that window the slot carries no model name and reads as idle, which
                is the window a shrink can take a slot the parent has already committed to. Zero collapses it,
                so the lane reports the load on the same tick the command was sent.
            preload_latency_seconds: How long a lane stays in ``PRELOADING_MODEL`` before its weights are
                staged. Zero materialises the load on the tick after the report, so a load is never observable
                mid-flight by a governance pass; a value of at least one tick puts the loading state inside a
                pass's view.
            disaggregated_encode_seconds: How long a disaggregation-class sampler waits for the encode lane
                after it is dispatched. The sampler is pinned and owns the job from dispatch, but its child
                reports nothing until the sample stage runs, so through this window the slot reads
                ``WAITING_FOR_JOB`` while owning a dispatched job. Zero samples immediately, which is the
                monolithic shape and what the rows that do not vary this run against.
            card_max_pixels: Per-card ``max_pixels`` for a multi-card pool, keyed by device index. The lanes
                are pinned round-robin over those indices and the scheduler is handed a per-card runtime whose
                effective config differs only in this ceiling, so a job's resolution alone decides which cards
                may serve it. Each index also gets its own entry in the VRAM ledger, so the cards are
                independent memory domains: every card carries its own total, its own committed tenants and its
                own measured free reading. None keeps the single card every other row runs on, where routing is
                a strict no-op and the ledger holds one entry.
            safety_card_index: The card the on-GPU safety process is pinned to, which is the card its charge
                lands on and the card its placement policy reasons about. The lowest index is the historical
                fixed pin and what every single-card row runs on.
        """
        self.card = card
        self.tick_seconds = tick_seconds
        self.safety_readiness_seconds = safety_readiness_seconds
        self.safety_load_transient_mb = safety_load_transient_mb
        self.closed_loop = closed_loop
        self.child_free_view_lie_mb = child_free_view_lie_mb
        self.footprint_undershoot = footprint_undershoot
        self.unload_release_delay_seconds = unload_release_delay_seconds
        self.child_evicts_granted_resident = child_evicts_granted_resident
        self.child_unload_leaks_mb = child_unload_leaks_mb
        self.preload_report_latency_seconds = preload_report_latency_seconds
        self.preload_latency_seconds = preload_latency_seconds
        self.disaggregated_encode_seconds = disaggregated_encode_seconds
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
        self._loading: dict[int, _PendingPreload] = {}
        """Per-lane preloads the parent has commanded and the child has not finished, keyed by lane."""
        self._encode_until: dict[int, float] = {}
        """Per-lane instant a pinned disaggregated sampler stops waiting on the encode lane and starts sampling.

        While one stands the lane owns its dispatched job and still reports itself idle, which is the state a
        victim selector reading child state alone cannot tell apart from a free slot."""
        self.committed_slot_retirements: list[str] = []
        """Every retirement of a lane the parent had already committed to, as the structural oracle read it."""
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
        self._safety_transition_until: float | None = None
        """World-clock instant the safety placement rebuild in flight reaches readiness, or None when none is.

        While one stands the lifecycle reports a placement transition pending, exactly as it does across a real
        respawn, so a policy that gathers evidence through the rebuild window is visible here."""
        self.safety_readiness_events: list[tuple[int, float]] = []
        """Every time a safety placement rebuild reached readiness, as (tick, world clock)."""
        self.snapshot: MeasuredVramSnapshot | None = None
        """The most recent cycle-frozen device measurement, the surface a row reads obligations back from."""
        self.decision_records: list[_DecisionRecord] = []
        """Every verdict the scheduler disclosed through its decision sink, stamped with its tick.

        The worker's own disclosure surface, wired here rather than re-derived: a hold the scheduler does not
        record is a hold an operator cannot see either, so an oracle that reads this is held to the same
        evidence a post-mortem has."""
        self.tick_observations: list[_TickObservation] = []
        """What each tick looked like to the verdicts that judge whether the card was earning."""

        self._service_contexts = service_contexts
        card_indices = sorted(card_max_pixels) if card_max_pixels else [0]
        self._card_totals: dict[int, float] = dict.fromkeys(card_indices, card.total_mb)
        """The conserved ledger, one entry per card: what that card's total is, and therefore what its free
        reading is once the tenants charged to it are taken off. Cards are independent memory domains, so
        every figure derived from the ledger is derived per index; a single-card row holds one entry and every
        such derivation collapses to the whole-pool figure it was before."""
        self._safety_card_index = safety_card_index if safety_card_index in self._card_totals else card_indices[0]
        """The card safety's charge lands on, and the card its placement policy is told it occupies."""
        self._lane_cards = {lane_id: card_indices[lane_id % len(card_indices)] for lane_id in range(lane_count)}
        processes: dict[int, HordeProcessInfo] = {}
        for lane_id in range(lane_count):
            lane = make_mock_process_info(
                lane_id,
                model_name=None,
                state=HordeProcessState.WAITING_FOR_JOB,
                device_index=self._lane_cards[lane_id],
            )
            processes[lane_id] = lane
        if service_contexts:
            processes[_SAFETY_PROCESS_ID] = make_mock_process_info(
                _SAFETY_PROCESS_ID,
                model_name=None,
                process_type=HordeProcessType.SAFETY,
                state=HordeProcessState.WAITING_FOR_JOB,
                device_index=self._safety_card_index,
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
        # The lifecycle's victim selection over this world's pool, and nothing else of the lifecycle: the
        # selector reads only the process map and the tracker, while a constructed manager would want a
        # multiprocessing context, live queues and real child pipes, and would answer the scheduler's
        # placement and lane predicates from a config these rows do not set. Wiring the two collaborators the
        # selection actually reads is what puts the shrink's choice in production's hands without standing a
        # whole manager up around it.
        self._scale_down_selector = ProcessLifecycleManager.__new__(ProcessLifecycleManager)
        self._scale_down_selector._process_map = self._process_map
        self._scale_down_selector._job_tracker = self._job_tracker

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
        card_runtimes: dict[int, CardRuntime] = {}
        if card_max_pixels:
            for device_index, max_pixels in sorted(card_max_pixels.items()):
                card_config = make_mock_bridge_data(
                    max_threads=max_threads,
                    queue_size=queue_depth,
                    image_models_to_load=[model.name for model in _MODEL_CLASSES],
                    max_pixels=max_pixels,
                )
                lanes_on_card = sum(1 for card in self._lane_cards.values() if card == device_index)
                card_runtimes.update(
                    make_test_card_runtimes(
                        device_indices=(device_index,),
                        config=card_config,
                        total_vram_mb=card.total_mb,
                        target_process_count=max(1, lanes_on_card),
                        max_concurrent_inference=max_threads,
                    ),
                )
        self._card_runtimes: dict[int, CardRuntime] | None = card_runtimes or None
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
            card_runtimes=self._card_runtimes,
            max_concurrent_inference_processes=max_threads,
            max_inference_processes=lane_count,
            lru=LRUCache(max(2, lane_count)),
            reserve_ledger=self._reserve_ledger,
            decision_sink=self._record_decision,
            clock=lambda: self.now,
        )
        if disaggregated:
            # Class-eligibility is the scheduler's own seam for "this job will run as a UNet-only sampler".
            # Pinning it is what makes a row's jobs priced, admitted, and dispatched on the disaggregated
            # path without standing up the orchestrator's lanes, which these rows do not vary.
            self._scheduler._is_disaggregation_class_eligible = lambda _job: True  # type: ignore[method-assign]
        self._scheduler.set_device_free_mb_provider(self.device_free_mb)
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
        self._healthy_since: dict[int, float] = {}
        self.governor_states_by_card: dict[int, list[GovernorState]] = {index: [] for index in self._card_totals}
        """Each card's committed governor state at every tick of a closed-loop run, in order."""
        self.ladder_actuations: list[_LadderActuation] = []
        """Every reclaim rung the ladder actually performed, in the order it performed them."""
        self.min_card_free_mb: dict[int, float] = {index: self.card_free_mb(index) for index in self._card_totals}
        """Per card, the lowest device-free reading it ever showed, sampled once per tick after the card moves."""
        self.sampling_slot_seconds = 0.0
        """Slot-seconds spent sampling: the numerator of the run's duty figure."""
        self.sampling_slot_seconds_by_card: dict[int, float] = dict.fromkeys(self._card_totals, 0.0)
        """Per card, the slot-seconds its own lanes spent sampling, so duty is a claim about each card.

        A pool where one card earns for both would hold any worker-wide duty floor while a card sits idle."""
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

    def _card_of(self, process_id: int) -> int:
        """The card a process's charges land on, whether or not the process is still in the pool.

        A lane the pool has retired keeps no map entry, so its pinning is read from the row's own round-robin
        rather than from the map; nothing a retired lane held may quietly move to another card's ledger.
        """
        process = self._process_map.get(process_id)
        if process is not None:
            return process.device_index
        return self._lane_cards.get(process_id, self._safety_card_index)

    def _context_charge_mb(self, device_index: int) -> float:
        """One card's total context cost: the one-time runtime, each further context, and safety's weights.

        Every card pays its own one-time runtime allocation: a context on one device buys nothing on another.
        The safety process carries resident classifier weights on top of its context, so it is charged its
        whole-process figure (the one the scheduler also prices it at) rather than a bare context, and only to
        the card it is pinned to; the post-processing lane holds a context and no at-rest model, so it costs
        the marginal on its own card. Safety paused off the card costs it nothing, which is the whole of what
        the reclaim rung that moves it buys.
        """
        lanes = max(1, sum(1 for lane in self._inference_lanes() if lane.device_index == device_index))
        charge = _FIRST_CONTEXT_MB + _MARGINAL_CONTEXT_MB * (lanes - 1)
        if self._service_contexts:
            if self._card_of(_POST_PROCESS_LANE_ID) == device_index:
                charge += _MARGINAL_CONTEXT_MB
            if self._safety_card_index == device_index and not self.safety_is_off_gpu():
                charge += _SAFETY_GPU_LOAD_CHARGE_MB
                if self._safety_transition_until is not None:
                    # A restore is still materialising: the classifier weights are being read and copied, so the
                    # card carries the load peak rather than the at-rest footprint until the process settles.
                    charge += self.safety_load_transient_mb
        return charge

    def safety_is_off_gpu(self) -> bool:
        """Whether the safety context is currently paused off the card."""
        return bool(self._lifecycle.is_safety_gpu_paused)

    def pause_safety_on_gpu(self, *, owner: PauseOwner) -> bool:
        """Take the safety context off the card for ``owner``, returning whether this call did it.

        Mirrors the lifecycle's single-owner pause: a context already off the card is not paused twice, and the
        owner is recorded so only that owner's restore can bring it back.
        """
        if self.safety_is_off_gpu() or self._safety_transition_until is not None:
            return False
        self._lifecycle.is_safety_gpu_paused = True
        self._lifecycle.safety_pause_owner = owner
        self.safety_pause_events.append((self.tick, self.now, owner))
        self._begin_safety_placement_transition()
        self._sync_reported_vram()
        return True

    def restore_safety_on_gpu(self, *, owner: PauseOwner) -> bool:
        """Put the safety context back on the card for ``owner``, returning whether this call did it."""
        if not self.safety_is_off_gpu() or self._lifecycle.safety_pause_owner is not owner:
            return False
        if self._safety_transition_until is not None:
            return False
        self._lifecycle.is_safety_gpu_paused = False
        self._lifecycle.safety_pause_owner = None
        self.safety_restore_events.append((self.tick, self.now))
        self._begin_safety_placement_transition()
        self._sync_reported_vram()
        return True

    def _begin_safety_placement_transition(self) -> None:
        """Open the rebuild window one placement flip costs, if the row models one at all."""
        if self.safety_readiness_seconds <= 0:
            return
        self._safety_transition_until = self.now + self.safety_readiness_seconds
        self._lifecycle.safety_placement_transition_pending = True

    def _advance_safety_placement_transition(self) -> None:
        """Close the rebuild window once the replacement safety process has had time to reach readiness."""
        until = self._safety_transition_until
        if until is None or self.now < until:
            return
        self._safety_transition_until = None
        self._lifecycle.safety_placement_transition_pending = False
        self.safety_readiness_events.append((self.tick, self.now))
        self._sync_reported_vram()

    def safety_readiness_latency_seconds(self) -> float:
        """What the lifecycle reports one placement flip costs, floored the way the real manager floors it."""
        return max(self.safety_readiness_seconds, SAFETY_READINESS_LATENCY_FLOOR_SECONDS)

    def card_free_mb(self, device_index: int) -> float:
        """The truthful device-free reading for one card: its total less its contexts, weights and activation.

        The sampling activation term is what makes a crater representable: weights are a persistent tenant a
        static fit can price ahead of time, while a sampling window adds gigabytes for its own duration only,
        and it is the sum of the two across concurrent slots that reaches a paging cliff. Only the tenants
        charged to this card count, so work on a sibling card neither consumes this card's room nor excuses it.
        """
        held = sum(
            charge_mb
            for charges in (self._resident_mb, self._transient_mb, self._held_component_mb)
            for lane_id, charge_mb in charges.items()
            if self._card_of(lane_id) == device_index
        )
        return max(0.0, self._card_totals[device_index] - self._context_charge_mb(device_index) - held)

    def device_free_mb(self, device_index: int | None = None) -> float:
        """The truthful device-free reading for ``device_index``, or the tightest card's when none is named.

        The scheduler is handed this as its per-card measured reading. An unkeyed read answers with the
        lowest card, which is the figure a floor is about; on a single-card row it is that one card, so every
        such read is the whole-pool reading it was before the ledger was keyed.
        """
        if device_index is None:
            return min(self.card_free_mb(index) for index in self._card_totals)
        return self.card_free_mb(device_index)

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
        believed = self.card_free_mb(self._card_of(lane_id)) + self.child_free_view_lie_mb
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
        card_free_mb = self.card_free_mb(self._card_of(lane_id))
        if committed_mb > card_free_mb:
            self.child_overcommits.append(
                f"tick {self.tick}: lane {lane_id} committed {committed_mb:.0f}MB to a card with "
                f"{card_free_mb:.0f}MB really free, believing it had "
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
        for lane in self._process_map.values():
            total_mb = self._card_totals[lane.device_index]
            lane.total_vram_mb = int(total_mb)
            lane.vram_usage_mb = int(total_mb - self.card_free_mb(lane.device_index))
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
        """Book the load of any model the scheduler has just told a lane to bring in.

        The command is booked against the world clock rather than applied to the lane: a lane reports the load
        only once its report latency has passed, so the parent's optimistic map entry and the slot's own state
        can disagree for as long as a real child's wake takes.
        """
        for name, info in list(self._model_map.root.items()):
            if info.horde_model_load_state != ModelLoadState.LOADING or info.process_id is None:
                continue
            if info.process_id in self._loading or info.process_id in self._staged_mb:
                continue
            model = _model_by_name(name)
            if model is None:
                continue
            report_at = self.now + self.preload_report_latency_seconds
            self._loading[info.process_id] = _PendingPreload(
                model=name,
                weights_mb=self._actual_charge_mb(model.weights_mb),
                report_at=report_at,
                ready_at=report_at + self.preload_latency_seconds,
            )
        self._advance_preloads()

    def _advance_preloads(self) -> None:
        """Move each booked preload through its report and its materialisation as the clock passes them."""
        for lane_id, pending in list(self._loading.items()):
            lane = self._process_map.get(lane_id)
            if lane is None:
                self._loading.pop(lane_id, None)
                continue
            if self.now < pending.report_at - _CLOCK_EPSILON:
                continue
            # A load booked this tick is staged on a later one: the boundary instant still belongs to the
            # loading state, so a zero-latency load reports now and stages on the next advance, which is the
            # one-tick load every row without a latency of its own is written against.
            if self.now <= pending.ready_at + _CLOCK_EPSILON:
                if lane.last_process_state != HordeProcessState.PRELOADING_MODEL:
                    lane.last_process_state = HordeProcessState.PRELOADING_MODEL
                continue
            self._loading.pop(lane_id, None)
            self._staged_mb[lane_id] = pending.weights_mb
            lane.loaded_horde_model_name = pending.model
            lane.last_process_state = HordeProcessState.PRELOADED_MODEL
            lane.last_control_flag = None
            self._model_map.update_entry(pending.model, load_state=ModelLoadState.LOADED_IN_RAM, process_id=lane_id)
        self._sync_reported_vram()

    def _materialise_preloads(self) -> None:
        """Complete the loads whose latency has run out, staging their weights on the lane."""
        self._advance_preloads()

    def _advance_encode_windows(self) -> None:
        """Start the sample stage on every pinned sampler whose encode wait has run out.

        The point the child finally has work of its own to report, which is where the slot stops looking idle
        and starts reading busy to everything that judges it on child state.
        """
        for lane_id, until in list(self._encode_until.items()):
            if self.now < until - _CLOCK_EPSILON:
                continue
            self._encode_until.pop(lane_id, None)
            lane = self._process_map.get(lane_id)
            if lane is None:
                continue
            lane.last_process_state = HordeProcessState.INFERENCE_STARTING

    def _release_sampler_pin(self, lane_id: int) -> None:
        """Give a pinned sampler's lane back to the available pool once its job is off it."""
        self._encode_until.pop(lane_id, None)
        self._process_map.release_disaggregation_reservation(lane_id)

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
            if lane_id is not None:
                self._release_sampler_pin(lane_id)
            if lane is not None:
                # A slot that kept its execution ownership would go on reading as the executor of a finished
                # job for the rest of the run, which every selector that spares an owning slot then honours.
                # The closed-loop completion path retires it in the same place, so both fidelities agree
                # about when a lane stops owning what it was dispatched.
                lane.retire_inference_ownership(job)
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
            # The encode wait belongs to a disaggregation-class job, judged through the scheduler's own class
            # seam, so a queue that mixes classes gives it only to the jobs that run as UNet-only samplers.
            encode_seconds = (
                self.disaggregated_encode_seconds
                if self._scheduler._is_disaggregation_class_eligible(admitted)
                else 0.0
            )
            if encode_seconds > 0.0:
                # A disaggregation-class dispatch pins its sampler and grants it execution ownership before
                # the sample stage exists: the UNet waits on the encode lane, and its child reports nothing
                # in the meantime, so the slot goes on reading WAITING_FOR_JOB while owning a dispatched job.
                # The reservation is what keeps a second dispatch off it; nothing but the ownership record
                # separates it from a free slot for anything else that reads child state.
                assert self._process_map[lanes[0]].current_inference_job() is not None, (
                    "a dispatched job must leave its lane owning it"
                )
                self._process_map.reserve_for_disaggregation(lanes[0])
                self._encode_until[lanes[0]] = self.now + encode_seconds
                self._process_map[lanes[0]].last_process_state = HordeProcessState.WAITING_FOR_JOB
            else:
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
                self._begin_occupancy(
                    admitted,
                    lane_id=lanes[0],
                    loaded_now=loaded_now,
                    encode_seconds=encode_seconds,
                )

    # -- closed-loop occupancy ----------------------------------------------------------------------------

    def _begin_occupancy(
        self,
        job: ImageGenerateJobPopResponse,
        *,
        lane_id: int,
        loaded_now: bool,
        encode_seconds: float = 0.0,
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
        # A disaggregated sampler's window opens only once the encode lane hands it the conditioning, so the
        # encode wait sits between the dispatch and the first step rather than inside the sampling figure.
        sample_from = self.now + load_seconds + encode_seconds
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
            sampled_seconds = max(0.0, sampled_until - max(window_start, occupancy.sample_from))
            self.sampling_slot_seconds += sampled_seconds
            card = self._card_of(occupancy.lane_id)
            self.sampling_slot_seconds_by_card[card] = self.sampling_slot_seconds_by_card[card] + sampled_seconds
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
        self._release_sampler_pin(occupancy.lane_id)
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

        Which lane a shrink takes is production's answer, not this world's: the disallowed set is built the
        way the lifecycle builds it (a residency's shrink protects its holder by model name through
        ``protected_model``, plus the slot the caller already committed to through ``spared_process_id``) and
        the victim itself comes from
        :meth:`~horde_worker_regen.process_management.lifecycle.process_lifecycle.ProcessLifecycleManager
        ._select_inference_process_to_scale_down`. A world-local idleness predicate would carry whatever the
        production selector's did, so it could never disagree with it, and the two defects that reached
        production are both selectors calling a committed slot idle. What stays local is the pool bookkeeping
        a chosen victim causes: retiring the lane returns its VRAM, its map entry and its context to the card.

        Growth restores empty lanes up to the pool's provisioned ceiling.
        """
        selector = self._scale_down_selector
        while len(self._inference_lanes()) > target_count:
            if protected_model is not None:
                disallowed = selector._whole_card_protected_processes(protected_model, device_index)
            else:
                # The stand-in lifecycle answers the queued-model protection with an empty list, which is the
                # pool contract these rows are written against: a lane holding a queued model is stoppable.
                disallowed = []
            if spared_process_id is not None and spared_process_id not in disallowed:
                disallowed = [*disallowed, spared_process_id]
            victim = selector._select_inference_process_to_scale_down(
                disallowed_processes=disallowed,
                pressure_shortfall_mb=pressure_shortfall_mb,
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
                device_index=self._lane_cards.get(lane_id, self._safety_card_index),
            )
        self._sync_reported_vram()
        return len(self._inference_lanes())

    def _retire_lane(self, lane_id: int) -> None:
        """Remove a lane from the pool, releasing its context, its resident weights, and its map entry."""
        lane = self._process_map.get(lane_id)
        if lane is not None:
            self._record_committed_slot_retirement(lane)
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

    def _record_committed_slot_retirement(self, lane: HordeProcessInfo) -> None:
        """Note a retirement of a slot the parent had already committed to, at the retirement itself.

        Three commitments outlive whatever the lane's child is currently reporting, and each one makes the
        slot something other than spare capacity: an owned dispatched job (a pinned sampler waiting on the
        encode lane owns its job while reporting itself idle), a load in flight, and the slot a whole-card
        pre-stage is loading its head into (recorded on the residency before any lane carries its model
        name). Reading them here rather than judging the run by its outcome is what makes the verdict name
        the seam: the alternative, a job that never finished or a residency that never converged, is several
        ticks downstream of the decision and reachable by unrelated causes.
        """
        owned = lane.current_inference_job()
        if owned is not None:
            self.committed_slot_retirements.append(
                f"tick {self.tick}: lane {lane.process_id} was retired while owning dispatched job "
                f"{str(owned.id_)[:8]} ({owned.model}), reporting {lane.last_process_state.name}",
            )
        if lane.last_process_state in (HordeProcessState.PRELOADING_MODEL, HordeProcessState.DOWNLOADING_MODEL):
            self.committed_slot_retirements.append(
                f"tick {self.tick}: lane {lane.process_id} was retired mid-load, reporting "
                f"{lane.last_process_state.name}",
            )
        for device_index, state in self._scheduler._whole_card_ledger.held():
            if state.prestage_process_id == lane.process_id:
                self.committed_slot_retirements.append(
                    f"tick {self.tick}: lane {lane.process_id} was retired while it was the pre-stage target "
                    f"for whole-card head {state.model} on device {device_index}",
                )

    def _drop_occupancy_on(self, lane_id: int) -> None:
        """Forget the hold a lane that has gone away was carrying, so no later tick completes its job."""
        self._granted_resident_evicted.discard(lane_id)
        self._release_sampler_pin(lane_id)
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
            for lane_id, pending in self._loading.items():
                if pending.model == model.name:
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
        replacement = make_mock_process_info(
            victim,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            device_index=self._card_of(victim),
        )
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
            device_free_mb_by_device={index: self.card_free_mb(index) for index in self._card_totals},
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
        here; the manager's own metrics recording is the part left out, since this world reads its verdicts
        from state rather than from records. Every card in the ledger is folded in on its own reading, its own
        total and its own reclaim episode, because a card is an independent memory domain and a state committed
        from a sibling's reading would govern work that cannot move between them.

        The per-step floor latch the manager can force the ladder with is absent: it is set from a sampling
        slot's crawling heartbeats, which this world does not model, so the ladder here is SATURATED-driven
        only. The stranded-lane and stranded-reduction backstops are likewise the manager's, not the
        governor's, and are left to the suites that drive a manager.
        """
        for device_index in sorted(self._card_totals):
            self._evaluate_one_card_governor(device_index)

    def _evaluate_one_card_governor(self, device_index: int) -> None:
        """Fold one card's reading through its governor and advance that card's reclaim episode."""
        device_free_mb = self.card_free_mb(device_index)
        sample = self._governor.observe(
            device_index,
            device_free_mb=device_free_mb,
            total_vram_mb=self._card_totals[device_index],
        )
        self.governor_states_by_card[device_index].append(sample.state)
        self._scheduler.set_vram_growth_hold(
            device_index,
            sample.state in (GovernorState.PRESSURE, GovernorState.SATURATED),
        )
        self._scheduler.set_governor_state(device_index, sample.state)
        if sample.state is GovernorState.HEALTHY:
            self._healthy_since.setdefault(device_index, self.now)
        else:
            self._healthy_since.pop(device_index, None)
        self._reclaim_ladder.on_tick(
            device_index,
            saturated=sample.state is GovernorState.SATURATED,
            healthy=sample.state is GovernorState.HEALTHY,
            device_free_mb=device_free_mb,
            actuator=self._actuator,
            ladder_builder=lambda: build_reclaim_ladder(
                self._scheduler.build_reclaim_ladder_candidates(device_index),
            ),
            context_restore_ready=self._context_restore_ready(device_index),
            now=self.now,
        )

    def _context_restore_ready(self, device_index: int) -> bool:
        """Whether a reclaim episode may regrow the pool a context reduction shrank on this card.

        The manager's ``_context_restore_ready`` on this world's clock: a card continuously HEALTHY for the
        dwell, and no head of queue still parked on the demand the reduction was made for.
        """
        healthy_since = self._healthy_since.get(device_index)
        if healthy_since is None:
            return False
        if (self.now - healthy_since) < HordeWorkerProcessManager._CONTEXT_RESTORE_DWELL_SECONDS:
            return False
        return not self._scheduler.head_of_queue_is_parked()

    async def step(self) -> None:
        """Advance one scheduling tick, in the control loop's order."""
        self.tick += 1
        self.now += self.tick_seconds
        self._advance_safety_placement_transition()
        self._apply_control_flags()
        self._materialise_preloads()
        self._advance_encode_windows()
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
        for device_index in self._card_totals:
            self.min_card_free_mb[device_index] = min(
                self.min_card_free_mb[device_index],
                self.card_free_mb(device_index),
            )
        self._sample_retained_resident_divergence()
        self.offers[self.tick] = self.advertised_models()
        claim = self._scheduler.whole_card_pop_claim()
        if claim is not None:
            self.claim_ticks.append(self.tick)
            self.claim_expires_at = claim.expires_at
        elif self.claim_ticks and self.claim_released_at == 0.0:
            self.claim_released_at = self.now
        self._observe_dispatch_tick()

    # -- dispatch observation ------------------------------------------------------------------------------

    def _record_decision(
        self,
        *,
        decision_kind: DecisionKind,
        subject: str,
        verdict: DecisionVerdict,
        reason: str = "",
        inputs: FlatScalarMap | None = None,
        timestamp: float | None = None,
    ) -> DecisionEvent | None:
        """Take the scheduler's disclosure of one verdict, stamped with the tick it landed on.

        The manager injects a recorder that coalesces repeats for the stats export; nothing here coalesces,
        because a verdict repeated on every tick of a hold is precisely what a per-tick oracle reads. Returns
        None: the scheduler ignores the return, and a run has no export to append to.
        """
        self.decision_records.append(
            _DecisionRecord(
                tick=self.tick,
                decision_kind=decision_kind,
                subject=subject,
                verdict=verdict,
                reason=reason,
                inputs=dict(inputs or {}),
            ),
        )
        return None

    def _pending_jobs_that_fit(self) -> tuple[_FittingJob, ...]:
        """Pending jobs whose priced demand the card's free VRAM covers right now.

        Priced through the scheduler's own measured-admission candidate arithmetic, with no target process,
        so the figure is the one admission would charge the job with no resident-weight credit taken: the
        conservative end, since a job that fits at full price fits at any credit. Jobs the card has already
        disclosed it cannot ever seat (the ceiling hold) and jobs still waiting on their pop-time preparation
        are not offers the card is refusing, so neither counts as work being passed over.

        A job whose model is the one a churn governor is currently deferring is left out too, and only that
        model. The deferral is a bounded brake on how fast this card may be rotated, and it is aimed at that
        head: for its dwell the head is deliberately not being served, and past the dwell it stops asking for
        the card and takes ordinary admission. What the deferral says nothing about is the rest of the queue,
        which is exactly why the work behind such a head stays counted here.
        """
        # Priced against the roomiest card, since a job the pool can seat anywhere is work the pool is
        # passing over; on a single-card row that is the one card's headroom.
        headroom_mb = max(
            self.card_free_mb(index) - admission_noise_buffer_mb(total_mb)
            for index, total_mb in self._card_totals.items()
        )
        deferred_model = self._scheduler._whole_card_ledger.governor_deferred_head(None, now=self.now)
        fitting: list[_FittingJob] = []
        for job in self._job_tracker.jobs_pending_inference:
            if job.id_ is None or job.model is None:
                continue
            if job.model == deferred_model:
                continue
            if self._job_tracker.is_model_held_by_ceiling(job.model):
                continue
            if self._scheduler._job_requires_aux_preparation(job):
                continue
            priced_mb = self._scheduler._measured_admission_candidate_delta_mb(
                job,
                self._scheduler._model_metadata.get_baseline(job.model),
                process_id=None,
                disaggregated=self._scheduler._is_disaggregation_class_eligible(job),
            )
            if priced_mb is None or priced_mb > headroom_mb:
                continue
            fitting.append(_FittingJob(job_id=str(job.id_), model=job.model, priced_mb=priced_mb))
        return tuple(fitting)

    def _tick_grace_reasons(self) -> tuple[str, ...]:
        """The named, bounded reasons a tick may legitimately pass with nothing dispatched.

        Each is a window some part of the worker opened deliberately and closes on its own clock: a residency
        being established or restored, a heavy head's load, a preload in flight, a lane on its way into or out
        of the pool. A governor deferral is deliberately absent: the head it defers has stood down from asking
        for the card and normal scheduling is meant to continue around it, so a deferral excuses nothing about
        the work queued behind that head.
        """
        reasons: list[str] = []
        if self._scheduler.whole_card_residency_grace_active():
            reasons.append("whole-card establish/restore grace")
        if self._scheduler.heavy_head_load_grace_active():
            reasons.append("heavy head load grace")
        if self._loading:
            loading = ", ".join(sorted(pending.model for pending in self._loading.values()))
            reasons.append(f"preload in flight ({loading})")
        cycling = [
            lane.process_id for lane in self._inference_lanes() if lane.last_process_state in _CYCLING_LANE_STATES
        ]
        if cycling:
            reasons.append(f"lane(s) cycling ({', '.join(str(lane_id) for lane_id in cycling)})")
        if self._encode_until:
            reasons.append("disaggregated encode window open")
        return tuple(reasons)

    def _head_is_progressing(self, head: ImageGenerateJobPopResponse) -> tuple[bool, str]:
        """Whether the head of queue is itself moving toward being served, and what the world saw.

        Moving means the card is doing something on this head's behalf: its weights are loading, they are
        staged on a lane awaiting the dispatch that commits them, or a lane already holds them. A head with
        none of those is cold, and reserving card room for it holds the card against a demand nothing is
        working on.
        """
        model = head.model
        if model is None:
            return False, "the head carries no model"
        if self._model_map.is_model_loading(model):
            return True, f"{model} is loading"
        staged = [lane_id for lane_id, lane in self._process_map.items() if lane.loaded_horde_model_name == model]
        if any(lane_id in self._staged_mb for lane_id in staged):
            return True, f"{model} is staged in RAM awaiting dispatch"
        if model in self._resident_model.values():
            return True, f"{model} is resident on the card"
        deferred = self._scheduler._whole_card_ledger.governor_deferred_head(None, now=self.now)
        cold = f"{model} is cold: not loading, not staged, not resident"
        if deferred == model:
            return False, f"{cold}, and its whole-card establishment is governor-deferred"
        return False, cold

    def _protected_dispatch_holds(
        self,
        head: ImageGenerateJobPopResponse | None,
        bucket: SlotDutyBucket | None,
    ) -> tuple[_ProtectedDispatchHold, ...]:
        """Every dispatch declined this tick on behalf of some other entity, with that entity's own state.

        Read from the scheduler's own disclosures. The two gate holds come from the dispatch decision the
        gate records as it holds, which exists only on a tick the gate really ran: the hold ledger it also
        keeps outlives the pass that wrote it, so a bucket derived from that ledger would report a hold on
        ticks where nothing was asked. The two residency holds come from the duty bucket the stall classifier
        names, which is derived per tick from live state. Nothing about a fit is re-derived here; what is
        added is whether the protected entity is itself progressing and which bounded grace, if any, it sits
        inside.
        """
        holds: list[_ProtectedDispatchHold] = []
        for record in self.decision_records:
            if record.tick != self.tick or record.decision_kind is not DecisionKind.INFERENCE_DISPATCH:
                continue
            if record.verdict is not DecisionVerdict.DEFER:
                continue
            if _HEAD_PROTECTION_REASON_MARKER in record.reason:
                if head is None:
                    continue
                progressing, detail = self._head_is_progressing(head)
                holds.append(
                    _ProtectedDispatchHold(
                        kind="head_protection",
                        held_subject=record.subject,
                        protected=f"head {str(head.id_)[:8]} ({head.model})",
                        progressing=progressing,
                        grace=self._whole_card_grace_label(head.model),
                        detail=detail,
                    ),
                )
            elif record.inputs.get("is_head_of_queue") is True:
                reconciliation = self._residency_reconciliation_hold(record)
                if reconciliation is not None:
                    holds.append(reconciliation)
        if head is None or bucket is None:
            return tuple(holds)
        held_subject = str(head.id_)
        if bucket is SlotDutyBucket.WHOLE_CARD_RESERVED:
            holds.append(self._whole_card_reserved_hold(held_subject))
        elif bucket is SlotDutyBucket.EXCLUSIVE_ISOLATION:
            holds.append(self._exclusive_isolation_hold(held_subject))
        return tuple(holds)

    def _whole_card_grace_label(self, model: str | None) -> str | None:
        """The bounded whole-card window a model sits inside right now, or None when it sits inside none."""
        if self._scheduler.whole_card_residency_grace_active():
            return "whole-card establish/restore grace"
        if model is not None and self._scheduler.heavy_head_load_grace_active():
            return "heavy head load grace"
        return None

    def _whole_card_reserved_hold(self, held_subject: str) -> _ProtectedDispatchHold:
        """The hold a whole-card residency held for a non-head model places on the head behind it."""
        state = self._scheduler._whole_card_ledger.get(None)
        residency_model = state.model if state is not None else None
        serving = any(job.model == residency_model for job in self._job_tracker.jobs_in_progress)
        loading = residency_model is not None and self._model_map.is_model_loading(residency_model)
        grace = self._whole_card_grace_label(residency_model)
        if grace is None and self._scheduler._whole_card_ledger.min_hold_active(None, now=self.now):
            grace = "whole-card minimum hold"
        detail = (
            f"residency holds {residency_model}: serving={serving} loading={loading}"
            if residency_model is not None
            else "a residency reserved the card with no model recorded against it"
        )
        return _ProtectedDispatchHold(
            kind="whole_card_reserved",
            held_subject=held_subject,
            protected=f"residency model {residency_model}",
            progressing=serving or loading,
            grace=grace,
            detail=detail,
        )

    def _exclusive_isolation_hold(self, held_subject: str) -> _ProtectedDispatchHold:
        """The hold an exclusively-admitted over-budget job places on everything else on the card."""
        running = self._job_tracker.has_exclusive_job_running(None)
        residency_live = self._scheduler._exclusive_residency_live(None)
        return _ProtectedDispatchHold(
            kind="exclusive_isolation",
            held_subject=held_subject,
            protected="the exclusively-admitted over-budget job",
            progressing=running,
            grace="exclusive whole-card residency live" if residency_live else None,
            detail=f"exclusive job running={running}, its residency live={residency_live}",
        )

    def _residency_reconciliation_hold(self, record: _DecisionRecord) -> _ProtectedDispatchHold | None:
        """The hold the dispatch reconciliation gate places while it asks the card's tenants for room back.

        None when the card carries no tenancy this job's dispatch could displace. The gate holds for two
        different reasons under one name: a job whose materialisation cannot fit beside somebody else's
        weights, and a job that does not fit the card at all with nothing on it. Only the first is a hold
        kept for another entity; the second is the head against the card's own ceiling, which the arbiter
        resolves on its own measured-attempt path and which has no protected entity to ask about.

        What this waits on is room another tenant holds, so it progresses on any of the ways that room comes
        back: an eviction the gate itself requested or applied (both carried on the gate's own record of the
        hold), an unload already in flight, or a job actually running whose completion returns what it took.
        What is left is a head held against tenancy that nothing is running, nothing is unloading and nothing
        has asked for, which is a wait for a fit nothing is producing.

        The gate's opening pass is exempt, through the gate's own standing predicate. A hold is stamped
        before it may escalate to asking idle lanes for tenancy the arbiter cannot name, so the first pass
        necessarily asks for nothing; the escalation runs on the next one. That is a bounded, disclosed
        single pass rather than a wait on nothing, and the bound is the gate's own.
        """
        held_model = record.inputs.get("model")
        displaceable = sorted(
            lane_id
            for lane_id in set(self._resident_mb) | set(self._held_component_mb)
            if self._resident_mb.get(lane_id, 0.0) > 0.0
            and self._resident_model.get(lane_id) != held_model
            or self._held_component_mb.get(lane_id, 0.0) > 0.0
        )
        if not displaceable:
            return None
        unloads_in_flight = sorted(self._unload_due_at)
        requested = bool(record.inputs.get("reclaim_requested")) or bool(record.inputs.get("reclaim_applied"))
        running = len(self._job_tracker.jobs_in_progress)
        held_job = next(
            (job for job in self._job_tracker.jobs_pending_inference if str(job.id_) == record.subject),
            None,
        )
        opening_pass = held_job is not None and not self._scheduler._dispatch_hold_is_standing(held_job)
        return _ProtectedDispatchHold(
            kind="residency_reconciliation",
            held_subject=record.subject,
            protected=f"the room lane(s) {displaceable} hold",
            progressing=requested or bool(unloads_in_flight) or running > 0,
            grace="the gate's opening pass, before its hold stands" if opening_pass else None,
            detail=(
                f"reclaim requested={requested}, unloads in flight on lane(s) {unloads_in_flight}, "
                f"jobs running={running}"
            ),
        )

    def _observe_dispatch_tick(self) -> None:
        """Record what this tick looked like to the verdicts that judge whether the card was earning."""
        head = self._scheduler._undispatched_head()
        bucket: SlotDutyBucket | None = None
        reason: str | None = None
        if head is not None:
            bucket, reason = self._scheduler._classify_dispatch_stall(
                head,
                self._scheduler._model_metadata.require_reference(),
            )
        self.tick_observations.append(
            _TickObservation(
                tick=self.tick,
                now=self.now,
                device_free_mb=self.device_free_mb(),
                jobs_in_progress=len(self._job_tracker.jobs_in_progress),
                fitting_pending=self._pending_jobs_that_fit(),
                grace_reasons=self._tick_grace_reasons(),
                head_job_id=None if head is None or head.id_ is None else str(head.id_),
                head_model=None if head is None else head.model,
                head_stall_bucket=None if bucket is None else bucket.value,
                head_stall_reason=reason,
                protected_holds=self._protected_dispatch_holds(head, bucket),
            ),
        )

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
    def governor_states(self) -> list[GovernorState]:
        """Per tick, the least healthy state any card was committed to, in order.

        A pool is only as healthy as its worst card, so this is what a run-wide governor verdict is about; on a
        single-card row it is that card's own series. A verdict about one card of several reads
        :attr:`governor_states_by_card`.
        """
        series = [self.governor_states_by_card[index] for index in sorted(self._card_totals)]
        return [max(states, key=_GOVERNOR_SEVERITY.__getitem__) for states in zip(*series, strict=True) if states]

    @property
    def min_device_free_mb(self) -> float:
        """The lowest device-free reading any card ever showed, which is the figure a free floor is about."""
        return min(self.min_card_free_mb.values())

    def card_indices(self) -> list[int]:
        """The cards the ledger holds, in index order (``[0]`` on every single-card row)."""
        return sorted(self._card_totals)

    def card_resident_mb(self, device_index: int) -> dict[int, float]:
        """What each lane on one card holds committed to that card's VRAM, keyed by lane.

        The card's resident set: contexts, activation and device-warm components are not in it, so this is the
        weights the card is carrying and nothing else.
        """
        return {
            lane_id: charge_mb
            for lane_id, charge_mb in sorted(self._resident_mb.items())
            if charge_mb > 0.0 and self._card_of(lane_id) == device_index
        }

    def card_resident_models(self, device_index: int) -> dict[int, str]:
        """The model each lane on one card most recently committed to that card, keyed by lane."""
        return {
            lane_id: model
            for lane_id, model in sorted(self._resident_model.items())
            if self._card_of(lane_id) == device_index
        }

    def safety_card_index(self) -> int:
        """The card the on-GPU safety process is pinned to, and whose ledger carries its charge."""
        return self._safety_card_index

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

    def lane_serving(self, job: ImageGenerateJobPopResponse) -> int | None:
        """The lane holding ``job``, or None once it is off every lane."""
        return self._lane_of.get(str(job.id_))

    def card_of_lane(self, lane_id: int) -> int:
        """The device index the pool pinned ``lane_id`` to (0 on every single-card row)."""
        return self._process_map[lane_id].device_index

    def inference_lane_ids(self) -> list[int]:
        """The inference lanes the pool currently holds, in map order."""
        return [lane.process_id for lane in self._inference_lanes()]

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
        cards = ", ".join(
            f"{index}:{self.card_free_mb(index):.0f}/{total_mb:.0f}MB"
            for index, total_mb in sorted(self._card_totals.items())
        )
        return (
            f"tick={self.tick} free=[{cards}] "
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
    # Truthful "which card is safety on", the manager's own contract: the card it was pinned to at its last
    # bring-up, and None while it is off the card. The placement policy reads it to pick the card whose
    # evidence decides the flip, so a pool of several cards judged by a stand-in that never says which one
    # would be judging safety's placement against a card it does not sit on.
    lifecycle.safety_gpu_card_index = lambda: None if world.safety_is_off_gpu() else world.safety_card_index()
    lifecycle.safety_readiness_latency_seconds = world.safety_readiness_latency_seconds
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
