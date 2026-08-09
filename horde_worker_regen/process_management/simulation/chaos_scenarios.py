"""Seeded generation of worker-load scenarios for the generated chaos sweep.

A scenario is a composition of three things a wedge needs in order to happen: a queue structure (how much
work arrives, for which models, with which payload features, in what order and on what schedule), a worker
configuration (sampling slots, lanes, queue depth, card, memory policy, residency policy, throughput posture,
and unload posture), and a schedule of disturbances fired part-way through (a lane dying, an outside reclaim,
a slow child, a resource fault, or a misrouted message). An integer seed fully determines all three, so a red
run replays from its seed alone.

The generator is deterministic: it draws from ``random.Random(seed)``, resolves requested thread/queue
settings through the worker's production configuration rules, and returns a frozen description. Two
runners consume it:

- the scheduling-loop runner drives the composition over a modelled card on a fake clock, where a lane
  death and an outside reclaim are expressible and the child-side faults are not;
- the full-worker runner uses the same seed over a bounded production-topology projection and drives it
  against real child processes scripted with a
  :class:`~horde_worker_regen.process_management.simulation.fault_injection.FaultProfile`, where the
  child-side faults are expressible and an outside reclaim is not.

:data:`WORLD_EVENT_KINDS` and :data:`CHILD_EVENT_KINDS` state that split. The scheduling runner requires
an effective receipt for every requested world event. The subprocess generator retains only one child event
on a single-lane topology, where the fake's process-local ordinal is the scenario-global ordinal, rather
than claiming coverage for events it cannot target. :data:`DISCLOSED_BOUNDS` states every axis the space fixes or
truncates, and why; the suites print it, so the sweep's coverage is never taken on trust.

Model identity carries two names because the two altitudes resolve models differently. The scheduling-loop
runner prices from a synthetic reference keyed by ``scheduler_name``; the full-worker runner uses the
harness's local name-keyed synthetic reference. Both names denote the same weight class in the space this
module generates, without requiring live model-reference state.
"""

from __future__ import annotations

import base64
import enum
import functools
import random
from dataclasses import dataclass, replace

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry, TIPayloadEntry

from horde_worker_regen.bridge_data.data_model import cap_queue_size
from horde_worker_regen.process_management.process_manager import (
    _EstimatedContextFootprint,
    cap_card_process_counts,
    cap_card_processes_to_vram_fit,
    resolve_card_concurrency,
)
from horde_worker_regen.process_management.resources.resource_budget import predict_job_sampling_vram_mb
from horde_worker_regen.process_management.scheduling.inference_scheduler import _SAFETY_GPU_LOAD_CHARGE_MB
from horde_worker_regen.process_management.simulation._canned_scenarios import make_canned_job
from horde_worker_regen.process_management.simulation._dummy_images import make_dummy_png_bytes

SEED_ENV_VAR = "HORDE_CHAOS_SEEDS"
"""Environment override for which seeds a sweep runs.

Accepts an inclusive-exclusive range (``100:200``) or a comma list (``7,19,23``). Set it to replay a single
red seed, or to widen a nightly sweep beyond its committed range, without editing the suite.
"""


@dataclass(frozen=True)
class ChaosCard:
    """One card profile a scenario runs on, with the host it is fitted to.

    Attributes:
        label: Short identity used in scenario labels and failure messages.
        total_vram_mb: The card's total VRAM, which decides which models are servable on it.
        host_ram_gb: Host memory, so a full-worker run's RAM admission gates see a plausible host.
        per_process_overhead_mb: The first CUDA context's cost against the card.
        marginal_process_overhead_mb: Each additional context's cost.
    """

    label: str
    total_vram_mb: int
    host_ram_gb: int
    per_process_overhead_mb: int
    marginal_process_overhead_mb: int


CARD_8GB = ChaosCard("8gb", 8192, 16, 1100, 500)
CARD_16GB = ChaosCard("16gb", 16384, 32, 1400, 600)
CARD_24GB = ChaosCard("24gb", 24576, 64, 1800, 700)

CARDS: tuple[ChaosCard, ...] = (CARD_8GB, CARD_16GB, CARD_24GB)


@dataclass(frozen=True)
class ChaosModel:
    """One checkpoint identity in the generated vocabulary.

    Attributes:
        label: Short identity used in scenario labels.
        scheduler_name: The name the scheduling-loop runner's synthetic reference prices.
        harness_name: The name a full-worker run pops for.
        min_card_vram_mb: The smallest card this model is servable on. A scenario never queues a model its
            card cannot serve, so every generated head has a positive outcome to assert.
        weight_rank: The model's weight class as an order (light 0, mid 1, whole-card 2), which is what the
            heavy-head queue shape picks its head by.
        baseline: The reference baseline the burden estimator prices this class from, so a generated job's
            activation cost is the one the worker would compute for it rather than a figure kept here.
        resident_weights_mb: What one resident copy costs the card, matching the weight seed the scheduling
            runner's modelled card charges, so a scenario's own fit arithmetic and the runner's derived free
            VRAM agree about what a co-resident sibling holds.
    """

    label: str
    scheduler_name: str
    harness_name: str
    min_card_vram_mb: int
    weight_rank: int
    baseline: KNOWN_IMAGE_GENERATION_BASELINE
    resident_weights_mb: float


MODEL_LIGHT_A = ChaosModel(
    "light_a",
    "sd15-checkpoint",
    "Deliberate",
    0,
    0,
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
    3200.0,
)
MODEL_LIGHT_B = ChaosModel(
    "light_b",
    "sd15-checkpoint-b",
    "Anything Diffusion",
    0,
    0,
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
    3200.0,
)
MODEL_MID = ChaosModel(
    "mid",
    "sdxl-checkpoint",
    "Juggernaut XL",
    0,
    1,
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
    4900.0,
)
MODEL_HEAVY = ChaosModel(
    "heavy",
    "Flux.1-Schnell fp8 (Compact)",
    "Flux.1-Schnell fp8 (Compact)",
    16384,
    2,
    KNOWN_IMAGE_GENERATION_BASELINE.flux_1,
    11500.0,
)
"""The whole-card class: heavy enough to take the exclusive-residency path on a card that can hold it."""

MODELS: tuple[ChaosModel, ...] = (MODEL_LIGHT_A, MODEL_LIGHT_B, MODEL_MID, MODEL_HEAVY)

LIGHT_MODELS: tuple[ChaosModel, ...] = (MODEL_LIGHT_A, MODEL_LIGHT_B)
"""The light class, which every card serves and which the heavy-head shape queues behind its head."""


class ChaosArrival(enum.StrEnum):
    """How the queue's jobs become available to the worker."""

    ALL_AT_ONCE = "all_at_once"
    BURSTS = "bursts"
    STEADY = "steady"


class ChaosQueueShape(enum.StrEnum):
    """The model ordering of the generated queue."""

    SAME_MODEL_BURST = "same_model_burst"
    """Every job wants one model, so a single residency serves the whole queue."""
    ALTERNATING = "alternating"
    """Two models in strict alternation, so residency must rotate on every job."""
    HEAVY_LIGHT = "heavy_light"
    """A heavy head with lighter work behind it, the line-skip and reclaim shape."""
    MIXED = "mixed"
    """A free draw over the card's servable models."""


class ChaosDemandShape(enum.StrEnum):
    """How per-job sampling demand changes as the queue advances."""

    UNIFORM = "uniform"
    """Every job uses one shared geometry."""
    LOW_HIGH_LOW = "low_high_low"
    """Demand rises at the second job and then falls, including within one model residency."""
    HIGH_LOW = "high_low"
    """A high-demand head is followed by lower-demand work."""
    MIXED = "mixed"
    """Each job independently draws a geometry its model/card pairing can serve."""


class ChaosInitialResidency(enum.StrEnum):
    """The modelled card state before the first queue arrival."""

    EMPTY = "empty"
    HEAD_IN_RAM = "head_in_ram"
    HEAD_IN_VRAM = "head_in_vram"
    FOREIGN_IN_VRAM = "foreign_in_vram"


class ChaosPerformance(enum.StrEnum):
    """Throughput posture applied to one generated worker."""

    NORMAL = "normal"
    MODERATE = "moderate"
    HIGH = "high"


class ChaosSourceMode(enum.StrEnum):
    """Source-image structure carried by one generated job."""

    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    INPAINTING = "inpainting"


class ChaosAuxKind(enum.StrEnum):
    """Auxiliary-model references carried by one generated job."""

    NONE = "none"
    LORA = "lora"
    TEXTUAL_INVERSION = "ti"
    BOTH = "lora_ti"


class ChaosControlKind(enum.StrEnum):
    """ControlNet input and output structure carried by one generated job."""

    NONE = "none"
    ANNOTATE = "annotate"
    PREANNOTATED = "preannotated"
    RETURN_MAP = "return_map"


class ChaosPostProcessing(enum.StrEnum):
    """Post-processing work requested after generation."""

    NONE = "none"
    FACE_FIX = "face_fix"
    UPSCALE = "upscale"
    CHAIN = "chain"


class ChaosSamplerProfile(enum.StrEnum):
    """Representative sampler and scheduler pair requested by one job."""

    EULER_NORMAL = "euler_normal"
    DPM_KARRAS = "dpm_karras"
    LCM_SIMPLE = "lcm_simple"


class ChaosActivationShape(enum.StrEnum):
    """How large a sampling activation the scenario's geometries ask for.

    Resolution and batch move the sampling working set by gigabytes while the resident weights stay put, so
    this axis is what separates a model's persistent residency cost from its transient peak. It is the regime
    where an activation spike can be read as a demand for permanent exclusive residency.
    """

    NOMINAL = "nominal"
    """Geometries from the ordinary generated vocabulary."""
    HIRES_BATCH = "hires_batch"
    """High-resolution, batched geometries, taken as far as the card prices as servable."""


class ChaosSiblingResidency(enum.StrEnum):
    """What a sibling lane holds when the scenario starts, beside whatever the head's own state is."""

    NONE = "none"
    FOREIGN_IDLE_IN_VRAM = "foreign_idle_in_vram"
    """A second lane holds a model no queued job wants, idle and resident, so the card starts with more than
    one live inference context carrying weights: the state a process-count-reduction claim proposes to
    reduce."""


class ChaosServiceTopology(enum.StrEnum):
    """Which non-inference tenants hold a context on the card."""

    BARE = "bare"
    SERVICE_CONTEXTS = "service_contexts"
    """Safety sits on the card and the service lane holds a context. Neither is reclaimable by stopping
    inference siblings, so the card's structural floor is squeezed by charges no teardown can give back."""


@dataclass(frozen=True)
class ChaosJob:
    """One queued image job with its own model, payload structure, and sampling demand."""

    model: ChaosModel
    width: int
    height: int
    steps: int
    n_iter: int = 1
    source_mode: ChaosSourceMode = ChaosSourceMode.TXT2IMG
    aux_kind: ChaosAuxKind = ChaosAuxKind.NONE
    control_kind: ChaosControlKind = ChaosControlKind.NONE
    post_processing: ChaosPostProcessing = ChaosPostProcessing.NONE
    hires_fix: bool = False
    sampler_profile: ChaosSamplerProfile = ChaosSamplerProfile.EULER_NORMAL

    @property
    def pixels(self) -> int:
        """Return the requested image area."""
        return self.width * self.height

    @property
    def feature_label(self) -> str:
        """Return a compact payload identity for scenario summaries and parametrized failures."""
        return (
            f"b{self.n_iter}-{self.source_mode.value}-{self.aux_kind.value}-{self.control_kind.value}-"
            f"{self.post_processing.value}-h{int(self.hires_fix)}-{self.sampler_profile.value}"
        )


@dataclass(frozen=True)
class ChaosTopology:
    """A requested worker configuration and the effective single-card process topology it resolves to."""

    max_threads: int
    requested_queue_size: int
    queue_size: int
    lanes: int
    num_models_to_load: int

    @property
    def label(self) -> str:
        """Return a stable compact identity for coverage reports."""
        return f"t{self.max_threads}-q{self.queue_size}-l{self.lanes}-m{self.num_models_to_load}"


class ChaosEventKind(enum.StrEnum):
    """A disturbance fired part-way through a scenario."""

    LANE_DEATH = "lane_death"
    """The lane holding (or loading) the current model dies and is replaced."""
    EXTERNAL_RECLAIM = "external_reclaim"
    """An idle resident model is evicted by an actor other than the scheduler."""
    SLOW_CHILD = "slow_child"
    """A child takes several times longer than expected while still reporting progress."""
    RESOURCE_FAULT = "resource_fault"
    """A job reports an out-of-memory result instead of images."""
    MISROUTED_MESSAGE = "misrouted_message"
    """A child emits a garbage or stale-stamped message before its real result."""


WORLD_EVENT_KINDS = frozenset({ChaosEventKind.LANE_DEATH, ChaosEventKind.EXTERNAL_RECLAIM})
"""Kinds the scheduling-loop runner can express against its modelled card."""

CHILD_EVENT_KINDS = frozenset(
    {
        ChaosEventKind.LANE_DEATH,
        ChaosEventKind.SLOW_CHILD,
        ChaosEventKind.RESOURCE_FAULT,
        ChaosEventKind.MISROUTED_MESSAGE,
    },
)
"""Kinds the full-worker runner can express through a fault profile."""


@dataclass(frozen=True)
class ChaosEvent:
    """One disturbance, carrying a job ordinal for deterministic placement and subprocess targeting.

    Attributes:
        kind: What happens.
        at_job_ordinal: The 1-based ordinal a full-worker child targets directly. The scheduling runner uses
            it as a release-relative time anchor except for an outside reclaim, which fires against its
            explicitly constructed initial foreign residency.
    """

    kind: ChaosEventKind
    at_job_ordinal: int


@dataclass(frozen=True)
class ChaosScenario:
    """One generated composition of queue, worker configuration, and disturbance schedule.

    Attributes:
        seed: The integer that produced this scenario. Printed on every failure so a red run replays.
        card: The card profile the worker runs on.
        shape: The model ordering the queue was drawn with.
        jobs: The queued jobs, including each job's model and sampling demand, head first.
        arrival: How the jobs become available.
        burst_size: Jobs released per burst (bursts arrival only).
        topology: Requested settings and the effective topology resolved from production rules.
        demand_shape: How sampling demand changes across the queue.
        initial_residency: Card residency before the first arrival.
        events: The disturbance schedule, ordered by job ordinal.
        sibling_residency: What a second lane holds beside the head's own starting state.
        service_topology: Which non-inference tenants hold a context on the card.
        activation_shape: How large a sampling activation the scenario's geometries ask for.
        disaggregation_class: Whether the scenario's jobs run as UNet-only samplers, so the whole-card
            decision meets a job whose own encode and decode lanes a claim would stop.
        probe_measured_marginal: Whether the host's startup probe measured the per-additional-context VRAM
            cost. Without that measurement a card-light head's teardown demand is declined as an
            over-counted-context phantom, so the process-count-reduction decision is unreachable however the
            card is arranged.
    """

    seed: int
    card: ChaosCard
    shape: ChaosQueueShape
    jobs: tuple[ChaosJob, ...]
    arrival: ChaosArrival
    burst_size: int
    topology: ChaosTopology
    demand_shape: ChaosDemandShape
    initial_residency: ChaosInitialResidency
    events: tuple[ChaosEvent, ...]
    enable_vram_budget: bool = True
    whole_card_enabled: bool = True
    performance: ChaosPerformance = ChaosPerformance.NORMAL
    unload_models_from_vram_often: bool = False
    sibling_residency: ChaosSiblingResidency = ChaosSiblingResidency.NONE
    service_topology: ChaosServiceTopology = ChaosServiceTopology.BARE
    activation_shape: ChaosActivationShape = ChaosActivationShape.NOMINAL
    disaggregation_class: bool = False
    probe_measured_marginal: bool = False

    @property
    def job_count(self) -> int:
        """How many jobs the scenario queues."""
        return len(self.jobs)

    @property
    def models(self) -> tuple[ChaosModel, ...]:
        """Return the queued models, head first."""
        return tuple(job.model for job in self.jobs)

    @property
    def max_threads(self) -> int:
        """Return the effective concurrent-sampling cap."""
        return self.topology.max_threads

    @property
    def lanes(self) -> int:
        """Return the effective number of inference lanes."""
        return self.topology.lanes

    @property
    def queue_size(self) -> int:
        """Return the effective local queue depth."""
        return self.topology.queue_size

    @property
    def model_switches(self) -> int:
        """How many times the queue changes model, which is how often residency must rotate."""
        return sum(1 for before, after in zip(self.models, self.models[1:], strict=False) if before is not after)

    @property
    def heavy_job_count(self) -> int:
        """How many queued jobs want the whole-card class."""
        return sum(1 for model in self.models if model is MODEL_HEAVY)

    @property
    def head_resident_at_dispatch(self) -> bool:
        """Whether the head's model starts resident in VRAM, so dispatch meets it already loaded.

        The dispatch-time whole-card decision is only reachable for such a head: one whose weights a lane
        already holds when it is chosen to sample, rather than one admitted through a preload.
        """
        return self.initial_residency is ChaosInitialResidency.HEAD_IN_VRAM

    @property
    def label(self) -> str:
        """A stable, readable identity for the scenario (used as the parametrized test id)."""
        return (
            f"seed{self.seed}-{self.card.label}-{self.shape.value}-{self.demand_shape.value}-"
            f"{self.topology.label}-n{self.job_count}-{self.activation_shape.value}-"
            f"{self.service_topology.value}"
        )

    def world_events(self) -> tuple[ChaosEvent, ...]:
        """The disturbances the scheduling-loop runner can express."""
        return tuple(event for event in self.events if event.kind in WORLD_EVENT_KINDS)

    def child_events(self) -> tuple[ChaosEvent, ...]:
        """The disturbances the full-worker runner can express."""
        return tuple(event for event in self.events if event.kind in CHILD_EVENT_KINDS)

    def summary(self) -> str:
        """A one-line description carrying everything needed to reproduce and read the scenario."""
        queue = ">".join(
            f"{job.model.label}:{job.width}x{job.height}@{job.steps}:{job.feature_label}" for job in self.jobs
        )
        events = (
            ",".join(f"{event.kind.value}@{event.at_job_ordinal}" for event in self.events) if self.events else "none"
        )
        return (
            f"seed={self.seed} card={self.card.label} shape={self.shape.value} queue=[{queue}] "
            f"arrival={self.arrival.value}(burst={self.burst_size}) threads={self.max_threads} "
            f"lanes={self.lanes} queue_size={self.queue_size} demand={self.demand_shape.value} "
            f"residency={self.initial_residency.value} sibling={self.sibling_residency.value} "
            f"vram_budget={self.enable_vram_budget} "
            f"whole_card={self.whole_card_enabled} performance={self.performance.value} "
            f"unload_often={self.unload_models_from_vram_often} "
            f"activation={self.activation_shape.value} services={self.service_topology.value} "
            f"disaggregated={self.disaggregation_class} marginal_measured={self.probe_measured_marginal} "
            f"events=[{events}]"
        )


_QUEUE_LENGTHS: tuple[int, ...] = (2, 3, 5, 8)
"""Short, retry-boundary, queue-boundary, and repeated-recovery sequence lengths."""
_TOPOLOGY_REQUESTS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, 1),
    (1, 4),
    (2, 0),
    (2, 4),
    (3, 0),
    (3, 4),
    (16, 0),
    (16, 4),
)
"""Semantic boundaries of the public thread and queue ranges.

The queue-four rows exercise the production cap to three when more than one sampler is configured. The
sixteen-thread rows cover the upper validation boundary without asking the subprocess tier to spawn that
many children; :func:`generate_full_worker_scenarios` projects onto the bounded subset below.
"""

_FULL_WORKER_TOPOLOGY_REQUESTS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, 1),
    (1, 4),
    (2, 0),
    (2, 4),
)
"""Production-valid topology boundaries affordable at the real-subprocess altitude."""
_JOB_SIZES: tuple[tuple[int, int, int], ...] = (
    (512, 512, 8),
    (512, 768, 12),
    (768, 768, 12),
    (1024, 1024, 8),
)
"""Generated job geometries. Steps stay low so a full-worker run's fake children finish promptly; the axis
varies the priced sampling peak, which is what admission reads, not how long a fake pretends to sample."""

_EVENT_COUNT_DRAW: tuple[int, ...] = (0, 0, 1, 1, 2, 3)
"""How many disturbances a scenario fires, drawn with the weighting this tuple encodes. Undisturbed runs
stay in the space on purpose: a wedge that only appears when nothing is being injected is still a wedge."""

_UNBOUNDED_PIXELS = 1 << 30
"""Stands for "this card serves this model at any geometry the vocabulary offers"."""

_PIXEL_BUDGETS: dict[tuple[str, str], int] = {
    (CARD_8GB.label, MODEL_MID.label): 768 * 768,
}
"""Measured caps on what a card serves outright, keyed by ``(card label, model label)``.

A model's weights fitting a card is not the whole servability question: the priced sampling peak grows with
the requested geometry, so a card serves a checkpoint up to a geometry and not beyond it. The cap is the
largest geometry measured as served by the scheduling loop alone, and a queued job above it would be a head
the card genuinely cannot seat, which is the unservable-head bound's subject rather than this space's.

A mid-class checkpoint on the smallest card is served across the whole range up to this cap, including the
geometries whose predicted peak clears the card's achievable ceiling: those are admitted through one real
measured load once the card converges, which is a served outcome and not a deferred one. The cap sits where
the demand stops being recoverable by that route.

Entries are absent for every combination the card serves unconditionally.
"""


_HIRES_JOB_SIZES: tuple[tuple[int, int, int], ...] = (
    (1024, 1024, 8),
    (1280, 1280, 8),
    (1536, 1536, 8),
)
"""Hires-class geometries, largest last. Each is offered at a batch of at least two, so the activation term
grows by both of the multipliers that move it while the requested weights stay identical. Which of them a
scenario takes is decided by pricing, never by a table: see :func:`hires_upgrade`."""

_HIRES_BATCH = 2
"""The smallest batch that makes a hires geometry's activation delta a multiple rather than a single frame."""

_RESERVE_ALLOWANCE_MB = 1024.0
"""Kept clear of a scenario's own fit arithmetic, above the tenants it can enumerate.

A card's inference reserve floor, the allocator's own slack, and the transient a load takes on its way to
residency are all charged by the worker and are not derivable from a scenario description. Reserving a fixed
allowance for them keeps the axes below from constructing a card state that prices as servable here and is
refused by the running worker, which would turn an axis into a false red."""


@functools.lru_cache(maxsize=512)
def _priced_sampling_peak_mb(
    baseline: KNOWN_IMAGE_GENERATION_BASELINE,
    width: int,
    height: int,
    n_iter: int,
    *,
    hires_fix: bool,
    has_source_image: bool,
    has_control: bool,
    has_lora: bool,
    has_ti: bool,
) -> float | None:
    """Return the worker's own predicted sampling-phase peak (MB) for one generated job description.

    Priced through :func:`predict_job_sampling_vram_mb`, the same seam preload admission and the residency
    forecast read, so a generated geometry's activation cost is whatever the worker would charge it rather
    than a number maintained here. Post-processing is deliberately absent from the priced job: that peak runs
    after sampling on the already-loaded model and is excluded from this figure by construction.

    Returns:
        The predicted peak, or None when no estimate could be produced.
    """
    job = make_canned_job(
        "sd15-checkpoint",
        width=width,
        height=height,
        ddim_steps=8,
        n_iter=n_iter,
        hires_fix=hires_fix,
        control_type="canny" if has_control else None,
        source_image_base64=(base64.b64encode(make_dummy_png_bytes()).decode() if has_source_image else None),
        loras=[LorasPayloadEntry(name="priced-lora")] if has_lora else None,
        tis=[TIPayloadEntry(name="priced-ti", inject_ti="prompt")] if has_ti else None,
    )
    return predict_job_sampling_vram_mb(job, baseline)


def priced_sampling_peak_mb(job: ChaosJob) -> float | None:
    """Return the predicted sampling-phase peak (MB) for a generated job, or None when it cannot be priced."""
    return _priced_sampling_peak_mb(
        job.model.baseline,
        job.width,
        job.height,
        job.n_iter,
        hires_fix=job.hires_fix,
        has_source_image=job.source_mode is not ChaosSourceMode.TXT2IMG,
        has_control=job.control_kind is not ChaosControlKind.NONE,
        has_lora=job.aux_kind in {ChaosAuxKind.LORA, ChaosAuxKind.BOTH},
        has_ti=job.aux_kind in {ChaosAuxKind.TEXTUAL_INVERSION, ChaosAuxKind.BOTH},
    )


def sampling_headroom_mb(
    card: ChaosCard,
    *,
    lanes: int,
    service_contexts: bool,
    foreign_resident_mb: float,
) -> float:
    """Return the VRAM one sampling peak has to fit into, given everything else the card is holding.

    The card's total less the contexts its topology provisions, less the service tenants' charges when they
    are present, less the weights a foreign resident sibling holds, less the reserve allowance.

    Every provisioned lane is charged, even though the scheduling loop may reduce them under pressure.
    Counting on that reduction is what would let a tightening axis construct a card whose head is servable
    only if the pool collapses far enough, and a head that is not servable at the topology it was configured
    with is outside this space by its first disclosed bound, not a wedge for it to find.
    """
    charge = card.per_process_overhead_mb + card.marginal_process_overhead_mb * max(0, lanes - 1)
    if service_contexts:
        charge += _SAFETY_GPU_LOAD_CHARGE_MB + card.marginal_process_overhead_mb
    return card.total_vram_mb - charge - foreign_resident_mb - _RESERVE_ALLOWANCE_MB


def _all_jobs_priced_to_fit(jobs: tuple[ChaosJob, ...], headroom_mb: float) -> bool:
    """Whether every job's priced sampling peak is known and fits ``headroom_mb``.

    An unpriceable job answers False: an axis that tightens the card is only taken where the tightening can
    be shown to leave the queue servable, so the absence of a price is not treated as room.
    """
    for job in jobs:
        peak_mb = priced_sampling_peak_mb(job)
        if peak_mb is None or peak_mb > headroom_mb:
            return False
    return True


def hires_upgrade(job: ChaosJob, headroom_mb: float) -> ChaosJob:
    """Return ``job`` at the largest hires-class geometry that prices inside ``headroom_mb``.

    The whole-card class keeps its geometry: its weights already fill the card by themselves, so growing its
    activation says nothing new about the activation-versus-residency distinction this axis exists to vary,
    and the class's payload vocabulary excludes the batching this upgrade applies.

    Returns:
        The upgraded job, or the job unchanged when no hires geometry is priced as servable.
    """
    if job.model is MODEL_HEAVY:
        return job
    batch = max(job.n_iter, _HIRES_BATCH)
    for width, height, steps in reversed(_HIRES_JOB_SIZES):
        if width * height * batch <= job.pixels * job.n_iter:
            continue
        candidate = replace(job, width=width, height=height, steps=steps, n_iter=batch)
        peak_mb = priced_sampling_peak_mb(candidate)
        if peak_mb is not None and peak_mb <= headroom_mb:
            return candidate
    return job


def servable_pixels(card: ChaosCard, model: ChaosModel) -> int:
    """Return the largest generated geometry, in pixels, that ``card`` serves for ``model``.

    Returns:
        The pixel budget, or 0 when the card cannot serve the model at any generated geometry.
    """
    if card.total_vram_mb < model.min_card_vram_mb:
        return 0
    return _PIXEL_BUDGETS.get((card.label, model.label), _UNBOUNDED_PIXELS)


def _servable_models(card: ChaosCard) -> tuple[ChaosModel, ...]:
    """Return the models the card can serve at the smallest generated geometry.

    Restricting the draw is what makes a generated scenario's verdict uniform: every queued job is servable,
    so the property is "all of it drains" rather than a per-row judgement about which head the card refuses.
    """
    smallest = min(width * height for width, height, _steps in _JOB_SIZES)
    return tuple(model for model in MODELS if servable_pixels(card, model) >= smallest)


def _servable_sizes(card: ChaosCard, models: tuple[ChaosModel, ...]) -> tuple[tuple[int, int, int], ...]:
    """Return the geometries the card serves for every model in the queue.

    One geometry is drawn per scenario, so the budget is the tightest one the queue's models impose.
    """
    budget = min(servable_pixels(card, model) for model in models)
    return tuple(size for size in _JOB_SIZES if size[0] * size[1] <= budget)


def _resolve_topology(
    *,
    max_threads: int,
    requested_queue_size: int,
    num_models_to_load: int,
    card: ChaosCard | None = None,
) -> ChaosTopology:
    """Resolve a generated configuration through the worker's configuration and hardware caps.

    ``card`` is supplied by the full-worker projection, whose declared lane count must include the manager's
    spawn-time VRAM and shared-RAM sizing. The fake-clock projection omits it because its modeled lane pool is
    intentionally the configuration-level topology, independent of OS-process footprint.
    """
    queue_size = cap_queue_size(
        max_threads=max_threads,
        queue_size=requested_queue_size,
        log=False,
    )
    concurrency = resolve_card_concurrency(
        max_threads=max_threads,
        queue_size=queue_size,
        num_models_to_load=num_models_to_load,
        gpu_sampling_lease_enabled=False,
        gpu_sampling_lease_slots=None,
        max_threads_ceiling=max_threads,
    )
    target_process_count = concurrency.target_process_count
    if card is not None:
        target_process_count = cap_card_processes_to_vram_fit(
            per_card_target_processes={0: target_process_count},
            total_vram_mb_by_card={0: float(card.total_vram_mb)},
            idle_context_overhead_mb=_EstimatedContextFootprint.IDLE_CONTEXT_VRAM_MB,
            working_set_footprint_mb=_EstimatedContextFootprint.SDXL_CONTEXT_VRAM_MB,
        )[0]
        total_ram_bytes = card.host_ram_gb * 1024 * 1024 * 1024
        target_ram_overhead_bytes = min(total_ram_bytes // 2, 9 * 1024 * 1024 * 1024)
        target_process_count = cap_card_process_counts(
            per_card_target_processes={0: target_process_count},
            total_ram_bytes=total_ram_bytes,
            target_ram_overhead_bytes=target_ram_overhead_bytes,
        )[0]
    return ChaosTopology(
        max_threads=max_threads,
        requested_queue_size=requested_queue_size,
        queue_size=queue_size,
        lanes=target_process_count,
        num_models_to_load=num_models_to_load,
    )


def _draw_jobs(
    rng: random.Random,
    *,
    feature_rng: random.Random,
    card: ChaosCard,
    models: tuple[ChaosModel, ...],
    demand_shape: ChaosDemandShape,
    uniform_size: tuple[int, int, int],
) -> tuple[ChaosJob, ...]:
    """Give every queued model an independently servable sampling demand."""
    jobs: list[ChaosJob] = []
    for index, model in enumerate(models):
        sizes = _servable_sizes(card, (model,))
        if demand_shape is ChaosDemandShape.UNIFORM:
            size = uniform_size if uniform_size in sizes else sizes[-1]
        elif demand_shape is ChaosDemandShape.LOW_HIGH_LOW:
            size = sizes[-1] if index == 1 else sizes[0]
        elif demand_shape is ChaosDemandShape.HIGH_LOW:
            size = sizes[-1] if index == 0 else sizes[0]
        else:
            size = rng.choice(sizes)
        control_kind = feature_rng.choice(list(ChaosControlKind))
        n_iter = feature_rng.choice((1, 2, 4))
        hires_fix = feature_rng.choice((False, True))
        source_mode = feature_rng.choice(list(ChaosSourceMode))
        aux_kind = feature_rng.choice(list(ChaosAuxKind))
        post_processing = feature_rng.choice(list(ChaosPostProcessing))
        sampler_profile = feature_rng.choice(list(ChaosSamplerProfile))
        # A positive generated verdict must stay inside the capability envelope it claims. Whole-card
        # checkpoints are represented by a Flux-class model, for which ControlNet and hires-fix are not valid
        # workload choices. On the smallest card, large and XL-class jobs use one sampling image without the
        # activation multipliers that would turn a servable model/geometry pair into an unservable payload.
        if model is MODEL_HEAVY:
            n_iter = 1
            hires_fix = False
            control_kind = ChaosControlKind.NONE
        elif card is CARD_8GB:
            if size[0] * size[1] > 512 * 512 or model is MODEL_MID:
                n_iter = 1
                hires_fix = False
                control_kind = ChaosControlKind.NONE
                if model is MODEL_MID and size[0] * size[1] > 512 * 512:
                    source_mode = ChaosSourceMode.TXT2IMG
                    post_processing = ChaosPostProcessing.NONE
            elif n_iter > 1:
                hires_fix = False
        jobs.append(
            ChaosJob(
                model=model,
                width=size[0],
                height=size[1],
                steps=size[2],
                n_iter=n_iter,
                source_mode=source_mode,
                aux_kind=aux_kind,
                control_kind=control_kind,
                post_processing=post_processing,
                hires_fix=hires_fix,
                sampler_profile=sampler_profile,
            ),
        )
    return tuple(jobs)


def _draw_queue(
    rng: random.Random,
    *,
    shape: ChaosQueueShape,
    servable: tuple[ChaosModel, ...],
    length: int,
) -> tuple[ChaosModel, ...]:
    """Draw the queue's models, head first, in the given shape."""
    if shape is ChaosQueueShape.SAME_MODEL_BURST:
        return (rng.choice(servable),) * length
    if shape is ChaosQueueShape.ALTERNATING:
        first = rng.choice(servable)
        second = rng.choice([model for model in servable if model is not first] or list(servable))
        return tuple((first, second)[index % 2] for index in range(length))
    if shape is ChaosQueueShape.HEAVY_LIGHT:
        # The heaviest class the card serves, so the shape stays a heavy-head shape on a card the
        # whole-card class does not fit rather than collapsing onto a queue of equals.
        head = max(servable, key=lambda model: model.weight_rank)
        return (head, *(rng.choice(LIGHT_MODELS) for _ in range(length - 1)))
    return tuple(rng.choice(servable) for _ in range(length))


def _draw_events(rng: random.Random, *, job_count: int) -> tuple[ChaosEvent, ...]:
    """Draw the disturbance schedule, with unique kinds and job ordinals.

    Repeating a reclaim or lane death without constructing a second eligible target makes the later draw a
    known no-op. Longer sequences vary composition depth; each disturbance kind appears at most once until
    the generator can construct and verify repeated preconditions explicitly.
    """
    count = min(rng.choice(_EVENT_COUNT_DRAW), job_count)
    if count == 0:
        return ()
    ordinals = sorted(rng.sample(range(1, job_count + 1), count))
    kinds = rng.sample(list(ChaosEventKind), count)
    return tuple(ChaosEvent(kind=kind, at_job_ordinal=ordinal) for ordinal, kind in zip(ordinals, kinds, strict=True))


def _foreign_sibling_model(models: tuple[ChaosModel, ...]) -> ChaosModel | None:
    """Return the model an idle sibling lane holds beside the queue, or None when none is left over.

    Light by class, so the sibling's weights are a co-resident charge on every card rather than a second
    demand for the card, and foreign to the queue, so nothing the scenario serves is riding on it: the
    residency it represents is purely a live context the card is holding for other work.
    """
    return next((model for model in LIGHT_MODELS if model not in models), None)


def foreign_sibling_model(scenario: ChaosScenario) -> ChaosModel | None:
    """Return the model a runner seeds onto the scenario's idle sibling lane, or None when it seeds none."""
    if scenario.sibling_residency is ChaosSiblingResidency.NONE:
        return None
    return _foreign_sibling_model(scenario.models)


def _generate_scenario(
    seed: int,
    *,
    topology_requests: tuple[tuple[int, int], ...],
    full_worker: bool,
) -> ChaosScenario:
    """Return the scenario the given seed denotes.

    The draw order is part of the contract: changing it renumbers every seed, so a committed seed list and
    the failures it has pinned would no longer mean the same scenarios. Add axes at the end of the draw
    rather than in the middle.

    Args:
        seed: The integer that fully determines the scenario.
        topology_requests: The production configuration boundaries this runner can afford.
        full_worker: Whether to enforce subprocess fault-targeting constraints.

    Returns:
        The frozen scenario description.
    """
    rng = random.Random(seed)

    card = rng.choice(CARDS)
    servable = _servable_models(card)
    shape = rng.choice(list(ChaosQueueShape))
    length = rng.choice(_QUEUE_LENGTHS)
    models = _draw_queue(rng, shape=shape, servable=servable, length=length)
    arrival = rng.choice(list(ChaosArrival))
    burst_size = rng.randint(1, max(1, length - 1))
    requested_threads, requested_queue_size = rng.choice(topology_requests)
    topology = _resolve_topology(
        max_threads=requested_threads,
        requested_queue_size=requested_queue_size,
        num_models_to_load=len(set(models)),
        card=card if full_worker else None,
    )
    uniform_size = rng.choice(_servable_sizes(card, models))
    events = _draw_events(rng, job_count=length)
    if full_worker:
        child_events = [event for event in events if event.kind in CHILD_EVENT_KINDS]
        events = tuple(child_events[:1])
        if child_events and topology.lanes != 1:
            topology = _resolve_topology(
                max_threads=1,
                requested_queue_size=0,
                num_models_to_load=len(set(models)),
                card=card,
            )

    demand_shape = rng.choice(list(ChaosDemandShape))
    jobs = _draw_jobs(
        rng,
        feature_rng=random.Random(f"{seed}:job-features"),
        card=card,
        models=models,
        demand_shape=demand_shape,
        uniform_size=uniform_size,
    )
    initial_residency = rng.choice(list(ChaosInitialResidency))
    if any(event.kind is ChaosEventKind.EXTERNAL_RECLAIM for event in events):
        initial_residency = ChaosInitialResidency.FOREIGN_IN_VRAM
        if topology.lanes < 2:
            topology = _resolve_topology(
                max_threads=1,
                requested_queue_size=1,
                num_models_to_load=len(set(models)),
                card=card if full_worker else None,
            )

    # The dispatch-time axes are drawn from their own seed-derived streams rather than from the sequence
    # above, so adding them leaves every previously generated seed denoting the scenario it always did. Each
    # is then admitted only where the card prices the queue as still servable under it, which is what keeps
    # one uniform positive verdict over a space whose card state these axes deliberately tighten.
    sibling_residency = ChaosSiblingResidency.NONE
    service_topology = ChaosServiceTopology.BARE
    activation_shape = ChaosActivationShape.NOMINAL
    disaggregation_class = False
    probe_measured_marginal = False
    if not full_worker:
        foreign = _foreign_sibling_model(models)
        if (
            topology.lanes >= 2
            and foreign is not None
            and random.Random(f"{seed}:sibling-residency").choice((False, True))
        ):
            sibling_residency = ChaosSiblingResidency.FOREIGN_IDLE_IN_VRAM
        foreign_resident_mb = (
            foreign.resident_weights_mb
            if foreign is not None and sibling_residency is not ChaosSiblingResidency.NONE
            else 0.0
        )
        if random.Random(f"{seed}:service-topology").choice((False, True)) and _all_jobs_priced_to_fit(
            jobs,
            sampling_headroom_mb(
                card,
                lanes=topology.lanes,
                service_contexts=True,
                foreign_resident_mb=foreign_resident_mb,
            ),
        ):
            service_topology = ChaosServiceTopology.SERVICE_CONTEXTS
        if random.Random(f"{seed}:activation-shape").choice((False, True)):
            headroom_mb = sampling_headroom_mb(
                card,
                lanes=topology.lanes,
                service_contexts=service_topology is ChaosServiceTopology.SERVICE_CONTEXTS,
                foreign_resident_mb=foreign_resident_mb,
            )
            upgraded = tuple(hires_upgrade(job, headroom_mb) for job in jobs)
            if upgraded != jobs:
                jobs = upgraded
                activation_shape = ChaosActivationShape.HIRES_BATCH
        # Disaggregation is a class the whole-card class is never eligible for, and a scenario's jobs are
        # eligible together or not at all: the runner pins class-eligibility, not a per-job property.
        if MODEL_HEAVY not in models and random.Random(f"{seed}:disaggregation-class").choice((False, True)):
            disaggregation_class = True
        # A head already resident on an idle lane is the only state from which the dispatch-time whole-card
        # decision is reachable, and a free draw reaches it in a quarter of scenarios. Half the empty-card
        # draws are converted to it where the card carries more than one lane and has also been tightened,
        # which is what makes the neighbourhood dense enough to generate over. Every other starting state is
        # drawn exactly as often as before, as is every empty card on which the decision would have nothing
        # to discriminate: a single-lane pool has no context for a reduction to take, and an untightened card
        # is not the state a claim is contested on. The single-lane boundary and the untightened card are
        # still reached wherever the residency draw lands on a resident head by itself.
        tightened = (
            sibling_residency is not ChaosSiblingResidency.NONE
            or service_topology is not ChaosServiceTopology.BARE
            or activation_shape is not ChaosActivationShape.NOMINAL
        )
        if (
            initial_residency is ChaosInitialResidency.EMPTY
            and tightened
            and topology.lanes >= 2
            and random.Random(f"{seed}:dispatch-residency").choice((False, True))
        ):
            initial_residency = ChaosInitialResidency.HEAD_IN_VRAM
        # The probe measurement is a host property, not a scheduling choice, and it is a precondition of the
        # decision this neighbourhood exists to reach rather than an axis to vary inside it: a card-light
        # head's teardown demand is declined outright where the marginal was never measured. It is therefore
        # fixed to measured exactly where the neighbourhood is constructed, and left as the unmeasured
        # fallback everywhere else, so no scenario outside the neighbourhood changes meaning.
        probe_measured_marginal = (
            initial_residency is ChaosInitialResidency.HEAD_IN_VRAM and tightened and topology.lanes >= 2
        )

    return ChaosScenario(
        seed=seed,
        card=card,
        shape=shape,
        jobs=jobs,
        arrival=arrival,
        burst_size=burst_size,
        topology=topology,
        demand_shape=demand_shape,
        initial_residency=initial_residency,
        events=events,
        enable_vram_budget=random.Random(f"{seed}:runtime-vram-budget").choice((False, True)),
        whole_card_enabled=random.Random(f"{seed}:runtime-whole-card").choice((False, True)),
        performance=random.Random(f"{seed}:runtime-performance").choice(list(ChaosPerformance)),
        unload_models_from_vram_often=random.Random(f"{seed}:runtime-unload").choice((False, True)),
        sibling_residency=sibling_residency,
        service_topology=service_topology,
        activation_shape=activation_shape,
        disaggregation_class=disaggregation_class,
        probe_measured_marginal=probe_measured_marginal,
    )


def generate_scenario(seed: int) -> ChaosScenario:
    """Return a scheduling-loop scenario for ``seed`` over the full topology boundary vocabulary."""
    return _generate_scenario(seed, topology_requests=_TOPOLOGY_REQUESTS, full_worker=False)


def generate_scenarios(seeds: tuple[int, ...]) -> tuple[ChaosScenario, ...]:
    """Return the scenarios for a list of seeds, in the order given."""
    return tuple(generate_scenario(seed) for seed in seeds)


def generate_full_worker_scenarios(seeds: tuple[int, ...]) -> tuple[ChaosScenario, ...]:
    """Return subprocess scenarios with production topology boundaries that keep the sweep affordable.

    Child-side disturbances are restricted to one per scenario and a single inference lane. This makes a
    fault profile's process-local job ordinal identical to the scenario's global ordinal; the subprocess
    runner rejects any scenario that violates that mapping.
    """
    return tuple(
        _generate_scenario(
            seed,
            topology_requests=_FULL_WORKER_TOPOLOGY_REQUESTS,
            full_worker=True,
        )
        for seed in seeds
    )


def parse_seed_spec(spec: str) -> tuple[int, ...]:
    """Parse a seed specification into a seed list.

    Args:
        spec: Either an inclusive-exclusive range (``"100:200"``) or a comma list (``"7,19,23"``).

    Returns:
        The seeds the specification names, in order.

    Raises:
        ValueError: If the specification is empty, malformed, or names an empty range.
    """
    text = spec.strip()
    if not text:
        raise ValueError("seed specification is empty")
    if ":" in text:
        start_text, _, stop_text = text.partition(":")
        start, stop = int(start_text), int(stop_text)
        if stop <= start:
            raise ValueError(f"seed range {text!r} is empty: stop must be greater than start")
        return tuple(range(start, stop))
    return tuple(int(part) for part in text.split(",") if part.strip())


CORE_SEEDS: tuple[int, ...] = (
    1,
    2,
    3,
    7,
    12,
    19,
    23,
    31,
    44,
    57,
    68,
    73,
    86,
    91,
    104,
    117,
    129,
)
"""The representative slice the default suite runs.

Chosen (not drawn) so the slice spans the axes rather than whatever a contiguous range happens to hit:
between them these seeds cover all cards, queue and demand shapes, topology boundaries, initial residency
states, arrivals, event kinds, and event counts. ``test_the_core_slice_spans_the_generated_axes`` holds that
coverage, so a draw-order change that collapses the slice fails loudly instead of quietly narrowing it.
"""

DISPATCH_RESIDENCY_SEEDS: tuple[int, ...] = (
    2050,
    2056,
    2065,
    2073,
    2116,
    2132,
    2167,
    2189,
    2199,
    2205,
    2221,
    2252,
    2264,
    2274,
    2288,
    2297,
    2325,
    2329,
    2354,
    2367,
    2421,
    2432,
    2464,
    2465,
    2531,
    2563,
    2568,
    2594,
    2605,
    2613,
    2623,
    2701,
    2783,
    2799,
    2833,
    2886,
    2988,
    3004,
    3011,
    3050,
)
"""A band selected for the dispatch-time whole-card decision rather than drawn as a contiguous range.

The decision is only reachable for a head whose weights a lane already holds when it is chosen to sample, on
a worker with VRAM budgeting and whole-card residency both on, and it discriminates only where something has
tightened the card. Those conditions coincide in a few per cent of a free draw, so a contiguous sweep judges
the neighbourhood a handful of times however wide it is.

This band is the first forty seeds at or above 2000 whose generated scenario satisfies all of: the head
starts resident in VRAM, whole-card residency is on, VRAM budgeting is on, and at least one of the service
tenants, a hires-class geometry, or an idle sibling holding foreign weights is present. The band's
composition (all three tightenings present, with and without disaggregation-class eligibility) is asserted by
the suite, so a draw-order or eligibility change that empties it fails loudly rather than running forty
scenarios that no longer reach the decision.
"""

SWEEP_SEEDS: tuple[int, ...] = tuple(range(1000, 1250))
"""The scheduling-loop sweep's committed range: wide enough to reach combinations the slice never draws."""

FULL_WORKER_CANDIDATE_SEEDS: tuple[int, ...] = tuple(range(1000, 1024))
"""The finite candidate corpus used to select the spawned-process smoke rows."""

SWEEP_FULL_WORKER_SEEDS: tuple[int, ...] = (1000, 1005, 1006, 1010, 1013, 1018, 1020)
"""The full-worker sweep's committed spawned-process rows.

Together these rows preserve every configuration, queue-shape, system-state, payload, arrival, topology,
and child-disturbance value drawn by :data:`FULL_WORKER_CANDIDATE_SEEDS`. The larger scheduling-loop sweep
still judges hundreds of compositions without process startup. This tier spends fresh interpreter startup
only where it adds process-boundary evidence: ordinary IPC, multi-lane topology, child replacement, slow
execution, resource failure, and stale-message rejection.
"""


DISCLOSED_BOUNDS: tuple[tuple[str, str], ...] = (
    (
        "unservable heads are outside the generated space",
        "a scenario only queues models its card can serve, at a geometry that card serves them at, so every "
        "generated verdict is a positive one. The ceiling-hold exit a genuinely unservable head takes is the "
        "bounded-dispatch matrix's subject; advertised-demand contradictions have pinned boundary contracts.",
    ),
    (
        "the mid class is generated on the smallest card only up to 768x768",
        "above that geometry its priced sampling peak exceeds what the card can seat at all, so such a job "
        "would be an unservable head: the bound above, and the bounded-dispatch matrix's subject. Up to it the "
        "class is generated at every geometry, including the ones the card serves only by converging and "
        "taking one real measured load.",
    ),
    (
        "the model vocabulary is four checkpoints in three weight classes",
        "the axis that matters to admission is the weight class (light, mid, whole-card), not the count of "
        "distinct names; two light checkpoints are carried so residency rotation between same-class models is "
        "still reachable.",
    ),
    (
        "queue length uses the semantic boundaries 2, 3, 5, and 8",
        "these reach a follower, the default retry boundary plus one, the configured queue boundary plus one, "
        "and repeated residency/recovery cycles. Longer stochastic sequences remain a nightly widening axis.",
    ),
    (
        "image-job features use semantic representatives rather than every SDK field value",
        "the generated vocabulary crosses source-image and masked jobs, LoRA and textual-inversion references, "
        "ControlNet annotation/delivery modes, post-processing chains, batches, hires-fix, and three sampler "
        "profiles. Exact numeric sampler knobs and custom workflow documents stay in focused contract tests.",
    ),
    (
        "alchemy jobs are outside the image-job generated space",
        "alchemy has a separate queue, coordinator, process role, and generated source; mixing its forms into "
        "an image queue would not exercise the same scheduling path.",
    ),
    (
        "multi-card hosts are not generated",
        "every stateful scenario runs one card. A compact pop-to-routeability contract crosses one-, two-, "
        "and three-card topologies, heterogeneous model/feature/limit profiles, and dynamic feature "
        "withholding without process startup. A dedicated spawned-worker canary crosses heterogeneous "
        "card-scoped offers with server-style request matching and exact dispatch routing; a second canary "
        "replaces inference children while both non-equivalent card routes carry accepted work.",
    ),
    (
        "runtime settings use the scheduling choices that alter this runner's state transitions",
        "the generated rows vary VRAM budgeting, whole-card residency, normal/moderate/high performance, eager "
        "VRAM unload, thread count, and queue depth. Safety placement, process exit policy, sampling leases, "
        "and dedicated post-processing topology remain in their focused stateful or exhaustive contracts.",
    ),
    (
        "the scheduling-loop tier expresses only the lane-death and outside-reclaim disturbances",
        "its children are advanced by hand a tick at a time, so a slow child, an out-of-memory result, and a "
        "misrouted message have no surface there. Those are expressed at the full-worker tier.",
    ),
    (
        "the full-worker tier expresses only the child-side disturbances",
        "an outside reclaim is an actor operating on the card's residency, which a fake child cannot stage.",
    ),
    (
        "the full-worker tier limits requested sampling concurrency to one or two",
        "its lane count is still resolved by the real manager and checked against the scenario, but larger "
        "thread boundaries stay in the fake-clock tier to avoid spawning dozens of processes per row.",
    ),
    (
        "the dispatch-time axes are generated at the scheduling-loop tier only",
        "a head already resident on an idle lane, an idle sibling holding foreign weights, and the service "
        "tenants' card charges are all states a runner seeds onto a modelled card before the queue arrives. "
        "The full-worker tier's card state is whatever its real children reach, and its service placement is "
        "a process-topology choice its committed rows hold fixed, so it generates these axes at their inert "
        "values.",
    ),
    (
        "disaggregation is expressible at the scheduling-loop tier only, and never beside the whole-card class",
        "the scheduling runner pins the scheduler's own class-eligibility seam, which is what makes a job "
        "priced, admitted, and dispatched UNet-only without standing up the orchestrator's encode and decode "
        "lanes. The full-worker tier would have to run those lanes for real, so it generates the axis off "
        "rather than faking eligibility over children that sample whole jobs. A queue carrying the whole-card "
        "class is excluded because that class is not disaggregation-eligible in the first place.",
    ),
    (
        "a tightening axis is admitted only where the queue still prices as servable under it",
        "the service tenants' charges, a foreign sibling's weights, and a hires-class geometry each take room "
        "off the card. Each is taken only when every queued job's predicted sampling peak (priced through the "
        "worker's own estimator) still fits what is left, so the space keeps one uniform positive verdict "
        "instead of generating heads it has quietly made unservable. The peak is priced from geometry, batch, "
        "and the activation-bearing payload features; a fixed allowance stands in for the reserve floors and "
        "load transients a scenario description cannot enumerate.",
    ),
    (
        "the host's per-additional-context measurement is fixed, not varied",
        "it is a precondition of the dispatch-time decision rather than a choice inside it: where the probe "
        "measured nothing, a card-light head's teardown demand is declined outright as an over-counted-context "
        "phantom and the process-count-reduction path is unreachable however the card is arranged. Scenarios "
        "that construct the dispatch-time neighbourhood are generated on a host that measured it; every other "
        "scenario keeps the unmeasured fallback the space has always modelled.",
    ),
    (
        "the reduction branch is reached, but its refusal is not discriminated by outcome here",
        "a refusal only changes what happens on a card squeezed enough that a moderate head's weights miss the "
        "siblings-present floor while the budget still says more contexts fit. That combination needs a pool "
        "provisioned past what the card can carry beside the service tenants, and such a head is not servable "
        "at its configured topology at all, which is this space's first bound rather than a wedge inside it. "
        "The generated scenarios reach the branch and the disaggregation-class exemption is what withholds "
        "the claim there; the refusal itself is pinned by the ledger's own contract tests.",
    ),
    (
        "hires-class geometry is generated for the light and mid classes only",
        "the whole-card class's weights fill the card on their own, so growing its activation says nothing "
        "further about the activation-versus-residency distinction the axis exists to vary, and that class's "
        "payload vocabulary excludes the batching the upgrade applies.",
    ),
    (
        "hang, crash-on-start, and preload-stall faults are outside the generated space",
        "each is detected by waiting out a watchdog window with a floor of fifteen seconds, so generating them "
        "would price the sweep in watchdog timeouts rather than in scenarios. They are covered as named "
        "probes in the hand-written chaos suite.",
    ),
)
"""Every way the generated space is fixed or truncated, with the reason. The suites print this, so a reader
of a green sweep can see what it did not explore."""
