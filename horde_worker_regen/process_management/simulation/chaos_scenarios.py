"""Seeded generation of worker-load scenarios for the generated chaos sweep.

A scenario is a composition of three things a wedge needs in order to happen: a queue structure (how much
work arrives, for which models, in what order and on what schedule), a worker configuration (how many
sampling slots and lanes it has, how deep its queue is, what card it runs on), and a schedule of
disturbances fired part-way through (a lane dying, an outside reclaim, a slow child, a resource fault, a
misrouted message). An integer seed fully determines all three, so a red run replays from its seed alone.

The generator is pure: it draws from ``random.Random(seed)`` and returns a frozen description. It commits
to no altitude. Two runners consume it, and each maps what it can express:

- the scheduling-loop runner drives the composition over a modelled card on a fake clock, where a lane
  death and an outside reclaim are expressible and the child-side faults are not;
- the full-worker runner drives it against real child processes scripted with a
  :class:`~horde_worker_regen.process_management.simulation.fault_injection.FaultProfile`, where the
  child-side faults are expressible and an outside reclaim is not.

:data:`WORLD_EVENT_KINDS` and :data:`CHILD_EVENT_KINDS` state that split, so a runner never silently drops
an event it cannot express. :data:`DISCLOSED_BOUNDS` states every axis the generated space fixes or
truncates, and why; the suites print it, so the sweep's coverage is never taken on trust.

Model identity carries two names because the two altitudes resolve models differently. The scheduling-loop
runner prices from a synthetic reference keyed by ``scheduler_name``; the full-worker runner resolves
``harness_name`` through the real image-model reference when one is present, and through a name-keyed
fallback otherwise. Both names denote the same weight class in the space this module generates.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass

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
    """

    label: str
    scheduler_name: str
    harness_name: str
    min_card_vram_mb: int
    weight_rank: int


MODEL_LIGHT_A = ChaosModel("light_a", "sd15-checkpoint", "Deliberate", 0, 0)
MODEL_LIGHT_B = ChaosModel("light_b", "sd15-checkpoint-b", "Anything Diffusion", 0, 0)
MODEL_MID = ChaosModel("mid", "sdxl-checkpoint", "Juggernaut XL", 0, 1)
MODEL_HEAVY = ChaosModel(
    "heavy",
    "Flux.1-Schnell fp8 (Compact)",
    "Flux.1-Schnell fp8 (Compact)",
    16384,
    2,
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
    """One disturbance, anchored to a job ordinal so both altitudes can place it.

    Attributes:
        kind: What happens.
        at_job_ordinal: The 1-based job ordinal the disturbance is anchored to. A full-worker run scripts
            its child against that ordinal directly; the scheduling-loop runner converts it to a tick.
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
        models: The queued jobs' models, head first.
        arrival: How the jobs become available.
        burst_size: Jobs released per burst (bursts arrival only).
        max_threads: The concurrent-sampling cap.
        lanes: How many inference lanes the pool holds.
        queue_size: The configured local queue depth.
        width: Generated job width in pixels.
        height: Generated job height in pixels.
        steps: Generated job sampling steps.
        events: The disturbance schedule, ordered by job ordinal.
    """

    seed: int
    card: ChaosCard
    shape: ChaosQueueShape
    models: tuple[ChaosModel, ...]
    arrival: ChaosArrival
    burst_size: int
    max_threads: int
    lanes: int
    queue_size: int
    width: int
    height: int
    steps: int
    events: tuple[ChaosEvent, ...]

    @property
    def job_count(self) -> int:
        """How many jobs the scenario queues."""
        return len(self.models)

    @property
    def model_switches(self) -> int:
        """How many times the queue changes model, which is how often residency must rotate."""
        return sum(1 for before, after in zip(self.models, self.models[1:], strict=False) if before is not after)

    @property
    def heavy_job_count(self) -> int:
        """How many queued jobs want the whole-card class."""
        return sum(1 for model in self.models if model is MODEL_HEAVY)

    @property
    def label(self) -> str:
        """A stable, readable identity for the scenario (used as the parametrized test id)."""
        return f"seed{self.seed}-{self.card.label}-{self.shape.value}-t{self.max_threads}-n{self.job_count}"

    def world_events(self) -> tuple[ChaosEvent, ...]:
        """The disturbances the scheduling-loop runner can express."""
        return tuple(event for event in self.events if event.kind in WORLD_EVENT_KINDS)

    def child_events(self) -> tuple[ChaosEvent, ...]:
        """The disturbances the full-worker runner can express."""
        return tuple(event for event in self.events if event.kind in CHILD_EVENT_KINDS)

    def summary(self) -> str:
        """A one-line description carrying everything needed to reproduce and read the scenario."""
        queue = ">".join(model.label for model in self.models)
        events = (
            ",".join(f"{event.kind.value}@{event.at_job_ordinal}" for event in self.events) if self.events else "none"
        )
        return (
            f"seed={self.seed} card={self.card.label} shape={self.shape.value} queue=[{queue}] "
            f"arrival={self.arrival.value}(burst={self.burst_size}) threads={self.max_threads} "
            f"lanes={self.lanes} queue_size={self.queue_size} job={self.width}x{self.height}@{self.steps} "
            f"events=[{events}]"
        )


MAX_QUEUE_SIZE = 4
"""The deepest local queue an operator can configure, mirroring the bridge data's own ceiling.

A generated worker configuration has to be one the worker would accept, or the scenario cannot be run at
the full-worker altitude at all.
"""

_QUEUE_LENGTHS: tuple[int, ...] = (2, 3, 4, 5)
_MAX_THREADS: tuple[int, ...] = (1, 2)
_LANE_COUNTS: tuple[int, ...] = (2, 3)
_JOB_SIZES: tuple[tuple[int, int, int], ...] = (
    (512, 512, 8),
    (512, 768, 12),
    (768, 768, 12),
    (1024, 1024, 8),
)
"""Generated job geometries. Steps stay low so a full-worker run's fake children finish promptly; the axis
varies the priced sampling peak, which is what admission reads, not how long a fake pretends to sample."""

_EVENT_COUNT_DRAW: tuple[int, ...] = (0, 0, 1, 1, 1, 2)
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
    """Draw the disturbance schedule, at most one disturbance per job ordinal."""
    count = min(rng.choice(_EVENT_COUNT_DRAW), job_count)
    if count == 0:
        return ()
    ordinals = sorted(rng.sample(range(1, job_count + 1), count))
    kinds = list(ChaosEventKind)
    return tuple(ChaosEvent(kind=rng.choice(kinds), at_job_ordinal=ordinal) for ordinal in ordinals)


def generate_scenario(seed: int) -> ChaosScenario:
    """Return the scenario the given seed denotes.

    The draw order is part of the contract: changing it renumbers every seed, so a committed seed list and
    the failures it has pinned would no longer mean the same scenarios. Add axes at the end of the draw
    rather than in the middle.

    Args:
        seed: The integer that fully determines the scenario.

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
    max_threads = rng.choice(_MAX_THREADS)
    lanes = rng.choice(_LANE_COUNTS)
    # A queue configured shallower than the work offered is the at-depth shape: intake is gated on the
    # local queue draining, which is where a full-but-frozen queue would hide. The worker's own config
    # caps the depth, so the draw is clamped to what an operator could actually set.
    deep = min(length, MAX_QUEUE_SIZE)
    queue_size = rng.choice((deep, max(1, deep - 1)))
    width, height, steps = rng.choice(_servable_sizes(card, models))
    events = _draw_events(rng, job_count=length)

    return ChaosScenario(
        seed=seed,
        card=card,
        shape=shape,
        models=models,
        arrival=arrival,
        burst_size=burst_size,
        max_threads=max_threads,
        lanes=lanes,
        queue_size=queue_size,
        width=width,
        height=height,
        steps=steps,
        events=events,
    )


def generate_scenarios(seeds: tuple[int, ...]) -> tuple[ChaosScenario, ...]:
    """Return the scenarios for a list of seeds, in the order given."""
    return tuple(generate_scenario(seed) for seed in seeds)


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
between them these seeds cover all three cards, all four queue shapes, both sampling caps, both lane
counts, every arrival kind, every event kind, and both the undisturbed and two-disturbance ends of the
event draw. ``test_the_core_slice_spans_the_generated_axes`` holds that coverage, so a change to the draw
order that collapses the slice fails loudly instead of quietly narrowing the default suite.
"""

SWEEP_SEEDS: tuple[int, ...] = tuple(range(1000, 1250))
"""The scheduling-loop sweep's committed range: wide enough to reach combinations the slice never draws."""

SWEEP_FULL_WORKER_SEEDS: tuple[int, ...] = tuple(range(1000, 1024))
"""The full-worker sweep's committed range.

Each of these boots a worker and spawns real child processes, so the range is a deliberate truncation of
:data:`SWEEP_SEEDS` (its first 24 scenarios) rather than a second sample: the two tiers then judge the same
compositions at both altitudes, and a seed red in one can be read against the other.
"""


DISCLOSED_BOUNDS: tuple[tuple[str, str], ...] = (
    (
        "unservable heads are outside the generated space",
        "a scenario only queues models its card can serve, at a geometry that card serves them at, so every "
        "generated verdict is a positive one. The ceiling-hold exit a genuinely unservable head takes is the "
        "bounded-dispatch matrix's subject.",
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
        "queue length is bounded to 2-5 jobs",
        "a longer queue repeats the same residency rotations at higher cost; depth of composition is varied "
        "through the shape, arrival, and disturbance axes instead.",
    ),
    (
        "no alchemy, LoRA, controlnet, or post-processing traffic is generated",
        "those paths have their own suites, and mixing them in would make a red seed's cause ambiguous "
        "between the scheduling composition under test and the auxiliary path.",
    ),
    (
        "multi-card hosts are not generated",
        "every scenario runs one card. Card-count topology is the canary simulations' axis.",
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
        "the full-worker tier does not vary lane count",
        "the pool size follows from the sampling cap there; lane count is varied at the scheduling-loop tier, "
        "where the pool is constructed directly.",
    ),
    (
        "hang, crash-on-start, and preload-stall faults are outside the generated space",
        "each is detected by waiting out a watchdog window with a floor of fifteen seconds, so generating them "
        "would price the sweep in watchdog timeouts rather than in scenarios. They are covered as named "
        "probes in the hand-written chaos suite.",
    ),
    (
        "a full-worker run without a real image-model reference prices mid-class checkpoints as light ones",
        "the name-keyed fallback recognises the whole-card checkpoint and treats everything else as the light "
        "class, so at that tier the weight-class axis genuinely varies between light and whole-card only.",
    ),
)
"""Every way the generated space is fixed or truncated, with the reason. The suites print this, so a reader
of a green sweep can see what it did not explore."""
