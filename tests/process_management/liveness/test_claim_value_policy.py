"""The pop claim acts on what its model earns, as a property over the whole scheduling loop.

A held whole-card grant normally narrows what the worker asks the horde for to the grant's model. The worker
measures what each model earns per card second, and a model that earns below the rest of its card does not get
that claim under the default policy. These rows drive the real scheduler over the dispatch world with a seeded
card earning accumulator, so the verdict is read from figures the way the running worker reads it, and assert
what the worker asks for and serves while the model holds the card.

The rows build on the claim rows of ``test_bounded_dispatch_matrix.py``: a Flux burst on a 16 GB card, with the
cooldown raised so the grant is still standing where each assertion is made. Each tick the horde offers one more
Flux job and one SD 1.5 job through whatever the worker is asking for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.scheduling.card_earning import (
    CARD_EARNING_BUCKET_SECONDS,
    CardEarningAccumulator,
    CardTick,
)
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import (
    _CARD_16GB,
    _FLUX,
    _MODEL_CLASSES,
    _SD15,
    _DispatchWorld,
)

pytestmark = pytest.mark.closed_loop
"""Every row here drives the real scheduler over the simulated world for many ticks."""

_RUN_TICKS = 16
_QUEUE_DEPTH = 3
"""The configured queue size. Each row pops three Flux jobs at the start, as the matrix's burst rows do."""
_INTAKE_TICKS = 3
"""How many ticks the horde offers a further Flux job, as the matrix's burst rows do."""

_FOREIGN_SUPPLY_TICKS = 8
"""How many ticks the horde offers an SD 1.5 job. Past the Flux burst, so foreign work arrives while Flux holds
the card, and short of the run's end, so the queue drains."""

_SEED_BUCKETS = 8
"""Time buckets of seeded figures. Eight readings make a 95% band 2.4 standard errors wide."""

_SEED_FIRST_BUCKET = 3
"""The first seeded bucket's number. The world's clock starts at 10,000 s, after the eighth seeded bucket."""

_CARD = 0
"""The single card's device index, which the card earning store keys by."""


@dataclass(frozen=True)
class _Seed:
    """Represents the figures seeded per bucket for the claimed model and for the rest of the card.

    Attributes:
        model_kudos_per_second: What the claimed model earns per attributed card second.
        rest_kudos_per_second: What the other model earns per attributed card second.
        model_jobs_per_bucket: How many of the claimed model's jobs each bucket counts.
    """

    model_kudos_per_second: float
    rest_kudos_per_second: float
    model_jobs_per_bucket: int


_BELOW_THE_REST = _Seed(model_kudos_per_second=2.0, rest_kudos_per_second=10.0, model_jobs_per_bucket=3)
"""Flux at a fifth of the rest's rate over 24 jobs, which reads below the rest."""
_PAYS = _Seed(model_kudos_per_second=12.0, rest_kudos_per_second=8.0, model_jobs_per_bucket=3)
"""Flux above the rest over 24 jobs, which reads as paying."""
_UNMEASURED = _Seed(model_kudos_per_second=2.0, rest_kudos_per_second=10.0, model_jobs_per_bucket=1)
"""The below-the-rest rates over 8 jobs, under the verdict's job floor, which reads as unmeasured."""


def _seeded_accumulator(seed: _Seed) -> CardEarningAccumulator:
    """Return an accumulator with ``_SEED_BUCKETS`` buckets of figures on the card, before the world's clock.

    In each bucket Flux and SD 1.5 share 450 seconds of the card. The rates alternate 10% above and below the
    seed's figures between buckets, so each rate has a spread and a standard error.
    """
    first_start = _SEED_FIRST_BUCKET * CARD_EARNING_BUCKET_SECONDS
    accumulator = CardEarningAccumulator(max_tick_seconds=1000.0, clock=lambda: float(first_start))
    for index in range(_SEED_BUCKETS):
        start = float(first_start + index * CARD_EARNING_BUCKET_SECONDS)
        accumulator.observe(start, {})
        accumulator.observe(start + 450.0, {_CARD: CardTick(in_flight_models=(_FLUX.name, _SD15.name))})
        swing = 1.1 if index % 2 == 0 else 0.9
        model_kudos = seed.model_kudos_per_second * 225.0 * swing
        for _ in range(seed.model_jobs_per_bucket):
            accumulator.credit_finished_job(
                device_index=_CARD,
                model=_FLUX.name,
                kudos=model_kudos / seed.model_jobs_per_bucket,
            )
        accumulator.credit_finished_job(
            device_index=_CARD,
            model=_SD15.name,
            kudos=seed.rest_kudos_per_second * 225.0 * swing,
        )
    return accumulator


@dataclass
class _RowOutcome:
    """Represents what a row observed over its run.

    Attributes:
        held_ticks: The ticks a whole-card grant for Flux stood on the card.
        foreign_offered_while_held: The ticks the worker asked for SD 1.5 while Flux held the card.
        foreign_taken: The SD 1.5 jobs the worker accepted, with the tick each was taken on.
        foreign_refused_under_claim: The ticks an SD 1.5 job was refused while a claim stood.
        accepted: Every job the worker accepted during the run, for the drain assertion.
    """

    held_ticks: list[int] = field(default_factory=list)
    foreign_offered_while_held: list[int] = field(default_factory=list)
    foreign_taken: dict[str, int] = field(default_factory=dict)
    foreign_refused_under_claim: list[int] = field(default_factory=list)
    accepted: list[ImageGenerateJobPopResponse] = field(default_factory=list)


async def _build_world(
    seed: _Seed | None,
) -> tuple[_DispatchWorld, list[ImageGenerateJobPopResponse]]:
    """Build the burst world with the seeded accumulator and pop three Flux jobs."""
    world = _DispatchWorld(
        card=_CARD_16GB,
        lane_count=2,
        max_threads=1,
        queue_depth=_QUEUE_DEPTH,
        cooldown_seconds=600,
        card_earning=_seeded_accumulator(seed) if seed is not None else None,
    )
    jobs = [make_job_pop_response(_FLUX.name, width=512, height=512, ddim_steps=8) for _ in range(3)]
    for job in jobs:
        await world.pop(job)
    return world, jobs


def _flux_held(world: _DispatchWorld) -> bool:
    """Return whether a whole-card grant for Flux stands on the card."""
    return any(state.model == _FLUX.name for _device_index, state in world.scheduler._whole_card_ledger.held())


async def _drive(world: _DispatchWorld) -> _RowOutcome:
    """Run the row, offering a Flux job on the first ticks and an SD 1.5 job on each tick of the supply window.

    The SD 1.5 supply stops after ``_FOREIGN_SUPPLY_TICKS`` so the queue drains inside the run.
    """
    outcome = _RowOutcome()
    for _ in range(_RUN_TICKS):
        await world.step()
        held = _flux_held(world)
        if held:
            outcome.held_ticks.append(world.tick)
            if _SD15.name in world.offers[world.tick]:
                outcome.foreign_offered_while_held.append(world.tick)
        claim_stands = world.scheduler.whole_card_pop_claim() is not None
        if world.tick <= _INTAKE_TICKS:
            flux_job = make_job_pop_response(_FLUX.name, width=512, height=512, ddim_steps=8)
            if await world.offer_job(flux_job):
                outcome.accepted.append(flux_job)
        if world.tick > _FOREIGN_SUPPLY_TICKS:
            continue
        if len(world.job_tracker.jobs_pending_inference) >= _QUEUE_DEPTH:
            # The popper does not pop while the queue is full.
            continue
        foreign_job = make_job_pop_response(_SD15.name, width=512, height=512, ddim_steps=8)
        if await world.offer_job(foreign_job):
            outcome.accepted.append(foreign_job)
            outcome.foreign_taken[str(foreign_job.id_)] = world.tick
        elif claim_stands:
            outcome.foreign_refused_under_claim.append(world.tick)
    return outcome


def _assert_the_claim_stood(world: _DispatchWorld, outcome: _RowOutcome, label: str) -> None:
    """Assert today's claim: it stood, it narrowed the offer to Flux, and it refused the foreign job."""
    assert world.claim_ticks, f"{label}: the grant never claimed the offer. {world.state_dump()}"
    for tick in world.claim_ticks:
        assert world.offers[tick] == frozenset({_FLUX.name}), (
            f"{label}: at tick {tick} the claim stood but the worker asked for {sorted(world.offers[tick])}. "
            f"{world.state_dump()}"
        )
    assert outcome.foreign_refused_under_claim, (
        f"{label}: no foreign job was refused while the claim stood, so the row does not show the claim acting. "
        f"{world.state_dump()}"
    )
    for job_id, taken_at in outcome.foreign_taken.items():
        assert taken_at not in world.claim_ticks, (
            f"{label}: foreign job {job_id} was taken at tick {taken_at}, while the claim stood. {world.state_dump()}"
        )


async def test_a_below_rest_model_does_not_get_a_claim_and_the_card_serves_the_mix() -> None:
    """A whole-card model that earns below the rest of its card holds the card without a claim.

    Flux reads below the rest. The grant still takes the card for Flux's burst, but the worker keeps asking the
    horde for every model, so SD 1.5 jobs are offered and accepted while Flux holds the card. Every job the worker
    accepted is served by the end of the run. The SD 1.5 jobs run once Flux's jobs drain, since a teardown keeps
    other models off the card until then.
    """
    world, jobs = await _build_world(_BELOW_THE_REST)

    outcome = await _drive(world)

    assert outcome.held_ticks, (
        f"precondition: Flux never held the card, so the row proves nothing. {world.state_dump()}"
    )
    assert world.claim_ticks == [], (
        f"a model that earns below the rest claimed the offer on ticks {world.claim_ticks}. {world.state_dump()}"
    )
    assert outcome.foreign_offered_while_held == outcome.held_ticks, (
        f"while Flux held the card the worker asked for SD 1.5 only on ticks {outcome.foreign_offered_while_held} "
        f"of {outcome.held_ticks}. {world.state_dump()}"
    )
    assert any(taken_at in outcome.held_ticks for taken_at in outcome.foreign_taken.values()), (
        f"no SD 1.5 job was accepted while Flux held the card. {world.state_dump()}"
    )
    for job in jobs + outcome.accepted:
        assert world.dispatch_tick(job) is not None, (
            f"job {job.id_} for {job.model} was accepted and never served. {world.state_dump()}"
        )
    assert world.offers[world.tick] == frozenset(model.name for model in _MODEL_CLASSES)


async def test_a_model_that_pays_keeps_its_claim() -> None:
    """A whole-card model that earns more than the rest of its card keeps today's claim."""
    world, _jobs = await _build_world(_PAYS)

    outcome = await _drive(world)

    _assert_the_claim_stood(world, outcome, "pays")


async def test_an_unmeasured_model_keeps_its_claim() -> None:
    """A model with too few jobs measured keeps today's claim, however far below the rest its rate reads."""
    world, _jobs = await _build_world(_UNMEASURED)

    outcome = await _drive(world)

    _assert_the_claim_stood(world, outcome, "unmeasured")


async def test_a_listed_model_keeps_its_claim_whatever_it_earns() -> None:
    """A below-the-rest model in ``whole_card_claim_always_models`` keeps today's claim."""
    world, _jobs = await _build_world(_BELOW_THE_REST)
    world.scheduler._runtime_config.bridge_data.whole_card_claim_always_models = [_FLUX.name]

    outcome = await _drive(world)

    _assert_the_claim_stood(world, outcome, "listed")


async def test_the_always_policy_keeps_a_below_rest_claim() -> None:
    """With ``whole_card_claim_policy`` at ``always`` a below-the-rest model keeps today's claim."""
    from horde_worker_regen.process_management.scheduling.governance.whole_card import WholeCardClaimPolicy

    world, _jobs = await _build_world(_BELOW_THE_REST)
    world.scheduler._runtime_config.bridge_data.whole_card_claim_policy = WholeCardClaimPolicy.ALWAYS

    outcome = await _drive(world)

    _assert_the_claim_stood(world, outcome, "always")


async def test_a_world_without_an_accumulator_keeps_its_claim() -> None:
    """A scheduler that does not measure card earning reads every verdict as unmeasured, so its claim stands."""
    world, _jobs = await _build_world(None)

    outcome = await _drive(world)

    _assert_the_claim_stood(world, outcome, "no accumulator")
