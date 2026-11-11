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


# -- The paced return of a below-rest model -----------------------------------------------------------------------

_RETURN_SECONDS = 180.0
"""The popper's ``whole_card_low_earning_return_seconds`` in these rows, its default."""

_PACED_QUEUE_DEPTH = 4
"""The configured queue size of the pacing rows, one above the opening Flux burst so foreign work is taken."""

_STEADY_SUPPLY_TICKS = 30
"""How many ticks the horde offers a Flux job and an SD 1.5 job in the steady-supply rows."""

_DRAIN_TICKS = 12
"""Ticks with no supply after the steady supply, so every accepted job can finish inside the run."""

_FOREIGN_JOB_TTL_SECONDS = 300.0
"""The longest an SD 1.5 job may wait from its pop to its dispatch. The horde gives a job a deadline and drops it
from the worker past it. A foreign job taken while Flux holds the card waits for the Flux jobs ahead of it, which
the per-model queue cap holds to three, and for the card to be given back."""


async def _build_paced_world(
    seed: _Seed | None,
    *,
    flux_jobs: int = 3,
) -> tuple[_DispatchWorld, list[ImageGenerateJobPopResponse]]:
    """Build the burst world with the popper's pacing modelled and pop ``flux_jobs`` Flux jobs.

    The cooldown is the world's default of zero, so a drained grant is released and the card serves the mix.
    """
    world = _DispatchWorld(
        card=_CARD_16GB,
        lane_count=2,
        max_threads=1,
        queue_depth=_PACED_QUEUE_DEPTH,
        card_earning=_seeded_accumulator(seed) if seed is not None else None,
        low_earning_return_seconds=_RETURN_SECONDS,
    )
    jobs = [make_job_pop_response(_FLUX.name, width=512, height=512, ddim_steps=8) for _ in range(flux_jobs)]
    for job in jobs:
        await world.pop(job)
    return world, jobs


def _flux_in_hand(world: _DispatchWorld) -> bool:
    """Return whether a Flux job is pending or in progress."""
    return any(job.model == _FLUX.name for job in world.job_tracker.jobs_pending_inference)


def _queue_full(world: _DispatchWorld) -> bool:
    """Return whether the popper would hold the pop for a full queue."""
    return len(world.job_tracker.jobs_pending_inference) >= _PACED_QUEUE_DEPTH


@dataclass
class _PacedOutcome:
    """Represents what a pacing row observed over its run.

    Attributes:
        accepted: Every job the worker accepted during the run, for the drain assertion.
        foreign_taken: The SD 1.5 jobs the worker accepted.
        flux_taken: The Flux jobs the worker accepted.
        flux_in_hand_by_tick: Per tick, whether a Flux job was pending or in progress after the tick.
    """

    accepted: list[ImageGenerateJobPopResponse] = field(default_factory=list)
    foreign_taken: list[ImageGenerateJobPopResponse] = field(default_factory=list)
    flux_taken: list[ImageGenerateJobPopResponse] = field(default_factory=list)
    flux_in_hand_by_tick: dict[int, bool] = field(default_factory=dict)


def _at_model_queue_cap(world: _DispatchWorld, model_name: str) -> bool:
    """Return whether the popper's per-model queue cap leaves ``model_name`` out of the offer.

    The cap is three pending jobs for a model loaded on a lane and two otherwise, as the popper's stage applies
    it. The world does not model that stage, and without it one model fills the whole queue.
    """
    loaded = any(lane.loaded_horde_model_name == model_name for lane in world._process_map.values())
    pending = sum(job.model == model_name for job in world.job_tracker.jobs_pending_inference)
    return pending >= (3 if loaded else 2)


async def _drive_steady_supply(world: _DispatchWorld) -> _PacedOutcome:
    """Offer a Flux job and then an SD 1.5 job on every supply tick while the queue has room, then drain."""
    outcome = _PacedOutcome()
    for _ in range(_STEADY_SUPPLY_TICKS + _DRAIN_TICKS):
        await world.step()
        outcome.flux_in_hand_by_tick[world.tick] = _flux_in_hand(world)
        if world.tick > _STEADY_SUPPLY_TICKS:
            continue
        for model in (_FLUX, _SD15):
            if _queue_full(world):
                break
            if _at_model_queue_cap(world, model.name):
                continue
            job = make_job_pop_response(model.name, width=512, height=512, ddim_steps=8)
            if not await world.offer_job(job):
                continue
            outcome.accepted.append(job)
            (outcome.flux_taken if model is _FLUX else outcome.foreign_taken).append(job)
    return outcome


def _seconds_waited(world: _DispatchWorld, job: ImageGenerateJobPopResponse) -> float | None:
    """Return the seconds from ``job``'s pop to its dispatch, or None when it never reached sampling."""
    dispatched_tick = world.dispatch_tick(job)
    if dispatched_tick is None:
        return None
    popped_at = world.popped_at[str(job.id_)]
    return dispatched_tick * world.tick_seconds + _world_start(world) - popped_at


def _world_start(world: _DispatchWorld) -> float:
    """Return the world clock before its first tick."""
    return world.now - world.tick * world.tick_seconds


def _paced_ticks(world: _DispatchWorld) -> list[int]:
    """Return the ticks on which the governor's pacing kept Flux out of the offer."""
    return sorted(tick for tick, models in world.paced_withholds.items() if _FLUX.name in models)


async def test_steady_supply_of_a_below_rest_model_does_not_starve_the_other_models() -> None:
    """Steady demand for a below-rest whole-card model does not hold jobs for other models past their deadline.

    Flux reads below the rest, so its grant does not claim the offer and the worker takes SD 1.5 jobs while Flux
    holds the card. The horde offers a Flux job and an SD 1.5 job on every tick. Flux leaves the offer while its
    claim is withheld, so its jobs in hand finish, the grant drains and the SD 1.5 jobs run.
    """
    world, jobs = await _build_paced_world(_BELOW_THE_REST)

    outcome = await _drive_steady_supply(world)

    assert outcome.foreign_taken, f"precondition: no SD 1.5 job was accepted. {world.state_dump()}"
    for job in outcome.foreign_taken:
        waited = _seconds_waited(world, job)
        assert waited is not None, f"SD 1.5 job {job.id_} was accepted and never served. {world.state_dump()}"
        assert waited <= _FOREIGN_JOB_TTL_SECONDS, (
            f"SD 1.5 job {job.id_} waited {waited:.0f}s from its pop to its dispatch, past the "
            f"{_FOREIGN_JOB_TTL_SECONDS:.0f}s deadline. {world.state_dump()}"
        )
    paced = _paced_ticks(world)
    assert paced, f"Flux was never paced, so the row does not show the pacing acting. {world.state_dump()}"
    for tick in paced:
        assert _FLUX.name not in world.offers[tick], (
            f"at tick {tick} Flux was paced but the worker asked the horde for it. {world.state_dump()}"
        )
    for job in jobs + outcome.accepted:
        assert world.dispatch_tick(job) is not None, (
            f"job {job.id_} for {job.model} was accepted and never served. {world.state_dump()}"
        )


async def test_a_paced_model_stays_out_for_the_interval_after_its_last_job_then_returns() -> None:
    """A paced model is not asked for until the interval has passed since its last job finished.

    The horde keeps offering both models, so the worker always has SD 1.5 work and the idle escape does not
    apply, and it does not answer an offer with nothing, so the empty-answer escape does not apply either.
    """
    world, _jobs = await _build_paced_world(_BELOW_THE_REST)

    outcome = await _drive_steady_supply(world)

    paced = _paced_ticks(world)
    assert paced, f"precondition: Flux was never paced. {world.state_dump()}"
    first_paced = paced[0]
    finished_tick = next(
        (
            tick
            for tick in sorted(outcome.flux_in_hand_by_tick)
            if tick >= first_paced and not outcome.flux_in_hand_by_tick[tick]
        ),
        None,
    )
    assert finished_tick is not None, f"precondition: Flux's jobs in hand never finished. {world.state_dump()}"
    returns_after = finished_tick + _RETURN_SECONDS / world.tick_seconds
    for tick in range(first_paced, int(returns_after)):
        assert _FLUX.name not in world.offers[tick], (
            f"at tick {tick} the worker asked the horde for Flux, before {_RETURN_SECONDS:.0f}s had passed since "
            f"its last job finished at tick {finished_tick}. {world.state_dump()}"
        )
    returned = [
        tick for tick in range(int(returns_after), _STEADY_SUPPLY_TICKS + 1) if _FLUX.name in world.offers[tick]
    ]
    assert returned, (
        f"Flux was not asked for again after {_RETURN_SECONDS:.0f}s had passed since its last job finished at "
        f"tick {finished_tick}. {world.state_dump()}"
    )
    assert returned[0] == int(returns_after) or returned[0] == int(returns_after) + 1, (
        f"Flux returned at tick {returned[0]}, later than the interval allows. {world.state_dump()}"
    )
    assert any(
        world.popped_at[str(job.id_)] > _world_start(world) + returned[0] * world.tick_seconds - 1.0
        for job in outcome.flux_taken
    ), f"no Flux job was taken after it returned to the offer. {world.state_dump()}"


async def test_a_paced_model_that_is_the_only_supply_returns_after_an_empty_answer() -> None:
    """When the horde has work only for the paced model, an empty answer returns it to the offer.

    The horde offers only Flux jobs. While Flux is paced the worker asks for the other models and the horde
    answers with nothing. Flux is then asked for again while a Flux job is still in hand, before the worker
    runs out of work, and its jobs are served back to back.
    """
    world, jobs = await _build_paced_world(_BELOW_THE_REST)
    accepted: list[ImageGenerateJobPopResponse] = []
    empty_answer_ticks: list[int] = []
    for _ in range(_STEADY_SUPPLY_TICKS):
        await world.step()
        if _queue_full(world):
            continue
        job = make_job_pop_response(_FLUX.name, width=512, height=512, ddim_steps=8)
        if await world.offer_job(job):
            accepted.append(job)
            continue
        empty_answer_ticks.append(world.tick)
        world.report_pop_answered_empty()
    for _ in range(_DRAIN_TICKS):
        await world.step()

    paced = _paced_ticks(world)
    assert paced, f"precondition: Flux was never paced. {world.state_dump()}"
    assert empty_answer_ticks, f"precondition: the horde never answered an offer with nothing. {world.state_dump()}"
    first_empty = empty_answer_ticks[0]
    assert len(empty_answer_ticks) == 1, (
        f"the horde answered with nothing on ticks {empty_answer_ticks}, so Flux did not stay in the offer after "
        f"the first empty answer. {world.state_dump()}"
    )
    for tick in range(first_empty + 1, _STEADY_SUPPLY_TICKS + 1):
        assert _FLUX.name in world.offers[tick], (
            f"at tick {tick}, after the empty answer at tick {first_empty}, the worker did not ask for Flux. "
            f"{world.state_dump()}"
        )
    assert accepted, f"no Flux job was taken after the empty answer. {world.state_dump()}"
    for job in jobs + accepted:
        assert world.dispatch_tick(job) is not None, (
            f"Flux job {job.id_} was accepted and never served. {world.state_dump()}"
        )


async def test_a_paced_model_is_offered_when_the_worker_has_no_work() -> None:
    """The idle escape returns a paced model as soon as the worker holds no work, inside the interval."""
    world, _jobs = await _build_paced_world(_BELOW_THE_REST)
    idle_ticks: list[int] = []
    for _ in range(_STEADY_SUPPLY_TICKS):
        await world.step()
        if world.job_tracker.num_jobs_total == 0:
            idle_ticks.append(world.tick)

    paced = _paced_ticks(world)
    assert paced, f"precondition: Flux was never paced. {world.state_dump()}"
    assert idle_ticks, f"precondition: the worker never ran out of work. {world.state_dump()}"
    first_idle = idle_ticks[0]
    assert (first_idle - paced[-1]) * world.tick_seconds < _RETURN_SECONDS, (
        f"precondition: the worker ran out of work at tick {first_idle}, after the interval. {world.state_dump()}"
    )
    assert _FLUX.name in world.offers[first_idle], (
        f"the worker held no work at tick {first_idle} and still did not ask the horde for Flux. {world.state_dump()}"
    )


@pytest.mark.parametrize(
    "case",
    ["listed", "always", "pays", "unmeasured"],
)
async def test_a_model_whose_claim_is_not_withheld_is_not_paced(case: str) -> None:
    """An exempt model, any model under policy ``always``, a paying model and an unmeasured model are not paced."""
    from horde_worker_regen.process_management.scheduling.governance.whole_card import WholeCardClaimPolicy

    seed = {"listed": _BELOW_THE_REST, "always": _BELOW_THE_REST, "pays": _PAYS, "unmeasured": _UNMEASURED}[case]
    world, _jobs = await _build_paced_world(seed)
    bridge_data = world.scheduler._runtime_config.bridge_data
    if case == "listed":
        bridge_data.whole_card_claim_always_models = [_FLUX.name]
    if case == "always":
        bridge_data.whole_card_claim_policy = WholeCardClaimPolicy.ALWAYS

    outcome = await _drive_steady_supply(world)

    assert world.claim_ticks, f"{case}: precondition: the grant never claimed the offer. {world.state_dump()}"
    assert _paced_ticks(world) == [], f"{case}: Flux was paced on ticks {_paced_ticks(world)}. {world.state_dump()}"
    assert outcome.flux_taken, f"{case}: no further Flux job was taken. {world.state_dump()}"
