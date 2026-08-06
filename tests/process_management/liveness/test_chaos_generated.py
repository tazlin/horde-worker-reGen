"""Generated chaos over the scheduling loop: seeded compositions, judged by end-to-end completion.

The bounded-dispatch matrix enumerates compositions a person chose. This module runs compositions nobody
chose: a seed draws a queue structure, a worker configuration, and a schedule of disturbances, and the
resulting scenario is driven through the same modelled card the matrix uses. What it asserts is not a
per-cell judgement but one property, stated once and applied to every scenario:

    All the work the scenario queues is servable, so all of it drains inside a bound derived from the
    scenario's own shape; no job is given up on along the way; and every obligation opened is closed.

Servability is guaranteed by construction rather than assumed: the generator only queues models the
scenario's card can serve (see
:data:`~horde_worker_regen.process_management.simulation.chaos_scenarios.DISCLOSED_BOUNDS`), which is what
lets one verdict cover an unbounded space.

Two tiers run here. The default suite runs the fixed :data:`CORE_SEEDS` slice. The wider committed range
runs under ``-m chaos_sweep``, and ``HORDE_CHAOS_SEEDS`` overrides which seeds either tier draws
(``HORDE_CHAOS_SEEDS=1000:1100`` for a range, ``HORDE_CHAOS_SEEDS=7,19`` for a list), so a red seed replays
with one command:

    pytest tests/process_management/liveness/test_chaos_generated.py -m chaos_sweep

Every failure message carries the seed and the full scenario summary, so a red run is reproducible from the
message alone. The full-worker counterpart of this module is ``tests/e2e/test_chaos_generated_e2e.py``,
which drives the same seeds against real child processes.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from itertools import combinations

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.jobs.job_tracker import JobStage
from horde_worker_regen.process_management.simulation.chaos_scenarios import (
    CORE_SEEDS,
    DISCLOSED_BOUNDS,
    SEED_ENV_VAR,
    SWEEP_SEEDS,
    ChaosArrival,
    ChaosDemandShape,
    ChaosEvent,
    ChaosEventKind,
    ChaosInitialResidency,
    ChaosModel,
    ChaosQueueShape,
    ChaosScenario,
    generate_scenarios,
    parse_seed_spec,
)
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness.test_bounded_dispatch_matrix import (
    _MODEL_CLASSES,
    _CardClass,
    _DispatchWorld,
    _ModelClass,
)

# --------------------------------------------------------------------------------------------------------
# Seed selection
# --------------------------------------------------------------------------------------------------------


def _seeds(default: tuple[int, ...]) -> tuple[int, ...]:
    """Return the seeds to run: the environment override when set, otherwise the committed list."""
    override = os.environ.get(SEED_ENV_VAR)
    return parse_seed_spec(override) if override else default


_CORE_SCENARIOS = generate_scenarios(_seeds(CORE_SEEDS))
_SWEEP_SCENARIOS = generate_scenarios(_seeds(SWEEP_SEEDS))


# --------------------------------------------------------------------------------------------------------
# Translating a scenario into the modelled card's vocabulary
# --------------------------------------------------------------------------------------------------------

_CHAOS_TICK_SECONDS = 10.0
"""How much of the world's clock one scheduling tick advances here.

The matrix drives its rows at a scheduling interval per tick, which reaches the governance windows sized in
tens of seconds that its rows turn on. Generated scenarios also turn on the scheduler's short budgets, and
the shortest of those is the affinity line-skip window with a fifteen-second floor: at a thirty-second tick
that budget is spent in a single advance, so a resident-model job can be let past a cold head exactly once
per head no matter what the queue looks like. Measured, that alone freezes queues the worker serves without
difficulty at any tick of ten seconds or less. Ten seconds samples the shortest budget several times while
still crossing the minute-scale windows inside a scenario's run.

Every budget below is therefore stated in seconds of world time, not in ticks, so the property does not
change meaning if this constant does.
"""

_BASE_BUDGET_SECONDS = 120.0
"""Allowed for the worker to reach its first dispatch: a preload admission, its materialisation, and the
governance the first residency of a run has to pass."""

_SECONDS_PER_DISPATCH = 90.0
"""Allowed per dispatch a sampling slot has to make in sequence."""

_SECONDS_PER_SWITCH = 60.0
"""Allowed per residency rotation the queue's model ordering forces."""

_SECONDS_PER_HEAVY_JOB = 90.0
"""Allowed per whole-card job, which pays an establishment and a restore the light classes do not."""

_SECONDS_PER_EVENT = 120.0
"""Allowed per disturbance: the recovery it forces, plus the reload it costs."""

_SETTLE_SECONDS = 120.0
"""Driven past the budget so the obligation readback describes a settled worker rather than the instant the
last job was dispatched. The age bound is judged against the budget, not against this."""

_ARRIVAL_BURST_INTERVAL_SECONDS = 30.0
"""Between bursts when a scenario's work arrives in bursts."""

_ARRIVAL_STEADY_INTERVAL_SECONDS = 20.0
"""Between arrivals when a scenario's work arrives steadily."""

_EVENT_DELAY_SECONDS = 30.0
"""After its job's release that a disturbance fires, so the worker has committed to a plan first."""


@dataclass(frozen=True)
class _EventReceipt:
    """Proof that a requested modelled-card disturbance changed its intended target state."""

    event: ChaosEvent
    fired_tick: int
    target_model: str


def _model_class(model: ChaosModel) -> _ModelClass:
    """Resolve a generated model to the matrix's model class, which the world prices from."""
    for candidate in _MODEL_CLASSES:
        if candidate.name == model.scheduler_name:
            return candidate
    raise LookupError(f"no modelled card class for {model.scheduler_name!r}")


def _release_offset_seconds(scenario: ChaosScenario, index: int) -> float:
    """Return how long after the run starts a queued job becomes available to the worker."""
    if scenario.arrival is ChaosArrival.STEADY:
        return index * _ARRIVAL_STEADY_INTERVAL_SECONDS
    if scenario.arrival is ChaosArrival.BURSTS:
        return (index // max(1, scenario.burst_size)) * _ARRIVAL_BURST_INTERVAL_SECONDS
    return 0.0


def _release_tick(scenario: ChaosScenario, index: int) -> int:
    """Return the tick a queued job becomes available to the worker."""
    return math.ceil(_release_offset_seconds(scenario, index) / _CHAOS_TICK_SECONDS)


def _dispatch_budget_seconds(scenario: ChaosScenario) -> float:
    """Return the seconds the scenario's work is allowed to take from its release to its dispatch.

    Derived from the scenario's own shape rather than pinned per seed: the sequential dispatches its
    sampling cap forces, the residency rotations its model ordering forces, the whole-card episodes its
    heavy jobs cost, and the recovery each disturbance costs.

    It is a ceiling, not a fit. Across the committed seeds the worst wait any scenario produces is about a
    third of its own bound and the median is a fourteenth, so what this catches is a latency regression of
    several times over, while a queue that stops moving is caught by the completion assertions instead.
    """
    sequential_dispatches = math.ceil(scenario.job_count / scenario.max_threads)
    return (
        _BASE_BUDGET_SECONDS
        + _SECONDS_PER_DISPATCH * sequential_dispatches
        + _SECONDS_PER_SWITCH * scenario.model_switches
        + _SECONDS_PER_HEAVY_JOB * scenario.heavy_job_count
        + _SECONDS_PER_EVENT * len(scenario.world_events())
    )


def _run_ticks(scenario: ChaosScenario) -> int:
    """Return how many ticks to drive: the last arrival, the dispatch budget, and the settle window."""
    last_release = max(
        (_release_offset_seconds(scenario, index) for index in range(scenario.job_count)),
        default=0.0,
    )
    span = last_release + _dispatch_budget_seconds(scenario) + _SETTLE_SECONDS
    return math.ceil(span / _CHAOS_TICK_SECONDS)


class _ChaosRun:
    """One scenario driven to completion over the modelled card, with its per-tick observations kept.

    The observations are what the verdicts read: the oldest waiting job's age at every tick (the census the
    worker's own status surface reports), and the tick each job reached sampling.
    """

    def __init__(self, scenario: ChaosScenario) -> None:
        """Build the world for ``scenario`` without driving it."""
        self.scenario = scenario
        self.world = _DispatchWorld(
            card=_CardClass(scenario.card.label, float(scenario.card.total_vram_mb)),
            lane_count=scenario.lanes,
            max_threads=scenario.max_threads,
            queue_depth=scenario.queue_size,
            tick_seconds=_CHAOS_TICK_SECONDS,
        )
        self.jobs: list[ImageGenerateJobPopResponse] = []
        self.event_receipts: list[_EventReceipt] = []
        self._fired_events: set[ChaosEvent] = set()
        self.oldest_waiting_age_seconds = 0.0
        """The largest age any job reached while still waiting for inference, over the whole run."""
        self._seed_initial_residency()

    def _seed_initial_residency(self) -> None:
        """Materialise the scenario's explicit starting card state before any job arrives."""
        residency = self.scenario.initial_residency
        if residency is ChaosInitialResidency.EMPTY:
            return
        head = _model_class(self.scenario.jobs[0].model)
        if residency is ChaosInitialResidency.HEAD_IN_RAM:
            self.world.seed_resident(0, head, in_vram=False)
            return
        if residency is ChaosInitialResidency.HEAD_IN_VRAM:
            self.world.seed_resident(0, head, in_vram=True)
            return
        foreign = next(
            model
            for model in _MODEL_CLASSES
            if model.name != head.name and model.weights_mb <= self.scenario.card.total_vram_mb - 512
        )
        self.world.seed_resident(0, foreign, in_vram=True)

    async def drive(self) -> None:
        """Pop the queue on its arrival schedule, fire its disturbances, and run out its ticks."""
        await self._release_due_jobs()
        self._fire_due_events()
        for _ in range(_run_ticks(self.scenario)):
            await self.world.step()
            await self._release_due_jobs()
            self._fire_due_events()
            self._observe_waiting_ages()

    async def _release_due_jobs(self) -> None:
        """Pop every job whose arrival tick has come, in queue order (arrival is monotonic in the index)."""
        while len(self.jobs) < self.scenario.job_count and _release_tick(self.scenario, len(self.jobs)) <= (
            self.world.tick
        ):
            job_spec = self.scenario.jobs[len(self.jobs)]
            job = make_job_pop_response(
                _model_class(job_spec.model).name,
                width=job_spec.width,
                height=job_spec.height,
                ddim_steps=job_spec.steps,
            )
            await self.world.pop(job)
            self.jobs.append(job)

    def _fire_due_events(self) -> None:
        """Apply due disturbances and retain only receipts for actions that changed state.

        A preconditioned disturbance remains pending until its target state exists. The final property then
        fails when the runner requested an event but never exercised it, instead of counting a no-op draw as
        coverage.
        """
        for event in self.scenario.world_events():
            if event in self._fired_events or self._event_tick(event) > self.world.tick:
                continue
            target = _model_class(self.scenario.jobs[event.at_job_ordinal - 1].model)
            if event.kind is ChaosEventKind.LANE_DEATH:
                effective = False
                for resident in _MODEL_CLASSES:
                    if self.world.kill_lane_holding(resident):
                        target = resident
                        effective = True
                        break
            else:
                effective = self.world.evict_idle_resident_sibling(except_model=target)
            if not effective:
                continue
            self._fired_events.add(event)
            self.event_receipts.append(
                _EventReceipt(event=event, fired_tick=self.world.tick, target_model=target.name),
            )

    def _event_tick(self, event: ChaosEvent) -> int:
        """Return the tick a disturbance fires on, measured from its own job's arrival."""
        if event.kind is ChaosEventKind.EXTERNAL_RECLAIM:
            return 0
        offset = _release_offset_seconds(self.scenario, event.at_job_ordinal - 1) + _EVENT_DELAY_SECONDS
        return math.ceil(offset / _CHAOS_TICK_SECONDS)

    def _observe_waiting_ages(self) -> None:
        """Record the oldest age any job has reached while still waiting for inference.

        Measured from the tracker's own ``time_popped``, which is the figure the worker's queue-age
        judgements read, rather than from the per-stage census: the census stamps stage entry with wall
        time, so on a world running an injected clock it reports every job as ageless.
        """
        for tracked in self.world.job_tracker.tracked_jobs():
            if tracked.stage is not JobStage.PENDING_INFERENCE or tracked.time_popped is None:
                continue
            self.oldest_waiting_age_seconds = max(
                self.oldest_waiting_age_seconds,
                self.world.now - tracked.time_popped,
            )

    def message(self, complaint: str) -> str:
        """Return a failure message carrying the complaint, the seed, and the whole scenario."""
        return (
            f"{complaint}\n"
            f"  scenario: {self.scenario.summary()}\n"
            f"  replay:   {SEED_ENV_VAR}={self.scenario.seed} pytest "
            f"tests/process_management/liveness/test_chaos_generated.py\n"
            f"  world:    {self.world.state_dump()}"
        )


async def _assert_scenario_is_served(scenario: ChaosScenario) -> None:
    """Drive one scenario and assert the property every generated scenario is held to.

    Args:
        scenario: The generated composition to drive.
    """
    run = _ChaosRun(scenario)
    await run.drive()

    age_bound_seconds = _dispatch_budget_seconds(scenario)

    assert {receipt.event for receipt in run.event_receipts} == set(scenario.world_events()), run.message(
        "one or more requested disturbances never found their required state and changed nothing",
    )

    for index, job in enumerate(run.jobs):
        assert job.id_ is not None
        assert not run.world.job_tracker.was_faulted_by_scheduling_recovery(job.id_), run.message(
            f"queued job {index} ({job.model}) was given up on by scheduling recovery rather than served",
        )
        assert run.world.stage(job) is not JobStage.PENDING_INFERENCE, run.message(
            f"queued job {index} ({job.model}) was still waiting for inference at the end of the run",
        )
        assert run.world.dispatch_tick(job) is not None, run.message(
            f"queued job {index} ({job.model}) never reached sampling",
        )

    assert run.oldest_waiting_age_seconds <= age_bound_seconds, run.message(
        f"a job waited {run.oldest_waiting_age_seconds:.0f}s for inference, past the "
        f"{age_bound_seconds:.0f}s this scenario's shape allows",
    )

    planned_mb = run.world.planned_overlay_mb()
    assert planned_mb == pytest.approx(0.0), run.message(
        f"{planned_mb:.0f} MB of planned preload charge outlived the work it was booked for",
    )
    assert run.world.scheduler.is_whole_card_residency_active() is False, run.message(
        "the card is still held for an exclusive residency after the queue drained",
    )
    assert run.world.scheduler.whole_card_residency_grace_active() is False, run.message(
        "a residency grace window is still open after the queue drained",
    )
    assert run.world.lifecycle.is_safety_gpu_paused is False, run.message(
        "safety was left paused off the GPU after the queue drained",
    )
    assert run.world.lifecycle.is_post_process_gpu_paused is False, run.message(
        "the post-processing lane was left paused after the queue drained",
    )


@pytest.mark.parametrize("scenario", _CORE_SCENARIOS, ids=[scenario.label for scenario in _CORE_SCENARIOS])
async def test_generated_scenario_is_served_end_to_end(scenario: ChaosScenario) -> None:
    """Every generated composition in the core slice drains completely, in bound, with nothing given up."""
    await _assert_scenario_is_served(scenario)


@pytest.mark.chaos_sweep
@pytest.mark.parametrize("scenario", _SWEEP_SCENARIOS, ids=[scenario.label for scenario in _SWEEP_SCENARIOS])
async def test_swept_scenario_is_served_end_to_end(scenario: ChaosScenario) -> None:
    """The same property over the wide committed seed range, run as a pre-release gate and nightly."""
    await _assert_scenario_is_served(scenario)


def test_the_core_slice_spans_the_generated_axes() -> None:
    """The committed slice is representative: it draws every value of every generated axis.

    The slice is what the default suite runs, so a change to the draw order that quietly collapses it onto
    one card or one queue shape must fail here rather than pass as a narrower suite.
    """
    if os.environ.get(SEED_ENV_VAR):
        pytest.skip(f"{SEED_ENV_VAR} overrides the committed slice, so its span is the caller's to choose")

    assert {scenario.card.label for scenario in _CORE_SCENARIOS} == {"8gb", "16gb", "24gb"}
    assert {scenario.shape for scenario in _CORE_SCENARIOS} == set(ChaosQueueShape)
    assert {scenario.arrival for scenario in _CORE_SCENARIOS} == set(ChaosArrival)
    assert {scenario.max_threads for scenario in _CORE_SCENARIOS} == {1, 2, 3, 16}
    assert {scenario.topology.requested_queue_size for scenario in _CORE_SCENARIOS} == {0, 1, 4}
    assert {scenario.queue_size for scenario in _CORE_SCENARIOS} == {0, 1, 3, 4}
    assert {scenario.demand_shape for scenario in _CORE_SCENARIOS} == set(ChaosDemandShape)
    assert {scenario.initial_residency for scenario in _CORE_SCENARIOS} == set(ChaosInitialResidency)
    assert {event.kind for scenario in _CORE_SCENARIOS for event in scenario.events} == set(ChaosEventKind)
    assert {len(scenario.events) for scenario in _CORE_SCENARIOS} >= {0, 1, 2}
    assert any(scenario.heavy_job_count > 0 for scenario in _CORE_SCENARIOS)


def _coverage_features(scenario: ChaosScenario) -> dict[str, object]:
    """Return the independent semantic axes whose pairwise coverage the committed sweep guarantees."""
    return {
        "card": scenario.card.label,
        "queue_shape": scenario.shape,
        "arrival": scenario.arrival,
        "topology_request": (scenario.max_threads, scenario.topology.requested_queue_size),
        "demand_shape": scenario.demand_shape,
        "initial_residency": scenario.initial_residency,
        "queue_length": scenario.job_count,
    }


def test_the_committed_sweep_covers_every_pair_of_semantic_axes() -> None:
    """Every pair of independently valid axis values appears in at least one executed sweep scenario."""
    if os.environ.get(SEED_ENV_VAR):
        pytest.skip(f"{SEED_ENV_VAR} overrides the committed sweep, so its coverage is the caller's to choose")

    rows = [_coverage_features(scenario) for scenario in _SWEEP_SCENARIOS]
    axis_values = {axis: {row[axis] for row in rows} for axis in rows[0]}
    missing: list[str] = []
    for first, second in combinations(axis_values, 2):
        expected = {(a, b) for a in axis_values[first] for b in axis_values[second]}
        actual = {(row[first], row[second]) for row in rows}
        for first_value, second_value in sorted(expected - actual, key=repr):
            missing.append(f"{first}={first_value!r} + {second}={second_value!r}")

    assert not missing, "the committed sweep lost pairwise coverage:\n  " + "\n  ".join(missing)


def test_the_queue_corpus_contains_job_dependent_demand_orderings() -> None:
    """The sweep executes heterogeneous demand within one model, not only model-name permutations."""
    if os.environ.get(SEED_ENV_VAR):
        pytest.skip(f"{SEED_ENV_VAR} overrides the committed sweep, so its coverage is the caller's to choose")

    same_model = [scenario for scenario in _SWEEP_SCENARIOS if len(set(scenario.models)) == 1]
    assert any(
        len(scenario.jobs) >= 3
        and scenario.jobs[0].pixels < scenario.jobs[1].pixels
        and scenario.jobs[2].pixels < scenario.jobs[1].pixels
        for scenario in same_model
    ), "the queue corpus contains no same-model low/high/low demand transition"
    assert any(
        len(scenario.jobs) >= 2 and scenario.jobs[0].pixels > scenario.jobs[1].pixels for scenario in same_model
    ), "the queue corpus contains no same-model high-head/lower-follower transition"
    assert any(
        scenario.queue_size == 0 and scenario.lanes == 1 and scenario.max_threads == 1 for scenario in _SWEEP_SCENARIOS
    ), "the minimum production topology is absent from the executed sweep"


def test_the_sweep_discloses_its_coverage_and_its_bounds(pytestconfig: pytest.Config) -> None:
    """The suite prints what it ran and what it did not explore, so no truncation is silent."""
    reporter = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    core_seeds = _seeds(CORE_SEEDS)
    sweep_seeds = _seeds(SWEEP_SEEDS)
    lines = [
        f"generated chaos (scheduling loop): core slice {len(_CORE_SCENARIOS)} scenarios, "
        f"sweep {len(_SWEEP_SCENARIOS)} scenarios",
        f"  seeds: core {min(core_seeds)}..{max(core_seeds)}, sweep {min(sweep_seeds)}..{max(sweep_seeds)} "
        f"(override with {SEED_ENV_VAR})",
        "  not explored:",
        *(f"    - {axis}: {reason}" for axis, reason in DISCLOSED_BOUNDS),
    ]
    if reporter is not None:
        reporter.write_line("")
        for line in lines:
            reporter.write_line(line)

    assert DISCLOSED_BOUNDS, "the generated space's truncations must be listed, never silent"
    for axis, reason in DISCLOSED_BOUNDS:
        assert axis and reason, "each bound on the generated space must carry its reason"
