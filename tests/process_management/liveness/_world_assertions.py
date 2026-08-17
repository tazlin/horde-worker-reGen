"""First-class verdicts over a completed closed-loop world run.

A closed-loop run of :class:`~tests.process_management.liveness._dispatch_world._DispatchWorld` produces
several series a scenario can be judged on: what the device-free governor did, how far down the reclaim
ladder the card was taken, how low device free ever went, how much of the run's slot time was spent sampling,
and what each tick looked like to dispatch. Each verdict here is one question about one of those series,
phrased so a failure names the mechanism rather than a number.

Two of them read the per-tick series rather than the run's totals, because the failures they exist for are
spans rather than averages: a card idle for a hundred seconds against work that fits barely moves a run-average
duty figure, and a dispatch held for an entity that is going nowhere is invisible in a queue that eventually
drains.

The duty figure follows :mod:`horde_worker_regen.analysis.session_duty`'s slot-duty convention rather than
recomputing that module's report: slot-seconds spent in the earning phase, normalised against the concurrent
sampling capacity for the elapsed wall (here, simulated) time. The analysis module reads a worker's own
cumulative counters out of stats files; this reads the same quantity off a simulated run, so a duty floor
means the same thing in both places.
"""

from __future__ import annotations

from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from horde_worker_regen.process_management.resources.reclaim_ladder import ReclaimRungKind
from tests.process_management.liveness._dispatch_world import _DispatchWorld, _TickObservation

_RUNG_ESCALATION_ORDER: tuple[ReclaimRungKind, ...] = (
    ReclaimRungKind.UNLOAD_IDLE_MODEL,
    ReclaimRungKind.RELEASE_IDLE_CACHE,
    ReclaimRungKind.PAUSE_PP_LANE,
    ReclaimRungKind.PAUSE_VAE_LANE,
    ReclaimRungKind.PAUSE_COMPONENT_LANE,
    ReclaimRungKind.SAFETY_OFF_GPU,
)
"""The ladder's kinds in escalation order, so a scenario can name a ceiling rather than a set.

Ordered as
:func:`~horde_worker_regen.process_management.resources.reclaim_ladder.build_reclaim_ladder` emits them.
Reclaiming an idle resident or an allocator cache costs a reload; pausing a lane disables a pipeline stage
the worker advertises; taking safety off the GPU is the last rung before a kill. A scenario that reaches the
lane rungs has stopped serving part of what it promised, which is why that is the line the incident scenarios
draw."""

LANE_TEARDOWN_RUNGS = frozenset(
    {
        ReclaimRungKind.PAUSE_PP_LANE,
        ReclaimRungKind.PAUSE_VAE_LANE,
        ReclaimRungKind.PAUSE_COMPONENT_LANE,
    },
)
"""The rungs that stop a dedicated lane: reaching one means a pipeline stage went off the air for memory."""

FIRST_LANE_TEARDOWN_RUNG = ReclaimRungKind.PAUSE_PP_LANE
"""The shallowest lane pause, and therefore the ceiling a scenario names to forbid every lane teardown.

Named rather than derived from :data:`LANE_TEARDOWN_RUNGS`, which is a set and has no shallowest member."""

SAFETY_TEARDOWN_RUNG = ReclaimRungKind.SAFETY_OFF_GPU
"""The rung that takes the safety classifier off the card, the deepest reclaim short of killing a process."""


def duty_fraction(world: _DispatchWorld) -> float:
    """Return the fraction of the run's slot-time that was spent sampling.

    Slot-seconds sampling over the sampling capacity (the concurrency cap) times the elapsed simulated
    seconds. One means every sampling slot sampled continuously for the whole run.
    """
    capacity = max(1, int(world.scheduler._runtime_config.bridge_data.max_threads))
    elapsed = max(1e-9, world.now - world.started_at)
    return world.sampling_slot_seconds / (capacity * elapsed)


def jobs_per_simulated_hour(world: _DispatchWorld) -> float:
    """Return the completed-job rate the run achieved, in jobs per simulated hour."""
    elapsed = max(1e-9, world.now - world.started_at)
    return world.completed_jobs * 3600.0 / elapsed


def assert_duty_floor(world: _DispatchWorld, floor: float, *, context: str) -> None:
    """Assert the run kept its sampling slots busy for at least ``floor`` of the available slot-time.

    The positive half of every memory verdict here: a worker that avoids every crater by refusing to dispatch
    satisfies a free-VRAM floor perfectly and earns nothing, so a scenario that constrains the card must also
    say what throughput it expected while doing so.
    """
    achieved = duty_fraction(world)
    assert achieved >= floor, (
        f"{context}: sampling slots were busy {achieved:.0%} of the run, under the {floor:.0%} floor "
        f"({world.completed_jobs} jobs, {jobs_per_simulated_hour(world):.0f}/simulated hour). "
        f"{world.state_dump()}"
    )


def assert_governor_never_reached(world: _DispatchWorld, state: GovernorState, *, context: str) -> None:
    """Assert the device-free governor's committed state never got as far as ``state``."""
    reached = [tick for tick, seen in enumerate(world.governor_states, start=1) if seen is state]
    assert not reached, (
        f"{context}: the card reached governor {state.value} at tick(s) {reached[:5]}, so its free VRAM "
        f"crossed a floor the workload was supposed to stay above (low water {world.min_device_free_mb:.0f}MB "
        f"of {world.card.total_mb:.0f}MB). {world.state_dump()}"
    )


def assert_governor_recovered_from(world: _DispatchWorld, state: GovernorState, *, context: str) -> None:
    """Assert the governor did reach ``state`` and was back to HEALTHY by the end of the run.

    The recovery half of a trajectory verdict: a card that is allowed to dip must be shown coming back, or the
    scenario has only established that the worker survived to the end of its ticks.
    """
    assert state in world.governor_states, (
        f"{context}: the card never reached governor {state.value}, so this run says nothing about recovering "
        f"from it (low water {world.min_device_free_mb:.0f}MB). {world.state_dump()}"
    )
    assert world.governor_states[-1] is GovernorState.HEALTHY, (
        f"{context}: the card ended the run at governor {world.governor_states[-1].value} rather than "
        f"recovering to healthy. {world.state_dump()}"
    )


def assert_ladder_stayed_below(world: _DispatchWorld, kind: ReclaimRungKind, *, context: str) -> None:
    """Assert the reclaim ladder never escalated as far as ``kind`` or beyond it.

    Escalation is read through :data:`_RUNG_ESCALATION_ORDER`, so naming the first lane pause forbids every
    lane pause and the safety teardown behind it, which is the shape a scenario actually wants to state.
    """
    ceiling = _RUNG_ESCALATION_ORDER.index(kind)
    breaches = [
        actuation for actuation in world.ladder_actuations if _RUNG_ESCALATION_ORDER.index(actuation.kind) >= ceiling
    ]
    assert not breaches, (
        f"{context}: the reclaim ladder escalated to {kind.value} or beyond ({breaches[:3]}), so the card was "
        f"relieved by taking the worker's own capacity down. {world.state_dump()}"
    )


def assert_free_floor(world: _DispatchWorld, floor_mb: float, *, context: str) -> None:
    """Assert device free never fell below ``floor_mb`` at any tick of the run."""
    assert world.min_device_free_mb >= floor_mb, (
        f"{context}: device free bottomed out at {world.min_device_free_mb:.0f}MB, under the "
        f"{floor_mb:.0f}MB floor on a {world.card.total_mb:.0f}MB card. {world.state_dump()}"
    )


def assert_no_duplicate_vram_copy(world: _DispatchWorld, model: str, *, context: str) -> None:
    """Assert no two lanes carry a committed VRAM copy of ``model`` at once.

    Read off the card's live state, so a scenario that needs the guarantee to hold at every instant calls it
    inside its own drive loop rather than only at the end.
    """
    lanes = world.vram_resident_lanes(model)
    assert len(lanes) <= 1, (
        f"{context}: lanes {lanes} each hold a full copy of {model}'s weights, so the card is carrying the "
        f"same checkpoint twice. {world.state_dump()}"
    )


def assert_no_committed_slot_retired(world: _DispatchWorld, *, context: str) -> None:
    """Assert no shrink took a slot the parent had already committed to.

    A structural verdict read at the retirement itself rather than off the run's outcome: a slot that owns a
    dispatched job, one with a load in flight, and the slot a whole-card pre-stage is loading its head into
    are each committed regardless of what their child is reporting, and a selector that judges idleness by
    child state alone will take any of them. What follows is several ticks downstream (an orphaned job, a
    residency that never converges) and has other causes, so the seam is where this has to be asked.
    """
    assert not world.committed_slot_retirements, (
        f"{context}: a shrink retired a slot the worker was already committed to:\n    "
        + "\n    ".join(world.committed_slot_retirements)
        + f"\n{world.state_dump()}"
    )


DEFAULT_MAX_IDLE_TICKS = 6
"""How many consecutive idle-with-fitting-work ticks a run may show before it is a hold rather than a gap.

Measured against the generated corpus and the bounded-dispatch table rather than chosen: the longest run any
of them produces legitimately is five, and it is a cross-model swap on an eight-gigabyte card with no room to
seat the incoming model beside the outgoing one. Three of those ticks are the reconciliation gate holding the
head while it evicts an idle resident, and the two after them are the incoming model's preload being declined
until the evicted block actually comes back off the card. Everything the world already recognises as a named
window (a residency establishing or restoring, a heavy head loading, a preload in flight, a lane cycling, a
disaggregated encode wait) is excluded before the run is measured, as are the jobs of a head a churn governor
is deferring, so this bound covers only the ticks nothing has claimed."""


def assert_never_idle_with_fitting_work(
    world: _DispatchWorld,
    *,
    max_idle_ticks: int = DEFAULT_MAX_IDLE_TICKS,
    context: str,
) -> None:
    """Assert the card never sat idle for long with pending work its free VRAM covered.

    An idle tick is one where nothing was dispatched, at least one pending job priced through the scheduler's
    own admission arithmetic fits the card's free VRAM, and no window the world recognises as legitimate was
    open. Those windows are the deliberate ones something will close on its own clock: a whole-card residency
    establishing or restoring, a heavy head inside its load grace, a preload actually in flight, a lane
    cycling into or out of the pool, a disaggregated sampler waiting on its encode lane.

    A governor deferral is not among them, and that is the point of the verdict. While a governor holds a
    head's whole-card establishment off the card the head has stood down: normal scheduling is meant to
    continue around it. A card that instead sits empty through the deferral, with smaller work that fits
    sitting in the queue behind that head, is the shape this exists to name, and it is a shape a run-average
    duty floor barely registers.

    Run-length rather than total: a worker that pauses briefly between jobs many times over a long run is
    doing something different from one that stops for a hundred seconds, and only the second is a hold.
    """
    longest: list[_TickObservation] = []
    current: list[_TickObservation] = []
    for observation in world.tick_observations:
        if observation.idle and observation.fitting_pending and not observation.grace_reasons:
            current.append(observation)
            if len(current) > len(longest):
                longest = list(current)
        else:
            current = []
    if len(longest) <= max_idle_ticks:
        return
    first = longest[0]
    last = longest[-1]
    fitting = ", ".join(str(job) for job in first.fitting_pending[:3])
    holds = "\n    ".join(f"tick {observation.tick}: {observation.hold_summary()}" for observation in longest[:6])
    raise AssertionError(
        f"{context}: the card was idle for {len(longest)} consecutive ticks "
        f"({first.tick}-{last.tick}, {last.now - first.now + world.tick_seconds:.0f}s) with "
        f"{first.device_free_mb:.0f}MB of {world.card.total_mb:.0f}MB free and pending work that fits it "
        f"({fitting}), past the {max_idle_ticks}-tick bound. What was recorded as holding those ticks:\n    "
        + holds
        + f"\n{world.state_dump()}",
    )


MAX_UNSERVABLE_HOLD_TICKS = 2
"""How many passes in a row a hold may name no progress and no grace before it is a wait on nothing.

The gate that produces most of these says a card momentarily full while work flows resolves its own holds
"within a pass or two", and that is the tolerance this reproduces in the world's ticks. Measured against the
generated corpus and the bounded-dispatch table, where the longest such run is a single pass: an eight-gigabyte
card whose free VRAM is held by contexts rather than by any resident checkpoint, so the eviction ladder has
nothing to name for a pass before it reaches the context rung. A hold the entity is genuinely going nowhere
behind runs far past this, since nothing about it changes."""


def assert_no_unservable_dispatch_hold(world: _DispatchWorld, *, context: str) -> None:
    """Assert no dispatch was declined for a protected entity that was itself going nowhere.

    The scheduler withholds a ready job on several entities' behalf: a head of queue whose room a line-skip
    must not consume, a whole-card residency held for a model that is not the head, the tenancy a residency
    reconciliation is evicting, an exclusively-admitted over-budget job that owns the card. Each is a
    legitimate thing to wait for while the entity waited for is going somewhere, or while it sits inside a
    grace that is bounded and disclosed.

    What is never legitimate is a hold whose protected entity is doing neither: a head that is cold, not
    loading, and standing down from the card behind a governor deferral; a residency whose model has no job
    and no grace; a reconciliation evicting nothing. Such a hold cannot end by the entity being served,
    because nothing is serving it, so it ends only when something outside gives up. Naming both ends is the
    whole verdict: which ready job was declined, and what the world could see of the entity it was declined
    for.

    Judged over consecutive passes of the same hold rather than at a single pass, because a ladder that
    reaches for its next rung spends a pass proposing nothing and that is not the same thing as a hold with
    nowhere to go.
    """
    runs: dict[tuple[str, str], list[str]] = {}
    longest: list[str] = []
    longest_key: tuple[str, str] | None = None
    for observation in world.tick_observations:
        standing = {
            (hold.kind, hold.held_subject)
            for hold in observation.protected_holds
            if not hold.progressing and hold.grace is None
        }
        for key in list(runs):
            if key not in standing:
                runs.pop(key)
        for hold in observation.protected_holds:
            key = (hold.kind, hold.held_subject)
            if key not in standing:
                continue
            run = runs.setdefault(key, [])
            run.append(f"tick {observation.tick} (free {observation.device_free_mb:.0f}MB): {hold}")
            if len(run) > len(longest):
                longest = list(run)
                longest_key = key
    if len(longest) <= MAX_UNSERVABLE_HOLD_TICKS:
        return
    assert longest_key is not None
    raise AssertionError(
        f"{context}: a {longest_key[0]} hold on job {longest_key[1][:8]} stood for {len(longest)} consecutive "
        f"passes with its protected entity neither progressing nor inside a bounded, disclosed grace, past "
        f"the {MAX_UNSERVABLE_HOLD_TICKS}-pass bound:\n    " + "\n    ".join(longest[:6]) + f"\n{world.state_dump()}",
    )
