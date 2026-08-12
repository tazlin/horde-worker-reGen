"""First-class verdicts over a completed closed-loop world run.

A closed-loop run of :class:`~tests.process_management.liveness._dispatch_world._DispatchWorld` produces four
series a scenario can be judged on: what the device-free governor did, how far down the reclaim ladder the
card was taken, how low device free ever went, and how much of the run's slot time was spent sampling. Each
verdict here is one question about one of those series, phrased so a failure names the mechanism rather than
a number.

The duty figure follows :mod:`horde_worker_regen.analysis.session_duty`'s slot-duty convention rather than
recomputing that module's report: slot-seconds spent in the earning phase, normalised against the concurrent
sampling capacity for the elapsed wall (here, simulated) time. The analysis module reads a worker's own
cumulative counters out of stats files; this reads the same quantity off a simulated run, so a duty floor
means the same thing in both places.
"""

from __future__ import annotations

from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from horde_worker_regen.process_management.resources.reclaim_ladder import ReclaimRungKind
from tests.process_management.liveness._dispatch_world import _DispatchWorld

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
