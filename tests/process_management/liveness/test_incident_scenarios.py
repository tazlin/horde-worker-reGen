"""Production failures of the VRAM retention subsystem, encoded as permanent closed-loop scenarios.

Each scenario drives the real scheduler, the real device-free governor and the real verified reclaim ladder
over a conserved card on a virtual clock (see :mod:`tests.process_management.liveness._dispatch_world`), for a
workload whose shape is the one the failure needs. What each asserts is the durable behavior the worker must
show, not the absence of a symptom: weights reused rather than re-uploaded, a card that stays above its floors,
a parent record that matches what the device actually holds.

Every scenario is paired with a *defect reinjection*: a companion that monkeypatches the specific production
decision off and asserts the scenario's load-bearing assertion then fails, in a named shape. A scenario whose
assertion survives its own defect is measuring nothing, so the pair is what keeps these honest.

The failures encoded here:

- **Retention is inert.** Retention is granted, logged and shipped on the dispatch, and for a whole release
  it held no byte on the card, because the child's executor returned the card at the end of every prompt below
  anything the grant could suppress. A same-model streak therefore paid a full host-to-device weight upload per
  job for weights that had never left. That is the ``retained copy`` scenario.
- **No waiter can stage a second copy.** A same-model successor cannot be staged onto a sibling lane while the
  running job holds the model, so no handoff or overlap scheme can hide the per-job re-upload; only same-slot
  reuse of retained weights can. That is a structural claim about routing, and the ``no second copy`` scenario
  pins it, because a diagnosis that attributes the tail-overlap loss to timing sends the fix to the wrong
  layer.
- **A phantom retained resident under the legacy hatch.** Under the escape hatch the child unloads at the end
  of every prompt whatever the dispatch asked for, so a recorded grant is a parent-side record of weights that
  are not on the device. Three other paths then act on that record: the retention static fit charges it, the
  dispatch admission gate holds a load behind it, and same-model routing sends a successor to a slot that holds
  nothing. That is the ``legacy hatch`` scenario.
"""

from __future__ import annotations

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import (
    _CARD_16GB,
    _SDXL,
    _CardClass,
    _DispatchWorld,
    _ModelClass,
)
from tests.process_management.liveness._world_assertions import (
    FIRST_LANE_TEARDOWN_RUNG,
    assert_duty_floor,
    assert_free_floor,
    assert_governor_never_reached,
    assert_ladder_stayed_below,
    assert_no_duplicate_vram_copy,
    duty_fraction,
)

_TICK_SECONDS = 2.0
"""Seconds of simulated time per scheduling tick.

Short enough that a job's phases (a 4.6 s weight upload, a 5.3 s sampling window, a 2 s decode) are each
sampled several times, so a transient peak is observed on the card rather than stepped over, and the governor's
two-sample debounce can commit on a dip that a coarser tick would never see."""

_STREAK_JOBS = 16
"""Jobs in a same-model streak: long enough that a per-job re-upload dominates the run's time budget."""

_QUEUE_DEPTH = 4
"""Jobs the worker is kept holding, so a successor is always available while the retainer is still busy."""

_LEGACY_JOBS = 12
"""Jobs driven through the legacy-hatch scenarios: enough job boundaries that a grant would be recorded on
several of them, and short enough that a run paying a full upload per job settles inside its tick ceiling."""

_SETTLE_TICKS = 10
"""Ticks driven past the last completion so a run's readback describes a settled worker."""

_MAX_TICKS = 400
"""Ceiling on a scenario's ticks, so a workload that stops moving ends the test instead of hanging."""

_SDXL_STREAK_FREE_FLOOR_MB = 4096.0
"""Device free (MB) a 16 GB card must stay above while serving an SDXL streak from a retained copy.

One retained SDXL checkpoint and one sampling window is what the card is asked to carry; a floor a quarter of
the card above zero is breached only by a second copy of something, which is what the scenario forbids."""

_SDXL_STREAK_DUTY_FLOOR = 0.20
"""Fraction of the run's slot-time the streak must spend sampling.

The positive half of the memory verdicts: a worker that never craters because it never dispatches would pass
every floor above. Two sampling slots are configured but an SDXL megapixel job's peak lets only one of them
run at a time on this card, so half is the ceiling and this is a comfortable fraction of it."""


async def _drive_streak(
    world: _DispatchWorld,
    model: _ModelClass,
    *,
    job_count: int = _STREAK_JOBS,
) -> list[ImageGenerateJobPopResponse]:
    """Feed ``world`` a same-model streak, keeping its queue full, and run until the work settles.

    Modelled on continuing demand rather than a fixed queue: the worker holds up to its queue depth and a
    successor is popped as soon as one drains, so a retainer is still finishing its own job when the job that
    would reuse its weights becomes available. That overlap is the condition retention exists for and the one
    the tail-overlap diagnosis is about.
    """
    jobs: list[ImageGenerateJobPopResponse] = []
    settling = 0
    for _ in range(_MAX_TICKS):
        while len(world.job_tracker.jobs_pending_inference) < _QUEUE_DEPTH and len(jobs) < job_count:
            job = make_job_pop_response(model.name, width=1024, height=1024, ddim_steps=30)
            await world.pop(job)
            jobs.append(job)
        await world.step()
        if world.completed_jobs >= job_count:
            settling += 1
            if settling >= _SETTLE_TICKS:
                break
    return jobs


def _streak_world(*, card: _CardClass = _CARD_16GB, legacy_comfy_vram_unload: bool = False) -> _DispatchWorld:
    """Build the two-slot closed-loop world every scenario in this module runs on."""
    return _DispatchWorld(
        card=card,
        lane_count=2,
        max_threads=2,
        queue_depth=_QUEUE_DEPTH,
        whole_card_enabled=True,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
        legacy_comfy_vram_unload=legacy_comfy_vram_unload,
    )


def _assert_streak_drained(world: _DispatchWorld, jobs: list[ImageGenerateJobPopResponse], *, context: str) -> None:
    """Assert every job in the streak reached sampling and the run completed all of them."""
    unsampled = [str(job.id_)[:8] for job in jobs if world.dispatch_tick(job) is None]
    assert not unsampled, f"{context}: {len(unsampled)} streak job(s) never reached sampling {unsampled[:4]}"
    assert world.completed_jobs == len(jobs), (
        f"{context}: {world.completed_jobs} of {len(jobs)} streak jobs completed. {world.state_dump()}"
    )


# --------------------------------------------------------------------------------------------------------
# The retained copy: a same-model streak pays one weight upload, not one per job
# --------------------------------------------------------------------------------------------------------


async def test_a_same_model_streak_is_served_from_one_retained_copy() -> None:
    """A streak of jobs for one model uploads its weights once and samples the rest from the retained copy.

    The failure this encodes: retention decisions reached the dispatch message but no byte of the grant ever
    survived on the card, so every successor in a streak paid a full host-to-device weight upload for a
    checkpoint the device had been holding moments earlier. On a megapixel SDXL job that upload is comparable
    to the sampling it precedes, so the loss is a large fraction of the card's earning time and it grows with
    the length of the streak.

    Read as one positive statement and its physical consequences: the streak uploads once, every job is seated
    on the slot holding those weights, the card never carries a second copy, and it stays above its floor
    while doing so. The duty floor is what stops the memory verdicts from being satisfiable by a worker that
    simply declines to dispatch.
    """
    world = _streak_world()

    jobs = await _drive_streak(world, _SDXL)

    context = "same-model streak"
    _assert_streak_drained(world, jobs, context=context)
    assert world.weight_uploads == 1, (
        f"{context}: the streak paid {world.weight_uploads} weight uploads for {len(jobs)} jobs, so the "
        f"retained copy is not being reused across the job boundary. {world.state_dump()}"
    )
    seated_lanes = {lane for _model, lane in world.dispatch_lanes}
    assert len(seated_lanes) == 1, (
        f"{context}: the streak was seated across lanes {sorted(seated_lanes)}; a successor seated away from "
        f"the retainer funds a second copy of weights the card already holds. {world.state_dump()}"
    )
    assert_no_duplicate_vram_copy(world, _SDXL.name, context=context)
    assert_free_floor(world, _SDXL_STREAK_FREE_FLOOR_MB, context=context)
    assert_governor_never_reached(world, GovernorState.PRESSURE, context=context)
    assert_governor_never_reached(world, GovernorState.SATURATED, context=context)
    assert_ladder_stayed_below(world, FIRST_LANE_TEARDOWN_RUNG, context=context)
    assert_duty_floor(world, _SDXL_STREAK_DUTY_FLOOR, context=context)


async def test_a_defect_reinjection_inert_grant_reuploads_every_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the grant made inert, the same streak re-uploads its weights once per job.

    The reinjection is the pre-actuation regime exactly: the scheduler's retention verdict is forced to a
    denial, so every job ends with the card returned and the next one reloads. It is what makes the scenario
    above a measurement rather than a restatement of whatever the tree happens to do.
    """
    monkeypatch.setattr(
        InferenceScheduler,
        "_should_keep_model_resident",
        lambda self, dispatched_job, *, process_with_model, device_index: False,
    )
    world = _streak_world()

    jobs = await _drive_streak(world, _SDXL)

    _assert_streak_drained(world, jobs, context="inert-grant streak")
    assert world.weight_uploads == len(jobs), (
        "with retention denied the streak must pay one weight upload per job, which is the cost the grant "
        f"exists to remove; it paid {world.weight_uploads} for {len(jobs)} jobs"
    )
    assert world.retained_residents() == {}, "a denied grant must leave no slot recorded as holding weights"
    assert duty_fraction(world) < _SDXL_STREAK_DUTY_FLOOR, (
        f"the re-uploading streak reached {duty_fraction(world):.0%} duty, at or above the "
        f"{_SDXL_STREAK_DUTY_FLOOR:.0%} floor the retained-copy scenario asserts, so that floor would pass "
        "on a tree where retention does nothing"
    )


# --------------------------------------------------------------------------------------------------------
# No second copy: the structural truth behind the tail-overlap loss
# --------------------------------------------------------------------------------------------------------


async def test_b_a_streak_never_stages_a_second_copy_beside_the_running_one() -> None:
    """While a job holds a model on the card, no sibling lane is ever staged with a second copy of it.

    This pins a diagnosis, not a policy. The overlap between one job's tail and the next job's start looks
    like a scheduling gap that a staged waiter would close, and that reading sends the fix to the handoff
    layer. It is wrong: a same-model successor is routed to the slot that already holds the model rather than
    staged beside it, and on a card that cannot carry two copies of a megapixel-class checkpoint no staging
    scheme could place one anyway. The per-job re-upload therefore has exactly one remedy, same-slot reuse of
    retained weights, and a change that adds waiters without changing residency cannot move it.

    Asserted over the whole run rather than at the end, because a second copy staged and dropped between two
    end-of-run readbacks would satisfy a final-state check while having done the damage.
    """
    world = _streak_world()
    staged_beside_resident: list[str] = []
    jobs: list[ImageGenerateJobPopResponse] = []
    settling = 0

    for _ in range(_MAX_TICKS):
        while len(world.job_tracker.jobs_pending_inference) < _QUEUE_DEPTH and len(jobs) < _STREAK_JOBS:
            job = make_job_pop_response(_SDXL.name, width=1024, height=1024, ddim_steps=30)
            await world.pop(job)
            jobs.append(job)
        await world.step()
        resident = world.vram_resident_lanes(_SDXL.name)
        staged = world.ram_staged_lanes(_SDXL.name)
        if resident and staged:
            staged_beside_resident.append(f"tick {world.tick}: resident on {resident}, staged on {staged}")
        if world.completed_jobs >= len(jobs) == _STREAK_JOBS:
            settling += 1
            if settling >= _SETTLE_TICKS:
                break

    _assert_streak_drained(world, jobs, context="streak staging")
    assert not staged_beside_resident, (
        "a second copy of the streak's model was staged while another lane held it on the device:\n    "
        + "\n    ".join(staged_beside_resident[:4])
    )


async def test_b_defect_reinjection_inert_grant_still_stages_no_waiter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Denying retention re-introduces the per-job upload but still produces no staged second copy.

    The companion for a structural claim runs the other way from a policy scenario's: rather than showing the
    assertion can fail, it shows the state the diagnosis says is unreachable stays unreachable when the policy
    the loss is often blamed on is removed. A staged waiter appearing here would mean the tail-overlap loss
    really is a staging problem and this diagnosis is wrong.
    """
    monkeypatch.setattr(
        InferenceScheduler,
        "_should_keep_model_resident",
        lambda self, dispatched_job, *, process_with_model, device_index: False,
    )
    world = _streak_world()
    staged_beside_resident: list[str] = []
    jobs: list[ImageGenerateJobPopResponse] = []
    settling = 0

    for _ in range(_MAX_TICKS):
        while len(world.job_tracker.jobs_pending_inference) < _QUEUE_DEPTH and len(jobs) < _STREAK_JOBS:
            job = make_job_pop_response(_SDXL.name, width=1024, height=1024, ddim_steps=30)
            await world.pop(job)
            jobs.append(job)
        await world.step()
        if world.vram_resident_lanes(_SDXL.name) and world.ram_staged_lanes(_SDXL.name):
            staged_beside_resident.append(f"tick {world.tick}")
        if world.completed_jobs >= len(jobs) == _STREAK_JOBS:
            settling += 1
            if settling >= _SETTLE_TICKS:
                break

    assert world.weight_uploads == len(jobs), (
        "precondition: the reinjection must actually restore the per-job upload this diagnosis is about, "
        f"but the streak paid {world.weight_uploads} uploads for {len(jobs)} jobs"
    )
    assert not staged_beside_resident, (
        "with retention denied, a second copy was staged beside the running one at "
        f"{staged_beside_resident[:4]}; the per-job re-upload would then be closeable by staging, and the "
        "diagnosis that only same-slot reuse can remove it is wrong"
    )


# --------------------------------------------------------------------------------------------------------
# The legacy hatch: no grant may be recorded where the child unloads regardless
# --------------------------------------------------------------------------------------------------------


async def test_c_the_legacy_hatch_records_no_retained_resident() -> None:
    """Under the legacy unload regime no grant is made, so no slot is ever recorded as holding weights.

    The escape hatch runs the child's executor in the mode that returns the card at the end of every prompt,
    below anything a grant can suppress. A grant recorded there is a phantom: the parent's retained-resident
    map names weights the device does not hold, and three separate paths then act on it. The static fit
    charges the phantom against the card and denies a grant that would have fitted; the dispatch admission
    gate holds a load waiting for weights that are already gone; and same-model routing seats a successor on
    a slot that holds nothing, paying a full upload while calling it reuse.

    The scenario runs full job cycles rather than a single dispatch, because the record is written at the job
    boundary: a check made before any job completes cannot distinguish a refused grant from an unfinished one.
    """
    world = _streak_world(legacy_comfy_vram_unload=True)
    retention_observed: list[str] = []
    jobs: list[ImageGenerateJobPopResponse] = []
    settling = 0

    for _ in range(_MAX_TICKS):
        while len(world.job_tracker.jobs_pending_inference) < _QUEUE_DEPTH and len(jobs) < _LEGACY_JOBS:
            job = make_job_pop_response(_SDXL.name, width=1024, height=1024, ddim_steps=30)
            await world.pop(job)
            jobs.append(job)
        await world.step()
        if world.retained_residents():
            retention_observed.append(f"tick {world.tick}: {world.retained_residents()}")
        if world.completed_jobs >= len(jobs) == _LEGACY_JOBS:
            settling += 1
            if settling >= _SETTLE_TICKS:
                break

    _assert_streak_drained(world, jobs, context="legacy hatch")
    assert not retention_observed, (
        "a slot was recorded as holding weights across a job under the regime where the child returns the "
        "card at the end of every prompt:\n    " + "\n    ".join(retention_observed[:4])
    )
    assert world.weight_uploads == len(jobs), (
        "precondition: under the legacy regime every job reloads its weights, so a run that uploaded "
        f"{world.weight_uploads} times for {len(jobs)} jobs is not exercising that regime"
    )


async def test_c_defect_reinjection_granting_under_the_hatch_records_a_phantom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the hatch no longer refusing grants, the parent records residency the device does not hold.

    Reinjected at the one decision the fix added: the retention verdict is evaluated with the hatch hidden
    from it and every other input untouched, so what fails is the refusal itself rather than the arithmetic
    around it. The observable is the phantom in full: a retained-resident record standing while the run's own
    upload count proves the weights were reloaded from host RAM every time.
    """
    original = InferenceScheduler._should_keep_model_resident

    def granting_under_the_hatch(
        self: InferenceScheduler,
        dispatched_job: ImageGenerateJobPopResponse,
        *,
        process_with_model: object,
        device_index: int | None,
    ) -> bool:
        bridge_data = self._runtime_config.bridge_data
        configured = bridge_data.legacy_comfy_vram_unload
        bridge_data.legacy_comfy_vram_unload = False
        try:
            return original(self, dispatched_job, process_with_model=process_with_model, device_index=device_index)
        finally:
            bridge_data.legacy_comfy_vram_unload = configured

    monkeypatch.setattr(InferenceScheduler, "_should_keep_model_resident", granting_under_the_hatch)
    world = _streak_world(legacy_comfy_vram_unload=True)
    retention_observed: list[str] = []
    jobs: list[ImageGenerateJobPopResponse] = []
    settling = 0

    for _ in range(_MAX_TICKS):
        while len(world.job_tracker.jobs_pending_inference) < _QUEUE_DEPTH and len(jobs) < _LEGACY_JOBS:
            job = make_job_pop_response(_SDXL.name, width=1024, height=1024, ddim_steps=30)
            await world.pop(job)
            jobs.append(job)
        await world.step()
        if world.retained_residents():
            retention_observed.append(f"tick {world.tick}: {world.retained_residents()}")
        if world.completed_jobs >= len(jobs) == _LEGACY_JOBS:
            settling += 1
            if settling >= _SETTLE_TICKS:
                break

    assert retention_observed, (
        "granting under the hatch must produce the phantom the refusal exists to prevent, but no slot was "
        "ever recorded as holding weights"
    )
    assert world.weight_uploads == len(jobs), (
        "the phantom is only a phantom while the child really unloads: the run must still pay one upload per "
        f"job, and it paid {world.weight_uploads} for {len(jobs)}"
    )
