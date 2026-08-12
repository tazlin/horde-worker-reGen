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
- **A child that computes its shortfall against its own view craters the card.** A child sees only its own
  allocations, and the process-local free reading runs ahead of the device, so the shortfall its executor
  computes before a load or a sampling window comes out too small: it gives nothing back, allocates anyway, and
  the card's real free figure collapses to the paging cliff. Retention is what makes that reachable, because it
  leaves a footprint standing across job boundaries that the child's own arithmetic is then the last defense
  over. The remedy is a dispatch that carries the parent's device-level reading, which the child clamps its view
  with. That is the ``device truth`` scenario, and its reinjection is a worker whose dispatches carry no reading.
- **Strict queue order evicts the copy the queue was about to reuse.** Retention holds weights on a slot
  between jobs, but placement order is the queue's own: on a model rotation wider than the lane pool the cold
  head's preload target is whichever slot is free, which is routinely a slot retaining another queued model's
  weights. The head loads over them, the job that would have reused them re-uploads onto the other slot, and
  the two models trade places for the rest of the run at a full upload per job. That is the ``rotation order``
  scenario, and its reinjection is the strict order itself. The remedy is a reorder that only ever takes a free
  win, so the scenario runs on a pool with a spare lane: where the head's own load has nowhere to go but the
  retaining lane, the head keeps strict priority and the queue is placed in its own order.
- **A reclaim rung graded on a clock the hardware cannot meet.** Every rung is an asynchronous actuation: an
  unload is an IPC the child services between allocations and the driver then returns the block over the
  following seconds. Graded on a fixed number of governor samples, a working multi-gigabyte release is called a
  failure and the engine escalates through the lane pauses to moving safety off the card, all inside the window
  the release was going to land in. That is the ``slow release`` scenario. Its consequence over a session is the
  ``safety thrash`` scenario: pressure recurs, every episode sprints to the deepest rung, and the safety process
  is ended and rebuilt every couple of minutes, stalling submits each time for nothing the previous cycle did
  not already fail to fix.
- **A record of weights nothing holds hides a model from the preload pass.** The parent keeps two records of
  residency: the slot's own loaded model, and a model map keyed by model. Loading a slot over rewrites the
  first and leaves the second, so the displaced model still reads as loaded. The preload pass counts that map
  in its already-loaded set, so the displaced model's pending job looks served and is never staged again; the
  only dispatchable job left is a line-skip the head-protection hold rightly withholds for that head, and the
  hold then waits on a head no pass will ever load. Both lanes idle with a full queue and a card that is
  almost entirely free. That is the ``displaced residency`` scenario.
- **A selection that outlives the cycle it was derived in.** Dispatch selection may cache the job-and-lane pair
  a cycle chose so the cycle's look-ahead and its dispatch agree, and that pair is valid only while the lane
  still holds the job's model. Applying the children's reports invalidates it, and because every dispatch gate
  below refuses an undispatchable pair without clearing it, selection stays pinned to a job no lane can serve
  while a free lane holds preloaded weights for the head. That is the ``cycle-scoped selection`` scenario, and
  its reinjection is the cycle boundary itself: it pins the scope, which is the only thing keeping the stalled
  pair unreachable, so the harness may never drive a cycle without opening one.
"""

from __future__ import annotations

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeInferenceControlMessage
from horde_worker_regen.process_management.resources import reclaim_ladder as reclaim_ladder_module
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from horde_worker_regen.process_management.resources.reclaim_ladder import ReclaimRungKind
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _SAFETY_PLACEMENT_RESTORE_STREAK,
    InferenceScheduler,
    NextJobAndProcess,
)
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import (
    _CARD_16GB,
    _CARD_24GB,
    _CHILD_FREE_MARGIN_MB,
    _SD15,
    _SD15_OTHER,
    _SDXL,
    _SDXL_OTHER,
    _CardClass,
    _DispatchWorld,
    _ModelClass,
)
from tests.process_management.liveness._world_assertions import (
    FIRST_LANE_TEARDOWN_RUNG,
    LANE_TEARDOWN_RUNGS,
    SAFETY_TEARDOWN_RUNG,
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

_STREAK_WEIGHT_UPLOADS = 2
"""Weight uploads a same-model streak is permitted, against one per job for a streak that retains nothing.

Retention is granted on evidence the slot's own trailing dispatches supply, so the first dispatch on a slot
has nothing behind it and is refused; the second supplies the repeat and every dispatch after it is served
from the retained copy. The streak therefore pays for its weights twice rather than once, and the difference
amortizes to nothing over any real streak. What the scenarios below are about is the gap between this and the
sixteen their reinjections produce, not the warmup itself."""

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


def _streak_world(
    *,
    card: _CardClass = _CARD_16GB,
    legacy_comfy_vram_unload: bool = False,
    child_free_view_lie_mb: float = 0.0,
    footprint_undershoot: float = 1.0,
    max_threads: int = 2,
    lane_count: int = 2,
) -> _DispatchWorld:
    """Build the closed-loop world the scenarios in this module run on, two lanes unless asked otherwise.

    ``max_threads`` is the sampling-slot capacity, which the worker's config expresses as one number for the
    concurrency cap and the slot count alike. At 1 the pool still holds two lanes but only one of them may sample
    at a time, which is the operator posture a great many workers run and a regime of its own for anything gated
    on there being headroom right now.

    ``lane_count`` widens the pool past that default. A placement question only has an answer where the queue
    head's load has somewhere to go other than the lane whose weights the reorder would preserve, so the
    scenarios about placement order need a pool with a spare lane in it.
    """
    return _DispatchWorld(
        card=card,
        lane_count=lane_count,
        max_threads=max_threads,
        queue_depth=_QUEUE_DEPTH,
        whole_card_enabled=True,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
        legacy_comfy_vram_unload=legacy_comfy_vram_unload,
        child_free_view_lie_mb=child_free_view_lie_mb,
        footprint_undershoot=footprint_undershoot,
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

    Read as one positive statement and its physical consequences: the streak uploads a bounded number of times
    rather than once per job, every job is seated
    on the slot holding those weights, the card never carries a second copy, and it stays above its floor
    while doing so. The duty floor is what stops the memory verdicts from being satisfiable by a worker that
    simply declines to dispatch.
    """
    world = _streak_world()

    jobs = await _drive_streak(world, _SDXL)

    context = "same-model streak"
    _assert_streak_drained(world, jobs, context=context)
    assert world.weight_uploads == _STREAK_WEIGHT_UPLOADS, (
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


# --------------------------------------------------------------------------------------------------------
# Device truth: the child's shortfall is computed against the card, not against its own view
# --------------------------------------------------------------------------------------------------------

_CHILD_VIEW_LIE_MB = 6144.0
"""How far a child's own free-VRAM view runs ahead of the card in these scenarios.

Six gigabytes is the order the process-local reading overstates a card whose driver has not returned freed
memory: enough that every shortfall a job's charges could raise reads as comfortably covered, which is what a
child computing against its own view concludes."""

_FOOTPRINT_UNDERSHOOT = 1.8
"""How much more of the card a job really wants than the scheduler's forecast priced it at.

Chosen so the parent's own defenses are all satisfied and none of them is the thing under test: the forecast
fits the job on the card with room, admission and residency accounting agree, and the load then asks for
almost the whole device. Below roughly 1.75 the true footprint still fits with the child's margin intact and
no shortfall arises anywhere, so the run would say nothing about who computes one."""

_TRUTHFUL_FREE_FLOOR_MB = _CHILD_FREE_MARGIN_MB - 1.0
"""Device free (MB) the card holds above while the child's shortfall is computed against device truth.

The child frees for its allocation plus its own margin, so the margin is exactly what a truthful shortfall
leaves standing; a megabyte of slack keeps the floor a statement about the mechanism rather than about float
arithmetic. A card below it was relieved by nobody."""

_TRUTHFUL_DUTY_FLOOR = 0.20
"""Fraction of the run's slot-time the streak must still spend sampling while fitting inside the card.

Relieving a shortfall out of the running job's own weights costs sampling speed, and refusing to dispatch
costs everything, so the floor is what stops either from being a way to satisfy the memory verdicts. The same
floor the retained-copy scenario states, since this is the same streak on the same card."""


def _blind_children_to_device_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the parent's device reading off every inference dispatch this run sends.

    Reinjects the regime the fix replaced, at the field itself: the scheduler still measures, decides and
    dispatches exactly as it does now, and the only difference is that the child is told nothing about the
    card and is left computing its shortfall against its own view.
    """
    original_init = HordeInferenceControlMessage.__init__

    def without_device_truth(self: HordeInferenceControlMessage, **data: object) -> None:
        data["device_free_mb"] = None
        original_init(self, **data)

    monkeypatch.setattr(HordeInferenceControlMessage, "__init__", without_device_truth)


async def test_d_a_retention_streak_fits_the_card_on_device_truth() -> None:
    """A streak whose true footprint nearly fills the card stays off the paging cliff and keeps earning.

    The failure this encodes: retention leaves a footprint standing across job boundaries, and the last thing
    between that footprint and the device is the child's own shortfall arithmetic. Computed against the
    process-local view, which runs ahead of the card, the shortfall comes out too small; the child gives
    nothing back, allocates anyway, and free VRAM collapses to the paging cliff, where the governor saturates
    and the worker starts relieving the card by taking its own capacity down.

    The card here is asked for more than the forecast priced, so a shortfall genuinely arises and somebody has
    to answer it. Read as one positive statement and its consequences: the dispatch carries the parent's
    device reading, the child's relief comes out of its own footprint, the card keeps its margin, the governor
    never saturates, no lane is taken off the air, and the streak still drains on its warmup upload alone.
    """
    world = _streak_world(
        child_free_view_lie_mb=_CHILD_VIEW_LIE_MB,
        footprint_undershoot=_FOOTPRINT_UNDERSHOOT,
    )

    jobs = await _drive_streak(world, _SDXL)

    context = "device-truth streak"
    _assert_streak_drained(world, jobs, context=context)
    assert world.child_shortfall_frees, (
        f"{context}: no charge in the run ever raised a shortfall, so nothing here depends on who computes "
        f"one. {world.state_dump()}"
    )
    assert world.child_overcommits == [], (
        f"{context}: the card was committed past what it held at "
        f"{world.child_overcommits[:2]}, so a charge went on regardless of the shortfall it raised. "
        f"{world.state_dump()}"
    )
    assert world.weight_uploads == _STREAK_WEIGHT_UPLOADS, (
        f"{context}: the streak paid {world.weight_uploads} weight uploads for {len(jobs)} jobs, so fitting "
        f"the card cost the retained copy the streak is served from. {world.state_dump()}"
    )
    assert_free_floor(world, _TRUTHFUL_FREE_FLOOR_MB, context=context)
    assert_governor_never_reached(world, GovernorState.SATURATED, context=context)
    assert_ladder_stayed_below(world, FIRST_LANE_TEARDOWN_RUNG, context=context)
    assert_duty_floor(world, _TRUTHFUL_DUTY_FLOOR, context=context)


async def test_d_defect_reinjection_a_child_blind_to_the_card_craters_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the dispatch carrying no device reading, the same streak drives the card onto the paging cliff.

    The reinjection is the pre-fix regime exactly: every decision the parent makes is unchanged and the child
    is simply not told what the card holds. Its shortfall then reads as covered, it relieves nothing, and the
    card's free figure and the governor's verdict both go where the incident took them.
    """
    _blind_children_to_device_truth(monkeypatch)
    world = _streak_world(
        child_free_view_lie_mb=_CHILD_VIEW_LIE_MB,
        footprint_undershoot=_FOOTPRINT_UNDERSHOOT,
    )

    jobs = await _drive_streak(world, _SDXL)

    assert world.child_overcommits, (
        "a child told nothing about the card must commit charges past what the card holds, which is the whole "
        f"of this defect; none was recorded. {world.state_dump()}"
    )
    assert world.child_shortfall_frees == [], (
        "a child computing its shortfall against an inflated view finds nothing to relieve, but it relieved "
        f"{world.child_shortfall_frees[:2]}"
    )
    assert world.min_device_free_mb < _TRUTHFUL_FREE_FLOOR_MB, (
        f"device free bottomed out at {world.min_device_free_mb:.0f}MB, still above the "
        f"{_TRUTHFUL_FREE_FLOOR_MB:.0f}MB floor the device-truth scenario asserts, so that floor would pass "
        f"on a tree whose dispatches carry no device reading. {world.state_dump()}"
    )
    assert GovernorState.SATURATED in world.governor_states, (
        "the card never reached governor saturated, so the scenario's governor verdict would pass unfixed "
        f"({world.completed_jobs} of {len(jobs)} jobs, low water {world.min_device_free_mb:.0f}MB)"
    )


async def test_d_a_truthful_child_view_needs_no_device_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    """A child whose own view matches the card fits the same streak with no reading on the dispatch at all.

    The structural half of the diagnosis, and the reason the remedy is a reading rather than a policy: the
    same workload, the same footprint, the same absent device figure, and the only thing removed is the gap
    between what the child believes and what the card holds. It fits. The overcommit is therefore caused by
    the view being wrong, not by the load being too large for the card, and the fix belongs where the view is
    corrected rather than in anything that would decline the work.
    """
    _blind_children_to_device_truth(monkeypatch)
    world = _streak_world(child_free_view_lie_mb=0.0, footprint_undershoot=_FOOTPRINT_UNDERSHOOT)

    jobs = await _drive_streak(world, _SDXL)

    context = "truthful child view"
    _assert_streak_drained(world, jobs, context=context)
    assert world.child_overcommits == [], (
        f"{context}: a child whose view matches the card committed past it at {world.child_overcommits[:2]}, "
        f"so the crater does not depend on the view being wrong and this diagnosis is incomplete. "
        f"{world.state_dump()}"
    )
    assert_free_floor(world, _TRUTHFUL_FREE_FLOOR_MB, context=context)
    assert_governor_never_reached(world, GovernorState.SATURATED, context=context)


# --------------------------------------------------------------------------------------------------------
# Rotation order: a retained copy is served before a cold head is loaded over it
# --------------------------------------------------------------------------------------------------------

_ROTATION = (_SDXL, _SDXL_OTHER, _SD15, _SD15_OTHER)
"""The checkpoints a rotation cycles through: more models than the pool has lanes.

The live shape this is about is a handful of checkpoints rotating over a pool that cannot hold them all, which
is what makes a cold head's preload target a slot that is retaining a model the queue still wants. A rotation no
wider than the pool converges on a lane per model whatever the placement order is, so it cannot discriminate one
order from another; this is the narrowest rotation that can."""

_ROTATION_LANES = 4
"""Lanes the rotation runs over, wider than the two-lane default the rest of the module uses.

The reorder is admitted only where it costs the head nothing: the head must need a load and that load must have
somewhere to go other than the lane whose weights the reorder would keep. Two lanes with one of them retaining
cannot offer that second target, so the placement order there is the queue's own by construction and a scenario
run on it would measure nothing about the reorder. Sampling capacity stays at two, so the extra lanes are
staging room rather than concurrency."""

_ROTATION_JOBS = 16
"""Jobs in the rotation: four per checkpoint, so a model's residency has successors to be reused by."""

_ROTATION_SHAPE = (512, 512)
"""Generation size the rotation's jobs ask for.

Small enough that two checkpoints and a sampling window co-reside on a 16 GB card, which is the precondition
for the loss being an ordering one at all: where the card can hold only one of them, every cross-model
placement pays an upload no order can avoid."""

_ROTATION_UPLOAD_CEILING = 13
"""Weight uploads the rotation must come in under.

The reachable figure is one upload per residency episode rather than per job, and with four checkpoints over a
pool that holds a couple of them at a time each model takes several episodes to serve its four jobs. The
reordering run comes in at twelve of the sixteen the strict order pays, so the ceiling sits between the two:
below the one-per-job cost and above what the reordering run needs, stating the difference rather than pinning
a schedule."""

_ROTATION_DUTY_FLOOR = 0.06
"""Fraction of the run's slot-time the rotation must spend sampling.

The positive half of the upload verdict: an order that avoided uploads by declining to dispatch would pass a
ceiling on uploads and fail this. A rotation spends much of its time loading whatever the order, so the floor
is far below a same-model streak's; it is set under what the reordering run reaches and above what the strict
order does, so it discriminates as well as guards."""


async def _drive_rotation(
    world: _DispatchWorld,
    models: tuple[_ModelClass, ...] = _ROTATION,
    *,
    job_count: int = _ROTATION_JOBS,
) -> list[ImageGenerateJobPopResponse]:
    """Feed ``world`` a rotation through ``models``, keeping its queue full, and run until the work settles.

    The queue is refilled in strict rotation, so the pending window always holds a job for a model some slot is
    retaining and a job for a model no slot holds. That is the choice the placement order is about, and holding
    the queue at depth is what keeps it present on every cycle rather than only at the start.
    """
    jobs: list[ImageGenerateJobPopResponse] = []
    settling = 0
    width, height = _ROTATION_SHAPE
    for _ in range(_MAX_TICKS):
        while len(world.job_tracker.jobs_pending_inference) < _QUEUE_DEPTH and len(jobs) < job_count:
            model = models[len(jobs) % len(models)]
            job = make_job_pop_response(model.name, width=width, height=height, ddim_steps=30)
            await world.pop(job)
            jobs.append(job)
        await world.step()
        if world.completed_jobs >= job_count:
            settling += 1
            if settling >= _SETTLE_TICKS:
                break
    return jobs


async def test_e_a_rotation_serves_retained_copies_before_loading_a_cold_head_over_them() -> None:
    """A rotation wider than the lane pool serves what slots retain, where doing so costs the head nothing.

    The failure this encodes: retention held weights across the job boundary exactly as designed, and the
    placement order threw them away. The cold head is preloaded first and its target is whichever slot is free,
    so it lands on a slot retaining a model still in the queue; that model's next job then re-uploads onto the
    other slot, whose retained copy the following head evicts in turn. Two models trade lanes for the rest of
    the run and every job pays a full host-to-device upload for weights that were on the card moments earlier.

    Read as one positive statement and its consequences: where the head needs a load anyway and that load has a
    lane other than the retainer to go to, the job the retainer can already serve is seated first and the head's
    load runs alongside its sampling, so the run's uploads fall below one per job, the placement order engages
    rather than being incidental, every job still drains (the reordering is bounded, so a cold head is never
    traded away), and the card carries the co-residency that makes the reuse possible without saturating.
    """
    world = _streak_world(lane_count=_ROTATION_LANES)

    jobs = await _drive_rotation(world)

    context = "rotation order"
    _assert_streak_drained(world, jobs, context=context)
    assert world.weight_uploads <= _ROTATION_UPLOAD_CEILING, (
        f"{context}: the rotation paid {world.weight_uploads} weight uploads for {len(jobs)} jobs, at or near "
        "the one-per-job cost of loading each cold head over whichever slot was free. "
        f"{world.state_dump()}"
    )
    assert world.scheduler.retention_affinity_reorders > 0, (
        f"{context}: no job was ever seated ahead of the head by the placement order, so the upload count above "
        f"says nothing about it. {world.state_dump()}"
    )
    assert_governor_never_reached(world, GovernorState.SATURATED, context=context)
    assert_ladder_stayed_below(world, FIRST_LANE_TEARDOWN_RUNG, context=context)
    assert_duty_floor(world, _ROTATION_DUTY_FLOOR, context=context)


async def test_e_defect_reinjection_strict_queue_order_reuploads_every_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the placement order back to the queue's own, the same rotation uploads once per job.

    Reinjected at the source the placement order reads: the scan is told no pending job is served by a retained
    copy, so nothing is ever promoted ahead of the head, and everything else (retention grants, admission,
    routing, the card) is untouched. What comes back is the strict order's schedule in full, a lane per queue
    position and an upload per job, which is what makes the ceiling above a measurement of the order rather than
    of the rotation.
    """
    monkeypatch.setattr(
        InferenceScheduler,
        "_retention_affinity_candidates",
        lambda self, head_job: [],
    )
    world = _streak_world(lane_count=_ROTATION_LANES)

    jobs = await _drive_rotation(world)

    _assert_streak_drained(world, jobs, context="strict-order rotation")
    assert world.weight_uploads == len(jobs), (
        "under the strict queue order every job in the rotation must pay its own weight upload, which is the "
        f"cost the placement order exists to remove; it paid {world.weight_uploads} for {len(jobs)} jobs"
    )
    assert world.weight_uploads > _ROTATION_UPLOAD_CEILING, (
        f"the strict order came in at {world.weight_uploads} uploads, under the "
        f"{_ROTATION_UPLOAD_CEILING} ceiling the rotation scenario asserts, so that ceiling would pass on a "
        "tree whose placement order never seats a retained copy's job ahead of a cold head"
    )
    assert duty_fraction(world) < _ROTATION_DUTY_FLOOR, (
        f"the strict order reached {duty_fraction(world):.1%} duty, at or above the "
        f"{_ROTATION_DUTY_FLOOR:.1%} floor the rotation scenario asserts, so that floor would pass unfixed"
    )


async def test_e_with_nothing_retained_the_placement_order_is_the_queues_own() -> None:
    """Where no slot ever retains weights, the rotation is placed in queue order and nothing is reordered.

    The degradation guarantee the reordering has to carry: it is keyed to a retained copy a job can be served
    from at no upload, so a worker holding none of those (a cold start, the legacy unload regime, traffic with
    no repeat within the queue window) must schedule exactly as it did before. The legacy hatch is the
    strongest form of that, since the child returns the card at the end of every prompt whatever was asked, and
    it is checked by the reorder count rather than by the schedule alone: a run that reordered and happened to
    land on the same lanes would otherwise read as unchanged.
    """
    world = _streak_world(legacy_comfy_vram_unload=True)

    jobs = await _drive_rotation(world)

    context = "nothing retained"
    _assert_streak_drained(world, jobs, context=context)
    assert world.retained_residents() == {}, (
        f"{context}: a slot was recorded as retaining weights under the regime where the child returns the "
        f"card at the end of every prompt, so this run is not the no-retention case. {world.state_dump()}"
    )
    assert world.scheduler.retention_affinity_reorders == 0, (
        f"{context}: the placement order seated {world.scheduler.retention_affinity_reorders} job(s) ahead of a "
        f"head with nothing on the card to seat them onto. {world.state_dump()}"
    )
    assert world.weight_uploads == len(jobs), (
        f"{context}: with no retained copy to reuse every job must upload its own weights, and the run paid "
        f"{world.weight_uploads} for {len(jobs)} jobs"
    )


_COLD_HEAD_TICK_CEILING = 30
"""Ticks a cold head may spend waiting behind an unbroken stream of retained work before it samples.

The wait is bounded by the affinity window (tens of seconds) and by the skip ceiling, and this world's tick is
two seconds, so a head still waiting after this many ticks is bounded by neither: it is being traded away for
the retained work indefinitely, which is the one failure mode a placement preference must be incapable of."""


@pytest.mark.parametrize("max_threads", [1, 2], ids=["serial", "concurrent"])
async def test_e_a_cold_head_still_samples_behind_an_unbroken_stream_of_retained_work(max_threads: int) -> None:
    """A head whose model no slot holds is served even while every other pending job could reuse a retained copy.

    The hostile case for any placement preference: a slot is already retaining the stream's model when the cold
    head arrives, and the queue never stops offering more of that stream, so a preference with no bound would
    keep choosing it and the head would age out its ttl having never run. What stops that is that the placement
    order asks the same budget the resident-model bypass asks, so the head's window and its skip ceiling are
    spent by a reorder exactly as they are by a line-skip, and once spent no candidate is named at all: the
    retainer becomes an ordinary target again and the head's own model loads.

    Run at both sampling capacities because the head reaches its dispatch by different routes: at capacity 2 it
    can be passed by a dispatch onto a free lane, at capacity 1 nothing may be seated ahead of it at all.
    """
    world = _streak_world(max_threads=max_threads)
    for _ in range(_MAX_TICKS):
        while len(world.job_tracker.jobs_pending_inference) < _QUEUE_DEPTH:
            await world.pop(make_job_pop_response(_SDXL.name, width=512, height=512, ddim_steps=30))
        await world.step()
        if world.retained_residents():
            break
    assert world.retained_residents(), (
        f"precondition: the stream's model must be retained on a slot before the cold head arrives, or there is "
        f"nothing for the placement order to prefer over it. {world.state_dump()}"
    )

    cold_head = make_job_pop_response(_SD15.name, width=512, height=512, ddim_steps=30)
    await world.pop(cold_head)
    popped_at = world.tick

    for _ in range(_MAX_TICKS):
        while len(world.job_tracker.jobs_pending_inference) < _QUEUE_DEPTH:
            await world.pop(make_job_pop_response(_SDXL.name, width=512, height=512, ddim_steps=30))
        await world.step()
        if world.dispatch_tick(cold_head) is not None:
            break

    head_tick = world.dispatch_tick(cold_head)
    assert head_tick is not None, (
        "the cold head never sampled while retained work kept arriving, so the placement preference has no "
        f"bound. {world.state_dump()}"
    )
    waited = head_tick - popped_at
    assert waited <= _COLD_HEAD_TICK_CEILING, (
        f"the cold head waited {waited} ticks behind the retained stream, past the "
        f"{_COLD_HEAD_TICK_CEILING}-tick ceiling its bounded wait allows. {world.state_dump()}"
    )


# --------------------------------------------------------------------------------------------------------
# Cycle-scoped selection: a rotation drains without help from the placement preference
# --------------------------------------------------------------------------------------------------------

_ROTATION_3 = (_SDXL, _SDXL_OTHER, _SD15)
"""A rotation one model wider than the lane pool, so a lane turns over every cycle of it.

Two models over two lanes converge on a lane each and nothing is ever displaced; four models (the rotation
above) displace so often that no queue position outlives a lane's residency. Three is the shape where a job's
model is resident when it is looked at and gone a tick later, which is what a selection cached across the
boundary that applies the child's report is wrong about."""

_ROTATION_3_JOBS = 18
"""Jobs in the rotation: six per checkpoint, so the rotation cycles several times over turned-over lanes."""

_ROTATION_3_TICK_CEILING = 120
"""Ticks the rotation may take to drain.

A rotation that keeps dispatching finishes in well under half this; a run that stops dispatching altogether
consumes the module's whole tick ceiling. The gap between the two is wide enough that this discriminates a
drain from a wedge without pinning a schedule."""

_ROTATION_3_DUTY_FLOOR = 0.05
"""Fraction of the run's slot-time the rotation must spend sampling.

The positive half of the drain verdict: a run that completed its jobs slowly enough would satisfy a count and
starve the card. This sits under what a rotation reaches while placing every job by the queue's own order and
far above what a stalled selection leaves."""


async def test_f_a_rotation_drains_when_no_placement_preference_reorders_it() -> None:
    """A rotation over turned-over lanes keeps dispatching with the placement preference taken out.

    The failure this encodes: selection may cache the job-and-lane pair it chose, so a cycle's look-ahead and
    its dispatch cannot disagree, and the cached pair's validity rests on the lane still holding that job's
    model. Applying the children's reports between cycles is what invalidates it: the lane gives its weights
    back, the pair becomes undispatchable, and while it is held every gate below it refuses the dispatch, so
    nothing clears the cache. Selection is then pinned to a job no lane can serve and never offers the head
    that a free lane is holding preloaded weights for. Both lanes idle with a full queue for the rest of the
    run.

    Read as one positive statement: dispatch selection is re-derived from the process state each cycle, so a
    rotation whose lanes turn over under it keeps finding the pending job some lane can serve, and the run
    drains while keeping the card fed. The companion below asserts the same drain with the placement preference
    neutered, because that preference is an upload optimisation and the queue must drain whether or not it
    engages: a liveness guarantee resting on a preference is one a configuration change can delete.
    """
    world = _streak_world()

    jobs = await _drive_rotation(world, _ROTATION_3, job_count=_ROTATION_3_JOBS)

    context = "three-model rotation"
    _assert_streak_drained(world, jobs, context=context)
    assert world.tick <= _ROTATION_3_TICK_CEILING, (
        f"{context}: the rotation took {world.tick} ticks to drain, past the {_ROTATION_3_TICK_CEILING}-tick "
        f"ceiling a run that keeps dispatching comes in under. {world.state_dump()}"
    )
    assert_duty_floor(world, _ROTATION_3_DUTY_FLOOR, context=context)
    assert_governor_never_reached(world, GovernorState.SATURATED, context=context)
    assert_ladder_stayed_below(world, FIRST_LANE_TEARDOWN_RUNG, context=context)


async def test_f_a_rotation_drains_without_the_placement_preference(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same rotation drains with the placement preference told there is nothing to yield to.

    The preference reorders a cold head behind a job a lane retains the weights for, and that reordering is
    enough on its own to keep this rotation off the pair the stalled selection is reachable through. Neutering
    it is what makes the drain above a statement about selection rather than about which order happened to be
    chosen.
    """
    monkeypatch.setattr(
        InferenceScheduler,
        "_retention_affinity_candidates",
        lambda self, head_job: [],
    )
    world = _streak_world()

    jobs = await _drive_rotation(world, _ROTATION_3, job_count=_ROTATION_3_JOBS)

    context = "unreordered three-model rotation"
    _assert_streak_drained(world, jobs, context=context)
    assert world.scheduler.retention_affinity_reorders == 0, (
        f"{context}: the placement order reordered {world.scheduler.retention_affinity_reorders} time(s) though "
        f"the preference was neutered, so this run is not the unreordered case. {world.state_dump()}"
    )
    assert world.tick <= _ROTATION_3_TICK_CEILING, (
        f"{context}: the rotation took {world.tick} ticks to drain, past the {_ROTATION_3_TICK_CEILING}-tick "
        f"ceiling a run that keeps dispatching comes in under. {world.state_dump()}"
    )
    assert_duty_floor(world, _ROTATION_3_DUTY_FLOOR, context=context)


async def test_f_defect_reinjection_selection_outliving_its_cycle_wedges_the_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let the cached selection outlive its premises and the same rotation stops keeping the card fed.

    Two guards keep a cached line-skip pair honest: the cycle boundary discards it before the tick that
    applies the children's reports, and the premise-level revalidation refuses a pair whose target no longer
    holds the job's model. Reinjected at both, with the placement preference neutered alongside so nothing
    reorders the rotation off the pair, the rotation spends its run pinned to a job no lane can serve: it
    misses the tick ceiling several times over and its duty falls under the floor. Each guard alone prevents
    that, so the ceiling and floor above are a measurement of the pair of them.

    The jobs do eventually drain, because the stalled pair is broken by the starvation backstops rather than
    by selection re-deriving itself. That is why what this asserts is the scenario's own drain verdict (a
    bounded schedule at a duty the card earns from) rather than a queue that never moves: a wedge a backstop
    escapes from after minutes is still the failure this scenario exists to keep out.
    """
    monkeypatch.setattr(
        InferenceScheduler,
        "_retention_affinity_candidates",
        lambda self, head_job: [],
    )
    monkeypatch.setattr(InferenceScheduler, "begin_scheduling_cycle", lambda self: None)

    def _residency_blind_validation(self: InferenceScheduler, cached: NextJobAndProcess) -> bool:
        cached_job = cached.next_job
        return (
            cached_job in self._job_tracker.jobs_pending_inference
            and cached_job not in self._job_tracker.jobs_in_progress
            and cached.process_with_model.can_accept_job()
        )

    monkeypatch.setattr(InferenceScheduler, "_line_skip_cache_valid", _residency_blind_validation)
    world = _streak_world()

    await _drive_rotation(world, _ROTATION_3, job_count=_ROTATION_3_JOBS)

    assert world.tick > _ROTATION_3_TICK_CEILING, (
        f"a selection allowed to outlive its cycle drained the rotation in {world.tick} ticks, inside the "
        f"{_ROTATION_3_TICK_CEILING}-tick ceiling the scenario asserts, so that ceiling would pass on a tree "
        f"that never re-derives it. {world.state_dump()}"
    )
    assert duty_fraction(world) < _ROTATION_3_DUTY_FLOOR, (
        f"the stalled rotation reached {duty_fraction(world):.1%} duty, at or above the "
        f"{_ROTATION_3_DUTY_FLOOR:.1%} floor the rotation scenario asserts, so that floor would pass unfixed"
    )


# --------------------------------------------------------------------------------------------------------
# The slow release: a rung that is working is not graded a failure for landing after the next sample
# --------------------------------------------------------------------------------------------------------

_UNLOAD_RELEASE_SECONDS = 5.0
"""Seconds between the parent sending an unload and the card getting the memory back in these scenarios.

An unload is an IPC the child services between its own allocations, after which the driver returns the block,
so a checkpoint-sized give-back lands seconds after the command rather than as the actuator returns. Five
seconds is the low end of what such a release costs and is already several governor samples at this module's
tick, which is the whole of what a sample-counted verification window gets wrong."""

_PRESSURE_JOBS = 24
"""Jobs driven through the slow-release scenarios: enough saturation episodes that a per-episode claim has a
population to be true of, rather than describing one dip."""

_PRESSURE_TICKS = 300
"""Ticks the slow-release scenarios run for, which at this module's tick is ten simulated minutes."""

_THRASH_JOBS = 48
"""Jobs driven through the thrash scenario: twice the slow-release load, so recurring pressure spans the whole
run rather than a burst at the start."""

_THRASH_TICKS = 900
"""Ticks the thrash scenario runs for: thirty simulated minutes, several times the safety rung's own dwell.

The signature being forbidden is a safety process ended and rebuilt every couple of minutes, so the run has to
be long enough that a worker doing that shows several cycles and a worker holding its dwell does not."""

_SHORT_BUDGET_BASE_SECONDS = 2.0 * _TICK_SECONDS
"""The in-process verification allowance a sample-counted window amounts to, for the defect reinjections."""

_SHORT_BUDGET_TEARDOWN_SECONDS = 3.0 * _TICK_SECONDS
"""The teardown verification allowance a sample-counted window amounts to, for the defect reinjections."""


def _reinject_sample_counted_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the verification window back to what a fixed couple of governor samples buys.

    The pre-fix grading exactly: a rung is given a flat couple of samples' worth of time whatever it promised
    and whatever the hardware needs, so a working release is judged to have freed nothing.
    """
    monkeypatch.setattr(reclaim_ladder_module, "_VERIFICATION_BASE_SECONDS", _SHORT_BUDGET_BASE_SECONDS)
    monkeypatch.setattr(reclaim_ladder_module, "_VERIFICATION_SECONDS_PER_GB", 0.0)
    monkeypatch.setattr(
        reclaim_ladder_module,
        "_TEARDOWN_VERIFICATION_BASE_SECONDS",
        _SHORT_BUDGET_TEARDOWN_SECONDS,
    )
    monkeypatch.setattr(reclaim_ladder_module, "_TEARDOWN_VERIFICATION_SECONDS_PER_GB", 0.0)


def _pressure_world(
    *,
    card: _CardClass = _CARD_16GB,
    unload_release_seconds: float = _UNLOAD_RELEASE_SECONDS,
) -> _DispatchWorld:
    """Build the closed-loop world the reclaim scenarios run on: a card that recurringly crosses the cliff.

    Safety sits on the card and the operator permits it to be moved off, so the ladder has its full depth and a
    run can say what reaching the deepest rung costs. Sampling is serial, which keeps an idle resident on the
    other lane for the ladder's cheap rungs to name.

    Getting the card over the cliff is the point: reclaim is what a worker does once every gate ahead of it has
    already been beaten, so a scenario about reclaim has to put the card there. The lever is the one the
    device-truth scenarios above characterise, a job that really wants more of the device than the forecast
    priced run by a child computing its shortfall against its own view, and callers arm it with
    :func:`_blind_children_to_device_truth`. What the ladder does about a card in that state is independent of
    how it got there.
    """
    return _DispatchWorld(
        card=card,
        lane_count=2,
        max_threads=1,
        queue_depth=_QUEUE_DEPTH,
        whole_card_enabled=True,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
        service_contexts=True,
        safety_off_gpu_allowed=True,
        child_free_view_lie_mb=_CHILD_VIEW_LIE_MB,
        footprint_undershoot=_FOOTPRINT_UNDERSHOOT,
        unload_release_delay_seconds=unload_release_seconds,
    )


_PRESSURE_WORKLOAD: dict[str, tuple[tuple[_ModelClass, ...], tuple[int, int]]] = {
    _CARD_16GB.label: ((_SDXL, _SD15), (1024, 1024)),
    _CARD_24GB.label: ((_SDXL, _SDXL_OTHER), (1024, 1024)),
}
"""The rotation and generation size each card class is driven with.

Sized per card so every class runs the regime the scenario is about: two checkpoints the card can hold at once
(so a lane retains an idle resident for the ladder to name) and a sampling window large enough to take the card
over its hard floor. One workload cannot do both on cards this far apart, and the sizes are what make the run a
statement about the verification budget rather than about one card's arithmetic."""


async def _drive_pressure(
    world: _DispatchWorld,
    *,
    job_count: int,
    ticks: int,
    models: tuple[_ModelClass, ...] | None = None,
    shape: tuple[int, int] | None = None,
) -> None:
    """Keep ``world``'s queue full with a rotation for ``ticks``, so pressure recurs rather than happens once.

    The rotation is what keeps an idle resident on the lane that is not sampling, so the ladder has a resident
    rung to issue; refilling the queue for the whole run is what makes the card cross the cliff repeatedly.
    """
    configured_models, configured_shape = _PRESSURE_WORKLOAD[world.card.label]
    models = models or configured_models
    width, height = shape or configured_shape
    popped = 0
    for _ in range(ticks):
        while len(world.job_tracker.jobs_pending_inference) < _QUEUE_DEPTH and popped < job_count:
            model = models[popped % len(models)]
            await world.pop(make_job_pop_response(model.name, width=width, height=height, ddim_steps=30))
            popped += 1
        await world.step()


def _saturation_episodes(world: _DispatchWorld) -> list[tuple[int, int]]:
    """Return each maximal run of consecutive SATURATED ticks as an inclusive ``(first, last)`` tick span.

    An episode is the unit the reclaim ladder works in: one crossing of the hard floor, one frozen ladder, one
    escalation sequence. Reading them off the governor series is what lets a scenario state a claim per episode
    rather than over a run's undifferentiated actuation list.
    """
    spans: list[tuple[int, int]] = []
    first: int | None = None
    for tick, state in enumerate(world.governor_states, start=1):
        if state is GovernorState.SATURATED and first is None:
            first = tick
        elif state is not GovernorState.SATURATED and first is not None:
            spans.append((first, tick - 1))
            first = None
    if first is not None:
        spans.append((first, len(world.governor_states)))
    return spans


def _escalations_while_a_release_is_in_flight(world: _DispatchWorld) -> list[str]:
    """Every teardown rung issued while an unload the engine had already ordered was still landing.

    The signature of grading a rung on a clock the hardware cannot meet: the engine gave back a checkpoint, the
    driver had not finished returning it, and the engine had already moved on to stopping a lane or taking
    safety off the card. Each unload is paired with the tick the world booked its release, so this says the
    engine escalated before the release could have completed rather than merely that it escalated: a rung whose
    promise was met by the time it was graded is not a misjudgement whatever came next.
    """
    releases = list(world.unload_releases)
    in_flight: list[tuple[int, int, int]] = []
    for actuation in world.ladder_actuations:
        if actuation.kind is not ReclaimRungKind.UNLOAD_IDLE_MODEL:
            continue
        landed_at = next(
            (tick for tick, lane in releases if lane == actuation.target_process_id and tick > actuation.tick),
            None,
        )
        if landed_at is not None:
            in_flight.append((actuation.tick, landed_at, actuation.target_process_id or -1))
    teardown_kinds = LANE_TEARDOWN_RUNGS | {SAFETY_TEARDOWN_RUNG}
    breaches: list[str] = []
    for actuation in world.ladder_actuations:
        if actuation.kind not in teardown_kinds:
            continue
        for issued_at, landed_at, lane in in_flight:
            if issued_at < actuation.tick < landed_at:
                breaches.append(
                    f"tick {actuation.tick}: {actuation.kind.value} while the unload issued to lane {lane} at "
                    f"tick {issued_at} was still landing (its charge came off the card at tick {landed_at})",
                )
                break
    return breaches


def _assert_slow_release_was_not_misjudged(world: _DispatchWorld, *, context: str) -> None:
    """Assert the run's reclaim gave every rung the time its release needed and still relieved the card."""
    unloads = [a for a in world.ladder_actuations if a.kind is ReclaimRungKind.UNLOAD_IDLE_MODEL]
    assert unloads, (
        f"{context}: the ladder never gave back an idle resident, so this run says nothing about how a slow "
        f"release is graded. {world.state_dump()}"
    )
    assert world.reclaim_ladder.verification_shortfalls == 0, (
        f"{context}: {world.reclaim_ladder.verification_shortfalls} rung(s) were recorded as having freed less "
        f"than they promised on a card where every rung the engine issued did what it was asked. "
        f"{world.state_dump()}"
    )
    breaches = _escalations_while_a_release_is_in_flight(world)
    assert not breaches, (
        f"{context}: the engine escalated past a release it had already ordered and that had not yet had time "
        "to land:\n    " + "\n    ".join(breaches[:4])
    )
    ordered_relief = [
        first for first, last in _saturation_episodes(world) if any(first <= a.tick <= last for a in unloads)
    ]
    assert ordered_relief, (
        f"{context}: no saturation episode contained the unload rung(s) recorded, so the reclaim measured here "
        f"was not the reclaim those episodes ordered. {world.state_dump()}"
    )
    assert world.governor_states[-1] is not GovernorState.SATURATED, (
        f"{context}: the card ended the run still over the cliff, so the reclaim it ordered never relieved it. "
        f"{world.state_dump()}"
    )


@pytest.mark.parametrize("card", [_CARD_16GB, _CARD_24GB], ids=["16gb", "24gb"])
async def test_g_a_slow_release_is_not_graded_a_failed_rung(
    card: _CardClass,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reclaim rung whose memory lands seconds later is verified rather than escalated past, on any card.

    The failure this encodes: no rung frees synchronously. An unload is a message the child services between
    its own allocations, and the driver returns a checkpoint-sized block over the seconds that follow; a lane
    pause returns nothing until the OS has torn the process down. An engine that grades a rung on a fixed
    number of governor samples therefore reads every working release as having freed nothing, records a
    shortfall against the tenant it named, and escalates. On a control loop that samples faster than the driver
    frees, that walks the whole ladder down to taking safety off the card inside the window the first rung's
    memory was going to arrive in, and the card then recovers from the rung that was declared failed.

    Read as one positive statement and its consequences: a rung keeps its verification budget for as long as it
    keeps realizing free, so no rung in the run is graded short, no teardown is issued while a release the
    engine ordered is still in flight, and the episodes that ordered one still end.

    Run across the card classes operators actually have, because the budget a release needs scales with the
    bytes being released: a verdict that only holds on the card it was measured on is a constant in disguise.
    """
    _blind_children_to_device_truth(monkeypatch)
    world = _pressure_world(card=card)

    await _drive_pressure(world, job_count=_PRESSURE_JOBS, ticks=_PRESSURE_TICKS)

    _assert_slow_release_was_not_misjudged(world, context=f"slow release on {card.label}")


@pytest.mark.parametrize("card", [_CARD_16GB, _CARD_24GB], ids=["16gb", "24gb"])
async def test_g_defect_reinjection_a_sample_counted_window_escalates_past_a_working_rung(
    card: _CardClass,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the verification budget cut to what a couple of samples buy, the same run escalates past its own relief.

    Reinjected at the budget alone: the engine, the ladder order, the workload and the card are untouched, and
    the only difference is that a rung is given less time than the hardware needs to return the memory. What
    comes back is the pre-fix regime in full: working rungs recorded as shortfalls and teardown rungs issued
    while the release that was going to relieve the card is still arriving. It fails on every card class, which
    is what makes the scaled budget the thing the scenario above measures rather than one card's arithmetic.
    """
    _blind_children_to_device_truth(monkeypatch)
    _reinject_sample_counted_verification(monkeypatch)
    world = _pressure_world(card=card)

    await _drive_pressure(world, job_count=_PRESSURE_JOBS, ticks=_PRESSURE_TICKS)

    assert _escalations_while_a_release_is_in_flight(world), (
        "the engine never escalated while one of its own releases was still landing, so the in-flight verdict "
        f"the scenario asserts would pass on a tree that grades rungs by sample count. {world.state_dump()}"
    )


# --------------------------------------------------------------------------------------------------------
# Safety thrash: the deepest rung is a process cycle, so it is spent at most once per dwell
# --------------------------------------------------------------------------------------------------------


def _safety_cycle_gaps(world: _DispatchWorld) -> list[float]:
    """Seconds between each pair of consecutive safety-off-GPU actuations in the run."""
    times = [when for _tick, when, _owner in world.safety_pause_events]
    return [later - earlier for earlier, later in zip(times, times[1:], strict=False)]


async def test_h_recurring_pressure_cycles_safety_at_most_once_per_dwell(monkeypatch: pytest.MonkeyPatch) -> None:
    """A card under recurring pressure moves safety off the GPU at most once per cooldown, and readmits it slowly.

    The failure this encodes: moving safety off the card ends the safety process, and the placement policy
    rebuilds it once the card fits it again, so the rung and its restore together are a full process cycle that
    stalls result submission while the rebuild runs. Nothing bounded how often that could be spent: every
    pressure episode reached the rung again within seconds, and every restore was taken on the first cycle the
    instantaneous gates passed, which is the moment the pause's own relief made them pass. A worker under
    recurring pressure then spends its session rebuilding safety, buying each time exactly the relief the
    previous cycle had already failed to hold.

    Read as one positive statement and its consequences: the rung carries a dwell, so consecutive safety
    actuations are a cooldown apart however often the card saturates; a ladder-owned pause earns its restore
    with the same sustained-fit evidence the placement policy requires rather than on the first passing sample;
    and the run still completes work, so the dwell is not bought by a worker that stopped serving.
    """
    _blind_children_to_device_truth(monkeypatch)
    world = _pressure_world()

    await _drive_pressure(world, job_count=_THRASH_JOBS, ticks=_THRASH_TICKS)

    context = "safety thrash"
    assert len(_saturation_episodes(world)) > 1, (
        f"{context}: the card saturated at most once, so nothing here would have asked for a second safety "
        f"cycle whatever the dwell was. {world.state_dump()}"
    )
    assert world.reclaim_ladder.safety_rungs_refused > 0, (
        f"{context}: the ladder never reached the safety rung a second time, so the run does not exercise the "
        f"dwell it asserts. {world.state_dump()}"
    )
    cooldown_seconds = reclaim_ladder_module._SAFETY_RUNG_COOLDOWN_SECONDS
    short_gaps = [gap for gap in _safety_cycle_gaps(world) if gap < cooldown_seconds]
    assert not short_gaps, (
        f"{context}: safety was cycled again {[f'{gap:.0f}s' for gap in short_gaps]} after the previous cycle, "
        f"inside its {cooldown_seconds:.0f}s dwell. Pauses at "
        f"{[f'{when:.0f}' for _t, when, _o in world.safety_pause_events]}. {world.state_dump()}"
    )
    for (pause_tick, _paused_at, _owner), (restore_tick, _restored_at) in zip(
        world.safety_pause_events,
        world.safety_restore_events,
        strict=False,
    ):
        assert restore_tick - pause_tick >= _SAFETY_PLACEMENT_RESTORE_STREAK, (
            f"{context}: safety was put back on the card {restore_tick - pause_tick} cycle(s) after the ladder "
            f"took it off, inside the {_SAFETY_PLACEMENT_RESTORE_STREAK}-cycle band of measured headroom a "
            f"restore has to earn. {world.state_dump()}"
        )
    assert world.completed_jobs > 0, (
        f"{context}: the run completed no jobs, so the dwell above was bought by a worker that stopped serving. "
        f"{world.state_dump()}"
    )


async def test_h_defect_reinjection_an_undwelt_safety_rung_thrashes_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the dwell removed, the same workload ends and rebuilds the safety process every couple of minutes.

    Reinjected at the dwell and the grading window together, because that is the pair that produced the
    signature: the shorter window sends every episode to the deepest rung, and the absent dwell lets every
    episode spend it. Everything else, including the restore band, is left as it is, so what fails is how often
    the cycle happens rather than the fact that one ever does.
    """
    dwell_seconds = reclaim_ladder_module._SAFETY_RUNG_COOLDOWN_SECONDS
    _blind_children_to_device_truth(monkeypatch)
    monkeypatch.setattr(reclaim_ladder_module, "_SAFETY_RUNG_COOLDOWN_SECONDS", 0.0)
    _reinject_sample_counted_verification(monkeypatch)
    world = _pressure_world()

    await _drive_pressure(world, job_count=_THRASH_JOBS, ticks=_THRASH_TICKS)

    gaps = _safety_cycle_gaps(world)
    assert gaps, (
        "an undwelt safety rung must be spent more than once over a run of recurring pressure, which is the "
        f"whole of this defect; it was spent {len(world.safety_pause_events)} time(s). {world.state_dump()}"
    )
    assert min(gaps) < dwell_seconds, (
        f"the closest pair of safety cycles was {min(gaps):.0f}s apart, still outside the {dwell_seconds:.0f}s "
        "dwell the scenario asserts, so that dwell would pass on a tree that has none"
    )


# --------------------------------------------------------------------------------------------------------
# Displaced residency: no record of weights a lane has given up may hide its model from the preload pass
# --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("max_threads", [1, 2], ids=["serial", "concurrent"])
async def test_i_a_rotation_restages_the_model_whose_lane_was_loaded_over(max_threads: int) -> None:
    """A rotation whose lanes are loaded over each other keeps staging the displaced models and drains.

    The failure this encodes: residency is recorded twice, once on the lane and once in a model-keyed map, and
    loading a lane over rewrites only the first. The displaced model's map entry survives naming a lane that
    holds something else, the preload pass counts that map in its already-loaded set, and the displaced
    model's pending job is therefore treated as served and never staged onto a free lane. What is left
    dispatchable is a job some lane does hold, which is not the head; the head-protection hold correctly
    withholds it so the head keeps the room it needs, and the head is one no pass will ever load. Both lanes
    idle, the queue stays full, and the card sits almost entirely free with nothing being reclaimed, because
    nothing is short of memory.

    The invariant, and the reason a hold is not the thing to relax: a hold may wait on a head, but no state
    the hold itself sustains may be what stops that head. The two residency records are reconciled every
    pass, so a lane loaded over leaves no record of weights it has given up, the displaced model is staged
    again, and the head that the hold defers to arrives. Read as one positive statement and its
    consequences: the parent never carries a record of weights no lane holds, every job reaches sampling, and
    the rotation drains inside a bounded schedule at a duty the card earns from.

    Run at both sampling capacities: the wedge is reached whether or not a second job may sample, so a fix
    verified at one of them says nothing about the operator posture that runs the other.
    """
    world = _streak_world(max_threads=max_threads)
    outlived_a_pass: list[str] = []
    standing: set[str] = set()
    jobs: list[ImageGenerateJobPopResponse] = []
    settling = 0
    width, height = _ROTATION_SHAPE

    for _ in range(_MAX_TICKS):
        while len(world.job_tracker.jobs_pending_inference) < _QUEUE_DEPTH and len(jobs) < _ROTATION_3_JOBS:
            model = _ROTATION_3[len(jobs) % len(_ROTATION_3)]
            job = make_job_pop_response(model.name, width=width, height=height, ddim_steps=30)
            await world.pop(job)
            jobs.append(job)
        await world.step()
        # A record is written as the load is commanded and reconciled by the next pass over the queue, so what
        # the invariant forbids is one that is still standing after a pass has run, not one seen mid-tick.
        current = set(world.phantom_model_records())
        outlived_a_pass.extend(f"tick {world.tick}: {record}" for record in sorted(current & standing))
        standing = current
        if world.completed_jobs >= _ROTATION_3_JOBS:
            settling += 1
            if settling >= _SETTLE_TICKS:
                break

    context = f"displaced residency ({max_threads} sampling slot(s))"
    _assert_streak_drained(world, jobs, context=context)
    assert not outlived_a_pass, (
        f"{context}: the parent kept recording a model as held on a lane that holds another one, across a "
        "pass that should have reconciled it:\n    " + "\n    ".join(outlived_a_pass[:4])
    )
    assert world.tick <= _ROTATION_3_TICK_CEILING, (
        f"{context}: the rotation took {world.tick} ticks to drain, past the {_ROTATION_3_TICK_CEILING}-tick "
        f"ceiling a run that keeps dispatching comes in under. {world.state_dump()}"
    )
    assert_duty_floor(world, _ROTATION_3_DUTY_FLOOR, context=context)


@pytest.mark.parametrize("max_threads", [1, 2], ids=["serial", "concurrent"])
async def test_i_defect_reinjection_a_displaced_record_hides_the_head_from_the_preload_pass(
    max_threads: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the displaced record left standing, the same rotation stops dispatching on a nearly empty card.

    Reinjected at the reconciliation alone: the lane is still rewritten, the map is still written, and the
    only difference is that a record naming a lane which holds another model is no longer read as describing
    weights nothing holds. What comes back is the wedge in full, and its signature is what distinguishes it
    from ordinary memory pressure: jobs pending against idle lanes, a card with most of its memory free, and
    no reclaim ordered at all, because nothing in the worker believes it is short of anything.
    """
    monkeypatch.setattr(
        InferenceScheduler,
        "_model_map_entry_is_displaced",
        staticmethod(lambda model_name, process_info: False),
    )
    world = _streak_world(max_threads=max_threads)

    jobs = await _drive_rotation(world, _ROTATION_3, job_count=_ROTATION_3_JOBS)

    unsampled = [str(job.id_)[:8] for job in jobs if world.dispatch_tick(job) is None]
    assert unsampled, (
        "a record of weights no lane holds must keep its model's jobs out of the preload pass entirely, which "
        f"is the whole of this defect; every job sampled. {world.state_dump()}"
    )
    assert world.job_tracker.jobs_pending_inference and not world.job_tracker.jobs_in_progress, (
        "the wedge idles every lane against a queue that still holds work; a run with a job sampling or an "
        f"empty queue is a slow rotation rather than this defect. {world.state_dump()}"
    )
    assert world.device_free_mb() > world.card.total_mb / 2, (
        f"the card ended with only {world.device_free_mb():.0f}MB of {world.card.total_mb:.0f}MB free, so this "
        f"run stopped for want of memory rather than for want of a record being reconciled. {world.state_dump()}"
    )
    assert world.reclaim_commands == 0, (
        f"the worker ordered {world.reclaim_commands} reclaim(s) while wedged, so something did believe the "
        f"card was short; this defect wedges a worker that believes it has everything it needs. "
        f"{world.state_dump()}"
    )
    assert duty_fraction(world) < _ROTATION_3_DUTY_FLOOR, (
        f"the wedged rotation reached {duty_fraction(world):.1%} duty, at or above the "
        f"{_ROTATION_3_DUTY_FLOOR:.1%} floor the scenario asserts, so that floor would pass unfixed"
    )
