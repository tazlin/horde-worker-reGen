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
- **Idle lanes holding device-warm components starve the head.** A component cache that keeps its entries on
  the device holds VRAM that belongs to no job: it survives every job boundary, no dispatch's footprint
  includes it, and the only thing that returns it is an unload the parent actuates on that lane. A card packed
  with two such lanes leaves the queue head unable to materialise, and the dispatch residency-reconciliation
  hold has nothing it will evict, so it waits for a fit that nothing is producing while both lanes sit idle.
  That is the ``held components`` scenario.
- **A child eviction the parent never learns about.** ComfyUI frees memory on the device to fund an
  allocation and the requirement it frees against is unbounded, so a checkpoint held under a retention grant
  can be gone before the job that was granted it ends. The grant suppresses the worker's own end-of-job
  evictor and nothing else. The parent's retained-resident record is a prediction made at dispatch and nothing
  the parent measures separates a slot whose weights are still there from one whose are not, so the record
  stands over an empty device for the rest of the session, charged by the retention fit, waited on by the
  dispatch gate, and routed to by same-model placement. The child is what closes it: a run granted the
  deferral that ends with an empty device reports the model out of VRAM, and the parent's ordinary unload
  reconciliation drops the record. That is the ``silent eviction`` scenario.
- **An unload the device refused, booked as room.** A full VRAM free is a request: the backend drops what it
  can and skips a model a live reference still pins, and the command reports nothing about the difference. A
  child that reports host-RAM residency because that is what it was asked to do hands the parent gigabytes of
  room the card is still holding, and the queue head is then held "not fitting" against a ledger that shows
  space after every evict. The child judging the unload by what the device still holds, and the parent keeping
  such a slot VRAM-resident and out of a reclaim that would ask it again, is what closes it. That is the
  ``refused unload`` scenario.
- **A selection that outlives the cycle it was derived in.** Dispatch selection may cache the job-and-lane pair
  a cycle chose so the cycle's look-ahead and its dispatch agree, and that pair is valid only while the lane
  still holds the job's model. Applying the children's reports invalidates it, and because every dispatch gate
  below refuses an undispatchable pair without clearing it, selection stays pinned to a job no lane can serve
  while a free lane holds preloaded weights for the head. That is the ``cycle-scoped selection`` scenario, and
  its reinjection is the cycle boundary itself: it pins the scope, which is the only thing keeping the stalled
  pair unreachable, so the harness may never drive a cycle without opening one.
- **A teardown that took the slot a dispatched job was pinned to.** A disaggregated sampler is reserved and
  granted execution ownership before its sample stage is sent, so it reports ``WAITING_FOR_JOB`` for the whole
  encode window. A whole-card teardown that reads idleness off child state alone sees a spare lane there and
  ends it, taking a job that was mid-flight with it. That is the ``pinned sampler`` scenario, and its
  reinjection is the ownership-blind selector.
- **A head that stood down from the card went on reserving it.** A churn governor may defer a whole-card
  head's establishment: a brake on how fast the card is rotated, and explicitly not a finding that the head
  cannot be served, so normal scheduling is supposed to continue around it. Head protection went on pricing
  that head's whole demand anyway, so every smaller ready job behind it was withheld to keep room for a demand
  nobody was making; the card sat empty for the length of the deferral with fitting work already on a lane,
  and the recovery ladder then fired constructive remedies against a governance decision. That is the
  ``governed head`` scenario, and its reinjection is head-protection pricing with the stand-down dropped.
- **A residency that tore down its own pre-stage target.** A pre-staged whole-card head carries its model on
  no lane until its load lands, and the model-name protection every residency shrink relies on cannot reach a
  slot with no name. A reprice tightening the target in that window is a shrink whose own target is the first
  idle lane it finds. That is the ``pre-stage target`` scenario, and its reinjection is the shrink called
  without the spare.
- **A copy on a card that cannot serve the job counted as the job's own.** Residency was judged card-blind
  while dispatch was judged per card, so a head whose model sat on a card its resolution excluded had no
  eligible copy to dispatch to and was called already loaded by the preload pass. Neither lane could move it
  and both cards idled against a full queue. That is the ``ineligible-card residency`` scenario, and its
  reinjection is the card-blind gate.
- **A sibling card's peak evicted safety from a card that was serving.** Runtime safety placement was judged
  against the heaviest sampling peak the whole worker was committed to. On a multi-card pool that peak is
  routinely a job this card can never be given, and one whose model it holds no weights for, so its entire
  footprint read as memory this card still had to find. The card's measured free never covered it, the demotion
  dwell was met within seconds, and the safety process was ended on a card comfortably serving its own traffic;
  the restore forecast read the same phantom, so the eviction never reversed. That is the ``per-card safety
  placement`` scenario, and its reinjection is the worker-wide peak.
"""

from __future__ import annotations

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeInferenceControlMessage, HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobStage
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import (
    SAFETY_READINESS_LATENCY_FLOOR_SECONDS,
    PauseOwner,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources import reclaim_ladder as reclaim_ladder_module
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from horde_worker_regen.process_management.resources.reclaim_ladder import ReclaimRungKind
from horde_worker_regen.process_management.scheduling.governance.whole_card import _GRACE_BUDGET_SECONDS
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _HEAD_PROTECTION_MAX_STARVE_SECONDS,
    _SAFETY_PLACEMENT_RESTORE_DWELL_FACTOR,
    _WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
    _WHOLE_CARD_RESTORE_GRACE_SECONDS,
    InferenceScheduler,
    NextJobAndProcess,
)
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import (
    _CARD_10GB,
    _CARD_16GB,
    _CARD_24GB,
    _CHILD_FREE_MARGIN_MB,
    _FLUX,
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
    assert_duty_floor_on_card,
    assert_free_floor,
    assert_governor_never_reached,
    assert_ladder_stayed_below,
    assert_never_idle_with_fitting_work,
    assert_no_committed_slot_retired,
    assert_no_duplicate_vram_copy,
    assert_no_unservable_dispatch_hold,
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

_SAFETY_READINESS_SECONDS = 30.0
"""How long one safety placement flip takes to reach readiness in these rows.

A safety child has to import the inference stack and materialise the classifier weights before it can clear a
single image, so a pause and its restore cost the pipeline this window twice. Modelling it is what makes an
in-flight placement rebuild observable, rather than collapsing a flip to an instant nothing can happen during."""

_SAFETY_LOAD_TRANSIENT_MB = 2000.0
"""How much more than its at-rest footprint safety charges the card while a restore is materialising.

The classifier weights are read and copied before the process settles, so the peak a restore imposes is above
the figure a fit priced it at, and a restore onto a card with barely enough room re-trips whatever evicted
safety in the first place."""

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
        safety_readiness_seconds=_SAFETY_READINESS_SECONDS,
        safety_load_transient_mb=_SAFETY_LOAD_TRANSIENT_MB,
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
    if models is None or shape is None:
        # Only a caller that leaves the workload to the card class needs the per-card table; one that names both
        # (a row about a card class the table does not cover) must not be held to having an entry there.
        configured_models, configured_shape = _PRESSURE_WORKLOAD[world.card.label]
        models = models or configured_models
        shape = shape or configured_shape
    width, height = shape
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


def _reinject_undwelt_safety_placement(monkeypatch: pytest.MonkeyPatch, world: _DispatchWorld) -> None:
    """Price a safety placement flip at nothing, which is what counting control cycles amounted to.

    The pre-fix band was a couple of control cycles to leave the card and a handful to come back. At a
    sub-second control loop that is a fraction of a second of evidence against a rebuild measured in tens of
    seconds, so the respawn window itself decided the next flip. Reinjected as a zero flip cost, which is the
    same statement without depending on how fast the row's ticks are.
    """
    monkeypatch.setattr(world.lifecycle, "safety_readiness_latency_seconds", lambda: 0.0)


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
    assert world.safety_pause_events, (
        f"{context}: safety was never taken off the card, so the run says nothing about how often that may "
        f"happen. {world.state_dump()}"
    )
    assert world.safety_restore_events, (
        f"{context}: safety never came back, so the run buys its dwell by leaving the worker without an "
        f"on-GPU safety process rather than by pacing the cycle. {world.state_dump()}"
    )
    ladder_owned = [owner for _tick, _when, owner in world.safety_pause_events if owner is PauseOwner.RECLAIM_LADDER]
    assert ladder_owned, (
        f"{context}: no safety pause was the ladder's, so the run does not exercise the ladder rung's dwell "
        f"or a ladder-owned restore. Owners: {[owner.name for _t, _w, owner in world.safety_pause_events]}. "
        f"{world.state_dump()}"
    )
    cooldown_seconds = reclaim_ladder_module._SAFETY_RUNG_COOLDOWN_SECONDS
    # A gap list is empty when only one pause happened, so it cannot by itself say the dwell paced anything;
    # bound the count directly by what the run's length allows under the cooldown.
    max_pauses_allowed = int((_THRASH_TICKS * _TICK_SECONDS) // cooldown_seconds) + 1
    assert len(world.safety_pause_events) <= max_pauses_allowed, (
        f"{context}: safety was taken off the card {len(world.safety_pause_events)} times in a run whose length "
        f"allows at most {max_pauses_allowed} under a {cooldown_seconds:.0f}s cooldown. {world.state_dump()}"
    )
    short_gaps = [gap for gap in _safety_cycle_gaps(world) if gap < cooldown_seconds]
    assert not short_gaps, (
        f"{context}: safety was cycled again {[f'{gap:.0f}s' for gap in short_gaps]} after the previous cycle, "
        f"inside its {cooldown_seconds:.0f}s dwell. Pauses at "
        f"{[f'{when:.0f}' for _t, when, _o in world.safety_pause_events]}. {world.state_dump()}"
    )
    restore_dwell_seconds = SAFETY_READINESS_LATENCY_FLOOR_SECONDS * _SAFETY_PLACEMENT_RESTORE_DWELL_FACTOR
    for (_pause_tick, paused_at, _owner), (_restore_tick, restored_at) in zip(
        world.safety_pause_events,
        world.safety_restore_events,
        strict=False,
    ):
        assert restored_at - paused_at >= _SAFETY_READINESS_SECONDS, (
            f"{context}: safety was put back on the card {restored_at - paused_at:.0f}s after it was taken off, "
            f"inside the {_SAFETY_READINESS_SECONDS:.0f}s the replacement process needs to reach readiness, so "
            f"the flip was undone by the rebuild it was still waiting on. {world.state_dump()}"
        )
        assert restored_at - paused_at >= restore_dwell_seconds, (
            f"{context}: safety was put back on the card {restored_at - paused_at:.0f}s after the ladder took "
            f"it off, inside the {restore_dwell_seconds:.0f}s of forecast headroom a restore has to earn. "
            f"{world.state_dump()}"
        )
    assert world.completed_jobs > 0, (
        f"{context}: the run completed no jobs, so the dwell above was bought by a worker that stopped serving. "
        f"{world.state_dump()}"
    )


async def test_h_defect_reinjection_an_undwelt_safety_rung_thrashes_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the dwell removed, the same workload ends and rebuilds the safety process every couple of minutes.

    Reinjected at the dwells and the grading window together, because that is the set that produced the
    signature: the shorter window sends every episode to the deepest rung, the absent rung cooldown lets every
    episode spend it, and a restore that earns nothing hands the card straight back into the pressure that
    evicted safety. What fails is how often the cycle happens rather than the fact that one ever does.
    """
    dwell_seconds = reclaim_ladder_module._SAFETY_RUNG_COOLDOWN_SECONDS
    _blind_children_to_device_truth(monkeypatch)
    monkeypatch.setattr(reclaim_ladder_module, "_SAFETY_RUNG_COOLDOWN_SECONDS", 0.0)
    _reinject_sample_counted_verification(monkeypatch)
    world = _pressure_world()
    _reinject_undwelt_safety_placement(monkeypatch, world)

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


# --------------------------------------------------------------------------------------------------------
# Held components: an idle lane's device-warm tenancy is still the card's, and something must ask for it back
# --------------------------------------------------------------------------------------------------------

_HELD_COMPONENT_MB = 6600.0
"""Device-warm component weights one idle lane carries in these scenarios.

Sized so two such lanes and three live contexts leave a 16 GB card with about a gigabyte free: enough that
nothing is broken until a head has to materialise, and far short of what one does."""

_HELD_COMPONENT_LANES = 2
"""Idle lanes carrying that tenancy: two, so the card's shortfall cannot be met by reclaiming one of them."""

_HELD_COMPONENT_DISPATCH_TICKS = 12
"""Ticks the head may take to reach sampling once the card is packed with idle held components.

A reclaim is an IPC the child services and a driver-side release that lands on a following tick, and the
hold re-asks each scheduling pass, so a couple of ticks is the floor and this is several times it. What it
excludes is a hold that waits for a fit nothing is producing."""

_HELD_COMPONENT_FREE_FLOOR_MB = 1024.0
"""Device free (MB) the card must hold above once the head has been let onto it.

The child's own margin: a card taken below it was committed against something other than device truth, which
is what a head admitted over an unreclaimed tenancy does."""


def _held_component_world() -> _DispatchWorld:
    """Build a 16 GB card whose two idle lanes hold device-warm components and whose third stages the head.

    The head is seeded staged rather than popped cold on purpose: staging is what puts the dispatch, not the
    preload, in charge of the fit, so the gate under test is the dispatch residency-reconciliation hold
    rather than the preload admission path that runs before it.
    """
    world = _DispatchWorld(
        card=_CARD_16GB,
        lane_count=_HELD_COMPONENT_LANES + 1,
        max_threads=1,
        queue_depth=_QUEUE_DEPTH,
        whole_card_enabled=True,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
    )
    for lane_id in range(_HELD_COMPONENT_LANES):
        world.seed_held_components(lane_id, _HELD_COMPONENT_MB)
    world.seed_resident(_HELD_COMPONENT_LANES, _SDXL, in_vram=False)
    return world


async def _drive_held_component_head(world: _DispatchWorld) -> ImageGenerateJobPopResponse:
    """Queue one head for the staged model and run until it samples or the tick ceiling is reached."""
    head = make_job_pop_response(_SDXL.name, width=1024, height=1024, ddim_steps=30)
    await world.pop(head)
    for _ in range(_MAX_TICKS):
        await world.step()
        if world.dispatch_tick(head) is not None:
            break
    return head


async def test_j_a_head_is_let_onto_a_card_packed_with_idle_held_components() -> None:
    """A head that cannot materialise over idle device-warm components gets them reclaimed, not waited on.

    The failure this encodes: two idle lanes held their component cache on the device, the queue head's
    materialisation did not fit, and the dispatch residency-reconciliation hold found nothing it would evict.
    The hold's only release is the arbiter verdicting a fit, and nothing was producing one: both lanes sat
    idle holding most of the card while the head re-asked every pass, until the run ended on the deadlock
    path. An idle tenancy the parent can see and can unload is not a reason to stop serving.

    Read as one statement and its consequences: the head samples inside a bounded number of ticks, something
    was actually reclaimed to let it, and the card it lands on still has the child's own margin standing. The
    escalation assertions are the other half: reaching a lane pause or moving safety off the card to recover
    an idle lane's cache would be a working worker paying a teardown for a reclaim it could have asked for.
    """
    world = _held_component_world()

    head = await _drive_held_component_head(world)

    context = "idle held components"
    dispatch_tick = world.dispatch_tick(head)
    assert dispatch_tick is not None and dispatch_tick <= _HELD_COMPONENT_DISPATCH_TICKS, (
        f"{context}: the head reached sampling at tick {dispatch_tick} against a "
        f"{_HELD_COMPONENT_DISPATCH_TICKS}-tick bound, so the hold is waiting on a fit nothing is producing "
        f"rather than asking an idle lane for its tenancy back. {world.state_dump()}"
    )
    assert world.reclaim_commands >= 1, (
        f"{context}: the head was let through without anything being reclaimed, so the card it materialised "
        f"onto still carries every idle tenancy. {world.state_dump()}"
    )
    assert_free_floor(world, _HELD_COMPONENT_FREE_FLOOR_MB, context=context)
    assert_ladder_stayed_below(world, FIRST_LANE_TEARDOWN_RUNG, context=context)


async def test_j_defect_reinjection_an_unreclaimable_tenancy_starves_the_head() -> None:
    """With the idle unload refused, the same card starves the head, which is what the bound above forbids.

    The reinjection is the actuator rather than a policy: whatever issues the reclaim, it issues it through
    the one idle-unload surface, so refusing that surface reproduces the regime the scenario is about without
    naming the decision that reaches it.
    """
    world = _held_component_world()
    world.scheduler.unload_idle_model = lambda process_id, device_index=None: False  # type: ignore[method-assign]

    head = await _drive_held_component_head(world)

    dispatch_tick = world.dispatch_tick(head)
    assert dispatch_tick is None or dispatch_tick > _HELD_COMPONENT_DISPATCH_TICKS, (
        f"the head reached sampling at tick {dispatch_tick} with every idle unload refused, so the bound the "
        f"scenario above asserts would pass on a worker that reclaims nothing. {world.state_dump()}"
    )
    assert world.reclaim_commands == 0, (
        f"an unload was booked with the actuator refusing every one: {world.state_dump()}"
    )


# --------------------------------------------------------------------------------------------------------
# The silent eviction: a retention record the device stopped honouring
# --------------------------------------------------------------------------------------------------------

_SILENT_EVICTION_UNDERSHOOT = 1.8
"""How much more of the card the streak's jobs really want than the scheduler's static fit priced them at.

The regime the eviction is reachable in. A lane funds a shortfall out of its own footprint first, so on a job
priced correctly its activation is always the smaller half and the copy it is running on is never the last
thing left. A job that costs the card most of a card more than it was admitted for is: the parent's own
defenses have already passed it, and only the child's freeing stands between the load and the device."""

_SILENT_EVICTION_JOBS = 8
"""Jobs in the streak these scenarios drive.

Every job in it re-uploads its weights, because that is what the eviction costs, and a worker's own model-churn
governance rightly holds a head off the card once a streak has thrashed the device that many times. This is
comfortably inside that, so what the scenarios measure is the record's honesty rather than the governor's."""


def _silent_eviction_world(*, child_evicts_granted_resident: bool) -> _DispatchWorld:
    """Build a 16 GB card serving a retained streak whose jobs cost it more than they were admitted for."""
    return _DispatchWorld(
        card=_CARD_16GB,
        lane_count=2,
        max_threads=1,
        queue_depth=_QUEUE_DEPTH,
        whole_card_enabled=True,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
        child_evicts_granted_resident=child_evicts_granted_resident,
        footprint_undershoot=_SILENT_EVICTION_UNDERSHOOT,
    )


async def test_k_a_child_side_eviction_is_reconciled_rather_than_left_as_a_resident() -> None:
    """When the child frees weights a grant covered, the parent's record follows the device, not the grant.

    The failure this encodes: retention was granted, the dispatch was priced for it, and ComfyUI freed the
    checkpoint during the run anyway to fund an allocation, because the grant suppresses the worker's own
    end-of-job evictor and nothing else. The parent settled the grant into its retained-resident record all
    the same, and from then on three paths acted on weights that were not there: the retention fit charged
    them, the dispatch admission gate held loads behind them, and same-model routing kept seating successors
    on the slot that held nothing.

    Read as one statement: no idle slot ever claims weights the card is not holding. Its consequences are
    that every dispatch after an eviction is priced as the cold load it really is, and that the streak still
    drains and still samples, so the record's honesty is not bought by declining to work.
    """
    world = _silent_eviction_world(child_evicts_granted_resident=True)

    jobs = await _drive_streak(world, _SDXL, job_count=_SILENT_EVICTION_JOBS)

    context = "silent child eviction"
    _assert_streak_drained(world, jobs, context=context)
    assert world.child_granted_resident_evictions, (
        f"{context}: the child never freed a granted copy, so the scenario measured nothing. {world.state_dump()}"
    )
    assert world.retained_resident_divergences == [], (
        f"{context}: an idle slot claimed weights the card was not holding on "
        f"{len(world.retained_resident_divergences)} tick(s), first at "
        f"{world.retained_resident_divergences[:1]}. {world.state_dump()}"
    )
    assert world.weight_uploads == len(jobs), (
        f"{context}: the streak paid {world.weight_uploads} weight uploads for {len(jobs)} jobs while the "
        f"child was freeing its copy on every one of them, so a dispatch is still being priced against a "
        f"copy the card does not hold. {world.state_dump()}"
    )
    assert_free_floor(world, _CHILD_FREE_MARGIN_MB, context=context)
    assert_ladder_stayed_below(world, FIRST_LANE_TEARDOWN_RUNG, context=context)


async def test_k_defect_reinjection_an_unreconciled_eviction_leaves_a_phantom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the unload reconciliation removed, the same run leaves a slot claiming an empty device.

    The reinjection is the parent half of the fix: the child still reports the model out of VRAM, and the
    handler that turns that report into a cleared record is made inert. What is left is the pre-fix worker
    exactly, and the record it keeps is the phantom every later pricing, hold and routing decision is made
    against.
    """
    monkeypatch.setattr(ProcessMap, "on_model_vram_clear", lambda self, process_id: None)
    world = _silent_eviction_world(child_evicts_granted_resident=True)

    jobs = await _drive_streak(world, _SDXL, job_count=_SILENT_EVICTION_JOBS)

    _assert_streak_drained(world, jobs, context="unreconciled eviction")
    assert world.child_granted_resident_evictions, "the reinjection must run the same evictions the scenario does"
    assert world.retained_resident_divergences != [], (
        "with the reconciliation inert an idle slot must go on claiming weights the card does not hold, or "
        f"the scenario's record assertion would pass on a worker that never reconciles. {world.state_dump()}"
    )


async def test_k_a_streak_the_child_never_disturbs_keeps_its_retained_copy() -> None:
    """The control: on the same packed card, a child that honours the grant reuses one copy all streak.

    Without this the scenario above would be satisfied by a worker whose retention never holds anything, and
    a record that claims nothing is trivially honest.
    """
    world = _silent_eviction_world(child_evicts_granted_resident=False)

    jobs = await _drive_streak(world, _SDXL, job_count=_SILENT_EVICTION_JOBS)

    context = "undisturbed streak"
    _assert_streak_drained(world, jobs, context=context)
    assert world.child_granted_resident_evictions == [], "the control must run with the child eviction off"
    assert world.weight_uploads == _STREAK_WEIGHT_UPLOADS, (
        f"{context}: the streak paid {world.weight_uploads} weight uploads for {len(jobs)} jobs, so the "
        f"retained copy is not surviving the job boundary on this card at all. {world.state_dump()}"
    )
    assert world.retained_resident_divergences == [], (
        f"{context}: the record diverged from the card without the child ever disturbing it. {world.state_dump()}"
    )


# --------------------------------------------------------------------------------------------------------
# The refused unload: room the parent counted and the card never gave back
# --------------------------------------------------------------------------------------------------------

_UNLOAD_LEAK_MB = 3000.0
"""Weights (MB) the child's backend cannot free out of a lane it was told to unload.

A full free drops what it can and skips a model a live reference still pins. Sized as a fraction of the
resident copy rather than all of it, so the unload genuinely helps and the question is only whether the
parent's ledger knows what it did not get back."""

_UNLOAD_LEAK_DISPATCH_TICKS = 8
"""Ticks the head may take to sample once the partial unload has landed.

A release is an IPC the child services and a driver-side give-back on a following tick, so a couple of ticks
is the floor. What this excludes is a head parked behind a refusal nothing escalates past."""


def _refused_unload_world() -> _DispatchWorld:
    """A 16 GB card holding a large idle resident that will only partly unload, and a staged head.

    The head is staged rather than popped cold so the fit is decided at dispatch, and the idle resident is
    the extra-large class so the head cannot materialise until something gives the card back.
    """
    world = _DispatchWorld(
        card=_CARD_16GB,
        lane_count=2,
        max_threads=1,
        queue_depth=_QUEUE_DEPTH,
        whole_card_enabled=True,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
        child_unload_leaks_mb=_UNLOAD_LEAK_MB,
    )
    world.seed_resident(0, _FLUX, in_vram=True)
    world.seed_resident(1, _SDXL, in_vram=False)
    return world


async def _drive_refused_unload_head(world: _DispatchWorld) -> ImageGenerateJobPopResponse:
    """Queue one head for the staged model and run until it samples or the tick ceiling is reached."""
    head = make_job_pop_response(_SDXL.name, width=1024, height=1024, ddim_steps=30)
    await world.pop(head)
    for _ in range(_MAX_TICKS):
        await world.step()
        if world.dispatch_tick(head) is not None:
            break
    return head


async def test_l_an_unload_the_device_refused_is_not_recorded_as_room() -> None:
    """A slot that could not give its weights back keeps reading as VRAM-resident, and the head still moves.

    The failure this encodes: an unload was issued, the backend freed one component out of nearly nine
    gigabytes and left the rest loaded behind a live reference, and the child reported the model moved to
    host RAM because that is what the command had asked for. The parent booked the whole footprint as room
    it had recovered, and for the rest of the session admitted against gigabytes the card was still holding;
    the queue head was held "not fitting" for minutes while the ledger showed space after every evict.

    Read as one statement: the parent never records host-RAM residency for weights the card is holding. Its
    consequences are that the refusal is remembered once rather than re-asked every tick, and that the head
    still reaches sampling, so honesty about the refusal is not bought by parking the queue behind it.
    """
    world = _refused_unload_world()

    head = await _drive_refused_unload_head(world)

    context = "refused unload"
    assert world.unload_leaks, (
        f"{context}: no unload was refused, so the scenario measured nothing. {world.state_dump()}"
    )
    assert world.ram_recorded_over_resident_weights() == [], (
        f"{context}: the parent recorded host-RAM residency for weights still on the card: "
        f"{world.ram_recorded_over_resident_weights()}. {world.state_dump()}"
    )
    assert world.unload_refused_lanes() == [0], (
        f"{context}: the refusing lane is not marked as one, so reclaim will keep choosing it. {world.state_dump()}"
    )
    dispatch_tick = world.dispatch_tick(head)
    assert dispatch_tick is not None and dispatch_tick <= _UNLOAD_LEAK_DISPATCH_TICKS, (
        f"{context}: the head reached sampling at tick {dispatch_tick} against a "
        f"{_UNLOAD_LEAK_DISPATCH_TICKS}-tick bound, so it is parked behind a refusal nothing escalates past. "
        f"{world.state_dump()}"
    )
    assert world.reclaim_commands == 1, (
        f"{context}: {world.reclaim_commands} unloads were served on a card with one thing to reclaim, so "
        f"the same refusal is being asked again every tick instead of escalating. {world.state_dump()}"
    )


async def test_l_defect_reinjection_an_unverified_unload_books_room_the_card_still_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the refusal bookkeeping inert, the same unload is booked as room the card never gave back.

    The reinjection is the parent half: the child still judges the unload by what the device holds and still
    reports the refusal, and the handler that turns that report into a slot kept VRAM-resident is made a
    no-op. What is left is the pre-fix worker, whose map follows the command it issued rather than the device
    it issued it to.
    """
    monkeypatch.setattr(ProcessMap, "on_vram_unload_refused", lambda self, process_id: None)
    world = _refused_unload_world()

    await _drive_refused_unload_head(world)

    assert world.unload_leaks, "the reinjection must run the same refused unload the scenario does"
    assert world.ram_recorded_over_resident_weights() != [], (
        "with the refusal bookkeeping inert the parent must book host-RAM residency for weights the card is "
        f"still holding, or the scenario's ledger assertion would pass unfixed. {world.state_dump()}"
    )


async def test_l_an_unload_the_device_honoured_marks_nothing() -> None:
    """The control: the ordinary complete unload leaves no refusal behind and gives the card everything back.

    Without this the scenario above would be satisfied by a worker that marks every unload refused, which
    would keep every slot out of reclaim for the rest of the session.
    """
    world = _refused_unload_world()
    world.child_unload_leaks_mb = 0.0

    head = await _drive_refused_unload_head(world)

    context = "honoured unload"
    assert world.unload_leaks == [], "the control must run with the leak off"
    assert world.unload_refused_lanes() == [], (
        f"{context}: a completed unload must leave no slot marked as refusing. {world.state_dump()}"
    )
    assert world.dispatch_tick(head) is not None, f"{context}: the head never sampled. {world.state_dump()}"


# --------------------------------------------------------------------------------------------------------
# Committed slots a shrink must not take
# --------------------------------------------------------------------------------------------------------

_COMMITTED_SLOT_TICK_SECONDS = 2.0
"""Seconds per tick for the two shrink scenarios, matching the rest of this module's closed-loop pace."""

_ENCODE_TICKS = 6
"""Ticks a pinned disaggregated sampler waits on the encode lane before its own sample stage runs.

Long enough that the whole-card head arriving behind it finds the sampler still pinned and still reporting
itself idle, which is the only window the defect exists in."""

_PRESTAGE_LOAD_TICKS = 3
"""Ticks a whole-card pre-stage's preload spends in flight, so the load is observable rather than atomic."""

_COMMITTED_SLOT_SETTLE_TICKS = 40
"""Ticks driven after the disturbance, so each scenario also says the queue kept moving afterwards."""


async def _seed_live_job(world: _DispatchWorld, model: _ModelClass, *, steps: int) -> ImageGenerateJobPopResponse:
    """Pop one job for ``model`` and run until it is dispatched, so a live job holds the card."""
    job = make_job_pop_response(model.name, width=1024, height=1024, ddim_steps=steps)
    await world.pop(job)
    for _ in range(_MAX_TICKS):
        await world.step()
        if world.dispatch_tick(job) is not None:
            return job
    raise AssertionError(f"the seeded job never reached a lane. {world.state_dump()}")


def _pinned_sampler_world() -> _DispatchWorld:
    """A card serving disaggregation-class work, with room for a whole-card head to demand the device.

    Three lanes on a card a Flux head fills on its own, so the head's residency has a real teardown to order
    and the pinned sampler is one of the lanes it can order away. Only the SDXL class runs disaggregated: the
    head is priced and admitted as the whole-job load it is, which is what makes it ask for the card at all.
    """
    world = _DispatchWorld(
        card=_CARD_16GB,
        lane_count=3,
        max_threads=2,
        queue_depth=4,
        whole_card_enabled=True,
        closed_loop=True,
        tick_seconds=_COMMITTED_SLOT_TICK_SECONDS,
        cooldown_seconds=120,
        disaggregated=True,
        disaggregated_encode_seconds=_COMMITTED_SLOT_TICK_SECONDS * _ENCODE_TICKS,
    )
    world.scheduler._is_disaggregation_class_eligible = lambda job: job.model == _SDXL.name  # type: ignore[method-assign]
    return world


async def _drive_pinned_sampler_scenario(world: _DispatchWorld) -> tuple[ImageGenerateJobPopResponse, int]:
    """Pin a sampler, put a whole-card head behind it, and run past the residency's teardown.

    Returns:
        The pinned job and the lane it was dispatched onto.
    """
    pinned = await _seed_live_job(world, _SDXL, steps=30)
    lane_id = world.lane_serving(pinned)
    assert lane_id is not None, f"the pinned job is on no lane. {world.state_dump()}"
    assert world.scheduler._process_map[lane_id].current_inference_job() is not None, (
        f"precondition: the dispatched sampler must own its job. {world.state_dump()}"
    )
    assert world.scheduler._process_map[lane_id].can_accept_job(), (
        "precondition: the pinned sampler must still report itself idle, or the scenario measures nothing. "
        f"{world.state_dump()}"
    )
    head = make_job_pop_response(_FLUX.name, width=1024, height=1024, ddim_steps=30)
    await world.pop(head)
    for _ in range(_COMMITTED_SLOT_SETTLE_TICKS):
        await world.step()
    return pinned, lane_id


async def test_m_a_whole_card_teardown_spares_the_lane_a_pinned_sampler_owns() -> None:
    """A residency's teardown leaves a slot that owns a dispatched job alone, and still collapses the pool.

    The failure this encodes: a disaggregated sampler is pinned to its slot and granted execution ownership
    before the sample stage is sent, so it sits reporting ``WAITING_FOR_JOB`` for the whole encode window. A
    teardown that reads idleness off child state alone sees a free lane, ends it, and takes the job with it.

    Read as one statement: ownership of a dispatched job makes a slot busy, whatever its child is reporting.
    The pool still collapses toward the residency's target around it, so the guarantee is not bought by a
    teardown that refuses to run.
    """
    world = _pinned_sampler_world()

    pinned, lane_id = await _drive_pinned_sampler_scenario(world)

    context = "pinned sampler"
    assert_no_committed_slot_retired(world, context=context)
    assert world.stage(pinned) is not JobStage.PENDING_INFERENCE, (
        f"{context}: the pinned job never left the queue. {world.state_dump()}"
    )
    assert world.completed_jobs >= 1, (
        f"{context}: the pinned sampler's job never completed, so the lane it was on was serving nothing. "
        f"{world.state_dump()}"
    )
    assert len(world.inference_lane_ids()) < 3, (
        f"{context}: the pool never shrank at all, so sparing the pinned lane cost the residency its "
        f"teardown rather than redirecting it. {world.state_dump()}"
    )


def _kill_selection_blind_to_ownership(
    self: ProcessMap,
    disallowed_processes: list[int] | None = None,
) -> HordeProcessInfo | None:
    """Victim selection that reads idleness from the child's reported state alone.

    The pre-fix selector: every skip it applies is a statement about what the child last said, so a slot the
    parent has dispatched a job onto and is waiting on another lane for reads exactly like a spare.
    """
    for process_info in self.values():
        if process_info.process_type != HordeProcessType.INFERENCE:
            continue
        if process_info.process_id in (disallowed_processes or []):
            continue
        if process_info.is_process_busy():
            continue
        if process_info.last_process_state in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED):
            continue
        return process_info
    return None


async def test_m_defect_reinjection_an_ownership_blind_teardown_takes_the_pinned_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ownership invisible to victim selection, the same teardown ends the lane that owns the job.

    Everything else is the scenario above: the same card, the same pinned sampler, the same head asking for
    the device. Only the selector's view of the slot changes.
    """
    monkeypatch.setattr(ProcessMap, "_get_first_inference_process_to_kill", _kill_selection_blind_to_ownership)
    world = _pinned_sampler_world()

    _pinned, lane_id = await _drive_pinned_sampler_scenario(world)

    assert any(
        f"lane {lane_id} was retired while owning dispatched job" in entry
        for entry in world.committed_slot_retirements
    ), (
        "with ownership invisible the teardown must take the pinned sampler's lane, or the scenario's "
        f"assertion would pass unfixed. retirements={world.committed_slot_retirements} {world.state_dump()}"
    )


def _prestage_target_world() -> _DispatchWorld:
    """A card whose whole-card head is pre-staged into a spare while a live job still holds the device."""
    return _DispatchWorld(
        card=_CARD_16GB,
        lane_count=3,
        max_threads=2,
        queue_depth=4,
        whole_card_enabled=True,
        closed_loop=True,
        tick_seconds=_COMMITTED_SLOT_TICK_SECONDS,
        cooldown_seconds=120,
        preload_latency_seconds=_COMMITTED_SLOT_TICK_SECONDS * _PRESTAGE_LOAD_TICKS,
    )


async def _drive_prestage_target_scenario(world: _DispatchWorld) -> int:
    """Pre-stage a whole-card head, lose its load mid-flight, and run the reprice ticks that follow.

    Returns:
        The lane the residency recorded as its pre-stage target.
    """
    await _seed_live_job(world, _SDXL, steps=60)
    head = make_job_pop_response(_FLUX.name, width=1024, height=1024, ddim_steps=30)
    await world.pop(head)
    prestage_lane: int | None = None
    for _ in range(_MAX_TICKS):
        await world.step()
        prestage_lane = world.scheduler._whole_card_ledger.state_for(None).prestage_process_id
        if prestage_lane is not None:
            break
    assert prestage_lane is not None, (
        f"the head was never pre-staged, so the scenario has no target. {world.state_dump()}"
    )
    assert world.scheduler._process_map[prestage_lane].last_process_state is HordeProcessState.PRELOADING_MODEL, (
        f"precondition: the pre-stage load must still be in flight when it is lost. {world.state_dump()}"
    )
    # The child carrying the pre-stage dies mid-load and its slot is rebuilt empty. The residency still names
    # that slot and no lane carries the head's model, so the convergence shrink (which waits for a holder)
    # stands down and the reprice is the only shrink left running over a pool it can still cut.
    assert world.kill_lane_holding(_FLUX), f"the pre-stage load had no lane to lose. {world.state_dump()}"
    for _ in range(_COMMITTED_SLOT_SETTLE_TICKS):
        await world.step()
    return prestage_lane


async def test_n_a_reprice_spares_the_slot_a_whole_card_head_is_pre_staging_into() -> None:
    """A tightened residency target does not take the slot its own head is being loaded into.

    The failure this encodes: a pre-staged head carries its model on no lane until its load lands, so the
    model-name protection every residency shrink relies on cannot reach the slot the pre-stage chose. A
    reprice that tightens the target in that window is a shrink with nothing protecting its own target, and
    the slot the residency is loading into is the first idle lane it finds.

    Read as one statement: the slot a residency has committed to is off-limits to that residency's own
    shrink. The pool is still cut around it, so sparing the target does not cost the reprice its reduction.
    """
    world = _prestage_target_world()

    prestage_lane = await _drive_prestage_target_scenario(world)

    context = "pre-stage target"
    assert_no_committed_slot_retired(world, context=context)
    assert prestage_lane in world.inference_lane_ids(), (
        f"{context}: lane {prestage_lane} left the pool while the residency was still loading its head into "
        f"it. {world.state_dump()}"
    )
    assert len(world.inference_lane_ids()) < 3, (
        f"{context}: the reprice never cut the pool at all, so sparing its target cost it the reduction "
        f"rather than redirecting it. {world.state_dump()}"
    )


def _scale_without_sparing(
    self: InferenceScheduler,
    target: int,
    *,
    device_index: int | None,
    protected_model: str | None,
    spared_process_id: int | None,
) -> int:
    """Scale the pool with the caller's committed slot dropped from what the shrink must not take."""
    del spared_process_id
    return self._process_lifecycle.scale_inference_processes(
        target,
        device_index=device_index,
        protected_model=protected_model,
    )


async def test_n_defect_reinjection_an_unspared_reprice_takes_its_own_pre_stage_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the spare dropped, the residency's own reprice ends the slot it is loading its head into.

    The reinjection drops the named spare from every residency shrink; the tick under test is the reprice,
    which is the only shrink running in this scenario's window (the convergence shrink stands down while the
    head has no holder, which is the same condition that leaves the slot unprotected by name).
    """
    monkeypatch.setattr(InferenceScheduler, "_scale_sparing", _scale_without_sparing)
    world = _prestage_target_world()

    prestage_lane = await _drive_prestage_target_scenario(world)

    assert any(
        f"lane {prestage_lane} was retired while it was the pre-stage target" in entry
        for entry in world.committed_slot_retirements
    ), (
        "with the spare dropped the reprice must end its own pre-stage target, or the scenario's assertion "
        f"would pass unfixed. retirements={world.committed_slot_retirements} {world.state_dump()}"
    )


# --------------------------------------------------------------------------------------------------------
# Governed head: a head that has stood down from the card may not go on reserving it from the queue behind it
# --------------------------------------------------------------------------------------------------------

_GOVERNED_HEAD_TICK_SECONDS = 2.0
"""Seconds per tick for the governed-head scenarios.

The windows that decide this scenario are the governor's deferral dwell (240s) and head protection's own
starvation release (120s), and what it is about is the span between them. A tick worth a scheduling interval
would step over that span in a handful of samples; at two seconds the idle run the defect produces is measured
in the same units the incident was."""

_GOVERNED_HEAD_TICKS = 90
"""Ticks the governed-head scenarios run for: inside the governor's dwell, so the deferral is still in force
at the end and the run says something about the deferral rather than about its expiry."""

_GOVERNED_LIGHT_JOBS = 3
"""Light jobs queued behind the governed head. Three is enough that serving them is a streak rather than a
single dispatch that could be luck, and few enough that the run stays inside the dwell."""


def _governed_head_world() -> _DispatchWorld:
    """A 16 GB card whose whole-card grace allowance is spent, with the light class already resident.

    The allowance is spent through the ledger's own charge record, which is what a card that has recently
    cycled a residency twice carries. The light checkpoint is seeded resident because the failure is about
    dispatch, not about staging: a line-skip only exists where a smaller ready job is already on a lane the
    card can dispatch it from.
    """
    world = _DispatchWorld(
        card=_CARD_16GB,
        lane_count=3,
        max_threads=2,
        queue_depth=4,
        whole_card_enabled=True,
        closed_loop=True,
        tick_seconds=_GOVERNED_HEAD_TICK_SECONDS,
        cooldown_seconds=120,
    )
    state = world.scheduler._whole_card_ledger.state_for(None)
    cycle_seconds = _WHOLE_CARD_ESTABLISH_GRACE_SECONDS + _WHOLE_CARD_RESTORE_GRACE_SECONDS
    for index in range(int(_GRACE_BUDGET_SECONDS // cycle_seconds) + 1):
        state.grace_charges.append((world.now - index, _WHOLE_CARD_ESTABLISH_GRACE_SECONDS))
        state.grace_charges.append((world.now - index, _WHOLE_CARD_RESTORE_GRACE_SECONDS))
    assert world.scheduler._whole_card_ledger.grace_budget_exhausted(None, now=world.now), (
        "precondition: the card's rolling grace allowance must be spent, or no governor defers anything"
    )
    world.seed_resident(0, _SD15, in_vram=True)
    return world


async def _drive_governed_head(
    world: _DispatchWorld,
) -> tuple[ImageGenerateJobPopResponse, list[ImageGenerateJobPopResponse]]:
    """Queue a whole-card head with light work behind it and run inside the governor's dwell.

    Returns:
        The head, and the light jobs queued behind it in order.
    """
    head = make_job_pop_response(_FLUX.name, width=1024, height=1024, ddim_steps=20)
    await world.pop(head)
    light: list[ImageGenerateJobPopResponse] = []
    for _ in range(_GOVERNED_LIGHT_JOBS):
        job = make_job_pop_response(_SD15.name, width=512, height=512, ddim_steps=20)
        await world.pop(job)
        light.append(job)
    for _ in range(_GOVERNED_HEAD_TICKS):
        await world.step()
    return head, light


def _assert_the_deferral_was_in_force(world: _DispatchWorld) -> None:
    """Assert the run really spent its ticks behind a governor deferral of the head's whole-card ask."""
    ledger = world.scheduler._whole_card_ledger
    assert ledger.governor_deferred_head(None, now=world.now) == _FLUX.name, (
        "precondition: the head's whole-card establishment must still be governor-deferred at the end of the "
        f"run, or these ticks say nothing about a deferral. {world.state_dump()}"
    )
    assert not world.scheduler.is_whole_card_residency_active(), (
        "precondition: the deferral must have stopped the residency being established, or the card was given "
        f"to the head after all. {world.state_dump()}"
    )


async def test_o_a_governor_deferred_head_lets_the_work_behind_it_run() -> None:
    """A head whose whole-card establishment is governor-deferred stops reserving the card from the queue.

    The failure this encodes: the churn governor deferred a whole-card head's establishment, which is a brake
    on how fast the card may be rotated and explicitly not a finding that the head cannot be served. Normal
    scheduling is meant to continue around such a head. Head protection went on pricing that head's whole
    fifteen-gigabyte demand anyway, so every smaller ready job behind it was withheld to keep room for a
    demand nobody was making, and the card sat empty for the length of the deferral with fitting work on a
    lane it could have been dispatched from. Save-our-ship remedies then fired against what was a governance
    decision.

    Read as one statement: a head that has stood down from asking for the card reserves nothing from the jobs
    behind it. Its consequences are that the light work runs, that no tick passes idle with work the card's
    free VRAM covers, and that no dispatch is held for an entity going nowhere.
    """
    world = _governed_head_world()

    head, light = await _drive_governed_head(world)

    context = "governor-deferred head"
    _assert_the_deferral_was_in_force(world)
    assert world.dispatch_tick(head) is None, (
        f"{context}: the deferred head itself must not have taken the card, or the run measures an ordinary "
        f"dispatch rather than what happens behind a deferral. {world.state_dump()}"
    )
    unserved = [str(job.id_)[:8] for job in light if world.dispatch_tick(job) is None]
    assert not unserved, (
        f"{context}: light job(s) {unserved} that fit the card were never dispatched while the head they sit "
        f"behind was standing down from asking for it. {world.state_dump()}"
    )
    assert_never_idle_with_fitting_work(world, context=context)
    assert_no_unservable_dispatch_hold(world, context=context)


def _price_the_deferred_head(
    self: InferenceScheduler,
    displaced_head: ImageGenerateJobPopResponse,
    *,
    device_index: int | None,
) -> float | None:
    """Head-protection pricing with the governance stand-down dropped, and nothing else changed.

    The starvation release is kept, so the reinjected worker is one that eventually notices the head is not
    converging; what it does not have is the knowledge that the head is not asking for the card at all.
    """
    del device_index
    if displaced_head.model is None:
        return None
    if self._head_starved_seconds(displaced_head) >= _HEAD_PROTECTION_MAX_STARVE_SECONDS:
        return None
    return self._measured_admission_candidate_delta_mb(
        displaced_head,
        self._model_metadata.get_baseline(displaced_head.model),
        process_id=None,
        disaggregated=self._is_disaggregation_class_eligible(displaced_head),
    )


async def test_o_defect_reinjection_a_governor_deferred_head_goes_on_reserving_the_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the stand-down dropped from head-protection pricing, the card idles behind a head nobody is loading.

    Reinjected at the pricing alone: the governor still defers, the disclosure still says normal scheduling
    continues, and the only difference is that the deferred head's demand is charged against the room a
    fitting sibling would take. Both liveness verdicts must then fail, and their messages are what say which
    failure this is: a run of idle ticks with fitting work on the card, and a dispatch held for a head that is
    cold, unloading, and standing down.
    """
    monkeypatch.setattr(InferenceScheduler, "_displaced_head_outstanding_mb", _price_the_deferred_head)
    world = _governed_head_world()

    _head, light = await _drive_governed_head(world)

    _assert_the_deferral_was_in_force(world)
    assert [job for job in light if world.dispatch_tick(job) is None], (
        "with the deferred head's demand charged again, work behind it must be withheld, or the reinjection "
        f"did not reach the pricing it aims at. {world.state_dump()}"
    )
    with pytest.raises(AssertionError, match="the card was idle for"):
        assert_never_idle_with_fitting_work(world, context="governor-deferred head")
    with pytest.raises(AssertionError, match="head_protection hold"):
        assert_no_unservable_dispatch_hold(world, context="governor-deferred head")


# --------------------------------------------------------------------------------------------------------
# Ineligible-card residency: a copy on a card that cannot serve the job is not a copy the job has
# --------------------------------------------------------------------------------------------------------

_INELIGIBLE_CARD_PIXELS = {0: 4_194_304, 1: 262_144}
"""Per-card resolution ceilings for the two-card pool: a large card and one whose ``max_power`` override caps
it at 512x512. Only the resolution axis differs, so which cards may serve a job is decided by its pixels
alone."""

_INELIGIBLE_CARD_HEAD_SHAPE = (1024, 1024)
"""The head's resolution: inside the large card's ceiling and four times the small card's, so the head can
only ever run on the large card while the model it needs sits on the small one."""

_INELIGIBLE_CARD_TICK_CEILING = 10
"""Ticks the head may take to reach sampling once its own copy has to be loaded onto the eligible card.

A preload is commanded on one pass, reported on the next and dispatched after that, so a handful of ticks is
the floor. What this excludes is a head that never loads at all."""

_INELIGIBLE_CARD_TICKS = 30
"""Ticks each of these rows runs, comfortably past the ceiling above so a run that wedges shows a long idle
stretch rather than merely an unfinished one."""


def _ineligible_card_world() -> _DispatchWorld:
    """Two cards of differing resolution ceilings, the head's model resident only on the card that cannot serve it.

    One lane per card, one sampling slot, and a queue depth of one: the operator posture the failure was found
    on. The model sits on the small card as a genuine, dispatchable residency, so nothing here is stale; the
    head simply needs a copy of it on the only card allowed to run it.
    """
    world = _DispatchWorld(
        card=_CARD_24GB,
        lane_count=2,
        max_threads=1,
        queue_depth=1,
        whole_card_enabled=False,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        card_max_pixels=_INELIGIBLE_CARD_PIXELS,
    )
    world.seed_resident(1, _SD15, in_vram=True)
    assert world.card_of_lane(1) == 1, "precondition: the seeded copy must sit on the small card"
    assert world.card_of_lane(0) == 0, "precondition: the free lane must sit on the large card"
    return world


async def _drive_ineligible_card_head(
    world: _DispatchWorld,
) -> tuple[ImageGenerateJobPopResponse, list[int]]:
    """Queue the oversized head against the small card's resident copy and run the row out.

    Returns:
        The head, and the ticks on which the missing-model recovery latch was standing.
    """
    width, height = _INELIGIBLE_CARD_HEAD_SHAPE
    head = make_job_pop_response(_SD15.name, width=width, height=height, ddim_steps=20)
    await world.pop(head)
    latched_ticks: list[int] = []
    for _ in range(_INELIGIBLE_CARD_TICKS):
        await world.step()
        if world.scheduler._model_recently_missing:
            latched_ticks.append(world.tick)
    return head, latched_ticks


async def test_p_a_head_gets_its_own_copy_when_the_resident_card_cannot_serve_it() -> None:
    """A model resident only on a card that cannot serve the head is loaded again onto one that can.

    The failure this encodes: residency was judged card-blind while dispatch was judged per card. The head's
    model was resident, on a card whose resolution ceiling excluded this job, so dispatch found nothing to
    dispatch to and the preload pass called the model already loaded. Neither lane could move the head:
    dispatch had no eligible copy and no pass would fund one, and both cards idled against a full queue until
    save-our-ship rebuilt the pools. Dispatch also read the empty per-card lookup as a stale residency record
    and expired a model-map entry that was telling the truth, latching the missing-model flag against a card
    that was serving.

    Read as one statement: a copy on a card that cannot serve a job is not a copy that job has. Its
    consequences are that the head is staged onto an eligible card and sampled inside a bounded schedule, that
    no card sits idle with work its memory covers, and that nothing treats the truthful residency as missing.
    """
    world = _ineligible_card_world()

    head, latched_ticks = await _drive_ineligible_card_head(world)

    context = "ineligible-card residency"
    dispatch_tick = world.dispatch_tick(head)
    assert dispatch_tick is not None, (
        f"{context}: the head never reached sampling, though the large card was free the whole run and the "
        f"only thing it needed was its own copy of {_SD15.name}. {world.state_dump()}"
    )
    assert dispatch_tick <= _INELIGIBLE_CARD_TICK_CEILING, (
        f"{context}: the head took {dispatch_tick} ticks to reach sampling, past the "
        f"{_INELIGIBLE_CARD_TICK_CEILING}-tick ceiling a load onto a free eligible lane comes in under. "
        f"{world.state_dump()}"
    )
    lanes = [lane for model, lane in world.dispatch_lanes if model == _SD15.name]
    assert lanes and all(world.card_of_lane(lane) == 0 for lane in lanes), (
        f"{context}: the head was dispatched to lane(s) {lanes}, which is not on the only card whose "
        f"resolution ceiling covers it. {world.state_dump()}"
    )
    assert not latched_ticks, (
        f"{context}: the missing-model recovery was latched on tick(s) {latched_ticks[:6]} over a model that "
        f"was resident throughout, so a truthful residency record was expired as stale. {world.state_dump()}"
    )
    assert_never_idle_with_fitting_work(world, context=context)


async def test_p_defect_reinjection_a_card_blind_residency_gate_wedges_both_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the already-loaded gate card-blind again, the head is never staged and both cards idle.

    Reinjected at that gate alone: dispatch still routes per card and still finds nothing, and the only
    difference is that the preload pass counts a copy on a card that cannot serve this job. What comes back is
    the wedge in full, and its signature is what distinguishes it from memory pressure: a queue that holds
    work, lanes that hold nothing to do, and a card with almost all of its memory free.
    """
    monkeypatch.setattr(
        InferenceScheduler,
        "_model_loaded_for_job",
        lambda self, job, loaded_models: job.model in loaded_models,
    )
    world = _ineligible_card_world()

    head, _latched_ticks = await _drive_ineligible_card_head(world)

    assert world.dispatch_tick(head) is None, (
        "a card-blind already-loaded gate must keep the head out of the preload pass entirely, which is the "
        f"whole of this defect; it sampled anyway. {world.state_dump()}"
    )
    assert world.job_tracker.jobs_pending_inference and not world.job_tracker.jobs_in_progress, (
        "the wedge idles every lane against a queue that still holds work; a run with a job sampling or an "
        f"empty queue is a slow schedule rather than this defect. {world.state_dump()}"
    )
    with pytest.raises(AssertionError, match="the card was idle for"):
        assert_never_idle_with_fitting_work(world, context="ineligible-card residency")


# --------------------------------------------------------------------------------------------------------
# Per-card safety placement: a peak that can only land on a sibling card is not this card's pressure
# --------------------------------------------------------------------------------------------------------

_TWO_CARD_PIXELS = {0: 4_194_304, 1: 1_048_576}
"""Per-card resolution ceilings for the two-card pool: the heavy card takes anything these rows generate, and
the card hosting safety is capped at a megapixel. Only the resolution axis differs, so which card may serve a
job is decided by its pixels alone, and the heavy class can only ever land on card 0."""

_TWO_CARD_SAFETY_CARD = 1
"""The card the safety process is pinned to: the one whose own work is light, so its measured evidence says it
is serving comfortably while its sibling carries the peak the worker cannot give it."""

_TWO_CARD_HEAVY_SHAPE = (1280, 1280)
_TWO_CARD_HEAVY_BATCH = 2
"""The heavy class: a hires batch four times the safety card's ceiling, so it is only ever eligible on card 0.
Its predicted sampling peak is around 6.1 GB, well past anything the safety card could absorb."""

_TWO_CARD_LIGHT_SHAPE = (1024, 1024)
_TWO_CARD_LIGHT_STEPS = 60
"""The light class: a megapixel job at the safety card's exact ceiling, with enough steps that its sampling
window spans several ticks and the card it runs on earns a real duty figure. Its peak is around 4.1 GB, which
the card already holds the weights and the context for."""

_TWO_CARD_PENDING_PER_CLASS = 2
"""Jobs of each class the queue is kept holding.

One is not enough: a class with a single job in flight leaves the moment between that job's completion and the
next pop with nothing of its own anywhere, and the peak a card is committed to then reads as absent for a tick.
Real traffic keeps a queue, and what this scenario is about is what a card is committed to while it has one."""

_TWO_CARD_TICKS = 300
"""Ticks these rows run for: ten simulated minutes at this module's tick, twenty times the dwell a placement
demotion has to sustain, so a policy that arms itself once has the whole run to act on it."""

_TWO_CARD_DUTY_FLOOR = 0.30
"""Fraction of its own slot-time each card must spend sampling.

The positive half of the placement verdict: a pool that keeps safety on the card by never dispatching would
satisfy every memory claim here. Each card serves one class with its weights already resident, so the ceiling
is the share of a job that is neither load nor decode, and this is a comfortable fraction of it."""


def _two_card_safety_world() -> _DispatchWorld:
    """Two independent 10 GB cards, one inference lane each, safety pinned to the lighter card's ledger.

    Both cards carry a resident checkpoint of their own and serve their own class of traffic; safety sits on
    card 1 with that card's context, its resident weights and its sampling activation beside it. Whole-card
    residency is off, so nothing but the placement policy and the reclaim ladder can move safety, and the
    ladder only runs on a saturated card.

    Ten gigabytes rather than eight because eight cannot express the regime. The card hosting safety carries
    the one-time runtime context (1354 MB), safety's whole-process charge (3044 MB) and a 3.2 GB checkpoint
    before any job runs: 7598 MB, leaving 594 MB of an 8192 MB card against a 1024 MB PRESSURE soft floor. Such
    a card is off HEALTHY with nothing sampling on it, and demoting its safety process is then measured
    evidence acted on correctly rather than the misattribution this scenario is about. Ten gigabytes is the
    smallest class that fits the light class's activation (944 MB) and a healthy margin on top of those
    tenants.
    """
    world = _DispatchWorld(
        card=_CARD_10GB,
        lane_count=2,
        max_threads=2,
        queue_depth=_QUEUE_DEPTH,
        whole_card_enabled=False,
        closed_loop=True,
        tick_seconds=_TICK_SECONDS,
        service_contexts=True,
        safety_readiness_seconds=_SAFETY_READINESS_SECONDS,
        safety_load_transient_mb=_SAFETY_LOAD_TRANSIENT_MB,
        card_max_pixels=_TWO_CARD_PIXELS,
        safety_card_index=_TWO_CARD_SAFETY_CARD,
    )
    world.seed_resident(0, _SD15_OTHER, in_vram=True)
    world.seed_resident(1, _SD15, in_vram=True)
    assert world.card_of_lane(0) == 0 and world.card_of_lane(1) == _TWO_CARD_SAFETY_CARD, (
        "precondition: each lane must sit on its own card, so each card's ledger carries one resident"
    )
    return world


async def _drive_two_card_traffic(world: _DispatchWorld) -> list[int]:
    """Keep both classes queued for the whole run and record when safety's card read as pressured.

    The pressure predicate is sampled once per tick rather than inferred from whether a flip happened, so the
    verdict does not depend on the demotion dwell: a card that never reads pressured cannot arm a demotion
    however long the run is, and a card that does is a card the policy is entitled to act on.

    Returns:
        The ticks on which safety's own card read as pressured.
    """
    heavy_width, heavy_height = _TWO_CARD_HEAVY_SHAPE
    light_width, light_height = _TWO_CARD_LIGHT_SHAPE
    pressured_ticks: list[int] = []
    for _ in range(_TWO_CARD_TICKS):
        pending = list(world.job_tracker.jobs_pending_inference)
        heavy_pending = sum(1 for job in pending if job.payload.width > light_width)
        if heavy_pending < _TWO_CARD_PENDING_PER_CLASS:
            await world.pop(
                make_job_pop_response(
                    _SD15_OTHER.name,
                    width=heavy_width,
                    height=heavy_height,
                    n_iter=_TWO_CARD_HEAVY_BATCH,
                    ddim_steps=20,
                ),
            )
        if len(pending) - heavy_pending < _TWO_CARD_PENDING_PER_CLASS:
            await world.pop(
                make_job_pop_response(
                    _SD15.name,
                    width=light_width,
                    height=light_height,
                    ddim_steps=_TWO_CARD_LIGHT_STEPS,
                ),
            )
        await world.step()
        if world.scheduler._safety_placement_card_is_pressured(_TWO_CARD_SAFETY_CARD):
            pressured_ticks.append(world.tick)
    return pressured_ticks


def _safety_placement_pauses(world: _DispatchWorld) -> list[tuple[int, float]]:
    """Every time the runtime placement policy itself took safety off the card, as (tick, world clock)."""
    return [
        (tick, when) for tick, when, owner in world.safety_pause_events if owner is PauseOwner.RUNTIME_SAFETY_PLACEMENT
    ]


def _price_the_peak_worker_wide(
    self: InferenceScheduler,
    device_index: int | None,
) -> list[ImageGenerateJobPopResponse]:
    """The pre-fix sampling-peak set: every job the worker holds, whatever card could run it.

    Card-blind exactly as it was, so the heaviest peak any queued or running job carries is charged against
    every card and a card is held responsible for demand its own config forbids it from being given.
    """
    del device_index
    return [*self._job_tracker.jobs_in_progress, *self._job_tracker.jobs_pending_inference]


async def test_q_safety_stays_on_a_card_committed_only_to_the_work_it_serves() -> None:
    """A card serving its own light work keeps its safety process while a sibling card carries the heavy peak.

    The failure this encodes: safety's placement was judged against a worker-wide sampling peak. On a
    multi-card pool the heaviest queued job is routinely one this card can never be given (its effective
    config excludes the resolution outright), and the card holds no weights for that job's model either, so the
    whole of that peak read as memory this card still had to find. Measured free never covered it, the demotion
    dwell was met within seconds of the first heavy job, and the safety process was ended on a card that was
    serving its own traffic comfortably. The restore forecast then asked for the same phantom peak plus
    safety's footprint, which no card in the pool could ever show, so the eviction was permanent and every
    result the worker produced afterwards was cleared on the CPU.

    Read as one statement: cards are independent memory domains, so a peak that can only land on a sibling card
    is not this card's pressure. Its consequences are that the placement policy never demotes safety, that
    neither card's governor leaves HEALTHY and no reclaim rung is spent, that each card keeps the resident it
    was serving from, and that both cards go on earning while all of that holds.
    """
    world = _two_card_safety_world()

    pressured_ticks = await _drive_two_card_traffic(world)

    context = "per-card safety placement"
    safety_card = world.safety_card_index()
    assert not pressured_ticks, (
        f"{context}: card {safety_card} read as short of memory on tick(s) {pressured_ticks[:6]} while it was "
        f"serving its own work with its weights already resident, so a demotion was armed against a card that "
        f"has nothing left to find. {world.state_dump()}"
    )
    assert not _safety_placement_pauses(world), (
        f"{context}: the placement policy took safety off card {safety_card} at "
        f"{_safety_placement_pauses(world)}, which costs the pipeline the rebuild twice and leaves every later "
        f"result to be cleared on the CPU. {world.state_dump()}"
    )
    assert not world.ladder_actuations and world.reclaim_commands == 0, (
        f"{context}: the worker spent {len(world.ladder_actuations)} reclaim rung(s) and "
        f"{world.reclaim_commands} unload command(s) on a pool where neither card ever left its floors, so it "
        f"took its own capacity down against nothing. {world.state_dump()}"
    )
    assert_governor_never_reached(world, GovernorState.PRESSURE, context=context)
    for device_index in world.card_indices():
        expected = _SD15_OTHER.name if device_index == 0 else _SD15.name
        assert list(world.card_resident_models(device_index).values()) == [expected], (
            f"{context}: card {device_index} ended holding {world.card_resident_models(device_index)} "
            f"({world.card_resident_mb(device_index)}) rather than the {expected} copy it was serving from, so "
            f"the run stopped being two cards each serving one class off its own resident weights. "
            f"{world.state_dump()}"
        )
        assert_duty_floor_on_card(world, device_index, _TWO_CARD_DUTY_FLOOR, context=context)
    heaviest_pool_wide = world.scheduler._largest_active_sampling_peak(None)
    heaviest_on_safety_card = world.scheduler._largest_active_sampling_peak(safety_card)
    assert heaviest_pool_wide is not None and heaviest_on_safety_card is not None, (
        f"{context}: the pool was committed to no sampling peak at all at the end of the run, so nothing here "
        f"was priced against a peak. {world.state_dump()}"
    )
    assert heaviest_pool_wide[0] > heaviest_on_safety_card[0], (
        f"{context}: the pool's heaviest peak ({heaviest_pool_wide}) was no heavier than what card "
        f"{safety_card} is committed to ({heaviest_on_safety_card}), so this run would pass with the peak "
        f"priced worker-wide and says nothing about per-card attribution. {world.state_dump()}"
    )


async def test_q_defect_reinjection_a_worker_wide_peak_evicts_safety_from_a_healthy_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the peak priced worker-wide again, a sibling's demand permanently evicts safety from a serving card.

    Reinjected at the attribution alone: the routing, the ledgers, the dwell and the restore forecast are
    untouched, and the only difference is that the heaviest peak any job carries is charged to every card. The
    card hosting safety is then judged unable to find memory it was never going to be asked for, and because
    the restore forecast reads the same phantom, what comes back is the pre-fix signature in full: one demotion
    the worker never undoes.
    """
    monkeypatch.setattr(InferenceScheduler, "_sampling_peak_jobs_for_card", _price_the_peak_worker_wide)
    world = _two_card_safety_world()

    pressured_ticks = await _drive_two_card_traffic(world)

    assert pressured_ticks, (
        "a worker-wide peak must make safety's card read as pressured, which is the whole of this defect; the "
        f"card read comfortable throughout. {world.state_dump()}"
    )
    pauses = _safety_placement_pauses(world)
    assert pauses, (
        f"the placement policy never acted on {len(pressured_ticks)} pressured tick(s), so the eviction the "
        f"scenario forbids would not have happened on a tree that prices the peak worker-wide. "
        f"{world.state_dump()}"
    )
    assert not world.safety_restore_events, (
        f"safety came back at {world.safety_restore_events}, so this defect is a cycle rather than the "
        f"permanent eviction the restore forecast's own phantom peak produces. {world.state_dump()}"
    )
