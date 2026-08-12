"""VRAM retention measured across the traffic signatures operators actually run.

Retention is a bet that a same-model successor will arrive on the same slot. Whether that bet pays depends
entirely on the shape of the traffic, and an operator's offer is theirs to choose: a slot locked to one model
repeats on every dispatch, while a slot serving a wide rotation may never repeat inside any window worth
holding weights for. A policy that grants on card health alone therefore reads identically in both cases and is
right in only one of them: on a diverse offer the holds accumulate, price each other's static fits, and are
handed back through the reclaim ladder having saved nothing, which is pressure the worker created for itself.

Each scenario here drives the real scheduler, governor and reclaim ladder over a conserved card
(:mod:`tests.process_management.liveness._dispatch_world`) for one signature, and asserts the outcome that
signature must produce:

- **Pool-locked.** One model throughout. Retention must be granted as freely as an ungated policy would grant
  it, so the run pays for its weights once rather than once per job. The only permitted cost is the warmup: the
  first dispatch on a slot has no history behind it and is refused, which is the price of granting on evidence
  rather than on assumption, and it amortizes to nothing over a streak.
- **Diverse.** A rotation wide enough that no model recurs within the evidence window on any slot. Retention
  must earn close to nothing here, and the run must not manufacture pressure out of holds nothing came back
  for. Its reinjection is the ungated policy, which brings the accumulated unused holds straight back.
- **Heterogeneous.** A strongly bimodal mix on a shared pool: one model dominates the traffic while the rest
  rotate. The holds must concentrate on the repeating model rather than being spread across the rotation. This
  is the aggregate form of the mixed-pool signature; the world routes jobs by its own placement rules and a
  test cannot pin one to a lane, so what is asserted is where the residency lands, not which lane it lands on.
- **Sustained pressure.** A card that stays off HEALTHY while one lane holds weights nothing has come back for
  and another holds weights its traffic has just used. The dead hold must be given up and the live one kept.
  What separates them is time, not the history that issued them: a live grant's own dispatch heads that
  history, so re-asking the issuance question passes for every retention there is.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from horde_worker_regen.process_management.scheduling import inference_scheduler as inference_scheduler_module
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _RETENTION_REPEAT_EVIDENCE_DISPATCHES,
    InferenceScheduler,
    RetentionDenialReason,
)
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import (
    _CARD_16GB,
    _CARD_24GB,
    _FLUX,
    _SD15,
    _SD15_OTHER,
    _SDXL,
    _SDXL_OTHER,
    _CardClass,
    _DispatchWorld,
    _ModelClass,
)
from tests.process_management.liveness._world_assertions import assert_governor_never_reached

_TICK_SECONDS = 2.0
"""Seconds of simulated time per scheduling tick, short enough that a job's load, sample and decode phases are
each sampled several times rather than stepped over in one advance."""

_QUEUE_DEPTH = 4
"""Jobs the worker is kept holding, so a successor is always available while a retainer is still busy."""

_SETTLE_TICKS = 10
"""Ticks driven past the last completion so a run's readback describes a settled worker."""

_MAX_TICKS = 500
"""Ceiling on a scenario's ticks, so a workload that stops moving ends the test instead of hanging."""

_SIGNATURE_JOBS = 18
"""Jobs per signature: long enough that a per-job re-upload dominates the run and that a rotation completes
several full cycles, so what is measured is the signature rather than its first few dispatches."""

_LANE_COUNT = 2
"""Lanes in the pool every signature runs on."""

_DIVERSE_ROTATION: tuple[_ModelClass, ...] = (_SD15, _SDXL, _SD15_OTHER, _SDXL_OTHER, _FLUX)
"""A five-model rotation, which is what makes the diverse signature diverse *per slot* rather than merely in
aggregate. With an even lane count a slot sees every other job, so an even-length rotation would hand each lane
a two-model alternation that repeats inside the evidence window and reads as pool-locked traffic. An odd-length
rotation walks the whole set past every lane, which is the property the signature needs."""

_DOMINANT_SHARE = 5
"""Jobs of the dominant model per job of the rotating remainder in the heterogeneous signature.

A pool-locked slot beside a diverse one produces a worker whose traffic is mostly one model with a minority
that keeps moving; this is that mix expressed as a job stream, since placement is the scheduler's to decide."""


def _signature_world(*, card: _CardClass = _CARD_16GB, max_threads: int = 2) -> _DispatchWorld:
    """Build the closed-loop world a signature scenario runs on: a two-lane pool over one conserved card."""
    return _DispatchWorld(
        card=card,
        lane_count=_LANE_COUNT,
        max_threads=max_threads,
        queue_depth=_QUEUE_DEPTH,
        whole_card_enabled=True,
        tick_seconds=_TICK_SECONDS,
        closed_loop=True,
    )


async def _drive_signature(
    world: _DispatchWorld,
    models: Sequence[_ModelClass],
    *,
    job_count: int = _SIGNATURE_JOBS,
    width: int = 1024,
    height: int = 1024,
    watch_retention: list[str] | None = None,
) -> list[ImageGenerateJobPopResponse]:
    """Feed ``world`` jobs cycling through ``models``, keeping its queue full, and run until the work settles.

    Continuing demand rather than a fixed queue: the worker holds up to its queue depth and pops a successor as
    soon as one drains, so a retainer is still finishing its own job when the job that might reuse its weights
    becomes available. That overlap is the condition retention exists for.

    ``watch_retention`` collects the model of every slot recorded as holding weights at the end of each tick, so
    a scenario can say where residency landed over the run rather than only what survived to the end of it.
    """
    jobs: list[ImageGenerateJobPopResponse] = []
    settling = 0
    for _ in range(_MAX_TICKS):
        while len(world.job_tracker.jobs_pending_inference) < _QUEUE_DEPTH and len(jobs) < job_count:
            model = models[len(jobs) % len(models)]
            job = make_job_pop_response(model.name, width=width, height=height, ddim_steps=30)
            await world.pop(job)
            jobs.append(job)
        await world.step()
        if watch_retention is not None:
            watch_retention.extend(world.retained_residents().values())
        if world.completed_jobs >= job_count:
            settling += 1
            if settling >= _SETTLE_TICKS:
                break
    return jobs


def _model_class_named(name: str) -> _ModelClass:
    """The model class the world prices ``name`` from, so a scenario can re-drive one lane's own model."""
    return next(model for model in (_SD15, _SD15_OTHER, _SDXL, _SDXL_OTHER, _FLUX) if model.name == name)


def _denials(world: _DispatchWorld, reason: RetentionDenialReason) -> int:
    """How many retention grants the named gate refused over the run."""
    return world.scheduler.retention_grant_denials.get(reason, 0)


def _assert_signature_drained(
    world: _DispatchWorld,
    jobs: list[ImageGenerateJobPopResponse],
    *,
    context: str,
) -> None:
    """Assert every job reached sampling and completed, so the memory verdicts describe a worker that served."""
    unsampled = [str(job.id_)[:8] for job in jobs if world.dispatch_tick(job) is None]
    assert not unsampled, f"{context}: {len(unsampled)} job(s) never reached sampling {unsampled[:4]}"
    assert world.completed_jobs == len(jobs), (
        f"{context}: {world.completed_jobs} of {len(jobs)} jobs completed. {world.state_dump()}"
    )


def _slot_repeats_within_window(world: _DispatchWorld) -> list[str]:
    """Every dispatch whose model had already run on that same lane inside the evidence window.

    Reads the run's own dispatch series, which is what makes a diverse signature a measured property of the run
    rather than an assumption about how the placement order happened to seat it. A signature with any of these
    is not diverse per slot, whatever its rotation looks like in aggregate.
    """
    seen_per_lane: dict[int, list[str]] = collections.defaultdict(list)
    repeats: list[str] = []
    for model, lane in world.dispatch_lanes:
        window = seen_per_lane[lane][-_RETENTION_REPEAT_EVIDENCE_DISPATCHES:]
        if model in window:
            repeats.append(f"lane {lane}: {model} recurred within {window}")
        seen_per_lane[lane].append(model)
    return repeats


# --------------------------------------------------------------------------------------------------------
# Pool-locked: the signature retention exists for must lose nothing to the evidence gate
# --------------------------------------------------------------------------------------------------------


async def test_a_pool_locked_traffic_is_granted_retention_after_its_warmup() -> None:
    """A slot serving one model earns retention on every dispatch past the first and uploads its weights once.

    This is the binding constraint on gating retention at all: an operator who locks a pool to one model is
    running exactly the traffic retention was built for, and a gate that costs them anything beyond the first
    job boundary has taken away the feature rather than aimed it. The evidence a grant needs is what the slot
    has already been asked to run, so a pool-locked slot supplies it from its second dispatch onward and is
    refused only while it has no history at all.

    Read as one positive statement and its consequences: the run pays for its weights a bounded number of
    times rather than once per job, nearly every dispatch lands on weights already on the card, and the
    lack-of-evidence refusals stop once each lane has run once.
    """
    world = _signature_world()

    jobs = await _drive_signature(world, [_SDXL])

    context = "pool-locked signature"
    _assert_signature_drained(world, jobs, context=context)
    scheduler = world.scheduler
    assert _denials(world, RetentionDenialReason.NO_REPEAT_EVIDENCE) <= _LANE_COUNT, (
        f"{context}: {_denials(world, RetentionDenialReason.NO_REPEAT_EVIDENCE)} dispatch(es) were refused for "
        f"lack of repeat evidence, more than the one warmup dispatch per lane a pool-locked slot may pay. "
        f"{world.state_dump()}"
    )
    assert world.weight_uploads <= _LANE_COUNT, (
        f"{context}: the run paid {world.weight_uploads} weight uploads for {len(jobs)} jobs, so it is not "
        f"being served from a retained copy. {world.state_dump()}"
    )
    assert scheduler.retention_reuses >= len(jobs) - _LANE_COUNT, (
        f"{context}: only {scheduler.retention_reuses} of {len(jobs)} dispatches landed on a retained copy, so "
        f"the grants issued are not the ones the traffic comes back for. {world.state_dump()}"
    )
    assert scheduler.retention_grants_issued > 0, (
        f"{context}: no retention was granted at all over the run. {world.state_dump()}"
    )


async def test_a_defect_reinjection_an_unsatisfiable_gate_reuploads_every_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the evidence window emptied, the pool-locked run pays a weight upload per job again.

    Reinjected at the window alone: a gate that can never find evidence is the shape a mis-aimed evidence
    requirement takes, and it costs a pool-locked operator the whole of the feature. It is what makes the
    scenario above a measurement of the warmup bound rather than a restatement of whatever the gate does.
    """
    monkeypatch.setattr(inference_scheduler_module, "_RETENTION_REPEAT_EVIDENCE_DISPATCHES", 0)
    world = _signature_world()

    jobs = await _drive_signature(world, [_SDXL])

    _assert_signature_drained(world, jobs, context="unsatisfiable-gate pool-locked signature")
    assert world.weight_uploads == len(jobs), (
        "with the evidence window emptied a pool-locked streak must pay one weight upload per job, which is "
        f"the cost the gate must not impose on it; it paid {world.weight_uploads} for {len(jobs)} jobs"
    )
    assert world.scheduler.retention_grants_issued == 0, (
        "an unsatisfiable evidence window must grant nothing, so the warmup bound the scenario asserts would "
        "pass on a tree whose gate never fires"
    )


# --------------------------------------------------------------------------------------------------------
# Diverse: the signature where an ungated grant is pure cost
# --------------------------------------------------------------------------------------------------------


async def test_b_diverse_traffic_earns_almost_no_retention_and_makes_no_pressure() -> None:
    """A rotation that never repeats within the window on a slot is granted nothing and holds nothing.

    The failure this encodes: retention granted on card health alone treats a wide offer exactly as it treats a
    locked pool. Every dispatch leaves weights behind, the next dispatch for a different model is priced
    against them, and they are handed back unused: the card accumulates copies nothing will come back for and
    the worker manufactures the pressure its own reclaim ladder then has to resolve. Over hours of a diverse
    mix that is a recurring saturation the traffic never called for.

    Read as one positive statement and its consequences: the signature is diverse *per slot* (asserted of the
    run's own dispatch series, not assumed), the refusals are for lack of repeat evidence rather than for a
    card that would not carry the weights, nothing is left holding the card, no copy is evicted unused because
    none was taken, and the card never reaches the band where growth is held.
    """
    world = _signature_world(card=_CARD_24GB)

    jobs = await _drive_signature(world, _DIVERSE_ROTATION, width=512, height=512)

    context = "diverse signature"
    _assert_signature_drained(world, jobs, context=context)
    repeats = _slot_repeats_within_window(world)
    assert not repeats, (
        f"{context}: the rotation repeated a model on a lane inside the evidence window, so this run is not the "
        "diverse signature it asserts about:\n    " + "\n    ".join(repeats[:4])
    )
    scheduler = world.scheduler
    assert scheduler.retention_grants_issued == 0, (
        f"{context}: {scheduler.retention_grants_issued} grant(s) were issued on traffic that never repeats a "
        f"model on a slot, so each one is a copy nothing can come back for. {world.state_dump()}"
    )
    assert _denials(world, RetentionDenialReason.NO_REPEAT_EVIDENCE) > 0, (
        f"{context}: nothing was refused for lack of repeat evidence, so the run never reached the gate it is "
        f"about. {world.state_dump()}"
    )
    assert scheduler.retention_evicted_unused == 0, (
        f"{context}: {scheduler.retention_evicted_unused} retained copies were given back unused, which on this "
        f"signature is the whole of what the gate exists to prevent. {world.state_dump()}"
    )
    assert world.retained_residents() == {}, (
        f"{context}: the run ended with {world.retained_residents()} still recorded as held. {world.state_dump()}"
    )
    assert_governor_never_reached(world, GovernorState.PRESSURE, context=context)


async def test_b_defect_reinjection_an_ungated_grant_brings_the_unused_holds_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the evidence gate removed, the same diverse run accumulates holds it never comes back for.

    Reinjected at the gate alone: the card, the rotation, the governor and the ladder are untouched, and the
    only difference is that a grant no longer has to be predicted by the slot's own traffic. What returns is
    the regime the scenario above forbids, counted where it costs: copies taken and given back with no dispatch
    having reused them.
    """
    monkeypatch.setattr(
        InferenceScheduler,
        "_slot_has_repeat_evidence",
        lambda self, process_id, model, *, exclude_latest=False: True,
    )
    world = _signature_world(card=_CARD_24GB)

    jobs = await _drive_signature(world, _DIVERSE_ROTATION, width=512, height=512)

    _assert_signature_drained(world, jobs, context="ungated diverse signature")
    scheduler = world.scheduler
    assert scheduler.retention_grants_issued > 0, (
        "an ungated policy must grant on this rotation, since that is the behaviour being reinjected; it "
        f"granted nothing. {world.state_dump()}"
    )
    assert scheduler.retention_evicted_unused > 0, (
        "an ungated policy on traffic that never repeats must leave copies to be given back unused, which is "
        "the cost the gated scenario asserts away; none were recorded, so that assertion would pass on a tree "
        f"with no gate at all. {world.state_dump()}"
    )


# --------------------------------------------------------------------------------------------------------
# Heterogeneous: holds follow the repeating model, not the rotation around it
# --------------------------------------------------------------------------------------------------------


async def test_c_a_bimodal_mix_concentrates_retention_on_the_repeating_model() -> None:
    """On a mix of one dominant model and a moving remainder, the card holds the dominant model.

    The mixed-pool posture, expressed as traffic: an operator whose offer is mostly one model with a minority
    that keeps changing must get retention for the part that repeats and pay nothing for the part that does
    not. Because placement is the scheduler's decision and a test cannot pin a job to a lane, what is asserted
    is where the residency lands rather than which lane holds it, which is the aggregate form of the same
    claim.

    The remainder is what makes this more than the pool-locked scenario again: an evidence rule that simply
    tracked the worker's busiest model would hold the dominant one here too, while granting the rotation
    whatever the slot happened to be doing. The share below is what separates the two.
    """
    world = _signature_world()
    stream: list[_ModelClass] = [*([_SDXL] * _DOMINANT_SHARE), _SD15]
    stream += [*([_SDXL] * _DOMINANT_SHARE), _SD15_OTHER]
    stream += [*([_SDXL] * _DOMINANT_SHARE), _SDXL_OTHER]
    held: list[str] = []

    jobs = await _drive_signature(world, stream, watch_retention=held)

    context = "bimodal signature"
    _assert_signature_drained(world, jobs, context=context)
    assert held, f"{context}: nothing was ever held on the card over the run. {world.state_dump()}"
    dominant_holds = sum(1 for model in held if model == _SDXL.name)
    assert dominant_holds / len(held) >= 0.9, (
        f"{context}: only {dominant_holds} of {len(held)} recorded residencies were the repeating model "
        f"({collections.Counter(held).most_common()}), so the holds are following the rotation rather than the "
        f"traffic that comes back for them. {world.state_dump()}"
    )
    assert world.scheduler.retention_reuses > 0, (
        f"{context}: no dispatch landed on a retained copy, so the holds concentrated above were never used. "
        f"{world.state_dump()}"
    )


# --------------------------------------------------------------------------------------------------------
# Sustained pressure: a hold nothing came back for is given up, one the traffic still uses is not
# --------------------------------------------------------------------------------------------------------


async def _hold_pressure(world: _DispatchWorld, seconds: float) -> None:
    """Keep the card's committed governor state at PRESSURE for ``seconds`` of the world's clock.

    Pushed through the same seam the parent's control loop pushes it through, one push per scheduling
    interval, so the scheduler sees a card that has stayed off HEALTHY rather than a single sample. The
    world's own governor would derive HEALTHY from a card this empty, and what the scenario is about is what
    the worker does about its own retained copies while a card stays under pressure, not how it got there.
    """
    elapsed = 0.0
    while elapsed < seconds:
        world.now += world.tick_seconds
        elapsed += world.tick_seconds
        world.scheduler.run_governance_tick()
        world.scheduler.set_governor_state(0, GovernorState.PRESSURE)


async def test_d_sustained_pressure_revokes_the_hold_nothing_came_back_for() -> None:
    """A card under lasting pressure gives up a retained copy that has gone unused, and keeps one still in use.

    The failure this encodes: a grant is a prediction, and before this nothing ever revisited one. Only an
    eviction actuation could end a retention, so a hold taken in a healthy moment outlived every change in what
    the slot was being asked to run, and a card under pressure carried copies whose successors were never
    coming while the reclaim ladder worked around them.

    What re-opens the question is not the evidence that issued the grant: a live grant's own dispatch heads the
    slot's history, so that test passes for every live retention by construction. It is the prediction being
    falsified by time. A hold no job has come back for within the horizon is a bet that has lost, and on a card
    that has stayed off HEALTHY it is the cheapest thing there to give back.

    Read as one positive statement and its consequences: the aged hold is revoked once the pressure has held
    for its debounce, the hold whose traffic is still arriving is left alone on the same card at the same
    moment, and the revocation goes out through the ordinary idle-model unload rather than a second eviction
    mechanism of its own.
    """
    world = _signature_world()

    await _drive_signature(world, [_SDXL, _SD15], job_count=8, width=512, height=512)
    retained = world.retained_residents()
    assert len(retained) == 2, (
        f"stale-hold revoke: the run left {retained} held, but the scenario needs two retaining lanes to say "
        f"that one is revoked and the other is not. {world.state_dump()}"
    )

    # Age both holds past the horizon with no work arriving, then let one lane's traffic come back for its
    # weights: the reuse ends that episode and starts a fresh one, which is the whole of what distinguishes
    # the two lanes when the sweep runs.
    world.now += inference_scheduler_module._RETENTION_STALE_HOLD_SECONDS + world.tick_seconds
    reused_model = sorted(retained.items())[0][1]
    await _drive_signature(world, [_model_class_named(reused_model)], job_count=1, width=512, height=512)
    still_retained = world.retained_residents()
    reused_lane = next(lane for lane, model in still_retained.items() if model == reused_model)
    stale_lane = next(lane for lane, model in still_retained.items() if model != reused_model)

    await _hold_pressure(world, inference_scheduler_module._RETENTION_PRESSURE_REVOKE_SECONDS + world.tick_seconds)

    context = "stale-hold revoke"
    scheduler = world.scheduler
    assert scheduler.retention_revokes == 1, (
        f"{context}: {scheduler.retention_revokes} copies were revoked, not the one aged hold. {world.state_dump()}"
    )
    assert world.retained_residents().get(stale_lane) is None, (
        f"{context}: lane {stale_lane} still holds weights no job came back for through the whole pressure "
        f"window. {world.state_dump()}"
    )
    assert world.retained_residents().get(reused_lane) == reused_model, (
        f"{context}: lane {reused_lane}'s copy was revoked though its traffic had just used it, so the sweep "
        f"is acting on pressure rather than on a falsified prediction. {world.state_dump()}"
    )


async def test_d_defect_reinjection_without_the_horizon_the_dead_hold_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the falsification horizon put out of reach, the unused copy holds the pressured card indefinitely.

    Reinjected at the horizon alone: the debounce, the sweep and the actuation are untouched, and the only
    difference is that a hold can never be old enough to have lost its bet. What comes back is the regime the
    scenario above ends, a retention that no evidence and no elapsed time can dislodge, which is what makes
    that scenario a measurement of the horizon rather than of the sweep merely running.
    """
    monkeypatch.setattr(inference_scheduler_module, "_RETENTION_STALE_HOLD_SECONDS", 1_000_000.0)
    world = _signature_world()

    await _drive_signature(world, [_SDXL, _SD15], job_count=8, width=512, height=512)
    held_before = world.retained_residents()
    assert held_before, f"the run retained nothing, so there is no hold to survive. {world.state_dump()}"

    world.now += 60.0 * 60.0
    await _hold_pressure(world, inference_scheduler_module._RETENTION_PRESSURE_REVOKE_SECONDS + world.tick_seconds)

    assert world.scheduler.retention_revokes == 0, (
        "with the horizon out of reach the sweep must revoke nothing, so the scenario's revoke assertion would "
        f"pass on a tree with no horizon at all. {world.state_dump()}"
    )
    assert world.retained_residents() == held_before, (
        "an hour of unused residency under sustained pressure left the card's holds untouched, which is the "
        f"state the horizon exists to end. {world.state_dump()}"
    )
