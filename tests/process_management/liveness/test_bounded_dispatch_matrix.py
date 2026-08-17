"""Bounded dispatch of a physically servable head, as a property over the whole scheduling loop.

A worker is wedged when it holds work it could serve and stops progressing it. The admission matrix in
``tests/process_management/resources/test_admission_liveness_matrix.py`` proves that property at the arbiter
surface, over a card model it prices directly. This module lifts the same property to the pipeline: a real
:class:`InferenceScheduler` driven across scheduling ticks against a card whose free VRAM is derived from what
is actually resident on it, so preload admission, whole-card residency, the churn governors, reclaim, and the
dispatch gate all compose the way they do in the running worker.

The property, stated once:

    For any head of queue that is physically servable (its weights fit the card outright, fit once the reclaim
    it is entitled to has run, or fit co-resident), the head reaches dispatch within a bounded number of
    scheduling ticks, and every obligation opened on the way there is closed on the way out.

"Obligation" is concrete: a planned reserve-ledger charge booked for an admitted preload, a whole-card
residency claim over the card, and the safety / post-processing GPU pauses a residency takes out. Each is
asserted released once the row's queue has drained, because a charge that outlives the work it was booked for
is exactly the double-count that wedges the next head.

Every assertion is a positive outcome read through real state: a job entering progress through the dispatch
path, a lane carrying the START_INFERENCE flag, a model named on the ceiling hold, a cycle-frozen measurement
carrying no planned charge. None of them can be satisfied by the failure they guard against.

Not every head is servable, and the table says which. A model whose sampling peak prices above the card's
achievable ceiling takes the worker's other disclosed exit: the model goes on the ceiling hold and its queued
job is faulted onward for reissue. Rows carrying that expectation are what stop the property from degenerating
into "everything asked of the worker is admitted".

Axes and their values are enumerated in ``_ROWS``; combinations deliberately left out are listed with their
reason in ``_PRUNED_COMBINATIONS`` so the table's coverage is never silently truncated.

Scope. This drives the scheduler's own loop (governance tick, preload pass, dispatch pass) with a hand-run
world standing in for the children: a preload materialises on the following tick, an unload command frees the
model's weights, a dispatched job completes on the following tick. It does not drive the process manager's
message pump, the recovery supervisor, or the pop path; those are covered by the manager and recovery suites.

The world itself is :mod:`tests.process_management.liveness._dispatch_world`, shared with the generated chaos
suite and the incident scenarios. These rows run it at its scheduling fidelity: no governor, no reclaim
ladder, and one tick of occupancy per dispatch, which is what makes a row's tick bound a statement about
admission ordering rather than about how long a job takes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.jobs.job_tracker import JobStage
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    _ESTABLISH_WINDOW_LIMIT,
    _GRACE_BUDGET_SECONDS,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    _WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
    _WHOLE_CARD_RESTORE_GRACE_SECONDS,
)
from tests.process_management.conftest import make_job_pop_response
from tests.process_management.liveness._dispatch_world import (
    _CARD_8GB,
    _CARD_16GB,
    _CARD_24GB,
    _FLUX,
    _MODEL_CLASSES,
    _SAME_CLASS_PARTNER,
    _SD15,
    _SD15_OTHER,
    _SDXL,
    _SHAPE_HIRES_BATCH,
    _SHAPE_SMALL,
    _TICK_SECONDS,
    _CardClass,
    _DispatchWorld,
    _JobShape,
    _ModelClass,
)
from tests.process_management.liveness._world_assertions import (
    assert_never_idle_with_fitting_work,
    assert_no_committed_slot_retired,
    assert_no_unservable_dispatch_hold,
)

# --------------------------------------------------------------------------------------------------------
# Axis values
# --------------------------------------------------------------------------------------------------------


class _Residency(Enum):
    """Where the head's checkpoint sits when the row starts."""

    ABSENT = "absent"
    """Nothing holds it; a lane must preload it."""
    RESIDENT_IDLE_TARGET = "resident_idle_target"
    """A lane already holds it in VRAM and is idle, so dispatching moves nothing."""
    RESIDENT_ON_SIBLING = "resident_on_sibling"
    """A lane holds it, but that lane is running another job, so the head waits for it or lands elsewhere."""
    RAM_STAGED = "ram_staged"
    """A lane has it staged in RAM (preloaded, not yet committed to VRAM)."""


class _QueueShape(Enum):
    """The structure of the queue the head sits at the front of."""

    SINGLE_HEAD = "single_head"
    HEAD_PLUS_SKIPPERS = "head_plus_skippers"
    """A heavy head followed by lighter jobs whose model is already resident (line-skip candidates)."""
    MULTI_MODEL_INTERLEAVE = "multi_model_interleave"
    """Alternating models, so residency must rotate to drain the queue."""
    SAME_MODEL_BURST = "same_model_burst"
    """Several jobs for the head's model, all riding one residency."""
    AT_DEPTH = "at_depth"
    """The queue at its configured depth, so no further pop room exists."""


class _GovernorState(Enum):
    """The whole-card churn governors' state when the row starts, driven through the real ledger."""

    FRESH = "fresh"
    GRACE_EXHAUSTED = "grace_exhausted"
    """The card's rolling grace allowance is spent, so a new establishment is refused for its dwell."""
    ESTABLISH_RATE_EXCEEDED = "rate_exceeded"
    """The per-card establishment allowance is spent, so a new establishment is refused for its window."""


class _MidSequenceEvent(Enum):
    """A disturbance injected part-way through the row's run."""

    NONE = "none"
    TARGET_DEATH_RESPAWN = "target_death"
    """The lane holding (or loading) the head's model dies and is replaced by a fresh empty lane."""
    EXTERNAL_RECLAIM = "external_reclaim"
    """An idle sibling's resident model is evicted by an actor other than this scheduler."""


class _ClaimScenario(Enum):
    """What a row asserts about the residency's claim over the worker's pop offer.

    A residency governs the card; the claim is the same commitment applied to intake, so that foreign work
    stops arriving to push the resident weights back out. Each value names one end of that arrangement: the
    burst it exists to serve, and the three ways it gives the pool back.
    """

    NONE = "none"
    """The row says nothing about intake; the claim is exercised only as far as the other properties reach."""
    SERVES_THE_BURST = "serves_the_burst"
    """Work for the resident model keeps arriving and is served, while foreign work is not asked for."""
    CAP_RETURNS_THE_POOL = "cap_returns_the_pool"
    """The maximum hold elapses over a still-wanted residency, returning the full offer and draining the
    foreign work the claim had been holding back."""
    EMPTY_POPS_RELEASE = "empty_pops_release"
    """The resident model's demand dries up, so the claim releases on that evidence well inside its cap."""
    FOREIGN_QUEUED_FIRST = "foreign_queued_first"
    """Foreign jobs accepted before the residency existed wait for the claim to end, then drain."""


class _Expected(Enum):
    """What the row asserts about the head."""

    DISPATCHED = "dispatched"
    """The head reaches sampling within the row's tick bound."""
    UNSERVABLE_HELD = "unservable_held"
    """The head's model prices above the card's achievable ceiling, so the worker holds the model off its
    offer and hands the queued job back for reissue rather than keeping it. The row asserts that disclosed,
    bounded exit: the model is on the ceiling hold and the job did not linger in the queue."""


_DEATH_TICK = 3
"""The tick a mid-sequence disturbance fires on: late enough that the row has committed to a plan, early
enough that the remaining ticks can still prove recovery within the bound."""


# --------------------------------------------------------------------------------------------------------
# The scenario table
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    """One point in the bounded-dispatch matrix.

    Attributes:
        label: The parametrized id.
        card: The device-free profile.
        head_model: The head-of-queue job's model class.
        residency: Where the head's checkpoint starts out.
        queue: The queue structure behind the head.
        governor: The whole-card churn governors' starting state.
        max_threads: The concurrent-sampling cap.
        lanes: How many inference lanes the pool holds.
        event: A disturbance injected part-way through the run.
        expected: What the row asserts about the head.
        tick_bound: The number of scheduling ticks the head must reach dispatch within.
        run_ticks: How many ticks to drive in total (at least the bound, plus drain ticks for the
            obligation-closure assertions).
        drain_all: Whether every queued job (not only the head) must drain within ``run_ticks``.
        claim: What the row asserts about the residency's claim over the pop offer.
        cooldown_seconds: How long a drained residency is retained for a follow-on heavy job.
        max_hold_seconds: The ceiling on one residency episode, which also bounds its claim.
        shape: The generation size the row's jobs ask for, which scales sampling activation.
        service_contexts: Whether safety sits on the card and the post-processing lane holds a context.
        disaggregated: Whether the row's jobs run as UNet-only samplers.
    """

    label: str
    card: _CardClass
    head_model: _ModelClass
    residency: _Residency
    queue: _QueueShape
    governor: _GovernorState = _GovernorState.FRESH
    max_threads: int = 1
    lanes: int = 2
    event: _MidSequenceEvent = _MidSequenceEvent.NONE
    expected: _Expected = _Expected.DISPATCHED
    tick_bound: int = 6
    run_ticks: int = 14
    drain_all: bool = True
    whole_card_enabled: bool = True
    claim: _ClaimScenario = _ClaimScenario.NONE
    cooldown_seconds: int = 0
    max_hold_seconds: int = 180
    shape: _JobShape = _SHAPE_SMALL
    service_contexts: bool = False
    disaggregated: bool = False


_PRUNED_COMBINATIONS: tuple[tuple[str, str], ...] = (
    (
        "8 GB card x SDXL or flux head, beyond one representative cell each",
        "neither model's sampling peak fits an 8 GB card however much is reclaimed, so every such cell takes "
        "the same ceiling-hold exit; one representative cell of each is driven (plus one with servable work "
        "queued behind it) and the rest are dropped as repeats of a single verdict.",
    ),
    (
        "whole-card governor states x non-EXTRA_LARGE head models",
        "the grace budget and the establishment rate limiter gate whole-card residency only; an SD15 or SDXL "
        "head never reaches that path, so charging its ledger varies nothing about that head's admission.",
    ),
    (
        "RESIDENT_ON_SIBLING x SINGLE_HEAD",
        "a sibling lane is only meaningful when a second job occupies it; with a single head the shape "
        "collapses onto RESIDENT_IDLE_TARGET.",
    ),
    (
        "TARGET_DEATH_RESPAWN x RESIDENT_ON_SIBLING",
        "the disturbance is defined against the lane holding the head's own copy; with the copy held by a busy "
        "sibling the kill is a live-job kill, which is the recovery suite's subject, not bounded dispatch.",
    ),
    (
        "max_threads=2 x SINGLE_HEAD",
        "a second sampling slot cannot change a one-job queue's outcome; the concurrency axis is varied only "
        "against queue shapes that hold more than one dispatchable job.",
    ),
    (
        "AT_DEPTH x max_threads=2 x every card class",
        "queue depth interacts with the pop gate, not with dispatch capacity; one representative at-depth cell "
        "per card class is driven and the concurrency cross-product is dropped as redundant.",
    ),
    (
        "EXTERNAL_RECLAIM x rows whose card is not under pressure",
        "an eviction that frees room nothing was waiting for varies nothing; the event is applied only where "
        "the head's admission actually turns on the freed VRAM.",
    ),
    (
        "pop-claim scenarios x every card, model, governor and event value",
        "the claim is a property of a held residency and of the clock, not of the card it is held on: the "
        "16 GB flux cell is the one where a residency is genuinely warranted, so the four claim scenarios are "
        "driven there and the cross-product with hardware that would either never establish a residency or "
        "never need one is dropped.",
    ),
    (
        "disaggregation and the high-resolution batch shape x every card, governor and event value",
        "both exist to put a moderate-weight model's activation far above its weights while the card's "
        "structural floor is squeezed by charges no inference teardown reclaims. That regime is the 16 GB "
        "card with the service contexts up; a roomier card has headroom the spike never reaches and an 8 GB "
        "card cannot serve the shape at all, so neither varies the verdict.",
    ),
    (
        "service contexts x whole-card governor states and mid-sequence events",
        "safety and the post-processing lane change what the structural floor is made of, not how a governor "
        "brakes an establishment or how a lane death is recovered; those axes are driven without them.",
    ),
    (
        "24 GB card x SD15 head x every governor and event value",
        "the roomy card with the smallest model admits on the first tick under every one of them; one "
        "representative cell is kept and the rest dropped as trivially-passing.",
    ),
)
"""Combinations enumerated by the axes but deliberately not driven, each with why. Listed so the table's
coverage is explicit: nothing is truncated silently."""


def _rows() -> tuple[_Row, ...]:
    """The driven matrix."""
    return (
        # -- Card class x model class, single head, nothing resident: the base servability sweep. ------------
        _Row("sd15_absent_8gb", _CARD_8GB, _SD15, _Residency.ABSENT, _QueueShape.SINGLE_HEAD),
        _Row("sd15_absent_16gb", _CARD_16GB, _SD15, _Residency.ABSENT, _QueueShape.SINGLE_HEAD),
        _Row("sd15_absent_24gb", _CARD_24GB, _SD15, _Residency.ABSENT, _QueueShape.SINGLE_HEAD),
        # SDXL on an 8 GB card is servable: its priced peak comes from the sampling estimate rather than an
        # unmeasured recommendation floor, and the modest remaining shortfall against the live reading is
        # decided by a measured attempt instead of a static decline. The card provably holds the model, so
        # the worker serves it.
        _Row(
            "sdxl_absent_8gb",
            _CARD_8GB,
            _SDXL,
            _Residency.ABSENT,
            _QueueShape.SINGLE_HEAD,
            tick_bound=8,
        ),
        _Row("sdxl_absent_16gb", _CARD_16GB, _SDXL, _Residency.ABSENT, _QueueShape.SINGLE_HEAD),
        _Row("sdxl_absent_24gb", _CARD_24GB, _SDXL, _Residency.ABSENT, _QueueShape.SINGLE_HEAD),
        _Row("flux_absent_16gb", _CARD_16GB, _FLUX, _Residency.ABSENT, _QueueShape.SINGLE_HEAD, tick_bound=8),
        _Row("flux_absent_24gb", _CARD_24GB, _FLUX, _Residency.ABSENT, _QueueShape.SINGLE_HEAD, tick_bound=8),
        # A head that cannot fit even an emptied card: the discriminating negative row.
        _Row(
            "flux_absent_8gb_unservable",
            _CARD_8GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SINGLE_HEAD,
            expected=_Expected.UNSERVABLE_HELD,
            drain_all=False,
            run_ticks=10,
        ),
        # -- Residency axis. ---------------------------------------------------------------------------------
        _Row("sd15_resident_idle_8gb", _CARD_8GB, _SD15, _Residency.RESIDENT_IDLE_TARGET, _QueueShape.SINGLE_HEAD),
        _Row("sdxl_resident_idle_16gb", _CARD_16GB, _SDXL, _Residency.RESIDENT_IDLE_TARGET, _QueueShape.SINGLE_HEAD),
        _Row("flux_resident_idle_24gb", _CARD_24GB, _FLUX, _Residency.RESIDENT_IDLE_TARGET, _QueueShape.SINGLE_HEAD),
        # A resident whole-card head on a card that demands exclusive residency. Its weights are already on
        # the device, so the residency's live-fit question is answered before it is asked and the head is not
        # left to the drain-settle backstop.
        _Row(
            "flux_resident_idle_16gb",
            _CARD_16GB,
            _FLUX,
            _Residency.RESIDENT_IDLE_TARGET,
            _QueueShape.SINGLE_HEAD,
            tick_bound=8,
        ),
        _Row("sdxl_ram_staged_16gb", _CARD_16GB, _SDXL, _Residency.RAM_STAGED, _QueueShape.SINGLE_HEAD),
        _Row("flux_ram_staged_24gb", _CARD_24GB, _FLUX, _Residency.RAM_STAGED, _QueueShape.SINGLE_HEAD),
        _Row(
            "sdxl_resident_on_busy_sibling_16gb",
            _CARD_16GB,
            _SDXL,
            _Residency.RESIDENT_ON_SIBLING,
            _QueueShape.SAME_MODEL_BURST,
            tick_bound=8,
        ),
        # -- Queue structure. --------------------------------------------------------------------------------
        _Row(
            "flux_head_plus_skippers_16gb",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.HEAD_PLUS_SKIPPERS,
            tick_bound=10,
            run_ticks=22,
        ),
        # An SDXL head with lighter work behind it on the smallest card that holds it: the head dispatches
        # via measured admission and the skippers drain with it.
        _Row(
            "sdxl_head_plus_skippers_8gb",
            _CARD_8GB,
            _SDXL,
            _Residency.ABSENT,
            _QueueShape.HEAD_PLUS_SKIPPERS,
            tick_bound=10,
            run_ticks=24,
        ),
        _Row(
            "sdxl_multi_model_interleave_16gb",
            _CARD_16GB,
            _SDXL,
            _Residency.ABSENT,
            _QueueShape.MULTI_MODEL_INTERLEAVE,
            tick_bound=8,
            run_ticks=26,
        ),
        _Row(
            "sd15_multi_model_interleave_8gb",
            _CARD_8GB,
            _SD15,
            _Residency.ABSENT,
            _QueueShape.MULTI_MODEL_INTERLEAVE,
            tick_bound=8,
            run_ticks=30,
        ),
        _Row(
            "flux_same_model_burst_24gb",
            _CARD_24GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SAME_MODEL_BURST,
            tick_bound=8,
            run_ticks=22,
        ),
        _Row(
            "flux_same_model_burst_16gb",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SAME_MODEL_BURST,
            tick_bound=8,
            run_ticks=22,
        ),
        _Row(
            "sdxl_queue_at_depth_16gb",
            _CARD_16GB,
            _SDXL,
            _Residency.ABSENT,
            _QueueShape.AT_DEPTH,
            tick_bound=8,
            run_ticks=24,
        ),
        _Row(
            "sd15_queue_at_depth_8gb",
            _CARD_8GB,
            _SD15,
            _Residency.ABSENT,
            _QueueShape.AT_DEPTH,
            tick_bound=8,
            run_ticks=26,
        ),
        # -- Concurrency. ------------------------------------------------------------------------------------
        _Row(
            "sdxl_interleave_two_threads_24gb",
            _CARD_24GB,
            _SDXL,
            _Residency.ABSENT,
            _QueueShape.MULTI_MODEL_INTERLEAVE,
            max_threads=2,
            lanes=3,
            tick_bound=8,
            run_ticks=24,
        ),
        _Row(
            "sd15_burst_two_threads_16gb",
            _CARD_16GB,
            _SD15,
            _Residency.ABSENT,
            _QueueShape.SAME_MODEL_BURST,
            max_threads=2,
            lanes=3,
            tick_bound=6,
            run_ticks=20,
        ),
        _Row(
            "flux_skippers_two_threads_24gb",
            _CARD_24GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.HEAD_PLUS_SKIPPERS,
            max_threads=2,
            lanes=3,
            tick_bound=10,
            run_ticks=24,
        ),
        # -- Governor / budget state (whole-card head only). -------------------------------------------------
        _Row(
            "flux_grace_exhausted_16gb",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SINGLE_HEAD,
            governor=_GovernorState.GRACE_EXHAUSTED,
            tick_bound=10,
            run_ticks=18,
        ),
        _Row(
            "flux_grace_exhausted_24gb",
            _CARD_24GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SINGLE_HEAD,
            governor=_GovernorState.GRACE_EXHAUSTED,
            tick_bound=10,
            run_ticks=18,
        ),
        _Row(
            "flux_rate_exceeded_16gb",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SINGLE_HEAD,
            governor=_GovernorState.ESTABLISH_RATE_EXCEEDED,
            tick_bound=10,
            run_ticks=18,
        ),
        _Row(
            "flux_rate_exceeded_burst_24gb",
            _CARD_24GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SAME_MODEL_BURST,
            governor=_GovernorState.ESTABLISH_RATE_EXCEEDED,
            tick_bound=10,
            run_ticks=24,
        ),
        _Row(
            "flux_grace_exhausted_resident_24gb",
            _CARD_24GB,
            _FLUX,
            _Residency.RESIDENT_IDLE_TARGET,
            _QueueShape.SINGLE_HEAD,
            governor=_GovernorState.GRACE_EXHAUSTED,
            tick_bound=10,
            run_ticks=18,
        ),
        # -- Mid-sequence events. ----------------------------------------------------------------------------
        _Row(
            "sdxl_target_death_respawn_16gb",
            _CARD_16GB,
            _SDXL,
            _Residency.RESIDENT_IDLE_TARGET,
            _QueueShape.SAME_MODEL_BURST,
            event=_MidSequenceEvent.TARGET_DEATH_RESPAWN,
            tick_bound=10,
            run_ticks=24,
        ),
        _Row(
            "flux_target_death_respawn_24gb",
            _CARD_24GB,
            _FLUX,
            _Residency.RESIDENT_IDLE_TARGET,
            _QueueShape.SAME_MODEL_BURST,
            event=_MidSequenceEvent.TARGET_DEATH_RESPAWN,
            tick_bound=12,
            run_ticks=26,
        ),
        _Row(
            "sd15_external_reclaim_8gb",
            _CARD_8GB,
            _SD15,
            _Residency.ABSENT,
            _QueueShape.HEAD_PLUS_SKIPPERS,
            event=_MidSequenceEvent.EXTERNAL_RECLAIM,
            tick_bound=10,
            run_ticks=24,
        ),
        _Row(
            "flux_external_reclaim_16gb",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.HEAD_PLUS_SKIPPERS,
            event=_MidSequenceEvent.EXTERNAL_RECLAIM,
            tick_bound=10,
            run_ticks=24,
        ),
        # -- Whole-card residency disabled: the same heads must still be served. -----------------------------
        _Row(
            "flux_absent_16gb_residency_off",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SINGLE_HEAD,
            whole_card_enabled=False,
            tick_bound=8,
        ),
        # -- The residency's claim over intake. --------------------------------------------------------------
        # Each row raises the cooldown so the residency is still standing where its assertion is made: with
        # the cooldown at zero every residency releases the instant its work drains, which would leave the
        # claim's own ends untested.
        _Row(
            "flux_16gb_claim_serves_the_burst",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SAME_MODEL_BURST,
            tick_bound=8,
            run_ticks=16,
            claim=_ClaimScenario.SERVES_THE_BURST,
            cooldown_seconds=600,
        ),
        _Row(
            "flux_16gb_claim_cap_returns_the_pool",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.HEAD_PLUS_SKIPPERS,
            tick_bound=8,
            run_ticks=16,
            claim=_ClaimScenario.CAP_RETURNS_THE_POOL,
            # The cooldown alone would retain this residency for the whole run, so the pool coming back is
            # attributable to the cap and to nothing else.
            cooldown_seconds=600,
            max_hold_seconds=120,
        ),
        _Row(
            "flux_16gb_claim_empty_pops_release",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.SINGLE_HEAD,
            tick_bound=8,
            run_ticks=24,
            claim=_ClaimScenario.EMPTY_POPS_RELEASE,
            # Both clocks are set past the run's own reach so the early release is the only thing that can
            # end the claim inside it; the run is long enough that the cap still terminates the row.
            cooldown_seconds=600,
            max_hold_seconds=600,
        ),
        # -- A disaggregated sampler beside the service contexts. --------------------------------------------
        # The card's structural floor is squeezed by charges no inference teardown can reclaim (safety's
        # resident weights, the post-processing lane's context, the image lane's decode spike) while the
        # head's own model is already resident on an idle lane. The head's activation-inclusive peak nearly
        # doubles its weights at this size, so the row is where an activation spike priced as residency would
        # turn a dispatch into a whole-card claim over the very lanes the sampler's own encode and decode run
        # on. The head must sample on the residency it already has.
        _Row(
            "sdxl_disagg_hires_resident_idle_16gb",
            _CARD_16GB,
            _SDXL,
            _Residency.RESIDENT_IDLE_TARGET,
            _QueueShape.SINGLE_HEAD,
            shape=_SHAPE_HIRES_BATCH,
            service_contexts=True,
            disaggregated=True,
        ),
        _Row(
            "sdxl_disagg_hires_burst_16gb",
            _CARD_16GB,
            _SDXL,
            _Residency.RESIDENT_IDLE_TARGET,
            _QueueShape.SAME_MODEL_BURST,
            shape=_SHAPE_HIRES_BATCH,
            service_contexts=True,
            disaggregated=True,
            tick_bound=8,
            run_ticks=22,
        ),
        # The companion that keeps the guard honest: the same card and the same service contexts, but a
        # genuinely weight-dominant head that still takes the device. A suppression wide enough to catch this
        # would have traded a false claim for a model left thrashing.
        _Row(
            "flux_resident_idle_service_contexts_16gb",
            _CARD_16GB,
            _FLUX,
            _Residency.RESIDENT_IDLE_TARGET,
            _QueueShape.SINGLE_HEAD,
            service_contexts=True,
            tick_bound=10,
            run_ticks=20,
        ),
        _Row(
            "flux_16gb_claim_foreign_jobs_queued_first",
            _CARD_16GB,
            _FLUX,
            _Residency.ABSENT,
            _QueueShape.HEAD_PLUS_SKIPPERS,
            tick_bound=8,
            run_ticks=16,
            claim=_ClaimScenario.FOREIGN_QUEUED_FIRST,
        ),
    )


_ROWS = _rows()


_QUEUE_LENGTHS: dict[_QueueShape, int] = {
    _QueueShape.SINGLE_HEAD: 1,
    _QueueShape.HEAD_PLUS_SKIPPERS: 3,
    _QueueShape.MULTI_MODEL_INTERLEAVE: 4,
    _QueueShape.SAME_MODEL_BURST: 3,
    _QueueShape.AT_DEPTH: 4,
}


def _queue_models(row: _Row) -> list[_ModelClass]:
    """The models of the jobs the row queues, head first."""
    head = row.head_model
    light = _SD15 if head is not _SD15 else _SD15_OTHER
    other = _SAME_CLASS_PARTNER[head.name]
    if row.queue is _QueueShape.SINGLE_HEAD:
        return [head]
    if row.queue is _QueueShape.HEAD_PLUS_SKIPPERS:
        return [head, light, light]
    if row.queue is _QueueShape.MULTI_MODEL_INTERLEAVE:
        return [head, other, head, other]
    if row.queue is _QueueShape.SAME_MODEL_BURST:
        return [head] * 3
    return [head] * 4


def _seed_governor_state(world: _DispatchWorld, row: _Row) -> None:
    """Drive the card's whole-card ledger into the row's governor state through its real charge APIs."""
    ledger = world.scheduler._whole_card_ledger
    if row.governor is _GovernorState.FRESH:
        return
    now = world.now
    state = ledger.state_for(None)
    if row.governor is _GovernorState.ESTABLISH_RATE_EXCEEDED:
        state.establishments.extend([now - index for index in range(_ESTABLISH_WINDOW_LIMIT)])
        assert ledger.establish_rate_exceeded(None, now=now) is True, (
            f"{row.label}: precondition, the card's establishment allowance is spent"
        )
        return
    cycle_cost = _WHOLE_CARD_ESTABLISH_GRACE_SECONDS + _WHOLE_CARD_RESTORE_GRACE_SECONDS
    for index in range(int(_GRACE_BUDGET_SECONDS // cycle_cost) + 1):
        state.grace_charges.append((now - index, _WHOLE_CARD_ESTABLISH_GRACE_SECONDS))
        state.grace_charges.append((now - index, _WHOLE_CARD_RESTORE_GRACE_SECONDS))
    assert ledger.grace_budget_exhausted(None, now=now) is True, (
        f"{row.label}: precondition, the card's rolling grace allowance is spent"
    )


async def _build_world(row: _Row) -> tuple[_DispatchWorld, list[ImageGenerateJobPopResponse]]:
    """Build the world for a row and pop its queue in order."""
    world = _DispatchWorld(
        card=row.card,
        lane_count=row.lanes,
        max_threads=row.max_threads,
        queue_depth=_QUEUE_LENGTHS[row.queue],
        whole_card_enabled=row.whole_card_enabled,
        cooldown_seconds=row.cooldown_seconds,
        max_hold_seconds=row.max_hold_seconds,
        service_contexts=row.service_contexts,
        disaggregated=row.disaggregated,
    )

    if row.residency is _Residency.RESIDENT_IDLE_TARGET:
        world.seed_resident(0, row.head_model, in_vram=True)
    elif row.residency is _Residency.RAM_STAGED:
        world.seed_resident(0, row.head_model, in_vram=False)
    elif row.residency is _Residency.RESIDENT_ON_SIBLING:
        # The copy sits on the last lane rather than the first, so dispatch must route to the lane that
        # already holds the weights instead of staging a second copy into the empty lane it would otherwise
        # reach for.
        world.seed_resident(row.lanes - 1, row.head_model, in_vram=True)

    _seed_governor_state(world, row)

    jobs: list[ImageGenerateJobPopResponse] = []
    for model in _queue_models(row):
        job = make_job_pop_response(
            model.name,
            width=row.shape.width,
            height=row.shape.height,
            n_iter=row.shape.batch,
            ddim_steps=8,
        )
        await world.pop(job)
        jobs.append(job)
    return world, jobs


def _fire_event(world: _DispatchWorld, row: _Row) -> None:
    """Apply the row's mid-sequence disturbance."""
    if row.event is _MidSequenceEvent.TARGET_DEATH_RESPAWN:
        world.kill_lane_holding(row.head_model)
    elif row.event is _MidSequenceEvent.EXTERNAL_RECLAIM:
        world.evict_idle_resident_sibling(except_model=row.head_model)


async def _drive(world: _DispatchWorld, row: _Row) -> None:
    """Run the row's ticks, firing its disturbance at the fixed disturbance tick."""
    for _ in range(row.run_ticks):
        await world.step()
        if world.tick == _DEATH_TICK:
            _fire_event(world, row)


# --------------------------------------------------------------------------------------------------------
# The property
# --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("row", _ROWS, ids=[row.label for row in _ROWS])
async def test_servable_head_reaches_dispatch_within_its_bound(row: _Row) -> None:
    """A physically servable head reaches sampling within the row's tick bound.

    A head whose model prices above the card's achievable ceiling takes the other disclosed exit: the model
    goes on the ceiling hold and the job is handed back for reissue. Keeping both outcomes in one table is
    what makes it a discriminating property rather than a claim that everything asked of the worker is
    admitted.
    """
    world, jobs = await _build_world(row)
    head = jobs[0]

    await _drive(world, row)

    if row.expected is _Expected.UNSERVABLE_HELD:
        assert world.dispatch_tick(head) is None, (
            f"{row.label}: a head the card cannot serve must not have been dispatched. {world.state_dump()}"
        )
        assert world.job_tracker.is_model_held_by_ceiling(row.head_model.name), (
            f"{row.label}: an unservable head must put its model on the disclosed ceiling hold rather than "
            f"leaving the queue quietly. {world.state_dump()}"
        )
        assert world.stage(head) is JobStage.PENDING_SUBMIT, (
            f"{row.label}: the held model's job must be faulted onward for reissue rather than kept in a "
            f"queue that cannot serve it. {world.state_dump()}"
        )
        return

    dispatched_at = world.dispatch_tick(head)
    assert dispatched_at is not None, (
        f"{row.label}: the servable head never reached sampling in {row.run_ticks} ticks. {world.state_dump()}"
    )
    assert dispatched_at <= row.tick_bound, (
        f"{row.label}: the head reached sampling at tick {dispatched_at}, past its bound of {row.tick_bound}. "
        f"{world.state_dump()}"
    )


@pytest.mark.parametrize("row", _ROWS, ids=[row.label for row in _ROWS])
async def test_whole_queue_drains_and_obligations_close(row: _Row) -> None:
    """Every queued job the row can serve drains, and every obligation opened along the way is released.

    Three obligations are checked at the far end of the run: the planned reserve-ledger charges booked for
    admitted preloads, the whole-card residency claim over the device, and the safety / post-processing GPU
    pauses a residency takes out. Each must be back at its released value once the queue has drained, because
    a charge or a pause that outlives the work it was taken for is what wedges the next head.

    The run's own ticks are then read back for the two liveness obligations an end-state check cannot see: a
    card that sat idle with work its free VRAM covered, and a dispatch held for an entity that was itself
    going nowhere. Both are compatible with a queue that drains, so neither is implied by anything above.
    """
    world, jobs = await _build_world(row)

    await _drive(world, row)

    for index, job in enumerate(jobs):
        # An unservable head's own jobs take the ceiling-hold exit; everything behind it whose model the card
        # can serve must still drain, or the unservable head has wedged the work it was standing in front of.
        if not row.drain_all and job.model == row.head_model.name:
            continue
        assert world.dispatch_tick(job) is not None, (
            f"{row.label}: queued job {index} ({job.model}) never reached sampling in {row.run_ticks} "
            f"ticks. {world.state_dump()}"
        )

    planned_mb = world.planned_overlay_mb()
    assert planned_mb == pytest.approx(0.0), (
        f"{row.label}: {planned_mb:.0f} MB of planned preload charge outlived the work it was booked for. "
        f"{world.state_dump()}"
    )
    assert world.scheduler.is_whole_card_residency_active() is False, (
        f"{row.label}: the card is still claimed for an exclusive residency after the queue drained. "
        f"{world.state_dump()}"
    )
    assert world.scheduler.whole_card_residency_grace_active() is False, (
        f"{row.label}: a residency grace window is still open after the queue drained. {world.state_dump()}"
    )
    assert world.lifecycle.is_safety_gpu_paused is False, (
        f"{row.label}: safety was left paused off the GPU after the queue drained. {world.state_dump()}"
    )
    assert world.lifecycle.is_post_process_gpu_paused is False, (
        f"{row.label}: the post-processing lane was left paused after the queue drained. {world.state_dump()}"
    )
    assert_no_committed_slot_retired(world, context=row.label)
    assert_never_idle_with_fitting_work(world, context=row.label)
    assert_no_unservable_dispatch_hold(world, context=row.label)


@pytest.mark.parametrize(
    "row",
    [row for row in _ROWS if row.expected is _Expected.DISPATCHED],
    ids=[row.label for row in _ROWS if row.expected is _Expected.DISPATCHED],
)
async def test_servable_work_is_never_faulted_out_of_the_queue(row: _Row) -> None:
    """No job the row queues is faulted while it waits: every one leaves the queue by being sampled.

    The wedge-class failure this guards is a servable job being given up on rather than served. Read
    positively: the count of jobs that reached sampling equals the count queued, so none left the queue by
    any other door.
    """
    world, jobs = await _build_world(row)

    await _drive(world, row)

    sampled = [job for job in jobs if world.dispatch_tick(job) is not None]
    still_queued = [job for job in jobs if world.stage(job) is JobStage.PENDING_INFERENCE]
    assert len(sampled) + len(still_queued) == len(jobs), (
        f"{row.label}: {len(jobs) - len(sampled) - len(still_queued)} queued job(s) left the queue without "
        f"being sampled. {world.state_dump()}"
    )
    assert still_queued == [], (
        f"{row.label}: {len(still_queued)} job(s) were still waiting at the end of the run. {world.state_dump()}"
    )


@pytest.mark.parametrize(
    "row",
    [row for row in _ROWS if row.governor is _GovernorState.GRACE_EXHAUSTED],
    ids=[row.label for row in _ROWS if row.governor is _GovernorState.GRACE_EXHAUSTED],
)
async def test_grace_governed_head_is_served_without_claiming_the_card(row: _Row) -> None:
    """A head the grace budget refuses the card is still served, co-resident, without ever claiming it.

    The budget's window (1200s) far outlasts the governor dwell, so the brake cannot release before the
    head stops asking for the card: the head must fall through to ordinary measured admission and sample
    without spending the establishment the governor withheld.
    """
    world, jobs = await _build_world(row)
    head = jobs[0]
    establishments_before = len(world.scheduler._whole_card_ledger.state_for(None).establishments)

    await _drive(world, row)

    assert world.dispatch_tick(head) is not None, (
        f"{row.label}: a governed head that the card can hold must still be served. {world.state_dump()}"
    )
    assert len(world.scheduler._whole_card_ledger.state_for(None).establishments) <= establishments_before, (
        f"{row.label}: the governed head claimed the card the governor refused it. {world.state_dump()}"
    )


@pytest.mark.parametrize(
    "row",
    [row for row in _ROWS if row.governor is _GovernorState.ESTABLISH_RATE_EXCEEDED],
    ids=[row.label for row in _ROWS if row.governor is _GovernorState.ESTABLISH_RATE_EXCEEDED],
)
async def test_rate_governed_head_claims_the_card_once_the_brake_lifts(row: _Row) -> None:
    """A head the rate limiter defers is served by claiming the card after the brake's own window lapses.

    The rate window and the governor dwell are the same span by design: a rate deferral always releases on
    its own before the whole-card preference is abandoned, so the head takes the residency it is entitled
    to (sole residency beats coerced co-resident streaming) rather than being downgraded. The limiter's
    ceiling is still honored: the seeded establishments age out before the new one is counted.
    """
    world, jobs = await _build_world(row)
    head = jobs[0]

    await _drive(world, row)

    assert world.dispatch_tick(head) is not None, (
        f"{row.label}: a rate-deferred head must be served once the brake lifts. {world.state_dump()}"
    )
    state = world.scheduler._whole_card_ledger.state_for(None)
    assert 1 <= len(state.establishments) <= 2, (
        f"{row.label}: the head's claim must be a single establishment inside the limiter's ceiling, found "
        f"{len(state.establishments)}. {world.state_dump()}"
    )


_DISAGGREGATED_ROWS = tuple(row for row in _ROWS if row.disaggregated)

_WEIGHT_DOMINANT_CLAIM_ROWS = tuple(
    row
    for row in _ROWS
    if row.head_model is _FLUX
    and row.residency is _Residency.RESIDENT_IDLE_TARGET
    and row.card is _CARD_16GB
    and row.governor is _GovernorState.FRESH
    and row.event is _MidSequenceEvent.NONE
)
"""Rows whose head is heavy enough that the device is genuinely its only home, reached through the
dispatch-time residency path (its weights are already resident on an idle lane).

The card must be the one with no room for a second model beside those weights: on a roomier card the same
head co-resides by design and no claim is expected. A governed or disturbed row is excluded for the same
reason, since each is driving a deliberate refusal or recovery of its own."""


@pytest.mark.parametrize("row", _DISAGGREGATED_ROWS, ids=[row.label for row in _DISAGGREGATED_ROWS])
async def test_a_disaggregated_head_is_served_without_claiming_the_card(row: _Row) -> None:
    """A UNet-only sampler is dispatched on the residency it has, never by reserving the device.

    Whole-card residency and disaggregation are mutually exclusive by construction: the sampler's own encode
    and decode run in the lanes a residency stops, so a claim raised for one of these heads takes down the
    pipeline serving it. The row reads the outcome positively (the head samples) and the mechanism
    negatively (no establishment was ever charged to the card), because a head that happened to be served
    after a claim and a restore would satisfy the first alone.
    """
    world, jobs = await _build_world(row)

    await _drive(world, row)

    assert world.dispatch_tick(jobs[0]) is not None, (
        f"{row.label}: the disaggregated head never reached sampling. {world.state_dump()}"
    )
    assert not world.scheduler._whole_card_ledger.state_for(None).establishments, (
        f"{row.label}: a disaggregated head claimed the card its own encode and decode lanes run on. "
        f"{world.state_dump()}"
    )


@pytest.mark.parametrize(
    "row",
    _WEIGHT_DOMINANT_CLAIM_ROWS,
    ids=[row.label for row in _WEIGHT_DOMINANT_CLAIM_ROWS],
)
async def test_a_weight_dominant_resident_head_still_claims_the_card(row: _Row) -> None:
    """A head whose weights fill the card takes the device even though its model is already resident.

    The discriminating half of the residency suppression: a claim is withheld when there is no inference
    context for it to reclaim, never because the head arrived at dispatch already resident. Without this the
    same table would pass with whole-card residency removed entirely.
    """
    world, jobs = await _build_world(row)

    await _drive(world, row)

    assert world.dispatch_tick(jobs[0]) is not None, (
        f"{row.label}: the whole-card head never reached sampling. {world.state_dump()}"
    )
    assert world.scheduler._whole_card_ledger.state_for(None).establishments, (
        f"{row.label}: the head that needs the device to itself was never granted it. {world.state_dump()}"
    )


_CLAIM_ROWS = tuple(row for row in _ROWS if row.claim is not _ClaimScenario.NONE)

_CLAIM_INTAKE_JOBS = 3
"""How many further jobs the horde offers a row that models continuing demand.

Enough that the burst outlives the establishment and the first dispatch, and few enough that the queue the
worker accepts still drains inside the row's ticks."""


async def _drive_claim_row(world: _DispatchWorld, row: _Row) -> list[ImageGenerateJobPopResponse]:
    """Run a claim row, offering the horde's answers through whatever the worker is currently asking for.

    Returns the resident-model jobs the worker accepted mid-run, which is what a row asserting the burst is
    served reads its outcome from. A foreign job is offered on the same ticks and is expected to be refused
    for as long as the claim stands: work the worker never advertised is work the horde never sends it.
    """
    accepted: list[ImageGenerateJobPopResponse] = []
    offered_resident = 0
    for _ in range(row.run_ticks):
        await world.step()
        if row.claim is _ClaimScenario.EMPTY_POPS_RELEASE:
            world.report_empty_pop()
            continue
        if row.claim is not _ClaimScenario.SERVES_THE_BURST or offered_resident >= _CLAIM_INTAKE_JOBS:
            continue
        offered_resident += 1
        resident_job = make_job_pop_response(row.head_model.name, width=512, height=512, ddim_steps=8)
        if await world.offer_job(resident_job):
            accepted.append(resident_job)
        foreign_job = make_job_pop_response(_SD15.name, width=512, height=512, ddim_steps=8)
        foreign_taken = await world.offer_job(foreign_job)
        if world.claim_ticks and world.claim_ticks[-1] == world.tick:
            assert foreign_taken is False, (
                f"{row.label}: a foreign job arrived while the residency claimed the offer, which is the "
                f"intake the claim exists to stop. {world.state_dump()}"
            )
    return accepted


@pytest.mark.parametrize("row", _CLAIM_ROWS, ids=[row.label for row in _CLAIM_ROWS])
async def test_the_residency_claims_intake_and_gives_it_back(row: _Row) -> None:
    """A held residency asks the horde for its own model alone, and every way that claim ends returns the pool.

    The claim is what makes a residency a burst-serving window rather than a card the worker keeps evicting
    itself from: while it stands, no foreign job is asked for, so nothing arrives to push the resident weights
    back to host RAM. Each row drives one of its ends and asserts the pool comes back, because a claim with no
    reachable end is a worker that has advertised itself down to one model permanently.
    """
    world, jobs = await _build_world(row)

    accepted = await _drive_claim_row(world, row)

    assert world.claim_ticks, (
        f"{row.label}: the residency never claimed the offer, so the row proves nothing about intake. "
        f"{world.state_dump()}"
    )
    for tick in world.claim_ticks:
        assert world.offers[tick] == frozenset({row.head_model.name}), (
            f"{row.label}: at tick {tick} the claim stood but the worker was still advertising "
            f"{sorted(world.offers[tick])}. {world.state_dump()}"
        )

    if row.claim is _ClaimScenario.SERVES_THE_BURST:
        assert accepted, f"{row.label}: the claim must keep taking work for the model it holds the card for"
        for job in jobs + accepted:
            assert world.dispatch_tick(job) is not None, (
                f"{row.label}: a job for the resident model was not served inside the window the residency "
                f"was held for. {world.state_dump()}"
            )

    if row.claim is _ClaimScenario.CAP_RETURNS_THE_POOL:
        assert row.cooldown_seconds > row.run_ticks * _TICK_SECONDS, (
            f"{row.label}: precondition, the cooldown outlasts the run, so the pool returning is attributable "
            "to the maximum hold and to nothing else"
        )
        assert world.claim_released_at > 0.0, (
            f"{row.label}: the maximum hold elapsed and the claim still stands. {world.state_dump()}"
        )
        assert world.claim_released_at >= world.claim_expires_at, (
            f"{row.label}: the claim ended before its cap, so this row is not measuring the cap"
        )

    if row.claim is _ClaimScenario.EMPTY_POPS_RELEASE:
        assert world.claim_released_at > 0.0, (
            f"{row.label}: a resident model the horde has no work for held the offer for the whole run. "
            f"{world.state_dump()}"
        )
        assert world.claim_released_at < world.claim_expires_at, (
            f"{row.label}: the claim was only ended by its cap; the empty answers must release it sooner, or a "
            "worker nobody is sending work to sits on its own claim"
        )

    if row.claim is _ClaimScenario.FOREIGN_QUEUED_FIRST:
        last_claimed_tick = world.claim_ticks[-1]
        for job in jobs:
            if job.model == row.head_model.name:
                continue
            dispatched_at = world.dispatch_tick(job)
            assert dispatched_at is not None, (
                f"{row.label}: a foreign job queued before the residency existed never drained after the "
                f"claim ended. {world.state_dump()}"
            )
            assert dispatched_at > last_claimed_tick, (
                f"{row.label}: a foreign job was pulled onto the card at tick {dispatched_at}, while the "
                f"residency still claimed it. {world.state_dump()}"
            )

    # However the claim ended, the worker is asking for its whole pool again.
    assert world.offers[world.tick] == frozenset(model.name for model in _MODEL_CLASSES), (
        f"{row.label}: the full model pool never came back to the offer. {world.state_dump()}"
    )


def test_the_matrix_states_its_own_coverage() -> None:
    """The table's axes are fully enumerated: every driven row is unique and every omission is explained."""
    labels = [row.label for row in _ROWS]
    assert len(labels) == len(set(labels)), "row labels must be unique so a failure names exactly one cell"
    assert len(_PRUNED_COMBINATIONS) > 0, "omissions from the cross-product must be listed, never silent"
    for combination, reason in _PRUNED_COMBINATIONS:
        assert combination and reason, "each pruned combination must carry its reason"
    driven_cards = {row.card.label for row in _ROWS}
    driven_models = {row.head_model.label for row in _ROWS}
    driven_queues = {row.queue for row in _ROWS}
    driven_residencies = {row.residency for row in _ROWS}
    driven_governors = {row.governor for row in _ROWS}
    driven_events = {row.event for row in _ROWS}
    assert driven_cards == {"8gb", "16gb", "24gb"}
    assert {"sd15", "sdxl", "flux"} <= driven_models
    assert driven_queues == set(_QueueShape)
    assert driven_residencies == set(_Residency)
    assert driven_governors == set(_GovernorState)
    assert driven_events == set(_MidSequenceEvent)
    assert {row.max_threads for row in _ROWS} == {1, 2}
    assert {row.claim for row in _ROWS} == set(_ClaimScenario)
    assert {row.shape.label for row in _ROWS} == {_SHAPE_SMALL.label, _SHAPE_HIRES_BATCH.label}
    assert {row.service_contexts for row in _ROWS} == {False, True}
    assert {row.disaggregated for row in _ROWS} == {False, True}
