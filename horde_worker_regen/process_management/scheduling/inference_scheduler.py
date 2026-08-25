"""Schedules model preloading, inference start, and model unloading."""

from __future__ import annotations

import enum
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from loguru import logger

from horde_worker_regen.compute_mode import is_cpu_only_install
from horde_worker_regen.process_management.config.runtime_config import RuntimeConfig
from horde_worker_regen.process_management.config.worker_state import PopPauseOwner, WorkerState
from horde_worker_regen.process_management.gpu.card_runtime import CardRuntime, safety_permitted_card_indices
from horde_worker_regen.process_management.gpu.gpu_eligibility import (
    CardEligibilityVerdict,
    card_eligibility_for,
)
from horde_worker_regen.process_management.ipc.action_ledger import LedgerEventType
from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeControlModelMessage,
    HordeEvictComponentsControlMessage,
    HordeInferenceControlMessage,
    HordePreloadInferenceModelMessage,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo, LineSkip, NextJobAndProcess
from horde_worker_regen.process_management.jobs.job_tracker import JobFaultOrigin, JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import (
    ALLOCATOR_CACHE_CAPABLE_PROCESS_TYPES,
    HordeProcessType,
)
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner, ProcessLifecycleManager
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.component_residency_map import ComponentResidencyMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.models.lru_cache import LRUCache
from horde_worker_regen.process_management.models.model_metadata import ModelMetadata
from horde_worker_regen.process_management.models.model_sizing import ModelSizeTier, model_size_tier
from horde_worker_regen.process_management.resources.admission_identity import (
    admission_noise_buffer_mb,
)
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from horde_worker_regen.process_management.resources.foreign_vram_floor import ForeignVramFloorTracker
from horde_worker_regen.process_management.resources.model_serviceability import (
    ModelServiceabilityVerdict,
    assess_model_serviceability,
    model_footprint_figures_for_baseline,
)
from horde_worker_regen.process_management.resources.reclaim_ladder import (
    CacheReleaseTarget,
    IdleResidentModel,
    LadderCandidates,
    LaneReclaimCandidate,
    ReclaimRung,
    ReclaimRungKind,
    VerifiedReclaimLadder,
    build_reclaim_ladder,
)
from horde_worker_regen.process_management.resources.resource_budget import (
    _SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB,
    BudgetVerdict,
    CommittedReserveLedger,
    RamBudget,
    RamPressureVerdict,
    StreamForecast,
    VramBudget,
    WholeCardResidencyState,
    assess_ram_pressure,
    effective_inference_reserve_mb,
    forecast_weight_streaming,
    is_model_locally_unservable_for,
    platform_context_constant_mb,
    predict_job_decode_spike_mb,
    predict_job_footprint_mb,
    predict_job_post_processing_vram_mb,
    predict_job_sampler_only_vram_mb,
    predict_job_sampling_vram_mb,
    predict_job_unet_only_ram_mb,
    predict_job_weight_mb,
    ram_pressure_floor_mb,
)
from horde_worker_regen.process_management.resources.run_metrics import (
    ChurnKind,
    DecisionKind,
    DecisionSink,
    DecisionVerdict,
)
from horde_worker_regen.process_management.resources.vram_arbiter import (
    ActuatorCommand,
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
    VramDisposition,
    VramRequest,
    VramRequestKind,
    VramVerdict,
)
from horde_worker_regen.process_management.resources.vram_attribution import _REPORT_STALENESS_SECONDS
from horde_worker_regen.process_management.resources.vram_footprints import (
    SAFETY_PROCESS_BASELINE,
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
    ResolutionBucket,
)
from horde_worker_regen.process_management.scheduling.clearance_lease import (
    TAIL_OVERLAP_MIN_PROGRESS_FOR_ESTIMATE,
    ActiveSampler,
    ClearanceInputs,
    ClearanceWaiter,
)
from horde_worker_regen.process_management.scheduling.context_overhead_model import ContextOverheadModel
from horde_worker_regen.process_management.scheduling.dispatch_affinity import (
    _AFFINITY_MAX_SKIPS,
    AffinitySkipState,
    affinity_budget_seconds,
    affinity_skip_allowed,
    affinity_skip_disclosure,
    record_affinity_skip,
)
from horde_worker_regen.process_management.scheduling.governance import (
    AdmissionDecision,
    CardProcessSnapshot,
    ClearProcessDraining,
    EvictIdleModels,
    GovernanceAction,
    HostMemorySnapshot,
    InferenceSlotSnapshot,
    MarkProcessDraining,
    PausePops,
    PreloadSlotSnapshot,
    RamGovernorState,
    RamReclaimOutcome,
    RecycleProcess,
    ReduceCardProcesses,
    ReduceWorkerProcesses,
    ResourceGovernor,
    RestoreCardProcess,
    RestoreWorkerProcess,
    SetPopHold,
    StopTrackingShedCard,
    StopTrackingWorkerShed,
    WholeCardGovernor,
    WholeCardPopClaim,
    WholeCardPopClaimRelease,
    WholeCardResidency,
    WholeCardResidencyMachine,
    WorkerProcessShedState,
    card_preload_order,
    compute_preload_disallowed_processes,
    decide_degrade_response,
    decide_process_reduction,
    decide_ram_reclaim_outcome,
    decide_shed_card_restore,
    max_coresident_for_peak,
    preload_concurrency_blocked,
    select_head_room_process_id,
)
from horde_worker_regen.process_management.scheduling.model_affinity import affinity_active
from horde_worker_regen.process_management.scheduling.performance_model import PerformanceModel, signature_from_job
from horde_worker_regen.process_management.scheduling.slot_duty import SlotDutyAccumulator, SlotDutyBucket
from horde_worker_regen.process_management.scheduling.workload_flow import (
    DISPATCH_ADMISSION_FLOW,
    POST_PROCESS_RESERVE_FLOW,
    PRELOAD_ADMISSION_FLOW,
)
from horde_worker_regen.telemetry_spans import span_preload_model
from horde_worker_regen.utils.config_coercion import config_number
from horde_worker_regen.utils.job_utils import (
    get_single_job_magnitude as _get_single_job_effective_megapixelsteps,
)
from horde_worker_regen.utils.vram_quota import effective_post_process_vram_quota_mb

if TYPE_CHECKING:
    from horde_model_reference.component_identity import ComponentIdentitySidecar

    from horde_worker_regen.bridge_data.data_model import reGenBridgeData


@dataclass(frozen=True)
class LatestPreloadAdmission:
    """Operator-facing record of the most recent preload-admission decision."""

    decision: AdmissionDecision
    """The admission gate's decision."""
    model: str | None
    """Model whose queued job was judged, when available."""
    process_id: int | None
    """Target inference process selected by the decision, when one was selected."""
    reason: str
    """Short human-readable explanation for the decision."""
    timestamp: float
    """Worker wall-clock time when the decision was recorded."""


_PRELOAD_ADMISSION_VERDICTS: dict[AdmissionDecision, DecisionVerdict] = {
    AdmissionDecision.ADMIT: DecisionVerdict.ADMIT,
    AdmissionDecision.ALREADY_LOADED: DecisionVerdict.NO_OP,
    AdmissionDecision.QUARANTINED: DecisionVerdict.DENY,
    AdmissionDecision.UNSERVICEABLE: DecisionVerdict.DENY,
    AdmissionDecision.NEXT_JOB: DecisionVerdict.WITHHOLD,
    AdmissionDecision.STOP_PASS: DecisionVerdict.WITHHOLD,
}
"""Preload-admission decisions whose recorded verdict is not a plain defer.

Only the resolving and terminal decisions are mapped: everything else holds the job for a later cycle and
records as ``DEFER``, which is what the coalescing recorder collapses while the hold persists."""

_STAGING_ENCODE_VRAM_MB = 2048.0
"""VRAM a staged (dispatched but not-yet-cleared) job actually charges the device under the clearance
lease: the text-encoder footprint plus the conditioning working set for the largest supported family
(SDXL's dual CLIP encode lands near 1.5-2GB). Under the clearance lease the diffusion weights load
*inside* the leased sample call, at clearance, not at dispatch, so a staged job's device footprint is
only this encode working set until it is cleared. Dispatch admits staging while measured device free net
of the reserve covers this charge; the full materialisation is priced at clearance instead."""


class StagingDeferReason(enum.Enum):
    """Why the in-progress cap was held at the sampling-slot count instead of allowing another staged job.

    Deferring staging is ordinary backpressure, but it is also the clause that leaves spare inference
    processes idle while jobs queue, so a session has to be able to read which of the two measurements held
    it: an unread card, or a card whose free reserve does not cover the encode footprint a staged job adds.
    """

    MEASUREMENT_UNREAD = "unread"
    """No GPU-bearing child has reported its VRAM yet, so there is no evidence to admit staging on."""
    ENCODE_HEADROOM_SHORT = "headroom"
    """Measured free VRAM net of the reserve does not cover a staged job's encode working set."""


class RetentionDenialReason(enum.Enum):
    """Which gate refused a VRAM retention grant, so a session can read what its retention policy costs.

    Retention is the difference between a same-model successor reusing weights already on the card and paying
    a full host-to-device upload for them, so a worker that grants nothing is indistinguishable at the duty
    figure from one whose grants are all being evicted before reuse. Bucketing the refusals separates the two:
    a run denied for lack of repeat evidence is serving traffic retention cannot help, while one denied on
    static fit or governor state is serving traffic it could help on a card that will not carry it.
    """

    ACTUATION_DISABLED = "actuation_disabled"
    """The legacy comfy unload regime is configured, so the child returns the card whatever the grant says."""
    BUDGET_INACTIVE = "budget_inactive"
    """Measured VRAM budgeting is off, so nothing here can price what holding the weights would cost."""
    WDDM_PAGING = "wddm_paging"
    """The driver is already demand-paging the worker's allocations; holding weights can only deepen it."""
    NO_REPEAT_EVIDENCE = "no_repeat_evidence"
    """The slot's trailing dispatches do not contain this model, so nothing predicts a same-model successor."""
    GOVERNOR_STATE = "governor_state"
    """The card is PRESSURE or SATURATED, so the reclaim ladder holds priority over a new resident."""
    UNPRICEABLE = "unpriceable"
    """A sibling context or an existing retained resident shares the card at a cost not yet measured."""
    STATIC_FIT = "static_fit"
    """The card's total cannot absorb this job's peak beside what retention already holds on it."""


_RETENTION_REPEAT_EVIDENCE_DISPATCHES = 3
"""How many of a slot's trailing dispatches are searched for a repeat before retention is granted on it.

Retention pays for itself only when a same-model successor arrives on the same slot, and a grant that is
evicted before one does is pure cost: the weights occupy the card, price every subsequent grant's static fit
against themselves, and are handed back through the reclaim ladder having saved nothing. A worker offering one
model repeats on every dispatch and clears this at once; a worker offering a wide rotation repeats rarely and
is granted correspondingly little, which is the adaptation the policy is for. No queue lookahead can supply
that evidence (see :meth:`InferenceScheduler._should_keep_model_resident`), so the trailing dispatch history
is what stands in for it.

A starting point pending a signature sweep, not a measured optimum: three is short enough that a slot which
has moved on to a different model stops earning grants within a job or two, and long enough that a two-model
alternation on one slot still reads as repeating."""

_RETENTION_STALE_HOLD_SECONDS = 60.0
"""How long a retained copy may go unreused before the prediction that issued it counts as falsified.

A grant asserts that a same-model successor is about to arrive on this slot. The dispatch history that issued
it can never say otherwise afterwards, because a live grant's own dispatch is what that history contains, so
the only thing able to refute the prediction is the successor failing to turn up. This is the horizon at which
it has: long enough that a slot waiting through a sibling's long job, a weight download, or an ordinary queue
lull is not robbed of weights its next job would have used, and short enough that a hold the traffic has moved
on from is not still occupying a pressured card minutes later.

A starting point pending a signature sweep, not a measured optimum, and deliberately expressed in seconds of
demand rather than in jobs or bytes so that it means the same thing on any card and any offer size."""

_RETENTION_PRESSURE_REVOKE_SECONDS = 15.0
"""How long a card must be continuously off HEALTHY before evidence-lacking retained residents are revoked.

Debounced rather than immediate because a reload costs seconds of the card's earning time: revoking on a
momentary dip pays that cost for pressure that was about to clear on its own. Long enough to outlast the
transient a single sampling window's activation peak produces, short enough that the weights are back well
before the card reaches the hard floor and the ladder has to take them anyway. A retention the slot's own
traffic still backs is never revoked here; genuine saturation remains the verified ladder's to resolve."""

_STAGING_DEFER_REPEAT_SECONDS = 300.0
"""How long an unchanged staging-defer reason stays suppressed before it is restated with its tally.

The cap is consulted on every dispatch decision, so a line per deferral would be noise: the reason edge
carries the information and a persistent hold is worth one restatement every few minutes."""

_AFFINITY_SCAN_TRACE_SECONDS = 30.0
"""Throttle window for the empty-affinity-scan diagnostic line.

The scan runs every scheduling cycle and empty is its common answer; one line per window while retained
weights exist keeps the gate that empties the scan visible without flooding the log."""


def format_staging_defer_tally(defers: Mapping[StagingDeferReason, int]) -> str | None:
    """A compact staging-deferral tally for the duty-cycle line, or None when staging was never held back.

    Reports the total and the share each measurement took of it, largest first, so a duty figure short of
    target can be read straight across to what kept spare inference processes out of the queue.
    """
    counted = {reason: count for reason, count in defers.items() if count}
    total = sum(counted.values())
    if not total:
        return None
    ranked = sorted(counted.items(), key=lambda entry: (-entry[1], entry[0].value))
    shares = ", ".join(f"{reason.value} {count / total:.0%}" for reason, count in ranked)
    return f"staging deferred: {total} ({shares})"


_MISSING_MODEL_LATCH_FALLBACK_SECONDS = 150.0
"""Bound for the missing-model recovery latch when no numeric ``preload_timeout`` is configured.
Matches the ``preload_timeout`` default."""

_RESIDENCY_GRACE_SECONDS = 30.0
"""How long a model stays protected from RAM eviction after its last live demand, in the
models-exceed-processes regime. Bridges the gap between a model's consecutive jobs so a
process does not disk-reload the very model it just used when the next job for it has not yet
been popped."""

_DEFAULT_VRAM_RESERVE_MB = 2048.0
"""Fallback VRAM reserve (MB) used until the live config value is read. Matches the
``vram_reserve_mb`` config default; covers transient spikes such as tiled VAE decode."""

_PP_OVERLAP_MEASURED_MARGIN_MB = 1024.0
"""Measured free-VRAM margin (MB) the post-processing/sampling co-residency admission path holds back on
top of the reserve and the sampling peak (and, for a not-yet-allocated pending chain, its predicted
reserve). The static co-residency gate prices against the card's reported total (safe under WDDM
demand-paging, which falsifies the driver's free figure); when it withholds, this second path consults the
parent's measured device-free reading, which during an active chain already reflects the chain's real
allocations and so is more accurate than the ledger's worst-case reserve. The margin absorbs allocator
fragmentation and foreign churn so the measured admission does not over-commit the card. Only consulted when
the driver is not paging: under active paging the measured free is a lie and the static fence stands."""

_DEFAULT_RAM_RESERVE_MB = 4096.0
"""Fallback system-RAM reserve (MB) used until the live config value is read. Matches the
``ram_reserve_mb`` config default; keeps resident-in-RAM weights from forcing the OS to page."""

_STALE_RAM_UNLOAD_REPLACE_BYTES = 1024 * 1024 * 1024
"""RSS threshold above which a model-less idle process is still materially holding RAM after unload."""

_FRESH_INFERENCE_CHILD_BASELINE_MB = 1100.0
"""Resident RSS (MB) a just-spawned inference child holds before it loads any model weights.

The interpreter, torch/CUDA import allocations, and IPC scaffolding a cold child carries (measured ~1.03-1.1GB
fresh). An idle process's RSS above this baseline is retained model pages the allocator kept after an unload,
which a subsequent preload onto that slot reuses rather than allocating anew; the excess is what the marginal
RAM credit is computed from. Erring toward the high end of the measured range keeps the credit conservative
(less RSS is treated as reusable) so the budget under-credits rather than over-admits."""

_CREEP_CONTAINMENT_RSS_BYTES = 18432 * 1024 * 1024
"""Idle-slot RSS above which a process is cycled for creep containment regardless of its unload state.

An inference child creeps ~400MB/job with the model unchanged; left unchecked it grows unbounded (observed
8->19.8GB) and cycling is the only containment. The ceiling mirrors the ``ram_per_process_max_mb`` default
(18432 MB), the figure the danger-floor reclaim already treats as a single process's balloon limit, so the two
reclaim paths agree on what "too big" means; unlike that path (which fires only under the danger floor) this
containment runs whenever a reclaim is attempted, so a genuine leak is bounded before the host reaches the
floor. It sits well above any single resident XL checkpoint plus its runtime, so a legitimate single-model
reuse target never trips it, but a genuinely crept slot does. Above this figure the retained RSS is leak, not
clean reusable pages, so creep containment overrides the reuse protection that otherwise spares the head's
staging target: such a slot is cycled and cold-loaded rather than reused, which is the correct trade when
reusing it would only perpetuate the leak. A slot that was just routed a preload is still spared (reaping it
mid-stage would fault the head's load), matching the stale-unload path's own mid-preload guard."""

_LANE_RAM_CONTAINMENT_RSS_BYTES = 10240 * 1024 * 1024
"""Idle service-lane RSS above which the lane is asked to unload its models from RAM.

A disaggregation service lane (the COMPONENT text-encode lane, and the VAE decode lane) keeps its
components resident across jobs and, alternating between the hot pool models, ratchets its resident set
well past what its live encoders occupy (observed to 19-26GB on the COMPONENT lane). Unlike an inference
slot, a lane self-reloads its encoders on the next stage, so the containment remedy is an in-process RAM
unload rather than a process cycle: the only cost is one reload. This ceiling sits far below the host RAM
danger floor so containment actuates as a bounded sawtooth well before the floor is threatened, yet high
enough that a lane holding only its working encoders never trips it."""

_LANE_RAM_CONTAINMENT_MIN_INTERVAL_SECONDS = 180.0
"""Minimum wall-time between two RAM-unload requests to the same idle service lane.

A lane re-pages its encoders on the next stage after an unload, so a too-eager re-unload would pay a reload
every few jobs. This spacing bounds the containment to at most one unload per lane per interval, keeping the
sawtooth's reload cost negligible against the RAM it returns."""

_REUSE_CREDIT_RECONCILE_SETTLE_SECONDS = 30.0
"""How long after a credited admission the target's RSS must settle before the credit is reconciled.

A credited preload's real RAM growth is only truthful once the load completes and transient churn subsides.
This grace lets the target reach steady state before the measured-truth check compares its actual growth to
the charged amount, so the reconciliation reads settled RSS rather than a mid-load spike."""

_REUSE_CREDIT_RECONCILE_SLACK_MB = 2048.0
"""How far a credited admission's measured RSS growth may exceed its charge before it is flagged too generous.

The measured-truth check on the marginal credit: if a target's settled RSS grew by more than the effective
charge plus this slack, the credit under-priced the swap and the discrepancy is logged once so the credit
constants can be retuned against field truth. Slack absorbs ordinary per-job creep and measurement noise so
only a materially over-generous credit is reported."""


_REUSE_CREDIT_KIND_PAGE_REUSE = "page_reuse"
"""A credited admission whose charge was reduced by a reusable staging target's retained resident pages."""

_REUSE_CREDIT_KIND_COMPONENT = "component"
"""A credited admission priced at a disaggregation-class job's UNet-only component charge, not the checkpoint."""


@dataclass(frozen=True)
class _ReuseCreditRecord:
    """A credited RAM admission awaiting the measured-truth check against its target's settled RSS.

    Records what the marginal credit assumed at admit time so the reconciliation can compare the target's
    real RSS growth to the effective charge once the load settles. Held per target process; superseded if
    that slot is credited again before it settles.
    """

    model: str
    """The model the credited preload was staging onto the target."""
    rss_at_admit_mb: float
    """The target's resident RSS (MB) at admit time, the baseline the settled growth is measured against."""
    effective_charge_mb: float
    """The credited charge (MB) the admission priced the swap at; the growth is reconciled against this."""
    admitted_at: float
    """Wall-clock time of the credited admission, gating the settle grace before reconciliation."""
    kind: str = _REUSE_CREDIT_KIND_PAGE_REUSE
    """Which marginal accounting priced the admission (:data:`_REUSE_CREDIT_KIND_PAGE_REUSE` for a retained-page
    reuse credit, :data:`_REUSE_CREDIT_KIND_COMPONENT` for a UNet-only component charge). The reconciliation
    reads settled growth against the charge identically for both; only the discrepancy wording differs."""


_PRELOAD_FIRST_REPORT_GRACE_SECONDS = 5.0
"""How long a just-sent preload may still look idle before its first child state report arrives.

The parent records the model as ``LOADING`` immediately after sending ``PRELOAD_MODEL``, but the child
may still read as ``WAITING_FOR_JOB`` until it drains the control pipe and publishes its first preload
state. This short grace keeps stale-entry cleanup from expiring a healthy, just-sent preload while still
letting genuinely abandoned loading entries clear promptly.
"""
_RELEASE_CACHE_MIN_RECLAIMABLE_MB = 256.0
"""Minimum reclaimable allocator cache (MB) a GPU process must hold to qualify as a RELEASE_CACHE target.

A RELEASE_CACHE actuation runs ``torch.cuda.empty_cache`` on the lane, which returns only the allocator's
reserved-but-unallocated blocks (``process_reserved_mb - process_allocated_mb``); a process whose reservation
is its resident weights or components (a component/VAE/post-process lane holding encoders, allocated close to
reserved) has no such cache, so asking it to release frees nothing. Requiring this measured margin keeps a
resident-weight lane out of the release-target set so the escalation ladder does not emit a rung that can
never yield, which would otherwise keep the ladder non-empty forever and defer a head that reclaim can
never actually relieve."""

_SAFETY_GPU_LOAD_CHARGE_MB = 3044.0
"""The device VRAM (MB) charged when the safety process is loaded onto the GPU, for the arbiter's SAFETY_LOAD
gate. A documented conservative seed for the idle CLIP model plus its CUDA context. DeepDanbooru, BLIP, the
aesthetic head, and evaluation activations are explicitly reclaimable and are not fixed safety residency.
Erring high keeps safety off-GPU one more cycle rather than restoring it onto a card it would over-commit.
This is the *seed* for the learned safety figure: :meth:`InferenceScheduler._safety_footprint_mb` is the one
price every consumer reads, and it raises this constant by any measured :attr:`FootprintStage.SAFETY`
watermark."""

_MEMORY_PRESSURE_PAUSE_OWNERS = frozenset({PauseOwner.RUNTIME_SAFETY_PLACEMENT, PauseOwner.RECLAIM_LADDER})
"""The safety-off-GPU requests taken because the card was short of memory, rather than to clear it for a model.

Their restores are the ones that have to earn forecast headroom: the memory such a pause returned is part of
what any instantaneous gate then reads, so a restore priced on that alone hands the card back into the pressure
that evicted safety. A whole-card residency's pause instead ends when its own model drains, and its restore is
the liveness path that gives a heavy-resident card its on-GPU safety process back."""

_SAFETY_PLACEMENT_RESTORE_DWELL_FACTOR = 2.0
"""How much longer restore evidence must persist than demotion evidence, as a multiple of the demotion dwell.

Both dwells are seconds derived from the measured cost of one placement flip
(:meth:`ProcessLifecycleManager.safety_readiness_latency_seconds`), never a cycle count: the control loop runs
several times a second, so a run of cycles is a fraction of a second of wall clock against tens of seconds of
safety unavailability, and counting cycles lets a sub-second reading spend that cost. Greater than one because
the asymmetry is deliberate: leave a card that is genuinely short of memory promptly, and come back only once
its room has proven durable, so a readmit does not immediately re-trip on the next heavy job."""

_SAFETY_BACKLOG_PRIORITY_DEPTH = 2
"""Safety backlog depth above which GPU safety restoration is prioritized over placement inertia."""

_SAFETY_RESTORE_PP_BACKLOG_DEPTH = 2
"""Post-processing backlog depth above which a paused safety process is kept off its card.

Deferring restoration while post-processing runs protects the lane's transient device demand from a
concurrent safety (re)load. The bound is a depth rather than mere presence because a worker serving
post-processed requests is rarely without some post-processing work in flight, and an absolute veto would
hand that steady trickle the power to keep safety off the card for the whole run. Matched to
:data:`_SAFETY_BACKLOG_PRIORITY_DEPTH`: a queue of this size is one ordinary job's worth of tail work, while
anything deeper is a lane genuinely under load."""

_SAFETY_RESTORE_PP_BACKLOG_MAX_AGE_SECONDS = 90.0
"""How long a shallow post-processing backlog may defer safety restoration before it stops counting.

Measured from when the post-processing backlog last became non-empty, so an emptying lane resets it and only
continuously-occupied work ages. Long enough to cover the tail of a batch of jobs (the burst this gate exists
for). The headroom evidence remains the term that decides restoration timing; this bound only stops an unbroken
trickle from pinning safety to the CPU indefinitely."""

_DISPATCH_STALL_MIN_SECONDS = 10.0
"""How long the head must be continuously undispatched before the dispatch-stall diagnostic speaks.

Reuses the head-starvation clock so an ordinary one-tick gap between jobs (or a model mid-preload) is
never reported; only a head that has been parked this long, with nothing dispatching, is explained."""

_DISPATCH_STALL_LOG_INTERVAL_SECONDS = 30.0
"""Minimum gap between repeats of the dispatch-stall diagnostic for an unchanged reason, so the
sub-second control loop cannot spam it. A changed reason logs immediately (the stall's cause shifted)."""

_CONTEXT_REDUCTION_MIN_INTERVAL_SECONDS = 60.0
"""Minimum gap between live-context reductions on one card.

Each reduction costs the cold start of the process it stops plus the cold start of the regrowth that unwinds
it, and the head whose peak was rejected asks again every scheduling cycle. Rate-limiting the relief keeps a
head that cannot be admitted from buying one teardown per cycle; it does not change what the reduction does
when it is taken. Paired with the restore dwell, so a reduction and its regrowth cannot chase each other."""

_HEAD_PROTECTION_MAX_STARVE_SECONDS = 120.0
"""How long a parked head may reserve card room from the jobs behind it before the reservation is released.

Head protection holds physical room so the head, not a line-skipper, gets the next opportunity. That is only
worth its cost while the head is actually converging on a dispatch: a head whose own admission keeps
declining otherwise holds an idle card against fitting siblings for as long as the queue lasts, and the
worker serves nothing at all. Set well above the ordinary drain of an in-flight job (the wait the protection
exists to cover) so a normal handoff never trips it, and far below the horizon at which a stalled queue
starts missing the horde's dispatch deadlines."""

_WHOLE_CARD_ESTABLISH_GRACE_SECONDS = 120.0
"""How long after a whole-card residency is established the worker may keep the queue intentionally held
(heavy head deferred while idle siblings stop, safety cycles off-GPU, and the model loads ~11GB) without
the recovery supervisor treating it as a structural wedge. The establishment is deliberately slow now
that it cycles the safety process, so the plain ``_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS`` (20s) window would
otherwise soft-reset the pools mid-setup. Bounded so a residency that genuinely never loads still trips
the supervisor."""

_WHOLE_CARD_DRAIN_SETTLE_SECONDS = 20.0
"""How long after a whole-card teardown reaches sole residency the head waits for the live free-VRAM reading to
confirm the drain before loading best-effort regardless.

A teardown frees the stopped siblings' VRAM asynchronously, so the live measurement can lag or be briefly
unavailable. The live reading dispatches the head the moment it confirms; this bound guarantees the head is
never parked indefinitely on a stuck or missing measurement; once the teardown has been structurally complete
this long it loads on the structural ``fits_alone`` guarantee (the grant precondition for a residency).
Comfortably under ``_WHOLE_CARD_ESTABLISH_GRACE_SECONDS`` so the head always dispatches before the recovery
supervisor would treat the held queue as a structural wedge."""

_WHOLE_CARD_RESTORE_GRACE_SECONDS = 60.0
"""How long after a whole-card residency is *restored* the recovery supervisor keeps ignoring a queue
wedge. Restoring respawns the torn-down sibling inference processes and cycles the safety process back
on-GPU, each a ~20s spawn during which the queue is briefly unservable. Without this grace that churn
looks like a structural wedge and soft-resets the pools, which then cascades into further whole-card
churn and more resets. Covers the respawn window; bounded so a genuine post-restore wedge still trips
the supervisor."""

_POP_CLAIM_RELEASE_VISIBLE_SECONDS = 120.0
"""How long a released pop claim's end is still reported alongside the residency posture.

The reason answers "why is the full pool being advertised again", which is only a live question while the
widening is what the operator is looking at. Long enough to cover the restore churn that usually follows a
release, short enough that a stale reason never sits beside an offer nothing has claimed for minutes."""

_HEAVY_HEAD_LOAD_GRACE_SECONDS = 120.0
"""How long after a heavy head is admitted on the over-budget classification path (for example under
foreign-pressure fit-into-reality, when a model streams even with the whole card to itself) the recovery
supervisor keeps ignoring a queue wedge. Such a head bypasses the whole-card branch, so it is not covered
by ``_WHOLE_CARD_ESTABLISH_GRACE_SECONDS``, yet its multi-gigabyte load equally holds the queue and must
not be mistaken for a structural wedge that faults the never-run backlog. Bounded so a head that genuinely
never loads still trips the supervisor."""

_RAM_RECLAIM_CYCLE_GRACE_SECONDS = 60.0
"""How long after the worker deliberately cycles an idle inference process to reclaim allocator-retained
RAM (``_replace_stale_ram_unload_process``) the recovery supervisor keeps ignoring a queue wedge. The
cycle restarts the slot (a ~20s spawn) and the next head must then preload onto it (another ~20s+), a
window in which the queue is legitimately unservable through no fault of the pool. Without this grace
that deliberate, bounded hold ages past ``_MIN_STRUCTURAL_QUEUE_WEDGE_SECONDS`` (20s) and is mistaken for
a structural wedge, soft-resetting the pools and faulting the perfectly-servable backlog (in a
sole-process configuration this drops every queued job over a window the worker itself created).
Covers the respawn + preload window; bounded so a cycle that genuinely never recovers still trips the
supervisor."""

_HEAD_RAM_DEFER_BARRIER_SECONDS = 60.0
"""How long the head-of-queue preload may be continuously system-RAM-deferred behind live work before the
scheduler latches the head-priority dispatch barrier. Below this the head is treated as ordinarily queued: a
running sibling legitimately holds the memory the head needs, and the RAM branch keeps reclaiming and
re-asking each cycle. Past it, with reclaim freeing nothing and a sibling still holding memory, the head can
never reach the no-live-consumer best-effort admit on its own, so the barrier withholds new dispatch to other
slots and lets the running jobs drain to that escape. Chosen above a normal reclaim-and-retry settle and above
``_RAM_RECLAIM_CYCLE_GRACE_SECONDS`` so a deliberate reclaim cycle is never mistaken for starvation, and short
enough that a genuinely wedged head is unblocked in seconds rather than only when the horde faults its job."""

_HEAD_RAM_DEFER_BARRIER_CAP_SECONDS = 180.0
"""How long the head-priority dispatch barrier may hold before the head is declined outright. New dispatch is
barred while the barrier holds, so the running siblings drain and the head normally admits well inside this
window; if it still cannot be admitted this long after the barrier engaged (a sibling that never completes, or
a head that will not fit even as the card empties), the head is faulted for reissue with retryable semantics
rather than left to rot, and the barrier releases. Three times the engage bound: long enough that draining
siblings win the race in practice, bounded so a permanently unfittable head fails fast."""

_DISPATCH_ANTI_STARVATION_TTL_FRACTION = 0.3
"""Fraction of a queued head's ttl past which resident-model bypass yields so the head's own preload runs.

The affinity skip budget measures from the head's first bypass, so it does not bound the head's total age
since it was popped: a head that waited in queue before its first skip can still be bypassed well past a
winnable window, and by the time it dispatches the horde has aborted it as too slow. This absolute age gate,
anchored at ``time_popped`` and the job's own ttl, closes that gap independently of the skip budget. Sized
below the affinity budget fraction so the head reclaims its slot with enough of the ttl left for its own
staging, sampling, and submission."""

_RETENTION_EVICTION_CONFIRMATION_PASSES = 3
"""Scheduling passes a dispatch waits for its own retention eviction to be evidenced at the device.

A child frees before it reports, so a dispatch that trusted the request rather than the report would load into
memory the card has not returned. Waiting on the evidence costs a tick or two, which is what the reload it
prevents would have cost many times over. The bound is what keeps that wait from becoming a wedge: a child
whose reports never arrive leaves the dispatch to the measured admission gate, which prices it against device
truth every pass, rather than parking the queue on evidence that is not coming."""

_SAFETY_RECOVERY_HOLD_TTL_SECONDS = 120.0
"""How long the safety-recovery admission hold may keep new preloads off a saturated card while the safety
pool crash-loops before it releases. The hold exists to let the card drain so a deferred safety GPU start can
succeed; if the safety pool still cannot start this long after the hold engaged, holding inference longer only
starves the card without helping, so the hold releases and logs CRITICAL plus a ledger event to surface the
stuck condition (the give-up machinery already watches safety-pool health, so this does not re-escalate).
Chosen above a normal drain-and-respawn window and bounded so a permanently stuck safety pool cannot wedge
inference intake."""

_SCHEDULER_DIAGNOSTIC_REPEAT_SECONDS = 30.0
"""Minimum cadence for unchanged high-frequency scheduler diagnostics.

These diagnostics are useful when reconstructing residency and performance behavior, but they sit inside
the scheduler's fast polling loop. Log immediately when the decision state changes, otherwise emit only
periodic reminders with a suppressed-repeat count.
"""

_SCHEDULER_DIAGNOSTIC_MB_BUCKET = 256.0
"""Bucket size for deciding whether memory telemetry changed enough to re-log a scheduler diagnostic."""


# The model-size tier classification (and its baseline value sets) lives in the shared, torch-free
# ``model_sizing`` module so the scheduler and the job popper's large-model pop limiters classify "very large"
# identically. Aliased to the historical private name so the existing references read unchanged.
_ModelSizeTier = ModelSizeTier

_OVERLAP_HEADWAY_MIXED_HEAVY = 0.5
"""Fraction of the in-flight job's sampling that must be done before a concurrent job joins it when
exactly one side of the overlap is heavy (e.g. an SDXL is running and a cheaper SD1.5 wants to join,
or vice versa). Gives the heavier job room to get past its memory-hungry startup before another
sampler adds pressure."""

_OVERLAP_HEADWAY_BOTH_HEAVY = 0.75
"""Fraction of the in-flight job's sampling that must be done before a *second heavy* job joins it.
Two SDXL jobs stacking their weight loads and activation peaks is the over-subscription that thrashes
a sampler into a watchdog teardown, so the running job must be most of the way done first."""

_OVERLAP_HEADWAY_AMPLE_VRAM = 0.15
"""Headway applied instead of the mixed/both-heavy fractions when the device's measured free VRAM
absorbs the candidate's full predicted sampling peak plus the configured reserve.

The strict fractions price every card as tight; on a high-VRAM card serving a heavy-only queue that
prices a second configured thread out of existence (a both-heavy candidate waits for 75% progress, so
two threads converge to ~one effective thread). When the measurement says the newcomer's whole peak
fits *now*, the over-subscription the strict headway guards against cannot occur; a small headway is
kept so the running job clears its memory-hungry startup before a sibling adds pressure."""

_OVERLAP_HEADWAY_SCALE_HIGH_PERFORMANCE = 0.5
"""Multiplier applied to the required overlap headway when the worker runs in high-performance mode.

High-performance operators have provisioned the card for aggressive co-sampling and want the next job's
sampling to overlap the tail of the current one sooner. Halving the headway brings the newcomer in
earlier while the VRAM arbiter still independently decides whether the card can hold the overlap."""

_OVERLAP_HEADWAY_SCALE_MODERATE_PERFORMANCE = 0.75
"""Multiplier applied to the required overlap headway in moderate-performance mode: a milder pull-in than
high-performance mode, still gated by the arbiter's memory verdict."""


def _performance_mode_headway_scale(bridge_data: reGenBridgeData) -> float:
    """Return the overlap-headway multiplier for the worker's performance mode (1.0 outside the fast modes).

    Higher performance modes shrink the sampling headway a newcomer must wait for, so concurrent inference
    starts sooner. The memory arbiter still gates whether the overlap fits, so this only moves *when* an
    admissible overlap begins, never *whether* an over-committing one is allowed.
    """
    if bridge_data.high_performance_mode:
        return _OVERLAP_HEADWAY_SCALE_HIGH_PERFORMANCE
    if bridge_data.moderate_performance_mode:
        return _OVERLAP_HEADWAY_SCALE_MODERATE_PERFORMANCE
    return 1.0


class VaeLanePauseRequester(enum.StrEnum):
    """Which subsystem is asking to pause the VAE lane, and therefore which decode-drain rule applies.

    The value is the log-facing subsystem name. The distinction is not cosmetic: it selects how much of the
    disaggregated pipeline counts as "still needs this lane" in
    :meth:`InferenceScheduler._vae_lane_pause_deferred_for_decode`.
    """

    RECLAIM_LADDER = "Reclaim ladder"
    """Pausing to relieve VRAM pressure now; only a queued or in-flight decode withholds the pause."""
    WHOLE_CARD_RESIDENCY = "Whole-card residency"
    """Pausing to clear a card it cannot claim until in-flight work drains; a sampling job withholds it too."""


class _WholeCardDemandOutcome(enum.Enum):
    """How the whole-card residency decision resolves a budget-gated head's preload.

    Returned by :meth:`InferenceScheduler._decide_whole_card_demand` so the budget-admission orchestrator
    can map each outcome to proceed/defer without re-deriving the residency state.
    """

    FALL_THROUGH = enum.auto()
    """No whole-card reservation applies (not demanded, or declined as untrustworthy); continue to the
    ordinary VRAM/RAM verdict."""
    PRESTAGE = enum.auto()
    """The head's weights are pre-staging into spare RAM while a live job drains; skip the verdict and
    send the preload now (convergence collapses the card to sole residency before it samples)."""
    DEFER = enum.auto()
    """The reservation is mid-teardown (idle siblings stopping, safety cycling off-GPU, freed VRAM
    draining); defer this cycle and re-evaluate against the reduced topology next tick."""


@dataclass(frozen=True)
class _WholeCardGovernorHold:
    """A churn governor's refusal to open a new whole-card residency on one card, with its arithmetic."""

    governor: WholeCardGovernor
    """Which governor refused, used to key the throttled disclosure so a changed objection always speaks."""
    reason: str
    """Operator-facing sentence naming the refusal; also the recorded defer reason.

    Stable while the same refusal persists: the recorded reason is compared across ticks by the stall-line
    throttle and across recovery settling windows by the remedy-relevance judgement, so figures that tick
    (spend, countdowns) belong in :attr:`detail`, never here."""
    detail: str | None = None
    """The refusal's current arithmetic (spend, remaining allowance, replenish wait), for disclosure only."""


class _PreloadJobOutcome(enum.Enum):
    """What one pending job's preload attempt means for the rest of this scheduling pass."""

    NEXT_JOB = enum.auto()
    """This job needs nothing (or was faulted); consider the next pending job."""
    STOP_PASS = enum.auto()
    """A gate deferred or consumed this cycle (RAM floor, no slot, serialization, budget); stop the pass."""
    PRELOAD_SENT = enum.auto()
    """A preload was issued for this job; the pass is done and reports success."""


@dataclass
class _PreloadActuation:
    """The head-preload context a described actuation needs when the adapter runs a deferred verdict.

    An EVICT_IDLE_MODEL or REDUCE_LIVE_CONTEXTS command targets the card on behalf of the specific head being
    adjudicated: the eviction spares the head's own target slot, and the reduction establishes whole-card
    residency for the head's job at the depth the verdict's rejected peak sized. The adapter records this for
    the current head immediately before running the verdict's commands and clears it once they have run.
    """

    job: ImageGenerateJobPopResponse
    available_process: HordeProcessInfo
    forecast: StreamForecast
    max_resident: int | None


@dataclass(frozen=True)
class _SafetyPlacementInputs:
    """Represents the per-card evidence the runtime safety-placement policy decides one cycle from.

    Every term is about the card safety occupies (or the card it would land on while it is off-GPU), because
    cards are independent VRAM domains and a sibling card's sampling says nothing about this one. Gathered once
    per cycle by :meth:`InferenceScheduler._safety_placement_inputs` so the demotion predicate, the restore
    forecast, and the diagnostics cannot read different pictures of the same card.
    """

    device_index: int | None
    measured_free_mb: float | None
    marginal_need_mb: float
    noise_buffer_mb: float
    governor_state: GovernorState
    safety_footprint_mb: float
    reclaimable_idle_mb: float = 0.0
    """Device memory (MB) idle inference processes on the card hold as retained residents.

    Those weights are kept warm between jobs on a grant the reclaim ladder and the clearance gate revoke the
    moment a peak needs the room, so for a placement decision they are room the card can produce within a
    tick, not room it lacks. Counting them as used is what armed a demotion for as long as retention held a
    resident, and what kept the restore forecast from ever passing on a card that retains between jobs."""

    def available_mb(self) -> float | None:
        """The measured free plus what idle retained residents would return; None without a measurement."""
        if self.measured_free_mb is None:
            return None
        return self.measured_free_mb + self.reclaimable_idle_mb

    def restore_requirement_mb(self) -> float:
        """The available room the card must show for safety to survive the peak it is committed to."""
        return self.safety_footprint_mb + self.marginal_need_mb + self.noise_buffer_mb

    def describe(self) -> str:
        """Return the evidence as one diagnostic clause, for the placement log lines."""
        free_display = "unreported" if self.measured_free_mb is None else f"{self.measured_free_mb:.0f}MB"
        return (
            f"card {self.device_index}: measured free {free_display}, reclaimable idle residents "
            f"{self.reclaimable_idle_mb:.0f}MB, marginal need "
            f"{self.marginal_need_mb:.0f}MB, noise buffer {self.noise_buffer_mb:.0f}MB, safety footprint "
            f"{self.safety_footprint_mb:.0f}MB, governor {self.governor_state.name}"
        )


@dataclass(frozen=True)
class _MaterializationOutcome:
    """The result of pricing a job's VRAM materialisation through the MONOLITHIC_DISPATCH arbiter identity.

    Returned by :meth:`InferenceScheduler._evaluate_materialization_admission` so the dispatch and clearance
    gates share one admission core while each keeps its own hold bookkeeping and diagnostics. Carries the
    arbiter verdict plus the terms the callers' decision-sink lines report, and whether the non-admit path
    already routed an eviction through the single reclaim owner (so the caller attributes the hold's release to
    reclaim versus the card freeing on its own).
    """

    verdict: VramVerdict
    candidate_delta_mb: float | None
    device_index: int | None
    actuations_requested: tuple[ActuatorCommand, ...]
    actuations_applied: tuple[ActuatorCommand, ...]


@dataclass
class _PendingRetentionEviction:
    """A retained resident whose eviction a dispatch issued and is still waiting to see land on the card.

    Retention tracking clears the instant the unload is sent, but the bytes come back only when the child has
    actually freed them. Holding the issuing dispatch against this record is what stops it racing its own
    eviction: it carries the card's free reading and the slot's reported reservation as they stood when the
    unload went out, so the child's post-free reports are what release the hold.

    The wait is bounded by :data:`_RETENTION_EVICTION_CONFIRMATION_PASSES` scheduling passes. Evidence is
    preferred, but an absence of evidence must never park the queue: past the bound the record is dropped and
    the measured admission gate, which prices every dispatch against device truth on each pass, is what stands
    between the job and the card.
    """

    model: str
    reserved_baseline_mb: float | None
    device_free_baseline_mb: float | None
    passes_waited: int = 0


def _preload_outcome_from_admission(decision: AdmissionDecision) -> _PreloadJobOutcome:
    """Map the public admission decision vocabulary onto the scheduler pass control enum."""
    match decision:
        case AdmissionDecision.ADMIT | AdmissionDecision.PRESTAGE:
            return _PreloadJobOutcome.PRELOAD_SENT
        case (
            AdmissionDecision.NEXT_JOB
            | AdmissionDecision.QUARANTINED
            | AdmissionDecision.UNSERVICEABLE
            | AdmissionDecision.ALREADY_LOADED
        ):
            return _PreloadJobOutcome.NEXT_JOB
        case _:
            return _PreloadJobOutcome.STOP_PASS


class InferenceScheduler:
    """Owns model preloading, inference start, and model unloading logic."""

    _state: WorkerState
    _process_map: ProcessMap
    _horde_model_map: HordeModelMap
    _component_residency_map: ComponentResidencyMap | None
    _job_tracker: JobTracker
    _process_lifecycle: ProcessLifecycleManager
    _runtime_config: RuntimeConfig
    _model_metadata: ModelMetadata
    _max_threads_ceiling: int
    _max_inference_processes: int
    _lru: LRUCache
    _performance_model: PerformanceModel | None

    _preload_delay_notified: bool
    _model_recently_missing: bool
    _model_recently_missing_time: float
    _pending_line_skip: NextJobAndProcess | None
    _model_last_in_demand: dict[str, float]
    _vram_budget: VramBudget
    _ram_budget: RamBudget
    _reserve_ledger: CommittedReserveLedger
    _ram_budget_defer_notified: bool
    _ram_pressure_notified: bool
    _scheduler_diagnostic_log_state: dict[str, tuple[tuple[object, ...], float, int]]
    _last_budget_defer_reason: str | None
    _context_reduction_at: dict[int | None, float]
    _last_preload_admission: LatestPreloadAdmission | None
    _post_processing_lane_commitments_provider: Callable[[], int]
    _pool_protected_models_provider: Callable[[], frozenset[str]]
    _on_pool_pressure_eviction: Callable[[str], None]

    def __init__(
        self,
        *,
        state: WorkerState,
        process_map: ProcessMap,
        horde_model_map: HordeModelMap,
        component_residency_map: ComponentResidencyMap | None = None,
        job_tracker: JobTracker,
        process_lifecycle: ProcessLifecycleManager,
        runtime_config: RuntimeConfig,
        model_metadata: ModelMetadata,
        card_runtimes: dict[int, CardRuntime] | None = None,
        max_concurrent_inference_processes: int,
        max_inference_processes: int,
        lru: LRUCache,
        performance_model: PerformanceModel | None = None,
        reserve_ledger: CommittedReserveLedger | None = None,
        post_processing_lane_commitments_provider: Callable[[], int] | None = None,
        pool_protected_models_provider: Callable[[], frozenset[str]] | None = None,
        on_pool_pressure_eviction: Callable[[str], None] | None = None,
        decision_sink: DecisionSink | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialize the scheduler with references to the components it needs to manage.

        Args:
            state (WorkerState): The worker's state object, containing all of the mutable flags
                relating to the worker's active state and lifecycle.
            process_map (ProcessMap): The worker's ProcessMap, which tracks all active processes and
                their states.
            horde_model_map (HordeModelMap): The worker's HordeModelMap, which tracks the load state of all models
                and which processes they are loaded on.
            component_residency_map (ComponentResidencyMap | None): The per-process view of components staged in
                each child's RAM cache. Read by the RAM-pressure component-eviction rung to reclaim idle,
                unprotected staged components before the coarser whole-RAM unload. ``None`` (unit tests, and a
                worker whose budgeted cache is disabled) skips that rung and takes the legacy path unchanged.
            job_tracker (JobTracker): The worker's JobTracker, which tracks all jobs in-flight
                and is responsible for managing their state transitions.
            process_lifecycle (ProcessLifecycleManager): The worker's ProcessLifecycleManager, which is responsible
                for launching, monitoring, and killing processes as needed.
            runtime_config (RuntimeConfig): Holds the current bridge configuration snapshot.
            model_metadata (ModelMetadata): Provides lookups against the stable-diffusion model reference.
            card_runtimes (dict[int, CardRuntime] | None): The per-card runtime plan, keyed by stable device
                index, used to route a job to a card that can serve it on a multi-GPU host. ``None`` or a
                single entry means single-GPU: dispatch takes the original card-agnostic path unchanged.
            max_concurrent_inference_processes (int): The maximum number of inference processes to run at once.
            max_inference_processes (int): The maximum number of inference processes to have launched at once,
                including those that are preloading or downloading models.
            lru (LRUCache): The worker's LRU cache, used to track recently used models for unloading decisions.
            performance_model (PerformanceModel | None): Supplies an expected sampling time per dispatched
                job for the audit ledger (and, in a later phase, slow-job remediation). May be ``None``.
            reserve_ledger (CommittedReserveLedger | None): The shared committed-VRAM/RAM ledger every
                workload flow contributes to, so image generation and alchemy cannot independently admit
                against the same free VRAM. When ``None`` (unit tests driving the scheduler alone) a private
                ledger is created, so the scheduler still accounts for its own post-processing reserve.
            post_processing_lane_commitments_provider (Callable[[], int] | None): Optional count of
                non-JobTracker work already committed to the shared post-processing lane, such as
                graph-backed alchemy forms waiting for or running on that lane.
            pool_protected_models_provider (Callable[[], frozenset[str]] | None): Optional source of the
                fixed model pool's currently-seated models. Their weights gain residency protection so the
                scheduler does not evict what the pool is advertising, except under true RAM pressure where a
                seat with no live job yields. ``None`` (unit tests, and a worker running no pool) reads an
                empty set, so seat protection is inert and residency behaves exactly as before the pool existed.
            on_pool_pressure_eviction (Callable[[str], None] | None): Optional callback invoked with each
                seated idle model whose staged components the RAM-pressure rung yields, so the manager can tell
                the pool a seat was reclaimed. ``None`` (unit tests, and until wired) is a no-op.
            decision_sink (DecisionSink | None): Optional callback the manager injects to record
                dispatch-residency admission decisions (holds and their release) to the stats export.
                ``None`` in unit tests and until wired; emission is a no-op then.
            clock (Callable[[], float] | None): The wall-clock source for every scheduling and governance
                window the scheduler owns: residency grace and cooldown, the churn governors' rolling windows
                and their deferral dwell, head starvation and the dispatch barriers, and the reclaim rate
                limits. It is injectable so a harness can drive those windows on its own timeline instead of
                having to spend the real seconds they are sized in. Timestamps shared with other components (a
                process's spawn time, IPC report ages, action-ledger entries) stay on the real clock, since
                they are compared against stamps this scheduler does not write. ``None`` reads the real clock
                through a fresh lookup on every call, so a test that patches ``time.time`` still governs it.
        """
        self._clock: Callable[[], float] = clock if clock is not None else lambda: time.time()
        self._state = state
        self._process_map = process_map
        self._horde_model_map = horde_model_map
        self._component_residency_map = component_residency_map
        self._job_tracker = job_tracker
        self._process_lifecycle = process_lifecycle
        self._runtime_config = runtime_config
        self._model_metadata = model_metadata
        self._post_processing_lane_commitments_provider = post_processing_lane_commitments_provider or (lambda: 0)
        # The fixed model pool's seat set and the pool's pressure-eviction notifier. Both default inert so a
        # worker running no pool (and every standalone unit test) keeps the pre-pool residency behaviour.
        self._pool_protected_models_provider = pool_protected_models_provider or (lambda: frozenset())
        self._on_pool_pressure_eviction = on_pool_pressure_eviction or (lambda _model_name: None)
        # Seat models whose staged components the RAM-pressure rung has already yielded, so the pool notifier
        # and the summary log fire only on the rising edge of a seat's yield rather than every pressure tick.
        self._pool_pressure_yielded_models: frozenset[str] = frozenset()
        # Injected by the manager: records dispatch-residency admission decisions (holds and their release)
        # to the stats export, coalesced on the receiving side. None in unit tests and until wired.
        self._decision_sink = decision_sink
        # Per-card runtime plan for multi-GPU routing. A single entry (or None) means single-GPU, where the
        # dispatch path stays card-agnostic and byte-identical to before multi-GPU existed.
        self._card_runtimes: dict[int, CardRuntime] = card_runtimes if card_runtimes is not None else {}
        # The constructor value is the provisioned ceiling; the *live* concurrent cap is read from
        # the runtime config (see the _max_concurrent_inference_processes property) so it can change
        # at runtime without resizing the inference semaphore.
        self._max_threads_ceiling = max_concurrent_inference_processes
        self._max_inference_processes = max_inference_processes
        self._lru = lru
        self._performance_model = performance_model
        # Optional sink for between-jobs reload/respawn events, set by the manager to
        # WorkerRunMetrics.record_churn. None in unit tests that drive the scheduler directly.
        self._churn_observer: Callable[[ChurnKind], None] | None = None

        # The ledger-driven admission overlay. The baseline provider yields the reconciler's measured
        # shared-device baseline (MB) per card, wired by the manager (None until wired, and in standalone unit
        # tests: the overlay then reads baseline 0, so capacity is the raw total and the measured gate matches
        # the predictive gate). The per-card counters/headroom feed run-metrics calibration visibility.
        self._admission_baseline_provider: Callable[[int | None], float | None] | None = None
        self._admission_denials_by_device: dict[int, int] = {}
        self._admission_headroom_mb_by_device: dict[int, float | None] = {}
        # The device-free governor's growth hold per card, set each tick by the parent. True while a card is at
        # PRESSURE or SATURATED (device-level free VRAM below the soft floor): the scheduler must not grow the
        # card's VRAM footprint (no new model brought to VRAM on a process that does not already hold it, no
        # safety GPU restore, no paused-lane restart). In-flight sampling is never touched by this hold; it is
        # a truthful WDDM-cliff brake, orthogonal to the ledger admission gate. Empty (no hold) in standalone
        # unit tests, where the parent never wires the governor.
        self._vram_growth_hold_by_device: dict[int, bool] = {}
        # The device-free governor's committed state per card, pushed each governor tick by the parent alongside
        # the growth hold. Retention reads the STATE (not the derived hold boolean): weights only stay resident
        # while the card is HEALTHY, since a PRESSURE or SATURATED card is one the verified reclaim ladder is or
        # may soon be taking residents back from. Empty (defaults to HEALTHY) in standalone unit tests, where the
        # parent never wires the governor.
        self._governor_states_by_device: dict[int, GovernorState] = {}
        # Count of verified reclaim-ladder shortfalls the engine reported (a rung freed less than half its
        # promised device memory). Recorded here as a calibration counter: at reclaim time the freed figure is
        # not a footprint peak and no complete (baseline, resolution, stage) key is reconstructable, so the
        # raise-only footprint store does not apply and the signal is kept as a count. Calibration visibility.
        self._reclaim_calibration_events = 0
        # The single VRAM arbiter, injected by the manager. It is the live authority for the gated
        # preload/overlap/disaggregation/post-processing seams, pricing each demand against the cycle-frozen
        # measurement. None until wired (and in standalone unit tests), where those seams fall back to their
        # measured floors.
        self._vram_arbiter: VramArbiter | None = None
        # Whether the arbiter above is one this scheduler made for itself rather than the manager's. Nothing
        # else freezes a private arbiter's cycle, so the scheduler must re-freeze it from live state on each
        # consult; an injected one is driven by the control loop and must be left on the cycle it froze.
        self._owns_private_vram_arbiter = False
        # The worker's single verified reclaim ladder, injected by the manager. The scheduler books a live-
        # context reduction with it as a restore obligation, so the engine that already owns the LIFO unwind of
        # ladder-issued pauses also owns growing the inference pool back when the card recovers. None until
        # wired (and in standalone unit tests), where a reduction is simply not booked.
        self._reclaim_ladder: VerifiedReclaimLadder | None = None
        # The truthful per-card device-free reading source, injected by the manager (parent NVML). The
        # manager-driven cycle passes its explicit reading map to build_vram_arbiter_snapshot; this provider is
        # the fallback for a self-primed snapshot (a scheduler consult before or outside a manager tick), so the
        # measured-truth identity keeps its primary input there too. None (unwired) leaves the reading absent,
        # and admission defers with the missing-reading diagnostic.
        self._device_free_mb_provider: Callable[[int], float | None] | None = None
        self._available_ram_mb_provider: Callable[[], float] | None = None
        # The head-preload context the current deferred verdict's actuations act on, set immediately before the
        # adapter runs a verdict's commands and cleared once they have. None outside that window.
        self._preload_actuation: _PreloadActuation | None = None

        # Dispatch-time residency reconciliation state. The dispatch gate re-uses the arbiter's
        # MONOLITHIC_DISPATCH identity to check that a staged job's VRAM materialisation fits the card before it
        # is handed to a child (the moment RAM-staged weights actually commit to VRAM). A conflicting verdict
        # holds the dispatch (the job keeps its queue position, never faulted) and routes idle-resident eviction
        # through the single reclaim owner. The per-job map stamps when each held job first held, so a release is
        # attributed to reclaim (this gate emitted eviction commands for it) versus natural free (device-free
        # recovered on its own); the counters are calibration visibility only.
        # Session tallies for the staged-job admission cap: how often each measurement held the cap at the
        # sampling-slot count, and the reason last logged with when it was logged and how many identical
        # deferrals were suppressed behind it. Read into the duty-cycle summary, so a duty figure short of
        # target can be read across to the clause leaving spare processes idle.
        self._staging_defers: dict[StagingDeferReason, int] = {}
        self._staging_defer_log_state: tuple[StagingDeferReason, float, int] | None = None

        self._dispatch_hold_since: dict[str, float] = {}
        self._dispatch_hold_reclaim_requested: set[str] = set()
        self._dispatch_reconciliation_holds = 0
        self._dispatch_reconciliation_conflicts = 0
        self._dispatch_reconciliation_hold_seconds = 0.0
        self._dispatch_reconciliation_released_by_reclaim = 0
        self._dispatch_reconciliation_released_by_natural_free = 0

        # When the exclusive-admit dispatch hold last disclosed itself, per card scope. The hold is re-evaluated
        # every dispatch selection, so the notice is throttled to keep a sustained hold from repeating a line
        # the operator has already read.
        self._exclusive_suppression_logged_at: dict[int | None, float] = {}

        # The bounded affinity line-skip window for the currently-tracked displaced head. Resident-model jobs
        # may pass a cold FIFO head only while it is inside this window (a wall-clock budget derived from the
        # job ttl plus a hard skip ceiling). The window advances only on committed dispatch (see
        # :meth:`start_inference`), so the information_only look-ahead never mutates it; dispatch selection reads
        # it purely. When the window is spent the head reclaims the slot via the same fall-through as a
        # non-forecast head. Calibration visibility only.
        self._affinity_skip_state = AffinitySkipState()
        # The head id whose anti-starvation age override was last logged, so the "resident bypass yields"
        # notice is edge-triggered (once per head) rather than repeated every scheduling cycle it engages.
        self._anti_starvation_logged_head_id: str | None = None

        # How many jobs this session were seated ahead of the queue head because a slot already retained their
        # weights, and the (model, slot) pair last logged for it so the notice is edge-triggered rather than per
        # cycle. Counted on the dispatch that seats one, so the figure is jobs served without an upload rather
        # than cycles the ordering was consulted in, and readable beside the reload-churn figures a duty window
        # reports.
        self._retention_affinity_reorders = 0
        self._retention_affinity_logged_edge: tuple[str | None, int] | None = None

        # The models most recently dispatched to each inference slot, newest first, which is the evidence
        # retention is granted on. Kept scheduler-side rather than on the process record because it describes
        # the slot's traffic rather than one process's residency: it must outlive every job boundary, and a
        # slot whose child was replaced is still serving the same shape of work. One entry deeper than the
        # window searched, so a live grant's own dispatch can be excluded and the revoke sweep asks exactly the
        # question issuance asked.
        self._slot_dispatch_history: dict[int, deque[str]] = {}
        # When each card was last seen off HEALTHY, so a revoke sweep can require the pressure to have held
        # rather than fire on a sample. Cleared the moment the governor commits HEALTHY again.
        self._governor_pressure_since: dict[int, float] = {}
        # Retention outcome tallies. Grants and their refusal reasons say what the policy decided; reuses and
        # unused evictions say what those decisions were worth, which is the pair a grant count alone cannot
        # give: a retained copy evicted before any successor arrives costs the card and saves nothing.
        self._retention_grants_issued = 0
        self._retention_grant_denials: dict[RetentionDenialReason, int] = {}
        self._retention_reuses = 0
        self._retention_evicted_unused = 0
        self._retention_revokes = 0
        # Reorders the admissibility condition refused, so a session that reorders nothing says the gate is
        # what held rather than only that nothing happened.
        self._retention_reorder_pareto_vetoes = 0
        # Wall-clock stamp of the last empty-affinity-scan trace, so the per-cycle scan logs its emptiness at
        # most once per throttle window while retained weights exist to protect.
        self._affinity_scan_trace_last = 0.0

        # The set of job ids whose dispatch the post-processing co-residency gate held on the pass that last
        # evaluated them. The gate computes this verdict during dispatch (``_should_defer_dispatch_for_post_
        # processing``); recording it here lets the read-only stall classifier name the same hold instead of
        # re-deriving the PP arithmetic, so the classifier and the gate can never disagree. Membership is set
        # or cleared each time the gate is consulted for a head, and abandoned entries (jobs no longer pending)
        # are pruned alongside the reconciliation holds.
        self._post_processing_defer_holds: set[str] = set()

        # The set of in-progress (staged, primed) job ids the clearance gate is currently holding: their
        # diffusion weights would over-commit the card at the clearance VRAM moment, so the parent withholds
        # the clearance grant while the single reclaim owner evicts idle residents. Recorded so the slot-duty
        # classifier attributes the empty sampling slot to ``CLEARANCE_HOLD`` and pruned by omission once a job
        # leaves the in-progress set. Only used under the clearance lease; empty otherwise.
        self._clearance_hold_ids: set[str] = set()

        # The learned-footprint store, injected by the manager (one shared instance, the same the message
        # dispatcher observes into). Admission pricing of a job's sampling peak reads it so a measured
        # activation high-water raises the static per-model seed the predictor returns; a static seed
        # systematically undershoots the reserved peak (calibration saw ~11GB against a 6158MB seed). None
        # until wired (and in standalone unit tests), where every estimate falls back to the static seed.
        self._footprint_store: LearnedFootprintStore | None = None

        # Pipeline-disaggregation hooks, wired by the manager via set_disaggregation_hooks. The predicate
        # decides whether a job takes the disaggregated path (so its verdicts charge the UNet-only sampler
        # figure); the router registers an eligible job with the orchestrator, pinned to the process it was
        # scheduled onto, in place of the monolithic START_INFERENCE. Defaults keep the scheduler on the pure
        # monolithic path (every job disaggregated=False) for unit tests that drive it alone.
        self._is_disaggregatable_job: Callable[[ImageGenerateJobPopResponse], bool] = lambda _job: False
        # The stable class-eligibility predicate (no liveness/residency coupling): forecasting and VRAM
        # charging use this, so a job that *will* run disaggregated is always priced sampler-only, even during
        # a whole-card window when the lane is transiently paused. Defaults monolithic for standalone tests.
        self._is_disaggregation_class_eligible: Callable[[ImageGenerateJobPopResponse], bool] = lambda _job: False
        # The process argument is None for a job staged ahead of a pin: its model's only copy is on a lane
        # pinned to the job in front of it, so it is admitted with the sampler unresolved and binds when that
        # pin releases.
        self._register_disaggregated_job: (
            Callable[[ImageGenerateJobPopResponse, HordeProcessInfo | None], Awaitable[bool]] | None
        ) = None
        # Read-only disaggregation diagnostics for the dispatch-stall classifier: the job pinning a given
        # process as its sampler, and the current in-flight sampling peaks. Defaults (no owner, empty peaks)
        # keep the classifier's disaggregation branch inert for standalone tests that never wire them.
        self._disaggregation_pin_owner: Callable[[int], str | None] = lambda _pid: None
        self._disaggregation_sampling_peaks: Callable[[], dict[str, float]] = dict
        # How many disaggregated jobs currently need the VAE lane for a decode (queued or dispatched to it),
        # wired by the manager from the orchestrator. The reclaim-ladder VAE-lane pause reads it so the lane is
        # not stopped out from under a decode whose sample already finished (which would reroute the job
        # monolithic and discard that sampling). Defaults to zero so a standalone scheduler never withholds a
        # pause for a decode it has no orchestrator to see. ``_vae_pause_deferred_for_decode`` edge-latches the
        # one INFO line so a sustained pressure run logs the deferral once, re-arming when a pause next proceeds.
        self._vae_decode_pending_count: Callable[[], int] = lambda: 0
        # The same wiring one stage wider: jobs sampling or awaiting decode, i.e. everything still bound for the
        # lane. The whole-card residency reads this instead, because it cannot claim the card until that work
        # drains anyway, so pausing under a sampling job only costs the sample. See VaeLanePauseRequester.
        self._vae_lane_bound_job_count: Callable[[], int] = lambda: 0
        # Staging a pin-waiting head ahead of its sampler starts its text encode against the sample in front of
        # it instead of after it. The encode charge it books at register is returned at bind (covered by the
        # reservation tests); the fake harness cannot reach the staging path with a single served model, since
        # `resolve_card_concurrency` collapses that pool to one process, so a multi-model harness case is the
        # remaining prediction to add. Kept as a switch so the path can be held off without a code change.
        self._stage_ahead_of_pin_enabled: bool = True
        self._vae_pause_deferred_for_decode = False

        # Runtime safety-placement evidence (see _reconcile_runtime_safety_placement). Each is the clock time
        # the corresponding condition has held continuously for, or None while it does not hold: pressure on
        # safety's own card while safety is resident, and forecast headroom on the chosen card while safety is
        # off it. The policy's verdict is derived from these against a dwell measured in seconds, so no intent
        # can outlive the evidence that produced it and a sub-second run of cycles cannot spend a flip that
        # costs tens of seconds of safety unavailability.
        self._safety_placement_pressure_since: float | None = None
        self._safety_placement_headroom_since: float | None = None
        # The placement inputs last logged, so the diagnostic speaks on a state change and repeats at TRACE.
        self._safety_placement_last_logged_inputs: tuple[object, ...] | None = None
        # The reclaim ladder files a one-shot safety placement request here. The recurring placement reconciler
        # is the only code allowed to turn that request into a lifecycle pause or restore, so reclaim,
        # residency, and fit hysteresis cannot issue overlapping safety rebuilds.
        self._safety_reclaim_pause_requested = False
        # When the post-processing backlog last became non-empty, or None while it is empty. Ages the
        # restore-side post-processing bound so an unbroken trickle of shallow post-processing cannot keep a
        # paused safety process off its card for the whole run.
        self._safety_restore_pp_backlog_since: float | None = None
        # Lifetime counts of runtime safety-placement policy actuations, for the run-metrics readback: a
        # demotion moves safety off-GPU (its charge did not fit beside the sampler), a promotion restores it
        # once the chosen card's measured free proved durable room. These count only policy-initiated moves,
        # not the whole-card residency's own safety pauses (which the lifecycle manager counts separately).
        self._safety_placement_demotions = 0
        self._safety_placement_promotions = 0

        self._preload_delay_notified = False
        self._model_recently_missing = False
        self._model_recently_missing_time = 0.0
        self._pending_line_skip = None
        self._model_last_in_demand = {}

        # Constructed with safe defaults; the live reserves are synced from the (reloadable) config each
        # scheduling cycle by _vram_budget_active(), which also tolerates partially-mocked test config.
        self._vram_budget = VramBudget(reserve_mb=_DEFAULT_VRAM_RESERVE_MB)
        self._ram_budget = RamBudget(reserve_mb=_DEFAULT_RAM_RESERVE_MB)
        self._reserve_ledger = reserve_ledger if reserve_ledger is not None else CommittedReserveLedger()
        self._ram_budget_defer_notified = False
        self._ram_pressure_notified = False
        self._last_budget_defer_reason = None
        self._context_reduction_at = {}
        # Credited RAM admissions awaiting the measured-truth reconciliation, keyed by target process id.
        self._pending_reuse_credits: dict[int, _ReuseCreditRecord] = {}
        # Last credited-admission log key, so an unchanged credited admit is not re-logged (edge-triggered).
        self._last_credited_admission_key: tuple[int, str | None, int] | None = None
        # Torch-free component-charge plumbing for disaggregation-class RAM staging. The checkpoint path per
        # model is stable once the reference is loaded, so it is resolved once and cached; the component-
        # identity sidecar is cached per model keyed on the checkpoint's on-disk size, so a replaced checkpoint
        # re-reads while an unchanged one is not re-parsed every sub-second cycle. The weights root is resolved
        # once (a filesystem walk) and reused. The fallback-logged set gates the missing-sidecar debug notice to
        # once per model so the loop cannot spam it.
        self._checkpoint_path_cache: dict[str, Path | None] = {}
        self._component_sidecar_cache: dict[str, tuple[int, ComponentIdentitySidecar]] = {}
        self._component_charge_fallback_logged: set[str] = set()
        self._weights_root: Path | None = None
        self._scheduler_diagnostic_log_state = {}
        self._last_preload_admission = None
        # One-shot log throttle, keyed by model, for the "held back as locally unservable" notice.
        self._unservable_admit_notified: dict[str, bool] = {}
        # Sustained per-card minimum of measured foreign (non-worker) VRAM usage, folded into each device
        # state's achievable ceiling so a model that can never fit even an emptied card is DENIED rather than
        # deferred forever. Driven once per snapshot build from measured device truth; see
        # build_vram_arbiter_device_state. The tracker runs on a monotonic clock exposed via
        # _foreign_floor_clock so tests can hand-advance the observation window.
        self._foreign_vram_floor = ForeignVramFloorTracker()
        self._foreign_floor_clock: Callable[[], float] = time.monotonic
        # The job tracker lifts a conditional ceiling hold by re-reading the card's current achievable ceiling;
        # wire that read here (the scheduler owns the foreign-floor tracker) so the tracker's live predicate
        # need not reach back into the scheduler.
        self._job_tracker.set_achievable_ceiling_provider(self.achievable_ceiling_mb)
        # One-shot log throttle, keyed by model, for the "declined a whole-card residency" notice (a teardown
        # demand the warrant gate did not trust; see _whole_card_warranted / _log_whole_card_declined).
        self._whole_card_declined_notified: dict[str, bool] = {}
        # The same one-shot throttle for the "refused a process-count reduction" notice (a reduction with no
        # context left to reclaim; see WholeCardResidencyMachine.residency_demanded).
        self._whole_card_reduction_suppressed_notified: dict[str, bool] = {}
        # Per-context VRAM overhead model: owns the startup-measured per-process and marginal context costs
        # and derives the figures the streaming forecast needs (see ContextOverheadModel). The manager feeds
        # it the probe measurements via set_measured_*, and its attribution tick feeds the truthful
        # NVML-derived bare-context readings via capture_idle_context_residency /
        # invalidate_idle_context_floor (per-child VRAM views are per-process artefacts under WDDM and are
        # never decomposed as device truth).
        self._overhead = ContextOverheadModel()
        # Whole-card exclusive-residency records, keyed by the device index a residency is held on. A heavy
        # model claims a card by stopping that card's idle sibling contexts (and cycling safety off-GPU on the
        # safety card); keying per card lets two heavy models on different cards each hold their own residency.
        # A single-GPU worker uses exactly one entry under the None key, identical to the prior scalar fields.
        # All reads/writes go through the ledger (see WholeCardResidencyLedger; _residency_state delegates).
        self._whole_card_ledger = WholeCardResidencyMachine()
        # The per-tick resource governor: the process manager ticks it once per control-loop iteration via
        # run_governance_tick(), independent of queue depth, so governance never depends on a particular
        # scheduling path executing (or on the inference queue being non-empty). It owns the RAM governor's
        # multi-tick bookkeeping (shed cards, draining processes), exposed under the historical attribute
        # names through the _ram_pressure_shed_cards / _processes_draining_for_ram properties.
        self._governor = ResourceGovernor(host=self)
        # When a heavy head was last admitted through the foreign-pressure physical-fit branch. Its load
        # equally holds the queue, so this bounds a wedge grace that the whole-card establishment grace does
        # not cover. 0.0 when none is loading.
        self._heavy_head_admitted_at: float = 0.0
        # Edge-trigger latch for the disclosure that whole-card residency is off because its config flag
        # never resolved to a value (as distinct from resolving to False, which is an operator choice).
        self._whole_card_flag_unresolved_disclosed: bool = False
        # The whole-card pop claim as it was last disclosed, so the engage/release lines are edge-triggered:
        # a claim standing over many ticks says so once. None when the last disclosure was a release.
        self._pop_claim_disclosed: WholeCardPopClaim | None = None
        # Whether the empty-pop evidence ended the standing claim, so the release line names that reason
        # rather than the cap. Consumed by the disclosure that reports the release.
        self._pop_claim_empty_release_pending: bool = False
        # How the last standing claim ended and when, so the status snapshot can still say why the offer
        # widened back out after the disclosure line has scrolled. None before any claim has been released.
        self._pop_claim_release: tuple[WholeCardPopClaimRelease, float] | None = None
        # Edge-trigger latch for the disclosure that a residency on this multi-card host does not claim the
        # worker-wide offer, so its absence reads as deliberate rather than as the feature failing.
        self._pop_claim_multi_gpu_disclosed: bool = False
        # Per-card edge-trigger latch for the whole-card minimum hold's disclosure: the residency model whose
        # floor was last reported as holding a ready different-model head off that card. Cleared when the
        # residency restores, so each episode's floor is stated once.
        self._min_hold_disclosed: dict[int | None, str | None] = {}
        # When an idle inference slot was last deliberately cycled to reclaim allocator-retained RAM
        # (_replace_stale_ram_unload_process). The respawn + the next head's preload leave the queue
        # briefly unservable through no fault of the pool, so this bounds a wedge grace covering that
        # deliberate window. 0.0 when no reclaim cycle is in flight. See _RAM_RECLAIM_CYCLE_GRACE_SECONDS.
        self._ram_reclaim_cycle_at: float = 0.0
        # Per-lane throttle for idle service-lane RAM containment (_contain_idle_lane_ram): maps a lane's
        # process id to the monotonic time its last RAM-unload was requested, so a lane is asked to unload
        # at most once per _LANE_RAM_CONTAINMENT_MIN_INTERVAL_SECONDS.
        self._lane_ram_containment_at: dict[int, float] = {}
        # Head-of-queue starvation clock. Tracks the id of the job currently at the head of the queue and
        # when it first became budget-deferred onto an idle device. It only feeds
        # the arbiter's starvation diagnostic (a warning naming the arithmetic once a head is deferred past
        # the diagnostic horizon with reclaim exhausted). Reset when the head changes, a job dispatches, or a
        # live job takes the device.
        self._head_starvation_job_id: str | None = None
        self._head_starvation_since: float = 0.0

        # Head-of-queue RAM-defer starvation clock and the dispatch barrier it latches. The clock tracks the
        # head job whose preload the system-RAM verdict is continuously deferring behind live work, and when
        # that defer began; the head-starvation clock above is forced to zero whenever any job is in progress,
        # so it cannot see this behind-a-busy-sibling case. Once the continuous defer outlives
        # _HEAD_RAM_DEFER_BARRIER_SECONDS with reclaim freeing nothing, the barrier latches (its job id and
        # latch time recorded): new inference dispatch to other slots is withheld so the running siblings drain
        # to the no-live-consumer best-effort admit that seats the head. Both clear on the head's admission,
        # dispatch, fault, or departure. See _apply_ram_verdict, start_inference, and _reconcile_head_priority_barrier.
        self._head_ram_defer_job_id: str | None = None
        self._head_ram_defer_since: float = 0.0
        self._head_priority_barrier_job_id: str | None = None
        self._head_priority_barrier_since: float = 0.0
        self._head_priority_barrier_withhold_logged: bool = False

        # Safety-recovery admission hold. When the safety pool is crash-looping while a safety GPU start stays
        # deferred on a saturated card, new preload admissions on that card are held and idle reclaim is nudged
        # so the card can drain and the safety pool can start. Records when the hold engaged (0.0 when inactive)
        # and whether its engage notice has been emitted, so the edge is announced once. Bounded by
        # _SAFETY_RECOVERY_HOLD_TTL_SECONDS. See _safety_recovery_hold_active.
        self._safety_recovery_hold_since: float = 0.0
        self._safety_recovery_hold_logged: bool = False
        # Per-episode latch set when the hold gives up at its TTL. While set (and the crash-loop condition
        # still persists) the hold does not re-engage or re-log, so a stuck safety pool admits normally rather
        # than re-holding one preload per TTL window forever. Cleared by _release_safety_recovery_hold once the
        # condition clears, so a later recurrence is a fresh episode that may hold and expire again.
        self._safety_recovery_hold_expired: bool = False

        # Dispatch-stall diagnostic throttle. When the queue has work but nothing dispatches, the scheduler
        # would otherwise return None silently; this records the last reason logged and when, so the
        # explanation is emitted at most once per interval (and immediately when the reason changes) rather
        # than every sub-second control-loop tick.
        self._dispatch_stall_last_reason: str | None = None
        self._dispatch_stall_log_time: float = 0.0

        # The parent's measured WDDM demand-paging verdict (per-process GPU shared-segment usage on the
        # worker's own children). While set, retention is denied; the rising edge triggers an idle-VRAM
        # reclaim. Always False on hosts without the telemetry.
        self._wddm_paging_active: bool = False
        # The parent's most recent WDDM paging attribution: the child PIDs whose VRAM the driver demoted to
        # system memory, mapped to their shared (system-backed) GPU MB, plus a monotonic stamp of when it was
        # recorded. Refreshed on every active verdict (not just the rising edge) so the paged-slowdown
        # watchdog reads a current victim set, and cleared the moment paging clears. See
        # :meth:`wddm_paging_victim_shared_mb_by_pid`.
        self._wddm_paging_victims_shared_mb_by_pid: dict[int, float] = {}
        self._wddm_paging_victims_updated_monotonic: float = 0.0
        # Retention evictions this dispatch path issued, keyed by the slot they were sent to, kept until the
        # child's own post-free reports evidence the room is back. A dispatch that made room for itself waits
        # on these rather than on the tracking it just cleared.
        self._pending_retention_evictions: dict[int, _PendingRetentionEviction] = {}

        # Edge-log throttle for the post-processing/sampling time-slice hold on dispatch.
        self._pp_mutex_hold_logged: bool = False
        # Edge-log throttle for a dispatch admitted via the measured-truth co-residency path that the
        # static reported-total gate would have held. Sibling of _pp_mutex_hold_logged so the two never
        # both fire and neither repeats per tick.
        self._pp_mutex_measured_admit_logged: bool = False

        # Capacity-normalized wall-clock accounting: every scheduler tick attributes each configured
        # inference slot's elapsed time to SAMPLING or to the gate/supply state that kept it empty, so
        # "active vs idle vs gated" is a direct read over any window. Fed once per scheduling cycle
        # (record_slot_duty); snapshotted into the stats stream and the periodic duty-cycle log line.
        self._slot_duty = SlotDutyAccumulator()
        self._slot_duty_current_hold: SlotDutyBucket | None = None

        # The head whose dispatch is currently held for post-processing-peak headroom (job id, shortfall MB),
        # or None. Set when a dispatch defers (the peak overflows the contended card now but fits it alone and
        # an in-flight sibling will free room); read by the dispatch-stall diagnostic so a held head reads as
        # an explained wait. Cleared the moment any job dispatches.

    def set_churn_observer(self, observer: Callable[[ChurnKind], None]) -> None:
        """Register the sink for between-jobs reload/respawn events (see :data:`ChurnKind`)."""
        self._churn_observer = observer

    def set_admission_baseline_provider(self, provider: Callable[[int | None], float | None]) -> None:
        """Register the source of the measured shared-device baseline (MB) per card for the admission overlay.

        The manager's :meth:`ProcessManager.latest_baseline_estimate_mb`; called with a device index (None for
        the single-GPU / worker-wide case). Until wired (and in standalone unit tests) the overlay reads a
        baseline of 0, so capacity is the raw device total and the measured gate never denies what the
        predictive gate admits.
        """
        self._admission_baseline_provider = provider

    def latest_admission_denials(self, *, device_index: int | None = None) -> int:
        """Return the count of measured-floor admission denials on a card this run (calibration visibility)."""
        return self._admission_denials_by_device.get(device_index if device_index is not None else 0, 0)

    def latest_admission_headroom_mb(self, *, device_index: int | None = None) -> float | None:
        """Return the last measured-floor admission headroom (MB) on a card, or None when the floor was unapplied."""
        return self._admission_headroom_mb_by_device.get(device_index if device_index is not None else 0)

    def set_disaggregation_hooks(
        self,
        *,
        is_disaggregatable: Callable[[ImageGenerateJobPopResponse], bool],
        is_disaggregation_class_eligible: Callable[[ImageGenerateJobPopResponse], bool],
        register_disaggregated: Callable[[ImageGenerateJobPopResponse, HordeProcessInfo | None], Awaitable[bool]],
        pin_owner: Callable[[int], str | None] | None = None,
        sampling_peaks: Callable[[], dict[str, float]] | None = None,
        vae_decode_pending_count: Callable[[], int] | None = None,
        vae_lane_bound_job_count: Callable[[], int] | None = None,
    ) -> None:
        """Wire the pipeline-disaggregation predicates and router (see the ``_is_disaggregatable_job`` attr).

        ``is_disaggregatable`` is the dispatch-time predicate (class-eligible AND role processes live AND no
        whole-card residency held): at the dispatch seam an eligible job is routed to ``register_disaggregated``
        (which pins the process the scheduler chose as its sampler) instead of being sent monolithic inference.
        ``is_disaggregation_class_eligible`` is the stable class predicate used by residency forecasting and
        VRAM charging, so a job that will run disaggregated is priced sampler-only regardless of transient lane
        state (a whole-card window pauses the lane without flipping the forecast to the monolithic footprint).

        ``pin_owner`` maps a process id to the job pinning it as its sampler, and ``sampling_peaks`` returns the
        in-flight sampling peaks; both are read-only, used only by the dispatch-stall classifier to name a head
        held behind a pinned sampler lane. ``vae_decode_pending_count`` returns how many disaggregated jobs need
        the VAE lane for a decode now, read by the reclaim-ladder VAE-lane pause so the lane is not stopped out
        from under a queued or in-flight decode; ``vae_lane_bound_job_count`` is the wider count (sampling as
        well as decoding) that the whole-card residency's lane pause reads instead. All four are optional so
        standalone tests need not wire the orchestrator.
        """
        self._is_disaggregatable_job = is_disaggregatable
        self._is_disaggregation_class_eligible = is_disaggregation_class_eligible
        self._register_disaggregated_job = register_disaggregated
        if pin_owner is not None:
            self._disaggregation_pin_owner = pin_owner
        if sampling_peaks is not None:
            self._disaggregation_sampling_peaks = sampling_peaks
        if vae_decode_pending_count is not None:
            self._vae_decode_pending_count = vae_decode_pending_count
        if vae_lane_bound_job_count is not None:
            self._vae_lane_bound_job_count = vae_lane_bound_job_count

    def _disaggregation_sibling_charge_mb(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        *,
        device_index: int | None,
    ) -> float:
        """The image lane's concurrent VAE-decode spike (MB) to charge against co-residency when disaggregating.

        Prefers the job's *bounded* tiled-decode activation (``predict_job_decode_spike_mb``): the lane
        decodes the previous job's latent while this one samples, so only that decode working set (not the
        lane's whole allocator-guard quota) is the concurrent commitment. Charging the full quota over-commits
        the card and denies a second sampler it can physically hold, collapsing the pipeline. Falls back to the
        full lane quota when the pinned hordelib does not yet expose the decode-spike figure (conservative:
        safe but not optimally packed).
        """
        decode_spike_mb = predict_job_decode_spike_mb(job, str(baseline) if baseline is not None else None)
        if decode_spike_mb is not None:
            return decode_spike_mb
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        return effective_post_process_vram_quota_mb(total_vram_mb)

    def latest_preload_admission(self) -> LatestPreloadAdmission | None:
        """Return the most recent preload-admission decision, for the supervisor snapshot."""
        return self._last_preload_admission

    def latest_host_memory_governance_snapshot(self) -> HostMemorySnapshot | None:
        """Return the latest host-memory governance input snapshot, or None before the first tick."""
        verdict = self._governor.last_ram_verdict
        if verdict is None:
            return None
        return self._build_host_memory_snapshot(verdict)

    def _record_churn(self, kind: ChurnKind) -> None:
        """Report one churn event to the observer if one is registered (no-op otherwise)."""
        if self._churn_observer is not None:
            self._churn_observer(kind)

    @property
    def _max_concurrent_inference_processes(self) -> int:
        """The live concurrent-inference cap (effective ``max_threads``), bounded by the ceiling."""
        return self._runtime_config.effective_max_threads

    def get_single_job_effective_megapixelsteps(self, job: ImageGenerateJobPopResponse) -> int:
        """Return the number of effective megapixelsteps for a single job."""
        return _get_single_job_effective_megapixelsteps(job)

    def _expected_sampling_seconds(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
    ) -> float | None:
        """The performance model's expected sampling seconds for a job, or ``None`` when unavailable.

        Returns ``None`` when no model is wired, the baseline is unknown, or the job's signature has no
        seeded or calibrated rate yet (cold start), so the absence of an expectation is never an error.
        """
        if self._performance_model is None:
            return None
        signature = signature_from_job(job, str(baseline) if baseline is not None else None)
        if signature is None:
            return None
        return self._performance_model.expected_sampling_seconds(signature)

    def _diagnostic_mb_bucket(self, value: float | None) -> int | None:
        """Bucket memory telemetry so harmless measurement jitter does not spam diagnostics."""
        if value is None:
            return None
        return round(value / _SCHEDULER_DIAGNOSTIC_MB_BUCKET)

    def _scheduler_diagnostic_suppressed_count(
        self,
        name: str,
        state_key: tuple[object, ...],
    ) -> int | None:
        """Return suppressed-repeat count when a high-frequency diagnostic should be emitted.

        The first observation logs, a semantic state change logs immediately, and an unchanged observation
        logs periodically. ``None`` means "do not emit this time".
        """
        now = self._clock()
        previous = self._scheduler_diagnostic_log_state.get(name)
        if previous is None:
            self._scheduler_diagnostic_log_state[name] = (state_key, now, 0)
            return 0

        previous_key, previous_emit, suppressed_count = previous
        if previous_key != state_key:
            self._scheduler_diagnostic_log_state[name] = (state_key, now, 0)
            return suppressed_count

        if (now - previous_emit) >= _SCHEDULER_DIAGNOSTIC_REPEAT_SECONDS:
            self._scheduler_diagnostic_log_state[name] = (state_key, now, 0)
            return suppressed_count

        self._scheduler_diagnostic_log_state[name] = (previous_key, previous_emit, suppressed_count + 1)
        return None

    def _suppressed_suffix(self, suppressed_count: int) -> str:
        """Return a compact suffix for diagnostics that skipped unchanged loop repeats."""
        if suppressed_count <= 0:
            return ""
        return f" (suppressed {suppressed_count} unchanged repeats)"

    def _log_stream_forecast(self, job: ImageGenerateJobPopResponse, forecast: StreamForecast) -> None:
        """Log the stream forecast when its decision or materially-relevant measurements change."""
        if not forecast.known:
            return

        job_id = str(job.id_) if job.id_ is not None else None
        state_key = (
            job.model,
            job_id,
            self._diagnostic_mb_bucket(forecast.weights_mb),
            self._diagnostic_mb_bucket(forecast.reserve_mb),
            self._diagnostic_mb_bucket(forecast.free_now_mb),
            self._diagnostic_mb_bucket(forecast.free_after_model_evict_mb),
            self._diagnostic_mb_bucket(forecast.free_if_alone_mb),
            self._process_map.num_loaded_inference_processes(),
            self._diagnostic_mb_bucket(self._per_process_overhead_mb()),
            self._diagnostic_mb_bucket(forecast.marginal_process_overhead_mb),
            forecast.fits_coresident,
            forecast.needs_exclusive_residency,
            forecast.needs_process_count_reduction,
            forecast.streams_unavoidably,
            forecast.measured_retires_whole_card_intent,
        )
        suppressed_count = self._scheduler_diagnostic_suppressed_count(f"stream_forecast:{job_id}", state_key)
        if suppressed_count is None:
            return

        marginal = self._overhead.marginal_breakdown(config_override_mb=self._config_overhead_override_mb())
        marginal_chosen = f"{marginal.chosen_mb:.0f}" if marginal.chosen_mb is not None else "?"
        marginal_probe = f"{marginal.probe_mb:.0f}" if marginal.probe_mb is not None else "?"
        marginal_floor = f"{marginal.idle_floor_mb:.0f}" if marginal.idle_floor_mb is not None else "?"
        # ``unreclaimable`` and ``max_resident`` are what attribute a process-count-reduction verdict: the
        # first is the share of the after-model-evict deduction that stopping inference siblings cannot give
        # back, the second the context count the budget says already fits. A reduction demanded while
        # max_resident already covers the live processes is a deficit made of the unreclaimable charges.
        # A whole-card baseline that stops claiming the card on measured evidence is a visible reversal of a
        # declared policy, so the figure and the number of runs behind it are named where the reversal happens.
        measured_note = ""
        if forecast.measured_retires_whole_card_intent and forecast.measured_footprint_mb is not None:
            measured_note = (
                f" [whole-card intent retired: measured {forecast.measured_footprint_mb:.0f}MB over "
                f"{forecast.measured_observation_count} observations fits beside a sibling context]"
            )
        logger.debug(
            f"Stream forecast for {job.model}: {forecast.reason()}{measured_note} "
            f"[free_now={forecast.free_now_mb}, after_model_evict={forecast.free_after_model_evict_mb}, "
            f"alone={forecast.free_if_alone_mb}, "
            f"unreclaimable={forecast.unreclaimable_charge_mb:.0f}MB, live_procs="
            f"{self._process_map.num_loaded_inference_processes()}, "
            f"overhead/proc={self._per_process_overhead_mb():.0f}MB, "
            f"marginal/ctx={marginal_chosen}MB(src={marginal.source},probe={marginal_probe},"
            f"idle_floor={marginal_floor})] -> "
            f"coresident={forecast.fits_coresident}, "
            f"needs_exclusive={forecast.needs_exclusive_residency}, "
            f"needs_process_count_reduction={forecast.needs_process_count_reduction}"
            f"(max_resident={forecast.max_resident_processes()}), "
            f"streams_unavoidably={forecast.streams_unavoidably}"
            f"{self._suppressed_suffix(suppressed_count)}",
        )

    def _log_next_models_for_vram_unload(
        self,
        next_n_models: list[str],
        *,
        under_pressure: bool,
        for_head_of_queue: bool,
    ) -> None:
        """Log the unload guard's next-model view without repeating it every reclaim attempt."""
        in_progress_models = tuple(sorted(str(job.model) for job in self._job_tracker.jobs_in_progress))
        state_key = (
            tuple(next_n_models),
            in_progress_models,
            under_pressure,
            for_head_of_queue,
            self._max_inference_processes,
        )
        suppressed_count = self._scheduler_diagnostic_suppressed_count("vram_unload_next_models", state_key)
        if suppressed_count is None:
            return
        logger.debug(f"Next n models: {next_n_models}{self._suppressed_suffix(suppressed_count)}")

    def _budget_active(self) -> bool:
        """Whether the measured VRAM/RAM budget gates preload/dispatch this cycle.

        Disabled by config (``enable_vram_budget=false``) restores the prior availability-only
        behavior. Both reserves are synced from the (live-reloadable) config here. Tests construct the
        scheduler with a mocked bridge_data whose attributes are Mocks rather than real values; in that
        case (or any partial config) fall back to the pre-budget behavior instead of acting on a
        non-numeric reserve.
        """
        bridge_data = self._runtime_config.bridge_data
        enabled = bridge_data.enable_vram_budget
        vram_reserve = config_number(bridge_data.vram_reserve_mb)
        ram_reserve = config_number(bridge_data.ram_reserve_mb)
        if not isinstance(enabled, bool) or vram_reserve is None or ram_reserve is None:
            return False
        if not enabled:
            return False
        self._vram_budget.set_reserve_mb(vram_reserve)
        self._ram_budget.set_reserve_mb(ram_reserve)
        return True

    def _is_model_locally_unservable(self, model: str | None, *, device_index: int | None = None) -> bool:
        """Return whether ``model`` is held back as locally unservable on a card (the shared breaker policy).

        Delegates to :func:`is_model_locally_unservable_for` so dispatch and the popper's model selection
        apply one identical policy: a model held back here is neither dispatched nor popped, so the worker
        stops dropping a model the device genuinely cannot run. ``device_index`` scopes the streak to the card
        the admit targets on a multi-GPU host; None is the single-GPU / worker-wide reading.
        """
        return is_model_locally_unservable_for(
            self._runtime_config.bridge_data,
            self._job_tracker,
            model,
            device_index=device_index,
        )

    def _log_overbudget_admit(self, job: ImageGenerateJobPopResponse) -> None:
        """Log a foreign-pressure physical-fit admit with the residency/measurement picture.

        Captures, in one greppable line, the model admitted outside the worker's own admission capacity
        because it physically fits the device-free read, whether it runs exclusively, its prior over-budget
        fault streak, and the per-slot residency plus device-wide free VRAM at admit time.
        """
        exclusive = self._job_tracker.is_admitted_exclusive(job)
        fault_count = self._job_tracker.get_model_overbudget_fault_count(job.model)
        logger.opt(colors=True).warning(
            "<fg #f0beff>VRAM budget cannot fit head-of-queue model {} even after reclaiming all idle "
            f"VRAM/RAM, but it physically fits measured device-free VRAM; admitting it "
            f"({'exclusive' if exclusive else 'shared'}, prior_overbudget_faults={fault_count}) rather than "
            "wedging the queue. {}</>",
            job.model,
            self._process_map.residency_snapshot(),
        )

    def _mark_overbudget_admit(
        self,
        job: ImageGenerateJobPopResponse,
        forecast: StreamForecast | None,
        *,
        device_index: int | None = None,
    ) -> None:
        """Tag ``job`` as an over-budget physical-fit admit, opening the heavy-head load grace on first admit.

        Records the load-grace start the first time the job is admitted (so its multi-gigabyte load is not
        mistaken for a structural wedge; see :meth:`heavy_head_load_grace_active`). When over-budget
        exclusive mode is configured *and* the forecast shows the model's footprint dominates the card,
        also marks it exclusive so the scheduler suppresses concurrent pre-staging and dispatch for its
        duration, leaving the device un-contended while it completes.

        Exclusivity guards a heavy model against a concurrent sibling load pushing its weights into
        host-RAM streaming; that risk needs a footprint that dominates the device *on a card too small
        to host a sibling beside it* (see :attr:`StreamForecast.admit_requires_isolation`). A card-light
        model can reach this path purely through reserve arithmetic (free VRAM depressed by retained
        sibling contexts), and a card-dominating model on a roomy card co-resides safely; isolating
        either caps a multi-thread card at one job for the admit's whole lifetime while blocking every
        other preload. An unsized or missing forecast keeps the conservative isolation.
        """
        if not self._job_tracker.is_admitted_over_budget(job):
            self._heavy_head_admitted_at = self._clock()
        self._job_tracker.mark_admitted_over_budget(job)
        if self._runtime_config.bridge_data.overbudget_exclusive_mode and (
            forecast is None or forecast.admit_requires_isolation
        ):
            self._job_tracker.mark_admitted_exclusive(job, device_index=device_index)

    def set_measured_per_process_overhead_mb(
        self,
        overhead_mb: int | float,
        *,
        device_index: int | None = None,
    ) -> None:
        """Record the startup-measured per-process VRAM overhead (MB) for the streaming forecast.

        Args:
            overhead_mb: The measured first/sole-context cost (MB).
            device_index: The card it was measured on, so that card's forecasts price its own context; None
                records the worker-wide figure every card-less caller and every unmeasured card reads.
        """
        self._overhead.set_per_process_overhead_mb(overhead_mb, device_index=device_index)

    def set_measured_marginal_overhead_mb(
        self,
        marginal_mb: int | float,
        *,
        device_index: int | None = None,
    ) -> None:
        """Record the startup-measured *marginal* per-additional-context VRAM cost (MB) from the probe.

        Hard data (the probe's second-context delta) available from the first scheduling tick, so it fixes the
        startup-window over-count without waiting for siblings to reach idle. 0 (or unmeasurable) leaves the
        scheduler on its idle-residency fallback.

        Args:
            marginal_mb: The measured per-additional-context cost (MB).
            device_index: The card it was measured on; None records the worker-wide figure.
        """
        self._overhead.set_marginal_overhead_mb(marginal_mb, device_index=device_index)

    def _config_overhead_override_mb(self) -> float | None:
        """Return the coerced ``vram_per_process_overhead_mb`` config override, or None when unset/non-numeric.

        Tolerant of partially-mocked config: a non-numeric reading coerces to None so the overhead model falls
        back to its measured figures.
        """
        return config_number(self._runtime_config.bridge_data.vram_per_process_overhead_mb)

    def _per_process_overhead_mb(self, device_index: int | None = None) -> float:
        """Return the per-process VRAM overhead (MB) to assume: configured override, else measured, else 0.

        An explicit ``vram_per_process_overhead_mb`` config value (> 0) wins so operators can tune; otherwise
        the startup-measured figure is used. This is the *first/sole* context cost (it includes the one-time
        CUDA runtime allocation), used to size ``free_if_alone``; the per-additional-context cost is
        :meth:`_marginal_process_overhead_mb`.

        Args:
            device_index: The card being priced, so a heterogeneous host charges each card its own measured
                context cost. None (or a card the probe could not measure) reads the worker-wide maximum.
        """
        return self._overhead.per_process_mb(
            config_override_mb=self._config_overhead_override_mb(),
            device_index=device_index,
        )

    def _bare_context_total_mb(
        self,
        *,
        device_used_mb: float,
        baseline_mb: float,
        device_index: int | None,
    ) -> tuple[float, int] | None:
        """Decompose a truthful device-used reading into the tenants' bare-context total and their count.

        The worker-attributable bare-context total is truthful device-used minus the shared device baseline
        minus every committed-ledger tenant's byte-exact allocator reservation: what remains is only the
        context costs (the one-time CUDA runtime plus one context each), the exact quantity the overhead
        model's marginal derivation is defined over. Charging anything else (the baseline, resident weights,
        another tenant's reservation) into that residual multiplies it across the process count and prices
        the card into a phantom over-commit. Keyed on the committed ledger's tenant set so the marginal
        derivation and the ledger can never disagree about who holds a context. Returns None when the card
        has no ledger tenants; the residual may be negative (a baseline estimate that absorbed context cost),
        which the capture path skips and the invalidation path clamps toward zero.
        """
        reserved_sum_mb = 0.0
        tenants = self._process_map.committed_ledger_processes(device_index)
        if not tenants:
            return None
        for process_info in tenants:
            reserved_sum_mb += (process_info.process_reserved_mb or 0.0) + (process_info.process_aimdo_mb or 0.0)
        context_total_mb = device_used_mb - baseline_mb - reserved_sum_mb
        return context_total_mb, len(tenants)

    def capture_idle_context_residency(
        self,
        *,
        device_used_mb: float,
        baseline_mb: float,
        device_index: int | None = None,
    ) -> None:
        """Record the tenants' bare-context total when every inference process is idle with no model resident.

        That measurement is the true combined cost of the GPU tenants' contexts (the one-time CUDA runtime
        plus one context each), which the forecast needs to size ``free_after_model_evict`` without
        multiplying the one-time cost by the process count. Inspects the process map for the clean
        precondition (every live inference process up, idle, and holding no model, and no GPU tenant busy)
        and feeds a confirmed reading to the overhead model, which keeps the relevant extremes.

        Fed by the parent's attribution tick, which owns the truthful NVML device-used reading and the
        reconciled shared-baseline estimate: per-child VRAM views are per-process artefacts under WDDM and
        must never be decomposed as if they were device truth.

        Args:
            device_used_mb: Truthful device-wide used VRAM (MB) from the parent-side NVML read.
            baseline_mb: The reconciler's shared-device baseline estimate (MB) for the card.
            device_index: The card the reading belongs to; None for the single-GPU/worker-wide case.
        """
        inference_count = 0
        for process_info in self._process_map.values():
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.last_process_state in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED):
                continue
            # Any busy GPU tenant (an in-flight safety evaluation, a post-processing form) is transient VRAM
            # the residual would misread as context cost, so the clean window requires full quiescence.
            if process_info.is_process_busy():
                return
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            inference_count += 1
            # A clean baseline requires every live inference process up, idle, and holding no model: any model
            # resident (even one offloaded to RAM but still tracked) means the reading includes weight VRAM.
            if (
                process_info.last_process_state != HordeProcessState.WAITING_FOR_JOB
                or process_info.loaded_horde_model_name is not None
            ):
                return
        if inference_count < 1:
            return
        decomposed = self._bare_context_total_mb(
            device_used_mb=device_used_mb,
            baseline_mb=baseline_mb,
            device_index=device_index,
        )
        if decomposed is None:
            return
        context_total_mb, context_count = decomposed
        if context_total_mb <= 0:
            # The baseline estimate absorbed the context cost (it was captured with tenants already up): there
            # is no attributable residual to latch, and the marginal correctly falls back to probe/seed.
            return
        self._overhead.observe_idle_residency(
            context_total_mb=context_total_mb,
            context_count=context_count,
            device_index=device_index,
        )

    def invalidate_idle_context_floor(
        self,
        *,
        device_used_mb: float,
        baseline_mb: float,
        device_index: int | None = None,
    ) -> None:
        """Lower a latched effective idle floor once the device proves it was not a sustained reading.

        Complements :meth:`capture_idle_context_residency`. The capture keeps the worst clean all-idle
        reading; a transient spike would otherwise pin the per-context marginal high for the whole session
        and route ordinary models into teardown/exclusive admits. Unlike the capture this does not require
        the clean precondition: resident weights are netted out via the byte-exact reservations, and any
        residual transient VRAM only makes the correction conservative, so a bare-context reading below the
        latched floor (with at least as many tenants live) is unambiguous proof it was too high.

        Args:
            device_used_mb: Truthful device-wide used VRAM (MB) from the parent-side NVML read.
            baseline_mb: The reconciler's shared-device baseline estimate (MB) for the card.
            device_index: The card the reading belongs to; None for the single-GPU/worker-wide case.
        """
        decomposed = self._bare_context_total_mb(
            device_used_mb=device_used_mb,
            baseline_mb=baseline_mb,
            device_index=device_index,
        )
        if decomposed is None:
            return
        context_total_mb, context_count = decomposed
        self._overhead.observe_device_residency(
            context_total_mb=max(0.0, context_total_mb),
            context_count=context_count,
            device_index=device_index,
        )

    def _marginal_process_overhead_mb(self, device_index: int | None = None) -> float | None:
        """Return the per-additional-context VRAM cost (MB), or None to fall back to the first-context overhead.

        Prefers the probe's directly-measured second-context delta (hard data, available from the first tick,
        so it also covers the startup window where siblings have not yet reached idle). Failing that (the probe
        could not measure it on this backend), derives it from the measured all-contexts idle residency.
        Returns None when neither is available, in which case the forecast conservatively reuses the
        first-context overhead per additional context.

        Args:
            device_index: The card being priced, so a heterogeneous host charges each card its own measured
                per-context cost. None (or a card with no measurement) reads the worker-wide figures.
        """
        return self._overhead.marginal_mb(
            config_override_mb=self._config_overhead_override_mb(),
            device_index=device_index,
        )

    def resolved_context_constant_mb(self) -> float:
        """Return the per-process CUDA-context VRAM charge (MB) for the committed-VRAM attribution ledger.

        The measured per-additional-context marginal when the overhead model has one, else the platform seed
        (243 MB Windows / 144 MB Linux / the generic fallback), resolved by
        :func:`platform_context_constant_mb`. Consumed by the observational committed-VRAM ledger, drift
        reconciliation, and reclaim-ladder lane pricing, not by admission.
        """
        return platform_context_constant_mb(self._marginal_process_overhead_mb())

    def _whole_card_residency_enabled(self) -> bool:
        """Whether preventative whole-card exclusive residency is on (config, tolerant of mocked config).

        The identity test against True is deliberate: a config surface that has not resolved the flag hands
        back None, and a None here would otherwise silently read as "operator turned it off". That case is
        disclosed once, because a worker that quietly forgoes whole-card residency streams heavy models
        instead of failing visibly, which is very hard to attribute after the fact.
        """
        enabled = self._runtime_config.bridge_data.whole_card_exclusive_residency
        if enabled is None and not self._whole_card_flag_unresolved_disclosed:
            self._whole_card_flag_unresolved_disclosed = True
            logger.warning(
                "Whole-card exclusive residency is disabled because its config flag never resolved to a "
                "value. Heavy models will load co-resident with sibling contexts and may stream their "
                "weights. Set `whole_card_exclusive_residency` explicitly to choose the behaviour.",
            )
        return enabled is True

    def _whole_card_warranted(self, forecast: StreamForecast) -> bool:
        """Whether a teardown demand is trustworthy enough to engage the whole-card residency machinery.

        Reserving the whole card has a large blast radius: it stops sibling processes (which may be serving
        other queued heads), moves safety off-GPU, and holds the device through a cooldown, so it must only
        fire on a demand that is not a measurement artifact. Two signals qualify it:

        - a genuinely card-demanding model (its persistent footprint dominates the device, or its baseline is
          declared whole-card on intent): the teardown is warranted regardless of how contexts are counted; or
        - a per-additional-context cost that was actually *measured* (the probe's second-context delta or a
          derived idle-floor): the contention the demand rests on is real, not an over-count.

        When neither holds (a card-light model on a host where the marginal context cost could not be
        measured), the per-context overhead falls back to the full first-context cost, which charges the
        one-time CUDA runtime against every context and can collapse the structural floor below a model that
        physically co-resides with room to spare. Engaging a whole-card residency off that phantom reserves the
        card for a model that never needed it (and, held through the cooldown, can then starve a later head of a
        different model). So the caller falls through to the ordinary model-eviction path instead, whose
        admission still gates on real free VRAM, rather than reserving the device on an unmeasured guess.
        """
        if forecast.is_card_demanding:
            return True
        return self._marginal_process_overhead_mb() is not None

    def _log_whole_card_declined(self, job: ImageGenerateJobPopResponse, forecast: StreamForecast) -> None:
        """Record (once per model) that a whole-card teardown demand was declined as untrustworthy.

        Names why a model that the budget/forecast wanted to give the whole card was instead served by
        ordinary eviction: its footprint does not dominate the device and the per-additional-context cost was
        not measured, so the demand rests on the fallback that charges the one-time runtime cost against every
        context. Surfaces the numbers behind that call (the model's weight share of the card and whether the
        marginal was measured) so a teardown that does *not* happen is as visible in the logs as one that does.
        """
        if self._whole_card_declined_notified.get(job.model or "", False):
            return
        self._whole_card_declined_notified[job.model or ""] = True
        weights = forecast.weights_mb
        total = forecast.total_vram_mb
        share = f"{(weights / total) * 100:.0f}%" if weights is not None and total else "unknown"
        logger.opt(colors=True).info(
            "<fg #7b7d7d>Declined a whole-card residency for {}: "
            f"its weights (~{weights or 0:.0f}MB, "
            f"{share} of the {total or 0:.0f}MB card) do not dominate the device and the per-context overhead is "
            f"unmeasured (using the conservative first-context fallback), so a teardown demand cannot be trusted. "
            f"Serving it co-resident via model eviction instead of reserving the card.</>",
            job.model,
        )

    def _log_whole_card_reduction_suppressed(
        self,
        job: ImageGenerateJobPopResponse,
        forecast: StreamForecast,
        *,
        live_inference_processes: int,
    ) -> None:
        """Record (once per model) that a process-count-reduction claim was refused for want of a remedy.

        The reduction branch asks for idle sibling *processes* to be stopped. Where the budget already sizes
        at least as many co-resident contexts as are running, and the shortfall is within the charges no
        inference teardown reclaims, stopping siblings buys the head nothing: the deficit is the safety
        footprint, the service lanes' contexts, and the disaggregated image lane's decode spike, which a
        teardown removes only by stopping the lanes the work runs on. Names the figures behind that refusal so
        a residency that does *not* happen is as visible as one that does. Latched per model, so a head
        re-asking every scheduling tick discloses once rather than at the tick rate.
        """
        if self._whole_card_reduction_suppressed_notified.get(job.model or "", False):
            return
        self._whole_card_reduction_suppressed_notified[job.model or ""] = True
        weights = forecast.weights_mb or 0.0
        floor = forecast.base_reserve_mb if forecast.base_reserve_mb is not None else forecast.reserve_mb
        after_evict = forecast.free_after_model_evict_mb or 0.0
        logger.opt(colors=True).info(
            "<fg #7b7d7d>Refused a whole-card process-count reduction for {}: "
            f"its weights (~{weights:.0f}MB) plus the {floor:.0f}MB floor exceed the {after_evict:.0f}MB "
            f"siblings-present figure by {max(0.0, weights + floor - after_evict):.0f}MB, but "
            f"{forecast.unreclaimable_charge_mb:.0f}MB of that figure is safety, service-lane and decode "
            f"charges no inference teardown reclaims, and the budget already holds "
            f"{forecast.max_resident_processes()} contexts against {live_inference_processes} live. "
            f"Serving it co-resident instead of reserving the card.</>",
            job.model,
        )

    def _residency_state(self, device_index: int | None) -> WholeCardResidency:
        """Return the (lazily-created) whole-card residency state for ``device_index``.

        ``None`` is the single-GPU / worker-wide key, so a single-GPU host keeps exactly one residency state
        and behaves as the pre-multi-GPU scalar fields did.
        """
        return self._whole_card_ledger.state_for(device_index)

    def _serving_under_whole_card(self, model: str | None, device_index: int | None) -> bool:
        """Whether a whole-card exclusive residency for ``model`` is held on the card a job is dispatched to.

        Recorded per job at dispatch so a cost analysis can separate work served with the card to itself
        (which carries the amortized cost of establishing that residency) from co-resident work of the same
        shape. A residency held for a different model is not this job's, so it reads False.
        """
        if model is None:
            return False
        return self._residency_state(device_index).model == model

    def _held_residencies(self) -> list[tuple[int | None, WholeCardResidency]]:
        """Return ``(device_index, state)`` for every card currently holding a whole-card residency.

        A residency is "held" while its model is set. Used by the per-cycle convergence/restore passes and the
        supervisor-facing grace checks, which must consider every card's residency, not just one.
        """
        return self._whole_card_ledger.held()

    # The worker-wide (single-GPU) whole-card residency is the entry under the ``None`` key. These properties
    # expose its fields under their historical scalar names so single-GPU callers and tests read/write the
    # worker-wide residency exactly as before the per-card ``_whole_card_residencies`` map existed. The
    # multi-GPU admission path keys residency by real device index and does not go through these.
    @property
    def _sibling_teardown_for_model(self) -> str | None:
        """The model holding the worker-wide whole-card residency (the ``None``-keyed entry)."""
        return self._residency_state(None).model

    @_sibling_teardown_for_model.setter
    def _sibling_teardown_for_model(self, value: str | None) -> None:
        self._residency_state(None).model = value

    @property
    def _whole_card_forecast(self) -> StreamForecast | None:
        """The forecast that established the worker-wide whole-card residency."""
        return self._residency_state(None).forecast

    @_whole_card_forecast.setter
    def _whole_card_forecast(self, value: StreamForecast | None) -> None:
        state = self._residency_state(None)
        state.forecast = value
        state.repriced_target = self._whole_card_ledger.target_process_count(value)

    @property
    def _whole_card_established_at(self) -> float:
        """When the worker-wide whole-card residency was established (0.0 when none)."""
        return self._residency_state(None).established_at

    @_whole_card_established_at.setter
    def _whole_card_established_at(self, value: float) -> None:
        self._residency_state(None).established_at = value

    @property
    def _whole_card_cooldown_until(self) -> float:
        """Cooldown deadline of the worker-wide whole-card residency."""
        return self._residency_state(None).cooldown_until

    @_whole_card_cooldown_until.setter
    def _whole_card_cooldown_until(self, value: float) -> None:
        self._residency_state(None).cooldown_until = value

    @property
    def _whole_card_restore_at(self) -> float:
        """When the worker-wide whole-card residency was last restored (0.0 when none)."""
        return self._residency_state(None).restore_at

    @_whole_card_restore_at.setter
    def _whole_card_restore_at(self, value: float) -> None:
        self._residency_state(None).restore_at = value

    def _max_coresident_for_peak_mb(
        self,
        peak_mb: float,
        reserve_mb: float,
        *,
        device_index: int | None = None,
    ) -> int | None:
        """Largest live inference-process count that still fits ``peak_mb`` plus ``reserve_mb``.

        Sizes the context-reduction depth from the *same* conservative figure the VRAM verdict rejects on
        (the burden estimate), not the forecast's resident-weight estimate. The two estimators differ: the
        forecast judges co-residence from the resident weight footprint while the admission verdict uses the
        fuller per-job burden peak, so a moderate head can read co-resident in the forecast yet be rejected
        by the verdict every tick, the gap that routes it into the evict-all admit. Reasoning the teardown
        depth from the verdict's own peak makes the structural remedy fire exactly when admission would
        otherwise reject and thrash. The loader's first context costs the full one-time overhead; each
        additional co-resident context costs only the marginal. Returns None when it cannot be sized.

        Args:
            peak_mb: The job's predicted peak VRAM (MB) that must fit alongside the live contexts.
            reserve_mb: The transient-spike reserve (MB) required on top of the peak.
            device_index: When given, size against that one card's total VRAM (the per-card context-reduction
                depth on a multi-GPU host); when None, the worker-wide total.
        """
        return max_coresident_for_peak(
            total_vram_mb=self._process_map.get_reported_total_vram_mb(device_index=device_index),
            per_process_overhead_mb=self._per_process_overhead_mb(device_index),
            marginal_overhead_mb=self._marginal_process_overhead_mb(device_index),
            peak_mb=peak_mb,
            reserve_mb=reserve_mb,
        )

    def _forecast_streaming(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        *,
        device_index: int | None = None,
    ) -> StreamForecast:
        """Return the weight-streaming forecast for loading ``job``'s model given the device's measured state.

        Combines the measured free VRAM and total VRAM (from the children's reports), the configured reserve
        floor, and the per-process overhead so the scheduler can tell a model that only streams because of
        co-resident siblings (curable by exclusive residency) from one that streams even alone.

        Args:
            job: The job whose model load is being forecast.
            baseline: The model's known image-generation baseline (or its string form), or None when unknown.
            device_index: When given, forecast against that one card's measured free/total VRAM and its live
                inference- and safety-context counts (the per-card forecast on a multi-GPU host); when None,
                the worker-wide reading. The per-context overhead is a CUDA-runtime/arch constant and stays
                worker-wide either way (per-card overhead probing is a hordelib-side follow-up).
        """
        configured_floor = self._runtime_config.bridge_data.vram_reserve_mb
        floor_mb = (
            float(configured_floor)
            if isinstance(configured_floor, (int, float))
            and not isinstance(
                configured_floor,
                bool,
            )
            else 0.0
        )
        # The structural floor (free once every process's context materialises) is keyed off the *live*
        # inference-process count, not the configured ceiling. Using the live count is what lets the
        # teardown converge: once idle siblings are stopped for a whole-card job the forecast re-evaluates
        # against the reduced contexts and admits the model, instead of perpetually demanding more teardown
        # against a ceiling that is no longer running. Processes are staged up front (or once a model is on
        # disk), so by the time a job is scheduled the live count already reflects the real contention.
        num_processes = self._process_map.num_loaded_inference_processes(device_index=device_index)
        # The safety process holds a CUDA context *and* resident classifier weights on the card when
        # safety_on_gpu is set; that VRAM is not reclaimable by stopping idle inference siblings, so the
        # forecast must charge it against the achievable-free figures (sole residency for a heavy model then
        # implies moving safety off-GPU too). It is priced at :meth:`_safety_footprint_mb`, the same figure
        # admission, placement, and reclaim charge, rather than as one more marginal context: the marginal
        # prices an empty context and would under-count safety's weights. Charge it only when safety is
        # *actually* on the GPU right now: once a whole-card job has paused it off-GPU its footprint is
        # freed, so continuing to charge it would keep the structural floor (free_after_model_evict) below
        # the model's demand forever and the whole-card branch would defer the model every tick without ever
        # loading it. The safety process is pinned to a single card, so on a per-card forecast it is charged
        # only against the card it actually sits on.
        safety_on_gpu = self._safety_on_gpu_permitted and (not self._process_lifecycle.is_safety_gpu_paused)
        num_safety_contexts = self._process_map.num_safety_processes(device_index=device_index) if safety_on_gpu else 0
        safety_context_charge_mb = num_safety_contexts * self._safety_footprint_mb()
        # The dedicated post-processing lane holds a CUDA context (and its resident post-processing models)
        # on the card it is pinned to; like safety, that is a real device-wide commitment idle
        # inference siblings cannot reclaim, so it is charged as an extra resident context here (at the
        # marginal per-context cost, the lane holding no learned at-rest figure of its own). Charge it only
        # while the lane is actually on the card: once a whole-card job has stopped the lane off-GPU its context
        # is freed, so continuing to charge it would keep the structural floor below the model's demand and
        # defer the head forever (the same reasoning as the paused safety context above).
        num_post_process_contexts = (
            0
            if self._process_lifecycle.is_post_process_gpu_paused
            else self._process_map.num_post_process_processes(device_index=device_index)
        )
        # The EXTRA_LARGE tier (extra-large baselines plus the named VRAM-heavy checkpoints) is the single
        # source of truth for "wants the whole card and never shares". Feed it to the forecast so a baseline
        # whose conservative weight seed happens to fit co-resident still claims sole residency on intent,
        # rather than co-residing and thrashing as Z-Image did.
        wants_whole_card = self._model_size_tier(job.model) >= _ModelSizeTier.EXTRA_LARGE
        # A disaggregated job's sampler holds only the UNet, so its forecast charges the sampler-only figure
        # (keeping two samplers co-resident where the whole-job charge collapses them), and the image lane's
        # concurrent decode spike is charged as a sibling context. Class-eligibility (not the liveness-coupled
        # dispatch predicate) is used so a job that will run disaggregated is charged sampler-only even during a
        # whole-card window when the lane is transiently paused.
        disaggregated = self._is_disaggregation_class_eligible(job)
        measured_resident_mb, measured_observation_count = self._measured_resident_footprint_mb(job.model, baseline)
        return forecast_weight_streaming(
            job,
            str(baseline) if baseline is not None else None,
            free_now_mb=self._measured_free_vram_mb(device_index=device_index),
            total_vram_mb=self._process_map.get_reported_total_vram_mb(device_index=device_index),
            per_process_overhead_mb=self._per_process_overhead_mb(device_index),
            num_inference_processes=num_processes,
            configured_reserve_floor_mb=floor_mb,
            num_extra_resident_contexts=num_post_process_contexts,
            safety_context_charge_mb=safety_context_charge_mb,
            learned_resident_footprint_mb=self._learned_resident_footprint_mb(job.model, baseline),
            measured_resident_footprint_mb=measured_resident_mb,
            measured_observation_count=measured_observation_count,
            committed_reserve_mb=self._committed_vram_reserve_mb(device_index=device_index),
            marginal_process_overhead_mb=self._marginal_process_overhead_mb(device_index),
            wants_whole_card=wants_whole_card,
            disaggregated=disaggregated,
            disaggregation_sibling_charge_mb=(
                self._disaggregation_sibling_charge_mb(job, baseline, device_index=device_index)
                if disaggregated
                else 0.0
            ),
        )

    def _residency_should_pause_safety(self, device_index: int | None) -> bool:
        """Whether a whole-card residency on this card should also move the single safety process off-GPU.

        Requires safety configured-and-on-GPU (:meth:`_whole_card_safety_off_gpu_enabled`) and that this is
        the card the one safety process is pinned to (:meth:`_safety_gpu_card`, headroom-chosen, not a fixed
        index). A residency on a non-safety card never disturbs safety. The worker-wide key (``None``,
        single-GPU) always qualifies.
        """
        if not self._whole_card_safety_off_gpu_enabled():
            return False
        if device_index is None or not self._card_runtimes:
            return True
        return device_index == self._safety_gpu_card()

    def _safety_clear_of_residency_card(self, device_index: int | None) -> bool:
        """Whether safety satisfies this residency's card-local teardown leg.

        A residency that does not displace safety is clear by definition. When it does, a globally paused
        safety process is clear, but so is a live safety process now pinned to another card. Reading only the
        global pause flag would strand a residency after headroom-aware placement moved safety elsewhere.
        """
        if not self._residency_should_pause_safety(device_index):
            return True
        if self._process_lifecycle.is_safety_gpu_paused:
            return True
        safety_card = self._safety_gpu_card()
        return device_index is not None and safety_card != device_index

    def _has_safety_backlog(self) -> bool:
        """Return whether safety has work that should not be interrupted by residency churn."""
        return self._safety_backlog_depth() > 0

    def _safety_backlog_depth(self) -> int:
        """Return the total safety backlog: pending checks plus checks awaiting a verdict."""
        return len(self._job_tracker.jobs_pending_safety_check) + len(self._job_tracker.jobs_being_safety_checked)

    def _has_priority_safety_backlog(self) -> bool:
        """Return whether the safety backlog is deep enough to prioritize GPU restoration."""
        return self._safety_backlog_depth() > _SAFETY_BACKLOG_PRIORITY_DEPTH

    def _track_post_processing_backlog_age(self) -> int:
        """Return the post-processing backlog depth, advancing the clock the restore bound reads.

        The clock marks when the backlog last became non-empty and is cleared the moment it drains, so only
        continuously-occupied post-processing work accrues age and an intermittent lane never does.
        """
        pp_backlog_depth = len(self._job_tracker.jobs_pending_post_processing) + len(
            self._job_tracker.jobs_being_post_processed,
        )
        if pp_backlog_depth == 0:
            self._safety_restore_pp_backlog_since = None
        elif self._safety_restore_pp_backlog_since is None:
            self._safety_restore_pp_backlog_since = self._clock()
        return pp_backlog_depth

    def _post_processing_defers_safety_restore(self, pp_backlog_depth: int) -> bool:
        """Whether post-processing work should keep a paused safety process off its card this cycle.

        A safety (re)load competes for the card with the post-processing lane's transient demand, so live
        post-processing defers restoration. The deferral is bounded on both depth and age rather than being
        absolute: a backlog deeper than :data:`_SAFETY_RESTORE_PP_BACKLOG_DEPTH` is a lane under real load and
        always defers, while a shallow backlog defers only until it has been continuously occupied for
        :data:`_SAFETY_RESTORE_PP_BACKLOG_MAX_AGE_SECONDS`. Without the age bound a worker with a steady
        trickle of post-processed requests never presents an empty lane, and safety stays on the CPU for the
        whole run no matter how much room the card has.
        """
        if pp_backlog_depth == 0:
            return False
        if pp_backlog_depth > _SAFETY_RESTORE_PP_BACKLOG_DEPTH:
            return True
        since = self._safety_restore_pp_backlog_since
        if since is None:
            return True
        return (self._clock() - since) < _SAFETY_RESTORE_PP_BACKLOG_MAX_AGE_SECONDS

    def _held_residency_requests_safety_off_gpu(self) -> bool:
        """Return whether any live residency requires safety to remain off its card."""
        return any(self._residency_should_pause_safety(device_index) for device_index, _ in self._held_residencies())

    def _safety_footprint_mb(self) -> float:
        """The device VRAM (MB) the safety process costs while it sits on the GPU: the single safety price.

        :data:`_SAFETY_GPU_LOAD_CHARGE_MB` is the static seed and any measured
        :attr:`FootprintStage.SAFETY` watermark raises it, the same raise-only overlay every other stage's
        pricing uses. The learned figure is already safety's *whole device* footprint (its at-rest allocation
        plus the platform context constant, as the observation seam records it), directly
        comparable with the seed, so no context term is added on top of it here.

        Every consumer of the safety charge routes through this accessor (the arbiter's ``SAFETY_LOAD``
        request, both runtime-placement predicates, the streaming forecast's safety term, the residency
        charge-back, and the reclaim rung's promise), so admission, placement, forecasting, and reclaim can
        never disagree about what safety costs the card.
        """
        store = self._footprint_store
        if store is None:
            return _SAFETY_GPU_LOAD_CHARGE_MB
        return store.estimate_mb(
            FootprintKey(
                model_baseline=SAFETY_PROCESS_BASELINE,
                resolution_bucket=None,
                platform=sys.platform,
                stage=FootprintStage.SAFETY,
            ),
            static_seed_mb=_SAFETY_GPU_LOAD_CHARGE_MB,
        )

    def _arbiter_admits_safety_gpu_load(self, device_index: int | None) -> bool:
        """Whether the safety process may (re)load onto the GPU now, the VRAM arbiter deciding the memory question.

        Charges :meth:`_safety_footprint_mb` against the card's measured admission floor as a
        :attr:`VramRequestKind.SAFETY_LOAD`: a FITS verdict admits, a DEFER or DENY keeps safety off-GPU this
        cycle so the load re-asks. An unwired or cold arbiter admits, matching the
        every-gate-admits-on-missing-telemetry contract. No actuations run here (reclaim is single-owner, driven
        only by the preload path).
        """
        arbiter = self._vram_arbiter
        if arbiter is None or not arbiter.has_cycle:
            return True
        verdict = arbiter.evaluate(
            VramRequest(
                kind=VramRequestKind.SAFETY_LOAD,
                job_label="safety_load",
                baseline=None,
                device_index=device_index,
                candidate_delta_mb=self._safety_footprint_mb(),
            ),
        )
        return verdict.admits

    @property
    def _safety_permitted_cards(self) -> frozenset[int]:
        """The driven cards whose effective config permits hosting the on-GPU safety process."""
        return safety_permitted_card_indices(self._card_runtimes)

    @property
    def _safety_on_gpu_permitted(self) -> bool:
        """Whether any driven card permits an on-GPU safety process.

        The scheduler's placement and VRAM-accounting decisions read this rather than the global
        ``safety_on_gpu``: a card that withholds the permission is never charged for (or handed) a safety
        context, and no card permitting it is the same as the flag being globally off. A scheduler wired
        without a card plan reads the global flag, which is the same answer a card with no override gives.
        """
        if not self._card_runtimes:
            return bool(self._runtime_config.bridge_data.safety_on_gpu)
        return bool(self._safety_permitted_cards)

    def _choose_safety_gpu_card(self) -> int | None:
        """Return the driven card safety should be placed on: the one with the most verified headroom.

        The single placement identity, consumed both at spawn (pushed to the lifecycle manager, which pins the
        safety process there) and when the runtime placement policy re-promotes safety onto the GPU, so the
        two never disagree about which card safety lands on. Headroom per card is its truthful measured
        device-free VRAM when reported (that figure already nets out whatever is resident and sampling on the
        card right now); absent a measured reading it falls back to the card total less the largest sampling
        peak that card is committed to. The card with the greatest headroom wins, ties resolving to
        the lowest index so the choice is stable. Only cards whose effective config permits an on-GPU safety
        process are candidates, and the result is None when no card permits it. On a single-GPU host this is
        the one card, and with no headroom evidence at all it is the lowest-index card, both byte-identical to
        the historical fixed pin.
        """
        permitted = self._safety_permitted_cards
        if not permitted:
            return None
        best_index: int | None = None
        best_headroom_mb = float("-inf")
        for device_index in sorted(permitted):
            measured_free_mb = self._measured_free_vram_mb(device_index=device_index)
            if measured_free_mb is not None:
                headroom_mb = measured_free_mb
            else:
                total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
                if total_vram_mb is None:
                    continue
                headroom_mb = total_vram_mb - (self._largest_active_sampling_peak_mb(device_index) or 0.0)
            if headroom_mb > best_headroom_mb:
                best_headroom_mb = headroom_mb
                best_index = device_index
        return best_index if best_index is not None else min(permitted)

    def _safety_gpu_card(self) -> int | None:
        """Return the card safety currently occupies, or the card it would be placed on when off-GPU.

        When safety is on-GPU this is the card it is physically pinned to (from the lifecycle manager), so
        residency and fit checks reason about the real card. When safety is off-GPU it is the headroom-aware
        candidate (:meth:`_choose_safety_gpu_card`) it would land on at the next re-promotion. None on a
        host with no driven cards.
        """
        if not self._card_runtimes:
            return None
        current_card = self._process_lifecycle.safety_gpu_card_index()
        if current_card is not None and current_card in self._card_runtimes:
            return current_card
        return self._choose_safety_gpu_card()

    def _safety_restore_headroom_fits(self, device_index: int | None) -> bool:
        """Whether the chosen card's measured free covers safety *and* the peak that card is committed to.

        The restore side is a forecast, not a snapshot: safety only has to fit beside the sampling the card is
        already committed to, so readmitting it on room that the next peak will take back is how one restore
        buys the next eviction. The requirement is therefore safety's whole footprint plus the marginal step the
        heaviest committed peak still has to make, plus the proportional noise buffer, measured against the
        card's truthful device-free between allocation peaks. The device-free governor must also be HEALTHY
        there, so a card hovering at the paging cliff never readmits safety. Missing telemetry (no measured
        free) does not restore: the policy promotes only on positive, measured evidence.
        """
        inputs = self._safety_placement_inputs(device_index)
        available_mb = inputs.available_mb()
        if available_mb is None or inputs.governor_state is not GovernorState.HEALTHY:
            return False
        return available_mb >= inputs.restore_requirement_mb()

    def _runtime_safety_placement_enabled(self) -> bool:
        """Whether the runtime safety-placement policy may act (safety configured on-GPU on a real device).

        The policy can only ever degrade the operator's placement (GPU to CPU), never promote it, so it is inert
        unless some driven card's effective ``safety_on_gpu`` grants the maximum permission. On a CPU-only
        install safety is always off-GPU already, so there is nothing to place.
        """
        return self._safety_on_gpu_permitted and not is_cpu_only_install()

    def _sampling_peak_jobs_for_card(self, device_index: int | None) -> list[ImageGenerateJobPopResponse]:
        """The jobs whose sampling peak a card is committed to: those running on it and those it may be given.

        Cards are independent VRAM domains, so a peak that can only ever land on a sibling card says nothing
        about this one. In-progress jobs are attributed by the card their inference process sits on, and queued
        jobs by whether the card's effective config can serve them at all (an unknown fact never excludes a
        card, so this only narrows on a genuine mismatch). The worker-wide key (``None``) and a single-GPU host
        keep the whole set, which is the same answer.
        """
        queued_jobs = self._job_tracker.jobs_pending_inference
        if device_index is None or not self._multi_gpu_routing_active:
            return [*self._job_tracker.jobs_in_progress, *queued_jobs]
        eligible_queued = [job for job in queued_jobs if device_index in self._eligible_card_indices(job)]
        return [*self._jobs_in_progress_on_card(device_index), *eligible_queued]

    def _largest_active_sampling_peak(self, device_index: int | None) -> tuple[float, str] | None:
        """The heaviest learned solo sampling peak (MB) a card is committed to, with the model that carries it.

        Each job's static sampling-peak seed is raised by any learned SAMPLE-stage watermark for its footprint
        before the maximum is taken, so callers price the heaviest activation peak from measured high-waters
        rather than a seed the hardware has already overshot. Returns None when no job on the card can be
        priced (nothing sampling there), in which case the card is committed to no peak at all.
        """
        heaviest: tuple[float, str] | None = None
        seen: set[int] = set()
        for job in self._sampling_peak_jobs_for_card(device_index):
            if id(job) in seen:
                continue
            seen.add(id(job))
            if job.model is None:
                continue
            baseline = self._model_metadata.get_baseline(job.model)
            static_peak_mb = predict_job_sampling_vram_mb(job, baseline)
            if static_peak_mb is None:
                continue
            peak_mb = self._learned_sampling_peak_mb(
                job,
                baseline,
                static_seed_mb=static_peak_mb,
                stage=FootprintStage.SAMPLE,
            )
            if heaviest is None or peak_mb > heaviest[0]:
                heaviest = (peak_mb, job.model)
        return heaviest

    def _largest_active_sampling_peak_mb(self, device_index: int | None = None) -> float | None:
        """The heaviest learned solo sampling peak (MB) a card is committed to, or None when it has none."""
        heaviest = self._largest_active_sampling_peak(device_index)
        return None if heaviest is None else heaviest[0]

    def _resident_committed_mb_for_model(self, device_index: int | None, model: str) -> float:
        """The device memory (MB) a card's inference process already holds for ``model``, contexts included.

        Priced the way the committed-VRAM ledger prices a lane: the process's measured allocator reservation
        plus the resolved context constant, so what the card has already paid for the model is not counted a
        second time when forecasting the peak it still has to absorb. Zero when no process on the card holds the
        model with a priced reservation, which leaves the caller forecasting the whole peak.
        """
        context_constant_mb = self.resolved_context_constant_mb()
        committed_mb = 0.0
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.loaded_horde_model_name != model or process_info.process_reserved_mb is None:
                continue
            committed_mb = max(committed_mb, context_constant_mb + float(process_info.process_reserved_mb))
        return committed_mb

    def _safety_placement_marginal_need_mb(self, device_index: int | None) -> float:
        """The device VRAM (MB) the card still has to find for the heaviest peak it is committed to.

        The peak a card is committed to is not new demand in full: the process that will sample it already holds
        its weights and its context, and those bytes read as used in the card's measured free, not as room the
        peak needs again. Charging the whole peak instead is what makes a modeled non-fit permanent on a small
        card, so placement forecasts the marginal step from what is resident to what the peak will reach. Zero
        when the card is committed to no peak, or when what it already holds covers it.
        """
        heaviest = self._largest_active_sampling_peak(device_index)
        if heaviest is None:
            return 0.0
        peak_mb, model = heaviest
        return max(0.0, peak_mb - self._resident_committed_mb_for_model(device_index, model))

    def _safety_placement_inputs(self, device_index: int | None) -> _SafetyPlacementInputs:
        """Gather the per-card evidence both placement predicates read, as one snapshot for one cycle."""
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        return _SafetyPlacementInputs(
            device_index=device_index,
            measured_free_mb=self._measured_free_vram_mb(device_index=device_index),
            marginal_need_mb=self._safety_placement_marginal_need_mb(device_index),
            noise_buffer_mb=admission_noise_buffer_mb(total_vram_mb),
            governor_state=self.governor_state(device_index),
            safety_footprint_mb=self._safety_footprint_mb(),
            reclaimable_idle_mb=self._idle_retained_resident_mb(device_index),
        )

    def _idle_retained_resident_mb(self, device_index: int | None) -> float:
        """Device memory (MB) idle inference processes on a card hold under a retention grant.

        Priced by each process's measured allocator reservation, which is what an eviction returns to the card;
        a slot with a retained resident but no reservation reading contributes nothing (missing telemetry
        never inflates the room). Busy slots are excluded: their weights are in use, not reclaimable.
        """
        reclaimable_mb = 0.0
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.retained_resident_model is None or process_info.is_process_busy():
                continue
            if process_info.process_reserved_mb is None:
                continue
            reclaimable_mb += float(process_info.process_reserved_mb)
        return reclaimable_mb

    def _safety_placement_card_is_pressured(self, device_index: int | None) -> bool:
        """Whether safety's own card is short of memory right now, as measured evidence rather than a model.

        Two facts demote, both about the card safety occupies: the device-free governor has left HEALTHY there,
        or its measured free no longer covers the marginal step the heaviest committed peak still has to take
        plus the proportional noise buffer. A modeled non-fit is deliberately not one of them: on a small card
        that arithmetic (device total less the whole peak, the noise buffer and safety's footprint) can be
        unsatisfiable by a margin narrower than the buffer itself, which arms a permanent eviction against a
        card that is in fact serving its work. Missing telemetry does not demote, matching the
        every-gate-admits-on-missing-measurement contract, and neither does a card with nothing left to find:
        a low free reading beside work the card already holds the memory for is a card full of weights, which is
        admission's and reclaim's subject rather than safety's placement.
        """
        inputs = self._safety_placement_inputs(device_index)
        if inputs.governor_state is not GovernorState.HEALTHY:
            return True
        available_mb = inputs.available_mb()
        if available_mb is None or inputs.marginal_need_mb <= 0.0:
            return False
        return available_mb < (inputs.marginal_need_mb + inputs.noise_buffer_mb)

    def _safety_fits_beside_largest_sampling_peak(
        self,
        device_index: int | None,
        *,
        require_margin: bool,
    ) -> bool:
        """Whether the safety charge fits on the card beside the largest active sampling peak, as arithmetic.

        Structural fit over (device total, largest learned sampling peak, proportional noise buffer, the safety
        charge): ``total - peak - noise - safety_charge >= 0``. No constant is tuned to a card size; the
        noise buffer scales with the device total. With ``require_margin`` an extra proportional buffer must
        also clear, so a caller wanting durable headroom asks for more than a bare fit. When the device total
        is unknown or nothing is sampling on the card, the charge trivially fits (the policy never forces safety
        off on missing telemetry, matching the every-gate-admits-on-missing-measurement contract).

        This is a forecast input and never a trigger on its own: on a small card the arithmetic can fail by less
        than the noise buffer while the card is comfortably serving its work, so a policy that demoted on it
        alone would hold safety off the card permanently.
        """
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        if total_vram_mb is None or total_vram_mb <= 0:
            return True
        peak_mb = self._largest_active_sampling_peak_mb(device_index)
        if peak_mb is None:
            return True
        noise_mb = admission_noise_buffer_mb(total_vram_mb)
        margin_mb = admission_noise_buffer_mb(total_vram_mb) if require_margin else 0.0
        return (total_vram_mb - peak_mb - noise_mb - self._safety_footprint_mb() - margin_mb) >= 0.0

    def _safety_placement_dwell_seconds(self) -> float:
        """Seconds of continuous evidence a safety demotion must have before it is worth actuating.

        A placement flip ends the safety process and brings a replacement up, so it costs the worker every
        second of that rebuild in safety unavailability, during which finished images cannot be cleared. The
        dwell is that measured cost (:meth:`ProcessLifecycleManager.safety_readiness_latency_seconds`): pressure
        that has not persisted on the order of what relieving it costs is not worth relieving this way. Counting
        control cycles instead prices the flip at a fraction of a second, which is how a respawn window comes to
        decide the next flip.
        """
        return self._process_lifecycle.safety_readiness_latency_seconds()

    def _safety_placement_restore_dwell_seconds(self) -> float:
        """Seconds of continuous forecast headroom a safety restore must have (never below the demotion dwell)."""
        return self._safety_placement_dwell_seconds() * _SAFETY_PLACEMENT_RESTORE_DWELL_FACTOR

    def _safety_placement_dwell_met(self, since: float | None, dwell_seconds: float) -> bool:
        """Whether evidence first seen at ``since`` has now held continuously for ``dwell_seconds``."""
        if since is None:
            return False
        return (self._clock() - since) >= dwell_seconds

    def _log_safety_placement_inputs(self, inputs: _SafetyPlacementInputs) -> None:
        """Log the placement evidence on a change, so a later capture can attribute a flip to what it saw.

        Edge-triggered on the rounded evidence: the control loop runs several times a second, so an unchanged
        picture repeats at TRACE and only a genuine change speaks at DEBUG.
        """
        free_mb = inputs.measured_free_mb
        signature: tuple[object, ...] = (
            inputs.device_index,
            None if free_mb is None else round(free_mb / 64.0),
            round(inputs.marginal_need_mb / 64.0),
            round(inputs.reclaimable_idle_mb / 64.0),
            inputs.governor_state,
            self._process_lifecycle.is_safety_gpu_paused,
        )
        message = f"Runtime safety placement inputs: {inputs.describe()}."
        if signature == self._safety_placement_last_logged_inputs:
            logger.trace(message)
            return
        self._safety_placement_last_logged_inputs = signature
        logger.debug(message)

    def _advance_safety_placement_evidence(
        self,
        inputs: _SafetyPlacementInputs,
        *,
        safety_backlog_depth: int,
        residency_veto: bool,
    ) -> None:
        """Advance the one evidence clock that applies to safety's current placement, clearing the other.

        Only one side can be gathering evidence at a time: a resident safety process accrues the pressure that
        would justify evicting it, and an evicted one accrues the forecast headroom that would justify bringing
        it back. Either clock resets the moment its condition stops holding, so an intent can never outlive the
        evidence that produced it.

        Two conditions reset the pressure clock outright rather than feeding it. A backlog waiting on safety,
        because evicting safety while the worker is behind on safety checks stalls exactly the stage it is behind
        on. And a whole-card residency on safety's card, because filling that card is what the residency is: the
        pressure is the residency's own, it already owns safety's placement while it holds the card, and a
        pressure-owned pause taken alongside it would outlive the residency that caused it.
        """
        now = self._clock()
        if self._process_lifecycle.is_safety_gpu_paused:
            self._safety_placement_pressure_since = None
            if not self._safety_restore_headroom_fits(inputs.device_index):
                self._safety_placement_headroom_since = None
            elif self._safety_placement_headroom_since is None:
                self._safety_placement_headroom_since = now
            return

        self._safety_placement_headroom_since = None
        pressure_counts = (
            not residency_veto
            and safety_backlog_depth == 0
            and self._safety_placement_card_is_pressured(inputs.device_index)
        )
        if not pressure_counts:
            self._safety_placement_pressure_since = None
        elif self._safety_placement_pressure_since is None:
            self._safety_placement_pressure_since = now

    def _reconcile_runtime_safety_placement(self, *, update_policy: bool = True) -> None:
        """Apply the single reconciled safety placement chosen from every resource-governance request.

        Runtime placement policy, whole-card residency, and the reclaim ladder contribute demand rather than
        cycling the process independently. This method is the sole caller of the lifecycle safety pause and
        restore actuators. A live residency is a restore veto even when another request initiated the pause,
        and the device-free governor similarly vetoes growth back onto the card. Accepted post-processing work
        defers restoration under a depth-and-age bound (see :meth:`_post_processing_defers_safety_restore`)
        rather than absolutely, so a steady trickle cannot hold safety off the card for the whole run.

        The policy demotes only on measured pressure on safety's own card, sustained for a dwell measured in
        seconds against what the flip costs, and restores only once that card's measured free covers safety
        beside the peak it is committed to, sustained for the longer restore dwell. Every owner's restore earns
        that same forecast, so a residency or ladder pause is not handed back into the pressure that took it.
        Whole-card and reclaim requests do not wait for the demotion dwell. The reclaim request is one-shot and
        is consumed only after an off-GPU safety process reaches readiness. No placement flip is issued while a
        prior intentional rebuild remains unready, and no evidence accrues across that window either, so the
        rebuild cannot decide the flip that follows it.

        Args:
            update_policy: Whether to advance the placement evidence clocks. Reclaim may ask this reconciler to
                apply a freshly filed request immediately without manufacturing an extra policy observation.
        """
        # A config reload can withdraw the permission of the card safety is sitting on. Getting it off that
        # card is the same "safety must leave here" request as memory pressure, so it goes through the one
        # placement actuator; the bring-up then re-reads the permission, which is what keeps safety off until
        # a card grants it again. Checked ahead of the policy gate because withdrawing every card's
        # permission is exactly what turns that gate off.
        self._process_lifecycle.demote_safety_from_unpermitted_card()

        placement_enabled = self._runtime_safety_placement_enabled()
        if not placement_enabled:
            self._safety_placement_pressure_since = None
            self._safety_placement_headroom_since = None
            self._safety_reclaim_pause_requested = False
            self._safety_restore_pp_backlog_since = None
            return

        safety_backlog_depth = self._safety_backlog_depth()
        safety_card = self._safety_gpu_card()
        pp_backlog_depth = self._track_post_processing_backlog_age()

        # The headroom-aware placement choice only takes effect at a bring-up, and pushing it while safety is
        # resident would migrate the process on its next respawn for reasons unrelated to that respawn. Pushed
        # only when a spawn could actually use it (safety off-GPU, or not yet placed). Single-GPU keeps the
        # historical fixed pin (None), so its spawn path is byte-identical.
        if len(self._card_runtimes) > 1 and self._process_lifecycle.safety_gpu_card_index() is None:
            self._process_lifecycle.set_desired_safety_card(self._choose_safety_gpu_card())

        inputs = self._safety_placement_inputs(safety_card)
        self._log_safety_placement_inputs(inputs)

        residency_veto = self._held_residency_requests_safety_off_gpu()
        residency_may_initiate = residency_veto and (
            self._process_lifecycle.is_safety_gpu_paused or safety_backlog_depth == 0
        )

        transition_pending = self._process_lifecycle.safety_placement_transition_pending is True
        if transition_pending:
            # An intentional rebuild is the actuation of a decision already taken. The card it is reshaping says
            # nothing about whether the next flip is warranted, so evidence is frozen and restarts from scratch
            # once the rebuild proves readiness; otherwise the respawn window itself decides the next flip.
            self._safety_placement_pressure_since = None
            self._safety_placement_headroom_since = None
        elif update_policy:
            self._advance_safety_placement_evidence(
                inputs,
                safety_backlog_depth=safety_backlog_depth,
                residency_veto=residency_veto,
            )
        placement_wants_off = (
            not self._process_lifecycle.is_safety_gpu_paused
            and self._safety_placement_pressure_since is not None
            and self._safety_placement_dwell_met(
                self._safety_placement_pressure_since,
                self._safety_placement_dwell_seconds(),
            )
        )
        requested_owner: PauseOwner | None = None
        if residency_may_initiate:
            requested_owner = PauseOwner.WHOLE_CARD
        elif self._safety_reclaim_pause_requested:
            requested_owner = PauseOwner.RECLAIM_LADDER
        elif placement_wants_off:
            requested_owner = PauseOwner.RUNTIME_SAFETY_PLACEMENT

        if transition_pending:
            return

        if self._process_lifecycle.is_safety_gpu_paused:
            if self._safety_reclaim_pause_requested:
                # The off-GPU child reached readiness, so the ladder's one-shot request has materialised. Other
                # live requests still keep it off; absent one, restoration is reconsidered on the next cycle.
                self._safety_reclaim_pause_requested = False
                return
            if requested_owner is not None or residency_veto:
                return
            if 0 < safety_backlog_depth <= _SAFETY_BACKLOG_PRIORITY_DEPTH:
                return
            if self._post_processing_defers_safety_restore(pp_backlog_depth):
                return
            if self.is_vram_growth_held(safety_card) or not self._arbiter_admits_safety_gpu_load(safety_card):
                return
            pause_owner = self._process_lifecycle.safety_pause_owner
            if pause_owner is None:
                return
            if pause_owner in _MEMORY_PRESSURE_PAUSE_OWNERS and not self._safety_placement_dwell_met(
                self._safety_placement_headroom_since,
                self._safety_placement_restore_dwell_seconds(),
            ):
                # A pause taken because the card was short of memory is part of why the instantaneous gates
                # above now pass: the memory it returned is the memory they are reading. Restoring on that alone
                # hands the card straight back into the pressure that evicted safety, and each round trip ends
                # the safety process twice. The forecast headroom clock is the evidence that the room is durable
                # and covers the peak the card is committed to. A whole-card residency's pause is not in this
                # set: it ends when its own model drains, and holding its restore to a memory forecast would
                # leave a card that hosts one heavy resident without an on-GPU safety process for the session.
                return
            headroom_since = self._safety_placement_headroom_since or self._clock()
            if not self._process_lifecycle.restore_safety_on_gpu(owner=pause_owner):
                return
            self._safety_placement_headroom_since = None
            if pause_owner is PauseOwner.RUNTIME_SAFETY_PLACEMENT:
                self._safety_placement_promotions += 1
                logger.info(
                    f"Runtime safety placement: restoring safety to card {safety_card} after "
                    f"{self._clock() - headroom_since:.0f}s of measured free at or above the "
                    f"{inputs.restore_requirement_mb():.0f}MB it needs to survive the peak that card is "
                    f"committed to ({inputs.describe()}).",
                )
            return

        if requested_owner is None:
            return
        pressure_since = self._safety_placement_pressure_since or self._clock()
        if not self._process_lifecycle.pause_safety_on_gpu(owner=requested_owner):
            return
        self._safety_placement_pressure_since = None
        if requested_owner is PauseOwner.RUNTIME_SAFETY_PLACEMENT:
            self._safety_placement_demotions += 1
            logger.info(
                f"Runtime safety placement: moving safety off card {safety_card} after "
                f"{self._clock() - pressure_since:.0f}s of sustained memory pressure there "
                f"({inputs.describe()}). Restoring it costs about "
                f"{self._safety_placement_dwell_seconds():.0f}s of safety unavailability, which is the dwell "
                f"this pressure had to outlast.",
            )

    def _residency_should_pause_post_process(self, device_index: int | None) -> bool:
        """Whether a whole-card residency on this card should also stop the dedicated post-processing lane.

        Requires the lane to be enabled and to sit on the residency's card: its permanent CUDA context (and any
        warm upscaler models) is real device-wide VRAM that a sibling teardown cannot reclaim, so on a card too
        tight to host a whole-card model beside it (Flux on 16GB) the lane must vacate the card exactly as safety
        does. A residency on a card the lane does not occupy leaves it untouched. The worker-wide key (``None``,
        single-GPU) always qualifies: the lane shares the one card.
        """
        if not self._process_lifecycle.post_process_lane_enabled():
            return False
        if device_index is None or not self._card_runtimes:
            return True
        return device_index == self._process_lifecycle.post_process_lane_card_index()

    def _residency_should_pause_vae_lane(self, device_index: int | None) -> bool:
        """Whether a whole-card residency on this card should also stop the dedicated VAE lane.

        Mirrors :meth:`_residency_should_pause_post_process`: the lane's permanent CUDA context is real
        device-wide VRAM a sibling teardown cannot reclaim, so on a card too tight to host a whole-card model
        beside it the lane must vacate the card exactly as safety and the post-processing lane do. Requires
        the lane to be enabled and to sit on the residency's card; the worker-wide key (``None``, single-GPU)
        always qualifies.
        """
        if not self._process_lifecycle.vae_lane_enabled():
            return False
        if device_index is None or not self._card_runtimes:
            return True
        return device_index == self._process_lifecycle.vae_lane_card_index()

    def _residency_should_pause_component_lane(self, device_index: int | None) -> bool:
        """Whether a whole-card residency on this card should also stop the dedicated component lane.

        Mirrors :meth:`_residency_should_pause_vae_lane`: the component lane's permanent CUDA context and its
        resident text encoders are real device-wide VRAM a sibling teardown cannot reclaim, so on a card too
        tight to host a whole-card model beside it the lane must vacate the card exactly as safety, the
        post-processing lane, and the VAE lane do. Requires the lane to be enabled and to sit on the
        residency's card; the worker-wide key (``None``, single-GPU) always qualifies.
        """
        if not self._process_lifecycle.component_lane_enabled():
            return False
        if device_index is None or not self._card_runtimes:
            return True
        return device_index == self._process_lifecycle.component_lane_card_index()

    def _has_post_process_backlog(self) -> bool:
        """Return whether a post-processing job is pending or actively on the lane.

        Whole-card residency must leave a bounded window for post-processing jobs that peel off the resident
        model. Pending work therefore counts as backlog: the normal residency lever is to unload the idle lane's
        modules from VRAM, not to remove the lane and strand its queue. The one exception is a structurally
        incompatible card/model/lane combination, which first disables post-processing for the session and then
        stops the idle lane so the heavy model can fit.
        """
        return bool(self._job_tracker.jobs_pending_post_processing or self._job_tracker.jobs_being_post_processed)

    def _post_process_context_fits_with_residency(
        self,
        forecast: StreamForecast,
        *,
        device_index: int | None,
    ) -> bool:
        """Whether the residency model can load with the post-processing lane's bare context alive."""
        if not self._residency_should_pause_post_process(device_index):
            return True
        if forecast.weights_mb is None or forecast.total_vram_mb is None:
            return True
        target = self._whole_card_ledger.target_process_count(forecast)
        if target is None:
            return True
        marginal = forecast._effective_marginal_overhead_mb  # noqa: SLF001 - same budget object owns the estimate.
        extra_contexts = max(0, target - 1) + 1  # surviving inference siblings plus the PP lane context.
        free_with_pp_lane_mb = max(
            0.0,
            float(forecast.total_vram_mb) - forecast.per_process_overhead_mb - marginal * extra_contexts,
        )
        return (free_with_pp_lane_mb - forecast.weights_mb) >= forecast._effective_base_reserve  # noqa: SLF001

    def _disable_post_processing_for_whole_card(self, model_name: str | None, forecast: StreamForecast) -> None:
        """Session-disable post-processing because a whole-card model cannot fit beside the lane context."""
        if self._state.post_processing_disabled_by_breaker:
            return
        model = model_name or "the whole-card model"
        self._state.post_processing_disabled_by_breaker = True
        self._state.post_processing_breaker_tripped_at = time.time()
        self._state.post_processing_disabled_reason = (
            f"Disabled: {model} needs whole-card residency and cannot fit beside the dedicated "
            "post-processing lane's GPU context. Disable post-processing or move this workload to a card "
            "with more VRAM, then restart."
        )
        logger.warning(
            f"Disabling post-processing for this session: {model} needs whole-card residency and cannot fit "
            "beside the dedicated post-processing lane's GPU context. Keeping post-processing enabled would "
            "thrash the large model in and out of VRAM for every post-processing job. To restore it, disable "
            "whole-card models or post-processing for this worker, move the workload to a larger card, and restart.",
        )

    def _pause_post_process_for_residency_if_idle(
        self,
        device_index: int | None,
        *,
        model_name: str | None,
        forecast: StreamForecast,
    ) -> bool:
        """Reclaim the post-processing lane for residency without stranding supported PP work."""
        if not self._residency_should_pause_post_process(device_index):
            return False
        if self._post_process_context_fits_with_residency(forecast, device_index=device_index):
            return self.unload_post_process_models_from_vram(device_index=device_index)
        self._disable_post_processing_for_whole_card(model_name, forecast)
        if self._process_lifecycle.is_post_process_gpu_paused or self._has_post_process_backlog():
            return False
        return self._process_lifecycle.pause_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD)

    def _pause_vae_lane_for_residency_if_drained(self, device_index: int | None) -> bool:
        """Stop the VAE lane off the residency's card once no disaggregated job still needs it.

        The establishment pass withholds this pause while any job dispatched before the residency is still
        sampling or awaiting its decode, so the lane can still be on the card once the residency is held.
        Retried every convergence cycle for the same reason the sibling scale-down is: the room the residency
        wants is only actually freed when the lane's context goes. Idempotent once the lane is paused under
        this owner, and it never takes over another owner's pause.

        Args:
            device_index: Card holding the residency; None is the single-GPU / worker-wide case.

        Returns:
            True when this call initiated the pause.
        """
        if not self._residency_should_pause_vae_lane(device_index):
            return False
        if self._process_lifecycle.vae_lane_pause_owner is PauseOwner.WHOLE_CARD:
            return False
        if self._vae_lane_pause_deferred_for_decode(requester=VaeLanePauseRequester.WHOLE_CARD_RESIDENCY):
            return False
        return self._process_lifecycle.pause_vae_lane_off_gpu(owner=PauseOwner.WHOLE_CARD)

    def _establish_whole_card_residency(
        self,
        job: ImageGenerateJobPopResponse,
        forecast: StreamForecast,
        *,
        announce: bool,
        target_override: int | None = None,
        device_index: int | None = None,
        target_process: HordeProcessInfo | None = None,
    ) -> None:
        """Claim the device for a whole-card model: stop idle siblings and move safety off-GPU.

        The siblings' fixed per-process CUDA contexts (not their models) over-commit the device, and a context
        is only reclaimed by the process exiting (``torch.cuda.empty_cache`` returns cached blocks but never a
        context). Reduce the live inference-process count to the largest that still leaves room for this model's
        weights plus its activation reserve, and, on the very edge (Flux on a 16GB card), also move the
        safety process off-GPU so its context is freed too. The model is remembered so the residency is held
        and then restored once its job drains (after the configured cooldown). Only idle inference processes
        are stopped; a busy sibling is left to finish its job.

        ``device_index`` scopes the residency to one card on a multi-GPU host (only that card's processes are
        reduced, and safety is paused only if it sits on that card); None is the single-GPU / worker-wide case.

        ``target_process`` is the slot the caller is about to load or dispatch this head on. It is spared from
        the scale-down: ``protected_model`` spares only lanes already carrying the model, and a head that is
        not staged anywhere yet has none, so its own empty idle target would otherwise be a legal victim of
        the teardown its residency ordered.
        """
        self._whole_card_ledger.record_grant(
            device_index,
            model=job.model,
            forecast=forecast,
            cooldown_until=self._clock() + self._whole_card_cooldown_seconds(),
            now=self._clock(),
            establish_grace_seconds=_WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
        )
        if target_override is not None:
            self._whole_card_ledger.tighten_target(device_index, target_override)

        # ``target_override`` lets a caller size the depth from the admission verdict's rejected peak rather
        # than the forecast's lighter resident-weight estimate, for the activation-peak context over-commit the
        # weight-based gates leave co-resident. A forecast that cannot size the card yields no target: the
        # scale-down is skipped and the live count stands, the same absent-target contract the converge path
        # and the ledger's target_process_count follow.
        target = target_override if target_override is not None else forecast.max_resident_processes()
        current = self._process_map.num_loaded_inference_processes(device_index=device_index)
        after = current
        if target is not None and target < current:
            after = self._scale_sparing(
                target,
                device_index=device_index,
                protected_model=job.model,
                spared_process_id=target_process.process_id if target_process is not None else None,
            )

        safety_pause_requested = self._residency_should_pause_safety(device_index) and not self._has_safety_backlog()
        post_process_paused = self._pause_post_process_for_residency_if_idle(
            device_index,
            model_name=job.model,
            forecast=forecast,
        )
        # The disaggregated pipeline's VAE lane holds an equivalent bare CUDA context; stop it off-GPU on the
        # residency's card so the heavy model's weights are not tipped into host-RAM streaming by it. New
        # disagg dispatch is suppressed while a residency is active, but a job dispatched before it is still
        # sampling or decoding, and the residency cannot claim the card until that job drains anyway; pausing
        # under it buys nothing and discards the sample the lane's short decode hold cannot outlast. The pause
        # is retried each cycle by _converge_whole_card_residency once that work drains. A no-op unless
        # disaggregation is enabled and the lane sits on this card.
        if self._residency_should_pause_vae_lane(device_index) and not self._vae_lane_pause_deferred_for_decode(
            requester=VaeLanePauseRequester.WHOLE_CARD_RESIDENCY,
        ):
            self._process_lifecycle.pause_vae_lane_off_gpu(owner=PauseOwner.WHOLE_CARD)
        # The disaggregated pipeline's component (text-encode) lane holds an equivalent bare CUDA context plus
        # resident encoders; stop it off-GPU on the residency's card for the same reason as the VAE lane.
        # Stopping it also drops the lane out of the disaggregation liveness predicate, so new jobs route
        # monolithic while the residency holds. A no-op unless disaggregation is enabled and the lane sits here.
        if self._residency_should_pause_component_lane(device_index):
            self._process_lifecycle.pause_component_off_gpu(owner=PauseOwner.WHOLE_CARD)

        if announce or after < current or safety_pause_requested or post_process_paused:
            safety_note = " and requesting safety off-GPU" if safety_pause_requested else ""
            total_mb = forecast.total_vram_mb
            card_phrase = f"the whole ~{total_mb / 1024:.0f}GB card" if total_mb else "nearly the whole card"
            logger.opt(colors=True).warning(
                "<fg #f0beff>Whole-card residency: reserving the device for {} "
                f"(inference processes {current} -> {after} of {self._max_inference_processes}, target "
                f"{target}){safety_note}. Its weights + activations need {card_phrase}; co-resident "
                f"siblings/safety would force the driver to stream activations to host RAM and run several "
                "times slower. {}</>",
                job.model,
                self._process_map.residency_snapshot(),
            )

    def _scale_sparing(
        self,
        target: int,
        *,
        device_index: int | None,
        protected_model: str | None,
        spared_process_id: int | None,
    ) -> int:
        """Scale the inference pool, naming a spared slot only when this caller holds one.

        The spare is an optional extra on top of the model-name protection: a residency that has no slot of
        its own committed yet (its head is resident somewhere, or the caller is a convergence with nothing
        pre-staged) asks for exactly the shrink it always did.
        """
        if spared_process_id is None:
            return self._process_lifecycle.scale_inference_processes(
                target,
                device_index=device_index,
                protected_model=protected_model,
            )
        return self._process_lifecycle.scale_inference_processes(
            target,
            device_index=device_index,
            protected_model=protected_model,
            spared_process_id=spared_process_id,
        )

    def _should_prestage_whole_card_head(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        forecast: StreamForecast,
        available_process: HordeProcessInfo,
        *,
        device_index: int | None = None,
    ) -> bool:
        """Whether a whole-card head should be pre-loaded into a spare's RAM while a live job holds the device.

        ``preload_model`` is a RAM-only load (the weights reach VRAM only at sampling), so a heavy head can
        load into an idle process's RAM concurrently with the in-flight job, and be ready to sample the
        instant the device frees, rather than its multi-GB disk->RAM load only starting after the drain.

        Pre-staging is worthwhile only when:

        - a live job actually holds the device (otherwise the normal whole-card path claims the idle card and
          loads immediately, with nothing to overlap);
        - the head is not already resident or loading somewhere (nothing left to pre-stage);
        - there is an idle spare to hand the preload to (never the live job's own process); and
        - system RAM can hold the head's *weights* alongside the in-flight job, i.e. the operator's "assuming the
          RAM can support it" (see :meth:`_prestage_weights_fit_ram`). A RAM shortfall falls back to the prior
          claim-the-card-and-wait behavior.

        ``device_index`` scopes "a live job holds the device" to one card on a multi-GPU host (the card the
        spare slot sits on); None is the single-GPU / worker-wide case.
        """
        if device_index is None:
            live_jobs_on_device = len(self._job_tracker.jobs_in_progress)
        else:
            live_jobs_on_device = len(self._jobs_in_progress_on_card(device_index))
        if live_jobs_on_device == 0:
            return False
        if self._is_model_forecast_to_load(job.model):
            return False
        if available_process.is_process_busy():
            return False
        return self._prestage_weights_fit_ram(job, baseline, forecast)

    def _prestage_weights_fit_ram(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        forecast: StreamForecast,
    ) -> bool:
        """Whether system RAM can hold the head's *weights* alongside the in-flight job.

        A RAM preload materialises only the model's weights on the CPU offload device; the activation working
        set that inflates the full :func:`predict_job_ram_mb` burden lives in VRAM at sampling time, not in
        RAM. Gating the pre-stage on that full burden over-rejects a head whose weights comfortably fit (a Flux
        fp8 head's ~11.5GB of weights versus its ~24GB activation-inclusive estimate), which is what forces the
        worker to tear every idle sibling down instead of staging. Worse, the establish path it then falls back
        to loads those same weights into RAM with no hard gate at all, so the burden gate held the pre-stage to
        a stricter standard than the path it defers to. So gate on the weight footprint (the forecast's
        ``weights_mb``, the persistent RAM cost of a preload) plus the configured RAM reserve.

        When the weight estimate is unavailable (it should not be once ``needs_exclusive_residency`` is True,
        which requires known weights) fall back to the conservative full-burden RAM budget, so a head whose
        footprint cannot be sized is never force-staged onto a RAM-pressured host.
        """
        available_ram_mb = self._measured_available_ram_mb()
        weights_mb = forecast.weights_mb
        committed_ram_mb = self._reserve_ledger.total_ram_mb()
        if weights_mb is None:
            return self._ram_budget.check_job(
                job,
                baseline,
                available_ram_mb,
                committed_reserve_mb=committed_ram_mb,
            ).fits
        return (available_ram_mb - committed_ram_mb) >= float(weights_mb) + self._ram_budget.reserve_mb

    def _begin_whole_card_residency(
        self,
        job: ImageGenerateJobPopResponse,
        forecast: StreamForecast,
        *,
        announce: bool,
        device_index: int | None = None,
        target_process: HordeProcessInfo | None = None,
    ) -> None:
        """Record a whole-card residency for a head being pre-staged into RAM, without claiming the card yet.

        The device cannot be claimed while a live job holds it, but the heavy head's weights can load into a
        spare's RAM now. This sets the same residency bookkeeping :meth:`_establish_whole_card_residency` does
        (so the cooldown, the restore, and the recovery-supervisor wedge grace all cover the pre-stage load and
        the convergence that follows), minus the process teardown and safety pause: those are deferred to
        :meth:`_converge_whole_card_residency`, which runs once the head is staged and the device frees.

        ``device_index`` scopes the pre-staged residency to one card on a multi-GPU host; None is the
        single-GPU / worker-wide case.

        ``target_process`` is the spare the pre-stage loads into. It is remembered on the residency so the
        convergence that follows spares it: that shrink spares the holder by model name, which a slot still
        mid-load does not carry, so the pre-stage's own target would otherwise be a legal victim of the
        collapse it is loading for.
        """
        self._whole_card_ledger.record_grant(
            device_index,
            model=job.model,
            forecast=forecast,
            cooldown_until=self._clock() + self._whole_card_cooldown_seconds(),
            now=self._clock(),
            establish_grace_seconds=_WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
        )
        if target_process is not None:
            self._whole_card_ledger.state_for(device_index).prestage_process_id = target_process.process_id
        if announce:
            logger.opt(colors=True).info(
                "<fg #f0beff>Pre-staging whole-card head {} into a spare process's RAM while the "
                f"in-flight job finishes; the device will be reserved for it (idle siblings stopped"
                f"{' and safety moved off-GPU' if self._residency_should_pause_safety(device_index) else ''}) "
                f"once it frees, so its weights are loaded before it samples instead of after. "
                "{}</>",
                job.model,
                self._process_map.residency_snapshot(),
            )

    def _converge_whole_card_residency(self) -> None:
        """Collapse an in-progress whole-card residency to sole VRAM residency once its model is staged.

        Driven each scheduling cycle while a residency is held. A pre-staged head is loaded into RAM before
        the device is claimed (see :meth:`_begin_whole_card_residency`); stopping idle siblings before the head
        is actually resident on a process could kill the very spare the pre-stage wants to use, so this waits
        until the head is resident or loading on a process. From then on the scale-down is told this is a
        whole-card collapse (``protected_model``), so it spares only that head's holder and stops the *other*
        idle siblings, including ones holding a model still queued behind the head, which the generic
        scale-down guard would otherwise protect and thereby pin the count above the target forever. Those
        queued jobs wait and reload once the head drains (see :meth:`_restore_siblings_after_whole_card`).
        Reclaiming the siblings' CUDA contexts and moving safety off-GPU leaves the staged head the whole card
        when it samples. The VAE lane is retried here as well: establishment withholds its pause while a
        disaggregated decode is still in flight on it, so the lane leaves the card on the first cycle after
        those decodes drain. A no-op until a residency is held and its model is staged; idempotent at the target.
        Converges every held residency, so on a multi-GPU host each card's pre-staged head collapses its own
        card independently.
        """
        for device_index, state in self._held_residencies():
            model = state.model
            if model is None or not self._whole_card_residency_has_holder(model, device_index):
                continue
            forecast = state.forecast
            target = self._whole_card_ledger.effective_target(state)
            # A forecast that cannot size the card names no depth to converge on, so the pool is left where it
            # is rather than collapsed to sole residency on a figure nobody measured. The rest of the
            # convergence (moving the service lanes off the card) is unaffected by the missing depth.
            live_count = self._process_map.num_loaded_inference_processes(device_index=device_index)
            if target is not None and live_count > target:
                self._scale_sparing(
                    target,
                    device_index=device_index,
                    protected_model=model,
                    spared_process_id=state.prestage_process_id,
                )
            if forecast is not None:
                self._pause_post_process_for_residency_if_idle(device_index, model_name=model, forecast=forecast)
            self._pause_vae_lane_for_residency_if_drained(device_index)

    def _whole_card_residency_has_holder(self, model: str, device_index: int | None) -> bool:
        """Whether a held whole-card model is staged or resident on a live process.

        Convergence must wait until the pre-staged head has a holder so the scale-down can spare that process.
        Once the holder reports ready it may sit in ``WAITING_FOR_JOB`` rather than a preload state, so the
        generic "forecast to load" predicate is too narrow here.
        """
        return any(
            process.process_type is HordeProcessType.INFERENCE
            and process.loaded_horde_model_name == model
            and (device_index is None or process.device_index == device_index)
            for process in self._process_map.values()
        )

    def _prestaged_whole_card_not_ready(self, job: ImageGenerateJobPopResponse) -> bool:
        """Whether ``job`` must wait for its in-progress whole-card residency to claim the card before sampling.

        A pre-staged whole-card head is loaded into RAM (see :meth:`_begin_whole_card_residency`) before the
        device is reserved, so dispatching it would commit its weights to VRAM while idle siblings (or the
        just-drained busy process) still hold their CUDA contexts, forcing the first step to stream. This
        returns True until the residency has converged, i.e. the live inference-process count is at the forecast's
        target, safety is off-GPU if this residency needs it, and the card has drained enough to load the
        weights (the same :meth:`_whole_card_teardown_exhausted` gate the non-pre-staged path loads under).

        Returns False for any job that is not the currently-held residency's model, so ordinary dispatch (and
        the non-pre-staged whole-card path, which only preloads once already at sole residency) is unaffected.
        """
        found, device_index = self._residency_holder_for_model(job.model)
        if not found:
            return False
        # Re-use the residency's stored forecast for the readiness check rather than re-deriving from the
        # current (possibly degraded) state: it carries the stable weight footprint, the budget-relative target,
        # and the fits_alone guarantee captured at establishment, which the live device reading and the bounded
        # drain backstop in _whole_card_teardown_exhausted then resolve against the real, post-teardown VRAM.
        stored = self._residency_state(device_index).forecast
        if stored is not None:
            # The stored forecast reflects the residency's actual budget-relative target and the
            # weight footprint at establishment time; only re-derive when it was never captured.
            return not self._whole_card_teardown_exhausted(stored, device_index=device_index)
        baseline = self._model_metadata.get_baseline(job.model) if job.model is not None else None
        forecast = self._forecast_streaming(job, baseline, device_index=device_index)
        return not self._whole_card_teardown_exhausted(forecast, device_index=device_index)

    def _resident_whole_card_head_ready(
        self,
        job: ImageGenerateJobPopResponse,
        process_with_model: HordeProcessInfo,
    ) -> bool:
        """Ensure an already-resident whole-card head has sole residency before it samples.

        The ordinary whole-card path runs during preload admission, so it used to miss a heavy head whose
        model was already resident on an idle process while sibling processes still held their own models.
        That job is entitled to the same residency as a to-be-loaded head: establish the residency, evict
        sibling VRAM, and defer dispatch until the teardown is complete.
        """
        if job.model is None:
            raise ValueError(f"job.model is None ({job})")
        if not self._budget_active() or not self._whole_card_residency_enabled():
            return True

        target_device_index = process_with_model.device_index if self._multi_gpu_routing_active else None
        baseline = self._model_metadata.get_baseline(job.model)
        forecast = self._forecast_streaming(job, baseline, device_index=target_device_index)
        live_inference_processes = self._process_map.num_loaded_inference_processes(
            device_index=target_device_index,
        )
        if not self._whole_card_ledger.residency_demanded(
            forecast,
            enabled=self._whole_card_residency_enabled(),
            is_head_blocker=True,
            live_inference_process_count=live_inference_processes,
        ):
            if forecast.needs_process_count_reduction:
                self._log_whole_card_reduction_suppressed(
                    job,
                    forecast,
                    live_inference_processes=live_inference_processes,
                )
            return True
        # A disaggregation-class job never claims the device, the same contract the admission-time decision
        # holds (see :meth:`_decide_whole_card_demand`): its sampler runs UNet-only and co-resides with the
        # encode lane and other samplers by design. Establishing here is worse than doing so at admission,
        # because the job has already been dispatched disaggregated: the teardown would stop the very lanes
        # its own encode and decode run on. Decided on class-eligibility, not liveness, so the contract holds
        # while a lane is transiently paused.
        if self._is_disaggregation_class_eligible(job):
            self._log_stream_forecast(job, forecast)
            return True
        if not self._whole_card_warranted(forecast):
            self._log_whole_card_declined(job, forecast)
            return True

        # The same churn governors that gate a new establishment at preload admission gate it here: this path
        # otherwise claims the card for a head whose weights are already resident, so a head the governors
        # coerced onto the co-resident path would win the card back at dispatch time, spending the exhausted
        # allowance and cycling siblings all over again. The dwell state is shared with the admission path
        # through the ledger, so a head that already sat out its dwell dispatches co-resident immediately.
        held = self._whole_card_ledger.get(target_device_index)
        if held is None or held.model is None or held.model != job.model:
            governor_hold = self._whole_card_governor_hold(target_device_index, now=self._clock())
            if governor_hold is not None:
                elapsed, dwell_exhausted = self._whole_card_ledger.note_governor_defer(
                    target_device_index,
                    model=job.model,
                    now=self._clock(),
                )
                self._disclose_whole_card_governor_hold(
                    job,
                    governor_hold,
                    device_index=target_device_index,
                    elapsed=elapsed,
                    downgraded=dwell_exhausted,
                )
                # Past the dwell the head runs with whatever residency it already has; inside it, the brief
                # hold gives the governor the chance to release before sole residency is forfeited.
                return dwell_exhausted

        first_time = not self._job_tracker.is_admitted_exclusive(job)
        # Disclose the arithmetic behind a dispatch-time claim, which is otherwise established silently:
        # without it a residency that appears between a job's admission and its dispatch carries no forecast
        # anywhere in the log. The disclosure coalesces on its own decision key, so re-asking every tick does
        # not repeat it.
        self._log_stream_forecast(job, forecast)
        self._job_tracker.mark_admitted_exclusive(job, device_index=target_device_index)
        self._establish_whole_card_residency(
            job,
            forecast,
            announce=first_time,
            device_index=target_device_index,
            target_process=process_with_model,
        )
        self.unload_models_from_vram(
            process_with_model,
            under_pressure=True,
            for_head_of_queue=True,
            device_index=target_device_index,
        )
        return self._whole_card_teardown_exhausted(forecast, device_index=target_device_index)

    def _residency_holder_for_model(self, model: str | None) -> tuple[bool, int | None]:
        """Return ``(found, device_index)`` for the card whose held whole-card residency is for ``model``.

        ``found`` distinguishes a genuine hit on the ``None`` (single-GPU / worker-wide) key from a miss, since
        ``None`` is itself a valid residency key.
        """
        return self._whole_card_ledger.holder_for_model(model)

    def _whole_card_teardown_exhausted(self, forecast: StreamForecast, *, device_index: int | None = None) -> bool:
        """Whether a whole-card residency has done all it can and the head can now load best-effort.

        The whole-card branch defers a heavy head while a teardown can still make room: idle siblings left to
        stop, the safety process still on-GPU, or their freed VRAM still draining. The first two are the
        *structural* hold and are decided on topology alone (live process count at or below the forecast's
        target, safety off-GPU if this residency needs it). Once both hold the teardown is structurally
        complete: the model fits alone (``fits_alone``, the grant precondition for a whole-card residency), so
        the only remaining question is whether the asynchronously-freed VRAM has actually materialised.

        That last step is resolved against the live device, not the stale establishment forecast (whose
        ``free_now_mb`` was captured before the teardown freed the siblings' VRAM, so reading it would park the
        head forever once it drains): the *live* free-VRAM reading dispatches the head the moment it confirms the
        drain (safe to read here, at sole residency, where it only rises as the stopped contexts release), and a
        bounded ``_WHOLE_CARD_DRAIN_SETTLE_SECONDS`` backstop admits it on the structural sole-residency
        guarantee if the measurement is unavailable or lags, so the head never parks indefinitely. That
        guarantee is sized for a card every other context has left, so the safety context this residency is
        leaving in place (:meth:`_resident_safety_charge_mb`) is priced back out of it: otherwise the backstop
        admits the head against room only a departure the configuration forbids would free, and the weights
        load into a card short by roughly the safety footprint. A model that still cannot fit co-resident even
        at sole residency loads best-effort the same way and samples slowly under the over-budget step grace
        rather than wedging the queue until the recovery supervisor soft-resets.

        ``device_index`` scopes the live-context count and the safety check to one card on a multi-GPU host;
        None is the single-GPU / worker-wide case.
        """
        state = self._residency_state(device_index)
        return self._whole_card_ledger.teardown_complete(
            forecast,
            loaded_process_count=self._process_map.num_loaded_inference_processes(device_index=device_index),
            safety_clear_of_card=self._safety_clear_of_residency_card(device_index),
            process_target=self._whole_card_ledger.effective_target(state),
            post_process_pause_required=(
                self._residency_should_pause_post_process(device_index)
                and not self._post_process_context_fits_with_residency(forecast, device_index=device_index)
            ),
            post_process_cleared=self._process_map.num_post_process_processes(device_index=device_index) == 0,
            component_lane_pause_required=self._residency_should_pause_component_lane(device_index),
            component_lane_cleared=self._process_map.num_component_processes(device_index=device_index) == 0,
            weights_fit_live=self._whole_card_weights_fit_live(
                forecast,
                device_index=device_index,
                model=state.model,
            ),
            drain_backstop_elapsed=self._whole_card_drain_backstop_elapsed(device_index),
            resident_context_charge_mb=self._resident_safety_charge_mb(device_index),
            device_index=device_index,
            now=self._clock(),
        )

    def _resident_safety_charge_mb(self, device_index: int | None) -> float:
        """The device charge of a safety context this card's residency is leaving where it is.

        A whole-card residency only reaches sole residency where it actually moves safety off the card. Where
        the configuration keeps safety on-GPU (:meth:`_residency_should_pause_safety` is False) and safety is
        still holding its context, the residency's structural sole-residency figure over-states the room the
        head will find by :meth:`_safety_footprint_mb`, so that charge is what a caller leaning on the
        figure must price back out. Zero once safety is off-GPU, or where this residency is going to move it.
        """
        if not self._safety_on_gpu_permitted:
            return 0.0
        if self._residency_should_pause_safety(device_index):
            return 0.0
        if self._process_lifecycle.is_safety_gpu_paused:
            return 0.0
        return self._safety_footprint_mb()

    def _whole_card_weights_fit_live(
        self,
        forecast: StreamForecast,
        *,
        device_index: int | None = None,
        model: str | None = None,
    ) -> bool:
        """Whether the residency model's weights fit the *live* measured free VRAM (read only at sole residency).

        Keyed on the live device reading rather than the forecast's stored ``free_now_mb`` (captured at
        establishment, before the teardown freed the siblings' VRAM). Only the caller's structural-completion
        guard makes this safe to trust: at sole residency the reading is monotonic, only rising as the
        stopped siblings' contexts release, so it never reads deceptively high the way an instantaneous
        reading does during startup (idle contexts not yet allocated reading as free). Unknown weight or
        measurement returns False so the bounded structural backstop, not a guess, drives the fallback.

        Weights already committed to VRAM are the answered case: a free-VRAM reading taken while the residency
        model itself is resident already excludes those weights, so comparing the model's full weight figure
        against it asks the card for room it is spending on this very model. That comparison can never pass
        while the model is resident, which parks every dispatch of a resident head on the drain-settle backstop
        even though the fit it is waiting to confirm is already a physical fact. Residency of the model is
        therefore the fit.
        """
        if model is not None and self._whole_card_model_weights_resident(model, device_index=device_index):
            return True
        if forecast.weights_mb is None:
            return False
        free_now = self._measured_free_vram_mb(device_index=device_index)
        if free_now is None:
            return False
        return (free_now - forecast.weights_mb) >= forecast._effective_base_reserve  # noqa: SLF001

    def _whole_card_model_weights_resident(self, model: str, *, device_index: int | None) -> bool:
        """Whether ``model``'s weights already occupy VRAM on an inference process of this card.

        ``device_index`` scopes the question to one card on a multi-GPU host; None accepts any inference
        process (the single-GPU / worker-wide case).
        """
        return any(
            process_info.process_type is HordeProcessType.INFERENCE
            and (device_index is None or process_info.device_index == device_index)
            and self._candidate_weights_resident_on_process(model, process_info.process_id)
            for process_info in self._process_map.values()
        )

    def _whole_card_drain_backstop_elapsed(self, device_index: int | None) -> bool:
        """Whether the bounded drain-settle window has elapsed since this residency was established.

        The deterministic backstop for the dispatch gate: once a structurally-complete teardown has held for
        ``_WHOLE_CARD_DRAIN_SETTLE_SECONDS`` without the live reading confirming the drain, the head is admitted
        on the structural ``fits_alone`` guarantee rather than parking forever. Measured from the moment the
        teardown's structural legs first all passed, so a slow establishment does not spend the backstop before
        the sole-residency guarantee it admits the head against exists; once that guarantee holds, a stuck or
        unavailable free-VRAM measurement can never wedge the head.
        """
        return self._whole_card_ledger.drain_backstop_elapsed(
            device_index,
            now=self._clock(),
            settle_seconds=_WHOLE_CARD_DRAIN_SETTLE_SECONDS,
        )

    def is_whole_card_residency_active(self) -> bool:
        """Whether any card currently holds a whole-card residency lease (its cooldown still running).

        Mirrors the ``active`` field of :meth:`whole_card_residency_state` but without building the full
        snapshot, so the job popper's large-model re-entry cooldown can cheaply ask "is the lease up?" every
        pop cycle: the lease is up exactly when this returns False (no card holds a residency model).
        """
        return self._whole_card_ledger.any_held()

    def whole_card_pop_claim(self) -> WholeCardPopClaim | None:
        """Return the standing whole-card claim over the pop offer, or None when the pool is unclaimed.

        The typed answer to the only question intake asks about a residency: is one claiming the offer, and
        for which model. While a claim stands the worker advertises exactly that model, so the horde stops
        sending work whose arrival would force the resident weights back to host RAM or off the card. The
        claim's end conditions live with the residency (``WholeCardResidencyLedger.pop_claim``); intake only
        reads it.

        Two conditions keep the claim off hosts it would harm. It is skipped entirely while more than one
        card serves work, because a residency is per-card and holding the whole worker's intake to one card's
        model would starve the others; and it is skipped when the operator has set the maximum hold to zero,
        which disables the claim along with the cap that bounds it.
        """
        if self._multi_gpu_routing_active:
            self._disclose_pop_claim_skipped_on_multi_gpu()
            return None
        max_hold_seconds = self._whole_card_max_hold_seconds()
        if max_hold_seconds <= 0.0:
            return None
        now = self._clock()
        for device_index, _state in self._held_residencies():
            claim = self._whole_card_ledger.pop_claim(
                device_index,
                now=now,
                max_hold_seconds=max_hold_seconds,
            )
            if claim is not None:
                return claim
        return None

    def note_whole_card_pop_outcome(self, *, served: bool) -> None:
        """Record what a pop taken under the standing claim came back with (the empty-pop evidence).

        Called by the popper once per concluded attempt made under a claim, never for an attempt that failed
        to reach the horde: a request that errored says nothing about whether the resident model has demand.
        A run of empty answers releases the claim early, which is what stops a resident model nobody wants
        from holding the whole pool until the maximum hold expires.
        """
        claim = self.whole_card_pop_claim()
        if claim is None:
            return
        released = self._whole_card_ledger.note_pop_outcome(
            claim.device_index,
            served=served,
            now=self._clock(),
        )
        if released:
            self._pop_claim_empty_release_pending = True

    def _disclose_pop_claim_skipped_on_multi_gpu(self) -> None:
        """State once that a residency on this host does not claim the offer, and why.

        The offer is worker-wide while a residency is per-card, so on a multi-card host narrowing intake to
        one card's resident model would leave the other cards with nothing to serve. Silence would read as
        the feature being broken rather than deliberately inapplicable, so the first residency to be held
        here says so.
        """
        if self._pop_claim_multi_gpu_disclosed or not self._whole_card_ledger.any_held():
            return
        self._pop_claim_multi_gpu_disclosed = True
        logger.info(
            "Whole-card residency is held on a multi-card host, so it does not claim the pop offer: the offer "
            "is worker-wide and narrowing it to one card's model would starve the others.",
        )

    def _whole_card_max_hold_seconds(self) -> float:
        """Longest one whole-card residency may own the card and the offer, from operator configuration.

        Coerced defensively because a partially-mocked configuration would otherwise put a non-numeric value
        into a comparison on the pop path; a value that is not a number reads as the feature being off.
        """
        configured = self._runtime_config.bridge_data.whole_card_residency_max_hold_seconds
        if isinstance(configured, bool) or not isinstance(configured, (int, float)):
            return 0.0
        return float(configured)

    def _disclose_pop_claim_edge(self) -> None:
        """Emit the one-line engage/release disclosure when the pop claim's state actually changes.

        Edge-triggered against the claim last surfaced, so a claim held across many ticks is stated once and
        a released one is stated once. The release names which of the claim's ends fired, since the remedies
        differ: a cap expiry says the burst outlasted its window, an empty-pop release says the demand went
        away, and neither reads the same as the residency simply finishing its work. The same end is retained
        for the status snapshot, which is asked about the offer widening back out after the line has scrolled.
        """
        claim = self.whole_card_pop_claim()
        previous = self._pop_claim_disclosed
        if claim is not None:
            if previous is not None and previous.model == claim.model:
                return
            self._pop_claim_disclosed = claim
            self._pop_claim_empty_release_pending = False
            self._pop_claim_release = None
            logger.info(
                f"Whole-card pop claim engaged for {claim.model}: advertising that model alone while it holds "
                f"the card, for at most {claim.expires_at - claim.held_since:.0f}s.",
            )
            return
        if previous is None:
            return
        self._pop_claim_disclosed = None
        if self._pop_claim_empty_release_pending:
            release = WholeCardPopClaimRelease.NO_FURTHER_WORK
            reason = "the horde had no further work for it"
        elif self._clock() >= previous.expires_at:
            release = WholeCardPopClaimRelease.MAXIMUM_HOLD
            reason = "the maximum hold elapsed"
        else:
            release = WholeCardPopClaimRelease.RESIDENCY_RELEASED
            reason = "the residency released"
        self._pop_claim_empty_release_pending = False
        self._pop_claim_release = (release, self._clock())
        logger.info(f"Whole-card pop claim released for {previous.model}: {reason}; advertising the full pool again.")

    def _recent_pop_claim_release(self, now: float) -> WholeCardPopClaimRelease | None:
        """Return why the last claim ended while that still explains the offer, else None.

        The release is only an answer to "why is the full pool being advertised again" for as long as the
        question is being asked about this release; past
        :data:`_POP_CLAIM_RELEASE_VISIBLE_SECONDS` an unclaimed offer is just the ordinary state and a stale
        reason beside it would read as a claim that had only just ended.
        """
        if self._pop_claim_release is None:
            return None
        release, released_at = self._pop_claim_release
        if (now - released_at) >= _POP_CLAIM_RELEASE_VISIBLE_SECONDS:
            return None
        return release

    def whole_card_residency_grace_active(self) -> bool:
        """Whether a whole-card residency is establishing, so the held queue is intentional (not a wedge).

        While true, the recovery supervisor must not treat the deliberately-deferred heavy head (waiting
        for idle siblings to stop, the safety process to cycle off-GPU, and ~11GB of weights to load) as a
        structural queue wedge and soft-reset the pools mid-setup. A granted window holds for its own
        duration (``_WHOLE_CARD_ESTABLISH_GRACE_SECONDS`` or ``_WHOLE_CARD_RESTORE_GRACE_SECONDS``), which is
        the liveness bound on a residency that never loads; how often a card may open a new window is
        governed at admission, where an establishment is deferred while the card's rolling grace budget is
        spent. Public: read by the process manager's wedge assessment.
        """
        return self._whole_card_ledger.grace_active(
            now=self._clock(),
            establish_grace_seconds=_WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
            restore_grace_seconds=_WHOLE_CARD_RESTORE_GRACE_SECONDS,
        )

    def whole_card_governor_defer_active(self) -> bool:
        """Whether a churn governor is deferring a head's whole-card establishment, so a held queue is chosen.

        The governor brakes how fast a card may be rotated; while it holds, the head does not take the card
        and the smaller work behind it is admitted by ordinary measured admission. That window can legitimately
        present as an idle card with pending work, which is the raw shape a structural wedge is read from, so
        the recovery supervisor must not answer a governance brake with constructive remedies or a pool reset.
        Bounded by the ledger's defer dwell, after which the head stops asking for the card and is served
        co-resident, so a governor that never releases still leaves the wedge assessment reachable. Public:
        read by the process manager's wedge assessment.
        """
        return self._whole_card_ledger.any_governor_defer_active(now=self._clock())

    def heavy_head_load_grace_active(self) -> bool:
        """Whether a heavy head admitted off the whole-card path is still inside its bounded load window.

        A model that streams even with the whole card to itself never enters the whole-card branch, so
        ``whole_card_residency_grace_active`` does not cover it; but its multi-gigabyte load holds the queue
        just the same. While true the recovery supervisor must not treat that deliberate hold as a structural
        wedge and give up the never-run backlog. Bounded by ``_HEAVY_HEAD_LOAD_GRACE_SECONDS`` so a head that
        genuinely never loads still trips the supervisor. Public: read by the process manager's wedge assessment.
        """
        if self._heavy_head_admitted_at == 0.0:
            return False
        return (self._clock() - self._heavy_head_admitted_at) < _HEAVY_HEAD_LOAD_GRACE_SECONDS

    def ram_reclaim_cycle_grace_active(self) -> bool:
        """Whether a deliberate RAM-reclaim process cycle is still inside its bounded respawn/preload window.

        When the RAM budget cannot fit the next head and cycles an idle slot to return allocator-retained
        RAM to the OS (:meth:`_replace_stale_ram_unload_process`), the slot respawns and the head must then
        preload onto it. The queue is unservable across that window, but by the worker's own deliberate,
        bounded action, not a wedge. While true the recovery supervisor must not treat the held queue as a
        structural wedge and fault the servable backlog. Bounded by ``_RAM_RECLAIM_CYCLE_GRACE_SECONDS`` so a
        cycle that genuinely never recovers still trips the supervisor. Public: read by the process manager's
        wedge assessment.
        """
        if self._ram_reclaim_cycle_at == 0.0:
            return False
        return (self._clock() - self._ram_reclaim_cycle_at) < _RAM_RECLAIM_CYCLE_GRACE_SECONDS

    def card_residency(self, device_index: int | None) -> tuple[str | None, str]:
        """Return ``(model, phase)`` for the whole-card residency held on ``device_index`` (per-card view).

        ``model`` is None when this card holds no residency; otherwise ``phase`` is ``establishing`` while the
        establish grace is still in effect, else ``holding``; this is the same phase split the worker-wide
        :meth:`whole_card_residency_state` reports. The single-GPU worker-wide residency lives under the
        ``None`` key, so a single-GPU caller reads it by passing ``device_index=None``. Reads without creating:
        a card with no residency is left absent from the map.
        """
        model, phase = self._whole_card_ledger.phase(
            device_index,
            now=self._clock(),
            establish_grace_seconds=_WHOLE_CARD_ESTABLISH_GRACE_SECONDS,
        )
        if model is None:
            return None, ""
        return model, str(phase)

    def whole_card_residency_state(self) -> WholeCardResidencyState:
        """Return a read-only view of the whole-card residency posture, for the status snapshot/TUI.

        ``possible`` is config + topology only (feature on, the VRAM budget on, and something is actually
        tear-down-able: more than one inference process, or a safety process that can be moved off-GPU);
        it powers the operator heads-up so a teardown is not a surprise. The remaining fields describe a
        residency that is currently held (its model, the establish/hold phase, the reduced process count,
        the safety-pause state, the establishing forecast's hard numbers for the detailed view, and the claim
        it holds over the pop offer or the reason its last claim ended).
        Tolerant of partially-mocked config (used in tests that build snapshots): config flags are read
        with boolean coercion so a non-bool never leaks a truthy Mock into ``possible``.
        """
        bridge_data = self._runtime_config.bridge_data
        enabled = self._whole_card_residency_enabled()
        budget_on = bridge_data.enable_vram_budget is True
        safety_off_enabled = bool(self._whole_card_safety_off_gpu_enabled())
        multi_process = self._max_inference_processes > 1
        possible = enabled and budget_on and (multi_process or safety_off_enabled)

        # Represent the posture with the first held residency (single-GPU has at most one).
        # ``active`` is true while any card holds a residency.
        held = self._held_residencies()
        representative_index, representative = held[0] if held else (None, None)
        model = representative.model if representative is not None else None
        active = model is not None
        forecast = representative.forecast if representative is not None else None
        now = self._clock()

        phase = ""
        cooldown_remaining: float | None = None
        processes_target = 0
        weights_mb = reserve_mb = free_now_mb = free_if_alone_mb = None
        max_resident_processes: int | None = None
        if active and representative is not None:
            establishing = (
                representative.established_at != 0.0
                and (now - representative.established_at) < _WHOLE_CARD_ESTABLISH_GRACE_SECONDS
            )
            phase = "establishing" if establishing else "holding"
            cooldown_remaining = max(0.0, representative.cooldown_until - now)
            if forecast is not None:
                weights_mb = forecast.weights_mb
                reserve_mb = forecast.reserve_mb
                free_now_mb = forecast.free_now_mb
                free_if_alone_mb = forecast.free_if_alone_mb
                max_resident_processes = self._whole_card_ledger.effective_target(representative)
            processes_target = max_resident_processes or 1

        total_vram_mb = (
            forecast.total_vram_mb
            if forecast is not None
            else self._process_map.get_reported_total_vram_mb(device_index=representative_index)
        )

        # The claim is a separate fact from the residency: a card can hold a model without narrowing the
        # offer to it, and the offer is what an operator watching models come and go actually sees change.
        pop_claim = self.whole_card_pop_claim()
        pop_claim_release = self._recent_pop_claim_release(now) if pop_claim is None else None

        return WholeCardResidencyState(
            possible=possible,
            enabled=enabled,
            safety_off_gpu_enabled=safety_off_enabled,
            cooldown_seconds=self._whole_card_cooldown_seconds(),
            per_process_overhead_mb=self._per_process_overhead_mb(representative_index),
            total_vram_mb=total_vram_mb,
            active=active,
            model=model,
            phase=phase,
            safety_paused=bool(self._process_lifecycle.is_safety_gpu_paused),
            processes_now=self._process_map.num_loaded_inference_processes(),
            processes_target=processes_target,
            processes_max=self._max_inference_processes,
            cooldown_remaining_seconds=cooldown_remaining,
            pop_claim_model=pop_claim.model if pop_claim is not None else None,
            pop_claim_remaining_seconds=max(0.0, pop_claim.expires_at - now) if pop_claim is not None else None,
            pop_claim_release=pop_claim_release.value if pop_claim_release is not None else None,
            weights_mb=weights_mb,
            reserve_mb=reserve_mb,
            free_now_mb=free_now_mb,
            free_if_alone_mb=free_if_alone_mb,
            max_resident_processes=max_resident_processes,
        )

    def _whole_card_safety_off_gpu_enabled(self) -> bool:
        """Whether a whole-card job should move the safety process off-GPU (config + safety actually on-GPU)."""
        return self._runtime_config.bridge_data.whole_card_residency_safety_off_gpu and self._safety_on_gpu_permitted

    def _whole_card_cooldown_seconds(self) -> float:
        """Operator-configured seconds to hold a whole-card residency after its last job drains."""
        return float(self._runtime_config.bridge_data.whole_card_residency_cooldown_seconds)

    def _residency_can_never_converge(
        self,
        device_index: int | None,
        state: WholeCardResidency,
        *,
        now: float,
    ) -> bool:
        """Whether a held residency has lost every route to either running its model or draining.

        A residency is normally retained while its model still has queued or in-flight work, because that work
        is what it was taken out to serve. The retention becomes self-defeating when the model has no holder
        left on the card (a pool rebuild can drop a pre-stage while the ledger entry survives), nothing is
        staging it, and the undispatched head is some other model: the preload pass targets the head, the head
        is barred by this very residency, and no path remains to give the residency a holder or let it drain.
        Such a residency is released through the normal restore path instead.

        The no-holder condition is only trusted past the establish grace, since a residency legitimately has no
        holder for as long as its pre-stage preload takes to land.
        """
        model = state.model
        if model is None:
            return False
        if state.established_at == 0.0 or (now - state.established_at) < _WHOLE_CARD_ESTABLISH_GRACE_SECONDS:
            return False
        if self._whole_card_residency_has_holder(model, device_index):
            return False
        if self._horde_model_map.is_model_loading(model):
            return False
        prestage_process_id = state.prestage_process_id
        if prestage_process_id is not None:
            # The pre-stage target is remembered for the life of the residency, so its mere existence says
            # nothing; only a slot still mid-load is a route to a holder.
            prestage_process = self._process_map.get(prestage_process_id)
            if (
                prestage_process is not None
                and prestage_process.is_process_alive()
                and prestage_process.last_process_state
                in (HordeProcessState.PRELOADING_MODEL, HordeProcessState.DOWNLOADING_MODEL)
            ):
                return False
        # Only a job actually sampling keeps the residency: a pending exclusive admit for this model is exactly
        # the work that has no route to a holder, and its own claim on the card is what bars every other head.
        if self._job_tracker.has_exclusive_job_running(device_index):
            return False
        head = self._undispatched_head()
        return not (head is not None and head.model == model)

    def _restore_siblings_after_whole_card(self) -> None:
        """Restore inference concurrency and safety-on-GPU after a whole-card residency has fully drained.

        Held while the residency model is still pending or in progress, and for the configured cooldown after
        that, so a burst of heavy jobs reuses one residency rather than each thrashing the process count and
        the safety process. Once neither condition holds, that card's sibling processes are grown back to its
        ceiling and, if the residency was on the safety card, the safety process is restored to the GPU.
        Restores every drained card's residency independently; a no-op when none is outstanding.

        The operator's maximum hold sits above both retention rules: past it (or once the empty-pop evidence
        has ended the residency's claim over the offer) the cooldown neither refreshes nor holds, so one
        residency episode cannot own the card indefinitely on the strength of demand it is itself the only
        source of.

        The restore's wedge-grace window is granted for the churn the restore actually creates: siblings still
        respawning, or a service lane restarted. A release that finds the pool already at its ceiling and no
        lane to restart changed nothing, so it takes no window; leaving one standing would tell the recovery
        supervisor to ignore a genuine wedge for its whole duration after every such release, and would charge
        the card's rolling allowance for teardown churn that never happened.
        """
        now = self._clock()
        max_hold_seconds = 0.0 if self._multi_gpu_routing_active else self._whole_card_max_hold_seconds()
        active_models = {j.model for j in self._job_tracker.jobs_in_progress}
        active_models.update(j.model for j in self._job_tracker.jobs_pending_inference)
        for device_index, state in self._held_residencies():
            model = state.model
            # Past the maximum hold (or once the empty-pop evidence has ended the claim) the residency stops
            # being retained: it neither refreshes its cooldown nor honours a standing one, so it lets go as
            # soon as its own accepted work has drained and the full model pool returns. Retention is all it
            # ends; work in flight finishes and a granted grace window still runs its own duration.
            retention_ended = self._whole_card_ledger.pop_claim_retention_ended(
                device_index,
                now=now,
                max_hold_seconds=max_hold_seconds,
            )
            if (
                model in active_models or self._job_tracker.has_exclusive_job_in_progress(device_index)
            ) and not self._residency_can_never_converge(device_index, state, now=now):
                # Still serving the residency; keep it (refresh the cooldown so it survives the lull between
                # back-to-back heavy jobs).
                if not retention_ended:
                    state.cooldown_until = now + self._whole_card_cooldown_seconds()
                continue
            # A ready head for a different model may cut the cooldown short, but not before the residency has
            # held long enough to amortize the teardown and the regrowth the release itself will pay for;
            # otherwise a queue alternating heavy and light heads rebuilds the pool on every job.
            ready_different_model_head = self._ready_different_model_head_on_device(
                residency_model=model,
                device_index=device_index,
            )
            min_hold_disclosure = (
                self._whole_card_ledger.min_hold_disclosure(device_index, now=now)
                if ready_different_model_head
                else None
            )
            if min_hold_disclosure is not None:
                self._disclose_whole_card_min_hold(device_index, model, min_hold_disclosure)
            preempt_cooldown = ready_different_model_head and min_hold_disclosure is None
            if self._clock() < state.cooldown_until and not preempt_cooldown and not retention_ended:
                # Drained, but hold the residency through the cooldown so an imminent heavy job reuses it.
                continue
            # The restore's own churn (respawning siblings, cycling safety back on-GPU) briefly makes the queue
            # unservable; mark its start so the wedge grace covers it (see _WHOLE_CARD_RESTORE_GRACE_SECONDS).
            # The window is granted here, before the churn is ordered, so no reader can see the release without
            # its excuse; it is withdrawn below if the release turns out to have had no churn to cover.
            granted_at = self._clock()
            self._whole_card_ledger.record_restore(
                device_index,
                now=granted_at,
                restore_grace_seconds=_WHOLE_CARD_RESTORE_GRACE_SECONDS,
            )
            self._min_hold_disclosed.pop(device_index, None)
            if self._reclaim_ladder is not None:
                self._reclaim_ladder.discharge_context_reduction(device_index)
            post_process_restored = (
                self._process_lifecycle.restore_post_process_off_gpu(owner=PauseOwner.WHOLE_CARD)
                if self._residency_should_pause_post_process(device_index)
                and not self._state.post_processing_disabled_by_breaker
                else False
            )
            vae_lane_restored = (
                self._process_lifecycle.restore_vae_lane_off_gpu(owner=PauseOwner.WHOLE_CARD)
                if self._residency_should_pause_vae_lane(device_index)
                else False
            )
            component_lane_restored = (
                self._process_lifecycle.restore_component_off_gpu(owner=PauseOwner.WHOLE_CARD)
                if self._residency_should_pause_component_lane(device_index)
                else False
            )
            lanes_restarting = post_process_restored or vae_lane_restored or component_lane_restored
            ceiling = self._residency_restore_ceiling(device_index)
            current = self._process_map.num_loaded_inference_processes(device_index=device_index)
            if current >= ceiling and not lanes_restarting:
                # The pool is already at its ceiling and no lane was restarted: this release changed nothing.
                self._whole_card_ledger.close_restore_window(device_index, granted_at=granted_at)
                continue
            after = self._process_lifecycle.scale_inference_processes(ceiling, device_index=device_index)
            self._reconcile_worker_shed_to_pool()
            regrown_to_ceiling = isinstance(after, int) and after >= ceiling
            if not lanes_restarting and regrown_to_ceiling:
                # Every sibling was back before this call returned, so there is no respawn still in flight for
                # the window to cover. A pool that has not reached its ceiling keeps it: the spawns it is
                # waiting on are exactly the churn the wedge grace exists to excuse.
                self._whole_card_ledger.close_restore_window(device_index, granted_at=granted_at)
            post_process_note = " and restarting the post-processing lane" if post_process_restored else ""
            vae_lane_note = " and restarting the VAE lane" if vae_lane_restored else ""
            component_lane_note = " and restarting the component lane" if component_lane_restored else ""
            logger.opt(colors=True).info(
                "<fg #7b7d7d>Whole-card residency for {} complete; restoring inference processes "
                f"({current} -> {after} of {ceiling})"
                f"{post_process_note}{vae_lane_note}{component_lane_note}.</>",
                model,
            )
        self._disclose_pop_claim_edge()

    def _disclose_whole_card_min_hold(
        self,
        device_index: int | None,
        model: str | None,
        disclosure: str,
    ) -> None:
        """State once per residency episode that the minimum hold is what kept a ready head off the card.

        The floor holds a different-model head without returning any admission verdict, so silence leaves the
        wait attributable only to the cooldown, which a ready head is otherwise allowed to cut short. Keyed to
        the card and the residency's model so one episode speaks once however many cycles it holds for, and a
        later residency speaks for itself; cleared when the residency restores.
        """
        if self._min_hold_disclosed.get(device_index) == model:
            return
        self._min_hold_disclosed[device_index] = model
        logger.info(f"A ready different-model head may not release this residency yet: {disclosure}.")

    def _ready_different_model_head_on_device(
        self,
        *,
        residency_model: str | None,
        device_index: int | None,
    ) -> bool:
        """Return whether a ready queue head on this card should preempt a drained residency cooldown."""
        head = self._undispatched_head()
        if head is None or head.model is None or head.model == residency_model:
            return False
        process_info = self._resident_process_for_job(head)
        if process_info is None or not process_info.can_accept_job():
            return False
        return device_index is None or process_info.device_index == device_index

    def _reconcile_worker_shed_to_pool(self) -> None:
        """Realign the RAM governor's worker-wide shed record with the live inference-process count.

        The RAM governor records a worker-wide shed so its own restore can grow the pool back once RAM
        proves headroom. When a different mechanism grows the pool instead (the whole-card residency
        restore), that record would otherwise persist as a stale claim that the pool is still short of
        plan, and while the host stays under its RAM floor the governor re-sheds the pool the residency
        just regrew. Recompute the record from the live count against the recorded plan: drop it once the
        pool is back at (or above) plan, otherwise set the shortfall to the true remaining gap. A no-op on a
        multi-GPU host, whose reduction tracks per-card shedding rather than a worker-wide record.
        """
        worker_shed = self._ram_governor_state.worker_shed
        if worker_shed is None:
            return
        loaded = self._process_map.num_loaded_inference_processes()
        if loaded >= worker_shed.planned_process_count:
            self._ram_governor_state.worker_shed = None
        else:
            worker_shed.shed_process_count = worker_shed.planned_process_count - loaded

    def _residency_restore_ceiling(self, device_index: int | None) -> int:
        """The process count to grow back to when a card's whole-card residency is restored.

        That card's own ``target_process_count`` on a multi-GPU host; the worker-wide launched-process
        ceiling for the single-GPU / worker-wide (``None``) case.
        """
        if device_index is not None and device_index in self._card_runtimes:
            return self._card_runtimes[device_index].target_process_count
        return self._max_inference_processes

    def _update_head_starvation_timer(self, head_job: ImageGenerateJobPopResponse | None) -> None:
        """Track how long the current head-of-queue job has been stuck on an otherwise-idle device.

        The clock only runs while no live job is sampling or otherwise using an unclassified in-progress
        slot: a head waiting behind such work is legitimately queued, not starved. It resets whenever the
        head changes (a different job reached the front) so the backstop measures *this* head's wait, not
        the queue's age.
        """
        head_id = str(head_job.id_) if head_job is not None and head_job.id_ is not None else None
        in_progress_blocks_idle_fill = len(self._job_tracker.jobs_in_progress) > 0
        if head_id is None or in_progress_blocks_idle_fill:
            self._head_starvation_job_id = None
            self._head_starvation_since = 0.0
            return
        if head_id != self._head_starvation_job_id:
            self._head_starvation_job_id = head_id
            self._head_starvation_since = self._clock()

    def _head_starved_seconds(self, job: ImageGenerateJobPopResponse) -> float:
        """Seconds this job has been the idle-device head, or 0.0 when it is not the tracked head."""
        job_id = str(job.id_) if job.id_ is not None else None
        if job_id is None or job_id != self._head_starvation_job_id or self._head_starvation_since == 0.0:
            return 0.0
        return self._clock() - self._head_starvation_since

    def _undispatched_head(self) -> ImageGenerateJobPopResponse | None:
        """Return the first queued job no process is running yet (the head of queue), or None when there is none."""
        return next(
            (job for job in self._job_tracker.jobs_pending_inference if job not in self._job_tracker.jobs_in_progress),
            None,
        )

    def head_model_materializing(self) -> bool:
        """Return whether the head-of-queue inference job's model is actively loading onto an idle pool.

        True when nothing is in progress and the first not-yet-running queued job's model is in the model
        map's LOADING state (a preload is underway), or a missing-model recovery has just kicked a fresh
        preload for the head (the brief window between expiring the stale entry and the new LOADING state).
        A ready lane whose head model is still materialising is capacity in flight, not a wedge over a
        healthy pool: the recovery coordinator reads this to hold save-our-ship give-up off (bounded by the
        preload budget) so it does not fault a job the pool is in the middle of loading. Read-only.
        """
        pending = self._job_tracker.jobs_pending_inference
        if not pending:
            return False
        if self._process_map.has_inference_in_progress():
            return False
        head = self._undispatched_head()
        if head is None or head.model is None:
            return False
        return self._horde_model_map.is_model_loading(head.model) or self._missing_model_recovery_latched()

    def _clear_head_starvation_timer(self) -> None:
        """Reset the head-starvation clock once a job is dispatched (the wedge, if any, is broken)."""
        self._head_starvation_job_id = None
        self._head_starvation_since = 0.0
        # A dispatch means the card is fed, so the idle-fill breaker (if armed) is done; disarm it and
        # restart the ladder so the next idle episode begins at the smallest, quickest rung.
        if self._state.wants_idle_fill_candidate:
            self._state.wants_idle_fill_candidate = False
            self._state.idle_fill_rung = 0

    def _update_idle_fill_arm(self, bridge_data: reGenBridgeData) -> None:
        """Arm or disarm the idle-fill breaker from the head-starvation clock and free-sibling availability.

        Arms when a queue head has sat on an idle device past ``idle_fill_threshold_seconds`` (its model still
        loading, nothing in progress) while an inference sibling is free to run a fill job. The head-starvation
        clock is already forced to zero whenever any job is in progress, so this is inert in steady state and
        fires only for the "stuck doing nothing but downloading" case. Disarms (and resets the ladder) once
        the head is no longer starved or no sibling is free.
        """
        threshold = bridge_data.idle_fill_threshold_seconds
        starved_long_enough = (
            threshold is not None
            and self._head_starvation_since > 0.0
            and (self._clock() - self._head_starvation_since) >= threshold
        )
        has_free_sibling = self._process_map.get_first_available_inference_process() is not None
        if starved_long_enough and has_free_sibling:
            self._state.wants_idle_fill_candidate = True
        elif self._state.wants_idle_fill_candidate:
            self._state.wants_idle_fill_candidate = False
            self._state.idle_fill_rung = 0

    def _diagnose_dispatch_stall(
        self,
        head: ImageGenerateJobPopResponse,
        stable_diffusion_reference: dict[str, ImageGenerationModelRecord],
    ) -> str:
        """Return why the head-of-queue job is not being dispatched (read-only; never raises into the loop)."""
        return self._classify_dispatch_stall(head, stable_diffusion_reference)[1]

    def _classify_dispatch_stall(
        self,
        head: ImageGenerateJobPopResponse,
        stable_diffusion_reference: dict[str, ImageGenerationModelRecord],
    ) -> tuple[SlotDutyBucket, str]:
        """Name the gate parking the head-of-queue job, as a duty bucket plus the operator-facing text.

        The scheduler returns ``None`` from :meth:`get_next_job_and_process` at several points without saying
        why, so a stuck queue with idle processes leaves no record of which gate parked the head. This
        re-derives that reason, with the most detail for the genuinely suspicious case -- the head's model is
        resident on an *idle* process yet nothing dispatches, since that is the scheduler-bug-shaped stall
        that is otherwise invisible. The bucket half feeds the slot-duty accounting every tick
        (:meth:`record_slot_duty`), so the same derivation prices the empty slot's wall clock; the text half
        feeds the throttled parked-head log line. Read-only; never raises into the loop.

        The text names the block and nothing that merely ticks with the clock. Callers compare it across
        cycles: the log line throttles on it being unchanged, and recovery judges whether a rung moved the
        head's blocker by whether it changed. A quantity that advances every cycle (elapsed seconds, a
        running total) would make every comparison read as a different block, so the line would repeat every
        cycle and every rung would look effective. Such figures belong to the formatted line instead
        (:meth:`_log_dispatch_stall_if_needed`), which prints them alongside the parked-seconds count.
        """
        process = self._resident_process_for_job(head)
        # Mirror dispatch's retention affinity: where a busy slot holds the head's model on the device and a
        # second copy does not fit beside it, that slot is the head's destination, so the wait is named as the
        # resident slot being busy rather than as a preload defer.
        retention_retainer = self._retention_affinity_retainer(head, process)
        if retention_retainer is not None:
            process = retention_retainer
        if process is None:
            if head.model is not None and self._horde_model_map.is_model_loading(head.model):
                return (
                    SlotDutyBucket.MODEL_LOADING,
                    "its model is loading (a preload is in progress)",
                )
            # The head's model can be resident only on a disaggregation-pinned sampler lane, which the dispatch
            # query excludes. That is not a budget defer: the head is deliberately held for the pin to release
            # (rather than funding a second copy), so name the pin, the job holding it, and the in-flight
            # sampling that keeps the card busy, instead of reporting a generic not-resident preload defer.
            pinned_lane = self._pinned_lane_resident_for_job(head)
            if pinned_lane is not None:
                owner = self._disaggregation_pin_owner(pinned_lane.process_id)
                owner_text = f" holding disaggregated job {owner[:8]}" if owner else ""
                peaks = self._disaggregation_sampling_peaks()
                # The count distinguishes a card busy with sampling from an idle one; the megabytes behind it
                # drift every cycle, so they are left to the arbiter's own diagnostics.
                peaks_text = f"; {len(peaks)} sampling(s) in flight" if peaks else "; no sampling currently in flight"
                return SlotDutyBucket.DISAGG_PIN_WAIT, (
                    f"its model is resident only on process {pinned_lane.process_id}, pinned as a disaggregation "
                    f"sampler{owner_text}; the head waits for that pin to release and dispatch onto the resident "
                    f"lane rather than fund a second copy that cannot fit beside the pinned residents{peaks_text}"
                )
            # A whole-card residency held for a *different* model reserves the card and tore its siblings down,
            # so a head of another model cannot load until that residency restores. Name it: otherwise this
            # reads as a generic VRAM-budget defer (the card looks idle with ample free VRAM) when the real
            # cause is a residency granted to a non-head model.
            nonhead_residency_model = next(
                (
                    state.model
                    for _, state in self._held_residencies()
                    if state.model is not None and state.model != head.model
                ),
                None,
            )
            if nonhead_residency_model is not None:
                return SlotDutyBucket.WHOLE_CARD_RESERVED, (
                    f"its model is not resident because a whole-card residency is held for non-head model "
                    f"{nonhead_residency_model!r}: the card is reserved for that model and its siblings were "
                    f"torn down, so this head cannot load until that residency restores"
                )
            # Quote the admission gate's own verdict rather than pointing at budget lines that may not exist:
            # the defer notice is coalesced on an unchanged reason, so a head that has been declined for the
            # same arithmetic all along has no live line to read. The record is only quoted when it names this
            # head's model, since it holds the most recent decision for any job.
            admission = self._last_preload_admission
            if admission is not None and admission.model == head.model and admission.reason:
                return SlotDutyBucket.PRELOAD_DEFERRED, (
                    f"its model is not resident and its preload was declined ({admission.decision.value}): "
                    f"{admission.reason}"
                )
            return SlotDutyBucket.PRELOAD_DEFERRED, (
                "its model is not resident and no preload has been attempted for it this cycle"
            )
        if not process.can_accept_job():
            retained_detail = (
                " and holds the model's weights on the device, which a second copy cannot fit beside"
                if process.retained_resident_model == head.model
                else ""
            )
            return SlotDutyBucket.RESIDENT_SLOT_BUSY, (
                f"its model is resident on process {process.process_id}, but that process is busy "
                f"({process.last_process_state.name}){retained_detail}"
            )

        # Resident on an idle process: the interesting case. Name the gate that is holding dispatch.
        in_progress = len(self._job_tracker.jobs_in_progress)
        cap = self._max_jobs_in_progress_allowed()
        if in_progress >= cap:
            # The exclusive-admit hold collapses the cap to the running job; name it distinctly so the
            # serialization is attributed to the admit, not to a generic cap the operator would chase
            # through max_threads.
            if self._job_tracker.has_exclusive_job_in_progress(process.device_index) and not (
                self._job_tracker.is_admitted_exclusive(head)
            ):
                return SlotDutyBucket.EXCLUSIVE_ISOLATION, (
                    f"its model is resident and idle on process {process.process_id}, but an exclusively-"
                    f"admitted over-budget job has the device to itself (in_progress={in_progress})"
                )
            return SlotDutyBucket.CONCURRENCY_CAP, (
                f"its model is resident and idle on process {process.process_id}, but the concurrency cap is "
                f"reached (in_progress={in_progress}, cap={cap})"
            )
        # Below the cap the same admit still holds dispatch whenever it holds a live claim on the card (a
        # running over-budget job, or a staged one whose whole-card residency is up). Attribute that slot to the
        # admit here too, so the wait is named rather than falling through to the gate-less stall report.
        if not self._job_tracker.is_admitted_exclusive(head) and self._exclusive_dispatch_suppression_active(
            process.device_index,
        ):
            return SlotDutyBucket.EXCLUSIVE_ISOLATION, (
                f"its model is resident and idle on process {process.process_id}, but an exclusively-admitted "
                f"over-budget job holds the card (in_progress={in_progress}, cap={cap})"
            )
        if not self._concurrent_overlap_allowed(head, target_device_index=process.device_index):
            return SlotDutyBucket.OVERLAP_HEADWAY, (
                f"its model is resident and idle on process {process.process_id}, but the overlap-headway gate "
                "is holding it (the in-flight job has not made enough progress to share the card)"
            )

        # A held whole-card residency parks its own pre-staged head until the live inference-process count
        # collapses to the forecast's target (sole residency). The convergence teardown is meant to stop the
        # idle siblings, including ones holding a model queued behind the head, sparing only the head's
        # holder. If the head is still parked with such a sibling un-torn-down, the convergence shrink has not
        # collapsed the pool, and the head will be deferred until the recovery supervisor soft-resets. Name
        # that specific state rather than reporting a gate-less "scheduler stall", so the post-mortem points at
        # the residency teardown rather than the dispatch path.
        found_residency, residency_device = self._residency_holder_for_model(head.model)
        if found_residency and self._prestaged_whole_card_not_ready(head):
            blockers = self._whole_card_convergence_blockers(process, residency_device)
            if blockers:
                pinned = ", ".join(f"process {pid} holds queued model {model!r}" for pid, model in blockers)
                return SlotDutyBucket.WHOLE_CARD_CONVERGENCE, (
                    f"its model is resident and idle on process {process.process_id}, but the whole-card "
                    f"residency stuck: cannot reach sole residency because {pinned}; the convergence teardown "
                    f"should have stopped that idle sibling (only the head's holder is spared), so the shrink "
                    f"has not collapsed the pool and the head never dispatches"
                )
            return SlotDutyBucket.WHOLE_CARD_CONVERGENCE, (
                f"its model is resident and idle on process {process.process_id}, but its whole-card residency "
                f"has not yet converged to sole residency (siblings still tearing down or the device draining)"
            )

        # A head whose next dispatch must run degraded (isolated) waits for the card to clear of other work
        # rather than share it. Named here so the isolation wait does not read as an unexplained scheduler
        # stall once the concurrency gates above have not claimed it.
        if self._job_tracker.is_degraded_dispatch_pending(head):
            return SlotDutyBucket.DEGRADED_ISOLATION_PENDING, (
                f"its model is resident and idle on process {process.process_id}, but its next dispatch must run "
                f"degraded/isolated and is waiting for the card to clear of other work"
            )

        # The post-processing co-residency defer gate holds an already-resident head while an in-flight (or
        # imminent) post-processing chain's committed VRAM would collide with this job's sampling peak on the
        # card. The dispatch path records that verdict in ``_post_processing_defer_holds`` on the pass it
        # computes it, so this reads the gate's own truthful state rather than re-deriving the PP fit (the two
        # must never disagree). The head dispatches once the chain finishes, so this is a real named head-park,
        # not the gate-less scheduler stall the fall-through reports.
        if head.id_ is not None and str(head.id_) in self._post_processing_defer_holds:
            return SlotDutyBucket.POST_PROCESSING_DEFER, (
                f"its model is resident and idle on process {process.process_id}, but dispatch is held while an "
                "in-flight post-processing chain finishes: the chain's committed VRAM and this job's sampling "
                "peak cannot share the card, and it dispatches once the chain releases the device"
            )

        # The dispatch-time residency-reconciliation gate holds an already-resident head while it evicts idle
        # sibling VRAM so the head's on-device materialisation fits before it commits. The gate stamps the held
        # job in ``_dispatch_hold_since`` and self-clears within a few ticks once the eviction frees room, so a
        # head parked here is a benign swap-churn wait, not the gate-less scheduler stall the fall-through
        # reports. Read the hold ledger directly rather than re-deriving the fit so the two never disagree.
        if head.id_ is not None and str(head.id_) in self._dispatch_hold_since:
            return SlotDutyBucket.RESIDENCY_RECONCILIATION, (
                f"its model is resident and idle on process {process.process_id}, but dispatch is held to "
                "reconcile residency (evicting idle VRAM): its materialisation would over-commit the card until "
                "an idle resident is evicted, and it dispatches once that eviction frees room"
            )

        # The same reconciliation in the retention direction: weights a sibling slot holds across jobs occupy
        # the card this head would load into, so the head waits for them to be returned. Read through the
        # gate's own read-only predicate (naming a wait must never evict anything), and bounded by the
        # confirmation window, so this too is a self-clearing wait rather than the gate-less stall below.
        if self._retained_resident_hold_applies(head, process):
            return SlotDutyBucket.RESIDENCY_RECONCILIATION, (
                f"its model is resident and idle on process {process.process_id}, but dispatch is held to "
                "reconcile residency (returning retained weights): a sibling slot holds model weights across "
                "jobs that this job's materialisation cannot fit beside, and it dispatches once the card gives "
                "them back"
            )

        return SlotDutyBucket.UNEXPLAINED, (
            f"its model is resident and idle on process {process.process_id} but dispatch was withheld with no "
            "matching gate; this is a scheduler stall worth reporting"
        )

    def _whole_card_convergence_blockers(
        self,
        head_process: HordeProcessInfo,
        device_index: int | None,
    ) -> list[tuple[int, str]]:
        """Return idle sibling processes still holding a queued model while a whole-card head is parked.

        Returns ``(process_id, model)`` for each inference process other than the head's own holder that is
        idle (not busy), pinned to ``device_index`` when scoped, and holds a model that is still queued. The
        whole-card convergence is meant to have torn these siblings down (sparing only the head's holder), so
        finding any while the head is still parked is the fingerprint of a teardown that did not collapse the
        pool. Read-only; used only to explain a stalled dispatch.
        """
        queued_models = {
            job.model
            for job in (*self._job_tracker.jobs_pending_inference, *self._job_tracker.jobs_in_progress)
            if job.model is not None
        }
        blockers: list[tuple[int, str]] = []
        for proc in self._process_map.values():
            if proc.process_type is not HordeProcessType.INFERENCE:
                continue
            if proc.process_id == head_process.process_id:
                continue
            if device_index is not None and proc.device_index != device_index:
                continue
            if proc.is_process_busy():
                continue
            model = proc.loaded_horde_model_name
            if model is not None and model in queued_models:
                blockers.append((proc.process_id, model))
        return blockers

    _EXCLUSIVE_SUPPRESSION_LOG_INTERVAL_SECONDS = 30.0
    """How often the exclusive-admit dispatch hold may name itself for one card scope.

    The hold is re-evaluated on every dispatch selection, so an unthrottled line would repeat many times a
    second for as long as the exclusive job owns the card while saying nothing new."""

    def _exclusive_dispatch_suppression_active(self, device_index: int | None) -> bool:
        """Whether an exclusive over-budget admit currently withholds co-dispatch in ``device_index``'s scope.

        Suppression follows a live claim on the card, not the per-job ``admitted_exclusive`` flag: that flag is
        sticky for the life of the job because it also carries fault attribution and the over-budget step
        grace. Two claims qualify. An exclusive job actually sampling has its over-budget footprint on the card
        right now, so co-dispatch stays suppressed for its whole run (on a single-GPU host that is worker-wide,
        which is the intended conservatism). A staged exclusive job qualifies only while its whole-card
        residency is held or being established, since that is when the card has actually been given to it. A
        marked job with neither claim (its establishment deferred by the residency rate limiter, for instance)
        holds nothing and must not stop unrelated work from dispatching.
        """
        if self._job_tracker.has_exclusive_job_running(device_index):
            return True
        if not self._job_tracker.has_exclusive_job_in_progress(device_index):
            return False
        return self._exclusive_residency_live(device_index)

    def _exclusive_residency_live(self, device_index: int | None) -> bool:
        """Whether a whole-card residency is held or being established in ``device_index``'s scope.

        A residency record with a model set covers both phases: the model is recorded at the grant, which is
        when the teardown begins, and cleared only on restore. A card is asked about its own residency and
        about the card-agnostic worker-wide record, mirroring how an unattributed exclusive admit is treated as
        applying to every card.
        """
        if device_index is None:
            return self._whole_card_ledger.any_held()
        return any(
            (state := self._whole_card_ledger.get(key)) is not None and state.model is not None
            for key in (device_index, None)
        )

    def _note_exclusive_dispatch_suppression(
        self,
        job: ImageGenerateJobPopResponse,
        device_index: int | None,
    ) -> None:
        """Disclose (throttled per card scope) that an exclusive admit is holding this job's dispatch back."""
        now = self._clock()
        last = self._exclusive_suppression_logged_at.get(device_index, 0.0)
        if (now - last) < self._EXCLUSIVE_SUPPRESSION_LOG_INTERVAL_SECONDS:
            return
        self._exclusive_suppression_logged_at[device_index] = now
        scope = "worker-wide" if device_index is None else f"device {device_index}"
        logger.debug(
            f"Dispatch of job {str(job.id_)[:8]} ({job.model}) is held {scope}: an exclusively-admitted "
            "over-budget job holds the card.",
        )

    def _affinity_bypass_note(self, head: ImageGenerateJobPopResponse) -> str:
        """Return how often, and over how long, this head has been passed by resident-model line-skips.

        Empty unless the skip window is tracking this head and has actually skipped it. Purely observed
        quantities, so it belongs to a formatted line rather than to the compared stall reason: the counters
        advance while the block that caused them stays the same. It reports what happened and not an
        allowance, since nothing enforces the window once a head is being fed past.
        """
        if (
            head.id_ is None
            or str(head.id_) != self._affinity_skip_state.head_job_id
            or self._affinity_skip_state.skip_count <= 0
        ):
            return ""
        return (
            f"; bypassed by {self._affinity_skip_state.skip_count} affinity line-skips over "
            f"{self.latest_affinity_skip_seconds():.0f}s"
        )

    @property
    def head_dispatch_block_reason(self) -> str | None:
        """The constraint last recorded as holding the head of queue back, or None while dispatch flows.

        The stable stall attribution (no ticking figures), cleared when a job dispatches. Read by the
        recovery coordinator to judge remedy relevance and by the work ledger for operator disclosure.
        """
        return self._dispatch_stall_last_reason

    def _log_dispatch_stall_if_needed(
        self,
        stable_diffusion_reference: dict[str, ImageGenerationModelRecord],
    ) -> None:
        """Emit a throttled explanation when a parked head is not dispatching despite pending work.

        Only fires once the head has been undispatched past :data:`_DISPATCH_STALL_MIN_SECONDS` (so a normal
        between-jobs gap is silent), then at most once per :data:`_DISPATCH_STALL_LOG_INTERVAL_SECONDS` for an
        unchanged reason. The line carries the quantities that advance every cycle (how long the head has been
        parked, how often it has been passed) so the compared reason can stay stable across a sustained block.
        Read-only: it explains the stall, it does not change scheduling.
        """
        head = self._undispatched_head()
        if head is None or self._head_starved_seconds(head) < _DISPATCH_STALL_MIN_SECONDS:
            return
        try:
            reason = self._diagnose_dispatch_stall(head, stable_diffusion_reference)
        except Exception as e:  # noqa: BLE001 - a diagnostic must never crash the scheduling cycle
            reason = f"undiagnosed ({type(e).__name__}: {e})"

        now = time.monotonic()
        if (
            reason == self._dispatch_stall_last_reason
            and (now - self._dispatch_stall_log_time) < _DISPATCH_STALL_LOG_INTERVAL_SECONDS
        ):
            return
        self._dispatch_stall_last_reason = reason
        self._dispatch_stall_log_time = now
        logger.opt(colors=True).warning(
            "<fg #ff8c69>Inference dispatch stalled: head {} ({}) has been parked "
            f"{self._head_starved_seconds(head):.0f}s: {reason}{self._affinity_bypass_note(head)}.</>",
            str(head.id_)[:8],
            head.model,
        )

    def record_slot_duty(self, stable_diffusion_reference: dict[str, ImageGenerationModelRecord]) -> None:
        """Attribute the wall clock since the last scheduling cycle across the configured inference slots.

        Called once per scheduling cycle. Busy slots accrue ``SAMPLING``; when capacity is spare and a
        queued job is waiting, the empty slots accrue the bucket the stall classifier names (the same
        derivation that explains a parked head, but priced every tick instead of only after a multi-second
        park); with no waiting work they accrue ``NO_LOCAL_WORK``. The classification is a read-only
        diagnostic: any failure inside it degrades to ``UNEXPLAINED`` rather than touching scheduling.

        Under the clearance lease a slot is productive only while its child is *actively sampling*: a staged
        child holds an in-progress job but its sampling slot is empty until the parent clears it. So busy is
        counted from processes in their denoise loop, and a staged-but-uncleared child (the clearance gate
        holding it, or waiting its turn for a slot) attributes the empty sampling slot to ``CLEARANCE_HOLD``.
        Without the lease dispatch is the sampling moment, so busy is the in-progress count exactly as before.
        """
        capacity = max(int(self._max_concurrent_inference_processes or 0), 0)
        in_progress = self._job_tracker.jobs_in_progress

        if self._clearance_lease_active():
            sampling_count = 0
            primed_count = 0
            for process_info in self._process_map.values():
                if process_info.process_type != HordeProcessType.INFERENCE:
                    continue
                if process_info.last_process_state == HordeProcessState.INFERENCE_STARTING:
                    sampling_count += 1
                elif process_info.last_process_state == HordeProcessState.INFERENCE_PRIMED:
                    primed_count += 1
            # Prune clearance-hold records for jobs that have left the in-progress set (self-healing).
            in_progress_ids = {str(job.id_) for job in in_progress if job.id_ is not None}
            self._clearance_hold_ids &= in_progress_ids

            busy = min(sampling_count, capacity)
            hold: SlotDutyBucket | None = None
            if busy < capacity and primed_count > 0:
                # A staged child not yet sampling is waiting on the clearance gate (held for VRAM fit, or its
                # turn for a slot): the empty sampling slot is the clearance gate's, not the gate-less stall.
                hold = SlotDutyBucket.CLEARANCE_HOLD
                waiting = primed_count
            else:
                head = self._undispatched_head()
                waiting = len(self._job_tracker.jobs_pending_inference) - len(in_progress)
                if head is not None and busy < capacity:
                    try:
                        hold = self._classify_dispatch_stall(head, stable_diffusion_reference)[0]
                    except Exception:  # noqa: BLE001 - a diagnostic must never crash the scheduling cycle
                        hold = SlotDutyBucket.UNEXPLAINED
            self._slot_duty_current_hold = hold
            self._slot_duty.observe(
                time.time(),
                capacity=capacity,
                busy_slots=busy,
                waiting_jobs=max(waiting, 0),
                hold=hold,
            )
            return

        busy = len(in_progress)
        head = self._undispatched_head()
        waiting = len(self._job_tracker.jobs_pending_inference) - busy

        hold = None
        if head is not None and busy < capacity:
            try:
                hold = self._classify_dispatch_stall(head, stable_diffusion_reference)[0]
            except Exception:  # noqa: BLE001 - a diagnostic must never crash the scheduling cycle
                hold = SlotDutyBucket.UNEXPLAINED
        self._slot_duty_current_hold = hold

        self._slot_duty.observe(
            time.time(),
            capacity=capacity,
            busy_slots=busy,
            waiting_jobs=max(waiting, 0),
            hold=hold,
        )

    def slot_duty_snapshot(self) -> tuple[dict[str, float], int, str | None]:
        """The cumulative slot-second totals, the current capacity, and the currently-named hold bucket.

        Consumers difference successive totals for a window's breakdown (the stats stream carries the
        cumulative figures; the periodic duty log line differences its own anchor).
        """
        hold = self._slot_duty_current_hold
        return self._slot_duty.totals(), self._slot_duty.capacity, str(hold) if hold is not None else None

    def _measured_free_vram_mb(self, *, device_index: int | None = None) -> float | None:
        """Return the most conservative measured free VRAM (MB), or None when not yet reported.

        Sourced from GPU-bearing child VRAM reports via :meth:`ProcessMap.get_free_vram_mb`, which the
        children compute through hordelib's backend-agnostic accelerator layer (comfy /
        ``torch.cuda.mem_get_info``, accurate and not NVIDIA-specific). The parent stays free of any direct
        GPU query, so this works on every backend the execution layer supports.

        A child's view of the card is process-local, and under WDDM it runs ahead of the device: memory the
        driver has not yet returned still reads as free, so admission priced on the child figure alone buys
        headroom the card does not have. The parent's own device-level reading (see
        :meth:`set_device_free_mb_provider`) is therefore taken as a ceiling on the answer whenever one exists
        for the card, leaving the more conservative of the two. Where the reading source is itself the child
        reports (a harness that injects the provider) the ceiling is a no-op, so nothing that prices against an
        injected device inventory changes.

        Args:
            device_index: When given, the free VRAM of that one card (the per-card budget on a multi-GPU
                host); when None, the most conservative figure across every card (the single-GPU reading).
        """
        reported_mb = self._process_map.get_free_vram_mb(device_index=device_index)
        if reported_mb is None:
            return None
        device_truth_mb = self._measured_device_free_mb(device_index)
        if device_truth_mb is None:
            return reported_mb
        return min(reported_mb, device_truth_mb)

    def _measured_available_ram_mb(self) -> float:
        """The measured system-wide available RAM (MB): the injected provider, else a live parent read."""
        if self._available_ram_mb_provider is not None:
            return self._available_ram_mb_provider()
        return psutil.virtual_memory().available / (1024 * 1024)

    def _measured_total_ram_mb(self) -> float:
        """The measured system-wide total RAM (MB), read live in the parent process."""
        return psutil.virtual_memory().total / (1024 * 1024)

    def _ram_pressure_floor_config(self) -> tuple[float, float]:
        """The configured (pause_percent, min_free_mb) for the absolute RAM danger floor, read defensively.

        Tolerant of a partially-mocked config (the scheduler unit tests): a non-numeric value falls back to
        the module default so the pressure check never crashes the scheduling cycle on a bad attribute.
        """
        bridge_data = self._runtime_config.bridge_data
        pause = config_number(bridge_data.ram_pressure_pause_percent)
        min_free = config_number(bridge_data.ram_pressure_min_free_mb)
        # Fallbacks match reGenBridgeData's defaults for these fields (85% used / 1 GB) so a partially-mocked
        # config sees the same danger floor a real worker does.
        pause_pct = pause if pause is not None else 85.0
        min_free_mb = min_free if min_free is not None else 1024.0
        return pause_pct, min_free_mb

    def _ram_pressure_verdict(self) -> RamPressureVerdict:
        """Assess whether the host is below its absolute system-RAM danger floor right now."""
        pause_pct, min_free_mb = self._ram_pressure_floor_config()
        return assess_ram_pressure(
            self._measured_available_ram_mb(),
            self._measured_total_ram_mb(),
            pause_percent=pause_pct,
            min_free_mb=min_free_mb,
        )

    @property
    def _ram_governor_state(self) -> RamGovernorState:
        """The RAM governor's multi-tick bookkeeping (owned by the resource governor)."""
        return self._governor.ram_state

    @property
    def _ram_pressure_shed_cards(self) -> set[int]:
        """Device indices the RAM-pressure reduction shed below plan (see ``RamGovernorState.shed_cards``)."""
        return self._ram_governor_state.shed_cards

    @_ram_pressure_shed_cards.setter
    def _ram_pressure_shed_cards(self, value: set[int]) -> None:
        self._ram_governor_state.shed_cards = set(value)

    @property
    def _processes_draining_for_ram(self) -> set[int]:
        """Inference process ids draining for RAM reclaim (see ``RamGovernorState.draining_process_ids``)."""
        return self._ram_governor_state.draining_process_ids

    @_processes_draining_for_ram.setter
    def _processes_draining_for_ram(self, value: set[int]) -> None:
        self._ram_governor_state.draining_process_ids = set(value)

    def _build_host_memory_snapshot(self, verdict: RamPressureVerdict) -> HostMemorySnapshot:
        """Capture the host-RAM state and governor bookkeeping one governance decision runs over.

        The single measurement site for RAM governance: every reading the pure decision functions in
        [`ram_governor`][horde_worker_regen.process_management.scheduling.governance.ram_governor] consume
        is taken here, once, so a decision never re-measures mid-flight. Config values are read
        defensively (a partially-mocked config falls back to the field default) so snapshotting never
        crashes the scheduling cycle.
        """
        margin_mb = config_number(self._runtime_config.bridge_data.ram_reserve_mb)
        if margin_mb is None:
            margin_mb = 4096.0
        inference_slots = tuple(
            InferenceSlotSnapshot(
                process_id=process_info.process_id,
                device_index=process_info.device_index,
                resident_ram_mb=process_info.ram_usage_bytes / (1024 * 1024),
                is_busy=process_info.is_process_busy(),
            )
            for process_info in self._process_map.values()
            if process_info.process_type == HordeProcessType.INFERENCE
        )
        residency_held_cards = {index for index, _residency in self._held_residencies()}
        cards = tuple(
            CardProcessSnapshot(
                device_index=device_index,
                loaded_process_count=self._process_map.num_loaded_inference_processes(device_index=device_index),
                busy_process_count=self._card_inference_load(device_index),
                planned_process_count=card_runtime.target_process_count,
                held_by_whole_card_residency=device_index in residency_held_cards,
            )
            for device_index, card_runtime in sorted(self._card_runtimes.items())
        )
        return HostMemorySnapshot(
            verdict=verdict,
            now=time.time(),
            pop_pause_active=self._state.self_throttle_paused,
            pop_pause_until=self._state.self_throttle_paused_until,
            pop_hold_margin_mb=margin_mb,
            per_process_ceiling_mb=self._ram_per_process_ceiling_mb(),
            multi_gpu_routing_active=self._multi_gpu_routing_active,
            in_flight_job_count=len(self._job_tracker.jobs_in_progress),
            loaded_worker_process_count=self._process_map.num_loaded_inference_processes(),
            planned_worker_process_count=self._max_inference_processes,
            inference_slots=inference_slots,
            cards=cards,
            draining_process_ids=frozenset(self._ram_governor_state.draining_process_ids),
            shed_card_indices=frozenset(self._ram_governor_state.shed_cards),
            restore_headroom_mb=self._ram_headroom_for_additional_context_mb(),
            per_context_ram_estimate_mb=self._estimated_resident_context_ram_mb(),
            worker_shed_planned_process_count=(
                self._ram_governor_state.worker_shed.planned_process_count
                if self._ram_governor_state.worker_shed is not None
                else None
            ),
            worker_shed_process_count=(
                self._ram_governor_state.worker_shed.shed_process_count
                if self._ram_governor_state.worker_shed is not None
                else 0
            ),
        )

    def _execute_governance_actions(self, actions: list[GovernanceAction]) -> None:
        """Execute governance decisions against the live worker: the single act site for RAM remedies.

        The governor's multi-tick bookkeeping (draining marks, shed-card tracking) is mutated here, at
        execution time and with the measured result of each remedy (a card is only recorded as shed when
        its count actually fell), so the decision layer stays a pure function of its snapshot.
        """
        governor_state = self._ram_governor_state
        for action in actions:
            match action:
                case SetPopHold(active=hold_active):
                    self._state.ram_pressure_pop_hold = hold_active
                case PausePops(
                    until_time=until_time,
                    pause_seconds=pause_seconds,
                    reason=reason,
                    available_mb=available_mb,
                    floor_mb=floor_mb,
                ):
                    prior_owner = self._state.self_throttle_pause_owner
                    pause_reason = f"host RAM pressure: {reason}"
                    self._state.self_throttle_paused = True
                    self._state.self_throttle_paused_until = until_time
                    self._state.self_throttle_pause_owner = PopPauseOwner.RAM_PRESSURE
                    self._state.self_throttle_pause_reason = pause_reason
                    self._process_lifecycle.action_ledger.record(
                        LedgerEventType.POP_PAUSE_ARMED,
                        reason=pause_reason,
                        detail={
                            "owner": PopPauseOwner.RAM_PRESSURE.value,
                            "duration_seconds": round(pause_seconds, 1),
                            "available_ram_mb": round(available_mb, 1) if available_mb is not None else None,
                            "floor_ram_mb": round(floor_mb, 1) if floor_mb is not None else None,
                        },
                    )
                    # A still-standing pause from a different backstop is only superseded here when this
                    # RAM deadline is the later one (the decision layer emits PausePops only then), so name
                    # the transition rather than silently relabelling the shared deadline.
                    takeover = (
                        f" (superseding a standing {prior_owner.value} pause)"
                        if prior_owner is not None and prior_owner is not PopPauseOwner.RAM_PRESSURE
                        else ""
                    )
                    logger.opt(ansi=True).warning(
                        f"<fg #ff8c69>System RAM below the danger floor ({reason}); pausing job pops for "
                        f"{pause_seconds:.0f}s{takeover} and shedding idle footprint so the host is not driven "
                        "into an OS OOM kill. In-flight jobs finish; pops resume once RAM recovers.</>",
                    )
                case EvictIdleModels():
                    # First reclaim the cheap, targeted way: drop idle, unprotected staged components from the
                    # budgeted RAM cache, keeping a queued job's staged model resident. Only when nothing
                    # unprotected can be evicted (or the budgeted cache is off) does the coarser whole-RAM
                    # unload run: unload an idle resident model, and when none remains, cycle the slot whose
                    # allocator kept the freed pages (only a process cycle returns them to the OS). Mirrors the
                    # preload reclaim path so sustained pressure with a drained queue still reclaims RAM.
                    if not self._evict_unprotected_components_under_pressure() and not self.unload_models(
                        under_pressure=True,
                    ):
                        self._replace_stale_ram_unload_process()
                case ReduceWorkerProcesses(
                    target_count=target_count,
                    planned_count=planned_count,
                    pressure_shortfall_mb=pressure_shortfall_mb,
                ):
                    current = self._process_map.num_loaded_inference_processes()
                    planned = planned_count if planned_count > 0 else self._max_inference_processes
                    after = self._process_lifecycle.scale_inference_processes(
                        target_count,
                        device_index=None,
                        pressure_shortfall_mb=pressure_shortfall_mb,
                    )
                    if not isinstance(after, int):
                        after = current
                    if after < current:
                        # The record is the live shortfall below plan, not an accumulation of reductions: a
                        # whole-card residency restore can regrow the pool between reductions, and a running
                        # total would over-count every cycle without bound while the pool is back at plan.
                        governor_state.worker_shed = WorkerProcessShedState(
                            planned_process_count=planned,
                            shed_process_count=max(0, planned - after),
                        )
                        shortfall_note = (
                            f", shortfall ~{pressure_shortfall_mb:.0f} MB" if pressure_shortfall_mb is not None else ""
                        )
                        logger.opt(ansi=True).info(
                            f"<fg #ff8c69>RAM pressure reduced worker inference contexts "
                            f"({current} -> {after} of {planned}{shortfall_note}); the pool will be "
                            "restored incrementally once RAM has headroom.</>",
                        )
                case ReduceCardProcesses(device_index=device_index, target_count=target_count):
                    current = self._process_map.num_loaded_inference_processes(device_index=device_index)
                    after = self._process_lifecycle.scale_inference_processes(
                        target_count,
                        device_index=device_index,
                    )
                    if not isinstance(after, int):
                        after = current
                    if after < current:
                        governor_state.shed_cards.add(device_index)
                case MarkProcessDraining(
                    process_id=process_id,
                    resident_ram_mb=resident_ram_mb,
                    ceiling_mb=ceiling_mb,
                ):
                    governor_state.draining_process_ids.add(process_id)
                    logger.opt(ansi=True).warning(
                        f"<fg #ff8c69>Inference process {process_id} holds {resident_ram_mb:.0f} MB RAM (>= the "
                        f"{ceiling_mb:.0f} MB per-process ceiling) while the host is under its RAM floor; "
                        "draining it (no new work) so it can be recycled once its in-flight job finishes.</>",
                    )
                case ClearProcessDraining(process_id=process_id):
                    governor_state.draining_process_ids.discard(process_id)
                case RecycleProcess(process_id=process_id, resident_ram_mb=resident_ram_mb, ceiling_mb=ceiling_mb):
                    process_info = self._process_map.get(process_id)
                    if process_info is None:
                        # The process exited between snapshot and execution; nothing to reclaim.
                        governor_state.draining_process_ids.discard(process_id)
                        continue
                    logger.opt(ansi=True).warning(
                        f"<fg #ff8c69>Inference process {process_id} holds {resident_ram_mb:.0f} MB RAM (>= the "
                        f"{ceiling_mb:.0f} MB per-process ceiling); "
                        "recycling it to return the retained RAM to the OS.</>",
                    )
                    governor_state.draining_process_ids.discard(process_id)
                    self._process_lifecycle._replace_inference_process(process_info, intentional_reclaim=True)
                    self._ram_reclaim_cycle_at = self._clock()
                    self._record_churn("process_cycle")
                case RestoreCardProcess(device_index=device_index, target_count=target_count, planned_count=planned):
                    current = self._process_map.num_loaded_inference_processes(device_index=device_index)
                    after = self._process_lifecycle.scale_inference_processes(
                        target_count,
                        device_index=device_index,
                    )
                    if not isinstance(after, int):
                        after = current
                    logger.opt(ansi=True).info(
                        f"<fg #7b7d7d>System RAM has headroom; restoring an inference context on device "
                        f"{device_index} ({current} -> {after} of {planned}) so the card resumes serving.</>",
                    )
                    if after >= planned:
                        governor_state.shed_cards.discard(device_index)
                case RestoreWorkerProcess(target_count=target_count, planned_count=planned):
                    current = self._process_map.num_loaded_inference_processes()
                    after = self._process_lifecycle.scale_inference_processes(target_count, device_index=None)
                    if not isinstance(after, int):
                        after = current
                    logger.opt(ansi=True).info(
                        f"<fg #7b7d7d>System RAM has headroom; restoring a worker inference context "
                        f"({current} -> {after} of {planned}).</>",
                    )
                    if after >= planned:
                        governor_state.worker_shed = None
                    elif governor_state.worker_shed is not None and after > current:
                        governor_state.worker_shed.shed_process_count = max(
                            0,
                            governor_state.worker_shed.shed_process_count - (after - current),
                        )
                case StopTrackingShedCard(device_index=device_index):
                    governor_state.shed_cards.discard(device_index)
                case StopTrackingWorkerShed():
                    governor_state.worker_shed = None

    def _govern_ram_pressure(self, verdict: RamPressureVerdict) -> None:
        """Degrade the worker's footprint and intake while system RAM is below the danger floor.

        The proactive counterpart to the marginal RAM budget: rather than admit a load that the absolute
        reading says will trip the kernel OOM-killer, the worker pauses job pops, evicts idle resident
        models, reduces the resident inference-process count, and reclaims a process whose resident RAM
        crossed the per-process ceiling. The decision logic lives in
        [`decide_degrade_response`][horde_worker_regen.process_management.scheduling.governance.ram_governor.decide_degrade_response].
        """
        self._reclaim_idle_alchemy_lanes_under_pressure()
        snapshot = self._build_host_memory_snapshot(verdict)
        self._execute_governance_actions(decide_degrade_response(snapshot))

    def _reclaim_idle_alchemy_lanes_under_pressure(self) -> None:
        """Unload idle safety/post-process alchemy residents during a host-RAM pressure episode."""
        for process_info in self._process_map.values():
            if process_info.process_type not in (HordeProcessType.SAFETY, HordeProcessType.POST_PROCESS):
                continue
            if not process_info.is_process_alive() or not process_info.can_accept_job():
                continue
            if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
                continue
            logger.opt(colors=True).info(
                f"<fg #ff8c69>Host RAM pressure: unloading idle {process_info.process_type.name} lane "
                f"{process_info.process_id} from RAM.</>",
            )
            self.unload_from_ram(process_info.process_id)

    def _govern_ram_pressure_if_pressured(self) -> bool:
        """Evaluate the absolute RAM danger floor and degrade the worker if it is breached.

        The per-tick entry point (distinct from the per-job :meth:`_preload_blocked_by_ram_pressure`),
        delegated to the resource governor: it updates the soft pop hold, runs the whole-host degrade
        response whenever the host is under its floor, and restores past shedding once the host recovers,
        so a worker that never attempts a new preload still throttles and reclaims instead of growing
        into an OS OOM kill. Returns whether the host was under pressure. Clears the one-shot notice when
        the host is healthy.
        """
        under_pressure = self._governor.tick()
        if not under_pressure:
            self._ram_pressure_notified = False
        return under_pressure

    def _reprice_held_whole_card_residencies(self) -> None:
        """Tighten held residency targets from current pricing without rewriting their grant forecasts.

        New allocator evidence and live reservation changes can show that a residency granted for two contexts
        now safely holds only one. Each governance tick re-forecasts pending or active jobs for the held model
        and keeps the strictest target. A target never grows mid-hold. Any resulting context reduction is booked
        with the verified reclaim ladder as the single restore obligation; the residency's own release still
        performs the physical regrowth and discharges that debt.
        """
        for device_index, state in self._held_residencies():
            model = state.model
            if model is None:
                continue
            matching_jobs = [
                job
                for job in (*self._job_tracker.jobs_in_progress, *self._job_tracker.jobs_pending_inference)
                if job.model == model
            ]
            targets = [
                target
                for job in matching_jobs
                if (
                    target := self._whole_card_ledger.target_process_count(
                        self._forecast_streaming(
                            job,
                            self._model_metadata.get_baseline(model),
                            device_index=device_index,
                        )
                    )
                )
                is not None
            ]
            if not targets:
                continue
            self._whole_card_ledger.tighten_target(device_index, min(targets))
            effective_target = self._whole_card_ledger.effective_target(state)
            live_count = self._process_map.num_loaded_inference_processes(device_index=device_index)
            if effective_target is None or live_count <= effective_target:
                continue
            # A tightened target can arrive while the head is still pre-staging into a slot that does not yet
            # carry its model name, so the shrink is told which slot that is, the same as the convergence shrink.
            after = self._scale_sparing(
                effective_target,
                device_index=device_index,
                protected_model=model,
                spared_process_id=state.prestage_process_id,
            )
            if after >= live_count:
                continue
            if self._reclaim_ladder is not None:
                self._reclaim_ladder.record_context_reduction(device_index)
            logger.info(
                f"Re-priced held whole-card residency for {model} on device {device_index}: "
                f"inference processes {live_count} -> {after}, tightened target {effective_target}."
            )

    def run_governance_tick(self) -> None:
        """Drive one resource-governance tick per control-loop iteration, independent of queue depth.

        The process manager calls this every iteration so the governor's degrade/restore response, the soft
        pop hold, and idle service-lane RAM containment (:meth:`_contain_idle_lane_ram`) are re-evaluated even
        when the inference queue is empty. Gated on the same budget switch as the rest of the memory
        machinery, which also no-ops against partial/mocked or early-startup config.
        """
        self._stamp_retention_hold_ages()
        if self._budget_active():
            self._govern_ram_pressure_if_pressured()
            self._reprice_held_whole_card_residencies()
            self._contain_idle_lane_ram()

    def reset_governance_to_baseline(self, reason: str) -> None:
        """Return RAM-governance state to a clean baseline, re-derived from live measurement next tick.

        Clears the soft pop hold and the governor's shed/draining episode bookkeeping, and drops the
        RAM-pressure entry from the pop-skip reasons so a stale count stops surfacing (other reasons are
        left intact). Deliberately leaves alone flags owned by other subsystems or latched for the session:
        the shared self-throttle pause (safety/self-maintenance, which self-expires), the operator
        supervisor pause, the downloads-only hold, and the post-processing / torch-compat breakers. Safe to
        call under genuine pressure: the next governance tick re-arms whatever the live host warrants.
        """
        logger.warning(f"Resetting RAM governance to baseline: {reason}")
        self._state.ram_pressure_pop_hold = False
        self._state.last_pop_skipped_reasons.pop("ram_pressure", None)
        self._ram_reclaim_cycle_at = 0.0
        self._ram_pressure_notified = False
        self._governor.reset_bookkeeping()

    def governance_healthy_but_held(self) -> bool:
        """Whether the soft RAM pop hold is engaged while the host is measurably healthy.

        The signature of a governance latch: pops are held for RAM pressure, yet the most recent
        danger-floor verdict is healthy and nothing is draining, so the hold should already have cleared.
        Distinct from a merely idle worker (which never sets the hold) and from the deliberate held-queue
        windows (whole-card establishment, heavy-head load, RAM-reclaim cycle), which own their own
        resolution. Returns False before the first tick has measured a verdict (treated as not-yet-healthy).
        Read by the recovery coordinator's healthy-hold watchdog.
        """
        if not self._state.ram_pressure_pop_hold:
            return False
        verdict = self._governor.last_ram_verdict
        if verdict is None or verdict.under_pressure:
            return False
        if self._ram_governor_state.draining_process_ids:
            return False
        return not (
            self.whole_card_residency_grace_active()
            or self.heavy_head_load_grace_active()
            or self.ram_reclaim_cycle_grace_active()
        )

    def _ram_per_process_ceiling_mb(self) -> float | None:
        """The configured per-process resident-RAM ceiling (MB), or None when disabled/unset.

        Read defensively (a partially-mocked config yields None) so the pressure path never crashes on a bad
        attribute; a non-positive value disables the ceiling.
        """
        ceiling = config_number(self._runtime_config.bridge_data.ram_per_process_max_mb)
        if ceiling is None or ceiling <= 0:
            return None
        return ceiling

    def _reduce_processes_under_ram_pressure(self) -> None:
        """Shed idle resident inference processes to return their resident-weight RAM to the OS.

        The RAM analogue of :attr:`StreamForecast.needs_process_count_reduction`: with the host over the
        danger floor, the structural remedy is fewer resident contexts, not another load on top. Only idle
        processes are stopped (``scale_inference_processes`` never kills a busy slot), so live work is
        spared. The reduction targets are decided by
        [`decide_process_reduction`][horde_worker_regen.process_management.scheduling.governance.ram_governor.decide_process_reduction]
        (per card on a multi-GPU host, worker-wide otherwise).
        """
        snapshot = self._build_host_memory_snapshot(self._ram_pressure_verdict())
        self._execute_governance_actions(decide_process_reduction(snapshot))

    def _estimated_resident_context_ram_mb(self) -> float:
        """Conservative system-RAM cost (MB) of one more resident inference context.

        Taken as the largest live inference process's measured resident RAM, which captures the model
        working set the allocator retains and will not free without a respawn. Falls back to the configured
        RAM reserve when no process has reported usage yet (only before any model has loaded; a card is only
        ever restored after a reduction that itself implies loaded, RAM-holding processes, so the measured
        value is the normal case).
        """
        live_context_ram_mb = [
            process_info.ram_usage_bytes / (1024 * 1024)
            for process_info in self._process_map.values()
            if process_info.process_type == HordeProcessType.INFERENCE and process_info.ram_usage_bytes > 0
        ]
        if live_context_ram_mb:
            return max(live_context_ram_mb)
        return self._ram_budget.reserve_mb

    def _ram_headroom_for_additional_context_mb(self) -> float:
        """Measured system-RAM headroom (MB) above the reserve and committed reserves for one more context."""
        available_ram_mb = self._measured_available_ram_mb()
        committed_ram_mb = self._reserve_ledger.total_ram_mb()
        return available_ram_mb - committed_ram_mb - self._ram_budget.reserve_mb

    def _restore_processes_after_ram_pressure(self) -> None:
        """Grow RAM-pressure-shed inference contexts back toward plan as system RAM proves headroom.

        The reduction sheds idle contexts to walk the host back above its absolute RAM floor; nothing else
        re-establishes them, so without this a card or single-GPU worker-wide pool that lost contexts to a
        RAM spike stays reduced for the rest of the run. The restore grants (incremental, RAM-gated,
        residency-aware) are decided by
        [`decide_shed_card_restore`][horde_worker_regen.process_management.scheduling.governance.ram_governor.decide_shed_card_restore].
        """
        if not self._ram_governor_state.shed_cards and self._ram_governor_state.worker_shed is None:
            return
        snapshot = self._build_host_memory_snapshot(self._ram_pressure_verdict())
        self._execute_governance_actions(decide_shed_card_restore(snapshot))

    def _committed_vram_reserve_mb(self, *, device_index: int | None = None) -> float:
        """Return the combined committed VRAM (MB) across every flow in the shared ledger.

        The dedicated post-processing lane has a fixed resident context charged by the process residency
        forecast, but each active post-processing job still registers its estimated upscaler/face-fixer peak
        here until the lane result arrives. Alchemy forms use the same ledger. Admission and
        residency-forecast gates subtract the combined figure so a freshly released slot is not handed VRAM
        concurrent work is about to claim.

        Args:
            device_index: Accepted for call-site symmetry; ledger flows are not card-attributed, so the
                worker-wide total is charged conservatively against any card.
        """
        del device_index
        return self._reserve_ledger.total_vram_mb()

    def _admission_baseline_mb(self, device_index: int | None) -> float:
        """Return the measured shared-device baseline (MB) for a card, or 0.0 when none is available.

        Falls back to 0.0 (raw-total capacity) whenever no provider is wired or none has been captured yet,
        so a cold start or a standalone unit test degrades to a capacity of the whole device total and the
        measured overlay never denies what the predictive gate admits.
        """
        if self._admission_baseline_provider is None:
            return 0.0
        baseline = self._admission_baseline_provider(device_index)
        return baseline if baseline is not None else 0.0

    def _committed_process_reserved_by_pid(self, device_index: int | None) -> dict[int, float]:
        """Return the live GPU processes' measured allocator reservation (MB) keyed by process id, for a card.

        The snapshot the planned-reserve overlay decays each entry against (a planned charge shrinks as its
        target's reservation materialises). Keyed by :attr:`HordeProcessInfo.process_id`, matching the id the
        planned entries are registered under.
        """
        reserved_by_pid: dict[int, float] = {}
        for process_info in self._process_map.values():
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.process_reserved_mb is None:
                continue
            reserved_by_pid[process_info.process_id] = float(process_info.process_reserved_mb)
        return reserved_by_pid

    def _in_flight_admitted_planned_units(self) -> set[str]:
        """Return the loading-process ids (as ledger units) whose admitted VRAM has not yet materialised.

        The authoritative in-flight-admitted set, derived from the model map rather than any parallel registry
        so it cannot leak: a model is counted while its load state is ``LOADING`` or ``LOADED_IN_RAM``, since a
        just-admitted preload sits in ``LOADED_IN_RAM`` before its allocator reservation grows, and two loads
        admitted the same cycle must both keep their planned charge until then (the double-admit guard). Once
        the model is ``IN_USE`` or its process finishes, faults, or dies, that unit stops appearing here and the
        next :meth:`CommittedReserveLedger.reconcile_planned` drops its planned charge by omission.

        ``LOADED_IN_RAM`` is overloaded: it is both the pre-materialisation state of a fresh preload and the
        state a resident model returns to when it is evicted from VRAM back to system RAM, so this set alone
        cannot tell a materialised-then-evicted anchor apart from one still in flight. The ledger closes that
        gap directly: :meth:`CommittedReserveLedger.effective_planned_vram_mb` consumes each anchor
        monotonically, so an anchor whose reservation has already grown stays consumed regardless of the load
        state it later revisits. Keyed by process id to match the unit :meth:`_send_preload` registers each
        grant under.

        A planned charge only survives while its target process could still materialise it. A model-map entry
        that still reads ``LOADING`` on a process that has since died or entered its terminal shutdown states can
        outlive the (throttled, once-per-cooldown) missing-process recovery that expires it, and a dead target's
        reservation never grows, so its charge would otherwise decay by neither materialisation nor omission and
        pin the overlay at full weight indefinitely; a head re-asking that same load then finds its own stale
        planned charge holding the card against it, a self-deadlock the identity cannot escape. Excluding process
        ids that are absent from the process map or in a terminal state drops such a charge here, through the
        same reconcile-by-omission that
        releases a finished load, with no separate death-path delete to keep in sync. Mirrors the committed
        ledger's own exclusion of ending/ended tenants, so the two overlays agree on which processes are live.
        """
        units: set[str] = set()
        for model_info in self._horde_model_map.root.values():
            if model_info.process_id is None:
                continue
            if model_info.horde_model_load_state not in (ModelLoadState.LOADING, ModelLoadState.LOADED_IN_RAM):
                continue
            process_info = self._process_map.get(model_info.process_id)
            if process_info is None:
                continue
            if process_info.last_process_state in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED):
                continue
            units.add(str(model_info.process_id))
        return units

    def _in_flight_dispatch_units(self, device_index: int | None) -> set[str]:
        """Return the in-progress job ids (as reservation units) whose dispatch reservation is still live.

        A dispatch reservation protects an admitted job's activation-inclusive peak until it materialises over
        the sampling window the device-free reading does not yet reflect. The authoritative live set is the
        job tracker's in-progress jobs (``INFERENCE_IN_PROGRESS``): a job that finalises, faults, or whose
        process dies leaves that set, and the next :meth:`CommittedReserveLedger.reconcile_planned` drops its
        reservation by omission with no death-path delete to keep in sync. On a multi-GPU host only jobs
        dispatched to ``device_index`` are counted, matching the per-card reservation view; a single-GPU
        (``device_index`` None) call counts every in-progress job.
        """
        jobs = (
            self._jobs_in_progress_on_card(device_index)
            if self._multi_gpu_routing_active and device_index is not None
            else self._job_tracker.jobs_in_progress
        )
        return {str(job.id_) for job in jobs if job.id_ is not None}

    def _measured_admission_candidate_delta_mb(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: str | None,
        *,
        process_id: int | None,
        disaggregated: bool,
    ) -> float | None:
        """Return the candidate job's marginal predicted VRAM (MB) for the measured overlay, net of resident credit.

        The gross charge is the sampler-only figure for a disaggregation-class job, else the whole-job sampling
        peak, matching how the predictive gate prices the same job, then raised by any learned watermark for the
        job's (baseline, resolution, platform) footprint at the matching stage (SAMPLE_ISOLATED for the
        disaggregated sampler-only figure, SAMPLE for the whole-job peak) so the measured floor is never priced
        below a peak the hardware has already demonstrated. When the job's model is already resident in
        its target process its weights are already in the measured committed floor, so they are credited out
        (the same resident-credit reasoning the retention gate applies) to avoid re-charging them. None (no
        estimate) propagates so the overlay treats the candidate as a zero marginal, never denying on an
        unpriceable cost.
        """
        static_gross_mb = (
            predict_job_sampler_only_vram_mb(job, baseline)
            if disaggregated
            else predict_job_sampling_vram_mb(job, baseline)
        )
        if static_gross_mb is None:
            return None
        gross_mb = self._learned_sampling_peak_mb(
            job,
            baseline,
            static_seed_mb=static_gross_mb,
            stage=FootprintStage.SAMPLE_ISOLATED if disaggregated else FootprintStage.SAMPLE,
        )
        resident_credit_mb = 0.0
        if self._candidate_weights_resident_on_process(job.model, process_id):
            resident_credit_mb = predict_job_weight_mb(job, baseline) or 0.0
        return max(0.0, gross_mb - resident_credit_mb)

    def _candidate_weights_resident_on_process(self, model_name: str | None, process_id: int | None) -> bool:
        """Whether ``model_name``'s weights already occupy VRAM on ``process_id`` (dispatch materialises nothing).

        The single residency truth two admission concerns share: the resident-weight credit that keeps a
        candidate delta from re-charging weights the committed floor already counts, and the arbiter's
        ``candidate_already_resident`` no-op admit. Read primarily from the horde model map's residency state on
        the matching process. The committed floor charges those weights by the process's own measured
        reservation, keyed by the process map's ``loaded_horde_model_name``; when the model map's process pointer
        transiently lags that record the two disagree, so a fallback also credits residency when the target
        process itself reports this model loaded and the model map agrees the model is VRAM-resident. Aligning
        the credit with the floor's own truth stops the divergence from double-charging resident weights (once
        in the committed floor, again as the candidate delta) and wedging a dispatch to an idle resident model.

        The parent's own retention record (``retained_resident_model``) is read first, because it is the only
        truth that covers a slot holding weights *between* jobs. The model map tracks load transitions a child
        reports, and a slot that finished a job under a retention grant reports its weights back in system RAM
        or, on a disaggregated sampler, reports no transition at all: the sample stage emits none, so such a
        slot never reads as VRAM-resident there however long it holds its UNet. Without this arm every job
        landing on a retained resident is charged its weights a second time (once through the committed floor
        that already contains them), which holds a job whose model is on the card at full materialisation
        price. The credit taken against it is the core weight figure, which is exactly what a component-only
        (UNet-alone) retention holds.
        """
        if model_name is None or process_id is None:
            return False
        retainer = self._process_map.get(process_id)
        if retainer is not None and retainer.retained_resident_model == model_name:
            return True
        model_info = self._horde_model_map.root.get(model_name)
        model_map_says_vram_resident = model_info is not None and model_info.horde_model_load_state in (
            ModelLoadState.LOADED_IN_VRAM,
            ModelLoadState.IN_USE,
        )
        if model_info is not None and model_info.process_id == process_id and model_map_says_vram_resident:
            return True
        return retainer is not None and retainer.loaded_horde_model_name == model_name and model_map_says_vram_resident

    def set_vram_arbiter(self, arbiter: VramArbiter) -> None:
        """Inject the single VRAM arbiter: the preload-admission authority and the observational overlay elsewhere."""
        self._vram_arbiter = arbiter
        self._owns_private_vram_arbiter = False

    def set_reclaim_ladder(self, reclaim_ladder: VerifiedReclaimLadder) -> None:
        """Inject the worker's single verified reclaim ladder, the owner of every reclaim restore obligation.

        A live-context reduction the admission path takes is booked with the engine here, so it is unwound
        (the pool regrown) on the same debounced-HEALTHY signal, in the same LIFO order, as the lane pauses the
        engine issues itself. Without the injection a reduction is not booked and nothing grows the pool back.
        """
        self._reclaim_ladder = reclaim_ladder

    def set_device_free_mb_provider(self, provider: Callable[[int], float | None]) -> None:
        """Inject the truthful per-card device-free reading source (the parent's NVML view).

        The manager-driven cycle passes its explicit reading map into :meth:`build_vram_arbiter_snapshot`;
        this provider covers the self-primed path (a scheduler consult before or outside a manager tick), so
        the measured-truth admission identity keeps its primary input on every snapshot the scheduler builds.
        """
        self._device_free_mb_provider = provider

    def set_available_ram_mb_provider(self, provider: Callable[[], float]) -> None:
        """Inject the available-system-RAM reading source, replacing the live psutil read.

        The RAM admission gates price a preload against the host the worker actually runs on, so production
        never overrides this. A harness constructing scheduling scenarios must, because otherwise every RAM
        gate silently prices against whatever machine happens to run the suite, and a scenario that admits a
        heavy model passes or fails with the runner's free RAM instead of the scenario's own state.
        """
        self._available_ram_mb_provider = provider

    def set_footprint_store(self, store: LearnedFootprintStore) -> None:
        """Inject the shared learned-footprint store the message dispatcher also observes into.

        Admission pricing of a job's sampling peak consults it so a measured activation high-water raises the
        static per-model seed; one instance is shared across the parent so every observed peak and every priced
        estimate reference the same watermarks.
        """
        self._footprint_store = store

    def _sampling_footprint_key(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: str | None,
        *,
        stage: FootprintStage,
    ) -> FootprintKey | None:
        """Build the footprint key for ``job`` at ``stage``, or None when it cannot be keyed.

        The key is (baseline, resolution bucket by the job's maximum dimension, host platform, stage). The stage
        distinguishes a whole-job monolithic peak (:attr:`FootprintStage.SAMPLE`) from a disaggregated UNet-only
        sampler peak (:attr:`FootprintStage.SAMPLE_ISOLATED`): the two are physically different quantities and,
        since watermarks are raise-only, must not share a key. A None baseline or an absent width/height cannot
        be attributed to a footprint population, so it returns None and the caller keeps the static seed.
        """
        if baseline is None:
            return None
        width = job.payload.width
        height = job.payload.height
        if width is None or height is None:
            return None
        return FootprintKey(
            model_baseline=str(baseline),
            resolution_bucket=ResolutionBucket.from_dimensions(width, height, job.payload.n_iter or 1),
            platform=sys.platform,
            stage=stage,
        )

    def _learned_resident_footprint_mb(
        self,
        model_name: str | None,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
    ) -> float | None:
        """The measured at-rest device footprint (MB) of this checkpoint, or None when nothing was measured.

        Reads the per-checkpoint :attr:`FootprintStage.RESIDENT` watermark: what a slot holding this specific
        file actually costs the card with no activation on it. Keyed on the checkpoint rather than the
        baseline because resident weights are a property of the file (two checkpoints of one baseline differ
        by gigabytes), which is what lets the streaming forecast price a heavy checkpoint by its own
        measurement instead of its architecture's seed.

        The store records the whole *device* charge, the process's CUDA context included, whereas the
        forecast charges contexts separately from model bytes (the loading process's first-context overhead
        plus a marginal per sibling). The context constant is therefore netted back out here so the returned
        figure is in the same context-exclusive terms as the static weight seeds it will raise, and the
        context is not paid for twice.

        Returns None for a cold key, an unkeyable model, or a store-less scheduler, which leaves the
        forecast's arithmetic exactly as the static seeds compute it.
        """
        store = self._footprint_store
        if store is None or model_name is None or baseline is None:
            return None
        watermark_mb = store.estimate_mb(
            FootprintKey(
                model_baseline=str(baseline),
                resolution_bucket=None,
                platform=sys.platform,
                stage=FootprintStage.RESIDENT,
                checkpoint=model_name,
            ),
            static_seed_mb=0.0,
        )
        if watermark_mb <= 0.0:
            return None
        return max(0.0, watermark_mb - self.resolved_context_constant_mb())

    def _measured_resident_footprint_mb(
        self,
        model_name: str | None,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
    ) -> tuple[float | None, int]:
        """The margined *measured* at-rest charge (MB) of this checkpoint and how many runs back it.

        The bidirectional counterpart to :meth:`_learned_resident_footprint_mb`: where that one can only
        raise the static seed, this one reports what the hardware actually did, which may be well below the
        seed. It is returned separately rather than folded into the seed overlay because the fit arithmetic
        must keep planning on the conservative figure; the measurement's only authority is to retire a
        whole-card residency claim the seed alone was making (see
        :attr:`~horde_worker_regen.process_management.resources.resource_budget.StreamForecast.measured_retires_whole_card_intent`).

        The context constant is netted out for the same reason it is in the raise-only accessor: the store
        keeps whole-device charges while the forecast charges contexts separately.

        Returns ``(None, 0)`` for an under-observed key, an unkeyable model, or a store-less scheduler.
        """
        store = self._footprint_store
        if store is None or model_name is None or baseline is None:
            return (None, 0)
        key = FootprintKey(
            model_baseline=str(baseline),
            resolution_bucket=None,
            platform=sys.platform,
            stage=FootprintStage.RESIDENT,
            checkpoint=model_name,
        )
        measured_mb = store.measured_estimate_mb(key)
        if measured_mb is None:
            return (None, 0)
        return (max(0.0, measured_mb - self.resolved_context_constant_mb()), store.observation_count(key))

    def _learned_sampling_peak_mb(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: str | None,
        *,
        static_seed_mb: float,
        stage: FootprintStage,
    ) -> float:
        """Raise a static sampling-peak seed by any learned watermark for this job's ``stage`` footprint key.

        The static predictor stays the seed; the learned overlay can only ever RAISE it (a cold key, a None
        baseline, or an unkeyable job returns the seed unchanged). This is the single seam admission pricing of
        sampling work routes through so a measured activation peak is never undershot. Callers pricing whole-job
        sampling pass :attr:`FootprintStage.SAMPLE`; callers pricing a disaggregated UNet-only sampler pass
        :attr:`FootprintStage.SAMPLE_ISOLATED` so a monolithic whole-job watermark never over-prices it.

        The activation keys are fed from the child's reserved-peak reports, which include allocator cache, so
        the margined measured estimate is deliberately not used here: pricing a sampling window at reserved
        peak times margin plus context charges more than the seed for an ordinary SDXL job and refuses
        co-residency the card can hold. Measured pricing is confined to the resident footprint, where the
        backend measures weights alone.
        """
        store = self._footprint_store
        if store is None:
            return static_seed_mb
        key = self._sampling_footprint_key(job, baseline, stage=stage)
        if key is None:
            return static_seed_mb
        return store.estimate_mb(key, static_seed_mb=static_seed_mb)

    def observe_disaggregated_sampling_peak(self, job_info: HordeJobInfo, peak_reserved_mb: float) -> None:
        """Fold a disaggregated sampler's measured peak into the store under this job's SAMPLE_ISOLATED key.

        The message dispatcher observes the monolithic case under :attr:`FootprintStage.SAMPLE`, but a
        disaggregated UNet-only sampler's peak arrives through the orchestrator (which alone binds the pinned
        sampler process to the job's stage), so this is the seam that closes that gap. It records under
        :attr:`FootprintStage.SAMPLE_ISOLATED`, a distinct key from the monolithic whole-job peak: mixed
        operation is designed (a stage fault re-routes a disaggregated job monolithic), so a single monolithic
        peak must not raise the isolated-sampler estimate and forfeit the second concurrent sampler. The peak
        figure is the pinned sampler process's latest reported ``process_peak_reserved_mb`` at sample completion:
        it is the allocator high-water since the process's previous memory report, so it can lag the true
        sampling peak by up to one report interval, but it is the best-attributable reading at this seam.
        Raise-only semantics apply (a non-positive reading is ignored by the store); a store-less, unkeyable, or
        model-less job is a no-op.
        """
        store = self._footprint_store
        if store is None:
            return
        job = job_info.sdk_api_job_info
        if job.model is None:
            return
        key = self._sampling_footprint_key(
            job,
            self._model_metadata.get_baseline(job.model),
            stage=FootprintStage.SAMPLE_ISOLATED,
        )
        if key is None:
            return
        store.observe_peak(key, peak_reserved_mb)

    def _gpu_process_activity_ids(self, device_index: int | None) -> tuple[frozenset[int], frozenset[int]]:
        """Return the idle and busy GPU-process ids on a card, for the arbiter's release-cache targeting.

        A RELEASE_CACHE target is an idle GPU process that plausibly still holds *reclaimable* allocator
        cache: its measured reservation exceeds its in-use allocation by at least
        ``_RELEASE_CACHE_MIN_RECLAIMABLE_MB``, so an ``empty_cache`` could return that reserved-but-unallocated
        margin to the card. A process whose reservation is its resident footprint (a component/VAE/post-process
        lane holding encoders or a still-loaded model, where allocated tracks reserved) has no such margin and
        is not a target: asking it to release frees nothing, and emitting a rung that can never yield would keep
        the escalation ladder non-empty forever. An idle process that still holds a horde model is an eviction
        candidate (a distinct ladder rung), not a cache-release target, so it is left out here to keep the two
        remedies separate. A busy process is never a target. When the in-use allocation has not yet been
        reported the reclaimable margin cannot be measured, so the process is not targeted rather than assumed
        to hold cache.
        """
        cache_bearing = (
            HordeProcessType.INFERENCE,
            HordeProcessType.POST_PROCESS,
            HordeProcessType.VAE_LANE,
            HordeProcessType.COMPONENT,
        )
        idle: set[int] = set()
        busy: set[int] = set()
        for process_info in self._process_map.values():
            if process_info.process_type not in cache_bearing:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.is_process_busy():
                busy.add(process_info.process_id)
                continue
            if process_info.loaded_horde_model_name is not None:
                continue
            reserved_mb = process_info.process_reserved_mb
            allocated_mb = process_info.process_allocated_mb
            if reserved_mb is None or allocated_mb is None:
                continue
            if reserved_mb - allocated_mb >= _RELEASE_CACHE_MIN_RECLAIMABLE_MB:
                idle.add(process_info.process_id)
        return frozenset(idle), frozenset(busy)

    def build_vram_arbiter_device_state(
        self,
        device_index: int | None,
        *,
        active_sampling_peaks_total_mb: float = 0.0,
        governor_state: GovernorState | None = None,
        device_free_mb: float | None = None,
        reclaim_unresolved: bool = False,
    ) -> DeviceVramState:
        """Assemble the frozen per-device VRAM measurement the arbiter prices this cycle's requests against.

        Sourced entirely from figures the scheduler already holds: the measured-truth admission identity's
        primary input (the frozen device-free reading, passed in) plus the outstanding reservations and the
        noise buffer, and the concurrent-sampling headroom's terms (baseline, fixed and marginal context
        overhead, the live context counts, the operator reserve, the lane decode spike). The committed floor
        and staleness are still assembled for diagnostics and telemetry but the admission path no longer reads
        them. No NVML read and no torch import; the measurement is the parent's already-reconciled state.

        Both admission-reservation flows are reconciled by omission before the outstanding total is read: a
        preload whose process finished, faulted, or died, or an in-progress job that finalised or died, drops
        its reservation here so a re-ask is never blocked by a dead unit's stale reservation. The preload
        flow's share is also carried separately, so the arbiter can price drain-side requests (post-processing)
        net of RAM-staged loads whose VRAM claim waits on that very drain completing.
        """
        raw_total_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        baseline_mb = self._admission_baseline_mb(device_index)
        committed_mb = self._process_map.committed_vram_mb(
            context_constant_mb=self.resolved_context_constant_mb(),
            device_index=device_index,
        )
        foreign_floor_mb = self._sustained_foreign_floor_mb(
            device_index,
            total_vram_mb=raw_total_mb,
            device_free_mb=device_free_mb,
            worker_footprint_mb=committed_mb,
        )
        oldest_report_age = self._process_map.oldest_committed_report_age_seconds(
            now=time.time(),
            device_index=device_index,
        )
        committed_is_stale = oldest_report_age is not None and oldest_report_age > _REPORT_STALENESS_SECONDS
        self._reserve_ledger.reconcile_planned(PRELOAD_ADMISSION_FLOW, self._in_flight_admitted_planned_units())
        self._reserve_ledger.reconcile_planned(DISPATCH_ADMISSION_FLOW, self._in_flight_dispatch_units(device_index))
        per_process_reserved = self._committed_process_reserved_by_pid(device_index)
        planned_mb = self._reserve_ledger.effective_planned_vram_mb(per_process_reserved)
        preload_planned_mb = self._reserve_ledger.effective_planned_vram_mb_for_flow(
            PRELOAD_ADMISSION_FLOW,
            per_process_reserved,
        )
        noise_buffer_mb = admission_noise_buffer_mb(raw_total_mb)
        self._admission_headroom_mb_by_device[device_index if device_index is not None else 0] = (
            None if device_free_mb is None else device_free_mb - planned_mb - noise_buffer_mb
        )

        override_mb = self._config_overhead_override_mb()
        per_process_mb = self._overhead.per_process_mb(config_override_mb=override_mb, device_index=device_index)
        marginal_mb = self._overhead.marginal_mb(config_override_mb=override_mb, device_index=device_index)
        if marginal_mb is None or marginal_mb <= 0:
            marginal_mb = _SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB
        safety_contexts = (
            1 if self._safety_on_gpu_permitted and not self._process_lifecycle.is_safety_gpu_paused else 0
        )
        post_process_contexts = (
            0
            if self._process_lifecycle.is_post_process_gpu_paused
            else self._process_map.num_post_process_processes(device_index=device_index)
        )
        vae_lane_contexts = (
            0
            if self._process_lifecycle.is_vae_lane_gpu_paused
            else self._process_map.num_vae_lane_processes(device_index=device_index)
        )
        component_lane_contexts = (
            0
            if self._process_lifecycle.is_component_gpu_paused
            else self._process_map.num_component_processes(device_index=device_index)
        )
        idle_process_ids, busy_process_ids = self._gpu_process_activity_ids(device_index)
        return DeviceVramState(
            total_vram_mb=raw_total_mb,
            baseline_mb=baseline_mb,
            committed_vram_mb=committed_mb,
            planned_unmaterialized_mb=planned_mb,
            preload_planned_unmaterialized_mb=preload_planned_mb,
            committed_is_stale=committed_is_stale,
            noise_buffer_mb=noise_buffer_mb,
            per_process_reserved_mb=per_process_reserved,
            idle_process_ids=idle_process_ids,
            busy_process_ids=busy_process_ids,
            num_loaded_inference_processes=self._process_map.num_loaded_inference_processes(
                device_index=device_index,
            ),
            safety_context_count=safety_contexts,
            safety_reclaim_allowed=(
                self._residency_should_pause_safety(device_index) and not self._has_safety_backlog()
            ),
            post_process_context_count=post_process_contexts,
            vae_lane_context_count=vae_lane_contexts,
            vae_lane_reclaim_allowed=(
                vae_lane_contexts > 0
                and self._has_idle_service_lane_for_reclaim(HordeProcessType.VAE_LANE, device_index)
            ),
            component_lane_context_count=component_lane_contexts,
            component_lane_reclaim_allowed=(
                component_lane_contexts > 0
                and self._has_idle_service_lane_for_reclaim(HordeProcessType.COMPONENT, device_index)
            ),
            per_process_overhead_mb=per_process_mb,
            marginal_mb=marginal_mb,
            vram_reserve_mb=self._vram_budget.reserve_mb,
            vae_lane_decode_spike_mb=self._vae_lane_decode_spike_charge_mb(device_index=device_index),
            active_sampling_peaks_total_mb=active_sampling_peaks_total_mb,
            governor_state=governor_state,
            device_free_mb=device_free_mb,
            foreign_floor_mb=foreign_floor_mb,
            reclaim_unresolved=reclaim_unresolved,
        )

    def _sustained_foreign_floor_mb(
        self,
        device_index: int | None,
        *,
        total_vram_mb: float | None,
        device_free_mb: float | None,
        worker_footprint_mb: float,
    ) -> float | None:
        """Return this card's sustained foreign-VRAM floor (MB), advancing the trailing-window tracker.

        The instantaneous foreign reading is ``total - device_free - worker_footprint``: the device VRAM the
        NVML reading shows used, less the worker's own committed footprint (the ledger already excludes the
        shared baseline), so what remains is the VRAM the OS/desktop/other processes hold. It is contributed
        as a sample only when the card is fully measured and at least one worker process has reported its VRAM
        footprint; otherwise (cold start, children not yet reporting) no sample is added and the floor stays
        None, preserving the arbiter's pre-foreign behaviour. The tracker returns the sustained minimum over
        its trailing window, or None until a full window has been observed.
        """
        device_key = device_index if device_index is not None else 0
        if (
            total_vram_mb is not None
            and device_free_mb is not None
            and self._process_map.committed_ledger_processes(device_index)
        ):
            foreign_now_mb: float | None = total_vram_mb - device_free_mb - worker_footprint_mb
        else:
            foreign_now_mb = None
        return self._foreign_vram_floor.update(device_key, foreign_now_mb, now=self._foreign_floor_clock())

    def achievable_ceiling_mb(self, device_index: int | None) -> float | None:
        """Return a card's current achievable VRAM ceiling (MB): total net of noise and the sustained foreign floor.

        The most VRAM this card could offer one load right now, read live so a conditional ceiling hold can lift
        the moment the foreign floor recedes. None when the card's total is unknown (no GPU child has reported
        yet). Reads the sustained foreign floor without recording a fresh sample, so a ceiling read never
        perturbs the observation window.
        """
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        if total_vram_mb is None:
            return None
        device_key = device_index if device_index is not None else 0
        foreign_floor_mb = self._foreign_vram_floor.current_floor_mb(device_key, now=self._foreign_floor_clock())
        return total_vram_mb - admission_noise_buffer_mb(total_vram_mb) - (foreign_floor_mb or 0.0)

    def build_vram_arbiter_snapshot(
        self,
        *,
        active_sampling_peaks_total_mb: float = 0.0,
        governor_states: Mapping[int, GovernorState] | None = None,
        device_free_mb_by_device: Mapping[int, float] | None = None,
        reclaim_unresolved_by_device: Mapping[int, bool] | None = None,
    ) -> MeasuredVramSnapshot:
        """Build the whole-worker frozen snapshot for one cycle, one entry per driven card.

        The single-GPU/worker-wide reading (device index None) is stored under card 0, matching how the
        scheduler keys its per-card admission counters, so a None-keyed request resolves to it. The parent
        supplies ``governor_states`` (the device-free governor's committed state per card), the truthful NVML
        ``device_free_mb_by_device`` reading, and ``reclaim_unresolved_by_device`` (whether the verified
        reclaim ladder has exhausted itself while SATURATED per card) so each device state carries the
        admission inputs. A missing readings map falls back to the injected device-free provider (see
        :meth:`set_device_free_mb_provider`) so a self-primed snapshot keeps the identity's primary input;
        with neither source the reading is absent and admission defers on it.
        """
        device_indices = {process_info.device_index for process_info in self._process_map.values()}
        if not device_indices:
            device_indices = {0}
        devices: dict[int, DeviceVramState] = {}
        for device_index in sorted(device_indices):
            governor_state = governor_states.get(device_index) if governor_states is not None else None
            if device_free_mb_by_device is not None:
                device_free_mb = device_free_mb_by_device.get(device_index)
            elif self._device_free_mb_provider is not None:
                device_free_mb = self._device_free_mb_provider(device_index)
            else:
                device_free_mb = None
            reclaim_unresolved = (
                reclaim_unresolved_by_device.get(device_index, False)
                if reclaim_unresolved_by_device is not None
                else False
            )
            devices[device_index] = self.build_vram_arbiter_device_state(
                device_index if self._multi_gpu_routing_active else None,
                active_sampling_peaks_total_mb=active_sampling_peaks_total_mb,
                governor_state=governor_state,
                device_free_mb=device_free_mb,
                reclaim_unresolved=reclaim_unresolved,
            )
        return MeasuredVramSnapshot(devices=devices)

    def _overlap_memory_verdict(
        self,
        candidate_job: ImageGenerateJobPopResponse,
        *,
        target_device_index: int | None,
    ) -> bool | None:
        """The arbiter's answer to ``candidate_job``'s overlap memory demand: admits, withholds, or unpriced.

        Prices the candidate's marginal device cost against the cycle-frozen admission floor as a
        :attr:`VramRequestKind.MONOLITHIC_DISPATCH`: True when a FITS verdict admits, False when a DEFER or
        DENY withholds. Returns None when the demand cannot be priced at all (the arbiter is unwired,
        no cycle snapshot is frozen, or the candidate is model-less): the caller then relaxes the memory answer
        to admit, matching the predictive gate's admit-on-missing-telemetry contract, without treating the
        absence of telemetry as positive confirmation of room. A disaggregation-class candidate is priced with
        its sampler-only delta (``disaggregated``), so the concurrent decode spike the sampling gate already
        reserves is never double-counted here. No actuations run: the arbiter verdict's actuations are ignored
        because reclaim stays single-owner (the preload path drives it).
        """
        if not self._budget_active():
            return None
        arbiter = self._vram_arbiter
        if arbiter is None or not arbiter.has_cycle or candidate_job.model is None:
            return None
        baseline = self._model_metadata.get_baseline(candidate_job.model)
        resident_model_info = self._horde_model_map.root.get(candidate_job.model)
        resident_pid = resident_model_info.process_id if resident_model_info is not None else None
        request = VramRequest(
            kind=VramRequestKind.MONOLITHIC_DISPATCH,
            job_label=str(candidate_job.model),
            baseline=baseline,
            device_index=target_device_index,
            target_process_id=resident_pid,
            candidate_delta_mb=self._measured_admission_candidate_delta_mb(
                candidate_job,
                baseline,
                process_id=resident_pid,
                disaggregated=self._is_disaggregation_class_eligible(candidate_job),
            ),
        )
        return arbiter.evaluate(request).admits

    def _concurrent_overlap_allowed(
        self,
        candidate_job: ImageGenerateJobPopResponse,
        *,
        target_device_index: int | None = None,
    ) -> bool:
        """Whether ``candidate_job`` may start while other jobs are already sampling, the arbiter deciding memory.

        The concurrency cap (``max_threads``) only counts in-flight jobs; it does not look at what those jobs
        are, how far along they are, or how much the card can hold. This gate adds both missing dimensions: a
        temporal/structural guard that keeps a newcomer off a running job's memory-hungry startup beat, and the
        VRAM arbiter's authoritative answer to whether the card can hold the overlap at all.

        The non-memory guards run first and can decline overlap on their own:
            * The first job (nothing in flight) always starts: with no overlap there is no memory question.
            * An extra-large (whole-card tier) model neither joins a busy card nor shares one, whatever the
              card's headroom; that contract is the tier's, not the card's.
            * A newcomer must let the running job make size-appropriate sampling headway (none for light+light,
              a startup beat when a memory-hungry pairing has room, the strictest headway behind a batch) before
              it joins, so two loads and activation peaks do not stack into a step-timeout teardown.

        The memory question is then the arbiter's: a :attr:`VramRequestKind.MONOLITHIC_DISPATCH` verdict that
        FITS admits the overlap, a DEFER or DENY withholds it this cycle and the dispatch re-asks
        naturally on the next scheduling pass. This seam runs no actuations on a DEFER (reclaim is single-owner,
        driven only by the preload path); a cold start or an unwired arbiter relaxes the memory answer to admit.

        The headway relaxation is driven by positive confirmation only: a heavy pairing's headway drops to the
        startup-beat constant (and a batch is bounded by the strictest headway rather than a hard block) only
        when the arbiter has actually confirmed room this cycle. A cold start (no cycle) keeps the strict
        headway fractions, since the admit-on-missing-telemetry relaxation is not evidence the card has room.

        A blocked job is not dropped: it keeps its queue position and dispatches once the in-flight job(s)
        progress or finish and the card has room.

        Args:
            candidate_job: The job being considered for dispatch.
            target_device_index: On a multi-GPU host, the card this candidate would run on; the headway check
                then considers only jobs already sampling on that same card (jobs on other cards do not contend
                for its VRAM or sampler), and the arbiter prices the demand against that card. ``None`` (and
                every single-GPU call) keeps the worker-wide comparison.
        """
        if self._multi_gpu_routing_active and target_device_index is not None:
            in_progress_jobs: tuple[ImageGenerateJobPopResponse, ...] | list[ImageGenerateJobPopResponse] = (
                self._jobs_in_progress_on_card(target_device_index)
            )
        else:
            in_progress_jobs = self._job_tracker.jobs_in_progress
        if not in_progress_jobs:
            return True

        candidate_tier = self._model_size_tier(candidate_job.model)
        if candidate_tier >= _ModelSizeTier.EXTRA_LARGE:
            return False

        # The memory question is the arbiter's, resolved once and only when a rule needs it. ``memory_ample``
        # is the arbiter's positive confirmation of room (a real cycle that admits), which relaxes the headway;
        # ``memory_admits`` is the veto, which withholds only when a real cycle denies and relaxes to admit when
        # the demand could not be priced. A cold start therefore keeps the strict headway yet admits on memory.
        memory_verdict_cache: bool | None = None
        memory_evaluated = False

        def memory_verdict() -> bool | None:
            nonlocal memory_verdict_cache, memory_evaluated
            if not memory_evaluated:
                memory_verdict_cache = self._overlap_memory_verdict(
                    candidate_job,
                    target_device_index=target_device_index,
                )
                memory_evaluated = True
            return memory_verdict_cache

        def memory_admits() -> bool:
            return memory_verdict() is not False

        def memory_ample() -> bool:
            return memory_verdict() is True

        candidate_batched = self._job_batch_amount(candidate_job) > 1
        if candidate_batched and not memory_ample():
            return False

        # Higher performance modes pull a newcomer's sampling into the current job's tail sooner by shrinking
        # the headway it must wait for; the arbiter's memory verdict below still independently gates the overlap.
        headway_scale = _performance_mode_headway_scale(self._runtime_config.bridge_data)

        for job in in_progress_jobs:
            running_tier = self._model_size_tier(job.model)
            if running_tier >= _ModelSizeTier.EXTRA_LARGE:
                return False

            if candidate_batched or self._job_batch_amount(job) > 1:
                # A batch multiplies the activation peak, so without confirmed room it keeps the hard block;
                # with room it is bounded instead by the strictest headway (never the startup-beat relaxation,
                # which is sized for single jobs).
                if not memory_ample():
                    return False
                required_headway = _OVERLAP_HEADWAY_BOTH_HEAVY
            else:
                required_headway = self._required_overlap_headway(running_tier, candidate_tier)
                if required_headway > 0.0 and memory_ample():
                    required_headway = _OVERLAP_HEADWAY_AMPLE_VRAM

            required_headway *= headway_scale

            if required_headway <= 0.0:
                continue
            if self._in_flight_progress_fraction(job) < required_headway:
                return False

        return memory_admits()

    def set_vram_growth_hold(self, device_index: int, active: bool) -> None:
        """Set or clear the device-free governor's growth hold for a card (parent control loop only).

        While ``active`` the scheduler withholds every action that would grow the card's VRAM footprint: a new
        model preload onto a process that does not already hold it, a safety GPU restore, and a paused-lane
        restart. In-flight sampling is untouched. Called each governor tick, so the hold tracks the card's
        live proximity-to-cliff state.
        """
        self._vram_growth_hold_by_device[device_index] = active

    def is_vram_growth_held(self, device_index: int | None) -> bool:
        """Whether the device-free governor is holding new VRAM growth on a card (default card 0 for None).

        The single-GPU/worker-wide key (None) maps to card 0, matching the arbiter snapshot and admission
        counters. False for any card the governor has not yet sampled (no hold), so a host without NVML never
        holds growth.
        """
        return self._vram_growth_hold_by_device.get(device_index if device_index is not None else 0, False)

    def set_governor_state(self, device_index: int, state: GovernorState) -> None:
        """Record the device-free governor's committed state for a card (parent control loop only).

        Pushed each governor tick alongside :meth:`set_vram_growth_hold`. Retention reads this state (not the
        derived hold) so a resident is only kept while the card is HEALTHY.

        This is also where a card's continuous time off HEALTHY is measured, since it is the one place the
        committed state arrives on the worker's own cadence. Once that has held for
        :data:`_RETENTION_PRESSURE_REVOKE_SECONDS`, retained copies no job has come back for are given back
        (:meth:`_revoke_stale_retentions_under_pressure`). Debouncing here
        rather than inventing a second governor keeps one committed state behind every decision that reads it.
        """
        self._governor_states_by_device[device_index] = state
        if state is GovernorState.HEALTHY:
            self._governor_pressure_since.pop(device_index, None)
            return
        pressure_since = self._governor_pressure_since.setdefault(device_index, self._clock())
        if (self._clock() - pressure_since) >= _RETENTION_PRESSURE_REVOKE_SECONDS:
            self._revoke_stale_retentions_under_pressure(device_index)

    def governor_state(self, device_index: int | None) -> GovernorState:
        """Return the device-free governor's committed state for a card (default card 0 for None).

        The single-GPU/worker-wide key (None) maps to card 0, matching the growth hold and the arbiter snapshot.
        HEALTHY for any card the governor has not yet sampled, so a host without NVML never denies retention on
        governor grounds.
        """
        return self._governor_states_by_device.get(
            device_index if device_index is not None else 0, GovernorState.HEALTHY
        )

    def reclaim_one_idle_model_under_pressure(self, *, device_index: int | None = None) -> bool:
        """Reclaim idle resident VRAM under a physical-overcommit pressure signal (no specific loading candidate).

        The no-candidate counterpart of the WDDM-paging rising-edge reclaim: one under-pressure sweep of idle
        resident models on the card, reusing :meth:`unload_models_from_vram`'s target selection and its existing
        protections (busy, in-progress, queued-lookahead, lane, and pinned processes are all skipped). The
        anchor passed is a process to protect (a busy inference process if one exists, else any inference
        process on the card); the sweep evicts the coldest eligible idle resident, never a live model. Returns
        True when an unload was issued.
        """
        anchor = self._pressure_reclaim_anchor(device_index=device_index)
        if anchor is None:
            return False
        return self.unload_models_from_vram(anchor, under_pressure=True, device_index=device_index)

    def recalibrate_committed_ledger(self, *, device_index: int | None = None) -> int:
        """Recalibrate the committed-VRAM ledger to device truth by releasing every idle lane's allocator cache.

        The committed ledger sums each lane's ``memory_reserved()``. That per-process figure can detach upward
        from device reality: an unloaded model's blocks the torch caching allocator has not returned, or a
        reservation the WDDM driver already spilled to host RAM, both keep counting against committed while the
        physical pages are free. Such a phantom over-count cannot be cured by evicting a model (there is nothing
        resident to evict) and would otherwise defer every admission forever on a figure the card does not hold.

        Emptying an idle lane's allocator cache returns those blocks and prompts a fresh memory report, so the
        parent's ``process_reserved_mb`` (hence committed) converges back to device truth. A busy (actively
        sampling) lane is skipped: its reservation is live, and its cache returns through the ordinary post-stage
        path. Terminal (ending/ended) lanes and lanes that have never reported a reservation are skipped too, as
        is any process whose dispatch contract does not include ``RELEASE_ALLOCATOR_CACHE`` (the fan-out targets
        only :data:`ALLOCATOR_CACHE_CAPABLE_PROCESS_TYPES`, so a routing-incapable process is never asked).
        Returns how many lanes were asked to release.

        Args:
            device_index: When given, recalibrate only lanes pinned to that card; when None, every card.
        """
        lanes_asked = 0
        for process_info in self._process_map.values():
            if process_info.process_type not in ALLOCATOR_CACHE_CAPABLE_PROCESS_TYPES:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.process_reserved_mb is None:
                continue
            if process_info.last_process_state in (
                HordeProcessState.PROCESS_ENDING,
                HordeProcessState.PROCESS_ENDED,
            ):
                continue
            if process_info.is_process_busy():
                continue
            if self.release_allocator_cache(process_info.process_id):
                lanes_asked += 1
        return lanes_asked

    def _pressure_reclaim_anchor(self, *, device_index: int | None) -> HordeProcessInfo | None:
        """Pick the process to protect for a no-candidate pressure reclaim: a busy inference process, else any.

        The reclaim sweep excludes its anchor, so anchoring on a busy inference process keeps a live job's
        model safe while the sweep evicts the coldest idle resident. Falls back to any inference process on the
        card (its own model is still protected by the in-progress guard) and to None when the card has none.
        """
        fallback: HordeProcessInfo | None = None
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if fallback is None:
                fallback = process_info
            if process_info.is_process_busy():
                return process_info
        return fallback

    def _clearance_lease_active(self) -> bool:
        """Whether the per-process GPU denoise clearance lease governs this worker's sampling.

        When true, dispatch stages a job (its diffusion weights load at clearance, inside the leased sample
        call) and the parent's clearance controller admits each child into its load-and-sample window; when
        false the whole-job inference semaphore is the sole denoise gate and dispatch is the VRAM moment.
        """
        return bool(self._runtime_config.bridge_data.gpu_sampling_lease_enabled)

    def _max_jobs_in_progress_allowed(
        self,
        *,
        card: CardRuntime | None = None,
    ) -> int:
        """The cap on concurrently in-progress jobs for this scheduling decision.

        Without the GPU sampling lease, the inference semaphore is the sole denoise gate, so this
        is the concurrent-sampling count; dispatching more would over-subscribe the GPU. With the
        lease enabled, the lease (not this cap) limits actual concurrent sampling, so spare
        inference processes are allowed to receive jobs and stage their pipeline (model load,
        prompt encode) *ahead* while others sample, filling the inter-job gaps where the GPU
        would otherwise go dark. That pre-staging is permitted up to the full inference-process
        count, but only while there is enough free VRAM to hold another staged model; otherwise it
        falls back to the sampling-slot cap so speculation never over-commits the device.

        Args:
            card: When the worker drives more than one card, the card this decision is scoped to: its
                own sampling-slot and process ceilings are used so the big card's spare threads never
                inflate a small card's allowance. ``None`` keeps the worker-wide global ceilings, which
                is exactly the single-GPU case (byte-identical to before). The free-VRAM staging
                headroom is that card's own measured free; ``None`` reads the worker-wide (tightest-card)
                figure.
        """
        # An exclusive admit suppresses only its planned card once routing has attributed it. Before attribution
        # (and on the worker-wide single-GPU path), the conservative worker-wide answer still blocks every card.
        exclusive_device = card.device_index if card is not None else None
        if self._job_tracker.has_exclusive_job_in_progress(exclusive_device):
            in_scope = (
                len(self._jobs_in_progress_on_card(card.device_index))
                if card is not None
                else len(self._job_tracker.jobs_in_progress)
            )
            return max(1, in_scope)

        if card is not None:
            concurrent_ceiling = card.max_concurrent_inference
            process_ceiling = card.target_process_count
        else:
            concurrent_ceiling = self._max_concurrent_inference_processes
            process_ceiling = self._max_inference_processes

        base = concurrent_ceiling
        if not self._runtime_config.bridge_data.gpu_sampling_lease_enabled:
            return base

        # Under the clearance lease a staged job only charges its encode working set to the device until it
        # is cleared (the diffusion weights load inside the leased sample call, at clearance). Admit staging
        # while measured device free net of the reserve covers that encode charge, rather than gating on a
        # flat multi-GB device-free floor that mid-sampling almost never clears (which is what stranded spare
        # processes idle while jobs queued). The full materialisation is priced at clearance, so speculation
        # here never over-commits the device: it only funds the encode footprint the staging actually incurs.
        reserve_mb = self._vram_budget.reserve_mb if self._budget_active() else 0.0
        free_vram_mb = self._measured_free_vram_mb(device_index=card.device_index if card is not None else None)
        if free_vram_mb is None:
            self._note_staging_defer(
                StagingDeferReason.MEASUREMENT_UNREAD,
                headroom_mb=None,
                slot_cap=base,
                process_ceiling=process_ceiling,
            )
            return base
        headroom_mb = free_vram_mb - reserve_mb
        if headroom_mb < _STAGING_ENCODE_VRAM_MB:
            self._note_staging_defer(
                StagingDeferReason.ENCODE_HEADROOM_SHORT,
                headroom_mb=headroom_mb,
                slot_cap=base,
                process_ceiling=process_ceiling,
            )
            return base

        return process_ceiling

    def _note_staging_defer(
        self,
        reason: StagingDeferReason,
        *,
        headroom_mb: float | None,
        slot_cap: int,
        process_ceiling: int,
    ) -> None:
        """Tally a staging deferral and log it when the reason changes (or after a long unchanged run).

        The cap is consulted several times per scheduling pass, so the reason edge is what carries
        information; an unchanged run is restated only every :data:`_STAGING_DEFER_REPEAT_SECONDS` with the
        count it stood for. Emitted at debug, since holding staging back is ordinary backpressure.
        """
        self._staging_defers[reason] = self._staging_defers.get(reason, 0) + 1

        now = self._clock()
        suppressed = 0
        previous = self._staging_defer_log_state
        if previous is not None:
            previous_reason, previous_emit, previous_suppressed = previous
            if previous_reason is reason and (now - previous_emit) < _STAGING_DEFER_REPEAT_SECONDS:
                self._staging_defer_log_state = (previous_reason, previous_emit, previous_suppressed + 1)
                return
            suppressed = previous_suppressed
        self._staging_defer_log_state = (reason, now, 0)

        measured = "no device reading yet" if headroom_mb is None else f"{headroom_mb:.0f}MB free net of reserve"
        suffix = f" (suppressed {suppressed} unchanged repeats)" if suppressed > 0 else ""
        logger.debug(
            f"Holding the in-progress cap at {slot_cap} sampling slot(s) rather than pre-staging onto "
            f"{process_ceiling}: {reason.value}, {measured} against the {_STAGING_ENCODE_VRAM_MB:.0f}MB a "
            f"staged job's encode working set charges.{suffix}",
        )

    @property
    def staging_defer_counts(self) -> Mapping[StagingDeferReason, int]:
        """How many staging deferrals each measurement accounted for this session, keyed by the reason."""
        return self._staging_defers

    @property
    def retention_affinity_reorders(self) -> int:
        """How many jobs this session were seated ahead of the queue head onto weights a slot already retained.

        One per job served that way, counted where the dispatch commits, so it reads as uploads the placement
        order removed rather than as cycles it was consulted in. A session churning models with this at zero is
        not being reordered at all.
        """
        return self._retention_affinity_reorders

    @property
    def retention_grants_issued(self) -> int:
        """How many dispatches this session were granted VRAM retention (their weights left on the card)."""
        return self._retention_grants_issued

    @property
    def retention_grant_denials(self) -> Mapping[RetentionDenialReason, int]:
        """How many retention grants each gate refused this session, keyed by the gate that refused."""
        return self._retention_grant_denials

    @property
    def retention_reuses(self) -> int:
        """How many dispatches this session landed on a slot already retaining that job's model.

        One per job served without a weight upload, which is the whole of what retention buys. Read against
        :attr:`retention_evicted_unused`: the two partition every retention episode, so their ratio is what
        says whether the policy is paying for itself on this worker's traffic.
        """
        return self._retention_reuses

    @property
    def retention_evicted_unused(self) -> int:
        """How many retained copies this session were given back before any successor reused them.

        Counted where the scheduler itself returns the weights (a cross-model dispatch onto the retaining slot,
        a reclaim-ladder unload, an eager VRAM sweep, or the stale-hold revoke). A slot whose child died carries
        its residency out with the process and is not counted here, since the loss is the process, not the
        grant.
        """
        return self._retention_evicted_unused

    @property
    def retention_revokes(self) -> int:
        """How many retained copies this session the sustained-pressure sweep took back as stale.

        Each is a hold that predicted a same-model successor, went :data:`_RETENTION_STALE_HOLD_SECONDS`
        without one, and was occupying a card that had stayed off HEALTHY.
        """
        return self._retention_revokes

    def _record_slot_dispatch(self, process_id: int, model: str) -> None:
        """Record that ``model`` was dispatched to this slot, newest first, for later repeat-evidence reads."""
        history = self._slot_dispatch_history.get(process_id)
        if history is None:
            history = deque(maxlen=_RETENTION_REPEAT_EVIDENCE_DISPATCHES + 1)
            self._slot_dispatch_history[process_id] = history
        history.appendleft(model)

    def _slot_has_repeat_evidence(self, process_id: int, model: str, *, exclude_latest: bool = False) -> bool:
        """Whether this slot's trailing dispatches show ``model`` repeating, so retention on it is predicted.

        Args:
            process_id: The inference slot whose dispatch history is read.
            model: The model a grant is being decided (or re-decided) for.
            exclude_latest: Whether to skip the newest dispatch. A grant is decided before its own dispatch is
                recorded, so at issuance the newest entry is the previous job and the window is read as it
                stands. Once the grant is live its own dispatch heads the history, and skipping it is what
                makes the revoke sweep ask the identical question rather than a looser one that every fresh
                grant would answer for itself.
        """
        history = self._slot_dispatch_history.get(process_id)
        if history is None:
            return False
        window = list(history)[1:] if exclude_latest else list(history)
        return model in window[:_RETENTION_REPEAT_EVIDENCE_DISPATCHES]

    def _note_retention_denial(self, reason: RetentionDenialReason) -> None:
        """Tally a refused retention grant against the gate that refused it."""
        self._retention_grant_denials[reason] = self._retention_grant_denials.get(reason, 0) + 1

    def _stamp_retention_hold_ages(self) -> None:
        """Start the clock on any retention episode that has begun holding since the last tick.

        The settle that creates a retention runs on the completion path, which has no clock of the scheduler's;
        every other retention window is measured on this one, so the stamp is taken here instead. Idempotent
        by construction: an episode already stamped keeps its original start, and an episode that has ended
        carries no stamp forward, so a reuse is a new hold rather than an old one that got longer.
        """
        for process_info in self._process_map.values():
            if process_info.retained_resident_model is None:
                process_info.retained_resident_since = None
            elif process_info.retained_resident_since is None:
                process_info.retained_resident_since = self._clock()

    def _retention_hold_is_stale(self, process_info: HordeProcessInfo) -> bool:
        """Whether this slot's retained weights have gone unreused past the prediction's falsification horizon.

        An unstamped hold is never stale: it has not been observed for a full tick yet, and a hold whose age is
        unknown must not be revoked on an assumption about it.
        """
        held_since = process_info.retained_resident_since
        if held_since is None:
            return False
        return (self._clock() - held_since) >= _RETENTION_STALE_HOLD_SECONDS

    def _note_retention_evicted_unused(self, process_info: HordeProcessInfo) -> None:
        """Tally a retained copy about to be given back without any successor having reused it."""
        if process_info.retained_resident_model is not None:
            self._retention_evicted_unused += 1

    def _model_size_tier(self, model_name: str | None) -> _ModelSizeTier:
        """Classify a model by how much of the device its inference is expected to want.

        Resolves the model's baseline from the loaded reference and delegates to the shared, torch-free
        :func:`~horde_worker_regen.process_management.models.model_sizing.model_size_tier`, so this and the
        popper's large-model pop limiters classify "very large" from the same single source of truth.
        """
        baseline = self._model_metadata.get_baseline(model_name) if model_name is not None else None
        baseline_value = baseline.value if isinstance(baseline, KNOWN_IMAGE_GENERATION_BASELINE) else baseline
        return model_size_tier(model_name, baseline_value)

    @staticmethod
    def _job_batch_amount(job: ImageGenerateJobPopResponse) -> int:
        """The batch size (``n_iter``) of a job, floored at 1 for malformed values."""
        n_iter = job.payload.n_iter
        return n_iter if isinstance(n_iter, int) and n_iter > 0 else 1

    def _process_running_job(self, job: ImageGenerateJobPopResponse) -> HordeProcessInfo | None:
        """The inference process currently dispatched the given in-flight job, if any.

        Matches on typed execution ownership so a preload attribution cannot make the overlap gate read model
        preparation as a running job.
        """
        job_id = job.id_
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            referenced = process_info.current_inference_job()
            if referenced is not None and referenced.id_ == job_id:
                return process_info
        return None

    @staticmethod
    def _progress_fraction_of_process(process_info: HordeProcessInfo) -> float:
        """Denoise progress in ``[0.0, 1.0]`` for a process, from its last reported step or heartbeat.

        A process that has not yet reported a step reads as ``0.0`` (its progress fields are unset), which is
        exactly when a heavy overlap is most dangerous.
        """
        total_steps = process_info.last_total_steps
        current_step = process_info.last_current_step
        if total_steps is not None and total_steps > 0 and current_step is not None:
            return max(0.0, min(1.0, current_step / total_steps))

        percent_complete = process_info.last_heartbeat_percent_complete
        if percent_complete is not None:
            return max(0.0, min(1.0, percent_complete / 100.0))

        return 0.0

    def _remaining_sampling_seconds(self, process_info: HordeProcessInfo) -> float | None:
        """Estimated wall seconds left in a process's denoise loop, or None while no rate can be trusted.

        The rate is taken from data the child already reports: its step position and the timestamp of its
        first step, so the estimate is the job's own average step time extrapolated over the steps it has
        left. Withheld until the job is at least :data:`TAIL_OVERLAP_MIN_PROGRESS_FOR_ESTIMATE` through and
        has advanced past its first step, because a rate measured across one step is dominated by the one-off
        cost of entering the loop and would read far slower than the job really is.
        """
        total_steps = process_info.last_total_steps
        current_step = process_info.last_current_step
        first_step_at = process_info.current_first_step_at
        if total_steps is None or current_step is None or first_step_at is None or total_steps <= 0:
            return None
        if current_step >= total_steps:
            return 0.0
        if current_step < 2 or (current_step / total_steps) < TAIL_OVERLAP_MIN_PROGRESS_FOR_ESTIMATE:
            return None
        elapsed = self._clock() - first_step_at
        if elapsed <= 0.0:
            return None
        return (total_steps - current_step) * (elapsed / current_step)

    def _in_flight_progress_fraction(self, job: ImageGenerateJobPopResponse) -> float:
        """How far along the in-flight job's sampling is, in ``[0.0, 1.0]``.

        A freshly dispatched job that has not yet reported a step reads as ``0.0`` (the slot's progress
        fields are unset), which is exactly when a heavy overlap is most dangerous.
        """
        process_info = self._process_running_job(job)
        if process_info is None:
            return 0.0
        return self._progress_fraction_of_process(process_info)

    def build_clearance_inputs(self, *, device_index: int) -> ClearanceInputs:
        """Snapshot the per-tick truth the clearance controller reads for ``device_index``.

        A child that has staged and primed its next job (:attr:`HordeProcessState.INFERENCE_PRIMED`) is a
        clearance waiter: it holds the job and its encode-staging reservation and is waiting for the parent to
        clear it into its load-and-sample window. A child inside its denoise loop
        (:attr:`HordeProcessState.INFERENCE_STARTING`) is an active sampler. The controller owns each child's
        grant state and derives the held-slot count and tail-overlap sampler from these populations plus the
        measured free/reserve/paging truth. Each sampler additionally carries its estimated remaining sampling
        seconds and the card carries the measured median weight-upload cost, the two quantities the handoff
        window is sized from. Waiter priority is dispatch order (earlier-dispatched first), so queue position
        decides who samples next; the process id breaks ties deterministically.
        """
        waiters: list[ClearanceWaiter] = []
        samplers: list[ActiveSampler] = []
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if process_info.device_index != device_index:
                continue
            state = process_info.last_process_state
            referenced = process_info.current_inference_job()
            if state == HordeProcessState.INFERENCE_PRIMED:
                # Earlier dispatch time sorts closer to the head; a process with no stamp yet sorts last.
                dispatched_at = process_info.current_inference_started_at or float("inf")
                primed_job_id = str(referenced.id_) if referenced is not None and referenced.id_ is not None else None
                waiters.append(
                    ClearanceWaiter(
                        process_id=process_info.process_id,
                        priority=int(dispatched_at * 1000.0) if dispatched_at != float("inf") else 2**62,
                        job_id=primed_job_id,
                    ),
                )
            elif state == HordeProcessState.INFERENCE_STARTING:
                if referenced is None or referenced.id_ is None:
                    continue
                # A sampler re-bound inside the previous job's result tick still reports that job's active
                # state for a moment; read against the new ownership it would look like the new job sampling
                # with no grant, the reconciler would mark it sampling, and the child's real PRIMED that follows
                # would never be granted. An active state older than the ownership belongs to the previous job.
                ownership = process_info.inference_ownership
                if ownership is not None and ownership.recorded_at > process_info.last_process_state_started_at:
                    continue
                samplers.append(
                    ActiveSampler(
                        process_id=process_info.process_id,
                        job_id=str(referenced.id_),
                        progress_fraction=self._progress_fraction_of_process(process_info),
                        remaining_sampling_seconds=self._remaining_sampling_seconds(process_info),
                    ),
                )
        return ClearanceInputs(
            staged_waiters=tuple(waiters),
            active_grants=tuple(samplers),
            device_free_mb=self._measured_device_free_mb(device_index),
            vram_reserve_mb=self._vram_budget.reserve_mb,
            paging_active=self._wddm_paging_active,
            incoming_load_seconds=self._process_map.recent_vram_load_seconds(device_index),
        )

    def clearance_admit_process(self, process_id: int) -> bool:
        """Full-price fit-or-evict for a staged child's job at the clearance VRAM moment.

        The injected ``admit_fn`` the clearance controller calls before granting a child its load-and-sample
        window: it prices the child's primed job's full materialisation (weights plus activation peak) against
        measured device truth through the shared MONOLITHIC_DISPATCH admission core, running the single reclaim
        owner's eviction on a non-fit exactly as the dispatch residency gate does. On a fit it upgrades the
        job's dispatch reservation from the encode-only staging charge to the full peak (so a second grant sees
        it) and clears the clearance-hold record; on a non-fit it records the hold for slot-duty attribution and
        withholds the grant, so the child stays staged until eviction frees room (or degrades into unpriced
        sampling via hordelib's lease-acquire timeout, liveness over pricing). The co-residency mutex is applied
        here too, since clearance, not dispatch, is now the VRAM moment for a leased job.
        """
        process_info = self._process_map.get(process_id)
        if process_info is None:
            return False
        job = process_info.current_inference_job()
        if job is None or job.model is None:
            # Nothing priceable to hold on; let the child proceed rather than wedge it.
            return True

        # The co-residency mutex protection moves with the VRAM moment: a leased job's weights land at
        # clearance, so the post-processing chain fit is checked here (the dispatch-side check still guards the
        # non-lease path). A deferral withholds the grant without faulting the job.
        if self._should_defer_dispatch_for_post_processing(job, process_with_model=process_info):
            self._note_clearance_hold(job, reclaim_applied=False)
            self._log_clearance_hold(process_id, job, reason="post_processing_coresidency", verdict=None)
            return False

        if not self._budget_active():
            self._resolve_clearance_hold(job)
            return True

        outcome = self._evaluate_materialization_admission(
            job,
            process_info,
            is_head_of_queue=True,
            head_outstanding_mb=None,
            nets_own_dispatch_reservation=True,
        )
        if outcome.verdict.admits:
            self._resolve_clearance_hold(job)
            self._upgrade_dispatch_reservation_to_full(
                job,
                process_info,
                baseline=self._model_metadata.get_baseline(job.model),
            )
            return True

        self._note_clearance_hold(job, reclaim_applied=bool(outcome.actuations_applied))
        self._log_clearance_hold(process_id, job, reason=outcome.verdict.disposition.value, verdict=outcome.verdict)
        return False

    def _log_clearance_hold(
        self,
        process_id: int,
        job: ImageGenerateJobPopResponse,
        *,
        reason: str,
        verdict: VramVerdict | None,
    ) -> None:
        """Edge-log a clearance hold with its concrete denial arithmetic, coalesced per distinct cause.

        A persistent clearance hold starving a card is otherwise opaque: the slot-duty line names the bucket
        but not why admission refused this specific child. This names the disposition and the measured terms
        (device free, reserve, candidate charge, outstanding reservations) once per distinct ``(process, reason)``
        rather than every tick, so a held card's cause is always readable without per-tick spam.
        """
        suppressed = self._scheduler_diagnostic_suppressed_count("clearance_hold", (str(process_id), reason))
        if suppressed is None:
            return
        if verdict is not None:
            measured = verdict.measured
            free = "n/a" if measured.device_free_mb is None else f"{measured.device_free_mb:.0f}MB"
            avail = "n/a" if measured.available_mb is None else f"{measured.available_mb:.0f}MB"
            detail = (
                f"device free {free}, available {avail}, reserve {self._vram_budget.reserve_mb:.0f}MB, "
                f"outstanding reservations {measured.outstanding_reservations_mb:.0f}MB, "
                f"noise buffer {measured.noise_buffer_mb:.0f}MB"
            )
        else:
            detail = "post-processing co-residency mutex held the card for an in-flight or pending chain"
        logger.info(
            f"Clearance held for process {process_id} ({job.model}): {reason}; {detail}. The staged child waits "
            f"for room (or samples via its lease-acquire timeout); this repeats at most once per distinct cause.",
        )

    def _note_clearance_hold(self, job: ImageGenerateJobPopResponse, *, reclaim_applied: bool) -> None:
        """Record that ``job``'s clearance was withheld this pass, so the empty sampling slot is attributed.

        The clearance counterpart of :meth:`_note_dispatch_hold`, keyed on the primed in-progress job rather
        than a pending head. ``reclaim_applied`` (unused for now beyond symmetry) records that an actuator
        accepted at least one eviction command, matching the dispatch gate's release attribution.
        """
        job_id = str(job.id_) if job.id_ is not None else None
        if job_id is None:
            return
        self._clearance_hold_ids.add(job_id)

    def _resolve_clearance_hold(self, job: ImageGenerateJobPopResponse) -> None:
        """Clear any clearance hold on ``job`` now that its materialisation fits (idempotent)."""
        job_id = str(job.id_) if job.id_ is not None else None
        if job_id is None:
            return
        self._clearance_hold_ids.discard(job_id)

    @staticmethod
    def _required_overlap_headway(running_tier: _ModelSizeTier, candidate_tier: _ModelSizeTier) -> float:
        """Progress the running job must have made before a candidate joins it concurrently.

        Only called once both jobs are known to be non-extra-large and non-batched (those are hard
        blocks handled earlier). Two light jobs thread together freely; any pairing involving a heavy
        job requires headway, and two heavy jobs require the most.
        """
        if running_tier <= _ModelSizeTier.LIGHT and candidate_tier <= _ModelSizeTier.LIGHT:
            return 0.0
        if running_tier >= _ModelSizeTier.HEAVY and candidate_tier >= _ModelSizeTier.HEAVY:
            return _OVERLAP_HEADWAY_BOTH_HEAVY
        return _OVERLAP_HEADWAY_MIXED_HEAVY

    def _jobs_in_progress_on_card(self, device_index: int) -> list[ImageGenerateJobPopResponse]:
        """The in-progress jobs whose live inference process is pinned to ``device_index``.

        Cards are independent sampling/VRAM domains, so the per-card concurrency gates compare a candidate
        only against the jobs sharing its card. A job whose running slot cannot be identified is omitted (it
        is attributed to no card), which only ever relaxes the per-card count, never inflates it.
        """
        on_card: list[ImageGenerateJobPopResponse] = []
        for job in self._job_tracker.jobs_in_progress:
            running_process = self._process_running_job(job)
            if running_process is not None and running_process.device_index == device_index:
                on_card.append(job)
        return on_card

    def _expire_stale_model_map_entries(self) -> list[str]:
        """Expire model-map entries whose owning process can no longer be holding or loading that model.

        A slot holds one model at a time, so an entry naming a slot that now names a different model is a
        record of weights nothing holds. It is not inert: the preload pass reads the map as part of the
        already-loaded set, so a displaced model's surviving entry makes its pending job look served and the
        job is never staged onto a free slot. Reconciling the two parent-side records here keeps a displaced
        head loadable rather than permanently skipped.
        """
        expired: list[str] = []
        loading_owner_states = {
            HordeProcessState.PROCESS_STARTING,
            HordeProcessState.DOWNLOADING_MODEL,
            HordeProcessState.PRELOADING_MODEL,
            HordeProcessState.UNLOADED_MODEL_FROM_RAM,
        }

        now = self._clock()

        for model_name, model_info in list(self._horde_model_map.root.items()):
            process_info = self._process_map.get(model_info.process_id)
            if process_info is None:
                self._horde_model_map.expire_entry(model_name)
                expired.append(model_name)
                logger.warning(
                    f"Expiring stale model-map entry for {model_name}: process {model_info.process_id} is gone.",
                )
                continue

            recent_preload_request = (
                model_info.horde_model_load_state == ModelLoadState.LOADING
                and process_info.last_control_flag == HordeControlFlag.PRELOAD_MODEL
                and process_info.loaded_horde_model_name == model_name
                and (now - process_info.last_preload_requested_at) <= _PRELOAD_FIRST_REPORT_GRACE_SECONDS
            )
            if (
                model_info.horde_model_load_state == ModelLoadState.LOADING
                and process_info.last_process_state not in loading_owner_states
                and not recent_preload_request
            ):
                self._horde_model_map.expire_entry(model_name)
                expired.append(model_name)
                logger.warning(
                    f"Expiring stale loading entry for {model_name} on process {process_info.process_id}: "
                    f"process is {process_info.last_process_state.name}.",
                )
                continue

            if self._model_map_entry_is_displaced(model_name, process_info):
                self._horde_model_map.expire_entry(model_name)
                expired.append(model_name)
                logger.warning(
                    f"Expiring displaced entry for {model_name} on process {process_info.process_id}: "
                    f"the slot now holds {process_info.loaded_horde_model_name}.",
                )

        return expired

    @staticmethod
    def _model_map_entry_is_displaced(model_name: str, process_info: HordeProcessInfo) -> bool:
        """Whether ``model_name``'s map entry names a slot that has since been loaded over.

        A slot holds one model at a time, so a slot naming a different model has given these weights up. A
        slot naming nothing is not read as displaced: that is the transient the process map reports between a
        load being commanded and the slot being attributed, and expiring on it would discard a live load's
        record.
        """
        return process_info.loaded_horde_model_name is not None and (
            process_info.loaded_horde_model_name != model_name
        )

    def _replace_stale_ram_unload_process(self, *, protect_process_id: int | None = None) -> bool:
        """Cycle an idle inference process to return retained RAM to the OS; return whether one was cycled.

        Two triggers, in priority order:

        - Creep containment: any idle inference slot whose RSS exceeds :data:`_CREEP_CONTAINMENT_RSS_BYTES` is
          cycled regardless of its unload state or resident model. A child creeps ~400MB/job and cycling is the
          only containment; RSS this high is leak, not clean reusable pages, so this override even cycles the
          protected staging target (reusing a crept slot would only perpetuate the leak).
        - Stale-unload reclaim (the RAM-verdict last resort): a model-less idle slot that did not actually
          release RAM after an ``UNLOAD_MODELS_FROM_RAM`` request. ``protect_process_id`` spares one slot here,
          the current head's staging target, so the reclaim does not destroy the retained pages the marginal
          RAM credit priced the head's preload against (which would force the very cold load the credit avoids).

        Creep victims are preferred over stale victims so an unbounded leak is contained ahead of an ordinary
        retained-page reclaim.
        """
        creep_victim: HordeProcessInfo | None = None
        stale_victim: HordeProcessInfo | None = None
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if process_info.is_process_busy():
                continue
            # A slot the admission pipeline just routed a preload onto is mid-stage: reaping it would fault the
            # head's load. Spared from both triggers, exactly as the stale-unload path spares a non-UNLOAD flag.
            if process_info.last_control_flag == HordeControlFlag.PRELOAD_MODEL:
                continue
            if creep_victim is None and process_info.ram_usage_bytes >= _CREEP_CONTAINMENT_RSS_BYTES:
                creep_victim = process_info
                continue
            if stale_victim is not None:
                continue
            if process_info.process_id == protect_process_id:
                continue
            if process_info.loaded_horde_model_name is not None:
                continue
            if process_info.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
                continue
            if process_info.ram_usage_bytes < _STALE_RAM_UNLOAD_REPLACE_BYTES:
                continue
            stale_victim = process_info

        victim = creep_victim if creep_victim is not None else stale_victim
        if victim is None:
            return False

        if creep_victim is not None:
            logger.warning(
                f"Idle process {victim.process_id} holds {victim.ram_usage_bytes} bytes "
                "(above the creep-containment ceiling); cycling it to return the crept RAM to the OS.",
            )
        else:
            logger.warning(
                f"Idle process {victim.process_id} still holds {victim.ram_usage_bytes} bytes "
                "after a RAM unload (the allocator retains the freed model's pages); cycling it to return "
                "the RAM to the OS.",
            )
        # A deliberate reclaim of a healthy idle slot, not a crash/hang: keep it out of the crash
        # bookkeeping (recovery count + crash-loop breaker) so sustained RAM pressure cannot
        # quarantine a perfectly healthy slot.
        self._process_lifecycle._replace_inference_process(victim, intentional_reclaim=True)
        # A cycled slot's pending reuse credit is void: its successor cold-loads, so the recorded swap will
        # never settle and must not be reconciled against the fresh process.
        self._pending_reuse_credits.pop(victim.process_id, None)
        # Open the bounded reclaim-cycle grace: the slot now respawns and the next head must preload
        # onto it, a window in which the queue is unservable by the worker's own deliberate action, not
        # a wedge. ram_reclaim_cycle_grace_active() reads this so the recovery supervisor does not
        # soft-reset the pools and fault the servable backlog mid-reclaim.
        self._ram_reclaim_cycle_at = self._clock()
        self._record_churn("process_cycle")
        return True

    def _preload_blocked_by_ram_pressure(self, job: ImageGenerateJobPopResponse) -> bool:
        """Return whether the host's absolute RAM danger floor forces this preload to defer.

        When system RAM is below its danger floor, governs the pressure (sheds idle footprint, pauses
        pops) and reports True so the caller defers rather than routing a new model's weights through a
        host already on the edge. Clears the one-shot notice and reports False when RAM is healthy.
        """
        # One scheduling cycle acts on one consistent reading: reuse the verdict the governor's tick
        # measured at the top of this cycle rather than re-measuring per job. Tests (and any path that
        # reaches here before a first tick) fall back to a live reading.
        ram_pressure = self._governor.last_ram_verdict
        if ram_pressure is None:
            ram_pressure = self._ram_pressure_verdict()
        if not ram_pressure.under_pressure:
            self._ram_pressure_notified = False
            return False
        # The governor's tick (run once per control-loop iteration via run_governance_tick) has already
        # driven the whole-host degrade response this cycle; here we only defer *this* preload and surface
        # the per-model notice once so the loop does not route a new model's weights through a host already
        # on the edge.
        if not self._ram_pressure_notified:
            logger.opt(colors=True).warning(
                "<fg #ff8c69>RAM danger floor reached: deferring preload of {} "
                f"({ram_pressure.reason()}). Shedding idle footprint and pausing pops.</>",
                job.model,
            )
            self._ram_pressure_notified = True
        return True

    def _send_preload(self, job: ImageGenerateJobPopResponse, available_process: HordeProcessInfo) -> bool:
        """Send the preload command for ``job``'s model to ``available_process`` and record the load.

        Resets the preload-delay and head-starvation trackers, sends the PRELOAD_MODEL message inside a
        telemetry span, and on a successful send records the churn/ledger entry and advances the model map
        and process map into the LOADING state. Returns True (a preload was issued this cycle).
        """
        if job.model is None:
            raise ValueError(f"job.model is None ({job})")

        self._preload_delay_notified = False
        self._clear_head_starvation_timer()
        logger.debug(f"Preloading model {job.model} on process {available_process.process_id}")
        logger.debug(f"Available inference processes: {self._process_map}")
        only_active_models = {
            model_name: model_info
            for model_name, model_info in self._horde_model_map.root.items()
            if model_info.horde_model_load_state.is_active()
        }
        logger.debug(f"Horde model map (active): {only_active_models}")

        will_load_loras = job.payload.loras is not None and len(job.payload.loras) > 0
        seamless_tiling_enabled = job.payload.tiling is not None and job.payload.tiling

        # A swap is a preload that displaces a *different* model already resident on this process;
        # that prior model's load work is thrown away. A fresh slot (None) or re-preload of the same
        # model is not churn. Captured before the send so the process's prior model is still readable.
        prior_model = available_process.loaded_horde_model_name
        is_model_swap = prior_model is not None and prior_model != job.model

        with span_preload_model(model_name=job.model, process_id=available_process.process_id):
            preload_sent = available_process.safe_send_message(
                HordePreloadInferenceModelMessage(
                    control_flag=HordeControlFlag.PRELOAD_MODEL,
                    horde_model_name=job.model,
                    will_load_loras=will_load_loras,
                    seamless_tiling_enabled=seamless_tiling_enabled,
                    sdk_api_job_info=job,
                    diffusion_model_only=self._is_disaggregation_class_eligible(job),
                ),
            )

        if preload_sent:
            available_process.last_control_flag = HordeControlFlag.PRELOAD_MODEL
            available_process.last_preload_requested_at = time.time()
            if is_model_swap:
                self._record_churn("model_swap")
            self._process_lifecycle.action_ledger.record(
                LedgerEventType.PRELOAD_REQUESTED,
                process_id=available_process.process_id,
                os_pid=available_process.os_pid,
                launch_identifier=available_process.process_launch_identifier,
                job_id=str(job.id_) if job.id_ is not None else None,
                detail={"model": job.model},
            )

            self._horde_model_map.update_entry(
                horde_model_name=job.model,
                load_state=ModelLoadState.LOADING,
                process_id=available_process.process_id,
            )

            model_baseline = self._model_metadata.get_baseline(job.model)

            self._process_map.on_model_load_state_change(
                process_id=available_process.process_id,
                horde_model_name=job.model,
                horde_model_baseline=model_baseline,
                last_job_referenced=job,
            )

            # Record the grant into the planned overlay the moment the load is admitted, so a second admission
            # in this same scheduling cycle (before the per-cycle reconcile runs) sees this charge and cannot
            # over-admit against the same measured floor. The charge is the candidate delta actually priced for
            # this preload (sampler-only or whole-job, net of any resident credit); the admit-time reservation
            # baseline is the target's reserved right now, so the charge decays one-for-one as this process's
            # own reservation materialises the load. The per-cycle reconcile then prunes it once the process
            # leaves the loading set (finished, faulted, or dead), with no explicit release on those paths.
            planned_charge_mb = self._preload_candidate_delta_mb(
                job,
                model_baseline,
                process_id=available_process.process_id,
            )
            self._reserve_ledger.set_planned(
                PRELOAD_ADMISSION_FLOW,
                str(available_process.process_id),
                vram_mb=planned_charge_mb if planned_charge_mb is not None else 0.0,
                target_process_id=available_process.process_id,
                reserved_at_admit_mb=float(available_process.process_reserved_mb or 0),
            )

        return True

    def _whole_card_governor_hold(self, device_index: int | None, *, now: float) -> _WholeCardGovernorHold | None:
        """Return the churn governor barring a *new* whole-card residency on this card, or None when free.

        Both governors act at admission and only on a card about to take a model on: a residency already held
        keeps converging, and a restore always gets its window. The rate limiter is answered first because it
        is the cheaper and shorter-lived of the two.
        """
        if self._whole_card_ledger.establish_rate_exceeded(device_index, now=now):
            return _WholeCardGovernorHold(
                governor=WholeCardGovernor.ESTABLISH_RATE,
                reason="this card has already cycled whole-card residency as often as the rolling window allows",
            )
        if self._whole_card_ledger.grace_budget_exhausted(device_index, now=now):
            status = self._whole_card_ledger.grace_budget_status(device_index, now=now)
            return _WholeCardGovernorHold(
                governor=WholeCardGovernor.GRACE_BUDGET,
                reason=(
                    "this card has spent its rolling recovery-supervisor grace allowance, so it may not open "
                    "another establish window yet"
                ),
                detail=(
                    f"{status.spent_seconds:.0f}s of grace opened against a "
                    f"{status.allowance_seconds:.0f}s rolling allowance, {status.remaining_seconds:.0f}s left, "
                    f"next replenish in {status.replenish_in_seconds:.0f}s"
                ),
            )
        return None

    def _disclose_whole_card_governor_hold(
        self,
        job: ImageGenerateJobPopResponse,
        hold: _WholeCardGovernorHold,
        *,
        device_index: int | None,
        elapsed: float,
        downgraded: bool,
    ) -> None:
        """Disclose a governor holding this head off the card, and the downgrade when the dwell is spent.

        A hold that persists is re-stated periodically rather than announced once: the operator needs the
        current spend and the wait, not a single line from whenever the hold began. Repeats inside the
        diagnostic cadence are counted and left at TRACE so a tick-rate loop cannot flood the log.
        """
        name = "whole_card_governor_downgrade" if downgraded else "whole_card_governor_hold"
        suppressed = self._scheduler_diagnostic_suppressed_count(name, (str(job.model), device_index, hold.governor))
        stated = hold.reason if hold.detail is None else f"{hold.reason} ({hold.detail})"
        if downgraded:
            message = (
                f"Whole-card residency downgraded for {job.model}: {stated}. It has asked for the card for "
                f"{elapsed:.0f}s, so the ask is dropped and ordinary admission decides; if the device holds its "
                "weights it runs co-resident (slower than sole residency) instead of waiting for the allowance."
            )
        else:
            message = (
                f"Deferring a new whole-card residency for {job.model}: {stated}. Normal scheduling "
                f"continues; after {self._whole_card_ledger.governor_defer_dwell_seconds:.0f}s held (currently "
                f"{elapsed:.0f}s) the head stops asking for the card and is served co-resident if the device "
                "can hold it."
            )
        if suppressed is None:
            logger.trace(message)
            return
        logger.warning(f"{message}{self._suppressed_suffix(suppressed)}")

    def _decide_whole_card_demand(
        self,
        job: ImageGenerateJobPopResponse,
        available_process: HordeProcessInfo,
        forecast: StreamForecast,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        *,
        is_head_blocker: bool,
        target_device_index: int | None,
    ) -> _WholeCardDemandOutcome:
        """Decide whether the head should claim the whole card and drive the residency side effects.

        Whole-card exclusive residency (preventative): the forecast judges whether loading this model
        alongside the currently-resident models would drive the device into weight streaming. A heavy model
        loaded while others stay resident across sibling processes can collapse free VRAM to near zero, at
        which point ComfyUI offloads weights or the driver's system-memory fallback spills per-step
        activations; both stream over the bus and the slow job risks being mistaken for a hang and killed.
        When the model would stream co-resident but fits with the card to itself, it is given sole residency
        before it loads: marked exclusive (so ``has_exclusive_job_in_progress`` suppresses other staging),
        then enough VRAM is freed. The forecast distinguishes two remedies, applying the least-disruptive:
        evicting sibling *models* (their processes stay up), or stopping idle sibling *processes* when their
        fixed per-process contexts are themselves the over-commit (a context is only reclaimed by exit).

        Only the head may claim the card: reserving it tears down the siblings serving the lighter heads
        ahead of a deeper-queue job, so a non-head heavy job returns ``FALL_THROUGH`` and defers via the
        ordinary verdict until it becomes the head. See :class:`_WholeCardDemandOutcome` for each result.
        """
        # A disaggregation-class job never demands exclusive device residency: it runs as a UNet-only sampler
        # whose sampler-only footprint co-resides with the encode lane and other samplers by design. Coupled
        # with sampler-only charging in the forecast, this breaks the loop where a whole-card window (which
        # pauses the lane) would otherwise flip the job to its monolithic footprint and re-demand the card,
        # starving the encode lane. Decided on class-eligibility, not liveness, so the contract holds even
        # while the lane is transiently paused.
        if self._is_disaggregation_class_eligible(job):
            return _WholeCardDemandOutcome.FALL_THROUGH
        # A model needs the teardown path either because it is weight-dominant (needs sole residency) or
        # because the live sibling process contexts have squeezed its bounded weights off the card though it
        # co-resides once the process count is reduced. Both are served by the same machinery: establish
        # residency, stop idle siblings down to max_resident_processes, and admit once the weights fit.
        live_inference_processes = self._process_map.num_loaded_inference_processes(
            device_index=target_device_index,
        )
        whole_card_demanded = self._whole_card_ledger.residency_demanded(
            forecast,
            enabled=self._whole_card_residency_enabled(),
            is_head_blocker=is_head_blocker,
            live_inference_process_count=live_inference_processes,
        )
        if not whole_card_demanded:
            # Only the head could have claimed the card, so only the head's refusal is worth naming: a
            # deeper-queue job reads the same forecast every tick and never had a claim to lose.
            if is_head_blocker and forecast.needs_process_count_reduction:
                self._log_whole_card_reduction_suppressed(
                    job,
                    forecast,
                    live_inference_processes=live_inference_processes,
                )
            return _WholeCardDemandOutcome.FALL_THROUGH
        if not self._whole_card_warranted(forecast):
            # The teardown demand is not trustworthy (a card-light model on a host with no measured
            # per-context cost): decline the reservation and fall through to ordinary eviction rather than
            # reserving the device on an over-counted-context phantom.
            self._log_whole_card_declined(job, forecast)
            return _WholeCardDemandOutcome.FALL_THROUGH

        held = self._whole_card_ledger.get(target_device_index)
        establishing_anew = held is None or held.model is None
        governor_hold = (
            self._whole_card_governor_hold(target_device_index, now=self._clock()) if establishing_anew else None
        )
        if governor_hold is not None:
            # A governor brakes how fast this card may be rotated; it never says the head cannot be served.
            # Hold the whole-card ask for a bounded dwell (the head re-asks every cycle), then stop preferring
            # the card and let the measured arbiter decide: a card that can demonstrably hold the weights
            # serves the head co-resident rather than standing idle behind a brake with no fallback.
            elapsed, dwell_exhausted = self._whole_card_ledger.note_governor_defer(
                target_device_index,
                model=job.model,
                now=self._clock(),
            )
            self._disclose_whole_card_governor_hold(
                job,
                governor_hold,
                device_index=target_device_index,
                elapsed=elapsed,
                downgraded=dwell_exhausted,
            )
            if dwell_exhausted:
                return _WholeCardDemandOutcome.FALL_THROUGH
            self._last_budget_defer_reason = governor_hold.reason
            return _WholeCardDemandOutcome.DEFER
        self._whole_card_ledger.clear_governor_defer(target_device_index)

        first_time = not self._job_tracker.is_admitted_exclusive(job)
        self._job_tracker.mark_admitted_exclusive(job, device_index=target_device_index)
        if self._should_prestage_whole_card_head(
            job,
            baseline,
            forecast,
            available_process,
            device_index=target_device_index,
        ):
            # A live job still holds the device, but the heavy head's weights can begin loading into a spare
            # process's RAM right now: preload_model is a RAM-only load (weights move to VRAM at sampling
            # time), so it does not contend with the in-flight job's VRAM. Record the residency and send the
            # preload; _converge_whole_card_residency then collapses the live process count to sole VRAM
            # residency before the staged model samples. The heavy disk->RAM load overlaps the in-flight job
            # instead of waiting for the device to drain first.
            self._begin_whole_card_residency(
                job,
                forecast,
                announce=first_time,
                device_index=target_device_index,
                target_process=available_process,
            )
            return _WholeCardDemandOutcome.PRESTAGE

        # Claim the device: stop idle siblings to the model's max-resident count and, on the very edge, move
        # safety off-GPU too. Announces (once) why, for the operator. Held through the cooldown so a burst of
        # heavy jobs reuses one residency instead of churning per job.
        self._establish_whole_card_residency(
            job,
            forecast,
            announce=first_time,
            device_index=target_device_index,
            target_process=available_process,
        )
        # Evict the idle resident models on the *other* processes (sparing the slot that will load this
        # model, and never a live in-progress model) so their VRAM returns to the driver. A live sibling is
        # left to drain; the preload simply waits until the device is clear.
        self.unload_models_from_vram(
            available_process,
            under_pressure=True,
            for_head_of_queue=True,
            device_index=target_device_index,
        )
        if not self._whole_card_teardown_exhausted(forecast, device_index=target_device_index):
            # Still tearing down idle siblings, cycling safety off-GPU, or waiting for their freed VRAM to
            # drain: defer and let a later tick re-evaluate against the reduced topology.
            return _WholeCardDemandOutcome.DEFER
        # Teardown is structurally exhausted (already at the target process count, safety settled). The card is
        # now cleared to sole residency, so the head's weights are priced against a drained card: fall through
        # to the measured arbiter evaluation, which admits when the weights fit the cleared card (the
        # activation peak is the sampling gate's concern, not preload admission) and denies when even the
        # cleared card cannot hold them (an unserviceable model the offering seam should have excluded).
        return _WholeCardDemandOutcome.FALL_THROUGH

    def _reclaim_ram_for_overbudget_admit(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
    ) -> None:
        """Reclaim idle system RAM ahead of an over-budget classified load, only when the host is short.

        A heavy head loads its checkpoint through system RAM before it reaches the device, so an admit on
        a RAM-tight host must first evict an idle resident copy. On a host with ample available RAM that
        eviction buys nothing and costs a warm cache: the sibling's model drops to disk, and the next job
        for it pays a full checkpoint reload (with the allocator-stuck slot the unload leaves behind then
        recycled, compounding the churn). So the reclaim is gated on the RAM budget's own verdict for the
        incoming load rather than performed unconditionally.
        """
        ram_verdict = self._ram_budget.check_job(
            job,
            baseline,
            self._measured_available_ram_mb(),
            committed_reserve_mb=self._reserve_ledger.total_ram_mb(),
        )
        if ram_verdict.fits:
            return
        self.unload_models(under_pressure=True, for_head_of_queue=True)

    def _staging_reuse_credit_mb(self, target: HordeProcessInfo) -> float:
        """The retained, reusable resident RSS (MB) a preload onto ``target`` can reuse instead of allocating.

        An idle inference slot keeps the freed model's pages resident after an unload (its RSS stays above a
        fresh child's baseline); a preload onto it reuses those pages, so its real system-RAM growth is a
        fraction of a cold load. This returns the retained excess over :data:`_FRESH_INFERENCE_CHILD_BASELINE_MB`
        that the marginal RAM credit is computed from. A busy target's pages are in live use and are not
        reusable, so it earns no credit; a fresh slot (RSS at baseline) yields zero, collapsing the verdict to
        the ordinary full charge.
        """
        if target.is_process_busy():
            return 0.0
        target_rss_mb = max(0, target.ram_usage_bytes) / (1024 * 1024)
        return max(0.0, target_rss_mb - _FRESH_INFERENCE_CHILD_BASELINE_MB)

    def _ram_danger_floor_mb(self) -> float:
        """The absolute available-RAM danger floor (MB) below which the host must degrade rather than load.

        The same floor the whole-host RAM-pressure governor uses, read here so the marginal RAM credit never
        credits a preload into a floor breach (see :meth:`_apply_ram_verdict`).
        """
        pause_pct, min_free_mb = self._ram_pressure_floor_config()
        return ram_pressure_floor_mb(
            self._measured_total_ram_mb(),
            pause_percent=pause_pct,
            min_free_mb=min_free_mb,
        )

    def _note_credited_admission(
        self,
        job: ImageGenerateJobPopResponse,
        target: HordeProcessInfo,
        verdict: BudgetVerdict,
    ) -> None:
        """Log a credited RAM admission (edge-triggered) and record it for the measured-truth reconciliation.

        Emitted once per distinct credited admission (target, model, rounded charge) so the sub-second loop
        cannot spam an unchanged decision. The record lets :meth:`_reconcile_reuse_credit` later compare the
        target's settled RSS growth to the charge the credit priced the swap at.
        """
        retained_mb = self._staging_reuse_credit_mb(target)
        charge_key = int(verdict.predicted_mb) if verdict.predicted_mb is not None else 0
        key = (target.process_id, job.model, charge_key)
        if key != self._last_credited_admission_key:
            uncredited = verdict.uncredited_predicted_mb if verdict.uncredited_predicted_mb is not None else 0.0
            logger.opt(colors=True).info(
                "<fg #8fd6a0>RAM credit admitting preload of {} "
                f"onto process {target.process_id}: "
                f"predicted ~{uncredited:.0f} MB, credit {verdict.reusable_credit_mb:.0f} MB "
                f"(retained reusable {retained_mb:.0f} MB), effective charge ~{charge_key:.0f} MB.</>",
                job.model,
            )
            self._last_credited_admission_key = key
        if job.model is not None:
            self._pending_reuse_credits[target.process_id] = _ReuseCreditRecord(
                model=job.model,
                rss_at_admit_mb=max(0, target.ram_usage_bytes) / (1024 * 1024),
                effective_charge_mb=verdict.predicted_mb if verdict.predicted_mb is not None else 0.0,
                admitted_at=self._clock(),
            )

    def _note_component_admission(
        self,
        job: ImageGenerateJobPopResponse,
        target: HordeProcessInfo,
        verdict: BudgetVerdict,
    ) -> None:
        """Log a UNet-only component admission (edge-triggered) and record it for the measured-truth check.

        The disaggregation-class analogue of :meth:`_note_credited_admission`: emitted once per distinct
        (target, model, rounded charge) so the sub-second loop cannot spam it, and recorded so
        :meth:`_reconcile_reuse_credit` can compare the target's settled RSS growth to the component charge the
        stage was priced at. The record carries :data:`_REUSE_CREDIT_KIND_COMPONENT` so the reconciliation
        wording reflects a UNet-only charge rather than a retained-page reuse credit.
        """
        charge_key = int(verdict.predicted_mb) if verdict.predicted_mb is not None else 0
        key = (target.process_id, job.model, charge_key)
        if key != self._last_credited_admission_key:
            whole = verdict.uncredited_predicted_mb if verdict.uncredited_predicted_mb is not None else 0.0
            logger.opt(colors=True).info(
                "<fg #8fd6a0>RAM UNet-only charge admitting preload of {} onto process "
                f"{target.process_id}: whole-checkpoint ~{whole:.0f} MB, component charge ~{charge_key:.0f} MB "
                "(disaggregation-class sampler stages the UNet only).</>",
                job.model,
            )
            self._last_credited_admission_key = key
        if job.model is not None:
            self._pending_reuse_credits[target.process_id] = _ReuseCreditRecord(
                model=job.model,
                rss_at_admit_mb=max(0, target.ram_usage_bytes) / (1024 * 1024),
                effective_charge_mb=verdict.predicted_mb if verdict.predicted_mb is not None else 0.0,
                admitted_at=self._clock(),
                kind=_REUSE_CREDIT_KIND_COMPONENT,
            )

    def _disaggregated_component_charge_mb(
        self,
        job: ImageGenerateJobPopResponse,
        target: HordeProcessInfo,
    ) -> float | None:
        """Return the UNet-only RAM staging charge (MB) for a disaggregation-class ``job``, or None.

        None means "price the whole checkpoint" and is returned whenever the component charge does not apply:
        the job is not disaggregation-class, it has no model, or no component-identity sidecar (hence no UNet
        residual) can be read for its checkpoint. A readable sidecar yields the floored UNet residual charge
        (:func:`predict_job_unet_only_ram_mb`), except that a checkpoint whose identity the residency map shows
        already staged on ``target`` is charged 0.0: its pages are resident, so the stage materialises nothing
        (the RAM analogue of the resident-weight credit the VRAM candidate delta applies). The class predicate
        is the stable one the VRAM side charges against, so RAM and VRAM stay class-consistent within a cycle.
        """
        if not self._is_disaggregation_class_eligible(job):
            return None
        if job.model is None:
            return None
        sidecar = self._read_component_sidecar(job.model)
        if sidecar is None:
            return None
        if self._checkpoint_identity_held_on(job.model, target.process_id):
            return 0.0
        return predict_job_unet_only_ram_mb(sidecar.residual_tensor_bytes)

    def _checkpoint_identity_held_on(self, model_name: str, process_id: int) -> bool:
        """Whether ``model_name``'s checkpoint is already staged in ``process_id``'s RAM component cache.

        A checkpoint entry's residency identity is the bare horde model name, so this is a direct membership
        test against the residency map. False when no residency map is wired (unit tests, or a worker whose
        budgeted component cache is disabled), so the charge then defaults to the full UNet residual.
        """
        if self._component_residency_map is None:
            return False
        return model_name in self._component_residency_map.checkpoint_models_held_on([process_id])

    def _read_component_sidecar(self, model_name: str) -> ComponentIdentitySidecar | None:
        """Return ``model_name``'s component-identity sidecar (torch-free, cached), or None when unavailable.

        Resolves the checkpoint path once per model (see :meth:`_resolve_checkpoint_path`) and reads the sidecar
        beside it. The parsed sidecar is cached keyed on the checkpoint's on-disk size, so an unchanged
        checkpoint is not re-parsed every sub-second cycle while a replaced one (a different size) re-reads;
        this leans on the sidecar's own ``ckpt_size_bytes`` staleness check, which
        :func:`horde_model_reference.component_identity.read_sidecar` applies on a fresh read. A missing path,
        an unstattable checkpoint, or an absent/malformed/stale sidecar returns None (the whole-checkpoint
        fallback) and logs a debug notice once per model so the loop cannot spam it.
        """
        ckpt_path = self._resolve_checkpoint_path(model_name)
        if ckpt_path is None:
            self._log_component_charge_fallback_once(model_name, "no checkpoint path resolved")
            return None
        try:
            current_size = ckpt_path.stat().st_size
        except OSError:
            self._log_component_charge_fallback_once(model_name, "checkpoint not on disk")
            return None
        cached = self._component_sidecar_cache.get(model_name)
        if cached is not None and cached[0] == current_size:
            return cached[1]

        from horde_model_reference.component_identity import read_sidecar

        sidecar = read_sidecar(ckpt_path)
        if sidecar is None:
            self._log_component_charge_fallback_once(model_name, "no component-identity sidecar")
            return None
        self._component_sidecar_cache[model_name] = (current_size, sidecar)
        return sidecar

    def _resolve_checkpoint_path(self, model_name: str) -> Path | None:
        """Resolve ``model_name``'s on-disk monolithic checkpoint path (torch-free, cached), or None.

        The path is stable once the reference is loaded, so it is resolved once and cached per model. Resolution
        uses the same on-disk layout authority the download path does
        (:func:`horde_model_reference.on_disk_layout.file_paths_for`) over the env-resolved weights root, picking
        the checkpoint file (the download entry that is not routed to a sibling component folder). None is
        returned and cached when no reference is loaded, the model has no record, or resolution raises, so the
        caller falls back to the whole-checkpoint charge.
        """
        if model_name in self._checkpoint_path_cache:
            return self._checkpoint_path_cache[model_name]
        path = self._compute_checkpoint_path(model_name)
        self._checkpoint_path_cache[model_name] = path
        return path

    def _compute_checkpoint_path(self, model_name: str) -> Path | None:
        """Resolve ``model_name``'s checkpoint path from the loaded reference, or None (see caller)."""
        reference = self._model_metadata.reference
        if reference is None:
            return None
        record = reference.get(model_name)
        if record is None:
            return None
        try:
            from horde_model_reference.on_disk_layout import resolve_weights_root

            from horde_worker_regen.model_download_plan import primary_checkpoint_path_for

            if self._weights_root is None:
                self._weights_root = resolve_weights_root()
            return primary_checkpoint_path_for(record, self._weights_root)
        except Exception as e:  # noqa: BLE001 - path resolution is best-effort; any failure fails safe to None.
            logger.debug(f"Checkpoint path resolution for {model_name!r} failed: {type(e).__name__}: {e}")
            return None

    def _log_component_charge_fallback_once(self, model_name: str, reason: str) -> None:
        """Log the whole-checkpoint fallback for a disaggregation-class model once, keyed by model name."""
        if model_name in self._component_charge_fallback_logged:
            return
        self._component_charge_fallback_logged.add(model_name)
        logger.debug(
            f"UNet-only RAM charge unavailable for {model_name!r} ({reason}); charging the whole checkpoint.",
        )

    def _reconcile_reuse_credit(self) -> None:
        """Compare each settled credited admission's real RSS growth to its charge (the measured-truth check).

        Once a credited target has settled past :data:`_REUSE_CREDIT_RECONCILE_SETTLE_SECONDS` with the staged
        model resident, its measured RSS growth is known. Growth exceeding the charge by more than
        :data:`_REUSE_CREDIT_RECONCILE_SLACK_MB` means the charge under-priced the load; log it once so the
        constants can be retuned. Growth below the charge is expected and silent: a page-reuse credit assumes a
        retaining slot, and a UNet-only component charge assumes mmap page sharing keeps a reload cheap, so
        settling below the charge is the intended outcome, not a discrepancy. The wording distinguishes the two
        record kinds; the threshold is identical. Records for vanished or re-tasked slots are dropped so the map
        does not accumulate.
        """
        now = self._clock()
        for process_id, record in list(self._pending_reuse_credits.items()):
            process_info = self._process_map.get(process_id)
            if process_info is None:
                del self._pending_reuse_credits[process_id]
                continue
            settled = (
                now - record.admitted_at >= _REUSE_CREDIT_RECONCILE_SETTLE_SECONDS
                and process_info.loaded_horde_model_name == record.model
                and not process_info.is_process_busy()
            )
            if not settled:
                continue
            del self._pending_reuse_credits[process_id]
            growth_mb = max(0, process_info.ram_usage_bytes) / (1024 * 1024) - record.rss_at_admit_mb
            if growth_mb <= record.effective_charge_mb + _REUSE_CREDIT_RECONCILE_SLACK_MB:
                continue
            if record.kind == _REUSE_CREDIT_KIND_COMPONENT:
                logger.opt(colors=True).warning(
                    "<fg #f0beff>UNet-only component charge for {} "
                    f"on process {process_id} "
                    f"under-priced the stage: measured RSS grew ~{growth_mb:.0f} MB against a component charge of "
                    f"~{record.effective_charge_mb:.0f} MB; the reload shared fewer pages than the residual "
                    "implied.</>",
                    record.model,
                )
            else:
                logger.opt(colors=True).warning(
                    "<fg #f0beff>RAM reuse credit for {} "
                    f"on process {process_id} was too generous: "
                    f"measured RSS grew ~{growth_mb:.0f} MB against an effective charge of "
                    f"~{record.effective_charge_mb:.0f} MB; the retained pages were less reusable than priced.</>",
                    record.model,
                )

    def _apply_ram_verdict(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        available_process: HordeProcessInfo,
        *,
        is_head_blocker: bool,
        no_live_resource_consumer: bool,
    ) -> bool:
        """Apply the system-RAM budget verdict for a preload: reclaim idle RAM or best-effort admit.

        When the predicted RAM cost fits, returns True immediately. The charge is credited for a reusable
        staging target: an idle ``available_process`` that retains its unloaded model's pages (or swaps a model
        in place) reuses those pages, so the verdict prices the swap at its marginal growth rather than a full
        cold load (see :meth:`_staging_reuse_credit_mb` and :meth:`RamBudget.check_job`). The credit is gated on
        the host RAM danger floor so it never admits into a floor breach.

        A disaggregation-class job (the same class predicate the VRAM side charges sampler-only against) is
        instead priced at its UNet-only component charge: its sampler stages only the UNet, so the whole-
        checkpoint charge would over-count by the text encoders and VAE that never enter this process. The
        component charge supersedes the page-reuse credit (they are alternative marginal accountings of the
        same load). When no component charge applies (the job is not disaggregation-class, or its checkpoint has
        no resolvable sidecar, e.g. a ``.ckpt`` pickle that can never carry one), the verdict is exactly the
        pre-feature path: the whole-checkpoint charge with the reusable-page credit still applied, so a model
        that lacks a sidecar is never admitted more strictly than before this feature existed (see
        :meth:`_disaggregated_component_charge_mb`). Otherwise runs the reclaim attempts (gentle
        eviction, escalated for the head, then cycling an allocator-stuck idle slot that is *not* this target)
        and dispatches on
        [`decide_ram_reclaim_outcome`][horde_worker_regen.process_management.scheduling.governance.preload_admission.decide_ram_reclaim_outcome]:
        reclaim progress is always worth waiting for, and only a head-of-queue blocker with no live job
        holding memory is admitted best-effort once nothing more can be reclaimed.
        """
        self._reconcile_reuse_credit()
        component_charge_mb = self._disaggregated_component_charge_mb(job, available_process)
        is_component_charge = component_charge_mb is not None
        ram_verdict = self._ram_budget.check_job(
            job,
            baseline,
            self._measured_available_ram_mb(),
            committed_reserve_mb=self._reserve_ledger.total_ram_mb(),
            reusable_credit_mb=0.0 if is_component_charge else self._staging_reuse_credit_mb(available_process),
            danger_floor_mb=self._ram_danger_floor_mb(),
            disaggregated=is_component_charge,
            component_charge_mb=component_charge_mb,
        )
        if ram_verdict.fits:
            self._ram_budget_defer_notified = False
            if is_component_charge:
                self._note_component_admission(job, available_process, ram_verdict)
            elif ram_verdict.reusable_credit_mb > 0.0:
                self._note_credited_admission(job, available_process, ram_verdict)
            self._resolve_head_ram_defer(job, reason="admitted")
            return True

        if not self._ram_budget_defer_notified:
            logger.opt(colors=True).warning(
                f"<fg #f0beff>RAM budget deferring preload of {{}}: {ram_verdict.reason()}. Reclaiming idle RAM.</>",
                job.model,
            )
            self._ram_budget_defer_notified = True
        reclaimed = self.unload_models(under_pressure=True)
        if not reclaimed and is_head_blocker:
            # Gentle reclaim freed nothing; for the head of the queue, escalate to reclaim a queued
            # model's RAM before falling back to cycling an allocator-stuck idle slot.
            reclaimed = self.unload_models(under_pressure=True, for_head_of_queue=True)
        # Spare the staging target from the cycle: its retained pages are the reuse the credited verdict
        # priced against, and cycling it would force the very cold load the credit exists to avoid. A
        # genuinely crept slot is still cycled by the creep-containment override inside the callee.
        cycled = (
            False
            if reclaimed
            else self._replace_stale_ram_unload_process(protect_process_id=available_process.process_id)
        )

        outcome = decide_ram_reclaim_outcome(
            reclaimed=reclaimed,
            cycled_stale_slot=cycled,
            is_head_blocker=is_head_blocker,
            no_live_resource_consumer=no_live_resource_consumer,
        )
        if outcome is RamReclaimOutcome.DEFER:
            if is_head_blocker:
                self._govern_head_ram_defer(job, made_reclaim_progress=reclaimed or cycled)
            return False

        logger.opt(colors=True).warning(
            "<fg #f0beff>RAM budget cannot fit head-of-queue model {} even after "
            "reclaiming all idle RAM, and no live job holds memory; admitting it best-effort "
            "rather than wedging the queue.</>",
            job.model,
        )
        self._resolve_head_ram_defer(job, reason="admitted best-effort")
        return True

    def _govern_head_ram_defer(self, job: ImageGenerateJobPopResponse, *, made_reclaim_progress: bool) -> None:
        """Advance the head RAM-defer starvation clock and latch or hard-cap the head-priority barrier.

        Reached only when the head-of-queue preload cannot fit system RAM and the RAM branch resolved to defer
        while a live job still holds memory (with no live consumer the caller best-effort admits instead, so
        the head never starves there). The continuous-defer clock starts at this head's first such defer and
        restarts on reclaim progress or a change of head, since only a head that keeps deferring with nothing
        freed is starving. Once the clock outlives ``_HEAD_RAM_DEFER_BARRIER_SECONDS`` the dispatch barrier
        latches so the running siblings drain to the best-effort escape; a barrier that has held past
        ``_HEAD_RAM_DEFER_BARRIER_CAP_SECONDS`` without admitting the head declines it for reissue. The barrier
        is suppressed while a deliberate RAM-reclaim process cycle is still inside its bounded grace, so an
        actively-resolving defer is never mistaken for starvation.
        """
        now = self._clock()
        job_id = str(job.id_) if job.id_ is not None else None
        if job_id is None:
            return

        if made_reclaim_progress or job_id != self._head_ram_defer_job_id:
            self._head_ram_defer_job_id = job_id
            self._head_ram_defer_since = now
            if made_reclaim_progress and self._head_priority_barrier_job_id == job_id:
                self._release_head_priority_barrier(reason="reclaim progress")
            return

        if (
            self._head_priority_barrier_job_id == job_id
            and (now - self._head_priority_barrier_since) >= _HEAD_RAM_DEFER_BARRIER_CAP_SECONDS
        ):
            self._decline_head_priority_barrier(job)
            return

        if (
            now - self._head_ram_defer_since
        ) < _HEAD_RAM_DEFER_BARRIER_SECONDS or self.ram_reclaim_cycle_grace_active():
            return

        self._engage_head_priority_barrier(job)

    def _resolve_head_ram_defer(self, job: ImageGenerateJobPopResponse, *, reason: str) -> None:
        """Clear the RAM-defer clock and release the barrier once this head's preload is admitted."""
        job_id = str(job.id_) if job.id_ is not None else None
        if job_id is None:
            return
        if job_id == self._head_ram_defer_job_id:
            self._head_ram_defer_job_id = None
            self._head_ram_defer_since = 0.0
        if job_id == self._head_priority_barrier_job_id:
            self._release_head_priority_barrier(reason=reason)

    def _engage_head_priority_barrier(self, job: ImageGenerateJobPopResponse) -> None:
        """Latch the head-priority dispatch barrier for a starved head (edge-triggered).

        While latched, :meth:`start_inference` withholds new dispatch to every slot but the head's own, so the
        running jobs drain to the no-live-consumer best-effort admit that finally seats the head.
        """
        job_id = str(job.id_) if job.id_ is not None else None
        if job_id is None or self._head_priority_barrier_job_id == job_id:
            return
        self._head_priority_barrier_job_id = job_id
        self._head_priority_barrier_since = self._clock()
        self._head_priority_barrier_withhold_logged = False
        logger.opt(colors=True).warning(
            "<fg #f0beff>Head-of-queue model {} has been RAM-deferred behind live work for over "
            f"{_HEAD_RAM_DEFER_BARRIER_SECONDS:.0f}s with reclaim freeing nothing; withholding new dispatch to "
            "other slots so the running jobs drain and the head can load.</>",
            job.model,
        )
        self._process_lifecycle.action_ledger.record(
            LedgerEventType.HEAD_PRIORITY_BARRIER_ENGAGED,
            job_id=job_id,
            reason="head RAM-deferred behind live work past the starvation bound",
            detail={"model": job.model},
        )

    def _release_head_priority_barrier(self, *, reason: str) -> None:
        """Release the head-priority dispatch barrier (edge-triggered) so normal dispatch resumes."""
        released_id = self._head_priority_barrier_job_id
        if released_id is None:
            return
        self._head_priority_barrier_job_id = None
        self._head_priority_barrier_since = 0.0
        self._head_priority_barrier_withhold_logged = False
        logger.opt(ansi=True).info(
            f"<fg #f0beff>Head-priority dispatch barrier released ({reason}); resuming normal dispatch.</>",
        )
        self._process_lifecycle.action_ledger.record(
            LedgerEventType.HEAD_PRIORITY_BARRIER_RELEASED,
            job_id=released_id,
            reason=reason,
        )

    def _decline_head_priority_barrier(self, job: ImageGenerateJobPopResponse) -> None:
        """Fault a head the barrier could not unblock within the cap, then release the barrier.

        The fault is retryable and resource-classed through the existing job fault machinery so the horde
        reissues the job promptly, the least-destructive terminal path: no model quarantine and no ceiling
        hold, because the block is transient host-RAM pressure behind live work, not an unroutable model.
        """
        logger.opt(colors=True).warning(
            "<fg #f0beff>Head-of-queue model {} could not be admitted within "
            f"{_HEAD_RAM_DEFER_BARRIER_CAP_SECONDS:.0f}s of holding the dispatch barrier; faulting it for "
            "reissue and releasing the barrier.</>",
            job.model,
        )
        self._job_tracker.handle_job_fault_now(
            job,
            is_resource_failure=True,
            retryable=True,
            fault_reason=(
                f"head-of-queue preload could not fit system RAM within "
                f"{_HEAD_RAM_DEFER_BARRIER_CAP_SECONDS:.0f}s of holding new dispatch; reissuing"
            ),
            fault_origin=JobFaultOrigin.SCHEDULING_RECOVERY,
        )
        job_id = str(job.id_) if job.id_ is not None else None
        if job_id is not None and job_id == self._head_ram_defer_job_id:
            self._head_ram_defer_job_id = None
            self._head_ram_defer_since = 0.0
        self._release_head_priority_barrier(reason="hard-cap decline")

    def _reconcile_head_priority_barrier(self, head_job: ImageGenerateJobPopResponse | None) -> None:
        """Clear the RAM-defer clock and release the barrier when their head is no longer the queue front.

        A head that dispatched, faulted, or was cancelled by the horde is no longer at the front, so its
        barrier must not outlive it and hold dispatch for a job that has departed.
        """
        head_id = str(head_job.id_) if head_job is not None and head_job.id_ is not None else None
        if self._head_ram_defer_job_id is not None and self._head_ram_defer_job_id != head_id:
            self._head_ram_defer_job_id = None
            self._head_ram_defer_since = 0.0
        if self._head_priority_barrier_job_id is not None and self._head_priority_barrier_job_id != head_id:
            self._release_head_priority_barrier(reason="head departed")

    def _head_priority_barrier_withholds_dispatch(self, next_job: ImageGenerateJobPopResponse) -> bool:
        """Whether the barrier withholds dispatching ``next_job`` so running siblings drain for the head.

        The barred head keeps its own path to dispatch; every other job is withheld so the card drains to the
        no-live-consumer best-effort admit that seats the head. Inert (returns False) when no barrier is held.
        """
        if self._head_priority_barrier_job_id is None:
            return False
        job_id = str(next_job.id_) if next_job.id_ is not None else None
        return job_id != self._head_priority_barrier_job_id

    def _safety_recovery_hold_active(self, target_device_index: int | None) -> bool:
        """Whether a crash-looping safety pool on a saturated card is holding new preload admissions here.

        Engages (edge-triggered) when the safety pool is crash-looping and a safety GPU start is deferred for
        want of device headroom, for a preload that would land on the same card as that deferred start (a
        single-GPU worker holds worker-wide). While held it nudges idle reclaim toward the card so it can drain
        and the safety pool can start. Releases when the safety pool starts or its crash-loop signal clears, or
        when the hold outlives ``_SAFETY_RECOVERY_HOLD_TTL_SECONDS`` (logged CRITICAL so a stuck safety pool is
        loud). A TTL expiry latches the episode: the hold does not re-engage while the same condition persists,
        so a permanently stuck safety pool no longer re-holds one preload per TTL window forever; it may hold
        again only once the condition clears and a fresh episode arises. The lifecycle reads are
        ``is True``-guarded so a mocked lifecycle never trips the hold.
        """
        failing = self._process_lifecycle.safety_pool_failing is True
        pending = self._process_lifecycle.has_pending_safety_starts() is True
        if not (failing and pending):
            self._release_safety_recovery_hold(reason="safety pool started or recovered")
            return False

        if (
            target_device_index is not None
            and target_device_index not in self._process_lifecycle.pending_safety_start_device_indices()
        ):
            return False

        if self._safety_recovery_hold_expired:
            # The TTL already gave up on this episode; do not re-hold or re-log while the same condition
            # persists. A fresh episode is possible only after the condition clears (the release path runs).
            return False

        now = self._clock()
        if self._safety_recovery_hold_since == 0.0:
            self._safety_recovery_hold_since = now
            self._engage_safety_recovery_hold(target_device_index)

        if (now - self._safety_recovery_hold_since) >= _SAFETY_RECOVERY_HOLD_TTL_SECONDS:
            self._expire_safety_recovery_hold(target_device_index)
            return False
        return True

    def _engage_safety_recovery_hold(self, target_device_index: int | None) -> None:
        """Announce the safety-recovery admission hold once and nudge idle reclaim so the card can drain."""
        self._safety_recovery_hold_logged = True
        logger.opt(ansi=True).warning(
            "<fg #f0beff>Safety pool is crash-looping while its GPU start is deferred on a saturated card; "
            "holding new inference preloads and reclaiming idle VRAM so the card can drain for safety.</>",
        )
        self._process_lifecycle.action_ledger.record(
            LedgerEventType.SAFETY_RECOVERY_HOLD_ENGAGED,
            reason="safety pool crash-looping with a deferred GPU start on a saturated card",
            detail={"device_index": target_device_index},
        )
        self.unload_models(under_pressure=True)

    def _release_safety_recovery_hold(self, *, reason: str) -> None:
        """Release the safety-recovery admission hold (edge-triggered) when safety can proceed.

        Also clears the per-episode expiry latch: once the crash-loop condition has cleared, a later
        recurrence is a fresh episode that may hold and expire again. A hold that had already given up at its
        TTL emits no second release notice (the expiry already logged one), so the edge stays single.
        """
        was_engaged = self._safety_recovery_hold_since != 0.0 or self._safety_recovery_hold_logged
        self._safety_recovery_hold_since = 0.0
        self._safety_recovery_hold_logged = False
        self._safety_recovery_hold_expired = False
        if not was_engaged:
            return
        logger.opt(ansi=True).info(
            f"<fg #f0beff>Safety-recovery admission hold released ({reason}); resuming inference preloads.</>",
        )
        self._process_lifecycle.action_ledger.record(
            LedgerEventType.SAFETY_RECOVERY_HOLD_RELEASED,
            reason=reason,
        )

    def _expire_safety_recovery_hold(self, target_device_index: int | None) -> None:
        """Release the safety-recovery hold at its TTL and latch the episode, logging CRITICAL once.

        The latch keeps the hold from re-engaging while the same crash-loop condition persists, so inference is
        not re-held one preload per TTL window; :meth:`_release_safety_recovery_hold` clears it when the
        condition finally clears.
        """
        self._safety_recovery_hold_since = 0.0
        self._safety_recovery_hold_logged = False
        self._safety_recovery_hold_expired = True
        logger.opt(ansi=True).critical(
            "<fg #ff5f5f>Safety pool still cannot start "
            f"{_SAFETY_RECOVERY_HOLD_TTL_SECONDS:.0f}s after the safety-recovery admission hold engaged; "
            "releasing the hold so inference is not starved. The safety pool needs operator attention.</>",
        )
        self._process_lifecycle.action_ledger.record(
            LedgerEventType.SAFETY_RECOVERY_HOLD_RELEASED,
            reason="ttl expired: safety pool still not started",
            detail={"device_index": target_device_index},
        )

    def _admit_preload_under_budget(
        self,
        job: ImageGenerateJobPopResponse,
        available_process: HordeProcessInfo,
        *,
        is_head_blocker: bool,
    ) -> bool:
        """Return whether ``job`` may be admitted for preload, with the VRAM arbiter as the deciding authority.

        True means proceed to send the preload; False means defer this cycle. The whole-card residency state
        machine is consulted first (pre-stage, defer, or fall through once its teardown has cleared the card to
        sole residency). Otherwise the arbiter prices the preload against the frozen cycle measurement: a FITS
        admits and runs the RAM verdict, and a DEFER (or the structural-impossibility DENY) runs the described
        pressure-relief actuations so the over-commit is relieved before the request re-asks next cycle. There
        is no overcommit-admit path: a head that never becomes admittable while the device is idle is rerouted by the
        structural-queue-wedge recovery supervisor. Every decision is scoped to the card this preload would land
        on (None keeps the worker-wide reading on a single-GPU host).
        """
        if job.model is None:
            raise ValueError(f"job.model is None ({job})")

        # Each return below names why it declined, so the recorded admission decision (and the dispatch-stall
        # line that quotes it) carries the deciding block instead of the generic gate label. The recorded text
        # is compared across cycles by the stall-line throttle and across recovery settling windows, so it is
        # the stable half of a verdict and never the arithmetic. Cleared per call so a later decline can never
        # inherit an earlier one's reason.
        self._last_budget_defer_reason = None

        baseline = self._model_metadata.get_baseline(job.model)
        target_device_index = available_process.device_index if self._multi_gpu_routing_active else None
        # A head waiting behind live work is queued, not starved. With no live job holding this card, the
        # starved-seconds value feeds the arbiter diagnostic and the RAM branch can still decide that exhausted
        # system-RAM reclaim should proceed.
        if target_device_index is None:
            no_live_resource_consumer = len(self._job_tracker.jobs_in_progress) == 0
        else:
            no_live_resource_consumer = len(self._jobs_in_progress_on_card(target_device_index)) == 0

        # Yield the card to a crash-looping safety pool whose GPU start is deferred for want of headroom on it:
        # holding new inference off the card (and nudging idle reclaim toward it) is the only way the card
        # drains enough for the safety pool to come up. Held whatever the arbiter's own verdict would be.
        if self._safety_recovery_hold_active(target_device_index):
            self._last_budget_defer_reason = "the card is held for a safety pool whose GPU start is deferred"
            return False

        forecast = self._forecast_streaming(job, baseline, device_index=target_device_index)
        # Trace the forecast for every budget-gated load so the logs show the residency dynamics, not just the
        # action taken. Unchanged observations are coalesced by _log_stream_forecast.
        self._log_stream_forecast(job, forecast)

        whole_card = self._decide_whole_card_demand(
            job,
            available_process,
            forecast,
            baseline,
            is_head_blocker=is_head_blocker,
            target_device_index=target_device_index,
        )
        if whole_card is _WholeCardDemandOutcome.DEFER:
            # The residency path names its own arithmetic where it has any (which governor, what it has spent);
            # the generic label only stands in when the deferral was the teardown itself still running.
            self._last_budget_defer_reason = (
                self._last_budget_defer_reason or "a whole-card residency demand for this model is deferred"
            )
            return False
        if whole_card is _WholeCardDemandOutcome.PRESTAGE:
            # A RAM-only pre-stage of a whole-card head: the VRAM budget deliberately does not fit it
            # co-resident (that is *why* it gets the whole card), so skip the verdict and send the preload.
            return True

        arbiter = self._ensure_preload_arbiter()

        # Price the predictive verdict once: it sources the candidate delta and the rejected peak the
        # context-reduction remedy is sized from.
        vram_verdict = self._vram_budget.check_job(
            job,
            baseline,
            self._measured_free_vram_mb(device_index=target_device_index),
            committed_reserve_mb=self._committed_vram_reserve_mb(device_index=target_device_index),
            disaggregated=self._is_disaggregation_class_eligible(job),
        )
        max_resident, can_reduce_live_contexts = self._context_reduction_demand(
            vram_verdict,
            forecast,
            is_head_blocker=is_head_blocker,
            target_device_index=target_device_index,
        )
        has_reclaimable_idle_model = self._has_reclaimable_idle_model(
            available_process,
            for_head_of_queue=is_head_blocker,
            device_index=target_device_index,
            make_room_for_model=job.model,
        )
        live_inference_processes = self._process_map.num_loaded_inference_processes(
            device_index=target_device_index,
        )
        idle_contexts_teardownable = (
            is_head_blocker
            and max_resident is not None
            and max_resident < live_inference_processes
            and self._has_teardownable_idle_context(
                available_process,
                device_index=target_device_index,
            )
        )
        request = self._build_preload_request(
            job,
            available_process,
            baseline,
            target_device_index=target_device_index,
            is_head_blocker=is_head_blocker,
            has_reclaimable_idle_model=has_reclaimable_idle_model,
            can_reduce_live_contexts=can_reduce_live_contexts,
            idle_contexts_teardownable=idle_contexts_teardownable,
        )
        verdict = arbiter.evaluate(request)

        if verdict.disposition is VramDisposition.FITS:
            if verdict.measured_attempt:
                self._mark_measured_attempt(job, request, device_index=target_device_index)
            # A FITS is a real fit against the truthful device-free reading (the identity already accounts for
            # baseline and foreign load, which are physically inside that reading). The model still loads
            # through system RAM, so the marginal RAM verdict runs.
            return self._apply_ram_verdict(
                job,
                baseline,
                available_process,
                is_head_blocker=is_head_blocker,
                no_live_resource_consumer=no_live_resource_consumer,
            )

        # A structural-impossibility DENY for the true head, with no other card that could ever seat it, is a
        # permanent wall: the model's demand exceeds this card's achievable ceiling (total net of the noise
        # buffer and the sustained foreign floor), so deferring only wedges the queue behind a head that no
        # reclaim can admit and no reroute can rescue. Fault the head terminally so the horde reissues it and
        # head protection releases, and quarantine the model for the session so it is not popped into the same
        # wall again. A DENY with a live reroute target, or for a non-head demand, still falls through to the
        # ordinary defer-and-reclaim path below.
        if (
            verdict.disposition is VramDisposition.DENY
            and is_head_blocker
            and not arbiter.any_other_device_can_seat(request, exclude_device_index=target_device_index)
        ):
            self._fault_unroutable_head(job, arbiter, request, device_index=target_device_index)
            return False

        # DEFER (or a structural DENY that can still reroute or is not the head): run the described
        # pressure-relief commands so the over-commit is relieved, then defer. There is no overcommit admit: a
        # head that stays deferred while the device is idle is rerouted by the structural-queue-wedge recovery
        # supervisor, and the arbiter emits a starvation diagnostic naming the arithmetic before then.
        # Completion is observed via the next cycle's frozen snapshot; the actuations are never awaited inline.
        # Emitted whatever the predictive forecast said. A defer against a candidate the static forecast calls
        # a fit is the case most worth naming: the arithmetic the arbiter objects to is then visible nowhere
        # else, and a head can sit deferred behind it for as long as the queue holds. Coalesced on the stable
        # reason rather than latched, so a changed objection always speaks and an unchanged one is counted;
        # the line itself renders the arithmetic behind that reason, which moves every cycle.
        self._last_budget_defer_reason = verdict.reason
        suppressed = self._scheduler_diagnostic_suppressed_count(
            "preload_budget_defer",
            (str(job.model), verdict.disposition.value, verdict.reason),
        )
        if suppressed is not None:
            forecast_note = "" if not vram_verdict.fits else " (the static forecast calls this a fit)"
            logger.opt(colors=True).warning(
                f"<fg #f0beff>VRAM arbiter deferring preload of {{}}: {verdict.stated}{forecast_note}. "
                f"Reclaiming idle VRAM.{self._suppressed_suffix(suppressed)}</>",
                job.model,
            )
        self._preload_actuation = _PreloadActuation(
            job=job,
            available_process=available_process,
            forecast=forecast,
            max_resident=max_resident,
        )
        try:
            self._execute_preload_actuations(
                verdict.required_actuations,
                device_index=target_device_index,
                for_head_of_queue=is_head_blocker,
            )
        finally:
            self._preload_actuation = None
        return False

    def _mark_measured_attempt(
        self,
        job: ImageGenerateJobPopResponse,
        request: VramRequest,
        *,
        device_index: int | None,
    ) -> None:
        """Tag a job admitted through the arbiter's measured-load escape hatch, recording the attempted demand.

        The arbiter emits the one attempt-start log line (edge-triggered on its own one-shot), so this only
        records the tag and the candidate the attempt was taken under. The tag rides on the job so that a
        subsequent child OOM is routed to a terminal scheduling-recovery fault that arms the ceiling hold with
        real-attempt evidence (the candidate the load actually failed at), rather than to the ordinary degraded
        retry that would bang the same converged-empty card again.
        """
        self._job_tracker.mark_measured_attempt(
            job,
            candidate_mb=request.candidate_delta_mb or 0.0,
            device_index=device_index,
        )

    def _fault_unroutable_head(
        self,
        job: ImageGenerateJobPopResponse,
        arbiter: VramArbiter,
        request: VramRequest,
        *,
        device_index: int | None,
    ) -> None:
        """Terminally fault an unroutable head and place a conditional ceiling hold on its model.

        Reached only for the true head when the measured achievable ceiling (total net of the noise buffer and
        the sustained foreign floor) cannot hold its predicted demand on any card. The job is faulted now as a
        scheduling-recovery action so the horde reissues it while it cannot fit (a job is time-bound) and the
        give-up/head-protection machinery sees the head resolved; the fault is excluded from the
        consecutive-failure pop pause (a card-fit verdict the recovery path owns, not a generation outcome). The
        model is placed on a conditional ceiling hold, not a permanent quarantine: pop advertising and dispatch
        stop offering it *while* the ceiling holds, and it becomes advertisable again on its own once the ceiling
        recedes (the foreign floor drops). The operator warning is edge-triggered on each fresh arm.
        """
        state = arbiter.device_state(device_index)
        candidate_mb = request.candidate_delta_mb or 0.0
        ceiling_mb = state.achievable_ceiling_mb() if state is not None else None
        foreign_mb = state.foreign_floor_mb if state is not None else None
        arithmetic = (
            f"model {job.model} needs {candidate_mb:.0f} MB; this card can currently offer at most "
            f"{ceiling_mb:.0f} MB with {foreign_mb:.0f} MB held by other processes"
            if ceiling_mb is not None and foreign_mb is not None
            else f"model {job.model} exceeds this card's current achievable VRAM ceiling"
        )
        newly_armed = self._job_tracker.hold_model_by_ceiling(
            job.model,
            device_index=device_index,
            candidate_mb=candidate_mb,
        )
        if newly_armed:
            logger.opt(colors=True).warning(
                "<fg #f0beff>VRAM: {}. It is held (not offered) while other processes hold that "
                "VRAM, and its jobs are faulted for reissue meanwhile. The hold frees automatically once that "
                "VRAM is released (close other GPU apps), or you can remove the model from this worker's "
                "config.</>",
                arithmetic,
            )
        self._job_tracker.handle_job_fault_now(
            job,
            retryable=False,
            fault_reason=f"unroutable head (ceiling hold): {arithmetic}",
            fault_origin=JobFaultOrigin.SCHEDULING_RECOVERY,
        )

    def _ensure_preload_arbiter(self) -> VramArbiter:
        """Return the preload-admission arbiter, priming a private one with the current measurement if unwired.

        In the running worker the manager injects the shared arbiter and freezes its cycle once per
        control-loop iteration, so this returns that instance with the tick's snapshot intact. A scheduler
        exercised on its own (no manager tick, no injected cycle) gets a private arbiter primed with a
        freshly-built snapshot, so admission stays fully governed rather than falling to an ungoverned path.
        """
        arbiter = self._vram_arbiter
        if arbiter is None:
            arbiter = VramArbiter()
            self._vram_arbiter = arbiter
            self._owns_private_vram_arbiter = True
        if self._owns_private_vram_arbiter:
            # Nothing else advances a private arbiter's cycle, so priming it only while it has none would
            # freeze the very first measurement for the rest of the session: every later admission would be
            # priced against a card state that has since been reclaimed, and a head the reclaim made room for
            # would be declined forever on a reading that no longer describes the device. Re-prime each time
            # so "the current measurement" stays true past the first request.
            arbiter.begin_cycle(self.build_vram_arbiter_snapshot())
        elif not arbiter.has_cycle:
            arbiter.begin_cycle(self.build_vram_arbiter_snapshot())
        return arbiter

    def _build_preload_request(
        self,
        job: ImageGenerateJobPopResponse,
        available_process: HordeProcessInfo,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        *,
        target_device_index: int | None,
        is_head_blocker: bool,
        has_reclaimable_idle_model: bool,
        can_reduce_live_contexts: bool,
        idle_contexts_teardownable: bool,
    ) -> VramRequest:
        """Assemble the arbiter request for one preload, priced identically to the measured admission overlay."""
        return VramRequest(
            kind=VramRequestKind.PRELOAD,
            job_label=str(job.model),
            baseline=baseline,
            device_index=target_device_index,
            target_process_id=available_process.process_id,
            candidate_delta_mb=self._preload_candidate_delta_mb(
                job,
                baseline,
                process_id=available_process.process_id,
            ),
            candidate_weights_mb=predict_job_weight_mb(job, baseline),
            accepted_work=job.id_ is not None and self._job_tracker.get_tracked_job(job.id_) is not None,
            candidate_already_resident=self._candidate_weights_resident_on_process(
                job.model,
                available_process.process_id,
            ),
            own_planned_unmaterialized_mb=self._own_planned_charge_mb(
                device_index=target_device_index,
                target_process_id=available_process.process_id,
            ),
            is_head_of_queue=is_head_blocker,
            head_job_id=str(job.id_) if job.id_ is not None else None,
            measured_attempt_in_progress=self._job_tracker.is_measured_attempt_on_device(
                job,
                target_device_index,
            ),
            measured_attempt_already_spent=self._job_tracker.has_spent_measured_attempt_on_device(
                job,
                target_device_index,
            ),
            starved_seconds=self._head_starved_seconds(job),
            has_reclaimable_idle_model=has_reclaimable_idle_model,
            can_reduce_live_contexts=can_reduce_live_contexts,
            idle_contexts_teardownable=idle_contexts_teardownable,
        )

    def _preload_candidate_delta_mb(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
        *,
        process_id: int | None,
    ) -> float | None:
        """Return the VRAM (MB) a preload of ``job`` charges the card at admission.

        Without the clearance lease a preload is the VRAM moment (the child loads the weights when the job
        starts), so it is priced at the job's full marginal sampling charge. Under the lease a preload only
        stages the job in system RAM and the weights load inside the leased sample call, so the charge is
        capped at the staging encode footprint, the same figure a staged dispatch books; the full fit-or-evict
        runs at clearance. Pricing the stage at the full peak parks the next model's disk read behind the
        current sample on every model switch, since a second peak almost never fits beside a running one, and
        that read is exactly the work the stage exists to overlap.
        """
        delta_mb = self._measured_admission_candidate_delta_mb(
            job,
            baseline,
            process_id=process_id,
            disaggregated=self._is_disaggregation_class_eligible(job),
        )
        if delta_mb is None or not self._clearance_lease_active():
            return delta_mb
        return min(delta_mb, _STAGING_ENCODE_VRAM_MB)

    def _own_planned_charge_mb(self, *, device_index: int | None, target_process_id: int | None) -> float:
        """Return the planned-overlay charge (MB) attributable to a request's own target process.

        The arbiter subtracts this from the device's planned overlay so a re-ask nets out the load it itself
        admitted on an earlier cycle (the candidate delta already represents it), preventing the head-of-queue
        self-deadlock where a load's own not-yet-materialised plan holds the card against its re-ask. Every
        other process's planned charge is left intact, so genuinely-concurrent admissions still count in full.
        """
        if target_process_id is None:
            return 0.0
        return self._reserve_ledger.planned_charge_for_unit(
            PRELOAD_ADMISSION_FLOW,
            str(target_process_id),
            self._committed_process_reserved_by_pid(device_index),
        )

    def _record_dispatch_reservation(
        self,
        job: ImageGenerateJobPopResponse,
        process_info: HordeProcessInfo,
        *,
        baseline: str | None,
        staging_only: bool = False,
    ) -> None:
        """Register the outstanding reservation for a dispatch the moment it is sent inference.

        Under the clearance lease a dispatch stages the job (checkpoint disk load, prompt encode) without
        loading the diffusion weights to VRAM, which happens inside the leased sample call at clearance. With
        ``staging_only`` the reservation is booked at the encode charge (:data:`_STAGING_ENCODE_VRAM_MB`)
        rather than the full activation-inclusive peak, so a staged job holds only the device footprint it
        actually incurs. :meth:`_upgrade_dispatch_reservation_to_full` re-books it at the full peak when the
        job is cleared (``set_planned`` refreshes the same ``(flow, unit)`` entry in place, re-charging in full
        and resetting its materialisation watermark, so the encode charge is upgraded, never double-booked).

        A dispatch onto an already-resident model materialises the job's activation-inclusive peak (net of the
        resident-weight credit the model already holds) over the sampling window the device-free reading does
        not yet reflect. Recording that peak as a reservation keyed by the job id, targeting the sampling
        process and baselined at the process's current reservation, means a second admission in the same window
        sees it and cannot over-admit into the same physical room; the reservation decays one-for-one as this
        process's own reservation materialises the activation, and drops by omission once the job leaves the
        in-progress set (finalised, faulted, or process-dead). An unpriceable job reserves nothing rather than
        pinning the overlay at a fabricated figure.

        The full charge is priced sampler-only for a disaggregation-class-eligible job (its weights load and
        decode run off the sampling process), matching how every other measured-admission site prices the same
        job, so the upgrade never over-books a disaggregated job's peak at the monolithic whole-job figure.
        """
        if job.id_ is None:
            return
        if staging_only:
            charge_mb: float | None = _STAGING_ENCODE_VRAM_MB
        else:
            charge_mb = self._measured_admission_candidate_delta_mb(
                job,
                baseline,
                process_id=process_info.process_id,
                disaggregated=self._is_disaggregation_class_eligible(job),
            )
        self._reserve_ledger.set_planned(
            DISPATCH_ADMISSION_FLOW,
            str(job.id_),
            vram_mb=charge_mb if charge_mb is not None else 0.0,
            target_process_id=process_info.process_id,
            reserved_at_admit_mb=float(process_info.process_reserved_mb or 0),
        )

    def _upgrade_dispatch_reservation_to_full(
        self,
        job: ImageGenerateJobPopResponse,
        process_info: HordeProcessInfo,
        *,
        baseline: str | None,
    ) -> None:
        """Re-book a staged job's dispatch reservation at its full materialisation peak when it is cleared.

        Called at clearance (the VRAM moment) so the ledger reflects the weights-plus-activation peak the
        leased sample call is about to materialise, rather than the encode-only staging charge. Re-registering
        the same ``(DISPATCH_ADMISSION_FLOW, job id)`` entry through :meth:`_record_dispatch_reservation`
        refreshes it in place: the full charge replaces the encode charge and the target's materialisation
        watermark resets, so the upgrade is not a second, additive booking.
        """
        self._record_dispatch_reservation(job, process_info, baseline=baseline, staging_only=False)

    def release_dispatch_reservation(self, job: ImageGenerateJobPopResponse) -> None:
        """Drop a dispatch's outstanding reservation on clean finalization (a latency tightener over omission).

        Reconcile-by-omission already releases a dispatch reservation the next cycle a job leaves the
        in-progress set, so this is not the correctness guarantee: it only shortens the window between a job
        completing and its reservation clearing, so the freed room is available to the next admission sooner.
        Idempotent and safe for a job that never reserved (no-op).
        """
        if job.id_ is None:
            return
        self._reserve_ledger.release(DISPATCH_ADMISSION_FLOW, str(job.id_))

    def _displaced_head_outstanding_mb(
        self,
        displaced_head: ImageGenerateJobPopResponse,
        *,
        device_index: int | None,
    ) -> float | None:
        """Return the head-of-queue demand (MB) a line-skipper must not consume, or None when unpriceable.

        The head a line-skip jumped is still downloading, so its weights are not yet resident: its outstanding
        demand is its full priced candidate (no resident-weight credit). Head protection uses this to hold the
        skipper when admitting it would leave the card short of the room the head needs. None (an unpriceable
        head, or a model-less one) skips the protection rather than fabricating a figure, degrading to admitting
        the skipper.

        Head protection is also released once the head has been parked past
        :data:`_HEAD_PROTECTION_MAX_STARVE_SECONDS` without dispatching. Reserving room for a head is only
        worth anything if the head eventually takes it: a head whose own admission keeps declining holds the
        card empty while runnable siblings that fit are turned away, which serves nobody. The head keeps its
        queue position and first claim on the next opportunity; it simply stops blocking work in the meantime.

        It is released the same way while a churn governor is deferring this head's whole-card establishment
        on this card. For the length of that deferral the head is not asking for the card at all (normal
        scheduling continues around it by design), so reserving its whole-card demand against smaller ready
        work would idle the card against a claim nobody is making. Once the deferral clears or its dwell is
        spent the head is served by ordinary admission and its normal charge applies again.
        """
        if displaced_head.model is None:
            return None
        if self._whole_card_ledger.governor_deferred_head(device_index, now=self._clock()) == displaced_head.model:
            self._note_head_protection_governor_deferred(displaced_head)
            return None
        if self._head_starved_seconds(displaced_head) >= _HEAD_PROTECTION_MAX_STARVE_SECONDS:
            self._note_head_protection_released(displaced_head)
            return None
        baseline = self._model_metadata.get_baseline(displaced_head.model)
        return self._measured_admission_candidate_delta_mb(
            displaced_head,
            baseline,
            process_id=None,
            disaggregated=self._is_disaggregation_class_eligible(displaced_head),
        )

    def _note_head_protection_governor_deferred(self, head: ImageGenerateJobPopResponse) -> None:
        """Disclose that a head whose whole-card ask is governor-deferred has stopped reserving card room."""
        if head.id_ is None:
            return
        if self._decision_sink is not None:
            self._decision_sink(
                decision_kind=DecisionKind.INFERENCE_DISPATCH,
                subject=str(head.id_),
                verdict=DecisionVerdict.NO_OP,
                reason="head protection released: the head's whole-card establishment is governor-deferred",
                inputs={"model": str(head.model)},
            )
        suppressed = self._scheduler_diagnostic_suppressed_count(
            "head_protection_governor_deferred",
            (str(head.id_),),
        )
        if suppressed is None:
            return
        logger.opt(colors=True).warning(
            "<fg #ff8c69>Head {} ({}) is not asking for this card while a governor defers its whole-card "
            "residency, so it no longer reserves card room from the jobs behind it; a fitting sibling may "
            f"dispatch. The head keeps its queue position.{self._suppressed_suffix(suppressed)}</>",
            str(head.id_)[:8],
            head.model,
        )

    def _note_head_protection_released(self, head: ImageGenerateJobPopResponse) -> None:
        """Disclose that a starved head has stopped reserving room from the jobs behind it."""
        if head.id_ is None:
            return
        starved_seconds = self._head_starved_seconds(head)
        if self._decision_sink is not None:
            self._decision_sink(
                decision_kind=DecisionKind.INFERENCE_DISPATCH,
                subject=str(head.id_),
                verdict=DecisionVerdict.NO_OP,
                reason="head protection released: the head has not dispatched within its protection window",
                inputs={
                    "model": str(head.model),
                    "starved_seconds": round(starved_seconds, 1),
                },
            )
        suppressed = self._scheduler_diagnostic_suppressed_count(
            "head_protection_released",
            (str(head.id_),),
        )
        if suppressed is None:
            return
        logger.opt(colors=True).warning(
            "<fg #ff8c69>Head {} ({}) has been parked {:.0f}s without dispatching, so it no longer reserves "
            "card room from the jobs behind it; a fitting sibling may dispatch. The head keeps its queue "
            f"position.{self._suppressed_suffix(suppressed)}</>",
            str(head.id_)[:8],
            head.model,
            starved_seconds,
        )

    def _context_reduction_demand(
        self,
        vram_verdict: BudgetVerdict,
        forecast: StreamForecast,
        *,
        is_head_blocker: bool,
        target_device_index: int | None,
    ) -> tuple[int | None, bool]:
        """Return the head's context-reduction target and whether reducing live contexts is a warranted remedy.

        A moderate head's weights fit after a model eviction but its activation peak does not while this many
        contexts are live (each extra context retains VRAM the allocator never returns). Reducing the live
        inference-process count to the largest that still seats the rejected peak plus its structural reserve
        is the remedy. The depth keys on the honest streaming floor, not the operator's configured margin, so
        only a genuinely card-filling peak pushes the co-resident count below the live pool; a demand resting
        on untrusted (unmeasured-fallback) overhead figures is not warranted.

        The warrant is measured, never an operator preference. ``whole_card_exclusive_residency`` governs
        whether the worker takes an exclusive *residency* (a lease with a cooldown, safety and the service
        lanes moved off the card); a context reduction is none of those, and both the actuator that performs it
        (:meth:`reduce_live_contexts`) and the reachability predicate behind it
        (:meth:`_has_teardownable_idle_context`) are deliberately independent of the flag. Gating the demand on
        it would leave a head that is servable only once an idle context is torn down with no ordinary route to
        that reclaim, deferring it against a card whose own idle contexts hold the deficit.
        """
        max_resident: int | None = None
        if vram_verdict.predicted_mb is not None:
            total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=target_device_index)
            structural_reserve_mb = (
                effective_inference_reserve_mb(total_vram_mb, 0.0)
                if total_vram_mb is not None
                else vram_verdict.reserve_mb
            )
            max_resident = self._max_coresident_for_peak_mb(
                vram_verdict.predicted_mb,
                structural_reserve_mb,
                device_index=target_device_index,
            )
        context_reduction_demanded = (
            is_head_blocker
            and max_resident is not None
            and self._process_map.num_loaded_inference_processes(device_index=target_device_index) > max_resident
        )
        can_reduce = context_reduction_demanded and self._whole_card_warranted(forecast)
        return max_resident, can_reduce

    def _target_slot_is_spared(
        self,
        process_info: HordeProcessInfo,
        *,
        make_room_for_model: str | None,
    ) -> bool:
        """Return whether the slot a load is aimed at must be spared from the reclaim being performed for it.

        Reclaim aimed at seating a model on a slot spares that slot: whatever it holds is about to be written
        anyway, so evicting it buys nothing and costs a reload. That reasoning covers the slot's own copy of
        the model being seated (its residency is credit against the load, not memory to reclaim) and a slot
        that is already mid-load, whose weights are arriving rather than idle. It does not cover a *different*
        model sitting idle on the slot: those weights are the ordinary cost of a model swap, and on a worker
        whose card has one inference lane they are the only memory a starved head can be given.

        ``make_room_for_model`` None means the caller named no model to seat (a bare pressure sweep, where the
        slot is passed purely as a process to protect), so the slot is spared outright.
        """
        if make_room_for_model is None:
            return True
        if process_info.process_type is not HordeProcessType.INFERENCE:
            return True
        # PRELOADING_MODEL / PRELOADED_MODEL / DOWNLOADING_MODEL all report busy, so a slot whose weights are
        # on their way is never mistaken for an idle resident.
        if process_info.is_process_busy():
            return True
        loaded_model = process_info.loaded_horde_model_name
        return loaded_model is None or loaded_model == make_room_for_model

    def _has_reclaimable_idle_model(
        self,
        process_with_model: HordeProcessInfo,
        *,
        for_head_of_queue: bool,
        device_index: int | None,
        make_room_for_model: str | None = None,
    ) -> bool:
        """Return whether an idle resident model could be evicted on the card to reclaim VRAM for this head.

        A read-only mirror of the eviction targeting :meth:`unload_models_from_vram` performs under pressure:
        a post-processing lane not already unloading, or an inference process holding a model that is not in
        progress, not spared by the queued-lookahead or residency guards (both of which the head escalation
        overrides), and not already unloading. It never counts an in-progress model, and it excludes the head's
        own target slot on the terms :meth:`_target_slot_is_spared` sets: unconditionally when the caller names
        no model to seat, otherwise only while that slot holds the head's own model or is mid-load. When this
        is False, and no idle cache and no warranted context reduction remain, reclamation is structurally
        exhausted for this head.
        """
        wanted_models = self._compute_wanted_models()
        next_n_models = list(self.get_next_n_models(self._max_inference_processes))
        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress}
        for process_info in self._process_map.values():
            if process_info.process_id == process_with_model.process_id and self._target_slot_is_spared(
                process_info,
                make_room_for_model=make_room_for_model,
            ):
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.process_type == HordeProcessType.POST_PROCESS:
                if process_info.is_process_busy():
                    continue
                if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                    continue
                return True
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if process_info.loaded_horde_model_name is None:
                continue
            # The single-model unload guard (skip when only one model is configured) applies only when the
            # reclaim is not under pressure; every preload reclaim here is under pressure, so it never spares.
            if process_info.loaded_horde_model_name in in_progress_models:
                continue
            if (
                process_info.loaded_horde_model_name in next_n_models
                and not for_head_of_queue
                and self._coresident_lookahead_affordable(
                    process_info.loaded_horde_model_name,
                    device_index=device_index,
                )
            ):
                continue
            if not for_head_of_queue and self._residency_protects_from_unload(
                process_info.loaded_horde_model_name,
                wanted_models,
                vram=True,
                under_pressure=True,
            ):
                continue
            if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                continue
            return True
        return False

    def _has_teardownable_idle_context(
        self,
        head_process: HordeProcessInfo,
        *,
        device_index: int | None,
    ) -> bool:
        """Return whether an idle sibling inference context could be torn down to reclaim VRAM for a starved head.

        A bare CUDA context's VRAM is reclaimed only when its process exits, which weight eviction (model
        unload, cache release) cannot achieve. An idle inference process on the card other than the head's own
        target slot, not busy and not serving an in-progress job, is a teardown candidate the starvation
        escalation can reduce via :meth:`reduce_live_contexts`. Excludes the head's target slot and every busy
        process, matching that actuator's own protections.

        This is independent of ``whole_card_exclusive_residency``: that flag governs whether the worker
        establishes exclusive residency as a steady-state preference, but the starvation escalation is an
        emergency liveness path (a head starved past the arbiter's threshold whose own idle contexts hold the
        deficit) that must be reachable regardless. The actuation runs through :meth:`reduce_live_contexts` ->
        ``scale_inference_processes``, neither of which gates on the flag, so tearing the idle contexts down
        proceeds when the flag is off.
        """
        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress}
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if process_info.process_id == head_process.process_id:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.is_process_busy():
                continue
            if (
                process_info.loaded_horde_model_name is not None
                and process_info.loaded_horde_model_name in in_progress_models
            ):
                continue
            return True
        return False

    def _execute_preload_actuations(
        self,
        commands: tuple[ActuatorCommand, ...],
        *,
        device_index: int | None,
        for_head_of_queue: bool,
    ) -> tuple[ActuatorCommand, ...]:
        """Run the pressure-relief commands a deferred preload verdict described, at most once each this cycle.

        RELEASE_CACHE returns an idle lane's cached allocator reservation to the card; EVICT_IDLE_MODEL frees
        an idle resident model's weights; REDUCE_LIVE_CONTEXTS collapses the live inference-process count so a
        retained per-context reservation returns; CYCLE_SAFETY_OFF_GPU frees the safety context. The arbiter
        guarantees RELEASE_CACHE targets only idle lanes, so a busy lane is never asked to release its cache.

        The command dispatch is routed through :meth:`VerifiedReclaimLadder.execute_arbiter_commands` so this
        DEFER path and the governor's SATURATED verified ladder share one reclaim execution surface (the
        single-owner rule): the two triggers can never become two mechanisms evicting the same card by
        different rules.
        """
        return VerifiedReclaimLadder.execute_arbiter_commands(
            commands,
            self,
            device_index=device_index,
            for_head_of_queue=for_head_of_queue,
        )

    def release_cache(self, process_id: int) -> bool:
        """Return an idle lane's cached allocator reservation to the device (:class:`VramActuator`)."""
        return self.release_allocator_cache(process_id)

    def evict_idle_model(self, device_index: int | None, *, for_head_of_queue: bool) -> bool:
        """Evict an idle resident model on the card to reclaim its weights (:class:`VramActuator`).

        The head being admitted keeps its own model's weights wherever they sit, including on its own target
        slot, and no live in-progress model is ever touched. A *different* model idle on that target slot is
        evictable: seating the head there is a model swap, and it is the only reclaim a single-lane card has.
        """
        actuation = self._preload_actuation
        anchor = (
            actuation.available_process
            if actuation is not None
            else self._pressure_reclaim_anchor(device_index=device_index)
        )
        if anchor is None:
            return False
        return self.unload_models_from_vram(
            anchor,
            under_pressure=True,
            for_head_of_queue=for_head_of_queue,
            device_index=device_index,
            make_room_for_model=actuation.job.model if actuation is not None else None,
        )

    def reduce_live_contexts(self, device_index: int | None) -> bool:
        """Reduce the live inference-context count for the current head (:class:`VramActuator`).

        Stops the card's idle inference contexts down to the depth the rejected peak sized and evicts the idle
        residents on the other processes, so the VRAM a live process retains for its context returns to the
        card. That is the whole of it: this is a reclaim actuation, not a policy grant. Reserving the worker
        for the head, stamping a residency lease and its cooldown, moving safety off the GPU, pausing the
        service lanes and opening the establish grace window are commitments of a *whole-card residency*, and
        they belong to the grant that asks for one (:meth:`_establish_whole_card_residency`), which may itself
        request a reduction through this same surface. Taking them here would impose them on an operator who
        declined that policy, since emergency reclaim is not gated on a steady-state preference.

        The reduction is booked with the reclaim ladder as a restore obligation, so the pool it shrank is
        grown back when the card recovers (:meth:`restore_live_contexts`) rather than leaving the worker at
        emergency depth for the rest of the session.

        A no-op when no head-preload context is recorded, the target could not be sized, the command is stale
        and the live pool is already at or below its target, or another reduction on this card is still inside
        :data:`_CONTEXT_REDUCTION_MIN_INTERVAL_SECONDS`.
        """
        actuation = self._preload_actuation
        if actuation is None:
            return False
        target = actuation.max_resident
        live_processes = self._process_map.num_loaded_inference_processes(device_index=device_index)
        if target is None or target >= live_processes:
            return False
        # A head re-asks every cycle, so without a floor on the rate one rejected peak can buy a reduction per
        # cycle: each costs a cold start, and the pool it shrinks is regrown between them.
        last_reduction = self._context_reduction_at.get(device_index)
        if last_reduction is not None and (self._clock() - last_reduction) < _CONTEXT_REDUCTION_MIN_INTERVAL_SECONDS:
            return False
        # The head's own holder is the process the reduction is making room for, so it is named here to spare
        # it and to drop the "spare any process whose model is queued" protection, which would otherwise let a
        # sibling holding a model queued *behind* the head pin the count above the target and free nothing.
        after = self._process_lifecycle.scale_inference_processes(
            target,
            device_index=device_index,
            protected_model=actuation.job.model,
        )
        self.unload_models_from_vram(
            actuation.available_process,
            under_pressure=True,
            for_head_of_queue=True,
            device_index=device_index,
        )
        if self._reclaim_ladder is not None:
            self._reclaim_ladder.record_context_reduction(device_index)
        # Stamped only once the reduction has actually been taken, so a bail-out added above can never charge
        # the rate limit for a reduction that did not happen.
        self._context_reduction_at[device_index] = self._clock()
        self._record_churn("context_reduction")
        logger.info(
            f"Reclaiming live inference contexts for {actuation.job.model} "
            f"(inference processes {live_processes} -> {after} of {self._max_inference_processes}, target "
            f"{target}); the pool is grown back once the card recovers.",
        )
        return True

    def restore_live_contexts(self, device_index: int | None) -> bool:
        """Regrow a card's inference pool after a reclaim reduction (reclaim-ladder actuator).

        The unwind of :meth:`reduce_live_contexts`, driven by the reclaim ladder once the governor calls the
        card HEALTHY. Stands down while that card holds a whole-card residency: the residency's own restore
        (:meth:`_restore_siblings_after_whole_card`) owns the regrowth then, and growing the pool underneath a
        held residency would re-add the very contexts it cleared. Reports no-op when the pool is already at
        its configured size.
        """
        if self._whole_card_ledger.any_held():
            return False
        ceiling = self._residency_restore_ceiling(device_index)
        current = self._process_map.num_loaded_inference_processes(device_index=device_index)
        if current >= ceiling:
            return False
        after = self._process_lifecycle.scale_inference_processes(ceiling, device_index=device_index)
        self._reconcile_worker_shed_to_pool()
        self._record_churn("context_restore")
        logger.info(
            f"Card recovered; restoring the inference contexts reclaimed under pressure "
            f"({current} -> {after} of {ceiling}).",
        )
        return True

    def cycle_safety_off_gpu(self, device_index: int | None) -> bool:
        """Cycle the safety model off the GPU to reclaim its context (:class:`VramActuator`)."""
        return self.safety_off_gpu(device_index)

    def build_reclaim_ladder_candidates(
        self,
        device_index: int | None,
        *,
        protected_models: frozenset[str] = frozenset(),
    ) -> LadderCandidates:
        """Assemble the idle-filtered inputs the verified reclaim ladder orders into rungs for a card.

        Every actively-sampling process is excluded, so a busy tenant can never become a rung. Idle inference
        processes still holding a model (and not serving in-progress or caller-protected demand) are unload
        candidates, ranked by recency via their last model-state report. ``protected_models`` lets SOS protect
        the exact resident capacity a wedged queue is waiting to dispatch without weakening ordinary
        device-pressure reclaim, which may evict queued lookahead to make room for the head. The
        reclaimable-cache targets reuse the arbiter's release-cache selection (idle processes whose reservation
        exceeds allocation by the release threshold, holding no model). Lane and safety candidates are included
        only while their context is on the GPU, in the fixed pause order the ladder escalates through.
        Promised-free figures are the tenants' measured reservations where known; lane pauses additionally
        charge each stopped process's CUDA-context constant, since stopping the process is what returns that
        VRAM. Verification compares realized frees against these expectations.
        """
        now_monotonic = time.monotonic()
        now_wall = time.time()
        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress}
        idle_residents: list[IdleResidentModel] = []
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.is_process_busy() or process_info.loaded_horde_model_name is None:
                continue
            if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                # An unload is already outstanding on this slot, so :meth:`unload_idle_model` would refuse it
                # and the rung would be a same-tick no-op the engine advances straight past. Leaving such a slot
                # in the ladder spends the cheap resident rungs on nothing and opens every later episode
                # directly onto the lane and safety rungs.
                continue
            if (
                process_info.loaded_horde_model_name in in_progress_models
                or process_info.loaded_horde_model_name in protected_models
            ):
                continue
            footprint_mb = (
                float(process_info.process_reserved_mb)
                if process_info.process_reserved_mb
                else self._idle_resident_footprint_mb(process_info, process_info.loaded_horde_model_name)
            )
            idle_residents.append(
                IdleResidentModel(
                    process_id=process_info.process_id,
                    tenant_label=process_info.loaded_horde_model_name,
                    materialized_monotonic=self._reclaim_recency_key(process_info, now_monotonic, now_wall),
                    footprint_mb=footprint_mb,
                ),
            )

        idle_cache_ids, _busy = self._gpu_process_activity_ids(device_index)
        cache_targets: list[CacheReleaseTarget] = []
        for process_id in idle_cache_ids:
            process_info = self._process_map.get(process_id)
            if process_info is None or process_info.process_reserved_mb is None:
                continue
            reclaimable_mb = float(process_info.process_reserved_mb) - float(process_info.process_allocated_mb or 0)
            cache_targets.append(
                CacheReleaseTarget(
                    process_id=process_id,
                    tenant_label=f"{process_info.process_type.name.lower()}#{process_id}",
                    materialized_monotonic=self._reclaim_recency_key(process_info, now_monotonic, now_wall),
                    reclaimable_mb=reclaimable_mb,
                ),
            )

        lanes = self._reclaim_lane_candidates(device_index)
        safety = self._reclaim_safety_candidate(device_index)
        return LadderCandidates(
            device_index=device_index,
            idle_residents=tuple(idle_residents),
            cache_targets=tuple(cache_targets),
            lanes=lanes,
            safety=safety,
        )

    def _idle_resident_footprint_mb(self, process_info: HordeProcessInfo, model_name: str) -> float:
        """What a slot holding ``model_name`` costs the card, for a slot whose reservation is unreported.

        The ladder grades a rung against what it promised and scales its verification budget by that promise, so
        an idle resident priced at zero is graded on the base allowance alone and its release is escalated past
        while it is still arriving. A model's resident cost is knowable independently of whether the slot's
        allocator reservation has been reported and independently of whether the slot is retention-tracked: the
        measured per-checkpoint watermark where one exists, otherwise the same static resident-footprint
        estimate :meth:`_retained_resident_footprint_mb` charges a tracked resident at.

        Returns zero only for a genuinely unknown footprint (no watermark, no job to key the estimate on, or no
        baseline), which leaves the unverifiable-rung path to grade it on an honest absence of evidence.
        """
        baseline = self._model_metadata.get_baseline(model_name)
        learned_mb = self._learned_resident_footprint_mb(model_name, baseline)
        if learned_mb is not None:
            return learned_mb
        job = process_info.last_job_referenced
        if job is None:
            return 0.0
        return predict_job_footprint_mb(job, baseline) or 0.0

    @staticmethod
    def _reclaim_recency_key(process_info: HordeProcessInfo, now_monotonic: float, now_wall: float) -> float:
        """Return a monotonic-scale recency key for LIFO reclaim ranking of a process.

        Prefers the dedicated ``vram_materialized_monotonic`` stamp (set when the parent observed the process
        materialize VRAM). When that is unset (an older child, or a process that has not materialized since
        start) it falls back to the report-time proxy, mapped onto the monotonic timeline
        (``now_monotonic - (now_wall - last_received_timestamp)``) so stamped and unstamped processes remain
        comparable in one ranking rather than one scale sorting entirely above the other.
        """
        if process_info.vram_materialized_monotonic is not None:
            return process_info.vram_materialized_monotonic
        return now_monotonic - (now_wall - process_info.last_received_timestamp)

    def _post_processing_lane_has_committed_work(self) -> bool:
        """Return true if the shared post-processing lane has queued or active work.

        Image post-processing lives in JobTracker. Graph-backed alchemy shares the same child process but
        owns its queue in AlchemyCoordinator, so the manager wires that count in through a provider.
        """
        if self._job_tracker.jobs_pending_post_processing or self._job_tracker.jobs_being_post_processed:
            return True
        try:
            return self._post_processing_lane_commitments_provider() > 0
        except Exception:
            logger.exception("Failed to read post-processing lane commitments; preserving the lane this cycle")
            return True

    def _has_idle_post_process_process_for_reclaim(self, device_index: int | None) -> bool:
        """Return true when a post-processing process is live, idle, and on the requested card."""
        return self._has_idle_service_lane_for_reclaim(HordeProcessType.POST_PROCESS, device_index)

    def _has_idle_service_lane_for_reclaim(
        self,
        process_type: HordeProcessType,
        device_index: int | None,
    ) -> bool:
        """Return whether a live service lane of ``process_type`` is idle on the requested card.

        Lane teardown returns a real CUDA context but is more disruptive than releasing cache or weights, so
        both candidate-specific PP reclaim and the verified saturation ladder use this same last-moment liveness
        check. A lane that is encoding, decoding, or post-processing is never offered as a reclaim target.
        """
        for process_info in self._process_map.values():
            if process_info.process_type != process_type:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.can_accept_job():
                return True
        return False

    def _reclaim_lane_candidates(self, device_index: int | None) -> tuple[LaneReclaimCandidate, ...]:
        """Build the lane-pause rungs in fixed escalation order for lanes currently on the GPU."""
        lifecycle = self._process_lifecycle
        bridge_data = self._runtime_config.bridge_data
        lanes: list[LaneReclaimCandidate] = []
        if (
            bridge_data.allow_post_processing
            and bridge_data.post_processing_lane_enabled
            and not lifecycle.is_post_process_gpu_paused
            and not self._post_processing_lane_has_committed_work()
            and self._has_idle_post_process_process_for_reclaim(device_index)
        ):
            lanes.append(
                LaneReclaimCandidate(
                    kind=ReclaimRungKind.PAUSE_PP_LANE,
                    tenant_label="post-processing lane",
                    promised_mb=self._lane_promised_free_mb(HordeProcessType.POST_PROCESS, device_index),
                ),
            )
        if (
            lifecycle.vae_lane_enabled()
            and not lifecycle.is_vae_lane_gpu_paused
            and self._process_map.num_vae_lane_processes(device_index=device_index) > 0
            and self._has_idle_service_lane_for_reclaim(HordeProcessType.VAE_LANE, device_index)
        ):
            lanes.append(
                LaneReclaimCandidate(
                    kind=ReclaimRungKind.PAUSE_VAE_LANE,
                    tenant_label="VAE lane",
                    promised_mb=self._lane_promised_free_mb(HordeProcessType.VAE_LANE, device_index),
                ),
            )
        if (
            lifecycle.component_lane_enabled()
            and not lifecycle.is_component_gpu_paused
            and self._process_map.num_component_processes(device_index=device_index) > 0
            and self._has_idle_service_lane_for_reclaim(HordeProcessType.COMPONENT, device_index)
        ):
            lanes.append(
                LaneReclaimCandidate(
                    kind=ReclaimRungKind.PAUSE_COMPONENT_LANE,
                    tenant_label="component lane",
                    promised_mb=self._lane_promised_free_mb(HordeProcessType.COMPONENT, device_index),
                ),
            )
        return tuple(lanes)

    def _reclaim_safety_candidate(self, device_index: int | None) -> LaneReclaimCandidate | None:
        """Build the safety-off-GPU rung when the operator allows safety to leave the GPU."""
        if (
            not self._safety_on_gpu_permitted
            or not self._runtime_config.bridge_data.whole_card_residency_safety_off_gpu
            or self._process_lifecycle.is_safety_gpu_paused
            or self._process_lifecycle.safety_placement_transition_pending is True
        ):
            return None
        reserved_mb = self._reserved_mb_for_type(HordeProcessType.SAFETY, device_index)
        return LaneReclaimCandidate(
            kind=ReclaimRungKind.SAFETY_OFF_GPU,
            tenant_label="safety",
            promised_mb=reserved_mb if reserved_mb > 0 else self._safety_footprint_mb(),
        )

    def _lane_promised_free_mb(self, process_type: HordeProcessType, device_index: int | None) -> float:
        """Price a lane pause's promised free as each live process's context charge plus its reservation.

        A lane pause stops the lane's process off the GPU, so the give-back is the process's full device
        footprint (``context_constant + process_reserved_mb``), not the allocator reservation alone: an idle
        lane whose allocator is empty still returns its CUDA context. Priced with the same resolved context
        constant the committed-VRAM ledger charges, so a lane rung never promises zero while a real context
        give-back stands behind it; where the driver has already demoted the context, the shortfall surfaces
        through the ladder's promised-vs-realized verification instead of a silent zero promise.
        """
        context_constant_mb = self.resolved_context_constant_mb()
        total_mb = 0.0
        for process_info in self._process_map.values():
            if process_info.process_type != process_type:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            total_mb += context_constant_mb + float(process_info.process_reserved_mb or 0)
        return total_mb

    def _reserved_mb_for_type(self, process_type: HordeProcessType, device_index: int | None) -> float:
        """Sum the measured device reservation (MB) of a process type's live processes on a card."""
        total_mb = 0.0
        for process_info in self._process_map.values():
            if process_info.process_type != process_type:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.process_reserved_mb is not None:
                total_mb += float(process_info.process_reserved_mb)
        return total_mb

    def unload_idle_model(self, process_id: int, device_index: int | None = None) -> bool:
        """Unload one idle process's resident model from VRAM to RAM (reclaim-ladder actuator).

        Targets a single named process rather than sweeping the card, so the verified ladder controls exactly
        which resident it gives back and in what order. Never touches an actively-sampling process, and treats
        a process already unloading (or without a resident model) as a no-op so the engine does not open a
        verification window on a rung that frees nothing.
        """
        process_info = self._process_map.get(process_id)
        if process_info is None:
            return False
        if process_info.is_process_busy() and not self._slot_is_reclaimable_while_busy(process_info):
            return False
        if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
            return False
        model_name = process_info.loaded_horde_model_name
        if model_name is None and not process_info.held_components:
            return False
        # A lane holding component-cache entries but no tracked checkpoint is still holding the card, and it
        # is precisely the tenancy no job boundary returns. Refusing it here made this rung a no-op against
        # the one holder a starved head cannot outwait, so the whole-slot unload is issued instead: the child
        # frees what it holds, named model or not.
        message: HordeControlMessage = (
            HordeControlMessage(control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM)
            if model_name is None
            else HordeControlModelMessage(
                control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM,
                horde_model_name=model_name,
            )
        )
        if not process_info.safe_send_message(message):
            return False
        process_info.clear_job_references()
        self._note_retention_evicted_unused(process_info)
        process_info.clear_retained_resident()
        process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
        self._record_churn("vram_eviction")
        logger.info(
            f"Reclaim ladder: unloading {model_name or 'the held component tenancy'} from VRAM on idle "
            f"process {process_id}",
        )
        return True

    def _slot_is_reclaimable_while_busy(self, process_info: HordeProcessInfo) -> bool:
        """Whether a slot the busy predicate covers may still have its residency reclaimed.

        The one case: a slot parked on a completed preload whose dispatch never came. It stays busy so no
        dispatch races it, and past the retention staleness horizon (the same falsified-prediction horizon a
        retained copy is judged on: a preload and a retention both assert an imminent same-slot job) its
        weights are the ladder's to take back.
        """
        return process_info.is_parked_preload(
            now=self._clock(),
            dwell_seconds=_RETENTION_STALE_HOLD_SECONDS,
        )

    def evict_retained_resident_for_model_change(
        self,
        process_info: HordeProcessInfo,
        dispatched_model: str,
    ) -> bool:
        """Return a slot's retained weights to the card before it loads a different model; True if issued.

        Retention leaves a model on the device with no pending eviction, so a dispatch for a different
        model onto the same slot would materialise the new weights beside the old ones and leave the card
        carrying both. The unload is issued on the same pipe ahead of START_INFERENCE, so the child frees
        before it loads, and the parent's residency record is cleared with it. This is deliberately not
        left to the child's own free-view: under WDDM that view is untruthful in exactly the regime the
        double residency creates.

        A same-model dispatch is the retention case proper and is left alone: those weights are what the
        successor reuses, which is the whole saving.
        """
        retained_model = process_info.retained_resident_model
        if retained_model is None or retained_model == dispatched_model:
            return False
        if not process_info.safe_send_message(
            HordeControlModelMessage(
                control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM,
                horde_model_name=retained_model,
            ),
        ):
            return False
        self._note_retention_evicted_unused(process_info)
        process_info.clear_retained_resident()
        self._record_churn("vram_eviction")
        logger.info(
            f"Evicting retained model {retained_model} from VRAM on process {process_info.process_id} "
            f"before it loads {dispatched_model}",
        )
        return True

    def _retained_resident_dispatch_holds(
        self,
        next_job: ImageGenerateJobPopResponse,
        process_with_model: HordeProcessInfo,
    ) -> bool:
        """Whether this dispatch must wait for sibling retained weights to leave the card before it loads.

        A dispatch that materialises weights is priced against every retained resident the card carries, not
        just against the slot it lands on: weights another slot holds across jobs are as real a tenant as a
        sampling peak, and nothing else in the dispatch path consults them. Where the load does not fit beside
        them, the idle ones are evicted through the reclaim actuator and the dispatch holds its queue position
        until the child's own reports evidence the room is back. Holding on that evidence rather than on the
        eviction being *issued* is the point: the tracking clears the moment the unload is sent, so a dispatch
        that trusted it would load into memory still occupied by the copy it just asked for back.

        The slot's own retained weights are not charged here: a cross-model dispatch onto a retaining slot is
        already evicted ahead of its START_INFERENCE (:meth:`evict_retained_resident_for_model_change`), so
        charging them would hold a dispatch against a tenant that leaves before the load. The two paths act on
        disjoint slots and never evict the same weights twice.

        A dispatch onto the slot that already retains this model loads nothing and is never held.
        """
        device_index = process_with_model.device_index if self._multi_gpu_routing_active else None
        self._prune_confirmed_retention_evictions()
        if not self._retained_resident_hold_applies(next_job, process_with_model):
            return False
        if self._retention_eviction_pending(device_index):
            self._note_retention_dispatch_hold(next_job, reclaiming=True)
            return True
        reclaiming = self._evict_retained_residents_for_dispatch(process_with_model, device_index=device_index)
        self._note_retention_dispatch_hold(next_job, reclaiming=reclaiming)
        return True

    def _retained_resident_hold_applies(
        self,
        next_job: ImageGenerateJobPopResponse,
        process_with_model: HordeProcessInfo,
    ) -> bool:
        """Whether this dispatch cannot yet load beside the card's retained residents. Read-only.

        The condition :meth:`_retained_resident_dispatch_holds` acts on, without the actuation: an eviction
        this path issued that the card has not evidenced yet, or a load that does not fit beside the weights
        sibling slots hold across jobs. The stall diagnostic reads it to name the wait, so the gate and the
        attribution can never disagree about whether a head is parked here, and naming a wait never evicts
        anything.
        """
        model = next_job.model
        if model is None:
            return False
        device_index = process_with_model.device_index if self._multi_gpu_routing_active else None
        if self._retention_eviction_pending(device_index):
            return True
        if process_with_model.retained_resident_model == model:
            return False
        if not self._card_has_sibling_retained_resident(process_with_model, device_index=device_index):
            # No weights are being held across jobs on this card, so this gate has nothing to price. Every
            # other admission concern belongs to the gates that already ran.
            return False
        return (
            self._fits_beside_retained_residents(
                next_job,
                target=process_with_model,
                device_index=device_index,
                include_target_retained=False,
            )
            is False
        )

    def _card_has_sibling_retained_resident(
        self,
        target: HordeProcessInfo,
        *,
        device_index: int | None,
    ) -> bool:
        """Whether a slot other than ``target`` is holding weights on this card between jobs."""
        for process_info in self._process_map.values():
            if process_info.process_id == target.process_id:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.retained_resident_model is not None:
                return True
        return False

    def _evict_retained_residents_for_dispatch(
        self,
        target: HordeProcessInfo,
        *,
        device_index: int | None,
    ) -> bool:
        """Return idle siblings' retained weights to the card for ``target``'s load; True if any unload issued.

        Routed through the reclaim ladder's own single-slot actuator, so the eviction the dispatch needs is
        the same actuation every other owner performs. A retainer that is busy is left alone: its weights go
        back when its job ends, and the dispatch simply keeps holding until they do.
        """
        issued = False
        for process_info in list(self._process_map.values()):
            if process_info.process_id == target.process_id:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if self._issue_retention_eviction(process_info, device_index=device_index):
                issued = True
        return issued

    def _issue_retention_eviction(self, process_info: HordeProcessInfo, *, device_index: int | None) -> bool:
        """Ask one idle slot's retained weights back through the reclaim actuator, tracked until evidenced.

        The single seam every owner of a retention eviction goes through, so the actuation and its in-flight
        record can never come apart. Recording the pending eviction is not bookkeeping: the residency tracking
        clears the moment the unload is *sent*, so anything reading that tracking would believe the room is
        already back. The record is what holds a dispatch on this card until the child's own reports evidence
        the free (:meth:`_prune_confirmed_retention_evictions`), and it is owed by every path that issues one,
        not only by the dispatch that made room for itself.

        A busy retainer is left alone: its weights go back when its job ends.
        """
        retained_model = process_info.retained_resident_model
        if retained_model is None or not process_info.can_accept_job():
            return False
        reserved_baseline_mb = (
            float(process_info.process_reserved_mb) if process_info.process_reserved_mb is not None else None
        )
        if not self.unload_idle_model(process_info.process_id, device_index):
            return False
        self._pending_retention_evictions[process_info.process_id] = _PendingRetentionEviction(
            model=retained_model,
            reserved_baseline_mb=reserved_baseline_mb,
            device_free_baseline_mb=self._measured_free_vram_mb(device_index=device_index),
        )
        return True

    def _revoke_stale_retentions_under_pressure(self, device_index: int) -> None:
        """Give back this card's retained weights whose predicted successor never turned up.

        A grant is issued on the evidence standing at one dispatch, and before this nothing afterwards re-asked
        the question: only an eviction actuation could end a retention, so a hold taken during a healthy moment
        survived every subsequent change in what the slot was actually being asked to run.

        Asking the *issuance* question again is not what re-opens it, and this is worth stating because it is
        the obvious design and it does not work: a live grant's own dispatch heads the slot's history, so the
        window the sweep would read is the window that issued the grant, and every live retention passes by
        construction. What can refute the prediction is the thing it predicted failing to happen. A hold that
        has gone unreused past :data:`_RETENTION_STALE_HOLD_SECONDS` has been falsified by events, whatever the
        history that issued it says, and on a card that has since gone and stayed off HEALTHY it is the
        cheapest thing on that card to give back.

        A hold the traffic is still coming back for is never touched, however long the pressure lasts, because
        each reuse ends its episode and starts a fresh one: a pool-locked slot's age therefore resets every job
        and can never reach the horizon. This is deliberately not a second reclaim ladder. Genuine saturation
        is the verified ladder's to resolve, and retained residents are already first-class candidates for it;
        this only removes holds that had stopped being a bet on anything.

        Actuation is the ordinary idle-model unload, tracked as an in-flight eviction like any other, so a
        dispatch priced against these weights waits for the card to evidence the free rather than for the
        request to have been sent. Busy slots are never touched.
        """
        scoped_device_index = device_index if self._multi_gpu_routing_active else None
        for process_info in list(self._process_map.values()):
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if scoped_device_index is not None and process_info.device_index != scoped_device_index:
                continue
            retained_model = process_info.retained_resident_model
            if retained_model is None or not self._retention_hold_is_stale(process_info):
                continue
            if not self._issue_retention_eviction(process_info, device_index=scoped_device_index):
                continue
            self._retention_revokes += 1
            logger.info(
                f"Revoking the retained copy of {retained_model} on process {process_info.process_id}: the card "
                f"has been off HEALTHY for {_RETENTION_PRESSURE_REVOKE_SECONDS:.0f}s and no job has come back "
                f"for those weights in {_RETENTION_STALE_HOLD_SECONDS:.0f}s.",
            )

    def _prune_confirmed_retention_evictions(self) -> None:
        """Drop the pending retention evictions the card has evidenced, leaving only the unlanded ones.

        A child frees first and reports after: it sends a fresh memory report as soon as its allocator
        reservation falls, and a model-state change moving the model out of device residency. A risen device
        free reading, a fallen slot reservation, the map no longer placing those weights on that slot, the
        slot no longer naming the model, or the slot being gone all evidence that the room is back.

        Absent every one of them the record is kept for
        :data:`_RETENTION_EVICTION_CONFIRMATION_PASSES` passes and then dropped: evidence is preferred, but a
        child whose reports never arrive must not park the queue on evidence that is not coming.
        """
        for process_id, pending in list(self._pending_retention_evictions.items()):
            process_info = self._process_map.get(process_id)
            if process_info is None or process_info.loaded_horde_model_name != pending.model:
                del self._pending_retention_evictions[process_id]
                continue
            reserved_mb = process_info.process_reserved_mb
            if (
                pending.reserved_baseline_mb is not None
                and reserved_mb is not None
                and float(reserved_mb) < pending.reserved_baseline_mb
            ):
                del self._pending_retention_evictions[process_id]
                continue
            device_free_mb = self._measured_free_vram_mb(
                device_index=process_info.device_index if self._multi_gpu_routing_active else None,
            )
            if (
                pending.device_free_baseline_mb is not None
                and device_free_mb is not None
                and device_free_mb > pending.device_free_baseline_mb
            ):
                del self._pending_retention_evictions[process_id]
                continue
            model_info = self._horde_model_map.root.get(pending.model)
            map_places_weights_here = (
                model_info is not None
                and model_info.process_id == process_id
                and model_info.horde_model_load_state
                in (
                    ModelLoadState.LOADED_IN_VRAM,
                    ModelLoadState.IN_USE,
                )
            )
            if not map_places_weights_here:
                del self._pending_retention_evictions[process_id]
                continue
            pending.passes_waited += 1
            if pending.passes_waited >= _RETENTION_EVICTION_CONFIRMATION_PASSES:
                logger.debug(
                    f"Retention eviction of {pending.model} on process {process_id} went unevidenced for "
                    f"{pending.passes_waited} passes; the dispatch waiting on it now stands on the measured "
                    "admission gate alone.",
                )
                del self._pending_retention_evictions[process_id]

    def _retention_eviction_pending(self, device_index: int | None) -> bool:
        """Whether a retention eviction issued for this card has not yet been evidenced at the device."""
        for process_id in self._pending_retention_evictions:
            process_info = self._process_map.get(process_id)
            if process_info is None:
                continue
            if device_index is None or process_info.device_index == device_index:
                return True
        return False

    def _note_retention_dispatch_hold(self, next_job: ImageGenerateJobPopResponse, *, reclaiming: bool) -> None:
        """Disclose (throttled) that a dispatch is waiting for retained weights to come back off the card."""
        suppressed = self._scheduler_diagnostic_suppressed_count(
            "retained_resident_dispatch_hold",
            (str(next_job.id_), reclaiming),
        )
        if suppressed is None:
            return
        detail = (
            "idle retained weights are being returned to the card"
            if reclaiming
            else "waiting for the retained weights already asked back to leave the device"
        )
        logger.opt(colors=True).info(
            "<fg #f0beff>Holding dispatch of {} until the card's retained residents fit beside it: {}.</>",
            next_job.model,
            detail,
        )

    def reclaim_idle_resident_for_post_processing(self, *, device_index: int | None = None) -> str | None:
        """Unload the coldest idle resident model's VRAM to make room for a post-processing peak; return its name.

        Picks the least-recently-demanded idle inference resident whose model no pending or in-progress job
        needs, and unloads its weights to RAM (the RAM copy is retained, so a later job re-stages it cheaply).
        Returns the unloaded model's name, or None when no such momentarily-idle resident exists (every resident
        is busy, is already unloading, or is a queued/in-progress job's model, so yielding one would only force
        an immediate reload).

        ``device_index`` scopes the search to one card on a multi-GPU host; None considers every card.
        """
        pending_models = {job.model for job in self._job_tracker.jobs_pending_inference}
        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress}
        now_monotonic = time.monotonic()
        now_wall = time.time()

        coldest_process_id: int | None = None
        coldest_model_name: str | None = None
        coldest_recency_key = float("inf")
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.is_process_busy() or process_info.loaded_horde_model_name is None:
                continue
            model_name = process_info.loaded_horde_model_name
            if model_name in pending_models or model_name in in_progress_models:
                continue
            if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                continue
            recency_key = self._reclaim_recency_key(process_info, now_monotonic, now_wall)
            if recency_key < coldest_recency_key:
                coldest_recency_key = recency_key
                coldest_process_id = process_info.process_id
                coldest_model_name = model_name

        if coldest_process_id is None:
            return None
        if not self.unload_idle_model(coldest_process_id, device_index=device_index):
            return None
        return coldest_model_name

    def release_idle_cache(self, process_id: int) -> bool:
        """Release an idle process's reclaimable allocator cache back to the card (reclaim-ladder actuator)."""
        return self.release_allocator_cache(process_id)

    def pause_post_process_lane(self, device_index: int | None) -> bool:
        """Pause the post-processing lane off the GPU to reclaim its context (reclaim-ladder actuator)."""
        return self._note_lane_cycle(
            self._process_lifecycle.pause_post_process_off_gpu(owner=PauseOwner.RECLAIM_LADDER)
        )

    def _vae_lane_pause_deferred_for_decode(self, *, requester: VaeLanePauseRequester) -> bool:
        """Whether a VAE-lane pause must be withheld because disaggregated work still needs the lane.

        Pausing the lane out from under a queued or in-flight decode strands a job whose sampling already
        finished: it reroutes monolithic and discards that sampling, to free room for a dispatch the decode
        itself clears within seconds. Every path that stops this lane asks here first (the reclaim ladder's
        actuator, which both the governor's saturation rung and a post-processing borrow route through, and the
        whole-card residency's establish and converge passes), so the eligibility rule is stated once and each
        caller simply treats the pause as not eligible this tick.

        ``requester`` selects how wide "still needs the lane" is. The reclaim ladder is buying VRAM relief now,
        so only a queued or in-flight decode outweighs it. A whole-card residency is pre-staging a head that
        cannot claim the card until the in-flight work drains regardless, so it also waits on a job that is
        merely sampling: pausing under one frees the lane's context seconds before the residency could use it
        and discards the sample the short decode hold cannot outlast.

        Emits one edge-latched INFO per deferral episode naming ``requester`` and the blocking count, re-armed
        once the work has drained, so a per-cycle caller cannot spam the log.

        Args:
            requester: The subsystem asking to pause the lane; names it in the log and picks the count.

        Returns:
            True when the pause must be withheld this tick.
        """
        whole_card = requester is VaeLanePauseRequester.WHOLE_CARD_RESIDENCY
        blocking = self._vae_lane_bound_job_count() if whole_card else self._vae_decode_pending_count()
        if blocking <= 0:
            self._vae_pause_deferred_for_decode = False
            return False
        if not self._vae_pause_deferred_for_decode:
            self._vae_pause_deferred_for_decode = True
            waiting_on = (
                f"{blocking} disaggregated job(s) still bound for the VAE lane"
                if whole_card
                else f"{blocking} disaggregated decode(s) are queued or in flight on it"
            )
            logger.info(
                f"{requester.value}: holding the VAE-lane pause while {waiting_on}; the lane frees itself "
                "sooner than the monolithic reroute a pause would force, which discards the finished sampling.",
            )
        return True

    def pause_vae_lane(self, device_index: int | None) -> bool:
        """Pause the VAE lane off the GPU to reclaim its context (reclaim-ladder actuator).

        Reports no-op (does not pause) while a disaggregated decode still needs the lane
        (:meth:`_vae_lane_pause_deferred_for_decode`). Both reclaim paths that stop this lane (the governor's
        saturation rung and a post-processing borrow) route through here, so a no-op makes each move to its next
        relief option exactly as any rung whose target has gone away does.
        """
        if self._vae_lane_pause_deferred_for_decode(requester=VaeLanePauseRequester.RECLAIM_LADDER):
            return False
        return self._note_lane_cycle(self._process_lifecycle.pause_vae_lane_off_gpu(owner=PauseOwner.RECLAIM_LADDER))

    def pause_component_lane(self, device_index: int | None) -> bool:
        """Pause the component lane off the GPU to reclaim its context (reclaim-ladder actuator)."""
        return self._note_lane_cycle(self._process_lifecycle.pause_component_off_gpu(owner=PauseOwner.RECLAIM_LADDER))

    def _note_lane_cycle(self, paused: bool) -> bool:
        """Count a service-lane pause that acted as churn, passing the actuator's answer through.

        Only a pause that acted costs anything: the lane is stopped now and cold-starts when it is restored,
        so the pause is the countable event and its restore is the second half of the same cycle.
        """
        if paused:
            self._record_churn("lane_cycle")
        return paused

    def safety_off_gpu(self, device_index: int | None) -> bool:
        """Move the on-GPU safety context off the card to reclaim it (reclaim-ladder actuator).

        Unlike lane pauses, this files a one-shot request with the recurring safety-placement reconciler. The
        reconciler applies it immediately without advancing fit hysteresis, then consumes it only after the
        off-GPU child reaches readiness. Restoration is also reconciled there, so the ladder cannot overlap a
        residency or runtime-policy transition.
        """
        if (
            self._safety_reclaim_pause_requested
            or self._process_lifecycle.is_safety_gpu_paused
            or self._process_lifecycle.safety_placement_transition_pending is True
        ):
            return False
        self._safety_reclaim_pause_requested = True
        self._reconcile_runtime_safety_placement(update_policy=False)
        return True

    def restore_post_process_lane(self, device_index: int | None) -> bool:
        """Restart a ladder-paused post-processing lane once the card has recovered (reclaim-ladder actuator)."""
        return self._process_lifecycle.restore_post_process_off_gpu(owner=PauseOwner.RECLAIM_LADDER)

    def restore_vae_lane(self, device_index: int | None) -> bool:
        """Restart a ladder-paused VAE lane once the card has recovered (reclaim-ladder actuator)."""
        return self._process_lifecycle.restore_vae_lane_off_gpu(owner=PauseOwner.RECLAIM_LADDER)

    def restore_component_lane(self, device_index: int | None) -> bool:
        """Restart a ladder-paused component lane once the card has recovered (reclaim-ladder actuator)."""
        return self._process_lifecycle.restore_component_off_gpu(owner=PauseOwner.RECLAIM_LADDER)

    def record_calibration_event(self, rung: ReclaimRung, *, promised_mb: float, realized_mb: float) -> None:
        """Record a verified reclaim shortfall as a calibration counter (reclaim-ladder actuator).

        The freed figure is not a footprint peak and no complete footprint key is reconstructable at reclaim
        time, so the raise-only footprint store does not apply; the signal is kept as a count for calibration
        visibility that a rung's promised free over-stated what the hardware returned.
        """
        self._reclaim_calibration_events += 1

    def preload_models(self) -> bool:
        """Preload models that are likely to be used soon.

        Housekeeping first (whole-card residency restore/convergence, stale model-map expiry, clearing
        preloads the queue no longer needs), then one pass over the pending queue: each job runs through
        the admission pipeline (:meth:`_attempt_preload_for_job`) until one preloads or a gate stops the
        pass for this cycle.

        Returns:
            True if a model was preloaded, False otherwise.
        """
        self._restore_siblings_after_whole_card()
        self._converge_whole_card_residency()
        self._reconcile_runtime_safety_placement()
        self._expire_stale_model_map_entries()

        if self._pending_post_processing_should_hold_preload():
            return False

        loaded_models = {process.loaded_horde_model_name for process in self._process_map.values()}
        loaded_models = loaded_models.union(
            model.horde_model_name
            for model in self._horde_model_map.root.values()
            if model.horde_model_load_state.is_loaded() or model.horde_model_load_state == ModelLoadState.LOADING
        )

        pending_models = {job.model for job in self._job_tracker.jobs_pending_inference}
        for process in self._process_map.values():
            if (
                process.last_process_state == HordeProcessState.PRELOADED_MODEL
                and process.loaded_horde_model_name not in pending_models
            ):
                logger.debug(
                    f"Clearing preloaded model {process.loaded_horde_model_name} "
                    f"from process {process.process_id} as it is no longer needed",
                )
                self._process_map.on_process_state_change(
                    process_id=process.process_id,
                    new_state=HordeProcessState.WAITING_FOR_JOB,
                )

        # The fast path out of the pass: every pending job's model is already accounted for. Set equality
        # alone is not enough on a multi-GPU host, where a model can be loaded on a card that cannot serve the
        # job that wants it, so each pending job must find its copy on a card eligible for it.
        if loaded_models == pending_models and all(
            self._model_loaded_for_job(job, loaded_models) for job in self._job_tracker.jobs_pending_inference
        ):
            return False

        # The first queued job not already in progress is the head of the queue. Only when *its* model
        # is the one that cannot be loaded may the budget gate escalate to evicting another queued
        # model (see the budget-defer branches in the admission pipeline); a later job whose turn has
        # not come never displaces a resident head.
        in_progress_jobs = self._job_tracker.jobs_in_progress
        # An aux-unprepared job never anchors preload as the priority head: it holds no sampling reservation
        # and nothing prices around it, so the first job eligible to hold capacity is the first pending one that
        # is both not in progress and not awaiting auxiliary preparation. A fitting sibling behind a gated job
        # is thus the head that may escalate eviction to become resident. Only when no such sibling exists may
        # the gated job take the strictly non-displacing, empty-slot preload path below.
        pending = self.pending_inference_in_placement_order()
        head_job = next(
            (j for j in pending if j not in in_progress_jobs and not self._job_requires_aux_preparation(j)),
            None,
        )
        self._update_head_starvation_timer(head_job)
        self._reconcile_head_priority_barrier(head_job)
        if self._resident_head_should_dispatch_before_preload(head_job):
            return False

        for job in pending:
            outcome = self._attempt_preload_for_job(
                job,
                head_job=head_job,
                loaded_models=loaded_models,
            )
            if outcome is _PreloadJobOutcome.NEXT_JOB:
                continue
            return outcome is _PreloadJobOutcome.PRELOAD_SENT

        return False

    def _pending_post_processing_should_hold_preload(self) -> bool:
        """Whether a pending post-processing chain should receive the next drain window before preloads."""
        return self._pending_post_processing_reserve_mb(device_index=None) > 0

    def _resident_head_should_dispatch_before_preload(self, head_job: ImageGenerateJobPopResponse | None) -> bool:
        """Whether the queue head can already try dispatch, so speculative preloading should yield."""
        return head_job is not None and self._job_dispatchable_now(head_job)

    def _job_dispatchable_now(self, job: ImageGenerateJobPopResponse) -> bool:
        """Whether ``job`` could be dispatched this cycle: its model is resident on a free slot under the cap.

        The question the preload pass asks before spending a cycle on a speculative load, of the head and of a
        retention-affinity candidate alike: dispatch beats staging whenever it is available, so both callers
        need the same answer and neither may reach a different one.
        """
        if job.model is None:
            return False
        if self._process_lifecycle.is_model_load_quarantined(job.model):
            return False

        process_with_model = self._resident_process_for_job(job)
        if process_with_model is None or not process_with_model.can_accept_job():
            return False

        target_card: CardRuntime | None = None
        if self._multi_gpu_routing_active:
            candidate_card = self._card_runtimes.get(process_with_model.device_index)
            if candidate_card is not None and (
                self._job_tracker.is_admitted_exclusive(job)
                or not self._job_tracker.has_exclusive_job_in_progress(candidate_card.device_index)
            ):
                target_card = candidate_card

        if target_card is not None:
            jobs_in_progress_count = len(self._jobs_in_progress_on_card(target_card.device_index))
        else:
            jobs_in_progress_count = len(self._job_tracker.jobs_in_progress)
        return jobs_in_progress_count < self._max_jobs_in_progress_allowed(card=target_card)

    def pending_inference_in_placement_order(self) -> tuple[ImageGenerateJobPopResponse, ...]:
        """The pending-inference queue in the order the scheduler places work from, retained copies first.

        The queue's own order is FIFO, and on a model rotation wider than the lane pool that puts a cold job at
        the head while a slot sits idle holding the retained weights of another model still queued. The cold head
        is then loaded onto whichever slot is free, which is routinely that retainer: the weights a queued job was
        about to reuse are evicted, that job re-uploads them onto another slot, and two models trade lanes for the
        rest of a session at a full host-to-device upload per job.

        This moves the jobs a retainer can already serve ahead of the head, keeping their relative order and the
        relative order of everything behind them. Every consumer of this ordering then reaches the same
        conclusion without being told about retention: the preload pass takes the promoted job as its head and so
        yields to dispatch, dispatch seats it on the retainer at no upload, a disaggregation-class promoted job
        goes to the pipeline exactly as it would from the head, and a preload that does run targets the lane its
        own job needs. Nothing is surrendered and no load is withheld, which is what keeps the pass from trading
        a cycle for a cycle.

        A read-only view, deliberately: the tracker owns queue order, and every other reader of
        ``jobs_pending_inference`` (age accounting, counts, telemetry, the submit path) means the order work
        arrived in. Only the two placement sites consume this.

        The bounds are the displaced head's own, unchanged: no reorder happens once it has aged past the
        anti-starvation fraction of its ttl, and none once it has been passed its skip ceiling. The ceiling is
        the operative one, because a reorder that seats a job *is* a committed pass of the head and advances that
        budget exactly as a line-skip does, so a head is passed a bounded number of times whatever its ttl.
        """
        pending = self._job_tracker.jobs_pending_inference
        head_job = next(
            (
                j
                for j in pending
                if j not in self._job_tracker.jobs_in_progress and not self._job_requires_aux_preparation(j)
            ),
            None,
        )
        promoted = [job for job, _retainer in self._retention_affinity_candidates(head_job)]
        if not promoted:
            return pending
        promoted_ids = {id(job) for job in promoted}
        return (*promoted, *(job for job in pending if id(job) not in promoted_ids))

    def _reordered_head_displaced_by(
        self,
        dispatched_job: ImageGenerateJobPopResponse,
    ) -> ImageGenerateJobPopResponse | None:
        """The queue head this dispatch was seated ahead of by the placement order, or None.

        Read at the dispatch commit, where the dispatched job is still pending, so the queue's own head is still
        resolvable. Selection reaches a job that is not that head only through the placement order (a line-skip
        is the other way past a head, and its callers handle it separately), so a mismatch here identifies a
        reorder without the selection path having to report one.
        """
        fifo_head = next(
            (
                job
                for job in self._job_tracker.jobs_pending_inference
                if job not in self._job_tracker.jobs_in_progress and not self._job_requires_aux_preparation(job)
            ),
            None,
        )
        if fifo_head is None or fifo_head.id_ == dispatched_job.id_:
            return None
        return fifo_head

    def _note_retention_reorder(
        self,
        job: ImageGenerateJobPopResponse,
        displaced_head: ImageGenerateJobPopResponse,
        process: HordeProcessInfo,
    ) -> None:
        """Log a reorder, edge-triggered on the model and slot it seated work onto.

        A rotation seats several jobs onto the same retainer in a row, so the (model, slot) pair is the edge
        worth a line; the count carries the rest.
        """
        edge = (job.model, process.process_id)
        if edge == self._retention_affinity_logged_edge:
            return
        self._retention_affinity_logged_edge = edge
        logger.debug(
            f"Seating job {str(job.id_)[:8]} ({job.model}) ahead of head {str(displaced_head.id_)[:8]} "
            f"({displaced_head.model}) on process {process.process_id}: process {process.process_id} already "
            "retains those weights, where serving the head first would evict them.",
        )

    def _retention_affinity_candidates(
        self,
        head_job: ImageGenerateJobPopResponse | None,
    ) -> list[tuple[ImageGenerateJobPopResponse, HordeProcessInfo]]:
        """Pending jobs a slot's retained weights can serve at no upload, once a sampling slot is free.

        The candidates the placement order promotes ahead of the head (see
        :meth:`pending_inference_in_placement_order`), in queue order.

        Candidacy is deliberately *not* conditioned on this instant's concurrency headroom. Where the worker
        samples one job at a time the cap is spent for the whole of every sampling window, which is exactly when
        the preload pass runs, so a candidate that had to be dispatchable this instant would never be named in
        the regime the loss lives in. What makes a candidate is that the weights are already on a slot that can
        take work (``retained_resident_model`` plus ``can_accept_job``), so serving it costs no upload as soon as
        a slot frees.

        Nor is candidacy conditioned on the retainer being dispatchable *at all*. The retainer is located with
        ``include_reserved=True``, the carve-out :meth:`_resident_process_for_job` documents for the residency and
        pricing queries: a disaggregation-pinned sampler lane is a lane no job may be dispatched onto yet, and it
        is still a lane carrying weights on the device. Whether those weights may be thrown away is a residency
        question and the answer does not change because a pin is standing, since the pin lifts when the pinned
        job's sampling ends and the queued job is seated there afterwards. Excluding pinned lanes empties the scan
        on a disaggregating worker, where a lane is pinned for most of every job.

        The priority tiers the selection loops honour are not bypassed: a job filtered by quarantine, auxiliary
        preparation, degraded-retry isolation, or in-progress status is never a candidate, so a reorder can only
        promote a job that was already eligible to be placed.
        """
        if head_job is None or head_job.model is None:
            return []
        head_job_id = str(head_job.id_) if head_job.id_ is not None else None
        if not affinity_skip_allowed(
            self._affinity_skip_state,
            head_job_id,
            self._clock(),
            affinity_budget_seconds(self._state.recent_job_ttl),
            _AFFINITY_MAX_SKIPS,
        ):
            self._trace_empty_affinity_scan("skip budget spent", head_job)
            return []
        if self._head_aged_past_anti_starvation(head_job):
            self._trace_empty_affinity_scan("head aged past anti-starvation bound", head_job)
            return []

        in_progress = self._job_tracker.jobs_in_progress
        candidates: list[tuple[ImageGenerateJobPopResponse, HordeProcessInfo]] = []
        rejections: list[str] = []
        for job in self._job_tracker.jobs_pending_inference:
            if job is head_job or job.model is None or job.model == head_job.model:
                rejections.append(f"{job.model}: head/same-model")
                continue
            if job in in_progress or self._job_requires_aux_preparation(job):
                rejections.append(f"{job.model}: in-progress/aux-gated")
                continue
            # A degraded retry must run isolated (see the diversity path), so it is never a bypass target.
            if self._job_tracker.is_degraded_dispatch_pending(job):
                rejections.append(f"{job.model}: degraded-retry")
                continue
            if self._process_lifecycle.is_model_load_quarantined(job.model):
                rejections.append(f"{job.model}: quarantined")
                continue
            resident = self._resident_process_for_job(job, include_reserved=True)
            if resident is None or resident.retained_resident_model != job.model:
                rejections.append(
                    f"{job.model}: no-retainer"
                    f" (resident={'none' if resident is None else resident.process_id},"
                    f" retained={'-' if resident is None else resident.retained_resident_model})",
                )
                continue
            # A slot running live work is not holding weights *for* this job; it is using them for its own. Note
            # that a pinned lane between disaggregation stages reports an accepting state while it waits for its
            # conditioning, which is precisely the window its retained weights are most exposed in.
            if not resident.can_accept_job():
                rejections.append(f"{job.model}: retainer-busy ({resident.last_process_state.name})")
                continue
            if not self._reorder_is_pareto_admissible(head_job, resident):
                rejections.append(f"{job.model}: pareto (head has no other load target)")
                self._retention_reorder_pareto_vetoes += 1
                continue
            candidates.append((job, resident))
        if not candidates and rejections:
            self._trace_empty_affinity_scan("; ".join(rejections), head_job)
        return candidates

    def _reorder_is_pareto_admissible(
        self,
        head_job: ImageGenerateJobPopResponse,
        retainer: HordeProcessInfo,
    ) -> bool:
        """Whether seating a job on ``retainer`` can be done without delaying ``head_job``'s own completion.

        Fairness before reuse. A reorder is only ever a free win when the head was going to spend this cycle
        loading anyway *and* that load has somewhere else to go: the head's load is then the critical path and
        runs in parallel with the promoted job's sampling, so the head finishes no later than it would have.

        Both halves are required. The head must actually need a load (a head that is resident and merely waiting
        on capacity is delayed by anything seated ahead of it), and there must be an admissible preload target
        that is not the retaining lane. Where the retainer is the head's only possible target, the head keeps
        strict priority and its load evicts the retained weights: a reorder there would buy an upload back by
        making the head wait for a whole other job, which is a trade the head's deadline cannot be asked to fund.
        """
        if self._resident_process_for_job(head_job) is not None:
            return False
        return self._select_preload_process(head_job, [retainer.process_id]) is not None

    def _trace_empty_affinity_scan(self, reason: str, head_job: ImageGenerateJobPopResponse) -> None:
        """Log why the affinity scan named no candidate, throttled, while any slot retains weights.

        The scan runs every scheduling cycle and empty is its common answer, so the line is emitted only
        while retained weights exist to protect and at most once per throttle window; the gate that empties
        the scan is the diagnosis when retained copies are being evicted with their reusers pending.
        """
        if not any(p.retained_resident_model is not None for p in self._process_map.values()):
            return
        now = self._clock()
        if now - self._affinity_scan_trace_last < _AFFINITY_SCAN_TRACE_SECONDS:
            return
        self._affinity_scan_trace_last = now
        head_age = None
        if head_job.id_ is not None:
            tracked = self._job_tracker.get_tracked_job(head_job.id_)
            if tracked is not None and tracked.time_popped is not None:
                head_age = now - tracked.time_popped
        logger.debug(
            f"Affinity scan named no candidate: {reason}. head={head_job.model}"
            f" age={'-' if head_age is None else f'{head_age:.0f}s'}"
            f" ttl={head_job.ttl} recent_ttl={self._state.recent_job_ttl}",
        )

    def _record_preload_admission(
        self,
        decision: AdmissionDecision,
        *,
        job: ImageGenerateJobPopResponse | None = None,
        process: HordeProcessInfo | None = None,
        reason: str = "",
    ) -> None:
        """Remember one preload-admission decision for operator diagnostics, and record it as a decision.

        The stored record answers "why is this head not loading?" for the status surfaces; the decision event
        gives the same answer to offline analysis, which previously saw nothing at all from this gate. The
        sink coalesces repeats, so a head declined for the same reason every cycle costs one event.
        """
        self._last_preload_admission = LatestPreloadAdmission(
            decision=decision,
            model=job.model if job is not None else None,
            process_id=process.process_id if process is not None else None,
            reason=reason,
            timestamp=time.time(),
        )
        if self._decision_sink is None or job is None or job.id_ is None:
            return
        self._decision_sink(
            decision_kind=DecisionKind.VRAM_ADMISSION,
            subject=str(job.id_),
            verdict=_PRELOAD_ADMISSION_VERDICTS.get(decision, DecisionVerdict.DEFER),
            reason=reason or decision.value,
            inputs={
                "model": str(job.model),
                "decision": decision.value,
                "process_id": process.process_id if process is not None else None,
            },
        )

    def _preload_outcome(
        self,
        decision: AdmissionDecision,
        *,
        job: ImageGenerateJobPopResponse | None = None,
        process: HordeProcessInfo | None = None,
        reason: str = "",
    ) -> _PreloadJobOutcome:
        """Record a public admission decision and map it onto the preload pass control enum."""
        self._record_preload_admission(decision, job=job, process=process, reason=reason)
        return _preload_outcome_from_admission(decision)

    def _attempt_preload_for_job(
        self,
        job: ImageGenerateJobPopResponse,
        *,
        head_job: ImageGenerateJobPopResponse | None,
        loaded_models: set[str | None],
    ) -> _PreloadJobOutcome:
        """Run one pending job through the preload admission pipeline.

        The gates, in order: quarantine (faults the job), already-resident, the absolute RAM danger
        floor, the exclusive-job hold, target selection, the
        cycle-on-model-change replacement, the per-device load serialization gate, and the VRAM/RAM
        budget admission. The returned :class:`_PreloadJobOutcome` tells the pass whether to consider the
        next pending job, stop for this cycle, or record that a preload was issued.

        Args:
            job: The pending job to consider loading a model for.
            head_job: The queue head, which alone may escalate to displacing another queued model.
            loaded_models: The models already resident or loading, so a repeat is not staged. Read through
                :meth:`_model_loaded_for_job`, which asks the question per job rather than card-blind.
        """
        bridge_data = self._runtime_config.bridge_data
        if job.model is None:
            raise ValueError(f"job.model is None ({job})")

        # An aux-unprepared job must not compete for capacity: staging its model reserves a lane and prices
        # VRAM around work that cannot sample yet, so any job able to run outranks it and the pass moves on so
        # that fitting sibling is preloaded instead. With no such sibling the reservation costs nobody
        # anything, and withholding it only serializes the checkpoint load behind the auxiliary download that
        # the load could have run alongside. The slot must still be unoccupied (checked once a target is
        # chosen), and dispatch stays gated on preparation either way: this moves when weights enter RAM,
        # never when the job samples.
        aux_gated = self._job_requires_aux_preparation(job)
        if aux_gated and head_job is not None:
            return _PreloadJobOutcome.NEXT_JOB

        if (unserviceable_reason := self._unserviceable_job_reason(job)) is not None:
            if job not in self._job_tracker.jobs_in_progress:
                self._fault_unserviceable_job(job, unserviceable_reason)
            return self._preload_outcome(
                AdmissionDecision.UNSERVICEABLE,
                job=job,
                reason=unserviceable_reason,
            )

        # A model quarantined for repeatedly failing to load must never be preloaded again: doing so only
        # re-arms the crash/recovery loop it was quarantined to stop. Fault the job so the horde reissues
        # it elsewhere rather than letting an unservable head wedge the queue.
        if self._process_lifecycle.is_model_load_quarantined(job.model):
            if job not in self._job_tracker.jobs_in_progress:
                logger.warning(
                    f"Skipping preload of quarantined model {job.model}; faulting its job for reissue.",
                )
                self._job_tracker.handle_job_fault_now(job, retryable=False)
            return self._preload_outcome(AdmissionDecision.QUARANTINED, job=job, reason="model load quarantined")

        if self._model_loaded_for_job(job, loaded_models) and not self._duplicate_copy_may_serve(job):
            return self._preload_outcome(
                AdmissionDecision.ALREADY_LOADED, job=job, reason="model already resident or loading"
            )

        # Absolute system-RAM floor (degrade, never crash): loading a new model routes its weights through
        # system RAM first, so admitting one while the host is already below its danger floor is the OS
        # OOM kill, not progress. This gates every preload path independent of the marginal RAM budget, which
        # can pass on a job's small estimate while the whole host is on the edge (resident weights + the
        # safety process + other apps). The governor's tick has already degraded the host this cycle;
        # this only defers the load. Gated on the budget being active (the same switch the rest of the
        # memory machinery uses).
        if self._budget_active() and self._preload_blocked_by_ram_pressure(job):
            return self._preload_outcome(
                AdmissionDecision.DEFER_RAM_PRESSURE, job=job, reason="system RAM danger floor"
            )

        is_head_blocker = head_job is not None and job is head_job

        # Which slots this preload may not displace: the queued-model guard, model->process affinity
        # (never displace the last resident copy of a still-wanted model; the working model set is
        # taken from live state, not bridge_data.image_models_to_load, because the harness/canned
        # path never resolves that config field), and slots draining for RAM reclaim. The guards are
        # target exclusions only, never a wedge: the head-starvation fallback below deliberately
        # overrides them, and the governor recycles a draining slot once it is idle.
        inference_process_models = {
            p.process_id: p.loaded_horde_model_name
            for p in self._process_map.values()
            if p.process_type == HordeProcessType.INFERENCE
        }
        wanted_models: set[str] = {m for m in inference_process_models.values() if m is not None}
        wanted_models.update(j.model for j in self._job_tracker.jobs_pending_inference if j.model is not None)
        wanted_models.update(j.model for j in self._job_tracker.jobs_in_progress if j.model is not None)
        preload_disallowed = compute_preload_disallowed_processes(
            queued_model_process_ids=self._process_lifecycle.get_processes_with_model_for_queued_job(),
            busy_process_ids=[p.process_id for p in self._process_map.values() if p.is_process_busy()],
            prefer_busy_only=self._process_map.num_loaded_inference_processes()
            < (len(self._job_tracker.jobs_pending_inference) + len(self._job_tracker.jobs_in_progress)),
            inference_process_models=inference_process_models,
            wanted_models=wanted_models,
            max_inference_processes=self._max_inference_processes,
            draining_process_ids=frozenset(self._processes_draining_for_ram),
        )
        if not self._job_tracker.is_admitted_exclusive(job):
            preload_disallowed.update(
                process.process_id
                for process in self._process_map.values()
                if process.process_type is HordeProcessType.INFERENCE
                and self._job_tracker.has_exclusive_job_in_progress(
                    process.device_index if self._multi_gpu_routing_active else None
                )
            )

        # On a multi-GPU host this also chooses *which* card to load onto: an eligible card already
        # holding the model first, then the least-loaded eligible card. Single-GPU returns the first
        # available slot exactly as before.
        available_process = self._select_preload_process(job, sorted(preload_disallowed))

        if available_process is None and is_head_blocker:
            # The head of the queue could not get a slot because affinity (or the queued-model
            # guard) protected every idle process. Affinity is provisioned against the
            # inference-process *ceiling*, so with more resident models than running processes it
            # can pin every slot and starve a genuinely-queued head, wedging the whole worker. The
            # head must make progress regardless of whether the measured budget is active, so fall
            # back to a displacement target that spares live work and prefers an idle resident model
            # no queued job needs. This is the budget-independent counterpart to the budget-gated
            # make-room escalation in the admission pipeline. A slot retaining weights a queued job reuses is
            # not spared here: the placement order has already moved that job ahead of this one, so a head that
            # still reaches this fallback is one the reorder is not protecting.
            available_process = self._select_head_room_process(job)

        if available_process is None:
            return self._preload_outcome(
                AdmissionDecision.NO_TARGET, job=job, reason="no idle inference slot available"
            )

        exclusive_scope = available_process.device_index if self._multi_gpu_routing_active else None
        if self._job_tracker.has_exclusive_job_in_progress(exclusive_scope) and not (
            self._job_tracker.is_admitted_exclusive(job)
        ):
            return self._preload_outcome(
                AdmissionDecision.EXCLUSIVE_IN_PROGRESS,
                job=job,
                process=available_process,
                reason="exclusive over-budget job in progress on target card",
            )

        # An aux-gated preload takes a wholly unoccupied slot or none at all. Displacing a resident model,
        # cycling a process, or evicting to make room are all costs paid on behalf of a job that cannot sample
        # until its auxiliary files land, and the model thrown away may be one a dispatchable job still wants.
        if aux_gated and not self._slot_is_unoccupied(available_process):
            return self._preload_outcome(
                AdmissionDecision.NEXT_JOB,
                job=job,
                process=available_process,
                reason="aux-gated preload would displace an occupied slot",
            )

        # Device-free governor growth hold: while the target card's device-level free VRAM sits below the
        # soft floor (PRESSURE or SATURATED), bringing this model to a slot that does not already hold it
        # would grow a footprint already near the WDDM paging cliff. Defer until the card recovers. A job
        # already in progress is exempt: its preload is part of live work, not new speculative growth, and
        # withholding it would wedge a job the card is already committed to.
        growth_hold_device = available_process.device_index if self._multi_gpu_routing_active else 0
        if self.is_vram_growth_held(growth_hold_device) and job not in self._job_tracker.jobs_in_progress:
            return self._preload_outcome(
                AdmissionDecision.DEFER_VRAM_GROWTH_HOLD,
                job=job,
                process=available_process,
                reason="device-free governor holding VRAM growth (device near paging cliff)",
            )

        if (
            available_process.last_process_state != HordeProcessState.WAITING_FOR_JOB
            and available_process.loaded_horde_model_name is not None
            and bridge_data.cycle_process_on_model_change
            and not self._state.shutting_down
        ):
            self._process_lifecycle._replace_inference_process(available_process, intentional_reclaim=True)
            return self._preload_outcome(
                AdmissionDecision.REPLACE_PROCESS,
                job=job,
                process=available_process,
                reason="cycling process for model change",
            )

        # Serialize preloads per card, not worker-wide: the gate exists so two checkpoints do not load
        # onto the same device at once (disk-read + VRAM-allocation spike). On a multi-GPU host a load
        # onto an idle card is independent of one happening on another card, so scope the in-flight count
        # to the card this preload would land on. Worker-wide (device_index=None) on a single-GPU host
        # keeps the original behavior byte-identical. Without this, a card that is almost always mid-load
        # (the busy card) perpetually blocks the idle card from ever getting its first model -> starvation.
        preload_scope_device = available_process.device_index if self._multi_gpu_routing_active else None
        num_preloading_processes = self._process_map.num_preloading_processes(
            device_index=preload_scope_device,
        )

        if preload_concurrency_blocked(
            num_preloading=num_preloading_processes,
            max_concurrent_inference_processes=self._max_concurrent_inference_processes,
            very_fast_disk_mode=bool(bridge_data.very_fast_disk_mode),
        ):
            if not self._preload_delay_notified:
                logger.opt(colors=True).info(
                    "<fg #7b7d7d>"
                    f"Already preloading {num_preloading_processes} models, waiting for one to finish before "
                    "preloading {}"
                    "</>",
                    job.model,
                )
                self._preload_delay_notified = True
            return self._preload_outcome(
                AdmissionDecision.DEFER_CONCURRENCY,
                job=job,
                process=available_process,
                reason="preload concurrency gate",
            )

        # Resource budget gate: a fresh preload loads this model's weights into the shared device
        # (VRAM) and into system RAM, so admit it only when both measured free VRAM and available
        # RAM cover its estimated cost plus their reserves. This is the proactive guard against the
        # multi-process over-commit that OOMs the GPU and against resident weights paging RAM to
        # disk. When a resource does not fit, start reclaiming it from idle resident models
        # (overriding residency under pressure) and defer this preload rather than over-committing.
        if self._budget_active() and not self._admit_preload_under_budget(
            job,
            available_process,
            is_head_blocker=is_head_blocker,
        ):
            return self._preload_outcome(
                AdmissionDecision.DEFER_BUDGET,
                job=job,
                process=available_process,
                reason=self._last_budget_defer_reason or "VRAM/RAM budget gate",
            )

        # Admission is not read-only: a whole-card residency establish scales the inference pool down to the
        # depth the head needs, and this preload's chosen target is by construction an idle, empty lane, which
        # is exactly what that scale-down selects as a victim. A lane the pool no longer has cannot be sent a
        # load command, so re-select onto a surviving one rather than addressing a retired slot. Deferring
        # instead would be worse than the crash it replaces: the head would lose its preload every cycle for
        # as long as the residency keeps choosing its own target.
        if self._process_map.get(available_process.process_id) is not available_process:
            available_process = self._select_preload_process(job, sorted(preload_disallowed))
            if available_process is None:
                return self._preload_outcome(
                    AdmissionDecision.NO_TARGET,
                    job=job,
                    reason="the preload target was retired during admission and no other slot is free",
                )

        if self._send_preload(job, available_process):
            return self._preload_outcome(
                AdmissionDecision.ADMIT, job=job, process=available_process, reason="preload sent"
            )
        return self._preload_outcome(
            AdmissionDecision.STOP_PASS, job=job, process=available_process, reason="preload send failed"
        )

    @staticmethod
    def _slot_is_unoccupied(process_info: HordeProcessInfo) -> bool:
        """Whether a slot holds no model and is idle, so loading onto it displaces nothing.

        Distinct from ``can_accept_job()``, which is also true of a slot holding a resident model: a caller
        that must not throw any load away needs the stronger property that there is nothing there to throw.
        """
        return (
            process_info.loaded_horde_model_name is None
            and process_info.last_process_state == HordeProcessState.WAITING_FOR_JOB
        )

    def _select_head_room_process(self, job: ImageGenerateJobPopResponse) -> HordeProcessInfo | None:
        """Pick an eligible idle process to free for a starved head-of-queue job, or None.

        Used when the normal preload picker found no slot because affinity (provisioned against the
        inference-process ceiling) or the queued-model guard protected every idle process. The head must
        still make progress, so this deliberately overrides those guards. It never overrides card eligibility:
        on a multi-card worker, a temporarily busy or restarting eligible card cannot make another card a valid
        preload target. Among eligible slots it never returns a process running live work (only
        ``can_accept_job()`` slots, and never one whose model is in progress) and prefers the cheapest
        displacement: an empty slot, then one holding a resident model no pending or in-progress job needs, then,
        as a last resort, one holding a merely-queued model.

        """
        eligible_cards = self._eligible_card_indices(job) if self._multi_gpu_routing_active else None
        slots = tuple(
            PreloadSlotSnapshot(
                process_id=process_info.process_id,
                model_name=process_info.loaded_horde_model_name,
                can_accept_job=process_info.can_accept_job(),
            )
            for process_info in self._process_map.values()
            if process_info.process_type == HordeProcessType.INFERENCE
            and (eligible_cards is None or process_info.device_index in eligible_cards)
            # A lane pinned as an in-flight disaggregated job's sampler is live work even while it idles
            # between its stages: its state reads WAITING_FOR_JOB, but preloading over it would evict the
            # weights the pinned job's sample stage is about to use.
            and not self._process_map.is_reserved_for_disaggregation(process_info.process_id)
        )
        chosen_id = select_head_room_process_id(
            slots,
            in_progress_models={job.model for job in self._job_tracker.jobs_in_progress},
            pending_models={job.model for job in self._job_tracker.jobs_pending_inference if job.model is not None},
        )
        return self._process_map.get(chosen_id) if chosen_id is not None else None

    def _select_idle_thread_diversity_job(
        self,
        head_job: ImageGenerateJobPopResponse,
        candidates: list[ImageGenerateJobPopResponse],
    ) -> tuple[ImageGenerateJobPopResponse, HordeProcessInfo] | None:
        """A pending distinct-model job resident on a free process that may overlap the in-flight work.

        When the head's own process cannot take work right now because it is busy sampling the head's model,
        a later job for a *different* model that is already resident on an idle process can run concurrently
        instead of leaving the thread idle. Preferring a distinct model also avoids loading a duplicate copy
        of the head's model onto a second process: with several same-model jobs ahead of a lone different
        model (a run of one checkpoint followed by another), threading the different model alongside the run
        processes it for free under the run, rather than idling a thread and tacking the second model on at
        the end as its own load. The overlap-headway gate still applies (two heavy models are not stacked
        without headway), degraded retries that must run isolated are skipped, and the head keeps its queue
        position (the caller records a line-skip) so it dispatches the moment its process frees.
        """
        for candidate_job in candidates:
            if candidate_job.model is None or candidate_job.model == head_job.model:
                continue
            if self._job_tracker.is_degraded_dispatch_pending(candidate_job):
                continue
            candidate_process = self._resident_process_for_job(candidate_job)
            if candidate_process is None or not candidate_process.can_accept_job():
                continue
            if not self._concurrent_overlap_allowed(
                candidate_job,
                target_device_index=candidate_process.device_index,
            ):
                continue
            return candidate_job, candidate_process
        return None

    @property
    def _multi_gpu_routing_active(self) -> bool:
        """Whether per-card dispatch routing applies (the worker drives more than one card).

        A single card (or the empty plan unit tests construct) keeps the original card-agnostic dispatch,
        so all multi-GPU routing below is a strict no-op on a single-GPU host.
        """
        return len(self._card_runtimes) > 1

    def _eligible_card_indices(self, job: ImageGenerateJobPopResponse) -> set[int]:
        """Device indices of the cards whose effective config can serve ``job`` (see ``gpu_eligibility``).

        Restricts dispatch (and the resident-process search) to cards that offer the job's model, fit its
        weights, enable every feature it needs, and allow its resolution. An unknown fact never excludes a
        card (the eligibility primitive abstains), so this only ever narrows routing on a genuine mismatch.
        """
        return self._card_eligibility(job).eligible_card_indices

    def _card_eligibility(self, job: ImageGenerateJobPopResponse) -> CardEligibilityVerdict:
        """Return exact per-card reasons behind the routing verdict for ``job``."""
        baseline = self._model_metadata.get_baseline(job.model) if job.model is not None else None
        baseline_value = baseline.value if isinstance(baseline, KNOWN_IMAGE_GENERATION_BASELINE) else baseline
        weight_mb = predict_job_weight_mb(job, baseline)
        return card_eligibility_for(
            job,
            self._card_runtimes,
            baseline=baseline_value,
            weight_mb=weight_mb,
        )

    def _baseline_value_for_job(self, job: ImageGenerateJobPopResponse) -> str | None:
        """Return the job model's baseline value, or None when metadata is unavailable."""
        if job.model is None:
            return None
        baseline = self._model_metadata.get_baseline(job.model)
        return baseline.value if isinstance(baseline, KNOWN_IMAGE_GENERATION_BASELINE) else baseline

    def _model_serviceability_verdicts(
        self,
        job: ImageGenerateJobPopResponse,
    ) -> list[tuple[CardRuntime, ModelServiceabilityVerdict]]:
        """Return per-serving-card serviceability verdicts for ``job``.

        The check abstains when the runtime card map or model footprint is unknown. It considers only cards
        whose effective config serves the model, matching the offer and placement surfaces.
        """
        if job.model is None or not self._card_runtimes:
            return []
        baseline = self._baseline_value_for_job(job)
        figures = model_footprint_figures_for_baseline(baseline)
        if figures is None:
            return []
        verdicts: list[tuple[CardRuntime, ModelServiceabilityVerdict]] = []
        for card in self._card_runtimes.values():
            if job.model not in set(card.config.image_models_to_load):
                continue
            baseline_mb = self._admission_baseline_mb(card.device_index)
            verdicts.append(
                (
                    card,
                    assess_model_serviceability(
                        total_vram_mb=card.total_vram_mb,
                        baseline_mb=baseline_mb,
                        noise_buffer_mb=None,
                        figures=figures,
                    ),
                ),
            )
        return verdicts

    def _unserviceable_job_reason(self, job: ImageGenerateJobPopResponse) -> str | None:
        """Return a fault reason when no serving card can ever host this job's model minimum."""
        verdicts = self._model_serviceability_verdicts(job)
        if not verdicts or any(verdict.serviceable for _, verdict in verdicts):
            return None
        arithmetic = "; ".join(f"device {card.device_index}: {verdict.reason()}" for card, verdict in verdicts)
        return f"model minimum footprint cannot fit any serving card; {arithmetic}"

    def _fault_unserviceable_job(self, job: ImageGenerateJobPopResponse, reason: str) -> None:
        """Fault an unserviceable queued job before any child process touches VRAM for it."""
        logger.warning(f"Faulting unserviceable job {job.id_} for model {job.model}: {reason}")
        self._job_tracker.handle_job_fault_now(
            job,
            is_resource_failure=True,
            retryable=False,
            fault_reason=reason,
        )

    def _fault_ineligible_job(self, job: ImageGenerateJobPopResponse, verdict: CardEligibilityVerdict) -> None:
        """Fault a job no configured card can execute and disclose every card's reasons."""
        reason = f"no configured card can serve the accepted job; {verdict.reason_summary()}"
        logger.warning(f"Faulting ineligible job {job.id_} for model {job.model}: {reason}")
        self._job_tracker.handle_job_fault_now(
            job,
            retryable=False,
            scheduling_fault=True,
            fault_reason=reason,
        )

    def _card_inference_load(self, device_index: int) -> int:
        """Count this card's inference processes currently busy: the least-loaded routing tie-breaker."""
        return sum(
            1
            for p in self._process_map.values()
            if p.process_type == HordeProcessType.INFERENCE and p.device_index == device_index and p.is_process_busy()
        )

    def _pick_best_resident_process(self, candidates: list[HordeProcessInfo]) -> HordeProcessInfo:
        """Choose which resident process to dispatch to: prefer one ready now, then the least-loaded card.

        The "sticky, then least-loaded" policy at the process level: every candidate already holds the model
        (sticky), so among them a process that can take work immediately is preferred, and ties break to the
        card running the fewest inference jobs so a hot model spreads across cards instead of queueing on one.
        """
        ready = [p for p in candidates if p.can_accept_job()]
        pool = ready or candidates
        return min(pool, key=lambda p: self._card_inference_load(p.device_index))

    def _resident_process_for_job(
        self,
        job: ImageGenerateJobPopResponse,
        *,
        include_reserved: bool = False,
    ) -> HordeProcessInfo | None:
        """The resident process to dispatch ``job`` to, honoring per-card eligibility on a multi-GPU host.

        Single-GPU: identical to :meth:`ProcessMap.get_process_by_horde_model_name` (the first resident
        process), so the dispatch path is byte-identical. Multi-GPU: restrict to processes pinned to cards
        eligible for this job, then apply the sticky-then-least-loaded policy. Returns None when the model is
        resident only on cards that cannot serve this particular job, or is not resident anywhere.

        Pinned disaggregation-sampler lanes are excluded by default (a dispatch may never land on a pinned
        lane). ``include_reserved=True`` includes them, for the residency and pricing queries that must see a
        model's weights even where they sit on a lane no job may be dispatched onto yet.

        A free slot that holds the model's *retained* weights wins over any other resident candidate. Both
        record the model as loaded, but only the retainer still has it on the device, so seating the job
        anywhere else pays a full RAM->VRAM re-upload for weights the card already carries.
        """
        if job.model is None:
            return None
        if not self._multi_gpu_routing_active:
            resident = self._process_map.get_process_by_horde_model_name(job.model, include_reserved=include_reserved)
            return self._prefer_free_retainer(job.model, resident, include_reserved=include_reserved)
        allowed = self._eligible_card_indices(job)
        candidates = self._process_map.get_processes_by_horde_model_name(
            job.model,
            allowed_cards=allowed,
            include_reserved=include_reserved,
        )
        if not candidates:
            return None
        free_retainers = [
            candidate
            for candidate in candidates
            if candidate.retained_resident_model == job.model and candidate.can_accept_job()
        ]
        return self._pick_best_resident_process(free_retainers or candidates)

    def _prefer_free_retainer(
        self,
        model: str,
        resident: HordeProcessInfo | None,
        *,
        include_reserved: bool,
    ) -> HordeProcessInfo | None:
        """Swap a resident-by-name choice for a free slot that holds ``model``'s retained weights, if one exists.

        ``loaded_horde_model_name`` records that a slot has served a model, not that its weights are still on
        the device; the retention record is the parent's device-level truth. Where the two name different
        slots, the retaining one is the dispatch destination.
        """
        if resident is None or resident.retained_resident_model == model:
            return resident
        for process_info in self._process_map.values():
            if process_info.retained_resident_model != model or process_info.loaded_horde_model_name != model:
                continue
            if not include_reserved and self._process_map.is_reserved_for_disaggregation(process_info.process_id):
                continue
            if not process_info.can_accept_job():
                continue
            return process_info
        return resident

    def _retention_affinity_retainer(
        self,
        job: ImageGenerateJobPopResponse,
        current_choice: HordeProcessInfo | None,
    ) -> HordeProcessInfo | None:
        """The busy slot holding ``job``'s model on the device that ``job`` must wait for, else None.

        A same-model successor seated on any other slot funds a second full copy of weights the card is
        already carrying. Where that copy does not fit beside the retained one, the head holds its queue
        position for the retainer instead, on the pinned-lane precedent: the wait is not funding a fresh copy,
        so it lets other resident work bypass it while the retainer finishes and becomes the destination.

        The hold is bounded by the ttl-derived affinity budget and ends the moment the retention record clears
        (an eviction, a death, a completed job that retained nothing). Falling through is safe rather than
        merely tolerable: the dispatch admission gate then prices the fresh copy against the same residents.
        A retainer that is free needs no wait, since ordinary dispatch already prefers it.

        Pinned disaggregation lanes stay excluded here, unlike in the retention-protection scan, and the
        difference is what the returned slot is *for*. This one becomes the caller's ``process_with_model``, the
        dispatch destination, and a job may never be dispatched onto a pinned lane; the only thing standing
        between a widened scan and that dispatch would be the busy check below, which a pinned lane fails for
        much of its life (it reports an accepting state between stages while it waits for conditioning). The
        residency question a widened scan would answer is already answered for pinned lanes by
        :meth:`_pinned_lane_resident_for_job`, whose branch holds the head's queue position until the pin
        releases and is deliberately exempt from this budget because it funds no second copy.
        """
        model = job.model
        if model is None:
            return None
        if current_choice is not None and current_choice.retained_resident_model == model:
            return None
        allowed = self._eligible_card_indices(job) if self._multi_gpu_routing_active else None
        for process_info in self._process_map.values():
            if process_info.retained_resident_model != model:
                continue
            if allowed is not None and process_info.device_index not in allowed:
                continue
            if self._process_map.is_reserved_for_disaggregation(process_info.process_id):
                continue
            if process_info.can_accept_job():
                continue
            if self._retention_affinity_budget_spent(job):
                return None
            target = current_choice or self._select_preload_process(
                job,
                disallowed_processes=[process_info.process_id],
            )
            if target is None:
                return None
            device_index = process_info.device_index if self._multi_gpu_routing_active else None
            if self._fits_beside_retained_residents(job, target=target, device_index=device_index) is False:
                return process_info
        return None

    def _retention_affinity_budget_spent(self, job: ImageGenerateJobPopResponse) -> bool:
        """Whether a head has held for a retaining slot longer than the affinity budget allows.

        Measured from the job's pop against the same ttl-derived budget the resident-bypass window uses, so a
        head can never age out its deadline behind a retainer that is not releasing. A head whose wait cannot
        be timed at all (no id, untracked, no pop time) counts as spent: an untimeable wait is the one shape
        that could become unbounded, and falling through prices the fresh copy through dispatch admission.
        """
        job_id = job.id_
        if job_id is None:
            return True
        tracked = self._job_tracker.get_tracked_job(job_id)
        if tracked is None or tracked.time_popped is None:
            return True
        ttl = float(job.ttl) if job.ttl is not None else self._state.recent_job_ttl
        return (self._clock() - tracked.time_popped) > affinity_budget_seconds(ttl)

    def _pinned_lane_resident_for_job(self, job: ImageGenerateJobPopResponse) -> HordeProcessInfo | None:
        """The disaggregation-pinned lane holding ``job``'s model when that is the only resident copy, else None.

        The dispatch query (:meth:`_resident_process_for_job`) excludes pinned lanes, so a job whose model is
        resident only on a pinned sampler lane reads as not-resident and would otherwise be priced a fresh
        preload that cannot fit beside the pinned residents. This names that case: an unreserved resident copy
        does not exist, but a pinned lane holds the model. The head then holds for the pin to release (dispatch
        onto the resident lane, priced as resident) instead of funding a second copy; a job is never dispatched
        onto the returned pinned lane.
        """
        if self._resident_process_for_job(job) is not None:
            return None
        resident = self._resident_process_for_job(job, include_reserved=True)
        if resident is not None and self._process_map.is_reserved_for_disaggregation(resident.process_id):
            return resident
        return None

    def _model_loaded_for_job(self, job: ImageGenerateJobPopResponse, loaded_models: set[str | None]) -> bool:
        """Whether ``job``'s model already counts as resident or loading *for this job's routing*.

        Single-GPU (or the empty plan unit tests construct): the card-blind membership test the preload pass
        has always used, so behaviour there is unchanged. Multi-GPU: a copy only counts when it sits on a card
        eligible to serve this job. A model resident solely on a card that cannot serve this job (its
        resolution, features or weights exceed that card's effective config) leaves the job needing its own
        copy on an eligible card. Dispatch will not seat it on the ineligible copy, and counting that copy
        here withholds the preload that would give the job one, so neither lane can move the job. A job with
        no eligible card at all keeps the card-blind answer; the eligibility fault path owns that case.
        """
        model = job.model
        if model is None:
            return False
        if not self._multi_gpu_routing_active:
            return model in loaded_models
        allowed = self._eligible_card_indices(job)
        if not allowed:
            return model in loaded_models
        if self._process_map.get_processes_by_horde_model_name(model, allowed_cards=allowed, include_reserved=True):
            return True
        model_info = self._horde_model_map.root.get(model)
        if model_info is None:
            return False
        if not (
            model_info.horde_model_load_state.is_loaded()
            or model_info.horde_model_load_state == ModelLoadState.LOADING
        ):
            return False
        owner = self._process_map.get(model_info.process_id)
        # An entry whose owning process is gone records a residency that no longer exists; counting it would
        # suppress the preload that replaces it.
        return owner is not None and owner.device_index in allowed

    def _duplicate_copy_may_serve(self, job: ImageGenerateJobPopResponse) -> bool:
        """Whether a second copy of ``job``'s already-resident model may be preloaded onto another card.

        The single-copy rule (a model that is resident or loading is never preloaded again) is load and
        VRAM economy, and on one card a duplicate is pure waste. Across cards it inverts when every copy
        is busy running other work: the queued job then waits a whole sampling window for weights an idle
        card could be given instead, and a pending queue dominated by such jobs leaves those cards doing
        nothing at all. A duplicate is therefore considered only when the worker routes across multiple
        cards, at least one eligible copy exists, every such copy is busy, and no copy is still loading
        (a load in flight is about to provide a serving copy; doubling it is the waste the rule exists to
        stop). Whether the duplicate actually lands stays the preload pipeline's decision: the displacement
        guards, the preload concurrency gate, VRAM admission, and the growth hold all apply to it unchanged,
        and the target selection already excludes the slots holding the existing copies.
        """
        if not self._multi_gpu_routing_active or job.model is None:
            return False
        model_info = self._horde_model_map.root.get(job.model)
        if model_info is not None and model_info.horde_model_load_state == ModelLoadState.LOADING:
            return False
        allowed = self._eligible_card_indices(job)
        if not allowed:
            return False
        copies = self._process_map.get_processes_by_horde_model_name(
            job.model,
            allowed_cards=allowed,
            include_reserved=True,
        )
        if not copies:
            # Resident only per the model map, with no live process holding it; the missing-model
            # recovery owns that inconsistency, not a duplicate load.
            return False
        return all(not copy.can_accept_job() for copy in copies)

    def _resident_only_on_ineligible_cards(self, job: ImageGenerateJobPopResponse) -> bool:
        """Whether ``job``'s model is resident somewhere, but on no card eligible to serve this job.

        This is not a missing model: the weights are where the model map says they are, on a card whose
        effective config cannot serve this particular job. The job needs its own copy on an eligible card,
        which is the preload pass's business, so dispatch falls through instead of treating a truthful
        residency record as stale and expiring it.
        """
        model = job.model
        if model is None or not self._multi_gpu_routing_active:
            return False
        if self._resident_process_for_job(job, include_reserved=True) is not None:
            return False
        return self._process_map.get_process_by_horde_model_name(model, include_reserved=True) is not None

    def _head_aged_past_anti_starvation(self, head_job: ImageGenerateJobPopResponse) -> bool:
        """Whether a queued head has waited past the anti-starvation fraction of its ttl and must not be bypassed.

        Measures the head's absolute age since pop against its own ttl (the per-job value when the horde supplied
        one, else the most recent ttl the worker saw). Returns False when neither a ttl nor a pop time is known,
        so a job the horde gave no ttl keeps the pure skip-budget behaviour and is never forced off the bypass
        path by this gate.
        """
        job_id = head_job.id_
        if job_id is None:
            return False
        ttl = float(head_job.ttl) if head_job.ttl is not None else self._state.recent_job_ttl
        if ttl is None or ttl <= 0:
            return False
        tracked = self._job_tracker.get_tracked_job(job_id)
        if tracked is None or tracked.time_popped is None:
            return False
        age_since_pop = self._clock() - tracked.time_popped
        return age_since_pop > _DISPATCH_ANTI_STARVATION_TTL_FRACTION * ttl

    def _note_anti_starvation_override(self, head_job: ImageGenerateJobPopResponse) -> None:
        """Log once (edge-triggered) when the age override first suppresses resident-model bypass for a head."""
        if head_job.id_ is None:
            return
        job_id = str(head_job.id_)
        if job_id == self._anti_starvation_logged_head_id:
            return
        self._anti_starvation_logged_head_id = job_id
        tracked = self._job_tracker.get_tracked_job(head_job.id_)
        age_since_pop = self._clock() - tracked.time_popped if tracked is not None and tracked.time_popped else 0.0
        logger.info(
            f"Anti-starvation override: head job {job_id[:8]} (model {head_job.model}) has waited "
            f"{age_since_pop:.0f}s since pop, past the anti-starvation fraction of its ttl; resident-model bypass "
            "yields so its preload runs.",
        )

    def _slots_holding_the_head_model(
        self,
        job: ImageGenerateJobPopResponse,
        disallowed_processes: list[int],
    ) -> list[int]:
        """Slots holding the queue head's model that a preload for ``job`` must not take, else an empty list.

        A preload placed on the head's own warm slot displaces the copy the head is waiting to run: the head
        loses its residency, its dispatch falls back to a cold load, and the slot it lost is now busy staging
        somebody else's model. That is a priority inversion, and it is worst exactly when it is most likely,
        because the head reaching its anti-starvation window is what makes the scheduler push a later job's
        preload past it in the first place.

        Two conditions, either sufficient. A slot the head's copy is on is refused outright whenever another
        idle slot can take the preload: the placement costs nothing to move and the head keeps its copy. With
        no other slot free the refusal holds only while the head is inside its anti-starvation window, so a
        single-slot worker still swaps models between jobs as it always has, and only a head the queue is
        actually starving is protected from having its own copy taken.

        A preload for the head itself, or for the model the head needs, displaces nothing and is never
        refused.
        """
        head = next(iter(self._job_tracker.jobs_pending_inference), None)
        if head is None or head.model is None:
            return []
        if head.id_ == job.id_ or head.model == job.model:
            return []
        holders = [
            process_info.process_id
            for process_info in self._process_map.values()
            if process_info.process_type == HordeProcessType.INFERENCE
            and process_info.loaded_horde_model_name == head.model
            and process_info.process_id not in disallowed_processes
        ]
        if not holders:
            return []
        alternative = self._process_map.get_first_available_inference_process(
            disallowed_processes=[*disallowed_processes, *holders],
        )
        if alternative is not None or self._head_aged_past_anti_starvation(head):
            return holders
        return []

    def _select_preload_process(
        self,
        job: ImageGenerateJobPopResponse,
        disallowed_processes: list[int],
    ) -> HordeProcessInfo | None:
        """The inference slot to preload ``job``'s model onto, choosing the card on a multi-GPU host.

        Single-GPU: identical to :meth:`ProcessMap.get_first_available_inference_process`, so the preload path
        is byte-identical. Multi-GPU: restrict to cards eligible for this job and pick the placement card by the
        same sticky-then-least-loaded policy dispatch uses: a card already holding this model first (avoid a
        duplicate load), then the card running the fewest inference jobs (balance fresh loads). Returns the
        first available slot on the best such card, or None when no eligible card has a free slot.

        Slots carrying the queue head's own copy are excluded first
        (:meth:`_slots_holding_the_head_model`), on either topology.
        """
        head_slots = self._slots_holding_the_head_model(job, disallowed_processes)
        if head_slots:
            disallowed_processes = [*disallowed_processes, *head_slots]
        if not self._multi_gpu_routing_active:
            return self._process_map.get_first_available_inference_process(disallowed_processes=disallowed_processes)
        eligible = self._eligible_card_indices(job)
        if not eligible:
            return None

        cards_already_serving_model = {
            process.device_index
            for process in self._process_map.values()
            if process.loaded_horde_model_name == job.model
        }
        placement_order = card_preload_order(
            eligible,
            cards_already_serving_model=cards_already_serving_model,
            card_busy_counts={device_index: self._card_inference_load(device_index) for device_index in eligible},
            card_free_vram_mb={
                device_index: self._measured_free_vram_mb(device_index=device_index) for device_index in eligible
            },
        )
        for device_index in placement_order:
            candidate = self._process_map.get_first_available_inference_process(
                disallowed_processes=disallowed_processes,
                device_index=device_index,
            )
            if candidate is not None:
                return candidate
        return None

    def _missing_model_recovery_latched(self) -> bool:
        """Whether the missing-model recovery latch still holds, within the preload budget.

        The latch is set when the head's model was expected resident but no process held it, and the next
        committed dispatch selection clears it. No dispatch is guaranteed to follow, so the latch also expires
        on ``preload_timeout``: the fresh preload the recovery released the job for either lands inside that
        window or has failed. Unbounded, one recovery would bar every later one for the rest of the run and
        would keep telling the recovery coordinator a load is still in flight. Read-only; the flag is left as
        written and only ever read through this bound.
        """
        if not self._model_recently_missing:
            return False
        budget = self._runtime_config.bridge_data.preload_timeout
        budget_seconds = float(budget) if isinstance(budget, int | float) else _MISSING_MODEL_LATCH_FALLBACK_SECONDS
        return (self._clock() - self._model_recently_missing_time) < budget_seconds

    async def _handle_process_missing(self, job: ImageGenerateJobPopResponse) -> None:
        """Recover when the head's model was expected resident but no process holds it.

        Reached only when the model map claims the model is loaded and no process is tagged with it, so the
        map entry is the stale half: it is expired and the job is released from in-progress so a fresh preload
        can be scheduled. Guarded by the missing-model latch so the recovery runs at most once per preload
        budget.
        """
        if self._missing_model_recovery_latched():
            return
        logger.warning(
            f"Expected to find a process with model {job.model} but none was found. Attempt to load it now...",
        )
        logger.debug(f"Horde model map: {self._horde_model_map}")
        logger.debug(f"Process map: {self._process_map}")

        if job.model is not None:
            logger.debug(f"Expiring entry for model {job.model}")
            self._horde_model_map.expire_entry(job.model)

            logger.debug(f"Horde model map: {self._horde_model_map}")
            logger.debug(f"Process map: {self._process_map}")

            self._model_recently_missing = True

            logger.debug(f"Last missing time: {self._model_recently_missing_time}")
            self._model_recently_missing_time = self._clock()

            if not await self._job_tracker.release_in_progress(job):
                logger.debug(f"Job {job.id_} not found in jobs_in_progress.")

    def _line_skip_cache_valid(self, cached: NextJobAndProcess) -> bool:
        """Whether a cached line-skip pair's premises still hold, including the residency it rested on.

        The pair was chosen because its target held the job's model; a RAM clear strips that name
        synchronously mid-cycle, and a pair that outlives its premise pins selection on a lane that can
        no longer serve it. The per-cycle cache scope bounds how long a stale pair could survive, and this
        check is the premise-level guard consulted where the cache is spent.
        """
        cached_job = cached.next_job
        return (
            cached_job in self._job_tracker.jobs_pending_inference
            and cached_job not in self._job_tracker.jobs_in_progress
            and cached.process_with_model.can_accept_job()
            and cached.process_with_model.loaded_horde_model_name == cached_job.model
        )

    async def get_next_job_and_process(
        self,
        information_only: bool = False,
    ) -> NextJobAndProcess | None:
        """Get the next job and process that can be started, if any.

        A single scheduling cycle calls this twice: once with ``information_only=True``
        (to look ahead for heavy-model / single-inference decisions) and once from
        :meth:`start_inference` to actually launch. The two calls must agree on which
        job is selected. Normal selection is deterministic given the queue and process
        states, but the line-skip branch below depends on transient process state
        (e.g. a process being mid aux-model download), so a line-skip decision is
        cached in ``self._pending_line_skip`` and returned on the second call. The
        cache is cleared at the start of each scheduling cycle (see
        :meth:`run_scheduling_cycle`) and at the end of :meth:`start_inference`.
        """
        cached = self._pending_line_skip
        if cached is not None:
            if self._line_skip_cache_valid(cached):
                return cached
            self._pending_line_skip = None

        next_job: ImageGenerateJobPopResponse | None = None
        next_n_jobs: list[ImageGenerateJobPopResponse] = []
        # The placement order, not the queue's own: a job a slot already retains the weights for is seated ahead
        # of a cold head rather than bypassing it, so it arrives here as the head and needs no line-skip. The
        # gates below are unchanged and still filter it, so a promoted job is one that was already placeable.
        for job in self.pending_inference_in_placement_order():
            if job in self._job_tracker.jobs_in_progress:
                continue
            # Never make a quarantined model the dispatch head: it can never become resident (preload_models
            # skips it and faults it for reissue), so selecting it here would only stall the scheduler.
            if self._process_lifecycle.is_model_load_quarantined(job.model):
                continue
            # An aux-unprepared job is invisible to dispatch: it holds nothing and cannot sample until the
            # pop-time prefetch pipeline clears its preparation gate. Skipping it here lets a fitting sibling
            # become the dispatch head and be GPU-fed while the gated job waits, so no lane idles behind it.
            if self._job_requires_aux_preparation(job):
                continue
            if next_job is None:
                next_job = job
            next_n_jobs.append(job)

        if next_job is None:
            return None

        if next_job.model is None:
            raise ValueError(f"next_job.model is None ({next_job})")

        if (unserviceable_reason := self._unserviceable_job_reason(next_job)) is not None:
            if not information_only:
                self._fault_unserviceable_job(next_job, unserviceable_reason)
            return None

        if self._card_runtimes:
            eligibility = self._card_eligibility(next_job)
            if not eligibility.eligible_card_indices:
                if not information_only:
                    self._fault_ineligible_job(next_job, eligibility)
                return None

        process_with_model = self._resident_process_for_job(next_job)
        # A slot busy with its own job may still be holding this model's weights on the device under an
        # earlier retention grant. Seating the head anywhere else would load a second copy of those same
        # weights; where that copy does not fit beside them, the retainer is the head's destination and the
        # head waits for it (the busy-slot handling below keeps the card fed with other resident work).
        retention_retainer = self._retention_affinity_retainer(next_job, process_with_model)
        if retention_retainer is not None:
            process_with_model = retention_retainer
        line_skip: LineSkip | None = None

        # On a multi-GPU host the head's resident process names the card this dispatch would land on, so the
        # concurrency cap and exclusive hold are scoped to that card. An unattributed exclusive job still
        # suppresses every card through JobTracker's conservative fallback.
        target_card: CardRuntime | None = None
        if self._multi_gpu_routing_active and process_with_model is not None:
            candidate_card = self._card_runtimes.get(process_with_model.device_index)
            if candidate_card is not None and (
                self._job_tracker.is_admitted_exclusive(next_job)
                or not self._job_tracker.has_exclusive_job_in_progress(candidate_card.device_index)
            ):
                target_card = candidate_card

        if target_card is not None:
            jobs_in_progress_count = len(self._jobs_in_progress_on_card(target_card.device_index))
        else:
            jobs_in_progress_count = len(self._job_tracker.jobs_in_progress)
        max_jobs_allowed = self._max_jobs_in_progress_allowed(card=target_card)
        if jobs_in_progress_count >= max_jobs_allowed:
            exclusive_scope = target_card.device_index if target_card is not None else None
            if self._job_tracker.has_exclusive_job_in_progress(exclusive_scope):
                logger.debug(
                    "Dispatch held at the concurrency cap while an exclusive in-progress job requires isolation "
                    f"(jobs_in_progress={jobs_in_progress_count}, cap={max_jobs_allowed}).",
                )
            return None

        if process_with_model is None:
            if next_job.model is None:
                raise ValueError(f"next_job.model is None ({next_job})")

            # The head's model may be resident only on a disaggregation-pinned sampler lane, which the dispatch
            # query excludes so no job is ever dispatched onto a pinned lane. That copy becomes dispatchable when
            # the pin releases (its disaggregated job's sampling finishes), so the head must wait for it rather
            # than fund a second copy that cannot fit beside the pinned residents.
            pinned_resident = self._pinned_lane_resident_for_job(next_job)

            # The head's model is not resident on any dispatchable process. Let a later already-resident job
            # bypass it so the card is not idle while the head is cold, but only within a bound: the affinity
            # skip window (a wall-clock budget from the job ttl plus a hard skip ceiling), keyed to the head's
            # identity and advanced only on committed dispatch. This covers both the forecast case (a preload is
            # already on the way) and the plain cold head during a preload-defer window; both are now bounded so
            # a head cannot age past its ttl behind a stream of resident work. The old perpetual-bypasser fear is
            # answered by the bound, not by refusing to bypass a non-forecast head. A pin-wait (the head's only
            # copy is on a disaggregation-pinned lane it is waiting to reuse) stays exempt: it is not funding a
            # fresh copy, so it bypasses unconditionally until the pin releases. When the window is spent (and no
            # pin-wait), the loop is skipped and dispatch falls through exactly as a non-forecast head does
            # today: the head keeps its slot claim and the room-making machinery runs, with no skips until it
            # dispatches.
            head_job_id = str(next_job.id_) if next_job.id_ is not None else None
            affinity_budget = affinity_budget_seconds(self._state.recent_job_ttl)
            within_affinity_budget = affinity_skip_allowed(
                self._affinity_skip_state,
                head_job_id,
                self._clock(),
                affinity_budget,
                _AFFINITY_MAX_SKIPS,
            )
            # Absolute age override: the affinity budget runs from the first bypass, not from pop, so a head
            # that aged in queue before its first skip could still be bypassed past a winnable window. Once the
            # head has waited more than the anti-starvation fraction of its ttl since pop, resident-model bypass
            # yields so its own preload runs (a pin-wait head stays exempt: it funds no fresh copy).
            if within_affinity_budget and self._head_aged_past_anti_starvation(next_job):
                within_affinity_budget = False
                self._note_anti_starvation_override(next_job)
            # While a whole-card residency claims the card, a foreign job must not be pulled forward onto it:
            # materialising another model there is precisely the weight eviction the residency was taken out
            # to prevent, and the job it would serve is one the queue can serve after the release. Only the
            # claimed model may still bypass, so a burst for it keeps flowing.
            pop_claim = self.whole_card_pop_claim()
            # A residency held for a model other than the head's bars the head from loading at all until it
            # drains, and the only work that drains it is that model's own queued jobs. Those jobs therefore
            # bypass regardless of the affinity budget, on the same ground as the pin-wait exemption: they
            # fund no fresh copy of anything, and the head was not going to run in the meantime, so no bypass
            # budget is being spent against it. The pop-claim restriction above still applies.
            residency_bypass_models = {
                residency.model
                for _residency_device_index, residency in self._held_residencies()
                if residency.model is not None and residency.model != next_job.model
            }
            if pinned_resident is not None or within_affinity_budget or residency_bypass_models:
                for candidate_job in next_n_jobs:
                    if candidate_job.model is None or candidate_job.model == next_job.model:
                        continue
                    if pop_claim is not None and candidate_job.model != pop_claim.model:
                        continue
                    if (
                        pinned_resident is None
                        and not within_affinity_budget
                        and candidate_job.model not in residency_bypass_models
                    ):
                        continue
                    # A degraded retry must run isolated (see the diversity path), so it is never a bypass target.
                    if self._job_tracker.is_degraded_dispatch_pending(candidate_job):
                        continue
                    candidate_process = self._resident_process_for_job(candidate_job)
                    if candidate_process is not None and candidate_process.can_accept_job():
                        line_skip = LineSkip(displaced_job=next_job, reason="resident_bypass")
                        next_job = candidate_job
                        process_with_model = candidate_process
                        break

            if process_with_model is None:
                if pinned_resident is not None:
                    # The head's only resident copy is on a disaggregation-pinned sampler lane. Never fund a
                    # fresh preload (it cannot fit beside the pinned residents and would wedge the card); hold
                    # the head's queue position until the pin releases and the resident lane becomes
                    # dispatchable, then it dispatches onto that lane priced as already resident.
                    #
                    # Holding does not mean idling. A disaggregation-eligible head needs the component lane and
                    # the encode working set to produce its conditioning, not a free sampler, so that half of
                    # its pipeline can run against the pinned lane's sample rather than behind it. Admit it into
                    # the disaggregated pipeline now with no sampler; it binds the lane when the pin releases.
                    if not information_only and self._stage_ahead_of_pin_enabled:
                        await self._stage_head_ahead_of_pin(next_job, pinned_resident)
                    return None

                next_job_model = next_job.model
                if next_job_model is None:
                    raise ValueError(f"next_job.model is None ({next_job})")

                # The model is resident, but only on a card that cannot serve this job. Nothing is
                # missing, so the recovery must not run: expiring the entry would discard a truthful
                # residency and latch the missing-model flag against a card that is serving. The job needs
                # its own copy on an eligible card, which the preload pass admits.
                if self._resident_only_on_ineligible_cards(next_job):
                    return None

                if (
                    self._preload_delay_notified
                    or self._horde_model_map.is_model_loading(next_job_model)
                    or information_only
                ):
                    return None
                await self._handle_process_missing(next_job)
                return None

        if not process_with_model.can_accept_job():
            # The head's own process is busy sampling its model, so the head cannot run yet. Rather than
            # idle a free inference process, fill it with a pending job for a *different* model already
            # resident there: a multi-threaded worker covers more distinct models per concurrent slot and
            # avoids duplicate-loading the head's model. The head keeps its queue position via the
            # line-skip. Falls through to None when nothing distinct is runnable (so a run of same-model
            # jobs still waits for the busy process rather than duplicating its model).
            diversity = self._select_idle_thread_diversity_job(next_job, next_n_jobs)
            if diversity is None:
                return None
            diversity_job, diversity_process = diversity
            line_skip = LineSkip(displaced_job=next_job, reason="diversity")
            next_job = diversity_job
            process_with_model = diversity_process

        dispatch_scope = process_with_model.device_index if self._multi_gpu_routing_active else None
        if not self._job_tracker.is_admitted_exclusive(next_job) and self._exclusive_dispatch_suppression_active(
            dispatch_scope,
        ):
            self._note_exclusive_dispatch_suppression(next_job, dispatch_scope)
            return None

        self._model_recently_missing = False

        if (
            not information_only
            and line_skip is None
            and not self._resident_whole_card_head_ready(next_job, process_with_model)
        ):
            self._pending_line_skip = None
            return None

        # Hold a would-be concurrent job back until the in-flight job(s) have made size-appropriate
        # headway, so two heavy models (or a batch / extra-large model) do not stack their loads and
        # activation peaks on the card and thrash a sampler into a watchdog teardown. The line-skip
        # bypass is exempt: it is a resident-model bypass or diversity fill that is already size-appropriate
        # and keeps the GPU fed while the displaced head waits on its own load. information_only look-ahead
        # is not gated here so callers still see the next job; the real dispatch path below enforces the hold.
        if (
            not information_only
            and line_skip is None
            and not self._concurrent_overlap_allowed(next_job, target_device_index=process_with_model.device_index)
        ):
            self._pending_line_skip = None
            return None

        next_job_and_process = NextJobAndProcess(
            next_job=next_job,
            process_with_model=process_with_model,
            line_skip=line_skip,
        )

        if line_skip is not None:
            self._pending_line_skip = next_job_and_process

        return next_job_and_process

    def _should_keep_model_resident(
        self,
        dispatched_job: ImageGenerateJobPopResponse,
        *,
        process_with_model: HordeProcessInfo,
        device_index: int | None,
    ) -> bool:
        """Whether ``dispatched_job``'s model should stay resident in VRAM after it runs.

        hordelib evicts the model from VRAM after every job so sibling GPU instances never collectively
        over-commit; that eviction forces a RAM->VRAM reload on the next job, which is the dominant
        non-sampling cost on small jobs (a same-model successor on the same process pays it for weights that
        were still on the card). Retention suppresses that eviction for one job. Because eviction is now both
        on-demand and *proven* (the device-free governor reads truthful NVML device-free, and the verified
        reclaim ladder takes residents back rung by rung with each free confirmed at the device level),
        retention no longer has to be preemptively stingy about *fit*: weights may stay resident while the card
        is healthy, and the ladder reclaims them the instant any overcommit picture appears. What it does have
        to be stingy about is *evidence*, because a copy nothing comes back for is not free: it holds the card,
        prices every later grant's static fit against itself, and is handed back having saved nothing. On a
        diverse offer that is most copies, and the accumulated holds recreate the pressure the ladder then has
        to resolve. The grant therefore needs:

        - **Repeat evidence on this slot**: the dispatched model appears among the slot's previous
          :data:`_RETENTION_REPEAT_EVIDENCE_DISPATCHES` dispatches. Retention only ever pays off through a
          same-model successor on the same slot, so a slot whose recent traffic has never repeated this model
          is being asked to hold weights on no prediction at all. This adapts to whatever the operator offers
          without encoding an assumed mix: a slot serving a single-model pool repeats on every dispatch and is
          granted exactly as freely as an ungated policy would grant it, while a slot rotating more models than
          the window holds earns close to nothing, which is the traffic shape where retention was pure cost.
        - **Card healthy**: the device-free governor's committed state for this card is HEALTHY. A PRESSURE
          or SATURATED card is one the ladder is or may soon be reclaiming from, so it is handed no new
          resident to evict. This state is derived from the one figure a WDDM driver cannot misreport under
          demand-paging (NVML device-free), so it holds precisely in the regime measured free VRAM lies.
        - **Static fit**: the card's reported total VRAM must absorb this job's sampling peak plus the
          measurement noise buffer (and any committed in-flight reserves), after charging the sibling CUDA
          contexts, the models other slots are already holding resident under earlier grants, and the job's
          own post-processing that share the card while the weights are held. Retention is cumulative: each
          grant leaves weights on the card that the next grant's peak must fit beside, so a fit that counts
          only contexts lets a run of grants sum past the card and hand the driver the overflow. The margin
          added on top of the peak is the admission noise buffer, not the operator's configured
          ``vram_reserve_mb``: that reserve is a sampling / co-residency headroom term, and enforcing it as a
          hard static-fit floor stacks it on an already activation-inclusive learned peak, denying retention on
          a small card by a few dozen MB and forcing a re-transfer every job. The total is a constant the driver
          cannot misreport under pressure, so a model too large to hold at all is refused regardless of what
          "free" claims.

        The measured admission floor is deliberately *not* re-checked here: it is the admission/dispatch
        gate's job, and retaining already-materialized weights adds zero new bytes to the card, so a measured
        veto in this seam only reintroduces the never-fires problem via committed-figure noise. Nor is sole
        residency required: the governor plus the verified ladder make a second idle resident safe (it is a
        first-class ladder reclaim candidate), so retention may keep weights warm even while a sibling holds
        its own resident model.

        The evidence is *trailing*, never a queue lookahead, and the distinction is the whole of why this gate
        can exist at all. The pop cycle refills the queue immediately *after* a dispatch drains it, so at the
        dispatch instant a same-model successor is almost never visible in the pending set even when one
        arrives milliseconds later; a gate asking what is queued would therefore refuse a pool-locked worker
        every grant and make retention structurally unreachable. What the slot has already been asked to run is
        under no such timing, and on any traffic whose model mix is stable enough for retention to help, it
        predicts the successor the queue cannot yet show.

        A grant that goes unused is still bounded rather than permanent. Eviction is just-in-time: a
        cross-model preload that no longer fits because idle residents hold the card defers while the ladder
        evicts them, the under-pressure reclaim overrides retention outright, and a hold nothing has come back
        for is swept off a card that stays under pressure
        (:meth:`_revoke_stale_retentions_under_pressure`).

        A missing budget, unreported total, or unpriceable sibling overhead yields False: retention is granted
        on evidence, never assumed. Even when granted, hordelib's force-load overflow guard remains the hard
        backstop, so a wrong call degrades to a reload rather than an OOM.
        """
        model = dispatched_job.model
        if model is None:
            return False
        if self._runtime_config.bridge_data.legacy_comfy_vram_unload is True:
            # Under the legacy regime the child's executor returns the card at the end of every prompt, below
            # anything the grant can suppress. Recording a grant there would have the parent track, wait on,
            # and charge for weights that are no longer resident.
            self._log_retention_decision(
                model,
                process_with_model,
                granted=False,
                reason="actuation disabled (legacy_comfy_vram_unload): the child unloads at end of prompt regardless",
                denial_reason=RetentionDenialReason.ACTUATION_DISABLED,
            )
            return False
        if not self._budget_active():
            self._note_retention_denial(RetentionDenialReason.BUDGET_INACTIVE)
            return False
        if self._wddm_paging_active:
            # The driver is already demand-paging the worker's allocations; holding weights across jobs
            # in that regime can only deepen it.
            self._note_retention_denial(RetentionDenialReason.WDDM_PAGING)
            return False
        governor_state = self.governor_state(device_index)
        if governor_state is not GovernorState.HEALTHY:
            self._log_retention_decision(
                model,
                process_with_model,
                granted=False,
                reason=f"governor: card {governor_state.value} (the reclaim ladder holds priority over new residents)",
                denial_reason=RetentionDenialReason.GOVERNOR_STATE,
            )
            return False
        if not self._slot_has_repeat_evidence(process_with_model.process_id, model):
            # Nothing this slot has recently run predicts a same-model successor, so the hold would be paid for
            # on no evidence. Checked ahead of the VRAM arithmetic because a grant that fits is still waste.
            self._log_retention_decision(
                model,
                process_with_model,
                granted=False,
                reason=(
                    f"no repeat evidence: {model} is absent from this slot's last "
                    f"{_RETENTION_REPEAT_EVIDENCE_DISPATCHES} dispatch(es)"
                ),
                denial_reason=RetentionDenialReason.NO_REPEAT_EVIDENCE,
            )
            return False
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        if total_vram_mb is None:
            self._note_retention_denial(RetentionDenialReason.STATIC_FIT)
            return False
        baseline = self._model_metadata.get_baseline(model)
        committed_reserve_mb = self._committed_vram_reserve_mb(device_index=device_index)
        static_charges_mb = self._retention_static_charges_mb(
            dispatched_job,
            baseline,
            process_with_model=process_with_model,
            device_index=device_index,
        )
        if static_charges_mb is None:
            self._log_retention_decision(
                model,
                process_with_model,
                granted=False,
                reason="static: sibling contexts present but per-context overhead not yet measured",
                denial_reason=RetentionDenialReason.UNPRICEABLE,
            )
            return False
        retained_resident_mb = self._retained_resident_charges_mb(
            dispatched_job,
            model,
            process_with_model=process_with_model,
            device_index=device_index,
        )
        if retained_resident_mb is None:
            self._log_retention_decision(
                model,
                process_with_model,
                granted=False,
                reason="static: a retained resident holds the card but its weight footprint is unpriceable",
                denial_reason=RetentionDenialReason.UNPRICEABLE,
            )
            return False
        static_available_mb = total_vram_mb - static_charges_mb - retained_resident_mb
        static_verdict = self._vram_budget.check_job(
            dispatched_job,
            baseline,
            static_available_mb,
            committed_reserve_mb=committed_reserve_mb,
            disaggregated=self._is_disaggregation_class_eligible(dispatched_job),
        )
        # De-stack the margin: the learned sampling peak is already activation-inclusive, and the operator's
        # configured vram_reserve_mb is a sampling / co-residency headroom term, not a static load-feasibility
        # floor. Enforcing that full reserve on top of the peak (as check_job's fits does) prices already
        # materialised weights off a small card and forces a re-transfer every job. The measurement margin for a
        # static fit is the admission noise buffer, the same slack the admission identity uses. The sibling
        # contexts (charged above at their truthful marginal) and the job's own post-processing are already
        # netted out of static_available_mb.
        predicted_mb = static_verdict.predicted_mb
        noise_mb = admission_noise_buffer_mb(total_vram_mb)
        effective_available_mb = static_available_mb - committed_reserve_mb
        granted = predicted_mb is None or (predicted_mb + noise_mb) <= effective_available_mb
        retained_detail = f", retained residents {retained_resident_mb:.0f}MB" if retained_resident_mb > 0 else ""
        self._log_retention_decision(
            model,
            process_with_model,
            granted=granted,
            reason=(
                f"static: peak {predicted_mb} + noise {noise_mb:.0f} vs {effective_available_mb:.0f}MB "
                f"(total {total_vram_mb:.0f}MB minus sibling contexts, the job's own post-processing, and "
                f"in-flight commitments{retained_detail})"
            ),
            denial_reason=None if granted else RetentionDenialReason.STATIC_FIT,
        )
        return granted

    def _log_retention_decision(
        self,
        model: str,
        process_with_model: HordeProcessInfo,
        *,
        granted: bool,
        reason: str,
        denial_reason: RetentionDenialReason | None = None,
    ) -> None:
        """Emit the per-dispatch retention verdict with the gate figures that produced it, and tally it.

        Args:
            model: The model the verdict covers.
            process_with_model: The slot the verdict was reached for.
            granted: Whether the weights are being left on the card.
            reason: The gate figures behind the verdict, for the log line.
            denial_reason: Which gate refused, for the session tally. None on a grant.
        """
        if granted:
            self._retention_grants_issued += 1
        elif denial_reason is not None:
            self._note_retention_denial(denial_reason)
        logger.debug(
            f"VRAM retention for {model} on process {process_with_model.process_id}: "
            f"{'granted' if granted else 'denied'} ({reason})",
        )

    def _retention_static_charges_mb(
        self,
        dispatched_job: ImageGenerateJobPopResponse,
        baseline: str | None,
        *,
        process_with_model: HordeProcessInfo,
        device_index: int | None,
    ) -> float | None:
        """VRAM (MB) the retention static gate must charge on top of the job's own sampling peak.

        Two costs share the card with retained weights but are invisible to the sampling-peak estimate
        and to the committed-reserve ledger at grant time:

        - **Sibling CUDA contexts**: every other live GPU process (inference siblings, the
          post-processing lane, the disaggregated VAE and component lanes, the on-GPU safety process)
          holds a context whether or not it holds a model. Charged at the measured marginal per-context
          cost (first-context overhead when no marginal was measured), matching how the streaming
          forecast counts them. Returns None when sibling contexts exist but no per-context cost has
          been measured yet: an unpriceable charge must deny the grant, not be waved through at zero.
        - **Idle lanes' held components**: a lane's component cache outlives every job boundary and is
          returned by nothing but an unload, so between jobs it is as real a tenant as a retained
          checkpoint. The parent already receives what each lane holds on every memory report.
        - **Concurrent clearance grants**: with more than one lease slot a sibling can be cleared into
          its own load-and-sample window while these weights are held, and its materialisation lands on
          the same card. A staged sibling has only its encode charge booked in the shared ledger, so the
          remainder of its full materialisation is charged here; a sibling already sampling had its
          reservation upgraded to that full peak at its clearance and is left to the ledger.
        - **The job's own post-processing chain**: a job that requests post-processing runs its
          upscaler/face-fixer right after sampling, precisely while retention is holding the weights.
          Its estimated peak only enters the committed ledger after inference finishes, one dispatch
          too late for this grant, so it is charged here up front.

        Both are static estimates: the gate must hold even when the driver's free figure cannot be
        trusted (WDDM demand-paging), so nothing here reads measured free VRAM.
        """
        safety_on_gpu = self._safety_on_gpu_permitted and not self._process_lifecycle.is_safety_gpu_paused
        sibling_contexts = 0
        for process_info in self._process_map.values():
            if process_info.process_id == process_with_model.process_id:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.process_type in (
                HordeProcessType.INFERENCE,
                HordeProcessType.POST_PROCESS,
                HordeProcessType.VAE_LANE,
                HordeProcessType.COMPONENT,
            ) or (process_info.process_type == HordeProcessType.SAFETY and safety_on_gpu):
                sibling_contexts += 1

        charges_mb = 0.0
        if sibling_contexts > 0:
            override_mb = self._config_overhead_override_mb()
            per_context_mb = self._overhead.marginal_mb(config_override_mb=override_mb, device_index=device_index)
            if per_context_mb is None:
                per_context_mb = self._overhead.per_process_mb(
                    config_override_mb=override_mb,
                    device_index=device_index,
                )
            if per_context_mb <= 0:
                return None
            charges_mb = sibling_contexts * per_context_mb

        if dispatched_job.payload.post_processing:
            own_post_processing_mb = predict_job_post_processing_vram_mb(dispatched_job, baseline)
            if own_post_processing_mb is not None:
                charges_mb += max(0.0, own_post_processing_mb)

        charges_mb += self._idle_lane_component_charges_mb(
            process_with_model=process_with_model,
            device_index=device_index,
        )
        concurrent_grant_mb = self._concurrent_clearance_grant_charges_mb(
            process_with_model=process_with_model,
            device_index=device_index,
        )
        if concurrent_grant_mb is None:
            return None
        charges_mb += concurrent_grant_mb

        return charges_mb

    def _idle_lane_component_charges_mb(
        self,
        *,
        process_with_model: HordeProcessInfo,
        device_index: int | None,
    ) -> float:
        """VRAM (MB) the card holds for component-cache entries idle lanes report between jobs.

        Read from the residency each lane reports rather than predicted, because a cache's contents are the
        lane's own history and nothing about the dispatched job says what they are. Only idle lanes are
        charged: a lane mid-stage has its work priced by that stage's own admission, and charging it here as
        well would price the same bytes twice. The target slot is excluded for the same reason its own
        retained weights are.
        """
        charges_mb = 0.0
        for process_info in self._process_map.values():
            if process_info.process_id == process_with_model.process_id:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.is_process_busy() or not process_info.held_components:
                continue
            charges_mb += sum(max(0.0, held.approx_ram_mb) for held in process_info.held_components)
        return charges_mb

    def _concurrent_clearance_grant_charges_mb(
        self,
        *,
        process_with_model: HordeProcessInfo,
        device_index: int | None,
    ) -> float | None:
        """VRAM (MB) a sibling's imminent lease window will materialise beside these retained weights.

        The clearance lease admits as many concurrent load-and-sample windows as it has slots. A retention
        grant priced against the card minus contexts alone assumes it is the only claim in flight, so on a
        multi-slot lease two grants can each fit "alone" and jointly overflow the card, which the driver then
        resolves by demand-paging or by failing the allocation outright.

        Only the part the shared ledger does not already carry is charged: a staged sibling has booked its
        encode working set and nothing else (its weights load at clearance), so the remainder of its full
        materialisation is charged here; a sibling already inside its window had its reservation upgraded to
        that same full peak when it was cleared and is left to the ledger the caller subtracts. With no lease
        a dispatch books its full charge at dispatch, so nothing is outstanding to add.

        Returns None when a sibling's staged job cannot be priced: an unpriceable concurrent claim denies the
        grant rather than being waved through, as every other unpriceable tenant here does.
        """
        if not self._runtime_config.bridge_data.gpu_sampling_lease_enabled:
            return 0.0
        charges_mb = 0.0
        for process_info in self._process_map.values():
            if process_info.process_id == process_with_model.process_id:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if process_info.last_process_state != HordeProcessState.INFERENCE_PRIMED:
                continue
            staged_job = process_info.current_inference_job()
            if staged_job is None or staged_job.model is None:
                continue
            staged_baseline = self._model_metadata.get_baseline(staged_job.model)
            materialisation_mb = (
                predict_job_sampler_only_vram_mb(staged_job, staged_baseline)
                if self._is_disaggregation_class_eligible(staged_job)
                else predict_job_sampling_vram_mb(staged_job, staged_baseline)
            )
            if materialisation_mb is None:
                return None
            charges_mb += max(0.0, materialisation_mb - _STAGING_ENCODE_VRAM_MB)
        return charges_mb

    def _retained_resident_charges_mb(
        self,
        dispatched_job: ImageGenerateJobPopResponse,
        dispatched_model: str,
        *,
        process_with_model: HordeProcessInfo,
        device_index: int | None,
        include_target_retained: bool = True,
    ) -> float | None:
        """VRAM (MB) of weights already held on the card by earlier retention grants.

        A grant is not free after it is issued: the retained weights stay on the device until an eviction
        actuates, so every later grant's sampling peak has to fit *beside* them. Charging only live CUDA
        contexts prices the card as if each grant were the only one, which lets a run of grants across
        sibling slots sum past the card total and leave the driver to demote the overflow.

        Charged per slot at the model's full resident footprint (weights plus the text encoders and VAE the
        engine force-loads), keyed on the baseline alone since weights do not scale with job shape:

        - every *other* slot on this card whose retained model the parent is tracking;
        - this slot's own retained model when it differs from the dispatched one, since those weights are
          still on the card at the grant instant. A same-model re-grant charges nothing: the retained
          weights and the dispatched job's weights are the same bytes, and the reuse is the point.

        ``include_target_retained`` is False for a caller whose load is preceded by an eviction of this
        slot's own retained weights (:meth:`evict_retained_resident_for_model_change`): those bytes are back
        on the card before the load, so charging them would price a tenant that is already leaving.

        Returns None when a tracked resident's footprint cannot be estimated: an unpriceable tenant denies
        the grant rather than being waved through at zero, matching how unpriceable sibling contexts are
        handled.
        """
        charges_mb = 0.0
        for process_info in self._process_map.values():
            if device_index is not None and process_info.device_index != device_index:
                continue
            retained_model = process_info.retained_resident_model
            if retained_model is None:
                continue
            if process_info.process_id == process_with_model.process_id and (
                retained_model == dispatched_model or not include_target_retained
            ):
                continue
            footprint_mb = self._retained_resident_footprint_mb(dispatched_job, process_info, retained_model)
            if footprint_mb is None:
                return None
            charges_mb += max(0.0, footprint_mb)
        return charges_mb

    def _retained_resident_footprint_mb(
        self,
        dispatched_job: ImageGenerateJobPopResponse,
        process_info: HordeProcessInfo,
        retained_model: str,
    ) -> float | None:
        """VRAM (MB) the weights ``process_info`` retains really occupy, or None when unpriceable.

        A whole-job residency is charged the model's full resident footprint (weights plus the text encoders
        and VAE the engine force-loads), keyed on the baseline alone since weights do not scale with job shape.

        A disaggregated sampler's residency is charged its UNet alone, from the checkpoint's component-identity
        sidecar, because that is all such a slot holds: its text encoders ran in the encode service and its VAE
        in the image lane. Charging it the whole checkpoint would price support weights no process holds, and
        this figure decides later grants and dispatch holds, so the over-charge would collapse exactly the
        co-residency disaggregation exists to buy. The component figure is the sidecar residual
        :func:`predict_job_unet_only_ram_mb` floors, the same reading
        :meth:`_disaggregated_component_charge_mb` admits a UNet stage against; that method's
        already-staged credit is deliberately not applied here, since it answers whether a *stage* materialises
        anything in RAM, while this answers what the device is holding right now.

        An unreadable sidecar returns None rather than falling back to the whole checkpoint: the fallback would
        be the over-charge this exists to remove, and an unpriceable tenant denies a grant instead of being
        waved through, matching how unpriceable sibling contexts are handled.
        """
        if process_info.retained_resident_component_only:
            sidecar = self._read_component_sidecar(retained_model)
            if sidecar is None:
                return None
            return predict_job_unet_only_ram_mb(sidecar.residual_tensor_bytes)
        # The footprint estimator keys on the baseline alone (weights do not scale with job shape);
        # the job argument only satisfies its signature.
        return predict_job_footprint_mb(dispatched_job, self._model_metadata.get_baseline(retained_model))

    def _fits_beside_retained_residents(
        self,
        job: ImageGenerateJobPopResponse,
        *,
        target: HordeProcessInfo,
        device_index: int | None,
        include_target_retained: bool = True,
    ) -> bool | None:
        """Whether loading ``job``'s weights onto ``target`` fits the card beside every retained resident.

        The static arithmetic of :meth:`_should_keep_model_resident`, asked in the other direction. A retained
        resident is a priced occupant of the card, never a bystander a fresh load can race: the reported card
        total (the one figure the driver cannot misreport while it is demand-paging), net of the sibling CUDA
        contexts and the job's own post-processing, net of the weights other slots hold under earlier grants,
        and net of the in-flight commitments, must absorb this job's sampling peak plus the admission noise
        buffer. Where that fails, a second copy of the same weights (or a second model beside them) is what
        the card is being asked to carry, which is the overcommit no later pricing can undo.

        ``include_target_retained`` is passed through to :meth:`_retained_resident_charges_mb`.

        Returns None when the fit cannot be judged on evidence: the budget is off, the card reported no total,
        a sibling context or a tracked resident is unpriceable, or the job has no peak estimate. Callers treat
        a None as no verdict and take their ordinary path, so this only ever acts on measurable arithmetic.
        """
        model = job.model
        if model is None or not self._budget_active():
            return None
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        if total_vram_mb is None:
            return None
        baseline = self._model_metadata.get_baseline(model)
        static_charges_mb = self._retention_static_charges_mb(
            job,
            baseline,
            process_with_model=target,
            device_index=device_index,
        )
        if static_charges_mb is None:
            return None
        retained_resident_mb = self._retained_resident_charges_mb(
            job,
            model,
            process_with_model=target,
            device_index=device_index,
            include_target_retained=include_target_retained,
        )
        if retained_resident_mb is None:
            return None
        committed_reserve_mb = self._committed_vram_reserve_mb(device_index=device_index)
        static_available_mb = total_vram_mb - static_charges_mb - retained_resident_mb
        static_verdict = self._vram_budget.check_job(
            job,
            baseline,
            static_available_mb,
            committed_reserve_mb=committed_reserve_mb,
            disaggregated=self._is_disaggregation_class_eligible(job),
        )
        predicted_mb = static_verdict.predicted_mb
        if predicted_mb is None:
            return None
        return (predicted_mb + admission_noise_buffer_mb(total_vram_mb)) <= (
            static_available_mb - committed_reserve_mb
        )

    def _coresident_lookahead_affordable(self, resident_model: str, *, device_index: int | None) -> bool:
        """Whether an idle resident copy of a queued model can coexist with the imminent job's sampling.

        Static accounting against the card's reported total (a constant the driver cannot misreport
        under memory pressure): the resident's full weight footprint plus the head-of-queue job's
        sampling peak plus the configured reserve must fit. On a card where they cannot, keeping the
        copy warm forces driver demand-paging during the head's sampling, which costs far more than the
        one reload the protection would have saved. Unknown figures (no total reported, no head job, no
        estimate) keep the protection: the affordability gate only ever *removes* protection on
        evidence.
        """
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        if total_vram_mb is None:
            return True
        pending = self._job_tracker.jobs_pending_inference
        head_job = pending[0] if len(pending) > 0 else None
        if head_job is None or head_job.model is None:
            return True
        head_peak_mb = predict_job_sampling_vram_mb(head_job, self._model_metadata.get_baseline(head_job.model))
        # The footprint estimator keys on the baseline alone (weights do not scale with job shape); the
        # job argument only satisfies its signature.
        resident_footprint_mb = predict_job_footprint_mb(head_job, self._model_metadata.get_baseline(resident_model))
        if head_peak_mb is None or resident_footprint_mb is None:
            return True
        return total_vram_mb - self._vram_budget.reserve_mb - head_peak_mb - resident_footprint_mb >= 0

    def pp_sampling_coresidency_affordable(
        self,
        *,
        sampling_peak_mb: float | None,
        pp_reserve_mb: float,
        device_index: int | None = None,
    ) -> bool:
        """Whether a sampling job and a post-processing chain can run on this card at the same time.

        Static accounting against the card's reported total (the driver's free figure is untrustworthy
        under WDDM demand-paging, precisely the failure this predicate prevents): the sampling peak, the
        chain's estimated peak, every extra live GPU context, and the configured reserve must all fit
        together. On a card where they cannot, co-running the two silently demand-pages both (sampling
        collapses to a fraction of its rate for the whole overlap), so the dispatch gates time-slice the
        card instead: whichever side arrives second waits for the first to finish. Unknown figures (no
        total, no peak estimate, unmeasured context cost) leave co-running allowed: this gate only ever
        restricts on evidence.
        """
        if sampling_peak_mb is None or pp_reserve_mb <= 0:
            return True
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        if total_vram_mb is None:
            return True
        override_mb = self._config_overhead_override_mb()
        per_context_mb = self._overhead.marginal_mb(config_override_mb=override_mb, device_index=device_index)
        if per_context_mb is None:
            per_context_mb = self._overhead.per_process_mb(config_override_mb=override_mb, device_index=device_index)
        if per_context_mb <= 0:
            return True
        safety_context = 1 if self._safety_on_gpu_permitted and not self._process_lifecycle.is_safety_gpu_paused else 0
        extra_contexts = (
            max(0, self._process_map.num_loaded_inference_processes(device_index=device_index) - 1) + safety_context
        )
        return (
            total_vram_mb
            - self._vram_budget.reserve_mb
            - sampling_peak_mb
            - pp_reserve_mb
            - extra_contexts * per_context_mb
            >= 0
        )

    def max_in_progress_sampling_peak_mb(self) -> float | None:
        """The largest sampling peak (MB) among jobs currently in progress, or None when idle.

        Each job's static sampling peak is raised by any learned SAMPLE-stage watermark for its footprint before
        the maximum is taken, so the post-processing co-residency gate this feeds prices in-flight sampling from
        measured activation high-waters, not a seed the hardware has already overshot.
        """
        peaks: list[float] = []
        for job in self._job_tracker.jobs_in_progress:
            if job.model is None:
                continue
            baseline = self._model_metadata.get_baseline(job.model)
            static_peak_mb = predict_job_sampling_vram_mb(job, baseline)
            if static_peak_mb is None:
                continue
            peaks.append(
                self._learned_sampling_peak_mb(
                    job,
                    baseline,
                    static_seed_mb=static_peak_mb,
                    stage=FootprintStage.SAMPLE,
                ),
            )
        if not peaks:
            return None
        return max(peaks)

    def estimate_disaggregated_sampling_peak_mb(self, job_info: HordeJobInfo) -> float | None:
        """Return a disaggregated job's estimated sampling-phase peak VRAM (MB), or None when unavailable.

        Injected into the disaggregation orchestrator's concurrent-sampling gate. Charges the whole-job
        sampling peak (:func:`predict_job_sampling_vram_mb`, weights plus the per-step activation working set),
        a deliberately conservative monolithic bound for v1 rather than the leaner sampler-only figure: the gate
        arbitrates whether two activation peaks may over-commit the card, and erring high there defers a second
        sampler that would otherwise drive the device into WDDM demand-paging. The static figure is the seed of a
        learned SAMPLE_ISOLATED-stage estimate (the disaggregated UNet-only sampler's own key, distinct from the
        monolithic whole-job SAMPLE key), so a measured isolated-sampler high-water for this (baseline,
        resolution, platform) raises the booked peak above the seed and never below it. None (no model, or no
        estimate) leaves the gate to admit, so a missing estimate never wedges the pipeline.
        """
        job = job_info.sdk_api_job_info
        if job.model is None:
            return None
        baseline = self._model_metadata.get_baseline(job.model)
        static_peak_mb = predict_job_sampling_vram_mb(job, baseline)
        if static_peak_mb is None:
            return None
        return self._learned_sampling_peak_mb(
            job,
            baseline,
            static_seed_mb=static_peak_mb,
            stage=FootprintStage.SAMPLE_ISOLATED,
        )

    def estimate_disaggregated_decode_spike_mb(self, job_info: HordeJobInfo) -> float | None:
        """Return a disaggregated job's VAE tiled-decode activation spike (MB), or None if unsizable.

        Injected into the disaggregation orchestrator's decode gate. The bounded tiled-decode activation working
        set (:func:`predict_job_decode_spike_mb`, the same figure the lane co-residency charge derives from) is
        the concurrent device commitment a decode adds while a sibling samples. None (no model, or a pinned
        hordelib that predates the decode-spike figure) prices the decode as unpriced, so the gate then only
        withholds a decode onto an already over-committed card rather than charging a phantom cost.
        """
        job = job_info.sdk_api_job_info
        if job.model is None:
            return None
        baseline = self._model_metadata.get_baseline(job.model)
        return predict_job_decode_spike_mb(job, str(baseline) if baseline is not None else None)

    def _vae_lane_decode_spike_charge_mb(self, *, device_index: int | None) -> float:
        """The VAE lane's concurrent decode spike (MB) to reserve out of sampling headroom, 0 when off-card.

        Under disaggregation the image lane VAE-decodes the previous job's latent while a sampler runs, so that
        bounded tiled-decode activation is a real concurrent device commitment that must not be handed to a
        second sampler. Charged only while the lane is enabled and on the GPU; sized from an in-flight job's
        bounded decode-spike estimate (via :meth:`_disaggregation_sibling_charge_mb`, which itself falls back to
        the conservative full lane quota when the pinned hordelib predates the decode-spike figure). Zero when
        no in-flight job can size it, so the headroom is never charged a phantom lane spike.
        """
        if not self._process_lifecycle.vae_lane_enabled() or self._process_lifecycle.is_vae_lane_gpu_paused:
            return 0.0
        if self._process_map.num_vae_lane_processes(device_index=device_index) <= 0:
            return 0.0
        for job in self._job_tracker.jobs_in_progress:
            if job.model is None:
                continue
            return self._disaggregation_sibling_charge_mb(
                job,
                self._model_metadata.get_baseline(job.model),
                device_index=device_index,
            )
        return 0.0

    def _sampling_peak_mb(self, job: ImageGenerateJobPopResponse) -> float | None:
        """Return the sampling peak estimate for ``job``, raised by any learned SAMPLE-stage watermark."""
        if job.model is None:
            return None
        baseline = self._model_metadata.get_baseline(job.model)
        static_peak_mb = predict_job_sampling_vram_mb(job, baseline)
        if static_peak_mb is None:
            return None
        return self._learned_sampling_peak_mb(
            job,
            baseline,
            static_seed_mb=static_peak_mb,
            stage=FootprintStage.SAMPLE,
        )

    def _pending_post_processing_reserve_mb(self, *, device_index: int | None) -> float:
        """Return the smallest known pending post-processing peak for an idle lane on ``device_index``.

        The lane orchestrator scans for the first pending chain that can run. For the dispatch-side hold, the
        smallest known pending peak is enough: if the next sampler cannot share the card with even that chain,
        starting it would extend the no-drain window for every pending chain. Unknown estimates do not hold
        inference; every memory gate in this scheduler restricts only on evidence.
        """
        post_process_process = self._process_map.get_first_available_post_process_process()
        if post_process_process is None:
            return 0.0
        if device_index is not None and post_process_process.device_index != device_index:
            return 0.0

        estimates_mb: list[float] = []
        for job_info in self._job_tracker.jobs_pending_post_processing:
            sdk_job = job_info.sdk_api_job_info
            baseline = self._model_metadata.get_baseline(sdk_job.model) if sdk_job.model is not None else None
            baseline_name = str(getattr(baseline, "value", baseline)) if baseline is not None else None
            estimate = predict_job_post_processing_vram_mb(sdk_job, baseline_name)
            if estimate is None or estimate <= 0:
                continue
            estimates_mb.append(estimate)

        if not estimates_mb:
            return 0.0
        return min(estimates_mb)

    def _measured_device_free_mb(self, device_index: int | None) -> float | None:
        """Return the parent's measured device-free VRAM (MB) for this card, or None when unread.

        A None ``device_index`` (single-GPU default) resolves to card 0, matching how the scheduler keys
        its per-card admission state. With no provider wired or no reading yet the answer is None, so the
        measured admission path is unavailable and the caller keeps the static verdict.
        """
        if self._device_free_mb_provider is None:
            return None
        return self._device_free_mb_provider(0 if device_index is None else device_index)

    def measured_device_free_mb(self, device_index: int | None = None) -> float | None:
        """The parent's measured device-free VRAM (MB) for a card, for the stage dispatchers to pass on.

        A disaggregated stage is dispatched by the orchestrator rather than through this scheduler's own
        dispatch path, so it needs the same reading that path puts on a monolithic dispatch: the child's
        process-local free view overstates the card under WDDM, and a sampler is the process whose loads and
        activation reach the card hardest.
        """
        return self._measured_device_free_mb(device_index)

    def _pp_overlap_margin_mb(self, next_job: ImageGenerateJobPopResponse) -> float:
        """Return the co-residency measured second-say margin (MB) that applies to this candidate job.

        Defaults to :data:`_PP_OVERLAP_MEASURED_MARGIN_MB`. A disaggregation-class-eligible job holds only its
        UNet-only sampler peak, so its true overlap headroom differs from a monolithic dispatch; when the
        operator has set ``pp_overlap_margin_mb_disaggregated`` to a real number and this job is class-eligible,
        that override is used in its place. The setting is read with explicit None/float coercion (never bare
        truthiness), so a partially-mocked or unset config falls back to the default. Monolithic-path jobs
        always use the default regardless of the setting.
        """
        override_mb = config_number(self._runtime_config.bridge_data.pp_overlap_margin_mb_disaggregated)
        if override_mb is not None and self._is_disaggregation_class_eligible(next_job):
            return override_mb
        return _PP_OVERLAP_MEASURED_MARGIN_MB

    def _pp_overlap_measured_admits(
        self,
        *,
        sampling_peak_mb: float | None,
        pending_pp_reserve_mb: float,
        device_index: int | None,
        margin_mb: float,
    ) -> bool:
        """Whether measured device truth admits a sampling/post-processing overlap the static gate withheld.

        Consulted only after :meth:`pp_sampling_coresidency_affordable` (static accounting against the
        card's reported total) has said the two cannot share the card. The measured device-free reading
        already reflects an active chain's real allocations, so it is a more accurate second admission path
        than the ledger's worst-case reserve. It admits when the measured free, net of the reserve, the
        sampling peak, and any not-yet-allocated pending chain's predicted reserve, clears ``margin_mb`` (the
        default :data:`_PP_OVERLAP_MEASURED_MARGIN_MB`, or the disaggregation override the caller resolved).

        The path is unavailable (returns False, so the static fence stands) whenever the driver is
        demand-paging (its free figure is then untrustworthy, the exact regime the static gate guards), the
        sampling peak is unpriceable, or no measured reading exists. ``pending_pp_reserve_mb`` is zero for an
        active chain (whose allocations the measured free already reflects) and the predicted reserve for a
        pending chain (which has not allocated yet, so its cost must still be charged).
        """
        if self._wddm_paging_active:
            return False
        if sampling_peak_mb is None:
            return False
        device_free_mb = self._measured_device_free_mb(device_index)
        if device_free_mb is None:
            return False
        headroom_mb = device_free_mb - self._vram_budget.reserve_mb - sampling_peak_mb - pending_pp_reserve_mb
        return headroom_mb >= margin_mb

    def _resolve_pp_coresidency_hold(
        self,
        next_job: ImageGenerateJobPopResponse,
        *,
        static_affordable: bool,
        sampling_peak_mb: float | None,
        pending_pp_reserve_mb: float,
        device_index: int | None,
        static_hold_message: str,
    ) -> bool:
        """Resolve one co-residency branch to a hold/admit verdict, consulting the measured path on a static miss.

        When the static gate admits, dispatch proceeds and both edge-log latches clear. When it withholds,
        the measured-truth path (:meth:`_pp_overlap_measured_admits`) gets a second say: if it admits, the
        dispatch proceeds and the admission is edge-logged once with its arithmetic; otherwise the hold
        stands and is edge-logged once with ``static_hold_message``.
        """
        if static_affordable:
            self._pp_mutex_hold_logged = False
            self._pp_mutex_measured_admit_logged = False
            return False

        margin_mb = self._pp_overlap_margin_mb(next_job)
        if self._pp_overlap_measured_admits(
            sampling_peak_mb=sampling_peak_mb,
            pending_pp_reserve_mb=pending_pp_reserve_mb,
            device_index=device_index,
            margin_mb=margin_mb,
        ):
            if not self._pp_mutex_measured_admit_logged:
                self._pp_mutex_measured_admit_logged = True
                device_free_mb = self._measured_device_free_mb(device_index)
                logger.info(
                    f"Admitting dispatch of {next_job.model} via measured device truth that static "
                    f"co-residency held: {device_free_mb:.0f}MB free, {self._vram_budget.reserve_mb:.0f}MB "
                    f"reserve, {(sampling_peak_mb or 0.0):.0f}MB sampling peak, "
                    f"{pending_pp_reserve_mb:.0f}MB pending chain reserve, "
                    f"{margin_mb:.0f}MB margin.",
                )
            self._pp_mutex_hold_logged = False
            return False

        self._pp_mutex_measured_admit_logged = False
        if not self._pp_mutex_hold_logged:
            self._pp_mutex_hold_logged = True
            logger.info(f"{static_hold_message} (measured second-say margin {margin_mb:.0f}MB)")
        return True

    def _should_defer_dispatch_for_post_processing(
        self,
        next_job: ImageGenerateJobPopResponse,
        *,
        process_with_model: HordeProcessInfo | None = None,
    ) -> bool:
        """Whether this dispatch must wait for post-processing to release or receive the card.

        The counterpart of the orchestrator's chain-admission gate: together they time-slice a card that
        cannot hold a sampling peak and an upscale chain at once. Active chains hold dispatch until their
        result lands. Pending chains can also hold dispatch before the next sampler starts; otherwise a
        fresh sampler can keep the card never-idle and prevent the pending lane work from ever getting its
        turn.

        Each branch first prices co-residency statically against the card's reported total, then, on a
        static miss, gives the parent's measured device-free reading a second say (see
        :meth:`_pp_overlap_measured_admits`): measured truth reflects an active chain's real allocations and
        so admits overlaps the ledger's worst-case reserve would needlessly hold, without over-committing
        the card or trusting the driver's free figure while it is paging.
        """
        device_index = process_with_model.device_index if process_with_model is not None else None

        pp_committed_mb = self._reserve_ledger.total_vram_mb() - self._reserve_ledger.total_vram_mb_excluding(
            POST_PROCESS_RESERVE_FLOW,
        )
        pp_busy = any(
            process_info.process_type == HordeProcessType.POST_PROCESS
            and process_info.is_process_busy()
            and (device_index is None or process_info.device_index == device_index)
            for process_info in self._process_map.values()
        )
        if pp_committed_mb > 0 and pp_busy:
            sampling_peak_mb = self._sampling_peak_mb(next_job)
            affordable = self.pp_sampling_coresidency_affordable(
                sampling_peak_mb=sampling_peak_mb,
                pp_reserve_mb=pp_committed_mb,
                device_index=device_index,
            )
            return self._resolve_pp_coresidency_hold(
                next_job,
                static_affordable=affordable,
                sampling_peak_mb=sampling_peak_mb,
                # The active chain has already allocated, so the measured free reflects it: charge no extra.
                pending_pp_reserve_mb=0.0,
                device_index=device_index,
                static_hold_message=(
                    f"Holding dispatch of {next_job.model}: an in-flight post-processing chain "
                    f"({pp_committed_mb:.0f}MB committed) and this job's sampling peak cannot share the card; "
                    "dispatching when the chain finishes."
                ),
            )

        pending_pp_reserve_mb = self._pending_post_processing_reserve_mb(device_index=device_index)
        if pending_pp_reserve_mb <= 0:
            self._pp_mutex_hold_logged = False
            self._pp_mutex_measured_admit_logged = False
            return False

        sampling_peak_mb = self._sampling_peak_mb(next_job)
        affordable = self.pp_sampling_coresidency_affordable(
            sampling_peak_mb=sampling_peak_mb,
            pp_reserve_mb=pending_pp_reserve_mb,
            device_index=device_index,
        )
        return self._resolve_pp_coresidency_hold(
            next_job,
            static_affordable=affordable,
            sampling_peak_mb=sampling_peak_mb,
            # The pending chain has not allocated yet, so its predicted reserve must still be charged.
            pending_pp_reserve_mb=pending_pp_reserve_mb,
            device_index=device_index,
            static_hold_message=(
                f"Holding dispatch of {next_job.model}: pending post-processing "
                f"({pending_pp_reserve_mb:.0f}MB estimated) needs the next drain window and this job's "
                "sampling peak cannot share the card; dispatching after the lane gets its turn."
            ),
        )

    def note_wddm_paging(self, elevated_shared_mb_by_pid: dict[int, float], *, active: bool) -> None:
        """Record the parent's WDDM demand-paging verdict and reclaim idle VRAM on its rising edge.

        ``elevated_shared_mb_by_pid`` names the worker child PIDs whose shared (system-backed) GPU usage
        crossed the paging threshold: measured attribution that the *worker's own* allocations were
        demoted out of dedicated VRAM. While active, retention is denied outright (holding weights in a
        regime the driver is already paging can only deepen it). The rising edge additionally reclaims idle
        resident VRAM, routed through the same LIFO reclaim policy the governor's ladder uses (newest idle
        resident first).

        The PDH-flagged process is deliberately NOT protected. The old sweep spared it on the assumption its
        model was the one in use, but under WDDM the driver demotes the least-recently-touched allocator, so
        the flagged process is usually the idle newcomer that just materialized weights, not the active
        sampler. Protecting it therefore spared exactly the squatter that should be evicted first. Immunity is
        instead structural: the reclaim ladder's candidate assembly excludes every actively-sampling process,
        so a busy slot is never swept whatever PDH flagged, and the newest idle resident (the likeliest
        squatter) is the first eviction target.
        """
        was_active = self._wddm_paging_active
        self._wddm_paging_active = active
        # Persist the victim set on every active verdict so recency tracks the latest sample, and clear it
        # the instant paging clears so a stale set cannot outlive the pressure that produced it.
        if active:
            self._wddm_paging_victims_shared_mb_by_pid = dict(elevated_shared_mb_by_pid)
            self._wddm_paging_victims_updated_monotonic = time.monotonic()
        else:
            self._wddm_paging_victims_shared_mb_by_pid = {}
        if not active or was_active:
            return

        detail = ", ".join(
            f"pid {pid}: {shared_mb:.0f}MB shared" for pid, shared_mb in sorted(elevated_shared_mb_by_pid.items())
        )
        logger.warning(
            "WDDM demand-paging detected on worker processes "
            f"({detail}); the driver demoted their VRAM allocations to system memory. "
            "Denying model retention and reclaiming idle resident VRAM (newest idle resident first).",
        )

        # Reclaim idle resident models in LIFO order through the single reclaim policy: build the ordered
        # ladder and issue each idle-model unload rung via the same actuator the governor's ladder uses. A
        # busy process is never a candidate, so an actively-sampling slot is untouched.
        ladder = build_reclaim_ladder(self.build_reclaim_ladder_candidates(None))
        for rung in ladder:
            if rung.kind is ReclaimRungKind.UNLOAD_IDLE_MODEL and rung.target_process_id is not None:
                self.unload_idle_model(rung.target_process_id, rung.device_index)

    def wddm_paging_victim_shared_mb_by_pid(self, max_age_seconds: float) -> dict[int, float]:
        """Return the fresh WDDM paging-victim map (os_pid -> shared MB), or empty when none is current.

        The map names the worker child PIDs whose VRAM the driver most recently demoted to system memory
        and by how much (their shared, system-backed GPU MB). It is returned only while it is younger than
        ``max_age_seconds``; a stale or absent verdict yields an empty map, so a caller can never act on a
        paging episode that has already cleared or whose telemetry has stopped arriving. The per-PID
        figures are diagnostic hints only: the counter is unreliable sample-to-sample and the demoted PID
        is usually the idle newcomer rather than the slow sampler, so no reclaim or kill decision gates on
        membership in this map.
        """
        if not self._wddm_paging_victims_shared_mb_by_pid:
            return {}
        if (time.monotonic() - self._wddm_paging_victims_updated_monotonic) > max_age_seconds:
            return {}
        return dict(self._wddm_paging_victims_shared_mb_by_pid)

    def _log_job_dispatch_details(self, next_job: ImageGenerateJobPopResponse) -> None:
        """Log the model, conditioning extras, and the resolution/steps/sampler line for a dispatching job.

        Side-effect-only diagnostics emitted just before an inference dispatch; it reads the job payload
        and writes log lines, mutating no scheduler state.
        """
        # Every line here carries payload text the horde supplied (model, sampler, workflow, control type),
        # so the text is passed as the ``message`` formatting argument rather than spliced into the colour
        # template: loguru parses only the template for tags, so payload markup cannot abort the line.
        color_format_string = "<fg #f0beff>{message}</>"

        logger.opt(colors=True).info(
            color_format_string,
            message=f"  Model: {next_job.model}",
        )
        if next_job.source_image is not None:
            logger.opt(colors=True).info(
                color_format_string,
                message="  Using source image",
            )

        extra_info = ""
        if next_job.payload.control_type is not None:
            extra_info += f"Control type: {next_job.payload.control_type}"
        if next_job.payload.loras:
            if extra_info:
                extra_info += ", "
            extra_info += f"{len(next_job.payload.loras)} LoRAs"
        if next_job.payload.tis:
            if extra_info:
                extra_info += ", "
            extra_info += f"{len(next_job.payload.tis)} TIs"
        if next_job.payload.post_processing is not None and len(next_job.payload.post_processing) > 0:
            if extra_info:
                extra_info += ", "
            extra_info += f"Post processing: {next_job.payload.post_processing}"
        if next_job.payload.hires_fix:
            if extra_info:
                extra_info += ", "
            extra_info += "HiRes fix"

        if next_job.payload.workflow is not None:
            if extra_info:
                extra_info += ", "
            extra_info += f"Workflow: {next_job.payload.workflow}"

        if extra_info:
            logger.opt(colors=True).info(
                color_format_string,
                message=f"  {extra_info}",
            )

        logger.opt(colors=True).info(
            color_format_string,
            message=f"  {next_job.payload.width}x{next_job.payload.height} for "
            f"{next_job.payload.ddim_steps} steps "
            f"with sampler {next_job.payload.sampler_name} for a batch of {next_job.payload.n_iter}",
        )

        logger.debug(f"All Batch IDs: {next_job.ids}")

    async def _stage_head_ahead_of_pin(
        self,
        next_job: ImageGenerateJobPopResponse,
        pinned_resident: HordeProcessInfo,
    ) -> None:
        """Admit a pin-waiting head into the disaggregated pipeline with no sampler, so its encode runs now.

        On a same-model streak the head's only copy of the weights is on the lane sampling the job in front of
        it, so the head cannot be dispatched and, until it is, nothing of its pipeline runs. Its text encode
        does not need that lane: it needs the component lane and the encode working set. Registering the job
        here starts that encode against the in-flight sample instead of after it, and leaves the sample stage
        the only thing the pin release still gates. The orchestrator binds the lane in ``_resolve_sampler`` the
        moment the reservation drops, which is also where the slot-side dispatch records are made
        (:meth:`note_disaggregated_sampler_bound`); no weights are preloaded anywhere by this.

        Nothing here changes what may be dispatched: the concurrency cap, the exclusive-admit suppression and
        the whole-card pop claim have all already been applied to this head by the caller, and the staged job
        counts against the cap exactly as a dispatched one does (its relaxation to the process ceiling is what
        prices the encode working set of a staged job in the first place). The charge is booked against the
        component lane, the process that actually holds the conditioning, and is returned at the bind
        (:meth:`note_disaggregated_sampler_bound`) or, on any exit that leaves the job unbound, by the dispatch
        flow's reconcile-by-omission once the job is out of the in-progress set.
        """
        if self._register_disaggregated_job is None or next_job.id_ is None:
            return
        if not self._is_disaggregatable_job(next_job):
            return
        encode_lane = self._process_map.get_component_process()
        if encode_lane is None:
            # Without a component lane there is no encode to run ahead, and admitting the job would only park
            # it in the pipeline until the pin releases, which is what holding here already does.
            return
        if not await self._register_disaggregated_job(next_job, None):
            return
        device_index = pinned_resident.device_index if self._multi_gpu_routing_active else None
        await self._job_tracker.mark_inference_started(
            next_job,
            device_index=device_index,
            whole_card=self._serving_under_whole_card(next_job.model, device_index),
        )
        self._record_dispatch_reservation(
            next_job,
            encode_lane,
            baseline=self._model_metadata.get_baseline(next_job.model) if next_job.model is not None else None,
            staging_only=True,
        )
        logger.info(
            f"Job {str(next_job.id_)[:8]} staged ahead of the sampler pin on process "
            f"{pinned_resident.process_id}: its text encode runs on the component lane while that lane samples.",
        )

    async def _dispatch_disaggregated(
        self,
        next_job: ImageGenerateJobPopResponse,
        process_with_model: HordeProcessInfo,
        *,
        keep_model_resident_after: bool = False,
        dispatched_device_index: int | None,
        degraded_dispatch: bool,
    ) -> bool:
        """Register a disaggregation-eligible job with the orchestrator, pinned to its scheduled process.

        Replaces the monolithic START_INFERENCE at this seam: the orchestrator reserves ``process_with_model``
        as the job's sampler (so the scheduler cannot double-book it), and this applies the same job-progress
        marking a monolithic dispatch does, so concurrency accounting and the orphaned-job watchdog see the
        job as owned. Returns False when the router declines (a role went unhealthy), so the caller falls back
        to a monolithic dispatch.

        The retention verdict is carried here exactly as a monolithic dispatch carries it, marked
        component-only: the sampler runs the same end-of-run eviction every other job does, so a stage
        dispatched without the grant returns the card and the next same-model sample re-uploads the UNet. What
        the slot then holds is that UNet alone, which is what the grant is recorded and priced as. It defaults
        to denied because that is the fail-safe direction: an absent grant costs a reload, while a grant nobody
        asked for records weights the device does not hold.
        """
        assert self._register_disaggregated_job is not None
        model = next_job.model
        if model is None:
            return False
        registered = await self._register_disaggregated_job(next_job, process_with_model)
        if not registered:
            return False

        await self._job_tracker.mark_inference_started(
            next_job,
            device_index=dispatched_device_index,
            whole_card=self._serving_under_whole_card(model, dispatched_device_index),
            process_age_seconds=time.time() - process_with_model.spawned_at,
        )
        self._commit_disaggregated_sampler_slot(
            next_job,
            process_with_model,
            keep_model_resident_after=keep_model_resident_after,
        )
        if degraded_dispatch:
            self._job_tracker.clear_degraded_dispatch(next_job)
        return True

    def _commit_disaggregated_sampler_slot(
        self,
        next_job: ImageGenerateJobPopResponse,
        process_with_model: HordeProcessInfo,
        *,
        keep_model_resident_after: bool,
    ) -> None:
        """Make the slot-side records of a dispatch on the process now pinned as ``next_job``'s sampler.

        Shared by the two ways a sampler becomes a job's: pinned at admission, or bound later for a job staged
        ahead of that pin. Everything recorded here describes a slot that is running the job, so it belongs to
        the binding instant in both cases; a staged-ahead job that has not bound owns no slot and must carry
        none of it.
        """
        model = next_job.model
        if model is None:
            return
        # The pinned process references this job so the orphaned-job watchdog credits it as owned across the
        # whole encode-and-sample window (the reservation, not a START_INFERENCE flag, is that ownership
        # record: see WorkerRecoveryCoordinator.inference_slot_owns_job). No sampling-timing stamp is set here;
        # the sampler reports its own INFERENCE_STARTING when the sample stage runs.
        tracked = self._job_tracker.get_tracked_job(next_job.id_) if next_job.id_ is not None else None
        process_with_model.record_inference_ownership(
            next_job,
            attempt_ordinal=(tracked.inference_attempts + 1) if tracked is not None else 1,
        )
        # The pinned sampler is now executing, the same as a monolithic START_INFERENCE. Without this the slot
        # keeps whatever control flag it last carried; a slot the ladder once told to unload would keep reading
        # as "unload in flight" across every disaggregated job it serves afterwards, since a disaggregated
        # sampler never reports a VRAM materialisation that would retire the flag, and both the reclaim
        # candidate set and the unload actuator would pass it over for good.
        process_with_model.last_control_flag = HordeControlFlag.START_INFERENCE
        process_with_model.loaded_horde_model_name = model
        process_with_model.loaded_horde_model_baseline = self._model_metadata.get_baseline(model)
        # Carry the retention verdict on the sampler lane: the sample stage's completion is synthesized by the
        # parent from the image lane's decode, so this record is the only thing that can tell that completion
        # whether the sampler's UNet stayed on the device.
        process_with_model.note_retention_grant(
            model if keep_model_resident_after else None,
            component_only=True,
        )
        self._process_lifecycle.action_ledger.record(
            LedgerEventType.INFERENCE_DISPATCHED,
            process_id=process_with_model.process_id,
            os_pid=process_with_model.os_pid,
            launch_identifier=process_with_model.process_launch_identifier,
            job_id=str(next_job.id_) if next_job.id_ is not None else None,
            detail={"model": model, "disaggregated": True},
        )
        logger.opt(colors=True).info(
            "<fg #f0beff>Job {} routed to the disaggregated pipeline; sampler pinned "
            f"to process {process_with_model.process_id}.</>",
            str(next_job.id_)[:8],
        )

    def note_disaggregated_sampler_bound(
        self,
        next_job: ImageGenerateJobPopResponse,
        process_with_model: HordeProcessInfo,
    ) -> None:
        """Commit the sampler slot for a staged-ahead job that has just bound one, retention verdict included.

        A job admitted ahead of a pin has no sampler at registration, so the dispatch bookkeeping the pinned
        path performs there runs here instead, at the instant the pin released and the sample is going out. The
        retention verdict is taken now for the same reason: it prices the card as it is when the weights would
        be held, not as it was a job earlier.

        The model-change eviction a monolithic dispatch runs is deliberately not repeated here: this job binds
        this lane precisely because the lane already holds its model, so there is no other model on it to evict.

        The encode charge :meth:`_stage_head_ahead_of_pin` booked against the component lane is returned here:
        the conditioning is produced by the time a sampler binds, and from this instant the job is priced
        exactly as a job dispatched disaggregated at admission is, which is to say it carries no dispatch
        reservation until clearance books its full materialisation peak against the sampler
        (:meth:`clearance_admit_process`). Leaving the encode charge outstanding would hold clearance on room
        the card really has, once per staged job, for as long as each job stayed in progress.
        """
        self.release_dispatch_reservation(next_job)
        model = next_job.model
        if model is None:
            return
        if process_with_model.retained_resident_model == model:
            self._retention_reuses += 1
            # The bet paid, so this episode's age is spent: the hold that follows this job is a new prediction.
            process_with_model.retained_resident_since = None
        keep_model_resident_after = self._should_keep_model_resident(
            next_job,
            process_with_model=process_with_model,
            device_index=process_with_model.device_index if self._multi_gpu_routing_active else None,
        )
        # Recorded after the verdict, never before: the gate asks what this slot ran *previously*.
        self._record_slot_dispatch(process_with_model.process_id, model)
        self._commit_disaggregated_sampler_slot(
            next_job,
            process_with_model,
            keep_model_resident_after=keep_model_resident_after,
        )

    async def _dispatch_inference_message(
        self,
        next_job: ImageGenerateJobPopResponse,
        process_with_model: HordeProcessInfo,
        *,
        keep_model_resident_after: bool,
        dispatched_device_index: int | None,
        degraded_dispatch: bool,
    ) -> None:
        """Send the START_INFERENCE command and record the outcome: mark started on success, fault on failure.

        On a successful send, marks the job started, stamps the slot's in-flight timing for the
        graded-slowdown monitor, records the dispatch in the action ledger, and advances the process state.
        On a failed send, faults the job so the horde reissues it.

        A disaggregation-eligible job is not sent monolithic inference here: this is the single admission
        point, so the scheduler has already preloaded its model onto ``process_with_model`` exactly as for a
        monolithic job, and instead of START_INFERENCE the job is registered with the orchestrator pinned to
        that process as its sampler. All the in-flight/job-progress marking a monolithic dispatch performs is
        applied identically (so the process cannot be double-booked), minus the sampling-timing stamps, which
        the sampler reports itself when the sample stage runs.
        """
        if next_job.model is None:
            raise ValueError(f"next_job.model is None ({next_job})")

        # A disaggregation-eligible job is registered with the orchestrator (its sampler pinned to
        # process_with_model) instead of being sent monolithic inference. If registration is declined (a role
        # went unhealthy between the eligibility check and here) the dispatch falls through to the monolithic
        # path so the job still runs whole rather than being dropped.
        if (
            self._register_disaggregated_job is not None
            and self._is_disaggregatable_job(next_job)
            and await self._dispatch_disaggregated(
                next_job,
                process_with_model,
                keep_model_resident_after=keep_model_resident_after,
                dispatched_device_index=dispatched_device_index,
                degraded_dispatch=degraded_dispatch,
            )
        ):
            return

        bridge_data = self._runtime_config.bridge_data
        # Carry the pre-annotated ControlNet control map, if the image-utilities lane derived one for this
        # job before dispatch; the inference child injects it so hordelib skips re-annotating in the main
        # venv. None (every non-annotated job) preserves the normal in-graph preprocessing path.
        tracked_for_dispatch = self._job_tracker.get_tracked_job(next_job.id_) if next_job.id_ is not None else None
        premade_control_map_bytes = (
            tracked_for_dispatch.premade_control_map_bytes if tracked_for_dispatch is not None else None
        )
        if process_with_model.safe_send_message(
            HordeInferenceControlMessage(
                control_flag=HordeControlFlag.START_INFERENCE,
                horde_model_name=next_job.model,
                sdk_api_job_info=next_job,
                keep_model_resident_after=keep_model_resident_after,
                premade_control_map_bytes=premade_control_map_bytes,
                skipped_aux_models=self._job_tracker.skipped_aux_for_job(next_job),
                # The child's own free-VRAM view overstates the card under WDDM, so it is handed the parent's
                # device-level reading (taken on the control tick that precedes this dispatch) to clamp the
                # shortfall arithmetic its freeing decisions are made on.
                device_free_mb=self._measured_device_free_mb(dispatched_device_index),
            ),
        ):
            await self._job_tracker.mark_inference_started(
                next_job,
                device_index=dispatched_device_index,
                whole_card=self._serving_under_whole_card(next_job.model, dispatched_device_index),
                process_age_seconds=time.time() - process_with_model.spawned_at,
            )
            horde_model_baseline = self._model_metadata.get_baseline(next_job.model)
            # Under the clearance lease a dispatch only stages the job: its weights load at clearance, so book
            # the encode-only charge now and upgrade to the full materialisation peak when the parent clears it.
            self._record_dispatch_reservation(
                next_job,
                process_with_model,
                baseline=horde_model_baseline,
                staging_only=self._clearance_lease_active(),
            )

            dispatch_detail: dict[str, str | int | float | bool | None] = {
                "model": next_job.model,
                "steps": next_job.payload.ddim_steps,
            }
            expected_seconds = self._expected_sampling_seconds(next_job, horde_model_baseline)
            if expected_seconds is not None:
                dispatch_detail["expected_sampling_seconds"] = round(expected_seconds, 2)

            # Stamp the in-flight timing onto the slot so the graded-slowdown monitor can measure this
            # job against its expected sampling time; the level resets so notices escalate per dispatch.
            process_with_model.current_inference_started_at = time.time()
            process_with_model.current_first_step_at = None
            process_with_model.current_job_expected_sampling_seconds = expected_seconds
            process_with_model.current_job_slowdown_level = 0
            process_with_model.consecutive_slow_per_steps = 0
            process_with_model.current_job_per_step_floor_tripped = False

            if degraded_dispatch:
                self._job_tracker.clear_degraded_dispatch(next_job)
                dispatch_detail["degraded_retry"] = True
                logger.warning(
                    f"  Degraded, isolated retry dispatched for job {str(next_job.id_)[:8]} "
                    "after a prior resource failure.",
                )

            self._process_lifecycle.action_ledger.record(
                LedgerEventType.INFERENCE_DISPATCHED,
                process_id=process_with_model.process_id,
                os_pid=process_with_model.os_pid,
                launch_identifier=process_with_model.process_launch_identifier,
                job_id=str(next_job.id_) if next_job.id_ is not None else None,
                detail=dispatch_detail,
            )

            process_with_model.last_control_flag = HordeControlFlag.START_INFERENCE
            tracked = self._job_tracker.get_tracked_job(next_job.id_) if next_job.id_ is not None else None
            process_with_model.record_inference_ownership(
                next_job,
                attempt_ordinal=(tracked.inference_attempts + 1) if tracked is not None else 1,
            )
            process_with_model.loaded_horde_model_name = next_job.model
            process_with_model.loaded_horde_model_baseline = horde_model_baseline
            # Carry the retention verdict on the slot: the completion path cannot see the dispatch decision,
            # and it is what decides whether this job leaves its weights on the card.
            process_with_model.note_retention_grant(next_job.model if keep_model_resident_after else None)
            # Optimistically mark the slot primed at dispatch (staging toward sampling, not yet in the
            # denoise loop); it advances to INFERENCE_STARTING on the first step. Keeps the slot readable
            # as busy-and-owning-the-job the instant START_INFERENCE is sent, before the child confirms.
            self._process_map.on_process_state_change(
                process_id=process_with_model.process_id,
                new_state=HordeProcessState.INFERENCE_PRIMED,
            )

        else:
            logger.error(
                f"Failed to start inference for job {next_job.id_} on process {process_with_model.process_id}",
            )
            await self._job_tracker.handle_job_fault(
                faulted_job=next_job,
                process_info=process_with_model,
                process_timeout=bridge_data.process_timeout,
            )

    def _prune_abandoned_dispatch_holds(self) -> None:
        """Drop hold bookkeeping for jobs no longer pending inference (rerouted, faulted, or dispatched).

        An abandoned hold (its job left the pending queue by some path other than a release through the gate)
        is not a release, so it advances neither release counter; it is simply forgotten so the maps stay
        bounded to the live queue.
        """
        pending_ids = {str(job.id_) for job in self._job_tracker.jobs_pending_inference if job.id_ is not None}
        for held_id in [held_id for held_id in self._dispatch_hold_since if held_id not in pending_ids]:
            self._dispatch_hold_since.pop(held_id, None)
            self._dispatch_hold_reclaim_requested.discard(held_id)
            self._resolve_dispatch_decision(held_id, reason="left_pending_queue")
        self._post_processing_defer_holds.intersection_update(pending_ids)

    def _note_dispatch_hold(self, job: ImageGenerateJobPopResponse, *, reclaim_applied: bool) -> None:
        """Record that the dispatch of ``job`` was held this pass, stamping the first hold and its cause.

        Every hold pass counts a conflict; the first hold for a job also stamps the hold-start instant and
        counts a distinct held dispatch. A pass whose actuator accepted an eviction command marks the job so
        its eventual release is attributed to reclaim rather than to the card recovering on its own.
        """
        job_id = str(job.id_) if job.id_ is not None else None
        if job_id is None:
            return
        self._dispatch_reconciliation_conflicts += 1
        if job_id not in self._dispatch_hold_since:
            self._dispatch_hold_since[job_id] = self._clock()
            self._dispatch_reconciliation_holds += 1
        if reclaim_applied:
            self._dispatch_hold_reclaim_requested.add(job_id)

    def _resolve_dispatch_hold(self, job: ImageGenerateJobPopResponse) -> None:
        """Close out any dispatch hold on ``job`` now that it fits, folding its duration and release cause.

        A no-op for a job that was never held (the common admit-first-pass case). A held job's accumulated
        wait folds into the cumulative hold seconds, and the release is attributed to reclaim when this gate
        emitted eviction commands during the hold, otherwise to the card freeing on its own.
        """
        job_id = str(job.id_) if job.id_ is not None else None
        if job_id is None:
            return
        held_since = self._dispatch_hold_since.pop(job_id, None)
        if held_since is None:
            self._dispatch_hold_reclaim_requested.discard(job_id)
            return
        self._dispatch_reconciliation_hold_seconds += max(0.0, self._clock() - held_since)
        if job_id in self._dispatch_hold_reclaim_requested:
            self._dispatch_reconciliation_released_by_reclaim += 1
            self._dispatch_hold_reclaim_requested.discard(job_id)
        else:
            self._dispatch_reconciliation_released_by_natural_free += 1
        self._resolve_dispatch_decision(job_id, reason="dispatch_admitted")

    def _note_post_processing_defer(self, job: ImageGenerateJobPopResponse, *, deferred: bool) -> None:
        """Record whether the post-processing co-residency gate held ``job``'s dispatch on this pass.

        The dispatch path already computed this verdict; caching it lets the read-only stall classifier name
        the same hold without re-deriving the PP arithmetic. A held job is remembered; a job that cleared the
        gate is forgotten, so the cache tracks only heads the gate is actively deferring right now.
        """
        job_id = str(job.id_) if job.id_ is not None else None
        if job_id is None:
            return
        if deferred:
            self._post_processing_defer_holds.add(job_id)
        else:
            self._post_processing_defer_holds.discard(job_id)

    def _resolve_dispatch_decision(self, job_id: str, *, reason: str) -> None:
        """Close any open dispatch-hold decision for ``job_id`` with a final resolving record.

        A resolving verdict for a subject the recorder is not tracking is dropped, so calling this for a
        job that was never held is harmless.
        """
        if self._decision_sink is None:
            return
        self._decision_sink(
            decision_kind=DecisionKind.INFERENCE_DISPATCH,
            subject=job_id,
            verdict=DecisionVerdict.NO_OP,
            reason=reason,
        )

    def _dispatch_hold_is_standing(self, job: ImageGenerateJobPopResponse) -> bool:
        """Whether ``job``'s dispatch hold has stood long enough to be a wedge rather than a tight moment.

        Read from the hold ledger the gate already keeps, so the bound is the hold's own age rather than a
        second clock. A card that is momentarily full while work flows resolves its own holds within a pass
        or two, and stripping an idle lane's tenancy there would trade a warm cache for room the finishing
        job was about to return anyway. Past the stall horizon nothing is finishing, and the tenancy is the
        only thing left to ask.
        """
        job_id = str(job.id_) if job.id_ is not None else None
        if job_id is None:
            return False
        held_since = self._dispatch_hold_since.get(job_id)
        return held_since is not None and (self._clock() - held_since) >= _DISPATCH_STALL_MIN_SECONDS

    def _reclaim_idle_tenancy_for_head(
        self,
        head_job: ImageGenerateJobPopResponse,
        target_process: HordeProcessInfo,
        *,
        device_index: int | None,
    ) -> bool:
        """Ask idle lanes on this card for tenancy the arbiter cannot name, for a head that does not fit.

        Two holders sit outside the arbiter's eviction description because neither is a resident checkpoint
        it tracks: a lane holding component-cache entries between jobs, and a slot parked on a preload whose
        dispatch never came. Both are the card's, both outlive every job boundary, and neither is returned by
        anything except an unload the parent actuates, so a hold that waits for them to clear waits forever.

        Issued through :meth:`unload_idle_model`, the single reclaim actuator, so this adds a caller rather
        than a second ladder, and one lane at a time: the actuator refuses a slot already unloading, so a
        card needing more than one lane's tenancy gives it up over successive holds rather than being
        stripped in one pass. The head's own target slot is spared, as is any lane holding the head's model
        (reclaiming the copy the head is about to use is the trade this exists to avoid).

        Returns True when at least one unload was issued, which the hold reports as room being on the way.
        """
        active_jobs_on_card = (
            self._jobs_in_progress_on_card(device_index)
            if self._multi_gpu_routing_active and device_index is not None
            else self._job_tracker.jobs_in_progress
        )
        if active_jobs_on_card:
            # A job is sampling on this card, so its completion is the fit the hold is waiting for. Taking an
            # idle lane's cache here would buy room the finishing job is about to return anyway, and pay for
            # it with that lane's next cold load. The tenancy is only worth asking for when nothing on the
            # card is producing a fit at all.
            return False
        head_model = head_job.model
        acted = False
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.process_id == target_process.process_id:
                continue
            if process_info.loaded_horde_model_name == head_model and head_model is not None:
                continue
            holds_component_tenancy = bool(process_info.held_components)
            if not holds_component_tenancy and not self._slot_is_reclaimable_while_busy(process_info):
                continue
            if self.unload_idle_model(process_info.process_id, device_index=device_index):
                acted = True
                break
        return acted

    def _dispatch_residency_reconciliation_holds(
        self,
        next_job: ImageGenerateJobPopResponse,
        process_with_model: HordeProcessInfo,
        *,
        is_head_of_queue: bool = True,
        head_outstanding_mb: float | None = None,
    ) -> bool:
        """Return whether ``next_job``'s dispatch must be held because its VRAM would over-commit the card now.

        The dispatch of an already-RAM-staged job is the moment its weights and first activation actually
        materialise on the device. Admission is consulted at preload and at the second-concurrent-sampler seam,
        but not here, so a job whose materialisation lands beside an idle sibling's still-resident weights can
        tip the card over the paging cliff faster than the tick-paced reclaim reacts. This gate closes that
        seam by pricing the dispatch through the arbiter's single MONOLITHIC_DISPATCH identity (the same
        measured-truth admission math the preload and overlap seams use, testing the candidate against the
        truthful device-free reading net of the outstanding reservations and the noise buffer): a FITS releases
        the dispatch, a DEFER or DENY holds it.

        On a hold the job is never faulted: it keeps its queue position and re-asks on the next scheduling pass.
        The conflicting idle residents are evicted through the one reclaim owner (the same
        :meth:`_execute_preload_actuations` surface the arbiter's preload-DEFER path drives), never inline and
        never through a second ladder; the head's own target slot is protected. The hold releases only once the
        arbiter next verdicts FITS, matching the verified-reclaim doctrine that a demand is admitted into
        measured reality rather than into hope. Can't-fit-ever models are excluded upstream by model
        serviceability, so this gate only ever holds a can't-fit-now dispatch.

        ``is_head_of_queue`` is the truth of whether this dispatch is the genuine head of queue: a line-skip
        dispatch (a smaller ready job selected ahead of a downloading head) is not, so it presents
        ``is_head_of_queue=False`` and ``head_outstanding_mb`` priced from the head it jumped. Head protection
        then holds the line-skipper when admitting it would leave the card without the room the head needs, so
        the head keeps first claim on the physical space rather than being starved behind the skipper. A
        line-skip likewise routes its reclaim through the non-head eviction path, which respects the residency
        and queued-lookahead guards the head escalation would override.
        """
        self._prune_abandoned_dispatch_holds()

        if next_job.model is None:
            return False

        # Under the clearance lease a dispatch only stages the job (weights load at clearance), so price the
        # encode charge here rather than the full materialisation: the full fit-or-evict runs at clearance.
        # Head protection is preserved either way (the head's outstanding charge still reserves room), and the
        # full-price gate that used to park staging on materialisation grounds moves to the clearance VRAM
        # moment. Without the lease this is the VRAM moment, so the full materialisation is priced as before.
        staging_charge_override = _STAGING_ENCODE_VRAM_MB if self._clearance_lease_active() else None

        outcome = self._evaluate_materialization_admission(
            next_job,
            process_with_model,
            is_head_of_queue=is_head_of_queue,
            head_outstanding_mb=head_outstanding_mb,
            candidate_delta_override_mb=staging_charge_override,
        )
        verdict = outcome.verdict
        candidate_delta_mb = outcome.candidate_delta_mb
        device_index = outcome.device_index

        if verdict.admits:
            self._resolve_dispatch_hold(next_job)
            return False

        # The arbiter describes evictions in terms of resident checkpoints, so a card held by idle tenancy it
        # cannot name (a lane's component cache, a slot parked on a preload nothing dispatched) yields no
        # command and the hold waits on a fit nothing is producing. Ask those holders for the card back
        # through the one reclaim actuator before recording a hold that reclaims nothing.
        tenancy_reclaimed = False
        if is_head_of_queue and not outcome.actuations_applied and self._dispatch_hold_is_standing(next_job):
            tenancy_reclaimed = self._reclaim_idle_tenancy_for_head(
                next_job,
                process_with_model,
                device_index=device_index,
            )

        self._note_dispatch_hold(
            next_job,
            reclaim_applied=bool(outcome.actuations_applied) or tenancy_reclaimed,
        )

        if self._decision_sink is not None and next_job.id_ is not None:
            measured = verdict.measured
            self._decision_sink(
                decision_kind=DecisionKind.INFERENCE_DISPATCH,
                subject=str(next_job.id_),
                verdict=DecisionVerdict.DEFER,
                reason=verdict.reason or verdict.disposition.value,
                inputs={
                    "model": str(next_job.model),
                    "device_index": device_index,
                    "candidate_delta_mb": None if candidate_delta_mb is None else round(candidate_delta_mb, 1),
                    "device_free_mb": None if measured.device_free_mb is None else round(measured.device_free_mb, 1),
                    "available_mb": None if measured.available_mb is None else round(measured.available_mb, 1),
                    "outstanding_reservations_mb": round(measured.outstanding_reservations_mb, 1),
                    "noise_buffer_mb": round(measured.noise_buffer_mb, 1),
                    "is_head_of_queue": is_head_of_queue,
                    "reclaim_requested": bool(outcome.actuations_requested),
                    "reclaim_applied": bool(outcome.actuations_applied),
                    "reclaim_requested_kinds": ",".join(
                        command.kind.value for command in outcome.actuations_requested
                    ),
                    "reclaim_applied_kinds": ",".join(command.kind.value for command in outcome.actuations_applied),
                },
            )

        suppressed = self._scheduler_diagnostic_suppressed_count(
            "dispatch_residency_hold",
            (str(next_job.id_), verdict.disposition.value),
        )
        if suppressed is not None:
            # Only claim the eviction when an actuator accepted it. A proposed command can be declined because
            # its target became busy, protected, or was already reclaimed; that leaves the measured state
            # unchanged and must not be described as room being on the way.
            reclaim_note = (
                " Evicting idle VRAM so the job's materialisation fits the card before it commits to VRAM."
                if outcome.actuations_applied or tenancy_reclaimed
                else (
                    " Reclaim was proposed but no actuator accepted it; the hold remains eligible for escalation."
                    if outcome.actuations_requested
                    else " Nothing is being reclaimed for this hold; it releases when the arbiter next verdicts a fit."
                )
            )
            logger.opt(colors=True).warning(
                f"<fg #f0beff>Holding dispatch of {{}} to reconcile residency: {verdict.stated}.{reclaim_note}</>",
                next_job.model,
            )
        return True

    def _evaluate_materialization_admission(
        self,
        next_job: ImageGenerateJobPopResponse,
        process_with_model: HordeProcessInfo,
        *,
        is_head_of_queue: bool,
        head_outstanding_mb: float | None,
        candidate_delta_override_mb: float | None = None,
        nets_own_dispatch_reservation: bool = False,
    ) -> _MaterializationOutcome:
        """Price a job's VRAM materialisation through the single MONOLITHIC_DISPATCH arbiter identity.

        The reusable core shared by the dispatch residency-reconciliation gate and the clearance gate: it
        builds the arbiter request against measured device truth (net of outstanding reservations and the
        noise buffer), evaluates it, and on a non-admit routes the described idle-resident eviction through the
        one reclaim owner (:meth:`_execute_preload_actuations`), protecting the head's own target slot. It owns
        no hold bookkeeping or logging: each caller records its own hold state (dispatch-pending versus
        clearance) and emits its own diagnostics from the returned outcome.

        ``candidate_delta_override_mb`` prices a smaller charge than the job's full materialisation peak (the
        encode-only staging charge under the clearance lease, where the weights have not yet loaded); ``None``
        prices the full activation-inclusive peak (the non-lease dispatch VRAM moment and the clearance moment).

        ``nets_own_dispatch_reservation`` is set by the clearance re-price of an already-dispatched job: its
        encode-only staging reservation is still outstanding in the dispatch flow, and the full peak priced
        here already covers it, so that entry is netted out of the overlay rather than counted against the
        job's own room. Left False at dispatch, where the job holds no reservation yet.

        Both callers guarantee a non-None model before pricing (an unmodelled job is not materialisable).
        """
        if next_job.model is None:
            raise ValueError("materialisation admission requires a job with a model")
        device_index = process_with_model.device_index if self._multi_gpu_routing_active else None
        own_dispatch_mb = 0.0
        if nets_own_dispatch_reservation and next_job.id_ is not None:
            own_dispatch_mb = self._reserve_ledger.planned_charge_for_unit(
                DISPATCH_ADMISSION_FLOW,
                str(next_job.id_),
                self._committed_process_reserved_by_pid(device_index),
            )
        baseline = self._model_metadata.get_baseline(next_job.model)
        has_reclaimable_idle_model = self._has_reclaimable_idle_model(
            process_with_model,
            for_head_of_queue=is_head_of_queue,
            device_index=device_index,
            make_room_for_model=next_job.model,
        )
        candidate_delta_mb = (
            candidate_delta_override_mb
            if candidate_delta_override_mb is not None
            else self._measured_admission_candidate_delta_mb(
                next_job,
                baseline,
                process_id=process_with_model.process_id,
                disaggregated=self._is_disaggregation_class_eligible(next_job),
            )
        )
        forecast = self._forecast_streaming(next_job, baseline, device_index=device_index)
        total_vram_mb = self._process_map.get_reported_total_vram_mb(device_index=device_index)
        structural_reserve_mb = (
            effective_inference_reserve_mb(total_vram_mb, 0.0)
            if total_vram_mb is not None
            else forecast._effective_base_reserve  # noqa: SLF001 - same budget owner sizes teardown depth.
        )
        max_resident = (
            self._max_coresident_for_peak_mb(
                candidate_delta_mb,
                structural_reserve_mb,
                device_index=device_index,
            )
            if candidate_delta_mb is not None
            else None
        )
        live_inference_processes = self._process_map.num_loaded_inference_processes(device_index=device_index)
        idle_contexts_teardownable = (
            is_head_of_queue
            and max_resident is not None
            and max_resident < live_inference_processes
            and self._has_teardownable_idle_context(process_with_model, device_index=device_index)
        )
        active_jobs_on_card = (
            self._jobs_in_progress_on_card(device_index)
            if self._multi_gpu_routing_active and device_index is not None
            else self._job_tracker.jobs_in_progress
        )
        prepared_head_reprices_activation = self._job_tracker.are_job_aux_models_prepared(next_job) and bool(
            active_jobs_on_card,
        )
        request = VramRequest(
            kind=VramRequestKind.MONOLITHIC_DISPATCH,
            job_label=str(next_job.model),
            baseline=baseline,
            device_index=device_index,
            target_process_id=process_with_model.process_id,
            candidate_delta_mb=candidate_delta_mb,
            candidate_weights_mb=predict_job_weight_mb(next_job, baseline),
            accepted_work=(next_job.id_ is not None and self._job_tracker.get_tracked_job(next_job.id_) is not None),
            # An ordinary dispatch onto resident weights is a no-materialisation fast path. A prepared head
            # re-entering while another job owns a live reservation is not: its weights may be resident, but
            # its activations would overlap that admitted peak. Reprice the activation-only delta so the
            # preparation boundary cannot turn resident-weight credit into an ungoverned second sampler.
            candidate_already_resident=(
                self._candidate_weights_resident_on_process(
                    next_job.model,
                    process_with_model.process_id,
                )
                and not prepared_head_reprices_activation
            ),
            own_planned_unmaterialized_mb=self._own_planned_charge_mb(
                device_index=device_index,
                target_process_id=process_with_model.process_id,
            ),
            own_dispatch_unmaterialized_mb=own_dispatch_mb,
            is_head_of_queue=is_head_of_queue,
            head_job_id=str(next_job.id_) if next_job.id_ is not None else None,
            measured_attempt_in_progress=self._job_tracker.is_measured_attempt_on_device(
                next_job,
                device_index,
            ),
            measured_attempt_already_spent=self._job_tracker.has_spent_measured_attempt_on_device(
                next_job,
                device_index,
            ),
            head_outstanding_mb=head_outstanding_mb,
            starved_seconds=self._head_starved_seconds(next_job),
            has_reclaimable_idle_model=has_reclaimable_idle_model,
            # An ordinary staged dispatch never reduces the live inference-context count: it evicts idle
            # residents to make room, it does not collapse the co-resident pool (can_reduce_live_contexts stays
            # False, so the ordinary activation-peak warrant never tears a context down). The one exception is a
            # starved head whose deficit is held by its own bare idle sibling contexts with no reality-admit and
            # no weight reclaim left: it escalates to the same verified teardown the preload seam uses, so this
            # head-only signal is reported for that path.
            can_reduce_live_contexts=False,
            idle_contexts_teardownable=idle_contexts_teardownable,
        )
        verdict = self._ensure_preload_arbiter().evaluate(request)

        if verdict.admits:
            if verdict.measured_attempt:
                self._mark_measured_attempt(next_job, request, device_index=device_index)
            return _MaterializationOutcome(
                verdict=verdict,
                candidate_delta_mb=candidate_delta_mb,
                device_index=device_index,
                actuations_requested=(),
                actuations_applied=(),
            )

        # The materialisation cannot land yet. Route the described idle-resident eviction through the single
        # reclaim owner, protecting the head's own slot. The caller holds (dispatch or clearance) and re-asks
        # next pass, releasing once the arbiter verdicts FITS (the governor having verified the reclaimed room).
        actuations = verdict.required_actuations
        self._preload_actuation = _PreloadActuation(
            job=next_job,
            available_process=process_with_model,
            forecast=forecast,
            max_resident=max_resident,
        )
        try:
            applied_actuations = self._execute_preload_actuations(
                actuations,
                device_index=device_index,
                for_head_of_queue=is_head_of_queue,
            )
        finally:
            self._preload_actuation = None
        return _MaterializationOutcome(
            verdict=verdict,
            candidate_delta_mb=candidate_delta_mb,
            device_index=device_index,
            actuations_requested=actuations,
            actuations_applied=applied_actuations,
        )

    def head_of_queue_is_parked(self) -> bool:
        """Whether the queue has stopped moving behind a head that is not dispatching.

        The same clock the dispatch-stall diagnostic reads, exposed so resource owners can tell a card that is
        merely idle from one whose queue is not moving. Two facts are required, because either alone is
        ordinary: a head undispatched past the stall threshold is normal backpressure while a sibling samples,
        and an idle pool is normal when there is nothing to serve. Together they are a queue no one is serving.
        """
        if self._process_map.has_inference_in_progress():
            return False
        head = self._undispatched_head()
        if head is None:
            return False
        return self._head_starved_seconds(head) >= _DISPATCH_STALL_MIN_SECONDS

    def latest_dispatch_reconciliation_holds(self) -> int:
        """Return the count of dispatches held for residency reconciliation this run (calibration visibility)."""
        return self._dispatch_reconciliation_holds

    def latest_dispatch_reconciliation_conflicts(self) -> int:
        """Return the count of dispatch-time residency conflicts detected this run (calibration visibility)."""
        return self._dispatch_reconciliation_conflicts

    def latest_dispatch_reconciliation_hold_seconds(self) -> float:
        """Return the cumulative seconds dispatches spent held for residency reconciliation (calibration)."""
        return self._dispatch_reconciliation_hold_seconds

    def latest_dispatch_reconciliation_released_by_reclaim(self) -> int:
        """Return the count of held dispatches released after this gate's eviction freed room (calibration)."""
        return self._dispatch_reconciliation_released_by_reclaim

    def latest_dispatch_reconciliation_released_by_natural_free(self) -> int:
        """Return the count of held dispatches released by the card recovering on its own (calibration)."""
        return self._dispatch_reconciliation_released_by_natural_free

    def latest_affinity_skips(self) -> int:
        """Return the committed affinity line-skips the currently-tracked displaced head has taken (visibility)."""
        return self._affinity_skip_state.skip_count

    def latest_affinity_skip_seconds(self) -> float:
        """Return the wall-clock seconds since the tracked displaced head's first affinity skip (0 if none)."""
        if self._affinity_skip_state.skip_count == 0:
            return 0.0
        return self._clock() - self._affinity_skip_state.first_skip_time

    def latest_safety_placement_demotions(self) -> int:
        """Return how many times the runtime safety-placement policy moved safety off-GPU this run."""
        return self._safety_placement_demotions

    def latest_safety_placement_promotions(self) -> int:
        """Return how many times the runtime safety-placement policy restored safety to the GPU this run."""
        return self._safety_placement_promotions

    def latest_safety_placement_card(self) -> int | None:
        """Return the card the safety process currently occupies, or None when safety is off-GPU (on CPU)."""
        return self._process_lifecycle.safety_gpu_card_index()

    async def start_inference(self) -> bool:
        """Start inference for the next job in jobs_pending_inference, if possible.

        During graceful shutdown the worker keeps draining the queue it already popped rather than
        faulting it: the job popper stops accepting NEW jobs once shutdown is armed, so the only jobs
        that can dispatch here are ones accepted before the stop. They are given a chance to finish,
        bounded by the per-job-scaled shutdown grace and the force-kill backstop; whatever genuinely
        cannot finish in time is still fault-reported so the horde reissues it promptly.
        """
        next_job_and_process = await self.get_next_job_and_process()

        if next_job_and_process is None:
            return False

        bridge_data = self._runtime_config.bridge_data
        process_with_model = next_job_and_process.process_with_model
        next_job = next_job_and_process.next_job

        if self._head_priority_barrier_withholds_dispatch(next_job):
            # A starved head has latched the head-priority barrier: withhold every dispatch but the head's own
            # so the running jobs drain and the head reaches its best-effort admit. The head keeps its queue
            # position and dispatches the moment the barrier releases.
            if not self._head_priority_barrier_withhold_logged:
                logger.opt(colors=True).info(
                    "<fg #7b7d7d><i>Holding dispatch of job {} behind the head-priority "
                    "barrier so running jobs drain for the starved head.</i></>",
                    str(next_job.id_)[:8],
                )
                self._head_priority_barrier_withhold_logged = True
            return False

        if next_job_and_process.line_skip is None and self._job_requires_aux_preparation(next_job):
            # An unprepared aux job is not dispatchable: it holds no lane and no VRAM reservation while the
            # pop-time prefetch pipeline places its LoRAs/TIs on disk and clears its preparation gate. It is
            # already skipped by dispatch selection and preload admission, so a fitting sibling flows past it;
            # this remains as the terminal gate so such a job can never seize a lane by any path.
            return False

        degraded_dispatch = self._job_tracker.is_degraded_dispatch_pending(next_job)
        if degraded_dispatch and len(self._job_tracker.jobs_in_progress) > 0:
            # A degraded retry (after a resource/OOM failure) runs in isolation to minimise VRAM
            # pressure: defer it until no other job is sampling. It keeps its head-of-queue position, so
            # it dispatches as soon as the in-flight jobs drain rather than being starved.
            return False

        if self._prestaged_whole_card_not_ready(next_job):
            # The head was pre-staged into RAM while another job held the device; sampling commits its
            # weights to VRAM, so it must wait for the residency to finish collapsing to sole residency
            # (idle siblings stopped, safety off-GPU, the card drained) before it starts. Otherwise a
            # lingering sibling context would force its first step to stream over the bus. It keeps its
            # head-of-queue position, so it dispatches the moment the residency converges.
            return False

        post_processing_deferred = self._should_defer_dispatch_for_post_processing(
            next_job,
            process_with_model=process_with_model,
        )
        self._note_post_processing_defer(next_job, deferred=post_processing_deferred)
        if post_processing_deferred:
            # Post-processing either holds the card or is waiting for the active sampler to drain, and this
            # job's sampling peak cannot share the card with it.
            return False

        line_skip = next_job_and_process.line_skip
        if self._budget_active() and self._dispatch_residency_reconciliation_holds(
            next_job,
            process_with_model,
            # A line-skip dispatch is not the true head of queue, so it presents is_head_of_queue=False and the
            # head it jumped priced as head_outstanding_mb. Head protection reserves the card's physical room
            # for a head that may begin sampling on its own once the lane it waits on frees.
            is_head_of_queue=line_skip is None,
            head_outstanding_mb=(
                None
                if line_skip is None
                else self._displaced_head_outstanding_mb(
                    line_skip.displaced_job,
                    device_index=process_with_model.device_index if self._multi_gpu_routing_active else None,
                )
            ),
        ):
            # The staged job's VRAM materialisation would over-commit the card against an idle sibling's
            # still-resident weights: hold the dispatch (the job keeps its head-of-queue position) while the
            # single reclaim owner evicts the idle residents, and re-ask next pass once the arbiter verifies the
            # reclaimed room fits it. This is the seam where RAM-staged weights actually commit to VRAM, which
            # neither the preload nor the second-sampler admission consult.
            return False

        if self._retained_resident_dispatch_holds(next_job, process_with_model):
            # Weights another slot holds across jobs occupy the card this dispatch would load into. The job
            # keeps its head-of-queue position while they are asked back, and dispatches once the card has
            # evidenced the free rather than the moment the request went out.
            return False

        # Every hold gate above is passed, so this dispatch is committed: advance the affinity skip window here
        # (never in get_next_job_and_process, which runs twice a cycle and must stay pure for information_only).
        # A resident_bypass skip is counted against the displaced head; a direct head dispatch (no line-skip)
        # closes the window. A diversity line-skip leaves the window untouched: the head is still pending, its
        # process merely busy, so it is not being aged by the affinity path.
        # A placement reorder is counted the same way. The promoted job arrives here as the head (no line-skip
        # was needed to reach it), so without this the direct-dispatch branch below would reset the window every
        # time and the ceiling could never accumulate: a head could then be passed without bound. Seating a job
        # ahead of the head *is* a committed pass of that head, which is exactly what the window counts, and the
        # ceiling is what bounds the reorder on a worker whose jobs carry no ttl for the age override to read.
        reordered_ahead_of = self._reordered_head_displaced_by(next_job) if line_skip is None else None
        if line_skip is not None and line_skip.reason == "resident_bypass":
            displaced_head_id = str(line_skip.displaced_job.id_) if line_skip.displaced_job.id_ is not None else None
            if displaced_head_id is not None:
                self._affinity_skip_state = record_affinity_skip(
                    self._affinity_skip_state,
                    displaced_head_id,
                    self._clock(),
                )
        elif reordered_ahead_of is not None:
            self._note_retention_reorder(next_job, reordered_ahead_of, process_with_model)
            self._retention_affinity_reorders += 1
            displaced_head_id = str(reordered_ahead_of.id_) if reordered_ahead_of.id_ is not None else None
            if displaced_head_id is not None:
                self._affinity_skip_state = record_affinity_skip(
                    self._affinity_skip_state,
                    displaced_head_id,
                    self._clock(),
                )
        elif line_skip is None:
            self._affinity_skip_state = AffinitySkipState()

        if line_skip is not None:
            if line_skip.reason == "resident_bypass":
                skip_detail = affinity_skip_disclosure(
                    self._affinity_skip_state,
                    now=self._clock(),
                    budget_seconds=affinity_budget_seconds(self._state.recent_job_ttl),
                    max_skips=_AFFINITY_MAX_SKIPS,
                )
            else:
                skip_detail = "the displaced job's process is busy sampling its own model"
            logger.info(
                f"Job {next_job.id_} skipped the line ({line_skip.reason}) and will run on process "
                f"{process_with_model.process_id} ahead of job {line_skip.displaced_job.id_}: {skip_detail}.",
            )

        if bridge_data.unload_models_from_vram_often:
            self.unload_models_from_vram(process_with_model)

        if degraded_dispatch:
            # Reclaim VRAM from idle slots before the degraded retry so the job has the best chance to
            # fit, independent of the unload_models_from_vram_often setting.
            self.unload_models_from_vram(process_with_model)

        if next_job.model is None:
            raise ValueError(f"next_job.model is None ({next_job})")

        process_with_model.batch_amount = next_job.payload.n_iter
        # Record the card this job runs on (None on a single-GPU host) so its over-budget fault streak is
        # kept per card: a model unservable on a small card can still be advertised and run on a larger one.
        dispatched_device_index = process_with_model.device_index if self._multi_gpu_routing_active else None
        # A dispatch landing on the model this slot is already holding is retention's whole return: the weights
        # it would otherwise have uploaded are on the card. Counted before the eviction below, which by
        # construction leaves a same-model retention alone.
        if process_with_model.retained_resident_model == next_job.model:
            self._retention_reuses += 1
            # The bet paid, so this episode's age is spent: the hold that follows this job is a new prediction
            # and is given the full horizon to be met rather than inheriting how long the previous one waited.
            process_with_model.retained_resident_since = None
        # Evict before the load, not after: a slot holding another model under an earlier grant would
        # otherwise carry both models' weights through this job. Ordered ahead of the retention verdict so
        # the verdict prices the card the dispatch will actually run on.
        self.evict_retained_resident_for_model_change(process_with_model, next_job.model)
        keep_model_resident_after = self._should_keep_model_resident(
            next_job,
            process_with_model=process_with_model,
            device_index=dispatched_device_index,
        )
        # Recorded after the verdict, never before: the gate asks what this slot ran *previously*, so a job
        # already in the history would satisfy the repeat test with itself and every first dispatch of a model
        # would be granted. This is the committed-dispatch point, which is the event the history is about.
        self._record_slot_dispatch(process_with_model.process_id, next_job.model)

        # Past every hold/fault gate: this job is dispatching now, so emit the start logging here rather than
        # before the reclaim decision (where a deferred or faulted job would mislead the log as "starting").
        color_format_string = "<fg #f0beff>{message}</>"
        logger.opt(colors=True).info(
            color_format_string,
            message=f"Starting inference for job {str(next_job.id_)[:8]} on process {process_with_model.process_id}",
        )
        self._log_job_dispatch_details(next_job)

        await self._dispatch_inference_message(
            next_job,
            process_with_model,
            keep_model_resident_after=keep_model_resident_after,
            dispatched_device_index=dispatched_device_index,
            degraded_dispatch=degraded_dispatch,
        )

        self._pending_line_skip = None

        # A job dispatched: any prior stall reason is now stale. Clear it so the
        # orchestrator intent's "Holding dispatch" does not stick after the stall resolves.
        self._dispatch_stall_last_reason = None

        return True

    def _job_requires_aux_preparation(self, job: ImageGenerateJobPopResponse) -> bool:
        """Return whether a pending job must resolve its auxiliary files before it may claim sampling admission.

        A base model can already be resident while the job's LoRAs or textual inversions are not.  The
        pop-time prefetch pipeline places those files on disk while the job stays pending, so preparation
        creates no dispatch reservation.  Once the prepared flag is set, START_INFERENCE still revalidates
        the files child-side and passes through every ordinary VRAM, concurrency, post-processing, and
        degraded-retry gate.
        """
        has_aux = bool(job.payload.loras) or bool(job.payload.tis)
        return has_aux and not self._job_tracker.are_job_aux_models_prepared(job)

    def _compute_wanted_models(self) -> set[str]:
        """The set of models the worker is actively serving right now.

        Derived from live scheduler state; every model currently resident on an inference
        process, plus every model referenced by a pending or in-progress job. This mirrors the
        affinity computation in :meth:`preload_models`; ``bridge_data.image_models_to_load`` is
        deliberately not used because the harness/canned path never resolves that config field,
        so live state is the only reliable source.

        Fixed-pool seats are deliberately NOT unioned in. This set sizes ``affinity_active`` and biases
        preload decisions, so a seat for a not-yet-loaded model would flip the whole card between the
        fits and overflow VRAM regimes and starve co-resident lanes of headroom. Seats are an advertising
        and RAM-idle-protection concept only: their RAM hold lives in the seat term of
        :meth:`_residency_protects_from_unload`, and under true RAM pressure an idle seat simply yields
        (:meth:`_evict_unprotected_components_under_pressure`).
        """
        wanted: set[str] = {
            p.loaded_horde_model_name
            for p in self._process_map.values()
            if p.process_type == HordeProcessType.INFERENCE and p.loaded_horde_model_name is not None
        }
        wanted.update(j.model for j in self._job_tracker.jobs_pending_inference if j.model is not None)
        wanted.update(j.model for j in self._job_tracker.jobs_in_progress if j.model is not None)
        return wanted

    def _seat_only_idle_models(self) -> frozenset[str]:
        """The seated pool models with no pending or in-progress job right now.

        A seat is "busy" while any pending or in-progress job references its model; those seats keep full
        residency protection. The remainder are seat-only idle models: the worker still advertises them, but
        no live work needs their weights this instant, so under true RAM pressure they yield their staged
        components to avoid a host-memory deadlock. Busy-versus-idle is derived from the same JobTracker demand
        that :meth:`_compute_wanted_models` reads, so no separate busy-seat provider is needed.
        """
        seats = self._pool_protected_models_provider()
        if not seats:
            return frozenset()
        busy_models = {job.model for job in self._job_tracker.jobs_pending_inference if job.model is not None}
        busy_models.update(job.model for job in self._job_tracker.jobs_in_progress if job.model is not None)
        return seats - busy_models

    def _is_model_forecast_to_load(self, model_name: str | None) -> bool:
        """Whether ``model_name`` is already on track to become resident soon.

        True when the model map marks it loading, or an inference process is currently preloading it or
        already holds it preloaded. In that case the job needing it will get a process shortly, so a
        later already-resident job may bypass it to keep the GPU fed rather than the worker idling until
        the load completes. When the model is *not* forecast to load, no bypass is allowed so the budget
        gate's room-making runs and that job makes progress instead of being starved behind bypassers.
        """
        if model_name is None:
            return False
        if self._horde_model_map.is_model_loading(model_name):
            return True
        return any(
            process.process_type == HordeProcessType.INFERENCE
            and process.loaded_horde_model_name == model_name
            and process.last_process_state in (HordeProcessState.PRELOADING_MODEL, HordeProcessState.PRELOADED_MODEL)
            for process in self._process_map.values()
        )

    def _refresh_model_demand(self) -> None:
        """Stamp the current time against every model with live demand (pending/in-progress job).

        Feeds the residency grace period (:meth:`_is_recently_demanded`). Only genuine demand,
        not mere residency, refreshes the stamp, so a loaded-but-idle model's grace still
        expires. Entries well past the grace window are pruned to bound the dict.
        """
        now = self._clock()
        for job in (*self._job_tracker.jobs_pending_inference, *self._job_tracker.jobs_in_progress):
            if job.model is not None:
                self._model_last_in_demand[job.model] = now

        cutoff = now - (_RESIDENCY_GRACE_SECONDS * 4)
        for model_name in [m for m, last in self._model_last_in_demand.items() if last < cutoff]:
            del self._model_last_in_demand[model_name]

    def _is_recently_demanded(self, model_name: str) -> bool:
        """Whether the model had a pending/in-progress job within the residency grace window."""
        last = self._model_last_in_demand.get(model_name)
        return last is not None and (self._clock() - last) <= _RESIDENCY_GRACE_SECONDS

    def _residency_protects_from_unload(
        self,
        model_name: str | None,
        wanted_models: set[str],
        *,
        vram: bool,
        under_pressure: bool = False,
    ) -> bool:
        """Whether residency policy should keep ``model_name`` loaded rather than evict it now.

        Two regimes:

        - **Working set fits the process count** (``affinity_active``): every actively-served
          model can have its own home process, so keep them all resident in both RAM and VRAM.
          This is the regime the soak measures and the dominant duty-cycle win; it stops a
          process evicting the very model it just used (and is about to reuse) the instant its
          next job has not yet been popped.
        - **More models than processes**: residency cannot be guaranteed, so apply only a RAM
          grace period; cheap to hold, and it avoids the expensive disk reload between a model's
          consecutive jobs. VRAM, the scarce resource, is still reclaimed promptly. A fixed-pool seat
          earns the same RAM protection as a recently-demanded model here, since seating is a standing
          commitment to keep the model ready.

        ``under_pressure`` is the measured-budget override (the WS-1 "aggregate budget"): the
        fits-regime assumption that model-count <= process-count implies the resident set fits the
        device only holds for sd15-class weights, so when the VRAM (or RAM) budget reports the resource
        cannot absorb the next job, residency protection for that resource is dropped to let an idle
        resident model be evicted. It never overrides the in-progress / next-model guards in the caller,
        so live and imminent work is still never evicted.
        """
        if model_name is None:
            return False

        if under_pressure:
            # A model holding a whole-card residency must never be evicted from VRAM, even under
            # budget pressure: evicting it undermines the residency convergence (the pre-staged head
            # cannot reach sole residency and dispatch is permanently blocked until save-our-ship
            # soft-resets the pools). Only the residency holder is spared; other models are still
            # reclaimable.
            return any(state.model == model_name for _, state in self._held_residencies())

        if affinity_active(len(wanted_models), self._max_inference_processes) and model_name in wanted_models:
            return True

        # A pool seat is a standing commitment to serve the model, so in the overflow regime it earns the same
        # RAM residency the demand grace grants a recently-run model: cheap to hold and it spares a disk reload
        # between the seat's jobs. VRAM, the scarce resource, is still reclaimed promptly (this is the non-
        # pressure branch; the ``under_pressure`` override above never reaches here).
        is_pool_seat = model_name in self._pool_protected_models_provider()
        held_by_grace_or_seat = self._is_recently_demanded(model_name) or is_pool_seat
        return not vram and held_by_grace_or_seat

    def unload_post_process_models_from_vram(self, *, device_index: int | None = None) -> bool:
        """Ask an idle post-processing lane to unload its modules while keeping the lane alive."""
        unloaded_any = False
        for process_info in self._process_map.values():
            if process_info.process_type is not HordeProcessType.POST_PROCESS:
                continue
            if device_index is not None and process_info.device_index != device_index:
                continue
            if process_info.is_process_busy():
                logger.debug(f"Post-processing process {process_info.process_id} is busy")
                continue
            if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                continue

            logger.info(f"Unloading post-processing models from VRAM on process {process_info.process_id}")
            if (
                not process_info.safe_send_message(
                    HordeControlMessage(control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM),
                )
                and not self._state.shutting_down
            ):
                logger.warning(
                    f"Failed to send UNLOAD_MODELS_FROM_VRAM to post-processing process "
                    f"{process_info.process_id}; marking the lane for replacement.",
                )
                self._process_lifecycle.post_process_processes_should_be_replaced = True
            process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
            unloaded_any = True
            self._record_churn("vram_eviction")
        return unloaded_any

    def release_allocator_cache(self, process_id: int) -> bool:
        """Ask one process to release its torch allocator cache without unloading its models.

        The cache-only reclaim actuator: an arbiter RELEASE_CACHE command and the post-stage lane policy
        both land here, returning an allocator's reserved-but-unused device blocks to the card without
        evicting any resident model. Mirrors the safe-send path of the unload senders. Returns True if
        the flag was delivered, False if the process is absent or the send failed.
        """
        process_info = self._process_map.get(process_id)
        if process_info is None:
            return False
        delivered = process_info.safe_send_message(
            HordeControlMessage(control_flag=HordeControlFlag.RELEASE_ALLOCATOR_CACHE),
        )
        if delivered:
            logger.debug(f"Asked process {process_id} to release its allocator cache")
        return delivered

    def unload_models_from_vram(
        self,
        process_with_model: HordeProcessInfo,
        *,
        under_pressure: bool = False,
        for_head_of_queue: bool = False,
        device_index: int | None = None,
        make_room_for_model: str | None = None,
    ) -> bool:
        """Unload models from VRAM from processes that are not running a job.

        ``under_pressure`` (set by the VRAM budget when the next job does not fit) drops residency
        protection and the single-model hold-back so the coldest idle resident copy is reclaimed,
        while still never touching an in-progress or next-up model.

        ``for_head_of_queue`` is the last-resort escalation when the head-of-queue job cannot be loaded
        and gentle reclaim freed nothing because every idle resident copy is another *queued* job's
        model: it additionally overrides the queued-lookahead guard so the head can be given room. It
        never evicts an in-progress (live) model.

        ``device_index`` restricts eviction to idle resident copies on that one card: reclaiming VRAM for a
        load onto card C must evict from card C, since freeing another card's model returns no VRAM to C.
        None (the single-GPU / worker-wide case) considers every card's idle residents.

        ``make_room_for_model`` names the model ``process_with_model`` is being cleared for, which decides how
        far that slot is spared (:meth:`_target_slot_is_spared`). A caller that names no model is performing a
        bare pressure sweep and the slot is spared outright; a caller that names one keeps the slot's own copy
        of that model and its in-flight load, while a different idle resident there is evicted as the ordinary
        model swap. On a worker with one inference lane that swap is the only reclaim the card has to offer.

        Returns True if an idle resident model's unload was issued (room is on the way), False if there
        was nothing to reclaim.
        """
        bridge_data = self._runtime_config.bridge_data
        wanted_models = self._compute_wanted_models()
        next_n_models = list(self.get_next_n_models(self._max_inference_processes))
        self._log_next_models_for_vram_unload(
            next_n_models,
            under_pressure=under_pressure,
            for_head_of_queue=for_head_of_queue,
        )

        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress}

        unloaded_any = False
        for process_info in self._process_map.values():
            if process_info.process_id == process_with_model.process_id and self._target_slot_is_spared(
                process_info,
                make_room_for_model=make_room_for_model,
            ):
                continue

            if process_info.process_type == HordeProcessType.POST_PROCESS:
                if device_index is not None and process_info.device_index != device_index:
                    continue

                if process_info.is_process_busy():
                    logger.debug(f"Post-processing process {process_info.process_id} is busy")
                    continue

                if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                    continue

                logger.info(f"Unloading post-processing models from VRAM on process {process_info.process_id}")
                if (
                    not process_info.safe_send_message(
                        HordeControlMessage(control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM),
                    )
                    and not self._state.shutting_down
                ):
                    logger.warning(
                        f"Failed to send UNLOAD_MODELS_FROM_VRAM to post-processing process "
                        f"{process_info.process_id}; marking the lane for replacement.",
                    )
                    self._process_lifecycle.post_process_processes_should_be_replaced = True
                process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
                unloaded_any = True
                self._record_churn("vram_eviction")
                continue

            if process_info.process_type != HordeProcessType.INFERENCE:
                continue

            if device_index is not None and process_info.device_index != device_index:
                continue

            if process_info.is_process_busy() and not self._slot_is_reclaimable_while_busy(process_info):
                logger.debug(f"Process {process_info.process_id} is busy")
                continue

            if process_info.loaded_horde_model_name is not None:
                if len(bridge_data.image_models_to_load) == 1 and not under_pressure:
                    logger.debug("Not unloading models from VRAM because there is only one model to load.")
                    continue

                if process_info.loaded_horde_model_name in in_progress_models:
                    continue

                # Spare the resident copy of ANY model in the queue lookahead, not just one of them:
                # evicting a queued model's weights trades the room gained for a guaranteed reload when
                # its job's turn comes. The protection is affordability-gated: on a card that cannot
                # physically hold the resident's footprint alongside the imminent job's sampling peak,
                # sparing it would force driver demand-paging (far costlier than the reload it saves),
                # so the copy is evicted. The head-of-queue escalation overrides the protection outright
                # (the head has priority for the room when every idle resident copy is another queued
                # job's model).
                if (
                    process_info.loaded_horde_model_name in next_n_models
                    and not for_head_of_queue
                    and self._coresident_lookahead_affordable(
                        process_info.loaded_horde_model_name,
                        device_index=device_index,
                    )
                ):
                    continue

                if not for_head_of_queue and self._residency_protects_from_unload(
                    process_info.loaded_horde_model_name,
                    wanted_models,
                    vram=True,
                    under_pressure=under_pressure,
                ):
                    continue

                if process_info.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                    logger.info(
                        f"Unloading model {process_info.loaded_horde_model_name} from VRAM on process "
                        f"{process_info.process_id}",
                    )
                    process_info.safe_send_message(
                        HordeControlModelMessage(
                            control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM,
                            horde_model_name=process_info.loaded_horde_model_name,
                        ),
                    )
                    process_info.clear_job_references()
                    self._note_retention_evicted_unused(process_info)
                    process_info.clear_retained_resident()
                    process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
                    unloaded_any = True
                    self._record_churn("vram_eviction")
            else:
                if process_info.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                    logger.debug(f"Unloading all models from VRAM on process {process_info.process_id}")
                    if (
                        not process_info.safe_send_message(
                            HordeControlMessage(
                                control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM,
                            ),
                        )
                        and not self._state.shutting_down
                    ):
                        logger.warning(
                            f"Failed to send UNLOAD_MODELS_FROM_VRAM to process {process_info.process_id}. ",
                            "This may indicate the process is unresponsive or has already exited. "
                            "Attempting to replace the process with a new one.",
                        )
                        self._process_lifecycle._replace_inference_process(process_info)
                    self._note_retention_evicted_unused(process_info)
                    process_info.clear_retained_resident()
                    process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
                    unloaded_any = True

        return unloaded_any

    def unload_from_ram(self, process_id: int) -> None:
        """Unload models from a process."""
        if process_id not in self._process_map:
            raise ValueError(f"process_id {process_id} is not in the process map")

        process_info = self._process_map[process_id]

        if process_info.process_type in (HordeProcessType.POST_PROCESS, HordeProcessType.SAFETY):
            if process_info.is_process_busy():
                logger.warning(
                    f"{process_info.process_type.name} process {process_id} is busy, not unloading models from RAM",
                )
                return

            logger.debug(f"Unloading {process_info.process_type.name} models from RAM on process {process_id}")
            process_info.safe_send_message(
                HordeControlMessage(
                    control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_RAM,
                ),
            )
            process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_RAM
            self._process_map.on_model_ram_clear(process_id=process_id)
            return

        if process_info.process_type in (HordeProcessType.COMPONENT, HordeProcessType.VAE_LANE):
            if process_info.is_process_busy():
                logger.warning(f"Service lane {process_id} is busy, not unloading models from RAM")
                return

            logger.debug(f"Unloading service-lane models from RAM on process {process_id}")
            process_info.safe_send_message(
                HordeControlMessage(
                    control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_RAM,
                ),
            )
            process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_RAM
            self._process_map.on_model_ram_clear(process_id=process_id)
            return

        if process_info.process_type != HordeProcessType.INFERENCE:
            logger.warning(
                f"Process {process_id} is not an inference, safety, post-processing, or service-lane process, "
                "not unloading models"
            )
            return

        if process_info.recently_unloaded_from_ram:
            return

        if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
            return

        if process_info.loaded_horde_model_name is not None and self._horde_model_map.is_model_loaded(
            process_info.loaded_horde_model_name,
        ):
            logger.debug(f"Unloading model {process_info.loaded_horde_model_name} from RAM on process {process_id}")
            process_info.safe_send_message(
                HordeControlModelMessage(
                    control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_RAM,
                    horde_model_name=process_info.loaded_horde_model_name,
                ),
            )

            self._horde_model_map.update_entry(
                horde_model_name=process_info.loaded_horde_model_name,
                load_state=ModelLoadState.ON_DISK,
                process_id=process_id,
            )

            process_info.clear_job_references()
            process_info.loaded_horde_model_name = None
            process_info.loaded_horde_model_baseline = None
            process_info.recently_unloaded_from_ram = True
            process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_RAM

        else:
            if (
                process_info.last_process_state == HordeProcessState.PROCESS_ENDING
                or process_info.last_process_state == HordeProcessState.PROCESS_ENDED
            ):
                return

            logger.debug(f"Unloading all models from RAM on process {process_id}")
            process_info.safe_send_message(
                HordeControlMessage(
                    control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_RAM,
                ),
            )
        logger.debug(f"Clearing process {process_id} of model {process_info.loaded_horde_model_name}")
        self._process_map.on_model_ram_clear(process_id=process_id)

    def _contain_idle_lane_ram(self) -> None:
        """Ask each idle service lane holding excess resident RAM to unload its models, throttled per lane.

        Runs every governance tick as a bounded sawtooth. A disaggregation service lane (the COMPONENT
        text-encode lane and the VAE decode lane) keeps its components resident across jobs and, alternating
        between the hot pool models, ratchets its resident set well above the RAM its working encoders occupy.
        When a lane is idle (:meth:`HordeProcessInfo.can_accept_job`) and its reported RSS exceeds
        :data:`_LANE_RAM_CONTAINMENT_RSS_BYTES`, it is sent ``UNLOAD_MODELS_FROM_RAM`` via
        :meth:`unload_from_ram`; the lane re-pages its encoders on its next stage, so the only cost is one
        reload. The unload never interrupts a stage: a lane handles its pipe messages serially and finishes any
        in-flight encode or decode before it reads the control message. A per-lane throttle
        (:data:`_LANE_RAM_CONTAINMENT_MIN_INTERVAL_SECONDS`) keeps the reload cost negligible against the RAM
        returned.

        The threshold sits well below the host RAM danger floor, so containment actuates both before and under
        host RAM pressure: service lanes participate in the RAM-pressure response rather than being exempt from
        it.
        """
        now = time.monotonic()
        for process_info in self._process_map.values():
            if process_info.process_type not in (HordeProcessType.COMPONENT, HordeProcessType.VAE_LANE):
                continue
            if not process_info.is_process_alive() or not process_info.can_accept_job():
                continue
            if process_info.ram_usage_bytes < _LANE_RAM_CONTAINMENT_RSS_BYTES:
                continue
            last_contained = self._lane_ram_containment_at.get(process_info.process_id)
            if last_contained is not None and (now - last_contained) < _LANE_RAM_CONTAINMENT_MIN_INTERVAL_SECONDS:
                continue
            logger.opt(ansi=True).info(
                f"<fg #ff8c69>Idle {process_info.process_type.name} lane {process_info.process_id} holds "
                f"{process_info.ram_usage_bytes / (1024 * 1024):.0f}MB resident (above the lane RAM "
                f"containment ceiling); unloading its models from RAM. It re-pages on its next stage.</>",
            )
            self._lane_ram_containment_at[process_info.process_id] = now
            self.unload_from_ram(process_info.process_id)

    def get_next_n_models(self, n: int) -> list[str]:
        """Get the next n models that will be used in the job deque."""
        next_n_models: list[str] = []
        jobs_traversed = 0
        while len(next_n_models) < n:
            if jobs_traversed >= len(self._job_tracker.jobs_pending_inference):
                break

            model_name = self._job_tracker.jobs_pending_inference[jobs_traversed].model

            if model_name is None:
                raise ValueError(f"job_deque[{jobs_traversed}].model is None")

            if model_name not in next_n_models:
                next_n_models.append(model_name)

            jobs_traversed += 1

        return next_n_models

    def _evict_unprotected_components_under_pressure(self) -> bool:
        """Evict idle, unprotected staged components from RAM ahead of the whole-RAM unload; return progress.

        The RAM-pressure path's gentle first rung. When the budgeted component cache holds staged checkpoints
        that no queued or in-flight job needs, dropping just those returns host RAM while every model a job
        still needs stays warm, unlike the coarser whole-RAM unload that clears a slot's entire cache. The
        protected set is every model with live demand (:meth:`_compute_wanted_models`: resident, queued, and
        in-progress, which covers the head-of-queue and every dispatched job); each slot's own tracked active
        model is also spared and left to the whole-RAM unload, which keeps the parent's per-slot model
        bookkeeping consistent, so this reclaims only the extra budgeted-cache residents.

        A fixed-pool seat with a live job is protected the same as any demanded model (it appears in the
        wanted set through that job); a seat with no pending or in-progress job carries no protection here at
        all, so it yields its staged components under true RAM pressure. This is the deadlock-avoidance
        posture: seat holds must never keep the host above its RAM floor. Each seat actually yielded
        (:meth:`_seat_only_idle_models` members among the evicted) is reported to the pool via the
        manager-supplied notifier on the rising edge of its yield.

        Returns:
            True when it commanded at least one eviction (this rung made progress, so the caller need not run
            the whole-RAM unload this tick); False when the cache is untracked or everything held is
            protected, so the caller falls through to the existing whole-RAM unload unchanged.
        """
        if self._component_residency_map is None:
            return False

        seat_only_idle_models = self._seat_only_idle_models()
        protected = self._compute_wanted_models()
        acted = False
        yielded_seat_models: set[str] = set()
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE or not process_info.is_process_alive():
                continue
            held = self._component_residency_map.checkpoint_models_held_on([process_info.process_id])
            evictable = held - protected
            if process_info.loaded_horde_model_name is not None:
                evictable = evictable - {process_info.loaded_horde_model_name}
            if not evictable:
                continue
            if process_info.safe_send_message(HordeEvictComponentsControlMessage(identities=sorted(evictable))):
                logger.opt(colors=True).info(
                    f"<fg #ff8c69>RAM pressure: evicting {len(evictable)} idle unprotected staged component(s) "
                    "{} "
                    f"from process {process_info.process_id} before a whole-RAM unload.</>",
                    sorted(evictable),
                )
                acted = True
                yielded_seat_models.update(evictable & seat_only_idle_models)

        self._announce_pool_pressure_yield(frozenset(yielded_seat_models))
        return acted

    def _announce_pool_pressure_yield(self, yielded_seat_models: frozenset[str]) -> None:
        """Notify the pool of seated idle models that yielded staged components under RAM pressure.

        Edge-triggered against the previously-yielded seat set: the manager-supplied notifier fires once per
        model that newly yields (so a seat is not re-reported every sub-second pressure tick while its child
        drains the eviction), and a single summary line names that rising-edge set. Passing an empty set (the
        common case, and every pressure tick that yields no seat) clears the latch so a later re-yield of the
        same model is announced afresh.

        Args:
            yielded_seat_models (frozenset[str]): The seated idle models whose staged components this pass
                actually evicted.
        """
        newly_yielded_models = yielded_seat_models - self._pool_pressure_yielded_models
        self._pool_pressure_yielded_models = yielded_seat_models
        if not newly_yielded_models:
            return

        for model_name in sorted(newly_yielded_models):
            self._on_pool_pressure_eviction(model_name)
        logger.opt(colors=True).info(
            f"<fg #ff8c69>RAM pressure: {len(newly_yielded_models)} seated idle pool model(s) "
            "{} yielded their staged components; the pool has been notified.</>",
            sorted(newly_yielded_models),
        )

    def unload_models(self, *, under_pressure: bool = False, for_head_of_queue: bool = False) -> bool:
        """Unload one idle model from RAM that is no longer needed; return True if one was unloaded.

        ``under_pressure`` (set by the RAM budget when the next job's RAM cost does not fit available
        system memory) drops the RAM residency grace so an idle resident copy is reclaimed rather than
        held, the guard against resident-in-RAM weights forcing the OS to page.

        ``for_head_of_queue`` is the last-resort escalation when the head-of-queue job cannot be loaded
        and gentle reclaim freed nothing because every idle resident copy is another *queued* job's
        model: it additionally overrides the still-needed-by-a-pending-job guard so the head can be
        given room. It never evicts an in-progress (live) model.
        """
        bridge_data = self._runtime_config.bridge_data

        # An empty queue short-circuits the *normal* path (nothing to make room for), but not the pressure
        # path: under the RAM danger floor the held pops drain the queue, and the idle resident footprint
        # left behind is precisely what must be reclaimed to get the host back off its floor. Gating the
        # reclaim on queued work would leave that footprint pinned forever, so the pop hold never lifts.
        if len(self._job_tracker.jobs_pending_inference) == 0 and not under_pressure:
            return False

        if (
            self._max_concurrent_inference_processes == 1
            and len(bridge_data.image_models_to_load) == 1
            and not under_pressure
        ):
            return False

        wanted_models = self._compute_wanted_models()
        in_progress_models = {job.model for job in self._job_tracker.jobs_in_progress}

        eligible: list[HordeProcessInfo] = []
        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue

            if process_info.is_process_busy() or process_info.last_process_state == HordeProcessState.PRELOADED_MODEL:
                continue

            if process_info.loaded_horde_model_name is not None:
                if self._horde_model_map.is_model_loading(process_info.loaded_horde_model_name):
                    continue

                # The map entry can be expired out from under a still-set loaded_horde_model_name (the
                # stale-loading sweep, or a dead process's entries): a missing entry is not IN_USE, so a
                # raw [] index here would crash the control loop. Treat absence as "free to reclaim".
                model_info = self._horde_model_map.root.get(process_info.loaded_horde_model_name)
                if model_info is not None and model_info.horde_model_load_state == ModelLoadState.IN_USE:
                    continue

                # Live (in-progress) work is never evicted, even when making room for the head. Pending
                # (merely queued) models are protected too in the normal path, but the head-of-queue
                # escalation may reclaim one of them since the head has priority for room.
                if process_info.loaded_horde_model_name in in_progress_models:
                    continue
                if not for_head_of_queue:
                    pending_models = {
                        job.model for job in self._job_tracker.jobs_pending_inference if job.model is not None
                    }
                    if process_info.loaded_horde_model_name in pending_models:
                        continue

                if not for_head_of_queue and self._residency_protects_from_unload(
                    process_info.loaded_horde_model_name,
                    wanted_models,
                    vram=False,
                    under_pressure=under_pressure,
                ):
                    continue

                eligible.append(process_info)

        if not eligible:
            return False

        # Among the reclaimable idle residents, sacrifice the cheapest cache to rebuild: a light model's
        # checkpoint reloads from disk in a fraction of a card-dominating one's time, so evicting by size
        # tier (map order breaking ties) keeps the most expensive warm copy alive whenever any cheaper
        # candidate can free the RAM instead. A lone heavy resident is still evicted, so the tier
        # preference can never wedge the reclaim.
        victim = min(eligible, key=lambda p: self._model_size_tier(p.loaded_horde_model_name))
        self.unload_from_ram(victim.process_id)
        return True

    def begin_scheduling_cycle(self) -> None:
        """Discard the selection state that is only valid within one scheduling cycle.

        The cached line-skip exists so the look-ahead and the dispatch of a single cycle agree on which job is
        selected, and its validity rests on process state the cycle itself does not re-derive: the target holds
        the cached job's model. Between cycles the child reports are applied, so that target may have given its
        weights back, and a cached pair naming a lane that holds nothing is undispatchable while suppressing
        selection of every job that is dispatchable. It is therefore scoped to the cycle rather than revalidated,
        and every driver of :meth:`preload_models` / :meth:`start_inference` must open its cycle here.
        """
        self._pending_line_skip = None

    async def run_scheduling_cycle(self, stable_diffusion_reference: dict[str, ImageGenerationModelRecord]) -> None:
        """Run a single scheduling cycle: preload, start inference, unload.

        This absorbs the inline orchestration block from _process_control_loop.
        """
        self.begin_scheduling_cycle()
        bridge_data = self._runtime_config.bridge_data

        self._refresh_model_demand()
        self.record_slot_duty(stable_diffusion_reference)

        # Resource governance is not driven here: the process manager runs run_governance_tick() every
        # control-loop iteration, so the danger-floor verdict and shed/restore response are already fresh
        # for this cycle regardless of whether any preload or dispatch happens.
        if not self.preload_models():
            # Fill every free inference slot this cycle rather than one per ~0.5s control-loop
            # tick: when several jobs complete close together, dispatching them one tick apart
            # leaves the GPU underfed. start_inference() returns False once no more can start
            # (its own concurrency gate: jobs_in_progress >= max_concurrent, no free process,
            # or no eligible job, stops the loop), so this cannot over-subscribe. There is no
            # worker-wide serialisation ahead of it: a workflow's extra weights are priced per
            # card by the admission and overlap gates, like a batch or a card-demanding model.
            started = 0
            while await self.start_inference():
                started += 1

            if not started:
                # Nothing dispatched this cycle though the queue has work: if the head has been parked
                # long enough to be a real stall (not a between-jobs gap), explain *why* it is not
                # dispatching. Throttled, read-only; it never changes scheduling.
                self._log_dispatch_stall_if_needed(stable_diffusion_reference)
                # Arm the idle-fill breaker off the same idle-head signal: if the head has been starved
                # long enough with a free sibling, let the popper over-pop a quick no-LoRA fill job.
                self._update_idle_fill_arm(bridge_data)
                self.unload_models()
