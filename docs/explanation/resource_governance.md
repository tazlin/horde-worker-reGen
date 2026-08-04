# Resource governance

How the worker decides, every scheduling cycle, whether the host can afford what it is about to do:
load another model, keep another process resident, start another job. This page explains the structure
of those decisions; the individual policies (the VRAM/RAM budget, the RAM danger floor, whole-card
residency) are described in [Performance and backpressure](performance_and_backpressure.md).

## Why governance is separated from scheduling

Resource incidents share a repeating shape: a protective check existed, but it was fused into one
scheduling path (a preload attempt, a dispatch), and the incident arrived through a different path
where the check never ran. A steady-state worker that never loads a new model can still grow its
resident RAM into an OS OOM kill; a model already resident on a card can still be dispatched into VRAM
another decision reserved. Protections tied to *how the work happens to flow* go dormant exactly when
the flow changes.

The worker therefore separates three roles that used to live in single methods:

- **Measurement**: reading live state (free VRAM per card, available host RAM, per-process resident
  RAM, process counts) happens in one place per decision and produces an immutable *snapshot*. A
  decision never re-measures midway, so one decision acts on one consistent picture of the host.
- **Decision**: pure functions over a snapshot return typed values (remedy *actions*, outcome enums).
  They touch no live state, so every policy is unit-testable by constructing a snapshot and asserting
  on the returned decision, with no process pool, no monkeypatching.
- **Execution**: a single dispatcher applies the returned actions to the live worker (pause pops, evict
  a model, shrink or grow a process pool, recycle a process). Multi-tick bookkeeping is mutated only
  here, with the measured result of each remedy, so what the governor believes always reflects what
  actually happened.

The decision layer lives in the
[`governance`][horde_worker_regen.process_management.scheduling.governance] package; the
[`InferenceScheduler`][horde_worker_regen.process_management.scheduling.inference_scheduler.InferenceScheduler]
provides the measurement and execution surfaces.

## The governor tick

[`ResourceGovernor.tick`][horde_worker_regen.process_management.scheduling.governance.governor.ResourceGovernor.tick]
runs once per control-loop iteration, unconditionally, driven by the process manager (via the scheduler's
`run_governance_tick`) rather than as a step inside a scheduling cycle. That placement is the structural
fix for the dormant-check shape above: governance does not depend on any particular scheduling path
executing, nor on the inference queue being non-empty, so a worker that never attempts a preload, or one
whose queue a pop hold has drained to empty, is governed exactly as often as one actively serving work.
Gating the tick on a non-empty queue would let the soft pop hold self-latch: the hold blocks pops, the
queue drains, and the only thing that clears the hold (the tick) would never run again.

One tick measures the RAM danger-floor verdict and one
[`HostMemorySnapshot`][horde_worker_regen.process_management.scheduling.governance.snapshots.HostMemorySnapshot],
then decides and executes both regimes:

- **Pressured host** (below the danger floor): the degrade response. Arm the self-throttle pop pause,
  set the soft pop hold, evict idle resident models, shed idle inference contexts (per card on a
  multi-GPU host, never emptying a card), and reclaim a process whose resident RAM crossed the
  per-process ceiling (recycled if idle, drained first if busy). The idle-model eviction is *not* gated on
  there being queued work: the pop hold drains the queue, so the footprint left to reclaim is exactly the
  idle resident set on an empty queue, and eviction must still reach it. When no idle model remains to
  unload, an idle slot that kept the freed model's allocator pages is cycled to return that RAM to the OS.
  Without this the host would stay pinned under its floor with the pop hold latched on indefinitely.
- **Recovered host**: the restore response, and the drain follow-through. Cards the reduction shed grow
  back toward their planned process count, one context per card per tick, gated on measured RAM headroom
  actually fitting another resident working set. A drain the pressure episode initiated but did not
  finish still resolves here
  ([`decide_draining_followthrough`][horde_worker_regen.process_management.scheduling.governance.ram_governor.decide_draining_followthrough]):
  the marked process is recycled once idle (or unmarked if it shrank under the ceiling or exited), because
  the mark holds job pops and shed restore closed until it resolves. New drains are never initiated on a
  recovered host.

The two regimes are mutually exclusive by construction, so one combined execution never both sheds and
restores. The tick's verdict is retained for the rest of the cycle: per-job gates (such as the preload
RAM-floor defer) read it instead of re-measuring, so a whole cycle acts on one reading.

The single-GPU shed record tracks the *live* shortfall below plan (planned minus loaded), not a running
total of reductions. This matters because the inference-process count is also moved by two other, unrelated
mechanisms: whole-card residency collapses the pool to the residency holder and grows it back when the
residency drains, and the VRAM arbiter's live-context reduction stops idle contexts under admission pressure
and has the verified reclaim ladder grow them back when the card recovers. When either restore regrows the
pool, it reconciles the RAM shed record against the live count, dropping it once the pool is back at plan.
Without that reconciliation the record would linger as a stale claim that the pool is still short, and while
the host stayed under its floor the governor would re-shed the pool the restore just regrew, cycle after
cycle, without ever returning to steady state.

The same unconditional tick re-prices every held whole-card residency. The forecast captured when the
residency was granted is immutable: it remains the evidence and fit guarantee that justified the grant.
Current allocator-derived checkpoint footprints and live reserve commitments instead produce a separate
`repriced_target`. That target may only tighten while the residency is held. If it falls below the live
context count, the scheduler stops idle siblings and records one `ContextReduction` restore obligation with
the verified reclaim ladder; a roomier later reading never grows contexts back underneath the heavy model.
The residency release owns physical regrowth and discharges the ladder debt. A foreign application's VRAM
growth after dispatch is not folded into this model price: the device-free governor observes that device-level
condition and drives verified reclaim independently of queue or residency pricing.

## Idle service-lane RAM containment

The degrade response above sheds and cycles inference slots, but the disaggregation service lanes (the
COMPONENT text-encode lane and the VAE decode lane) are a separate RAM source: each keeps its encoders
resident across jobs and, alternating between the hot pool models, ratchets its resident set well past
what its working encoders occupy. The governor tick therefore also runs
[`_contain_idle_lane_ram`][horde_worker_regen.process_management.scheduling.inference_scheduler.InferenceScheduler._contain_idle_lane_ram]
every iteration: an idle lane (one accepting a job, so never mid-stage) whose reported resident RAM
crosses `_LANE_RAM_CONTAINMENT_RSS_BYTES` is sent `UNLOAD_MODELS_FROM_RAM`. Unlike an inference slot, a
lane self-reloads its encoders on its next stage, so containment is an in-process RAM unload rather than a
process cycle; the only cost is one reload. A control message cannot interrupt a stage (a lane reads its
pipe serially and finishes any in-flight encode or decode first), and a per-lane throttle
(`_LANE_RAM_CONTAINMENT_MIN_INTERVAL_SECONDS`) keeps the reload cost negligible against the RAM returned.

The containment ceiling sits well below the host RAM danger floor, so this actuates as a bounded sawtooth
before the floor is threatened as well as under active pressure: the lanes participate in the RAM-pressure
response rather than being exempt from it. Lane processes are deliberately left out of the creep-cycling
path (which only cycles inference slots); the in-process unload is the whole containment for a lane.

A complementary release happens inside `hordelib`: when the component cache drops an entry (a budget
eviction or a single-slot replacement), it issues a throttled, best-effort `gc.collect()` plus host trim
so the displaced component's cold pages return to the OS. That trim reclaims only components the comfy
model-management layer has already released; the guaranteed reclaim of every held component is this
worker-driven RAM unload.

### Honest RAM accounting and per-process residency

The RAM figures the containment logic reacts to are also what the operator sees, and both must be honest.
The per-role RAM breakdown (the console status block's *Worker* line and the Live tab's *Worker RAM by
role* panel) sums every worker process's resident-set size into its role. Every child on the process map
is bucketed: inference and safety alongside the component, VAE, post-processing, and image-utilities lanes.
A lane's footprint is therefore attributed to the worker rather than being mislabelled as *Other (OS +
apps)*, which stays a genuine remainder (system-used RAM minus the worker subtotal, clamped at zero because
resident-set sizes over-count pages shared between processes). See
[`system_memory`][horde_worker_regen.process_management.resources.system_memory] for the role set and the
derivation.

Each GPU-bearing child also reports which model components it holds resident in its RAM component cache
(identity, kind, and the cache's approximate MB for each entry), carried on every memory report and
surfaced per process on the Live tab. The listing is the cache's own view, so it is approximate: a
process's resident-set size minus the summed listed approximation is a *retained/unattributed* remainder,
the allocator-retained or comfy-held pages the cache no longer lists. Surfacing that remainder makes the
resident-set ratchet the containment logic fights visible rather than hidden inside one opaque RAM total.

## Governance baseline and the healthy-hold watchdog

Because the soft pop hold and the governor's shed/draining bookkeeping live in worker state rather than in
the process pool, rebuilding the pool does not clear them. The scheduler exposes a single
`reset_governance_to_baseline` that drops the RAM pop hold, clears the shed-card / draining / single-GPU
shed records, and forgets the RAM-pressure pop-skip reason, leaving flags owned by other subsystems (the
shared self-throttle pause, the operator supervisor pause, the downloads-only hold, the post-processing
and torch-compat breakers) untouched. It is safe to call at any time: the next governance tick re-derives
whatever the live host warrants. The save-our-ship soft reset calls it so a pool rebuild also returns
governance to a clean slate.

A standalone watchdog in the recovery coordinator (`maybe_reset_stuck_governance_hold`) is the last-resort
guard against a pop hold that stays engaged after the host is healthy. It fires only when the pop hold is
set, the most recent danger-floor verdict is healthy, nothing is draining, none of the deliberate
held-queue graces (whole-card establishment, heavy-head load, RAM-reclaim cycle) are active, no inference
is in progress, and the queue is empty, sustained past a grace window. The pop-hold-set term is what
distinguishes a genuine latch from a merely idle worker with no matching jobs. It escalates in tiers: first
a governance-baseline reset, and only if the hold re-latches despite a healthy host does it rebuild the
(all-idle) inference pool. It is deliberately not an `assess_wedge` trigger, because that would apply the
soft reset's unconditional pool churn to a pool that is actually healthy.

## Decisions are values

Remedies are expressed as inert command objects
([`GovernanceAction`][horde_worker_regen.process_management.scheduling.governance.actions]) and policy
outcomes as enums such as
[`RamReclaimOutcome`][horde_worker_regen.process_management.scheduling.governance.preload_admission.RamReclaimOutcome].
This buys three things:

- **Testability**: target exclusion, load serialization, card ordering, and system-RAM reclaim outcomes are
  pure functions tested with plain value inputs.
- **One execution site per side effect**: every RAM remedy executes through the scheduler's single
  action dispatcher, so there is exactly one place a remedy's log line, bookkeeping, and measured
  result live.
- **Reviewability**: a change to *policy* is a change to a pure function and its table of cases, not to
  a method interleaving measurement, judgment, and process manipulation.

## The admission pipeline

The preload loop walks the pending queue and, per job, runs a sequence of named gates. The pass uses
[`AdmissionDecision`][horde_worker_regen.process_management.scheduling.governance.preload_admission.AdmissionDecision]
as its shared vocabulary (continue, stop, admit, defer by RAM pressure/concurrency/budget, pre-stage,
unserviceable). The scheduler keeps a tiny private adapter from those decisions to pass control, while
the judgment calls live as pure functions in
[`preload_admission`][horde_worker_regen.process_management.scheduling.governance.preload_admission]:

1. **Target exclusion**: which slots this preload may not displace (the queued-model guard, model to
   process affinity, slots draining for RAM reclaim), composed by
   [`compute_preload_disallowed_processes`][horde_worker_regen.process_management.scheduling.governance.preload_admission.compute_preload_disallowed_processes].
   The guards are exclusions only, never a wedge: the starved-head fallback
   ([`select_head_room_process_id`][horde_worker_regen.process_management.scheduling.governance.preload_admission.select_head_room_process_id])
   deliberately overrides them while still never displacing live work.
2. **Placement**: on a multi-GPU host, which card receives a fresh load
   ([`card_preload_order`][horde_worker_regen.process_management.scheduling.governance.preload_admission.card_preload_order]:
   a card already serving the model first, then the least-loaded card, then the card with the most measured
   free VRAM). The final tie-break keeps an otherwise-idle card that is carrying safety or post-processing
   context from winning a preload that another equally idle card can seat with more headroom.
3. **Load serialization**: whether another checkpoint may load on this device right now
   ([`preload_concurrency_blocked`][horde_worker_regen.process_management.scheduling.governance.preload_admission.preload_concurrency_blocked]).
4. **Budget verdicts**: VRAM admission is owned by
   [`VramArbiter`][horde_worker_regen.process_management.resources.vram_arbiter.VramArbiter]. A non-fitting
   request defers while verified reclaim can still make progress, and a foreign-pressure request admits only
   when it physically fits measured device-free VRAM net of the noise buffer. Before admitting, the scheduler
   reclaims idle system RAM only when the RAM budget judges the incoming load short of available memory: a
   heavy head does route its checkpoint through RAM first, but purging a sibling's warm RAM copy on a host
   with ample headroom only converts that sibling's next job into a disk reload. When RAM reclaim has run,
   [`decide_ram_reclaim_outcome`][horde_worker_regen.process_management.scheduling.governance.preload_admission.decide_ram_reclaim_outcome]
   resolves whether to wait for the reclaimed memory to show up or proceed. Same-tick reclaim side effects
   stay behind the
   [`ReclamationExecutor`][horde_worker_regen.process_management.scheduling.governance.preload_admission.ReclamationExecutor]
   protocol, implemented by the scheduler because it owns live process state and logging. The RAM charge is
   marginal-clean: a reload onto a warm slot is credited its reusable retained pages, and a
   disaggregation-class job is charged its UNet-only component cost (from the checkpoint's component-identity
   sidecar) rather than the whole checkpoint the sampler never fully stages. Both are charging-honesty
   corrections that never mix the device-level baseline into a per-job marginal; see
   [Performance and backpressure](performance_and_backpressure.md#the-vram-and-ram-budget).

Two scoping rules keep these last-resort remedies from taxing a healthy host:

- **Exclusivity follows the footprint and the room.** An over-budget admit runs with the device to
  itself (`overbudget_exclusive_mode`) only when the streaming forecast's `admit_requires_isolation`
  holds: the model's persistent footprint dominates the card *and* the card lacks room for a sibling
  model beside it. The footprint charges every component the engine force-loads over a job (core
  diffusion weights plus text encoders and the VAE), not the checkpoint's core weights alone: a
  multi-component model judged by its core weights looks co-residable on a card where its own
  components will in fact evict each other all job long. Isolation protects a heavy checkpoint from a concurrent sibling load spilling its
  weights to host RAM; a card-light model that reaches the admit through reserve arithmetic alone (free
  VRAM depressed by retained sibling contexts) shares the device, and so does a card-dominating model on
  a genuinely roomy card, whose no-co-sampling contract the overlap gate enforces without freezing the
  sibling lane through the admit. Once routing has chosen a card, the exclusive marker carries that planned
  device and suppresses only that card's preload and dispatch consumers. A marker created before attribution
  remains worker-wide until a card is known, so missing routing evidence is conservative rather than permissive.
  If dispatch commits on a different card, that actual card replaces the plan before later consumers run.
  Dispatch suppression additionally requires a live claim on the card: the exclusive job is sampling, or its
  whole-card residency is held or being established. The marker itself lasts the life of the job (it also
  carries fault attribution and the over-budget step grace), so a staged job whose establishment is deferred
  by the residency rate limiter does not hold back unrelated work while it waits.
- **RAM eviction sacrifices the cheapest cache.** When an idle RAM resident must be reclaimed, the
  victim is the smallest size-tier candidate (map order breaking ties), never a card-dominating
  checkpoint whose disk reload costs several times an ordinary model's, unless it is the only candidate.

## Whole-card residency state

The whole-card exclusive-residency records (which model holds which card, when the hold was
established, its cooldown and restore stamps) live in the
[`WholeCardResidencyMachine`][horde_worker_regen.process_management.scheduling.governance.whole_card.WholeCardResidencyMachine].
It extends the ledger queries (phase, grace windows, model holder lookup, drain backstop) with the pure
transition decisions the scheduler can ask without touching live process objects: whether a head demands
residency, what process count the residency targets, and whether teardown is complete enough for dispatch.
The transitions that touch live processes (establishing a residency, converging it to sole residency,
restoring siblings afterward) remain scheduler methods, but they read and write state exclusively through
the machine. A resident heavy head goes through the same machine-backed readiness gate as a newly-loaded
head, so an already-resident whole-card model cannot co-sample while sibling models still occupy the card.

The process target is a *measured* verdict, so it can be absent. When the forecast cannot size the card (no
reported total VRAM, or no per-process overhead to reason about) the machine reports no target rather than
one, and a caller leaves the live process count where it is: collapsing to sole residency there would tear
the pool down on the strength of a figure nobody measured, on exactly the hosts where the measurement is
missing. Readiness then turns on the residency's other legs (safety and the service lanes off the card, the
live weight fit or the drain backstop) instead of waiting forever on a depth that was never named. A
whole-card-intent baseline is the deliberate exception: it still sizes to one when its footprint cannot be
sized at all, since its tier asserts it never shares the card well.

The grant forecast and the target used while holding have intentionally different lifetimes. The former is
captured once and never rewritten. The governance tick derives a separate target from current raise-only,
allocator-attributable evidence and may lower it when the held model now needs more room. Both convergence
and dispatch readiness consult that tighten-only target. They never raise it mid-hold; capacity returns only
after the residency drains, through the normal restore path.

The dispatch gate releases the head either on a live free-VRAM reading that genuinely holds the weights, or,
once the bounded drain window is spent, on the forecast's sole-residency guarantee. That guarantee is sized
for a card every other context has left, safety included, so the scheduler passes in the charge of whatever
the residency is leaving on the card: where the operator has not permitted the residency to move safety
off-GPU, the resident safety process's whole device footprint (the one safety price: its seed raised by any
measured `SAFETY` watermark) is priced back out of the alone figure. Without that the backstop admits
the head against room only a departure the configuration forbids would free, and the weights load into a card
short by roughly the safety footprint and stream. Which contexts stay is a configuration and lifecycle fact,
not residency state, so the machine takes the charge as an input and remains a pure state machine.

Safety readiness is card-local as well. The scheduler passes `safety_clear_of_card`, computed from the safety
process's actual/headroom-selected card, rather than the global `safety_paused` flag. A live safety process on
card 1 is already clear of a residency on card 0; only the residency on the card safety occupies waits for it
to move. This avoids parking one GPU behind a service process physically resident on another.

The drain backstop runs from the moment the teardown's structural legs *first all pass*, not from the moment
the residency was granted. A slow establishment (siblings still exiting, safety still cycling off-GPU) would
otherwise spend the backstop before there was any sole-residency guarantee for it to admit the head against,
and the head would load into a card that had not yet cleared. The machine latches that moment when it answers
the readiness question, since that is where the structural legs are computed, and clears the latch on every
fresh grant so a re-established residency starts its own backstop rather than inheriting a spent one.

### Bounding residency churn

Entering and leaving sole residency are both expensive: the establishment stops sibling inference processes
and may move safety off-GPU, and the restore respawns each of those siblings (a ~20s spawn apiece, plus the
preload that follows). A demand signal that oscillates can therefore rebuild the pool continuously without
ever serving more work. Three governors, all held in the residency machine and actuated at the scheduler's
call sites, bound that churn:

- **Minimum hold.** A ready queue head for a *different* model may cut a drained residency's cooldown short,
  but not before the grant has held long enough to amortize the teardown and the regrowth the release itself
  will pay for. Without the floor, a queue alternating heavy and light heads turns into a continuous pool
  rebuild. The floor is shorter than the establish grace window, so it can never keep a residency alive past
  the point the recovery supervisor is watching.
- **Establishment rate limit.** Establishments are counted per card over a short rolling window. Past the
  allowance the head is **deferred**, never declined: deferring keeps it on the whole-card path so it re-asks
  the next cycle, whereas declining would send it down the ordinary co-resident path the forecast has already
  said streams its weights. The window is deliberately far shorter than the pop-gate structural-wedge
  backstop, so a deferred head can never accrue toward a wedge verdict. The limiter gates *new*
  establishments only; a residency already held keeps being driven to convergence every cycle.
- **Grace budget.** Each establish and each restore claims a window in which the recovery supervisor ignores
  a held queue (see [Resilience and recovery](resilience_and_recovery.md)). Those claims are charged against
  a per-card rolling budget, because otherwise repeated cycling re-arms a fresh window faster than the
  previous one expires and the supervisor stays disarmed for as long as the churn continues, which is exactly
  the state in which a real wedge goes unnoticed. Once the budget is spent, the grace claim is refused even
  inside a nominal window, and the scheduler discloses the refusal once on the transition. The budget
  replenishes as old grants age out of the window.

Only one reconciler may change safety placement. Whole-card residency contributes a persistent off-GPU veto,
the reclaim ladder contributes a one-shot request, and runtime placement contributes its hysteretic fit wish;
none calls lifecycle's safety pause or restore directly. The reconciler keeps the initiating `PauseOwner` for
attribution, waits until the replacement reaches readiness, and restores only after every request and veto has
cleared, the arbiter admits the safety load, and the device-free governor permits growth. A residency that ends
under a runtime-policy wish therefore cannot promote safety only for the policy to demote it again, and two
opposing wishes cannot chain intentional replacement windows.

Whole-card residency is off unless its config flag resolves to true. A flag that never resolved at all is
disclosed once at the scheduler, because a worker that quietly forgoes the residency simply loads heavy
models co-resident and streams them, which is very hard to attribute after the fact; an explicit `false` is
an operator decision and says nothing.
The pure sizing rule for how many live contexts a rejected peak can co-reside with is
[`max_coresident_for_peak`][horde_worker_regen.process_management.scheduling.governance.whole_card.max_coresident_for_peak].

## Extending governance

When adding a new resource protection, follow the same shape:

1. Put the readings it needs into the snapshot (or a new snapshot type) in
   [`snapshots`][horde_worker_regen.process_management.scheduling.governance.snapshots]: measurement
   stays in the scheduler, captured once per decision.
2. Express the remedy as a new action in
   [`actions`][horde_worker_regen.process_management.scheduling.governance.actions] and its execution
   as a new dispatcher case, so the side effect has one home.
3. Write the policy as a pure decide function with unit tests under
   `tests/process_management/governance/`.
4. Wire it into the governor tick, not into a scheduling path, unless it is genuinely scoped to one
   job's admission.

The scheduler-integrated behavior stays covered by the regression suites under
`tests/process_management/regressions/`, which drive real scheduling passes.
