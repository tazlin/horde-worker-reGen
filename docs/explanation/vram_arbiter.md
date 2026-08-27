# The VRAM arbiter

How the worker concentrates its device-memory admission decisions into one authority, and how cross-job
weight retention is governed separately by the device-free governor and the verified reclaim ladder.

This page assumes the separation of measurement, decision, and execution described in
[Resource governance](resource_governance.md), and the reserve/admission arithmetic in
[Performance and backpressure](performance_and_backpressure.md).

## Why one authority

Device VRAM on the worker is contended by several independent consumers: model preloads, monolithic
job dispatch, the disaggregated encode/sample/decode lanes, post-processing, safety, and the choice to
keep a just-run model resident across the next job. Each historically priced the card with its own
arithmetic and its own reading of free VRAM. On Windows/WDDM that reading lies exactly when it matters:
the driver silently demand-pages an over-commit into host RAM and keeps reporting healthy free VRAM, so
several gates independently trusting that figure cannot be reconciled into one coherent admission
picture.

The
[`VramArbiter`][horde_worker_regen.process_management.resources.vram_arbiter.VramArbiter]
is the single object those decisions can be expressed through. It reasons about one frozen measurement
per control-loop cycle and prices each request with the same ledger-driven identity, so admission is a
single inequality rather than a set of competing ones.

## The truthful signal hierarchy

Not every "free VRAM" figure is the same figure, and under WDDM they disagree exactly when it matters. The
arbiter's admission arithmetic is deliberately built to survive that disagreement, and a second, enforcement
layer (the device-free governor) reads the one figure that stays honest.

- **Per-process reads lie near the ceiling.** A child's `mem_get_info`, and the per-PID shared-segment
  counters behind Task Manager's "Shared GPU memory", cannot be trusted once the driver demotes an allocator
  to system memory: WDDM demotes the *least-recently-touched* allocator, so the process that goes slow (the
  active sampler) and the process whose shared memory grows (the idle newcomer) are usually **different**
  process ids, and the per-PID magnitude read for a given process varies run to run for one physical state.
  The arbiter therefore never prices against a per-process free figure; it prices against the committed
  *ledger* (what the worker itself placed on the card) plus its own planned overlay.
- **The NVML device-level total is truthful.** Read from the torch-free parent, outside any CUDA workload,
  the device used/free total does not lie. Throughput does not degrade gradually as it falls; it falls off a
  hard cliff the instant device-free reaches roughly zero, then plateaus, so the whole defense is to keep
  device-free from ever reaching it.

The **device-free governor** turns that truthful figure into a small hysteretic state machine per card
(HEALTHY / PRESSURE / SATURATED), sampled once per control-loop tick and debounced over two samples. Its
committed per-card state is carried into the arbiter's
[`DeviceVramState`][horde_worker_regen.process_management.resources.vram_arbiter.DeviceVramState] as a
read-only field, so the admission substrate can see the same proximity-to-cliff truth the enforcement layer
acts on. On a card at PRESSURE the scheduler holds new VRAM growth (no new model brought to VRAM on a process
that does not already hold it, no safety GPU restore, no paused-lane restart); at SATURATED the reclaim ladder
runs. In-flight sampling is never touched by the growth hold.

## Verified reclaim, the per-step floor, and the kill as last rung

Reclaim is single-owner. The governor's SATURATED ladder and the arbiter's per-cycle DEFER actuations run
through one engine, so the two triggers can never become two mechanisms evicting the same card by different
rules. The engine reclaims in LIFO order (newest idle resident first, since the driver demotes the
least-recently-touched allocator), and it **verifies**: after issuing a rung it compares the realized NVML
device-free gain against the rung's promised figure on each following governor sample, escalating only once the
rung has realized nothing further for a verification budget scaled by the memory it promised, and marking the
episode *unresolved* only once every rung has run without relieving the card. Because the budget scales with
the promise, an idle-resident rung is priced from what the slot is known to cost the card: its measured
allocator reservation where the slot has reported one, otherwise the checkpoint's measured resident watermark
or, failing that, the same static resident-footprint estimate a retention grant charges a tracked resident at.
Only a genuinely unknowable footprint prices at zero, which takes the unverifiable path (full budget, no
credit, escalate on an honest absence of evidence) rather than being graded on the base allowance alone.
Nothing frees synchronously, so
grading a rung on a sample count reads a working multi-gigabyte release as a failure; the deepest rung (safety
off the GPU) additionally carries a per-card dwell, because spending it is a whole process cycle. The
**per-step floor** is the fast detector that forces that ladder early: two
consecutive sampling steps each several times their expected per-step time, on a PRESSURE-or-SATURATED card,
mean a job is being demand-paged (not merely heavy) and reclaim should run without waiting for the whole-job
elapsed-ratio grade. Replacing the crawling sampler is the ladder's **last rung**: it fires only once the card
has been SATURATED past the kill horizon, the ladder is exhausted, and the slot is crawling. That kill gates on
device-level truth, never on the per-PID paging-victim map, which the LRU physics make structurally
unsatisfiable. The full mechanics live in
[Performance and backpressure](performance_and_backpressure.md#the-truthful-signal-hierarchy-and-the-device-free-governor).

A lane pause among the ladder's rungs (post-processing, VAE, or component lane off-GPU) frees a real CUDA
context that no external trigger restores, so each executed lane pause must have exactly one responsible restore
owner. A pause the governor's saturation episode issued is restored by that episode's LIFO unwind when the card
returns `HEALTHY`; a lane the per-cycle post-processing DEFER path borrowed is restored by that drain's
applied-action receipt (see
[Process lanes and chaining](process_lanes_and_chaining.md#restore-ownership-every-lane-pause-has-exactly-one-responsible-owner)).
The `execute_arbiter_commands` receipt returns exactly the commands whose actuator reported it acted. Decision
records retain both the ordered requested tuple and the ordered applied tuple; a proposed cache release,
eviction, context reduction, or lane pause is never reported as performed merely because the arbiter requested
it. A borrow therefore restores only a pause it truly acquired, never a same-owner pause an independent episode already held. A
conservative self-heal backstop in the governor tick reclaims any reclaim-ladder lane pause that has outlived
both owners (no live episode, no borrow receipt) once the card has been debounced-`HEALTHY`, so a lost claimant
can never strand a lane off the GPU indefinitely. The recovery coordinator's constructive remedy takes its lane
pauses through the same actuator, so the lifecycle records the ladder as their owner while the coordinator holds
the receipt: the backstop consults that claim too, or it would read a live remedy as an orphan, lift the pause
inside the remedy's yield window, and cold-start the lane process on every re-issue.

A live-context reduction is the episode's other restore obligation, and it unwinds on different terms because
undoing it costs a process cold start. Regrowing the pool on the first `HEALTHY` sample after a reduction reads
the reduction's own success as evidence the pressure has passed: the freed per-context VRAM is *why* the card is
healthy, so the regrowth re-inflates the footprint, the head whose rejected peak bought the reduction is
rejected again, and the pair oscillates at one cold start per cycle. The unwind therefore takes the reduction
only once the card has been continuously `HEALTHY` for a dwell and no head of queue is still parked, and the
reduction itself is rate-limited per card so a head that re-asks every cycle cannot buy one teardown per cycle.
Lane rungs keep the plain LIFO behaviour.

The reduction's only responsible restore is that same LIFO unwind, which runs on `HEALTHY`; a card that leaves
saturation but settles below the soft floor never reaches it, so the pool stays at emergency depth and the
worker serves at reduced concurrency long after the pressure that bought the reduction. A second backstop in the
governor tick closes that: while the card is no longer `SATURATED` and an episode still owes a reduction, it
regrows the pool through the same actuator the unwind uses once that has held for the same debounce interval. It
arms only off saturation, so a card the ladder is still working keeps the contexts it reclaimed and a card that
dips and re-saturates restarts the clock, and it holds like the unwind while a head of queue is still parked, so
the debounce cannot discharge an obligation the dwell is deliberately retaining. The actuator stands down while a whole-card residency owns the pool (that
residency's own restore owns the regrowth), and the obligation is discharged only when the actuator reports it
acted, so a stood-down card is retried rather than left shrunk.

The VAE-lane pause carries one further eligibility rule the arbiter's idle-target guarantee cannot express.
The arbiter only names a lane whose process is idle this instant, but an idle VAE lane may still have imminent
work: the disaggregation orchestrator holds jobs at the decode stage whose sampling already finished and whose
only remaining step is a short VAE decode on that lane. Pausing the lane out from under such a job reroutes it
monolithic and discards the finished sampling, to make room for a dispatch the decode itself would clear in
seconds. So the executing actuator reports the VAE-lane pause as a no-op while any disaggregated decode is
queued or in flight on the lane, and both reclaim paths (the governor rung and the post-processing borrow) then
move to their next relief option, exactly as they do for any rung whose target has gone away. The whole-card
residency's own VAE-lane pause asks the same predicate and retries on later convergence cycles. A job merely
sampling does not withhold the pause, and the component/text-encode lane is not gated, because rerouting a job
that has not yet finished sampling discards no completed work (see
[Process lanes and chaining](process_lanes_and_chaining.md#decode-drain-eligibility-a-vae-lane-pause-defers-to-a-queued-decode)).

## The four layers

The arbiter keeps four concerns deliberately separate:

- **Measurement** arrives from outside as a
  [`MeasuredVramSnapshot`][horde_worker_regen.process_management.resources.vram_arbiter.MeasuredVramSnapshot],
  assembled once per control-loop iteration from figures the parent already holds (the scheduler's
  committed ledger and headroom terms, the reconciler baseline, the orchestrator's in-flight sampling
  peaks). The arbiter performs no NVML read and imports no torch: it is pure decision state in the
  torch-free parent.
- **Estimation** prices a request's marginal device cost. The caller supplies the priced delta (or a
  stage's static spike figure); an unpriceable candidate is charged nothing, matching the predictive
  gate's admit-on-unknown-cost contract. Where a request prices *sampling* work (the disaggregated
  concurrent-sampling estimate, the measured-overlay candidate delta, and the post-processing
  co-residency gates), the scheduler prices it from a learned per-(baseline, resolution, platform, stage)
  peak held in the
  [`LearnedFootprintStore`][horde_worker_regen.process_management.resources.vram_footprints.LearnedFootprintStore],
  with the static per-model predictor as a floor: a measured SAMPLE-stage activation high-water can only
  ever *raise* the priced peak, never lower it, closing the systematic undershoot where a static
  weights-plus-step seed (~6GB for SDXL) plans below the ~10.5GB a 1024-class sampler actually reserves. A
  whole-job monolithic peak and a disaggregated UNet-only sampler peak are physically different quantities
  and are kept under distinct stages (`SAMPLE` vs `SAMPLE_ISOLATED`), so a single monolithic peak never
  over-prices the isolated sampler and forfeits the second concurrent sampler (mixed operation is designed:
  a stage fault re-routes a disaggregated job monolithic). Monolithic peaks are observed from child memory
  reports; isolated-sampler peaks from the disaggregation orchestrator at sample completion. A cold key
  prices at the static seed unchanged, so a first-of-kind job and small-resolution buckets keep their
  smaller peaks and their concurrency.

    The raise-only overlay is the *under-observed* policy, not the only one. Once a key carries at least
    `_MIN_OBSERVATIONS_FOR_MEASURED` (5) observations, `measured_estimate_mb` answers from the measurements
    alone and may sit well below the seed. The seeds are wrong in both directions: a Flux fp8 checkpoint
    seeded at 16.4GB measures 13.5GB device-used on the card that was reserving itself entirely for it. The
    measured answer is the maximum of a bounded recent window (so a genuine downward shift eventually lands
    while one high job still counts), times a single explicit margin `_MEASURED_ESTIMATE_MARGIN` (1.10), plus
    the platform's context charge. That margin is the whole of the conservatism and it is one constant at one
    seam: an under-estimate is punished asymmetrically (the Linux OOM killer, WDDM paging to host RAM), which
    is why the margin exists, but smearing that fear across the seeds is what produced the over-statement in
    the first place.

    The measurements have two sources. The parent infers peaks from child memory reports as described above,
    and the backend reports a measured per-job footprint (`JobPhaseMetrics.vram_footprint`) that arrives
    already attributed to a model, geometry, batch and execution shape. The second is the stronger evidence
    and needs no inference: its resident figure feeds the `RESIDENT` key for its checkpoint. Its device-wide
    high-water is deliberately not folded into `SAMPLE` or `SAMPLE_ISOLATED`: it carries every sibling's
    resident weights, and an activation key priced from it makes an ordinary preload look like it needs the
    whole card. The activation keys keep the memory-report path as their only source, and admission prices
    them raise-only from the seed; the margined measured estimate is used for the resident footprint alone.
    A backend that reports no footprint (an older one, or a dry run) leaves the memory-report path as the
    only source for every key. The store persists to `.horde_worker_regen/vram_footprints.json` (schema-versioned,
    atomic write, debounced at 10 observations plus a save at shutdown), so a restart keeps its calibration
    instead of re-learning it; a missing or corrupt file starts cold.

    Not every footprint is an activation peak. The same store also carries two at-rest stages under the same
    raise-only contract: `RESIDENT` (a loaded checkpoint's weights, keyed per checkpoint rather than per
    baseline, since two checkpoints of one architecture differ by gigabytes and weights do not move with the
    request size) and `SAFETY` (the safety process's device residency). Both are observed from allocator
    bookkeeping only, as the process's own reservation plus the platform's fixed CUDA-context constant;
    device-view VRAM readings are never folded in, since they report device-wide occupancy on one platform
    and a per-process view on the other. The gates are strict in both cases, because a raise-only watermark
    keeps whatever it is given: a resident observation is taken only from an idle, model-loaded inference
    slot with no tracked job of its own in progress whose residency has been stable for a short settle
    window (a reservation is still in motion for seconds around a model load), and a safety observation only
    from a safety process that is not mid-evaluation, folding its steady reservation rather than its peak,
    because an evaluation's spike is reclaimable and is not what safety costs the card while it waits.

    Both at-rest stages are priced through a single seam each. **Safety** has one price worker-wide,
    `InferenceScheduler._safety_footprint_mb`: the documented `_SAFETY_GPU_LOAD_CHARGE_MB` seed raised by any
    `SAFETY` watermark. The arbiter's `SAFETY_LOAD` charge, both runtime-placement predicates, the streaming
    forecast, the whole-card residency charge-back, and the safety-off reclaim rung all read it, so admission,
    placement, forecasting, and reclaim cannot come to disagree about what the same process costs. The
    streaming forecast charges that whole-process figure rather than counting safety as one more context at
    the per-additional-context marginal: the marginal prices an empty CUDA context, while safety additionally
    holds resident classifier weights, so a marginal-priced safety term understated the card by gigabytes. The
    service lanes, which hold contexts and no learned at-rest figure of their own, keep marginal pricing.

    **Resident** footprints feed the streaming forecast's per-checkpoint weight term. The per-baseline seed
    prices every checkpoint of an architecture alike, so a heavy file is granted sibling room it does not have;
    a measured at-rest figure raises the seed and re-keys the sibling-room and card-dominance judgments on what
    the file actually costs. A family member whose weight set differs from its family's (a quantized DiT, a
    different text encoder) instead carries its own burden entry keyed by model name, which is why the
    worker passes the job's model alongside its baseline when it asks for a seed. Two conversions happen at
    that seam. The store records a whole *device* charge, so
    the context constant is netted back out before the figure reaches a forecast that charges contexts
    separately. And the raise lands on the forecast's full-footprint term only, never on the core-weight term:
    the core weights are deliberately the smaller quantity (support components time-share the card via
    per-phase swaps, which is why load feasibility keys on them), and an at-rest measurement carries no
    attribution of that split. The disaggregated branch ignores resident measurements entirely, for the same
    reason `SAMPLE_ISOLATED` is distinct from `SAMPLE`: its sampler process holds the UNet alone. On a roomy
    card this pricing is expected to stop the heaviest checkpoints from co-residing and to have them claim the
    card instead, which is the honest verdict for a file measured near the card's size; the residency churn
    limiter, not a buffer on the estimate, is what bounds the cost of that flip.

    The measured resident figure additionally carries one authority the raise-only overlay does not: it can
    **retire a whole-card residency claim**. A `wants_whole_card` baseline reserves the device unless the
    forecast finds room for another full model beside it, and that room question is otherwise answered by a
    seed that can over-state the model by gigabytes. Where the checkpoint has been measured enough times,
    `StreamForecast.measured_retires_whole_card_intent` re-asks it of the measurement: if the margined
    measured footprint leaves the card room for one more process context at sole residency, the intent branch
    yields and the ordinary co-residency rules apply. It is a weaker claim than the seed-based floor makes
    (room for a context, not for a whole sibling model), and deliberately so: the premise of tearing the
    siblings down is that the card is full, and the measurement is what shows it is not. The retirement is
    logged once per model, with the figure and the number of runs behind it, through the stream-forecast
    diagnostic. Without measurements nothing changes.
- **Arbitration** evaluates the
  [ledger-driven admission identity][horde_worker_regen.process_management.resources.admission_identity]
  plus the concurrent-sampling headroom, then resolves an actuator escalation ladder. It never overcommit-admits
  a request into a measured over-commit.
- **Actuation** is expressed as
  [`ActuatorCommand`][horde_worker_regen.process_management.resources.vram_arbiter.ActuatorCommand]
  values on the verdict. The arbiter itself executes nothing: it describes what would relieve the pressure
  and a caller that implements the
  [`VramActuator`][horde_worker_regen.process_management.resources.vram_arbiter.VramActuator] surface runs
  those commands. For preload admission the scheduler is that caller, mapping each command onto the worker
  mechanism that already performs it (allocator-cache release, idle-model eviction, live-context reduction,
  safety off-GPU cycling).

## The decision pipeline

For a request the arbiter first evaluates the measured admission identity: committed floor plus planned
overlay plus candidate delta against capacity (device total net of the measured shared baseline, less a
noise buffer). The buffer absorbs measurement noise and the inter-report activation transients a child's
allocator holds before the next memory report reflects them, and scales with device capacity (the greater
of a 512MB floor and 5% of the device total) so a large card keeps proportional headroom while a small card
is never starved below the floor. If that fits, the disposition is `FITS`.

The planned overlay carries each admitted-but-not-yet-materialised preload as an anchor that decays as its
target process's measured reservation grows to cover it. Consumption is monotonic: an anchor is measured
against the greatest growth ever seen for it, so once a preload has materialised, a later eviction that
returns its VRAM to the card cannot resurrect the charge. A materialised anchor never re-charges; only a
genuinely new admission on that process charges again. An anchor whose target process dies or ends before the
load materialises decays by neither route (a dead target's reservation never grows), so the scheduler
excludes ended and missing processes from the in-flight set it reconciles the overlay against: the charge is
then released by omission, the same self-healing path a finished load takes, with no death-path delete to keep
in sync.

A request's own footprint counts at most once in the identity. Two adjustments enforce this so a head can
never wedge on state it alone produced. First, the request nets its own target process's outstanding planned
charge out of the overlay before the inequality: that charge is the same load the candidate delta already
represents, so leaving it in would count the load twice and let a re-ask (whose earlier plan lingered after a
reclaim or a target death) defer forever on its own weight. Only the target process's own charge is removed;
every other process's planned load stays fully charged, so genuinely-concurrent admissions still stack. The
same netting covers the dispatch flow at clearance: a leased job's encode-only staging reservation is still
outstanding when the job is re-priced at its full peak, and that peak already covers it, so the job's own
staging entry is subtracted as well (`own_dispatch_unmaterialized_mb`). Without it a lone staged child on a
card with room for its peak reads as short by exactly the encode charge and waits out its lease-acquire
timeout for room the card had all along. Second,
a candidate whose weights already occupy VRAM on the target process is admitted directly as a no-op: dispatching
(or preloading) onto an already-resident idle model materialises nothing, its weights are already in the
committed floor, and its next activation is the monolithic status quo the card has already served. The ledger
identity cannot express that no-op (the resident model's own reservation can legitimately sit above the
noise-adjusted ceiling, which would otherwise withhold a dispatch that needs no memory), so this is the
whole-card analogue of the disaggregated stage dispatch a resident lane never withholds.

Two request classes are priced against a narrower overlay, because ordering guarantees they commit VRAM
before any staged load can. A preload-flow anchor is a load still staged in system RAM whose VRAM claim only
happens once its dispatch is later re-priced against fresh measured truth; charging it against work that
necessarily precedes that dispatch inverts the dependency into a circular wait that only the recovery
supervisor's soft resets and give-up faults can break.

The first class is the drain side. A post-processing chain (`PP_JOB`) completes a job that has already
sampled: finishing it is what releases the job's holds and frees the room a staged head is waiting on, so a
drain deferring on the head's bookkeeping ages out and faults the finished job. The second class is the true
head of queue itself (`PRELOAD` or `MONOLITHIC_DISPATCH` with `is_head_of_queue`): every other staged load
sits behind the head, and a staged sibling's materialisation is gated on its own dispatch admission, which
cannot precede the head's. Charging the head a queued sibling's staged plan parks the head on room that can
only ever be claimed after the head itself runs; with two staged loads the standoff is mutual and the queue
wedges outright. Both classes are therefore priced against physical truth plus the dispatch-flow
reservations only (in-flight sampling genuinely about to spike); the requester's own preload-flow charge is
part of the excluded share, so the per-target self-netting does not stack on top of it. A non-head request
(a line-skip) normally stays fully charged, so it cannot consume the room a staged head is waiting on. The
one bounded exception is a line-skip past a `PREPARE_AUX_MODELS` head that is still pending rather than
in-progress: auxiliary completion returns that child to `WAITING_FOR_JOB`, so the head cannot materialise
until it re-enters this arbiter. The skipper may use measured room now and records its own dispatch
reservation; that reservation then holds the prepared head if their sampling peaks cannot coexist. An
in-progress/legacy auxiliary download receives no exception because it can flow directly into sampling. This is the
same reasoning that admits the disaggregated decode stage unconditionally: work whose turn has come must not
be starved by claims whose own turn comes after it.

When the prepared head re-enters dispatch while another job is in progress on its card, retained base weights
do not make the request a complete no-op. The scheduler credits those resident weights but reprices the head's
activation-only delta against the other job's live reservation. Ordinary resident dispatches retain their
no-materialisation fast path; this narrower rule prevents auxiliary preparation from hiding a second peak
behind resident-weight credit.

Before reclaim is consulted, a non-fitting request is checked against the **phantom-ledger** judgement. The
committed floor is bookkeeping, and the worker cannot hold more device VRAM than the device itself reports
used: when committed exceeds the truthful device-used reading beyond a tolerance
([`committed_ledger_is_phantom`][horde_worker_regen.process_management.resources.vram_attribution.committed_ledger_is_phantom],
the same predicate the drift reconciler keys its recalibration on), the rejection is arithmetically
impossible for a truthful ledger and the over-count is fiction. Handing that rejection to the reclaim ladder
would spend destructive actuation (model eviction, context teardown, whole-card residency) on memory the
device never held, so instead the head of queue is re-priced against the truthful device-free reading:
candidate plus the (self-netted) planned overlay against device-free minus the noise buffer. When it
physically fits, the verdict is a `FITS` flagged `phantom_truth_admit` and counted in
`phantom_truth_admissions`; nothing is marked over-budget, because the card genuinely has the room. The
bypass keeps the same head-of-queue priority rule as the foreign-pressure admit, and the device-free
governor outranks it: while a SATURATED card's verified ladder is still working, even a phantom-rejected
head keeps deferring, because SATURATED is itself a device-level truth. While the phantom holds, the
escalation ladder below describes only its cache-release rungs (the recalibration actuation) and the
starved-head context teardown is suppressed: destructive reclaim under a lying ledger is how a free card
gets torn down. The reconciler's recalibration (asking idle lanes to release their allocator cache and
re-report) runs on its own cadence to converge the ledger back to truth.

If it does not fit, the arbiter next asks whether reclaim can still make progress. Reclaim can still make
progress when the arbiter's own ladder emits a command, or when the device-free governor is SATURATED and
its verified ladder has not proven the card unresolved. In that state the disposition is `DEFER`: for
preload the scheduler runs the described actuation and the request re-asks next cycle once the device-level
verification has either shown reclaimed memory or advanced to the next rung.

Once reclaim is exhausted and the demand still does not fit, the verdict depends on the shortfall. If the
worker's own committed load plus the (self-netted) planned overlay still exceeds capacity, live worker work is
holding the card and the head stays queued until a slot drains. Because the request's own footprint is netted
out first, this branch can only be reached by load that is genuinely other than the request itself (a live
sibling holding the card): it can never be composed from the request's own resident weights, its own lingering
plan, or its own candidate delta, which is exactly the self-deadlock the netting closes. If the worker's own
committed load fits capacity but the candidate tips the inequality over, the shortfall is foreign pressure. Foreign pressure admits only when the
candidate physically fits the truthful device-free read minus the noise buffer at that moment, and only for
the true head of queue. That is the remaining useful "best effort" case: fitting into measured reality, not
hoping an over-commit will work. A non-head request (a line-skip job selected ahead of a downloading head) is
denied that admit even when the card physically has room right now, because materialising into it can starve a
head already authorized to proceed from download into sampling. A preparation-only pending head is not yet
authorized, so its bounded skipper omits `head_outstanding_mb` and both jobs take their turns through ordinary
measured admission. If the candidate does not physically fit (or
the requester is not the head), the disposition is `DEFER` and the `admission_foreign_pressure_defers` counter
advances. The dispatch-reconciliation gate plumbs the same truth, presenting `is_head_of_queue=False` for a
line-skip dispatch and retaining head protection except for that preparation-only pending state.

Head protection is bounded. Reserving physical room for a head is only worth its cost while the head is
converging on a dispatch; a head whose own admission keeps declining otherwise holds an idle card against
siblings that measurably fit, for as long as the queue lasts, and the worker serves nothing. Once the head has
been parked past its protection window without dispatching, the gate stops presenting its outstanding demand, a
fitting sibling is admitted, and the release is disclosed and recorded as a dispatch decision. The head keeps
its queue position and first claim on the next opportunity.

Protection is released the same way, and immediately, while a churn governor is deferring that head's
whole-card establishment on the target card (see
[Bounding residency churn](resource_governance.md#bounding-residency-churn)). For the length of that deferral
the head is deliberately not asking for the card: the governor's own disclosure says normal scheduling
continues around it. Charging its whole-card demand against the ready work behind it would reserve the card
for a claim nobody is making, so the card stands empty while smaller jobs that measurably fit are turned
away. The pricing seam reads the deferral from the whole-card ledger and omits the head's demand for as long
as it stands; once the governor releases the card, or the deferral's dwell is spent and the head is served
co-resident by ordinary admission, the head's normal charge applies again.

The "foreign" label is earned, not assumed. Before a non-fitting head is charged to foreign pressure, the
arbiter separates a shortfall the worker can itself reclaim: a head whose deficit is held by its own idle
sibling contexts (a bare CUDA context whose VRAM returns only when the process exits, so no model-unload or
cache-release rung reclaims it), with no physically-available VRAM to admit into, is deferred as reclaimable
first-party residency and advances `first_party_context_defers`, not the foreign counter, and emits no reroute
diagnostic. Reclaim is not exhausted while its own context teardown is still pending; the head simply waits out
the short teardown grace for those contexts to age into the verified teardown below. This is what keeps the
worker's own idle contexts from being mistaken for unreachable desktop load and rerouted to the recovery
supervisor while surgical room sat one teardown away.

There is no starved-head overcommit admit, but a starved head does escalate reclaim, and it does so on
evidence rather than a long clock. Two timings apply, deliberately different:

* **First-party context teardown fires after a short grace (`_FIRST_PARTY_TEARDOWN_GRACE_SECONDS`, 10s).** When
  weight eviction is exhausted, no physically-available VRAM exists to admit into, and the head's remaining
  deficit is exactly its own idle sibling contexts, no alternative remedy can ever arrive: evicting a model or
  releasing a cache frees nothing a bare context holds, and a busy sibling finishing does not surrender its
  context. Waiting longer is pure idle-card loss, so the arbiter escalates quickly. The grace exists only to
  ride out transient state churn and measurement noise (a sibling about to pick up work, a snapshot mid
  reconciliation), not to wait for a remedy that cannot come. The escalation defers with a
  `REDUCE_LIVE_CONTEXTS` actuation that reduces the live inference-context count (protecting the head's own
  target slot and every busy process) and advances `starvation_context_teardowns`. The freed room is verified
  at device level before the head is admitted, so the escalation never force-admits; it only makes room the
  re-ask can then fit. Because the trigger is short, a re-ask arrives every scheduler cycle; that is safe
  because the teardown scales to a fixed target and retires its victims from the process map synchronously, so
  a repeated command sees the count already at target and tears nothing more down (the actuation only scales
  while the live count exceeds the target, and books one restore obligation however many times the head asks).
  Both the preload and
  `MONOLITHIC_DISPATCH` paths may use this escape only when the candidate peak computes a maximum
  resident-process target below the current live pool. Merely finding an idle sibling does not qualify: a
  target at or above the live count proves pruning that context cannot address the deficit, so the request
  stays out of whole-card residency. The actuator repeats that check against the current live count before it
  stops anything, because a correctly-issued command can become stale after an earlier scheduler tick has
  already retired its victims.
* **The genuinely-foreign starvation diagnostic keeps its 60s threshold (`_STARVATION_DIAGNOSTIC_SECONDS`).** A
  head whose shortfall is real foreign load with no first-party context reclaim has no surgical remedy the
  arbiter can apply, so its long-wait warning stays at 60s (see below); shortening it would only spam the log.

Thrash between distinct large models (or large/small alternation) is not damped by lengthening the escalation
timer, which would just cost idle-card time. It is damped on the pop side by `large_model_switch_min_seconds`
and the large-model re-entry cooldown, which stop the worker offering a churning sequence of heavy models in
the first place.

**The `whole_card_exclusive_residency` flag governs steady-state preference, never this emergency liveness.**
That config flag decides whether the worker proactively establishes exclusive whole-card residency as a matter
of course (the pre-staging and forecast-driven teardown described under whole-card residency). It does **not**
gate the starvation escalation. A weight-dominant head starved behind its own idle sibling contexts must reach
the verified teardown even with the flag off, because the alternative is the catastrophic save-our-ship pool
reset resolving a situation the arbiter could relieve surgically. The scheduler therefore reports
`idle_contexts_teardownable` on this seam independent of the flag, and the actuation (stop the idle contexts
down to the head's target, then evict the idle siblings' VRAM) runs through machinery that does not itself
consult the flag, so the contexts are torn down and the head admits regardless of the steady-state preference.

**The reduction is an actuation, not a residency grant.** `REDUCE_LIVE_CONTEXTS` removes inference contexts
and does nothing else. Reserving the worker for the head, stamping a residency lease and its cooldown, moving
safety off the GPU, pausing the service lanes, and opening the establish grace window that tells the recovery
supervisor a held queue is intentional are commitments of a *whole-card residency*. They belong to the grant
that asks for one, which may itself request a reduction through this same surface; taking them on the rung
would impose the whole policy on an operator who declined it, since emergency reclaim is not gated on a
steady-state preference. The pool the reduction shrank is booked with the verified reclaim ladder as a restore
obligation and regrown when that card recovers, so a reduction is not a permanent capacity loss (see
[The verified LIFO reclaim ladder](performance_and_backpressure.md#the-verified-lifo-reclaim-ladder)).

A head deferred past the 60s diagnostic threshold with reclaim genuinely exhausted and no such teardown target
(no first-party context reclaim remains) emits a warning with the full arithmetic and increments
`starvation_diagnostics`; it still does not admit. The job stays queued for the structural queue wedge recovery
supervisor, which detects a stuck queue with no dispatch progress, soft-resets the pools, and then faults
wedged jobs non-retryably so the horde can reissue them elsewhere.

`DENY` is reserved for a candidate that could not fit even an emptied card: one whose demand exceeds the
card's **achievable ceiling**, the card total minus the noise buffer minus the sustained foreign floor (the
VRAM the OS/desktop/other processes hold, measured outside the arbiter as the trailing-window minimum of
`total - device-free - worker-committed` and carried on the device state; an unknown floor collapses the
ceiling to `total - noise`, preserving the pre-foreign boundary). Model-level prevention keeps most such jobs
out earlier, and when a runtime `DENY` does fire for the true head with no other card that could seat it, the
scheduler faults the head for reissue and places the model on a conditional, self-lifting ceiling hold
(re-checked against the current ceiling, lifted once the foreign floor recedes) rather than deferring into a
permanent wedge (see [the achievable ceiling in Performance and Backpressure](performance_and_backpressure.md#the-achievable-ceiling-a-model-that-can-never-fit-this-card)).
The first concurrent sampling of its kind admits on an empty ledger.

Between a plain `DEFER` and a structural `DENY` sits one bounded escape hatch: the **measured-load attempt**. A
candidate under the achievable ceiling is possible on this card in principle, so a static sampling-VRAM
prediction that misses the instantaneous available reading may simply be conservative, or the foreign VRAM may
have breathed up for one reading. When the true head has starved past the 60s diagnostic horizon at a
*converged-empty* card (no worker reservations outstanding, no worker-resident model to evict and no idle
sibling context to tear down: make-room has nothing left to reclaim), the arbiter stops deferring an idle card
on arithmetic that may be wrong and admits the head for exactly one real load, so measured reality decides. The
attempt rides the ordinary `FITS` path, so every downstream safety (the per-step floors, the watchdogs, the OOM
classification, the whole-card residency machinery) applies unchanged. It is one-shot per accepted job and
card. The job tracker owns the spent-card receipt, so replacing the arbiter or inference process cannot re-arm
the same exceptional attempt; an active preload-to-dispatch continuation remains admitted because it is still
the same attempt. A success teaches the learned peak the true figure and arms no hold. An out-of-memory failure
ends the continuation and immediately arms that card's conditional ceiling hold. The ordinary inference retry
cap may offer the job to a different eligible card, but the failure never earns a degraded retry on the same
card: the card was already converged and alone. When the normal cap is spent, the job faults through the usual
terminal path.

For work the worker has not accepted, the shortfall's *size* does not gate an under-ceiling attempt, and the
over-ceiling escape stays limited to the prediction-error allowance. Accepted RAM-staged work adds one case:
when the checkpoint weights themselves fit the achievable ceiling, an activation-inclusive static prediction
above that ceiling cannot become an immutable refusal after intake. That job/card pair receives the same single
measured attempt, still subject to the true-head, converged-empty, and no-better-card guards.

A band expresses how far the arithmetic
may be wrong before waiting beats trying, and waiting is only a bet on a card whose available reading can still
improve. On a converged card it cannot: the choice is exactly two-way, take the load or hold the head against a
figure that will never move, so refusing a wider shortfall parks the head with nothing to wait for. What bounds
the demand there is the achievable ceiling, not a band. The 1024 MB uncertainty figure still sizes the
*ceiling* allowance below.

A card that has **not** converged is different: something the worker can free is still outstanding, so the
reading can improve and deferring has a purpose. There the head defers and the reclaim is routed. That also
decides what happens to a candidate *over* the ceiling by less than the ceiling allowance (10% of the ceiling,
capped at 1024 MB): it is granted its one real load only from a converged card, and while the card still holds
reclaimable state it takes a `DEFER` that drives the teardown rather than the terminal `DENY`. Denying it
earlier would settle servability from a pool shape that is still changing: the identical head on a pool some
other mechanism happened to collapse first would be served, while this one is faulted for reissue and its model
put on the ceiling hold. The terminal verdict stays reachable, but only from a card that has actually run out
of things to free.

Disaggregated sampling is priced differently: the static concurrent-sampling headroom now lives on the
device state alone (device total net of baseline, minus the fixed and marginal context overheads, the
operator reserve, and the image lane's bounded decode spike), and a later sampling admits when the summed
in-flight sampling peaks plus this one fit that headroom. The in-flight total is taken live from the request
so a peak booked earlier in the same tick is counted before the cycle snapshot is next refrozen. Charging
the lane's bounded decode spike rather than its full allocator-guard quota is what lets two samplers
co-reside on a card that holds them; the full-quota charge collapses the pipeline to one sampler.

## Admission authority

The arbiter is the deciding authority at every device-memory admission seam: model preloads,
monolithic-dispatch overlap, the disaggregated concurrent-sampling gate, the disaggregated encode and decode
stages, post-processing chains, and safety GPU loads. Each proceeds only on a `FITS` verdict. Cross-job
weight retention is not an arbiter seam: it neither adds new bytes to the card nor consults the arbiter, and
is governed instead by the device-free governor and the verified reclaim ladder (see below).

**Preloads.** The scheduler's preload adapter consults the whole-card residency state machine first (which
stays external, pre-staging or deferring a whole-card head), then prices the preload through the arbiter and
acts on the single verdict: a `FITS` admits and runs the marginal RAM verdict, and a `DEFER` runs the
described actuations and re-asks. There is no second, parallel admission arithmetic: the
ledger-driven identity is the only gate. Because the reserve is a sampling-headroom term and never a
load-feasibility floor, a preload is never denied by `vram_reserve_mb`; a model whose weights fit the drained
card admits even when the operator's reserve would have read it as unloadable. Under the clearance lease a
preload only stages the job in system RAM (the diffusion weights load inside the leased sample call), so its
candidate charge is capped at the staging encode footprint, the same figure a staged dispatch books, and the
full fit-or-evict runs at clearance; without the lease the preload is the VRAM moment and is priced at the
job's full marginal sampling charge. Pricing a lease-staged preload at the full peak parked the next model's
disk read behind the running sample on every model switch, which is the work the stage exists to overlap.

**Overlap.** The scheduler's overlap adapter runs its non-memory guards first (the whole-card tier's
no-co-sampling contract, and the size-scaled sampling headway that keeps a newcomer off a running job's
startup beat), then lets the arbiter decide the memory question through a `MONOLITHIC_DISPATCH` verdict: a
`FITS` admits the overlap, a `DEFER` or `DENY` withholds it for the cycle. The headway relaxation
fires only on positive confirmation of room (a cycle that admits), so a cold start keeps the strict headway
fractions rather than reading the admit-on-missing-telemetry relaxation as evidence of room.

**Dispatch reconciliation.** The overlap gate reasons only about jobs already sampling; it says nothing about
an idle sibling whose weights are still resident from a prior job. Yet the instant a RAM-staged job is handed
to its child, its weights and first activation commit to VRAM, and that materialisation lands on top of any
idle resident. Neither the preload nor the second-sampler seam prices that moment, so a dispatch is the last
uncrossed admission point. This gate prices only a genuine materialisation: a dispatch whose model is already
resident in VRAM on its target moves nothing, so it is released as a no-op (the identity's
`candidate_already_resident` admit) rather than priced against a card its resident weights already legitimately
overshoot. The scheduler's dispatch adapter closes the remaining seam with the same `MONOLITHIC_DISPATCH`
identity: before a staged (not yet VRAM-resident) job is dispatched, it prices the job's expected
materialisation against the card
(the learned per-signature peak the admission overlay already uses, against the truthful device-free reading
net of the proportional buffer that the identity's foreign-pressure branch enforces). A `FITS` releases the
dispatch. A conflict holds it: the job keeps its head-of-queue position and is never faulted, the idle
residents that tip the card over are evicted through the one reclaim owner (the head's own target slot is
protected, so its staged weights are spared), and the dispatch re-asks each pass, releasing only once the
arbiter next verdicts `FITS` on the governor's verified device-free reading. Can't-fit-ever jobs are already
excluded by model serviceability, so this gate only ever holds a can't-fit-now dispatch.

The dispatch head has the same two starved-head escapes the preload head does, so it can never wedge on a
ledger fiction while its own idle sibling contexts hold the card. The dispatch candidate is an
activation-inclusive learned high-watermark peak, so it already carries its own headroom; stacking the full
admission noise buffer on top of it prices demonstrated-fine dispatches out of existence on a small card.
Past the starved-head grace, a dispatch head that physically fits the truthful device-free reading net of the
governor's **hard floor** (`hard_floor_mb`, the band the governor actually defends, not the larger noise
buffer) admits into reality and advances `dispatch_reality_admits`. Only when even that hard-floor reading has
no room, and the deficit is held by the head's own bare idle sibling contexts (a context weight eviction
cannot reclaim), does the dispatch head escalate to the same verified `REDUCE_LIVE_CONTEXTS` teardown the
preload seam uses, after the same grace. An ordinary (un-starved) dispatch still never collapses the pool: the
reality admit and the teardown are the starved head's alone, and the reality admit is tried first so no
teardown happens when physically-available room already exists.

**Pinned-lane residency.** A disaggregated job's sampler lane is pinned (reserved out of the availability
pool) from the moment it is scheduled until its sampling finishes, and while pinned it is excluded from the
dispatch selection so no job is ever dispatched onto it. A monolithic head whose model is resident *only* on
such a pinned lane must therefore not read as not-resident and fund a fresh second copy that cannot fit beside
the pinned residents. Residency and pricing queries include pinned lanes (the dispatch query still excludes
them), so the head is priced as already resident and held for the pin to release rather than preloaded afresh;
when the pin releases, the lane returns to the availability pool and the head dispatches onto that resident
copy, priced through the `candidate_already_resident` no-op admit. The dispatch-stall classifier names this
wait (the pin, the disaggregated job holding it, and the in-flight sampling peaks) rather than reporting a
generic budget defer.

A held dispatch is not mistaken for a wedge. The job stays queued with its model resident and never enters
in-progress, so the clocks that time the preloaded-to-inference-started transition have nothing to reap: the
stale-entry expiry only touches a `LOADING` entry (not a resident one), the resident-cleanup spares any model
a pending job still wants, and the lost-result reap and orphaned-in-progress reconciler act only on a job that
actually ran. The deadlock detector does see an all-idle queue whose head model is resident as a queue
deadlock, but only a queue deadlock sustained past the structural-wedge horizon reaches the recovery
supervisor: a transient hold (reclaim frees the idle resident within a few ticks) clears far below it, while a
hold that genuinely never clears (foreign pressure, reclaim exhausted) is exactly the case the recovery
supervisor exists to reroute, identical to a never-admittable preload.

**Source-latent routing.** Whether a disaggregated job enters at the source-latent encode stage or straight
at conditioning derives from the SDK's effective (post-fallback) source processing, never the raw pop field. A
source-requiring mode (img2img, inpainting/outpainting, remix) whose source image is unusable resolves to
txt2img and enters at conditioning, so a job the converter runs as txt2img is never routed through a
source-latent encode of a placeholder image. The same effective mode governs disaggregation class-eligibility,
so a mislabeled job is eligible as the txt2img job it actually runs. The resolution is the SDK's single
authority (`horde_sdk.worker.dispatch.ai_horde.image.source_image`), shared with the image parameter
converter, so the routing decision and the executed generation cannot disagree.

**Disaggregated sampling.** The orchestrator's concurrent-sampling gate admits a first-of-kind sampling on an
empty ledger, then defers to the arbiter's `DISAGG_SAMPLE` verdict for every later sampling. It passes the
live in-flight sampling total with the request so a peak booked earlier in the same tick is counted before
the cycle snapshot is next refrozen. The gate may serialise samplers but must never deadlock: a deferral is
healthy backpressure only while a sampling is verifiably in flight (a ledger entry whose owner is still
sampling, whose sample was dispatched to a live process launch, and whose process reports busy on the device).
When no sampling is verifiably in flight, the deferral escalates within a tick, not at the sanity bound: the
provably-stale peaks (owner gone, dispatch launch dead, or an idle sampler whose result was lost past a short
grace) are cleared so the sample re-admits, because a candidate that fits alone on an idle card must always
run. A far larger sanity bound is the last resort for a ledger that looks live yet yields no system-wide
sampling progress for its whole window. The one protection never relaxed is the second-concurrent-sampler
memory check itself: two peaks that do not co-fit are never admitted together.

**Disaggregated encode and decode.** A stage dispatch targets a process already resident on the card, so it
is not a new admission and is never withheld. The concurrent-sampling gate downstream is the pipeline's real
admission point: an encode only leads to sampling if that gate admits the job, so gating the stages adds no
admission control, and any stage gate serialises the stage overlap the disaggregated pipeline exists for
(during 1024-class sampling the committed floor legitimately exceeds the admission ceiling, so an
identity-shaped stage gate would defer every encode for the whole sampling duration, and a distress-shaped
one freezes finished work behind transient paging blips). Decode in particular drains the pipeline:
completing it releases the job's sampler hold, latents, and submit path, which is precisely how memory
pressure ends; the image lane's tiled decode and its allocation self-heal bound the transient spike. Decode
returns raw images: it never runs post-processing, so the VAE lane is never blocked on upscale/face-fix work,
and the decode gate prices only the tiled-decode activation spike. A disaggregated job that requested
post-processing routes to the dedicated post-processing lane after decode, on the identical path a monolithic
completion takes (see **Post-processing** below); the disaggregation flag forces that lane on. The
resource-defer window and monolithic re-route remain reserved for genuine resource-class stage faults
reported by a child process, never for a parent-side verdict.

**Post-processing.** The lane's memory admission is the arbiter's `PP_JOB` verdict (replacing the banned
free-VRAM read): a `FITS` admits, a `DEFER` or `DENY` holds the chain and the lane's own deferral
bookkeeping (each newly available reclaim plan at most once, throttled warning, patience age-out) remains
bounded. The orchestrator retains
the verdict rather than reducing it to a boolean, and executes its reclaim commands through the same shared
reclaim owner as preload and dispatch admission. For a post-processing head this plan may move safety off-GPU
after idle cache/weight reclaim when `whole_card_residency_safety_off_gpu` permits it. The reserve bypass is
preserved: a disabled VRAM budget or a zero-peak chain always admits. The lane's non-memory guards (the
allocator-guard cap fault and sampling co-residency hold) stay.

**Safety GPU load.** The recurring safety-on-GPU seam is gated on the arbiter's `SAFETY_LOAD` verdict, charging
the learned safety-context footprint. One per-tick placement reconciler owns that seam for every initiator:
runtime fit policy, whole-card residency, and verified reclaim contribute wishes or vetoes rather than calling
lifecycle independently. A `DEFER` keeps safety off-GPU this cycle and the reconciler re-asks, so a deferred
load is not stranded after the card has room. The initial cold-start safety load onto the GPU (at worker
bring-up, before any heavy residency pressure) is not gated and always proceeds.

### Runtime safety placement

The single safety process (slot 0) runs on-GPU only where a driven card's effective `safety_on_gpu` permits
it. The flag is a per-card permission to host rather than a worker-wide switch: the placement chooser
considers only permitted cards, and with none permitting it safety runs off-GPU exactly as a globally
disabled flag makes it. Withdrawing the permission from the card safety currently occupies goes through the
same pause actuator every other placement request uses, so there is one path that ends an on-GPU safety
process. On a card too tight to
hold safety's context beside the model that is sampling on it, that CUDA context competes for VRAM the sampler
needs. The scheduler-owned **runtime safety-placement policy**
([`_reconcile_runtime_safety_placement`][horde_worker_regen.process_management.scheduling.inference_scheduler.InferenceScheduler])
generalises the whole-card safety-off lever to that ordinary case: it moves safety to a CPU-only process when
safety's own card is really short of the memory its work needs, and re-promotes it once that card proves durable
room. The per-card permission remains the operator's maximum grant; the policy only degrades GPU to CPU and
back, never beyond it.

**Every term the policy reads is about safety's own card** (the card it occupies, or the card it would land on
while it is off). Cards are independent VRAM domains, so the peak a sibling card is committed to says nothing
about this one: the peak is taken over the jobs running on this card plus the queued jobs its effective config
can serve, and on a single-GPU host that is the same set the worker-wide figure would give. The `per-card safety
placement` scenario in `tests/process_management/liveness/test_incident_scenarios.py` holds this over two
independent card ledgers: a card serving its own light class keeps its safety process while its sibling carries a
peak it can never be given, and pricing that peak worker-wide again evicts safety from the serving card for the
rest of the run.

GPU placement treats CLIP plus its CUDA context as the lane's fixed residency, not the last evaluation's
high-water mark. DeepDanbooru stays in host RAM until an anime check calls it; BLIP and the aesthetic head are
offloaded after use; and transient allocator blocks are cleared before the child reports itself idle. The
device-free reading therefore sees those allocations as reclaimable work, rather than evidence that the whole
safety process must be replaced or that an ordinary SDXL-class dispatch needs exclusive residency.

Both sides forecast against the **marginal need**, not the whole peak. The process that will sample a peak
already holds its weights and its context, and those bytes read as *used* in the card's measured free rather
than as room the peak needs again, so the figure the policy compares against is
`max(0, per-card peak - that process's committed device memory)`. Charging the whole peak instead is what makes
a fit unsatisfiable on a small card for the entire session: on an 8GB card with a 4.6GB peak,
`total - peak - noise - safety` is negative by a few tens of megabytes, a margin far inside the noise buffer
and one no amount of good behaviour can close.

**Demotion** therefore needs measured evidence about that card: the device-free governor has left `HEALTHY`
there, or its measured free no longer covers the marginal need plus the proportional noise buffer. A card
committed to no peak, or one whose residents already cover every peak it is committed to, is not pressured
however little it reports free: a card full of weights is admission's and reclaim's subject. The **modeled**
non-fit (`total - peak - noise - safety_footprint < 0`) is a forecast input and never a trigger on its own.

**Re-promotion** is the mirror-image forecast: the chosen card's measured device-free (the governor's truthful
NVML-derived figure) must cover safety's whole footprint *plus* the marginal need plus the buffer, with the
governor `HEALTHY`. Safety has to survive the peak the card is already committed to, or one restore simply buys
the next eviction. On a box where no card can host safety beside its sampler (two small cards, a large model on
each) that forecast never holds, and **CPU safety is the correct steady state**, with the post-inference
backpressure above bounding intake to CPU-safety throughput.

Both sides are **dwelt in seconds**, never in control cycles. A flip ends the safety process and brings a
replacement up, so it costs the worker that whole rebuild in safety unavailability; the dwell is that measured
cost ([`safety_readiness_latency_seconds`][horde_worker_regen.process_management.lifecycle.process_lifecycle.ProcessLifecycleManager],
floored for a cold start), and the restore dwell is a multiple of it (`_SAFETY_PLACEMENT_RESTORE_DWELL_FACTOR`):
leave a card that is genuinely short promptly, come back only once its room has proven durable. Counting control
cycles instead prices the flip at a fraction of a second, which is how a respawn window comes to decide the flip
that follows it.

**Evidence is frozen across an intentional rebuild** and restarts from scratch after it, so a placement change
cannot be decided by its own actuation window. Only one clock runs at a time: a resident safety process accrues
pressure, an evicted one accrues forecast headroom, and each resets the moment its condition stops holding, so
no off-GPU intent can outlive the evidence that produced it. A pending safety backlog resets the pressure clock
(evicting safety while the worker is behind on safety checks stalls exactly the stage it is behind on), and so
does a whole-card residency on safety's card, since filling that card is what the residency *is*.

Re-promotion is additionally withheld while a whole-card residency still needs safety off its card, accepted
post-processing needs the drain window, or the device-free governor is holding growth. Residency and reclaim
bypass the demotion dwell when requesting a pause, but they still use this same reconciler and cannot start a
second placement rebuild before the current one reaches readiness. A **reclaim-ladder** pause earns its restore
with the same forecast dwell (the memory that pause returned is part of what the instantaneous gates then read).
A **whole-card residency** pause does not: it ends when its own model drains, and holding it to a memory
forecast would leave a card that hosts one heavy resident without an on-GPU safety process for the session.

The verified reclaim ladder uses the same operator permission as whole-card safety movement: if
`whole_card_residency_safety_off_gpu` is false, safety is not added as a reclaim rung even when it is on GPU.

**Placement is headroom-aware across cards, not a fixed device 0.** One identity
([`_choose_safety_gpu_card`][horde_worker_regen.process_management.scheduling.inference_scheduler.InferenceScheduler])
picks the driven card with the most verified headroom (measured device-free when reported, else card total less
the peak that card is committed to) and is pushed to the lifecycle manager **only while a spawn could use it**,
that is while safety is off-GPU or not yet placed. While safety is resident the desired card is the card it is
pinned to, so a crash rebuild or a residency restore puts it back where it was rather than wherever read
roomiest that cycle. Demotion, promotion, and the current placement card (or `None` for CPU) are surfaced in the
run metrics.

Reclaim stays single-owner across three seams: preload, dispatch reconciliation, and post-processing all run a
`DEFER` verdict's actuations through the one reclaim engine (`execute_arbiter_commands`), which the governor's
SATURATED verified ladder shares. Every other authoritative seam (overlap, disaggregated sampling, safety)
simply withholds the demand and re-asks next cycle, so no second mechanism evicts the same card by different
rules.

### Cache reclaim is on-demand only

A stage process's retained allocator pool (the ~4-5GB a sampler or the image lane holds between slices) is
deliberately left in place while nothing competes for the card: releasing it costs a collection pause plus a
full pool rebuild on the process's next slice, which is paid on every job, while the reservation it returns
is only worth anything when another demand actually needs the memory. The arbiter's escalation ladder is
therefore the sole reclaim path: a deferred demand emits a release command targeting a specific idle
process's cache, the preload adapter executes it, and the freed reservation shows up in the next cycle's
measurement.

## Doomed model prevention

Some models cannot ever run on a card because their minimum footprint exceeds the card's usable capacity
before a child process starts work. The worker checks that arithmetically at the model offering seam and again
before preload or dispatch in case a stale offer returns a job anyway:

```text
resident weights + minimum 512x512 batch-1 activation <= device total - shared baseline - admission noise
```

The resident weight and minimum activation figures come from the same torch-free hordelib burden seeds used
by the scheduler. The shared baseline is the VRAM the reconciler attributes to the OS, desktop, and foreign
apps; if it has not been captured yet the offer filter reads it as zero rather than inventing pressure. The
noise term is the same proportional admission buffer used by runtime admission.

At pop time, a model is excluded only when every card in the current offer scope that serves it fails that
inequality. A heterogeneous offer scope contains one card; equivalent cards may share a combined scope and
keep a model if any serving card can host it. Queue imbalance can prioritize an under-fed card, so the same
arithmetic applies to the card being advertised. Each excluded model logs one INFO line naming the arithmetic
per card.

If a doomed job still arrives because the horde answered an older offer or the reference changed, the
scheduler faults it before any `PRELOAD_MODEL` or inference control message is sent. The fault is
non-retryable and carries the arithmetic in the job diagnostic, so the submit path reports a no-image fault
and the horde can route the request to a larger worker.

## Runtime placement policies

Measured admission decides whether a *new* demand may join the card. It cannot, by construction, prevent the
overflow a single admitted job causes on its own: once a job is sampling, its activation peak is already on the
card, and on a WDDM host the driver answers the resulting over-commit by streaming weights to host RAM rather
than failing, so the job runs several times slower instead of erroring. The winning regime on a tight card is
therefore one healthy sampler at a time with every reclaimable context off the card beside it. Two scheduler
policies enforce that placement each control cycle, both as arithmetic over `(device total, learned or seeded
footprints, job resolution and batch)` with no constant tuned to a particular card size.

**Safety placement as arithmetic.** The operator's `safety_on_gpu` is a *maximum* permission: `False` keeps the
safety process off the GPU forever. When it is `True`, a runtime policy may still degrade the placement from GPU
to CPU (never the reverse) whenever the safety context cannot fit beside the largest sampling peak the device is
committed to. The fit is structural: `total - largest_learned_sampling_peak - proportional_noise_buffer -
safety_charge >= 0`, where the largest peak is taken across the in-progress and queued jobs (each job's static
seed raised by any learned `SAMPLE`-stage watermark, so the policy prices from measured high-waters rather than
a seed the hardware has already overshot) and the noise buffer scales with the device total. The decision is
hysteresis-gated: safety moves off after a short run of consecutive non-fitting cycles and is readmitted only
after a longer run of cycles that fit *with an added proportional margin*, so a card oscillating around the fit
boundary does not flap the safety process on and off the GPU every cycle. This generalises the whole-card
safety-off lever (which stops safety only while a genuinely-heavy model holds the whole card) to the ordinary
tight-card case; the two share the single pause/restore machinery, and the placement latch withholds *both*
residency-side restores (the drain reconciler and the residency-end restore) so the two controllers never
fight over the safety process's placement. A residency that ends while the latch holds safety off leaves it
off; the placement policy's own re-promotion is the single path back on-GPU, which is what keeps one ending
residency from costing two full safety process rebuilds.

**Lane yield parity.** The disaggregated pipeline's component (text-encode) lane, like its VAE lane and the
post-processing lane, holds a permanent CUDA context plus resident weights that a sibling teardown cannot
reclaim. On a card too tight to host a whole-card model beside that context, the lane must vacate the card
exactly as safety does. Each of these lanes is therefore stopped wholesale (context and models freed) when a
whole-card residency claims its card, whether at establishment or through the convergence loop a pre-staged
head claims the card with, is a member of the residency's teardown-complete gate (the heavy model is not
admitted until the lane's process has actually exited, not merely been asked to), and is restarted once the
residency drains. Every pause the gate waits on is ordered on both paths: a leg nothing acts on never passes,
and the gate's drain backstop only starts once every structural leg has. Stopping the component lane also drops it from the disaggregation liveness
predicate, so while it is down new jobs route through the monolithic path rather than dispatching encodes into a
card reserved for the heavy model; the demotion is automatic and a job never faults for the paused lane.

Staleness drops only the measured committed floor (child telemetry), never the planned overlay: that
overlay is the parent's own admission ledger and needs no child report, so it always counts. A stale
ledger with a known total therefore still tests `planned + candidate` against capacity and can deny a
stacked-admission over-commit even before the first child report, while staleness alone (no planned
demand) never denies. Only a cold start with no known total relaxes every verdict fully to admit, since
with no capacity nothing is knowable; the caller then falls back to its predictive path.

## Cross-job retention

hordelib evicts a job's model from VRAM after every run. That eviction forces a RAM->VRAM weight
re-transfer on the next job, the dominant non-sampling cost on small jobs: even a same-model successor on
the same process re-uploads weights that were still on the card. Retention suppresses that eviction for one
dispatch (it sets the child's `defer_vram_unload` flag), and the child then reports the model still
`LOADED_IN_VRAM` so the parent's model map keeps its residency and the next same-model job skips the
re-transfer.

Retention is not routed through the arbiter, because holding already-materialized weights adds no new bytes
to the card. It is instead a governed live gate that grants only when:

- **Retention can actuate at all.** The `legacy_comfy_vram_unload` escape hatch restores the flag regime in
  which the child's executor returns the card at the end of every prompt, below anything a grant reaches.
  Every grant is denied there, so the parent never tracks (nor waits on, nor charges for) weights the child
  has already unloaded.
- **The card is healthy.** The device-free governor's committed state for the card is `HEALTHY`. A
  `PRESSURE` or `SATURATED` card is one the verified reclaim ladder is or may soon be reclaiming from, so it
  is handed no new resident to evict. This reads the one figure a WDDM driver cannot misreport under
  demand-paging (NVML device-free), so it holds precisely in the regime where measured free VRAM lies.
- **The slot's own recent traffic repeats this model.** The dispatched model must appear among the slot's
  previous `_RETENTION_REPEAT_EVIDENCE_DISPATCHES` (3) dispatches. See
  [Retention is granted on repeat evidence](#retention-is-granted-on-repeat-evidence).
- **The card statically fits the job.** The card's reported total (a constant the driver cannot misreport)
  must absorb the job's sampling peak plus the reserve, after charging everything else that shares the card
  while the weights are held: the sibling CUDA contexts (inference siblings, the post-processing lane, the
  disaggregated VAE and component lanes, the on-GPU safety process), the models other slots are already
  holding resident under earlier grants, the component-cache tenancy idle lanes report holding between jobs,
  the job's own post-processing, and any sibling the clearance lease is about to admit. That last term is
  what a multi-slot lease adds: a staged sibling's weights land at *its* clearance, so at this instant the
  shared ledger carries only its encode charge and the rest of its materialisation is charged here. Priced
  without it, two grants each fit "alone" and jointly overflow the card, which is the shape every observed
  out-of-memory on a two-slot lease took.

A grant is settled into the slot's retained-resident record only when its job **succeeds**. A fault is
evidence about the job and none at all about the device: a job that failed part-way through may have left
its weights standing, freed them, or never loaded them, and the parent cannot tell which. Recording
residency on that guess is what makes the static fit above charge weights that are not there and same-model
routing seat a successor on an empty slot, so a faulted result drops the grant and the record together.

### Retention is granted on repeat evidence

Card health and static fit say a copy *can* be held; neither says anything about whether one *should* be. A
copy nothing comes back for is not free: it occupies the card, prices every later grant's static fit against
itself, and is eventually handed back through the reclaim ladder having saved nothing. On a worker offering a
wide model mix that describes most copies, and the accumulated holds recreate the pressure the ladder then has
to resolve.

The evidence a grant needs is therefore what the slot has already been asked to run: the dispatched model must
appear among that slot's previous three dispatches. This adapts to whatever an operator offers without
encoding an assumed mix or a machine's capacity. A slot serving a single-model pool supplies the evidence on
every dispatch after its first and is granted exactly as freely as an ungated policy would grant it; a slot
rotating more models than the window holds earns close to nothing.

The evidence is **trailing**, never a queue lookahead, and that distinction is why the gate can exist at all.
The pop cycle refills the queue immediately *after* a dispatch drains it, so at the dispatch instant a
same-model successor is almost never visible in the pending set even when one arrives milliseconds later; a
gate reading the queue would refuse a pool-locked worker every grant and make retention unreachable. What a
slot has already run is under no such timing.

The one cost is warmup: the first dispatch on a slot has no history behind it and is refused, so a streak pays
for its weights twice rather than once. That amortizes to nothing over any real streak, and it is the price of
granting on evidence rather than on assumption.

The trailing history lives on the scheduler, keyed by slot, because it describes the slot's traffic rather
than one process's residency: it must outlive every job boundary, and a slot whose child was replaced is still
serving the same shape of work.

### A hold nothing comes back for is revoked under sustained pressure

A grant is a prediction, and before this nothing revisited one: only an eviction actuation could end a
retention, so a hold taken in a healthy moment outlived every subsequent change in what its slot was being
asked to run.

Re-asking the *issuance* question is not what re-opens it. A live grant's own dispatch heads the slot's
history, so the window a sweep would read is the window that issued the grant and every live retention passes
by construction. What can refute the prediction is the predicted successor failing to arrive. When a card has
been off `HEALTHY` continuously for `_RETENTION_PRESSURE_REVOKE_SECONDS` (15s), any retained copy that has
gone unreused for `_RETENTION_STALE_HOLD_SECONDS` (60s) is given back. Both constants are starting points
pending a signature sweep, and both are expressed in seconds of demand so they mean the same thing on any card
and any offer size.

A hold the traffic is still using is never revoked, however long the pressure lasts: each reuse ends its
episode and starts a fresh one, so a pool-locked slot's hold age resets every job and can never reach the
horizon. `HordeProcessInfo.retained_resident_since` carries the episode's start, stamped by the scheduler
(which owns the clock every other retention window is measured on) rather than at the settle, which runs on
the completion path.

This is not a second reclaim ladder. Genuine saturation remains the verified ladder's to resolve, and retained
residents are already first-class candidates for it; the sweep only removes holds that had stopped being a bet
on anything. Revocation actuates through the ordinary idle-model unload and registers the same in-flight
eviction record a dispatch-time eviction does, so a dispatch priced against those weights waits for the card
to evidence the free rather than for the request to have been sent. A busy slot is never touched.

### Reading retention back

Retention decides on evidence that is only visible in aggregate, so the counters are what say whether the
policy is paying for itself on a given worker rather than merely how often it fired. They reach
`RunMetricsSnapshot` and the `GPU duty cycle` line beside the reload-churn figures:

| Counter | What it says |
| --- | --- |
| `retention_grants_issued` | Dispatches whose weights were left on the card. |
| `retention_grant_denials` | Refusals bucketed by the gate that refused (`no_repeat_evidence`, `governor_state`, `static_fit`, `unpriceable`, `wddm_paging`, `budget_inactive`, `actuation_disabled`). |
| `retention_reuses` | Dispatches that landed on a slot already retaining that model: one per job served without an upload. |
| `retention_evicted_unused` | Copies given back before any successor reused them. |
| `retention_revokes` | Copies the sustained-pressure sweep took back as stale. |

Reuses and unused evictions partition every retention episode, so their ratio is the read: a run where unused
evictions dominate is paying for holds its traffic never came back for. The denial buckets separate a worker
whose traffic retention cannot help (`no_repeat_evidence`) from one whose card will not carry what it could
(`static_fit`, `governor_state`).

Disaggregated sampling is granted the same way, and priced differently. A sampler runs the identical
end-of-run eviction, so without a grant reaching it the stage returns the card after every sample and the next
same-model sample re-uploads the UNet; the grant rides `HordeSampleControlMessage.keep_model_resident_after`
into `hordelib`'s `sample_stage(defer_vram_unload=…)`, and the same dispatch carries the parent's device
reading (`device_free_mb`) so the sampler's shortfall arithmetic is computed against the card rather than its
own view. What such a slot then holds is the UNet alone (its text encoders ran in the encode service, its VAE
in the image lane), which is recorded as `retained_resident_component_only` and charged at the checkpoint's
component-identity residual rather than the whole checkpoint. Pricing it as a whole checkpoint would charge the
card for support weights no process holds and collapse exactly the co-residency disaggregation exists to buy;
an unreadable sidecar denies the grant rather than falling back to the over-charge.

The grant is settled when the *sample stage* ends, not at the job's completion: that is the instant the
sampler's device either holds the UNet or does not, and the decode that follows runs on the image lane and
cannot change it. Settling at completion would leave the parent blind to those weights for the whole decode
window (a window the scheduler may preload in) and would let a decode fault clear a residency the sampler
really holds. A stage that faults, or a job that never reaches its sampler, discards its unsettled grant so no
later completion on that slot can settle it into a phantom; weights the slot retains from an earlier job are
left standing. The sampler's ownership record is retired at the same point, since the slot owns nothing once
sampling has ended and the synthesized completion retires ownership on the decode lane.

Retention is cumulative, which is why the fit charges the residents it has already granted. Each grant
leaves weights on the card until an eviction actuates, so the next grant's sampling peak has to fit beside
them; a fit that counted only live contexts would price every grant as though it were the only one and let a
run of grants across sibling slots sum past the card. The parent tracks what each inference slot retains
(`HordeProcessInfo.retained_resident_model`, set from the dispatch verdict when the job's result arrives and
cleared by every eviction actuation and at slot death), and charges each tracked resident at its full weight
footprint. A slot's own retained model is charged only when it differs from the dispatched job's: a
same-model streak's re-grant is reusing exactly those bytes. A tracked resident whose footprint cannot be
estimated denies the grant rather than being charged zero.

The record is a prediction, and the child reconciles it. ComfyUI satisfies an allocation by freeing memory
on the device, and hordelib's `load_models_gpu` hijack falls back to an unbounded requirement, so one load
inside a run can unload every other model on the card, the granted checkpoint included. Nothing the parent
can measure distinguishes that from a grant that held: the slot still reports a model, and the freed bytes
read as ordinary headroom. Left alone, the phantom is charged by the retention fit, waited on by the
dispatch admission gate, and routed to by same-model placement for the rest of the session, all for weights
that have to be re-uploaded anyway. So the engine reports it: a run granted `defer_vram_unload` that ends
with no model on the inference device carries `retained_weights_evicted` on its result (see hordelib's
[ComfyUI bridge](https://github.com/Haidra-Org/hordelib/blob/main/docs/comfyui-bridge.md)), the inference
child turns that into the `UNLOADED_MODEL_FROM_VRAM` state change and fresh memory report a
parent-commanded unload already sends, and `ProcessMap.on_model_vram_clear` drops the slot's retained
record. The clear takes the in-flight grant with it, because the report arrives around the end of the job
whose result settles that grant into the record; clearing only the settled half would let the settle write
the phantom straight back. The next dispatch to that slot is then priced as the cold load it really is.

The same discipline runs in the other direction, for an unload. A full VRAM free is a request, not a
guarantee: the child's backend drops what it can and skips any model a live reference still pins, and the
command reports nothing about the difference. A child that reported host-RAM residency because that is what
the command asked for would hand the parent gigabytes of room the card is still holding, and the head would
then be held "not fitting" against a ledger showing space after every evict. So the child judges the unload
by what the device is left holding (hordelib's `VramUnloadResult`, see its ComfyUI bridge doc) and reports
the residency the device really has, flagging the refusal as `vram_unload_refused` on the model-state
message. `ProcessMap.on_vram_unload_refused` then keeps the slot recorded as VRAM-resident, leaves its
outstanding-unload flag standing, and does not restamp its materialization recency. Those three together are
what make the reclaim ladder pass the slot over: every candidate set and the unload actuator already read a
standing unload flag as a reason to skip, so the episode escalates to its next rung instead of asking the
same refusal the same question every tick. The refusal is retired by a genuine re-materialization, by a
verified clear, or by the slot's death, so a slot that recovers is a reclaim candidate again.

A retained resident is a dispatch destination, not only a charge. `loaded_horde_model_name` records that a
slot has served a model, not that its weights are still there, so two slots can read as equally resident
while only one carries the bytes; dispatch selection therefore prefers the slot that retains the model, and a
same-model head whose retainer is busy waits for it rather than funding a second full copy that cannot fit
beside the first. The wait is bounded by the same ttl-derived affinity budget the resident-bypass window
uses, ends the moment the retention record clears, and lets other resident work bypass it meanwhile, so the
card is never idled by it. Where a second copy does fit, nothing is held back.

Placement order defers to a retained copy. The preload pass walks the queue in its own order, so on a model
rotation wider than the lane pool the cold head's load target is whichever slot is free, which is routinely a
slot retaining a model still in the queue: the head loads over those weights, the job that would have reused
them re-uploads onto the other slot, and two models can trade lanes for the rest of a session at a full upload
per job. `_retention_affinity_candidates` names the pending jobs a slot's retained weights can serve at no
upload (residency plus `can_accept_job`, never this instant's concurrency headroom), and
`pending_inference_in_placement_order` moves those jobs ahead of the head, keeping their relative order and the
relative order of everything behind them.

The reorder is the whole mechanism. No load is withheld and no lane is excluded as a preload target; every
consumer of the ordering then reaches the same conclusion without being told about retention. The preload pass
takes the promoted job as its head, finds it dispatchable on the lane already holding its weights and so yields
the cycle to dispatch; dispatch seats it there at no upload; a preload that does run targets the lane its own
job needs. `retention_affinity_reorders` counts the reorders at the dispatch commit, where the dispatched job is
compared against the queue's own head, so a mismatch identifies one without the selection path reporting it.

FIFO first: a reorder is admitted only as a free win. `_reorder_is_pareto_admissible` requires both halves. The
head must actually need a load, since a head that is resident and merely waiting on capacity is delayed by
anything seated ahead of it; and that load must have an admissible target other than the retaining lane, so it
runs alongside the promoted job's sampling and the head finishes no later than it would have. Where the retainer
is the head's only possible target, the head keeps strict priority and its load evicts the retained weights:
buying an upload back there means making the head wait for a whole other job to sample and finish, and the head
is racing a server-side ttl this worker cannot observe, so its deadline is never asked to fund a reuse win. The
refusals are counted in `_retention_reorder_pareto_vetoes`. The gate is deliberately unsatisfiable on a two-lane
worker whose retainer is the one idle lane, which is where line-skipping would risk job aging for the least
benefit; the win belongs to pools wide enough to stage the head's load elsewhere.

Candidacy locates the retainer with `include_reserved=True`. A disaggregation-pinned sampler lane is a lane no
job may be dispatched onto *yet*, and it is still a lane carrying weights: the pin is taken when a job is
registered on the lane and released when its sampling ends, and for much of that the lane sits idle awaiting its
conditioning while reporting an accepting state. Whether its weights may be thrown away is a residency question,
and the pin lifting is precisely why the queued job will be seated there afterwards. Locating the retainer
through the dispatch-legal query instead empties the scan on a disaggregating worker, where a lane is pinned for
most of every job. Dispatch takes the opposite view of a pin, because it names a destination:
`ProcessMap.get_process_by_horde_model_name` skips pinned lanes by default, so a promoted job whose only
retainer is pinned is simply not dispatchable this cycle and the head's own load proceeds.

The reorder is bounded by exactly the budget that bounds the bypass (the ttl-derived affinity window, the skip
ceiling, and the anti-starvation age override). The ceiling is the operative bound, because seating a job ahead
of the head *is* a committed pass of that head and advances the budget exactly as a line-skip does, so a head is
passed a bounded number of times whatever its ttl and a head behind a steady stream of retained work sees no
candidate named at all. It is keyed to a retained copy, so a worker holding none (a cold start, the legacy
hatch, traffic with no repeat inside the queue window) schedules exactly as it did before.

Dispatch admission charges the residents too. A dispatch that materializes weights is priced against every
retained resident the card carries, not just the slot it lands on: on a non-fit the idle ones are evicted
through the ladder's actuator and the job keeps its queue position until the child's own reports (a risen
device-free reading, a fallen slot reservation, or the model map no longer placing those weights there)
evidence the room is back, bounded so a child whose reports never arrive leaves the dispatch to the measured
admission gate rather than parking the queue. The gate charges only *sibling* residents, because a
cross-model dispatch onto a retaining slot already evicts that slot's own weights ahead of its
`START_INFERENCE`; the two paths act on disjoint slots and never ask for the same weights twice.

Model changes evict before they load. When a slot's next job is for a different model than the one it
retains, the dispatch path sends that slot an explicit VRAM unload ahead of `START_INFERENCE` and clears its
residency record, so the child frees the old weights before materializing the new ones instead of carrying
both through the job. This is not left to the child's own free-view, which is untruthful under WDDM in
exactly the regime double residency creates.

Every dispatch carries the parent's device reading for that reason. `HordeInferenceControlMessage`'s
`device_free_mb` is the parent's NVML device-level free figure at dispatch, and the child forwards it to
hordelib as `device_free_truth_mb` so the executor's shortfall arithmetic (what it must free before a weight
load or a sampling window) is computed against the card rather than against the process. A child sees only its
own allocations and, under WDDM, memory the driver has not returned still reads as free, so an unclamped
shortfall comes out too small: the child frees less than the load needs, allocates anyway, and real free VRAM
craters to the paging cliff, where the governor saturates and the reclaim ladder starts taking the worker's own
capacity down. Retention is what makes that reachable at all, since it is what leaves a footprint standing
across a job boundary for the child's own arithmetic to be the last defense over. The parent's figure is
likewise taken as a ceiling on the child-reported free VRAM the scheduler prices admission and headroom from,
so the same overstatement cannot buy headroom on the parent side either. The scenarios in
`tests/process_management/liveness/test_incident_scenarios.py` hold this: a streak whose true footprint nearly
fills the card keeps its margin and its duty when the reading is on the dispatch, and craters when it is not.

No measured-floor veto and no sole-residency rule apply in this seam. The measured identity is the
admission/dispatch gate's job; re-imposing it on retention only reintroduces the never-fires problem via
committed-figure noise. Sole residency is unnecessary because a second idle resident is safe: it is a
first-class candidate of the verified reclaim ladder, which reclaims newest-idle-first (LIFO) and confirms
each free at the device level.

Eviction is therefore just-in-time. A cross-model preload that no longer fits because idle retained
residents hold the card defers while the ladder evicts them (the head-of-queue reclaim targets the idle
resident and re-asks once its free verifies), and the under-pressure reclaim overrides retention outright.
The dispatch-reconciliation gate is the same reclaim in the other direction: where the preload gate makes
room to bring a model *toward* the card, the dispatch gate makes room for an already-staged job to *commit*
to the card, evicting the retained idle resident that would otherwise share the sampling peak. Between them
these bound what an unused hold can cost: the next cross-model dispatch on the card takes it back, the ladder
takes it back the instant any overcommit picture appears, and on a card that simply stays pressured the stale
sweep takes it back without waiting for either. What keeps unused holds rare in the first place is the repeat
evidence a grant needs, so the just-in-time paths are the backstop rather than the policy. This dispatch-time
reconciliation is the precondition for defaulting cross-job retention on: until a staged dispatch is priced
against the card, retention's idle residents can only be reclaimed after the fact, so the retention default
stays off pending that regime's validation at system scale.
