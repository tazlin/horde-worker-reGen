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
device-free gain against the rung's promised figure over the next one or two governor samples, escalating on a
shortfall rather than trusting the estimate, and marking the episode *unresolved* only once every rung has run
without relieving the card. The **per-step floor** is the fast detector that forces that ladder early: two
consecutive sampling steps each several times their expected per-step time, on a PRESSURE-or-SATURATED card,
mean a job is being demand-paged (not merely heavy) and reclaim should run without waiting for the whole-job
elapsed-ratio grade. Replacing the crawling sampler is the ladder's **last rung**: it fires only once the card
has been SATURATED past the kill horizon, the ladder is exhausted, and the slot is crawling. That kill gates on
device-level truth, never on the per-PID paging-victim map, which the LRU physics make structurally
unsatisfiable. The full mechanics live in
[Performance and backpressure](performance_and_backpressure.md#the-truthful-signal-hierarchy-and-the-device-free-governor).

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
every other process's planned load stays fully charged, so genuinely-concurrent admissions still stack. Second,
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
(a line-skip) stays fully charged, so it can never consume the room a staged head is waiting on. This is the
same reasoning that admits the disaggregated decode stage unconditionally: work whose turn has come must not
be starved by claims whose own turn comes after it.

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
denied that admit even when the card physically has room right now, because materialising into it starves the
head the skipper jumped: the head needs the same space and took precedence. If it does not physically fit (or
the requester is not the head), the disposition is `DEFER` and the `admission_foreign_pressure_defers` counter
advances. The dispatch-reconciliation gate plumbs the same truth, presenting `is_head_of_queue=False` for a
line-skip dispatch so a line-skipper is held rather than committing over the head at the dispatch seam.

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
  a repeated command sees the count already at target and tears nothing more down (`_establish_whole_card_residency`
  only scales while the live count exceeds the target, and re-stamps the residency once). Both the preload and
  `MONOLITHIC_DISPATCH` paths may use this escape only when the candidate peak computes a maximum
  resident-process target below the current live pool. Merely finding an idle sibling does not qualify: a
  target at or above the live count proves pruning that context cannot address the deficit, so the request
  stays out of whole-card residency. The actuator repeats that check against the current live count before it
  acquires exclusivity or pauses service lanes, because a correctly-issued command can become stale after an
  earlier scheduler tick has already retired its victims.
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
`idle_contexts_teardownable` on this seam independent of the flag, and the actuation (establish the head's
residency, then evict the idle siblings' VRAM) runs through machinery that does not itself consult the flag, so
the contexts are torn down and the head admits regardless of the steady-state preference.

A head deferred past the 60s diagnostic threshold with reclaim genuinely exhausted and no such teardown target
(no first-party context reclaim remains) emits a warning with the full arithmetic and increments
`starvation_diagnostics`; it still does not admit. The job stays queued for the structural queue wedge recovery
supervisor, which detects a stuck queue with no dispatch progress, soft-resets the pools, and then faults
wedged jobs non-retryably so the horde can reissue them elsewhere.

`DENY` is reserved for a candidate that could not fit even an empty card. Model-level prevention keeps those
jobs out earlier, so a runtime `DENY` is a diagnostic boundary rather than a throughput path. The first
concurrent sampling of its kind admits on an empty ledger.

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
card admits even when the operator's reserve would have read it as unloadable.

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

**Safety GPU load.** The recurring safety-on-GPU seam, bringing the safety process back onto the card after a
whole-card residency freed its context, is gated on the arbiter's `SAFETY_LOAD` verdict, charging a documented
safety-context footprint. A `DEFER` keeps safety off-GPU this cycle; a per-tick reconciler re-asks so a
deferred safety load is not stranded off-GPU, restoring it once the card has room and no held residency still
requires safety off its card. The initial cold-start safety load onto the GPU (at worker bring-up, before any
heavy residency pressure) is not gated and always proceeds.

### Runtime safety placement

The single safety process (slot 0) runs on-GPU only when `safety_on_gpu` is configured. On a card too tight to
hold safety's context beside the model that is sampling on it, that CUDA context competes for VRAM the sampler
needs. The scheduler-owned **runtime safety-placement policy**
([`_reconcile_runtime_safety_placement`][horde_worker_regen.process_management.scheduling.inference_scheduler.InferenceScheduler])
generalises the whole-card safety-off lever to that ordinary case: it moves safety to a CPU-only process when
its charge cannot fit, and re-promotes it to the GPU once a card proves durable room. `safety_on_gpu` remains
the operator's maximum permission; the policy only degrades GPU to CPU and back, never beyond that grant.

GPU placement treats CLIP plus its CUDA context as the lane's fixed residency, not the last evaluation's
high-water mark. DeepDanbooru stays in host RAM until an anime check calls it; BLIP and the aesthetic head are
offloaded after use; and transient allocator blocks are cleared before the child reports itself idle. The
device-free reading therefore sees those allocations as reclaimable work, rather than evidence that the whole
safety process must be replaced or that an ordinary SDXL-class dispatch needs exclusive residency.

The two sides read **different signals**, which is what makes re-promotion satisfiable under sustained load.
Demotion prices a *modeled* worst case: the charge must fail to fit beside the largest learned sampling peak
(device total less that peak, a proportional noise buffer, and the safety charge), a predictive eviction that
acts before the card reaches the paging cliff. Re-promotion instead reads the chosen card's *measured*
device-free VRAM between allocation peaks (the governor's truthful NVML-derived figure) and requires the card
to be governor-`HEALTHY`. The modeled peak is always populated while jobs flow, so a modeled restore predicate
could never be satisfied under load; the measured free rises whenever the card genuinely has room, so it can.
On a box where no card can host safety beside its sampler (two small cards, a large model on each) the measured
streak never accrues, and **CPU safety is the correct steady state**, with the post-inference backpressure above
bounding intake to CPU-safety throughput.

Both sides are **hysteresis-gated**. The off-latch turns on only after a short run of modeled-non-fit cycles
(`_SAFETY_PLACEMENT_PAUSE_STREAK`) and off only after a longer run of measured-headroom cycles
(`_SAFETY_PLACEMENT_RESTORE_STREAK`), with a deadband (modeled fit but measured room not yet proven) that
advances neither streak. The asymmetric streaks double as a demote-again cooldown: a promotion resets the miss
streak, so a fresh run of non-fit cycles must pass before safety can be evicted again. Actuation is skipped
while a safety check is pending or active (no mid-backlog churn), and re-promotion is additionally withheld
while a whole-card residency still needs safety off its card or the device-free governor is holding growth, so
this policy fights neither the residency machinery nor the cliff brake.

The verified reclaim ladder uses the same operator permission as whole-card safety movement: if
`whole_card_residency_safety_off_gpu` is false, safety is not added as a reclaim rung even when it is on GPU.

**Placement is headroom-aware across cards, not a fixed device 0.** One identity
([`_choose_safety_gpu_card`][horde_worker_regen.process_management.scheduling.inference_scheduler.InferenceScheduler])
picks the driven card with the most verified headroom (measured device-free when reported, else card total less
the modeled peak) and is pushed to the lifecycle manager each cycle, so both the spawn-time pin and every
re-promotion respawn land on the same chosen card. Demotion, promotion, and the current placement card (or
`None` for CPU) are surfaced in the run metrics.

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
inequality. Multi-GPU union pops keep a model if any serving card can host it; targeted pops are scoped to
the under-fed card's capabilities, so the same arithmetic applies to the card being advertised. Each excluded
model logs one INFO line naming the arithmetic per card.

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
tight-card case; the two share the single pause/restore machinery, and the placement latch withholds the
residency-drain safety restore so the two controllers never fight over the safety process's placement.

**Lane yield parity.** The disaggregated pipeline's component (text-encode) lane, like its VAE lane and the
post-processing lane, holds a permanent CUDA context plus resident weights that a sibling teardown cannot
reclaim. On a card too tight to host a whole-card model beside that context, the lane must vacate the card
exactly as safety does. Each of these lanes is therefore stopped wholesale (context and models freed) when a
whole-card residency is established on its card, is a member of the residency's teardown-complete gate (the
heavy model is not admitted until the lane's process has actually exited, not merely been asked to), and is
restarted once the residency drains. Stopping the component lane also drops it from the disaggregation liveness
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

- **The card is healthy.** The device-free governor's committed state for the card is `HEALTHY`. A
  `PRESSURE` or `SATURATED` card is one the verified reclaim ladder is or may soon be reclaiming from, so it
  is handed no new resident to evict. This reads the one figure a WDDM driver cannot misreport under
  demand-paging (NVML device-free), so it holds precisely in the regime where measured free VRAM lies.
- **The card statically fits the job.** The card's reported total (a constant the driver cannot misreport)
  must absorb the job's sampling peak plus the reserve, after charging the sibling CUDA contexts and the
  job's own post-processing that share the card while the weights are held.

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
to the card, evicting the retained idle resident that would otherwise share the sampling peak. An unused hold
costs only the interval until the next dispatch, so retention can stay generous while the card is healthy and
the ladder takes the weights back the instant any overcommit picture appears. This dispatch-time
reconciliation is the precondition for defaulting cross-job retention on: until a staged dispatch is priced
against the card, retention's idle residents can only be reclaimed after the fact, so the retention default
stays off pending that regime's validation at system scale.
