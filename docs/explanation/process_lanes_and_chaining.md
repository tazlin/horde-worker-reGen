# Process lanes and job chaining

The worker treats a job as a small pipeline of stages rather than a single blob of work: generate, then
optionally post-process, then safety-check, then submit. Each stage is served by a *lane*: a process (or pool)
that holds the models that stage needs resident and does nothing else. This page explains the lane topology,
the chain model that describes a job's route through it, and why post-processing gets a dedicated lane.

## The lanes

| Lane | Process type | Resident models | Work served |
| --- | --- | --- | --- |
| Inference | `INFERENCE` (pool, per card) | The image-generation checkpoint(s) | Image generation (`IMAGE_GEN`) |
| Post-processing | `POST_PROCESS` (single) | Upscalers, face-fixers, background removal | Job post-processing phases and graph alchemy forms (`ALCHEMY_GRAPH`) |
| Safety | `SAFETY` (single) | The CLIP safety stack | Safety evaluation (`SAFETY_EVAL`) and CLIP alchemy forms (`ALCHEMY_CLIP`) |
| Image utilities | `UTILITIES` (single) | ControlNet annotators, background-removal stack | ControlNet annotation and background removal (`IMAGE_UTILITIES`) |

The image-utilities lane is the odd one out: it is not a multiprocessing child but an out-of-venv
subprocess bridged by a parent-side adapter. See [Image utilities lane](image_utilities_lane.md).

Dispatch keys on the capability flags (`WorkerCapability`), not on process types, so a new work kind adds a
flag and a handler rather than special cases. The download process sits outside the lanes entirely.

The post-processing lane is a single shared process pinned to the non-safety card (the first configured card
when the safety process is CPU-only). It is controlled by `dedicated_post_processing`:

- `auto` (default): the lane runs whenever post-processing is allowed (`allow_post_processing`).
- `on`: the lane always runs.
- `off`: no lane, and the worker does not offer post-processing at all. The lane is the only place
  post-processing runs; there is no inline fallback inside the inference processes.

Pipeline disaggregation (`enable_pipeline_disaggregation`) forces the lane on regardless of this control, the
same way it forces the VAE lane on. A disaggregated job's VAE lane returns raw decoded images, and its
requested post-processing runs on this dedicated lane afterward (never inline on the VAE lane), so the lane is
required whenever disaggregation is enabled.

A disaggregated job whose next stage has no live role process distinguishes why the process is missing. A
crashed lane ages through the stage-patience window and is faulted so the horde reissues the job. A lane that
is deliberately paused off-GPU (whole-card residency claiming the card for a heavy head, or the reclaim
ladder) is a routing decision, not a failure: the job is rerouted to the monolithic path at once, where the
normal claim/dispatch queue arbitrates waiting on the card, instead of parking against a pause whose restore
may be minutes away and forfeiting the job at the patience fault.

Because that reroute is silent (every job simply takes the monolithic path), a **silence-breaker** guards
against an outage going unnoticed: when disaggregation is enabled but its role lanes stay unavailable (paused
or absent) continuously past a short window, the parent emits one edge-latched WARNING naming which role is
down and why, re-armed when routing returns. This surfaces a lane that fails to come back rather than leaving
the advertised disaggregation silently dead.

When safety runs on GPU, only CLIP remains resident while the lane is idle. DeepDanbooru stages from CPU for a
conditional anime check; BLIP captioning and the aesthetic head are also offloaded after use, and completion
trims the allocator before `WAITING_FOR_JOB`. This bounds the lane's fixed card cost without making every
safety evaluation pay a CLIP reload.

## Why a dedicated post-processing lane

Post-processing used to run inline on the inference process that generated the images. That welded an
unpredictable, output-scaled VRAM spike (a 4x upscale of an SDXL batch peaks at several gigabytes) onto a
process whose budget was sized for sampling, and it forced every inference process to load its own copies of
the upscaler/face-fixer models on demand. The scheduler carried a family of compensating mechanisms
(committed/imminent post-processing reserves, a pre-dispatch reclaim planner, an overlap bump) to predict and
dodge those spikes.

The lane replaces prediction with structure:

- The spike happens in one process whose models stay resident, so there is no per-job model reload and the
  peak lands on a process the parent can observe and control.
- The inference slot frees the moment sampling and VAE decode finish; the next job can start generating while
  the previous one post-processes on the lane.
- The scheduler charges the lane's fixed CUDA context in its residency forecast and charges each active
  post-processing job's estimated peak in the shared committed-reserve ledger. The hold is released when the
  lane result arrives, when a retired result is known lost, or when orphan recovery requeues or faults.
- The lane reports VRAM like an inference process. Its sample participates in `ProcessMap.get_free_vram_mb`,
  so the parent sees the same low-free-VRAM condition that would make ComfyUI tile or stream.
- Idle reclaim commands apply to the lane too: under pressure, the scheduler can ask an idle `POST_PROCESS`
  process to unload modules from VRAM/RAM. Active post-processing is never interrupted for reclaim, and a
  queued image post-processing job or graph-backed alchemy form keeps the lane out of whole-lane pause
  candidates. The lane is considered for whole-lane pause only when post-processing is enabled, the worker is
  still offering it, the process is idle, and no shared-lane work is committed.

Admission to the lane is governed so a chain the card cannot host never wedges it. The orchestrator admits a
job only when its marginal candidate fits measured device-free VRAM minus outstanding dispatch-flow
reservations and the proportional noise margin (with the budget on; see
[Performance and backpressure](performance_and_backpressure.md)). Preload-flow planned charges are excluded
from that arithmetic: a RAM-staged load's VRAM claim is re-priced at its own dispatch and cannot precede the
drain whose completion frees the room it waits on, so counting it against the lane would deadlock the drain
behind bookkeeping until the aging escape faults the finished job. The manager refreshes the arbiter's
device-memory snapshot immediately before driving the lane, and the selected post-processing job is priced as
the head of the drain queue: once inference dispatch is being held to let the lane run, the lane must be able
to take the same reality-based admit a head-of-queue model would get after reclaim is exhausted. Three
behaviors keep an unfittable chain from parking the queue:

- **Queue scan**: the first *fittable* pending job is dispatched, so an unfittable head never blocks the
  fittable jobs queued behind it. The same rule applies when active sampling temporarily blocks one chain's
  co-residency: that chain keeps its patience record, and the scan continues so a later chain that can share
  the card may use the idle lane.
- **Aging escape**: a job that stays unfittable past the admission-patience window is submitted as a no-image
  fault, so the horde reissues it to another worker rather than letting it park forever. This feeds the
  circuit breaker because the worker accepted post-processing work it could not host.
- **Bounded reclaim**: an unfittable job executes each newly available arbiter reclaim plan once per starvation
  episode through the shared reclaim owner, rather than issuing an unrelated model-only sweep every scheduling
  tick. The first rejection leaves service-lane borrowing disabled while idle caches, idle weights, reducible
  inference contexts, and any policy-permitted safety action run in their existing order. Only if a fresh
  measured re-check still does not fit and those softer actions are exhausted may the drain borrow one idle VAE
  or component lane context. That one-context loan is the episode limit; a continuing non-fit falls through to
  normal admission aging and recovery instead of stacking lane teardown. Busy service lanes are never
  candidates. An applied-action receipt and the existing pause-owner guard ensure this path restores only a
  pause it acquired. Safety may move off-GPU only under the operator's existing policy. This gives accepted lane
  work a path to measured room without repeatedly churning inference residency or weakening the recovery
  fences.

### Restore ownership: every lane pause has exactly one responsible owner

A service-lane off-GPU pause frees a real CUDA context, so it has no external trigger to bring it back: whoever
paused it must own its restore. There are two responsible owners, keyed by `PauseOwner` at pause time so
neither lifts the other's hold:

- a **whole-card residency** pause is restored by the residency completion loop when the residency drains; and
- a **reclaim-ladder** pause is restored by the ladder's LIFO unwind when the card's saturation episode ends,
  or, for a lane the post-processing drain borrowed, by that drain's applied-action receipt.

The borrowed lane's release is not gated on the *whole* PP queue draining, because a borrowed lane is a
disaggregation lane: pausing it disables disaggregation, which routes work monolithic, raises card pressure,
and sustains a PP backlog that never fully drains. A queue gate alone would then hold the lane hostage
forever. Instead the loan is held while a job is actively being post-processed (adjacent dispatched jobs share
the one loan), and returned once no job has been actively post-processed for a bounded idle window, even while
jobs remain queued. A loan released this way is not re-borrowed for the same stalled episode (so the lane is
not thrashed), and re-borrowing is re-enabled once the queue fully drains.

Behind both owners sits a conservative **self-heal backstop** in the parent's governor tick: a reclaim-ladder
lane pause that no live saturation episode and no PP-borrow receipt still claims is restored once the card has
been governor-`HEALTHY` for a debounced interval, with a WARNING naming what was stranded. It never lifts a
whole-card pause or a pause a live claimant holds; it exists only to reclaim an orphan neither responsible
owner will.

### Decode-drain eligibility: a VAE-lane pause defers to a queued decode

The arbiter guarantees a lane it names for a pause is *idle* at the instant of the pause (its process is not
busy), but a lane being idle this instant does not mean it has no imminent work: the disaggregation
orchestrator holds jobs at `AWAITING_LATENT_DECODE` whose sampling already finished and whose only remaining
step is a ~1-2s VAE decode on that same lane. Pausing the lane out from under such a job strands the finished
sample, since the job then reroutes monolithic and re-runs whole, discarding the completed sampling to free
room for a dispatch the decode itself would have cleared within seconds.

So the reclaim-ladder VAE-lane pause is **decode-drain-aware**: it reports no-op while the orchestrator has any
job needing the lane for a decode (queued or dispatched to it but not yet resulted). Both reclaim paths that
stop this lane, the governor's saturation rung and a post-processing borrow, execute through the one reclaim
owner, so a single no-op there makes each move to its next relief option exactly as any rung whose target has
gone away does; the pause is simply not eligible this tick and pressure logic proceeds. The eligibility reads a
cheap orchestrator accessor (the count of jobs at the decode stage) injected into the scheduler as a callable,
and emits one edge-latched INFO line naming the pending-decode count when it withholds a pause, so the lever is
visible in live forensics.

A job that is *merely sampling* does not withhold the pause. Rerouting an unfinished sample discards no
completed work, the existing defer window already covers a sampler whose lane a pause outlasts, and relieving
device pressure matters more than protecting a sample that has not yet produced a latent. The mirror case for
the component/text-encode lane is deliberately left ungated: encode sits at the front of the pipeline, so
rerouting an `AWAITING_CONDITIONING` job monolithic discards nothing, and the wasted-work argument that
motivates the VAE-lane gate does not apply.

### Owner-aware decode hold: waiting out a bounded pause instead of rerouting

The decode-drain eligibility gate only stops a *new* pause from executing while a decode is already pending. It
cannot cover the window where the pause lands earlier: because a job merely sampling does not withhold the
pause, a reclaim-ladder pause can execute while a job is still sampling, and that job then finishes sampling and
reaches `AWAITING_LATENT_DECODE` a few ticks later to find the lane already paused off-GPU. The general rule for
a deliberately-paused role lane is an immediate monolithic reroute (waiting on the card is arbitrated by the
monolithic queue, and parking the job only ages it toward a patience fault). At the decode stage that rule is
made **owner-aware**, because the two pause owners differ in how long they last:

- a **reclaim-ladder** pause is bounded by construction (its idle-release restores the lane within seconds of
  the borrower going idle, and the self-heal backstop covers an orphan), so a job whose sampling has already
  finished **holds** at `AWAITING_LATENT_DECODE` for the restore rather than discarding that sampling to a
  reroute. The hold is kept clear of the no-role patience clock, so a legitimate wait never ages toward a fault,
  and the next tick re-attempts the decode dispatch with no manual kick, dispatching the moment the lane returns.
  Should the lane not restore within the bounded hold window, the job reroutes monolithically as a backstop, so a
  pause that never lifts can never strand it. One edge-latched INFO names the held-job count for live forensics.
- a **whole-card residency** pause (or an unknown or absent owner while paused) lasts a heavy model's whole
  residency (minutes), so the instant reroute remains correct: waiting is not worth protecting the finished
  sample.

The owner reaches the orchestrator through the same injected-predicate seam as the lane-paused check: lifecycle
exposes the VAE-lane pause owner and the manager injects an accessor callable alongside the paused predicate.
Only the decode stage holds; encode and source-latent stages keep the instant reroute, since rerouting them
discards nothing worth waiting on.

### Class eligibility: which jobs the pipeline accepts

A job is class-eligible for disaggregation only when disaggregation is enabled and the job is one the staged
graph can build faithfully: its effective (post-fallback) source processing is txt2img/img2img/remix; it
carries no control_type; it is not transparent (the layerdiffuse decode graph is not identity-validated on the
staged path); its model is **not an inpainting-variant checkpoint**; its model's baseline is an SD1.5 or SDXL
family; and it has not already been re-routed out of the pipeline. An inpainting checkpoint's UNet takes the
masked-image-plus-mask input channels, which the txt2img sample graph the staged path builds does not supply,
so a staged sampler would fault it on the first slice. The inpainting flag is read from the loaded model
reference record (the same lookup that resolves the baseline); an absent or `None` flag is treated as not
inpainting, so incomplete reference data never excludes a model.

### Structural stage fault: decline the pipeline for the retry

A disaggregated stage can fault for two distinct reasons, handled differently. A **resource-class** fault
(the stage was denied device VRAM under pressure) defers, and re-routes monolithically only if the pressure
does not clear within the defer window, since the demand may simply not fit the card right now. A
**structural** fault (a non-resource-class error the whole-job graph would not hit, such as a checkpoint the
staged graph cannot faithfully build) is not device pressure and re-sampling on the pipeline would fault it
again. So a structural stage fault latches the job **disaggregation-declined** as it requeues: the retry is
kept off the pipeline by the class-eligibility predicate and runs on the whole-job path, whose graph
construction accommodates the job. The terminal path is unchanged: if the monolithic retry also fails, the job
faults for the horde to reissue. Only the fault reason authoritative to the orchestrator drives this; a
resource-class fault keeps its existing defer-then-reroute behavior.

Pending post-processing work also owns lane liveness. If the queue is non-empty and no `POST_PROCESS` process
exists, the orchestrator asks lifecycle to start the lane; lifecycle still owns the actual admission checks
(the operator setting, GPU-start headroom, pending starts, and deliberate off-GPU pauses). If the lane remains
absent after that request, the same admission-patience clock runs so the job cannot sit in
`PENDING_POST_PROCESSING` forever. The exception is an active whole-card residency pause: that pause is expected
to restore itself when the resident model releases the card, so the patience clock stays unarmed while residency
is still active. If a whole-card pause is left behind after residency and inference work have drained, pending
post-processing restores that owner-scoped pause and then re-enters the normal lane-start path.
When a whole-card lease itself releases, accepted post-processing keeps safety off-GPU until the lane queue
drains; restoring safety first would consume the room the downstream job is waiting for. A drained lease's
speculative cooldown also yields immediately to a ready different-model inference head on the same card, so
the cooldown cannot park useful resident work.

The overlap gate is keyed to active sampling on the lane's card, not to every job in the inference
`IN_PROGRESS` stage. A job that is only blocked in `DOWNLOADING_AUX_MODEL` holds an in-flight slot but is not
using the GPU for denoising, so already-popped post-processing work is admitted and drained before any
line-skip candidate is launched to keep the card busy during that download. If sampling is active and one
pending chain cannot safely co-run with it, the orchestrator records that chain's deferral but keeps scanning
for later pending work whose estimated peak can co-reside. If no pending chain can co-run, or if the current
sampler has just drained and the next sampler would also be unable to co-reside, the inference scheduler holds
that next sampler so the lane gets the next drain window instead of extending the never-idle period. Model
preloading is treated as speculative work in this state and yields to the same pending chain.

The popper protects the lane from growing its own downstream queue without stopping ordinary image work. Once
two or more accepted jobs still need post-processing, the next image pop temporarily withholds
`allow_post_processing` while keeping normal generation capabilities advertised. The count includes jobs still
queued or running inference and graph-backed alchemy forms waiting for or running on the same lane, so
back-to-back batched PP jobs and alchemy lane occupancy are visible before the image job reaches the lane.
When the commitment count drains below that point, post-processing is offered again, subject to the normal
operator setting, model-readiness gate, and fault breaker.

A post-processing failure never falls back to raw submission. Requested post-processing is part of the
worker's contract for that job; if the lane cannot honor it, the worker submits a no-image fault so the horde
reissues the job to another worker. Repeated lane faults feed the post-processing circuit breaker
(`post_processing_fault_breaker_enabled`), which stops advertising post-processing before the horde forces
maintenance. A queued job that was already popped when the breaker trips is faulted on the next orchestrator
pass; a job already running on the lane is allowed a best-effort chance to finish, then the orphan watchdog
requeues it a bounded number of times before faulting it without images.

Whole-card residency treats the lane separately from its resident models. When the lane's bare GPU context
fits beside the resident model, the scheduler asks the idle `POST_PROCESS` process to unload its modules from
VRAM and keeps the lane alive so jobs peeling off the resident model can still post-process. If the bare PP
context cannot coexist with a model that requires whole-card residency, post-processing is disabled for the
session with an operator warning (logs, and the TUI health row when attached); stopping the lane is then a
structural compatibility decision, not the normal residency lever.

## The chain model

The route a job takes is described by a *chain flow* from the SDK
(`horde_sdk.worker.chaining`): a validated DAG of stage nodes, each naming the kind of work, the lane
capability it requires, and the generation-progress states that bound it (for example the generate stage runs
from `GENERATING` to its `GENERATION_COMPLETE` milestone). `image_generation_flow(post_processing=...,
safety_check=...)` builds the canonical route at pop time, and every tracked job carries a
`ChainExecutionContext` over it.

The chain is descriptive, not a scheduler: queue membership (`JobStage` in the job tracker) remains the
executor of record, and every stage transition is mirrored into the chain via
`ChainExecutionContext.advance_for_progress`. What this buys:

- Any observer can read a job's position in its routing plan (`snapshot()`) without re-deriving it from
  queue membership across several collections.
- The routing decision "what does this job need next, and which lane serves it" is data
  (`ready_nodes()` plus each node's `required_capability`) instead of hardcoded stage-to-stage branching.
- New stages and lanes (text, audio/video, chained jobs that feed one generation's output into another) are
  new nodes and edges over the same machinery rather than new special cases.

A terminal fault marks the executing node failed and skips everything downstream. Post-processing faults are
terminal no-image faults rather than raw-image fallbacks, so the chain reflects that the requested downstream
stage did not complete.

## The job's path through the lanes

1. **Pop**: the job registers with the tracker; its chain flow is built from the payload (post-processing
   requested or not).
2. **Pre-annotate** (ControlNet jobs the utilities lane can serve): the control loop parks the job in the
   `PENDING_ANNOTATION` stage and derives its control map off-GPU on the
   [image-utilities lane](image_utilities_lane.md) before it is eligible for inference. This stage precedes
   the generate node, so it drives no chain milestone; the job returns to `PENDING_INFERENCE` carrying the
   map on success, or with no map (in-graph fallthrough) on any failure, and is never re-parked. A job the
   lane cannot serve skips this step entirely.
3. **Generate**: the scheduler dispatches to an inference process; the raw images return to the main process
   at the generate stage's completion milestone.
4. **Post-process** (when requested): the dispatcher queues the job for the lane; the post-processing
   orchestrator dispatches the first fittable pending job, sends the raw images and the requested operations,
   and the processed images replace the raw ones. A chain that never fits the card ages out to a no-image
   fault; an orphan watchdog requeues a job whose result was lost (bounded), then faults without images.
   `strip_background` is excluded from this pass (it has no in-graph path) and runs last on the
   [image-utilities lane](image_utilities_lane.md) instead (step 5).
5. **Background strip** (when `strip_background` was requested): the last image transform, run on the
   image-utilities lane in the `PENDING_STRIP` stage after any upscale/face-fix. A strip-only job reaches
   here straight from generation, skipping the post-processing lane. With no in-graph fallback, a fault,
   age-out, or lane death here is a no-image fault (reissue), matching the post-processing lane.
6. **Safety**: unchanged; the safety orchestrator sends the (possibly post-processed) images for evaluation.
7. **Submit**: unchanged; the chain closes out (`SUBMIT_COMPLETE`) when the job finalizes.

Graph alchemy forms (standalone upscale/facefix jobs) ride the same lane via the `ALCHEMY_GRAPH` capability
and count as lane commitments while they are pending or in flight; `strip_background` instead routes to the
out-of-venv [image-utilities lane](image_utilities_lane.md) (`IMAGE_UTILITIES` capability), where its
`rembg` stack stays out of the main environment; CLIP forms (caption/interrogation/nsfw) stay on the safety
process. The alchemy coordinator owns its own pop/submit loop, but the image popper reads its graph-lane
commitment count for post-processing offer shaping.
