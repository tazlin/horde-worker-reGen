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

Dispatch keys on the capability flags (`WorkerCapability`), not on process types, so a new work kind adds a
flag and a handler rather than special cases. The download process sits outside the lanes entirely.

The post-processing lane is a single shared process pinned to the non-safety card (the first configured card
when the safety process is CPU-only). It is controlled by `dedicated_post_processing`:

- `auto` (default): the lane runs whenever post-processing is allowed (`allow_post_processing`).
- `on`: the lane always runs.
- `off`: no lane, and the worker does not offer post-processing at all. The lane is the only place
  post-processing runs; there is no inline fallback inside the inference processes.

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

Admission to the lane is governed so a chain the card cannot host never wedges it. The orchestrator admits
a job only when its estimated peak plus the VRAM reserve fits the lane card's free VRAM (with the budget on;
see [Performance and backpressure](performance_and_backpressure.md)). The manager refreshes the arbiter's
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
- **One-shot reclaim**: an unfittable job asks the scheduler to evict idle VRAM once per starvation episode,
  not once per scheduling tick, so deferral does not churn idle inference residency in a loop.

Pending post-processing work also owns lane liveness. If the queue is non-empty and no `POST_PROCESS` process
exists, the orchestrator asks lifecycle to start the lane; lifecycle still owns the actual admission checks
(the operator setting, GPU-start headroom, pending starts, and deliberate off-GPU pauses). If the lane remains
absent after that request, the same admission-patience clock runs so the job cannot sit in
`PENDING_POST_PROCESSING` forever. The exception is an active whole-card residency pause: that pause is expected
to restore itself when the resident model releases the card, so the patience clock stays unarmed while residency
is still active. If a whole-card pause is left behind after residency and inference work have drained, pending
post-processing restores that owner-scoped pause and then re-enters the normal lane-start path.

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
2. **Generate**: the scheduler dispatches to an inference process; the raw images return to the main process
   at the generate stage's completion milestone.
3. **Post-process** (when requested): the dispatcher queues the job for the lane; the post-processing
   orchestrator dispatches the first fittable pending job, sends the raw images and the requested operations,
   and the processed images replace the raw ones. A chain that never fits the card ages out to a no-image
   fault; an orphan watchdog requeues a job whose result was lost (bounded), then faults without images.
4. **Safety**: unchanged; the safety orchestrator sends the (possibly post-processed) images for evaluation.
5. **Submit**: unchanged; the chain closes out (`SUBMIT_COMPLETE`) when the job finalizes.

Graph alchemy forms (standalone upscale/facefix/strip_background jobs) ride the same lane via the
`ALCHEMY_GRAPH` capability and count as lane commitments while they are pending or in flight; CLIP forms
(caption/interrogation/nsfw) stay on the safety process. The alchemy coordinator owns its own pop/submit
loop, but the image popper reads its graph-lane commitment count for post-processing offer shaping.
