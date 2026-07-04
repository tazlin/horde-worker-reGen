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
  lane result arrives, when a retired result is known lost, or when orphan recovery requeues/falls back.
- The lane reports VRAM like an inference process. Its sample participates in `ProcessMap.get_free_vram_mb`,
  so the parent sees the same low-free-VRAM condition that would make ComfyUI tile or stream.
- Idle reclaim commands apply to the lane too: under pressure, the scheduler can ask an idle `POST_PROCESS`
  process to unload modules from VRAM/RAM. Active post-processing is never interrupted for reclaim.

Admission to the lane is governed so a chain the card cannot host never wedges it. The orchestrator admits
a job only when its estimated peak plus the VRAM reserve fits the lane card's free VRAM (with the budget on;
see [Performance and backpressure](performance_and_backpressure.md)), and three behaviors keep an unfittable
chain from parking the queue:

- **Queue scan**: the first *fittable* pending job is dispatched, so an unfittable head never blocks the
  fittable jobs queued behind it.
- **Aging escape**: a job that stays unfittable past the admission-patience window is submitted with its raw
  images, so its finished inference is delivered un-post-processed rather than deferred forever. This is an
  admission decision, not a lane fault, and does not feed the circuit breaker; the lane never ran the job.
- **One-shot reclaim**: an unfittable job asks the scheduler to evict idle VRAM once per starvation episode,
  not once per scheduling tick, so deferral does not churn idle inference residency in a loop.

A post-processing failure never forfeits the job: the raw inference images are still held by the main
process, so after bounded retries the job proceeds to safety with the raw images and the fault is recorded in
its generation metadata. Repeated lane faults feed the post-processing circuit breaker
(`post_processing_fault_breaker_enabled`), which stops advertising post-processing before the horde forces
maintenance.

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

A terminal fault marks the executing node failed and skips everything downstream; a job that falls back
(e.g. raw images after a post-processing failure) records the fault in metadata while the chain reflects the
stages that actually ran.

## The job's path through the lanes

1. **Pop**: the job registers with the tracker; its chain flow is built from the payload (post-processing
   requested or not).
2. **Generate**: the scheduler dispatches to an inference process; the raw images return to the main process
   at the generate stage's completion milestone.
3. **Post-process** (when requested): the dispatcher queues the job for the lane; the post-processing
   orchestrator dispatches the first fittable pending job, sends the raw images and the requested operations,
   and the processed images replace the raw ones. A chain that never fits the card ages out to a raw-image
   submit; an orphan watchdog requeues a job whose result was lost (bounded), then falls back to the raw
   images.
4. **Safety**: unchanged; the safety orchestrator sends the (possibly post-processed) images for evaluation.
5. **Submit**: unchanged; the chain closes out (`SUBMIT_COMPLETE`) when the job finalizes.

Graph alchemy forms (standalone upscale/facefix/strip_background jobs) ride the same lane via the
`ALCHEMY_GRAPH` capability; CLIP forms (caption/interrogation/nsfw) stay on the safety process. The alchemy
coordinator did not change: only the capability-to-process mapping moved.
