# Image utilities lane

The image-utilities lane runs the `horde_image_utilities` capability service: the home of heavier image
features (ControlNet annotators, background removal) whose native, accelerator-gated dependency stack the
worker deliberately keeps out of its main environment. Unlike every other lane, this one is not a
multiprocessing child; it is a subprocess launched from a *second virtual environment*, bridged into the
ordinary child contract by a parent-side adapter. This page explains what it is, why it runs the way it
does, and how the worker supervises it.

## Its own virtual environment

The capability service is launched as `python -m horde_image_utilities` using the interpreter from a second
venv (`worker_bootstrap.paths.utilities_python`), provisioned separately from the worker's own `.venv`. That
isolation is the point: the annotator and background-removal stack can
carry native, accelerator-matched dependencies that would otherwise have to co-resolve with the worker's
own torch build. Because the two environments never share a resolution, one cannot constrain or break the
other.

The service is a small uvicorn server bound to a loopback ephemeral port. The worker talks to it only over
loopback HTTP, so the consuming side stays lean: the worker's main environment carries a dependency-free
client, never the server's stack.

### How the venv is provisioned

The bootstrap brain (`worker_bootstrap`) builds and maintains the utilities venv; the worker process only
checks that its interpreter exists and never creates it. Provisioning runs after a successful worker sync
and is also repaired on a launch that finds the main `.venv` already current, so an install that gained the
utilities project in a later release still converges without a forced resync. It is best-effort: a failure
warns but never turns an otherwise-complete worker install into a reported failure.

The dependency set is a self-contained uv project committed at `requirements/utilities/` (a `pyproject.toml`
plus a checked-in `uv.lock`, one lock with a build extra per locked torch build), shipped in the release
bundle alongside the worker. The bootstrap provisions the venv with `uv sync --locked --extra <build>`,
pointing `UV_PROJECT_ENVIRONMENT` at the peered utilities venv. Provisioning is skipped for a backend whose
resolved feature set is empty (a lean build that did not opt in), and when a stamp shows the venv already
matches the committed lock. When the lane *is* wanted for a full build but the lock is absent (an install
that never received the `requirements/utilities/` directory), the bootstrap warns and disables the lane
rather than skipping silently, so the cause is diagnosable instead of surfacing only as a missing interpreter
at start.

Provisioning is also deferred while the worker's own environment is limping along on an older torch than the
utilities lock pins (a held optional update): syncing the lane from that lock would download the very torch
the hold declined, and leave the two environments on different builds. The deferral ends by itself, on the
first sync that takes the newer torch. An existing utilities venv remains unchanged during the deferral; if
the venv has not been provisioned yet, the capabilities that need it remain unavailable until then.

Provisioning from a lock is what makes the two environments share a single download. A locked sync installs
strictly from the lockfile's recorded sources and never re-resolves, so it is deterministic (identical wheels
on every machine), reuses the main `.venv`'s already-cached torch/CUDA stack from the peered uv cache, and
can never be pulled onto a box's ambient extra-index (the failure a floating `uv pip install -r
requirements.txt` is prone to: re-resolving torch's CUDA dependencies onto a different, sometimes unreachable,
index). The lock pins `torch`/`torchvision` to the same build the root worker's `uv.lock` resolves, so the
cached wheels actually match; a guard test fails if the two drift. The sync runs through the same hardened
child environment as every other uv call, so it shares the worker's peered cache and managed CPython, and it
runs *before* the post-sync `uv cache prune` while that cache is still warm: a prune reclaims the reusable
unpacked wheels, and a venv's own hardlinks survive a prune but can no longer serve a fresh install, so
pruning first would force the utilities venv to re-download the exact wheels the main sync just fetched.
Building it first lets both environments share one download; the prune then tidies both. `uv sync` is
inherently non-interactive and idempotent, so a reprovision (only when the venv is absent or the lock
changed) reconciles the environment without prompting.

## The adapter bridge

Every other worker child speaks the [IPC message vocabulary](ipc_and_messaging.md) from inside its own
process loop. A separate-venv subprocess cannot: it does not import the worker's message types. The
`UtilitiesProcessAdapter` closes that gap from the parent side. It:

- Owns the capability-service subprocess (start, health-gated bring-up, teardown).
- Runs a control thread that drains the same control pipe the lifecycle sends on and translates each
  message into an HTTP call.
- Runs a cadence thread that polls the service's health and memory endpoints and emits the state,
  heartbeat, and memory messages the child would otherwise send itself.

The result is that the [process lifecycle](process_lifecycle.md), the process map, the message queue, and
the dashboard all see a perfectly ordinary child. Job routing keys on the `IMAGE_UTILITIES` capability, not
on the process type, so the lane is discovered the same way every other lane is. The lane is enabled by the
`enable_image_utilities` config flag, which defaults to true; the lifecycle still refuses to start it until
the separate environment has been provisioned.

### Control verbs

The adapter translates the standard control flags into the service's ops endpoints:

| Control flag | Action |
| --- | --- |
| `START_ANNOTATION` | POST the source image to `/annotators/{control_type}`, return the control map as an annotation result |
| `START_ALCHEMY` (strip_background) | POST the source image to `/rembg/remove-background`, return the WebP result as a standard alchemy result |
| `RELEASE_ALLOCATOR_CACHE` | POST `/ops/release-cache` (drop framework caches, keep models resident) |
| `END_PROCESS` | POST `/ops/shutdown`, then stop the subprocess |

A form runs one at a time inline on the control thread; the service enforces its own concurrency. The
`strip_background` alchemy result carries WebP `image_bytes` in exactly the shape the post-processing lane
emits, so the alchemy coordinator's submit path (R2 upload, then submit) consumes it without knowing which
lane produced it.

## When the lane runs

The lane is started only while work needs it. An idle lane still holds a CUDA context on the
card and spends tens of seconds of CPU priming its annotators at every start, and the reclaim ladder would stop
it again for the next post-processing chain that does not fit beside it.

One pure function, `utilities_lane_demand` in `scheduling/workload_flow.py`, returns whether the lane should be
up and why. Its rules, first match wins:

1. A lane that is disabled, not wanted by the configured workloads or not provisioned, or a worker shutting
   down, is never wanted.
2. An alchemist offering `annotation` before any lane has reported a servable set this session, or since the
   kept set was dropped. No offer keeps the lane up (see [Alchemy forms](#alchemy-forms)), but the annotation
   offer has to list control types and only a lane run reports them. Strips and ControlNet jobs depend on traffic, so
   no setting guarantees another start, and this rule starts the lane once to read the set. It lapses at the
   lane's first report. The extended-ControlNet opt-in does not keep the lane up either (see
   [Extended ControlNet](#extended-controlnet)).
3. A job in `PENDING_STRIP`.
4. A job pending or in inference whose post-processing is background removal alone. Its queue wait and
   generation cover a lane start. A job that also requests an upscaler or face fixer counts from rule 3 only,
   once its post-processing chain has left the post-processing lane, so a lane is not started for the chain and
   stopped again.
5. A ControlNet job that needs pre-annotation and has not been offered to the lane, or a job in
   `PENDING_ANNOTATION`. An extended-ControlNet job with a source image counts here like any other.
6. A lane-bound alchemy form pending dispatch or in flight. The form was offered and popped with the lane
   down, so this rule is what starts the lane for it.

Every start goes through `ProcessLifecycleManager.start_utilities_processes`, which refuses without demand: the
boot bring-up, a VRAM-deferred retry (dropped once the demand lapses), the replacement's final start after a
crash or recycle, and the reclaim ladder's restore. A crash with no demand therefore leaves the lane down and
still counts one recovery.

A per-tick driver, `reconcile_utilities_demand`, brings the lane up on the tick after work arrives that needs
it, whichever path the work arrived by (an image pop, an alchemy pop, a strip handed back after a lane loss). It
acts only after the boot bring-up, with no lane in the map, no pause, no replacement under way and no deferred
start pending, and a start that fails outright is tried again only once the demand lapses and returns. Each
start it makes is recorded as an `IMAGE_UTILITIES_ROUTING` decision on the subject `utilities_lane`, which
opens with `withhold`/`no_demand` while the lane is held down and resolves with `admit` and the demand that
started it.

The reclaim ladder's restore (`restore_utilities_off_gpu`) lifts its pause and discharges its obligation either
way, but starts the lane only with demand. Without it the lane stays down and the restore logs
`Reclaim ladder: leaving the image-utilities lane down until a job needs it.` once per stretch the lane stays
down. A lane that is down does not hold a context, so the arbiter does not offer a utilities rung for it. A ladder pause that
outlives every owner that could restore it (no saturation episode, post-processing borrow or recovery remedy
claims it, held for the stranded-lane debounce) is restored by the driver once a job needs the lane.

A lane that is down because no job needs it is not a failure. Recovery, the hung-process watchdog and the
crash reaper act on processes in the map only, the strip and annotation stages wait or fall through as they do
while the lane restarts, and the dashboard's process table lists the lane only while it runs.

### Alchemy forms

The alchemy forms that route to the lane (`strip_background`, `annotation`) cannot run anywhere else, but a lane that
is down by policy starts for a popped form, so the offer does not need a live lane. The alchemy pop gate
(`AlchemyCoordinator._utilities_lane_healthy`) offers them while a lane is loaded or while
`utilities_lane_will_serve` holds, and withholds them while the lane is paused, crashed, refused for VRAM or
disabled.

`annotation` also needs control types to list in the pop. The worker keeps the servable set the lane reported on
its last run (`HordeWorkerProcessManager._kept_servable_set`) and offers it while the lane will serve and no live
lane has reported. A crash or hang replacement and a disabled lane drop the kept set, so a lane that comes back
reports its set afresh before annotation is offered from it. Before the lane has reported once there is no set,
which is rule 2 above.

A popped lane-bound form is lane demand under rule 6, so the driver starts the lane on the next tick. At
dispatch (`AlchemyCoordinator._utilities_lane_fault_cause`) a form with no loaded lane waits while the lane will
serve, bounded by `dispatch_affinity.utilities_lane_hold_expired`, the clock extended-ControlNet jobs wait on.
The form's dispatch deadline (`PENDING_FORM_DISPATCH_DEADLINE_SECONDS`) stands in for its ttl and there is no
sampling figure, so the wait ends at the anti-starvation share of that deadline. A form whose lane start is deferred
for VRAM headroom (`ProcessLifecycleManager.utilities_lane_start_deferred`, outlook `START_DEFERRED`) waits on the
same clock, since the deferred-start drain starts the lane once the card has room. The offer stays withheld for the
length of the deferral. A form whose lane crashed, whose start failed or that was disabled is faulted back to the
horde at once. A form whose lane a reclaim or residency
owner holds paused, or whose loaded lane is busy with a form, annotation or strip, waits for the full dispatch
deadline.

### Extended ControlNet

Extended ControlNet types do not have an in-graph preprocessor, so an extended job can only run once the lane has
derived its control map. The lane is not kept up for them. `ProcessLifecycleManager.utilities_lane_will_serve`
returns whether a job may wait for the lane. It is true while the lane runs or starts, and while the lane is down
by policy (enabled, provisioned, not paused by a reclaim or residency owner, no start deferred for VRAM, no crash
replacement under way, no demand start that failed outright). A crashed, paused or refused lane is not waited
for.

The worker offers extended ControlNet at pop (`_extended_controlnet_ready`) while the live servable set covers
every extended type, or while the lane will serve and the extended annotator weights are on disk. A popped
extended job is lane demand under rule 5, so the driver starts a lane that is down on the next tick. A job already
popped is also held while the lane's start is deferred for VRAM headroom, on the same hold, while the offer is
withheld.

Each control-loop tick, before the annotation pass and the scheduling cycle,
`_fault_unready_extended_controlnet_jobs` sorts the queued extended jobs:

- A job whose type is in the live servable set, or that already carries its derived map, passes on to
  pre-annotation or inference.
- A job that brings its own control map (`image_is_control`) needs the extended-ControlNet feature and that map,
  and no annotator, since hordelib runs the preprocessor as `none`. It passes on to inference whatever the lane's
  state, because a lane state does not withdraw the offer the job was popped under. Only the feature being off
  faults it.
- A job that needs pre-annotation of a type the lane cannot serve yet is held while the lane will serve, no live
  lane has reported its servable set yet, and the type's weights are on disk. `InferenceScheduler.reconcile_utilities_lane_holds` records the hold in the
  `DispatchHoldLedger`, logs `Holding extended-ControlNet job <id> for the image-utilities lane, which cannot
  serve control type <type> yet.` and records an `INFERENCE_DISPATCH` decision (`withhold`,
  `awaiting_utilities_lane`) once per hold, resolved when the hold is released. Selection skips a held job, as it
  skips an aux-unprepared one, so the jobs behind it dispatch, and an empty slot behind a held head is
  attributed to the `utilities_lane_bring_up` duty bucket. The deadlock detector exempts held jobs.
- A held job past `dispatch_affinity.utilities_lane_hold_expired`, and every other extended job the worker
  cannot serve, is faulted non-retryably with `extended_controlnet_not_offered` so the horde reissues it. The
  hold lasts until the job's ttl from pop, less its expected sampling seconds from the performance model, less
  the skip budget (`affinity_budget_seconds`), which stands in for the tail after sampling. This is longer than
  the anti-starvation share, since a held job has not been offered a lane and there is no head to step aside.
  With no sampling figure the hold ends at the anti-starvation share. A job with no ttl is measured against
  `dispatch_affinity.fallback_ttl_seconds`, the most recent ttl the worker received or the skip budget's default. A live lane that has reported a set without the job's
  type is not waited for.

The weights check (`_extended_annotators_on_disk`) probes each extended type once and caches the result until the
download process reports a change in ControlNet or post-processor presence, so the pop offer and the per-tick
hold check read a cached result.

The hold is released on the tick the lane reports the job's type servable, and the annotation pass parks the job
for its map in the same tick.

## Liveness and recovery

The adapter polls the service's health endpoint on the child heartbeat cadence and emits a heartbeat only
while the service answers. It reports `WAITING_FOR_JOB` when idle and healthy, and `ALCHEMY_STARTING` →
`ALCHEMY_COMPLETE` around an annotation, so the process map treats it like any lane doing alchemy work.

There is no bespoke silence watchdog for this lane. Instead, a service that stops answering health while its
subprocess is still alive (an unresponsive-but-not-dead hang) is converted into a recoverable death: after a
short grace window the adapter stops the subprocess outright and reports `PROCESS_ENDING` then `PROCESS_ENDED`,
so the process map stops offering the lane at once. The handle then reports not-alive, and the lifecycle's
crash reaper recovers the lane through the same end → delete → start replacement machine the
post-processing lane uses. A subprocess that exits on its own is caught the same way. Dispatch to the lane
(`ProcessMap.get_first_available` for `IMAGE_UTILITIES`, and the annotation and strip passes) also requires a live
process, since a lane that died keeps its last reported state until the reaper runs.

The adapter handles each control message on the thread that reads the control pipe, and a form, annotation or
strip runs as an HTTP call with a timeout of minutes. A second image-bearing message sent meanwhile waits for that
read, and the sender is the parent's main loop. The lane therefore takes one unit of work at a time across all
three senders. Each sets `HordeProcessInfo.utilities_work_sent_at` on send, the result handler clears it
(`ProcessMap.on_utilities_work_result`), and so does the lane's end. A lane holding work is not selected and
`can_accept_job` is false for it. The mark does not follow the reported state, since the adapter can report idle
before it reads the message.

The capability service launcher exposes its subprocess pid, so this lane surfaces an OS pid the same way a
spawned child does: the handle reports it, and every emitted state / heartbeat / memory / availability
message carries it as `reported_os_pid`. The parent adopts that value over its handle-derived pid, so
per-PID telemetry (WDDM paging attribution, the owned-PID registry) attributes the utilities subprocess
correctly. Teardown is still driven through the control pipe and the handle's terminate/kill.

## Logging

The lane is not a hordelib child, so it never receives the per-slot `HordeLog` sinks (`bridge_<n>.log`,
`trace_<n>.log`, `stdout_<n>.log`, `stderr_<n>.log`) every spawned child gets, and the service itself
configures no logging of its own. Left alone its only output (uvicorn's stream logs plus whatever the
native stack writes) would inherit the orchestrator's file descriptors: commingled into
`bridge_main_console.log` when supervised, or lost to the terminal when not. To give it a durable,
attributable home, the launcher redirects the subprocess's stdout and stderr to a dedicated
`logs/bridge_utilities_<slot>.log`, keyed by the lane's process slot so a per-device multi-GPU deployment
keeps each lane's output separate. The file is opened append (a crash-reaper restart reuses the slot and
its path, so the prior run's tail survives) and is declared as a `raw_stream` family in the
[log-file registry][horde_worker_regen.log_file_registry], so the startup purge, the TUI Logs tab, and the
support bundle all recognize and group it.

## Job flow: pre-annotation, strip rerouting, pop gating

### ControlNet pre-annotation (availability-driven, in-graph fallthrough)

A ControlNet image job whose source image is not already a control map, and which did not request the
control map as its output, needs its control map derived before generation. Rather than run that annotation
in the worker's main venv, the control loop parks such a job in the `PENDING_ANNOTATION` stage and
dispatches `START_ANNOTATION` to an idle utilities lane process. A parked job is held out of the
inference-eligible set until its control map arrives; the derived control map (PNG bytes) is then stored on
the tracked job, the job returns to `PENDING_INFERENCE`, and the scheduler carries the bytes on the
inference control message so the inference child injects them and hordelib runs the `none` preprocessor over
them instead of re-annotating. An annotation goes only to a lane that does not hold another unit of work (see
[Liveness and recovery](#liveness-and-recovery)), so the single serial lane is paced and the rest of the queue
keeps feeding inference.

Which control types are pre-annotated is **availability-driven**, not hardcoded. The utilities process
cannot necessarily serve every control type: a detector whose heavy backend is not importable in the lane's
environment, or whose weights are not pre-placed, is not servable. The adapter polls `GET /annotators`
(per-detector `available` + `weights_present`) on bring-up and on its memory cadence and emits the servable
set as an annotator-availability snapshot; the parent caches it per lane process. A control type in that set
is pre-annotated on the lane; anything **not** in it falls through to hordelib's in-graph preprocessor,
which runs today with no extra dependencies. The set grows automatically as the lane gains detectors, with
nothing to maintain per control type.

The liveness contract is that a parked job always resolves back into the normal generation flow and is
never lost. Every failure mode releases it to **in-graph fallthrough** (returned to `PENDING_INFERENCE`
carrying no premade map, so hordelib annotates it during generation), not a fault:

- **Annotation fault**: the lane returns an error result; the job is released in-graph.
- **Age-out**: a park that outlives a bounded timeout (comfortably longer than the lane's own annotate
  timeout, so a legitimately slow result is not raced) is released in-graph.
- **Lane death or replacement**: the moment the owning lane process leaves the map (died, or was replaced),
  every job it owned is released in-graph, so no job is left parked behind a dead lane.

An **anti-ping-pong latch** guarantees each job is offered to the lane at most once: a job released by any
of the above is marked annotation-attempted and is never re-parked by a later scan, so it dispatches
in-graph exactly once-decided rather than oscillating between the stages.

### return_control_map: the map is the deliverable

When a job's requested output *is* the control map (`return_control_map`), a successful annotation is the
whole job: the map becomes the job's single image result and the job moves straight from `PENDING_ANNOTATION`
to the safety stage (the same post-generation path a normal completion takes), never touching an inference
process. This is served only when the control type is servable and the lane is healthy; otherwise, and on
any annotation fault, age-out, or lane death, the job falls through to `PENDING_INFERENCE` exactly like a
pre-annotation fallthrough, and hordelib's in-graph annotation-only path produces the map end to end. The
anti-ping-pong latch applies identically.

### strip_background rerouting

`strip_background` runs on the utilities lane (its `rembg` stack never enters the main venv), so
`capability_for_alchemy_form` routes a standalone `strip_background` alchemy form straight to the lane, and
its result reaches submit through the alchemy coordinator's normal path.

Inside a generation job's post-processing, background removal is the **last** image transform. It is split
from the pure-torch transforms: a job with any upscaler or face-fixer runs those on the post-processing lane
first (the lane's pass drops `strip_background`, which it can no longer run), and the job then enters the
`PENDING_STRIP` stage, where the control loop dispatches `START_BACKGROUND_STRIP` to an idle utilities lane
process. A job whose only post-processing is background removal skips the post-processing lane entirely and
goes straight to `PENDING_STRIP`. On success the stripped images move on to safety; the chain is
inference -> post-processing lane (upscale/face-fix) -> utilities strip (last) -> safety -> submit, dropping
the post-processing-lane hop when no pure-torch transform was requested.

Background removal has **no in-graph fallback**, so its liveness contract differs from pre-annotation: a
strip that faults or exhausts its budgets is a **no-image fault** (the horde reissues the job), never a
silent submit of un-stripped images. This matches the post-processing lane, which likewise faults without
images a job whose requested post-processing could not run. No job is left parked indefinitely, but the
write-off is deliberately hard to reach, because a reissue asks the horde to buy the work twice:

- **Execution budget** (180s): how long one strip may run, timed from the moment it is dispatched to a lane.
  A job's wait for its turn is never counted against its own pass.
- **Head-of-queue budget** (180s): a queued strip is clocked only once it reaches the head of the queue,
  the first queued strip with no dispatched strip ahead of it. Time spent behind a running pass is the lane's
  time and is not charged to the waiter. Each tick dispatches before it ages strips out, so a waiter is never
  faulted in the tick a lane frees for it. The clock restarts when a strip is handed back to the queue.
- **Stage cap**: one execution budget for each job the worker admits to intake at once (`queue_size` plus
  `max_threads`, summed across cards) plus the job's own pass, measured from stage entry. A queued strip is
  faulted once a fresh pass could no longer finish inside the cap, so a dispatched strip always keeps its full
  execution budget.
- **The breaker**: strip faults (a timeout, a service failure, a lost lane, the lane disabled) affect only the
  job that faulted. They do not feed the post-processing fault breaker, which counts VRAM over-commit
  only.
- **Lane replacement**: a lane going away does not spoil the job. Its images are still held, so the strip is
  re-queued for the replacement lane; only after a bounded number of lanes (two) is it written off. This is
  what the orphan reconcile does instead of faulting on sight.
- **Late results**: a strip that finishes after its job was written off is still adopted, as long as the
  job's delivery to the horde has not begun. Finished images serve the requester and earn the job's kudos,
  where the write-off only asks for the work to be done again. Once the submit is in flight the outcome is
  settled and the late result is discarded with the reason logged.

The lane-side watchdog respects the same accounting: the adapter never stops a lane that is busy with a
strip the worker dispatched and still considers inside its budget, so a slow pass cannot be mistaken for a
hang and killed for it.

### Pop gating

ControlNet annotation and background removal are advertised only when the lane can serve them, but on
different terms because their fallback stories differ. `capabilities.utilities_available` (the utilities
venv is provisioned and `enable_image_utilities` is not disabled) gates the config coercion, so a worker
without the lane provisioned never advertises `allow_controlnet` / `allow_sdxl_controlnet` and drops the
whole post-processing bucket (the API cannot accept upscale/face-fix while refusing `strip_background`).

Beyond that static provisioning gate the two features diverge at runtime:

- **`strip_background` has no in-graph fallback** (the main venv is purged of `rembg`), so its alchemy
  offer is additionally gated on a lane that is loaded or will serve (see [Alchemy forms](#alchemy-forms)). A
  lane that is down by policy keeps the offer, since a popped form starts it. A paused, crashed, refused or
  disabled lane withholds the offer until it recovers.
- **ControlNet pops are not gated on runtime lane health.** A provisioned lane that is momentarily down
  still pops controlnet work, because the in-graph preprocessor is a real dynamic fallback: the job is
  simply dispatched with no premade map. The provisioning coercion, not runtime health, is the controlnet
  gate.

`allow_extended_controlnet` carries a further, server-side gate on top of the operator opt-in
(`extended_controlnet` in `bridgeData`) and annotator readiness (see
[Extended ControlNet](#extended-controlnet)). The `allow_extended_controlnet` pop
field only exists on the AI Horde server from the release that ships extended ControlNet, and a server
that does not recognise the field rejects the whole pop. So the worker advertises it only once the
server-capability probe (see [`server_capabilities`][horde_worker_regen.server_capabilities]) confirms the
field is present on the server's `PopInputStable` schema. The gate is fail-closed: a worker talking to an
older server never offers extended ControlNet, and begins offering it within the probe TTL of the server
going live, no restart required.

## Weight pre-placement

The capability service runs with downloads disabled (`HIU_ALLOW_DOWNLOADS=false`), so it never fetches its
own weights. The worker's download process pre-places the rembg `u2net.onnx` weight into the lane's isolated
rembg cache (`AIWORKER_CACHE_HOME/horde/image-utilities/rembg`), verified against rembg's published checksum,
so `strip_background` finds its model where the service looks for it. The weight is an ordinary queued
download, not a side thread: it appears in the Downloads tab with its size and progress, and is retried and
reported like any other model. It is queued only for a worker that can serve background removal: one that
offers post-processing (`allow_post_processing`) or is an alchemist with the post-process form configured,
and has the utilities lane enabled (`enable_image_utilities`). ControlNet annotator checkpoints are queued by
the same [feature reconcile](model_downloads.md#feature-reconcile-queueing-what-is-missing).

## VRAM admission

The lane is a GPU-context tenant, so it is admitted through the same free-VRAM headroom gate as the other
GPU lanes, with a provisional expected-footprint charge (`UTILITIES_PROCESS_EXPECTED_VRAM_MB`, a
rough-order-of-magnitude seed to be recalibrated) reserved on top of the generic requirement. When the card
lacks headroom the start is deferred and retried through the same drain-pending machinery as a deferred
inference or lane start, so an enabled lane never wedges the control loop on a pressured card.
