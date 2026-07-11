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
`enable_image_utilities` config flag and is off by default.

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

## Liveness and recovery

The adapter polls the service's health endpoint on the child heartbeat cadence and emits a heartbeat only
while the service answers. It reports `WAITING_FOR_JOB` when idle and healthy, and `ALCHEMY_STARTING` →
`ALCHEMY_COMPLETE` around an annotation, so the process map treats it like any lane doing alchemy work.

There is no bespoke silence watchdog for this lane. Instead, a service that stops answering health while its
subprocess is still alive (an unresponsive-but-not-dead hang) is converted into a recoverable death: after a
short grace window the adapter stops the subprocess outright. The handle then reports not-alive, and the
lifecycle's existing crash reaper recovers the lane through the same end → delete → start replacement machine
the post-processing lane uses. A subprocess that exits on its own is caught the same way.

The capability service launcher exposes its subprocess pid, so this lane surfaces an OS pid the same way a
spawned child does: the handle reports it, and every emitted state / heartbeat / memory / availability
message carries it as `reported_os_pid`. The parent adopts that value over its handle-derived pid, so
per-PID telemetry (WDDM paging attribution, the owned-PID registry) attributes the utilities subprocess
correctly. Teardown is still driven through the control pipe and the handle's terminate/kill.

## Job flow: pre-annotation, strip rerouting, pop gating

### ControlNet pre-annotation (availability-driven, in-graph fallthrough)

A ControlNet image job whose source image is not already a control map, and which did not request the
control map as its output, needs its control map derived before generation. Rather than run that annotation
in the worker's main venv, the control loop parks such a job in the `PENDING_ANNOTATION` stage and
dispatches `START_ANNOTATION` to an idle utilities lane process. A parked job is held out of the
inference-eligible set until its control map arrives; the derived control map (PNG bytes) is then stored on
the tracked job, the job returns to `PENDING_INFERENCE`, and the scheduler carries the bytes on the
inference control message so the inference child injects them and hordelib runs the `none` preprocessor over
them instead of re-annotating. At most one annotation is dispatched per idle lane process per control-loop
tick, so the single serial lane is paced and the rest of the queue keeps feeding inference.

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
strip that faults, ages out, or whose lane dies mid-pass is a **no-image fault** (the horde reissues the
job), never a silent submit of un-stripped images. This matches the post-processing lane, which likewise
faults without images a job whose requested post-processing could not run. As with the strip pass being
dispatched one per idle lane per tick, no job is ever left parked indefinitely: the bounded age-out is the
backstop, and a dead lane is caught immediately by the orphan reconcile.

### Pop gating

ControlNet annotation and background removal are advertised only when the lane can serve them, but on
different terms because their fallback stories differ. `capabilities.utilities_available` (the utilities
venv is provisioned and `enable_image_utilities` is not disabled) gates the config coercion, so a worker
without the lane provisioned never advertises `allow_controlnet` / `allow_sdxl_controlnet` and drops the
whole post-processing bucket (the API cannot accept upscale/face-fix while refusing `strip_background`).

Beyond that static provisioning gate the two features diverge at runtime:

- **`strip_background` has no in-graph fallback** (the main venv is purged of `rembg`), so its alchemy
  offer is additionally gated on a **healthy** utilities lane process existing right now: a provisioned but
  crashed or restarting lane withholds the `strip_background` offer until it recovers.
- **ControlNet pops are not gated on runtime lane health.** A provisioned lane that is momentarily down
  still pops controlnet work, because the in-graph preprocessor is a real dynamic fallback: the job is
  simply dispatched with no premade map. The provisioning coercion, not runtime health, is the controlnet
  gate.

`allow_extended_controlnet` carries a further, server-side gate on top of the operator opt-in
(`extended_controlnet` in `bridgeData`) and live annotator readiness. The `allow_extended_controlnet` pop
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
so `strip_background` finds its model where the service looks for it. ControlNet annotator checkpoints are
downloaded through the existing controlnet-annotator aux pass.

## VRAM admission

The lane is a GPU-context tenant, so it is admitted through the same free-VRAM headroom gate as the other
GPU lanes, with a provisional expected-footprint charge (`UTILITIES_PROCESS_EXPECTED_VRAM_MB`, a
rough-order-of-magnitude seed to be recalibrated) reserved on top of the generic requirement. When the card
lacks headroom the start is deferred and retried through the same drain-pending machinery as a deferred
inference or lane start, so an enabled lane never wedges the control loop on a pressured card.
