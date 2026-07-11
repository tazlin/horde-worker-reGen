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

### ControlNet pre-annotation (availability-driven)

A ControlNet image job whose source image is not already a control map, and which did not request the
control map as its output, needs its control map derived before generation. Rather than run that annotation
in the worker's main venv, the job flow dispatches it to the utilities lane (`START_ANNOTATION`) once the
source images are downloaded; the job is not eligible for inference dispatch until the annotation result
arrives, and the derived control map (PNG bytes) is carried on the inference control message so the
inference child injects it and hordelib runs the `none` preprocessor over it instead of re-annotating.

Which control types are pre-annotated is **availability-driven**, not hardcoded. The utilities process
cannot necessarily serve every control type: a detector whose heavy backend is not importable in the lane's
environment (for example `seg` today), or whose weights are not pre-placed, is not servable. The adapter
polls `GET /annotators` (per-detector `available` + `weights_present`) on bring-up and on its memory
cadence, caches the servable set, and emits it as an annotator-availability snapshot. A control type in that
set is pre-annotated on the lane; anything **not** in it (unavailable backend, or missing weights) falls
through to hordelib's in-graph preprocessor, which runs today with no extra dependencies. `return_control_map`
jobs for an unservable control type fall through the same way (hordelib's annotation-only pipeline still
produces the map). The carve-out therefore shrinks automatically as the lane gains detectors, with nothing
to maintain per control type.

If a pre-annotation faults (the lane returns an error, or the utilities process dies mid-annotation), the
job is faulted through the normal job-fault path with metadata and reissued, never left parked.

### strip_background rerouting

`strip_background` runs on the utilities lane (its `rembg` stack never enters the main venv), so
`capability_for_alchemy_form` routes it to `IMAGE_UTILITIES`. A standalone `strip_background` alchemy form
is dispatched straight to the lane. Inside a generation job's post-processing chain, the pure-torch steps
(upscale, face-fix) still run on the post-processing lane; `strip_background` is then applied last on the
utilities lane, preserving the request ordering, before the job proceeds to safety and submit. The flow
degrades gracefully to fewer hops when steps are absent (inference → post-processing → utilities → safety →
submit). A utilities failure at any hop faults the job rather than wedging it.

### Pop gating

ControlNet annotation and background removal are advertised only when the lane can serve them.
`capabilities.utilities_available` (the utilities venv is provisioned and `enable_image_utilities` is not
disabled) gates the config coercion, so a worker without the lane never advertises `allow_controlnet` /
`allow_sdxl_controlnet` and drops the whole post-processing bucket (the API cannot accept upscale/face-fix
while refusing `strip_background`). At runtime, offers are additionally gated on a **healthy** utilities
process existing, re-evaluated on process state changes rather than polled, so a crashed or restarting lane
withholds its offers until it recovers.

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
