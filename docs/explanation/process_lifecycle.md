# Process Lifecycle

- [Process Lifecycle](#process-lifecycle)
    - [Why a dedicated lifecycle manager?](#why-a-dedicated-lifecycle-manager)
    - [Process creation](#process-creation)
    - [Semaphores and locks](#semaphores-and-locks)
    - [Hung-process detection](#hung-process-detection)
    - [Process replacement](#process-replacement)
    - [Model preloading lifecycle](#model-preloading-lifecycle)
    - [See also](#see-also)

`ProcessLifecycleManager` owns everything related to starting, stopping,
monitoring, and replacing child processes. It is the only component that creates
`multiprocessing.Process` objects.

## Why a dedicated lifecycle manager?

Process management is cross-cutting: the inference scheduler needs processes to
be healthy, the safety orchestrator needs a safety process to exist, the
shutdown manager needs to kill everything, and the message dispatcher needs to
know when a process has died so it can discard stale messages. Without a single
owner, these responsibilities scatter across components and create ordering
dependencies that are hard to reason about.

## Process creation

When a process is started, the lifecycle manager:

1. Creates a new `multiprocessing.Pipe` for parent→child control messages.
2. Creates a `HordeProcessInfo` with the pipe, process type, ID, and a fresh
   `process_launch_identifier`.
3. Adds it to `ProcessMap`.
4. Launches a `multiprocessing.Process` targeting the appropriate entry point
   (`ProcessEntryPoints`).
5. The child sends `PROCESS_STARTING` state-change messages as it initialises, to
   confirm it's alive.

A cold start is the one window in which a child cannot heartbeat: the heartbeat
and memory-reporter threads only start with its main loop, and there is no
intermediate state between `PROCESS_STARTING` and readiness to announce. Its only
liveness signal is therefore a repeat of the state it is already in, which the
parent counts as a report (it refreshes the silence clock the watchdogs read, and
applies none of the transition side effects). Without that, the whole cold start
reads as one unbroken silence, and a start slower than `preload_timeout` is reaped
as "stuck starting" and respawned into the same window.

A child that dies before its log sink opens writes its traceback only to
`logs/bridge_<role>_<id>_startup.log` (`child_crash_capture.write_startup_crash`).
When the parent replaces a slot stuck starting, the "replacing it" line and the
recovery diagnostics line carry that file's exception summary, if this launch wrote
one. The record must be stamped after the launch was spawned and, when it names a
launch identifier, name this one. The file is appended across launches, and a
replaced child can still write to it after its successor started, so an older or
foreign record is never attributed to the slot being replaced.

Inference processes are started up to `max_inference_processes` (derived
`queue_size + max_threads`). Safety processes are started up to
`max_safety_processes` (typically 1).

### Startup hygiene: inherited `sys.argv`

The `spawn` start method restores the launcher's full `sys.argv` in each child,
but a worker child consumes none of it (its configuration arrives through the
entry point's arguments and IPC, not the command line). That inherited argv is
not inert: libraries loaded later can call `argparse.parse_known_args()` against
`sys.argv` at runtime, and abbreviation matching means an inherited flag can
ambiguously match one of their options and trigger `sys.exit(2)`. Because
`SystemExit` is a `BaseException`, the worker's `except Exception` guards miss
it, so the child exits with no fault message and no fatal signal, surfacing only
as an unexplained mid-inference process recovery. Each spawned entry point
therefore calls `neutralize_inherited_argv()` immediately after enabling crash
capture, reducing argv to the program name so no inherited flag can reach such a
parser. (The ComfyUI controlnet depth/normal annotator preprocessors are one
such runtime `sys.argv` reader.)

## Semaphores and locks

Four shared synchronization primitives gate process concurrency:

| Primitive              | Purpose                                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| `inference_semaphore`  | Limits processes running inference at once to `max_concurrent_inference_processes` (`max_threads`) |
| `vae_decode_semaphore` | Limits concurrent VAE decode operations across processes                                           |
| `disk_lock`            | Serializes model downloads to avoid disk contention                                                |
| `aux_model_lock`       | Serializes auxiliary model (ControlNet, LoRA) loading                                              |

These are standard `multiprocessing` primitives; they work across process
boundaries.

## Hung-process detection

Each control-loop tick, `replace_hung_processes()` checks every process for
being stuck. Processes continuously report progress via heartbeats
(`HordeProcessHeartbeatMessage`) and other status messages; "stuck" is measured
as `min(now − last_received, now − last_heartbeat)` exceeding a **per-state**
timeout:

| Stuck in…                        | Timeout                                | Default window                         |
| -------------------------------- | -------------------------------------- | -------------------------------------- |
| Mid-inference                    | `inference_step_timeout`               | `20s` base, widened as described below |
| Preloading a model / starting up | `preload_timeout`                      | `150s`                                 |
| Post-processing                  | `post_process_timeout + 3 × max_batch` | `120s + 3 × max_batch`                 |
| Running an alchemy form          | `post_process_timeout + 3 × max_batch` | `120s + 3 × max_batch`                 |
(`process_timeout` and these timeouts are affected by performance modes.) No child
process downloads model weights: the post-processing lane faults a job whose
upscaler or face fixer is not on disk, and the download process fetches it, so no
stuck-downloading row exists. `download_timeout` bounds a popped job's wait for its
auxiliary prefetch instead (see
[Model downloads](model_downloads.md#pop-time-auxiliary-prefetch)).

The mid-inference timeout is not a single flat value. Before a job's first
sampling step (its `last_current_step` is still `None`) the slot is doing
one-time pre-sampling work (streaming a large checkpoint through VRAM, the
initial prompt encode) that emits no step, so the longer
`inference_first_step_timeout` applies. Once sampling has started it tightens to
`inference_step_timeout`, but that flat value suits a light job on an
uncontended device. On a multi-process worker two healthy cases legitimately go
heartbeat-silent for longer with no sampling step: a step stretched by
co-residence contention, and a feature phase that emits no step for its duration
(the hires-fix second pass, VAE decode, post-processing setup, a ControlNet
graph). The watchdog therefore widens the per-step grace up to
`contended_step_timeout` when there is positive evidence of such work (a
non-step pipeline phase is running, the slot was graded contention-slowed, or
the job's signature is feature-heavy), otherwise scaling the grace with the
job's expected sampling time. A genuinely wedged slot is still reaped once it
has been continuously silent past that bound. An over-budget admit keeps its own
`overbudget_step_timeout`.

Alongside the hard timeout, `_grade_running_inference()` runs each tick as a
soft, advisory ladder: it logs and audits a job sampling measurably slower than
its [performance-model](performance_and_backpressure.md#performance-model-scoring) expectation, escalating
`current_job_slowdown_level`. It measures sampling time **from the first step**,
not from dispatch, so a long cold start or feature-heavy startup is never
mislabelled as slow sampling; it only logs and feeds the widened timeout above,
and never replaces a slot itself.

Most timeouts above measure *silence*, the time since the last message or
heartbeat. Alchemy is the exception: its synchronous backends emit periodic
heartbeats to distinguish a live call from a dead child, while the watchdog
measures total time in `ALCHEMY_STARTING` so a live but paging/deadlocked backend
still has a hard bound. A silence-only timeout also misses another wedge: a
generation that loops on a single sampling
step without ever returning. ComfyUI keeps invoking the progress callback at
that step, so the child keeps emitting heartbeats; the slot is never silent, and
the per-step timeout never fires. The slot would sit in `INFERENCE_STARTING`
indefinitely, holding VRAM and a queue slot. The child therefore counts
consecutive *non-advancing* progress reports and forwards the running count on
its heartbeats. Once it crosses the effective ceiling
(`inference_stuck_step_repeat_limit` below the final step, twice the step count
at it, where an adaptive solver legitimately overshoots: see
[the watchdog's final-step allowance](resilience_and_recovery.md#the-stuck-step-watchdog-and-its-final-step-allowance))
the stuck-step watchdog reaps the slot despite its liveness. (The child cannot abort
the wedged call itself: hordelib swallows exceptions raised inside the progress
callback, so reaping is the parent's job.) The `detect_stuck_inference_step`
[log detector](../reference/logs.md) recognizes the reap line after the fact.

A reap in the **post-processing** state most often means the post-processing peak
or result handoff went silent for too long. A common cause is VRAM over-commit:
the upscaler/face-fixer peak lands after sampling and is never charged against
the job's placement, so on a contended card it allocates into near-zero free VRAM
and tile-thrashes silently until the `post_process_timeout + 3 × max_batch`
silence reaps the slot. The same watchdog also covers a child that completed the
post-processing pipeline but then stalls while packaging or sending the final
result to the parent. Each such reap feeds the post-processing fault breaker (see
[Resilience and recovery](resilience_and_recovery.md)); the
`detect_post_processing_vram_stall` detector attributes the reap line to the
post-processing stall. Post-processing runs on the dedicated lane (see
[Process lanes and job chaining](process_lanes_and_chaining.md)), so a reap replaces
only that lane; the affected job is requeued by the orphan watchdog and, after
bounded re-attempts, is reported as a no-image fault so the horde can reissue it.

One of these conditions is conditional on the horde having work. A slot silent in
`WAITING_FOR_JOB` "while there is work to do" is only stuck if there *was* work,
so that condition alone is suppressed while the last completed pop reported no
jobs available, and a genuine lull cannot churn healthy idle processes. The
suppression requires that evidence to be **fresh**
(`WorkerState.pop_no_jobs_evidence_fresh`, a 60s window): the no-jobs flag
describes the last pop attempt that reached the horde, so a worker whose pop loop
has stopped attempting freezes it at its last value. Without the freshness
requirement a pool whose every slot wedges in `PROCESS_STARTING` would silence the
popper (it never reaches the API), which would in turn disable the only watchdog
that could free a slot. The startup, preload, and post-processing conditions are
never suppressed: those wedges have nothing to do with what the horde is sending.

The pop loop's own silence is disclosed separately. Each control-loop tick,
`_check_pop_liveness()` compares now against the last completed pop attempt; past
60 seconds it logs one warning naming the gate the pop coroutine is currently
held at (`WorkerState.last_pop_gate`, for example `no_inference_process` or
`queue_full`) and how long that gate has held, escalating once to an error at 300
seconds. It stays quiet when intake is deliberately paused worker-wide or a
tracked [pop governor](performance_and_backpressure.md) has an open spell, since
both already account for the silence. The pop error-backoff spell is the one
exception: it only stretches the pop cadence to a few seconds and closes solely
when an attempt completes, so it can never account for a total absence of
attempts and is not accepted as an explanation. A silence with *no* gate named means a pop
request is still outstanding or the loop is gone, and the line says so. The
sentinel only discloses; recovery belongs to the watchdogs above.

When a process exceeds its timeout, it is ended immediately within the same call
(see below); there is no separate notification sent to the message dispatcher.
The replacement start normally happens in that same pass, but GPU-bearing
children are first admitted against the card's device-free headroom. If the card
is under pressure, the start is queued as a deferred GPU start and retried on
later control-loop ticks instead of spawning a fresh CUDA context into an already
tight device. After any recovery, a short `recently_recovered` guard suppresses
repeated replacements.

If **every** process is unresponsive past `process_timeout`, a global "hung"
state is entered. After roughly 20 s of that, the worker purges outstanding jobs
and either aborts (when `exit_on_unhandled_faults` is set, or it is already
shutting down) or replaces all inference processes.

## Process replacement

When a process dies unexpectedly (crash, OOM kill, hung timeout):

1. The old process entry is removed from `ProcessMap`.
2. Any model ownership entries in `HordeModelMap` tied to that process are
   cleared.
3. If inference was in progress on that process, the job **that exact launch was running**
   (its typed execution-ownership record) is faulted via `JobTracker.handle_job_fault_now`
   as *retryable*: it returns to `PENDING_INFERENCE` for a fresh attempt while any
   remain, and only skips to `PENDING_SUBMIT` once the attempt budget is exhausted
   (see [Layer 1](resilience_and_recovery.md#layer-1-bounded-and-degraded-job-retry)).
   A job left stranded in progress despite this (e.g. a lost result) is caught by
   the [orphaned-job backstops](resilience_and_recovery.md#stranded-in-progress-jobs).
   The one exception is the stuck-step watchdog's final-step overtime reap, which
   passes `sampler_overtime_reap` so the job is faulted non-retryably: that reap is
   a verdict on the payload, which the next slot would burn on identically (see
   [the stuck-step watchdog](resilience_and_recovery.md#the-stuck-step-watchdog-and-its-final-step-allowance)).
   A preload intent is deliberately excluded: preparation associates a model with a
   job for scheduling and display, but it does not prove an inference attempt began.
4. A new process is started with a fresh `process_launch_identifier`, or queued
   as a deferred GPU start if the assigned card is below its start headroom.
5. The `process_launch_identifier` bump ensures any stale messages from the dead
   process (still in the IPC queue) are discarded.

Deferred GPU starts apply to inference slots and GPU-resident auxiliary lanes
(safety-on-GPU, post-processing, component, and VAE lanes). The admission floor is
the device-free governor's soft floor plus one measured/estimated CUDA context
cost, so a respawn waits until there is enough room both to stay out of the
saturation band and to pay the context it is about to create. Missing device-free
telemetry is permissive: on hosts that cannot report a trustworthy device-level
free value, lifecycle falls back to the old immediate-start behavior rather than
inventing a floor. Each deferred start records `PROCESS_START_DEFERRED` in the
action ledger and is drained by `replace_hung_processes()` before ordinary hung
detection runs.

Headroom is the only thing that retries a deferred start, and a card whose
pressure comes from a tenant the reclaim ladder cannot evict (a worker-managed
text backend, another program) never reaches the floor. A deferred **safety**
start therefore escalates: once it has waited `PENDING_GPU_START_NO_PROGRESS_SECONDS`
(600 s) with no free-VRAM progress on its card, the safety process starts on the
CPU instead. The escalation goes through the existing off-GPU placement: it is
recorded as a runtime safety placement pause, so
[runtime safety placement](vram_arbiter.md#runtime-safety-placement) returns
safety to a card that permits it once that card shows durable room, under the
same restore dwell a pressure demotion earns. It logs one WARNING naming the
card, its free reading, the start requirement and the wait, and records
`SAFETY_START_ESCALATED_TO_CPU` in the action ledger. Drain progress inside the
window (a free reading rising by at least `PENDING_GPU_START_PROGRESS_EPSILON_MB`)
keeps the start waiting for the GPU.

The window covers only a shortfall the card could drain. When the start
requirement exceeds the card's achievable room (its total less the admission
margin and its standing foreign floor, which includes the worker's managed text
backend on its card; see [the VRAM arbiter](vram_arbiter.md#the-decision-pipeline)),
no eviction can meet it and the wait is structural. The next drain escalates a
safety start to the CPU, or retires an inference slot, exactly as the window
would. The WARNING and the ledger reason say "structural shortfall" and carry
the requirement and the room, and the deferral reason logged when the start is
first deferred carries the same note. A shortfall below the room keeps waiting.

A deferred **inference** start has no CPU fallback. After the same window with no
progress it is retired, provided another inference lane exists:
the pending entry is dropped, the slot comes off its card's `target_process_count`
(the per-card plan the scheduler reads) and off the worker-wide ceiling, so no pool
start or scale-up places it on that card again, and the worker serves on its other
lanes. It logs one WARNING naming the card, its free reading, the start
requirement, the wait and the card's new lane count, and records
`INFERENCE_START_RETIRED` in the action ledger. The slot is not moved to another
card, since each card's target is sized from that card's own config. The worker's
last inference lane is never retired: it stays pending, and once the window has
passed `pending_gpu_starts_backing_off()` stops excusing it, so the recovery
escalation treats the pool normally.

A retired slot comes back on its own card once that card has durably had room for
it (`restore_retired_inference_slots`, run each control tick beside the
deferred-start drain). The condition is the one a start needs: measured free at or
above the start requirement, a `HEALTHY` governor, and the requirement within the
card's achievable room. It must hold without a break for
`RETIRED_INFERENCE_SLOT_RESTORE_DWELL_SECONDS` (half the no-progress window), and
any reading that breaks it restarts the clock, so memory a lane frees between jobs
does not bring the slot back into a start that defers again. A structural shortfall
never restores. The restore raises the card's target by one, refreshes the
worker-wide ceiling, records `INFERENCE_SLOT_RESTORED`, logs one INFO line and
starts the slot on that card. One slot per card is restored per dwell. This matters
most on a single card, where the retired slot is the second lane on the only card:
it returns without a restart once the tenant that took its room leaves. Nothing is
restored while image generation is not served, including after a runtime CPU-only
torch build.

Every change to the per-card targets goes through `refresh_max_inference_processes`,
which publishes the new worker-wide ceiling to one listener. The process manager
updates its own `max_inference_processes`, the scheduler's and the job popper's
copies from it, so pricing lookahead, affinity and the planned-count budget split
follow a retirement or a restore. The runtime CPU-only collapse caps each card at
the smaller of its target and one (`cap_inference_targets`): a card that retired its
only lane stays at zero, since the collapse only ever removes lanes. The
recovery coordinator counts retired and quarantined slots together against the
planned ceiling, so a worker whose remaining lane is quarantined after the others
were retired reads as an unrecoverable pool.

A supervised safety rebuild (`rebuild_safety_pool`, which the soft reset runs)
is a fresh placement. On a multi-card host its respawn re-runs the scheduler's
headroom card choice instead of keeping the card the previous process was pinned
to, so a card that has since filled with a tenant the reclaim ladder cannot evict
is not chosen again while another card has room. Each card's `safety_on_gpu`
permission and whole-card residency apply as at any bring-up. A crash respawn
keeps its card, and a single-card host is unchanged.

Each replacement is also reported to a **process-recovery observer**
(`set_process_recovery_observer`); the process manager wires this to
`WorkerRunMetrics.record_process_crash`, so every crash/hang/replacement lands in
the run-metrics snapshot (process id, launch identifier, last state, reason) for
the benchmark and e2e harness to inspect.

The headline `_num_process_recoveries` counter is cumulative for the worker's lifetime (it is only
ever incremented). The warm benchmark worker reuses one process pool across levels, so the manager
zeroes it at each level boundary via `ProcessLifecycleManager.reset_recovery_counter()` (called from
`install_benchmark_scenario`, alongside `WorkerRunMetrics.reset()`); otherwise the first level to
recover would leave every later level reading a non-zero count it never earned. The slot-recovery
*history* behind the crash-loop breaker is deliberately left intact across that reset, so a genuine
crash loop spanning levels is still caught.

The safety process gets special treatment: if it dies, the
`safety_processes_should_be_replaced` flag is set, and any jobs in
`jobs_being_safety_checked` are requeued to `jobs_pending_safety_check`.

The safety, post-processing, image-utilities, component and VAE lanes are replaced
over several control-loop ticks (end, retire, start), and a replaced child is ended
as an OS process before its map
entry is retired, whatever state it was in. Retiring an entry only forgets the
child, and every teardown path walks the process map, so a child that was never
ended would outlive both its replacement and the parent. A child still in
`PROCESS_STARTING` has not reached its control loop and cannot read `END_PROCESS`,
so it is terminated. A child past startup was sent `END_PROCESS` when the
replacement began and gets the same end grace an inference slot does. A straggler
is killed. Asking a lane to end does not drop its child from the owned-PID
registry. A child confirmed dead leaves the registry and records
`PROCESS_ENDED`; one the kill could not end stays in the registry. The shutdown
reap and the hard kill both kill whatever the registry still holds outside the map
before clearing it, so a front end that drives the main loop directly, without the
atexit backstop, still ends those children. A safety
placement change also replaces the process, but that is intentional rather than
crash recovery. Runtime fit policy, whole-card residency, and verified reclaim do
not issue those replacements independently: they contribute demand to the
scheduler's one placement reconciler. The resulting pause records a `PauseOwner`,
and only that reconciler may restore it after every remaining request and restore
veto has cleared.

An intentional placement replacement must reach a loaded state before the
reconciler may issue a contrary placement change. This prevents a slow but healthy
CPU-only or GPU respawn from being replaced again inside its still-open intentional
window. The window remains bounded for genuine crash-on-start churn; readiness
clears both its unready-rebuild count and the independent consecutive-start-failure
streak, after which later unexpected safety replacement counts normally.

Readiness is also where the **cost of one placement flip** is measured. The manager
times each safety start to the first readiness that follows it and keeps a running
average (`safety_readiness_latency_seconds`), floored for a cold start and capped so
one pathological start cannot distort it. That figure is what the scheduler's
placement policy prices its dwells from: an eviction that leaves the worker without
an on-GPU safety process for the length of a rebuild has to be justified by pressure
that lasted about that long. See
[VRAM arbiter](vram_arbiter.md#runtime-safety-placement).

## Model preloading lifecycle

Model loading is a multi-step operation managed cooperatively by the inference
scheduler and the lifecycle manager:

1. **Scheduler** picks a job, determines the required model, finds a free
   process via `ProcessMap.get_first_available_inference_process`.
2. **Scheduler** sends `PRELOAD_MODEL` to that process via its pipe.
3. **Scheduler** marks the model `LOADING` in `HordeModelMap`.
4. **Child process** downloads the model (if needed), loads it into RAM, then
   into VRAM, sending `ModelLoadState` change messages at each step.
5. **Message dispatcher** updates `HordeModelMap` as each
   `HordeModelStateChangeMessage` arrives.
6. When the model reaches `LOADED_IN_VRAM` or `IN_USE`, it's eligible for
   inference dispatch.

Residency is recorded twice: on the slot (`HordeProcessInfo.loaded_horde_model_name`) and in the
model-keyed `HordeModelMap`. Loading a slot over rewrites the first, so the scheduler reconciles the
second on every preload pass (`_expire_stale_model_map_entries`): an entry naming a slot that now holds
a different model describes weights nothing holds and is expired. This matters for liveness rather than
tidiness, because the preload pass counts that map in its already-loaded set; a surviving entry makes the
displaced model's pending job look served, so it is never staged again and its job can wedge behind a
dispatch hold held for it.

Model **unloading** works in reverse: the scheduler picks models to evict based
on an LRU-informed heuristic, sends `UNLOAD_MODELS_FROM_VRAM` /
`UNLOAD_MODELS_FROM_RAM`, and the child acknowledges with state-change messages.

## See also

- [IPC and Messaging](ipc_and_messaging.md): the messages this manager sends
  and receives
- [Performance and Backpressure](performance_and_backpressure.md): model
  eviction and LRU policy
- [Shutdown and Faults](shutdown_and_faults.md): how processes are killed
  during shutdown
- [`ProcessLifecycleManager`][horde_worker_regen.process_management.lifecycle.process_lifecycle.ProcessLifecycleManager]
- [`ProcessMap`][horde_worker_regen.process_management.lifecycle.process_map.ProcessMap]
