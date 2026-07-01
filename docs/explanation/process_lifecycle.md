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
5. The child immediately sends a `PROCESS_STARTING` state-change message to
   confirm it's alive.

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

| Stuck in…                        | Timeout                                |
| -------------------------------- | -------------------------------------- |
| Mid-inference                    | `inference_step_timeout`               |
| Preloading a model / starting up | `preload_timeout`                      |
| Downloading an auxiliary model   | `download_timeout`                     |
| Post-processing                  | `post_process_timeout + 3 × max_batch` |

(`process_timeout` and these timeouts are affected by performance modes.)

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

Every timeout above measures *silence*, the time since the last message or
heartbeat. That misses one wedge: a generation that loops on a single sampling
step without ever returning. ComfyUI keeps invoking the progress callback at
that step (in practice the **final** step, after a corrupt or incompatible
model+LoRA pairing), so the child keeps emitting heartbeats; the slot is never
silent, and the per-step timeout never fires. The slot would sit in
`INFERENCE_STARTING` indefinitely, holding VRAM and a queue slot. A healthy job
reports each step (including the last) exactly once, so the child counts
consecutive *non-advancing* progress reports and forwards the running count on
its heartbeats. Once it crosses `inference_stuck_step_repeat_limit` the
stuck-step watchdog reaps the slot despite its liveness. (The child cannot abort
the wedged call itself: hordelib swallows exceptions raised inside the progress
callback, so reaping is the parent's job.) The `detect_stuck_inference_step`
[log detector](../reference/logs.md) recognizes the reap line after the fact.

A reap in the **post-processing** state has a specific cause worth naming: a VRAM
over-commit. The upscaler/face-fixer peak lands after sampling and is never
charged against the job's placement, so on a contended card it allocates into
near-zero free VRAM and tile-thrashes silently until the
`post_process_timeout + 3 × max_batch` silence reaps the slot. Each such reap
feeds the post-processing fault breaker (see
[Resilience and recovery](resilience_and_recovery.md)); the
`detect_post_processing_vram_stall` detector attributes the reap line to the
over-commit. The preventative fix is the scheduler's active post-processing
reclaim (`post_processing_active_reclaim_enabled`, see
[bridge config](bridge_config.md#post-processing-vram-over-commit)), which frees
cross-process VRAM before the peak lands so the reap never happens.

When a process exceeds its timeout, it is **replaced immediately** within the
same call (see below); there is no separate notification sent to the message
dispatcher. After any recovery, a short `recently_recovered` guard suppresses
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
3. If inference was in progress on that process, the job **that slot was running**
   (its own `last_job_referenced`) is faulted via `JobTracker.handle_job_fault_now`
   as *retryable*: it returns to `PENDING_INFERENCE` for a fresh attempt while any
   remain, and only skips to `PENDING_SUBMIT` once the attempt budget is exhausted
   (see [Layer 1](resilience_and_recovery.md#layer-1-bounded-and-degraded-job-retry)).
   A job left stranded in progress despite this (e.g. a lost result) is caught by
   the [orphaned-job backstops](resilience_and_recovery.md#stranded-in-progress-jobs).
4. A new process is started with a fresh `process_launch_identifier`.
5. The `process_launch_identifier` bump ensures any stale messages from the dead
   process (still in the IPC queue) are discarded.

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
