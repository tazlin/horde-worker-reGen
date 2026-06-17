# Process Lifecycle

- [Process Lifecycle](#process-lifecycle)
    - [Why a dedicated lifecycle manager?](#why-a-dedicated-lifecycle-manager)
    - [Process creation](#process-creation)
    - [Semaphores and locks](#semaphores-and-locks)
    - [Hung-process detection](#hung-process-detection)
    - [Process replacement](#process-replacement)
    - [Model preloading lifecycle](#model-preloading-lifecycle)

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
3. If inference was in progress on that process, the job is faulted via
   `JobTracker.handle_job_fault_now`; it skips straight to `PENDING_SUBMIT`.
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
- [`ProcessLifecycleManager`][horde_worker_regen.process_management.process_lifecycle.ProcessLifecycleManager]
- [`ProcessMap`][horde_worker_regen.process_management.process_map.ProcessMap]
