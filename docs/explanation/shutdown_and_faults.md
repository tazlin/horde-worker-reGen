# Shutdown, Faults, and Error Recovery

- [Shutdown, Faults, and Error Recovery](#shutdown-faults-and-error-recovery)
    - [Shutdown vs. abort](#shutdown-vs-abort)
        - [Graceful shutdown sequence](#graceful-shutdown-sequence)
        - [Abort sequence](#abort-sequence)
    - [Signal handling](#signal-handling)
    - [Fault propagation](#fault-propagation)
    - [Consecutive failure backoff](#consecutive-failure-backoff)
    - [Deadlock detection](#deadlock-detection)
    - [The `.abort` file](#the-abort-file)

The worker must never lose a job, even during a crash, a SIGINT, or a hung
process. This page explains how shutdown coordination, fault propagation, and
error recovery work together.

## Shutdown vs. abort

There are two termination paths:

| Path                  | Trigger                                                 | Behavior                                                       |
| --------------------- | ------------------------------------------------------- | -------------------------------------------------------------- |
| **Graceful shutdown** | SIGINT, SIGTERM, or `shutdown()` call                   | Finish in-progress jobs, submit all pending results, then exit |
| **Abort**             | Three SIGINTs, `.abort` file created, or `abort()` call | Purge all jobs, hard-kill all processes, exit immediately      |

### Graceful shutdown sequence

1. `WorkerState.initiate_shutdown()` sets `shutting_down = True`.
2. `JobPopper` sees the flag and stops popping new jobs.
3. `InferenceScheduler` stops dispatching new inference jobs.
4. `SafetyOrchestrator` stops dispatching new safety checks.
5. `AlchemyCoordinator` stops its pop/dispatch/submit loop (it checks the
   shutdown manager each iteration).
6. `JobSubmitter` continues submitting pending results.
7. When all jobs are finalized (all stage collections empty), the main loop
   exits.
8. If jobs don't drain within a grace period, `start_timed_shutdown()`
   force-kills all processes.

### Abort sequence

1. `JobTracker._purge_jobs()` clears all job collections (jobs are lost).
2. `ProcessLifecycleManager._hard_kill_processes()` kills all children
   immediately.
3. `start_timed_shutdown()` launches a background thread that `sys.exit(1)`
   after a grace period; a last-resort measure if the main loop is stuck.

## Signal handling

`ShutdownManager.signal_handler` is registered for `SIGINT` and `SIGTERM`. The
first two signals initiate graceful shutdown. The third signal triggers an
immediate `sys.exit(1)`. This gives the operator a way to escalate: Ctrl+C once
for graceful, three times for "I mean it."

## Fault propagation

Jobs can fault at any stage. The fault-propagation chain is:

1. **Source image download failure** → `SourceImageDownloader` records a
   `GenMetadataEntry` fault keyed by `GenerationID`. The job still proceeds to
   inference (with a placeholder/missing image).
2. **Inference failure** (child crash, OOM, model error) →
   `JobTracker.handle_job_fault` resolves it: the job is **requeued for another
   attempt** if it has any of its `max_inference_attempts` budget left (a
   resource/OOM fault gets one degraded, isolated retry), and only reported
   faulted to the API once attempts are exhausted. See
   [Resilience and Recovery → bounded and degraded job retry](resilience_and_recovery.md#layer-1-bounded-and-degraded-job-retry).
3. **Safety evaluation failure** (classifier crash, missing fields) →
   `SafetyOrchestrator` faults the job terminally and queues it for submit
   (re-running inference cannot help a post-inference failure).
4. **Submission failure** (R2 upload timeout, API error) → `JobSubmitter`
   retries with exponential backoff per image.

At every stage, faults are recorded as `GenMetadataEntry` objects in
`JobTracker.job_faults`. The submitter includes all accumulated faults in the
API submission so the horde can track why a job failed.

## Consecutive failure backoff

`WorkerState` tracks `consecutive_failed_jobs`. After 3 consecutive failures,
`JobPopper`'s `_handle_consecutive_failures` gate pauses popping for
`CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS` (180 s; the constant is defined in
`pop_throttler.py`). This prevents a broken worker from rapidly consuming and
failing jobs. (With `exit_on_unhandled_faults` enabled, the worker shuts down
instead of pausing.)

Successful submissions reset the counter.

## Deadlock detection

Two independent mechanisms guard against the worker getting stuck:

1. **Hung-process recovery**: heartbeat- and timeout-based detection in
   `ProcessLifecycleManager` that actually _replaces_ stuck processes (see
   [Process Lifecycle](process_lifecycle.md#hung-process-detection)).
2. **Deadlock diagnostics**: `MessageDispatcher.detect_deadlock` inspects
   job/process state and logs when all inference processes are idle while jobs
   are still pending inference, or when jobs are tracked but no process is busy.
   These checks are purely informational: they emit diagnostics after a short
   grace period and do **not** kill or replace anything.

## Worker-wide recovery

The mechanisms above handle *individual* faults and stuck processes. A separate
escalation layer handles the case where the worker **as a whole** has stopped
making progress: a "save-our-ship" supervisor first soft-resets the process pools
in place (rebuild every child, reduce concurrency a notch for "limp-by"), and
only if that clearly is not helping does it give up cleanly on jobs it cannot
serve (faulting them so the horde reissues them) while continuing to run. Two
durable records (the action ledger and the owned-PID registry) support diagnosis
and orphan cleanup across crashes. These are covered in full in
[Resilience and Recovery](resilience_and_recovery.md).

## The `.abort` file

Writing any content to `.abort` in the worker's working directory triggers an
immediate abort on the next control-loop tick. This is a convenience for
external process managers (systemd, Docker, etc.) that can't send signals
easily.

## See also

- [Resilience and Recovery](resilience_and_recovery.md): bounded/degraded retry,
  crash-loop quarantine, the SOS escalation, and the durable diagnostic records
- [Architecture](architecture.md): the five asyncio tasks this machinery halts
- [Job State Machine](job_state_machine.md): how faults interact with job
  stages
- [Process Lifecycle](process_lifecycle.md): process killing and replacement
- [`ShutdownManager`][horde_worker_regen.process_management.shutdown_manager.ShutdownManager]
- [`WorkerState`][horde_worker_regen.process_management.worker_state.WorkerState]
