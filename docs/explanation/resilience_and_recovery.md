# Resilience and Recovery

- [Resilience and Recovery](#resilience-and-recovery)
    - [The layered recovery model](#the-layered-recovery-model)
    - [Layer 1: bounded and degraded job retry](#layer-1-bounded-and-degraded-job-retry)
    - [Layer 2: slot replacement and crash-loop quarantine](#layer-2-slot-replacement-and-crash-loop-quarantine)
    - [Layer 3: save-our-ship (SOS) escalation](#layer-3-save-our-ship-sos-escalation)
    - [The action ledger](#the-action-ledger)
    - [The owned-PID registry](#the-owned-pid-registry)
    - [Fault injection and chaos testing](#fault-injection-and-chaos-testing)
    - [See also](#see-also)

The worker's overriding goal is to **keep serving jobs**. A crash in one inference
slot, a wedged model load, or even a hard kill of the parent process should not
take the whole worker down or leave a job silently lost. This page describes the
recovery machinery that sits on top of the [process lifecycle](process_lifecycle.md)
and the [fault propagation](shutdown_and_faults.md#fault-propagation) chain.

## The layered recovery model

Recovery is organised as nested layers, each handling a more severe failure than
the one below and only escalating when the lower layer cannot cope:

| Layer | Scope | Mechanism | Owner |
| ----- | ----- | --------- | ----- |
| 1 | A single job faulted | Bounded retry; one degraded (isolated) retry for resource faults | `JobTracker` + `failure_classification.py` |
| 2 | A single slot crashed | Replace the process; quarantine it if it crash-loops | `ProcessLifecycleManager` |
| 3 | The whole worker is wedged | Soft-reset the pools (limp-by), then give up cleanly on unservable jobs | `RecoverySupervisor` (policy) + `HordeWorkerProcessManager` (actions) |

Cutting across all three are two durable records used for diagnosis and orphan
cleanup: the [action ledger](#the-action-ledger) and the
[owned-PID registry](#the-owned-pid-registry).

## Layer 1: bounded and degraded job retry

When inference faults (a slot crash, a hung timeout, a failed dispatch, or an
error reported by the child), the job is **not** immediately reported faulted to
the horde. `JobTracker` resolves the fault via `handle_job_fault` /
`handle_job_fault_now`, which returns an
[`InferenceFailureResolution`][horde_worker_regen.process_management.job_tracker.InferenceFailureResolution]:

- **Retry**: the job has attempts left, so it returns to `PENDING_INFERENCE` for
  a fresh dispatch. The attempt budget is `max_inference_attempts` (bridge config,
  default `2`, range `1`–`5`); `1` restores the pre-resiliency "one shot, then
  fault" behaviour.
- **Retry degraded**: a job that faulted with a **resource failure**
  (CUDA/HIP out-of-memory) earns *one* degraded, isolated retry. The tracker sets
  `needs_degraded_dispatch`; the scheduler consumes it and re-dispatches the job
  more conservatively (alone, without competing VRAM pressure). A job spends this
  degraded retry only once (`degraded_retry_used`).
- **Faulted**: attempts are exhausted, the fault is not retryable (e.g. a
  post-inference safety failure, where re-running cannot help), or the job was
  never formally queued. The job is reported faulted to the API **with
  diagnostics**, so the horde reissues it elsewhere.

[`failure_classification.is_resource_failure`][horde_worker_regen.process_management.failure_classification.is_resource_failure]
decides resource-vs-other by substring-matching the faulted result's `info`
string (it recognises both real allocator messages and the chaos harness's
injected OOM marker). It is deliberately dependency-free so it cannot itself
raise on a surprising message.

## Layer 2: slot replacement and crash-loop quarantine

A single crashed or hung slot is handled by the
[`ProcessLifecycleManager`](process_lifecycle.md#process-replacement): the dead
process is removed from `ProcessMap`, its model ownership is cleared, any
in-flight job is faulted into Layer 1, and a replacement is spawned with a fresh
`process_launch_identifier`.

A slot that **crash-loops** (repeatedly dies shortly after being replaced) is
*quarantined* rather than respawned forever: the lifecycle manager tracks
`quarantined_inference_slots` so a deterministically-broken slot (a model that
always OOMs on load, say) stops consuming respawn churn. A merely slow,
replacing, or model-loading slot is **not** quarantined; only repeated fast
crashes trip the breaker.

## Layer 3: save-our-ship (SOS) escalation

Layers 1 and 2 handle *individual* failures. The SOS layer answers a different
question: *the worker as a whole has stopped making progress on work it has
accepted: now what?* This is split into a pure **policy** object and the
manager-side **actions**:

- [`RecoverySupervisor`][horde_worker_regen.process_management.recovery_supervisor.RecoverySupervisor]
  is the policy. It tracks how long the worker has been wedged and returns a
  [`RecoveryAction`][horde_worker_regen.process_management.recovery_supervisor.RecoveryAction].
  Keeping it pure (it takes a wedge boolean and a clock) makes the escalation
  timing unit-testable with a fake clock.
- `HordeWorkerProcessManager` owns the wedge **assessment** and the **actions**.
  `_assess_wedge()` decides the worker is wedged only on definitive signals (every
  inference slot quarantined, or the safety pool crash-looping with no healthy
  process). A busy, slow, replacing, or model-loading worker is never wedged.
  `_run_recovery_supervisor()` runs each control-loop tick and applies the
  returned action.

The escalation, in order:

1. **Soft reset (bounded)** (`_perform_soft_reset`): rebuild the process pools
   in place (kill and respawn every child, un-quarantine slots) and drop one
   **limp-by** notch, reducing effective concurrency (`max_threads`) so a worker
   that wedges under load can keep limping along. The parent process and the TUI
   stay attached. A transient wedge (a bad model load, a one-off deadlock)
   recovers here. If the episode later recovers after a sustained clean streak,
   limp-by is cleared exactly once and configured concurrency is restored.
2. **Give up cleanly** (`_give_up_on_wedged_jobs`): once resets clearly are not
   helping (e.g. a deterministic crash-on-start), stop fighting: fault the jobs
   that cannot be served so the horde reissues them, rather than wedging forever.
   The worker keeps running and keeps popping.

With `exit_on_unhandled_faults` set, the worker exits instead of limping; SOS is
the default-on alternative that prioritises continued operation.

## The action ledger

[`ActionLedger`][horde_worker_regen.process_management.action_ledger.ActionLedger]
is an append-only, self-audited record of the lifecycle actions the parent takes
on its children: when each slot was spawned (and its OS pid), when inference was
dispatched, when a held semaphore was released on its behalf, when a timeout
fired and why a slot was replaced. When a child hangs or crashes, this ordered
account is the single most useful diagnostic.

It keeps a bounded in-memory ring (always on, cheap, queried for the timeout
diagnostics dump) and optionally mirrors each event to a size-rotated JSONL file
(`.horde_worker_regen/action_ledger.jsonl`) so the record survives a restart. It
**never raises**: a file IO error degrades to in-memory only, so auditing cannot
itself wedge the worker. (Mirroring is disabled under `AI_HORDE_TESTING`.)

## The owned-PID registry

[`OwnedProcessRegistry`][horde_worker_regen.process_management.owned_process_registry.OwnedProcessRegistry]
persists which OS pids the worker started, so the *next* startup can find and
kill any that are still alive after a hard parent death (SIGKILL, OOM-kill, power
loss) that skipped the graceful shutdown path. Orphaned children otherwise keep a
GPU resident and a model loaded, and a relaunched worker contends with its own
zombies.

The single hazard with pid-based reaping is pid reuse, so each record stores the
child's `create_time` (and a name fragment); a survivor is killed only when both
still match. The file lives at `.horde_worker_regen/owned_pids.json`; reads never
raise and writes are atomic, so it can never block startup. An `atexit` handler
(`_kill_owned_children_on_exit`) is the in-process backstop for the cases that do
unwind cleanly.

## Fault injection and chaos testing

Because this machinery only matters when things go wrong, the worker ships a
typed fault-injection harness to exercise it without a GPU or a real failure.
[`FaultProfile`][horde_worker_regen.process_management.fault_injection.FaultProfile]
tells one of the [fake worker processes](architecture.md#dry-run-mode) to
misbehave in a specific, reproducible way: hang, crash, drop heartbeats, run
slow, exhaust resources, or emit a malformed message. Profiles are plain pydantic
models so they pickle cleanly across the spawn boundary.

The chaos tests drive the *real* process manager, scheduler, safety
orchestrator, and job tracker through these faults and assert the worker
recovers: the job eventually completes-or-faults, the slot is replaced, no
semaphore is orphaned, and the worker keeps running.

## See also

- [Process Lifecycle](process_lifecycle.md): slot replacement, semaphores, and
  hung-process detection that Layer 2 builds on
- [Shutdown and Faults](shutdown_and_faults.md): the fault-propagation chain and
  graceful-vs-abort shutdown
- [Job State Machine](job_state_machine.md): the stages a retried or faulted job
  moves through
- [`RecoverySupervisor`][horde_worker_regen.process_management.recovery_supervisor.RecoverySupervisor]
- [`ActionLedger`][horde_worker_regen.process_management.action_ledger.ActionLedger]
- [`OwnedProcessRegistry`][horde_worker_regen.process_management.owned_process_registry.OwnedProcessRegistry]
