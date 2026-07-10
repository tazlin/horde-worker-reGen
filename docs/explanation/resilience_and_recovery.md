# Resilience and Recovery

- [Resilience and Recovery](#resilience-and-recovery)
    - [The layered recovery model](#the-layered-recovery-model)
    - [Layer 1: bounded and degraded job retry](#layer-1-bounded-and-degraded-job-retry)
    - [Layer 2: slot replacement and crash-loop quarantine](#layer-2-slot-replacement-and-crash-loop-quarantine)
    - [Stranded in-progress jobs](#stranded-in-progress-jobs)
    - [Stranded safety-check jobs](#stranded-safety-check-jobs)
    - [Stranded post-processing jobs](#stranded-post-processing-jobs)
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
| 3 | The whole worker is wedged | Soft-reset the pools (concurrency preserved), then give up cleanly on unservable jobs | `RecoverySupervisor` (policy) + `WorkerRecoveryCoordinator` (assessment/actions) |

Cutting across all three are two durable records used for diagnosis and orphan
cleanup: the [action ledger](#the-action-ledger) and the
[owned-PID registry](#the-owned-pid-registry).

## Layer 1: bounded and degraded job retry

When inference faults (a slot crash, a hung timeout, a failed dispatch, or an
error reported by the child), the job is **not** immediately reported faulted to
the horde. `JobTracker` resolves the fault via `handle_job_fault` /
`handle_job_fault_now`, which returns an
[`InferenceFailureResolution`][horde_worker_regen.process_management.jobs.job_tracker.InferenceFailureResolution]:

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

[`failure_classification.is_resource_failure`][horde_worker_regen.process_management.jobs.failure_classification.is_resource_failure]
decides resource-vs-other by substring-matching the faulted result's `info`
string (it recognises both real allocator messages and the chaos harness's
injected OOM marker). It is deliberately dependency-free so it cannot itself
raise on a surprising message.

## Layer 2: slot replacement and crash-loop quarantine

A single crashed or hung slot is handled by the
[`ProcessLifecycleManager`](process_lifecycle.md#process-replacement): the dead
process is removed from `ProcessMap`, its model ownership is cleared, the job
**that slot was running** (taken from its own `last_job_referenced`, never by
scanning the map for the first in-flight job) is faulted into Layer 1, and a
replacement is spawned with a fresh `process_launch_identifier`.

A slot that **crash-loops** (repeatedly dies shortly after being replaced) is
*quarantined* rather than respawned forever: the lifecycle manager tracks
`quarantined_inference_slots` so a deterministically-broken slot (a model that
always OOMs on load, say) stops consuming respawn churn. A merely slow,
replacing, or model-loading slot is **not** quarantined; only repeated fast
crashes trip the breaker.

## Stranded in-progress jobs

Per-slot replacement faults the job of the slot it replaces, but a job can still
end up marked `in_progress` with nothing left to move it on: its
`HordeInferenceResultMessage` can be **lost** (dropped by the launch-identifier
guard while the slot was being replaced), or it can be mis-associated by a
requeue race. No result will ever arrive for such a job, so it would pin the head
of the queue and count against the concurrent-job cap forever. Two independent
backstops guarantee the [no-loss invariant](job_lifecycle.md#pipeline-invariants)
holds anyway:

- **Prompt detector** (`MessageDispatcher._reap_lost_inference_result`): the
  moment a slot reports it is back to `WAITING_FOR_JOB` *from an inference-active
  state* while still referencing a job that is still `in_progress`, the result must
  have been lost. Because results and state changes share one ordered message
  stream, a real result is always processed *before* the idle transition, so this
  cannot misfire on a normally completed job. The job is released retryably
  (Layer 1) the tick the loss becomes observable. The slot is alive, so this is
  **not** treated as a process crash. The "from an inference-active state"
  qualifier is essential, but not sufficient by itself: `last_job_referenced` and
  the in-progress mark are stamped by the scheduler the instant it *dispatches* a
  job, before the child has acknowledged it, so a slot can carry a freshly
  dispatched job while it is still draining state messages from *before* the
  dispatch (the idle it reports after unloading the previous model to free VRAM,
  say). The reaper also compares the active dispatch timestamp with the state
  transition it is closing; an idle report older than the current dispatch is left
  alone. Reaping on that idle would fault a job that never ran, a window that
  widens on slower disks and larger models, so only a return to idle from a state
  where inference actually ran *for the same dispatch epoch* can mean a result was
  lost. The periodic watchdog below covers the remaining shapes.
- **Periodic watchdog** (`WorkerRecoveryCoordinator.reconcile_orphaned_in_progress_jobs`):
  each control-loop tick, any `in_progress` job that **no live slot is actively
  working** is punted (retryably) once it has been un-owned for a short grace
  window. The grace rides out the brief dispatch race between marking a job
  in-progress and the slot reporting `INFERENCE_STARTING`. The key subtlety is
  *ownership*: a slot only owns its job while it cannot accept new work. An
  **idle** slot (`can_accept_job()` is true) whose `last_job_referenced` still
  points at the job does **not** shield it, because that reference is retained
  across completion and is not a "currently running" flag. Treating a stale idle
  reference as ownership is what previously let a single stranded job wedge the
  whole worker.

A *recurring* storm of orphan punts means something upstream keeps stranding jobs
(a flaky GPU, say); that feeds the wedge assessment below so SOS can limp the
worker by rather than punting forever.

## Stranded safety-check jobs

The safety stage has the same shape of loss. A job handed to the safety process
sits in `SAFETY_CHECKING` until its verdict returns; if the safety process is
**replaced** while the check is in flight, the verdict arrives from a now-retired
launch and is dropped by the same launch-identifier guard. Safety-process
replacement is routine, not exceptional: whole-card residency moves the safety
process off the GPU while a card-filling model holds the device and restarts it
when the residency lifts, so a model mix that alternates between a card-filler and
co-resident models replaces the safety process repeatedly. Nothing else moves a
job whose verdict was dropped, so each one would pin a pipeline slot until
recovered; let enough pile up and the pipeline wedges into an SOS soft reset.

`WorkerRecoveryCoordinator.reconcile_orphaned_safety_jobs` recovers them each
control-loop tick, with the same two-signal split as the in-progress case:

- **Prompt signal**: when the dispatcher drops a safety result because its launch
  was retired, it flags that job's verdict as *known lost* (positive evidence, not
  a timeout suspicion). The reconcile pass drains those flags and re-checks the job
  on the next tick, skipping the grace it would otherwise wait out.
- **Periodic watchdog**: any job that has sat in `SAFETY_CHECKING` past a grace
  window with no verdict is requeued for a fresh check, covering losses with no
  corresponding dropped message at all.

Both routes share one bounded requeue/escalation counter, so a verdict that keeps
being lost is requeued only a fixed number of times before the job is dropped with
its images cleared (an image the safety check never cleared is **never** submitted)
and popping is soft-paused until safety recovers. Re-checked images are always
preserved, never submitted unchecked.

## Stranded post-processing jobs

Post-processing uses a dedicated GPU-bearing lane, so it can be replaced or
temporarily stopped independently from inference. A job handed to that lane sits
in `POST_PROCESSING` until its processed images return. If the lane is retired
while the result is already in flight, the normal launch-identifier guard would
otherwise discard a valid result and leave the job waiting for images that will
never arrive.

The dispatcher keeps the retired-launch guard, but makes one narrow exception:
a successful `HordePostProcessResultMessage` from a retired post-processing
launch is accepted only when `JobTracker` still records that exact process id and
launch identifier as the owner of the job's current post-processing attempt. The
result is then handled through the same path as an ordinary live-lane result,
which adopts the processed images, releases the active post-processing reserve,
and queues the job for safety.

The ownership stamp is cleared whenever the job leaves the active
post-processing attempt: successful completion, watchdog requeue, or explicit
detachment. A result from an older attempt therefore cannot overwrite a newer
attempt, and a faulted retired-lane result still enters the known-lost path. The
watchdog requeues those known-lost jobs for a bounded number of fresh
post-processing attempts; once the retry budget is exhausted, the job is faulted
without images rather than submitting raw images that did not satisfy the
requested post-processing contract.

## Layer 3: save-our-ship (SOS) escalation

Layers 1 and 2 handle *individual* failures. The SOS layer answers a different
question: *the worker as a whole has stopped making progress on work it has
accepted: now what?* This is split into a pure **policy** object and the
manager-side **actions**:

- [`RecoverySupervisor`][horde_worker_regen.process_management.lifecycle.recovery_supervisor.RecoverySupervisor]
  is the policy. It tracks how long the worker has been wedged, whether its
  rebuilt pool has become ready, and returns a
  [`RecoveryAction`][horde_worker_regen.process_management.lifecycle.recovery_supervisor.RecoveryAction].
  Keeping it pure (it takes a wedge boolean, a pool-ready boolean, and a clock)
  makes the escalation timing unit-testable with a fake clock.
- [`WorkerRecoveryCoordinator`][horde_worker_regen.process_management.lifecycle.worker_recovery_coordinator.WorkerRecoveryCoordinator]
  owns the wedge **assessment** and the **actions**; `HordeWorkerProcessManager` calls it directly from the control loop. `assess_wedge()` decides the worker is wedged only on definitive signals: every
  inference slot quarantined, the safety pool crash-looping with no healthy
  process, a **sustained** structural queue deadlock (pending inference work with
  every process idle, held long enough to rule out the transient all-idle gap
  between jobs), or a recurring [orphaned-job](#stranded-in-progress-jobs) punt
  storm. A busy, slow, replacing, or model-loading worker is never wedged, and a
  queue deliberately held while a heavy model establishes whole-card residency is
  excused by a bounded grace. The same grace covers lifecycle's deferred
  GPU-process starts: a slot killed for recovery or RAM reclamation may be absent
  from `ProcessMap` while its respawn waits for device-free headroom, but SOS
  treats that as recoverable capacity while the wait is young or free-VRAM
  readings show drain progress. If the card never recovers past the bounded
  no-progress window, the normal unrecoverable-pool checks resume.
  `run_recovery_supervisor()` runs each control-loop tick and applies the
  returned action.

  A pending post-processing drain can deliberately hold new inference sampling,
  so admission gets one bounded chance to reclaim ordinary idle memory and, only
  after a fresh non-fitting measurement, borrow one verified-idle VAE or component
  service-lane context. That resolves the known all-idle mutex before it matures
  into a structural queue deadlock. It does **not** excuse the queue from wedge
  assessment, reset the structural-deadlock timer, or suppress SOS. When no safe
  context can be reclaimed—or the operator's safety-placement policy forbids the
  remaining action—the ordinary admission-patience fault and SOS escalation stay
  authoritative. The local fix removes the avoidable resource cycle without
  weakening the recovery system that catches every other cause.

  A wedge episode closes (and its escalation counter resets) on a clean streak
  alone only before any soft reset has been attempted. Once a soft reset has been
  spent, the streak must be corroborated by real forward progress since the most
  recent soft reset: the reset requirement is belt-and-suspenders, the quiet-wedge
  time streak **and** an objective progress signal. This is because a rebuild
  transiently reads as not-wedged (the un-quarantine to re-quarantine window, or a
  queue deadlock that momentarily clears while the pool boots), and that window can
  outlast the streak. If the streak alone reset the counter, a doomed pool would
  close its episode on the transient window, open a fresh one on the next wedge, and
  log every soft reset as the first, so `limp_by_level` never climbs and the
  readiness-gated give-up is never reached. Requiring progress holds the counter
  across the transient window, so a pool that keeps rebuilding without ever serving
  work climbs the ladder to give-up instead.

  The progress signal is objective forward movement: a job completion or a new
  inference start proves upstream accepted work is moving again, so stale
  unrecoverable-pool state is also cleared instead of carrying a past queue deadlock
  into another soft reset. When accepted post-processing work is still pending or
  running, though, upstream starts are not enough: only a post-processing drain
  transition (dispatch to the lane, requeue after a lost attempt, processed result
  moving to safety, or no-image fault) proves that the downstream lane itself is
  making headway. The baseline these compare against is captured when the episode
  opens and re-captured on every soft reset, so the credit is progress since the
  latest recovery attempt, not since the episode began.

The escalation, in order:

1. **Soft reset (bounded)** (`perform_soft_reset`): rebuild the process pools
   in place (kill and respawn every child, un-quarantine slots), preserving the
   configured concurrency (`max_threads`). The rebuild alone clears a transient
   wedge; the cap is deliberately not lowered, because shedding a lane on every
   wedge let a one-off blip (including one provoked by aggressive co-sampling
   tripping a sampler watchdog) ratchet throughput down and outlast its cause. The
   escalation policy still **counts** each soft reset (`limp_by_level`) so a
   persistent wedge still escalates to give-up. The parent process and the TUI stay
   attached. A transient wedge (a bad model load, a one-off deadlock) recovers here.
2. **Give up cleanly** (`give_up_on_wedged_jobs`): once resets clearly are not
   helping (e.g. a deterministic crash-on-start), stop fighting: fault the jobs
   that cannot be served so the horde reissues them, rather than wedging forever.
   If the pool is still structurally usable (for example, a queue-deadlock give-up
   with live capacity), the worker keeps running. If inference or safety capacity
   cannot be restored, SOS escalates through abort so the worker process exits
   non-zero after killing its children; the TUI supervisor then relaunches it via
   the normal unexpected-exit path.

   Give-up is **readiness-gated**, not fixed-age. A soft reset rebuilds the pool,
   and the replacement children spend real time booting (importing torch) before
   any lane can accept a job. Faulting during that boot window drops jobs the
   just-rebuilt pool was about to run. So the escalation clock does not advance
   while the pool is still booting: give-up fires only once an inference lane has
   reached an accepting state (`is_inference_pool_ready()`, the same accepting-state
   fact whose absence the queue-deadlock detector reports as "some processes are
   starting. Waiting.") *and* the wedge has then persisted for a further grace. A
   pool whose children never come up is bounded by a boot allowance so give-up is
   still reachable. Give-up is also **latched once per cycle** (repeat wedged ticks
   are no-ops, so the ledger is not spammed), and its `recovery_abandoned` ledger
   record is written only when the pass actually did something (faulted at least
   one job, or made a terminal abort decision), never a `jobs_faulted=0` no-op. If
   a wedge persists over a ready pool past the first give-up, a **bounded
   continuation** permits exactly one more soft-reset cycle after a cool-down; a
   second give-up is then flagged terminal and abandons ship deliberately rather
   than faulting jobs on every tick forever.

With `exit_on_unhandled_faults` set, the worker exits instead of limping; SOS is
the default-on alternative that prioritises continued operation.

## Self-protective feature throttles

The horde forces a worker into maintenance when it "drops too many jobs". Layers
1-3 keep a *struggling* worker serving, but some failures are **structural**: a
capability the worker advertises that this hardware simply cannot honour. Faulting
those jobs and waiting for the next one only feeds the forced-maintenance spiral.
So the worker also withdraws the failing capability before that happens.

The **post-processing fault breaker** is the instance of this for post-processing.
A post-processing peak that cannot be hosted *at all* (see
[post-processing VRAM over-commit](bridge_config.md#post-processing-vram-over-commit))
faults the job, and a watchdog-reaped post-processing stall does the same. A peak
that only *transiently* overflows a contended card it would fit once drained, including pressure from
speculative preloads, is instead held until the card gives the lane a drain window, not faulted, so it never reaches
the breaker. The unhostable-peak fault is **terminal** (non-retryable): a local retry would only
re-dispatch the job into the same unchanged, still-overflowing card (a guaranteed
second fault), so the job is reissued by the horde elsewhere instead, and one
placement failure feeds the breaker exactly one count. Both sources feed a
rolling-window counter
([`JobTracker.count_recent_post_processing_faults`][horde_worker_regen.process_management.jobs.job_tracker.JobTracker.count_recent_post_processing_faults]);
once it exceeds `post_processing_fault_threshold` within
`post_processing_fault_window_seconds`, the worker stops advertising
post-processing at pop time, so the horde stops sending it upscale/face-fix jobs,
and logs an operator advisory to downgrade settings. The suppression is
**session-latched** (it survives a soft reset and clears only on restart)
because the over-commit is structural and auto-recovery would simply re-trip it.
It mirrors the per-model unservable breaker and the self-maintenance throttle: a
worker that protects its own standing on the horde rather than bleeding dropped
jobs until the server intervenes. The dedicated post-processing lane (see
[Process lanes and job chaining](process_lanes_and_chaining.md)) is the structural
complement that keeps the breaker from being needed in the first place: its
fixed resident footprint replaces the transient per-job peak that caused the
over-commits.

## The action ledger

[`ActionLedger`][horde_worker_regen.process_management.ipc.action_ledger.ActionLedger]
is an append-only, self-audited record of the lifecycle actions the parent takes
on its children: when each slot was spawned (and its OS pid), when a GPU-bearing
start was deferred for device-free headroom, when inference was dispatched, when
a held semaphore was released on its behalf, when a timeout fired and why a slot
was replaced. When a child hangs or crashes, this ordered
account is the single most useful diagnostic. It also records worker-level pop
governance transitions: the shared self-throttle pop-pause is ledgered when armed
(`POP_PAUSE_ARMED`) and when it lapses (`POP_PAUSE_LAPSED`), each carrying the
[`PopPauseOwner`][horde_worker_regen.process_management.config.worker_state.PopPauseOwner]
that set the deadline, the duration, and the numeric context (the measured free
and floor MB for a RAM-pressure pause), so an operator can attribute a pop-pause
spell to the backstop that caused it.

It keeps a bounded in-memory ring (always on, cheap, queried for the timeout
diagnostics dump) and optionally mirrors each event to a size-rotated JSONL file
(`.horde_worker_regen/action_ledger.jsonl`) so the record survives a restart. It
**never raises**: a file IO error degrades to in-memory only, so auditing cannot
itself wedge the worker. (Mirroring is disabled under `AI_HORDE_TESTING`.)

## The owned-PID registry

[`OwnedProcessRegistry`][horde_worker_regen.process_management.lifecycle.owned_process_registry.OwnedProcessRegistry]
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
[`FaultProfile`][horde_worker_regen.process_management.simulation.fault_injection.FaultProfile]
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
- [`RecoverySupervisor`][horde_worker_regen.process_management.lifecycle.recovery_supervisor.RecoverySupervisor]
- [`ActionLedger`][horde_worker_regen.process_management.ipc.action_ledger.ActionLedger]
- [`OwnedProcessRegistry`][horde_worker_regen.process_management.lifecycle.owned_process_registry.OwnedProcessRegistry]
