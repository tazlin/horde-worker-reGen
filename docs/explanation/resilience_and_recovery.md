# Resilience and Recovery

- [Resilience and Recovery](#resilience-and-recovery)
    - [The layered recovery model](#the-layered-recovery-model)
    - [Layer 1: bounded and degraded job retry](#layer-1-bounded-and-degraded-job-retry)
    - [Layer 2: slot replacement and crash-loop quarantine](#layer-2-slot-replacement-and-crash-loop-quarantine)
    - [Poison-model quarantine](#poison-model-quarantine)
    - [Stranded in-progress jobs](#stranded-in-progress-jobs)
    - [Stranded safety-check jobs](#stranded-safety-check-jobs)
    - [Stranded post-processing jobs](#stranded-post-processing-jobs)
    - [Layer 3: save-our-ship (SOS) escalation](#layer-3-save-our-ship-sos-escalation)
    - [Self-protective feature throttles](#self-protective-feature-throttles)
    - [The terminal-fault-rate breaker](#the-terminal-fault-rate-breaker)
    - [The background download process](#the-background-download-process)
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
| 3 | The whole worker is wedged | Soft-reset the pools (concurrency preserved), then give up cleanly on unservable jobs, and finally take a fresh process when one is reachable | `RecoverySupervisor` (policy) + `WorkerRecoveryCoordinator` (assessment/actions) |

The escalation is ordered least-destructive first, and its endpoint is a **working worker**, not a stopped
one. A fresh process is the last rung rather than an outcome: the worker exits non-zero so whoever
launched it starts a new one, which clears state an in-place rebuild cannot. That rung is only taken when
something is listening for the exit, either the dashboard supervising its worker child or an operator who
set `exit_on_unhandled_faults` alongside a service manager (see
[Run the worker as a system service](../how-to/run-as-a-system-service.md)). With neither, the worker does
not exit: it keeps escalating with the remedies it can apply in place, because exiting where nothing would
restart it ends the worker's usefulness instead of restoring it.

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
**that exact process launch was running** (taken from typed execution ownership,
never from preload intent or by scanning the map for the first in-flight job) is
faulted into Layer 1, and a
replacement is spawned with a fresh `process_launch_identifier`.

Process association has two explicit meanings. `PreloadJobIntent` attributes model
preparation for display and scheduling, but grants no retry-spending ownership.
`InferenceExecutionOwnership` records the dispatched job, launch identifier, and
attempt ordinal. Replacement and orphan recovery accept the latter only when its
launch still matches the live slot; a preload-only replacement cannot fault work
that never ran.

A slot that **crash-loops** (repeatedly dies shortly after being replaced) is
*quarantined* rather than respawned forever: the lifecycle manager tracks
`quarantined_inference_slots` so a deterministically-broken slot (a model that
always OOMs on load, say) stops consuming respawn churn. A merely slow,
replacing, or model-loading slot is **not** quarantined; only repeated fast
crashes trip the breaker.

### The safety pool's breakers

The safety pool is rebuilt rather than quarantined (a worker without safety
capacity cannot serve anything, so there is no useful "leave it out of the pool"
state), which makes the bounds on that rebuilding the whole of its protection.
Three of them apply, and they cover different shapes of the same failure:

- **The sliding window** (`SAFETY_CRASH_LOOP_MAX`): more rebuilds than this
  within `CRASH_LOOP_WINDOW_SECONDS` reports the pool as failing.
- **The consecutive start-failure streak**
  (`SAFETY_CRASH_LOOP_MAX_START_FAILURES`): this many rebuilds in a row without
  one reaching readiness declares the pool unable to start, whatever their
  spacing. It exists because a deterministic start failure pays a full cold start
  per attempt, so its rebuilds can arrive spaced wider than the window can
  accumulate and age their own evidence out; the pool would then respawn for as
  long as the worker runs while every breaker reads clear. The streak resets the
  moment a safety process reaches readiness, so a pool that comes up, serves, and
  later dies is an ordinary crash the rebuild path is for. This is the safety-side
  companion of `CRASH_LOOP_MAX_START_FAILURES` on inference slots, and either
  signal makes the pool *unrecoverable* to the wedge assessment.
- **The respawn backoff** (`SAFETY_RESPAWN_BACKOFF_BASE_SECONDS` doubling to
  `SAFETY_RESPAWN_BACKOFF_MAX_SECONDS`): consecutive failed rebuilds wait longer
  before the next start attempt. The first respawn after a lone crash is
  immediate, since that is the ordinary recovery path; only a repeat failure
  earns a wait, so a pool that cannot start stops consuming a spawn, a cold
  start, and a reap several times a second while the escalation decides. Only the
  respawn waits: a hung child is still ended and reaped promptly.

Deliberate safety-pool churn is kept out of the record: a whole-card
pause/restore cycle and a supervised soft-reset rebuild both mark the replacement
*intentional*, and the suppression holds until the replacement reaches readiness,
because the startup churn in between is part of the same placement change. A pool
that cannot start never reaches readiness, so that window is additionally bounded
by `SAFETY_INTENTIONAL_WINDOW_MAX_UNREADY_REBUILDS` attempts. Past the bound the
window is no longer describing a placement change and its rebuilds are counted
like any other, so a crash loop cannot hide inside an open intentional window and
report the pool healthy indefinitely. Readiness clears all of it at once: the
window, the streak, and the backoff.

Reading these incidents, a child that died before it ever reported to the parent
(`HordeProcessInfo.has_ever_reported`) failed in its own startup, before its
message loop; one that had reported got past that and died later. The futility
disclosure names which it was.

### Coercing the adaptive solver instead of reaping it

The watchdog below is a **backstop**, not the first response to a runaway
adaptive sampler. The inference engine bounds the solver itself: hordelib's
`hordelib.execution.adaptive_sampler_bound` replaces ComfyUI's `dpm_adaptive`
sampler function with one that stops the solver's loop at 1.25 times the nominal
step count and returns the sample it has. Iterations past the nominal schedule
are the solver polishing against its error tolerance, which buys approximately no
perceptible quality while costing a full model evaluation each, so the bounded
run delivers what the requester asked for at the cost the schedule advertised.

Because the bound produces a usable sample, the job **succeeds**. That makes the
coercion invisible unless it is declared, so the child turns the engine's
truncation record into one `information` / `see_ref` `gen_metadata` entry naming
the iterations run and the schedule they were measured against
(`sampler_truncation_disclosure`). `information` is a non-reportable metadata
type, so the disclosure does not inflate the submission's fault count: the
generation was delivered, not faulted.

Both inference paths disclose it, in the same shape. On the monolithic path the
child reads the record straight off its result. On the disaggregated path the
sampler and the decode that produces the image run in different processes, so the
sample stage returns the record alongside its LATENT
(`hordelib.horde.SampleStageResult`), forwards it on the optional
`SampleSliceResult.sampler_truncation` field, and the orchestrator holds it with
the rest of the job's stage state until the decode returns the images, attaching
the entry at the completion hand-off. The record is held per job and dropped with
the job's state at completion or `release_job`, so it cannot follow a job out of
the pipeline or attach to a neighbour.

The ordering matters for reading incidents. A bounded solver terminates on its
own well inside the watchdog's ceiling, so a `sampler_overtime_reap` after this
landed means the bound did not hold (a sampler the patch does not cover, an
engine build without it, or a hang outside the solver loop) rather than an
ordinary long adaptive run.

### The stuck-step watchdog and its final-step allowance

Every other hang check measures **silence**. The stuck-step watchdog covers the
opposite shape: a generation that keeps invoking the progress callback on the
same step, so the slot never goes quiet and the silence checks never fire. The
child counts consecutive non-advancing progress reports and forwards the running
count on its heartbeats; `inference_stuck_step_repeat_limit` reaps on it.

That count alone is not evidence of a wedge at the **end** of sampling. An
error-controlled ("adaptive") solver such as `k_dpm_adaptive` chooses its own
iteration count from the local error estimate and routinely runs past the
nominal schedule, and the backend clamps the reported position at the total, so
each overtime iteration arrives as another report of the final step at full
per-iteration cadence while real GPU work continues. The watchdog therefore
applies `effective_stuck_step_repeat_limit`: repeats below the final step keep
the configured limit, and repeats at the final step (the parent knows the true
`N/N` because the saturated heartbeats carry the step counts) are judged against
`max(configured_limit, 2 × total_steps)`. An overshoot of roughly the schedule's
own length is within what such a solver asks for; beyond that the reports are no
longer explainable as overtime. The reap line names which ceiling was crossed.

A generation that has genuinely stopped doing work stops reporting altogether,
and that silence is what `inference_step_timeout` proves. The repeat count only
ever sees a slot that is still calling back.

An overtime reap is **terminal for its job**, unlike every other slot
replacement. The two reaps carry different verdicts: a mid-run repeat loop says
nothing about the payload, so its job takes the ordinary Layer 1 retry, but a
run past the doubled ceiling is a statement about how many iterations *this*
payload asks for, and the solver is deterministic. Requeuing it buys an
identical second burn on another slot, whose queue shadow can push a healthy
neighbouring job past the horde's dispatch deadline before the retry is
abandoned anyway. The watchdog therefore passes `sampler_overtime_reap` to
`_replace_inference_process`, which faults the in-flight job with
`retryable=False` so the horde reissues it immediately. Both reaps still record
a `SAMPLER_HANG` incident against the model, keyed to the job that hung.

The terminal fault carries `SAMPLER_OVERTIME_FAULT_REASON` as its fault reason,
which rides the faulted submission's `gen_metadata` (see
[Shutdown and faults → fault propagation](shutdown_and_faults.md#fault-propagation)),
so the horde is told what the worker concluded about the payload.

## Poison-model quarantine

The slot breaker keys on the **slot**, so it cannot see a bad *model*. A
checkpoint that kills whichever slot it is dispatched to is re-dispatched
round-robin across fresh slots, and no single slot is ever replaced often enough
to trip its own breaker while the whole pool burns down. `ProcessLifecycleManager`
therefore keeps a second, model-keyed counter: `record_model_incident` records a
`ModelIncidentKind` against the model, and once one kind crosses its threshold
within `MODEL_INCIDENT_WINDOW_SECONDS` (600 s) the model joins
`quarantined_models`.

Two kinds are counted, with separate thresholds because the evidence differs in
strength:

| Kind | Threshold | Fed by |
| ---- | --------- | ------ |
| `LOAD_FAILURE` | `MODEL_LOAD_FAILURE_QUARANTINE_THRESHOLD` (3) | A child reporting `PRELOADING_FAILED`, a child that dies natively (no report) while its last known state was `PRELOADING_MODEL`, **and** a live child reaped for outstaying `preload_timeout` in that state |
| `SAMPLER_HANG` | `MODEL_SAMPLER_HANG_QUARANTINE_THRESHOLD` (2) | The stuck-step watchdog killing a slot that had this model loaded |

The second feed for `LOAD_FAILURE` covers a checkpoint that crashes the process
outright (an access violation while the backend mmaps a bad file), which never
produces the child-side report the first feed depends on. The third covers the
same silence from the other direction: a checkpoint whose load never returns
leaves the child alive and mute, and the preload-window watchdog reaps it without
any report either. Only that watchdog charges a live slot's model; a bulk
replacement that happens to catch a slot mid-preload carries no evidence about
what it was loading. The hang feed covers a
model that loads perfectly and then wedges the sampler: each occurrence costs a
full stuck-step timeout, a kill, and a double-faulted job, so two within the
window is already a verdict.

Hangs are counted per **distinct job** (`count_model_incidents`), which is why
`record_model_incident` takes the `job_id` the watchdog read from the slot's
current execution ownership. A hung job is requeued when its slot is replaced and can
reach the watchdog again on the next slot, and one generation failing repeatedly
is a single piece of evidence however many slots it costs; without the dedupe a
lone unlucky job could quarantine a model that runs everything else fine. An
incident recorded without a job id identifies nothing, so it stands on its own
rather than merging with other unknowns. Load failures happen before any
job-specific work and are still counted per occurrence.

Attributing a hang to the model does
**not** change the slot bookkeeping: the slot really did die, so its own breaker
still counts the replacement.

A reported `PRELOADING_FAILED` does not always cost a slot. The child ends itself
after reporting a load failure because a failure part-way through a real load leaves
torch and the backend in an unknown state, and a fresh process is the only way back
to a known one. Two failures are exempt, because they happen *before* the backend
touches a weight: a blank model name, and a name the backend refuses to resolve
because it does not know it. Neither leaves anything half-loaded, so the child
reports the failure and stays in its main loop, ready for the next job. Without the
exemption a per-job data error costs a full process replacement with a backend
re-initialisation, and a repeating one becomes a churn loop; the quarantine ladder
still terminates it, now after three load failures rather than three replacements.

Crossing a threshold calls the one quarantine handler every feed site shares
(`HordeWorkerProcessManager._on_model_quarantined`, registered via
`set_model_quarantine_handler`), and that handler is the single downstream:

- Every **queued** job for the model is faulted non-retryably, so the horde
  reissues them to a worker that can run the model instead of the worker
  re-dispatching a model that kills slots.
- The **scheduler** refuses to preload a quarantined model, or to select a head
  job that needs one.
- The **popper** stops advertising it. This is the part that actually stops the
  bleeding: a model left on the offer keeps being assigned, every assignment is
  faulted, and a steady drop stream is exactly what makes the horde server
  force-set maintenance for "dropping too many jobs".

The pop exclusion has a **non-empty floor**. If dropping the quarantined models
would leave nothing to advertise (a worker configured with only that model), the
offer is sent unchanged and a warning is logged once per episode. A worker that
advertises nothing is sent nothing, so it can never produce the work that would
let it recover; taking the faults is a recoverable state, going silent is not.

### Only a real model may be quarantined

A quarantine entry is permanent for the session and takes its model off the offer,
off preload, and off head selection. That makes the *identity* it is keyed on
load-bearing: an entry keyed on a name no model has can never be matched by a real
load or cleared by a real success, so it would sit in the set forever, refusing a
model that does not exist.

`record_model_incident` therefore refuses a blank (empty or whitespace-only) model
name outright: nothing is counted, nothing is quarantined, and the refusal is
surfaced once at `WARNING` since whatever produced the name will keep producing it.
[`HordeModelMap.update_entry`][horde_worker_regen.process_management.models.horde_model_map.HordeModelMap.update_entry]
refuses one the same way, so the load-state map cannot report a residency for a
model that does not exist either. The site that reported the failure still handles
it; only the per-model bookkeeping declines to name a culprit it does not have.

## Pop-boundary validation

A pop response can arrive naming no model at all. Such a job is unservable by
construction: there is no checkpoint to preload and no card to route it to. Left to
flow through the worker it becomes the empty string as a model *identity*, and every
model-keyed mechanism downstream then treats that string as a model: the scheduler
preloads it, the child's preload fails on it, and the per-model incident counter
counts the failures against it until it is quarantined as if it were a checkpoint.

`JobPopper` rejects it at the boundary instead. A popped job whose model name is
absent or blank is never placed in the pending-inference queue: it is registered and
immediately faulted terminally with `JobFaultOrigin.MALFORMED_POP`, so the horde
reissues it at once rather than waiting out its ttl, and the log line names the
condition as a malformed pop rather than a model failure. The rejection happens
before source media is fetched and before auxiliary prefetch is triggered, since
neither can lead anywhere for a job that is already being handed back.

The fault is deliberately routed to two different places:

- It is **excluded** from the consecutive-failure pop pause, along with the other
  non-generation origins. Nothing was generated, so it says nothing about whether
  this worker can generate.
- It **is** counted by the [terminal-fault-rate breaker](#the-terminal-fault-rate-breaker).
  The horde counts the returned job as dropped like any other fault, so a steady
  stream of malformed pops earns the worker forced maintenance whether or not the
  worker faulted them politely. Backing intake off is the correct response.

Each pop request also logs the model set it advertised at `DEBUG`, which is what
makes an empty name attributable after the fact: without it, a blank name the worker
advertised and a blank name the horde answered with look identical in a capture.

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
  qualifier is essential, but not sufficient by itself: typed execution ownership
  and the in-progress mark are stamped by the scheduler the instant it *dispatches*
  a job, before the child has acknowledged it, so a slot can carry a freshly
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
  in-progress and the slot reporting `INFERENCE_PRIMED`. The key subtlety is
  *ownership*: a live slot shields only the typed execution record stamped for its
  current launch. A preload intent and a compatibility/display reference do not
  shield the job. An **idle** slot (`can_accept_job()` is true) likewise does not
  prove execution is still making progress; the active-state and dispatch-epoch
  checks decide the prompt lost-result path, while the periodic grace remains the
  backstop.

A *recurring* storm of orphan punts means something upstream keeps stranding jobs
(a flaky GPU, say); that feeds the wedge assessment below so SOS can limp the
worker by rather than punting forever.

## Stranded safety-check jobs

The safety stage has the same shape of loss. A job handed to the safety process
sits in `SAFETY_CHECKING` until its verdict returns; if the safety process is
**replaced** while the check is in flight, the verdict arrives from a now-retired
launch and is dropped by the same launch-identifier guard. Safety-process
replacement is routine, not exceptional: the scheduler's placement reconciler may
move safety off the GPU for whole-card residency, verified reclaim, or a runtime
fit decision, then restart it when every request and veto clears. A model mix that
changes that placement can therefore replace the safety process repeatedly.
Nothing else moves a job whose verdict was dropped, so each one would pin a
pipeline slot until recovered; let enough pile up and the pipeline wedges into an
SOS soft reset.

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
  Keeping it pure (it takes wedge, pool-ready, and live-boot-in-progress booleans
  and a clock, reading no process map or dispatcher itself) makes the escalation
  timing unit-testable with a fake clock.
- [`WorkerRecoveryCoordinator`][horde_worker_regen.process_management.lifecycle.worker_recovery_coordinator.WorkerRecoveryCoordinator]
  owns the wedge **assessment** and the **actions**; `HordeWorkerProcessManager` calls it directly from the control loop. `assess_wedge()` decides the worker is wedged only on definitive signals: every
  inference slot quarantined, the safety pool crash-looping with no healthy
  process, a **sustained** structural queue deadlock (pending inference work with
  every process idle, held long enough to rule out the transient all-idle gap
  between jobs), a recurring [orphaned-job](#stranded-in-progress-jobs) punt
  storm, or job pops held at one gate past `POP_GATE_HELD_WEDGE_SECONDS` with
  nothing completing behind it. A busy, slow, replacing, or model-loading worker is never wedged, and a
  queue deliberately held while a heavy model establishes whole-card residency is
  excused by a bounded grace. A pending job whose
  [auxiliary prefetch](model_downloads.md#job-driven-auxiliary-prefetch) is still
  in flight is likewise excused: it holds no inference lane by design while its
  LoRAs or textual inversions download, so the deadlock detector excludes it from
  both the queue- and general-deadlock conditions rather than reading "pending plus
  every process idle" as a wedge. That exclusion is bounded by the coordinator's
  per-job prefetch deadline, so a genuinely stalled download stops shielding the
  queue the moment its deadline lapses (the job is then served without its
  never-in-flight file, or faulted if its transfer stalled in flight, and either way
  stops fuelling the shield as any resolved job would). The same grace covers lifecycle's deferred
  GPU-process starts: a slot killed for recovery or RAM reclamation may be absent
  from `ProcessMap` while its respawn waits for device-free headroom, but SOS
  treats that as recoverable capacity while the wait is young or free-VRAM
  readings show drain progress. If the card never recovers past the bounded
  no-progress window, the normal unrecoverable-pool checks resume.
  `run_recovery_supervisor()` runs each control-loop tick and applies the
  returned action.

  Every signal above names a condition, and a condition nobody modelled holds the
  intake path just as effectively, so the assessment also carries one
  **failure-independent backstop**: pops held at the same gate for longer than
  `POP_GATE_HELD_WEDGE_SECONDS` (15 minutes), with no pop attempt reaching the
  horde in that span and no job completed since the hold began, is a wedge
  whichever gate it is. Completed work is the proof it keys on precisely because
  no gate-holding failure can produce one on its own, which is what lets the check
  stand over gates added later. The two liveness clauses are also what keep
  ordinary backpressure out: a worker whose local queue is full holds a gate
  continuously and attempts no pops while doing so, and its completions excuse the
  hold whatever its length. The threshold sits above every in-place remedy window
  the worker gives a condition to clear itself (600 s at the widest), so the
  watchdog that owns the condition always acts first and this only sees what it
  left unresolved. A held-gate wedge escalates through the same ladder as any
  other: an episode opens, the remedies run in order, and the give-up reaches the
  terminal rung if nothing restores work flow.

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

  When accepted work exists, the progress signal is movement of the episode's
  recovery frontier: the oldest unresolved accepted job must advance to a later stage
  or complete successfully. Faulting or requeueing that head is not progress the
  recovery path may use to prove itself. A follower starting or completing cannot prove that an unchanged
  blocked head recovered. Aggregate completion and stage counters remain the
  fallback for pool-level episodes with no accepted frontier. If unrelated work
  advances while the frontier does not, it may earn one observation interval for
  that throughput to reach the head; this delay neither closes the episode nor
  consumes a reset/give-up rung, and it cannot repeat within the episode. The
  baseline is captured when the episode opens and re-captured after a soft reset,
  so recovery credit always describes motion since the latest recovery attempt.

The escalation, in order:

1. **Constructive reclaim (bounded)**: before rebuilding healthy children, borrow the verified reclaim ladder's
   remaining idle-only rungs and allow each issued action a settling window. The candidate list is frozen per
   episode, its cursor only advances, and both its issue allotment and aggregate time are bounded. SOS protects
   models demanded by pending work, so the rung cannot unload the exact resident model the wedged head needs.
   A post-processing module unload has one independent wall-clock grace and is issued at most once per episode;
   it cannot reset its own deadline indefinitely. A rung whose own promised free is effectively zero is skipped
   rather than issued: it has nothing to give back, so spending a settling window on it is indistinguishable
   from doing nothing while the wedge stands. Lane rungs are priced as their stopped process's CUDA-context
   charge plus its allocator reservation, since stopping the process is what returns the context, so a lane
   with a context on the card never prices as zero and remains available as a remedy. The lane pauses the
   remedy does take are stamped with the reclaim ladder's owner (it acts through
   that actuator), so the ladder's stranded-pause backstop consults the coordinator's own receipt before
   treating one as an orphan; the coordinator's unwind remains the responsible restore. The ladder also
   judges its own relevance: every rung on it frees card memory, which addresses the head only when a
   shortfall is what holds it. When the scheduler publishes the constraint blocking the head, each issued
   rung is judged once, after its full settling window, on whether the named constraint or the
   inference-start count moved. Two consecutive rungs that moved neither set the ladder aside for the rest
   of the episode: it stops being issued and, critically, its settling windows stop counting as "a remedy
   remains" toward deferring the give-up. Without that judgement a ladder answering a constraint it has no
   purchase on cycles children on its own cadence indefinitely, each cycle both churning resident state and
   renewing the excuse that holds the give-up backstop off.
2. **Soft reset (bounded)** (`perform_soft_reset`): rebuild the process pools
   in place (kill and respawn every child, un-quarantine slots), preserving the
   configured concurrency (`max_threads`). The rebuild alone clears a transient
   wedge; the cap is deliberately not lowered, because shedding a lane on every
   wedge let a one-off blip (including one provoked by aggressive co-sampling
   tripping a sampler watchdog) ratchet throughput down and outlast its cause. The
   escalation policy still **counts** each soft reset (`limp_by_level`) so a
   persistent wedge still escalates to give-up. The parent process and the TUI stay
   attached. A transient wedge (a bad model load, a one-off deadlock) recovers here.
3. **Give up cleanly** (`give_up_on_wedged_jobs`): once resets clearly are not
   helping (e.g. a deterministic crash-on-start), stop fighting: fault the jobs
   that cannot be served so the horde reissues them, rather than wedging forever.
   If the pool is still structurally usable (for example, a queue-deadlock give-up
   with live capacity), the worker keeps running. If inference or safety capacity
   cannot be restored, SOS escalates through abort so the worker process exits
   non-zero after killing its children; the TUI supervisor then relaunches it via
   the normal unexpected-exit path. Safety capacity counts here for the same reason
   inference capacity does: it sits on every job's path, so a pool that cannot serve
   safety checks cannot serve work at all, and faulting that backlog without
   recording the pool as structurally broken would drop the queue on every cycle
   while the escalation read the pool as healthy.

   The abort is gated on a reachable restart. When no supervisor is attached and
   `exit_on_unhandled_faults` is unset, the worker declines to exit and continues
   escalating in place instead, so a refused abort never freezes the escalation that
   still has remedies to try.

   That decision crosses the lifecycle boundary as a typed `RecoveryDisposition`, not
   as an abort callback whose success is guessed from mutable shutdown state. The
   coordinator consumes the disposition directly when deciding whether to park; the
   process manager retains the same value when constructing the run record; and the
   outer entry point converts `RESTART_PROCESS` to a non-zero status only after session
   state has been persisted. Thus recovery policy, durable history, and the service
   manager always observe one outcome.

   Continuing in place is only safe while remedies remain. When the terminal rung is
   withheld *and* the escalation is spent (the terminal give-up, or a process-recovery
   rate past its rolling ceiling), the coordinator **parks recovery**
   (`WorkerState.recovery_parked`): the worker stays up, but every workload popper
   (image generation and alchemy) stops accepting work through the shared
   `WorkerState.workload_intake_paused` contract, and every automatic child
   replacement or late model-ready pool-start path is suspended across inference, safety, post-processing,
   component, VAE, and utilities lanes. Already-accepted work may still drain, and a
   shutdown that begins during the park retains its unconditional hard-abort path.
   Thus a doomed pool costs a bounded amount instead of endless process churn and job
   faulting. A park is an indefinite hold, kept off the
   time-bounded pop-pause deadline for that reason, and modelled on
   `downloads_only_hold`. It is exitable: what dooms a pool is often external (a
   co-tenant process holding the card's VRAM, an exhausted disk), so after
   `RECOVERY_PARK_REPROBE_SECONDS` the park lifts. Every episode-owned wedge latch,
   progress baseline, reclaim cursor, lane-pause receipt, and remedy budget returns to
   baseline before one fresh attempt runs. A worker whose condition cleared resumes
   serving; one still doomed parks again, bounding the churn to a cycle per interval.
   Entry is logged and recorded in the ledger once (`recovery_parked`), and the lift
   is logged once.

   Give-up is **readiness-gated**, not fixed-age. A soft reset rebuilds the pool,
   and the replacement children spend real time booting (importing torch) before
   any lane can accept a job. Faulting during that boot window drops jobs the
   just-rebuilt pool was about to run. So the escalation clock does not advance
   while the pool is still booting: give-up fires only once an inference lane has
   reached an accepting state (`is_inference_pool_ready()`, the same accepting-state
   fact whose absence the queue-deadlock detector reports as "some processes are
   starting. Waiting.") *and* the wedge has then persisted for a further grace. Two
   backstops bound the boot so a pool that never comes up still escalates: a boot
   **allowance** for an *absent* boot (no replacement child is alive and
   progressing), and a larger boot **hard cap** for a *hung* boot. While a
   replacement child is alive and still `PROCESS_STARTING` (the coordinator passes a
   liveness-aware `boot_in_progress` fact, so a child that died mid-boot does not
   count), the allowance is suppressed up to the hard cap: a merely slow-but-healthy
   boot has been observed to outlast the allowance yet still succeed, and faulting
   there would drop the very jobs the finishing boot serves. Past the hard cap even a
   still-live boot is deemed hung and the allowance applies again, so a permanently
   stuck boot still escalates. Give-up is also **latched once per cycle** (repeat wedged ticks
   are no-ops, so the ledger is not spammed), and its `recovery_abandoned` ledger
   record is written only when the pass actually did something (faulted at least
   one job, or made a terminal abort decision), never a `jobs_faulted=0` no-op. If
   a wedge persists over a ready pool past the first give-up, a **bounded
   continuation** permits exactly one more soft-reset cycle after a cool-down; a
   second give-up is then flagged terminal and abandons ship deliberately rather
   than faulting jobs on every tick forever.

   Readiness gates on a lane *accepting*, but an accepting lane can still be
   mid-preload of the head-of-queue job's model: it reads ready while the very job
   the pool is loading is still queued. Faulting there drops a job the card is
   actively materialising. So the ready-wedged clock also **defers to head-model
   materialisation**: while the scheduler reports the head-of-queue model in its
   loading state (`head_model_materializing()`, driven from the model map's loading
   entry) over an otherwise idle pool, the give-up anchor is held. The deferral is
   **bounded by the preload budget** (`preload_timeout`): a load that never lands (a
   stuck preload) stops deferring once the budget elapses, so the wedge still
   escalates. A ready lane whose head model is loading is capacity in flight, not a
   wedge over a healthy pool.

   A soft-reset rebuild requeues each live slot's in-flight job (`inference attempt
   N/max, requeuing`), granting it another attempt. So the same cycle's give-up must
   not immediately fault the very retries the rebuild just granted: a **recovery-
   granted retry is spared by the non-terminal give-up** until the rebuilt pool has
   had a dispatch opportunity (the grant is marked on the job at requeue and cleared
   when it is dispatched). The terminal give-up faults it regardless, so this defers
   the drop by at most one continuation cycle rather than preventing it.

   A queue-deadlock give-up over available capacity first checks for a **reachable
   lane-reclaim remedy**: a head that does not fit only because the idle
   post-processing lane's resident module weights hold its room is servable the
   moment that lane unloads, so faulting its backlog would drop work the card can
   run. The give-up drives the unload itself and yields for a bounded grace so the
   freed VRAM can materialise and the head re-admit. A yielded give-up refunds its
   escalation cycle (the supervisor's cool-down still applies), so the eventual
   real give-up escalates undiminished; once the grace expires without the wedge
   clearing, the ordinary give-up faults the backlog, keeping the safety valve
   reachable. A broken pool's give-up never yields: no lane reclaim restores dead
   capacity.

With `exit_on_unhandled_faults` set, the worker exits instead of limping; SOS is
the default-on alternative that prioritises continued operation. The same flag is
how an unattended operator opts into the escalation's fresh-process rung, since it
declares that something outside the worker will restart it. See
[Run the worker as a system service](../how-to/run-as-a-system-service.md).

## What counts as progress

An escalation resets only on forward motion the failure could not have produced itself. A completed job
counts unconditionally: it proves every downstream stage cleared. An inference *start* is only an attempt,
so it counts only while no downstream stage is holding accepted work. A post-processing or safety backlog
keeps admitting fresh starts while nothing leaves the stage, so crediting those starts would let the
stalled stage manufacture its own proof of recovery and close the very episode that should be climbing.
The queue-deadlock clock applies the same rule to "capacity is on its way": only an *inference* slot
booting is worth restarting it for, because only an inference slot can take the pending work. Safety and
service-lane children are cycled by recovery remedies on their own cadence, and crediting their boots as
progress would let a looping remedy hold the wedge clock at zero for as long as it keeps running.

The same rule governs the wedge verdict itself. A structural queue deadlock is excused while the scheduler
is deliberately holding the queue (a whole-card model establishing residency, a heavy head loading, a RAM
reclaim cycle, backing-off process starts) or while inference is actually running, and every consumer reads
that one verdict.

Each of those excuses is bounded by its own window, but a window alone is not enough for the whole-card one:
an establish/restore cycle can re-arm a fresh window faster than the previous one expires, which would leave
the supervisor disarmed for as long as the churn continued. So the whole-card grace is additionally charged
against a per-card rolling budget, spent at admission and charged only for physical teardown events, never
for jobs reusing a residency the card already holds. While a card is over its allowance it may not open
another establish window: the new establishment is deferred for a bounded dwell and the refusal is
re-disclosed periodically with the current spend and replenish wait, so the operator can see that residency
churn, not a genuine setup, is what is holding the queue. Past the dwell the head stops asking for the card
and ordinary measured admission serves it co-resident if the device holds its weights, so the budget brakes
rotation without ever parking a servable head. A window already granted always runs to its own duration.
Withdrawing it part-way would have the supervisor classify a teardown the scheduler itself commanded as a
wedge and reset the pools mid-teardown, adding a rebuild to the very churn the budget exists to stop. See
[Bounding residency churn](resource_governance.md#bounding-residency-churn). The wedge assessment and the give-up that acts on it must not diverge: a give-up applying
a narrower set of excuses would fault exactly the backlog the scheduler is holding for capacity that is
about to arrive.

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
post-processing at pop time, so the horde stops sending it upscale/face-fix jobs.
Recovery is **headroom-gated rather than restart-only**: because the over-commit
is a VRAM shortage the card can grow out of (a heavy resident unloads, a fixed
pool seat rotates to a smaller model), the latch clears once the parent measures
the card's free VRAM back above the post-processing peak plus a safety margin,
and a fresh fault first attempts a one-shot idle-resident VRAM reclaim so the
peak may fit without ever latching. A **proactive** gate closes the same loop
from the front: whenever the parent measures free VRAM below that requirement it
withholds post-processing advertising before any fault, killing the boot-window
burst a relaunch into heavy residents would otherwise pay. Both paths need a
truthful NVML device-free reading; a host without one keeps the reactive breaker
alone, **session-latched** (it survives a soft reset and clears only on restart).
The structural whole-card conflict that shares the latch never auto-recovers. It
mirrors the per-model unservable breaker and the self-maintenance throttle: a
worker that protects its own standing on the horde rather than bleeding dropped
jobs until the server intervenes. The dedicated post-processing lane (see
[Process lanes and job chaining](process_lanes_and_chaining.md)) is the structural
complement that keeps the breaker from being needed in the first place: its
fixed resident footprint replaces the transient per-job peak that caused the
over-commits.

## The terminal-fault-rate breaker

Every throttle above withdraws a *specific* thing: a model, a capability, a card.
That leaves the generic case uncovered. A drop stream attributable to no single
model and to no single capability still counts against the worker on the horde's
side, and the server's own breaker fires on the raw rate: a worker faulting jobs
steadily is force-set into maintenance for "dropping too many jobs" whether or not
it has diagnosed why. The **terminal-fault-rate breaker** is the worker's own
reading of that same rate, so it can stop taking work on its own terms first.

It is armed from the tracker's terminal-fault *decision*
([`JobTracker.set_terminal_fault_observer`][horde_worker_regen.process_management.jobs.job_tracker.JobTracker.set_terminal_fault_observer]),
not from the session faulted counter. That counter only moves when a faulted
result is submitted, so a breaker reading it would learn of a burst well after
the horde had already counted the drops.

Faults whose [`JobFaultOrigin`][horde_worker_regen.process_management.jobs.job_tracker.JobFaultOrigin]
is `GENERATION` or `MALFORMED_POP` are counted. A scheduling-recovery fault, an
auxiliary-prefetch give-up, and a remote-submit failure are each a verdict on
something other than the worker's ability to generate, so none may pause intake of
unrelated work; a retryable failure that requeues dropped nothing at all. The
shutdown drain is excluded too: it faults the remaining backlog deliberately, and a
worker on its way out accepts nothing either way.

`MALFORMED_POP` is the one origin counted here that the consecutive-failure pause
still excludes. It marks a job the popper handed straight back because the pop
carried no usable model name (see [Pop-boundary validation](#pop-boundary-validation)).
Nothing was generated, so it is no verdict on generating; but the horde counts the
returned job as dropped exactly like any other fault, and a stream of them is
precisely the raw rate this breaker exists to notice.

The policy is fixed in module constants (no configuration keys):
`TERMINAL_FAULT_BREAKER_THRESHOLD` (3) faults within
`TERMINAL_FAULT_BREAKER_WINDOW_SECONDS` (600 s) pause new pops for
`TERMINAL_FAULT_BREAKER_COOLDOWN_SECONDS` (300 s). A re-trip within
`TERMINAL_FAULT_BREAKER_ESCALATION_DECAY_SECONDS` (3600 s) doubles the cooldown,
up to `TERMINAL_FAULT_BREAKER_MAX_COOLDOWN_SECONDS` (1800 s); going a full decay
window without tripping resets the escalation to the base. The faults a trip acts
on are consumed by it, so the same evidence cannot re-trip the breaker the instant
the cooldown lifts.

The pause reuses the shared self-throttle deadline (see
[Self-throttle pause ownership](performance_and_backpressure.md#self-throttle-pause-ownership)),
stamped with `PopPauseOwner.FAULT_THROTTLE`, so there is one pop-pause surface
rather than a parallel one. In-flight jobs are unaffected; only new pops stop.
**Nothing but the deadline lifts the pause.** A lift conditioned on the faults
stopping would be a liveness proof the failure itself could withhold, and a
persistent condition would then hold the worker silent indefinitely; instead each
pause is bounded and the breaker simply re-trips, which is a repeating signal an
operator can read.

The poison-model quarantine feeds this breaker rather than bypassing it. The
backlog sweep's non-retryable faults are real drops the horde counts, so a
quarantine with a deep backlog trips the breaker at once. That is the intent: the
pause lets the backlog drain and the horde reissue that work elsewhere, and the
worker rejoins with the poison model already off its offer.

## The intake path is bounded and audible

Every throttle and breaker above assumes the pop loop itself is running. It is a
single coroutine, so any await inside it that never returns silences the
worker's entire intake with no error and no log line; in production that looked
like a healthy worker that simply stopped serving. Two bounds and one sentinel
close that class:

- **The pop request is bounded.** A single pop HTTP request is capped at
  `POP_REQUEST_TIMEOUT_SECONDS` (30 s), well under the transport default, so an
  unanswered peer or a stale pooled connection costs one bounded attempt instead
  of minutes of silence. A pop is cheap to retry; the loop re-issues one on the
  next tick.
- **Source media is bounded.** Fetching an already-popped job's source images is
  capped at `SOURCE_IMAGE_DOWNLOAD_TIMEOUT_SECONDS` (120 s); on expiry the job
  carries the same per-item download faults an exhausted retry loop would have
  produced and proceeds, so a dead media host cannot hold intake hostage behind
  one job.
- **Silence is disclosed.** Each early return in the pop coroutine stamps the
  gate that held it (`WorkerState.last_pop_gate`), and a per-tick sentinel warns
  when no attempt has reached the horde for 60 s with nothing deliberate to
  account for it, naming that gate; see
  [hung-process detection](process_lifecycle.md) for the sentinel and the
  watchdogs that own each condition's remedy. The full-queue gate gets one extra
  test, because a full local queue is the ordinary shape of a busy worker and
  would otherwise fully excuse the silence: full is only healthy while it drains.
  When the queue has been full with nothing dispatched and nothing completed for
  the whole span, the sentinel escalates to a distinct error naming the frozen
  span, the waiting head, and the constraint the scheduler says is blocking it.
  Disclosure is not the only consumer
  of that stamp: a gate that holds far past the sentinel's warning with no work
  completing behind it is [a wedge in its own right](#layer-3-save-our-ship-sos-escalation),
  so a condition no watchdog owns still reaches the escalation.

## Rejoining after horde-forced maintenance

Every throttle above exists so the horde never has to force this worker into
maintenance. When one does fire too late, the worker has to be able to come back,
and until this existed it could not: **horde-forced maintenance has no expiry and
no horde-side release**. The local latch
(`WorkerState.last_pop_maintenance_mode`) is cleared only by a successful pop,
which cannot happen while every pop is rejected, so a forced pause lasted until an
operator noticed and cleared it by hand. Worse, it was nearly invisible: the pop
rejection is logged once, the periodic status print is suppressed while the latch
holds, and the kudos loop returns early, so hours of five-second retries read in
the log exactly like a dead pop loop.

The episode is driven once per control-loop tick by
`HordeWorkerProcessManager._drive_server_maintenance_recovery`, and it has two
halves.

**The heartbeat.** While the latch holds, a WARNING every
`SERVER_MAINTENANCE_HEARTBEAT_INTERVAL_SECONDS` (600 s) restates how long the
worker has been in maintenance, how many pops have been rejected since it engaged
(counted in the popper, since the log line is edge-triggered), and what the worker
intends to do next: the time of the next clear attempt, the health condition
holding that attempt back, or the reason auto-clear will not run at all. One line
makes a maintenance episode unmistakable for as long as it lasts.

**The bounded auto-clear.** The worker may lift the horde's own pause, and only
the horde's own pause:

- **Eligibility.** The rejected pop carries the horde's maintenance reason, and
  the return code is the same whichever side set the flag, so that reason is the
  only discriminator the API offers. A reason the horde writes itself ("dropping
  too many jobs") marks the episode server-forced; any other reason is treated as
  somebody's deliberate choice and left standing. Separately, every deliberate
  local set (the dashboard key, a supervisor command, the attach supervisor's
  frozen-parent guard) arrives as the same `SET_SERVER_MAINTENANCE` command and
  records `WorkerState.server_maintenance_locally_intended`, which disqualifies
  auto-clear until the same surface unsets it.
- **Fitness.** An attempt only goes out while the worker can actually serve: an
  inference process free to take a job, no pop pause standing (self-throttle,
  fault-rate breaker, or operator), and no terminal fault in the last
  `SERVER_MAINTENANCE_FAULT_QUIET_SECONDS` (180 s). This is remediation evidence
  rather than a timer: after a poison-model storm the quarantine has already taken
  the cause out of rotation, and a quiet, healthy pool is the signal that rejoining
  is safe. An unfit worker **defers without consuming the attempt**, so the clear
  goes out the moment the pool recovers instead of waiting out another interval.
- **Backoff.** `SERVER_MAINTENANCE_CLEAR_BACKOFF_SECONDS` schedules the attempts
  at 600 s after the latch engaged, then 1800 s, then 3600 s, then every 3600 s.
  It never stops: a worker that gives up on rejoining is a worker an operator has
  to rescue by hand, which is the outcome this exists to prevent.
- **Re-trip escalation.** If the horde re-forces maintenance within
  `SERVER_MAINTENANCE_RETRIP_WINDOW_SECONDS` (1800 s) of a clear that worked, the
  worker was not as fit as its checks believed, so the next episode's intervals
  double, capped at `SERVER_MAINTENANCE_CLEAR_MAX_INTERVAL_SECONDS` (6 h). Going
  `SERVER_MAINTENANCE_ESCALATION_DECAY_SECONDS` (3600 s) of healthy popping
  between the clear and the next pause resets the escalation.

A successful API call is **not** treated as the end of the episode. The worker
leaves its own latch alone and lets the next successful pop clear it, exactly as
before, because work arriving is the only end-to-end proof the horde is sending
this worker jobs again. That successful pop is also what dates the escalation:
an episode the worker made attempts in and then left is the evidence one of those
attempts worked.

`auto_clear_server_maintenance` (default `true`) is the recovery path's own
config key, not a preference flag that could switch it off as a side effect (see
[Bridge configuration](bridge_config.md#recovery-from-horde-forced-maintenance)).

## The background download process

The singleton [download process](model_downloads.md) lives **outside** the process
map: it serves no jobs, so the Layer 2 hung-process sweep deliberately never
touches it. That exclusion leaves a gap the parent closes separately, because a
hard termination of the downloader mid-fetch is otherwise invisible: nothing reads
its liveness, and its last status snapshot stays frozen, forever reporting a
transfer that will never complete.

A liveness sweep on the periodic loop detects when the owned download process is
no longer alive. On the first tick that sees the death it reports it once (with
the exit code), forgets the corpse, and restarts the process, bounded as a
crash-loop: at most three automatic restarts in a rolling ten-minute window. Past
the bound the loss is reported once and the process is left down until an operator
revives it (see below), rather than spinning restarts forever.

Two knock-on effects follow the death immediately, so a frozen snapshot can never
strand work:

- The [auxiliary-prefetch](model_downloads.md#job-driven-auxiliary-prefetch)
  in-flight view yields nothing while the downloader is dead, so a job's prefetch
  deadline resolves on its **first** budget instead of deferring against a process
  that can no longer make progress: with nothing in flight the job is served without
  its reference (the inference child fetches it) rather than faulted, the death
  detection and restart being separately owned. On a restart the coordinator's
  deadlines are reset, so each still-pending auxiliary job is re-requested against the
  fresh downloader within a single download-timeout budget rather than waiting out the
  full deferral cap.
- LoRA and auxiliary feature advertising is withheld while no downloader exists:
  with nothing able to place a job's LoRAs or textual inversions on disk, offering
  the capability would only pop work the worker cannot serve.

An operator can force a revival of a dead or stuck downloader with the
`RESTART_PROCESS` supervisor command targeting the download process id, which
routes to the dedicated download-restart path (the command otherwise addresses
inference slots).

The same command also recycles a service lane (COMPONENT, VAE_LANE,
POST_PROCESS, or UTILITIES) when its id is targeted: a lane accrues host commit
charge (its permanent CUDA context and allocator arenas) that an in-process model
unload cannot return, so a full process recycle is the only way to reset a lane
whose commit has ballooned. The recycle routes through the lane's normal end and
respawn machine, marked intentional so it is not counted as a crash recovery; any
stage in flight is re-dispatched from held state by the disaggregation
orchestrator once the fresh lane appears. The safety process (id 0) and the
download process remain excluded from the lane-recycle path.

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

That degrade is a **cooldown, not a latch**. A failed write pauses the mirror for
a bounded interval and the next event after it lapses retries the file, so a
transient fault (a full disk, a briefly locked file) costs the events inside the
outage rather than every event for the rest of the run. The worker logs the pause
once when it starts, and logs on recovery how many events were held in memory
only, so the gap in the file reads as an outage instead of as a period in which
the worker did nothing. Readers of the file (`horde-log`, the support bundle)
skip a line torn by a mid-write crash and keep every whole record around it.

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

That same handler also **detaches the parent's status-queue feeder** before
multiprocessing's own exit finalizers run. The parent both owns and (through the
in-parent image-utilities lane adapter) produces to the single cross-process status
queue, which spins up a feeder thread. Once the parent stops draining that queue at
teardown, a feeder mid-send toward a child whose read end is already gone would
block indefinitely, and multiprocessing's own atexit join of that feeder has no
timeout: the whole exit hangs until an external watchdog force-kills the tree.
Cancelling the feeder's join (the buffered bytes are never read again once the
parent is exiting) makes shutdown bounded. The same detach runs at the end of a
forced hard-kill, so both the clean and the aborted exit paths are covered.

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

The generated scheduling sweep also drives a fake-clock composition over the
real scheduler and job tracker. Its jobs carry independent geometries and
payload structures, so the queue includes low/high/low and high/low demand
changes within one checkpoint alongside source-image and masked work, LoRA and
textual-inversion references, ControlNet annotation and delivery modes,
post-processing chains, batches, hires-fix, and representative sampler/scheduler
pairs. Requested thread and queue settings are resolved through the production
concurrency rules; the corpus includes the one-lane/zero-prefetch boundary, the
queue cap, and the maximum thread boundary. Initial empty, RAM-staged,
VRAM-resident, and foreign-resident states are axes, as are VRAM budgeting,
whole-card residency, normal/moderate/high performance, and eager VRAM unload.

The hand-driven child boundary completes auxiliary preparation and
post-processing through the same tracker transitions the real child messages
use. This keeps the scheduling projection deterministic while ensuring those
payloads are not merely labels around plain jobs. The subprocess projection
materializes the same descriptions as real SDK responses and sends them through
the complete manager and protocol-faithful fake children.

Generated valid traffic and capability extraction share one objective job-feature
descriptor for source-image modes, painting, known ControlNet workflows, extended
and SDXL ControlNet, LoRA, textual inversion, and post-processing. The generator
therefore cannot make a row pass by silently withholding a valid payload its worker
configuration should accept. Invalid traffic remains an explicit scripted-source
test rather than being mixed into the valid generator.

Coverage is contractual rather than inferred from the seed count. The committed
sweep must cover every pair of its scenario axes, every pair of payload axes,
and each payload choice with the queue, model, card, worker-state, and runtime
setting that receives it. A modelled-card disturbance counts only when it
changes its intended state and produces a receipt; a drawn event that remains
inapplicable fails the scenario. Generated positive rows are constrained by
model/card capability (for example, the Flux-class whole-card representative
does not request ControlNet or hires-fix). The rejected side of those boundaries
is retained in a separate fast popped-job contract rather than silently pruned.
The subprocess projection keeps at most one child event on one inference lane,
where the fake process's local job ordinal has one unambiguous global target,
and verifies that the real manager resolves the lane count declared by the
scenario.

Full-worker rows also carry per-stage deadlines for child boot, pending inference,
inference, post-processing, safety, and submission. A violation captures the oldest
stage subject, stage age, every tracked job stage and age, process state and launch,
model availability, planned VRAM, and recent action names before aborting. The
overall scenario timeout remains a final backstop, but it is no longer the first
diagnostic for a stage-local stall.

The popped-job contract constructs real SDK responses from a pairwise array over
baseline family, source mode, auxiliary-reference shape, ControlNet/workflow
shape, post-processing depth, censor posture, and geometry. It independently
asserts the capability requirements extracted from each response, then crosses
the payload corpus with every pair of ControlNet, SDXL-ControlNet, LoRA,
post-processing, img2img, painting, and NSFW settings. The simulated job source
also feeds representative plain, LoRA, post-processing, legacy/extended
ControlNet, SD1, and SDXL jobs through one-, two-, and three-card offer scopes.
Those rows vary model assignment, features, resolution ceilings, and dynamic
feature withdrawal, and require every simulated returned job to retain at least
one exact route. Heterogeneous cards must therefore remain card-scoped; only
equivalent externally visible offers may be combined. The source is held to the
same source-mode advertising rules. Explicit higher-order rows
retain XL ControlNet with both auxiliary classes, masked ControlNet combined with
a post-processing chain, and uncensored XL outpainting. These decisions are pure
and cheap, so rejected combinations and boundary arithmetic are checked without
turning the subprocess sweep into a Cartesian product.

A second fake-clock composition covers the state that a scheduling-only world
cannot represent: process generations and recovery episodes. It constructs a
testable process manager and leaves its tracker, process map, scheduler,
lifecycle manager, dispatcher, recovery coordinator, and recovery supervisor
wired by reference exactly as they are in the worker. Only child creation,
termination, and queue transport are replaced with deterministic boundaries.

Its generated replacement array pairwise-covers:

- no job reference, a preload reference, and a started inference attempt;
- unexpected child exit, recovery soft reset, and resource-driven replacement;
- zero and one prior failed attempt;
- one- and two-lane process maps; and
- an empty tail, one follower, a same-model burst, and an alternating-model tail.

Those rows assert generation changes, attempt deltas, terminal stages, stable
tail ordering, and stale-message receipts. Sequence tests then exercise the
paths a Cartesian settings test cannot: crash then successful retry, two crashes
through retry exhaustion, crash-on-start quarantine then soft-reset revival,
and persistent wedges with ready, absent, or live-but-hung replacement boots.
Recovery-episode rows hold the same queue head blocked while no motion, head
motion, follower start, follower completion, and follower fault are observed.
This distinguishes global counters changing from the blocked condition actually
changing. Follower throughput may defer one action interval, but cannot reset the
frontier or defer escalation again. Positive flows accompany the non-firing assertions, so a disabled
watchdog or inert replacement path cannot make the suite pass.

The same composition exhaustively crosses orphaned inference, safety, and
post-processing ownership with a single recoverable loss or loss through the
stage's retry bound, against both a single job and a queued burst. This verifies
that each stage requeues to its own pending state, reaches its own bounded
terminal path, and leaves unrelated followers ordered and pending.

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
