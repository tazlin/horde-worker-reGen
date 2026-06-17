# Job State Machine

- [Job State Machine](#job-state-machine)
    - [Why a state machine?](#why-a-state-machine)
    - [The stages](#the-stages)
    - [The stage dual-presence rule](#the-stage-dual-presence-rule)
    - [Allowed transitions](#allowed-transitions)
    - [The DETACHED stage](#the-detached-stage)
    - [Job faults](#job-faults)
    - [Lookup lifetime](#lookup-lifetime)
    - [Identity stability](#identity-stability)

Every **image** job the worker knows about is exactly one `TrackedJob` in a
single `dict[GenerationID, TrackedJob]`, with an explicit `JobStage`. All stage
changes go through one transition method that validates legality; "a job is in
exactly one stage" is enforced structurally, not by convention. (Alchemy forms
are tracked separately by `AlchemyCoordinator` and never enter this machine.)

Each `TrackedJob` also records a `stage_timestamps` map: the first time it
entered each stage (plus `FINALIZED`). On finalize, a registered observer folds
those into the worker run metrics (queue-wait, end-to-end, and safety latencies);
see [Architecture → Metrics and observability](architecture.md#metrics-and-observability).

## Why a state machine?

The previous implementation used separate collections (`jobs_pending_inference`,
`jobs_in_progress`, etc.) and moved jobs between them with ad-hoc mutations
spread across multiple components. A job could silently end up in two
collections, or in none. The state machine eliminates this class of bug: every
transition is validated, and the current stage is a single source of truth.

## The stages

| Stage                   | Meaning                                                         |
| ----------------------- | --------------------------------------------------------------- |
| `PENDING_INFERENCE`     | Popped, waiting for dispatch to an inference process            |
| `INFERENCE_IN_PROGRESS` | Sent to an inference process, awaiting result                   |
| `DETACHED`              | Tracked but not in any queue; transient hand-off between stages |
| `PENDING_SAFETY_CHECK`  | Inference finished; waiting for a safety process                |
| `SAFETY_CHECKING`       | Sent to safety process, awaiting verdict                        |
| `PENDING_SUBMIT`        | Ready for API submission (success or fault)                     |

## The stage dual-presence rule

The one intentional exception to "one stage at a time": a job in
`INFERENCE_IN_PROGRESS` is **also** visible in the `jobs_pending_inference`
derived view. This is because queue-size accounting depends on knowing how many
jobs are still "in the pipeline" before submission. The job leaves
`jobs_pending_inference` only when the inference result (or a fault) arrives.
The underlying `JobStage` is still `INFERENCE_IN_PROGRESS`; the dual visibility
is a derived-view concern, not a stage violation.

## Allowed transitions

```
PENDING_INFERENCE → INFERENCE_IN_PROGRESS   (inference started)
                  → DETACHED                 (transient)
                  → PENDING_SAFETY_CHECK     (inference complete)
                  → PENDING_SUBMIT           (faulted before inference)

INFERENCE_IN_PROGRESS → PENDING_INFERENCE   (requeued)
                      → DETACHED             (transient)
                      → PENDING_SAFETY_CHECK (result received)
                      → PENDING_SUBMIT       (faulted during inference)

DETACHED → PENDING_SAFETY_CHECK
         → PENDING_SUBMIT
         → PENDING_INFERENCE

PENDING_SAFETY_CHECK → SAFETY_CHECKING       (dispatched to safety)
                     → PENDING_SUBMIT        (faulted, skip safety)
                     → DETACHED

SAFETY_CHECKING → DETACHED                   (transient)
                → PENDING_SAFETY_CHECK       (requeued)
                → PENDING_SUBMIT             (safety verdict received)

PENDING_SUBMIT → (job removed from tracker after finalization)
```

## The DETACHED stage

`DETACHED` is a transient stage for hand-offs between components. A job must not
remain `DETACHED` across loop iterations. It exists so that `queue_for_safety`
and `queue_for_submit` can atomically move a job out of its current stage
without requiring the next component to "know" where it came from.

## Job faults

Jobs can fault at any stage: source image download failure, inference crash,
safety evaluation error, submission timeout. Faults are recorded as
`GenMetadataEntry` objects keyed by `GenerationID` in the `job_faults`
collection. Faulted jobs skip straight to `PENDING_SUBMIT` (or
`PENDING_SAFETY_CHECK` if the fault occurred before inference). The submitter
reports all accumulated faults back to the API.

Fault entries are **kept independent of job lifetime**; they survive
`finalize_submitted` and must be explicitly cleared via `clear_faults_for_job`.
This prevents faults being lost if they are recorded after the job is finalized.

## Lookup lifetime

- `jobs_lookup[response] = HordeJobInfo` is set at pop time and removed at
  `finalize_submitted`.
- `job_pop_timestamps[job_id]` is set at pop time and removed at
  `finalize_submitted`.
- Stage collections are derived views; adding a job inserts it, removing it from
  the tracker removes it from all views.

## Identity stability

Stage collections are keyed by the `ImageGenerateJobPopResponse` object value
(pydantic `__eq__` / `__hash__`). **Any code that rebuilds the response object
must do so before `record_popped_job`**, or lookups will fail silently. The
`_apply_sdk_workarounds` function in `job_popper.py` is the only place that
rebuilds the response, and it runs before the job is recorded.

## See also

- [Job Lifecycle](job_lifecycle.md): how the stages connect to subsystems
- [Architecture](architecture.md): where
  [`JobTracker`][horde_worker_regen.process_management.job_tracker.JobTracker]
  fits in the shared-state pattern
- [Shutdown and Faults](shutdown_and_faults.md): fault propagation across
  stages
- [`JobStage`][horde_worker_regen.process_management.job_tracker.JobStage]
- [`TrackedJob`][horde_worker_regen.process_management.job_tracker.TrackedJob]
