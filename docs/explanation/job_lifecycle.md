# Job Lifecycle and Major Subsystems

- [Job Lifecycle and Major Subsystems](#job-lifecycle-and-major-subsystems)
    - [The pipeline at a glance](#the-pipeline-at-a-glance)
    - [Shared state objects](#shared-state-objects)
    - [IPC model](#ipc-model)
    - [Stage-by-stage walkthrough](#stage-by-stage-walkthrough)
        - [1. Pop (`job_popper.py`)](#1-pop-job_popperpy)
        - [2. Schedule and dispatch to a process (`inference_scheduler.py`)](#2-schedule-and-dispatch-to-a-process-inference_schedulerpy)
        - [3. Inference (child process, `inference_process.py`)](#3-inference-child-process-inference_processpy)
        - [4. Result intake (`message_dispatcher.py`)](#4-result-intake-message_dispatcherpy)
        - [5. Safety (`safety_orchestrator.py` → child → dispatcher)](#5-safety-safety_orchestratorpy--child--dispatcher)
        - [6. Submit (`job_submitter.py`)](#6-submit-job_submitterpy)
    - [Pipeline invariants](#pipeline-invariants)
    - [See also](#see-also)

A **job** is the worker's unit of work: one image request popped from the AI Horde, generated,
checked, and submitted back. This page follows a single job through that journey and names the
subsystem that owns it at each step. It is the spine the other explanation pages hang off: once you
know which component holds a job in which phase, the [scheduling](performance_and_backpressure.md),
[IPC](ipc_and_messaging.md), [process-management](process_lifecycle.md), and
[recovery](resilience_and_recovery.md) pages each go deeper on one part of the path traced here. Read
this first; reach for those when you need the detail.

Where a phase has subtle rules (queue accounting, backpressure, fault handling), this page summarises
them and links to the page that treats them in full rather than repeating them. File references are
relative to `horde_worker_regen/process_management/` unless noted.

## The pipeline at a glance

```
                         (asyncio task)            (asyncio task, 0.2s tick)              (asyncio task, 0.02s tick)
                        ┌────────────┐   ┌──────────────────────────────────────────┐   ┌──────────────┐
   AI Horde API ──pop──►│ JobPopper  │   │        _process_control_loop             │   │ JobSubmitter │──submit──► AI Horde API
                        └─────┬──────┘   │  MessageDispatcher / InferenceScheduler  │   └──────▲───────┘            + R2 upload
                              │          │  SafetyOrchestrator / ProcessLifecycle   │          │
                              │          └──────┬───────────────▲───────────────────┘          │
                              ▼                 │ Pipe (control)│ mp.Queue (status)            │
                       ╔══════════════════════════════════════════════════════════════════════╗
                       ║                      JobTracker (stage collections)                   ║
                       ║  pending_inference ─► in_progress ─► pending_safety_check             ║
                       ║       ─► being_safety_checked ─► pending_submit ─► (removed)          ║
                       ╚══════════════════════════════════════════════════════════════════════╝
                                                │                       ▲
                                                ▼                       │
                                  ┌──────────────────────┐   ┌──────────────────────┐
                                  │ HordeInferenceProcess│   │  HordeSafetyProcess  │   (child processes)
                                  └──────────────────────┘   └──────────────────────┘
```

Five long-lived asyncio tasks are started by
[`HordeWorkerProcessManager`][horde_worker_regen.process_management.process_manager.HordeWorkerProcessManager]'s
`_main_loop` (`process_manager.py`). They share no thread; they cooperate only through the
[shared-state objects](#shared-state-objects) below, which is what keeps the design analysable:

| Task                        | Cadence | Role                                                                      |
| --------------------------- | ------- | ------------------------------------------------------------------------- |
| `JobPopper.run()`           | 1s      | Pop image jobs from the API (or `CannedJobSource` in dry-run)            |
| `_process_control_loop()`   | 0.2s    | Drain IPC messages, schedule inference, dispatch safety, manage processes |
| `JobSubmitter.run()`        | 0.02s   | Upload images to R2 and submit results                                    |
| `_api_get_user_info_loop()` | 15s     | Fetch user/kudos info                                                     |
| `AlchemyCoordinator.run()`  | 1s      | Pop, dispatch, and submit alchemy forms (only when `alchemist: true`)    |

A sixth (`_bridge_data_loop`, 1s) hot-reloads `bridgeData.yaml` into
[`RuntimeConfig`][horde_worker_regen.process_management.runtime_config.RuntimeConfig] unless config
came from environment variables.

This page traces an **image** job. Alchemy jobs follow a parallel but separate pop→dispatch→submit
loop owned entirely by
[`AlchemyCoordinator`][horde_worker_regen.process_management.alchemy_popper.AlchemyCoordinator]; they
do **not** pass through the `JobTracker` or the stages below. See
[Architecture](architecture.md#what-this-program-does),
[Bridge Configuration → Alchemy](bridge_config.md#alchemy), and
[Performance → Alchemy backpressure](performance_and_backpressure.md#alchemy-backpressure).

## Shared state objects

All sub-managers receive these by reference at construction (`process_manager.py` `__init__`); none
are reassigned afterwards, so every task sees the same live state. The `JobTracker` is the heart of
the pipeline (it holds the job itself); the rest describe the processes, models, and config the job
flows through.

| Object                    | File                   | Owns                                                                                                           |
| ------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------- |
| [`JobTracker`][horde_worker_regen.process_management.job_tracker.JobTracker] | `job_tracker.py` | Per-ID `TrackedJob` state (one `JobStage` each); faults, pop timestamps, counters. The legacy stage collections (`jobs_pending_inference`, …) are now read-only derived views over the `TrackedJob` map. See [Job State Machine](job_state_machine.md). |
| [`ProcessMap`][horde_worker_regen.process_management.process_map.ProcessMap] | `process_map.py` | `HordeProcessInfo` per child process: last reported state, loaded model, last control flag sent, batch amount. |
| [`HordeModelMap`][horde_worker_regen.process_management.horde_model_map.HordeModelMap] | `horde_model_map.py` | Model name → load state (`ModelLoadState`) + owning process id.                                                |
| [`WorkerState`][horde_worker_regen.process_management.worker_state.WorkerState] | `worker_state.py` | Cross-cutting flags: shutdown, last pop times, consecutive failures, kudos events.                             |
| [`RuntimeConfig`][horde_worker_regen.process_management.runtime_config.RuntimeConfig] | `runtime_config.py` | Current `reGenBridgeData` snapshot (hot-reloadable).                                                           |
| [`ApiSessions`][horde_worker_regen.process_management.api_sessions.ApiSessions] | `api_sessions.py` | The aiohttp and horde-sdk client sessions.                                                                     |
| [`ModelMetadata`][horde_worker_regen.process_management.model_metadata.ModelMetadata] | `model_metadata.py` | The stable-diffusion model reference and baseline lookups.                                                     |
| [`ProcessLifecycleManager`][horde_worker_regen.process_management.process_lifecycle.ProcessLifecycleManager] | `process_lifecycle.py` | Start/stop/replace child processes, hung-process detection.                                                    |
| [`ShutdownManager`][horde_worker_regen.process_management.shutdown_manager.ShutdownManager] | `shutdown_manager.py` | Shutdown/abort coordination.                                                                                   |

## IPC model

This is the short version; the [IPC and messaging](ipc_and_messaging.md) page is the full treatment
(message catalogue, the optimistic-send pattern, and replacement-incarnation filtering).

- **Parent → child:** control messages (`HordeControlMessage` subclasses, `messages.py`) sent over a
  per-process `Pipe` via `HordeProcessInfo.safe_send_message`. Sends are _optimistic_: the parent
  updates `ProcessMap`/`HordeModelMap` immediately after a successful send, before the child confirms.
- **Child → parent:** status messages (`HordeProcessMessage` subclasses) on a single shared
  `multiprocessing.Queue`, drained by
  [`MessageDispatcher`][horde_worker_regen.process_management.message_dispatcher.MessageDispatcher]'s
  `receive_and_handle_process_messages` twice per control-loop tick. Messages carry a
  `process_launch_identifier` so messages from a replaced process incarnation are discarded.

## Stage-by-stage walkthrough

### 1. Pop (`job_popper.py`)

[`JobPopper`][horde_worker_regen.process_management.job_popper.JobPopper]'s `api_job_pop` runs a
gauntlet of gates before any network call, so the worker never pulls work it cannot place. In brief:

1. Not shutting down; not in consecutive-failure backoff (`_handle_consecutive_failures`, 3 failures →
   pause `CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS`).
2. Queue not full (`_is_queue_full`: `queue_size + 1 + (max_threads - 1)`).
3. Hold-back gate: while jobs are pending inference but none has completed yet this session, skip. This
   is a warm-up guard, letting the very first job finish before pulling more.
4. A safety process and an inference process are available.
5. Megapixelstep backpressure (`PopThrottler.should_wait_for_megapixelsteps`) and pop-rate throttle
   (`is_pop_too_soon`).

Each gate, and the reasoning behind the `+ 1 + (max_threads - 1)` queue headroom, is treated in full
in [Performance → The pop gauntlet](performance_and_backpressure.md#the-pop-gauntlet).

Model selection (`_select_models_for_pop`) applies [model
stickiness](performance_and_backpressure.md#model-stickiness), removes models that already have ≥2
queued jobs, and adds custom models. The pop response then goes through `_apply_sdk_workarounds`
(seed/denoise fixups; note this _rebuilds_ the response object, which the [identity-stability
invariant](#pipeline-invariants) depends on),
[`SourceImageDownloader`][horde_worker_regen.process_management.source_image_downloader.SourceImageDownloader]'s
`download_source_images` (img2img source fetch, faults recorded as `GenMetadataEntry` keyed by
`GenerationID`), and finally `JobTracker.record_popped_job`, which registers a `TrackedJob` in stage
`PENDING_INFERENCE` (surfaced through the `jobs_pending_inference`, `jobs_lookup`, and
`job_pop_timestamps` derived views).

In dry-run (`dry_run_skip_api`), `CannedJobSource` replaces the API call; everything downstream is
identical.

### 2. Schedule and dispatch to a process (`inference_scheduler.py`)

`_process_control_loop` invokes
[`InferenceScheduler`][horde_worker_regen.process_management.inference_scheduler.InferenceScheduler]'s
`run_scheduling_cycle` when there are pending jobs and a free process or preloaded model. One cycle:

1. **`preload_models()`**: for the first pending job whose model isn't loaded anywhere, pick an
   available process (`ProcessMap.get_first_available_inference_process`), send `PRELOAD_MODEL`, mark
   the model `LOADING` in `HordeModelMap`. The preload is subject to the [VRAM and RAM
   budget](performance_and_backpressure.md#the-vram-and-ram-budget); concurrent-preload limits differ
   under `very_fast_disk_mode`.
2. **Look-ahead**: `get_next_job_and_process(information_only=True)` peeks at the next runnable job to
   decide heavy-model/batch blocking. This method is called _twice_ per cycle (peek, then launch) and
   must agree with itself; line-skip decisions (a small job jumping ahead of one blocked on a LoRA
   download) are cached in `_pending_line_skip` to keep the two calls consistent.
3. **Blocking rules**: `ProcessMap.keep_single_inference` (active batch, VRAM-heavy model,
   ControlNet-XL, post-processing overlap) and a batch/heavy-workflow check can defer launch.
4. **`start_inference()`**: sends `START_INFERENCE` with the full `ImageGenerateJobPopResponse`; on
   success `JobTracker.mark_inference_started` moves the job to `INFERENCE_IN_PROGRESS`. By the
   [dual-presence rule](job_state_machine.md#the-stage-dual-presence-rule) it stays visible in the
   `jobs_pending_inference` view until the result arrives. On send failure the job faults straight to
   `PENDING_SUBMIT` (`handle_job_fault`).
5. **`unload_models()` / `unload_models_from_vram()`**: evict idle models not needed by the upcoming
   queue (LRU-informed; see [model eviction](performance_and_backpressure.md#model-eviction-lru)).

The full scheduling-priority rationale (performance-model scoring, model affinity, the line-skip
cache) is in [Performance → Inference scheduling
priorities](performance_and_backpressure.md#inference-scheduling-priorities).

### 3. Inference (child process, `inference_process.py`)

[`HordeInferenceProcess`][horde_worker_regen.process_management.inference_process.HordeInferenceProcess]
receives the control message, runs hordelib, and streams back heartbeats, memory reports, and state
changes, ending with `HordeInferenceResultMessage` (state `ok`/`censored`/`faulted` + base64 images).
A job that stops reporting progress is graded and recovered by the watchdogs in [Resilience and
Recovery](resilience_and_recovery.md). In dry-run, `fake_worker_processes.py` substitutes canned image
results.

### 4. Result intake (`message_dispatcher.py`)

`_handle_inference_result`: take the job out of the `jobs_in_progress`/`jobs_pending_inference` views;
on success copy state/timing/images into the `HordeJobInfo` and `queue_for_safety` (→
`PENDING_SAFETY_CHECK`); on fault, `queue_for_submit` directly (faults are still reported to the API).

If this result message never arrives (it can be dropped during a concurrent slot replacement), the job
is recovered rather than stranded in `INFERENCE_IN_PROGRESS`: see [stranded in-progress
jobs](resilience_and_recovery.md#stranded-in-progress-jobs).

### 5. Safety (`safety_orchestrator.py` → child → dispatcher)

[`SafetyOrchestrator`][horde_worker_regen.process_management.safety_orchestrator.SafetyOrchestrator]'s
`start_evaluate_safety` takes the head of `jobs_pending_safety_check`, validates required fields
(faulting the job on any `None`), and sends `EVALUATE_SAFETY` to the
[`HordeSafetyProcess`][horde_worker_regen.process_management.safety_process.HordeSafetyProcess]; on
send success `begin_safety_check` moves it to `SAFETY_CHECKING`. If the safety process died, jobs are
requeued and the process flagged for replacement.

The reply (`HordeSafetyResultMessage`) is handled by `_handle_safety_result`:
`take_being_safety_checked(job_id)`, apply per-image censorship/CSAM/NSFW metadata and final
`GENERATION_STATE`, merge accumulated source-image faults, then `queue_for_submit`.

### 6. Submit (`job_submitter.py`)

[`JobSubmitter`][horde_worker_regen.process_management.job_submitter.JobSubmitter]'s `api_submit_job`
takes the head of `jobs_pending_submit` and fans out one `PendingSubmitJob` task per image: PUT the
PNG to the job's `r2_upload` URL (10s timeout, retry on 500/timeouts), then a `JobSubmitRequest` to
the API. `PendingSubmitJob` owns retry/fault bookkeeping. Kudos are recorded into `WorkerState`;
consecutive-failure counters reset on success. Finally `ensure_submitted_job_info` +
`finalize_submitted` remove the job from the tracker entirely and stamp `last_job_submitted_time`.
Dry-run short-circuits the whole method per image.

## Pipeline invariants

The pipeline depends on a handful of invariants. The first three are now enforced **structurally** by
the `JobTracker` state machine, which is the authority on them: see [Job State
Machine](job_state_machine.md). The last two are properties the wider pipeline maintains by
construction and recovery rather than by a single guard.

1. **Single stage:** a job is in exactly one `JobStage` at a time, the one intentional exception being
   that an `INFERENCE_IN_PROGRESS` job stays visible in the `jobs_pending_inference` derived view until
   its result arrives (the [dual-presence rule](job_state_machine.md#the-stage-dual-presence-rule)).
   Every stage change goes through one validated transition method.
2. **Identity stability:** stage lookups are keyed by the `ImageGenerateJobPopResponse` _object value_
   (pydantic equality/hash), so any code that rebuilds the response (e.g. `_apply_sdk_workarounds`)
   must do so **before** `record_popped_job`, or lookups will miss. See [Identity
   stability](job_state_machine.md#identity-stability).
3. **Lookup lifetime:** `jobs_lookup` and `job_pop_timestamps` hold a job from `record_popped_job`
   until `finalize_submitted`.
4. **No loss:** every popped job eventually reaches `finalize_submitted` (success or fault); nothing is
   silently dropped. This is upheld across crashes and dropped messages by the watchdogs in
   [Resilience and Recovery](resilience_and_recovery.md), not by a single check.
5. **Optimistic sends converge:** the parent-side `ProcessMap`/`HordeModelMap` mutations made at send
   time are eventually confirmed or corrected by child state messages. See the optimistic-send pattern
   in [IPC and messaging](ipc_and_messaging.md).

## See also

- [Job State Machine](job_state_machine.md): how `JobTracker` enforces the stages and transitions
  named above
- [Architecture](architecture.md): the shared-state pattern and the asyncio loop these tasks run on
- [IPC and messaging](ipc_and_messaging.md): the pipe/queue model summarised under [IPC
  model](#ipc-model)
- [Performance and backpressure](performance_and_backpressure.md): the pop gauntlet, scheduling, and
  eviction this page links into at each stage
- [Process lifecycle](process_lifecycle.md): starting, monitoring, and replacing the child processes
- [Resilience and recovery](resilience_and_recovery.md): how faults and lost results are recovered so
  the no-loss invariant holds
