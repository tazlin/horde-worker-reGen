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
    - [Intended invariants](#intended-invariants)

This document traces a job from pop to submit and names every subsystem that
touches it. File references are relative to
`horde_worker_regen/process_management/` unless noted.

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
`HordeWorkerProcessManager._main_loop` (`process_manager.py`):

| Task                        | Cadence | Role                                                                      |
| --------------------------- | ------- | ------------------------------------------------------------------------- |
| `JobPopper.run()`           | 1s      | Pop image jobs from the API (or `CannedJobSource` in dry-run)            |
| `_process_control_loop()`   | 0.2s    | Drain IPC messages, schedule inference, dispatch safety, manage processes |
| `JobSubmitter.run()`        | 0.02s   | Upload images to R2 and submit results                                    |
| `_api_get_user_info_loop()` | 15s     | Fetch user/kudos info                                                     |
| `AlchemyCoordinator.run()`  | 1s      | Pop, dispatch, and submit alchemy forms (only when `alchemist: true`)    |

A sixth (`_bridge_data_loop`, 1s) hot-reloads `bridgeData.yaml` into
`RuntimeConfig` unless config came from environment variables.

This page traces an **image** job. Alchemy jobs follow a parallel but separate
pop→dispatch→submit loop owned entirely by `AlchemyCoordinator`; they do **not**
pass through the `JobTracker` or the stages below. See
[Architecture](architecture.md#what-this-program-does) and
[Bridge Configuration → Alchemy](bridge_config.md#alchemy).

## Shared state objects

All sub-managers receive these by reference at construction
(`process_manager.py` `__init__`); none are reassigned afterwards.

| Object                    | File                   | Owns                                                                                                           |
| ------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------- |
| `JobTracker`              | `job_tracker.py`       | Per-ID `TrackedJob` state (one `JobStage` each); faults, pop timestamps, counters. The legacy stage collections (`jobs_pending_inference`, …) are now read-only derived views over the `TrackedJob` map. |
| `ProcessMap`              | `process_map.py`       | `HordeProcessInfo` per child process: last reported state, loaded model, last control flag sent, batch amount. |
| `HordeModelMap`           | `horde_model_map.py`   | Model name → load state (`ModelLoadState`) + owning process id.                                                |
| `WorkerState`             | `worker_state.py`      | Cross-cutting flags: shutdown, last pop times, consecutive failures, kudos events.                             |
| `RuntimeConfig`           | `runtime_config.py`    | Current `reGenBridgeData` snapshot (hot-reloadable).                                                           |
| `ApiSessions`             | `api_sessions.py`      | The aiohttp and horde-sdk client sessions.                                                                     |
| `ModelMetadata`           | `model_metadata.py`    | The stable-diffusion model reference and baseline lookups.                                                     |
| `ProcessLifecycleManager` | `process_lifecycle.py` | Start/stop/replace child processes, hung-process detection.                                                    |
| `ShutdownManager`         | `shutdown_manager.py`  | Shutdown/abort coordination.                                                                                   |

## IPC model

- **Parent → child:** control messages (`HordeControlMessage` subclasses,
  `messages.py`) sent over a per-process `Pipe` via
  `HordeProcessInfo.safe_send_message`. Sends are _optimistic_: the parent
  updates `ProcessMap`/`HordeModelMap` immediately after a successful send,
  before the child confirms.
- **Child → parent:** status messages (`HordeProcessMessage` subclasses) on a
  single shared `multiprocessing.Queue`, drained by
  `MessageDispatcher.receive_and_handle_process_messages` twice per control-loop
  tick. Messages carry a `process_launch_identifier` so messages from a replaced
  process incarnation are discarded.

## Stage-by-stage walkthrough

### 1. Pop (`job_popper.py`)

`api_job_pop` runs a gauntlet of gates before any network call:

1. Not shutting down; not in consecutive-failure backoff
   (`_handle_consecutive_failures`, 3 failures → pause
   `CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS`).
2. Queue not full (`_is_queue_full`: `queue_size + 1 + (max_threads - 1)`).
3. Hold-back gate: if jobs are pending inference and nothing is pending submit,
   skip (intended as a "let the very first job complete" warm-up check; see
   _Known sharp edges_).
4. A safety process and an inference process are available.
5. Megapixelstep backpressure (`PopThrottler.should_wait_for_megapixelsteps`)
   and pop-rate throttle (`is_pop_too_soon`).

Model selection (`_select_models_for_pop`) applies model stickiness, removes
models that already have ≥2 queued jobs, and adds custom models. The pop
response then goes through `_apply_sdk_workarounds` (seed/denoise fixups; note:
this _rebuilds_ the response object),
`SourceImageDownloader.download_source_images` (img2img source fetch, faults
recorded as `GenMetadataEntry` keyed by `GenerationID`), and finally
`JobTracker.record_popped_job`, which registers a `TrackedJob` in stage
`PENDING_INFERENCE` (surfaced through the `jobs_pending_inference`,
`jobs_lookup`, and `job_pop_timestamps` derived views).

In dry-run (`dry_run_skip_api`), `CannedJobSource` replaces the API call;
everything downstream is identical.

### 2. Schedule and dispatch to a process (`inference_scheduler.py`)

`_process_control_loop` invokes `InferenceScheduler.run_scheduling_cycle` when
there are pending jobs and a free process or preloaded model. One cycle:

1. **`preload_models()`**: for the first pending job whose model isn't loaded
   anywhere, pick an available process
   (`ProcessMap.get_first_available_inference_process`), send `PRELOAD_MODEL`,
   mark the model `LOADING` in `HordeModelMap`. Concurrent-preload limits differ
   under `very_fast_disk_mode`.
2. **Look-ahead**: `get_next_job_and_process(information_only=True)` peeks at
   the next runnable job to decide heavy-model/batch blocking. This method is
   called _twice_ per cycle (peek, then launch) and must agree with itself;
   line-skip decisions (a small job jumping ahead of one blocked on a LoRA
   download) are cached in `_pending_line_skip` to keep the two calls
   consistent.
3. **Blocking rules**: `ProcessMap.keep_single_inference` (active batch,
   VRAM-heavy model, ControlNet-XL, post-processing overlap) and a
   batch/heavy-workflow check can defer launch.
4. **`start_inference()`**: sends `START_INFERENCE` with the full
   `ImageGenerateJobPopResponse`; on success `JobTracker.mark_inference_started`
   adds the job to `jobs_in_progress`. **The job stays in
   `jobs_pending_inference` too**; it is removed only when the result message
   arrives. On send failure the job faults straight to `jobs_pending_submit`
   (`handle_job_fault`).
5. **`unload_models()` / `unload_models_from_vram()`**: evict idle models not
   needed by the upcoming queue (LRU-informed).

### 3. Inference (child process, `inference_process.py`)

`HordeInferenceProcess` receives the control message, runs hordelib, and streams
back heartbeats, memory reports, and state changes, ending with
`HordeInferenceResultMessage` (state `ok`/`censored`/`faulted` + base64 images).
In dry-run, `fake_worker_processes.py` substitutes canned image results.

### 4. Result intake (`message_dispatcher.py`)

`_handle_inference_result`: remove from `jobs_in_progress` and
`jobs_pending_inference`; on success copy state/timing/images into the
`HordeJobInfo` and `queue_for_safety` (→ `jobs_pending_safety_check`); on fault,
`queue_for_submit` directly (faults are still reported to the API).

### 5. Safety (`safety_orchestrator.py` → child → dispatcher)

`start_evaluate_safety` takes the head of `jobs_pending_safety_check`, validates
required fields (faulting the job on any `None`), and sends `EVALUATE_SAFETY` to
the safety process; on send success `begin_safety_check` moves it to
`jobs_being_safety_checked`. If the safety process died, jobs are requeued and
the process flagged for replacement.

The reply (`HordeSafetyResultMessage`) is handled by `_handle_safety_result`:
`take_being_safety_checked(job_id)`, apply per-image censorship/CSAM/NSFW
metadata and final `GENERATION_STATE`, merge accumulated source-image faults,
then `queue_for_submit`.

### 6. Submit (`job_submitter.py`)

`api_submit_job` takes the head of `jobs_pending_submit` and fans out one
`PendingSubmitJob` task per image: PUT the PNG to the job's `r2_upload` URL (10s
timeout, retry on 500/timeouts), then a `JobSubmitRequest` to the API.
`PendingSubmitJob` owns retry/fault bookkeeping. Kudos are recorded into
`WorkerState`; consecutive-failure counters reset on success. Finally
`ensure_submitted_job_info` + `finalize_submitted` remove the job from every
collection and stamp `last_job_submitted_time`. Dry-run short-circuits the whole
method per image.

## Intended invariants

These are the properties the pipeline assumes but does not currently enforce
structurally:

1. **Single stage:** at any instant a job is in exactly one of
   {pending*inference, in_progress, pending_safety_check, being_safety_checked,
   pending_submit}, \_except* that in_progress jobs intentionally also remain
   in pending_inference until their result arrives.
2. **Lookup lifetime:** `jobs_lookup` and `job_pop_timestamps` contain a job
   from `record_popped_job` until `finalize_submitted`.
3. **No loss:** every popped job eventually reaches `finalize_submitted`
   (success or fault); nothing is silently dropped.
4. **Identity stability:** stage collections are keyed by the
   `ImageGenerateJobPopResponse` _object value_ (pydantic equality/hash); any
   code that rebuilds the response (e.g. `_apply_sdk_workarounds`) must do so
   **before** `record_popped_job`, or lookups will miss.
5. **Optimistic sends converge:** parent-side `ProcessMap`/`HordeModelMap`
   mutations made at send time are eventually confirmed or corrected by child
   state messages.
