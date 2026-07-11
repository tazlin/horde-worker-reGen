# Architecture

- [Architecture](#architecture)
    - [What this program does](#what-this-program-does)
    - [Why multiple processes?](#why-multiple-processes)
    - [The shared-state pattern](#the-shared-state-pattern)
    - [The asyncio tasks](#the-asyncio-tasks)
    - [The three process types](#the-three-process-types)
    - [IPC: pipes and a queue](#ipc-pipes-and-a-queue)
    - [Metrics and observability](#metrics-and-observability)
    - [The optional frontend](#the-optional-frontend)
    - [Dry-run mode](#dry-run-mode)
    - [Where the code lives](#where-the-code-lives)
    - [See also](#see-also)

This page describes the architecture of the horde-worker-reGen process at the
level you need to understand how the pieces fit together. Read this first; the
subsystem-specific pages assume you know the concepts here.

## What this program does

The worker sits in a tight loop: **pop a job from the AI Horde API → run
inference in a child process → run the NSFW/CSAM safety filter in another child
process → upload the result images to R2 → submit the result to the API**. Every
job passes through every stage; faults at any stage are still reported back to
the API so the horde can track them.

A worker can additionally opt into **alchemy** jobs (`alchemist: true`):
post-processing/interrogation work pulled from `/v2/interrogate/pop`: upscalers,
face-fixers, background removal, captioning, interrogation, and NSFW
classification. Alchemy runs on a separate pop→dispatch→submit loop
([`AlchemyCoordinator`][horde_worker_regen.process_management.jobs.alchemy_popper.AlchemyCoordinator])
that reuses the same child processes the image pipeline already owns; it does not
have its own process pool. See [Bridge Configuration](bridge_config.md#alchemy).

## Workloads (flows)

Image generation and alchemy are two **workload flows** the worker orchestrates over one shared pool
of child processes ([`WorkloadKind`][horde_worker_regen.process_management.scheduling.workload_flow.WorkloadKind];
audio and video generation are the intended next flows). Each flow is its own pop→dispatch→submit
loop. What they share is threefold:

- **The process pool**, routed by capability rather than by process type: each process declares a
  [`WorkerCapability`][horde_worker_regen.process_management.lifecycle.horde_process.WorkerCapability]
  (inference processes serve `IMAGE_GEN` and `ALCHEMY_GRAPH`; the safety process serves `SAFETY_EVAL`
  and `ALCHEMY_CLIP`), and work is dispatched to the first process declaring the capability it needs.
- **One resource budget.** A single
  [`CommittedReserveLedger`][horde_worker_regen.process_management.resources.resource_budget.CommittedReserveLedger]
  records the VRAM (and RAM) each flow has admitted but not yet allocated, so every admission gate
  subtracts the *same* combined figure. This is what stops image generation and alchemy independently
  admitting work against the same free VRAM and over-committing the device. Admission is two layers: a
  *fairness* layer (image work wins contended lanes; alchemy backfills or takes only spare lanes) sits
  above a *capacity* layer (the shared budget) that decides whether the device can physically hold the
  work at all.

Each flow presents a uniform
[`FlowCoordinator`][horde_worker_regen.process_management.scheduling.workload_flow.FlowCoordinator]
surface (`kind`, `num_in_flight`, `run`). Alchemy satisfies it directly with `AlchemyCoordinator`;
image generation does so through
[`ImageGenerationCoordinator`][horde_worker_regen.process_management.jobs.image_coordinator.ImageGenerationCoordinator],
which wraps the existing image popper, submitter, and
[`JobTracker`][horde_worker_regen.process_management.jobs.job_tracker.JobTracker] (still the owner of
image-generation job state) rather than replacing them. The process manager holds a `WorkloadKind`-keyed
registry of these flows and launches each flow's `run` uniformly, so the snapshot and dashboard read
per-workload work counts the same way for every flow, and a future audio/video flow plugs into the
registry rather than getting a bespoke launch path. Which workloads a worker actually serves is the
separate, declarative question answered by
[`capabilities.enabled_workloads`][horde_worker_regen.capabilities.enabled_workloads] (derived from the
`dreamer`/`alchemist` role flags and the install), which drives process sizing and the dashboard's mode
identity; an alchemist-only worker is simply one whose served set is `{alchemy}`.

## Why multiple processes?

AI generation inference is VRAM-heavy and stateful. ComfyUI, which is used
under the hood, is generally not thread-safe, and so to avoid race conditions
and crashes, each GPU slot gets its own process. This allows for situations like
pre-loading a model in RAM while another process is running inference, and
running safety evaluation in parallel with inference for a different job.

The typical worker configuration has 2-4 inference processes and 1 safety
process. The main process orchestrates everything but does not use torch or
touch the GPU directly. The benefits of this architecture are:

- **Isolation**: a crash in one inference process does not take down the others
  or the safety process. Recovery means simply restarting the failed process,
  not the whole worker.
- **Model persistence**: a child can keep a model loaded in VRAM across multiple
  jobs, avoiding reload latency.
- **Parallelism**: inference and safety evaluation can happen concurrently for
  different jobs.

## The shared-state pattern

All sub-managers receive references to the same shared objects at construction
time. Nothing is reassigned afterwards; the references are set once and held.
This means any component can read the latest state of any other component
without callbacks, global variables, or event buses.

The shared objects are:

| Object          | File                 | What it owns                                                           |
| --------------- | -------------------- | ---------------------------------------------------------------------- |
| `JobTracker`    | `job_tracker.py`     | Per-ID `TrackedJob` stages (collections are derived views), faults, pop timestamps, completion counters |
| `ProcessMap`    | `process_map.py`     | `HordeProcessInfo` per child process                                   |
| `HordeModelMap` | `horde_model_map.py` | Model name → load state + owning process id                            |
| `WorkerState`   | `worker_state.py`    | Cross-cutting flags: shutdown, pop times, consecutive failures, kudos  |
| `RuntimeConfig` | `runtime_config.py`  | Live `reGenBridgeData` snapshot (hot-reloadable)                       |
| `ApiSessions`   | `api_sessions.py`    | aiohttp and horde-sdk client sessions                                  |
| `ModelMetadata` | `model_metadata.py`  | Stable-diffusion model reference lookups                               |

## The asyncio tasks

Several long-lived asyncio tasks are started by `HordeWorkerProcessManager._main_loop`
(`process_manager.py`): the non-flow loops below, plus one `run()` per registered workload flow.

| Task                          | Cadence | Role                                                                                      |
| ----------------------------- | ------- | ----------------------------------------------------------------------------------------- |
| `_process_control_loop()`     | 0.2 s   | Drain IPC messages, schedule inference, dispatch safety, manage processes                 |
| `_api_get_user_info_loop()`   | 15 s    | Fetch user/kudos info                                                                     |
| `_periodic_update_check_loop()` | -     | Check for a newer worker release                                                          |
| `ImageGenerationCoordinator.run()` | -  | Supervise the image `JobPopper` (1 s) and `JobSubmitter` (0.02 s) loops                   |
| `AlchemyCoordinator.run()`    | 1 s     | Pop, dispatch, and submit alchemy forms (only when `alchemist: true`; otherwise idle)     |

The two coordinator tasks are launched uniformly from the flow registry (`self._flows`, keyed by
`WorkloadKind`). The image flow keeps the same per-loop shutdown supervision the popper and submitter
had as top-level tasks: each is launched carrying `_handle_exception`, so if one ends the worker shuts
down gracefully while the other keeps draining in-flight work. A further task
(`BridgeDataReloader.bridge_data_loop`, 1 s) hot-reloads `bridgeData.yaml` into `RuntimeConfig` unless
the config came from environment variables.

These tasks run concurrently in the same asyncio event loop. The
`_process_control_loop` is the "brain". Every 200 ms it drains the IPC queue,
decides whether to preload models, launch inference, or dispatch safety checks,
and monitors child process health.

## The three process types

1. **Main process**: runs the asyncio event loop and all the sub-managers. Never
   touches a GPU directly.

2. **Inference process(es)** (always process `1` and up): The number is
   configurable via `queue_size` / `max_threads`. Receives `PREPARE_AUX_MODELS`
   for pending LoRA resolution and `START_INFERENCE` control messages with full job payloads, runs hordelib (Stable Diffusion
   pipeline), streams back heartbeats and progress, and finally sends a
   `HordeInferenceResultMessage` with base64-encoded images. When alchemy is
   enabled, these same processes also handle **graph** alchemy forms (upscalers,
   face-fixers, background removal) via `START_ALCHEMY`.

3. **Safety process** (always process `0`): receives `EVALUATE_SAFETY` messages
   containing completed inference results, runs the NSFW/CSAM classifier, and
   sends back per-image safety evaluations (NSFW/CSAM flags and optional
   replacement images). The dispatcher derives the job's final
   `GENERATION_STATE` from those evaluations. It also runs **CLIP/caption**
   alchemy forms (interrogation, NSFW classification, captioning) via
   `START_ALCHEMY`.

A fourth, **download process** runs *outside* the inference/safety process map: it
owns a hordelib model manager (without a ComfyUI init) and fetches model weights
in the background while the worker serves whatever is already on disk. Because it
serves no jobs, the hung-process logic must never sweep it up. See
[Model Downloads and Availability](model_downloads.md).

## IPC: pipes and a queue

Parent → child communication uses per-process **multiprocessing `Pipe`s**. Child
→ parent communication uses a single shared **`multiprocessing.Queue`**.

Sends from the parent are **optimistic**: the parent updates `ProcessMap` /
`HordeModelMap` immediately after a successful send, before the child confirms.
The child's eventual state-change messages confirm or correct the parent's
bookkeeping. Messages carry a `process_launch_identifier` so messages from a
replaced (killed and restarted) process incarnation are silently discarded.
This allows the parent to react immediately to events without waiting for round-trip
confirmation, while still ensuring the child is the source of truth for its own state.

## Metrics and observability

Two layers of telemetry run alongside the pipeline:

- **Logfire** is the human-facing observability mirror (spans, histograms,
  gauges). Child processes forward a `trace_context` (W3C traceparent) on control
  messages so per-job spans correlate across the process boundary. The hordelib
  library's logfire init is suppressed in worker children via
  `HORDELIB_EXTERNAL_LOGFIRE=1` so it does not clobber the worker's own setup.
- **In-process run metrics** give the benchmark controller (and the e2e harness)
  the same numbers programmatically, without needing an OTel backend. Children
  emit `HordeJobMetricsMessage` / `HordeDownloadMetricsMessage` (sourced from
  hordelib's metrics collector: model-load phase timings, sampling it/s, VRAM/RAM
  high-water marks, ad-hoc download bandwidth); the main process folds these,
  per-job stage latencies (from the `JobTracker` finalize observer), disk
  free-space samples (`DiskSpaceMonitor`), and process-crash events into
  [`WorkerRunMetrics`][horde_worker_regen.process_management.resources.run_metrics.WorkerRunMetrics],
  exposed as a `RunMetricsSnapshot`. This is the foundation the progressive
  benchmark (`horde_worker_regen/benchmark/`) reads.

## The optional frontend

Everything above is headless. An optional Textual **dashboard** (`horde-worker`)
launches the worker as a child process and renders its live state over a
structured supervisor channel; the same app runs in a terminal or, in served
mode, in a browser. A small amount of **durable state** (last benchmark,
last-known-good settings, owned PIDs, the action ledger) persists between runs in
a `.horde_worker_regen/` working-directory folder. Neither is required to run the
worker. See [Frontend and Durable State](frontend_and_state.md).

## Dry-run mode

When `dry_run_skip_api` is active,
[`CannedJobSource`][horde_worker_regen.process_management.simulation._canned_scenarios.CannedJobSource]
substitutes synthetic jobs instead of calling the API, and
`fake_worker_processes` substitutes fake download, inference, and safety child
processes instead of running real model downloads, inference, or safety
evaluation. Everything else (the job tracker, the scheduler, the message
dispatcher) runs identically. The e2e harness can also inject synthetic system
resources (RAM, card count, per-card VRAM, backend kind, and process-overhead
estimates) and script the fake download process's initial model availability,
which lets canary simulations exercise representative volunteer-host topologies,
cold-start model availability, and background-download queue pressure without
touching the network or a GPU.

## Where the code lives

Almost all of the orchestration lives in
`horde_worker_regen/process_management/`. The entry point is `run_worker.py`;
the main-process orchestrator is `HordeWorkerProcessManager` in
`process_manager.py`.

The large process-management modules are grouped into responsibility-based
subpackages (`jobs/`, `scheduling/`, `lifecycle/`, `workers/`, `ipc/`, and so
on). Top-level files with the old flat names are compatibility shims so existing
imports continue to resolve while new code can use the canonical subpackage
paths.

For a complete file-to-responsibility map, including the pop pipeline, scheduler,
IPC, and process lifecycle, see the **[Codebase Map](../reference/codebase-map.md)**.

## See also

- [Codebase Map](../reference/codebase-map.md): file→responsibility map and entry points
- [Job Lifecycle](job_lifecycle.md): traces a job through every stage
- [Job State Machine](job_state_machine.md): how
  [`JobTracker`][horde_worker_regen.process_management.jobs.job_tracker.JobTracker]
  enforces stage transitions
- [IPC and Messaging](ipc_and_messaging.md): the pipe/queue model and message
  types
- [Process Lifecycle](process_lifecycle.md): starting, monitoring, and
  replacing child processes
- [Bridge Configuration](bridge_config.md): config file fields and hot-reload
- [Performance and Backpressure](performance_and_backpressure.md): throttling
  and scheduling
- [Shutdown and Faults](shutdown_and_faults.md): graceful vs. abort shutdown
- [Resilience and Recovery](resilience_and_recovery.md): bounded retry, SOS
  escalation, and orphan cleanup
- [Model Downloads and Availability](model_downloads.md): the download process
  and on-disk availability tracking
- [Frontend and Durable State](frontend_and_state.md): the dashboard, supervisor
  channel, and persisted state
