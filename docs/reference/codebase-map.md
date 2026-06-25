# Codebase Map

A navigation aid for finding your way around the source. For the concepts behind
these files, read [Architecture](../explanation/architecture.md) first.

> **TL;DR**: Execution starts in `run_worker.py`. Almost all of the interesting
> logic lives in `horde_worker_regen/process_management/`, coordinated by
> `HordeWorkerProcessManager` (`process_manager.py`). The main process never
> touches a GPU; it drives child **inference** and **safety** processes over
> IPC.

## Where do I look for...?

| If you're working on...                | Start in...                                                        |
| -------------------------------------- | ------------------------------------------------------------------ |
| Top-level orchestration                | `process_management/process_manager.py` (`HordeWorkerProcessManager`) |
| Process startup                        | `process_management/main_entry_point.py`, `worker_entry_points.py` |
| Job popping / pop gates                | `process_management/jobs/job_popper.py` (`JobPopper`)              |
| Job stages, faults, invariants         | `process_management/jobs/job_tracker.py` (`JobTracker`)            |
| Scheduling inference & model preloads  | `process_management/scheduling/inference_scheduler.py` (`InferenceScheduler`) |
| Pop-rate & megapixelstep throttling    | `process_management/scheduling/pop_throttler.py` (`PopThrottler`)  |
| Safety-check dispatch                  | `process_management/workers/safety_orchestrator.py` (`SafetyOrchestrator`) |
| Alchemy pop / dispatch / submit        | `process_management/jobs/alchemy_popper.py` (`AlchemyCoordinator`) |
| Parsing child-to-parent messages       | `process_management/ipc/message_dispatcher.py` (`MessageDispatcher`) |
| IPC message and enum definitions       | `process_management/ipc/messages.py`                               |
| Starting / replacing child processes   | `process_management/lifecycle/process_lifecycle.py` (`ProcessLifecycleManager`) |
| Live process state                     | `process_management/lifecycle/process_map.py`, `process_info.py`   |
| Result upload & submission             | `process_management/jobs/job_submitter.py` (`JobSubmitter`)        |
| Config fields & hot-reload             | `bridge_data/`, `process_management/config/runtime_config.py`      |
| Dashboard state channel                | `process_management/ipc/supervisor_channel.py`                     |
| What runs inside a child process       | `process_management/workers/inference_process.py`, `safety_process.py`, `download_process.py` |
| Model availability and downloads       | `process_management/models/`, `process_management/workers/download_process.py` |
| VRAM/RAM budgeting and metrics         | `process_management/resources/`                                    |
| Multi-GPU routing                      | `process_management/gpu/`                                          |
| Dry-run / test doubles                 | `process_management/testing/`                                      |
| Progressive benchmark / ramp CLI       | `benchmark/` (`horde-benchmark`)                                   |

## Program entry points

The startup path, in order:

| Step                  | Location                                                                                     | Role                                                                                                                                 |
| --------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Process launch     | `run_worker.py` (`init` -> `main`)                                                           | Sets the `spawn` start method, clears a stale `.abort`, parses CLI args, validates config, ensures the model reference DB is present |
| 2. Driver bootstrap   | `process_management/main_entry_point.py` (`start_working`)                                   | Builds and runs the main-process orchestrator                                                                                        |
| 3. Main orchestrator  | `process_management/process_manager.py` (`HordeWorkerProcessManager._main_loop`)             | Owns the asyncio event loop and the long-lived tasks                                                                                 |
| 4. Child entry points | `process_management/worker_entry_points.py` (`ProcessEntryPoints`)                           | The callables each spawned child process runs                                                                                        |
| 5. Child workloads    | `process_management/workers/inference_process.py`, `workers/safety_process.py`               | The actual GPU/classifier work, subclassing `HordeProcess` (`lifecycle/horde_process.py`)                                            |

## Core orchestration: `process_management/`

The main process is a set of single-responsibility sub-managers that all share
state by reference (see
[Architecture](../explanation/architecture.md#the-shared-state-pattern)).
`process_manager.py`, `main_entry_point.py`, and `worker_entry_points.py` stay
at the top level because they are the facade and entry points. The other
process-management modules live in canonical subpackages; import those grouped
module paths directly.

| Subpackage | Responsibility |
| ---------- | -------------- |
| `lifecycle/` | Parent-side process machinery: spawn, supervise, reap, replace, shutdown, recovery, crash capture, process maps, and process metadata. |
| `workers/` | Child-process bodies and worker-side orchestration: inference, safety, safety dispatch, and background downloads. |
| `scheduling/` | What to run, when, where, and why: inference scheduling, pop throttling, model affinity, performance model, and workload flow routing. |
| `jobs/` | The unit of work: pop, submit, track, classify failures, alchemy coordination, job data models, and source-image downloads. |
| `models/` | On-disk model state and feature readiness: desired state, availability, metadata, load map, cache, LoRA guards/backoff, and download scheduling. |
| `resources/` | Runtime resource accounting: VRAM/RAM budgets, device info, system memory, duty-cycle summaries, and run metrics. |
| `gpu/` | Multi-GPU routing primitives: card runtime state, eligibility checks, and advertised pop-shaping capabilities. |
| `ipc/` | Message types, channels, dispatch, supervisor protocol, action ledger, and API sessions. |
| `config/` | Live runtime config, mutable worker state, and worker identity preflight. |
| `testing/` | Dry-run and test-double modules that are imported by both tests and the harness. |
| `_internal/` | Cross-cutting internal helpers that do not belong to a domain package. |

## Supporting top-level packages

| Path                          | Responsibility                                                  |
| ----------------------------- | --------------------------------------------------------------- |
| `bridge_data/`                | Loading and validating `bridgeData.yaml` into `reGenBridgeData` |
| `models/`                     | Worker-side model reference and metadata helpers                |
| `reporting/`                  | Status reporting and run statistics                             |
| `utils/`                      | Image, job, and kudos helpers (e.g. `job_queue_analyzer.py`, `disk_monitor.py`) |
| `telemetry.py`, `telemetry_spans.py` | Logfire/OTel setup, cross-process trace context, and span/histogram definitions |
| `locale_info/`, `localize.py` | Localization                                                    |
| `benchmark/`                  | Progressive worker benchmark: tier/feature matrix in `ladder.py` (typed by `enums.py`), `sizing.py` (VRAM-derived post-processing resolution), level runner, controller, criteria, `report.py` (capabilities + conservative recommendation + per-setting provenance), `history.py` (enumerate/compare past runs), and the `horde-benchmark` CLI |
| `amd_go_fast/`                | AMD/ROCm-specific optimizations                                 |
| `capabilities/`               | Placeholder for future optional "capability" processes (heavy features split out of the base worker; see the README) |
| `consts.py`                   | Shared constants and filenames                                  |
| `load_env_vars.py`            | Environment-variable configuration path                         |
| `download_models.py`          | Standalone model-download CLI                                   |
| `harness.py`                  | E2E harness: runs the real orchestration against canned scenarios with selectable real/fake API and processes |

## Conceptual docs

- [Architecture](../explanation/architecture.md)
- [Job Lifecycle](../explanation/job_lifecycle.md)
- [Job State Machine](../explanation/job_state_machine.md)
- [Bridge Configuration](../explanation/bridge_config.md)
- [IPC and Messaging](../explanation/ipc_and_messaging.md)
- [Process Lifecycle](../explanation/process_lifecycle.md)
- [Performance and Backpressure](../explanation/performance_and_backpressure.md)
- [Shutdown and Faults](../explanation/shutdown_and_faults.md)
- [Resilience and Recovery](../explanation/resilience_and_recovery.md)
- [Model Downloads and Availability](../explanation/model_downloads.md)
- [Frontend and Durable State](../explanation/frontend_and_state.md)

The full auto-generated API reference lives under
[Horde Worker Regen Code Reference](../horde_worker_regen/).
