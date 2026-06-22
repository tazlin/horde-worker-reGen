# Codebase Map

A navigation aid for finding your way around the source. For the concepts behind
these files, read [Architecture](../explanation/architecture.md) first.

> **TL;DR**: Execution starts in `run_worker.py`. Almost all of the interesting
> logic lives in `horde_worker_regen/process_management/`, coordinated by
> `HordeWorkerProcessManager` (`process_manager.py`). The main process never
> touches a GPU; it drives child **inference** and **safety** processes over
> IPC.

## Where do I look for…?

| If you're working on…                 | Start in…                                             |
| ------------------------------------- | ----------------------------------------------------- |
| Job popping / pop gates               | `job_popper.py` (`JobPopper`)                         |
| Pop-rate & megapixelstep throttling   | `pop_throttler.py` (`PopThrottler`)                   |
| Job stages, faults, invariants        | `job_tracker.py` (`JobTracker`)                       |
| Scheduling inference & model preloads | `inference_scheduler.py` (`InferenceScheduler`)       |
| Safety-check dispatch                 | `safety_orchestrator.py` (`SafetyOrchestrator`)       |
| Alchemy pop / dispatch / submit       | `alchemy_popper.py` (`AlchemyCoordinator`)            |
| Parsing child→parent messages         | `message_dispatcher.py` (`MessageDispatcher`)         |
| Starting / replacing child processes  | `process_lifecycle.py` (`ProcessLifecycleManager`)    |
| Result upload & submission            | `job_submitter.py` (`JobSubmitter`)                   |
| Run metrics / disk / telemetry        | `run_metrics.py`, `utils/disk_monitor.py`, `telemetry.py`, `telemetry_spans.py` |
| Shutdown / abort / signals            | `shutdown_manager.py` (`ShutdownManager`)             |
| Bounded/degraded retry & SOS recovery | `job_tracker.py`, `failure_classification.py`, `recovery_supervisor.py` |
| Crash audit & orphan reaping          | `action_ledger.py`, `owned_process_registry.py`      |
| Job-cost / "slow job" scoring         | `performance_model.py`; model→process pinning: `model_affinity.py` |
| Background model downloads            | `download_process.py`, `model_availability.py`, `model_download_plan.py` |
| Chaos / fault-injection testing       | `fault_injection.py`, `fake_worker_processes.py`     |
| Config fields & hot-reload            | `bridge_data/`, `runtime_config.py` (`RuntimeConfig`) |
| IPC message & enum definitions        | `messages.py`                                         |
| What runs inside a child process      | `inference_process.py`, `safety_process.py`           |
| The dashboard / TUI                   | `tui/` (`horde-worker`); state channel: `supervisor_channel.py` |
| Durable cross-run state               | `app_state.py` (`.horde_worker_regen/state.json`)    |
| Worker-name preflight                 | `worker_identity.py`                                  |
| Progressive benchmark / ramp CLI      | `benchmark/` (`horde-benchmark`)                      |

## Program entry points

The startup path, in order:

| Step                  | Location                                                                                     | Role                                                                                                                                 |
| --------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Process launch     | `run_worker.py` (`init` → `main`)                                                            | Sets the `spawn` start method, clears a stale `.abort`, parses CLI args, validates config, ensures the model reference DB is present |
| 2. Driver bootstrap   | `main_entry_point.py` (`start_working`)                                                      | Builds and runs the main-process orchestrator                                                                                        |
| 3. Main orchestrator  | `process_manager.py` (`HordeWorkerProcessManager._main_loop`)                                | Owns the asyncio event loop and the long-lived tasks                                                                                 |
| 4. Child entry points | `worker_entry_points.py` (`ProcessEntryPoints`)                                              | The callables each spawned child process runs                                                                                        |
| 5. Child workloads    | `inference_process.py` (`HordeInferenceProcess`), `safety_process.py` (`HordeSafetyProcess`) | The actual GPU/classifier work, subclassing `HordeProcess` (`horde_process.py`)                                                      |

## Core orchestration: `process_management/`

The main process is a set of single-responsibility sub-managers that all share
state by reference (see
[Architecture](../explanation/architecture.md#the-shared-state-pattern)).

### Coordination

| File                    | Primary type                | Responsibility                                                                                        |
| ----------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------- |
| `process_manager.py`    | `HordeWorkerProcessManager` | Top-level orchestrator; the asyncio loop, the long-lived tasks, signal registration, performance-mode setup, run-metrics/disk-monitor wiring |
| `message_dispatcher.py` | `MessageDispatcher`         | Drains the child→parent queue, applies results, discards stale messages, logs deadlock diagnostics    |

### Job pipeline

| File                         | Primary type            | Responsibility                                                                                         |
| ---------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------ |
| `job_popper.py`              | `JobPopper`             | The "pop gauntlet" of gates; consecutive-failure backoff                                               |
| `pop_throttler.py`           | `PopThrottler`          | Pop-rate frequencies and megapixelstep wait timing (`CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS` lives here) |
| `inference_scheduler.py`     | `InferenceScheduler`    | Decides which job/model to preload and launch; the line-skip cache, size/progress-aware concurrent-overlap gating, idle-thread diversity dispatch, and the VRAM-aware co-resident-context teardown |
| `safety_orchestrator.py`     | `SafetyOrchestrator`    | Dispatches completed images to the safety process                                                      |
| `alchemy_popper.py`          | `AlchemyCoordinator`    | The separate alchemy pop/dispatch/submit loop and its concurrency gating (`AlchemyHeadroomEstimator`)  |
| `job_submitter.py`           | `JobSubmitter`          | Uploads images to R2 and submits results to the API                                                    |
| `source_image_downloader.py` | `SourceImageDownloader` | Fetches source images for img2img / remix jobs                                                         |

### Process & IPC management

| File                   | Primary type              | Responsibility                                                                                         |
| ---------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------ |
| `process_lifecycle.py` | `ProcessLifecycleManager` | Starts, stops, replaces, and hung-checks child processes; owns the shared semaphores/locks             |
| `process_map.py`       | `ProcessMap`              | The live `HordeProcessInfo` per process; validates state transitions                                   |
| `process_info.py`      | `HordeProcessInfo`        | Per-process bookkeeping record                                                                         |
| `process_temperature.py` | `ProcessTemperature`    | Pure classifier turning a slot's raw state + resident/pending models into a "temperature" (hot/next/warm/priming/cold/down) for the status line and TUI |
| `messages.py`          | message classes + enums   | All IPC message types, `HordeProcessState`, `ModelLoadState`, `HordeControlFlag`, `HordeHeartbeatType` |
| `shutdown_manager.py`  | `ShutdownManager`         | Graceful shutdown, abort, and the signal handler                                                       |

### State, config, and models

| File                 | Primary type                                         | Responsibility                                                                       |
| -------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `job_tracker.py`     | `JobTracker` (+ `JobStage`, `TrackedJob`)            | The single source of truth for job stages, faults, counters, and megapixelstep state |
| `job_models.py`      | `HordeJobInfo`, `PendingJob`, `NextJobAndProcess`, … | Job-related data models passed between sub-managers                                  |
| `worker_state.py`    | `WorkerState`                                        | Cross-cutting flags: shutdown, pop times, consecutive failures, kudos                |
| `runtime_config.py`  | `RuntimeConfig`                                      | Live, hot-reloadable `reGenBridgeData` snapshot                                      |
| `horde_model_map.py` | `HordeModelMap`                                      | Model name → load state + owning process                                             |
| `model_metadata.py`  | `ModelMetadata`                                      | Stable-diffusion model reference lookups                                             |
| `api_sessions.py`    | `ApiSessions`                                        | aiohttp and horde-sdk client sessions                                                |
| `lru_cache.py`       | `LRUCache`                                           | Recency ordering used by model eviction                                              |
| `run_metrics.py`     | `WorkerRunMetrics` (+ `RunMetricsSnapshot`)          | Aggregates per-job latencies, child phase/download metrics, and crashes for the benchmark/e2e harness |
| `system_memory.py`   | `SystemMemorySummary`                                | Samples total/available system RAM and per-role RSS (orchestrator/inference/safety/download) via psutil, for the supervisor snapshot and TUI |

### Resilience and recovery

| File                        | Primary type           | Responsibility                                                                                  |
| --------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------- |
| `failure_classification.py` | (functions)            | Classify a faulted result as a resource (OOM) failure vs other, to choose the retry strategy    |
| `recovery_supervisor.py`    | `RecoverySupervisor`   | Pure save-our-ship escalation policy: soft-reset (limp-by) vs give-up timing                     |
| `action_ledger.py`          | `ActionLedger`         | Append-only audit ring (+ optional JSONL) of lifecycle actions for crash diagnosis              |
| `owned_process_registry.py` | `OwnedProcessRegistry` | Durable record of owned OS pids so orphans can be reaped after a hard parent death              |
| `performance_model.py`      | `PerformanceModel`     | Expected sampling it/s per job signature (benchmark-seeded + self-calibrating) to grade "slow"  |
| `model_affinity.py`         | (functions)            | Compute which processes hold the last copy of a wanted model, to protect them from displacement |
| `fault_injection.py`        | `FaultProfile`         | Typed misbehaviour profiles for the fake processes (hang/crash/slow/OOM/malformed)              |

### Downloads and model availability

| File                      | Primary type        | Responsibility                                                                       |
| ------------------------- | ------------------- | ------------------------------------------------------------------------------------ |
| `download_process.py`     | (process entry)     | Dedicated background weight-download process (outside the process map); phase/pause/rate reporting |
| `model_availability.py`   | `ModelAvailability` | On-disk model set + live download status; gates the popper's advertised models       |
| `model_download_plan.py`  | (top-level)         | Torch-free planning: which configured models are on disk, disk usage, and fit        |
| `reference_helper.py`     | (top-level)         | Offline (never-download) reference manager for subprocesses; parent owns downloading |

### Frontend and durable state

| Path                    | Primary type            | Responsibility                                                                  |
| ----------------------- | ----------------------- | ------------------------------------------------------------------------------- |
| `tui/`                  | Textual app (`horde-worker`) | The dashboard: launches/supervises the worker, renders live state, wizard, config editor |
| `supervisor_channel.py` | `WorkerStateSnapshot`, `SupervisorControlMessage` | Structured state/control protocol between a frontend and the worker (`SUPERVISOR_PROTOCOL_VERSION`, currently 6; snapshot carries `SystemMemorySnapshot`) |
| `app_state.py`          | (top-level)             | Durable cross-run state (`.horde_worker_regen/state.json`)                       |
| `worker_identity.py`    | (functions)             | Startup fail-fast checks that configured worker names are valid and owned        |

### Dry-run / test doubles

`fake_worker_processes.py` (`FakeInferenceProcess`, `FakeSafetyProcess`),
`_canned_scenarios.py` (`CannedJobSource`, `CannedAlchemySource`),
`_dummy_images.py`, and `_dummy_jobs.py` substitute synthetic jobs and results so
the pipeline can run without the network or a GPU (see
[Architecture](../explanation/architecture.md#dry-run-mode)).

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
