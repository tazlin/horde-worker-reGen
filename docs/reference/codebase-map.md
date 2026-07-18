# Codebase Map

A navigation aid for finding your way around the source. For the concepts behind
these files, read [Architecture](../explanation/architecture.md) first.

> **TL;DR**: Execution starts in `run_worker.py`. Almost all of the interesting
> logic lives in `horde_worker_regen/process_management/`, coordinated by
> `HordeWorkerProcessManager` (`process_manager.py`). The main process never
> touches a GPU; it drives child **inference**, **safety**, **post-processing**,
> and optional disaggregated pipeline lane processes over IPC.

## Where do I look for...?

| If you're working on...                | Start in...                                                        |
| -------------------------------------- | ------------------------------------------------------------------ |
| Top-level orchestration                | `process_management/process_manager.py` (`HordeWorkerProcessManager`) |
| Process startup                        | `process_management/main_entry_point.py`, `worker_entry_points.py` |
| Job popping / pop gates                | `process_management/jobs/job_popper.py` (`JobPopper`)              |
| Job stages, faults, invariants         | `process_management/jobs/job_tracker.py` (`JobTracker`)            |
| Scheduling inference & model preloads  | `process_management/scheduling/inference_scheduler.py` (`InferenceScheduler`) |
| Pop-rate & megapixelstep throttling    | `process_management/scheduling/pop_throttler.py` (`PopThrottler`)  |
| Pop/scheduling hold visibility         | `process_management/scheduling/pop_governor_registry.py`           |
| Safety-check dispatch                  | `process_management/workers/safety_orchestrator.py` (`SafetyOrchestrator`) |
| Post-processing lane dispatch          | `process_management/workers/post_process_orchestrator.py` (`PostProcessOrchestrator`) |
| Disaggregated pipeline stages          | `process_management/workers/disaggregation_orchestrator.py`, `component_lane_process.py`, `inference_process.py`, `vae_lane_process.py` |
| Alchemy pop / dispatch / submit        | `process_management/jobs/alchemy_popper.py` (`AlchemyCoordinator`) |
| Parsing child-to-parent messages       | `process_management/ipc/message_dispatcher.py` (`MessageDispatcher`) |
| IPC message and enum definitions       | `process_management/ipc/messages.py`                               |
| Starting / replacing child processes   | `process_management/lifecycle/process_lifecycle.py` (`ProcessLifecycleManager`) |
| Live process state                     | `process_management/lifecycle/process_map.py`, `process_info.py`   |
| Result upload & submission             | `process_management/jobs/job_submitter.py` (`JobSubmitter`)        |
| Config fields & hot-reload             | `bridge_data/`, `process_management/config/runtime_config.py`, `process_management/config/bridge_data_reloader.py` |
| Dashboard state channel                | `process_management/ipc/supervisor_channel.py`                     |
| What runs inside a child process       | `process_management/workers/inference_process.py`, `safety_process.py`, `post_process_process.py`, `download_process.py` |
| Model availability and downloads       | `process_management/models/` (`ModelDownloadCoordinator`), `process_management/workers/download_process.py` |
| VRAM/RAM budgeting and metrics         | `process_management/resources/`, especially `vram_arbiter.py` and `resource_budget.py` |
| Host-resource governance               | `process_management/scheduling/governance/` (`ResourceGovernor`)   |
| Multi-GPU routing                      | `process_management/gpu/`                                          |
| File descriptor limit preflight        | `process_management/fd_limits.py`                                  |
| Dry-run / test doubles                 | `process_management/simulation/`                                   |
| Progressive benchmark / ramp CLI       | `benchmark/` (`horde-benchmark`)                                   |

## Program entry points

The startup path, in order:

| Step                  | Location                                                                                     | Role                                                                                                                                 |
| --------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Process launch     | `run_worker.py` (`init` -> `main`)                                                           | Sets the `spawn` start method, clears a stale `.abort`, parses CLI args, validates config, ensures the model reference DB is present |
| 2. Driver bootstrap   | `process_management/main_entry_point.py` (`start_working`)                                   | Builds and runs the main-process orchestrator                                                                                        |
| 3. Main orchestrator  | `process_management/process_manager.py` (`HordeWorkerProcessManager._main_loop`)             | Owns the asyncio event loop and the long-lived tasks                                                                                 |
| 4. Child entry points | `process_management/worker_entry_points.py` (`ProcessEntryPoints`)                           | The callables each spawned child process runs                                                                                        |
| 5. Child workloads    | `process_management/workers/inference_process.py`, `workers/safety_process.py`, `workers/post_process_process.py`, `workers/component_lane_process.py`, `workers/vae_lane_process.py` | The actual GPU/classifier/lane work, subclassing `HordeProcess` (`lifecycle/horde_process.py`) |

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
| `lifecycle/` | Parent-side process machinery: spawn, supervise, reap, replace, shutdown, worker-level recovery, crash capture, process maps, and process metadata. |
| `workers/` | Child-process bodies and worker-side orchestration: inference, safety, safety dispatch, post-processing, disaggregated text/VAE lanes, and background downloads. |
| `scheduling/` | What to run, when, where, and why: inference scheduling, resource governance, pop throttling, model affinity, performance model, pop-governor visibility, and workload flow routing. |
| `jobs/` | The unit of work: pop, submit, track, classify failures, alchemy coordination, job data models, and source-image downloads. |
| `models/` | On-disk model state and feature readiness: desired state, availability, metadata, load map, the per-process RAM component-residency map, cache, LoRA guards/backoff, download coordination, and download scheduling. |
| `resources/` | Runtime resource accounting: VRAM/RAM budgets, VRAM arbitration, attribution, device info, system memory, duty-cycle summaries, and run metrics. |
| `gpu/` | Multi-GPU routing primitives: card runtime state, eligibility checks, and advertised pop-shaping capabilities. |
| `ipc/` | Message types, channels, dispatch, supervisor protocol, action ledger, and API sessions. |
| `config/` | Live runtime config, bridge-data reload orchestration, mutable worker state, and worker identity preflight. |
| `simulation/` | Dry-run and test-double modules that are imported by both tests and the harness. |
| `_internal/` | Cross-cutting internal helpers that do not belong to a domain package. |

## Supporting top-level packages

| Path                          | Responsibility                                                  |
| ----------------------------- | --------------------------------------------------------------- |
| `analysis/`                   | Log/session/support-bundle analysis and triage tools             |
| `bridge_data/`                | Loading and validating `bridgeData.yaml` into `reGenBridgeData` |
| `bridge_data/disagg_model_selection.py` | Pure ranking for the `disagg_optimized N` model rule: shared-VAE cluster (via `derive_canonical_registry` over merged record/sidecar hashes) then popularity |
| `tui/`                        | Textual dashboard, web dashboard, config form, and worker host/launcher |
| `models/`                     | Worker-side model reference and metadata helpers                |
| `reporting/`                  | Status reporting and run statistics                             |
| `utils/`                      | Image, job, system, accelerator-probe, quota, and diagnostics helpers |
| `telemetry.py`, `telemetry_spans.py` | Logfire/OTel setup, cross-process trace context, and span/histogram definitions |
| `locale_info/`, `localize.py` | Localization                                                    |
| `benchmark/`                  | Progressive worker benchmark: tier/feature matrix in `ladder.py` (typed by `enums.py`), `sizing.py` (VRAM-derived post-processing resolution), level runner, controller, criteria, `report.py` (capabilities + conservative recommendation + per-setting provenance), `history.py` (enumerate/compare past runs), the sustained-load `soak.py` mixes, and the `horde-benchmark` CLI |
| `benchmark/disagg_mixes.py`   | Named disagg-optimization payload mixes (`DisaggGateMix`): component-churn and VAE-cluster scenarios for the gate driver |
| `benchmark/gate_driver.py`    | Disagg A/B measurement gate: runs a mix through the harness in ABBA order, scores kudos/hr, and derives per-stage reload/latency mechanism metrics (`python -m horde_worker_regen.benchmark.gate_driver`) |
| `amd_go_fast/`                | AMD/ROCm-specific optimizations                                 |
| `capabilities/`               | Placeholder for future optional "capability" processes (heavy features split out of the base worker; see the README) |
| `app_state.py`                | Durable dashboard/worker state path helpers                     |
| `compute_mode.py`, `server_capabilities.py` | Torch-free backend intent and advertised capability helpers |
| `consts.py`                   | Shared constants and filenames                                  |
| `load_env_vars.py`            | Environment-variable configuration path                         |
| `log_file_registry.py`, `logging_purge.py` | Declared worker log-file families and startup retention sweep |
| `model_download_core.py`, `model_download_plan.py` | Shared download planning/execution helpers used by the CLI and process manager (`primary_checkpoint_path_for` resolves a record's primary checkpoint file for the scheduler's component charge and config-load sidecar lookup) |
| `reference_helper.py`         | Parent/child-safe access to model-reference data                 |
| `runtime_version.py`, `update_check.py`, `version_meta.py` | Runtime version metadata and update checks |
| `torch_gpu_preflight.py`      | Container startup preflight for torch/GPU architecture mismatches |
| `download_models.py`          | Standalone model-download CLI                                   |
| `harness.py`                  | E2E harness: runs the real orchestration against canned scenarios with selectable real/fake API and processes |

## Conceptual docs

- [Architecture](../explanation/architecture.md)
- [Compute Backends](../explanation/compute_backends.md)
- [Job Lifecycle](../explanation/job_lifecycle.md)
- [Job State Machine](../explanation/job_state_machine.md)
- [Bridge Configuration](../explanation/bridge_config.md)
- [IPC and Messaging](../explanation/ipc_and_messaging.md)
- [Process Lifecycle](../explanation/process_lifecycle.md)
- [Process Lanes and Chaining](../explanation/process_lanes_and_chaining.md)
- [Performance and Backpressure](../explanation/performance_and_backpressure.md)
- [Resource Governance](../explanation/resource_governance.md)
- [VRAM Arbiter](../explanation/vram_arbiter.md)
- [Duty Cycle](../explanation/duty-cycle.md)
- [Log Diagnostics Contract](../explanation/log_diagnostics_contract.md)
- [Shutdown and Faults](../explanation/shutdown_and_faults.md)
- [Resilience and Recovery](../explanation/resilience_and_recovery.md)
- [Model Downloads and Availability](../explanation/model_downloads.md)
- [Frontend and Durable State](../explanation/frontend_and_state.md)
- [Telemetry](../explanation/telemetry.md)

The full auto-generated API reference lives under
[Horde Worker Regen Code Reference](../horde_worker_regen/).
