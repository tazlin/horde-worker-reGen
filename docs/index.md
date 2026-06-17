# Horde Worker reGen documentation

The AI Horde worker pulls image-generation jobs from the [AI Horde](https://aihorde.net/) API, runs
the inference in child processes, checks the results for safety, and submits them back. It can also
serve **alchemy** jobs (post-processing, interrogation, captioning) on the same processes.

These docs are organised by what you are trying to do.

## Using the worker

New here? Start with the tutorial, then dip into the how-to guides and reference as needed.

- **[Getting started](tutorials/getting-started.md)**: a guided walkthrough from install to your first
  kudos. No command line required.
- **How-to guides** (task-focused recipes):
    - [install](how-to/install.md)
    - [configure for your GPU](how-to/configure-for-your-gpu.md)
    - [use the dashboard](how-to/use-the-dashboard.md)
    - [run headless](how-to/run-headless.md)
    - [run on AMD ROCm](how-to/run-on-amd-rocm.md)
    - [run multiple GPUs](how-to/run-multiple-gpus.md)
    - [add custom models](how-to/add-custom-models.md)
    - [run in Docker](how-to/run-in-docker.md)
    - [update](how-to/update-the-worker.md)
    - [troubleshoot](how-to/troubleshoot.md)
- **Reference** (look something up):
  [command line](reference/cli.md),
  [logs](reference/logs.md),
  [codebase map](reference/codebase-map.md).

## Understanding the worker

Want to know how it works under the hood, or contribute a change? The **explanation** pages describe
the design. Read them in this order:

1. **[Architecture](explanation/architecture.md)**: the shared-state pattern, the process model, and
   the asyncio loop. Start here.
2. **[Job lifecycle](explanation/job_lifecycle.md)**: traces a job from pop to submit through every
   subsystem.
3. **[Job state machine](explanation/job_state_machine.md)**: how the unified `JobTracker` enforces
   stage transitions and invariants.
4. **[Bridge configuration](explanation/bridge_config.md)**: what every `bridgeData.yaml` field
   controls and how config flows at runtime.
5. **[IPC and messaging](explanation/ipc_and_messaging.md)**: the pipe/queue model, message types, and
   the optimistic-send pattern.
6. **[Process lifecycle](explanation/process_lifecycle.md)**: starting, monitoring, replacing, and
   killing child processes.
7. **[Performance and backpressure](explanation/performance_and_backpressure.md)**: pop throttling,
   model stickiness, megapixelstep backpressure, and the LRU eviction policy.
8. **[Compute backends](explanation/compute_backends.md)**: the backend-agnostic device/VRAM
   abstraction, optional feature extras, and what limits non-NVIDIA support.
9. **[Shutdown and faults](explanation/shutdown_and_faults.md)**: graceful versus abort shutdown,
   signal handling, and fault propagation.
10. **[Resilience and recovery](explanation/resilience_and_recovery.md)**: bounded/degraded job
    retry, crash-loop quarantine, the save-our-ship escalation, and orphan cleanup.
11. **[Model downloads and availability](explanation/model_downloads.md)**: the background download
    process, on-disk availability tracking, and how popping stays aligned with it.
12. **[Frontend and durable state](explanation/frontend_and_state.md)**: the dashboard/TUI, the
    supervisor channel, and the state that persists between runs.
13. **[Telemetry](explanation/telemetry.md)**: the Logfire/OpenTelemetry layer and in-process run
    metrics.

The **[codebase map](reference/codebase-map.md)** is a file-to-responsibility quick reference, and the
auto-generated API reference for every module lives under
**[Code reference](horde_worker_regen/)**.

## For contributors

- **[Code issues for remediation](explanation/code_issues_for_remediation.md)**: known rough edges and
  potential bugs identified during documentation review, intended for developer triage.
- **[Contributing](https://github.com/Haidra-Org/horde-worker-reGen/blob/main/CONTRIBUTING.md)**:
  development setup and guidelines.
