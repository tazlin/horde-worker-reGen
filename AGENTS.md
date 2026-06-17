# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository. This file is a high-level
map; the [`docs/`](docs/index.md) tree is the source of truth for depth and is kept current. Prefer
linking a reader to a doc page over duplicating it here.

## What this is and why it exists

**Horde Worker reGen** is the local GPU worker for the [AI Horde](https://aihorde.net/): a free,
decentralized network where people donate GPU time to generate AI images. The worker pulls jobs from
the AI Horde API, runs Stable Diffusion / Flux inference (via `hordelib`/ComfyUI), screens results
through an NSFW/CSAM safety classifier, uploads the images to R2, and submits the result back to the
API. Operators earn **kudos** for completed work. A worker can additionally opt into **alchemy** jobs
(upscaling, face-fixing, interrogation, captioning) on the same processes.

**Why multiprocess.** Inference is VRAM-heavy and stateful, and ComfyUI is not thread-safe, so each
GPU slot runs in its own OS process. The main process never touches the GPU; it orchestrates child
**inference** and **safety** processes (and a separate background **download** process) over IPC. This
buys crash isolation, model persistence across jobs, and parallelism (preload one model while another
samples; run safety in parallel with inference). See [Architecture](docs/explanation/architecture.md).

## Start here (docs)

- [Documentation home](docs/index.md): Diátaxis tree (tutorials / how-to / reference / explanation).
- [Architecture](docs/explanation/architecture.md): process model, shared-state pattern, asyncio loop.
- [Codebase map](docs/reference/codebase-map.md): file → responsibility quick reference.
- [Job lifecycle](docs/explanation/job_lifecycle.md) and
  [Job state machine](docs/explanation/job_state_machine.md): a job from pop to submit.
- [Resilience and recovery](docs/explanation/resilience_and_recovery.md),
  [Model downloads](docs/explanation/model_downloads.md), and
  [Frontend and durable state](docs/explanation/frontend_and_state.md): the newer subsystems.

## The map (most important files & classes)

Almost all orchestration lives in `horde_worker_regen/process_management/`. The main process is a set
of single-responsibility sub-managers that **share state by reference** (set once at construction,
never reassigned), coordinated by `HordeWorkerProcessManager`.

| Concern | File · primary type |
| ------- | ------------------- |
| Top-level orchestrator (asyncio loop, long-lived tasks, signals) | `process_manager.py` · `HordeWorkerProcessManager` |
| Single source of truth for job stages/faults/counters | `job_tracker.py` · `JobTracker` (`JobStage`, `TrackedJob`) |
| Pop "gauntlet" of gates + model selection | `job_popper.py` · `JobPopper` |
| Pop-rate / megapixelstep throttling | `pop_throttler.py` · `PopThrottler` |
| Decide which model/job to preload & launch | `inference_scheduler.py` · `InferenceScheduler` |
| Drain child→parent queue, apply results | `message_dispatcher.py` · `MessageDispatcher` |
| Dispatch completed images to safety | `safety_orchestrator.py` · `SafetyOrchestrator` |
| Upload to R2 + submit to API | `job_submitter.py` · `JobSubmitter` |
| Alchemy pop/dispatch/submit loop | `alchemy_popper.py` · `AlchemyCoordinator` |
| Start/stop/replace/hung-check child processes | `process_lifecycle.py` · `ProcessLifecycleManager` |
| Per-process live state + transition validation | `process_map.py` · `ProcessMap` (`process_info.py` · `HordeProcessInfo`) |
| All IPC message types + enums | `messages.py` |
| What runs inside a child | `inference_process.py` · `HordeInferenceProcess`, `safety_process.py` · `HordeSafetyProcess` (base: `horde_process.py`) |
| Shutdown / abort / signals | `shutdown_manager.py` · `ShutdownManager` |
| Bounded/degraded retry, SOS recovery | `job_tracker.py`, `failure_classification.py`, `recovery_supervisor.py` · `RecoverySupervisor` |
| Crash audit + orphan reaping | `action_ledger.py` · `ActionLedger`, `owned_process_registry.py` · `OwnedProcessRegistry` |
| "Slow job" scoring + model pinning | `performance_model.py` · `PerformanceModel`, `model_affinity.py` |
| Background weight downloads + availability | `download_process.py`, `model_availability.py` · `ModelAvailability`, `model_download_plan.py` |
| Live config (hot-reload) | `bridge_data/` (`reGenBridgeData`, `BridgeDataLoader`), `runtime_config.py` · `RuntimeConfig` |
| Dashboard / TUI + supervisor channel | `tui/` (`horde-worker`), `supervisor_channel.py`, `app_state.py` |
| Telemetry (Logfire/OTel) + run metrics | `telemetry.py`, `telemetry_spans.py`, `run_metrics.py` · `WorkerRunMetrics` |
| Dry-run / fault-injection test doubles | `fake_worker_processes.py`, `fault_injection.py`, `_canned_scenarios.py`, `harness.py` |

**Entry points:** `run_worker.py` (`run_worker`, headless) → `main_entry_point.py:start_working` →
`HordeWorkerProcessManager._main_loop`. The TUI dashboard is `tui/app.py` (`horde-worker`), which
launches the headless worker as a child. Other console scripts: `download_models`, `horde-worker-web`,
`horde-worker-host`, `horde-benchmark`. Full flag/env reference: [CLI](docs/reference/cli.md).

**Durable state** lives in a `.horde_worker_regen/` working-directory folder (state.json, owned_pids
.json, action_ledger.jsonl, perf_model.json), alongside `bridgeData.yaml`, `logs/`, and
`benchmark_results/`.

## Code quality

Follow the **[Haidra Python style guide](docs/haidra-assets/docs/meta/python.md)** (it is the
canonical reference). In brief: complete type hints on all public surfaces; `| None` over `Optional`;
`StrEnum`/`Enum` and small classes over magic strings and bare dicts; guard clauses over deep nesting;
never silently swallow exceptions; Google-style docstrings on public APIs; descriptive names. The
codebase is written for static analysis (pyrefly) and pydantic models are used DataClass-like.

- **Python:** `>=3.12,<3.13`. Dependencies/venv via **uv** (`uv sync`, `uv run <cmd>`; this repo and
  hordelib are often run with `uv run --no-sync`).
- **Line length:** 119. `ruff format` is the canonical formatter.

## Lint, format, type-check, test

```bash
# Format + lint (ruff is the linter and the formatter)
uv run ruff format .
uv run ruff check . --fix

# Type check (pyrefly is the type checker for this repo, not mypy)
uv run pyrefly check

# All hooks at once (ruff + pyrefly + file hygiene)
prek run --all-files

# Tests
uv run pytest                       # full suite (asyncio_mode = auto)
uv run pytest -m "not e2e"          # skip the slower full-lifecycle tests
uv run pytest tests/process_management/
```

- Tests live in `tests/`. `tests/process_management/` builds a testable manager via
  `make_testable_process_manager()`; `tests/e2e/` exercises the dry-run/fake flow end to end (marked
  `e2e`). `tests/test_chaos*` drive the fault-injection harness.
- Most pipeline tests run **without a GPU or network** using dry-run mode (`CannedJobSource` +
  `fake_worker_processes`); see [Architecture → Dry-run mode](docs/explanation/architecture.md#dry-run-mode)
  and `harness.py`.
- `AI_HORDE_TESTING=1` is read at runtime to suppress side effects (e.g. action-ledger file mirroring)
  during tests/harness runs.
- `prek` (not `pre-commit`) runs the hooks; the pinned `ruff`/`pyrefly` versions in
  `.pre-commit-config.yaml` must match `pyproject.toml` (there is a test that enforces this).

## Gotchas

- **Subprocesses must never download model references.** The parent owns reference downloading; use
  `reference_helper` to get an offline (read-only) reference manager in a child. On-disk layout facts
  live in `horde_model_reference.on_disk_layout`, not in worker-local code.
- **Telemetry is forced off by default** (it is expensive even with no collector); opt in only with a
  collector running. See [Telemetry](docs/explanation/telemetry.md).
- **Optimistic IPC sends:** the parent updates `ProcessMap`/`HordeModelMap` immediately after a send,
  before the child confirms; `process_launch_identifier` discards messages from replaced processes.
- **Config flows by reference:** sub-managers read `RuntimeConfig.bridge_data`; the file hot-reloads
  every 1 s unless config came from env vars (`-e`).

## See also

- [CONTRIBUTING.md](CONTRIBUTING.md): toolchain (uv, prek, ruff, pyrefly) and PR guidelines.
- [README.md](README.md): user-facing overview, support matrix, and install.
