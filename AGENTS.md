# AGENTS.md

Guidance for coding agents working in this repository. This file is a high-level map; the
[`docs/`](docs/index.md) tree is the source of truth for depth. Link a reader to a doc page rather
than duplicating it here.

> [!IMPORTANT]
> **Docs land in the same change that alters behavior.** A new module, config field, IPC message,
> scheduling/budget/recovery rule, or renamed entry point ships with the matching `docs/` edits in the
> same commit. Narrative pages (`explanation/`, `how-to/`, `reference/codebase-map.md`, `reference/logs.md`)
> are hand-written and rot silently; find the page that owns the changed concept via
> [docs/index.md](docs/index.md) and grep `docs/` for the touched terms. API stubs regenerate via
> `uv run --no-sync python docs/build_docs.py` (commit new/removed stubs; `git checkout --` any
> line-ending-only churn it causes). If a doc cannot be fully reconciled in the change, say so in the
> change description.

## What this is

**Horde Worker reGen** is the local GPU worker for the [AI Horde](https://aihorde.net/): it pops jobs
from the API, runs image inference (via `hordelib`/ComfyUI), screens results through a safety
classifier, uploads to R2, and submits back. Operators earn **kudos**; workers can also serve
**alchemy** (upscaling, face-fixing, interrogation, captioning).

**Why multiprocess:** inference is VRAM-heavy and ComfyUI is not thread-safe, so each GPU slot is its
own OS process. The main process never touches the GPU; it orchestrates inference/safety/download
children over IPC. See [Architecture](docs/explanation/architecture.md).

## Start here (docs)

- [Documentation home](docs/index.md): Diátaxis tree (tutorials / how-to / reference / explanation).
- [Architecture](docs/explanation/architecture.md): process model, shared-state pattern, asyncio loop.
- [Codebase map](docs/reference/codebase-map.md): file → responsibility quick reference.
- [Job lifecycle](docs/explanation/job_lifecycle.md),
  [Job state machine](docs/explanation/job_state_machine.md),
  [Resilience and recovery](docs/explanation/resilience_and_recovery.md),
  [Model downloads](docs/explanation/model_downloads.md),
  [Frontend and durable state](docs/explanation/frontend_and_state.md).
- [Admission pipeline](docs/explanation/admission_pipeline.md): where each preload, dispatch and
  clearance decision lives, what it reads, and how to change one;
  [Resource governance](docs/explanation/resource_governance.md) and
  [VRAM arbiter](docs/explanation/vram_arbiter.md) for the memory machinery behind it.

## Diagnosing a log bundle

Run the tool before hand-rolling anything. `horde-log sessions` lists each launch in an appended
`bridge.log`; `diagnose` gives ranked findings with remediation; `jobs` splits every job's wall clock
into pop->dispatch / generation / finished->submit plus the sampling-concurrency histogram (which
answers "is a multi-GPU worker actually using its cards"); `timeline` is the raw merged parent/child
event stream; `bundle` builds a redacted zip for a maintainer. Every finding id is catalogued in
[docs/reference/log_findings.md](docs/reference/log_findings.md).

**When a session needs a hand-rolled script, the tool needs a finding: add it there.** The copy a finding
shows a reader follows [docs/how-to/write-a-finding.md](docs/how-to/write-a-finding.md), enforced by
`tests/analysis/test_finding_copy_style.py`. And the log lines
`horde_worker_regen/analysis/` parses are a contract, registered in `analysis/log_signatures.py` with
their emitting sites. Before changing a log message in `process_management/`, grep that registry; if the
line is registered, update the pattern, the sample, and any detector in the same change, then re-run
`tests/analysis/test_log_signatures.py` and `-m slow tests/analysis/test_log_contract_dry_run.py`.

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
| Decide which model/job to preload & launch (the act site, the ticks, the actuators) | `scheduling/inference_scheduler.py` · `InferenceScheduler` |
| Admission decisions over a frozen per-cycle snapshot: preload gates, pricing, the materialisation request, clearance, the executor | `scheduling/admission/` · `SchedulingSnapshot`, `decide_preload_gates`, `price_preload`, `decide_clearance_admit`, `PlanExecutor` |
| Scheduler state ledgers: retention, safety placement, RAM reclaim, head admission, dispatch holds | `scheduling/ledgers/` · `RetentionLedger`, `SafetyPlacementLedger`, `RamReclaimLedger`, `HeadAdmissionLedger`, `DispatchHoldLedger` |
| Host RAM governance and the whole-card residency machine | `scheduling/governance/` · `ResourceGovernor`, `WholeCardResidencyLedger` |
| Model serviceability (one verdict for the pop offer and the preload gate) | `resources/model_serviceability.py` · `model_serviceability_verdicts` |
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

**Durable state** lives in `.horde_worker_regen/` in the working directory (state.json,
owned_pids.json, action_ledger.jsonl, perf_model.json, vram_footprints.json), alongside `bridgeData.yaml`, `logs/`, and
`benchmark_results/`.

## Code quality

Follow the **[Haidra Python style guide](docs/haidra-assets/docs/meta/python.md)** (canonical). In
brief: complete type hints on public surfaces; `| None` over `Optional`; `StrEnum`/`Enum` and small
classes over magic strings and bare dicts; guard clauses over deep nesting; never silently swallow
exceptions; Google-style docstrings on public APIs. The codebase is written for static analysis
(pyrefly); pydantic models are used dataclass-like.

- **Python** `>=3.12,<3.13`; dependencies via **uv** (`uv sync`, `uv run <cmd>`, often
  `uv run --no-sync` here and in hordelib).
- **Line length** 119; `ruff format` is the canonical formatter.

## Lint, format, type-check, test

```bash
uv run ruff format . && uv run ruff check . --fix
uv run pyrefly check                # pyrefly, not mypy
prek run --all-files                # all hooks (prek, not pre-commit)

uv run pytest                       # default sweep (opt-in bands skipped)
uv run pytest -m slow               # real subprocess spawns / multi-second workloads
uv run pytest -m gpu                # needs a real accelerator; auto-skips without CUDA
uv run pytest -m chaos_sweep        # generated wedge-liveness sweep (pre-release gate)
```

- `tests/process_management/` is grouped by subsystem and builds a testable manager via
  `make_testable_process_manager()`; `tests/e2e/` exercises the dry-run/fake flow (marked `e2e`).
- **Marker contract:** the `slow`, `gpu`, and `chaos_sweep` bands run only when the `-m` expression
  names them (`-m "not slow"` and `-m slow` both behave as written). The chaos sweep's full-worker
  half is also `slow`: `-m "chaos_sweep and slow"`. `CONTRIBUTING.md` carries the gate commands and
  the `HORDE_CHAOS_SEEDS` replay/widen override. `golden_regen` is a fourth opt-in band, and the only one
  that writes: it rewrites the committed scheduler traces under
  `tests/process_management/liveness/golden/` (`-m golden_regen`), so a regeneration is always a reviewed
  diff. The default sweep compares against them instead. `closed_loop` marks the simulator rows (incident
  scenarios, golden traces, the bounded matrix, the chaos core slice): they are in the default sweep, ordered
  after the rest of `process_management`, and cost seconds each, so an iteration gate is
  `-m "not closed_loop"` plus the rows named for the change, and the whole band runs once before a commit.
  That whole-band run takes `-n auto` (`pytest-xdist`, a dev dependency that is never on by default) on its
  own selection; `CONTRIBUTING.md` carries the command and the two caveats, chiefly that the suite's
  fastest-first run order is a serial property.
  Every run prints its thirty slowest tests (`--durations` is in `addopts`); read them off the tee before
  reaching for a stopwatch.
- **Rerun the failure, not the band.** Every `FAILED` line carries a node id; rerun exactly that
  (quote parametrized ids verbatim). One chaos seed replays via `HORDE_CHAOS_SEEDS=<seed>`. Tee long
  runs to a file once and grep the tee. Concurrent pytest runs in one checkout are safe: every test gets
  its own run root (`HORDE_WORKER_RUN_ROOT`, set by an autouse fixture), so the `.abort` sentinel, `logs/`
  and the state directory never collide. Keep CPU contention in mind for timing-sensitive rows.
- **Wait on a run, do not poll it.** Run pytest at full verbosity (never `-q`) redirected to a tee, in the
  foreground with a timeout, or once in the background and then wait for its completion. Sleep-and-tail
  loops and monitors on the tee are forbidden; an empty tee is a buffered redirect, not a hang, so check
  the python process's CPU time before calling a run stuck. A run that outlives its timeout is bisected by
  directory rather than waited on.
- Most pipeline tests run **without GPU or network** via dry-run mode (`CannedJobSource` +
  `fake_worker_processes`); see [Architecture → Dry-run mode](docs/explanation/architecture.md#dry-run-mode).
- `AI_HORDE_TESTING=1` suppresses runtime side effects (orphan reaping, action-ledger mirroring)
  during tests. The flip side: a killed test run can leave child processes alive holding VRAM
  indefinitely, and every real worker in a checkout shares the `logs/bridge_<n>.log` namespace. Before
  blaming a GPU-band wedge on the code, check for stale python children.
- **One real worker per card at a time.** A test that cold-boots its own worker cannot run beside a
  live warm session: the second worker's preloads starve under VRAM admission. This is why
  `tests/gpu/test_capability_probes.py` orders its cold baselines before the first warm probe.
- **The harness derives worker config from the scenario.** A cold run builds bridge data per scenario
  (`build_harness_bridge_data`); a warm session is provisioned once to the union ceiling of every
  scenario it will host (`WarmHarnessSession(scenarios=...)`), both through
  `_workload_capability_bridge_fields` in `harness.py`. A new job feature, workflow, or resolution
  axis must extend that shared derivation, or the warm path's jobs are ineligible at dispatch and the
  level scores on faults instead of running.
- **A level's completed/faulted split reads from per-level run metrics**, with tracker deltas as
  floor/fallback (`WarmHarnessSession._level_job_counts`). Per-job records trail the tracker's
  counters by a message pump, so they are read through a bounded settle
  (`_settle_level_metrics`), never at the instant a drain returns.
- The pinned `ruff`/`pyrefly` versions in `.pre-commit-config.yaml` must match `pyproject.toml`
  (a test enforces this).

## Gotchas

- **The orchestrator must stay torch-free.** The main process never loads torch (~500MB RSS). Two
  traps: (1) `hordelib.api` eagerly loads torch on any import; in parent/host/planning code import the
  torch-free origin submodule instead (`hordelib.feature_impact`, `hordelib.feature_requirements`,
  `hordelib.metrics`, `hordelib.utils.logger`, `hordelib.pipeline.constants`, `hordelib.preload`,
  `hordelib.utils.torch_memory`). (2) Device queries (`enumerate_accelerators`, `get_torch_*_vram_mb`)
  load torch when called; run them out-of-process via `utils/accelerator_probe.py::probe_accelerators`.
  `tests/process_management/manager/test_orchestrator_torch_free.py` is the tripwire.
- **Subprocesses must never download model references.** The parent owns reference downloading; use
  `reference_helper` for an offline reference manager in a child. On-disk layout facts live in
  `horde_model_reference.on_disk_layout`.
- **Telemetry is off by default** (expensive even with no collector). See
  [Telemetry](docs/explanation/telemetry.md).
- **Optimistic IPC sends:** the parent updates `ProcessMap`/`HordeModelMap` immediately after a send;
  `process_launch_identifier` discards messages from replaced processes.
- **Config flows by reference:** sub-managers read `RuntimeConfig.bridge_data`; the file hot-reloads
  every 1 s unless config came from env vars (`-e`).
- **Job acceptance is promised twice, and both promises must agree.** The popper advertises from live
  `bridge_data`; dispatch eligibility judges each card's effective config (`CardRuntime.config`, via
  `gpu/gpu_eligibility.py::reasons_card_cannot_serve`). A new field that gates jobs (an `allow_*`
  flag, a resolution/limit) must reach `resolve_all_effective_gpu_configs`, which both the boot build
  and the hot-reload refresh (`_refresh_card_configs`) derive from. When the surfaces disagree the
  worker accepts jobs it then hard-faults, and those faults feed the consecutive-failure pop pause and
  the fault-rate breaker.
- **`total_num_completed_jobs` is a movement counter: terminal jobs, faults included.**
  `num_jobs_faulted` rises at fault time with a per-job latch. Success is never `completed` alone;
  it is completed minus faulted, or the per-job run-metrics records.
- **Production breakers run under canned sources too.** The benchmark/harness path does not disable
  the pop pause or fault-rate breaker; a canned job that terminally faults is never re-served
  (`CannedJobSource`'s terminal-fault ledger), so a refusing worker fails a level in seconds instead
  of looping through breaker pauses.
- **A liveness proof the failure can satisfy is not a proof.** For any watchdog, escalation counter,
  or "did it recover" signal: prefer completion counters over attempt counters, scale expectations by
  the concurrency actually present, and never let a counter be reset by state the escalation's own
  remedy produces. Pair every "did not fire" test with a positive-liveness test that work flows. An
  adaptive sampler legally repeats its final step at full speed, so a stall check keyed on repeated
  steps faults healthy samplers.
- **Recovery must end in a working worker.** The escalation ladder is ordered least-destructive first
  and its endpoint is a running worker. Exiting is only a rung when something restarts the process (the
  dashboard, or `exit_on_unhandled_faults` plus a service manager). A rung that reproduces its own
  trigger is not a rung. See [Resilience and recovery](docs/explanation/resilience_and_recovery.md).
- **Do not gate a recovery action behind a preference flag.** Operators will switch it off without
  knowing they removed a recovery capability; give recovery paths their own keys (default enabled).
- **`getattr` on typed models is forbidden**, including to tolerate a partially-mocked test; read the
  field. `isinstance` narrowing is fine.
- **Mock bridge data is a maintenance contract.** Many `tests/process_management` tests use a `Mock`
  bridge data whose every attribute is truthy: a new field in `bridge_data/data_model.py` goes into
  `make_mock_bridge_data` (`tests/process_management/conftest.py`) in the same change, and an
  off-by-default branch needs `is True` or it fires in every mocked test.
- **Faulting a doomed job back to the horde sends the same burn to the next worker.** Before
  automating what follows a failure (retries, quarantines, breakers), ask whether the failure has to
  happen at all.

## Changing scheduling, memory, or recovery behaviour

These fail as interactions, not as units, so component tests stay green through most of what matters.

- **Decide over the snapshot, act through the executor.** Preload, dispatch and clearance admission are
  decisions over a frozen per-cycle `SchedulingSnapshot` that return plans (`scheduling/admission/`);
  the `PlanExecutor` is the one act site. A new gate reads a snapshot field (add it to the builder and
  pin it in `test_scheduling_snapshot.py`) and returns a decision or a command; it never reads a
  collaborator or sends a message. Where a decision needs to see an action's result, re-snapshot behind
  the action instead of reading live state. See [Admission pipeline](docs/explanation/admission_pipeline.md).
- **The golden traces are the parity gate for scheduler refactors.**
  `tests/process_management/liveness/test_golden_traces.py` diffs ten dispatch-world scenarios against
  `golden/*.json`; an intended behaviour change regenerates them with `-m golden_regen` and explains the
  diff in the commit. Anything else the diff surfaces is a regression.
- **Test in the simulator first.** `tests/process_management/liveness/_dispatch_world.py` runs the
  real scheduler, governor and ladder over fake children with a conserved VRAM ledger, keyed per card so a
  multi-card row states each card's own total, tenants and measured free (one entry on a single card). Changes to
  admission, retention, leases, governor thresholds or recovery rungs ship with a closed-loop test
  asserting an outcome (a duty floor, no safety teardown while serving, no free-VRAM crater). A live
  run confirms; it is not where you find out. Three fidelities are opt-in per row: `clearance_lease=True`
  makes dispatch a staging step (the lane sits `INFERENCE_PRIMED` holding its encode working set alone,
  a real `ClearanceController` is stepped over the scheduler's own `build_clearance_inputs` with
  `clearance_admit_process` as its `admit_fn`, and the weights land at clearance), and a lane held for
  `CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS` samples unpriced and is recorded in `world.clearance_timeouts`;
  `learned_footprints=True` sends each lane's per-tick memory report through a real `MessageDispatcher` into
  a `LearnedFootprintStore` the scheduler prices from (`world.footprint_store`), so a row can state what the
  parent learns from what its children report. It is opt-in because a parent that learns the children's
  true peaks stops over-committing within a few jobs, which would dissolve the premise of every row about
  over-commit under a fixed misprice. `child_reports_card_sized_peaks=True` makes a sampling lane report the
  whole card as its high-water, the reading an overflowed or checkpoint-caching child sends.
- **Every production incident becomes a permanent scenario** in `test_incident_scenarios.py`, written
  so that undoing its fix makes it fail. If the simulator cannot express an incident, extend the
  simulator.
- **Do not weaken an old defensive default** without reproducing the failure it guards at full scale.
  A single-process benchmark cannot disprove a multi-process problem.
- **Aggressive operator settings are supported targets.** Duty loss, VRAM pressure, model thrash and
  process recoveries are admission/scheduling/lifecycle defects; lowering `queue_size` or
  `max_threads` is not a fix.
- **Multiprocess is settled** (ComfyUI is not thread-safe), and the NVIDIA "Prefer No Sysmem
  Fallback" setting is off the table for volunteer operators, so on WDDM admission has to prevent
  overcommit itself.
- **The dispatch mix is operator configuration plus traffic randomness, never a stable property.**
  Operators offer one model or all of them and may lock any subset of slots to a model pool. Policy
  adapts to observed traffic per slot; validate scheduling/retention changes across signatures
  (diverse, pooled, heterogeneous), each judged by its own criteria. Log-derived numbers calibrate a
  scenario, never justify a hard-coded threshold.
- **No constant encodes one machine.** Cards run 8-24GB and system RAM 16-64GB. Budgets, windows and
  gates derive from measured capacity or the size of the work at hand, with a floor for the small end.
- **`VramArbiter` and `VerifiedReclaimLadder` return verdicts and emit nothing.** Telemetry goes
  through a `DecisionSink` at the call site; `WorkerRunMetrics.record_decision` deduplicates.
- **Any log line that can fire every tick** must be edge-triggered or rate-limited, with suppressed
  repeats dropped to TRACE.

## Measuring

- **Score against what a second of wall clock earns.** Per-iteration scores seat the cheapest work;
  many small jobs are worse than fewer large ones at identical iteration rates.
- **What you advertise at pop time is a promise about later.** A gate reading free VRAM right now
  leaks jobs it cannot serve. Start closed, require the condition to hold, open with margin, close at
  the bare requirement.
- **One hour of live traffic cannot rank two configurations**; alternate hourly or run for hours.
- **Kudos numbers from before the v22 pricing retrain** cannot be compared with ones after it.
- `state.jsonl` `kudos_hr` is this worker; the bridge log's "Total Kudos Accumulated" covers every
  worker on the account. Do not mix them.

## Working conventions

- **Do not run tests or builds during a soak or live run in the same checkout.** The CPU contention ruins
  the measurement. (Tests no longer share the live run's `.abort` sentinel or `logs/`: each test has its
  own run root. A live worker launched in the checkout still watches the checkout's `.abort`, so never
  drop one there to stop a test.)

## See also

- [CONTRIBUTING.md](CONTRIBUTING.md): toolchain (uv, prek, ruff, pyrefly) and PR guidelines.
- [README.md](README.md): user-facing overview, support matrix, and install.
