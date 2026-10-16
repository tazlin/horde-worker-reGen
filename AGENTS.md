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
- Responsibility maps, one per cluster, each a summary, a diagram and tables of decisions with their owner and
  rule: [runs and residency](docs/reference/responsibility-map-runs-and-residency.md),
  [clearance and child loads](docs/reference/responsibility-map-clearance-and-child-loads.md),
  [footprints and pricing](docs/reference/responsibility-map-footprints-and-pricing.md),
  [recovery and safety](docs/reference/responsibility-map-recovery-and-safety.md),
  [host RAM](docs/reference/responsibility-map-host-ram.md). Read the map for the cluster a change touches
  before the code; a decision with two owners there is the first thing to fix.
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
[docs/reference/log_findings.md](docs/reference/log_findings.md). A benchmark run (`bridge_harness.log`)
is a session of its own kind: it is listed after the worker sessions, numbered from #0 within its kind,
and addressed with `--benchmark N` or `--last-benchmark`; `--session` and `--last` are worker-only, so
worker numbering never moves when a harness log is present.

**When a session needs a hand-rolled script, the tool needs a finding: add it there.** The copy a finding
shows a reader follows [docs/how-to/write-a-finding.md](docs/how-to/write-a-finding.md), enforced by
`tests/analysis/test_finding_copy_style.py`. And the log lines
`horde_worker_regen/analysis/` parses are a contract, registered in `analysis/log_signatures.py` with
their emitting sites. Before changing a log message in `process_management/`, grep that registry; if the
line is registered, update the pattern, the sample, and any detector in the same change, then re-run
`tests/analysis/test_log_signatures.py` and `-m slow tests/analysis/test_log_contract_dry_run.py`.

**Symptom to first command:**

| Symptom | Finding | First command | Bundle file |
| ------- | ------- | ------------- | ----------- |
| Image intake held | `pop_liveness_full_queue` | `horde-log diagnose --last`, then `pop_gate` and `pop_gate_since` on the snapshot | `bridge.log` |
| Text backend wedged | `text_backend_wedged` | `horde-log diagnose --last`, then the backend counters in the hold line | `text_backend.log` |
| Inference slot retired, safety moved to the CPU | `inference_slot_retired`, `safety_start_escalated_to_cpu` | `horde-log timeline --last` for the `INFERENCE_START_RETIRED`, `INFERENCE_SLOT_RESTORED` and `SAFETY_START_ESCALATED_TO_CPU` ledger events | `action_ledger.jsonl` |
| Benchmark beside a worker | findings tagged as the benchmark run | `horde-log diagnose --last-benchmark`, stopped with `.abort_benchmark` | `bridge_harness.log` |
| GPU duty below target | the duty report | `horde-log duty --stats <bundle>/stats --logs <bundle>/logs` | `stats/*.jsonl`, `bridge.log` |
| Jobs dropped as "took too long" behind a run | none yet | grep `skipped the line (` by reason and count skips per displaced head (`ahead of job <id>`) | `bridge.log` |
| Card idle, queue full, a large model parked | none yet | grep `Deferring a new whole-card residency` and `does not open a recovery-supervisor grace window` | `bridge.log` |
| Sampler crawling under WDDM, partial load turned full | none yet | lane log `loaded partially` followed by `loaded completely` for one job; `reserve_mb=0` on the overflow check | `bridge_<n>.log` |
| Safety on the CPU for a whole session | none yet | `safety_placement` restore lines; replay `restore_headroom_fits` over the status inputs | `bridge.log` |
| A small model's verdict or price jumps to a large model's | none yet | `Dominance verdict for device` lines; a different model dispatched onto a lane holding this one just before | `bridge.log`, `config/vram_footprints.json` |

**Reading duty.** The duty report's "GPU time lost" labels are an order of precedence over what was outstanding
(a pending submit outranks a pending safety check, which outranks every lane state), not costs; the costs are the
per-job phase medians and the `job_completed` rows' `phase_metrics` (sampling seconds and per-step rates, encode,
decode, `ram_to_vram`, the post-processing chain, safety, and the job's measured device peaks). The
`clearance_hold` slot bucket carries the cleared child's load as well as the arbiter's hold. The stats rows are
`{"event": "stats_sample", "sample": {...}}` plus `decision`, `resource_state`, `job_completed` and one
`session_start`; their epochs are UTC while the logs are the host's zone. A recorded session's `job_completed`
rows are also the source of a replay fixture for the dispatch-world simulator (`tests/process_management/liveness/replay/`),
built with `horde-log replay-fixture <bundle> --session N <out_dir>` (see `docs/reference/cli.md`).

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

### Scheduling and admission, by mechanism

Start from the mechanism, not the file. Each line states the owner of one question; read outward from it.

- **What an empty sampling slot was waiting on:** `InferenceScheduler.record_slot_duty` and `slot_duty.py`.
  The `clearance_hold` bucket is keyed on the lane state and a cleared child stays `INFERENCE_PRIMED` until its
  first step, so the bucket carries the child's weight upload and prompt encode as well as the arbiter's hold.
  Split it with the `Clearance held` and `cleared process` lines: dispatch to clearance is the gate, clearance to
  first step is the child.
- **Whether a staged child may load and sample:** `scheduling/clearance_lease.py` (`ClearanceController.step`,
  `decide_clearances`, `_clear`, `_clear_stage`; the child's `ClearanceLeaseProxy` serves the grant's device-free
  and activation-reserve figures once each) and the scheduler's `clearance_admit_process` and
  `clearance_stage_admit_process`. The arbiter's verdict is `resources/vram_arbiter.py` (`evaluate`,
  `_partial_seat_admission`, `_probe_due`, `_card_converged_empty`); the identity it tests is
  `resources/admission_identity.py` (`candidate <= device free - outstanding - margin`). A waiter's starvation
  clock runs through an applied reclaim rung and restarts only when measured free rises by the progress epsilon
  since the send.
- **What a job costs the card:** `scheduling/admission/pricing.py` (`partial_seat_mb`, `sampling_reserve_mb`,
  `has_reclaimable_idle_model`, `sweep_spares_idle_copy`, the ready seats), `resources/vram_footprints.py`
  (`LearnedFootprintStore`, stages `SAMPLE`, `SAMPLE_ACTIVATION`, `RESIDENT`, keyed by platform) and
  `resources/resource_budget.py` (`StreamForecast`, the static seeds, the booking decay). A dispatch booking is
  set at clearance against the lane's report and decays only by growth past the lowest report since, so a job
  past its first step can still carry gigabytes of booking.
- **Whether the card has room nobody holds:** `resources/foreign_vram_floor.py` (a 120 s trailing minimum of
  `total - free - worker footprint`, clamped at 0 and withheld until the window fills) and the scheduler's
  `_sustained_foreign_floor_mb` and `current_foreign_floor_mb`; the worker footprint is
  `process_map.committed_vram_mb` (context constant plus reservation per committed process). A reading at or
  below zero comes from the worker's own stale or over-charged footprint, never from the card.
- **Whether a lane keeps its weights between jobs:** the scheduler's `_should_keep_model_resident`
  (governor gate, repeat evidence, static fit through `_retention_fit`) and `ledgers/retention.py`
  (`RetentionFit`, `slot_has_repeat_evidence`); the child honours the verdict as `defer_vram_unload`.
- **Whether a model takes the whole card:** `scheduling/governance/whole_card.py` (`WholeCardResidencyMachine`,
  `WholeCardResidencyLedger`, `teardown_complete`, the pop claim and `WholeCardPopClaimRelease`), the scheduler's
  `_whole_card_*` methods, `wants_whole_card` in `pricing.py`, and the settings `whole_card_exclusive_residency`,
  `whole_card_residency_safety_off_gpu`, `whole_card_residency_max_hold_seconds`,
  `large_model_reentry_cooldown_seconds`. The verdict is a two-rung dominance ladder (`card_dominance`): rung 1
  (`CANNOT_CO_SAMPLE`) is a dominant-model run, rung 2 (`CANNOT_CO_RESIDE`) the teardown residency. `held()`
  returns every rung and `teardown_held()` rung 2 only, so a reader whose premise is a teardown reads the latter.
  A run starts at its first step and ends in `_restore_siblings_after_whole_card`, either drained or with a
  different-model head past `_head_aged_past_anti_starvation`, or when a rung-2 grant for a different model takes
  the card (`_end_run_superseded_by`). Every run end is written by `_end_dominant_run`, and every reader of a run
  goes through the ledger.
  The establish-rate limiter is the only governor that defers (`DEFER_RESIDENCY_GOVERNOR`, which does not stop the
  preload pass, and a head it withholds at dispatch is stepped past by a `diversity` line-skip); the grace
  allowance only decides whether an excuse window opens (`grace_seconds_to_open`). The owners and rules, and
  which head each reader means, are in
  [the runs and residency map](docs/reference/responsibility-map-runs-and-residency.md).
- **Whether the host has room for a load:** `scheduling/admission/preload.py` (`decide_ram_admission` against
  physical RAM, `decide_commit_admission` against commit at the checkpoint's file size), the governor's tiers in
  `scheduling/governance/ram_governor.py`, and the child's private reading in `utils/private_memory.py`. Every
  consumer of each figure is in [the host RAM map](docs/reference/responsibility-map-host-ram.md).
- **How long a queue head may wait:** `scheduling/dispatch_affinity.py` owns the two clocks that read the horde
  deadline (the skip budget, 6 skips or 0.2 x ttl, and the anti-starvation override at 0.3 x ttl from pop). Every
  other hold has its own fixed clock (head protection, the RAM defer barrier, the establish-rate window, the claim's max
  hold, the 60 s lease-acquire timeout, the structural-wedge minimum). Pin-wait, `residency_bypass_models` and the
  `dominant_run` seat do not consult the deadline clocks. A new hold that can delay a head reads
  `dispatch_affinity`'s clocks rather than adding one.
- **Where a learned sampling reading is filed:** three feeders write the footprint store (the monolithic feeder
  `MessageDispatcher._observe_footprint_peak`, the disaggregated feeder in the scheduler and the `RESIDENT` feeder
  in `vram_footprints.py`), every one under the job's model, and a sampling reading counts only while the lane
  holds that model. The store bounds readings at the feeder, at commit and at load
  (`sanitize_sampling_observations`). Four readers price different figures for one job (dispatch, clearance,
  residency head need, verdict), listed with their owners in
  [the footprints and pricing map](docs/reference/responsibility-map-footprints-and-pricing.md).
- **Which lane a preload or a sweep may touch:** `scheduling/admission/preload.py` (`lanes_committed_elsewhere`,
  the three target selectors) and `pricing.sweep_spares_idle_copy`: a lane loading one model is no target for
  another, a lane that owns a dispatched job is busy whatever it last reported, and the queue head's ready seat is
  spared by the clearance and pressure sweeps. The commitment is bound to the model map's `LOADING` entry
  (`models/horde_model_map.py`, `PRELOAD_FIRST_REPORT_GRACE_SECONDS`), not to the control flag.
- **What the child does between its clearance and its first step:** hordelib `execution/sampling_lease.py`
  (the `CFGGuider.sample` hook: encoder release, stage upload, the sample permit) and
  `execution/comfy_patches.py` (`_load_models_gpu_hijack`, the class-list skip, the overflow check charging the
  held reserve, the free-view clamp); the worker's `workers/inference_process.py` handles one control message at
  a time, so a preload or unload queued ahead of `START_INFERENCE` delays the job. The force-load skip list is
  `worker_entry_points.FORCE_LOAD_SKIP_BASELINES`. The grant can reach the child before the child starts the job.
  The parent clears a staged waiter at dispatch, `ClearanceLeaseProxy.begin_job` unsets the shared figures, and
  `_refresh_unconsumed_grant_figures` rewrites both the device free and the reserve of every unconsumed grant on
  each step from the staged waiter's price, so a grant issued before the job start carries its reserve one step
  later. The lease brackets the stage upload and
  the denoise only; the encoder load, the encode and the VAE decode run outside it. In a run, the released
  encoder's reload for each job pages part of the kept diffusion weights out of the card.
- **Who may replace a process:** eleven paths, with no shared debounce or view of boot state.
  `process_lifecycle.py` (`_reap_if_crashed`, `replace_hung_processes` and its watchdogs, the last reclaim rung
  `_replace_if_paged_and_slow`, `rebuild_inference_pool`), `worker_recovery_coordinator.py` (the orphaned-verdict
  watchdog, `perform_soft_reset`, `assess_wedge`), `recovery_supervisor.py` (`RecoverySupervisor`, whose give-up
  and soft-reset branches both read `boot_in_progress`, bounded by the boot hard cap), and the scheduler's
  clearance-refused and unload-failure paths. Only the `replace_hung_processes` branches share
  `_recently_recovered`. The orphaned-verdict watchdog replaces safety only when nothing is alive to progress
  (`is_safety_capacity_available`); `is_safety_pool_ready` (idle and accepting) is the intake question, never a
  replacement trigger. A new replacement path reads boot and busy state and goes through that debounce.
- **Where safety runs:** several mechanisms move safety off the GPU (runtime placement demotion, reclaim-ladder
  requests from the governor, the arbiter and the SOS remedy, whole-card residency, start escalation, the per-check
  CPU route, weight demotion in place); one reconciler, `_reconcile_runtime_safety_placement`, restores it,
  gated by `SafetyPlacementInputs.restore_headroom_fits` and the restore dwell (`safety_placement.py`). Weight
  promotion (`_promote_safety_weights_if_room`) has its own promoter.
- **Lanes held down for a beneficiary:** the reclaim ladder's restore obligations stay held while
  `ProcessManager._beneficiary_waiting` reports the booked job waiting, asked with the paused lane's kind, so a job
  waiting for post-processing never holds the post-processing lane. The post-processing orchestrator's 90 s
  age-out runs under a ladder pause regardless.
- **Whether a LoRA is on disk:** two owners, the parent's `JobTracker._cached_lora_keys` and hordelib's ad-hoc
  cache in the download process (eviction pins from `aux_prefetch_coordinator.py` cover every queued job). The
  cache does not report an eviction to the parent.

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
- **How to test is written down in `TESTING.md`**, for human contributors: install per compute backend,
  the bands and markers, what CI covers, non-NVIDIA and CPU-only checks, the bootstrap, real hardware,
  text generation, the dashboard, the live maintenance-mode run, and reading a run. Read it
  before running anything beyond a targeted file. It names only committed tooling; scripts under
  `docs/plans/` never belong in it. When a band, a marker, a harness mode or a live-run step changes,
  update `TESTING.md` in the same change. The rules below (tee, rerun, wait) are agent discipline and
  stay here.
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
- **A warm level ends, and is counted, on its finalized jobs.** A job's run-metrics record is written at
  finalization, after safety and submit, and the tracker's completion counter moves a whole safety check
  earlier, at the inference result. The drain therefore waits for a record per job, and the count
  (`WarmHarnessSession._level_job_counts`) reads only records popped after the level was installed, so a
  job the previous level left in safety is never counted. Tracker deltas are the fallback for a driver
  with no run metrics and the floor for a fault that never finalizes. A level's elapsed time and timeout
  include safety and submit.
- **Two locks must agree.** `requirements/utilities/pyproject.toml` pins torch to the version the root
  `uv.lock` resolves, and a root bump does not move that pin. After one, re-pin and run `uv lock` in
  `requirements/utilities`. CI's `lock-drift` job fails otherwise.
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
- **A card can carry a tenant nobody lists.** A graphics or driver tenancy can hold gigabytes that NVML's
  compute-process list does not show; the worker learns it only through the foreign floor. The floor is learned
  from positive readings only (a reading at or below zero is an over-charged worker footprint, never evidence of
  zero foreign), and where the driver reports per-process usage the worker's own share is measured from its
  pids rather than estimated from the ledger. Price against the floor; never assume the card offers its total.
- **The encoder's device residency is decided at its own job's diffusion load.** ComfyUI's `free_memory` pass
  partially unloads the job's text encoder to fit the diffusion weights (a copy back and pin per module), unless
  hordelib has released it whole through its CPU origins first; a waiter's encoder also fills the room a stage grant
  needs. The job-end component rule decides the RAM cache, not the device.
- **A partial-load seat assumes the child leaves the priced activation room.** The grant carries that room as a
  figure hordelib bounds ComfyUI's partial sizing with; without it ComfyUI leaves only its own estimate and a
  whole-card model faults in the sampler with the weights fully loaded. The room is the measured activation of
  jobs that sampled with their weights whole, never the raise-only watermark: a partial load's peak less its
  resident share reads streamed weights as activation, and a watermark priced from it forces every later load
  deeper into partial mode.
- **`PRELOADED_MODEL` is busy to every reader but a residency's teardown.** A preload predicts a dispatch, so the
  state is busy for dispatch, lease and reclaim; a held whole-card residency withholds that dispatch, so its shrink
  and its safety move treat a lane that holds only a preload as stoppable (`holds_only_a_preload`). Left busy, the
  residency can never reach its target and the card idles with both jobs pending.
- **A lane's last control flag is the last command sent.** The VRAM sweep skips a lane whose flag already reads
  an unload, and a service lane never sends the `LOADED_IN_VRAM` report that retires it, so a dispatch to a
  service lane records its own command (`START_POST_PROCESS`) or the lane is unloaded once per launch and keeps
  every module it loads afterwards.
- **The overlap headway is read over sampling jobs.** Under the clearance lease a dispatched job without a grant
  is not a running sampler; held as one, it parks every later job behind a job that may be waiting for the room
  those jobs' lanes hold.
- **The dispatch world's five fidelity traps.** A served class outside its catalogue is charged nothing unless
  it resolves through the row's own set; the lanes report no context unless `lane_reports_carry_context` is on,
  so the scheduler's foreign reading runs above the scripted tenancy; the admission margin is the WDDM buffer
  unless `admission_noise_mb` states the session's; regrown lanes, absent post-processing lanes and RAM-held idle
  lanes each had a path that wedged or crashed the row before they were modelled; and a replay that reads far from
  live is a simulator finding before it is a scheduling one.
- **The world starts the child's job at dispatch.** It calls `begin_job` and enters the wait in the dispatch tick,
  so a grant never precedes the job start and a clear never precedes the encode, and a run's per-job encoder
  reload does not page the kept weights out. A clearance-transport or run-duty claim needs a row that drives that
  order, or a live measurement.
- **On one card the preload pass ends at its first stop of any kind.** `stop_is_card_scoped` in
  `admission/preload.py` matters only with multi-GPU routing active; on a single card, whatever decision a parked
  head records decides whether anything behind it preloads. `DEFER_RESIDENCY_GOVERNOR` is not a stop, since a head
  standing down for the establish-rate limiter lets the pass go on.

## Changing scheduling, memory, or recovery behaviour

These fail as interactions, not as units, so component tests stay green through most of what matters.

- **Decide over the snapshot, act through the executor.** Preload, dispatch and clearance admission are
  decisions over a frozen per-cycle `SchedulingSnapshot` that return plans (`scheduling/admission/`);
  the `PlanExecutor` is the one act site. A new gate reads a snapshot field (add it to the builder and
  pin it in `test_scheduling_snapshot.py`) and returns a decision or a command; it never reads a
  collaborator or sends a message. Where a decision needs to see an action's result, re-snapshot behind
  the action instead of reading live state. See [Admission pipeline](docs/explanation/admission_pipeline.md).
- **The golden traces are the parity gate for scheduler refactors.**
  `tests/process_management/liveness/test_golden_traces.py` diffs eleven dispatch-world scenarios against
  `golden/*.json`; an intended behaviour change regenerates them with `-m golden_regen` and explains the
  diff in the commit. Anything else the diff surfaces is a regression.
- **Test in the simulator first.** `tests/process_management/liveness/_dispatch_world.py` runs the
  real scheduler, governor and ladder over fake children with a conserved VRAM ledger, keyed per card so a
  multi-card row states each card's own total, tenants and measured free (one entry on a single card). Changes to
  admission, retention, leases, governor thresholds or recovery rungs ship with a closed-loop test
  asserting an outcome (a duty floor, no safety teardown while serving, no free-VRAM crater). A live
  run confirms; it is not where you find out. Every lane reports what the card charges it, and every
  `UNLOAD_MODELS_FROM_VRAM` the scheduler sends frees the lane's device weights, including one sent down the
  pipe alone with no control flag (the eviction ahead of a cross-model dispatch). Reports arrive every tick,
  where a child reports every `MEMORY_REPORT_INTERVAL_SECONDS`. Five fidelities are opt-in per row:
  `clearance_lease=True` makes dispatch a staging step (the lane sits `INFERENCE_PRIMED` holding its encode
  working set alone and reporting it, a real `ClearanceController` is stepped over the scheduler's own `build_clearance_inputs` with
  `clearance_admit_process` as its `admit_fn`, the weights land at clearance, and the lane stays
  `INFERENCE_PRIMED` through their load until its first step, as a production child does), and a lane held for
  `CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS` samples unpriced and is recorded in `world.clearance_timeouts`.
  `two_phase=True` (with the lease) gives each lane a stage permit and steps the controller with the scheduler's
  `clearance_stage_admit_process`: a stage grant commits the lane's weights alone while another lane samples
  (recorded in `world.stage_grants`), and the activation and sampling window land at the sample grant.
  The child acquires through a real `ClearanceLeaseProxy` built with a shared device-free figure, so a granted
  child rebases its free-VRAM view on the figure its grant carries and ages it from the grant, as hordelib's
  clamp does; a lane that times out keeps its dispatch figure.
  Under the lease, `staged_lane_prefetch=True` starts a staged lane's read of its reclaimed checkpoint pages at
  dispatch and touches them each staged tick, so clearance pays only the unfinished read, as hordelib's
  prefetch-and-touch does; `tail_overlap=True` turns on the controller's handoff window and feeds it what
  production does, each sampler's step position through `ProcessMap.on_heartbeat` and each paid load as a
  `ram_to_vram` phase through `ProcessMap.on_job_metrics`; `gpu_sampling_lease_slots` sets the controller's cap
  apart from `max_threads`. `idle_lane_warm=True` has a lane honour the parent's warm message
  (`InferenceScheduler.warm_idle_lanes`): an idle lane holding the named model re-reads its reclaimed pages
  ahead of dispatch (off, the lane ignores the message); a dispatch adopts a running warm and clearance pays only
  the unfinished part. All four default to the off or `max_threads` value. `performance_model=True` gives
  the scheduler an in-memory, unseeded `PerformanceModel` fed each finished job's sampling rate and its
  finalization, so a dispatched slot carries an expected sampling time once a signature has enough samples;
  `learned_footprints=True` sends each lane's per-tick memory report through a real `MessageDispatcher` into
  a `LearnedFootprintStore` the scheduler prices from (`world.footprint_store`), so a row can state what the
  parent learns from what its children report. It is opt-in because a parent that learns the children's
  true peaks stops over-committing within a few jobs, which would dissolve the premise of every row about
  over-commit under a fixed misprice. `child_reports_card_sized_peaks=True` makes a sampling lane report the
  whole card as its high-water, the reading an overflowed or checkpoint-caching child sends.
  `host_ram=HostRamLedger(...)` conserves host RAM (private pages, transients, reclaimable checkpoint cache):
  a load allocates when it completes and reports its peak, and a device unload keeps the model in RAM.
  With it, a device load of a RAM-held checkpoint also reads back whatever pages the host reclaimed (least
  recently used first, at the ledger's `read_mb_per_second`, under foreign RAM a row can script per tick with
  `foreign_mb_by_tick`), and a monolithic load carries its text encoder and VAE.
  `phase_resident_components=True` charges a sampling job its sampler-only footprint and a decode tail its
  weights plus the decode spike, as ComfyUI loads components separately; the encode phase is not modelled.
  `partial_loads=True` models hordelib's and ComfyUI's partial loads: a lane's offloaded remainder survives the
  job and the next same-model load pays to return it at the load rate, the decode spike unloads the lane's own
  weights first, and a model on the worker's force-load skip list (`worker_entry_points.FORCE_LOAD_SKIP_BASELINES`)
  loads only into believed free and keeps its support components resident as a lane tenant; each window records
  `load_moved_mb`, `load_returned_mb` and `decode_unloaded_mb`. `lane_reports_carry_context=True` has every
  process report its share of the card's context charge, so the scheduler's instantaneous foreign reading equals
  the tenancy a row scripts with `foreign_vram_mb_by_tick`; off, the reading exceeds it by the world's context
  charge less the scheduler's resolved constant. Rows that state a true workload can also pass
  `extra_model_classes` (served classes resolve and are charged whatever the module lists), a `true_footprint_mb`
  hook (resident and transient per job), `sample_seconds_by_job` and `decode_seconds_by_job`, and a seeded
  `footprint_store`; `tests/process_management/liveness/replay/` holds a live session's job stream and learned
  store for that purpose, built with `horde-log replay-fixture`. A replay measures a run with the pop driver's
  bounded look-ahead (`pop_driver(look_ahead_rows=N)`; a refused row keeps its order), the large-model pop
  governor under `large_model_switch_min_seconds` and `large_model_reentry_cooldown_seconds` (0 and 0 skip it;
  a faithful replay of the recorded host passes 45 for re-entry), and the figures `kudos_per_hour`,
  `load_totals` per model, `partial_loads` and `complete_loads`. The driver has two supply models. At the
  recorded offsets it offers each row when the live worker popped it, which calibrates the world against the
  session but starves a world that outruns the live one on the fixture rather than on the card.
  `pop_driver(on_demand=True)` offers the next row whenever the world asks, keeping the queue budget and the
  look-ahead, and is the model a duty claim is proven on. A tick spans several of the manager's 0.1 s loop
  iterations, so the tick closes with a clearance pass for the jobs its dispatch staged
  (`_clear_waiters_dispatched_this_tick`), as the next iteration would price them; without it every dispatched
  job waited a whole tick for its first clearance pass and the on-demand whole session read seven points low.
  The world prices for the WDDM platform shape (`pricing_platform="win32"`: the admission margin's fraction and
  ComfyUI's inference reserve both differ on Windows), so goldens and rows read the same on a Linux runner as
  on the host they were sized on; a replay of a session recorded elsewhere passes that host's token.
  Under `learned_footprints` the world also hands the dispatcher a finished job's resident weight set
  (`_report_sampled_weight_set`), so the `SAMPLE_ACTIVATION` key is learned as production learns it. Under
  `performance_model` the world feeds
  `note_first_step` at each job's first step, so a rule that reads learned dispatch overheads can fire in a
  replay, and a service lane keeps its served unload flag as production does, so a card reads converged-empty
  once nothing is left to reclaim. `text_encoder_released_before_wait` defaults to on wherever the lease,
  `partial_loads` and `phase_resident_components` are all on, and drops the staged encode and the support
  tenant before the clearance wait, as hordelib releases the encoder whole; `lane_start_seconds` holds a
  regrown lane in `PROCESS_STARTING` with no context for that long (default 0). A lane's reported high-water
  carries no context share and no decode spike, since a child's allocator never holds either, so the store the
  replay learns from does not climb; the fixture's truth hook charges the store's figures without their margin,
  which the parent adds once.
  `seed_ram_held_checkpoint` makes a lane report its seated checkpoint as a held component sized by its RAM
  pages with nothing of it on the device, the report of a pooled monolithic lane. `post_process_chain_peak_mb`
  gives the post-processing lane modelled chains that go through production's sampling co-residency check
  against the lanes cleared to sample, where a lane holding only a stage grant does not hold the card. With
  `post_process_chain_measured_admission=True`, which is off by default, a real `PostProcessOrchestrator` prices
  each chain against the tick's frozen measurement and runs a refused chain's reclaim plan through the
  scheduler, one verified rung per settle. Every run records in `world.structural_queue_wedges` each queue
  deadlock that production's detector held past the structural-wedge horizon, which is the state save-our-ship
  acts on. Every run keeps
  `world.job_windows` (each job's dispatch, clearance, load and sampling instants, and what its load paid) and
  each lane's state on `tick_observations`, from which `_world_assertions.py` reads handoff gaps, the staging
  overlap share, the two-sampler share, dispatch-to-first-step seconds, each load's read split at clearance,
  disk reads on lanes booked as holding their model, and each child's weight-load decision (believed free
  against true free, and what it left in host RAM) through `child_load_decisions`.
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
  drop one there to stop a test. A benchmark claims its own `.abort_benchmark` sentinel
  (`run_root.AbortSentinelKind`), so aborting a benchmark never stops a live worker and its startup
  cleanup never removes the worker's file.)
- **Delegated agents run targeted test files only; the orchestrator runs the long suites.** A subagent's
  context cache lives five minutes, so a delegate waiting on a long run pays for its whole context again
  and a finished delegate cannot be resumed cheaply. Briefs name specific test files; whole packages, the
  slow bands and the full suite are run by the session that owns the tree, after the reports are in.
- **A liveness proof the failure's own bookkeeping can satisfy is not a proof.** Take an acknowledgement
  marker after the request is sent (the send path may restamp it), and count end-to-end completions
  (jobs reaching submit), never inference results, as progress: results parked behind a missing safety
  process moved `total_num_completed_jobs` for hours.
- **Every replaced child is ended, whatever its state.** A child still in `PROCESS_STARTING` cannot read
  an end request; retiring only its map entry leaves an OS process that outlives the parent and holds
  the card. Replacement terminates, joins and drops the owned pid; a harness run kills owned children at
  exit itself because it bypasses `start()`.

## See also

- [CONTRIBUTING.md](CONTRIBUTING.md): toolchain (uv, prek, ruff, pyrefly) and PR guidelines.
- [README.md](README.md): user-facing overview, support matrix, and install.
