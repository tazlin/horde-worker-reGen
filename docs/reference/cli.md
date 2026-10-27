# Command-line reference

Entry points, flags, and environment variables. For task-oriented walkthroughs, see the
[how-to guides](../how-to/install.md).

## Launcher scripts

These wrappers prepare the environment, then call the entry points below. Use the `.cmd` form on
Windows, `.sh` on Linux/macOS, and the `-rocm` variants on AMD Linux.

| Script | Purpose |
|--------|---------|
| `install.ps1` / `install.sh` | One-line installer (download release, bootstrap runtime, launch). |
| `update` | Update managed source and dependencies. In a git checkout, sync dependencies without changing source. |
| `update-runtime` | Install or update dependencies into the managed environment. |
| `horde-worker` | Launch the worker: web browser dashboard by default, `--terminal` for the in-terminal UI, or `--headless` for no UI (foreground worker, downloads models first). |
| `horde-bridge` | Run the headless worker (downloads/verifies models first). |

`horde-worker` and `horde-bridge` pass any extra arguments through to the underlying program.

All managed launcher paths verify the private uv and synchronize stale dependencies before importing worker
code. See [Updates and bootstrap](../explanation/updates_and_bootstrap.md) for the update model and its
failure guarantees, or [Update the worker](../how-to/update-the-worker.md) for the operator procedure.

## Update and dependency sync

`update` checks for a release, applies it when the installation is managed, then synchronizes dependencies.
In a git checkout, an applying `update` skips release discovery and synchronizes the checkout produced by
`git pull`. `update --check` remains a release query and does not synchronize dependencies.
`update-runtime` performs only the dependency synchronization.

If release discovery fails, an applying managed `update` still synchronizes the installed source and
lockfile. It returns exit code 1 because it could not establish whether newer source exists, and reports
that the worker can start when local synchronization succeeded.

### Update-only flags

| Flag | Effect |
|------|--------|
| `--check` | Report whether a release is available without applying or synchronizing it. |
| `--yes` | Apply an available release without prompting. |
| `--repo OWNER/REPO` | Use and record a different GitHub release repository for this update. |

### Shared update and sync flags

These flags work with both `update` and `update-runtime`.

| Flag | Effect |
|------|--------|
| `--backend BUILD` | Select `cu126`, `cu130`, `cu132`, `cpu`, `rocm`, or `rocm-windows`. |
| `--cu126`, `--cu130`, `--cu132`, `--cpu`, `--rocm`, `--rocm-windows` | Shortcuts for `--backend BUILD`. |
| `--hold-torch` | Keep the installed torch and torchvision when the new dependency graph permits it. The image-utilities venv is deferred with it, so the held torch is not downloaded a second time. |
| `--no-hold-torch` | Take the locked torch and torchvision versions. |
| `--no-sync-preview` | Skip the dependency-change preview. |
| `--confirm-above-mb N` | Prompt before an optional download larger than `N` MB. The default is 1500 MB. |
| `--headless-policy proceed\|hold` | Choose whether a non-interactive optional torch update proceeds or is held. |
| `--no-prune` | Keep superseded files in the worker-owned uv cache after a successful sync. |
| `--skip-utilities` | Skip provisioning the separate image-utilities environment. |
| `--cache-mode isolated\|shared` | Use the worker-owned cache or uv's system cache. Shared caches are never pruned automatically. |

The normal sync uses the locked dependency versions. A requested torch hold drops the lock only for the
held resolution and succeeds only when all current dependency requirements still resolve. The
image-utilities venv is left unchanged until a sync takes the newer torch. When that venv does not exist yet,
its capabilities remain unavailable during the hold.

### Release channels

The default `stable` channel considers final `vX.Y.Z` releases. The `beta` channel also considers
pre-releases. A beta installation continues receiving newer betas and advances to the corresponding final
release when it becomes available. Version comparison never offers an older release.

The release repository is resolved in this order:

1. `HORDE_WORKER_UPDATE_REPO`;
2. the origin baked into `worker_bootstrap/release-origin` by the release workflow;
3. `repo=` in `bin/install-info`, written by the installer or `update --repo`; and
4. `Haidra-Org/horde-worker-reGen`.

The baked origin travels with a release and can redirect an installed channel on its next update. Use the
environment override when an installation must remain pinned to a repository.

### Update and sync environment variables

| Variable | Values and effect |
|----------|-------------------|
| `HORDE_WORKER_AUTO_UPDATE` | `prompt` (default), `auto`, or `off` for launch-time managed updates. |
| `HORDE_WORKER_UPDATE_CHANNEL` | `stable` (default) or `beta`. An installed prerelease implies `beta`. |
| `HORDE_WORKER_UPDATE_REPO` | `OWNER/REPO` release-origin override. |
| `HORDE_WORKER_SYNC_PREVIEW` | Set to `0` to disable the dependency preview. |
| `HORDE_WORKER_SYNC_HOLD` | Set to `1` to request a compatible torch hold. |
| `HORDE_WORKER_SYNC_CONFIRM_MB` | Optional-download confirmation threshold in MB. Default: `1500`. |
| `HORDE_WORKER_SYNC_HEADLESS_POLICY` | `proceed` (default) or `hold`. |
| `HORDE_WORKER_SYNC_PRUNE` | Set to `0` to retain superseded files in a worker-owned cache. |
| `HORDE_WORKER_SKIP_UTILITIES` | Set to `1` to skip the image-utilities environment. |
| `HORDE_WORKER_UV_CACHE_MODE` | `isolated` (default) or `shared`. |
| `UV_CACHE_DIR` | Explicit uv cache location. An explicitly selected cache is never pruned automatically. |
| `HORDE_WORKER_UV_VERSION` | Override the launcher's pinned uv version for bootstrap maintenance. It must satisfy `[tool.uv] required-version`. |

## Console entry points

Installed as console scripts (defined in `pyproject.toml`):

| Command | Module | Role |
|---------|--------|------|
| `run_worker` | `horde_worker_regen.run_worker:start` | Headless worker. |
| `download_models` | `horde_worker_regen.download_models:main` | Download and verify configured models. |
| `horde-worker` | `horde_worker_regen.tui.app:main` | The dashboard (TUI). |
| `horde-worker-web` | `horde_worker_regen.tui.web:main` | Serve the dashboard over the web. |
| `horde-worker-host` | `horde_worker_regen.tui.worker_host:main` | Background worker host the web dashboard attaches to. |
| `horde-benchmark` | `horde_worker_regen.benchmark.cli:main` | Progressive benchmark. |
| `horde-duty-report` | `horde_worker_regen.analysis.duty_log_report:main` | Per-session GPU duty-cycle report over stats JSONL, with `bridge.log` fallback. Also `horde-log duty`. |
| `horde-log` | `horde_worker_regen.analysis.log_triage_cli:main` | Triage worker logs: sessions, timelines, and what-went-wrong findings. |
| `horde-stats` | `horde_worker_regen.stats_operations:main` | Compress or downsample retained stats JSONL files. |

## `horde-worker` (dashboard)

| Flag | Meaning |
|------|---------|
| `--process-mode {real,fake}` | `real` runs the GPU worker (default); `fake` runs a synthetic worker. |
| `-e`, `--load-config-from-env-vars` | Configure from `AIWORKER_*` env vars instead of `bridgeData.yaml`. |
| `-n`, `--worker-name NAME` | Override the worker name. |
| `--amd`, `--amd-gpu` | Enable AMD GPU optimisations. |
| `--config PATH` | `bridgeData.yaml` the config editor reads and writes (default `bridgeData.yaml`). |
| `--no-auto-restart` | Do not relaunch the worker if it crashes. |
| `--attach [HOST:PORT]` | Attach to a running worker host instead of owning the worker; the worker survives this session closing. With no value, attaches to `127.0.0.1:7717`. |
| `--remote-exposed` | Withhold the credential fields from the config editor. Set by the web launcher when it binds a non-loopback address; not normally passed by hand. |
| `--directml N` | DirectML device index. DirectML is currently unavailable, so this has no working backend. |

The launcher-only flags `--terminal` (in-terminal UI), `--headless` (no UI; runs the foreground worker
via `horde-bridge`), and `--host HOST` (bind the served dashboard, unauthenticated) are handled by the
`horde-worker` script before this program runs. In the default browser mode on a machine with no
graphical display, the launcher falls back to the in-terminal UI automatically.

## `horde-worker-web` (served dashboard / host control)

Serves the browser dashboard, spawning a background worker host if one is not already running. The root URL
serves the complete terminal dashboard; `/native` serves a dependency-free responsive overview with selected
live/session stats and start, graceful-stop, pause/resume, and Horde-maintenance controls. The command also
carries two control options that act on an already-running host and exit without starting anything:

| Flag | Meaning |
|------|---------|
| `--status` | Report whether a worker host is running here and whether its worker is working, then exit (non-zero when nothing is running). |
| `--stop` | Ask a running host to drain in-flight jobs and stop the worker and host cleanly, then exit. |
| `--host-port N` | Worker-host socket port the commands target. Without the flag, `$HORDE_WORKER_HOST_PORT` is used, then the default `7717`. |
| `--host HOST`, `--port N` | Bind address/port of the web server itself. Each falls back to `$HORDE_WORKER_WEB_HOST` / `$HORDE_WORKER_WEB_PORT`, then `dashboard_web_host` / `dashboard_web_port` in `bridgeData.yaml`, then `127.0.0.1:8000`. |

Binding anything but loopback serves an **unauthenticated** dashboard: anyone who can reach the port can
start and stop the worker, change every setting, and read its logs. The launcher prints that warning and
tells each dashboard session to withhold the API key and Civitai token fields, which is the only
protection it adds. See
[What a network-bound dashboard exposes](../how-to/use-the-dashboard.md#what-a-network-bound-dashboard-exposes).

On Windows the worker host also shows a **system-tray icon** while it runs, with *Open dashboard* and
*Exit now (stop the worker first)* actions. The host remembers the launcher's resolved web port, so *Open dashboard*
does not assume 8000 when `--port`, the environment, or bridge data selected another one. A detached
or orphaned worker therefore stays visible and stoppable. See
[Keep the worker running when you close the dashboard](../how-to/use-the-dashboard.md#keep-the-worker-running-when-you-close-the-dashboard).

## `horde-duty-report`

Analyze GPU duty-cycle loss across worker sessions. By default, the command reads retained stats JSONL
from `.horde_worker_regen/stats/` when files are present, because those samples carry structured worker
state and finalized-job phase metrics. If no stats session is available, or when you pass a log path, it
falls back to the legacy epoch-aware `bridge.log` parser. `horde-log duty` runs the same report with the same
options.

| Flag / argument | Meaning |
|-----------------|---------|
| `LOG` | Optional `bridge.log` path for the legacy log parser. Supplying this makes the command log-focused unless `--stats` is also supplied. |
| `--stats DIR` | Analyze `stats-v*.jsonl` and `stats-v*.jsonl.gz` files from a stats directory. Rotated files with the same filename stamp are grouped into one session. |
| `--logs PATH` | Log file or directory containing `bridge.log`. Beside `--stats` it adds each session's clearance-hold digest and idle attribution; alone it selects log-only analysis. |
| `--footprints PATH` | Learned VRAM footprint store to list sampling prices from. Defaults to the bundle's `config/vram_footprints.json` when `--stats` sits in a support bundle, else the worker's `.horde_worker_regen/vram_footprints.json`; a missing default store prints nothing. |
| `--last` | Only report the newest stats session or log epoch. |
| `--json` | Emit the machine-readable report schema. Stats-backed JSON uses `SessionDutyReport.to_dict()`. |

Examples:

```bash
horde-duty-report
horde-duty-report --stats .horde_worker_regen/stats --last
horde-duty-report --logs logs --json
horde-duty-report logs/bridge.log
horde-log duty --stats <bundle>/stats --logs <bundle>/logs --last
```

The stats-backed report separates idle loss from partial-utilization loss, reports popped-job `inference_queue_wait` before inference starts, reports sampled `inference_dispatch_gap` when queued inference work has no active inference process, and keeps `unknown` as a first-class bucket when older files lack enough state to attribute a sampled interval.

Four more views read what a support bundle carries:

- **Sampling concurrency.** `jobs in progress at once` counts jobs in progress, and a staged job counts there
  while its lane waits. `image processes sampling at once` counts the inference lanes in `INFERENCE_STARTING`
  per stats sample (0, 1, 2+). The zero share is split by what the lanes were doing with a job in progress
  (preparing to sample, loading a model, other) or with none. Each row below the limit is broken down by the
  sample's `dispatch_hold_bucket` and by whether jobs were queued.
- **Clearance holds.** With `--logs`, the `bridge.log` epoch the stats session belongs to (matched by the stats
  filename's start stamp, which is the worker host's clock) is read for `Clearance held for process` lines.
  Per model it prints the hold count, the median VRAM needed, available and reserved for other jobs,
  and how many holds took each set of steps to free VRAM (`none` included). The worker logs a hold once per process and
  decision, so the count is distinct holds, not held time.
- **Idle attribution.** With `--logs`, every idle card-second of the same epoch is placed on the timeline of the
  job that ended it. A stats sample with no inference lane in `INFERENCE_STARTING` is idle and stands for the
  gap to the next sample, at most 5 s. Consecutive idle samples form a stretch, which ends at the first sample
  where a lane samples; the job most recently dispatched on that lane (`Starting inference for job`) owns the
  stretch. Each idle second lands in one segment of that job's timeline: `pre-dispatch` (named from the sample:
  no local work and why, a rung-2 residency, lanes starting, a preloading lane, or the dispatch hold bucket),
  `arbiter` (dispatch to the `cleared process N into its load-and-sample window` grant, or `no sample grant
  line` when none was logged, marked `unpriced after timeout` when the lane entered its denoise loop without a
  recorded grant), or `child load after grant` (grant to first step, split by the `from dispatch to first step`
  kind, `cold` or `resident`, and by model). A stretch that reaches the epoch's end is `window end`, named from
  its samples. Every line carries its seconds split by context: a rung-2 residency spell, a rung-1
  dominant-model run, or neither. The header gives the idle share and the idle seconds above the 90% card-busy
  target, so the biggest segment is where that gap is. The log's host clock is aligned to the stats' epoch
  seconds from the dispatch stamps both record; a session whose epoch has no dispatch of a recorded job, or
  whose log has no epoch near its start, prints no section.
- **Learned footprints.** Each sampling key in the footprint store, tagged `[all]` for the pooled key or with
  its job class (`[b4]`, `[b1 hires]`), and each learned post-processing chain, tagged with its operations, with
  the two terms admission prices it from: the raise-only watermark and the margined measured figure net of the context charge (`n/a` until the
  key's window holds enough jobs), beside the `sample_activation` watermark for the same band and the
  resident figures of the baseline's checkpoints. The text report prints figures only; the JSON output
  carries the sampling concurrency, the clearance holds and the idle attribution (`idle_attribution`: totals,
  per-segment seconds and every bucket; the per-stretch rows are left out), not the footprints.

## `horde-stats`

Operate on retained stats JSONL files under `.horde_worker_regen/stats/`. The same implementation is
importable from `horde_worker_regen.stats_operations` for the dashboard or other Python callers. By default,
both operations skip the newest stats file so a running worker's active export file is not rewritten; pass
`--include-latest` only when the worker is stopped or you intentionally want to touch it.

| Subcommand | Meaning |
|------------|---------|
| `compress` | Gzip-compress existing uncompressed `stats-v*.jsonl` files, except the latest by default. |
| `downsample SECONDS` | Rewrite stats files so `stats_sample` events are at least `SECONDS` apart. `job_completed` and unknown events are preserved. |

Examples:

```bash
horde-stats compress
horde-stats downsample 15
horde-stats --stats-dir /path/to/.horde_worker_regen/stats downsample 60 --include-latest
```

## `run_worker` (headless)

| Flag | Meaning |
|------|---------|
| `-v` | Increase console verbosity. Repeatable (`-vvv`). |
| `--no-logging` | Disable console logging. |
| `-e`, `--load-config-from-env-vars` | Load config only from environment variables (useful in containers). |
| `--amd`, `--amd-gpu` | Enable AMD GPU optimisations. |
| `-n`, `--worker-name NAME` | Override the worker name (for running multiple workers on one machine). |
| `--directml N` | Enable DirectML on the given device index (currently unavailable). |
| `--gpu-device-indices INDICES` | The cards this worker uses, as comma-separated indices in PCI-bus order (`0` or `0,2`). Overrides `gpu_device_indices` in the config. With neither set, the worker uses every card it detects. |

## `horde-benchmark`

Progressive worker benchmarking. Subcommands:

| Subcommand | Purpose |
|------------|---------|
| `run` | Run the benchmark: test jobs of increasing difficulty on one worker, then suggested settings for `bridgeData.yaml`. |
| `plan` | Show what each probe needs and whether it would run or be skipped. No worker is started. |
| `download` | Download the models the selected tiers need, so the timed benchmark is not slowed by downloading mid-run. |
| `corpus-preflight` | Check that this machine can produce admissible pricing-corpus rows, and print the fix for each problem found. |
| `pricing-corpus` | Run the pricing corpus, the fixed set of jobs whose stats records are used to fit a pricing model. |
| `soak` | Run one mix of jobs continuously for a fixed time and score it, to compare performance before and after a change. |
| `report OUT_DIR` | Re-render the markdown report from an existing output directory. |
| `monitor OUT_DIR` | Follow a run's `progress.jsonl`, live or as a replay. |
| `live` | Generate load against a live AI Horde API (not implemented yet). |

### `run`: operator assessment

<a id="run-bounded-operator-assessment"></a>

The default real-mode run starts one normal worker connected to a local server implementing the
required Horde endpoints. It uses the normal startup, admission, safety, upload and submission
paths. Results retain generation ids and worker counters throughout the run; cold loads and
recoveries count as evidence. The shared harness remains available through `--exhaustive` for
developer checks and through fake/dry-run mode.

The default run finishes selected representative checks without an overall time limit. The work
limit is derived from those checks, two increases per capacity axis and combined-settings
confirmation. Concurrency checks offer two jobs per requested thread at twice the baseline step
count. The worker keeps one model on one lane, so a concurrency check assigns the other lane a
second installed model and alternates jobs between them. The companion model's family need not be
selected for the run. With a single installed model, the check is reported as not applicable and
does not reserve jobs. The thread search stops at the number of installed models, since each lane
needs its own model. After the family baselines, the controller tests the first capacity increases
before feature checks. Further capacity searches follow those feature checks.

A concurrency check runs its jobs twice. The first run is one lane under the operator's
sampling-lease settings. The second is the requested thread count under the candidate configuration,
which enables the sampling lease at one slot so one lane samples while the others load, encode and
stage their next job. An operator whose configuration already enables the lease keeps their
configured slot count for the second run. Both runs share the check's budget and both sets of jobs
belong to its evidence. The combined-settings confirmation runs only the thread count under the
lease settings its thread check ranked. Its result does not depend on the wall-clock gain, and its
lane-equivalent references come from the thread check's single-lane jobs of the same model and
shape.

The combined-settings confirmation is priced before it runs. The largest passed resolution and batch are
forecast together with the worker's own sampling-VRAM estimate, one job per lane, against the card's total
minus the arbiter's admission margin. A joint the worker could not admit is reduced along the passed rungs,
resolution first and then batch, and the report notes each reduction. Resolution goes first because batch
jobs are a large share of production pops while jobs near the largest resolutions are rare. Without this, a
joint of two limits that each passed alone can price above the card, wait out the clearance timeout and
sample unpriced.
A joint on two or more lanes that is unconfirmed only for a thread reason (no overlap, a lane-equivalent
throughput below one lane, or an unmeasured or noise-level wall gain) is run once more at one thread with the
same shape, so its batch, resolution and queue size still confirm. The work limit reserves that follow-up for
every multi-lane joint.

The two runs are compared by throughput: image-job megapixelsteps divided by elapsed time from the
earliest pop to the latest finalization. This includes time spent loading, encoding, checking safety
and submitting results. When any job lacks either timestamp, the run uses the sampling windows'
edges instead. The gain is the ratio of throughput at the tested thread count to single-lane
throughput. It must exceed one and reach one plus the noise margin, the median absolute deviation of
the single-lane jobs' pop-to-finalize times over their median. The report and the progress line
state the gain and the jobs per minute of both runs for the tested shape. A gain within the margin
is inconclusive with `wall_gain_below_noise`. A check whose jobs did not report the timestamps or
the megapixelsteps is inconclusive with `throughput_unmeasured`. A check with an image job that waited
out the clearance lease, or sampled without a grant, is inconclusive with `clearance_timeout`.

Confirmation also requires observed denoise overlap to reach the slot count permitted by the second
run's lease. Lanes beyond the slots stage the next job and are reported as such. With the operator's
lease at two or more slots a check whose sampling does not overlap repeats its second run once.

Lane-equivalent throughput measures combined sampling speed relative to one lane. Divide each job's
steps by the model's single-lane reference rate, sum the resulting durations, then divide by the
time covered by the second run's sampling windows. When two or more windows overlap, a result below
1.0 prevents confirmation. Partly unloaded weights can cause this slowdown. That check is
inconclusive with `throughput_below_single_lane` and is not repeated. References come from first-run
jobs of the same model and shape. The reference is the median rate of those that sampled alone, or
the fastest rate observed when none did. The report and progress line state the result and each
model's rate beside its single-lane rate.

A confirmed thread count measured under the lease candidate is suggested together with
`gpu_sampling_lease_enabled: true` and `gpu_sampling_lease_slots`. A count measured under the
operator's own lease settings leaves both keys out, so Apply does not change them.

Optional retries retain the work reserved for joint confirmation, including other cards and models.
Batch images belong to one image job. `--seconds` and `--job-budget` set optional explicit limits.
Slow cards can continue while loading or sampling progresses. Lost communication and unchanged work
are bounded separately using the worker's configured loading/inference timeouts. Repeated identical
heartbeats do not reset the work-progress clock. A worker that sends no update for 30 seconds ends
the run with that stop reason. A worker that exits ends it with the last error line from its startup
log.

Installed checkpoints from the selected family can replace the catalog's canonical checkpoint.
Selection respects global model exclusions and exclusions on every participating card, including
meta rules such as `all sdxl` and `top N`. Exclusions on unselected cards do not constrain
selection. Apply preserves those rules. If every installed representative is excluded, the run does
not start a worker. The report records the actual checkpoint and card. Representative feature checks
and success-dependent batch, thread and resolution checks replace blanket scenario repetitions. Each
result is classified as a measured success or failure, a prediction, or unknown. Predictions are
extrapolated from measurements. They do not count as additional observations or confidence
estimates. Shared service work retains its ordinary placement.

Thread checks require the sampling overlap their lease permits, reported separately from configured
threads, and a wall-clock gain over one lane for the same jobs. New lanes must become ready before
work is offered. Existing performance modes are retained and included in confirmed settings. A check
whose settings equal the previous check's, with no issued job still awaiting its submit, keeps the
running worker and only confirms its lanes are ready. The report states the total time the checks
spent applying settings beside the total sampling time.

Features the worker reports as withheld for missing preparation are left unknown before any job is
offered, so an ineligible feature cannot consume the remaining assessment time waiting for a pop.

Only jointly confirmed, enforceable operating profiles can be applied. Different model limits on one
card are not merged into an untested combination. The export does not set `models_to_load`, globally
or per card. The tested checkpoints represent their families, so Apply and the exported YAML keep
the operator's model lists. The report's "Model guidance" section, the console summary and the TUI's
Models panel carry model advice derived from the evidence. Each card's summary lists the families
that passed a measured baseline, failed it, were only estimated or were not tested, with the GPU
name and total VRAM when the hardware probe reported them. The guidance suggests less commonly
offered models from a passing family to help cover gaps in availability. The offline reference
does not include worker counts, so the advice does not give numerical estimates.

The storage notes state whether the model directories are on rotational or solid-state storage (read
from sysfs on Linux and `Get-PhysicalDisk` on Windows, otherwise undetermined) and, on rotational
storage, how many models stay resident at the resolved inference-process count. The large families
are the ladder's `HUGE_TIERS`. They can earn more kudos per job, and the guidance states whether
each one passed, failed or was not selected and can be selected for a rerun. Processor and
controlnet variants outside the representative checks remain unconfirmed, and broad feature flags
stay off in the conservative export. Review the report before enabling additional features. Without
a confirmed configuration, Apply is disabled.

The plain-workload export explicitly disables LoRA, img2img, painting, ControlNet and
post-processing globally and, when exported, per card. Single-card runs use global settings unless
`--per-card` is set. Apply removes only conflicting fields from that card's existing overrides,
preserving unrelated policies and other cards. The cleanup is retained for restoring known-good
settings. The exported YAML includes comments listing the existing card fields to remove before
merging it manually, including the painting alias. Copying only the global values leaves card
overrides active.

A joint run that required an inference retry, degraded dispatch or host commit retry cannot confirm
settings, even if it ultimately submitted successfully. Operator reports older than schema 7 remain
readable but must be rerun before applying their recommendations.

Alchemy results describe shared-worker service capacity. The image card is the companion generation's
card; a form's reported lane affinity is separate, and an unreported execution device stays unknown.

#### Assessment controls and results

The dashboard's Advanced options use the same assessment settings as the CLI:

| Control | Behavior |
| --- | --- |
| Time limit (min) | Limits the entire assessment, including startup. Leave blank to finish selected checks. |
| Per-card settings | Includes settings for the tested card even in a single-GPU run, equivalent to `--per-card`. Multi-card results always use per-card settings. |
| Confirm combined settings | Tests proposed settings together before enabling Apply suggested config. Turning it off is equivalent to `--no-validate`. |

A concurrency result reports both the single-lane and multiple-lane jobs-per-minute measurements.
If a run replaced a worker process or required an inference retry, the result explains why the
settings could not be confirmed. Incomplete results identify missing final records or submissions,
missing card identity, jobs that ran on another card, or missing device peak-memory measurements.
A timeout or lost worker connection retains its specific stop reason.

Model-loading results include the first observed load and repeat loads. These durations include
deserialization and may use the OS cache. They do not measure physical disk throughput. Storage
advice identifies the detected media type and remains conditional when detection fails. The advice
does not change settings.

#### Exhaustive developer catalog

`run --exhaustive` uses the shared capability-probe engine. Each probe defines a workload, pass/fail
criteria and prerequisite capabilities. The engine runs probes in dependency order on one warm
worker, reducing repeated startup and model loading. See [GPU duty
cycle](../explanation/duty-cycle.md#per-probe-timing-warmup-versus-inference) for the effect of
startup time on isolated cold probes.

Before each probe, the worker applies the probe's bridge-data settings over the session's base
configuration. Settings from the previous probe do not carry over. This keeps each result tied to
the configuration tested.

The warm driver also asks idle safety and post-processing processes to unload optional models from
RAM, then waits for acknowledgments. The probe's warm-up loads the models needed for measurement. If
warm-up does not finish within the budget, the probe times out. The driver does not start a measured
run while warm-up forms may still be running.

Before the next probe, the driver drops unfinished alchemy forms from a timed-out probe and discards
any late results for those forms. If a process does not acknowledge RAM cleanup within 30 seconds,
the next probe times out without running. The result's `exit_reason` begins with
`boundary_cleanup_timeout` and lists the processes that did not respond. The run then continues.

A **capability** is a `(tier, kind, magnitude)` triple, such as "SD1.5 can run batch size 4". A
probe's verdict is `PROVEN`, `DISPROVEN`, `SKIPPED` or `CRASHED`. The engine skips probes with
unproven prerequisites or resource requirements the machine cannot meet. After the catalog checks,
it tests the suggested settings under sustained load for each passing tier. Failed soaks lower the
recommendation.

The report lists confirmed capabilities separately from suggested bridgeData. Every suggested value
includes the reason it was enabled, disabled or limited. A consistency check rejects settings that
lack a supporting pass.

Key `run` options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--tiers` | `sd15,sdxl` | Comma-separated tiers (`sd15`, `sdxl`, `flux`, `qwen`, `zimage`). `flux`, `qwen` and `zimage` are very large, so they run only when listed, and are skipped when the machine cannot run them. |
| `--process-mode {fake,dry_run,real}` | `real` | `real` runs jobs on the GPU. `fake` and `dry_run` test the benchmark itself and run no inference. |
| `--out PATH` | `benchmark_results/<timestamp>` | Output directory (`report.json`, `report.md`, `progress.jsonl`). |
| `--seconds SECONDS` | none | Optional overall time limit, including startup and model loading. |
| `--job-budget N` | derived | Optional limit on image jobs and alchemy forms across all selected cards. |
| `--per-card` | off | Include per-card settings when testing one card. Multi-card results use per-card settings automatically. |
| `--exhaustive` | off | Use the shared developer harness, full catalog and per-tier validation soaks. |
| `--jobs-per-level N` | `4` | Jobs per exhaustive probe. |
| `--probe-timeout SECONDS` | `900` | Per-probe timeout for the exhaustive engine. The operator assessment uses progress-aware waits. |
| `--only SLUG` | none | Run one probe by capability slug plus its minimal prerequisite closure; an unknown slug is an error. |
| `--no-validate` | off | Skip joint confirmation; the report remains useful but produces no applicable configuration. With `--exhaustive`, skip the per-tier soaks. |
| `--strict-duty` | off | With `--exhaustive`, fail a soak whose GPU duty cycle is under the 90% target. Without this flag a miss is only reported. |
| `--exclude-capability KIND` | none | Skip one capability kind, in addition to anything the `--no-*` flags skip. Repeatable. The choices are the `CapabilityKind` values, e.g. `post_processing`, `controlnet`, `alchemy_concurrent`. |
| `--no-concurrency` / `--no-features` / `--no-alchemy` / `--include-downloads` | none | Coarse stage selection (drop a whole group of probes; downloads are opt-in). |

```bash
# Prove sd15 + sdxl on this GPU and write the report:
horde-benchmark run --tiers sd15,sdxl

# Check one capability and its family anchor within the shared budget:
horde-benchmark run --only sd15-controlnet --no-validate
```

#### Exhaustive catalog coverage

The exhaustive catalog tests the workflows supported by hordelib. Classic **controlnet**
(canny/depth/openpose preprocessors) is tested on SD1.5 only. The **qr_code** workflow, gated by
`allow_sdxl_controlnet`, is tested on SD1.5 and SDXL.

Post-processing tests every upscaler and face-fixer this installation can serve at 512², 1024² and a
VRAM-derived maximum. Models must be recognized by the SDK and resolved by the local model
reference. The reference includes canonical on-disk records and, for categories with beta opt-in,
the PRIMARY's pending queue. The log reports how many models came from each source, so coverage that
depends on access to the pending queue is visible. Models absent from both sources are excluded
because they would fault during execution.

Alchemy checks cover the CLIP lane (caption/interrogation/NSFW on the safety process) and the graph
lane (upscalers/face-fixers/strip-background on the post-processing process). A separate check runs
alchemy alongside image generation. Graph checks have separate warm probes for ordinary forms, the
memory-intensive `4xNomos2_hq_dat2` upscaler (`sd15-alchemy_graph-1`) and face-fixers
(`sd15-alchemy_graph-2`). This isolates memory failures and prevents one probe's resident models
from affecting the others.

Heavy tiers (`flux`/`qwen`/`zimage`) require explicit selection and are skipped when the machine
cannot hold them or the checkpoint is absent. A tier whose model is only in the pending reference
needs `HORDE_MODEL_REFERENCE_PRIMARY_API_URL`. The benchmark sets the beta opt-in automatically.

Every suggested value carries a **provenance basis** so you can tell a setting that is off because it
was *tested and failed* from one that is off only because its probe was *skipped* (never tested),
*not in this run*, or *held back* for VRAM headroom or an unstable soak. The basis is printed under the
completion line, written into the report's "Why each value" table, and shown beside each value in the
dashboard. A built-in consistency check flags (and never silently ships) any recommendation that would
enable a capability on anything weaker than a real pass.

### `plan`: preview requirements before you run

The exhaustive catalog uses the same scenarios on every machine. Hardware and available resources
determine which probes run. The default operator assessment can substitute installed checkpoints
from the same family and adapt checks to earlier results. `plan` makes that decision visible up
front, without starting a worker: it builds the exhaustive catalog, then prints one row per probe
with its estimated VRAM, the disk it needs free, whether it needs network or a CivitAI token, and
the predicted verdict (`RUN`, or `SKIP` with the reason) against the detected hardware.

```bash
# What would run if I benchmarked just sd15 and sdxl on this box?
horde-benchmark plan --tiers sd15,sdxl

# Machine-readable rows (the TUI's "Preview plan" button uses this):
horde-benchmark plan --tiers flux --json
```

`plan` accepts the same selection flags as `run` (`--tiers`, `--process-mode`, `--no-concurrency`,
`--no-features`, `--no-alchemy`, `--exclude-capability`, `--include-downloads`, `--force`), so the
preview identifies category requirements. The operator assessment chooses checks based on earlier
results and may leave some categories unresolved. The same plan table is printed at the top of every
`run` (emitted on the progress channel as a `RampPlanned` event), so `monitor` and the dashboard
show it too. Fake-mode plans skip device discovery because fake runs do not use or gate on GPU
resources; real-mode plans still probe the machine for their fit verdicts. Pass `--force` to see
probes that do not fit (or lack a token) reported as `RUN` instead of `SKIP`.

### `pricing-corpus`: measure what each payload axis costs

A cost model can only price an axis it has seen vary, so this subcommand runs a workload that sweeps
one axis at a time against a per-model anchor: steps, resolution, batch, cfg, sampler, schedule, source
processing, post-processing, controlnet, hires fix, LoRA cache state, textual inversions, and
base-model cold loads. The expansion is deterministic (same tier, same jobs, same order, on every
machine), and every job's cell is recorded in a **definition artifact** written next to the session's
stats stream, which is what lets a later fit label each stats record with the axis values that produced
it.

| Flag | Default | Meaning |
|------|---------|---------|
| `--tier {smoke,standard,census,heavy}` | `smoke` | `standard` is the marginal-cost fit set (multi-hour); `census` covers every value of every categorical axis the kudos manifest encodes (about four hours at the default budget); `heavy` measures the model families a small card cannot hold, on a machine that can, beside the SD1.5 and SDXL anchors it shares with every other machine; `smoke` is a short subset that proves the corpus runs. |
| `--machine ID` | — | The measuring machine's id (`<owner>-<gpu>` by convention, lowercase and dash-separated). Required for every tier but `smoke`; stamped into the definition artifact with the machine's GPU, driver and version facts so the rows stay attributable. |
| `--keep-learned-footprints` | off | Keep the worker's learned VRAM footprint store for this run. By default a corpus sets the store aside (kept beside it under a stamped name) and starts empty, so one tier's measured peaks never price the next; keeping it trades that predictability for a warm start. |
| `--skip-preflight` | off | Start without the preflight below. The run still needs everything the preflight checks; skipping only removes the early refusal. |
| `--job-budget N` | `950` | Census tier only: jobs the run may spend, warmup included. The vocabulary sweeps are fixed; a smaller budget shrinks only the jointly-varied conflation block. |
| `--model NAME` | every heavy model | Heavy tier only: restrict the tier to these models (repeatable), for a machine that holds only some of them. |
| `--emit-definition PATH` | — | Write the definition artifact here instead of next to the session stats. |
| `--dry-list` | off | Print the ordered cell ids and exit; no worker is started. |
| `--lora-version-id ID` | pinned set | Override a pinned CivitAI LoRA *version* id for the LoRA cells (repeatable; the standard and census tiers need five). |
| `--ti-name NAME` | pinned id | Override the pinned textual-inversion reference for the TI cell. |
| `--no-lora-eviction` | off | Skip evicting the pinned LoRAs before the run; any still cached turns its miss cell into a hit measurement. |
| `--out DIR` | `benchmark_results/<ts>` | Output directory for the controller log. |
| `--timeout SECONDS` | `21600` | Overall run timeout. |

```bash
# See what the full corpus would run, without starting a worker:
horde-benchmark pricing-corpus --tier standard --dry-list

# Short end-to-end check on this GPU:
horde-benchmark pricing-corpus --tier smoke

# The real fit run (the pinned LoRA/TI references apply unless overridden):
horde-benchmark pricing-corpus --tier standard

# Full vocabulary coverage, for a model that must price values the standard tier never runs:
horde-benchmark pricing-corpus --tier census --machine alice-l40s

# The same coverage in about two and a half hours, with a smaller conflation block:
horde-benchmark pricing-corpus --tier census --machine alice-l40s --job-budget 600

# The heavy families, on a card that holds them:
horde-benchmark pricing-corpus --tier heavy --machine alice-l40s
```

Every tier but `smoke` runs `corpus-preflight` first and exits with code 2 if it fails. A run given a
machine id names its definition `pricing-corpus-<tier>-<machine>-<UTC stamp>.json` and, when it finishes,
copies that definition and the run's stats parts into `benchmark_results/corpus-<machine>-<tier>-<UTC
stamp>/` with a `bundle.json` manifest: the directory a contributor sends back. The how-to
[Run a pricing corpus on your machine](../how-to/run-a-pricing-corpus-on-another-machine.md) walks a
developer through the whole sequence.

#### The `heavy` tier

Per heavy model: the native anchor, a second trajectory length where the model's reference record allows
one, a smaller and a wide resolution, a two-image batch, and a cold cell. Jobs are ordered model by
model, one block per replicate with the cold cell first, because heavy checkpoints do not share host RAM
and every switch between them is a disk read that would otherwise land in whichever cell followed it.
The SDXL and SD1.5 anchors run in the same session, warm and cold, so the machine's speed can be
calibrated to the reference machine on cells both have measured.

### `corpus-preflight`: prove a machine is ready before spending hours

```bash
horde-benchmark corpus-preflight --tier census --machine alice-l40s
```

Prints one row per check (machine id, CUDA, VRAM, hordelib and its kudos manifest, the ComfyUI pins,
cache home, the tier's models on disk, free disk, CivitAI token, no live worker), each with the exact
command or edit that clears it, and exits non-zero while anything is outstanding. A row is `OK`, `WARN`
(reported with its remedy, does not stop the run) or `FAIL`. `--model` narrows the heavy tier's model
check the same way it narrows the run.

The `vram` check judges the card only against what this machine has already measured. hordelib streams a
model's components on and off the card, so a checkpoint larger than the card still runs and its size
predicts nothing; the check reads the learned footprint store
(`.horde_worker_regen/vram_footprints.json`) for each of the tier's models and reports the highest peak
measured for it against the card. The row never fails: a peak above the card is paging on a demand-paging
driver or a failed job, and only a run tells which, so it is a `WARN` naming the model. Where a model has
never been measured here the row is also `WARN`: it names the unverified models and the largest checkpoint
on disk, and asks for the tier's one-model smoke first rather than refusing a run that may well work. The `civitai token`
check is skipped for the `heavy` tier, which carries no LoRA cells; the run skips the pre-run LoRA
eviction there for the same reason.

#### The `census` tier

The standard tier moves each axis a few steps so a marginal cost can be read off it. The census answers
a different question: does the corpus contain *every value* a request can name? Its vocabularies are read
from the kudos feature manifest that hordelib ships, not from lists restated in the worker, so a value the
manifest gains is a value the census sweeps. Each value gets its own replicated cell against a fixed
anchor (every sampler, every schedule, every post-processor, every control type, every source-processing
mode, hires fix, the LoRA count levels, textual inversions), and the remaining budget goes to a
conflation block whose axes vary jointly, so the marginals can be checked against conditions where
several of them move at once. The combination space is sampled, not covered: full pairwise over these
vocabularies is orders of magnitude past any sitting.

The sampler sweep is the one axis whose values get more than a cell each: every sampler runs at two
trajectory lengths (15 and 30 steps). A sampler's cost is part per-trajectory-step and part per-job, and a
sampler observed at a single step count leaves the two confounded, so a fit can only price it by borrowing
another sampler's slope. Both levels sit on the steps sweep's own grid and span the range requests
mostly fall in. The extra cells come out of the fixed job budget rather than adding to it: the
conflation block is sized to whatever the sweeps leave, so the tier still runs about four hours.

A value the census cannot run is named in the definition artifact's `census.exclusions`, with the reason,
alongside a per-axis `census.coverage` count and the projected job count and runtime. Absence is always a
declared decision: `k_dpm_adaptive` is out because it picks its own step count, the baselines with no
model in the corpus are out, and `dpmpp_3m_sde` is not paired with the `normal` schedule because the
solver diverges there and hordelib substitutes another schedule underneath the request. The census run
advertises the whole capability surface it exercises (controlnet, extended controlnet, img2img,
inpainting, LoRA, post-processing), because a capability the bridge does not advertise narrows what the
worker accepts and would leave the definition's jobs unpaired.

Requires a hordelib that ships `hordelib.kudos_training.manifest`; without it the tier refuses to build
rather than falling back to a hand-written vocabulary. Sweeping every control type also fetches every
annotator checkpoint on first use (several gigabytes, one time, on top of the run itself).

The LoRA and TI cells reference pinned, real CivitAI version ids (near the ad-hoc LoRA size cap, so the
miss cells measure the worst legitimate per-job download). Run preparation must evict the five pinned
LoRAs through the LoRA model manager's `delete_lora` so the first use of each is a genuine cache miss;
the run itself re-downloads them, leaving the cache as it was. The ordering is constrained,
not free, because three of the costs are properties of a job's *position* rather than its payload: a
cold-load cell is always preceded by a different model, a LoRA reference is always first used by a
cache-miss cell, and one cell deliberately runs three post-processing jobs back to back while every
other post-processing job is spaced apart.

### `soak`: the standalone before-and-after soak

`soak` runs one sustained-traffic mix against the worker for a fixed period and nothing else: no capability
probing, no recommendation, and no contact with the horde. It exists to answer "did this change make the
worker faster", which needs two runs that differ only in the setting under test. Live traffic cannot give
that (the model mix and job sizes vary between any two hours), so the mix is generated locally from the
same templates the `run` soak uses.

Two mixes are available. `production_replay` (the default) reproduces the shape of a measured production
pop stream: SDXL-family dominant with an SD1.5 and a rare Flux minority, batch sizes and job magnitudes
matching the observed distribution, LoRAs on a realistic minority of pops, post-processing on about a
third. `lora_storm` is the adversarial download-pressure mix (roughly two thirds of pops carry LoRAs).
Both are described under [LoRA backpressure](../explanation/performance_and_backpressure.md).

The run measures the **operator's own configuration**: the throughput-shaping fields of `bridgeData.yaml`
(`max_threads`, `queue_size`, `high_performance_mode`, `moderate_performance_mode`, `safety_on_gpu`,
`unload_models_from_vram_often`, `legacy_comfy_vram_unload`, `whole_card_exclusive_residency`,
`gpu_sampling_lease_enabled`, `dedicated_post_processing`, `enable_pipeline_disaggregation`,
`gpu_device_indices`, `vram_to_leave_free`, `ram_to_leave_free`) are carried over, and only those the file
actually sets. Everything else is the harness's, because it is derived from the workload rather than
chosen: the model set and `max_power` come from the mix (an operator `max_power` below what a template
needs would make the job ineligible and fault it at dispatch), and the `allow_*` capability flags come from
the templates' own features. `--override key=value` layers on top, and is the A/B knob.

| Flag | Default | Meaning |
|------|---------|---------|
| `--mix {production_replay,lora_storm}` | `production_replay` | Which traffic mix to sustain. |
| `--minutes N` | `20` | How long to sustain it. Short runs are noisy; prefer 20 minutes or more per arm. |
| `--process-mode {fake,dry_run,real}` | `real` | Only `real` measures anything; `fake`/`dry_run` exercise the plumbing and export no stats. |
| `--out DIR` | `benchmark_results/<ts>` | Run directory for `controller.log` and the `soak.json` manifest. |
| `--label TEXT` | — | A label for this run, such as `baseline` or `unload-on`. It appears in the manifest and the closing summary. |
| `--bridge-data PATH` | `bridgeData.yaml` | Worker config to copy the throughput settings from. |
| `--ignore-bridge-data` | off | Copy nothing from the worker config. The run uses the benchmark defaults plus any `--override`. |
| `--override KEY=VALUE` | — | Set one config value for this run, over the worker config. Repeatable. |
| `--no-loras` | off | Remove every LoRA and textual-inversion reference from the mix. |
| `--shared-lora REFERENCE` | synthetic pool | A downloadable LoRA reference for the pool the mix reuses across jobs (the cache-hit pool). Repeatable. Give at least 3. |
| `--unique-lora REFERENCE` | synthetic pool | A downloadable LoRA reference for the pool the mix uses once each (the download-pressure pool). Repeatable. Give at least 8. |

```bash
# Arm A: the worker as configured today.
horde-benchmark soak --minutes 20 --label baseline --no-loras

# Arm B: the same traffic with one setting changed.
horde-benchmark soak --minutes 20 --label unload-on --no-loras --override legacy_comfy_vram_unload=true

# The adversarial download mix, with real LoRA references so the prefetch path is exercised:
horde-benchmark soak --mix lora_storm --minutes 30 \
  --shared-lora GlowingRunesAI --shared-lora ... --unique-lora ...
```

**The LoRA caveat.** Both mixes ship *synthetic* default LoRA and textual-inversion reference names, which
exist only so a fake-mode run has something to mint. In a real-mode run they cannot resolve, so every
LoRA-bearing job would measure the prefetch failure path instead of production cadence. Either supply
resolvable references (`--shared-lora` / `--unique-lora`, at least 3 and 8 respectively) or pass
`--no-loras`, which strips LoRA *and* textual-inversion references from every template while leaving the
model, resolution, batch and post-processing shape untouched. `--no-loras` makes the mix slightly cheaper
than production (no LoRA load, no download pressure); that is a fair comparison between two arms run the
same way, but not a measurement of production LoRA cost. A real-mode run that does neither is warned about
rather than silently accepted.

**What it prints.** Real-mode runs export stats
(`.horde_worker_regen/stats/stats-v*.jsonl`) stamped with the mix's scenario id, so `horde-duty-report` and
`horde-log` can score the session afterwards exactly as they score a live one. At the end the subcommand
names the stats files it wrote, renders the session's duty report (the same one `horde-duty-report --last`
would produce), and closes with the throughput headline: completed jobs, the sampling slot-duty seconds
that are the denoise time those jobs spent on the card, and the top non-sampling slot-duty buckets.
**Kudos is not among them**: nothing priced this work, so an arm is compared on denoise seconds and on
where the empty slot-seconds went. The `soak.json` manifest in the run directory records the label, the
resolved bridge overrides, and the stats session id, which is the join from an arm back to its stats.

## `horde-log`

Post-mortem triage of worker logs. A `bridge.log` is appended across every restart (so one file holds
many worker lifetimes), the per-subprocess logs are separate files, rotations are zipped, and the real
crash cause is usually a traceback in a child's `bridge_inference_<N>_startup.log`. `horde-log` does that
archeology for you: it segments the file into per-launch sessions, stitches the orchestrator log to the
subprocess that actually crashed (and to the `action_ledger.jsonl` when present), and reports what went
wrong. It is pure-stdlib and read-only; point it at a `logs/` directory, a single log file, or a `.zip`
an operator sent you.

| Subcommand | Purpose |
|------------|---------|
| `sessions [PATH]` | List each worker launch in the log with its span, version, end-reason, and peak process recoveries, then each benchmark run from `bridge_harness.log` as a second list. |
| `diagnose [PATH]` | Run the detectors and print ranked findings (root cause + remediation) per session. |
| `timeline [PATH]` | Merged, time-ordered parent + child + ledger event stream for a session. |
| `jobs [PATH]` | Per-job lifecycle table for a session: which lane and card ran each job, and how its wall clock split across pop->dispatch, generation, and finished->submit. |
| `job <ID> [PATH]` | Trace one job across the parent and the inference slot that ran it. |
| `watch [PATH]` | Live-poll the logs and alert when a new warning/critical finding or a rising recovery count appears. |
| `bundle [PATH]` | Build a single redacted `.zip` (logs + diagnosis + config + system/cache info) to send a maintainer. |
| `duty` | The GPU duty report, with the options of [`horde-duty-report`](#horde-duty-report): stats-backed sampling concurrency, the session's clearance holds and idle attribution, and the learned footprint prices. |

`PATH` defaults to `logs/`. `sessions`, `diagnose`, `timeline`, and `jobs` take `--session N` or
`--last` to select a worker session, `--benchmark N` or `--last-benchmark` to select a benchmark run, and
`--json` for machine-readable output. Worker sessions and benchmark runs are each numbered from #0, and
selectors of both kinds combine (worker sessions first); `timeline` also takes
`--process N`, `--grep RE`, and `--child` (include every line of the child process logs), and `jobs` takes
`--limit N` (`0` for every row; the summary always covers every job whatever the table shows). Each detector recognizes one incident
class (an inference pool crashing on start, repeated process restarts that never give up, GPU OOM, the swallowed
"no images produced" OOM, a run of jobs dropped because their process disappeared, a local queue that
filled and stopped moving, a
model-reference read that faulted a running sample, a pop that named no model, a model that ends every
slot it is loaded onto, and the horde repeatedly refusing this worker's pops) and emits the child's
exception as the root cause where it can. Every session also carries a census of the jobs it faulted,
with each one's model and cause. Every finding id, with its severity and what it means, is catalogued in
[Log findings](log_findings.md); a finding whose remedy needs background prints a `see:` line naming the
docs page that explains the subsystem behind it. See
[Troubleshoot](../how-to/troubleshoot.md#diagnose-a-crash-or-repeated-process-restarts-from-the-logs).

Findings quote the worker's own words wherever the log carries them, and only fall back to listing
candidate causes when it does not: a pop rejection is reported with the horde's verbatim message and
keyed to whether an operator can act on it, a safety stall names the subsystem whose pause/restore lines
cycled the safety process, and a start-up crash reads the child's failure text before suggesting a fix
(a git clone/checkout failure points at the shared ComfyUI environment directory, not at torch).

### `jobs`: where a session's wall clock went

A slow worker is one of three different problems, and the remedies are opposite. `horde-log jobs` splits
every job's wall clock into the three segments the worker logs separately, so the question is answered
from the log rather than guessed:

| Segment | Bounded by | A large median means |
|---------|------------|----------------------|
| `pop->disp` | the pop line to `Starting inference for job` | a scheduling problem: the job waited for a lane |
| `generate` | the child's own reported generation time | a GPU/config problem: it sampled slowly |
| `->submit` | `Inference finished for job` to the submit line | a pipeline-balance problem: safety, upload or submit backed up |

Under the table it prints the median and p90 of each segment, the card count and worker-wide intake
budget, the **sampling-concurrency histogram** (the share of time 0, 1, 2 ... cards held an in-flight
job, which is the fleet's real parallelism rather than its lane count), the status-print census (median
queue depth, idle lanes, and pending jobs whose model was already resident on an idle lane), and the
line-skip census. On a multi-card host this is the view that separates "each card is slow" from "only
two cards are doing anything":

```text
1000 jobs; wait segments (seconds):
  pre_inference   n=992   median=    87.8 p90=   154.3
  inference       n=1002  median=    15.9 p90=    37.6
  post_inference  n=969   median=     4.9 p90=    13.6

Cards driven: 8 | worker-wide intake budget: 16
Sampling concurrency (time-weighted mean cards busy: 2.50)
  0 cards 2% | 1 card 28% | 2 cards 25% | 3 cards 20% | 4 cards 13% | 5 cards 8% | 6 cards 2%
```

The same parse backs the `multi_card_dispatch_serialization`, `model_churn`, `lane_placement`, and
`parent_loop_stall` findings and the wait attribution in `slow_generation_drop_spiral`, so `diagnose`
and `jobs` never disagree about where a session's time went.

### Rotated logs are stitched back into their run

The parent log rotates at a size cap mid-run, so the active `bridge.log` usually holds only the tail of
the run in it. Naming a directory reads every rotation it finds; naming a single log file walks back
through the rotated predecessors in the same directory and folds in each one whose range hands directly
over to the next, stopping at the first that does not (that is a different launch). Zipped rotations are
read in place, without unpacking anything next to the logs.

Either way the report says what it read: `sessions` prints a `Rotation:` line and the session-summary
finding carries the same disclosure, naming the archives that were included and, when the walk stopped,
the older ones that were left out. A session whose launch line is still missing is reported as starting
*at or before* its first surviving line, with the span and duration marked as bounds.

The dashboard exposes the same `diagnose` analysis without a shell: the **Diagnostics** tab
runs the detectors over `logs/` and renders the ranked findings, with a selector to pick which worker
session to view. It reads the log files directly, so it **works whether or not the worker is running**.
Each finding separates the *diagnosis* (what went wrong) from the *suggested fix*, and a timing line
dates the analysis, shows the current time, and flips to a clear **stale** warning once the displayed
analysis is more than five minutes old.

A **scope** selector (left of the session/Run cluster) chooses how much history a pass reads:
*Current session* and *Last 3 sessions* read only the live `bridge.log` (fast); *All logs* also
decompresses every rotation. Analysis is **only started by pressing Run analysis**, opening the tab or
changing the scope does not run anything (changing the scope shows a "press Run analysis to apply"
hint), so a slow pass is never triggered just by browsing. The parse and detectors are CPU-bound, so
the work runs in a **separate worker process** (only a lightweight, record-free summary is returned)
and the TUI stays responsive while it runs. It calls the same `diagnose()` facade the CLI does, so the
two never disagree. See [How the diagnostics stay in sync](../explanation/log_diagnostics_contract.md)
for the contract that keeps the detectors, the logs they read, and this tab from drifting apart.

### `bundle`: a redacted archive for a maintainer

`horde-log bundle` collects everything a maintainer needs into one shareable `.zip`: the diagnosis
(`diagnose.txt`), the worker's logs, retained stats JSONL files, the action ledger, the redacted config,
and a system/cache report (OS, worker version, RAM/disk, on-disk model listing). **Every text artifact is
scrubbed before it is written**. The horde `api_key` and CivitAI token always, and (by default)
personal identifiers (home path, username, worker name) too. The command prints how many occurrences it
redacted and reminds you to skim the result before sending; redaction is best-effort, not a guarantee.

Anything too large for the per-file budget keeps its **most recent** end, logs and JSONL artifacts alike,
so every artifact in one bundle covers the same window. A trimmed file says so on its first line: a
`[... truncated to the most recent N MB ...]` note in a log, and the same wording as a JSON object under
`_bundle_truncation` in a JSONL artifact, which is cut on a record boundary. When a trimmed log no longer
contains a session's launch line, `sessions.txt` and `diagnose.txt` report that session as starting *at or
before* its first surviving line, with the span and duration marked as lower bounds (`<=`/`>=`).

```bash
# The usual: bundle the current logs into horde_support_<timestamp>.zip
horde-log bundle
```

| Flag | Meaning |
|------|---------|
| `--out FILE.zip` | Output path (default `horde_support_<timestamp>.zip`). |
| `--last` / `--session N` | Diagnose only the most recent / a specific worker session (the logs are still included). |
| `--last-benchmark` / `--benchmark N` | Diagnose only the most recent / a specific benchmark run from `bridge_harness.log`. Combines with the worker selectors. |
| `--full-logs` | Include rotation archives, every retained stats file, and do not tail-cap large artifacts (a much larger bundle). By default only the active logs are bundled and analysed, stats files last written before the earliest bundled session are skipped; oversized plain logs are read from their tail and bundled already-trimmed, and the active `bridge.log` already spans many sessions. When the earliest bundled session begins mid-run (the active `bridge.log` is a size roll-over continuation), up to three of its abutting predecessor rotations and the active log itself are shipped uncapped and the stats window reaches back to where they start, so a long run's earlier hours are not lost. The manifest's `scope` records `stats_files` and `stats_export_enabled`, and the README says why a bundle has no `stats/` member (the export is off by default). The learned VRAM footprint store (`.horde_worker_regen/vram_footprints.json`) ships as `config/vram_footprints.json`. |
| `--no-cache-inventory` | Skip the on-disk model listing. |
| `--probe-gpu` | Run the GPU probe for the system-info block (slower; the logs already record the GPUs). |
| `--keep-identifiers` | Do not scrub home path / username / worker name (the keys are still redacted). |
| `--config PATH` | Worker config to redact and source secrets/cache from (default `bridgeData.yaml`). |

The same generator is available in the dashboard: the **Logs** tab has a **Support bundle** button
(also `Ctrl+B`) that writes the zip and reports the path.

### `replay-fixture`: a recorded session as simulator input

`horde-log replay-fixture <path> (--session N | --last) <out_dir>` turns one worker session's `job_completed`
stats rows and the learned VRAM footprint store into the two files the dispatch-world session replay reads
(`session_jobs.jsonl`, one row per image job with its pop offset, parameters, stage offsets, phase seconds,
load sums, post-processing window, kudos and measured peaks; `footprints.json`, the store with every key's
platform rewritten to `<platform>`). `path` is a bundle root, a bundle's `logs/` directory or a worker's logs
directory. `N` is the worker session number `horde-log sessions` prints, and the stats group is the one whose
stamp sits within two minutes of that session's first log line. `--stats DIR` and `--footprints FILE` override
the discovery. A session captured mid-run has no matching stats stamp and is reported as such.

```bash
horde-log replay-fixture logs/external_logs/<box>/horde_support_<stamp> --session 3 /tmp/replay
```

## Environment variables

### Install and runtime

| Variable | Effect |
|----------|--------|
| `HORDE_WORKER_DIR` | Install location for the one-line installer. |
| `HORDE_WORKER_BACKEND` | Force a PyTorch build: `cu126`, `cu130`, `cu132`, `rocm`, `rocm-windows`, or `cpu` (default: detected). |
| `HORDE_WORKER_NO_SHORTCUTS` | Skip creating Desktop/Start Menu shortcuts. |
| `HORDE_WORKER_NO_LAUNCH` | Skip auto-launching the dashboard after install. |
| `HORDE_WORKER_ENABLE_LONG_PATHS` | Opt in to Windows system-wide long-path support (changes an HKLM setting; needs administrator). |
| `HORDE_WORKER_ROCM_TORCH` | Override the Linux ROCm torch version/spec installed by the ad-hoc ROCm path. |
| `HORDE_WORKER_ROCM_INDEX` | Override the Linux ROCm wheel index used by the ad-hoc ROCm path. |
| `HORDE_WORKER_ROCM_WINDOWS_VERSION` | Override the AMD ROCm Windows wheel version (default `7.2.1`). |
| `HORDE_WORKER_ROCM_WINDOWS_BASE` | Override the AMD ROCm Windows wheel URL base. |
| `CUDA_VISIBLE_DEVICES` | Pin a worker to a specific GPU (see [Run multiple GPUs](../how-to/run-multiple-gpus.md)). |

### Behaviour

| Variable | Effect |
|----------|--------|
| `HORDE_WORKER_NO_UPDATE_CHECK` | Disable the dashboard's background release check. |
| `AIWORKER_LIMITED_CONSOLE_MESSAGES` | Cap console verbosity at level 2. |
| `AIWORKER_REGEN_ENABLE_TELEMETRY` | Opt in to OpenTelemetry tracing (off by default; see [Telemetry](../explanation/telemetry.md)). |

Worker configuration itself can be supplied through `AIWORKER_*` variables instead of
`bridgeData.yaml` when you pass `-e`; this is the container path described in
[Run headless](../how-to/run-headless.md#configure-from-environment-variables-containers).
