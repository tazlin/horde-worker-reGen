# Bridge Configuration

- [Bridge Configuration](#bridge-configuration)
    - [How configuration loads](#how-configuration-loads)
    - [Config tab save checks and presets](#config-tab-save-checks-and-presets)
    - [Critical sections](#critical-sections)
        - [Image models and model stickiness](#image-models-and-model-stickiness)
        - [Queue sizing](#queue-sizing)
        - [Performance modes](#performance-modes)
        - [Memory modes](#memory-modes)
        - [Very fast disk mode](#very-fast-disk-mode)
        - [Custom models](#custom-models)
        - [Post-processing overlap](#post-processing-overlap)
        - [Dormant experimental flags](#dormant-experimental-flags)
        - [Alchemy](#alchemy)
    - [Configuration flow at a glance](#configuration-flow-at-a-glance)
    - [See also](#see-also)

`bridgeData.yaml` is the single configuration file that controls every aspect of
the worker's behavior. This page explains what each section does and how
configuration flows through the system at runtime.

## How configuration loads

At startup, `BridgeDataLoader` reads `bridgeData.yaml` (or
`bridgeData_template.yaml`) and validates it against the pydantic
[`reGenBridgeData`][horde_worker_regen.bridge_data.data_model.reGenBridgeData]
model. The validated model is stored in
[`RuntimeConfig`][horde_worker_regen.process_management.config.runtime_config.RuntimeConfig].

During normal operation, [`BridgeDataReloader`][horde_worker_regen.process_management.config.bridge_data_reloader.BridgeDataReloader]
runs the `bridge_data_loop` asyncio task, re-reads the file every second, and
applies valid changes through the process manager. The apply step updates
`RuntimeConfig`, then components read `RuntimeConfig.bridge_data` whenever they
need the current values; no locks, no notifications. The hot-reload is best-effort; there is no atomicity guarantee
across multiple reads.

A file reload is applied live, including the download subsystem: the pause /
bandwidth / parallelism controls and the download-gating flags
(`allow_lora`, `allow_controlnet`, `allow_sdxl_controlnet`,
`allow_post_processing`, `nsfw`, `purge_loras_on_download`) are forwarded to the
running download process, which re-arms its one-shot auxiliary pass when a
category is newly enabled. None of these require a worker or download-process
restart. The set of fields that genuinely cannot change live (worker identity,
GPU selection, and other structural choices) is small; those need a restart.

If configuration was provided via environment variables, the `bridge_data_loop`
is **not started** and the config is **restart-only**: change the environment
variables and restart the worker to apply them. This reflects the typical use
case of env var config (e.g. Docker), where dynamic updates are less common and
file-watching would add complexity for little benefit. Operators who want live
config changes should run from `bridgeData.yaml` rather than environment variables.

## Config tab save checks and presets

The dashboard's **Config** tab is the preferred editor for operator-facing
settings. It keeps advanced and restart-locked fields away from the everyday
pages, preserves comments and untouched YAML keys, and shows a live process-count
preview for the throughput levers (`max_threads`, `queue_size`, and the model
set). A plain **Save** writes only fields that actually changed; **Save + restart
worker** is only needed for fields marked with the restart marker.

Before writing the file, the editor runs import-light interlock checks so
configuration mistakes are caught even when no worker process is running. Errors
block the save and jump to the relevant tab; warnings allow the save but are
shown in the status line. The guarded combinations include inpainting without
img2img, SDXL ControlNet without ControlNet, post-processing enabled while the
post-processing lane is off, LoRA or `TOP`/`ALL` model rules without a CivitAI
token, both worker roles disabled, extra-slow mode combined with incompatible
throughput settings, and exact model names present in both load and skip lists.

The **Apply preset** action offers built-in hardware starting points (for example
4090/64 GB SDXL balanced, 4090/64 GB large-model, 2080/32 GB SD1.5-safe, and
midrange 12-16 GB balanced). Applying a preset does not save immediately. The
operator sees every setting the preset would change, each change is individually
checkable, and changes such as enabling LoRA are left unchecked when their
precondition (a CivitAI token) is missing.

## Critical sections

### Image models and model stickiness

`image_models_to_load` is the list of model names the worker will accept jobs
for. `horde_model_stickiness` (0.0-1.0) controls how likely the worker is to
prefer models already loaded in VRAM when popping. Higher stickiness reduces
model-switching overhead on slow disks at the cost of occasionally missing jobs
for other models.

Stickiness only activates when the number of configured models exceeds
`max_inference_processes` (`queue_size + max_threads`) **and** every inference
process already has a model loaded. When it fires, the pop is restricted to
models already loaded on idle processes.

`only_models_on_disk` (default `false`) constrains the served set to models whose
weights are already present. The load rules are resolved as usual (literal names
plus any `TOP n` / `ALL` expansions), then any resolved model that is not on disk
is dropped rather than downloaded. This pins the worker to what you already have
without curating an explicit list, and guarantees a config edit never triggers a
large download. Presence is resolved against the configured weights root
(`cache_home` / `AIWORKER_CACHE_HOME`), the same location the worker downloads to.

### Queue sizing

`queue_size` controls how many jobs the worker will hold in its internal
pipeline at once (pending + in-progress + pending-safety + pending-submit). The
pop gate checks `queue_size + 1 + (max_threads - 1)` as the ceiling; the extra
headroom accounts for the fact that jobs "in progress" briefly occupy two stages
(pending_inference + in_progress).

### Process count

How many inference processes the worker spawns is not a single setting; it is
derived from the concurrency fields, and the result is easy to under- or
over-estimate. Per card the worker runs:

```
inference processes = queue_size + max_threads
```

with these adjustments:

- **Single-model collapse:** a worker offering exactly one concrete model at
  `max_threads: 1` collapses to a single inference process (a `top N`/`all` meta
  command counts as many models, so it does not collapse).
- **Queue cap:** `queue_size` is capped to `3` once `max_threads >= 2`.
- **Extra-slow clamp:** `extra_slow_worker` forces `max_threads` to `1` and
  `queue_size` to `0` first, so the count becomes `1`.
- **Alchemist-only:** a worker with `dreamer: false` (or a CPU-only install)
  runs a single inference process regardless of the above; image concurrency is
  what the process pool exists for.

There is always **one additional safety process**. The formula above is an
**upper bound**: two plan-time caps can lower it, each only ever reducing a card
and never below one context per card. First, a **per-card VRAM-fit cap** bounds
every card (single-GPU included) to the inference contexts its VRAM physically
holds: each planned context's idle CUDA/runtime baseline plus one copy of a single
typical (SDXL-class) working set, within the card total net of the proportional
admission noise buffer. A very large model (Flux, Cascade, Qwen, Z-Image) does not
raise this bound: it never co-samples, and its footprint is paid just-in-time by the
whole-card residency machinery when one actually dispatches, not reserved at spawn,
so the spare contexts stay free to preload the next model. Second, a **shared-RAM
cap** (`cap_card_process_counts`) applies to every host, single-GPU included, and
lowers the count so the worker-wide resident set fits the single system-RAM pool
(each resident context retains system RAM that no card's VRAM offsets; a second card
doubles VRAM, not RAM, and one card's plan can over-commit the pool alone). Both are
spawn-time bounds; the running worker's runtime governor down-regulates live
concurrency further under memory pressure, and the `horde-benchmark` path is exempt
from both caps so it can probe higher concurrency than steady-state operation would
keep. When a cap reduces the configured count the worker logs the arithmetic so the
smaller spawn is explained.

The **Config tab** shows a live estimate of this count under the Throughput
fields, updating as you edit, so the consequence of a concurrency change is
visible before you save.

### Performance modes

Three boolean flags interact to determine timings and parallelism:

| Flag                        | Effect                                                                                                                                                                                                                             |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `high_performance_mode`     | Cuts `process_timeout` to 1/3 of default; allows post-processing overlap                                                                                                                                                           |
| `moderate_performance_mode` | Cuts `process_timeout` to 1/2 of default; allows post-processing overlap                                                                                                                                                           |
| `extra_slow_worker`         | Disables both performance modes (`high_performance_mode`, `moderate_performance_mode`); forces `queue_size=0`, `max_threads=1`, and `preload_timeout` to at least `150`                                                             |

These flags also affect safety-check timeouts and how aggressively the scheduler
preloads models.

### VRAM and RAM budget

Because several inference processes share one GPU, a large model set or a deep
`queue_size` can over-commit the device and crash with an out-of-memory error.
The VRAM/RAM budget guards against this by gating model preloads on **measured**
free VRAM and available RAM rather than process counts. It also decides when to
keep a model resident versus evict it, so the worker no longer needs a manual
"keep models resident" switch. See
[The VRAM and RAM budget](performance_and_backpressure.md#the-vram-and-ram-budget)
for how it works.

| Field                | Default | Effect                                                                                                                                                                  |
| -------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enable_vram_budget` | `true`  | Gate preloads and concurrent dispatch on measured free VRAM/RAM, and evict idle resident models under pressure. Disable only to restore the prior availability-only behavior (not recommended on a shared/consumer GPU). |
| `vram_reserve_mb`    | `2048`  | Co-residency safety margin: free VRAM (MB) kept in reserve on top of a job's estimated peak while a model samples *beside* others. Covers transient spikes such as tiled VAE decode and sizes how many models co-reside and how deep a whole-card teardown goes. It is *not* a hard load-feasibility floor: whether a model's weights can load at all is governed by ComfyUI's own streaming threshold (`minimum_inference_memory`), so a large checkpoint whose weights fit the drained card (e.g. an ~11.5 GB Flux on 16 GB) still loads via whole-card residency even when this margin exceeds the leftover headroom. Larger trades co-resident throughput for safety.                |
| `ram_reserve_mb`     | `4096`  | Available system RAM (MB) kept in reserve so resident-in-RAM models do not force the OS to page to disk.                                                               |
| `ram_pressure_pause_percent` | `85.0` | Absolute whole-host RAM danger floor, evaluated every scheduling tick. At/above this usage percentage the worker degrades (refuses new model loads, sheds idle resident processes, recycles an over-ceiling process, pauses pops) until RAM recovers, rather than loading weights through an out-of-RAM host and being OS OOM-killed. The default leaves ~15% free because a resident process can allocate several GB in a single step. |
| `ram_pressure_min_free_mb`   | `1024` | Free-RAM (MB) companion floor: the worker also degrades below this many MB free. The effective floor is `max((100 - ram_pressure_pause_percent)% of total RAM, this)`, so the percentage protects large-RAM hosts and the absolute floor protects small ones. |
| `ram_per_process_max_mb`     | `18432` | Resident RAM (MB) one inference process may hold before it is a reclaim candidate *while the host is under the danger floor*. Over it, an idle process is recycled immediately and a busy one is drained (fed no new work) then recycled once its job finishes, so a single process's balloon cannot drive the summed footprint into an OS OOM kill. Consulted only under the floor, so a roomy host never recycles. `0` disables. |
| `post_processing_fault_breaker_enabled` | `true` | Disable post-processing on this worker after repeated post-processing over-commit faults, so it stops feeding the horde's forced-maintenance spiral (see below). |
| `post_processing_fault_threshold` | `4` | The breaker trips when *more than* this many post-processing over-commit faults occur within the window (tolerates 4, trips on the 5th). |
| `post_processing_fault_window_seconds` | `1800` | Rolling window (seconds) over which `post_processing_fault_threshold` is counted. |

The `ram_pressure_*` floor is distinct from `ram_reserve_mb`: the reserve is a
*marginal* per-job admission check, while the pressure floor is the *absolute*
whole-host guard that drives the degrade response (shed footprint, recycle an
over-ceiling process, throttle pops) the moment available RAM crosses it,
independent of any one job's cost. Raising `ram_reserve_mb` alone does **not**
prevent a system-RAM OOM kill: it gates only a *new* preload's marginal cost, not
the resident set that a generous residency policy accumulates across processes.
The `ram_per_process_max_mb` ceiling and the danger floor are the levers for that.
A softer, pre-floor pop hold (reusing `ram_reserve_mb` as its approach margin)
stops the popper starting a new job's time-to-live clock once available RAM is
within that margin of the floor, so a job does not age past its ttl on a degraded
worker and get aborted as too slow. A SIGKill (`exitcode -9`) reaped while RAM is
below this floor is classified as an OS OOM kill rather than a slot crash, so it
is not mislabelled "crashed or hung" and does not quarantine an otherwise-healthy
slot.

On a host running several worker roles together (a dreamer plus an alchemist
and/or a scribe), the plan-time process-count sizing additionally reserves RAM for
those co-tenants up front, so the resident image-context count is sized against
what is actually left of the shared pool rather than the whole of it.

When the budget evicts resident models to reclaim VRAM/RAM under pressure, it
logs prominently; frequent eviction/reload churn is a signal to reduce the model
set or the queue depth.

### Very fast disk mode

`very_fast_disk_mode` tells the scheduler that model loading from disk is
essentially instant. When enabled, the scheduler allows more concurrent model
preloads and is less aggressive about keeping models in VRAM.

### Custom models

`custom_models` allows the worker to accept jobs for model names not in the
standard horde model reference. These are added to the pop request alongside the
configured `image_models_to_load`.

### The dedicated post-processing lane

Post-processing (upscalers, face-fixers, background removal) runs on a dedicated
process lane rather than inside the inference processes, controlled by
`dedicated_post_processing` (`auto`/`on`/`off`; `off` also stops the worker
offering post-processing, since the lane is the only place it runs). The lane
keeps the post-processing models resident (no per-job reload), frees the
inference slot the moment sampling and VAE decode finish, and reports its VRAM
telemetry into the same device-free budget view as inference processes. The
budget charges the lane's fixed CUDA context through the normal process-count
forecast and charges each active post-processing job's estimated upscale/face-fix
peak through the shared committed-reserve ledger until its result returns. Under
pressure, the scheduler can evict idle inference models and ask an idle
post-processing lane to unload its modules from VRAM/RAM before starting more
work. A pending chain that fits the card once drained gets the next drain window before a fresh sampler that
would be unable to co-reside with it, and speculative model preloads yield to the same drain window so they
do not recreate the pressure the lane is waiting to clear. Only structurally unhostable chains become
no-image faults. See
[Process lanes and job chaining](process_lanes_and_chaining.md) for the full
picture, including how lane failures are reported as no-image faults rather than
silently submitting raw images for jobs that requested post-processing.

`post_processing_fault_breaker_enabled` is the self-protective backstop. If
post-processing peaks keep failing to host (more than
`post_processing_fault_threshold` over-commit faults within
`post_processing_fault_window_seconds`), the worker stops advertising
post-processing so the horde stops sending it upscale/face-fix jobs it cannot
host, ending the fault-to-forced-maintenance spiral, and logs an advisory to
downgrade settings. The suppression is session-latched: the over-commit is
structural, so it clears only on restart. See
[Resilience and recovery](resilience_and_recovery.md) for how it sits alongside
the other self-protective throttles. The same session-latched suppression is
used when a whole-card model cannot fit beside even the post-processing lane's
bare GPU context; in that case the worker logs an operator warning and the TUI
health view shows post-processing as disabled until restart.

### Dormant experimental flags

`enable_pipeline_disaggregation` is currently accepted for config compatibility
but forced off during `reGenBridgeData` validation. Setting it in
`bridgeData.yaml` or through `AIWORKER_REGEN_ENABLE_PIPELINE_DISAGGREGATION`
does not start the component/VAE lane processes or route jobs through the
disaggregated path. The config editor keeps the field in the Advanced catalog
but hides it until the pipeline is ready to expose again.

### Alchemy

`alchemist: true` opts the worker into **alchemy** jobs (`/v2/interrogate/pop`) in
addition to image generation. Alchemy work reuses the existing child processes
rather than spawning its own: graph forms (upscalers, face-fixers, background
removal) run on the inference processes; CLIP/caption and other text-output forms
(interrogation, NSFW, caption, `vectorize`, `palette`, `describe`, `aesthetic`) run
on the safety process. The forms the worker offers come from the `forms` field;
`forms` is left empty by default, which advertises the default set (caption, NSFW,
interrogation, post-process, and the text-output forms `vectorize`/`palette`/
`describe`/`aesthetic`). A newly-added text-output form is withheld until the
server advertises it, so a worker can ship ahead of the server go-live without
risking a rejected pop.

The text-output forms return their result inline (no R2 upload): `palette` extracts
a dominant-colour list, `describe` bundles a blurhash + perceptual hashes + geometry,
and `aesthetic` returns a LAION 0-10 quality score (reusing the CLIP embedding the
safety process already computes).

| Field                       | Default | Effect                                                                                                     |
| --------------------------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `alchemist`                 | `false` | Enables alchemy popping/processing at all.                                                                  |
| `alchemy_caption_enabled`   | `false` | Opts into caption forms. Off by default because captioning loads BLIP into the safety process (extra RAM/VRAM). |
| `aesthetic_scoring_enabled` | `true`  | Opt into attaching a LAION aesthetic score to **every** image generation as `gen_metadata` (computed for free from the safety pass's CLIP embedding). Independent of offering the `aesthetic` alchemy form. The worker still withholds the score until it detects the server accepts the `aesthetic_score` metadata type (the server rejects a submit carrying an unknown type), so leaving this on is safe ahead of the server's go-live. Set `false` to skip it entirely (and the one-time predictor-weight download). |
| `alchemy_allow_concurrent`  | `true`  | Allow alchemy to run alongside image jobs. When `false`, alchemy is strict **backfill**: it pops only when the image queue is empty. |
| `alchemy_max_concurrency`   | `1`     | Maximum alchemy forms in flight (dispatched, awaiting result) at once. Raise it on machines with spare compute/VRAM. |
| `alchemy_vram_headroom_mb`  | `2000`  | Minimum free VRAM (MB) required before popping a graph form in concurrent mode; acts as the floor for the headroom estimator, which raises the bar toward the observed median cost of recent forms. |

Image jobs always win contention for a process; in concurrent mode, alchemy only
uses a process lane no waiting image job needs **and** only when the VRAM-headroom
gate passes. See
[Performance and Backpressure → Alchemy backpressure](performance_and_backpressure.md#alchemy-backpressure).

### Log retention

The worker's `logs/` directory holds several families of files: the rotated, zipped
`bridge*.log` and `trace*.log` archives (written by the loguru sinks in hordelib),
plus one-per-run `stdout`/`stderr` redirections, `bridge_*_startup.log` crash
backstops, the supervised `bridge_main_console.log`, and `*.faulthandler` dumps. The
loguru sinks keep a bounded *count* of each rotated family, but nothing ages files
out over time or bounds the directory as a whole, so a long-lived install would grow
it without limit.

At each startup the orchestrator runs a single cleanup sweep of `logs/` before any
child process is spawned: it first deletes files older than the age limit, then, if
the directory still exceeds the size budget, deletes the oldest files first until it
fits. The currently active logs are always the newest files, so the age-out never
touches them and the size trim reaches them last. The sweep is best-effort: a file it
cannot delete (an actively-held handle on Windows) is skipped rather than failing.

The sweep is deliberately incapable of removing the wrong file. It only ever considers
a **direct child** of `logs/` (non-recursive, so a nested `logs/remote_support/` tree
is never inspected), only a **regular file** (symlinks are skipped, never followed to a
target elsewhere), and only a file that
[`log_file_registry`][horde_worker_regen.log_file_registry] **positively recognizes** as
a worker log. That registry is the single declared source of truth for the worker's
log-file families (the hordelib `bridge*`/`trace*` sinks, the per-child `stdout`/`stderr`
redirects, the supervisor sink, the startup-crash backstops, and the faulthandler dumps).
It is **fail-closed**: a file the registry does not describe is left untouched, not
guessed at. A CI check introspects the loguru sinks the worker and hordelib actually
register at runtime and asserts every one is described by a registry entry, so the
registry cannot silently drift out of step with the code that writes the logs.

| Field                    | Default | Effect                                                                                             |
| ------------------------ | ------- | -------------------------------------------------------------------------------------------------- |
| `log_purge_max_age_days` | `30`    | Delete any log file older than this many days at startup. `0` disables the age-out.                |
| `log_purge_max_total_gb` | `5`     | After the age-out, if `logs/` still exceeds this many GB, delete oldest-first until it fits. `0` disables the size cap. |

Both limits are independent; set both to `0` to keep all logs indefinitely (only the
loguru per-sink rotation count then applies).

## Configuration flow at a glance

```
bridgeData.yaml ──(BridgeDataLoader)──► reGenBridgeData (pydantic)
                                              │
                    ┌─────────────────────────┘
                    ▼
             RuntimeConfig (hot-reloaded every 1 s)
                    │
     ┌──────────────┼──────────────────┐
     ▼              ▼                  ▼
  JobPopper   InferenceScheduler   JobSubmitter
  (pop gates) (model loading,      (timeouts,
              VRAM management)     kudos tracking)
```

Every sub-manager that needs config simply reads
`self._runtime_config.bridge_data`; there is no config-propagation machinery to
understand.

## See also

- [Architecture](architecture.md): where `RuntimeConfig` fits in the
  shared-state pattern
- [Performance and Backpressure](performance_and_backpressure.md): how config
  fields drive throttling and scheduling
- [`BRIDGE_CONFIG_FILENAME`][horde_worker_regen.consts.BRIDGE_CONFIG_FILENAME]
- [`reGenBridgeData`][horde_worker_regen.bridge_data.data_model.reGenBridgeData]
- [`RuntimeConfig`][horde_worker_regen.process_management.config.runtime_config.RuntimeConfig]
