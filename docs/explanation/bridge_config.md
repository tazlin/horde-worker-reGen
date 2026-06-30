# Bridge Configuration

- [Bridge Configuration](#bridge-configuration)
    - [How configuration loads](#how-configuration-loads)
    - [Critical sections](#critical-sections)
        - [Image models and model stickiness](#image-models-and-model-stickiness)
        - [Queue sizing](#queue-sizing)
        - [Performance modes](#performance-modes)
        - [Memory modes](#memory-modes)
        - [Very fast disk mode](#very-fast-disk-mode)
        - [Custom models](#custom-models)
        - [Post-processing overlap](#post-processing-overlap)
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

### Performance modes

Three boolean flags interact to determine timings and parallelism:

| Flag                        | Effect                                                                                                                                                                                                                             |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `high_performance_mode`     | Cuts `process_timeout` to 1/3 of default; allows post-processing overlap                                                                                                                                                           |
| `moderate_performance_mode` | Cuts `process_timeout` to 1/2 of default; allows post-processing overlap                                                                                                                                                           |
| `extra_slow_worker`         | Disables both performance modes (`high_performance_mode`, `moderate_performance_mode`); forces `queue_size=0`, `max_threads=1`, and `preload_timeout` to at least `120`                                                             |

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
| `vram_reserve_mb`    | `2048`  | Free VRAM (MB) kept in reserve on top of a job's estimated peak. Covers transient spikes such as tiled VAE decode. Larger trades throughput for safety.                |
| `ram_reserve_mb`     | `4096`  | Available system RAM (MB) kept in reserve so resident-in-RAM models do not force the OS to page to disk.                                                               |
| `ram_pressure_pause_percent` | `90.0` | Absolute whole-host RAM danger floor. At/above this usage percentage the worker degrades (refuses new model loads, sheds idle resident processes, pauses pops) until RAM recovers, rather than loading weights through an out-of-RAM host and being OS OOM-killed. |
| `ram_pressure_min_free_mb`   | `1024` | Free-RAM (MB) companion floor: the worker also degrades below this many MB free. The effective floor is `max((100 - ram_pressure_pause_percent)% of total RAM, this)`, so the percentage protects large-RAM hosts and the absolute floor protects small ones. |
| `post_processing_budget_reserve_enabled` | `true` | Subtract the predicted post-processing peak of in-flight jobs from the free VRAM the budget gates new dispatch/overlap against, so a freshly-released slot is not handed VRAM an in-flight upscaler is about to claim. Self-scales to zero when nothing is post-processing. |
| `post_processing_active_reclaim_enabled` | `true` | Proactively reclaim cross-process VRAM before a job's *own* post-processing peak lands (see below). |
| `post_processing_fault_breaker_enabled` | `true` | Disable post-processing on this worker after repeated post-processing over-commit faults, so it stops feeding the horde's forced-maintenance spiral (see below). |
| `post_processing_fault_threshold` | `4` | The breaker trips when *more than* this many post-processing over-commit faults occur within the window (tolerates 4, trips on the 5th). |
| `post_processing_fault_window_seconds` | `1800` | Rolling window (seconds) over which `post_processing_fault_threshold` is counted. |

The `ram_pressure_*` floor is distinct from `ram_reserve_mb`: the reserve is a
*marginal* per-job admission check, while the pressure floor is the *absolute*
whole-host guard that drives the degrade response (shed footprint, throttle
pops) the moment available RAM crosses it, independent of any one job's cost. A
SIGKill (`exitcode -9`) reaped while RAM is below this floor is classified as an
OS OOM kill rather than a slot crash, so it is not mislabelled "crashed or hung"
and does not quarantine an otherwise-healthy slot.

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

### Post-processing overlap

When `post_process_job_overlap` is enabled (and a performance mode is active),
inference processes can start a new job while the previous job's post-processing
(image encoding, etc.) is still running. This is a throughput optimization for
fast GPUs. (Distinct from `allow_post_processing`, which controls whether the
worker advertises post-processing capability to the API at pop time.)

### Post-processing VRAM over-commit

A job's upscaler/face-fixer peak lands *after* sampling and can be far larger
than its sampling footprint: a 4x upscale on an SDXL image needs roughly 8.5 GB
at peak. The preload budget deliberately admits a job on its sampling cost alone
(folding the transient post-processing spike into placement would misroute
ordinary upscale jobs onto the heavy-head path). On a contended card (warm
sibling models plus several process contexts already resident), that peak can
allocate into near-zero free VRAM and tile-thrash silently until the
post-process watchdog reaps the slot, faulting the job. ComfyUI's own
`free_memory` can only release *this* process's weights; the sibling models and
contexts that fill the card are cross-process and only the orchestrator can
reclaim them.

Two protections close that gap, both on by default. The decisive one is an
**overlap gate** on the imminent peak, part of
`post_processing_budget_reserve_enabled`. The over-commit usually emerges
*mid-flight*: while one job is still sampling, `post_process_job_overlap`
pre-stages a second concurrent sample, and by the time the first job reaches its
upscaler the card already holds both. A dispatch-time check cannot see that --
at each job's dispatch neither peak is live yet. So the reserve also charges the
*imminent* post-processing peak of any in-flight job that is still sampling
against the overlap/pre-staging cap: a second concurrent sample is withheld when
the card is already owed a large upscale peak. It self-scales to zero when
nothing in flight will post-process, so ordinary overlap is unaffected.

`post_processing_active_reclaim_enabled` is the complement for the non-overlap
saturated case. At dispatch the scheduler sizes the dispatching job's own
post-processing peak against the *effective* headroom: the measured free VRAM less
the room in-flight sibling work has committed or will imminently commit (the same
not-yet-realised reserve the overlap gate subtracts), so an optimistic or stale
free reading cannot make the peak look like it fits a card that is about to fill.
When the peak overflows that headroom it frees cross-process room, choosing for
what it can actually reclaim. If an idle sibling holds an evictable model it frees
that first -- even when freeing the dispatching job's own weights would nominally
cover the peak -- because on a contended card that sibling model is the room the
upscaler needs and the child's in-process `free_memory` cannot reach it. Only when
no reclaimable sibling holds room does it defer to that in-child own-weights free,
then to stopping an idle context on the contended card. When none of those can free
room *now*, it asks one more question before faulting: can the card host the peak at
all? If the peak fits the card drained to this job's process alone and a sibling is
mid-inference whose completion will free the room, the dispatch is *held* (the job
keeps its head-of-queue position) until that room appears, rather than faulting a job
the card can serve moments from now. This matters most on the large-card overlap
case: a 24 GB card running several processes can have two or more siblings mid-upscale
at once, whose committed peaks pull the effective free below zero, yet a fresh ~5 GB
peak still fits the card the instant a sibling finishes, so waiting beats faulting.
The hold is self-bounding and wedge-safe: it only ever spans a window the recovery
supervisor already exempts as inference-in-progress, and the moment no sibling is left
in flight to free room the plan re-evaluates and faults instead of parking forever.

Only a peak that cannot fit even the drained card (or one with no in-flight sibling to
wait on, such as the tiny single-process card) faults gracefully so the horde reissues
the job, rather than dispatching it into a guaranteed stall. That fault is terminal
(non-retryable): a local retry would only re-dispatch into the same unchanged,
still-overflowing card, so the job is left for the horde to reissue elsewhere. The
whole mechanism is evidence-gated: with the peak unknown or free VRAM unmeasured it
does nothing, and on a roomy card where the peak already fits it is a no-op. Each
decision is logged at debug with the peak, the effective free, and the chosen action,
and a held dispatch logs its hold and surfaces in the dispatch-stall diagnostic, so a
stall the reclaim declined to prevent (or chose to wait out) leaves a trace.

`post_processing_fault_breaker_enabled` is the self-protective backstop. If
post-processing peaks keep failing to host (more than
`post_processing_fault_threshold` over-commit faults within
`post_processing_fault_window_seconds`), the worker stops advertising
post-processing so the horde stops sending it upscale/face-fix jobs it cannot
host, ending the fault-to-forced-maintenance spiral, and logs an advisory to
downgrade settings. The suppression is session-latched: the over-commit is
structural, so it clears only on restart. See
[Resilience and recovery](resilience_and_recovery.md) for how it sits alongside
the other self-protective throttles.

### Alchemy

`alchemist: true` opts the worker into **alchemy** jobs (`/v2/interrogate/pop`) in
addition to image generation. Alchemy work reuses the existing child processes
rather than spawning its own: graph forms (upscalers, face-fixers, background
removal) run on the inference processes; CLIP/caption forms (interrogation, NSFW,
caption) run on the safety process. The forms the worker offers come from the
`forms` field; `forms` is left empty by default, which advertises the default set
(caption, NSFW, interrogation, post-process).

| Field                       | Default | Effect                                                                                                     |
| --------------------------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `alchemist`                 | `false` | Enables alchemy popping/processing at all.                                                                  |
| `alchemy_caption_enabled`   | `false` | Opts into caption forms. Off by default because captioning loads BLIP into the safety process (extra RAM/VRAM). |
| `alchemy_allow_concurrent`  | `true`  | Allow alchemy to run alongside image jobs. When `false`, alchemy is strict **backfill**: it pops only when the image queue is empty. |
| `alchemy_max_concurrency`   | `1`     | Maximum alchemy forms in flight (dispatched, awaiting result) at once. Raise it on machines with spare compute/VRAM. |
| `alchemy_vram_headroom_mb`  | `2000`  | Minimum free VRAM (MB) required before popping a graph form in concurrent mode; acts as the floor for the headroom estimator, which raises the bar toward the observed median cost of recent forms. |

Image jobs always win contention for a process; in concurrent mode, alchemy only
uses a process lane no waiting image job needs **and** only when the VRAM-headroom
gate passes. See
[Performance and Backpressure → Alchemy backpressure](performance_and_backpressure.md#alchemy-backpressure).

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
