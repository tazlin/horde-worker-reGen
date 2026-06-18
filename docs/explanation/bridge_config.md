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
[`RuntimeConfig`][horde_worker_regen.process_management.runtime_config.RuntimeConfig].

During normal operation, the `_bridge_data_loop` asyncio task re-reads the file
every second and calls `RuntimeConfig.update()` if it changed. Components read
`RuntimeConfig.bridge_data` whenever they need the current values; no locks, no
notifications. The hot-reload is best-effort; there is no atomicity guarantee
across multiple reads.

If configuration was provided via environment variables, the `_bridge_data_loop`
is **not started** and the config is effectively immutable. This reflects
the typical use case of env var config (e.g. Docker) where dynamic updates are
less common and file-watching would add unnecessary complexity for little benefit.

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
| `extra_slow_worker`         | Disables all performance **and** memory modes (`high_performance_mode`, `moderate_performance_mode`, `high_memory_mode`, `very_high_memory_mode`); forces `queue_size=0`, `max_threads=1`, and `preload_timeout` to at least `120` |

These flags also affect safety-check timeouts and how aggressively the scheduler
preloads models.

### Memory modes

`high_memory_mode` and `very_high_memory_mode` signal that the worker has
abundant RAM/VRAM. They relax constraints on concurrent model preloading and
model eviction.

### VRAM and RAM budget

Because several inference processes share one GPU, residency-favoring settings
(`high_memory_mode`, a large model set, deep `queue_size`) can over-commit the
device and crash with an out-of-memory error. The VRAM/RAM budget guards against
this by gating model preloads on **measured** free VRAM and available RAM rather
than process counts. See
[The VRAM and RAM budget](performance_and_backpressure.md#the-vram-and-ram-budget)
for how it works.

| Field                | Default | Effect                                                                                                                                                                  |
| -------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enable_vram_budget` | `true`  | Gate preloads and concurrent dispatch on measured free VRAM/RAM, and evict idle resident models under pressure. Disable only to restore the prior availability-only behavior (not recommended on a shared/consumer GPU). |
| `vram_reserve_mb`    | `2048`  | Free VRAM (MB) kept in reserve on top of a job's estimated peak. Covers transient spikes such as tiled VAE decode. Larger trades throughput for safety.                |
| `ram_reserve_mb`     | `4096`  | Available system RAM (MB) kept in reserve so resident-in-RAM models do not force the OS to page to disk.                                                               |

When the budget overrides `high_memory_mode` residency to reclaim VRAM/RAM under
pressure, it logs prominently; frequent eviction/reload churn is a signal to
reduce the model set or disable `high_memory_mode`.

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
- [`RuntimeConfig`][horde_worker_regen.process_management.runtime_config.RuntimeConfig]
