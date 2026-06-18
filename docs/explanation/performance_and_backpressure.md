# Performance and Backpressure

- [Performance and Backpressure](#performance-and-backpressure)
    - [The pop gauntlet](#the-pop-gauntlet)
    - [Megapixelstep backpressure](#megapixelstep-backpressure)
    - [Model stickiness](#model-stickiness)
    - [Pop-rate throttling](#pop-rate-throttling)
    - [Queue sizing and the hold-back gate](#queue-sizing-and-the-hold-back-gate)
    - [Inference scheduling priorities](#inference-scheduling-priorities)
        - [Performance-model scoring](#performance-model-scoring)
        - [Model affinity (high-throughput regime)](#model-affinity-high-throughput-regime)
        - [The line-skip cache](#the-line-skip-cache)
    - [Model eviction (LRU)](#model-eviction-lru)
    - [The VRAM and RAM budget](#the-vram-and-ram-budget)
    - [Alchemy backpressure](#alchemy-backpressure)
    - [See also](#see-also)

The worker sits between two external systems: the AI Horde API (which can flood
it with jobs) and the GPU (which has finite VRAM and throughput). This page
explains the throttling, scheduling, and backpressure mechanisms that keep the
worker stable under load.

## The pop gauntlet

Before `JobPopper` makes any network call, it runs a series of gates:

1. **Shutdown check**: if `WorkerState.shutting_down`, skip.
2. **Consecutive failure backoff**: if ≥ 3 consecutive failures, pause for
   `CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS` (180 s).
3. **Queue full**: if `queue_size + 1 + (max_threads - 1)` jobs are in the
   pipeline, skip.
4. **Hold-back gate**: if jobs are pending inference but no jobs have been
   completed yet this session, skip. This is a warm-up guard: let the very first
   job complete before pulling more.
5. **Process availability**: at least one safety process and one inference
   process must exist and be healthy.
6. **Megapixelstep backpressure**: if pending megapixelsteps exceed the
   configured threshold, skip (see below).
7. **Pop-rate throttle**: if less than `current_pop_frequency` seconds have
   elapsed since the last pop, skip.

If any gate says "no," the pop is skipped for this cycle. The popper retries on
the next 1-second tick.

Model selection within an allowed pop is further constrained by **on-disk
availability**: `_select_models_for_pop` filters the advertised models down to
those actually present on disk (`ModelAvailability.filter_present`), so the worker
never accepts a job for a model that is still downloading. While availability is
not yet known (no download process, or before its first scan) every configured
model is treated as present. See
[Model Downloads and Availability](model_downloads.md#model-availability-and-the-pop-gate).

## Megapixelstep backpressure

"Megapixelsteps" are a rough measure of GPU work:
`width × height × steps / 1,000,000`. The job tracker sums the megapixelsteps of
pending jobs; when that sum exceeds a threshold, `PopThrottler` pauses popping
until the backlog drains. The threshold is **not** a config field; it is
derived from the active performance mode: `15` (normal), `60`
(`moderate_performance_mode`), or `80` (`high_performance_mode`). How long
popping pauses also scales with the backlog and performance mode.

This prevents the worker from accepting a large number of high-resolution jobs
that would take hours to complete, starving smaller jobs behind them.

## Model stickiness

When the worker has more configured models than inference processes, every new
job might require unloading one model and loading another. On slow disks, model
loading can take minutes, during which the GPU is idle.

Model stickiness (`horde_model_stickiness` in bridge config, 0.0-1.0) is the
probability that the pop request will only ask for models currently loaded in
VRAM. A sticky worker stays "locked" to its loaded models, trading job variety
for throughput. Stickiness automatically disengages when no jobs are available
for the loaded models.

## Pop-rate throttling

`PopThrottler` manages two pop frequencies:

| Frequency               | Used when         | Default |
| ----------------------- | ----------------- | ------- |
| `default_pop_frequency` | Normal operation  | 1.0 s   |
| `error_pop_frequency`   | After a pop error | 5.0 s   |

On a successful pop (even if no jobs were returned), the frequency resets to
default. On an API error, the frequency slows to `error_pop_frequency` to avoid
hammering the API.

When no jobs are available, `PopThrottler` tracks cumulative idle time. If idle
time grows too large, the worker logs a diagnostic.

## Queue sizing and the hold-back gate

The hold-back gate (`_is_queue_full` logic) deserves special attention:

```python
queue_size + 1 + (max_threads - 1)
```

The `+ 1` accounts for the fact that one job can be "in flight" during the pop
itself (not yet recorded in the tracker). The `+ (max_threads - 1)` accounts for
jobs that are in `INFERENCE_IN_PROGRESS` but also still counted in
`PENDING_INFERENCE` (the dual-presence rule). Without this headroom, the queue
would appear full when it still has capacity.

## Inference scheduling priorities

`InferenceScheduler.run_scheduling_cycle` runs every 200 ms and makes decisions
in this order:

1. **Preload models**: for the first pending job whose model isn't loaded, pick
   a free process and send `PRELOAD_MODEL`, subject to the
   [VRAM and RAM budget](#the-vram-and-ram-budget), which defers the preload (and
   reclaims idle resident models) when the device cannot absorb the new model.
2. **Peek ahead**: call `get_next_job_and_process(information_only=True)` to
   decide whether to block on a heavy model or batch.
3. **Blocking rules**: defer launch if:
    - `keep_single_inference` is active (batch mode, VRAM-heavy model,
      ControlNet-XL, post-processing overlap).
    - A batch/heavy-workflow job is waiting but not enough jobs have accumulated
      for the batch.
4. **Start inference**: send `START_INFERENCE` with the job payload. A
   per-dispatch *expected sampling time* is computed and attached to the process
   (see [Performance-model scoring](#performance-model-scoring) below) so a
   running job can later be graded as "slow".
5. **Unload models**: evict models not needed by the upcoming queue
   (LRU-informed), subject to **model affinity** (below).

### Performance-model scoring

The worker has no way to tell a genuinely stuck job from a merely large one
without a reference for how fast a job of that shape *should* sample. The
[`PerformanceModel`][horde_worker_regen.process_management.performance_model.PerformanceModel]
supplies that reference as expected sampling iterations-per-second, keyed by a
coarse `JobSignature` (baseline + resolution/steps/batch buckets +
controlnet/hires flags). It is seeded from a prior benchmark `report.json` and
then **self-calibrates** from every completed job's observed it/s, persisting the
learned table to `.horde_worker_regen/perf_model.json`. Nothing is enforced until
a signature has a seed or enough samples, so a cold start raises no false alarms.
The scheduler attaches `expected_sampling_seconds` to each dispatch; this module
only *measures*, the grading happens elsewhere (see
[Resilience and Recovery](resilience_and_recovery.md)).

### Model affinity (high-throughput regime)

When the worker serves at least as many inference processes as distinct models,
every model can have a permanent home process and never needs reloading. The
default preload-target picker, however, can fall back to displacing a still-wanted
model when no empty process is free, forcing a needless disk reload.
[`model_affinity.compute_protected_processes`][horde_worker_regen.process_management.model_affinity]
identifies the processes holding the *last remaining copy* of a still-wanted
model and marks them off-limits as displacement targets. Surplus copies and
processes holding no-longer-wanted models stay displaceable, so spare capacity is
still usable. It is pure and table-testable, with no scheduler imports.

### The line-skip cache

`get_next_job_and_process` is called twice per cycle: once to peek and once to
launch. When a small job "skips" ahead of a larger job blocked on a LoRA
download, the skip decision is cached in `_pending_line_skip` so the second call
agrees with the first. Without this cache, the launch call could pick a
different job, causing the block decision to be wasted.

## Model eviction (LRU)

When the scheduler needs to free VRAM for a new model, it uses the `LRUCache` to
decide which loaded model to evict. The LRU tracks which models were most
recently used. Models that are `IN_USE` are never evicted. Models needed by
upcoming jobs in the queue are also protected.

Eviction happens in two stages:

1. **Unload from VRAM**: the model stays in system RAM, enabling fast reload.
2. **Unload from RAM**: the model is fully evicted; next use requires a disk
   read.

`very_fast_disk_mode` makes the scheduler more aggressive about RAM eviction
(since reloading from disk is cheap) and less aggressive about VRAM eviction
(since keeping models in VRAM is the real win).

## The VRAM and RAM budget

Each inference process loads models into the **same** GPU independently. Without a
shared accountant, their combined resident footprint can exceed device VRAM,
producing an out-of-memory crash (and, for system RAM, paging to disk that
collapses throughput). The eviction rules above are *count*-based. They assume
that if the working set fits the process count it fits the device, which only
holds for SD1.5-class weights. The budget makes the worker decide on **measured**
resources instead.

[`VramBudget` and `RamBudget`][horde_worker_regen.process_management.resource_budget]
predict a job's peak VRAM and RAM cost from hordelib's per-job burden estimate
([`estimate_job_burden`][hordelib.feature_impact.estimate_job_burden], the same
estimate the benchmark pre-flight trusts) and compare it against:

- **measured device-wide free VRAM**: the conservative minimum across inference
  processes' memory reports
  ([`ProcessMap.get_free_vram_mb`][horde_worker_regen.process_management.process_map.ProcessMap.get_free_vram_mb]),
  plus
- **measured available system RAM**: read live from the parent via psutil.

A job is admitted only when free VRAM and available RAM each cover the prediction
plus a reserve (`vram_reserve_mb`, default 2048; `ram_reserve_mb`, default 4096).
The reserve absorbs transient spikes the steady-state estimate misses, most
notably tiled VAE decode (the phase that produced the observed live OOM).

When a resource does not fit, the scheduler **defers** the preload for that cycle
and starts **reclaiming** the resource from idle resident models, overriding the
count-based residency protection (`under_pressure`) so an idle copy is evicted
even in the affinity regime but never an in-progress or next-up model. This is
the "auto-throttle" behavior: it overrides the residency that `high_memory_mode`
would otherwise hold, and logs prominently when it does so.

Cold start (no VRAM telemetry yet) and a missing burden estimate both **admit**,
so the budget never wedges a worker that has not yet reported memory. Set
`enable_vram_budget: false` to restore the prior availability-only behavior (not
recommended on a shared or consumer GPU).

> The prediction is the conservative hordelib estimate, not a learned per-job
> measurement. The only measurement the worker has (per-process VRAM high-water)
> is device-wide (it reflects *every* resident model, not one job's marginal
> cost), so feeding it back would over-throttle a multi-model worker. A true
> marginal per-job measurement is a hordelib-side follow-up.

To keep measurements fresh, inference processes emit an interval-driven memory
report (every `_memory_report_interval`, 5 s) in addition to the event-driven
reports at model load/unload, and a dead process's stale VRAM figure is cleared on
recovery so it cannot be counted as either used or free.

## Alchemy backpressure

When `alchemist: true`, `AlchemyCoordinator` runs its own pop loop (≈ every 1 s,
popping at most every 4 s) independent of the image pop gauntlet. Because alchemy
shares the inference and safety processes with image work, it has its own gating
so it never starves image jobs:

- **In-flight cap**: at most `alchemy_max_concurrency` forms may be dispatched
  and awaiting a result at once.
- **Spare-lane gate**: in concurrent mode (`alchemy_allow_concurrent: true`), a
  graph form pops only when an inference lane is idle beyond what the undispatched
  image queue needs. Image jobs always win contention for a process.
- **VRAM-headroom gate**: a graph form pops only when free VRAM exceeds
  `alchemy_vram_headroom_mb`. An `AlchemyHeadroomEstimator` tracks the rolling
  median VRAM cost of recent forms and raises the requirement toward it; free VRAM
  is read from the worker's per-process memory reports. With no VRAM telemetry yet
  (cold start / CPU-only), it falls back to backfill.
- **Backfill fallback**: with `alchemy_allow_concurrent: false`, all of the above
  collapses to the legacy rule: pop only when the image queue is fully drained.

Periods where only alchemy forms are in flight do **not** count as "idle" for the
no-jobs-available accounting (`WorkerState.alchemy_forms_in_flight` gates it).

## See also

- [Bridge Configuration](bridge_config.md): the config fields that drive
  throttling behavior
- [Job Lifecycle](job_lifecycle.md): where popping and scheduling fit in the
  pipeline
- [Process Lifecycle](process_lifecycle.md): model preloading lifecycle
- [`PopThrottler`][horde_worker_regen.process_management.pop_throttler.PopThrottler]
- [`InferenceScheduler`][horde_worker_regen.process_management.inference_scheduler.InferenceScheduler]
- [`LRUCache`][horde_worker_regen.process_management.lru_cache.LRUCache]
- [`VramBudget` / `RamBudget`][horde_worker_regen.process_management.resource_budget]
