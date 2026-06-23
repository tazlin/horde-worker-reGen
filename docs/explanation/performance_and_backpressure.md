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
        - [Concurrent-overlap gating](#concurrent-overlap-gating)
        - [Idle-thread diversity scheduling](#idle-thread-diversity-scheduling)
    - [Model eviction (LRU)](#model-eviction-lru)
    - [The VRAM and RAM budget](#the-vram-and-ram-budget)
        - [Per-context overhead and the effective idle floor](#per-context-overhead-and-the-effective-idle-floor)
    - [Alchemy backpressure](#alchemy-backpressure)
    - [See also](#see-also)

The worker sits between two external systems: the AI Horde API (which can flood
it with jobs) and the GPU (which has finite VRAM and throughput). This page
explains the throttling, scheduling, and backpressure mechanisms that keep the
worker stable under load.

## The pop gauntlet

Before [`JobPopper`][horde_worker_regen.process_management.job_popper.JobPopper] makes any network
call, it runs a series of gates:

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
`width × height × steps / 1,000,000`. The
[`JobTracker`][horde_worker_regen.process_management.job_tracker.JobTracker] sums the megapixelsteps of
pending jobs; when that sum exceeds a threshold,
[`PopThrottler`][horde_worker_regen.process_management.pop_throttler.PopThrottler] pauses popping
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

Model stickiness (0.0-1.0) is the probability that a pop request will only ask
for models currently loaded in VRAM. A sticky worker stays "locked" to its
loaded models, trading job variety for throughput, and automatically disengages
when no jobs are available for the loaded models.

The `bridgeData.yaml` key is **`model_stickiness`**. (The field is read
internally as `horde_model_stickiness`, but that is the *alias*, not the YAML
key: because the config model accepts unknown extras, writing
`horde_model_stickiness:` is silently ignored and the value stays `0.0`.) In
practice this is mainly a slow-disk lever: on a card with a fast disk and diverse
demand it gives little or no throughput gain, because reload churn already
overlaps across the spare inference processes. See
[GPU duty cycle → Tuning levers](duty-cycle.md#tuning-levers-and-what-they-cannot-do)
for the measured analysis.

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
`PENDING_INFERENCE` (the [dual-presence
rule](job_state_machine.md#the-stage-dual-presence-rule)). Without this headroom,
the queue would appear full when it still has capacity.

## Inference scheduling priorities

[`InferenceScheduler`][horde_worker_regen.process_management.inference_scheduler.InferenceScheduler]'s
`run_scheduling_cycle` runs every 200 ms and makes decisions in this order:

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

### Concurrent-overlap gating

`max_threads` caps how many jobs may be *in flight* at once, but it only counts jobs; it does not look at
what those jobs are or how far along they are. That count-only cap will happily let two heavy SDXL jobs
(plus a speculatively-staged third) stack their weight loads and activation peaks on the same card at the
same moment, thrashing a sampler badly enough to trip its step-timeout watchdog into a teardown.

`InferenceScheduler._concurrent_overlap_allowed` adds the missing dimension: a new job may join work that
is already sampling only when the in-flight work can tolerate it. Models are classed into size tiers
(`_model_size_tier`): a model in the VRAM-heavy list or carrying an extra-large baseline is
**extra-large**; SDXL is **heavy**; SD1.5/SD2 are **light**; an unknown baseline falls back to light so a
missing reference does not starve dispatch. The rules, scaled by tier:

- The first job (nothing in flight) always starts.
- An extra-large or batched candidate never joins a busy card, and an extra-large or batched job already
  in flight never lets anything share the card: these want the device to themselves.
- Otherwise the running job must have made size-appropriate **headway** before a candidate joins: none for
  light + light (they thread freely), modest when one side is heavy, and considerable for two heavy jobs.
  Progress is read live from the running slot's step counters (a freshly dispatched job that has not yet
  reported a step reads as `0.0`, which is exactly when a heavy overlap is most dangerous).

A blocked candidate is never dropped; it keeps its queue position and dispatches the moment the in-flight
jobs progress past the headway threshold or finish.

On a multi-GPU worker both this gate and the in-flight **count** cap (`_max_jobs_in_progress_allowed`) are
scoped to the card the candidate would run on: each card is its own sampling and VRAM domain, so the count
is taken against that card's own in-progress jobs and its own concurrency ceiling, and the headway check
compares the candidate only against jobs already sampling on the same card. A heavy job on the big card
therefore never holds back a job destined for an idle small card, and a one-thread small card never borrows
the big card's headroom to admit a second concurrent job. A single-GPU worker has one card, so every
comparison is worker-wide exactly as before.

### Idle-thread diversity scheduling

When the head-of-queue job's process is busy sampling the head's own model, the naive choice is to wait,
leaving other inference processes idle, or to load a second copy of the head's model onto a spare process.
Both waste the card. `InferenceScheduler._select_idle_thread_diversity_job` instead looks for a *later*
pending job whose (distinct) model is **already resident on an idle process** and dispatches that
concurrently. Preferring a distinct model means a run of several same-model jobs followed by one different
model processes the different model "for free" alongside the run, rather than idling a thread and tacking
it on at the end as its own load.

The diversity pick still respects the [concurrent-overlap gate](#concurrent-overlap-gating) (two heavy
models are not stacked without headway), skips degraded-retry jobs that must run isolated, and records a
[line-skip](#the-line-skip-cache) for the displaced head so it keeps its queue position and dispatches the
moment its process frees.

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
(`hordelib.feature_impact.estimate_job_burden`, the same estimate the benchmark
pre-flight trusts) and compare it against:

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
the "auto-throttle" behavior: it evicts resident models the worker would
otherwise keep staged for fast reload, and logs prominently when it does so.

Cold start (no VRAM telemetry yet) and a missing burden estimate both **admit**,
so the budget never wedges a worker that has not yet reported memory. Set
`enable_vram_budget: false` to restore the prior availability-only behavior (not
recommended on a shared or consumer GPU).

On a multi-GPU worker the whole admission decision is scoped to the card a preload
would land on (the slot chosen for it). A device-pinned child reports only its own
card's VRAM, so [`get_free_vram_mb`][horde_worker_regen.process_management.process_map.ProcessMap.get_free_vram_mb]
and the total/live-context counts take a `device_index` and read just that card;
the budget compares the job against *that card's* free VRAM plus its own committed
reserve, eviction reclaims only that card's idle residents, and a whole-card
exclusive residency claims (and later restores) one card's process pool
independently of the others. A fresh preload also chooses *which* eligible card to
load onto by the same sticky-then-least-loaded policy dispatch uses: a card already
holding the model first (no duplicate load), then the eligible card running the
fewest jobs. The single safety process is moved off-GPU only for a
residency on the card it is pinned to (the lowest-index card). System RAM stays a
single shared pool sized from the *total* process count across all cards. A
single-GPU worker passes `device_index=None` throughout, so every reading is
worker-wide exactly as before. (The per-context CUDA overhead the residency
forecast assumes is a runtime/architecture constant and stays worker-wide; per-card
overhead probing is a hordelib-side follow-up.)

> The prediction is the conservative hordelib estimate, not a learned per-job
> measurement. The only measurement the worker has (per-process VRAM high-water)
> is device-wide (it reflects *every* resident model, not one job's marginal
> cost), so feeding it back would over-throttle a multi-model worker. A true
> marginal per-job measurement is a hordelib-side follow-up.

To keep measurements fresh, inference processes emit an interval-driven memory
report (every `_memory_report_interval`, 5 s) in addition to the event-driven
reports at model load/unload, and a dead process's stale VRAM figure is cleared on
recovery so it cannot be counted as either used or free.

### Per-context overhead and the effective idle floor

The job-burden prediction above is *per job*. Sizing residency across several processes that share one
card needs a second, *per-process* quantity: how much VRAM each inference context costs **on top of** its
model weights. The first context pays a large one-time cost (the CUDA runtime allocation); each additional
co-resident context costs only a smaller **marginal** amount. Multiplying the one-time cost by the process
count would phantom away most of the card and force needless teardowns; ignoring the marginal entirely
would over-promise reclaimable VRAM. The scheduler needs both figures to forecast what is actually free
once idle models are evicted.

It derives them from two measurements, preferring hard data and erring conservative:

- **The startup accelerator probe** measures the marginal per-additional-context cost directly (its
  second-context delta). This is available from the first scheduling tick, so it also covers the startup
  window where sibling processes have not yet reached idle. `vram_per_process_overhead_mb` (config) can
  override the first-context figure when an operator knows their card.
- **The effective idle floor** is the *worst* (highest) device-wide used-VRAM reading observed when every
  inference process is up, idle, and holding no model. That is ground truth for what reclaim can never
  return: if a real inference context retains more allocator cache than the probe's minimal holder did,
  the probe under-counts the marginal and the forecast would over-promise free VRAM. The scheduler takes
  the **max** of the probe estimate and the effective-floor derivation, so it never believes in headroom
  the device will not give back. The floor only rises above the probe once contexts genuinely over-commit
  (the `threads > 1` regime), so a roomy card keeps the probe estimate unchanged.

These feed two decisions. The streaming forecast uses them to size `free_after_model_evict` (the VRAM
achievable once idle resident models are gone) without multiplying the one-time cost by the process count.
And when an over-commit is due to retained per-context cache rather than resident model *weights*, the
scheduler computes the largest live-context count that still fits the job's peak plus reserve
(`_max_coresident_for_peak_mb`, sized from the *same* burden peak the admission verdict rejects on) and
**reduces live contexts** (stops idle sibling processes) to that depth, instead of evicting every resident
model and forcing a full reload storm. This is the structural remedy for the `threads > 1` co-residence
thrash: it fires exactly when the admission verdict would otherwise reject the head every tick and route
it into an evict-all admit.

## Alchemy backpressure

When `alchemist: true`,
[`AlchemyCoordinator`][horde_worker_regen.process_management.alchemy_popper.AlchemyCoordinator] runs
its own pop loop (≈ every 1 s, popping at most every 4 s) independent of the image pop gauntlet.
Because alchemy
shares the inference and safety processes with image work, it has its own gating
so it never starves image jobs:

- **In-flight cap**: at most `alchemy_max_concurrency` forms may be dispatched
  and awaiting a result at once.
- **Spare-lane gate**: in concurrent mode (`alchemy_allow_concurrent: true`), a
  graph form pops only when an inference lane is idle beyond what the undispatched
  image queue needs. Image jobs always win contention for a process.
- **VRAM-headroom gate**: a form pops only when *effective* free VRAM exceeds
  `alchemy_vram_headroom_mb`, where effective free is the measured device-wide free
  VRAM minus everything the shared
  [`CommittedReserveLedger`][horde_worker_regen.process_management.resource_budget.CommittedReserveLedger]
  records as already committed by in-flight image and alchemy work. Image generation
  reads the same combined figure, so the two flows cannot independently admit against
  the same free VRAM. An `AlchemyHeadroomEstimator` tracks the rolling median VRAM cost
  of recent forms and raises the requirement toward it; free VRAM is read from the
  worker's per-process memory reports. With no VRAM telemetry yet (cold start /
  CPU-only), it falls back to backfill.
- **RAM-headroom gate**: because graph forms keep weights resident in system RAM, a
  form also pops only when effective available RAM clears `alchemy_ram_headroom_mb`,
  keeping alchemy from pushing a memory-resident worker into paging. When RAM cannot
  be read, this gate does not apply.
- **Backfill fallback**: with `alchemy_allow_concurrent: false`, all of the above
  collapses to the legacy rule: pop only when the image queue is fully drained.

Each in-flight form's shared-ledger reserve is charged by what it actually allocates:
a graph form reserves the predicted VRAM cost (and an `alchemy_ram_headroom_mb` RAM
hold for its resident weights), while a CLIP form runs on
the safety process's already-resident model and reserves nothing, so it never holds image
generation back. The reserve is reconciled to the in-flight set each cycle. A form whose
process dies hard before reporting a result (so no faulted result is ever sent) is detected
by the lost-form reaper (its owning process launch is no longer active) and dropped, so its
reserve is released rather than leaking and starving image generation.

Periods where only alchemy forms are in flight do **not** count as "idle" for the
no-jobs-available accounting (`WorkerState.alchemy_forms_in_flight` gates it).

## The LoRA cache and its disk floor

When `allow_lora: true`, the worker downloads CivitAI LoRAs on demand into a size-bounded local
cache (hordelib's `LoraModelManager`). Two independent bounds keep that cache from filling the disk:

- **Byte budget** (`max_lora_cache_size`, in GB): the primary bound. Ad-hoc LoRAs beyond it are
  evicted least-recently-used. This is the figure you tune for steady-state cache size. (Internally
  it reaches the manager as the megabyte env var `AIWORKER_LORA_CACHE_SIZE`.)
- **Free-space floor** (`min_lora_disk_free_gb`, default 1 GB): a safety net for when the byte budget
  is not enough on its own, e.g. a cache volume shared with other data, or weights larger than
  expected. Below the floor, every weight write risks an `ENOSPC` that can take co-located worker
  data down with it, so the worker treats the floor as a hard constraint.

The floor is enforced in two places, split across the processes that can act on it:

- **In the inference process** (which owns the live LoRA manager),
  [`constrain_lora_cache_to_disk`][horde_worker_regen.process_management.lora_disk_guard.constrain_lora_cache_to_disk]
  runs before each LoRA-bearing job: it shrinks the effective ad-hoc budget to fit free space and
  evicts least-recently-used ad-hoc LoRAs until the floor is clear, *making room for the job's
  LoRAs*. It relies only on measured free space and LRU eviction, so it self-limits correctly even
  if the byte-budget arithmetic is off.
- **In the main process** (which has no LoRA manager),
  [`is_lora_disk_exhausted`][horde_worker_regen.process_management.lora_disk_guard.is_lora_disk_exhausted]
  decides whether to stop advertising LoRA support at all. Crucially, it suppresses LoRAs **only**
  when evicting every ad-hoc LoRA (read from the persisted `lora.json`) still would not clear the
  floor, so a *recoverable* shortfall is left to the inference-side eviction rather than latching the
  worker out of LoRA work. When the shortfall is structural (non-LoRA data filling the disk), it sets
  `WorkerState.lora_disk_exhausted`, which folds into `effective_allow_lora` so new pops stop
  advertising LoRA support, and the TUI's health panel raises a prominent "LoRA disabled: disk full"
  alert. Both clear automatically once free space recovers above the floor.

This division means a tight-but-recoverable disk quietly trims the cache and keeps serving LoRAs,
while a genuinely full disk stops the worker accepting LoRA jobs it could not fulfil, instead of
crashing on a failed write.

## Multi-GPU pop shaping

A worker driving several cards presents one identity and pops one job stream, so per-card capability
differences become pop-side shaping. By default the pop advertises the **union** of every card's
capabilities (every model any card serves, a feature/NSFW flag if any card allows it, the largest
`max_power`, the summed thread count;
[`advertised_capabilities`][horde_worker_regen.process_management.gpu_pop_shaping.advertised_capabilities]),
so the horde returns work at least one card can run; the worker then routes each returned job to an
eligible card (the same
[`eligible_card_indices_for`][horde_worker_regen.process_management.gpu_eligibility.eligible_card_indices_for]
that preload, dispatch, and placement share) and never dispatches a job to a card that cannot serve it.

When the local queue becomes lopsided -- a card cannot serve at least `gpu_pop_balance_threshold` (default
0.5) of the held work -- the next pop is instead **scoped** to that under-fed card's capabilities
([`under_fed_card`][horde_worker_regen.process_management.gpu_pop_shaping.under_fed_card]), so the horde
returns work the starved card can actually run rather than more for the already-fed cards.

The "locally unservable" breaker (above) is likewise **per card**: a model's over-budget fault streak is
keyed to the card it faulted on, so a model the small card cannot run is still advertised and dispatched to a
larger one, and the popper holds a model back only when *every* card that serves it has flagged it
unservable. A single-GPU worker has one card, so the union is that card's config, no pop is ever targeted,
and the streak is worker-wide -- identical to before.

## See also

- [Bridge Configuration](bridge_config.md): the config fields that drive
  throttling behavior
- [GPU duty cycle](duty-cycle.md): how the reload churn and hand-off gaps from
  this page show up as measured GPU idle, and the tuning levers
- [Job Lifecycle](job_lifecycle.md): where popping and scheduling fit in the
  pipeline
- [Job State Machine](job_state_machine.md): the stages and the dual-presence
  rule the queue accounting depends on
- [Process Lifecycle](process_lifecycle.md): model preloading lifecycle
- [`PopThrottler`][horde_worker_regen.process_management.pop_throttler.PopThrottler]
- [`InferenceScheduler`][horde_worker_regen.process_management.inference_scheduler.InferenceScheduler]
- [`LRUCache`][horde_worker_regen.process_management.lru_cache.LRUCache]
- [`VramBudget` / `RamBudget`][horde_worker_regen.process_management.resource_budget]
