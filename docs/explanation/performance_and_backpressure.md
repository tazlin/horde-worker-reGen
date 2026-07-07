# Performance and Backpressure

- [Performance and Backpressure](#performance-and-backpressure)
    - [The pop gauntlet](#the-pop-gauntlet)
    - [Megapixelstep backpressure](#megapixelstep-backpressure)
    - [Post-inference (safety) backpressure](#post-inference-safety-backpressure)
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
    - [Keeping a model resident between same-model jobs](#keeping-a-model-resident-between-same-model-jobs)
    - [The VRAM and RAM budget](#the-vram-and-ram-budget)
        - [Per-context overhead and the effective idle floor](#per-context-overhead-and-the-effective-idle-floor)
    - [Alchemy backpressure](#alchemy-backpressure)
    - [The LoRA cache and its disk floor](#the-lora-cache-and-its-disk-floor)
    - [LoRA download stalls: backoff, cap, and fast-fault](#lora-download-stalls-backoff-cap-and-fast-fault)
    - [Multi-GPU pop shaping](#multi-gpu-pop-shaping)
    - [See also](#see-also)

The worker sits between two external systems: the AI Horde API (which can flood
it with jobs) and the GPU (which has finite VRAM and throughput). This page
explains the throttling, scheduling, and backpressure mechanisms that keep the
worker stable under load.

## The pop gauntlet

Before [`JobPopper`][horde_worker_regen.process_management.jobs.job_popper.JobPopper] makes any network
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
6. **Post-inference (safety) backpressure**: if the post-inference backlog
   cannot clear within the job deadline, skip (see below).
7. **Megapixelstep backpressure**: if pending megapixelsteps exceed the
   configured threshold, skip (see below).
8. **Pop-rate throttle**: if less than `current_pop_frequency` seconds have
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
[`JobTracker`][horde_worker_regen.process_management.jobs.job_tracker.JobTracker] sums the megapixelsteps of
pending jobs; when that sum exceeds a threshold,
[`PopThrottler`][horde_worker_regen.process_management.scheduling.pop_throttler.PopThrottler] pauses popping
until the backlog drains. The threshold is **not** a config field; it is
derived from the active performance mode: `15` (normal), `60`
(`moderate_performance_mode`), or `80` (`high_performance_mode`). How long
popping pauses also scales with the backlog and performance mode.

When local queueing is enabled, the megapixelstep gate preserves the first
standby inference slot before it starts pausing pops. Pending megapixelsteps
include the active inference job, so a single large SDXL job can exceed the
normal-mode threshold by itself; without this exception, the worker can finish
that job and then discover it has no already-popped successor to preload or run.
After one non-running inference job is queued, the usual threshold applies again.

This prevents the worker from accepting a large number of high-resolution jobs
that would take hours to complete, starving smaller jobs behind them.

## Post-inference (safety) backpressure

Megapixelstep and queue-size backpressure bound the *pre-inference* queue. The
*post-inference* queue - jobs that have finished generating and are waiting for
the safety stage - has a different throughput constraint: safety is often a single
process, and CPU-bound when `safety_on_gpu` is off. When inference is faster than
safety, the post-inference backlog grows until jobs exceed the horde-supplied `ttl`
("seconds before this job is considered stale and aborted") and the horde aborts
them as "too slow", counting each as a dropped job.

The popper applies backpressure against this stage too: it measures the safety
stage's average wall-clock cost and stops popping while the current backlog cannot
drain within the job deadline. The tolerated backlog is

```
max(2 × num_safety, int(ttl × 0.5 × num_safety / avg_safety_seconds))
```

where `avg_safety_seconds` is an exponential moving average of measured safety
checks ([`WorkerState.record_safety_duration`][horde_worker_regen.process_management.config.worker_state.WorkerState.record_safety_duration])
and `ttl` is the most recent horde-supplied deadline (falling back to conservative
constants before either is known). The cap is self-tuning with no config knob: a
faster safety stage or longer deadline raises the tolerated backlog; a slower one
tightens it. The practical effect is that throughput settles at the slowest pipeline
stage's rate rather than at the rate the GPU produces images.

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

[`InferenceScheduler`][horde_worker_regen.process_management.scheduling.inference_scheduler.InferenceScheduler]'s
`run_scheduling_cycle` runs (when there are pending jobs and capacity) and makes decisions in this order.
Resource governance is *not* part of this cycle: the process manager drives `run_governance_tick` every
control-loop iteration regardless of queue depth (see
[Resource governance](resource_governance.md#the-governor-tick)), so the RAM pop hold and shed/restore
response stay live even when the queue is empty.

1. **Preload models**: for the first pending job whose model isn't loaded, pick
   a free process and send `PRELOAD_MODEL`, subject to the
   [VRAM and RAM budget](#the-vram-and-ram-budget), which defers the preload (and
   reclaims idle resident models) when the device cannot absorb the new model.
   The scheduler records the model as `LOADING` as soon as the command is sent;
   stale-entry cleanup must keep that entry while the child is in early preload
   states, including the short `WAITING_FOR_JOB` first-report race and the
   `UNLOADED_MODEL_FROM_RAM` -> `DOWNLOADING_AUX_MODEL` ->
   `DOWNLOAD_AUX_COMPLETE` aux-model download path.
2. **Peek ahead**: call `get_next_job_and_process(information_only=True)` to
   decide whether to block on a heavy model or batch.
3. **Blocking rules**: defer launch if:
    - `keep_single_inference` is active (batch mode, VRAM-heavy model,
      ControlNet-XL).
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
[`PerformanceModel`][horde_worker_regen.process_management.scheduling.performance_model.PerformanceModel]
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
[`model_affinity.compute_protected_processes`][horde_worker_regen.process_management.scheduling.model_affinity]
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

#### Sourcing a skip job when none is queued

A line-skip needs a small, non-LoRA job that is already resident on an idle
sibling process. Often one is: the queue holds a mix and the scheduler simply
picks it. But when the blocked head sits behind an auxiliary-model download and
*nothing queued qualifies* (every other queued job shares the head's model, is
itself LoRA-bearing, or is too large), the GPU would idle for the whole
download. `aux_model_download_line_skip_threshold_seconds` bounds how long the
scheduler tolerates that before acting: once a slot has been in
`DOWNLOADING_AUX_MODEL` past the threshold with no usable candidate, it arms
`WorkerState.wants_line_skip_candidate`. Two things then happen:

- The in-flight **count** cap is bypassed for that cycle, so a skip job that
  materialises can dispatch even though the cap is nominally reached.
- The job popper reads the flag and **biases its next pop** toward a small,
  non-LoRA job: it stops advertising LoRA support (a LoRA candidate would only
  block on its own download) and caps `max_power` down to the line-skip
  resolution ceiling (`line_skip_pop_max_power`, derived from the same
  per-performance-mode eMPS limit the scheduler uses to accept a skip candidate).

Biasing the pop is not enough on its own: the situation that arms the flag is a
*full* local queue whose head is stuck, so the popper's steady-state gates would
otherwise refuse the pop before the bias could take effect. Because a skip job is
expected to dispatch onto the idle sibling immediately rather than buffer, an
armed flag also relaxes those throughput/pacing governors for that one pop: it is
treated as **urgent** (skipping the inter-pop cadence gate), the megapixelstep
wait is bypassed (the small skip job must not be held behind the very backlog it
relieves), and the queue-depth cap is loosened by exactly one slot so a single
skip job can be admitted past a full queue. Genuinely protective gates
(shutdown, RAM-pressure and safety-backlog holds, the consecutive-failure pause,
and requiring a free process) still apply. The per-model "one running plus one
queued" cap is untouched by design: the queue is full of the head's model, so
that cap simply steers the pop toward the *other* configured models a skip job
would use.

The freshly popped small job is preloaded onto the idle sibling on the next
cycle, becomes a valid skip candidate, and keeps the GPU sampling while the
download finishes. The scheduler clears the flag as soon as no aux download
still exceeds the threshold, so the cap bypass and the pop bias last only as long
as the stall does. Set `aux_model_download_line_skip_threshold_seconds` to unset
to disable the breaker entirely.

Already-popped post-processing work has priority over the line-skip dispatch while the blocked head is only
downloading auxiliary models. The post-processing overlap gate waits for active sampling on the PP lane's
card, not for the broader in-progress job stage, so a `DOWNLOADING_AUX_MODEL` slot does not make the PP lane
idle. The control loop drains pending PP work first; the line-skip job is then free to use the card once the
lane has no immediately admissible image post-processing work.

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
- An extra-large candidate never joins a busy card, and an extra-large job already in flight never lets
  anything share the card: the whole-card tier's contract holds whatever the headroom.
- A batched candidate or a batched in-flight job blocks overlap on a tight card; with ample measured
  headroom (below) the batch instead imposes the strictest (both-heavy) headway.
- Otherwise the running job must have made size-appropriate **headway** before a candidate joins: none for
  light + light (they thread freely), modest when one side is heavy, and considerable for two heavy jobs.
  Progress is read live from the running slot's step counters (a freshly dispatched job that has not yet
  reported a step reads as `0.0`, which is exactly when a heavy overlap is most dangerous).

The heavy fractions are sized for a tight card and would otherwise price a high-VRAM card identically: on
a heavy-only queue (an all-SDXL worker) a 75% both-heavy headway converges two configured threads to
roughly one effective thread. So the gate conditions its strictness on measurement
(`_overlap_headroom_ample`): when the device's live free VRAM absorbs the candidate's full predicted
sampling peak plus the configured reserve (the same verdict the VRAM admission budget uses, and doubly
conservative here because a dispatchable candidate's weights are already resident), the heavy headway
drops to a small constant that still gives the running job its memory-hungry startup beat. No measurement
(cold start) or a disabled VRAM budget keeps the strict fractions.

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

## Keeping a model resident between same-model jobs

The eviction above is *scheduler*-driven and happens when a different model needs the space. There is a
second, finer eviction underneath it: the engine force-loads a job's model fully into VRAM to sample it,
then by default evicts it from VRAM the moment the job finishes. That blanket post-job eviction is what
lets N inference processes share one GPU without their resident footprints colliding, but it also means
the *next* job re-streams the same weights from RAM back into VRAM, a per-job RAM→VRAM cost that dominates
non-sampling time on small jobs and shows up as the `vram_transfer` loss in the
[duty-cycle report](duty-cycle.md). It is paid even when the very next job uses the *same* model on the
*same* process.

The scheduler suppresses that eviction for one dispatch under a governed live gate. Because eviction is now
both on-demand and *proven* (the [device-free governor](vram_arbiter.md) reads the truthful NVML device-free
figure, and the verified reclaim ladder takes residents back rung by rung with each free confirmed at the
device level), retention no longer has to be preemptively stingy. It grants when both hold:

- **Card healthy**: the device-free governor's committed state for the card is `HEALTHY`. A `PRESSURE` or
  `SATURATED` card is one the ladder is or may soon be reclaiming from, so it is handed no new resident. This
  reads the one figure a WDDM driver cannot misreport under demand-paging (NVML device-free), so it holds
  precisely in the regime where measured free VRAM lies.
- **Static fit**: the card's reported *total* VRAM (a constant the driver cannot misreport under pressure)
  must absorb the job's sampling peak plus the configured reserve and any committed reserves, after charging
  the sibling CUDA contexts and the job's own post-processing that share the card while the weights are held.

The [VRAM budget](#the-vram-and-ram-budget) measured-free check is deliberately *not* re-applied in this
seam: it is the admission/dispatch gate's job, and retaining already-materialized weights adds no new bytes
to the card, so a measured veto here only reintroduces the never-fires problem via committed-figure noise.
Nor is sole residency required: a second idle resident is safe because it is a first-class candidate of the
verified reclaim ladder, so retention may keep weights warm even while a sibling holds its own resident
model.

No queue lookahead gates the grant. The pop cycle refills the queue immediately *after* a dispatch drains
it, so at the dispatch instant a same-model successor is almost never visible in the pending set even
when one arrives milliseconds later; conditioning retention on seeing one makes it structurally
unreachable. Reclaim is instead just-in-time, by the parties that can actually see the demand: a cross-model
preload that no longer fits because idle retained residents hold the card defers while the ladder evicts
them (the head-of-queue reclaim targets the idle resident, newest-idle-first), and the under-pressure
reclaim overrides retention outright. An unused hold therefore costs only the interval until the next
dispatch. The sweep spares the resident copy of a model still in the queue lookahead, but only when the card
can statically afford that copy alongside the head-of-queue job's sampling peak; on a card where they cannot
coexist, keeping the copy warm would force silent driver demand-paging during sampling, which costs far more
than the one reload the protection saves.

Underneath the scheduler's retention lever sits ComfyUI's own model management, and it has the final say on
whether weights actually stay on the card. ComfyUI's *smart memory* keeps a just-used model resident in VRAM
across executions; disabling it (`--disable-smart-memory`) makes ComfyUI aggressively offload every model to
RAM the instant an execution finishes. That offload runs below the worker's `defer_vram_unload` request, so
with smart memory off a same-model back-to-back job re-uploads the full UNet, CLIP, and VAE (three RAM→VRAM
transfers) even when both the worker's retention and the parent model map agree the model is still resident:
the retention lever is silently defeated one layer down. Cross-job residency (`comfy_smart_memory: true`)
lets a same-model successor pay zero re-upload, and the parent retains the authority to undo it: the
device-free governor reads truthful NVML device-free, and the verified reclaim ladder forces an actual VRAM
free on any idle child (ComfyUI's own `free_memory` honors a full-card reclaim request regardless of the
smart-memory setting). It nonetheless ships **off** by default: residency is not yet reconciled at dispatch
time, so on a tight card a sampling peak landing beside an idle sibling's resident weights overcommits the
device faster than the ladder's tick-paced eviction can clear it, and the driver demotes VRAM to system
memory, a card-wide slowdown that costs more than the re-uploads residency saves. Until dispatch evicts
conflicting residents before the peak materializes, per-job offloading remains the safe default; the field
exists for cards whose headroom comfortably exceeds one sampling peak plus one resident model.

The support processes are additionally held to allocator-enforced VRAM quotas on CUDA hosts: the
dedicated post-processing lane and the on-GPU safety process cap their own caching allocators. The parent
can schedule when work runs, but only the allocator can bound how much a process *keeps*: freed tensors
stay in a process's pool, and under WDDM an over-committed card silently demand-pages every process
instead of failing. On non-CUDA backends the quota is a logged no-op.

The post-processing lane's quota is a *runaway guard*, not a per-job limiter. It is sized to the card:
above the largest realistic upscale/face-fix working set where the card has room, so legitimate chains
run in VRAM, while still bounding a pool that would otherwise squat the whole card between chains (a
small card keeps the guard tight and leaves the inference pool its share). Because the guard sits above
real jobs, three cooperating behaviors keep it from faulting work the card can host:

- **Gate reconciliation.** The dispatch gate knows the lane's effective cap. A chain whose estimated
  peak exceeds it can never run in the lane no matter how idle the card becomes, so the gate faults it
  without images at admission (the horde reissues it to a larger worker) rather than dispatching it into
  a guaranteed out-of-memory.
- **Self-reclaim and retry.** If a chain does hit an out-of-memory (typically a previous chain's pool
  still resident), the lane evicts its own post-processing models, empties the cache, and retries the
  chain once before reporting a fault. Only a second out-of-memory becomes a no-image fault.
- **Crisp attribution.** A genuine overstep still becomes an out-of-memory inside the offender, on a
  path that degrades deliberately (the faulted chain is reported without images).

The safety process's quota is a fixed cap sized to its resident set (the safety models plus one
evaluation); a faulted safety evaluation recycles the process.

On Windows the worker also watches the one signal the driver cannot fake: the per-process
`GPU Process Memory` counters (the data behind Task Manager's "Shared GPU memory" column). When a worker
child's *shared* (system-backed) GPU usage climbs past a threshold for consecutive samples, its
allocations were demoted out of dedicated VRAM: measured, PID-attributed demand-paging by the worker
itself, as opposed to external VRAM pressure from another application's process. That verdict denies
retention outright while active and triggers one under-pressure sweep of idle resident models on its
rising edge. The telemetry is WDDM-level and vendor-neutral; on Linux, or any host where the counters
are unavailable, the monitor simply collects nothing and the verdict stays off.

### The truthful signal hierarchy and the device-free governor

Not every "free VRAM" number is the same number, and under WDDM they disagree exactly when it matters.
The per-process view (`mem_get_info` inside a child, or the per-PID shared-segment counters above) is
unreliable near the ceiling: the driver demotes the *least-recently-touched* allocator to system memory,
so the process that goes slow and the process whose memory was demoted are usually **different** process
ids, and the per-PID shared magnitude read for a given process varies run to run for the same physical
state. The one figure that stays truthful throughout is the **NVML device-level** used/free total, read
from the torch-free parent (outside any CUDA workload). Throughput does not degrade gradually as that
figure falls; it falls off a hard cliff the instant device-free reaches roughly zero, then plateaus, so
the depth of an over-commit past saturation barely matters. The whole defense is therefore to keep
device-free from ever reaching zero.

The **device-free governor** turns that one truthful figure into a small hysteretic state machine, sampled
once per monitor tick per card:

- **HEALTHY** — device-free is above the *soft floor* (`max(1024MB, 2x` the proportional admission noise
  buffer`)`); nothing to do.
- **PRESSURE** — device-free is below the soft floor. The scheduler **holds new VRAM growth**: no new model
  is brought to VRAM on a process that does not already hold it, no safety process is restored to the GPU,
  and no paused lane is restarted. Work already sampling is left alone, because it is not the active sampler
  the driver demotes. Held preloads simply re-ask on later cycles and proceed once the card recovers.
- **SATURATED** — device-free is below the *hard floor* (`max(256MB, 0.5x` the noise buffer`)`). The card is
  at or past the cliff, so a reclaim pass runs immediately on the same tick.

The floors scale with card capacity through the same proportional noise buffer admission uses, so a large
card keeps proportional headroom and a small card is never starved below an absolute floor. State changes
are debounced over two consecutive samples: NVML is stable, but the confirm removes any chance a lone
transient (an allocator mid-materialisation, a foreign app's momentary spike) flips the state, at a cost of
one tick of latency. The governor's latest device-free reading and its PRESSURE/SATURATED transition counts
are surfaced on the run-metrics snapshot (`device_free_mb`, `governor_pressure_events`,
`governor_saturation_events`), and its committed per-card state is carried into the VRAM arbiter's device
snapshot so admission can see the same truth. On any host without NVML the free read returns nothing and no
card is ever governed.

### The verified LIFO reclaim ladder

When the governor calls a card SATURATED, a single parent-side engine owns the reclaim so there are never
two mechanisms evicting the same card by different rules. It differs from a naive "unload something" in two
ways that the WDDM physics demand.

It reclaims **LIFO** (most-recently-materialized tenant first). The driver demotes the least-recently-touched
allocator, so the newest idle resident is both the likeliest squatter and the cheapest to give back (its
weights are still warm in RAM). The rung order is fixed: unload the newest idle resident model, then release
the reclaimable allocator caches on idle processes (reserved-minus-allocated at or above the release
threshold), then evict the older idle residents, then pause the post-processing, VAE, and component lanes in
turn, then move safety off the GPU. An **actively-sampling process is never a rung**: it is the one process
the driver did not demote, and tearing it down would trade a slow job for a faulted one. Both the candidate
assembly and the targeted unload refuse a busy process, so immunity holds at two layers.

It **verifies**. A real release shows up in NVML device-used within a sample or two, so after issuing a rung
the engine watches the next one or two governor samples and compares the realized device-free gain against
the rung's promised figure (the tenant's measured reservation). A rung that yields less than half its promise
is logged against the tenant it named, recorded as a calibration event, and the engine escalates to the next
rung rather than trusting the estimate. One rung is issued per tick; a rung whose target has already gone
away frees nothing and is skipped immediately. When the whole ladder is exhausted and the card is still
SATURATED, the episode is marked **unresolved**: nothing the worker can give back relieved the card, the
signal a later, harder rung reads. The rung count, cumulative verified frees, and shortfall count are
surfaced on the run-metrics snapshot (`ladder_rungs_issued`, `ladder_verified_frees_mb`,
`ladder_verification_shortfalls`). The arbiter's deferred actuations, both the preload path's and the
dispatch-reconciliation gate's, run through this same engine, so the governor's SATURATED ladder and the
arbiter's per-cycle DEFER ladder share one reclaim execution surface. The dispatch gate's own activity is
surfaced separately (`dispatch_reconciliation_holds`, `dispatch_reconciliation_conflicts`,
`dispatch_reconciliation_hold_seconds`, and the reclaim-versus-natural-free release split) so a soak can
attribute the re-transfer cost of holding a dispatch to reconcile co-resident weights.

The verification window is **wider for teardown-class rungs**. A model unload or cache release frees its memory
synchronously as the actuator returns, but a lane pause and a safety off-GPU cycle free their memory only once
the *process has exited*, which takes longer than one governor sample: a lane pause has been measured returning
almost none of its promised context one sample later, then the full figure a sample or two after that. Teardown
rungs are therefore given one extra verification sample before a shortfall is declared, so the engine does not
falsely grade a pause that is still tearing down as short and escalate past a rung that is in fact working.

It also **restores what it paused**. A paused lane, unlike safety, has no independent mechanism to bring it
back (the runtime safety-placement policy re-promotes safety on its own; a lane stays down until something
restarts it). Each lane pause is therefore tagged with an **owner** so the whole-card residency and the reclaim
ladder can never clear each other's hold: the residency's completion loop restores only residency-owned pauses,
and the ladder restores only ladder-owned pauses. When a card's saturation episode ends and it returns fully
**HEALTHY** (the pauses are held through the intermediate PRESSURE band, since restarting a lane's CUDA context
while the card is still tight would risk re-crossing the cliff), the ladder **unwinds** its own pauses in
reverse rung order (LIFO): the last lane stopped is the first restarted. Safety is excluded from the unwind on
purpose, because the placement policy already owns its restore.

That ownership closes a liveness gap. Pending post-processing work behind an off-GPU lane suppresses its
patience clock only for a **residency-owned** pause, which has a bounded, self-restoring hold. A **ladder-owned**
pause offers no such guarantee (a card stuck saturated may never recover to HEALTHY, so the lane may never come
back), so a job stranded behind one runs the normal admission-patience countdown and, if the lane has not
returned within the window, takes the existing raw-image fallback (faulted without images so the horde reissues
it) rather than sitting in `PENDING_POST_PROCESSING` forever and wedging the drain.

### The per-step floor: fast crawl detection

The whole-job elapsed-ratio [grading](#performance-model-scoring) is slow to fire: a job only reaches its
WARN rung once its *total* elapsed sampling time is a large multiple of expected, which for a long job is
minutes. The per-step floor detects a crawl within seconds instead. On each `INFERENCE_STEP` heartbeat the
parent compares the slot's observed per-step time (the interval since its previous beat) against its expected
per-step time (the performance model's expected sampling seconds divided by the job's payload steps, with
every division guarded and the detector skipping rather than guessing when a term is missing). A demand-paged
step runs at system-memory bandwidth, measured at 3x-13x its healthy pace with nothing between on the
reference card, while a legitimately heavy job crawls uniformly near its own expected pace; three times
expected cleanly separates them. Two consecutive steps at or above that floor, with the card's governor at
PRESSURE or SATURATED, mark the slot crawling and force the reclaim ladder to run this cycle without waiting
for the elapsed-ratio rungs. The first step of a job is skipped (its inter-beat gap includes the one-time
cold load/encode work), and a single step back at healthy pace clears the crawl, so a slot that recovers in
place the instant a squatter frees is no longer flagged. The trigger count is surfaced on the run-metrics
snapshot (`per_step_floor_triggers`); the elapsed-ratio grade stays as telemetry alongside it.

### The kill as the last reclaim rung

The hung-process watchdog reaps a slot on **heartbeat silence**: a job that stops emitting progress past its
step timeout is torn down and requeued. That detector is blind to a specific, expensive failure. When the
driver demotes a sampling process's VRAM to system memory, the job does not go silent: it keeps emitting
steps, each one crawling at system-memory bandwidth. Every step refreshes the heartbeat, so the silence
timeout never trips, and the card is effectively lost for minutes while the job limps to completion.

Replacing that slot is the **terminal rung of the reclaim ladder**, reached only after every softer rung has
failed, and it gates on device-level truth rather than per-PID attribution. Three conditions must all hold:
the card has been continuously SATURATED (device-free below the hard floor) for at least the kill horizon;
the verified reclaim ladder has exhausted itself on that card without relieving it (its idle-model unloads,
cache releases, lane pauses, and safety off-GPU all ran and the card is still over the cliff); and the slot
is crawling (its per-step floor tripped, or its whole-job elapsed grade reached WARN). Only then is the
crawling sampler the last thing left to give the card back.

The per-PID PDH paging-victim map is deliberately **not** a gate here. The measured LRU physics make it
structurally unsatisfiable: WDDM demotes the least-recently-touched allocator, so the process that goes slow
(the active sampler) and the process whose shared memory grows (the idle squatter) are usually different
pids. Requiring the slow slot's own pid to appear in the victim set almost never held. The map is retained
only as a logging hint. The killed job faults as a **resource** failure, so it earns the single degraded,
isolated retry that clears the card for it rather than a plain re-dispatch onto another over-committed slot.
The crawl signals reset at the next job boundary, so each job is acted on at most once. The count of such
last-rung replacements is surfaced on the run-metrics snapshot (`paging_victim_replacements`, the counter's
name retained), alongside the in-flight WARN-slow grading count (`job_slowdown_events`). This is a
Windows/WDDM path: where NVML device-free is unavailable no card is governed, so the SATURATION gate never
holds and the kill never fires.

When granted, the dispatch carries a "keep resident after" hint to the engine, which skips the post-job
VRAM cleanup so the following same-model job samples immediately with no reload. Retention is granted on
evidence, never assumed: a disabled budget, an unreported card total, or a card the governor has not
graded `HEALTHY` all fall back to eviction, and even a granted retention is a soft hold. The engine's force-load overflow guard and the worker's under-pressure
reclaim can still evict it, so a wrong call degrades to a reload rather than an out-of-memory. This is the
mechanism that lets a sticky or homogeneous workload realise back-to-back sampling on its hot model
without paying the [structural reload](duty-cycle.md#the-structural-ceiling-on-vram-constrained-cards)
every job.

## The VRAM and RAM budget

Each inference process loads models into the **same** GPU independently. Without a
shared accountant, their combined resident footprint can exceed device VRAM,
producing an out-of-memory crash (and, for system RAM, paging to disk that
collapses throughput). The eviction rules above are *count*-based. They assume
that if the working set fits the process count it fits the device, which only
holds for SD1.5-class weights. The budget makes the worker decide on **measured**
resources instead. (The *structure* those decisions run in, one snapshot per
decision, pure policy functions, a single per-cycle governor tick, is described
in [Resource governance](resource_governance.md).)

[`VramBudget` and `RamBudget`][horde_worker_regen.process_management.resources.resource_budget]
predict a job's peak VRAM and RAM cost from hordelib's per-job burden estimate
(`hordelib.feature_impact.estimate_job_burden`, the same estimate the benchmark
pre-flight trusts) and compare it against:

- **measured device-wide free VRAM**: the conservative minimum across
  GPU-bearing child-process memory reports (inference and the dedicated
  post-processing lane)
  ([`ProcessMap.get_free_vram_mb`][horde_worker_regen.process_management.lifecycle.process_map.ProcessMap.get_free_vram_mb]),
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

Separately from the *marginal* fit check above, an **absolute system-RAM danger
floor** guards against the kernel OOM-killer. It is evaluated **every scheduling
tick**, not only when a new model needs preloading: a steady-state worker whose
queued jobs all target already-resident models would otherwise never reach the
check (the preload routine returns early when nothing needs loading) and could
grow its resident set into an OOM with the governor asleep. When measured available
RAM falls below the floor (default: 15% free, `ram_pressure_pause_percent`) the
worker pauses job pops (the self-throttle, auto-resumed on recovery), evicts idle
resident models, and **reduces the resident inference-process count** so each idle
context's pinned weights return to the OS. On a single-GPU host this reduction is
incremental: one idle context is stopped per governance tick, the host is measured
again, and the smallest idle context that can plausibly clear the current RAM
shortfall is preferred before a larger model-holding process. The worker records the
pre-pressure process target and grows the pool back one context at a time once RAM
has headroom, the pop-pause has lapsed, and no over-ceiling process is draining. On
a multi-GPU host the reduction is applied **per card**, leaving at least one context
on every driven card: a worker-wide collapse would let the victim search empty a
whole card of contexts and idle that GPU until something restored it. Each card the
reduction shrank is then grown back to its planned per-card process count once RAM
clears the floor and the pop-pause has lapsed (a card a whole-card residency is
deliberately holding down is left to that residency's own restore).

Idle-shedding cannot help when *every* process is busy, so a single process whose
resident RAM has ballooned past a **per-process ceiling** (`ram_per_process_max_mb`,
default 18 GB) is reclaimed directly while the host is under the floor: an idle
over-ceiling process is recycled immediately (its allocator-retained pages return to
the OS on respawn), and a busy one is **drained** (fed no new work by the preload
target selection) so its in-flight job finishes before it is recycled. The largest
over-ceiling process is acted on first, one per cycle, so a multi-GPU host never
empties every card at once. New drains are *initiated* only under the danger floor, so
a roomy host never recycles needlessly, but an already-placed mark **follows through**
even after the floor clears: the degrade response itself (pausing pops, shedding
footprint) routinely lifts the pressure before the drained process finishes its job,
and the mark holds the popper closed and blocks shed restore until it resolves.
Once the drained process goes idle it is recycled on the recovered host too; a mark
whose process shrank back under the ceiling or exited is simply cleared. This bounds
the per-process balloon that a generous residency policy accumulates (weights the
allocator will not return without a respawn), which is what a shed-idle-only response
cannot reach.

A softer, **pre-floor pop hold** sits above the hard floor: once available RAM is
within the marginal RAM reserve of the danger floor while work is in flight (or a
process is mid-drain) the popper stops accepting new jobs, so a job does not start
its time-to-live clock on a worker too degraded to serve it promptly and then get
aborted by the horde as too slow. In-flight work is unaffected; the hold clears as
soon as RAM recovers, the in-flight work finishes, or the drain resolves. An idle
worker whose steady-state resident footprint merely sits inside the margin is not
held: nothing on an idle host frees RAM on its own, so a hold there could never
clear, and a popped job is served immediately (no time-to-live risk) with the hard
floor still guarding actual overgrowth.

The reductions above are the runtime backstop; the **plan-time process count** is
sized to the hardware up front so the worker rarely has to reach for them. The
resolved per-card plan is `queue_size + ceiling` processes, which is sound per card
but, summed across a multi-GPU host, double-counts the single shared system-RAM
pool (a second card doubles VRAM, not RAM). So when the worker drives more than
one card,
[`cap_card_process_counts`][horde_worker_regen.process_management.process_manager.cap_card_process_counts]
lowers each card's spawned-process count so the worker-wide resident-context count
fits system RAM, never below one context per card and only ever reducing the
resolved plan. It uses conservative footprint estimates (no model reference or
measurement exists at startup); the measured runtime budget then refines the live
count downward under real pressure. A single-GPU host never double-counts the RAM
pool, so the cap is multi-GPU only and the single-card plan stays byte-identical.
Post-processing VRAM is split across two budget paths: the lane's fixed CUDA
context is part of the resident-process forecast, while each active
post-processing job's estimated upscale/face-fix peak is entered in the shared
committed-reserve ledger until the result arrives or orphan recovery clears it.
The lane reports VRAM into the same `get_free_vram_mb` view as inference, and an
idle lane can receive the same unload-from-VRAM/RAM commands used for pressure
reclaim (see [Process lanes and job chaining](process_lanes_and_chaining.md)).

On a multi-GPU worker the whole admission decision is scoped to the card a preload
would land on (the slot chosen for it). A device-pinned child reports only its own
card's VRAM, so [`get_free_vram_mb`][horde_worker_regen.process_management.lifecycle.process_map.ProcessMap.get_free_vram_mb]
and the total/live-context counts take a `device_index` and read just that card;
the budget compares the job against *that card's* free VRAM plus its own committed
reserve, eviction reclaims only that card's idle residents, and a whole-card
exclusive residency claims (and later restores) one card's process pool
independently of the others. The residency reduces the live process count to the
largest the card's VRAM proves can co-reside with the heavy model (its weights plus
the activation-inclusive reserve, against the measured per-context cost), not a blanket
collapse to one: on a high-VRAM card a small-fraction heavy model (an fp8 Flux
checkpoint on a 24 GB card) keeps an idle sibling context so the next job pipelines
without a respawn, while on a card with no such room (the same checkpoint on a 16 GB
card) it collapses to sole residency. The whole-card *intent* still governs that the
model never co-*samples* (the concurrency overlap gate), so only the teardown depth is
hardware-relative. The teardown deliberately stops idle siblings even when their model
is still queued *behind* the heavy head: the head owns the card's sampling, so those
queued jobs wait and their models reload once it drains.
The generic scale-down spares any queued-model process, which would otherwise pin the
count above the target and wedge both the initial residency establishment and later
convergence forever; the residency instead tells the scale-down it is a whole-card
collapse so it spares only the head's holder. The holder test is based on the model
being staged or resident on a live process, so a pre-staged head still converges after it finishes loading and returns to
`WAITING_FOR_JOB`. A fresh preload also chooses *which* eligible card to
load onto by the same sticky-then-least-loaded policy dispatch uses: a card already
holding the model first (no duplicate load), then the eligible card running the
fewest jobs. The single safety process is moved off-GPU only for a
residency on the card it is pinned to (the lowest-index card), and only after
already-pending or active safety checks have drained; this avoids interrupting a
completed job's safety pass and paying a full safety-process restart in the
middle of the pipeline. System RAM stays a single shared pool sized from the
*total* process count across all cards. A
single-GPU worker passes `device_index=None` throughout, so every reading is
worker-wide exactly as before. (The per-context CUDA overhead the residency
forecast assumes is a runtime/architecture constant and stays worker-wide; per-card
overhead probing is a hordelib-side follow-up.)

> The prediction is the conservative hordelib estimate, not a learned per-job
> measurement. The only measurement the worker has (per-process VRAM high-water)
> is device-wide (it reflects *every* resident model, not one job's marginal
> cost), so feeding it back would over-throttle a multi-model worker. A true
> marginal per-job measurement is a hordelib-side follow-up.

To keep measurements fresh, GPU-bearing processes emit an interval-driven memory
report (every `_memory_report_interval`, 5 s) in addition to the event-driven
reports at model load/unload, and a dead process's stale VRAM figure is cleared on
recovery so it cannot be counted as either used or free. That interval report runs
on a dedicated reporter thread, not the main loop: the main loop is blocked for the
entire duration of a GPU operation (a 20-150 s sample), so a main-loop report would
only ever be an idle-boundary snapshot taken after post-job cleanup, with the
multi-GB mid-job working set systematically invisible. The thread reads only
allocator/`mem_get_info` statistics (safe cross-thread once the device context is
initialised) and never touches compute state; it withholds the VRAM read until the
main thread has initialised the device, so it never triggers a device init off the
main thread. Each report carries the wall-clock instant it was sampled, and the
attribution reconciler treats a contributor whose report has aged past a staleness
bound (3x the report interval) as an UNKNOWN, incomparable tenant: it skips the
drift computation for that card rather than raising a false alarm or a false
all-clear against a ledger it can no longer trust.

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
  second-context delta, with the one-time runtime and the fixed device baseline already subtracted). This is
  the structural per-context cost, available from the first scheduling tick, so it also covers the startup
  window where sibling processes have not yet reached idle. `vram_per_process_overhead_mb` (config) can
  override the first-context figure when an operator knows their card.
- **The idle floor** is derived from the device-wide used-VRAM observed when every inference process is up,
  idle, and holding no model. A real inference context can retain more allocator/runtime VRAM than the
  probe's minimal matmul holder did (emptying the cache does not return it), so the floor can legitimately
  refine the probe upward. The scheduler takes the **max** of the probe and the idle-floor derivation, so it
  never believes in headroom the device will not give back.

The risk in that `max` is a *transient* spike: with `unload_models_from_vram_often` off the caching allocator
holds a just-unloaded model's freed blocks for a while, so a clean all-idle reading taken in that window
reads high and, kept as the worst-ever floor, would pin an inflated per-context cost for the whole session
and route ordinary models into needless teardowns. The floor is therefore kept honest by **invalidation**:
any later device-wide used reading below the latched floor (with at least as many inference contexts live)
proves the device runs below that level, so the retained VRAM was reclaimable and the floor is lowered to the
demonstrated reading. A *sustained* retention is never contradicted and stands; a transient spike is
corrected down toward the minimum the device has actually shown. (A reading with resident models only makes
the correction conservative, since residency adds VRAM, so the invalidation does not need the clean
all-idle precondition the capture does.)

The streaming-forecast log reports the chosen marginal alongside its inputs (`marginal/ctx=…MB(src=…,
probe=…,idle_floor=…)`), so a bundle shows which signal won and the value the (corrected) idle floor settled
on.

These feed two decisions. The streaming forecast uses them to size `free_after_model_evict` (the VRAM
achievable once idle resident models are gone) without multiplying the one-time cost by the process count.
And when an over-commit is due to retained per-context cache rather than resident model *weights*, the
scheduler computes the largest live-context count that still fits the job's peak plus reserve
(`_max_coresident_for_peak_mb`, sized from the *same* burden peak the admission verdict rejects on) and
**reduces live contexts** (stops idle sibling processes) to that depth, instead of evicting every resident
model and forcing a full reload storm. This is the structural remedy for the `threads > 1` co-residence
thrash: it fires exactly when the admission verdict would otherwise reject the head every tick and route
it into an evict-all admit.

Both of those teardown paths (the streaming forecast's `needs_teardown` and the verdict-driven context
reduction) are gated on the demand being **trustworthy** before the worker engages the disruptive whole-card
machinery (reserve the device, move safety off-GPU, hold through a cooldown). A teardown is trusted when the
model is genuinely **card-demanding**; its weights-plus-floor occupy a meaningful fraction of total VRAM, or
its baseline wants the whole card on intent **or** when the per-additional-context cost was actually
*measured* (a probe delta or a clean idle floor), so the contention it rests on is real. When neither holds,
a model whose weights are a small fraction of the card on a host that could not measure the marginal. The
per-context overhead is the conservative first-context fallback, which over-counts contexts and can manufacture
a teardown demand a model that physically co-resides never needs. The scheduler **declines** the reservation
there and serves the model by ordinary eviction (whose terminal admit still gates on real free VRAM), logging
why. This keeps a small-weight model on a large card (an SDXL checkpoint on a 24 GB device, say) co-resident
rather than reserving the card for it; on a smaller card the same checkpoint *is* card-demanding, so a teardown
there remains available. It is the difference between a teardown justified by measured contention and one
conjured by an unmeasured over-count.

### Large-model pop limiters

Whole-card residency is disruptive even when correctly sized: a queue that *alternates* distinct very-large
models (Flux → Z-Image → Flux → Flux → Z-Image) makes every switch tear the pool down and stream a fresh
multi-GB checkpoint from disk, so the worker spends most of its time loading rather than generating. Two
optional limiters act at the only point the worker controls what work it takes: the set of models it
*offers* in the horde pop request. Because they shape the offer, no job is ever popped and then dropped
(dropping is what trips the horde's "too many drops" maintenance). Both classify "very large" from the same
`EXTRA_LARGE` tier the residency machinery uses
([`is_extra_large_model`][horde_worker_regen.process_management.models.model_sizing.is_extra_large_model]:
the Flux/Cascade/Qwen/Z-Image baselines and the named VRAM-heavy checkpoints), so the set they throttle is
exactly the set that would claim the card.

- **The switch throttle** (`large_model_switch_min_seconds`): once a very-large model is loaded or queued,
  a *different* very-large model is withheld from the offer until that many seconds have elapsed since the
  last distinct large model was introduced. Jobs for the large model already in play stay offerable; only
  churning to a new one is throttled.
- **The re-entry cooldown** (`large_model_reentry_cooldown_seconds`): once the whole-card residency lease is
  up *and* no very-large model remains loaded or queued, *any* very-large model is withheld for the
  cooldown, so the worker does ordinary work for a beat instead of immediately re-thrashing. `-1` inherits
  `whole_card_residency_cooldown_seconds` (the lease it complements); `0` disables it.

Both yield to an idle escape: when the worker holds no local work at all (an empty queue, nothing in
flight), nothing is withheld, so a limiter never leaves the worker idle when the only work it could take is
a large model. The timing state lives in a pure, worker-wide
[`LargeModelPopGovernor`][horde_worker_regen.process_management.jobs.large_model_pop_governor.LargeModelPopGovernor];
both limiters are off by default (zero / inherited-zero durations) and independent of
`whole_card_exclusive_residency`.

### Governor observability

These limiters, whole-card residency, and the other conditions that hold back or reshape pops (post-inference
backpressure, the unservable-model holdback, the consecutive-failure and self-throttle pauses, pop
error-backoff, the LoRA pop backoff, the megapixelstep wait, model stickiness) are all *governors*. They funnel
into one
[`PopGovernorRegistry`][horde_worker_regen.process_management.scheduling.pop_governor_registry.PopGovernorRegistry],
fed once per control-loop tick, which tracks each governor's current *spell* (when it engaged, why, how much
longer it is expected to last) and its session totals (how many times it engaged and how long in aggregate).
This makes every governor visible the same way:

- **Logs.** The registry emits a grep-friendly `Pop governor ENTER: <name> (<reason>); expected ~<N>s` at each
  spell start and `Pop governor EXIT: <name> after <N>s (<count>x, <N>s total)` at each end, independent of
  whether a TUI is attached, so the boundaries are always in `bridge.log`.
- **TUI.** The Overview shows a *Pop governors* strip naming whatever is engaged with a live countdown; the
  Stats tab shows a per-governor table (engagements, total time, % of session).
- **Tooling.** `horde-log` ingests the ENTER/EXIT lines and a
  [`detect_pop_governor_dominance`][horde_worker_regen.analysis.detectors.detect_pop_governor_dominance] finding
  flags a governor that consumed a large share of the session; `horde-duty-report` attributes per-epoch
  engaged time to each governor alongside the duty-band shortfall, so idle/non-pop time has a named cause.

The grep-friendly boundary format is the contract between the worker and the log/duty tooling; the regexes live
in [`governor_signatures`][horde_worker_regen.analysis.governor_signatures].

## Alchemy backpressure

When `alchemist: true`,
[`AlchemyCoordinator`][horde_worker_regen.process_management.jobs.alchemy_popper.AlchemyCoordinator] runs
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
  [`CommittedReserveLedger`][horde_worker_regen.process_management.resources.resource_budget.CommittedReserveLedger]
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
  [`constrain_lora_cache_to_disk`][horde_worker_regen.process_management.models.lora_disk_guard.constrain_lora_cache_to_disk]
  runs before each LoRA-bearing job: it shrinks the effective ad-hoc budget to fit free space and
  evicts least-recently-used ad-hoc LoRAs until the floor is clear, *making room for the job's
  LoRAs*. It relies only on measured free space and LRU eviction, so it self-limits correctly even
  if the byte-budget arithmetic is off.
- **In the main process** (which has no LoRA manager),
  [`is_lora_disk_exhausted`][horde_worker_regen.process_management.models.lora_disk_guard.is_lora_disk_exhausted]
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

## LoRA download stalls: backoff, cap, and fast-fault

A full disk is not the only way LoRA work goes wrong. When the ad-hoc download source itself is slow or
flaky (CivitAI `ReadTimeout`s, an over-loaded mirror), a LoRA-bearing job can sit minutes in
`DOWNLOADING_AUX_MODEL` before the orchestrator's stuck-aux watchdog tears the slot down and requeues
it. Three mechanisms keep one bad download path from collapsing worker throughput, since the
[line-skip](#the-line-skip-cache) path deliberately rejects LoRA candidates (a skip job must be quick,
not one that itself blocks on a download) and so cannot route around a LoRA job stuck at the head.

- **Escalating pop backoff**
  ([`LoraDownloadBackoff`][horde_worker_regen.process_management.models.lora_download_backoff.LoraDownloadBackoff]):
  every time the lifecycle reaps a slot stuck downloading aux models it registers a *strike*. While a
  strike's window is active the popper stops advertising LoRA support (folded into `_lora_disk_permits`
  alongside the disk guard), so the worker stops feeding jobs into a failing download path. The window
  starts at 60 s and doubles per consecutive strike up to 30 min, then resets after a trouble-free
  stretch - a brief blip pauses LoRA work briefly, a sustained outage pauses it for a long time.
- **Concurrent-LoRA cap** (`JobPopper._lora_queue_cap_reached`): independent of any stall, the popper
  withholds LoRA support once the local queue already holds `max(1, inference_processes - 1)` LoRA
  jobs. This guarantees at least one slot's worth of room for a non-LoRA job, which *can* line-skip past
  a LoRA job blocked at the head - so the GPU keeps working even while one slot waits on a download.
- **Child-side graceful abort (no teardown)**: the real cost of a stalled aux download was the response
  to it - the parent's watchdog tearing the whole inference process down (losing its resident model and
  paying a full hordelib re-init) just because one download was slow. Instead, the parent hands each
  dispatched job an `aux_download_deadline_seconds` (its backoff-aware watchdog timeout minus a margin;
  `ProcessLifecycleManager.aux_download_deadline_for_dispatch`). When the child's `download_aux_models`
  blows that deadline it **cancels** the stalled downloads (a manager-scoped
  `cancel_active_downloads()` in the engine that abandons the retry ladder without killing the shared
  download pool) and faults the job back to the parent through the ordinary faulted-result path, then
  returns to `WAITING_FOR_JOB` with its model intact. A whole-process teardown+respawn becomes a
  slot-local fault. The parent's stuck-aux **watchdog stays as the backstop**: it only fires (and tears
  down) if the child fails to self-abort in time (e.g. a genuinely wedged process).
- **Backoff-aware fault, registered once**: the child-reported aux fault arms the same LoRA-download
  backoff strike and the same retry policy as a watchdog teardown would have - not a resource/OOM
  failure, and during an active outage dropped (handed back to the horde) rather than requeued straight
  back into the same failing download. A lone transient stall (no active incident) keeps its one retry.
  This is what stops one bad job from costing two process recoveries.

Together these turn an all-LoRA queue facing a flaky source from a near-total stall (every slot idle
behind one choking download, each stall costing a process teardown) into a graceful degradation: LoRA
intake pauses and escalates, non-LoRA work keeps flowing, and a stalled download faults its single job
and frees its slot with the model still resident - no teardown, no respawn, no doubled-up recoveries.

## Multi-GPU pop shaping

A worker driving several cards presents one identity and pops one job stream, so per-card capability
differences become pop-side shaping. By default the pop advertises the **union** of every card's
capabilities (every model any card serves, a feature/NSFW flag if any card allows it, the largest
`max_power`, the summed thread count;
[`advertised_capabilities`][horde_worker_regen.process_management.gpu.gpu_pop_shaping.advertised_capabilities]),
so the horde returns work at least one card can run; the worker then routes each returned job to an
eligible card (the same
[`eligible_card_indices_for`][horde_worker_regen.process_management.gpu.gpu_eligibility.eligible_card_indices_for]
that preload, dispatch, and placement share) and never dispatches a job to a card that cannot serve it.

When the local queue becomes lopsided - a card cannot serve at least `gpu_pop_balance_threshold` (default
0.5) of the held work - the next pop is instead **scoped** to that under-fed card's capabilities
([`under_fed_card`][horde_worker_regen.process_management.gpu.gpu_pop_shaping.under_fed_card]), so the horde
returns work the starved card can actually run rather than more for the already-fed cards.

The "locally unservable" breaker (above) is likewise **per card**: a model's over-budget fault streak is
keyed to the card it faulted on, so a model the small card cannot run is still advertised and dispatched to a
larger one, and the popper holds a model back only when *every* card that serves it has flagged it
unservable. A single-GPU worker has one card, so the union is that card's config, no pop is ever targeted,
and the streak is worker-wide - identical to before.

## See also

- [Resource governance](resource_governance.md): the decide/act structure the
  memory protections on this page run in
- [Bridge Configuration](bridge_config.md): the config fields that drive
  throttling behavior
- [GPU duty cycle](duty-cycle.md): how the reload churn and hand-off gaps from
  this page show up as measured GPU idle, and the tuning levers
- [Job Lifecycle](job_lifecycle.md): where popping and scheduling fit in the
  pipeline
- [Job State Machine](job_state_machine.md): the stages and the dual-presence
  rule the queue accounting depends on
- [Process Lifecycle](process_lifecycle.md): model preloading lifecycle
- [`PopThrottler`][horde_worker_regen.process_management.scheduling.pop_throttler.PopThrottler]
- [`InferenceScheduler`][horde_worker_regen.process_management.scheduling.inference_scheduler.InferenceScheduler]
- [`LRUCache`][horde_worker_regen.process_management.models.lru_cache.LRUCache]
- [`VramBudget` / `RamBudget`][horde_worker_regen.process_management.resources.resource_budget]
