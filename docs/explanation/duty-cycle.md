# GPU duty cycle: measurement, attribution, and tuning

**Duty cycle** is the fraction of wall-clock time the GPU core is actually doing work while the worker
has jobs to run. It is the single best proxy for how much of your hardware's earning potential the
worker is realising: a card sampling at 90% duty earns roughly twice the kudos of the same card at
45%, because the other half of the clock is spent loading models, decoding, or waiting rather than
generating. The worker drives toward **90%** on a reference machine and treats anything below that as
GPU time left on the table, though as this page explains, some hardware and workload combinations have
a *structural* ceiling well below 90% that no amount of configuration can lift.

This page is about understanding that number: what it measures, how to tell whether a low reading is
your worker's fault or simply the absence of jobs, where the lost time went, how to read it live and
across sessions, and which configuration levers move it (and which only appear to).

## Two kinds of idle, and why the distinction is everything

A naive low duty cycle reading is ambiguous, and the ambiguity matters because the two causes call for
opposite responses:

- **Demand-limited idle**: the GPU is idle because the AI Horde had no jobs to hand the worker. This is
  not a worker fault and nothing you change locally will fix it; the only levers are advertising more
  models and features so you match more of the available demand (which assumes you have run the
  [benchmark](../how-to/configure-for-your-gpu.md) and have the disk space and inclination to load
  more). The worker never raises an alarm for this.
- **Efficiency loss**: the GPU is idle *despite* jobs being queued, because wall-clock is going to
  worker-side hand-off (model loading, eviction, safety checking, submitting) instead of sampling. This
  is the loss worth investigating, and the rest of this page is largely about attributing and reducing
  it.

The worker draws this line for you. A window where no completed jobs ran and at least 10% of the time
had no work available is reported as *demand-limited* and logged calmly; a low reading *with* jobs
queued escalates to a warning. The split is computed in
[`DutyCycleSummary.is_demand_limited`][horde_worker_regen.process_management.resources.duty_cycle.DutyCycleSummary.is_demand_limited],
and it is the first thing to check before reaching for any tuning knob.

## How it is measured

A background [`GpuUtilizationSampler`][horde_worker_regen.utils.gpu_monitor.GpuUtilizationSampler]
polls GPU core utilization at 10 Hz for the life of the worker. It reads through hordelib's
backend-agnostic accelerator helper (NVIDIA via NVML today, other backends as they gain telemetry), so
the worker itself makes no NVIDIA assumption and never touches `pynvml` directly. On hardware with no
utilization source (CPU, fake, or a backend without a telemetry path) the sampler collects nothing and
reports `None`, and the worker falls back to the phase-derived proxy described below.

Every 180 seconds the worker emits one `GPU duty cycle` line (see
[`HordeWorkerProcessManager._maybe_log_duty_cycle`][horde_worker_regen.process_management.process_manager.HordeWorkerProcessManager._maybe_log_duty_cycle]),
where the window each report covers is exactly that same 180 seconds, so the utilization figure, the
per-job attribution, and the no-jobs share all describe one consistent slice of time. The line's
severity is matched to the cause, which lets you grep many workers' logs and triage by log level
alone:

- **DEBUG** at or above the 90% target (healthy, kept quiet),
- **INFO** between 75% and 90%, or whenever the window was demand-limited (idle the horde caused, not
  the worker),
- **WARNING** below 75% with jobs queued (genuine worker inefficiency worth investigating).

### The headline and the busy fraction

The line carries two GPU numbers, and reading them together tells you *how* the GPU is being
underused, not just that it is:

- The **mean** utilization is the headline duty cycle: the average core load across the window.
- The **busy fraction** is the share of samples with *any* GPU activity at all (utilization at or above
  a low 5% threshold).

When both are high the GPU is saturated. A large gap between them, busy high but mean low, means the
GPU is *on* most of the time but rarely *saturated*: it is doing light, latency-bound work such as
streaming weights into VRAM, VAE decode, encode, or IPC hand-off rather than running the sampler flat
out. That signature points you at the inter-job phases rather than at the sampler itself.

### When there is no NVML: the phase-derived proxy

On a backend that cannot report utilization, the worker still produces a duty figure from the job
timings alone.
[`span_derived_busy_ratio`][horde_worker_regen.process_management.resources.duty_cycle.span_derived_busy_ratio]
divides the GPU-touching phases of a typical job (`vram_load`, `sampling`, `vae`, `encode`) by its
whole wall-clock, giving a phase-attributed duty estimate that needs no tracing backend. The log line's
`source=` field tells you which signal backed the headline: `nvml` for a measured figure or
`phase-derived` for the proxy. The proxy is also what makes the attribution meaningful on CPU-only and
CI runs, where there is no hardware counter to read.

## Per-job attribution: where the wall-clock went

The same 180-second line names where the time went, so a low duty cycle is *explained* on the spot
without standing up a tracing backend. There is deliberately no separate per-event logging, which would
spam the log; everything rides on the one throttled line. Two attributions are folded in.

### Phase breakdown

[`phase_breakdown`][horde_worker_regen.process_management.resources.duty_cycle.phase_breakdown] reports the
median seconds a job spent in each lifecycle phase, in pipeline order: `queue_wait`, `model_unload`,
`disk_load` (disk to RAM), `vram_load` (RAM to VRAM), `sampling`, `vae` (VAE decode), `encode`
(CLIP/VAE prompt and image encode), `graph_overhead` (ComfyUI graph build, validate, and teardown),
`other_inference` (node and IPC residual), `safety`, and `submit`. Only the four phases in
[`GPU_BUSY_PHASES`][horde_worker_regen.process_management.resources.duty_cycle.GPU_BUSY_PHASES] (`vram_load`,
`sampling`, `vae`, `encode`) put the GPU core to work; the rest are worker-side hand-off the
[scheduler](performance_and_backpressure.md) can try to shrink, and the line surfaces the two largest
of them as the "biggest worker-side gaps".

The non-`other_inference` engine phases come from hordelib's per-job `phase_seconds` carried over the
job-metrics IPC. Engines that predate those keys simply do not report them, in which case the breakdown
degrades gracefully: the missing buckets are omitted and their time folds back into `other_inference`
exactly as before, so an older engine produces a coarser but still correct picture.

### Reload churn

The second attribution counts the between-jobs reload and respawn events in the window, rendered as
`reload churn: N model swaps, M VRAM evictions, ...`:

- A **model swap** is a preload that displaced a *different* model already resident on that process.
- A **VRAM eviction** is an idle model unloaded to make room (see
  [model eviction](performance_and_backpressure.md#model-eviction-lru)).
- A **process cycle** is a healthy idle process restarted to reclaim system RAM.

None of these are faults; they are the normal mechanics of fitting more models than processes onto a
finite card. What matters is their *rate*. High churn inflates `queue_wait` and `disk_load`, so naming
the counts on the duty line points you straight at the reload behaviour behind a low reading and at the
levers, [model stickiness](#tuning-levers-and-what-they-cannot-do) and residency, that suppress it.

A distinct cost hides inside `vram_load` even when no swap or eviction is counted: the engine evicts a
job's model from VRAM after every run, so the next job re-streams the same weights from RAM, a RAM→VRAM
reload paid even for the *same* model back-to-back. The scheduler now suppresses that per-job eviction
whenever the VRAM budget confirms the card can afford to carry the weights, reclaiming an unused hold
just-in-time at the next dispatch that needs the card (see
[keeping a model resident between same-model jobs](performance_and_backpressure.md#keeping-a-model-resident-between-same-model-jobs)),
so a homogeneous or sticky workload stops paying `vram_transfer` on its hot model and the residual reload
loss narrows to genuine model *switches*.

## Per-probe timing: warmup versus inference

Duty cycle as described above is a *steady-state* measure: it deliberately discounts cold-start so it
reflects a running worker. A single benchmark **capability probe** measured in isolation tells the
opposite story, and the difference is worth naming because it routinely surprises. A probe run on its
own (for example `pytest -m gpu -k controlnet`, or any probe the supervisor cannot serve from a warm
worker) boots its *own* worker: it spawns the process, imports torch, initialises the inference engine,
and cold-loads a checkpoint before the first pixel is sampled. On a warm worker that cost is paid once
and amortised across many jobs; on a per-probe cold boot it is paid every time, so an isolated probe
can read as minutes of wall-clock at a low GPU-core duty cycle even though the actual generation was
fast. The low duty there is an artifact of the measurement boundary, not a worker fault.

[`probe_timing`][horde_worker_regen.benchmark.capabilities.timing.probe_timing] makes that split
explicit from the timestamps the harness already records, with no extra instrumentation. It attributes
a probe's whole wall-clock to three segments:

- **startup**: run start to the first job's inference (process spawn plus engine init),
- **active window**: the first job's inference start to the last job's completion (where work is
  produced, including the one-time cold model load surfaced separately as `cold_model_load_seconds`),
- **teardown**: the last completion to the end of the run (shutdown and drain),

and reports `gpu_active_seconds` (summed sampling, VAE, encode, and VRAM load) with its
`gpu_active_fraction` of the whole, which is the headline that explains a low isolated reading: a cold
boot can leave only a small fraction of the run actually computing. The result rides on each
[`CapabilityProbeResult`][horde_worker_regen.benchmark.capabilities.result.CapabilityProbeResult] and is
logged per probe; the pytest probe suites also print a session-end table so a whole run's
warmup-versus-inference cost is visible at a glance. The lesson the numbers teach is the same one the
warm-session driver acts on: amortise the boot by reusing a worker across probes rather than paying
startup once per capability.

## Where to read it

The same [`DutyCycleSummary`][horde_worker_regen.process_management.resources.duty_cycle.DutyCycleSummary]
surfaces in three places, so you can watch it live or reconstruct it after the fact.

### The live log line

A representative line, with jobs queued and the worker below target:

```text
GPU duty cycle 47% over last 183s (target 90%, source=nvml, busy=82%). biggest worker-side gaps:
model load (disk) 1.8s/job, safety 0.9s/job; reload churn: 23 model swaps, 18 VRAM evictions.
jobs: 14 done | 3 pending | 1 in-flight; processes: ...
```

Read left to right: the GPU was busy 82% of the time but only averaged 47% load (light, latency-bound
work, not saturation); none of the window was demand-limited (no "had no jobs available" clause, so
this is efficiency loss, not lack of demand); the biggest worker-side sinks were disk model loads and
safety checks; and 23 swaps plus 18 evictions in three minutes is the churn driving those gaps. The
trailing context confirms there was always a job in flight, ruling out an idle horde.

### Slot-duty attribution: what the configured capacity was doing

The device counters cannot distinguish a saturated one-thread worker from a two-thread worker running
at half capacity: one busy sampler reads as "busy" either way. The same duty line therefore also
carries a **slot attribution** fragment from
[`SlotDutyAccumulator`][horde_worker_regen.process_management.scheduling.slot_duty.SlotDutyAccumulator]:
every scheduler tick attributes each configured inference slot's elapsed time to `sampling` or to
exactly one named reason the slot stayed empty, so the shares sum to 100% of `capacity x wall` and
"active vs idle vs gated" is a direct read:

```text
slot attribution (capacity 2): sampling 61%, overlap_headway 17%, no_local_work 12%, model_loading 6%
```

`no_local_work` is supply-side (no queued job wanted the slot: horde demand or a pop governor, which
the pop-governor registry names). Every other bucket is a scheduler gate, derived by the same
classification that explains a parked head
([`InferenceScheduler._classify_dispatch_stall`][horde_worker_regen.process_management.scheduling.inference_scheduler.InferenceScheduler._classify_dispatch_stall]),
so the attribution line and the `Inference dispatch stalled` text never name different causes.
`unexplained` growing is itself a finding: an empty slot with queued work that no gate claims is the
scheduler-bug-shaped case. The cumulative totals also ride every stats sample (`slot_duty_totals`),
so `horde-log duty` reports the same breakdown per session offline, alongside a **concurrency
occupancy** line (share of wall at each in-flight count) computed from the samples' `jobs_in_progress`,
which works even on stats files predating the slot-duty fields.

### The dashboard

The [TUI dashboard](../how-to/use-the-dashboard.md) shows the same sampled figure live: the health
panel reports a `% duty cycle` check (and flags the GPU sitting near-idle while a job is supposedly
running), and the Overview's Trends panel plots the rolling mean. The dashboard is the quickest way to
see the number move while you change a setting; the log line and the report below are for attribution
and for comparing runs.

### Across sessions: `horde-duty-report`

The preferred offline source is the worker-owned stats JSONL export under `.horde_worker_regen/stats/`.
Those files carry one-second `stats_sample` records plus finalized `job_completed` records, so the
report can account for every sampled interval, split **idle-time loss** from **partial-utilization
loss**, and then enrich the buckets with per-job phase metrics, process-state summaries, scheduler
intent, maintenance/backoff flags, and churn counters when the export contains them. Older stats files
with fewer fields still parse; missing attribution is reported as `unknown` rather than dropped.

`bridge.log` remains the fallback for older sessions or workers with stats export disabled. The
`horde-duty-report` CLI ([`horde_worker_regen.analysis.duty_log_report`][horde_worker_regen.analysis.duty_log_report])
tries stats first when no explicit log path is supplied, then falls back to the epoch-aware log parser.
Point it at either data source explicitly when comparing artifacts from a support bundle:

```bash
horde-duty-report                              # stats JSONL if present, else logs/bridge.log
horde-duty-report --stats .horde_worker_regen/stats --last
horde-duty-report --logs logs --json
horde-duty-report path/to/bridge.log           # legacy log-only path
```

A stats-backed session renders roughly like this:

```text
== Stats session 20260620-140201 v12.16.0 | 2026-06-20 14:02-14:26 | 1440 samples ==
   duty: mean 44%  busy 82%  completed jobs 61
   top loss buckets: model_load 420s (idle 180s, partial 240s)  scheduler_wait 210s (idle 210s, partial 0s)
   phase medians: sampling 8.2s/job  model_load 1.8s/job  vram_transfer 1.2s/job  safety 0.9s/job
   inference queue wait: total 210s (3.4s/job; median 2.1s, p90 8.8s, max 22.0s; 0.42x sampling time; overlaps active inference)
   queue wait by model: model-a 90s/18j  model-b 54s/12j
   inference dispatch gap: 32s queued with no active inference (15% of queued sample time)
   dispatch gap states: inf#0=PRELOADING_MODEL 20s  inf#0=WAITING_FOR_JOB 12s
   churn: model_swap 0.48/job  vram_eviction 0.36/job
   operator view: Model load/transfer dominated loss; reduce model churn or use a smaller served model set for this VRAM size.
   maintainer view: Unknown/unattributed time remains; inspect sample state/intent coverage around the evidence timestamps.
```

The report intentionally keeps three related views separate. Idle seconds are wall-clock intervals where
the GPU was mostly not doing work, attributed to demand limits, scheduler waits, local pause, API
backoff, safety/submit queues, recovery, or `unknown`. Partial-utilization seconds are the remaining
shortfall when the GPU was active but the mean utilization stayed below the target; those are attributed
to the dominant phase or worker state (for example model load, VRAM transfer, safety, or
post-processing). `inference_queue_wait` is popped-to-inference-start latency and can overlap active
sampling when a standby job waits behind the one legal inference slot. `inference_dispatch_gap` is the
narrower scheduler-delay signal: sampled time where inference work was queued and no inference job was
active. Because each stats session is grouped by the export filename stamp, rotated `.jsonl` and
`.jsonl.gz` files stay together for A/B comparisons. The same demand caveat still applies: short
comparisons are noisy because the horde's model mix and job sizes vary, so prefer longer sessions and
large deltas.

## The structural ceiling on VRAM-constrained cards

The most important thing to understand about duty cycle is that **a low reading is not always fixable**,
and chasing 90% on hardware that cannot reach it wastes effort.

Reaching high duty means the GPU never stops sampling, which requires the *next* job's model to already
be resident in VRAM while the *current* job is still sampling, so the switch costs nothing. That
overlap needs enough VRAM to hold two large models at once. On a card that can only hold one large
model, and with a single sampling slot (`max_threads: 1`, which is correct on such a card), the next
model's disk-to-RAM-to-VRAM load *cannot* overlap the current job's sampling. The GPU therefore goes
idle on every model switch, and with a large, diverse model set switches are frequent. This is a
hardware limit, not a scheduling bug: no configuration knob can synthesise VRAM that is not there.

In practice a memory-constrained card (for example, a 16 GB card running a 100+ model set spanning
SD1.5, SDXL, and Flux) settles around 45-55% mean duty, with **zero** demand-limited windows: the horde
always has work, yet a third or more of the wall-clock is the inter-job GPU stall. Eliminating
`queue_wait` and swaps barely moves the headline in this regime, which is the tell that queue_wait was
never the binding constraint (it overlaps across the spare inference processes); the binding constraint
is the VRAM that would let the next model load while the current one samples. Crossing into the
high-duty regime on a large model set is a "buy more VRAM" (24 GB+) outcome, not a "tune harder" one.

Not every load that *looks* structural is. On a contended card, ComfyUI's VAE-decode step frees its
worst-case decode estimate up front, which used to evict the just-sampled diffusion model to host a few
hundred MB of autoencoder: an inflated `vae` phase plus a `vram_load` on the very next same-model job,
misread as an unavoidable swap. The engine now clamps that estimate for small support-model loads
(falling back to tiled decoding on a genuine shortfall), and the worker's same-model VRAM retention
credits the resident weights when judging fit, so back-to-back same-model jobs can sample with no
reload even when the free-VRAM reading looks tight.

This is exactly why the attribution above matters: it lets you distinguish a genuinely fixable
efficiency loss (high churn from an avoidable swap pattern, a slow disk inflating `disk_load`) from the
structural ceiling, so you stop tuning when there is nothing left to win.

The same logic governs the [benchmark](../how-to/configure-for-your-gpu.md) soak, whose duty target is
**advisory by default** for this reason: it measures and reports duty against the 90% reference but does
not fail a level for missing it. Pass `--strict-duty` to make the soak enforce the gate when you are
deliberately validating a machine expected to reach it.

## Tuning levers (and what they cannot do)

These are the configuration fields that bear on duty cycle. None of them can lift the structural ceiling
above; their job is to reduce *avoidable* efficiency loss and to suit the worker to its hardware. See
[Performance and backpressure](performance_and_backpressure.md) for how each fits into scheduling and
[Bridge configuration](bridge_config.md) for the full field reference.

- **`model_stickiness`** (0.0 to 1.0) biases job pops toward already-resident models, trading job
  variety for fewer swaps. Note the trap: the bridge-data field is read internally as
  `horde_model_stickiness` but its YAML key is the alias **`model_stickiness`**. Because the config
  model accepts unknown extras, writing `horde_model_stickiness:` in `bridgeData.yaml` is *silently
  ignored* and the value stays 0.0, so always use `model_stickiness:`. On a memory-constrained card with
  diverse demand, stickiness gave **no duty improvement and lower throughput** in testing, because swaps
  already overlap across the spare inference processes; it earns its keep mainly on slow-disk workers
  where avoiding a reload is a large, real saving. It is not a general duty lever. See
  [model stickiness](performance_and_backpressure.md#model-stickiness).
- **`high_performance_mode`** is a **24 GB+** setting: it cuts the process timeouts to one third, which
  only makes sense when the card has the headroom to keep models resident and switch fast. On a 16 GB
  card it is too aggressive; turning it **off** measured cleaner (fewer swaps, much smaller
  `queue_wait`, slightly higher duty, and fewer transient wedges). Leave it off below 24 GB.
- **`unload_models_from_vram_often`** is recommended **on** for cards under 16 GB, where freeing VRAM
  between jobs is worth the reload cost. On larger cards leaving it **off** lets the worker keep
  recently-used models staged in RAM for fast reload (the VRAM/RAM budget decides what stays resident),
  which is the pairing that actually raises duty, and only when the working set genuinely fits.
- **`gpu_sampling_lease_enabled`** attacks the inter-job stall head-on: one process holds the lease and
  runs the denoise loop while the others stage their next pipeline (RAM->VRAM load, prompt encode), so
  when the sampler finishes the next job starts immediately instead of the GPU idling through warm-up.
  It needs residency to overlap, so it is **counterproductive with `unload_models_from_vram_often: true`**
  (the model is fully evicted between jobs, so the lease just serializes the reload behind sampling).
  - **`gpu_sampling_lease_slots`** is the active-denoise gate (distinct from `max_threads`, which is the
    job-*admission* gate). Leave it **unset** so it tracks `max_threads`: enabling the lease then never
    samples fewer jobs at once than the worker already runs concurrently. Set it to **1** only on
    hardware without CUDA MPS (e.g. Windows WDDM), where concurrent denoise loops merely time-slice the
    GPU: there, one denoise loop plus staged-ahead siblings is the efficient shape and raising the count
    only inflates the coverage-based duty number without lifting throughput.

## See also

- [Performance and backpressure](performance_and_backpressure.md): the scheduling, eviction, and budget
  machinery whose hand-off gaps the duty line attributes
- [Job lifecycle](job_lifecycle.md): the phases the per-job breakdown is measured against
- [Configure for your GPU](../how-to/configure-for-your-gpu.md): choosing models, modes, and the
  benchmark that sets them
- [Telemetry](telemetry.md): the broader run-metrics and tracing layer this builds on
- [`DutyCycleSummary`][horde_worker_regen.process_management.resources.duty_cycle.DutyCycleSummary] and
  [`summarize_duty_cycle`][horde_worker_regen.process_management.resources.duty_cycle.summarize_duty_cycle]: the
  shared summary used by both the live worker and the benchmark
- [`GpuUtilizationSampler`][horde_worker_regen.utils.gpu_monitor.GpuUtilizationSampler]: the background
  utilization sampler
- [`horde_worker_regen.analysis.session_duty`][horde_worker_regen.analysis.session_duty]: the
  stats-backed session analyzer behind the preferred `horde-duty-report` path
- [`horde_worker_regen.analysis.duty_log_report`][horde_worker_regen.analysis.duty_log_report]: the
  CLI and legacy epoch-aware log analyzer behind `horde-duty-report`
