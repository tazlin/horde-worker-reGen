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
[`DutyCycleSummary.is_demand_limited`][horde_worker_regen.process_management.duty_cycle.DutyCycleSummary.is_demand_limited],
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
[`span_derived_busy_ratio`][horde_worker_regen.process_management.duty_cycle.span_derived_busy_ratio]
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

[`phase_breakdown`][horde_worker_regen.process_management.duty_cycle.phase_breakdown] reports the
median seconds a job spent in each lifecycle phase, in pipeline order: `queue_wait`, `model_unload`,
`disk_load` (disk to RAM), `vram_load` (RAM to VRAM), `sampling`, `vae` (VAE decode), `encode`
(CLIP/VAE prompt and image encode), `graph_overhead` (ComfyUI graph build, validate, and teardown),
`other_inference` (node and IPC residual), `safety`, and `submit`. Only the four phases in
[`GPU_BUSY_PHASES`][horde_worker_regen.process_management.duty_cycle.GPU_BUSY_PHASES] (`vram_load`,
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

## Where to read it

The same [`DutyCycleSummary`][horde_worker_regen.process_management.duty_cycle.DutyCycleSummary]
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

### The dashboard

The [TUI dashboard](../how-to/use-the-dashboard.md) shows the same sampled figure live: the health
panel reports a `% duty cycle` check (and flags the GPU sitting near-idle while a job is supposedly
running), and the Overview's Trends panel plots the rolling mean. The dashboard is the quickest way to
see the number move while you change a setting; the log line and the report below are for attribution
and for comparing runs.

### Across sessions: `horde-duty-report`

`bridge.log` is appended across worker restarts, so one file holds many sessions, and a fair before/after
comparison means reading each session separately against the config that produced it. The
`horde-duty-report` CLI ([`horde_worker_regen.analysis.duty_log_report`][horde_worker_regen.analysis.duty_log_report])
splits the log into **session epochs** on the once-per-launch process-manager init banner and, per
epoch, prints the duty distribution, the biggest per-job gaps, the churn totals, the effective config,
and any disk-pressure dips:

```bash
horde-duty-report                 # every epoch in logs/bridge.log
horde-duty-report --last          # only the most recent session
horde-duty-report path/to/bridge.log --json
```

A single epoch renders roughly like this:

```text
== Epoch 2 | 2026-06-20 14:02-14:26 (24 min) | 8 windows ==
   config: models=111, threads=1, queue=2, max_power=32, high_perf
   verdict: below 90% target: mean 44%; ~18% idle (hand-off/no-work) + ~38% partial-utilization
   duty: mean 44%  min 31%  max 58%
   bands: <40% 3  40-60% 4  60-75% 1  75-90% 0  >=90% 0
   top per-job gaps: queue wait 24.7s/job  model load (disk) 1.8s/job  safety 0.9s/job
   reload churn: model swaps 29  VRAM evictions 22
```

Because each epoch's numbers are tied to the config that produced them, an A/B tuning comparison is
meaningful in a way that a single blended average is not. The caveat is that the horde's demand (job
sizes, model spread) varies between epochs, so short (roughly 24-minute) comparisons are noisy: prefer
longer epochs and look for large deltas rather than reading significance into a few points of duty.

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

## See also

- [Performance and backpressure](performance_and_backpressure.md): the scheduling, eviction, and budget
  machinery whose hand-off gaps the duty line attributes
- [Job lifecycle](job_lifecycle.md): the phases the per-job breakdown is measured against
- [Configure for your GPU](../how-to/configure-for-your-gpu.md): choosing models, modes, and the
  benchmark that sets them
- [Telemetry](telemetry.md): the broader run-metrics and tracing layer this builds on
- [`DutyCycleSummary`][horde_worker_regen.process_management.duty_cycle.DutyCycleSummary] and
  [`summarize_duty_cycle`][horde_worker_regen.process_management.duty_cycle.summarize_duty_cycle]: the
  shared summary used by both the live worker and the benchmark
- [`GpuUtilizationSampler`][horde_worker_regen.utils.gpu_monitor.GpuUtilizationSampler]: the background
  utilization sampler
- [`horde_worker_regen.analysis.duty_log_report`][horde_worker_regen.analysis.duty_log_report]: the
  epoch-aware log analyzer behind `horde-duty-report`
