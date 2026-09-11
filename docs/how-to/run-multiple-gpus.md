# Run multiple GPUs

One worker now drives **every GPU on the machine** under a single horde identity, queue, and download
path. You no longer need to launch a separate worker per card. By default the worker auto-detects all
accelerators and spreads jobs across them; each card can optionally take its own config.

> The older "one worker process per GPU" approach still works and is documented at the end as an
> alternative, but it is no longer the recommended default.

## The default: drive every card

Start the worker normally. It enumerates every GPU (stable PCI-bus order, indices `0`, `1`, …) and serves
jobs from all of them. The GPUs tab in the dashboard shows per-card VRAM, contexts, throughput, and
residency. Nothing extra is required for the homogeneous case.

## Choosing which cards to drive

To pin or subset the cards this worker owns, set `gpu_device_indices` in `bridgeData.yaml`. Indices are
stable across reboots:

```yaml
gpu_device_indices:
  - 0
  - 2
```

Leave it unset to drive all detected cards.

In the dashboard, the **Config → Per-GPU** tab does this for you with a card strip:
`All GPUs (auto)` keeps the list empty (drive everything), while the numbered chips (`GPU 0`, `GPU 1`, …,
plus `+ card` for higher indices) pick an explicit set. A chip is green when the running worker actually
detected that card and blue when you have selected it, so you never have to type an index.

## Per-card overrides

A heterogeneous box (say a 24 GB card alongside a 12 GB card) can give each card its own settings without
standing up separate workers. Each card sets only the fields that should differ from the global config;
everything else inherits.

Per-card overrides work the same whether the worker reads `bridgeData.yaml`, JSON, or environment variables;
the YAML parser's private state is not carried into the resolved per-card runtime configs.

The easiest path is the **Config → Per-GPU** tab: each driven, detected, or selected card gets a
collapsible section (two laid out side by side on a wide terminal, so comparing a pair of cards is easy).
Inside, every overridable knob has an *Override* toggle that is off (the disabled control shows the
inherited global value, tagged `inherited`) until you flip it (`custom`). Only toggled-on fields are
written, so a single-GPU or homogeneous machine never grows an override block. On a single-GPU machine the
tab shows a banner reminding you the per-card rules only apply once multiple cards are driven.

The equivalent YAML is a `gpu_overrides` map keyed by device index:

```yaml
gpu_overrides:
  0:                         # the 24 GB card
    max_threads: 2
    high_performance_mode: true
  1:                         # the 12 GB card
    allow_lora: false
    models_to_load:
      - "top 3"
```

Overridable per card: `max_threads`, `queue_size`, `high_performance_mode`, `moderate_performance_mode`,
`extra_slow_worker`, `preload_timeout`, `models_to_load`, `models_to_skip`, `dynamic_models`, `allow_lora`,
`allow_controlnet`, `allow_sdxl_controlnet`, `allow_post_processing`, `allow_painting`, `allow_img2img`,
`nsfw`, `max_power`, `max_batch`, `safety_on_gpu`, `enable_vram_budget`, `vram_reserve_mb`,
`vram_to_leave_free`, `whole_card_exclusive_residency`. Global-only fields (API key, downloader settings,
alchemy, …) cannot be overridden per card and are rejected if you try.

`max_batch` is a per-card ceiling on the images one request may ask for: the pop asks for the offered card's
own ceiling, and a job arriving above a card's ceiling is not dispatched there.

`nsfw` is enforced at the offer rather than per card: a popped job carries no NSFW marker, so the worker
cannot tell which returned job was NSFW. Cards that disagree on `nsfw` therefore advertise SFW for the
combined offer and for every card-scoped offer; NSFW work is only requested when every card permits it.

`safety_on_gpu` is a per-card *permission to host*, not a request. The safety check is one process on one
card, so the worker places it on a card that permits it (the one with the most measured headroom) and runs it
off-GPU when no card does. Turn it off for a card you want kept clear of safety's CUDA context; on a
single-GPU worker with no override it means exactly what the global flag has always meant.

The auxiliary lanes (post-processing, and the image-utilities lane when it is enabled) are one process each
for the whole worker, not one per card. Each is placed at its first spawn onto the card best able to host
it: safety's card is skipped when safety is on a GPU and another card is available, and among the rest the
worker prefers a card that is not already carrying another auxiliary lane, then the one with the most
measured free VRAM. That placement is then pinned. Stopping a lane to free VRAM (whole-card residency, or
the reclaim ladder relieving a saturated card) and restarting it does not re-shop the cards: the lane comes
back on the card it held, so a temporary pause cannot land it on top of a card already hosting safety or a
large model. Only removing that card from `gpu_device_indices` re-places the lane. If safety later moves
onto a card an auxiliary lane holds, the bridge log says so once; nothing moves automatically.

When cards advertise different models, features, policy, resolution ceilings, or batch ceilings, the worker
rotates complete card-scoped offers. This preserves the relationship between those fields; separately unioning them could ask
for a model from one card with a feature or size supported only by another. `gpu_pop_balance_threshold`
(default `0.5`) lets local queue imbalance prioritize the most under-fed card ahead of that fair rotation.
Cards with equivalent externally visible offers can safely share a combined request.

## Concurrency and intake are per card

Every driven card runs **its own pool of inference processes**: the plan is `queue_size + max_threads`
processes per card (VRAM-fit and RAM-fit sizing may reduce a card's count, never below one). `max_threads`,
`queue_size`, and the performance modes are therefore per-card values, and the worker-wide intake scales as
their sum:

- The **job intake budget** (how many jobs the worker holds at once, running jobs included) is the sum of
  every driven card's `queue_size + max_threads`. Four cards at `queue_size: 1` / `max_threads: 1` hold 8
  jobs, not 2.
- The **megapixelstep budget** (how much work-in-magnitude may be pending before pops pause) is the sum of
  every card's performance-mode figure (15 normal / 60 moderate / 80 high), each card contributing per its
  own effective mode.
- The **concurrent-sampling ceiling** (how many jobs may sample at once across the whole worker) is the sum
  of every card's `max_threads`. Eight cards at `max_threads: 1` run 8 concurrent jobs, not 1.
- A model busy sampling on one card may be **loaded a second time onto an idle card** when queued demand
  warrants it; the VRAM budget and displacement guards still gate the load. The second copy goes to a card
  that can run it now: a card already serving the model and sitting at its `max_threads` ranks last, since a
  copy on its sibling lane could not sample until the running job ends. Copies are capped at the model's own
  outstanding jobs and at one per card, so a burst of one model cannot take the whole pool and leave the next
  model's job reloading from disk wherever it lands.
- The displacement guard counts a model as **wanted while it is resident on any lane**, whichever card that
  lane is on, so a worker holding a distinct model per lane protects every lane and ordinary target selection
  finds a backed-up model nowhere to put a second copy.
- **Copies follow demand** where that leaves a queued job no lane at all: a job whose model is resident only
  where it is busy may displace one idle lane whose resident model has strictly fewer outstanding jobs than
  its own, taking the least demanded of them, then the one whose model has gone longest without work. The
  lane must be on a card under its `max_threads` (a copy that cannot sample buys nothing), must not hold a
  model a running job is using, and is never the queue head's own copy. The displaced model keeps its
  host-RAM copy, so its next job pays a weight upload rather than a disk read, and the total number of copies
  is still capped at the model's outstanding jobs and at one per card. Single-card workers are untouched.
- Separately from that guard, a **residency grace period** keeps a finished job's weights on the card for a
  window derived from the horde's recent job ttl, so a follow-on job for the same model is served without
  re-uploading. That window governs eviction, not which lane a load may displace; the two are independent.

Dispatch selection follows the same rule. The head of the queue can only be seated on a card that can serve
it, so a head whose card is at its own `max_threads`, whose resident lane is busy, or whose model is not
loaded anywhere yet is a fact about **one** card. The scheduler carries on down the queue and seats the
first job another card can run immediately; the head keeps its queue position and its claim on the card it
is waiting for. The card the head waits on is left alone, so nothing takes the capacity it is queued for,
and where the head is cold with a load already in flight, the card carrying that load is left alone too.

Five more worker-wide behaviours are per card for the same reason:

- A scheduling cycle **preloads and dispatches**. Staging a model onto one slot of one card no longer costs
  every other card a control-loop tick; one preload per cycle is still the ceiling.
- The **pending post-processing hold** applies to the post-processing lane's own card. A load or a dispatch
  aimed at another card proceeds while the lane waits for its drain window.
- The **preload pass** continues past a stop that names a card (its growth hold, an exclusive job on it, its
  load-serialization gate, the post-processing hold) and considers jobs that would load elsewhere. A stop
  about the host (the RAM danger floor) or about the head's own escalation still stops everything.
- The **head-priority barrier** withholds dispatch only onto the card the starved head's load is aimed at,
  and a **degraded retry** waits for its own card to empty rather than for the whole worker.
- A **dispatch a gate withholds** stops that job's card, not the cycle. The withheld job keeps its queue
  position and its card is left to it, while the dispatch loop goes on seating work the other cards can run
  now. This matters most at an aggressive `max_threads`, where a card has a second lane free for a hold to
  strand.

At startup the worker logs how the per-card figures compose ("Driving N cards, each with its own inference
process pool …" and the megapixelstep budget line), so the effective worker-wide appetite is always stated
rather than inferred.

## Memory

Driving several cards needs plenty of RAM (32 to 64 GB+). Both `queue_size` and `max_threads` multiply
memory use **per card**, so account for them across every driven card, not once for the machine. See
[Configure for your GPU](configure-for-your-gpu.md) and
[Performance and backpressure](../explanation/performance_and_backpressure.md).

## Watching each card

Every duty signal is measured per card. The periodic `GPU duty cycle` log line states each card's own
figure beside the worker-wide one, the dashboard's Trends row and GPUs panel show per-card duty, and
the near-idle alert names the card it means. The worker-wide number is a mean across the driven cards,
so treat it as a summary and read the per-card figures when it looks middling: one saturated card
beside a starved one produces exactly the same average as two evenly-fed ones. See
[GPU duty cycle](../explanation/duty-cycle.md).

## Alternative: one worker per GPU

You can still run a separate worker instance per card, each pinned to a device and given its own name.
This trades the unified queue/identity for full process isolation.

### Linux

```bash
CUDA_VISIBLE_DEVICES=0 ./horde-bridge.sh -n "GPU-0"
CUDA_VISIBLE_DEVICES=1 ./horde-bridge.sh -n "GPU-1"
```

Run each command in its own terminal (or as its own service). Each instance needs its own
`bridgeData.yaml` and a distinct worker name.
