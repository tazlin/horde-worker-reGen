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

When cards advertise different models, features, policy, resolution ceilings, or batch ceilings, the worker
rotates complete card-scoped offers. This preserves the relationship between those fields; separately unioning them could ask
for a model from one card with a feature or size supported only by another. `gpu_pop_balance_threshold`
(default `0.5`) lets local queue imbalance prioritize the most under-fed card ahead of that fair rotation.
Cards with equivalent externally visible offers can safely share a combined request.

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
