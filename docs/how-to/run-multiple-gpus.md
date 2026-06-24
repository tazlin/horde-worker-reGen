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
`nsfw`, `max_power`, `enable_vram_budget`, `vram_reserve_mb`, `vram_to_leave_free`,
`whole_card_exclusive_residency`. Global-only fields (API key, downloader settings, alchemy, …) cannot be
overridden per card and are rejected if you try.

When cards advertise different capabilities, `gpu_pop_balance_threshold` (default `0.5`) controls when the
worker stops popping the union of all cards' capabilities and instead targets the most under-fed card, so
the horde returns work that card can actually run.

## Memory

Driving several cards needs plenty of RAM (32 to 64 GB+). Both `queue_size` and `max_threads` multiply
memory use **per card**, so account for them across every driven card, not once for the machine. See
[Configure for your GPU](configure-for-your-gpu.md) and
[Performance and backpressure](../explanation/performance_and_backpressure.md).

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
