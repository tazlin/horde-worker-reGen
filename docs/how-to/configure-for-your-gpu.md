# Configure the worker for your GPU

The worker is configured through `bridgeData.yaml`. If you used the dashboard wizard, this file was
created for you and you can tune it later from the **Config** tab or in any text editor. This page
gives sensible starting points by GPU class and explains the hardware that matters.

For what each field means and how config is loaded at runtime, see
[Bridge configuration](../explanation/bridge_config.md).

## Basic settings

If you are setting up by hand:

1. Copy `bridgeData_template.yaml` to `bridgeData.yaml`.
2. Set your `api_key` (from [aihorde.net/register](https://aihorde.net/register)). Keep this secret.
3. Set a unique `dreamer_name`. If it is already taken you will get a "Wrong Credentials" error.

## Starting points by GPU

Pick the block closest to your card and adjust from there. The benchmark (on the dashboard's
**Benchmark** tab, or `horde-benchmark ramp`) can suggest values tuned to your actual machine.

### 24 GB+ VRAM (RTX 3090, 4090)

```yaml
queue_size: 1        # <32 GB RAM: 0, 32 GB: 1, >32 GB: 2
safety_on_gpu: true
high_performance_mode: true
unload_models_from_vram_often: false
max_threads: 1       # 2 is often viable for xx90 cards
post_process_job_overlap: true
max_power: 64        # Reduce if max_threads: 2
max_batch: 8         # Increase if max_threads: 1, decrease if max_threads: 2
allow_sdxl_controlnet: true
```

### 12 to 16 GB VRAM (RTX 3080 Ti, 4070 Ti, 4080)

```yaml
queue_size: 1        # <32 GB RAM: 0, 32 GB: 1, >32 GB: 2
safety_on_gpu: true  # Consider false if using Cascade/Flux
moderate_performance_mode: true
unload_models_from_vram_often: false
max_threads: 1
max_power: 50
max_batch: 4         # Or higher
```

### 8 to 10 GB VRAM (RTX 2080, 3060, 4060, 4060 Ti)

```yaml
queue_size: 1        # <32 GB RAM: 0, 32 GB: 1, >32 GB: 2
safety_on_gpu: false
max_threads: 1
max_power: 32        # No higher
max_batch: 4         # No higher
allow_post_processing: false  # If using SDXL/Flux, else can be true
allow_sdxl_controlnet: false
```

Minimise other VRAM-consuming apps while the worker runs.

### Lower-end or under-performing GPUs

- `extra_slow_worker: true` gives more time per job, but requesters must opt in. Only use it if you
  are consistently under 0.3 MPS/s or 3000 kudos/hr with an otherwise-correct config.
- `limit_max_steps: true` caps total steps per job based on model type.
- `preload_timeout: 120` allows longer model load times.

### Systems with less than 32 GB RAM

- Set `queue_size: 0` and stick to SD 1.5 models only.
- Set `load_large_models: false`.
- Add `ALL SDXL`, `ALL SD21`, and the unpruned models to `models_to_skip`.

## Hardware tips

- **Use an SSD.** HDDs are too slow for multiple models; limit to one model with under 60 s load time.
- **Configure 8 GB+ swap** (16 GB+ preferred), even on Linux.
- **Keep `max_threads` at 2 or below** unless you have a 48 GB+ VRAM data-center GPU. Remember that
  `queue_size` and `max_threads` together set how many inference processes spawn (roughly
  `queue_size + max_threads` per card); the Config tab shows a live estimate, and
  [Process count](../explanation/bridge_config.md#process-count) explains the formula and its
  interlocks.
- **Disable sleep and power-saving** while the worker runs.
- SDXL needs around 9 GB free RAM (32 GB+ total recommended). Flux and Cascade need around 20 GB free
  RAM (48 GB+ total recommended).

## See also

- [Bridge configuration](../explanation/bridge_config.md): what each field controls
- [Performance and backpressure](../explanation/performance_and_backpressure.md): how these fields
  drive throttling and scheduling
- [Run multiple GPUs](run-multiple-gpus.md)
- [Troubleshooting](troubleshoot.md)
