# Run multiple GPUs

A single worker process uses one GPU. To use several GPUs on one machine, run one worker per GPU, each
pinned to a different device and given its own name.

> Future versions will not require multiple worker instances.

## Linux

Use `CUDA_VISIBLE_DEVICES` to pin each worker to a device, and `-n` to give each a distinct name:

```bash
CUDA_VISIBLE_DEVICES=0 ./horde-bridge.sh -n "GPU-0"
CUDA_VISIBLE_DEVICES=1 ./horde-bridge.sh -n "GPU-1"
```

Run each command in its own terminal (or as its own service).

## Memory

Running multiple workers needs plenty of RAM (32 to 64 GB+). Both `queue_size` and `max_threads`
multiply memory use, so account for them per worker, not per machine. See
[Configure for your GPU](configure-for-your-gpu.md) and
[Performance and backpressure](../explanation/performance_and_backpressure.md).
