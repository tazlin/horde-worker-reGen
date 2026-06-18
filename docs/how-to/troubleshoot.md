# Troubleshooting

Need a hand? Ask in [#local-workers on Discord](https://discord.com/channels/781145214752129095/1076124012305993768)
or [open an issue](https://github.com/Haidra-Org/horde-worker-reGen/issues).

When sharing debug information, do **not** post `.log` files in public channels: send them to a
maintainer directly, as we cannot guarantee your API key is not present in them.

## Common problems

| Problem | Fix |
|---------|-----|
| Download failures | Check disk space and your internet connection. |
| Antivirus blocks downloads (`CRYPT_E_NO_REVOCATION_CHECK`) | Some antivirus (for example Avast) interferes with downloads. Temporarily disable it. |
| "Path too long" or file-not-found during install (Windows) | Use a short install path (the default already is). If it persists, opt in to system-wide long-path support: set `$env:HORDE_WORKER_ENABLE_LONG_PATHS=1` before installing. This changes an HKLM setting and needs administrator rights. |
| SmartScreen "Windows protected your PC" | The installer is not code-signed yet. Click **More info**, then **Run anyway**. `winget install` avoids the prompt. |
| Job timeouts | Remove large models (Flux, Cascade, SDXL), lower `max_power`, and disable post-processing, controlnet, or LoRA. |
| Out of memory | The worker's VRAM/RAM budget (`enable_vram_budget`, on by default) guards against this by gating model loads on measured free memory; if you still hit it, lower `max_threads`, `max_batch`, or `queue_size`, disable `high_memory_mode`, or raise `vram_reserve_mb`, and close other programs. See [Configure for your GPU](configure-for-your-gpu.md) and [the VRAM and RAM budget](../explanation/performance_and_backpressure.md#the-vram-and-ram-budget). |
| Less kudos than expected | New workers have 50% of job kudos and 100% of uptime kudos held in escrow for around a week, until you become trusted. |
| Worker stuck in maintenance mode | Log into [artbot](https://tinybots.net/artbot/settings?panel=workers) with the worker running and click "unpause". Check the logs for `ERROR` entries to find the root cause. |

## Reading the logs

Logs live in the `logs/` directory. `bridge.log` is the main log; per-process logs and the
errors-only `trace.log` help pin down a specific failure. The full table is in
[Logs](../reference/logs.md). You can tail a log live: `Get-Content bridge.log -Wait` on Windows
PowerShell, or `less +F bridge.log` on Linux.

## Still stuck?

- Confirm your machine is supported on the
  [README](https://github.com/Haidra-Org/horde-worker-reGen#readme).
- For AMD or Windows-without-NVIDIA setups, see [Run on AMD ROCm](run-on-amd-rocm.md).
- For PyTorch or CUDA version mismatches, see [Choose a PyTorch build](choose-a-pytorch-build.md).
