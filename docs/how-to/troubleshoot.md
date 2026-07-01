# Troubleshooting

Need a hand? Ask in [#local-workers on Discord](https://discord.com/channels/781145214752129095/1076124012305993768)
or [open an issue](https://github.com/Haidra-Org/horde-worker-reGen/issues).

When sharing debug information, do **not** post `.log` files in public channels: send them to a
maintainer directly, as we cannot guarantee your API key is not present in them.

## Common problems

| Problem | Fix |
|---------|-----|
| Download failures | Check disk space and your internet connection. |
| TUI shows "Worker name problem" / logs say a name is "already registered to another account" | Worker names are unique **horde-wide** and are tied to the API key that first registered them. Your `dreamer_name` (and `alchemist_name`, when alchemy is enabled) must be unique, must not be left at the template default, and the two must differ from each other. Pick a different name in `bridgeData.yaml` (or in the dashboard's Config tab, which blocks saving an invalid name). The startup check only fails fast when the horde *proves* the name belongs to another account (it reports that account's owner); it will not keep restarting until that is fixed. |
| Logs warn a name's "ownership could not be confirmed" (often the `alchemist_name`) but the worker starts | This is a warning, not an error: the horde reveals a worker's owner only to moderators or accounts that enabled public workers, and alchemist workers are additionally absent from your account's worker id list, so the startup check cannot positively confirm the name is yours. It proceeds anyway rather than blocking boot. If the name really is yours, ignore it. If pops later fail with "wrong credentials", the name belongs to another account: pick a different one. |
| LoRAs stopped being served ("lora OFF (disk full)") | The LoRA cache disk fell below its free-space floor (`min_lora_disk_free_gb`, default 1 GB) and evicting cached LoRAs could not clear it, so the worker stopped offering LoRA jobs. Free disk space (or move the cache to a larger volume); LoRA support resumes automatically. The LoRA cache itself is bounded by `max_lora_cache_size` (GB). See [the LoRA cache and its disk floor](../explanation/performance_and_backpressure.md#the-lora-cache-and-its-disk-floor). |
| GPU mostly idle while logs show repeated LoRA `ReadTimeout` / "withholding LoRA job pops" | The ad-hoc LoRA download source (e.g. CivitAI) is slow or flaky, so LoRA jobs stall in their aux-download phase. The worker reacts on its own: it pauses LoRA pops for an escalating window, caps how many LoRA jobs queue at once so non-LoRA work keeps flowing, and reaps stuck slots faster. No action is needed; LoRA intake resumes automatically once downloads recover. If it persists, the source is likely down. See [LoRA download stalls](../explanation/performance_and_backpressure.md#lora-download-stalls-backoff-cap-and-fast-fault). |
| Antivirus blocks downloads (`CRYPT_E_NO_REVOCATION_CHECK`) | Some antivirus (for example Avast) interferes with downloads. Temporarily disable it. |
| "Path too long" or file-not-found during install (Windows) | Use a short install path (the default already is). If it persists, opt in to system-wide long-path support: set `$env:HORDE_WORKER_ENABLE_LONG_PATHS=1` before installing. This changes an HKLM setting and needs administrator rights. |
| SmartScreen "Windows protected your PC" | The installer is not code-signed yet. Click **More info**, then **Run anyway**. |
| Every job fails / TUI shows "PyTorch cannot run this GPU" (`no kernel image is available for execution on the device`) | The installed PyTorch has no CUDA kernels for your GPU's architecture: the wheel was built for a different set of GPUs than the one installed. This is common on a brand-new GPU (e.g. an RTX 50-series / Blackwell card that needs a CUDA 13 build) or a very old one (Maxwell/Pascal/Volta, dropped by CUDA 13). The inference process detects this at startup and the worker stops popping jobs rather than failing every one silently. Run `update.cmd` (or re-run the installer): the sync re-reads your GPU and corrects a stale/wrong build automatically, so this normally fixes itself on the next update. Update your NVIDIA driver if asked (a Blackwell card still needs a CUDA 13 driver before `cu130` can load). See [Choose a PyTorch build](choose-a-pytorch-build.md). |
| Job timeouts | Remove large models (Flux, Cascade, SDXL), lower `max_power`, and disable post-processing, controlnet, or LoRA. |
| Out of memory (VRAM) | The worker's VRAM/RAM budget (`enable_vram_budget`, on by default) guards against this by gating model loads on measured free memory; if you still hit it, lower `max_threads`, `max_batch`, or `queue_size`, reduce your model set, or raise `vram_reserve_mb`, and close other programs. See [Configure for your GPU](configure-for-your-gpu.md) and [the VRAM and RAM budget](../explanation/performance_and_backpressure.md#the-vram-and-ram-budget). |
| The whole worker (and TUI) vanishes to the desktop | This is a **system-RAM** OS OOM kill: the kernel reaped an inference process because the host ran out of RAM (`dmesg`/`journalctl -k` shows `Out of memory: Killed process … (python)`). It is most likely when several inference processes (a multi-GPU host) keep model weights resident *and* the box also runs a co-tenant (an alchemist, a scribe): their summed footprint exceeds RAM. The worker guards against it (an absolute RAM danger floor that sheds/throttles, a per-process RAM ceiling that recycles a ballooned process), but if it still happens: lower `ram_per_process_max_mb` so a process is recycled sooner, lower `ram_pressure_pause_percent` (degrade with more headroom), reduce the number of driven GPUs (`gpu_device_indices`) or `max_threads`/`queue_size`, move the alchemist/scribe to another machine, or add RAM. On a host running a dreamer + alchemist + scribe together, plan for each inference process to hold tens of GB. See [the VRAM and RAM budget](../explanation/performance_and_backpressure.md#the-vram-and-ram-budget). |
| Less kudos than expected | New workers have 50% of job kudos and 100% of uptime kudos held in escrow for around a week, until you become trusted. |
| Worker stuck in maintenance mode | The horde usually forces this *because the worker dropped too many jobs*, so unpausing alone will re-trigger it. Run `horde-log diagnose --last` to find the drops. They are usually generations the horde aborted as "too slow", but that has two different causes the diagnosis distinguishes: (1) generation itself is slow (lower `max_power`/`max_threads`/`queue_size`/`max_batch`, put models on an SSD); or (2) generation is fast but jobs *age in the pipeline queue* because a downstream stage (typically CPU safety, with `safety_on_gpu` off) is slower than inference. For (2), lowering `max_power` does not help: enable `safety_on_gpu` (or otherwise speed up safety). The worker also applies post-inference backpressure that bounds (2) automatically. Fix the cause, then unpause in [artbot](https://tinybots.net/artbot/settings?panel=workers). |
| Logs show "requeued it for a fresh safety check" or "Soft-pausing job pops … safety could not check a result" | A completed job was sent to the safety process but its verdict never returned (the safety process was replaced, or a result message was dropped). The worker recovers on its own: it re-checks the job (an image is never submitted without passing safety), and if safety cannot be relied on it briefly soft-pauses popping and reissues the affected job to the horde with no image. No action is needed unless it persists, which points to a failing safety process (check the `bridge_safety_*` logs). |

## Reading the logs

Logs live in the `logs/` directory. `bridge.log` is the main log; per-process logs and the
errors-only `trace.log` help pin down a specific failure. The full table is in
[Logs](../reference/logs.md). You can tail a log live: `Get-Content bridge.log -Wait` on Windows
PowerShell, or `less +F bridge.log` on Linux.

## Diagnose a crash or recovery storm from the logs

Reading `bridge.log` by hand is slow: it is appended across every restart, the real crash cause is
usually a traceback in a *different* file (a child's `bridge_inference_<N>_startup.log`), and the
interesting failures are buried in thousands of lines. The `horde-log` command does that archeology for
you. Run it from your worker directory:

```bash
# Ensure the venv is active, or use the full path to `horde-log` in the venv's `Scripts`/`bin` directory.

# Windows activate:
.\venv\Scripts\activate.ps1 # or .cmd if you are in cmd.exe
# Linux/macOS activate:
source venv/bin/activate

# Which worker launches are in the log, and how did each end?
horde-log sessions

# What went wrong in the most recent run (root cause + remediation)?
horde-log diagnose --last
```

`sessions` lists every launch with its span, version, end-reason (clean exit, gave-up-and-aborted,
operator shutdown, or killed/crashed), and peak process-recovery count, so a session that thrashed
stands out. `diagnose` then runs detectors over a session and prints ranked findings: it recognizes an
inference pool that crashes on start (and surfaces the child's actual exception, e.g. a CPU-only torch
reporting `Torch not compiled with CUDA enabled`), a recovery storm that never gave up, GPU
out-of-memory, the horde forcing the worker into maintenance for dropping too many jobs (and ties it
back to the local cause), a slow-generation spiral where the horde aborts late submissions as too slow
until it forces maintenance, a scheduler wedge from an over-conservative VRAM budget deferring
head-of-queue jobs on an idle device with ample free VRAM, and more, each with a remediation.

For a deeper look, `horde-log timeline --session N` interleaves the parent log, the per-slot child logs,
and the action ledger into one time-ordered stream, and `horde-log job <id>` traces a single job across
the parent and the slot that ran it. `horde-log watch` tails a live worker and alerts the moment a new
problem appears. All of it is read-only, accepts a `.zip` of logs someone sent you, and has a `--json`
mode; see the [command reference](../reference/cli.md#horde-log).

### Send your logs to a maintainer

When a maintainer asks for your logs, run `horde-log bundle` (or press **Support bundle** / `Ctrl+B` on
the dashboard's **Logs** tab). It writes one `horde_support_<timestamp>.zip` containing the diagnosis,
your logs, retained stats JSONL files when present, the redacted config, and a system/cache report. It
**scrubs your API key and CivitAI token** (and, by default, your home-directory path, username, and
worker name) before writing, and tells you how
many things it redacted. Redaction is best-effort, so skim the archive before you send it. See
[`horde-log bundle`](../reference/cli.md#bundle-a-redacted-archive-for-a-maintainer) for the options
(e.g. `--full-logs` for the complete history, `--keep-identifiers` to leave paths and the worker name in).

## Still stuck?

- Confirm your machine is supported on the
  [README](https://github.com/Haidra-Org/horde-worker-reGen#readme).
- For AMD or Windows-without-NVIDIA setups, see [Run on AMD ROCm](run-on-amd-rocm.md).
- For PyTorch or CUDA version mismatches, see [Choose a PyTorch build](choose-a-pytorch-build.md).
