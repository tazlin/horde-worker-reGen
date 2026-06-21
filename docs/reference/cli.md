# Command-line reference

Entry points, flags, and environment variables. For task-oriented walkthroughs, see the
[how-to guides](../how-to/install.md).

## Launcher scripts

These wrappers prepare the environment, then call the entry points below. Use the `.cmd` form on
Windows, `.sh` on Linux/macOS, and the `-rocm` variants on AMD Linux.

| Script | Purpose |
|--------|---------|
| `install.ps1` / `install.sh` | One-line installer (download release, bootstrap runtime, launch). |
| `update-runtime` | Install or update dependencies into the managed environment. |
| `horde-worker` | Launch the worker: web browser dashboard by default, `--terminal` for the in-terminal UI, or `--headless` for no UI (foreground worker, downloads models first). |
| `horde-bridge` | Run the headless worker (downloads/verifies models first). |

`horde-worker` and `horde-bridge` pass any extra arguments through to the underlying program.

## Console entry points

Installed as console scripts (defined in `pyproject.toml`):

| Command | Module | Role |
|---------|--------|------|
| `run_worker` | `horde_worker_regen.run_worker:start` | Headless worker. |
| `download_models` | `horde_worker_regen.download_models:main` | Download and verify configured models. |
| `horde-worker` | `horde_worker_regen.tui.app:main` | The dashboard (TUI). |
| `horde-worker-web` | `horde_worker_regen.tui.web:main` | Serve the dashboard over the web. |
| `horde-worker-host` | `horde_worker_regen.tui.worker_host:main` | Background worker host the web dashboard attaches to. |
| `horde-benchmark` | `horde_worker_regen.benchmark.cli:main` | Progressive benchmark. |

## `horde-worker` (dashboard)

| Flag | Meaning |
|------|---------|
| `--process-mode {real,fake}` | `real` runs the GPU worker (default); `fake` runs a synthetic worker. |
| `-e`, `--load-config-from-env-vars` | Configure from `AIWORKER_*` env vars instead of `bridgeData.yaml`. |
| `-n`, `--worker-name NAME` | Override the worker name. |
| `--amd`, `--amd-gpu` | Enable AMD GPU optimisations. |
| `--config PATH` | `bridgeData.yaml` the config editor reads and writes (default `bridgeData.yaml`). |
| `--no-auto-restart` | Do not relaunch the worker if it crashes. |
| `--attach HOST:PORT` | Attach to a running worker host instead of owning the worker. |
| `--directml N` | DirectML device index. DirectML is currently unavailable, so this has no working backend. |

The launcher-only flags `--terminal` (in-terminal UI), `--headless` (no UI; runs the foreground worker
via `horde-bridge`), and `--host HOST` (bind the served dashboard, unauthenticated) are handled by the
`horde-worker` script before this program runs. In the default browser mode on a machine with no
graphical display, the launcher falls back to the in-terminal UI automatically.

## `run_worker` (headless)

| Flag | Meaning |
|------|---------|
| `-v` | Increase console verbosity. Repeatable (`-vvv`). |
| `--no-logging` | Disable console logging. |
| `-e`, `--load-config-from-env-vars` | Load config only from environment variables (useful in containers). |
| `--amd`, `--amd-gpu` | Enable AMD GPU optimisations. |
| `-n`, `--worker-name NAME` | Override the worker name (for running multiple workers on one machine). |
| `--directml N` | Enable DirectML on the given device index (currently unavailable). |

## `horde-benchmark`

Progressive worker benchmarking. Subcommands:

| Subcommand | Purpose |
|------------|---------|
| `ramp` | Run the progressive ramp benchmark via the canned-job harness. |
| `plan` | Show each level's resource requirements and predicted run/skip verdict (no worker is started). |
| `report OUT_DIR` | Re-render the markdown report from an existing output directory. |
| `monitor OUT_DIR` | Tail a run's `progress.jsonl` live (attach to or replay a run). |
| `live` | Open-loop load generation against a live API (not yet implemented). |

The ramp walks an ordered ladder per tier: a conservative **baseline** (stage A), then **concurrency**
(queue/threads/batch, B), **features** (C), **alchemy** (D), and optional **downloads** (E), followed by
a sustained-load **validation soak** (V). Stage C is grounded in what hordelib actually supports:
classic **controlnet** (canny/depth/openpose preprocessors) is exercised on SD1.5 only, while the
**qr_code** workflow (the real SDXL controlnet capability, gated by `allow_sdxl_controlnet`) is
exercised on SD1.5 and SDXL. Post-processing sweeps every known upscaler and face-fixer at 512²,
1024², and a VRAM-derived maximum. Alchemy is tested on both lanes independently — the CLIP lane
(caption/interrogation/NSFW, on the safety process) and the graph lane (upscalers/face-fixers/
strip-background, on the inference processes) — plus a concurrent-with-image rung.

The report separates **Capabilities** (everything the worker proved it can do) from a **conservative
recommended bridgeData** (only models that fit with VRAM headroom are loaded, the batch size is the
largest that passed cleanly, and concurrent alchemy is enabled only if the soak held up).

Every suggested value carries a **provenance basis** so you can tell a setting that is off because it
was *tested and failed* from one that is off only because its level was *skipped* (never tested),
*not in this run*, or *held back* for VRAM headroom or an unstable soak. The basis is printed under the
completion line, written into the report's "Why each value" table, and shown beside each value in the
dashboard. A built-in consistency check flags (and never silently ships) any recommendation that would
enable a capability on anything weaker than a real pass.

Key `ramp` options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--tiers` | `sd15,sdxl` | Comma-separated model tiers (`sd15`, `sdxl`, `flux`, `qwen`). `flux`/`qwen` are opt-in: they are very large (17-20 GB download, 13-16 GB VRAM), the run warns and auto-skips them when the machine cannot hold them or the checkpoint is absent, and `qwen` is a beta model sourced from the pending reference (needs `HORDE_MODEL_REFERENCE_PRIMARY_API_URL`; the beta opt-in env is set automatically). |
| `--process-mode {fake,dry_run,real}` | `real` | `real` benchmarks the GPU; `fake`/`dry_run` exercise the ramp without inference. |
| `--out PATH` | `benchmark_results/<timestamp>` | Output directory. |
| `--jobs-per-level N` | `4` | Jobs run per ramp level. |
| `--level-timeout SECONDS` | `900` | Per-level timeout. |
| `--warm` / `--no-warm` | on | Reuse one warm worker across fixed-scenario levels instead of cold-starting a fresh worker (and respawning every inference process) per level. Feature and alchemy levels pre-warm their models (one throwaway job/form) before being measured, so the one-time cold load of a controlnet/QR checkpoint, upscaler, or BLIP model is not counted against the level; the measured pass reflects steady state. `--no-warm` runs each level in its own isolated subprocess. |
| `--resume` | off | Reuse existing level results in `--out`. |
| `--no-validate` | off | Skip the post-ramp sustained-load soak (`--soak-minutes` sets its length). |
| `--force` | off | Attempt levels that would otherwise be skipped for not fitting this machine (insufficient VRAM/disk) or lacking a CivitAI token. An absent checkpoint is still skipped (there is nothing to run). |

Other toggles narrow the run. The coarse stage flags drop a whole stage: `--no-concurrency`,
`--no-features`, `--no-alchemy` (plus `--only-level`, `--skip-downloads`, `--include-downloads`).
For finer control, `--exclude-axis AXIS` (repeatable) drops one individual capability while leaving its
stage siblings in place, so you can benchmark, say, post-processing without controlnet. The axes are
`queue_size`, `threads`, `batch` (concurrency); `hires_fix`, `post_processing`, `controlnet`,
`qr_code` (features); and `alchemy_clip`, `alchemy_graph`, `alchemy_concurrent` (alchemy). A level is
built only if its stage is included *and* its axis is not excluded.

### `plan`: preview requirements before you run

The benchmark keeps every scenario identical across machines (apples-to-apples), so what changes from
machine to machine is only *whether* a level runs. `plan` makes that decision visible up front, without
starting a worker: it builds the same ladder `ramp` would, then prints one row per level with its
estimated VRAM, the disk it needs free, whether it needs network or a CivitAI token, and the predicted
verdict (`RUN`, or `SKIP` with the reason) against the detected hardware.

```bash
# What would run if I benchmarked just sd15 and sdxl on this box?
horde-benchmark plan --tiers sd15,sdxl

# Machine-readable rows (the TUI's "Preview plan" button uses this):
horde-benchmark plan --tiers flux --json
```

`plan` accepts the same selection flags as `ramp` (`--tiers`, `--process-mode`, `--no-concurrency`,
`--no-features`, `--no-alchemy`, `--exclude-axis`, `--include-downloads`, `--force`), so the preview
matches exactly what the ramp would run. The same plan table is also printed
at the top of every `ramp` (emitted on the progress channel as a `RampPlanned` event), so `monitor` and
the dashboard show it too. Pass `--force` to see levels that do not fit (or lack a token) reported as
`RUN` instead of `SKIP`.

## Environment variables

### Install and runtime

| Variable | Effect |
|----------|--------|
| `HORDE_WORKER_DIR` | Install location for the one-line installer. |
| `HORDE_WORKER_BACKEND` | Force a PyTorch build: `cu126`, `cu130`, `cu132`, `rocm`, or `cpu` (default: detected). |
| `HORDE_WORKER_NO_SHORTCUTS` | Skip creating Desktop/Start Menu shortcuts. |
| `HORDE_WORKER_NO_LAUNCH` | Skip auto-launching the dashboard after install. |
| `HORDE_WORKER_ENABLE_LONG_PATHS` | Opt in to Windows system-wide long-path support (changes an HKLM setting; needs administrator). |
| `HORDE_WORKER_ROCM_TORCH` | Override the ROCm torch version installed by `update-runtime-rocm`. |
| `CUDA_VISIBLE_DEVICES` | Pin a worker to a specific GPU (see [Run multiple GPUs](../how-to/run-multiple-gpus.md)). |

### Behaviour

| Variable | Effect |
|----------|--------|
| `HORDE_WORKER_NO_UPDATE_CHECK` | Disable the dashboard's background release check. |
| `AIWORKER_LIMITED_CONSOLE_MESSAGES` | Cap console verbosity at level 2. |
| `AIWORKER_REGEN_ENABLE_TELEMETRY` | Opt in to OpenTelemetry tracing (off by default; see [Telemetry](../explanation/telemetry.md)). |

Worker configuration itself can be supplied through `AIWORKER_*` variables instead of
`bridgeData.yaml` when you pass `-e`; this is the container path described in
[Run headless](../how-to/run-headless.md#configure-from-environment-variables-containers).
