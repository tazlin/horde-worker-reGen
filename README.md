# AI Horde Worker reGen

Share your GPU with the world. Earn [kudos](https://github.com/Haidra-Org/haidra-assets/blob/main/docs/kudos.md). Generate AI images faster.

The [AI Horde](https://aihorde.net/) is a free, open, decentralized platform where anyone can contribute GPU power to generate images. When your worker completes jobs, you earn **kudos**; the more you have, the faster your own image requests get processed.

## Quick Start

> **Prerequisites**: an NVIDIA GPU (8 GB+ VRAM recommended) and a free [AI Horde API key](https://aihorde.net/register). You do **not** need git or Python installed; the installer fetches everything it needs, including its own Python and PyTorch.

### Install

**Windows** — use `winget` (most trusted), or paste the one-liner into PowerShell:

```powershell
winget install Haidra.HordeWorker
```

```powershell
irm https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/install.ps1 | iex
```

**Linux** — paste into a terminal:

```bash
curl -LsSf https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/install.sh | sh
```

The installer downloads the worker, builds its environment (the first run pulls Python and PyTorch and can take several minutes), then opens the **dashboard in your browser**. A short wizard walks you through:

1. Entering your API key and choosing a worker name
2. Picking which models to serve (a sensible default is chosen for your GPU)
3. Optionally running a benchmark to auto-tune your settings
4. Starting the worker

After you click **Start**, your chosen models download in the background (shown on the **Downloads** tab). The first run can take 30-60 minutes depending on your selection and connection; the worker serves each model as it finishes, so keep the window open. That's it, you're contributing to the horde. Re-run the same install command any time to update.

> **Windows SmartScreen**: the raw `irm … | iex` one-liner may show "Windows protected your PC". Click **More info → Run anyway** (the same step used to install tools like `uv`). `winget install` avoids the prompt entirely.

> **Tip**: keep the install path free of spaces. The installer picks a safe default; override it with `$env:HORDE_WORKER_DIR` (Windows) or `HORDE_WORKER_DIR` (Linux).

> **What the installer does (and doesn't)**: it installs into a per-user folder (no administrator rights needed), downloads Python, PyTorch, and your chosen models (several GB) into that folder and your per-user package cache, and adds per-user "AI Horde Worker" shortcuts. It does **not** change any system-wide settings. Opt out of the shortcuts with `HORDE_WORKER_NO_SHORTCUTS`, or skip the auto-launch with `HORDE_WORKER_NO_LAUNCH`.

### Opening it again later

The worker runs as long as the launcher window is open: closing that window (or pressing Ctrl+C in it) stops the worker. Closing just the dashboard window or browser tab leaves the worker running, so you can reopen it to reconnect. To start the worker again after it has been stopped:

- **Windows**: click the **AI Horde Worker** shortcut the installer added to your Desktop and Start Menu (or, if you installed with winget, run `horde-worker` in a terminal). You can also run `horde-worker.cmd` in the install folder.
- **Linux**: launch **AI Horde Worker** from your applications menu, or run `./horde-worker.sh` in the install folder.

The first run does the slow one-time setup; reopening after that is quick. On a return visit the worker remembers your settings and skips the setup wizard.

### Power-user and headless options

- **In-terminal UI instead of the browser**: `horde-worker.cmd --terminal` (Windows) or `./horde-worker.sh --terminal` (Linux).
- **Remote / LAN access to the dashboard**: `horde-worker.cmd --host 0.0.0.0`. This binds all interfaces and the dashboard is unauthenticated, so only do this on a trusted network.
- **Fully unattended (no UI)**: edit `bridgeData.yaml`, then run `horde-bridge.cmd` (Windows) or `./horde-bridge.sh` (Linux).

### Manual install (git or zip)

<details>
<summary>Prefer to clone or download a zip?</summary>

```bash
git clone https://github.com/Haidra-Org/horde-worker-reGen.git
cd horde-worker-reGen
```

No git? Download the [latest zip](https://github.com/Haidra-Org/horde-worker-reGen/archive/refs/heads/main.zip) and extract it (use a path without spaces).

Then run `horde-worker.cmd` (Windows) or `./horde-worker.sh` (Linux): it installs dependencies on first run and opens the dashboard. Or use the non-interactive scripts: `update-runtime.cmd` to install, copy `bridgeData_template.yaml` to `bridgeData.yaml` and fill in your details, then `horde-bridge.cmd` to run.
</details>

## Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [GPU-Specific Setup](#gpu-specific-setup)
- [Running the Worker](#running-the-worker)
- [Updating](#updating)
- [Custom Models](#custom-models)
- [Docker](#docker)
- [Support & Troubleshooting](#support--troubleshooting)
- [Architecture & developer docs](#architecture--developer-docs)

## Configuration

### Basic Settings

If you used the interactive launcher, your config was created automatically. To edit it later, open `bridgeData.yaml` in any text editor.

If you're setting up manually:

1. Copy `bridgeData_template.yaml` to `bridgeData.yaml`.
2. Set your `api_key` (from [aihorde.net/register](https://aihorde.net/register)). **Keep this secret.**
3. Set a unique `dreamer_name`. If it's already taken, you'll get a "Wrong Credentials" error.

### Recommended Settings by GPU (Click to expand)

<details>
<summary><strong>24 GB+ VRAM</strong> (RTX 3090, 4090)</summary>

```yaml
queue_size: 1        # <32 GB RAM: 0, 32 GB: 1, >32 GB: 2
safety_on_gpu: true
high_memory_mode: true
high_performance_mode: true
unload_models_from_vram_often: false
max_threads: 1       # 2 is often viable for xx90 cards
post_process_job_overlap: true
max_power: 64        # Reduce if max_threads: 2
max_batch: 8         # Increase if max_threads: 1, decrease if max_threads: 2
allow_sdxl_controlnet: true
```
</details>

<details>
<summary><strong>12–16 GB VRAM</strong> (RTX 3080 Ti, 4070 Ti, 4080)</summary>

```yaml
queue_size: 1        # <32 GB RAM: 0, 32 GB: 1, >32 GB: 2
safety_on_gpu: true  # Consider false if using Cascade/Flux
moderate_performance_mode: true
unload_models_from_vram_often: false
max_threads: 1
max_power: 50
max_batch: 4         # Or higher
```
</details>

<details>
<summary><strong>8–10 GB VRAM</strong> (RTX 2080, 3060, 4060, 4060 Ti)</summary>

```yaml
queue_size: 1        # <32 GB RAM: 0, 32 GB: 1, >32 GB: 2
safety_on_gpu: false
max_threads: 1
max_power: 32        # No higher
max_batch: 4         # No higher
allow_post_processing: false  # If using SDXL/Flux, else can be true
allow_sdxl_controlnet: false
```

Minimize other VRAM-consuming apps while the worker runs.
</details>

<details>
<summary><strong>Lower-end GPUs / Under-performing workers</strong></summary>

- `extra_slow_worker: true`: gives more time per job, but requesters must opt-in. Only use if consistently under 0.3 MPS/s or 3000 kudos/hr with correct config.
- `limit_max_steps: true`: caps total steps per job based on model type.
- `preload_timeout: 120`: allows longer model load times.
</details>

<details>
<summary><strong>Systems with less than 32 GB RAM</strong></summary>

- Set `queue_size: 0` and stick to SD 1.5 models only.
- Set `load_large_models: false`.
- Add `ALL SDXL`, `ALL SD21`, and the unpruned models to `models_to_skip`.
</details>

### Hardware Tips

- **Use an SSD.** HDDs are too slow for multiple models; limit to one model with <60 s load time.
- **Configure 8 GB+ swap** (16 GB+ preferred), even on Linux.
- **Keep `max_threads` ≤ 2** unless you have a 48 GB+ VRAM data center GPU.
- **Disable sleep/power-saving** while the worker runs.
- SDXL needs ~9 GB free RAM (32 GB+ total recommended). Flux/Cascade need ~20 GB free RAM (48 GB+ total recommended).

## GPU-Specific Setup

### NVIDIA (default)

No extra steps. The standard scripts and the interactive launcher default to CUDA.

### AMD (ROCm) (Linux only)

AMD support is **experimental** and Linux-only.

- Use `update-runtime-rocm.sh` and `horde-bridge-rocm.sh` instead of the standard versions.
- [WSL support](README_advanced.md#advanced-users-amd-rocm-inside-windows-wsl) is highly experimental.
- Join the [AMD discussion on Discord](https://discord.com/channels/781145214752129095/1076124012305993768) for help.

### DirectML (Windows)

DirectML is **temporarily unavailable**: its PyTorch build is incompatible with the current torch version.
Windows AMD/Intel users without CUDA should run on Linux (ROCm) for now. This note will be updated when a
compatible DirectML build returns.

## Running the Worker

### Starting

> The worker is resource-intensive. Avoid gaming or other heavy tasks while it runs.

**Recommended**: run `horde-worker.cmd` (Windows) or `./horde-worker.sh` (Linux). By default this opens the **dashboard in your browser** (served locally via `textual serve`). On first run it launches the setup wizard; after that it shows a live overview, a per-process view, logs, a configuration editor, downloads, benchmarking, and recommendations.

Browser mode runs the worker in a persistent background host, so **closing the browser tab leaves the worker running**; closing the launcher window stops it. Reopen the dashboard any time to reconnect.

Prefer a terminal? Add `--terminal` for the in-terminal UI. You can try the dashboard without a GPU using `horde-worker --process-mode fake`. See [Worker TUI](docs/worker_tui.md) for details.

**Alternative (headless)**: `horde-bridge.cmd` / `./horde-bridge.sh` (or the `-rocm` variant) runs the worker with no UI. The headless `run_worker` path is unchanged.

### Stopping

Press `Ctrl+C` in the worker's terminal. It will finish any in-progress jobs before exiting.

### Logs

Logs are saved in the `logs/` directory:

| File | Contents |
|------|----------|
| `bridge.log` | Main log (all info) |
| `bridge_n.log` | Per-process log |
| `trace.log` | Errors and warnings only |
| `trace_n.log` | Per-process errors |

### Multiple GPUs

> Future versions will not require multiple worker instances.

For now, run one worker per GPU. On Linux:

```bash
CUDA_VISIBLE_DEVICES=0 ./horde-bridge.sh -n "GPU-0"
CUDA_VISIBLE_DEVICES=1 ./horde-bridge.sh -n "GPU-1"
```

Running multiple workers needs high RAM (32–64 GB+). `queue_size` and `max_threads` multiply memory use.

## Updating

Stay up to date via our [Discord](https://discord.gg/3DxrhksKzn). Script names below assume Windows + NVIDIA; for Linux use `.sh`, for AMD use `-rocm` variants.

1. **Stop** the worker (`Ctrl+C`, or Quit in the dashboard).
2. **Update**, matching how you installed:
   - One-line installer: re-run the same `irm … | iex` (Windows) or `curl … | sh` (Linux) command; it updates in place.
   - winget: `winget upgrade Haidra.HordeWorker`.
   - Git users: `git pull`, then `update-runtime.cmd` (or the relevant variant).
   - Zip users: download the [latest zip](https://github.com/Haidra-Org/horde-worker-reGen/archive/refs/heads/main.zip), extract over the existing folder, then `update-runtime.cmd`.
3. **Start** the worker again.

> **Antivirus note**: Some antivirus (e.g. Avast) may interfere with downloads. If you see `CRYPT_E_NO_REVOCATION_CHECK` errors, temporarily disable it.

## Custom Models

Serving custom models requires the `customizer` role; request it on [Discord](https://discord.gg/3DxrhksKzn).

With the role:

1. Download your model files locally.
2. Add them to `bridgeData.yaml`:

   ```yaml
   custom_models:
     - name: My Custom Model
       baseline: stable_diffusion_xl
       filepath: /path/to/model/file.safetensors
   ```

   Supported baselines: `stable_diffusion_1`, `stable_diffusion_2_768`, `stable_diffusion_2_512`, `stable_diffusion_xl`, `stable_cascade`, `flux_1`.

   > **Warning**: Only Flux.schnell models are allowed. Flux.dev and its derivatives are **not** permitted.

3. Add the model `name` to your `models_to_load` list.

Custom model names can't conflict with existing horde model names. The horde treats them as SD 1.5 for kudos and safety purposes.

## Docker

Docker images: <https://hub.docker.com/r/tazlin/horde-worker-regen/tags>

See the [Docker guide](Dockerfiles/README.md) for setup instructions.

## Support & Troubleshooting

**Get help**: [#local-workers on Discord](https://discord.com/channels/781145214752129095/1076124012305993768) or [open an issue](https://github.com/Haidra-Org/horde-worker-reGen/issues).

| Problem | Fix |
|---------|-----|
| Download failures | Check disk space and internet connection. |
| "Path too long" / file-not-found during install (Windows) | Use a short install path (the default already is). If it persists, opt in to system-wide long-path support: set `$env:HORDE_WORKER_ENABLE_LONG_PATHS=1` before installing (this changes an HKLM setting and needs administrator). |
| Job timeouts | Remove large models (Flux, Cascade, SDXL), lower `max_power`, disable post-processing/controlnet/lora. |
| Out of memory | Lower `max_threads`, `max_batch`, or `queue_size`. Close other programs. |
| Less kudos than expected | New workers have 50% of job kudos and 100% of uptime kudos held in escrow for ~1 week until you become trusted. |
| Worker in maintenance mode | Log into [artbot](https://tinybots.net/artbot/settings?panel=workers) with the worker running and click "unpause". Check [logs](logs/README.md) for ERROR entries to find the root cause. |

For advanced setup options (manual `uv` usage, custom environments, etc.), see [README_advanced.md](README_advanced.md).

## Architecture & developer docs

Want to understand how the worker works under the hood, or contribute a change?

- **[Documentation home](docs/index.md)**: start here.
- **[Architecture overview](docs/architecture.md)**: what runs where, the process model, and IPC.
- **[Codebase Map](docs/navigation.md)**: a file→responsibility quick reference and program entry points.
- **[Job Lifecycle](docs/job_lifecycle.md)**: traces a job from pop to submit.
- **[Contributing](CONTRIBUTING.md)**: development setup and guidelines.

**Where things live (top level):**

| Path | What it is |
|------|------------|
| `run_worker.py` | Worker entry point |
| `horde_worker_regen/process_management/` | Main- and child-process orchestration (the core) |
| `horde_worker_regen/bridge_data/` | Configuration loading and models |
| `bridgeData.yaml` | Your worker configuration |
| `docs/` | Full documentation: see the [Codebase Map](docs/navigation.md) |

## Model Usage & Licenses

Many bundled models use the [CreativeML OpenRAIL License](https://huggingface.co/spaces/CompVis/stable-diffusion-license). Please review it before use.
