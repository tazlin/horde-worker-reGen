# Use the dashboard (`horde-worker`)

`horde-worker` is an optional [Textual](https://textual.textualize.io/) frontend that launches and
supervises the worker and shows its live state. It runs in your terminal or, unchanged, in a web
browser. The headless `run_worker` path is untouched and remains the right choice for unattended,
containerised, and remote deployments (see [Run headless](run-headless.md)); the dashboard is a
convenience layer on top.

## Launch it

The launcher scripts pick the interface for you:

```bash
# Windows
horde-worker.cmd            # opens the dashboard in your web browser (default)
horde-worker.cmd --terminal # runs the dashboard in this terminal instead

# Linux / macOS
./horde-worker.sh
./horde-worker.sh --terminal
```

The window you launch from is the worker: closing it (or pressing `Ctrl+C` in it) stops the worker.
In browser mode the worker runs in a persistent background host, so closing the browser tab leaves
the worker running and you can reopen the dashboard to reconnect.

To try the whole interface without a GPU, models, or an API key, run a synthetic worker:

```bash
horde-worker --process-mode fake
```

## First run: the setup wizard

The first time you start a real worker whose `bridgeData.yaml` is not yet filled in, the dashboard
opens a guided setup wizard. It collects your API key and worker name, lets you pick which models to
serve, and offers to run a benchmark to tune your settings. When you finish you can start the worker
straight away, start a benchmark first, or stay stopped and start later with `F3`. Existing installs
whose config is already complete skip the wizard. The wizard is also skipped for the synthetic worker
and for env-var config (`-e`), both of which are power-user paths.

After you start, your selected models download in the background and the dashboard switches to the
**Downloads** tab so you can watch progress. On first run this can take 30 to 60 minutes depending on
your selection and connection. The worker serves each model as soon as it finishes, so keep the
window open.

## Tabs

| Tab | What it shows |
|-----|---------------|
| **Overview** | Headline metrics (jobs submitted/faulted, queue depth, GPU duty cycle, kudos/hr), worker identity, and a per-process summary. |
| **Live** | One panel per inference process: state, current model and job, a sampling progress bar, iterations/second, VRAM/RAM (current and peak), and heartbeat freshness. |
| **Downloads** | Model download progress, with pause/resume and an optional bandwidth cap. |
| **Logs** | Tail `logs/bridge.log` or any subprocess `logs/bridge_n.log`, with level and substring filters. These are the same files the worker already writes (see [Logs](../reference/logs.md)). |
| **Config** | A form over `bridgeData.yaml`, grouped by section with inline help, enforced bounds, and masked secrets. Comments and untouched keys are preserved. Includes a dedicated models editor and a searchable model picker (filter by baseline/SFW/NSFW/inpainting; inspect a model's full record before adding it). |
| **Insights** | Live, actionable recommendations (low GPU duty cycle, VRAM pressure, fault rate, idle time, configuration mismatches) and a recent-activity rollup. |
| **Benchmark** | A guided, plan-first flow: **Preview plan** shows what each level needs and what will run on this machine (no GPU), **Run benchmark** measures it, and **Apply suggested config** writes the recommendation. Model tiers are individual toggles (sd15/sdxl on by default; flux/qwen opt-in). Advanced options are collapsed by default with inline explanations, and each capability is separately selectable: queue depth / thread count / batch size, hires-fix / post-processing / controlnet / QR-code, and the alchemy CLIP / graph / concurrent lanes, so you can measure exactly the features you run. The suggested config shows per-setting **provenance** (proven / untested / failed / capped), and **History** browses and compares past runs. For the full benchmark CLI, see [`horde-benchmark`](../reference/cli.md#horde-benchmark). |

Saving on the Config tab gives you three choices: *Save*, *Save and apply* (hot-reload), and *Save and
restart worker*. Fields that only take effect after a restart are marked as such.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `F2` | Pause / resume popping new jobs (in-flight jobs finish) |
| `F3` | Start / stop the worker without quitting |
| `F4` | Toggle whether the worker auto-starts when the dashboard launches |
| `F5` | Reload `bridgeData.yaml` on the worker |
| `F7` | Pause / resume model downloads |
| `F8` | Jump to the Benchmark tab |
| `F9` | Restart the worker process |
| `Ctrl+Q` / `Ctrl+C` | Stop the worker and quit |

## Command-line options

These are options for the `horde-worker` program itself. The wrapper-script flags above (`--terminal`,
`--host`) are handled before this program runs.

| Flag | Meaning |
|------|---------|
| `--process-mode {real,fake}` | `real` runs the GPU worker (default); `fake` runs a synthetic worker. |
| `-e`, `--load-config-from-env-vars` | Configure the worker from `AIWORKER_*` env vars instead of `bridgeData.yaml`. |
| `-n`, `--worker-name NAME` | Override the worker name. |
| `--amd`, `--amd-gpu` | Enable AMD GPU optimisations. |
| `--config PATH` | Path to the `bridgeData.yaml` the config editor reads and writes (default `bridgeData.yaml`). |
| `--no-auto-restart` | Do not relaunch the worker if it crashes. |
| `--attach HOST:PORT` | Attach to a running worker host instead of owning the worker (used by the web launcher); the worker survives this session closing. |
| `--directml N` | Select a DirectML device index. DirectML is currently unavailable (see [Run on AMD ROCm](run-on-amd-rocm.md)), so this flag has no working backend at present. |

When it owns the worker, the dashboard relaunches it automatically if it crashes (bounded by a restart
budget, which `--no-auto-restart` disables) and stops it cleanly on exit.

## Serve the dashboard over the web yourself

Because it is a Textual app, the same UI can be served over the web. The launcher does this for you in
the default (browser) mode. To bind the dashboard to your LAN, pass `--host`:

```bash
./horde-worker.sh --host 0.0.0.0
```

The served dashboard is unauthenticated, so only do this on a trusted network. Run the serve command
on the worker host; it serves that host's worker.

## How it works

When the dashboard owns the worker, it spawns the worker as a child process and talks to it over a
duplex pipe, with no on-disk state file. The worker pushes compact state snapshots and accepts control
commands (pause/resume, reload config, restart). State publishing never blocks the worker's control
loop, so a slow or closed UI can never stall job processing. In browser mode a separate host process
owns the worker and the served dashboard attaches to it over a socket, which is why closing the tab
leaves the worker running.

See also: [Frontend and durable state](../explanation/frontend_and_state.md) for the supervisor
channel, served/attached modes, and persisted state; [Architecture](../explanation/architecture.md);
and [IPC and messaging](../explanation/ipc_and_messaging.md).
