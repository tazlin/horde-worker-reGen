# Use the dashboard (`horde-worker`)

`horde-worker` is an optional [Textual](https://textual.textualize.io/) frontend that launches and
supervises the worker and shows its live state. It runs in your terminal or, unchanged, in a web
browser. The headless `run_worker` path is untouched and remains the right choice for unattended,
containerised, and remote deployments (see [Run headless](run-headless.md)); the dashboard is a
convenience layer on top.

## Launch it

The launcher scripts offer three peer interfaces:

```bash
# Windows
horde-worker.cmd            # web dashboard in your browser (default)
horde-worker.cmd --terminal # the dashboard in this terminal (no browser)
horde-worker.cmd --headless # no UI: run the worker in the foreground, printing to this console

# Linux / macOS
./horde-worker.sh
./horde-worker.sh --terminal
./horde-worker.sh --headless
```

The window you launch from is the worker. In the in-terminal and headless modes, closing that window
(or pressing `Ctrl+C`) stops the worker. In the default browser mode the worker runs in a persistent
background **host**, so closing the browser tab leaves it running and reopening the dashboard
reconnects. See [Closing and reattaching](#closing-and-reattaching) for exactly what keeps the worker
alive, how to reattach to a running one, and the Windows tray icon.

`--headless` is the no-UI path: it downloads/verifies your models and then runs the worker directly
(equivalent to `run_worker`; see [Run headless](run-headless.md)), so a server or service sees the
worker's own log output rather than a UI. On a machine with no graphical display, the default
(browser) mode detects that a browser cannot be opened and automatically falls back to the in-terminal
dashboard; with no terminal either, it tells you to use `--headless` (or `--host` to serve a browser
on another machine).

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
| **Overview** | Headline metrics (jobs submitted/faulted, queue depth, GPU duty cycle, kudos/hr), health, and a **Now / Next / Why** strip that explains the scheduler's current intent. Job-owned facts live in a **Work ledger** (state, progress, model, process/GPU target, size, and reason); press `J` to collapse recently finished rows into a one-line count while keeping in-progress work visible. Worker identity/model settings and enabled alchemy are visible in normal mode. The process table stays process-owned (slot state, resident model, GPU, memory, heartbeat, completed count). Trends show their configured time window and can be cycled across 5m / 15m / 30m / 60m / 120m / All; changing the window filters retained session history rather than resetting it. Each finite window is rendered as fixed buckets from `now - window` to `now`, so a warming-up worker shows early empty buckets instead of shrinking the graph to only the samples seen so far. Config changes and explicit soft resets mark the trends as stabilizing. Multi-GPU workers also get a compact per-card strip (one row per GPU: VRAM bar, contexts, active jobs). In details density, the status block stays pinned above the scroll body; Health and Now / Next / Why form the left column, while GPUs + Job pipeline and Trends form the wider right column once the terminal is wide enough. 80-column terminals stay stacked. |
| **Stats** | Session statistics owned by the worker: submitted/faulted jobs, kudos/hr, GPU duty, recoveries, slowdowns, no-work time, pipeline depth, and alchemy totals when alchemy is enabled. The tables roll finalized image jobs up by model and by baseline, with jobs, MPxsteps (`width x height / 1,000,000 x steps x batch`), sampling time, end-to-end time, and batch>1 job counts. The **JSONL export** button toggles session-scoped stats export under `.horde_worker_regen/stats/`; export is off by default, rotates at 5 MiB per file, and warns once retained stats JSONL files exceed 50 MiB. Retained files can be compressed or downsampled with `horde-stats` or the importable `stats_operations` helpers. |
| **GPUs** | A per-card breakdown for multi-GPU operators (a single-GPU host shows one collapsed card): each GPU's VRAM headroom with a near-OOM pressure flag, its inference contexts against their target, throughput (combined it/s and a jobs/hr trend), its concurrency ceiling and a busy-context duty proxy, and — in the details density (`F6`) — the whole-card residency it is holding and any models gone *locally unservable* on that card. |
| **Live** | One panel per inference process: state (and its temperature phrase, e.g. *sampling*, *primed*, *loading*), current model and job, a sampling progress bar, iterations/second, VRAM/RAM (current and peak), and heartbeat freshness. |
| **Downloads** | Model download progress, with pause/resume and an optional bandwidth cap. |
| **Control** | Worker lifecycle and lower-frequency controls: start/stop, local pause/resume, auto-start on launch, restart, and horde-side maintenance. |
| **Logs** | Tail `logs/bridge.log` or any subprocess `logs/bridge_n.log`, with level and substring filters. These are the same files the worker already writes (see [Logs](../reference/logs.md)). |
| **Config** | A form over `bridgeData.yaml`, grouped by section with inline help, enforced bounds, and masked secrets. Comments and untouched keys are preserved. Includes a dedicated models editor and a searchable model picker: search across name/description/tags/triggers, filter by baseline/SFW/NSFW/inpainting and by on-disk status, sort by any column, see which list (load/skip) each model is already in, and inspect a model's full record before adding it. |
| **Insights** | Live, actionable recommendations (low GPU duty cycle, VRAM pressure, fault rate, idle time, configuration mismatches) and a recent-activity rollup. |
| **Benchmark** | A guided, plan-first flow: **Preview plan** shows what each level needs and what will run on this machine (no GPU), **Run benchmark** measures it, and **Apply suggested config** writes the recommendation. Model tiers are individual toggles (sd15/sdxl on by default; flux/qwen opt-in). Advanced options are collapsed by default with inline explanations, and each capability is separately selectable: queue depth / thread count / batch size, hires-fix / post-processing / controlnet / QR-code, and the alchemy CLIP / graph / concurrent lanes, so you can measure exactly the features you run. The suggested config shows per-setting **provenance** (proven / untested / failed / capped), and **History** browses and compares past runs. For the full benchmark CLI, see [`horde-benchmark`](../reference/cli.md#horde-benchmark). |

The Config tab offers *Reload from disk*, *Save*, and *Save + restart worker*. A plain *Save* is enough
to apply most changes: the running worker watches `bridgeData.yaml` and hot-reloads it on its own. Only
fields marked with ⟳ need a restart, which is what *Save + restart worker* is for.

Saving only writes the fields you actually changed: values you never touched (including settings the
form merely shows at their default) are left exactly as they are on disk, so a fresh Save adds nothing
surprising and a no-op Save changes nothing at all. If an edited value is out of range, the save is
blocked, every problem is listed at once, and the editor jumps to the first offending field; a value
that was already invalid on disk but that you did not touch will not block an unrelated change.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `F3` | Start / stop the worker without quitting |
| `F6` | Cycle dashboard density: normal, details, thin |
| `F7` | Pause / resume model downloads |
| `F11` | Restart the worker process |
| `M` | Toggle horde-side maintenance |
| `T` | Cycle the Overview trend window: 5m, 15m, 30m, 60m, 120m, All |
| `R` | Soft-reset the Overview trend epoch |
| `Ctrl+Q` / `Ctrl+C` | Stop the worker and quit |

## Closing and reattaching

Browser mode (the default) splits the dashboard from the worker: a persistent **host** process owns
the worker, and each browser tab is just a viewer attached to it over a loopback socket. That is why a
closed tab does not stop the worker, but it also makes "is it still running?" a fair question. What
each kind of close does:

| You close... | In this mode | The worker... |
|--------------|--------------|---------------|
| The browser tab | browser (default) | keeps running on the host; reopen to reconnect |
| The launcher window, cleanly | browser (default) | stops (the launcher tells the host to drain and exit) |
| The launcher window, hard-killed | browser (default) | keeps running, now orphaned (use the tray icon or `--stop`) |
| The terminal, or `Ctrl+C` | `--terminal` / `--headless` | stops |

### Reattach to a running worker

- **Browser:** run `horde-worker` again. It detects the running host and opens a fresh dashboard tab
  attached to it (it does not start a second worker).
- **Terminal:** `horde-worker --terminal --attach` attaches an in-terminal dashboard to the running
  host (defaults to `127.0.0.1:7717`; pass `--attach HOST:PORT` for another). Closing it detaches
  without stopping the worker.
- **Inspect or stop from a terminal, with no UI:**

  ```bash
  horde-worker-web --status   # is a worker host running here, and is its worker working?
  horde-worker-web --stop     # ask it to drain in-flight jobs and exit cleanly
  ```

  `--status` exits non-zero when nothing is running, so scripts can branch on it.

### The tray icon (Windows)

On Windows the worker host shows a **system-tray icon** while it runs, so a worker is never invisible,
even after the browser and the launcher window are gone. Its menu offers:

- **Open dashboard** -- reopen the browser dashboard attached to this worker (reusing a running web
  server if there is one).
- **Stop worker & exit** -- drain in-flight jobs, then stop the worker and host cleanly. A still-running
  launcher window notices the host going away and closes itself too, so nothing is left lingering.

The line at the top of the menu shows whether the worker is currently running. The icon is the simplest
way to find and stop an orphaned worker after a hard-closed launcher window; the `--stop` command above
does the same from a terminal. (Linux and macOS have no tray icon yet; use `--status` / `--stop`.)

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
| `--attach [HOST:PORT]` | Attach to a running worker host instead of owning the worker; the worker survives this session closing. With no value, attaches to `127.0.0.1:7717`. |
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
commands (pause/resume, maintenance, restart). State publishing never blocks the worker's control
loop, so a slow or closed UI can never stall job processing. In browser mode a separate host process
owns the worker and the served dashboard attaches to it over a socket, which is why closing the tab
leaves the worker running.

See also: [Frontend and durable state](../explanation/frontend_and_state.md) for the supervisor
channel, served/attached modes, and persisted state; [Architecture](../explanation/architecture.md);
and [IPC and messaging](../explanation/ipc_and_messaging.md).
