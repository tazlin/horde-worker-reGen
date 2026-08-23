# Use the dashboard (`horde-worker`)

`horde-worker` launches the worker and shows you what it is doing, either in a browser tab or in your
terminal, from the same command. It is optional: on a server, in a container, or anywhere unattended,
[run headless](run-headless.md) instead.

In brief:

- `horde-worker.cmd` (Windows) or `./horde-worker.sh` opens the dashboard in a browser and leaves the
  worker running when you close the tab. `--terminal` draws it in the terminal instead, and
  `--headless` runs the worker with no UI at all.
- A worker whose `bridgeData.yaml` is not filled in gets a setup page on first launch, so you never
  have to hand-edit YAML to start.
- `F3` starts and stops the worker, `?` lists every key, and `Ctrl+P` searches every tab and command.
- The dashboard opens in its **Simple** level. Advanced and Developer add detail to the same tabs.
- Binding the dashboard to anything other than loopback hands full control of your worker to anyone
  who can reach the port. Read [what a network-bound dashboard exposes](#what-a-network-bound-dashboard-exposes)
  before you do it.

## Before you start

You need the worker installed ([install](install.md)) and, to earn kudos, an
[AI Horde](https://aihorde.net/) API key. Models are not required up front: the setup page picks them
and downloads them for you.

To try the whole interface with no GPU, no models, and no API key, run a synthetic worker:

```bash
horde-worker --process-mode fake
```

## Launch it

The launcher scripts offer three peer interfaces:

```bash
# Windows
horde-worker.cmd            # web dashboard in your browser (default)
horde-worker.cmd --terminal # the dashboard in this terminal (no browser)
horde-worker.cmd --headless # no UI: the worker runs in the foreground, printing to this console

# Linux / macOS
./horde-worker.sh
./horde-worker.sh --terminal
./horde-worker.sh --headless
```

| Mode | Where the UI is | What closing it does |
|------|-----------------|----------------------|
| default | A browser tab, served from `127.0.0.1:8000` | The worker keeps running in a background host. Reopen to reconnect. |
| `--terminal` | The terminal you launched from | Stops the worker. |
| `--headless` | No UI. The worker prints to the console | Stops the worker. |

`--headless` is the same path as `run_worker`: it verifies your models, then runs the worker directly,
so a service or a log collector sees the worker's own output. See [Run headless](run-headless.md).

On a machine with no graphical display, the default mode notices that no browser can be opened and
falls back to the in-terminal dashboard. With no terminal either, it tells you to use `--headless`, or
`--host` to serve the dashboard to another machine.

You should see the status bar name your worker and show a lifecycle badge within a few seconds. If the
worker is not started yet, press `F3`.

## Set up a new worker

The first launch of a worker whose `bridgeData.yaml` is not yet filled in opens the **Getting started**
page. It says what the AI Horde is and what a worker needs (a public worker name, an API key so your
kudos are kept, and models on disk), then collects all three inline.

Pick one of three presets. Each is a model selection plus a stance on which kinds of work you accept:

| Preset | Models | Accepts |
|--------|--------|---------|
| **Essentials** | The single most-requested model. Smallest download. | Post-processing (upscaling, face fixing). No LoRA, no ControlNet. |
| **Recommended** | The most-requested models your card can hold. | Post-processing and LoRA work. LoRA files download as jobs ask for them. |
| **Showcase** | Adds larger SDXL models. Largest download. | Post-processing, LoRA, and ControlNet. |

Each preset states what it downloads and how much room is left where models are kept. A preset that
does not fit stays visible but cannot be chosen, and says how much more space it needs. **Choose my own
models instead** opens the same model picker the Config tab uses.

The LoRA presets also ask for an optional **Civitai token**. Some Civitai downloads refuse anonymous
requests, and a token from a free account gets those. Leaving the field empty never removes a token you
already have.

Saving writes only the settings this page owns and leaves the rest of `bridgeData.yaml` alone. Press
`F3`, or **Start contributing** on the home screen, when you are ready.

Verify it worked: the **Downloads** tab lists your selected models with live progress. A first run of
30 to 60 minutes is normal, depending on the preset and your connection. The worker serves each model
as it finishes, so leave the window open. Once one is done, the Overview shows a job in flight.

To change any of this later, the **Getting started** action stays on the Simple home screen. Existing
installs with complete config are never sent to the page, and it is skipped entirely for the synthetic
worker and for env-var config (`-e`).

## Choose how much detail you see

| Level | What it is for |
|-------|----------------|
| **Simple** (default) | Contributing without needing to understand the worker: plain wording, live per-request progress, and the settings most contributors change. |
| **Advanced** | The full operator surface: queues, per-process state, scheduler behaviour, download detail, and the whole configuration. |
| **Developer** | Advanced plus the worker's internal safety levers: hung-process timeouts, the VRAM and RAM budgets, and the fault breakers. Entering it asks you to confirm once. |

Change level on the Config tab's **Dashboard** page, or press `Ctrl+P` and search for the level. The
choice is remembered between runs.

Every tab exists at every level. The level changes how much each tab shows, never which tabs you have,
so anything you learn to find in Simple is in the same place in Developer.

Simple holds back three things:

- The Config tab's tuning pages, whose settings only mean something beside the measurements that would
  justify changing them. Simple keeps Dashboard, Essentials, Models, Content, Features, Alchemy, and
  LoRA & Downloads.
- The GPU table's tuning columns.
- The three shortcuts that act on the Advanced Overview (`C`, `H`, `F6`), since Simple shows a
  different view.

Developer adds the settings the worker uses to police itself: the timeouts that decide when a process
counts as hung, the VRAM and RAM budgets the arbiter enforces, whole-card residency, and the breakers
for post-processing faults, unservable models, and self-maintenance. Set carelessly, these remove a
protection without saying so, or trip on healthy work. That is why they sit behind a level you choose
on purpose.

Settings you cannot see are still preserved. Saving from any level writes back everything already in
`bridgeData.yaml`, so moving between levels never rewrites a tuned config.

### The Simple home screen

Simple's Overview opens with a line naming what the worker is doing, then which worker this is: its
name and version, the horde account it contributes for, how long this session has run, what it offers
requesters, how many models it serves, and how many requests it takes at once. Under that sit the
session totals, requests completed and kudos earned, with the hourly rate the worker measures while it
works. The rate reads as unknown rather than zero until there is enough to measure.

The small chart under each total is a rate rather than the total drawn again: it shows how much
finished in each slice of the last fifteen minutes, so a worker that stops earning flattens to the
baseline within a couple of minutes. Below it are the requests in flight with their progress, and the
last few finished.

One card appears only when something is off: a health finding, maintenance holding new requests back,
the worker waiting after repeated trouble reaching the horde, or processes it restarted on its own this
session. A healthy worker shows no such card, so seeing one is itself the signal.

### Appearance

The Config tab's **Dashboard** page also chooses a theme (**Horde Dark**, **Horde Light**, or
**Terminal colours**, which follows your terminal's own 16-colour palette and suits low-colour or
high-contrast setups) and a spacing density for the Advanced and Developer surfaces.

## The tabs

| Tab | What it answers |
|-----|-----------------|
| [**Overview**](#overview) | What is the worker doing right now, and is anything wrong? |
| [**Stats**](#stats) | What has this session produced, by model and by baseline? |
| **GPUs** | How is each card doing: VRAM headroom, contexts, throughput, and duty? |
| [**Live**](#live) | What is each worker process doing, and where is the RAM going? |
| **Downloads** | How far along are the model downloads? Pause, resume, or cap the bandwidth here. |
| **Control** | Start, stop, pause, restart, auto-start on launch, and horde-side maintenance. |
| **Logs** | What did the worker write? Tail any log with level and substring filters. |
| [**Config**](#config) | Everything in `bridgeData.yaml`, as a form. |
| [**Insights**](#insights) | What should I change, given what this worker has measured? |
| [**Benchmark**](#benchmark) | What can this machine actually sustain? |

**GPUs** shows one collapsed card on a single-GPU host, and flags a card under near-OOM pressure.
Details density (`F6`) adds the whole-card residency each card holds and any models gone locally
unservable on it. **Logs** reads the same `logs/bridge.log` and `logs/bridge_n.log` files the worker
already writes; see [Logs](../reference/logs.md) for what is in them.

The descriptions below are the Advanced presentation. Simple shows plain-language equivalents on
Overview, Live, and Downloads, and puts a one-line explanation above the other tabs' widgets.

On Stats, Control, Logs, and Insights, a **What these numbers say right now** panel appears whenever
the live figures are worth a comment: a share of requests faulting, restarts the worker made on its
own, failures in a row, a session spent mostly waiting for work. Each sentence quotes your worker's own
numbers and says what to do, if anything. It is absent when there is nothing to say, so its presence is
the signal. Fold it away with its ▼ once you have read it.

### Overview

The Overview carries the headline metrics (jobs submitted and faulted, queue depth, GPU duty cycle,
kudos/hr), the **Health** checklist, and a **Now / Next / Why** strip stating what the scheduler is
doing and what it is waiting on. Panel titles carry counts (`Health · 2 models`,
`Processes · 3/4 alive · 2 hot`, `Queue · 5 pending · 1,234 MP`, `GPUs · 2 cards · 41% duty`), so the
scale of things reads without expanding a row.

Two tables split the same worker along different axes. The **Work ledger** is job-owned: one row per
job, listed in the order the worker will serve them. The process table is process-owned: slot state,
resident model, GPU, memory, heartbeat, and completed count.

A **Governance** panel consolidates the pop governors and the scheduler's RAM and preload diagnostics;
its title says how many governors are actively holding work back. Multi-GPU workers also get a
per-card strip (one row per GPU: VRAM bar, contexts, active jobs).

#### Reading a work-ledger row

Each row names the job's pop order, age, stage, model, the process and GPU it is dispatched to, and its
progress. `J` collapses finished rows into a one-line count while keeping in-progress work visible.

The **Size** cell states everything that decides how much work a job is, and a legend under the table
names its parts:

| Part | Example | Meaning |
|------|---------|---------|
| resolution | `832×1216` | The image size the request asks for. |
| steps | `28s` | Denoising steps. |
| batch | `n4` | Images per job. Absent when the job asks for one. |
| sampler | `dpmpp_sde` | The sampler the request names. |
| order | `²` | Model evaluations one step costs. Absent for a first-order sampler. |
| cost | `2.22×` | Measured per-step cost against `k_euler`, published by `horde_sdk`. |

An adaptive sampler shows neither an order nor a cost, since it chooses its own iteration count from
the requested steps.

Alchemy forms appear in the ledger and the Queue as their own rows, with the form where a model name
goes (prefixed `⚗`) and the source image's resolution as the size. In pipeline-disaggregation mode the
ledger also carries a per-job stage line (prefixed `Disagg:`) showing where each in-flight job sits in
its pipeline and which process holds its current stage.

#### Trends

`T` cycles the trend window across 5m, 15m, 30m, 60m, 120m, and All; `R` resets the buffers without
touching session totals. Changing the window filters retained history rather than resetting it, and
each finite window draws fixed buckets from `now - window` to `now`, so a warming-up worker shows empty
early buckets instead of a graph that only covers the samples it has. Config changes and explicit
resets mark the trends as stabilizing.

**Kudos/hr** is an active rate: kudos earned in the window divided by the time the pipeline actually
held work. It therefore reads steadily instead of sawtoothing on each submit, and it charges neither
the cold start before your first job nor an idle or maintenance stretch. A pause still counts while it
drains queued work, since kudos keep landing. The session figure in the hero line and the status bar is
the same measure over the whole session. An alchemist worker gets a **Forms/hr** row beside Jobs/hr.

If the GPU is near-idle while a job is in flight, the Trends border and its GPU-duty row turn orange
with a `(!)`. On a multi-GPU worker the row states each card's own duty and the alert names the card it
means, because the headline figure is a mean across the driven cards.

#### Layout and customization

The Overview is built from individually hideable panels. `C` opens the customize overlay (`A` hides
all, `N` shows all, `Esc` saves), and the choice persists across restarts. `H` temporarily reveals
everything you have hidden, so you can glance at a demoted panel without editing the layout again.
Nothing is lost by hiding: the per-role RAM breakdown lives on the **Live** tab, and health and overall
RAM are always in the status bar.

The layout follows the width in every density. Below about 100 columns everything stacks in one column,
and tables shed their least-important columns rather than truncating (an 80-column terminal is
first-class). At about 100 columns Health sits beside GPUs and the job pipeline. At about 165 columns
the worker, alchemy, and residency panels spread three-up.

### Stats

Session statistics the worker owns: submitted and faulted jobs, kudos/hr, GPU duty (each card's own
figure beside it on a multi-GPU worker), recoveries, slowdowns, no-work time, pipeline depth, and
alchemy totals when alchemy is on.

The tables roll finalized image jobs up by model and by baseline, carrying jobs, MPxsteps
(`width x height / 1,000,000 x steps x batch`), sampling time, end-to-end time, and how many jobs ran a
batch above one. An alchemist worker also gets a per-form table: kind (graph or CLIP), forms completed,
faults, average and total pop-to-submit time, and peak VRAM.

**JSONL export** toggles session-scoped stats export under `.horde_worker_regen/stats/`. It is off by
default, rotates at 5 MiB per file, and warns once retained files exceed 50 MiB. Compress or downsample
them with `horde-stats` or the importable `stats_operations` helpers.

### Live

A **Scheduling** strip, a **Worker RAM by role** panel, then one panel per inference process.

The strip shows RAM-governor status and the latest preload-admission decision. Details density (`F6`)
adds the active pop hold or pause, the reclaim target, and the gate reason.

**Worker RAM by role** breaks the worker's resident RAM into inference, the component, VAE,
post-processing, and utilities lanes, safety, orchestrator, and download, with the machine-wide figure
in its title and everything else left as a genuine *Other (OS + apps)* remainder.

Each process panel shows its state and temperature phrase (*sampling*, *primed*, *loading*), current
model and job, a sampling progress bar, iterations per second, VRAM and RAM (current and peak), and
heartbeat freshness. A GPU-bearing process holding model components in RAM also lists them under
**Resident** (name, kind, approximate MB) with a **Retained** line for the remainder its component
cache does not account for. In pipeline-disaggregation mode the pinned sampler shows the same per-step
bar, and the VAE lane shows one during a tiled decode.

### Config

A form over `bridgeData.yaml`, grouped by operator workflow, with inline help, enforced bounds, masked
secrets, and checkboxes for the alchemy forms. Comments and keys you never touch are preserved.

The tab offers *Reload from disk*, *Apply preset*, *Save*, and *Save + restart worker*. A plain *Save*
is enough for most changes: the running worker watches `bridgeData.yaml` and hot-reloads it. Only
fields marked ⟳ need a restart.

Saving writes only the fields you changed. Values you never touched, including settings the form shows
at their default, are left exactly as they are on disk, so a no-op Save changes nothing at all. An
out-of-range value blocks the save, lists every problem at once, and jumps to the first offending
field. A value that was already invalid on disk and that you did not touch never blocks an unrelated
change.

The tab also holds a models editor and a searchable model picker: search across name, description,
tags, and triggers, filter by baseline, by SFW/NSFW/inpainting, and by on-disk status, sort by any
column, mark models for the load or skip lists, and inspect a model's full record. Clear actions
confirm before dropping staged work.

#### Deciding whether to turn the model pool on

The **Model pool** sub-tab controls a trade-off worth deciding on purpose. Left off (the default), the
worker advertises its normal eligible model set. Turned on, it commits to a small set of logical seats
and shapes its pops toward them. That can cut swap time when model churn is the bottleneck, at the cost
of serving a narrower mix. Demand still decides which jobs arrive, so the gain is not guaranteed.

Turn it on when the worker either sits idle waiting for rare-model demand or records model-swap churn.
The Insights tab reads the measured swap counter rather than inferring churn from model variety.

**Demand-following pool preset (50 GB admission)** is the one-switch aggressive version; the individual
knobs give tighter control. The form shows the effective values inherited from the switch. Editing one
makes it an explicit override, while saving the switch alone leaves them inherited.

- `seats` at or below your inference-process count gives every seat the possibility of a resident home
  process.
- The demand `ranker` fills the seats you do not pin.
- The rotation and dwell windows decide how fast the seat set re-contests.
- Pins are a validated `{name, affinity}` YAML list, edited in the same tab. Affinity biases seating
  and rotation. A pin excluded by your load or skip rules is refused.

Verify the result on the Insights **Model pool** panel: a recent resident match proves that pop avoided
a cold load, while rising empty pops mean the horde has little demand for that model on this card. To
undo, switch the pool off. The worker returns to advertising its normal eligible set on the next
config reload.

### Insights

Live recommendations drawn from what this worker measured: low GPU duty cycle, VRAM pressure, fault
rate, idle time, configuration mismatches, and model-pool guidance, plus a recent-activity rollup.

The **Model pool** panel is always present. With the pool off it reads `Model pool: off` and notes the
throughput-versus-variety trade, and a recommendation appears only after measured `model_swap` churn.
With the pool on it lists each seat, its measured readiness (`resident`, `cold`, or `empty`), dwell,
last pop-match age and whether that match was resident, empty pops, rescue countdown, the most recent
routed lane, the demand-snapshot age, and the benched models. It flags stale demand and seats that keep
taking empty pops.

### Benchmark

A plan-first flow: **Preview plan** shows what each level needs and what will run on this machine (no
GPU required), **Run benchmark** measures it, and **Apply suggested config** writes the recommendation.

Model tiers are individual toggles, with sd15 and sdxl on by default and flux and qwen opt-in. Advanced
options are collapsed with inline explanations, and each capability is separately selectable: queue
depth, thread count, and batch size; hires-fix, post-processing, controlnet, and QR-code; and the
alchemy CLIP, graph, and concurrent lanes. Measure exactly the features you run.

The suggested config shows per-setting provenance (proven, untested, failed, or capped), and
**History** browses and compares past runs. For the command-line equivalent, see
[`horde-benchmark`](../reference/cli.md#horde-benchmark).

## The status bar

A one-line status bar sits above the tabs and stays visible everywhere. It leads with the worker's
lifecycle phase as a coloured badge and a health summary (the worst outstanding check, or an `N/N ok`
tally), then the live vitals: the job pipeline (`q▸inf▸post▸saf▸sub` when post-processing is enabled),
GPU duty, system RAM, kudos/hr, jobs done and faulted, and the worker name. On a narrow terminal the
lowest-priority segments drop from the right rather than wrap, so the phase and health are never pushed
off the line.

## Keyboard shortcuts

The bar at the bottom shows as many shortcuts as the terminal is wide enough to hold. The full list is
always two keys away: `?` for help, or `Ctrl+P` for the command palette, which lists every tab and
shortcut by name with its key beside it. Both stay visible at 80 columns.

| Key | Action |
|-----|--------|
| `F3` | Start or stop the worker without quitting |
| `?` | Help for the level you are using, and the complete shortcut list |
| `Ctrl+P` | Command palette: jump to any tab, run any shortcut, or change level |
| `F6` | Cycle dashboard density: normal, details, thin |
| `C` | Open the customize-layout overlay for the Overview panels |
| `H` | Reveal all hidden Overview panels (press again to re-hide) |
| `J` | Collapse or show finished rows in the Work ledger |
| `F7` | Pause or resume model downloads |
| `F11` | Restart the worker process |
| `M` | Toggle horde-side maintenance |
| `T` | Cycle the Overview trend window: 5m, 15m, 30m, 60m, 120m, All |
| `R` | Reset the Overview trend buffers (view-only; session totals keep running) |
| `Ctrl+Q` / `Ctrl+C` | Stop the worker and quit |

## Keep the worker running when you close the dashboard

Browser mode splits the dashboard from the worker: a persistent **host** process owns the worker, and
each browser tab is a viewer attached to it over a loopback socket. That is why a closed tab does not
stop the worker, and it is also why "is it still running?" is a fair question.

| You close | In this mode | The worker |
|-----------|--------------|------------|
| The browser tab | browser (default) | Keeps running on the host. Reopen to reconnect. |
| The launcher window, cleanly | browser (default) | Stops. The launcher tells the host to drain and exit. |
| The launcher window, hard-killed | browser (default) | Keeps running, now orphaned. Use the tray icon or `--stop`. |
| The terminal, or `Ctrl+C` | `--terminal` / `--headless` | Stops. |

### Reattach to a running worker

- **Browser:** run `horde-worker` again. It finds the running host and opens a fresh tab attached to
  it, rather than starting a second worker.
- **Terminal:** `horde-worker --terminal --attach` attaches an in-terminal dashboard to the running
  host, defaulting to `127.0.0.1:7717`. Pass `--attach HOST:PORT` for another. Closing it detaches
  without stopping the worker.
- **No UI at all:**

  ```bash
  horde-worker-web --status   # is a worker host running here, and is its worker working?
  horde-worker-web --stop     # drain in-flight jobs and exit cleanly
  ```

  `--status` exits non-zero when nothing is running, so scripts can branch on it.

### The tray icon (Windows)

The worker host shows a system-tray icon while it runs, so a worker is never invisible even after the
browser and the launcher window are gone. The line at the top of its menu says whether the worker is
running. The menu offers:

- **Open dashboard**: reopen the browser dashboard attached to this worker, reusing a running web
  server if there is one.
- **Stop worker & exit**: drain in-flight jobs, then stop the worker and host cleanly. A still-running
  launcher notices the host going away and closes its own dashboard sessions. Your browser and its
  other tabs are left alone.

The icon is the simplest way to find and stop an orphaned worker after a hard-closed launcher window.
`horde-worker-web --stop` does the same from a terminal. Linux and macOS have no tray icon yet; use
`--status` and `--stop`.

## Serve the dashboard to another machine

The default mode binds `127.0.0.1:8000`, so only this machine can reach it. To reach the dashboard from
elsewhere, bind an address other than loopback:

```bash
./horde-worker.sh --host 0.0.0.0 --port 8000
```

Run this on the machine the worker runs on: it serves that machine's worker.

The address and port come from the first of these that is set:

| Source | Example |
|--------|---------|
| The launcher flags | `--host 0.0.0.0 --port 8080` |
| Environment variables | `HORDE_WORKER_WEB_HOST=0.0.0.0`, `HORDE_WORKER_WEB_PORT=8080` |
| `bridgeData.yaml` | `dashboard_web_host: 0.0.0.0`, `dashboard_web_port: 8080` |

The `bridgeData.yaml` keys are on the Config tab's **Dashboard** page at the Advanced level or above,
so the binding survives between launches without a flag. The dashboard launcher reads them, not the
worker, and they take effect the next time you start the dashboard. To go back to loopback, clear them
and launch without `--host`.

### What a network-bound dashboard exposes

**There is no authentication and no encryption.** Anyone who can reach the port can start and stop your
worker, change every setting, and read your logs. Bind the network only on a network you trust, and
never expose the port to the internet.

When the bind is not loopback, the dashboard withholds the **API key** and **Civitai token** fields
from the config editor, so a visitor can neither read nor replace your credentials. That is the only
protection it adds. It does not cover:

- The keys themselves, which stay in `bridgeData.yaml` on the worker machine.
- Anything already written to the logs, which the Logs tab shows in full.
- The **Getting started** page's key fields on a worker that is not yet configured. They are masked as
  you type but not withheld, because withholding them would block first-run setup.

The launcher prints this same summary whenever it binds a non-loopback address, so the exposure is
never silent.

### The lightweight `/native` page

`http://<worker-machine>:<dashboard-port>/native` is a browser-native status page built from ordinary
responsive HTML rather than the terminal canvas, which makes it the practical quick-check surface on a
phone. It shows worker identity and uptime, lifecycle and maintenance state, pipeline depth, session
job and kudos totals, GPU duty, active models, recent horde messages, active-job stage and progress,
per-process liveness, model, VRAM and heartbeat state, and alchemy totals when enabled. Its **At a
glance** sentence leads with whatever best explains what the worker is doing now.

Its controls are deliberately few: start, graceful stop, local pause/resume, and horde maintenance.
**Pause** stops this worker from accepting new work locally; **Horde maintenance** changes the worker's
advertised state at the horde. They are independent. The **Full terminal dashboard** link goes to
everything else.

**Glance view** keeps lifecycle controls, four headline metrics, the pipeline, active jobs, and all
process states in one viewport. On a phone the page itself does not scroll: metrics form a 2×2 grid,
the pipeline stays on one row, and jobs and processes use compact cards that become a two-column matrix
as the inventory grows. The preference is remembered in that browser, and `/native?view=glance`
requests it directly for a home-screen shortcut.

The page polls the same host the full browser dashboard uses, so refreshing or closing it neither
restarts the worker nor resets session statistics. It omits config, credentials, logs, and snapshot
internals, but it is not an authentication boundary: its controls are unauthenticated, and the full
dashboard is still at `/` on the same port. The trusted-network rule above applies to both.

### On a phone

Bind the network as above, then open `http://<worker-machine>:<dashboard-port>` on the device (port
8000 unless you configured another). Prefer [`/native`](#the-lightweight-native-page) for a
conventional mobile overview and the core controls; use the terminal view when you need the full
operator surface.

Three things adapt on their own:

- **Text size.** The page sizes the terminal's text to give the dashboard enough columns to lay itself
  out, targeting about 52 columns and never shrinking below 12px. Append `?fontsize=N` to override it:
  `?fontsize=14` for larger text and fewer columns, `?fontsize=8` only if you would rather have columns
  than readability.
- **Typing.** The dashboard is painted into a terminal canvas, so the browser cannot tell a tab from a
  text field. Touch keyboard input is therefore off by default, and tapping around does not summon the
  software keyboard. Tap the keyboard button in the bottom dock when you mean to type into the selected
  field, and again to return to navigation-only taps.
- **Layout.** Below the 80-column terminal floor, cards lose their side padding and borders, action
  bars become two-column touch grids, form rows stack, buttons and tabs gain taller touch targets, tab
  labels shorten, dialogs clamp to the viewport, Logs wrap, and tables shed down to the columns that
  identify a row. The terminal follows the browser's visual viewport, so the bottom stays reachable as
  the address bar moves and while the keyboard is open.

Gestures and the dock:

- Swipe vertically anywhere in the terminal to scroll; you do not need to catch the narrow scrollbar.
- Swipe sideways over the main or Config tab strip to move to the adjacent tab. The gesture locks to
  its dominant axis after a short movement, so a mostly vertical drag still scrolls. Two-finger pinch
  zoom stays available.
- The 52-pixel dock below the terminal holds three controls: `☰` opens the command palette, the
  up/down triangle hides or restores the tab strip without changing page, and the keyboard button
  enables typing.

Two rough edges worth knowing:

- Benchmark **History** scrolls sideways rather than shedding columns.
- Modifier shortcuts, including `Ctrl+P`, stay awkward with a software keyboard.

## Command-line options

These are options for the `horde-worker` program. The wrapper-script flags above (`--terminal`,
`--host`) are handled before it runs.

| Flag | Meaning |
|------|---------|
| `--process-mode {real,fake}` | `real` runs the GPU worker (default); `fake` runs a synthetic worker. |
| `-e`, `--load-config-from-env-vars` | Configure the worker from `AIWORKER_*` env vars instead of `bridgeData.yaml`. |
| `-n`, `--worker-name NAME` | Override the worker name. |
| `--amd`, `--amd-gpu` | Enable AMD GPU optimisations. |
| `--config PATH` | The `bridgeData.yaml` the config editor reads and writes (default `bridgeData.yaml`). |
| `--no-auto-restart` | Do not relaunch the worker if it crashes. |
| `--attach [HOST:PORT]` | Attach to a running worker host instead of owning the worker, so the worker survives this session closing. Defaults to `127.0.0.1:7717`. |
| `--remote-exposed` | Treat this session as reachable from other machines, withholding the credential fields from the config editor. The web launcher sets it for you when it binds a non-loopback address. |
| `--directml N` | Select a DirectML device index. DirectML has no working backend at present (see [Run on AMD ROCm](run-on-amd-rocm.md)). |

When it owns the worker, the dashboard relaunches it after a crash, bounded by a restart budget that
`--no-auto-restart` disables, and stops it cleanly on exit.

Stop, Restart, Save + restart, and dashboard exit are non-blocking: the UI stays responsive while
accepted work drains. Restart stays labelled **Restarting** and launches the replacement only after the
previous worker's PID has exited, so two worker trees never overlap on the GPU. Pressing quit again
during a graceful exit escalates to an immediate process-tree kill.

## How it works

When the dashboard owns the worker, it spawns the worker as a child process and talks to it over a
duplex pipe, with no on-disk state file. The worker pushes compact state snapshots and accepts control
commands. Publishing never blocks the worker's control loop, so a slow or closed UI cannot stall job
processing.

In browser mode a separate host process owns the worker and the served dashboard attaches to it over a
socket. Uptime, cumulative totals, and the trend-history backfill therefore continue across browser
sessions; only a real worker restart begins a new session.

See also: [Frontend and durable state](../explanation/frontend_and_state.md) for the supervisor
channel, served and attached modes, and persisted state; [Architecture](../explanation/architecture.md);
and [IPC and messaging](../explanation/ipc_and_messaging.md).
