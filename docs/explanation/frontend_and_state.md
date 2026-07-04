# Frontend and Durable State

- [Frontend and Durable State](#frontend-and-durable-state)
    - [Two ways to run the worker](#two-ways-to-run-the-worker)
    - [The supervisor channel](#the-supervisor-channel)
    - [Terminal, served, and attached modes](#terminal-served-and-attached-modes)
    - [Worker-owned stats history](#worker-owned-stats-history)
    - [The first-run wizard](#the-first-run-wizard)
    - [Worker identity preflight](#worker-identity-preflight)
    - [Durable app state](#durable-app-state)
    - [See also](#see-also)

The inference engine described in the rest of these docs is headless. On top of
it sits an optional supervising **frontend** (the dashboard most users see) and a
small amount of **durable state** the application remembers between runs. This
page describes both and the channel that connects them to the worker.

## Two ways to run the worker

| Entry point | Console script | Role |
| ----------- | -------------- | ---- |
| `run_worker.py` | `run_worker` | The headless worker. Unchanged by the frontend; this is what the TUI launches under the hood and what servers/automation run directly. |
| `tui/app.py` | `horde-worker` | The Textual dashboard. Launches and supervises the worker as a **child process** and renders its live state. |

The headless path is fully self-sufficient; the TUI is purely additive. See
[Run headless](../how-to/run-headless.md) and
[Use the dashboard](../how-to/use-the-dashboard.md).

## The supervisor channel

A supervising frontend launches the worker as a child and holds one end of a
duplex pipe. [`supervisor_channel.py`][horde_worker_regen.process_management.ipc.supervisor_channel]
defines the structured protocol over it:

- The worker pushes
  [`WorkerStateSnapshot`][horde_worker_regen.process_management.ipc.supervisor_channel.WorkerStateSnapshot]
  objects at a steady cadence (the same data the overview, per-process view, and
  Downloads tab render), including an `orchestration_intent` summary (what the
  scheduler/popper is doing next and why), a `work_ledger` of active/recent job
  state (including post-processing stage rows and each active image job's current pop-order),
  post-processing lane counters, a `SystemMemorySnapshot` (machine total/available
  RAM plus per-role worker RSS) and a `per_card` list of
  [`CardSnapshot`][horde_worker_regen.process_management.ipc.supervisor_channel.CardSnapshot]
  (one per driven GPU: VRAM headroom, inference contexts, whole-card residency, and
  per-card fault/unservable-model health) that the GPUs tab and the Overview per-card
  strip render. Each `ProcessSnapshot` also carries the `device_index` of the card its
  slot is pinned to. A single-GPU host reports exactly one `CardSnapshot`. The snapshot also carries
  worker-owned stats data: the latest one-second `StatsSample`, bounded stats-history backfill for
  reconnecting frontends, model/baseline `StatsRollupRow` tables, and `StatsExportState` for the JSONL
  export toggle and disk-size warning. The snapshot is versioned by `SUPERVISOR_PROTOCOL_VERSION`
  (currently 16) so a frontend can detect a mismatch with a worker built from different code.
- The worker drains
  [`SupervisorControlMessage`][horde_worker_regen.process_management.ipc.supervisor_channel.SupervisorControlMessage]
  commands each loop tick (start/stop intent, download pause/resume and rate
  limit, stats JSONL export enable/disable, etc.). Server-side maintenance is reported separately from local pause: the
  dashboard shows horde maintenance as a distinct **MAINTENANCE** phase, labels the
  API connectivity row as maintenance instead of disconnected, and treats a pending
  Maintenance (horde) command as active until the worker-details poll confirms it or
  a later successful job pop proves the horde is sending work again. The worker-side
  maintenance latch follows the same rule: a real popped job clears the latch immediately
  and suppresses any stale worker-details `maintenance=True` cache until the poll catches up.

This mirrors the worker's own internal IPC (see
[IPC and Messaging](ipc_and_messaging.md)) and is the structured upgrade of the
[`.abort` sentinel](shutdown_and_faults.md#the-abort-file) external-supervision
hook. The models are deliberately pure-data and JSON-round-trippable: the default
transport is a `multiprocessing` pipe (pickle), but the same models serialize
cleanly for the localhost-socket fallback the launcher uses in served mode.

## Worker-owned stats history

The dashboard renders trend graphics, but the worker owns the underlying statistics samples. During snapshot
construction the process manager appends at most one `StatsSample` per second from counters it already has in
memory: submitted/faulted jobs, kudos/hr, GPU duty, queue and in-progress counts, no-work time, process
recoveries, slowdowns, and alchemy totals. Finalized image jobs update incremental model and baseline rollups
inside `WorkerRunMetrics`, so the Stats tab does not recompute those tables from the full job list on every
frame. Alchemy forms remain in run metrics but are excluded from the image model/baseline tables.

Reconnects receive a `StatsHistoryBackfill`: exact recent samples for the largest finite trend window plus a
decimated all-session series. Consumers still bucket and render locally. Finite trend windows are interpreted
as fixed spans from `now - window` to `now`; empty early buckets render as no activity, which keeps a 5m or
60m graph visually spanning the selected duration even while the worker is warming up. Changing the selected
window is only a view change over retained history; it does not move the trend epoch. `All` spans from the
worker session start to now, while the explicit reset shortcut starts a new display epoch.

The Stats tab can toggle worker-side JSONL export for the current session. Export writes typed `stats_sample`
and `job_completed` events under `.horde_worker_regen/stats/`, uses version-and-session-stamped filenames,
rotates at 5 MiB, and only warns when retained stats files exceed 50 MiB. IO failures disable export and appear
in `StatsExportState`; they do not affect worker operation. Retained files can be operated on later via
`horde_worker_regen.stats_operations`: compressing older JSONL files to `.jsonl.gz`, or downsampling
`stats_sample` events to a caller-selected interval while preserving finalized-job events. The `horde-stats`
CLI uses the same functions.

## Terminal, served, and attached modes

The same app runs in a terminal or in a browser, with one important wrinkle:
`textual-serve` runs a fresh TUI subprocess per browser session, so the worker
cannot live inside any one session. The frontend therefore has two supervisor
implementations behind a common `SupervisorLike` interface:

- **Owning supervisor** (`worker_launcher.WorkerSupervisor`): spawns and owns the
  worker directly. Used in terminal mode.
- **Attached supervisor** (`attach.AttachedWorkerSupervisor`): connects to a
  persistent [`WorkerHost`][horde_worker_regen.tui.worker_host.WorkerHost] over a
  localhost socket, reflects its streamed snapshots, and forwards commands. Used
  in served/browser mode.

In served mode (`tui/web.py`, the default for non-technical users) a single
`WorkerHost` owns one worker independently of any browser session, so closing a
browser tab detaches the client but **leaves the worker running**. Network
exposure is conservative: the web server and the worker host both bind
`127.0.0.1` by default; binding the LAN is a deliberate power-user action
(`--host` / `HORDE_WORKER_WEB_HOST`) that exposes an unauthenticated dashboard.

The host's lifetime is decoupled from the launcher that started it. `tui/web.py`
spawns a host only when one is not already listening, and on a *clean* exit it
sends `LIFECYCLE_SHUTDOWN` so the host drains and stops the worker. Two cases
break that tidy ownership: a launcher that is *hard*-killed (the window's close
button or `taskkill`) skips that shutdown and orphans the host, and a host
started directly (`horde-worker-host`) has no launcher to stop it at all. In both,
the worker keeps running with nothing on screen. Two affordances keep it
discoverable and stoppable: `horde-worker-web --status` / `--stop` (the same
status frame and `LIFECYCLE_SHUTDOWN` the host already speaks), and, on Windows, a
**system-tray icon** the host itself shows (`tui/tray.py`). The tray lives on the
host rather than the launcher precisely because the host is what survives, so an
orphaned worker surfaces as a visible icon with *Open dashboard* and *Stop*
actions instead of an invisible process. The tray is best-effort and
import-guarded (`pystray`/`Pillow`, Windows-only), so its absence never affects
the worker.

The coupling runs the other way too. The launcher tells the host to stop on its
own clean exit, but the host can also exit *first*: the tray's *Stop worker &
exit* ends the host directly, and a host can crash. A launcher blocked in
`textual-serve`'s `serve()` cannot otherwise notice that, so it would keep serving
a dead host as exactly the kind of invisible orphaned console this whole design
fights. So the launcher holds a **liveness leash**: a background thread keeps a
connection to the host's control socket and, the moment that socket drops (a clean
close, an explicit `host_shutdown` farewell frame, or a reset), winds the launcher
down. It first reaps the per-session TUI subprocesses `textual-serve` spawned so
none of *them* orphan, then exits. The socket is the authoritative signal,
immune to the pid-reuse hazard a pid-file leash would carry, and it works whether
this launcher spawned the host or merely attached to a pre-existing one.

Discoverability is not enough on its own: a worker the host spawns is a child
process tree (the worker and its own inference/safety processes), and on Windows
a child outlives its parent, so a host that *itself* dies the hard way would
leave that tree resident on the GPU with nothing left to stop it. Two guards make
the tree's lifetime track the host's. First, the supervisor binds the worker to a
Windows **Job Object** created with kill-on-close (`tui/job_object.py`); because
the host holds the only handle and a job member's children join the job
automatically, the OS terminates the whole tree the instant the host process
ends, however it ends. Second, the host records the worker pid it owns in a
dedicated registry
([`OwnedProcessRegistry`][horde_worker_regen.process_management.lifecycle.owned_process_registry.OwnedProcessRegistry],
in `host_owned_pids.json`) and, on startup, reaps any tree a previous host
orphaned before serving (`reap_orphans_from_previous_run(kill_tree=True)`). The
job object is the immediate guarantee; the registry sweep is the backstop for
when it could not apply (a job-assignment that lost the spawn race, or a host from
an older build). Both verify process identity against pid reuse, and both are
best-effort and Windows-centric, so neither can wedge startup.

## The first-run wizard

On first launch, when `bridgeData.yaml` is unconfigured
([`is_setup_incomplete`][horde_worker_regen.tui.wizard.is_setup_incomplete]), the
TUI shows a guided, linear setup wizard. It collects the two things a worker
cannot run without (an API key and a unique worker name) plus an initial model
selection, writes them through the same light YAML path the config editor uses,
and hands off to the benchmark/start flow. It reuses existing controls (the same
model picker the Config tab uses) and never blocks the dashboard: cancelling
leaves the worker stopped and the tabs available for manual configuration.

## Guarding unsaved config edits

Config edits live only in the form widgets until **Save** writes them to
`bridgeData.yaml` (a running worker then hot-reloads the file on its own). Because
switching tabs does not destroy the form, leaving the Config tab does not lose the
edits outright, but it is an easy way to forget to save them. The app therefore
gates navigation away from a *dirty* Config tab: Textual switches the tab before
the app sees it, so the guard snaps back to Config and shows a modal offering to
**leave** (keep the edits live in the form), **discard** (revert the form to
disk), **stay**, or **never** warn again for the rest of the session. Dirty
detection is a best-effort comparison of the raw widget values against a baseline
captured on mount/save/reload, so a malformed in-progress entry never raises and a
detection glitch can never trap the operator on the tab. The "never" choice is
intentionally session-scoped (not persisted): it is a per-sitting convenience, not
a durable preference.

## Worker identity preflight

Worker names are unique horde-wide and tied to the API key that first registers
them, and each worker *type* (the image "dreamer" and the alchemy "alchemist")
registers as a separate, uniquely-named worker. Getting this wrong otherwise
surfaces only as a late, cryptic "Wrong credentials to submit as this worker" at
pop time. [`worker_identity.py`][horde_worker_regen.process_management.config.worker_identity]
fails fast *before* any process spawns:

1. A **local** check (no network): names must not be the reserved template
   defaults, and the alchemist name must differ from the dreamer name when
   alchemy is enabled.
2. A **network** check: each enabled name must be either unregistered (a
   brand-new worker) or already owned by the configured API key. The name is
   resolved through the single-worker-by-name endpoint, not the all-workers list:
   the list only returns workers that are currently *active*, so an idle worker
   registered under the name would be invisible there and a collision would slip
   past the check. The endpoint's `WorkerNotFound` response is the one signal read
   as "name is free"; every other error is treated as a failure to verify. Per the
   chosen policy this hard-fails on any such failure, including the API being
   unreachable (after a small bounded retry), so the worker never silently runs
   under a name the horde will reject.

## Durable app state

[`app_state.py`][horde_worker_regen.app_state] is the structured, on-disk
counterpart to the in-memory [`WorkerState`](architecture.md#the-shared-state-pattern):
it records what the application needs to remember *between* invocations: the last
benchmark and where its results live, the last worker run, the last-known-good
settings, which worker version last ran (so a version bump can mark a stale
benchmark for re-running), and the operator's durable UI preferences (the Overview
density mode, the trend window, and which Overview panels are hidden). A hidden-panel
key that no longer names a live element is dropped on load, so a stale preference can
never block the Overview from rendering.

The store lives in a grouped working-directory folder
(`.horde_worker_regen/state.json`), alongside `bridgeData.yaml`, `logs/`,
`benchmark_results/`, and the [action ledger / owned-PID
registry](resilience_and_recovery.md). Reads never raise (a missing or corrupt
file yields fresh state, so it cannot block startup) and writes are atomic. The
module is dependency-light so it can be imported early in startup and by the TUI.

## See also

- [Use the dashboard](../how-to/use-the-dashboard.md): the dashboard from a
  user's perspective
- [Model Downloads and Availability](model_downloads.md): the Downloads tab and
  the controls the supervisor channel carries
- [Resilience and Recovery](resilience_and_recovery.md): the owned-PID registry
  and action ledger that share the `.horde_worker_regen/` state directory
- [Telemetry](telemetry.md): the separate observability layer
