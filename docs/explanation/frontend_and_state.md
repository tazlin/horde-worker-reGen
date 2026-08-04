# Frontend and Durable State

- [Frontend and Durable State](#frontend-and-durable-state)
    - [Two ways to run the worker](#two-ways-to-run-the-worker)
    - [The supervisor channel](#the-supervisor-channel)
    - [Terminal, served, and attached modes](#terminal-served-and-attached-modes)
    - [Worker-owned stats history](#worker-owned-stats-history)
    - [The Getting started page](#the-getting-started-page)
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
  (currently 22) so a frontend can detect a mismatch with a worker built from different code.
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

Lifecycle changes are cooperative state-machine intents owned by the same thread
that calls `WorkerSupervisor.tick()`. Stop sends `SHUTDOWN` and returns; later
ticks keep draining progress until the old PID exits or the 150-second outer
deadline tree-kills it. Restart is the same stop followed by a spawn only after
the old PID is confirmed dead. Either way the supervisor drops the outgoing
worker's snapshot and liveness stamp immediately and pins a status of its own,
`STOPPING` or `RESTARTING`, so late frames from the draining process cannot
change the display back to `RUNNING` or age into `UNRESPONSIVE`. The presentation
therefore follows the supervisor's own intent and does not depend on the worker
reporting `shutting_down` before it goes quiet. A repeat stop request re-sends
the shutdown but keeps the original force-kill deadline, so pressing stop again
cannot defer the backstop that ends a stop the worker is ignoring. A restart
requested mid-stop upgrades that stop in place rather than starting a second
teardown. Start carries one intent, "have a worker running", and the supervisor
resolves it against its own state: it spawns when nothing is running, becomes the
replacement the stop already plans when a drain is under way, and does nothing at
all (beyond one explanatory log line) when the worker is already healthy. Callers
therefore never have to inspect the lifecycle first, and the host forwards a
client's start unconditionally instead of guessing from liveness; replacing a
healthy worker is the separate `restart` intent. Terminal and served modes use
this same supervisor transition; socket
clients merely enqueue the intent on the host's single owner thread and present
`STOPPING` locally until the host reports a status of its own.

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

## The Getting started page

When `bridgeData.yaml` is unconfigured
([`is_setup_incomplete`][horde_worker_regen.tui.wizard.is_setup_incomplete]), the
TUI opens the Getting started page
([`GettingStartedScreen`][horde_worker_regen.tui.wizard.GettingStartedScreen]). It
is the whole setup surface: it explains what the horde is and why a worker needs a
name, a key and models on disk, then collects those inline, so the Config tab is
where a working worker is refined rather than where setup happens. It stays
reachable from the Simple home afterwards, because the presets and the
explanations keep their value once a worker runs.

The three presets pair a model selection with a feature stance. Each is priced
against the disk its models will really need: the entries are expanded exactly as
the worker expands them
([`resolve_effective_models`][horde_worker_regen.tui.model_resolution.resolve_effective_models])
and the resulting models are costed with
[`compute_download_plan`][horde_worker_regen.model_download_plan.compute_download_plan].
A preset the volume cannot hold is shown disabled with the shortfall rather than
hidden, so the constraint is legible instead of surfacing as a failed download
later.

Saving writes only the keys the page owns (identity, `models_to_load`, `nsfw` and
the preset's feature flags) through the same comment-preserving YAML path the
config editor uses, so every other setting in an existing config survives. Nothing
the page does depends on the network being up: key validation, the name-taken
check and the model catalog all degrade to a quieter page, and closing without
saving leaves the worker stopped and every tab available for manual configuration.

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

## Progressive experience levels

The dashboard serves two audiences whose needs pull in opposite directions: a
contributor who wants to donate GPU time without learning what a sampling lane is,
and an operator tuning throughput. Rather than averaging them into one surface that
serves neither,
[`ExperienceLevel`][horde_worker_regen.app_state.ExperienceLevel] selects how much
implementation detail each destination renders.

The load-bearing constraint is that the level changes *depth*, never *navigation*.
Every tab exists and is reachable at every level. Overview, Live, and Downloads each
host both presentations and swap which is displayed; the rest keep their operator
widgets and gain only an explanatory line. Consolidating destinations by level was
considered and rejected: hiding tabs makes the dashboard's shape depend on a setting
the user may not know they changed, and someone who learns where a thing lives in
Simple would have to relearn it on promotion.

Withholding is deliberately reversible and non-destructive. Config fields are gated
through the stylesheet rather than skipped during composition, so a withheld widget
stays mounted holding its value and is written back on save; skipping composition
would silently drop those keys from `bridgeData.yaml` the next time an unrelated
setting was saved. The Config editor's Dashboard page is never withheld, so the
control that changes level cannot be hidden by the level.

The seven destinations that keep their operator widget at every level are framed in
Simple by a [`TabPrimer`][horde_worker_regen.tui.widgets.simple.TabPrimer]: one line
on what the page is for, plus a collapsible callout saying what the live figures mean
right now. Each [`PrimerCallout`][horde_worker_regen.tui.widgets.simple.PrimerCallout]
is conditional and renders only while its condition holds, so the callout carries the
worker's current anomalies and nothing else; with no condition holding it is hidden
outright and the framing line stands alone. Conditions are drawn from sticky signals
(session counters and latched flags) so a callout does not flicker between frames, and
the sentences quote the snapshot's own numbers. Stats, Control, Logs and Insights carry
callouts; GPUs, Diagnostics and Benchmark keep the framing line alone, since their
widgets already name and explain every figure they show. Relabelling the columns
themselves was rejected: it would fork the vocabulary by level, so a contributor who
learned a term in Simple would meet a different one on promotion. Several observations
at once run longer than a short terminal has rows, which is why the callout folds; it
opens expanded so it is read before it is dismissed.

Config audience is decided at two granularities, which answer different questions.
The sub-tab list decides which *pages* a level offers, and
[`ConfigField.risk_level`][horde_worker_regen.tui.config_form.ConfigField] decides
which *fields* within an offered page it shows. The `dangerous` tier is what
distinguishes Developer from Advanced: it carries the worker's self-policing settings
(hung-process timeouts, the VRAM and RAM budget, whole-card residency, the fault
breakers), which fail differently from a setting that merely trades throughput. A
section whose fields all share one tier carries that tier on its heading as well, so a
withheld group does not leave a titled empty block behind.

The level and the F6 density mode both narrow the Config editor's sub-tabs, so a
single arbiter reads both and decides visibility. Letting each setter write that
state directly made the result depend on which ran last: cycling density in Simple
re-showed every tuning page Simple had withheld.

Shortcuts follow the same rule as tabs, with one narrow exception. Customising the
Overview layout, revealing elements hidden from it, and cycling its density all act on
a widget Simple replaces with its own view, so `check_action` withholds those three
keys at that level: a footer hint for a control that is off screen offers something
with nothing to act on. The exception extends no further. Every destination, and every
other shortcut, remains available.

Because the default is Simple, an installation that predates the levels would
otherwise appear to have lost its detail. A state file stamped before schema 2 sets
`needs_experience_introduction`, and the notice is answered before the setup and
start prompts, so the changed default is visible rather than inferred. This is the
one thing the schema version is used for: reads never reject an older file, since
every field carries a default.

### Liveness that the failure cannot satisfy

Simple's indicator exists to distinguish a working worker from a wedged one, which
constrains what may drive it. A spinner advanced by the render loop keeps turning over
a dead worker, so it carries no information (see
[Liveness proofs must be failure-independent](resilience_and_recovery.md)). Only
signals the worker itself produces may advance it, and which of those is truthful
depends on whether there is work in hand.

While a job is in hand, the frame advances on the child processes' own reporting:
their sampling counter, or their heartbeat timestamps. The supervisor's snapshot
timestamp is excluded in that state. A supervisor whose loop is healthy goes on
stamping snapshots over a wedged child, so admitting it would let the failure supply
its own proof of life. With nothing in hand the snapshot timestamp becomes the correct
signal: the worker is alive with nothing to do, and no child is reporting.

Both child signals are read, because they answer different questions. The worker
resets `heartbeats_inference_steps` to zero on every heartbeat that is not a sampling
step, so alone it reads a model load or a post-processing pass as a wedge, and those
routinely run for tens of seconds. The heartbeat timestamp advances on any heartbeat,
which separates a busy non-sampling stage from an absent process.

Whether a stall amounts to a *fault* is settled by
[`derive`][horde_worker_regen.tui.health.derive], against tuned, download-aware
thresholds the whole dashboard shares. Reaching a second verdict here, from a subset
of the same evidence, would put the Simple view at odds with every other surface. The
indicator supplies the animation and takes its alarm state from the health report.

Progress percentages come from the worker's own reported values and fall back to its
step counters; when it reports neither, the view shows an indeterminate state rather
than inventing a number.

### Trends that can show a stall

The same constraint governs Simple's headline trends. The figures they sit under are
cumulative session counters, and a sparkline drawn from a counter can only rise: it
plateaus at whatever the session reached and goes on reading as a full chart right
through an outage, which is precisely the state a contributor most needs to see. Both
Home trends therefore chart the per-interval *delta* over a fixed recent window,
through the same [`trends`][horde_worker_regen.tui.trends] helpers the operator
Overview uses, so a worker that stops finishing work draws a flat baseline while its
totals stay where they are. The window is fixed rather than selectable here: the
question Home answers is whether the worker is earning *now*, and a span long enough to
average a stall away would defeat it.

Rates the worker measures itself are quoted rather than recomputed, and named as
unmeasured when it has not reported one, on the same grounds as an unreported progress
percentage.

The attention card follows the inverse convention to the rest of the view: it renders
only for a health finding or a posture that explains a running worker earning nothing
(maintenance, pop backoff, processes restarted during the session), and is absent
otherwise, so its presence is the signal and a nominal worker pays no visual cost for
it.

Identity and offered features sit in the status hero rather than in a card of their
own. The home view already fills the terminal it is designed for, and another card
would push the action buttons past the bottom of it.

## Durable app state

[`app_state.py`][horde_worker_regen.app_state] is the structured, on-disk
counterpart to the in-memory [`WorkerState`](architecture.md#the-shared-state-pattern):
it records what the application needs to remember *between* invocations: the last
benchmark and where its results live, the last worker run, the last-known-good
settings, which worker version last ran (so a version bump can mark a stale
benchmark for re-running), and the operator's durable UI preferences (the experience
level, the theme and display density, the Overview density mode, the trend window,
and which Overview panels are hidden). A hidden-panel key that no longer names a live
element is dropped on load, so a stale preference can never block the Overview from
rendering; a `theme_name` this build cannot restore falls back to the default for the
same reason. The known theme names are duplicated in `app_state` rather than imported
from the TUI, because the worker reads this module too and it stays free of Textual.

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
