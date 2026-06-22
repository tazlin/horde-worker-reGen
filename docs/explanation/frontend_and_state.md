# Frontend and Durable State

- [Frontend and Durable State](#frontend-and-durable-state)
    - [Two ways to run the worker](#two-ways-to-run-the-worker)
    - [The supervisor channel](#the-supervisor-channel)
    - [Terminal, served, and attached modes](#terminal-served-and-attached-modes)
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
duplex pipe. [`supervisor_channel.py`][horde_worker_regen.process_management.supervisor_channel]
defines the structured protocol over it:

- The worker pushes
  [`WorkerStateSnapshot`][horde_worker_regen.process_management.supervisor_channel.WorkerStateSnapshot]
  objects at a steady cadence (the same data the overview, per-process view, and
  Downloads tab render), including a `SystemMemorySnapshot` (machine total/available
  RAM plus per-role worker RSS). The snapshot is versioned by
  `SUPERVISOR_PROTOCOL_VERSION` (currently 6) so a frontend can detect a mismatch
  with a worker built from different code.
- The worker drains
  [`SupervisorControlMessage`][horde_worker_regen.process_management.supervisor_channel.SupervisorControlMessage]
  commands each loop tick (start/stop intent, download pause/resume and rate
  limit, etc.).

This mirrors the worker's own internal IPC (see
[IPC and Messaging](ipc_and_messaging.md)) and is the structured upgrade of the
[`.abort` sentinel](shutdown_and_faults.md#the-abort-file) external-supervision
hook. The models are deliberately pure-data and JSON-round-trippable: the default
transport is a `multiprocessing` pipe (pickle), but the same models serialize
cleanly for the localhost-socket fallback the launcher uses in served mode.

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
pop time. [`worker_identity.py`][horde_worker_regen.process_management.worker_identity]
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
settings, and which worker version last ran (so a version bump can mark a stale
benchmark for re-running).

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
