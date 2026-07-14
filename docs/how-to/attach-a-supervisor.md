# Attach a headless supervisor (file-driven observe and control)

The attach supervisor runs the worker with no UI and no socket, and instead exposes it through three
JSONL files under a session directory. It is built for an autonomous operator (a script or an agent) that
needs to watch the worker in near-real-time and steer it without a terminal: it observes by tailing files
and controls by appending to a file inbox.

It reuses the same worker ownership as the dashboard and the worker host (`horde-worker-host`): spawn over
a pipe, auto-restart on crash, the alive-but-frozen wedge backstop, and orphan-proof shutdown. It never
loads the inference stack itself; it runs the orchestrator role only.

## Start it

```bash
horde-worker-attach --session-dir ./session
```

`--session-dir` is required and is where the three files are written. Common options:

- `--interval` -- seconds between polls (default `5`): how often state is sampled, alerts are
  re-evaluated, and the command inbox is read.
- `--config` -- path to a `bridgeData.yaml`. Its directory becomes the working directory, so the config,
  the `logs/` sink, and the live-log watch all read from the same place.
- `-n/--worker-name`, `-e/--load-config-from-env-vars`, `--amd`, `--directml`, `--no-auto-restart` --
  the usual worker launch options, forwarded to the worker.

Stop it with `Ctrl+C` (or `SIGTERM`); it shuts the worker down through the launcher's orphan-proof path,
draining in-flight jobs before force-killing the tree if it overruns.

## The three files

### `state.jsonl` (observe)

One compact JSON line per interval summarizing the latest worker snapshot. Each line is kept under 2 KB
(per-process rows are shed first if needed). The keys:

| Key | Meaning |
| --- | --- |
| `t` | Wall-clock time the line was written. |
| `worker_up` | `false` until the worker sends its first frame. |
| `snap_t` | Timestamp of the snapshot being summarized. |
| `liveness_age` | Seconds since the worker's control loop last advanced (freshness of the parent). |
| `procs` | Per-process rows: `id`, `type`, `state`, `busy`, `model` (`procs_truncated` if any were dropped). |
| `queue` / `in_progress` | Jobs pending inference / currently generating. |
| `popped` / `submitted` / `faulted` | Cumulative job counters. |
| `kudos_hr` | Kudos per hour, when the worker reports it. |
| `maintenance` / `server_maintenance` | Local maintenance flag / the horde's worker-details maintenance. |
| `download` | `phase`, current `file`, `done_bytes`, `total_bytes` (or `null` when downloads are off). |
| `gpu_duty` | GPU duty percentage / busy fraction, when measured. |

### `alerts.jsonl` (wake me)

Append-only, edge-triggered alerts worth an operator's attention. Each fires once per episode and re-arms
only after its condition clears, so a steady problem does not spam. Two sources feed it:

- **Live-log findings** -- an incremental `watch_pass` over the worker's `logs/` each interval, emitting a
  line the first time a warning/critical diagnosis appears (the same detectors as `horde-log`).
- **Snapshot threshold rules**:

| Rule | Fires when |
| --- | --- |
| `frozen_parent` | The worker's control-loop liveness stamp is more than 30 s stale. |
| `consecutive_failure_pause` | The worker armed its consecutive-failure pop pause. |
| `fault_burst` | 5 or more job faults landed in the last 10 minutes. |
| `gpu_idle_with_pending` | Jobs are pending while every inference process has sat idle for over 120 s. |
| `download_no_progress` | An active download's byte count has not advanced for over 120 s. |

Each line carries a `severity` (`warning`/`critical`), a human `summary`, and the key numbers behind it.

### `commands.jsonl` (control)

An append-only inbox the supervisor polls each interval. Each line is one JSON object with a `command`
verb plus any fields that verb needs. Every line is applied exactly once (a processed-offset is tracked);
malformed or unknown lines are logged, surfaced as a `command_rejected` alert, and skipped, never fatal.

The verbs are the worker's full control surface. `SHUTDOWN` (or the `GRACEFUL_SHUTDOWN` alias) routes
through the orphan-proof graceful path rather than a raw command.

```jsonl
{"command": "SET_SERVER_MAINTENANCE", "server_maintenance_enabled": true}
{"command": "SET_CONCURRENCY", "target_threads": 2, "target_processes": 3}
{"command": "SET_DOWNLOAD_RATE_LIMIT", "download_rate_limit_kbps": 5000}
{"command": "RESTART_PROCESS", "process_id": 1}
{"command": "DOWNLOAD_MODELS", "download_model_names": ["AlbedoBase XL (SDXL)"], "download_include_aux": false}
{"command": "PAUSE"}
{"command": "GRACEFUL_SHUTDOWN"}
```

The full verb set: `PAUSE`, `RESUME`, `DRAIN`, `RESTART_PROCESS`, `RELOAD_CONFIG`, `SET_CONCURRENCY`,
`PAUSE_DOWNLOADS`, `RESUME_DOWNLOADS`, `SET_DOWNLOAD_RATE_LIMIT`, `DOWNLOADS_ONLY_HOLD`, `GO_LIVE`,
`DOWNLOAD_MODELS`, `SET_SERVER_MAINTENANCE`, `SET_STATS_EXPORT`, and `SHUTDOWN` / `GRACEFUL_SHUTDOWN`.

## The auto-guard (one pre-authorized action)

The supervisor is otherwise observe-only, with one exception. When either the `gpu_idle_with_pending` or
the `frozen_parent` condition persists continuously for the confirmation window (180 s), the guard sends a
single `SET_SERVER_MAINTENANCE=true` so the horde stops routing jobs to a worker that cannot serve them,
and appends a `critical` `auto_guard_server_maintenance` alert explaining what it did and why.

The guard is deliberately conservative:

- It acts **once** per episode (edge-triggered), never repeatedly.
- It **never** restarts a process or shuts the worker down. Those remain operator decisions via the inbox.
- Appending a `SET_SERVER_MAINTENANCE` with `server_maintenance_enabled: false` to the inbox re-arms it, so
  after you lift maintenance the guard must observe the condition persist again before acting.

## See also

- [Run the worker headless](run-headless.md) for the plain no-UI worker and the persistent worker host.
- [Logs](../reference/logs.md) and the `horde-log` triage tool that share the live-log detectors used here.
