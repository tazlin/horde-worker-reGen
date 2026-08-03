# Logs

The worker writes logs to the `logs/` directory. The dashboard's **Logs** tab reads these same files,
and the **Diagnostics** tab runs the [`horde-log diagnose`](cli.md#horde-log) detectors over
them to surface what went wrong, without you needing a shell. See
[How the diagnostics stay in sync](../explanation/log_diagnostics_contract.md) for how a detector, the
log line it reads, and the dashboard are kept from drifting apart.

| File | Contents |
|------|----------|
| `bridge.log` | Main log (all info). |
| `bridge_n.log` | Per-process log. |
| `trace.log` | Errors, criticals, and `TRACE`-level lines (including the suppressed repeats described below). |
| `trace_n.log` | The same, per process. |
| `bridge_tui.log` / `bridge_host.log` | The supervisor (parent) process's own log: TUI dashboard or `--host` wrapper. Captures worker launch, crash-loop, and TUI-process crash diagnostics that never reach `bridge.log`. |

## Repeating telemetry is time-boxed

Some lines describe a condition that is re-evaluated on every control-loop tick: a child's memory report,
or a reason the parent skipped a periodic reconciliation. Emitting each one at `DEBUG` puts several lines
a second into `bridge.log` and crowds out the job-flow story an operator is actually reading.

Such lines are time-boxed per subject (per process, per device): at most one emission per 30 seconds lands
at `DEBUG`, and the repeats in between are emitted at `TRACE` instead. Nothing is dropped. `bridge.log`
carries a periodic sample and `trace.log` carries the full series, so a per-tick reconstruction is still
possible after the fact.

If you are adding a log line that fires on a timer rather than on an event, use
`throttled_log_level()` from `horde_worker_regen/process_management/_internal/util.py` and log at the level
it returns.

## Rotation and retention

The supervisor logs (`bridge_tui.log` / `bridge_host.log`) and the benchmark run logs rotate at a **25 MB**
size cap, are compressed to `.zip` once rotated, and keep a bounded number of older files (20 for the
supervisor, 10 for benchmark runs). This keeps total disk use bounded under a heavy or long-running session,
and keeps any single file small enough that the **Logs** tab can tail it without buffering a multi-GB file.
The dashboard reads only the trailing window of a log, so a large file scrolls quickly to the latest lines.

## Headless model-pool status

The headless worker's periodic status block always names the model-pool mode. With the pool off, it says that
normal eligible-model advertising has no persistent seat bias. With the pool on, one compact line lists the
first six seats (plus an overflow count), their source (`M`, `R`, or `S`) and readiness (`resident` with GPU,
`cold`, or `downloading`), the most recent fixed/free advertising lane, each lane's matched/popped rate and
resident-match count, bench size, demand-reading age, and charged/session download admission budget. The charge is the
reference-declared model size booked when a request starts, not measured disk use or bandwidth. A pending seat
names its target model and reads `downloading`; reaching the limit does not cancel that already-started transfer.

## Which process is which

The numbered logs map to the worker's child processes:

- `bridge_0.log` is the **safety** process.
- `bridge_1.log` and higher are **inference** processes.

For why the worker runs separate inference and safety processes, see
[Architecture](../explanation/architecture.md).

## Tailing a log live

```powershell
# Windows PowerShell
Get-Content bridge_1.log -Wait
```

```bash
# Linux/macOS
less +F bridge_1.log
```

## VRAM admission diagnostics

A warning beginning `Deferring post-processing for job` records a lane-admission miss. Its candidate is the
post-processing chain's marginal VRAM requirement. The available-room expression reports the same arithmetic
the arbiter decided from: current device-free VRAM minus outstanding reservations and the proportional noise
margin. Outstanding means memory not yet materialized in the device-free sample; already-realized commitments
are not subtracted again.

The stable prefix is part of the diagnostics contract used by `horde-log diagnose`. The numeric wording may
become more precise without invalidating older logs, so the detector accepts both the current measured-room
format and the former free-after-commitments format. Repeated warnings for one job, with no later PP completion,
surface as `post_processing_deferral_starvation` in the dashboard's Diagnostics tab.

## Sharing logs

Do **not** post `.log` files in public channels. Send them to a maintainer directly: we cannot
guarantee your API key is not present in a log. See [Troubleshooting](../how-to/troubleshoot.md).
