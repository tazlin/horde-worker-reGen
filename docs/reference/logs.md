# Logs

The worker writes logs to the `logs/` directory. The dashboard's **Logs** tab reads these same files.

| File | Contents |
|------|----------|
| `bridge.log` | Main log (all info). |
| `bridge_n.log` | Per-process log. |
| `trace.log` | Errors and warnings only. |
| `trace_n.log` | Per-process errors and warnings. |
| `bridge_tui.log` / `bridge_host.log` | The supervisor (parent) process's own log: TUI dashboard or `--host` wrapper. Captures worker launch, crash-loop, and TUI-process crash diagnostics that never reach `bridge.log`. |

## Rotation and retention

The supervisor logs (`bridge_tui.log` / `bridge_host.log`) and the benchmark run logs rotate at a **25 MB**
size cap, are compressed to `.zip` once rotated, and keep a bounded number of older files (20 for the
supervisor, 10 for benchmark runs). This keeps total disk use bounded under a heavy or long-running session,
and keeps any single file small enough that the **Logs** tab can tail it without buffering a multi-GB file.
The dashboard reads only the trailing window of a log, so a large file scrolls quickly to the latest lines.

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

## Sharing logs

Do **not** post `.log` files in public channels. Send them to a maintainer directly: we cannot
guarantee your API key is not present in a log. See [Troubleshooting](../how-to/troubleshoot.md).
