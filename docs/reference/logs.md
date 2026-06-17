# Logs

The worker writes logs to the `logs/` directory. The dashboard's **Logs** tab reads these same files.

| File | Contents |
|------|----------|
| `bridge.log` | Main log (all info). |
| `bridge_n.log` | Per-process log. |
| `trace.log` | Errors and warnings only. |
| `trace_n.log` | Per-process errors and warnings. |

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
