# Run the worker as a system service

Use this when the worker runs unattended and no dashboard is watching it. A service manager gives the
worker something the dashboard would otherwise provide: a way to come back after the recovery escalation
decides a fresh process is the only remaining remedy.

If you run the worker with the dashboard (`horde-worker`), you already have this. The dashboard
supervises its worker child and relaunches it on an unexpected exit, so you do not need a service unit.

## Why this matters

The worker recovers from most trouble in place: it retries jobs, replaces crashed processes, reclaims
VRAM, and rebuilds its process pools. A few failures outlive all of that, usually because something has
leaked device state that only a new process can clear. For those, the last rung of the escalation is to
exit non-zero and let whoever launched the worker start a fresh one.

That rung is only a recovery if something is actually listening for the exit. The worker therefore checks
before taking it:

| Situation | What the worker does when escalation asks for a fresh process |
| --------- | ------------------------------------------------------------- |
| Dashboard attached | Exits non-zero. The dashboard relaunches it. |
| `exit_on_unhandled_faults: true` | Exits non-zero. Your service manager restarts it. |
| Neither | Does **not** exit. Keeps escalating in place, and holds quiescent rather than churning if nothing it can do in place helps. |

So on an unattended host you have two valid choices. Leave `exit_on_unhandled_faults` off and the worker
never exits on its own, trading the strongest remedy for the certainty that the process stays alive. Or
set it and pair it with a service manager, which is the configuration this guide describes.

## Linux: systemd

Create `/etc/systemd/system/horde-worker.service`, replacing the user and paths:

```ini
[Unit]
Description=AI Horde worker (horde-worker-reGen)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=horde
Group=horde
WorkingDirectory=/home/horde/horde-worker-reGen
ExecStart=/home/horde/horde-worker-reGen/horde-bridge.sh
Restart=always
RestartSec=15
# The worker shuts its children down itself; give it room to finish in-flight jobs first.
TimeoutStopSec=180
KillMode=mixed

[Install]
WantedBy=multi-user.target
```

Then set `exit_on_unhandled_faults: true` in `bridgeData.yaml` and enable the unit:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now horde-worker
sudo systemctl status horde-worker
journalctl -u horde-worker -f
```

Notes on the settings that matter:

- `Restart=always` is the point of the exercise. `Restart=on-failure` also works, but `always` also
  covers a clean exit you did not intend.
- `RestartSec=15` keeps a genuinely broken install from restarting in a tight loop. Raise it if your
  models take a long time to verify on startup.
- `TimeoutStopSec=180` is deliberately generous. The worker finishes or faults its accepted jobs during
  shutdown, and killing it early strands them, which the horde sees as dropped work.
- `KillMode=mixed` sends the stop signal to the main process only, letting the worker tear down its own
  children in the right order.

Use the `WorkingDirectory` shown: the worker resolves `bridgeData.yaml` and writes `logs/` relative to it.

## Windows: Task Scheduler

Windows has no direct systemd equivalent. Create a scheduled task that runs `horde-bridge.cmd`:

- **Trigger**: At startup.
- **Action**: Start a program, `horde-bridge.cmd`, with "Start in" set to your worker directory.
- **Settings**: enable "If the task fails, restart every" and set an interval of a few minutes.
- Run it as a user that owns the worker directory, and tick "Run whether user is logged on or not".

Set `exit_on_unhandled_faults: true` as above so the worker exits for the scheduler to restart.

If you want closer supervision than Task Scheduler offers, run the dashboard instead, or use the
project's own headless supervisor (see [Attach a supervisor](attach-a-supervisor.md)), which spawns and
relaunches the worker over a pipe and exposes state and control files.

## Confirming it works

After a restart, check that the worker came back on its own:

```bash
systemctl show horde-worker --property=NRestarts
```

The worker also records escalation decisions in its action ledger, so
`horde-log bundle` captures why a restart happened. See
[Resilience and recovery](../explanation/resilience_and_recovery.md) for what each rung of the escalation
does, and [Troubleshoot](troubleshoot.md) if the worker restarts repeatedly rather than settling.

## See also

- [Run the worker headless](run-headless.md)
- [Attach a supervisor](attach-a-supervisor.md)
- [Resilience and recovery](../explanation/resilience_and_recovery.md)
- [Command-line reference](../reference/cli.md)
