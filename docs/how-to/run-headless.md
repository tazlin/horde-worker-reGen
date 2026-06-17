# Run the worker headless

The dashboard is optional. The headless worker is the right choice for unattended, automated,
containerised, and remote machines. It reads `bridgeData.yaml` (or environment variables), downloads
your models, and starts working with no UI.

## Set up the config

If you have not already configured the worker:

1. Copy `bridgeData_template.yaml` to `bridgeData.yaml`.
2. Set at least `api_key` and `dreamer_name`.
3. Tune the rest for your hardware: see [Configure for your GPU](configure-for-your-gpu.md).

## Start it

```bash
# Windows
horde-bridge.cmd

# Linux (use the -rocm variant on AMD)
./horde-bridge.sh
```

These scripts ensure the environment, download and verify your configured models, then run the
worker. Model downloads happen before the worker starts, so the first run can take a while.

To run more than one worker on the same machine (one per GPU), pass a distinct name through to the
worker:

```bash
./horde-bridge.sh -n "GPU-0"
```

See [Run multiple GPUs](run-multiple-gpus.md) for the full pattern.

## Stopping

Press `Ctrl+C` in the worker's terminal. It finishes any in-progress jobs before exiting. Avoid hard
killing it unless you are seeing many major errors; you can force a stop by pressing `Ctrl+C`
repeatedly or sending `SIGKILL`.

## Configure from environment variables (containers)

Instead of a config file, the worker can read its configuration from `AIWORKER_*` environment
variables. This suits Docker and other immutable deployments. Pass `-e` (or
`--load-config-from-env-vars`) to `run_worker`:

```bash
run_worker -e
```

When config comes from environment variables, the live config-reload loop is not started and the
configuration is effectively immutable for the run. See
[Bridge configuration](../explanation/bridge_config.md#how-configuration-loads) for the details, and
[Run in Docker](run-in-docker.md) for container images.

## Under the hood

`horde-bridge` runs two steps you can also run yourself in a prepared environment: `download_models`
(fetch and verify every model in your config) followed by `run_worker` (start working). The
`download_models` step is what guarantees a model is on disk before the worker advertises it.

For the full list of entry points, flags, and environment variables, see the
[CLI reference](../reference/cli.md). Logs are written to `logs/`; see [Logs](../reference/logs.md).
