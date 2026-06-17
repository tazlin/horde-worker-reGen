# Run in Docker

Prebuilt CUDA images are published to the GitHub Container Registry at
[ghcr.io/haidra-org/horde-worker-regen](https://github.com/Haidra-Org/horde-worker-reGen/pkgs/container/horde-worker-regen).
Pull `:latest` for the newest `main` build or a `:vX.Y.Z` tag for a specific release:

```bash
docker pull ghcr.io/haidra-org/horde-worker-regen:latest
```

The images are immutable: the worker code and its dependencies are baked in at build time, so a
container does not pull or reinstall anything at startup. To update, pull a newer tag and recreate the
container.

The container worker is configured from `AIWORKER_*` environment variables rather than a config file,
which keeps the image immutable. That is the same env-var path described in
[Run headless](run-headless.md#configure-from-environment-variables-containers).

For the full, supported container setup (image tags, required environment variables, GPU passthrough,
and volume mounts for the model cache), follow the
[Docker guide](https://github.com/Haidra-Org/horde-worker-reGen/blob/main/Dockerfiles/README.md) in
the repository.
