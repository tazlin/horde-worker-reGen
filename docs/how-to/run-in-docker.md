# Run in Docker

Prebuilt images are published at
[hub.docker.com/r/tazlin/horde-worker-regen](https://hub.docker.com/r/tazlin/horde-worker-regen/tags).

The container worker is configured from `AIWORKER_*` environment variables rather than a config file,
which keeps the image immutable. That is the same env-var path described in
[Run headless](run-headless.md#configure-from-environment-variables-containers).

For the full, supported container setup (image tags, required environment variables, GPU passthrough,
and volume mounts for the model cache), follow the
[Docker guide](https://github.com/Haidra-Org/horde-worker-reGen/blob/main/Dockerfiles/README.md) in
the repository.
