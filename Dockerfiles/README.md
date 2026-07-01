# horde-worker-reGen in Docker (CUDA and ROCm)

These images run the worker in a container for either NVIDIA (CUDA) or AMD (ROCm) GPUs.

The images are **immutable**: the worker source and all of its dependencies are baked in at build
time. A container does **not** clone the repo or reinstall dependencies on startup; it just runs the
worker. To update, pull (or rebuild) a newer image and recreate the container.

## Prerequisites

- Docker installed on your system
- NVIDIA GPU with current drivers + the NVIDIA Container Toolkit (for the CUDA image), or
- AMD GPU with current ROCm-capable drivers (for the ROCm image)

## Option A: Use the prebuilt CUDA image (recommended)

Prebuilt CUDA images are published to the GitHub Container Registry:

```bash
docker pull ghcr.io/haidra-org/horde-worker-regen:latest   # newest main build
# or a specific release, e.g.:
# docker pull ghcr.io/haidra-org/horde-worker-regen:v12.7.4
```

### Image variants (CUDA build per GPU architecture)

One torch CUDA build cannot cover every GPU: `cu126` carries kernels for `sm_50`–`sm_90` (Maxwell
through Hopper, incl. Ada/Ampere) but **not** Blackwell, while `cu130`/`cu132` cover `sm_75`–`sm_120`
(Turing through Blackwell) but drop pre-Turing. A build with no kernel image for the host card crashes on
the first GPU model load, so each build is published as its own tag:

| Tag | Torch build | GPUs |
| --- | --- | --- |
| `:latest`, `:vX.Y.Z` (unsuffixed) | `cu130` | Turing → **Blackwell** (RTX 50-series, datacenter) |
| `:latest-cu130`, `:vX.Y.Z-cu130` | `cu130` | same as the default, named explicitly |
| `:latest-cu132`, `:vX.Y.Z-cu132` | `cu132` | Turing → Blackwell on a CUDA 13.2 driver |
| `:latest-cu126`, `:vX.Y.Z-cu126` | `cu126` | Maxwell → Hopper (**pre-Turing / older cards**) |

The unsuffixed tags default to `cu130` because the common container case is current datacenter / Blackwell
hardware. If your card is **pre-Turing** (Maxwell/Pascal/Volta, e.g. GTX 10-series), pull the `-cu126`
variant instead. `cu130`/`cu132` also require a **CUDA-13-capable NVIDIA driver** on the host; update the
host driver if the container reports an incompatibility. The entrypoint runs a preflight that fails fast
with the exact rebuild/pull instruction if the image's build has no kernels for the detected GPU, rather
than crashing deep inside the first model load.

Configure the worker entirely through `AIWORKER_*` environment variables (see
[Configuration](#configuration)) and run it with the GPU passed through and a host directory mounted
for the model cache:

```bash
docker run -it --gpus all \
  -e AIWORKER_API_KEY=your_api_key_here \
  -e AIWORKER_DREAMER_NAME=your_worker_name_here \
  -e AIWORKER_CACHE_HOME=/horde-worker-reGen/models \
  -v "$(pwd)/models":/horde-worker-reGen/models \
  ghcr.io/haidra-org/horde-worker-regen:latest
```

> ROCm images are not currently published; build them locally (see
> [Building locally](#option-c-build-locally)).

## Option B: Use docker compose

A compose file is provided for each GPU type. By default it builds the image locally; to use the
prebuilt CUDA image instead, edit `compose.cuda.yaml` to comment out the `build:` block and uncomment
the `image:` line.

Set up your `bridgeData.yaml` (see the repository
[configuration guide](https://github.com/Haidra-Org/horde-worker-reGen?tab=readme-ov-file#configure)),
then from the repository root:

```bash
docker compose -f Dockerfiles/compose.cuda.yaml up -dV   # or compose.rocm.yaml
```

> **Warning**: The compose files mount your `bridgeData.yaml` into the container. If any setting points
> at an absolute or Windows-style path (**especially `cache_home`**), the worker inside the container
> will not behave as expected. Use `AIWORKER_BRIDGE_DATA_LOCATION` to point at a different config file
> and `AIWORKER_CACHE_HOME` to set the host models directory.

The compose file mounts a `models` directory next to the repository so selected models are not
re-downloaded each time.

### Start, monitor, and stop a compose container

```bash
docker start -ai reGen   # attach to (or start) the container; CTRL+C detaches but leaves it running
docker start reGen       # start detached (background)
docker stop reGen        # stop
```

> Note: To reduce the chance of dropping jobs when `docker stop` times out, set your worker into
> maintenance mode first whenever possible (the AI Horde [API](https://aihorde.net/api/) PUT endpoint
> `/v2/workers/{worker_id}`, or a frontend like [artbot.site](https://artbot.site/)).

### Updating with compose

Pull the latest source, rebuild, and let compose recreate the container:

```bash
git pull
docker compose -f Dockerfiles/compose.cuda.yaml build --pull
docker compose -f Dockerfiles/compose.cuda.yaml up -dV
```

## Option C: Build locally

The build context is the **repository root** (the image copies the source in), so build from the root
and point `-f` at the Dockerfile:

```bash
# NVIDIA (CUDA)
docker build -f Dockerfiles/Dockerfile.cuda -t horde-worker-regen:cuda .

# AMD (ROCm)
docker build -f Dockerfiles/Dockerfile.rocm -t horde-worker-regen:rocm .
```

To build a fork or a feature branch, simply check it out first (`git switch <branch>`) and build; the
image bakes in whatever source tree you build from.

### Build arguments

CUDA (`Dockerfile.cuda`):

- `CUDA_VERSION` (default `12.8.1`) selects the `nvidia/cuda:<version>-runtime-ubuntu22.04` base. Use a
  CUDA-13 base (e.g. `13.0.1`) when building a `cu130`/`cu132` image.
- `TORCH_BACKEND` (default `cu126`) selects the PyTorch build extra, which must match the host GPU's
  architecture (see [Image variants](#image-variants-cuda-build-per-gpu-architecture)): `cu126` for
  Maxwell–Hopper, `cu130`/`cu132` for Turing–Blackwell (RTX 50-series). The published `:latest` uses
  `cu130`; the local build default stays `cu126` for the widest older-card compatibility, so pass
  `--build-arg TORCH_BACKEND=cu130 --build-arg CUDA_VERSION=13.0.1` to build a Blackwell-capable image.

ROCm (`Dockerfile.rocm`):

- `ROCM_VERSION` (default `6.2.1`) selects the `rocm/rocm-terminal:<version>` base.
- `HORDE_WORKER_ROCM_TORCH` (default `2.9.1`) and `HORDE_WORKER_ROCM_INDEX`
  (default `https://download.pytorch.org/whl/rocm6.4`) pin the ROCm PyTorch overlay, which is installed
  ad-hoc because ROCm builds are not in `uv.lock`.

### Running a locally built image

```bash
# NVIDIA (CUDA)
docker run -it --gpus all horde-worker-regen:cuda

# AMD (ROCm)
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video horde-worker-regen:rocm
```

## Configuration

The entrypoint applies the right GPU-specific runtime setup based on the image (`GPU_TYPE`) and then
launches the worker. Configure the worker one of two ways:

- Mount a `bridgeData.yaml` at `/horde-worker-reGen/bridgeData.yaml`, or
- Set `AIWORKER_*` environment variables (used when no `bridgeData.yaml` is present).

### Setting config by environment variables

Any option in `bridgeData_template.yaml` can be set by prefixing it with `AIWORKER_`. A typical config
(adjust for your machine; these values will not suit every system):

```
AIWORKER_API_KEY=your_api_key_here          # Important
AIWORKER_CACHE_HOME=/horde-worker-reGen/models  # Important
AIWORKER_DREAMER_NAME=your_worker_name_here # Important
AIWORKER_ALLOW_CONTROLNET=True
AIWORKER_ALLOW_LORA=True
AIWORKER_MAX_LORA_CACHE_SIZE=50
AIWORKER_ALLOW_PAINTING=True
AIWORKER_MAX_POWER=38
AIWORKER_MAX_THREADS=1 # Only set to 2 on high end or xx90 machines
AIWORKER_MODELS_TO_LOAD=['TOP 3', 'AlbedoBase XL (SDXL)'] # Mind download times; ~2-8 GB each
AIWORKER_MODELS_TO_SKIP=['pix2pix', 'SDXL_beta::stability.ai#6901']
AIWORKER_QUEUE_SIZE=2
AIWORKER_MAX_BATCH=4
AIWORKER_SAFETY_ON_GPU=True
AIWORKER_CIVITAI_API_TOKEN=your_token_here
```

See `bridgeData_template.yaml` for the full set of options.

#### Generating an `.env` file from a `bridgeData.yaml`

If you have a local install of the worker, convert a `bridgeData.yaml` into a `.env` file suitable for
`docker run --env-file`:

```bash
uv run python -m convert_config_to_env --file ./bridgeData.yaml
```

This writes `bridgeData.env` to the current directory. Note that `models_to_load`/`models_to_skip`
meta-commands such as `TOP 5` are resolved to a concrete list at the time of conversion (not kept
dynamic); specify models manually if you want the dynamic behavior.

## Troubleshooting

1. Ensure the host has current GPU drivers (and, for NVIDIA, the NVIDIA Container Toolkit).
2. For ROCm, confirm the host supports the `ROCM_VERSION` the image was built against.
