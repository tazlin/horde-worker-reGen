#!/bin/bash
set -e

# Immutable image: the worker code and its dependencies are already baked in at build time. This
# entrypoint does NOT pull code or re-sync dependencies; it only applies host-GPU-dependent runtime
# setup and launches the worker. To update, pull a newer image tag.

cd "${APP_HOME}"

# Put the baked virtualenv on PATH so bare `python` (e.g. inside the AMD setup script) resolves to it.
export PATH="${APP_HOME}/.venv/bin:${PATH}"

if [ "${GPU_TYPE}" = "rocm" ]; then
    # Enable AMD flash-attention only on cards that support it (host-dependent, so decided at runtime).
    SUPPORTED_CARD=$(rocminfo | grep -c -e gfx1100 -e gfx1101 -e gfx1102 || true)
    if [ "${SUPPORTED_CARD}" -gt 0 ]; then
        export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:=TRUE}"
    fi
    export MIOPEN_FIND_MODE="FAST"

    # Installs/cleans flash-attention + the AMD GO FAST node based on the detection above. The ROCm
    # torch stack itself is already baked into the image.
    bash "${APP_HOME}/horde_worker_regen/amd_go_fast/install_amd_go_fast.sh"
else
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}

    # Fail fast with an actionable rebuild instruction when the baked torch build has no kernels for the
    # host GPU. An immutable image cannot switch CUDA builds at runtime, so otherwise the first GPU model
    # load crashes with a cryptic deep traceback (e.g. open_clip's convert_weights_to_lp). set -e turns a
    # non-zero exit here into a clean stop carrying the message, instead of a confusing mid-download crash.
    uv run --no-sync python -m horde_worker_regen.torch_gpu_preflight
fi

# Configure from a mounted bridgeData.yaml if present, otherwise from AIWORKER_* environment variables.
if [ -e bridgeData.yaml ]; then
    uv run --no-sync python download_models.py
    exec uv run --no-sync python run_worker.py
else
    uv run --no-sync python download_models.py -e
    exec uv run --no-sync python run_worker.py -e
fi
