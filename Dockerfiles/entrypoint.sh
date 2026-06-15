#!/bin/bash
set -e

# Source environment variables from /env_vars file
if [ -f "/env_vars" ]; then
    . /env_vars
else
    echo "/env_vars file not found. Exiting."
    exit 1
fi

cd ${APP_HOME}
git fetch
git reset --hard origin/${GIT_BRANCH:-main}

# Determine GPU type and install the right PyTorch build.
if [ ! -z "${GPU_TYPE}" ] && [ "${GPU_TYPE}" = "rocm" ]; then
    # Determine if the user has a flash attention supported card.
    SUPPORTED_CARD=$(rocminfo | grep -c -e gfx1100 -e gfx1101 -e gfx1102)
    if [ "$SUPPORTED_CARD" -gt 0 ]; then export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:=TRUE}"; fi

    export MIOPEN_FIND_MODE="FAST"

    # ROCm is not in uv.lock (torch 2.12.0 has no lockable ROCm build); sync the base env then overlay
    # the ROCm PyTorch stack ad-hoc (torch 2.9.1 on ROCm 6.4 by default; see update-runtime-rocm.sh).
    uv sync --locked --extra cpu
    uv pip install --reinstall \
        "torch==${HORDE_WORKER_ROCM_TORCH:-2.9.1}" torchvision torchaudio pytorch-triton-rocm \
        --extra-index-url "${HORDE_WORKER_ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.4}"
else
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    export PATH=${CUDA_HOME}/bin:${PATH}
    # The CUDA base image is CUDA 12.x, so default to the cu126 build of the latest torch (it runs on
    # any CUDA 12.6+ driver). Override with HORDE_WORKER_BACKEND (e.g. cu130 on a CUDA 13 base image).
    uv sync --locked --extra "${HORDE_WORKER_BACKEND:-cu126}"
fi

# Run GPU-specific setup scripts if they exist
if [ -f "${APP_HOME}/setup_${GPU_TYPE:-cuda}.sh" ]; then
    bash "${APP_HOME}/setup_${GPU_TYPE:-cuda}.sh"
fi

# Run the worker
if [ -e bridgeData.yaml ]; then
    uv run --no-sync python download_models.py
    exec uv run --no-sync python run_worker.py
else
    uv run --no-sync python download_models.py -e
    exec uv run --no-sync python run_worker.py -e
fi
