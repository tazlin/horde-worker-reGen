#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Keep uv's package cache next to the install instead of on the home drive (~/.cache). The cache is several
# GB; defaulting it here keeps it on the same volume as .venv so uv can hardlink instead of full-copying, and
# off the home drive. Respect an existing UV_CACHE_DIR so power users can still point at a shared global cache.
: "${UV_CACHE_DIR:=$SCRIPT_DIR/bin/uv_cache}"
export UV_CACHE_DIR

echo "============================================"
echo "  AI Horde Worker - Install / Update (ROCm)"
echo "============================================"
echo ""

# Install uv if not present
if [ ! -f "$SCRIPT_DIR/bin/uv" ]; then
    echo "Downloading uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCRIPT_DIR/bin" sh
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Failed to download uv. Check your internet connection."
        exit 1
    fi
    echo "Done."
    echo ""
fi

# Determine if the user has a flash attention supported card.
SUPPORTED_CARD=$(rocminfo | grep -c -e gfx1100 -e gfx1101 -e gfx1102)
if [ "$SUPPORTED_CARD" -gt 0 ]; then export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:=TRUE}"; fi

# ROCm is NOT in uv.lock: torch 2.12.0 (the locked line) has no ROCm wheel, and PyTorch has not
# published the pytorch-triton-rocm that the torch 2.10-2.12 rocm7.x wheels hard-depend on. So we
# install the ROCm PyTorch stack AD-HOC -- torch 2.9.1 on ROCm 6.4 by default -- on top of a base sync.
# This path is best-effort and not pinned by uv.lock. Override with HORDE_WORKER_ROCM_TORCH /
# HORDE_WORKER_ROCM_INDEX if you need a different torch version or ROCm index.
ROCM_TORCH="${HORDE_WORKER_ROCM_TORCH:-2.9.1}"
ROCM_INDEX="${HORDE_WORKER_ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.4}"

echo "Installing the base environment (everything except the GPU torch build)..."
echo "(This may take a few minutes on first run...)"
echo ""
"$SCRIPT_DIR/bin/uv" sync --locked --extra cpu
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: base installation failed."
    echo "  - Try deleting the .venv folder and running this script again."
    echo "  - If the problem persists, ask for help in #local-workers on Discord."
    exit 1
fi

echo ""
echo "Installing the ROCm PyTorch stack ad-hoc (torch ${ROCM_TORCH} from ${ROCM_INDEX})..."
echo "(Not locked by uv.lock; see README_advanced.md.)"
"$SCRIPT_DIR/bin/uv" pip install --reinstall \
    "torch==${ROCM_TORCH}" torchvision torchaudio pytorch-triton-rocm \
    --extra-index-url "${ROCM_INDEX}"
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: ad-hoc ROCm PyTorch install failed. Check that ${ROCM_INDEX} publishes a"
    echo "       torch==${ROCM_TORCH} build (and its pytorch-triton-rocm), or set"
    echo "       HORDE_WORKER_ROCM_TORCH / HORDE_WORKER_ROCM_INDEX and re-run."
    exit 1
fi

# Ensure no NVIDIA packages leaked in
"$SCRIPT_DIR/bin/uv" pip uninstall pynvml nvidia-ml-py 2>/dev/null

# Check if we are running in WSL2
WSL_KERNEL=$(uname -a | grep -c -e WSL2 )
if [ "$WSL_KERNEL" -gt 0 ]; then
    export IN_WSL="TRUE"
    echo "WSL environment detected. Patching ROCm libhsa-runtime64.so"
    for i in $(find ./ -iname libhsa-runtime64.so); do cp /opt/rocm/lib/libhsa-runtime64.so "$i"; done
fi

# Install AMD Go Fast optimizations
"$SCRIPT_DIR/bin/uv" run "$SCRIPT_DIR/horde_worker_regen/amd_go_fast/install_amd_go_fast.sh"
