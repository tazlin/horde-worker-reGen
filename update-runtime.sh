#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep uv's package cache next to the install instead of on the home drive (~/.cache). The cache is several
# GB; defaulting it here keeps it on the same volume as .venv so uv can hardlink instead of full-copying, and
# off the home drive. Respect an existing UV_CACHE_DIR so power users can still point at a shared global cache.
: "${UV_CACHE_DIR:=$SCRIPT_DIR/bin/uv_cache}"
export UV_CACHE_DIR

echo "============================================"
echo "  AI Horde Worker - Install / Update"
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

# Parse arguments for GPU backend selection (default cu128). ROCm has its own update-runtime-rocm.sh.
GPU_EXTRA="cu128"
for arg in "$@"; do
    case "$arg" in
        --cpu) GPU_EXTRA="cpu" ;;
        --rocm) GPU_EXTRA="rocm" ;;
        --directml) GPU_EXTRA="directml" ;;
    esac
done

echo "Installing dependencies for GPU backend: $GPU_EXTRA"
echo "(This may take a few minutes on first run...)"
echo ""
"$SCRIPT_DIR/bin/uv" sync --locked --extra "$GPU_EXTRA"
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Installation failed."
    echo "  - Try deleting the .venv folder and running this script again."
    echo "  - If the problem persists, ask for help in #local-workers on Discord."
    exit 1
fi

echo ""
echo "============================================"
echo "  Installation complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit bridgeData.yaml with your API key and worker name"
echo "  2. Run ./horde-bridge.sh to start the worker"
echo "     (or ./horde-worker.sh for the interactive launcher)"
echo ""
