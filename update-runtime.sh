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

# The uv extra is the torch *build* (cu126/cu130/cu132/cpu), all on the latest torch (2.12.0).
#   BUILD precedence: explicit flag (--cu126/--cu130/--cu132/--cpu) > HORDE_WORKER_BACKEND > bin/backend
#                     (written by an installer from the GPU driver) > cu126 fallback (the broadest
#                     CUDA-12 build: runs on any CUDA 12.6+ driver and, via driver back-compat, CUDA 13).
# Older torch lines and ROCm are not locked; install those ad-hoc (see pyproject.toml / README_advanced),
# or use ./update-runtime-rocm.sh for AMD.
BUILD=""
for arg in "$@"; do
    case "$arg" in
        --cpu) BUILD="cpu" ;;
        --cu126) BUILD="cu126" ;;
        --cu130) BUILD="cu130" ;;
        --cu132) BUILD="cu132" ;;
    esac
done
if [ -z "$BUILD" ] && [ -n "${HORDE_WORKER_BACKEND:-}" ]; then BUILD="$HORDE_WORKER_BACKEND"; fi
if [ -z "$BUILD" ] && [ -f "$SCRIPT_DIR/bin/backend" ]; then BUILD="$(tr -d '[:space:]' < "$SCRIPT_DIR/bin/backend")"; fi
if [ -z "$BUILD" ]; then BUILD="cu126"; fi

# torch 2.12.0 publishes cu126 (not cu128) for CUDA 12; transparently map a legacy cu128 request so an
# existing install whose bin/backend still says cu128 keeps working.
if [ "$BUILD" = "cu128" ]; then
    echo "Note: torch 2.12.0 has no cu128 build; using cu126 (runs on any CUDA 12.6+ driver)."
    BUILD="cu126"
fi

GPU_EXTRA="$BUILD"
# Only the builds 2.12.0 actually publishes are locked. If an unsupported build was requested (e.g.
# rocm, or some other token), explain the ad-hoc paths rather than failing cryptically.
if [ -f "$SCRIPT_DIR/pyproject.toml" ] && ! grep -q "^${GPU_EXTRA} = " "$SCRIPT_DIR/pyproject.toml"; then
    echo "ERROR: '${GPU_EXTRA}' is not a locked build. The lock provides the latest torch (2.12.0) for:" >&2
    echo "       cu126 (CUDA 12.6+), cu130 (CUDA 13.x), cu132 (CUDA 13.2), cpu." >&2
    echo "       For AMD/ROCm run ./update-runtime-rocm.sh; for an older torch line install ad-hoc, e.g." >&2
    echo "       UV_TORCH_BACKEND=auto uv pip install torch torchvision torchaudio" >&2
    exit 1
fi

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
# When runtime.sh bootstraps on first launch it sets HORDE_WORKER_FROM_LAUNCHER and then starts the
# worker itself, so skip the manual "next steps" in that flow. A direct run still shows them.
if [ -z "${HORDE_WORKER_FROM_LAUNCHER:-}" ]; then
    echo "Next steps:"
    echo "  1. Edit bridgeData.yaml with your API key and worker name"
    echo "  2. Run ./horde-bridge.sh to start the worker"
    echo "     (or ./horde-worker.sh for the interactive launcher)"
    echo ""
fi
