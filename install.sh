#!/usr/bin/env bash
# One-line installer for the AI Horde Worker (Linux).
#
#   curl -LsSf https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/install.sh | sh
#
# Downloads the latest release, builds the environment with uv (no pre-installed Python or git needed),
# seeds the config, and starts the dashboard. Re-running it updates in place.
#
# Options (environment variables, so they work with the curl | sh form):
#   HORDE_WORKER_DIR        install location (default: ~/.local/share/horde-worker)
#   HORDE_WORKER_BACKEND    cu128 | rocm | cpu (default: detected from the GPU)
#   HORDE_WORKER_NO_LAUNCH  set to skip auto-launching the dashboard after install
set -eu

OWNER="Haidra-Org"
REPO="horde-worker-reGen"
ASSET="horde-worker-reGen.zip"
RELEASE_URL="https://github.com/$OWNER/$REPO/releases/latest/download/$ASSET"

INSTALL_DIR="${HORDE_WORKER_DIR:-$HOME/.local/share/horde-worker}"
if [ "${1:-}" != "" ]; then INSTALL_DIR="$1"; fi
case "$INSTALL_DIR" in
    *" "*) echo "ERROR: the install path must not contain spaces: $INSTALL_DIR" >&2; exit 1 ;;
esac

echo ""
echo "=== AI Horde Worker installer ==="
echo "Install location: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

tmp_dir="$(mktemp -d)"
zip_path="$tmp_dir/horde-worker.zip"
echo "Downloading the latest release..."
if command -v curl >/dev/null 2>&1; then
    curl -LsSf "$RELEASE_URL" -o "$zip_path"
elif command -v wget >/dev/null 2>&1; then
    wget -qO "$zip_path" "$RELEASE_URL"
else
    echo "ERROR: need curl or wget to download the release." >&2
    exit 1
fi

echo "Extracting..."
if command -v unzip >/dev/null 2>&1; then
    unzip -oq "$zip_path" -d "$INSTALL_DIR"
elif command -v python3 >/dev/null 2>&1; then
    python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$zip_path" "$INSTALL_DIR"
else
    echo "ERROR: need unzip or python3 to extract the release." >&2
    exit 1
fi
rm -rf "$tmp_dir"

cd "$INSTALL_DIR"
if ! chmod +x ./*.sh 2>/dev/null; then
    echo "Note: could not mark the .sh scripts executable. If you later hit 'permission denied'," >&2
    echo "      run:  chmod +x \"$INSTALL_DIR\"/*.sh" >&2
fi

# GPU detection helpers. NVIDIA detection is broadened aggressively (a working CUDA card frequently
# has no nvidia-smi on PATH), but AMD is kept conservative: the ROCm build only works if the ROCm
# runtime is actually present, so a bare Radeon in lspci with no runtime is treated as "needs a
# decision" rather than silently installing a doomed build or the (~100x slower) CPU build.
has_nvidia_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 && return 0
    [ -e /proc/driver/nvidia/version ] && return 0
    [ -e /dev/nvidia0 ] && return 0
    command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -iq 'nvidia' && return 0
    return 1
}
has_amd_runtime() {
    command -v rocminfo >/dev/null 2>&1 && return 0
    [ -e /dev/kfd ] && return 0
    return 1
}
has_amd_hardware() {
    command -v lspci >/dev/null 2>&1 || return 1
    lspci 2>/dev/null | grep -iqE 'amd/ati|advanced micro devices|radeon' && return 0
    return 1
}

# GPU backend: explicit override wins; otherwise detect. Never silently fall back to CPU when a GPU
# is present, because the CPU build is ~100x slower and just looks like the worker is "broken".
BACKEND="${HORDE_WORKER_BACKEND:-}"
if [ -n "$BACKEND" ]; then
    echo "GPU backend: $BACKEND (from HORDE_WORKER_BACKEND)"
elif has_nvidia_gpu; then
    BACKEND="cu128"
    command -v nvidia-smi >/dev/null 2>&1 || \
        echo "Note: an NVIDIA GPU was detected but 'nvidia-smi' is not on PATH; using the CUDA build anyway."
    echo "GPU backend: cu128 (NVIDIA GPU detected)"
elif has_amd_runtime; then
    BACKEND="rocm"
    echo "GPU backend: rocm (AMD ROCm runtime detected)"
elif has_amd_hardware; then
    echo "ERROR: an AMD GPU was detected, but the ROCm runtime was not found (no rocminfo, no /dev/kfd)." >&2
    echo "Installing now would use the CPU build, which is roughly 100x slower." >&2
    echo "Install ROCm (https://rocm.docs.amd.com) and re-run, or force a choice with one of:" >&2
    echo "    HORDE_WORKER_BACKEND=rocm   # try the ROCm build" >&2
    echo "    HORDE_WORKER_BACKEND=cpu    # run on CPU (slow)" >&2
    exit 1
else
    echo "WARNING: no NVIDIA or AMD GPU detected; using the CPU build." >&2
    echo "CPU is roughly 100x slower than a GPU and is mainly useful for testing." >&2
    echo "If you have an NVIDIA GPU, install its drivers and re-run, or set HORDE_WORKER_BACKEND=cu128." >&2
    BACKEND="cpu"
fi

# Seed the config from the template on a fresh install (never clobbers an existing bridgeData.yaml).
if [ ! -f bridgeData.yaml ] && [ -f bridgeData_template.yaml ]; then
    cp bridgeData_template.yaml bridgeData.yaml
fi

# Co-locate uv's package cache with the install so it lands on the chosen drive (not the home drive) and
# stays on the same volume as .venv for hardlinking. update-runtime*.sh apply the same default; setting it
# here too makes the decision visible at the entry point. Respect a user-set UV_CACHE_DIR. ($PWD is the
# install dir: we cd'd into it above.)
: "${UV_CACHE_DIR:=$PWD/bin/uv_cache}"
export UV_CACHE_DIR

echo "Setting up the environment. The first run downloads Python and PyTorch and can take several minutes..."
export HORDE_WORKER_NONINTERACTIVE=1
case "$BACKEND" in
    rocm) ./update-runtime-rocm.sh ;;
    cpu) ./update-runtime.sh --cpu ;;
    *) ./update-runtime.sh ;;
esac

echo ""
echo "Installation complete (installed at $INSTALL_DIR)."

# Per-user application-menu entry so the dashboard is easy to reopen later. Never system-wide; opt out
# with HORDE_WORKER_NO_SHORTCUTS. Best-effort either way.
if [ -n "${HORDE_WORKER_NO_SHORTCUTS:-}" ]; then
    echo "Skipping the app-menu entry (HORDE_WORKER_NO_SHORTCUTS is set)."
else
    apps_dir="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    desktop_file="$apps_dir/ai-horde-worker.desktop"
    if mkdir -p "$apps_dir" 2>/dev/null && printf '%s\n' \
            '[Desktop Entry]' \
            'Type=Application' \
            'Name=AI Horde Worker' \
            'Comment=Share your GPU with the AI Horde' \
            "Exec=$INSTALL_DIR/horde-worker.sh" \
            "Path=$INSTALL_DIR" \
            'Terminal=true' \
            'Categories=Utility;' \
            > "$desktop_file" 2>/dev/null; then
        echo "Added an 'AI Horde Worker' entry to your applications menu."
    else
        echo "Note: could not add an applications-menu entry; launch $INSTALL_DIR/horde-worker.sh directly instead." >&2
    fi
fi

echo ""
echo "To open the dashboard again later:"
echo "  - run $INSTALL_DIR/horde-worker.sh  (add --terminal for the in-terminal UI), or"
echo "  - launch 'AI Horde Worker' from your applications menu."
echo "To update later: re-run the same install command."
echo ""

if [ -n "${HORDE_WORKER_NO_LAUNCH:-}" ]; then
    echo "Start it whenever you're ready with the command above."
else
    echo "Starting the worker dashboard..."
    exec ./horde-worker.sh
fi
