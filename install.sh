#!/usr/bin/env bash
# One-line installer for the AI Horde Worker (Linux / macOS).
#
#   curl -LsSf https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/install.sh | sh
#
# Downloads the latest release, then hands off to the bundled runtime.sh, which installs uv and runs the
# Python bootstrap (GPU detection, dependency sync, config seeding, launch). No pre-installed Python or git
# needed. Re-running it updates in place.
#
# Options (environment variables, so they work with the curl | sh form):
#   HORDE_WORKER_DIR        install location (default: ./horde-worker in the current directory)
#   HORDE_WORKER_BACKEND    cu126 | cu130 | cu132 | rocm | cpu (default: detected from the GPU/driver)
#   HORDE_WORKER_NO_LAUNCH  set to skip auto-launching the dashboard after install
set -eu

OWNER="tazlin"
REPO="horde-worker-reGen"
ASSET="horde-worker-reGen.zip"
RELEASE_URL="https://github.com/$OWNER/$REPO/releases/latest/download/$ASSET"

# Default into a named subfolder of the current directory, not the home drive: the worker plus its model
# downloads run to many GB. A subfolder keeps the loose-file bundle self-contained.
INSTALL_DIR="${HORDE_WORKER_DIR:-$PWD/horde-worker}"
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

# Everything else (install uv, detect the GPU, seed bridgeData.yaml, sync dependencies) is the bootstrap's
# job now, so the one-liner, the regular launchers and every platform run identical logic. runtime.sh
# installs uv and runs bootstrap.py. --no-launch: we start the dashboard ourselves below. A pre-set
# HORDE_WORKER_BACKEND still overrides detection (e.g. 'cpu', or 'rocm' for AMD on Linux).
echo "Setting up the environment. The first run downloads Python and PyTorch and can take several minutes..."
export HORDE_WORKER_NONINTERACTIVE=1
if ! ./runtime.sh install --no-launch; then
    echo "" >&2
    echo "ERROR: environment setup failed (see the output above). Deleting .venv and re-running often helps." >&2
    exit 1
fi
# Trust the artifact, not just the exit code: a real install must have produced a virtual environment.
if [ ! -d .venv ]; then
    echo "ERROR: environment setup did not produce a .venv. See the output above; delete .venv and re-run." >&2
    exit 1
fi

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
