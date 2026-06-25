#!/usr/bin/env bash
# One-line installer for the AI Horde Worker (Linux / macOS).
#
#   curl -LsSf https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/install.sh | sh
#
# Downloads the latest release, shows a notice of what will be installed and from where, asks for
# confirmation, then hands off to the bundled runtime.sh, which installs uv and runs the Python bootstrap
# (GPU detection, dependency sync, config seeding, launch). It provides its own private Python; git must
# already be installed (on Linux/macOS it is a one-line package install). Re-running it updates in place.
#
# Options (environment variables, so they work with the curl | sh form):
#   HORDE_WORKER_DIR         install location (default: ./horde-worker in the current directory)
#   HORDE_WORKER_REPO        install from a fork (owner/repo; default Haidra-Org/horde-worker-reGen)
#   HORDE_WORKER_BACKEND     cu126 | cu130 | cu132 | rocm | cpu (default: detected from the GPU/driver)
#   HORDE_WORKER_FEATURES    optional feature extras to install: comma/space list of post-processing,
#                            controlnet, or 'none' (default: all on NVIDIA/CPU, none on other backends)
#   HORDE_WORKER_ASSUME_YES  accept the install notice without prompting (required when piped, no terminal)
#   HORDE_WORKER_SHORTCUTS   create the applications-menu entry without prompting
#   HORDE_WORKER_NO_SHORTCUTS skip the applications-menu entry entirely
#   HORDE_WORKER_NO_LAUNCH   skip the "Start now?" prompt and do not launch after install
set -eu

# The owner/repo to install from. Defaults to the canonical production repo; a fork overrides it by setting
# HORDE_WORKER_REPO (e.g. baked into its own one-liner) rather than editing this file, so the committed
# default never diverges from upstream. The resolved value is recorded in bin/install-info, so the in-place
# self-updater pulls future releases from the same origin (see worker_bootstrap/updater.py resolve_update_repo).
REPO_SLUG="${HORDE_WORKER_REPO:-Haidra-Org/horde-worker-reGen}"
case "$REPO_SLUG" in
    */*/*) echo "ERROR: HORDE_WORKER_REPO must be 'owner/repo' (got '$REPO_SLUG')." >&2; exit 1 ;;
    */*) ;;
    *) echo "ERROR: HORDE_WORKER_REPO must be 'owner/repo' (got '$REPO_SLUG')." >&2; exit 1 ;;
esac
OWNER="${REPO_SLUG%/*}"
REPO="${REPO_SLUG#*/}"
ASSET="horde-worker-reGen.zip"
RELEASE_URL="https://github.com/$OWNER/$REPO/releases/latest/download/$ASSET"

# Default into a named subfolder of the current directory, not the home drive: the worker plus its model
# downloads run to many GB. A subfolder keeps the loose-file bundle self-contained.
INSTALL_DIR="${HORDE_WORKER_DIR:-$PWD/horde-worker}"
if [ "${1:-}" != "" ]; then INSTALL_DIR="$1"; fi
case "$INSTALL_DIR" in
    *" "*) echo "ERROR: the install path must not contain spaces: $INSTALL_DIR" >&2; exit 1 ;;
esac

# Guard against running the installer from inside an existing horde-worker installation.
# When no explicit destination was given, the default ($PWD/horde-worker) would create a
# nested copy. Presence of runtime.sh in the CWD is a reliable sentinel for an existing install.
if [ -z "${HORDE_WORKER_DIR:-}" ] && [ -z "${1:-}" ] && [ -f "$PWD/runtime.sh" ]; then
    echo "ERROR: the current directory looks like an existing horde-worker installation (runtime.sh is here)." >&2
    echo "       Installing from here would create a nested copy at: $INSTALL_DIR" >&2
    echo "       To update the current install, run:  $PWD/update.sh" >&2
    echo "       To install elsewhere, cd to another directory first, or set:" >&2
    echo "         HORDE_WORKER_DIR=/path/to/new-location" >&2
    exit 1
fi

echo ""
echo "=== AI Horde Worker installer ==="
echo "Install location: $INSTALL_DIR"

# Download and extract the release into a temp dir, then lay it down and overlay it onto the install with
# the same mirror-pruning logic the self-updater uses (runtime.sh apply-bundle, below). Extracting straight
# into the install with a plain `unzip -o` only overwrites files and never removes ones the new release
# dropped, so a reinstall over an older version would leave a renamed/removed module on the import path to
# shadow the new code.
tmp_dir="$(mktemp -d)"
zip_path="$tmp_dir/horde-worker.zip"
bundle_dir="$tmp_dir/bundle"
mkdir -p "$bundle_dir"

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
    unzip -oq "$zip_path" -d "$bundle_dir"
elif command -v python3 >/dev/null 2>&1; then
    python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$zip_path" "$bundle_dir"
else
    echo "ERROR: need unzip or python3 to extract the release." >&2
    exit 1
fi

# Lay the bundle down (shims, bootstrap, source). This overwrites in place but, like the old plain unzip,
# does not by itself prune modules the new release dropped; the apply-bundle overlay below removes those.
# The shims must be written here, not by the overlay, so the overlay can run *through* runtime.sh without
# overwriting the running launcher mid-run.
mkdir -p "$INSTALL_DIR"
cp -R "$bundle_dir/." "$INSTALL_DIR/"

# Record how this worker was installed and from where, so the in-place self-updater pulls future releases
# from the same origin (this fork/account) rather than a hardcoded default. Lives under bin/ (preserved
# across updates, removed on uninstall).
mkdir -p "$INSTALL_DIR/bin"
printf 'method=one-line\nrepo=%s/%s\n' "$OWNER" "$REPO" > "$INSTALL_DIR/bin/install-info"

cd "$INSTALL_DIR"
if ! chmod +x ./*.sh 2>/dev/null; then
    echo "Note: could not mark the .sh scripts executable. If you later hit 'permission denied'," >&2
    echo "      run:  chmod +x \"$INSTALL_DIR\"/*.sh" >&2
fi

# Show what is about to be installed (and from where) and get consent before any heavy download. Under
# `curl | sh` our stdin is the piped script, so we read the answer from the controlling terminal
# (/dev/tty); when there is none (true headless), require HORDE_WORKER_ASSUME_YES instead of guessing.
if [ -f "$INSTALL_DIR/INSTALL_NOTICE.txt" ]; then
    echo ""
    cat "$INSTALL_DIR/INSTALL_NOTICE.txt"
    echo ""
fi
if [ -z "${HORDE_WORKER_ASSUME_YES:-}" ]; then
    if [ -r /dev/tty ] && [ -w /dev/tty ]; then
        printf 'Proceed with installation? [y/N] ' > /dev/tty
        reply=""
        read -r reply < /dev/tty || reply=""
        case "$reply" in
            [Yy]|[Yy][Ee][Ss]) ;;
            *) echo "Installation cancelled. The downloaded files are in $INSTALL_DIR; delete that folder to remove them."; exit 1 ;;
        esac
        export HORDE_WORKER_ASSUME_YES=1
    else
        echo "ERROR: no interactive terminal to accept the notice above (stdin is the piped script)." >&2
        echo "       Re-run accepting it explicitly, e.g.:" >&2
        echo "         curl -LsSf https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/install.sh | HORDE_WORKER_ASSUME_YES=1 sh" >&2
        exit 1
    fi
fi

# Overlay the freshly downloaded bundle onto the install through the shared, mirror-pruning overlay so the
# one-line (re)install and the in-place self-updater behave identically: a module renamed or removed since
# the installed version is pruned from the import roots, while .venv, bin, models, and bridgeData.yaml are
# preserved. This also fetches uv (into bin/), which the dependency sync below reuses.
if ! ./runtime.sh apply-bundle "$bundle_dir"; then
    echo "ERROR: could not lay down the worker files (see the output above)." >&2
    exit 1
fi
rm -rf "$tmp_dir"

# Everything else (install uv, detect the GPU, seed bridgeData.yaml, sync dependencies) is the bootstrap's
# job now, so the one-liner, the regular launchers and every platform run identical logic. runtime.sh
# installs uv and runs bootstrap.py. --no-launch: we start the dashboard ourselves below. A pre-set
# HORDE_WORKER_BACKEND still overrides detection (e.g. 'cpu', or 'rocm' for AMD on Linux).
echo "Setting up the environment. The first run downloads Python and PyTorch and can take several minutes..."
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
echo "Models, the uv cache, and Python live in $INSTALL_DIR-data (a sibling folder), which is preserved if"
echo "you delete or reinstall the worker folder, so your models are not lost. Set HORDE_WORKER_DATA_DIR"
echo "before installing to put it elsewhere (e.g. another drive)."

# Per-user application-menu entry so the dashboard is easy to reopen later. Opt-in (conservative default):
# we ask, defaulting to No. HORDE_WORKER_SHORTCUTS adds it without asking; HORDE_WORKER_NO_SHORTCUTS skips
# it. Never system-wide; best-effort.
make_shortcut=""
if [ -n "${HORDE_WORKER_NO_SHORTCUTS:-}" ]; then
    echo "Skipping the app-menu entry (HORDE_WORKER_NO_SHORTCUTS is set)."
elif [ -n "${HORDE_WORKER_SHORTCUTS:-}" ]; then
    make_shortcut=1
elif [ -r /dev/tty ] && [ -w /dev/tty ]; then
    printf "Add an 'AI Horde Worker' entry to your applications menu? [y/N] " > /dev/tty
    reply=""
    read -r reply < /dev/tty || reply=""
    case "$reply" in [Yy]|[Yy][Ee][Ss]) make_shortcut=1 ;; esac
fi
if [ -n "$make_shortcut" ]; then
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
echo "  - run $INSTALL_DIR/horde-worker.sh  (add --terminal for the in-terminal UI, --headless for no UI), or"
echo "  - launch 'AI Horde Worker' from your applications menu."
echo "To update later: run $INSTALL_DIR/update.sh or re-run the same install command (both keep $INSTALL_DIR-data intact)."
echo ""

if [ -n "${HORDE_WORKER_NO_LAUNCH:-}" ]; then
    echo "Start it whenever you're ready with the command above."
elif [ -r /dev/tty ] && [ -w /dev/tty ]; then
    printf "Start the worker now? [(y)es / (n)o / (t)erminal UI / (h)eadless]: " > /dev/tty
    while true; do
        reply=""
        read -r reply < /dev/tty || reply=""
        case "$reply" in
            [Yy]|[Yy][Ee][Ss])
                echo "Starting the worker dashboard..."
                exec ./horde-worker.sh
                ;;
            [Nn]|[Nn][Oo]|"")
                echo "Start it whenever you're ready with the command above."
                break
                ;;
            [Tt])
                echo "Starting the in-terminal UI..."
                exec ./horde-worker.sh --terminal </dev/tty
                ;;
            [Hh])
                echo "Starting the worker in headless mode..."
                exec ./horde-worker.sh --headless
                ;;
            *)
                printf "Please enter y, n, t, or h: " > /dev/tty
                ;;
        esac
    done
else
    echo "Start it whenever you're ready with the command above."
fi
