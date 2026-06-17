#!/bin/bash
# Single POSIX entry point: make sure uv exists, then hand every argument to the Python bootstrap brain
# (bootstrap.py). All install/update/launch logic lives in Python now; this script's only irreducible job
# is getting uv, the one thing that cannot yet be done in Python.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Isolation: ignore user site-packages, a stray PYTHONPATH, and a half-activated conda env.
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset CONDA_SHLVL
# Keep uv's cache, the managed Python, and downloaded models in a peered data dir: a sibling of the worker
# folder (same name with a -data suffix) that is preserved when the worker folder is deleted or reinstalled,
# so a user starting fresh cannot lose their cached deps or model weights. HORDE_WORKER_DATA_DIR overrides
# the location (e.g. another drive). This must match worker_bootstrap/paths.py:data_root. Use a uv-managed
# CPython (only-managed) so the install is self-contained. Respect caller-set values for each.
HORDE_WORKER_DATA_DIR="${HORDE_WORKER_DATA_DIR:-${SCRIPT_DIR}-data}"
export HORDE_WORKER_DATA_DIR
mkdir -p "$HORDE_WORKER_DATA_DIR"
# Cache mode: "shared" leaves UV_CACHE_DIR unset so uv uses its own default (system) cache a power user
# already populates for other projects (no 7-10 GB duplicate); the worker then never auto-prunes it.
# "isolated" (default) keeps a private cache in the data dir that we can prune safely. Must match
# worker_bootstrap/paths.py:uv_cache_mode. A caller-set UV_CACHE_DIR is respected in either mode.
if [ "$HORDE_WORKER_UV_CACHE_MODE" != "shared" ]; then
    : "${UV_CACHE_DIR:=$HORDE_WORKER_DATA_DIR/uv_cache}"; export UV_CACHE_DIR
fi
: "${UV_PYTHON_INSTALL_DIR:=$HORDE_WORKER_DATA_DIR/python}"; export UV_PYTHON_INSTALL_DIR
: "${UV_PYTHON_PREFERENCE:=only-managed}"; export UV_PYTHON_PREFERENCE
# Deliberately NOT setting AIWORKER_CACHE_HOME here. It would outrank `cache_home` in bridgeData.yaml (the
# worker treats a pre-set env var as higher precedence than config). The peered <data>/models default is
# applied at the LOWEST precedence inside the worker (load_env_vars.py, from HORDE_WORKER_DATA_DIR) so the
# ladder stays env var > cache_home > peered default.

ensure_uv() {
    [ -x "$SCRIPT_DIR/bin/uv" ] && return 0
    echo "Downloading uv package manager..."
    mkdir -p "$SCRIPT_DIR/bin"
    local version triple os arch
    version="${HORDE_WORKER_UV_VERSION:-0.11.21}"
    os="$(uname -s)"; arch="$(uname -m)"
    case "$os" in
        Linux)  case "$arch" in
                    x86_64) triple="x86_64-unknown-linux-gnu" ;;
                    aarch64|arm64) triple="aarch64-unknown-linux-gnu" ;;
                esac ;;
        Darwin) case "$arch" in
                    x86_64) triple="x86_64-apple-darwin" ;;
                    arm64|aarch64) triple="aarch64-apple-darwin" ;;
                esac ;;
    esac
    # Preferred: a direct, pinned download of the uv standalone tarball with curl + tar (no shell piping of
    # remote code). The tarball nests the binaries under uv-<triple>/, so --strip-components=1 flattens them
    # into bin/.
    if [ -n "$triple" ] && command -v curl >/dev/null 2>&1 && command -v tar >/dev/null 2>&1; then
        local url="https://github.com/astral-sh/uv/releases/download/${version}/uv-${triple}.tar.gz"
        if curl -fL --retry 3 "$url" | tar -xz --strip-components=1 -C "$SCRIPT_DIR/bin" 2>/dev/null; then
            [ -x "$SCRIPT_DIR/bin/uv" ] && { echo "Done."; return 0; }
        fi
    fi
    # Fallback: the official installer script (covers platforms without our pinned tarball mapping).
    echo "Falling back to the astral install script..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$SCRIPT_DIR/bin" sh
    [ -x "$SCRIPT_DIR/bin/uv" ]
}

if ! ensure_uv; then
    echo "" >&2
    echo "ERROR: Could not install uv (the package manager)." >&2
    echo "  - Confirm GitHub and astral.sh are reachable (proxy/firewall?)." >&2
    echo "  - Or place a uv binary at \"$SCRIPT_DIR/bin/uv\" and re-run." >&2
    exit 1
fi

# --no-project + PEP 723 inline metadata means uv ignores the project and runs bootstrap.py in a tiny
# stdlib-only environment, so it works before .venv exists. --python 3.12 pins a managed CPython.
exec "$SCRIPT_DIR/bin/uv" run --python 3.12 --no-project --script "$SCRIPT_DIR/bootstrap.py" "$@"
