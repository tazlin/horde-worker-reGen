#!/bin/bash
# Single POSIX entry point: make sure uv exists, then hand every argument to the Python bootstrap brain
# (bootstrap.py). All install/update/launch logic lives in Python now; this script's only irreducible job
# is getting uv, the one thing that cannot yet be done in Python.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Isolation: ignore user site-packages, a stray PYTHONPATH, and a half-activated conda env.
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset CONDA_SHLVL
# Keep uv's cache and the managed Python it downloads on the install drive (next to .venv), not the home
# drive. Use a uv-managed CPython (only-managed) so the install is self-contained. Respect caller-set values.
: "${UV_CACHE_DIR:=$SCRIPT_DIR/bin/uv_cache}"; export UV_CACHE_DIR
: "${UV_PYTHON_INSTALL_DIR:=$SCRIPT_DIR/bin/python}"; export UV_PYTHON_INSTALL_DIR
: "${UV_PYTHON_PREFERENCE:=only-managed}"; export UV_PYTHON_PREFERENCE

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
