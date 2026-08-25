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
    local version existing_output existing_version triple os arch archive url tmp_dir candidate expected actual
    # This version MUST match [tool.uv] required-version in pyproject.toml. test_uv_version_consistency.py
    # enforces this: uv checks its version at runtime against required-version, so the version we download
    # here must satisfy it. Override with HORDE_WORKER_UV_VERSION to bump without editing this file.
    version="${HORDE_WORKER_UV_VERSION:-0.12.1}"
    existing_version=""
    if [ -x "$SCRIPT_DIR/bin/uv" ]; then
        # Probe from outside the project: an older uv may enforce the new pyproject.toml pin even for a
        # version check, which would hide the version we need in order to repair it.
        existing_output="$(cd "$HORDE_WORKER_DATA_DIR" && "$SCRIPT_DIR/bin/uv" --version 2>/dev/null)" || true
        case "$existing_output" in
            "uv "*) existing_version="${existing_output#uv }"; existing_version="${existing_version%% *}" ;;
        esac
        if [ "$existing_version" = "$version" ]; then
            return 0
        fi
        echo "Updating uv package manager from ${existing_version:-unknown} to ${version}..."
    else
        echo "Downloading uv package manager..."
    fi
    mkdir -p "$SCRIPT_DIR/bin"
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
    if [ -z "$triple" ]; then
        echo "Automatic uv bootstrap does not support ${os:-unknown}/${arch:-unknown}." >&2
        return 1
    fi
    if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
        echo "Automatic uv bootstrap requires curl and tar." >&2
        return 1
    fi
    archive="uv-${triple}.tar.gz"
    url="https://github.com/astral-sh/uv/releases/download/${version}/${archive}"
    tmp_dir="$(mktemp -d "$SCRIPT_DIR/bin/.uv-download.XXXXXX")" || return 1
    if ! curl -fL --retry 3 -o "$tmp_dir/$archive.sha256" "$url.sha256" || \
       ! curl -fL --retry 3 -o "$tmp_dir/$archive" "$url"; then
        rm -rf "$tmp_dir"
        return 1
    fi
    expected="$(awk 'NR == 1 { print $1 }' "$tmp_dir/$archive.sha256")"
    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "$tmp_dir/$archive" | awk '{ print $1 }')"
    elif command -v shasum >/dev/null 2>&1; then
        actual="$(shasum -a 256 "$tmp_dir/$archive" | awk '{ print $1 }')"
    else
        echo "Automatic uv bootstrap requires sha256sum or shasum." >&2
        rm -rf "$tmp_dir"
        return 1
    fi
    if [ -z "$expected" ] || [ "$actual" != "$expected" ]; then
        echo "Downloaded uv ${version} failed SHA-256 verification." >&2
        rm -rf "$tmp_dir"
        return 1
    fi
    if ! tar -xzf "$tmp_dir/$archive" -C "$tmp_dir"; then
        rm -rf "$tmp_dir"
        return 1
    fi
    candidate="$tmp_dir/uv-${triple}/uv"
    if [ ! -x "$candidate" ]; then
        echo "The verified uv archive did not contain the expected executable." >&2
        rm -rf "$tmp_dir"
        return 1
    fi
    existing_output="$(cd "$HORDE_WORKER_DATA_DIR" && "$candidate" --version 2>/dev/null)" || true
    existing_version=""
    case "$existing_output" in
        "uv "*) existing_version="${existing_output#uv }"; existing_version="${existing_version%% *}" ;;
    esac
    if [ "$existing_version" != "$version" ]; then
        echo "The verified uv artifact reported ${existing_version:-unknown}; expected ${version}." >&2
        rm -rf "$tmp_dir"
        return 1
    fi
    if ! chmod 755 "$candidate" || ! mv -f "$candidate" "$SCRIPT_DIR/bin/uv"; then
        rm -rf "$tmp_dir"
        return 1
    fi
    rm -rf "$tmp_dir"
    echo "Done."
    return 0
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
#
# --cache-dir gives THIS parent `uv run` its own tiny cache, deliberately NOT the worker UV_CACHE_DIR the
# children use. `uv run --script` holds a shared (read) flock on its cache's .lock for the whole script
# lifetime, while the post-sync `uv cache prune` child wants an exclusive (write) flock on the same file.
# Pointing them at the same cache deadlocks prune until it times out (a ~5 min apparent hang). UV_CACHE_DIR
# is still exported, so the sync/prune children below inherit the worker cache; only this parent is moved.
exec "$SCRIPT_DIR/bin/uv" run --python 3.12 --no-project \
    --cache-dir "$HORDE_WORKER_DATA_DIR/bootstrap_cache" \
    --script "$SCRIPT_DIR/bootstrap.py" "$@"
