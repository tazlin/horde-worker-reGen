#!/bin/bash
# ROCm launch wrapper: ensure the (ad-hoc) ROCm environment exists, then run the passed command in it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Peered data dir (sibling of the worker folder, preserved across reinstalls); must match runtime.sh and
# worker_bootstrap/paths.py:data_root. HORDE_WORKER_DATA_DIR overrides the location.
HORDE_WORKER_DATA_DIR="${HORDE_WORKER_DATA_DIR:-${SCRIPT_DIR}-data}"
export HORDE_WORKER_DATA_DIR
mkdir -p "$HORDE_WORKER_DATA_DIR"
# Cache mode: "shared" leaves UV_CACHE_DIR unset (uv uses its own default cache, never auto-pruned);
# "isolated" (default) keeps a private, prunable cache here. Must match worker_bootstrap/paths.py.
if [ "$HORDE_WORKER_UV_CACHE_MODE" != "shared" ]; then
    : "${UV_CACHE_DIR:=$HORDE_WORKER_DATA_DIR/uv_cache}"; export UV_CACHE_DIR
fi
: "${UV_PYTHON_INSTALL_DIR:=$HORDE_WORKER_DATA_DIR/python}"; export UV_PYTHON_INSTALL_DIR
: "${UV_PYTHON_PREFERENCE:=only-managed}"; export UV_PYTHON_PREFERENCE
# AIWORKER_CACHE_HOME intentionally unset here so it cannot outrank bridgeData.yaml `cache_home`; the worker
# applies the peered <data>/models default at lowest precedence from HORDE_WORKER_DATA_DIR. See runtime.sh.

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    "$SCRIPT_DIR/update-runtime-rocm.sh" || exit 1
fi
# --no-sync so launching never re-syncs over the ad-hoc ROCm torch (which is not in uv.lock).
exec "$SCRIPT_DIR/bin/uv" run --no-sync "$@"
