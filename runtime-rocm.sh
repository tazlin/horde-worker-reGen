#!/bin/bash
# ROCm launch wrapper: ensure the (ad-hoc) ROCm environment exists, then run the passed command in it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${UV_CACHE_DIR:=$SCRIPT_DIR/bin/uv_cache}"; export UV_CACHE_DIR
: "${UV_PYTHON_INSTALL_DIR:=$SCRIPT_DIR/bin/python}"; export UV_PYTHON_INSTALL_DIR
: "${UV_PYTHON_PREFERENCE:=only-managed}"; export UV_PYTHON_PREFERENCE

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    "$SCRIPT_DIR/update-runtime-rocm.sh" || exit 1
fi
# --no-sync so launching never re-syncs over the ad-hoc ROCm torch (which is not in uv.lock).
exec "$SCRIPT_DIR/bin/uv" run --no-sync "$@"
