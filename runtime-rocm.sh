#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Match update-runtime-rocm.sh: keep uv's cache on the install drive (next to .venv) rather than the home drive.
: "${UV_CACHE_DIR:=$SCRIPT_DIR/bin/uv_cache}"
export UV_CACHE_DIR
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    "$SCRIPT_DIR/update-runtime-rocm.sh"
fi
"$SCRIPT_DIR/bin/uv" run "$@"
