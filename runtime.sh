#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Match update-runtime.sh: keep uv's cache on the install drive (next to .venv) rather than the home drive.
: "${UV_CACHE_DIR:=$SCRIPT_DIR/bin/uv_cache}"
export UV_CACHE_DIR
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    # Flag the bootstrap as launcher-driven so update-runtime.sh skips its standalone "next steps" and we
    # flow straight into starting the worker.
    HORDE_WORKER_FROM_LAUNCHER=1 "$SCRIPT_DIR/update-runtime.sh"
fi
# Prefer the bundled uv (installer/end-user setups); fall back to a uv on PATH (dev checkouts).
UV_BIN="$SCRIPT_DIR/bin/uv"
if [ ! -x "$UV_BIN" ]; then
    UV_BIN="uv"
fi
"$UV_BIN" run --no-sync "$@"
