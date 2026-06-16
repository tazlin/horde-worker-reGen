#!/bin/bash
# Install / update the environment for AMD / ROCm (Linux). Thin wrapper: delegates to runtime.sh, which
# ensures uv and runs `bootstrap.py sync --backend rocm`. ROCm is installed ad-hoc (it is not in uv.lock);
# override the torch version / wheel index with HORDE_WORKER_ROCM_TORCH / HORDE_WORKER_ROCM_INDEX.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/runtime.sh" sync --backend rocm "$@"
