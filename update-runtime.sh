#!/bin/bash
# Install / update the environment. Thin wrapper kept under its historical name: it delegates to
# runtime.sh, which ensures uv and runs `bootstrap.py sync`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/runtime.sh" sync "$@"
