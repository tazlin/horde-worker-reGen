#!/bin/bash
# Download/verify the configured models, then exit (no worker started).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/runtime.sh" preload
