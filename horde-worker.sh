#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default: serve the dashboard in your web browser. Pass --terminal for the in-terminal UI.
# Power users can bind the LAN with: ./horde-worker.sh --host 0.0.0.0  (unauthenticated; opt-in)
if [ "$1" = "--terminal" ]; then
    shift
    exec "$SCRIPT_DIR/runtime.sh" launch terminal "$@"
fi

echo "Starting the AI Horde Worker dashboard..."
echo "This window runs the worker: closing it (or pressing Ctrl+C here) stops the worker."
echo "Closing just the dashboard window/tab leaves the worker running; reopen to reconnect."
echo "Pass --terminal for the in-terminal UI."
echo ""
exec "$SCRIPT_DIR/runtime.sh" launch web "$@"
