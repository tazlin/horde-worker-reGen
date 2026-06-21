#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Three peer launch options:
#   (default)    web dashboard in your browser (a pop-up window).
#   --terminal   the in-terminal Textual UI (console, no browser).
#   --headless   no UI: run the worker in the foreground, printing to this console.
# Power users can bind the LAN with: ./horde-worker.sh --host 0.0.0.0  (unauthenticated; opt-in)
if [ "$1" = "--web" ]; then
    shift
    exec "$SCRIPT_DIR/runtime.sh" launch web "$@"
fi

if [ "$1" = "--terminal" ]; then
    shift
    exec "$SCRIPT_DIR/runtime.sh" launch terminal "$@"
fi

if [ "$1" = "--headless" ]; then
    shift
    exec "$SCRIPT_DIR/runtime.sh" launch bridge "$@"
fi

echo "Starting the AI Horde Worker dashboard..."
echo "This window runs the worker: closing it (or pressing Ctrl+C here) stops the worker."
echo "Closing just the dashboard window/tab leaves the worker running; reopen to reconnect."
echo "Pass --terminal for the in-terminal UI, or --headless for no UI."
echo ""
exec "$SCRIPT_DIR/runtime.sh" launch web "$@"
