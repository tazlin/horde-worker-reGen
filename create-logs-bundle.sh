#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "This will create a logs bundle for the Horde Worker. It will include the following:"
echo "- A scrubbed version of the configuration file bridgeData.yaml"
echo "- The worker's log files"
echo "- Any worker state files from .horde_worker_regen, including captures in .horde_worker_regen/stats/"
echo "- Your downloaded models and their metadata"
echo "- System information (OS, Python version, etc.)"
echo "All of the above will be anonymized, details such as system usernames in paths (like C:\Users\YourName) and API keys will be removed."
echo "However, this is only a best-effort attempt, and you should review the bundle before sharing it if you have any concerns about sensitive information being included."
echo ""
exec "$SCRIPT_DIR/runtime.sh" horde-log bundle "$@"
