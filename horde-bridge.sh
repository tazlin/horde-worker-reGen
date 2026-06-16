#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# jemalloc noticeably reduces the worker's memory fragmentation; preload it when present (Linux-specific,
# so it stays in this shell shim rather than the cross-platform Python brain).
for dir in /usr/lib /usr/local/lib /lib /lib64 /usr/lib/x86_64-linux-gnu; do
    if [ -f "$dir/libjemalloc.so.2" ]; then
        export LD_PRELOAD="$dir/libjemalloc.so.2"
        printf "Using jemalloc from %s\n" "$dir"
        break
    fi
done
if [ -z "${LD_PRELOAD:-}" ]; then
    printf "WARNING: jemalloc not found. You may run into memory issues! We recommend 'sudo apt install libjemalloc2'\n"
    read -n 1 -s -r -p "Press q to quit or any other key to continue: " key
    if [ "$key" = "q" ]; then printf "\n"; exit 1; fi
    printf "\n"
fi

echo "============================================"
echo "  AI Horde Worker"
echo "============================================"
echo ""
# The bridge path: ensure the environment, download/verify models, then run the headless worker.
exec "$SCRIPT_DIR/runtime.sh" launch bridge "$@"
