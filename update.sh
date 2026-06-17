#!/bin/bash
# One-touch full in-place upgrade for the AI Horde Worker (Linux / macOS).
#
# Re-runs the official one-line installer against THIS folder: it downloads the latest release, extracts
# it over the worker source, and re-syncs dependencies. The peered data dir (this folder's `-data` sibling
# holding the uv cache, managed Python, and models) is left untouched, so an upgrade never re-downloads
# models. For a deps-only re-sync without fetching a new release, use update-runtime.sh instead.
#
# The installer is run from memory (curl | sh), never from a copy inside this folder, because the upgrade
# overwrites this folder's files; piping the freshly downloaded script avoids overwriting a running script.
set -eu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Point the installer at this install, accept the notice (consent was recorded at first install), and do
# not auto-launch the dashboard after the upgrade.
export HORDE_WORKER_DIR="$SCRIPT_DIR"
export HORDE_WORKER_ASSUME_YES=1
export HORDE_WORKER_NO_LAUNCH=1

exec sh -c 'curl -LsSf https://raw.githubusercontent.com/tazlin/horde-worker-reGen/main/install.sh | sh'
