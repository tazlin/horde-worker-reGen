#!/bin/bash
# One-touch full in-place upgrade for the AI Horde Worker (Linux / macOS).
#
# Delegates to the bootstrap brain's `update` action: it queries the latest release, downloads the release
# bundle directly, verifies its published SHA-256 BEFORE writing anything, then overlays the worker's
# Python source and lockfile and re-syncs dependencies. The peered data dir (this folder's `-data` sibling
# holding the uv cache, managed Python, and models) is left untouched, so an upgrade never re-downloads
# models. The shell shims (these .cmd/.sh launchers) are intentionally not overwritten, so the script
# driving this update is never modified out from under itself.
#
# This replaces piping a remote installer into a shell: nothing is executed from the network, and the
# download is integrity-checked before it touches the install. For a deps-only re-sync without fetching a
# new release, use update-runtime.sh instead.
set -eu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/runtime.sh" update --yes "$@"
