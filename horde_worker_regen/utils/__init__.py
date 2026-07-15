"""Utility modules for horde-worker-reGen."""

import sys


def get_system_appropriate_updater() -> str | None:
    """Get the system-appropriate updater script name.

    Returns:
        The updater script name for the current platform, or None if the platform is unsupported.
    """
    if sys.platform == "win32":
        return "update.cmd"
    if sys.platform in ("linux", "darwin"):
        return "update.sh"

    return "update.cmd / update.sh (unsupported platform?)"
