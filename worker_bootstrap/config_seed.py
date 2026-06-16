"""Seed ``bridgeData.yaml`` from the bundled template on a fresh install (never clobbers an existing one)."""

from __future__ import annotations

import shutil
from pathlib import Path


def seed_config(*, template: Path, target: Path) -> bool:
    """Copy ``template`` to ``target`` when the target is absent and the template exists.

    Returns:
        True if a new config file was created, False if one already existed or the template was missing.
    """
    if target.exists() or not template.exists():
        return False
    shutil.copyfile(template, target)
    return True
