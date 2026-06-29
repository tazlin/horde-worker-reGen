"""Seed ``bridgeData.yaml`` from the bundled template on a fresh install (never clobbers an existing one)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

_CPU_TOKEN = "cpu"
_ALCHEMIST_LINE_RE = re.compile(r"^\s*alchemist\s*:.*$")


def seed_config(*, template: Path, target: Path, backend_token: str | None = None) -> bool:
    """Copy ``template`` to ``target`` when the target is absent and the template exists.

    On a CPU-only install the freshly seeded config is adjusted so the worker is useful out of the box:
    image generation is impractical on CPU, so ``alchemist`` is enabled (the worker runs the CPU-friendly
    alchemy forms). This only ever touches a config we just created from the template, never an existing
    user config, so it is a default rather than an override. The image model list is left as the template
    has it; the runtime coerces it away in CPU mode.

    Returns:
        True if a new config file was created, False if one already existed or the template was missing.
    """
    if target.exists() or not template.exists():
        return False
    shutil.copyfile(template, target)
    if backend_token == _CPU_TOKEN:
        _enable_alchemist_only(target)
    return True


def _enable_alchemist_only(target: Path) -> None:
    """Flip the seeded config's ``alchemist`` flag on (best-effort line edit; stdlib-only).

    A CPU install has image generation disabled, so without alchemist the worker would have nothing to do.
    Leaves the file untouched if no ``alchemist:`` line is found (an unexpectedly shaped template).
    """
    try:
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return
    for index, line in enumerate(lines):
        if _ALCHEMIST_LINE_RE.match(line):
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f"alchemist: true{newline}"
            break
    else:
        return
    try:
        target.write_text("".join(lines), encoding="utf-8")
    except OSError:
        return
