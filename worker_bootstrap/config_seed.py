"""Seed ``bridgeData.yaml`` from the bundled template on a fresh install (never clobbers an existing one)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

_CPU_TOKEN = "cpu"
_ALCHEMIST_LINE_RE = re.compile(r"^\s*alchemist\s*:.*$")
_DREAMER_LINE_RE = re.compile(r"^\s*dreamer\s*:.*$")


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
    """Make the seeded config explicitly alchemist-only (best-effort line edits; stdlib-only).

    A CPU install has image generation disabled, so the worker is seeded as ``alchemist: true`` (it runs
    the CPU-friendly alchemy forms) and ``dreamer: false`` (image generation deselected), so the role is
    explicit in the file rather than relying on the runtime coercing an empty model list. Each flag is
    edited only if its line is present, so an unexpectedly shaped template is left as-is for that flag.
    """
    try:
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return

    changed = False
    for index, line in enumerate(lines):
        newline = "\n" if line.endswith("\n") else ""
        if _ALCHEMIST_LINE_RE.match(line):
            lines[index] = f"alchemist: true{newline}"
            changed = True
        elif _DREAMER_LINE_RE.match(line):
            lines[index] = f"dreamer: false{newline}"
            changed = True

    if not changed:
        return
    try:
        target.write_text("".join(lines), encoding="utf-8")
    except OSError:
        return
