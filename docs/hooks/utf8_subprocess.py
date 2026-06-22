"""mkdocs hook: make text-mode subprocesses use UTF-8 regardless of platform locale.

With ``separate_signature: true`` the mkdocstrings Python handler formats every rendered signature
by piping it through ``ruff``/``black`` via ``subprocess.run(..., text=True)``. On Windows that pipe
encodes stdin with the locale codec (cp1252), which cannot represent characters that legitimately
appear in our docstrings and attribute defaults (e.g. U+2264 ``<=``), so the build aborts with a
``UnicodeEncodeError``. The same build succeeds on UTF-8 platforms (Linux CI) and under Python's
UTF-8 mode, which is why it only bites local Windows builds.

Defaulting text-mode subprocess pipes to UTF-8 fixes it at the source without changing any rendered
output and without requiring the build to be launched with ``PYTHONUTF8=1`` / ``-X utf8``.
"""

from __future__ import annotations

import subprocess
from typing import Any

_original_run = subprocess.run


def _utf8_run(*args: Any, **kwargs: Any) -> Any:
    # Only text-mode pipes carry an encoding; leave binary pipes untouched. setdefault respects an
    # explicit caller-provided encoding.
    if kwargs.get("text") or kwargs.get("universal_newlines"):
        kwargs.setdefault("encoding", "utf-8")
    return _original_run(*args, **kwargs)


subprocess.run = _utf8_run  # type: ignore[assignment]


def on_startup(**kwargs: Any) -> None:
    """No-op event handler so mkdocs registers this module as a hook; the patch is applied on import."""
