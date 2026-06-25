"""Minimal placeholder image generation for dry-run and fake worker processes.

Kept free of any heavy imports (no PIL, no torch) so it can be used by processes
that must start without the ML dependency stack.
"""

from __future__ import annotations

import base64
import functools
import struct
import zlib


@functools.cache
def make_dummy_png_bytes() -> bytes:
    """Return the bytes of a valid 1x1 black RGB PNG."""
    sig = b"\x89PNG\r\n\x1a\n"

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + ctype + data
        crc = struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
        return c + crc

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


@functools.cache
def make_dummy_png_base64() -> str:
    """Return a valid 1x1 black RGB PNG as a base64 string."""
    return base64.b64encode(make_dummy_png_bytes()).decode("utf-8")
