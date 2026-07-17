"""Minimal placeholder image generation for dry-run and fake worker processes.

Kept free of any heavy imports (no PIL, no torch) so it can be used by processes
that must start without the ML dependency stack.
"""

from __future__ import annotations

import functools
import struct
import zlib

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Assemble one length-prefixed, CRC-checked PNG chunk."""
    body = struct.pack(">I", len(data)) + chunk_type + data
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return body + crc


@functools.cache
def make_dummy_png_bytes() -> bytes:
    """Return the bytes of a valid 1x1 black RGB PNG."""
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return _PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


@functools.lru_cache(maxsize=128)
def make_dummy_source_png_bytes(width: int, height: int, seed: int = 0) -> bytes:
    """Return a deterministic RGB PNG at the requested size, its content derived from ``seed``.

    Stdlib-only (no PIL, no torch) so any process can build it, and byte-identical for a given
    ``(width, height, seed)`` so two builds of the same job present the same source image. The content
    is a seed-coloured horizontal-band fill: the pixels are irrelevant to the VAE-encode cost a source
    image incurs, whereas the resolution is exactly what makes that cost realistic, so the requested
    size is honoured here rather than collapsing to a 1x1 placeholder when no imaging library is present.
    """
    safe_width = max(1, width)
    safe_height = max(1, height)
    primary = (seed & 0xFF, (seed >> 8) & 0xFF, (seed >> 16) & 0xFF)
    secondary = (primary[0] ^ 0xFF, primary[1] ^ 0xFF, primary[2] ^ 0xFF)
    band_height = 16 + (seed % 48)  # seed-derived band thickness, always >= 1

    # Each PNG scanline is a leading filter byte (0 = no filter) then packed RGB triples; building a
    # whole row with one bytes multiplication keeps generation at C speed even at 1024x1024.
    primary_row = b"\x00" + bytes(primary) * safe_width
    secondary_row = b"\x00" + bytes(secondary) * safe_width
    raw = bytearray()
    for row_index in range(safe_height):
        raw += primary_row if (row_index // band_height) % 2 == 0 else secondary_row

    ihdr = struct.pack(">IIBBBBB", safe_width, safe_height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw))
    return _PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")
