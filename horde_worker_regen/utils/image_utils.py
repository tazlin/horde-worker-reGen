"""Image processing utility functions."""

from __future__ import annotations

from io import BytesIO

import PIL.Image
from loguru import logger


def image_bytes_to_stream_buffer(image_bytes: bytes) -> BytesIO | None:
    """Convert encoded image bytes to a WebP BytesIO stream buffer.

    Args:
        image_bytes: The encoded image bytes to convert.

    Returns:
        A BytesIO stream buffer containing the image, or None if the conversion failed.
    """
    try:
        image_as_pil = PIL.Image.open(BytesIO(image_bytes))
        image_buffer = BytesIO()
        image_as_pil.save(
            image_buffer,
            format="WebP",
            quality=95,  # FIXME # TODO
            method=6,
        )

        return image_buffer
    except Exception as e:
        logger.error(f"Failed to convert image bytes to stream buffer: {e}")
        return None
