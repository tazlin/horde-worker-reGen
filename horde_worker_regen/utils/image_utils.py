"""Image processing utility functions."""

from __future__ import annotations

import base64
from io import BytesIO

import PIL.Image
from loguru import logger


def base64_image_to_stream_buffer(image_base64: str) -> BytesIO | None:
    """Convert a base64 image to a BytesIO stream buffer.

    Args:
        image_base64: The base64 image to convert.

    Returns:
        A BytesIO stream buffer containing the image, or None if the conversion failed.
    """
    try:
        image_as_pil = PIL.Image.open(BytesIO(base64.b64decode(image_base64)))
        image_buffer = BytesIO()
        image_as_pil.save(
            image_buffer,
            format="WebP",
            quality=95,  # FIXME # TODO
            method=6,
        )

        return image_buffer
    except Exception as e:
        logger.error(f"Failed to convert base64 image to stream buffer: {e}")
        return None
