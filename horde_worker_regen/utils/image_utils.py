"""Image processing utility functions."""

from __future__ import annotations

from enum import StrEnum, auto
from io import BytesIO

import PIL.Image
from loguru import logger


class ImageEncoding(StrEnum):
    """How a stage's image bytes are encoded."""

    PNG = auto()
    WEBP = auto()


UPLOAD_IMAGE_ENCODING = ImageEncoding.WEBP
"""The encoding the horde receives; :func:`encode_image_for_upload` produces it."""

UPLOAD_WEBP_METHOD = 2
"""libwebp's effort setting (0 fastest, 6 slowest) for the upload encode.

The encode runs in the process that produced the image and that process holds its card reservation until the
result is sent, so encode seconds are seconds of held card. At quality 95 the higher efforts buy almost nothing:
measured against 6, this takes about a quarter of the time for a file about 7% larger and a quarter of a decibel
less PSNR, on an encode that already sits 43 to 47 dB from its source."""


def encode_image_for_upload(image: PIL.Image.Image) -> BytesIO:
    """Encode a PIL image in the upload encoding (:data:`UPLOAD_IMAGE_ENCODING`).

    Args:
        image: The decoded image to encode.

    Returns:
        A BytesIO stream buffer containing the encoded image.
    """
    image_buffer = BytesIO()
    image.save(
        image_buffer,
        format="WebP",
        quality=95,  # FIXME # TODO
        method=UPLOAD_WEBP_METHOD,
    )
    return image_buffer


def image_bytes_to_stream_buffer(image_bytes: bytes) -> BytesIO | None:
    """Decode encoded image bytes and re-encode them in the upload encoding.

    Args:
        image_bytes: The encoded image bytes to convert.

    Returns:
        A BytesIO stream buffer containing the image, or None if the conversion failed.
    """
    try:
        return encode_image_for_upload(PIL.Image.open(BytesIO(image_bytes)))
    except Exception as e:
        logger.error(f"Failed to convert image bytes to stream buffer: {e}")
        return None
