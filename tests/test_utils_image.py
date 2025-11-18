"""Tests for image utility functions."""

import base64
from io import BytesIO

import PIL.Image

from horde_worker_regen.utils.image_utils import base64_image_to_stream_buffer


def test_base64_image_to_stream_buffer_valid_image() -> None:
    """Test converting a valid base64 image to a stream buffer."""
    # Create a simple test image
    test_image = PIL.Image.new("RGB", (100, 100), color="red")
    image_bytes = BytesIO()
    test_image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    # Convert to base64
    image_base64 = base64.b64encode(image_bytes.read()).decode("utf-8")

    # Test the function
    result = base64_image_to_stream_buffer(image_base64)

    assert result is not None
    assert isinstance(result, BytesIO)

    # Verify the result is a valid WebP image
    result.seek(0)
    result_image = PIL.Image.open(result)
    assert result_image.format == "WEBP"
    assert result_image.size == (100, 100)


def test_base64_image_to_stream_buffer_invalid_base64() -> None:
    """Test that invalid base64 returns None."""
    result = base64_image_to_stream_buffer("not-valid-base64!")
    assert result is None


def test_base64_image_to_stream_buffer_empty_string() -> None:
    """Test that empty string returns None."""
    result = base64_image_to_stream_buffer("")
    assert result is None


def test_base64_image_to_stream_buffer_valid_base64_but_not_image() -> None:
    """Test that valid base64 but not an image returns None."""
    # Create valid base64 that's not an image
    not_an_image = base64.b64encode(b"This is just text").decode("utf-8")
    result = base64_image_to_stream_buffer(not_an_image)
    assert result is None


def test_base64_image_to_stream_buffer_different_formats() -> None:
    """Test converting images in different formats."""
    for format_name in ["PNG", "JPEG", "BMP"]:
        test_image = PIL.Image.new("RGB", (50, 50), color="blue")
        image_bytes = BytesIO()
        test_image.save(image_bytes, format=format_name)
        image_bytes.seek(0)

        image_base64 = base64.b64encode(image_bytes.read()).decode("utf-8")
        result = base64_image_to_stream_buffer(image_base64)

        assert result is not None, f"Failed to convert {format_name} image"
        result.seek(0)
        result_image = PIL.Image.open(result)
        assert result_image.format == "WEBP"
