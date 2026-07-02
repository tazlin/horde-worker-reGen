"""Tests for image utility functions."""

from io import BytesIO

import PIL.Image

from horde_worker_regen.utils.image_utils import image_bytes_to_stream_buffer


def test_image_bytes_to_stream_buffer_valid_image() -> None:
    """Test converting valid image bytes to a stream buffer."""
    # Create a simple test image
    test_image = PIL.Image.new("RGB", (100, 100), color="red")
    image_buffer = BytesIO()
    test_image.save(image_buffer, format="PNG")

    # Test the function
    result = image_bytes_to_stream_buffer(image_buffer.getvalue())

    assert result is not None
    assert isinstance(result, BytesIO)

    # Verify the result is a valid WebP image
    result.seek(0)
    result_image = PIL.Image.open(result)
    assert result_image.format == "WEBP"
    assert result_image.size == (100, 100)


def test_image_bytes_to_stream_buffer_empty_bytes() -> None:
    """Test that empty bytes returns None."""
    result = image_bytes_to_stream_buffer(b"")
    assert result is None


def test_image_bytes_to_stream_buffer_not_an_image() -> None:
    """Test that bytes that are not an image returns None."""
    result = image_bytes_to_stream_buffer(b"This is just text")
    assert result is None


def test_image_bytes_to_stream_buffer_different_formats() -> None:
    """Test converting images in different formats."""
    for format_name in ["PNG", "JPEG", "BMP"]:
        test_image = PIL.Image.new("RGB", (50, 50), color="blue")
        image_buffer = BytesIO()
        test_image.save(image_buffer, format=format_name)

        result = image_bytes_to_stream_buffer(image_buffer.getvalue())

        assert result is not None, f"Failed to convert {format_name} image"
        result.seek(0)
        result_image = PIL.Image.open(result)
        assert result_image.format == "WEBP"
