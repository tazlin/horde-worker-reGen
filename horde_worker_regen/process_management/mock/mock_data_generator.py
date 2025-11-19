"""Generate fake data for mock processes.

This module creates realistic placeholder data (images, metadata, etc.) that can be
used in place of real GPU-generated content for testing purposes.
"""

from __future__ import annotations

import base64
import hashlib
import io
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def generate_fake_image(
    width: int,
    height: int,
    *,
    job_id: str | None = None,
    model_name: str | None = None,
    seed: int | None = None,
    steps: int | None = None,
) -> str:
    """Generate a fake placeholder image for testing.

    Creates a simple colored image with text overlay containing job metadata.
    The image is encoded as base64 PNG data.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        job_id: Optional job ID to display on image.
        model_name: Optional model name to display.
        seed: Optional seed to display.
        steps: Optional step count to display.

    Returns:
        Base64-encoded PNG image data.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # Fallback: generate a minimal valid PNG without PIL
        return _generate_minimal_png(width, height)

    # Generate a color based on job_id (deterministic but varied)
    if job_id:
        hash_val = int(hashlib.md5(job_id.encode()).hexdigest()[:6], 16)
        r = (hash_val >> 16) & 0xFF
        g = (hash_val >> 8) & 0xFF
        b = hash_val & 0xFF
        # Lighten colors a bit for better text contrast
        color = (
            min(255, r + 80),
            min(255, g + 80),
            min(255, b + 80),
        )
    else:
        # Random pastel color
        color = (
            random.randint(150, 220),
            random.randint(150, 220),
            random.randint(150, 220),
        )

    # Create image
    img = Image.new("RGB", (width, height), color=color)
    draw = ImageDraw.Draw(img)

    # Try to use a font, fall back to default if not available
    try:
        # Try to find a monospace font
        font_size = max(12, min(width, height) // 30)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Build text overlay
    text_lines = ["[MOCK IMAGE]"]
    if model_name:
        text_lines.append(f"Model: {model_name}")
    if job_id:
        # Truncate long job IDs
        display_id = job_id[:16] + "..." if len(job_id) > 16 else job_id
        text_lines.append(f"Job: {display_id}")
    if seed is not None:
        text_lines.append(f"Seed: {seed}")
    if steps is not None:
        text_lines.append(f"Steps: {steps}")

    text_lines.append(f"Size: {width}x{height}")

    # Draw text with background for readability
    y_offset = 10
    for line in text_lines:
        # Get text bounding box
        bbox = draw.textbbox((10, y_offset), line, font=font)
        # Draw semi-transparent background
        draw.rectangle(
            [(bbox[0] - 2, bbox[1] - 2), (bbox[2] + 2, bbox[3] + 2)],
            fill=(0, 0, 0, 180),
        )
        # Draw text
        draw.text((10, y_offset), line, fill=(255, 255, 255), font=font)
        y_offset += bbox[3] - bbox[1] + 5

    # Add a diagonal watermark
    watermark = "MOCK DATA - NOT REAL IMAGE"
    try:
        watermark_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        watermark_font = ImageFont.load_default()

    # Calculate position for centered watermark
    bbox = draw.textbbox((0, 0), watermark, font=watermark_font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Draw rotated watermark
    watermark_img = Image.new("RGBA", (text_width + 20, text_height + 20), (0, 0, 0, 0))
    watermark_draw = ImageDraw.Draw(watermark_img)
    watermark_draw.text((10, 10), watermark, fill=(255, 255, 255, 100), font=watermark_font)
    watermark_img = watermark_img.rotate(45, expand=True)

    # Paste watermark in center
    img.paste(
        watermark_img,
        (width // 2 - watermark_img.width // 2, height // 2 - watermark_img.height // 2),
        watermark_img,
    )

    # Encode to PNG
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    # Return base64 encoded
    return base64.b64encode(buffer.read()).decode("utf-8")


def _generate_minimal_png(width: int, height: int) -> str:
    """Generate a minimal valid PNG without PIL (fallback).

    Creates a simple 1x1 gray PNG and returns it base64-encoded.
    This is used when PIL is not available.

    Args:
        width: Requested width (ignored in minimal version).
        height: Requested height (ignored in minimal version).

    Returns:
        Base64-encoded minimal PNG.
    """
    # Minimal 1x1 gray PNG (89 bytes)
    minimal_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xa8\xa9\xa9"
        b"\x01\x00\x02\x8c\x00\xed\x8e\xb3\xd5\xca\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return base64.b64encode(minimal_png).decode("utf-8")


def calculate_mock_kudos(
    width: int,
    height: int,
    steps: int,
    *,
    has_controlnet: bool = False,
    has_loras: bool = False,
    has_source_image: bool = False,
    num_images: int = 1,
) -> float:
    """Calculate approximate kudos for a mock job.

    This mimics the real kudos calculation to provide realistic values.

    Args:
        width: Image width.
        height: Image height.
        steps: Number of inference steps.
        has_controlnet: Whether ControlNet is used.
        has_loras: Whether LoRAs are used.
        has_source_image: Whether this is img2img.
        num_images: Number of images in batch.

    Returns:
        Estimated kudos value.
    """
    # Simplified kudos calculation (megapixelsteps based)
    megapixels = (width * height) / 1_000_000
    megapixelsteps = megapixels * steps

    # Base kudos
    kudos = megapixelsteps * 0.5

    # Modifiers
    if has_controlnet:
        kudos *= 1.3
    if has_loras:
        kudos *= 1.1
    if has_source_image:
        kudos *= 0.9  # img2img is slightly cheaper

    # Multiply by batch size
    kudos *= num_images

    return round(kudos, 2)


def calculate_mock_inference_time(
    width: int,
    height: int,
    steps: int,
    *,
    speed_multiplier: float = 1.0,
    slowdown_multiplier: float = 1.0,
) -> float:
    """Calculate realistic mock inference time.

    Args:
        width: Image width.
        height: Image height.
        steps: Number of inference steps.
        speed_multiplier: Speed multiplier (higher = faster).
        slowdown_multiplier: Slowdown multiplier for slow jobs (higher = slower).

    Returns:
        Estimated inference time in seconds.
    """
    # Base time per step (milliseconds)
    base_time_per_step = 100  # 0.1s per step baseline

    # Adjust for resolution
    megapixels = (width * height) / 1_000_000
    resolution_multiplier = 0.5 + (megapixels * 0.5)

    # Total time before modifiers
    total_ms = steps * base_time_per_step * resolution_multiplier

    # Apply modifiers
    total_ms *= slowdown_multiplier
    total_ms /= speed_multiplier

    # Convert to seconds
    return total_ms / 1000.0


def generate_fake_nsfw_score() -> float:
    """Generate a fake NSFW score for testing.

    Returns a value between 0.0 and 1.0, with bias toward low values
    (most images are SFW in testing).

    Returns:
        NSFW score between 0.0 and 1.0.
    """
    # Bias heavily toward safe images (90% will be < 0.3)
    if random.random() < 0.9:
        return random.uniform(0.0, 0.3)
    else:
        return random.uniform(0.3, 1.0)


def generate_fake_csam_score() -> float:
    """Generate a fake CSAM score for testing.

    Always returns very low values since CSAM should never appear in testing.

    Returns:
        CSAM score between 0.0 and 0.05.
    """
    # Always very low scores for testing
    return random.uniform(0.0, 0.05)
