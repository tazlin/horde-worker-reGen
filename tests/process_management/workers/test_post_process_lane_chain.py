"""The post-processing lane runs one hordelib chain per image and encodes its result for upload."""

from __future__ import annotations

import io
from unittest.mock import Mock

import PIL.Image
import pytest

from horde_worker_regen.process_management.ipc.messages import HordePostProcessControlMessage
from horde_worker_regen.process_management.workers.post_process_process import HordePostProcessProcess
from horde_worker_regen.utils.image_utils import UPLOAD_IMAGE_ENCODING


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    PIL.Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def _make_lane(chain_image: PIL.Image.Image | None) -> tuple[HordePostProcessProcess, Mock]:
    lane = HordePostProcessProcess.__new__(HordePostProcessProcess)
    horde = Mock()
    horde.post_process_chain.return_value = Mock(image=chain_image, faults=[])
    lane._horde = horde
    lane.send_heartbeat_message = Mock()  # type: ignore[method-assign]
    lane._require_post_processor_on_disk = Mock()  # type: ignore[method-assign]
    return lane, horde


def _message(facefixer_strength: float | None) -> HordePostProcessControlMessage:
    return HordePostProcessControlMessage(
        job_id="00000000-0000-0000-0000-000000000001",
        images_bytes=[_png_bytes()],
        post_processing=["GFPGAN", "RealESRGAN_x4plus"],
        facefixer_strength=facefixer_strength,
    )


def test_lane_calls_the_chain_once_with_operations_and_strength() -> None:
    """One chain call per image carries the requested operations and strength; the result is upload-encoded."""
    lane, horde = _make_lane(PIL.Image.new("RGB", (16, 16), "blue"))

    results = lane._post_process_all_images(_message(0.4))

    horde.post_process_chain.assert_called_once()
    call = horde.post_process_chain.call_args
    assert call.args[1] == ["GFPGAN", "RealESRGAN_x4plus"]
    assert call.kwargs["facefixer_strength"] == 0.4
    assert len(results) == 1
    assert results[0].image_encoding == UPLOAD_IMAGE_ENCODING
    assert PIL.Image.open(io.BytesIO(results[0].image_bytes)).format == "WEBP"
    lane.send_heartbeat_message.assert_called()


def test_chain_without_an_image_raises() -> None:
    """A chain that yields no image faults the job as before."""
    lane, _ = _make_lane(None)

    with pytest.raises(RuntimeError, match="no output image"):
        lane._post_process_all_images(_message(None))
