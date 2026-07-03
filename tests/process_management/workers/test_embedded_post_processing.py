"""The worker-driven embedded post-processing chain for image jobs.

The worker (not hordelib) applies a job's requested post-processing to each generated image
via hordelib's typed `post_process`. These tests pin the execution policy: facefixers run
last, the requested order is otherwise preserved, faults accumulate across conversion,
inference, and post-processing, and an image whose chain produces no output is dropped.
"""

from __future__ import annotations

import io
from typing import Any

import PIL.Image
from horde_sdk.ai_horde_api.apimodels import GenMetadataEntry
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from horde_sdk.generation_parameters.alchemy import (
    FacefixAlchemyParameters,
    SingleAlchemyParameters,
    UpscaleAlchemyParameters,
)
from horde_sdk.generation_parameters.alchemy.consts import KNOWN_ALCHEMY_FORMS, KNOWN_MISC_POST_PROCESSORS

from horde_worker_regen.process_management.workers.inference_process import HordeInferenceProcess


def _upscale_op(upscaler: str = "RealESRGAN_x2plus") -> UpscaleAlchemyParameters:
    return UpscaleAlchemyParameters(
        result_id="test",
        form=KNOWN_ALCHEMY_FORMS.post_process,
        source_image=None,
        upscaler=upscaler,
    )


def _facefix_op(facefixer: str = "CodeFormers") -> FacefixAlchemyParameters:
    return FacefixAlchemyParameters(
        result_id="test",
        form=KNOWN_ALCHEMY_FORMS.post_process,
        source_image=None,
        facefixer=facefixer,
    )


def _strip_background_op() -> SingleAlchemyParameters:
    return SingleAlchemyParameters(
        result_id="test",
        form=KNOWN_MISC_POST_PROCESSORS.strip_background,
        source_image=None,
    )


def _fault(ref: str | None = None) -> GenMetadataEntry:
    return GenMetadataEntry(type=METADATA_TYPE.source_image, value=METADATA_VALUE.parse_failed, ref=ref)


class _FakeHordeLib:
    """Record post_process calls and hand back scripted results."""

    def __init__(self, fail_on_call_index: int | None = None) -> None:
        self.calls: list[Any] = []
        self._fail_on_call_index = fail_on_call_index

    def post_process(self, payload: Any) -> Any:
        from hordelib.api import ResultingImageReturn

        call_index = len(self.calls)
        self.calls.append(payload)

        if self._fail_on_call_index is not None and call_index == self._fail_on_call_index:
            return ResultingImageReturn(image=None, rawpng=None, faults=[_fault(ref="pp")])

        return ResultingImageReturn(
            image=PIL.Image.new("RGB", (16, 16)),
            rawpng=io.BytesIO(b"post-processed png"),
            faults=[],
        )


class _FakeInferenceProcess:
    """The minimal surface `_run_embedded_post_processing` touches on its process."""

    _facefixers_last = staticmethod(HordeInferenceProcess._facefixers_last)

    def __init__(self, fail_on_call_index: int | None = None) -> None:
        self._horde = _FakeHordeLib(fail_on_call_index)
        self._in_post_processing = False
        self._post_processing_memory_report_sent = False
        self._start_inference_time = 0.0
        self.state_changes: list[Any] = []
        self.slot_released = False
        self.memory_reports = 0

    def send_process_state_change_message(self, **kwargs: Any) -> None:
        self.state_changes.append(kwargs)

    def _release_inference_slot(self) -> None:
        self.slot_released = True

    def _send_inference_memory_report(self) -> None:
        self.memory_reports += 1


def _make_result(faults: list[GenMetadataEntry] | None = None) -> Any:
    from hordelib.api import ResultingImageReturn

    return ResultingImageReturn(
        image=PIL.Image.new("RGB", (8, 8)),
        rawpng=io.BytesIO(b"original png"),
        faults=faults or [],
    )


def _run(
    fake: _FakeInferenceProcess,
    results: list[Any],
    operations: list[SingleAlchemyParameters],
    conversion_faults: list[GenMetadataEntry] | None = None,
) -> list[Any]:
    return HordeInferenceProcess._run_embedded_post_processing(
        fake,  # type: ignore[arg-type]
        results,
        operations,
        conversion_faults or [],
    )


def test_facefixers_sort_last_preserving_order() -> None:
    """Facefixers move to the end; the requested order is otherwise stable."""
    operations = [
        _facefix_op("GFPGAN"),
        _strip_background_op(),
        _upscale_op("RealESRGAN_x2plus"),
        _facefix_op("CodeFormers"),
        _upscale_op("RealESRGAN_x4plus"),
    ]

    ordered = HordeInferenceProcess._facefixers_last(operations)

    assert [type(operation).__name__ for operation in ordered] == [
        "SingleAlchemyParameters",
        "UpscaleAlchemyParameters",
        "UpscaleAlchemyParameters",
        "FacefixAlchemyParameters",
        "FacefixAlchemyParameters",
    ]
    assert ordered[1].upscaler == "RealESRGAN_x2plus"
    assert ordered[2].upscaler == "RealESRGAN_x4plus"
    assert ordered[3].facefixer == "GFPGAN"
    assert ordered[4].facefixer == "CodeFormers"


def test_chain_runs_per_image_and_signals_post_processing_state() -> None:
    """Each image runs the full chain and the process signals the post-processing transition."""
    from hordelib.api import FacefixPayload, UpscalePayload

    from horde_worker_regen.process_management.ipc.messages import HordeProcessState

    fake = _FakeInferenceProcess()
    results = [_make_result(), _make_result()]

    post_processed = _run(fake, results, [_upscale_op(), _facefix_op()])

    assert len(post_processed) == 2
    # Two operations per image, upscale before facefix
    assert len(fake._horde.calls) == 4
    assert isinstance(fake._horde.calls[0], UpscalePayload)
    assert isinstance(fake._horde.calls[1], FacefixPayload)

    assert fake.slot_released is True
    assert fake._in_post_processing is True
    assert fake.memory_reports == 1
    assert fake.state_changes[0]["process_state"] == HordeProcessState.INFERENCE_POST_PROCESSING


def test_faults_accumulate_across_stages() -> None:
    """Conversion and inference faults carry through onto the post-processed result."""
    fake = _FakeInferenceProcess()
    conversion_fault = _fault(ref="conversion")
    inference_fault = _fault(ref="inference")

    post_processed = _run(fake, [_make_result(faults=[inference_fault])], [_upscale_op()], [conversion_fault])

    assert len(post_processed) == 1
    assert conversion_fault in post_processed[0].faults
    assert inference_fault in post_processed[0].faults


def test_image_dropped_when_chain_produces_no_output() -> None:
    """A failed operation with no output image drops the image from the results."""
    fake = _FakeInferenceProcess(fail_on_call_index=0)

    post_processed = _run(fake, [_make_result()], [_upscale_op(), _facefix_op()])

    assert post_processed == []
    # The chain aborted after the failed first operation
    assert len(fake._horde.calls) == 1


def test_strip_background_keeps_pre_strip_rawpng() -> None:
    """strip_background updates the image but keeps the pre-strip raw PNG (legacy parity)."""
    from hordelib.api import StripBackgroundPayload

    fake = _FakeInferenceProcess()
    original = _make_result()
    original_rawpng = original.rawpng

    post_processed = _run(fake, [original], [_strip_background_op()])

    assert len(post_processed) == 1
    assert isinstance(fake._horde.calls[0], StripBackgroundPayload)
    assert post_processed[0].rawpng is original_rawpng
