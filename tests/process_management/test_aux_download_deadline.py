"""The child aborts a stalled aux download at its dispatch deadline and faults the job (no teardown).

When a job's LoRA downloads blow the deadline the parent hands it, ``download_aux_models`` cancels the
stalled downloads and raises ``AuxDownloadDeadlineExceeded`` instead of letting the parent's watchdog tear
the whole inference process down. The control-message wrapper turns that into a faulted result + a return
to ``WAITING_FOR_JOB``, keeping the process (and its resident model) alive.
"""

from __future__ import annotations

import multiprocessing
import threading
from multiprocessing import synchronize
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry

from horde_worker_regen.process_management.inference_process import (
    AuxDownloadDeadlineExceeded,
    HordeInferenceProcess,
)

from .conftest import make_job_pop_response


class _StallingLoraManager:
    """A lora manager whose downloads never complete, standing in for a wedged ad-hoc download."""

    def __init__(self, *, downloads_complete: bool = False) -> None:
        self.cancel_active_downloads = Mock()
        self.fetched: list[object] = []
        self._downloads_complete = downloads_complete

    def load_model_database(self) -> None:
        return None

    def reset_adhoc_cache(self) -> None:
        return None

    def is_model_available(self, name: object) -> bool:
        return False

    def fetch_adhoc_lora(self, name: object, timeout: object = None, is_version: bool = False) -> None:
        self.fetched.append(name)

    def wait_for_downloads(self, timeout: float | None = None) -> None:
        return None

    def are_downloads_complete(self) -> bool:
        return self._downloads_complete

    def save_reference_to_disk(self) -> None:
        return None


def _bare_inference_proc(aux_lock: synchronize.Lock, lora_manager: _StallingLoraManager) -> HordeInferenceProcess:
    """Build a HordeInferenceProcess wired only for the ``download_aux_models`` path."""
    proc = object.__new__(HordeInferenceProcess)
    proc._aux_model_lock = aux_lock
    proc._shared_model_manager = SimpleNamespace(manager=SimpleNamespace(lora=lora_manager))  # pyrefly: ignore
    proc.process_id = 1
    proc.send_aux_model_message = Mock()  # pyrefly: ignore
    proc.send_heartbeat_message = Mock()  # pyrefly: ignore
    proc._start_aux_download_heartbeat_thread = Mock(  # pyrefly: ignore
        side_effect=lambda: (threading.Event(), Mock()),
    )
    proc._send_download_metrics_if_any = Mock()  # pyrefly: ignore
    proc._enforce_lora_disk_floor = Mock()  # pyrefly: ignore
    return proc


def _lora_job() -> ImageGenerateJobPopResponse:
    return make_job_pop_response(
        model="WAI-NSFW-illustrious-SDXL",
        loras=[LorasPayloadEntry(name="1683285", model=1.0, clip=1.0, is_version=True)],
    )


def test_aux_download_aborts_and_cancels_past_deadline() -> None:
    """A blown deadline cancels the active downloads and raises, leaving the aux lock free."""
    aux_lock = multiprocessing.Lock()
    lora_manager = _StallingLoraManager(downloads_complete=False)
    proc = _bare_inference_proc(aux_lock, lora_manager)

    with pytest.raises(AuxDownloadDeadlineExceeded):
        proc.download_aux_models(_lora_job(), aux_download_deadline_seconds=0.0)

    lora_manager.cancel_active_downloads.assert_called_once()
    # The shared aux lock must be released even though we aborted via an exception.
    assert aux_lock.acquire(block=False) is True
    aux_lock.release()


def test_aux_download_without_deadline_does_not_abort() -> None:
    """With no deadline the old behaviour holds: a never-completing wait does not raise the deadline error."""
    aux_lock = multiprocessing.Lock()
    lora_manager = _StallingLoraManager(downloads_complete=False)
    proc = _bare_inference_proc(aux_lock, lora_manager)

    # Returns normally (the per-LoRA wait is mocked to return); no deadline means no abort.
    proc.download_aux_models(_lora_job(), aux_download_deadline_seconds=None)

    lora_manager.cancel_active_downloads.assert_not_called()


def test_aux_download_completes_within_deadline() -> None:
    """A deadline that is not exceeded (downloads complete) returns normally without cancelling."""
    aux_lock = multiprocessing.Lock()
    lora_manager = _StallingLoraManager(downloads_complete=True)
    proc = _bare_inference_proc(aux_lock, lora_manager)

    proc.download_aux_models(_lora_job(), aux_download_deadline_seconds=600.0)

    lora_manager.cancel_active_downloads.assert_not_called()


def test_fault_job_for_aux_deadline_sends_faulted_result_and_idles() -> None:
    """The graceful-abort handler reports a faulted result (with the marker) and returns to WAITING_FOR_JOB."""
    from horde_worker_regen.process_management.messages import AUX_DOWNLOAD_FAILED_INFO

    proc = object.__new__(HordeInferenceProcess)
    proc.process_id = 1
    proc._last_inference_error = None
    proc._active_model_name = "WAI-NSFW-illustrious-SDXL"
    sent_results: list[dict] = []
    proc.send_inference_result_message = Mock(  # pyrefly: ignore
        side_effect=lambda **kw: sent_results.append({"info_marker": proc._last_inference_error, **kw}),
    )
    proc.send_process_state_change_message = Mock()  # pyrefly: ignore

    job = _lora_job()
    proc._fault_job_for_aux_deadline(job)

    assert len(sent_results) == 1
    # The marker must be in place at send time (it is cleared again afterwards).
    assert sent_results[0]["info_marker"] == AUX_DOWNLOAD_FAILED_INFO
    assert sent_results[0]["results"] is None
    assert proc._last_inference_error is None
    proc.send_process_state_change_message.assert_called_once()
