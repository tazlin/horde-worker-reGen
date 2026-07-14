"""Tests for the inference child's pop-time auxiliary-model resolve at START_INFERENCE.

The dedicated download process places a job's LoRAs and textual inversions on disk before the job becomes
dispatchable. By inference time the child only confirms their presence read-only: no network fetch may
happen on the inference path, or the ``inference_step_timeout`` watchdog would mistake a download for a
hung sampler. These tests pin that contract through the messages the parent would receive. A resolvable
job proceeds into sampling with no fault; an unresolvable one faults the job retryably (tagged
``AUX_RESOLVE_FAILED_INFO``), returns to ``WAITING_FOR_JOB`` with its model kept resident, and never
preloads, samples, or ends the process.
"""

from __future__ import annotations

import io
import queue
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, LorasPayloadEntry, TIPayloadEntry
from horde_sdk.ai_horde_api.consts import GENERATION_STATE

from horde_worker_regen.process_management.ipc.messages import (
    AUX_RESOLVE_FAILED_INFO,
    AuxModelKind,
    AuxModelRef,
    HordeControlFlag,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
)
from horde_worker_regen.process_management.workers.inference_process import HordeInferenceProcess
from tests.process_management.conftest import make_job_pop_response

_RESIDENT_MODEL = "CyberRealistic Pony"


class _FakeLoraManager:
    """A LoRA manager that answers presence probes from memory and records any download attempt.

    The presence-only resolve path may consult ``refresh_reference_if_stale`` and ``is_lora_available``
    but must never reach the download surface; each download method records its call so a regression that
    fetches on the inference path is caught rather than silently tolerated.
    """

    def __init__(self, *, available: bool) -> None:
        self._available = available
        self.refresh_calls = 0
        self.presence_probes: list[tuple[str, bool]] = []
        self.fetch_calls: list[str] = []

    def refresh_reference_if_stale(self) -> bool:
        self.refresh_calls += 1
        return False

    def is_lora_available(self, lora_name: str | int, timeout: float = 45, is_version: bool = False) -> bool:
        self.presence_probes.append((str(lora_name), bool(is_version)))
        return self._available

    def fetch_adhoc_lora(self, *args: object, **kwargs: object) -> str | None:
        self.fetch_calls.append("fetch_adhoc_lora")
        return None

    def download_default_models(self, *args: object, **kwargs: object) -> None:
        self.fetch_calls.append("download_default_models")

    def download_model(self, *args: object, **kwargs: object) -> bool:
        self.fetch_calls.append("download_model")
        return False


class _FakeTiManager:
    """A textual-inversion manager mirroring the LoRA fake: presence lookups only, download recorded."""

    def __init__(self, *, key: str | None) -> None:
        self._key = key
        self.refresh_calls = 0
        self.presence_probes: list[str] = []
        self.fetch_calls: list[str] = []

    def refresh_reference_if_stale(self) -> bool:
        self.refresh_calls += 1
        return False

    def fuzzy_find_ti_key(self, ti_name: str | int) -> str | None:
        self.presence_probes.append(str(ti_name))
        return self._key

    def fetch_adhoc_ti(self, *args: object, **kwargs: object) -> str | None:
        self.fetch_calls.append("fetch_adhoc_ti")
        return None


class _FakeSharedModelManager:
    """Stands in for the shared model manager, exposing only the ``.manager.lora`` / ``.manager.ti`` path."""

    def __init__(self, lora: _FakeLoraManager | None, ti: _FakeTiManager | None) -> None:
        self.manager = SimpleNamespace(lora=lora, ti=ti)


def _fake_result() -> SimpleNamespace:
    """A minimal stand-in for a sampled image result the result-message builder can serialize."""
    return SimpleNamespace(rawpng=io.BytesIO(b"\x89PNG\r\n"), faults=[])


def _make_proc(
    *,
    shared_model_manager: _FakeSharedModelManager | None,
    active_model: str | None = _RESIDENT_MODEL,
    dry_run: bool = False,
) -> HordeInferenceProcess:
    """Build a HordeInferenceProcess wired only for the START_INFERENCE resolve path.

    Real construction spins up HordeLib and the shared model managers; only a handful of methods are
    reached here, so the heavy downstream (preload, sampling, model-state and memory side channels) is
    replaced while the message-emitting surface (result and state-change messages onto the queue) stays
    real, so the parent-facing behavior is observed rather than mocked away.
    """
    proc = object.__new__(HordeInferenceProcess)
    proc.process_id = 2
    proc.process_launch_identifier = 0
    proc.process_message_queue = Mock(spec=queue.Queue)  # pyrefly: ignore
    proc._active_model_name = active_model
    proc._shared_model_manager = shared_model_manager  # pyrefly: ignore
    proc._dry_run_skip_inference = dry_run
    proc._last_inference_error = None
    proc._last_job_inference_rate = None
    proc._current_job_kept_model_resident = False
    proc._end_process = False
    proc.preload_model = Mock()  # pyrefly: ignore
    proc.start_inference = Mock(return_value=[_fake_result()])  # pyrefly: ignore
    proc.on_horde_model_state_change = Mock()  # pyrefly: ignore
    proc.send_memory_report_message = Mock(return_value=True)  # pyrefly: ignore
    return proc


def _start_inference_message(
    job: ImageGenerateJobPopResponse, model: str = _RESIDENT_MODEL
) -> HordeInferenceControlMessage:
    return HordeInferenceControlMessage(
        control_flag=HordeControlFlag.START_INFERENCE,
        horde_model_name=model,
        sdk_api_job_info=job,
    )


def _emitted_messages(proc: HordeInferenceProcess) -> list[Any]:
    """Return, in order, every message the child placed on the queue to the parent."""
    return [call.args[0] for call in proc.process_message_queue.put.call_args_list]  # pyrefly: ignore


def _result_messages(proc: HordeInferenceProcess) -> list[HordeInferenceResultMessage]:
    return [m for m in _emitted_messages(proc) if isinstance(m, HordeInferenceResultMessage)]


def _assert_no_downloads(
    lora: _FakeLoraManager | None = None,
    ti: _FakeTiManager | None = None,
) -> None:
    """No download/fetch method on either manager may be reached from the inference resolve path."""
    if lora is not None:
        assert lora.fetch_calls == []
    if ti is not None:
        assert ti.fetch_calls == []


def test_job_without_aux_models_proceeds_to_inference() -> None:
    """A job referencing neither LoRAs nor TIs resolves and reaches sampling with no fault emitted."""
    lora = _FakeLoraManager(available=True)
    ti = _FakeTiManager(key="anything")
    proc = _make_proc(shared_model_manager=_FakeSharedModelManager(lora, ti))
    job = make_job_pop_response(model=_RESIDENT_MODEL)

    proc._receive_and_handle_control_message(_start_inference_message(job))

    proc.start_inference.assert_called_once()  # pyrefly: ignore
    proc.preload_model.assert_not_called()  # pyrefly: ignore
    results = _result_messages(proc)
    assert len(results) == 1
    assert results[0].state is GENERATION_STATE.ok
    assert results[0].info != AUX_RESOLVE_FAILED_INFO
    # An aux-free job never consults the managers at all, let alone downloads.
    assert lora.refresh_calls == 0 and ti.refresh_calls == 0
    _assert_no_downloads(lora, ti)


def test_present_aux_models_resolve_after_stale_refresh_without_downloading() -> None:
    """Every referenced LoRA/TI present on disk resolves True after a stale-reference refresh, no fetch."""
    lora = _FakeLoraManager(available=True)
    ti = _FakeTiManager(key="badhands")
    proc = _make_proc(shared_model_manager=_FakeSharedModelManager(lora, ti))
    job = make_job_pop_response(
        model=_RESIDENT_MODEL,
        loras=[LorasPayloadEntry(name="2498503", model=0.9, clip=1.0, is_version=True)],
        tis=[TIPayloadEntry(name="badhands")],
    )

    proc._receive_and_handle_control_message(_start_inference_message(job))

    # The read-only staleness reload runs so files another process wrote are picked up before probing.
    assert lora.refresh_calls == 1 and ti.refresh_calls == 1
    assert lora.presence_probes == [("2498503", True)]
    assert ti.presence_probes == ["badhands"]
    proc.start_inference.assert_called_once()  # pyrefly: ignore
    results = _result_messages(proc)
    assert len(results) == 1
    assert results[0].state is GENERATION_STATE.ok
    assert results[0].info != AUX_RESOLVE_FAILED_INFO
    _assert_no_downloads(lora, ti)


def test_missing_lora_faults_job_retryably_and_stays_idle_resident() -> None:
    """A missing file faults the job with the aux-resolve tag, returns to idle, and never samples or ends.

    The unresolved reference is a raced eviction between prefetch and dispatch, so the child reports a
    retryable fault the parent classifies via ``AUX_RESOLVE_FAILED_INFO``, transitions back to
    ``WAITING_FOR_JOB`` with its model resident, and does not preload, sample, or end the process.
    """
    lora = _FakeLoraManager(available=False)
    ti = _FakeTiManager(key="unused")
    proc = _make_proc(shared_model_manager=_FakeSharedModelManager(lora, ti))
    job = make_job_pop_response(
        model=_RESIDENT_MODEL,
        loras=[LorasPayloadEntry(name="2498503", model=0.9, clip=1.0, is_version=True)],
    )

    proc._receive_and_handle_control_message(_start_inference_message(job))

    messages = _emitted_messages(proc)
    results = _result_messages(proc)
    assert len(results) == 1
    fault = results[0]
    assert fault.info == AUX_RESOLVE_FAILED_INFO
    assert fault.state is GENERATION_STATE.faulted

    fault_index = messages.index(fault)
    waiting_after = [
        m
        for m in messages[fault_index + 1 :]
        if isinstance(m, HordeProcessStateChangeMessage) and m.process_state is HordeProcessState.WAITING_FOR_JOB
    ]
    assert waiting_after, "expected a WAITING_FOR_JOB state change after the aux-resolve fault"

    proc.preload_model.assert_not_called()  # pyrefly: ignore
    proc.start_inference.assert_not_called()  # pyrefly: ignore
    assert proc._end_process is False
    _assert_no_downloads(lora, ti)


def test_missing_ti_faults_job_retryably() -> None:
    """A present LoRA but a missing textual inversion still faults the job with the aux-resolve tag."""
    lora = _FakeLoraManager(available=True)
    ti = _FakeTiManager(key=None)
    proc = _make_proc(shared_model_manager=_FakeSharedModelManager(lora, ti))
    job = make_job_pop_response(
        model=_RESIDENT_MODEL,
        loras=[LorasPayloadEntry(name="2498503", model=0.9, clip=1.0, is_version=True)],
        tis=[TIPayloadEntry(name="missing-embedding")],
    )

    proc._receive_and_handle_control_message(_start_inference_message(job))

    results = _result_messages(proc)
    assert len(results) == 1
    assert results[0].info == AUX_RESOLVE_FAILED_INFO
    assert results[0].state is GENERATION_STATE.faulted
    proc.start_inference.assert_not_called()  # pyrefly: ignore
    proc.preload_model.assert_not_called()  # pyrefly: ignore
    _assert_no_downloads(lora, ti)


def test_dry_run_resolve_short_circuits_without_touching_managers() -> None:
    """With dry-run inference the resolve returns True without consulting the managers at all."""
    lora = _FakeLoraManager(available=False)
    ti = _FakeTiManager(key=None)
    proc = _make_proc(shared_model_manager=_FakeSharedModelManager(lora, ti), dry_run=True)
    job = make_job_pop_response(
        model=_RESIDENT_MODEL,
        loras=[LorasPayloadEntry(name="2498503", model=0.9, clip=1.0, is_version=True)],
        tis=[TIPayloadEntry(name="badhands")],
    )

    assert proc._resolve_aux_models(job) is True
    assert lora.refresh_calls == 0 and ti.refresh_calls == 0
    assert lora.presence_probes == [] and ti.presence_probes == []
    _assert_no_downloads(lora, ti)


def test_skipped_lora_absent_on_disk_resolves_and_proceeds_to_inference() -> None:
    """A LoRA not on disk but listed in ``skipped_aux_models`` is tolerated: the job samples with no fault.

    A terminal ad-hoc rejection (invalid, too large, or NSFW on an SFW-only worker) means the file will never
    land on disk, so the generator skips it rather than the child faulting the whole job.
    """
    lora = _FakeLoraManager(available=False)
    ti = _FakeTiManager(key="unused")
    proc = _make_proc(shared_model_manager=_FakeSharedModelManager(lora, ti))
    job = make_job_pop_response(
        model=_RESIDENT_MODEL,
        loras=[LorasPayloadEntry(name="2498503", model=0.9, clip=1.0, is_version=True)],
    )
    message = HordeInferenceControlMessage(
        control_flag=HordeControlFlag.START_INFERENCE,
        horde_model_name=_RESIDENT_MODEL,
        sdk_api_job_info=job,
        skipped_aux_models=[AuxModelRef(kind=AuxModelKind.LORA, name="2498503", is_version=True)],
    )

    proc._receive_and_handle_control_message(message)

    proc.start_inference.assert_called_once()  # pyrefly: ignore
    results = _result_messages(proc)
    assert len(results) == 1
    assert results[0].state is GENERATION_STATE.ok
    assert results[0].info != AUX_RESOLVE_FAILED_INFO
    _assert_no_downloads(lora, ti)


def test_skipped_set_only_tolerates_the_named_file_not_a_different_missing_one() -> None:
    """A different missing LoRA (not the one in the skip set) still faults the job with the aux-resolve tag."""
    lora = _FakeLoraManager(available=False)
    ti = _FakeTiManager(key="unused")
    proc = _make_proc(shared_model_manager=_FakeSharedModelManager(lora, ti))
    job = make_job_pop_response(
        model=_RESIDENT_MODEL,
        loras=[LorasPayloadEntry(name="2498503", model=0.9, clip=1.0, is_version=True)],
    )

    # The skip set names a different file, so this job's genuinely-missing LoRA is not tolerated.
    assert (
        proc._resolve_aux_models(job, [AuxModelRef(kind=AuxModelKind.LORA, name="other", is_version=False)]) is False
    )
    _assert_no_downloads(lora, ti)


def test_resolve_never_downloads_in_present_or_missing_branch() -> None:
    """Neither the resolvable branch nor the first-miss branch may reach any manager download method."""
    present_lora = _FakeLoraManager(available=True)
    present_ti = _FakeTiManager(key="badhands")
    present_proc = _make_proc(shared_model_manager=_FakeSharedModelManager(present_lora, present_ti))
    present_job = make_job_pop_response(
        model=_RESIDENT_MODEL,
        loras=[LorasPayloadEntry(name="2498503", model=0.9, clip=1.0, is_version=True)],
        tis=[TIPayloadEntry(name="badhands")],
    )
    assert present_proc._resolve_aux_models(present_job) is True
    _assert_no_downloads(present_lora, present_ti)

    missing_lora = _FakeLoraManager(available=False)
    missing_ti = _FakeTiManager(key="badhands")
    missing_proc = _make_proc(shared_model_manager=_FakeSharedModelManager(missing_lora, missing_ti))
    missing_job = make_job_pop_response(
        model=_RESIDENT_MODEL,
        loras=[LorasPayloadEntry(name="2498503", model=0.9, clip=1.0, is_version=True)],
        tis=[TIPayloadEntry(name="badhands")],
    )
    assert missing_proc._resolve_aux_models(missing_job) is False
    _assert_no_downloads(missing_lora, missing_ti)
