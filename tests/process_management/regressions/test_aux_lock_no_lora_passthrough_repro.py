"""A LoRA download on one slot must not strand no-LoRA fill work on another, nor look dead while it waits.

``download_aux_models`` runs on the inference slot that will sample the job, and it serializes on a single
process-wide ``_aux_model_lock`` shared by every inference child. Two failure modes fed the "download ties
up the whole worker" pathology:

* A job with **no LoRAs** (the quick, resident work the scheduler line-skips onto an idle lane precisely to
  keep the GPU fed while another slot downloads) used to acquire that shared lock before discovering it had
  nothing to fetch. So it blocked for the entire duration of an unrelated slot's CivitAI download, stranding
  the card the line-skip was meant to feed. ``test_no_lora_job_does_not_block_on_held_aux_lock`` holds the
  lock and asserts a no-LoRA job sails through anyway.

* A job **with LoRAs** that serializes behind the head's download used to publish its
  ``DOWNLOADING_AUX_MODEL`` busy state and start its liveness heartbeat only *after* the blocking acquire.
  While blocked on the acquire it therefore stayed in its pre-dispatch state with no heartbeat, so the
  parent read the slot as idle (``can_accept_job``) and punted the in-progress job, and the stale heartbeat
  fed the "crashed or hung" verdict, both escalating to a soft reset.
  ``test_lora_job_marks_busy_and_heartbeats_while_blocked_on_aux_lock`` holds the lock and asserts both
  protective signals are already live while the call is still blocked on the acquire.
"""

from __future__ import annotations

import multiprocessing
import threading
from multiprocessing import synchronize
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.workers.inference_process import HordeInferenceProcess
from tests.process_management.conftest import make_job_pop_response


class _ExplodingLoraManager:
    """A lora manager that fails if touched, proving the no-LoRA path never reaches the download machinery."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"a no-LoRA job must not call the lora manager (called {name!r})")


class _AvailableLoraManager:
    """A lora manager whose single LoRA is already on disk, so the only blocking is the aux lock itself."""

    def load_model_database(self) -> None:
        return None

    def reset_adhoc_cache(self) -> None:
        return None

    def is_model_available(self, name: str | int) -> bool:
        return True

    def wait_for_downloads(self, timeout: float | None = None) -> None:
        return None

    def are_downloads_complete(self) -> bool:
        return True

    def save_reference_to_disk(self) -> None:
        return None

    def fetch_adhoc_lora(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - not reached
        raise AssertionError("the available LoRA must not be re-fetched")


def _bare_inference_proc(aux_lock: synchronize.Lock, lora_manager: object, observations: dict[str, Any]) -> Any:
    """Build a HordeInferenceProcess wired only for the ``download_aux_models`` path."""
    proc = object.__new__(HordeInferenceProcess)
    proc._aux_model_lock = aux_lock
    proc._shared_model_manager = SimpleNamespace(manager=SimpleNamespace(lora=lora_manager))  # pyrefly: ignore
    proc.process_id = 1

    def _record_busy(*args: object, **kwargs: object) -> None:
        if kwargs.get("process_state") == HordeProcessState.DOWNLOADING_AUX_MODEL:
            observations["busy_state_sent"] = True

    def _start_heartbeat() -> tuple[threading.Event, Any]:
        observations["heartbeat_started"] = True
        return threading.Event(), Mock()

    proc.send_aux_model_message = Mock(side_effect=_record_busy)  # pyrefly: ignore
    proc.send_heartbeat_message = Mock()  # pyrefly: ignore
    proc._start_aux_download_heartbeat_thread = Mock(side_effect=_start_heartbeat)  # pyrefly: ignore
    proc._send_download_metrics_if_any = Mock()  # pyrefly: ignore
    proc._enforce_lora_disk_floor = Mock()  # pyrefly: ignore
    return proc


def test_no_lora_job_does_not_block_on_held_aux_lock() -> None:
    """A no-LoRA job completes even while another slot holds the shared aux lock for a long download.

    This is the "keep the GPU fed" guarantee: the quick, resident, no-LoRA work that gets line-skipped onto
    an idle lane must not serialize behind the very download it was chosen to skip past.
    """
    observations: dict[str, Any] = {"busy_state_sent": False, "heartbeat_started": False}
    aux_lock = multiprocessing.Lock()
    proc = _bare_inference_proc(aux_lock, _ExplodingLoraManager(), observations)
    job = make_job_pop_response(model="stable_diffusion", loras=None)

    # Stand in for another inference slot mid-download: the shared lock is held for the whole test.
    assert aux_lock.acquire(block=False) is True

    result: dict[str, Any] = {}

    def _run() -> None:
        result["value"] = proc.download_aux_models(job)

    worker = threading.Thread(target=_run, name="no-lora-aux", daemon=True)
    worker.start()
    worker.join(timeout=5.0)

    assert not worker.is_alive(), "a no-LoRA job blocked on the held aux lock (it should never take the lock)"
    assert result["value"] is None
    assert observations["busy_state_sent"] is False, "a no-LoRA job should not publish a DOWNLOADING_AUX_MODEL state"

    aux_lock.release()


def test_lora_job_marks_busy_and_heartbeats_while_blocked_on_aux_lock() -> None:
    """A LoRA job serializing behind a held aux lock is visibly busy-and-alive during the wait.

    Otherwise, while blocked on the acquire it keeps its pre-dispatch state with no heartbeat, so the parent
    reads the slot as idle and punts the in-progress job, and the stale heartbeat feeds a hung verdict.
    """
    observations: dict[str, Any] = {"busy_state_sent": False, "heartbeat_started": False}
    aux_lock = multiprocessing.Lock()
    proc = _bare_inference_proc(aux_lock, _AvailableLoraManager(), observations)
    job = make_job_pop_response(
        model="WAI-NSFW-illustrious-SDXL",
        loras=[LorasPayloadEntry(name="1683285", model=1.0, clip=1.0, is_version=True)],
    )

    # Hold the lock so the job's download_aux_models blocks on the acquire, mid-call.
    assert aux_lock.acquire(block=False) is True

    finished = threading.Event()

    def _run() -> None:
        proc.download_aux_models(job)
        finished.set()

    worker = threading.Thread(target=_run, name="lora-aux", daemon=True)
    worker.start()

    # Give the call time to reach and block on the acquire, then assert liveness *before* releasing.
    def _both_signals_live() -> bool:
        return observations["busy_state_sent"] and observations["heartbeat_started"]

    deadline = threading.Event()
    for _ in range(50):
        if _both_signals_live():
            break
        deadline.wait(0.02)

    try:
        assert observations["busy_state_sent"] is True, (
            "DOWNLOADING_AUX_MODEL must be published before the blocking aux-lock acquire, or the parent "
            "reads the slot as idle and punts the in-progress job"
        )
        assert observations["heartbeat_started"] is True, (
            "the aux-download heartbeat must start before the blocking aux-lock acquire, or the slot's "
            "heartbeat goes stale during the wait and the parent grades it hung"
        )
        assert not finished.is_set(), "the job should still be blocked on the held lock at this point"
    finally:
        aux_lock.release()

    assert finished.wait(timeout=5.0), "the job did not complete after the aux lock was released"
