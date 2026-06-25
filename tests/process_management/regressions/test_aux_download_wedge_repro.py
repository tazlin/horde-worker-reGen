"""Reproduces the disk-full aux-download wedge that soft-resets the pools (a process-recovery storm).

When the cache drive fills (``[Errno 28] No space left on device``), a CivitAI ad-hoc LoRA download keeps
retrying in hordelib's background download thread. The inference child then dispatches its next
resident-model job and calls ``download_aux_models``,
which drains in-flight downloads via ``lora_manager.reset_adhoc_cache()`` and, for an already-available
LoRA, ``lora_manager.wait_for_downloads(0)``. Both of those waits are unbounded and run *before*
``download_aux_models`` sends its ``DOWNLOADING_AUX_MODEL`` busy state or starts the aux-download
heartbeat thread. So while the child sits blocked on the wedged background download:

* the parent still sees the slot as ``WAITING_FOR_JOB`` (``can_accept_job()`` stays True), so
  ``_inference_slot_owns_job`` reports no live owner and the orphaned-job watchdog punts the
  in-progress job every 30s; the repeated punts escalate to a Save-our-ship soft reset, and
* the slot's heartbeat goes stale, feeding the same "crashed or hung" verdict.

The soft reset rebuilds both pools and counts two process recoveries for a single wedge.

The fix is to publish the busy state and start the heartbeat loop *before* any blocking drain, so the
parent keeps seeing the slot as busy-and-alive for the whole aux phase. This test asserts that ordering:
by the time the lora manager performs its first blocking drain, both protective signals are already in
place. On the unfixed code they are not, so it fails exactly where the theory predicts.
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


class _WedgedDrainLoraManager:
    """A lora manager whose blocking drains stand in for a wedged background download.

    The job's own LoRA is reported already-available, so ``download_aux_models`` performs no fresh
    fetch: the only blocking it does is draining in-flight downloads (``reset_adhoc_cache`` and the
    available-LoRA ``wait_for_downloads``). Each drain records whether the child had already published
    its busy state and started its heartbeat loop, which is the behaviour under test.
    """

    def __init__(self, observations: dict[str, Any]) -> None:
        self._obs = observations

    def _snapshot(self, where: str) -> None:
        self._obs.setdefault("drain_order", []).append(where)
        self._obs.setdefault("busy_state_at_first_drain", self._obs["busy_state_sent"])
        self._obs.setdefault("heartbeat_at_first_drain", self._obs["heartbeat_started"])

    def load_model_database(self) -> None:
        return None

    def reset_adhoc_cache(self) -> None:
        # First thing download_aux_models does after the no-LoRA short-circuit; this is what blocks on an
        # ENOSPC-retrying background download.
        self._snapshot("reset_adhoc_cache")

    def is_model_available(self, name: str | int) -> bool:
        return True

    def wait_for_downloads(self, timeout: float | None = None) -> None:
        # The available-LoRA path: wait_for_downloads(0) is also unbounded and also blocks here.
        self._snapshot("wait_for_downloads")

    def fetch_adhoc_lora(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - not reached
        raise AssertionError("the available LoRA must not be re-fetched")

    def save_reference_to_disk(self) -> None:
        return None


def _bare_inference_proc(aux_lock: synchronize.Lock, observations: dict[str, Any]) -> HordeInferenceProcess:
    """Build a HordeInferenceProcess wired only for the ``download_aux_models`` drain path."""
    proc = object.__new__(HordeInferenceProcess)
    proc._aux_model_lock = aux_lock
    proc._shared_model_manager = SimpleNamespace(  # pyrefly: ignore
        manager=SimpleNamespace(lora=_WedgedDrainLoraManager(observations)),
    )
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
    return proc


def test_aux_download_marks_busy_and_heartbeats_before_blocking_drains() -> None:
    """The busy state + heartbeat loop must be live before the first blocking drain.

    Otherwise a wedged in-flight download (e.g. a full disk retrying ENOSPC) stalls the child while the
    parent still reads the slot as idle and unheartbeating, which punts the in-progress job (orphan
    watchdog) and feeds the Save-our-ship soft reset that produced the recovery storm.
    """
    observations: dict[str, Any] = {"busy_state_sent": False, "heartbeat_started": False}
    aux_lock = multiprocessing.Lock()
    proc = _bare_inference_proc(aux_lock, observations)
    job = make_job_pop_response(
        model="WAI-NSFW-illustrious-SDXL",
        loras=[LorasPayloadEntry(name="1683285", model=1.0, clip=1.0, is_version=True)],
    )

    proc.download_aux_models(job)

    assert observations.get("drain_order"), "expected download_aux_models to reach a blocking drain"
    assert observations["heartbeat_at_first_drain"] is True, (
        "aux-download heartbeat loop must start before the first blocking drain; otherwise the slot's "
        "heartbeat goes stale during a wedged download and the parent grades it 'crashed or hung'"
    )
    assert observations["busy_state_at_first_drain"] is True, (
        "DOWNLOADING_AUX_MODEL busy state must be published before the first blocking drain; otherwise "
        "the parent still sees WAITING_FOR_JOB (can_accept_job) and the orphaned-job watchdog punts the "
        "in-progress job, escalating to a soft reset"
    )
    # The shared aux lock must be left free, not wedged.
    assert aux_lock.acquire(block=False) is True
    aux_lock.release()
