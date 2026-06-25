"""Reproduces the aux-model-lock over-release that tears down a still-alive inference child.

``_aux_model_lock`` is a *single, bounded* ``multiprocessing.Lock`` created once in the manager and
shared by every inference child **and** the supervisor. The supervisor force-releases it on a slot it
is replacing (``HordeProcessLifecycleManager._release_held_primitives``). When that reclaim lands
while a still-alive child is inside the aux-download critical section (e.g. a slow LoRA download the
watchdog flagged as stuck), the child's own block-exit release pushes the bounded lock past its
ceiling and raises ``ValueError: semaphore or lock released too many times``. Because
``download_aux_models`` is ``@logger.catch(reraise=True)``, a naive ``with self._aux_model_lock:``
lets that over-release propagate up to ``receive_and_handle_control_messages`` and end the *whole*
inference process, which the supervisor then reaps as "crashed or hung" and restarts, re-loading the
model: a self-sustaining churn loop that re-loads on every LoRA job under supervisor pressure.

The protected work is already finished when the block-exit release runs and the lock is genuinely
free, so the over-release is benign and must not be fatal. This mirrors the supervisor side, which
already swallows the symmetric ``ValueError`` in ``_release_held_primitives``.
"""

from __future__ import annotations

import multiprocessing
import queue
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeInferenceControlMessage,
)
from horde_worker_regen.process_management.workers.inference_process import HordeInferenceProcess

from .conftest import make_job_pop_response


class _LockReclaimingManager:
    """Lora-manager holder that force-releases the shared aux lock the first time it is touched.

    Stands in for the supervisor concurrently reclaiming the shared aux-model lock (via
    ``_release_held_primitives``) while this still-alive child is mid critical-section.
    """

    def __init__(self, aux_lock: Any) -> None:  # noqa: ANN401
        self._aux_lock = aux_lock
        self._fired = False

    @property
    def lora(self) -> Mock:
        if not self._fired:
            self._fired = True
            # The supervisor's force-release lands here, inside the child's `with` block.
            self._aux_lock.release()
        return Mock()  # a non-None lora manager; a no-LoRA job returns before it is used


def _bare_inference_proc(aux_lock: Any, active_model: str) -> HordeInferenceProcess:  # noqa: ANN401
    """Build a HordeInferenceProcess wired only for the resident-model aux-download path.

    A real construction spins up HordeLib/SharedModelManager; only ``download_aux_models`` and the
    resident START_INFERENCE branch are exercised here, so everything else is stubbed.
    """
    proc = object.__new__(HordeInferenceProcess)
    proc._aux_model_lock = aux_lock
    proc._shared_model_manager = SimpleNamespace(manager=_LockReclaimingManager(aux_lock))  # pyrefly: ignore
    proc._active_model_name = active_model
    proc._end_process = False
    proc._control_inbox = queue.Queue()
    proc.on_horde_model_state_change = Mock()  # pyrefly: ignore
    proc.send_process_state_change_message = Mock()  # pyrefly: ignore
    # The resident START_INFERENCE branch runs inference once the aux download returns; stub it (and
    # the result hand-off) so the test isolates the lock behaviour, not the HordeLib inference path.
    proc.start_inference = Mock(return_value=[object()])  # pyrefly: ignore
    proc.send_inference_result_message = Mock()  # pyrefly: ignore
    return proc


def test_download_aux_models_survives_supervisor_lock_reclaim() -> None:
    """A supervisor-forced release of the shared aux lock must not crash ``download_aux_models``.

    Directly reproduces the logged ``with self._aux_model_lock:`` over-release: the lock is released
    out from under the critical section, so the block's exit release would exceed the bound. The
    method must complete normally (its work is done, the lock is free) instead of raising.
    """
    aux_lock = multiprocessing.Lock()
    proc = _bare_inference_proc(aux_lock, active_model="CyberRealistic Pony")
    job = make_job_pop_response(model="CyberRealistic Pony", loras=None)

    result = proc.download_aux_models(job)

    assert result is None
    # The shared lock must be left usable (free), not wedged: a fresh acquire/release round-trips.
    assert aux_lock.acquire(block=False) is True
    aux_lock.release()


def test_aux_lock_over_release_does_not_end_the_inference_process() -> None:
    """The benign over-release must not tear the inference process down (the churn-loop trigger).

    Drives the real control-message pump down the resident-model START_INFERENCE branch, the path that
    calls ``download_aux_models`` for an already-loaded model. On the buggy code the over-release
    propagates out of ``download_aux_models`` and ``receive_and_handle_control_messages`` flips
    ``_end_process`` -> the supervisor then reaps and restarts the slot. The process must stay alive.
    """
    aux_lock = multiprocessing.Lock()
    model = "CyberRealistic Pony"
    proc = _bare_inference_proc(aux_lock, active_model=model)
    job = make_job_pop_response(model=model, loras=None)

    proc._control_inbox.put(
        HordeInferenceControlMessage(
            control_flag=HordeControlFlag.START_INFERENCE,
            horde_model_name=model,
            sdk_api_job_info=job,
        ),
    )

    proc.receive_and_handle_control_messages()

    assert proc._end_process is False
    proc.on_horde_model_state_change.assert_called_once()  # pyrefly: ignore
