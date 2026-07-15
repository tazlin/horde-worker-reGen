"""Reproduces a spurious deadlock latch when a pending job's auxiliary prefetch is still in flight.

A job that carries LoRAs (or textual inversions) not yet on disk is popped and then held pending while the
background download process places its files. By design that job holds no inference lane: its dispatch is
gated on the auxiliary prefetch completing. The observed failure: seconds after such a pop, with every
inference process idle (``WAITING_FOR_JOB``) and the sole pending job aux-gated, ``detect_deadlock`` latched
both the queue- and general-deadlock flags. "Pending plus every process idle" is the aux-prefetch gate
working exactly as designed, not a wedge, yet the latched flag escalated through the structural-wedge window
into a save-our-ship soft reset that pointlessly rebuilt both pools mid-download.

The coordinator's per-job prefetch deadline is the bound on this hold: while it is live the job must not fuel
either deadlock condition, and once it lapses (the download genuinely stalled) detection engages as before.
"""

from __future__ import annotations

import time

from horde_sdk.ai_horde_api.apimodels import LorasPayloadEntry

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)

_STRUCTURAL_WEDGE_AGE = 25.0
"""Seconds a latched queue-deadlock flag would have to persist to become a structural wedge."""


async def _pop_aux_gated_job(pm: object, *, model: str = "resident") -> object:
    """Track a LoRA-bearing job as pending inference and arm its live aux-prefetch deadline."""
    job = make_job_pop_response(model=model, loras=[LorasPayloadEntry(name="a-lora")])
    await track_popped_job_async(pm._job_tracker, job)  # type: ignore[attr-defined]
    pm._aux_prefetch_coordinator.on_job_popped(job)  # type: ignore[attr-defined]
    return job


async def test_aux_gated_pending_job_does_not_latch_deadlock() -> None:
    """A pending job whose prefetch is still live must not latch either deadlock flag while all slots idle.

    This is the incident: one LoRA job pending, its files downloading in the background, every inference
    slot ``WAITING_FOR_JOB``. Repeated detection ticks across the structural-wedge window must leave both
    flags clear, so nothing escalates to a soft reset.
    """
    pm = make_testable_process_manager()
    pm._state.last_job_pop_time = time.time() - 60  # last pop not recent; detection is live

    idle = make_mock_process_info(1, model_name="resident", state=HordeProcessState.WAITING_FOR_JOB)
    pm._process_map[1] = idle

    job = await _pop_aux_gated_job(pm)
    assert job.id_ is not None
    assert job.id_ in pm._aux_prefetch_coordinator.job_ids_with_live_deadlines()

    for _ in range(10):
        pm.detect_deadlock()

    snapshot = pm._message_dispatcher.get_deadlock_snapshot()
    assert snapshot.in_queue_deadlock is False
    assert snapshot.in_deadlock is False
    # The flag was never set, so the wedge assessment cannot fire even at the structural-wedge age.
    pm._message_dispatcher._last_queue_deadlock_detected_time = time.time() - _STRUCTURAL_WEDGE_AGE
    assert snapshot.indicates_structural_wedge() is False
    assert pm._recovery_coordinator.assess_wedge() is False


async def test_deadlock_engages_once_aux_deadline_lapses() -> None:
    """Bounded patience: once the prefetch deadline expires, the still-pending job fuels deadlock again.

    The hold is only for as long as the coordinator reports a live deadline. Backdate the job's deadline
    into the past (a genuinely stalled prefetch) and the same all-idle picture must now latch, proving the
    aux shield cannot suppress a real stall indefinitely.
    """
    pm = make_testable_process_manager()
    pm._state.last_job_pop_time = time.time() - 60

    idle = make_mock_process_info(1, model_name="resident", state=HordeProcessState.WAITING_FOR_JOB)
    pm._process_map[1] = idle

    job = await _pop_aux_gated_job(pm)
    assert job.id_ is not None
    # The prefetch never resolved and its deadline has passed: no live hold remains.
    pm._aux_prefetch_coordinator._deadlines[job.id_] = time.time() - 5
    assert pm._aux_prefetch_coordinator.job_ids_with_live_deadlines() == set()

    pm.detect_deadlock()

    assert pm._message_dispatcher.get_deadlock_snapshot().in_queue_deadlock is True


async def test_aux_hold_does_not_shield_a_second_plain_pending_job() -> None:
    """A plain pending job stalled alongside an aux-held one must still latch deadlock (shield is scoped).

    The aux hold suppresses only the aux-gated job's contribution. A second job with no auxiliary files,
    pending on an idle pool, is a genuine stall and must still latch, so the shield never masks a real wedge.
    """
    pm = make_testable_process_manager()
    pm._state.last_job_pop_time = time.time() - 60

    idle = make_mock_process_info(1, model_name="resident", state=HordeProcessState.WAITING_FOR_JOB)
    pm._process_map[1] = idle

    aux_job = await _pop_aux_gated_job(pm)
    assert aux_job.id_ is not None
    assert aux_job.id_ in pm._aux_prefetch_coordinator.job_ids_with_live_deadlines()

    plain_job = make_job_pop_response(model="resident")
    await track_popped_job_async(pm._job_tracker, plain_job)

    pm.detect_deadlock()

    assert pm._message_dispatcher.get_deadlock_snapshot().in_queue_deadlock is True
