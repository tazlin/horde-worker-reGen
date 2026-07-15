"""Reproduces an owner-aware decode hold for a reclaim-ladder-paused VAE lane.

The decode-drain eligibility gate (its sibling repro) stops a NEW VAE-lane pause from executing while a
disaggregated decode is already queued. It cannot cover the window where the pause lands earlier: a
reclaim-ladder pause is executed while a job is still encoding or sampling (a job merely sampling does not
block the pause), and the job then finishes sampling and reaches ``AWAITING_LATENT_DECODE`` a few ticks later
to find the lane already paused off-GPU.

A reclaim-ladder pause is bounded by construction: the post-processing borrow's idle-release restores the lane
within seconds of the borrower going idle, and a self-heal backstop covers an orphaned pause. So for that
finished-sampling job the orchestrator holds at the decode stage for the restore rather than rerouting
monolithic and discarding the completed sampling to avoid a wait that is typically under a minute.

The specification here:

* a job at latent decode whose VAE lane is paused by the reclaim ladder holds (no reroute, no fault) for the
  restore, up to a bound, then dispatches its decode and completes disaggregated once the lane returns;
* a whole-card residency pause of the same lane still reroutes at once (its pause lasts a heavy model's whole
  residency, so waiting is not worth the finished sample);
* a reclaim-ladder pause that never lifts reroutes at the bound as a backstop, losing no job and raising no
  fault; and
* the hold never ages the job toward the no-role patience fault.
"""

from __future__ import annotations

from types import SimpleNamespace

from loguru import logger

from horde_worker_regen.process_management.ipc.messages import (
    GENERATION_STATE,
    HordeImageResult,
    HordeVaeDecodeResultMessage,
)
from horde_worker_regen.process_management.lifecycle.process_lifecycle import PauseOwner
from horde_worker_regen.process_management.workers.disaggregation_orchestrator import (
    _DECODE_PAUSE_WAIT_SECONDS,
    _STAGE_PATIENCE_SECONDS,
    DisaggJobStage,
)

from ..test_disaggregation_orchestrator import (
    _SAMPLER_PID,
    _drive_to_decode_pending,
    _job,
    _make_harness,
)

_IMAGE_LANE_PID = 3


def _pause_image_lane(h: SimpleNamespace, *, owner: PauseOwner | None) -> None:
    """Model the VAE lane paused off-GPU with ``owner`` holding the pause: no lane process, pause predicate True."""
    h.orchestrator._find_image_lane = lambda: None
    h.orchestrator._image_lane_paused = lambda: True
    h.orchestrator._image_lane_pause_owner = lambda: owner


def _restore_image_lane(h: SimpleNamespace) -> None:
    """Model the VAE lane restored on-GPU: the lane process returns and the pause predicates clear."""
    h.orchestrator._find_image_lane = lambda: h.image_lane
    h.orchestrator._image_lane_paused = lambda: False
    h.orchestrator._image_lane_pause_owner = lambda: None


async def test_reclaim_ladder_paused_decode_holds_then_completes_on_restore() -> None:
    """A finished-sampling job holds at decode for a reclaim-ladder-paused VAE lane, then completes on restore.

    Expected-RED before the owner-aware hold: the decode stage saw a deliberately-paused lane and rerouted
    monolithically on the first tick, discarding the finished sampling. The fix holds the job (no reroute, no
    fault) for the bounded restore, then dispatches the decode and completes the job disaggregated once the lane
    returns.
    """
    h = _make_harness()
    job = _job()
    job_id = job.sdk_api_job_info.id_
    await _drive_to_decode_pending(h, job, _SAMPLER_PID)

    _pause_image_lane(h, owner=PauseOwner.RECLAIM_LADDER)

    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(message.record["message"]), level="INFO")
    try:
        # Several ticks within the wait bound: the job holds, never rerouting and never faulting.
        for elapsed in (0.0, 5.0, 20.0, _DECODE_PAUSE_WAIT_SECONDS - 1.0):
            h.virtual_now[0] = elapsed
            h.orchestrator.tick()
            assert h.rerouted == [], (
                "the finished-sampling decode was rerouted instead of held for the bounded restore"
            )
            assert h.completed == [], "the held decode was faulted rather than waiting for the lane to restore"
            assert h.orchestrator.has_job(job)
            assert h.orchestrator._jobs[str(job_id)].stage == DisaggJobStage.AWAITING_LATENT_DECODE
    finally:
        logger.remove(sink_id)

    # One edge-latched INFO names the held-job count for live forensics, logged once for the whole hold episode.
    hold_lines = [line for line in lines if "latent decode" in line and "reclaim-ladder-paused" in line]
    assert len(hold_lines) == 1, "the decode-pause hold must emit exactly one edge-latched INFO naming the held count"

    # The lane restores: the very next tick re-attempts the held decode with no manual kick and dispatches it.
    _restore_image_lane(h)
    h.orchestrator.tick()
    assert len(h.image_lane.sent) == 1, "the held decode did not dispatch once the VAE lane restored"

    await h.orchestrator.handle_stage_result(
        HordeVaeDecodeResultMessage(
            process_id=_IMAGE_LANE_PID,
            process_launch_identifier=0,
            info="",
            job_id=job_id,
            job_image_results=[HordeImageResult(image_bytes=b"img")],
            state=GENERATION_STATE.ok,
        ),
    )
    assert len(h.completed) == 1, "the job did not complete disaggregated after its decode landed"
    _ji, images, state, fault = h.completed[0]
    assert state == GENERATION_STATE.ok
    assert len(images) == 1
    assert fault is None
    assert h.rerouted == [], "the job was rerouted monolithic despite completing its disaggregated decode"
    assert not h.orchestrator.has_job(job)


async def test_whole_card_paused_decode_reroutes_at_once() -> None:
    """Control: a whole-card residency pause of the VAE lane still reroutes the decode immediately (today's behavior).

    Expected-RED is not applicable: this pins the unchanged whole-card behavior so the owner-aware hold does not
    accidentally start holding a residency pause that genuinely lasts minutes.
    """
    h = _make_harness()
    job = _job()
    await _drive_to_decode_pending(h, job, _SAMPLER_PID)

    _pause_image_lane(h, owner=PauseOwner.WHOLE_CARD)
    h.orchestrator.tick()

    assert h.rerouted == [job], "a whole-card-paused decode must reroute at once, not hold"
    assert h.completed == []
    assert not h.orchestrator.has_job(job)
    assert h.reserved == set()


async def test_reclaim_ladder_pause_that_never_lifts_reroutes_at_the_bound() -> None:
    """Backstop: a reclaim-ladder pause that never restores reroutes at the bound, losing no job and no fault.

    Expected-RED before the hold: the job rerouted on the first paused tick (no bound involved). The fix holds it
    until the wait bound elapses, then reroutes monolithically as the backstop so a pause that never lifts can
    never strand the job.
    """
    h = _make_harness()
    job = _job()
    await _drive_to_decode_pending(h, job, _SAMPLER_PID)

    _pause_image_lane(h, owner=PauseOwner.RECLAIM_LADDER)

    h.virtual_now[0] = 0.0
    h.orchestrator.tick()  # anchors the hold; no reroute yet
    assert h.rerouted == []
    assert h.orchestrator.has_job(job)

    h.virtual_now[0] = _DECODE_PAUSE_WAIT_SECONDS + 1.0
    h.orchestrator.tick()

    assert h.rerouted == [job], "the pause outlasted the hold window but the job was not rerouted as the backstop"
    assert h.completed == [], "the backstop reroute must not fault the job"
    assert not h.orchestrator.has_job(job)
    assert h.reserved == set()


async def test_decode_hold_does_not_age_toward_the_patience_fault() -> None:
    """The hold is kept clear of the no-role patience clock, so a legitimate wait never accumulates a fault.

    Expected-RED before the hold: the decode never held at all (it rerouted at once), so there was no hold whose
    patience-clock interaction to pin. The fix must hold without ever anchoring ``first_stalled_at``: were the
    hold to age like a genuine no-role stall it would fault at ``_STAGE_PATIENCE_SECONDS``.
    """
    h = _make_harness()
    job = _job()
    job_id = job.sdk_api_job_info.id_
    await _drive_to_decode_pending(h, job, _SAMPLER_PID)

    _pause_image_lane(h, owner=PauseOwner.RECLAIM_LADDER)

    # Tick repeatedly across the whole hold window: the patience anchor must never be set, so no fault can accrue.
    assert _DECODE_PAUSE_WAIT_SECONDS < _STAGE_PATIENCE_SECONDS
    for elapsed in (0.0, 10.0, 30.0, _DECODE_PAUSE_WAIT_SECONDS - 1.0):
        h.virtual_now[0] = elapsed
        h.orchestrator.tick()
        state = h.orchestrator._jobs[str(job_id)]
        assert state.first_stalled_at is None, "the decode hold aged into the no-role patience clock"
        assert h.completed == [], "a legitimate decode hold must never fault the job"
        assert h.rerouted == []
