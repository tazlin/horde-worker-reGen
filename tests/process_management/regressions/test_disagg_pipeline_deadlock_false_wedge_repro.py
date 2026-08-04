"""Reproduces a spurious deadlock latch while a job advances through the disaggregated pipeline.

A disaggregated job occupies an inference slot only for its sampling stage. The moment the sampler emits its
latent it returns to ``WAITING_FOR_JOB`` while the service lanes finish the job (conditioning, VAE
encode/decode, post-processing). The observed failure: for that window every inference process is idle with
the job still in progress, so ``detect_deadlock`` latched the general deadlock flag and cleared it again a
fraction of a second later, once per job, filling the log with detected/cleared pairs and priming the
recovery machinery against a worker that was serving normally.

A job the orchestrator holds is doing lane work by design, so it must fuel neither deadlock condition. The
hold is bounded by the orchestrator's own stage patience: a job whose stage never returns is faulted or
rerouted monolithic and leaves the pipeline, after which the same all-idle picture latches as before.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    mark_job_in_progress_async,
    track_popped_job_async,
)


async def _register_disaggregated_job(pm: object, *, model: str = "resident") -> object:
    """Track a job as inference-in-progress and admit it into the disaggregated pipeline."""
    job = make_job_pop_response(model=model)
    await track_popped_job_async(pm._job_tracker, job)  # type: ignore[attr-defined]
    await mark_job_in_progress_async(pm._job_tracker, job)  # type: ignore[attr-defined]
    pm._disaggregation_orchestrator.register(  # type: ignore[attr-defined]
        SimpleNamespace(sdk_api_job_info=job),
        needs_source_latent=False,
        pinned_sampler_process_id=1,
    )
    return job


async def test_job_in_the_disaggregated_pipeline_does_not_latch_deadlock() -> None:
    """A held disaggregated job with every inference slot idle must leave both deadlock flags clear."""
    pm = make_testable_process_manager()
    pm._state.last_job_pop_time = time.time() - 60  # last pop not recent; detection is live

    pm._process_map[1] = make_mock_process_info(1, model_name="resident", state=HordeProcessState.WAITING_FOR_JOB)

    job = await _register_disaggregated_job(pm)
    assert str(job.id_) in pm._disaggregation_orchestrator.held_job_ids()
    assert len(pm._job_tracker.jobs_in_progress) == 1

    for _ in range(10):
        pm.detect_deadlock()

    snapshot = pm._message_dispatcher.get_deadlock_snapshot()
    assert snapshot.in_deadlock is False
    assert snapshot.in_queue_deadlock is False


async def test_deadlock_engages_once_the_job_leaves_the_pipeline() -> None:
    """Bounded shield: a job the orchestrator no longer holds fuels detection exactly as before."""
    pm = make_testable_process_manager()
    pm._state.last_job_pop_time = time.time() - 60

    pm._process_map[1] = make_mock_process_info(1, model_name="resident", state=HordeProcessState.WAITING_FOR_JOB)

    job = await _register_disaggregated_job(pm)
    pm._disaggregation_orchestrator.release_job(job.id_)
    assert pm._disaggregation_orchestrator.held_job_ids() == set()

    pm.detect_deadlock()

    assert pm._message_dispatcher.get_deadlock_snapshot().in_deadlock is True


async def test_disaggregated_hold_does_not_shield_a_second_stalled_job() -> None:
    """The shield is scoped to the held job: a plain pending job on an idle pool still latches."""
    pm = make_testable_process_manager()
    pm._state.last_job_pop_time = time.time() - 60

    pm._process_map[1] = make_mock_process_info(1, model_name="resident", state=HordeProcessState.WAITING_FOR_JOB)

    await _register_disaggregated_job(pm)
    plain_job = make_job_pop_response(model="resident")
    await track_popped_job_async(pm._job_tracker, plain_job)

    pm.detect_deadlock()

    snapshot = pm._message_dispatcher.get_deadlock_snapshot()
    assert snapshot.in_deadlock is True
    assert snapshot.in_queue_deadlock is True
