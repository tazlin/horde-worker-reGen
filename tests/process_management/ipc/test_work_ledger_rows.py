"""Tests for what a work-ledger row states about its job, and the order the rows arrive in.

The ledger is the operator's per-job view. A row therefore has to carry what decides how much work its
job actually is (resolution, steps, batch count, and what the sampler asks the model for per step), and
the rows have to arrive in pop order, which is not the order the tracker holds them in once jobs start
advancing through stages.
"""

from __future__ import annotations

import time

import pytest
from horde_sdk.generation_parameters.image.constraints import SAMPLER_CONSTRAINTS
from horde_sdk.generation_parameters.image.consts import KNOWN_IMAGE_SAMPLERS

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    WorkLedgerProgressUnit,
    WorkLedgerStage,
)
from horde_worker_regen.process_management.jobs.text_generation_coordinator import TextJobInFlight
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind
from horde_worker_regen.text_backends import TextGenerationProgress
from tests.process_management.conftest import (
    make_job_pop_response,
    make_testable_process_manager,
    mark_job_in_progress_async,
    track_popped_job_async,
)

_SDE = SAMPLER_CONSTRAINTS[KNOWN_IMAGE_SAMPLERS.k_dpmpp_sde]


def test_sampler_summary_states_the_order_and_the_measured_cost() -> None:
    """A fixed-rate sampler reports the evaluations one step costs and its measured ratio to k_euler."""
    summary = HordeWorkerProcessManager._sampler_summary("k_dpmpp_sde", "stable_diffusion_xl")

    assert summary is not None
    assert summary.name == "k_dpmpp_sde"
    assert summary.work_per_step == 2
    assert summary.adaptive is False
    assert summary.cost_ratio == _SDE.measured_cost_ratio_sdxl


def test_sampler_summary_prices_a_512_class_baseline_from_the_512_class_column() -> None:
    """The two published ratios differ for the same sampler, so the baseline decides which one is read."""
    small = HordeWorkerProcessManager._sampler_summary("k_dpmpp_sde", "stable_diffusion_1")
    large = HordeWorkerProcessManager._sampler_summary("k_dpmpp_sde", "flux_1")

    assert small is not None
    assert large is not None
    assert small.cost_ratio == _SDE.measured_cost_ratio_sd15
    assert large.cost_ratio == _SDE.measured_cost_ratio_sdxl
    assert small.cost_ratio != large.cost_ratio


def test_sampler_summary_marks_an_adaptive_sampler_and_claims_no_cost() -> None:
    """An adaptive sampler picks its own iteration count, so neither an order nor a ratio applies."""
    summary = HordeWorkerProcessManager._sampler_summary("k_dpm_adaptive", "stable_diffusion_xl")

    assert summary is not None
    assert summary.adaptive is True
    assert summary.work_per_step is None
    assert summary.cost_ratio is None


def test_sampler_summary_names_a_sampler_the_sdk_does_not_know() -> None:
    """The horde asked for it, so the operator sees it; the cost fields stay unstated rather than guessed."""
    summary = HordeWorkerProcessManager._sampler_summary("k_not_a_real_sampler", "stable_diffusion_xl")

    assert summary is not None
    assert summary.name == "k_not_a_real_sampler"
    assert summary.work_per_step is None
    assert summary.adaptive is False
    assert summary.cost_ratio is None


def test_sampler_summary_is_absent_without_a_sampler() -> None:
    """An alchemy form or a record built without a pop payload names no sampler at all."""
    assert HordeWorkerProcessManager._sampler_summary(None, "stable_diffusion_xl") is None


@pytest.mark.asyncio
async def test_work_ledger_row_carries_batch_count_and_sampler() -> None:
    """A row states the batch and the sampler, so two same-size jobs are not read as the same work."""
    manager = make_testable_process_manager()
    job = make_job_pop_response(width=832, height=1216, ddim_steps=28, n_iter=4, sampler_name="k_heun")
    await track_popped_job_async(manager._job_tracker, job)

    row = manager._build_work_ledger([])[0]

    assert row.batch_size == 4
    assert row.sampler is not None
    assert row.sampler.name == "k_heun"
    assert row.sampler.work_per_step == 2


@pytest.mark.asyncio
async def test_work_ledger_rows_arrive_in_pop_order() -> None:
    """Rows follow pop order even after a later-popped job advances a stage ahead of an earlier one.

    The tracker orders by stage entry, so the job that starts inference first would otherwise jump the
    row it was popped behind, and the ledger would disagree with the pop-order column it carries.
    """
    manager = make_testable_process_manager()
    jobs = [make_job_pop_response(model=f"model_{index}") for index in range(3)]
    for job in jobs:
        await track_popped_job_async(manager._job_tracker, job)
    await mark_job_in_progress_async(manager._job_tracker, jobs[2])

    rows = manager._build_work_ledger([])

    assert [row.queue_order for row in rows] == [1, 2, 3]
    assert [row.job_id for row in rows] == [str(job.id_) for job in jobs]


def _text_manager() -> HordeWorkerProcessManager:
    """Build a scribe worker whose text flow exists but has generated nothing."""
    manager = make_testable_process_manager(scribe=True)
    assert manager._text_coordinator is not None, "a scribe worker registers a text flow"
    return manager


def _hold_text_jobs(manager: HordeWorkerProcessManager, *jobs: TextJobInFlight) -> None:
    """Put jobs in the text flow's hand, as a pop would, without running a pop."""
    coordinator = manager._text_coordinator
    assert coordinator is not None
    coordinator._in_flight = {job.job_id: job for job in jobs}


@pytest.mark.asyncio
async def test_a_queued_and_a_generating_text_job_become_ledger_rows() -> None:
    """The three workloads share one ledger, and a text row says what it counts and where it runs.

    A generation runs in a separate program the worker reaches over HTTP, so there is no lane serving it:
    naming a process would point an operator at one doing something else.
    """
    manager = _text_manager()
    _hold_text_jobs(
        manager,
        TextJobInFlight(job_id="waiting", payload={"max_length": 160}, time_popped=time.time() - 3.0),
        TextJobInFlight(
            job_id="running",
            payload={"max_length": 160},
            time_popped=time.time() - 2.0,
            model_name="koboldcpp/Llama-3.2-3B-Instruct-Q4_K_M",
            time_generation_started=time.time() - 1.0,
            progress=TextGenerationProgress(
                chunks_received=12,
                characters_received=64,
                completion_tokens=40,
                elapsed_seconds=2.5,
                tokens_per_second=16.0,
                last_chunk_at=time.monotonic(),
            ),
        ),
    )

    rows = {row.job_id: row for row in manager._build_work_ledger([])}

    queued = rows["waiting"]
    assert queued.workload is WorkloadKind.TEXT_GENERATION
    assert queued.stage is WorkLedgerStage.QUEUED
    assert queued.progress_current is None
    assert queued.process_id is None
    assert queued.device_index is None

    generating = rows["running"]
    assert generating.stage is WorkLedgerStage.INFERENCE
    assert generating.model == "koboldcpp/Llama-3.2-3B-Instruct-Q4_K_M"
    assert generating.progress_unit is WorkLedgerProgressUnit.TOKENS
    assert generating.progress_current == 40
    assert generating.progress_total == 160
    assert generating.iterations_per_second == 16.0
    assert generating.progress_age_seconds is not None
    assert generating.process_id is None


@pytest.mark.asyncio
async def test_a_text_row_counts_chunks_when_its_backend_counts_no_tokens() -> None:
    """A stream record is not a token, so a backend that counts none leaves the row without a total."""
    manager = _text_manager()
    _hold_text_jobs(
        manager,
        TextJobInFlight(
            job_id="running",
            payload={"max_length": 160},
            time_popped=time.time() - 2.0,
            time_generation_started=time.time() - 1.0,
            progress=TextGenerationProgress(chunks_received=12, characters_received=64, elapsed_seconds=2.5),
        ),
    )

    row = manager._build_work_ledger([])[0]

    assert row.progress_unit is WorkLedgerProgressUnit.CHUNKS
    assert row.progress_current == 12
    assert row.progress_total is None
    assert row.iterations_per_second is None


@pytest.mark.asyncio
async def test_an_image_row_is_unchanged_by_the_text_rows_beside_it() -> None:
    """A mixed worker's image rows still lead the ledger and still read as sampler steps."""
    manager = _text_manager()
    job = make_job_pop_response(width=832, height=1216, ddim_steps=28)
    await track_popped_job_async(manager._job_tracker, job)
    _hold_text_jobs(
        manager,
        TextJobInFlight(job_id="running", payload={}, time_popped=time.time(), time_generation_started=time.time()),
    )

    rows = manager._build_work_ledger([])

    assert rows[0].job_id == str(job.id_)
    assert rows[0].workload is WorkloadKind.IMAGE_GENERATION
    assert rows[0].progress_unit is WorkLedgerProgressUnit.STEPS
    assert rows[0].progress_age_seconds is None
    assert rows[-1].workload is WorkloadKind.TEXT_GENERATION


@pytest.mark.asyncio
async def test_the_backend_row_carries_what_the_flow_has_in_flight() -> None:
    """Only the flow can see a generation, so the backend's row is filled from it at snapshot time."""
    manager = _text_manager()
    _hold_text_jobs(
        manager,
        TextJobInFlight(job_id="waiting", payload={}, time_popped=time.time()),
        TextJobInFlight(
            job_id="running",
            payload={},
            time_popped=time.time(),
            time_generation_started=time.time(),
            progress=TextGenerationProgress(
                chunks_received=3,
                characters_received=9,
                completion_tokens=8,
                elapsed_seconds=1.0,
                tokens_per_second=21.0,
            ),
        ),
    )

    activity = manager._text_backend_activity()

    assert activity.active_requests == 1
    assert activity.queued_requests == 1
    assert activity.tokens_per_second == 21.0


@pytest.mark.asyncio
async def test_a_worker_with_no_text_flow_reports_no_text_rows_and_no_activity() -> None:
    """A dreamer worker has no text flow at all, which must read as nothing rather than raise."""
    manager = make_testable_process_manager()

    assert manager._text_coordinator is None
    assert manager._text_in_flight_rows() == []
    assert manager._text_backend_activity().active_requests == 0
    assert manager._build_work_ledger([]) == []
